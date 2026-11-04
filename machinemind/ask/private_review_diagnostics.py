"""Bounded encrypted observation, never a source/answer approval or a replay.

Only an existing protected ASK owner may publish the returned ciphertext. No
credential, key, rejected prose or source body is logged or persisted here.
Secret-only redaction preserves ordinary citation identity and whitespace.
"""
from dataclasses import asdict, is_dataclass
import base64
import hashlib
import json
import os
import re
import zlib

VERSION = 'ask-private-factual-review-v1'
ALGORITHM = 'AES-256-GCM+HKDF-SHA256'
MAX_PLAINTEXT_BYTES = 512 * 1024
MAX_COMPRESSED_BYTES = 64 * 1024
MIN_SECRET_BYTES = 32
_DOMAIN = b'MachineMind/ASK/private-factual-review/v1'
_HASH = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_REQUEST = re.compile(r'[0-9a-f]{32}')
_BASE_KEYS = {'commit_sha', 'company_id', 'machine_id', 'request_id', 'query_sha256'}
_AAD_KEYS = _BASE_KEYS | {'version', 'source_packet_sha256', 'capture_sha256', 'redaction_changed'}
_BODY_KEYS = {'first_wire_reply', 'expanded_candidate', 'review_frame', 'table_catalog',
              'actual_review_reply', 'sources_block', 'question', 'response_language',
              'selected_records_only'}
_FORBIDDEN_KEYS = {'headers', 'authorization', 'auth_headers', 'private_context',
                   'authority_context', 'credentials', 'credential_values', 'secret_values',
                   'encryption_key', 'api_key', 'password', 'client_secret', 'access_token',
                   'refresh_token', 'id_token', 'ai_internal_secret'}
_CREDENTIAL_NAME = re.compile(r'(?:^|_)(?:SECRET|TOKEN|API_KEY|PASSWORD)$', re.I)
_ASSIGNMENT = re.compile(
    r'''(?i)(\b(?:[A-Z0-9_-]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD)|authorization)\b["']?\s*[:=]\s*)'''
    r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)''')
_BEARER = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*')
_KEY_PATTERN = re.compile(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,})\b')


class DiagnosticError(ValueError):
    """Fixed content-free codes only."""


def _require(value, code):
    if not value:
        raise DiagnosticError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _secret_bytes(secret):
    _require(type(secret) is str, 'diagnostic_secret_unavailable')
    data = secret.encode('utf-8')
    _require(MIN_SECRET_BYTES <= len(data) <= 4096, 'diagnostic_secret_unavailable')
    return data


def _base(binding):
    _require(type(binding) is dict and set(binding) == _BASE_KEYS, 'diagnostic_binding_invalid')
    _require(all(type(v) is str for v in binding.values()), 'diagnostic_binding_invalid')
    _require(_COMMIT.fullmatch(binding['commit_sha']) and _REQUEST.fullmatch(binding['request_id'])
             and _HASH.fullmatch(binding['query_sha256']), 'diagnostic_binding_invalid')
    _require(0 < len(binding['company_id']) <= 256 and len(binding['machine_id']) <= 256,
             'diagnostic_binding_invalid')
    return dict(binding)


def _plain(value, depth=0):
    _require(depth <= 32, 'diagnostic_structure_invalid')
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if type(value) is dict:
        _require(all(type(k) is str for k in value), 'diagnostic_structure_invalid')
        _require(not any(k.casefold() in _FORBIDDEN_KEYS for k in value), 'diagnostic_structure_invalid')
        table_token = (set(value) == {'token', 'text', 'citation_id', 'record_ids', 'source_sha256', 'table_lines'}
                       and type(value.get('token')) is str
                       and re.fullmatch(r'\[\[SOURCE_TABLE:G[1-9][0-9]*\]\]', value['token']))
        _require(not any(_CREDENTIAL_NAME.search(k) and not (k == 'token' and table_token) for k in value),
                 'diagnostic_structure_invalid')
        return {k: _plain(v, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_plain(v, depth + 1) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise DiagnosticError('diagnostic_structure_invalid')


def _redact(value, values):
    if type(value) is dict:
        return {k: _redact(v, values) for k, v in value.items()}
    if type(value) is list:
        return [_redact(v, values) for v in value]
    if type(value) is not str:
        return value
    for secret in values:
        variants = {secret, json.dumps(secret, ensure_ascii=False)[1:-1],
                    json.dumps(secret, ensure_ascii=True)[1:-1]}
        for variant in sorted(variants, key=len, reverse=True):
            value = value.replace(variant, '[REDACTED]')
    value = _ASSIGNMENT.sub(lambda match: match[1] + '[REDACTED]', value)
    value = _BEARER.sub('Bearer [REDACTED]', value)
    return _KEY_PATTERN.sub('[REDACTED]', value)


def _credential_values(secret, environment):
    # The mapping is the MachineMind server process environment, not an OS
    # credential file, request header or caller-supplied private context.
    values = {secret}
    for name, value in environment.items():
        if type(name) is str and _CREDENTIAL_NAME.search(name) and type(value) is str and value:
            values.add(value)
    _require(len(values) <= 64 and all(len(v.encode('utf-8')) <= 4096 for v in values),
             'diagnostic_secret_set_invalid')
    return sorted(values, key=len, reverse=True)


def _derive(secret, salt):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=_DOMAIN).derive(secret)


def seal(*, body, binding, secret, environment):
    """Return a complete authenticated envelope or raise a fixed local error.

    Secret material is used only for key derivation/redaction in this call.
    Neither it nor the values list is part of the whitelist payload.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    base = _base(binding)
    secret_bytes = _secret_bytes(secret)
    _require(type(body) is dict and set(body) == _BODY_KEYS, 'diagnostic_structure_invalid')
    _require(type(body['first_wire_reply']) is dict and type(body['expanded_candidate']) is dict
             and type(body['actual_review_reply']) is dict
             and type(body['sources_block']) is str and type(body['question']) is str
             and type(body['response_language']) is str and type(body['selected_records_only']) is bool,
             'diagnostic_structure_invalid')
    salt, nonce = os.urandom(16), os.urandom(12)
    key = _derive(secret_bytes, salt)  # Before redaction, never serialized.
    plain = _plain(body)
    raw = canonical(plain)
    _require(len(raw) <= MAX_PLAINTEXT_BYTES, 'diagnostic_plaintext_capacity')
    redacted = _redact(plain, _credential_values(secret, environment))
    # The original query digest is an external binding. A credential-bearing
    # query cannot safely publish that digest, so suppress its entire capture
    # rather than weakening/reinterpreting the request binding.
    _require(redacted['question'] == plain['question'], 'diagnostic_sensitive_query')
    redacted_data = canonical(redacted)
    redaction_changed = redacted_data != raw
    # Original digests stay encrypted. Public digests cover redacted content
    # only, so they cannot become a guess oracle for a weak source credential.
    private_manifest = {'raw_capture_sha256': _hash(raw),
                        'raw_source_packet_sha256': _hash(body['sources_block'].encode('utf-8')),
                        'redacted_capture_sha256': _hash(redacted_data)}
    data = canonical({'body': redacted, 'private_manifest': private_manifest})
    _require(len(data) <= MAX_PLAINTEXT_BYTES, 'diagnostic_plaintext_capacity')
    compressed = zlib.compress(data)
    _require(len(compressed) <= MAX_COMPRESSED_BYTES, 'diagnostic_compressed_capacity')
    aad = {**base, 'version': VERSION, 'source_packet_sha256': _hash(redacted['sources_block'].encode('utf-8')),
           # A redacted frame can still contain original internal binding
           # digests. Keep its aggregate digest private to avoid an indirect
           # guess oracle through those values.
           'capture_sha256': None if redaction_changed else _hash(redacted_data),
           'redaction_changed': redaction_changed}
    ciphertext = AESGCM(key).encrypt(nonce, compressed, canonical(aad))
    return {'version': VERSION, 'algorithm': ALGORITHM, 'compression': 'zlib', 'aad': aad,
            'salt_b64': base64.b64encode(salt).decode('ascii'),
            'nonce_b64': base64.b64encode(nonce).decode('ascii'),
            'ciphertext_b64': base64.b64encode(ciphertext).decode('ascii')}


def _decode(value, maximum):
    _require(type(value) is str and len(value) <= ((maximum + 2) // 3) * 4,
             'diagnostic_envelope_invalid')
    data = base64.b64decode(value, validate=True)
    _require(len(data) <= maximum, 'diagnostic_envelope_invalid')
    return data


def open_envelope(envelope, *, secret, expected_binding):
    """Private offline decrypt; caller must redact known secrets before sharing.

    This returns no source/answer authority. The original model verdict remains
    rejected if it was rejected. No network, files or automatic replay.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    base = _base(expected_binding)
    _require(type(envelope) is dict and set(envelope) == {'version', 'algorithm', 'compression',
             'aad', 'salt_b64', 'nonce_b64', 'ciphertext_b64'}, 'diagnostic_envelope_invalid')
    _require(envelope['version'] == VERSION and envelope['algorithm'] == ALGORITHM
             and envelope['compression'] == 'zlib', 'diagnostic_envelope_invalid')
    aad = envelope['aad']
    _require(type(aad) is dict and set(aad) == _AAD_KEYS and all(aad.get(k) == v for k, v in base.items())
             and aad.get('version') == VERSION and type(aad.get('redaction_changed')) is bool,
             'diagnostic_binding_invalid')
    _require(type(aad.get('source_packet_sha256')) is str and _HASH.fullmatch(aad['source_packet_sha256']),
             'diagnostic_binding_invalid')
    _require((aad['capture_sha256'] is None if aad['redaction_changed'] else
              type(aad['capture_sha256']) is str and _HASH.fullmatch(aad['capture_sha256'])),
             'diagnostic_binding_invalid')
    salt, nonce = _decode(envelope['salt_b64'], 16), _decode(envelope['nonce_b64'], 12)
    _require(len(salt) == 16 and len(nonce) == 12, 'diagnostic_envelope_invalid')
    ciphertext = _decode(envelope['ciphertext_b64'], MAX_COMPRESSED_BYTES + 16)
    compressed = AESGCM(_derive(_secret_bytes(secret), salt)).decrypt(nonce, ciphertext, canonical(aad))
    decompressor = zlib.decompressobj()
    data = decompressor.decompress(compressed, MAX_PLAINTEXT_BYTES + 1)
    _require(len(data) <= MAX_PLAINTEXT_BYTES and decompressor.eof and not decompressor.unconsumed_tail
             and not decompressor.unused_data, 'diagnostic_plaintext_capacity')
    package = json.loads(data)
    _require(type(package) is dict and set(package) == {'body', 'private_manifest'} and canonical(package) == data,
             'diagnostic_structure_invalid')
    body, private_manifest = package['body'], package['private_manifest']
    _require(type(body) is dict and set(body) == _BODY_KEYS and type(private_manifest) is dict
             and set(private_manifest) == {'raw_capture_sha256', 'raw_source_packet_sha256', 'redacted_capture_sha256'}
             and all(type(v) is str and _HASH.fullmatch(v) for v in private_manifest.values()),
             'diagnostic_structure_invalid')
    _require(_hash(canonical(body)) == private_manifest['redacted_capture_sha256']
             and _hash(body['sources_block'].encode('utf-8')) == aad['source_packet_sha256'],
             'diagnostic_payload_hash_invalid')
    if not aad['redaction_changed']:
        _require(private_manifest['raw_capture_sha256'] == aad['capture_sha256']
                 and private_manifest['redacted_capture_sha256'] == aad['capture_sha256']
                 and private_manifest['raw_source_packet_sha256'] == aad['source_packet_sha256'],
                 'diagnostic_payload_hash_invalid')
    return body
