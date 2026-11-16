"""Open one authenticated Root documentary rejection without exposing debug text."""

import base64
import hashlib
import json
import os

from machinemind.ask import private_review_diagnostics as _private


VERSION = "root-private-documentary-rejection-v1"
MAX_BYTES = 32 * 1024
_DOMAIN = b"MachineMind/Root/documentary-rejection/v1"


def _sha(value):
    return hashlib.sha256(_private.canonical(value)).hexdigest()


def _aad(binding):
    checked = _private._base(binding)
    return {"version": VERSION, "request_id": checked["request_id"],
            "binding_sha256": _sha(checked)}


def _key(secret, salt):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=_DOMAIN).derive(
        _private._secret_bytes(secret))


def seal(*, record, binding, secret, environment):
    """Emit only authenticated ciphertext, with existing credential redaction."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    plain = _private._plain(record)
    redacted = _private._redact(plain, _private._credential_values(secret, environment))
    manifest = {"capture_sha256": _sha(redacted), "redaction_changed": plain != redacted}
    if "claim_text" in redacted:
        manifest.update(claim_sha256=_sha(redacted["claim_text"]),
                        proof_parts_sha256=_sha(redacted["proof_parts"]))
    data = _private.canonical({"record": redacted, "manifest": manifest})
    if len(data) > MAX_BYTES:
        raise ValueError("root_rejection_capture_capacity")
    aad = _aad(binding)
    salt, nonce = os.urandom(16), os.urandom(12)
    ciphertext = AESGCM(_key(secret, salt)).encrypt(nonce, data, _private.canonical(aad))
    return {**aad, "salt_b64": base64.b64encode(salt).decode("ascii"),
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii")}


def open_envelope(envelope, *, secret, expected_binding):
    """Read the single protected rejection using the independently known scope."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aad = _aad(expected_binding)
    fields = {"salt_b64", "nonce_b64", "ciphertext_b64"}
    if type(envelope) is not dict or set(envelope) != set(aad) | fields or any(
            envelope[key] != value for key, value in aad.items()):
        raise ValueError("root_rejection_binding_invalid")
    salt = _private._decode(envelope["salt_b64"], 16)
    nonce = _private._decode(envelope["nonce_b64"], 12)
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("root_rejection_envelope_invalid")
    ciphertext = _private._decode(envelope["ciphertext_b64"], MAX_BYTES + 16)
    data = AESGCM(_key(secret, salt)).decrypt(nonce, ciphertext, _private.canonical(aad))
    package = json.loads(data)
    record, manifest = package["record"], package["manifest"]
    if (_private.canonical(package) != data or manifest["capture_sha256"] != _sha(record)
            or ("claim_text" in record and (manifest.get("claim_sha256") != _sha(record["claim_text"])
                or manifest.get("proof_parts_sha256") != _sha(record["proof_parts"])))):
        raise ValueError("root_rejection_capture_invalid")
    return package
