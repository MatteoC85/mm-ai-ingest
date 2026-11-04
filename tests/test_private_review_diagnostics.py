"""Private diagnostics are confidential exact observations, never model calls."""
from copy import deepcopy
from dataclasses import asdict
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from machinemind.ask import private_review_diagnostics as p
from machinemind.ask.factual_context_review import ReviewFrame
from machinemind.ask.source_table_review import TableCatalog

SECRET = 'offline-synthetic-service-secret-0123456789'
BINDING = {'commit_sha': 'a' * 40, 'company_id': 'offline-company', 'machine_id': 'offline-machine',
           'request_id': 'b' * 32, 'query_sha256': hashlib.sha256(b'question').hexdigest()}


def body():
    wire = {'outcome': 'rewrite', 'answer': '  Supported selection.\n[[FACTUAL_INVENTORY]]',
            'expected_list_items': ['[[SOURCE_TABLE:G1]]'], 'citation_ids': ['ps:0123456789:p1'],
            'table_record_coverage': [], 'selection_scope_note': 'A useful selection.'}
    frame = ReviewFrame(first_json=json.dumps(wire, ensure_ascii=False), answer='  Literal answer.\n',
        template=wire['answer'], items=('  Literal table.\n',), immutable_indices=(0,),
        admitted_citation_ids=('ps:0123456789:p1',), current_citation_ids=('ps:0123456789:p1',),
        editable_targets=('template-answer',), table_owners=((0, 'ps:0123456789:p1'),),
        source_bodies=(('ps:0123456789:p1', '  Original\tspacing.\n[Q1]Retain source.[/Q]'),),
        source_packet_sha256='c' * 64, quote_view='[S1]  exact\tview\n[Q1]unchanged[/Q]',
        quote_units=(('Q1', 'Retain source.', ('ps:0123456789:p1',)),))
    return {'first_wire_reply': wire, 'expanded_candidate': deepcopy(wire), 'review_frame': frame,
            'table_catalog': TableCatalog(), 'actual_review_reply': {'outcome': 'no_sources', 'reason': 'uneditable_candidate'},
            'sources_block': '[ps:0123456789:p1] source_type=ps; p=1\n  Keep\ttabs and\nwhitespace.\n',
            'question': 'question', 'response_language': 'en', 'selected_records_only': True}


class PrivateDiagnosticsTests(unittest.TestCase):
    def seal(self, value=None, **kwargs):
        return p.seal(body=value if value is not None else body(), binding=BINDING,
                      secret=kwargs.pop('secret', SECRET), environment=kwargs.pop('environment', {}), **kwargs)

    def open(self, envelope, **kwargs):
        return p.open_envelope(envelope, secret=kwargs.pop('secret', SECRET),
                               expected_binding=kwargs.pop('binding', BINDING), **kwargs)

    def test_exact_roundtrip_preserves_frame_citation_ids_aliases_and_whitespace(self):
        value = body(); before = deepcopy(value)
        envelope = self.seal(value); actual = self.open(envelope)
        self.assertEqual(actual, p._plain(value))
        self.assertEqual(value, before)
        self.assertEqual(actual['review_frame'], p._plain(asdict(value['review_frame'])))
        self.assertEqual(actual['review_frame']['first_json'], value['review_frame'].first_json)
        self.assertIn('ps:0123456789:p1', actual['sources_block'])
        self.assertIn('  Keep\ttabs and\nwhitespace.\n', actual['sources_block'])
        self.assertIn('[Q1]', actual['review_frame']['quote_view'])
        self.assertFalse(envelope['aad']['redaction_changed'])
        self.assertNotIn('Literal answer', json.dumps(envelope))
        self.assertNotIn('uneditable_candidate', json.dumps(envelope))

    def test_explicit_process_credentials_and_patterns_are_redacted_before_encryption(self):
        value = body(); token = 'OFFLINE_PRIVATE_TOKEN_0123456789'
        key = 'sk-proj-' + 'X' * 32
        value['expanded_candidate']['answer'] = SECRET + ' ' + token + ' API_KEY=synthetic-pattern-only ' + key + ' Bearer ABCDEFGHIJKLMNOP'
        before = deepcopy(value)
        envelope = self.seal(value, environment={'MM_ACCESS_TOKEN': token, 'UNRELATED_NONCREDENTIAL': 'do not redact'})
        actual = json.dumps(self.open(envelope))
        for sensitive in (SECRET, token, key, 'synthetic-pattern-only', 'ABCDEFGHIJKLMNOP'):
            self.assertNotIn(sensitive, actual)
        self.assertTrue(envelope['aad']['redaction_changed'])
        self.assertNotIn('capture_raw_sha256', envelope['aad'])
        self.assertIsNone(envelope['aad']['capture_sha256'])
        self.assertEqual(value, before)

    def test_redacted_source_credentials_have_no_original_digest_in_public_aad(self):
        value = body(); value['sources_block'] += '\nDB_PASSWORD=weak-source-password\n'
        raw_body_hash = hashlib.sha256(p.canonical(p._plain(value))).hexdigest()
        raw_source_hash = hashlib.sha256(value['sources_block'].encode()).hexdigest()
        envelope = self.seal(value)
        public = json.dumps(envelope)
        self.assertNotIn(raw_body_hash, public); self.assertNotIn(raw_source_hash, public)
        decoded = self.open(envelope)
        self.assertNotIn('weak-source-password', json.dumps(decoded))
        self.assertEqual(envelope['aad']['source_packet_sha256'], hashlib.sha256(decoded['sources_block'].encode()).hexdigest())

    def test_random_salt_and_nonce_prevent_reused_ciphertext_for_same_request(self):
        first, second = self.seal(), self.seal()
        self.assertNotEqual(first['nonce_b64'], second['nonce_b64'])
        self.assertNotEqual(first['salt_b64'], second['salt_b64'])
        self.assertNotEqual(first['ciphertext_b64'], second['ciphertext_b64'])

    def test_wrong_key_tampering_and_scope_request_or_commit_substitution_fail(self):
        envelope = self.seal()
        with self.assertRaises(Exception): self.open(envelope, secret='wrong-synthetic-service-secret-0123456789')
        for field, replacement in [('company_id', 'foreign'), ('machine_id', 'foreign'), ('request_id', 'd' * 32),
                                   ('commit_sha', 'e' * 40), ('query_sha256', 'f' * 64)]:
            with self.subTest(field=field), self.assertRaises(p.DiagnosticError):
                self.open(envelope, binding={**BINDING, field: replacement})
        for field in ('ciphertext_b64', 'salt_b64', 'nonce_b64'):
            changed = deepcopy(envelope); raw = bytearray(base64.b64decode(changed[field])); raw[0] ^= 1
            changed[field] = base64.b64encode(raw).decode()
            with self.subTest(field=field), self.assertRaises(Exception): self.open(changed)
        changed = deepcopy(envelope); changed['aad']['capture_sha256'] = 'f' * 64
        with self.assertRaises(Exception): self.open(changed)

    def test_missing_short_secret_invalid_structure_and_private_context_never_capture(self):
        for secret in (None, '', 'short', 'x' * 4097):
            with self.subTest(secret_type=type(secret).__name__), self.assertRaises(p.DiagnosticError): self.seal(secret=secret)
        for key in ('headers', 'authorization', 'private_context', 'access_token', 'DB_PASSWORD', 'CUSTOM_API_KEY', 'token'):
            value = body(); value['expanded_candidate'][key] = 'MUST_NOT_CAPTURE'
            with self.subTest(key=key), self.assertRaises(p.DiagnosticError): self.seal(value)
        value = body(); value['unknown_body_field'] = 'MUST_NOT_CAPTURE'
        with self.assertRaises(p.DiagnosticError): self.seal(value)

    def test_sensitive_query_is_wholly_suppressed_instead_of_exposing_its_original_digest(self):
        value = body(); value['question'] = 'DB_PASSWORD=weak-synthetic-password'
        with self.assertRaisesRegex(p.DiagnosticError, 'diagnostic_sensitive_query'): self.seal(value)

    def test_only_the_exact_literal_table_catalog_token_is_allowed_as_noncredential_structure(self):
        from machinemind.ask.source_table_review import TableEntry
        value = body(); value['table_catalog'] = TableCatalog(entries=(TableEntry('[[SOURCE_TABLE:G1]]',
            'unchanged table', 'ps:0123456789:p1', ('T1',), 'a' * 64, ('literal row',)),))
        self.assertEqual(self.open(self.seal(value))['table_catalog'], p._plain(value['table_catalog']))

    def test_uncompressed_and_compressed_limits_reject_whole_payload_without_clipping(self):
        value = body(); value['question'] = 'x' * (p.MAX_PLAINTEXT_BYTES + 1)
        with self.assertRaisesRegex(p.DiagnosticError, 'diagnostic_plaintext_capacity'): self.seal(value)
        value = body(); value['question'] = base64.b64encode(os.urandom(100000)).decode()
        with self.assertRaisesRegex(p.DiagnosticError, 'diagnostic_compressed_capacity'): self.seal(value)

    def test_authenticated_decompression_cannot_expand_beyond_limit(self):
        envelope = self.seal(); raw = b'X' * (p.MAX_PLAINTEXT_BYTES + 1)
        salt = base64.b64decode(envelope['salt_b64']); nonce = base64.b64decode(envelope['nonce_b64'])
        cipher = AESGCM(p._derive(SECRET.encode(), salt)).encrypt(nonce, zlib.compress(raw), p.canonical(envelope['aad']))
        envelope['ciphertext_b64'] = base64.b64encode(cipher).decode()
        with self.assertRaisesRegex(p.DiagnosticError, 'diagnostic_plaintext_capacity'): self.open(envelope)


if __name__ == '__main__':
    unittest.main(verbosity=2)
