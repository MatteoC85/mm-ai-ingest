"""Complete-record capacity and accounted START failures; all external I/O denied."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import smart_evidence as evidence

SCOPE = {'company_id': 'capacity-fixture-company', 'machine_id': 'capacity-fixture-machine', 'ai_scope': 'machine_all'}


def row(number, body):
    return {'citation_id': f'capacity-source-{number}', 'bubble_document_id': f'capacity-document-{number}',
            'source_id': f'capacity-record-{number}', 'source_type': 'document',
            'company_id': SCOPE['company_id'], 'machine_id': SCOPE['machine_id'],
            'page_from': 1, 'page_to': 1, 'chunk_full': body}


def denied(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class SmartStartCapacityTests(unittest.TestCase):
    def select(self, rows, **kwargs):
        return evidence.select_start_sources(rows, scope=SCOPE, max_items=kwargs.pop('max_items', 8), **kwargs)

    def test_eight_valid_complete_records_fit_before_packet_without_clipping(self):
        rows = [row(i, (f'Documented pressure conditions and precautions {i}. ' * 60)[:2400]) for i in range(8)]
        before = deepcopy(rows)
        default = evidence.select_complete_sources(rows, scope=SCOPE, max_items=8)
        self.assertEqual(default, rows)
        with self.assertRaisesRegex(evidence.SmartEvidenceError, 'evidence_packet_capacity_exceeded'):
            evidence.build(default, scope=SCOPE)
        selected, meta = self.select(rows)
        self.assertTrue(selected)
        self.assertLess(len(selected), len(rows))
        packet = evidence.build(selected, scope=SCOPE)
        review = evidence.review_packet(packet)
        self.assertLessEqual(review['summary']['packet_chars'], evidence.MAX_CHARS)
        self.assertLessEqual(review['summary']['evidence_payload_chars'], evidence.MAX_CHARS)
        expected = {r['citation_id']: r['chunk_full'] for r in rows}
        self.assertEqual([r['text'] for r in review['validator_records']], [expected[r['citation_id']] for r in selected])
        self.assertTrue(meta['capacity_omitted'])
        self.assertFalse(meta['semantic_coverage_proven'])
        self.assertTrue(meta['complete_selected_records'])
        self.assertEqual(rows, before)
        selected[0]['chunk_full'] = 'caller mutation'
        self.assertEqual(rows, before)

    def test_oversized_priority_source_does_not_hide_later_whole_source(self):
        rows = [row(1, 'X' * 30000), row(2, 'Complete later record, including its final safety condition. ')]
        selected, meta = self.select(rows, max_items=1, relevant_ids=[rows[0]['citation_id']])
        self.assertEqual(selected, rows[1:])
        self.assertEqual(meta['capacity_omitted'], [{'citation_id': rows[0]['citation_id'], 'reason': 'evidence_packet_capacity_exceeded'}])
        self.assertEqual(meta['selected_ids'], [rows[1]['citation_id']])

    def test_foreign_or_conflicting_late_row_cannot_hide_after_count_or_capacity(self):
        first = row(1, 'Complete first record.')
        for extra, code in (({**row(2, 'Foreign'), 'company_id': 'other'}, 'evidence_company_mismatch'),
                            ({**row(2, 'Foreign'), 'machine_id': 'other'}, 'evidence_machine_mismatch'),
                            ({**first, 'chunk_full': 'Conflicting original identity'}, 'evidence_duplicate_citation_conflict')):
            for prefix in ([first], [row(0, 'X' * 30000), first]):
                with self.subTest(code=code, prefix=len(prefix)), self.assertRaisesRegex(evidence.SmartEvidenceError, '^' + code + '$'):
                    self.select(prefix + [extra], max_items=1)

    def test_expanded_review_limit_is_checked_even_when_nested_storage_fits(self):
        body = '0123456789' * 420
        rows = [row(i, body[:length]) for i, length in enumerate(range(3500, 4300, 100))]
        for item in rows:
            item.update(bubble_document_id='same-document', source_id='same-record')
        with self.assertRaisesRegex(evidence.SmartEvidenceError, 'evidence_review_capacity_exceeded'):
            evidence.build(rows, scope=SCOPE)
        selected, meta = self.select(rows)
        self.assertTrue(any(item['reason'] == 'evidence_review_capacity_exceeded' for item in meta['capacity_omitted']))
        evidence.validate(evidence.build(selected, scope=SCOPE), scope=SCOPE,
                          allowed_ids=[r['citation_id'] for r in selected])

    def test_nothing_fits_returns_no_excerpt_and_validates_config_limits(self):
        rows = [row(0, 'X' * 30000 + 'RETAIN FINAL SAFETY')]
        selected, meta = self.select(rows)
        self.assertEqual(selected, [])
        self.assertEqual(len(meta['capacity_omitted']), 1)
        self.assertTrue(rows[0]['chunk_full'].endswith('RETAIN FINAL SAFETY'))
        for limit in (None, 0, True, evidence.MAX_CHARS + 1):
            with self.subTest(limit=limit), self.assertRaisesRegex(evidence.SmartEvidenceError, 'evidence_limit_invalid'):
                self.select(rows, max_chars=limit)

    def test_priority_alias_and_unexpected_build_failure_remain_strict(self):
        first = row(1, 'Whole record with complete safety tail.\n')
        alias = {**first, 'citation_id': 'preferred-real-id'}
        selected, _ = self.select([first, alias], relevant_ids=[alias['citation_id']])
        self.assertEqual(selected, [alias])
        with patch.object(evidence, 'build', side_effect=evidence.SmartEvidenceError('evidence_owner_mismatch')):
            with self.assertRaisesRegex(evidence.SmartEvidenceError, '^evidence_owner_mismatch$'):
                self.select([first])


class SmartStartCapacityEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT': 'off',
            'AI_INTERNAL_SECRET': 'offline-capacity-only', 'OPENAI_API_KEY': 'offline-unused',
            'MM_INGEST_LEDGER_AUTO_DDL': '0'}))
        cls.guards.enter_context(patch.object(socket.socket, 'connect', denied))
        cls.guards.enter_context(patch.object(socket, 'create_connection', denied))
        cls.guards.enter_context(patch('requests.sessions.Session.request', denied))
        cls.guards.enter_context(patch('psycopg2.connect', denied))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch('machinemind.authority.request_admission.before_egress', return_value=None))

    def payload(self):
        return self.m.SmartDiagnosticStartRequest(**{k: SCOPE[k] for k in ('company_id', 'machine_id')},
            session_id='offline-capacity-session', symptom_text='Material missing signal after stop', language='en', debug=False,
            options=self.m.SmartDiagnosticOptions(top_k=8))

    def run_core(self, sources):
        from test_smart_review import fixture, parsed_review, wire_review
        _, draft = fixture()
        router = {'request_kind': 'guided_diagnostic', 'effective_mode': 'smart_diagnostic',
            'confidence': .95, 'requested_mode_fit': True, 'evidence_state': 'supported',
            'evidence_policy': 'machine_sources_required', 'information_task': 'fault_diagnostic',
            'required_answer_types': ['diagnostic_causes', 'checklist'],
            'relevant_evidence_ids': [source['citation_id'] for source in sources],
            'preferred_source_types': ['document'], 'source_type_policy': 'prefer',
            'dense_queries': [], 'lexical_queries': [], 'exact_terms': [], 'required_facets': [], 'facet_queries': [],
            'diagnostic_subsystems': ['material signal'], 'diagnostic_observables': ['Material missing signal'],
            'diagnostic_operating_conditions': [], 'diagnostic_discriminants': [], 'diagnostic_exclusions': [],
            'missing_information': [], 'clarification_question': '', 'safety_reason': '', 'out_of_scope_reason': '',
            'rationale': 'Synthetic capacity-test decision.'}
        prepared, purposes, generated_sources = [], [], []
        original_prepare = self.m._smart_review.prepare
        def prepare(**kwargs):
            result = original_prepare(**kwargs)
            prepared.append(result)
            return result
        def provider(*args, json, **kwargs):
            purpose = self.m._v13_current_budget().call_log[-1]['purpose']
            purposes.append(purpose)
            if purpose == 'assistant_core_v2_semantic_router':
                value = router
            elif purpose == 'smart_diagnostic_step_v1:start':
                generated_sources.append(json)
                value = draft
            else:
                self.assertEqual(purpose, 'smart_diagnostic_independent_review')
                value = wire_review(parsed_review(prepared[-1], rejected=()))
            return SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
                'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': __import__('json').dumps(value)})
        hooks = replace(self.m._ASSISTANT_CORE_SMART_ENGINE.hooks,
            retrieve_neutral=lambda request: {'candidates': sources, 'citations': sources},
            refine_retrieval=lambda request, retrieval, decision: retrieval)
        with patch.object(self.m, '_ASSISTANT_CORE_SMART_ENGINE', self.m.AssistantCoreV2(hooks)), \
             patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
             patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-capacity-only'), \
             patch.object(self.m, 'SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_ENABLED', False), \
             patch.object(self.m, '_build_rg_links', return_value=[]), \
             patch.object(self.m._smart_review, 'prepare', side_effect=prepare), \
             patch.object(self.m.requests, 'post', side_effect=provider), redirect_stdout(io.StringIO()):
            result = self.m._assistant_core_smart_start_sync(self.payload(), 'offline-capacity-only')
        return result, purposes, generated_sources

    def test_real_core_start_packs_then_generates_reviews_and_signs_only_selected_full_records(self):
        from test_smart_review import fixture
        sources, _ = fixture()
        for source in sources:
            source.update(company_id=SCOPE['company_id'], machine_id=SCOPE['machine_id'], similarity=.8)
        sources += [{**row(i, f'Documented material sensor signal condition and safety {i}. ' * 150),
                     'similarity': .8} for i in range(2, 8)]
        before = deepcopy(sources)
        with self.assertRaises(evidence.SmartEvidenceError):
            evidence.build(sources, scope=SCOPE)
        result, purposes, generation = self.run_core(sources)
        self.assertTrue(result['ok'])
        self.assertEqual(len(purposes), 3)
        self.assertEqual(len(generation), 1)
        state = json.loads(result['session_state_json'])
        self.assertTrue(state['state_signature'])
        packet = state['grounding_packet']
        selected_ids = {item['citation_id'] for item in packet['sources']}
        self.assertLess(len(selected_ids), len(sources))
        self.assertEqual(selected_ids, {item['citation_id'] for item in state['evidence']})
        expected = {r['citation_id']: r['chunk_full'] for r in sources}
        for item in evidence.review_packet(packet)['validator_records']:
            self.assertEqual(item['text'], expected[item['citation_id']])
        capacity = state['retrieval_assurance']['canonical_context']['capacity_selection']
        self.assertTrue(capacity['capacity_omitted'], capacity)
        self.assertFalse(capacity['semantic_coverage_proven'])
        self.assertTrue(result['meta']['v13_accounting_complete'])
        self.assertEqual(result['meta']['v13_llm_calls'], 3)
        self.assertEqual(sources, before)

    def test_start_public_projection_keeps_existing_sanitization_while_packet_retains_raw_body(self):
        raw = row(1, ' \nComplete admitted body.\nRetain the final safety condition.\n ')
        raw.update(arbitrary_private='PRIVATE_NOT_PUBLIC', url='PRIVATE_NOT_PUBLIC',
                   snippet='Display excerpt only', similarity=.8)
        with patch.object(self.m, '_fetch_document_file_map', return_value={}), \
             patch.object(self.m, '_v13_current_budget', return_value=None):
            expected = self.m._sd_select_complete_citations([raw], company_id=SCOPE['company_id'],
                machine_id=SCOPE['machine_id'], max_items=8)
            actual, packet, _ = self.m._sd_prepare_start_grounding([raw], company_id=SCOPE['company_id'],
                machine_id=SCOPE['machine_id'], max_items=8)
        self.assertEqual(actual, expected)
        self.assertNotIn('chunk_full', actual[0])
        self.assertNotIn('PRIVATE_NOT_PUBLIC', json.dumps(actual))
        self.assertEqual(evidence.review_packet(packet)['validator_records'][0]['text'], raw['chunk_full'])

    def test_real_core_all_too_large_is_accounted_technical_failure_without_generation(self):
        sources = [row(1, 'Documented material signal condition. ' * 1000)]
        result, purposes, generation = self.run_core(sources)
        self.assertEqual(purposes, ['assistant_core_v2_semantic_router'])
        self.assertEqual(generation, [])
        self.assertFalse(result['ok'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(result['detail']['reason'], 'evidence_start_capacity_unavailable')
        self.assertIn('more specific symptom', result['operator_summary'])
        self.assertEqual(result['session_state_json'], '')
        self.assertEqual(result['hypotheses'], [])
        self.assertEqual(result['meta']['v13_route'], 'assistant_core_smart_evidence_error')
        self.assertTrue(result['meta']['v13_accounting_complete'])
        self.assertEqual(result['meta']['v13_llm_calls'], 1)
        self.assertGreater(result['meta']['v13_estimated_cost_usd'], 0)

    def test_legacy_start_capacity_error_also_publishes_original_request_accounting(self):
        sources = [row(1, 'Material signal complete record. ' * 1000)]
        with patch.object(self.m, 'ASSISTANT_CORE_V2_ENABLED', False), \
             patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-capacity-only'), \
             patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
             patch.object(self.m, '_diagnostic_evidence_pipeline', return_value={'citations': sources}), \
             patch.object(self.m, '_v13_deterministic_evidence_state', return_value=('supported', {})), \
             patch.object(self.m, '_sd_semantic_evidence_gate', return_value={'accepted': True}), \
             patch.object(self.m, '_sd_run_retrieval_assurance', return_value=(True, sources, {})), \
             patch.object(self.m, '_sd_llm_step_start', side_effect=AssertionError('MUST_NOT_GENERATE')), \
             redirect_stdout(io.StringIO()):
            result = self.m.smart_diagnostic_start_v1(self.payload(), 'offline-capacity-only')
        self.assertEqual(result['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(result['detail']['reason'], 'evidence_start_capacity_unavailable')
        self.assertTrue(result['meta']['v13_accounting_complete'])
        self.assertEqual(result['meta']['v13_llm_calls'], 0)
        self.assertEqual(result['session_state_json'], '')

    def test_packet_adapter_does_not_mask_auth_signed_state_or_unknown_errors(self):
        from fastapi import HTTPException
        budget = self.m._assistant_core_new_budget('smart_diagnostic', company_id=SCOPE['company_id'])
        for status, code in ((401, 'SMART_DIAGNOSTIC_EVIDENCE_PACKET_INVALID'),
                             (403, 'SCOPE_DENIED'), (400, 'SMART_DIAGNOSTIC_STATE_TAMPERED'),
                             (409, 'SMART_DIAGNOSTIC_EVIDENCE_PACKET_INVALID'),
                             (409, 'SMART_DIAGNOSTIC_SESSION_RESTART_REQUIRED'),
                             (502, 'SMART_DIAGNOSTIC_EVIDENCE_PACKET_INVALID_EXTRA')):
            with self.subTest(status=status, code=code):
                self.assertIsNone(self.m._sd_error_response(HTTPException(status_code=status, detail={'code': code}),
                    'en', budget=budget, phase='start'))
        result = self.m._sd_error_response(HTTPException(status_code=502,
            detail={'code': 'SMART_DIAGNOSTIC_EVIDENCE_PACKET_INVALID', 'reason': 'PRIVATE_VALUE', 'message': 'PRIVATE_BODY'}),
            'en', budget=budget, phase='start')
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(result['detail']['reason'], 'evidence_packet_invalid')
        self.assertEqual(result['session_state_json'], '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
