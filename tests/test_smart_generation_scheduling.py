"""Offline Smart scheduling/accounting boundaries, using the real provider ledger."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def denied(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class SmartGenerationSchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT': 'off',
            'AI_INTERNAL_SECRET': 'offline-scheduling-only', 'OPENAI_API_KEY': 'offline-unused',
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
        self.patches.enter_context(patch.object(self.m, 'ASSISTANT_CORE_SMART_MODEL', 'gpt-5.6-terra'))
        self.patches.enter_context(patch('machinemind.authority.request_admission.before_egress', return_value=None))
        self.budget = self.m._assistant_core_new_budget('smart_diagnostic', company_id='offline-company')
        self.cap = self.budget.max_estimated_cost_usd
        token = self.m._V13_BUDGET_CTX.set(self.budget)
        self.addCleanup(self.m._V13_BUDGET_CTX.reset, token)
        self.schema = {'name': 'smart_diagnostic_step_v1', 'strict': True,
                       'schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}

    def generate(self, **kwargs):
        return self.m._assistant_core_sd_json_models([{'role': 'user', 'content': 'Offline generic diagnosis'}],
            json_schema=self.schema, timeout=70, phase='start', **kwargs)

    def test_default_fast_model_and_explicit_override_preserve_cost_limits(self):
        for override, expected in (('', 'gpt-5.6-terra'), ('gpt-5.6-sol', 'gpt-5.6-sol')):
            with self.subTest(override=override), patch.dict(os.environ, {
                'MM_ASSISTANT_CORE_SMART_MODEL': override, 'MM_V13_FAST_MODEL': 'gpt-5.6-terra',
                'MM_ASSISTANT_CORE_MAX_COST_SMART_START_USD': '0.35',
                'MM_ASSISTANT_CORE_MAX_COST_SMART_TURN_USD': '0.25'}):
                config = runpy.run_path(str(ROOT / 'machinemind/config/assistant_runtime.py'))
                self.assertEqual(config['ASSISTANT_CORE_SMART_MODEL'], expected)
                self.assertEqual(config['ASSISTANT_CORE_MAX_COST_SMART_START_USD'], .35)
                self.assertEqual(config['ASSISTANT_CORE_MAX_COST_SMART_TURN_USD'], .25)
                with patch.object(self.m, 'ASSISTANT_CORE_SMART_MODEL', expected), \
                     patch.object(self.m, '_v13_json_models', return_value=({}, expected)) as provider:
                    self.assertEqual(self.generate(models=['gpt-5.6-luna']), {})
                    self.assertEqual(provider.call_args.kwargs['models'], [expected])
                    self.assertEqual(provider.call_count, 1)
                    self.assertEqual(self.budget.max_estimated_cost_usd, self.cap)

    def test_dispatched_timeout_keeps_liability_without_retry_or_second_model(self):
        import requests
        with patch.object(self.m.requests, 'post', side_effect=requests.ReadTimeout('PRIVATE_ERROR_BODY')) as post:
            with self.assertRaises(RuntimeError):
                self.generate()
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.budget.llm_calls, 1)
        self.assertEqual(self.budget.call_log[0]['accounting_state'], 'uncertain')
        self.assertTrue(self.budget.call_log[0]['dispatched'])
        self.assertGreater(self.budget.committed_cost_usd, 0)
        self.assertLessEqual(self.budget.committed_cost_usd, self.cap)
        self.assertEqual(self.budget.max_estimated_cost_usd, self.cap)
        self.assertEqual(self.budget.retry_allowance_calls, 0)
        self.assertEqual(self.budget.retry_events, [])

    def test_paid_malformed_generation_is_not_retried_even_with_settled_usage(self):
        response = SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
            'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': 'invalid JSON'})
        with patch.object(self.m.requests, 'post', return_value=response) as post:
            with self.assertRaises(RuntimeError):
                self.generate()
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.budget.call_log[0]['accounting_state'], 'settled')
        self.assertGreater(self.budget.estimated_cost_usd, 0)
        self.assertEqual(self.budget.retry_allowance_calls, 0)

    def test_32_second_reserve_clamps_generation_and_preserves_review_slot(self):
        for remaining, expected in ((38.99, None), (39, 7), (45, 13), (72, 40), (96, 64), (110, 70)):
            with self.subTest(remaining=remaining), patch.object(self.budget, 'remaining', return_value=remaining), \
                 patch.object(self.m, '_v13_json_models', return_value=({}, 'offline')) as provider:
                if expected is None:
                    with self.assertRaisesRegex(self.m._V13BudgetExceeded, 'smart_generation_deadline_reserve'):
                        self.generate()
                    provider.assert_not_called()
                else:
                    self.generate()
                    self.assertEqual(provider.call_args.kwargs['timeout'], expected)
                    self.assertGreaterEqual(remaining - expected, 32)
        self.budget.llm_calls = self.budget.max_llm_calls - 1
        with patch.object(self.m, '_v13_json_models', side_effect=denied) as provider:
            with self.assertRaisesRegex(self.m._V13BudgetExceeded, 'call budget reserved'):
                self.generate()
            provider.assert_not_called()

    def test_no_budget_or_too_low_cost_cannot_use_unmetered_fallback(self):
        with patch.object(self.m, '_v13_current_budget', return_value=None), \
             patch.object(self.m, '_openai_chat_json_models', side_effect=denied) as legacy:
            with self.assertRaisesRegex(RuntimeError, 'requires its request budget'):
                self.generate()
            legacy.assert_not_called()
        self.budget.max_estimated_cost_usd = .000001
        with patch.object(self.m.requests, 'post', side_effect=denied) as post:
            with self.assertRaises(self.m._V13BudgetExceeded):
                self.generate()
            post.assert_not_called()
        self.assertEqual(self.budget.max_estimated_cost_usd, .000001)
        self.assertEqual(self.budget.llm_calls, 0)

    def test_guard_diagnostic_whitelists_values_and_never_exposes_arbitrary_text(self):
        self.budget.call_log = [{'call': 1, 'model': 'gpt-5.6-terra',
            'purpose': 'smart_diagnostic_step_v1:start', 'accounting_state': 'uncertain',
            'reserved_cost_usd': .03, 'estimated_cost_usd': .01, 'timeout_seconds': 40,
            'max_output_tokens': 5200, 'dispatched': True, 'error': 'PRIVATE_TOKEN',
            'raw_body': 'PRIVATE_SOURCE', 'headers': {'Authorization': 'PRIVATE_AUTH'}}]
        self.budget.call_log += [{'model': 'PRIVATE_MODEL', 'purpose': 'PRIVATE_PURPOSE',
            'accounting_state': 'PRIVATE_STATE', 'timeout_seconds': float('nan'),
            'reserved_cost_usd': 0, 'estimated_cost_usd': 'PRIVATE_COST', 'dispatched': 'PRIVATE_BOOL'}] * 8
        diagnostic = self.m._sd_budget_guard_diagnostic(
            self.m._V13BudgetExceeded('V13 insufficient reserved cost for useful output PRIVATE_EXCEPTION'), self.budget)
        self.assertEqual(diagnostic['category'], 'insufficient_reservation')
        self.assertEqual(len(diagnostic['calls']), 6)
        self.assertEqual(diagnostic['calls'][0]['model'], 'gpt-5.6-terra')
        self.assertEqual(diagnostic['calls'][1]['model'], 'unknown')
        self.assertIsNone(diagnostic['calls'][1]['timeout_seconds'])
        self.assertNotIn('PRIVATE', json.dumps(diagnostic))
        self.budget.call_log = []
        self.budget.accounting_anomalies = ['PRIVATE_ANOMALY']
        diagnostic = self.m._sd_budget_guard_diagnostic(self.m._V13BudgetExceeded('committed cost budget exhausted'), self.budget)
        self.assertEqual(diagnostic['category'], 'accounting_anomaly')
        self.assertNotIn('PRIVATE', json.dumps(diagnostic))

    def test_start_and_answer_cost_guards_emit_safe_diagnostic_without_debug(self):
        def fail(*args, **kwargs):
            raise self.m._V13BudgetExceeded('insufficient reserved cost PRIVATE_ERROR')
        payload = self.m.SmartDiagnosticStartRequest(company_id='offline-company', machine_id='offline-machine',
            session_id='offline-session', symptom_text='Offline reported stop', language='en', debug=False)
        stdout = io.StringIO()
        with patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
             patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-scheduling-only'), \
             patch.object(self.m._ASSISTANT_CORE_SMART_ENGINE, 'run', side_effect=fail), redirect_stdout(stdout):
            start = self.m._assistant_core_smart_start_sync(payload, 'offline-scheduling-only')
            with patch.object(self.m, '_assistant_core_run_smart_with_hard_timeout', side_effect=fail):
                answer = self.m._assistant_core_budgeted_sd_turn('answer')(lambda *a: {})(payload)
        for result in (start, answer):
            self.assertEqual(result['status'], 'budget_exceeded')
            self.assertFalse(result['meta']['cacheable'])
            self.assertEqual(result['meta']['smart_budget_guard']['category'], 'insufficient_reservation')
            self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertNotIn('PRIVATE', stdout.getvalue())

    def test_generation_timeout_returns_original_uncertain_accounting_on_start(self):
        import requests
        payload = self.m.SmartDiagnosticStartRequest(company_id='offline-company', machine_id='offline-machine',
            session_id='offline-session', symptom_text='Offline reported stop', language='en', debug=False)
        def generate(request):
            return self.m._sd_llm_step_start(symptom_text=request.query, language='en', max_questions=6,
                max_hypotheses=4, evidence_block='Offline evidence', evidence_ids=['offline:p1'])
        stdout = io.StringIO()
        with patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
             patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-scheduling-only'), \
             patch.object(self.m._ASSISTANT_CORE_SMART_ENGINE, 'run', side_effect=generate), \
             patch.object(self.m.requests, 'post', side_effect=requests.ReadTimeout('PRIVATE_PROVIDER_BODY')) as post, \
             redirect_stdout(stdout):
            result = self.m._assistant_core_smart_start_sync(payload, 'offline-scheduling-only')
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result['error_code'], 'SMART_DIAGNOSTIC_GENERATION_FAILED')
        self.assertEqual(result['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(result['meta']['v13_llm_calls'], 1)
        self.assertEqual(result['meta']['v13_route'], 'assistant_core_smart_generation_error')
        self.assertFalse(result['meta']['v13_accounting_complete'])
        self.assertGreater(result['meta']['v13_uncertain_cost_usd'], 0)
        self.assertEqual(result['meta']['v13_committed_cost_usd'], result['meta']['v13_uncertain_cost_usd'])
        diagnostic = result['meta']['smart_generation']
        self.assertEqual((diagnostic['stage'], diagnostic['category'], diagnostic['error_class']), ('start', 'timeout', 'ReadTimeout'))
        self.assertTrue(diagnostic['generation_attempt_recorded'])
        self.assertTrue(diagnostic['calls'][0]['dispatched'])
        self.assertEqual(diagnostic['calls'][0]['status'], 'failed')
        self.assertIsNotNone(diagnostic['calls'][0]['elapsed_seconds'])
        self.assertEqual(result['hypotheses'], [])
        self.assertNotIn('PRIVATE', json.dumps(result) + stdout.getvalue())
        self.assertIn('V13_REQUEST', stdout.getvalue())

    def test_answer_and_legacy_finalize_failures_keep_settled_usage(self):
        scope = {'company_id': 'offline-company', 'machine_id': 'offline-machine', 'ai_scope': 'machine_all'}
        source = {'citation_id': 'offline:p1', 'bubble_document_id': 'offline',
            'chunk_full': 'Offline complete evidence', **scope}
        state = {**scope, 'symptom_text': 'Offline stop', 'history': [], 'hypotheses': [],
            'evidence': [{'citation_id': 'offline:p1', 'snippet': 'Offline evidence'}],
            'grounding_packet': self.m._smart_evidence.build([source], scope=scope)}
        payload = SimpleNamespace(company_id='offline-company', language='en', debug=False)
        malformed = SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
            'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': 'PRIVATE_NON_JSON_BODY'})
        for phase, expected_code in (('answer', 'SMART_DIAGNOSTIC_GENERATION_FAILED'), ('finalize', 'SMART_DIAGNOSTIC_FINALIZE_FAILED')):
            with self.subTest(phase=phase):
                def generate(_payload, _secret):
                    if phase == 'answer':
                        return self.m._sd_llm_step_answer(state=state, answer={'value': 'unknown'}, language='en', max_hypotheses=4)
                    return self.m._sd_llm_finalize(state=state, language='en')
                stdout = io.StringIO()
                with patch.object(self.m.requests, 'post', return_value=malformed) as post, redirect_stdout(stdout):
                    result = self.m._assistant_core_budgeted_sd_turn(phase)(generate)(payload)
                self.assertEqual(post.call_count, 1)
                self.assertEqual(result['error_code'], expected_code)
                self.assertEqual(result['meta']['smart_generation']['stage'], phase)
                self.assertEqual(result['meta']['v13_route'], f'assistant_core_smart_{phase}_generation_error')
                self.assertEqual(result['meta']['smart_generation']['error_class'], 'JSONDecodeError')
                self.assertEqual(result['meta']['smart_generation']['category'], 'response_format')
                self.assertEqual(result['meta']['v13_llm_calls'], 1)
                self.assertTrue(result['meta']['v13_accounting_complete'])
                self.assertGreater(result['meta']['v13_estimated_cost_usd'], 0)
                self.assertEqual(result['meta']['v13_uncertain_cost_usd'], 0)
                self.assertNotIn('PRIVATE', json.dumps(result) + stdout.getvalue())

    def test_local_generator_failure_is_not_mislabeled_as_router_timeout(self):
        self.budget.call_log = [{'call': 1, 'purpose': 'assistant_core_v2_semantic_router',
            'model': 'gpt-5.6-terra', 'accounting_state': 'uncertain', 'reserved_cost_usd': .02,
            'dispatched': True, 'failed': True, 'error': 'ReadTimeout'}]
        diagnostic = self.m._sd_generation_failure_diagnostic(budget=self.budget, phase='start', error_class='NameError')
        self.assertEqual(diagnostic['category'], 'generation_contract')
        self.assertEqual(diagnostic['error_class'], 'NameError')
        self.assertFalse(diagnostic['generation_attempt_recorded'])
        self.assertEqual(diagnostic['calls'][0]['error_class'], 'ReadTimeout')
        timeout = self.m._sd_generation_failure_diagnostic(budget=self.budget, phase='start', error_class='TimeoutError')
        self.assertEqual(timeout['category'], 'timeout')
        self.assertEqual(timeout['error_class'], 'TimeoutError')

    def test_generation_error_adapter_does_not_intercept_auth_scope_or_unknown_errors(self):
        from fastapi import HTTPException
        for status, detail in ((401, 'Unauthorized'), (403, {'code': 'SCOPE_DENIED'}),
            (400, {'code': 'SMART_DIAGNOSTIC_STATE_TAMPERED'}),
            (409, {'code': 'SMART_DIAGNOSTIC_STALE_QUESTION'}),
            (502, {'code': 'SMART_DIAGNOSTIC_GENERATION_FAILED_EXTRA'})):
            with self.subTest(detail=detail):
                exc = HTTPException(status_code=status, detail=detail)
                self.assertIsNone(self.m._sd_error_response(exc, 'en', budget=self.budget, phase='start'))
        exc = HTTPException(status_code=502, detail={'code': 'SMART_DIAGNOSTIC_GENERATION_FAILED',
            'message': 'PRIVATE_PROVIDER_MESSAGE', 'generation_error_class': 'PRIVATE_CLASS',
            'generation_diagnostic': {'body': 'PRIVATE_SOURCE'}})
        result = self.m._sd_error_response(exc, 'en', budget=self.budget, phase='start')
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(result['meta']['smart_generation']['error_class'], 'unknown')
        self.assertEqual(result['session_state_json'], '')

    def test_real_start_engine_router_failure_is_technical_and_never_generates(self):
        """Only retrieval I/O is replaced; actual router/engine/prepare/generator run."""
        import requests
        source = {'citation_id': 'manual:p1', 'bubble_document_id': 'manual', 'source_type': 'document',
            'company_id': 'offline-company', 'machine_id': 'offline-machine', 'page_from': 1, 'page_to': 1,
            'chunk_full': 'Pressure loss can prevent motion. Check the pressure indicator before further checks.',
            'snippet': 'Pressure loss can prevent motion.', 'similarity': .8, 'exact_machine_scope': True}
        for router_elapsed in (3.0, 64.0):
            with self.subTest(router_elapsed=router_elapsed):
                clock = [self.m.time_module.monotonic()]
                budgets, purposes = [], []
                def retrieve(_request):
                    budgets.append(self.m._v13_current_budget())
                    return {'candidates': [source], 'citations': [source]}
                def provider(*args, json, **kwargs):
                    budget = self.m._v13_current_budget()
                    budgets.append(budget)
                    purpose = budget.call_log[-1]['purpose']
                    purposes.append(purpose)
                    if purpose == 'assistant_core_v2_semantic_router':
                        clock[0] += router_elapsed
                        raise requests.ReadTimeout('Offline router timeout')
                    self.assertEqual(purpose, 'smart_diagnostic_step_v1:start')
                    return SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
                        'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': 'invalid JSON'})
                hooks = replace(self.m._ASSISTANT_CORE_SMART_ENGINE.hooks, retrieve_neutral=retrieve,
                    refine_retrieval=lambda request, retrieval, decision: retrieval)
                engine = self.m.AssistantCoreV2(hooks)
                payload = self.m.SmartDiagnosticStartRequest(company_id='offline-company', machine_id='offline-machine',
                    session_id='offline-session', symptom_text='Pressure loss after stop', language='en', debug=False)
                with patch.object(self.m.time_module, 'monotonic', side_effect=lambda: clock[0]), \
                     patch.object(self.m, '_ASSISTANT_CORE_SMART_ENGINE', engine), \
                     patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
                     patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-scheduling-only'), \
                     patch.object(self.m, 'SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_ENABLED', False), \
                     patch.object(self.m, '_build_rg_links', return_value=[]), \
                     patch.object(self.m.requests, 'post', side_effect=provider), redirect_stdout(io.StringIO()):
                    result = self.m._assistant_core_smart_start_sync(payload, 'offline-scheduling-only')
                self.assertEqual(result['status'], 'error')
                self.assertEqual(result['error_code'], 'SMART_DIAGNOSTIC_ROUTER_FAILED')
                self.assertEqual(result['meta']['v13_route'], 'assistant_core_smart_router_error')
                self.assertTrue(budgets and all(b is budgets[0] and b is not None for b in budgets))
                self.assertEqual(result['meta']['v13_llm_calls'], len(purposes))
                self.assertFalse(result['meta']['v13_accounting_complete'])
                self.assertGreater(result['meta']['v13_uncertain_cost_usd'], 0)
                self.assertEqual(purposes, ['assistant_core_v2_semantic_router'])
                self.assertEqual(result['meta']['smart_router']['error_class'], 'ReadTimeout')
                self.assertEqual(result['meta']['smart_router']['category'], 'timeout')

    def test_real_start_success_and_valid_router_exhausting_time_use_same_ledger(self):
        from test_smart_review import fixture, parsed_review, wire_review
        sources, draft = fixture()
        for source in sources:
            source.update(company_id='offline-company', machine_id='offline-machine', similarity=.8)
        router = {'request_kind': 'guided_diagnostic', 'effective_mode': 'smart_diagnostic',
            'confidence': .95, 'requested_mode_fit': True, 'evidence_state': 'supported',
            'evidence_policy': 'machine_sources_required', 'information_task': 'fault_diagnostic',
            'required_answer_types': ['diagnostic_causes', 'checklist'], 'relevant_evidence_ids': ['manual:p1'],
            'preferred_source_types': ['document'], 'source_type_policy': 'prefer',
            'dense_queries': [], 'lexical_queries': [], 'exact_terms': [], 'required_facets': [], 'facet_queries': [],
            'diagnostic_subsystems': ['material signal'], 'diagnostic_observables': ['Material missing signal'],
            'diagnostic_operating_conditions': [], 'diagnostic_discriminants': [], 'diagnostic_exclusions': [],
            'missing_information': [], 'clarification_question': '', 'safety_reason': '', 'out_of_scope_reason': '',
            'rationale': 'Offline synthetic router decision.'}
        for router_elapsed in (2.0, 64.0):
            with self.subTest(router_elapsed=router_elapsed):
                clock = [self.m.time_module.monotonic()]
                budgets, purposes, prepared = [], [], []
                def retrieve(_request):
                    budgets.append(self.m._v13_current_budget())
                    return {'candidates': sources, 'citations': sources}
                original_prepare = self.m._smart_review.prepare
                def prepare(**kwargs):
                    result = original_prepare(**kwargs)
                    prepared.append(result)
                    return result
                def provider(*args, json, **kwargs):
                    budget = self.m._v13_current_budget()
                    budgets.append(budget)
                    purpose = budget.call_log[-1]['purpose']
                    purposes.append(purpose)
                    if purpose == 'assistant_core_v2_semantic_router':
                        self.assertEqual(json['reasoning']['effort'], 'low')
                        clock[0] += router_elapsed
                        response = router
                    elif purpose == 'smart_diagnostic_step_v1:start':
                        self.assertEqual(json['reasoning']['effort'], self.m.ASSISTANT_CORE_SMART_EFFORT)
                        response = draft
                    else:
                        self.assertEqual(purpose, 'smart_diagnostic_independent_review')
                        response = wire_review(parsed_review(prepared[-1], rejected=()))
                    self.assertIsInstance(json['text']['format']['schema'], dict)
                    return SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
                        'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': __import__('json').dumps(response)})
                hooks = replace(self.m._ASSISTANT_CORE_SMART_ENGINE.hooks, retrieve_neutral=retrieve,
                    refine_retrieval=lambda request, retrieval, decision: retrieval)
                payload = self.m.SmartDiagnosticStartRequest(company_id='offline-company', machine_id='offline-machine',
                    session_id='offline-session', symptom_text='Material missing signal after stop', language='en', debug=False)
                with patch.object(self.m.time_module, 'monotonic', side_effect=lambda: clock[0]), \
                     patch.object(self.m, '_ASSISTANT_CORE_SMART_ENGINE', self.m.AssistantCoreV2(hooks)), \
                     patch.object(self.m, 'SMART_DIAGNOSTIC_ENABLED', True), \
                     patch.object(self.m, 'AI_INTERNAL_SECRET', 'offline-scheduling-only'), \
                     patch.object(self.m, 'SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_ENABLED', False), \
                     patch.object(self.m, '_build_rg_links', return_value=[]), \
                     patch.object(self.m._smart_review, 'prepare', side_effect=prepare), \
                     patch.object(self.m.requests, 'post', side_effect=provider), redirect_stdout(io.StringIO()):
                    result = self.m._assistant_core_smart_start_sync(payload, 'offline-scheduling-only')
                self.assertTrue(all(b is budgets[0] and b is not None for b in budgets))
                self.assertTrue(result['meta']['v13_accounting_complete'])
                self.assertEqual(result['meta']['v13_llm_calls'], len(purposes))
                if router_elapsed == 2.0:
                    self.assertTrue(result['ok'])
                    self.assertEqual(result['status'], 'in_progress')
                    self.assertEqual(purposes, ['assistant_core_v2_semantic_router', 'smart_diagnostic_step_v1:start',
                                                'smart_diagnostic_independent_review'])
                    state = __import__('json').loads(result['session_state_json'])
                    self.assertTrue(state['state_signature'])
                else:
                    self.assertEqual(purposes, ['assistant_core_v2_semantic_router'])
                    self.assertEqual(result['status'], 'timeout')
                    self.assertEqual(result['meta']['smart_budget_guard']['category'], 'deadline')


if __name__ == '__main__':
    unittest.main(verbosity=2)
