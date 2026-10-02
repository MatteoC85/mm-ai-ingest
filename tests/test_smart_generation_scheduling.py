"""Offline Smart scheduling/accounting boundaries, using the real provider ledger."""
from contextlib import ExitStack, redirect_stdout
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
        for remaining, expected in ((38.99, None), (39, 7), (45, 13), (72, 40), (96, 40)):
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
