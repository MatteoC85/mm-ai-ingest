"""Protected ASK router policy through the real composition and prompt emitters.

All network/SQL/provider boundaries are denied or synthetic. These tests prove
scope, wiring, unchanged bounds and emitted contracts, not live model quality.
"""
from contextlib import ExitStack
from contextvars import Context
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import routing_runtime as policy
from machinemind.ask.request_completion import RequestCompletion
from machinemind.ask.numeric_premise_contract import NUMERIC_PREMISE_CORRECTION_INSTRUCTION
import test_ask_source_identifiers as identifier_fixture
from test_ask_conditions_family_admission import OwnedFamilyFixture


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class AskRouterCostPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identifier_fixture.AskSourceIdentifierTests.setUpClass()
        cls.m = identifier_fixture.AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        identifier_fixture.AskSourceIdentifierTests.tearDownClass()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.budget = self.m._assistant_core_new_budget('ask', company_id='offline-company')
        token = self.m._V13_BUDGET_CTX.set(self.budget)
        self.addCleanup(self.m._V13_BUDGET_CTX.reset, token)
        self.stack.enter_context(patch.object(self.m, 'ASSISTANT_CORE_ROUTER_MODEL', 'configured-primary'))
        self.stack.enter_context(patch.object(self.m, 'ASSISTANT_CORE_ROUTER_FALLBACK_MODEL', 'configured-fallback'))
        self.stack.enter_context(patch.object(self.m, 'ASSISTANT_CORE_ROUTER_EFFORT', 'high'))
        self.stack.enter_context(patch.object(self.m, 'ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS', 15))
        self.assertIsNone(policy._ACTIVE.get(), 'A prior test leaked protected execution')

    def activate(self):
        execution, token = policy.activate()
        self.addCleanup(policy.deactivate, token)
        return execution

    def request(self, mode='ask', **kwargs):
        return self.m.AssistantCoreRequest(query='What operating pressure is documented?',
            requested_mode=mode, response_language='en', company_id='offline-company',
            machine_id='offline-machine', ai_scope='machine_all', top_k=6,
            allowed_effective_modes=(mode,), **kwargs)

    def decision(self, mode='ask'):
        return dict(effective_mode=mode, request_kind='factual' if mode == 'ask' else 'guided_diagnostic',
            confidence=.9, evidence_state='supported', evidence_policy='machine_sources_required',
            information_task='document_explanation' if mode == 'ask' else 'fault_diagnostic',
            required_answer_types=['explanation'] if mode == 'ask' else ['diagnostic_causes'],
            relevant_evidence_ids=[], dense_queries=[], lexical_queries=[], required_facets=[])

    def capture_router(self, mode='ask', **request_kwargs):
        with patch.object(self.m, '_v13_json_models',
                          return_value=(self.decision(mode), 'offline-fixture')) as provider:
            self.m._assistant_core_router_call(self.request(mode, **request_kwargs), {'candidates': []})
        provider.assert_called_once()
        return SimpleNamespace(args=deepcopy(provider.call_args.args),
                               kwargs=deepcopy(provider.call_args.kwargs))

    def test_off_and_non_ask_policy_preserve_order_effort_and_input(self):
        configured = ['configured-primary', 'configured-fallback']
        for mode in ('ask', 'root_cause', 'smart_diagnostic', 'other'):
            with self.subTest(mode=mode):
                result = policy.router_models(mode, configured)
                self.assertEqual(result, configured)
                self.assertIsNot(result, configured)
                self.assertEqual(policy.router_effort(mode, 'high'), 'high')
        self.assertEqual(configured, ['configured-primary', 'configured-fallback'])

    def test_only_active_ask_prefers_existing_fallback_without_adding_attempts(self):
        self.activate()
        for models in ([], ['single'], ['primary', 'fallback'], ['one', 'two', 'three']):
            before = list(models)
            with self.subTest(models=models):
                expected = ['fallback', 'primary'] if len(models) == 2 else before
                self.assertEqual(policy.router_models('ask', models), expected)
                self.assertEqual(policy.router_effort('ask', 'high'), 'low')
                self.assertEqual(models, before)
                for mode in ('root_cause', 'smart_diagnostic', 'other'):
                    self.assertEqual(policy.router_models(mode, models), before)
                    self.assertEqual(policy.router_effort(mode, 'high'), 'high')

    def test_active_policy_is_context_local_and_stops_when_owner_is_closed(self):
        execution = self.activate()
        configured = ['primary', 'fallback']
        self.assertEqual(policy.router_models('ask', configured), ['fallback', 'primary'])
        self.assertEqual(Context().run(policy.router_models, 'ask', configured), configured)
        self.assertEqual(Context().run(policy.router_effort, 'ask', 'high'), 'high')
        with self.assertRaisesRegex(RuntimeError, 'ALREADY_ACTIVE'):
            policy.activate()
        execution.active = False
        self.assertEqual(policy.router_models('ask', configured), configured)
        self.assertEqual(policy.router_effort('ask', 'high'), 'high')

    def test_real_main_changes_only_ask_order_and_effort_not_prompt_schema_or_limits(self):
        baseline = self.capture_router()
        bounds = (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd,
                  self.budget.max_llm_calls, self.budget.retry_allowance_calls)
        self.activate()
        protected = self.capture_router()
        self.assertEqual(baseline.kwargs['models'], ['configured-primary', 'configured-fallback'])
        self.assertEqual(protected.kwargs['models'], ['configured-fallback', 'configured-primary'])
        self.assertEqual(baseline.kwargs['effort'], 'high')
        self.assertEqual(protected.kwargs['effort'], 'low')
        base_kwargs, protected_kwargs = dict(baseline.kwargs), dict(protected.kwargs)
        for key in ('models', 'effort'):
            base_kwargs.pop(key); protected_kwargs.pop(key)
        self.assertEqual(base_kwargs, protected_kwargs)
        self.assertEqual(baseline.args, protected.args)
        self.assertEqual(protected.kwargs['timeout'], 15)
        self.assertEqual(protected.kwargs['max_output_tokens'], self.m.ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS)
        self.assertEqual(protected.kwargs['purpose'], 'assistant_core_v2_semantic_router')
        self.assertEqual(bounds, (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd,
                                 self.budget.max_llm_calls, self.budget.retry_allowance_calls))
        self.assertEqual(self.budget.llm_calls, 0, 'Synthetic provider capture must not dispatch')

    def test_root_and_smart_real_router_calls_are_identical_with_active_ask_context(self):
        baseline = {mode: self.capture_router(mode) for mode in ('root_cause', 'smart_diagnostic')}
        self.activate()
        for mode, expected in baseline.items():
            with self.subTest(mode=mode):
                actual = self.capture_router(mode)
                self.assertEqual(actual, expected)
        self.assertEqual(baseline['root_cause'].kwargs['models'], ['configured-primary'])
        self.assertEqual(baseline['root_cause'].kwargs['timeout'], 30)
        self.assertEqual(baseline['smart_diagnostic'].kwargs['models'], ['configured-primary'])

    def test_payload_metadata_cannot_enable_protected_router_policy(self):
        call = self.capture_router(metadata={'protected_execution': True,
            'router_order': 'configured_fallback_first', 'router_effort': 'low',
            'mm_protected_ask_routing_runtime': {'active': True}})
        self.assertEqual(call.kwargs['models'], ['configured-primary', 'configured-fallback'])
        self.assertEqual(call.kwargs['effort'], 'high')
        self.assertIsNone(policy._ACTIVE.get())

    def composition_fixture(self):
        events = []
        test = self
        class Completion:
            def __init__(self, **kwargs):
                events.append('completion_created')
            def publish(self, response, guard):
                return guard(response)
            def finish(self, response):
                return response
            def close(self):
                events.append('completion_closed')
        owner = SimpleNamespace(guards=(), final=lambda response: response,
                                close=lambda: events.append('source_owner_closed'))
        authorized = SimpleNamespace(admission=None, scope=SimpleNamespace())
        def authorize(*args, **kwargs):
            test.assertIsNone(policy._ACTIVE.get(), 'Routing policy activated before authentication')
            events.append('authenticated')
            return authorized
        def delegate(value, secret, **kwargs):
            test.assertEqual(events[:2], ['completion_created', 'authenticated'])
            events.append('delegate')
            call = test.capture_router()
            test.assertEqual(call.kwargs['models'], ['configured-fallback', 'configured-primary'])
            test.assertEqual(call.kwargs['effort'], 'low')
            return {'ok': True, 'status': 'answered', 'meta': {}}
        def protected_call(payload, secret, **kwargs):
            return kwargs['response_guard'](kwargs['delegate'](payload, secret))
        self.stack.enter_context(patch('machinemind.ask.request_completion.RequestCompletion', Completion))
        self.stack.enter_context(patch.object(self.m._application_authority, 'ResponseGuardOwner', return_value=owner))
        self.stack.enter_context(patch.object(self.m._application_authority, 'protected_call', side_effect=protected_call))
        self.stack.enter_context(patch.object(self.m._ask_request_flow, 'run_sync', side_effect=delegate))
        self.stack.enter_context(patch.object(self.m, '_PRECISION_FACT_RUNTIME', return_value=SimpleNamespace()))
        return events, authorize

    def test_real_composition_activates_after_authority_and_always_releases_context(self):
        events, authorize = self.composition_fixture()
        with patch.object(self.m._application_authority, 'authorize_http_request', side_effect=authorize):
            response = self.m._assistant_core_authorized_ask_sync(SimpleNamespace(), 'offline-service',
                x_mm_app_authority='offline-app', authority_environment={})
        self.assertEqual(events, ['completion_created', 'authenticated', 'delegate',
                                  'source_owner_closed', 'completion_closed'])
        self.assertIsNone(policy._ACTIVE.get())
        self.assertEqual(response['meta']['protected_execution']['router_order'], 'configured_fallback_first')

    def test_failed_authority_never_activates_policy_or_runs_router(self):
        events, _ = self.composition_fixture()
        with patch.object(self.m._application_authority, 'authorize_http_request',
                          side_effect=self.m._AuthorityError('SCOPE_DENIED')), \
             patch.object(policy, 'activate', side_effect=denied) as activate, \
             patch.object(self.m, '_assistant_core_router_call', side_effect=denied) as router:
            response = self.m._assistant_core_authorized_ask_sync(SimpleNamespace(), 'offline-service',
                x_mm_app_authority='offline-app', authority_environment={})
        activate.assert_not_called(); router.assert_not_called()
        self.assertEqual(events, ['completion_created', 'completion_closed'])
        self.assertIsNone(policy._ACTIVE.get())
        self.assertNotIn('protected_execution', response.get('meta') or {})

    def test_delegate_failure_closes_authenticated_execution_without_leaking_policy(self):
        events, authorize = self.composition_fixture()
        with patch.object(self.m._application_authority, 'authorize_http_request', side_effect=authorize), \
             patch.object(self.m._ask_request_flow, 'run_sync', side_effect=RuntimeError('offline delegate failed')):
            with self.assertRaisesRegex(RuntimeError, 'offline delegate failed'):
                self.m._assistant_core_authorized_ask_sync(SimpleNamespace(), 'offline-service',
                    x_mm_app_authority='offline-app', authority_environment={})
        self.assertIsNone(policy._ACTIVE.get())
        self.assertEqual(events[-2:], ['source_owner_closed', 'completion_closed'])

    def test_actual_generic_producer_and_reviewer_emit_same_numeric_premise_contract(self):
        completion = RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company', debug=False), started=time.monotonic())
        self.addCleanup(completion.close)
        answer = "Il documento indica il grado QTR-7 e la frequenza di 600 ore."
        result, calls, _ = identifier_fixture.AskSourceIdentifierTests.run_pipeline(self, answer)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual([kind for kind, _, _ in calls], ['writer', 'reviewer'])
        for kind, messages, kwargs in calls:
            with self.subTest(stage=kind):
                self.assertIn(NUMERIC_PREMISE_CORRECTION_INSTRUCTION, messages[0]['content'])
                self.assertEqual(messages[0]['content'].count(NUMERIC_PREMISE_CORRECTION_INSTRUCTION), 1)
                self.assertIn('without repeating its numeric literal', messages[0]['content'])
                self.assertIn('does not authorize any fact absent from SOURCES', messages[0]['content'])
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])

    def test_actual_owned_structured_producer_and_reviewer_emit_numeric_premise_contract(self):
        completion = RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company', debug=False), started=time.monotonic())
        self.addCleanup(completion.close)
        h = OwnedFamilyFixture(self, language='en')
        self.addCleanup(h.close)
        result = h.run()
        self.assertEqual(result['status'], 'answered')
        self.assertEqual([kind for kind, _, _ in h.calls], ['selector', 'writer', 'reviewer'])
        for kind, messages, _ in h.calls:
            if kind in ('writer', 'reviewer'):
                with self.subTest(stage=kind):
                    self.assertIn(NUMERIC_PREMISE_CORRECTION_INSTRUCTION, messages[0]['content'])
                    self.assertEqual(messages[0]['content'].count(NUMERIC_PREMISE_CORRECTION_INSTRUCTION), 1)
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
