"""Smart router gets one bounded wait without consuming review's time slot."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.retrieval import smart_routing
from test_smart_generation_scheduling import SmartGenerationSchedulingTests, denied


class SmartRouterAllocationTests(unittest.TestCase):
    setUpClass = classmethod(SmartGenerationSchedulingTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartGenerationSchedulingTests.tearDownClass.__func__)
    setUp = SmartGenerationSchedulingTests.setUp

    def request(self, mode='smart_diagnostic'):
        return self.m.AssistantCoreRequest(query='Pressure drops; actual circuit remains unknown.',
            requested_mode=mode, response_language='en', company_id='offline-company',
            machine_id='offline-machine', ai_scope='machine_all', top_k=8,
            allowed_effective_modes=(mode,))

    def decision(self, mode='smart_diagnostic'):
        return dict(effective_mode=mode, request_kind='guided_diagnostic', confidence=.9,
            evidence_state='supported', evidence_policy='machine_sources_required',
            information_task='fault_diagnostic', required_answer_types=['diagnostic_causes'],
            relevant_evidence_ids=[], dense_queries=[], lexical_queries=[], required_facets=[])

    def test_full_budget_and_upstream_delays_clip_single_wait_without_moving_deadline(self):
        deadline = self.budget.deadline_monotonic
        for remaining, expected in ((96,30),(69,30),(68.99,29),(55,16),(50,11),(47,8)):
            with self.subTest(remaining=remaining), patch.object(self.budget,'remaining',return_value=remaining), \
                 patch.object(self.m,'_v13_json_models',return_value=(self.decision(),'offline-primary')) as provider:
                self.m._assistant_core_router_call(self.request(), {'candidates':[]})
                provider.assert_called_once()
                self.assertEqual(provider.call_args.kwargs['timeout'],expected)
                self.assertGreaterEqual(remaining-expected,39)
                self.assertEqual(provider.call_args.kwargs['models'],[self.m.ASSISTANT_CORE_ROUTER_MODEL])
                self.assertEqual(provider.call_args.kwargs['max_output_tokens'],self.m.ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS)
                self.assertEqual(self.budget.deadline_monotonic,deadline)
                self.assertEqual(self.budget.max_estimated_cost_usd,self.cap)

    def test_insufficient_time_stops_before_provider_and_keeps_no_uncertain_liability(self):
        for remaining in (46.99,46.9,46,40,39.99,39,10,0):
            with self.subTest(remaining=remaining), patch.object(self.budget,'remaining',return_value=remaining), \
                 patch.object(self.m,'_v13_json_models',side_effect=denied) as provider:
                with self.assertRaises(smart_routing.SmartRouterUnavailable):
                    self.m._assistant_core_router_call(self.request(),{'candidates':[]})
                provider.assert_not_called()
                self.assertEqual(self.budget.llm_calls,0)
                self.assertEqual(self.budget.committed_cost_usd,0)

    def test_ask_and_root_keep_existing_timeout_models_and_skip_smart_allocator(self):
        with patch.object(self.m,'ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS',15), \
             patch.object(self.m,'ASSISTANT_CORE_ROUTER_MODEL','configured-primary'), \
             patch.object(self.m,'ASSISTANT_CORE_ROUTER_FALLBACK_MODEL','configured-fallback'), \
             patch.object(smart_routing,'allocated_timeout',side_effect=denied):
            for mode,timeout,models in (('ask',15,['configured-primary','configured-fallback']),
                                        ('root_cause',30,['configured-primary'])):
                with self.subTest(mode=mode), patch.object(self.m,'_v13_json_models',
                    return_value=(self.decision(mode),'configured-primary')) as provider:
                    self.m._assistant_core_router_call(self.request(mode),{'candidates':[]})
                    provider.assert_called_once()
                    self.assertEqual(provider.call_args.kwargs['timeout'],timeout)
                    self.assertEqual(provider.call_args.kwargs['models'],models)

    def test_real_transport_reserves_same_caps_and_keeps_timeout_unknown_without_retry(self):
        import requests
        with patch.object(self.m.requests,'post',side_effect=requests.ReadTimeout('offline-only')) as provider:
            with self.assertRaises(smart_routing.SmartRouterUnavailable):
                self.m._assistant_core_router_call(self.request(),{'candidates':[]})
        provider.assert_called_once()
        self.assertEqual(provider.call_args.kwargs['timeout'],30)
        self.assertEqual(self.budget.llm_calls,1)
        call = self.budget.call_log[0]
        self.assertEqual(call['accounting_state'],'uncertain')
        self.assertTrue(call['dispatched'])
        self.assertGreater(self.budget.committed_cost_usd,0)
        self.assertLessEqual(self.budget.committed_cost_usd,self.cap)
        self.assertEqual(self.budget.max_estimated_cost_usd,self.cap)
        self.assertEqual(call['max_output_tokens'],self.m.ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS)
        self.assertEqual(self.budget.retry_allowance_calls,0)
        self.assertEqual(self.budget.retry_events,[])

    def test_real_success_after_former_cutoff_leaves_generation_and_review_time(self):
        start=self.m.time_module.monotonic()
        clock=[start]
        def provider(*args,**kwargs):
            self.assertEqual(kwargs['timeout'],30)
            clock[0]+=20
            return SimpleNamespace(status_code=200,json=lambda:dict(status='completed',
                usage=dict(input_tokens=100,output_tokens=100),output_text=json.dumps(self.decision())))
        with patch.object(self.m.time_module,'monotonic',side_effect=lambda:clock[0]), \
             patch.object(self.m.requests,'post',side_effect=provider) as post:
            self.m._assistant_core_router_call(self.request(),{'candidates':[]})
            self.assertGreaterEqual(self.budget.remaining(),39)
            # Fake-clock subtraction can leave a subnanosecond float residual.
            self.assertLessEqual(self.budget.remaining(),76 + 1e-9)
        post.assert_called_once()
        self.assertEqual(self.budget.call_log[0]['accounting_state'],'settled')
        self.assertEqual(self.budget.retry_events,[])

    def test_invalid_or_unbounded_time_is_not_accepted(self):
        for value in (float('nan'),float('inf'),-1,True,'96',None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smart_routing.allocated_timeout(value,downstream_reserve_seconds=39)
        with self.assertRaises(ValueError):
            smart_routing.allocated_timeout(96,downstream_reserve_seconds=-1)


del SmartGenerationSchedulingTests
if __name__=='__main__':unittest.main(verbosity=2)
