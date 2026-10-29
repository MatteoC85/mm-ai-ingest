"""Real owned validation/finish with settled transports and bounded repair credit.

SQL, authorization and HTTP replies are fixtures. Session/source guards, model
transport, RequestBudget, validation and final handoff are the production code.
The rejection is synthetic: T20 did not retain its exact provider verdict.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_conditions_family_admission as fixtures
from assistant_core_v2 import _finish_ask_validation, _choose_monotonic_response, decorate_response
from machinemind.ask import execution, validation, request_completion
from machinemind.infrastructure import request_budget, openai_transport


class AskRepairCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m = fixtures.AskConditionsFamilyAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        fixtures.AskConditionsFamilyAdmissionTests.tearDownClass()

    def first_transport(self, h, *, failed):
        # A completed but invalid provider response still settles its usage and
        # remains eligible for the existing transport retry policy.
        class Response:
            status_code = 200
            def json(self):
                return {'status': 'incomplete' if failed else 'completed',
                    'output_text': '{}', 'usage': {'input_tokens': 100, 'output_tokens': 100}}
        def post(url, **kw):
            h.posts.append(kw['json'])
            return Response()
        def call():
            return openai_transport.responses_json([{'role':'user', 'content':'offline first transport'}],
            model=self.m.V13_FAST_MODEL,
            json_schema={'name':'offline', 'strict':True, 'schema':{
                'type':'object', 'properties':{}, 'required':[], 'additionalProperties':False}},
            effort='low', reasoning_mode='', timeout=15, max_output_tokens=3200,
            company_id=h.request.company_id, purpose='assistant_core_v2_semantic_router',
            api_key='offline-unused', url='https://offline.invalid/responses', post_fn=post,
            current_budget_fn=request_budget._v13_current_budget,
            response_text_fn=openai_transport.response_text,
            safety_identifier_fn=openai_transport.safety_identifier)
        if failed:
            with self.assertRaisesRegex(RuntimeError, 'status=incomplete'):call()
        else:self.assertEqual(call(), {})

    def scenario(self, *, failed=False, remaining=24.48, missing_budget=False):
        now = [1000.0]
        with patch.object(request_completion, '_monotonic', side_effect=lambda:now[0]), \
             patch.object(request_budget, '_monotonic', side_effect=lambda:now[0]):
            completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'), started=now[0])
            h = fixtures.OwnedFamilyFixture(self, transport=True, eight_points=True)
            try:
                h.prepared['assistant_core_decision'] = {'supported':True, 'strict_supported':True}
                h.owner._prepared_retrieval = deepcopy(h.prepared)
                self.first_transport(h, failed=failed)
                h.generated = execution.synthesize_ask(h.request, h.prepared, h.decision, runtime=h.runtime)
                def reject(messages, **kw):
                    return h.reply({'outcome':'no_sources', 'reply_mode':'replace', 'answer':'', 'edits':[],
                        'reason':'offline semantic rejection; exact live verdict unavailable',
                        'covered_facets':[h.decision.required_facets[0]],
                        'missing_facets':[h.decision.required_facets[1]],
                        'covered_answer_types':list(h.decision.required_answer_types), 'missing_answer_types':[],
                        'enumeration_requested':False, 'expected_list_items':[], 'covered_list_items':[],
                        'missing_list_items':['offline missing safety item'], 'citation_ids':[],
                        'operation_boundary':{'complete':False, 'checked_blocks':[], 'changes':[]}},
                        'offline-reviewer', messages, kw)
                eruntime = replace(h.runtime, _v13_json_models=reject,
                    _v13_current_budget=lambda:completion.budget)
                # Use the same binding as the Core adapter, including citation
                # recovery. Replacing only execution_runtime would omit that edge.
                vruntime = validation.bind_ask_validation(h.owner._validation_factory(), execution_runtime=eruntime)
                vruntime = replace(vruntime, _v13_current_budget=lambda:None if missing_budget else completion.budget)
                now[0] = 1100.0 - remaining
                result = validation.validate_response(h.generated, h.request, h.prepared, h.decision, runtime=vruntime)
                budget = completion.budget
                self.assertEqual(budget.llm_calls, 4)
                self.assertEqual(len(h.posts), 4)
                self.assertEqual(sum(r['accounting_state']=='settled' for r in budget.call_log), 4)
                self.assertEqual(sum(bool(r.get('failed')) for r in budget.call_log), int(failed))
                contract = result['meta']['assistant_core_validation']['answer_contract']
                self.assertFalse(contract['passed'])
                self.assertEqual(contract['semantic_verifier']['outcome'], 'no_sources')
                self.assertTrue(contract['missing_list_items'])
                if missing_budget:
                    self.assertTrue(result['_assistant_core_repair_needed'])
                    with self.assertRaisesRegex(execution.EvidenceContractError, 'protected repair budget missing'):
                        execution.repair_response(result, h.request, h.prepared, h.decision,
                            runtime=replace(eruntime, _v13_current_budget=lambda:None))
                    self.assertEqual(len(h.posts), 4)
                    return
                if failed and remaining >= 12:
                    self.assertTrue(result['_assistant_core_repair_needed'])
                    self.assertEqual(budget.retry_allowance_calls, 1)
                    self.assertEqual(budget.max_llm_calls, 5)
                    # Exercise the existing repair on the same owned source packet.
                    repaired_runtime = replace(eruntime, _v13_json_models=h.reviewer)
                    repaired_runtime = replace(repaired_runtime,
                        _assistant_core_verify_or_repair_answer=lambda **kw:
                            execution.verify_or_repair_answer(**kw, runtime=repaired_runtime))
                    repaired = execution.repair_response(result, h.request, h.prepared, h.decision, runtime=repaired_runtime)
                    self.assertEqual(budget.llm_calls, 5)
                    self.assertEqual(len(h.posts), 5)
                    self.assertEqual(budget.grant_retry_allowance(failed_attempts=1, reason='no duplicate credit'), 0)
                    final_runtime = validation.bind_ask_validation(h.owner._validation_factory(), execution_runtime=repaired_runtime)
                    final_runtime = replace(final_runtime, _v13_current_budget=lambda:budget)
                    validated_first = result
                    repaired['_assistant_core_repair_attempted'] = True  # Core's revalidation marker.
                    result = validation.validate_response(repaired, h.request, h.prepared, h.decision, runtime=final_runtime)
                    self.assertEqual(result['status'], 'answered')
                    result = _choose_monotonic_response(validated_first, result)
                else:
                    self.assertFalse(result.get('_assistant_core_repair_needed'), 'A settled rejection cannot schedule an unfunded fifth call')
                    self.assertEqual(budget.retry_allowance_calls, 0)
                    self.assertEqual(budget.max_llm_calls, 4)
                    self.assertEqual(result['status'], 'no_sources')
                    self.assertEqual(result['result_code'], 'INCOMPLETE_ANSWER_CONTRACT')
                    self.assertEqual(result['citations'], [])
                projected = decorate_response(_finish_ask_validation(result), h.request, h.decision)
                h.owner._finalize_core(h.request, projected)
                self.assertTrue(h.owner._response_flow.completed)
                self.assertFalse(any(k.startswith('_assistant_core_') for k in projected))
                self.assertEqual(budget.deadline_monotonic, 1100.0)
                self.assertEqual(budget.max_estimated_cost_usd, .25)
            finally:
                h.close()
                completion.close()

    def test_four_settled_calls_keep_semantic_failure_and_finish_without_fifth_call(self):
        self.scenario()

    def test_real_failed_transport_retains_single_existing_retry_credit(self):
        self.scenario(failed=True)

    def test_missing_protected_budget_still_raises_invariant(self):
        self.scenario(missing_budget=True)


if __name__ == '__main__':unittest.main(verbosity=2)
