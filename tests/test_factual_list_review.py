"""Real review adapter with recorded-style replies; no provider or database I/O.

Checks the empty-inventory defect and concrete omitted-item rejection, not live
semantic correctness. Admission is isolated here; authority has separate suites.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_final_review_window as window_fixtures
from machinemind.ask import execution, enumeration_review, final_contract, request_completion


class FactualListReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        window_fixtures.FinalReviewWindowTests.setUpClass()
        cls.m = window_fixtures.FinalReviewWindowTests.m

    @classmethod
    def tearDownClass(cls):
        window_fixtures.FinalReviewWindowTests.tearDownClass()

    def case(self, *, items=(), answer='Inspect the frame.', protected=True,
             mode='ask', kind='factual', task='document_explanation', state='supported',
             outcome='pass', budget=None, provider_schedule=None, runtime_overrides=None):
        m=self.m
        request=m.AssistantCoreRequest(query='Which inspection points and intervals are documented?',
            requested_mode=mode,response_language='en',company_id='offline-company',
            machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision=m.AssistantCoreDecision(request_kind=kind,effective_mode=mode,confidence=.9,
            requested_mode_fit=True,evidence_state=state,evidence_policy='evidence_required',
            information_task=task,required_facets=('inspection',),required_answer_types=('checklist',))
        source='Inspect the frame daily.\nCheck pressure sensor calibration every 200 hours.'
        rows=[{'citation_id':'manual:inspection','source_type':'document','chunk_full':source}]
        seen=[]
        def provider(messages,**kwargs):
            if provider_schedule is not None:
                provider_schedule(messages, kwargs)
            seen.append((messages,kwargs))
            return {'outcome':outcome,'answer':answer,'citation_ids':['manual:inspection'],
                'covered_facets':['inspection'],'missing_facets':[],
                'covered_answer_types':['checklist'],'missing_answer_types':[],
                'expected_list_items':list(items),'covered_list_items':list(items),
                'missing_list_items':[],'reason':'offline fixture'},'offline-model'
        schema=m._assistant_core_contract_verifier_schema()
        baseline=deepcopy(schema)
        runtime=replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=(lambda *a,**kw:None) if protected else None,
            _assistant_core_enumeration_requested=lambda *a:True,
            _assistant_core_contract_verifier_schema=lambda:schema,
            _v13_json_models=provider,
            _v13_current_budget=lambda:budget if budget is not None else m._assistant_core_new_budget('ask'))
        if runtime_overrides:
            runtime=replace(runtime,**runtime_overrides)
        with patch.object(execution,'_admit_input',side_effect=lambda req,data,dec,**kw:data):
            result=execution.verify_or_repair_answer(request=request,decision=decision,
                answer=answer,candidates=rows,runtime=runtime)
        self.assertEqual(len(seen),1)
        self.assertEqual(schema,baseline)
        return result,seen[0],baseline

    def window_case(self, *, remaining=53.517522, explicit_timeout=None,
                    protected=True, owner_present=True, task='document_explanation'):
        # The actual verifier selects its own factual-list profile. Only the
        # provider reply and remaining clock are fixtures; completion scheduling
        # and the per-request reservation ledger are the real implementations.
        m=self.m
        owner=request_completion.RequestCompletion(
            runtime=m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'),
            started=time.monotonic()) if owner_present else None
        budget=owner.budget if owner else m._assistant_core_new_budget('ask')
        budget.llm_calls=2
        budget.max_llm_calls=3
        if owner:
            owner.largest_fence_seconds=1.400963
        deadline=budget.deadline_monotonic
        attempts=[]
        dispatches=[]

        def schedule(messages,kwargs):
            attempts.append(kwargs['timeout'])
            timeout=request_completion.provider_timeout(kwargs['purpose'],kwargs['timeout'])
            timeout,cap,_=budget.reserve_call(model='gpt-5.6-terra',purpose=kwargs['purpose'],
                requested_timeout=timeout,max_output_tokens=kwargs['max_output_tokens'],messages=messages)
            dispatches.append({'requested':kwargs['timeout'],'scheduled':timeout,'cap':cap})

        try:
            with patch.object(budget,'remaining',return_value=remaining):
                try:
                    result,_,_=self.case(items=['Inspect the frame daily'],
                        answer='Inspect the frame daily.',protected=protected,task=task,
                        budget=budget,provider_schedule=schedule,runtime_overrides={
                            'review_uses_remaining_window':explicit_timeout is None,
                            'V13_FAST_TIMEOUT_SECONDS':26 if explicit_timeout is None else explicit_timeout})
                    error=None
                except m._V13BudgetExceeded as exc:
                    result=None
                    error=exc
            self.assertEqual(budget.deadline_monotonic,deadline)
            self.assertEqual(budget.max_estimated_cost_usd,.25)
            self.assertLessEqual(budget.llm_calls,3)
            return result,dispatches,attempts,error
        finally:
            if owner:
                owner.close()

    def test_factual_review_consumes_owned_window_without_extending_total_budget(self):
        result,calls,_,error=self.window_case()
        self.assertIsNone(error)
        self.assertEqual(result['outcome'],'pass')
        self.assertEqual(calls,[{'requested':53,'scheduled':50,'cap':3200}])

    def test_factual_review_respects_explicit_lower_fast_timeout(self):
        result,calls,_,error=self.window_case(explicit_timeout=15)
        self.assertIsNone(error)
        self.assertEqual(result['outcome'],'pass')
        self.assertEqual(calls,[{'requested':15,'scheduled':15,'cap':3200}])

    def test_factual_late_review_cannot_borrow_publication_reserve(self):
        result,calls,attempts,error=self.window_case(remaining=11.4)
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(calls,[])
        self.assertEqual(attempts,[11])

    def test_other_final_ask_review_uses_the_same_owned_window(self):
        _,calls,_,error=self.window_case(task='general_technical')
        self.assertIsNone(error)
        self.assertEqual(calls,[{'requested':53,'scheduled':50,'cap':3200}])

    def test_unprotected_and_ownerless_keep_legacy_review_timeout(self):
        for overrides in ({'protected':False},{'owner_present':False}):
            with self.subTest(overrides=overrides):
                _,calls,_,error=self.window_case(**overrides)
                self.assertIsNone(error)
                self.assertEqual(calls,[{'requested':20,'scheduled':20,'cap':3200}])

    def test_empty_claimed_pass_and_rewrite_cannot_certify_list(self):
        for outcome in ('pass','rewrite'):
            with self.subTest(outcome=outcome):
                result,_,_=self.case(outcome=outcome)
                self.assertEqual(result['outcome'],'no_sources')
                self.assertEqual(result['reason'],'supported_list_inventory_missing')

    def test_concrete_missing_row_rejected_even_when_provider_claims_coverage(self):
        item='Check pressure sensor calibration every 200 hours'
        result,_,_=self.case(items=[item])
        self.assertEqual(result['outcome'],'rewrite')
        self.assertEqual(result['missing_list_items'],[item])
        self.assertEqual(result['covered_list_items'],[])

    def test_complete_inventory_preserved_in_response_language_and_source_first(self):
        items=['Inspect the frame daily','Check pressure sensor calibration every 200 hours']
        result,(messages,kwargs),_=self.case(items=items,answer='. '.join(items)+'.')
        self.assertEqual(result['outcome'],'pass')
        self.assertEqual(result['covered_list_items'],items)
        self.assertEqual(result['missing_list_items'],[])
        self.assertEqual(next(iter(kwargs['json_schema']['schema']['properties'])),'expected_list_items')
        self.assertLess(messages[1]['content'].index('SOURCES:\n'),messages[1]['content'].index('CURRENT_ANSWER:\n'))
        self.assertIn('RESPONSE_LANGUAGE',messages[0]['content'])

    def test_nonfactual_procedural_unsupported_and_unprotected_keep_legacy_schema(self):
        for overrides in ({'protected':False},{'mode':'root_cause'},
                {'mode':'smart_diagnostic'},{'kind':'procedure','task':'procedure_segment'},
                {'task':'general_technical'},{'state':'unsupported'}):
            with self.subTest(overrides=overrides):
                result,(messages,kwargs),baseline=self.case(**overrides)
                self.assertEqual(result['outcome'],'pass')
                self.assertEqual(kwargs['json_schema'],baseline)
                self.assertNotIn('FACTUAL_LIST_REVIEW:',messages[0]['content'])

    def test_no_sources_is_not_upgraded_by_nonempty_inventory(self):
        result,_,_=self.case(items=['Inspect the frame'],outcome='no_sources')
        self.assertEqual(result['outcome'],'no_sources')

    def test_final_contract_consumes_real_reviewer_inventory_verdict(self):
        complete_items = ['Inspect the frame daily',
                          'Check pressure sensor calibration every 200 hours']
        for items, answer, expected in (
                ([], 'Inspect the frame.', False),
                (complete_items, 'Inspect the frame daily.', False),
                (complete_items, '. '.join(complete_items)+'.', True)):
            with self.subTest(items=items, answer=answer):
                reviewed, _, _ = self.case(items=items, answer=answer)
                # Exercise the real final gate with the real reviewer result.
                # Even optimistic upstream booleans cannot upgrade no_sources
                # or missing inventory items to a complete answer. This checks
                # contract consumption, not model semantic correctness or ACLs.
                result = final_contract.final_contract(
                    deterministic={'passed': True, 'reason': 'optimistic fixture'},
                    semantic=reviewed, reviewed_answer=reviewed['answer'],
                    answer=reviewed['answer'], semantic_complete=True,
                    semantic_partial=False)
                self.assertEqual(result['passed'], expected)
                self.assertEqual(result['complete'], expected)
                self.assertTrue(result['semantic_answer_bound'])
                self.assertEqual(result['semantic_verifier'], reviewed)
                if not items:
                    self.assertEqual(reviewed['reason'], 'supported_list_inventory_missing')
                    self.assertEqual(result['reason'], 'semantic_contract_incomplete')
                elif not expected:
                    self.assertEqual(result['missing_list_items'], [complete_items[1]])


if __name__=='__main__':unittest.main(verbosity=2)
