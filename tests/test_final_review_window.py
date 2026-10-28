"""The final protected review consumes its owned window, preserving publication.

No wall-clock waits/provider/network calls: the real review caller, scheduler and
ledger run with a fixed remaining-window fixture from a recorded timeout.
"""
from contextlib import ExitStack
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution,procedure_review as review,request_completion


def denied(*args,**kwargs):raise AssertionError('OFFLINE_NETWORK_DENIED')


class FinalReviewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards=ExitStack()
        cls.guards.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for target in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guards.enter_context(patch(target,denied))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):cls.guards.close()

    def case(self,*,remaining=33.115698,explicit_timeout=None,protected=True,compact=True):
        m=self.m;calls=[];attempted=[]
        request=m.AssistantCoreRequest(query='Describe inspection.',requested_mode='ask',response_language='en',
            company_id='offline-company',machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision=m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,effective_mode='ask',confidence=.99,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,evidence_policy='evidence_required',
            information_task='procedure_full',required_facets=('inspection',),required_answer_types=(m.REQ_ORDERED_ACTIONS,))
        answer='1. Keep the supply isolated.\n2. Inspect the marked guide.'
        rows=[{'citation_id':'step:one','source_type':'step','chunk_full':answer}]
        layout=review.block_layout(answer)
        owner=request_completion.RequestCompletion(runtime=m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=request.company_id),started=time.monotonic()) if protected else None
        budget=owner.budget if owner else m._assistant_core_new_budget('ask')
        budget.llm_calls=2;budget.max_llm_calls=3
        if owner:owner.largest_fence_seconds=2.02741
        original_deadline=budget.deadline_monotonic
        def provider(messages,**kwargs):
            attempted.append(kwargs['timeout'])
            timeout=request_completion.provider_timeout(kwargs['purpose'],kwargs['timeout'])
            timeout,cap,index=budget.reserve_call(model='gpt-5.6-terra',purpose=kwargs['purpose'],
                requested_timeout=timeout,max_output_tokens=kwargs['max_output_tokens'],messages=messages)
            calls.append({'requested':kwargs['timeout'],'scheduled':timeout,'cap':cap})
            # Mock wire reply only; binding and audit resolution remain real.
            result={'outcome':'pass','answer':'' if compact else answer,'reason':'verified fixture',
                'covered_facets':['inspection'],'missing_facets':[],
                'covered_answer_types':[m.REQ_ORDERED_ACTIONS],'missing_answer_types':[],
                'missing_list_items':[],'citation_ids':['step:one']}
            props=kwargs['json_schema']['schema']['properties']
            if 'edits' in props:
                result['citation_ids']=props['citation_ids']['items'].get('enum',['step:one'])
                result.update(reply_mode='retain',edits=[],operation_boundary={
                    'complete':True,'checked_blocks':props['edits']['items']['properties']['block_id']['enum'],
                    'change_edges':[],'precedence':[]})
            else:result['answer']=answer
            return result,'offline-model'
        runtime=replace(m._assistant_core_ask_execution_runtime(),evidence_admission=denied if protected else None,
            review_uses_remaining_window=explicit_timeout is None,
            V13_FAST_TIMEOUT_SECONDS=26 if explicit_timeout is None else explicit_timeout,
            _v13_current_budget=lambda:budget,_v13_json_models=provider,
            _v13_sources_block=lambda *a,**kw:answer)
        structure={'usable':True,'answer_sha256':review.digest(answer),'basis':'offline-source-structure',
            'expected_numbers':[1,2],'visible_numbers':[1,2],'sequence_complete':True,
            'source_notes':0,'source_notes_present':True,'semantic_coverage_proven':False,
            'source_sequence_required':True,'units':[{'citation_id':'step:one','number':1}]}
        try:
            with patch.object(budget,'remaining',return_value=remaining), \
                 patch.object(execution,'_admit_input',side_effect=lambda req,data,dec,**kw:data):
                try:
                    result=execution.verify_or_repair_answer(request=request,decision=decision,answer=answer,
                        candidates=rows,repair_context={'procedure_structure':structure} if compact else {},runtime=runtime)
                    error=None
                except m._V13BudgetExceeded as exc:
                    result=None;error=exc
            self.assertEqual(budget.deadline_monotonic,original_deadline)
            self.assertEqual(budget.max_estimated_cost_usd,.25)
            self.assertLessEqual(budget.llm_calls,3)
            return result,calls,attempted,error
        finally:
            if owner:owner.close()

    def test_recorded_window_is_used_without_extending_deadline_or_cutting_reserve(self):
        result,calls,attempted,error=self.case()
        self.assertIsNone(error);self.assertEqual(result['outcome'],'pass')
        self.assertEqual(calls,[{'requested':33,'scheduled':29,'cap':3200}])
        self.assertTrue(result['operation_boundary_validation']['precedence_checked'])

    def test_explicit_lower_fast_timeout_remains_a_limit(self):
        result,calls,_,error=self.case(explicit_timeout=15)
        self.assertIsNone(error);self.assertEqual(result['outcome'],'pass')
        self.assertEqual(calls[0]['requested'],15);self.assertEqual(calls[0]['scheduled'],15)

    def test_late_review_fails_before_dispatch_instead_of_borrowing_publication(self):
        result,calls,attempted,error=self.case(remaining=12.4)
        self.assertIsNotNone(error);self.assertIsNone(result)
        self.assertEqual(calls,[]);self.assertEqual(attempted,[12])

    def test_legacy_and_noncompact_consumers_keep_previous_timeout(self):
        for kwargs in ({'protected':False},{'compact':False}):
            with self.subTest(kwargs=kwargs):
                _,calls,_,error=self.case(**kwargs)
                self.assertIsNone(error);self.assertEqual(calls[0]['requested'],20)

    def test_factory_distinguishes_default_from_explicit_environment_limit(self):
        with patch.dict(os.environ,{}):
            os.environ.pop('MM_V13_FAST_TIMEOUT_SECONDS',None)
            self.assertTrue(self.m._assistant_core_ask_execution_runtime().review_uses_remaining_window)
        with patch.dict(os.environ,{'MM_V13_FAST_TIMEOUT_SECONDS':'17'}):
            self.assertFalse(self.m._assistant_core_ask_execution_runtime().review_uses_remaining_window)


if __name__=='__main__':unittest.main(verbosity=2)
