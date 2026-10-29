"""Protected ASK review keeps output room without new calls, caps or deadlines."""
from contextlib import ExitStack
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution,request_completion
from machinemind.infrastructure import openai_transport,request_budget


def deny(*a,**kw):raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class AskReviewReasoningBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=ExitStack()
        cls.guard.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for target in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guard.enter_context(patch(target,deny))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):cls.guard.close()

    def dispatch(self,*,mode='ask',guarded=True,active=True):
        m=self.m;payloads=[]
        request=m.AssistantCoreRequest(query='Which checks precede insertion?',requested_mode=mode,
            response_language='en',company_id='offline-company',machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision=m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,effective_mode=mode,confidence=.99,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,evidence_policy='machine_sources_required',
            information_task='procedure_segment',required_facets=('isolation check',),required_answer_types=(m.REQ_SAFETY_CONDITIONS,))
        answer='Verify that the supply is isolated before insertion.'
        rows=[{'citation_id':'manual:safety','source_type':'document','chunk_full':answer}]
        owner=request_completion.RequestCompletion(runtime=m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=request.company_id),started=1000.0) if active else None
        budget=owner.budget if owner else m._assistant_core_new_budget('ask')
        original=(budget.deadline_monotonic,budget.max_llm_calls,budget.max_estimated_cost_usd)
        def provider(messages,**kwargs):
            verdict={'outcome':'pass','answer':answer,'reason':'offline fixture only','citation_ids':['manual:safety'],
                'covered_facets':['isolation check'],'missing_facets':[],
                'covered_answer_types':[m.REQ_SAFETY_CONDITIONS],'missing_answer_types':[],
                'missing_list_items':[]}
            class Response:
                status_code=200
                def json(self):return {'status':'completed','output_text':json.dumps(verdict),
                    'usage':{'input_tokens':100,'output_tokens':100,'output_tokens_details':{'reasoning_tokens':20}}}
            def post(url,**kw):payloads.append(kw['json']);return Response()
            parsed=openai_transport.responses_json(messages,model='gpt-5.6-terra',json_schema=kwargs['json_schema'],
                effort=kwargs['effort'],reasoning_mode=kwargs['reasoning_mode'],timeout=kwargs['timeout'],
                max_output_tokens=kwargs['max_output_tokens'],company_id=request.company_id,purpose=kwargs['purpose'],
                api_key='offline-unused',url='https://offline.invalid/responses',post_fn=post,
                current_budget_fn=lambda:budget,response_text_fn=openai_transport.response_text,
                safety_identifier_fn=openai_transport.safety_identifier)
            return parsed,'offline-model'
        runtime=replace(m._assistant_core_ask_execution_runtime(),evidence_admission=deny if guarded else None,
            V13_FAST_EFFORT='medium',_v13_current_budget=lambda:budget,_v13_json_models=provider,
            _v13_sources_block=lambda *a,**kw:answer)
        try:
            with patch.object(execution,'_admit_input',side_effect=lambda q,data,d,**kw:data):
                execution.verify_or_repair_answer(request=request,decision=decision,answer=answer,candidates=rows,runtime=runtime)
            self.assertEqual(len(payloads),1)
            self.assertEqual(payloads[0]['max_output_tokens'],3200)
            self.assertEqual(payloads[0]['model'],'gpt-5.6-terra')
            self.assertTrue(payloads[0]['text']['format']['strict'])
            self.assertIn(answer,payloads[0]['input'][-1]['content'])
            self.assertEqual((budget.deadline_monotonic,budget.max_llm_calls,budget.max_estimated_cost_usd),original)
            self.assertEqual(budget.llm_calls,1)
            return payloads[0]['reasoning']['effort']
        finally:
            if owner:owner.close()

    def test_active_protected_ask_uses_low_effort_in_actual_transport_payload(self):
        with patch.object(request_completion,'_monotonic',return_value=1000.0), \
             patch.object(request_budget,'_monotonic',return_value=1000.0):
            self.assertEqual(self.dispatch(),'low')

    def test_legacy_root_and_smart_keep_configured_effort(self):
        with patch.object(request_completion,'_monotonic',return_value=1000.0), \
             patch.object(request_budget,'_monotonic',return_value=1000.0):
            for settings in ({'guarded':False},{'active':False},{'mode':'root_cause'},{'mode':'smart_diagnostic'}):
                with self.subTest(settings=settings):self.assertEqual(self.dispatch(**settings),'medium')

    def test_recorded_a13_late_fallback_still_cannot_borrow_publication_time(self):
        now=[1000.0]
        with patch.object(request_completion,'_monotonic',side_effect=lambda:now[0]), \
             patch.object(request_budget,'_monotonic',side_effect=lambda:now[0]):
            owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'),started=1000.0)
            try:
                now[0]+=88.217782
                with self.assertRaises(request_budget._V13BudgetExceeded):
                    owner.provider_timeout('assistant_core_answer_contract_verifier',55)
                self.assertEqual(owner.provider_schedules[-1]['timeout_seconds'],8)
                self.assertEqual(owner.budget.llm_calls,0)
                self.assertEqual(owner.publication_reserve,3)
                self.assertEqual(owner.budget.deadline_monotonic,1100.0)
            finally:owner.close()


if __name__=='__main__':unittest.main(verbosity=2)
