"""Offline admission tests: skip an unfundable full-output attempt before its ledger row."""
from contextlib import ExitStack
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import request_budget as budgets, openai_transport as transport

def deny(*args,**kwargs):raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')

class Response:
    status_code=200
    def json(self):return {'status':'completed','output_text':'{"ok":true}',
                           'usage':{'input_tokens':100,'output_tokens':40}}

class ReviewOutputCapacityTests(unittest.TestCase):
    def setUp(self):
        self.guard=ExitStack();self.addCleanup(self.guard.close)
        for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request'):
            self.guard.enter_context(patch(name,deny))
        self.guard.enter_context(patch.object(budgets,'_monotonic',return_value=1000.0))
        self.guard.enter_context(patch.object(budgets,'_text_input_bound',return_value=(12000,4000)))
        self.guard.enter_context(patch.object(budgets,'_RUNTIME_GLOBALS',{
            'V13_MAX_ESTIMATED_COST_ASK_USD':.06,'V13_MAX_LLM_CALLS_ASK':4,'V13_ASK_DEADLINE_SECONDS':100,
            'V13_PRICE_TERRA_INPUT':2.5,'V13_PRICE_TERRA_OUTPUT':15,
            'V13_PRICE_LUNA_INPUT':1,'V13_PRICE_LUNA_OUTPUT':6}))
        self.budget=budgets._V13RequestBudget('ask');self.posts=[];self.attempted=[]
        self.schema={'name':'capacity_fixture','strict':True,'schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}

    def reserve(self,model='gpt-5.6-terra',full=True):
        return self.budget.reserve_call(model=model,purpose='fixture',requested_timeout=30,max_output_tokens=3200,
                                       messages=[],**({'require_full_output':True}if full else {}))

    def delegate(self,messages,model,**kw):
        self.attempted.append(model)
        def post(url,**request):
            self.assertFalse(request['allow_redirects']);self.posts.append(request['json']);return Response()
        return transport.responses_json(messages,model=model,api_key='offline-unused',url='https://offline.invalid/responses',
            post_fn=post,current_budget_fn=lambda:self.budget,response_text_fn=transport.response_text,
            safety_identifier_fn=transport.safety_identifier,**kw)

    def run_models(self,full=True,models=None):
        return transport.json_models([],models=models or ['gpt-5.6-terra','gpt-5.6-luna'],json_schema=self.schema,
            effort='low',reasoning_mode='',timeout=30,max_output_tokens=3200,company_id='offline-scope',purpose='fixture',
            default_model='gpt-5.6-terra',normalize_models_fn=transport.normalize_model_candidates,
            current_budget_fn=lambda:self.budget,responses_json_fn=self.delegate,chat_json_fn=deny,
            budget_exceeded_type=budgets._V13BudgetExceeded,**({'require_full_output':True}if full else {}))

    def test_starved_model_is_denied_before_any_reservation_or_counter_change(self):
        before=deepcopy(self.budget.public_meta())
        with self.assertRaises(budgets._V13OutputCapacityInsufficient):self.reserve()
        self.assertEqual(self.budget.public_meta(),before)
        self.assertEqual(self.posts,[])

    def test_existing_next_model_receives_full_output_without_skip_retry_credit(self):
        limits=(self.budget.deadline_monotonic,self.budget.max_estimated_cost_usd,self.budget.max_llm_calls)
        answer,model=self.run_models()
        self.assertEqual((answer,model),({'ok':True},'gpt-5.6-luna'))
        self.assertEqual(self.attempted,['gpt-5.6-terra','gpt-5.6-luna'])
        self.assertEqual([p['model']for p in self.posts],['gpt-5.6-luna'])
        self.assertEqual(self.posts[0]['max_output_tokens'],3200)
        self.assertEqual(self.budget.llm_calls,1);self.assertEqual(len(self.budget.call_log),1)
        self.assertEqual(self.budget.call_log[0]['accounting_state'],'settled')
        self.assertEqual((self.budget.retry_allowance_calls,self.budget.retry_events),(0,[]))
        self.assertEqual(limits,(self.budget.deadline_monotonic,self.budget.max_estimated_cost_usd,self.budget.max_llm_calls))

    def test_both_models_ineligible_dispatch_nothing_and_raise_capacity_error(self):
        self.budget.max_estimated_cost_usd=.02;before=deepcopy(self.budget.public_meta())
        with self.assertRaises(budgets._V13OutputCapacityInsufficient):self.run_models()
        self.assertEqual(self.attempted,['gpt-5.6-terra','gpt-5.6-luna'])
        self.assertEqual(self.posts,[]);self.assertEqual(self.budget.public_meta(),before)

    def test_exact_full_cost_boundary_and_one_output_token_below(self):
        full_cost=12000*1*1.25/1_000_000+3200*6/1_000_000
        self.budget.max_estimated_cost_usd=full_cost-6/1_000_000
        with self.assertRaises(budgets._V13OutputCapacityInsufficient):self.reserve('gpt-5.6-luna')
        self.assertEqual(self.budget.call_log,[])
        self.budget.max_estimated_cost_usd=full_cost
        _,cap,index=self.reserve('gpt-5.6-luna')
        self.assertEqual((cap,index),(3200,1))
        self.assertAlmostEqual(self.budget.committed_cost_usd,full_cost,places=12)

    def test_default_partial_output_floor_and_original_model_order_remain(self):
        answer,model=self.run_models(full=False)
        self.assertEqual(model,'gpt-5.6-terra');self.assertEqual(self.attempted,[model])
        self.assertGreaterEqual(self.posts[0]['max_output_tokens'],800)
        self.assertLess(self.posts[0]['max_output_tokens'],3200)
        self.assertEqual(self.budget.retry_allowance_calls,0)

    def test_default_below_800_floor_still_aborts_without_model_fallback(self):
        self.budget.max_estimated_cost_usd=.0375+799*15/1_000_000
        with self.assertRaises(budgets._V13BudgetExceeded)as caught:self.run_models(full=False)
        self.assertNotIsInstance(caught.exception,budgets._V13OutputCapacityInsufficient)
        self.assertEqual(self.attempted,['gpt-5.6-terra']);self.assertEqual(self.posts,[])
        self.assertEqual(self.budget.call_log,[])

    def test_chat_fallback_also_reserves_full_output_without_leaking_flag(self):
        def chat_delegate(messages,model,**kwargs):
            self.attempted.append(model)
            def post(url,**request):
                self.posts.append(request['json'])
                return SimpleNamespace(status_code=200,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}],
                    'usage':{'prompt_tokens':100,'completion_tokens':40}})
            return transport.chat_json(messages,model=model,api_key='offline-unused',default_model='alpha',
                url='https://offline.invalid/chat',post_fn=post,current_budget_fn=lambda:self.budget,**kwargs)
        with patch.object(budgets,'_v13_model_rates',side_effect=lambda model:(2.5,15)if model=='alpha'else(1,6)):
            result=transport.json_models([],models=['alpha','beta'],json_schema=self.schema,effort='low',reasoning_mode='',
                timeout=30,max_output_tokens=3200,company_id='offline',purpose='fixture',default_model='alpha',
                normalize_models_fn=transport.normalize_model_candidates,current_budget_fn=lambda:self.budget,
                responses_json_fn=deny,chat_json_fn=chat_delegate,budget_exceeded_type=budgets._V13BudgetExceeded,
                require_full_output=True)
        self.assertEqual(result,({'ok':True},'beta'));self.assertEqual(self.attempted,['alpha','beta'])
        self.assertEqual(len(self.posts),1);self.assertEqual(self.posts[0]['max_completion_tokens'],3200)
        self.assertNotIn('require_full_output',self.posts[0])
        self.assertEqual(self.budget.llm_calls,1);self.assertEqual(self.budget.retry_allowance_calls,0)

    def test_unknown_previous_hold_is_not_released_to_fit_either_model(self):
        self.budget.max_estimated_cost_usd=.119
        _,_,index=self.reserve();self.budget.mark_dispatched(index)
        self.budget.mark_call_failed(index,TimeoutError('offline'))
        before=deepcopy(self.budget.public_meta())
        with self.assertRaises(budgets._V13OutputCapacityInsufficient):self.run_models()
        self.assertEqual(self.posts,[]);self.assertEqual(self.budget.public_meta(),before)
        self.assertGreater(self.budget.public_meta()['uncertain_cost_usd'],0)

    def test_unknown_previous_hold_remains_after_cheaper_model_fits(self):
        self.budget.max_estimated_cost_usd=.12
        _,_,index=self.reserve();self.budget.mark_dispatched(index)
        self.budget.mark_call_failed(index,TimeoutError('offline'))
        prior=deepcopy(self.budget.call_log[0]);hold=self.budget.public_meta()['uncertain_cost_usd']
        self.run_models()
        self.assertEqual(self.budget.call_log[0],prior)
        self.assertEqual(self.budget.public_meta()['uncertain_cost_usd'],hold)
        self.assertEqual(self.budget.llm_calls,2)
        self.assertEqual(self.budget.retry_allowance_calls,0)
        self.assertFalse(self.budget.public_meta()['accounting_complete'])

    def test_other_budget_guards_abort_instead_of_trying_later_models(self):
        for guard in ('invalid_cost','anomaly','deadline','calls'):
            with self.subTest(guard=guard):
                self.budget=budgets._V13RequestBudget('ask');self.attempted=[];self.posts=[]
                if guard=='invalid_cost':self.budget.max_estimated_cost_usd=float('nan')
                elif guard=='anomaly':self.budget.accounting_anomalies.append('fixture')
                elif guard=='deadline':self.budget.deadline_monotonic=999.0
                else:self.budget.llm_calls=self.budget.max_llm_calls
                with self.assertRaises(budgets._V13BudgetExceeded)as caught:self.run_models()
                self.assertNotIsInstance(caught.exception,budgets._V13OutputCapacityInsufficient)
                self.assertEqual(self.attempted,['gpt-5.6-terra'])
                self.assertEqual(self.posts,[]);self.assertEqual(self.budget.call_log,[])
                self.assertEqual(self.budget.retry_allowance_calls,0)

    def test_legacy_delegate_kwargs_unchanged_when_option_is_absent(self):
        def old_delegate(messages,*,model,json_schema,effort,reasoning_mode,timeout,max_output_tokens,company_id,purpose):
            return {'ok':True}
        with patch.object(self,'delegate',old_delegate):
            self.assertEqual(self.run_models(full=False),({'ok':True},'gpt-5.6-terra'))

    def test_composition_root_wrappers_propagate_only_explicit_requirement(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf-8-sig'))
        names={'_v13_json_models','_v13_responses_json','_openai_chat_json'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
        calls=[]
        fake=SimpleNamespace(**{name:(lambda *a,**kw:calls.append(kw))for name in ('json_models','responses_json','chat_json')})
        namespace={'Optional':Optional,'_openai_transport':fake,'OPENAI_API_KEY':'offline-unused',
            'OPENAI_RESPONSES_URL':'https://offline.invalid/responses','OPENAI_CHAT_URL':'https://offline.invalid/chat',
            'OPENAI_CHAT_MODEL':'legacy','V13_FAST_MODEL':'gpt-5.6-terra','requests':SimpleNamespace(post=deny),
            '_v13_current_budget':lambda:self.budget,'_v13_response_text':transport.response_text,
            '_v13_safety_identifier':transport.safety_identifier,'_normalize_model_candidates':transport.normalize_model_candidates,
            '_V13BudgetExceeded':budgets._V13BudgetExceeded}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'main-wrapper-fixture','exec'),namespace)
        module=ModuleType('machinemind.authority.request_admission');module.before_egress=lambda purpose:None
        shared=dict(json_schema=self.schema,timeout=30,max_output_tokens=3200,purpose='fixture')
        variants=[('_v13_json_models',dict(shared,models=['gpt-5.6-terra'],effort='low',reasoning_mode='',company_id='offline')),
                  ('_v13_responses_json',dict(shared,model='gpt-5.6-terra',effort='low',company_id='offline')),
                  ('_openai_chat_json',dict(shared,model='legacy'))]
        with patch.dict(sys.modules,{module.__name__:module}):
            for name,kwargs in variants:
                namespace[name]([],**kwargs);self.assertNotIn('require_full_output',calls[-1])
                namespace[name]([],**kwargs,require_full_output=True);self.assertIs(calls[-1]['require_full_output'],True)

if __name__=='__main__':unittest.main()
