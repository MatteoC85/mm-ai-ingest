"""ASK-local attempt allocation; provider replies/transport failures are fixtures.

Real Core ownership and request-budget accounting run with external I/O denied.
No test asserts that a provider will respond within the allocated ceiling.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys,time,unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import routing_runtime as policy,request_completion
from machinemind.infrastructure import openai_transport,request_budget
import test_ask_conditions_family_admission as owned
import test_ask_router_family_context as core_fixture

def deny(*a,**k):raise AssertionError('OFFLINE_ROUTER_POLICY_IO_OR_DOWNSTREAM_DENIED')

class AskRouterAttemptPlanTests(unittest.TestCase):
 def activate(self):
  execution,token=policy.activate();self.addCleanup(policy.deactivate,token);return execution
 def plan(self,models=('preferred','old-fallback'),timeout=15,**kw):
  return policy.router_attempt_plan('ask',list(models),timeout,
   allowed_effective_modes=kw.pop('allowed_effective_modes',('ask',)),compact_wire=kw.pop('compact_wire',True),**kw)
 def test_only_owned_compact_ask_exact_tuple_can_opt_in(self):
  self.assertIsNone(self.plan())
  self.activate()
  for compact,modes in ((False,('ask',)),('true',('ask',)),(1,('ask',)),(True,['ask']),
    (True,('ask','root_cause')),(True,())):
   with self.subTest(compact=compact,modes=modes):
    self.assertIsNone(self.plan(compact_wire=compact,allowed_effective_modes=modes))
  for mode in ('root_cause','smart_diagnostic','other'):
   self.assertIsNone(policy.router_attempt_plan(mode,['p','f'],15,allowed_effective_modes=('ask',),compact_wire=True))
 def test_one_already_preferred_existing_model_without_reordering_or_new_model(self):
  self.activate();models=['existing-preferred','existing-old-primary'];before=deepcopy(models)
  plan=policy.router_attempt_plan('ask',models,15,allowed_effective_modes=('ask',),compact_wire=True)
  self.assertEqual(models,before)
  self.assertEqual(plan['models'],[models[0]])
  self.assertEqual(plan['attempt_limit'],1);self.assertFalse(plan['automatic_fallback'])
  self.assertEqual(plan['timeout_seconds'],20)
  self.assertEqual(plan['previous_stage_allowance_seconds'],30)
 def test_timeout_borrows_only_old_stage_allowance_with_twenty_second_ceiling(self):
  self.activate()
  for models,timeout,expected in ((['p'],15,15),(['p','f'],15,20),(['p','f'],3,6),
    (['p','p'],15,15),(['p'],30,20),(['p','f'],30,20)):
   with self.subTest(models=models,timeout=timeout):
    plan=self.plan(models,timeout)
    self.assertEqual(plan['timeout_seconds'],expected)
    self.assertLessEqual(plan['timeout_seconds'],plan['previous_stage_allowance_seconds'])
 def test_invalid_active_configuration_fails_without_adding_default_models(self):
  self.activate()
  for models in ([],[''],['p','f','third'],'p',[None]):
   with self.subTest(models=models),self.assertRaises(ValueError):
    policy.router_attempt_plan('ask',models,15,allowed_effective_modes=('ask',),compact_wire=True)
  for timeout in (True,0,-1,1.5):
   with self.subTest(timeout=timeout),self.assertRaises(ValueError):self.plan(timeout=timeout)
 def test_closed_owner_has_no_policy_and_error_string_contains_no_caller_detail(self):
  execution=self.activate();execution.active=False
  self.assertIsNone(self.plan())
  self.assertEqual(str(policy.AskRouterUnavailable()),'ASK_SEMANTIC_ROUTER_UNAVAILABLE')

class RealRouterAttemptBoundaryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def setUp(self):
  self.guard=ExitStack();self.addCleanup(self.guard.close)
  for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request'):
   self.guard.enter_context(patch(name,deny))
  for name,value in (('ASSISTANT_CORE_ROUTER_MODEL','gpt-5.6-terra'),
   ('ASSISTANT_CORE_ROUTER_FALLBACK_MODEL','gpt-5.6-luna'),
   ('ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS',15),('ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS',3000)):
   self.guard.enter_context(patch.object(self.m,name,value))
 def activate(self):
  execution,token=policy.activate();self.addCleanup(policy.deactivate,token);return execution
 def request(self,mode='ask',allowed=None):
  return self.m.AssistantCoreRequest(query='Why is the mechanism stopped?',requested_mode=mode,
   allowed_effective_modes=allowed or (mode,),response_language='en',company_id='offline-company',
   machine_id='offline-machine',ai_scope='machine_all',top_k=6)
 def decision(self,mode):
  return dict(effective_mode=mode,request_kind='factual' if mode=='ask' else 'guided_diagnostic',
   confidence=.9,evidence_state='supported',evidence_policy='machine_sources_required',
   information_task='document_explanation' if mode=='ask' else 'fault_diagnostic',
   required_answer_types=['explanation'] if mode=='ask' else ['diagnostic_causes'],
   relevant_evidence_ids=[],dense_queries=[],lexical_queries=[],required_facets=[])
 def capture(self,request):
  captured=[]
  def provider(messages,**kw):captured.append((deepcopy(messages),deepcopy(kw)));return self.decision(request.requested_mode),'fixture'
  with patch.object(self.m,'_v13_json_models',provider):self.m._assistant_core_router_call(request,{'candidates':[]})
  self.assertEqual(len(captured),1);return captured[0]
 def test_Root_Smart_and_inactive_full_wire_keep_same_models_schema_limits_and_prompt(self):
  completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
   payload=SimpleNamespace(company_id='offline-company'),started=time.monotonic())
  self.addCleanup(completion.close)
  baseline={mode:self.capture(self.request(mode)) for mode in ('root_cause','smart_diagnostic')}
  self.activate()
  for mode,expected in baseline.items():
   with self.subTest(mode=mode):self.assertEqual(self.capture(self.request(mode)),expected)
  # No exact owned ASK-only permission: this is the existing full-wire branch.
  request=self.request('ask',('ask','root_cause'))
  call=self.capture(request)
  self.assertEqual(call[1]['models'],['gpt-5.6-luna','gpt-5.6-terra'])
  self.assertEqual(call[1]['timeout'],15)
  self.assertEqual(call[1]['max_output_tokens'],3000)
 def test_real_Core_timeout_has_one_POST_retains_uncertainty_and_stops_before_refine(self):
  import requests
  execution=self.activate();h=core_fixture.RouterFixture(self)
  posts=[];captured={}
  def provider(messages,**kw):
   h.calls.append(('router',deepcopy(messages),deepcopy(kw)))
   self.assertEqual(kw['models'],['gpt-5.6-luna']);self.assertEqual(kw['timeout'],20)
   self.assertEqual(kw['max_output_tokens'],3000)
   budget=request_budget._v13_current_budget();captured['budget']=budget
   # A synthetic completed embedding gives a known charge that must survive.
   _,index=budget.reserve_embedding(model='text-embedding-3-small',texts=['fixture'],requested_timeout=12)
   budget.mark_dispatched(index,embedding=True);budget.record_embedding_usage(index,{'prompt_tokens':3,'total_tokens':3})
   captured['known']=budget.estimated_cost_usd
   captured['caps']=(budget.deadline_monotonic,budget.max_estimated_cost_usd,budget.max_llm_calls)
   def post(url,**request):
    posts.append(request);self.assertEqual(request['json']['service_tier'],'default')
    raise requests.exceptions.ReadTimeout('offline-router-timeout')
   def responses(messages,**options):
    return openai_transport.responses_json(messages,**options,api_key='offline-unused',
     url='https://api.openai.com/v1/responses',post_fn=post,current_budget_fn=lambda:budget,
     response_text_fn=openai_transport.response_text,safety_identifier_fn=openai_transport.safety_identifier)
   return openai_transport.json_models(messages,**kw,default_model='gpt-5.6-terra',
    normalize_models_fn=openai_transport.normalize_model_candidates,current_budget_fn=lambda:budget,
    responses_json_fn=responses,chat_json_fn=deny,budget_exceeded_type=request_budget._V13BudgetExceeded)
  h.router=provider
  try:
   with patch.object(h.owner,'refine',side_effect=deny),self.assertRaises(policy.AskRouterUnavailable) as raised:h.run_core()
   budget=captured['budget'];meta=budget.public_meta()
   self.assertEqual(len(posts),1);self.assertEqual([x[0] for x in h.calls],['router'])
   self.assertEqual(len(meta['calls']),1);self.assertEqual(meta['calls'][0]['accounting_state'],'uncertain')
   self.assertGreater(meta['uncertain_cost_usd'],0);self.assertFalse(meta['accounting_complete'])
   self.assertEqual(budget.estimated_cost_usd,captured['known'])
   self.assertEqual((budget.deadline_monotonic,budget.max_estimated_cost_usd,budget.max_llm_calls),captured['caps'])
   self.assertEqual(meta['retry_allowance_calls'],0)
   response=self.m._assistant_core_technical_error(requested_mode='ask',q=h.request.query,
    language=h.request.response_language,budget=budget,exc=raised.exception)
   response=execution.finish(response)
   self.assertEqual(response['status'],'error');self.assertFalse(response['ok'])
   self.assertFalse(response['meta']['protected_execution']['accounting']['complete'])
   self.assertFalse(response['meta']['protected_execution']['accounting']['cache_eligible'])
   self.assertEqual(response['meta']['protected_execution']['accounting']['uncertain_reserved_usd'],meta['uncertain_cost_usd'])
   self.assertEqual(response.get('citations',[]),[])
  finally:h.close()

if __name__=='__main__':unittest.main(verbosity=2)
