"""Exact-count capabilities: actual native budget/transport, no external I/O.

Provider token counts/usages are controlled observations. They do not certify
live model output quality or actual tokenization; the parent validates the FREE
full-payload endpoint separately before any paid inference.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib,json,time,types,unittest
from pathlib import Path
import sys
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.infrastructure import ask_input_count as counter,openai_transport as transport,request_budget as budget_module
from machinemind.ask import request_completion,routing_runtime
import test_ask_conditions_family_admission as owned

SCHEMA={'name':'offline_reply','strict':True,'schema':{'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}}
PURPOSE='ask_final_synthesis'
def deny(*a,**kw):raise AssertionError('EXTERNAL_NETWORK_DENIED')
class CountResponse:
 status_code=200
 def __init__(self,value):self.value=value;self.closed=False
 def iter_content(self,**kw):yield json.dumps(self.value).encode()
 def close(self):self.closed=True

class ExactInputCountTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def setUp(self):
  self.stack=ExitStack();self.addCleanup(self.stack.close)
  for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request'):self.stack.enter_context(patch(name,deny))
  self.stack.enter_context(patch.object(self.m,'OPENAI_API_KEY','offline-unused'))
  self.stack.enter_context(patch.object(self.m,'OPENAI_RESPONSES_URL',counter.CREATE_ENDPOINT))
  self.posts=[];self.count_tokens=7000;self.usage_input=7000;self.usage_output=800
  self.counter_reply=None;self.on_count=None;self.on_create=None
  self.stack.enter_context(patch.object(self.m.requests,'post',self.post))
 def activate(self,protected=True):
  if protected:
   self.execution,token=routing_runtime.activate();self.addCleanup(routing_runtime.deactivate,token)
  self.completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),payload=types.SimpleNamespace(company_id='offline-company'),started=time.monotonic())
  self.addCleanup(self.completion.close);self.budget=self.completion.budget
  self.assertEqual(self.budget.route,'unselected')
  return self.budget
 def post(self,url,**kw):
  self.posts.append((url,deepcopy(kw)))
  if url==counter.ENDPOINT:
   self.assertFalse(kw['allow_redirects']);self.assertTrue(kw['stream']);self.assertNotIn('json',kw)
   if self.on_count:self.on_count(kw)
   value=self.counter_reply if self.counter_reply is not None else {'object':'response.input_tokens','input_tokens':self.count_tokens}
   return CountResponse(value)
  self.assertEqual(url,counter.CREATE_ENDPOINT)
  if self.on_create:self.on_create(kw)
  return types.SimpleNamespace(status_code=200,json=lambda:{'status':'completed','service_tier':'default','output_text':'{"answer":"supported"}',
   'usage':{'input_tokens':self.usage_input,'output_tokens':self.usage_output}})
 def call(self,*,size=40000,purpose=PURPOSE,model='gpt-5.6-sol',output=6000,full=False):
  return self.m._v13_responses_json([{'role':'user','content':'Source evidence: '+('é'*size)}],model=model,json_schema=SCHEMA,
   effort='low',timeout=30,max_output_tokens=output,company_id='offline-company',purpose=purpose,require_full_output=full)
 def test_actual_main_unselected_route_uses_owned_count_and_binds_monotone_final_cap(self):
  b=self.activate();b.estimated_cost_usd=.01677029;self.count_tokens=10000
  limits=(b.max_estimated_cost_usd,b.max_llm_calls,b.deadline_monotonic)
  self.assertEqual(self.call(),{'answer':'supported'})
  self.assertEqual([p[0]for p in self.posts],[counter.ENDPOINT,counter.CREATE_ENDPOINT])
  projection=json.loads(self.posts[0][1]['data']);sent=self.posts[1][1]['json']
  self.assertEqual(projection,counter._projection(sent));self.assertNotIn('max_output_tokens',projection)
  self.assertLess(sent['max_output_tokens'],6000);self.assertEqual(sent['service_tier'],'default')
  event=b.public_meta()['input_count_preflight']['events'][0]
  self.assertNotEqual(event['planned_create_sha256'],event['sent_create_sha256'])
  self.assertEqual(event['sent_create_sha256'],counter._sha(sent));self.assertEqual(event['input_projection_sha256'],counter._sha(projection))
  self.assertEqual(b.llm_calls,1);self.assertEqual(b.public_meta()['uncertain_cost_usd'],0)
  self.assertEqual((b.max_estimated_cost_usd,b.max_llm_calls,b.deadline_monotonic),limits)
 def test_small_fitting_ask_and_inactive_Root_Smart_other_roles_do_not_count(self):
  b=self.activate();self.call(size=30,output=800)
  self.assertEqual(len(self.posts),1);self.assertNotIn('input_count_preflight',b.public_meta())
  for purpose in ('assistant_core_v2_semantic_router','assistant_core_root_cause_adjudicator','sd_finalize'):
   with self.subTest(purpose=purpose),self.assertRaises(budget_module._V13BudgetExceeded):self.call(purpose=purpose)
  self.assertEqual(len(self.posts),1)
  self.execution.active=False
  with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(len(self.posts),1)
 def test_full_reviewer_uses_exact_count_and_preserves_complete_output_cap(self):
  b=self.activate();b.estimated_cost_usd=.09
  self.call(purpose='assistant_core_answer_contract_verifier',model='gpt-5.6-terra',output=3200,full=True)
  self.assertEqual(self.posts[-1][1]['json']['max_output_tokens'],3200)
  self.assertEqual(b.call_log[-1]['reserved_input_tokens'],7000)
 def test_failed_invalid_or_oversized_count_keeps_old_refusal_without_paid_POST(self):
  b=self.activate()
  for value in ({'object':'response.input_tokens','input_tokens':True},{'object':'response.input_tokens','input_tokens':0},
   {'object':'response.input_tokens','input_tokens':-1},{'object':'response.input_tokens','input_tokens':7,'extra':'x'},
   {'object':'response.input_tokens','input_tokens':1048577}):
   with self.subTest(value=value):
    # A new actual private owner is necessary for a new role allowance.
    if hasattr(b,'_ask_input_count_session'):del b._ask_input_count_session
    self.counter_reply=value
    with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
    self.assertEqual(b.llm_calls,0)
  self.assertTrue(all(url==counter.ENDPOINT for url,_ in self.posts));self.assertEqual(b.call_log,[])
 def test_count_timeout_no_retry_and_one_attempt_per_role(self):
  import requests
  b=self.activate();self.on_count=lambda kw:(_ for _ in ()).throw(requests.exceptions.ReadTimeout('offline'))
  for _ in range(2):
   with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(len(self.posts),1);self.assertEqual(b.llm_calls,0);self.assertEqual(b.public_meta()['retry_allowance_calls'],0)
 def test_output_alone_call_limit_deadline_and_uncertainty_cannot_be_bypassed(self):
  b=self.activate();b.estimated_cost_usd=.24
  with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(self.posts,[])
  b.estimated_cost_usd=0;b.llm_calls=b.max_llm_calls
  with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(self.posts,[])
  b.llm_calls=0;b.cancel()
  with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(self.posts,[])
 def test_existing_unknown_cost_is_retained_after_successful_exact_count(self):
  b=self.activate();b.call_log.append(dict(call=1,purpose='old_unknown',model='gpt-5.6-luna',accounting_state='uncertain',reserved_cost_usd=.04,dispatched=True));b.llm_calls=1
  self.call();meta=b.public_meta()
  self.assertEqual(meta['uncertain_cost_usd'],.04);self.assertFalse(meta['accounting_complete']);self.assertEqual(meta['retry_allowance_calls'],0)
 def test_count_cannot_outlive_owner_or_deadline_and_rechecks_authority(self):
  b=self.activate();self.on_count=lambda kw:b.cancel()
  with self.assertRaises(budget_module._V13BudgetExceeded):self.call()
  self.assertEqual(len(self.posts),1);self.assertEqual(b.llm_calls,0)
 def test_input_projection_preserves_every_supported_input_field_and_rejects_unknowns(self):
  base={'model':'gpt-5.6-sol','input':[{'role':'developer','content':[{'type':'input_text','text':'é schema'}]}],
   'instructions':' unchanged  ','reasoning':{'effort':'low','context':'current_turn','mode':'pro','summary':'auto','generate_summary':'detailed'},
   'text':{'format':SCHEMA,'verbosity':'high'},'truncation':'disabled','store':False,'service_tier':'default','safety_identifier':'offline','max_output_tokens':6000}
  result=counter._projection(base)
  for key in counter._INPUT_KEYS:self.assertEqual(result[key],base[key])
  for key,value in (('personality','friendly'),('tools',[]),('previous_response_id','old'),('conversation','old'),('parallel_tool_calls',False)):
   with self.subTest(key=key),self.assertRaises(ValueError):counter._projection({**base,key:value})
  with self.assertRaises(ValueError):counter._projection({**base,'truncation':'auto'})
  with self.assertRaises(ValueError):counter._projection({**base,'service_tier':'flex'})
 def test_fake_or_reused_proofs_and_changed_schema_or_body_cannot_authorize_dispatch(self):
  b=self.activate();session=counter.for_owner(b)
  payload={'model':'gpt-5.6-sol','input':[{'role':'user','content':'é'*40000}],'store':False,'service_tier':'default','text':{'format':SCHEMA}}
  proof=session.count(purpose=PURPOSE,payload=payload,requested=6000,api_key='offline-unused',post_fn=self.post,url=counter.CREATE_ENDPOINT,old_error=budget_module._V13BudgetExceeded('capacity'))
  self.assertIsNotNone(proof)
  for invalid in ({'tokens':7},object()):
   with self.assertRaises(budget_module._V13BudgetExceeded):b.reserve_call(model=payload['model'],purpose=PURPOSE,requested_timeout=30,max_output_tokens=6000,messages=payload['input'],request_payload=payload,_input_count_proof=invalid)
  changed=deepcopy(payload);changed['input'][0]['content']+='changed'
  with self.assertRaises(budget_module._V13BudgetExceeded):b.reserve_call(model=payload['model'],purpose=PURPOSE,requested_timeout=30,max_output_tokens=6000,messages=changed['input'],request_payload=changed,_input_count_proof=proof)
  timeout,cap,index=b.reserve_call(model=payload['model'],purpose=PURPOSE,requested_timeout=30,max_output_tokens=6000,messages=payload['input'],request_payload=payload,_input_count_proof=proof)
  final={**payload,'max_output_tokens':cap};counter.finalize(proof,budget=b,payload=final,index=index)
  with self.assertRaises(ValueError):counter.finalize(proof,budget=b,payload=final,index=index)
  with self.assertRaises(budget_module._V13BudgetExceeded):b.reserve_call(model=payload['model'],purpose=PURPOSE,requested_timeout=30,max_output_tokens=6000,messages=payload['input'],request_payload=payload,_input_count_proof=proof)
 def test_final_proof_failure_stops_json_model_fallback_and_preserves_not_sent_row(self):
  b=self.activate()
  with patch.object(counter,'finalize',side_effect=ValueError('changed')):
   with self.assertRaises(budget_module._V13BudgetExceeded):
    self.m._v13_json_models([{'role':'user','content':'é'*40000}],models=['gpt-5.6-sol','gpt-5.6-terra'],json_schema=SCHEMA,
     effort='low',reasoning_mode='',timeout=30,max_output_tokens=6000,company_id='offline-company',purpose=PURPOSE)
  self.assertEqual(len(self.posts),1);self.assertEqual(b.call_log[0]['accounting_state'],'not_sent');self.assertEqual(b.public_meta()['retry_allowance_calls'],0)

 def authority(self):
  from machinemind.authority import request_admission
  from test_authority_catalog_pipeline import AuthorityCatalogPipelineTests
  helper=AuthorityCatalogPipelineTests();fake,directory,authority,grant,scope=helper.fixture(parallel=False)
  self.addCleanup(helper.doCleanups)
  admission=request_admission.RequestAdmission(provider=authority,grant=grant,scope=scope,
   payload=types.SimpleNamespace(company_id='company-a',machine_id='machine-a'),resolve=lambda:None)
  token=request_admission.activate(admission);self.addCleanup(request_admission.deactivate,token)
  return fake,admission
 def test_real_authority_revoked_during_count_stops_paid_POST_and_all_model_fallback(self):
  from machinemind.authority.contracts import AuthorityError
  b=self.activate();fake,admission=self.authority()
  old=dict(call=1,purpose='old_unknown',model='gpt-5.6-luna',accounting_state='uncertain',reserved_cost_usd=.04,dispatched=True)
  b.call_log.append(deepcopy(old));b.llm_calls=1
  self.on_count=lambda kw:setattr(fake,'machine_company','company-other')
  with self.assertRaises(AuthorityError):
   self.m._v13_json_models([{'role':'user','content':'é'*40000}],models=['gpt-5.6-sol','gpt-5.6-terra'],json_schema=SCHEMA,
    effort='low',reasoning_mode='',timeout=30,max_output_tokens=6000,company_id='offline-company',purpose=PURPOSE)
  self.assertEqual(len(self.posts),1);self.assertEqual(self.posts[0][0],counter.ENDPOINT)
  self.assertEqual(b.call_log,[old]);self.assertEqual(b.llm_calls,1)
  self.assertEqual(b.public_meta()['uncertain_cost_usd'],.04);self.assertEqual(b.public_meta()['retry_allowance_calls'],0)
  self.assertEqual(admission.fence_attempts,2);self.assertEqual(admission.fence_failures,1)
 def test_fitting_old_bound_keeps_exactly_one_real_authority_fence(self):
  self.activate();fake,admission=self.authority()
  self.call(size=30,output=800)
  self.assertEqual(admission.fence_attempts,1);self.assertEqual(admission.fence_failures,0)
  self.assertEqual(len(self.posts),1);self.assertEqual(self.posts[0][0],counter.CREATE_ENDPOINT)

 def test_final_output_cap_must_equal_actual_native_reserved_cap_not_just_configured_ceiling(self):
  b=self.activate();b.estimated_cost_usd=.04;session=counter.for_owner(b)
  payload={'model':'gpt-5.6-sol','input':[{'role':'user','content':'é'*40000}],'store':False,'service_tier':'default','text':{'format':SCHEMA}}
  proof=session.count(purpose=PURPOSE,payload=payload,requested=6000,api_key='offline-unused',post_fn=self.post,url=counter.CREATE_ENDPOINT,old_error=budget_module._V13BudgetExceeded('capacity'))
  _,cap,index=b.reserve_call(model=payload['model'],purpose=PURPOSE,requested_timeout=30,max_output_tokens=6000,messages=payload['input'],request_payload=payload,_input_count_proof=proof)
  self.assertLess(cap,6000)
  with self.assertRaises(ValueError):counter.finalize(proof,budget=b,payload={**payload,'max_output_tokens':cap+1},index=index)
  self.assertFalse(b.call_log[-1]['dispatched']);self.assertEqual(len(self.posts),1)
  counter.finalize(proof,budget=b,payload={**payload,'max_output_tokens':cap},index=index)

if __name__=='__main__':unittest.main(verbosity=2)
