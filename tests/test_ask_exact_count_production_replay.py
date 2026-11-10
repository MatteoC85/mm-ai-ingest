"""Qualified actual-source Core replay under production native role/cost limits.

R7 normalized decision/24 rows + current canonical PROC002 are real pinned inputs.
The unobserved private parent choice, added-manual ranking, model replies, counts
and provider output usage are explicit fixtures, not a recovered live R8 prompt.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib,json,time,types,unittest
from pathlib import Path
import sys
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from assistant_core_v2 import AssistantCoreV2
from machinemind.ask import routing_runtime,request_completion
from machinemind.ask.request_binding import run_core_request,AskRequestEvidence
from machinemind.infrastructure import request_budget as bm,ask_input_count as counter
from machinemind.retrieval.ask_composition import AskSelection
from machinemind.retrieval.chunk_evidence import build_chunk_read,_source_from_columns
import test_ask_conditions_family_admission as owned
import test_ask_procedure_context_recovery as actual
import test_ask_router_family_context as rf
from test_ask_exact_input_count import CountResponse,deny
W=Path(__file__).resolve().parents[2]

class CaptureComplete(Exception):pass

def replay(case,output_tokens,*,free_post=None,include_real_manual_pool=True,boundary_padding=0):
 """No network unless the parent supplies the explicitly FREE-only callback."""
 record={'output_tokens_fixture':output_tokens,'synthetic_writer_boundary_padding_characters':boundary_padding,'fake_create_posts':0,'count_posts':0,'quotes':[],'count_payloads':[]}
 def wrapper(h):
  prior_path=W/'phase6-ask-correction-r8-20261109/live-client/receipts/R8-A-ASK-02.json'
  assert hashlib.sha256(prior_path.read_bytes()).hexdigest()=='d04c98d67a59a98806e4d950713f92096024b992f6beb21cd83ec3f7e78a862f'
  prior=json.loads(prior_path.read_text(encoding='utf-8-sig'))['response']['meta']['v13_runtime']
  role_path=W/'phase6-ask-correction-r8-20261109/review/native-role-config/ROLE-CONFIG-READONLY.json'
  assert hashlib.sha256(role_path.read_bytes()).hexdigest()=='ea92ce9f1bdd210db1f6126fda872b577370f33cc9f94496056ca07ccf10820c'
  # Restore the real chooser, not the legacy fixture's always-FAST callback.
  h.owner.generation=replace(h.owner.generation,_v13_choose_ask_model=h.m._v13_choose_ask_model)
  case.assertEqual(h.owner.generation.V13_HEAVY_MODEL,'gpt-5.6-sol')
  case.assertEqual(h.owner.generation.V13_FAST_MODEL,'gpt-5.6-terra')
  case.assertEqual(h.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS,6000)
  case.assertEqual(h.owner.generation.V13_FAST_MAX_OUTPUT_TOKENS,3000)
  case.assertEqual(h.m._v13_choose_ask_model(h.request.query,h.prepared,narrow_scope=h.request.narrow_scope)[0],'gpt-5.6-sol')
  if include_real_manual_pool:
   pool_path=W/'phase6-ask-correction-r5-20261109/review/read-only-manual-capture/AURELIS-CURRENT-MANUAL-POOL.json'
   assert hashlib.sha256(pool_path.read_bytes()).hexdigest()=='806142ca4ed85b51d220800b489ca379f952f8185a2f4f54ac7d2d2109f4873e'
   pool=json.loads(pool_path.read_text(encoding='utf-8-sig'));extra=[]
   seen={c['citation_id']for c in h.prepared['candidates']}
   for row in pool['chunks']:
    cid=str(row['bubble_document_id'])+':p'+str(row['page_from'])+'-'+str(row['page_to'])+':c'+str(row['chunk_index'])
    if cid in seen:continue
    body=row['chunk_text'];assert hashlib.sha256(body.encode()).hexdigest()==row['body_sha256']
    extra.append(dict(bubble_document_id=row['bubble_document_id'],chunk_index=row['chunk_index'],page_from=row['page_from'],page_to=row['page_to'],citation_id=cid,
     source_type='document',snippet=body[:800].strip(),chunk_full=body.strip(),similarity=.6,semantic_similarity=.6,embedding_list=[]));seen.add(cid)
   h.allowance|=frozenset(_source_from_columns(h.request.company_id,h.request.machine_id,c['bubble_document_id'])for c in extra)
   rows=[(c['bubble_document_id'],c['chunk_index'],c['page_from'],c['page_to'],c['snippet'],c['chunk_full'],c['similarity'],[],True,h.request.company_id,h.request.machine_id)for c in extra]
   registered=h.session.register(request=h.request,receipt=build_chunk_read(scope=h.scope,kind='dense',rows=rows,candidates=extra,limits=owned.limits_fixture(1,1)[0].limits),current_allowed_sources=h.allowance)
   values=[dict(i.record)for i in h.session.records(request=h.request,handles=registered.records,current_allowed_sources=h.allowance)]
   handles=h.owner._selection[0].records+registered.records
   h.prepared={**h.prepared,'candidates':h.prepared['candidates']+values,'citations':h.prepared['citations']+values}
   h.owner._selection=(AskSelection('candidates',handles),AskSelection('citations',handles))
   record['added_real_manual_records']=len(extra);record['manual_relevance_scores_fixture']=.6
  h.seed_selections=h.owner._selection;neutral=deepcopy(h.prepared)
  h.owner._initial=h.initial
  hooks=replace(h.owner.core.hooks,retrieve_neutral=lambda request:h.owner._add_procedure_catalog(request,deepcopy(neutral)),
   route_semantically=h.owner.route,refine_retrieval=h.owner.refine,prepare_evidence=h.owner.prepare,build_no_evidence=h.owner._build_empty_response)
  h.owner.core=AssistantCoreV2(replace(h.owner.core.hooks,route_semantically=lambda request,data:h.m._assistant_core_router_call(request,data)))
  completion=request_completion.RequestCompletion(runtime=h.m._assistant_core_request_flow_runtime(),payload=types.SimpleNamespace(company_id=h.request.company_id),started=time.monotonic())
  b=completion.budget;record['caps_before']=(b.max_estimated_cost_usd,b.max_llm_calls,b.deadline_monotonic)
  case.assertEqual(b.max_estimated_cost_usd,.25);case.assertEqual(b.route,'unselected')
  for attr in ('estimated_cost_usd','input_tokens','cached_input_tokens','cache_write_tokens','output_tokens','reasoning_tokens','embedding_calls','embedding_input_tokens','embedding_estimated_cost_usd'):
   if hasattr(b,attr):setattr(b,attr,prior.get(attr,0))
  b.llm_calls=prior['llm_calls'];b.call_log=deepcopy(prior['calls']);b.embedding_log=deepcopy(prior['embeddings'])
  prior_calls=deepcopy(b.call_log);prior_embeddings=deepcopy(b.embedding_log)
  token=None;last_count={}
  original_router=h.router
  def model_router(messages,**kw):
   nonlocal token
   value=original_router(messages,**kw)
   # The router's canonical semantic fixture is already parsed by the real main
   # normalizer. Activate the real protected owner before all downstream work;
   # compact-wire protocol itself has its independent R7 regression suite.
   execution,token=routing_runtime.activate()
   return value
  def reply(self,oracle,model,messages,kw):
   selected=kw['models'][0]
   if boundary_padding and kw['purpose']=='ask_final_synthesis':
    messages=deepcopy(messages);messages[0]['content']+='\n'*boundary_padding
   if kw['purpose']=='ask_final_synthesis':
    case.assertEqual(selected,'gpt-5.6-terra');case.assertEqual(kw['effort'],'low')
    case.assertEqual(kw['max_output_tokens'],6000)
   if kw['purpose']=='assistant_core_answer_contract_verifier':case.assertEqual(selected,'gpt-5.6-terra')
   def post(url,**options):
    if url==counter.ENDPOINT:
     record['count_posts']+=1;projection=json.loads(options['data'])
     record['count_payloads'].append({'purpose':kw['purpose'],'model':projection['model'],'projection_sha256':counter._sha(projection),'input_UTF8_bytes':len(options['data'])})
     if free_post is not None:
      response=free_post(url,**options)
      # The production private counter consumes and verifies the real FREE reply.
      return response
     tokens=11000 if kw['purpose']=='ask_final_synthesis' else 14000
     last_count[kw['purpose']]=tokens
     return CountResponse({'object':'response.input_tokens','input_tokens':tokens})
    case.assertEqual(url,counter.CREATE_ENDPOINT);record['fake_create_posts']+=1
    payload=options['json'];used_output=output_tokens if kw['purpose']=='ask_final_synthesis' else 1800
    case.assertGreaterEqual(payload['max_output_tokens'],used_output)
    record['quotes'].append({'purpose':kw['purpose'],'model':selected,'requested_output_tokens':kw['max_output_tokens'],
     'effective_output_tokens':payload['max_output_tokens'],'actual_fixture_output_tokens':used_output,
     'native_reserved_input_tokens':b.call_log[-1]['reserved_input_tokens'],'native_reserved_cost_usd':b.call_log[-1]['reserved_cost_usd'],
     'remaining_before_settlement_usd':b.remaining_cost_usd,'full_payload_sha256':counter._sha(payload)})
    # Controlled positive usage, not a tokenization or live quality assertion.
    input_usage=last_count.get(kw['purpose'],11000 if kw['purpose']=='ask_final_synthesis' else 14000)
    return types.SimpleNamespace(status_code=200,json=lambda:{'status':'completed','service_tier':'default','output_text':json.dumps(oracle),
     'usage':{'input_tokens':input_usage,'output_tokens':used_output}})
   with patch.object(h.m.requests,'post',post):
    result=h.m._v13_responses_json(messages,model=selected,json_schema=kw['json_schema'],effort=kw['effort'],reasoning_mode=kw.get('reasoning_mode',''),
     timeout=kw['timeout'],max_output_tokens=kw['max_output_tokens'],company_id=h.request.company_id,purpose=kw['purpose'],require_full_output=kw.get('require_full_output',False))
   return result,selected
  h.reply=types.MethodType(reply,h)
  def efactory():
   case.assertFalse(h.owner._prepared)
   def review(messages,**kw):h.generated=h.owner._generated_response;return h.reviewer(messages,**kw)
   return replace(h.owner._execution_factory(),_v13_json_models=review,_v13_current_budget=bm._v13_current_budget)
  try:
   with patch.object(h.m,'_v13_json_models',model_router):
    result=run_core_request(h.request,core=AssistantCoreV2(hooks),runtimes=replace(h.owner.runtimes,execution=efactory,
     validation=lambda:replace(h.owner._validation_factory(),_v13_current_budget=bm._v13_current_budget)),
     evidence=AskRequestEvidence(session=h.session,authorize=h.owner.authorize,select=h.owner.select,finalize=h.owner._finalize_core))
   record['status']=result['status'];case.assertEqual(result['status'],'answered',result)
   case.assertEqual(record['fake_create_posts'],2);case.assertLessEqual(record['count_posts'],2)
   case.assertEqual(record['caps_before'],(b.max_estimated_cost_usd,b.max_llm_calls,b.deadline_monotonic))
   case.assertEqual(b.call_log[0],prior_calls[0]);case.assertEqual(b.embedding_log,prior_embeddings)
   case.assertIn('Non sostare sotto carichi sospesi.',h.writer_packets[0]);case.assertIn('Non sostare sotto carichi sospesi.',h.review_packets[0])
   packet=h.owner._response_flow._review_packet
   generation=h.owner._generated_response['meta']['generation_evidence']
   case.assertEqual(generation['source_sha256'],packet.diagnostic(reused=False)['source_sha256'])
   family=[body for key,_,_,body in h.rows if key=='procedure:'+h.chosen or key.startswith('step:')]
   case.assertEqual(len(family),11)
   for body in family:case.assertIn(body,h.writer_packets[0]);case.assertIn(body,h.review_packets[0])
   record['required_family_records']=len(family);record['prepared_records']=len(h.owner._prepared_retrieval['candidates'])
   record['packet']=generation;record['budget']=b.public_meta()
   case.assertEqual(record['budget']['uncertain_cost_usd'],0);case.assertTrue(record['budget']['accounting_complete'])
   case.assertLessEqual(b.committed_cost_usd,.25);case.assertEqual(b.llm_calls,3)
  except bm._V13BudgetExceeded as exc:
   record['status']='budget_exceeded';record['budget']=b.public_meta();record['error_class']=type(exc).__name__
   raise
  finally:
   if token is not None:routing_runtime.deactivate(token)
   completion.close()
  raise CaptureComplete()
 with patch.object(rf.RouterFixture,'run_core',wrapper):
  try:case.test_actual_r7_required_parent_type_keeps_complete_family_and_manual_support()
  except CaptureComplete:pass
 return record

class ExactCountProductionReplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):actual.AskProcedureContextRecoveryTests.setUpClass();cls.m=actual.AskProcedureContextRecoveryTests.m
 @classmethod
 def tearDownClass(cls):actual.AskProcedureContextRecoveryTests.tearDownClass()
 def test_actual_owned_family_and_manuals_use_production_roles_with_800_1500_2600_output_usage(self):
  self.records=[]
  with ExitStack()as stack:
   for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request'):stack.enter_context(patch(name,deny))
   for output in (800,1500,2600):
    with self.subTest(output=output):
     case=actual.AskProcedureContextRecoveryTests();case.m=self.m
     self.records.append(replay(case,output,boundary_padding=60000))
  self.assertTrue(all(row['count_posts']>=1 for row in self.records))
 def test_unpadded_actual_source_packet_fits_old_writer_bound_without_free_writer_count(self):
  with ExitStack()as stack:
   for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request'):stack.enter_context(patch(name,deny))
   for output in (800,1500):
    case=actual.AskProcedureContextRecoveryTests();case.m=self.m
    row=replay(case,output)
    self.assertEqual(row['quotes'][0]['effective_output_tokens'],6000)
    self.assertEqual(row['quotes'][0]['actual_fixture_output_tokens'],output)
    self.assertFalse(any(p['purpose']=='ask_final_synthesis'for p in row['count_payloads']))

if __name__=='__main__':unittest.main(verbosity=2)
