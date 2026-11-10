"""Explicit ASK intent, not a deterministic reclassification of old router output.

Actual R12 A03 normalized semantic fields are replayed with one NEW provider-intent
value. Original pre-router pool/private choice and raw provider replies were not
retained. Real typed intake/readers/owner/writer/reviewer/projection run on pinned
real source text and controlled model replies. No live quality/speed claim.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,re,socket,sqlite3,sys,unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from assistant_core_v2 import normalize_decision
from machinemind.ask import factual_coverage as scope, final_contract
from machinemind.retrieval.context_expansion import assistant_core_enumeration_requested,AssistantCoreEnumerationRequestedRuntime
import test_ask_conditions_family_admission as owned
import test_ask_router_family_context as route
_WORK_CANDIDATES=(Path(__file__).resolve().parents[2],Path(__file__).resolve().parents[3])
W=next((p for p in _WORK_CANDIDATES if (p/'phase6-ask-correction-r12-20261110/live-client/receipts/R12-A-ASK-03.json').is_file()),_WORK_CANDIDATES[0])
R=W/'phase6-ask-correction-r12-20261110/live-client/receipts/R12-A-ASK-03.json'
SOURCE=W/'phase6-ask-source-audit-20261109/current-aurelis-proc002-and-catalog.json'
PAGES=W/'aurelis-incident/BBX-300_40T AME1000SN IT.pages.json'
RECEIPT_SHA='10cbd8a5277c2f253a68e4767a95b660d4c38fcd77973c0a50bdaa97b11a6456'
SOURCE_SHA='89620322db9e553fe738fba1e2a17b81c9dd6c1d361936ee2e996d5d5e1b8ee5'

def deny(*a,**kw):raise AssertionError('NETWORK_SQL_PROVIDER_FORBIDDEN')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def enum(request,decision):return assistant_core_enumeration_requested(request,decision,
 runtime=AssistantCoreEnumerationRequestedRuntime(_normalize_unicode_advanced=lambda s:s,re=re))
def original():
 assert hashlib.sha256(R.read_bytes()).hexdigest()==RECEIPT_SHA
 r=read(R);return deepcopy(r['response']['meta']['assistant_core_v2']),r

def manual_text():
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
 p=read(SOURCE)['procedure'];pages=read(PAGES)
 return '\n'.join([p['short_description'],*[s['description']for s in p['steps'][:3]],
  *[x['text']for x in pages if x['page']in(66,84)]]).strip()

class WholeManualFixture(route.RouterFixture):
 def _register_manual(self,text):
  # Existing short-manual fixture used snippet=full text. Preserve this actual
  # long source body in full while using its exact reader-defined800-char view.
  source=owned.SourceIdentity(owned.SourceScope(self.request.company_id,owned.ScopeLevel.MACHINE,self.request.machine_id),
   owned.SourceType.DOCUMENT,'manual-only',owned.SourceFormat.UNKNOWN)
  self.allowance|=frozenset((source,))
  capture=owned._PageReadCapture(self.scope,owned.limits_fixture(1,1)[0].limits,'structured_direct',snippet_chars=800)
  capture.expect(1,6000);capture.capture([('manual-only',self.request.machine_id,1,text,self.request.company_id,self.request.machine_id)])
  row=dict(bubble_document_id='manual-only',citation_id='manual-only:p1',page_from=1,page_to=1,
   source_type='document',snippet=text[:800],chunk_full=text,similarity=.8,semantic_similarity=.8)
  registered=self.session.register(request=self.request,receipt=capture.finish([row]),current_allowed_sources=self.allowance)
  records=[dict(item.record)for item in self.session.records(request=self.request,handles=registered.records,current_allowed_sources=self.allowance)]
  self.manual_pack=(dict(candidates=records,citations=records,plan={},metrics={}),
   (owned.AskSelection('candidates',registered.records),owned.AskSelection('citations',registered.records)))

class SpecificAssertionsScopeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def test_actual_old_contract_remains_inventory_absent_new_semantic_information(self):
  raw,r=original();before=deepcopy(raw)
  self.assertIs(scope.bind_specific_assertions(raw,enabled=True),raw)
  self.assertEqual(raw,before)
  h=route.RouterFixture(self,choice=None,raw=raw,query=r['response']['meta']['assistant_core_repair']['first_answer_contract']['semantic_verifier']['answer'])
  try:self.assertTrue(enum(h.request,normalize_decision(h.request,raw)))
  finally:h.close()
 def test_new_intent_preserves_every_facet_probe_qualifier_and_policy(self):
  raw,_=original();raw['factual_coverage']='specific_assertions';before=deepcopy(raw)
  result=scope.bind_specific_assertions(raw,enabled=True)
  self.assertEqual(raw,before)
  self.assertNotIn('checklist',result['required_answer_types']);self.assertIn('explanation',result['required_answer_types'])
  for a,b in zip(before['facet_queries'],result['facet_queries']):
   self.assertEqual({k:v for k,v in a.items()if k!='answer_type'},{k:v for k,v in b.items()if k!='answer_type'})
  for k in set(before)-{'required_answer_types','facet_queries'}:self.assertEqual(result[k],before[k])
 def test_absent_unknown_collection_and_maintenance_keep_legacy_contract(self):
  raw,_=original()
  for intent in (None,'useful','exhaustive','unspecified','invented'):
   r=deepcopy(raw)
   if intent is not None:r['factual_coverage']=intent
   self.assertIs(scope.bind_specific_assertions(r,enabled=True),r)
   self.assertIn('checklist',r['required_answer_types'])
 def test_conflicting_operations_partial_unsafe_other_modes_do_not_get_scope(self):
  raw,_=original();raw['factual_coverage']='specific_assertions'
  variants=[]
  for typ in ('ordered_actions','state_sequence','interface_locations','diagnostic_causes'):
   r=deepcopy(raw);r['required_answer_types'].append(typ);variants.append(r)
   r=deepcopy(raw);r['facet_queries'][0]['answer_type']=typ;variants.append(r)
  for key,value in [('request_kind','unsafe_request'),('request_kind','ambiguous'),('effective_mode','root_cause'),('information_task','procedure_segment'),('procedure_scope','conditions_only'),('evidence_state','unsupported'),('evidence_state','needs_clarification')]:
   r=deepcopy(raw);r[key]=value;variants.append(r)
  for r in variants:
   with self.subTest(r=r):
    result=scope.bind_specific_assertions(r,enabled=True)
    self.assertEqual(result['factual_coverage'],'unspecified')
    self.assertEqual(result['required_answer_types'],r['required_answer_types']);self.assertEqual(result['facet_queries'],r['facet_queries'])
  disabled=scope.bind_specific_assertions(raw,enabled=False)
  self.assertEqual(disabled['factual_coverage'],'unspecified');self.assertEqual(disabled['required_answer_types'],raw['required_answer_types'])
 def test_schema_old_default_exact_and_explicit_enum_only_when_opted_in(self):
  s={'schema':{'properties':{},'required':[]}}
  self.assertEqual(scope.router_schema(s)['schema']['properties']['factual_coverage']['enum'],['useful','exhaustive','unspecified'])
  self.assertEqual(scope.router_schema(s,specific_assertions=True)['schema']['properties']['factual_coverage']['enum'],['useful','exhaustive','unspecified','specific_assertions'])
  self.assertEqual(s,{'schema':{'properties':{},'required':[]}})
 def test_actual_router_profile_new_intent_runs_owned_core_writer_judge_projection(self):
  raw,r=original();raw['factual_coverage']='specific_assertions'
  query='Mi confermi che posso caricare una matassa da 2500 kg sull’aspo e che tenere premuto STOP equivale ad avere isolato tutte le energie? Se una di queste premesse è sbagliata, correggila usando la documentazione.'
  body=manual_text();self.assertNotIn('2500',body);self.assertIn('1000 kg',body)
  h=WholeManualFixture(self,choice=None,raw=raw,query=query,manual=body,language='it')
  try:
   normalized=scope.bind_specific_assertions(raw,enabled=True);h.decision=normalize_decision(h.request,normalized)
   h.owner.generation=replace(h.owner.generation,_v13_choose_ask_model=self.m._v13_choose_ask_model)
   def provider(messages,**kw):
    h.calls.append(('writer',deepcopy(messages),deepcopy(kw)));h.writer_packets.append(messages[1]['content'])
    self.assertIn(scope.SPECIFIC_ASSERTIONS_ANSWER_INSTRUCTION,messages[0]['content'])
    return {'answer_status':'answered','grounded_points':[
     {'text':'La portata massima documentata è 1000 kg: la matassa proposta supera il limite e non deve essere caricata.','citation_ids':['manual-only:p1']},
     {'text':'Tenere premuto STOP non equivale all’isolamento delle energie. Per carico e infilaggio applicare le procedure aziendali di isolamento delle energie e impiegare personale addestrato.','citation_ids':['manual-only:p1']}]},kw['models'][0]
   h.owner.generation=replace(h.owner.generation,_v13_json_models=provider)
   def review(messages,**kw):
    self.assertIn(scope.SPECIFIC_ASSERTIONS_ANSWER_INSTRUCTION,messages[0]['content'])
    return owned.OwnedFamilyFixture.reviewer(h,messages,**kw)
   h.reviewer=review
   result=h.run_core()
   self.assertEqual(result['status'],'answered',result)
   self.assertEqual([x[0]for x in h.calls],['router','writer','reviewer'])
   wire=h.calls[0][2]['json_schema']['schema']['properties']
   self.assertIn('specific_assertions',wire.get('factual_coverage',wire.get('fc'))['enum'])
   self.assertIn('1000 kg',result['answer']);self.assertIn('STOP',result['answer']);self.assertNotIn('2500',result['answer'])
   self.assertEqual(result['meta']['assistant_core_validation']['unsupported_numeric_or_code_claims_removed'],[])
   self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
   self.assertFalse(h.owner._prepared_retrieval['assistant_core_decision']['enumeration_requested'])
   self.assertEqual(h.owner._prepared_retrieval['assistant_core_contract']['factual_coverage'],'specific_assertions')
   self.assertIn(body,h.writer_packets[0]);self.assertIn(body,h.review_packets[0])
  finally:h.close()
 def test_real_maintenance_and_conditions_profiles_keep_collection_requirements(self):
  for name in ('R12-A-ASK-02','R11-N-ASK-01'):
   parent='phase6-ask-correction-r12-20261110' if name.startswith('R12') else 'phase6-ask-correction-r11-20261110'
   prior=read(W/parent/'live-client/receipts'/(''+name+'.json'))['response']['meta']['assistant_core_v2']
   raw=deepcopy(prior);raw['factual_coverage']='useful'
   self.assertIs(scope.bind_specific_assertions(raw,enabled=True),raw)
   self.assertEqual(raw['required_answer_types'],prior['required_answer_types'])
   if name=='R12-A-ASK-02':self.assertIn('checklist',raw['required_answer_types'])
   self.assertEqual(raw['facet_queries'],prior['facet_queries'])
   self.assertEqual(raw['preferred_source_types'],prior['preferred_source_types'])
 def test_off_actual_router_keeps_original_schema_and_does_not_enable_intent(self):
  raw,r=original();raw['factual_coverage']='specific_assertions'
  h=route.RouterFixture(self,choice=None,raw=raw)
  try:
   calls=[]
   def provider(messages,**kw):calls.append((messages,kw));return deepcopy(raw),'offline-router'
   with patch.object(self.m,'_v13_json_models',provider):out=self.m._assistant_core_router_call(h.request,h.prepared)
   self.assertEqual(len(calls),1)
   self.assertNotIn('specific_assertions',calls[0][1]['json_schema']['schema']['properties']['factual_coverage']['enum'])
   self.assertNotIn(scope.SPECIFIC_ASSERTIONS_ROUTER_INSTRUCTION,calls[0][0][0]['content'])
   self.assertEqual(out['factual_coverage'],'unspecified');self.assertIn('checklist',out['required_answer_types'])
   for mode in ('root_cause','smart_diagnostic','general'):
    self.assertFalse(scope.is_specific_assertions(raw,requested_mode=mode,effective_mode=mode,request_kind='factual'))
  finally:h.close()
 def test_unowned_metadata_cannot_add_scope_instruction_or_change_model(self):
  from machinemind.ask import generation
  raw,_=original();raw['factual_coverage']='specific_assertions'
  bound=scope.bind_specific_assertions(raw,enabled=True)
  h=route.RouterFixture(self,choice=None,raw=raw)
  try:
   retrieval={'candidates':[deepcopy(h.prepared['candidates'][0])],
    'assistant_core_contract':{**bound,'fail_closed':True},
    'assistant_core_decision':{'request_kind':'factual','effective_mode':'ask','enumeration_requested':False}}
   calls=[]
   def provider(messages,**kw):calls.append((messages,kw));return {'answer_status':'no_sources','grounded_points':[]},kw['models'][0]
   rt=replace(h.owner.generation,_v13_json_models=provider,_v13_sources_block=lambda *a,**kw:'source',_finalize_ask_response_for_ui=lambda r,**kw:r)
   self.assertFalse(rt.preserve_procedure_points)
   generation.generate_ask_response(q=h.request.query,company_id=h.request.company_id,response_language='it',top_k=6,
    retrieval=retrieval,narrow_scope=False,debug=False,runtime=rt)
   self.assertEqual(len(calls),1);self.assertNotIn(scope.SPECIFIC_ASSERTIONS_ANSWER_INSTRUCTION,calls[0][0][0]['content'])
  finally:h.close()
 def test_actual_review_reintroducing_unsupported_premise_still_loses_binding(self):
  _,r=original();first=r['response']['meta']['assistant_core_repair']['first_answer_contract'];semantic=first['semantic_verifier']
  answer=semantic['answer'];projected,removed=self.m._assistant_core_filter_unsupported_claim_sentences(answer,manual_text())
  self.assertIn('2500',removed)
  self.assertEqual(final_contract.answer_digest(projected),first['answer_sha256'])
  self.assertNotIn('una matassa da 2500 kg',projected)
  result=final_contract.final_contract(deterministic={'passed':True},semantic=semantic,reviewed_answer=answer,
   answer=projected,semantic_complete=True,semantic_partial=False)
  self.assertFalse(result['passed']);self.assertEqual(result['reason'],'semantic_answer_changed_after_review')

if __name__=='__main__':
 with patch.object(socket.socket,'connect',deny),patch.object(socket,'create_connection',deny),patch.object(sqlite3,'connect',deny):unittest.main(verbosity=2)
