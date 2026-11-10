"""Owned finite factual context recall, real-data counterfactual and strict fallback.

Actual R17 public decision and retained 24-source diagnostic subset are fixtures,
not its unretained initial pool/private parent choice. Current Bubble catalog and
PROC002 bodies are pinned. SQL and model replies are controlled; real Core,
canonical compiler, producer, numeric projection, independent review and final
binding execute. No live model/retrieval/quality claim is made.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace
import hashlib,json,re,socket,sqlite3,sys,unittest
C=Path(__file__).resolve().parents[1]
W=next(p for p in C.parents if(p/'phase6-ask-unit-projection-r17-20261110').is_dir())
sys.path[:0]=[str(C),str(C/'tests')]
import test_ask_router_family_context as route
import test_ask_conditions_family_admission as owned
from assistant_core_v2 import normalize_decision
from machinemind.ask import core_intake
from machinemind.retrieval.ask_composition import AskSelection
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.chunk_evidence import build_chunk_read
from machinemind.evidence.contracts import SourceType,SourceIdentity,SourceScope,ScopeLevel,SourceFormat

def checked(p,sha):
 data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 return json.loads(data.decode('utf-8-sig'))
R=checked(W/'phase6-ask-unit-projection-r17-20261110/live-client/receipts/R17-A-ASK-EN-FIX.json',
 '90329fd42e04b2f7bfc1434b3ee2731e6b1f3e6db881e4fbe0fa0ee19f160374')
S=checked(W/'phase6-ask-source-audit-20261109/current-aurelis-proc002-and-catalog.json',
 '89620322db9e553fe738fba1e2a17b81c9dd6c1d361936ee2e996d5d5e1b8ee5')
P=json.loads((W/'phase6-ask-unit-projection-r17-20261110/live-client/plan.ACCEPTED.json').read_text(encoding='utf-8-sig'))['initial_payloads']['R17-A-ASK-EN-FIX']
RAW=deepcopy(R['response']['meta']['assistant_core_v2'])
POOL=R['response']['meta']['assistant_core_evidence_admission']['deterministic_signals']['candidates']
PARENT=S['procedure']['procedure_id']
class FactualFixture(route.RouterFixture):
 def __init__(self,test,*,raw=None,choice=None,rows=None,text=None,review_reject=False,request_updates=None,**kw):
  self.actual_text=text;self.review_reject=review_reject;self.document_valid=False;self.relevance_override=None;self.omit_parent=False
  rows=rows or [dict(procedure_id=p['procedure_id'],title=p['title'],short_description=p['short_description'],
   steps=deepcopy(S['procedure']['steps'])if p['procedure_id']==PARENT else[])
   for p in sorted(S['procedures_catalog'],key=lambda p:p['procedure_id']==PARENT)]
  original=owned.condition_fixtures.AskConditionsPipelineTests.request_decision
  def rd(case,language):
   req,d=original(case,language)
   return replace(req,company_id=P['company_id'],machine_id=P['machine_id'],**(request_updates or{})),d
  with patch.object(owned.condition_fixtures.AskConditionsPipelineTests,'request_decision',rd):
   super().__init__(test,choice=choice,rows=rows,raw=raw or RAW,query=P['query'],**kw)
  self.decision=normalize_decision(self.request,raw or RAW)
  self.structured_rows=deepcopy(self.prepared['candidates']);self.structured_handles=self.owner._selection[0].records
  self.register_actual_pool()
 def register_actual_pool(self):
  sources=frozenset(SourceIdentity(SourceScope(self.request.company_id,ScopeLevel.MACHINE,self.request.machine_id),
   SourceType.DOCUMENT,row['bubble_document_id'],SourceFormat.UNKNOWN)for row in POOL)
  self.allowance|=sources
  limits=owned.limits_fixture(1,1)[0].limits
  rows=[{k:deepcopy(v)for k,v in x.items()if k!='embedding_list'}for x in POOL]
  chunks=[x for x in rows if ':v13page'not in x['citation_id']]
  pages=[x for x in rows if ':v13page'in x['citation_id']]
  sql=[(x['bubble_document_id'],x['chunk_index'],x['page_from'],x['page_to'],x['snippet'],x['chunk_full'],
    x['similarity'],None,None,self.request.company_id,self.request.machine_id)for x in chunks]
  dense=self.session.register(request=self.request,receipt=build_chunk_read(scope=self.scope,kind='dense',
   rows=sql,candidates=chunks,limits=limits),current_allowed_sources=self.allowance)
  capture=_PageReadCapture(self.scope,limits,'structured_direct',snippet_chars=700)
  capture.expect(len(pages),6000)
  capture.capture([(x['bubble_document_id'],self.request.machine_id,x['page_from'],x['chunk_full'],
   self.request.company_id,self.request.machine_id)for x in pages])
  paged=self.session.register(request=self.request,receipt=capture.finish(pages),current_allowed_sources=self.allowance)
  combined=dense.records+paged.records
  observed=self.session.records(request=self.request,handles=combined,current_allowed_sources=self.allowance)
  by_id={item.record['citation_id']:(h,dict(item.record))for h,item in zip(combined,observed)}
  ordered=tuple(by_id[x['citation_id']][0]for x in rows)
  values=[by_id[x['citation_id']][1]for x in rows]
  self.prepared={**self.prepared,'candidates':values,'citations':values}
  self.owner._selection=(AskSelection('candidates',ordered),AskSelection('citations',ordered))
 def router(self,messages,**kw):
  out,model=super().router(messages,**kw)
  if self.relevance_override is not None:out['relevant_evidence_ids']=deepcopy(self.relevance_override)
  if self.omit_parent:out.pop('selected_procedure_parent_id',None)
  return out,model
 def provider(self,messages,**kw):
  self.calls.append(('writer',deepcopy(messages),deepcopy(kw)))
  block=messages[1]['content'].split('SOURCES:\n',1)[1]
  self.writer_packets.append(block)
  cids=re.findall(r'\[([^\]\n]+)\]',block)
  numeric=next((cid for cid in cids if '1785239918835x940708430393348800'in cid),None)
  if numeric is None and self.document_valid:numeric=next((cid for cid in cids if cid=='manual-only:p1'),None)
  if numeric is None:return {'answer_status':'no_sources','grounded_points':[]},'offline-writer'
  self.test.assertIn('1000 kg',block)
  self.test.assertIn('STOP',block)
  text=self.actual_text or 'The documented maximum compatible coil weight is 1000 kg.'
  return {'answer_status':'answered','grounded_points':[
   dict(text=text,citation_ids=[numeric]),
   dict(text='STOP is not complete energy isolation. Only trained personnel may perform loading with the machine stopped and the applicable energy-isolation procedure.',
    citation_ids=[numeric])]},'offline-writer'
 def reviewer(self,messages,**kw):
  if self.owner._selected_catalog_parent is None and not self.document_valid:
   self.calls.append(('reviewer',deepcopy(messages),deepcopy(kw)))
   self.review_packets.append(messages[1]['content'])
   return dict(outcome='no_sources',answer='',reason='No capacity in this fixture source packet.',
    covered_facets=[],missing_facets=list(self.decision.required_facets),
    covered_answer_types=[],missing_answer_types=list(self.decision.required_answer_types),
    expected_list_items=[],covered_list_items=[],missing_list_items=[],enumeration_requested=False,citation_ids=[]),'offline-reviewer'
  reply,model=owned.OwnedFamilyFixture.reviewer(self,messages,**kw)
  if self.review_reject:
   reply.update(outcome='no_sources',reply_mode='replace',answer='',
    covered_facets=[],missing_facets=list(self.decision.required_facets),
    covered_answer_types=[],missing_answer_types=list(self.decision.required_answer_types))
  return reply,model
class AskFactualCatalogContextTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def test_actual_public_profile_and_real_catalog_with_null_parent_recovers_capacity_through_real_core(self):
  f=FactualFixture(self)
  try:
   self.assertEqual(len(f.prepared['candidates']),24)
   self.assertTrue(all(x['source_type']=='document'for x in f.prepared['candidates']))
   self.assertFalse(any('1000 kg'in x['chunk_full']for x in f.prepared['candidates']))
   result=f.run_core()
   self.assertEqual(result['status'],'answered',result)
   self.assertIn('1000 kg',result['answer'])
   self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer'])
   self.assertTrue(f.owner._response_flow.completed)
   self.assertTrue(f.owner._procedure_family_context)
   for _,_,_,body in f.rows:
    if body.startswith('SOURCE_TYPE: step'):
     self.assertIn(body,f.writer_packets[0]);self.assertIn(body,f.review_packets[0])
   self.assertIn('STOP',f.review_packets[0]);self.assertTrue(f.refine_calls)
   self.assertNotIn('selected_procedure_parent_id',result['meta']['assistant_core_v2'])
  finally:f.close()
 def test_explicit_nonexclusive_none_policy_reuses_same_real_owned_family_and_judge(self):
  f=FactualFixture(self,raw={**RAW,'source_type_policy':'none'})
  try:
   result=f.run_core()
   self.assertEqual(result['status'],'answered',result)
   self.assertIn('1000 kg',result['answer'])
   self.assertTrue(f.owner._procedure_family_context)
   self.assertIn('1000 kg',f.writer_packets[0]);self.assertIn('1000 kg',f.review_packets[0])
   self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer'])
  finally:f.close()
 def test_require_document_invalid_or_missing_parent_and_legacy_procedure_null_are_not_upgraded(self):
  for updates,choice in [(dict(source_type_policy='require'),None),({},'foreign'),({},13),
    (dict(request_kind='procedure'),None)]:
   with self.subTest(updates=updates,choice=choice):
    f=FactualFixture(self,raw={**RAW,**updates},choice=choice)
    try:
     result=f.run_core();self.assertEqual(result['status'],'no_sources')
     self.assertFalse(f.owner._procedure_family_context)
    finally:f.close()
 def test_ambiguous_title_or_one_term_probes_do_not_select_a_family(self):
  for mode in ('ambiguous','split'):
   raw=deepcopy(RAW);rows=None
   if mode=='split':
    raw['facet_queries'][0].update(lexical_queries=['matassa','aspo'],dense_queries=[])
   else:
    rows=[dict(procedure_id='other-current',title=S['procedure']['title'],short_description='Different current operation.',steps=[]),
     dict(procedure_id=PARENT,title=S['procedure']['title'],short_description=S['procedure']['short_description'],steps=deepcopy(S['procedure']['steps']))]
   f=FactualFixture(self,raw=raw,rows=rows)
   try:
    self.assertEqual(f.run_core()['status'],'no_sources');self.assertFalse(f.owner._procedure_family_context)
   finally:f.close()
 def test_current_revocation_cannot_be_replaced_by_parent_title(self):
  f=FactualFixture(self)
  try:
   original=f.router
   def revoke(messages,**kw):
    out=original(messages,**kw)
    f.allowance=frozenset(s for s in f.allowance if s.source_id!=PARENT)
    return out
   f.router=revoke
   with self.assertRaises(Exception):f.run_core()
   self.assertEqual([x[0]for x in f.calls],['router'])
  finally:f.close()
 def test_guard_scope_and_all_mandatory_types_without_keyword_rules(self):
  for change in ('narrow','no_numeric','mandatory_checklist','degraded','root','smart'):
   f=FactualFixture(self)
   try:
    f.prepared=f.owner._add_procedure_catalog(f.request,f.prepared)
    f.owner._procedure_catalog_null_choice=True
    d=normalize_decision(f.request,RAW)
    if change=='narrow':
     self.assertIsNone(f.owner._factual_catalog_parent(replace(f.request,narrow_scope=True),d));continue
    if change=='no_numeric':d=replace(d,required_answer_types=('explanation',),facet_queries=tuple(replace(x,answer_type='explanation')for x in d.facet_queries))
    if change=='mandatory_checklist':d=replace(d,facet_queries=d.facet_queries+(replace(d.facet_queries[0],answer_type='checklist'),))
    if change=='degraded':d=replace(d,degraded=True)
    if change in ('root','smart'):d=replace(d,effective_mode='root_cause'if change=='root'else'smart_diagnostic')
    self.assertIsNone(f.owner._factual_catalog_parent(f.request,d))
   finally:f.close()
 def test_independent_reject_is_still_terminal_with_complete_source_family(self):
  f=FactualFixture(self,review_reject=True)
  try:
   self.assertEqual(f.run_core()['status'],'no_sources')
   self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer','reviewer'])
  finally:f.close()
 def test_r15_and_r16_actual_numeric_writer_forms_keep_guard_and_final_binding_on_family_lane(self):
  traces=[(W/'phase6-ask-english-forensics-20261110/review/ACTUAL-GENERATED-ANSWER-TRACE-v2.json',
    '76d7cd9291fa650bcee1f57945de676b56e7b226d8c044109400906810b6a427'),
   (W/'phase6-ask-postreview-correction-r16-20261110/review/ACTUAL-R16-GENERATED-ANSWER-TRACE.json',
    '03a0ef832c3cfdcc54104be693a6db2d9c5316dd0e09229e35de3ca262a8bba2')]
  # Exact first numeric generated points; citations are deliberately rebound to
  # this current family capacity occurrence, not forged historical receipt IDs.
  texts=[checked(path,sha)['generated_answer_trace']['first_writer_grounded_points'][0]['text']['text']for path,sha in traces]
  for text in texts:
   with self.subTest(text=text):
    f=FactualFixture(self,text=text)
    try:
     result=f.run_core();self.assertEqual(result['status'],'answered',result)
     self.assertIn('1000 kg',result['answer']);self.assertNotIn('2500',result['answer'])
     self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer'])
     self.assertTrue(f.owner._response_flow.completed)
    finally:f.close()
 def test_optional_oversized_family_keeps_existing_complete_document_path_and_limits(self):
  rows=[dict(procedure_id=p['procedure_id'],title=p['title'],short_description=p['short_description'],
   steps=deepcopy(S['procedure']['steps'])if p['procedure_id']==PARENT else[])
   for p in sorted(S['procedures_catalog'],key=lambda p:p['procedure_id']==PARENT)]
  f=FactualFixture(self,rows=rows)
  try:
   # Change only this controlled subsequent canonical SQL fixture. The initial
   # fixture's exact small seed projection stays valid and is not an input seed.
   key='procedure:'+PARENT
   f.children[key]=[row[:4]+(row[4]+'\nDocumented supporting source context. '*85,)for row in f.children[key]]
   f.document_valid=True
   f._register_manual('The documented maximum compatible coil weight is 1000 kg. STOP is not complete energy isolation. Only trained personnel may perform loading with the machine stopped and the applicable energy-isolation procedure.')
   before=(f.owner.generation.V13_FAST_CONTEXT_CHARS,f.owner.generation.V13_HEAVY_CONTEXT_CHARS,
    f.owner.generation.V13_FAST_MAX_OUTPUT_TOKENS,f.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS)
   result=f.run_core()
   self.assertEqual(result['status'],'answered',result);self.assertIn('1000 kg',result['answer'])
   self.assertFalse(f.owner._procedure_family_context);self.assertTrue(f.owner._factual_catalog_declined)
   self.assertNotIn('Documented supporting source context.',f.writer_packets[0])
   self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer'])
   self.assertEqual(before,(f.owner.generation.V13_FAST_CONTEXT_CHARS,f.owner.generation.V13_HEAVY_CONTEXT_CHARS,
    f.owner.generation.V13_FAST_MAX_OUTPUT_TOKENS,f.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS))
  finally:f.close()
 def test_family_between_fast_and_review_limits_cannot_inflate_postP4_fast_writer(self):
  f=FactualFixture(self)
  try:
   f.document_valid=True
   f._register_manual('The documented maximum compatible coil weight is 1000 kg. STOP is not complete energy isolation. Only trained personnel may perform loading with the machine stopped and the applicable energy-isolation procedure.')
   gen=f.owner.generation;fast=gen.V13_FAST_CONTEXT_CHARS
   review=min(24000,max(fast,22000));self.assertLess(fast,review)
   key='procedure:'+PARENT;unit='Generic documented source detail. '
   raw=[dict(bubble_document_id=row[0],citation_id=row[0]+':fixture',source_type='step'if row[0].startswith('step:')else'procedure',chunk_full=row[3],snippet=row[3])for row in f.rows if row[0]==key or row[0].startswith('step:')]
   base_chars=len(gen._v13_sources_block(raw,max_context_chars=sys.maxsize))
   n=max(1,(fast+1000-base_chars)//(len(f.children[key])*len(unit)))
   f.children[key]=[row[:4]+(row[4]+unit*n,)for row in f.children[key]]
   sizes=[];policies=[]
   def render(rows,**kw):
    result=gen._v13_sources_block(rows,**kw)
    if kw.get('max_context_chars')==sys.maxsize and len(rows)==11:sizes.append(len(result))
    return result
   def changing_policy(query,retrieval,**kw):
    prepared='assistant_core_decision'in retrieval;policies.append(prepared)
    return (gen.V13_FAST_MODEL if prepared else gen.V13_HEAVY_MODEL),'low',''
   f.owner.generation=replace(gen,_v13_sources_block=render,_v13_choose_ask_model=changing_policy)
   result=f.run_core()
   self.assertEqual(result['status'],'answered',result);self.assertIn('1000 kg',result['answer'])
   self.assertTrue(sizes);self.assertGreater(sizes[0],fast);self.assertLess(sizes[0],review)
   self.assertFalse(f.owner._procedure_family_context);self.assertTrue(f.owner._factual_catalog_declined)
   self.assertEqual(policies,[True])
   self.assertEqual([x[0]for x in f.calls],['router','writer','reviewer'])
  finally:f.close()
 def test_missing_parent_or_invalid_relevance_does_not_become_empty_null_selection(self):
  for mode in ('missing_parent','invalid_relevance','wrong_relevance_type'):
   f=FactualFixture(self)
   try:
    if mode=='missing_parent':f.omit_parent=True
    elif mode=='invalid_relevance':f.relevance_override=['unadmitted-Step-occurrence']
    else:f.relevance_override='not a list'
    self.assertEqual(f.run_core()['status'],'no_sources')
    self.assertFalse(f.owner._procedure_family_context)
   finally:f.close()
 def test_current_selected_step_is_preserved_instead_of_acquiring_another_family(self):
  f=FactualFixture(self)
  try:
   row,handle=next((x,h)for x,h in zip(f.structured_rows,f.structured_handles)
    if x['bubble_document_id']=='step:'+S['procedure']['steps'][0]['step_id'])
   f.prepared={**f.prepared,'candidates':[row]+f.prepared['candidates'],'citations':[row]+f.prepared['citations']}
   f.owner._selection=tuple(AskSelection(s.name,(handle,)+s.records)for s in f.owner._selection)
   f.relevance_override=[row['citation_id']]
   self.assertEqual(f.run_core()['status'],'no_sources')
   self.assertFalse(f.owner._procedure_family_context)
   self.assertFalse(f.owner._procedure_catalog_null_choice)
   self.assertEqual(f.relevance_override,[row['citation_id']])
   self.assertFalse(any(x[0]=='read_related_step_page_evidence'for x in f.sql))
  finally:f.close()
 def test_parent_hidden_by_snippet_deduplication_cannot_claim_complete_catalog(self):
  parent=deepcopy(S['procedure']);parent['procedure_id']='other-identical-current-parent';parent['steps']=[]
  actual=dict(procedure_id=PARENT,title=S['procedure']['title'],short_description=S['procedure']['short_description'],steps=deepcopy(S['procedure']['steps']))
  f=FactualFixture(self,rows=[parent,actual])
  try:
   self.assertEqual(f.run_core()['status'],'no_sources')
   self.assertFalse(f.owner._procedure_family_context)
  finally:f.close()
if __name__=='__main__':
 def denied(*a,**k):raise AssertionError('REMOTE_OR_SQL_FORBIDDEN')
 with ExitStack()as stack:
  for name in ('socket.create_connection','socket.socket.connect','sqlite3.connect'):stack.enter_context(patch(name,denied))
  unittest.main(verbosity=2)
