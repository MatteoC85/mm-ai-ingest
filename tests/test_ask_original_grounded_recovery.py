"""Real R15 generated text replay; controlled provider verdicts, no live-quality claim."""
from pathlib import Path
import sys,json,hashlib,socket,sqlite3,time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from assistant_core_v2 import AssistantCoreV2
from machinemind.ask import execution,validation,final_contract,request_completion
from machinemind.ask.request_binding import AskRequestEvidence,run_core_request
from machinemind.evidence.ask_input import ask_request_key
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_conditions_family_admission as owned
import test_ask_specific_assertions_scope as fixture
W=next(p for p in (ROOT.parent,ROOT.parents[1],ROOT.parents[2])if(p/'phase6-ask-english-forensics-20261110').is_dir())
TRACE=W/'phase6-ask-english-forensics-20261110/review/ACTUAL-GENERATED-ANSWER-TRACE-v2.json'
RECEIPT=W/'phase6-ask-english-forensics-20261110/live-client/receipts/R15-A-ASK-EN-DIAG.json'
TRACE_SHA='76d7cd9291fa650bcee1f57945de676b56e7b226d8c044109400906810b6a427'
RECEIPT_SHA='a356becbb9de2a557f46746e3a579b97f359fceaad27847f1b32bbe63bcf84e2'
def deny(*a,**k):raise AssertionError('NETWORK_SQL_PROVIDER_FORBIDDEN')
def load(p,digest):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==digest
 return json.loads(b.decode('utf-8-sig'))
class RecoveryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
  cls.trace=load(TRACE,TRACE_SHA)['generated_answer_trace'];cls.receipt=load(RECEIPT,RECEIPT_SHA)
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def replay(self,*,second='pass',first='rewrite',original=None,missing=False,revoke=False,disable=False,no_budget=False):
  raw=deepcopy(self.receipt['response']['meta']['assistant_core_v2'])
  query='Can you confirm that I can load a 2500 kg coil onto the decoiler and that holding STOP is equivalent to isolating all energy sources? If either premise is wrong, correct it using the documentation.'
  # Retain the actual model designation in its real manual page as well as the
  # operational source fixture. Text/number support is not fabricated here.
  pages=json.loads(fixture.PAGES.read_text(encoding='utf-8-sig'))
  self.assertEqual(hashlib.sha256(fixture.SOURCE.read_bytes()).hexdigest(),fixture.SOURCE_SHA)
  procedure=json.loads(fixture.SOURCE.read_text(encoding='utf-8-sig'))['procedure']
  body='\n'.join([procedure['short_description'],*[s['description']for s in procedure['steps'][:3]],
    *[p['text']for p in pages if p['page']in(80,81,84)]]).strip()
  h=fixture.WholeManualFixture(self,choice=None,raw=raw,query=query,manual=body)
  h.calls=[];checks=[];budget=self.m._assistant_core_new_budget('ask')
  points=[dict(text=p['text']['text'],citation_ids=['manual-only:p1'])for p in self.trace['first_writer_grounded_points']]
  if original is not None:points=[dict(text=original,citation_ids=['manual-only:p1'])]
  def writer(messages,**kw):
   h.calls.append(('writer',deepcopy(messages),deepcopy(kw)));h.writer_packets.append(messages[1]['content'])
   return {'answer_status':'answered','grounded_points':deepcopy(points)},kw['models'][0]
  def review(messages,**kw):
   h.calls.append(('reviewer',deepcopy(messages),deepcopy(kw)));h.review_packets.append(messages[1]['content'])
   current=messages[1]['content'].split('CURRENT_ANSWER:\n',1)[1].split('\n\nSOURCES:',1)[0]
   nth=len(h.review_packets)
   outcome=first if nth==1 else second
   body=self.trace['captured_first_actual_review_replacement']['text'] if nth==1 and first=='rewrite' else ''
   if nth==1 and first=='validrewrite':
    outcome='rewrite'
    body='The documented maximum load is 1000 kg.\nHolding STOP is not energy isolation. Stop the cycle and apply company energy-isolation procedures using trained personnel.'
   if nth>1 and second=='rewrite':body=self.trace['latest_public_meta_semantic_replacement']['text']
   if nth>1 and second=='changedpass':outcome='pass';body=current+'\nForged added content.'
   facets=list(h.owner._synthesis_decision.required_facets);types=list(h.owner._synthesis_decision.required_answer_types)
   if nth>1:
    checks.append(current)
    if missing:types=[x for x in types if x!='safety_conditions']
   if revoke and nth==1:h.allowance=frozenset(s for s in h.allowance if s.source_id!='manual-only')
   if no_budget and nth==1:budget.llm_calls=budget.max_llm_calls
   return {'outcome':outcome,'answer':body,'reason':'Controlled actual-rewrite replay, second judgment is independent.',
    'covered_facets':facets,'missing_facets':[],
    'covered_answer_types':types,'missing_answer_types':['safety_conditions']if missing and nth>1 else [],
    'expected_list_items':[],'covered_list_items':[],'missing_list_items':[],'enumeration_requested':False,
    'citation_ids':['manual-only:p1']},kw['models'][0]
  h.owner.generation=replace(h.owner.generation,_v13_json_models=writer)
  h.owner.core=AssistantCoreV2(replace(h.owner.core.hooks,route_semantically=lambda req,data:self.m._assistant_core_router_call(req,data)))
  h.seed_selections=h.owner._selection;neutral=deepcopy(h.prepared);h.owner._initial=h.initial
  def er():
   return replace(h.owner._execution_factory(),_v13_json_models=review,_v13_current_budget=lambda:budget)
  hooks=replace(h.owner.core.hooks,retrieve_neutral=lambda req:h.owner._add_procedure_catalog(req,deepcopy(neutral)),
   route_semantically=h.owner.route,refine_retrieval=h.owner.refine,prepare_evidence=h.owner.prepare,build_no_evidence=h.owner._build_empty_response)
  completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
   payload=SimpleNamespace(company_id=h.request.company_id),started=time.monotonic())
  try:
   with patch.object(self.m,'_v13_json_models',h.router),(patch.object(final_contract,'numeric_recovery_scope',return_value=False)if disable else nullcontext()):
    result=run_core_request(h.request,core=AssistantCoreV2(hooks),
     runtimes=replace(h.owner.runtimes,execution=er,
      validation=lambda:replace(h.owner._validation_factory(),_v13_current_budget=lambda:budget)),
     evidence=AskRequestEvidence(session=h.session,authorize=h.owner.authorize,select=h.owner.select,finalize=h.owner._finalize_core))
   return result,h.calls,checks,h
  except Exception:
   h.close();raise
  finally:completion.close()
 def test_actual_rewrite_old_failure_and_explicit_second_pass_recovers_exact_grounded_original(self):
  old,calls,checks,h=self.replay(disable=True,second='rewrite')
  try:self.assertEqual(old['status'],'no_sources');self.assertNotIn('1000 kg',checks[0])
  finally:h.close()
  result,calls,checks,h=self.replay()
  try:
   self.assertEqual(result['status'],'answered',result)
   expected=self.trace['captured_CURRENT_ANSWER_before_first_consumer']['text']
   self.assertEqual(checks,[expected]);self.assertEqual(result['answer'],expected)
   self.assertEqual([x[0]for x in calls],['router','writer','reviewer','reviewer'])
   final=result['meta']['assistant_core_validation']['answer_contract']
   self.assertTrue(final['semantic_answer_bound']);self.assertTrue(final['complete'])
   self.assertEqual(final['semantic_verifier']['review_profile'],'grounded_original_retain_only_v1')
   first,second=[x for x in calls if x[0]=='reviewer']
   self.assertEqual(first[2]['models'],second[2]['models']);self.assertEqual(first[2]['max_output_tokens'],second[2]['max_output_tokens'])
   self.assertEqual(second[2]['json_schema']['schema']['properties']['outcome']['enum'],['pass','no_sources'])
   self.assertEqual(second[2]['json_schema']['schema']['properties']['answer']['enum'],[''])
   self.assertIsNone(getattr(h.owner._response_flow,'_grounded_numeric_recovery',None))
  finally:h.close()
 def test_second_reject_rewrite_changed_pass_and_missing_coverage_never_return_original(self):
  for options in ({'second':'no_sources'},{'second':'rewrite'},{'second':'changedpass'},{'missing':True}):
   with self.subTest(options=options):
    result,calls,checks,h=self.replay(**options)
    try:self.assertEqual(result['status'],'no_sources',result);self.assertEqual(len(checks),1);self.assertEqual(len(calls),4)
    finally:h.close()
 def test_normal_first_pass_and_valid_substantive_rewrite_remain_one_judge(self):
  for first in ('pass','validrewrite'):
   with self.subTest(first=first):
    result,calls,checks,h=self.replay(first=first)
    try:
     self.assertEqual(result['status'],'answered');self.assertEqual(len(calls),3);self.assertFalse(checks)
     self.assertNotIn('original_retain_only',calls[-1][2]['json_schema']['name'])
    finally:h.close()
 def test_original_with_numeric_but_missing_safety_must_earn_new_judgment(self):
  result,calls,checks,h=self.replay(original='The documented maximum load is 1000 kg.',second='no_sources')
  try:
   self.assertEqual(result['status'],'no_sources')
   self.assertEqual(len(checks),1);self.assertNotIn('STOP',checks[0])
   self.assertEqual(len(calls),4)
  finally:h.close()
 def test_original_lost_numeric_or_unsupported_same_sentence_cannot_be_restored(self):
  for text in ('Holding STOP is not isolation. Apply energy-isolation and trained personnel.',
   'The maximum load is 1000 kg and a 2500 kg coil is permitted. Holding STOP is not isolation.'):
   with self.subTest(text=text):
    result,calls,checks,h=self.replay(original=text,second='rewrite')
    try:
     self.assertEqual(result['status'],'no_sources',result)
     for call in calls:
      if call[0]=='reviewer':self.assertNotIn('original_retain_only',call[2]['json_schema']['name'])
    finally:h.close()
 def test_current_source_revocation_stops_without_second_provider(self):
  with self.assertRaises(EvidenceContractError):self.replay(revoke=True)
 def test_missing_second_call_budget_does_not_create_recovery_call(self):
  result,calls,checks,h=self.replay(no_budget=True)
  try:self.assertEqual(result['status'],'no_sources');self.assertEqual(len(calls),3);self.assertFalse(checks)
  finally:h.close()
 def test_private_snapshot_ownership_and_source_digest_are_not_forgeable_metadata(self):
  owner=SimpleNamespace();key=('one',)
  draft=final_contract.capture_reviewed_draft('Maximum1000 kg.',request_key=key,required_facets=('mass',),required_types=('numeric_value',))
  source=[{'citation_id':'one','text':'Maximum1000 kg.'}]
  snap=final_contract.GroundedNumericDraft(draft,owner,(('one',final_contract.answer_digest(source[0]['text'])),))
  owner._grounded_numeric_recovery=snap
  snap.check(owner=owner,request_key=key,candidates=source,evidence_text=lambda r:r['text'])
  for kw in ({'owner':SimpleNamespace(_grounded_numeric_recovery=snap)},{'request_key':('foreign',)},
   {'candidates':[{'citation_id':'one','text':'changed'}]}):
   args=dict(owner=owner,request_key=key,candidates=source,evidence_text=lambda r:r['text']);args.update(kw)
   with self.assertRaises(EvidenceContractError):snap.check(**args)
  owner._grounded_numeric_recovery=deepcopy(snap)
  with self.assertRaises(EvidenceContractError):snap.check(owner=owner,request_key=key,candidates=source,evidence_text=lambda r:r['text'])
 def test_scope_keeps_operations_collections_and_other_modes_out(self):
  d=SimpleNamespace(effective_mode='ask',request_kind='factual',information_task='document_explanation',
    required_answer_types=['numeric_value','safety_conditions','explanation'],facet_queries=[])
  r=SimpleNamespace(requested_mode='ask')
  self.assertTrue(final_contract.numeric_recovery_scope(r,d,enumeration=False))
  for name,value in (('effective_mode','root_cause'),('request_kind','procedure'),('information_task','procedure_segment'),
    ('required_answer_types',['numeric_value','checklist'])):
   old=getattr(d,name);setattr(d,name,value)
   self.assertFalse(final_contract.numeric_recovery_scope(r,d,enumeration=False));setattr(d,name,old)
  self.assertFalse(final_contract.numeric_recovery_scope(r,d,enumeration=True))
if __name__=='__main__':
 with patch.object(socket.socket,'connect',deny),patch.object(socket,'create_connection',deny),patch.object(sqlite3,'connect',deny):
  unittest.main(verbosity=2)
