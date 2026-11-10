"""Owned canonical context changes draft role only; provider replies remain fixtures.

These controls verify source/authority/output/reviewer contracts, not live quality
or a latency improvement. Actual R7 replay's unrecorded private parent is an
explicit valid-choice counterfactual, never historical telemetry.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,unittest,time
from types import SimpleNamespace
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import generation,request_completion
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_conditions_family_admission as owned
import test_ask_procedure_context_recovery as context
import test_ask_router_family_context as route

class OwnedSourceFastCompositionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def real_policy(self,h):
  h.owner.generation=replace(h.owner.generation,_v13_choose_ask_model=self.m._v13_choose_ask_model)
  h.prepared['metrics']={'confidence':'low'}
 def assert_fast_draft_unchanged_review(self,h):
  writer=next(c for c in h.calls if c[0]=='writer');reviewer=next(c for c in h.calls if c[0]=='reviewer')
  baseline=self.m._v13_choose_ask_model(h.request.query,h.prepared,narrow_scope=h.request.narrow_scope)
  self.assertEqual(baseline[0],self.m.V13_HEAVY_MODEL)
  self.assertEqual(writer[2]['models'],[self.m.V13_FAST_MODEL]);self.assertEqual(writer[2]['effort'],'none')
  self.assertEqual(writer[2]['reasoning_mode'],'')
  self.assertEqual(writer[2]['max_output_tokens'],h.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS)
  self.assertEqual(writer[2]['timeout'],h.owner.generation.V13_HEAVY_TIMEOUT_SECONDS)
  self.assertEqual(writer[2]['purpose'],'ask_final_synthesis')
  self.assertEqual(reviewer[2]['models'][0],self.m.V13_FAST_MODEL);self.assertEqual(reviewer[2]['effort'],'low')
  self.assertEqual(reviewer[2]['purpose'],'assistant_core_answer_contract_verifier')
  self.assertEqual(h.owner._generated_response['meta']['generation_evidence']['source_sha256'],
      h.owner._response_flow._review_packet.diagnostic(reused=False)['source_sha256'])
 def completion(self,h):
  return request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
   payload=SimpleNamespace(company_id=h.request.company_id),started=time.monotonic())
 def test_current_owned_family_uses_fast_none_with_original_heavy_packet_caps_and_judge(self):
  h=context.ContextFixture(self,query='Before inserting the cartridge, which documented mass limit, lifting checks, electrical checks and safety conditions apply, and which low-speed test belongs later?')
  completion=self.completion(h)
  try:
   self.real_policy(h);h.prepare();self.assertTrue(h.runtime.preserve_procedure_context)
   result=h.run();self.assertEqual(result['status'],'answered');self.assert_fast_draft_unchanged_review(h)
   for key,_,_,body in h.rows:
    self.assertIn(body,h.writer_packets[0]);self.assertIn(body,h.review_packets[0])
   self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
  finally:h.close();completion.close()
 def test_independent_judge_can_still_reject_complete_family_fast_draft(self):
  h=context.ContextFixture(self,reject_review=True,query='Before cartridge insertion provide the maximum mass and every required initial safety check, keeping the later low-speed test separate.')
  completion=self.completion(h)
  try:
   self.real_policy(h);h.prepare();result=h.run();self.assertEqual(result['status'],'no_sources')
   self.assertEqual([c[0]for c in h.calls],['writer','reviewer'])
   self.assertEqual(h.calls[0][2]['models'],[self.m.V13_FAST_MODEL]);self.assertEqual(h.calls[1][2]['effort'],'low')
  finally:h.close();completion.close()
 def test_revocation_during_fast_draft_still_aborts_before_review(self):
  h=context.ContextFixture(self,query='Before inserting the cartridge, which maximum permitted mass and documented lifting, connection and safety checks must be respected?')
  try:
   self.real_policy(h);h.prepare();provider=h.owner.generation._v13_json_models
   def revoke(messages,**kw):
    reply=provider(messages,**kw);h.allowance=frozenset();return reply
   h.owner.generation=replace(h.owner.generation,_v13_json_models=revoke)
   with self.assertRaises(EvidenceContractError):h.run()
   self.assertEqual([c[0]for c in h.calls],['writer'])
  finally:h.close()
 def test_unowned_partial_contract_flags_do_not_change_legacy_model_effort_or_bounds(self):
  h=context.ContextFixture(self)
  try:
   self.real_policy(h);row=deepcopy(h.prepared['candidates'][0]);q='State the documented numerical requirement and explain every relevant condition, including which interpretation is unsupported by the available manual sources.'
   retrieval={'candidates':[row],'citations':[row],'metrics':{'confidence':'low'},
    'assistant_core_contract':{'fail_closed':True,'preserve_procedure_context':True,'source_backed_composition':True},
    'plan':{'preserve_procedure_context':True}}
   calls=[];contexts=[]
   def provider(messages,**kw):calls.append(kw);return {'answer_status':'no_sources','grounded_points':[]},kw['models'][0]
   def sources(rows,**kw):contexts.append(kw);return 'SOURCE: exact partial fixture'
   runtime=replace(h.owner.generation,_v13_json_models=provider,_v13_sources_block=sources,
     _finalize_ask_response_for_ui=lambda r,**kw:r)
   self.assertFalse(runtime.preserve_procedure_context);self.assertFalse(runtime.preserve_procedure_points)
   baseline=self.m._v13_choose_ask_model(q,retrieval,narrow_scope=False)
   result=generation.generate_ask_response(q=q,company_id=h.request.company_id,response_language='en',top_k=6,
     retrieval=retrieval,narrow_scope=False,debug=False,runtime=runtime)
   self.assertEqual(result['status'],'no_sources');self.assertEqual(len(calls),1)
   self.assertEqual(calls[0]['models'],[baseline[0],self.m.V13_FAST_MODEL]);self.assertEqual(calls[0]['effort'],baseline[1])
   self.assertEqual(calls[0]['reasoning_mode'],baseline[2]);self.assertEqual(calls[0]['max_output_tokens'],runtime.V13_HEAVY_MAX_OUTPUT_TOKENS)
   self.assertEqual(contexts[0]['max_context_chars'],runtime.V13_HEAVY_CONTEXT_CHARS)
  finally:h.close()
 def test_actual_pinned_r7_type_family_manual_and_rejection_matrix_uses_new_draft_role_only(self):
  # The existing replay proves11currentfamilyrecords incl suspended-load note,
  # optional manual-only facts, exclusion of another family, and reject path.
  work=Path(__file__).resolve().parents[2]
  required=work/'phase6-ask-correction-r7-20261109/live-client/receipts/R7-A-ASK-02.json'
  self.assertTrue(required.exists(),'Pinned private corpus must be present; this qualification may not skip')
  fixtures=[];original=route.RouterFixture.__init__
  def init(fixture,*a,**kw):
   original(fixture,*a,**kw);self.real_policy(fixture);fixtures.append(fixture)
  with patch.object(route.RouterFixture,'__init__',init):
   context.AskProcedureContextRecoveryTests.test_actual_r7_required_parent_type_keeps_complete_family_and_manual_support(self)
  self.assertEqual(len(fixtures),3)
  for h in fixtures:
   writer=next(c for c in h.calls if c[0]=='writer');reviewer=next(c for c in h.calls if c[0]=='reviewer')
   self.assertEqual(writer[2]['models'],[self.m.V13_FAST_MODEL]);self.assertEqual(writer[2]['effort'],'none')
   self.assertEqual(writer[2]['max_output_tokens'],h.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS)
   self.assertEqual(reviewer[2]['effort'],'low');self.assertIn('Non sostare sotto carichi sospesi.',h.writer_packets[0])
   self.assertIn('Non sostare sotto carichi sospesi.',h.review_packets[0])

 def test_real_owned_finite_numeric_manual_draft_uses_fast_without_claiming_family_completeness(self):
  raw={'request_kind':'factual','effective_mode':'ask','confidence':.98,'requested_mode_fit':True,
   'evidence_state':'supported','evidence_policy':'machine_sources_required','information_task':'numeric_specification',
   'procedure_scope':'other','required_answer_types':['numeric_value','safety_conditions','explanation'],
   'preferred_source_types':['document'],'source_type_policy':'require','required_facets':['maximum mass and proposed mass','STOP versus energy isolation','why the proposed mass is outside the limit'],
   'facet_queries':[{'facet':'maximum mass and proposed mass','answer_type':'numeric_value','must_cover':True,
      'dense_queries':['maximum cartridge mass'],'lexical_queries':['maximum cartridge mass'],'exact_terms':['40 kg'], 'preferred_source_types':['document']},
     {'facet':'STOP versus energy isolation','answer_type':'safety_conditions','must_cover':True,
      'dense_queries':['STOP energy isolation'],'lexical_queries':['STOP energy isolation'],'exact_terms':['STOP'],'preferred_source_types':['document']},
     {'facet':'why the proposed mass is outside the limit','answer_type':'explanation','must_cover':True,
      'dense_queries':['proposed mass exceeds limit'],'lexical_queries':['proposed mass limit'],'exact_terms':[],'preferred_source_types':['document']}]}
  text='Maximum permitted cartridge mass: 40 kg. A proposed 50 kg cartridge exceeds that limit. STOP holds the cycle but does not isolate energy sources. Full energy isolation uses the approved company procedure.'
  h=route.RouterFixture(self,choice=None,raw=raw,query='The proposed cartridge mass is 50 kg: is that within the documented maximum mass, and does holding STOP establish full energy isolation? Explain the documented distinction.',manual=text)
  try:
   self.real_policy(h)
   def provider(messages,**kw):
    h.calls.append(('writer',deepcopy(messages),deepcopy(kw)));h.writer_packets.append(messages[1]['content'])
    return {'answer_status':'answered','grounded_points':[
     {'text':'No. The maximum permitted cartridge mass is 40 kg, so 50 kg exceeds that limit.','citation_ids':['manual-only:p1']},
     {'text':'STOP holds the cycle; it does not isolate energy sources. Use the approved company isolation procedure.','citation_ids':['manual-only:p1']}]},kw['models'][0]
   h.owner.generation=replace(h.owner.generation,_v13_json_models=provider)
   h.reviewer=lambda messages,**kw:owned.OwnedFamilyFixture.reviewer(h,messages,**kw)
   result=h.run_core();self.assertEqual(result['status'],'answered',result)
   self.assertFalse(h.owner._procedure_family_context);self.assertIsNone(h.owner._selected_catalog_parent)
   self.assertEqual([c[0]for c in h.calls],['router','writer','reviewer'])
   writer=next(c for c in h.calls if c[0]=='writer');judge=next(c for c in h.calls if c[0]=='reviewer')
   self.assertEqual(writer[2]['models'],[self.m.V13_FAST_MODEL]);self.assertEqual(writer[2]['effort'],'none')
   self.assertEqual(writer[2]['max_output_tokens'],h.owner.generation.V13_HEAVY_MAX_OUTPUT_TOKENS)
   self.assertEqual(writer[2]['timeout'],h.owner.generation.V13_HEAVY_TIMEOUT_SECONDS)
   self.assertEqual(judge[2]['effort'],'low');self.assertEqual(judge[2]['models'][0],self.m.V13_FAST_MODEL)
   self.assertIn(text,h.writer_packets[0]);self.assertIn(text,h.review_packets[0])
   self.assertIn('40 kg',result['answer']);self.assertIn('does not isolate',result['answer'])
  finally:h.close()
 def test_mandatory_facet_union_nonfactual_enum_catalog_or_missing_proof_keep_original_writer(self):
  h=context.ContextFixture(self)
  try:
   self.real_policy(h);q='State the numerical capacity and explain whether the named stop control provides complete isolation according to the supplied documentation, with all relevant source qualifications.'
   base={'candidates':[deepcopy(h.prepared['candidates'][0])],'metrics':{'confidence':'low'},
    'assistant_core_decision':{'request_kind':'factual','effective_mode':'ask','enumeration_requested':False},
    'assistant_core_contract':{'fail_closed':True,'information_task':'numeric_specification',
      'required_answer_types':['numeric_value','safety_conditions'],
      'facet_queries':[{'facet':'numeric limit','must_cover':True,'answer_type':'numeric_value'}]}}
   variants=[]
   for answer_type in ('checklist','ordered_actions','state_sequence','interface_locations'):
    v=deepcopy(base);v['assistant_core_contract']['facet_queries'].append({'facet':'mandatory secondary obligation','must_cover':True,'answer_type':answer_type});variants.append(v)
   for key,value in (('request_kind','procedure'),('effective_mode','root_cause'),('enumeration_requested',True)):
    v=deepcopy(base);v['assistant_core_decision'][key]=value;variants.append(v)
   v=deepcopy(base);v['assistant_core_decision'].pop('enumeration_requested');variants.append(v)
   for key,value in (('overview_catalog_requested',True),('machine_catalog_digest','complete catalog'),('information_task','procedure_segment')):
    v=deepcopy(base);v['assistant_core_contract'][key]=value;variants.append(v)
   for v in variants:
    with self.subTest(contract=v['assistant_core_contract'],decision=v['assistant_core_decision']):
     calls=[]
     def provider(messages,**kw):calls.append(kw);return {'answer_status':'no_sources','grounded_points':[]},kw['models'][0]
     rt=replace(h.owner.generation,preserve_procedure_points=True,_v13_json_models=provider,
      _v13_sources_block=lambda *a,**kw:'bounded admitted fixture',_finalize_ask_response_for_ui=lambda r,**kw:r)
     generation.generate_ask_response(q=q,company_id=h.request.company_id,response_language='en',top_k=6,
      retrieval=v,narrow_scope=False,debug=False,runtime=rt)
     self.assertEqual(len(calls),1);self.assertEqual(calls[0]['models'][0],self.m.V13_HEAVY_MODEL)
     self.assertEqual(calls[0]['effort'],self.m._v13_choose_ask_model(q,v,narrow_scope=False)[1])
  finally:h.close()
 def test_unverified_fast_model_override_keeps_original_complete_family_model_policy(self):
  h=context.ContextFixture(self,query='Before inserting the cartridge, state its maximum permitted mass and all documented lifting, connection and safety checks, distinguishing the later test from initial conditions.')
  try:
   self.real_policy(h);h.owner.generation=replace(h.owner.generation,V13_FAST_MODEL='custom-unverified-model')
   h.prepare();self.assertEqual(h.run()['status'],'answered')
   writer=next(c for c in h.calls if c[0]=='writer')
   self.assertEqual(writer[2]['models'],[self.m.V13_HEAVY_MODEL,'custom-unverified-model'])
   self.assertEqual(writer[2]['effort'],self.m._v13_choose_ask_model(h.request.query,h.prepared,narrow_scope=False)[1])
  finally:h.close()

if __name__=='__main__':unittest.main(verbosity=2)
