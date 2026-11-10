"""Offline categorical-label generation contract; semantic verdicts are controlled fixtures."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,hashlib,importlib.util,json,sys,unittest
HOME=Path(__file__).resolve().parents[1]
REPO=HOME if(HOME/'main.py').exists()else HOME.parent/'mm-ai-ingest'
sys.path.insert(0,str(REPO))
from machinemind.retrieval import smart_evidence,review_references as refs
SOURCE=HOME/'source/machinemind/retrieval/smart_review.py'
if not SOURCE.exists():SOURCE=REPO/'machinemind/retrieval/smart_review.py'
spec=importlib.util.spec_from_file_location('machinemind.retrieval._label_candidate_for_test',SOURCE)
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def fixture():
 scope={'company_id':'fixture-company','machine_id':'fixture-machine','ai_scope':'machine_all'}
 sources=[{'citation_id':'delivery-manual','bubble_document_id':'delivery-manual','source_type':'document',**scope,'page_from':1,'page_to':1,
  'chunk_full':'Message: Delivery rate. Type: Notice marks the caution range. Type: Trip marks a value outside the specified range. Record only the message already observed; if unavailable report Unknown. No new inspection or operation is requested.'}]
 packet=smart_evidence.review_packet(smart_evidence.build(sources,scope=scope))
 step={'status':'in_progress','final_ready':False,'operator_summary':'','final_result':{},
  'hypotheses':[{'id':'H1','rank':1,'label':'Possible delivery condition outside its applicable range','description':'A reported delivery message is compatible with a conditional out-of-range condition.','why':'The recorded type and value remain unknown.','status':'open','probability_pct':0,'probability_band':'unknown','checks':[],'evidence_ids':['delivery-manual']}],
  'question':{'question_id':'Q1','question_number':1,'question_type':'single_choice','question_text':'Which type was already recorded for the delivery message? Do not inspect now.','why_asked':'The literal type distinguishes its documented range; the type is unknown.','safety_level':'normal','safety_note':'','target_hypotheses':['H1'],
   'options':[{'id':'notice','label_en':'Notice','label_it':'Notice'},{'id':'trip','label_en':'Trip','label_it':'Trip'},{'id':'unknown','label_en':'Unknown','label_it':'Non so'}]}}
 return packet,step

def decisions(prepared,reject_question=False):
 frozen=prepared['references']['frozen'];rows=[]
 for p in frozen['proposals']:
  question=p['kind']=='diagnostic_question';reject=question and reject_question
  rows.append({'proposal_index':p['proposal_index'],'verdict':'reject'if reject else'accept','reason':'unsupported_check'if reject else'supported','blocking_checks':[2]if reject else[],'note':'Merged labels are not documented as equivalent.'if reject else'',
   'proofs':[]if reject else[{'source_index':0,'supports_cause':not question,'observation_units':[]if question else frozen['observed_ids'],'source_units':frozen['source_sets'][0],'target_units':frozen['target_sets'][0],'applicability':'same_target','support_type':'documented_check'if question else'bounded_inference','check_indices':[c['check_index']for c in p['checks']]}]})
 return {'decisions':rows}

class CategoricalLabelContract(unittest.TestCase):
 def setUp(self):
  self.guard=ExitStack();self.addCleanup(self.guard.close)
  for name in('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','sqlite3.connect','google.auth.default'):
   self.guard.enter_context(patch(name,side_effect=AssertionError('OFFLINE_NETWORK_ADC_SQL_DENIED')))
 def test_generation_guidance_preserves_literal_categories_and_existing_unknown_safety(self):
  text=review.generation_hypothesis_instruction()
  for phrase in('literal source labels','separate options','explicitly documents their equivalence','same target and conditions','neighboring message or subsystem','preserve Unknown','no new screen navigation','retrospection grants no safety exemption','Unknown is neither a positive nor a negative observation'):
   self.assertIn(phrase,text)
  self.assertNotIn('Water pressure',text);self.assertNotIn('Fault / alarm',text)
 def test_reviewer_and_every_other_contract_are_identical_to_preclarification(self):
  tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'));tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='generation_hypothesis_instruction')]
  self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),'e33701c27af6176ad2ecc6b7d1a0fe2029ac516d5dc952f944c71094b416599d')
 def test_actual_START_and_ANSWER_prompt_hooks_keep_one_call_and_existing_evidence_wire(self):
  tree=ast.parse((REPO/'main.py').read_text(encoding='utf-8-sig'));functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in{'_sd_llm_step_start','_sd_llm_step_answer'}]
  captures=[]
  def provider(messages,**kwargs):captures.append((deepcopy(messages),deepcopy(kwargs)));return {}
  ns={'json':json,'_smart_review':review,'_assistant_core_sd_json_models':provider,'SMART_DIAGNOSTIC_MODEL':'first','DIAGNOSTIC_EVIDENCE_MODEL':'second','OPENAI_CHAT_MODEL':'third','SMART_DIAGNOSTIC_LLM_TIMEOUT':70,'SMART_DIAGNOSTIC_MAX_QUESTIONS':6,
   '_sd_schema':lambda **k:{'name':'unchanged_schema','limits':k},'_sd_clamp_int':lambda v,d,a,z:int(v or d),
   '_sd_require_grounding_packet':lambda s:s['grounding_packet'],'_smart_evidence':SimpleNamespace(evidence_block=lambda p:'Exact original source block'),
   '_V13BudgetExceeded':RuntimeError,'HTTPException':RuntimeError}
  exec(compile(ast.Module(body=functions,type_ignores=[]),'actual_main_Smart_prompt_hooks','exec'),ns)
  ns['_sd_llm_step_start'](symptom_text='Recorded delivery message; type unknown.',language='en',max_questions=6,max_hypotheses=4,evidence_block='Exact original source block',evidence_ids=['s1'])
  ns['_sd_llm_step_answer'](state={'symptom_text':'Recorded delivery message; type unknown.','max_questions':6,'history':[],'evidence':[{'citation_id':'s1'}],'grounding_packet':{},'hypotheses':[],'current_question':{'question_number':1}},answer={'value':'unknown','api_value':'unknown','label':'Unknown'},language='it',max_hypotheses=4)
  self.assertEqual(len(captures),2)
  for (messages,kw),phase in zip(captures,('start','answer')):
   self.assertEqual([m['role']for m in messages],['system','user']);self.assertIn('literal source labels',messages[0]['content']);self.assertIn('Exact original source block',messages[1]['content'])
   self.assertEqual(kw['models'],['first','second','third']);self.assertEqual(kw['timeout'],70);self.assertEqual(kw['phase'],phase)
 def test_literal_distinct_labels_pass_only_with_controlled_source_bound_proofs(self):
  packet,step=fixture();prepared=review.prepare(step=step,packet=packet,symptom_text='A delivery message was recorded; its type and value are unknown.',history=[],language='en')
  out,seal,summary=review.resolve(prepared=prepared,parsed=decisions(prepared),probability_band=lambda p:'unknown',continuity_policy=review.continuity.POLICY_VERSION)
  self.assertEqual([o['id']for o in out['question']['options']],['notice','trip','unknown']);self.assertTrue(seal);self.assertEqual(summary['accepted_hypothesis_ids'],['H1']);self.assertTrue(summary['question_validated'])
 def test_merged_label_rejection_is_not_repaired_or_bypassed(self):
  packet,step=fixture();step['question']['options'][1]['label_en']='Trip / unrelated class'
  prepared=review.prepare(step=step,packet=packet,symptom_text='A delivery message was recorded; its type and value are unknown.',history=[],language='en')
  with self.assertRaisesRegex(review.SmartReviewError,'question_not_supported'):review.resolve(prepared=prepared,parsed=decisions(prepared,True),probability_band=lambda p:'unknown')
  out,seal,summary=review.resolve(prepared=prepared,parsed=decisions(prepared,True),probability_band=lambda p:'unknown',continuity_policy=review.continuity.POLICY_VERSION)
  self.assertFalse(summary['question_validated']);self.assertTrue(summary['rejected_question_discarded']);self.assertTrue(summary['application_question'])
  self.assertNotIn('unrelated class',json.dumps(out['question']));self.assertNotEqual(out['question']['question_text'],step['question']['question_text'])
 def test_retrospective_question_without_complete_proofs_still_fails(self):
  packet,step=fixture();prepared=review.prepare(step=step,packet=packet,symptom_text='A delivery message was recorded; its type and value are unknown.',history=[],language='en');parsed=decisions(prepared);parsed['decisions'][-1]['proofs']=[]
  with self.assertRaises(refs.ReferenceError):review.resolve(prepared=prepared,parsed=parsed,probability_band=lambda p:'unknown',continuity_policy=review.continuity.POLICY_VERSION)
if __name__=='__main__':unittest.main(verbosity=2)
