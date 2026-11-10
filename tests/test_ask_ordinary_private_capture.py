"""Ordinary ASK diagnostic hooks: actual owner/call boundaries, no model spending.

The EN route is an actual recorded profile; source/LLM replies in the replay are
controlled fixtures. It is a counterexample to inferring packet absence from
compact no_sources counts, not recovery of the missing production prompt.
"""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,io,json,os,socket,sqlite3,sys,time,unittest,zipfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from machinemind.ask import request_completion as c,private_review_diagnostics as p
import test_private_review_capture_owned as capture
import test_ask_conditions_family_admission as owned
import test_ask_specific_assertions_scope as manual_fixture
from test_private_review_diagnostics import SECRET
W=next(x for x in (ROOT.parent,ROOT.parents[1],ROOT.parents[2])if(x/'phase6-r14-webapp-validation-20261110').is_dir())
QUALITY=W/'phase6-r14-webapp-validation-20261110/collector/UI-A-ASK-03-quality-response.json'
QUALITY_SHA='307f3c00b68a36761d9ffdeb3fdeb0f49c01486f1bc623fbb984cabcfccf3459'

def deny(*a,**k):raise AssertionError('NETWORK_SQL_PROVIDER_FORBIDDEN')
def input_frame(sources='Capacity 1000 kg.',schema=None):
 return {'sources_block':sources,'messages':[{'role':'user','content':sources}],
         'json_schema_json':json.dumps(schema or {'type':'object','properties':{'answer':{'type':'string'}}},sort_keys=True),
         'models':['gpt-5.6-terra'],'effort':'none'}

class OrdinaryCaptureOwnerTests(unittest.TestCase):
 def setUp(self):
  self.stack=ExitStack();self.addCleanup(self.stack.close)
  self.stack.enter_context(patch.dict(os.environ,{'COMMIT_SHA':'a'*40},clear=True))
  self.stack.enter_context(patch.object(socket,'create_connection',side_effect=deny))
  self.stack.enter_context(patch.object(sqlite3,'connect',side_effect=deny))
 def owner(self,debug=True):
  request=capture.request();request.debug=debug;request.response_language='en'
  owner=c.RequestCompletion(runtime=capture.runtime(),payload=request,started=time.monotonic());self.addCleanup(owner.close)
  return owner,request
 def record(self,owner,request,*,writer='writer packet has8 records',reviewer='review packet has11 records; capacity1000 kg',foreign=False):
  token=c.begin_private_ordinary_writer(query=request.query,company_id=request.company_id,provider_input=input_frame(writer))
  c.finish_private_ordinary_writer(token,actual_reply={'answer_status':'no_sources','grounded_points':[]},model_used='gpt-5.6-terra')
  if foreign:request=deepcopy(request);request.company_id='foreign'
  review=c.begin_private_ordinary_review(request=request,answer='',sources_block=reviewer,provider_input=input_frame(reviewer))
  c.finish_private_ordinary_review(review,actual_reply={'outcome':'no_sources','covered_facets':['one','two'],'missing_facets':['three']},model_used='gpt-5.6-terra')
 def finish(self,owner,allowed=True):
  value={'ok':True,'status':'no_sources','answer':'unchanged','meta':{}}
  if allowed:value=owner.publish(value,capture.publish_allowed)
  return owner.finish(value)
 def test_both_exact_packets_and_no_sources_wires_are_encrypted_and_separately_bound(self):
  owner,request=self.owner();self.record(owner,request)
  final=self.finish(owner);envelope=final['meta']['private_factual_review_diagnostics']
  decoded=p.open_envelope(envelope,secret=SECRET,expected_binding=owner._private_review_binding())
  self.assertEqual(decoded['first_wire_reply'],{'answer_status':'no_sources','grounded_points':[]})
  self.assertEqual(decoded['expanded_candidate']['writer_input']['sources_block'],'writer packet has8 records')
  self.assertEqual(decoded['sources_block'],'review packet has11 records; capacity1000 kg')
  self.assertNotIn('capacity1000',json.dumps(final));self.assertIsNone(owner._private_ordinary_writer)
 def test_debug_false_foreign_and_unpublished_never_emit_or_change_response(self):
  for debug,foreign,allowed in [(False,False,True),(True,True,True),(True,False,False)]:
   owner,request=self.owner(debug);self.record(owner,request,foreign=foreign);final=self.finish(owner,allowed)
   self.assertNotIn('private_factual_review_diagnostics',final['meta']);self.assertEqual(final['answer'],'unchanged');owner.close()
 def test_input_snapshot_cannot_be_mutated_after_boundary_or_reused_by_foreign_owner(self):
  owner,request=self.owner();source=input_frame('original writer packet')
  token=c.begin_private_ordinary_writer(query=request.query,company_id=request.company_id,provider_input=source)
  source['sources_block']='mutated later';source['messages'][0]['content']='mutated later'
  c.finish_private_ordinary_writer(token,actual_reply={'answer_status':'no_sources','grounded_points':[]},model_used='model')
  self.assertEqual(owner._private_ordinary_writer['input']['sources_block'],'original writer packet')
  owner.close();other,req=self.owner()
  c.finish_private_ordinary_writer(token,actual_reply={'answer_status':'answered'},model_used='model')
  self.assertIsNone(other._private_ordinary_writer)
 def test_duplicate_writer_and_close_keep_diagnostic_fail_closed(self):
  owner,request=self.owner();kwargs=dict(query=request.query,company_id=request.company_id,provider_input=input_frame())
  c.begin_private_ordinary_writer(**kwargs);c.begin_private_ordinary_writer(**kwargs)
  self.assertTrue(owner._private_review_invalidated);self.assertIsNone(owner._private_ordinary_writer)
  self.assertNotIn('private_factual_review_diagnostics',self.finish(owner)['meta']);owner.close()
  self.assertIsNone(owner._private_ordinary_writer)
 def test_schema_string_preserves_real_schema_metadata_without_forbiddenkey_bypass(self):
  owner,request=self.owner();schema={'type':'object','properties':{'source_table_token':{'type':'string'},'answer':{'type':'string'}}}
  frame=input_frame('safe '+SECRET,schema);token=c.begin_private_ordinary_writer(query=request.query,company_id=request.company_id,provider_input=frame)
  c.finish_private_ordinary_writer(token,actual_reply={'answer_status':'no_sources','grounded_points':[]},model_used='model')
  rev=c.begin_private_ordinary_review(request=request,answer='',sources_block='safe '+SECRET,provider_input=input_frame('safe'))
  c.finish_private_ordinary_review(rev,actual_reply={'outcome':'no_sources'},model_used='model')
  final=self.finish(owner);serialized=json.dumps(final)
  self.assertNotIn(SECRET,serialized)
  decoded=p.open_envelope(final['meta']['private_factual_review_diagnostics'],secret=SECRET,expected_binding=owner._private_review_binding())
  self.assertNotIn(SECRET,json.dumps(decoded));self.assertIn('[REDACTED]',json.dumps(decoded))
  # Existing forbidden dictionary handling is not relaxed to carry schema.
  with self.assertRaises(p.DiagnosticError):p._plain({'access_token':'forbidden'})

class OrdinaryCaptureRealCoreTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
 @classmethod
 def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()
 def replay(self,debug):
  assert hashlib.sha256(QUALITY.read_bytes()).hexdigest()==QUALITY_SHA
  data=json.loads(QUALITY.read_text(encoding='utf-8-sig'));raw=deepcopy(data['calls'][0]['quality_meta']['assistant_core_v2'])
  query='Can you confirm that I can load a 2500 kg coil onto the decoiler and that holding STOP is equivalent to isolating all energy sources? If either premise is wrong, correct it using the documentation.'
  h=manual_fixture.WholeManualFixture(self,choice=None,raw=raw,query=query,manual=manual_fixture.manual_text())
  recorded=[];original_init=c.RequestCompletion.__init__;original_close=c.RequestCompletion.close
  def init(owner,*,runtime,payload,started):
   payload=SimpleNamespace(company_id=h.request.company_id,machine_id=h.request.machine_id,query=query,debug=debug)
   return original_init(owner,runtime=replace(runtime,AI_INTERNAL_SECRET=SECRET),payload=payload,started=started)
  def close(owner):
   if owner._private_review_envelope is not None:recorded.append((deepcopy(owner._private_review_envelope),owner._private_review_binding()))
   return original_close(owner)
  def writer(messages,**kw):
   h.calls.append(('writer',deepcopy(messages),deepcopy(kw)));h.writer_packets.append(messages[1]['content'])
   return {'answer_status':'no_sources','grounded_points':[]},kw['models'][0]
  def reviewer(messages,**kw):
   h.calls.append(('reviewer',deepcopy(messages),deepcopy(kw)));h.review_packets.append(messages[1]['content'])
   return {'outcome':'no_sources','answer':'','reason':'Controlled missing explanation counterexample; not an actual recovered production verdict.',
     'covered_facets':list(raw['required_facets'][:2]),'missing_facets':list(raw['required_facets'][2:]),
     'covered_answer_types':['numeric_value','safety_conditions'],'missing_answer_types':['explanation'],
     'enumeration_requested':False,'expected_list_items':[],'covered_list_items':[],'missing_list_items':[],'citation_ids':[]},kw['models'][0]
  h.owner.generation=replace(h.owner.generation,_v13_json_models=writer);h.reviewer=reviewer
  try:
   with patch.dict(os.environ,{'COMMIT_SHA':'b3916685230b253fdb1f90f5453faa1de9791e9f'}),patch.object(c.RequestCompletion,'__init__',init),patch.object(c.RequestCompletion,'close',close):
    result=h.run_core()
   return result,h.calls,recorded
  finally:h.close()
 def test_actual_EN_contract_through_real_Core_captures_real_provider_schemas_and_separate_packets(self):
  result,calls,recorded=self.replay(True)
  self.assertEqual(result['status'],'no_sources');self.assertEqual([x[0]for x in calls],['router','writer','reviewer'])
  self.assertEqual(len(recorded),1,'The ordinary branch must emit one owned encrypted capture, not an inventory-only capture')
  body=p.open_envelope(recorded[0][0],secret=SECRET,expected_binding=recorded[0][1])
  writer=next(x for x in calls if x[0]=='writer');reviewer=next(x for x in calls if x[0]=='reviewer')
  self.assertEqual(body['expanded_candidate']['writer_input']['messages'],writer[1])
  self.assertEqual(json.loads(body['expanded_candidate']['writer_input']['json_schema_json']),writer[2]['json_schema'])
  self.assertEqual(body['review_frame']['messages'],reviewer[1])
  self.assertEqual(json.loads(body['review_frame']['json_schema_json']),reviewer[2]['json_schema'])
  self.assertIn('1000',body['expanded_candidate']['writer_input']['sources_block'])
  self.assertEqual(len(body['actual_review_reply']['covered_facets']),2);self.assertEqual(len(body['actual_review_reply']['missing_facets']),1)
 def test_debug_false_keeps_same_provider_protocol_and_response_without_capture(self):
  plain,plaincalls,nocapture=self.replay(False);debug,debugcalls,capture=self.replay(True)
  self.assertEqual(plain,debug)
  def fixed_protocol(calls):
   normalized=deepcopy(calls)
   for _,_,kw in normalized:
    timeout=kw.pop('timeout',None)
    if timeout is not None:self.assertGreater(timeout,0);self.assertLessEqual(timeout,100)
   return normalized
  # Remaining wall time is an observation, not a fixed provider-policy field.
  # The separate AST control proves original timeout/cap expressions unchanged.
  self.assertEqual(fixed_protocol(plaincalls),fixed_protocol(debugcalls));self.assertEqual(nocapture,[]);self.assertEqual(len(capture),1)
 def test_original_provider_timeout_and_output_cap_expressions_are_unchanged(self):
  archive=W/'phase6-ask-english-forensics-20261110/baseline-b391.zip'
  with zipfile.ZipFile(archive)as z:
   for rel in ('machinemind/ask/generation.py','machinemind/ask/execution.py'):
    def controls(text):
     return sorted((ast.dump(k.value,include_attributes=False)for n in ast.walk(ast.parse(text))if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_v13_json_models'for k in n.keywords if k.arg in ('timeout','max_output_tokens','effort','reasoning_mode','company_id','purpose','require_full_output')))
    self.assertEqual(controls(z.read(rel).decode('utf-8')),controls((ROOT/rel).read_text(encoding='utf-8')))

if __name__=='__main__':unittest.main(verbosity=2)
