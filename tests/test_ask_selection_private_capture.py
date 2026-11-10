"""Encrypted-only typed router/preP4 observations; controlled real Core, no remote I/O."""
from pathlib import Path
import sys,json,socket,sqlite3,unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import test_ask_ordinary_private_capture as old
from machinemind.ask import request_completion as c,core_intake as intake,private_review_diagnostics as private
from test_private_review_diagnostics import SECRET

def deny(*a,**k):raise AssertionError('NETWORK_SQL_ADC_PROVIDER_FORBIDDEN')
class SelectionCapture(unittest.TestCase):
 @classmethod
 def setUpClass(cls):old.OrdinaryCaptureRealCoreTests.setUpClass()
 @classmethod
 def tearDownClass(cls):old.OrdinaryCaptureRealCoreTests.tearDownClass()
 def replay(self,debug=True):return old.OrdinaryCaptureRealCoreTests().replay(debug)
 def decode(self,recorded):
  self.assertEqual(len(recorded),1)
  return private.open_envelope(recorded[0][0],secret=SECRET,expected_binding=recorded[0][1])
 def test_real_core_typed_selection_and_preP4_inventory_stay_inside_cipher(self):
  result,calls,recorded=self.replay();body=self.decode(recorded)
  selected=body['expanded_candidate']['source_selection'];self.assertEqual(set(selected),{'router','pre_p4'})
  self.assertIsNone(selected['router']['selected_parent_id'])
  self.assertTrue(selected['router']['owner_catalog_is_not_claimed_actual_emitted_enum'])
  self.assertLessEqual(len(selected['router']['owner_available_catalog_ids']),12)
  p4=selected['pre_p4'];self.assertGreater(p4['candidate_count'],0)
  self.assertEqual(sum(p4['candidate_type_counts'].values()),p4['candidate_count'])
  self.assertEqual(len(p4['candidate_rows']),min(256,p4['candidate_count']))
  self.assertEqual(p4['candidate_rows_truncated'],p4['candidate_count']>256)
  self.assertEqual(p4['snapshot_stage'],'after_family_restore_before_P4_prepare')
  self.assertTrue(all(set(row)=={'citation_id','source_type'}for row in p4['candidate_rows']))
  raw=json.dumps(result);self.assertNotIn('source_selection',raw);self.assertNotIn('candidate_type_counts',raw)
  self.assertEqual([row[0]for row in calls],['router','writer','reviewer'])
 def test_foreign_company_binding_never_invokes_typed_snapshot_or_changes_verdict(self):
  original=c.record_private_source_selection;observed=[]
  def record(*,intake,stage,router_result=None):
   owner=c.current();company=owner.payload.company_id;owner.payload.company_id='FOREIGN_PRIVATE_COMPANY'
   try:return original(intake=intake,stage=stage,router_result=router_result)
   finally:owner.payload.company_id=company
  with patch.object(c,'record_private_source_selection',side_effect=record),patch.object(intake._CoreIntake,'_private_source_selection_snapshot',side_effect=lambda **kw:observed.append(kw)):
   result,calls,recorded=self.replay()
  self.assertEqual(observed,[]);self.assertNotIn('source_selection',self.decode(recorded)['expanded_candidate'])
  self.assertEqual(result['status'],'no_sources');self.assertEqual(len(calls),3)
 def test_oversize_or_forbidden_diagnostic_context_is_omitted_without_fallback(self):
  real=intake._CoreIntake._private_source_selection_snapshot
  for malformed in ({'oversized':'x'*65537},{'access_token':'SECRET_SENTINEL_NOT_ALLOWED'}):
   def snapshot(obj,**kw):real(obj,**kw);return deepcopy(malformed)
   with self.subTest(kind=next(iter(malformed))),patch.object(intake._CoreIntake,'_private_source_selection_snapshot',snapshot):
    result,calls,recorded=self.replay()
    decoded=self.decode(recorded);self.assertNotIn('source_selection',decoded['expanded_candidate'])
    self.assertNotIn('SECRET_SENTINEL_NOT_ALLOWED',json.dumps(result));self.assertEqual(len(calls),3)
 def test_debug_false_and_fake_intake_cannot_create_selection_context(self):
  for debug in(False,'true'):
   result,calls,recorded=self.replay(debug)
   self.assertEqual(recorded,[]);self.assertNotIn('source_selection',json.dumps(result));self.assertEqual(len(calls),3)
  case=old.OrdinaryCaptureOwnerTests();case.setUp()
  try:
   owner,request=case.owner();called=[]
   fake=SimpleNamespace(request=request,_private_source_selection_snapshot=lambda **kw:called.append(kw))
   c.record_private_source_selection(intake=fake,stage='router',router_result={})
   self.assertFalse(called);self.assertIsNone(owner._private_source_selection)
   owner.close();self.assertIsNone(owner._private_source_selection)
  finally:case.doCleanups()
if __name__=='__main__':
 with patch.object(socket.socket,'connect',deny),patch.object(socket,'create_connection',deny),patch.object(sqlite3,'connect',deny):unittest.main(verbosity=2)
