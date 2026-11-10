"""Current Root router context survives P4 caps without becoming factual proof."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.retrieval import root_clue_support as cs
import test_root_router_selected_context as selected_context
BODY=selected_context.BODY
from test_root_supported_context_capacity import candidate

class RootContextRetentionTests(unittest.TestCase):
 setUpClass=classmethod(selected_context.SelectedContextTests.setUpClass.__func__)
 tearDownClass=classmethod(selected_context.SelectedContextTests.tearDownClass.__func__)
 request=selected_context.SelectedContextTests.request
 route=selected_context.SelectedContextTests.route
 def predicate(self,decision):
  return lambda row: not bool(row.get('hard_excluded'))and cs.support_match(decision,self.m._v13_candidate_text(row),source_id=row['citation_id']).get('router_context_selected')is True
 def prepare(self,request,decision,rows):
  calls=[];original=self.m._v13_rescore_root_candidates
  def traced(q,values,**kw):
   predicate=kw.get('context_selected');calls.append({'callback':predicate,'ids':[r['citation_id']for r in values]})
   return original(q,values,**kw)
  with patch.object(self.m,'_assistant_core_root_diagnostic_evidence_assurance',side_effect=lambda req,ret,d:ret),patch.object(self.m,'_v13_rescore_root_candidates',traced):
   result=self.m._assistant_core_prepare_evidence(request,{'candidates':deepcopy(rows),'citations':deepcopy(rows)},decision)
  return result,calls
 def test_selected_context_survives_caps_without_inventing_observation_coverage(self):
  req,d,row=self.route();row.update(causal_strength_score=.1,semantic_score=.1,retrieval_score=.01,similarity=.01)
  others=[candidate('noise-'+str(i),source='manual-'+str(i),causal_strength_score=.5,semantic_score=.5,retrieval_score=1,similarity=.8)for i in range(30)]
  rows=others+[row];before=deepcopy(rows);current=self.predicate(d)
  old=self.m._prioritize_root_cause_coverage(self.m._dedup_root_cause_candidates_semantic(rows,24),20)
  new=self.m._prioritize_root_cause_coverage(self.m._dedup_root_cause_candidates_semantic(rows,24,context_selected=current),20,context_selected=current)
  self.assertNotIn(row['citation_id'],[r['citation_id']for r in old]);self.assertIn(row['citation_id'],[r['citation_id']for r in new]);self.assertEqual(len(new),20)
  self.assertEqual(rows,before);self.assertEqual(cs.support_match(d,BODY)['matched_support_part_ids'],())
 def test_identical_alias_keeps_exact_selected_original_without_duplicate_source(self):
  req,d,row=self.route();alias={**row,'citation_id':'higher-scored-alias','retrieval_score':99,'similarity':.99,'causal_strength_score':.99,'semantic_score':.99}
  row.update(retrieval_score=.01,similarity=.01,causal_strength_score=.1,semantic_score=.1)
  current=self.predicate(d)
  for fn in(self.m._dedup_citations_by_snippet,self.m._dedup_root_cause_candidates_semantic):
   self.assertEqual(fn([alias,row],1)[0]['citation_id'],'higher-scored-alias')
   kept=fn([alias,row],1,context_selected=current);self.assertEqual(len(kept),1);self.assertEqual(kept[0]['citation_id'],row['citation_id']);self.assertEqual(kept[0]['chunk_full'],BODY)
  prepared,calls=self.prepare(req,d,[alias,row]);self.assertTrue(calls and callable(calls[0]['callback']));self.assertIn(row['citation_id'],calls[0]['ids']);self.assertNotIn('higher-scored-alias',calls[0]['ids'])
 def test_P4_uses_only_fresh_current_router_CID_and_body(self):
  req,d,row=self.route();out,calls=self.prepare(req,d,[row]);self.assertTrue(calls and callable(calls[0]['callback']))
  for changed in({**row,'chunk_full':'Packaging dimensions only.'},{**row,'citation_id':'unseen-id'},{**row,'company_id':'foreign-company'},{**row,'machine_id':'foreign-machine'}):
   with self.subTest(changed=changed['citation_id']):
    out,calls=self.prepare(req,d,[changed]);self.assertTrue(calls);self.assertIsNone(calls[0]['callback'])
 def test_unselected_forged_flags_and_modified_carrier_do_not_enable_P4(self):
  req,d,row=self.route(chosen=False);row.update(router_context_selected=True,core_context_eligible=True,matched_support_part_ids=['0:30:observation'])
  out,calls=self.prepare(req,d,[row]);self.assertIsNone(calls[0]['callback'])
  req,d,row=self.route();tampered=replace(d.root_clue_support,router_source_hashes=((row['citation_id'],'0'*64),))
  for bad in(replace(d,root_clue_support=None),replace(d,root_clue_support=tampered),replace(d,degraded=True)):
   out,calls=self.prepare(req,bad,[row]);self.assertIsNone(calls[0]['callback'])
 def test_changed_request_scope_and_other_modes_keep_legacy_selection(self):
  req,d,row=self.route()
  for request in(self.request(company_id='other-company'),self.request(machine_id='other-machine'),self.request(query=req.query+' Another issue.')):
   out,calls=self.prepare(request,d,[row]);self.assertTrue(calls);self.assertIsNone(calls[0]['callback'])
  for mode in('ask','smart_diagnostic'):
   out,calls=self.prepare(self.request(requested_mode=mode),replace(d,effective_mode=mode),[row])
   self.assertTrue(all(c['callback']is None for c in calls))
 def test_hard_excluded_current_row_receives_no_context_priority(self):
  req,d,row=self.route();row['hard_excluded']=True
  out,calls=self.prepare(req,d,[row]);self.assertIsNone(calls[0]['callback'])
 def test_overselected_context_and_zero_limit_cannot_expand_existing_caps(self):
  rows=[candidate('owned-'+str(i),source='source-'+str(i))for i in range(35)]
  for limit in(0,1,8,20,24):
   for fn in(self.m._dedup_citations_by_snippet,self.m._dedup_root_cause_candidates_semantic,self.m._prioritize_root_cause_coverage):
    out=fn(rows,limit,context_selected=lambda row:True);self.assertLessEqual(len(out),limit);self.assertEqual(len({r['citation_id']for r in out}),len(out))
 def test_absent_or_false_callback_is_exact_legacy_order_and_content(self):
  rows=[candidate('owned-'+str(i),source='source-'+str(i%3),causal_strength_score=(i%7)/10,retrieval_score=(i%5)/10)for i in range(35)]
  for limit in(1,8,20,24):
   for fn in(self.m._dedup_citations_by_snippet,self.m._dedup_root_cause_candidates_semantic,self.m._prioritize_root_cause_coverage):
    self.assertEqual(fn(rows,limit),fn(rows,limit,context_selected=None));self.assertEqual(fn(rows,limit),fn(rows,limit,context_selected=lambda row:False))
if __name__=='__main__':unittest.main(verbosity=2)
