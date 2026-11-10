"""Actual R15/R16 failure variants; controlled whole-core second judgment."""
from pathlib import Path
import ast,hashlib,json,socket,sqlite3,sys,unittest,zipfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from machinemind.ask import unit_grounding as u,final_contract
import test_ask_original_grounded_recovery as replay
W=next(p for p in(ROOT.parent,ROOT.parents[1],ROOT.parents[2])if(p/'phase6-ask-postreview-correction-r16-20261110').is_dir())
TRACE=W/'phase6-ask-postreview-correction-r16-20261110/review/ACTUAL-R16-GENERATED-ANSWER-TRACE.json'
RECEIPT=W/'phase6-ask-postreview-correction-r16-20261110/live-client/receipts/R16-A-ASK-EN-FIX.json'
def deny(*a,**k):raise AssertionError('NETWORK_SQL_PROVIDER_FORBIDDEN')
class Projection(unittest.TestCase):
 def test_actual_R16_denial_colon_salvages_only_supported_independent_fact(self):
  t=replay.load(TRACE,'03a0ef832c3cfdcc54104be693a6db2d9c5316dd0e09229e35de3ca262a8bba2')['generated_answer_trace']
  text=t['first_writer_grounded_points'][0]['text']['text']
  source='Portata massima di 1000 kg.'
  old,removed=u._assistant_core_filter_unsupported_claim_sentences(text,source)
  new,dropped=u.finite_factual_claim_filter(text,source)
  self.assertNotIn('1000',old);self.assertIn('1000 kg',new)
  self.assertNotIn('2500',new);self.assertEqual(dropped,removed)
  self.assertIn('the documented maximum compatible coil weight is1000 kg'.replace('is1000','is 1000'),new)
 def test_R15_Markdown_sentence_boundary_preserves_source_value_but_binding_is_still_required(self):
  text='The documented maximum is **1,000 kg**. Do not load a 2,500 kg coil.'
  old,_=u._assistant_core_filter_unsupported_claim_sentences(text,'Maximum 1000 kg.')
  new,removed=u.finite_factual_claim_filter(text,'Maximum 1000 kg.')
  self.assertNotIn('1,000',old);self.assertIn('1,000',new);self.assertEqual(removed,['2,500'])
  verdict=final_contract.final_contract(deterministic={'passed':True},semantic={'outcome':'rewrite'},
    reviewed_answer=text,answer=new,semantic_complete=True,semantic_partial=False)
  self.assertFalse(verdict['passed']);self.assertEqual(verdict['reason'],'semantic_answer_changed_after_review')
 def test_general_EN_IT_units_and_semicolon_do_not_whitelist_user_values(self):
  cases=[
   ('A12 bar setting is not permitted: the documented pressure is6 bar.','Pressure6 bar.','6 bar','12'),
   ('Una tensione di48 V non è ammessa; la tensione documentata è24 V.','Tensione24 V.','24 V','48'),
   ('The documented pressure is6 bar; a12 bar setting is not permitted.','Pressure6 bar.','6 bar','12'),
   ('La pressione documentata è6 bar. Non usare una pressione di12 bar.','Pressione6 bar.','6 bar','12')]
  for text,source,kept,bad in cases:
   # Spaces are actual quantity separators; avoid embedding small values in words.
   text=text.replace('A12','A 12').replace('is6','is 6').replace('è24','è 24').replace('è6','è 6').replace('a12','a 12').replace('di12','di 12').replace('di48','di 48')
   source=source.replace('Pressure6','Pressure 6').replace('Tensione24','Tensione 24').replace('Pressione6','Pressione 6')
   with self.subTest(text=text):
    value,removed=u.finite_factual_claim_filter(text,source)
    self.assertIn(kept,value);self.assertNotIn(bad,value);self.assertIn(bad,removed)
 def test_scope_conditions_negated_context_wrong_entities_and_shared_units_fail_closed(self):
  examples=[
   'Only if a2500 kg coil is not permitted: the maximum load is1000 kg.',
   'For another model a2500 kg coil is not permitted: the maximum load is1000 kg.',
   'A2500 kg coil is not permitted for another model: the maximum load is1000 kg.',
   'A2500 kg coil is not permitted on another machine: the maximum load is1000 kg.',
   'A2500 kg coil is not permitted on the machine OMEGA: the maximum load is1000 kg.',
   'Una matassa da2500 kg non è ammessa per un altro modello: la portata massima è1000 kg.',
   'It is false that a2500 kg coil is not permitted: the maximum load is1000 kg.',
   'A2500 kg coil is not permitted when the guard is open: the maximum load is1000 kg.',
   'A2500 kg coil is not permitted: it is1000 kg.',
   'A2500 kg coil is not permitted: 1000 kg.',
   'A2500 kg coil is not permitted: the range is500 to1000 kg.',
   'A2500 kg coil is not permitted: the tolerance is1000 ±10 kg.',
   'A2500 kg coil on WRONG-999 is not permitted: the maximum load is1000 kg.',
   'A2500 kg coil is permitted: the maximum load is1000 kg.',
   'A2500 kg coil exceeds the maximum1000 kg.',
   '"A2500 kg coil is not permitted: the maximum load is1000 kg."',
   '“A2500 kg coil is not permitted: the maximum load is1000 kg.”',
   '‘A2500 kg coil is not permitted: the maximum load is1000 kg.’',
   '(A2500 kg coil is not permitted: the maximum load is1000 kg.)',
   'A2500 kg coil is not permitted: the maximum load is1000 V.']
  for text in examples:
   text=text.replace('A2500','A 2500').replace('a2500','a 2500').replace('is1000','is 1000').replace('is500','is 500').replace('to1000','to 1000').replace('±10','± 10').replace('maximum1000','maximum 1000').replace('da2500','da 2500').replace('è1000','è 1000')
   with self.subTest(text=text):
    value,removed=u.finite_factual_claim_filter(text,'Maximum 1000 kg. Range 500 to 1000 kg. Tolerance 10 kg.')
    self.assertNotIn('1000',value);self.assertIn('2500',removed)
 def test_quoted_Markdown_and_unbalanced_markup_do_not_create_new_boundaries(self):
  for text in ('"Maximum **1000 kg**. Load2500 kg."','Maximum1000 kg**. Load2500 kg.',
    'Only if the guard is rated **2500 kg**. The maximum is 1000 kg.',
    'For another model rated **2500 kg**. The maximum is 1000 kg.',
    'It is false that the setting is **2500 kg**. The maximum is 1000 kg.',
    'The WRONG-999 device uses **2500 kg**. The maximum is 1000 kg.',
    '“Maximum **1000 kg**. Load2500 kg.”'):
   text=text.replace('Maximum1000','Maximum 1000').replace('Load2500','Load 2500')
   value,removed=u.finite_factual_claim_filter(text,'Maximum 1000 kg.')
   self.assertNotIn('1000',value);self.assertIn('2500',removed)
 def test_default_filter_function_and_Root_sources_are_byte_AST_unchanged(self):
  archive=W/'phase6-ask-unit-projection-r17-20261110/baseline-7add.zip'
  with zipfile.ZipFile(archive)as z:
   old=ast.parse(z.read('machinemind/ask/unit_grounding.py').decode())
   new=ast.parse((ROOT/'machinemind/ask/unit_grounding.py').read_text(encoding='utf-8'))
   old_functions={n.name:ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,ast.FunctionDef)}
   new_functions={n.name:ast.dump(n,include_attributes=False)for n in new.body if isinstance(n,ast.FunctionDef)}
   self.assertTrue(all(new_functions[k]==v for k,v in old_functions.items()))
   # R18 adds only this private router-null proof; all other main AST stays exact.
   private_nodes = [
    'relevance = out.get("relevant_evidence_ids")',
    'out["_ask_catalog_explicit_null"] = ("selected_procedure_parent_id" in out and selected_parent is None and type(relevance) is list and all(type(cid) is str and cid in supplied_ids for cid in relevance))',
    'out.pop("_ask_catalog_explicit_null", None)',
   ]
   expected = [ast.dump(ast.parse(text).body[0],include_attributes=False)for text in private_nodes]
   class PrivateNullOnly(ast.NodeTransformer):
    def __init__(self):self.seen=[0,0,0]
    def visit_FunctionDef(self,node):
     if node.name!='_assistant_core_router_call':return node
     return self.generic_visit(node)
    def strip_exact(self,node):
     value=ast.dump(node,include_attributes=False)
     if value in expected:
      self.seen[expected.index(value)]+=1
      return None
     return self.generic_visit(node)
    visit_Assign=strip_exact
    visit_Expr=strip_exact
   scrub=PrivateNullOnly()
   current_main=scrub.visit(ast.parse((ROOT/'main.py').read_text(encoding='utf-8')))
   self.assertEqual(scrub.seen,[1,1,1])
   self.assertEqual(ast.dump(ast.parse(z.read('main.py').decode()),include_attributes=False),
    ast.dump(current_main,include_attributes=False))
   for name in('machinemind/ask/execution.py','machinemind/ask/final_contract.py'):
    self.assertEqual(z.read(name),(ROOT/name).read_bytes())
class WholeCore(unittest.TestCase):
 @classmethod
 def setUpClass(cls):replay.RecoveryTests.setUpClass()
 @classmethod
 def tearDownClass(cls):replay.RecoveryTests.tearDownClass()
 def run_trace(self,actual,**kw):
  h=replay.RecoveryTests()
  if actual=='R16':
   h.trace=replay.load(TRACE,'03a0ef832c3cfdcc54104be693a6db2d9c5316dd0e09229e35de3ca262a8bba2')['generated_answer_trace']
   h.receipt=replay.load(RECEIPT,'431d47e2c57b8ea55451596ef11cffdcb10e082ad164100ee3e396e16b17f4fc')
  return h.replay(**kw)
 def test_actual_R15_and_R16_baselines_fail_and_new_second_PASS_remains_exactly_bound(self):
  for actual in('R15','R16'):
   with self.subTest(actual=actual):
    result,calls,checks,h=self.run_trace(actual,disable=True,second='rewrite')
    try:self.assertEqual(result['status'],'no_sources')
    finally:h.close()
    result,calls,checks,h=self.run_trace(actual)
    try:
     self.assertEqual(result['status'],'answered',result['meta'].get('assistant_core_validation'))
     self.assertEqual([x[0]for x in calls],['router','writer','reviewer','reviewer'])
     self.assertIn('1000 kg',result['answer']);self.assertNotIn('2500',result['answer'])
     self.assertEqual(result['answer'],checks[0])
     c=result['meta']['assistant_core_validation']['answer_contract']
     self.assertTrue(c['complete']);self.assertTrue(c['semantic_answer_bound'])
     self.assertEqual(c['semantic_verifier']['review_profile'],'grounded_original_retain_only_v1')
    finally:h.close()
 def test_second_rewrite_rejection_and_missing_coverage_are_never_automatic_pass(self):
  for kwargs in({'second':'rewrite'},{'second':'no_sources'},{'second':'changedpass'},{'missing':True}):
   with self.subTest(kwargs=kwargs):
    result,calls,checks,h=self.run_trace('R16',**kwargs)
    try:self.assertEqual(result['status'],'no_sources');self.assertEqual(len(calls),4)
    finally:h.close()
if __name__=='__main__':
 with patch.object(socket.socket,'connect',deny),patch.object(socket,'create_connection',deny),patch.object(sqlite3,'connect',deny):
  unittest.main(verbosity=2)
