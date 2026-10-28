"""Exact witness profile fusion: immutable shape, bytes and bounds stay equal."""
from dataclasses import replace
from collections import UserDict
from pathlib import Path
import random,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.evidence import legacy_compatibility as l

LIMITS=l.LegacyLimits(32,100000,16000000)

class WitnessProfileFusionTests(unittest.TestCase):
 def check_oracle(self,value):
  actual=l._LegacyWitness.capture(value,LIMITS)
  freezer=l._Freezer(LIMITS);tree=freezer.freeze(value)
  expected=l._LegacyWitness(tree,l._frozen_canonical_size(tree),(freezer.nodes,freezer.scalar_bytes,freezer.max_depth_seen))
  self.assertEqual(l._wire_json(actual._tree),l._wire_json(expected._tree))
  self.assertEqual(actual.size_bytes,expected.size_bytes)
  self.assertEqual(actual.retained_metadata_bytes,expected.retained_metadata_bytes)
  self.assertEqual(actual._limits_profile,expected._limits_profile)
  self.assertEqual(actual.numeric_payloads,expected.numeric_payloads)
  self.assertTrue(actual.matches(value))
  return actual

 def test_profile_exact_for_scalars_nested_unicode_aliases_and_numeric_pool(self):
  vector=[0.0,-0.0,1e-200,1e200]
  shared={'string':'accento à 漢字 😀 \n \t \\ "', 'vector':vector}
  values=[None,False,True,0,-1,10**100,0.0,-0.0,1e-300,1e300,'',[],{},(),[shared,shared],{'a':shared,'b':tuple(vector)}]
  pool=l._PayloadPool(16000000)
  with l._payload_scope(pool):
   for value in values:
    with self.subTest(value_type=type(value).__name__):self.check_oracle(value)
  pool.close()

 def test_composed_profiles_preserve_duplicate_occurrences_and_order(self):
  pool=l._PayloadPool(16000000)
  with l._payload_scope(pool):
   values=[{'first':[1.0,-0.0],'second':'é\n'},{'second':'x','first':[1.0,-0.0]}]
   witnesses=tuple(self.check_oracle(v) for v in values)
   for kind in ('list','tuple'):
    for order in ((0,1),(1,0),(0,0,1),()):
     assembled=l._LegacyWitness.sequence(kind,tuple(witnesses[i] for i in order),LIMITS)
     raw=[values[i] for i in order];raw=raw if kind=='list' else tuple(raw)
     expected=self.check_oracle(raw)
     self.assertEqual(l._wire_json(assembled._tree),l._wire_json(expected._tree))
     self.assertEqual((assembled.size_bytes,assembled.retained_metadata_bytes,assembled._limits_profile),(expected.size_bytes,expected.retained_metadata_bytes,expected._limits_profile))
     self.assertEqual(assembled.numeric_payloads,expected.numeric_payloads)
  pool.close()

 def test_exact_old_allocation_boundaries_and_invalid_values(self):
  value={'escaped':'é\n"','items':([1.0,-0.0],{'x':10**80})}
  w=self.check_oracle(value)
  for field,exact in zip(('max_nodes','max_bytes','max_depth'),w._limits_profile):
   # scalar byte bound, not canonical size, is the existing freeze check.
   self.assertEqual(l._LegacyWitness.capture(value,replace(LIMITS,**{field:exact}))._limits_profile,w._limits_profile)
   with self.assertRaises(l.EvidenceAssemblyError):l._LegacyWitness.capture(value,replace(LIMITS,**{field:exact-1}))
  cycle=[];cycle.append(cycle)
  for invalid in (cycle,float('nan'),float('inf'),[1.0,float('nan')],{1:'invalid'},object()):
   with self.subTest(kind=type(invalid).__name__),self.assertRaises(l.EvidenceAssemblyError):l._LegacyWitness.capture(invalid,LIMITS)

 def test_mapping_size_uses_actual_items_as_the_original_tree_did(self):
  class UnusualMapping(UserDict):
   def __len__(self):return 1
  value=UnusualMapping({'a':'escaped\n', 'b':[1.0,-0.0], 'c':{'nested':'x'}})
  actual=l._LegacyWitness.capture(value,LIMITS)
  freezer=l._Freezer(LIMITS);tree=freezer.freeze(value)
  old=l._LegacyWitness(tree,l._frozen_canonical_size(tree),(freezer.nodes,freezer.scalar_bytes,freezer.max_depth_seen))
  self.assertEqual((actual.size_bytes,actual.retained_metadata_bytes),(old.size_bytes,old.retained_metadata_bytes))
  self.assertEqual(l._wire_json(actual._tree),l._wire_json(old._tree))

 def test_component_identity_rechecks_each_value_and_distinct_floats_fall_back(self):
  vector=[float(i)/13 for i in range(32)] + [-0.0]
  witness=l._LegacyWitness.capture(vector,LIMITS)
  with patch.object(l,'array',wraps=l.array) as binary:
   self.assertTrue(witness.matches(list(vector)))
   self.assertEqual(binary.call_count,0)
   distinct=[float(str(v)) for v in vector]
   self.assertTrue(witness.matches(distinct))
   self.assertEqual(binary.call_count,1)
  for index,replacement in ((0,0),(0,False),(1,999.0),(32,0.0),(1,float('nan'))):
   changed=list(vector);changed[index]=replacement
   self.assertFalse(witness.matches(changed))
  self.assertFalse(witness.matches(vector[:-1]))

 def test_random_nested_profiles_match_original_three_pass_oracle(self):
  rng=random.Random(8675309)
  def value(depth=0):
   leaf=rng.choice([None,True,False,-0.0,1e-200,-10**60,'é 漢字\n\\"',rng.randrange(-1000,1000)])
   if depth>3 or rng.randrange(3)==0:return leaf
   if rng.randrange(2):return [value(depth+1) for _ in range(rng.randrange(5))]
   return {f'key-{i}-é':value(depth+1) for i in range(rng.randrange(5))}
  for _ in range(200):self.check_oracle(value())

if __name__=='__main__':unittest.main()
