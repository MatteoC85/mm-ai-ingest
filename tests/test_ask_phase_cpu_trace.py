"""Content-free diagnostic timings; no functional/authority policy changes."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from pathlib import Path
import sys,time,unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import phase_trace as p
from machinemind.evidence import legacy_compatibility as l
from machinemind.retrieval import receipt_producers as producers

class Clock:
 def __init__(self):self.wall=0.;self.cpu=0;self.thread=1
 def now(self):return self.wall
 def ticks(self):return self.cpu
 def who(self):return self.thread
 def advance(self,wall,cpu):self.wall+=wall;self.cpu+=int(cpu*1_000_000_000)
 def trace(self):return p.PhaseTrace(started=0,clock=self.now,cpu_clock=self.ticks,thread_identity=self.who)

class AskPhaseCpuTraceTests(unittest.TestCase):
 def test_macro_same_thread_cpu_union_preserves_existing_wall(self):
  c=Clock();trace=c.trace();token=p.activate(trace)
  try:
   with p.span('core.refine'):
    c.advance(1,1)
    with p.span('retrieval.compose'):c.advance(4,2)
    c.advance(3,2)
   outer,child=trace.summary()['spans']
   self.assertEqual((outer['wall_seconds'],outer['exclusive_wall_seconds']),(8.,4.))
   self.assertEqual((outer['thread_cpu_seconds'],outer['exclusive_thread_cpu_seconds']),(5.,3.))
   self.assertEqual(child['thread_cpu_seconds'],2.)
  finally:p.deactivate(token)
 def test_off_thread_child_is_not_subtracted_and_cross_thread_end_unknown(self):
  c=Clock();trace=c.trace();token=p.activate(trace)
  try:
   with p.span('core.refine'):
    c.advance(1,1);saved=c.cpu;c.thread=2;c.cpu=0
    with p.span('retrieval.compose'):c.advance(2,2)
    c.thread=1;c.cpu=saved;c.advance(1,1)
   outer,child=trace.summary()['spans']
   self.assertEqual(outer['thread_cpu_seconds'],2.)
   self.assertEqual(outer['exclusive_thread_cpu_seconds'],2.)
   self.assertEqual(child['thread_cpu_seconds'],2.)
   n=trace.begin('core.prepare');c.thread=3;trace.end(n,'completed')
   self.assertIsNone(trace.summary()['spans'][-1]['thread_cpu_seconds'])
   self.assertFalse(trace.summary()['spans'][-1]['thread_cpu_valid'])
  finally:p.deactivate(token)
 def test_copied_context_child_uses_same_owner_without_exposing_thread_identity(self):
  trace=p.PhaseTrace(started=time.monotonic());token=p.activate(trace)
  try:
   with p.span('core.refine'):
    context=copy_context()
    def worker():
     with p.span('retrieval.compose'):
      return sum(i*i for i in range(10000))
    with ThreadPoolExecutor(max_workers=1)as pool:self.assertGreater(pool.submit(context.run,worker).result(),0)
   summary=trace.summary();self.assertEqual([r['parent']for r in summary['spans']],[None,1])
   self.assertTrue(all(r['thread_cpu_valid']for r in summary['spans']))
   serialized=json.dumps(summary)
   self.assertNotIn('cpu_thread',serialized);self.assertNotIn('cpu_start',serialized)
   self.assertNotIn('cpu_end',serialized)
  finally:p.deactivate(token)
 def test_hot_aggregate_separates_blocking_wall_from_cpu_and_preserves_value(self):
  c=Clock();trace=c.trace();token=p.activate(trace);value=object()
  @p.hot('legacy.capture')
  def operation():c.advance(2,.01);return value
  try:
   self.assertIs(operation(),value)
   row=trace.summary()['hot_operations']['legacy.capture']
   self.assertEqual((row['calls'],row['completed_calls'],row['error_calls']),(1,1,0))
   self.assertEqual((row['inclusive_wall_seconds'],row['inclusive_thread_cpu_seconds']),(2.,.01))
   self.assertEqual(trace.summary()['spans'],[])
  finally:p.deactivate(token)
 def test_cpu_clock_failure_and_recording_failure_never_replace_original_exception(self):
  c=Clock();trace=c.trace();trace.cpu_clock=lambda:(_ for _ in ()).throw(OSError('private clock text'))
  token=p.activate(trace);error=ValueError('private original failure')
  @p.hot('receipt.same_value')
  def operation():raise error
  try:
   with p.span('core.prepare'):c.advance(1,0)
   with self.assertRaises(ValueError)as got:operation()
   self.assertIs(got.exception,error)
   self.assertFalse(trace.summary()['spans'][0]['thread_cpu_valid'])
   self.assertEqual(trace.summary()['hot_operations']['receipt.same_value']['cpu_invalid_calls'],1)
   with patch.object(trace,'record_hot',side_effect=RuntimeError('private record failure')):
    with self.assertRaises(ValueError)as got:operation()
    self.assertIs(got.exception,error)
    self.assertEqual(p.hot('receipt.same_value')(lambda:42)(),42)
   self.assertTrue(trace.summary()['hot_operations_incomplete'])
   self.assertNotIn('private',json.dumps(trace.summary()))
  finally:p.deactivate(token)
 def test_invalid_wall_or_cpu_ticks_are_unavailable_not_fabricated(self):
  c=Clock();trace=c.trace();token=p.activate(trace)
  try:
   fn=p.hot('legacy.capture')(lambda:'ok')
   for bad in (float('nan'),'invalid',None):
    trace.clock=lambda:bad;trace.cpu_clock=lambda:bad
    self.assertEqual(fn(),'ok')
   trace.clock=c.now;trace.cpu_clock=c.ticks
   summary=trace.summary();row=summary['hot_operations']['legacy.capture']
   self.assertEqual((row['calls'],row['wall_invalid_calls'],row['cpu_invalid_calls']),(3,3,3))
  finally:p.deactivate(token)
 def test_inactive_off_root_smart_wrappers_never_sample_clocks(self):
  self.assertIsNone(p._ACTIVE.get())
  with patch.object(p.PhaseTrace,'_hot_sample',side_effect=AssertionError('inactive sample')):
   for name in p.HOT_NAMES:
    value=object();self.assertIs(p.hot(name)(lambda:value)(),value)
   limits=l.LegacyLimits(16,1000,10000)
   self.assertTrue(l._LegacyWitness.capture({'x':[-0.0,True]},limits).matches({'x':[-0.0,True]}))
   self.assertTrue(producers._same_value({'x':-0.0},{'x':-0.0}))
 def test_fixed_three_buckets_saturate_without_per_call_span_growth(self):
  c=Clock();trace=c.trace();token=p.activate(trace)
  try:
   f=p.hot('receipt.same_value')(lambda:True)
   for _ in range(10000):self.assertTrue(f())
   row=trace.hot_rows['receipt.same_value'];row['calls']=p._MAX_COUNTER
   f();summary=trace.summary()
   self.assertEqual(set(summary['hot_operations']),p.HOT_NAMES)
   self.assertEqual(summary['spans'],[])
   self.assertEqual(summary['hot_operations']['receipt.same_value']['calls'],p._MAX_COUNTER)
   self.assertTrue(summary['hot_operations_saturated'])
   with self.assertRaises(ValueError):p.hot('source-secret-id')
  finally:p.deactivate(token)
 def test_actual_capture_and_adapter_hooks_preserve_wire_limits_and_mutation_checks(self):
  from test_lineage_copy_reuse import fixture
  trace=p.PhaseTrace(started=time.monotonic());token=p.activate(trace)
  try:
   lineage,records=fixture(vector_size=32)
   row=records[0];witness=lineage.entry(row)[1]
   original=l._LegacyWitness.capture.__func__.__wrapped__(l._LegacyWitness,row,lineage.limits.legacy)
   self.assertEqual(l._wire_json(witness._tree),l._wire_json(original._tree))
   self.assertEqual((witness.size_bytes,witness.retained_metadata_bytes,witness._limits_profile),
    (original.size_bytes,original.retained_metadata_bytes,original._limits_profile))
   self.assertTrue(producers._same_value(row,witness))
   rows=trace.summary()['hot_operations']
   self.assertGreaterEqual(rows['legacy.capture']['calls'],1)
   self.assertGreaterEqual(rows['receipt.adapt_candidate']['calls'],1)
   self.assertGreaterEqual(rows['receipt.same_value']['calls'],1)
   row['page_from']=99
   with self.assertRaises(producers.EvidenceProductionError):lineage.entry(row)
   bounds=l.LegacyLimits(1,1,1)
   with self.assertRaises(l.EvidenceAssemblyError):l._LegacyWitness.capture(row,bounds)
  finally:p.deactivate(token)

if __name__=='__main__':unittest.main(verbosity=2)
