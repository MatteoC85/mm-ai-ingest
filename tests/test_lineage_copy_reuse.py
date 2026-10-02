"""Offline exact-copy lineage optimization; no database, auth or provider I/O."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.evidence.adapter_types import AdapterContext, AdapterLimits
from machinemind.evidence.assembly import AssemblyLimits
from machinemind.evidence.contracts import Provenance, SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.evidence.legacy_compatibility import LegacyLimits, LegacyRecordInput, _LegacyWitness, _PayloadPool, _payload_scope
from machinemind.evidence.manifest import ManifestLimits
from machinemind.retrieval.chunk_evidence import ChunkEvidenceLimits
from machinemind.retrieval.ask_composition import RecordHandle
from machinemind.retrieval import receipt_producers as production


def fixture(count=1, vector_size=1536):
    limits = ChunkEvidenceLimits(AdapterLimits(16000, 128, 16000),
        AssemblyLimits(ManifestLimits(3000, 64_000_000), 3000, 64_000_000),
        LegacyLimits(32, 8_000_000, 128_000_000))
    trace = production._PreparationTrace(limits=limits, max_records=3000, max_bytes=128_000_000)
    source = SourceIdentity(SourceScope('offline-company', ScopeLevel.MACHINE, 'offline-machine'),
                            SourceType.DOCUMENT, 'offline-document', SourceFormat.PDF)
    context = AdapterContext(source, Provenance('offline-observed', 'offline-original'), frozenset({source}))
    handles, inputs = [], []
    owner = object()
    for index in range(count):
        record = {'bubble_document_id': source.source_id, 'page_from': index+1, 'page_to': index+1,
            'chunk_index': index, 'citation_id': f'fixture-{index}',
            'chunk_full': f'Observed passage {index}. ' + 'Applicable operation and safety context. '*48,
            'similarity': 0.5, 'semantic_similarity': 0.5,
            'embedding': [float(index+i)/10000 for i in range(vector_size)]}
        handles.append(RecordHandle(owner, index))
        inputs.append(LegacyRecordInput('retrieval_candidate', record, context))
    records = trace.materialize(tuple(handles), tuple(inputs))
    return trace, records


def copy_once(trace, source, *, reuse):
    copied = dict(source)
    entry = trace.entry(source)
    if not production._same_value(source, copied):
        raise AssertionError('fixture copy mismatch')
    trace.add(copied, entry[2], copy_entry=entry if reuse else None)
    return copied


class LineageCopyReuseTests(unittest.TestCase):
    def test_materialization_captures_once_and_reuses_exact_parent_witness(self):
        with patch.object(_LegacyWitness, 'capture', wraps=_LegacyWitness.capture) as capture, \
             patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt, \
             patch.object(production, '_check_derived_locator', wraps=production._check_derived_locator) as locator:
            trace, records = fixture(3, vector_size=4)
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(adapt.call_count, 3)
        self.assertEqual(locator.call_count, 3)
        for record in records:
            entry = trace.entry(record)
            self.assertIs(entry[1], trace.roots[entry[2][0]].record)
        self.assertEqual(trace._witness_budget.logical_bytes,
                         sum(trace.entry(row)[1].size_bytes for row in records))

    def test_root_copy_cannot_change_any_parent_field_or_origin(self):
        for field, value in (('embedding', [9.0]*4), ('chunk_full', 'unsafe injected'),
                             ('page_from', 90), ('similarity', True)):
            with self.subTest(field=field):
                trace, rows = fixture(vector_size=4)
                entry = trace.entry(rows[0])
                changed = production.deepcopy(rows[0])
                changed[field] = value
                with self.assertRaisesRegex(production.EvidenceProductionError, 'immutable parent'):
                    trace.add(changed, entry[2], root_copy=True)
        trace, rows = fixture(vector_size=4)
        entry = trace.entry(rows[0])
        with self.assertRaisesRegex(production.EvidenceProductionError, 'one explicit'):
            trace.add(dict(rows[0]), entry[2]*2, root_copy=True)
        rows[0]['embedding'][2] = 7.0
        with self.assertRaises(production.EvidenceProductionError):
            trace.entry(rows[0])

    def test_root_copy_preserves_bounds_and_changed_context_conversion(self):
        trace, rows = fixture(vector_size=4)
        entry = trace.entry(rows[0])
        trace.max_bytes = trace.bytes
        with self.assertRaisesRegex(production.EvidenceProductionError, 'byte limit'):
            trace.add(dict(rows[0]), entry[2], root_copy=True)
        trace.max_bytes = 128_000_000
        context = replace(trace.roots[entry[2][0]].context,
                          provenance=Provenance('another-provider', 'another-reference'))
        with patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt:
            trace.add(dict(rows[0]), entry[2], context=context, root_copy=True)
        self.assertEqual(adapt.call_count, 1)

    def test_real_outer_copy_event_keeps_same_origins_and_detects_later_change(self):
        base, records = fixture(vector_size=4)
        # Exercise the exact event method; authority/session collaborators are
        # irrelevant to this pure local event and are not replaced in production.
        trace = production._OuterRetrievalTrace.__new__(production._OuterRetrievalTrace)
        trace.__dict__.update(base.__dict__)
        trace.pending, trace.completed = {}, None
        source = records[0]
        copied = trace.event('copy', source, dict(source))
        self.assertEqual(trace.entry(copied)[2], trace.entry(source)[2])
        self.assertIs(trace.entry(copied)[1], trace.entry(source)[1])
        copied['page_from'] = 2
        with self.assertRaises(production.EvidenceProductionError):
            trace.event('copy', copied, dict(copied))

    def test_explicit_copy_reuses_witness_and_pure_adapter_but_checks_parent_locator(self):
        trace, records = fixture()
        source = records[0]
        witness = trace.entry(source)[1]
        with patch.object(_LegacyWitness, 'capture', wraps=_LegacyWitness.capture) as capture, \
             patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt, \
             patch.object(production, '_check_derived_locator', wraps=production._check_derived_locator) as locator:
            copied = copy_once(trace, source, reuse=True)
        self.assertEqual(capture.call_count, 0)
        self.assertEqual(adapt.call_count, 0)
        self.assertEqual(locator.call_count, 1)
        self.assertIs(trace.entry(copied)[1], witness)
        self.assertEqual(copied, source)
        self.assertIsNot(copied, source)
        self.assertEqual(trace._witness_budget.logical_bytes, witness.size_bytes*2)

    def test_mutated_original_and_mutated_copy_both_fail(self):
        for field in ('chunk_full', 'embedding', 'page_from', 'similarity'):
            for target in ('original', 'copy'):
                with self.subTest(field=field, target=target):
                    trace, records = fixture(vector_size=4)
                    source = records[0]
                    entry = trace.entry(source)
                    copied = production.deepcopy(source)
                    changed = source if target == 'original' else copied
                    changed[field] = {'chunk_full':'injected', 'embedding':[9.0]*4,
                                      'page_from':9, 'similarity':True}[field]
                    with self.assertRaises(production.EvidenceProductionError):
                        trace.add(copied, entry[2], copy_entry=entry)

    def test_foreign_occurrence_cannot_supply_copy_witness(self):
        trace, records = fixture(vector_size=4)
        other, foreign = fixture(vector_size=4)
        entry = other.entry(foreign[0])
        with self.assertRaises(production.EvidenceProductionError):
            trace.add(dict(records[0]), entry[2], copy_entry=entry)

    def test_copy_keeps_capacity_limits_and_mutation_is_detected_later(self):
        trace, records = fixture(vector_size=4)
        trace.max_records = 1
        with self.assertRaisesRegex(production.EvidenceProductionError, 'record limit'):
            copy_once(trace, records[0], reuse=True)
        trace.max_records = 3
        trace.max_bytes = trace.bytes
        with self.assertRaisesRegex(production.EvidenceProductionError, 'byte limit'):
            copy_once(trace, records[0], reuse=True)
        trace.max_bytes = 128_000_000
        copied = copy_once(trace, records[0], reuse=True)
        copied['embedding'][0] = -0.0
        with self.assertRaises(production.EvidenceProductionError):
            trace.entry(copied)

    def test_changed_context_or_limits_rerun_pure_adapter(self):
        trace, records = fixture(vector_size=4)
        entry = trace.entry(records[0])
        context = replace(trace.roots[entry[2][0]].context, provenance=Provenance('new-provider', 'new-reference'))
        with patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt:
            trace.add(dict(records[0]), entry[2], context=context, copy_entry=entry)
        self.assertEqual(adapt.call_count, 1)

    def test_released_witness_adapter_does_not_accumulate(self):
        trace, records = fixture(vector_size=4)
        copied = copy_once(trace, records[0], reuse=True)
        witness = trace.entry(copied)[1]
        del trace.entries[id(copied)]
        trace.release_witness(witness)
        self.assertEqual(trace._adapted_views[id(witness)][4], 1)
        del trace.entries[id(records[0])]
        trace.release_witness(witness)
        self.assertFalse(trace._adapted_views)


def benchmark(count=160, rounds=3):
    output = []
    for reuse in (False, True):
        pool = _PayloadPool(64_000_000)
        with _payload_scope(pool):
            trace, rows = fixture(count)
            started = time.perf_counter()
            for _ in range(rounds):
                rows = [copy_once(trace, row, reuse=reuse) for row in rows]
            for row in rows:
                trace.entry(row)
            elapsed = time.perf_counter()-started
            output.append({'reuse': reuse, 'original_records': count, 'copy_rounds': rounds,
                'registered_views': len(trace.entries), 'wall_seconds': round(elapsed, 6),
                'logical_witness_bytes': trace._witness_budget.logical_bytes,
                'retained_witness_bytes': trace.bytes})
        pool.close()
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
