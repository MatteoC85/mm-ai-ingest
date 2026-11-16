"""Exact, bounded private transport for already-owned ASK embedding vectors."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from test_lineage_copy_reuse import fixture
from machinemind.evidence import legacy_compatibility as legacy
from machinemind.retrieval import receipt_producers as producer


class AskJournalVectorTransportTests(unittest.TestCase):
    def setUp(self):
        self.limits = legacy.LegacyLimits(8, 100, 2000)
        self.input = {"source": "source", "embedding_list": [-0.0, 0.2, 0.3]}
        self.snapshot = legacy._LegacyWitness.capture(self.input, self.limits)

    def test_scoped_capture_preserves_exact_shape_bits_and_original_charges(self):
        record = self.snapshot.journal_record()
        vector = record["embedding_list"]
        self.assertIs(type(vector), legacy._JournalVector)
        self.assertEqual(tuple(vector), tuple(self.input["embedding_list"]))
        with self.assertRaises(TypeError):
            vector[0] = 1.0
        self.assertFalse(self.snapshot.matches(record))
        self.assertTrue(self.snapshot.matches(record, journal_vectors=True))
        with self.assertRaises(legacy.EvidenceAssemblyError):
            legacy._LegacyWitness.capture(record, self.limits)
        converted = legacy._LegacyWitness.capture(record, self.limits, journal_vectors=True)
        self.assertEqual(converted._limits_profile, self.snapshot._limits_profile)
        self.assertEqual(converted.size_bytes, self.snapshot.size_bytes)
        self.assertEqual(converted.retained_metadata_bytes, self.snapshot.retained_metadata_bytes)
        self.assertEqual(legacy._wire_json(converted._tree), legacy._wire_json(self.snapshot._tree))
        self.assertIs(type(converted.detached()["embedding_list"]), list)

    def test_vector_scope_never_covers_an_unrelated_field(self):
        vector = self.snapshot.journal_record()["embedding_list"]
        with self.assertRaises(legacy.EvidenceAssemblyError):
            legacy._LegacyWitness.capture({"notes": vector}, self.limits, journal_vectors=True)
        unrelated = legacy._LegacyWitness.capture({"notes": [1.0, 2.0]}, self.limits)
        self.assertIs(type(unrelated.journal_record()["notes"]), list)

    def test_signed_zero_and_numeric_types_remain_exact(self):
        for bad in ((0.0, 0.2, 0.3), (-0.0, 0, 0.3), (-0.0, 0.2, 0.4)):
            session_witness = legacy._LegacyWitness.capture({**self.input, "embedding_list": list(bad)}, self.limits)
            self.assertFalse(self.snapshot.matches(session_witness.journal_record(), journal_vectors=True))
        self.assertFalse(self.snapshot.matches({**self.input, "embedding_list": tuple(self.input["embedding_list"])},
                                               journal_vectors=True))

    def test_snapshot_bounds_apply_to_the_whole_logical_vector(self):
        compact = self.snapshot.journal_record()
        for key, value in (("max_depth", 1), ("max_nodes", 5), ("max_bytes", 20)):
            limits = replace(self.limits, **{key: value})
            with self.assertRaises(legacy.EvidenceAssemblyError):
                legacy._LegacyWitness.capture(self.input, limits)
            with self.assertRaises(legacy.EvidenceAssemblyError):
                legacy._LegacyWitness.capture(compact, limits, journal_vectors=True)

    def test_trace_exports_fresh_lists_and_rejects_untracked_or_mutated_output(self):
        original = producer._PreparationTrace.__init__

        def create(trace, **kwargs):
            original(trace, **kwargs, compact_vectors=True)

        with patch.object(producer._PreparationTrace, "__init__", create):
            trace, records = fixture(2, vector_size=16)
        first = records[0]
        vector = first["embedding"]
        self.assertIs(type(vector), legacy._JournalVector)
        with self.assertRaises(producer.EvidenceProductionError):
            trace.public_views((dict(first),))
        first["embedding"] = list(vector)
        first["embedding"][0] = 100.0
        with self.assertRaises(producer.EvidenceProductionError):
            trace.public_views(records)
        first["embedding"] = vector
        views = trace.public_views(records)
        trace.export(records, views)
        self.assertIs(first, records[0])
        self.assertIs(type(first["embedding"]), list)
        self.assertEqual(tuple(first["embedding"]), tuple(vector))
        first["embedding"][0] = 100.0
        self.assertNotEqual(first["embedding"][0], vector[0])


if __name__ == "__main__":
    unittest.main()
