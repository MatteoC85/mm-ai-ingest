"""Exact, bounded private transport for already-owned ASK embedding vectors."""
from dataclasses import replace
import json
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


class AskJournalPublicWireTests(unittest.TestCase):
    def test_debug_response_with_dense_candidate_details_is_json_serializable(self):
        from test_ask_conditions_pipeline import AskConditionsPipelineTests
        from test_ask_router_family_context import AskRouterFamilyContextTests, RouterFixture
        from machinemind.retrieval.ask_composition import AskSelection
        from machinemind.retrieval.chunk_evidence import _source_from_columns, build_chunk_read
        from machinemind.retrieval.dense import raw_rows_to_dense_candidates

        fixtures = AskRouterFamilyContextTests
        fixtures.setUpClass()
        self.addCleanup(fixtures.tearDownClass)
        self.m = fixtures.m
        request_decision = AskConditionsPipelineTests.request_decision

        def debug_request(fixture, language="en"):
            request, decision = request_decision(fixture, language)
            return replace(request, debug=True), decision

        with patch.object(AskConditionsPipelineTests, "request_decision", debug_request):
            owner = RouterFixture(self)
        self.addCleanup(owner.close)
        self.assertIs(owner.request.debug, True)
        company, machine = owner.request.company_id, owner.request.machine_id
        source = "manual-journal-transport"
        owner.allowance |= frozenset({_source_from_columns(company, machine, source)})
        _, limits = owner.session.read_contract(request=owner.request,
                                                current_allowed_sources=owner.allowance)
        body = "Before installing a cartridge, verify energy isolation and use trained personnel."
        vector = [-0.0, 0.2, 0.5] * 512
        row = (source, 0, 1, 1, body, body, 0.88, vector, True)
        receipt = build_chunk_read(scope=owner.scope, kind="dense", rows=[(*row, company, machine)],
            candidates=raw_rows_to_dense_candidates([row]), limits=limits)
        registered = owner.session.register(request=owner.request, receipt=receipt,
                                             current_allowed_sources=owner.allowance)
        record, = owner.session.records(request=owner.request, handles=registered.records,
                                        current_allowed_sources=owner.allowance)
        candidates = [*owner.prepared["candidates"], dict(record.record)]
        owner.prepared.update(candidates=candidates, citations=candidates)
        handles = owner.owner._selection[0].records + registered.records
        owner.owner._selection = (AskSelection("candidates", handles), AskSelection("citations", handles))

        result = owner.run_core()
        self.assertEqual(result["status"], "answered")
        self.assertEqual([call[0] for call in owner.calls], ["router", "writer", "reviewer"])
        payload = json.loads(json.dumps(result, ensure_ascii=False))
        details = payload["meta"]["assistant_core_evidence_admission"]["deterministic_signals"]["candidates"]
        document = next(item for item in details if item["bubble_document_id"] == source)
        self.assertEqual(document["chunk_full"], body)
        self.assertEqual(document["embedding_list"], vector)
        self.assertEqual(document["embedding_list"][0].hex(), "-0x0.0p+0")


if __name__ == "__main__":
    unittest.main()
