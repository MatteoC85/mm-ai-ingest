"""Fresh page expansions share Root's canonical score scale before the cap.

Readers return offline fixtures; preparation, canonical scoring, deduplication,
Root viability and selection are real. No provider or database calls are allowed.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import AssistantCoreDecision, AssistantCoreRequest
from machinemind.retrieval import evidence_orchestration as orchestration
import test_ask_bounded_acquisition as harness


class RootExpansionScoreScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.AskBoundedAcquisitionTests.setUpClass()
        cls.m = harness.AskBoundedAcquisitionTests.m

    @classmethod
    def tearDownClass(cls):
        harness.AskBoundedAcquisitionTests.tearDownClass()

    def row(self, page, text, *, semantic=0.0, score=0.0):
        return {"citation_id": f"offline-manual:p{page}:c0",
            "bubble_document_id": "offline-manual", "source_type": "document",
            "page_from": page, "page_to": page, "chunk_index": page,
            "chunk_full": text, "snippet": text[:800], "exact_machine_scope": True,
            "semantic_similarity": semantic, "similarity": semantic,
            "v13_score": score, "retrieval_score": score,
            "causal_strength_score": .25 if semantic else 0.0,
            "subsystem_score": .2 if semantic else 0.0,
            "context_fit_score": .1 if semantic else 0.0,
            "role_group": "core", "matched_subsystems": ["pump"] if semantic else []}

    def fixture(self, *, mode="root_cause", raw_score=40.0):
        request = AssistantCoreRequest(query="Which documented causes explain pump delivery stopping?",
            requested_mode=mode, response_language="en", company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=8)
        decision = AssistantCoreDecision(request_kind="fault_diagnostic", effective_mode=mode,
            confidence=.95, requested_mode_fit=True, evidence_state="refine",
            evidence_policy="machine_sources_required", information_task="fault_diagnostic",
            required_answer_types=("diagnostic_causes",), preferred_source_types=("document",))
        initial = [self.row(1, "Pump delivery stops when the isolation valve is closed. "
            "Consult recorded valve position; do not open guards.", semantic=.85, score=.85)]
        initial[0]["assistant_core_facet_hits"] = ["preserved original facet"]
        neighbors = [self.row(30+i, f"Packaging section {i}: storage crates and shipping labels.",
            score=raw_score+i) for i in range(26)]
        sections = [self.row(90, "Pump delivery: an obstructed inlet prevents liquid flow. "
            + "Preserve documented inspection conditions. " * 100 + "COMPLETE_SECTION_END",
            semantic=.6, score=raw_score*2)]
        return request, decision, initial, neighbors, sections

    def prepare(self, fixture, *, expansions=True, lineage=None):
        request, decision, initial, neighbors, sections = fixture
        before = deepcopy((initial, neighbors, sections))
        runtime = self.m._assistant_core_prepare_evidence_runtime()
        calls = {"score": [], "read": [], "capped": []}

        def score(query, rows, **kwargs):
            result = runtime._v13_score_candidates(query, rows, **kwargs)
            calls["score"].append({"before": deepcopy(rows), "after": deepcopy(result),
                "preserve_complete": kwargs.get("preserve_complete")})
            return result

        def neighbors_read(**kwargs):
            calls["read"].append("neighbors")
            return deepcopy(neighbors) if expansions else []

        def section_read(**kwargs):
            calls["read"].append("sections")
            return deepcopy(sections) if expansions else []

        def rescore(query, rows):
            calls["capped"] = deepcopy(rows)
            return runtime._v13_rescore_root_candidates(query, rows)

        local = replace(runtime,
            _assistant_core_root_diagnostic_evidence_assurance=lambda r, p, d: p,
            _assistant_core_enumeration_requested=lambda r, d: True,
            _v13_current_budget=lambda: None,
            _v13_assurance_fetch_neighbor_pages=neighbors_read,
            _assistant_core_expand_enumeration_sections=section_read,
            _v13_score_candidates=score, _v13_rescore_root_candidates=rescore)
        result = orchestration.assistant_core_prepare_evidence(request,
            {"candidates": deepcopy(initial)}, decision, runtime=local, lineage=lineage)
        self.assertEqual((initial, neighbors, sections), before)
        return result, calls

    def test_raw_reader_scale_does_not_displace_existing_root_evidence_at_cap(self):
        low, low_calls = self.prepare(self.fixture(raw_score=.01))
        high, high_calls = self.prepare(self.fixture(raw_score=4000.0))
        projection = lambda result: [(r["citation_id"], r["chunk_full"], r["v13_score"])
            for r in result["retrieval"]["candidates"]]
        self.assertEqual(projection(low), projection(high))
        self.assertTrue(high["supported"])
        for calls in (low_calls, high_calls):
            self.assertEqual(len(calls["capped"]), 24)
            self.assertIn("offline-manual:p1:c0", [r["citation_id"] for r in calls["capped"]])
        self.assertIn("offline-manual:p1:c0", [r[0] for r in projection(high)])

    def test_only_fresh_neighbors_and_sections_scored_with_complete_text(self):
        fixture = self.fixture()
        _, calls = self.prepare(fixture)
        self.assertEqual(calls["read"], ["neighbors", "sections"])
        self.assertEqual(len(calls["score"]), 2)
        for scored, expected in zip(calls["score"], fixture[3:]):
            self.assertEqual(scored["before"], expected)
            self.assertIs(scored["preserve_complete"], True)
            self.assertEqual({r["citation_id"]: r["chunk_full"] for r in scored["after"]},
                {r["citation_id"]: r["chunk_full"] for r in expected})
        initial = next(r for r in calls["capped"] if r["page_from"] == 1)
        self.assertEqual(initial["assistant_core_facet_hits"], ["preserved original facet"])
        section = next(r for r in calls["capped"] if r["page_from"] == 90)
        self.assertTrue(section["chunk_full"].endswith("COMPLETE_SECTION_END"))

    def test_empty_expansions_do_not_rescore_initial_pool(self):
        fixture = self.fixture()
        result, calls = self.prepare(fixture, expansions=False)
        self.assertEqual(calls["score"], [])
        self.assertEqual([r["citation_id"] for r in result["retrieval"]["candidates"]],
            [fixture[2][0]["citation_id"]])
        self.assertEqual(calls["capped"][0]["chunk_full"], fixture[2][0]["chunk_full"])

    def test_owned_ask_keeps_existing_fresh_scoring_behavior(self):
        _, calls = self.prepare(self.fixture(mode="ask"), lineage=lambda *a, **kw: None)
        self.assertEqual(len(calls["score"]), 2)
        self.assertTrue(all(c["preserve_complete"] for c in calls["score"]))

    def test_legacy_ask_does_not_gain_root_scoring(self):
        _, calls = self.prepare(self.fixture(mode="ask"))
        self.assertEqual(calls["read"], ["neighbors", "sections"])
        self.assertEqual(calls["score"], [])

    def test_smart_keeps_early_selection_without_expansion_or_scoring(self):
        fixture = self.fixture(mode="smart_diagnostic")
        with_expansions, calls = self.prepare(fixture)
        without_expansions, empty_calls = self.prepare(fixture, expansions=False)
        self.assertEqual(with_expansions, without_expansions)
        self.assertEqual(calls, {"score": [], "read": [], "capped": []})
        self.assertEqual(empty_calls, calls)


if __name__ == "__main__":
    unittest.main()
