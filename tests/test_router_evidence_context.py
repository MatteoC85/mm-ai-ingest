"""Offline router evidence packing: no model, database or HTTP."""
import copy
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval.evidence_assurance import (
    V13GateCandidateBlockRuntime, v13_gate_candidate_block,
)


def runtime(*, reverse=False, limit=10):
    def summary(query, candidates):
        rows = [{**copy.deepcopy(c), "gate_similarity": 0.4,
                 "gate_overlap": 0.3, "gate_exact_code_hit": False}
                for c in candidates if isinstance(c, dict)]
        if reverse:
            rows.reverse()
        return {"candidates": rows}
    return V13GateCandidateBlockRuntime(
        V13_EVIDENCE_GATE_MAX_CANDIDATES=limit,
        _clean_display_text=lambda value, max_len=180: str(value or "")[:max_len],
        _source_type_from_document_id=lambda value: "document",
        _v13_candidate_text=lambda c: str(c.get("chunk_full") or c.get("text") or c.get("snippet") or ""),
        _v13_evidence_signal_summary=summary, json=json, re=re)


def candidate(index, text="Documented operation and its applicable conditions."):
    return {"citation_id": f"unit-{index}", "bubble_document_id": "fixture-document",
            "source_type": "document", "chunk_full": text}


def prompt_ids(block):
    return re.findall(r"(?m)^\[([^\]\n]+)\] source_type=", block)


class RouterEvidenceContextTests(unittest.TestCase):
    def test_late_query_passage_is_visible_without_changing_source_or_budget(self):
        for query, passage in (("come regolo il dosatore", "Per regolare il dosatore, chiudere il circuito prima della taratura."),
                               ("how do I align the coupling", "For alignment of the coupling, isolate the drive before adjustment.")):
            with self.subTest(query=query):
                text = "DOCUMENT TITLE. " + ("General introduction. " * 90) + passage + (" Additional context." * 30)
                original = candidate(0, text)
                before = copy.deepcopy(original)
                block, supplied = v13_gate_candidate_block(query, [original], runtime=runtime())
                self.assertIn("DOCUMENT TITLE.", block)
                self.assertIn(passage, block)
                self.assertIn(" […] ", block)
                self.assertEqual(original, before)
                self.assertEqual(supplied[0]["chunk_full"], text)
                self.assertLessEqual(len(block.split("\n", 1)[1]), 1200)

    def test_common_query_words_do_not_displace_rare_late_match(self):
        text = "Machine information. " + "General description. " * 100 + "Calibrate the actuator after isolation. " + "Continuation. " * 30
        candidates = [candidate(0, text), candidate(1, "Machine reference table."), candidate(2, "Machine specifications.")]
        block, _ = v13_gate_candidate_block("calibrate actuator machine", candidates, runtime=runtime())
        self.assertIn("Calibrate the actuator after isolation.", block)

    def test_no_match_preserves_original_head(self):
        text = "Original paragraph. " * 100
        block, _ = v13_gate_candidate_block("unmatched query", [candidate(0, text)], runtime=runtime())
        self.assertEqual(block.split("\n", 1)[1], text[:1200].strip())

    def test_conflicting_signal_ranking_retains_both_rankings(self):
        candidates = [candidate(index) for index in range(12)]
        before = copy.deepcopy(candidates)
        block, supplied = v13_gate_candidate_block("Generic machine operation", candidates, runtime=runtime(reverse=True))
        ids = {row["citation_id"] for row in supplied}
        self.assertTrue({row["citation_id"] for row in candidates[:5]}.issubset(ids))
        self.assertTrue({row["citation_id"] for row in candidates[-5:]}.issubset(ids))
        self.assertEqual(len(supplied), 10)
        self.assertEqual(prompt_ids(block), [row["citation_id"] for row in supplied])
        self.assertEqual(candidates, before)
        self.assertTrue(all(row["gate_similarity"] == 0.4 for row in supplied))

    def test_unrenderable_record_does_not_hide_later_valid_evidence(self):
        oversized = candidate(0)
        oversized["citation_id"] = "x" * 15000
        block, supplied = v13_gate_candidate_block("Generic operation", [oversized, candidate(1), candidate(2)], runtime=runtime())
        self.assertEqual([row["citation_id"] for row in supplied], ["unit-1", "unit-2"])
        self.assertEqual(prompt_ids(block), ["unit-1", "unit-2"])
        self.assertLessEqual(len(block), 14000)

    def test_budget_includes_separators_between_records(self):
        block, _ = v13_gate_candidate_block("Generic operation", [candidate(0, "t" * 1200)], runtime=runtime(limit=1))
        padding = 1399 - len(block)
        self.assertGreater(padding, 0)
        candidates = []
        for index in range(10):
            row = candidate(index, "t" * 1200)
            row["citation_id"] += "x" * padding
            candidates.append(row)
        block, supplied = v13_gate_candidate_block("Generic operation", candidates, runtime=runtime())
        self.assertLessEqual(len(block), 14000)
        self.assertLess(len(supplied), 10)
        self.assertEqual(prompt_ids(block), [row["citation_id"] for row in supplied])

    def test_invalid_and_duplicate_units_are_not_reported_as_supplied(self):
        candidates = [candidate(0), candidate(0), candidate(1, ""),
                      {"citation_id": "", "chunk_full": "No usable citation identity"}, candidate(2)]
        block, supplied = v13_gate_candidate_block("Generic operation", candidates, runtime=runtime())
        self.assertEqual([row["citation_id"] for row in supplied], ["unit-0", "unit-2"])
        self.assertEqual(prompt_ids(block), ["unit-0", "unit-2"])

    def test_excerpt_and_candidate_limits_remain_bounded(self):
        candidates = [candidate(index, "x" * 1200 + "TAIL_NOT_SENT") for index in range(12)]
        block, supplied = v13_gate_candidate_block("Generic operation", candidates, runtime=runtime(limit=3))
        self.assertEqual(len(supplied), 3)
        self.assertNotIn("TAIL_NOT_SENT", block)
        self.assertLessEqual(len(block), 14000)
        self.assertEqual(prompt_ids(block), [row["citation_id"] for row in supplied])


if __name__ == "__main__":
    unittest.main(verbosity=2)
