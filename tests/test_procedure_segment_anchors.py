"""Offline semantic seed recovery without lexical, tenant or ordinal guessing."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import task_generation as tasks, review_evidence


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


def step(number, title, action, *, family="family", note="Keep protective equipment on."):
    key = f"step:{family}-{number}"
    body = (f"SOURCE_TYPE: step\ntitle: {title}\nstep_number: {number}\n"
            f"procedure_id: {family}\ndescription: OPERATIONAL ACTION\n{action}\nSAFETY NOTE\n{note}")
    return dict(citation_id=key + ":p1:c1", bubble_document_id=key,
                source_type="step", chunk_full=body, snippet=body, page_from=1, page_to=1)


class ProcedureSegmentAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {"MM_USAGE_ENFORCEMENT": "off",
            "AI_INTERNAL_SECRET": "offline-fixture-only", "OPENAI_API_KEY": "offline-unused",
            "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def setUp(self):
        self.rows = [step(1, "Safe setup", "Isolate the supply."),
            step(2, "Previous operation", "Load the reel."),
            step(3, "Previous operation", "Center the reel."),
            step(4, "Previous operation", "Clamp the reel."),
            step(5, "Upstream route", "Guide strip through the upstream rollers.", note="Keep the stop pressed."),
            step(6, "Entry alignment", "Align strip with the entry."),
            step(7, "Open the mechanism", "Release the lever and loosen the guide."),
            step(8, "Passage to carriage", "Push strip to the carriage gripper."),
            step(9, "Restore guides", "Close the lever and adjust the guide."),
            step(10, "Following operation", "Select the production program.")]
        self.seeds = [self.rows[i - 1] for i in (1, 5, 6, 8)]
        self.planner = {"information_task": "procedure_segment",
                        "required_answer_types": ["ordered_actions", "safety_conditions"]}
        self.runtime = replace(self.m._assistant_core_task_synthesis_runtime(), preserve_complete=True)

    def select(self, q, seeds=None):
        args = dict(all_steps=self.rows, model_used_citations=[], q=q, planner=self.planner)
        selected = self.m._v12_select_response_steps(**args, selected_step_ids=[])
        if not selected and self.m._v12_query_range_anchors(q) == ("", ""):
            anchors = tasks.admitted_segment_anchors(self.rows, self.seeds if seeds is None else seeds,
                                                      runtime=self.runtime)
            if anchors:
                selected = self.m._v12_select_response_steps(**args, selected_step_ids=anchors)
        return [self.m._v12_step_sort_key(row)[0] for row in selected]

    def test_it_and_en_zero_overlap_recover_middle_and_restoration(self):
        for q in ("come si infila il filo in macchina", "How is the wire threaded?"):
            with self.subTest(q=q):
                self.assertTrue(all(self.m._v12_step_direct_query_score(row, q) == 0 for row in self.rows))
                self.assertEqual(self.select(q), [5, 6, 7, 8, 9])

    def test_safety_note_does_not_erase_an_operational_anchor(self):
        anchors = tasks.admitted_segment_anchors(self.rows, self.seeds, runtime=self.runtime)
        self.assertEqual(anchors, [self.rows[i - 1]["citation_id"] for i in (5, 6, 8)])

    def test_other_family_same_ordinals_and_fetched_siblings_are_not_seeds(self):
        seeds = [step(5, "Other", "Other action", family="other"), self.rows[7]]
        self.assertEqual(self.select("How is the wire threaded?", seeds), [8, 9])

    def test_safety_only_or_parent_only_cannot_authorize_arbitrary_operation(self):
        for seeds in ([self.rows[0]], [{"source_type": "procedure", "bubble_document_id": "procedure:family"}], []):
            self.assertEqual(self.select("How is the wire threaded?", seeds), [])

    def test_explicit_range_keeps_existing_limits(self):
        q = "from upstream route to entry alignment"
        expected = self.m._v12_select_response_steps(all_steps=self.rows,
            selected_step_ids=[], model_used_citations=[], q=q, planner=self.planner)
        self.assertTrue(expected)
        self.assertEqual(self.select(q), [self.m._v12_step_sort_key(row)[0] for row in expected])

    def structured_result(self, q="How is the wire threaded?"):
        m = self.m
        parent = {"citation_id": "procedure:family:p1:c1", "bubble_document_id": "procedure:family",
            "source_type": "procedure", "page_from": 1, "page_to": 1,
            "chunk_full": "SOURCE_TYPE: procedure\ntitle: Material insertion\nshort_description: Operating sequence"}
        calls = []
        def model(*args, **kwargs):
            calls.append(kwargs["purpose"])
            return ({"answer_status": "answered", "grounded_points": [
                {"text": "Push strip to the carriage gripper.", "citation_ids": [self.rows[7]["citation_id"]]}]}, "offline")
        runtime = replace(self.runtime, _structured_rescue_query_intent=lambda *args: False,
            _v12_curate_structured_sources=lambda **kw: [parent] + deepcopy(self.rows),
            _v13_fetch_manual_support_deterministic=lambda **kw: [], _v13_json_models=model,
            _sanitize_citations_for_response=lambda rows, **kw: rows,
            _build_rg_links=lambda *args, **kw: [],
            _finalize_ask_response_for_ui=lambda response, **kw: response)
        result = tasks.structured_ask(q=q, company_id="offline-company",
            machine_id="offline-machine", response_language="en", top_k=8, planner=self.planner,
            seed_citations=self.seeds, debug=False, runtime=runtime)
        return result, calls

    def test_unresolved_explicit_range_cannot_be_replaced_by_semantic_seed_boundaries(self):
        for q in ("from zephyr to quasar", "dalla bobina fino al morsetto"):
            with self.subTest(q=q):
                self.assertNotEqual(self.m._v12_query_range_anchors(q), ("", ""))
                result, calls = self.structured_result(q)
                self.assertIsNone(result)
                self.assertEqual(calls, [])

    def test_real_structured_path_renders_all_selected_source_actions_even_if_model_omits_them(self):
        result, calls = self.structured_result()
        self.assertIsNotNone(result)
        self.assertEqual(calls, ["ask_structured_synthesis"])
        self.assertEqual(result["meta"]["procedure_source_selection"]["selected_step_numbers"], [5, 6, 7, 8, 9])
        self.assertIn("Release the lever and loosen the guide", result["answer"])
        self.assertIn("Close the lever and adjust the guide", result["answer"])
        self.assertIn("Isolate the supply", result["answer"])
        self.assertNotIn("Load the reel", result["answer"])
        self.assertTrue(result["_assistant_core_source_ordered_procedure"])

    def test_packet_identity_telemetry_is_bounded_to_emitted_sources_without_bodies(self):
        packet = review_evidence.compile_packet(primary=self.rows[:2], extension=self.rows[2:4],
            render=lambda rows, **kw: "\n\n".join(row["chunk_full"] for row in rows),
            max_records=3, max_context_chars=10000)
        diagnostic = packet.diagnostic(reused=False)
        self.assertEqual(diagnostic["required_citation_ids"], [r["citation_id"] for r in self.rows[:2]])
        self.assertEqual(diagnostic["emitted_citation_ids"], [r["citation_id"] for r in self.rows[:3]])
        self.assertNotIn("Isolate the supply", repr(diagnostic))


if __name__ == "__main__":
    unittest.main(verbosity=2)
