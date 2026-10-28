"""Offline operational spans retain query/facet anchors without forcing full families."""
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
from machinemind.ask import task_generation as tasks


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


def step(number, title, action, note="Keep the documented protective conditions."):
    key = f"step:offline-operation-{number}"
    text = (f"SOURCE_TYPE: step\nTITLE: {title}\nSTEP_NUMBER: {number}\n"
            f"PROCEDURE_ID: offline-family\nDESCRIPTION: OPERATIONAL ACTION\n{action}\n"
            f"SAFETY NOTE\n{note}")
    return {"citation_id": key + ":p1:c1", "bubble_document_id": key, "source_type": "step",
            "chunk_full": text, "snippet": text, "page_from": 1, "page_to": 1}


def facet(text, answer_type, *, must_cover=True):
    return {"facet": text, "answer_type": answer_type, "must_cover": must_cover,
            "dense_queries": [text], "lexical_queries": [text], "exact_terms": []}


class CompoundProcedureSelectionTests(unittest.TestCase):
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
        self.rows = [
            step(1, "Safe setup", "Isolate the supply."),
            step(2, "External cleaning", "Clean the exterior enclosure."),
            step(3, "Mount cartridge", "Place cartridge onto spindle."),
            step(4, "Retain cartridge", "Secure cartridge retaining clamp."),
            step(5, "Ribbon routing", "Pass ribbon through feed rollers."),
            step(6, "Open guide", "Release the guide lever."),
            step(7, "Insert ribbon", "Feed ribbon into carriage jaw."),
            step(8, "Restore guide", "Close the guide lever."),
            step(9, "Final checks", "Check ribbon alignment and tension."),
            step(10, "Optical calibration", "Calibrate the optical sensor."),
        ]
        self.q = "How do I load the consumable and check it?"
        self.planner = {"information_task": "procedure_segment",
            "required_answer_types": ["ordered_actions", "checklist", "safety_conditions"],
            "facet_queries": [facet("Place cartridge onto spindle", "ordered_actions"),
                facet("Pass ribbon through feed rollers into carriage jaw", "explanation"),
                facet("Check ribbon alignment and tension", "checklist")]}

    def select(self, q=None, planner=None):
        return self.m._v12_select_response_steps(all_steps=self.rows, selected_step_ids=[],
            model_used_citations=[], q=self.q if q is None else q,
            planner=self.planner if planner is None else planner)

    def numbers(self, rows):
        return [self.m._v12_step_sort_key(row)[0] for row in rows]

    def test_operations_take_precedence_over_sparse_checklist(self):
        for operation in ("ordered_actions", "state_sequence"):
            for in_facets in (False, True):
                planner = {"required_answer_types": ["checklist", "safety_conditions"]}
                if in_facets:
                    planner["facet_queries"] = [facet("Operating sequence", operation)]
                else:
                    planner["required_answer_types"].append(operation)
                self.assertEqual(self.m._v12_procedure_selection_mode(self.q, planner), "contiguous_span")
        pure = {"required_answer_types": ["checklist", "safety_conditions"]}
        self.assertEqual(self.m._v12_procedure_selection_mode("Which checks apply?", pure), "sparse_ordered_steps")

    def test_compound_facets_preserve_inbetween_actions_and_exclude_unrelated_ends(self):
        # The user's word 'consumable' differs from the admitted source term.
        # Router-provided operational variants anchor the beginning and end.
        selected = self.select()
        self.assertEqual(self.numbers(selected), list(range(3, 10)))
        self.assertIn(self.rows[3], selected)  # retaining prerequisite
        self.assertIn(self.rows[5], selected)  # open mechanism before insertion
        self.assertIn(self.rows[7], selected)  # restore before final check
        self.assertNotIn(self.rows[1], selected)
        self.assertNotIn(self.rows[9], selected)
        self.assertGreater(len(selected), 6)  # no sparse presentation cap

    def test_mandatory_facets_can_anchor_when_question_has_no_literal_overlap(self):
        self.assertEqual(self.numbers(self.select(q="zephyr quasar")), list(range(3, 10)))
        empty = {**self.planner, "facet_queries": []}
        self.assertEqual(self.select(q="zephyr quasar", planner=empty), [])

    def test_optional_or_pure_safety_facets_cannot_expand_operational_span(self):
        planner = deepcopy(self.planner)
        planner["facet_queries"] += [facet("Clean the exterior enclosure", "ordered_actions", must_cover=False),
            facet("Isolate the supply", "safety_conditions")]
        self.assertEqual(self.numbers(self.select(planner=planner)), list(range(3, 10)))

    def test_explicit_range_overrides_compound_facets_and_keeps_closure(self):
        q = "from open guide to insert ribbon"
        selected = self.select(q=q)
        self.assertEqual(self.numbers(selected), [6, 7, 8])
        self.assertNotIn(self.rows[2], selected)
        self.assertNotIn(self.rows[8], selected)

    def test_unresolved_explicit_endpoint_cannot_fall_back_to_compound_facets(self):
        for q in ("from open guide to zephyr quasar", "from zephyr quasar to insert ribbon"):
            self.assertNotEqual(self.m._v12_query_range_anchors(q), ("", ""))
            self.assertEqual(self.select(q=q), [])

    def test_checks_only_request_remains_sparse(self):
        planner = {"information_task": "procedure_segment",
            "required_answer_types": ["checklist", "safety_conditions"],
            "facet_queries": [facet("Isolate the supply", "safety_conditions"),
                facet("Check ribbon alignment and tension", "checklist")]}
        numbers = self.numbers(self.select(q="Which checks apply?", planner=planner))
        self.assertIn(1, numbers)
        self.assertIn(9, numbers)
        self.assertNotIn(6, numbers)
        self.assertNotIn(8, numbers)

    def test_compound_sequence_keeps_independent_verifier_obligation(self):
        m = self.m
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE, effective_mode="ask", confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task=m.INFO_PROCEDURE_SEGMENT,
            required_answer_types=(m.REQ_ORDERED_ACTIONS, m.REQ_CHECKLIST, m.REQ_SAFETY_CONDITIONS))
        self.assertTrue(m._assistant_core_should_semantic_verify_answer(decision))
        self.assertFalse(m._assistant_core_should_semantic_verify_answer(replace(decision,
            required_answer_types=(m.REQ_ORDERED_ACTIONS, m.REQ_SAFETY_CONDITIONS))))

    def test_real_producer_preserves_safety_step_without_unrelated_intermediate_operation(self):
        m = self.m
        parent = {"citation_id": "procedure:offline-family:p1:c1", "bubble_document_id": "procedure:offline-family",
            "source_type": "procedure", "page_from": 1, "page_to": 1,
            "chunk_full": "SOURCE_TYPE: procedure\nTITLE: Consumable loading\nSHORT_DESCRIPTION: Operational sequence"}
        calls = []
        def model(*args, **kwargs):
            calls.append(kwargs["purpose"])
            return {"answer_status": "answered", "grounded_points": [
                {"text": "Check ribbon alignment and tension.", "citation_ids": [self.rows[8]["citation_id"]]}]}, "offline"
        runtime = replace(m._assistant_core_task_synthesis_runtime(), preserve_complete=True,
            _structured_rescue_query_intent=lambda *args: False,
            _v12_curate_structured_sources=lambda **kw: [parent] + deepcopy(self.rows),
            _v13_fetch_manual_support_deterministic=lambda **kw: [], _v13_json_models=model,
            _sanitize_citations_for_response=lambda rows, **kw: rows,
            _build_rg_links=lambda *args, **kw: [],
            _finalize_ask_response_for_ui=lambda response, **kw: response)
        result = tasks.structured_ask(q=self.q, company_id="offline-company", machine_id="offline-machine",
            response_language="en", top_k=8, planner=self.planner, seed_citations=self.rows, debug=False, runtime=runtime)
        self.assertIsNotNone(result)
        self.assertEqual(calls, ["ask_structured_synthesis"])
        self.assertEqual(result["meta"]["procedure_source_selection"]["selected_step_numbers"], list(range(3, 10)))
        for phrase in ("Isolate the supply", "Place cartridge", "Secure cartridge", "Pass ribbon",
                       "Release the guide lever", "Feed ribbon into carriage jaw", "Close the guide lever", "Check ribbon"):
            self.assertIn(phrase, result["answer"])
        self.assertNotIn("Clean the exterior enclosure", result["answer"])
        self.assertNotIn("Calibrate the optical sensor", result["answer"])
        self.assertTrue(result["_assistant_core_source_ordered_procedure"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
