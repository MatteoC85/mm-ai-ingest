"""Offline regressions for manual actions and protected synthesis scheduling.

Model output is fixed; network/database access is denied. These tests exercise
the real renderer and completion scheduler, not live model quality or authority.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, generation, request_completion


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class ManualProcedureSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0",
        }))
        cls.guards.enter_context(patch.object(socket, "create_connection", denied))
        cls.guards.enter_context(patch.object(socket.socket, "connect", denied))
        cls.guards.enter_context(patch("requests.sessions.Session.request", denied))
        cls.guards.enter_context(patch("psycopg2.connect", denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def generation_case(self, *, protected=True, task="procedure_full", count=8,
                        unknown_citation=False):
        actions = ["Isolate the supply", "Release the lever", "Open the guide",
                   "Insert the material", "Align the rollers", "Close the guide",
                   "Lock the lever", "Verify the final position", "Extra item"][:count]
        rows = [{"citation_id": f"manual:p{6+i}:c{i}", "bubble_document_id": "manual",
                 "source_type": "document", "page_from": 6+i, "page_to": 6+i,
                 "chunk_index": i, "snippet": text, "chunk_full": text}
                for i, text in enumerate(actions)]
        points = [{"text": text, "citation_ids": [rows[i]["citation_id"]]}
                  for i, text in enumerate(actions)]
        if unknown_citation:
            points[1]["citation_ids"] = ["unadmitted-source"]
        retrieval = {"citations": rows, "assistant_core_contract": {
            "information_task": task, "required_answer_types": [],
            "required_facets": [], "fail_closed": True}}
        before = deepcopy(retrieval)
        selected = []

        def sanitize(items, **kwargs):
            selected.extend(items)
            return items

        runtime = replace(self.m._assistant_core_generation_runtime(),
            ASK_UI_MAX_POINTS=5, V13_MAX_EVIDENCE_ITEMS_ASK=12,
            preserve_procedure_points=protected,
            _v13_choose_ask_model=lambda *a, **k: ("offline-model", "low", ""),
            _v13_sources_block=lambda *a, **k: "Offline admitted manual operations",
            _v13_json_models=lambda *a, **k: (
                {"answer_status": "answered", "grounded_points": points}, "offline-model"),
            _sanitize_citations_for_response=sanitize,
            _build_rg_links=lambda *a, **k: [])
        result = generation.generate_ask_response(q="Describe the operating procedure.",
            company_id="offline-company", response_language="en", top_k=8,
            retrieval=retrieval, narrow_scope=True, debug=False, runtime=runtime)
        self.assertEqual(retrieval, before)
        for item in selected:
            self.assertIs(item, rows[rows.index(item)])
        self.assertEqual([c["citation_id"] for c in result["citations"]],
                         [c["citation_id"] for c in selected])
        return result, actions, selected

    def test_manual_procedure_keeps_six_to_eight_actions_and_their_citations(self):
        for task in ("procedure_full", "procedure_segment"):
            for count in (6, 8):
                with self.subTest(task=task, count=count):
                    result, actions, selected = self.generation_case(task=task, count=count)
                    self.assertEqual(result["status"], "answered")
                    self.assertEqual(len(selected), count)
                    self.assertEqual([line for line in result["answer"].splitlines() if line.strip()],
                        [f"{i}. {action}." for i, action in enumerate(actions, 1)])

    def test_legacy_and_nonprocedure_keep_existing_five_point_limit(self):
        self.assertFalse(self.m._assistant_core_generation_runtime().preserve_procedure_points)
        for kwargs in ({"protected": False}, {"task": "document_explanation"}):
            with self.subTest(kwargs=kwargs):
                result, actions, selected = self.generation_case(**kwargs)
                self.assertEqual(len(selected), 5)
                self.assertNotIn(actions[5], result["answer"])

    def test_manual_retention_stays_bounded_and_does_not_admit_unknown_citations(self):
        result, actions, selected = self.generation_case(count=9)
        self.assertEqual(len(selected), 8)
        self.assertNotIn(actions[8], result["answer"])
        result, actions, selected = self.generation_case(unknown_citation=True)
        self.assertEqual(len(selected), 7)
        self.assertNotIn(actions[1], result["answer"])
        self.assertIn(actions[7], result["answer"])
        self.assertNotIn("unadmitted-source", repr(result))

    def schedule_case(self, *, protected=True, task="procedure_full",
                      structured=False, outer_review=False):
        m = self.m
        request = m.AssistantCoreRequest(query="Describe the operating procedure.",
            requested_mode="ask", response_language="en", company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=8)
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,
            effective_mode="ask", confidence=.99, requested_mode_fit=True,
            evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task=task, required_answer_types=(m.REQ_ORDERED_ACTIONS,))
        provider_calls = []

        def generate(**kwargs):
            provider_calls.append("generic")
            request_completion.provider_timeout("ask_final_synthesis", 60)
            return {"ok": True, "status": "answered", "answer": "Offline manual answer"}

        def bundle(**kwargs):
            provider_calls.append("structured")
            request_completion.provider_timeout("ask_structured_synthesis", 60)
            return {"ok": True, "status": "answered", "answer": "Offline ProcedureBundle"}

        runtime = replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=denied if protected else None,
            _v13_generate_ask_response=generate, _v13_structured_ask=bundle,
            _assistant_core_verify_or_repair_answer=denied,
            _assistant_core_recover_ask_from_evidence=denied)
        retrieval = {"citations": [{"citation_id": "offline:source",
            "source_type": "step" if structured else "document"}], "plan": {}}
        owner = request_completion.RequestCompletion(
            runtime=m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id="offline-company"), started=time.monotonic())
        try:
            # Isolate scheduling from authority: this fixture asserts no admission
            # behavior. Production admission and provenance functions are untouched.
            with patch.object(execution, "_admit_input", side_effect=lambda req, data, dec, **kw: data):
                with request_completion.synthesis_review_reserve(required=outer_review):
                    result = execution.synthesize_ask(request, retrieval, decision, runtime=runtime)
                    request_completion.provider_timeout("ask_final_synthesis", 60)
            rows = list(owner.provider_schedules)
            self.assertEqual(owner.budget.llm_calls, 0)
            self.assertEqual(result["status"], "answered")
            return rows, provider_calls
        finally:
            owner.close()

    def test_protected_manual_procedure_reserves_repair_and_restores_outer_policy(self):
        for task in ("procedure_full", "procedure_segment"):
            with self.subTest(task=task):
                rows, calls = self.schedule_case(task=task)
                self.assertEqual(calls, ["generic"])
                self.assertEqual(rows[0]["review_reserve_seconds"], 10.5)
                self.assertEqual(rows[0]["publication_reserve_seconds"], 3)
                self.assertEqual(rows[1]["review_reserve_seconds"], 0)

    def test_successful_procedure_bundle_keeps_existing_review_bypass(self):
        rows, calls = self.schedule_case(structured=True)
        self.assertEqual(calls, ["structured"])
        self.assertEqual(rows[0]["review_reserve_seconds"], 0)

    def test_legacy_and_nonprocedure_review_policy_is_unchanged(self):
        for kwargs in ({"protected": False}, {"task": "document_explanation"}):
            with self.subTest(kwargs=kwargs):
                rows, calls = self.schedule_case(**kwargs)
                self.assertEqual(calls, ["generic"])
                self.assertEqual(rows[0]["review_reserve_seconds"], 0)
        rows, _ = self.schedule_case(task="document_explanation", outer_review=True)
        self.assertEqual(rows[0]["review_reserve_seconds"], 10.5)
        self.assertEqual(rows[1]["review_reserve_seconds"], 10.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
