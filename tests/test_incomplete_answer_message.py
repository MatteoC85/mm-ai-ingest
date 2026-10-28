"""A failed answer with admitted material is not a missing-source diagnosis."""
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
from machinemind.ask import validation


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class IncompleteAnswerMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {"MM_USAGE_ENFORCEMENT": "off",
            "AI_INTERNAL_SECRET": "offline-fixture-only", "OPENAI_API_KEY": "offline-unused",
            "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guards.enter_context(patch(name, denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def validate(self, *, language="it", sources=True, code="INCOMPLETE_ANSWER_CONTRACT"):
        m = self.m
        request = m.AssistantCoreRequest(query="Describe the operation.", requested_mode="ask",
            response_language=language, company_id="offline-company", machine_id="offline-machine",
            ai_scope="machine_all", top_k=8)
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE, effective_mode="ask", confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task=m.INFO_PROCEDURE_SEGMENT,
            required_answer_types=(m.REQ_ORDERED_ACTIONS, m.REQ_SAFETY_CONDITIONS))
        rows = [{"citation_id": "offline-source:p1:c1", "source_type": "document",
                 "chunk_full": "Admitted material alone does not prove a complete answer."}] if sources else []
        old = "REJECTED_PREVIOUS_DRAFT"
        response = {"ok": True, "status": "no_sources", "result_code": code, "answer": old,
                    "answer_html": "<p>" + old + "</p>",
                    "_assistant_ui_model": {"kind": "procedure", "title": old, "steps": []},
                    "citations": [], "rg_links": [],
                    "meta": {"cacheable": False, "semantic_cacheable": False,
                             "assistant_core_repair": {"attempted": True, "completed": False,
                                 "first_answer_contract": {"passed": False, "complete": False,
                                                           "reason": "semantic_contract_rejected_grounded_answer"}}}}
        snapshot = deepcopy(response)
        runtime = replace(m._assistant_core_ask_validation_runtime(),
            _assistant_core_candidate_evidence_text=lambda row: row["chunk_full"],
            _assistant_core_media_metadata_only=lambda value, *args: value,
            _assistant_core_redact_internal_text=lambda value: value,
            _assistant_core_verify_or_repair_answer=denied,
            _assistant_core_recover_citations=denied)
        with patch.object(validation, "_inputs", side_effect=lambda r,q,t,d,**kw: (r,t)), \
             patch.object(validation, "_output", side_effect=lambda r,*args,**kw: r):
            out = validation.validate_response(response, request, {"citations": rows}, decision, runtime=runtime)
        self.assertEqual(response, snapshot)
        return out, old

    def test_admitted_material_changes_only_failure_explanation_it_en(self):
        for language, expected in (("it", "Ho trovato fonti autorizzate"), ("en", "I found authorized sources")):
            with self.subTest(language=language):
                out, old = self.validate(language=language)
                self.assertTrue(out["answer"].startswith(expected))
                self.assertNotIn(old, out["answer"])
                self.assertEqual(out["result_code"], "INCOMPLETE_ANSWER_CONTRACT")
                self.assertEqual(out["status"], "no_sources")
                self.assertTrue(out["ok"])
                self.assertEqual(out["citations"], [])
                self.assertEqual(out["rg_links"], [])
                self.assertFalse(out["meta"]["cacheable"])
                self.assertFalse(out["meta"]["semantic_cacheable"])
                self.assertFalse(out["meta"]["assistant_core_validation"]["answer_contract"]["passed"])
                self.assertNotIn("answer_html", out)
                self.assertNotIn("_assistant_ui_model", out)

    def test_safe_html_is_rebuilt_from_failure_message_without_previous_draft(self):
        out, old = self.validate()
        rendered = self.m._assistant_ui_finalize_response(out, language="it")
        self.assertIn("Ho trovato fonti autorizzate", rendered["answer_html"])
        self.assertNotIn(old, rendered["answer_html"])
        self.assertNotIn(old, rendered["answer"])
        self.assertEqual(rendered["status"], "no_sources")
        self.assertEqual(rendered["result_code"], "INCOMPLETE_ANSWER_CONTRACT")

    def test_actual_absent_material_keeps_previous_missing_source_message(self):
        out, old = self.validate(sources=False)
        self.assertEqual(out["answer"], old)
        self.assertEqual(out["status"], "no_sources")
        self.assertEqual(out["result_code"], "INCOMPLETE_ANSWER_CONTRACT")

    def test_no_machine_evidence_code_is_not_reinterpreted_as_completion_failure(self):
        out, old = self.validate(code="NO_MACHINE_EVIDENCE")
        self.assertEqual(out["answer"], old)
        self.assertEqual(out["status"], "no_sources")
        self.assertEqual(out["result_code"], "NO_MACHINE_EVIDENCE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
