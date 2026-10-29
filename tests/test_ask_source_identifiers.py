"""Literal identifier prompt and grounding regressions; no provider or external I/O.

The writer/verifier replies are fixtures. These tests prove the real prompt
contract and unchanged source/hash gates, not compliance by a live model.
Authority and file-link adapters are isolated because no admission code changes.
"""
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
from machinemind.ask import execution, generation, validation


def deny(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class AskSourceIdentifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, deny))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def run_pipeline(self, reviewed, *, language="it", source_code="QTR-7"):
        m = self.m
        source = (f"Documento del costruttore: grado {source_code}. "
                  "Frequenza documentata: ogni 600 ore. Non sono elencate alternative equivalenti.")
        row = {"citation_id": "offline:p1:c0", "bubble_document_id": "offline",
               "source_type": "document", "page_from": 1, "page_to": 1,
               "chunk_full": source, "snippet": source}
        request = m.AssistantCoreRequest(query="Quali grado e frequenza sono documentati?",
            requested_mode="ask", response_language=language, company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=6)
        decision = m.AssistantCoreDecision(request_kind="general_technical", effective_mode="ask",
            confidence=.99, requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED,
            evidence_policy="machine_sources_required", information_task="document_explanation",
            required_answer_types=("explanation",), required_facets=("grado e frequenza",))
        calls = []

        def writer(messages, **kwargs):
            calls.append(("writer", deepcopy(messages), deepcopy(kwargs)))
            return {"answer_status": "answered", "grounded_points": [{
                "text": f"Il documento indica il grado {source_code} e la frequenza di 600 ore.",
                "citation_ids": [row["citation_id"]]}]}, "offline-model"

        def reviewer(messages, **kwargs):
            calls.append(("reviewer", deepcopy(messages), deepcopy(kwargs)))
            return {"outcome": "rewrite", "answer": reviewed, "reason": "fixture",
                "covered_facets": list(decision.required_facets), "missing_facets": [],
                "covered_answer_types": ["explanation"], "missing_answer_types": [],
                "expected_list_items": [], "covered_list_items": [], "missing_list_items": [],
                "enumeration_requested": False, "citation_ids": [row["citation_id"]]}, "offline-reviewer"

        gruntime = replace(m._assistant_core_generation_runtime(),
            _v13_json_models=writer,
            _v13_choose_ask_model=lambda *a, **kw:(m.V13_FAST_MODEL, "low", ""),
            _sanitize_citations_for_response=lambda rows, **kw:list(rows),
            _build_rg_links=lambda *a, **kw:[])
        eruntime = replace(m._assistant_core_ask_execution_runtime(), evidence_admission=deny,
            _v13_json_models=reviewer, _v13_current_budget=lambda:m._assistant_core_new_budget("ask"))
        vruntime = replace(m._assistant_core_ask_validation_runtime(), execution_runtime=eruntime,
            final_contract_enabled=True, source_fields=m._procedure_ui_fields,
            source_sections=m._procedure_ui_sections,
            _assistant_core_recover_citations=lambda response, **kw:(response["citations"], []),
            _assistant_core_verify_or_repair_answer=lambda **kw:execution.verify_or_repair_answer(**kw, runtime=eruntime),
            _sanitize_citations_for_response=lambda rows, **kw:list(rows),
            _build_rg_links=lambda *a, **kw:[], _v13_current_budget=lambda:None)
        retrieval = {"citations": [row], "candidates": [row], "plan": {},
            "assistant_core_contract": {"fail_closed": True, "information_task": decision.information_task,
                "required_answer_types": list(decision.required_answer_types),
                "required_facets": list(decision.required_facets)}}
        generated = generation.generate_ask_response(q=request.query, company_id=request.company_id,
            response_language=language, top_k=6, retrieval=retrieval, narrow_scope=False,
            debug=False, runtime=gruntime)
        # A failed final gate must remain visible; this fixture never dispatches
        # a repair. The unchanged production retry policy has separate tests.
        generated["_assistant_core_repair_attempted"] = True
        with patch.object(execution, "_admit_input", side_effect=lambda q,d,s,**kw:d), \
             patch.object(validation, "_inputs", side_effect=lambda r,q,t,d,**kw:(r,t)), \
             patch.object(validation, "_collection", side_effect=lambda q,items,d,**kw:items), \
             patch.object(validation, "_output", side_effect=lambda r,*a,**kw:r):
            result = validation.validate_response(generated, request, retrieval, decision, runtime=vruntime)
        return result, calls, source

    def test_source_exact_identifier_survives_real_writer_review_and_final_gate(self):
        text = "Il documento indica il grado QTR-7.\nLa frequenza documentata e' di 600 ore."
        final, calls, source = self.run_pipeline(text)
        self.assertEqual([c[0] for c in calls], ["writer", "reviewer"])
        self.assertEqual(final["status"], "answered", final)
        contract = final["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertTrue(contract["passed"], contract)
        self.assertTrue(contract["semantic_answer_bound"])
        self.assertIn("QTR-7", final["answer"])
        self.assertEqual([c["citation_id"] for c in final["citations"]], ["offline:p1:c0"])
        for kind, messages, _ in calls:
            self.assertIn(source, messages[1]["content"], kind)
            prompt = messages[0]["content"]
            self.assertIn("letter order and digits", prompt)
            self.assertIn("apparent typo", prompt)
            self.assertIn("prior knowledge", prompt)
            self.assertIn("unsupported optional identifier", prompt)
            self.assertIn("mandatory fact", prompt)

    def test_source_spelling_is_preserved_when_explanation_is_translated(self):
        final, calls, _ = self.run_pipeline(
            "The documented designation is QTR-7; verify this source spelling if uncertain. "
            "The documented interval is 600 hours.", language="en")
        self.assertEqual(final["status"], "answered", final)
        self.assertIn("QTR-7", final["answer"])
        self.assertTrue(final["meta"]["assistant_core_validation"]["answer_contract"]["complete"])
        for _, messages, _ in calls:
            self.assertIn("RESPONSE_LANGUAGE: en", messages[1]["content"])

    def test_silent_letter_reordering_or_digit_change_still_fails_closed(self):
        for bad in ("QRT-7", "QTR-8"):
            with self.subTest(identifier=bad):
                final, calls, _ = self.run_pipeline(
                    f"Il documento indica il grado {bad}.\nLa frequenza documentata e' di 600 ore.")
                self.assertEqual(len(calls), 2)
                self.assertEqual(final["status"], "no_sources", final)
                meta = final["meta"]["assistant_core_validation"]
                self.assertIn(bad, meta["unsupported_numeric_or_code_claims_removed"])
                self.assertEqual(meta["answer_contract"]["reason"], "semantic_answer_changed_after_review")
                self.assertFalse(meta["answer_contract"]["semantic_answer_bound"])
                self.assertEqual(final["citations"], [])

    def test_unrelated_unsupported_quantity_still_fails_closed(self):
        final, _, _ = self.run_pipeline("Il documento indica il grado QTR-7.\nLa frequenza e' di 900 ore.")
        self.assertEqual(final["status"], "no_sources", final)
        contract = final["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertFalse(contract["passed"])
        self.assertEqual(contract["reason"], "semantic_answer_changed_after_review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
