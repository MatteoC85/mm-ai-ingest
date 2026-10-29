"""Conditions-only routing through real writer, UI and final review, offline.

Provider replies are fixtures, not a quality assertion about a live model.
The authority adapters are isolated; all network and database I/O is denied.
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
from machinemind.ask import execution, generation, procedure_review, request_scope, validation


def deny(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class AskConditionsPipelineTests(unittest.TestCase):
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

    def request_decision(self, language="en"):
        m = self.m
        query = ("Before installing a cartridge, which checks and safety conditions must I verify?"
                 if language == "en" else "Quali controlli e condizioni di sicurezza verifico prima di inserire la cartuccia?")
        request = m.AssistantCoreRequest(query=query, requested_mode="ask", response_language=language,
            company_id="offline-company", machine_id="offline-machine", ai_scope="machine_all", top_k=6)
        facets = tuple(m.AssistantCoreFacetQuery(facet=name, answer_type=kind, must_cover=True)
                       for name, kind in (("preparatory checks", "checklist"), ("safety", "safety_conditions")))
        decision = m.AssistantCoreDecision(request_kind="general_technical", effective_mode="ask",
            confidence=.99, requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED,
            evidence_policy="machine_sources_required", information_task="procedure_segment",
            procedure_scope="conditions_only", required_answer_types=("checklist", "safety_conditions"),
            required_facets=("preparatory checks", "safety"), facet_queries=facets)
        return request, decision

    def contract(self, decision):
        return dict(procedure_scope=decision.procedure_scope, information_task=decision.information_task,
            required_answer_types=decision.required_answer_types, facet_queries=decision.facet_queries)

    def test_profile_is_explicit_and_conservative_for_conflicting_or_legacy_shapes(self):
        _, decision = self.request_decision()
        self.assertTrue(request_scope.conditions_only_contract(**self.contract(decision)))
        cases = [dict(procedure_scope=value) for value in (None, "", "other", "execution_steps")]
        cases += [dict(information_task="procedure_full"), dict(facet_queries=[]),
                  dict(facet_queries=[{"must_cover": False, "answer_type": "checklist"}]),
                  dict(facet_queries=[{"must_cover": True, "answer_type": "unknown"}])]
        cases += [dict(required_answer_types=("checklist", item))
                  for item in ("ordered_actions", "state_sequence", "numeric_value")]
        cases += [dict(facet_queries=[{"must_cover": True, "answer_type": "ordered_actions"}])]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertFalse(request_scope.conditions_only_contract(**{**self.contract(decision), **changes}))

    def rows(self):
        return [
            {"citation_id": "step:prepare", "bubble_document_id": "step:prepare", "source_type": "step",
             "chunk_full": "SOURCE_TYPE: step\nSTEP_NUMBER: 1\nTITLE: Preparare in sicurezza\n"
                 "DESCRIPTION: ISTRUZIONE OPERATIVA: Arrestare il ciclo e applicare l'isolamento delle energie.\n"
                 "NOTA DI SICUREZZA: Utilizzare personale addestrato."},
            {"citation_id": "step:guide", "bubble_document_id": "step:guide", "source_type": "step",
             "chunk_full": "SOURCE_TYPE: step\nSTEP_NUMBER: 4\nTITLE: Verificare la guida\n"
                 "DESCRIPTION: ISTRUZIONE OPERATIVA: Verificare che la guida sia libera da ostacoli prima di inserire la cartuccia.\n"
                 "NOTA DI SICUREZZA: Tenere le mani lontane dalle parti mobili."},
            {"citation_id": "manual:transport", "bubble_document_id": "manual:transport", "source_type": "document",
             "chunk_full": "Trasporto dell'intera linea: imballare i componenti e utilizzare un carroponte."},
        ]

    def run_pipeline(self, language="en", reject=False, wrong_quote=False):
        m = self.m
        request, decision = self.request_decision(language)
        rows = self.rows()
        calls, captures = [], {}
        bodies = (["Before installation, verify that the cycle is stopped and energy is isolated. Use trained personnel.",
                   "Check that the guide is clear of obstructions. Keep hands away from moving parts."]
                  if language == "en" else
                  ["Prima dell'inserimento verificare che il ciclo sia arrestato e le energie isolate. Utilizzare personale addestrato.",
                   "Verificare che la guida sia libera da ostacoli. Tenere le mani lontane dalle parti mobili."])

        def writer(messages, **kwargs):
            calls.append(("writer", messages, kwargs))
            return {"answer_status": "answered", "grounded_points": [
                {"text": text, "citation_ids": [row["citation_id"]]} for row, text in zip(rows, bodies)]}, "offline-model"

        gruntime = replace(m._assistant_core_generation_runtime(), preserve_procedure_points=True,
            _v13_json_models=writer, _v13_choose_ask_model=lambda *a, **kw:(m.V13_FAST_MODEL, "low", ""),
            _sanitize_citations_for_response=lambda items, **kw:list(items), _build_rg_links=lambda *a, **kw:[],
            _finalize_ask_response_for_ui=lambda r, **kw:m._finalize_ask_response_for_ui(r, **kw, preserve_complete=True))

        def generate(**kw):
            captures["retrieval"] = deepcopy(kw["retrieval"])
            result = generation.generate_ask_response(**kw, runtime=gruntime)
            captures["generated"] = deepcopy(result)
            return result

        def reviewer(messages, **kwargs):
            calls.append(("reviewer", messages, kwargs))
            current = messages[1]["content"].split("CURRENT_ANSWER:\n", 1)[1].split("\n\nSOURCES:", 1)[0]
            layout = procedure_review.block_layout(current)
            boundary = {"complete": not reject, "checked_blocks": [b["block_id"] for b in layout["blocks"]], "changes": []}
            if wrong_quote:
                boundary["changes"] = [{"block_id": layout["blocks"][0]["block_id"],
                    "action_quote": "energy is isolated", "resolution": "documented_terminal_state",
                    "closing_answer_quote": "energy is isolated", "closing_source_citation_id": "step:prepare",
                    "closing_source_quote": "Arrestare il ciclo e applicare l'isolamento."}]
            return {"outcome": "no_sources" if reject else "pass", "reply_mode": "replace" if reject else "retain",
                "answer": "", "edits": [], "reason": "fixture safety missing" if reject else "fixture verified",
                "covered_facets": [] if reject else list(decision.required_facets),
                "missing_facets": ["safety"] if reject else [],
                "covered_answer_types": list(decision.required_answer_types), "missing_answer_types": [],
                "expected_list_items": [], "covered_list_items": [], "missing_list_items": [],
                "enumeration_requested": False, "citation_ids": [r["citation_id"] for r in rows[:2]],
                "operation_boundary": boundary}, "offline-reviewer"

        eruntime = replace(m._assistant_core_ask_execution_runtime(), evidence_admission=deny,
            _v13_structured_ask=deny, _v13_generate_ask_response=generate,
            _assistant_core_recover_ask_from_evidence=deny, _v13_json_models=reviewer,
            _v13_current_budget=lambda:m._assistant_core_new_budget('ask'))
        runtime = replace(m._assistant_core_ask_validation_runtime(), execution_runtime=eruntime,
            final_contract_enabled=True, source_fields=m._procedure_ui_fields, source_sections=m._procedure_ui_sections,
            _assistant_core_recover_citations=lambda response, **kw:(response["citations"], []),
            _assistant_core_verify_or_repair_answer=lambda **kw:execution.verify_or_repair_answer(**kw, runtime=eruntime),
            _sanitize_citations_for_response=lambda items, **kw:list(items), _build_rg_links=lambda *a, **kw:[],
            _v13_current_budget=lambda:None)
        retrieval = {"citations": rows, "candidates": rows, "plan": {}, "assistant_core_contract": {"fail_closed": True}}
        with patch.object(execution, "_admit_input", side_effect=lambda q, data, d, **kw:data), \
             patch.object(validation, "_inputs", side_effect=lambda r, q, t, d, **kw:(r, t)), \
             patch.object(validation, "_collection", side_effect=lambda q, items, d, **kw:items), \
             patch.object(validation, "_output", side_effect=lambda r, *a, **kw:r):
            generated = execution.synthesize_ask(request, retrieval, decision, runtime=eruntime)
            # Final validation failure must be observable, not scheduled as a new
            # provider call by this fixture. The ordinary one-repair policy stays.
            if reject or wrong_quote:
                generated["_assistant_core_repair_attempted"] = True
            final = validation.validate_response(generated, request, captures["retrieval"], decision, runtime=runtime)
        from assistant_core_v2 import _finish_ask_validation
        final = m._finalize_ask_response_for_ui(_finish_ask_validation(final), language=language, preserve_complete=True)
        return final, calls, captures

    def test_english_checks_use_real_writer_renderer_and_mixed_review(self):
        final, calls, captures = self.run_pipeline()
        self.assertEqual([c[0] for c in calls], ["writer", "reviewer"])
        self.assertEqual(final["status"], "answered", final)
        self.assertIn("guide is clear", final["answer"])
        self.assertNotIn("carroponte", final["answer"])
        self.assertNotIn("_assistant_core_source_ordered_procedure", captures["generated"])
        self.assertNotIn("_assistant_ui_model", final)
        self.assertFalse(any(k.startswith("_assistant_core_") for k in final))
        self.assertIn("guide is clear", final["answer_html"])
        contract = final["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertTrue(contract["passed"], contract)
        self.assertTrue(contract["semantic_verifier"]["operation_boundary_validation"]["complete"])
        self.assertFalse(contract["procedure_structure"]["source_sequence_required"])
        self.assertIn("Step notes and safety precautions", final["answer"])
        self.assertIn("Utilizzare personale addestrato", final["answer"])
        for kind, messages, _ in calls:
            self.assertIn("PROCEDURE_SCOPE: conditions_only", messages[1]["content"])
            required = messages[1]["content"].split("REQUIRED_ANSWER_TYPES: ", 1)[1].splitlines()[0]
            self.assertNotIn("ordered_actions", required, kind)

    def test_italian_checks_use_same_pipeline_without_extra_provider(self):
        final, calls, _ = self.run_pipeline("it")
        self.assertEqual([c[0] for c in calls], ["writer", "reviewer"])
        self.assertEqual(final["status"], "answered", final)
        self.assertIn("guida sia libera", final["answer"])
        self.assertNotIn("carroponte", final["answer"])

    def test_missing_safety_still_fails_closed(self):
        final, calls, _ = self.run_pipeline(reject=True)
        self.assertEqual(final["status"], "no_sources")
        self.assertEqual(final["result_code"], "NO_MACHINE_EVIDENCE")
        self.assertEqual(final["citations"], [])
        self.assertFalse(final["meta"]["assistant_core_validation"]["answer_contract"]["passed"])
        self.assertEqual(len(calls), 2)

    def test_execution_full_and_legacy_dispatch_still_use_structured_producer(self):
        request, base = self.request_decision()
        cases = [replace(base, procedure_scope=""), replace(base, procedure_scope="execution_steps"),
                 replace(base, information_task="procedure_full"),
                 replace(base, required_answer_types=("ordered_actions", "checklist", "safety_conditions")),
                 replace(base, required_answer_types=("checklist", "safety_conditions"),
                         facet_queries=base.facet_queries + (self.m.AssistantCoreFacetQuery(
                             facet="execute operation", answer_type="ordered_actions", must_cover=True),))]
        for decision in cases:
            with self.subTest(decision=decision):
                called = []
                def structured(**kwargs):
                    called.append(kwargs)
                    return {"ok": True, "status": "answered", "answer": "existing canonical producer"}
                runtime = replace(self.m._assistant_core_ask_execution_runtime(), evidence_admission=deny,
                    _v13_structured_ask=structured, _v13_generate_ask_response=deny)
                with patch.object(execution, "_admit_input", side_effect=lambda q, data, d, **kw:data):
                    result = execution.synthesize_ask(request, {"citations": self.rows(), "plan": {}},
                        decision, runtime=runtime)
                self.assertEqual(result["answer"], "existing canonical producer")
                self.assertEqual(len(called), 1)

    def test_inexact_source_closure_quote_still_fails_closed(self):
        final, calls, _ = self.run_pipeline(wrong_quote=True)
        self.assertEqual(final["status"], "no_sources")
        self.assertEqual(final["citations"], [])
        contract = final["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertFalse(contract["semantic_verifier"]["operation_boundary_validation"]["complete"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
