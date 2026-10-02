"""Minimized ASK003 structure with synthetic text and no external I/O.

Admission/grounding callbacks are isolated: this tests the final numbering and
semantic coverage contract, not source authorization or live model quality.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, generation, procedure_review, validation


def denied(*a, **kw):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class MixedProcedureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        cls.guards.enter_context(patch.object(socket, "create_connection", denied))
        cls.guards.enter_context(patch.object(socket.socket, "connect", denied))
        cls.guards.enter_context(patch("requests.sessions.Session.request", denied))
        cls.guards.enter_context(patch("psycopg2.connect", denied))
        cls.m = importlib.import_module("main")
        cls.fixture = json.loads((Path(__file__).parent / "fixtures/ask_mixed_procedure_review.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def validate_fixture(self, *, source_ordered=False, outcome="rewrite", missing=None,
                         omit_safety_coverage=False, answer=None, source_numbers=None):
        m = self.m
        f = self.fixture
        answer = f["answer"] if answer is None else answer
        numbers = f["source_numbers"] if source_numbers is None else source_numbers
        rows = [{"citation_id": f"offline-step-{number}", "source_type": "step",
                 "fields": {"step_number": str(number), "description": "",
                            "notes": f["source_notes"].get(str(number), "")}}
                for number in numbers]
        rows.append({"citation_id": "offline-manual", "source_type": "document"})
        request = m.AssistantCoreRequest(query="Describe the threading operations and precautions.",
            requested_mode="ask", response_language="it", company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=8)
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,
            effective_mode="ask", confidence=.99, requested_mode_fit=True,
            evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task="procedure_segment",
            required_facets=tuple(f["verdict"]["covered_facets"]),
            required_answer_types=(m.REQ_ORDERED_ACTIONS, m.REQ_SAFETY_CONDITIONS))
        calls = []

        def verify(**kwargs):
            calls.append(kwargs)
            result = deepcopy(f["verdict"])
            result.update(outcome=outcome, answer=answer, citation_ids=[c["citation_id"] for c in rows])
            # This suite fixes a reviewed semantic verdict; source-quote/order
            # validation itself is exercised in test_operation_boundary.py.
            result["operation_boundary_validation"] = {
                "version": "ask-operation-boundary-v1", "complete": True,
                "answer_sha256": procedure_review.digest(answer.strip())}
            if missing:
                result["missing_facets"] = [missing]
            if omit_safety_coverage:
                result["covered_answer_types"] = [m.REQ_ORDERED_ACTIONS]
            if outcome == "unavailable":
                return {"outcome": "unavailable", "reason": "offline_timeout"}
            return result

        runtime = replace(m._assistant_core_ask_validation_runtime(),
            execution_runtime=replace(m._assistant_core_ask_execution_runtime(), evidence_admission=denied),
            final_contract_enabled=True,
            source_fields=lambda c: c.get("fields", {}), source_sections=lambda text: {},
            _assistant_core_candidate_source_type=lambda c: c["source_type"],
            _assistant_core_candidate_evidence_text=lambda c: answer,
            _assistant_core_recover_citations=lambda *a, **k: (rows, []),
            _assistant_core_redact_internal_text=lambda text: text,
            _assistant_core_filter_unsupported_claim_sentences=lambda text, sources: (text, []),
            _assistant_core_media_metadata_only=lambda text, *a: text,
            # Production's legacy policy skips ordinary procedure reviews. The
            # new protected manual/mixed rule must force one nevertheless.
            _assistant_core_should_semantic_verify_answer=lambda d: False,
            _assistant_core_verify_or_repair_answer=verify,
            _assistant_core_answer_contract_check=lambda **kw: {"passed": True, "reason": "offline_deterministic"},
            _sanitize_citations_for_response=lambda items, **kw: list(items),
            _build_rg_links=lambda *a, **kw: [], _v13_current_budget=lambda: None)
        response = {"ok": True, "status": "answered", "answer": answer,
                    "citations": rows, "meta": {}}
        if source_ordered:
            response["_assistant_core_source_ordered_procedure"] = True
        with patch.object(validation, "_inputs", side_effect=lambda r, q, t, d, **kw: (r, t)), \
             patch.object(validation, "_collection", side_effect=lambda q, items, d, **kw: items), \
             patch.object(validation, "_output", side_effect=lambda r, *a, **kw: r):
            result = validation.validate_response(response, request, {"citations": rows}, decision, runtime=runtime)
        return result, calls

    def test_live_mixed_answer_is_not_forced_to_sparse_retrieved_step_numbers(self):
        result, calls = self.validate_fixture()
        contract = result["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["complete"])
        structure = contract["procedure_structure"]
        self.assertFalse(structure["source_sequence_required"])
        self.assertEqual(structure["source_numbers"], [1, 5, 6, 7, 8, 9])
        self.assertEqual(structure["visible_numbers"], [1, 2, 3, 4, 5, 6, 7])
        self.assertIsNone(structure["sequence_complete"])
        self.assertEqual(result["answer"], self.fixture["answer"])

    def test_explicit_bundle_still_rejects_missing_or_reordered_source_steps(self):
        for answer in ("1. Isolate.\n2. Open.\n3. Close.", "5. Open.\n1. Isolate.\n9. Close."):
            with self.subTest(answer=answer):
                result, _ = self.validate_fixture(source_ordered=True, answer=answer)
                contract = result["meta"]["assistant_core_validation"]["answer_contract"]
                self.assertFalse(contract["passed"])
                self.assertEqual(contract["reason"], "source_step_sequence_or_notes_incomplete")

    def test_explicit_bundle_accepts_original_sparse_numbers(self):
        answer = "\n".join(f"{n}. Documented operation." for n in self.fixture["source_numbers"])
        result, _ = self.validate_fixture(source_ordered=True, answer=answer)
        contract = result["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["procedure_structure"]["sequence_complete"])
        self.assertNotIn("_assistant_core_source_ordered_procedure", result)

    def test_explicit_bundle_cannot_fall_back_when_source_structure_is_unusable(self):
        for numbers in ([1, 1], [], ["not-a-step-number"]):
            with self.subTest(numbers=numbers):
                result, calls = self.validate_fixture(source_ordered=True, source_numbers=numbers,
                                                       answer="1. Isolate.")
                contract = result["meta"]["assistant_core_validation"]["answer_contract"]
                self.assertEqual(calls, [])
                self.assertFalse(contract["passed"])
                self.assertEqual(contract["reason"], "source_step_structure_unverifiable")

    def test_manual_mixed_answer_requires_complete_order_and_safety_review(self):
        for kwargs in ({"outcome": "unavailable"}, {"missing": "operational order"},
                       {"missing": "safety precautions"}, {"omit_safety_coverage": True}):
            with self.subTest(kwargs=kwargs):
                result, calls = self.validate_fixture(**kwargs)
                self.assertEqual(len(calls), 1)
                self.assertFalse(result["meta"]["assistant_core_validation"]["answer_contract"]["passed"])

    def test_manual_with_no_step_records_still_gets_compact_independent_review(self):
        result, calls = self.validate_fixture(source_numbers=[])
        self.assertTrue(calls[0]["repair_context"]["procedure_structure"]["usable"])
        self.assertTrue(result["meta"]["assistant_core_validation"]["answer_contract"]["passed"])

    def test_mixed_source_precautions_are_preserved_after_reviewer_shortens_them(self):
        result, calls = self.validate_fixture(answer="1. Stop.\n2. Move material manually.")
        self.assertEqual(len(calls), 1)
        for note in self.fixture["source_notes"].values():
            self.assertIn(note, result["answer"])
        proof = result["meta"]["assistant_core_validation"]["answer_contract"]["source_preservation"]
        self.assertTrue(proof["all_present"])
        self.assertEqual(proof["notes"], 6)

    def test_model_and_retrieval_metadata_cannot_select_source_ordered_mode(self):
        marker = "_assistant_core_source_ordered_procedure"
        row = {"citation_id": "offline-manual", "source_type": "document",
               "snippet": "Isolate before opening.", "chunk_full": "Isolate before opening."}
        parsed = {"answer_status": "answered", marker: True,
                  "meta": {marker: True}, "grounded_points": [
                      {"text": "1. Isolate before opening.", "citation_ids": [row["citation_id"]], marker: True}]}
        runtime = replace(self.m._assistant_core_generation_runtime(),
            preserve_procedure_points=True,
            _v13_choose_ask_model=lambda *a, **kw: ("offline-model", "low", ""),
            _v13_sources_block=lambda *a, **kw: row["snippet"],
            _v13_json_models=lambda *a, **kw: (parsed, "offline-model"),
            _sanitize_citations_for_response=lambda items, **kw: list(items),
            _build_rg_links=lambda *a, **kw: [])
        result = generation.generate_ask_response(q="Describe the procedure.", company_id="offline-company",
            response_language="en", top_k=8, narrow_scope=True, debug=False,
            retrieval={"citations": [row], marker: True, "meta": {marker: True},
                       "assistant_core_contract": {"information_task": "procedure_full", "fail_closed": True}},
            runtime=runtime)
        self.assertEqual(result["status"], "answered")
        self.assertNotIn(marker, result)
        self.assertNotIn(marker, result.get("meta", {}))

    def test_actual_doubled_draft_prefix_is_removed_before_generic_rendering(self):
        raw = self.fixture["answer"].split("\n\nStep notes", 1)[0]
        blocks = procedure_review.block_layout(raw)["blocks"]
        points = [{"text": re.sub(r"^[1-9][0-9]*[.)] ", "", b["text"]).strip(),
                   "citation_ids": ["manual"]} for b in blocks]
        original = deepcopy(points)
        cleaned = generation._procedure_point_bodies(points)
        rendered, citations = self.m._render_grounded_answer_points(
            grounded_points=cleaned, citations=[{"citation_id": "manual"}], max_points=8)
        self.assertEqual(points, original)
        self.assertNotRegex(rendered, r"(?m)^([1-9][0-9]*)\. \1\.")
        self.assertEqual(len(procedure_review.block_layout(rendered)["blocks"]), 7)
        self.assertIn("Move the material manually", rendered)
        self.assertIn("stop button pressed", rendered)
        self.assertIn("do not apply excessive pressure", rendered)
        self.assertEqual(citations, [{"citation_id": "manual"}])

    def test_number_normalization_preserves_decimal_values_and_nested_operations(self):
        points = [{"text": "2.5 bar maximum.", "citation_ids": ["a"]},
                  {"text": "3. Isolate.\n4. Verify zero energy.", "citation_ids": ["b"]}]
        result = generation._procedure_point_bodies(points)
        self.assertEqual(result[0], points[0])
        self.assertEqual(result[1]["text"], "Isolate.\n4. Verify zero energy.")

    def test_generic_procedure_keeps_prepared_closure_source_after_default_eight(self):
        rows = [{"citation_id": f"offline:{i}", "source_type": "document", "bubble_document_id": "manual",
                 "snippet": f"Operation {i}.", "chunk_full": f"Operation {i}."} for i in range(12)]
        rows[-1]["chunk_full"] = "Restore the clamp after inserting the material."
        calls = []
        def provider(messages, **kw):
            calls.append(messages)
            return {"answer_status": "answered", "grounded_points": [
                {"text": "Insert material, then restore the clamp.", "citation_ids": ["offline:11"]}]}, "offline-model"
        runtime = replace(self.m._assistant_core_generation_runtime(),
            V13_MAX_EVIDENCE_ITEMS_ASK=8, preserve_procedure_points=True,
            _v13_choose_ask_model=lambda *a, **kw: ("offline-model", "low", ""),
            _v13_json_models=provider, _sanitize_citations_for_response=lambda items, **kw: list(items),
            _build_rg_links=lambda *a, **kw: [])
        result = generation.generate_ask_response(q="Describe the procedure.", company_id="offline-company",
            response_language="en", top_k=8, narrow_scope=True, debug=False,
            retrieval={"citations": rows, "assistant_core_contract": {
                "information_task": "procedure_segment", "fail_closed": True}}, runtime=runtime)
        self.assertEqual(len(calls), 1)
        self.assertIn(rows[-1]["chunk_full"], calls[0][1]["content"])
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["citation_id"], "offline:11")

    def test_existing_reviewer_repairs_closure_with_one_call_and_bound_audit(self):
        m = self.m
        draft = "1. Open the guide.\n2. Insert material."
        closure = "Close the guide before feeding."
        rows = [{"citation_id": "manual:one", "source_type": "document", "bubble_document_id": "manual",
                 "chunk_full": draft + "\n" + closure}]
        request = m.AssistantCoreRequest(query="Describe the operation.", requested_mode="ask",
            response_language="en", company_id="offline-company", machine_id="offline-machine", ai_scope="machine_all", top_k=8)
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE, effective_mode="ask", confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task="procedure_segment", required_answer_types=(m.REQ_ORDERED_ACTIONS,))
        calls = []
        def provider(messages, **kw):
            calls.append(kw)
            return {"outcome": "rewrite", "reply_mode": "replace", "answer": draft + "\n3. " + closure,
                "edits": [], "covered_facets": [], "missing_facets": [], "covered_answer_types": [m.REQ_ORDERED_ACTIONS],
                "missing_answer_types": [], "missing_list_items": [], "citation_ids": ["manual:one"],
                "operation_boundary": {"complete": True, "checked_blocks": ["step_1", "step_2"], "changes": [{
                    "block_id": "step_1", "action_quote": "Open the guide.", "resolution": "restored",
                    "closing_source_citation_id": "manual:one", "closing_source_quote": closure,
                    "closing_answer_quote": closure}]}}, "offline-model"
        runtime = replace(m._assistant_core_ask_execution_runtime(), evidence_admission=denied,
            _v13_json_models=provider, _v13_current_budget=lambda: SimpleNamespace(
                llm_calls=0, max_llm_calls=3, remaining=lambda: 30))
        observation = procedure_review.observe_structure(draft, [], fields=lambda c: {}, notes=(),
            notes_present=lambda *a: True, source_ordered=False)
        with patch.object(execution, "_admit_input", side_effect=lambda q, data, d, **kw: data):
            result = execution.verify_or_repair_answer(request=request, decision=decision, answer=draft,
                candidates=rows, repair_context={"procedure_structure": observation}, runtime=runtime)
        self.assertEqual(len(calls), 1)
        self.assertIn("operation_boundary", calls[0]["json_schema"]["schema"]["required"])
        self.assertTrue(result["operation_boundary_validation"]["complete"])
        self.assertEqual(result["answer"], draft + "\n3. " + closure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
