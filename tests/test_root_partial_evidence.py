"""Scoped Root admission to existing review; no provider or database I/O."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_core_v2 import AssistantCoreHooks, AssistantCoreV2
from machinemind.infrastructure import request_budget as native


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_TEST_EXTERNAL_IO_DENIED")


class RootPartialEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-root-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        cls.guards.enter_context(patch.object(socket, "create_connection", denied))
        cls.guards.enter_context(patch("requests.sessions.Session.request", denied))
        cls.guards.enter_context(patch("psycopg2.connect", denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def setUp(self):
        m = self.m
        self.request = m.AssistantCoreRequest(query="Pump delivery decreased under the same operating conditions.",
            requested_mode="root_cause", response_language="en", company_id="fixture-company",
            machine_id="fixture-machine", ai_scope="machine_all", top_k=8, debug=True)
        self.facets = ("Interpretazione complessiva dei riscontri disponibili",
                       "Condizioni preliminari alla sostituzione del componente")
        self.decision = m.AssistantCoreDecision(request_kind=m.KIND_FAULT_DIAGNOSTIC,
            effective_mode="root_cause", confidence=0.98, requested_mode_fit=True,
            evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="machine_sources_required",
            information_task=m.INFO_FAULT_DIAGNOSTIC,
            required_answer_types=(m.REQ_DIAGNOSTIC_CAUSES, m.REQ_CHECKLIST, m.REQ_EXPLANATION),
            required_facets=self.facets,
            facet_queries=tuple(m.AssistantCoreFacetQuery(facet=f) for f in self.facets),
            diagnostic_observables=("Pump delivery decreased",), diagnostic_subsystems=("pump",),
            relevant_evidence_ids=("fixture-manual:p7:c1",))
        texts = [
            "A clogged discharge filter reduces pump delivery. Record the differential pressure already available. Do not start a new cycle to obtain a reading.",
            "An air leak in the suction line reduces pump delivery. Qualified personnel must isolate the pump before inspecting the suction line seal."]
        self.rows = [{"citation_id": f"fixture-manual:p7:c{i}", "bubble_document_id": "fixture-manual",
            "company_id": self.request.company_id, "machine_id": self.request.machine_id,
            "source_type": "document", "page_from": 7, "page_to": 7, "chunk_index": i,
            "snippet": text, "chunk_full": text, "similarity": 0.8, "semantic_similarity": 0.8,
            "retrieval_score": 0.8, "exact_machine_scope": True}
            for i, text in enumerate(texts, 1)]
        self.original = deepcopy(self.rows)
        self.assurance = patch.object(m, "_assistant_core_root_diagnostic_evidence_assurance",
            lambda request, retrieval, decision: retrieval)
        self.assurance.start()
        self.addCleanup(self.assurance.stop)

    def prepare(self, *, request=None, decision=None, rows=None):
        return self.m._assistant_core_prepare_evidence(
            request or self.request, {"candidates": deepcopy(self.rows if rows is None else rows)},
            decision or self.decision)

    def test_viable_cross_language_sources_reach_review_without_invented_facet_coverage(self):
        prepared = self.prepare()
        self.assertTrue(prepared["supported"])
        retrieval = prepared["retrieval"]
        admission = retrieval["assistant_core_decision"]
        self.assertEqual(admission["facet_evidence_coverage"], 0)
        self.assertFalse(admission["facet_threshold_pass"])
        self.assertFalse(admission["strict_supported"])
        self.assertEqual(admission["covered_facets"], [])
        partial = admission["root_partial_evidence_admission"]
        self.assertTrue(partial["requires_semantic_review"])
        self.assertFalse(partial["coverage_complete"])
        self.assertEqual(partial["missing_facets"], list(self.facets))
        self.assertEqual(retrieval["assistant_core_contract"]["root_partial_evidence_admission"], partial)
        self.assertEqual(retrieval["assistant_core_contract"]["required_facets"], list(self.facets))
        self.assertTrue(retrieval["citations"])
        original = {row["citation_id"]: row for row in self.original}
        for row in retrieval["citations"]:
            for key in ("company_id", "machine_id", "bubble_document_id", "page_from", "page_to", "chunk_full"):
                self.assertEqual(row[key], original[row["citation_id"]][key])
            self.assertEqual(row["assistant_core_missing_facets"], list(self.facets))
        self.assertEqual(self.rows, self.original)

    def test_strict_supported_path_is_unchanged(self):
        decision = replace(self.decision, required_facets=(), facet_queries=())
        result = self.prepare(decision=decision)
        self.assertTrue(result["supported"])
        self.assertTrue(result["retrieval"]["assistant_core_decision"]["strict_supported"])
        self.assertNotIn("root_partial_evidence_admission", result["retrieval"]["assistant_core_decision"])

    def test_no_causal_sources_or_empty_evidence_still_fail(self):
        for rows in ([],
                     [{**self.rows[0], "chunk_full": "General cleaning guidance only.",
                       "snippet": "General cleaning guidance only.", "similarity": 0.01,
                       "semantic_similarity": 0.01, "retrieval_score": 0.01}]):
            with self.subTest(rows=len(rows)):
                self.assertFalse(self.prepare(rows=rows)["supported"])

    def test_current_rescoring_hard_exclusions_are_not_rescued(self):
        rescore = self.m._v13_rescore_root_candidates
        def excluded(*args, **kwargs):
            # Rescoring recomputes this flag; exercise the current exclusion,
            # rather than a stale incoming annotation that it may replace.
            return [{**row, "hard_excluded": True} for row in rescore(*args, **kwargs)]
        with patch.object(self.m, "_v13_rescore_root_candidates", excluded):
            self.assertFalse(self.prepare()["supported"])

    def test_partial_and_refine_router_states_still_require_independent_viability(self):
        for state in (self.m.EVIDENCE_PARTIAL, self.m.EVIDENCE_REFINE):
            with self.subTest(state=state):
                decision = replace(self.decision, evidence_state=state)
                self.assertTrue(self.prepare(decision=decision)["supported"])
                self.assertFalse(self.prepare(decision=decision, rows=[])["supported"])

    def test_degraded_and_unsupported_router_are_not_rescued(self):
        for changes in ({"degraded": True}, {"evidence_state": self.m.EVIDENCE_UNSUPPORTED}):
            with self.subTest(changes=changes):
                self.assertFalse(self.prepare(decision=replace(self.decision, **changes))["supported"])

    def test_general_technical_deterministic_veto_is_not_a_facet_only_failure(self):
        decision = replace(self.decision, request_kind=self.m.KIND_GENERAL_TECHNICAL,
            evidence_policy=self.m.POLICY_GENERAL_ALLOWED)
        signals = {"top_similarity": 0.8, "top_overlap": 0.1, "exact_code_count": 1}
        with patch.object(self.m, "_v13_deterministic_evidence_state", return_value=("borderline", signals)):
            result = self.prepare(decision=decision)
        self.assertFalse(result["supported"])
        admission = result["retrieval"]["assistant_core_decision"]
        self.assertGreater(admission["root_viable_count"], 0)
        self.assertEqual(admission["facet_evidence_coverage"], 0)
        self.assertNotIn("root_partial_evidence_admission", admission)

    def test_missing_required_source_type_is_not_rescued(self):
        decision = replace(self.decision, preferred_source_types=("procedure",), source_type_policy="require")
        self.assertFalse(self.prepare(decision=decision)["supported"])
        decision = replace(decision, preferred_source_types=("document",))
        self.assertTrue(self.prepare(decision=decision)["supported"])

    def test_missing_numeric_interface_and_sequence_contracts_are_not_rescued(self):
        for requirement, helper, value in (
            (self.m.REQ_NUMERIC_VALUE, "_assistant_core_numeric_signal", {"has_number": False}),
            (self.m.REQ_INTERFACE_LOCATIONS, "_assistant_core_interface_navigation_signal", False),
            (self.m.REQ_STATE_SEQUENCE, "_assistant_core_sequence_signal", False)):
            with self.subTest(requirement=requirement), patch.object(self.m, helper, return_value=value):
                decision = replace(self.decision, required_answer_types=(*self.decision.required_answer_types, requirement))
                self.assertFalse(self.prepare(decision=decision)["supported"])

    def test_missing_ordered_action_support_is_not_rescued(self):
        decision = replace(self.decision, required_answer_types=(*self.decision.required_answer_types, self.m.REQ_ORDERED_ACTIONS))
        signals = {"top_similarity": 0.1, "top_overlap": 0.01, "exact_code_count": 1}
        with patch.object(self.m, "_v13_deterministic_evidence_state", return_value=("borderline", signals)):
            self.assertFalse(self.prepare(decision=decision)["supported"])

    def test_empty_diversified_pool_is_not_rescued(self):
        from types import SimpleNamespace
        with patch.object(self.m, "_assistant_core_root_source_selection",
                          return_value=SimpleNamespace(candidates=(), summary={"selected_count": 0})):
            self.assertFalse(self.prepare()["supported"])

    def test_ask_smart_and_rerouted_pairs_do_not_use_partial_root_admission(self):
        for requested, effective in (("ask", "ask"), ("smart_diagnostic", "smart_diagnostic"),
                                     ("ask", "root_cause"), ("root_cause", "ask")):
            with self.subTest(requested=requested, effective=effective):
                result = self.prepare(request=replace(self.request, requested_mode=requested),
                    decision=replace(self.decision, effective_mode=effective))
                self.assertNotIn("root_partial_evidence_admission", result["retrieval"].get("assistant_core_decision", {}))
                if effective != "smart_diagnostic":
                    self.assertFalse(result["supported"])

    def run_core(self, *, review="reject", rows=None):
        m = self.m
        events = []
        captured = {}
        budget = native._V13RequestBudget("root_cause")
        def producer(**kwargs):
            events.append("producer")
            captured["retrieval"] = kwargs["retrieval"]
            return {"ok": True, "status": "answered", "possible_causes": [{
                "cause": "Possible discharge filter restriction",
                "why": "Restriction could explain the reduced delivery; the filter condition is unknown.",
                "checks": ["Record the differential pressure already available."]}], "meta": {}}
        def records(request, candidates):
            return [{"citation_id": c["citation_id"], "text": c["chunk_full"],
                "source_type": c["source_type"], "page_from": c["page_from"], "page_to": c["page_to"],
                "context_pages": []} for c in candidates]
        def reviewer(messages, **kwargs):
            events.append("review")
            self.assertEqual(kwargs["purpose"], "assistant_core_root_cause_adjudicator")
            if review == "error":
                raise RuntimeError("offline reviewer failure")
            # A semantic rejection is a provider fixture, never a fabricated acceptance.
            return {"decisions": [{"proposal_index": 0, "verdict": "reject", "reason": "unsupported",
                "selected_check_indices": [], "blocking_checks": [0], "safe_check_subset": False,
                "selected_why_indices": [], "omitted_why_indices": [0], "safe_why_subset": False,
                "proofs": [], "note": "Applicable causal relation not established."}],
                "ranked_proposal_indices": []}, "offline-review-model"
        def prepare(request, retrieval, decision):
            result = m._assistant_core_prepare_evidence(request, retrieval, decision)
            captured["prepared"] = result
            return result
        def no_evidence(*args):
            events.append("no_evidence")
            return m._assistant_core_build_no_evidence(*args)
        hooks = AssistantCoreHooks(retrieve_neutral=lambda request: {"candidates": deepcopy(self.rows if rows is None else rows)},
            route_semantically=lambda request, retrieval: asdict(self.decision), prepare_evidence=prepare,
            synthesize_root_cause=m._assistant_core_synthesize_root_cause,
            synthesize_ask=denied, synthesize_general=denied, build_no_evidence=no_evidence,
            build_clarification=denied, build_out_of_scope=denied, build_safety_refusal=denied)
        with ExitStack() as stack:
            stack.enter_context(patch.object(m, "_v13_current_budget", return_value=budget))
            stack.enter_context(patch.object(m, "_v13_generate_root_cause_response", producer))
            stack.enter_context(patch.object(m, "_assistant_core_root_applicability_records", records))
            stack.enter_context(patch.object(m, "_v13_json_models", reviewer))
            result = AssistantCoreV2(hooks).run(self.request)
        return result, events, captured

    def test_actual_core_and_main_route_rejected_review_cannot_restore_draft(self):
        result, events, captured = self.run_core()
        self.assertEqual(events, ["producer", "review"])
        self.assertTrue(captured["prepared"]["supported"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["result_code"], "TECHNICAL_ERROR")
        self.assertEqual(result["possible_causes"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["meta"]["root_response_quality"]["outcome"], "review_rejected_proposals")

    def test_actual_core_review_failure_is_not_an_acceptance_fallback(self):
        result, events, _ = self.run_core(review="error")
        self.assertEqual(events, ["producer", "review"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["possible_causes"], [])
        self.assertFalse(result["meta"]["cacheable"])

    def test_actual_review_keeps_company_machine_and_document_scope_guards(self):
        for field, value in (("company_id", "other-company"), ("machine_id", "other-machine"),
                             ("bubble_document_id", "other-document")):
            with self.subTest(field=field):
                rows = deepcopy(self.rows)
                for row in rows:
                    row[field] = value
                original_request = self.request
                if field == "bubble_document_id":
                    self.request = replace(self.request, ai_scope="document_ids",
                        metadata={"document_ids": ["fixture-manual"]})
                try:
                    result, events, _ = self.run_core(rows=rows)
                    self.assertEqual(events, ["producer"])
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["possible_causes"], [])
                    self.assertIn("scope_mismatch", result["meta"]["root_review_execution"]["validation_error"])
                finally:
                    self.request = original_request


if __name__ == "__main__":
    unittest.main(verbosity=2)
