"""Offline real-Core tests for Smart's single semantic routing failure boundary."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_core_v2 import AssistantCoreHooks, AssistantCoreRequest, AssistantCoreV2
from machinemind.retrieval import smart_routing


class SmartRoutingTests(unittest.TestCase):
    def setUp(self):
        self.request = AssistantCoreRequest(
            query="Pressure drops intermittently after maintenance; the indicator was not observed.",
            requested_mode="smart_diagnostic", response_language="en",
            company_id="offline-company", machine_id="offline-machine", ai_scope="machine", top_k=8,
            allowed_effective_modes=("smart_diagnostic",),
        )
        self.retrieval = {"candidates": [{"citation_id": "offline-source", "text": "Pressure inspection."}]}

    def engine(self, route):
        self.calls = {name: Mock(name=name, side_effect=AssertionError("Unexpected downstream callback"))
                      for name in ("prepare_evidence", "synthesize_ask", "synthesize_root_cause",
                                   "synthesize_general", "build_no_evidence", "build_clarification",
                                   "build_out_of_scope", "build_safety_refusal", "refine_retrieval",
                                   "synthesize_smart_start")}
        return AssistantCoreV2(AssistantCoreHooks(
            retrieve_neutral=Mock(return_value=self.retrieval), route_semantically=route, **self.calls))

    def allow(self, name, value):
        self.calls[name].side_effect = None
        self.calls[name].return_value = value

    def decision(self, **overrides):
        return {"effective_mode": "smart_diagnostic", "request_kind": "guided_diagnostic",
                "confidence": .8, "evidence_state": "supported",
                "evidence_policy": "machine_sources_required", "information_task": "fault_diagnostic",
                "required_answer_types": ["diagnostic_causes", "checklist"],
                "dense_queries": ["pressure drops", "calo pressione"],
                "lexical_queries": ["pressure inspection", "verifica pressione"],
                "required_facets": ["pressure drop after maintenance"],
                "facet_queries": [{"facet": "pressure drop after maintenance", "answer_type": "diagnostic_causes",
                    "must_cover": True, "dense_queries": ["pressure drop", "calo pressione"],
                    "lexical_queries": ["pressure", "pressione"], "exact_terms": [],
                    "preferred_source_types": ["procedure", "ps"]}],
                "diagnostic_subsystems": ["pressure circuit"],
                "diagnostic_discriminants": ["after maintenance"],
                "missing_information": ["indicator state"], **overrides}

    def test_plan_preserves_configured_primary_and_explicit_effort_without_fallback(self):
        for primary in ("configured-primary", "another-configured-primary"):
            for override, expected in ((None, "low"), ("", "low"), ("  ", "low"), ("medium", "medium"), ("high", "high")):
                with self.subTest(primary=primary, override=override):
                    plan = smart_routing.attempt_plan(primary, effort_override=override)
                    self.assertEqual(plan["models"], [primary])
                    self.assertEqual(plan["effort"], expected)
                    self.assertEqual(plan["attempt_limit"], 1)
                    self.assertNotIn("timeout", plan)
                    self.assertNotIn("max_output_tokens", plan)
                    self.assertNotIn("cost_allowance", plan)

    def test_invalid_configuration_does_not_select_a_hidden_model(self):
        for primary in (None, "", "   ", [], 3):
            with self.subTest(primary=primary), self.assertRaises(ValueError):
                smart_routing.attempt_plan(primary)
        with self.assertRaises(ValueError):
            smart_routing.attempt_plan("configured-primary", effort_override={})

    def test_unavailable_router_blocks_every_later_callback_despite_candidates(self):
        for rows in ([], self.retrieval["candidates"], self.retrieval["candidates"] * 30):
            with self.subTest(candidate_count=len(rows)):
                self.retrieval = {"candidates": rows}
                engine = self.engine(Mock(side_effect=smart_routing.SmartRouterUnavailable()))
                self.allow("build_no_evidence", {"ok": False, "status": "error",
                    "error_code": smart_routing.FAILURE_CODE, "meta": {"cacheable": False}})
                result = engine.run(self.request)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error_code"], smart_routing.FAILURE_CODE)
                decision = self.calls["build_no_evidence"].call_args.args[1]
                self.assertTrue(decision.degraded)
                self.assertEqual(decision.degraded_reason, smart_routing.FAILURE_REASON)
                self.assertEqual(decision.evidence_state, "unsupported")
                self.assertEqual(decision.relevant_evidence_ids, ())
                self.assertEqual(decision.effective_mode, "smart_diagnostic")
                for name, callback in self.calls.items():
                    if name != "build_no_evidence":
                        callback.assert_not_called()

    def test_unavailable_reason_has_no_provider_error_input(self):
        self.assertEqual(str(smart_routing.SmartRouterUnavailable()), smart_routing.FAILURE_REASON)
        with self.assertRaises(TypeError):
            smart_routing.SmartRouterUnavailable("PRIVATE_PROVIDER_ERROR")

    def test_semantic_unsafe_and_out_of_scope_still_stop_before_evidence_or_generation(self):
        for kind, hook, status in (("unsafe_request", "build_safety_refusal", "safety_refusal"),
                                   ("out_of_scope", "build_out_of_scope", "out_of_scope")):
            with self.subTest(kind=kind):
                engine = self.engine(Mock(return_value=self.decision(request_kind=kind)))
                self.allow(hook, {"ok": False, "status": status})
                result = engine.run(self.request)
                self.assertEqual(result["status"], status)
                self.calls[hook].assert_called_once()
                for name, callback in self.calls.items():
                    if name != hook:
                        callback.assert_not_called()

    def test_success_retains_bilingual_facets_and_scope_through_refinement(self):
        engine = self.engine(Mock(return_value=self.decision()))
        self.allow("refine_retrieval", self.retrieval)
        self.allow("prepare_evidence", {"supported": True, "retrieval": self.retrieval})
        self.allow("synthesize_smart_start", {"ok": True, "status": "in_progress"})
        result = engine.run(self.request)
        self.assertTrue(result["ok"])
        passed_request, _, decision = self.calls["refine_retrieval"].call_args.args
        self.assertIs(passed_request, self.request)
        self.assertEqual(decision.dense_queries, ("pressure drops", "calo pressione"))
        self.assertEqual(decision.facet_queries[0].dense_queries, ("pressure drop", "calo pressione"))
        self.assertEqual(decision.diagnostic_discriminants, ("after maintenance",))
        self.assertEqual(decision.missing_information, ("indicator state",))
        self.assertFalse(decision.degraded)
        self.calls["synthesize_smart_start"].assert_called_once()
        self.calls["synthesize_ask"].assert_not_called()
        self.calls["synthesize_root_cause"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
