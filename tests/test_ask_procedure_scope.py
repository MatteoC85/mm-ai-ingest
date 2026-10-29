"""Offline router-contract propagation, not a claim of live classification quality."""
import copy
import ast
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_core_v2 import AssistantCoreRequest, build_router_schema, normalize_decision
import assistant_core_v2 as core
from machinemind.ask import procedure_scope, request_readiness


def request(query="Which safety conditions must hold before setup?", language="en", mode="ask"):
    return AssistantCoreRequest(query=query, response_language=language,
        requested_mode=mode, company_id="fixture-tenant", machine_id="fixture-machine",
        ai_scope="machine_all", top_k=6)


def payload(scope="conditions_only"):
    return {"effective_mode": "ask", "request_kind": "general_technical", "confidence": .9,
        "evidence_state": "partial", "evidence_policy": "machine_sources_required",
        "information_task": "procedure_segment", "procedure_scope": scope,
        "required_answer_types": ["checklist", "safety_conditions"],
        "required_facets": ["preparatory checks", "safety conditions"],
        "facet_queries": [
            {"facet": "preparatory checks", "answer_type": "checklist", "must_cover": True,
             "dense_queries": ["preparatory checks"], "lexical_queries": ["checks"]},
            {"facet": "safety conditions", "answer_type": "safety_conditions", "must_cover": True,
             "dense_queries": ["safety conditions"], "lexical_queries": ["safety"]}]}


class AskProcedureScopeTests(unittest.TestCase):
    def test_semantic_conditions_contract_survives_normalization_in_both_languages(self):
        for query, language in (("Before setup, which checks and safety conditions must hold?", "en"),
                                ("Quali controlli e condizioni servono prima dell'attrezzaggio?", "it")):
            with self.subTest(language=language):
                raw = payload(); before = copy.deepcopy(raw)
                decision = normalize_decision(request(query, language), raw)
                self.assertEqual(decision.procedure_scope, "conditions_only")
                self.assertEqual(decision.required_answer_types, ("checklist", "safety_conditions"))
                self.assertEqual(decision.information_task, "procedure_segment")
                self.assertEqual(decision.evidence_policy, "machine_sources_required")
                self.assertEqual(len(decision.facet_queries), 2)
                self.assertEqual(raw, before)

    def test_execution_or_combined_contract_never_loses_ordered_requirements(self):
        for query in ("How do I thread the machine?", "Come infilo il materiale?",
                      "Show Steps 3 through 6 and the checks before restart",
                      "Quale percorso seguire e cosa ripristinare prima del riavvio?"):
            raw = payload("execution_steps")
            raw["required_answer_types"].append("ordered_actions")
            raw["required_facets"].append("execution sequence")
            raw["facet_queries"].append({"facet": "execution sequence", "answer_type": "ordered_actions",
                "must_cover": True, "dense_queries": ["execution sequence"]})
            decision = normalize_decision(request(query), raw)
            self.assertEqual(decision.procedure_scope, "execution_steps")
            self.assertIn("ordered_actions", decision.required_answer_types)
            self.assertTrue(any(f.answer_type == "ordered_actions" for f in decision.facet_queries))

    def test_legacy_or_invalid_scope_never_acquires_conditions_only(self):
        for value in (None, "", "checks", False, [], {"conditions_only": True}):
            raw = payload(value)
            raw["required_answer_types"].append("ordered_actions")
            decision = normalize_decision(request(), raw)
            self.assertEqual(decision.procedure_scope, "")
            self.assertIn("ordered_actions", decision.required_answer_types)
        raw = payload(); raw.pop("procedure_scope")
        self.assertEqual(normalize_decision(request(), raw).procedure_scope, "")

    def test_scope_cannot_override_safety_refusal_or_other_modes(self):
        for kind in ("unsafe_request", "out_of_scope"):
            raw = payload(); raw["request_kind"] = kind
            decision = normalize_decision(request(), raw)
            self.assertEqual(decision.procedure_scope, "")
            self.assertEqual(decision.evidence_state, "unsupported")
            self.assertFalse(decision.facet_queries)
        for mode in ("root_cause", "smart_diagnostic"):
            self.assertEqual(normalize_decision(request(mode=mode), payload()).procedure_scope, "")

    def test_contradictory_contract_is_preserved_for_downstream_fail_closed_guard(self):
        raw = payload(); raw["information_task"] = "procedure_full"
        raw["required_answer_types"].append("ordered_actions")
        decision = normalize_decision(request(), raw)
        self.assertEqual(decision.information_task, "procedure_full")
        self.assertIn("ordered_actions", decision.required_answer_types)

    def test_strict_ask_schema_composes_without_mutating_base_or_duplicate_fields(self):
        base = build_router_schema(["ask"]); before = copy.deepcopy(base)
        modified = procedure_scope.router_schema(request_readiness.router_schema(base))
        modified = procedure_scope.router_schema(modified)
        self.assertEqual(base, before)
        self.assertNotIn("procedure_scope", base["schema"]["properties"])
        self.assertEqual(set(modified["schema"]["required"]), set(modified["schema"]["properties"]))
        self.assertEqual(modified["schema"]["required"].count("procedure_scope"), 1)
        self.assertFalse(modified["schema"]["additionalProperties"])

    def test_real_final_contract_keeps_checks_only_but_preserves_legacy_ordered_default(self):
        from machinemind.ask import request_scope
        source = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_assistant_core_answer_contract_check")
        namespace = {k: getattr(core, k) for k in dir(core) if k.startswith(("REQ_", "INFO_"))}
        namespace.update(AssistantCoreDecision=core.AssistantCoreDecision, re=re,
            _ask_request_scope=request_scope,
            _assistant_core_required_facet_metrics=lambda *_: {"coverage": 1., "missing": []},
            _assistant_core_sequence_signal=lambda _: False)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        check = namespace["_assistant_core_answer_contract_check"]
        for scope in ("conditions_only", "execution_steps", ""):
            decision = normalize_decision(request(), payload(scope))
            result = check(answer="- Required condition A\n- Required condition B",
                           evidence_text="Admitted conditions", decision=decision)
            if scope == "conditions_only":
                self.assertTrue(result["passed"])
                self.assertEqual(result["required_answer_types"], ["checklist", "safety_conditions"])
            else:
                self.assertFalse(result["passed"])
                self.assertIn("ordered_actions", result["required_answer_types"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
