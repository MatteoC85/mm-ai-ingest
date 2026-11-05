"""Actual request-flow cache policy guard; all collaborators are offline fixtures."""
from contextlib import ExitStack
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import request_flow as flow
from machinemind.retrieval import diagnostic_query, review_packet, review_references
from machinemind.retrieval import root_response_quality as quality

QUERY = "The Model Q clamp slips; its pressure reading is unknown."
QUALITY_SCOPE_KEY = "_root_response_quality_policy"


class FixtureHTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


def unexpected(*args, **kwargs):
    raise AssertionError("Unexpected collaborator; test permits no provider or network I/O")


def current_response():
    return {"ok": True, "status": "answered", "result_code": "ANSWERED",
        "answer": "Fixture response", "possible_causes": [{"cause": "Conditional pressure loss",
            "why": "The pressure remains unknown.", "checks": [], "citations": ["model-q-manual"]}],
        "meta": {
            diagnostic_query.BASIS_METADATA_KEY: {
                "policy_version": diagnostic_query.DIAGNOSTIC_BASIS_POLICY,
                "query_sha256": diagnostic_query.query_fingerprint(QUERY),
                "coverage_complete": True, "interpretation_status": "valid"},
            "root_review_packet": {"policy_version": review_packet.POLICY_VERSION},
            "root_review_decisions": {"policy_version": review_references.POLICY_VERSION,
                "quality_policy_version": quality.POLICY_VERSION}}}


class RootQualityCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("socket.create_connection", side_effect=unexpected))
        self.stack.enter_context(patch("socket.socket.connect", side_effect=unexpected))

    def run_flow(self, cached, mode="root_cause"):
        lookups, stores, core_requests = [], [], []
        budget = SimpleNamespace(company_id="", semantic_cache="miss", route="")
        context = ContextVar("offline_root_quality_cache", default=None)
        scope = {"company_id": "fixture-company", "machine_id": "fixture-machine",
            "ai_scope": "machine_all", "document_ids": None, "bubble_document_id": None}
        payload = SimpleNamespace(query=QUERY, language="en", top_k=5, max_causes=3,
            debug=False, **scope)

        def lookup(**kwargs):
            lookups.append(deepcopy(kwargs))
            return deepcopy(cached)

        def store(**kwargs):
            stores.append(deepcopy(kwargs))

        def core(request):
            core_requests.append(request)
            result = current_response()
            result["origin"] = "fresh-core"
            return result

        def technical_error(**kwargs):
            raise AssertionError("Unexpected request-flow exception") from kwargs["exc"]

        values = {field.name: unexpected for field in fields(flow.RequestFlowRuntime)}
        values.update(AI_INTERNAL_SECRET="offline-fixture-only", ASK_MAX_TOP_K=20,
            AssistantCoreRequest=lambda **kwargs: SimpleNamespace(**kwargs),
            HTTPException=FixtureHTTPError, MODE_ASK="ask", MODE_ROOT_CAUSE="root_cause",
            _ROOT_DIAGNOSTIC_QUERY_PROFILE_KEY="root_query_profile", _V13BudgetExceeded=type("BudgetExceeded", (Exception,), {}),
            _V13_BUDGET_CTX=context,
            _assistant_core_attach_runtime_meta=lambda response, *args, **kwargs: response,
            _assistant_core_clear_unsupported_sources=lambda response: response,
            _assistant_core_new_budget=lambda *args, **kwargs: budget,
            _assistant_core_technical_error=technical_error,
            _assistant_ui_finalize_response=lambda response, **kwargs: response,
            _resolve_query_scope=lambda **kwargs: deepcopy(scope),
            _retrieval_diagnostic_query=diagnostic_query,
            _retrieval_precision_facts=SimpleNamespace(SCALAR_TARGET_KEY="scalar_target"),
            _retrieval_review_packet=review_packet, _retrieval_review_references=review_references,
            _root_cause_response_language=lambda *args, **kwargs: "en",
            _select_response_language=lambda *args, **kwargs: "en",
            _v13_cache_lookup=lookup, _v13_cache_store=store,
            response_has_rejected_answer=lambda response: False, run_core=core)
        response = flow.run_sync(payload, "offline-fixture-only", requested_mode=mode,
            runtime=flow.RequestFlowRuntime(**values))
        self.assertIsNone(context.get(), "Request budget context must be restored")
        return SimpleNamespace(response=response, lookups=lookups, stores=stores,
            core_requests=core_requests, budget=budget, scope=scope)

    def test_missing_or_old_quality_policy_cannot_reuse_cached_causes(self):
        for policy in (None, "root-response-quality-old"):
            with self.subTest(policy=policy):
                cached = current_response()
                verdict = cached["meta"]["root_review_decisions"]
                if policy is None:
                    verdict.pop("quality_policy_version")
                else:
                    verdict["quality_policy_version"] = policy
                result = self.run_flow(cached)
                self.assertEqual(result.response.get("origin"), "fresh-core")
                self.assertEqual(len(result.core_requests), 1)
                self.assertEqual(len(result.stores), 1)
                self.assertEqual(result.budget.semantic_cache, "bypass_unvalidated_observation_basis")

    def test_current_quality_and_existing_proofs_reuse_cache_without_new_core(self):
        cached = current_response()
        cached["origin"] = "current-cache"
        result = self.run_flow(cached)
        self.assertEqual(result.response["origin"], "current-cache")
        self.assertEqual(result.core_requests, [])
        self.assertEqual(result.stores, [])
        self.assertEqual(result.budget.route, "assistant_core_semantic_cache")

    def test_read_and_store_share_current_quality_scoped_key(self):
        result = self.run_flow(None)
        self.assertEqual(len(result.lookups), 1)
        self.assertEqual(len(result.stores), 1)
        self.assertEqual(result.lookups[0]["scope"], result.stores[0]["scope"])
        self.assertEqual(result.lookups[0]["scope"][QUALITY_SCOPE_KEY], quality.POLICY_VERSION)
        for key, value in result.scope.items():
            self.assertEqual(result.lookups[0]["scope"][key], value)

    def test_next_quality_policy_changes_root_key_without_touching_other_scope(self):
        before = self.run_flow(None).lookups[0]["scope"]
        with patch.object(quality, "POLICY_VERSION", "root-response-quality-next"):
            after = self.run_flow(None).lookups[0]["scope"]
        self.assertNotEqual(before, after)
        self.assertEqual(after[QUALITY_SCOPE_KEY], "root-response-quality-next")
        self.assertEqual({k: v for k, v in before.items() if k != QUALITY_SCOPE_KEY},
                         {k: v for k, v in after.items() if k != QUALITY_SCOPE_KEY})

    def test_current_quality_never_overrides_existing_observation_or_proof_guards(self):
        changes = [(diagnostic_query.BASIS_METADATA_KEY, "policy_version", "old-basis"),
            (diagnostic_query.BASIS_METADATA_KEY, "query_sha256", "foreign-query"),
            (diagnostic_query.BASIS_METADATA_KEY, "coverage_complete", False),
            (diagnostic_query.BASIS_METADATA_KEY, "interpretation_status", "invalid"),
            ("root_review_packet", "policy_version", "old-packet"),
            ("root_review_decisions", "policy_version", "old-proof-policy")]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                cached = current_response()
                cached["meta"][section][key] = value
                result = self.run_flow(cached)
                self.assertEqual(result.response.get("origin"), "fresh-core")
                self.assertEqual(len(result.core_requests), 1)

    def test_ask_cache_behavior_and_key_do_not_depend_on_root_quality(self):
        cached = {"ok": True, "status": "answered", "answer": "Existing ASK answer.",
            "origin": "ask-cache", "meta": {}}
        before = self.run_flow(cached, mode="ask")
        with patch.object(quality, "POLICY_VERSION", "root-response-quality-next"):
            after = self.run_flow(cached, mode="ask")
        self.assertEqual(before.response, cached)
        self.assertEqual(after.response, cached)
        self.assertEqual(before.core_requests + after.core_requests, [])
        self.assertEqual(before.stores + after.stores, [])
        self.assertEqual(before.lookups[0]["scope"], after.lookups[0]["scope"])
        self.assertEqual(before.lookups[0]["scope"],
            {**before.scope, "_v13_top_k": 5, "_v13_max_causes": 3})
        self.assertNotIn(QUALITY_SCOPE_KEY, before.lookups[0]["scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
