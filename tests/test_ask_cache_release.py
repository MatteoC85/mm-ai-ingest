"""Offline cache regression: a deploy cannot inherit an older answer contract."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import semantic_cache as cache
from machinemind.ask import protected_cache as protected
from machinemind.ask.application_authority import AuthorizedCall, ResponseGuardOwner
from machinemind.ask.request_flow import RequestFlowRuntime
from machinemind.evidence.adapter_types import AdapterLimits
from machinemind.evidence.assembly import AssemblyLimits
from machinemind.evidence.contracts import SourceIdentity, SourceScope, SourceType, ScopeLevel
from machinemind.evidence.legacy_compatibility import LegacyLimits
from machinemind.evidence.manifest import ManifestLimits
from machinemind.evidence.response_authority import ResponseSourceAuthorityError
from machinemind.retrieval.chunk_evidence import ChunkReadScope, ChunkEvidenceLimits
from machinemind.retrieval.document_readers import FetchDocumentFileMapRuntime

A, B = "a" * 40, "b" * 40
QUERY = "Describe the operation."


class MemoryDatabase:
    """Execute the real cache code against its SQL parameters, without I/O."""
    def __init__(self):
        self.rows, self.statements, self.selected = [], [], []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=()):
        self.statements.append((sql, params))
        if "INSERT INTO" in sql:
            self.rows.append(tuple(params))
        elif "SELECT id, query_text" in sql:
            self.selected = [(i + 1, r[9], json.loads(r[11]), r[12], "offline")
                for i, r in enumerate(self.rows) if r[:8] == tuple(params[:8])]
        elif "SELECT id, query_embedding" in sql:
            self.selected = [(i + 1, json.loads(r[10])) for i, r in enumerate(self.rows)
                             if i + 1 in params[0]]

    def fetchall(self):
        return self.selected

    def close(self):
        pass

    commit = rollback = close


class CacheReleaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"COMMIT_SHA": A}))
        self.stack.enter_context(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("network forbidden")))
        self.db = MemoryDatabase()
        self.budget = SimpleNamespace(embedding_cache={("embed", QUERY): [1., 0.]},
            semantic_cache="miss", estimated_cost_usd=0, llm_calls=0)
        self.rt = {
            "_v13_current_budget": lambda: self.budget,
            "V13_SEMANTIC_CACHE_ENABLED": True,
            "_v13_cache_bootstrap": Mock(return_value=True),
            "_v13_get_knowledge_version": lambda company: 1,
            "_v13_scope_key": lambda scope: json.dumps(scope, sort_keys=True),
            "_db_conn": Mock(return_value=self.db),
            "V13_ENGINE_KEY": "stable-engine",
            "V13_SEMANTIC_CACHE_MIN_QUALITY": .5,
            "V13_SEMANTIC_CACHE_SCAN_LIMIT": 20,
            "V13_SEMANTIC_CACHE_THRESHOLD_ASK": .9,
            "V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE": .9,
            "_v13_normalize_query": lambda q: q.lower().strip(),
            "_v13_jsonb_to_python": lambda value, fallback: value,
            "_v13_semantic_cache_compatible": lambda *args: True,
            "_openai_embed_texts": Mock(return_value=[[1., 0.]]),
            "_cosine_sim": lambda *args: 1.,
            "_build_rg_links": lambda *args: [],
            "_assistant_core_cache_certified": lambda *args: True,
            "_v13_response_quality": lambda *args: 1.,
            "OPENAI_EMBED_MODEL": "embed",
            "V13_SEMANTIC_CACHE_TTL_SECONDS": 3600,
            "V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY": 100,
        }
        self.params = dict(mode="ask", q=QUERY, company_id="tenant", machine_id="machine",
            scope={"ai_scope": "machine_all"}, language="en", debug=False)
        self.answer = {"ok": True, "status": "answered", "answer": "Old procedure.",
            "citations": [], "meta": {"v13_estimated_cost_usd": .5}}
        self.stack.enter_context(patch("machinemind.accounting.cache_receipt.allow_cache", return_value=True))

    def store(self, **changes):
        cache.cache_store(**{**self.params, **changes}, response=self.answer, runtime_globals=self.rt)

    def lookup(self, **changes):
        return cache.cache_lookup(**{**self.params, **changes}, runtime_globals=self.rt)

    def test_exact_and_semantic_hits_stay_within_the_same_release(self):
        self.store()
        for query in (QUERY, "Explain this operation."):
            with self.subTest(query=query):
                self.assertIsNotNone(self.lookup(q=query))
        self.assertEqual(self.budget.estimated_cost_usd, 0)
        self.assertEqual(self.budget.llm_calls, 0)

    def test_new_release_misses_exact_and_semantic_old_answers_without_embedding(self):
        self.store()
        self.rt["_openai_embed_texts"].reset_mock()
        with patch.dict(os.environ, {"COMMIT_SHA": B}):
            for query in (QUERY, "Explain this operation."):
                self.assertIsNone(self.lookup(q=query))
            self.store()
            self.assertIsNotNone(self.lookup())
        # Only storing the new answer used the already mocked/reused embedding.
        self.assertEqual(self.rt["_openai_embed_texts"].call_count, 1)
        self.assertNotEqual(self.db.rows[0][3], self.db.rows[1][3])

    def test_unversioned_old_sql_namespace_is_unreachable(self):
        self.store()
        row = list(self.db.rows[0])
        row[3] = self.rt["_v13_scope_key"](self.params["scope"])
        self.db.rows[0] = tuple(row)
        self.assertIsNone(self.lookup())

    def test_missing_invalid_identity_never_bootstraps_reads_or_writes(self):
        for value in ("", "a" * 39, "g" * 40, A + "\n", "main"):
            with self.subTest(value=value), patch.dict(os.environ, {"COMMIT_SHA": value}):
                self.assertIsNone(self.lookup())
                self.store()
        self.rt["_v13_cache_bootstrap"].assert_not_called()
        self.rt["_db_conn"].assert_not_called()
        self.rt["_openai_embed_texts"].assert_not_called()
        self.assertEqual(self.budget.semantic_cache, "bypass_release_identity")

    def test_payload_cannot_override_deploy_and_tenant_selectors_remain_distinct(self):
        scoped = {**self.params["scope"], "COMMIT_SHA": A, "commit_sha": A}
        self.store(scope=scoped)
        with patch.dict(os.environ, {"COMMIT_SHA": B}):
            self.assertIsNone(self.lookup(scope=scoped))
        for changes in ({"company_id": "other"}, {"machine_id": "other"},
                        {"scope": {"ai_scope": "document_ids", "document_ids": ["other"]}}):
            self.assertIsNone(self.lookup(**changes))

    def test_root_cause_existing_namespace_is_unchanged_even_without_sha(self):
        with patch.dict(os.environ, {"COMMIT_SHA": ""}):
            self.store(mode="root_cause")
            self.assertIsNotNone(self.lookup(mode="root_cause"))
        self.assertEqual(self.db.rows[0][3], self.rt["_v13_scope_key"](self.params["scope"]))

    def test_same_release_hits_and_writes_still_require_supplied_authority_guard(self):
        denied = Mock(side_effect=ResponseSourceAuthorityError("revoked"))
        cache.cache_store(**self.params, response=self.answer, runtime_globals=self.rt,
                          source_guard_fn=denied)
        self.assertEqual(self.db.rows, [])
        self.store()
        for query in (QUERY, "Explain this operation."):
            self.assertIsNone(cache.cache_lookup(**{**self.params, "q": query},
                runtime_globals=self.rt, source_guard_fn=denied))
        self.assertEqual(denied.call_count, 3)

    def owner(self):
        payload = SimpleNamespace(query=QUERY, language="en", debug=False, top_k=5,
            company_id="tenant", machine_id="machine", ai_scope="machine_all",
            document_ids=None, bubble_document_id=None, metadata={"COMMIT_SHA": A})
        authorized = AuthorizedCall(payload, ChunkReadScope("tenant", "machine", "machine_all"),
                                    None, None, None)
        flow = {f.name: Mock() for f in fields(RequestFlowRuntime)}
        flow.update(ASK_MAX_TOP_K=20, _select_response_language=lambda *args, **kwargs: "en")
        limits = ChunkEvidenceLimits(AdapterLimits(1000, 30, 1000),
            AssemblyLimits(ManifestLimits(30, 10000), 30, 10000), LegacyLimits(10, 1000, 10000))
        return protected.ProtectedCacheOwner(authorized=authorized,
            source_owner=ResponseGuardOwner(authorized), flow_runtime=RequestFlowRuntime(**flow),
            cache_runtime=self.rt, signing_key="offline-test-key-" * 3,
            file_runtime=FetchDocumentFileMapRuntime(Mock()), limits=limits)

    def owner_context(self, owner):
        owner._set_context({**self.params, "scope": {"ai_scope": "machine_all", "_v13_top_k": 5}})
        owner._epoch = 1

    def test_protected_sql_scope_and_valid_hmac_proof_bind_captured_deploy(self):
        source = SourceIdentity(SourceScope("tenant", ScopeLevel.MACHINE, "machine"),
                                SourceType.DOCUMENT, "manual")
        first = self.owner()
        self.owner_context(first)
        response = deepcopy(self.answer)
        response["meta"][protected.ARTIFACT_KEY] = first._proof(response, frozenset({source}))
        first._current = Mock(return_value=frozenset({source}))
        with patch.object(protected.semantic_cache, "assistant_core_cache_certified", return_value=True), \
             patch.object(protected, "make_response_guard", return_value=lambda response: response):
            self.assertIsNotNone(first.lookup(response))
            first._current.assert_called_once()
            first._current.return_value = frozenset()
            with self.assertRaises(ResponseSourceAuthorityError):
                first.lookup(response)
            with patch.dict(os.environ, {"COMMIT_SHA": B}):
                second = self.owner()
                self.owner_context(second)
                self.assertEqual(second._context["commit_sha"], B)
                self.assertNotEqual(first._runtime()["_v13_scope_key"](self.params["scope"]),
                                    second._runtime()["_v13_scope_key"](self.params["scope"]))
                with self.assertRaises(ResponseSourceAuthorityError):
                    second.lookup(response)  # authentic old artifact still cannot transfer
        self.assertEqual(first._context["commit_sha"], A)

    def test_protected_missing_identity_bypasses_cache_and_preserves_owner_checks(self):
        with patch.dict(os.environ, {"COMMIT_SHA": ""}):
            owner = self.owner()
        args = {**self.params, "scope": {"ai_scope": "machine_all", "_v13_top_k": 5}}
        with patch.object(cache, "cache_lookup") as lookup, patch.object(cache, "cache_store") as store:
            self.assertIsNone(owner.lookup_entry(**args, source_guard_fn=owner.lookup))
            owner.store_entry(**args, response=self.answer, source_guard_fn=owner.store)
            lookup.assert_not_called()
            store.assert_not_called()
        owner.authorized.payload.machine_id = "changed"
        with self.assertRaises(Exception):
            owner.check()


if __name__ == "__main__":
    unittest.main()
