"""Request-owned exact cache reuse with current source authority.

Reuses the production semantic-cache SQL/TTL/quality/knowledge-version policies
in a separate scope namespace. Protected hits require the EXACT query/selectors;
semantic similarity is not a proof that a previously validated answer answers a
new question. No model/embedding call is added for caching. OFF/RC keep their
existing cache unchanged.

A domain-separated HMAC authenticates the server-produced cache artifact, NOT
permissions. Every reuse separately checks the current application allowance for
ALL snapshot dependencies (including uncited reads and relation targets). Cached
handles are never imported into a live EvidenceSession. Links are rebuilt from a
fresh typed file-reference read. Key rotation makes old entries misses.

The normal process trusts its internal callbacks and DB transport, not client
metadata. This is not an atomic revocation transaction across Bubble/SQL/HTTP.
Content freshness uses existing company knowledge-version invalidation and TTL;
out-of-band content edits without invalidation are not certified here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import hmac
import json
import math
from typing import Callable

from .application_authority import AuthorizedCall, ResponseGuardOwner
from .routing_runtime import accounting_state
from .final_contract import response_contract_bound
from .request_flow import RequestFlowRuntime, RequestFlowGuards
from ..authority.contracts import AuthorityError
from ..evidence.contracts import SourceIdentity, to_primitive
from ..evidence.response_authority import (make_response_guard,
    citation_storage_key, ResponseSourceAuthorityError)
from ..retrieval.chunk_evidence import ChunkEvidenceLimits
from ..retrieval.document_readers import (FetchDocumentFileMapRuntime,
    read_document_file_references)
from ..retrieval.supplemental_evidence import storage_key
from ..infrastructure import semantic_cache

PROTECTED_CACHE_VERSION = "ask-canonical-exact-cache-p6b4o-final-contract-v4"
ARTIFACT_KEY = "_mm_canonical_cache_artifact"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_DEPENDENCIES = 8192
_DOMAIN = b"MachineMind/ASK/protected-cache/v1\x00"

# Only request accounting added by main after UI rendering is excluded when
# checking that request_flow did not alter the observed canonical response.
_RUNTIME_META = frozenset({
    "v13_runtime", "v13_engine", "v13_code_marker", "v13_engine_key", "v13_route",
    "v13_semantic_cache", "v13_semantic_cache_detail", "v13_evidence_gate",
    "v13_retrieval_assurance", "v13_elapsed_seconds", "v13_llm_calls",
    "v13_estimated_cost_usd", "v13_budget_policy_version", "v13_committed_cost_usd",
    "v13_uncertain_cost_usd", "v13_accounting_complete", "assistant_core_enabled",
    "assistant_core_code_marker", "assistant_core_release_id",
    "assistant_core_hard_timeout_seconds", "response_flow_guard",
    "response_source_manifest", "request_authority", "protected_cache", ARTIFACT_KEY,
})


def _json(value) -> bytes:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError("bounded canonical cache artifact required")
    return data


def _hash(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _storage_view(response: dict) -> dict:
    """Exactly the existing cache-store projection, minus our detached seal."""
    out = deepcopy(response)
    out.pop("debug", None)
    out["rg_links"] = []
    meta = dict(out.get("meta") or {})
    for key in ("v13_runtime", "v13_semantic_cache", ARTIFACT_KEY):
        meta.pop(key, None)
    out["meta"] = meta
    return out


def _flow_view(response: dict) -> dict:
    out = deepcopy(response)
    meta = out.get("meta") or {}
    out["meta"] = {key: val for key, val in meta.items() if key not in _RUNTIME_META}
    out["meta"].setdefault("cacheable", True)
    return out


def _sources(sources: frozenset[SourceIdentity]) -> list[dict]:
    if (type(sources) is not frozenset or len(sources) > MAX_DEPENDENCIES
            or any(type(source) is not SourceIdentity for source in sources)):
        raise ValueError("bounded typed snapshot dependencies required")
    return sorted((to_primitive(source) for source in sources), key=_json)


class ProtectedCacheOwner:
    """One cache/access lifetime shared by the protected HTTP request.

    observe_response is an INTERNAL completion callback from RequestEvidenceOwner,
    after Core finalization or the registered scalar producer. It is never called
    on cache lookup or on client-deserialized data. Default callers without this
    observer cannot mint cache entries, even with a forged canonical flag.
    """
    def __init__(self, *, authorized: AuthorizedCall, source_owner: ResponseGuardOwner,
                 flow_runtime: RequestFlowRuntime, cache_runtime: dict,
                 signing_key: str, file_runtime: FetchDocumentFileMapRuntime,
                 limits: ChunkEvidenceLimits):
        if (type(authorized) is not AuthorizedCall
                or type(source_owner) is not ResponseGuardOwner
                or type(flow_runtime) is not RequestFlowRuntime
                or type(cache_runtime) is not dict
                or type(file_runtime) is not FetchDocumentFileMapRuntime
                or type(limits) is not ChunkEvidenceLimits
                or type(signing_key) is not str or not signing_key.isascii()
                or not 32 <= len(signing_key) <= 256):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self.authorized, self.source_owner = authorized, source_owner
        self.scope, self.flow, self.cache = authorized.scope, flow_runtime, dict(cache_runtime)
        self.file_runtime, self.limits = file_runtime, limits
        self._key = hmac.new(signing_key.encode("ascii"), _DOMAIN, hashlib.sha256).digest()
        self._active, self._fault = True, None
        self._context = self._parameters = None
        self._epoch = 0
        self._observed = None
        self._dependencies = frozenset()
        self._hit_dependencies = None
        self._hit_expected = self._final = self._terminal = None
        self._mode = "miss"
        self.guards = RequestFlowGuards(self.lookup, self.store, self.final, self.check)
        self.runtime = replace(flow_runtime, _v13_cache_lookup=self.lookup_entry,
                               _v13_cache_store=self.store_entry,
                               _assistant_core_budget_response=self.budget_response)

    def _fail(self, code):
        if self._fault is None:
            self._fault = AuthorityError(code)
        raise self._fault

    def check(self):
        if self._fault is not None:
            raise self._fault
        if not self._active:
            self._fail("AUTHORITY_REQUEST_EXPIRED")
        self.source_owner.check()

    def _current(self):
        self.check()
        return self.source_owner._current()

    def _version(self):
        value = self.cache["_v13_get_knowledge_version"](self.scope.company_id)
        return value if type(value) is int and value > 0 else 0

    def _set_context(self, parameters):
        self.check()
        p = self.authorized.payload
        scope = parameters.get("scope")
        if type(scope) is not dict:
            self._fail("AUTHORITY_CACHE_SCOPE_CHANGED")
        expected = dict(mode="ask", q=str(p.query or "").strip(),
            company_id=self.scope.company_id, machine_id=self.scope.machine_id,
            language=self.flow._select_response_language(str(p.query or "").strip(), preferred=p.language),
            debug=bool(p.debug))
        if any(parameters.get(k) != v for k, v in expected.items()):
            self._fail("AUTHORITY_CACHE_SCOPE_CHANGED")
        if (scope.get("ai_scope") != self.scope.ai_scope
                or tuple(scope.get("document_ids") or ()) != self.scope.document_ids
                or (scope.get("bubble_document_id") or None) != self.scope.bubble_document_id
                or scope.get("_v13_top_k") != max(1, min(int(p.top_k or 5), self.flow.ASK_MAX_TOP_K))):
            self._fail("AUTHORITY_CACHE_SCOPE_CHANGED")
        context = {**expected, "scope": deepcopy(scope), "version": PROTECTED_CACHE_VERSION,
                   "engine": self.cache["V13_ENGINE_KEY"]}
        if self._context is not None and self._context != context:
            self._fail("AUTHORITY_CACHE_SCOPE_CHANGED")
        self._context = context
        self._parameters = deepcopy(parameters)

    def _reused_embedding(self, texts, **kwargs):
        # Never call a paid embedding endpoint merely to store a cached answer.
        budget = self.cache["_v13_current_budget"]()
        if texts != [self._context["q"]] or budget is None:
            raise ValueError("request embedding unavailable")
        value = budget.embedding_cache.get((self.cache["OPENAI_EMBED_MODEL"], texts[0]))
        if not isinstance(value, (tuple, list)) or not value or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ValueError("valid previously computed embedding required")
        return [list(value)]

    def _runtime(self, *, storing=False):
        rt = dict(self.cache)
        base_key = self.cache["_v13_scope_key"]
        rt["_v13_scope_key"] = lambda scope: _hash({
            "namespace": PROTECTED_CACHE_VERSION, "legacy_scope": base_key(scope),
            "request": self._context})[:40]
        # Query fingerprint in the scope prevents normalization collisions too.
        rt["_v13_semantic_cache_compatible"] = lambda *args: False
        rt["_openai_embed_texts"] = self._reused_embedding
        rt["_build_rg_links"] = self._refresh_links
        def version(company):
            if company != self.scope.company_id:
                self._fail("AUTHORITY_CACHE_SCOPE_CHANGED")
            value = self._version()
            if storing:
                return value if value == self._epoch else 0
            self._epoch = value
            return value
        rt["_v13_get_knowledge_version"] = version
        return rt

    def observe_response(self, response: dict, dependencies: frozenset[SourceIdentity]):
        self.check()
        if self._hit_expected is not None or self._context is None or type(response) is not dict:
            self._fail("AUTHORITY_CACHE_COMPLETION_INVALID")
        # Typed identities originate ONLY from the already-owned session.
        _sources(dependencies)
        self._observed, self._dependencies = deepcopy(response), dependencies

    def _accounting_allows_store(self):
        # The request-local ledger, not public response flags, controls sealing.
        # The new namespace also rejects old artifacts minted without this gate.
        try:
            budget = self.cache["_v13_current_budget"]()
            state = accounting_state(budget.public_meta()) if budget is not None else {}
            allowed = state.get("cache_eligible") is True
        except Exception:
            allowed = False
        if not allowed:
            self._mode = "miss_accounting_incomplete_no_store"
        return allowed

    def _proof(self, response, dependencies):
        payload = dict(version=PROTECTED_CACHE_VERSION, context=self._context,
            knowledge_version=self._epoch, dependencies=_sources(dependencies),
            response_sha256=_hash(_storage_view(response)))
        return {**payload, "seal": hmac.new(self._key, _json(payload), hashlib.sha256).hexdigest()}

    def lookup(self, response):
        """Malformed/old/revoked cached artifacts miss; provider faults latch."""
        self.check()
        try:
            if type(response) is not dict or type(response.get("meta")) is not dict:
                raise ValueError()
            proof = response["meta"][ARTIFACT_KEY]
            if type(proof) is not dict or set(proof) != {
                    "version", "context", "knowledge_version", "dependencies", "response_sha256", "seal"}:
                raise ValueError()
            value = {key: val for key, val in proof.items() if key != "seal"}
            seal = proof["seal"]
            if (type(seal) is not str or not seal.isascii()
                    or not hmac.compare_digest(seal, hmac.new(self._key, _json(value), hashlib.sha256).hexdigest())
                    or proof["version"] != PROTECTED_CACHE_VERSION
                    or proof["context"] != self._context
                    or type(proof["knowledge_version"]) is not int
                    or proof["knowledge_version"] != self._epoch or self._epoch < 1
                    or proof["response_sha256"] != _hash(_storage_view(response))
                    or not (semantic_cache.assistant_core_cache_certified("ask", response)
                            and response_contract_bound(response))):
                raise ValueError()
            deps = proof["dependencies"]
            if type(deps) is not list or not deps or len(deps) > MAX_DEPENDENCIES:
                raise ValueError()
            tokens = [_json(d) for d in deps]
            if tokens != sorted(set(tokens)):
                raise ValueError()
        except (KeyError, ValueError, TypeError, OverflowError, RecursionError):
            raise ResponseSourceAuthorityError("invalid canonical cache artifact") from None
        if self.authorized.admission is not None:
            self.authorized.admission.refresh("cache.admission")
        current = self._current()  # provider exceptions are NOT cache misses
        index = {_json(to_primitive(source)): source for source in current}
        if any(token not in index for token in tokens):
            raise ResponseSourceAuthorityError("cache dependency revoked or reassigned")
        dependencies = frozenset(index[token] for token in tokens)
        result = make_response_guard(company_id=self.scope.company_id,
            machine_id="" if self.scope.ai_scope == "company_general" else self.scope.machine_id,
            ai_scope=self.scope.ai_scope, current_allowed_sources=current,
            require_existing_manifest=True)(deepcopy(response))
        by_key = {storage_key(source) for source in dependencies}
        if any(citation_storage_key(row)[0] not in by_key for row in result.get("citations", [])):
            raise ResponseSourceAuthorityError("citation absent from signed dependencies")
        self._hit_dependencies = dependencies
        return result

    def _refresh_links(self, company_id, citations):
        self.check()
        try:
            if self._hit_dependencies is None or company_id != self.scope.company_id:
                self._fail("AUTHORITY_CACHE_LINKS_INVALID")
            current = self._current()
            if not self._hit_dependencies.issubset(current):
                self._fail("AUTHORITY_CACHE_DEPENDENCY_CHANGED")
            by_key = {storage_key(source): source for source in self._hit_dependencies}
            anchors = tuple(by_key[key] for key in sorted({citation_storage_key(c)[0] for c in citations}))
            receipt = read_document_file_references(scope=self.scope, sources=anchors,
                current_allowed_sources=current, limits=self.limits, runtime=self.file_runtime,
                allow_structured=True)
            current = self._current()
            if not self._hit_dependencies.issubset(current):
                self._fail("AUTHORITY_CACHE_DEPENDENCY_CHANGED")
            files = receipt.as_file_map(scope=self.scope, current_allowed_sources=current)
            expected = sorted(storage_key(source) for source in anchors)
            def file_map(company, keys):
                if company != company_id or keys != expected:
                    self._fail("AUTHORITY_CACHE_LINKS_INVALID")
                return dict(files)
            return self.flow._build_rg_links(company_id, deepcopy(citations), file_map_fn=file_map)
        except AuthorityError:
            raise
        except Exception:
            self._fail("AUTHORITY_CACHE_LINKS_UNAVAILABLE")

    def lookup_entry(self, *, source_guard_fn=None, **parameters):
        self.check()
        if self._context is not None or source_guard_fn != self.lookup:
            self._fail("AUTHORITY_CACHE_LOOKUP_INVALID")
        self._set_context(parameters)
        result = semantic_cache.cache_lookup(**parameters, runtime_globals=self._runtime(),
                                             source_guard_fn=self.lookup)
        self.check()  # cache infrastructure may swallow a latched error
        if result is not None and self._version() != self._epoch:
            result = None
            self._mode = "miss_knowledge_changed"
        if result is None:
            self._hit_dependencies = None
            return None
        # Exact pure rendering expected at the cache-return branch of run_sync.
        self._hit_expected = self.flow._assistant_ui_finalize_response(deepcopy(result),
                                                     language=self._context["language"])
        self._mode = "hit"
        return result

    def final(self, response):
        self.check()
        # Internal validation is not publication. Store performs its own fresh
        # dependency admission, and release() ALWAYS performs the final remote
        # source/scope fence before this body may reach the HTTP caller.
        if type(response) is not dict:
            self._fail("AUTHORITY_RESPONSE_INVALID")
        if response.get("ok") is False and response.get("status") == "error":
            out = self.source_owner.final(response, _refresh_current=False)
        else:
            expected = self._hit_expected
            if expected is None and self._observed is not None:
                expected = self.flow._assistant_ui_finalize_response(
                    self.flow._assistant_core_clear_unsupported_sources(deepcopy(self._observed)),
                    language=self._context["language"])
            if expected is None or _flow_view(response) != _flow_view(expected):
                self._fail("AUTHORITY_CACHE_RESPONSE_CHANGED")
            out = self.source_owner.final(response, _refresh_current=False)
            if self._hit_dependencies is not None:
                if not self._hit_dependencies.issubset(self._current()):
                    self._fail("AUTHORITY_CACHE_DEPENDENCY_CHANGED")
                if self._version() != self._epoch:
                    self._fail("AUTHORITY_CACHE_KNOWLEDGE_CHANGED")
            elif (self._epoch > 0 and self._dependencies
                    and self._accounting_allows_store()
                    and (semantic_cache.assistant_core_cache_certified("ask", out)
                         and response_contract_bound(out))
                    and out.get("meta", {}).get("cacheable") is not False
                    and out.get("meta", {}).get("semantic_cacheable") is not False):
                current = self._current()
                # Conservative dependency invalidation is a cache skip, NOT
                # veto of a fresh answer whose selected sources were admitted.
                if self._dependencies.issubset(current):
                    try:
                        proof = self._proof(out, self._dependencies)
                    except (ValueError, TypeError, OverflowError, RecursionError):
                        self._mode = "miss_uncacheable_serialization"
                    else:
                        out["meta"][ARTIFACT_KEY] = proof
                        self._mode = "miss_sealed"
                else:
                    self._mode = "miss_dependencies_changed_no_store"
        self._final = deepcopy(out)
        return out

    def store(self, response):
        self.check()
        if self._final is None or response != self._final or self._hit_expected is not None:
            self._fail("AUTHORITY_CACHE_STORE_INVALID")
        if ARTIFACT_KEY not in response.get("meta", {}):
            raise ResponseSourceAuthorityError("response not cache eligible")
        # Revalidate the complete artifact and independent current permissions.
        # Do not turn an invalid STORE into authorization to release another body.
        saved = self._hit_dependencies
        try:
            return self.lookup(response)
        finally:
            self._hit_dependencies = saved

    def store_entry(self, *, response, source_guard_fn=None, **parameters):
        self.check()
        if source_guard_fn != self.store:
            self._fail("AUTHORITY_CACHE_STORE_INVALID")
        self._set_context(parameters)
        if ARTIFACT_KEY not in response.get("meta", {}):
            return None
        if not self._accounting_allows_store():
            return None
        semantic_cache.cache_store(**parameters, response=deepcopy(response),
            runtime_globals=self._runtime(storing=True), source_guard_fn=self.store)
        self.check()
        return None

    def budget_response(self, **kwargs):
        """Observe the existing deterministic time/cost response without changing it."""
        self.check()
        result = self.flow._assistant_core_budget_response(**kwargs)
        if type(result) is not dict or result.get("citations") or result.get("rg_links"):
            self._fail("AUTHORITY_CACHE_BUDGET_RESPONSE_INVALID")
        self._terminal = deepcopy(result)
        return result

    def release(self, response):
        self.check()
        if self._final is None:
            # run_sync may have returned a budget/technical error outside its
            # normal finalizer. Source-bearing success cannot use this path.
            if (type(response) is not dict or
                    (response.get("ok") is not False and response != self._terminal)):
                self._fail("AUTHORITY_CACHE_RESPONSE_UNOBSERVED")
            out = self.source_owner.final(response)
        else:
            if response != self._final:
                self._fail("AUTHORITY_CACHE_RESPONSE_CHANGED")
            out = self.source_owner.final(response)
            if self._hit_dependencies is not None:
                if not self._hit_dependencies.issubset(self._current()):
                    self._fail("AUTHORITY_CACHE_DEPENDENCY_CHANGED")
                if self._version() != self._epoch:
                    self._fail("AUTHORITY_CACHE_KNOWLEDGE_CHANGED")
        out = deepcopy(out)
        out.setdefault("meta", {}).pop(ARTIFACT_KEY, None)
        out["meta"]["protected_cache"] = {"version": PROTECTED_CACHE_VERSION,
            "mode": "exact_request", "outcome": self._mode,
            "worker_cacheable": False, "canonical_activation_certified": False}
        return out

    def completed(self):
        self.check()
        return self._final is not None and (self._observed is not None or self._hit_expected is not None)

    def close(self):
        self._active = False
        self._key = b""
        self._observed = self._hit_expected = self._final = self._terminal = None
        self._dependencies = frozenset()
        self._hit_dependencies = self._context = self._parameters = None
        self.cache.clear()
        self.authorized = self.flow = self.file_runtime = None

    def __repr__(self):
        return "ProtectedCacheOwner(<request-owned>)"
