"""B4l trusted-application request boundary, reusable by HTTP ingress and pre-cache checks.

Required mode is an explicit deployment migration, not enabled by import.
Unknown configuration fails closed. Source adapters use the same concrete
provider; canonical Core activation remains the separate B4o gate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hmac
import math
import time
from typing import Any, Callable, Mapping

from ..authority.contracts import (AUTHORITY_VERSION, ApplicationBoundary,
    ApplicationGrant, AuthorityError, AuthorityLimits, AuthorityMeter)
from ..authority.bubble_directory import BubbleConnection, BubbleDirectory, _json_object
from ..authority.policy import BubbleAuthority, BubbleSchema, SourceSchema
from ..evidence.contracts import SourceType, SourceIdentity
from ..evidence.response_authority import make_response_guard, ResponseSourceAuthorityError
from ..retrieval.supplemental_evidence import storage_key
from .request_flow import RequestFlowGuards
from ..retrieval.chunk_evidence import ChunkReadScope

MODE_VARIABLE = "MM_ASK_REQUEST_AUTHORITY"


def required(env: Mapping[str, str]) -> bool:
    value = env.get(MODE_VARIABLE, "off")
    if value not in {"off", "required"}:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
    return value == "required"


def _config(env, key):
    value = env.get(key)
    if type(value) is not str or not value or len(value) > 32768:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
    return value


def _load_schema(value: str) -> BubbleSchema:
    try:
        fields = _json_object(value.encode("utf-8"))
        specs = fields.pop("sources")
        # B4l originally re-read Bubble User role/company on every ASK. The
        # trusted Bubble bridge now owns that end-user authorization. Keep
        # accepting the already-provisioned schema JSON during cleanup by
        # discarding only those known-deprecated fields.
        for key in ("user_type", "user_company_field", "user_role_field", "superadmin_value"):
            fields.pop(key, None)
        if type(specs) is not list:
            raise ValueError()
        fields["sources"] = tuple(SourceSchema(**{**s, "source_type": SourceType(s["source_type"])}) for s in specs)
        return BubbleSchema(**fields)
    except Exception:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID") from None


def scope_from_resolved(value: dict) -> ChunkReadScope:
    if type(value) is not dict:
        raise AuthorityError("AUTHORITY_SCOPE_INVALID", 400)
    try:
        return ChunkReadScope(value["company_id"], value["machine_id"], value["ai_scope"],
            tuple(value.get("document_ids") or ()), value.get("bubble_document_id") or None)
    except Exception:
        raise AuthorityError("AUTHORITY_SCOPE_INVALID", 400) from None


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizedCall:
    """Request-local capability; only server application code can construct one."""
    payload: Any = field(repr=False)
    scope: ChunkReadScope
    grant: ApplicationGrant = field(repr=False)
    provider: BubbleAuthority = field(repr=False)
    resolve: Callable = field(repr=False)

    def check(self, payload) -> None:
        if payload is not self.payload:
            raise AuthorityError("AUTHORITY_REQUEST_EXPIRED", 403)
        now = self.resolve(company_id=payload.company_id, machine_id=payload.machine_id,
            bubble_document_id=payload.bubble_document_id, document_ids=payload.document_ids,
            ai_scope=payload.ai_scope)
        if scope_from_resolved(now) != self.scope:
            raise AuthorityError("AUTHORITY_SCOPE_CHANGED", 403)
        self.provider.authorize_scope(self.grant, self.scope)

    def public_context(self) -> dict:
        return {"ok": True, "status": "authorized", "result_code": "REQUEST_AUTHORIZED",
            "authority_version": AUTHORITY_VERSION,
            "company_id": self.scope.company_id, "machine_id": self.scope.machine_id,
            "ai_scope": self.scope.ai_scope,
            "document_ids": list(self.scope.document_ids),
            "bubble_document_id": self.scope.bubble_document_id,
            "canonical_evidence_active": False}


def authorize_http_request(payload, *, service_secret: object, application_secret: object,
        env: Mapping[str, str], resolve: Callable,
        clock: Callable = time.monotonic, opener=None) -> AuthorizedCall:
    if not required(env):
        raise AuthorityError("AUTHORITY_DISABLED")
    old_secret = _config(env, "AI_INTERNAL_SECRET")
    if (type(service_secret) is not str or not service_secret.isascii()
            or not old_secret.isascii() or not hmac.compare_digest(service_secret, old_secret)):
        raise AuthorityError("AUTH_REQUIRED", 401)
    boundary = ApplicationBoundary(secret=_config(env, "MM_APP_AUTHORITY_SECRET"),
        legacy_secret=old_secret)
    grant = boundary.authenticate(supplied_secret=application_secret)
    # All configuration is resolved before reading the application directory.
    schema = _load_schema(_config(env, "MM_BUBBLE_AUTHORITY_SCHEMA_JSON"))
    try:
        limits = AuthorityLimits(**_json_object(_config(env, "MM_AUTHORITY_LIMITS_JSON").encode("utf-8")))
    except Exception:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID") from None
    connection = BubbleConnection(base_url=_config(env, "MM_BUBBLE_AUTHORITY_BASE_URL"),
        allowed_host=_config(env, "MM_BUBBLE_AUTHORITY_HOST"),
        token=_config(env, "MM_BUBBLE_AUTHORITY_TOKEN"))
    directory = BubbleDirectory(connection=connection, meter=AuthorityMeter(limits, clock), opener=opener)
    provider = BubbleAuthority(directory=directory, schema=schema, boundary=boundary)
    scope = scope_from_resolved(resolve(company_id=payload.company_id, machine_id=payload.machine_id,
        bubble_document_id=payload.bubble_document_id, document_ids=payload.document_ids, ai_scope=payload.ai_scope))
    authorized = AuthorizedCall(payload, scope, grant, provider, resolve)
    authorized.check(payload)
    return authorized


def protected_call(payload, service_secret, *, authorized: AuthorizedCall,
                   delegate: Callable, response_guard: Callable | None = None) -> dict:
    """Check current authorized application context before execution and release.

    Delegate must have its response caches disabled until B4n adds source-safe
    re-use. This function does not claim to authorize legacy citation provenance
    or source revocation hidden inside a still-legacy callback.
    """
    if (type(authorized) is not AuthorizedCall or not callable(delegate)
            or (response_guard is not None and not callable(response_guard))):
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
    authorized.check(payload)
    result = None
    try:
        result = delegate(payload, service_secret)
        authorized.check(payload)
        if response_guard is not None:
            result = response_guard(result)
            authorized.check(payload)
        if type(result) is not dict or type(result.get("meta", {})) is not dict:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    except AuthorityError as exc:
        # A revoked request must not release content, but denying the answer
        # does NOT make a model call already performed free. Preserve only
        # existing server accounting scalars; never return evidence/debug data.
        exc.execution_accounting = _accounting_after_execution(result)
        raise
    result = dict(result)
    result["meta"] = {**dict(result.get("meta") or {}),
        "request_authority": {**authorized.provider.directory.meter.summary(),
            "scope_authorized": True, "canonical_evidence_active": False,
            "cache_reuse": "disabled_pending_B4n"},
        "cacheable": False, "semantic_cacheable": False}
    return result


def _accounting_after_execution(result) -> dict:
    meta = result.get("meta", {}) if type(result) is dict else {}
    meta = meta if type(meta) is dict else {}
    out = {"authority_denied_after_execution": True}
    numbers = ("v13_elapsed_seconds", "v13_llm_calls", "v13_estimated_cost_usd",
               "v13_committed_cost_usd", "v13_uncertain_cost_usd")
    for key in numbers:
        value = meta.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            out[key] = value
    complete = meta.get("v13_accounting_complete") is True and all(k in out for k in numbers)
    out["v13_accounting_complete"] = complete
    out["authority_execution_accounting"] = "retained" if complete else "pending_or_unknown"
    version = meta.get("v13_budget_policy_version")
    if type(version) is str and len(version) <= 128 and all(c.isalnum() or c in "_-.:" for c in version):
        out["v13_budget_policy_version"] = version
    return out


def public_error(exc: AuthorityError) -> dict:
    accounting = getattr(exc, "execution_accounting", {})
    return {"ok": False, "status": "error", "result_code": exc.code,
            "error": {"code": exc.code, "message": exc.code},
            "answer": "", "citations": [], "rg_links": [],
            "meta": {**accounting, "cacheable": False, "semantic_cacheable": False}}


RESPONSE_FLOW_GUARD_VERSION = "ask-response-flow-guards-p6b4o-v1"


class ResponseGuardOwner:
    """One HTTP request's CURRENT source guards; B4o response-flow substep.

    The application boundary remains Bubble. This owner does not create grants,
    EvidenceSession, new readers or another Core. It does not infer ownership
    from citation text, cached manifests or retrieval. Every successful source
    check obtains a fresh allowance from the existing application provider.

    Lookup proof mismatch is deliberately a cache miss, not a poisoned request.
    Provider/configuration/lifetime failures are sticky. Store/final proof
    failures are sticky too: a swallowed store denial cannot release an answer.
    protected_call performs a final current-source check before release and
    retains execution accounting when authority is lost after model execution.

    This is source identity/ownership validation, NOT full occurrence lineage,
    URL validity, answer entailment or end-to-end canonical activation. Protected
    response cache remains disabled until the complete B4o composition gate.
    """
    def __init__(self, authorized: AuthorizedCall):
        if type(authorized) is not AuthorizedCall:
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self._authorized = authorized
        self._payload = authorized.payload
        self._scope = authorized.scope
        self._key = self._payload_key()
        self._active = True
        self._fault = None
        self._guards = RequestFlowGuards(self.lookup, self.store, self.final, self.check)

    def _payload_key(self):
        value = self._payload
        return deepcopy(tuple(getattr(value, name, None) for name in (
            "company_id", "machine_id", "ai_scope", "document_ids", "bubble_document_id",
            "query", "language", "top_k", "debug")))

    @property
    def guards(self) -> RequestFlowGuards:
        self.check()
        return self._guards

    def _fail(self, code: str):
        if self._fault is None:
            self._fault = AuthorityError(code)
        raise self._fault

    def check(self) -> None:
        if self._fault is not None:
            raise self._fault
        if not self._active:
            self._fail("AUTHORITY_REQUEST_EXPIRED")
        try:
            if (self._authorized.payload is not self._payload
                    or self._authorized.scope != self._scope
                    or self._payload_key() != self._key):
                self._fail("AUTHORITY_SCOPE_CHANGED")
        except AuthorityError:
            raise
        except Exception:
            self._fail("AUTHORITY_REQUEST_INVALID")

    def _current(self) -> frozenset[SourceIdentity]:
        self.check()
        try:
            self._authorized.check(self._payload)
            current = self._authorized.provider.current_sources(
                self._authorized.grant, self._scope)
            self.check()
            if (type(current) is not frozenset
                    or any(type(source) is not SourceIdentity
                        or not self._scope.permits(source.scope.company_id,
                            source.scope.machine_id, storage_key(source)) for source in current)):
                self._fail("AUTHORITY_RESPONSE_INVALID")
            return current
        except AuthorityError as exc:
            self._fault = exc
            raise
        except Exception:
            self._fail("AUTHORITY_PROVIDER_UNAVAILABLE")

    def _validate(self, response: dict, *, cached: bool, terminal: bool) -> dict:
        current = self._current()
        try:
            # Pass detached data: a cache entry must not gain or share authority
            # metadata by aliasing a response object mutated later by presentation.
            result = make_response_guard(
                company_id=self._scope.company_id,
                machine_id="" if self._scope.ai_scope == "company_general" else (self._scope.machine_id or ""),
                ai_scope=self._scope.ai_scope,
                current_allowed_sources=current,
                require_existing_manifest=cached,
            )(deepcopy(response))
        except ResponseSourceAuthorityError:
            if terminal:
                self._fail("AUTHORITY_RESPONSE_INVALID")
            raise
        except Exception:
            self._fail("AUTHORITY_RESPONSE_INVALID")
        self.check()
        return result

    def lookup(self, response: dict) -> dict:
        return self._validate(response, cached=True, terminal=False)

    def store(self, response: dict) -> dict:
        # A miss is allowed for cache lookup, never as release authorization.
        return self._validate(response, cached=True, terminal=True)

    def final(self, response: dict) -> dict:
        self.check()
        if type(response) is not dict or type(response.get("meta", {})) is not dict:
            self._fail("AUTHORITY_RESPONSE_INVALID")
        if response.get("ok") is False and response.get("status") == "error":
            # Do not make an already content-free technical failure depend on a
            # new provider read. Never release an error carrying a draft/source.
            result = deepcopy(response)
            for name in ("answer", "answer_html"):
                result[name] = ""
            for name in ("citations", "rg_links", "candidates"):
                if name in result or name in ("citations", "rg_links"):
                    result[name] = []
            result.pop("_assistant_core_validation_evidence", None)
            result["meta"] = {**result.get("meta", {}),
                "cacheable": False, "semantic_cacheable": False}
            return result
        result = self._validate(response, cached=False, terminal=True)
        result["meta"] = {**result.get("meta", {}),
            "response_flow_guard": {"version": RESPONSE_FLOW_GUARD_VERSION,
                "current_source_authority_checked": True,
                "canonical_evidence_active": False}}
        return result

    def close(self) -> None:
        # Idempotent even on exceptions. Retained guard callbacks then expire.
        self._active = False
        self._authorized = None
        self._payload = None
        self._key = None

    def __repr__(self):
        return "ResponseGuardOwner(<request-owned>)"
