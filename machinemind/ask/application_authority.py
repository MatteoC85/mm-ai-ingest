"""B4l production request boundary, reusable by HTTP ingress and pre-cache checks.

Required mode is an explicit deployment migration, not enabled by import.
Unknown configuration fails closed. Source adapters use the same concrete
provider; canonical Core activation remains the separate B4o gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import json
import math
import time
from typing import Any, Callable, Mapping

from ..authority.contracts import (AUTHORITY_VERSION, ApplicationBoundary,
    ApplicationPrincipal, AuthorityError, AuthorityLimits, AuthorityMeter)
from ..authority.bubble_directory import BubbleConnection, BubbleDirectory, _json_object
from ..authority.policy import BubbleAuthority, BubbleSchema, SourceSchema
from ..evidence.contracts import SourceType
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
    principal: ApplicationPrincipal = field(repr=False)
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
        self.provider.authorize_scope(self.principal, self.scope)

    def public_context(self) -> dict:
        return {"ok": True, "status": "authorized", "result_code": "REQUEST_AUTHORIZED",
            "authority_version": AUTHORITY_VERSION,
            "company_id": self.scope.company_id, "machine_id": self.scope.machine_id,
            "ai_scope": self.scope.ai_scope,
            "document_ids": list(self.scope.document_ids),
            "bubble_document_id": self.scope.bubble_document_id,
            "canonical_evidence_active": False}


def authorize_http_request(payload, *, service_secret: object, application_secret: object,
        principal_id: object, env: Mapping[str, str], resolve: Callable,
        clock: Callable = time.monotonic, opener=None) -> AuthorizedCall:
    if not required(env):
        raise AuthorityError("AUTHORITY_DISABLED")
    old_secret = _config(env, "AI_INTERNAL_SECRET")
    if (type(service_secret) is not str or not service_secret.isascii()
            or not old_secret.isascii() or not hmac.compare_digest(service_secret, old_secret)):
        raise AuthorityError("AUTH_REQUIRED", 401)
    boundary = ApplicationBoundary(secret=_config(env, "MM_APP_AUTHORITY_SECRET"),
        legacy_secret=old_secret, audience=_config(env, "MM_AUTHORITY_AUDIENCE"))
    principal = boundary.authenticate(supplied_secret=application_secret, user_id=principal_id)
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
    authorized = AuthorizedCall(payload, scope, principal, provider, resolve)
    authorized.check(payload)
    return authorized


def protected_call(payload, service_secret, *, authorized: AuthorizedCall,
                   delegate: Callable) -> dict:
    """Check current user/context both before execution and before release.

    Delegate must have its response caches disabled until B4n adds source-safe
    re-use. This function does not claim to authorize legacy citation provenance
    or source revocation hidden inside a still-legacy callback.
    """
    if type(authorized) is not AuthorizedCall or not callable(delegate):
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
    authorized.check(payload)
    result = None
    try:
        result = delegate(payload, service_secret)
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
