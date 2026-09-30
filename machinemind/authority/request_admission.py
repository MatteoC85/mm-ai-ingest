"""Request-local admission windows and external-consumption fences.

An immutable catalog is an admission for in-memory bookkeeping, NOT a cached
permission grant. It never survives an HTTP request. Authoritative Bubble reads
are repeated before external model consumption, response/cache publication and
cache release. The same fixed AuthorityMeter accounts for every remote read.

This separates pure lineage validation (potentially thousands of calls) from
remote reauthorization. No new source allowance is inferred from SQL or model
output. A changed grant is sticky; already-sent model calls cannot be revoked.
"""
from __future__ import annotations
from contextvars import ContextVar
from copy import deepcopy
import re
import math
from .contracts import AuthorityError

_ACTIVE: ContextVar = ContextVar("mm_request_admission", default=None)


def _transport_counters(meter):
    """Optional observation cannot change an authorization decision.

    Production uses AuthorityMeter; trusted offline/legacy directories may
    expose only its limits. Missing instrumentation is explicit, never zero.
    """
    try:
        value = meter.summary()
        keys = ("http_calls", "response_bytes", "elapsed_seconds")
        if any(type(value[k]) not in (int, float) or not math.isfinite(value[k])
                or value[k] < 0 for k in keys):
            return None
        return {k: value[k] for k in keys}
    except Exception:
        return None


class RequestAdmission:
    def __init__(self, *, provider, grant, scope, payload, resolve):
        from .policy import BubbleAuthority
        if type(provider) is not BubbleAuthority or not callable(resolve):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self.provider, self.grant, self.scope = provider, grant, scope
        self.payload, self.resolve = payload, resolve
        self.key = self._key()
        self.active, self.fault = True, None
        self._sources, self._session, self._request = None, None, None
        self.fences = []
        self.catalog_epoch = 0
        self.catalog_reuses = 0
        self.fence_events = []
        self.fence_attempts = self.fence_failures = 0
        self.fence_wall_seconds = 0.0

    def _key(self):
        return deepcopy(tuple(getattr(self.payload, name, None) for name in
            ("company_id", "machine_id", "ai_scope", "document_ids", "bubble_document_id",
             "query", "language", "top_k", "debug")))

    def check(self):
        if self.fault is not None:
            raise self.fault
        if not self.active or self.key != self._key():
            self._fail("AUTHORITY_REQUEST_EXPIRED")

    def _fail(self, code):
        if self.fault is None:
            self.fault = AuthorityError(code, 403)
        raise self.fault

    def sources(self):
        self.check()
        if self._sources is None:
            return self.refresh("request.admission")
        self.catalog_reuses += 1
        return self._sources

    def refresh(self, stage, *, observed=False):
        self.check()
        if not isinstance(stage, str) or not stage or len(stage) > 128:
            self._fail("AUTHORITY_FENCE_INVALID")
        from ..ask.request_completion import record_fence, _CacheWindowExpired
        from ..infrastructure.request_budget import _monotonic, _V13BudgetExceeded
        started = _monotonic()
        meter = self.provider.directory.meter
        before = _transport_counters(meter)
        outcome = "interrupted"
        mode, dependency_count = "full_catalog", 0
        self.fence_attempts += 1
        from ..ask.phase_trace import span
        try:
            with span("authority.fence"):
                if observed and self._session is not None:
                    deps = self._session.cache_dependencies(request=self._request)
                    mode, dependency_count = "observed_sources", len(deps)
                    checked = self.provider.validate_sources(self.grant, self.scope, deps)
                    self.check()
                    if checked != deps:
                        self._fail("AUTHORITY_CONSUMPTION_REVOKED")
                    # Keep the immutable request-admission catalog for internal lineage.
                    # A later source can only reach an external consumer after appearing
                    # in deps and passing this fresh compact fence.
                    sources = self._sources if self._sources is not None else self.provider.current_sources(self.grant, self.scope)
                else:
                    sources = self.provider.current_sources(self.grant, self.scope)
                    self.check()
                    self._sources = sources
                self.fences.append({"stage": stage, "source_count": len(sources),
                    "checked_dependency_count": dependency_count, "fence_mode": mode})
                # A reader may reuse one immutable snapshot only within this
                # epoch; every successful external-consumption fence ends it.
                self.catalog_epoch += 1
                outcome = "completed"
                return sources
        except _CacheWindowExpired:
            outcome = "cache_window_expired"
            raise
        except _V13BudgetExceeded:
            outcome = "deadline"
            raise
        except Exception as exc:
            outcome = "denied" if isinstance(exc, AuthorityError) else "provider_error"
            self.fault = exc if isinstance(exc, AuthorityError) else AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
            raise self.fault from None
        finally:
            elapsed = max(0.0, _monotonic() - started)
            record_fence(elapsed)
            self.fence_wall_seconds += elapsed
            self.fence_failures += int(outcome != "completed")
            after = _transport_counters(meter)
            measured = before is not None and after is not None
            if len(self.fence_events) < 256:
                # Only internal phase labels are public; never serialize
                # provider URLs/bodies, source identifiers or exception text.
                label = stage if (stage in {"request.admission", "cache.admission", "response.publication"}
                    or (stage.startswith("model.") and re.fullmatch(r"model\.[A-Za-z_][A-Za-z0-9_.-]{0,99}", stage))) else "other"
                self.fence_events.append({"stage": label, "outcome": outcome,
                    "fence_mode": mode, "checked_dependency_count": dependency_count,
                    "wall_seconds": round(elapsed, 6),
                    "transport_measured": measured,
                    "http_calls": max(0, after["http_calls"] - before["http_calls"]) if measured else None,
                    "response_bytes": max(0, after["response_bytes"] - before["response_bytes"]) if measured else None,
                    "http_work_seconds": round(max(0.0, after["elapsed_seconds"] - before["elapsed_seconds"]), 6) if measured else None})

    def summary(self):
        return {"catalog_reuse": "request_local_lineage_only", "catalog_reuses": self.catalog_reuses,
            "catalog_epoch": self.catalog_epoch, "remote_fence_attempt_count": self.fence_attempts,
            "remote_fence_failed_count": self.fence_failures,
            "remote_fence_wall_seconds": round(self.fence_wall_seconds, 6),
            "fence_events": [dict(event) for event in self.fence_events],
            "fence_events_truncated": self.fence_attempts > len(self.fence_events),
            "elapsed_seconds_measure": "summed_http_transport_work_not_wall_time"}

    def attach(self, request, session):
        from ..retrieval.ask_composition import AskEvidenceSession
        self.check()
        if self._session is not None or type(session) is not AskEvidenceSession:
            self._fail("AUTHORITY_SESSION_INVALID")
        actual_scope, _ = session.read_contract(request=request, current_allowed_sources=self.sources())
        if actual_scope != self.scope:
            self._fail("AUTHORITY_SCOPE_CHANGED")
        self._session, self._request = session, request

    def before_egress(self, purpose):
        self.check()
        # Query-only embedding before acquisition carries no source evidence.
        if purpose == "embedding" and (self._session is None or not
                self._session.cache_dependencies(request=self._request)):
            return
        self.refresh("model." + str(purpose)[:100], observed=True)

    def close(self):
        self.active = False
        provider = self.provider
        self._sources = self._session = self._request = None
        self.payload = self.provider = self.grant = self.resolve = None
        from .bubble_directory import BubbleDirectory
        if provider is not None and type(provider.directory) is BubbleDirectory:
            provider.directory.close()

    def __repr__(self):
        return "RequestAdmission(<request-owned>)"


def activate(admission):
    if type(admission) is not RequestAdmission or _ACTIVE.get() is not None:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
    admission.check()
    return _ACTIVE.set(admission)


def deactivate(token):
    _ACTIVE.reset(token)


def before_egress(purpose):
    admission = _ACTIVE.get()
    if admission is not None:
        admission.before_egress(purpose)


def current_summary():
    """IO-free accounting snapshot, including a failed active admission.

    Do not call check()/sources(): a terminal denial or deadline must retain
    measurements without obtaining another grant or claiming authorization.
    """
    admission = _ACTIVE.get()
    if type(admission) is not RequestAdmission or not admission.active:
        return {}
    try:
        meter = admission.provider.directory.meter
        counters = meter.summary()
        if type(counters) is not dict:
            counters = {}
        else:
            # Only the established scalar meter schema may leave this owner.
            allowed = {"version", "http_calls", "response_bytes", "failed_http_calls",
                "elapsed_seconds", "grants_cached", "prices_measured"}
            counters = {key: value for key, value in counters.items()
                if key in allowed and type(value) in (str, int, float, bool)
                and (type(value) is not float or math.isfinite(value))}
        return {**counters, **admission.summary(),
            "remote_fence_count": len(admission.fences),
            "admission_policy": "request-local-observed-source-fences-v2"}
    except Exception:
        # Optional telemetry never replaces the original deadline/denial.
        return {}
