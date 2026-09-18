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
from .contracts import AuthorityError

_ACTIVE: ContextVar = ContextVar("mm_request_admission", default=None)


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
        return self._sources

    def refresh(self, stage, *, observed=False):
        self.check()
        if not isinstance(stage, str) or not stage or len(stage) > 128:
            self._fail("AUTHORITY_FENCE_INVALID")
        try:
            if observed and self._session is not None:
                deps = self._session.cache_dependencies(request=self._request)
                checked = self.provider.validate_sources(self.grant, self.scope, deps)
                self.check()
                if checked != deps:
                    self._fail("AUTHORITY_CONSUMPTION_REVOKED")
                # Keep the immutable request-admission catalog for internal lineage.
                # A later source can only reach an external consumer after appearing
                # in deps and passing this fresh compact fence.
                sources = self._sources if self._sources is not None else self.provider.current_sources(self.grant, self.scope)
                self.fences.append({"stage": stage, "source_count": len(sources),
                    "checked_dependency_count": len(deps), "fence_mode": "observed_sources"})
                return sources
            sources = self.provider.current_sources(self.grant, self.scope)
            self.check()
            self._sources = sources
            self.fences.append({"stage": stage, "source_count": len(sources),
                "checked_dependency_count": 0, "fence_mode": "full_catalog"})
            return sources
        except Exception as exc:
            self.fault = exc if isinstance(exc, AuthorityError) else AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
            raise self.fault from None

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
