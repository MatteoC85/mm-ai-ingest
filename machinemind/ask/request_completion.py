"""Protected ASK completion lifetime: one ledger through final publication.

Only the trusted ASK composition root activates this owner. OFF/RC do not use it.
The optional cache write has a child time window; expiry never authorizes content
or a cache entry. Final publication still performs its own current authority
check after any store. All child workers are joined before that check.

Cancellation is cooperative: blocking transports/OS scheduling are not preempted
by Python. A late result is vetoed, never labelled ANSWERED within the deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from typing import Any
from . import phase_trace

def _monotonic() -> float:
    # The completion observer can be imported by pure request-bound Core tests.
    # Load infrastructure only when a protected lifetime actually asks for time.
    from ..infrastructure.request_budget import _monotonic as clock
    return clock()

VERSION = "ask-completion-through-publication-v1"
PHASE6_CANDIDATE_VERSION = "phase6-closure-2026-09-29-v1"
_ACTIVE = ContextVar("mm_ask_request_completion", default=None)
_CACHE_WINDOW = ContextVar("mm_ask_optional_cache_window", default=None)
_REVIEW_RESERVE = ContextVar("mm_ask_synthesis_review_reserve", default=None)
_SYNTHESIS_PURPOSES = frozenset({"ask_structured_synthesis", "ask_final_synthesis",
    "assistant_core_machine_overview_inventory"})
_REVIEW_PURPOSE = "assistant_core_answer_contract_verifier"
_KNOWN_PURPOSES = _SYNTHESIS_PURPOSES | {_REVIEW_PURPOSE,
    "assistant_core_v2_semantic_router", "assistant_core_general_technical_answer", "embedding"}


class _CacheWindowExpired(BaseException):
    """Internal cancellation, NOT a failed permission check.

    Must cross legacy broad Exception handlers without becoming a sticky
    authority-provider failure or a successful cache write. Caught only by the
    request owner's optional_cache boundary after all workers have joined.
    """


@dataclass(frozen=True, slots=True)
class _Window:
    owner: Any
    deadline: float


def current():
    owner = _ACTIVE.get()
    return owner if type(owner) is RequestCompletion and owner.active else None


def check_io_time() -> None:
    owner = current()
    if owner is None:
        return
    window = _CACHE_WINDOW.get()
    if type(window) is _Window and window.owner is owner:
        if _monotonic() >= window.deadline:
            raise _CacheWindowExpired("optional cache deadline reached")
    owner.budget.ensure_time(0.0)


def transport_timeout(configured: float) -> float:
    """Keep existing transport/accounting ceilings; only clamp to live time."""
    owner = current()
    if owner is None:
        return configured
    check_io_time()
    remaining = owner.budget.remaining()
    window = _CACHE_WINDOW.get()
    if type(window) is _Window and window.owner is owner:
        remaining = min(remaining, window.deadline - _monotonic())
    value = min(float(configured), remaining)
    if not math.isfinite(value) or value <= 0:
        check_io_time()
        from ..infrastructure.request_budget import _V13BudgetExceeded
        raise _V13BudgetExceeded("protected request deadline before transport")
    return value


def record_fence(seconds: float) -> None:
    owner = current()
    if owner is not None and type(seconds) in (int, float) and math.isfinite(seconds):
        owner.largest_fence_seconds = max(owner.largest_fence_seconds, float(seconds), 0.0)


@contextmanager
def synthesis_review_reserve(*, required: bool):
    """Reserve the EXISTING nine-second review admission threshold, not a SLA.

    Applied only to synthesis dispatch purposes. A nested recovery verifier can
    use its reserved window without inheriting a synthesis child deadline.
    """
    owner = current()
    if owner is None:
        yield
        return
    token = _REVIEW_RESERVE.set((owner, bool(required)))
    try:
        yield
    finally:
        _REVIEW_RESERVE.reset(token)


def synthesize_observed(callback, review_required):
    """Injected observer; querying the review policy is protected-request-only."""
    if current() is None:
        return callback()
    with synthesis_review_reserve(required=bool(review_required())):
        return callback()


def provider_timeout(purpose: str, configured: int) -> int:
    owner = current()
    return configured if owner is None else owner.provider_timeout(purpose, configured)


class RequestCompletion:
    """One explicit outer owner, never selected by client request metadata."""
    def __init__(self, *, runtime, payload, started: float):
        from .request_flow import RequestFlowRuntime
        if type(runtime) is not RequestFlowRuntime or type(started) not in (int, float) or not math.isfinite(started):
            raise TypeError("typed runtime and finite protected entry time required")
        if current() is not None:
            raise RuntimeError("ASK_COMPLETION_ALREADY_ACTIVE")
        self.runtime, self.payload = runtime, payload
        self.budget = runtime._assistant_core_new_budget("ask", company_id=str(payload.company_id or ""))
        # The same original duration, now starting at protected entry. Never
        # extend a shorter budget supplied by the application or outer control.
        start = min(float(started), self.budget.started_monotonic)
        self.budget.started_monotonic = start
        self.budget.deadline_monotonic = min(self.budget.deadline_monotonic,
                                             start + float(self.budget.deadline_seconds))
        self.started = start
        self.active = True
        self._budget_token = runtime._V13_BUDGET_CTX.set(self.budget)
        self._owner_token = _ACTIVE.set(self)
        self.trace = phase_trace.PhaseTrace(started=start, clock=_monotonic)
        self._trace_token = phase_trace.activate(self.trace)
        self._terminal = None
        self.largest_fence_seconds = 0.0
        self.core_finished = None
        self.publication_started = self.publication_finished = None
        self.cache_state = "not_attempted"
        self.cache_seconds = 0.0
        self.publication_reserve = 0.0
        self.cache_allowance = 0.0
        self.cache_committed = False
        self.provider_schedules = []

    def measure(self, name, callback, /, *args, **kwargs):
        if not self.active or current() is not self:
            raise RuntimeError("ASK_COMPLETION_EXPIRED")
        return phase_trace.call(name, callback, *args, **kwargs)

    def provider_timeout(self, purpose, configured):
        self.budget.ensure_time(0.0)
        # Observed request-local fence wall time plus existing local DB/admission
        # allowance. The floor reserves three seconds before the first fence.
        # This is a scheduling estimate; fresh final authority still decides.
        publication = max(1.5, self.largest_fence_seconds) + 1.5
        review = 0.0
        planned = _REVIEW_RESERVE.get()
        if (purpose in _SYNTHESIS_PURPOSES and planned is not None
                and planned[0] is self and planned[1]):
            review = 9.0 + max(1.5, self.largest_fence_seconds)
        available = self.budget.remaining() - publication - review
        minimum = 9.0 if purpose == _REVIEW_PURPOSE else 1.0
        timeout = min(int(configured), math.floor(available))
        row = {"purpose": purpose if purpose in _KNOWN_PURPOSES else "other",
               "at_seconds": round(self.budget.elapsed(), 6),
               "publication_reserve_seconds": round(publication, 6),
               "review_reserve_seconds": round(review, 6),
               "remaining_seconds": round(self.budget.remaining(), 6),
               "timeout_seconds": max(0, timeout), "admitted": timeout >= minimum}
        # At most six LLM calls and bounded embeddings; don't let diagnostics
        # create an unbounded payload if a caller violates those assumptions.
        if len(self.provider_schedules) < 64:
            self.provider_schedules.append(row)
        self.publication_reserve = max(self.publication_reserve, publication)
        if timeout < minimum:
            from ..infrastructure.request_budget import _V13BudgetExceeded
            raise _V13BudgetExceeded("protected ASK deadline lacks time for provider and required finalization")
        return timeout

    def budget_for(self, payload, runtime):
        if (not self.active or current() is not self or payload is not self.payload
                or runtime._V13_BUDGET_CTX is not self.runtime._V13_BUDGET_CTX
                or runtime._V13_BUDGET_CTX.get() is not self.budget):
            raise RuntimeError("ASK_COMPLETION_OWNERSHIP_INVALID")
        return self.budget

    def core_done(self):
        self.budget.ensure_time(0.0)
        self.core_finished = self.budget.elapsed()

    def mark_terminal(self, response):
        if (type(response) is not dict or response.get("result_code") not in {"TIMEOUT", "BUDGET_EXCEEDED"}
                or response.get("citations") or response.get("rg_links")):
            raise RuntimeError("ASK_COMPLETION_TERMINAL_INVALID")
        self._terminal = deepcopy(response)
        return response

    def timeout_response(self, exc):
        p = self.payload
        result = self.runtime._assistant_core_budget_response(
            requested_mode="ask", q=str(p.query or "").strip(),
            language=self.runtime._select_response_language(str(p.query or "").strip(), preferred=p.language),
            top_k=max(1, min(int(p.top_k or 5), self.runtime.ASK_MAX_TOP_K)),
            budget=self.budget, exc=exc)
        return self.mark_terminal(result)

    def publish(self, response, guard):
        if self._terminal is not None and response == self._terminal:
            # Only the original server-produced content-free terminal envelope
            # can avoid new source I/O after cancellation, never an arbitrary
            # success/body with a client-controlled timeout flag.
            return deepcopy(response)
        if type(response) is dict and response.get("ok") is False and response.get("status") == "error":
            return guard(response)  # original error sanitization, no content
        self.budget.ensure_time(0.0)
        self.publication_started = self.budget.elapsed()
        with phase_trace.span("flow.publication"):
            result = guard(response)
        self.budget.ensure_time(0.0)
        self.publication_finished = self.budget.elapsed()
        return result

    @contextmanager
    def optional_cache(self, *, authority_timeout: float):
        from ..infrastructure.request_budget import _REQUEST_CONTROL_CTX, _RequestControl
        self.budget.ensure_time(0.0)
        if _CACHE_WINDOW.get() is not None:
            raise RuntimeError("ASK_COMPLETION_NESTED_CACHE")
        if type(authority_timeout) not in (int, float) or not math.isfinite(authority_timeout) or authority_timeout <= 0:
            raise TypeError("configured authority timeout required")
        # Reserve the larger of the existing GET ceiling and the largest fresh
        # fence already observed in THIS request, plus the existing DB-connect
        # minimum. This is scheduling, not a grant or a statistical SLA. The final
        # guard is always checked and a late answer is still vetoed.
        reserve = max(float(authority_timeout), self.largest_fence_seconds) + 1.5
        allowance = max(0.0, self.budget.remaining() - reserve)
        self.publication_reserve, self.cache_allowance = reserve, allowance
        if allowance < float(authority_timeout):
            self.cache_state = "skipped_time_reserved_for_publication"
            yield False
            return
        began = _monotonic()
        window = _Window(self, began + allowance)
        wt = _CACHE_WINDOW.set(window)
        control = _RequestControl(allowance, allow_llm=False, parent=_REQUEST_CONTROL_CTX.get())
        ct = _REQUEST_CONTROL_CTX.set(control)
        self.cache_state = "attempted"
        try:
            yield True
            check_io_time()
            self.cache_state = "committed" if self.cache_committed else "not_stored"
        except _CacheWindowExpired:
            self.cache_state = "committed_at_window_edge" if self.cache_committed else "skipped_cache_window_expired"
        finally:
            control.cancel()
            _REQUEST_CONTROL_CTX.reset(ct)
            _CACHE_WINDOW.reset(wt)
            self.cache_seconds += max(0.0, _monotonic() - began)
        self.budget.ensure_time(0.0)

    def cache_connection(self, connect, *, entry_write=True):
        check_io_time()
        conn = connect()
        try:
            check_io_time()
            return _CacheConnection(conn, self, entry_write=entry_write)
        except BaseException:
            conn.close()
            raise

    def finish(self, response):
        if type(response) is not dict:
            raise TypeError("response mapping required")
        if str(response.get("status") or "").lower() == "answered":
            self.budget.ensure_time(0.0)
        result = self.runtime._assistant_core_attach_runtime_meta(
            response, self.budget, debug=bool(getattr(self.payload, "debug", False)))
        commit = os.environ.get("COMMIT_SHA", "").lower()
        commit = commit if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) else None
        result["meta"] = {**dict(result.get("meta") or {}), "request_completion": {
            "version": VERSION, "timing_scope": "protected_entry_through_publication",
            "elapsed_seconds": round(self.budget.elapsed(), 6),
            "deadline_seconds": float(self.budget.deadline_seconds),
            "remaining_seconds": round(self.budget.remaining(), 6),
            "core_finished_seconds": self.core_finished,
            "publication_started_seconds": self.publication_started,
            "publication_finished_seconds": self.publication_finished,
            "cache_write": self.cache_state, "cache_write_seconds": round(self.cache_seconds, 6),
            "publication_reserve_seconds": round(self.publication_reserve, 6),
            "cache_allowance_seconds": round(self.cache_allowance, 6),
            "cache_commit_observed": self.cache_committed,
            "late_success_forbidden": True,
            "provider_scheduling": list(self.provider_schedules),
            "reserve_basis": "observed_fence_wall_plus_local_allowance_not_SLA",
        }, "ask_phase_trace": self.trace.summary(), "runtime_commit_sha": commit,
            "phase6_candidate_version": PHASE6_CANDIDATE_VERSION}
        # Read-only diagnostics must survive timeout/revocation. Never trigger
        # fresh authority I/O here or turn a fault snapshot into scope approval.
        from ..authority.request_admission import current_summary
        authority = current_summary()
        if authority is not None:
            existing = result["meta"].get("request_authority")
            result["meta"]["request_authority"] = {
                **authority, **(existing if type(existing) is dict else {})}
        if str(result.get("status") or "").lower() == "answered":
            # Metadata/trace serialization belongs to this request too. A
            # callback or scheduler pause after the entry check cannot release
            # an answer that the same clock already knows is late.
            self.budget.ensure_time(0.0)
        return result

    def close(self):
        if not self.active:
            return
        self.active = False
        try:
            self.runtime._V13_BUDGET_CTX.reset(self._budget_token)
        finally:
            try:
                phase_trace.deactivate(self._trace_token)
            finally:
                _ACTIVE.reset(self._owner_token)
                self._terminal = self.payload = self.runtime = None


class _CacheConnection:
    """Keep the original SQL/transaction; check the child window between steps."""
    def __init__(self, conn, owner, *, entry_write=True):
        self._conn, self._owner = conn, owner
        self._entry_write = entry_write
        self._committed = False
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def cursor(self, *args, **kwargs):
        check_io_time()
        return _CacheCursor(self._conn.cursor(*args, **kwargs))
    def commit(self):
        check_io_time()
        self._conn.commit()
        self._committed = True
        if self._entry_write:
            self._owner.cache_committed = True
        check_io_time()
    def rollback(self):
        return self._conn.rollback()
    def close(self):
        try:
            if not self._committed:
                self._conn.rollback()
        finally:
            self._conn.close()


class _CacheCursor:
    def __init__(self, cursor):
        self._cursor = cursor
    def __getattr__(self, name):
        return getattr(self._cursor, name)
    def __enter__(self):
        self._cursor.__enter__()
        return self
    def __exit__(self, *args):
        return self._cursor.__exit__(*args)
    def execute(self, *args, **kwargs):
        check_io_time()
        value = self._cursor.execute(*args, **kwargs)
        check_io_time()
        return value
