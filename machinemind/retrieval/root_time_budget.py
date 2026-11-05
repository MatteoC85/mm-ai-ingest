"""Root-only time reservations within the existing request and cost ledger.

Optional retrieval uses a synchronous child deadline. Completed evidence keeps
the existing admission/scoring rules; a time limit never grants causal support.
No worker is detached and no provider attempt or cost allowance is added.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from machinemind.infrastructure import request_budget as native

POLICY_VERSION = "root-time-reservation-v1"
FINALIZATION_RESERVE_SECONDS = 4.0
MIN_PRODUCER_SECONDS = 12.0
_WINDOW = ContextVar("machinemind_root_time_window", default=None)


class RootWindowExpired(native._V13BudgetExceeded):
    def __init__(self):
        super().__init__("root_optional_retrieval_deadline")


@dataclass(frozen=True)
class Window:
    deadline: float
    budget: Any
    optional: bool

    def remaining(self) -> float:
        return max(0.0, min(self.deadline - native._monotonic(), float(self.budget.remaining())))

    def check(self, minimum_seconds: float = 0.05) -> None:
        if self.remaining() <= 0 or self.remaining() < minimum_seconds:
            raise RootWindowExpired()

    def timeout_ms(self, configured: int) -> int:
        if type(configured) is not int or configured <= 0:
            raise ValueError("Root SQL timeout must be a positive integer")
        self.check(0.05)
        milliseconds = min(configured, math.floor((self.remaining() - 0.05) * 1000))
        if milliseconds < 1:
            raise RootWindowExpired()
        return milliseconds


def current_window():
    return _WINDOW.get()


def optional_transport_timeout(configured: int) -> int:
    """Bound optional connect/read waits immediately before existing dispatch.

    The scalar Requests timeout applies independently to connect and read. Leave
    room for both inside this optional slot. It is not a hard wall-clock bound
    against an indefinitely streaming peer; the request's outer guard remains.
    """
    window = current_window()
    if window is None or not window.optional:
        return configured
    window.check(3.0)
    timeout = min(configured, math.floor((window.remaining() - 1.0) / 2.0))
    if timeout < 1:
        raise RootWindowExpired()
    return timeout


def _root_request(request, decision) -> bool:
    return (getattr(request, "requested_mode", None) == "root_cause"
            and getattr(decision, "effective_mode", None) == "root_cause")


def _seconds(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Root stage timeout must be positive and finite")
    return float(value)


def reserved_final_seconds(*, producer_timeout, review_timeout) -> float:
    return _seconds(producer_timeout) + _seconds(review_timeout) + FINALIZATION_RESERVE_SECONDS


@contextmanager
def _window(budget, seconds, *, optional):
    # Never modify the native request deadline, call allowance or cost ledger.
    control, token = native._v13_push_operation_limits(seconds=seconds, allow_llm=not optional)
    window = Window(control.deadline, budget, optional)
    window_token = _WINDOW.set(window)
    try:
        yield window
    finally:
        _WINDOW.reset(window_token)
        native._v13_pop_operation_limits(token)


def _annotate(retrieval, stage, status, reserve, allowance):
    prior = dict(retrieval.get("assistant_core_root_time_budget") or {})
    stages = list(prior.get("stages") or [])
    stages.append({"stage": stage, "status": status,
                   "reserved_final_seconds": reserve,
                   "optional_allowance_seconds": round(max(0.0, allowance), 6)})
    return {**retrieval, "assistant_core_root_time_budget": {
        "policy_version": POLICY_VERSION, "stages": stages,
        "evidence_admission_unchanged": True}}


def optional_retrieval(*, request, decision, retrieval, budget,
                       producer_timeout, review_timeout, stage, run):
    if not _root_request(request, decision) or budget is None:
        return run(retrieval)
    if stage not in {"facet_refinement", "diagnostic_assurance"}:
        raise ValueError("Unknown optional Root stage")
    reserve = reserved_final_seconds(producer_timeout=producer_timeout, review_timeout=review_timeout)
    allowance = float(budget.remaining()) - reserve
    # Preserve the existing entry requirements INSIDE the optional allowance.
    minimum = 18.0 if stage == "facet_refinement" else 12.0
    if allowance < minimum:
        return _annotate(retrieval, stage, "skipped_reserved_time", reserve, allowance)
    exhausted = False
    try:
        with _window(budget, allowance, optional=True) as window:
            candidate = deepcopy(retrieval)
            window.check(minimum_seconds=minimum)
            result = run(candidate)
            exhausted = window.remaining() <= 0
    except RootWindowExpired:
        # The original baseline is untouched even if a partial stage mutated its
        # private working copy. Cost/authority failures are not caught here.
        if budget.remaining() <= 0:
            raise
        return _annotate(retrieval, stage, "optional_deadline_reached", reserve, allowance)
    if not isinstance(result, dict):
        raise TypeError("Root retrieval must return its existing dictionary contract")
    return _annotate(result, stage,
                     "optional_deadline_reached" if exhausted else "completed",
                     reserve, allowance)


@contextmanager
def producer_window(*, request, decision, budget, review_timeout):
    if not _root_request(request, decision) or budget is None:
        yield
        return
    reserve = _seconds(review_timeout) + FINALIZATION_RESERVE_SECONDS
    allowance = float(budget.remaining()) - reserve
    if allowance < MIN_PRODUCER_SECONDS:
        # Do not buy a short producer attempt that cannot leave a review window.
        raise native._V13BudgetExceeded("root_producer_deadline_reserve")
    with _window(budget, allowance, optional=False):
        yield
