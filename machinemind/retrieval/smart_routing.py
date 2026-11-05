"""Bounded Smart-only semantic routing; no deterministic evidence admission.

The caller retains the existing router prompt/schema and provider accounting.
An unavailable router raises a fixed, non-sensitive exception so Core's existing
degraded-decision path stops before refinement, evidence preparation or synthesis.
The Smart adapter must project that path as a technical failure, not no sources.
"""
from __future__ import annotations
import math


POLICY_VERSION = "smart-router-single-attempt-v1"
EFFORT_ENV = "MM_ASSISTANT_CORE_SMART_ROUTER_EFFORT"
FAILURE_REASON = "smart_semantic_router_unavailable"
FAILURE_CODE = "SMART_DIAGNOSTIC_ROUTER_FAILED"
MAX_ROUTER_WAIT_SECONDS = 30
MIN_ROUTER_WAIT_SECONDS = 8


class SmartRouterUnavailable(RuntimeError):
    """Do not embed the provider error, request, source text or configuration."""

    def __init__(self):
        super().__init__(FAILURE_REASON)


def attempt_plan(primary_model: str, *, effort_override: str | None = None) -> dict:
    """Keep the configured primary model and an explicit Smart effort override.

    No fallback model, timeout, cost/call allowance or source contract is changed
    here. The caller applies this plan only to the fixed Smart endpoint. Effort is
    passed through consistently with existing provider configuration semantics.
    """
    if not isinstance(primary_model, str) or not primary_model.strip():
        raise ValueError("Smart semantic router requires its configured primary model")
    if effort_override is not None and not isinstance(effort_override, str):
        raise ValueError("Smart semantic router effort must be configured as text")
    return {
        "policy_version": POLICY_VERSION,
        "models": [primary_model.strip()],
        "effort": (effort_override or "").strip() or "low",
        "attempt_limit": 1,
    }


def allocated_timeout(remaining_seconds: float, *, downstream_reserve_seconds: float) -> int:
    """One router wait within the existing request deadline, reserving later work.

    A zero result means no useful dispatch is admitted. Monetary, output and
    call limits remain with the native budget; this function only allocates time.
    """
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (remaining_seconds, downstream_reserve_seconds)):
        raise ValueError("Smart router time allocation must be finite and nonnegative")
    available = math.floor(remaining_seconds - downstream_reserve_seconds)
    return min(MAX_ROUTER_WAIT_SECONDS, available) if available >= MIN_ROUTER_WAIT_SECONDS else 0
