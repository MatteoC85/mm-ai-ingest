"""Bounded Smart-only semantic routing; no deterministic evidence admission.

The caller retains the existing router prompt/schema and provider accounting.
An unavailable router raises a fixed, non-sensitive exception so Core's existing
degraded-decision path stops before refinement, evidence preparation or synthesis.
The Smart adapter must project that path as a technical failure, not no sources.
"""
from __future__ import annotations


POLICY_VERSION = "smart-router-single-attempt-v1"
EFFORT_ENV = "MM_ASSISTANT_CORE_SMART_ROUTER_EFFORT"
FAILURE_REASON = "smart_semantic_router_unavailable"
FAILURE_CODE = "SMART_DIAGNOSTIC_ROUTER_FAILED"


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
