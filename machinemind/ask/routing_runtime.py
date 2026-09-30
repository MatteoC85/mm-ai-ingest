"""Protected ASK routing policy and truthful execution diagnostics.

Only the authenticated composition root creates this context. It cannot be
selected with payload metadata; OFF, Root Cause and Smart are unchanged.
The only provider change is low reasoning effort for the ASK semantic router.
Prompts, schema, model order, output caps, retries and deadlines are unchanged.
A successful fallback never settles a previous unknown provider outcome.
"""
from __future__ import annotations
from contextvars import ContextVar
from copy import deepcopy
import math
import re
import time

VERSION = "ask-router-low-accounting-v1"
_ACTIVE = ContextVar("mm_protected_ask_routing_runtime", default=None)


def _safe_name(value):
    return value if (type(value) is str and len(value) <= 128
        and re.fullmatch(r"[A-Za-z0-9_.:-]+", value)) else "unknown"


def _provider_attempts(ledger):
    """Safe accounting/timing projection; no prompts, IDs or exception messages."""
    result = []
    for kind, rows in (("llm", ledger.get("calls", [])), ("embedding", ledger.get("embeddings", []))):
        for row in rows[:32]:
            if type(row) is not dict:
                continue
            item = {"kind": kind, "purpose": _safe_name(row.get("purpose", "embedding")),
                    "model": _safe_name(row.get("model")),
                    "accounting_state": _safe_name(row.get("accounting_state")),
                    "failed": row.get("failed") is True,
                    "dispatched": row.get("dispatched") is True}
            for key in ("started_at_elapsed_seconds", "completed_at_elapsed_seconds",
                        "timeout_seconds", "max_output_tokens", "reserved_input_tokens",
                        "input_tokens", "output_tokens", "reasoning_tokens",
                        "reserved_cost_usd", "estimated_cost_usd"):
                value = row.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    item[key] = value
            start, end = item.get("started_at_elapsed_seconds"), item.get("completed_at_elapsed_seconds")
            if start is not None and end is not None:
                item["reservation_through_outcome_seconds"] = round(max(0.0, end - start), 6)
            result.append(item)
    return result


def accounting_state(meta: dict) -> dict:
    """Classify the actual request ledger; invalid/missing values fail closed.

    All monetary values are runtime configured-price estimates, not invoices.
    Do not mutate the ledger or infer missing usage from another successful call.
    """
    invalid = {"state": "invalid_or_unavailable", "bounded": False,
               "complete": False, "cache_eligible": False}
    if type(meta) is not dict:
        return invalid
    names = ("estimated_cost_usd", "reserved_cost_usd", "uncertain_cost_usd",
             "committed_cost_usd", "max_estimated_cost_usd")
    if any(type(meta.get(n)) not in (int, float) or not math.isfinite(meta[n])
           or meta[n] < 0 for n in names):
        return invalid
    if type(meta.get("accounting_complete")) is not bool:
        return invalid
    if type(meta.get("accounting_anomalies")) is not list:
        return invalid
    known, pending, uncertain, committed, cap = (float(meta[n]) for n in names)
    # public_meta independently rounds the three components to eight decimals.
    if abs(known + pending + uncertain - committed) > 0.00000003:
        return invalid
    bounded = committed <= cap + 1e-12 and not meta["accounting_anomalies"]
    complete = meta["accounting_complete"] and pending == 0 and uncertain == 0
    if meta["accounting_complete"] and not complete:
        return invalid
    state = ("over_limit_or_anomalous" if not bounded else
             "pending" if pending > 0 else
             "bounded_uncertain" if uncertain > 0 or not complete else "complete")
    return {"state": state, "bounded": bounded, "complete": complete,
            "cache_eligible": bounded and complete,
            "known_estimate_usd": known, "pending_reserved_usd": pending,
            "uncertain_reserved_usd": uncertain, "committed_usd": committed,
            "limit_usd": cap}


class ProtectedExecution:
    def __init__(self, started: float):
        self.started = started
        self.ledger = None
        self.active = True

    def record(self, ledger):
        if not self.active:
            raise RuntimeError("PROTECTED_EXECUTION_CLOSED")
        self.ledger = deepcopy(ledger)

    def finish(self, response: dict) -> dict:
        if not self.active or type(response) is not dict:
            raise RuntimeError("PROTECTED_EXECUTION_INVALID")
        ledger = self.ledger or {}
        router = [row for row in ledger.get("calls", [])
                  if type(row) is dict and row.get("purpose") == "assistant_core_v2_semantic_router"]
        # No error messages, prompts, IDs, payloads or credentials in this view.
        attempts = [{"model": str(r.get("model") or ""),
                     "accounting_state": str(r.get("accounting_state") or "unknown"),
                     "failed": r.get("failed") is True,
                     "error_type": str(r.get("error") or ""),
                     "timeout_seconds": r.get("timeout_seconds")}
                    for r in router]
        out = dict(response)
        out["meta"] = {**dict(response.get("meta") or {}), "protected_execution": {
            "version": VERSION, "router_effort": "low",
            "elapsed_sync_seconds": round(max(0.0, time.monotonic() - self.started), 3),
            "timing_scope": "protected_sync_entry_to_after_final_authority_checks",
            "accounting": accounting_state(ledger), "router_attempts": attempts,
            "provider_attempts": _provider_attempts(ledger),
            "router_fallback_used": len(router) > 1,
            "end_user_privacy_certified": False}}
        return out


def now():
    return time.monotonic()


def activate(started=None):
    if _ACTIVE.get() is not None:
        raise RuntimeError("PROTECTED_EXECUTION_ALREADY_ACTIVE")
    execution = ProtectedExecution(time.monotonic() if started is None else started)
    return execution, _ACTIVE.set(execution)


def deactivate(token):
    execution = _ACTIVE.get()
    if execution is not None:
        execution.active = False
        execution.ledger = None
    _ACTIVE.reset(token)


def router_effort(requested_mode, configured):
    execution = _ACTIVE.get()
    if execution is not None and execution.active and requested_mode == "ask":
        return "low"
    return configured


def record_ledger(ledger):
    execution = _ACTIVE.get()
    if execution is not None:
        execution.record(ledger)
