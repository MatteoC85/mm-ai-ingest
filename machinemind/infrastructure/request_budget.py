"""Request-local estimated-cost reservations shared by ASK, Root Cause and Smart.

The ledger guards *configured-price exposure*, not an external billing account. Input
reservation uses a conservative UTF-8 text envelope, including the JSON schema; it is
not a tokenizer or a guarantee about provider billing. Reported usage always wins. An
underestimate is visible, never clamped, and closes further admission. Pending and
unknown provider outcomes remain committed; a failed parse is not a free request.
"""
from __future__ import annotations

import contextvars
import json
import math
import threading
import time as _time
from collections.abc import Mapping
from typing import Any, Optional

from machinemind.config import assistant_runtime as _assistant_defaults

BUDGET_POLICY_VERSION = _assistant_defaults.V13_BUDGET_POLICY_VERSION
_RUNTIME_GLOBALS: Mapping[str, Any] = {}


def configure_request_budget_runtime(runtime_globals: Mapping[str, Any]) -> None:
    global _RUNTIME_GLOBALS
    _RUNTIME_GLOBALS = runtime_globals


def _runtime_value(name: str, default: Any = None) -> Any:
    if name in _RUNTIME_GLOBALS:
        return _RUNTIME_GLOBALS[name]
    return getattr(_assistant_defaults, name, default)


def _monotonic() -> float:
    clock = _RUNTIME_GLOBALS.get("time_module")
    return float(clock.monotonic() if clock is not None and callable(getattr(clock, "monotonic", None)) else _time.monotonic())


class _V13BudgetExceeded(RuntimeError):
    pass


class _RequestControl:
    """Cooperative cancellation shared with copied worker contexts; no thread kill."""
    def __init__(self, seconds: float, *, allow_llm: bool = True, parent=None):
        value = float(seconds)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid operation deadline")
        self.deadline = _monotonic() + value
        self.allow_llm = bool(allow_llm)
        self.parent = parent
        self.stopped = threading.Event()

    def remaining(self) -> float:
        if self.stopped.is_set():
            return 0.0
        own = max(0.0, self.deadline - _monotonic())
        return min(own, self.parent.remaining()) if self.parent is not None else own

    def permits_llm(self) -> bool:
        return self.allow_llm and (self.parent is None or self.parent.permits_llm())

    def cancel(self) -> None:
        self.stopped.set()


_REQUEST_CONTROL_CTX = contextvars.ContextVar("machinemind_request_control", default=None)


def _v13_push_operation_limits(*, seconds: float, allow_llm: bool = True):
    control = _RequestControl(seconds, allow_llm=allow_llm, parent=_REQUEST_CONTROL_CTX.get())
    return control, _REQUEST_CONTROL_CTX.set(control)


def _v13_pop_operation_limits(token) -> None:
    _REQUEST_CONTROL_CTX.reset(token)


def _positive_rate(value: Any) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate < 0:
        raise _V13BudgetExceeded("Invalid configured token price")
    return rate


def _usage_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Usage is not a non-negative integer")
    return value


def _text_input_bound(messages: list[dict], request_payload: Optional[dict] = None) -> tuple[int, int]:
    # This contract is for inline TEXT only. Remote files/tools/images/audio must not
    # be priced by their short URL. Their provider needs a separate metering contract.
    for msg in messages or []:
        if not isinstance(msg, dict):
            raise _V13BudgetExceeded("Unmetered message type")
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in {"text", "input_text"} or not isinstance(item.get("text"), str):
                    raise _V13BudgetExceeded("Non-text provider input needs explicit metering")
        elif not isinstance(content, str):
            raise _V13BudgetExceeded("Non-text provider input needs explicit metering")
    payload = request_payload if request_payload is not None else {"input": messages}
    if any(payload.get(key) for key in ("tools", "previous_response_id", "conversation", "prompt")):
        raise _V13BudgetExceeded("Remote/provider-managed context needs explicit metering")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # A byte-level bound for inline content, plus a deliberately explicit envelope
    # allowance. It includes schema and message metadata; cache discounts are not
    # assumed before execution. Usage violations remain visible as ledger breaches.
    bound = max(1, len(serialized.encode("utf-8")) + 1024 + 128 * len(messages or []))
    approx = max(1, math.ceil(len(serialized) / 3.0))
    return bound, approx


class _V13RequestBudget:
    def __init__(self, mode: str):
        self.mode = str(mode or "ask").strip().lower()
        self.started_monotonic = _monotonic()
        root = self.mode == "root_cause"
        self.deadline_seconds = _runtime_value("V13_ROOT_CAUSE_DEADLINE_SECONDS" if root else "V13_ASK_DEADLINE_SECONDS")
        self.deadline_monotonic = self.started_monotonic + float(self.deadline_seconds)
        self.max_llm_calls = _runtime_value("V13_MAX_LLM_CALLS_ROOT_CAUSE" if root else "V13_MAX_LLM_CALLS_ASK")
        self.base_max_llm_calls = int(self.max_llm_calls)
        self.absolute_max_llm_calls = min(6, int(self.max_llm_calls) + 2)
        self.retry_allowance_calls = 0
        self.retry_events: list[dict] = []
        self.max_estimated_cost_usd = _runtime_value("V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD" if root else "V13_MAX_ESTIMATED_COST_ASK_USD")
        self.llm_calls = 0
        self.estimated_cost_usd = 0.0
        self.input_tokens = self.cached_input_tokens = self.cache_write_tokens = 0
        self.output_tokens = self.reasoning_tokens = 0
        self.embedding_calls = self.embedding_input_tokens = self.embedding_cache_hits = 0
        self.embedding_estimated_cost_usd = 0.0
        self.call_log: list[dict] = []
        self.embedding_log: list[dict] = []
        self.embedding_cache: dict[tuple[str, str], list[float]] = {}
        self.route = "unselected"
        self.refinement_used = False
        self.semantic_cache = "miss"
        self.evidence_gate: dict = {}
        self.retrieval_assurance: dict = {}
        self._lock = threading.RLock()
        self._control = _REQUEST_CONTROL_CTX.get()
        self._cancelled = threading.Event()
        self._retry_credited: set[int] = set()
        self.accounting_anomalies: list[str] = []

    def elapsed(self) -> float:
        return max(0.0, _monotonic() - self.started_monotonic)

    def remaining(self) -> float:
        if self._cancelled.is_set():
            return 0.0
        remaining = max(0.0, self.deadline_monotonic - _monotonic())
        for control in (self._control, _REQUEST_CONTROL_CTX.get()):
            if control is not None:
                remaining = min(remaining, control.remaining())
        return remaining

    def cancel(self) -> None:
        self._cancelled.set()

    def ensure_time(self, minimum_seconds: float = 2.0) -> None:
        minimum = max(0.0, float(minimum_seconds))
        if self.remaining() <= 0.0 or self.remaining() < minimum:
            raise _V13BudgetExceeded(f"V13 {self.mode} deadline/cancellation reached after {self.elapsed():.2f}s")

    def _rows(self):
        return self.call_log + self.embedding_log

    def _liability(self, state: str) -> float:
        return sum(float(row["reserved_cost_usd"]) for row in self._rows() if row.get("accounting_state") == state)

    @property
    def committed_cost_usd(self) -> float:
        with self._lock:
            return self.estimated_cost_usd + self._liability("pending") + self._liability("uncertain")

    @property
    def remaining_cost_usd(self) -> float:
        return max(0.0, float(self.max_estimated_cost_usd) - self.committed_cost_usd)

    def _admission(self) -> None:
        self.ensure_time(2.0)
        limit = float(self.max_estimated_cost_usd)
        if not math.isfinite(limit) or limit < 0:
            raise _V13BudgetExceeded("Invalid configured request cost limit")
        if self.accounting_anomalies or self.committed_cost_usd >= limit:
            raise _V13BudgetExceeded(f"V13 {self.mode} committed cost budget exhausted ({self.committed_cost_usd:.6f}/{limit:.6f} USD)")

    def _timeout(self, requested: int) -> int:
        # Never round UP to five seconds when only four seconds remain.
        value = 30 if requested is None else int(requested)
        if value <= 0:
            raise _V13BudgetExceeded("Invalid provider timeout")
        timeout = min(value, int(self.remaining() - 1.0))
        if timeout < 1:
            raise _V13BudgetExceeded("Insufficient deadline for a provider attempt")
        return timeout

    def reserve_call(self, *, model: str, purpose: str, requested_timeout: int,
                     max_output_tokens: int, messages: list[dict], request_payload: Optional[dict] = None) -> tuple[int, int, int]:
        with self._lock:
            self._admission()
            for control in (self._control, _REQUEST_CONTROL_CTX.get()):
                if control is not None and not control.permits_llm():
                    raise _V13BudgetExceeded("LLM calls prohibited in retrieval-only operation")
            if self.llm_calls >= self.max_llm_calls:
                raise _V13BudgetExceeded(f"V13 {self.mode} LLM call budget exhausted ({self.llm_calls}/{self.max_llm_calls})")
            bound, approx = _text_input_bound(messages, request_payload)
            input_rate, output_rate = (_positive_rate(v) for v in _v13_model_rates(model))
            # Preserve the existing configured cache-write premium (1.25). Reserve
            # worst-case input; settle actual cache details only when usage arrives.
            input_cost = bound * input_rate * 1.25 / 1_000_000.0
            requested = 2000 if max_output_tokens is None else int(max_output_tokens)
            if requested <= 0:
                raise _V13BudgetExceeded("Invalid output token ceiling")
            available = self.remaining_cost_usd - input_cost
            cap = min(requested, max(0, math.floor(available * 1_000_000.0 / output_rate))) if output_rate else (requested if available >= 0 else 0)
            if cap < min(800, requested):
                raise _V13BudgetExceeded(f"V13 {self.mode} insufficient reserved cost for useful output")
            timeout = self._timeout(requested_timeout)
            amount = input_cost + cap * output_rate / 1_000_000.0
            if self.committed_cost_usd + amount > float(self.max_estimated_cost_usd) + 1e-12:
                raise _V13BudgetExceeded("Reservation would exceed configured budget")
            self.llm_calls += 1
            row = {"call": self.llm_calls, "model": str(model or ""), "purpose": str(purpose or "reasoning"),
                   "timeout_seconds": timeout, "max_output_tokens": cap, "approx_input_tokens": approx,
                   "reserved_input_tokens": bound, "reserved_cost_usd": amount,
                   "accounting_state": "pending", "dispatched": False,
                   "input_rate": input_rate, "output_rate": output_rate,
                   "pricing_basis": ("configured_model_rates" if any(name in str(model).lower() for name in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")) else "configured_sol_fallback_not_verified_tariff"), "started_at_elapsed_seconds": round(self.elapsed(), 3)}
            self.call_log.append(row)
            return timeout, cap, self.llm_calls

    def _row(self, index: int, *, embedding: bool = False) -> dict:
        rows = self.embedding_log if embedding else self.call_log
        key = "embedding_call" if embedding else "call"
        for row in rows:
            if row[key] == index:
                return row
        raise ValueError("Unknown provider reservation")

    def mark_dispatched(self, call_index: int, *, embedding: bool = False) -> None:
        with self._lock:
            self.ensure_time(0.0)
            row = self._row(call_index, embedding=embedding)
            if row["accounting_state"] != "pending" or row["dispatched"]:
                raise _V13BudgetExceeded("Reservation is not eligible for dispatch")
            row["dispatched"] = True

    def mark_call_failed(self, call_index: int, error: Any, *, embedding: bool = False) -> None:
        with self._lock:
            row = self._row(call_index, embedding=embedding)
            row.update(failed=True, error=type(error).__name__, completed_at_elapsed_seconds=round(self.elapsed(), 3))
            if row["accounting_state"] == "pending":
                row["accounting_state"] = "uncertain" if row["dispatched"] else "not_sent"
            # Settled usage is retained, including JSON parse/refusal/incomplete errors.

    def _settle(self, row: dict, amount: float, fingerprint: tuple, counters: dict, *, embedding: bool = False) -> None:
        if row.get("accounting_state") == "not_sent":
            self.accounting_anomalies.append("usage_after_not_sent")
        previous = row.get("_usage_fingerprint")
        if previous is not None:
            if previous == fingerprint:
                return  # Exactly-once accounting, including repeated failure handlers.
            self.accounting_anomalies.append("conflicting_usage")
            # Never refund a conflicting receipt. Record the larger monetary exposure.
            extra = max(0.0, amount - float(row.get("estimated_cost_usd") or 0.0))
            self.estimated_cost_usd += extra
            if embedding:
                self.embedding_estimated_cost_usd += extra
            row["estimated_cost_usd"] = max(amount, float(row.get("estimated_cost_usd") or 0.0))
            return
        row.update(counters)
        row.update(accounting_state="settled", usage_source="provider_usage", estimated_cost_usd=amount,
                   _usage_fingerprint=fingerprint, completed_at_elapsed_seconds=round(self.elapsed(), 3))
        self.estimated_cost_usd += amount
        if embedding:
            self.embedding_estimated_cost_usd += amount
            self.embedding_input_tokens += counters["input_tokens"]
        else:
            for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens"):
                setattr(self, key, getattr(self, key) + counters[key])
        if amount > float(row["reserved_cost_usd"]) + 1e-12:
            row["reservation_exceeded"] = True
            self.accounting_anomalies.append("provider_usage_exceeded_reservation")

    def record_usage(self, call_index: int, model: str, usage: dict) -> None:
        with self._lock:
            row = self._row(call_index)
            if str(model) != row["model"]:
                self.accounting_anomalies.append("usage_model_mismatch")
                row["accounting_state"] = "uncertain" if row["accounting_state"] == "pending" else row["accounting_state"]
                return
            try:
                u = usage if isinstance(usage, dict) else {}
                input_tokens = _usage_int(u["input_tokens"] if "input_tokens" in u else u["prompt_tokens"])
                output_tokens = _usage_int(u["output_tokens"] if "output_tokens" in u else u["completion_tokens"])
                ins = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
                outs = u.get("output_tokens_details") or u.get("completion_tokens_details") or {}
                cached = _usage_int(ins.get("cached_tokens", 0))
                writes = _usage_int(ins.get("cache_write_tokens", 0))
                reasoning = _usage_int(outs.get("reasoning_tokens", u.get("reasoning_tokens", 0)))
                if cached + writes > input_tokens or reasoning > output_tokens:
                    raise ValueError("Inconsistent token details")
            except (KeyError, TypeError, ValueError, AttributeError):
                if row["accounting_state"] == "pending":
                    row["accounting_state"] = "uncertain"
                    row["usage_source"] = "missing_or_invalid"
                return
            cost = ((input_tokens - cached - writes) * row["input_rate"] + cached * row["input_rate"] * 0.10 + writes * row["input_rate"] * 1.25 + output_tokens * row["output_rate"]) / 1_000_000.0
            counters = dict(input_tokens=input_tokens, cached_input_tokens=cached, cache_write_tokens=writes, output_tokens=output_tokens, reasoning_tokens=reasoning)
            self._settle(row, cost, (input_tokens, cached, writes, output_tokens, reasoning), counters)
            if input_tokens > row["reserved_input_tokens"] or output_tokens > row["max_output_tokens"]:
                row["token_bound_exceeded"] = True
                self.accounting_anomalies.append("provider_tokens_exceeded_limit")

    def grant_retry_allowance(self, *, failed_attempts: int, reason: str) -> int:
        with self._lock:
            if self.remaining() < 8.0 or self.accounting_anomalies or self.remaining_cost_usd <= 0.0:
                return 0
            failed = [r["call"] for r in self.call_log if r.get("failed") and r["call"] not in self._retry_credited]
            granted = min(max(0, int(failed_attempts or 0)), len(failed), max(0, self.absolute_max_llm_calls - self.max_llm_calls))
            if granted:
                self._retry_credited.update(failed[:granted])
                self.max_llm_calls += granted
                self.retry_allowance_calls += granted
                self.retry_events.append(dict(reason=str(reason or "model_fallback")[:160], failed_attempts=int(failed_attempts), granted_calls=granted, max_llm_calls_after=self.max_llm_calls, at_elapsed_seconds=round(self.elapsed(), 3)))
            return granted

    def reserve_embedding(self, *, model: str, texts: list[str], requested_timeout: int) -> tuple[int, int]:
        with self._lock:
            self._admission()
            tokens = sum(max(1, len(t.encode("utf-8")) + 16) for t in texts)
            rate = _positive_rate(_runtime_value("V13_PRICE_EMBED_INPUT", 0.0))
            amount = tokens * rate / 1_000_000.0
            if self.committed_cost_usd + amount > float(self.max_estimated_cost_usd) + 1e-12:
                raise _V13BudgetExceeded("Embedding reservation would exceed request budget")
            timeout = self._timeout(requested_timeout)
            self.embedding_calls += 1
            self.embedding_log.append(dict(embedding_call=self.embedding_calls, model=str(model), reserved_input_tokens=tokens, reserved_cost_usd=amount, input_rate=rate, timeout_seconds=timeout, accounting_state="pending", dispatched=False, started_at_elapsed_seconds=round(self.elapsed(), 3)))
            return timeout, self.embedding_calls

    def record_embedding_usage(self, call_index: int, usage: dict, *, cache_hits: int = 0) -> None:
        with self._lock:
            row = self._row(call_index, embedding=True)
            if not row.get("_cache_hits_recorded"):
                self.embedding_cache_hits += max(0, int(cache_hits))
                row["_cache_hits_recorded"] = True
            try:
                tokens = _usage_int(next(usage[k] for k in ("prompt_tokens", "input_tokens", "total_tokens") if k in usage))
            except (TypeError, ValueError, StopIteration, KeyError):
                if row["accounting_state"] == "pending":
                    row.update(accounting_state="uncertain", usage_source="missing_or_invalid")
                return
            self._settle(row, tokens * row["input_rate"] / 1_000_000.0, (tokens,), {"input_tokens": tokens}, embedding=True)
            if tokens > row["reserved_input_tokens"]:
                self.accounting_anomalies.append("embedding_tokens_exceeded_reservation")

    def record_embedding(self, *, input_tokens: int, cache_hits: int = 0) -> None:
        # Compatibility only: transports must reserve BEFORE I/O and use the method
        # above. An unreserved legacy receipt is reported and blocks later spending.
        with self._lock:
            tokens = max(0, int(input_tokens or 0))
            self.embedding_calls += 1
            self.embedding_input_tokens += tokens
            self.embedding_cache_hits += max(0, int(cache_hits or 0))
            cost = tokens * _positive_rate(_runtime_value("V13_PRICE_EMBED_INPUT", 0.0)) / 1_000_000.0
            self.embedding_estimated_cost_usd += cost
            self.estimated_cost_usd += cost
            self.accounting_anomalies.append("unreserved_embedding_usage")

    def public_meta(self) -> dict:
        with self._lock:
            pending, uncertain = self._liability("pending"), self._liability("uncertain")
            clean = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
            return dict(engine="v13", budget_policy_version=BUDGET_POLICY_VERSION, route=self.route,
                        elapsed_seconds=round(self.elapsed(), 3), deadline_seconds=self.deadline_seconds,
                        llm_calls=self.llm_calls, max_llm_calls=self.max_llm_calls, base_max_llm_calls=self.base_max_llm_calls,
                        absolute_max_llm_calls=self.absolute_max_llm_calls, retry_allowance_calls=self.retry_allowance_calls,
                        retry_events=list(self.retry_events), input_tokens=self.input_tokens, cached_input_tokens=self.cached_input_tokens,
                        cache_write_tokens=self.cache_write_tokens, output_tokens=self.output_tokens, reasoning_tokens=self.reasoning_tokens,
                        embedding_calls=self.embedding_calls, embedding_input_tokens=self.embedding_input_tokens,
                        embedding_cache_hits=self.embedding_cache_hits, embedding_estimated_cost_usd=round(self.embedding_estimated_cost_usd, 8),
                        estimated_cost_usd=round(self.estimated_cost_usd, 8), max_estimated_cost_usd=self.max_estimated_cost_usd,
                        reserved_cost_usd=round(pending, 8), uncertain_cost_usd=round(uncertain, 8),
                        committed_cost_usd=round(self.committed_cost_usd, 8), remaining_cost_usd=round(self.remaining_cost_usd, 8),
                        accounting_complete=not (any(row.get("accounting_state") in {"pending", "uncertain"} for row in self._rows()) or self.accounting_anomalies),
                        accounting_anomalies=list(dict.fromkeys(self.accounting_anomalies)),
                        cost_limit_exceeded=self.committed_cost_usd > float(self.max_estimated_cost_usd) + 1e-12,
                        refinement_used=bool(self.refinement_used), semantic_cache=self.semantic_cache,
                        evidence_gate=dict(self.evidence_gate or {}), retrieval_assurance=dict(self.retrieval_assurance or {}),
                        calls=[clean(row) for row in self.call_log], embeddings=[clean(row) for row in self.embedding_log])


_V13_BUDGET_CTX = contextvars.ContextVar("machinemind_v13_budget", default=None)


def _v13_current_budget() -> Optional[_V13RequestBudget]:
    value = _V13_BUDGET_CTX.get()
    return value if isinstance(value, _V13RequestBudget) else None


def _v13_model_rates(model: str) -> tuple[float, float]:
    name = str(model or "").strip().lower()
    if "gpt-5.6-sol" in name:
        return (
            _runtime_value("V13_PRICE_SOL_INPUT"),
            _runtime_value("V13_PRICE_SOL_OUTPUT"),
        )
    if "gpt-5.6-terra" in name:
        return (
            _runtime_value("V13_PRICE_TERRA_INPUT"),
            _runtime_value("V13_PRICE_TERRA_OUTPUT"),
        )
    if "gpt-5.6-luna" in name:
        return (
            _runtime_value("V13_PRICE_LUNA_INPUT"),
            _runtime_value("V13_PRICE_LUNA_OUTPUT"),
        )
    # Conservative fallback used only for runtime accounting.
    return (
        _runtime_value("V13_PRICE_SOL_INPUT"),
        _runtime_value("V13_PRICE_SOL_OUTPUT"),
    )


def _v13_estimate_model_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    input_rate, output_rate = _v13_model_rates(model)
    total_input = max(0, int(input_tokens or 0))
    cached = max(0, min(total_input, int(cached_input_tokens or 0)))
    writes = max(
        0,
        min(total_input - cached, int(cache_write_tokens or 0)),
    )
    uncached = max(0, total_input - cached - writes)
    return (
        uncached * input_rate
        + cached * input_rate * 0.10
        + writes * input_rate * 1.25
        + max(0, int(output_tokens or 0)) * output_rate
    ) / 1_000_000.0
