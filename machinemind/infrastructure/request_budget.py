"""Request-scoped time/cost accounting used by ASK, Root Cause and Smart Diagnostic.

This module is a behavior-preserving extraction from the production monolith.  During
migration it deliberately reads configuration through a live runtime mapping supplied
by ``main``.  That keeps historical monkey-patching, environment-derived constants and
rollback behavior intact while removing the implementation from ``main.py``.
"""

import contextvars
import json
import math
import time as _time
from collections.abc import Mapping
from typing import Any, Optional

from machinemind.config import assistant_runtime as _assistant_defaults

_RUNTIME_GLOBALS: Mapping[str, Any] = {}


def configure_request_budget_runtime(runtime_globals: Mapping[str, Any]) -> None:
    """Bind the legacy runtime namespace without importing ``main`` circularly."""
    global _RUNTIME_GLOBALS
    _RUNTIME_GLOBALS = runtime_globals


def _runtime_value(name: str, default: Any = None) -> Any:
    if name in _RUNTIME_GLOBALS:
        return _RUNTIME_GLOBALS[name]
    return getattr(_assistant_defaults, name, default)


def _monotonic() -> float:
    runtime_time = _RUNTIME_GLOBALS.get("time_module")
    if runtime_time is not None and callable(getattr(runtime_time, "monotonic", None)):
        return float(runtime_time.monotonic())
    return float(_time.monotonic())


class _V13BudgetExceeded(RuntimeError):
    pass


class _V13RequestBudget:
    def __init__(self, mode: str):
        self.mode = str(mode or "ask").strip().lower()
        self.started_monotonic = _monotonic()
        self.deadline_seconds = (
            _runtime_value("V13_ROOT_CAUSE_DEADLINE_SECONDS")
            if self.mode == "root_cause"
            else _runtime_value("V13_ASK_DEADLINE_SECONDS")
        )
        self.deadline_monotonic = self.started_monotonic + float(self.deadline_seconds)
        self.max_llm_calls = (
            _runtime_value("V13_MAX_LLM_CALLS_ROOT_CAUSE")
            if self.mode == "root_cause"
            else _runtime_value("V13_MAX_LLM_CALLS_ASK")
        )
        # Failed provider/model attempts are infrastructure retries, not completed
        # reasoning stages. A bounded retry allowance preserves the downstream
        # synthesis/verifier budget without relaxing the time or monetary ceilings.
        self.base_max_llm_calls = int(self.max_llm_calls)
        self.absolute_max_llm_calls = min(6, int(self.max_llm_calls) + 2)
        self.retry_allowance_calls = 0
        self.retry_events: list[dict] = []
        self.max_estimated_cost_usd = (
            _runtime_value("V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD")
            if self.mode == "root_cause"
            else _runtime_value("V13_MAX_ESTIMATED_COST_ASK_USD")
        )
        self.llm_calls = 0
        self.estimated_cost_usd = 0.0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.embedding_calls = 0
        self.embedding_input_tokens = 0
        self.embedding_cache_hits = 0
        self.embedding_estimated_cost_usd = 0.0
        self.call_log: list[dict] = []
        self.embedding_cache: dict[tuple[str, str], list[float]] = {}
        self.route = "unselected"
        self.refinement_used = False
        self.semantic_cache = "miss"
        self.evidence_gate: dict = {}
        self.retrieval_assurance: dict = {}

    def elapsed(self) -> float:
        return max(0.0, _monotonic() - self.started_monotonic)

    def remaining(self) -> float:
        return max(0.0, self.deadline_monotonic - _monotonic())

    def ensure_time(self, minimum_seconds: float = 2.0) -> None:
        if self.remaining() < float(minimum_seconds):
            raise _V13BudgetExceeded(
                f"V13 {self.mode} deadline exhausted after {self.elapsed():.2f}s"
            )

    def reserve_call(
        self,
        *,
        model: str,
        purpose: str,
        requested_timeout: int,
        max_output_tokens: int,
        messages: list[dict],
    ) -> tuple[int, int, int]:
        self.ensure_time(4.0)
        if self.llm_calls >= self.max_llm_calls:
            raise _V13BudgetExceeded(
                f"V13 {self.mode} LLM call budget exhausted ({self.llm_calls}/{self.max_llm_calls})"
            )
        if self.estimated_cost_usd >= self.max_estimated_cost_usd:
            raise _V13BudgetExceeded(
                f"V13 {self.mode} estimated cost budget exhausted ({self.estimated_cost_usd:.4f} USD)"
            )

        message_chars = 0
        for msg in messages or []:
            try:
                message_chars += len(json.dumps(msg, ensure_ascii=False))
            except Exception:
                message_chars += len(str(msg))
        approx_input_tokens = max(1, int(math.ceil(message_chars / 3.0)))

        remaining = self.remaining()
        timeout = max(
            5,
            min(
                int(requested_timeout or 30),
                int(max(5.0, remaining - 2.0)),
            ),
        )
        output_cap = max(800, int(max_output_tokens or 2000))

        # Enforce the monetary budget before the request. Reasoning tokens are part of
        # output_tokens, so max_output_tokens is also the hard cost ceiling.
        input_rate, output_rate = _v13_model_rates(model)
        remaining_cost = max(
            0.0,
            self.max_estimated_cost_usd - self.estimated_cost_usd,
        )
        affordable_output = int(
            max(
                0.0,
                (
                    remaining_cost * 1_000_000.0
                    - approx_input_tokens * input_rate
                )
                / max(0.000001, output_rate),
            )
        )
        output_cap = min(output_cap, affordable_output)
        if output_cap < 800:
            raise _V13BudgetExceeded(
                f"V13 {self.mode} call would exceed cost budget ({self.estimated_cost_usd:.4f}/{self.max_estimated_cost_usd:.4f} USD)"
            )

        self.llm_calls += 1
        call_index = self.llm_calls
        self.call_log.append(
            {
                "call": call_index,
                "model": str(model or ""),
                "purpose": str(purpose or "reasoning"),
                "timeout_seconds": timeout,
                "max_output_tokens": output_cap,
                "approx_input_tokens": approx_input_tokens,
                "started_at_elapsed_seconds": round(self.elapsed(), 3),
            }
        )
        return timeout, output_cap, call_index

    def mark_call_failed(self, call_index: int, error: Any) -> None:
        """Annotate a failed provider attempt without treating it as a reasoning result."""
        for row in self.call_log:
            if int(row.get("call") or 0) == int(call_index):
                row["failed"] = True
                row["error"] = str(error or "")[:700]
                row["completed_at_elapsed_seconds"] = round(self.elapsed(), 3)
                break

    def grant_retry_allowance(self, *, failed_attempts: int, reason: str) -> int:
        """Restore stage capacity consumed by failed model attempts, within hard caps."""
        requested = max(0, int(failed_attempts or 0))
        if requested <= 0:
            return 0
        # Never extend a request that no longer has enough time for a useful call.
        if (
            self.remaining() < 8.0
            or self.estimated_cost_usd >= self.max_estimated_cost_usd
        ):
            return 0
        room = max(
            0,
            int(self.absolute_max_llm_calls) - int(self.max_llm_calls),
        )
        granted = min(requested, room)
        if granted <= 0:
            return 0
        self.max_llm_calls += granted
        self.retry_allowance_calls += granted
        self.retry_events.append(
            {
                "reason": str(reason or "model_fallback")[:160],
                "failed_attempts": requested,
                "granted_calls": granted,
                "max_llm_calls_after": int(self.max_llm_calls),
                "at_elapsed_seconds": round(self.elapsed(), 3),
            }
        )
        return granted

    def record_usage(self, call_index: int, model: str, usage: dict) -> None:
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        input_details = (
            usage.get("input_tokens_details")
            or usage.get("prompt_tokens_details")
            or {}
        )
        output_details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        cached_input_tokens = int((input_details or {}).get("cached_tokens") or 0)
        cache_write_tokens = int(
            (input_details or {}).get("cache_write_tokens") or 0
        )
        reasoning_tokens = int(
            (output_details or {}).get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or 0
        )
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached_input_tokens
        self.cache_write_tokens += cache_write_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens
        call_cost = _v13_estimate_model_cost_usd(
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        self.estimated_cost_usd += call_cost

        for row in self.call_log:
            if int(row.get("call") or 0) == int(call_index):
                row["input_tokens"] = input_tokens
                row["cached_input_tokens"] = cached_input_tokens
                row["cache_write_tokens"] = cache_write_tokens
                row["output_tokens"] = output_tokens
                row["reasoning_tokens"] = reasoning_tokens
                row["estimated_cost_usd"] = round(call_cost, 6)
                row["completed_at_elapsed_seconds"] = round(self.elapsed(), 3)
                break

    def record_embedding(self, *, input_tokens: int, cache_hits: int = 0) -> None:
        tokens = max(0, int(input_tokens or 0))
        self.embedding_calls += 1
        self.embedding_input_tokens += tokens
        self.embedding_cache_hits += max(0, int(cache_hits or 0))
        cost = (
            tokens * float(_runtime_value("V13_PRICE_EMBED_INPUT", 0.0) or 0.0)
        ) / 1_000_000.0
        self.embedding_estimated_cost_usd += cost
        self.estimated_cost_usd += cost

    def public_meta(self) -> dict:
        return {
            "engine": "v13",
            "route": self.route,
            "elapsed_seconds": round(self.elapsed(), 3),
            "deadline_seconds": self.deadline_seconds,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "base_max_llm_calls": self.base_max_llm_calls,
            "absolute_max_llm_calls": self.absolute_max_llm_calls,
            "retry_allowance_calls": self.retry_allowance_calls,
            "retry_events": list(self.retry_events),
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "embedding_calls": self.embedding_calls,
            "embedding_input_tokens": self.embedding_input_tokens,
            "embedding_cache_hits": self.embedding_cache_hits,
            "embedding_estimated_cost_usd": round(
                self.embedding_estimated_cost_usd,
                8,
            ),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
            "refinement_used": bool(self.refinement_used),
            "semantic_cache": self.semantic_cache,
            "evidence_gate": dict(self.evidence_gate or {}),
            "retrieval_assurance": dict(self.retrieval_assurance or {}),
            "calls": list(self.call_log),
        }


_V13_BUDGET_CTX = contextvars.ContextVar(
    "machinemind_v13_budget",
    default=None,
)


def _v13_current_budget() -> Optional[_V13RequestBudget]:
    try:
        value = _V13_BUDGET_CTX.get()
        return value if isinstance(value, _V13RequestBudget) else None
    except Exception:
        return None


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
