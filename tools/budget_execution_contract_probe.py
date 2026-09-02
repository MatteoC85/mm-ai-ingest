from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
import types
from pathlib import Path
from typing import Any


def install_stubs() -> None:
    psycopg2 = types.ModuleType("psycopg2")
    psycopg2.connect = lambda **kwargs: None
    sys.modules["psycopg2"] = psycopg2

    google = sys.modules.get("google") or types.ModuleType("google")
    if not hasattr(google, "__path__"):
        google.__path__ = []
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    tasks = types.ModuleType("google.cloud.tasks_v2")

    class CloudTasksClient:
        pass

    tasks.CloudTasksClient = CloudTasksClient
    cloud.tasks_v2 = tasks
    google.cloud = cloud
    sys.modules["google"] = google
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.tasks_v2"] = tasks


install_stubs()
sys.path.insert(0, str(Path.cwd()))
import main  # noqa: E402


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): normalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize(v) for v in value)
    if isinstance(value, float):
        return round(value, 9)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def capture(fn, *args, **kwargs) -> dict:
    try:
        return {"value": normalize(fn(*args, **kwargs)), "error": None}
    except Exception as exc:
        return {
            "value": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "status_code": getattr(exc, "status_code", None),
                "detail": normalize(getattr(exc, "detail", None)),
            },
        }


def public_meta_without_timing(budget) -> dict:
    meta = dict(budget.public_meta())
    # All clocks are frozen below, but retain a defensive normalized representation.
    if "elapsed_seconds" in meta:
        meta["elapsed_seconds"] = round(float(meta["elapsed_seconds"]), 6)
    return normalize(meta)


output: dict[str, Any] = {}

names = [
    "_V13BudgetExceeded",
    "_V13RequestBudget",
    "_v13_current_budget",
    "_v13_model_rates",
    "_v13_estimate_model_cost_usd",
    "_v13_stream_json_response",
    "_assistant_core_json_with_hard_timeout",
    "_assistant_core_run_smart_with_hard_timeout",
]
def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"


output["surface"] = {
    name: {
        "name": getattr(getattr(main, name), "__name__", None),
        "qualname": getattr(getattr(main, name), "__qualname__", None),
        "module": getattr(getattr(main, name), "__module__", None),
        "signature": safe_signature(getattr(main, name)),
    }
    for name in names
}

# Request budget and cost accounting with a deterministic clock.
clock = {"now": 1000.0}
original_monotonic = main.time_module.monotonic
main.time_module.monotonic = lambda: clock["now"]
try:
    cases: dict[str, Any] = {}
    for mode in ["ask", "root_cause", "unexpected"]:
        budget = main._V13RequestBudget(mode)
        cases[f"{mode}_initial"] = public_meta_without_timing(budget)

    budget = main._V13RequestBudget("ask")
    clock["now"] = 1005.25
    cases["elapsed_remaining"] = {
        "elapsed": budget.elapsed(),
        "remaining": budget.remaining(),
    }
    reserved = budget.reserve_call(
        model=main.V13_FAST_MODEL,
        purpose="contract-probe",
        requested_timeout=99,
        max_output_tokens=1800,
        messages=[{"role": "user", "content": "abc"}],
    )
    cases["reserved"] = reserved
    budget.record_usage(
        reserved[2],
        main.V13_FAST_MODEL,
        {
            "input_tokens": 1000,
            "output_tokens": 200,
            "input_tokens_details": {
                "cached_tokens": 100,
                "cache_write_tokens": 50,
            },
            "output_tokens_details": {"reasoning_tokens": 75},
        },
    )
    budget.record_embedding(input_tokens=400, cache_hits=2)
    cases["after_usage"] = public_meta_without_timing(budget)
    budget.mark_call_failed(reserved[2], RuntimeError("provider down"))
    cases["after_failed"] = public_meta_without_timing(budget)
    budget.max_llm_calls = budget.llm_calls
    clock["now"] = 1006.0
    cases["retry_granted"] = budget.grant_retry_allowance(
        failed_attempts=2,
        reason="provider-fallback",
    )
    cases["after_retry"] = public_meta_without_timing(budget)

    token = main._V13_BUDGET_CTX.set(budget)
    try:
        cases["context_bound"] = main._v13_current_budget() is budget
    finally:
        main._V13_BUDGET_CTX.reset(token)
    cases["context_reset"] = main._v13_current_budget() is None

    clock["now"] = 2000.0
    near_deadline = main._V13RequestBudget("ask")
    clock["now"] = near_deadline.deadline_monotonic - 1.0
    cases["ensure_time_error"] = capture(near_deadline.ensure_time, 2.0)

    clock["now"] = 3000.0
    call_limited = main._V13RequestBudget("ask")
    call_limited.llm_calls = call_limited.max_llm_calls
    cases["call_limit_error"] = capture(
        call_limited.reserve_call,
        model=main.V13_FAST_MODEL,
        purpose="limited",
        requested_timeout=10,
        max_output_tokens=1000,
        messages=[],
    )

    cost_limited = main._V13RequestBudget("ask")
    cost_limited.estimated_cost_usd = cost_limited.max_estimated_cost_usd
    cases["cost_limit_error"] = capture(
        cost_limited.reserve_call,
        model=main.V13_FAST_MODEL,
        purpose="limited",
        requested_timeout=10,
        max_output_tokens=1000,
        messages=[],
    )

    unaffordable = main._V13RequestBudget("ask")
    unaffordable.max_estimated_cost_usd = 0.000001
    cases["unaffordable_error"] = capture(
        unaffordable.reserve_call,
        model=main.V13_FAST_MODEL,
        purpose="too-expensive",
        requested_timeout=10,
        max_output_tokens=800,
        messages=[{"content": "x" * 100}],
    )

    cases["rates"] = {
        model: main._v13_model_rates(model)
        for model in [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "unknown-model",
        ]
    }
    cases["costs"] = {
        "terra_cached": main._v13_estimate_model_cost_usd(
            "gpt-5.6-terra",
            1000,
            200,
            cached_input_tokens=100,
            cache_write_tokens=50,
        ),
        "fallback": main._v13_estimate_model_cost_usd(
            "unknown-model",
            1000,
            200,
        ),
    }

    # Production configuration is late-bound from main. Changes must be observed by
    # the extracted implementation exactly as they were by the monolith.
    config_names = [
        "V13_ASK_DEADLINE_SECONDS",
        "V13_MAX_LLM_CALLS_ASK",
        "V13_MAX_ESTIMATED_COST_ASK_USD",
        "V13_PRICE_TERRA_INPUT",
        "V13_PRICE_TERRA_OUTPUT",
        "V13_PRICE_EMBED_INPUT",
    ]
    saved = {name: getattr(main, name) for name in config_names}
    try:
        main.V13_ASK_DEADLINE_SECONDS = 33
        main.V13_MAX_LLM_CALLS_ASK = 3
        main.V13_MAX_ESTIMATED_COST_ASK_USD = 0.123
        main.V13_PRICE_TERRA_INPUT = 9.0
        main.V13_PRICE_TERRA_OUTPUT = 21.0
        main.V13_PRICE_EMBED_INPUT = 0.5
        clock["now"] = 4000.0
        custom = main._V13RequestBudget("ask")
        custom_call = custom.reserve_call(
            model="gpt-5.6-terra",
            purpose="custom",
            requested_timeout=60,
            max_output_tokens=2500,
            messages=[{"x": "y"}],
        )
        custom.record_usage(
            custom_call[2],
            "gpt-5.6-terra",
            {"input_tokens": 100, "output_tokens": 20},
        )
        custom.record_embedding(input_tokens=50, cache_hits=1)
        cases["late_bound_config"] = {
            "call": custom_call,
            "meta": public_meta_without_timing(custom),
        }
    finally:
        for name, value in saved.items():
            setattr(main, name, value)

    output["budget"] = normalize(cases)
finally:
    main.time_module.monotonic = original_monotonic


async def exercise_async_execution() -> dict[str, Any]:
    result: dict[str, Any] = {}

    class Payload:
        query = "ciao"
        symptom_text = ""
        language = "it"

    payload = Payload()

    def ok(_payload, _secret):
        return {"ok": True, "value": 1}

    def error(_payload, _secret):
        raise ValueError("boom")

    def invalid(_payload, _secret):
        return "invalid"

    result["json_success"] = await main._assistant_core_json_with_hard_timeout(
        mode=main.MODE_ASK,
        sync_func=ok,
        payload=payload,
        x_ai_internal_secret="secret",
        hard_timeout_seconds=2,
    )
    result["json_error"] = await main._assistant_core_json_with_hard_timeout(
        mode=main.MODE_ASK,
        sync_func=error,
        payload=payload,
        x_ai_internal_secret="secret",
        hard_timeout_seconds=2,
    )
    result["json_invalid"] = await main._assistant_core_json_with_hard_timeout(
        mode=main.MODE_ASK,
        sync_func=invalid,
        payload=payload,
        x_ai_internal_secret="secret",
        hard_timeout_seconds=2,
    )

    for name, function in [
        ("stream_success", ok),
        ("stream_error", error),
        ("stream_invalid", invalid),
    ]:
        response = await main._v13_stream_json_response(
            mode="ask",
            sync_func=function,
            payload=payload,
            x_ai_internal_secret="secret",
            hard_timeout_seconds=2,
        )
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            chunks.append(chunk)
        result[name] = {
            "media_type": response.media_type,
            "headers": dict(response.headers),
            "chunks": chunks,
        }

    def slow_stream(_payload, _secret):
        time.sleep(0.35)
        return {"ok": True, "late": True}

    response = await main._v13_stream_json_response(
        mode="ask",
        sync_func=slow_stream,
        payload=payload,
        x_ai_internal_secret="secret",
        hard_timeout_seconds=0.02,
    )
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        chunks.append(chunk)
    result["stream_timeout"] = {
        "media_type": response.media_type,
        "headers": dict(response.headers),
        "chunks": chunks,
    }

    def slow_json(_payload, _secret):
        time.sleep(1.1)
        return {"ok": True, "late": True}

    result["json_timeout"] = await main._assistant_core_json_with_hard_timeout(
        mode=main.MODE_ROOT_CAUSE,
        sync_func=slow_json,
        payload=payload,
        x_ai_internal_secret="secret",
        hard_timeout_seconds=0.01,
    )
    return result


output["async_execution"] = normalize(asyncio.run(exercise_async_execution()))


class SmartPayload:
    symptom_text = "x"
    language = "it"
    state_json = '{"opaque":true}'
    session_id = "session"
    company_id = "company"
    machine_id = "machine"


smart_payload = SmartPayload()
output["smart_success"] = normalize(
    main._assistant_core_run_smart_with_hard_timeout(
        lambda _payload, _secret: {"ok": True},
        smart_payload,
        "secret",
        turn_kind="answer",
        hard_timeout_seconds=1,
    )
)


def sleepy(_payload, _secret):
    time.sleep(0.05)
    return {"ok": True}


output["smart_timeout"] = normalize(
    main._assistant_core_run_smart_with_hard_timeout(
        sleepy,
        smart_payload,
        "secret",
        turn_kind="answer",
        hard_timeout_seconds=0.01,
    )
)

# ContextVar propagation into the Smart worker thread is a required invariant.
probe_budget = main._V13RequestBudget("ask")
probe_token = main._V13_BUDGET_CTX.set(probe_budget)
try:
    output["smart_context_propagated"] = main._assistant_core_run_smart_with_hard_timeout(
        lambda _payload, _secret: main._v13_current_budget() is probe_budget,
        smart_payload,
        "secret",
        turn_kind="answer",
        hard_timeout_seconds=1,
    )
finally:
    main._V13_BUDGET_CTX.reset(probe_token)

print(json.dumps(normalize(output), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
