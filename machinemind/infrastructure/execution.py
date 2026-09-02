"""Generic execution guards for synchronous AI work.

The module contains no ASK/Root Cause/Smart reasoning policy.  Callers provide the
language selector and error/timeout envelope builders so API behavior remains owned by
the application layer while thread, asyncio and streaming mechanics live here.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import time as time_module
from collections.abc import Callable
from typing import Any, Optional

from fastapi.responses import StreamingResponse


async def stream_json_response(
    *,
    mode: str,
    sync_func: Callable[..., Any],
    payload: Any,
    x_ai_internal_secret: Optional[str],
    hard_timeout_seconds: Optional[int],
    heartbeat_seconds: float,
    heartbeat_bytes: int,
    timeout_result_code: str,
    select_response_language: Callable[..., str],
    error_payload: Callable[[str, Exception], dict],
):
    """Run blocking work in a thread and stream JSON whitespace heartbeats."""
    result_task = asyncio.create_task(
        asyncio.to_thread(sync_func, payload, x_ai_internal_secret)
    )
    stream_started = time_module.monotonic()
    stream_query = str(
        getattr(payload, "query", "")
        or getattr(payload, "symptom_text", "")
        or ""
    ).strip()
    stream_language = select_response_language(
        stream_query,
        preferred=getattr(payload, "language", None),
    )

    async def stream_json():
        heartbeat = (" " * int(heartbeat_bytes)) + "\n"
        yield heartbeat
        while True:
            hard_remaining = None
            if hard_timeout_seconds is not None:
                hard_remaining = max(
                    0.0,
                    float(hard_timeout_seconds)
                    - (time_module.monotonic() - stream_started),
                )
                if hard_remaining <= 0.0:
                    timeout_message = (
                        "Maximum response time exceeded."
                        if str(stream_language or "").lower().startswith("en")
                        else "Tempo massimo di risposta superato."
                    )
                    result = {
                        "ok": True,
                        "status": "timeout",
                        "result_code": timeout_result_code,
                        "requested_mode": mode,
                        "effective_mode": mode,
                        "routed": False,
                        "language": stream_language,
                        "answer": timeout_message,
                        "problem_summary": timeout_message,
                        "possible_causes": [],
                        "recommended_next_checks": [],
                        "citations": [],
                        "rg_links": [],
                        "meta": {
                            "cacheable": False,
                            "semantic_cacheable": False,
                            "hard_timeout": True,
                        },
                    }
                    try:
                        result_task.cancel()
                    except Exception:
                        pass
                    yield json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    break
            wait_seconds = float(heartbeat_seconds)
            if hard_remaining is not None:
                wait_seconds = max(
                    0.25,
                    min(wait_seconds, hard_remaining),
                )
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(result_task),
                    timeout=wait_seconds,
                )
            except asyncio.TimeoutError:
                yield heartbeat
                continue
            except Exception as exc:
                result = error_payload(mode, exc)

            if not isinstance(result, dict):
                result = {
                    "ok": False,
                    "status": "error",
                    "error": {
                        "code": f"{mode.upper()}_FAILED",
                        "message": "Invalid backend payload",
                        "detail": str(type(result)),
                    },
                }
            yield json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            break

    return StreamingResponse(
        stream_json(),
        media_type="application/json",
        headers={
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
            "X-MachineMind-V13-Heartbeat": "1",
        },
    )


async def json_with_hard_timeout(
    *,
    mode: str,
    sync_func: Callable[..., Any],
    payload: Any,
    x_ai_internal_secret: Optional[str],
    hard_timeout_seconds: int,
    timeout_result_code: str,
    root_cause_mode: str,
    select_response_language: Callable[..., str],
    error_payload: Callable[[str, Exception], dict],
) -> dict:
    """Return a normal JSON payload while enforcing an outer asyncio timeout."""
    query = str(
        getattr(payload, "query", "")
        or getattr(payload, "symptom_text", "")
        or ""
    ).strip()
    language = select_response_language(
        query,
        preferred=getattr(payload, "language", None),
    )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(sync_func, payload, x_ai_internal_secret),
            timeout=max(1.0, float(hard_timeout_seconds)),
        )
        if isinstance(result, dict):
            return result
        raise RuntimeError(f"Invalid backend payload: {type(result)}")
    except asyncio.TimeoutError:
        message = (
            "Maximum response time exceeded."
            if str(language or "").lower().startswith("en")
            else "Tempo massimo di risposta superato."
        )
        common = {
            "ok": True,
            "status": "timeout",
            "result_code": timeout_result_code,
            "requested_mode": mode,
            "effective_mode": mode,
            "routed": False,
            "language": language,
            "citations": [],
            "rg_links": [],
            "meta": {
                "cacheable": False,
                "semantic_cacheable": False,
                "hard_timeout": True,
                "hard_timeout_seconds": hard_timeout_seconds,
            },
        }
        if mode == root_cause_mode:
            common.update(
                {
                    "symptom": query,
                    "problem_summary": message,
                    "possible_causes": [],
                    "recommended_next_checks": [],
                }
            )
        else:
            common["answer"] = message
        return common
    except Exception as exc:
        return error_payload(mode, exc)


def run_sync_with_hard_timeout(
    func: Callable[..., Any],
    payload: Any,
    x_ai_internal_secret: Optional[str],
    *,
    hard_timeout_seconds: float,
    thread_name: str,
    on_timeout: Callable[[Any, float], Any],
):
    """Run a synchronous callable in a copied ContextVar context with a hard wait cap."""
    timeout = max(0.01, float(hard_timeout_seconds))
    ctx = contextvars.copy_context()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=str(thread_name),
    )
    future = executor.submit(
        ctx.run,
        func,
        payload,
        x_ai_internal_secret,
    )
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return on_timeout(payload, timeout)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
