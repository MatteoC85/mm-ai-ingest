"""Provider-facing OpenAI transport extracted from the production monolith.

Runtime settings and callbacks are supplied by ``main`` so that this module has no
reverse dependency on the composition root. The wrappers in ``main`` preserve the
historical function names, signatures, late-bound monkeypatch points, request budget
semantics and ingest metering behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any, Optional, Type

PostFn = Callable[..., Any]
BudgetResolver = Callable[[], Any]
IngestMeterResolver = Callable[[], Any]


def normalize_model_candidates(models: Optional[list[str]]) -> list[str]:
    out: list[str] = []
    seen = set()
    for model_name in models or []:
        model_name = str(model_name or "").strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        out.append(model_name)
    return out


def safety_identifier(company_id: str) -> str:
    raw = str(company_id or "").encode("utf-8", errors="ignore")
    return "mm_" + hashlib.sha256(raw).hexdigest()[:40]


def response_text(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    refusals: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            content_type = str(content.get("type") or "")
            if content_type == "output_text" and str(content.get("text") or "").strip():
                parts.append(str(content.get("text") or "").strip())
            elif content_type == "refusal" and str(content.get("refusal") or "").strip():
                refusals.append(str(content.get("refusal") or "").strip())

    if parts:
        return "\n".join(parts).strip()
    if refusals:
        raise RuntimeError("OpenAI refusal: " + " | ".join(refusals)[:500])
    raise RuntimeError("OpenAI Responses API returned no output_text")


def embed_texts(
    texts: list[str],
    *,
    timeout: int = 60,
    api_key: str,
    model: str,
    url: str,
    post_fn: PostFn,
    current_budget_fn: BudgetResolver,
    current_ingest_meter_fn: IngestMeterResolver,
) -> list[list[float]]:
    if not api_key:
        raise Exception("OPENAI_API_KEY missing")

    normalized_texts = [str(value or "") for value in (texts or [])]
    if not normalized_texts:
        return []

    budget = None
    try:
        budget = current_budget_fn() if callable(current_budget_fn) else None
    except Exception:
        budget = None

    cache: dict[tuple[str, str], list[float]] = (
        budget.embedding_cache if budget is not None else {}
    )
    keys = [(model, value) for value in normalized_texts]
    unique_missing: list[str] = []
    seen_missing: set[tuple[str, str]] = set()
    cache_hits = 0
    for key, value in zip(keys, normalized_texts):
        if key in cache:
            cache_hits += 1
            continue
        if key in seen_missing:
            cache_hits += 1
            continue
        seen_missing.add(key)
        unique_missing.append(value)

    if unique_missing:
        request_timeout = max(5, int(timeout or 60))
        if budget is not None:
            budget.ensure_time(3.0)
            request_timeout = max(
                5,
                min(request_timeout, int(max(5.0, budget.remaining() - 1.0))),
            )

        payload = {"model": model, "input": unique_missing}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = post_fn(
            url,
            headers=headers,
            json=payload,
            timeout=request_timeout,
        )
        if response.status_code != 200:
            raise Exception(
                f"OpenAI embeddings failed: {response.status_code} {response.text}"
            )
        data = response.json()

        approx_tokens = sum(
            max(1, int(math.ceil(len(value) / 4.0)))
            for value in unique_missing
        )
        provider_usage = data.get("usage") if isinstance(data, dict) else {}
        provider_usage = provider_usage if isinstance(provider_usage, dict) else {}
        provider_tokens = int(
            provider_usage.get("prompt_tokens")
            or provider_usage.get("input_tokens")
            or provider_usage.get("total_tokens")
            or 0
        )
        ingest_meter = current_ingest_meter_fn()
        if ingest_meter is not None:
            ingest_meter.record_embedding(
                input_tokens=(provider_tokens if provider_tokens > 0 else approx_tokens),
                usage_source=(
                    "provider_usage" if provider_tokens > 0 else "character_fallback"
                ),
            )

        vectors: list[Optional[list[float]]] = [None] * len(unique_missing)
        for item in data.get("data", []):
            idx = int(item["index"])
            if 0 <= idx < len(vectors):
                vectors[idx] = item["embedding"]
        if any(vector is None for vector in vectors):
            raise Exception("OpenAI embeddings response missing some items")

        for value, vector in zip(unique_missing, vectors):
            cache[(model, value)] = list(vector or [])
        if budget is not None:
            budget.record_embedding(input_tokens=approx_tokens, cache_hits=cache_hits)
    elif budget is not None:
        budget.embedding_cache_hits += len(normalized_texts)

    out: list[list[float]] = []
    for key in keys:
        vector = cache.get(key)
        if vector is None:
            raise Exception("OpenAI embeddings cache reconstruction failed")
        out.append(vector)
    return out


def chat_text(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: str,
    default_model: str,
    url: str,
    post_fn: PostFn,
) -> str:
    if not api_key:
        raise Exception("OPENAI_API_KEY missing")
    payload = {
        "model": (model or default_model),
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = post_fn(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"OpenAI chat failed: {response.status_code} {response.text}")
    data = response.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


def chat_json(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    json_schema: Optional[dict] = None,
    timeout: int = 60,
    api_key: str,
    default_model: str,
    url: str,
    post_fn: PostFn,
) -> dict:
    if not api_key:
        raise Exception("OPENAI_API_KEY missing")
    payload = {
        "model": (model or default_model),
        "messages": messages,
        "temperature": 0,
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = post_fn(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise Exception(
            f"OpenAI chat JSON failed: {response.status_code} {response.text}"
        )
    data = response.json()
    msg = (data.get("choices", [{}])[0].get("message", {}) or {})
    content = msg.get("content", "")
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = str(content or "").strip()
    if not text:
        raise Exception("OpenAI chat JSON empty response")
    try:
        return json.loads(text)
    except Exception as exc:
        raise Exception(
            f"OpenAI chat JSON parse failed: {str(exc)} | raw={text[:500]}"
        )


def chat_json_models(
    messages: list[dict],
    *,
    models: Optional[list[str]] = None,
    json_schema: Optional[dict] = None,
    timeout: int = 60,
    default_model: str,
    normalize_models_fn: Callable[[Optional[list[str]]], list[str]],
    chat_json_fn: Callable[..., dict],
) -> dict:
    tried = normalize_models_fn(models) or [default_model]
    last_error: Optional[Exception] = None
    for model_name in tried:
        try:
            return chat_json_fn(
                messages,
                model=model_name,
                json_schema=json_schema,
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise Exception("No model candidates available for JSON chat call")


def responses_json(
    messages: list[dict],
    *,
    model: str,
    json_schema: dict,
    effort: str,
    reasoning_mode: str = "",
    timeout: int,
    max_output_tokens: int,
    company_id: str,
    purpose: str,
    api_key: str,
    url: str,
    post_fn: PostFn,
    current_budget_fn: BudgetResolver,
    response_text_fn: Callable[[dict], str],
    safety_identifier_fn: Callable[[str], str],
) -> dict:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    budget = current_budget_fn()
    if budget is None:
        raise RuntimeError("V13 budget context missing")
    timeout, output_cap, call_index = budget.reserve_call(
        model=model,
        purpose=purpose,
        requested_timeout=timeout,
        max_output_tokens=max_output_tokens,
        messages=messages,
    )
    schema_name = str(json_schema.get("name") or "machinemind_v13_output")
    schema_body = json_schema.get("schema") or {}
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
        "store": False,
        "reasoning": {"effort": effort or "medium", "context": "current_turn"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": bool(json_schema.get("strict", True)),
                "schema": schema_body,
            }
        },
        "max_output_tokens": output_cap,
        "safety_identifier": safety_identifier_fn(company_id),
    }
    if reasoning_mode:
        payload["reasoning"]["mode"] = reasoning_mode
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = post_fn(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI Responses failed: HTTP {response.status_code}: {response.text[:1200]}"
        )
    data = response.json()
    budget.record_usage(call_index, model, data.get("usage") or {})
    status = str(data.get("status") or "completed").strip().lower()
    if status == "incomplete":
        details = json.dumps(data.get("incomplete_details") or {}, ensure_ascii=False)[:700]
        raise RuntimeError(f"OpenAI Responses incomplete: {details}")
    if status in {"failed", "cancelled"}:
        raise RuntimeError(
            f"OpenAI Responses status={status}: "
            f"{json.dumps(data.get('error') or {}, ensure_ascii=False)[:700]}"
        )
    text = response_text_fn(data)
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"OpenAI Responses JSON parse failed: {exc}; raw={text[:800]}")
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI Responses structured output is not an object")
    return parsed


def json_models(
    messages: list[dict],
    *,
    models: list[str],
    json_schema: dict,
    effort: str,
    reasoning_mode: str,
    timeout: int,
    max_output_tokens: int,
    company_id: str,
    purpose: str,
    default_model: str,
    normalize_models_fn: Callable[[Optional[list[str]]], list[str]],
    current_budget_fn: BudgetResolver,
    responses_json_fn: Callable[..., dict],
    chat_json_fn: Callable[..., dict],
    budget_exceeded_type: Type[BaseException],
) -> tuple[dict, str]:
    candidates = normalize_models_fn(models)
    if not candidates:
        candidates = [default_model]
    errors: list[str] = []
    failed_provider_attempts = 0
    for model in candidates:
        budget = current_budget_fn()
        calls_before = int(budget.llm_calls) if budget is not None else 0
        try:
            if str(model).startswith("gpt-5.6"):
                parsed = responses_json_fn(
                    messages,
                    model=model,
                    json_schema=json_schema,
                    effort=effort,
                    reasoning_mode=reasoning_mode,
                    timeout=timeout,
                    max_output_tokens=max_output_tokens,
                    company_id=company_id,
                    purpose=purpose,
                )
            else:
                budget = current_budget_fn()
                if budget is None:
                    raise RuntimeError("V13 budget context missing")
                call_timeout, _output_cap, call_index = budget.reserve_call(
                    model=model,
                    purpose=purpose,
                    requested_timeout=timeout,
                    max_output_tokens=max_output_tokens,
                    messages=messages,
                )
                parsed = chat_json_fn(
                    messages,
                    model=model,
                    json_schema=json_schema,
                    timeout=call_timeout,
                )
                budget.record_usage(call_index, model, {})
            budget = current_budget_fn()
            if budget is not None and failed_provider_attempts:
                budget.grant_retry_allowance(
                    failed_attempts=failed_provider_attempts,
                    reason=f"{purpose}:model_fallback_succeeded",
                )
            return parsed, model
        except budget_exceeded_type as exc:
            if errors:
                errors.append(f"{model}: fallback skipped by budget: {str(exc)[:300]}")
                break
            raise
        except Exception as exc:
            budget = current_budget_fn()
            if budget is not None and int(budget.llm_calls) > calls_before:
                budget.mark_call_failed(int(budget.llm_calls), exc)
                failed_provider_attempts += 1
            errors.append(f"{model}: {str(exc)[:500]}")
    raise RuntimeError("All V13 model calls failed: " + " | ".join(errors)[:1800])
