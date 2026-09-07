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


def _send_json(*, url: str, headers: dict, payload: dict, timeout: int,
               post_fn: PostFn, budget=None, call_index: Optional[int] = None,
               embedding: bool = False, cache_hits: int = 0) -> dict:
    """One reservation, one HTTP dispatch. Never perform transparent HTTP retries.

    Usage is reconciled before content parsing/refusal/status checks. Any outcome
    after dispatch without usable usage stays uncertain. Closing a client timeout
    does not prove the provider performed no work.
    """
    try:
        if budget is not None:
            budget.mark_dispatched(call_index, embedding=embedding)
        response = post_fn(url, headers=headers, json=payload, timeout=timeout, allow_redirects=False)
        data = response.json()
        if budget is not None:
            usage = data.get("usage") if isinstance(data, dict) else None
            if embedding:
                budget.record_embedding_usage(call_index, usage, cache_hits=cache_hits)
            else:
                budget.record_usage(call_index, str(payload["model"]), usage)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI provider returned HTTP {response.status_code}")
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI provider returned non-object JSON")
        if budget is not None:
            budget.ensure_time(0.0)  # account late result, but do not use it after cancellation
        return data
    except BaseException as exc:
        if budget is not None and call_index is not None:
            budget.mark_call_failed(call_index, exc, embedding=embedding)
        raise


def embed_texts(texts: list[str], *, timeout: int = 60, api_key: str, model: str,
                url: str, post_fn: PostFn, current_budget_fn: BudgetResolver,
                current_ingest_meter_fn: IngestMeterResolver) -> list[list[float]]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    values = [str(value or "") for value in (texts or [])]
    if not values:
        return []
    # Failure to resolve a request budget must not silently switch to unmetered I/O.
    budget = current_budget_fn() if callable(current_budget_fn) else None
    cache = budget.embedding_cache if budget is not None else {}
    keys = [(model, value) for value in values]
    missing, seen, cache_hits = [], set(), 0
    for key, value in zip(keys, values):
        if key in cache or key in seen:
            cache_hits += 1
        else:
            missing.append(value)
            seen.add(key)
    if missing:
        request_timeout = max(1, int(timeout or 60))
        call_index = None
        if budget is not None:
            request_timeout, call_index = budget.reserve_embedding(model=model, texts=missing, requested_timeout=request_timeout)
        try:
            data = _send_json(url=url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                              payload={"model": model, "input": missing}, timeout=request_timeout,
                              post_fn=post_fn, budget=budget, call_index=call_index, embedding=True, cache_hits=cache_hits)
            # Ingest ledger behavior is intentionally unchanged. AI request accounting
            # above uses provider usage or keeps its pre-dispatch reservation.
            approx_tokens = sum(max(1, int(math.ceil(len(value) / 4.0))) for value in missing)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            provider_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("total_tokens") or 0)
            ingest_meter = current_ingest_meter_fn()
            if ingest_meter is not None:
                ingest_meter.record_embedding(input_tokens=provider_tokens if provider_tokens > 0 else approx_tokens,
                                               usage_source="provider_usage" if provider_tokens > 0 else "character_fallback")
            vectors = [None] * len(missing)
            for item in data.get("data", []):
                idx = int(item["index"])
                if 0 <= idx < len(vectors):
                    if vectors[idx] is not None:
                        raise RuntimeError("OpenAI embeddings returned duplicate index")
                    vector = item["embedding"]
                    if not isinstance(vector, list) or not vector or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                        raise RuntimeError("OpenAI embeddings returned invalid vector")
                    vectors[idx] = vector
            if any(vector is None for vector in vectors):
                raise RuntimeError("OpenAI embeddings response missing some items")
            for value, vector in zip(missing, vectors):
                cache[(model, value)] = list(vector)
        except BaseException as exc:
            if budget is not None and call_index is not None:
                budget.mark_call_failed(call_index, exc, embedding=True)
            raise
    elif budget is not None:
        budget.embedding_cache_hits += len(values)
    return [cache[key] for key in keys]


def _chat_response(messages: list[dict], *, model: str, json_schema: Optional[dict],
                   timeout: int, temperature: float, max_output_tokens: Optional[int],
                   purpose: str, api_key: str, url: str, post_fn: PostFn,
                   current_budget_fn: Optional[BudgetResolver]) -> tuple[dict, Any, Optional[int]]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    budget = current_budget_fn() if callable(current_budget_fn) else None
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if json_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    call_index = None
    if budget is not None:
        timeout, cap, call_index = budget.reserve_call(model=model, purpose=purpose, requested_timeout=timeout,
                                                      max_output_tokens=2000 if max_output_tokens is None else max_output_tokens, messages=messages,
                                                      request_payload=payload)
        payload["max_completion_tokens"] = cap
    elif max_output_tokens is not None:
        if int(max_output_tokens) <= 0:
            raise ValueError("Invalid output limit")
        payload["max_completion_tokens"] = int(max_output_tokens)
    data = _send_json(url=url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      payload=payload, timeout=timeout, post_fn=post_fn, budget=budget, call_index=call_index)
    return data, budget, call_index


def _chat_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenAI chat returned no choices")
    choice = choices[0]
    if choice.get("finish_reason") in {"length", "content_filter"}:
        raise RuntimeError("OpenAI chat output incomplete or filtered")
    msg = choice.get("message") or {}
    if msg.get("refusal"):
        raise RuntimeError("OpenAI chat refusal")
    content = msg.get("content", "")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text").strip()
    return str(content or "").strip()


def chat_text(messages: list[dict], *, model: Optional[str] = None, temperature: float = 0.0,
              api_key: str, default_model: str, url: str, post_fn: PostFn,
              current_budget_fn: Optional[BudgetResolver] = None,
              max_output_tokens: Optional[int] = None) -> str:
    data, budget, call_index = _chat_response(messages, model=model or default_model, json_schema=None,
                                             timeout=60, temperature=temperature, max_output_tokens=max_output_tokens,
                                             purpose="legacy_chat_text", api_key=api_key, url=url,
                                             post_fn=post_fn, current_budget_fn=current_budget_fn)
    try:
        return _chat_content(data)
    except BaseException as exc:
        if budget is not None:
            budget.mark_call_failed(call_index, exc)
        raise


def chat_json(messages: list[dict], *, model: Optional[str] = None, json_schema: Optional[dict] = None,
              timeout: int = 60, api_key: str, default_model: str, url: str, post_fn: PostFn,
              current_budget_fn: Optional[BudgetResolver] = None,
              max_output_tokens: Optional[int] = None, purpose: str = "legacy_chat_json") -> dict:
    data, budget, call_index = _chat_response(messages, model=model or default_model, json_schema=json_schema,
                                             timeout=timeout, temperature=0, max_output_tokens=max_output_tokens,
                                             purpose=purpose, api_key=api_key, url=url,
                                             post_fn=post_fn, current_budget_fn=current_budget_fn)
    try:
        text = _chat_content(data)
        if not text:
            raise RuntimeError("OpenAI chat JSON empty response")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI chat structured output is not an object")
        return parsed
    except BaseException as exc:
        if budget is not None:
            budget.mark_call_failed(call_index, exc)
        raise


def chat_json_models(messages: list[dict], *, models: Optional[list[str]] = None,
                     json_schema: Optional[dict] = None, timeout: int = 60, default_model: str,
                     normalize_models_fn: Callable[[Optional[list[str]]], list[str]],
                     chat_json_fn: Callable[..., dict]) -> dict:
    # Each delegate owns and records its attempt. A shared exhausted/cancelled budget
    # makes all further delegates fail BEFORE I/O; no unaccounted fallback is possible.
    tried = normalize_models_fn(models) or [default_model]
    last_error = None
    for model_name in tried:
        try:
            return chat_json_fn(messages, model=model_name, json_schema=json_schema, timeout=timeout)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model candidates available for JSON chat call")


def responses_json(messages: list[dict], *, model: str, json_schema: dict, effort: str,
                   reasoning_mode: str = "", timeout: int, max_output_tokens: int,
                   company_id: str, purpose: str, api_key: str, url: str, post_fn: PostFn,
                   current_budget_fn: BudgetResolver, response_text_fn: Callable[[dict], str],
                   safety_identifier_fn: Callable[[str], str]) -> dict:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    budget = current_budget_fn()
    if budget is None:
        raise RuntimeError("V13 budget context missing")
    payload = {"model": model, "input": messages, "store": False,
               "reasoning": {"effort": effort or "medium", "context": "current_turn"},
               "text": {"format": {"type": "json_schema", "name": str(json_schema.get("name") or "machinemind_v13_output"),
                                     "strict": bool(json_schema.get("strict", True)), "schema": json_schema.get("schema") or {}}},
               "safety_identifier": safety_identifier_fn(company_id)}
    if reasoning_mode:
        payload["reasoning"]["mode"] = reasoning_mode
    timeout, cap, index = budget.reserve_call(model=model, purpose=purpose, requested_timeout=timeout,
                                              max_output_tokens=max_output_tokens, messages=messages,
                                              request_payload=payload)
    payload["max_output_tokens"] = cap
    try:
        data = _send_json(url=url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          payload=payload, timeout=timeout, post_fn=post_fn, budget=budget, call_index=index)
        status = str(data.get("status") or "completed").lower()
        if status != "completed":
            raise RuntimeError(f"OpenAI Responses status={status}")
        parsed = json.loads(response_text_fn(data))
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI Responses structured output is not an object")
        return parsed
    except BaseException as exc:
        budget.mark_call_failed(index, exc)
        raise


def json_models(messages: list[dict], *, models: list[str], json_schema: dict, effort: str,
                reasoning_mode: str, timeout: int, max_output_tokens: int, company_id: str,
                purpose: str, default_model: str,
                normalize_models_fn: Callable[[Optional[list[str]]], list[str]],
                current_budget_fn: BudgetResolver, responses_json_fn: Callable[..., dict],
                chat_json_fn: Callable[..., dict], budget_exceeded_type: Type[BaseException]) -> tuple[dict, str]:
    candidates = normalize_models_fn(models) or [default_model]
    errors, failed_attempts = [], 0
    for model in candidates:
        budget = current_budget_fn()
        if budget is None:
            raise RuntimeError("V13 budget context missing")
        calls_before = int(budget.llm_calls)
        try:
            if str(model).startswith("gpt-5.6"):
                parsed = responses_json_fn(messages, model=model, json_schema=json_schema, effort=effort,
                                           reasoning_mode=reasoning_mode, timeout=timeout, max_output_tokens=max_output_tokens,
                                           company_id=company_id, purpose=purpose)
            else:
                # This delegate now reserves, caps and settles its own receipt, exactly
                # like Responses. Never fake an empty usage receipt after success.
                parsed = chat_json_fn(messages, model=model, json_schema=json_schema, timeout=timeout,
                                      max_output_tokens=max_output_tokens, purpose=purpose)
            if failed_attempts:
                budget.grant_retry_allowance(failed_attempts=failed_attempts, reason=f"{purpose}:model_fallback_succeeded")
            return parsed, model
        except budget_exceeded_type:
            raise  # do not disguise a budget/deadline guard as a provider failure
        except Exception as exc:
            if int(budget.llm_calls) > calls_before:
                failed_attempts += 1
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:300]}")
    raise RuntimeError("All V13 model calls failed: " + " | ".join(errors)[:1800])
