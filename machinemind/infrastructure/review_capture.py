"""Opt-in, request-local observation of the existing Root Cause provider call.

This module never sends a request, retries, changes a payload, or inspects auth
headers. A trace is returned only to the already-authorized debug response. It
contains customer source texts and must be treated as a confidential test file.
No files are persisted by the server. Serialization/size failures fail capture,
not the existing diagnosis. Hashes cover canonical JSON, not raw HTTP bytes.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable
import hashlib
import json
import time
from urllib.parse import urlsplit

POLICY_VERSION = "root-review-capture-v1"
PURPOSE = "assistant_core_root_cause_adjudicator"
MAX_CAPTURE_BYTES = 2 * 1024 * 1024
BODY_KEYS = frozenset({"model", "input", "store", "reasoning", "text", "safety_identifier", "max_output_tokens"})
RESPONSE_HEADERS = ("x-request-id", "openai-processing-ms", "openai-version", "content-type")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def clone(value: Any) -> Any:
    data = canonical(value).encode("utf-8")
    if len(data) > MAX_CAPTURE_BYTES:
        raise ValueError("capture_size_limit")
    return json.loads(data)


@dataclass
class Capture:
    fixture: dict = field(default_factory=dict)
    provider: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    dispatch_count: int = 0


_ACTIVE: ContextVar[Capture | None] = ContextVar("mm_root_review_capture", default=None)


def begin(*, enabled: bool, fixture: dict) -> tuple[Capture | None, Token | None]:
    if not enabled:
        # Clear a possibly enclosing context: an explicit non-debug call must not
        # inherit any trace belonging to another scope or nested debug operation.
        return None, _ACTIVE.set(None)
    capture = Capture()
    try:
        capture.fixture = clone(fixture)
    except Exception as exc:
        capture.errors.append("fixture_capture_" + type(exc).__name__)
    return capture, _ACTIVE.set(capture)


def end(token: Token | None) -> None:
    if token is not None:
        _ACTIVE.reset(token)


def observe_post(post_fn: Callable[..., Any], *, purpose: str) -> Callable[..., Any]:
    capture = _ACTIVE.get()
    if capture is None or purpose != PURPOSE:
        return post_fn

    def traced(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        capture.dispatch_count += 1
        event: dict[str, Any] = {}
        if capture.dispatch_count != 1:
            capture.errors.append("multiple_provider_dispatches")
        try:
            body = kwargs.get("json")
            if not isinstance(body, dict) or not set(body).issubset(BODY_KEYS):
                raise ValueError("unrecognized_request_body")
            body_copy = clone(body)
            url = args[0] if args else kwargs.get("url", "")
            parsed = urlsplit(str(url))
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("unsafe_endpoint_to_capture")
            event.update({
                "endpoint": str(url), "method": "POST", "body": body_copy,
                "body_sha256": digest(body_copy), "timeout": clone(kwargs.get("timeout")),
                "allow_redirects": kwargs.get("allow_redirects"),
                "headers_captured": False, "output_cap_is_effective": True,
            })
        except Exception as exc:
            capture.errors.append("request_capture_" + type(exc).__name__)
        if capture.dispatch_count == 1:
            capture.provider = event
        # The ORIGINAL arguments and object identities are forwarded exactly once.
        # Do not read response body: leave accounting and parsing to the transport.
        try:
            response = post_fn(*args, **kwargs)
        except BaseException as exc:
            event.update({"outcome": "exception", "exception_type": type(exc).__name__,
                          "elapsed_seconds": round(time.monotonic() - started, 6)})
            raise
        else:
            event.update({"outcome": "http_response", "elapsed_seconds": round(time.monotonic() - started, 6)})
            try:
                event["http_status"] = int(response.status_code)
                headers = response.headers
                event["response_headers"] = {name: str(headers[name])[:256]
                                               for name in RESPONSE_HEADERS if name in headers}
            except Exception:
                event["response_metadata_available"] = False
            return response
    return traced


def snapshot(capture: Capture | None) -> dict | None:
    if capture is None:
        return None
    try:
        result = {
            "policy_version": POLICY_VERSION,
            "confidential_source_material": True,
            "headers_captured": False,
            "automatic_replay": False,
            "dispatch_count": capture.dispatch_count,
            "capture_errors": list(capture.errors),
            "fixture": capture.fixture,
            "provider_request": capture.provider,
        }
        result["complete"] = bool(capture.fixture and capture.dispatch_count == 1
                                  and capture.provider.get("body_sha256") and not capture.errors)
        result["fingerprint"] = digest(result)
        return clone(result)
    except Exception as exc:
        return {"policy_version": POLICY_VERSION, "complete": False,
                "capture_errors": ["snapshot_" + type(exc).__name__], "automatic_replay": False}
