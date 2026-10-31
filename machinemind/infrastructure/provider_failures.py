"""Bounded failure observations; no provider text, billing or retry decisions."""
from __future__ import annotations

import json

FAILURE_CODES = frozenset({
    "output_limit", "content_filter", "response_incomplete", "response_failed",
    "response_cancelled", "response_not_completed", "refusal", "empty_output",
    "invalid_json", "invalid_shape", "duplicate_key", "http_4xx", "http_5xx",
    "http_other", "timeout", "unclassified",
})


class ProviderResponseError(RuntimeError):
    """A local classification from an observed provider field, never raw text."""
    def __init__(self, code: str):
        self.failure_code = code if type(code) is str and code in FAILURE_CODES else "unclassified"
        super().__init__("OpenAI provider failure: " + self.failure_code)


def http_failure(status) -> ProviderResponseError:
    if type(status) is int and 400 <= status < 500:
        return ProviderResponseError("http_4xx")
    if type(status) is int and 500 <= status < 600:
        return ProviderResponseError("http_5xx")
    return ProviderResponseError("http_other")


def response_status_failure(data: dict) -> ProviderResponseError:
    status = str(data.get("status") or "completed").lower()
    if status == "incomplete":
        details = data.get("incomplete_details")
        reason = details.get("reason") if type(details) is dict else None
        if reason == "max_output_tokens":
            return ProviderResponseError("output_limit")
        if reason == "content_filter":
            return ProviderResponseError("content_filter")
        return ProviderResponseError("response_incomplete")
    return ProviderResponseError({
        "failed": "response_failed", "cancelled": "response_cancelled",
    }.get(status, "response_not_completed"))


def failure_code(error) -> str:
    if isinstance(error, ProviderResponseError):
        code = error.failure_code
        return code if type(code) is str and code in FAILURE_CODES else "unclassified"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if type(error) is ValueError and error.args == ("structured_output_duplicate_key",):
        return "duplicate_key"
    if isinstance(error, TimeoutError) or any(
        base.__module__ == "requests.exceptions" and base.__name__ == "Timeout"
        for base in type(error).__mro__
    ):
        return "timeout"
    return "unclassified"
