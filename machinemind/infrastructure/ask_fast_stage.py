"""Opt-in ASK stage provider tariff and request-local authorization."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import re


FLAG = "MM_ASK_NATIVE_FAST_PROFILE"
VERSION = "stage-gpt6-fast-v1"
ROUTER = "assistant_core_v2_semantic_router"
WRITER = "ask_final_synthesis"
REVIEW = "assistant_core_answer_contract_verifier"
ENDPOINT = "https://api.openai.com/v1/responses"
MAX_SHORT_INPUT_TOKENS = 200_000
TIERS = frozenset({"fast", "priority", "default"})
_MODELS = {
    "gpt-5.6-terra": ("gpt-6.1-terra", (2.5, 15.0)),
    "gpt-5.6-luna": ("gpt-6-luna", (1.0, 6.0)),
}
_PERMIT = ContextVar("ask_fast_stage_permit", default=None)
_ATTEMPT = ContextVar("ask_fast_stage_attempt", default=None)


class UnsupportedEnvelope(ValueError):
    """The Fast tariff is inapplicable before any reservation or POST."""


@dataclass(frozen=True, slots=True)
class Attempt:
    budget: object
    purpose: str
    configured_model: str
    model: str
    rates: tuple[float, float]


def _owner(budget):
    from machinemind.ask import request_completion, routing_runtime
    from . import request_budget

    completion, execution = request_completion.current(), routing_runtime._ACTIVE.get()
    if (type(execution) is routing_runtime.ProtectedExecution and execution.active
            and completion is not None and completion.budget is budget
            and request_budget._v13_current_budget() is budget and budget.mode == "assistant_core_ask"):
        return completion
    return None


@contextmanager
def allow(purpose, *, eligible):
    """The owned producer/consumer supplies eligibility, never request text."""
    token = _PERMIT.set(purpose if eligible else None)
    try:
        yield
    finally:
        _PERMIT.reset(token)


def select(budget, *, purpose, model, candidates, json_schema, company_id):
    if (os.environ.get(FLAG) != VERSION or model not in _MODELS
            or any(item not in _MODELS for item in candidates)):
        return None
    if purpose == ROUTER:
        if (len(candidates) != 1 or type(json_schema) is not dict
                or json_schema.get("name") != "machinemind_ask_compact_router_v1"):
            return None
    elif purpose not in {WRITER, REVIEW} or _PERMIT.get() != purpose:
        return None
    owner = _owner(budget)
    if (owner is None or str(owner.payload.company_id) != str(company_id)
            or not 0 < budget.max_estimated_cost_usd <= .25):
        return None
    from . import request_budget

    actual, floor = _MODELS[model]
    rates = tuple(request_budget._positive_rate(rate) for rate in request_budget._v13_model_rates(model))
    if any(rate < minimum for rate, minimum in zip(rates, floor, strict=True)):
        return None
    return Attempt(budget, purpose, model, actual, rates)


@contextmanager
def bind(attempt):
    token = _ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _ATTEMPT.reset(token)


def current(budget, *, purpose, model):
    attempt = _ATTEMPT.get()
    if (attempt is not None and attempt.budget is budget and attempt.purpose == purpose
            and attempt.model == model and _owner(budget) is not None):
        return attempt
    return None


def check_payload(attempt, payload):
    return (type(payload) is dict and payload.get("model") == attempt.model
            and payload.get("store") is False and payload.get("service_tier") == "fast"
            and payload.get("prompt_cache_options") == {"ttl": "30m"})


def response_model_matches(requested, actual):
    return (type(actual) is str and (actual == requested or re.fullmatch(
        re.escape(requested) + r"-[0-9]{4}-[0-9]{2}-[0-9]{2}", actual) is not None))
