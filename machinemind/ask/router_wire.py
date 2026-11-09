"""Lossless private JSON wire for the existing protected ASK semantic router.

Only wire names and two deterministic/duplicate fields differ. Semantic values,
global probes, source policy, literal quotes and ownership choices stay with the
same model call. Expansion precedes the existing semantic normalizers.
"""
from __future__ import annotations

from copy import deepcopy
import math

VERSION = "protected-ask-compact-wire-v1"
ALIASES = {
    "request_kind": "k", "information_task": "t", "required_answer_types": "a",
    "confidence": "c", "requested_mode_fit": "fit", "evidence_state": "e",
    "evidence_policy": "p", "relevant_evidence_ids": "i", "preferred_source_types": "s",
    "source_type_policy": "sp", "dense_queries": "d", "lexical_queries": "l",
    "exact_terms": "x", "facet_queries": "f", "diagnostic_subsystems": "ds",
    "diagnostic_observables": "do", "diagnostic_operating_conditions": "dc",
    "diagnostic_discriminants": "dd", "diagnostic_exclusions": "de",
    "missing_information": "mi", "clarification_question": "cq", "safety_reason": "sr",
    "out_of_scope_reason": "or", "rationale": "r", "scalar_target": "st",
    "request_readiness": "rr", "procedure_scope": "ps", "procedure_phase": "ph",
    "factual_coverage": "fc", "selected_procedure_parent_id": "pp",
}
FACET_ALIASES = {"facet": "f", "answer_type": "a", "must_cover": "m",
    "dense_queries": "d", "lexical_queries": "l", "exact_terms": "x",
    "preferred_source_types": "s"}
DERIVED = frozenset({"effective_mode", "required_facets"})
INSTRUCTION = (
    " PRIVATE COMPACT ASK WIRE: return only the schema's short keys. Each field description "
    "gives its original semantic name; its meaning is unchanged. effective_mode is fixed to "
    "ask by the request owner, so do not emit it. The mandatory required_facets list is derived "
    "from f entries whose m (must_cover) is true: put EVERY independently requested mandatory "
    "category there, at its full original breadth, even when its retrieval probes are empty. "
    "Do not omit a mandatory category, shape, unit, applicability qualifier or phase exclusion. "
    "Global d/l/x are the original global dense_queries/lexical_queries/exact_terms: keep any "
    "useful full-request probes or anchors absent from individual facet probes; they are not "
    "derived or replaced by a facet union. All scalar targets, literal phase quotes, readiness, "
    "factual coverage, populated diagnostic arrays, unknown information and refusal reasons "
    "retain their original meanings. pp is the existing nullable selected_procedure_parent_id, "
    "when present: choose only an exactly shown emitted ID or null. Do not copy an optional "
    "relevance ID into pp merely because it exists."
)


class RouterWireError(ValueError):
    """Malformed private semantic wire; never a guessed/degraded decision."""


def _need(value, reason):
    if not value:
        raise RouterWireError(reason)


def schema(full_schema):
    """Clone the existing supported JSON schema; do not mutate another mode."""
    out = deepcopy(full_schema)
    body = out["schema"]
    props = body["properties"]
    _need(set(props).issubset(set(ALIASES) | DERIVED), "unknown_canonical_router_field")
    _need(len(set(ALIASES.values())) == len(ALIASES), "router_alias_collision")
    _need(props.get("effective_mode", {}).get("enum") == ["ask"], "ask_only_wire_required")
    facets = props.get("facet_queries", {}).get("items", {})
    _need(set(facets.get("properties", {})) == set(FACET_ALIASES), "unknown_facet_router_field")
    translated = {}
    for name, field in props.items():
        if name in DERIVED:
            continue
        field = deepcopy(field)
        field["description"] = name + (": " + field["description"] if field.get("description") else "")
        if name == "facet_queries":
            item = field["items"]
            item["properties"] = {
                FACET_ALIASES[key]: {**value, "description": key +
                    (": " + value["description"] if value.get("description") else "")}
                for key, value in item["properties"].items()
            }
            item["required"] = [FACET_ALIASES[key] for key in item["required"]]
        translated[ALIASES[name]] = field
    body["properties"] = translated
    body["required"] = [ALIASES[name] for name in body["required"] if name not in DERIVED]
    out["name"] = "machinemind_ask_compact_router_v1"
    return out


def _validate(value, rule):
    """Strict local validation of the schema subset used by this private wire."""
    if "anyOf" in rule:
        for choice in rule["anyOf"]:
            try:
                _validate(value, choice)
                return
            except RouterWireError:
                pass
        raise RouterWireError("invalid_wire_alternative")
    expected = rule.get("type")
    types = expected if type(expected) is list else [expected]
    matches = {
        "object": type(value) is dict, "array": type(value) is list,
        "string": type(value) is str, "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) in (int, float) and math.isfinite(value),
        "null": value is None,
    }
    _need(any(matches.get(kind, False) for kind in types), "invalid_wire_type")
    if "enum" in rule:
        _need(any(type(value) is type(item) and value == item for item in rule["enum"]), "invalid_wire_enum")
    if type(value) is dict:
        props = rule.get("properties", {})
        _need(set(rule.get("required", ())).issubset(value), "missing_wire_field")
        _need(rule.get("additionalProperties") is False and set(value).issubset(props), "unknown_wire_field")
        for key, item in value.items():
            _validate(item, props[key])
    elif type(value) is list:
        _need(len(value) >= rule.get("minItems", 0) and len(value) <= rule.get("maxItems", math.inf), "wire_array_capacity")
        for item in value:
            _validate(item, rule["items"])
    elif type(value) is str:
        _need(len(value) >= rule.get("minLength", 0) and len(value) <= rule.get("maxLength", math.inf), "wire_string_capacity")
    elif type(value) in (int, float):
        _need(rule.get("minimum", -math.inf) <= value <= rule.get("maximum", math.inf), "wire_number_range")


def expand(value, full_schema, *, allowed_effective_modes):
    """Decode the exact model wire, preserving every semantic field verbatim."""
    _need(type(allowed_effective_modes) is tuple and allowed_effective_modes == ("ask",), "ask_only_wire_required")
    compact = schema(full_schema)
    _validate(value, compact["schema"])
    reverse = {alias: name for name, alias in ALIASES.items()}
    result = {reverse[key]: deepcopy(item) for key, item in value.items()}
    result["facet_queries"] = [
        {next(name for name, alias in FACET_ALIASES.items() if alias == key): deepcopy(item)
            for key, item in facet.items()}
        for facet in result["facet_queries"]
    ]
    result["effective_mode"] = "ask"
    result["required_facets"] = [facet["facet"] for facet in result["facet_queries"] if facet["must_cover"] is True]
    _validate(result, full_schema["schema"])
    return result
