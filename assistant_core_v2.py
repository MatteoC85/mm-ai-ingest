"""MachineMind unified Assistant Core V2.

This module contains only orchestration and response-contract logic for ASK,
Root Cause and Smart Diagnostic. Infrastructure-specific retrieval, database
access, model calls and synthesis are injected by ``main.py`` through hooks.

Design goals:
- one semantic understanding layer shared by all three tools;
- the UI-selected tool is a preference, not a reason for a false ``no_sources``;
- machine-specific claims require indexed evidence;
- generic engineering knowledge is allowed only for genuinely generic technical
  questions and is labelled as such;
- non-technical and unsafe requests get explicit outcomes;
- response citations and links are kept on one evidence manifest;
- response shapes remain compatible with the existing Worker and Bubble UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


MODE_ASK = "ask"
MODE_ROOT_CAUSE = "root_cause"
MODE_SMART_DIAGNOSTIC = "smart_diagnostic"
VALID_MODES = {MODE_ASK, MODE_ROOT_CAUSE, MODE_SMART_DIAGNOSTIC}

RESULT_ANSWERED = "ANSWERED"
RESULT_MODE_ROUTED = "MODE_ROUTED"
RESULT_NO_MACHINE_EVIDENCE = "NO_MACHINE_EVIDENCE"
RESULT_GENERAL_GUIDANCE = "GENERAL_GUIDANCE"
RESULT_NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
RESULT_OUT_OF_SCOPE = "OUT_OF_SCOPE"
RESULT_SAFETY_REFUSAL = "SAFETY_REFUSAL"
RESULT_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
RESULT_TIMEOUT = "TIMEOUT"
RESULT_TECHNICAL_ERROR = "TECHNICAL_ERROR"

EVIDENCE_SUPPORTED = "supported"
EVIDENCE_PARTIAL = "partial"
EVIDENCE_REFINE = "refine"
EVIDENCE_UNSUPPORTED = "unsupported"
EVIDENCE_CLARIFY = "clarify"
VALID_EVIDENCE_STATES = {
    EVIDENCE_SUPPORTED,
    EVIDENCE_PARTIAL,
    EVIDENCE_REFINE,
    EVIDENCE_UNSUPPORTED,
    EVIDENCE_CLARIFY,
}

POLICY_MACHINE_REQUIRED = "machine_sources_required"
POLICY_MACHINE_PREFERRED = "machine_sources_preferred"
POLICY_GENERAL_ALLOWED = "general_technical_allowed"
VALID_EVIDENCE_POLICIES = {
    POLICY_MACHINE_REQUIRED,
    POLICY_MACHINE_PREFERRED,
    POLICY_GENERAL_ALLOWED,
}

KIND_FACTUAL = "factual"
KIND_PROCEDURE = "procedure"
KIND_SOURCE_RETRIEVAL = "source_retrieval"
KIND_COMPARISON = "comparison"
KIND_FAULT_DIAGNOSTIC = "fault_diagnostic"
KIND_GUIDED_DIAGNOSTIC = "guided_diagnostic"
KIND_GENERAL_TECHNICAL = "general_technical"
KIND_AMBIGUOUS = "ambiguous"
KIND_OUT_OF_SCOPE = "out_of_scope"
KIND_UNSAFE_REQUEST = "unsafe_request"

REQUEST_KINDS = {
    KIND_FACTUAL,
    KIND_PROCEDURE,
    KIND_SOURCE_RETRIEVAL,
    KIND_COMPARISON,
    KIND_FAULT_DIAGNOSTIC,
    KIND_GUIDED_DIAGNOSTIC,
    KIND_GENERAL_TECHNICAL,
    KIND_AMBIGUOUS,
    KIND_OUT_OF_SCOPE,
    KIND_UNSAFE_REQUEST,
}

# Fine-grained semantic output contract. These values are language-independent and
# describe the shape of information the answer must contain. They are deliberately
# separate from request_kind, which chooses the broad assistant mode.
INFO_PROCEDURE_FULL = "procedure_full"
INFO_PROCEDURE_SEGMENT = "procedure_segment"
INFO_NUMERIC_SPECIFICATION = "numeric_specification"
INFO_INTERFACE_NAVIGATION = "interface_navigation"
INFO_SEQUENCE_SYNCHRONIZATION = "sequence_or_synchronization"
INFO_DOCUMENT_EXPLANATION = "document_explanation"
INFO_SOURCE_RETRIEVAL = "source_retrieval"
INFO_FAULT_DIAGNOSTIC = "fault_diagnostic"
INFO_COMPARISON = "comparison"
INFO_GENERAL_TECHNICAL = "general_technical"
INFO_OUT_OF_SCOPE = "out_of_scope"
INFO_OTHER = "other"

INFORMATION_TASKS = {
    INFO_PROCEDURE_FULL,
    INFO_PROCEDURE_SEGMENT,
    INFO_NUMERIC_SPECIFICATION,
    INFO_INTERFACE_NAVIGATION,
    INFO_SEQUENCE_SYNCHRONIZATION,
    INFO_DOCUMENT_EXPLANATION,
    INFO_SOURCE_RETRIEVAL,
    INFO_FAULT_DIAGNOSTIC,
    INFO_COMPARISON,
    INFO_GENERAL_TECHNICAL,
    INFO_OUT_OF_SCOPE,
    INFO_OTHER,
}

PRECISION_INFORMATION_TASKS = {
    INFO_NUMERIC_SPECIFICATION,
    INFO_INTERFACE_NAVIGATION,
    INFO_SEQUENCE_SYNCHRONIZATION,
    INFO_PROCEDURE_SEGMENT,
    INFO_FAULT_DIAGNOSTIC,
}

# A request can require more than one output shape. ``information_task`` remains
# the primary routing class, while these requirements make composite questions
# (for example a numeric limit plus a checklist) explicit and enforceable.
REQ_NUMERIC_VALUE = "numeric_value"
REQ_ORDERED_ACTIONS = "ordered_actions"
REQ_CHECKLIST = "checklist"
REQ_SAFETY_CONDITIONS = "safety_conditions"
REQ_INTERFACE_LOCATIONS = "interface_locations"
REQ_STATE_SEQUENCE = "state_sequence"
REQ_DIAGNOSTIC_CAUSES = "diagnostic_causes"
REQ_COMPARISON = "comparison"
REQ_SOURCE_LOCATIONS = "source_locations"
REQ_EXPLANATION = "explanation"

ANSWER_REQUIREMENTS = {
    REQ_NUMERIC_VALUE,
    REQ_ORDERED_ACTIONS,
    REQ_CHECKLIST,
    REQ_SAFETY_CONDITIONS,
    REQ_INTERFACE_LOCATIONS,
    REQ_STATE_SEQUENCE,
    REQ_DIAGNOSTIC_CAUSES,
    REQ_COMPARISON,
    REQ_SOURCE_LOCATIONS,
    REQ_EXPLANATION,
}

PRECISION_ANSWER_REQUIREMENTS = {
    REQ_NUMERIC_VALUE,
    REQ_ORDERED_ACTIONS,
    REQ_CHECKLIST,
    REQ_INTERFACE_LOCATIONS,
    REQ_STATE_SEQUENCE,
    REQ_DIAGNOSTIC_CAUSES,
}

SOURCE_TYPES = {
    "document",
    "procedure",
    "step",
    "ps",
    "md_photo",
    "md_video",
}


@dataclass(frozen=True)
class AssistantCoreFacetQuery:
    """Language-independent retrieval plan for one mandatory answer facet.

    The semantic router may provide faithful Italian and English query variants.
    Retrieval uses these variants independently so a strong result for one facet
    cannot hide a missing result for another facet in a composite question.
    """

    facet: str
    answer_type: str = REQ_EXPLANATION
    must_cover: bool = True
    dense_queries: tuple[str, ...] = ()
    lexical_queries: tuple[str, ...] = ()
    exact_terms: tuple[str, ...] = ()
    preferred_source_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssistantCoreRequest:
    query: str
    requested_mode: str
    response_language: str
    company_id: str
    machine_id: str
    ai_scope: str
    top_k: int
    max_causes: int = 3
    narrow_scope: bool = False
    allowed_effective_modes: tuple[str, ...] = (MODE_ASK, MODE_ROOT_CAUSE)
    debug: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantCoreDecision:
    request_kind: str
    effective_mode: str
    confidence: float
    requested_mode_fit: bool
    evidence_state: str
    evidence_policy: str
    information_task: str = INFO_OTHER
    required_answer_types: tuple[str, ...] = ()
    relevant_evidence_ids: tuple[str, ...] = ()
    preferred_source_types: tuple[str, ...] = ()
    source_type_policy: str = "none"
    dense_queries: tuple[str, ...] = ()
    lexical_queries: tuple[str, ...] = ()
    exact_terms: tuple[str, ...] = ()
    required_facets: tuple[str, ...] = ()
    facet_queries: tuple[AssistantCoreFacetQuery, ...] = ()
    diagnostic_subsystems: tuple[str, ...] = ()
    diagnostic_observables: tuple[str, ...] = ()
    diagnostic_operating_conditions: tuple[str, ...] = ()
    diagnostic_discriminants: tuple[str, ...] = ()
    diagnostic_exclusions: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    clarification_question: str = ""
    safety_reason: str = ""
    out_of_scope_reason: str = ""
    rationale: str = ""
    router_model: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def routed(self) -> bool:
        return self.effective_mode not in {"", "unknown"}

    @property
    def diagnostic_clues(self) -> tuple[str, ...]:
        """All positive diagnostic observations, ordered by discriminating value."""
        values = (
            list(self.diagnostic_discriminants)
            + list(self.diagnostic_operating_conditions)
            + list(self.diagnostic_observables)
            + list(self.diagnostic_subsystems)
        )
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = _clean_text(value, 300)
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= 16:
                break
        return tuple(out)


@dataclass(frozen=True)
class AssistantCoreHooks:
    retrieve_neutral: Callable[[AssistantCoreRequest], dict]
    route_semantically: Callable[[AssistantCoreRequest, dict], Mapping[str, Any]]
    prepare_evidence: Callable[[AssistantCoreRequest, dict, AssistantCoreDecision], Mapping[str, Any]]
    synthesize_ask: Callable[[AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    synthesize_root_cause: Callable[[AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    synthesize_general: Callable[[AssistantCoreRequest, AssistantCoreDecision], dict]
    build_no_evidence: Callable[[AssistantCoreRequest, AssistantCoreDecision, dict], dict]
    build_clarification: Callable[[AssistantCoreRequest, AssistantCoreDecision], dict]
    build_out_of_scope: Callable[[AssistantCoreRequest, AssistantCoreDecision], dict]
    build_safety_refusal: Callable[[AssistantCoreRequest, AssistantCoreDecision], dict]
    refine_retrieval: Optional[
        Callable[[AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    ] = None
    synthesize_smart_start: Optional[
        Callable[[AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    ] = None
    validate_response: Optional[
        Callable[[dict, AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    ] = None
    repair_response: Optional[
        Callable[[dict, AssistantCoreRequest, dict, AssistantCoreDecision], dict]
    ] = None


def _clean_text(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _unique_strings(values: Any, limit: int) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value, 240)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


def _normalize_facet_queries(
    values: Any,
    required_facets: Sequence[str],
    *,
    default_answer_types: Sequence[str] = (),
    default_source_types: Sequence[str] = (),
) -> tuple[AssistantCoreFacetQuery, ...]:
    """Normalize the router's per-facet retrieval plan.

    Strict router output normally supplies every field. The defaults keep the
    contract useful during a bounded fallback or a rolling deployment: when one
    global answer requirement/source family exists, a missing facet annotation
    inherits it instead of silently degrading to a generic explanation.
    """
    raw_values = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []
    default_types = tuple(
        item.lower()
        for item in _unique_strings(default_answer_types, 8)
        if item.lower() in ANSWER_REQUIREMENTS
    )
    default_sources = tuple(
        item.lower()
        for item in _unique_strings(default_source_types, 6)
        if item.lower() in SOURCE_TYPES
    )
    fallback_type = default_types[0] if len(default_types) == 1 else REQ_EXPLANATION

    by_key: dict[str, AssistantCoreFacetQuery] = {}
    for raw in raw_values:
        if not isinstance(raw, Mapping):
            continue
        facet = _clean_text(raw.get("facet"), 240)
        if not facet:
            continue
        answer_type = _clean_text(raw.get("answer_type"), 80).lower()
        if answer_type not in ANSWER_REQUIREMENTS:
            answer_type = fallback_type
        preferred = tuple(
            item.lower()
            for item in _unique_strings(raw.get("preferred_source_types"), 6)
            if item.lower() in SOURCE_TYPES
        ) or default_sources
        item = AssistantCoreFacetQuery(
            facet=facet,
            answer_type=answer_type,
            must_cover=bool(raw.get("must_cover", True)),
            dense_queries=_unique_strings(raw.get("dense_queries"), 4),
            lexical_queries=_unique_strings(raw.get("lexical_queries"), 5),
            exact_terms=_unique_strings(raw.get("exact_terms"), 8),
            preferred_source_types=preferred,
        )
        by_key[facet.casefold()] = item
        if len(by_key) >= 10:
            break

    # Every required facet gets an explicit retrieval plan even if the router did
    # not provide translations. The facet text itself remains a safe fallback.
    for facet in _unique_strings(required_facets, 10):
        key = facet.casefold()
        if key not in by_key:
            by_key[key] = AssistantCoreFacetQuery(
                facet=facet,
                answer_type=fallback_type,
                must_cover=True,
                dense_queries=(facet,),
                lexical_queries=(facet,),
                exact_terms=(),
                preferred_source_types=default_sources,
            )
    return tuple(by_key.values())


def build_router_schema(allowed_modes: Sequence[str]) -> dict:
    modes = [m for m in allowed_modes if m in VALID_MODES]
    if not modes:
        modes = [MODE_ASK]
    return {
        "name": "machinemind_assistant_core_router_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "request_kind": {
                    "type": "string",
                    "enum": sorted(REQUEST_KINDS),
                },
                "information_task": {
                    "type": "string",
                    "enum": sorted(INFORMATION_TASKS),
                },
                "required_answer_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(ANSWER_REQUIREMENTS)},
                    "maxItems": 8,
                },
                "effective_mode": {"type": "string", "enum": modes},
                "confidence": {"type": "number"},
                "requested_mode_fit": {"type": "boolean"},
                "evidence_state": {
                    "type": "string",
                    "enum": sorted(VALID_EVIDENCE_STATES),
                },
                "evidence_policy": {
                    "type": "string",
                    "enum": sorted(VALID_EVIDENCE_POLICIES),
                },
                "relevant_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 16,
                },
                "preferred_source_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(SOURCE_TYPES)},
                    "maxItems": 6,
                },
                "source_type_policy": {
                    "type": "string",
                    "enum": ["none", "prefer", "require"],
                },
                "dense_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
                "lexical_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 7,
                },
                "exact_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 12,
                },
                "required_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10,
                },
                "facet_queries": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "facet": {"type": "string"},
                            "answer_type": {
                                "type": "string", "enum": sorted(ANSWER_REQUIREMENTS)
                            },
                            "must_cover": {"type": "boolean"},
                            "dense_queries": {
                                "type": "array", "items": {"type": "string"}, "maxItems": 4
                            },
                            "lexical_queries": {
                                "type": "array", "items": {"type": "string"}, "maxItems": 5
                            },
                            "exact_terms": {
                                "type": "array", "items": {"type": "string"}, "maxItems": 8
                            },
                            "preferred_source_types": {
                                "type": "array",
                                "items": {"type": "string", "enum": sorted(SOURCE_TYPES)},
                                "maxItems": 6,
                            },
                        },
                        "required": [
                            "facet", "answer_type", "must_cover", "dense_queries",
                            "lexical_queries", "exact_terms", "preferred_source_types"
                        ],
                    },
                },
                "diagnostic_subsystems": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 6
                },
                "diagnostic_observables": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 8
                },
                "diagnostic_operating_conditions": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 8
                },
                "diagnostic_discriminants": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 8
                },
                "diagnostic_exclusions": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 8
                },
                "missing_information": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "clarification_question": {"type": "string"},
                "safety_reason": {"type": "string"},
                "out_of_scope_reason": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": [
                "request_kind",
                "information_task",
                "required_answer_types",
                "effective_mode",
                "confidence",
                "requested_mode_fit",
                "evidence_state",
                "evidence_policy",
                "relevant_evidence_ids",
                "preferred_source_types",
                "source_type_policy",
                "dense_queries",
                "lexical_queries",
                "exact_terms",
                "required_facets",
                "facet_queries",
                "diagnostic_subsystems",
                "diagnostic_observables",
                "diagnostic_operating_conditions",
                "diagnostic_discriminants",
                "diagnostic_exclusions",
                "missing_information",
                "clarification_question",
                "safety_reason",
                "out_of_scope_reason",
                "rationale",
            ],
        },
    }


def _fallback_decision(request: AssistantCoreRequest, reason: str) -> AssistantCoreDecision:
    # ASK is the safest universal degraded mode: it can explain a symptom without
    # forcing a root-cause format, while a procedural question must never be forced
    # through Root Cause merely because that button was pressed.
    allowed = tuple(m for m in request.allowed_effective_modes if m in VALID_MODES)
    if not allowed:
        allowed = (MODE_ASK,)
    effective = MODE_ASK if MODE_ASK in allowed else allowed[0]
    return AssistantCoreDecision(
        request_kind="ambiguous",
        effective_mode=effective,
        confidence=0.0,
        requested_mode_fit=(effective == request.requested_mode),
        evidence_state=EVIDENCE_UNSUPPORTED,
        evidence_policy=POLICY_MACHINE_REQUIRED,
        information_task=INFO_OTHER,
        required_answer_types=(),
        rationale="Semantic router unavailable; fail closed rather than answer from unrelated evidence.",
        degraded=True,
        degraded_reason=_clean_text(reason, 400) or "router_unavailable",
    )


def normalize_decision(
    request: AssistantCoreRequest,
    raw: Optional[Mapping[str, Any]],
    *,
    router_error: str = "",
) -> AssistantCoreDecision:
    if not isinstance(raw, Mapping):
        return _fallback_decision(request, router_error or "invalid_router_payload")

    allowed = tuple(m for m in request.allowed_effective_modes if m in VALID_MODES)
    if not allowed:
        allowed = (MODE_ASK,)

    effective = _clean_text(raw.get("effective_mode"), 40).lower()
    if effective not in allowed:
        effective = MODE_ASK if MODE_ASK in allowed else allowed[0]

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
    except Exception:
        confidence = 0.0

    request_kind = _clean_text(raw.get("request_kind"), 60).lower()
    # Backward-compatible alias from early candidates; the production schema emits
    # ``out_of_scope``.
    if request_kind == "non_technical":
        request_kind = KIND_OUT_OF_SCOPE
    if request_kind not in REQUEST_KINDS:
        request_kind = KIND_AMBIGUOUS

    evidence_state = _clean_text(raw.get("evidence_state"), 40).lower()
    if evidence_state not in VALID_EVIDENCE_STATES:
        evidence_state = EVIDENCE_UNSUPPORTED

    evidence_policy = _clean_text(raw.get("evidence_policy"), 60).lower()
    if evidence_policy not in VALID_EVIDENCE_POLICIES:
        evidence_policy = POLICY_MACHINE_REQUIRED

    source_type_policy = _clean_text(raw.get("source_type_policy"), 20).lower()
    if source_type_policy not in {"none", "prefer", "require"}:
        source_type_policy = "none"

    preferred = tuple(
        x.lower()
        for x in _unique_strings(raw.get("preferred_source_types"), 6)
        if x.lower() in SOURCE_TYPES
    )
    if not preferred:
        source_type_policy = "none"

    information_task = _clean_text(raw.get("information_task"), 80).lower()
    if information_task not in INFORMATION_TASKS:
        information_task = INFO_OTHER

    required_answer_types = tuple(
        x.lower()
        for x in _unique_strings(raw.get("required_answer_types"), 8)
        if x.lower() in ANSWER_REQUIREMENTS
    )

    required_facets = _unique_strings(raw.get("required_facets"), 10)
    facet_queries = _normalize_facet_queries(
        raw.get("facet_queries"),
        required_facets,
        default_answer_types=required_answer_types,
        default_source_types=preferred,
    )

    clarification = _clean_text(raw.get("clarification_question"), 500)
    if evidence_state == EVIDENCE_CLARIFY and not clarification:
        clarification = (
            "Please add the missing machine condition or the exact information you need."
            if request.response_language.lower().startswith("en")
            else "Aggiungi la condizione della macchina o l'informazione esatta che ti serve."
        )

    # A non-technical or unsafe request must never be converted into general
    # engineering knowledge merely because evidence is absent.
    if request_kind in {KIND_OUT_OF_SCOPE, KIND_UNSAFE_REQUEST}:
        evidence_policy = POLICY_MACHINE_REQUIRED
        evidence_state = EVIDENCE_UNSUPPORTED
        information_task = INFO_OUT_OF_SCOPE if request_kind == KIND_OUT_OF_SCOPE else INFO_OTHER
        required_facets = ()
        facet_queries = ()
        raw = dict(raw)
        raw["diagnostic_subsystems"] = []
        raw["diagnostic_observables"] = []
        raw["diagnostic_operating_conditions"] = []
        raw["diagnostic_discriminants"] = []
        raw["diagnostic_exclusions"] = []
        if MODE_ASK in allowed:
            effective = MODE_ASK

    return AssistantCoreDecision(
        request_kind=request_kind,
        effective_mode=effective,
        confidence=confidence,
        requested_mode_fit=bool(raw.get("requested_mode_fit", effective == request.requested_mode)),
        evidence_state=evidence_state,
        evidence_policy=evidence_policy,
        information_task=information_task,
        required_answer_types=required_answer_types,
        relevant_evidence_ids=_unique_strings(raw.get("relevant_evidence_ids"), 16),
        preferred_source_types=preferred,
        source_type_policy=source_type_policy,
        dense_queries=_unique_strings(raw.get("dense_queries"), 5),
        lexical_queries=_unique_strings(raw.get("lexical_queries"), 7),
        exact_terms=_unique_strings(raw.get("exact_terms"), 12),
        required_facets=required_facets,
        facet_queries=facet_queries,
        diagnostic_subsystems=_unique_strings(raw.get("diagnostic_subsystems"), 6),
        diagnostic_observables=_unique_strings(raw.get("diagnostic_observables"), 8),
        diagnostic_operating_conditions=_unique_strings(raw.get("diagnostic_operating_conditions"), 8),
        diagnostic_discriminants=_unique_strings(raw.get("diagnostic_discriminants"), 8),
        diagnostic_exclusions=_unique_strings(raw.get("diagnostic_exclusions"), 8),
        missing_information=_unique_strings(raw.get("missing_information"), 8),
        clarification_question=clarification,
        safety_reason=_clean_text(raw.get("safety_reason"), 500),
        out_of_scope_reason=_clean_text(raw.get("out_of_scope_reason"), 500),
        rationale=_clean_text(raw.get("rationale"), 700),
        router_model=_clean_text(raw.get("router_model"), 100),
    )


def _status_to_result_code(response: Mapping[str, Any], routed: bool) -> str:
    explicit = _clean_text(response.get("result_code"), 80).upper()
    if explicit:
        return explicit
    status = _clean_text(response.get("status"), 80).lower()
    if status == "answered":
        return RESULT_MODE_ROUTED if routed else RESULT_ANSWERED
    if status == "no_sources":
        return RESULT_NO_MACHINE_EVIDENCE
    if status == "needs_clarification":
        return RESULT_NEEDS_CLARIFICATION
    if status == "out_of_scope":
        return RESULT_OUT_OF_SCOPE
    if status == "safety_refusal":
        return RESULT_SAFETY_REFUSAL
    if status == "budget_exceeded":
        return RESULT_BUDGET_EXCEEDED
    if status == "timeout":
        return RESULT_TIMEOUT
    if status == "error":
        return RESULT_TECHNICAL_ERROR
    return RESULT_MODE_ROUTED if routed else RESULT_ANSWERED


def _align_evidence_manifest(out: MutableMapping[str, Any]) -> None:
    citations = [c for c in (out.get("citations") or []) if isinstance(c, Mapping)]
    links = [l for l in (out.get("rg_links") or []) if isinstance(l, Mapping)]

    dedup_citations: list[dict] = []
    citation_ids: set[str] = set()
    for item in citations:
        cid = _clean_text(item.get("citation_id"), 260)
        if not cid or cid in citation_ids:
            continue
        citation_ids.add(cid)
        dedup_citations.append(dict(item))

    dedup_links: list[dict] = []
    link_keys: set[tuple[str, str]] = set()
    for item in links:
        cid = _clean_text(item.get("citation_id"), 260)
        url = _clean_text(item.get("url"), 2000)
        if cid and citation_ids and cid not in citation_ids:
            continue
        key = (cid, url)
        if key in link_keys:
            continue
        link_keys.add(key)
        dedup_links.append(dict(item))

    out["citations"] = dedup_citations
    out["rg_links"] = dedup_links


def decorate_response(
    response: Mapping[str, Any],
    request: AssistantCoreRequest,
    decision: AssistantCoreDecision,
) -> dict:
    out: MutableMapping[str, Any] = dict(response or {})
    routed = decision.effective_mode != request.requested_mode
    out["requested_mode"] = request.requested_mode
    out["effective_mode"] = decision.effective_mode
    out["routed"] = routed
    out["request_kind"] = decision.request_kind
    out["information_task"] = decision.information_task
    out["required_answer_types"] = list(decision.required_answer_types)
    out["evidence_state"] = decision.evidence_state
    out["evidence_policy"] = decision.evidence_policy
    out["result_code"] = _status_to_result_code(out, routed)

    grounding = _clean_text(out.get("grounding"), 80)
    if not grounding:
        grounding = (
            "general_technical_knowledge"
            if out["result_code"] == RESULT_GENERAL_GUIDANCE
            else "none"
            if out["result_code"] in {
                RESULT_OUT_OF_SCOPE,
                RESULT_SAFETY_REFUSAL,
                RESULT_NEEDS_CLARIFICATION,
                RESULT_TECHNICAL_ERROR,
                RESULT_TIMEOUT,
                RESULT_BUDGET_EXCEEDED,
            }
            else "indexed_machine_sources"
        )
    out["grounding"] = grounding

    _align_evidence_manifest(out)

    # Existing Worker/Bubble compatibility. Root Cause formats ``problem_summary``
    # and ignores an ASK-only answer. Therefore a semantically routed ASK answer is
    # copied there without inventing pseudo-causes.
    if decision.effective_mode == MODE_ASK:
        answer = str(out.get("answer") or "").strip()
        out.setdefault("possible_causes", [])
        out.setdefault("recommended_next_checks", [])
        if answer:
            out.setdefault("problem_summary", answer)
    elif decision.effective_mode == MODE_ROOT_CAUSE:
        out.setdefault("answer", "")

    meta = dict(out.get("meta") or {})
    meta["assistant_core_v2"] = {
        "requested_mode": request.requested_mode,
        "effective_mode": decision.effective_mode,
        "routed": routed,
        "request_kind": decision.request_kind,
        "information_task": decision.information_task,
        "required_answer_types": list(decision.required_answer_types),
        "confidence": round(decision.confidence, 4),
        "requested_mode_fit": bool(decision.requested_mode_fit),
        "evidence_state": decision.evidence_state,
        "evidence_policy": decision.evidence_policy,
        "preferred_source_types": list(decision.preferred_source_types),
        "source_type_policy": decision.source_type_policy,
        "required_facets": list(decision.required_facets),
        "facet_queries": [
            {
                "facet": item.facet,
                "answer_type": item.answer_type,
                "must_cover": item.must_cover,
                "dense_queries": list(item.dense_queries),
                "lexical_queries": list(item.lexical_queries),
                "exact_terms": list(item.exact_terms),
                "preferred_source_types": list(item.preferred_source_types),
            }
            for item in decision.facet_queries
        ],
        "diagnostic_subsystems": list(decision.diagnostic_subsystems),
        "diagnostic_observables": list(decision.diagnostic_observables),
        "diagnostic_operating_conditions": list(decision.diagnostic_operating_conditions),
        "diagnostic_discriminants": list(decision.diagnostic_discriminants),
        "diagnostic_exclusions": list(decision.diagnostic_exclusions),
        "missing_information": list(decision.missing_information),
        "router_model": decision.router_model,
        "router_degraded": bool(decision.degraded),
        "router_degraded_reason": decision.degraded_reason,
    }
    out["meta"] = meta
    return dict(out)



def _response_quality_tuple(response: Mapping[str, Any]) -> tuple:
    """Monotonic quality ordering used only between one baseline and one repair.

    A technical timeout/no_sources can never replace an already grounded answered
    response. Contract completion, evidence and missing-facet count dominate length.
    """
    if not isinstance(response, Mapping):
        return (0, 0, 0, -999, 0)
    status = _clean_text(response.get("status"), 40).lower()
    answered = status == "answered"
    citations = [c for c in (response.get("citations") or []) if isinstance(c, Mapping)]
    meta = response.get("meta") if isinstance(response.get("meta"), Mapping) else {}
    validation = meta.get("assistant_core_validation") if isinstance(meta.get("assistant_core_validation"), Mapping) else {}
    contract = validation.get("answer_contract") if isinstance(validation.get("answer_contract"), Mapping) else {}
    passed = bool(contract.get("passed"))
    missing = len(contract.get("missing_answer_facets") or []) + len(contract.get("missing_evidence_facets") or []) + len(contract.get("missing_list_items") or [])
    if _clean_text(response.get("effective_mode"), 40).lower() == MODE_ROOT_CAUSE:
        body = _clean_text(response.get("problem_summary"), 2000) + " " + " ".join(_clean_text(c.get("cause"), 300) for c in (response.get("possible_causes") or []) if isinstance(c, Mapping))
    else:
        body = _clean_text(response.get("answer"), 4000)
    return (1 if answered else 0, 1 if passed else 0, 1 if citations else 0, -missing, min(len(body), 4000))


def _choose_monotonic_response(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    base = dict(baseline or {})
    cand = dict(candidate or {})
    if not base:
        return cand
    if not cand:
        return base
    bq = _response_quality_tuple(base)
    cq = _response_quality_tuple(cand)
    # Never permit a timeout/error/no_sources repair to erase an answered baseline.
    if bq[0] and not cq[0]:
        out = base
        meta = dict(out.get("meta") or {})
        meta["monotonic_repair"] = {"selected": "baseline", "baseline_quality": list(bq), "candidate_quality": list(cq)}
        out["meta"] = meta
        return out
    out = cand if cq > bq else base
    meta = dict(out.get("meta") or {})
    meta["monotonic_repair"] = {"selected": "candidate" if out is cand else "baseline", "baseline_quality": list(bq), "candidate_quality": list(cq)}
    out["meta"] = meta
    return out

class AssistantCoreV2:
    def __init__(self, hooks: AssistantCoreHooks):
        self.hooks = hooks

    def run(self, request: AssistantCoreRequest) -> dict:
        if request.requested_mode not in VALID_MODES:
            raise ValueError(f"Unsupported requested mode: {request.requested_mode}")
        if not request.query.strip():
            raise ValueError("query is required")

        retrieval = self.hooks.retrieve_neutral(request)

        router_error = ""
        raw_decision: Optional[Mapping[str, Any]] = None
        try:
            raw_decision = self.hooks.route_semantically(request, retrieval)
        except Exception as exc:  # bounded degraded mode, never false no_sources
            router_error = str(exc)

        decision = normalize_decision(request, raw_decision, router_error=router_error)

        if decision.request_kind == KIND_UNSAFE_REQUEST:
            return decorate_response(
                self.hooks.build_safety_refusal(request, decision), request, decision
            )

        if decision.request_kind == KIND_OUT_OF_SCOPE:
            return decorate_response(
                self.hooks.build_out_of_scope(request, decision), request, decision
            )

        if decision.evidence_state == EVIDENCE_CLARIFY:
            return decorate_response(
                self.hooks.build_clarification(request, decision), request, decision
            )

        # A router timeout/technical degradation must never be converted into an
        # answer merely because a nearby retrieval candidate exists. Fail closed;
        # the user may retry and the exact-response cache will not store this result.
        if decision.degraded and decision.evidence_state == EVIDENCE_UNSUPPORTED:
            return decorate_response(
                self.hooks.build_no_evidence(request, decision, retrieval),
                request,
                decision,
            )

        has_refinement_queries = bool(
            decision.dense_queries
            or decision.lexical_queries
            or decision.exact_terms
            or decision.required_facets
            or decision.facet_queries
        )
        precision_task_requires_refinement = (
            (
                decision.information_task in PRECISION_INFORMATION_TASKS
                or bool(set(decision.required_answer_types) & PRECISION_ANSWER_REQUIREMENTS)
            )
            and has_refinement_queries
        )
        if (
            (
                decision.evidence_state in {EVIDENCE_REFINE, EVIDENCE_PARTIAL}
                or precision_task_requires_refinement
            )
            and has_refinement_queries
            and self.hooks.refine_retrieval is not None
        ):
            retrieval = self.hooks.refine_retrieval(request, retrieval, decision)

        prepared = dict(self.hooks.prepare_evidence(request, retrieval, decision) or {})
        prepared_retrieval = dict(prepared.get("retrieval") or retrieval or {})
        evidence_supported = bool(prepared.get("supported"))

        if not evidence_supported:
            if (
                decision.request_kind == "general_technical"
                and decision.effective_mode == MODE_ASK
                and decision.evidence_policy in {
                    POLICY_GENERAL_ALLOWED,
                    POLICY_MACHINE_PREFERRED,
                }
            ):
                general = dict(self.hooks.synthesize_general(request, decision) or {})
                general.setdefault("ok", True)
                general.setdefault("status", "answered")
                general["result_code"] = RESULT_GENERAL_GUIDANCE
                general["grounding"] = "general_technical_knowledge"
                return decorate_response(general, request, decision)

            return decorate_response(
                self.hooks.build_no_evidence(request, decision, prepared_retrieval),
                request,
                decision,
            )

        if decision.effective_mode == MODE_ROOT_CAUSE:
            response = self.hooks.synthesize_root_cause(
                request, prepared_retrieval, decision
            )
        elif decision.effective_mode == MODE_SMART_DIAGNOSTIC:
            if self.hooks.synthesize_smart_start is None:
                raise RuntimeError("Smart Diagnostic synthesis hook is not configured")
            response = self.hooks.synthesize_smart_start(
                request, prepared_retrieval, decision
            )
        else:
            response = self.hooks.synthesize_ask(
                request, prepared_retrieval, decision
            )

        if self.hooks.validate_response is not None:
            response = self.hooks.validate_response(
                dict(response or {}), request, prepared_retrieval, decision
            )
        validated_baseline = dict(response or {})

        # One bounded repair cycle is available only when the first grounded answer
        # misses a mandatory facet/answer type. It reuses the semantic contract,
        # retrieves only the missing facets and rewrites once. A second failure is
        # converted to no_sources by the validator; the loop can never repeat.
        if (
            isinstance(response, Mapping)
            and bool(response.get("_assistant_core_repair_needed"))
            and not bool(response.get("_assistant_core_repair_attempted"))
            and self.hooks.repair_response is not None
            and decision.effective_mode == MODE_ASK
        ):
            repaired = dict(
                self.hooks.repair_response(
                    dict(response), request, prepared_retrieval, decision
                )
                or {}
            )
            repair_retrieval = repaired.pop("_assistant_core_repair_retrieval", None)
            if isinstance(repair_retrieval, Mapping):
                prepared_retrieval = dict(repair_retrieval)
            repaired["_assistant_core_repair_attempted"] = True
            response = repaired
            if self.hooks.validate_response is not None:
                response = self.hooks.validate_response(
                    dict(response or {}), request, prepared_retrieval, decision
                )
            response = _choose_monotonic_response(validated_baseline, dict(response or {}))

        return decorate_response(response, request, decision)
