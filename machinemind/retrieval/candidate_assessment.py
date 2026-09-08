"""P4-C7: extracted candidate assessment implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class AskEvidenceScoreTextRuntime:
    _ask_evidence_code_tokens: Callable[..., Any]
    _ask_evidence_number_tokens: Callable[..., Any]
    _ask_evidence_stopwords: Callable[..., Any]
    _ask_evidence_tokenize: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_evidence_score_text(q: str, text: str, profile: dict, *, runtime: AskEvidenceScoreTextRuntime) -> float:
    _ask_evidence_code_tokens = runtime._ask_evidence_code_tokens
    _ask_evidence_number_tokens = runtime._ask_evidence_number_tokens
    _ask_evidence_stopwords = runtime._ask_evidence_stopwords
    _ask_evidence_tokenize = runtime._ask_evidence_tokenize
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    if not text:
        return 0.0
    tn = _normalize_unicode_advanced(text).lower()
    qn = _normalize_unicode_advanced(q or "").lower()

    q_tokens = [t for t in _ask_evidence_tokenize(qn) if len(t) >= 3]
    terms = []
    for key in ["search_phrases", "search_terms_it", "search_terms_en", "required_information", "important_codes_or_numbers"]:
        terms.extend([str(x or "").strip() for x in (profile.get(key) or [])])
    terms.extend(q_tokens)
    terms.extend(_ask_evidence_code_tokens(q))
    terms.extend(_ask_evidence_number_tokens(q))
    terms = _dedup_text_values(terms, limit=90)

    score = 0.0
    hit_terms = 0
    for term in terms:
        norm = _normalize_unicode_advanced(term).lower().strip()
        if not norm or norm in _ask_evidence_stopwords():
            continue
        if norm in tn:
            hit_terms += 1
            # Phrases, codes and numeric/unit values matter more than isolated generic words.
            if len(norm) >= 12 or re.search(r"\d", norm):
                score += 5.0
            elif len(norm) >= 6:
                score += 2.2
            else:
                score += 1.0

    # Token-level recall from the original question.
    q_unique = _dedup_text_values(q_tokens, limit=40)
    if q_unique:
        matched = sum(1 for t in q_unique if t in tn)
        score += 10.0 * (matched / max(1, len(q_unique)))
        if matched >= 2:
            score += 2.0

    # Exact code/number tokens from the question are strong anchors.
    for x in _ask_evidence_code_tokens(q) + _ask_evidence_number_tokens(q):
        xn = _normalize_unicode_advanced(x).lower()
        if xn and xn in tn:
            score += 8.0

    # Prefer pages/records that are information dense and contain multiple query anchors.
    if hit_terms >= 4:
        score += min(8.0, hit_terms * 0.8)

    # Penalize very generic safety/intro pages unless the question itself asks about them.
    generic_markers = ["informazioni generali", "general information", "proprietà delle informazioni", "all rights reserved"]
    if any(x in tn for x in generic_markers) and hit_terms < 3:
        score -= 3.0

    return max(0.0, score)


@dataclass(frozen=True)
class AskStructuredManualSupportSafetyTermsRuntime:
    pass


def ask_structured_manual_support_safety_terms( *, runtime: AskStructuredManualSupportSafetyTermsRuntime) -> list[str]:
    return [
        "sicurezza", "safety", "messa a punto", "operatore qualificato", "qualified operator",
        "dpi", "ppe", "guanti", "gloves", "occhiali", "goggles", "protezione", "protection",
        "alimentazione elettrica", "electrical", "alimentazione pneumatica", "pneumatic",
        "sezionatore", "interruttore generale", "lucchetto", "lockout", "disconnect",
    ]


@dataclass(frozen=True)
class AskStructuredManualSupportScoreDetailsRuntime:
    _ask_structured_manual_support_safety_terms: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_manual_support_score_details(text: str, terms: list[str], *, runtime: AskStructuredManualSupportScoreDetailsRuntime) -> dict:
    _ask_structured_manual_support_safety_terms = runtime._ask_structured_manual_support_safety_terms
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    low = _normalize_unicode_advanced(text or "").lower()
    if not low:
        return {"operation_score": 0.0, "safety_score": 0.0, "total_score": 0.0}

    operation_score = 0.0
    matched_terms = 0
    for t in terms:
        if t and t in low:
            matched_terms += 1
            operation_score += 1.0

    # Phrase synergy: if an operation noun and operation verb both appear, prefer
    # that page over generic safety pages.
    change_words = ["change", "cambio", "cambiare", "sostituzione", "sostituire", "replacement", "replace"]
    coil_words = ["coil", "bobina", "bobine"]
    if any(w in low for w in change_words) and any(w in low for w in coil_words):
        operation_score += 5.0
    if any(w in low for w in ["procedura", "procedure", "sequenza", "sequence", "operazione", "operation"]):
        operation_score += 1.2

    safety_score = 0.0
    for marker in _ask_structured_manual_support_safety_terms():
        if marker in low:
            safety_score += 1.0

    # Operational relevance dominates. Safety is still useful, but it cannot be
    # the only reason a manual page is selected when the user asked how to perform
    # an operation.
    total_score = (operation_score * 3.0) + (safety_score * 0.35)
    return {
        "operation_score": float(operation_score),
        "safety_score": float(safety_score),
        "total_score": float(total_score),
        "matched_operation_terms": int(matched_terms),
    }


@dataclass(frozen=True)
class AskStructuredManualSupportScoreRuntime:
    _ask_structured_manual_support_score_details: Callable[..., Any]


def ask_structured_manual_support_score(text: str, terms: list[str], *, runtime: AskStructuredManualSupportScoreRuntime) -> float:
    _ask_structured_manual_support_score_details = runtime._ask_structured_manual_support_score_details
    return float(_ask_structured_manual_support_score_details(text, terms).get("total_score") or 0.0)


@dataclass(frozen=True)
class AskStructuredManualSupportCandidateScoreRuntime:
    _ask_structured_manual_support_score_details: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_manual_support_candidate_score(text: str, profile_terms: list[str], fallback_terms: list[str], *, runtime: AskStructuredManualSupportCandidateScoreRuntime) -> float:
    _ask_structured_manual_support_score_details = runtime._ask_structured_manual_support_score_details
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    low = _normalize_unicode_advanced(text or "").lower()
    if not low:
        return 0.0
    score = 0.0
    for t in profile_terms or []:
        tt = _normalize_unicode_advanced(str(t or "")).lower().strip()
        if len(tt) < 3:
            continue
        if tt in low:
            # Multi-word concepts are stronger because they usually reflect a
            # reasoned manual phrase rather than a generic single term.
            score += 2.5 if " " in tt else 1.2
    # Fallback terms are useful but must not dominate the LLM-inferred manual profile.
    details = _ask_structured_manual_support_score_details(text, fallback_terms or [])
    score += float(details.get("total_score") or 0.0) * 0.18
    return float(score)


@dataclass(frozen=True)
class V12FamilyScoreRuntime:
    _assistant_core_required_facet_metrics: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v12_step_direct_query_score: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def v12_family_score(
    *,
    q: str,
    planner: Optional[dict],
    procedure: dict,
    seed_steps: list[dict],
    complete_steps: list[dict],
    raw_procedure_present: bool,
    runtime: V12FamilyScoreRuntime,
) -> dict:
    _assistant_core_required_facet_metrics = runtime._assistant_core_required_facet_metrics
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _term_overlap_score = runtime._term_overlap_score
    _v12_step_direct_query_score = runtime._v12_step_direct_query_score
    _v13_candidate_text = runtime._v13_candidate_text
    facets = _dedup_text_values((planner or {}).get("required_facets") or [], limit=12)
    complete_text = "\n".join(
        _v13_candidate_text(candidate)
        for candidate in [procedure] + list(complete_steps or [])
        if isinstance(candidate, dict)
    )
    seed_text = "\n".join(
        _v13_candidate_text(candidate)
        for candidate in list(seed_steps or [])
        if isinstance(candidate, dict)
    )
    complete_metrics = _assistant_core_required_facet_metrics(complete_text, facets)
    seed_metrics = _assistant_core_required_facet_metrics(seed_text, facets)
    direct_scores = [
        _v12_step_direct_query_score(step, q)
        for step in (complete_steps or [])
        if isinstance(step, dict)
    ]
    procedure_terms = _content_term_set(_v13_candidate_text(procedure), limit=160)
    query_terms = _content_term_set(q, limit=80)
    procedure_overlap = (
        _term_overlap_score(query_terms, procedure_terms)
        if query_terms and procedure_terms else 0.0
    )
    seed_quality = max(
        [
            float(step.get("v13_score", step.get("retrieval_score", step.get("similarity", 0.0))) or 0.0)
            for step in (seed_steps or [])
        ]
        or [0.0]
    )
    relation_seed_count = sum(
        1
        for step in (seed_steps or [])
        if str(step.get("_v10_5_parent_source_key") or "").strip()
    )
    exact_machine = any(bool(step.get("exact_machine_scope")) for step in (seed_steps or []))
    complete_coverage = float(complete_metrics.get("coverage") or 0.0)
    seed_coverage = float(seed_metrics.get("coverage") or 0.0)
    max_direct = max(direct_scores or [0.0])
    seed_count = len({str(step.get("bubble_document_id") or "") for step in (seed_steps or []) if str(step.get("bubble_document_id") or "")})
    score = (
        4.0 * complete_coverage
        + 2.2 * seed_coverage
        + 0.38 * min(seed_count, 6)
        + 1.30 * max_direct
        + 0.70 * procedure_overlap
        + 0.28 * min(relation_seed_count, 4)
        + 0.18 * min(1.0, seed_quality)
        + (0.12 if exact_machine else 0.0)
        + (0.08 if raw_procedure_present else 0.0)
    )
    return {
        "score": float(score),
        "facet_coverage": complete_coverage,
        "seed_facet_coverage": seed_coverage,
        "covered_facets": list(complete_metrics.get("covered") or []),
        "missing_facets": list(complete_metrics.get("missing") or []),
        "seed_count": seed_count,
        "relation_seed_count": relation_seed_count,
        "max_direct_step_score": float(max_direct),
        "procedure_overlap": float(procedure_overlap),
        "seed_quality": float(seed_quality),
        "raw_procedure_present": bool(raw_procedure_present),
    }


@dataclass(frozen=True)
class V12StepDirectQueryScoreRuntime:
    _content_term_set: Callable[..., Any]
    _procedure_ui_fields: Callable[..., Any]
    _procedure_ui_sections: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]


def v12_step_direct_query_score(step: dict, q: str, *, runtime: V12StepDirectQueryScoreRuntime) -> float:
    _content_term_set = runtime._content_term_set
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_sections = runtime._procedure_ui_sections
    _term_overlap_score = runtime._term_overlap_score
    q_terms = _content_term_set(q, limit=80)
    if not q_terms:
        return 0.0
    fields = _procedure_ui_fields(step)
    sections = _procedure_ui_sections(fields.get("description") or "")
    # Score the Step's own title and operational action. Parent Procedure labels
    # and repeated technical-reference boilerplate are present in every Step and
    # would otherwise make a component name appear relevant to the whole family.
    text = " ".join(
        [
            str(fields.get("title") or ""),
            str(sections.get("instruction") or sections.get("body") or ""),
        ]
    )
    return _term_overlap_score(q_terms, _content_term_set(text, limit=180))


@dataclass(frozen=True)
class V12StepPhraseScoreRuntime:
    _content_term_set: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _procedure_ui_fields: Callable[..., Any]
    _procedure_ui_sections: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]


def v12_step_phrase_score(step: dict, phrase: str, *, runtime: V12StepPhraseScoreRuntime) -> float:
    _content_term_set = runtime._content_term_set
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_sections = runtime._procedure_ui_sections
    _term_overlap_score = runtime._term_overlap_score
    phrase_terms = _content_term_set(phrase, limit=40)
    if not phrase_terms:
        return 0.0
    fields = _procedure_ui_fields(step)
    sections = _procedure_ui_sections(fields.get("description") or "")
    step_text = _normalize_unicode_advanced(
        " ".join(
            [
                str(fields.get("title") or ""),
                str(sections.get("instruction") or sections.get("body") or ""),
            ]
        )
    ).lower()
    step_terms = _content_term_set(step_text, limit=180)
    overlap = _term_overlap_score(phrase_terms, step_terms)
    exact_hits = sum(1 for term in phrase_terms if term in step_text)
    return float(overlap + 0.08 * exact_hits)


@dataclass(frozen=True)
class V12StepFacetScoreRuntime:
    _assistant_core_required_facet_metrics: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v12_step_contract_text: Callable[..., Any]
    _v12_step_direct_query_score: Callable[..., Any]
    re: Any


def v12_step_facet_score(step: dict, facet_query: dict, q: str, *, runtime: V12StepFacetScoreRuntime) -> dict:
    _assistant_core_required_facet_metrics = runtime._assistant_core_required_facet_metrics
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _term_overlap_score = runtime._term_overlap_score
    _v12_step_contract_text = runtime._v12_step_contract_text
    _v12_step_direct_query_score = runtime._v12_step_direct_query_score
    re = runtime.re
    text = _normalize_unicode_advanced(_v12_step_contract_text(step)).lower()
    text_terms = _content_term_set(text, limit=220)
    values = _dedup_text_values(
        [facet_query.get("facet")]
        + list(facet_query.get("dense_queries") or [])
        + list(facet_query.get("lexical_queries") or []),
        limit=20,
    )
    phrase_score = 0.0
    for value in values:
        terms = _content_term_set(value, limit=45)
        if not terms or not text_terms:
            continue
        overlap = _term_overlap_score(terms, text_terms)
        exact_hits = sum(1 for term in terms if term in text)
        phrase_score = max(phrase_score, float(overlap + 0.06 * exact_hits))

    exact_terms = _dedup_text_values(facet_query.get("exact_terms") or [], limit=12)
    matched_exact: list[str] = []
    for term in exact_terms:
        normalized = re.sub(
            r"\s+", " ", _normalize_unicode_advanced(str(term or "")).lower()
        ).strip()
        if normalized and normalized in text:
            matched_exact.append(term)
            continue
        term_set = _content_term_set(normalized, limit=20)
        if term_set and len(term_set & text_terms) / max(1, len(term_set)) >= 0.67:
            matched_exact.append(term)

    facet = str(facet_query.get("facet") or "").strip()
    facet_coverage = float(
        _assistant_core_required_facet_metrics(text, [facet]).get("coverage") or 0.0
    ) if facet else 0.0
    direct = _v12_step_direct_query_score(step, q)
    score = (
        phrase_score
        + 0.15 * min(len(matched_exact), 4)
        + 0.22 * facet_coverage
        + 0.14 * direct
    )
    return {
        "score": float(score),
        "phrase_score": float(phrase_score),
        "facet_coverage": facet_coverage,
        "matched_exact_terms": matched_exact,
        "direct_query_score": float(direct),
    }


@dataclass(frozen=True)
class AssistantCoreFacetCandidateConfidenceRuntime:
    ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD: Any
    REQ_CHECKLIST: Any
    REQ_EXPLANATION: Any
    REQ_INTERFACE_LOCATIONS: Any
    REQ_NUMERIC_VALUE: Any
    REQ_ORDERED_ACTIONS: Any
    REQ_SAFETY_CONDITIONS: Any
    REQ_STATE_SEQUENCE: Any
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_interface_navigation_signal: Callable[..., Any]
    _assistant_core_numeric_signal: Callable[..., Any]
    _assistant_core_required_facet_metrics: Callable[..., Any]
    _assistant_core_sequence_signal: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]
    re: Any


def assistant_core_facet_candidate_confidence(
    *,
    candidate: dict,
    facet: str,
    answer_type: str,
    dense_queries: list[str] | tuple[str, ...],
    lexical_queries: list[str] | tuple[str, ...],
    exact_terms: list[str] | tuple[str, ...],
    preferred_source_types: list[str] | tuple[str, ...],
    rank: int,
    runtime: AssistantCoreFacetCandidateConfidenceRuntime,
) -> dict:
    """Independent support signal for one facet-specific retrieval result.

    A result is not marked as covering a facet merely because it ranked high in a
    search. It needs semantic, lexical, exact-title or answer-shape support. This
    prevents an HMI page mentioning a component from satisfying a capacity/checklist
    facet, while preserving cross-language evidence through cosine similarity.
    """
    ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD = runtime.ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD
    REQ_CHECKLIST = runtime.REQ_CHECKLIST
    REQ_EXPLANATION = runtime.REQ_EXPLANATION
    REQ_INTERFACE_LOCATIONS = runtime.REQ_INTERFACE_LOCATIONS
    REQ_NUMERIC_VALUE = runtime.REQ_NUMERIC_VALUE
    REQ_ORDERED_ACTIONS = runtime.REQ_ORDERED_ACTIONS
    REQ_SAFETY_CONDITIONS = runtime.REQ_SAFETY_CONDITIONS
    REQ_STATE_SEQUENCE = runtime.REQ_STATE_SEQUENCE
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_interface_navigation_signal = runtime._assistant_core_interface_navigation_signal
    _assistant_core_numeric_signal = runtime._assistant_core_numeric_signal
    _assistant_core_required_facet_metrics = runtime._assistant_core_required_facet_metrics
    _assistant_core_sequence_signal = runtime._assistant_core_sequence_signal
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _term_overlap_score = runtime._term_overlap_score
    _v13_candidate_text = runtime._v13_candidate_text
    re = runtime.re
    text = _v13_candidate_text(candidate)
    normalized_text = re.sub(
        r"\s+", " ", _normalize_unicode_advanced(text).lower()
    ).strip()
    query_text = " ".join(
        _dedup_text_values(
            [facet] + list(dense_queries or []) + list(lexical_queries or []),
            limit=18,
        )
    )
    query_terms = _content_term_set(query_text, limit=140)
    text_terms = _content_term_set(text, limit=220)
    lexical_overlap = _term_overlap_score(query_terms, text_terms) if query_terms else 0.0
    semantic = max(
        0.0,
        float(
            candidate.get(
                "semantic_similarity",
                candidate.get("gate_similarity", candidate.get("similarity", 0.0)),
            )
            or 0.0
        ),
    )
    facet_metrics = _assistant_core_required_facet_metrics(text, [facet])
    facet_coverage = float(facet_metrics.get("coverage") or 0.0)

    normalized_exact_hits: list[str] = []
    for raw_term in exact_terms or []:
        term = re.sub(
            r"\s+", " ", _normalize_unicode_advanced(str(raw_term or "")).lower()
        ).strip()
        if not term:
            continue
        if term in normalized_text:
            normalized_exact_hits.append(str(raw_term).strip())
    strong_exact = any(
        len(re.sub(r"\W+", "", _normalize_unicode_advanced(term))) >= 4
        or bool(re.search(r"\d", term))
        for term in normalized_exact_hits
    )
    title_support = bool(
        candidate.get("structured_title_support")
        or (
            candidate.get("structured_title_match")
            and float(candidate.get("structured_title_match_score") or 0.0) >= 0.40
        )
    )
    fts_support = bool(candidate.get("fts_v13") and lexical_overlap >= 0.025)
    source_type = _assistant_core_candidate_source_type(candidate)
    preferred = {
        str(item or "").strip().lower()
        for item in preferred_source_types or []
        if str(item or "").strip()
    }
    source_preferred = bool(preferred and source_type in preferred)

    answer_type = str(answer_type or REQ_EXPLANATION).strip().lower()
    numeric_signal = _assistant_core_numeric_signal(text)
    if answer_type == REQ_NUMERIC_VALUE:
        answer_shape_score = (
            1.0 if numeric_signal.get("has_number_with_unit")
            else 0.45 if numeric_signal.get("has_number")
            else 0.0
        )
    elif answer_type == REQ_INTERFACE_LOCATIONS:
        answer_shape_score = 1.0 if _assistant_core_interface_navigation_signal(text) else 0.0
    elif answer_type == REQ_STATE_SEQUENCE:
        answer_shape_score = 1.0 if _assistant_core_sequence_signal(text) else 0.0
    elif answer_type in {REQ_ORDERED_ACTIONS, REQ_CHECKLIST, REQ_SAFETY_CONDITIONS}:
        list_signal = len(
            re.findall(r"(?m)^\s*(?:[-•*]|\d{1,2}[.)])\s+", str(text or ""))
        )
        structured_signal = source_type in {"procedure", "step", "ps"}
        answer_shape_score = 1.0 if list_signal >= 2 else 0.75 if structured_signal else 0.0
    else:
        answer_shape_score = 0.55

    rank_score = max(0.0, 1.0 - min(max(1, int(rank)) - 1, 20) / 20.0)
    score = (
        0.40 * min(1.0, semantic)
        + 0.24 * min(1.0, lexical_overlap * 5.0)
        + 0.16 * min(1.0, facet_coverage)
        + 0.12 * answer_shape_score
        + 0.04 * rank_score
        + (0.18 if strong_exact else 0.0)
        + (0.10 if title_support else 0.0)
        + (0.06 if fts_support else 0.0)
        + (0.04 if source_preferred else 0.0)
    )
    score = min(1.0, max(0.0, score))

    # Rank alone is never sufficient. Precision facets use stricter shape-aware
    # admission so a nearby page containing an unrelated number/menu/sequence does
    # not satisfy the contract. The router supplies bilingual lexical variants, so
    # a real cross-language match still has an independent contextual signal.
    if answer_type == REQ_NUMERIC_VALUE:
        credible = bool(
            strong_exact
            or title_support
            or facet_coverage >= 0.50
            or (
                lexical_overlap >= 0.10
                and answer_shape_score >= 0.45
                and (semantic >= 0.24 or fts_support)
            )
        )
    elif answer_type == REQ_INTERFACE_LOCATIONS:
        credible = bool(
            strong_exact
            or title_support
            or facet_coverage >= 0.50
            or (
                answer_shape_score >= 0.90
                and (lexical_overlap >= 0.055 or semantic >= 0.46 or fts_support)
            )
        )
    elif answer_type == REQ_STATE_SEQUENCE:
        credible = bool(
            strong_exact
            or title_support
            or facet_coverage >= 0.50
            or (
                answer_shape_score >= 0.90
                and (lexical_overlap >= 0.045 or semantic >= 0.44 or fts_support)
            )
        )
    elif answer_type in {REQ_ORDERED_ACTIONS, REQ_CHECKLIST, REQ_SAFETY_CONDITIONS}:
        credible = bool(
            strong_exact
            or title_support
            or facet_coverage >= 0.50
            or (
                answer_shape_score >= 0.70
                and (lexical_overlap >= 0.055 or semantic >= 0.45 or fts_support)
            )
        )
    else:
        credible = bool(
            strong_exact
            or title_support
            or facet_coverage >= 0.50
            or lexical_overlap >= 0.10
            or (semantic >= 0.42 and lexical_overlap >= 0.025)
            or (semantic >= 0.54 and answer_shape_score >= 0.50)
            or fts_support
        )
    threshold = float(ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD)
    credible = credible and score >= threshold
    return {
        "credible": bool(credible),
        "score": round(score, 6),
        "semantic": round(semantic, 6),
        "lexical_overlap": round(lexical_overlap, 6),
        "facet_coverage": round(facet_coverage, 6),
        "answer_shape_score": round(answer_shape_score, 6),
        "exact_hits": normalized_exact_hits[:8],
        "title_support": bool(title_support),
        "fts_support": bool(fts_support),
        "source_preferred": bool(source_preferred),
    }


@dataclass(frozen=True)
class AssistantCoreRequiredFacetMetricsRuntime:
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def assistant_core_required_facet_metrics(text: str, facets: list[str] | tuple[str, ...], *, runtime: AssistantCoreRequiredFacetMetricsRuntime) -> dict:
    """Deterministic coverage signal for the semantic contract produced by the router.

    The router is responsible for multilingual semantic interpretation. This helper
    only verifies that the admitted evidence/answer still contains the concepts the
    router declared mandatory; it never invents missing facets.
    """
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    normalized_text = re.sub(
        r"\s+", " ", _normalize_unicode_advanced(str(text or "")).lower()
    ).strip()
    clean_facets = _dedup_text_values(list(facets or []), limit=12)
    if not normalized_text or not clean_facets:
        return {"coverage": 0.0, "covered": [], "missing": clean_facets}

    text_terms = _content_term_set(normalized_text, limit=260)
    covered: list[str] = []
    missing: list[str] = []
    for facet in clean_facets:
        normalized_facet = re.sub(
            r"\s+", " ", _normalize_unicode_advanced(str(facet or "")).lower()
        ).strip()
        facet_terms = _content_term_set(normalized_facet, limit=30)
        phrase_hit = bool(normalized_facet and normalized_facet in normalized_text)
        if phrase_hit:
            covered.append(facet)
            continue
        if not facet_terms:
            missing.append(facet)
            continue
        overlap = len(facet_terms & text_terms) / max(1, len(facet_terms))
        if overlap >= 0.50 or (len(facet_terms) == 1 and overlap >= 1.0):
            covered.append(facet)
        else:
            missing.append(facet)
    return {
        "coverage": len(covered) / max(1, len(clean_facets)),
        "covered": covered,
        "missing": missing,
    }


@dataclass(frozen=True)
class AssistantCoreDiagnosticPriorityMetricsRuntime:
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_ps_is_substantive: Callable[..., Any]
    _assistant_core_required_facet_metrics: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def assistant_core_diagnostic_priority_metrics(
    candidate: dict,
    decision: AssistantCoreDecision,
    *, runtime: AssistantCoreDiagnosticPriorityMetricsRuntime,
) -> dict:
    """Score explicit user clues without hardcoding a machine or vocabulary."""
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_ps_is_substantive = runtime._assistant_core_ps_is_substantive
    _assistant_core_required_facet_metrics = runtime._assistant_core_required_facet_metrics
    _dedup_text_values = runtime._dedup_text_values
    _v13_candidate_text = runtime._v13_candidate_text
    text = _v13_candidate_text(candidate)
    tag_text = " ".join(
        str(x or "").strip()
        for x in (candidate.get("assistant_core_facet_hits") or [])
        if str(x or "").strip()
    )
    combined = f"{text}\n{tag_text}".strip()
    groups = [
        ("discriminants", tuple(decision.diagnostic_discriminants), 0.40),
        ("operating_conditions", tuple(decision.diagnostic_operating_conditions), 0.24),
        ("observables", tuple(decision.diagnostic_observables), 0.22),
        ("subsystems", tuple(decision.diagnostic_subsystems), 0.14),
    ]
    covered: list[str] = []
    missing: list[str] = []
    weighted = 0.0
    active_weight = 0.0
    group_scores: dict[str, float] = {}
    for name, values, weight in groups:
        if not values:
            continue
        metrics = _assistant_core_required_facet_metrics(combined, values)
        coverage = float(metrics.get("coverage") or 0.0)
        group_scores[name] = coverage
        weighted += weight * coverage
        active_weight += weight
        covered.extend(metrics.get("covered") or [])
        missing.extend(metrics.get("missing") or [])
    score = weighted / active_weight if active_weight > 0.0 else 0.0
    source_type = _assistant_core_candidate_source_type(candidate)
    exact_case_bonus = 0.0
    if source_type == "ps" and _assistant_core_ps_is_substantive(candidate):
        # A P&S is privileged only when it matches a high-value observation, not
        # simply because it belongs to the same machine.
        if group_scores.get("discriminants", 0.0) >= 0.34:
            exact_case_bonus = 0.20
        elif group_scores.get("observables", 0.0) >= 0.34 and group_scores.get("subsystems", 0.0) > 0.0:
            exact_case_bonus = 0.12
    return {
        "score": min(1.0, score + exact_case_bonus),
        "base_score": score,
        "exact_case_bonus": exact_case_bonus,
        "covered": _dedup_text_values(covered, limit=16),
        "missing": _dedup_text_values(missing, limit=16),
        "groups": group_scores,
    }


@dataclass(frozen=True)
class AssistantCoreRootCandidateViableRuntime:
    _assistant_core_candidate_facet_metrics: Callable[..., Any]
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_diagnostic_priority_metrics: Callable[..., Any]
    _assistant_core_ps_is_substantive: Callable[..., Any]
    _assistant_core_retrieval_query: Callable[..., Any]
    _root_cause_target_subsystems: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def assistant_core_root_candidate_viable(
    request: AssistantCoreRequest,
    decision: AssistantCoreDecision,
    candidate: dict,
    *, runtime: AssistantCoreRootCandidateViableRuntime,
) -> bool:
    _assistant_core_candidate_facet_metrics = runtime._assistant_core_candidate_facet_metrics
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_diagnostic_priority_metrics = runtime._assistant_core_diagnostic_priority_metrics
    _assistant_core_ps_is_substantive = runtime._assistant_core_ps_is_substantive
    _assistant_core_retrieval_query = runtime._assistant_core_retrieval_query
    _root_cause_target_subsystems = runtime._root_cause_target_subsystems
    _v13_candidate_text = runtime._v13_candidate_text
    if bool(candidate.get("hard_excluded")):
        return False
    if not _assistant_core_ps_is_substantive(candidate):
        return False

    text = _v13_candidate_text(candidate)
    core_facets = [
        item.facet for item in (decision.facet_queries or ())
        if bool(item.must_cover) and str(item.facet or "").strip()
    ] or list(decision.required_facets)
    facets = _assistant_core_candidate_facet_metrics(candidate, core_facets)
    facet_coverage = float(facets.get("coverage") or 0.0)
    router_selected = bool(candidate.get("assistant_core_router_id_bonus"))
    semantic = float(
        candidate.get("semantic_similarity", candidate.get("gate_similarity", candidate.get("similarity", 0.0)))
        or 0.0
    )
    causal = float(candidate.get("causal_strength_score") or 0.0)
    subsystem = float(candidate.get("subsystem_score") or 0.0)
    context_fit = float(candidate.get("context_fit_score") or 0.0)
    diag = dict(candidate.get("assistant_core_diagnostic_priority") or {})
    if not diag:
        diag = _assistant_core_diagnostic_priority_metrics(candidate, decision)
    diagnostic_priority = float(diag.get("score") or 0.0)
    source_type = _assistant_core_candidate_source_type(candidate)
    exact_documented_case = bool(
        source_type == "ps"
        and bool(diag.get("exact_case_bonus"))
        and (
            causal >= 0.02
            or context_fit > 0.0
            or subsystem > 0.0
            or semantic >= 0.30
            or facet_coverage >= 0.40
        )
    )
    if exact_documented_case:
        return True

    # A diagnostic source must preserve at least one real clue/facet from the
    # semantic contract.  Request-level similarity plus a generic word such as
    # "manual" is not enough to turn a nearby maintenance page into causal
    # evidence.  Strong cross-language semantic evidence remains admissible when
    # it is also corroborated by a causal/subsystem/context signal.
    diagnostic_contract_present = bool(decision.diagnostic_clues or core_facets)
    diagnostic_group_support = max(
        [float(value or 0.0) for value in (diag.get("groups") or {}).values()] or [0.0]
    )
    strongly_corroborated_crosslingual = bool(
        semantic >= 0.60
        and causal >= 0.12
        and (subsystem > 0.0 or context_fit > 0.0)
    )
    query_subsystems = set(_root_cause_target_subsystems(_assistant_core_retrieval_query(request), []))
    candidate_subsystems = {
        str(value or "").strip()
        for value in (candidate.get("matched_subsystems") or [])
        if str(value or "").strip()
    }
    subsystem_mismatch = bool(
        query_subsystems
        and candidate_subsystems
        and not (query_subsystems & candidate_subsystems)
    )
    # A candidate from a known but different subsystem is not causal evidence
    # merely because it is semantically nearby.  Apply the same mismatch guard to
    # every identified subsystem; strong cross-language corroboration remains the
    # only bounded exception.
    if subsystem_mismatch and not strongly_corroborated_crosslingual:
        return False
    if (
        diagnostic_contract_present
        and facet_coverage < 0.50
        and not strongly_corroborated_crosslingual
        and (
            (
                diagnostic_priority <= 0.0
                and diagnostic_group_support <= 0.0
            )
            or (
                subsystem_mismatch
                and diagnostic_priority < 0.35
            )
        )
    ):
        return False
    if (
        diagnostic_priority >= 0.28
        and (causal >= 0.02 or subsystem > 0.0 or context_fit > 0.0 or semantic >= 0.40)
    ):
        return True
    if (
        router_selected
        and causal >= 0.02
        and (
            facet_coverage >= 0.20
            or diagnostic_priority >= 0.20
            or subsystem > 0.0
            or context_fit > 0.0
        )
    ):
        return True
    if (facet_coverage >= 0.25 or diagnostic_priority >= 0.24) and (
        causal >= 0.02 or subsystem > 0.0 or context_fit > 0.0
    ):
        return True
    if semantic >= 0.46 and causal >= 0.08 and (subsystem > 0.0 or context_fit > 0.0):
        return True
    return False


@dataclass(frozen=True)
class AssistantCoreCandidateFacetMetricsRuntime:
    _assistant_core_required_facet_metrics: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def assistant_core_candidate_facet_metrics(
    candidate: dict,
    facets: tuple[str, ...] | list[str],
    *, runtime: AssistantCoreCandidateFacetMetricsRuntime,
) -> dict:
    _assistant_core_required_facet_metrics = runtime._assistant_core_required_facet_metrics
    _dedup_text_values = runtime._dedup_text_values
    _v13_candidate_text = runtime._v13_candidate_text
    clean_facets = _dedup_text_values(list(facets or []), limit=12)
    text_metrics = _assistant_core_required_facet_metrics(
        _v13_candidate_text(candidate), clean_facets
    )
    tagged = {
        str(x or "").strip().casefold(): str(x or "").strip()
        for x in (candidate.get("assistant_core_facet_hits") or [])
        if str(x or "").strip()
    }
    covered: list[str] = []
    missing: list[str] = []
    text_covered_keys = {
        str(x or "").strip().casefold()
        for x in (text_metrics.get("covered") or [])
        if str(x or "").strip()
    }
    for facet in clean_facets:
        key = facet.casefold()
        if key in tagged or key in text_covered_keys:
            covered.append(facet)
        else:
            missing.append(facet)
    return {
        "coverage": len(covered) / max(1, len(clean_facets)) if clean_facets else 0.0,
        "covered": covered,
        "missing": missing,
        "tagged": [tagged[k] for k in sorted(tagged)],
        "text_coverage": float(text_metrics.get("coverage") or 0.0),
    }


@dataclass(frozen=True)
class AssistantCoreFacetBalancedPoolRuntime:
    _assistant_core_candidate_stable_key: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]


def assistant_core_facet_balanced_pool(
    candidates: list[dict],
    facets: tuple[str, ...] | list[str],
    *,
    limit: int,
    runtime: AssistantCoreFacetBalancedPoolRuntime,
) -> list[dict]:
    """Keep one strong source per mandatory facet before filling by global rank."""
    _assistant_core_candidate_stable_key = runtime._assistant_core_candidate_stable_key
    _dedup_text_values = runtime._dedup_text_values
    facets = _dedup_text_values(list(facets or []), limit=12)
    if not candidates or not facets:
        return list(candidates or [])[:limit]

    selected: list[dict] = []
    seen: set[str] = set()
    for facet in facets:
        facet_key = facet.casefold()
        best = None
        for candidate in candidates:
            hits = {
                str(x or "").strip().casefold()
                for x in (candidate.get("assistant_core_facet_hits") or [])
                if str(x or "").strip()
            }
            covered = {
                str(x or "").strip().casefold()
                for x in (candidate.get("assistant_core_covered_facets") or [])
                if str(x or "").strip()
            }
            if facet_key not in hits and facet_key not in covered:
                continue
            best = candidate
            break
        if best is None:
            continue
        key = _assistant_core_candidate_stable_key(best)
        if key in seen:
            continue
        seen.add(key)
        selected.append(best)
        if len(selected) >= limit:
            return selected

    for candidate in candidates:
        key = _assistant_core_candidate_stable_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


@dataclass(frozen=True)
class AssistantCoreOverviewCandidateScoreRuntime:
    _assistant_core_candidate_evidence_text: Callable[..., Any]
    re: Any


def assistant_core_overview_candidate_score(candidate: dict, *, runtime: AssistantCoreOverviewCandidateScoreRuntime) -> float:
    """Stable ordering score for source records used by the overview builder."""
    _assistant_core_candidate_evidence_text = runtime._assistant_core_candidate_evidence_text
    re = runtime.re
    score = float(
        candidate.get(
            "v13_score",
            candidate.get("retrieval_score", candidate.get("similarity", 0.0)),
        )
        or 0.0
    )
    text = _assistant_core_candidate_evidence_text(candidate)
    if len(text) >= 240:
        score += 1.5
    if bool(candidate.get("assistant_core_section_expansion")):
        score += 1.5
    if str(candidate.get("retrieval_assurance_kind") or "").strip():
        score += 1.0
    # Table-of-contents fragments and title pages remain available as support but
    # rank below narrative technical pages when scores are otherwise similar.
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    if digit_ratio > 0.12:
        score -= 2.5
    if len(re.findall(r"(?m)^\s*\d{1,3}\s*$", text)) >= 4:
        score -= 2.0
    return score


@dataclass(frozen=True)
class SelectPromptCitationsFromMatrixRuntime:
    ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA: Any
    _root_cause_evidence_family_key: Callable[..., Any]


def select_prompt_citations_from_matrix(
    rescored_candidates: list[dict],
    diagnostic_matrix: dict,
    *,
    max_prompt: int,
    runtime: SelectPromptCitationsFromMatrixRuntime,
) -> list[dict]:
    ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA = runtime.ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA
    _root_cause_evidence_family_key = runtime._root_cause_evidence_family_key
    if not rescored_candidates:
        return []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in rescored_candidates
        if c.get("citation_id")
    }

    out: list[dict] = []
    used_ids = set()
    used_families = set()

    def try_add(cid: str, prefer_new_family: bool = True) -> bool:
        cid = str(cid or "").strip()
        if not cid or cid not in by_id or cid in used_ids:
            return False

        item = by_id[cid]
        fam = _root_cause_evidence_family_key(item)

        if prefer_new_family and fam in used_families:
            return False

        used_ids.add(cid)
        used_families.add(fam)
        out.append(item)
        return True

    for row in (diagnostic_matrix or {}).get("cause_hypotheses") or []:
        per_cause = 0
        for cid in row.get("evidence_ids") or []:
            added = try_add(cid, prefer_new_family=True)
            if not added:
                added = try_add(cid, prefer_new_family=False)
            if added:
                per_cause += 1
            if per_cause >= max(1, ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA):
                break
        if len(out) >= max_prompt:
            return out[:max_prompt]

    for cid in (diagnostic_matrix or {}).get("keep_ids") or []:
        if try_add(cid, prefer_new_family=True) or try_add(cid, prefer_new_family=False):
            if len(out) >= max_prompt:
                return out[:max_prompt]

    for item in rescored_candidates:
        cid = str(item.get("citation_id") or "").strip()
        if try_add(cid, prefer_new_family=True) or try_add(cid, prefer_new_family=False):
            if len(out) >= max_prompt:
                break

    return out[:max_prompt]


