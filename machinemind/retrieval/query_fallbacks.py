"""P4-C9: extracted query fallbacks implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class LlmClassifyRootCauseQueryIntentRuntime:
    DIAGNOSTIC_EVIDENCE_MODEL: Any
    OPENAI_CHAT_MODEL: Any
    ROOT_CAUSE_INTENT_MODEL: Any
    _openai_chat_json_models: Callable[..., Any]


def llm_classify_root_cause_query_intent(q: str, *, runtime: LlmClassifyRootCauseQueryIntentRuntime) -> dict:
    DIAGNOSTIC_EVIDENCE_MODEL = runtime.DIAGNOSTIC_EVIDENCE_MODEL
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    ROOT_CAUSE_INTENT_MODEL = runtime.ROOT_CAUSE_INTENT_MODEL
    _openai_chat_json_models = runtime._openai_chat_json_models
    schema = {
        "name": "root_cause_intent_classifier",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent_class": {
                    "type": "string",
                    "enum": [
                        "technical_fault_symptom",
                        "technical_information_question",
                        "non_technical_or_nonsense",
                        "ambiguous",
                    ],
                },
                "confidence": {
                    "type": "number",
                },
                "rationale": {
                    "type": "string",
                },
            },
            "required": ["intent_class", "confidence", "rationale"],
        },
    }

    system_msg = (
        "You classify a user query in an industrial machinery context. "
        "Work semantically, not by keyword matching. "
        "The query may be in Italian, English, or mixed language. "
        "The machinery type is unknown and can be any industrial machine. "
        "Classes:\n"
        "- technical_fault_symptom: the query expresses a fault symptom, anomaly, missing condition, malfunction, abnormal behavior, or a concise diagnostic complaint that could justify root cause analysis.\n"
        "- technical_information_question: the query is technical and relevant to machinery, but it is explanatory/informational rather than a fault symptom.\n"
        "- non_technical_or_nonsense: the query is outside the technical machinery domain, casual, or nonsense.\n"
        "- ambiguous: not enough certainty.\n"
        "Very short phrases can still be technical_fault_symptom if they express a real machine condition."
    )

    user_msg = f"QUERY:\n{q}"

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_INTENT_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=20,
        )
        if not isinstance(parsed, dict):
            return {
                "intent_class": "ambiguous",
                "confidence": 0.0,
                "rationale": "invalid classifier response",
            }

        parsed["intent_class"] = str(parsed.get("intent_class") or "ambiguous").strip()
        parsed["confidence"] = float(parsed.get("confidence") or 0.0)
        parsed["rationale"] = str(parsed.get("rationale") or "").strip()
        return parsed

    except Exception as e:
        return {
            "intent_class": "ambiguous",
            "confidence": 0.0,
            "rationale": f"classifier_error: {str(e)[:160]}",
        }


@dataclass(frozen=True)
class RootCausePreliminaryRetrievalSignalRuntime:
    ASK_SIM_THRESHOLD: Any
    ROOT_CAUSE_GATE_MIN_PRELIM_SIM: Any
    ROOT_CAUSE_GATE_PRELIM_TOP_K: Any
    _fetch_dense_chunk_candidates: Callable[..., Any]
    _vector_literal: Callable[..., Any]


def root_cause_preliminary_retrieval_signal(
    *,
    company_id: str,
    machine_id: str,
    q_vec: list[float],
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    runtime: RootCausePreliminaryRetrievalSignalRuntime,
) -> dict:
    ASK_SIM_THRESHOLD = runtime.ASK_SIM_THRESHOLD
    ROOT_CAUSE_GATE_MIN_PRELIM_SIM = runtime.ROOT_CAUSE_GATE_MIN_PRELIM_SIM
    ROOT_CAUSE_GATE_PRELIM_TOP_K = runtime.ROOT_CAUSE_GATE_PRELIM_TOP_K
    _fetch_dense_chunk_candidates = runtime._fetch_dense_chunk_candidates
    _vector_literal = runtime._vector_literal
    if not q_vec:
        return {
            "chunks_matching_filter": None,
            "rows_found": 0,
            "similarity_max": None,
            "hits_over_prelim_threshold": 0,
            "hits_over_ask_threshold": 0,
        }

    q_vec_lit = _vector_literal(q_vec)

    chunks_matching_filter, raw_rows = _fetch_dense_chunk_candidates(
        company_id=company_id,
        machine_id=machine_id,
        q_vec_lit=q_vec_lit,
        candidate_k=max(1, ROOT_CAUSE_GATE_PRELIM_TOP_K),
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    sims = [float(r[6]) for r in raw_rows] if raw_rows else []
    sim_max = max(sims) if sims else None

    return {
        "chunks_matching_filter": chunks_matching_filter,
        "rows_found": len(raw_rows),
        "similarity_max": sim_max,
        "hits_over_prelim_threshold": sum(1 for s in sims if s >= ROOT_CAUSE_GATE_MIN_PRELIM_SIM),
        "hits_over_ask_threshold": sum(1 for s in sims if s >= ASK_SIM_THRESHOLD),
    }


@dataclass(frozen=True)
class RootCauseQuerySignalSummaryRuntime:
    _extract_code_tokens: Callable[..., Any]
    _llm_classify_root_cause_query_intent: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _openai_embed_texts: Callable[..., Any]
    _root_cause_preliminary_retrieval_signal: Callable[..., Any]
    re: Any


def root_cause_query_signal_summary(
    q: str,
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: Optional[str] = None,
    doc_ids: Optional[list[str]] = None,
    debug: bool = False,
    runtime: RootCauseQuerySignalSummaryRuntime,
) -> dict:
    _extract_code_tokens = runtime._extract_code_tokens
    _llm_classify_root_cause_query_intent = runtime._llm_classify_root_cause_query_intent
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _openai_embed_texts = runtime._openai_embed_texts
    _root_cause_preliminary_retrieval_signal = runtime._root_cause_preliminary_retrieval_signal
    re = runtime.re
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    q_low = q_norm.lower()

    tokens = re.findall(r"[a-zà-öø-ÿ0-9]{2,}", q_low)
    code_hits = len(_extract_code_tokens(q_norm))

    classifier_used = True
    classified = _llm_classify_root_cause_query_intent(q_norm)
    intent_class = str(classified.get("intent_class") or "ambiguous").strip()
    intent_confidence = float(classified.get("confidence") or 0.0)
    intent_rationale = str(classified.get("rationale") or "").strip()

    q_vec = _openai_embed_texts([q_norm])[0] if q_norm else []

    preliminary = _root_cause_preliminary_retrieval_signal(
        company_id=company_id,
        machine_id=machine_id,
        q_vec=q_vec,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    return {
        "query_norm": q_norm,
        "token_count": len(tokens),
        "code_hits": code_hits,
        "query_vector": q_vec,
        "preliminary_retrieval": preliminary,
        "intent_class": intent_class,
        "intent_confidence": intent_confidence,
        "intent_rationale": intent_rationale,
        "classifier_used": classifier_used,
    }


@dataclass(frozen=True)
class ShouldFailClosedRootCauseQueryRuntime:
    ROOT_CAUSE_GATE_MIN_PRELIM_HITS: Any
    ROOT_CAUSE_GATE_MIN_PRELIM_SIM: Any


def should_fail_closed_root_cause_query(signal_summary: dict, *, runtime: ShouldFailClosedRootCauseQueryRuntime) -> bool:
    ROOT_CAUSE_GATE_MIN_PRELIM_HITS = runtime.ROOT_CAUSE_GATE_MIN_PRELIM_HITS
    ROOT_CAUSE_GATE_MIN_PRELIM_SIM = runtime.ROOT_CAUSE_GATE_MIN_PRELIM_SIM
    if not signal_summary:
        return True

    token_count = int(signal_summary.get("token_count", 0) or 0)
    intent_class = str(signal_summary.get("intent_class") or "ambiguous").strip()
    intent_confidence = float(signal_summary.get("intent_confidence", 0.0) or 0.0)

    prelim = signal_summary.get("preliminary_retrieval") or {}
    prelim_sim_max = prelim.get("similarity_max")
    prelim_hits = int(prelim.get("hits_over_prelim_threshold", 0) or 0)

    if token_count <= 0:
        return True

    if intent_class == "technical_fault_symptom":
        return False

    if intent_class == "non_technical_or_nonsense":
        return True

    strong_preliminary_signal = (
        prelim_sim_max is not None
        and float(prelim_sim_max) >= ROOT_CAUSE_GATE_MIN_PRELIM_SIM + 0.04
        and prelim_hits >= max(1, ROOT_CAUSE_GATE_MIN_PRELIM_HITS)
    )

    if intent_class == "technical_information_question":
        return not strong_preliminary_signal

    # ambiguous
    if intent_confidence < 0.80 and strong_preliminary_signal:
        return False

    return True


@dataclass(frozen=True)
class SimpleQueryLanguageRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def simple_query_language(q: str, *, runtime: SimpleQueryLanguageRuntime) -> str:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    toks = re.findall(r"[a-zà-öø-ÿ']{2,}", _normalize_unicode_advanced(q or "").lower())
    if not toks:
        return "it"

    it_markers = {
        "il", "lo", "la", "gli", "le", "di", "del", "della", "dei", "delle", "con",
        "per", "quando", "durante", "mentre", "dopo", "prima", "non", "si", "una", "un",
    }
    en_markers = {
        "the", "with", "for", "when", "during", "while", "after", "before", "not",
        "does", "is", "are", "can", "cannot", "won't", "will", "a", "an",
    }

    it_hits = sum(1 for t in toks if t in it_markers)
    en_hits = sum(1 for t in toks if t in en_markers)

    if en_hits > it_hits:
        return "en"
    return "it"


@dataclass(frozen=True)
class ShouldRouteAskThroughRootCauseRuntime:
    _is_lookup_or_identifier_query: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]


def should_route_ask_through_root_cause(q: str, *, runtime: ShouldRouteAskThroughRootCauseRuntime) -> bool:
    _is_lookup_or_identifier_query = runtime._is_lookup_or_identifier_query
    _query_symptom_profile = runtime._query_symptom_profile
    if _is_lookup_or_identifier_query(q):
        return False
    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])
    if "no_start" in classes:
        return True
    if classes & {"vibration", "noise", "jam"}:
        return True
    return False


@dataclass(frozen=True)
class IsLookupOrIdentifierQueryRuntime:
    EMAIL_HINTS: Any
    PHONE_HINTS: Any
    URL_HINTS: Any
    _extract_code_tokens: Callable[..., Any]
    _q_has_any: Callable[..., Any]


def is_lookup_or_identifier_query(q: str, *, runtime: IsLookupOrIdentifierQueryRuntime) -> bool:
    EMAIL_HINTS = runtime.EMAIL_HINTS
    PHONE_HINTS = runtime.PHONE_HINTS
    URL_HINTS = runtime.URL_HINTS
    _extract_code_tokens = runtime._extract_code_tokens
    _q_has_any = runtime._q_has_any
    if _q_has_any(q, URL_HINTS) or _q_has_any(q, EMAIL_HINTS) or _q_has_any(q, PHONE_HINTS):
        return True
    if _extract_code_tokens(q):
        return True
    return False


@dataclass(frozen=True)
class V13FallbackPlanRuntime:
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _extract_code_tokens: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _should_route_ask_through_root_cause: Callable[..., Any]
    _simple_query_language: Callable[..., Any]
    _v13_query_number_tokens: Callable[..., Any]
    re: Any


def v13_fallback_plan(q: str, *, runtime: V13FallbackPlanRuntime) -> dict:
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _extract_code_tokens = runtime._extract_code_tokens
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _should_route_ask_through_root_cause = runtime._should_route_ask_through_root_cause
    _simple_query_language = runtime._simple_query_language
    _v13_query_number_tokens = runtime._v13_query_number_tokens
    re = runtime.re
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    return {
        "intent": "diagnostic" if _should_route_ask_through_root_cause(q_norm) else "other",
        "normalized_query": q_norm,
        "query_language": _simple_query_language(q_norm),
        "dense_queries": [q_norm] if q_norm else [],
        "lexical_queries": [q_norm] if q_norm else [],
        "exact_terms": _dedup_text_values(_extract_code_tokens(q_norm) + _v13_query_number_tokens(q_norm), limit=16),
        "required_facets": _dedup_text_values(list(_content_term_set(q_norm, limit=12)), limit=10),
        "ambiguities": [],
    }


@dataclass(frozen=True)
class V13BuildProfileFromPlanRuntime:
    _ask_evidence_fallback_profile: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]


def v13_build_profile_from_plan(q: str, language: str, plan: Optional[dict], *, runtime: V13BuildProfileFromPlanRuntime) -> dict:
    _ask_evidence_fallback_profile = runtime._ask_evidence_fallback_profile
    _dedup_text_values = runtime._dedup_text_values
    profile = _ask_evidence_fallback_profile(q, language)
    plan = dict(plan or {})
    profile["search_phrases"] = _dedup_text_values(
        list(profile.get("search_phrases") or [])
        + list(plan.get("dense_queries") or [])
        + list(plan.get("lexical_queries") or []),
        limit=24,
    )
    profile["search_terms_it"] = _dedup_text_values(
        list(profile.get("search_terms_it") or []) + list(plan.get("exact_terms") or []) + list(plan.get("required_facets") or []),
        limit=32,
    )
    profile["search_terms_en"] = _dedup_text_values(
        list(profile.get("search_terms_en") or []) + list(plan.get("exact_terms") or []) + list(plan.get("required_facets") or []),
        limit=32,
    )
    profile["required_information"] = _dedup_text_values(
        list(profile.get("required_information") or []) + list(plan.get("required_facets") or []),
        limit=20,
    )
    profile["important_codes_or_numbers"] = _dedup_text_values(
        list(profile.get("important_codes_or_numbers") or []) + list(plan.get("exact_terms") or []),
        limit=24,
    )
    intent = str(plan.get("intent") or "").strip().lower()
    if intent in {"factual", "procedural", "listing", "comparison", "diagnostic"}:
        profile["answer_type"] = {
            "listing": "list",
            "diagnostic": "diagnostic",
        }.get(intent, intent)
    return profile


