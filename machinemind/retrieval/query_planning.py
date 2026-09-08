"""P4-C7: extracted query planning implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class QueryTranslationSchemaRuntime:
    pass


def query_translation_schema( *, runtime: QueryTranslationSchemaRuntime) -> dict:
    return {
        "name": "query_translation_for_retrieval_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    }


@dataclass(frozen=True)
class TranslateQueryForRetrievalRuntime:
    SEMANTIC_QUERY_PLANNER_MODEL: Any
    _normalize_unicode_advanced: Callable[..., Any]
    _openai_chat_json: Callable[..., Any]
    _query_translation_schema: Callable[..., Any]
    re: Any


def translate_query_for_retrieval(text: str, target_language: str, *, runtime: TranslateQueryForRetrievalRuntime) -> str:
    SEMANTIC_QUERY_PLANNER_MODEL = runtime.SEMANTIC_QUERY_PLANNER_MODEL
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _openai_chat_json = runtime._openai_chat_json
    _query_translation_schema = runtime._query_translation_schema
    re = runtime.re
    text = re.sub(r"\s+", " ", _normalize_unicode_advanced(text or "")).strip()
    target_language = str(target_language or "").strip().lower()
    if not text or target_language not in {"it", "en"}:
        return text

    system_msg = (
        "Translate the user's technical query for document retrieval. "
        "Preserve meaning exactly, keep it short, do not add explanations, "
        "do not assume a domain, and preserve codes, identifiers, and proper names exactly. "
        "Prefer natural industrial wording. For symptoms, use physically plausible verbs like "
        "'vibrates', 'makes noise', 'jams', 'stops', 'does not start', 'automatic mode', "
        "'vibra', 'fa rumore', 'si blocca', 'si inceppa', 'non parte', 'in automatico'. "
        "Avoid software-like or colloquial mistranslations such as 'freezes' for a machine stop."
    )
    user_msg = (
        f"TARGET_LANGUAGE: {target_language}\n\n"
        f"QUERY:\n{text}"
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=SEMANTIC_QUERY_PLANNER_MODEL,
            json_schema=_query_translation_schema(),
            timeout=20,
        )
        translated = re.sub(r"\s+", " ", str((parsed or {}).get("text") or "")).strip()
        return translated or text
    except Exception:
        return text


@dataclass(frozen=True)
class SymptomCrosslingualExpansionsRuntime:
    ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL: Any
    _dedup_text_values: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]
    re: Any


def symptom_crosslingual_expansions(q: str, source_language: str, *, runtime: SymptomCrosslingualExpansionsRuntime) -> list[str]:
    ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL = runtime.ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL
    _dedup_text_values = runtime._dedup_text_values
    _query_symptom_profile = runtime._query_symptom_profile
    re = runtime.re
    if not ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL:
        return []

    source_language = str(source_language or "").strip().lower()
    if source_language not in {"it", "en"}:
        return []

    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])
    out: list[str] = []

    def add(x: str):
        x = re.sub(r"\s+", " ", str(x or "").strip())
        if x:
            out.append(x)

    if source_language == "it":
        if "vibration" in classes:
            if profile.get("has_bending_anchor"):
                add("the machine vibrates during bending")
            elif profile.get("has_feed_anchor"):
                add("the machine vibrates during feed")
            else:
                add("the machine vibrates")
        if "noise" in classes:
            if profile.get("has_bending_anchor"):
                add("the machine makes noise during bending")
            elif profile.get("has_feed_anchor"):
                add("the machine makes noise during feed")
            else:
                add("the machine makes noise")
        if "jam" in classes:
            if profile.get("has_feed_anchor"):
                add("the machine jams during feed")
                add("the machine stops during feed")
            else:
                add("the machine jams")
                add("the machine stops unexpectedly")
        if "no_start" in classes:
            if profile.get("automatic_mode"):
                add("the machine does not start in automatic mode")
                add("the machine does not start in automatic cycle")
            else:
                add("the machine does not start")
    else:
        if "vibration" in classes:
            if profile.get("has_bending_anchor"):
                add("la macchina vibra durante la piegatura")
            elif profile.get("has_feed_anchor"):
                add("la macchina vibra durante l'avanzamento")
            else:
                add("la macchina vibra")
        if "noise" in classes:
            if profile.get("has_bending_anchor"):
                add("la macchina fa rumore durante la piegatura")
            elif profile.get("has_feed_anchor"):
                add("la macchina fa rumore durante l'avanzamento")
            else:
                add("la macchina fa rumore")
        if "jam" in classes:
            if profile.get("has_feed_anchor"):
                add("la macchina si inceppa durante l'avanzamento")
                add("la macchina si blocca durante l'avanzamento")
            else:
                add("la macchina si blocca")
                add("la macchina si inceppa")
        if "no_start" in classes:
            if profile.get("automatic_mode"):
                add("la macchina non parte in automatico")
                add("la macchina non si avvia in ciclo automatico")
            else:
                add("la macchina non parte")
                add("la macchina non si avvia")

    return _dedup_text_values(out, limit=3)


@dataclass(frozen=True)
class AugmentCrosslingualQueryPlanRuntime:
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _simple_query_language: Callable[..., Any]
    _symptom_crosslingual_expansions: Callable[..., Any]
    _translate_query_for_retrieval: Callable[..., Any]
    re: Any


def augment_crosslingual_query_plan(q: str, planner: Optional[dict], *, runtime: AugmentCrosslingualQueryPlanRuntime) -> dict:
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _simple_query_language = runtime._simple_query_language
    _symptom_crosslingual_expansions = runtime._symptom_crosslingual_expansions
    _translate_query_for_retrieval = runtime._translate_query_for_retrieval
    re = runtime.re
    planner = dict(planner or {})
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    query_language = str(planner.get("query_language") or _simple_query_language(q_norm)).strip().lower()

    planner["crosslingual_dense_queries"] = []
    planner["crosslingual_lexical_queries"] = []

    if not q_norm or query_language not in {"it", "en"}:
        return planner

    target_language = "en" if query_language == "it" else "it"
    base_text = re.sub(r"\s+", " ", str(planner.get("normalized_query") or q_norm)).strip() or q_norm
    translated = _translate_query_for_retrieval(base_text, target_language)
    deterministic = _symptom_crosslingual_expansions(base_text, query_language)

    cross_texts = []
    if translated and translated.lower() != base_text.lower():
        cross_texts.append(translated)
    cross_texts.extend(deterministic)

    planner["crosslingual_dense_queries"] = _dedup_text_values(cross_texts, limit=3)
    planner["crosslingual_lexical_queries"] = _dedup_text_values(cross_texts, limit=3)

    return planner


@dataclass(frozen=True)
class SemanticQueryPlanRuntime:
    SEMANTIC_MAX_DENSE_QUERIES: Any
    SEMANTIC_MAX_LEXICAL_QUERIES: Any
    SEMANTIC_QUERY_PLANNER_MODEL: Any
    SEMANTIC_QUERY_PLANNER_TIMEOUT: Any
    _count_query_tokens: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _openai_chat_json: Callable[..., Any]
    _simple_query_language: Callable[..., Any]
    re: Any


def semantic_query_plan(q: str, *, mode: str = "ask", runtime: SemanticQueryPlanRuntime) -> dict:
    SEMANTIC_MAX_DENSE_QUERIES = runtime.SEMANTIC_MAX_DENSE_QUERIES
    SEMANTIC_MAX_LEXICAL_QUERIES = runtime.SEMANTIC_MAX_LEXICAL_QUERIES
    SEMANTIC_QUERY_PLANNER_MODEL = runtime.SEMANTIC_QUERY_PLANNER_MODEL
    SEMANTIC_QUERY_PLANNER_TIMEOUT = runtime.SEMANTIC_QUERY_PLANNER_TIMEOUT
    _count_query_tokens = runtime._count_query_tokens
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _openai_chat_json = runtime._openai_chat_json
    _simple_query_language = runtime._simple_query_language
    re = runtime.re
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    fallback_style = "telegraphic" if _count_query_tokens(q_norm) <= 6 else "natural"
    fallback = {
        "normalized_query": q_norm,
        "dense_queries": [q_norm] if q_norm else [],
        "lexical_queries": [q_norm] if q_norm else [],
        "query_style": fallback_style,
        "query_language": _simple_query_language(q_norm),
    }

    if not q_norm:
        return fallback

    schema = {
        "name": "semantic_query_plan_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "normalized_query": {"type": "string"},
                "dense_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(1, SEMANTIC_MAX_DENSE_QUERIES),
                },
                "lexical_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(1, SEMANTIC_MAX_LEXICAL_QUERIES),
                },
                "query_style": {
                    "type": "string",
                    "enum": [
                        "telegraphic",
                        "natural",
                        "identifier_lookup",
                        "contact_lookup",
                    ],
                },
                "query_language": {
                    "type": "string",
                    "enum": ["it", "en", "mixed", "other"],
                },
            },
            "required": [
                "normalized_query",
                "dense_queries",
                "lexical_queries",
                "query_style",
                "query_language",
            ],
        },
    }

    system_msg = (
        "You prepare retrieval plans for a technical documentation assistant that must work across different machine sectors. "
        "Queries and documents may be in Italian or English. "
        "Work semantically and domain-agnostically. "
        "Preserve the user's meaning exactly. "
        "Do not inject unsupported components, causes, sectors, or jargon. "
        "Produce a small set of retrieval-ready rewrites: "
        "dense_queries for semantic embedding recall, lexical_queries for keyword/FTS rescue. "
        "You may include one careful translation between Italian and English if it improves mixed-language recall, "
        "but do not broaden the meaning. "
        "query_style should describe the surface form of the query, not the machine domain."
    )

    user_msg = (
        f"MODE: {mode}\n"
        f"QUERY:\n{q_norm}"
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=SEMANTIC_QUERY_PLANNER_MODEL,
            json_schema=schema,
            timeout=SEMANTIC_QUERY_PLANNER_TIMEOUT,
        )
        if not isinstance(parsed, dict):
            return fallback

        normalized_query = re.sub(r"\s+", " ", str(parsed.get("normalized_query") or q_norm)).strip() or q_norm
        dense_queries = _dedup_text_values(
            [q_norm, normalized_query] + list(parsed.get("dense_queries") or []),
            limit=max(2, SEMANTIC_MAX_DENSE_QUERIES + 1),
        )
        lexical_queries = _dedup_text_values(
            [q_norm, normalized_query] + list(parsed.get("lexical_queries") or []),
            limit=max(2, SEMANTIC_MAX_LEXICAL_QUERIES + 1),
        )
        query_style = str(parsed.get("query_style") or fallback_style).strip().lower()
        if query_style not in {"telegraphic", "natural", "identifier_lookup", "contact_lookup"}:
            query_style = fallback_style

        query_language = str(parsed.get("query_language") or fallback["query_language"]).strip().lower()
        if query_language not in {"it", "en", "mixed", "other"}:
            query_language = fallback["query_language"]

        return {
            "normalized_query": normalized_query,
            "dense_queries": dense_queries or [q_norm],
            "lexical_queries": lexical_queries or [q_norm],
            "query_style": query_style,
            "query_language": query_language,
        }
    except Exception:
        return fallback


@dataclass(frozen=True)
class AskEvidenceStopwordsRuntime:
    pass


def ask_evidence_stopwords( *, runtime: AskEvidenceStopwordsRuntime) -> set[str]:
    return {
        # IT
        "che", "cosa", "come", "quale", "quali", "quanto", "quanti", "quando", "dove", "perche", "perché",
        "sono", "devo", "deve", "fare", "faccio", "indica", "indicati", "indicate", "della", "delle", "degli",
        "dell", "alla", "allo", "alle", "con", "per", "sul", "sulla", "sulle", "nel", "nella", "nelle",
        "documento", "documenti", "macchina", "manuale", "principali", "richiesti", "richieste", "alcune",
        # EN
        "what", "which", "how", "when", "where", "why", "does", "must", "should", "with", "from", "about",
        "document", "documents", "machine", "manual", "main", "required", "requirements", "some",
    }


@dataclass(frozen=True)
class AskEvidenceTokenizeRuntime:
    _ask_evidence_stopwords: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_evidence_tokenize(text: str, *, runtime: AskEvidenceTokenizeRuntime) -> list[str]:
    _ask_evidence_stopwords = runtime._ask_evidence_stopwords
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    t = _normalize_unicode_advanced(text or "").lower()
    toks = re.findall(r"[a-z0-9à-öø-ÿ_+\-.,/°²≤>=]+", t)
    stop = _ask_evidence_stopwords()
    out = []
    for tok in toks:
        tok = tok.strip(".,;:!?()[]{}\"'")
        if not tok or tok in stop:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(tok)
    return out


@dataclass(frozen=True)
class AskEvidenceCodeTokensRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_evidence_code_tokens(text: str, *, runtime: AskEvidenceCodeTokensRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    raw = _normalize_unicode_advanced(text or "")
    # Codes/part numbers often include hyphens, digits, commas and letters.
    candidates = re.findall(r"\b[A-Z0-9][A-Z0-9_+./,\-]{4,}[A-Z0-9]\b", raw.upper())
    out = []
    seen = set()
    for x in candidates:
        x = x.strip(".,;:!?()[]{}")
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out[:24]


@dataclass(frozen=True)
class AskEvidenceNumberTokensRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_evidence_number_tokens(text: str, *, runtime: AskEvidenceNumberTokensRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    raw = _normalize_unicode_advanced(text or "")
    vals = re.findall(r"(?<!\w)[+\-]?(?:\d{1,4}(?:[.,]\d{1,6})?|\d{2,})(?:\s?(?:mm|cm|m/s²|m/s2|m/s|bar|n|kn|s|ore|hours|hz|kw|v|a|arcmin|°c|°))?", raw.lower())
    out = []
    seen = set()
    for v in vals:
        v = re.sub(r"\s+", " ", v.strip())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:30]


@dataclass(frozen=True)
class AskEvidenceQuerySchemaRuntime:
    pass


def ask_evidence_query_schema( *, runtime: AskEvidenceQuerySchemaRuntime) -> dict:
    return {
        "name": "ask_evidence_query_profile_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question_language": {"type": "string", "enum": ["it", "en", "other"]},
                "answer_type": {
                    "type": "string",
                    "enum": ["factual", "procedural", "list", "table", "component_spec", "diagnostic", "comparison", "no_answer_check", "general"],
                },
                "search_phrases": {"type": "array", "items": {"type": "string"}, "maxItems": 18},
                "search_terms_it": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
                "search_terms_en": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
                "required_information": {"type": "array", "items": {"type": "string"}, "maxItems": 18},
                "important_codes_or_numbers": {"type": "array", "items": {"type": "string"}, "maxItems": 18},
            },
            "required": ["question_language", "answer_type", "search_phrases", "search_terms_it", "search_terms_en", "required_information", "important_codes_or_numbers"],
        },
    }


@dataclass(frozen=True)
class AskEvidenceFallbackProfileRuntime:
    _ask_evidence_code_tokens: Callable[..., Any]
    _ask_evidence_number_tokens: Callable[..., Any]
    _ask_evidence_tokenize: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]


def ask_evidence_fallback_profile(q: str, response_language: str = "it", *, runtime: AskEvidenceFallbackProfileRuntime) -> dict:
    _ask_evidence_code_tokens = runtime._ask_evidence_code_tokens
    _ask_evidence_number_tokens = runtime._ask_evidence_number_tokens
    _ask_evidence_tokenize = runtime._ask_evidence_tokenize
    _dedup_text_values = runtime._dedup_text_values
    toks = _ask_evidence_tokenize(q)
    codes = _ask_evidence_code_tokens(q)
    nums = _ask_evidence_number_tokens(q)
    return {
        "question_language": response_language if response_language in {"it", "en"} else "it",
        "answer_type": "general",
        "search_phrases": _dedup_text_values([q] + codes + nums, limit=18),
        "search_terms_it": _dedup_text_values(toks + codes + nums, limit=24),
        "search_terms_en": _dedup_text_values(toks + codes + nums, limit=24),
        "required_information": _dedup_text_values(toks[:12], limit=18),
        "important_codes_or_numbers": _dedup_text_values(codes + nums, limit=18),
    }


@dataclass(frozen=True)
class AskEvidenceQueryProfileRuntime:
    ASK_EVIDENCE_ANALYZER_MODEL: Any
    OPENAI_API_KEY: Any
    OPENAI_CHAT_MODEL: Any
    OPENAI_RERANK_MODEL: Any
    _ask_evidence_fallback_profile: Callable[..., Any]
    _ask_evidence_query_schema: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _openai_chat_json_models: Callable[..., Any]


def ask_evidence_query_profile(q: str, response_language: str, *, runtime: AskEvidenceQueryProfileRuntime) -> dict:
    """Extract query needs without using any document-specific or benchmark-specific facts."""
    ASK_EVIDENCE_ANALYZER_MODEL = runtime.ASK_EVIDENCE_ANALYZER_MODEL
    OPENAI_API_KEY = runtime.OPENAI_API_KEY
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    _ask_evidence_fallback_profile = runtime._ask_evidence_fallback_profile
    _ask_evidence_query_schema = runtime._ask_evidence_query_schema
    _dedup_text_values = runtime._dedup_text_values
    _openai_chat_json_models = runtime._openai_chat_json_models
    fallback = _ask_evidence_fallback_profile(q, response_language)
    if not OPENAI_API_KEY:
        return fallback

    system_msg = (
        "You analyze industrial-document questions for retrieval. Do not answer the question. "
        "Extract generic search phrases, bilingual Italian/English terms, requested attributes, codes and numbers. "
        "Do not add facts that are not in the user question. Do not use any hidden benchmark knowledge."
    )
    user_msg = (
        f"QUESTION:\n{q}\n\n"
        "Return a retrieval profile. Include both Italian and English equivalents when useful, because documents and questions may be in either language. "
        "For technical/specification questions, include component names, attribute labels, units, table labels and code-like tokens found in the question."
    )
    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ASK_EVIDENCE_ANALYZER_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_ask_evidence_query_schema(),
            timeout=35,
        )
        if isinstance(parsed, dict):
            # Merge deterministic tokens so exact codes/numbers from the question cannot be lost.
            parsed["search_phrases"] = _dedup_text_values(list(parsed.get("search_phrases") or []) + fallback["search_phrases"], limit=24)
            parsed["search_terms_it"] = _dedup_text_values(list(parsed.get("search_terms_it") or []) + fallback["search_terms_it"], limit=32)
            parsed["search_terms_en"] = _dedup_text_values(list(parsed.get("search_terms_en") or []) + fallback["search_terms_en"], limit=32)
            parsed["important_codes_or_numbers"] = _dedup_text_values(list(parsed.get("important_codes_or_numbers") or []) + fallback["important_codes_or_numbers"], limit=24)
            return parsed
    except Exception as e:
        print("ASK_EVIDENCE_PROFILE_FAIL", str(e)[:300])
    return fallback


@dataclass(frozen=True)
class AskStructuredManualSupportTermsRuntime:
    _ask_structured_direct_stopwords: Callable[..., Any]
    _ask_structured_direct_terms: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_manual_support_terms(q: str, planner: Optional[dict], structured_citations: list[dict], *, runtime: AskStructuredManualSupportTermsRuntime) -> list[str]:
    """Operation-specific manual support terms.

    Structured records remain primary. Manual support must be relevant to the
    user operation, not merely a generic safety page. The terms are generated
    from the question and the structured records, with small bilingual IT/EN
    expansions for common industrial verbs/nouns. No document ids, expected test
    answers or machine-specific values are encoded here.
    """
    _ask_structured_direct_stopwords = runtime._ask_structured_direct_stopwords
    _ask_structured_direct_terms = runtime._ask_structured_direct_terms
    _content_term_set = runtime._content_term_set
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    raw_terms = list(_ask_structured_direct_terms(q, planner=planner, limit=22))
    structured_text = "\n".join(str(c.get("chunk_full") or c.get("snippet") or "") for c in (structured_citations or []))
    raw_terms.extend(list(_content_term_set(structured_text, limit=24)))

    joined = _normalize_unicode_advanced((q or "") + "\n" + structured_text).lower()
    expansions: list[str] = []
    bilingual_groups = [
        (["coil", "bobina", "bobine"], ["coil", "bobina", "bobine"]),
        (["change", "cambio", "cambiare", "sostitu", "replacement", "replace"], ["change", "cambio", "cambiare", "sostituzione", "sostituire", "replacement", "replace"]),
        (["old", "vecchio", "vecchia", "remove", "rimuovere", "togliere"], ["old", "vecchio", "vecchia", "remove", "rimuovere", "togliere"]),
        (["new", "nuovo", "nuova", "insert", "inserire", "mettere", "install"], ["new", "nuovo", "nuova", "insert", "inserire", "mettere", "installare", "montare"]),
        (["procedure", "procedura", "procedimento", "sequence", "sequenza"], ["procedure", "procedura", "procedimento", "sequence", "sequenza"]),
        (["step", "passo", "fase"], ["step", "passo", "fase"]),
        (["operation", "operazione", "operativo", "operativa"], ["operation", "operazione", "operativo", "operativa"]),
    ]
    for triggers, adds in bilingual_groups:
        if any(t in joined for t in triggers):
            expansions.extend(adds)

    # Keep only terms that help find the same operation in the manual. Generic
    # safety words are handled separately so they cannot outrank operational pages.
    generic_safety = {
        "sicurezza", "safety", "manuale", "manual", "manutenzione", "maintenance",
        "operatore", "operator", "qualificato", "qualified", "dpi", "ppe",
        "guanti", "gloves", "occhiali", "goggles", "protezione", "protection",
        "elettrica", "electrical", "pneumatica", "pneumatic", "sezionatore",
        "disconnect", "interruttore", "switch", "lucchetto", "lock", "blocco", "lockout",
    }
    out: list[str] = []
    seen: set[str] = set()
    stop = _ask_structured_direct_stopwords()
    for raw in list(raw_terms) + expansions:
        t = _normalize_unicode_advanced(str(raw or "")).lower().strip(" -–—:;,.")
        if len(t) < 3 or t in stop or t in generic_safety or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 44:
            break
    return out


@dataclass(frozen=True)
class AskStructuredManualSupportSearchSchemaRuntime:
    pass


def ask_structured_manual_support_search_schema( *, runtime: AskStructuredManualSupportSearchSchemaRuntime) -> dict:
    return {
        "name": "ask_structured_manual_support_search_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "manual_search_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 18},
                "manual_search_concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "reason": {"type": "string"},
            },
            "required": ["manual_search_terms", "manual_search_concepts", "reason"],
        },
    }


@dataclass(frozen=True)
class AskStructuredManualSupportSearchTermsWithLlmRuntime:
    ASK_EVIDENCE_ANALYZER_MODEL: Any
    ASK_STRUCTURED_DIRECT_MODEL: Any
    ASK_STRUCTURED_DIRECT_TIMEOUT: Any
    OPENAI_API_KEY: Any
    OPENAI_CHAT_MODEL: Any
    OPENAI_RERANK_MODEL: Any
    _ask_full_context_sources_block: Callable[..., Any]
    _ask_structured_direct_stopwords: Callable[..., Any]
    _ask_structured_manual_support_search_schema: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _openai_chat_json_models: Callable[..., Any]
    re: Any


def ask_structured_manual_support_search_terms_with_llm(
    *,
    q: str,
    response_language: str,
    structured_citations: list[dict],
    runtime: AskStructuredManualSupportSearchTermsWithLlmRuntime,
) -> list[str]:
    """Infer how a manual may describe support for a structured operation.

    This is not a dictionary of expected answers. The model receives the user
    question plus the primary structured records and produces search expressions
    that an official manual might use for the same operation, its immediate
    prerequisite, or its immediate continuation. This is needed because shop-floor
    structured procedures can use shorthand while manuals use formal wording.
    """
    ASK_EVIDENCE_ANALYZER_MODEL = runtime.ASK_EVIDENCE_ANALYZER_MODEL
    ASK_STRUCTURED_DIRECT_MODEL = runtime.ASK_STRUCTURED_DIRECT_MODEL
    ASK_STRUCTURED_DIRECT_TIMEOUT = runtime.ASK_STRUCTURED_DIRECT_TIMEOUT
    OPENAI_API_KEY = runtime.OPENAI_API_KEY
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    _ask_full_context_sources_block = runtime._ask_full_context_sources_block
    _ask_structured_direct_stopwords = runtime._ask_structured_direct_stopwords
    _ask_structured_manual_support_search_schema = runtime._ask_structured_manual_support_search_schema
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _openai_chat_json_models = runtime._openai_chat_json_models
    re = runtime.re
    if not OPENAI_API_KEY or not structured_citations:
        return []

    structured_block = _ask_full_context_sources_block(
        structured_citations,
        max_context_chars=7000,
    )
    system_msg = (
        "You prepare search terms for an industrial machine manual. You are not answering the user. "
        "Given a user question and primary structured procedure/step records, infer the formal wording the official manual may use for: "
        "the same operation, an immediate prerequisite, or an immediate continuation needed to complete that operation. "
        "Use semantic reasoning, not fixed keywords. Include terms in the user's language and likely manual language when useful. "
        "Do not invent values, ids, page numbers, or facts. Do not add broad generic safety unless it is necessary to find directly applicable prerequisites."
    )
    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"PRIMARY STRUCTURED SOURCES:\n{structured_block}\n\n"
        "Return concise search terms/concepts only. Prefer manual phrasing, component names, actions, materials, and immediate before/after operations. "
        "If a structured procedure uses workshop shorthand, infer plausible formal manual terms without asserting they exist."
    )
    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ASK_STRUCTURED_DIRECT_MODEL, ASK_EVIDENCE_ANALYZER_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_ask_structured_manual_support_search_schema(),
            timeout=min(int(ASK_STRUCTURED_DIRECT_TIMEOUT or 60), 45),
        )
    except Exception as e:
        print("ASK_STRUCTURED_MANUAL_SEARCH_TERMS_FAIL", str(e)[:700])
        return []

    if not isinstance(parsed, dict):
        return []

    out: list[str] = []
    seen: set[str] = set()
    raw_items = list(parsed.get("manual_search_terms") or []) + list(parsed.get("manual_search_concepts") or [])
    stop = _ask_structured_direct_stopwords()
    for raw in raw_items:
        t = _normalize_unicode_advanced(str(raw or "")).lower().strip(" -–—:;,.()[]{}")
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 3 or t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 26:
            break
    return out


@dataclass(frozen=True)
class V12FamilyFacetQueriesRuntime:
    _dedup_text_values: Callable[..., Any]


def v12_family_facet_queries(planner: Optional[dict], *, runtime: V12FamilyFacetQueriesRuntime) -> list[dict]:
    _dedup_text_values = runtime._dedup_text_values
    rows: list[dict] = []
    for raw in list((planner or {}).get("facet_queries") or []):
        if not isinstance(raw, dict):
            continue
        facet = str(raw.get("facet") or "").strip()
        if not facet:
            continue
        rows.append(
            {
                "facet": facet,
                "answer_type": str(raw.get("answer_type") or "").strip().lower(),
                "must_cover": bool(raw.get("must_cover", True)),
                "dense_queries": _dedup_text_values(raw.get("dense_queries") or [], limit=8),
                "lexical_queries": _dedup_text_values(raw.get("lexical_queries") or [], limit=8),
                "exact_terms": _dedup_text_values(raw.get("exact_terms") or [], limit=10),
            }
        )
    return rows


@dataclass(frozen=True)
class V13QueryPlanSchemaRuntime:
    pass


def v13_query_plan_schema( *, runtime: V13QueryPlanSchemaRuntime) -> dict:
    return {
        "name": "machinemind_v13_retrieval_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["factual", "procedural", "diagnostic", "listing", "comparison", "explanation", "other"],
                },
                "normalized_query": {"type": "string"},
                "query_language": {"type": "string", "enum": ["it", "en", "mixed", "other"]},
                "dense_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "lexical_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "exact_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                "required_facets": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            },
            "required": [
                "intent", "normalized_query", "query_language", "dense_queries",
                "lexical_queries", "exact_terms", "required_facets", "ambiguities"
            ],
        },
    }


