"""P4-B: existing retrieval behavior behind explicit dependencies.

Pure extraction from the P4-A baseline. Existing source selection heuristics,
SQL, ranking weights, tie-breaking and failure behavior are preserved, including
legacy limitations. No application, database or provider client is imported.
Dependencies are resolved by the composition root at call time.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
import re
import math
import json
from difflib import SequenceMatcher

@dataclass(frozen=True)
class DbFetchRelatedStepPagesRuntime:
    STRUCTURED_RELATION_PROCEDURE_STEP: str
    _db_conn: Callable[..., Any]


def db_fetch_related_step_pages(*, company_id: str, machine_id: str, parent_source_key: str, text_chars: int, runtime: DbFetchRelatedStepPagesRuntime) -> list[tuple]:
    """Return canonical Step children in Bubble order without semantic guessing."""
    STRUCTURED_RELATION_PROCEDURE_STEP = runtime.STRUCTURED_RELATION_PROCEDURE_STEP
    _db_conn = runtime._db_conn
    if not (company_id and machine_id and parent_source_key):
        return []

    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.child_source_key,
                    r.ordinal,
                    p.machine_id,
                    p.page_number,
                    LEFT(COALESCE(p.text, ''), %s) AS page_text
                FROM public.structured_source_relations AS r
                JOIN public.document_pages AS p
                  ON p.company_id = r.company_id
                 AND p.bubble_document_id = r.child_source_key
                WHERE r.company_id = %s
                  AND r.machine_id = %s
                  AND r.parent_source_key = %s
                  AND r.relation_type = %s
                  AND p.text IS NOT NULL
                  AND length(p.text) > 10
                ORDER BY
                    r.ordinal NULLS LAST,
                    r.child_source_key,
                    p.page_number;
                """,
                (
                    int(text_chars),
                    company_id,
                    machine_id,
                    parent_source_key,
                    STRUCTURED_RELATION_PROCEDURE_STEP,
                ),
            )
            return list(cur.fetchall())
    except Exception as exc:
        print("STRUCTURED_RELATION_READ_FALLBACK", str(exc)[:700])
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


@dataclass(frozen=True)
class StructuredRescueQueryIntentRuntime:
    _count_query_tokens: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def structured_rescue_query_intent(q: str, planner: Optional[dict]=None, *, runtime: StructuredRescueQueryIntentRuntime) -> bool:
    """Return True when a user query should explicitly consider structured sources.

    Dense retrieval can prefer long PDF manual chunks. For questions about procedures,
    steps, P&S, photos/videos, or practical how-to requests, structured Bubble records
    are first-class evidence and must be allowed into the final citation set.
    """
    _count_query_tokens = runtime._count_query_tokens
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    text_parts = [str(q or "")]
    if isinstance(planner, dict):
        text_parts.append(str(planner.get("normalized_query") or ""))
        text_parts.extend(str(x or "") for x in (planner.get("lexical_queries") or []))
        text_parts.extend(str(x or "") for x in (planner.get("dense_queries") or []))

    low = _normalize_unicode_advanced(" ".join(text_parts)).lower()
    low = re.sub(r"\s+", " ", low).strip()
    if not low:
        return False

    strong_markers = [
        "procedur", "procedure", "step", "passagg", "istruzion", "operativ",
        "p&s", "problem solution", "problema", "problematic", "soluzione", "solution",
        "foto", "photo", "immagin", "image", "video", "media",
    ]
    if any(m in low for m in strong_markers):
        return True

    howto_markers = [
        "come faccio", "come fare", "come posso", "cosa devo fare", "cosa devo controllare",
        "how to", "how do i", "what should i do", "esiste", "conosci", "conosci sulla macchina",
        "hai info", "hai informazioni", "quali altre informazioni",
    ]
    # Generic Italian how-to form: "come si <verbo/azione> ...".
    # This is not operation-specific; it prevents practical questions such as
    # "come si raddrizza il filo?" from falling back to long PDF manuals only.
    generic_howto = bool(re.search(r"\bcome\s+si\s+[a-zà-öø-ÿ0-9][a-zà-öø-ÿ0-9_\-/]{2,}", low))
    return (any(m in low for m in howto_markers) or generic_howto) and _count_query_tokens(low) >= 3


@dataclass(frozen=True)
class StructuredRescuePrefixesForQueryRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def structured_rescue_prefixes_for_query(q: str, planner: Optional[dict]=None, *, runtime: StructuredRescuePrefixesForQueryRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    text_parts = [str(q or "")]
    if isinstance(planner, dict):
        text_parts.append(str(planner.get("normalized_query") or ""))
    low = _normalize_unicode_advanced(" ".join(text_parts)).lower()

    prefixes: list[str] = []
    def add(prefix: str) -> None:
        if prefix not in prefixes:
            prefixes.append(prefix)

    if any(x in low for x in ["procedur", "procedure", "operativ", "istruzion"]):
        add("procedure")
        add("step")
    if any(x in low for x in ["step", "passagg", "fase"]):
        add("step")
        add("procedure")
    if any(x in low for x in ["p&s", "problem solution", "problema", "problematic", "soluzione", "solution"]):
        add("ps")
    if any(x in low for x in ["foto", "photo", "immagin", "image"]):
        add("md_photo")
    if "video" in low:
        add("md_video")

    if not prefixes:
        prefixes = ["procedure", "step", "ps", "md_photo", "md_video"]

    return prefixes


@dataclass(frozen=True)
class StructuredRescueTermsRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def structured_rescue_terms(q: str, planner: Optional[dict]=None, limit: int=10, *, runtime: StructuredRescueTermsRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    texts = [str(q or "")]
    if isinstance(planner, dict):
        texts.append(str(planner.get("normalized_query") or ""))
        texts.extend(str(x or "") for x in (planner.get("lexical_queries") or []))
        texts.extend(str(x or "") for x in (planner.get("dense_queries") or []))

    raw = _normalize_unicode_advanced(" ".join(texts)).lower()
    raw = re.sub(r"[^a-z0-9à-öø-ÿ]+", " ", raw)

    stop = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from",
        "this", "that", "these", "those", "question", "answer", "issue", "problem", "machine",
        "document", "documents", "manual", "what", "should", "how", "does", "there", "exist",
        "il", "lo", "la", "i", "gli", "le", "con", "per", "quando", "durante", "mentre", "dopo", "prima",
        "questo", "questa", "questi", "queste", "domanda", "risposta", "documenti", "documento",
        "macchina", "sistema", "esiste", "conosci", "info", "informazioni", "quali", "altre", "questa",
        "come", "faccio", "fare", "posso", "devo", "cosa", "controllare", "hai", "sulla", "sul",
        # Do not use source-type words as content terms; prefixes handle them.
        "procedura", "procedure", "step", "passaggio", "passaggi", "problema", "soluzione", "foto", "video",
    }

    terms: list[str] = []
    seen = set()
    for tok in raw.split():
        tok = tok.strip()
        if len(tok) < 3 or tok in stop or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= limit:
            break
    return terms


@dataclass(frozen=True)
class FetchStructuredRescueCandidatesRuntime:
    ASK_SNIPPET_CHARS: int
    STRUCTURED_RESCUE_ENABLED: bool
    STRUCTURED_RESCUE_MAX_HITS: int
    STRUCTURED_RESCUE_SCAN_LIMIT: int
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _structured_rescue_prefixes_for_query: Callable[..., Any]
    _structured_rescue_query_intent: Callable[..., Any]
    _structured_rescue_terms: Callable[..., Any]


def fetch_structured_rescue_candidates(*, company_id: str, machine_id: str, q: str, planner: Optional[dict], top_k: int, doc_ids: Optional[list[str]]=None, bubble_document_id: Optional[str]=None, runtime: FetchStructuredRescueCandidatesRuntime) -> list[dict]:
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    STRUCTURED_RESCUE_ENABLED = runtime.STRUCTURED_RESCUE_ENABLED
    STRUCTURED_RESCUE_MAX_HITS = runtime.STRUCTURED_RESCUE_MAX_HITS
    STRUCTURED_RESCUE_SCAN_LIMIT = runtime.STRUCTURED_RESCUE_SCAN_LIMIT
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _source_type_from_document_id = runtime._source_type_from_document_id
    _structured_rescue_prefixes_for_query = runtime._structured_rescue_prefixes_for_query
    _structured_rescue_query_intent = runtime._structured_rescue_query_intent
    _structured_rescue_terms = runtime._structured_rescue_terms
    if not STRUCTURED_RESCUE_ENABLED:
        return []
    if doc_ids or bubble_document_id:
        return []
    if not _structured_rescue_query_intent(q, planner):
        return []

    prefixes = _structured_rescue_prefixes_for_query(q, planner)
    terms = _structured_rescue_terms(q, planner)

    like_clauses = " OR ".join(["bubble_document_id LIKE %s" for _ in prefixes])
    params: list[Any] = [ASK_SNIPPET_CHARS]
    params.extend([f"{p}:%" for p in prefixes])
    params.extend([company_id, machine_id, STRUCTURED_RESCUE_SCAN_LIMIT])

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       left(chunk_text, 2000) AS chunk_full
                FROM public.document_chunks
                WHERE ({like_clauses})
                  AND company_id = %s
                  AND embedding IS NOT NULL
                  AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                ORDER BY bubble_document_id, chunk_index
                LIMIT %s;
                """,
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    low_q = _normalize_unicode_advanced(str(q or "")).lower()
    scored: list[dict] = []

    for (bdid, chunk_index, page_from, page_to, snippet, chunk_full) in rows:
        bdid_s = str(bdid or "")
        st = _source_type_from_document_id(bdid_s)
        text = _normalize_unicode_advanced((chunk_full or snippet or "")).lower()

        term_hits = sum(1 for t in terms if t in text)
        phrase_bonus = 0.0
        if len(terms) >= 2:
            for i in range(len(terms) - 1):
                if f"{terms[i]} {terms[i + 1]}" in text:
                    phrase_bonus += 1.25

        source_bonus = 0.0
        if st == "procedure" and any(x in low_q for x in ["procedur", "procedure", "operativ", "istruzion"]):
            source_bonus += 1.20
        if st == "step" and any(x in low_q for x in ["step", "passagg", "fase"]):
            source_bonus += 1.00
        if st == "ps" and any(x in low_q for x in ["p&s", "problem", "problema", "soluzione", "solution"]):
            source_bonus += 1.00
        if st == "md_photo" and any(x in low_q for x in ["foto", "photo", "immagin", "image"]):
            source_bonus += 1.00
        if st == "md_video" and "video" in low_q:
            source_bonus += 1.00

        # If the query has content terms, require at least one content hit.
        # If it has no content terms but is a pure listing query, source_bonus/source type is enough.
        if terms and term_hits <= 0:
            continue

        score = float(term_hits) + phrase_bonus + source_bonus
        if score <= 0:
            continue

        similarity = min(0.84, 0.58 + 0.045 * term_hits + 0.045 * phrase_bonus + 0.04 * source_bonus)
        retrieval_score = min(0.92, similarity + 0.09)
        citation_id = f"{bdid_s}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"

        scored.append(
            {
                "citation_id": citation_id,
                "bubble_document_id": bdid_s,
                "chunk_index": int(chunk_index),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": (snippet or "").strip(),
                "chunk_full": (chunk_full or "").strip(),
                "similarity": float(similarity),
                "retrieval_score": float(retrieval_score),
                "source_type": st,
                "structured_rescue": True,
                "structured_rescue_score": float(score),
                "overlap_score": min(1.0, 0.18 * term_hits + 0.10 * phrase_bonus),
                "specificity_score": 0.08,
                "embedding_list": [],
            }
        )

    scored.sort(
        key=lambda x: (
            -float(x.get("structured_rescue_score") or 0.0),
            0 if str(x.get("source_type")) == "procedure" else 1,
            str(x.get("bubble_document_id") or ""),
            int(x.get("chunk_index") or 0),
        )
    )
    return _dedup_citations_by_snippet(scored, max_items=max(1, min(top_k, STRUCTURED_RESCUE_MAX_HITS)))


def ask_structured_direct_stopwords() -> set[str]:
    return {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from", "into",
        "this", "that", "these", "those", "what", "which", "how", "does", "there", "exist", "exists",
        "machine", "manual", "document", "documents", "source", "sources", "content", "contents",
        "procedure", "procedures", "step", "steps", "photo", "photos", "image", "images", "video", "videos",
        "problem", "problems", "solution", "solutions", "issue", "issues", "fault", "faults",
        "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "del", "della", "dei", "delle",
        "con", "per", "quando", "durante", "mentre", "dopo", "prima", "come", "cosa", "quali", "quale",
        "questa", "questo", "queste", "questi", "macchina", "manuale", "documento", "documenti",
        "fonte", "fonti", "contenuto", "contenuti", "informazioni", "info", "conosci", "presenti",
        "procedura", "procedure", "step", "passaggio", "passaggi", "fase", "fasi", "foto", "immagine", "immagini",
        "video", "problema", "problemi", "soluzione", "soluzioni", "errore", "errori", "operativo", "operativi",
        "extra", "oltre", "riassumi", "fammi", "dimmi", "hai", "c'è", "sono",
        # Generic action words: useful for routing but not for lexical matching.
        "fare", "faccio", "fai", "fa", "eseguire", "eseguo", "esegui", "esegue",
        "operazione", "operazioni", "attività", "attivita", "intervento", "interventi",
        "task", "activity", "activities", "operation", "operations", "execute", "perform",
    }


@dataclass(frozen=True)
class AskStructuredDirectTermsRuntime:
    _ask_structured_direct_stopwords: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_direct_terms(q: str, planner: Optional[dict]=None, limit: int=16, *, runtime: AskStructuredDirectTermsRuntime) -> list[str]:
    _ask_structured_direct_stopwords = runtime._ask_structured_direct_stopwords
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    texts = [str(q or "")]
    if isinstance(planner, dict):
        texts.append(str(planner.get("normalized_query") or ""))
        texts.extend(str(x or "") for x in (planner.get("lexical_queries") or []))
        texts.extend(str(x or "") for x in (planner.get("dense_queries") or []))

    raw = _normalize_unicode_advanced(" ".join(texts)).lower()
    tokens = re.findall(r"[a-zà-öø-ÿ0-9][a-zà-öø-ÿ0-9_\-/]{2,}", raw)
    stop = _ask_structured_direct_stopwords()
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        tok = tok.strip("_-/")
        if len(tok) < 3 or tok in stop or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class AskStructuredDirectIntentRuntime:
    _ask_structured_direct_terms: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_direct_intent(q: str, planner: Optional[dict]=None, *, runtime: AskStructuredDirectIntentRuntime) -> dict:
    """Generic routing profile for Bubble structured sources.

    This does not know benchmark questions or object ids. It only detects whether the
    user is asking for first-class structured records (procedures, steps, P&S, photos,
    videos) rather than broad manual reading.
    """
    _ask_structured_direct_terms = runtime._ask_structured_direct_terms
    _count_query_tokens = runtime._count_query_tokens
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    parts = [str(q or "")]
    if isinstance(planner, dict):
        parts.append(str(planner.get("normalized_query") or ""))
        parts.extend(str(x or "") for x in (planner.get("lexical_queries") or []))
    low = _normalize_unicode_advanced(" ".join(parts)).lower()
    low = re.sub(r"\s+", " ", low).strip()

    prefixes: list[str] = []

    def add_many(values: list[str]) -> None:
        for v in values:
            if v not in prefixes:
                prefixes.append(v)

    # Explicit source-type requests.
    if any(x in low for x in ["procedur", "procedure", "istruzion", "instruction", "operativ", "operating sequence", "sequenza", "operazione", "operation"]):
        add_many(["procedure", "step"])
    if any(x in low for x in ["step", "passagg", "fase", "fasi", "passo", "passi"]):
        add_many(["step", "procedure"])
    if any(x in low for x in ["p&s", "problem solution", "problema", "problemi", "problematic", "soluzione", "solution", "errore", "error", "fault", "issue", "reset"]):
        add_many(["ps"])
    if any(x in low for x in ["foto", "photo", "immagin", "image", "picture", "visual"]):
        add_many(["md_photo"])
    if "video" in low or "filmato" in low or "recording" in low:
        add_many(["md_video"])

    # Practical/how-to and operation-execution requests should consider operational
    # records before manuals. This is generic source hierarchy, not test-specific:
    # user-authored procedures/steps/P&S are more authoritative than a manual for
    # "how do I perform this operation?" questions.
    how_to_markers = [
        "come faccio", "come fare", "come si fa", "come si esegue", "come eseguire",
        "come posso", "in che modo", "cosa devo fare", "cosa fare",
        "how to", "how do i", "how can i", "how should i", "how is", "what should i do",
        "procedere", "eseguire", "esecuzione", "operazione", "operazioni",
        "sequenza", "sequenza operativa", "intervento", "attività", "attivita",
        "operation", "operations", "operational sequence", "task", "workflow",
        "sostituire", "sostituzione", "cambiare", "cambio", "change", "replacement",
        "installare", "install", "montare", "montaggio", "smontare", "smontaggio",
        "rimuovere", "remove", "togliere", "mettere",
    ]
    generic_howto = bool(re.search(r"\bcome\s+si\s+[a-zà-öø-ÿ0-9][a-zà-öø-ÿ0-9_\-/]{2,}", low))
    if any(x in low for x in how_to_markers) or generic_howto:
        add_many(["procedure", "step", "ps"])

    # Machine knowledge overview, especially when the user asks for non-manual content.
    overview_markers = [
        "extra manuale", "oltre al manuale", "non manuale", "contenuti operativi", "fonti operative",
        "procedure, problemi", "procedure problemi", "immagini e video", "foto e video", "riassumi procedure",
        "operational content", "beyond the manual", "outside the manual",
    ]
    broad_overview = any(x in low for x in overview_markers)
    if broad_overview:
        add_many(["procedure", "step", "ps", "md_photo", "md_video"])

    # Existence/listing questions with a content term should prefer structured records.
    if any(x in low for x in ["esiste", "ci sono", "hai un", "hai una", "do you have", "are there", "is there"]):
        if not prefixes:
            add_many(["procedure", "step", "ps", "md_photo", "md_video"])

    terms = _ask_structured_direct_terms(q, planner=planner)
    # Do not require many query tokens: a short request such as "coil change" or
    # "reset error" can be a valid structured-source request if it has content terms.
    enabled = bool(prefixes) and (_count_query_tokens(q) >= 2 or bool(terms))
    return {"enabled": enabled, "prefixes": prefixes, "terms": terms, "broad_overview": broad_overview, "query_text": low}


@dataclass(frozen=True)
class AskStructuredDirectScoreRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def ask_structured_direct_score(*, q: str, text: str, source_type: str, terms: list[str], broad_overview: bool, runtime: AskStructuredDirectScoreRuntime) -> float:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    low_text = _normalize_unicode_advanced(text or "").lower()
    low_q = _normalize_unicode_advanced(q or "").lower()
    if not low_text:
        return 0.0

    term_hits = sum(1 for t in terms if t and t in low_text)
    score = float(term_hits) * 2.0

    if len(terms) >= 2:
        for i in range(len(terms) - 1):
            phrase = f"{terms[i]} {terms[i+1]}"
            if phrase in low_text:
                score += 1.25

    # Generic source-type affinity; this does not encode object-specific facts.
    if source_type in {"procedure", "step"} and any(x in low_q for x in ["procedur", "step", "passagg", "come", "how", "operativ", "istruzion", "operazione", "operation", "cambio", "change", "sostitu", "replace", "montar", "smontar", "rimuov", "remove", "togliere", "mettere"]):
        score += 2.0
    if source_type == "ps" and any(x in low_q for x in ["problema", "problem", "soluzione", "solution", "errore", "error", "fault", "reset"]):
        score += 2.0
    if source_type == "md_photo" and any(x in low_q for x in ["foto", "photo", "immagin", "image", "picture", "mostra", "show"]):
        score += 2.0
    if source_type == "md_video" and any(x in low_q for x in ["video", "filmato", "recording"]):
        score += 2.0

    if broad_overview:
        score += 1.5

    return score


@dataclass(frozen=True)
class AskStructuredDirectFetchSourcesRuntime:
    ASK_SNIPPET_CHARS: int
    ASK_STRUCTURED_DIRECT_ENABLED: bool
    ASK_STRUCTURED_DIRECT_MAX_ITEMS: int
    ASK_STRUCTURED_DIRECT_SCAN_LIMIT: int
    ASK_STRUCTURED_DIRECT_TEXT_CHARS: int
    COMPANY_GENERAL_MACHINE_SENTINEL: str
    _ask_structured_direct_intent: Callable[..., Any]
    _ask_structured_direct_score: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]


def ask_structured_direct_fetch_sources(*, company_id: str, machine_id: str, q: str, planner: Optional[dict], top_k: int, runtime: AskStructuredDirectFetchSourcesRuntime) -> list[dict]:
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    ASK_STRUCTURED_DIRECT_ENABLED = runtime.ASK_STRUCTURED_DIRECT_ENABLED
    ASK_STRUCTURED_DIRECT_MAX_ITEMS = runtime.ASK_STRUCTURED_DIRECT_MAX_ITEMS
    ASK_STRUCTURED_DIRECT_SCAN_LIMIT = runtime.ASK_STRUCTURED_DIRECT_SCAN_LIMIT
    ASK_STRUCTURED_DIRECT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_TEXT_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_structured_direct_intent = runtime._ask_structured_direct_intent
    _ask_structured_direct_score = runtime._ask_structured_direct_score
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _safe_int = runtime._safe_int
    _source_type_from_document_id = runtime._source_type_from_document_id
    if not ASK_STRUCTURED_DIRECT_ENABLED:
        return []
    if not machine_id or str(machine_id).strip() == COMPANY_GENERAL_MACHINE_SENTINEL:
        return []

    intent = _ask_structured_direct_intent(q, planner=planner)
    if not intent.get("enabled"):
        return []

    prefixes = list(intent.get("prefixes") or [])
    if not prefixes:
        return []

    terms = _dedup_text_values(list(intent.get("terms") or []), limit=18)
    broad_overview = bool(intent.get("broad_overview"))
    text_chars = max(800, int(ASK_STRUCTURED_DIRECT_TEXT_CHARS or 5000))
    scan_limit = max(
        100,
        int(ASK_STRUCTURED_DIRECT_SCAN_LIMIT or 1200),
        int(ASK_STRUCTURED_DIRECT_MAX_ITEMS or 12) * 20,
    )
    like_clauses = " OR ".join(["bubble_document_id LIKE %s" for _ in prefixes])

    def fetch_rows(*, require_term_match: bool) -> list[tuple]:
        term_clauses = ""
        term_params: list[Any] = []
        if require_term_match and terms:
            term_clauses = " AND (" + " OR ".join(
                ["LOWER(COALESCE(text, '')) LIKE %s" for _ in terms]
            ) + ")"
            term_params = [f"%{_normalize_unicode_advanced(t).lower()}%" for t in terms]

        params: list[Any] = [text_chars, company_id]
        params.extend([f"{p}:%" for p in prefixes])
        params.append(machine_id)
        params.extend(term_params)
        params.append(scan_limit)

        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT bubble_document_id, machine_id, page_number,
                           LEFT(COALESCE(text, ''), %s) AS page_text
                    FROM public.document_pages
                    WHERE company_id = %s
                      AND ({like_clauses})
                      AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                      AND text IS NOT NULL
                      AND length(text) > 10
                      {term_clauses}
                    ORDER BY
                      CASE WHEN machine_id = %s THEN 0 ELSE 1 END,
                      bubble_document_id,
                      page_number
                    LIMIT %s;
                    """,
                    params[:-1] + [machine_id, params[-1]],
                )
                return cur.fetchall()
        finally:
            conn.close()

    # Narrow queries first ask PostgreSQL only for pages containing at least one
    # semantic/planner term, so relevant records cannot be pushed out by alphabetical
    # ordering. If inflection/translation prevents a lexical hit, fall back to the
    # bounded full structured scan and score in Python.
    rows = fetch_rows(require_term_match=bool(terms and not broad_overview))
    if not rows and terms and not broad_overview:
        rows = fetch_rows(require_term_match=False)

    if not rows:
        return []

    scored: list[dict] = []
    for idx, (bdid, mid, page_number, page_text) in enumerate(rows, start=1):
        bdid_s = str(bdid or "").strip()
        st = _source_type_from_document_id(bdid_s)
        txt = str(page_text or "").strip()
        if not bdid_s or not txt:
            continue

        score = _ask_structured_direct_score(
            q=q,
            text=txt,
            source_type=st,
            terms=terms,
            broad_overview=broad_overview,
        )
        if score <= 0.0:
            continue
        if terms and not broad_overview and not any(
            t in _normalize_unicode_advanced(txt).lower() for t in terms
        ):
            continue

        exact_machine_scope = str(mid or "").strip() == str(machine_id or "").strip()
        if exact_machine_scope:
            score += 0.75

        page_no = _safe_int(page_number, 1)
        similarity = min(0.95, 0.62 + 0.045 * score)
        citation_id = f"{bdid_s}:p{page_no}-{page_no}:structured:{idx}"
        scored.append(
            {
                "citation_id": citation_id,
                "bubble_document_id": bdid_s,
                "chunk_index": 1,
                "page_from": page_no,
                "page_to": page_no,
                "snippet": txt[: int(ASK_SNIPPET_CHARS or 900)],
                "snippet_clean": txt[: int(ASK_SNIPPET_CHARS or 900)],
                "chunk_full": txt,
                "similarity": float(similarity),
                "retrieval_score": float(similarity + 0.06),
                "source_type": st,
                "ask_structured_direct": True,
                "structured_direct_score": float(score),
                "exact_machine_scope": bool(exact_machine_scope),
                "embedding_list": [],
            }
        )

    if not scored:
        return []

    scored.sort(
        key=lambda x: (
            -float(x.get("structured_direct_score") or 0.0),
            0 if bool(x.get("exact_machine_scope")) else 1,
            str(x.get("source_type") or ""),
            str(x.get("bubble_document_id") or ""),
        )
    )

    max_items = max(1, int(ASK_STRUCTURED_DIRECT_MAX_ITEMS or 12))
    if broad_overview:
        out: list[dict] = []
        used_ids: set[str] = set()
        desired_order = ["procedure", "step", "ps", "md_photo", "md_video"]
        for st in desired_order:
            added_for_type = 0
            for row in scored:
                if str(row.get("source_type") or "") != st:
                    continue
                cid = str(row.get("citation_id") or "")
                if cid in used_ids:
                    continue
                out.append(row)
                used_ids.add(cid)
                added_for_type += 1
                if added_for_type >= (2 if st in {"procedure", "step"} else 1):
                    break
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
        for row in scored:
            if len(out) >= max_items:
                break
            cid = str(row.get("citation_id") or "")
            if cid not in used_ids:
                out.append(row)
                used_ids.add(cid)
        return out[:max_items]

    return _dedup_citations_by_snippet(scored, max_items=max_items)


@dataclass(frozen=True)
class V12ExpandPrimaryProcedureStepsRuntime:
    ASK_SNIPPET_CHARS: int
    ASK_STRUCTURED_DIRECT_SCAN_LIMIT: int
    ASK_STRUCTURED_DIRECT_TEXT_CHARS: int
    _db_conn: Callable[..., Any]
    _db_fetch_related_step_pages: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _v12_step_matches_procedure: Callable[..., Any]
    _v12_step_sort_key: Callable[..., Any]
    _v12_structured_rank: Callable[..., Any]


def v12_expand_primary_procedure_steps(*, company_id: str, machine_id: str, procedure: dict, existing_steps: list[dict], runtime: V12ExpandPrimaryProcedureStepsRuntime) -> list[dict]:
    """Load all Step children deterministically; text parsing is compatibility only."""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    ASK_STRUCTURED_DIRECT_SCAN_LIMIT = runtime.ASK_STRUCTURED_DIRECT_SCAN_LIMIT
    ASK_STRUCTURED_DIRECT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_TEXT_CHARS
    _db_conn = runtime._db_conn
    _db_fetch_related_step_pages = runtime._db_fetch_related_step_pages
    _safe_int = runtime._safe_int
    _v12_step_matches_procedure = runtime._v12_step_matches_procedure
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v12_structured_rank = runtime._v12_structured_rank
    matched: list[dict] = []
    parent_source_key = str((procedure or {}).get("bubble_document_id") or "").strip()
    for c in existing_steps or []:
        exact_parent = str(
            (c or {}).get("_v10_5_parent_source_key")
            or (c or {}).get("parent_source_key")
            or ""
        ).strip()
        relation = (exact_parent == parent_source_key) if exact_parent else _v12_step_matches_procedure(c, procedure)
        if relation is True:
            cc = dict(c)
            cc.setdefault("structured_relation_source", "indexed_parent_metadata")
            if parent_source_key:
                cc["parent_source_key"] = parent_source_key
                cc["_v10_5_parent_source_key"] = parent_source_key
            matched.append(cc)

    text_chars = max(800, int(ASK_STRUCTURED_DIRECT_TEXT_CHARS or 5000))

    def make_candidate(
        *,
        bdid: str,
        mid: Any,
        page_number: Any,
        page_text: Any,
        idx: int,
        ordinal: Optional[int],
        relation_source: str,
    ) -> dict:
        bdid_s = str(bdid or "").strip()
        txt = str(page_text or "").strip()
        page_no = _safe_int(page_number, 1)
        candidate = {
            "citation_id": f"{bdid_s}:p{page_no}-{page_no}:structured:v12:{idx}",
            "bubble_document_id": bdid_s,
            "chunk_index": 1,
            "page_from": page_no,
            "page_to": page_no,
            "snippet": txt[: int(ASK_SNIPPET_CHARS or 900)],
            "snippet_clean": txt[: int(ASK_SNIPPET_CHARS or 900)],
            "chunk_full": txt,
            "similarity": 0.94,
            "retrieval_score": 0.94,
            "source_type": "step",
            "evidence_role": "step",
            "ask_structured_direct": True,
            "structured_direct_score": 10.0,
            "exact_machine_scope": str(mid or "").strip() == str(machine_id or "").strip(),
            "embedding_list": [],
            "structured_relation_source": relation_source,
            "parent_source_key": parent_source_key,
            "_v10_5_parent_source_key": parent_source_key,
        }
        if ordinal is not None:
            candidate["structured_relation_ordinal"] = int(ordinal)
        return candidate

    # Preferred path: exact Bubble Step -> Procedure relation stored in Cloud SQL.
    relation_rows = _db_fetch_related_step_pages(
        company_id=company_id,
        machine_id=machine_id,
        parent_source_key=parent_source_key,
        text_chars=text_chars,
    )
    for idx, (bdid, ordinal, mid, page_number, page_text) in enumerate(relation_rows, start=1):
        candidate = make_candidate(
            bdid=bdid,
            mid=mid,
            page_number=page_number,
            page_text=page_text,
            idx=idx,
            ordinal=_safe_int(ordinal, 0) or None,
            relation_source="structured_source_relations",
        )
        if candidate.get("bubble_document_id") and candidate.get("chunk_full"):
            matched.append(candidate)

    # Compatibility path for already-indexed sources: no reindex is required.
    # It reads the legacy "PROCEDURA/PROCEDURE: PROC-xxx" prefix from Step text.
    if not relation_rows:
        scan_limit = max(500, int(ASK_STRUCTURED_DIRECT_SCAN_LIMIT or 1200))
        rows: list[tuple] = []
        try:
            conn = _db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT bubble_document_id, machine_id, page_number,
                               LEFT(COALESCE(text, ''), %s) AS page_text
                        FROM public.document_pages
                        WHERE company_id = %s
                          AND bubble_document_id LIKE 'step:%%'
                          AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                          AND text IS NOT NULL
                          AND length(text) > 10
                        ORDER BY
                          CASE WHEN machine_id = %s THEN 0 ELSE 1 END,
                          bubble_document_id,
                          page_number
                        LIMIT %s;
                        """,
                        (text_chars, company_id, machine_id, machine_id, scan_limit),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()
        except Exception as exc:
            print("ASK_V12_STEP_EXPANSION_FAIL", str(exc)[:500])
            rows = []

        for idx, (bdid, mid, page_number, page_text) in enumerate(rows, start=1):
            candidate = make_candidate(
                bdid=str(bdid or ""),
                mid=mid,
                page_number=page_number,
                page_text=page_text,
                idx=idx,
                ordinal=None,
                relation_source="legacy_parent_text",
            )
            if (
                candidate.get("bubble_document_id")
                and candidate.get("chunk_full")
                and _v12_step_matches_procedure(candidate, procedure) is True
            ):
                matched.append(candidate)

    best_by_doc: dict[str, dict] = {}
    for c in matched:
        bdid = str(c.get("bubble_document_id") or "").strip()
        if not bdid:
            continue
        prev = best_by_doc.get(bdid)
        if prev is None or _v12_structured_rank(c, set()) < _v12_structured_rank(prev, set()):
            best_by_doc[bdid] = c
    return sorted(best_by_doc.values(), key=_v12_step_sort_key)


@dataclass(frozen=True)
class V13FetchStructuredDenseCandidatesRuntime:
    ASK_SNIPPET_CHARS: int
    COMPANY_GENERAL_MACHINE_SENTINEL: str
    STRUCTURED_SOURCE_TYPES: set[str]
    V13_DENSE_QUERY_LIMIT: int
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _raw_rows_to_dense_candidates: Callable[..., Any]
    _rrf_merge_candidates: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _vector_literal: Callable[..., Any]


def v13_fetch_structured_dense_candidates(*, company_id: str, machine_id: str, query_vectors: list[tuple[str, list[float]]], top_k: int=18, runtime: V13FetchStructuredDenseCandidatesRuntime) -> list[dict]:
    """Semantic structured-source retrieval without keyword-gating or extra LLM calls."""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    V13_DENSE_QUERY_LIMIT = runtime.V13_DENSE_QUERY_LIMIT
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _raw_rows_to_dense_candidates = runtime._raw_rows_to_dense_candidates
    _rrf_merge_candidates = runtime._rrf_merge_candidates
    _source_type_from_document_id = runtime._source_type_from_document_id
    _vector_literal = runtime._vector_literal
    if not company_id or not machine_id or machine_id == COMPANY_GENERAL_MACHINE_SENTINEL:
        return []
    if not query_vectors:
        return []

    patterns = [f"{prefix}:%" for prefix in sorted(STRUCTURED_SOURCE_TYPES)]
    ranked_lists: list[list[dict]] = []
    per_query_k = max(8, min(24, int(top_k or 18)))

    for query_text, vector in query_vectors[:V13_DENSE_QUERY_LIMIT]:
        if not vector:
            continue
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to,
                           LEFT(chunk_text, %s) AS snippet,
                           LEFT(chunk_text, 2400) AS chunk_full,
                           1 - (embedding <=> %s::vector) AS similarity,
                           embedding,
                           CASE WHEN machine_id = %s THEN TRUE ELSE FALSE END AS exact_machine_scope
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id LIKE ANY(%s)
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                    ORDER BY embedding <=> %s::vector,
                             CASE WHEN machine_id = %s THEN 0 ELSE 1 END,
                             bubble_document_id, page_from, chunk_index
                    LIMIT %s;
                    """,
                    (
                        ASK_SNIPPET_CHARS,
                        _vector_literal(vector),
                        machine_id,
                        company_id,
                        patterns,
                        machine_id,
                        _vector_literal(vector),
                        machine_id,
                        per_query_k,
                    ),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        ranked = _raw_rows_to_dense_candidates(rows, query_used=query_text)
        for c in ranked:
            source_type = _source_type_from_document_id(c.get("bubble_document_id") or "")
            c["source_type"] = source_type
            c["structured_semantic_v13"] = True
            c["ask_structured_direct"] = True
            c["structured_direct_score"] = max(
                float(c.get("structured_direct_score") or 0.0),
                10.0 * float(c.get("similarity") or 0.0),
            )
        if ranked:
            ranked_lists.append(ranked)

    merged = _rrf_merge_candidates(ranked_lists, k=40) if ranked_lists else []
    return _dedup_citations_by_snippet(merged, max_items=max(1, int(top_k or 18)))


@dataclass(frozen=True)
class V13FetchStructuredTitleCandidatesRuntime:
    ASK_SNIPPET_CHARS: int
    ASK_STRUCTURED_DIRECT_TEXT_CHARS: int
    COMPANY_GENERAL_MACHINE_SENTINEL: str
    STRUCTURED_SOURCE_TYPES: set[str]
    V13_SOURCE_RETRIEVAL_ENABLED: bool
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES: int
    V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS: int
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE: float
    V13_SOURCE_RETRIEVAL_SCAN_LIMIT: int
    _clean_display_text: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _v13_source_sql_match_patterns: Callable[..., Any]
    _v13_source_title_match_metrics: Callable[..., Any]
    _v13_source_title_tokens: Callable[..., Any]


def v13_fetch_structured_title_candidates(*, q: str, company_id: str, machine_id: str, ai_scope: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], runtime: V13FetchStructuredTitleCandidatesRuntime) -> list[dict]:
    """Bounded cross-type title/description retrieval, independent of source keywords.

    It runs only for machine-wide ASK and only returns strong content-title matches. The
    returned rows are candidates; they never bypass the shared semantic evidence gate.
    """
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    ASK_STRUCTURED_DIRECT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_TEXT_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    V13_SOURCE_RETRIEVAL_ENABLED = runtime.V13_SOURCE_RETRIEVAL_ENABLED
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES = runtime.V13_SOURCE_RETRIEVAL_MAX_CANDIDATES
    V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS = runtime.V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE = runtime.V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE
    V13_SOURCE_RETRIEVAL_SCAN_LIMIT = runtime.V13_SOURCE_RETRIEVAL_SCAN_LIMIT
    _clean_display_text = runtime._clean_display_text
    _count_query_tokens = runtime._count_query_tokens
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    _safe_int = runtime._safe_int
    _source_type_from_document_id = runtime._source_type_from_document_id
    _v13_source_sql_match_patterns = runtime._v13_source_sql_match_patterns
    _v13_source_title_match_metrics = runtime._v13_source_title_match_metrics
    _v13_source_title_tokens = runtime._v13_source_title_tokens
    if not V13_SOURCE_RETRIEVAL_ENABLED:
        return []
    if ai_scope != "machine_all" or doc_ids or bubble_document_id:
        return []
    if not company_id or not machine_id or machine_id == COMPANY_GENERAL_MACHINE_SENTINEL:
        return []
    if _count_query_tokens(q) > V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS:
        return []

    query_tokens = _v13_source_title_tokens(q, limit=18)
    if len(query_tokens) < 2:
        return []

    # Longest terms are the most selective. Each term also receives conservative
    # singular/plural inflection patterns. SQL is only a bounded prefilter; final
    # title ranking remains in Python and every returned candidate still requires the
    # shared semantic gate before a direct response.
    search_terms = sorted(query_tokens, key=lambda token: (-len(token), token))[:10]
    term_groups = [
        (term, _v13_source_sql_match_patterns(term))
        for term in search_terms
    ]
    term_groups = [(term, variants) for term, variants in term_groups if variants]
    patterns = [f"{prefix}:%" for prefix in sorted(STRUCTURED_SOURCE_TYPES)]
    term_match_expression = " + ".join(
        [
            "CASE WHEN (" + " OR ".join(
                ["LOWER(COALESCE(text, '')) LIKE %s" for _ in variants]
            ) + ") THEN 1 ELSE 0 END"
            for _term, variants in term_groups
        ]
    )
    if not term_match_expression:
        return []
    min_term_matches = 2 if len(term_groups) >= 4 else 1
    term_pattern_params = [
        f"%{variant}%"
        for _term, variants in term_groups
        for variant in variants
    ]

    rows: list[tuple] = []
    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bubble_document_id, machine_id, page_number,
                       LEFT(COALESCE(text, ''), %s) AS source_text
                FROM public.document_pages
                WHERE company_id=%s
                  AND bubble_document_id LIKE ANY(%s)
                  AND (machine_id=%s OR machine_id IS NULL OR machine_id='')
                  AND text IS NOT NULL
                  AND length(text) > 10
                  AND ({term_match_expression}) >= %s
                ORDER BY CASE WHEN machine_id=%s THEN 0 ELSE 1 END,
                         bubble_document_id, page_number
                LIMIT %s;
                """,
                [
                    ASK_STRUCTURED_DIRECT_TEXT_CHARS,
                    company_id,
                    patterns,
                    machine_id,
                    *term_pattern_params,
                    min_term_matches,
                    machine_id,
                    V13_SOURCE_RETRIEVAL_SCAN_LIMIT,
                ],
            )
            rows = cur.fetchall()
    except Exception as exc:
        print("V13_SOURCE_TITLE_SCAN_FAIL", str(exc)[:500])
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    out: list[dict] = []
    for bdid, mid, page_number, source_text in rows or []:
        bdid_s = str(bdid or "").strip()
        text = str(source_text or "").strip()
        if not bdid_s or not text:
            continue
        fields = _parse_structured_source_fields(text)
        title = _clean_display_text(
            fields.get("title")
            or fields.get("short_description")
            or fields.get("description")
            or "",
            max_len=180,
        )
        description = _clean_display_text(fields.get("description") or "", max_len=700)
        if not title:
            continue
        metrics = _v13_source_title_match_metrics(q, title, description)
        score = float(metrics.get("score") or 0.0)
        matched_terms = int(metrics.get("matched_title_terms") or 0)
        title_term_count = int(metrics.get("title_term_count") or 0)
        if score < V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE:
            continue
        if matched_terms < 2 and not (title_term_count == 1 and score >= 0.90):
            continue

        page = _safe_int(page_number, 1)
        source_type = _source_type_from_document_id(bdid_s)
        out.append(
            {
                "citation_id": f"{bdid_s}:p{page}-{page}:c1",
                "bubble_document_id": bdid_s,
                "chunk_index": 1,
                "page_from": page,
                "page_to": page,
                "snippet": text[: int(ASK_SNIPPET_CHARS or 900)],
                "snippet_clean": text[: int(ASK_SNIPPET_CHARS or 900)],
                "chunk_full": text,
                "similarity": 0.0,
                "semantic_similarity": 0.0,
                "retrieval_score": score,
                "v13_score": score,
                "source_type": source_type,
                "exact_machine_scope": str(mid or "").strip() == str(machine_id or "").strip(),
                "ask_structured_direct": True,
                "structured_direct_score": 10.0 * score,
                "structured_title_match": True,
                "structured_title_match_score": score,
                "structured_title_coverage": float(metrics.get("title_coverage") or 0.0),
                "structured_title_query_coverage": float(metrics.get("query_coverage") or 0.0),
                "structured_title_strict_coverage": float(metrics.get("strict_coverage") or 0.0),
                "structured_title_strict_query_coverage": float(metrics.get("strict_query_coverage") or 0.0),
                "structured_title_matched_terms": matched_terms,
                "structured_title_matched_query_terms": int(metrics.get("matched_query_terms") or 0),
                "structured_title_term_count": title_term_count,
                "structured_title_description_support": float(metrics.get("description_support") or 0.0),
                "structured_title": title,
                "structured_description": description,
                "embedding_list": [],
            }
        )

    out.sort(
        key=lambda c: (
            -float(c.get("structured_title_match_score") or 0.0),
            -float(c.get("structured_title_coverage") or 0.0),
            -int(c.get("structured_title_matched_terms") or 0),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
        )
    )
    return _dedup_citations_by_snippet(out, max_items=V13_SOURCE_RETRIEVAL_MAX_CANDIDATES)


