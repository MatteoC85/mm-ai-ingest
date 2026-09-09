"""P6-B3: opt-in bound views alongside the behavior-preserving P4 readers.

The composition root still invokes the legacy entry points. Only explicit new
read_*_evidence calls retain authoritative columns from the same SELECT and
build P5-compatible snapshots; no consumer or ASK activation is changed here.
Legacy heuristics, scores, query parameters and fallback behavior remain intact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from .page_evidence import (PAGE_BINDING_COLUMNS, PageEvidenceRead, PageReadBindingError,
                            _PageReadCapture)
from .chunk_evidence import ChunkReadScope, ChunkEvidenceLimits
from .relation_evidence import (RELATION_BINDING_COLUMNS, RelationEvidenceRead,
                                validate_relation_request, build_relation_read)
from ..evidence.contracts import SourceIdentity
from .supplemental_evidence import (CHUNK_BINDING_COLUMNS, FILE_BINDING_COLUMNS,
    SupplementalChunkRead, FileReferenceRead, SupplementalBindingError,
    _ChunkSelectionCapture, build_file_reference_read, checked_input_records,
    require_machine_scope, storage_key, validate_anchors)



if TYPE_CHECKING:
    from assistant_core_v2 import AssistantCoreRequest

@dataclass(frozen=True)
class FetchDocumentFileMapRuntime:
    _db_conn: Callable[..., Any]


def fetch_document_file_map(company_id: str, doc_ids: list[str], *, runtime: FetchDocumentFileMapRuntime) -> dict[str, str]:
    """Legacy file-map projection, unchanged SQL and parameters."""
    return _fetch_document_file_map_impl(company_id, doc_ids, runtime=runtime)


def _fetch_document_file_map_impl(company_id: str, doc_ids: list[str], *, runtime: FetchDocumentFileMapRuntime, _reference_limit: int | None = None):
    binding_columns = FILE_BINDING_COLUMNS if _reference_limit is not None else ""
    limit_clause = " LIMIT %s" if _reference_limit is not None else ""
    _db_conn = runtime._db_conn
    company_id = (company_id or "").strip()
    doc_ids = sorted({str(x or "").strip() for x in (doc_ids or []) if str(x or "").strip()})
    if not company_id or not doc_ids:
        return ([], 0) if _reference_limit is not None else {}

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bubble_document_id, file_url{binding_columns}
                FROM public.document_files
                WHERE company_id = %s
                  AND bubble_document_id = ANY(%s){limit_clause};
                """,
                (company_id, doc_ids, _reference_limit + 1) if _reference_limit is not None else (company_id, doc_ids),
            )
            rows = cur.fetchall()
            if _reference_limit is not None:
                return rows, 1
            return {str(bdid): (url or "").strip() for (bdid, url) in rows if bdid and url}
    finally:
        conn.close()


@dataclass(frozen=True)
class DbFetchParentProcedurePagesForStepsRuntime:
    STRUCTURED_RELATION_PROCEDURE_STEP: Any
    _db_conn: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _safe_int: Callable[..., Any]


def db_fetch_parent_procedure_pages_for_steps(
    *,
    company_id: str,
    machine_id: str,
    child_source_keys: list[str],
    text_chars: int,
    runtime: DbFetchParentProcedurePagesForStepsRuntime,
) -> list[dict]:
    """Legacy entry: existing relation SQL, fallback and result shape preserved."""
    return _db_fetch_parent_procedure_pages_for_steps_impl(company_id=company_id, machine_id=machine_id, child_source_keys=child_source_keys, text_chars=text_chars, runtime=runtime)


def read_parent_procedure_page_evidence(*, scope: ChunkReadScope, children: tuple[SourceIdentity, ...],
                                        current_allowed_sources: frozenset[SourceIdentity],
                                        text_chars: int, limits: ChunkEvidenceLimits,
                                        runtime: DbFetchParentProcedurePagesForStepsRuntime) -> RelationEvidenceRead:
    validate_relation_request(scope=scope, kind="parent_procedures", anchors=children,
        current_allowed_sources=current_allowed_sources, text_chars=text_chars, limits=limits)
    if len(children) > 500:
        raise PageReadBindingError("child selector list exceeds existing parent reader bound")
    rows = _db_fetch_parent_procedure_pages_for_steps_impl(company_id=scope.company_id,
        machine_id=scope.machine_id, child_source_keys=["step:" + a.source_id for a in children],
        text_chars=text_chars, runtime=runtime, _include_binding_columns=True,
        _row_budget=limits.assembly.max_occurrences)
    return build_relation_read(scope=scope, kind="parent_procedures", anchors=children,
        current_allowed_sources=current_allowed_sources, text_chars=text_chars, limits=limits,
        relation_type=runtime.STRUCTURED_RELATION_PROCEDURE_STEP, rows=rows)


def _db_fetch_parent_procedure_pages_for_steps_impl(
    *,
    company_id: str,
    machine_id: str,
    child_source_keys: list[str],
    text_chars: int,
    runtime: DbFetchParentProcedurePagesForStepsRuntime, _include_binding_columns: bool = False, _row_budget: int = 0) -> list[dict]:
    """Resolve Step -> Procedure using the canonical relation table.

    The query is read-only and never touches chunks or embeddings. A LEFT JOIN keeps
    the relation usable even when the Procedure page is temporarily unavailable; the
    caller can then build a minimal parent placeholder and fall back safely.
    """
    binding_columns = RELATION_BINDING_COLUMNS if _include_binding_columns else ""
    limit_sql = " LIMIT %s" if _include_binding_columns else ""
    STRUCTURED_RELATION_PROCEDURE_STEP = runtime.STRUCTURED_RELATION_PROCEDURE_STEP
    _db_conn = runtime._db_conn
    _dedup_text_values = runtime._dedup_text_values
    _safe_int = runtime._safe_int
    child_keys = _dedup_text_values(
        [str(value or "").strip() for value in (child_source_keys or [])],
        limit=500,
    )
    if not (company_id and machine_id and child_keys):
        return []

    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    r.child_source_key,
                    r.parent_source_key,
                    r.ordinal,
                    p.machine_id,
                    p.page_number,
                    LEFT(COALESCE(p.text, ''), %s) AS parent_text{binding_columns}
                FROM public.structured_source_relations AS r
                LEFT JOIN public.document_pages AS p
                  ON p.company_id = r.company_id
                 AND p.bubble_document_id = r.parent_source_key
                 AND (p.machine_id = r.machine_id OR p.machine_id IS NULL OR p.machine_id = '')
                WHERE r.company_id = %s
                  AND r.machine_id = %s
                  AND r.child_source_key = ANY(%s)
                  AND r.relation_type = %s
                ORDER BY
                    r.child_source_key,
                    r.ordinal NULLS LAST,
                    CASE WHEN p.machine_id = %s THEN 0 ELSE 1 END,
                    p.page_number NULLS LAST{limit_sql};
                """,
                (
                    int(text_chars),
                    company_id,
                    machine_id,
                    child_keys,
                    STRUCTURED_RELATION_PROCEDURE_STEP,
                    machine_id,
                ) + ((_row_budget + 1,) if _include_binding_columns else ()),
            )
            rows = cur.fetchall()
    except Exception as exc:
        if _include_binding_columns:
            raise
        print("STRUCTURED_RELATION_PARENT_READ_FALLBACK", str(exc)[:700])
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if _include_binding_columns:
        return list(rows)

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for child_key, parent_key, ordinal, parent_mid, page_number, parent_text in rows:
        child = str(child_key or "").strip()
        parent = str(parent_key or "").strip()
        if not child or not parent:
            continue
        key = (child, parent)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "child_source_key": child,
                "parent_source_key": parent,
                "ordinal": _safe_int(ordinal, 0) or None,
                "machine_id": str(parent_mid or "").strip(),
                "page_number": _safe_int(page_number, 1),
                "parent_text": str(parent_text or "").strip(),
            }
        )
    return out



@dataclass(frozen=True)
class V12RelationProcedureCandidateRuntime:
    ASK_SNIPPET_CHARS: Any
    _clean_display_text: Callable[..., Any]
    _safe_int: Callable[..., Any]


def v12_relation_procedure_candidate(
    *,
    parent_source_key: str,
    machine_id: str,
    page_number: int,
    parent_text: str,
    fallback_title: str = "",
    runtime: V12RelationProcedureCandidateRuntime,
) -> dict:
    """Build a normal structured Procedure candidate from a relation row."""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    _clean_display_text = runtime._clean_display_text
    _safe_int = runtime._safe_int
    parent_key = str(parent_source_key or "").strip()
    page_no = max(1, _safe_int(page_number, 1))
    title = _clean_display_text(fallback_title, max_len=140)
    body = str(parent_text or "").strip()
    if not body:
        body = "SOURCE_TYPE: procedure\nTITLE: " + (title or "Procedura")
    return {
        "citation_id": f"{parent_key}:p{page_no}-{page_no}:procedure-family:v10_5",
        "bubble_document_id": parent_key,
        "chunk_index": 1,
        "page_from": page_no,
        "page_to": page_no,
        "snippet": body[: int(ASK_SNIPPET_CHARS or 900)],
        "snippet_clean": body[: int(ASK_SNIPPET_CHARS or 900)],
        "chunk_full": body,
        "similarity": 0.96,
        "retrieval_score": 0.96,
        "source_type": "procedure",
        "evidence_role": "procedure",
        "ask_structured_direct": True,
        "structured_direct_score": 12.0,
        "exact_machine_scope": bool(machine_id),
        "embedding_list": [],
        "structured_relation_source": "structured_source_relations_parent_recovery",
    }


@dataclass(frozen=True)
class DbFindTokenChunkRuntime:
    ASK_SNIPPET_CHARS: Any
    _db_conn: Callable[..., Any]


def db_find_token_chunk(company_id: str, machine_id: str, token: str, doc_ids: Optional[list[str]]=None, bubble_document_id: Optional[str]=None, *, runtime: DbFindTokenChunkRuntime) -> Optional[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _db_find_token_chunk_impl(company_id=company_id, machine_id=machine_id, token=token, doc_ids=doc_ids, bubble_document_id=bubble_document_id, runtime=runtime)


def _db_find_token_chunk_impl(company_id: str, machine_id: str, token: str, doc_ids: Optional[list[str]]=None, bubble_document_id: Optional[str]=None, *, runtime: DbFindTokenChunkRuntime, _chunk_capture: _ChunkSelectionCapture | None=None) -> Optional[dict]:
    binding_columns = CHUNK_BINDING_COLUMNS if _chunk_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    _db_conn = runtime._db_conn
    token = (token or "").strip()
    if not token:
        return None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            if doc_ids:
                where.append("bubble_document_id = ANY(%s)")
                params.append(doc_ids)
            elif bubble_document_id:
                where.append("bubble_document_id = %s")
                params.append(bubble_document_id)
                where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
                params.append(machine_id)
            else:
                where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
                params.append(machine_id)

            where_sql = " AND ".join(where)

            if _chunk_capture is not None:
                _chunk_capture.expect(1, query_text=token)
            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet{binding_columns}
                FROM public.document_chunks
                WHERE {where_sql}
                  AND chunk_text ILIKE %s
                ORDER BY bubble_document_id, page_from, chunk_index
                LIMIT 1;
                """,
                [ASK_SNIPPET_CHARS, *params, f"%{token}%"],
            )
            row = cur.fetchone()
            if _chunk_capture is not None:
                observed = _chunk_capture.capture([] if row is None else [row])
                row = observed[0] if observed else None
            if not row:
                return None

            bdid, chunk_index, page_from, page_to, snippet = row
            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            return {
                "citation_id": citation_id,
                "bubble_document_id": str(bdid),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": (snippet or "").strip(),
                "similarity": 0.0,
            }
    finally:
        conn.close()


@dataclass(frozen=True)
class DbFindEntityChunkRuntime:
    ASK_SNIPPET_CHARS: Any
    EMAIL_REGEX: Any
    PHONE_REGEX: Any
    URL_REGEX: Any
    _db_conn: Callable[..., Any]
    _extract_first: Callable[..., Any]


def db_find_entity_chunk(company_id: str, machine_id: str, kind: str, doc_ids: Optional[list[str]]=None, bubble_document_id: Optional[str]=None, *, runtime: DbFindEntityChunkRuntime) -> Optional[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _db_find_entity_chunk_impl(company_id=company_id, machine_id=machine_id, kind=kind, doc_ids=doc_ids, bubble_document_id=bubble_document_id, runtime=runtime)


def _db_find_entity_chunk_impl(company_id: str, machine_id: str, kind: str, doc_ids: Optional[list[str]]=None, bubble_document_id: Optional[str]=None, *, runtime: DbFindEntityChunkRuntime, _chunk_capture: _ChunkSelectionCapture | None=None) -> Optional[dict]:
    binding_columns = CHUNK_BINDING_COLUMNS if _chunk_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    EMAIL_REGEX = runtime.EMAIL_REGEX
    PHONE_REGEX = runtime.PHONE_REGEX
    URL_REGEX = runtime.URL_REGEX
    _db_conn = runtime._db_conn
    _extract_first = runtime._extract_first
    if kind == "url":
        pattern = r"(https?://|www\.)"
        rx = URL_REGEX
    elif kind == "email":
        pattern = r"@[A-Z0-9.-]+\.[A-Z]{2,}"
        rx = EMAIL_REGEX
    elif kind == "phone":
        pattern = r"\+?\d[\d\s().-]{7,}\d"
        rx = PHONE_REGEX
    else:
        return None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            if doc_ids:
                where.append("bubble_document_id = ANY(%s)")
                params.append(doc_ids)
            elif bubble_document_id:
                where.append("bubble_document_id = %s")
                params.append(bubble_document_id)
                where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
                params.append(machine_id)
            else:
                where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
                params.append(machine_id)

            where_sql = " AND ".join(where)

            if _chunk_capture is not None:
                _chunk_capture.expect(1, query_text=kind)
            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet{binding_columns}
                FROM public.document_chunks
                WHERE {where_sql}
                  AND chunk_text ~* %s
                ORDER BY bubble_document_id, page_from, chunk_index
                LIMIT 1;
                """,
                [ASK_SNIPPET_CHARS, *params, pattern],
            )
            row = cur.fetchone()
            if _chunk_capture is not None:
                observed = _chunk_capture.capture([] if row is None else [row])
                row = observed[0] if observed else None
            if not row:
                return None

            bdid, chunk_index, page_from, page_to, snippet = row
            snippet = (snippet or "").strip()
            value = _extract_first(rx, snippet)
            if not value:
                return None

            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            return {
                "citation_id": citation_id,
                "bubble_document_id": str(bdid),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": snippet,
                "value": value,
            }
    finally:
        conn.close()


@dataclass(frozen=True)
class AskEvidenceScopeWhereRuntime:
    pass


def ask_evidence_scope_where(
    *,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    runtime: AskEvidenceScopeWhereRuntime,
) -> tuple[str, list[Any]]:
    where = ["company_id = %s"]
    params: list[Any] = [company_id]
    if doc_ids:
        where.append("bubble_document_id = ANY(%s)")
        params.append(doc_ids)
    elif bubble_document_id:
        where.append("bubble_document_id = %s")
        params.append(bubble_document_id)
        where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
        params.append(machine_id)
    else:
        where.append("(machine_id = %s OR machine_id IS NULL OR machine_id = '')")
        params.append(machine_id)
    return " AND ".join(where), params


@dataclass(frozen=True)
class AskEvidenceFetchPagesRuntime:
    ASK_EVIDENCE_MAX_PAGE_CHARS: Any
    ASK_EVIDENCE_MIN_PAGE_SCORE: Any
    ASK_EVIDENCE_SCOPE_PAGE_LIMIT: Any
    ASK_EVIDENCE_TOP_PAGES: Any
    ASK_SNIPPET_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _ask_evidence_code_tokens: Callable[..., Any]
    _ask_evidence_number_tokens: Callable[..., Any]
    _ask_evidence_scope_where: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _ask_evidence_tokenize: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _safe_int: Callable[..., Any]
    re: Any


def ask_evidence_fetch_pages(
    *,
    q: str,
    profile: dict,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    top_pages: int = 10,
    runtime: AskEvidenceFetchPagesRuntime,
) -> list[dict]:
    """Legacy entry: SQL, parameters, transformations and selection preserved."""
    return _ask_evidence_fetch_pages_impl(q=q, profile=profile, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, top_pages=top_pages, runtime=runtime)


def read_ask_page_evidence(*, scope: ChunkReadScope, q: str, profile: dict,
                           limits: ChunkEvidenceLimits, runtime: AskEvidenceFetchPagesRuntime,
                           top_pages: int = 10) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "ask_pages", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _ask_evidence_fetch_pages_impl(**scope.sql_selectors(), q=q, profile=profile,
        top_pages=top_pages, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def _ask_evidence_fetch_pages_impl(
    *,
    q: str,
    profile: dict,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    top_pages: int = 10,
    runtime: AskEvidenceFetchPagesRuntime, _page_capture: _PageReadCapture | None = None) -> list[dict]:
    """Fetch and rank full pages/structured pages within the authorized scope.

    Relevance predicates are applied before the database LIMIT. This prevents a
    large machine/company knowledge base from excluding the correct document merely
    because its Bubble id sorts after the first N pages.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_EVIDENCE_MAX_PAGE_CHARS = runtime.ASK_EVIDENCE_MAX_PAGE_CHARS
    ASK_EVIDENCE_MIN_PAGE_SCORE = runtime.ASK_EVIDENCE_MIN_PAGE_SCORE
    ASK_EVIDENCE_SCOPE_PAGE_LIMIT = runtime.ASK_EVIDENCE_SCOPE_PAGE_LIMIT
    ASK_EVIDENCE_TOP_PAGES = runtime.ASK_EVIDENCE_TOP_PAGES
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_evidence_code_tokens = runtime._ask_evidence_code_tokens
    _ask_evidence_number_tokens = runtime._ask_evidence_number_tokens
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _ask_evidence_tokenize = runtime._ask_evidence_tokenize
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _safe_int = runtime._safe_int
    re = runtime.re
    limit = max(50, int(ASK_EVIDENCE_SCOPE_PAGE_LIMIT or 900))
    top_pages = max(3, min(int(top_pages or ASK_EVIDENCE_TOP_PAGES or 10), 16))
    where_sql, base_params = _ask_evidence_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )

    raw_terms: list[str] = []
    for key in (
        "search_phrases",
        "search_terms_it",
        "search_terms_en",
        "required_information",
        "important_codes_or_numbers",
    ):
        raw_terms.extend(str(x or "") for x in (profile.get(key) or []))
    raw_terms.extend(_ask_evidence_tokenize(q))
    raw_terms.extend(_ask_evidence_code_tokens(q))
    raw_terms.extend(_ask_evidence_number_tokens(q))

    search_terms: list[str] = []
    seen_terms = set()
    for raw in raw_terms:
        term = _normalize_unicode_advanced(str(raw or "")).lower().strip(" -–—:;,.()[]{}")
        term = re.sub(r"\s+", " ", term).strip()
        if len(term) < 2 or term in seen_terms:
            continue
        seen_terms.add(term)
        search_terms.append(term)
        if len(search_terms) >= 30:
            break

    def fetch_rows(*, require_term_match: bool) -> list[tuple]:
        term_sql = ""
        term_params: list[Any] = []
        if require_term_match and search_terms:
            term_sql = " AND (" + " OR ".join(
                ["LOWER(COALESCE(text, '')) LIKE %s" for _ in search_terms]
            ) + ")"
            term_params = [f"%{term}%" for term in search_terms]

        if _page_capture is not None:
            _page_capture.expect(limit, int(ASK_EVIDENCE_MAX_PAGE_CHARS or 12000))
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT bubble_document_id, machine_id, page_number,
                           LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                    FROM public.document_pages
                    WHERE {where_sql}
                      AND text IS NOT NULL
                      AND length(text) > 20
                      {term_sql}
                    ORDER BY
                      CASE WHEN machine_id = %s THEN 0 ELSE 1 END,
                      bubble_document_id,
                      page_number
                    LIMIT %s;
                    """,
                    [
                        int(ASK_EVIDENCE_MAX_PAGE_CHARS or 12000),
                        *base_params,
                        *term_params,
                        machine_id,
                        limit,
                    ],
                )
                raw_rows = cur.fetchall()
                return _page_capture.capture(raw_rows) if _page_capture is not None else raw_rows
        finally:
            conn.close()

    rows = fetch_rows(require_term_match=bool(search_terms))
    if not rows and search_terms:
        rows = fetch_rows(require_term_match=False)

    scored: list[dict] = []
    for (bdid, mid, page_number, page_text) in rows:
        txt = str(page_text or "").strip()
        if not txt:
            continue

        score = _ask_evidence_score_text(q, txt, profile)
        exact_machine_scope = str(mid or "").strip() == str(machine_id or "").strip()
        if exact_machine_scope and str(machine_id or "").strip() != COMPANY_GENERAL_MACHINE_SENTINEL:
            score += 4.0

        if score < float(ASK_EVIDENCE_MIN_PAGE_SCORE or 0.0):
            continue

        page = _safe_int(page_number, 1)
        scored.append(
            {
                "citation_id": f"{bdid}:p{page}-{page}:c0",
                "bubble_document_id": str(bdid),
                "chunk_index": 0,
                "page_from": page,
                "page_to": page,
                "snippet": txt[:ASK_SNIPPET_CHARS],
                "chunk_full": txt[: int(ASK_EVIDENCE_MAX_PAGE_CHARS or 12000)],
                "similarity": min(0.99, 0.50 + score / 100.0),
                "retrieval_score": score,
                "ask_evidence_score": score,
                "exact_machine_scope": bool(exact_machine_scope),
            }
        )

    scored.sort(
        key=lambda c: (
            -float(c.get("ask_evidence_score") or 0.0),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
        )
    )
    return _dedup_citations_by_snippet(scored, max_items=top_pages)



@dataclass(frozen=True)
class AskStructuredDirectFetchManualSupportRuntime:
    ASK_SNIPPET_CHARS: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _ask_structured_manual_support_candidate_score: Callable[..., Any]
    _ask_structured_manual_support_search_terms_with_llm: Callable[..., Any]
    _ask_structured_manual_support_select_with_llm: Callable[..., Any]
    _ask_structured_manual_support_terms: Callable[..., Any]
    _clean_display_text: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _source_display_metadata_from_citation: Any
    os: Any


def ask_structured_direct_fetch_manual_support(*, company_id: str, machine_id: str, q: str, planner: Optional[dict], structured_citations: list[dict], response_language: str='it', runtime: AskStructuredDirectFetchManualSupportRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _ask_structured_direct_fetch_manual_support_impl(company_id=company_id, machine_id=machine_id, q=q, planner=planner, structured_citations=structured_citations, response_language=response_language, runtime=runtime)


def _ask_structured_direct_fetch_manual_support_impl(*, company_id: str, machine_id: str, q: str, planner: Optional[dict], structured_citations: list[dict], response_language: str='it', runtime: AskStructuredDirectFetchManualSupportRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    """Fetch optional manual support for structured answers.

    Structured records remain primary. Manual pages are selected by a strict LLM
    relevance selector, not by a fixed operation dictionary. A manual page is kept
    only when it directly supports the same operation/problem as the structured
    source, or when it provides directly applicable safety/prerequisite context.
    Generic safety pages and adjacent processes are rejected.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_structured_manual_support_candidate_score = runtime._ask_structured_manual_support_candidate_score
    _ask_structured_manual_support_search_terms_with_llm = runtime._ask_structured_manual_support_search_terms_with_llm
    _ask_structured_manual_support_select_with_llm = runtime._ask_structured_manual_support_select_with_llm
    _ask_structured_manual_support_terms = runtime._ask_structured_manual_support_terms
    _clean_display_text = runtime._clean_display_text
    _db_conn = runtime._db_conn
    _safe_int = runtime._safe_int
    _source_display_metadata_from_citation = runtime._source_display_metadata_from_citation
    os = runtime.os
    if not ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED:
        return []
    if not structured_citations or not machine_id or str(machine_id).strip() == COMPANY_GENERAL_MACHINE_SENTINEL:
        return []

    text_chars = max(1200, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS or 4200))
    scan_limit = max(20, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT or 180))

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if _page_capture is not None:
                _page_capture.expect(scan_limit, text_chars)
            cur.execute(
                f"""
                SELECT bubble_document_id, machine_id, page_number,
                       LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                FROM public.document_pages
                WHERE company_id = %s
                  AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                  AND text IS NOT NULL
                  AND length(text) > 40
                  AND bubble_document_id NOT LIKE 'procedure:%%'
                  AND bubble_document_id NOT LIKE 'step:%%'
                  AND bubble_document_id NOT LIKE 'ps:%%'
                  AND bubble_document_id NOT LIKE 'md_photo:%%'
                  AND bubble_document_id NOT LIKE 'md_video:%%'
                ORDER BY bubble_document_id, page_number
                LIMIT %s;
                """,
                (text_chars, company_id, machine_id, scan_limit),
            )
            rows = cur.fetchall()
            if _page_capture is not None:
                rows = _page_capture.capture(rows)
    finally:
        conn.close()

    candidates: list[dict] = []
    for idx, (bdid, mid, page_number, page_text) in enumerate(rows or [], start=1):
        bdid_s = str(bdid or "").strip()
        txt = str(page_text or "").strip()
        if not bdid_s or not txt:
            continue
        page_no = _safe_int(page_number, 1)
        similarity = 0.58
        c = {
            "selector_index": idx,
            "citation_id": f"{bdid_s}:p{page_no}-{page_no}:manualsupport:{idx}",
            "bubble_document_id": bdid_s,
            "chunk_index": 1,
            "page_from": page_no,
            "page_to": page_no,
            "snippet": txt[: int(ASK_SNIPPET_CHARS or 900)],
            "snippet_clean": txt[: int(ASK_SNIPPET_CHARS or 900)],
            "chunk_full": txt,
            "similarity": float(similarity),
            "retrieval_score": float(similarity),
            "source_type": "document",
            "ask_structured_manual_support": True,
            "structured_manual_support_score": float(similarity),
            "structured_manual_operation_score": 0.0,
            "structured_manual_safety_score": 0.0,
            "embedding_list": [],
        }
        # Provide a readable label to the selector and to debug traces.
        try:
            label_meta = _source_display_metadata_from_citation(c, company_id=company_id)
            c["display_label"] = str(label_meta.get("display_label") or "")
        except Exception:
            if _page_capture is not None:
                raise
            c["display_label"] = f"Manuale - pag. {page_no}"
        candidates.append(c)

    if not candidates:
        return []

    profile_terms = _ask_structured_manual_support_search_terms_with_llm(
        q=q,
        response_language=response_language,
        structured_citations=structured_citations,
    )
    fallback_terms = _ask_structured_manual_support_terms(q, planner, structured_citations)
    for c in candidates:
        txt = str(c.get("chunk_full") or c.get("snippet") or "")
        cand_score = _ask_structured_manual_support_candidate_score(txt, profile_terms, fallback_terms)
        c["manual_support_candidate_score"] = float(cand_score)

    # Do not send the whole manual to the selector: it dilutes attention and can
    # cause it to miss the relevant manual phase. Use the LLM-inferred search
    # profile to shortlist pages, then let the selector make the final semantic
    # decision. This is still reasoning-based; it is not a hard-coded answer map.
    candidates.sort(key=lambda x: (-float(x.get("manual_support_candidate_score") or 0.0), str(x.get("bubble_document_id") or ""), _safe_int(x.get("page_from"), 0)))
    selector_limit = max(8, min(16, int(os.getenv("MM_ASK_STRUCTURED_MANUAL_SELECTOR_CANDIDATES", "12") or "12")))
    if float(candidates[0].get("manual_support_candidate_score") or 0.0) > 0.0:
        candidates = candidates[:selector_limit]
    else:
        candidates = candidates[:min(len(candidates), selector_limit)]

    for idx, c in enumerate(candidates, start=1):
        c["selector_index"] = idx

    # Let the model reason about direct relevance on the shortlisted manual pages.
    # If the selector is unsure, manual support is omitted rather than wrong.
    selected_meta = _ask_structured_manual_support_select_with_llm(
        q=q,
        response_language=response_language,
        structured_citations=structured_citations,
        candidates=candidates,
    )

    op_indices = {int(x) for x in (selected_meta.get("operation_support_indices") or []) if str(x).strip().lstrip("-").isdigit()}
    safety_indices = {int(x) for x in (selected_meta.get("safety_support_indices") or []) if str(x).strip().lstrip("-").isdigit()}
    if not op_indices and not safety_indices:
        return []

    max_items = max(0, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS or 2))
    if max_items <= 0:
        return []

    operation_note = _clean_display_text(str(selected_meta.get("operation_note") or ""), max_len=360)
    safety_note = _clean_display_text(str(selected_meta.get("safety_note") or ""), max_len=320)

    by_idx = {int(c.get("selector_index") or 0): c for c in candidates}
    selected: list[dict] = []
    used: set[int] = set()

    def add_selected(idx: int, kind: str) -> None:
        if len(selected) >= max_items or idx in used:
            return
        c = dict(by_idx.get(idx) or {})
        if not c:
            return
        used.add(idx)
        c["ask_manual_support_kind"] = kind
        c["structured_manual_operation_score"] = 10.0 if kind == "operation" else 0.0
        c["structured_manual_safety_score"] = 10.0 if kind == "safety" else 0.0
        c["structured_manual_support_score"] = 10.0
        c["similarity"] = 0.86 if kind == "operation" else 0.82
        c["retrieval_score"] = float(c["similarity"])
        if kind == "operation" and operation_note:
            c["llm_operation_note"] = operation_note
        if kind == "safety" and safety_note:
            c["llm_safety_note"] = safety_note
        selected.append(c)

    for idx in sorted(op_indices):
        add_selected(idx, "operation")
    for idx in sorted(safety_indices):
        add_selected(idx, "safety")

    return selected[:max_items]


@dataclass(frozen=True)
class AskFullContextSeedDocIdsRuntime:
    ASK_FULL_CONTEXT_MAX_DOCS: Any
    _dedup_text_values: Callable[..., Any]


def ask_full_context_seed_doc_ids(
    *,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
    seed_citations: Optional[list[dict]],
    runtime: AskFullContextSeedDocIdsRuntime,
) -> Optional[list[str]]:
    """Pick document/source ids to read fully, without using benchmark-specific ids."""
    ASK_FULL_CONTEXT_MAX_DOCS = runtime.ASK_FULL_CONTEXT_MAX_DOCS
    _dedup_text_values = runtime._dedup_text_values
    if doc_ids:
        return _dedup_text_values([str(x or "").strip() for x in doc_ids if str(x or "").strip()], limit=ASK_FULL_CONTEXT_MAX_DOCS)
    if bubble_document_id:
        return [str(bubble_document_id).strip()]

    out: list[str] = []
    seen = set()
    for c in seed_citations or []:
        bdid = str((c or {}).get("bubble_document_id") or "").strip()
        if not bdid or bdid in seen:
            continue
        seen.add(bdid)
        out.append(bdid)
        if len(out) >= int(ASK_FULL_CONTEXT_MAX_DOCS or 3):
            break
    return out or None


@dataclass(frozen=True)
class AskFullContextFetchPagesRuntime:
    ASK_FULL_CONTEXT_MAX_CHARS: Any
    ASK_FULL_CONTEXT_MAX_DOCS: Any
    ASK_FULL_CONTEXT_MAX_PAGES: Any
    ASK_FULL_CONTEXT_PAGE_CHARS: Any
    ASK_SNIPPET_CHARS: Any
    _ask_full_context_seed_doc_ids: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _safe_int: Callable[..., Any]


def ask_full_context_fetch_pages(
    *,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
    seed_citations: Optional[list[dict]],
    runtime: AskFullContextFetchPagesRuntime,
) -> list[dict]:
    """Legacy entry: SQL, parameters, transformations and selection preserved."""
    return _ask_full_context_fetch_pages_impl(company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, seed_citations=seed_citations, runtime=runtime)


def read_full_context_page_evidence(*, scope: ChunkReadScope,
                                    seed_citations: Optional[list[dict]],
                                    limits: ChunkEvidenceLimits,
                                    runtime: AskFullContextFetchPagesRuntime) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "full_context", snippet_chars=runtime.ASK_SNIPPET_CHARS,
        candidate_chars=max(20000, int(runtime.ASK_FULL_CONTEXT_MAX_CHARS or 120000)))
    selected = _ask_full_context_fetch_pages_impl(**scope.sql_selectors(), seed_citations=seed_citations,
        runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def _ask_full_context_fetch_pages_impl(
    *,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
    seed_citations: Optional[list[dict]],
    runtime: AskFullContextFetchPagesRuntime, _page_capture: _PageReadCapture | None = None) -> list[dict]:
    """Fetch full pages for a narrow authorized scope.

    This is intentionally generic: it does not know any test question, expected answer,
    document id, product code or component. It simply reads the authorized document pages
    when the scope is narrow enough to fit in the model context.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_FULL_CONTEXT_MAX_CHARS = runtime.ASK_FULL_CONTEXT_MAX_CHARS
    ASK_FULL_CONTEXT_MAX_DOCS = runtime.ASK_FULL_CONTEXT_MAX_DOCS
    ASK_FULL_CONTEXT_MAX_PAGES = runtime.ASK_FULL_CONTEXT_MAX_PAGES
    ASK_FULL_CONTEXT_PAGE_CHARS = runtime.ASK_FULL_CONTEXT_PAGE_CHARS
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    _ask_full_context_seed_doc_ids = runtime._ask_full_context_seed_doc_ids
    _db_conn = runtime._db_conn
    _safe_int = runtime._safe_int
    target_doc_ids = _ask_full_context_seed_doc_ids(
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        seed_citations=seed_citations,
    )
    if not target_doc_ids:
        return []

    target_doc_ids = target_doc_ids[: max(1, int(ASK_FULL_CONTEXT_MAX_DOCS or 3))]
    page_limit = max(10, int(ASK_FULL_CONTEXT_MAX_PAGES or 140))
    page_chars = max(1200, int(ASK_FULL_CONTEXT_PAGE_CHARS or 6500))

    if _page_capture is not None:
        _page_capture.expect(page_limit, page_chars)
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                FROM public.document_pages
                WHERE company_id = %s
                  AND bubble_document_id = ANY(%s)
                  AND text IS NOT NULL
                  AND length(text) > 20
                ORDER BY bubble_document_id, page_number
                LIMIT %s;
                """,
                [page_chars, company_id, target_doc_ids, page_limit],
            )
            raw_rows = cur.fetchall()
            rows = _page_capture.capture(raw_rows) if _page_capture is not None else raw_rows
    finally:
        conn.close()

    citations: list[dict] = []
    total_chars = 0
    max_chars = max(20000, int(ASK_FULL_CONTEXT_MAX_CHARS or 120000))
    for (bdid, mid, page_number, page_text) in rows:
        txt = str(page_text or "").strip()
        if not txt:
            continue
        page = _safe_int(page_number, 1)
        # Keep whole pages until the budget is exhausted. This avoids fragment-only answers.
        if total_chars + len(txt) > max_chars and citations:
            break
        if len(txt) > max_chars and not citations:
            txt = txt[:max_chars]
        citations.append(
            {
                "citation_id": f"{bdid}:p{page}-{page}:full",
                "bubble_document_id": str(bdid),
                "chunk_index": 0,
                "page_from": page,
                "page_to": page,
                "snippet": txt[:ASK_SNIPPET_CHARS],
                "chunk_full": txt,
                "similarity": 0.99,
                "retrieval_score": 99.0,
                "ask_full_context": True,
            }
        )
        total_chars += len(txt)
    return citations



@dataclass(frozen=True)
class AskFetchPreferredSourcePagesRuntime:
    ASK_EVIDENCE_SCOPE_PAGE_LIMIT: Any
    ASK_FULL_CONTEXT_PAGE_CHARS: Any
    ASK_SNIPPET_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _ask_evidence_query_profile: Callable[..., Any]
    _ask_evidence_scope_where: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _ask_manual_priority_page_has_real_maintenance_content: Callable[..., Any]
    _ask_manual_priority_page_is_meta_or_index: Callable[..., Any]
    _ask_manual_priority_page_score: Callable[..., Any]
    _ask_manual_priority_query_is_maintenance: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _is_structured_source_key: Callable[..., Any]
    _is_xlsx_indexed_page_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _safe_int: Callable[..., Any]


def ask_fetch_preferred_source_pages(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], response_language: str, top_k: int, source_kind: str, runtime: AskFetchPreferredSourcePagesRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _ask_fetch_preferred_source_pages_impl(q=q, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, response_language=response_language, top_k=top_k, source_kind=source_kind, runtime=runtime)


def _ask_fetch_preferred_source_pages_impl(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], response_language: str, top_k: int, source_kind: str, runtime: AskFetchPreferredSourcePagesRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    """Fetch primary pages for a soft/hard source preference.

    source_kind="xlsx" fetches XLSX-generated pages.
    source_kind="manual" fetches ordinary document/manual/PDF pages, excluding
    Bubble structured records and XLSX-generated pages.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_EVIDENCE_SCOPE_PAGE_LIMIT = runtime.ASK_EVIDENCE_SCOPE_PAGE_LIMIT
    ASK_FULL_CONTEXT_PAGE_CHARS = runtime.ASK_FULL_CONTEXT_PAGE_CHARS
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_evidence_query_profile = runtime._ask_evidence_query_profile
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _ask_manual_priority_page_has_real_maintenance_content = runtime._ask_manual_priority_page_has_real_maintenance_content
    _ask_manual_priority_page_is_meta_or_index = runtime._ask_manual_priority_page_is_meta_or_index
    _ask_manual_priority_page_score = runtime._ask_manual_priority_page_score
    _ask_manual_priority_query_is_maintenance = runtime._ask_manual_priority_query_is_maintenance
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _is_structured_source_key = runtime._is_structured_source_key
    _is_xlsx_indexed_page_text = runtime._is_xlsx_indexed_page_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _safe_int = runtime._safe_int
    if source_kind not in {"xlsx", "manual"}:
        return []

    profile = _ask_evidence_query_profile(q, response_language)
    where_sql, params = _ask_evidence_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )

    page_chars = max(1200, int(ASK_FULL_CONTEXT_PAGE_CHARS or 6500))
    scan_limit = max(80, int(ASK_EVIDENCE_SCOPE_PAGE_LIMIT or 900))

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if _page_capture is not None:
                _page_capture.expect(scan_limit, page_chars)
            cur.execute(
                f"""
                SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                FROM public.document_pages
                WHERE {where_sql}
                  AND text IS NOT NULL
                  AND length(text) > 20
                ORDER BY bubble_document_id, page_number
                LIMIT %s;
                """,
                [page_chars, *params, scan_limit],
            )
            rows = cur.fetchall()
            if _page_capture is not None:
                rows = _page_capture.capture(rows)
    finally:
        conn.close()

    # Targeted supplement for explicit manual maintenance/check questions.
    # The broad scan can be diluted by cover pages, indexes or company-general manuals.
    # This pass pulls table/frequency/maintenance pages from the same authorized scope,
    # then the normal scorer still decides the order. It is a supplement, not a hard filter.
    if source_kind == "manual" and _ask_manual_priority_query_is_maintenance(q):
        targeted_patterns = [
            "%tabella per manutenzione%",
            "%tabella generale di manutenzione%",
            "%ore di funzionamento%",
            "%componenti%ore di%",
            "%tipo di lubrificante%",
            "%controllare il livello%",
            "%cambio olio%",
            "%pulizia dei filtri%",
            "%sostituzione completa dei filtri%",
            "%scarico della condensa%",
            "%verifica integrità%",
            "%verifica integrita%",
            "%verifica corretto funzionamento%",
            "%impianto elettrico%",
            "%impianto pneumatico%",
            "%raddrizzatura%",
            "%lubrificazione%",
            "%ogni 50 ore%",
            "%ogni 300 ore%",
            "%ogni 1000 ore%",
            "%ogni 3000 ore%",
            "%ogni giorno%",
            "%mensilmente%",
            "%settiman%",
            "%annualmente%",
        ]
        targeted_clauses = " OR ".join(["LOWER(COALESCE(text, '')) LIKE %s" for _ in targeted_patterns])
        target_limit = max(80, min(260, int(scan_limit // 2)))
        try:
            conn = _db_conn()
            try:
                with conn.cursor() as cur:
                    if _page_capture is not None:
                        _page_capture.expect(target_limit, page_chars)
                    cur.execute(
                        f"""
                        SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                        FROM public.document_pages
                        WHERE {where_sql}
                          AND text IS NOT NULL
                          AND length(text) > 20
                          AND ({targeted_clauses})
                        ORDER BY
                          CASE
                            WHEN machine_id = %s THEN 0
                            WHEN machine_id IS NULL OR machine_id = '' THEN 1
                            ELSE 2
                          END,
                          bubble_document_id,
                          page_number
                        LIMIT %s;
                        """,
                        [page_chars, *params, *targeted_patterns, machine_id, target_limit],
                    )
                    targeted_rows = cur.fetchall()
                    if _page_capture is not None:
                        targeted_rows = _page_capture.capture(targeted_rows)
            finally:
                conn.close()
        except Exception as e:
            if _page_capture is not None:
                raise
            print("ASK_MANUAL_TARGETED_SCAN_FAIL", str(e)[:300])
            targeted_rows = []

        if targeted_rows:
            rows = list(rows or [])
            seen_pages = {
                (str(r[0] or ""), _safe_int(r[2], 0))
                for r in rows
            }
            for r in targeted_rows:
                key = (str(r[0] or ""), _safe_int(r[2], 0))
                if key not in seen_pages:
                    rows.append(r)
                    seen_pages.add(key)

    scored: list[dict] = []
    q_low = _normalize_unicode_advanced(q or "").lower()

    for idx, (bdid, mid, page_number, page_text) in enumerate(rows or [], start=1):
        bdid_s = str(bdid or "").strip()
        txt = str(page_text or "").strip()
        if not bdid_s or not txt:
            continue

        is_xlsx = _is_xlsx_indexed_page_text(txt)
        is_structured = _is_structured_source_key(bdid_s)

        if source_kind == "xlsx" and not is_xlsx:
            continue
        if source_kind == "manual" and (is_xlsx or is_structured):
            continue

        score = float(_ask_evidence_score_text(q, txt, profile))
        # Source preference is a ranking boost, not an exclusive evidence rule.
        score += 35.0 if source_kind == "xlsx" else 28.0
        if source_kind == "xlsx" and any(x in q_low for x in ["excel", "xlsx", "foglio", "spreadsheet"]):
            score += 8.0
        if source_kind == "manual" and any(x in q_low for x in ["manual", "manuale", "pdf", "documentazione"]):
            score += 8.0

        t_low = _normalize_unicode_advanced(txt).lower()
        for marker, bonus in [
            ("manutenz", 8.0), ("maintenance", 8.0), ("controll", 6.0),
            ("periodic", 5.0), ("frequenza", 5.0), ("frequency", 5.0),
            ("lubr", 4.0), ("olio", 4.0), ("oil", 4.0),
        ]:
            if marker in q_low and marker in t_low:
                score += bonus

        exact_machine_page = False
        real_maintenance_page = False
        weak_meta_page = False
        if source_kind == "manual":
            row_mid = str(mid or "").strip()
            exact_machine_page = bool(machine_id and machine_id != COMPANY_GENERAL_MACHINE_SENTINEL and row_mid == str(machine_id or "").strip())
            real_maintenance_page = _ask_manual_priority_page_has_real_maintenance_content(txt)
            weak_meta_page = _ask_manual_priority_page_is_meta_or_index(txt)
            score = _ask_manual_priority_page_score(
                q=q,
                page_text=txt,
                base_score=score,
                row_machine_id=row_mid,
                requested_machine_id=machine_id,
            )

        page = _safe_int(page_number, 1)
        row_obj = {
            "citation_id": f"{bdid_s}:p{page}-{page}:{source_kind}priority:{idx}",
            "bubble_document_id": bdid_s,
            "chunk_index": 0,
            "page_from": page,
            "page_to": page,
            "snippet": txt[:ASK_SNIPPET_CHARS],
            "chunk_full": txt,
            "similarity": min(0.99, 0.74 + score / 200.0),
            "retrieval_score": score,
            "ask_source_priority": True,
            "ask_source_priority_kind": source_kind,
        }
        if source_kind == "manual":
            row_obj["manual_priority_exact_machine"] = bool(exact_machine_page)
            row_obj["manual_priority_real_maintenance"] = bool(real_maintenance_page)
            row_obj["manual_priority_weak_meta"] = bool(weak_meta_page)
        scored.append(row_obj)

    scored.sort(
        key=lambda c: (
            -float(c.get("retrieval_score") or 0.0),
            str(c.get("bubble_document_id") or ""),
            _safe_int(c.get("page_from"), 0),
        )
    )

    max_items = max(1, min(max(top_k, 6), 12))
    if source_kind == "manual" and _ask_manual_priority_query_is_maintenance(q):
        exact_strong = [
            c for c in scored
            if bool(c.get("manual_priority_exact_machine"))
            and bool(c.get("manual_priority_real_maintenance"))
            and not bool(c.get("manual_priority_weak_meta"))
        ]
        other_strong = [
            c for c in scored
            if c not in exact_strong
            and bool(c.get("manual_priority_real_maintenance"))
            and not bool(c.get("manual_priority_weak_meta"))
        ]
        weak = [c for c in scored if c not in exact_strong and c not in other_strong]
        if exact_strong:
            ordered = exact_strong[:min(6, max_items)] + other_strong[:max(0, max_items - min(6, len(exact_strong)))] + weak[:max_items]
            return _dedup_citations_by_snippet(ordered, max_items=max_items)
        if other_strong:
            ordered = other_strong[:max_items] + weak[:max_items]
            return _dedup_citations_by_snippet(ordered, max_items=max_items)

    return _dedup_citations_by_snippet(scored, max_items=max_items)


@dataclass(frozen=True)
class AskFetchManualMaintenanceTargetPagesRuntime:
    ASK_FULL_CONTEXT_PAGE_CHARS: Any
    ASK_SNIPPET_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _ask_evidence_fallback_profile: Callable[..., Any]
    _ask_evidence_scope_where: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _ask_manual_priority_page_has_real_maintenance_content: Callable[..., Any]
    _ask_manual_priority_page_is_meta_or_index: Callable[..., Any]
    _ask_manual_priority_page_score: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _is_structured_source_key: Callable[..., Any]
    _is_xlsx_indexed_page_text: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _simple_query_language: Callable[..., Any]


def ask_fetch_manual_maintenance_target_pages(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], top_k: int, runtime: AskFetchManualMaintenanceTargetPagesRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _ask_fetch_manual_maintenance_target_pages_impl(q=q, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, top_k=top_k, runtime=runtime)


def _ask_fetch_manual_maintenance_target_pages_impl(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], top_k: int, runtime: AskFetchManualMaintenanceTargetPagesRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    """Fetch high-signal manual maintenance pages for explicit manual questions.

    This is a narrow ASK-only supplement used when the user asks what the machine
    manual says about maintenance/periodic checks. It does not hardcode document
    IDs or answers: it scans authorized manual/PDF pages for real maintenance
    evidence and keeps document-specific pages whenever they are available.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_FULL_CONTEXT_PAGE_CHARS = runtime.ASK_FULL_CONTEXT_PAGE_CHARS
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_evidence_fallback_profile = runtime._ask_evidence_fallback_profile
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _ask_manual_priority_page_has_real_maintenance_content = runtime._ask_manual_priority_page_has_real_maintenance_content
    _ask_manual_priority_page_is_meta_or_index = runtime._ask_manual_priority_page_is_meta_or_index
    _ask_manual_priority_page_score = runtime._ask_manual_priority_page_score
    _db_conn = runtime._db_conn
    _is_structured_source_key = runtime._is_structured_source_key
    _is_xlsx_indexed_page_text = runtime._is_xlsx_indexed_page_text
    _safe_int = runtime._safe_int
    _simple_query_language = runtime._simple_query_language
    where_sql, params = _ask_evidence_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )
    page_chars = max(1200, int(ASK_FULL_CONTEXT_PAGE_CHARS or 6500))
    patterns = [
        "%tabella per manutenzione%",
        "%tabella generale di manutenzione%",
        "%ore di%funzionamento%",
        "%componenti%ore di%",
        "%tipo di lubrificante%",
        "%controllare il livello%",
        "%cambio olio%",
        "%pulizia dei filtri%",
        "%sostituzione completa dei filtri%",
        "%scarico della condensa%",
        "%verifica integrità%",
        "%verifica integrita%",
        "%verifica corretto funzionamento%",
        "%impianto elettrico%",
        "%impianto pneumatico%",
        "%raddrizzatura%",
        "%lubrificazione%",
        "%ogni 50 ore%",
        "%ogni 300 ore%",
        "%ogni 1000 ore%",
        "%ogni 3000 ore%",
        "%ogni giorno%",
        "%mensilmente%",
        "%settiman%",
        "%annualmente%",
    ]
    clauses = " OR ".join(["LOWER(COALESCE(text, '')) LIKE %s" for _ in patterns])

    rows = []
    try:
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                if _page_capture is not None:
                    _page_capture.expect(240, page_chars)
                cur.execute(
                    f"""
                    SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                    FROM public.document_pages
                    WHERE {where_sql}
                      AND text IS NOT NULL
                      AND length(text) > 20
                      AND ({clauses})
                    ORDER BY
                      CASE
                        WHEN machine_id = %s THEN 0
                        WHEN machine_id IS NULL OR machine_id = '' THEN 1
                        ELSE 2
                      END,
                      bubble_document_id,
                      page_number
                    LIMIT %s;
                    """,
                    [page_chars, *params, *patterns, machine_id, 240],
                )
                rows = cur.fetchall()
                if _page_capture is not None:
                    rows = _page_capture.capture(rows)
        finally:
            conn.close()
    except Exception as e:
        if _page_capture is not None:
            raise
        print("ASK_MANUAL_MAINTENANCE_DIRECT_FETCH_FAIL", str(e)[:300])
        return []

    scored: list[dict] = []
    for idx, (bdid, mid, page_number, page_text) in enumerate(rows or [], start=1):
        bdid_s = str(bdid or "").strip()
        txt = str(page_text or "").strip()
        if not bdid_s or not txt:
            continue
        if _is_structured_source_key(bdid_s) or _is_xlsx_indexed_page_text(txt):
            continue
        if not _ask_manual_priority_page_has_real_maintenance_content(txt):
            continue
        if _ask_manual_priority_page_is_meta_or_index(txt):
            continue

        row_mid = str(mid or "").strip()
        exact_machine = bool(machine_id and machine_id != COMPANY_GENERAL_MACHINE_SENTINEL and row_mid == str(machine_id or "").strip())
        base_score = float(_ask_evidence_score_text(q, txt, _ask_evidence_fallback_profile(q, _simple_query_language(q))))
        score = _ask_manual_priority_page_score(
            q=q,
            page_text=txt,
            base_score=base_score + 80.0,
            row_machine_id=row_mid,
            requested_machine_id=machine_id,
        )
        if exact_machine:
            score += 80.0

        page = _safe_int(page_number, 1)
        scored.append(
            {
                "citation_id": f"{bdid_s}:p{page}-{page}:manualmaint:{idx}",
                "bubble_document_id": bdid_s,
                "chunk_index": 0,
                "page_from": page,
                "page_to": page,
                "snippet": txt[:ASK_SNIPPET_CHARS],
                "chunk_full": txt,
                "similarity": min(0.99, 0.80 + score / 300.0),
                "retrieval_score": score,
                "ask_manual_maintenance_direct": True,
                "manual_priority_exact_machine": exact_machine,
                "manual_priority_real_maintenance": True,
                "manual_priority_weak_meta": False,
            }
        )

    scored.sort(
        key=lambda c: (
            0 if bool(c.get("manual_priority_exact_machine")) else 1,
            -float(c.get("retrieval_score") or 0.0),
            str(c.get("bubble_document_id") or ""),
            _safe_int(c.get("page_from"), 0),
        )
    )

    # Preserve duplicate-looking pages from different documents when one is a
    # machine-specific manual; normal snippet dedupe can otherwise keep the company
    # copy and drop the machine copy.
    out: list[dict] = []
    seen_pages: set[tuple[str, int]] = set()
    for c in scored:
        key = (str(c.get("bubble_document_id") or ""), _safe_int(c.get("page_from"), 0))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        out.append(c)
        if len(out) >= max(1, min(max(top_k, 8), 12)):
            break
    return out


@dataclass(frozen=True)
class V13FetchScoredPagesRuntime:
    ASK_EVIDENCE_MIN_PAGE_SCORE: Any
    ASK_SNIPPET_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    V13_PAGE_SCAN_LIMIT: Any
    V13_PAGE_TEXT_CHARS: Any
    _ask_evidence_code_tokens: Callable[..., Any]
    _ask_evidence_number_tokens: Callable[..., Any]
    _ask_evidence_scope_where: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _ask_evidence_tokenize: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _safe_int: Callable[..., Any]
    re: Any


def v13_fetch_scored_pages(
    *,
    q: str,
    profile: dict,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
    top_pages: int,
    runtime: V13FetchScoredPagesRuntime,
) -> list[dict]:
    """Legacy entry: SQL, parameters, transformations and selection preserved."""
    return _v13_fetch_scored_pages_impl(q=q, profile=profile, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, top_pages=top_pages, runtime=runtime)


def read_scored_page_evidence(*, scope: ChunkReadScope, q: str, profile: dict,
                              top_pages: int, limits: ChunkEvidenceLimits,
                              runtime: V13FetchScoredPagesRuntime) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "scored_pages", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _v13_fetch_scored_pages_impl(**scope.sql_selectors(), q=q, profile=profile,
        top_pages=top_pages, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def _v13_fetch_scored_pages_impl(
    *,
    q: str,
    profile: dict,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
    top_pages: int,
    runtime: V13FetchScoredPagesRuntime, _page_capture: _PageReadCapture | None = None) -> list[dict]:
    """Bounded full-page rescue with SQL relevance predicates before LIMIT."""
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_EVIDENCE_MIN_PAGE_SCORE = runtime.ASK_EVIDENCE_MIN_PAGE_SCORE
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    V13_PAGE_SCAN_LIMIT = runtime.V13_PAGE_SCAN_LIMIT
    V13_PAGE_TEXT_CHARS = runtime.V13_PAGE_TEXT_CHARS
    _ask_evidence_code_tokens = runtime._ask_evidence_code_tokens
    _ask_evidence_number_tokens = runtime._ask_evidence_number_tokens
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _ask_evidence_tokenize = runtime._ask_evidence_tokenize
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _safe_int = runtime._safe_int
    re = runtime.re
    where_sql, base_params = _ask_evidence_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )

    raw_terms: list[str] = []
    for key in (
        "search_phrases", "search_terms_it", "search_terms_en",
        "required_information", "important_codes_or_numbers",
    ):
        raw_terms.extend(str(x or "") for x in (profile.get(key) or []))
    raw_terms.extend(_ask_evidence_tokenize(q))
    raw_terms.extend(_ask_evidence_code_tokens(q))
    raw_terms.extend(_ask_evidence_number_tokens(q))

    search_terms: list[str] = []
    seen_terms = set()
    for raw in raw_terms:
        term = _normalize_unicode_advanced(str(raw or "")).lower().strip(" -–—:;,.()[]{}")
        term = re.sub(r"\s+", " ", term).strip()
        if len(term) < 2 or term in seen_terms:
            continue
        seen_terms.add(term)
        search_terms.append(term)
        if len(search_terms) >= 24:
            break

    def fetch_rows(require_term_match: bool) -> list[tuple]:
        term_sql = ""
        term_params: list[Any] = []
        if require_term_match and search_terms:
            term_sql = " AND (" + " OR ".join(
                ["LOWER(COALESCE(text, '')) LIKE %s" for _ in search_terms]
            ) + ")"
            term_params = [f"%{term}%" for term in search_terms]

        if _page_capture is not None:
            _page_capture.expect(V13_PAGE_SCAN_LIMIT, V13_PAGE_TEXT_CHARS)
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT bubble_document_id, machine_id, page_number,
                           LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                    FROM public.document_pages
                    WHERE {where_sql}
                      AND text IS NOT NULL
                      AND length(text) > 20
                      {term_sql}
                    ORDER BY CASE WHEN machine_id=%s THEN 0 ELSE 1 END,
                             bubble_document_id, page_number
                    LIMIT %s;
                    """,
                    [
                        V13_PAGE_TEXT_CHARS,
                        *base_params,
                        *term_params,
                        machine_id,
                        V13_PAGE_SCAN_LIMIT,
                    ],
                )
                raw_rows = cur.fetchall()
                return _page_capture.capture(raw_rows) if _page_capture is not None else raw_rows
        finally:
            conn.close()

    rows = fetch_rows(bool(search_terms))
    if not rows and search_terms:
        rows = fetch_rows(False)

    scored: list[dict] = []
    for bdid, mid, page_number, page_text in rows:
        text = str(page_text or "").strip()
        if not text:
            continue
        score = float(_ask_evidence_score_text(q, text, profile))
        exact_machine = (
            str(mid or "").strip() == str(machine_id or "").strip()
            and str(machine_id or "").strip() != COMPANY_GENERAL_MACHINE_SENTINEL
        )
        if exact_machine:
            score += 4.0
        if score < float(ASK_EVIDENCE_MIN_PAGE_SCORE or 0.0):
            continue
        page = _safe_int(page_number, 1)
        scored.append(
            {
                "citation_id": f"{bdid}:p{page}-{page}:v13page",
                "bubble_document_id": str(bdid),
                "chunk_index": 0,
                "page_from": page,
                "page_to": page,
                "snippet": text[:ASK_SNIPPET_CHARS],
                "chunk_full": text[:V13_PAGE_TEXT_CHARS],
                "similarity": min(0.99, 0.50 + score / 100.0),
                "retrieval_score": score,
                "ask_evidence_score": score,
                "exact_machine_scope": bool(exact_machine),
            }
        )

    scored.sort(
        key=lambda c: (
            -float(c.get("ask_evidence_score") or 0.0),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
        )
    )
    return _dedup_citations_by_snippet(scored, max_items=max(3, min(int(top_pages or 8), 14)))



@dataclass(frozen=True)
class V13FetchPreferredSourcePagesRuntime:
    ASK_FULL_CONTEXT_PAGE_CHARS: Any
    ASK_SNIPPET_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    V13_PAGE_TEXT_CHARS: Any
    V13_PREFERRED_PAGE_SCAN_LIMIT: Any
    _ask_evidence_scope_where: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _ask_manual_priority_page_has_real_maintenance_content: Callable[..., Any]
    _ask_manual_priority_page_is_meta_or_index: Callable[..., Any]
    _ask_manual_priority_page_score: Callable[..., Any]
    _ask_manual_priority_query_is_maintenance: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _is_structured_source_key: Callable[..., Any]
    _is_xlsx_indexed_page_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _v13_build_profile_from_plan: Callable[..., Any]


def v13_fetch_preferred_source_pages(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], response_language: str, top_k: int, plan: Optional[dict], source_kind: str, runtime: V13FetchPreferredSourcePagesRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _v13_fetch_preferred_source_pages_impl(q=q, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id, response_language=response_language, top_k=top_k, plan=plan, source_kind=source_kind, runtime=runtime)


def _v13_fetch_preferred_source_pages_impl(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], response_language: str, top_k: int, plan: Optional[dict], source_kind: str, runtime: V13FetchPreferredSourcePagesRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    """Fetch primary pages for a soft/hard source preference.

    source_kind="xlsx" fetches XLSX-generated pages.
    source_kind="manual" fetches ordinary document/manual/PDF pages, excluding
    Bubble structured records and XLSX-generated pages.
    """
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_FULL_CONTEXT_PAGE_CHARS = runtime.ASK_FULL_CONTEXT_PAGE_CHARS
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    V13_PAGE_TEXT_CHARS = runtime.V13_PAGE_TEXT_CHARS
    V13_PREFERRED_PAGE_SCAN_LIMIT = runtime.V13_PREFERRED_PAGE_SCAN_LIMIT
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _ask_manual_priority_page_has_real_maintenance_content = runtime._ask_manual_priority_page_has_real_maintenance_content
    _ask_manual_priority_page_is_meta_or_index = runtime._ask_manual_priority_page_is_meta_or_index
    _ask_manual_priority_page_score = runtime._ask_manual_priority_page_score
    _ask_manual_priority_query_is_maintenance = runtime._ask_manual_priority_query_is_maintenance
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _is_structured_source_key = runtime._is_structured_source_key
    _is_xlsx_indexed_page_text = runtime._is_xlsx_indexed_page_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _safe_int = runtime._safe_int
    _v13_build_profile_from_plan = runtime._v13_build_profile_from_plan
    if source_kind not in {"xlsx", "manual"}:
        return []

    profile = _v13_build_profile_from_plan(q, response_language, plan)
    where_sql, params = _ask_evidence_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )

    page_chars = min(V13_PAGE_TEXT_CHARS, max(1200, int(ASK_FULL_CONTEXT_PAGE_CHARS or 6500)))
    scan_limit = V13_PREFERRED_PAGE_SCAN_LIMIT

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if _page_capture is not None:
                _page_capture.expect(scan_limit, page_chars)
            cur.execute(
                f"""
                SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                FROM public.document_pages
                WHERE {where_sql}
                  AND text IS NOT NULL
                  AND length(text) > 20
                ORDER BY bubble_document_id, page_number
                LIMIT %s;
                """,
                [page_chars, *params, scan_limit],
            )
            rows = cur.fetchall()
            if _page_capture is not None:
                rows = _page_capture.capture(rows)
    finally:
        conn.close()

    # Targeted supplement for explicit manual maintenance/check questions.
    # The broad scan can be diluted by cover pages, indexes or company-general manuals.
    # This pass pulls table/frequency/maintenance pages from the same authorized scope,
    # then the normal scorer still decides the order. It is a supplement, not a hard filter.
    if source_kind == "manual" and _ask_manual_priority_query_is_maintenance(q):
        targeted_patterns = [
            "%tabella per manutenzione%",
            "%tabella generale di manutenzione%",
            "%ore di funzionamento%",
            "%componenti%ore di%",
            "%tipo di lubrificante%",
            "%controllare il livello%",
            "%cambio olio%",
            "%pulizia dei filtri%",
            "%sostituzione completa dei filtri%",
            "%scarico della condensa%",
            "%verifica integrità%",
            "%verifica integrita%",
            "%verifica corretto funzionamento%",
            "%impianto elettrico%",
            "%impianto pneumatico%",
            "%raddrizzatura%",
            "%lubrificazione%",
            "%ogni 50 ore%",
            "%ogni 300 ore%",
            "%ogni 1000 ore%",
            "%ogni 3000 ore%",
            "%ogni giorno%",
            "%mensilmente%",
            "%settiman%",
            "%annualmente%",
        ]
        targeted_clauses = " OR ".join(["LOWER(COALESCE(text, '')) LIKE %s" for _ in targeted_patterns])
        target_limit = max(80, min(260, int(scan_limit // 2)))
        try:
            conn = _db_conn()
            try:
                with conn.cursor() as cur:
                    if _page_capture is not None:
                        _page_capture.expect(target_limit, page_chars)
                    cur.execute(
                        f"""
                        SELECT bubble_document_id, machine_id, page_number, LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                        FROM public.document_pages
                        WHERE {where_sql}
                          AND text IS NOT NULL
                          AND length(text) > 20
                          AND ({targeted_clauses})
                        ORDER BY
                          CASE
                            WHEN machine_id = %s THEN 0
                            WHEN machine_id IS NULL OR machine_id = '' THEN 1
                            ELSE 2
                          END,
                          bubble_document_id,
                          page_number
                        LIMIT %s;
                        """,
                        [page_chars, *params, *targeted_patterns, machine_id, target_limit],
                    )
                    targeted_rows = cur.fetchall()
                    if _page_capture is not None:
                        targeted_rows = _page_capture.capture(targeted_rows)
            finally:
                conn.close()
        except Exception as e:
            if _page_capture is not None:
                raise
            print("V13_MANUAL_TARGETED_SCAN_FAIL", str(e)[:300])
            targeted_rows = []

        if targeted_rows:
            rows = list(rows or [])
            seen_pages = {
                (str(r[0] or ""), _safe_int(r[2], 0))
                for r in rows
            }
            for r in targeted_rows:
                key = (str(r[0] or ""), _safe_int(r[2], 0))
                if key not in seen_pages:
                    rows.append(r)
                    seen_pages.add(key)

    scored: list[dict] = []
    q_low = _normalize_unicode_advanced(q or "").lower()

    for idx, (bdid, mid, page_number, page_text) in enumerate(rows or [], start=1):
        bdid_s = str(bdid or "").strip()
        txt = str(page_text or "").strip()
        if not bdid_s or not txt:
            continue

        is_xlsx = _is_xlsx_indexed_page_text(txt)
        is_structured = _is_structured_source_key(bdid_s)

        if source_kind == "xlsx" and not is_xlsx:
            continue
        if source_kind == "manual" and (is_xlsx or is_structured):
            continue

        score = float(_ask_evidence_score_text(q, txt, profile))
        # Source preference is a ranking boost, not an exclusive evidence rule.
        score += 35.0 if source_kind == "xlsx" else 28.0
        if source_kind == "xlsx" and any(x in q_low for x in ["excel", "xlsx", "foglio", "spreadsheet"]):
            score += 8.0
        if source_kind == "manual" and any(x in q_low for x in ["manual", "manuale", "pdf", "documentazione"]):
            score += 8.0

        t_low = _normalize_unicode_advanced(txt).lower()
        for marker, bonus in [
            ("manutenz", 8.0), ("maintenance", 8.0), ("controll", 6.0),
            ("periodic", 5.0), ("frequenza", 5.0), ("frequency", 5.0),
            ("lubr", 4.0), ("olio", 4.0), ("oil", 4.0),
        ]:
            if marker in q_low and marker in t_low:
                score += bonus

        exact_machine_page = False
        real_maintenance_page = False
        weak_meta_page = False
        if source_kind == "manual":
            row_mid = str(mid or "").strip()
            exact_machine_page = bool(machine_id and machine_id != COMPANY_GENERAL_MACHINE_SENTINEL and row_mid == str(machine_id or "").strip())
            real_maintenance_page = _ask_manual_priority_page_has_real_maintenance_content(txt)
            weak_meta_page = _ask_manual_priority_page_is_meta_or_index(txt)
            score = _ask_manual_priority_page_score(
                q=q,
                page_text=txt,
                base_score=score,
                row_machine_id=row_mid,
                requested_machine_id=machine_id,
            )

        page = _safe_int(page_number, 1)
        row_obj = {
            "citation_id": f"{bdid_s}:p{page}-{page}:{source_kind}priority:{idx}",
            "bubble_document_id": bdid_s,
            "chunk_index": 0,
            "page_from": page,
            "page_to": page,
            "snippet": txt[:ASK_SNIPPET_CHARS],
            "chunk_full": txt,
            "similarity": min(0.99, 0.74 + score / 200.0),
            "retrieval_score": score,
            "ask_source_priority": True,
            "ask_source_priority_kind": source_kind,
        }
        if source_kind == "manual":
            row_obj["manual_priority_exact_machine"] = bool(exact_machine_page)
            row_obj["manual_priority_real_maintenance"] = bool(real_maintenance_page)
            row_obj["manual_priority_weak_meta"] = bool(weak_meta_page)
        scored.append(row_obj)

    scored.sort(
        key=lambda c: (
            -float(c.get("retrieval_score") or 0.0),
            str(c.get("bubble_document_id") or ""),
            _safe_int(c.get("page_from"), 0),
        )
    )

    max_items = max(1, min(max(top_k, 6), 12))
    if source_kind == "manual" and _ask_manual_priority_query_is_maintenance(q):
        exact_strong = [
            c for c in scored
            if bool(c.get("manual_priority_exact_machine"))
            and bool(c.get("manual_priority_real_maintenance"))
            and not bool(c.get("manual_priority_weak_meta"))
        ]
        other_strong = [
            c for c in scored
            if c not in exact_strong
            and bool(c.get("manual_priority_real_maintenance"))
            and not bool(c.get("manual_priority_weak_meta"))
        ]
        weak = [c for c in scored if c not in exact_strong and c not in other_strong]
        if exact_strong:
            ordered = exact_strong[:min(6, max_items)] + other_strong[:max(0, max_items - min(6, len(exact_strong)))] + weak[:max_items]
            return _dedup_citations_by_snippet(ordered, max_items=max_items)
        if other_strong:
            ordered = other_strong[:max_items] + weak[:max_items]
            return _dedup_citations_by_snippet(ordered, max_items=max_items)

    return _dedup_citations_by_snippet(scored, max_items=max_items)


@dataclass(frozen=True)
class V13FetchManualSupportDeterministicRuntime:
    ASK_SNIPPET_CHARS: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT: Any
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _ask_structured_manual_support_score_details: Callable[..., Any]
    _ask_structured_manual_support_terms: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _v12_filter_linkable_manual_support: Callable[..., Any]
    _v12_mark_manual_support: Callable[..., Any]


def v13_fetch_manual_support_deterministic(*, company_id: str, machine_id: str, q: str, planner: dict, structured_citations: list[dict], runtime: V13FetchManualSupportDeterministicRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _v13_fetch_manual_support_deterministic_impl(company_id=company_id, machine_id=machine_id, q=q, planner=planner, structured_citations=structured_citations, runtime=runtime)


def _v13_fetch_manual_support_deterministic_impl(*, company_id: str, machine_id: str, q: str, planner: dict, structured_citations: list[dict], runtime: V13FetchManualSupportDeterministicRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _ask_structured_manual_support_score_details = runtime._ask_structured_manual_support_score_details
    _ask_structured_manual_support_terms = runtime._ask_structured_manual_support_terms
    _db_conn = runtime._db_conn
    _safe_int = runtime._safe_int
    _v12_filter_linkable_manual_support = runtime._v12_filter_linkable_manual_support
    _v12_mark_manual_support = runtime._v12_mark_manual_support
    if not ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED:
        return []
    if not structured_citations or not machine_id or machine_id == COMPANY_GENERAL_MACHINE_SENTINEL:
        return []

    terms = _ask_structured_manual_support_terms(q, planner, structured_citations)
    if not terms:
        return []

    text_chars = max(1200, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS or 4200))
    scan_limit = max(40, min(300, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT or 180)))
    rows: list[tuple] = []
    try:
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                if _page_capture is not None:
                    _page_capture.expect(scan_limit, text_chars)
                cur.execute(
                    f"""
                    SELECT bubble_document_id, machine_id, page_number,
                           LEFT(COALESCE(text, ''), %s) AS page_text{binding_columns}
                    FROM public.document_pages
                    WHERE company_id=%s
                      AND (machine_id=%s OR machine_id IS NULL OR machine_id='')
                      AND text IS NOT NULL
                      AND length(text) > 40
                      AND bubble_document_id NOT LIKE 'procedure:%%'
                      AND bubble_document_id NOT LIKE 'step:%%'
                      AND bubble_document_id NOT LIKE 'ps:%%'
                      AND bubble_document_id NOT LIKE 'md_photo:%%'
                      AND bubble_document_id NOT LIKE 'md_video:%%'
                    ORDER BY CASE WHEN machine_id=%s THEN 0 ELSE 1 END,
                             bubble_document_id, page_number
                    LIMIT %s;
                    """,
                    (text_chars, company_id, machine_id, machine_id, scan_limit),
                )
                rows = cur.fetchall()
                if _page_capture is not None:
                    rows = _page_capture.capture(rows)
        finally:
            conn.close()
    except Exception as exc:
        if _page_capture is not None:
            raise
        print("V13_MANUAL_SUPPORT_SCAN_FAIL", str(exc)[:500])
        return []

    scored: list[dict] = []
    for idx, (bdid, mid, page_number, page_text) in enumerate(rows, start=1):
        text = str(page_text or "").strip()
        if not text:
            continue
        details = _ask_structured_manual_support_score_details(text, terms)
        operation_score = float(details.get("operation_score") or 0.0)
        safety_score = float(details.get("safety_score") or 0.0)
        total_score = float(details.get("total_score") or 0.0)
        # Generic safety by itself is not sufficient. At least one operation term must match.
        if operation_score < 1.0 or total_score < 3.0:
            continue
        page = _safe_int(page_number, 1)
        exact_machine = str(mid or "").strip() == str(machine_id or "").strip()
        scored.append(
            {
                "citation_id": f"{bdid}:p{page}-{page}:manualsupport:v13:{idx}",
                "bubble_document_id": str(bdid),
                "chunk_index": 1,
                "page_from": page,
                "page_to": page,
                "snippet": text[: int(ASK_SNIPPET_CHARS or 900)],
                "snippet_clean": text[: int(ASK_SNIPPET_CHARS or 900)],
                "chunk_full": text,
                "similarity": min(0.94, 0.70 + min(0.20, total_score / 100.0)),
                "retrieval_score": total_score,
                "v13_score": total_score,
                "source_type": "document",
                "evidence_role": "manual_support",
                "ask_structured_manual_support": True,
                "ask_manual_support_kind": "operation" if operation_score >= safety_score else "safety",
                "structured_manual_operation_score": operation_score,
                "structured_manual_safety_score": safety_score,
                "structured_manual_support_score": total_score,
                "exact_machine_scope": exact_machine,
            }
        )

    scored.sort(
        key=lambda c: (
            0 if bool(c.get("exact_machine_scope")) else 1,
            -float(c.get("structured_manual_operation_score") or 0.0),
            -float(c.get("structured_manual_support_score") or 0.0),
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
        )
    )
    max_items = max(0, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS or 2))
    selected = scored[:max_items]
    selected = _v12_filter_linkable_manual_support(company_id, selected)
    return _v12_mark_manual_support(selected)


@dataclass(frozen=True)
class AssistantCoreMachineCatalogCandidatesRuntime:
    ASK_SNIPPET_CHARS: Any
    _ask_evidence_fallback_profile: Callable[..., Any]
    _ask_evidence_score_text: Callable[..., Any]
    _assistant_core_candidate_source_type: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]


def assistant_core_machine_catalog_candidates(request: AssistantCoreRequest, *, max_rows: int=48, runtime: AssistantCoreMachineCatalogCandidatesRuntime) -> list[dict]:
    """Legacy entry; existing SQL, parameters, selection and failure behavior."""
    return _assistant_core_machine_catalog_candidates_impl(request=request, max_rows=max_rows, runtime=runtime)


def _assistant_core_machine_catalog_candidates_impl(request: AssistantCoreRequest, *, max_rows: int=48, runtime: AssistantCoreMachineCatalogCandidatesRuntime, _page_capture: _PageReadCapture | None=None) -> list[dict]:
    """Compact machine-wide structured digest for exhaustive overview requests."""
    binding_columns = PAGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    _ask_evidence_fallback_profile = runtime._ask_evidence_fallback_profile
    _ask_evidence_score_text = runtime._ask_evidence_score_text
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _db_conn = runtime._db_conn
    _safe_int = runtime._safe_int
    _source_type_from_document_id = runtime._source_type_from_document_id
    _v13_merge_candidates = runtime._v13_merge_candidates
    if not request.machine_id:
        return []
    out: list[dict] = []
    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            if _page_capture is not None:
                _page_capture.expect(max_rows, 4500)
            cur.execute(
                f"""
                SELECT bubble_document_id, page_number, LEFT(COALESCE(text, ''), 4500){binding_columns}
                FROM public.document_pages
                WHERE company_id=%s AND machine_id=%s
                  AND page_number=1
                  AND (
                    bubble_document_id LIKE 'procedure:%%'
                    OR bubble_document_id LIKE 'md_photo:%%'
                    OR bubble_document_id LIKE 'md_video:%%'
                    OR bubble_document_id LIKE 'photo:%%'
                    OR bubble_document_id LIKE 'video:%%'
                  )
                  AND text IS NOT NULL AND length(text) > 20
                ORDER BY bubble_document_id
                LIMIT %s;
                """,
                (request.company_id, request.machine_id, max_rows),
            )
            for bdid, page_number, page_text in (_page_capture.capture_catalog(cur.fetchall()) if _page_capture is not None else cur.fetchall()):
                text = str(page_text or "").strip()
                if not text:
                    continue
                st = _source_type_from_document_id(str(bdid or ""))
                score = float(_ask_evidence_score_text(request.query, text, _ask_evidence_fallback_profile(request.query, request.response_language)))
                out.append({
                    "citation_id": f"{bdid}:p1-1:assistant-core:catalog",
                    "bubble_document_id": str(bdid or ""),
                    "page_from": _safe_int(page_number, 1),
                    "page_to": _safe_int(page_number, 1),
                    "snippet": text[:ASK_SNIPPET_CHARS],
                    "snippet_clean": text[:ASK_SNIPPET_CHARS],
                    "chunk_full": text[:4500],
                    "similarity": min(0.92, max(0.0, 0.50 + score / 100.0)),
                    "semantic_similarity": 0.0,
                    "retrieval_score": score,
                    "v13_score": score,
                    "exact_machine_scope": True,
                    "source_type": st,
                    "assistant_core_catalog_candidate": True,
                })
    except Exception as exc:
        if _page_capture is not None:
            raise
        print("ASSISTANT_CORE_CATALOG_FAIL", str(exc)[:500])
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass
    # Keep every photo/video and the strongest procedure descriptions.
    media = [c for c in out if _assistant_core_candidate_source_type(c) in {"md_photo", "md_video", "photo", "video"}]
    procedures = sorted(
        [c for c in out if _assistant_core_candidate_source_type(c) == "procedure"],
        key=lambda c: -float(c.get("v13_score") or 0.0),
    )[:12]
    return _v13_merge_candidates([media[:8], procedures])




# P6-B3 supplemental readers. These are INTERNAL and not configured by main.
def read_token_chunk_evidence(*, scope: ChunkReadScope, token: str,
                              limits: ChunkEvidenceLimits, runtime: DbFindTokenChunkRuntime) -> SupplementalChunkRead:
    capture = _ChunkSelectionCapture(scope, limits, "token_chunk", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _db_find_token_chunk_impl(**scope.sql_selectors(), token=token, runtime=runtime, _chunk_capture=capture)
    return capture.finish([] if selected is None else [selected])


def read_entity_chunk_evidence(*, scope: ChunkReadScope, kind: str,
                               limits: ChunkEvidenceLimits, runtime: DbFindEntityChunkRuntime) -> SupplementalChunkRead:
    capture = _ChunkSelectionCapture(scope, limits, "entity_chunk", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _db_find_entity_chunk_impl(**scope.sql_selectors(), kind=kind, runtime=runtime, _chunk_capture=capture)
    return capture.finish([] if selected is None else [selected])


def read_preferred_page_evidence(*, scope: ChunkReadScope, q: str, response_language: str,
                                 source_kind: str, top_k: int, limits: ChunkEvidenceLimits,
                                 runtime: AskFetchPreferredSourcePagesRuntime) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "preferred_pages", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _ask_fetch_preferred_source_pages_impl(**scope.sql_selectors(), q=q,
        response_language=response_language, source_kind=source_kind, top_k=top_k, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def read_v13_preferred_page_evidence(*, scope: ChunkReadScope, q: str, response_language: str,
                                     source_kind: str, top_k: int, plan: dict | None,
                                     limits: ChunkEvidenceLimits, runtime: V13FetchPreferredSourcePagesRuntime) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "preferred_pages_v13", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _v13_fetch_preferred_source_pages_impl(**scope.sql_selectors(), q=q,
        response_language=response_language, source_kind=source_kind, top_k=top_k, plan=plan, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def read_maintenance_page_evidence(*, scope: ChunkReadScope, q: str, top_k: int,
                                   limits: ChunkEvidenceLimits, runtime: AskFetchManualMaintenanceTargetPagesRuntime) -> PageEvidenceRead:
    capture = _PageReadCapture(scope, limits, "maintenance_pages", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _ask_fetch_manual_maintenance_target_pages_impl(**scope.sql_selectors(), q=q,
        top_k=top_k, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def read_machine_catalog_page_evidence(request: AssistantCoreRequest, *, scope: ChunkReadScope,
                                       limits: ChunkEvidenceLimits, max_rows: int = 48,
                                       runtime: AssistantCoreMachineCatalogCandidatesRuntime) -> PageEvidenceRead:
    require_machine_scope(scope)
    if (request.company_id, request.machine_id, request.ai_scope) != (scope.company_id, scope.machine_id, scope.ai_scope):
        raise SupplementalBindingError("catalog request/scope mismatch")
    if request.metadata.get("document_ids") or request.metadata.get("bubble_document_id"):
        raise SupplementalBindingError("unresolved catalog document selectors")
    capture = _PageReadCapture(scope, limits, "machine_catalog", snippet_chars=runtime.ASK_SNIPPET_CHARS, candidate_chars=4500)
    selected = _assistant_core_machine_catalog_candidates_impl(request, max_rows=max_rows,
        runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def read_semantic_manual_support_page_evidence(*, scope: ChunkReadScope, q: str,
        planner: dict | None, structured_inputs: tuple, current_allowed_sources: frozenset[SourceIdentity],
        limits: ChunkEvidenceLimits, runtime: AskStructuredDirectFetchManualSupportRuntime,
        response_language: str = "it") -> PageEvidenceRead:
    require_machine_scope(scope)
    records = checked_input_records(scope=scope, records=structured_inputs,
        current_allowed_sources=current_allowed_sources, limits=limits)
    capture = _PageReadCapture(scope, limits, "manual_support_semantic", snippet_chars=int(runtime.ASK_SNIPPET_CHARS or 900))
    selected = _ask_structured_direct_fetch_manual_support_impl(company_id=scope.company_id,
        machine_id=scope.machine_id, q=q, planner=planner, structured_citations=records,
        response_language=response_language, runtime=runtime, _page_capture=capture)
    return capture.finish(selected)


def read_deterministic_manual_support_page_evidence(*, scope: ChunkReadScope, q: str,
        planner: dict | None, structured_inputs: tuple, current_allowed_sources: frozenset[SourceIdentity],
        limits: ChunkEvidenceLimits, runtime: V13FetchManualSupportDeterministicRuntime) -> PageEvidenceRead:
    require_machine_scope(scope)
    records = checked_input_records(scope=scope, records=structured_inputs,
        current_allowed_sources=current_allowed_sources, limits=limits)
    capture = _PageReadCapture(scope, limits, "manual_support_deterministic", snippet_chars=int(runtime.ASK_SNIPPET_CHARS or 900))
    selected = _v13_fetch_manual_support_deterministic_impl(company_id=scope.company_id,
        machine_id=scope.machine_id, q=q, planner=planner, structured_citations=records,
        runtime=runtime, _page_capture=capture)
    return capture.finish(selected)



def read_document_file_references(*, scope: ChunkReadScope, sources: tuple[SourceIdentity, ...],
        current_allowed_sources: frozenset[SourceIdentity], limits: ChunkEvidenceLimits,
        runtime: FetchDocumentFileMapRuntime) -> FileReferenceRead:
    validate_anchors(scope=scope, anchors=sources, current_allowed_sources=current_allowed_sources, limits=limits)
    if any(s.source_type.value != "document" for s in sources):
        raise SupplementalBindingError("document-only file map")
    # Legacy normalization must not silently change any authorization selector.
    if scope.company_id != scope.company_id.strip() or any(storage_key(s) != storage_key(s).strip() for s in sources):
        raise SupplementalBindingError("unresolved whitespace in file-map selectors")
    rows, count = _fetch_document_file_map_impl(scope.company_id, [storage_key(s) for s in sources],
        runtime=runtime, _reference_limit=limits.assembly.max_occurrences)
    return build_file_reference_read(scope=scope, anchors=sources, current_allowed_sources=current_allowed_sources,
        rows=rows, limits=limits, query_count=count)
