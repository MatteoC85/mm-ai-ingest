"""P4-A extraction of existing retrieval behavior.

No app/DB/provider is imported here. Dependencies are supplied by the composition
root at call time. SQL, thresholds, ordering, scope precedence and error behavior
are deliberately preserved, including historical quirks. This is not a semantic
rewrite or a new query-scope policy.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
import re

from .chunk_evidence import (ChunkEvidenceLimits, ChunkEvidenceRead, ChunkReadScope,
                             ChunkReadBindingError,
                             build_chunk_read, strip_binding_columns, validate_read)


@dataclass(frozen=True)
class LexicalRuntime:
    connect_db: Callable[[], Any]
    snippet_chars: int


@dataclass(frozen=True)
class LexicalMultiQueryRuntime:
    dedup_text_values: Callable[..., list[str]]
    max_lexical_queries: int
    search_chunks: Callable[..., list[dict]]
    dedup_citations: Callable[..., list[dict]]


def build_prefix_tsquery_from_texts(texts: list[str], limit: int = 10, *, normalize_unicode: Callable[[str], str]) -> Optional[str]:
    _normalize_unicode_advanced = normalize_unicode
    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from",
        "this", "that", "these", "those", "into", "onto", "about", "question",
        "machine", "system", "document", "documents", "manual", "answer", "issue", "problem",
        "il", "lo", "la", "i", "gli", "le", "con", "per", "quando", "durante", "dopo", "prima",
        "questo", "questa", "questi", "queste", "domanda", "documento", "documenti",
        "macchina", "sistema", "problema", "guasto", "risposta",
    }

    toks: list[str] = []
    seen = set()

    for text in texts or []:
        for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", _normalize_unicode_advanced(text or "").lower()):
            if tok in stopwords:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            toks.append(tok)
            if len(toks) >= limit:
                break
        if len(toks) >= limit:
            break

    if not toks:
        return None

    return " | ".join(f"{tok}:*" for tok in toks)



def fts_search_chunks_prefix(
    company_id: str,
    machine_id: str,
    texts: list[str],
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    *,
    runtime: LexicalRuntime,
    build_prefix_query: Callable[..., Optional[str]],
) -> list[dict]:
    """Unchanged legacy view; no canonical conversion is activated here."""
    return _read_fts_search_chunks_prefix(
        company_id, machine_id, texts, top_k, doc_ids, bubble_document_id,
        runtime=runtime, build_prefix_query=build_prefix_query,
        include_binding_columns=False,
    )[0]


def read_prefix_chunk_evidence(
    *, scope: ChunkReadScope, texts: list[str], top_k: int,
    runtime: LexicalRuntime, build_prefix_query: Callable[..., Optional[str]],
    limits: ChunkEvidenceLimits,
) -> ChunkEvidenceRead:
    """Bound prefix-FTS view; same query builder and one scoped SQL statement."""
    validate_read(scope, top_k, limits)
    candidates, rows = _read_fts_search_chunks_prefix(
        **scope.sql_selectors(), texts=texts, top_k=top_k, runtime=runtime,
        build_prefix_query=build_prefix_query, include_binding_columns=True,
    )
    if len(rows) > top_k:
        raise ChunkReadBindingError("SQL returned more rows than the requested LIMIT")
    return build_chunk_read(scope=scope, kind="fts_prefix", rows=rows,
                            candidates=candidates, limits=limits)


def _read_fts_search_chunks_prefix(
    company_id: str,
    machine_id: str,
    texts: list[str],
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    *,
    runtime: LexicalRuntime,
    build_prefix_query: Callable[..., Optional[str]],
    include_binding_columns: bool,
) -> tuple[list[dict], list[tuple]]:
    _build_prefix_tsquery_from_texts = build_prefix_query
    _db_conn = runtime.connect_db
    ASK_SNIPPET_CHARS = runtime.snippet_chars
    binding_columns = ", company_id AS evidence_company_id, machine_id AS evidence_machine_id" if include_binding_columns else ""
    ts_query = _build_prefix_tsquery_from_texts(texts, limit=10)
    if not ts_query:
        return [], []

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

            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       ts_rank_cd(
                           to_tsvector('simple', chunk_text),
                           to_tsquery('simple', %s)
                       ) AS rank{binding_columns}
                FROM public.document_chunks
                WHERE {where_sql}
                  AND to_tsvector('simple', chunk_text) @@ to_tsquery('simple', %s)
                ORDER BY rank DESC, bubble_document_id, page_from, chunk_index
                LIMIT %s;
                """,
                [ASK_SNIPPET_CHARS, ts_query, *params, ts_query, top_k],
            )
            rows = cur.fetchall()
            bound_rows = rows if include_binding_columns else []
            if include_binding_columns:
                rows = strip_binding_columns(rows, width=6)

        out: list[dict] = []
        for (bdid, chunk_index, page_from, page_to, snippet, _rank) in rows:
            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            out.append(
                {
                    "citation_id": citation_id,
                    "bubble_document_id": str(bdid),
                    "chunk_index": int(chunk_index),
                    "page_from": int(page_from),
                    "page_to": int(page_to),
                    "snippet": (snippet or "").strip(),
                    "similarity": 0.0,
                }
            )
        return out, bound_rows
    finally:
        conn.close()



def fts_search_chunks(
    company_id: str,
    machine_id: str,
    q: str,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    *,
    runtime: LexicalRuntime,
) -> list[dict]:
    """Unchanged legacy view; no canonical conversion is activated here."""
    return _read_fts_search_chunks(
        company_id, machine_id, q, top_k, doc_ids, bubble_document_id,
        runtime=runtime, include_binding_columns=False,
    )[0]


def read_fts_chunk_evidence(
    *, scope: ChunkReadScope, q: str, top_k: int,
    runtime: LexicalRuntime, limits: ChunkEvidenceLimits,
) -> ChunkEvidenceRead:
    """Bound plain-FTS view; raw FTS rank retained separately, never cosine."""
    validate_read(scope, top_k, limits)
    candidates, rows = _read_fts_search_chunks(
        **scope.sql_selectors(), q=q, top_k=top_k, runtime=runtime,
        include_binding_columns=True,
    )
    if len(rows) > top_k:
        raise ChunkReadBindingError("SQL returned more rows than the requested LIMIT")
    return build_chunk_read(scope=scope, kind="fts", rows=rows,
                            candidates=candidates, limits=limits)


def _read_fts_search_chunks(
    company_id: str,
    machine_id: str,
    q: str,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    *,
    runtime: LexicalRuntime,
    include_binding_columns: bool,
) -> tuple[list[dict], list[tuple]]:
    _db_conn = runtime.connect_db
    ASK_SNIPPET_CHARS = runtime.snippet_chars
    binding_columns = ", company_id AS evidence_company_id, machine_id AS evidence_machine_id" if include_binding_columns else ""
    q = (q or "").strip()
    if not q:
        return [], []

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

            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       ts_rank_cd(
                           to_tsvector('simple', chunk_text),
                           plainto_tsquery('simple', %s)
                       ) AS rank{binding_columns}
                FROM public.document_chunks
                WHERE {where_sql}
                  AND to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC, bubble_document_id, page_from, chunk_index
                LIMIT %s;
                """,
                [ASK_SNIPPET_CHARS, q, *params, q, top_k],
            )
            rows = cur.fetchall()
            bound_rows = rows if include_binding_columns else []
            if include_binding_columns:
                rows = strip_binding_columns(rows, width=6)

            out: list[dict] = []
            for (bdid, chunk_index, page_from, page_to, snippet, _rank) in rows:
                citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
                out.append(
                    {
                        "citation_id": citation_id,
                        "bubble_document_id": bdid,
                        "chunk_index": int(chunk_index),
                        "page_from": int(page_from),
                        "page_to": int(page_to),
                        "snippet": (snippet or "").strip(),
                        "similarity": 0.0,
                    }
                )
            return out, bound_rows
    finally:
        conn.close()



def fts_search_chunks_multi(
    company_id: str,
    machine_id: str,
    queries: list[str],
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    *,
    runtime: LexicalMultiQueryRuntime,
) -> list[dict]:
    _dedup_text_values = runtime.dedup_text_values
    SEMANTIC_MAX_LEXICAL_QUERIES = runtime.max_lexical_queries
    _fts_search_chunks = runtime.search_chunks
    _dedup_citations_by_snippet = runtime.dedup_citations
    merged: list[dict] = []

    for q in _dedup_text_values(queries, limit=max(1, SEMANTIC_MAX_LEXICAL_QUERIES + 1)):
        merged.extend(
            _fts_search_chunks(
                company_id=company_id,
                machine_id=machine_id,
                q=q,
                top_k=top_k,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
            )
        )

    return _dedup_citations_by_snippet(merged, max_items=top_k)

