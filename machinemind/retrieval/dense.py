"""P4-A extraction of existing retrieval behavior.

No app/DB/provider is imported here. Dependencies are supplied by the composition
root at call time. SQL, thresholds, ordering, scope precedence and error behavior
are deliberately preserved, including historical quirks. This is not a semantic
rewrite or a new query-scope policy.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class DenseRuntime:
    connect_db: Callable[[], Any]
    snippet_chars: int


@dataclass(frozen=True)
class DenseMultiQueryRuntime:
    dedup_text_values: Callable[..., list[str]]
    max_dense_queries: int
    embed_texts: Callable[..., list[list[float]]]
    vector_literal: Callable[[list[float]], str]
    fetch_candidates: Callable[..., Any]
    rows_to_candidates: Callable[..., list[dict]]
    merge_ranked_lists: Callable[..., list[dict]]


def fetch_dense_chunk_candidates(
    *,
    company_id: str,
    machine_id: str,
    q_vec_lit: str,
    candidate_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    runtime: DenseRuntime,
) -> tuple[Optional[int], list[tuple]]:
    _db_conn = runtime.connect_db
    ASK_SNIPPET_CHARS = runtime.snippet_chars
    chunks_matching_filter = None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if doc_ids:
                if debug:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM public.document_chunks
                        WHERE company_id=%s
                          AND bubble_document_id = ANY(%s)
                          AND embedding IS NOT NULL;
                        """,
                        (company_id, doc_ids),
                    )
                    chunks_matching_filter = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to,
                           left(chunk_text, %s) AS snippet,
                           left(chunk_text, 2000) AS chunk_full,
                           1 - (embedding <=> %s::vector) AS similarity,
                           embedding,
                           CASE WHEN machine_id = %s THEN TRUE ELSE FALSE END AS exact_machine_scope
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = ANY(%s)
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, machine_id, company_id, doc_ids, q_vec_lit, candidate_k),
                )

            elif bubble_document_id:
                bdid = bubble_document_id

                if debug:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM public.document_chunks
                        WHERE company_id=%s
                          AND bubble_document_id=%s
                          AND embedding IS NOT NULL
                          AND (machine_id=%s OR machine_id IS NULL OR machine_id = '');
                        """,
                        (company_id, bdid, machine_id),
                    )
                    chunks_matching_filter = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to,
                           left(chunk_text, %s) AS snippet,
                           left(chunk_text, 2000) AS chunk_full,
                           1 - (embedding <=> %s::vector) AS similarity,
                           embedding,
                           CASE WHEN machine_id = %s THEN TRUE ELSE FALSE END AS exact_machine_scope
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, machine_id, company_id, bdid, machine_id, q_vec_lit, candidate_k),
                )

            else:
                if debug:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM public.document_chunks
                        WHERE company_id=%s
                          AND embedding IS NOT NULL
                          AND (machine_id=%s OR machine_id IS NULL OR machine_id = '');
                        """,
                        (company_id, machine_id),
                    )
                    chunks_matching_filter = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to,
                           left(chunk_text, %s) AS snippet,
                           left(chunk_text, 2000) AS chunk_full,
                           1 - (embedding <=> %s::vector) AS similarity,
                           embedding,
                           CASE WHEN machine_id = %s THEN TRUE ELSE FALSE END AS exact_machine_scope
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL OR machine_id = '')
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, machine_id, company_id, machine_id, q_vec_lit, candidate_k),
                )

            raw_rows = cur.fetchall()
            return chunks_matching_filter, raw_rows
    finally:
        conn.close()



def raw_rows_to_dense_candidates(
    raw_rows: list[tuple],
    *,
    query_used: Optional[str] = None,
) -> list[dict]:
    candidates: list[dict] = []

    for row in raw_rows:
        if len(row) == 9:
            (
                bdid,
                chunk_index,
                page_from,
                page_to,
                snippet,
                chunk_full,
                similarity,
                embedding,
                exact_machine_scope,
            ) = row
        elif len(row) == 8:
            # Backward-compatible for tests or old fixtures.
            (
                bdid,
                chunk_index,
                page_from,
                page_to,
                snippet,
                chunk_full,
                similarity,
                embedding,
            ) = row
            exact_machine_scope = False
        else:
            raise ValueError(f"Unexpected dense candidate row width: {len(row)}")

        if embedding is None:
            emb_list = None
        elif isinstance(embedding, list):
            emb_list = embedding
        else:
            value = str(embedding).strip().strip("[]")
            emb_list = [float(x) for x in value.split(",") if x.strip()]

        item = {
            "citation_id": f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}",
            "bubble_document_id": str(bdid),
            "chunk_index": int(chunk_index),
            "page_from": int(page_from),
            "page_to": int(page_to),
            "snippet": (snippet or "").strip(),
            "chunk_full": (chunk_full or "").strip(),
            # Keep the raw cosine similarity separate from later routing/ranking
            # scores. Deterministic page/structured helpers also expose a field named
            # ``similarity`` but those values are synthetic and must never prove
            # evidence sufficiency.
            "similarity": float(similarity),
            "semantic_similarity": float(similarity),
            "embedding_list": emb_list or [],
            "exact_machine_scope": bool(exact_machine_scope),
        }

        if query_used is not None:
            item["query_used"] = query_used

        candidates.append(item)

    return candidates



def dense_candidates_multi_query(
    *,
    query_texts: list[str],
    company_id: str,
    machine_id: str,
    candidate_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    runtime: DenseMultiQueryRuntime,
) -> tuple[Optional[int], list[dict], dict[str, list[float]]]:
    _dedup_text_values = runtime.dedup_text_values
    SEMANTIC_MAX_DENSE_QUERIES = runtime.max_dense_queries
    _openai_embed_texts = runtime.embed_texts
    _vector_literal = runtime.vector_literal
    _fetch_dense_chunk_candidates = runtime.fetch_candidates
    _raw_rows_to_dense_candidates = runtime.rows_to_candidates
    _rrf_merge_candidates = runtime.merge_ranked_lists
    cleaned_queries = _dedup_text_values(query_texts, limit=max(1, SEMANTIC_MAX_DENSE_QUERIES + 2))
    if not cleaned_queries:
        return None, [], {}

    vectors = _openai_embed_texts(cleaned_queries)
    query_vectors: dict[str, list[float]] = {}
    dense_ranked_lists: list[list[dict]] = []
    chunks_matching_filter = None

    for qq, vec in zip(cleaned_queries, vectors):
        query_vectors[qq] = vec
        q_vec_lit = _vector_literal(vec)

        current_chunks_matching_filter, raw_rows = _fetch_dense_chunk_candidates(
            company_id=company_id,
            machine_id=machine_id,
            q_vec_lit=q_vec_lit,
            candidate_k=candidate_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
            debug=debug,
        )

        if chunks_matching_filter is None:
            chunks_matching_filter = current_chunks_matching_filter

        ranked = _raw_rows_to_dense_candidates(raw_rows, query_used=qq)
        if ranked:
            dense_ranked_lists.append(ranked)

    merged = _rrf_merge_candidates(dense_ranked_lists, k=60)
    return chunks_matching_filter, merged, query_vectors



def search_chunk_previews(
    *,
    company_id: str,
    q_vec_lit: str,
    top_k: int,
    bubble_document_id: Optional[str] = None,
    runtime: DenseRuntime,
) -> dict:
    """Preserve /v1/ai/search company/document scope, not machine_all scope.

    Authentication, input validation, top_k clamping and the single embedding
    remain in the existing HTTP adapter. Do not reuse the machine-scoped SQL here.
    """
    _db_conn = runtime.connect_db
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if bubble_document_id:
                bubble_document_id = bubble_document_id.strip()
                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to, left(chunk_text, 400) AS preview,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
                    LIMIT %s;
                    """,
                    (q_vec_lit, company_id, bubble_document_id, q_vec_lit, top_k),
                )
                rows = cur.fetchall()

                results = []
                for (bdid, chunk_index, page_from, page_to, preview, similarity) in rows:
                    citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
                    results.append(
                        {
                            "citation_id": citation_id,
                            "bubble_document_id": bdid,
                            "chunk_index": int(chunk_index),
                            "page_from": int(page_from),
                            "page_to": int(page_to),
                            "similarity": float(similarity),
                            "preview": preview,
                        }
                    )

                return {"ok": True, "top_k": top_k, "results": results}

            cur.execute(
                """
                SELECT bubble_document_id, chunk_index, page_from, page_to, left(chunk_text, 400) AS preview,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM public.document_chunks
                WHERE company_id = %s
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
                LIMIT %s;
                """,
                (q_vec_lit, company_id, q_vec_lit, top_k),
            )
            rows = cur.fetchall()

            results = []
            for (bubble_document_id, chunk_index, page_from, page_to, preview, similarity) in rows:
                citation_id = f"{bubble_document_id}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
                results.append(
                    {
                        "citation_id": citation_id,
                        "bubble_document_id": bubble_document_id,
                        "chunk_index": int(chunk_index),
                        "page_from": int(page_from),
                        "page_to": int(page_to),
                        "similarity": float(similarity),
                        "preview": preview,
                    }
                )

            return {"ok": True, "top_k": top_k, "results": results}
    finally:
        conn.close()

