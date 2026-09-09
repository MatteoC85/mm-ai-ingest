"""P4-C9: extracted context expansion implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional
from .chunk_evidence import ChunkReadScope, ChunkEvidenceLimits
from .page_evidence import PAGE_RANGE_BINDING_COLUMNS, PageEvidenceRead, _PageReadCapture
from .supplemental_evidence import (CHUNK_BINDING_COLUMNS, SupplementalChunkRead, SupplementalBindingError,
    _ChunkSelectionCapture, checked_input_records, require_request_scope, storage_key, validate_anchors)
from ..evidence.contracts import SourceIdentity

@dataclass(frozen=True)
class ExpandWithNeighborChunksRuntime:
    ASK_SNIPPET_CHARS: Any
    _db_conn: Callable[..., Any]
    re: Any


def expand_with_neighbor_chunks(company_id: str, bubble_document_id: str, citation_ids: list[str], *, radius: int=1, runtime: ExpandWithNeighborChunksRuntime) -> list[dict]:
    """Legacy entry, preserving the existing reader implementation."""
    return _expand_with_neighbor_chunks_impl(company_id=company_id, bubble_document_id=bubble_document_id, citation_ids=citation_ids, radius=radius, runtime=runtime)


def _expand_with_neighbor_chunks_impl(company_id: str, bubble_document_id: str, citation_ids: list[str], *, radius: int=1, runtime: ExpandWithNeighborChunksRuntime, _chunk_capture: _ChunkSelectionCapture | None = None) -> list[dict]:
    binding_columns = CHUNK_BINDING_COLUMNS if _chunk_capture is not None else ""
    limit_clause = " LIMIT %s" if _chunk_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    _db_conn = runtime._db_conn
    re = runtime.re
    if not citation_ids:
        return []

    parsed = []
    for cid in citation_ids:
        m = re.match(r"^(.*):p(\d+)-(\d+):c(\d+)$", str(cid).strip())
        if not m:
            continue
        bdid = m.group(1).strip()
        chunk_index = int(m.group(4))
        if bdid != bubble_document_id:
            continue
        parsed.append(chunk_index)

    if not parsed:
        return []

    min_idx = max(1, min(parsed) - radius)
    max_idx = max(parsed) + radius

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if _chunk_capture is not None:
                _chunk_capture.expect(_chunk_capture.limits.assembly.max_occurrences)
            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       left(chunk_text, 2000) AS chunk_full{binding_columns}
                FROM public.document_chunks
                WHERE company_id=%s
                  AND bubble_document_id=%s
                  AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index{limit_clause};
                """,
                (
                    ASK_SNIPPET_CHARS,
                    company_id,
                    bubble_document_id,
                    min_idx,
                    max_idx,
                ) + ((_chunk_capture.limits.assembly.max_occurrences + 1,) if _chunk_capture is not None else ()),
            )
            rows = cur.fetchall()
            if _chunk_capture is not None:
                rows = _chunk_capture.capture(rows)
    finally:
        conn.close()

    out = []
    for (bdid, chunk_index, page_from, page_to, snippet, chunk_full) in rows:
        cid = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
        out.append(
            {
                "citation_id": cid,
                "bubble_document_id": str(bdid),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": (snippet or "").strip(),
                "chunk_full": (chunk_full or "").strip(),
                "similarity": 0.0,
            }
        )

    return out


@dataclass(frozen=True)
class AssistantCoreEnumerationRequestedRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def assistant_core_enumeration_requested(
    request: AssistantCoreRequest,
    decision: AssistantCoreDecision,
    *, runtime: AssistantCoreEnumerationRequestedRuntime,
) -> bool:
    """Detect a request for an exhaustive list without machine-specific keywords.

    The semantic router still defines the facets. This language-level signal only
    asks retrieval/verification to preserve complete option lists instead of a few
    representative examples. Italian and English are supported symmetrically.
    """
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    query = _normalize_unicode_advanced(str(request.query or "")).lower()
    interrogative = re.search(
        r"\b(?:quali|elenca(?:mi|re)?|tutt[ei]|which|what|list|all|available)\b",
        query,
    )
    enumerable = re.search(
        r"\b(?:tip[oi]|modalit[aà]|opzion[ei]|impostazion[ei]|parametr[oi]|"
        r"controll[oi]|component[ei]|grupp[oi]|voc[ei]|stat[oi]|azion[ei]|"
        r"selezion[ei]|types?|modes?|options?|settings?|parameters?|controls?|"
        r"components?|groups?|items?|states?|actions?|selections?)\b",
        query,
    )
    if interrogative and enumerable:
        return True
    # Router facets are semantic and may expose the same intent even when the user
    # uses terse wording such as "configurable controls?".
    facet_text = _normalize_unicode_advanced(
        " ".join(list(decision.required_facets or []))
    ).lower()
    return bool(
        re.search(r"\b(?:tipo|modalit[aà]|opzion|impostazion|parametr|control|type|mode|option|setting|parameter)\b", facet_text)
        and re.search(r"\b(?:quali|which|what|elenca|list|tutti|all)\b", query)
    )


@dataclass(frozen=True)
class AssistantCoreExtractEnumeratedItemsRuntime:
    re: Any


def assistant_core_extract_enumerated_items(text: str, *, limit: int = 48, runtime: AssistantCoreExtractEnumeratedItemsRuntime) -> list[str]:
    """Extract short option labels from bullets/slash lists for verifier context.

    These are candidates, not automatically trusted requirements. The semantic
    verifier keeps only labels relevant to the user's requested category.
    """
    re = runtime.re
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    out: list[str] = []
    bullet_pending = False

    def add(label: str) -> None:
        value = re.sub(r"^[\-•*\u2022\s]+", "", str(label or "")).strip(" .;:-")
        if not value or len(value) < 2 or len(value) > 72:
            return
        words = value.split()
        if len(words) > 9:
            return
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", value):
            return
        low = value.casefold()
        if low in {"note", "attention", "description", "section", "source_type"}:
            return
        if low not in {x.casefold() for x in out}:
            out.append(value)

    for line in raw_lines:
        if not line:
            continue
        if line in {"-", "•", "*", "–"}:
            bullet_pending = True
            continue
        marked = bool(re.match(r"^[\-•*–]\s*", line))
        candidate = re.sub(r"^[\-•*–]\s*", "", line).strip()
        if bullet_pending or marked:
            label = re.split(r"\s*[:;–—]\s*", candidate, maxsplit=1)[0]
            add(label)
            bullet_pending = False
        else:
            bullet_pending = False

        # Short inline alternatives are common in structured Step records.
        if "/" in line and re.search(
            r"(?i)\b(?:tipo|type|modalit|mode|azione|action|arrest|stop|polar|control|controll)\b",
            line,
        ):
            tail = line.split(":", 1)[-1]
            for part in re.split(r"\s*/\s*", tail):
                add(re.split(r"\s*[,;.]\s*", part, maxsplit=1)[0])
        if len(out) >= limit:
            break

    # PDF/HMI extraction often flattens bullets into one paragraph. Recover short
    # labels immediately followed by a colon without relying on a specific machine
    # or language. The semantic verifier later keeps only labels in the requested
    # category, so this expands recall without declaring every label mandatory.
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    for match in re.finditer(
        r"(?:(?<=^)|(?<=[.;•]))\s*([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9 /_-]{1,46})\s*:\s*",
        flat,
    ):
        add(match.group(1))
        if len(out) >= limit:
            break
    return out[:limit]


@dataclass(frozen=True)
class AssistantCoreEnumerationMetricsRuntime:
    _assistant_core_extract_enumerated_items: Callable[..., Any]
    re: Any


def assistant_core_enumeration_metrics(text: str, *, runtime: AssistantCoreEnumerationMetricsRuntime) -> dict:
    _assistant_core_extract_enumerated_items = runtime._assistant_core_extract_enumerated_items
    re = runtime.re
    items = _assistant_core_extract_enumerated_items(text, limit=48)
    headings = len(re.findall(
        r"(?m)^\s*[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 _/\-]{2,45}:\s*$",
        str(text or ""),
    ))
    return {"items": items, "item_count": len(items), "heading_count": headings}


@dataclass(frozen=True)
class AssistantCoreExpandEnumerationSectionsRuntime:
    ASK_SNIPPET_CHARS: Any
    V13_PAGE_TEXT_CHARS: Any
    _db_conn: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _is_structured_source_key: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]


def assistant_core_expand_enumeration_sections(*, request: AssistantCoreRequest, retrieval: dict, candidates: list[dict], max_documents: int=4, page_radius: int=3, max_pages: int=18, runtime: AssistantCoreExpandEnumerationSectionsRuntime) -> list[dict]:
    """Legacy entry, preserving the existing reader implementation."""
    return _assistant_core_expand_enumeration_sections_impl(request=request, retrieval=retrieval, candidates=candidates, max_documents=max_documents, page_radius=page_radius, max_pages=max_pages, runtime=runtime)


def _assistant_core_expand_enumeration_sections_impl(*, request: AssistantCoreRequest, retrieval: dict, candidates: list[dict], max_documents: int=4, page_radius: int=3, max_pages: int=18, runtime: AssistantCoreExpandEnumerationSectionsRuntime, _page_capture: _PageReadCapture | None = None) -> list[dict]:
    """Fetch complete nearby manual/HMI pages for exhaustive-list requests.

    The semantic hit selects the document; this function only expands the same
    document around that hit. It never crosses company/machine scope and therefore
    improves recall without replacing the ranked baseline.
    """
    binding_columns = PAGE_RANGE_BINDING_COLUMNS if _page_capture is not None else ""
    ASK_SNIPPET_CHARS = runtime.ASK_SNIPPET_CHARS
    V13_PAGE_TEXT_CHARS = runtime.V13_PAGE_TEXT_CHARS
    _db_conn = runtime._db_conn
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _is_structured_source_key = runtime._is_structured_source_key
    _safe_int = runtime._safe_int
    _source_type_from_document_id = runtime._source_type_from_document_id
    docs: dict[str, dict] = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        bdid = str(c.get("bubble_document_id") or "").strip()
        if not bdid or _is_structured_source_key(bdid):
            continue
        p1 = _safe_int(c.get("page_from"), 0)
        p2 = _safe_int(c.get("page_to"), p1)
        if p1 <= 0:
            continue
        score = float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0)
        row = docs.setdefault(bdid, {"low": p1, "high": max(p1, p2), "score": score})
        row["low"] = min(int(row["low"]), p1)
        row["high"] = max(int(row["high"]), max(p1, p2))
        row["score"] = max(float(row["score"]), score)
    selected_docs = sorted(docs.items(), key=lambda x: -float(x[1]["score"]))[:max_documents]
    if not selected_docs:
        return []
    out: list[dict] = []
    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            for bdid, meta in selected_docs:
                low = max(1, int(meta["low"]) - int(page_radius))
                high = int(meta["high"]) + int(page_radius)
                if _page_capture is not None:
                    _page_capture.expect(max_pages, V13_PAGE_TEXT_CHARS)
                cur.execute(
                    f"""
                    SELECT machine_id, page_number, LEFT(COALESCE(text, ''), %s){binding_columns}
                    FROM public.document_pages
                    WHERE company_id=%s AND bubble_document_id=%s
                      AND page_number BETWEEN %s AND %s
                      AND text IS NOT NULL AND length(text) > 20
                    ORDER BY page_number
                    LIMIT %s;
                    """,
                    (V13_PAGE_TEXT_CHARS, request.company_id, bdid, low, high, max_pages),
                )
                for mid, page_number, page_text in (_page_capture.capture_range(cur.fetchall(), bdid) if _page_capture is not None else cur.fetchall()):
                    text = str(page_text or "").strip()
                    if not text:
                        continue
                    page = _safe_int(page_number, 1)
                    out.append({
                        "citation_id": f"{bdid}:p{page}-{page}:assistant-core:section",
                        "bubble_document_id": bdid,
                        "chunk_index": 0,
                        "page_from": page,
                        "page_to": page,
                        "snippet": text[:ASK_SNIPPET_CHARS],
                        "snippet_clean": text[:ASK_SNIPPET_CHARS],
                        "chunk_full": text[:V13_PAGE_TEXT_CHARS],
                        "similarity": min(0.92, max(0.0, 0.50 + float(meta["score"]) * 0.08)),
                        "semantic_similarity": 0.0,
                        "retrieval_score": float(meta["score"]),
                        "v13_score": float(meta["score"]),
                        "exact_machine_scope": str(mid or "").strip() == str(request.machine_id or "").strip(),
                        "source_type": _source_type_from_document_id(bdid),
                        "assistant_core_section_expansion": True,
                    })
    except Exception as exc:
        if _page_capture is not None:
            raise
        print("ASSISTANT_CORE_SECTION_EXPANSION_FAIL", str(exc)[:500])
    finally:
        if conn is not None:
            try: conn.close()
            except Exception:
                if _page_capture is not None:
                    raise
                pass
    return _dedup_citations_by_snippet(out, max_items=max_pages)


@dataclass(frozen=True)
class AssistantCoreRootApplicabilityRecordsRuntime:
    _ask_evidence_scope_where: Callable[..., Any]
    _assistant_core_candidate_evidence_text: Callable[..., Any]
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_scope_value: Callable[..., Any]
    _db_conn: Callable[..., Any]
    _is_structured_source_key: Callable[..., Any]
    _safe_int: Callable[..., Any]


def assistant_core_root_applicability_records(
    request: AssistantCoreRequest, candidates: list[dict],
    *, runtime: AssistantCoreRootApplicabilityRecordsRuntime,
) -> list[dict]:
    """Read bounded owner context for already-authorized Root Cause excerpts.

    Context pages are not new retrieval candidates and cannot widen the user's
    document/company scope. No machine vocabulary, headings or page constants
    are used to choose them: only each excerpt's own ordered page neighbourhood.
    """
    _ask_evidence_scope_where = runtime._ask_evidence_scope_where
    _assistant_core_candidate_evidence_text = runtime._assistant_core_candidate_evidence_text
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_scope_value = runtime._assistant_core_scope_value
    _db_conn = runtime._db_conn
    _is_structured_source_key = runtime._is_structured_source_key
    _safe_int = runtime._safe_int
    ranges: dict[str, set[int]] = {}
    for c in candidates[:14]:
        if _assistant_core_candidate_source_type(c) != "document":
            continue
        bdid = str(c.get("bubble_document_id") or "").strip()
        first = _safe_int(c.get("page_from"), 0)
        last = max(first, _safe_int(c.get("page_to"), first))
        if not bdid or first <= 0 or _is_structured_source_key(bdid):
            continue
        if bdid not in ranges and len(ranges) >= 6:
            continue
        ranges.setdefault(bdid, set()).update(range(max(1, first - 2), min(last, first + 2) + 1))
    pages: dict[tuple[str, int], str] = {}
    context_error = ""
    if ranges:
        where, params = _ask_evidence_scope_where(
            company_id=request.company_id, machine_id=request.machine_id,
            doc_ids=_assistant_core_scope_value(request, "document_ids"),
            bubble_document_id=_assistant_core_scope_value(request, "bubble_document_id"),
        )
        predicates = []
        for bdid, page_numbers in ranges.items():
            predicates.append("(bubble_document_id = %s AND page_number = ANY(%s))")
            params.extend([bdid, sorted(page_numbers)[:12]])
        conn = None
        try:
            conn = _db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT bubble_document_id, page_number, LEFT(COALESCE(text, ''), 6500) "
                    "FROM public.document_pages WHERE " + where + " AND (" +
                    " OR ".join(predicates) + ") ORDER BY bubble_document_id, page_number LIMIT 72",
                    params,
                )
                for doc, page, text in cur.fetchall():
                    key = (str(doc or ""), int(page or 0))
                    if key[0] in ranges and key[1] in ranges[key[0]]:
                        pages[key] = str(text or "")
        except Exception as exc:
            # Lack of owner context must remain visible; never infer the owner.
            context_error = type(exc).__name__
            print("ROOT_OWNER_CONTEXT_UNAVAILABLE", context_error)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as exc:
                    print("ROOT_OWNER_CONTEXT_CLOSE_ERROR", type(exc).__name__)
    records: list[dict] = []
    for c in candidates[:14]:
        cid = str(c.get("citation_id") or "").strip()
        text = _assistant_core_candidate_evidence_text(c)
        if not cid or not text:
            continue
        bdid = str(c.get("bubble_document_id") or "")
        first = max(1, _safe_int(c.get("page_from"), 1))
        last = max(first, _safe_int(c.get("page_to"), first))
        context_pages = [
            {"page_number": page, "text": pages[(bdid, page)],
             "read_limit_reached": len(pages[(bdid, page)]) >= 6500}
            for page in range(max(1, first - 2), min(last, first + 2) + 1)
            if pages.get((bdid, page))
        ]
        records.append({
            "citation_id": cid,
            "source_type": _assistant_core_candidate_source_type(c),
            "page_from": first, "page_to": last,
            "text": text,
            "context_pages": context_pages,
            "context_status": (
                "unavailable:" + context_error if context_error and bdid in ranges else
                "loaded" if context_pages else "no_neighbour_context"
            ),
        })
    return records




def read_neighbor_chunk_evidence(*, scope: ChunkReadScope, document_source: SourceIdentity,
        seed_inputs: tuple, current_allowed_sources: frozenset[SourceIdentity],
        limits: ChunkEvidenceLimits, runtime: ExpandWithNeighborChunksRuntime, radius: int = 1) -> SupplementalChunkRead:
    validate_anchors(scope=scope, anchors=(document_source,), current_allowed_sources=current_allowed_sources, limits=limits)
    records = checked_input_records(scope=scope, records=seed_inputs,
        current_allowed_sources=current_allowed_sources, limits=limits)
    if any(r.context.source != document_source for r in seed_inputs):
        raise SupplementalBindingError("neighbor seed belongs to a different source")
    capture = _ChunkSelectionCapture(scope, limits, "neighbor_chunks", snippet_chars=runtime.ASK_SNIPPET_CHARS)
    selected = _expand_with_neighbor_chunks_impl(scope.company_id, storage_key(document_source),
        [r.get("citation_id", "") for r in records], radius=radius, runtime=runtime, _chunk_capture=capture)
    return capture.finish(selected)


def read_enumeration_page_evidence(*, scope: ChunkReadScope, request: AssistantCoreRequest,
        retrieval: dict, candidate_inputs: tuple, current_allowed_sources: frozenset[SourceIdentity],
        limits: ChunkEvidenceLimits, runtime: AssistantCoreExpandEnumerationSectionsRuntime,
        max_documents: int = 4, page_radius: int = 3, max_pages: int = 18) -> PageEvidenceRead:
    require_request_scope(scope, request)
    records = checked_input_records(scope=scope, records=candidate_inputs,
        current_allowed_sources=current_allowed_sources, limits=limits)
    capture = _PageReadCapture(scope, limits, "enumeration_pages", snippet_chars=runtime.ASK_SNIPPET_CHARS,
        candidate_chars=runtime.V13_PAGE_TEXT_CHARS, max_queries=max_documents)
    selected = _assistant_core_expand_enumeration_sections_impl(request=request, retrieval=retrieval,
        candidates=records, max_documents=max_documents, page_radius=page_radius, max_pages=max_pages,
        runtime=runtime, _page_capture=capture)
    return capture.finish(selected)
