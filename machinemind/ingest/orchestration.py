"""Document, structured-source and indexing orchestration boundaries.

The production FastAPI routes remain in ``main.py`` with their certified names,
signatures and decorators.  Their behavior is delegated here through immutable,
late-bound runtime objects.  The module does not import ``main`` and does not own
routing, retrieval, source priority, ASK, Root Cause or Smart Diagnostic logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable


@dataclass(frozen=True)
class DocumentIngestRuntime:
    ai_internal_secret: str
    http_exception_cls: type[Exception]
    normalize_ai_scope: Callable[[Any], str]
    normalize_month_key: Callable[[Any], str]
    decimal_value: Callable[[Any], Decimal]
    month_usage: Callable[[str, str], dict]
    load_document_file: Callable[[Any, str], dict]
    effective_request_key: Callable[..., str]
    build_usage_event_id: Callable[[str, str, str], str]
    upsert_document_file: Callable[[str, str, str], None]
    detect_source_file_type: Callable[..., str]
    looks_like_xlsx_document: Callable[[bytes, str, str], bool]
    extract_pdf_pages: Callable[[bytes], list[str]]
    detect_repeated_headers_footers: Callable[[list[str]], tuple[set[str], set[str]]]
    upsert_cleaning_meta: Callable[[str, str, set[str], set[str]], None]
    remove_headers_footers_from_page: Callable[[str, set[str], set[str]], str]
    reflow_paragraphs: Callable[[str], str]
    maybe_remove_toc: Callable[[str], str]
    extract_xlsx_pages: Callable[..., list[str]]
    xlsx_error_cls: type[Exception]
    replace_document_pages: Callable[..., None]
    invalidate_company_knowledge: Callable[[str], Any]
    prepare_event: Callable[..., bool]
    enqueue_index_task: Callable[..., None]
    max_pdf_bytes: int
    max_xlsx_bytes: int
    min_page_chars: int
    min_text_chars_short: int
    min_text_chars: int
    min_pages_with_text_abs: int
    min_pages_with_text_pct: float
    xlsx_ingest_enabled: bool
    xlsx_min_text_chars: int
    chunk_target_chars: int
    chunk_overlap_chars: int
    pricing_version: str
    metering_version: str
    sha256_fn: Callable[[bytes], Any]
    ceil_fn: Callable[[float], int]
    log_fn: Callable[..., Any]


@dataclass(frozen=True)
class StructuredIngestRuntime:
    ai_internal_secret: str
    http_exception_cls: type[Exception]
    normalize_source_type: Callable[[Any], str]
    build_source_key: Callable[[str, str], str]
    compose_source_text: Callable[[Any], str]
    estimate_storage_bytes: Callable[[int], int]
    get_index_usage: Callable[..., dict]
    replace_document_pages: Callable[..., None]
    invalidate_company_knowledge: Callable[[str], Any]
    upsert_document_file: Callable[[str, str, str], None]
    index_request_cls: type
    index_document: Callable[..., dict]
    parent_procedure_source_key: Callable[[Any], str]
    upsert_structured_relation: Callable[..., bool]


@dataclass(frozen=True)
class IndexDocumentRuntime:
    ai_internal_secret: str
    http_exception_cls: type[Exception]
    normalize_ai_scope: Callable[[Any], str]
    get_cleaning_meta: Callable[[str, str], tuple[set[str], set[str]]]
    connect_db: Callable[[], Any]
    is_xlsx_page_text: Callable[[str], bool]
    chunk_xlsx_pages: Callable[..., list[dict]]
    chunk_sentences_with_pages: Callable[..., list[dict]]
    is_structured_source_key: Callable[[str], bool]
    collapse_structured_chunks: Callable[[list[dict]], list[dict]]
    strip_hf_from_chunk_text: Callable[[str, set[str], set[str]], str]
    get_table_columns: Callable[[Any, str], set[str]]
    embed_texts: Callable[[list[str]], list[list[float]]]
    vector_literal: Callable[[list[float]], str]
    invalidate_company_knowledge: Callable[[str], Any]
    chunk_target_chars: int
    chunk_overlap_chars: int
    chunk_min_chars: int
    embed_model: str
    search_text_fn: Callable[[str, str], Any]


def ingest_document(
    payload: Any,
    x_ai_internal_secret: str | None,
    *,
    runtime: DocumentIngestRuntime,
) -> dict:
    if not runtime.ai_internal_secret:
        raise runtime.http_exception_cls(
            status_code=500, detail="AI_INTERNAL_SECRET missing"
        )
    if (x_ai_internal_secret or "").strip() != runtime.ai_internal_secret:
        raise runtime.http_exception_cls(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()

    ingest_scope = runtime.normalize_ai_scope(payload.ai_scope)
    if ingest_scope == "document_ids":
        ingest_scope = "machine_all"

    if ingest_scope == "company_general":
        machine_id = ""
    elif not machine_id:
        raise runtime.http_exception_cls(status_code=400, detail="Missing machine_id")

    if not (company_id and bubble_document_id):
        raise runtime.http_exception_cls(
            status_code=400, detail="Missing company_id/bubble_document_id"
        )

    ingest_month_key_pre = runtime.normalize_month_key(payload.ingest_month_key)
    ingest_limit = max(
        Decimal("0"), runtime.decimal_value(payload.ingest_credits_limit_month)
    )
    ingest_enforced = bool(payload.ingest_credits_enforced)
    ingest_already_admitted = bool(payload.ingest_request_already_admitted)
    if ingest_enforced and ingest_limit > 0 and not ingest_already_admitted:
        usage_before = runtime.month_usage(company_id, ingest_month_key_pre)
        used_before = runtime.decimal_value(
            usage_before.get("ingest_credits_used_month")
        )
        if bool(usage_before.get("ledger_available")) and used_before >= ingest_limit:
            request_key_pre = str(
                payload.ingest_request_key or bubble_document_id
            ).strip()
            return {
                "ok": False,
                "status": "limit_exceeded",
                "reason": "PLAN_INGEST_CREDITS_LIMIT_EXCEEDED",
                "error_code": "PLAN_INGEST_CREDITS_LIMIT_EXCEEDED",
                "ai_quota_exceeded": True,
                "error": {
                    "code": "PLAN_INGEST_CREDITS_LIMIT_EXCEEDED",
                    "message": (
                        "Limite mensile di elaborazione AI documenti già raggiunto. "
                        "Il documento resta salvato in Bubble ma non viene indicizzato."
                    ),
                },
                "request_key": request_key_pre,
                "ingest_request_key": request_key_pre,
                "ingest_usage_event_id": "",
                "ingest_month_key": ingest_month_key_pre,
                "ingest_credits_actual": 0,
                "ingest_credits_used_month": float(used_before),
                "ingest_credits_limit_month": float(ingest_limit),
                "ingest_credits_remaining": 0,
                "ingest_pricing_version": runtime.pricing_version,
                "ingest_metering_status": "not_started_quota_blocked",
                "ingest_metering_version": runtime.metering_version,
                "ingest_usage": {},
            }

    loaded_file = runtime.load_document_file(payload, bubble_document_id)

    data = loaded_file["data"]
    url = loaded_file["url"]
    content_type = loaded_file["content_type"]
    content_disposition = loaded_file["content_disposition"]
    detected_filename = loaded_file["detected_filename"]
    detected_extension = loaded_file["detected_extension"]
    source_mode = loaded_file["source_mode"]
    file_sha256 = runtime.sha256_fn(data).hexdigest()
    ingest_month_key = runtime.normalize_month_key(payload.ingest_month_key)
    ingest_request_key = runtime.effective_request_key(
        company_id=company_id,
        bubble_document_id=bubble_document_id,
        requested_key=payload.ingest_request_key,
        file_sha256=file_sha256,
    )
    ingest_usage_event_id = runtime.build_usage_event_id(
        company_id, ingest_month_key, ingest_request_key
    )

    if url:
        runtime.upsert_document_file(company_id, bubble_document_id, url)

    source_file_type = runtime.detect_source_file_type(
        data,
        detected_extension,
        content_type,
        looks_like_xlsx_fn=runtime.looks_like_xlsx_document,
    )

    if not source_file_type:
        return {
            "ok": False,
            "error": {
                "code": "NOT_INDEXABLE",
                "message": "Documento non indicizzabile: formato file non supportato per ingest.",
            },
            "reason": "UNSUPPORTED_FILE_TYPE",
            "detected_content_type": content_type or None,
            "detected_filename": detected_filename or None,
            "detected_extension": detected_extension or None,
        }

    pages_text: list[str] = []
    pages_total = 0
    pages_with_text = 0
    text_chars = 0
    if source_file_type == "pdf":
        if len(data) > runtime.max_pdf_bytes:
            raise runtime.http_exception_cls(status_code=413, detail="PDF too large")

        try:
            raw_pages = runtime.extract_pdf_pages(data)
            pages_total = len(raw_pages)

            header_norm, footer_norm = runtime.detect_repeated_headers_footers(
                raw_pages
            )

            try:
                runtime.upsert_cleaning_meta(
                    company_id, bubble_document_id, header_norm, footer_norm
                )
            except Exception as exc:
                runtime.log_fn("CLEANING_META_UPSERT_FAIL", str(exc))

            for text in raw_pages:
                cleaned = runtime.remove_headers_footers_from_page(
                    text, header_norm, footer_norm
                )
                cleaned = runtime.reflow_paragraphs(cleaned)
                cleaned = runtime.maybe_remove_toc(cleaned)
                pages_text.append(cleaned)
                text_chars += len(cleaned)
                if len(cleaned) >= runtime.min_page_chars:
                    pages_with_text += 1
        except Exception:
            raise runtime.http_exception_cls(status_code=422, detail="PDF parse failed")

    else:
        if not runtime.xlsx_ingest_enabled:
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": "Documento non indicizzabile: supporto XLSX non ancora abilitato nel backend.",
                },
                "reason": "XLSX_INGEST_DISABLED",
                "detected_content_type": content_type or None,
                "detected_filename": detected_filename or None,
                "detected_extension": detected_extension or None,
            }

        if len(data) > runtime.max_xlsx_bytes:
            raise runtime.http_exception_cls(status_code=413, detail="XLSX too large")

        try:
            pages_text = runtime.extract_xlsx_pages(
                data, detected_filename=detected_filename
            )
            pages_total = len(pages_text)
            text_chars = sum(len(text or "") for text in pages_text)
            pages_with_text = sum(
                1 for text in pages_text if len(text or "") >= runtime.min_page_chars
            )
        except runtime.xlsx_error_cls as exc:
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": exc.message,
                    "detail": exc.detail or {},
                },
                "reason": exc.reason,
                "detected_content_type": content_type or None,
                "detected_filename": detected_filename or None,
                "detected_extension": detected_extension or None,
            }
        except Exception as exc:
            raise runtime.http_exception_cls(
                status_code=422,
                detail=f"XLSX parse failed: {str(exc)[:200]}",
            )

    if source_file_type == "xlsx":
        if pages_with_text < 1 or text_chars < max(1, runtime.xlsx_min_text_chars):
            reason = "LOW_TEXT_COVERAGE" if pages_with_text < 1 else "LOW_TEXT_CHARS"
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": "Documento XLSX non indicizzabile: testo leggibile insufficiente.",
                },
                "reason": reason,
                "pages_total": pages_total,
                "pages_with_text": pages_with_text,
                "pages_detected": pages_total,
                "text_chars": text_chars,
                "source_file_type": source_file_type,
            }
    elif pages_total <= 2:
        if pages_with_text < 1 or text_chars < runtime.min_text_chars_short:
            reason = "LOW_TEXT_COVERAGE" if pages_with_text < 1 else "LOW_TEXT_CHARS"
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": f"Documento non indicizzabile: troppo poco testo per {pages_total} pagina/e.",
                },
                "reason": reason,
                "pages_total": pages_total,
                "pages_with_text": pages_with_text,
                "pages_detected": pages_total,
                "text_chars": text_chars,
            }
    else:
        min_pages_required = max(
            runtime.min_pages_with_text_abs,
            int(runtime.ceil_fn(pages_total * runtime.min_pages_with_text_pct)),
        )
        if text_chars < runtime.min_text_chars or pages_with_text < min_pages_required:
            reason = (
                "LOW_TEXT_CHARS"
                if text_chars < runtime.min_text_chars
                else "LOW_TEXT_COVERAGE"
            )
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": "Documento non indicizzabile: testo insufficiente o troppo poco distribuito sulle pagine.",
                },
                "reason": reason,
                "pages_total": pages_total,
                "pages_with_text": pages_with_text,
                "pages_detected": pages_total,
                "text_chars": text_chars,
            }

    plan_chars_limit = int(payload.plan_embed_chars_limit_total or 0)
    plan_storage_limit = int(payload.plan_index_storage_limit_bytes or 0)

    effective_step = max(
        1,
        runtime.chunk_target_chars
        - min(runtime.chunk_overlap_chars, runtime.chunk_target_chars - 1),
    )
    est_chunks = int(runtime.ceil_fn(max(1, text_chars) / effective_step))

    bytes_per_char = 3
    bytes_per_chunk = 2000
    est_storage_bytes = int(
        text_chars * bytes_per_char + est_chunks * bytes_per_chunk
    )

    if plan_chars_limit > 0 and text_chars > plan_chars_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Documento troppo grande per il piano (limite caratteri indicizzabili).",
            },
            "reason": "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "limit_chars": plan_chars_limit,
        }

    if plan_storage_limit > 0 and est_storage_bytes > plan_storage_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Documento troppo grande per il piano (limite storage AI indicizzato).",
            },
            "reason": "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "est_storage_bytes": est_storage_bytes,
            "limit_storage_bytes": plan_storage_limit,
        }

    used_chars_total = int(payload.embed_chars_used_total or 0)
    used_storage_total = int(payload.index_storage_used_total or 0)
    prev_doc_chars = int(payload.doc_prev_embed_chars or 0)
    prev_doc_storage = int(payload.doc_prev_index_storage_bytes or 0)

    new_total_chars = used_chars_total - prev_doc_chars + int(text_chars)
    new_total_storage = used_storage_total - prev_doc_storage + int(
        est_storage_bytes
    )

    if plan_chars_limit > 0 and new_total_chars > plan_chars_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Limite totale caratteri AI superato per questa Company (used - prev + new).",
            },
            "reason": "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "limit_chars": plan_chars_limit,
            "used_chars_total": used_chars_total,
            "doc_prev_chars": prev_doc_chars,
            "new_total_chars": new_total_chars,
        }

    if plan_storage_limit > 0 and new_total_storage > plan_storage_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Limite totale storage AI superato per questa Company (used - prev + new).",
            },
            "reason": "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "est_storage_bytes": est_storage_bytes,
            "limit_storage_bytes": plan_storage_limit,
            "used_storage_total": used_storage_total,
            "doc_prev_storage_bytes": prev_doc_storage,
            "new_total_storage_bytes": new_total_storage,
        }

    runtime.replace_document_pages(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        pages_text=pages_text,
    )

    runtime.invalidate_company_knowledge(company_id)

    ledger_prepared = runtime.prepare_event(
        usage_event_id=ingest_usage_event_id,
        request_key=ingest_request_key,
        company_id=company_id,
        bubble_document_id=bubble_document_id,
        month_key=ingest_month_key,
    )

    runtime.enqueue_index_task(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        ingest_scope=ingest_scope,
        ingest_request_key=ingest_request_key,
        ingest_month_key=ingest_month_key,
        ingest_usage_event_id=ingest_usage_event_id,
    )

    month_usage_before = runtime.month_usage(company_id, ingest_month_key)
    return {
        "ok": True,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_detected": pages_total,
        "text_chars": text_chars,
        "est_storage_bytes": est_storage_bytes,
        "source_file_type": source_file_type,
        "request_key": ingest_request_key,
        "ingest_request_key": ingest_request_key,
        "ingest_usage_event_id": ingest_usage_event_id,
        "ingest_month_key": ingest_month_key,
        "ingest_credits_actual": 0,
        "ingest_credits_used_month": float(
            month_usage_before.get("ingest_credits_used_month") or 0
        ),
        "ingest_pricing_version": runtime.pricing_version,
        "ingest_metering_status": (
            "pending_async_index"
            if ledger_prepared
            else "pending_async_ledger_unavailable"
        ),
        "ingest_metering_version": runtime.metering_version,
        "ingest_usage": {
            "source_file_type": source_file_type,
            "file_sha256": file_sha256,
            "status": "queued",
        },
    }


def ingest_structured_source(
    payload: Any,
    x_ai_internal_secret: str | None,
    *,
    runtime: StructuredIngestRuntime,
) -> dict:
    if not runtime.ai_internal_secret:
        raise runtime.http_exception_cls(
            status_code=500, detail="AI_INTERNAL_SECRET missing"
        )
    if (x_ai_internal_secret or "").strip() != runtime.ai_internal_secret:
        raise runtime.http_exception_cls(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    source_id = (payload.source_id or "").strip()

    if not (company_id and machine_id and source_id):
        raise runtime.http_exception_cls(
            status_code=400,
            detail="Missing company_id/machine_id/source_id",
        )

    source_type = runtime.normalize_source_type(payload.source_type)
    source_key = runtime.build_source_key(source_type, source_id)

    text = runtime.compose_source_text(payload)
    text_chars = len(text)

    if text_chars <= 0:
        raise runtime.http_exception_cls(
            status_code=400, detail="Structured source text is empty"
        )

    source_url = (payload.source_url or "").strip()
    if source_url.startswith("//"):
        source_url = "https:" + source_url

    est_storage_bytes = runtime.estimate_storage_bytes(text_chars)

    plan_chars_limit = int(payload.plan_embed_chars_limit_total or 0)
    plan_storage_limit = int(payload.plan_index_storage_limit_bytes or 0)

    prev_usage = runtime.get_index_usage(
        company_id=company_id, bubble_document_id=source_key
    )
    company_usage = runtime.get_index_usage(company_id=company_id)

    new_total_chars = (
        int(company_usage["text_chars"])
        - int(prev_usage["text_chars"])
        + int(text_chars)
    )
    new_total_storage = (
        int(company_usage["est_storage_bytes"])
        - int(prev_usage["est_storage_bytes"])
        + int(est_storage_bytes)
    )

    if plan_chars_limit > 0 and text_chars > plan_chars_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Fonte testuale troppo grande per il piano (limite caratteri indicizzabili).",
            },
            "reason": "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "limit_chars": plan_chars_limit,
        }

    if plan_storage_limit > 0 and est_storage_bytes > plan_storage_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Fonte testuale troppo grande per il piano (limite storage AI indicizzato).",
            },
            "reason": "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "est_storage_bytes": est_storage_bytes,
            "limit_storage_bytes": plan_storage_limit,
        }

    if plan_chars_limit > 0 and new_total_chars > plan_chars_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Limite totale caratteri AI superato per questa Company.",
            },
            "reason": "PLAN_EMBED_CHARS_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "limit_chars": plan_chars_limit,
            "new_total_chars": new_total_chars,
            "prev_source_chars": int(prev_usage["text_chars"]),
            "company_chars_before": int(company_usage["text_chars"]),
        }

    if plan_storage_limit > 0 and new_total_storage > plan_storage_limit:
        return {
            "ok": False,
            "error": {
                "code": "LIMIT_EXCEEDED",
                "message": "Limite totale storage AI superato per questa Company.",
            },
            "reason": "PLAN_INDEX_STORAGE_LIMIT_EXCEEDED",
            "text_chars": text_chars,
            "est_storage_bytes": est_storage_bytes,
            "limit_storage_bytes": plan_storage_limit,
            "new_total_storage_bytes": new_total_storage,
            "prev_source_storage_bytes": int(prev_usage["est_storage_bytes"]),
            "company_storage_before": int(company_usage["est_storage_bytes"]),
        }

    runtime.replace_document_pages(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=source_key,
        pages_text=[text],
    )

    runtime.invalidate_company_knowledge(company_id)

    if source_url:
        runtime.upsert_document_file(company_id, source_key, source_url)

    index_result = runtime.index_document(
        runtime.index_request_cls(
            company_id=company_id,
            machine_id=machine_id,
            bubble_document_id=source_key,
            trace_id="structured_ingest",
        ),
        x_ai_internal_secret=runtime.ai_internal_secret,
    )

    parent_source_key = ""
    relation_written = False
    if source_type == "step":
        parent_source_key = runtime.parent_procedure_source_key(payload)
        if parent_source_key:
            relation_written = runtime.upsert_structured_relation(
                company_id=company_id,
                machine_id=machine_id,
                child_source_key=source_key,
                parent_source_key=parent_source_key,
                ordinal=payload.step_number,
                relation_source="bubble_ingest",
                metadata={
                    "parent_procedure_code": str(
                        payload.parent_procedure_code or ""
                    ),
                    "parent_procedure_title": str(
                        payload.parent_procedure_title or ""
                    ),
                },
            )

    return {
        "ok": True,
        "status": "indexed",
        "source_type": source_type,
        "source_key": source_key,
        "pages_total": 1,
        "pages_with_text": 1,
        "pages_detected": 1,
        "text_chars": text_chars,
        "est_storage_bytes": est_storage_bytes,
        "chunks_written": int(index_result.get("chunks_written") or 0),
        "parent_source_key": parent_source_key,
        "structured_relation_written": bool(relation_written),
    }


def index_document(
    payload: Any,
    x_ai_internal_secret: str | None,
    *,
    runtime: IndexDocumentRuntime,
) -> dict:
    if not runtime.ai_internal_secret:
        raise runtime.http_exception_cls(
            status_code=500, detail="AI_INTERNAL_SECRET missing"
        )
    if (x_ai_internal_secret or "").strip() != runtime.ai_internal_secret:
        raise runtime.http_exception_cls(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()
    trace_id = (payload.trace_id or "").strip() or None

    index_scope = runtime.normalize_ai_scope(payload.ai_scope)
    if index_scope == "document_ids":
        index_scope = "machine_all"

    if index_scope == "company_general":
        machine_id = ""
    elif not machine_id:
        raise runtime.http_exception_cls(status_code=400, detail="Missing machine_id")

    if not (company_id and bubble_document_id):
        raise runtime.http_exception_cls(
            status_code=400, detail="Missing company_id/bubble_document_id"
        )

    header_norm, footer_norm = runtime.get_cleaning_meta(
        company_id, bubble_document_id
    )

    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT page_number, text
                FROM document_pages
                WHERE company_id=%s AND bubble_document_id=%s
                ORDER BY page_number;
                """,
                (company_id, bubble_document_id),
            )
            page_rows = cur.fetchall()
            if not page_rows:
                return {
                    "ok": True,
                    "status": "indexed",
                    "company_id": company_id,
                    "machine_id": machine_id,
                    "bubble_document_id": bubble_document_id,
                    "trace_id": trace_id,
                    "chunks_written": 0,
                    "pages_detected": 0,
                    "note": "No pages found in document_pages for given ids",
                }

            pages = [(int(page_number), text or "") for page_number, text in page_rows]
            if pages and all(
                runtime.is_xlsx_page_text(text) for _, text in pages
            ):
                chunks = runtime.chunk_xlsx_pages(
                    pages=pages,
                    target_chars=runtime.chunk_target_chars,
                    min_chars=runtime.chunk_min_chars,
                    chunk_page_fn=runtime.chunk_sentences_with_pages,
                )
            else:
                chunks = runtime.chunk_sentences_with_pages(
                    pages=pages,
                    target_chars=runtime.chunk_target_chars,
                    overlap_chars=runtime.chunk_overlap_chars,
                    min_chars=runtime.chunk_min_chars,
                )

            source_is_structured = runtime.is_structured_source_key(
                bubble_document_id
            )
            if source_is_structured:
                chunks = runtime.collapse_structured_chunks(chunks)

            filtered_chunks = []
            for chunk in chunks:
                text = (chunk.get("chunk_text") or "").strip()

                if source_is_structured:
                    if len(text) < 20:
                        continue
                else:
                    if len(text) < 120:
                        continue

                if not runtime.search_text_fn(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text):
                    continue
                if len(text.split()) <= 4 and text.isupper():
                    continue

                filtered_chunks.append(chunk)

            chunks = filtered_chunks

            for chunk in chunks:
                chunk["chunk_text"] = runtime.strip_hf_from_chunk_text(
                    chunk.get("chunk_text", ""), header_norm, footer_norm
                )

            chunks = [
                chunk
                for chunk in chunks
                if (chunk.get("chunk_text") or "").strip()
            ]

            chunk_cols = runtime.get_table_columns(cur, "document_chunks")
            required = {
                "company_id",
                "machine_id",
                "bubble_document_id",
                "chunk_index",
                "page_from",
                "page_to",
                "chunk_text",
            }
            missing = sorted(list(required - set(chunk_cols)))
            if missing:
                raise runtime.http_exception_cls(
                    status_code=500,
                    detail=(
                        f"document_chunks missing columns: {missing}. "
                        f"Found: {sorted(chunk_cols)}"
                    ),
                )

            cur.execute(
                "DELETE FROM document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

            insert_query = """
                INSERT INTO document_chunks(
                    company_id, machine_id, bubble_document_id,
                    chunk_index, page_from, page_to,
                    chunk_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """

            for chunk in chunks:
                cur.execute(
                    insert_query,
                    (
                        company_id,
                        machine_id,
                        bubble_document_id,
                        int(chunk["chunk_index"]),
                        int(chunk["page_from"]),
                        int(chunk["page_to"]),
                        chunk["chunk_text"],
                    ),
                )

            batch_size = 32
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            for batch_start in range(0, len(chunks), batch_size):
                batch_texts = chunk_texts[batch_start : batch_start + batch_size]

                try:
                    vectors = runtime.embed_texts(batch_texts)
                except Exception as exc:
                    for batch_index in range(len(batch_texts)):
                        index = batch_start + batch_index
                        cur.execute(
                            """
                            UPDATE document_chunks
                            SET embedding_error=%s, embedding_model=%s, embedded_at=NOW()
                            WHERE company_id=%s AND bubble_document_id=%s AND chunk_index=%s;
                            """,
                            (
                                str(exc),
                                runtime.embed_model,
                                company_id,
                                bubble_document_id,
                                int(chunks[index]["chunk_index"]),
                            ),
                        )
                    continue

                for batch_index, vector in enumerate(vectors):
                    index = batch_start + batch_index
                    chunk_index = int(chunks[index]["chunk_index"])
                    vector_value = runtime.vector_literal(vector)

                    cur.execute(
                        """
                        UPDATE document_chunks
                        SET embedding = %s::vector,
                            embedding_model = %s,
                            embedded_at = NOW(),
                            embedding_error = NULL
                        WHERE company_id=%s AND bubble_document_id=%s AND chunk_index=%s;
                        """,
                        (
                            vector_value,
                            runtime.embed_model,
                            company_id,
                            bubble_document_id,
                            chunk_index,
                        ),
                    )

        conn.commit()
    finally:
        conn.close()

    runtime.invalidate_company_knowledge(company_id)

    return {
        "ok": True,
        "status": "indexed",
        "company_id": company_id,
        "machine_id": machine_id,
        "bubble_document_id": bubble_document_id,
        "trace_id": trace_id,
        "chunks_written": len(chunks),
        "pages_detected": len(pages),
        "chunk_target_chars": runtime.chunk_target_chars,
        "chunk_overlap_chars": runtime.chunk_overlap_chars,
    }
