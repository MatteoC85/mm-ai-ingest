import os
import re
import json
import hashlib
import unicodedata
from datetime import datetime
from typing import Any, Optional

import requests
import psycopg2

# Self-contained Phase 1D semantic normalizer. It reads only persisted electrical inventory.
DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_CHAT_URL = (os.environ.get("OPENAI_CHAT_URL") or "https://api.openai.com/v1/chat/completions").strip()

ELECTRICAL_SEMANTIC_ENABLED = (os.environ.get("MM_ELECTRICAL_SEMANTIC_ENABLED") or "0").strip() == "1"
ELECTRICAL_SEMANTIC_MODEL = (os.environ.get("MM_ELECTRICAL_SEMANTIC_MODEL") or "gpt-5.4-mini").strip()
ELECTRICAL_SEMANTIC_PROMPT_VERSION = (os.environ.get("MM_ELECTRICAL_SEMANTIC_PROMPT_VERSION") or "mm-electrical-semantic-v1").strip()
ELECTRICAL_SEMANTIC_BATCH_SIZE = int(os.environ.get("MM_ELECTRICAL_SEMANTIC_BATCH_SIZE", "24"))
ELECTRICAL_SEMANTIC_TEXT_SAMPLE_CHARS = int(os.environ.get("MM_ELECTRICAL_SEMANTIC_TEXT_SAMPLE_CHARS", "1400"))
ELECTRICAL_SEMANTIC_MIN_CONFIDENCE = float(os.environ.get("MM_ELECTRICAL_SEMANTIC_MIN_CONFIDENCE", "0.78"))
ELECTRICAL_SEMANTIC_TIMEOUT = int(os.environ.get("MM_ELECTRICAL_SEMANTIC_TIMEOUT_SECONDS", "120"))
ELECTRICAL_SEMANTIC_INPUT_USD_PER_MILLION = float(os.environ.get("MM_ELECTRICAL_SEMANTIC_INPUT_USD_PER_MILLION", "0"))
ELECTRICAL_SEMANTIC_OUTPUT_USD_PER_MILLION = float(os.environ.get("MM_ELECTRICAL_SEMANTIC_OUTPUT_USD_PER_MILLION", "0"))


def _db_conn():
    if not (DB_HOST and DB_USER and DB_PASSWORD):
        raise RuntimeError("DB env missing")
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def get_electrical_semantic_runtime_config() -> dict:
    return {
        "enabled": bool(ELECTRICAL_SEMANTIC_ENABLED),
        "model": ELECTRICAL_SEMANTIC_MODEL,
        "prompt_version": ELECTRICAL_SEMANTIC_PROMPT_VERSION,
        "batch_size": ELECTRICAL_SEMANTIC_BATCH_SIZE,
        "text_sample_chars": ELECTRICAL_SEMANTIC_TEXT_SAMPLE_CHARS,
        "min_confidence": ELECTRICAL_SEMANTIC_MIN_CONFIDENCE,
    }

ELECTRICAL_CANONICAL_PAGE_TYPES = (
    "cover",
    "index",
    "symbol_legend",
    "nameplate",
    "schematic",
    "plc_configuration",
    "safety_plc_configuration",
    "plc_io_table",
    "safety_io_table",
    "terminal_table",
    "bom_table",
    "layout",
    "network_layout",
    "unknown",
)

ELECTRICAL_CANONICAL_COVER_FIELDS = (
    "machine_code",
    "description",
    "machine_type",
    "scheme_number",
    "reference_bom",
    "order_number",
    "serial_number",
    "declared_sheet_count",
    "drawing_date",
    "operating_voltage",
    "auxiliary_voltage",
    "signal_voltage",
    "frequency",
    "nominal_current",
    "total_power",
    "protection_rating",
)


def _electrical_semantic_price(input_tokens: int, output_tokens: int) -> float:
    input_rate = max(0.0, float(ELECTRICAL_SEMANTIC_INPUT_USD_PER_MILLION or 0.0))
    output_rate = max(0.0, float(ELECTRICAL_SEMANTIC_OUTPUT_USD_PER_MILLION or 0.0))
    return round(
        (max(0, int(input_tokens or 0)) / 1_000_000.0) * input_rate
        + (max(0, int(output_tokens or 0)) / 1_000_000.0) * output_rate,
        6,
    )


def _electrical_openai_json_with_usage(
    messages: list[dict],
    *,
    json_schema: dict,
) -> tuple[dict, dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    payload = {
        "model": ELECTRICAL_SEMANTIC_MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(
        OPENAI_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=ELECTRICAL_SEMANTIC_TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI electrical semantic call failed: {r.status_code} {r.text[:1200]}")

    data = r.json()
    msg = (data.get("choices", [{}])[0].get("message", {}) or {})
    content = msg.get("content", "")
    if isinstance(content, list):
        text = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = str(content or "").strip()

    if not text:
        raise RuntimeError("OpenAI electrical semantic call returned empty content")

    try:
        parsed = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Electrical semantic JSON parse failed: {str(e)} | raw={text[:800]}")

    usage = data.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    return parsed, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _electrical_semantic_price(input_tokens, output_tokens),
        "model": ELECTRICAL_SEMANTIC_MODEL,
    }


def _electrical_semantic_fingerprint(task_type: str, payload: Any) -> tuple[str, str]:
    request_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    fingerprint_source = "|".join(
        [
            str(task_type or "").strip(),
            ELECTRICAL_SEMANTIC_PROMPT_VERSION,
            ELECTRICAL_SEMANTIC_MODEL,
            request_sha256,
        ]
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    return fingerprint, request_sha256


def _db_get_electrical_ai_artifact(version_id: int, fingerprint: str) -> Optional[dict]:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, response_json, input_tokens, output_tokens,
                       reasoning_tokens, cost_usd, model, prompt_version
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s AND fingerprint=%s
                LIMIT 1;
                """,
                (int(version_id), str(fingerprint)),
            )
            row = cur.fetchone()
            if not row:
                return None
            response_json = row[2]
            if isinstance(response_json, str):
                response_json = json.loads(response_json)
            return {
                "id": int(row[0]),
                "status": str(row[1] or ""),
                "response_json": response_json,
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "reasoning_tokens": int(row[5] or 0),
                "cost_usd": float(row[6] or 0),
                "model": str(row[7] or ""),
                "prompt_version": str(row[8] or ""),
            }
    finally:
        conn.close()


def _db_start_electrical_ai_artifact(
    *,
    version_id: int,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    page_id: Optional[int],
    fingerprint: str,
    task_type: str,
    request_sha256: str,
    request_metadata: dict,
) -> int:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.electrical_ai_artifacts(
                    version_id, company_id, machine_id, bubble_document_id,
                    page_id, fingerprint, task_type, region_hash,
                    model, prompt_version, request_sha256, request_metadata,
                    response_json, input_tokens, output_tokens, reasoning_tokens,
                    cost_usd, status, error_message, created_at, completed_at
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, NULL,
                    %s, %s, %s, %s::jsonb,
                    NULL, 0, 0, 0,
                    0, 'pending', NULL, NOW(), NULL
                )
                ON CONFLICT (version_id, fingerprint)
                DO UPDATE SET
                    page_id=EXCLUDED.page_id,
                    task_type=EXCLUDED.task_type,
                    model=EXCLUDED.model,
                    prompt_version=EXCLUDED.prompt_version,
                    request_sha256=EXCLUDED.request_sha256,
                    request_metadata=EXCLUDED.request_metadata,
                    status='pending',
                    error_message=NULL,
                    completed_at=NULL
                RETURNING id;
                """,
                (
                    int(version_id),
                    str(company_id),
                    str(machine_id),
                    str(bubble_document_id),
                    int(page_id) if page_id is not None else None,
                    str(fingerprint),
                    str(task_type),
                    ELECTRICAL_SEMANTIC_MODEL,
                    ELECTRICAL_SEMANTIC_PROMPT_VERSION,
                    str(request_sha256),
                    json.dumps(request_metadata, ensure_ascii=False),
                ),
            )
            artifact_id = int(cur.fetchone()[0])
        conn.commit()
        return artifact_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_complete_electrical_ai_artifact(
    *,
    artifact_id: int,
    response_json: dict,
    usage: dict,
    reused: bool = False,
) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.electrical_ai_artifacts
                SET response_json=%s::jsonb,
                    input_tokens=%s,
                    output_tokens=%s,
                    reasoning_tokens=%s,
                    cost_usd=%s,
                    status=%s,
                    error_message=NULL,
                    completed_at=COALESCE(completed_at, NOW())
                WHERE id=%s;
                """,
                (
                    json.dumps(response_json, ensure_ascii=False),
                    int(usage.get("input_tokens") or 0),
                    int(usage.get("output_tokens") or 0),
                    int(usage.get("reasoning_tokens") or 0),
                    float(usage.get("cost_usd") or 0.0),
                    "reused" if reused else "completed",
                    int(artifact_id),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_fail_electrical_ai_artifact(artifact_id: int, error_message: str) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.electrical_ai_artifacts
                SET status='failed',
                    error_message=%s,
                    completed_at=NOW()
                WHERE id=%s;
                """,
                (str(error_message or "")[:2000], int(artifact_id)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _electrical_cached_semantic_call(
    *,
    version_id: int,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    page_id: Optional[int],
    task_type: str,
    request_payload: dict,
    messages: list[dict],
    json_schema: dict,
    force: bool,
) -> tuple[dict, dict, bool, str]:
    fingerprint, request_sha256 = _electrical_semantic_fingerprint(task_type, request_payload)
    existing = _db_get_electrical_ai_artifact(version_id, fingerprint)

    if (
        not force
        and existing
        and existing.get("status") in {"completed", "reused"}
        and isinstance(existing.get("response_json"), dict)
    ):
        _db_complete_electrical_ai_artifact(
            artifact_id=int(existing["id"]),
            response_json=existing["response_json"],
            usage=existing,
            reused=True,
        )
        return existing["response_json"], existing, True, fingerprint

    artifact_id = _db_start_electrical_ai_artifact(
        version_id=version_id,
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        page_id=page_id,
        fingerprint=fingerprint,
        task_type=task_type,
        request_sha256=request_sha256,
        request_metadata={
            "prompt_version": ELECTRICAL_SEMANTIC_PROMPT_VERSION,
            "model": ELECTRICAL_SEMANTIC_MODEL,
            "request_payload": request_payload,
        },
    )

    try:
        response_json, usage = _electrical_openai_json_with_usage(
            messages,
            json_schema=json_schema,
        )
        _db_complete_electrical_ai_artifact(
            artifact_id=artifact_id,
            response_json=response_json,
            usage=usage,
            reused=False,
        )
        return response_json, usage, False, fingerprint
    except Exception as e:
        _db_fail_electrical_ai_artifact(artifact_id, str(e))
        raise


def _electrical_clean_sample(value: Any, max_chars: int) -> str:
    text = str(value or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _electrical_visual_rows(words: Any, max_rows: int = 80) -> list[dict]:
    valid: list[list[Any]] = []
    for item in words or []:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            x0, y0, x1, y1 = map(float, item[:4])
        except Exception:
            continue
        text = re.sub(r"\s+", " ", str(item[4] or "")).strip()
        if not text:
            continue
        valid.append([x0, y0, x1, y1, text])

    rows: list[dict] = []
    for word in sorted(valid, key=lambda w: (((w[1] + w[3]) / 2.0), w[0])):
        yc = (word[1] + word[3]) / 2.0
        target = None
        for row in rows:
            tolerance = max(3.0, float(row.get("median_height") or 0.0) * 0.35)
            if abs(float(row["y_center"]) - yc) <= tolerance:
                target = row
                break
        if target is None:
            target = {
                "y_center": yc,
                "median_height": max(1.0, word[3] - word[1]),
                "words": [],
            }
            rows.append(target)
        target["words"].append(word)
        heights = sorted(max(1.0, w[3] - w[1]) for w in target["words"])
        target["median_height"] = heights[len(heights) // 2]
        target["y_center"] = sum((w[1] + w[3]) / 2.0 for w in target["words"]) / len(target["words"])

    out: list[dict] = []
    for row_id, row in enumerate(sorted(rows, key=lambda r: float(r["y_center"])), start=1):
        ws = sorted(row["words"], key=lambda w: w[0])
        if not ws:
            continue
        gap_threshold = max(18.0, float(row.get("median_height") or 8.0) * 1.8)
        segments: list[dict] = []
        current: list[list[Any]] = []
        for word in ws:
            if current and (word[0] - current[-1][2]) > gap_threshold:
                segments.append({
                    "text": " ".join(w[4] for w in current),
                    "bbox": [
                        round(min(w[0] for w in current), 2),
                        round(min(w[1] for w in current), 2),
                        round(max(w[2] for w in current), 2),
                        round(max(w[3] for w in current), 2),
                    ],
                })
                current = []
            current.append(word)
        if current:
            segments.append({
                "text": " ".join(w[4] for w in current),
                "bbox": [
                    round(min(w[0] for w in current), 2),
                    round(min(w[1] for w in current), 2),
                    round(max(w[2] for w in current), 2),
                    round(max(w[3] for w in current), 2),
                ],
            })

        row_text = " | ".join(str(seg["text"]) for seg in segments)
        if not row_text:
            continue
        out.append({
            "row_id": row_id,
            "text_original": row_text[:1000],
            "segments": segments[:12],
        })
        if len(out) >= max_rows:
            break
    return out


def _electrical_page_semantic_schema() -> dict:
    return {
        "name": "electrical_page_semantic_normalization_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "page_id": {"type": "integer"},
                            "canonical_page_type": {
                                "type": "string",
                                "enum": list(ELECTRICAL_CANONICAL_PAGE_TYPES),
                            },
                            "confidence": {"type": "number"},
                            "language": {"type": "string"},
                            "evidence_basis": {
                                "type": "string",
                                "enum": [
                                    "outline",
                                    "page_text",
                                    "table_structure",
                                    "layout_geometry",
                                    "mixed",
                                    "insufficient",
                                ],
                            },
                        },
                        "required": [
                            "page_id",
                            "canonical_page_type",
                            "confidence",
                            "language",
                            "evidence_basis",
                        ],
                    },
                },
            },
            "required": ["pages"],
        },
    }


def _electrical_cover_semantic_schema() -> dict:
    return {
        "name": "electrical_cover_semantic_mapping_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "document_language": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "canonical_field": {
                                "type": "string",
                                "enum": list(ELECTRICAL_CANONICAL_COVER_FIELDS),
                            },
                            "row_id": {"type": "integer"},
                            "label_original": {"type": "string"},
                            "value_original": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "canonical_field",
                            "row_id",
                            "label_original",
                            "value_original",
                            "confidence",
                        ],
                    },
                },
            },
            "required": ["document_language", "fields"],
        },
    }


def _db_load_electrical_semantic_context(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
) -> dict:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            params: list[Any] = [str(company_id), str(machine_id), str(bubble_document_id)]
            version_clause = ""
            if version_id is not None:
                version_clause = " AND v.id=%s"
                params.append(int(version_id))

            cur.execute(
                f"""
                SELECT
                    d.id,
                    v.id,
                    v.version_no,
                    v.status,
                    v.metadata,
                    v.pdf_page_count,
                    v.declared_sheet_count
                FROM public.electrical_documents d
                JOIN public.electrical_versions v
                  ON v.electrical_document_id=d.id
                WHERE d.company_id=%s
                  AND d.machine_id=%s
                  AND d.bubble_document_id=%s
                  {version_clause}
                ORDER BY v.version_no DESC
                LIMIT 1;
                """,
                params,
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("Electrical version not found for semantic normalization")

            metadata = row[4] if isinstance(row[4], dict) else {}
            if metadata.get("inventory_status") != "ready":
                raise ValueError("Electrical inventory is not ready")

            cur.execute(
                """
                SELECT
                    id,
                    pdf_page_number,
                    sheet_code,
                    sheet_title,
                    group_code,
                    page_type,
                    classification_confidence,
                    structural_page_type,
                    structural_confidence,
                    is_vector_pdf,
                    has_internal_links,
                    raw_text,
                    text_spans_json,
                    links_json,
                    page_sha256
                FROM public.electrical_pages
                WHERE version_id=%s
                ORDER BY pdf_page_number;
                """,
                (int(row[1]),),
            )
            pages = []
            for p in cur.fetchall():
                pages.append({
                    "id": int(p[0]),
                    "pdf_page_number": int(p[1]),
                    "sheet_code": p[2],
                    "sheet_title": p[3],
                    "group_code": p[4],
                    "page_type": p[5],
                    "classification_confidence": float(p[6]) if p[6] is not None else None,
                    "structural_page_type": p[7],
                    "structural_confidence": float(p[8]) if p[8] is not None else None,
                    "is_vector_pdf": bool(p[9]) if p[9] is not None else None,
                    "has_internal_links": bool(p[10]),
                    "raw_text": str(p[11] or ""),
                    "text_spans_json": p[12] if isinstance(p[12], list) else [],
                    "links_json": p[13] if isinstance(p[13], list) else [],
                    "page_sha256": str(p[14] or ""),
                })

            if not pages:
                raise ValueError("Electrical version has no pages")

            return {
                "electrical_document_id": int(row[0]),
                "version_id": int(row[1]),
                "version_no": int(row[2]),
                "version_status": str(row[3] or ""),
                "metadata": metadata,
                "pdf_page_count": int(row[5] or len(pages)),
                "declared_sheet_count": int(row[6]) if row[6] is not None else None,
                "pages": pages,
            }
    finally:
        conn.close()


def _electrical_page_descriptors(pages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for page in pages:
        out.append({
            "page_id": int(page["id"]),
            "pdf_page_number": int(page["pdf_page_number"]),
            "sheet_code_original": str(page.get("sheet_code") or ""),
            "sheet_title_original": str(page.get("sheet_title") or ""),
            "group_title_original": str(page.get("group_code") or ""),
            "provisional_page_type": str(
                page.get("structural_page_type") or page.get("page_type") or "unknown"
            ),
            "provisional_confidence": float(
                page.get("structural_confidence")
                if page.get("structural_confidence") is not None
                else (page.get("classification_confidence") or 0.0)
            ),
            "word_count": len(page.get("text_spans_json") or []),
            "link_count": len(page.get("links_json") or []),
            "is_vector_pdf": page.get("is_vector_pdf"),
            "has_internal_links": bool(page.get("has_internal_links")),
            "text_sample_original": _electrical_clean_sample(
                page.get("raw_text"),
                ELECTRICAL_SEMANTIC_TEXT_SAMPLE_CHARS,
            ),
        })
    return out


def _electrical_normalize_page_batches(
    *,
    context: dict,
    force: bool,
) -> tuple[list[dict], dict]:
    descriptors = _electrical_page_descriptors(context["pages"])
    by_id = {int(page["id"]): page for page in context["pages"]}
    all_results: list[dict] = []
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "calls": 0,
        "reused_calls": 0,
        "fingerprints": [],
    }

    batch_size = max(1, min(50, int(ELECTRICAL_SEMANTIC_BATCH_SIZE or 24)))
    for offset in range(0, len(descriptors), batch_size):
        batch = descriptors[offset : offset + batch_size]
        request_payload = {
            "task": "page_semantic_normalization",
            "canonical_page_types": list(ELECTRICAL_CANONICAL_PAGE_TYPES),
            "pages": batch,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify pages from an industrial electrical-document package. "
                    "The source can be Italian, English, mixed, or another language. "
                    "Infer each page's FUNCTION semantically from the original outline, title, page text, "
                    "and structural signals. Do not use a fixed vocabulary or assume that a specific word "
                    "must be present. provisional_page_type is only a weak hint and can be wrong. "
                    "Return one result for every supplied page_id. Use unknown when evidence is insufficient. "
                    "Do not translate or alter identifiers, sheet codes, tags, or source text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(request_payload, ensure_ascii=False),
            },
        ]
        response, usage, reused, fingerprint = _electrical_cached_semantic_call(
            version_id=int(context["version_id"]),
            company_id=context["company_id"],
            machine_id=context["machine_id"],
            bubble_document_id=context["bubble_document_id"],
            page_id=None,
            task_type="page_semantic_normalization",
            request_payload=request_payload,
            messages=messages,
            json_schema=_electrical_page_semantic_schema(),
            force=force,
        )
        totals["calls"] += 1
        totals["reused_calls"] += 1 if reused else 0
        totals["fingerprints"].append(fingerprint)
        if not reused:
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["output_tokens"] += int(usage.get("output_tokens") or 0)
            totals["reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
            totals["cost_usd"] += float(usage.get("cost_usd") or 0.0)

        expected_ids = {int(item["page_id"]) for item in batch}
        seen_ids: set[int] = set()
        for item in response.get("pages") or []:
            try:
                page_id = int(item.get("page_id"))
            except Exception:
                continue
            if page_id not in expected_ids or page_id in seen_ids or page_id not in by_id:
                continue
            seen_ids.add(page_id)
            page_type = str(item.get("canonical_page_type") or "unknown")
            if page_type not in ELECTRICAL_CANONICAL_PAGE_TYPES:
                page_type = "unknown"
            confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
            all_results.append({
                "page_id": page_id,
                "canonical_page_type": page_type,
                "confidence": confidence,
                "language": str(item.get("language") or "unknown")[:32],
                "evidence_basis": str(item.get("evidence_basis") or "insufficient")[:64],
                "artifact_fingerprint": fingerprint,
            })

        for missing_id in sorted(expected_ids - seen_ids):
            all_results.append({
                "page_id": missing_id,
                "canonical_page_type": "unknown",
                "confidence": 0.0,
                "language": "unknown",
                "evidence_basis": "insufficient",
                "artifact_fingerprint": fingerprint,
            })

    totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
    return all_results, totals


def _normalize_for_semantic_validation(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _electrical_normalize_cover_fields(
    *,
    context: dict,
    force: bool,
) -> tuple[list[dict], dict]:
    first_page = min(context["pages"], key=lambda p: int(p["pdf_page_number"]))
    rows = _electrical_visual_rows(first_page.get("text_spans_json") or [])
    if not rows:
        return [], {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
            "reused_calls": 0,
            "fingerprints": [],
        }

    request_payload = {
        "task": "cover_field_semantic_mapping",
        "canonical_fields": list(ELECTRICAL_CANONICAL_COVER_FIELDS),
        "page_id": int(first_page["id"]),
        "pdf_page_number": int(first_page["pdf_page_number"]),
        "rows": rows,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Map label/value rows from the cover or title block of an industrial electrical document "
                "to canonical metadata fields. The source may be Italian, English, mixed, or another language. "
                "Infer meaning semantically; do not depend on a fixed keyword list. Preserve exact original "
                "label and value text. Map only values explicitly present in a supplied row. Do not infer, "
                "calculate, translate, normalize, or invent missing values. A row can contain multiple pairs."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request_payload, ensure_ascii=False),
        },
    ]
    response, usage, reused, fingerprint = _electrical_cached_semantic_call(
        version_id=int(context["version_id"]),
        company_id=context["company_id"],
        machine_id=context["machine_id"],
        bubble_document_id=context["bubble_document_id"],
        page_id=int(first_page["id"]),
        task_type="cover_field_semantic_mapping",
        request_payload=request_payload,
        messages=messages,
        json_schema=_electrical_cover_semantic_schema(),
        force=force,
    )

    rows_by_id = {int(row["row_id"]): row for row in rows}
    results: list[dict] = []
    seen_fields: set[str] = set()
    for item in response.get("fields") or []:
        canonical = str(item.get("canonical_field") or "").strip()
        if canonical not in ELECTRICAL_CANONICAL_COVER_FIELDS or canonical in seen_fields:
            continue
        try:
            row_id = int(item.get("row_id"))
        except Exception:
            continue
        row = rows_by_id.get(row_id)
        if not row:
            continue
        value_original = re.sub(r"\s+", " ", str(item.get("value_original") or "")).strip()
        label_original = re.sub(r"\s+", " ", str(item.get("label_original") or "")).strip()
        if not value_original:
            continue
        row_norm = _normalize_for_semantic_validation(row.get("text_original"))
        value_norm = _normalize_for_semantic_validation(value_original)
        if not value_norm or value_norm not in row_norm:
            continue
        confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
        seen_fields.add(canonical)
        results.append({
            "canonical_field": canonical,
            "row_id": row_id,
            "label_original": label_original[:300],
            "value_original": value_original[:500],
            "confidence": confidence,
            "artifact_fingerprint": fingerprint,
            "document_language": str(response.get("document_language") or "unknown")[:32],
        })

    return results, {
        "input_tokens": 0 if reused else int(usage.get("input_tokens") or 0),
        "output_tokens": 0 if reused else int(usage.get("output_tokens") or 0),
        "reasoning_tokens": 0 if reused else int(usage.get("reasoning_tokens") or 0),
        "cost_usd": 0.0 if reused else float(usage.get("cost_usd") or 0.0),
        "calls": 1,
        "reused_calls": 1 if reused else 0,
        "fingerprints": [fingerprint],
    }


def _db_apply_electrical_semantic_normalization(
    *,
    context: dict,
    page_results: list[dict],
    cover_results: list[dict],
) -> dict:
    version_id = int(context["version_id"])
    electrical_document_id = int(context["electrical_document_id"])
    company_id = str(context["company_id"])
    machine_id = str(context["machine_id"])
    bubble_document_id = str(context["bubble_document_id"])
    min_confidence = max(0.0, min(1.0, float(ELECTRICAL_SEMANTIC_MIN_CONFIDENCE)))

    page_by_id = {int(page["id"]): page for page in context["pages"]}
    ambiguous = 0
    semantic_type_counts: dict[str, int] = {}
    now_iso = datetime.now().astimezone().isoformat()

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            for result in page_results:
                page_id = int(result["page_id"])
                page = page_by_id.get(page_id)
                if not page:
                    continue
                semantic_type = str(result.get("canonical_page_type") or "unknown")
                confidence = float(result.get("confidence") or 0.0)
                accepted = semantic_type != "unknown" and confidence >= min_confidence
                final_type = semantic_type if accepted else "unknown"
                if not accepted:
                    ambiguous += 1

                semantic_type_counts[final_type] = int(semantic_type_counts.get(final_type, 0)) + 1
                classification_metadata = {
                    "semantic_prompt_version": ELECTRICAL_SEMANTIC_PROMPT_VERSION,
                    "semantic_model": ELECTRICAL_SEMANTIC_MODEL,
                    "semantic_accepted": accepted,
                    "semantic_evidence_basis": result.get("evidence_basis"),
                    "artifact_fingerprint": result.get("artifact_fingerprint"),
                    "normalized_at": now_iso,
                }

                cur.execute(
                    """
                    UPDATE public.electrical_pages
                    SET structural_page_type=COALESCE(structural_page_type, page_type),
                        structural_confidence=COALESCE(structural_confidence, classification_confidence),
                        semantic_page_type=%s,
                        semantic_confidence=%s,
                        page_type=%s,
                        classification_confidence=%s,
                        classification_method=%s,
                        classification_language=%s,
                        classification_metadata=COALESCE(classification_metadata, '{}'::jsonb) || %s::jsonb,
                        updated_at=NOW()
                    WHERE id=%s AND version_id=%s;
                    """,
                    (
                        semantic_type,
                        confidence,
                        final_type,
                        confidence if accepted else 0.0,
                        "openai_semantic_v1" if accepted else "semantic_review_required_v1",
                        str(result.get("language") or "unknown")[:32],
                        json.dumps(classification_metadata, ensure_ascii=False),
                        page_id,
                        version_id,
                    ),
                )

                issue_key = f"semantic-page-classification:{page_id}"
                if accepted:
                    cur.execute(
                        "DELETE FROM public.electrical_review_issues WHERE version_id=%s AND issue_key=%s;",
                        (version_id, issue_key),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO public.electrical_review_issues(
                            version_id, company_id, machine_id, bubble_document_id,
                            page_id, issue_key, issue_type, severity, status,
                            message, candidates_json, properties, created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, %s,
                            %s, %s, 'page_classification_ambiguous', 'warning', 'open',
                            %s, %s::jsonb, %s::jsonb, NOW(), NOW()
                        )
                        ON CONFLICT (version_id, issue_key)
                        DO UPDATE SET
                            status='open',
                            message=EXCLUDED.message,
                            candidates_json=EXCLUDED.candidates_json,
                            properties=EXCLUDED.properties,
                            updated_at=NOW();
                        """,
                        (
                            version_id,
                            company_id,
                            machine_id,
                            bubble_document_id,
                            page_id,
                            issue_key,
                            "Semantic page type is uncertain and requires review.",
                            json.dumps([result], ensure_ascii=False),
                            json.dumps({"minimum_confidence": min_confidence}, ensure_ascii=False),
                        ),
                    )

            semantic_cover_fields: dict[str, dict] = {}
            simple_cover_values: dict[str, Any] = {}
            declared_sheet_count_candidate: Optional[int] = None
            for item in cover_results:
                canonical = str(item.get("canonical_field") or "")
                confidence = float(item.get("confidence") or 0.0)
                semantic_cover_fields[canonical] = {
                    "value_original": item.get("value_original"),
                    "label_original": item.get("label_original"),
                    "confidence": confidence,
                    "row_id": item.get("row_id"),
                    "document_language": item.get("document_language"),
                    "artifact_fingerprint": item.get("artifact_fingerprint"),
                }
                if confidence >= min_confidence:
                    simple_cover_values[canonical] = item.get("value_original")
                    if canonical == "declared_sheet_count":
                        m = re.search(r"\d{1,4}", str(item.get("value_original") or ""))
                        if m:
                            declared_sheet_count_candidate = int(m.group(0))

            cur.execute(
                "SELECT metadata, declared_sheet_count FROM public.electrical_versions WHERE id=%s FOR UPDATE;",
                (version_id,),
            )
            version_row = cur.fetchone()
            existing_metadata = version_row[0] if version_row and isinstance(version_row[0], dict) else {}
            existing_declared = int(version_row[1]) if version_row and version_row[1] is not None else None
            existing_cover = existing_metadata.get("cover_fields") if isinstance(existing_metadata.get("cover_fields"), dict) else {}

            merged_cover = dict(existing_cover)
            for key, value in simple_cover_values.items():
                if key == "declared_sheet_count" and existing_declared is not None:
                    continue
                merged_cover[key] = value
            final_declared = existing_declared if existing_declared is not None else declared_sheet_count_candidate
            if final_declared is not None:
                merged_cover["declared_sheet_count"] = int(final_declared)
                if "declared_sheet_count_source" not in merged_cover:
                    merged_cover["declared_sheet_count_source"] = "semantic_cover_mapping"
                if "declared_sheet_count_confidence" not in merged_cover:
                    semantic_declared = semantic_cover_fields.get("declared_sheet_count") or {}
                    merged_cover["declared_sheet_count_confidence"] = float(semantic_declared.get("confidence") or 0.0)

            cur.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(reasoning_tokens), 0),
                    COALESCE(SUM(cost_usd), 0)
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s
                  AND status IN ('completed', 'reused');
                """,
                (version_id,),
            )
            token_row = cur.fetchone() or (0, 0, 0, 0)

            cur.execute(
                "SELECT COUNT(*) FROM public.electrical_review_issues WHERE version_id=%s AND status='open';",
                (version_id,),
            )
            review_issue_count = int(cur.fetchone()[0] or 0)
            next_status = "review_required" if review_issue_count > 0 else "queued"

            semantic_metadata = {
                "semantic_status": "review_required" if review_issue_count > 0 else "ready",
                "semantic_prompt_version": ELECTRICAL_SEMANTIC_PROMPT_VERSION,
                "semantic_model": ELECTRICAL_SEMANTIC_MODEL,
                "semantic_min_confidence": min_confidence,
                "semantic_page_type_counts": semantic_type_counts,
                "semantic_cover_fields": semantic_cover_fields,
                "semantic_completed_at": now_iso,
            }

            cur.execute(
                """
                UPDATE public.electrical_versions
                SET status=%s,
                    deterministic_only=false,
                    openai_used=true,
                    declared_sheet_count=%s,
                    review_issue_count=%s,
                    ai_input_tokens=%s,
                    ai_output_tokens=%s,
                    ai_cost_usd=%s,
                    metadata=COALESCE(metadata, '{}'::jsonb)
                             || %s::jsonb
                             || jsonb_build_object('cover_fields', %s::jsonb),
                    error_code=NULL,
                    error_message=NULL,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (
                    next_status,
                    final_declared,
                    review_issue_count,
                    int(token_row[0] or 0),
                    int(token_row[1] or 0),
                    float(token_row[3] or 0.0),
                    json.dumps(semantic_metadata, ensure_ascii=False),
                    json.dumps(merged_cover, ensure_ascii=False),
                    version_id,
                ),
            )
            cur.execute(
                """
                UPDATE public.electrical_documents
                SET index_status=%s,
                    last_error_code=NULL,
                    last_error_message=NULL,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (next_status, electrical_document_id),
            )

        conn.commit()
        return {
            "electrical_document_id": electrical_document_id,
            "electrical_version_id": version_id,
            "status": next_status,
            "semantic_status": "review_required" if review_issue_count > 0 else "ready",
            "pages_total": len(context["pages"]),
            "pages_ambiguous": ambiguous,
            "review_issue_count": review_issue_count,
            "semantic_page_type_counts": semantic_type_counts,
            "declared_sheet_count": final_declared,
            "cover_fields_mapped": len(semantic_cover_fields),
            "ai_input_tokens_total": int(token_row[0] or 0),
            "ai_output_tokens_total": int(token_row[1] or 0),
            "ai_reasoning_tokens_total": int(token_row[2] or 0),
            "ai_cost_usd_total": float(token_row[3] or 0.0),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def normalize_electrical_version_semantics(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
    force: bool,
) -> dict:
    if not ELECTRICAL_SEMANTIC_ENABLED:
        raise ValueError("MM_ELECTRICAL_SEMANTIC_ENABLED is disabled")

    context = _db_load_electrical_semantic_context(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        version_id=version_id,
    )
    context["company_id"] = str(company_id)
    context["machine_id"] = str(machine_id)
    context["bubble_document_id"] = str(bubble_document_id)

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.electrical_versions SET status='resolving_ambiguities', updated_at=NOW() WHERE id=%s;",
                (int(context["version_id"]),),
            )
            cur.execute(
                "UPDATE public.electrical_documents SET index_status='resolving_ambiguities', updated_at=NOW() WHERE id=%s;",
                (int(context["electrical_document_id"]),),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        page_results, page_usage = _electrical_normalize_page_batches(
            context=context,
            force=bool(force),
        )
        cover_results, cover_usage = _electrical_normalize_cover_fields(
            context=context,
            force=bool(force),
        )
        applied = _db_apply_electrical_semantic_normalization(
            context=context,
            page_results=page_results,
            cover_results=cover_results,
        )
        applied["calls"] = int(page_usage.get("calls") or 0) + int(cover_usage.get("calls") or 0)
        applied["reused_calls"] = int(page_usage.get("reused_calls") or 0) + int(cover_usage.get("reused_calls") or 0)
        applied["new_input_tokens"] = int(page_usage.get("input_tokens") or 0) + int(cover_usage.get("input_tokens") or 0)
        applied["new_output_tokens"] = int(page_usage.get("output_tokens") or 0) + int(cover_usage.get("output_tokens") or 0)
        applied["new_cost_usd"] = round(float(page_usage.get("cost_usd") or 0.0) + float(cover_usage.get("cost_usd") or 0.0), 6)
        return applied
    except Exception as e:
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.electrical_versions
                    SET status='review_required',
                        error_code='ELECTRICAL_SEMANTIC_FAILED',
                        error_message=%s,
                        metadata=COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                        updated_at=NOW()
                    WHERE id=%s;
                    """,
                    (
                        str(e)[:2000],
                        json.dumps({
                            "semantic_status": "failed",
                            "semantic_failed_at": datetime.now().astimezone().isoformat(),
                        }),
                        int(context["version_id"]),
                    ),
                )
                cur.execute(
                    """
                    UPDATE public.electrical_documents
                    SET index_status='review_required',
                        last_error_code='ELECTRICAL_SEMANTIC_FAILED',
                        last_error_message=%s,
                        updated_at=NOW()
                    WHERE id=%s;
                    """,
                    (str(e)[:2000], int(context["electrical_document_id"])),
                )
            conn.commit()
        finally:
            conn.close()
        raise
