import os
import re
import json
import hashlib
import unicodedata
from datetime import datetime
from typing import Any, Optional

import requests
import psycopg2

# Phase 2: one-time multilingual structured extraction from persisted electrical page inventory.
# The module never re-downloads the source PDF and never relies on fixed Italian/English keywords.

DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_CHAT_URL = (
    os.environ.get("OPENAI_CHAT_URL")
    or "https://api.openai.com/v1/chat/completions"
).strip()

ELECTRICAL_STRUCTURED_ENABLED = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_ENABLED") or "0"
).strip() == "1"
ELECTRICAL_STRUCTURED_MODEL = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_MODEL") or "gpt-5.4-mini"
).strip()
ELECTRICAL_STRUCTURED_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_PROMPT_VERSION")
    or "mm-electrical-structured-v1"
).strip()
ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION")
    or "mm-electrical-structured-materializer-v1"
).strip()
ELECTRICAL_STRUCTURED_MIN_CONFIDENCE = float(
    os.environ.get("MM_ELECTRICAL_STRUCTURED_MIN_CONFIDENCE", "0.78")
)
ELECTRICAL_STRUCTURED_TIMEOUT = int(
    os.environ.get("MM_ELECTRICAL_STRUCTURED_TIMEOUT_SECONDS", "180")
)
ELECTRICAL_STRUCTURED_MAX_WORDS_PER_PAGE = int(
    os.environ.get("MM_ELECTRICAL_STRUCTURED_MAX_WORDS_PER_PAGE", "5000")
)
ELECTRICAL_STRUCTURED_INPUT_USD_PER_MILLION = float(
    os.environ.get("MM_ELECTRICAL_STRUCTURED_INPUT_USD_PER_MILLION", "0")
)
ELECTRICAL_STRUCTURED_OUTPUT_USD_PER_MILLION = float(
    os.environ.get("MM_ELECTRICAL_STRUCTURED_OUTPUT_USD_PER_MILLION", "0")
)

ELIGIBLE_PAGE_TYPES = {
    "plc_io_table",
    "safety_io_table",
    "terminal_table",
    "bom_table",
}

IO_TYPES = {
    "digital_input",
    "digital_output",
    "analog_input",
    "analog_output",
    "safety_input",
    "safety_output",
    "mixed",
    "other",
}


def _db_conn():
    if not (DB_HOST and DB_USER and DB_PASSWORD):
        raise RuntimeError("DB env missing")
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def get_electrical_structured_runtime_config() -> dict:
    return {
        "enabled": bool(ELECTRICAL_STRUCTURED_ENABLED),
        "model": ELECTRICAL_STRUCTURED_MODEL,
        "prompt_version": ELECTRICAL_STRUCTURED_PROMPT_VERSION,
        "materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
        "min_confidence": ELECTRICAL_STRUCTURED_MIN_CONFIDENCE,
        "max_words_per_page": ELECTRICAL_STRUCTURED_MAX_WORDS_PER_PAGE,
    }


def _price(input_tokens: int, output_tokens: int) -> float:
    return round(
        max(0, int(input_tokens or 0))
        / 1_000_000.0
        * max(0.0, ELECTRICAL_STRUCTURED_INPUT_USD_PER_MILLION)
        + max(0, int(output_tokens or 0))
        / 1_000_000.0
        * max(0.0, ELECTRICAL_STRUCTURED_OUTPUT_USD_PER_MILLION),
        6,
    )


def _openai_json_with_usage(
    messages: list[dict],
    *,
    json_schema: dict,
) -> tuple[dict, dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    payload = {
        "model": ELECTRICAL_STRUCTURED_MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }
    r = requests.post(
        OPENAI_CHAT_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=ELECTRICAL_STRUCTURED_TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"OpenAI structured electrical call failed: {r.status_code} {r.text[:1400]}"
        )

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
        raise RuntimeError("OpenAI structured electrical call returned empty content")

    try:
        parsed = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            f"Structured electrical JSON parse failed: {str(e)} | raw={text[:900]}"
        )

    usage = data.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    return parsed, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _price(input_tokens, output_tokens),
        "model": ELECTRICAL_STRUCTURED_MODEL,
    }


def _fingerprint(task_type: str, payload: Any) -> tuple[str, str]:
    request_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    raw = "|".join(
        [
            str(task_type or "").strip(),
            ELECTRICAL_STRUCTURED_PROMPT_VERSION,
            ELECTRICAL_STRUCTURED_MODEL,
            request_sha256,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), request_sha256


def _db_get_artifact(version_id: int, fingerprint: str) -> Optional[dict]:
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


def _db_start_artifact(
    *,
    version_id: int,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    page_id: int,
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
                    int(page_id),
                    str(fingerprint),
                    str(task_type),
                    ELECTRICAL_STRUCTURED_MODEL,
                    ELECTRICAL_STRUCTURED_PROMPT_VERSION,
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


def _db_complete_artifact(
    *,
    artifact_id: int,
    response_json: dict,
    usage: dict,
    reused: bool,
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


def _db_fail_artifact(artifact_id: int, error_message: str) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.electrical_ai_artifacts
                SET status='failed', error_message=%s, completed_at=NOW()
                WHERE id=%s;
                """,
                (str(error_message or "")[:2000], int(artifact_id)),
            )
        conn.commit()
    finally:
        conn.close()


def _cached_call(
    *,
    context: dict,
    page: dict,
    task_type: str,
    request_payload: dict,
    messages: list[dict],
    json_schema: dict,
    force: bool,
) -> tuple[dict, dict, bool, str]:
    fingerprint, request_sha256 = _fingerprint(task_type, request_payload)
    existing = _db_get_artifact(int(context["version_id"]), fingerprint)
    if (
        not force
        and existing
        and existing.get("response_json")
        and existing.get("status") in {"completed", "reused"}
    ):
        usage = {
            "input_tokens": int(existing.get("input_tokens") or 0),
            "output_tokens": int(existing.get("output_tokens") or 0),
            "reasoning_tokens": int(existing.get("reasoning_tokens") or 0),
            "cost_usd": float(existing.get("cost_usd") or 0.0),
            "model": str(existing.get("model") or ELECTRICAL_STRUCTURED_MODEL),
        }
        _db_complete_artifact(
            artifact_id=int(existing["id"]),
            response_json=existing["response_json"],
            usage=usage,
            reused=True,
        )
        return existing["response_json"], usage, True, fingerprint

    artifact_id = _db_start_artifact(
        version_id=int(context["version_id"]),
        company_id=context["company_id"],
        machine_id=context["machine_id"],
        bubble_document_id=context["bubble_document_id"],
        page_id=int(page["id"]),
        fingerprint=fingerprint,
        task_type=task_type,
        request_sha256=request_sha256,
        request_metadata={
            "pdf_page_number": int(page["pdf_page_number"]),
            "sheet_code": str(page.get("sheet_code") or ""),
            "page_type": str(page.get("page_type") or ""),
            "materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
        },
    )
    try:
        response, usage = _openai_json_with_usage(messages, json_schema=json_schema)
        _db_complete_artifact(
            artifact_id=artifact_id,
            response_json=response,
            usage=usage,
            reused=False,
        )
        return response, usage, False, fingerprint
    except Exception as e:
        _db_fail_artifact(artifact_id, str(e))
        raise


def _load_context(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
    page_types: Optional[list[str]],
) -> dict:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            params: list[Any] = [company_id, machine_id, bubble_document_id]
            version_clause = ""
            if version_id is not None:
                version_clause = "AND v.id=%s"
                params.append(int(version_id))
            cur.execute(
                f"""
                SELECT d.id, v.id, v.version_no, v.status, v.metadata,
                       v.pdf_page_count, v.declared_sheet_count
                FROM public.electrical_documents d
                JOIN public.electrical_versions v
                  ON v.electrical_document_id=d.id
                 AND v.company_id=d.company_id
                 AND v.machine_id=d.machine_id
                 AND v.bubble_document_id=d.bubble_document_id
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
                raise ValueError("Electrical version not found for supplied scope")
            metadata = row[4]
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")

            requested_page_types = sorted(set(page_types or ELIGIBLE_PAGE_TYPES))
            invalid_page_types = set(requested_page_types) - ELIGIBLE_PAGE_TYPES
            if invalid_page_types:
                raise ValueError(
                    "Unsupported structured page_types: "
                    + ", ".join(sorted(invalid_page_types))
                )
            cur.execute(
                """
                SELECT id, pdf_page_number, sheet_code, sheet_title, group_code,
                       page_type, page_width_pt, page_height_pt, page_sha256,
                       raw_text, text_spans_json, classification_language,
                       semantic_confidence, classification_metadata
                FROM public.electrical_pages
                WHERE version_id=%s
                  AND page_type = ANY(%s)
                ORDER BY pdf_page_number;
                """,
                (int(row[1]), requested_page_types),
            )
            pages = []
            for p in cur.fetchall():
                words = p[10]
                class_meta = p[13]
                if isinstance(words, str):
                    words = json.loads(words or "[]")
                if isinstance(class_meta, str):
                    class_meta = json.loads(class_meta or "{}")
                pages.append(
                    {
                        "id": int(p[0]),
                        "pdf_page_number": int(p[1]),
                        "sheet_code": str(p[2] or ""),
                        "sheet_title": str(p[3] or ""),
                        "group_code": str(p[4] or ""),
                        "page_type": str(p[5] or "unknown"),
                        "page_width_pt": float(p[6] or 1.0),
                        "page_height_pt": float(p[7] or 1.0),
                        "page_sha256": str(p[8] or ""),
                        "raw_text": str(p[9] or ""),
                        "words": list(words or []),
                        "classification_language": str(p[11] or "unknown"),
                        "semantic_confidence": float(p[12] or 0.0),
                        "classification_metadata": class_meta or {},
                    }
                )

            return {
                "electrical_document_id": int(row[0]),
                "version_id": int(row[1]),
                "version_no": int(row[2]),
                "version_status": str(row[3] or ""),
                "metadata": metadata or {},
                "pdf_page_count": int(row[5] or 0),
                "declared_sheet_count": int(row[6]) if row[6] is not None else None,
                "pages": pages,
            }
    finally:
        conn.close()


def _clean_text(value: Any, max_len: int = 1000) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _canonical_key_part(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:100] or "none"


def _page_payload(page: dict) -> tuple[dict, dict[int, dict]]:
    width = max(1.0, float(page.get("page_width_pt") or 1.0))
    height = max(1.0, float(page.get("page_height_pt") or 1.0))
    words = list(page.get("words") or [])[: max(1, ELECTRICAL_STRUCTURED_MAX_WORDS_PER_PAGE)]
    out_words: list[list[Any]] = []
    word_map: dict[int, dict] = {}
    for idx, w in enumerate(words, start=1):
        if not isinstance(w, (list, tuple)) or len(w) < 5:
            continue
        try:
            x0, y0, x1, y1 = [float(w[i]) for i in range(4)]
        except Exception:
            continue
        text = str(w[4] or "").replace("\x00", "")
        if not text.strip():
            continue
        block_no = int(w[5] or 0) if len(w) > 5 else 0
        line_no = int(w[6] or 0) if len(w) > 6 else 0
        word_no = int(w[7] or 0) if len(w) > 7 else 0
        nx0 = int(round(max(0.0, min(1.0, x0 / width)) * 1000))
        ny0 = int(round(max(0.0, min(1.0, y0 / height)) * 1000))
        nx1 = int(round(max(0.0, min(1.0, x1 / width)) * 1000))
        ny1 = int(round(max(0.0, min(1.0, y1 / height)) * 1000))
        out_words.append([idx, nx0, ny0, nx1, ny1, text, block_no, line_no, word_no])
        word_map[idx] = {
            "id": idx,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "text": text,
            "block_no": block_no,
            "line_no": line_no,
            "word_no": word_no,
        }
    return {
        "page_id": int(page["id"]),
        "pdf_page_number": int(page["pdf_page_number"]),
        "sheet_code_original": str(page.get("sheet_code") or ""),
        "sheet_title_original": str(page.get("sheet_title") or ""),
        "group_title_original": str(page.get("group_code") or ""),
        "canonical_page_type": str(page.get("page_type") or "unknown"),
        "classification_language_hint": str(page.get("classification_language") or "unknown"),
        "page_sha256": str(page.get("page_sha256") or ""),
        "coordinate_system": "normalized_0_1000",
        "word_format": [
            "word_id",
            "x0",
            "y0",
            "x1",
            "y1",
            "text_original",
            "block_no",
            "line_no",
            "word_no",
        ],
        "words": out_words,
    }, word_map


def _source_evidence(
    source_word_ids: Any,
    word_map: dict[int, dict],
) -> Optional[dict]:
    ids: list[int] = []
    seen: set[int] = set()
    for raw in source_word_ids or []:
        try:
            wid = int(raw)
        except Exception:
            continue
        if wid not in word_map or wid in seen:
            continue
        seen.add(wid)
        ids.append(wid)
    if not ids:
        return None
    words = [word_map[i] for i in ids]
    words.sort(key=lambda w: (w["y0"], w["x0"], w["id"]))
    source_text = _clean_text(" ".join(w["text"] for w in words), 5000)
    return {
        "source_word_ids": ids,
        "source_text": source_text,
        "x0": min(w["x0"] for w in words),
        "y0": min(w["y0"] for w in words),
        "x1": max(w["x1"] for w in words),
        "y1": max(w["y1"] for w in words),
    }


def _common_issue_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issue_type": {"type": "string"},
            "message": {"type": "string"},
            "source_word_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 120,
            },
            "confidence": {"type": "number"},
        },
        "required": ["issue_type", "message", "source_word_ids", "confidence"],
    }


def _io_schema() -> dict:
    return {
        "name": "electrical_io_page_extraction_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "language": {"type": "string"},
                "tables": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "table_label_original": {"type": "string"},
                            "module_tag": {"type": "string"},
                            "module_model": {"type": "string"},
                            "io_type": {"type": "string", "enum": sorted(IO_TYPES)},
                            "is_safety": {"type": "boolean"},
                            "table_source_word_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 120,
                            },
                            "confidence": {"type": "number"},
                            "rows": {
                                "type": "array",
                                "maxItems": 256,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "channel_ref": {"type": "string"},
                                        "plc_address": {"type": "string"},
                                        "wire_reference": {"type": "string"},
                                        "terminal_reference": {"type": "string"},
                                        "signal_name": {"type": "string"},
                                        "description_original": {"type": "string"},
                                        "expected_normal_state": {"type": "string"},
                                        "is_placeholder": {"type": "boolean"},
                                        "source_word_ids": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "maxItems": 100,
                                        },
                                        "confidence": {"type": "number"},
                                    },
                                    "required": [
                                        "channel_ref",
                                        "plc_address",
                                        "wire_reference",
                                        "terminal_reference",
                                        "signal_name",
                                        "description_original",
                                        "expected_normal_state",
                                        "is_placeholder",
                                        "source_word_ids",
                                        "confidence",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "table_label_original",
                            "module_tag",
                            "module_model",
                            "io_type",
                            "is_safety",
                            "table_source_word_ids",
                            "confidence",
                            "rows",
                        ],
                    },
                },
                "issues": {"type": "array", "maxItems": 100, "items": _common_issue_schema()},
            },
            "required": ["page_id", "language", "tables", "issues"],
        },
    }


def _terminal_schema() -> dict:
    return {
        "name": "electrical_terminal_page_extraction_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "language": {"type": "string"},
                "tables": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "table_label_original": {"type": "string"},
                            "strip_tag": {"type": "string"},
                            "table_source_word_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 120,
                            },
                            "confidence": {"type": "number"},
                            "rows": {
                                "type": "array",
                                "maxItems": 512,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "terminal_number": {"type": "string"},
                                        "level_ref": {"type": "string"},
                                        "side_a_origin": {"type": "string"},
                                        "side_b_destination": {"type": "string"},
                                        "wire_number": {"type": "string"},
                                        "cable_reference": {"type": "string"},
                                        "potential": {"type": "string"},
                                        "conductor_color": {"type": "string"},
                                        "conductor_cross_section": {"type": "string"},
                                        "description_original": {"type": "string"},
                                        "source_word_ids": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "maxItems": 120,
                                        },
                                        "confidence": {"type": "number"},
                                    },
                                    "required": [
                                        "terminal_number",
                                        "level_ref",
                                        "side_a_origin",
                                        "side_b_destination",
                                        "wire_number",
                                        "cable_reference",
                                        "potential",
                                        "conductor_color",
                                        "conductor_cross_section",
                                        "description_original",
                                        "source_word_ids",
                                        "confidence",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "table_label_original",
                            "strip_tag",
                            "table_source_word_ids",
                            "confidence",
                            "rows",
                        ],
                    },
                },
                "issues": {"type": "array", "maxItems": 100, "items": _common_issue_schema()},
            },
            "required": ["page_id", "language", "tables", "issues"],
        },
    }


def _bom_schema() -> dict:
    return {
        "name": "electrical_bom_page_extraction_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "language": {"type": "string"},
                "tables": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "table_label_original": {"type": "string"},
                            "table_source_word_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 120,
                            },
                            "confidence": {"type": "number"},
                            "rows": {
                                "type": "array",
                                "maxItems": 512,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "item_position": {"type": "string"},
                                        "component_tag": {"type": "string"},
                                        "quantity_text": {"type": "string"},
                                        "unit": {"type": "string"},
                                        "manufacturer": {"type": "string"},
                                        "part_number": {"type": "string"},
                                        "description_original": {"type": "string"},
                                        "source_word_ids": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "maxItems": 160,
                                        },
                                        "confidence": {"type": "number"},
                                    },
                                    "required": [
                                        "item_position",
                                        "component_tag",
                                        "quantity_text",
                                        "unit",
                                        "manufacturer",
                                        "part_number",
                                        "description_original",
                                        "source_word_ids",
                                        "confidence",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "table_label_original",
                            "table_source_word_ids",
                            "confidence",
                            "rows",
                        ],
                    },
                },
                "issues": {"type": "array", "maxItems": 100, "items": _common_issue_schema()},
            },
            "required": ["page_id", "language", "tables", "issues"],
        },
    }


def _prompt_for_page(page_type: str, request_payload: dict) -> tuple[list[dict], dict, str]:
    common = (
        "You extract structured records from an industrial electrical-document page. "
        "The source may be Italian, English, mixed, or any other language. "
        "Infer table meaning from layout, repeated row/column structure, identifiers, values, "
        "titles, and the relationships among cells. Do not require or depend on a fixed vocabulary. "
        "Preserve original component tags, addresses, channel identifiers, wire references, part numbers, "
        "manufacturer names, descriptions, and language. Map their semantic roles into the canonical fields. "
        "Every extracted row must cite source_word_ids from the supplied word list. "
        "Never invent a value and never copy words from unrelated title blocks or page borders into data rows. "
        "When evidence is insufficient, omit the row and report an issue. Empty canonical fields must be empty strings. "
        "Coordinates are normalized from 0 to 1000."
    )
    if page_type in {"plc_io_table", "safety_io_table"}:
        extra = (
            "The page may contain multiple PLC or safety modules. Identify each module/table independently. "
            "Each repeated channel or pin line is one row. Distinguish inputs from outputs semantically and structurally. "
            "Unused or placeholder channels may be returned with is_placeholder=true. "
            "Do not treat a circuit schematic as an I/O table merely because it mentions PLC or safety."
        )
        schema = _io_schema()
        task_type = "structured_io_page"
    elif page_type == "terminal_table":
        extra = (
            "The page may contain one or more terminal strips arranged horizontally or vertically. "
            "Each physical terminal number is one row. Determine origin, destination, wire/potential, cable, color, "
            "cross-section, and descriptions only when supported. Keep strip tags and terminal numbers exactly."
        )
        schema = _terminal_schema()
        task_type = "structured_terminal_page"
    elif page_type == "bom_table":
        extra = (
            "Extract one BOM/material row per listed component or article. Reconstruct descriptions split across visual cells, "
            "but keep manufacturer, part number, component tag, quantity, and unit separate. "
            "Do not merge distinct consecutive BOM rows."
        )
        schema = _bom_schema()
        task_type = "structured_bom_page"
    else:
        raise ValueError(f"Unsupported structured page type: {page_type}")

    return [
        {"role": "system", "content": common + " " + extra},
        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
    ], schema, task_type


def _clamp_conf(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except Exception:
        return 0.0


def _quantity_numeric(quantity_text: str) -> Optional[float]:
    s = str(quantity_text or "").strip()
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:[\.,]\d+)?", s)
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", "."))
        return value if value >= 0 else None
    except Exception:
        return None


def _issue_record(
    *,
    context: dict,
    page: dict,
    issue_type: str,
    message: str,
    source_word_ids: Any,
    word_map: dict[int, dict],
    confidence: float,
    sequence_no: int,
    properties: Optional[dict] = None,
) -> dict:
    evidence = _source_evidence(source_word_ids, word_map)
    return {
        "issue_key": (
            f"structured:{int(page['pdf_page_number'])}:"
            f"{_canonical_key_part(issue_type)}:{sequence_no}"
        ),
        "issue_type": f"structured_{_canonical_key_part(issue_type)}",
        "severity": "warning" if confidence >= 0.5 else "high",
        "message": _clean_text(message or issue_type, 1200),
        "candidates_json": [],
        "properties": {
            "phase": "structured_v1",
            "page_type": page.get("page_type"),
            "pdf_page_number": page.get("pdf_page_number"),
            "source_word_ids": (evidence or {}).get("source_word_ids", []),
            "source_text": (evidence or {}).get("source_text", ""),
            "confidence": confidence,
            **(properties or {}),
        },
    }


def _materialize_io(
    *,
    context: dict,
    page: dict,
    response: dict,
    word_map: dict[int, dict],
    fingerprint: str,
) -> tuple[list[dict], list[dict]]:
    rows_out: list[dict] = []
    issues: list[dict] = []
    issue_no = 0
    row_no = 0
    language = _clean_text(response.get("language"), 32) or "unknown"

    for table_index, table in enumerate(response.get("tables") or [], start=1):
        module_tag = _clean_text(table.get("module_tag"), 160)
        module_model = _clean_text(table.get("module_model"), 200)
        table_label = _clean_text(table.get("table_label_original"), 500)
        io_type = str(table.get("io_type") or "other")
        if io_type not in IO_TYPES:
            io_type = "other"
        is_safety = bool(table.get("is_safety")) or io_type.startswith("safety_")
        table_conf = _clamp_conf(table.get("confidence"))

        for row in table.get("rows") or []:
            row_no += 1
            evidence = _source_evidence(row.get("source_word_ids"), word_map)
            confidence = min(table_conf or 1.0, _clamp_conf(row.get("confidence")))
            channel_ref = _clean_text(row.get("channel_ref"), 120)
            wire_ref = _clean_text(row.get("wire_reference"), 180)
            terminal_ref = _clean_text(row.get("terminal_reference"), 180)
            description = _clean_text(row.get("description_original"), 1000)
            signal_name = _clean_text(row.get("signal_name"), 500)
            is_placeholder = bool(row.get("is_placeholder"))

            invalid_reason = ""
            if not evidence:
                invalid_reason = "missing_valid_source_word_ids"
            elif confidence < ELECTRICAL_STRUCTURED_MIN_CONFIDENCE:
                invalid_reason = "row_confidence_below_threshold"
            elif not module_tag:
                invalid_reason = "missing_module_tag"
            elif not channel_ref:
                invalid_reason = "missing_channel_ref"
            elif not is_placeholder and not any([description, signal_name, wire_ref, terminal_ref]):
                invalid_reason = "row_has_no_supported_signal_content"

            if invalid_reason:
                issue_no += 1
                issues.append(
                    _issue_record(
                        context=context,
                        page=page,
                        issue_type=invalid_reason,
                        message=f"I/O row was not materialized: {invalid_reason}",
                        source_word_ids=row.get("source_word_ids"),
                        word_map=word_map,
                        confidence=confidence,
                        sequence_no=issue_no,
                        properties={"table_index": table_index, "row_index": row_no},
                    )
                )
                continue

            key_seed = "|".join(
                [
                    str(page["pdf_page_number"]),
                    module_tag,
                    channel_ref,
                    wire_ref,
                    terminal_ref,
                    ",".join(map(str, evidence["source_word_ids"])),
                ]
            )
            key_hash = hashlib.sha256(key_seed.encode("utf-8")).hexdigest()[:20]
            rows_out.append(
                {
                    "io_key": f"io:{int(page['pdf_page_number'])}:{key_hash}",
                    "module_tag": module_tag,
                    "module_model": module_model or None,
                    "rack_ref": None,
                    "slot_ref": None,
                    "channel_ref": channel_ref,
                    "plc_address": _clean_text(row.get("plc_address"), 180) or None,
                    "io_type": io_type,
                    "is_safety": is_safety,
                    "signal_name": signal_name or None,
                    "description": description or signal_name or None,
                    "expected_normal_state": _clean_text(row.get("expected_normal_state"), 300) or None,
                    "wire_reference": wire_ref or None,
                    "terminal_reference": terminal_ref or None,
                    "evidence": evidence,
                    "confidence": confidence,
                    "properties": {
                        "language": language,
                        "table_label_original": table_label,
                        "table_index": table_index,
                        "row_index": row_no,
                        "is_placeholder": is_placeholder,
                        "source_word_ids": evidence["source_word_ids"],
                        "artifact_fingerprint": fingerprint,
                        "prompt_version": ELECTRICAL_STRUCTURED_PROMPT_VERSION,
                        "materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
                    },
                }
            )

    for item in response.get("issues") or []:
        issue_no += 1
        issues.append(
            _issue_record(
                context=context,
                page=page,
                issue_type=str(item.get("issue_type") or "model_reported_issue"),
                message=str(item.get("message") or "Structured extraction issue"),
                source_word_ids=item.get("source_word_ids"),
                word_map=word_map,
                confidence=_clamp_conf(item.get("confidence")),
                sequence_no=issue_no,
                properties={"reported_by_model": True},
            )
        )
    return rows_out, issues


def _materialize_terminals(
    *,
    context: dict,
    page: dict,
    response: dict,
    word_map: dict[int, dict],
    fingerprint: str,
) -> tuple[list[dict], list[dict]]:
    rows_out: list[dict] = []
    issues: list[dict] = []
    issue_no = 0
    row_no = 0
    language = _clean_text(response.get("language"), 32) or "unknown"

    for table_index, table in enumerate(response.get("tables") or [], start=1):
        strip_tag = _clean_text(table.get("strip_tag"), 160)
        table_label = _clean_text(table.get("table_label_original"), 500)
        table_conf = _clamp_conf(table.get("confidence"))
        for row in table.get("rows") or []:
            row_no += 1
            evidence = _source_evidence(row.get("source_word_ids"), word_map)
            confidence = min(table_conf or 1.0, _clamp_conf(row.get("confidence")))
            terminal_number = _clean_text(row.get("terminal_number"), 120)
            invalid_reason = ""
            if not evidence:
                invalid_reason = "missing_valid_source_word_ids"
            elif confidence < ELECTRICAL_STRUCTURED_MIN_CONFIDENCE:
                invalid_reason = "row_confidence_below_threshold"
            elif not strip_tag:
                invalid_reason = "missing_strip_tag"
            elif not terminal_number:
                invalid_reason = "missing_terminal_number"

            if invalid_reason:
                issue_no += 1
                issues.append(
                    _issue_record(
                        context=context,
                        page=page,
                        issue_type=invalid_reason,
                        message=f"Terminal row was not materialized: {invalid_reason}",
                        source_word_ids=row.get("source_word_ids"),
                        word_map=word_map,
                        confidence=confidence,
                        sequence_no=issue_no,
                        properties={"table_index": table_index, "row_index": row_no},
                    )
                )
                continue

            seed = "|".join(
                [
                    str(page["pdf_page_number"]),
                    strip_tag,
                    terminal_number,
                    ",".join(map(str, evidence["source_word_ids"])),
                ]
            )
            key_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
            rows_out.append(
                {
                    "terminal_key": f"terminal:{int(page['pdf_page_number'])}:{key_hash}",
                    "strip_tag": strip_tag,
                    "terminal_number": terminal_number,
                    "level_ref": _clean_text(row.get("level_ref"), 120) or None,
                    "side_a_origin": _clean_text(row.get("side_a_origin"), 500) or None,
                    "side_b_destination": _clean_text(row.get("side_b_destination"), 500) or None,
                    "wire_number": _clean_text(row.get("wire_number"), 180) or None,
                    "cable_reference": _clean_text(row.get("cable_reference"), 180) or None,
                    "potential": _clean_text(row.get("potential"), 180) or None,
                    "conductor_color": _clean_text(row.get("conductor_color"), 120) or None,
                    "conductor_cross_section": _clean_text(row.get("conductor_cross_section"), 120) or None,
                    "description_original": _clean_text(row.get("description_original"), 1000),
                    "evidence": evidence,
                    "confidence": confidence,
                    "properties": {
                        "language": language,
                        "table_label_original": table_label,
                        "table_index": table_index,
                        "row_index": row_no,
                        "source_word_ids": evidence["source_word_ids"],
                        "artifact_fingerprint": fingerprint,
                        "prompt_version": ELECTRICAL_STRUCTURED_PROMPT_VERSION,
                        "materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
                    },
                }
            )

    for item in response.get("issues") or []:
        issue_no += 1
        issues.append(
            _issue_record(
                context=context,
                page=page,
                issue_type=str(item.get("issue_type") or "model_reported_issue"),
                message=str(item.get("message") or "Structured extraction issue"),
                source_word_ids=item.get("source_word_ids"),
                word_map=word_map,
                confidence=_clamp_conf(item.get("confidence")),
                sequence_no=issue_no,
                properties={"reported_by_model": True},
            )
        )
    return rows_out, issues


def _materialize_bom(
    *,
    context: dict,
    page: dict,
    response: dict,
    word_map: dict[int, dict],
    fingerprint: str,
) -> tuple[list[dict], list[dict]]:
    rows_out: list[dict] = []
    issues: list[dict] = []
    issue_no = 0
    row_no = 0
    language = _clean_text(response.get("language"), 32) or "unknown"

    for table_index, table in enumerate(response.get("tables") or [], start=1):
        table_label = _clean_text(table.get("table_label_original"), 500)
        table_conf = _clamp_conf(table.get("confidence"))
        for row in table.get("rows") or []:
            row_no += 1
            evidence = _source_evidence(row.get("source_word_ids"), word_map)
            confidence = min(table_conf or 1.0, _clamp_conf(row.get("confidence")))
            description = _clean_text(row.get("description_original"), 2000)
            part_number = _clean_text(row.get("part_number"), 300)
            manufacturer = _clean_text(row.get("manufacturer"), 300)
            component_tag = _clean_text(row.get("component_tag"), 180)
            invalid_reason = ""
            if not evidence:
                invalid_reason = "missing_valid_source_word_ids"
            elif confidence < ELECTRICAL_STRUCTURED_MIN_CONFIDENCE:
                invalid_reason = "row_confidence_below_threshold"
            elif not description:
                invalid_reason = "missing_description"
            elif not any([part_number, manufacturer, component_tag]):
                invalid_reason = "missing_identifying_bom_fields"

            if invalid_reason:
                issue_no += 1
                issues.append(
                    _issue_record(
                        context=context,
                        page=page,
                        issue_type=invalid_reason,
                        message=f"BOM row was not materialized: {invalid_reason}",
                        source_word_ids=row.get("source_word_ids"),
                        word_map=word_map,
                        confidence=confidence,
                        sequence_no=issue_no,
                        properties={"table_index": table_index, "row_index": row_no},
                    )
                )
                continue

            seed = "|".join(
                [
                    str(page["pdf_page_number"]),
                    component_tag,
                    part_number,
                    description,
                    ",".join(map(str, evidence["source_word_ids"])),
                ]
            )
            key_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
            quantity_text = _clean_text(row.get("quantity_text"), 120)
            rows_out.append(
                {
                    "bom_key": f"bom:{int(page['pdf_page_number'])}:{key_hash}",
                    "item_position": _clean_text(row.get("item_position"), 160) or None,
                    "component_tag": component_tag or None,
                    "quantity": _quantity_numeric(quantity_text),
                    "quantity_text": quantity_text or None,
                    "unit": _clean_text(row.get("unit"), 80) or None,
                    "manufacturer": manufacturer or None,
                    "part_number": part_number or None,
                    "description": description,
                    "evidence": evidence,
                    "confidence": confidence,
                    "properties": {
                        "language": language,
                        "table_label_original": table_label,
                        "table_index": table_index,
                        "row_index": row_no,
                        "source_word_ids": evidence["source_word_ids"],
                        "artifact_fingerprint": fingerprint,
                        "prompt_version": ELECTRICAL_STRUCTURED_PROMPT_VERSION,
                        "materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
                    },
                }
            )

    for item in response.get("issues") or []:
        issue_no += 1
        issues.append(
            _issue_record(
                context=context,
                page=page,
                issue_type=str(item.get("issue_type") or "model_reported_issue"),
                message=str(item.get("message") or "Structured extraction issue"),
                source_word_ids=item.get("source_word_ids"),
                word_map=word_map,
                confidence=_clamp_conf(item.get("confidence")),
                sequence_no=issue_no,
                properties={"reported_by_model": True},
            )
        )
    return rows_out, issues


def _apply_materialized(
    *,
    context: dict,
    io_rows: list[tuple[dict, dict]],
    terminal_rows: list[tuple[dict, dict]],
    bom_rows: list[tuple[dict, dict]],
    issues: list[tuple[dict, dict]],
    usage_totals: dict,
    languages: set[str],
) -> dict:
    version_id = int(context["version_id"])
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            processed_page_ids = sorted({int(p["id"]) for p in context["pages"]})
            processed_page_types = {str(p.get("page_type") or "") for p in context["pages"]}

            if processed_page_types & {"plc_io_table", "safety_io_table"}:
                cur.execute(
                    """
                    DELETE FROM public.electrical_io
                    WHERE version_id=%s
                      AND page_id = ANY(%s)
                      AND extraction_method='openai_structured_v1';
                    """,
                    (version_id, processed_page_ids),
                )
            if "terminal_table" in processed_page_types:
                cur.execute(
                    """
                    DELETE FROM public.electrical_terminals
                    WHERE version_id=%s
                      AND page_id = ANY(%s)
                      AND extraction_method='openai_structured_v1';
                    """,
                    (version_id, processed_page_ids),
                )
            if "bom_table" in processed_page_types:
                cur.execute(
                    """
                    DELETE FROM public.electrical_bom
                    WHERE version_id=%s
                      AND page_id = ANY(%s)
                      AND extraction_method='openai_structured_v1';
                    """,
                    (version_id, processed_page_ids),
                )
            cur.execute(
                """
                DELETE FROM public.electrical_review_issues
                WHERE version_id=%s
                  AND page_id = ANY(%s)
                  AND issue_key LIKE 'structured:%%';
                """,
                (version_id, processed_page_ids),
            )

            for page, row in io_rows:
                e = row["evidence"]
                cur.execute(
                    """
                    INSERT INTO public.electrical_io(
                        version_id, company_id, machine_id, bubble_document_id,
                        page_id, source_entity_id, io_key, module_tag, module_model,
                        rack_ref, slot_ref, channel_ref, plc_address, io_type, is_safety,
                        signal_name, description, expected_normal_state, wire_reference,
                        terminal_reference, x0, y0, x1, y1, source_text, properties,
                        confidence, extraction_method, is_verified, created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                        'openai_structured_v1',false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, io_key) DO UPDATE SET
                        module_tag=EXCLUDED.module_tag,
                        module_model=EXCLUDED.module_model,
                        channel_ref=EXCLUDED.channel_ref,
                        plc_address=EXCLUDED.plc_address,
                        io_type=EXCLUDED.io_type,
                        is_safety=EXCLUDED.is_safety,
                        signal_name=EXCLUDED.signal_name,
                        description=EXCLUDED.description,
                        expected_normal_state=EXCLUDED.expected_normal_state,
                        wire_reference=EXCLUDED.wire_reference,
                        terminal_reference=EXCLUDED.terminal_reference,
                        x0=EXCLUDED.x0,y0=EXCLUDED.y0,x1=EXCLUDED.x1,y1=EXCLUDED.y1,
                        source_text=EXCLUDED.source_text,
                        properties=EXCLUDED.properties,
                        confidence=EXCLUDED.confidence,
                        extraction_method=EXCLUDED.extraction_method,
                        updated_at=NOW();
                    """,
                    (
                        version_id,
                        context["company_id"],
                        context["machine_id"],
                        context["bubble_document_id"],
                        int(page["id"]),
                        row["io_key"],
                        row["module_tag"],
                        row["module_model"],
                        row["rack_ref"],
                        row["slot_ref"],
                        row["channel_ref"],
                        row["plc_address"],
                        row["io_type"],
                        bool(row["is_safety"]),
                        row["signal_name"],
                        row["description"],
                        row["expected_normal_state"],
                        row["wire_reference"],
                        row["terminal_reference"],
                        e["x0"],e["y0"],e["x1"],e["y1"],e["source_text"],
                        json.dumps(row["properties"], ensure_ascii=False),
                        row["confidence"],
                    ),
                )

            for page, row in terminal_rows:
                e = row["evidence"]
                properties = dict(row["properties"])
                if row.get("description_original"):
                    properties["description_original"] = row["description_original"]
                cur.execute(
                    """
                    INSERT INTO public.electrical_terminals(
                        version_id, company_id, machine_id, bubble_document_id,
                        page_id, source_entity_id, terminal_key, strip_tag, terminal_number,
                        level_ref, side_a_origin, side_b_destination, wire_number,
                        cable_reference, potential, conductor_color, conductor_cross_section,
                        x0,y0,x1,y1,source_text,properties,confidence,
                        extraction_method,is_verified,created_at,updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s::jsonb,%s,'openai_structured_v1',false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, terminal_key) DO UPDATE SET
                        strip_tag=EXCLUDED.strip_tag,
                        terminal_number=EXCLUDED.terminal_number,
                        level_ref=EXCLUDED.level_ref,
                        side_a_origin=EXCLUDED.side_a_origin,
                        side_b_destination=EXCLUDED.side_b_destination,
                        wire_number=EXCLUDED.wire_number,
                        cable_reference=EXCLUDED.cable_reference,
                        potential=EXCLUDED.potential,
                        conductor_color=EXCLUDED.conductor_color,
                        conductor_cross_section=EXCLUDED.conductor_cross_section,
                        x0=EXCLUDED.x0,y0=EXCLUDED.y0,x1=EXCLUDED.x1,y1=EXCLUDED.y1,
                        source_text=EXCLUDED.source_text,
                        properties=EXCLUDED.properties,
                        confidence=EXCLUDED.confidence,
                        extraction_method=EXCLUDED.extraction_method,
                        updated_at=NOW();
                    """,
                    (
                        version_id,
                        context["company_id"],context["machine_id"],context["bubble_document_id"],
                        int(page["id"]),row["terminal_key"],row["strip_tag"],row["terminal_number"],
                        row["level_ref"],row["side_a_origin"],row["side_b_destination"],
                        row["wire_number"],row["cable_reference"],row["potential"],
                        row["conductor_color"],row["conductor_cross_section"],
                        e["x0"],e["y0"],e["x1"],e["y1"],e["source_text"],
                        json.dumps(properties, ensure_ascii=False),row["confidence"],
                    ),
                )

            for page, row in bom_rows:
                e = row["evidence"]
                cur.execute(
                    """
                    INSERT INTO public.electrical_bom(
                        version_id, company_id, machine_id, bubble_document_id,
                        page_id, source_entity_id, bom_key, item_position, component_tag,
                        quantity, quantity_text, unit, manufacturer, part_number, description,
                        x0,y0,x1,y1,source_text,properties,confidence,
                        extraction_method,is_verified,created_at,updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s::jsonb,%s,'openai_structured_v1',false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, bom_key) DO UPDATE SET
                        item_position=EXCLUDED.item_position,
                        component_tag=EXCLUDED.component_tag,
                        quantity=EXCLUDED.quantity,
                        quantity_text=EXCLUDED.quantity_text,
                        unit=EXCLUDED.unit,
                        manufacturer=EXCLUDED.manufacturer,
                        part_number=EXCLUDED.part_number,
                        description=EXCLUDED.description,
                        x0=EXCLUDED.x0,y0=EXCLUDED.y0,x1=EXCLUDED.x1,y1=EXCLUDED.y1,
                        source_text=EXCLUDED.source_text,
                        properties=EXCLUDED.properties,
                        confidence=EXCLUDED.confidence,
                        extraction_method=EXCLUDED.extraction_method,
                        updated_at=NOW();
                    """,
                    (
                        version_id,context["company_id"],context["machine_id"],context["bubble_document_id"],
                        int(page["id"]),row["bom_key"],row["item_position"],row["component_tag"],
                        row["quantity"],row["quantity_text"],row["unit"],row["manufacturer"],
                        row["part_number"],row["description"],
                        e["x0"],e["y0"],e["x1"],e["y1"],e["source_text"],
                        json.dumps(row["properties"], ensure_ascii=False),row["confidence"],
                    ),
                )

            for page, issue in issues:
                cur.execute(
                    """
                    INSERT INTO public.electrical_review_issues(
                        version_id, company_id, machine_id, bubble_document_id,
                        page_id, entity_id, edge_id, issue_key, issue_type,
                        severity, status, message, candidates_json, properties,
                        created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,NULL,NULL,%s,%s,%s,'open',%s,%s::jsonb,%s::jsonb,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, issue_key) DO UPDATE SET
                        issue_type=EXCLUDED.issue_type,
                        severity=EXCLUDED.severity,
                        status='open',
                        message=EXCLUDED.message,
                        candidates_json=EXCLUDED.candidates_json,
                        properties=EXCLUDED.properties,
                        updated_at=NOW();
                    """,
                    (
                        version_id,context["company_id"],context["machine_id"],context["bubble_document_id"],
                        int(page["id"]),issue["issue_key"],issue["issue_type"],issue["severity"],
                        issue["message"],json.dumps(issue["candidates_json"], ensure_ascii=False),
                        json.dumps(issue["properties"], ensure_ascii=False),
                    ),
                )

            cur.execute("SELECT COUNT(*) FROM public.electrical_io WHERE version_id=%s;", (version_id,))
            io_count = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM public.electrical_terminals WHERE version_id=%s;", (version_id,))
            terminal_count = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM public.electrical_bom WHERE version_id=%s;", (version_id,))
            bom_count = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM public.electrical_review_issues WHERE version_id=%s AND status='open';", (version_id,))
            review_count = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(cost_usd),0)
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s;
                """,
                (version_id,),
            )
            token_row = cur.fetchone()
            all_input_tokens = int(token_row[0] or 0)
            all_output_tokens = int(token_row[1] or 0)
            all_cost = float(token_row[2] or 0.0)

            previous_completed = set(
                str(x)
                for x in (context.get("metadata", {}).get("structured_completed_page_types") or [])
                if str(x)
            )
            completed_page_types = previous_completed | processed_page_types
            previous_languages = set(
                str(x)
                for x in (context.get("metadata", {}).get("structured_languages") or [])
                if str(x)
            )
            all_languages = previous_languages | {x for x in languages if x}
            all_complete = ELIGIBLE_PAGE_TYPES.issubset(completed_page_types)

            if review_count > 0:
                structured_status = "review_required"
            elif all_complete:
                structured_status = "ready"
            else:
                structured_status = "partial"

            version_status = "review_required" if review_count > 0 else "queued"
            document_status = version_status
            metadata_patch = {
                "structured_status": structured_status,
                "structured_extracted_at": datetime.now().astimezone().isoformat(),
                "structured_prompt_version": ELECTRICAL_STRUCTURED_PROMPT_VERSION,
                "structured_model": ELECTRICAL_STRUCTURED_MODEL,
                "structured_materializer_version": ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION,
                "structured_completed_page_types": sorted(completed_page_types),
                "structured_languages": sorted(all_languages),
                "structured_io_count": io_count,
                "structured_terminal_count": terminal_count,
                "structured_bom_count": bom_count,
                "structured_review_issue_count": review_count,
            }
            cur.execute(
                """
                UPDATE public.electrical_versions
                SET status=%s,
                    deterministic_only=false,
                    openai_used=true,
                    io_count=%s,
                    terminal_count=%s,
                    bom_count=%s,
                    review_issue_count=%s,
                    ai_input_tokens=%s,
                    ai_output_tokens=%s,
                    ai_cost_usd=%s,
                    metadata=COALESCE(metadata,'{}'::jsonb) || %s::jsonb,
                    error_code=NULL,
                    error_message=NULL,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (
                    version_status,io_count,terminal_count,bom_count,review_count,
                    all_input_tokens,all_output_tokens,all_cost,
                    json.dumps(metadata_patch, ensure_ascii=False),version_id,
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
                (document_status, int(context["electrical_document_id"])),
            )
        conn.commit()
        return {
            "status": document_status,
            "structured_status": structured_status,
            "io_count": io_count,
            "terminal_count": terminal_count,
            "bom_count": bom_count,
            "review_issue_count": review_count,
            "ai_input_tokens_total": all_input_tokens,
            "ai_output_tokens_total": all_output_tokens,
            "ai_cost_usd_total": round(all_cost, 6),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def extract_electrical_structured_version(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
    page_types: Optional[list[str]],
    force: bool,
) -> dict:
    if not ELECTRICAL_STRUCTURED_ENABLED:
        raise ValueError("MM_ELECTRICAL_STRUCTURED_ENABLED is disabled")

    context = _load_context(
        company_id=str(company_id),
        machine_id=str(machine_id),
        bubble_document_id=str(bubble_document_id),
        version_id=version_id,
        page_types=page_types,
    )
    context["company_id"] = str(company_id)
    context["machine_id"] = str(machine_id)
    context["bubble_document_id"] = str(bubble_document_id)
    if not context["pages"]:
        raise ValueError("No semantically classified structured electrical pages found")

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.electrical_versions SET status='parsing', updated_at=NOW() WHERE id=%s;",
                (int(context["version_id"]),),
            )
            cur.execute(
                "UPDATE public.electrical_documents SET index_status='parsing', updated_at=NOW() WHERE id=%s;",
                (int(context["electrical_document_id"]),),
            )
        conn.commit()
    finally:
        conn.close()

    io_rows: list[tuple[dict, dict]] = []
    terminal_rows: list[tuple[dict, dict]] = []
    bom_rows: list[tuple[dict, dict]] = []
    issues: list[tuple[dict, dict]] = []
    languages: set[str] = set()
    totals = {
        "calls": 0,
        "reused_calls": 0,
        "new_input_tokens": 0,
        "new_output_tokens": 0,
        "new_reasoning_tokens": 0,
        "new_cost_usd": 0.0,
        "page_type_call_counts": {},
    }

    try:
        for page in context["pages"]:
            page_payload, word_map = _page_payload(page)
            request_payload = {
                "task": "structured_electrical_page_extraction",
                "source_rules": {
                    "preserve_original_language": True,
                    "require_source_word_ids": True,
                    "no_fixed_vocabulary": True,
                    "no_unsupported_inference": True,
                },
                "page": page_payload,
            }
            messages, schema, task_type = _prompt_for_page(page["page_type"], request_payload)
            response, usage, reused, fingerprint = _cached_call(
                context=context,
                page=page,
                task_type=task_type,
                request_payload=request_payload,
                messages=messages,
                json_schema=schema,
                force=bool(force),
            )
            totals["calls"] += 1
            totals["reused_calls"] += 1 if reused else 0
            totals["page_type_call_counts"][page["page_type"]] = (
                int(totals["page_type_call_counts"].get(page["page_type"], 0)) + 1
            )
            if not reused:
                totals["new_input_tokens"] += int(usage.get("input_tokens") or 0)
                totals["new_output_tokens"] += int(usage.get("output_tokens") or 0)
                totals["new_reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
                totals["new_cost_usd"] += float(usage.get("cost_usd") or 0.0)
            languages.add(_clean_text(response.get("language"), 32) or "unknown")

            returned_page_id = int(response.get("page_id") or 0)
            if returned_page_id != int(page["id"]):
                issues.append(
                    (
                        page,
                        _issue_record(
                            context=context,
                            page=page,
                            issue_type="page_id_mismatch",
                            message=(
                                f"Model returned page_id={returned_page_id}; expected {int(page['id'])}"
                            ),
                            source_word_ids=[],
                            word_map=word_map,
                            confidence=0.0,
                            sequence_no=1,
                        ),
                    )
                )
                continue

            if page["page_type"] in {"plc_io_table", "safety_io_table"}:
                rows, page_issues = _materialize_io(
                    context=context,page=page,response=response,word_map=word_map,fingerprint=fingerprint
                )
                io_rows.extend((page, r) for r in rows)
            elif page["page_type"] == "terminal_table":
                rows, page_issues = _materialize_terminals(
                    context=context,page=page,response=response,word_map=word_map,fingerprint=fingerprint
                )
                terminal_rows.extend((page, r) for r in rows)
            else:
                rows, page_issues = _materialize_bom(
                    context=context,page=page,response=response,word_map=word_map,fingerprint=fingerprint
                )
                bom_rows.extend((page, r) for r in rows)
            issues.extend((page, i) for i in page_issues)

        applied = _apply_materialized(
            context=context,
            io_rows=io_rows,
            terminal_rows=terminal_rows,
            bom_rows=bom_rows,
            issues=issues,
            usage_totals=totals,
            languages=languages,
        )
        return {
            "electrical_document_id": int(context["electrical_document_id"]),
            "electrical_version_id": int(context["version_id"]),
            "eligible_pages_total": len(context["pages"]),
            "requested_page_types": sorted(set(page_types or ELIGIBLE_PAGE_TYPES)),
            "languages": sorted(languages),
            **applied,
            **totals,
            "new_cost_usd": round(float(totals["new_cost_usd"]), 6),
        }
    except Exception as e:
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.electrical_versions
                    SET status='review_required',
                        error_code='ELECTRICAL_STRUCTURED_FAILED',
                        error_message=%s,
                        metadata=COALESCE(metadata,'{}'::jsonb) || %s::jsonb,
                        updated_at=NOW()
                    WHERE id=%s;
                    """,
                    (
                        str(e)[:2000],
                        json.dumps({
                            "structured_status": "failed",
                            "structured_failed_at": datetime.now().astimezone().isoformat(),
                        }),
                        int(context["version_id"]),
                    ),
                )
                cur.execute(
                    """
                    UPDATE public.electrical_documents
                    SET index_status='review_required',
                        last_error_code='ELECTRICAL_STRUCTURED_FAILED',
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
