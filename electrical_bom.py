import base64
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime
from itertools import combinations
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import fitz
import psycopg2
import requests

from electrical_source_store import download_electrical_source_pdf

# MachineMind Phase 2B V1.3
# Isolated multimodal bill-of-material extraction.
# Publication is geometry-first, multilingual and fail-closed. No deterministic
# rule depends on Italian/English labels, a specific PDF, fixed coordinates,
# component tags, manufacturers or part-number shapes.


def _env_int(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int = 1_000_000,
) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except Exception:
        value = int(default)
    return max(minimum, min(maximum, value))


def _env_float(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
) -> float:
    try:
        value = float(str(os.environ.get(name, default)).strip())
    except Exception:
        value = float(default)
    return max(minimum, min(maximum, value))


DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_CHAT_URL = (
    os.environ.get("OPENAI_CHAT_URL")
    or "https://api.openai.com/v1/chat/completions"
).strip()

BOM_ENABLED = (
    os.environ.get("MM_ELECTRICAL_BOM_ENABLED") or "0"
).strip() == "1"

DETECTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_BOM_DETECTOR_MODEL") or "gpt-5.4"
).strip()
EXTRACTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_BOM_EXTRACTOR_MODEL") or "gpt-5.4"
).strip()
VERIFIER_MODEL = (
    os.environ.get("MM_ELECTRICAL_BOM_VERIFIER_MODEL") or "gpt-5.4"
).strip()

DETECTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_BOM_DETECTOR_PROMPT_VERSION")
    or "mm-electrical-bom-detector-v1"
).strip()
EXTRACTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_BOM_EXTRACTOR_PROMPT_VERSION")
    or "mm-electrical-bom-table-extractor-v1.1"
).strip()
VERIFIER_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_BOM_VERIFIER_PROMPT_VERSION")
    or "mm-electrical-bom-page-verifier-v1.1"
).strip()
MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_BOM_MATERIALIZER_VERSION")
    or "mm-electrical-bom-materializer-v1.4"
).strip()

OPENAI_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_BOM_TIMEOUT_SECONDS", 240, 30, 600
)
FETCH_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_BOM_FETCH_TIMEOUT_SECONDS", 60, 10, 300
)
RENDER_DPI = _env_int(
    "MM_ELECTRICAL_BOM_RENDER_DPI", 220, 120, 360
)
MAX_COMPLETION_TOKENS = _env_int(
    "MM_ELECTRICAL_BOM_MAX_COMPLETION_TOKENS", 20000, 1000, 64000
)
ROW_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_BOM_ROW_MIN_CONFIDENCE", 0.82, 0.0, 1.0
)
PAGE_PASS_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_BOM_PAGE_PASS_MIN_CONFIDENCE", 0.90, 0.0, 1.0
)
INPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_BOM_INPUT_USD_PER_MILLION", 0.0
)
OUTPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_BOM_OUTPUT_USD_PER_MILLION", 0.0
)
MAX_SOURCE_BYTES = _env_int(
    "MM_ELECTRICAL_BOM_MAX_SOURCE_BYTES",
    100_000_000,
    1_000_000,
    500_000_000,
)

PIPELINE_MARKER = "phase2-bom-v1.4-source-snapshot"
EXTRACTION_METHOD = "openai_vision_bom_v1"
PHASE_NAME = "bom_vision_v1"
PAGE_TYPE = "bom_table"
SEVERITIES = {"info", "warning", "high", "critical"}
ROW_ROLES = {"item", "placeholder_item"}
NON_ITEM_ROW_KINDS = {
    "header",
    "footer",
    "annotation",
    "blank_separator",
    "other_non_item",
}
CANONICAL_COLUMN_ROLES = {
    "item_position",
    "component_tag",
    "quantity",
    "unit",
    "description",
    "part_number",
    "manufacturer",
    "other_data",
}

BASE_FIELDS = (
    "item_position",
    "component_tag",
    "quantity_text",
    "unit",
    "description",
    "part_number",
    "manufacturer",
)
ORIGINAL_FIELDS = tuple(f"{name}_original" for name in BASE_FIELDS)
NORMALIZED_FIELDS = tuple(f"{name}_normalized" for name in BASE_FIELDS)
OVERRIDABLE_FIELDS = set(ORIGINAL_FIELDS + NORMALIZED_FIELDS)
PUBLISHED_FIELD_TO_COLUMN = {
    "item_position": "item_position",
    "component_tag": "component_tag",
    "quantity_text": "quantity_text",
    "unit": "unit",
    "description": "description",
    "part_number": "part_number",
    "manufacturer": "manufacturer",
}
ORIGINAL_FIELD_TO_COLUMN_ROLE = {
    "item_position_original": "item_position",
    "component_tag_original": "component_tag",
    "quantity_text_original": "quantity",
    "unit_original": "unit",
    "description_original": "description",
    "part_number_original": "part_number",
    "manufacturer_original": "manufacturer",
}


def get_electrical_bom_runtime_config() -> dict:
    return {
        "enabled": bool(BOM_ENABLED),
        "pipeline_marker": PIPELINE_MARKER,
        "detector_model": DETECTOR_MODEL,
        "extractor_model": EXTRACTOR_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "detector_prompt_version": DETECTOR_PROMPT_VERSION,
        "extractor_prompt_version": EXTRACTOR_PROMPT_VERSION,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "row_min_confidence": ROW_MIN_CONFIDENCE,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
        "render_dpi": RENDER_DPI,
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


def _clean_text(value: Any, max_len: int = 4000) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _json_obj(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_bigint_id(*parts: Any) -> int:
    """Stable positive bigint without depending on a database sequence."""
    raw = "|".join(str(part or "") for part in parts).encode("utf-8")
    # 15 hex digits remain safely inside PostgreSQL signed bigint.
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def _semantic_character_signature(value: Any) -> str:
    """Character-preserving signature used for safe display normalization.

    Whitespace and dash glyph variants may change. Alphanumeric source content,
    masking symbols and their order may not change. Original values remain
    stored in properties.
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
        .replace("−", "-")
        .upper()
    )
    # Only whitespace and visually equivalent dash glyphs may change.
    # Punctuation remains significant, especially inside part numbers, tags,
    # quantities and manufacturer codes.
    return "".join(ch for ch in text if not ch.isspace())


def _source_evidence_signature(value: Any) -> str:
    """Loose signature for vector-word evidence versus visual transcription.

    CAD/PDF word extraction can omit visible separators such as dashes and
    slashes. The independent visual verifier remains responsible for those
    glyphs. This signature therefore checks alphanumeric/masking content and
    order without forcing unreliable vector punctuation.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return "".join(
        ch for ch in text if ch.isalnum() or ch in {"*", "#"}
    )


def _evidence_atom(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z]+", "", text)


def _price(input_tokens: int, output_tokens: int) -> float:
    return round(
        max(0, int(input_tokens or 0)) / 1_000_000.0
        * INPUT_USD_PER_MILLION
        + max(0, int(output_tokens or 0)) / 1_000_000.0
        * OUTPUT_USD_PER_MILLION,
        6,
    )


def _parse_chat_content(data: dict) -> str:
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(
            f"OpenAI refused BOM vision request: {str(refusal)[:800]}"
        )
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content or "").strip()


def _openai_json_with_usage(
    *,
    model: str,
    messages: list[dict],
    json_schema: dict,
) -> tuple[dict, dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    response = requests.post(
        OPENAI_CHAT_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "OpenAI BOM vision call failed: "
            f"{response.status_code} {response.text[:1800]}"
        )

    data = response.json()
    text = _parse_chat_content(data)
    if not text:
        raise RuntimeError("OpenAI BOM vision call returned empty content")
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            f"BOM vision JSON parse failed: {exc}; raw={text[:1200]}"
        ) from exc

    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    input_tokens = int(
        usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    )
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    return parsed, {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _price(input_tokens, output_tokens),
    }


def _fingerprint(
    *,
    task_type: str,
    prompt_version: str,
    model: str,
    request_payload: dict,
) -> tuple[str, str]:
    # Local materializer changes never invalidate already-paid AI responses.
    request_sha256 = _sha256_json(request_payload)
    raw = "|".join([task_type, prompt_version, model, request_sha256])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), request_sha256


def _db_get_artifact(version_id: int, fingerprint: str) -> Optional[dict]:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, response_json, input_tokens, output_tokens,
                       reasoning_tokens, cost_usd, model, prompt_version,
                       fingerprint
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s AND fingerprint=%s
                LIMIT 1;
                """,
                (int(version_id), str(fingerprint)),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "status": str(row[1] or ""),
                "response_json": _json_obj(row[2], None),
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "reasoning_tokens": int(row[5] or 0),
                "cost_usd": float(row[6] or 0.0),
                "model": str(row[7] or ""),
                "prompt_version": str(row[8] or ""),
                "fingerprint": str(row[9] or ""),
            }
    finally:
        conn.close()


def _db_get_artifact_by_request(
    *,
    version_id: int,
    task_type: str,
    model: str,
    prompt_version: str,
    request_sha256: str,
) -> Optional[dict]:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, response_json, input_tokens, output_tokens,
                       reasoning_tokens, cost_usd, model, prompt_version,
                       fingerprint
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s
                  AND task_type=%s
                  AND model=%s
                  AND prompt_version=%s
                  AND request_sha256=%s
                  AND status IN ('completed','reused')
                  AND response_json IS NOT NULL
                ORDER BY completed_at DESC NULLS LAST, id DESC
                LIMIT 1;
                """,
                (
                    int(version_id),
                    task_type,
                    model,
                    prompt_version,
                    request_sha256,
                ),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(row[0]),
                "status": str(row[1] or ""),
                "response_json": _json_obj(row[2], None),
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "reasoning_tokens": int(row[5] or 0),
                "cost_usd": float(row[6] or 0.0),
                "model": str(row[7] or ""),
                "prompt_version": str(row[8] or ""),
                "fingerprint": str(row[9] or ""),
            }
    finally:
        conn.close()


def _db_start_artifact(
    *,
    context: dict,
    page_id: int,
    fingerprint: str,
    task_type: str,
    region_hash: str,
    model: str,
    prompt_version: str,
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
                    response_json, input_tokens, output_tokens,
                    reasoning_tokens, cost_usd, status, error_message,
                    created_at, completed_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,
                    NULL,0,0,0,0,'pending',NULL,NOW(),NULL
                )
                ON CONFLICT (version_id, fingerprint)
                DO UPDATE SET
                    page_id=EXCLUDED.page_id,
                    task_type=EXCLUDED.task_type,
                    region_hash=EXCLUDED.region_hash,
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
                    int(context["version_id"]),
                    context["company_id"],
                    context["machine_id"],
                    context["bubble_document_id"],
                    int(page_id),
                    fingerprint,
                    task_type,
                    region_hash or None,
                    model,
                    prompt_version,
                    request_sha256,
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
                    completed_at=COALESCE(completed_at,NOW())
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


def _db_fail_artifact(artifact_id: int, message: str) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.electrical_ai_artifacts
                SET status='failed', error_message=%s, completed_at=NOW()
                WHERE id=%s;
                """,
                (_clean_text(message, 2000), int(artifact_id)),
            )
        conn.commit()
    finally:
        conn.close()


def _cached_call(
    *,
    context: dict,
    page: dict,
    task_type: str,
    region_hash: str,
    model: str,
    prompt_version: str,
    request_payload: dict,
    messages: list[dict],
    json_schema: dict,
    force: bool,
    request_metadata: dict,
) -> tuple[dict, dict, bool, str]:
    fingerprint, request_sha256 = _fingerprint(
        task_type=task_type,
        prompt_version=prompt_version,
        model=model,
        request_payload=request_payload,
    )
    existing = _db_get_artifact(int(context["version_id"]), fingerprint)
    if not force and not existing:
        existing = _db_get_artifact_by_request(
            version_id=int(context["version_id"]),
            task_type=task_type,
            model=model,
            prompt_version=prompt_version,
            request_sha256=request_sha256,
        )
    if (
        not force
        and existing
        and existing.get("response_json")
        and existing.get("status") in {"completed", "reused"}
    ):
        usage = {
            "model": existing.get("model") or model,
            "input_tokens": int(existing.get("input_tokens") or 0),
            "output_tokens": int(existing.get("output_tokens") or 0),
            "reasoning_tokens": int(existing.get("reasoning_tokens") or 0),
            "cost_usd": float(existing.get("cost_usd") or 0.0),
        }
        _db_complete_artifact(
            artifact_id=int(existing["id"]),
            response_json=existing["response_json"],
            usage=usage,
            reused=True,
        )
        return (
            existing["response_json"],
            usage,
            True,
            str(existing.get("fingerprint") or fingerprint),
        )

    artifact_id = _db_start_artifact(
        context=context,
        page_id=int(page["id"]),
        fingerprint=fingerprint,
        task_type=task_type,
        region_hash=region_hash,
        model=model,
        prompt_version=prompt_version,
        request_sha256=request_sha256,
        request_metadata={
            "phase": PHASE_NAME,
            "pipeline_marker": PIPELINE_MARKER,
            "materializer_version": MATERIALIZER_VERSION,
            "pdf_page_number": int(page["pdf_page_number"]),
            **request_metadata,
        },
    )
    try:
        result, usage = _openai_json_with_usage(
            model=model,
            messages=messages,
            json_schema=json_schema,
        )
        _db_complete_artifact(
            artifact_id=artifact_id,
            response_json=result,
            usage=usage,
            reused=False,
        )
        return result, usage, False, fingerprint
    except Exception as exc:
        _db_fail_artifact(artifact_id, str(exc))
        raise


def _load_context(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
    pdf_page_numbers: Optional[list[int]],
) -> dict:
    page_numbers = sorted(
        {int(x) for x in (pdf_page_numbers or []) if int(x) > 0}
    )
    if len(page_numbers) != 1:
        raise ValueError(
            "BOM extraction requires exactly one pdf_page_numbers value per "
            "request to keep publication atomic."
        )

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
                SELECT d.id, d.source_filename,
                       v.id, v.version_no, v.status, v.metadata,
                       v.pdf_page_count, v.declared_sheet_count,
                       v.source_sha256, f.file_url
                FROM public.electrical_documents d
                JOIN public.electrical_versions v
                  ON v.electrical_document_id=d.id
                 AND v.company_id=d.company_id
                 AND v.machine_id=d.machine_id
                 AND v.bubble_document_id=d.bubble_document_id
                LEFT JOIN public.document_files f
                  ON f.company_id=d.company_id
                 AND f.bubble_document_id=d.bubble_document_id
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
                raise ValueError(
                    "Electrical version not found for supplied scope"
                )

            cur.execute(
                """
                SELECT id, pdf_page_number, sheet_code, sheet_title,
                       group_code, page_type, page_width_pt,
                       page_height_pt, page_sha256, raw_text,
                       text_spans_json, classification_language,
                       semantic_confidence, classification_metadata
                FROM public.electrical_pages
                WHERE version_id=%s
                  AND page_type=%s
                  AND pdf_page_number=ANY(%s)
                ORDER BY pdf_page_number;
                """,
                (int(row[2]), PAGE_TYPE, page_numbers),
            )
            p = cur.fetchone()
            if not p:
                raise ValueError(
                    "Requested page was not found among classified bom_table pages"
                )
            page = {
                "id": int(p[0]),
                "pdf_page_number": int(p[1]),
                "sheet_code": str(p[2] or ""),
                "sheet_title": str(p[3] or ""),
                "group_code": str(p[4] or ""),
                "page_type": str(p[5] or ""),
                "page_width_pt": float(p[6] or 1.0),
                "page_height_pt": float(p[7] or 1.0),
                "page_sha256": str(p[8] or ""),
                "raw_text": str(p[9] or ""),
                "words": list(_json_obj(p[10], []) or []),
                "classification_language": str(p[11] or "unknown"),
                "semantic_confidence": float(p[12] or 0.0),
                "classification_metadata": _json_obj(p[13], {}) or {},
            }

            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_pages
                WHERE version_id=%s AND page_type=%s;
                """,
                (int(row[2]), PAGE_TYPE),
            )
            total_pages = int(cur.fetchone()[0] or 0)

            version_metadata = _json_obj(row[5], {}) or {}
            source_snapshot = version_metadata.get("source_snapshot") or {}
            if not isinstance(source_snapshot, dict):
                source_snapshot = {}

            return {
                "electrical_document_id": int(row[0]),
                "source_filename": str(row[1] or ""),
                "version_id": int(row[2]),
                "version_no": int(row[3]),
                "version_status": str(row[4] or ""),
                "metadata": version_metadata,
                "pdf_page_count": int(row[6] or 0),
                "declared_sheet_count": (
                    int(row[7]) if row[7] is not None else None
                ),
                "source_sha256": str(row[8] or ""),
                "source_snapshot_uri": str(
                    source_snapshot.get("uri")
                    or version_metadata.get("source_snapshot_uri")
                    or ""
                ).strip(),
                "file_url": str(row[9] or ""),
                "page": page,
                "all_bom_pages_total": total_pages,
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
            }
    finally:
        conn.close()


def _fetch_source_pdf(context: dict) -> tuple[bytes, fitz.Document]:
    expected_sha = str(context.get("source_sha256") or "").strip().lower()
    snapshot_uri = str(context.get("source_snapshot_uri") or "").strip()

    if snapshot_uri:
        try:
            data = download_electrical_source_pdf(
                uri=snapshot_uri,
                expected_sha256=expected_sha or None,
                max_bytes=MAX_SOURCE_BYTES,
            )
        except Exception as exc:
            raise ValueError(
                "SOURCE_SNAPSHOT_READ_FAILED: the private persisted "
                "electrical PDF could not be read: "
                f"{str(exc)[:700]}"
            ) from exc
    else:
        url = str(context.get("file_url") or "").strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url:
            raise ValueError(
                "SOURCE_SNAPSHOT_MISSING: no private source snapshot "
                "and no usable legacy URL."
            )
        response = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
        data = response.content

    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError(
            "Electrical source PDF is empty or exceeds configured limit"
        )
    actual_sha = _sha256_bytes(data)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            "Electrical source PDF SHA-256 differs from indexed version"
        )
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(
            f"Electrical source PDF cannot be opened: {exc}"
        ) from exc
    if len(doc) != int(context.get("pdf_page_count") or len(doc)):
        doc.close()
        raise ValueError(
            "Electrical source PDF page count differs from indexed version"
        )
    return data, doc


def _word_map(page: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for idx, word in enumerate(page.get("words") or [], start=1):
        if not isinstance(word, (list, tuple)) or len(word) < 5:
            continue
        try:
            x0, y0, x1, y1 = [float(word[i]) for i in range(4)]
        except Exception:
            continue
        text = str(word[4] or "").replace("\x00", "")
        if not text.strip():
            continue
        out[idx] = {
            "id": idx,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "text": text,
            "block_no": int(word[5] or 0) if len(word) > 5 else 0,
            "line_no": int(word[6] or 0) if len(word) > 6 else 0,
            "word_no": int(word[7] or 0) if len(word) > 7 else 0,
        }
    return out


def _word_center(word: dict) -> tuple[float, float]:
    return (
        (word["x0"] + word["x1"]) / 2.0,
        (word["y0"] + word["y1"]) / 2.0,
    )


def _ids_in_rect(
    word_map: dict[int, dict],
    rect: fitz.Rect,
) -> list[int]:
    ids: list[int] = []
    for wid, word in word_map.items():
        cx, cy = _word_center(word)
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            ids.append(wid)
    return sorted(
        ids,
        key=lambda wid: (
            word_map[wid]["y0"],
            word_map[wid]["x0"],
            wid,
        ),
    )


def _text_for_ids(
    ids: list[int],
    word_map: dict[int, dict],
    max_len: int = 5000,
) -> str:
    words = [word_map[i] for i in ids if i in word_map]
    words.sort(key=lambda w: (w["y0"], w["x0"], w["id"]))
    return _clean_text(
        " ".join(w["text"] for w in words),
        max_len,
    )


def _rect_list(rect: fitz.Rect, digits: int = 2) -> list[float]:
    return [
        round(float(rect.x0), digits),
        round(float(rect.y0), digits),
        round(float(rect.x1), digits),
        round(float(rect.y1), digits),
    ]


def _rect_from(value: Any) -> fitz.Rect:
    if isinstance(value, fitz.Rect):
        return value
    return fitz.Rect(*[float(x) for x in value])


def _table_is_outer_frame(table: Any, page_rect: fitz.Rect) -> bool:
    bbox = _rect_from(table.bbox)
    area_ratio = bbox.get_area() / max(1.0, page_rect.get_area())
    touches = (
        bbox.x0 <= page_rect.x0 + 10
        and bbox.y0 <= page_rect.y0 + 10
        and bbox.x1 >= page_rect.x1 - 10
        and bbox.y1 >= page_rect.y1 - 10
    )
    return area_ratio >= 0.82 or touches


def _bbox_iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    inter_area = max(0.0, float(inter.get_area()))
    if inter_area <= 0:
        return 0.0
    union = float(a.get_area()) + float(b.get_area()) - inter_area
    return inter_area / max(1.0, union)


def _candidate_tables(source_page: fitz.Page) -> list[Any]:
    finder = source_page.find_tables()
    raw = []
    for table in finder.tables or []:
        bbox = _rect_from(table.bbox)
        area_ratio = bbox.get_area() / max(
            1.0,
            source_page.rect.get_area(),
        )
        if _table_is_outer_frame(table, source_page.rect):
            continue
        if int(table.row_count or 0) < 4:
            continue
        if int(table.col_count or 0) < 2:
            continue
        if area_ratio < 0.04 or area_ratio > 0.78:
            continue
        if bbox.width < source_page.rect.width * 0.28:
            continue
        if bbox.height < source_page.rect.height * 0.12:
            continue
        raw.append(table)

    raw.sort(
        key=lambda table: (
            -int(table.row_count or 0),
            -_rect_from(table.bbox).get_area(),
            _rect_from(table.bbox).y0,
            _rect_from(table.bbox).x0,
        )
    )
    deduped: list[Any] = []
    for table in raw:
        bbox = _rect_from(table.bbox)
        if any(_bbox_iou(bbox, _rect_from(x.bbox)) >= 0.92 for x in deduped):
            continue
        deduped.append(table)
    deduped.sort(key=lambda t: (_rect_from(t.bbox).y0, _rect_from(t.bbox).x0))
    return deduped[:12]


def _table_row_cells(
    table: Any,
    row_index: int,
) -> list[Any]:
    rows = list(table.rows or [])
    if row_index < 0 or row_index >= len(rows):
        return []
    return list(rows[row_index].cells or [])


def _detect_sidecar_column_specs(
    *,
    source_page: fitz.Page,
    table_bbox: fitz.Rect,
    row_rects: list[fitz.Rect],
    word_map: dict[int, dict],
) -> list[dict]:
    """Recover visually populated columns adjacent to a detected table.

    Some CAD PDFs omit one vertical ruling line. ``PyMuPDF.find_tables`` can
    then stop at the last fully bordered column even though a further column is
    visibly populated on nearly every physical row. This geometry-only rescue
    looks for a compact word band immediately to the left or right of the
    detected table and accepts it only when it is aligned with a substantial
    fraction of the table rows. No header vocabulary, manufacturer name, page
    number or fixed coordinate is used.
    """
    if len(row_rects) < 4 or not word_map:
        return []

    page_rect = source_page.rect
    page_width = max(1.0, float(page_rect.width))
    max_search_gap = max(10.0, min(72.0, page_width * 0.10))
    max_band_width = max(36.0, min(190.0, page_width * 0.25))
    minimum_support = max(3, int(len(row_rects) * 0.30 + 0.999))

    specs: list[dict] = []
    for side in ("left", "right"):
        support_rows: list[int] = []
        supported_word_ids: set[int] = set()
        nearest_gap: Optional[float] = None

        for row_index, row_rect in enumerate(row_rects):
            row_ids: list[int] = []
            for word_id, word in word_map.items():
                cx, cy = _word_center(word)
                if not (row_rect.y0 <= cy <= row_rect.y1):
                    continue

                if side == "right":
                    gap = float(word["x0"]) - float(table_bbox.x1)
                    outside = cx > float(table_bbox.x1) + 0.25
                    within = gap <= max_search_gap
                else:
                    gap = float(table_bbox.x0) - float(word["x1"])
                    outside = cx < float(table_bbox.x0) - 0.25
                    within = gap <= max_search_gap

                if outside and 0.0 <= gap <= max_search_gap and within:
                    row_ids.append(int(word_id))
                    nearest_gap = gap if nearest_gap is None else min(
                        nearest_gap,
                        gap,
                    )

            if row_ids:
                support_rows.append(row_index)
                supported_word_ids.update(row_ids)

        if len(support_rows) < minimum_support or not supported_word_ids:
            continue

        words = [word_map[word_id] for word_id in sorted(supported_word_ids)]
        word_x0 = min(float(word["x0"]) for word in words)
        word_x1 = max(float(word["x1"]) for word in words)
        band_width = word_x1 - word_x0
        if band_width <= 0.0 or band_width > max_band_width:
            continue

        support_ratio = len(support_rows) / max(1, len(row_rects))
        if support_ratio < 0.30:
            continue

        margin = 3.0
        if side == "right":
            x0 = max(float(table_bbox.x1), word_x0 - margin)
            x1 = min(float(page_rect.x1), word_x1 + margin)
        else:
            x0 = max(float(page_rect.x0), word_x0 - margin)
            x1 = min(float(table_bbox.x0), word_x1 + margin)
        if x1 - x0 < 4.0:
            continue

        specs.append(
            {
                "side": side,
                "bbox_pt": _rect_list(
                    fitz.Rect(x0, table_bbox.y0, x1, table_bbox.y1)
                ),
                "support_row_indexes": support_rows,
                "support_row_count": len(support_rows),
                "support_ratio": round(support_ratio, 4),
                "source_word_ids": sorted(supported_word_ids),
                "nearest_gap_pt": round(float(nearest_gap or 0.0), 2),
                "band_width_pt": round(float(band_width), 2),
            }
        )

    # At most one adjacent recovered column per side. Both sides are supported
    # for generality; the final physical order is assigned from x coordinates.
    return sorted(specs, key=lambda item: item["bbox_pt"][0])


def _build_table_proposal(
    *,
    table: Any,
    proposal_index: int,
    source_page: fitz.Page,
    inventory_page: dict,
    word_map: dict[int, dict],
) -> dict:
    original_bbox = _rect_from(table.bbox)
    extracted = table.extract() or []
    rows = list(table.rows or [])
    row_rects = [
        fitz.Rect(
            original_bbox.x0,
            _rect_from(table_row.bbox).y0,
            original_bbox.x1,
            _rect_from(table_row.bbox).y1,
        )
        for table_row in rows
    ]

    sidecar_specs = _detect_sidecar_column_specs(
        source_page=source_page,
        table_bbox=original_bbox,
        row_rects=row_rects,
        word_map=word_map,
    )
    sidecar_rects = [_rect_from(item["bbox_pt"]) for item in sidecar_specs]

    extended_bbox = fitz.Rect(original_bbox)
    for sidecar_rect in sidecar_rects:
        extended_bbox.include_rect(sidecar_rect)

    crop = fitz.Rect(
        max(source_page.rect.x0, extended_bbox.x0 - 8.0),
        max(source_page.rect.y0, extended_bbox.y0 - 12.0),
        min(source_page.rect.x1, extended_bbox.x1 + 8.0),
        min(source_page.rect.y1, extended_bbox.y1 + 12.0),
    )

    row_candidates: list[dict] = []
    base_column_count = int(table.col_count or 0)
    for row_index, table_row in enumerate(rows):
        original_row_bbox = _rect_from(table_row.bbox)
        row_rect = fitz.Rect(
            extended_bbox.x0,
            original_row_bbox.y0,
            extended_bbox.x1,
            original_row_bbox.y1,
        )
        raw_cells: list[dict] = []
        cell_rects = _table_row_cells(table, row_index)

        for column_index in range(base_column_count):
            raw_cell = (
                cell_rects[column_index]
                if column_index < len(cell_rects)
                else None
            )
            if raw_cell:
                cell_rect = _rect_from(raw_cell)
            else:
                width = original_bbox.width / max(1, base_column_count)
                cell_rect = fitz.Rect(
                    original_bbox.x0 + column_index * width,
                    original_row_bbox.y0,
                    original_bbox.x0 + (column_index + 1) * width,
                    original_row_bbox.y1,
                )
            deterministic_text = ""
            if row_index < len(extracted):
                values = extracted[row_index] or []
                if column_index < len(values):
                    deterministic_text = _clean_text(
                        values[column_index],
                        1600,
                    )
            raw_cells.append(
                {
                    "bbox": cell_rect,
                    "deterministic_text": deterministic_text,
                    "geometry_source": "pymupdf_find_tables",
                }
            )

        for sidecar_spec, sidecar_rect in zip(sidecar_specs, sidecar_rects):
            cell_rect = fitz.Rect(
                sidecar_rect.x0,
                original_row_bbox.y0,
                sidecar_rect.x1,
                original_row_bbox.y1,
            )
            word_ids = _ids_in_rect(word_map, cell_rect)
            raw_cells.append(
                {
                    "bbox": cell_rect,
                    "deterministic_text": _text_for_ids(
                        word_ids,
                        word_map,
                        1600,
                    ),
                    "geometry_source": "row_aligned_sidecar_recovery_v1",
                    "side": sidecar_spec["side"],
                }
            )

        raw_cells.sort(key=lambda item: (item["bbox"].x0, item["bbox"].x1))
        cells: list[dict] = []
        for source_column_index, raw_cell in enumerate(raw_cells):
            cell_rect = raw_cell["bbox"]
            word_ids = _ids_in_rect(word_map, cell_rect)
            cells.append(
                {
                    "source_column_index": source_column_index,
                    "bbox_pt": _rect_list(cell_rect),
                    "word_ids": word_ids,
                    "word_text_original": _text_for_ids(
                        word_ids,
                        word_map,
                        2000,
                    ),
                    "deterministic_cell_text_original": raw_cell[
                        "deterministic_text"
                    ],
                    "geometry_source": raw_cell["geometry_source"],
                }
            )

        row_word_ids = sorted(
            {
                wid
                for cell in cells
                for wid in (cell.get("word_ids") or [])
            },
            key=lambda wid: (
                word_map.get(wid, {}).get("y0", 0.0),
                word_map.get(wid, {}).get("x0", 0.0),
                wid,
            ),
        )
        row_candidates.append(
            {
                "source_row_candidate_id": f"R{row_index + 1:03d}",
                "source_row_index": row_index,
                "bbox_pt": _rect_list(row_rect),
                "word_ids": row_word_ids,
                "word_text_original": _text_for_ids(
                    row_word_ids,
                    word_map,
                    5000,
                ),
                "cells": cells,
            }
        )

    region_id = (
        f"P{int(inventory_page['pdf_page_number'])}-BOM{proposal_index:02d}"
    )
    proposal = {
        "region_id": region_id,
        "geometry_method": (
            "pymupdf_find_tables_plus_sidecar_columns_v1"
            if sidecar_specs
            else "pymupdf_find_tables_v1"
        ),
        "table_bbox_pt": _rect_list(extended_bbox),
        "original_table_bbox_pt": _rect_list(original_bbox),
        "crop_bbox_pt": _rect_list(crop),
        "deterministic_row_count": int(table.row_count or 0),
        "deterministic_column_count": (
            base_column_count + len(sidecar_specs)
        ),
        "geometry_recovery": {
            "version": "row-aligned-sidecar-column-recovery-v1",
            "base_column_count": base_column_count,
            "recovered_column_count": len(sidecar_specs),
            "recovered_columns": sidecar_specs,
        },
        "row_candidates": row_candidates,
    }
    proposal["region_hash"] = _sha256_json(
        {
            "page_sha256": inventory_page.get("page_sha256"),
            **proposal,
        }
    )
    return proposal

def _fallback_page_proposal(
    *,
    source_page: fitz.Page,
    inventory_page: dict,
    word_map: dict[int, dict],
) -> dict:
    # Fail-safe fallback: the complete page is still sent to the visual stages.
    # No row count is fabricated when deterministic table geometry is absent.
    bbox = source_page.rect
    region_id = f"P{int(inventory_page['pdf_page_number'])}-BOM01"
    proposal = {
        "region_id": region_id,
        "geometry_method": "full_page_visual_fallback_v1",
        "table_bbox_pt": _rect_list(bbox),
        "crop_bbox_pt": _rect_list(bbox),
        "deterministic_row_count": 0,
        "deterministic_column_count": 0,
        "row_candidates": [],
        "fallback_page_word_ids": sorted(word_map),
        "fallback_page_words": [
            {
                "word_id": int(word_id),
                "bbox_pt": [
                    round(float(word_map[word_id]["x0"]), 2),
                    round(float(word_map[word_id]["y0"]), 2),
                    round(float(word_map[word_id]["x1"]), 2),
                    round(float(word_map[word_id]["y1"]), 2),
                ],
                "text_original": _clean_text(
                    word_map[word_id].get("text"),
                    500,
                ),
            }
            for word_id in sorted(word_map)
        ],
    }
    proposal["region_hash"] = _sha256_json(
        {
            "page_sha256": inventory_page.get("page_sha256"),
            **proposal,
        }
    )
    return proposal


def _detect_geometry_proposals(
    *,
    source_page: fitz.Page,
    inventory_page: dict,
    word_map: dict[int, dict],
) -> list[dict]:
    tables = _candidate_tables(source_page)
    proposals = [
        _build_table_proposal(
            table=table,
            proposal_index=index,
            source_page=source_page,
            inventory_page=inventory_page,
            word_map=word_map,
        )
        for index, table in enumerate(tables, start=1)
    ]
    return proposals or [
        _fallback_page_proposal(
            source_page=source_page,
            inventory_page=inventory_page,
            word_map=word_map,
        )
    ]


def _data_url_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _render_page(
    source_doc: fitz.Document,
    page_index: int,
    proposals: list[dict],
    rotation: int,
) -> bytes:
    source_page = source_doc[page_index]
    overlay = fitz.open()
    try:
        page = overlay.new_page(
            width=source_page.rect.width,
            height=source_page.rect.height,
        )
        page.show_pdf_page(page.rect, source_doc, page_index)
        for proposal in proposals:
            rect = _rect_from(proposal["crop_bbox_pt"])
            page.draw_rect(rect, color=(1, 0, 0), width=0.7, overlay=True)
            page.insert_text(
                (rect.x0 + 2, max(8, rect.y0 + 8)),
                proposal["region_id"],
                fontsize=7,
                color=(1, 0, 0),
                overlay=True,
            )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0)
            .prerotate(rotation),
            alpha=False,
        )
        return pix.tobytes("png")
    finally:
        overlay.close()


def _render_region(
    source_doc: fitz.Document,
    page_index: int,
    proposal: dict,
    rotation: int,
) -> bytes:
    crop = _rect_from(proposal["crop_bbox_pt"])
    overlay = fitz.open()
    try:
        page = overlay.new_page(width=crop.width, height=crop.height)
        page.show_pdf_page(
            page.rect,
            source_doc,
            page_index,
            clip=crop,
        )
        for row in proposal.get("row_candidates") or []:
            original = _rect_from(row["bbox_pt"])
            local = fitz.Rect(
                original.x0 - crop.x0,
                original.y0 - crop.y0,
                original.x1 - crop.x0,
                original.y1 - crop.y0,
            )
            page.draw_rect(local, color=(1, 0, 0), width=0.35, overlay=True)
            page.insert_text(
                (
                    max(1.0, local.x0 + 1.0),
                    max(6.0, local.y0 + 6.0),
                ),
                str(row.get("source_row_candidate_id") or ""),
                fontsize=4.5,
                color=(1, 0, 0),
                overlay=True,
            )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0)
            .prerotate(rotation),
            alpha=False,
        )
        return pix.tobytes("png")
    finally:
        overlay.close()


def _issue_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issue_type": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": sorted(SEVERITIES),
            },
            "message": {"type": "string"},
            "region_id": {"type": "string"},
            "row_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 400,
            },
            "confidence": {"type": "number"},
        },
        "required": [
            "issue_type",
            "severity",
            "message",
            "region_id",
            "row_ids",
            "confidence",
        ],
    }


def _verifier_issue_schema() -> dict:
    override_ref_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "region_id": {"type": "string"},
            "row_id": {"type": "string"},
            "field_name": {
                "type": "string",
                "enum": sorted(OVERRIDABLE_FIELDS),
            },
        },
        "required": ["region_id", "row_id", "field_name"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issue_type": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": sorted(SEVERITIES),
            },
            "message": {"type": "string"},
            "region_id": {"type": "string"},
            "row_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 500,
            },
            "confidence": {"type": "number"},
            "resolution_status": {
                "type": "string",
                "enum": [
                    "open",
                    "resolved_by_exact_overrides",
                    "informational",
                ],
            },
            "related_overrides": {
                "type": "array",
                "items": override_ref_schema,
                "maxItems": 1000,
            },
        },
        "required": [
            "issue_type",
            "severity",
            "message",
            "region_id",
            "row_ids",
            "confidence",
            "resolution_status",
            "related_overrides",
        ],
    }


def _detector_schema() -> dict:
    return {
        "name": "electrical_bom_page_detector_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "language": {"type": "string"},
                "preferred_reading_rotation_degrees": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                },
                "all_visible_bom_tables_accounted_for": {"type": "boolean"},
                "proposal_assessments": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "visible": {"type": "boolean"},
                            "distinct_table": {"type": "boolean"},
                            "kind": {
                                "type": "string",
                                "enum": ["bom_table", "other_table", "not_table"],
                            },
                            "expected_header_rows": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 30,
                            },
                            "expected_item_rows": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 500,
                            },
                            "expected_column_count": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 40,
                            },
                            "confidence": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "visible",
                            "distinct_table",
                            "kind",
                            "expected_header_rows",
                            "expected_item_rows",
                            "expected_column_count",
                            "confidence",
                            "notes",
                        ],
                    },
                },
                "missing_visible_bom_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 40,
                },
            },
            "required": [
                "page_id",
                "language",
                "preferred_reading_rotation_degrees",
                "all_visible_bom_tables_accounted_for",
                "proposal_assessments",
                "missing_visible_bom_tables",
                "confidence",
                "issues",
            ],
        },
    }


def _field_evidence_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "field_name": {
                "type": "string",
                "enum": sorted(ORIGINAL_FIELDS),
            },
            "source_column_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": 50,
            },
            "source_word_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 400,
            },
        },
        "required": [
            "field_name",
            "source_column_index",
            "source_word_ids",
        ],
    }


def _bom_row_schema() -> dict:
    properties: dict[str, Any] = {
        "row_id": {"type": "string"},
        "source_row_candidate_id": {"type": "string"},
        "visual_order": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
        },
        "row_role": {
            "type": "string",
            "enum": sorted(ROW_ROLES),
        },
        "field_evidence": {
            "type": "array",
            "items": _field_evidence_schema(),
            "maxItems": 20,
        },
        "source_word_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": 800,
        },
        "bbox_pt": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number"},
        "evidence_notes": {"type": "string"},
    }
    for field_name in ORIGINAL_FIELDS + NORMALIZED_FIELDS:
        properties[field_name] = {"type": "string"}
    required = [
        "row_id",
        "source_row_candidate_id",
        "visual_order",
        "row_role",
        *ORIGINAL_FIELDS,
        *NORMALIZED_FIELDS,
        "field_evidence",
        "source_word_ids",
        "bbox_pt",
        "confidence",
        "evidence_notes",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _extractor_schema() -> dict:
    return {
        "name": "electrical_bom_table_extractor_v1_1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "region_id": {"type": "string"},
                "header_row_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 30,
                },
                "non_item_rows": {
                    "type": "array",
                    "maxItems": 100,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_row_candidate_id": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": sorted(NON_ITEM_ROW_KINDS),
                            },
                            "reason": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "source_row_candidate_id",
                            "kind",
                            "reason",
                            "confidence",
                        ],
                    },
                },
                "source_column_roles": {
                    "type": "array",
                    "maxItems": 50,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_column_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 50,
                            },
                            "canonical_roles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": sorted(CANONICAL_COLUMN_ROLES),
                                },
                                "minItems": 1,
                                "maxItems": 8,
                            },
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "source_column_index",
                            "canonical_roles",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "rows": {
                    "type": "array",
                    "items": _bom_row_schema(),
                    "maxItems": 500,
                },
                "unaccounted_row_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 500,
                },
                "duplicate_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 500,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 60,
                },
            },
            "required": [
                "page_id",
                "region_id",
                "header_row_candidate_ids",
                "non_item_rows",
                "source_column_roles",
                "rows",
                "unaccounted_row_candidate_ids",
                "duplicate_row_ids",
                "confidence",
                "issues",
            ],
        },
    }


def _verifier_schema() -> dict:
    return {
        "name": "electrical_bom_page_verifier_v1_1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "review_required"],
                },
                "all_visible_bom_tables_accounted_for": {"type": "boolean"},
                "all_visible_item_rows_accounted_for": {"type": "boolean"},
                "all_visible_columns_accounted_for": {"type": "boolean"},
                "all_published_fields_visually_supported": {"type": "boolean"},
                "all_source_evidence_represented": {"type": "boolean"},
                "duplicates_preserved": {"type": "boolean"},
                "region_checks": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "expected_item_rows": {"type": "integer"},
                            "verified_item_rows": {"type": "integer"},
                            "verified_row_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 500,
                            },
                            "verified_component_tag_sequence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 500,
                            },
                            "pass": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "expected_item_rows",
                            "verified_item_rows",
                            "verified_row_ids",
                            "verified_component_tag_sequence",
                            "pass",
                            "confidence",
                            "notes",
                        ],
                    },
                },
                "field_overrides": {
                    "type": "array",
                    "maxItems": 1000,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "row_id": {"type": "string"},
                            "field_name": {
                                "type": "string",
                                "enum": sorted(OVERRIDABLE_FIELDS),
                            },
                            "approved_text": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "row_id",
                            "field_name",
                            "approved_text",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "missing_region_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "missing_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 500,
                },
                "duplicate_physical_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 500,
                },
                "unaccounted_visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 100,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _verifier_issue_schema(),
                    "maxItems": 80,
                },
            },
            "required": [
                "page_id",
                "verdict",
                "all_visible_bom_tables_accounted_for",
                "all_visible_item_rows_accounted_for",
                "all_visible_columns_accounted_for",
                "all_published_fields_visually_supported",
                "all_source_evidence_represented",
                "duplicates_preserved",
                "region_checks",
                "field_overrides",
                "missing_region_ids",
                "missing_row_ids",
                "duplicate_physical_keys",
                "unaccounted_visual_evidence",
                "confidence",
                "issues",
            ],
        },
    }


def _detector_messages(
    page: dict,
    proposals: list[dict],
    image_original: bytes,
    image_rotated: bytes,
) -> list[dict]:
    summary = [
        {
            "region_id": p["region_id"],
            "geometry_method": p["geometry_method"],
            "table_bbox_pt": p["table_bbox_pt"],
            "crop_bbox_pt": p["crop_bbox_pt"],
            "deterministic_row_count": p["deterministic_row_count"],
            "deterministic_column_count": p["deterministic_column_count"],
            "geometry_recovery": p.get("geometry_recovery") or {},
        }
        for p in proposals
    ]
    system = (
        "You are the visual perception stage of an industrial bill-of-material "
        "reader. The page may use any language, font, orientation, CAD system, "
        "or drawing standard. Work semantically from the complete images and "
        "geometry proposals. Identify every physical material-list/BOM table. "
        "Do not confuse title blocks, revision tables, page frames, coordinate "
        "rulers, legends, or decorative grids with BOM data. Count physical "
        "item rows exactly. Multi-line text inside one bordered row remains one "
        "item. Repeated component tags, manufacturers, descriptions or part "
        "numbers are valid separate physical rows and must not be deduplicated. "
        "Never infer a missing row or correct source values."
    )
    user_text = (
        "Audit every red proposal, classify whether it is a BOM table, count "
        "header and item rows separately, count visible data columns, and list "
        "any visible BOM table not covered by a proposal.\n\n"
        + json.dumps(
            {
                "page_id": page["id"],
                "pdf_page_number": page["pdf_page_number"],
                "sheet_code_original": page.get("sheet_code"),
                "sheet_title_original": page.get("sheet_title"),
                "page_type": page.get("page_type"),
                "page_width_pt": page.get("page_width_pt"),
                "page_height_pt": page.get("page_height_pt"),
                "geometry_proposals": summary,
            },
            ensure_ascii=False,
        )
    )
    content = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "ORIGINAL PAGE ORIENTATION"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(image_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "PAGE ROTATED 90 DEGREES"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(image_rotated),
                "detail": "original",
            },
        },
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _extractor_messages(
    page: dict,
    proposal: dict,
    detector_assessment: dict,
    crop_original: bytes,
    crop_rotated: bytes,
) -> list[dict]:
    system = (
        "You are extracting exactly one physical bill-of-material table from "
        "an industrial electrical drawing. The source may use any language, "
        "font or orientation. Understand columns from visual structure and "
        "table meaning, not a fixed vocabulary. Return every physical item row "
        "in visual order. Multi-line text inside one bordered row is one item. "
        "Preserve repeated rows and repeated values; never aggregate or dedupe. "
        "Preserve placeholders such as asterisks exactly. Do not invent quantity, "
        "unit, component tag, manufacturer, description or part number. If no "
        "quantity/position column exists, leave those fields empty. For every "
        "field, original preserves the visible source transcription. Normalized "
        "may repair only artificial spacing/word boundaries while preserving the "
        "same alphanumeric characters, masking symbols and order. Never translate, paraphrase, "
        "expand abbreviations, repair a source typo or alter a technical code. "
        "Every source word in an item row must belong to exactly one field_evidence "
        "entry. Use only supplied source word IDs. Return exactly one semantic "
        "source_column_roles decision for every supplied source column index, "
        "including geometry-recovered adjacent columns. Every deterministic row "
        "candidate must be classified exactly once: header rows belong only in "
        "header_row_candidate_ids, item rows belong only in rows, and all other "
        "rows belong only in non_item_rows."
    )
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "region_id": proposal["region_id"],
        "geometry_method": proposal["geometry_method"],
        "geometry_recovery": proposal.get("geometry_recovery") or {},
        "detector_assessment": detector_assessment,
        "table_bbox_pt": proposal["table_bbox_pt"],
        "original_table_bbox_pt": proposal.get("original_table_bbox_pt"),
        "crop_bbox_pt": proposal["crop_bbox_pt"],
        "row_candidates": proposal.get("row_candidates") or [],
        "fallback_page_word_ids": proposal.get("fallback_page_word_ids") or [],
        "fallback_page_words": proposal.get("fallback_page_words") or [],
    }
    content = [
        {
            "type": "text",
            "text": (
                "Extract the table completely and account for all physical rows "
                "and source evidence. Red labels identify deterministic row "
                "candidates when available.\n\n"
                + json.dumps(request, ensure_ascii=False)
            ),
        },
        {"type": "text", "text": "TABLE REGION IN SOURCE ORIENTATION"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(crop_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "TABLE REGION ROTATED 90 DEGREES"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(crop_rotated),
                "detail": "original",
            },
        },
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _verifier_messages(
    page: dict,
    proposals: list[dict],
    detector: dict,
    extractions: list[dict],
    page_original: bytes,
    page_rotated: bytes,
    region_images: dict[str, tuple[bytes, bytes]],
) -> list[dict]:
    system = (
        "You are the independent visual verifier of an industrial bill-of-material "
        "reader. Re-read the full page and every high-resolution table crop. The "
        "source may use any language or drawing standard. Verify every BOM table, "
        "physical item row, column, field and source token. Physical row identity "
        "is authoritative and consists of region plus source_row_candidate_id and "
        "visual_order. duplicate_physical_keys must contain only duplicated physical "
        "row identities, never repeated component tags, descriptions, part numbers "
        "or manufacturers. Identical field values on different physical rows are "
        "valid separate items and must be preserved. "
        "Do not merge accessory rows sharing one component tag. Verify that wrapped "
        "description text remains in its own row and no text leaks to adjacent rows. "
        "No visible token may disappear or be assigned to two fields. Original text "
        "must remain faithful. Normalized text may change spacing only while keeping "
        "the same alphanumeric content, masking symbols and order. Never translate, paraphrase, infer "
        "missing values, normalize a real source typo or repair a technical code. "
        "Provide a field_override only when the image supports one unambiguous exact "
        "transcription; otherwise require review. Evaluate the final candidate after "
        "applying every proposed override. An issue completely corrected by exact, "
        "high-confidence overrides must use resolution_status "
        "resolved_by_exact_overrides, list every related override, and must not remain "
        "high or critical. Keep resolution_status open for every unresolved problem. "
        "Return pass only when the post-override candidate accounts for all visible "
        "tables, item rows, columns and source evidence."
    )
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "detector": detector,
        "geometry_proposals": [
            {
                "region_id": proposal["region_id"],
                "geometry_method": proposal.get("geometry_method"),
                "deterministic_row_count": int(
                    proposal.get("deterministic_row_count") or 0
                ),
                "deterministic_column_count": int(
                    proposal.get("deterministic_column_count") or 0
                ),
                "geometry_recovery": proposal.get("geometry_recovery") or {},
            }
            for proposal in proposals
        ],
        "proposal_region_ids": [p["region_id"] for p in proposals],
        "extractions": extractions,
        "row_min_confidence": ROW_MIN_CONFIDENCE,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
    }
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Audit this page and every extracted row. Confirm exact row "
                "identity/order and component-tag sequence. Return overrides only "
                "when required by direct visual evidence, then judge the resulting "
                "post-override rows and explicitly link any resolved issue to its "
                "applied overrides.\n\n"
                + json.dumps(request, ensure_ascii=False)
            ),
        },
        {"type": "text", "text": "FULL PAGE ORIGINAL"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(page_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "FULL PAGE ROTATED 90 DEGREES"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(page_rotated),
                "detail": "original",
            },
        },
    ]
    for proposal in proposals:
        rid = proposal["region_id"]
        images = region_images.get(rid)
        if not images:
            continue
        content.extend(
            [
                {"type": "text", "text": f"REGION {rid} ORIGINAL"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url_png(images[0]),
                        "detail": "original",
                    },
                },
                {"type": "text", "text": f"REGION {rid} ROTATED 90 DEGREES"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url_png(images[1]),
                        "detail": "original",
                    },
                },
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _usage_accumulator() -> dict:
    return {
        "calls": 0,
        "reused_calls": 0,
        "new_input_tokens": 0,
        "new_output_tokens": 0,
        "new_reasoning_tokens": 0,
        "new_cost_usd": 0.0,
        "task_call_counts": {},
    }


def _add_usage(
    totals: dict,
    task: str,
    usage: dict,
    reused: bool,
) -> None:
    totals["calls"] += 1
    totals["reused_calls"] += 1 if reused else 0
    totals["task_call_counts"][task] = int(
        totals["task_call_counts"].get(task, 0)
    ) + 1
    if not reused:
        totals["new_input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["new_output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["new_reasoning_tokens"] += int(
            usage.get("reasoning_tokens") or 0
        )
        totals["new_cost_usd"] = round(
            float(totals["new_cost_usd"])
            + float(usage.get("cost_usd") or 0.0),
            6,
        )


def _normalize_issue(
    issue: Any,
    *,
    default_type: str,
    default_region_id: str = "",
    source_stage: str,
) -> dict:
    issue = issue if isinstance(issue, dict) else {}
    severity = str(issue.get("severity") or "warning").lower()
    if severity not in SEVERITIES:
        severity = "warning"
    return {
        "issue_type": _clean_text(
            issue.get("issue_type") or default_type,
            180,
        ),
        "severity": severity,
        "message": _clean_text(
            issue.get("message") or "BOM extraction issue",
            1600,
        ),
        "region_id": _clean_text(
            issue.get("region_id") or default_region_id,
            120,
        ),
        "row_ids": [
            _clean_text(x, 120)
            for x in (issue.get("row_ids") or [])
            if _clean_text(x, 120)
        ][:500],
        "confidence": max(
            0.0,
            min(1.0, float(issue.get("confidence") or 0.0)),
        ),
        "source_stage": source_stage,
    }


def _applied_override_keys(
    extractions: list[dict],
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        for row in extraction.get("rows") or []:
            row_id = _clean_text(row.get("row_id"), 120)
            for field_name, audit in (
                row.get("verifier_overrides") or {}
            ).items():
                if (
                    field_name in OVERRIDABLE_FIELDS
                    and isinstance(audit, dict)
                    and float(audit.get("confidence") or 0.0)
                    >= PAGE_PASS_MIN_CONFIDENCE
                ):
                    keys.add((rid, row_id, field_name))
    return keys


def _normalize_verifier_issue_after_overrides(
    issue: Any,
    *,
    extractions: list[dict],
) -> dict:
    normalized = _normalize_issue(
        issue,
        default_type="bom-verifier-issue",
        source_stage="verifier",
    )
    raw = issue if isinstance(issue, dict) else {}
    resolution_status = _clean_text(
        raw.get("resolution_status") or "open",
        80,
    )
    related: list[dict] = []
    related_keys: set[tuple[str, str, str]] = set()
    for ref in raw.get("related_overrides") or []:
        if not isinstance(ref, dict):
            continue
        rid = _clean_text(ref.get("region_id"), 120)
        row_id = _clean_text(ref.get("row_id"), 120)
        field_name = _clean_text(ref.get("field_name"), 120)
        if not rid or not row_id or field_name not in OVERRIDABLE_FIELDS:
            continue
        related.append(
            {
                "region_id": rid,
                "row_id": row_id,
                "field_name": field_name,
            }
        )
        related_keys.add((rid, row_id, field_name))

    applied = _applied_override_keys(extractions)
    resolution_validated = bool(
        resolution_status == "resolved_by_exact_overrides"
        and related_keys
        and related_keys.issubset(applied)
    )
    if resolution_validated:
        normalized["severity"] = "info"
        normalized["source_stage"] = "verifier_post_override_resolved"
    elif resolution_status == "resolved_by_exact_overrides":
        normalized["severity"] = "high"
        normalized["source_stage"] = "deterministic_validator"
        normalized["message"] = (
            normalized["message"]
            + " [Invalid post-override resolution claim: one or more exact "
            "high-confidence overrides were not applied.]"
        )[:1600]

    normalized["post_override_resolution"] = {
        "status": resolution_status,
        "related_overrides": related,
        "validated": resolution_validated,
    }
    return normalized


def _apply_overrides(
    extractions: list[dict],
    overrides: list[dict],
) -> None:
    lookup: dict[tuple[str, str], dict] = {}
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        for row in extraction.get("rows") or []:
            lookup[(rid, _clean_text(row.get("row_id"), 120))] = row
    for override in overrides or []:
        rid = _clean_text(override.get("region_id"), 120)
        row_id = _clean_text(override.get("row_id"), 120)
        field_name = _clean_text(override.get("field_name"), 120)
        confidence = float(override.get("confidence") or 0.0)
        row = lookup.get((rid, row_id))
        if not row or field_name not in OVERRIDABLE_FIELDS:
            continue
        if confidence < PAGE_PASS_MIN_CONFIDENCE:
            continue
        approved = _clean_text(override.get("approved_text"), 3000)
        row.setdefault("verifier_overrides", {})[field_name] = {
            "before": _clean_text(row.get(field_name), 3000),
            "after": approved,
            "confidence": confidence,
            "reason": _clean_text(override.get("reason"), 1200),
        }
        row[field_name] = approved


def _valid_transcription_override(
    row: dict,
    field_name: str,
) -> Optional[dict]:
    """Return a validated post-override transcription authority.

    Two authorities are accepted, both fail-closed:

    * deterministic source-word geometry, which may restore the physical
      left-to-right order of fragments from one source cell;
    * an exact high-confidence visual verifier override, but only when it
      preserves the complete multiset of source alphanumeric/masking
      characters. This permits CAD/PDF word-order repair without silently
      accepting substitutions, additions or deletions.
    """
    current = _clean_text(row.get(field_name), 5000)

    deterministic = (
        row.get("deterministic_overrides") or {}
    ).get(field_name)
    if isinstance(deterministic, dict):
        after = _clean_text(deterministic.get("after"), 5000)
        confidence = float(deterministic.get("confidence") or 0.0)
        if (
            after == current
            and confidence >= PAGE_PASS_MIN_CONFIDENCE
            and deterministic.get("validated") is True
            and deterministic.get("source_word_ids")
        ):
            return {
                "authority": "deterministic_source_geometry",
                **deterministic,
            }

    visual = (row.get("verifier_overrides") or {}).get(field_name)
    if not isinstance(visual, dict):
        return None
    before = _clean_text(visual.get("before"), 5000)
    after = _clean_text(visual.get("after"), 5000)
    confidence = float(visual.get("confidence") or 0.0)
    before_signature = _source_evidence_signature(before)
    after_signature = _source_evidence_signature(after)
    same_source_character_multiset = bool(
        before_signature
        and after_signature
        and sorted(before_signature) == sorted(after_signature)
    )
    if (
        after == current
        and confidence >= PAGE_PASS_MIN_CONFIDENCE
        and same_source_character_multiset
    ):
        return {
            "authority": "exact_visual_override_same_source_characters",
            **visual,
            "same_source_character_multiset": True,
        }
    return None


def _source_word_ids_for_field(
    row: dict,
    field_name: str,
) -> list[int]:
    matches = [
        item
        for item in (row.get("field_evidence") or [])
        if isinstance(item, dict)
        and _clean_text(item.get("field_name"), 120) == field_name
    ]
    if len(matches) != 1:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw_id in matches[0].get("source_word_ids") or []:
        try:
            word_id = int(raw_id)
        except Exception:
            continue
        if word_id in seen:
            return []
        seen.add(word_id)
        out.append(word_id)
    return out



def _signature_counter(value: Any) -> Counter:
    return Counter(_source_evidence_signature(value))


def _field_evidence_entry(
    row: dict,
    field_name: str,
) -> Optional[dict]:
    matches = [
        item
        for item in (row.get("field_evidence") or [])
        if isinstance(item, dict)
        and _clean_text(item.get("field_name"), 120) == field_name
    ]
    return matches[0] if len(matches) == 1 else None


def _unique_source_word_subset_for_counter(
    *,
    source_word_ids: list[int],
    target_counter: Counter,
    word_map: dict[int, dict],
    maximum_words: int = 4,
) -> Optional[list[int]]:
    """Find one unambiguous small set of whole source words.

    Cross-column CAD segmentation errors normally move one short vector word
    across an adjacent column boundary. The rescue intentionally refuses
    character slicing inside a word and refuses ambiguous solutions.
    """
    target_signature = "".join(target_counter.elements())
    if not target_signature or len(target_signature) > 64:
        return None

    candidates: list[tuple[int, Counter]] = []
    for raw_id in source_word_ids:
        try:
            word_id = int(raw_id)
        except Exception:
            continue
        word = word_map.get(word_id)
        if not word:
            continue
        signature = _source_evidence_signature(word.get("text"))
        if not signature:
            continue
        counter = Counter(signature)
        # The whole word must fit inside the exact transferred character set.
        if counter - target_counter:
            continue
        candidates.append((word_id, counter))

    solutions: list[list[int]] = []
    for size in range(1, min(maximum_words, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            combined: Counter = Counter()
            for _word_id, counter in combo:
                combined.update(counter)
            if combined == target_counter:
                solutions.append([word_id for word_id, _ in combo])
                if len(solutions) > 1:
                    return None
    return solutions[0] if len(solutions) == 1 else None


def _word_ids_are_at_shared_column_edge(
    *,
    moved_word_ids: list[int],
    donor_word_ids: list[int],
    donor_column_index: int,
    receiver_column_index: int,
    word_map: dict[int, dict],
) -> bool:
    """Require transferred whole words to sit on the donor's shared edge."""
    if abs(donor_column_index - receiver_column_index) != 1:
        return False
    if not moved_word_ids or not donor_word_ids:
        return False
    if any(word_id not in word_map for word_id in donor_word_ids):
        return False

    ordered = sorted(
        donor_word_ids,
        key=lambda word_id: (
            float(word_map[word_id].get("x0") or 0.0),
            float(word_map[word_id].get("y0") or 0.0),
            word_id,
        ),
    )
    moved = set(moved_word_ids)
    if donor_column_index > receiver_column_index:
        edge_ids = set(ordered[: len(moved_word_ids)])
    else:
        edge_ids = set(ordered[-len(moved_word_ids) :])
    return moved == edge_ids


def _reconcile_cross_field_evidence_transfers(
    *,
    row: dict,
    word_map: dict[int, dict],
) -> list[dict]:
    """Reassign exact whole-word evidence across one adjacent column boundary.

    A CAD/PDF vector stream can place the last word of one visible cell in the
    neighboring cell. This adjudicator activates only when two high-confidence
    verifier overrides form an exact closed transfer: one field removes a
    character multiset and an adjacent field adds precisely the same multiset.
    It then requires one unique whole-word subset at the donor's shared edge,
    moves only those word IDs, and revalidates both final field transcriptions.
    Anything ambiguous remains untouched and therefore continues to block.
    """
    override_map = row.get("verifier_overrides") or {}
    if not isinstance(override_map, dict):
        return []

    all_assigned_before: list[int] = []
    for evidence in row.get("field_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        for raw_id in evidence.get("source_word_ids") or []:
            try:
                all_assigned_before.append(int(raw_id))
            except Exception:
                continue
    if len(all_assigned_before) != len(set(all_assigned_before)):
        return []

    donors: list[dict] = []
    receivers: list[dict] = []
    for field_name in ORIGINAL_FIELDS:
        audit = override_map.get(field_name)
        evidence = _field_evidence_entry(row, field_name)
        if not isinstance(audit, dict) or not isinstance(evidence, dict):
            continue
        confidence = float(audit.get("confidence") or 0.0)
        if confidence < PAGE_PASS_MIN_CONFIDENCE:
            continue
        before_counter = _signature_counter(audit.get("before"))
        after_counter = _signature_counter(audit.get("after"))
        removed = before_counter - after_counter
        added = after_counter - before_counter
        try:
            column_index = int(evidence.get("source_column_index"))
        except Exception:
            continue
        base = {
            "field_name": field_name,
            "audit": audit,
            "evidence": evidence,
            "column_index": column_index,
            "confidence": confidence,
        }
        if removed and not added:
            donors.append({**base, "transfer_counter": removed})
        elif added and not removed:
            receivers.append({**base, "transfer_counter": added})

    transfers: list[dict] = []
    used_donor_fields: set[str] = set()
    used_receiver_fields: set[str] = set()

    for receiver in receivers:
        matching_donors = [
            donor
            for donor in donors
            if donor["field_name"] not in used_donor_fields
            and donor["transfer_counter"] == receiver["transfer_counter"]
            and abs(
                int(donor["column_index"])
                - int(receiver["column_index"])
            )
            == 1
        ]
        if len(matching_donors) != 1:
            continue
        donor = matching_donors[0]
        donor_field = str(donor["field_name"])
        receiver_field = str(receiver["field_name"])
        if receiver_field in used_receiver_fields:
            continue

        donor_ids = _source_word_ids_for_field(row, donor_field)
        receiver_ids = _source_word_ids_for_field(row, receiver_field)
        moved_ids = _unique_source_word_subset_for_counter(
            source_word_ids=donor_ids,
            target_counter=receiver["transfer_counter"],
            word_map=word_map,
        )
        if not moved_ids:
            continue
        if not _word_ids_are_at_shared_column_edge(
            moved_word_ids=moved_ids,
            donor_word_ids=donor_ids,
            donor_column_index=int(donor["column_index"]),
            receiver_column_index=int(receiver["column_index"]),
            word_map=word_map,
        ):
            continue

        donor_after_ids = [
            word_id for word_id in donor_ids if word_id not in set(moved_ids)
        ]
        receiver_after_ids = sorted(
            set(receiver_ids + moved_ids),
            key=lambda word_id: (
                float(word_map.get(word_id, {}).get("y0") or 0.0),
                float(word_map.get(word_id, {}).get("x0") or 0.0),
                word_id,
            ),
        )

        donor_entry = donor["evidence"]
        receiver_entry = receiver["evidence"]
        donor_before_ids = list(donor_entry.get("source_word_ids") or [])
        receiver_before_ids = list(receiver_entry.get("source_word_ids") or [])
        donor_entry["source_word_ids"] = donor_after_ids
        receiver_entry["source_word_ids"] = receiver_after_ids

        all_assigned_after: list[int] = []
        for evidence in row.get("field_evidence") or []:
            if not isinstance(evidence, dict):
                continue
            for raw_id in evidence.get("source_word_ids") or []:
                try:
                    all_assigned_after.append(int(raw_id))
                except Exception:
                    continue

        donor_source_signature = _source_evidence_signature(
            _text_for_ids(donor_after_ids, word_map, 5000)
        )
        receiver_source_signature = _source_evidence_signature(
            _text_for_ids(receiver_after_ids, word_map, 5000)
        )
        donor_field_signature = _source_evidence_signature(
            row.get(donor_field)
        )
        receiver_field_signature = _source_evidence_signature(
            row.get(receiver_field)
        )
        assignment_is_exact = bool(
            sorted(all_assigned_after) == sorted(all_assigned_before)
            and len(all_assigned_after) == len(set(all_assigned_after))
            and donor_source_signature == donor_field_signature
            and receiver_source_signature == receiver_field_signature
        )
        if not assignment_is_exact:
            donor_entry["source_word_ids"] = donor_before_ids
            receiver_entry["source_word_ids"] = receiver_before_ids
            continue

        transfer_signature = "".join(
            receiver["transfer_counter"].elements()
        )
        transfer_id = hashlib.sha256(
            "|".join(
                [
                    _clean_text(row.get("row_id"), 120),
                    donor_field,
                    receiver_field,
                    ",".join(str(word_id) for word_id in moved_ids),
                    transfer_signature,
                ]
            ).encode("utf-8")
        ).hexdigest()[:20]
        confidence = min(
            float(donor["confidence"]),
            float(receiver["confidence"]),
        )
        moved_tokens = [
            _clean_text(word_map[word_id].get("text"), 500)
            for word_id in moved_ids
        ]
        audit = {
            "version": "paired-cross-field-source-evidence-transfer-v1",
            "transfer_id": transfer_id,
            "donor_field": donor_field,
            "receiver_field": receiver_field,
            "donor_column_index": int(donor["column_index"]),
            "receiver_column_index": int(receiver["column_index"]),
            "moved_source_word_ids": moved_ids,
            "moved_source_tokens": moved_tokens,
            "transferred_source_signature": transfer_signature,
            "confidence": confidence,
            "validated": True,
        }
        row.setdefault("cross_field_evidence_transfers", []).append(audit)

        for field_name, field_ids, field_source_signature in [
            (donor_field, donor_after_ids, donor_source_signature),
            (receiver_field, receiver_after_ids, receiver_source_signature),
        ]:
            visual = override_map.get(field_name) or {}
            row.setdefault("deterministic_overrides", {})[field_name] = {
                "before": _clean_text(visual.get("before"), 5000),
                "after": _clean_text(row.get(field_name), 5000),
                "confidence": confidence,
                "reason": (
                    "A closed pair of exact verifier overrides transferred one "
                    "unambiguous whole source-word group across the shared edge "
                    "of two adjacent physical columns."
                ),
                "method": "paired_cross_field_source_evidence_transfer_v1",
                "source_word_ids": field_ids,
                "source_signature": field_source_signature,
                "cross_field_transfer_id": transfer_id,
                "validated": True,
            }
            if isinstance(visual, dict):
                visual["cross_field_transfer_validated"] = True
                visual["cross_field_transfer_id"] = transfer_id

        transfers.append(audit)
        used_donor_fields.add(donor_field)
        used_receiver_fields.add(receiver_field)

    return transfers


def _single_line_source_code_candidate(
    *,
    row: dict,
    field_name: str,
    word_map: dict[int, dict],
) -> Optional[dict]:
    """Reconstruct a fragmented technical code from exact source geometry.

    This does not use page numbers, language, manufacturers, known codes or
    fixed coordinates. It operates only on the word IDs already bound to one
    canonical code field and requires a single horizontal visual line. The
    candidate must contain exactly the same source characters as the original
    vector transcription; only their physical left-to-right order may change.
    """
    ids = _source_word_ids_for_field(row, field_name)
    if len(ids) < 2 or any(word_id not in word_map for word_id in ids):
        return None

    words = [word_map[word_id] for word_id in ids]
    centers_y = [
        (float(word["y0"]) + float(word["y1"])) / 2.0
        for word in words
    ]
    heights = [
        max(0.1, float(word["y1"]) - float(word["y0"]))
        for word in words
    ]
    line_tolerance = max(3.0, max(heights) * 1.25)
    if max(centers_y) - min(centers_y) > line_tolerance:
        return None

    ordered = sorted(
        words,
        key=lambda word: (
            float(word["x0"]),
            float(word["y0"]),
            int(word.get("id") or 0),
        ),
    )
    tokens = [
        re.sub(r"\s+", "", _clean_text(word.get("text"), 500))
        for word in ordered
    ]
    if any(not token for token in tokens):
        return None
    candidate = "".join(tokens)
    if not candidate or len(candidate) > 300:
        return None

    candidate_signature = _source_evidence_signature(candidate)
    if not candidate_signature:
        return None

    visual_audit = (row.get("verifier_overrides") or {}).get(field_name)
    source_before = (
        _clean_text(visual_audit.get("before"), 5000)
        if isinstance(visual_audit, dict)
        else _clean_text(row.get(field_name), 5000)
    )
    source_before_signature = _source_evidence_signature(source_before)
    current_signature = _source_evidence_signature(row.get(field_name))

    # The exact vector characters must be preserved. Only their sequence may
    # be repaired from physical x geometry.
    if (
        not source_before_signature
        or sorted(source_before_signature) != sorted(candidate_signature)
        or current_signature == candidate_signature
    ):
        return None

    return {
        "candidate": candidate,
        "source_word_ids": [int(word.get("id") or 0) for word in ordered],
        "source_tokens": tokens,
        "source_before": source_before,
        "source_before_signature": source_before_signature,
        "candidate_signature": candidate_signature,
        "method": "single_line_source_x_order_v1",
        "confidence": 1.0,
        "validated": True,
    }


def _reconcile_post_override_rows(
    *,
    extractions: list[dict],
    word_map: dict[int, dict],
) -> dict:
    """Finalize rows after exact verifier overrides, before publication gates."""
    source_order_rows: list[str] = []
    normalized_rows: list[str] = []
    cross_field_rows: list[str] = []
    cross_field_transfers: list[dict] = []

    for extraction in extractions:
        for row in extraction.get("rows") or []:
            row_id = _clean_text(row.get("row_id"), 120)

            transfers = _reconcile_cross_field_evidence_transfers(
                row=row,
                word_map=word_map,
            )
            if transfers:
                cross_field_rows.append(row_id)
                cross_field_transfers.extend(
                    [
                        {
                            "row_id": row_id,
                            **transfer,
                        }
                        for transfer in transfers
                    ]
                )

            # Part numbers are technical codes. If the extractor emitted their
            # fragments in non-physical order, exact source x geometry is a
            # stronger authority than model token order.
            field_name = "part_number_original"
            candidate = _single_line_source_code_candidate(
                row=row,
                field_name=field_name,
                word_map=word_map,
            )
            if candidate:
                before = _clean_text(row.get(field_name), 5000)
                row.setdefault("deterministic_overrides", {})[field_name] = {
                    "before": before,
                    "after": candidate["candidate"],
                    "confidence": candidate["confidence"],
                    "reason": (
                        "Exact source-word x geometry restored the physical "
                        "left-to-right order of one technical-code cell."
                    ),
                    "method": candidate["method"],
                    "source_word_ids": candidate["source_word_ids"],
                    "source_tokens": candidate["source_tokens"],
                    "source_before_signature": candidate[
                        "source_before_signature"
                    ],
                    "candidate_signature": candidate[
                        "candidate_signature"
                    ],
                    "validated": True,
                }
                visual = (row.get("verifier_overrides") or {}).get(field_name)
                if isinstance(visual, dict):
                    visual["superseded_by"] = candidate["method"]
                row[field_name] = candidate["candidate"]
                source_order_rows.append(row_id)

            # Once an original field has a validated exact authority, a stale
            # pre-correction normalized field must not survive publication.
            for base_name in BASE_FIELDS:
                original_field = f"{base_name}_original"
                normalized_field = f"{base_name}_normalized"
                authority = _valid_transcription_override(
                    row,
                    original_field,
                )
                if not authority:
                    continue
                original = _clean_text(row.get(original_field), 5000)
                normalized = _clean_text(row.get(normalized_field), 5000)
                if normalized == original:
                    continue
                row.setdefault(
                    "post_override_normalization",
                    {},
                )[normalized_field] = {
                    "before": normalized,
                    "after": original,
                    "authority": authority.get("authority"),
                    "source_original_field": original_field,
                    "validated": True,
                }
                row[normalized_field] = original
                normalized_rows.append(row_id)

    return {
        "version": "bom-post-override-reconciliation-v2",
        "source_order_row_ids": sorted(set(source_order_rows)),
        "normalized_row_ids": sorted(set(normalized_rows)),
        "cross_field_transfer_row_ids": sorted(set(cross_field_rows)),
        "cross_field_evidence_transfers": cross_field_transfers,
    }


def _post_override_text_issue_resolution(
    *,
    raw_issue: dict,
    normalized_issue: dict,
    extractions: list[dict],
) -> dict:
    """Resolve only specifically provable post-override text findings."""
    if normalized_issue.get("severity") not in {"high", "critical"}:
        return normalized_issue

    issue_type = _clean_text(raw_issue.get("issue_type"), 180)
    row_ids = [
        _clean_text(value, 120)
        for value in (raw_issue.get("row_ids") or [])
        if _clean_text(value, 120)
    ]
    row_lookup = {
        _clean_text(row.get("row_id"), 120): row
        for extraction in extractions
        for row in (extraction.get("rows") or [])
        if _clean_text(row.get("row_id"), 120)
    }

    resolved = False
    evidence: list[dict] = []

    if issue_type in {
        "normalized_text_not_spacing_only",
        "normalized_fields_not_faithful_to_source",
    } and row_ids:
        resolved = True
        for row_id in row_ids:
            row = row_lookup.get(row_id)
            if not row:
                resolved = False
                break
            coverage = row.get("source_evidence_coverage") or {}
            if coverage.get("complete") is not True:
                resolved = False
                break
            row_evidence: list[dict] = []
            for base_name in BASE_FIELDS:
                original = _clean_text(
                    row.get(f"{base_name}_original"),
                    5000,
                )
                normalized = _clean_text(
                    row.get(f"{base_name}_normalized"),
                    5000,
                )
                if bool(original) != bool(normalized):
                    resolved = False
                    break
                if (
                    original
                    and _semantic_character_signature(original)
                    != _semantic_character_signature(normalized)
                ):
                    resolved = False
                    break
                authority = _valid_transcription_override(
                    row,
                    f"{base_name}_original",
                )
                if authority and normalized != original:
                    resolved = False
                    break
                if authority:
                    row_evidence.append(
                        {
                            "field": base_name,
                            "authority": authority.get("authority"),
                        }
                    )
            if not resolved:
                break
            evidence.append({"row_id": row_id, "fields": row_evidence})

    elif issue_type == "reading_order_error_in_part_number" and row_ids:
        resolved = True
        for row_id in row_ids:
            row = row_lookup.get(row_id)
            if not row:
                resolved = False
                break
            audit = (
                row.get("deterministic_overrides") or {}
            ).get("part_number_original")
            original = _clean_text(
                row.get("part_number_original"),
                5000,
            )
            normalized = _clean_text(
                row.get("part_number_normalized"),
                5000,
            )
            if not (
                isinstance(audit, dict)
                and audit.get("validated") is True
                and float(audit.get("confidence") or 0.0)
                >= PAGE_PASS_MIN_CONFIDENCE
                and _clean_text(audit.get("after"), 5000) == original
                and normalized == original
                and audit.get("source_word_ids")
            ):
                resolved = False
                break
            evidence.append(
                {
                    "row_id": row_id,
                    "method": audit.get("method"),
                    "source_word_ids": audit.get("source_word_ids"),
                    "final_value": original,
                }
            )

    if not resolved:
        return normalized_issue

    normalized_issue = dict(normalized_issue)
    normalized_issue["severity"] = "info"
    normalized_issue["source_stage"] = (
        "deterministic_post_override_adjudicator"
    )
    normalized_issue["post_override_resolution"] = {
        "status": "resolved_by_post_override_revalidation",
        "validated": True,
        "issue_type": issue_type,
        "evidence": evidence,
    }
    return normalized_issue


def _field_evidence_audit(
    *,
    row: dict,
    expected_word_ids: list[int],
    word_map: dict[int, dict],
) -> dict:
    expected = [int(x) for x in expected_word_ids if int(x) in word_map]
    expected_set = set(expected)
    assignments: dict[int, list[str]] = {}
    invalid_ids: list[int] = []
    evidence_by_field: dict[str, list[int]] = {}

    for item in row.get("field_evidence") or []:
        if not isinstance(item, dict):
            continue
        field_name = _clean_text(item.get("field_name"), 120)
        if field_name not in ORIGINAL_FIELDS:
            continue
        ids: list[int] = []
        for raw_id in item.get("source_word_ids") or []:
            try:
                wid = int(raw_id)
            except Exception:
                continue
            if wid not in word_map or wid not in expected_set:
                invalid_ids.append(wid)
                continue
            ids.append(wid)
            assignments.setdefault(wid, []).append(field_name)
        evidence_by_field.setdefault(field_name, []).extend(ids)

    duplicated_ids = sorted(
        wid for wid, fields in assignments.items() if len(fields) > 1
    )
    assigned_set = set(assignments)
    missing_ids = sorted(expected_set - assigned_set)
    missing_field_evidence = sorted(
        field_name
        for field_name in ORIGINAL_FIELDS
        if _clean_text(row.get(field_name), 5000)
        and not evidence_by_field.get(field_name)
    )

    unresolved_field_text_mismatches: list[dict] = []
    adjudicated_field_text_mismatches: list[dict] = []
    for field_name, ids in evidence_by_field.items():
        source_text = _text_for_ids(sorted(set(ids)), word_map, 5000)
        source_signature = _source_evidence_signature(source_text)
        field_signature = _source_evidence_signature(row.get(field_name))
        if source_signature == field_signature:
            continue
        mismatch = {
            "field_name": field_name,
            "source_text": source_text,
            "field_text": _clean_text(row.get(field_name), 5000),
            "source_signature": source_signature,
            "field_signature": field_signature,
        }
        authority = _valid_transcription_override(row, field_name)
        if authority:
            mismatch["post_override_authority"] = authority
            adjudicated_field_text_mismatches.append(mismatch)
        else:
            unresolved_field_text_mismatches.append(mismatch)

    return {
        "expected_word_ids": sorted(expected_set),
        "assigned_word_ids": sorted(assigned_set),
        "missing_word_ids": missing_ids,
        "duplicated_word_ids": duplicated_ids,
        "invalid_word_ids": sorted(set(invalid_ids)),
        "missing_field_evidence": missing_field_evidence,
        "field_text_mismatches": unresolved_field_text_mismatches,
        "adjudicated_field_text_mismatches": (
            adjudicated_field_text_mismatches
        ),
        "complete": not any(
            [
                missing_ids,
                duplicated_ids,
                invalid_ids,
                missing_field_evidence,
                unresolved_field_text_mismatches,
            ]
        ),
    }

def _parse_quantity(value: Any) -> Optional[Decimal]:
    text = _clean_text(value, 120)
    if not text:
        return None
    compact = text.replace(" ", "")
    if re.fullmatch(r"\d+(?:[.,]\d{1,3})?", compact) is None:
        return None
    try:
        parsed = Decimal(compact.replace(",", "."))
    except InvalidOperation:
        return None
    if parsed < 0:
        return None
    return parsed


def _published_value(row: dict, base_name: str) -> str:
    normalized = _clean_text(row.get(f"{base_name}_normalized"), 4000)
    original = _clean_text(row.get(f"{base_name}_original"), 4000)
    return normalized or original


def _physical_bom_key(
    *,
    version_id: int,
    page_id: int,
    region_id: str,
    visual_order: int,
) -> str:
    # Values are intentionally excluded so duplicate source rows remain separate.
    return hashlib.sha256(
        "|".join(
            [
                str(version_id),
                str(page_id),
                region_id,
                str(visual_order),
            ]
        ).encode("utf-8")
    ).hexdigest()



def _canonical_row_candidate_accounting(
    extraction: dict,
) -> tuple[set[str], set[str], set[str]]:
    """Return mutually exclusive header, non-item and item candidate IDs.

    Older/less precise model responses can echo a header in both the dedicated
    ``header_row_candidate_ids`` list and ``non_item_rows`` with kind=header.
    That is one semantic classification, not two physical rows. Header aliases
    are therefore canonicalized before the exact-once accounting check. Any
    overlap with an item row or with a genuine non-header non-item row remains
    blocking.
    """
    explicit_headers = {
        _clean_text(value, 120)
        for value in (extraction.get("header_row_candidate_ids") or [])
        if _clean_text(value, 120)
    }
    non_item_rows = [
        item
        for item in (extraction.get("non_item_rows") or [])
        if isinstance(item, dict)
    ]
    header_aliases = {
        _clean_text(item.get("source_row_candidate_id"), 120)
        for item in non_item_rows
        if _clean_text(item.get("kind"), 120) == "header"
        and _clean_text(item.get("source_row_candidate_id"), 120)
    }
    headers = explicit_headers | header_aliases
    non_items = {
        _clean_text(item.get("source_row_candidate_id"), 120)
        for item in non_item_rows
        if _clean_text(item.get("kind"), 120) != "header"
        and _clean_text(item.get("source_row_candidate_id"), 120)
    }
    items = {
        _clean_text(item.get("source_row_candidate_id"), 120)
        for item in (extraction.get("rows") or [])
        if isinstance(item, dict)
        and _clean_text(item.get("source_row_candidate_id"), 120)
    }
    return headers, non_items, items


def _sequence_matches_source_characters(
    actual_values: list[Any],
    verified_values: list[Any],
) -> bool:
    """Compare visual sequences while tolerating artificial PDF whitespace."""
    if len(actual_values) != len(verified_values):
        return False
    return all(
        _semantic_character_signature(actual)
        == _semantic_character_signature(verified)
        for actual, verified in zip(actual_values, verified_values)
    )


def _component_tag_sequence_source_exact_adjudication(
    *,
    actual_rows: list[dict],
    verified_row_ids: list[str],
    verified_tags: list[str],
    word_map: dict[int, dict],
) -> dict:
    """Adjudicate verifier tag punctuation only from exact source evidence.

    The verifier can occasionally compact a source identifier for display and
    omit punctuation that is genuinely printed in the BOM tag column. The
    source value must never be rewritten merely to match that compact sequence.

    Adjudication is allowed only when row count and row identity already match,
    every differing tag has the same alphanumeric/masking content, and the
    published tag (including punctuation) is independently identical to the
    exact vector words assigned to ``component_tag_original``. Any missing,
    duplicated, invalid or non-exact field evidence keeps the mismatch blocking.
    """
    actual_row_ids = [
        _clean_text(row.get("row_id"), 120)
        for row in actual_rows
    ]
    result = {
        "version": "source-exact-component-tag-sequence-v1",
        "validated": False,
        "row_identity_matches": False,
        "exact_match_count": 0,
        "source_authoritative_difference_count": 0,
        "source_authoritative_row_ids": [],
        "differences": [],
        "failure_reason": "",
    }

    if (
        len(actual_rows) != len(verified_tags)
        or len(actual_rows) != len(verified_row_ids)
    ):
        result["failure_reason"] = "row_count_mismatch"
        return result
    if actual_row_ids != verified_row_ids:
        result["failure_reason"] = "row_identity_or_order_mismatch"
        return result
    result["row_identity_matches"] = True

    differences: list[dict] = []
    exact_match_count = 0
    for row, verified_tag in zip(actual_rows, verified_tags):
        row_id = _clean_text(row.get("row_id"), 120)
        original_tag = _clean_text(
            row.get("component_tag_original"),
            1000,
        )
        published_tag = _clean_text(
            row.get("component_tag_normalized") or original_tag,
            1000,
        )
        verified_tag = _clean_text(verified_tag, 1000)

        published_signature = _semantic_character_signature(published_tag)
        verified_signature = _semantic_character_signature(verified_tag)
        if published_signature == verified_signature:
            exact_match_count += 1
            continue

        if not original_tag or not published_tag or not verified_tag:
            result["failure_reason"] = (
                f"empty_differing_component_tag:{row_id}"
            )
            return result
        if (
            _semantic_character_signature(original_tag)
            != published_signature
        ):
            result["failure_reason"] = (
                f"published_tag_not_source_character_equivalent:{row_id}"
            )
            return result

        # The disagreement may contain punctuation/spacing only. Any
        # alphanumeric or masking-symbol difference remains blocking.
        source_atom = _source_evidence_signature(original_tag)
        verifier_atom = _source_evidence_signature(verified_tag)
        if not source_atom or source_atom != verifier_atom:
            result["failure_reason"] = (
                f"non_punctuation_component_tag_difference:{row_id}"
            )
            return result

        evidence_entries = [
            item
            for item in (row.get("field_evidence") or [])
            if isinstance(item, dict)
            and _clean_text(item.get("field_name"), 120)
            == "component_tag_original"
        ]
        if len(evidence_entries) != 1:
            result["failure_reason"] = (
                f"component_tag_evidence_not_unique:{row_id}"
            )
            return result
        evidence = evidence_entries[0]
        try:
            source_column_index = int(evidence.get("source_column_index"))
        except Exception:
            result["failure_reason"] = (
                f"component_tag_source_column_invalid:{row_id}"
            )
            return result

        source_word_ids: list[int] = []
        for raw_id in evidence.get("source_word_ids") or []:
            try:
                source_word_ids.append(int(raw_id))
            except Exception:
                result["failure_reason"] = (
                    f"component_tag_word_id_invalid:{row_id}"
                )
                return result
        if (
            not source_word_ids
            or len(source_word_ids) != len(set(source_word_ids))
            or any(word_id not in word_map for word_id in source_word_ids)
        ):
            result["failure_reason"] = (
                f"component_tag_word_evidence_invalid:{row_id}"
            )
            return result

        declared_row_word_ids = {
            int(raw_id)
            for raw_id in (row.get("source_word_ids") or [])
            if isinstance(raw_id, int) or str(raw_id).isdigit()
        }
        if not set(source_word_ids).issubset(declared_row_word_ids):
            result["failure_reason"] = (
                f"component_tag_word_evidence_outside_row:{row_id}"
            )
            return result

        source_text = _text_for_ids(
            source_word_ids,
            word_map,
            1000,
        )
        if (
            _semantic_character_signature(source_text)
            != _semantic_character_signature(original_tag)
        ):
            result["failure_reason"] = (
                f"component_tag_not_exact_vector_source:{row_id}"
            )
            return result

        coverage = row.get("source_evidence_coverage") or {}
        if coverage and coverage.get("complete") is not True:
            result["failure_reason"] = (
                f"row_source_evidence_not_complete:{row_id}"
            )
            return result
        if float(row.get("confidence") or 0.0) < ROW_MIN_CONFIDENCE:
            result["failure_reason"] = (
                f"component_tag_row_confidence_below_threshold:{row_id}"
            )
            return result

        audit = {
            "row_id": row_id,
            "source_column_index": source_column_index,
            "source_word_ids": source_word_ids,
            "source_text_original": source_text,
            "published_component_tag": published_tag,
            "verifier_component_tag": verified_tag,
            "source_exact_signature": (
                _semantic_character_signature(source_text)
            ),
            "shared_alphanumeric_signature": source_atom,
            "difference_kind": "verifier_punctuation_or_spacing_omission",
            "source_authority": "exact_component_tag_vector_evidence",
            "validated": True,
        }
        row["component_tag_sequence_adjudication"] = {
            "version": result["version"],
            **audit,
        }
        differences.append(audit)

    result["exact_match_count"] = exact_match_count
    result["source_authoritative_difference_count"] = len(differences)
    result["source_authoritative_row_ids"] = [
        item["row_id"] for item in differences
    ]
    result["differences"] = differences
    result["validated"] = True
    return result


def _adjudicate_verifier_duplicate_physical_keys(
    *,
    reported_keys: list[Any],
    rows: list[dict],
) -> tuple[list[str], list[dict]]:
    """Separate real duplicate row identities from repeated BOM values.

    The deterministic physical key intentionally excludes component values. A
    verifier occasionally reports repeated component tags in the field named
    ``duplicate_physical_keys``. When every row identity is unique and the
    reported token is demonstrably a value repeated on distinct rows, preserve
    it as audit information rather than blocking publication. Unknown tokens or
    actual duplicate row identities remain unresolved.
    """
    cleaned = [
        _clean_text(value, 500)
        for value in (reported_keys or [])
        if _clean_text(value, 500)
    ]
    if not cleaned:
        return [], []

    physical_identities = [
        (
            _clean_text(row.get("region_id"), 120),
            _clean_text(row.get("source_row_candidate_id"), 120),
            int(row.get("visual_order") or 0),
        )
        for row in rows
    ]
    identities_unique = len(physical_identities) == len(
        set(physical_identities)
    )

    repeated_values: dict[str, dict] = {}
    for field_name in ORIGINAL_FIELDS:
        occurrences: dict[str, list[dict]] = {}
        for row in rows:
            value = _clean_text(row.get(field_name), 4000)
            atom = _evidence_atom(value)
            if not atom:
                continue
            occurrences.setdefault(atom, []).append(
                {
                    "region_id": _clean_text(row.get("region_id"), 120),
                    "row_id": _clean_text(row.get("row_id"), 120),
                    "source_row_candidate_id": _clean_text(
                        row.get("source_row_candidate_id"), 120
                    ),
                    "visual_order": int(row.get("visual_order") or 0),
                    "value": value,
                }
            )
        for atom, matches in occurrences.items():
            if len(matches) >= 2:
                repeated_values.setdefault(
                    atom,
                    {"fields": set(), "rows": []},
                )
                repeated_values[atom]["fields"].add(field_name)
                repeated_values[atom]["rows"].extend(matches)

    unresolved: list[str] = []
    resolved: list[dict] = []
    for reported in cleaned:
        atom = _evidence_atom(reported)
        evidence = repeated_values.get(atom)
        if identities_unique and evidence:
            unique_rows = {
                (
                    item["region_id"],
                    item["source_row_candidate_id"],
                    item["visual_order"],
                )
                for item in evidence["rows"]
            }
            if len(unique_rows) >= 2:
                resolved.append(
                    {
                        "reported_value": reported,
                        "matched_fields": sorted(evidence["fields"]),
                        "distinct_physical_rows": len(unique_rows),
                        "row_examples": evidence["rows"][:12],
                    }
                )
                continue
        unresolved.append(reported)
    return unresolved, resolved

def _validate_page(
    *,
    page: dict,
    proposals: list[dict],
    detector: dict,
    extractions: list[dict],
    verifier: dict,
    word_map: dict[int, dict],
) -> tuple[bool, list[dict], list[dict]]:
    issues: list[dict] = []
    rows: list[dict] = []

    reconciliation = _reconcile_post_override_rows(
        extractions=extractions,
        word_map=word_map,
    )
    if reconciliation["cross_field_transfer_row_ids"]:
        issues.append(
            {
                "issue_type": (
                    "bom-cross-field-source-evidence-reconciled-post-override"
                ),
                "severity": "info",
                "message": (
                    "Exact paired verifier corrections moved unambiguous whole "
                    "source words across adjacent physical column boundaries "
                    "before publication."
                ),
                "region_id": "",
                "row_ids": reconciliation[
                    "cross_field_transfer_row_ids"
                ],
                "confidence": 1.0,
                "source_stage": (
                    "deterministic_post_override_adjudicator"
                ),
                "post_override_resolution": reconciliation,
            }
        )
    if reconciliation["source_order_row_ids"]:
        issues.append(
            {
                "issue_type": "bom-source-order-reconciled-post-override",
                "severity": "info",
                "message": (
                    "Exact source-word geometry restored one or more "
                    "technical-code fragment sequences before publication."
                ),
                "region_id": "",
                "row_ids": reconciliation["source_order_row_ids"],
                "confidence": 1.0,
                "source_stage": "deterministic_post_override_adjudicator",
                "post_override_resolution": reconciliation,
            }
        )
    if reconciliation["normalized_row_ids"]:
        issues.append(
            {
                "issue_type": "bom-normalized-fields-synchronized-post-override",
                "severity": "info",
                "message": (
                    "Normalized display values were synchronized from their "
                    "validated final original transcriptions."
                ),
                "region_id": "",
                "row_ids": reconciliation["normalized_row_ids"],
                "confidence": 1.0,
                "source_stage": "deterministic_post_override_adjudicator",
                "post_override_resolution": reconciliation,
            }
        )

    proposal_by_id = {p["region_id"]: p for p in proposals}
    proposal_ids = [str(p["region_id"]) for p in proposals]

    raw_assessments = [
        x
        for x in (detector.get("proposal_assessments") or [])
        if isinstance(x, dict)
    ]
    assessment_ids = [
        _clean_text(x.get("region_id"), 120)
        for x in raw_assessments
    ]
    assessments = {
        _clean_text(x.get("region_id"), 120): x
        for x in raw_assessments
        if _clean_text(x.get("region_id"), 120)
    }

    raw_extractions = [x for x in extractions if isinstance(x, dict)]
    extraction_ids = [
        _clean_text(x.get("region_id"), 120)
        for x in raw_extractions
    ]
    extraction_by_id = {
        _clean_text(x.get("region_id"), 120): x
        for x in raw_extractions
        if _clean_text(x.get("region_id"), 120)
    }

    if int(detector.get("page_id") or 0) != int(page["id"]):
        issues.append(
            {
                "issue_type": "bom-detector-page-id-mismatch",
                "severity": "high",
                "message": "Detector returned a different page_id",
                "region_id": "",
                "row_ids": [],
                "confidence": 0.0,
                "source_stage": "deterministic_validator",
            }
        )
    duplicate_assessment_ids = sorted({
        rid for rid in assessment_ids if rid and assessment_ids.count(rid) > 1
    })
    if duplicate_assessment_ids:
        issues.append(
            {
                "issue_type": "bom-detector-duplicate-proposal-assessment",
                "severity": "high",
                "message": (
                    "Detector returned more than one assessment for the same "
                    "geometry proposal"
                ),
                "region_id": "",
                "row_ids": duplicate_assessment_ids,
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    if set(assessment_ids) != set(proposal_ids) or len(assessment_ids) != len(
        proposal_ids
    ):
        issues.append(
            {
                "issue_type": "bom-detector-proposal-accounting-mismatch",
                "severity": "high",
                "message": (
                    "Detector did not return exactly one assessment for every "
                    "deterministic geometry proposal"
                ),
                "region_id": "",
                "row_ids": sorted(
                    set(proposal_ids).symmetric_difference(set(assessment_ids))
                ),
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    if float(detector.get("confidence") or 0.0) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(
            {
                "issue_type": "bom-detector-confidence-below-threshold",
                "severity": "high",
                "message": "BOM detector confidence is below the page threshold",
                "region_id": "",
                "row_ids": [],
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    if not detector.get("all_visible_bom_tables_accounted_for"):
        issues.append(
            {
                "issue_type": "bom-visible-table-coverage-failed",
                "severity": "high",
                "message": "Detector reports unaccounted visible BOM tables",
                "region_id": "",
                "row_ids": [],
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    if detector.get("missing_visible_bom_tables"):
        issues.append(
            {
                "issue_type": "bom-missing-visible-table",
                "severity": "high",
                "message": "Detector listed visible BOM tables without proposals",
                "region_id": "",
                "row_ids": [],
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    for raw in detector.get("issues") or []:
        issues.append(
            _normalize_issue(
                raw,
                default_type="bom-detector-issue",
                source_stage="detector",
            )
        )

    active_region_ids = [
        rid
        for rid in proposal_ids
        if (assessments.get(rid) or {}).get("visible")
        and (assessments.get(rid) or {}).get("distinct_table")
        and (assessments.get(rid) or {}).get("kind") == "bom_table"
    ]
    for rid in active_region_ids:
        assessment = assessments.get(rid) or {}
        if (
            float(assessment.get("confidence") or 0.0)
            < PAGE_PASS_MIN_CONFIDENCE
        ):
            issues.append(
                {
                    "issue_type": "bom-detector-region-confidence-below-threshold",
                    "severity": "high",
                    "message": (
                        f"Detector confidence for BOM region {rid} is below "
                        "the page threshold"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(assessment.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    duplicate_extraction_ids = sorted({
        rid for rid in extraction_ids if rid and extraction_ids.count(rid) > 1
    })
    if duplicate_extraction_ids:
        issues.append(
            {
                "issue_type": "bom-duplicate-extraction-region",
                "severity": "high",
                "message": "A BOM region was extracted more than once",
                "region_id": "",
                "row_ids": duplicate_extraction_ids,
                "confidence": 0.0,
                "source_stage": "deterministic_validator",
            }
        )
    if set(extraction_ids) != set(active_region_ids) or len(extraction_ids) != len(
        active_region_ids
    ):
        issues.append(
            {
                "issue_type": "bom-extractor-region-accounting-mismatch",
                "severity": "high",
                "message": (
                    "Extractor results do not exactly match the active BOM "
                    "geometry regions"
                ),
                "region_id": "",
                "row_ids": sorted(
                    set(active_region_ids).symmetric_difference(
                        set(extraction_ids)
                    )
                ),
                "confidence": 0.0,
                "source_stage": "deterministic_validator",
            }
        )

    if not active_region_ids:
        issues.append(
            {
                "issue_type": "bom-no-active-table",
                "severity": "high",
                "message": "No visible BOM table was approved for extraction",
                "region_id": "",
                "row_ids": [],
                "confidence": float(detector.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    for rid in active_region_ids:
        proposal = proposal_by_id.get(rid)
        assessment = assessments.get(rid) or {}
        extraction = extraction_by_id.get(rid)
        if not proposal or not extraction:
            issues.append(
                {
                    "issue_type": "bom-region-not-extracted",
                    "severity": "high",
                    "message": f"BOM region {rid} is missing extraction",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
            continue

        if int(extraction.get("page_id") or 0) != int(page["id"]):
            issues.append(
                {
                    "issue_type": "bom-extractor-page-id-mismatch",
                    "severity": "high",
                    "message": f"Extractor returned a different page_id for {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
        if _clean_text(extraction.get("region_id"), 120) != rid:
            issues.append(
                {
                    "issue_type": "bom-extractor-region-id-mismatch",
                    "severity": "high",
                    "message": f"Extractor returned a different region_id for {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
        if (
            float(extraction.get("confidence") or 0.0)
            < PAGE_PASS_MIN_CONFIDENCE
        ):
            issues.append(
                {
                    "issue_type": "bom-extractor-confidence-below-threshold",
                    "severity": "high",
                    "message": (
                        f"Extractor confidence for BOM region {rid} is below "
                        "the page threshold"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        expected_item_rows = int(assessment.get("expected_item_rows") or 0)
        expected_header_rows = int(assessment.get("expected_header_rows") or 0)
        expected_column_count = int(
            assessment.get("expected_column_count") or 0
        )
        source_column_roles = [
            x
            for x in (extraction.get("source_column_roles") or [])
            if isinstance(x, dict)
        ]
        source_column_indexes: list[int] = []
        for decision in source_column_roles:
            try:
                source_column_index = int(decision.get("source_column_index"))
            except Exception:
                source_column_index = -1
            source_column_indexes.append(source_column_index)
            canonical_roles = [
                _clean_text(role, 120)
                for role in (decision.get("canonical_roles") or [])
                if _clean_text(role, 120)
            ]
            if (
                source_column_index < 0
                or not canonical_roles
                or any(
                    role not in CANONICAL_COLUMN_ROLES
                    for role in canonical_roles
                )
                or ("other_data" in canonical_roles and len(canonical_roles) > 1)
            ):
                issues.append(
                    {
                        "issue_type": "bom-source-column-role-invalid",
                        "severity": "high",
                        "message": (
                            f"Invalid semantic column binding in region {rid} "
                            f"for source column {source_column_index}"
                        ),
                        "region_id": rid,
                        "row_ids": [],
                        "confidence": float(decision.get("confidence") or 0.0),
                        "source_stage": "deterministic_validator",
                    }
                )
            if float(decision.get("confidence") or 0.0) < ROW_MIN_CONFIDENCE:
                issues.append(
                    {
                        "issue_type": "bom-source-column-confidence-below-threshold",
                        "severity": "high",
                        "message": (
                            f"Column binding confidence is below threshold in "
                            f"{rid}/{source_column_index}"
                        ),
                        "region_id": rid,
                        "row_ids": [],
                        "confidence": float(decision.get("confidence") or 0.0),
                        "source_stage": "deterministic_validator",
                    }
                )
        source_column_roles_by_index = {
            int(decision.get("source_column_index")): {
                _clean_text(role, 120)
                for role in (decision.get("canonical_roles") or [])
                if _clean_text(role, 120)
            }
            for decision in source_column_roles
            if str(decision.get("source_column_index", "")).lstrip("-").isdigit()
        }
        expected_column_indexes = set(range(max(0, expected_column_count)))
        if (
            expected_column_count <= 0
            or set(source_column_indexes) != expected_column_indexes
            or len(source_column_indexes) != expected_column_count
        ):
            issues.append(
                {
                    "issue_type": "bom-source-column-accounting-mismatch",
                    "severity": "high",
                    "message": (
                        f"Region {rid} did not return exactly one semantic "
                        "binding for every visible source column"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        extracted_rows = list(extraction.get("rows") or [])
        if expected_item_rows <= 0 or len(extracted_rows) != expected_item_rows:
            issues.append(
                {
                    "issue_type": "bom-row-count-mismatch",
                    "severity": "high",
                    "message": (
                        f"Region {rid} expected {expected_item_rows} item rows "
                        f"but extracted {len(extracted_rows)}"
                    ),
                    "region_id": rid,
                    "row_ids": [
                        _clean_text(x.get("row_id"), 120)
                        for x in extracted_rows
                    ],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        candidate_rows = proposal.get("row_candidates") or []
        candidate_by_id = {
            _clean_text(x.get("source_row_candidate_id"), 120): x
            for x in candidate_rows
            if isinstance(x, dict)
        }
        if candidate_rows:
            (
                header_ids,
                non_item_ids,
                item_candidate_ids,
            ) = _canonical_row_candidate_accounting(extraction)
            all_candidate_ids = set(candidate_by_id)
            accounted = header_ids | non_item_ids | item_candidate_ids
            overlaps = (
                (header_ids & non_item_ids)
                | (header_ids & item_candidate_ids)
                | (non_item_ids & item_candidate_ids)
            )
            if len(header_ids) != expected_header_rows:
                issues.append(
                    {
                        "issue_type": "bom-header-row-count-mismatch",
                        "severity": "high",
                        "message": (
                            f"Region {rid} expected {expected_header_rows} "
                            f"header rows but classified {len(header_ids)}"
                        ),
                        "region_id": rid,
                        "row_ids": sorted(header_ids),
                        "confidence": float(extraction.get("confidence") or 0.0),
                        "source_stage": "deterministic_validator",
                    }
                )
            if accounted != all_candidate_ids or overlaps:
                issues.append(
                    {
                        "issue_type": "bom-row-candidate-accounting-failed",
                        "severity": "high",
                        "message": (
                            f"Region {rid} did not account for every deterministic "
                            "row candidate exactly once"
                        ),
                        "region_id": rid,
                        "row_ids": sorted(all_candidate_ids - accounted),
                        "confidence": float(extraction.get("confidence") or 0.0),
                        "source_stage": "deterministic_validator",
                    }
                )
        if extraction.get("unaccounted_row_candidate_ids"):
            issues.append(
                {
                    "issue_type": "bom-unaccounted-row-candidates",
                    "severity": "high",
                    "message": f"Region {rid} has unaccounted row candidates",
                    "region_id": rid,
                    "row_ids": [
                        _clean_text(x, 120)
                        for x in extraction.get("unaccounted_row_candidate_ids") or []
                    ],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
        if extraction.get("duplicate_row_ids"):
            issues.append(
                {
                    "issue_type": "bom-duplicate-row-ids-returned",
                    "severity": "high",
                    "message": f"Region {rid} returned duplicate row IDs",
                    "region_id": rid,
                    "row_ids": [
                        _clean_text(x, 120)
                        for x in extraction.get("duplicate_row_ids") or []
                    ],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        seen_row_ids: set[str] = set()
        seen_visual_orders: set[int] = set()
        seen_source_candidate_ids: set[str] = set()
        for item in sorted(
            extracted_rows,
            key=lambda x: int(x.get("visual_order") or 0),
        ):
            row_id = _clean_text(item.get("row_id"), 120)
            visual_order = int(item.get("visual_order") or 0)
            role = _clean_text(item.get("row_role"), 120)
            confidence = float(item.get("confidence") or 0.0)
            source_candidate_id = _clean_text(
                item.get("source_row_candidate_id"), 120
            )

            if not row_id or row_id in seen_row_ids:
                issues.append(
                    {
                        "issue_type": "bom-row-id-invalid",
                        "severity": "high",
                        "message": f"Missing or duplicate row_id in region {rid}",
                        "region_id": rid,
                        "row_ids": [row_id] if row_id else [],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            seen_row_ids.add(row_id)
            if (
                not source_candidate_id
                or source_candidate_id in seen_source_candidate_ids
            ):
                issues.append(
                    {
                        "issue_type": "bom-source-row-candidate-duplicate",
                        "severity": "high",
                        "message": (
                            f"Missing or duplicate physical source row identity "
                            f"in {rid}/{row_id}"
                        ),
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            seen_source_candidate_ids.add(source_candidate_id)
            if visual_order < 1 or visual_order in seen_visual_orders:
                issues.append(
                    {
                        "issue_type": "bom-visual-order-invalid",
                        "severity": "high",
                        "message": f"Invalid or duplicate visual_order in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            seen_visual_orders.add(visual_order)
            if role not in ROW_ROLES:
                issues.append(
                    {
                        "issue_type": "bom-row-role-invalid",
                        "severity": "high",
                        "message": f"Invalid BOM row role in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            if confidence < ROW_MIN_CONFIDENCE:
                issues.append(
                    {
                        "issue_type": "bom-row-confidence-below-threshold",
                        "severity": "high",
                        "message": f"Low confidence BOM row {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            bbox = list(item.get("bbox_pt") or [])
            bbox_valid = False
            if len(bbox) == 4:
                try:
                    x0, y0, x1, y1 = [float(value) for value in bbox]
                    page_width = float(page.get("page_width_pt") or 0.0)
                    page_height = float(page.get("page_height_pt") or 0.0)
                    bbox_valid = bool(x0 < x1 and y0 < y1)
                    if page_width > 0.0 and page_height > 0.0:
                        bbox_valid = bool(
                            bbox_valid
                            and x0 >= -2.0
                            and y0 >= -2.0
                            and x1 <= page_width + 2.0
                            and y1 <= page_height + 2.0
                        )
                except Exception:
                    bbox_valid = False
            if not bbox_valid:
                issues.append(
                    {
                        "issue_type": "bom-row-bbox-invalid",
                        "severity": "high",
                        "message": f"Invalid row bbox in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            if candidate_rows:
                candidate = candidate_by_id.get(source_candidate_id)
                if not candidate:
                    issues.append(
                        {
                            "issue_type": "bom-source-row-candidate-invalid",
                            "severity": "high",
                            "message": (
                                f"Row {rid}/{row_id} references an unknown "
                                "source row candidate"
                            ),
                            "region_id": rid,
                            "row_ids": [row_id],
                            "confidence": confidence,
                            "source_stage": "deterministic_validator",
                        }
                    )
                    expected_word_ids: list[int] = []
                else:
                    expected_word_ids = [
                        int(x)
                        for x in (candidate.get("word_ids") or [])
                        if int(x) in word_map
                    ]
            else:
                try:
                    row_rect = _rect_from(bbox)
                    expected_word_ids = _ids_in_rect(word_map, row_rect)
                except Exception:
                    expected_word_ids = []

            declared_source_ids = {
                int(x)
                for x in (item.get("source_word_ids") or [])
                if (isinstance(x, int) or str(x).isdigit())
            }
            if set(expected_word_ids) != declared_source_ids:
                issues.append(
                    {
                        "issue_type": "bom-row-source-word-accounting-mismatch",
                        "severity": "high",
                        "message": (
                            f"Row {rid}/{row_id} source_word_ids do not exactly "
                            "match its physical source row"
                        ),
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            invalid_field_column_bindings: list[dict] = []
            for evidence in item.get("field_evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                field_name = _clean_text(evidence.get("field_name"), 120)
                try:
                    source_column_index = int(
                        evidence.get("source_column_index")
                    )
                except Exception:
                    source_column_index = -1
                expected_role = ORIGINAL_FIELD_TO_COLUMN_ROLE.get(field_name)
                bound_roles = source_column_roles_by_index.get(
                    source_column_index,
                    set(),
                )
                if (
                    not expected_role
                    or source_column_index not in source_column_roles_by_index
                    or expected_role not in bound_roles
                ):
                    invalid_field_column_bindings.append(
                        {
                            "field_name": field_name,
                            "source_column_index": source_column_index,
                            "expected_role": expected_role,
                            "bound_roles": sorted(bound_roles),
                        }
                    )
            if invalid_field_column_bindings:
                issues.append(
                    {
                        "issue_type": "bom-field-column-binding-mismatch",
                        "severity": "high",
                        "message": (
                            f"One or more fields in {rid}/{row_id} are not "
                            "supported by their physical source columns"
                        ),
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            evidence_audit = _field_evidence_audit(
                row=item,
                expected_word_ids=expected_word_ids,
                word_map=word_map,
            )
            evidence_audit["invalid_field_column_bindings"] = (
                invalid_field_column_bindings
            )
            evidence_audit["complete"] = bool(
                evidence_audit.get("complete")
                and not invalid_field_column_bindings
            )
            item["source_evidence_coverage"] = {
                "version": "bom-row-source-evidence-v1.3",
                **evidence_audit,
            }
            if not evidence_audit["complete"]:
                issues.append(
                    {
                        "issue_type": "bom-visible-source-evidence-unrepresented",
                        "severity": "high",
                        "message": (
                            f"Visible source evidence is incomplete or duplicated "
                            f"for {rid}/{row_id}"
                        ),
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            for base_name in BASE_FIELDS:
                original = _clean_text(
                    item.get(f"{base_name}_original"), 4000
                )
                normalized = _clean_text(
                    item.get(f"{base_name}_normalized"), 4000
                )
                if not original and normalized:
                    issues.append(
                        {
                            "issue_type": "bom-normalized-text-without-source",
                            "severity": "high",
                            "message": (
                                f"Normalized {base_name} has no source value in "
                                f"{rid}/{row_id}"
                            ),
                            "region_id": rid,
                            "row_ids": [row_id],
                            "confidence": confidence,
                            "source_stage": "deterministic_validator",
                        }
                    )
                if original and not normalized:
                    issues.append(
                        {
                            "issue_type": "bom-normalized-text-missing",
                            "severity": "high",
                            "message": (
                                f"Normalized {base_name} is missing in {rid}/{row_id}"
                            ),
                            "region_id": rid,
                            "row_ids": [row_id],
                            "confidence": confidence,
                            "source_stage": "deterministic_validator",
                        }
                    )
                if (
                    original
                    and normalized
                    and _semantic_character_signature(original)
                    != _semantic_character_signature(normalized)
                ):
                    issues.append(
                        {
                            "issue_type": "bom-normalized-text-changed-source-content",
                            "severity": "high",
                            "message": (
                                f"Normalized {base_name} changes source "
                                f"alphanumeric content in {rid}/{row_id}"
                            ),
                            "region_id": rid,
                            "row_ids": [row_id],
                            "confidence": confidence,
                            "source_stage": "deterministic_validator",
                        }
                    )

            substantive = any(
                _clean_text(item.get(field_name), 4000)
                for field_name in (
                    "component_tag_original",
                    "description_original",
                    "part_number_original",
                    "manufacturer_original",
                )
            )
            if not substantive:
                issues.append(
                    {
                        "issue_type": "bom-item-row-empty",
                        "severity": "high",
                        "message": f"BOM item row {rid}/{row_id} has no item data",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            item["region_id"] = rid
            item["source_column_roles"] = extraction.get(
                "source_column_roles"
            ) or []
            rows.append(item)

        for raw in extraction.get("issues") or []:
            issues.append(
                _normalize_issue(
                    raw,
                    default_type="bom-extractor-issue",
                    default_region_id=rid,
                    source_stage="extractor",
                )
            )

    if int(verifier.get("page_id") or 0) != int(page["id"]):
        issues.append(
            {
                "issue_type": "bom-verifier-page-id-mismatch",
                "severity": "high",
                "message": "Verifier returned a different page_id",
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    for raw in verifier.get("issues") or []:
        normalized_verifier_issue = (
            _normalize_verifier_issue_after_overrides(
                raw,
                extractions=extractions,
            )
        )
        normalized_verifier_issue = (
            _post_override_text_issue_resolution(
                raw_issue=raw if isinstance(raw, dict) else {},
                normalized_issue=normalized_verifier_issue,
                extractions=extractions,
            )
        )
        issues.append(normalized_verifier_issue)

    raw_verifier_checks = [
        x
        for x in (verifier.get("region_checks") or [])
        if isinstance(x, dict)
    ]
    verifier_check_ids = [
        _clean_text(x.get("region_id"), 120)
        for x in raw_verifier_checks
    ]
    verifier_checks = {
        _clean_text(x.get("region_id"), 120): x
        for x in raw_verifier_checks
        if _clean_text(x.get("region_id"), 120)
    }
    if (
        set(verifier_check_ids) != set(active_region_ids)
        or len(verifier_check_ids) != len(active_region_ids)
    ):
        issues.append(
            {
                "issue_type": "bom-verifier-region-accounting-mismatch",
                "severity": "high",
                "message": (
                    "Verifier did not return exactly one check for every active "
                    "BOM region"
                ),
                "region_id": "",
                "row_ids": sorted(
                    set(active_region_ids).symmetric_difference(
                        set(verifier_check_ids)
                    )
                ),
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    for rid in active_region_ids:
        check = verifier_checks.get(rid)
        actual_rows = sorted(
            [row for row in rows if row.get("region_id") == rid],
            key=lambda x: int(x.get("visual_order") or 0),
        )
        if not check:
            issues.append(
                {
                    "issue_type": "bom-verifier-region-missing",
                    "severity": "high",
                    "message": f"Verifier did not return region {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
            continue
        check_confidence = float(check.get("confidence") or 0.0)
        if check_confidence < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(
                {
                    "issue_type": "bom-verifier-region-confidence-below-threshold",
                    "severity": "high",
                    "message": _clean_text(
                        check.get("notes") or f"Verifier confidence failed {rid}",
                        1600,
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": check_confidence,
                    "source_stage": "deterministic_validator",
                }
            )
        actual_row_ids = [
            _clean_text(row.get("row_id"), 120) for row in actual_rows
        ]
        verified_row_ids = [
            _clean_text(x, 120)
            for x in (check.get("verified_row_ids") or [])
        ]
        actual_tags = [
            _clean_text(
                row.get("component_tag_normalized")
                or row.get("component_tag_original"),
                1000,
            )
            for row in actual_rows
        ]
        verified_tags = [
            _clean_text(x, 1000)
            for x in (check.get("verified_component_tag_sequence") or [])
        ]
        count_and_identity_match = bool(
            int(check.get("expected_item_rows") or 0) == len(actual_rows)
            and int(check.get("verified_item_rows") or 0) == len(actual_rows)
            and verified_row_ids == actual_row_ids
        )
        exact_sequence_matches = bool(
            count_and_identity_match
            and _sequence_matches_source_characters(
                actual_tags,
                verified_tags,
            )
        )
        component_tag_sequence_adjudication: dict = {}
        if count_and_identity_match and not exact_sequence_matches:
            component_tag_sequence_adjudication = (
                _component_tag_sequence_source_exact_adjudication(
                    actual_rows=actual_rows,
                    verified_row_ids=verified_row_ids,
                    verified_tags=verified_tags,
                    word_map=word_map,
                )
            )
        sequence_matches = bool(
            exact_sequence_matches
            or component_tag_sequence_adjudication.get("validated") is True
        )
        if (
            component_tag_sequence_adjudication.get("validated") is True
            and component_tag_sequence_adjudication.get(
                "source_authoritative_difference_count"
            )
        ):
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-component-tag-sequence-"
                        "source-exact-adjudicated"
                    ),
                    "severity": "info",
                    "message": (
                        "Verifier component-tag sequence omitted source "
                        "punctuation or spacing; exact vector field evidence "
                        "preserved the printed identifiers."
                    ),
                    "region_id": rid,
                    "row_ids": component_tag_sequence_adjudication[
                        "source_authoritative_row_ids"
                    ],
                    "confidence": check_confidence,
                    "source_stage": (
                        "deterministic_source_evidence_adjudicator"
                    ),
                    "component_tag_sequence_adjudication": (
                        component_tag_sequence_adjudication
                    ),
                }
            )
        if not sequence_matches:
            issues.append(
                {
                    "issue_type": "bom-verifier-row-sequence-mismatch",
                    "severity": "high",
                    "message": (
                        f"Verifier row count/order/tag sequence differs from "
                        f"materialized region {rid}"
                    ),
                    "region_id": rid,
                    "row_ids": actual_row_ids,
                    "confidence": check_confidence,
                    "source_stage": "deterministic_validator",
                    "component_tag_sequence_adjudication": (
                        component_tag_sequence_adjudication
                    ),
                }
            )
        elif not check.get("pass"):
            prior_blocking = [
                issue
                for issue in issues
                if issue.get("severity") in {"high", "critical"}
            ]
            if not prior_blocking and check_confidence >= PAGE_PASS_MIN_CONFIDENCE:
                issues.append(
                    {
                        "issue_type": (
                            "bom-verifier-region-verdict-superseded-post-override"
                        ),
                        "severity": "info",
                        "message": (
                            "The verifier region decision described the "
                            "pre-correction candidate; exact post-override "
                            "row and field validation is now clean."
                        ),
                        "region_id": rid,
                        "row_ids": actual_row_ids,
                        "confidence": check_confidence,
                        "source_stage": (
                            "deterministic_post_override_adjudicator"
                        ),
                        "post_override_resolution": {
                            "status": "superseded_after_exact_revalidation",
                            "validated": True,
                            "verifier_notes": _clean_text(
                                check.get("notes"), 1600
                            ),
                        },
                    }
                )
            else:
                issues.append(
                    {
                        "issue_type": "bom-verifier-region-failed",
                        "severity": "high",
                        "message": _clean_text(
                            check.get("notes") or f"Verifier failed {rid}",
                            1600,
                        ),
                        "region_id": rid,
                        "row_ids": [],
                        "confidence": check_confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

    structural_required_flags = [
        "all_visible_bom_tables_accounted_for",
        "all_visible_item_rows_accounted_for",
        "all_visible_columns_accounted_for",
    ]
    for flag in structural_required_flags:
        if not verifier.get(flag):
            issues.append(
                {
                    "issue_type": f"bom-verifier-{flag}",
                    "severity": "high",
                    "message": f"Verifier returned {flag}=false",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    if not verifier.get("all_source_evidence_represented"):
        all_final_source_evidence_complete = bool(rows) and all(
            (row.get("source_evidence_coverage") or {}).get("complete")
            is True
            for row in rows
        )
        prior_blocking = [
            issue
            for issue in issues
            if issue.get("severity") in {"high", "critical"}
        ]
        if all_final_source_evidence_complete and not prior_blocking:
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-source-evidence-flag-"
                        "superseded-post-override"
                    ),
                    "severity": "info",
                    "message": (
                        "The verifier source-evidence flag described the "
                        "pre-correction candidate; every final source word is "
                        "represented exactly once after deterministic "
                        "post-override reconciliation."
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": (
                        "deterministic_post_override_adjudicator"
                    ),
                    "post_override_resolution": {
                        "status": "superseded_after_exact_revalidation",
                        "validated": True,
                        "final_row_count": len(rows),
                    },
                }
            )
        else:
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-all_source_evidence_represented"
                    ),
                    "severity": "high",
                    "message": (
                        "Verifier returned "
                        "all_source_evidence_represented=false"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    if not verifier.get("all_published_fields_visually_supported"):
        prior_blocking = [
            issue
            for issue in issues
            if issue.get("severity") in {"high", "critical"}
        ]
        if not prior_blocking:
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-fields-flag-superseded-post-override"
                    ),
                    "severity": "info",
                    "message": (
                        "The verifier field-support flag described the "
                        "pre-correction candidate; every final field passed "
                        "exact post-override evidence validation."
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": (
                        "deterministic_post_override_adjudicator"
                    ),
                    "post_override_resolution": {
                        "status": "superseded_after_exact_revalidation",
                        "validated": True,
                    },
                }
            )
        else:
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-all_published_fields_visually_supported"
                    ),
                    "severity": "high",
                    "message": (
                        "Verifier returned "
                        "all_published_fields_visually_supported=false"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    unresolved_duplicate_keys, resolved_repeated_values = (
        _adjudicate_verifier_duplicate_physical_keys(
            reported_keys=verifier.get("duplicate_physical_keys") or [],
            rows=rows,
        )
    )
    if resolved_repeated_values:
        issues.append(
            {
                "issue_type": "bom-repeated-values-preserved-as-distinct-rows",
                "severity": "info",
                "message": (
                    "Verifier-reported duplicate values were confirmed as "
                    "separate physical BOM rows by deterministic row identity"
                ),
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_duplicate_adjudicator",
                "duplicate_value_adjudication": resolved_repeated_values,
            }
        )
    if unresolved_duplicate_keys:
        issues.append(
            {
                "issue_type": "bom-verifier-duplicate-physical-keys",
                "severity": "high",
                "message": (
                    "Verifier returned unresolved duplicate physical row keys"
                ),
                "region_id": "",
                "row_ids": unresolved_duplicate_keys,
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    if not verifier.get("duplicates_preserved"):
        if resolved_repeated_values and not unresolved_duplicate_keys:
            issues.append(
                {
                    "issue_type": "bom-verifier-duplicates-preserved-adjudicated",
                    "severity": "info",
                    "message": (
                        "The verifier flag was conservative, but deterministic "
                        "physical row identities prove repeated values were preserved"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_duplicate_adjudicator",
                    "duplicate_value_adjudication": resolved_repeated_values,
                }
            )
        else:
            issues.append(
                {
                    "issue_type": "bom-verifier-duplicates_preserved",
                    "severity": "high",
                    "message": "Verifier returned duplicates_preserved=false",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    for field_name, issue_type in [
        ("missing_region_ids", "bom-verifier-missing-regions"),
        ("missing_row_ids", "bom-verifier-missing-rows"),
        ("unaccounted_visual_evidence", "bom-verifier-unaccounted-evidence"),
    ]:
        if verifier.get(field_name):
            issues.append(
                {
                    "issue_type": issue_type,
                    "severity": "high",
                    "message": f"Verifier returned non-empty {field_name}",
                    "region_id": "",
                    "row_ids": [
                        _clean_text(x, 120)
                        for x in (verifier.get(field_name) or [])
                    ],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
    if str(verifier.get("verdict") or "") != "pass":
        prior_blocking = [
            issue
            for issue in issues
            if issue.get("severity") in {"high", "critical"}
        ]
        if (
            not prior_blocking
            and float(verifier.get("confidence") or 0.0)
            >= PAGE_PASS_MIN_CONFIDENCE
        ):
            issues.append(
                {
                    "issue_type": (
                        "bom-verifier-verdict-superseded-post-override"
                    ),
                    "severity": "info",
                    "message": (
                        "Independent verifier review_required was based on the "
                        "pre-correction candidate; exact post-override "
                        "revalidation completed without blocking findings."
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": (
                        "deterministic_post_override_adjudicator"
                    ),
                    "post_override_resolution": {
                        "status": "superseded_after_exact_revalidation",
                        "validated": True,
                    },
                }
            )
        else:
            issues.append(
                {
                    "issue_type": "bom-verifier-blocked-page",
                    "severity": "high",
                    "message": "Independent verifier did not pass BOM page",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
    if float(verifier.get("confidence") or 0.0) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(
            {
                "issue_type": "bom-page-confidence-below-threshold",
                "severity": "high",
                "message": "BOM page confidence below threshold",
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    blocking = [
        issue
        for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    return not blocking and bool(rows), rows, issues


def _db_replace_page_issues(
    *,
    context: dict,
    page_id: int,
    issues: list[dict],
) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.electrical_review_issues
                WHERE version_id=%s
                  AND page_id=%s
                  AND properties ->> 'phase'=%s;
                """,
                (int(context["version_id"]), int(page_id), PHASE_NAME),
            )
            for index, issue in enumerate(issues, start=1):
                issue_key = hashlib.sha256(
                    "|".join(
                        [
                            str(context["version_id"]),
                            str(page_id),
                            PHASE_NAME,
                            str(index),
                            str(issue.get("issue_type") or ""),
                            str(issue.get("region_id") or ""),
                            str(issue.get("message") or ""),
                        ]
                    ).encode("utf-8")
                ).hexdigest()
                props = {
                    "phase": PHASE_NAME,
                    "pipeline_marker": PIPELINE_MARKER,
                    "materializer_version": MATERIALIZER_VERSION,
                    "pdf_page_number": int(context["page"]["pdf_page_number"]),
                    "region_id": issue.get("region_id") or "",
                    "row_ids": issue.get("row_ids") or [],
                    "confidence": float(issue.get("confidence") or 0.0),
                    "source_stage": issue.get("source_stage") or "",
                    "post_override_resolution": issue.get(
                        "post_override_resolution"
                    )
                    or {},
                    "duplicate_value_adjudication": issue.get(
                        "duplicate_value_adjudication"
                    )
                    or [],
                    "component_tag_sequence_adjudication": issue.get(
                        "component_tag_sequence_adjudication"
                    )
                    or {},
                }
                cur.execute(
                    """
                    INSERT INTO public.electrical_review_issues(
                        version_id, company_id, machine_id,
                        bubble_document_id, page_id, issue_key,
                        issue_type, severity, status, message,
                        candidates_json, properties, created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,'open',%s,
                        '[]'::jsonb,%s::jsonb,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, issue_key)
                    DO UPDATE SET
                        issue_type=EXCLUDED.issue_type,
                        severity=EXCLUDED.severity,
                        status='open',
                        message=EXCLUDED.message,
                        properties=EXCLUDED.properties,
                        updated_at=NOW();
                    """,
                    (
                        int(context["version_id"]),
                        context["company_id"],
                        context["machine_id"],
                        context["bubble_document_id"],
                        int(page_id),
                        issue_key,
                        _clean_text(issue.get("issue_type"), 180),
                        issue.get("severity")
                        if issue.get("severity") in SEVERITIES
                        else "warning",
                        _clean_text(issue.get("message"), 1600),
                        json.dumps(props, ensure_ascii=False),
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_publish_page_rows(
    *,
    context: dict,
    page: dict,
    rows: list[dict],
    detector_fingerprint: str,
    extractor_fingerprints: dict[str, str],
    verifier_fingerprint: str,
) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.electrical_bom
                WHERE version_id=%s AND page_id=%s;
                """,
                (int(context["version_id"]), int(page["id"])),
            )
            for item in rows:
                region_id = _clean_text(item.get("region_id"), 120)
                visual_order = int(item.get("visual_order") or 0)
                bbox = list(item.get("bbox_pt") or [0, 0, 0, 0])
                if len(bbox) != 4:
                    bbox = [0, 0, 0, 0]
                bom_key = _physical_bom_key(
                    version_id=int(context["version_id"]),
                    page_id=int(page["id"]),
                    region_id=region_id,
                    visual_order=visual_order,
                )
                row_id = _stable_bigint_id(
                    "electrical_bom",
                    context["version_id"],
                    bom_key,
                )
                quantity_text = _published_value(item, "quantity_text")
                quantity = _parse_quantity(quantity_text)
                source_text = _clean_text(
                    " | ".join(
                        _clean_text(item.get(field_name), 4000)
                        for field_name in ORIGINAL_FIELDS
                        if _clean_text(item.get(field_name), 4000)
                    ),
                    12000,
                )
                properties = {
                    "phase": PHASE_NAME,
                    "pipeline_marker": PIPELINE_MARKER,
                    "materializer_version": MATERIALIZER_VERSION,
                    "pdf_page_number": int(page["pdf_page_number"]),
                    "sheet_code": page.get("sheet_code"),
                    "region_id": region_id,
                    "row_id": item.get("row_id"),
                    "source_row_candidate_id": item.get(
                        "source_row_candidate_id"
                    ),
                    "visual_order": visual_order,
                    "row_role": item.get("row_role"),
                    "original_fields": {
                        field_name: item.get(field_name) or ""
                        for field_name in ORIGINAL_FIELDS
                    },
                    "normalized_fields": {
                        field_name: item.get(field_name) or ""
                        for field_name in NORMALIZED_FIELDS
                    },
                    "source_column_roles": item.get("source_column_roles") or [],
                    "field_evidence": item.get("field_evidence") or [],
                    "source_word_ids": item.get("source_word_ids") or [],
                    "source_evidence_coverage": item.get(
                        "source_evidence_coverage"
                    )
                    or {},
                    "verifier_overrides": item.get("verifier_overrides") or {},
                    "deterministic_overrides": item.get(
                        "deterministic_overrides"
                    ) or {},
                    "post_override_normalization": item.get(
                        "post_override_normalization"
                    ) or {},
                    "cross_field_evidence_transfers": item.get(
                        "cross_field_evidence_transfers"
                    ) or [],
                    "component_tag_sequence_adjudication": item.get(
                        "component_tag_sequence_adjudication"
                    ) or {},
                    "evidence_notes": item.get("evidence_notes") or "",
                    "detector_fingerprint": detector_fingerprint,
                    "extractor_fingerprint": extractor_fingerprints.get(
                        region_id,
                        "",
                    ),
                    "verifier_fingerprint": verifier_fingerprint,
                    "page_passed": True,
                }
                cur.execute(
                    """
                    INSERT INTO public.electrical_bom(
                        id, version_id, company_id, machine_id,
                        bubble_document_id, page_id, source_entity_id,
                        bom_key, item_position, component_tag,
                        quantity, quantity_text, unit, manufacturer,
                        part_number, description, x0, y0, x1, y1,
                        source_text, properties, confidence,
                        extraction_method, is_verified, created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, bom_key)
                    DO UPDATE SET
                        item_position=EXCLUDED.item_position,
                        component_tag=EXCLUDED.component_tag,
                        quantity=EXCLUDED.quantity,
                        quantity_text=EXCLUDED.quantity_text,
                        unit=EXCLUDED.unit,
                        manufacturer=EXCLUDED.manufacturer,
                        part_number=EXCLUDED.part_number,
                        description=EXCLUDED.description,
                        x0=EXCLUDED.x0,
                        y0=EXCLUDED.y0,
                        x1=EXCLUDED.x1,
                        y1=EXCLUDED.y1,
                        source_text=EXCLUDED.source_text,
                        properties=EXCLUDED.properties,
                        confidence=EXCLUDED.confidence,
                        extraction_method=EXCLUDED.extraction_method,
                        updated_at=NOW();
                    """,
                    (
                        row_id,
                        int(context["version_id"]),
                        context["company_id"],
                        context["machine_id"],
                        context["bubble_document_id"],
                        int(page["id"]),
                        bom_key,
                        _published_value(item, "item_position") or None,
                        _published_value(item, "component_tag") or None,
                        quantity,
                        quantity_text or None,
                        _published_value(item, "unit") or None,
                        _published_value(item, "manufacturer") or None,
                        _published_value(item, "part_number") or None,
                        _published_value(item, "description") or None,
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        source_text,
                        json.dumps(properties, ensure_ascii=False),
                        max(
                            0.0,
                            min(1.0, float(item.get("confidence") or 0.0)),
                        ),
                        EXTRACTION_METHOD,
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_update_version_state(
    *,
    context: dict,
    page: dict,
    page_passed: bool,
    published_rows: int,
    blocking_count: int,
) -> dict:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata
                FROM public.electrical_versions
                WHERE id=%s
                FOR UPDATE;
                """,
                (int(context["version_id"]),),
            )
            row = cur.fetchone()
            metadata = _json_obj(row[0] if row else {}, {}) or {}
            page_results = metadata.get("bom_page_results") or {}
            if not isinstance(page_results, dict):
                page_results = {}
            page_results[str(page["pdf_page_number"])] = {
                "page_passed": bool(page_passed),
                "published_rows": int(published_rows),
                "blocking_issue_count": int(blocking_count),
                "pipeline_marker": PIPELINE_MARKER,
                "materializer_version": MATERIALIZER_VERSION,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            passed_pages = sum(
                1
                for value in page_results.values()
                if isinstance(value, dict) and value.get("page_passed")
            )
            total_pages = int(context["all_bom_pages_total"])
            bom_status = (
                "bom_ready"
                if passed_pages == total_pages and total_pages > 0
                else ("partial" if passed_pages > 0 else "review_required")
            )
            if not page_passed:
                bom_status = "review_required"

            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_bom
                WHERE version_id=%s
                  AND extraction_method=%s;
                """,
                (int(context["version_id"]), EXTRACTION_METHOD),
            )
            bom_count = int(cur.fetchone()[0] or 0)

            metadata["bom_page_results"] = page_results
            metadata["bom_structured_status"] = bom_status
            metadata["bom_pipeline_marker"] = PIPELINE_MARKER
            metadata["bom_materializer_version"] = MATERIALIZER_VERSION
            metadata["bom_passed_pages"] = passed_pages
            metadata["bom_total_pages"] = total_pages
            metadata["bom_rows"] = bom_count

            version_status = "queued" if page_passed else "review_required"
            cur.execute(
                """
                UPDATE public.electrical_versions
                SET metadata=%s::jsonb,
                    status=%s,
                    error_code=%s,
                    error_message=%s,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (
                    json.dumps(metadata, ensure_ascii=False),
                    version_status,
                    None if page_passed else "BOM_REVIEW_REQUIRED",
                    None
                    if page_passed
                    else "BOM page requires review before publication",
                    int(context["version_id"]),
                ),
            )
        conn.commit()
        return {
            "status": version_status,
            "bom_status": bom_status,
            "bom_count": bom_count,
            "passed_pages": passed_pages,
            "total_pages": total_pages,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_ai_totals(version_id: int) -> dict:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(input_tokens),0),
                       COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(reasoning_tokens),0),
                       COALESCE(SUM(cost_usd),0)
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s;
                """,
                (int(version_id),),
            )
            row = cur.fetchone() or (0, 0, 0, 0)
            return {
                "ai_input_tokens_total": int(row[0] or 0),
                "ai_output_tokens_total": int(row[1] or 0),
                "ai_reasoning_tokens_total": int(row[2] or 0),
                "ai_cost_usd_total": float(row[3] or 0.0),
            }
    finally:
        conn.close()


def _severity_counts(issues: list[dict]) -> dict:
    return {
        severity: sum(
            1 for issue in issues if issue.get("severity") == severity
        )
        for severity in sorted(SEVERITIES)
    }


def extract_electrical_bom_page(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int] = None,
    pdf_page_numbers: Optional[list[int]] = None,
    force: bool = False,
) -> dict:
    if not BOM_ENABLED:
        raise ValueError("Electrical BOM extraction is disabled")

    context = _load_context(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        version_id=version_id,
        pdf_page_numbers=pdf_page_numbers,
    )
    page = context["page"]
    context["page"] = page
    word_map = _word_map(page)
    _, source_doc = _fetch_source_pdf(context)
    usage_totals = _usage_accumulator()

    try:
        page_index = int(page["pdf_page_number"]) - 1
        source_page = source_doc[page_index]
        proposals = _detect_geometry_proposals(
            source_page=source_page,
            inventory_page=page,
            word_map=word_map,
        )

        page_original = _render_page(
            source_doc,
            page_index,
            proposals,
            0,
        )
        page_rotated = _render_page(
            source_doc,
            page_index,
            proposals,
            90,
        )

        detector_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "sheet_code": page.get("sheet_code"),
            "sheet_title": page.get("sheet_title"),
            "proposals": [
                {
                    "region_id": p["region_id"],
                    "geometry_method": p["geometry_method"],
                    "table_bbox_pt": p["table_bbox_pt"],
                    "crop_bbox_pt": p["crop_bbox_pt"],
                    "deterministic_row_count": p[
                        "deterministic_row_count"
                    ],
                    "deterministic_column_count": p[
                        "deterministic_column_count"
                    ],
                    "geometry_recovery": p.get("geometry_recovery") or {},
                }
                for p in proposals
            ],
            "render_dpi": RENDER_DPI,
        }
        detector, detector_usage, detector_reused, detector_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_bom_region_detector_v1",
            region_hash=_sha256_json(detector_request),
            model=DETECTOR_MODEL,
            prompt_version=DETECTOR_PROMPT_VERSION,
            request_payload=detector_request,
            messages=_detector_messages(
                page,
                proposals,
                page_original,
                page_rotated,
            ),
            json_schema=_detector_schema(),
            force=force,
            request_metadata={"proposal_count": len(proposals)},
        )
        _add_usage(
            usage_totals,
            "detector",
            detector_usage,
            detector_reused,
        )

        assessment_by_region = {
            _clean_text(x.get("region_id"), 120): x
            for x in (detector.get("proposal_assessments") or [])
            if isinstance(x, dict)
            and x.get("visible")
            and x.get("distinct_table")
            and x.get("kind") == "bom_table"
        }
        extractions: list[dict] = []
        extractor_fingerprints: dict[str, str] = {}
        region_images: dict[str, tuple[bytes, bytes]] = {}
        for proposal in proposals:
            rid = proposal["region_id"]
            assessment = assessment_by_region.get(rid)
            if not assessment:
                continue
            crop_original = _render_region(
                source_doc,
                page_index,
                proposal,
                0,
            )
            crop_rotated = _render_region(
                source_doc,
                page_index,
                proposal,
                int(
                    detector.get("preferred_reading_rotation_degrees")
                    or 90
                ),
            )
            region_images[rid] = (crop_original, crop_rotated)
            extractor_request = {
                "page_sha256": page.get("page_sha256"),
                "pdf_page_number": page["pdf_page_number"],
                "region_id": rid,
                "region_hash": proposal["region_hash"],
                "detector_assessment": assessment,
                "geometry_method": proposal["geometry_method"],
                "geometry_recovery": proposal.get("geometry_recovery") or {},
                "row_candidates": proposal.get("row_candidates") or [],
                "fallback_page_word_ids": proposal.get(
                    "fallback_page_word_ids"
                )
                or [],
                "fallback_page_words": proposal.get(
                    "fallback_page_words"
                )
                or [],
                "render_dpi": RENDER_DPI,
            }
            result, usage, reused, fp = _cached_call(
                context=context,
                page=page,
                task_type="vision_bom_table_extractor_v1",
                region_hash=proposal["region_hash"],
                model=EXTRACTOR_MODEL,
                prompt_version=EXTRACTOR_PROMPT_VERSION,
                request_payload=extractor_request,
                messages=_extractor_messages(
                    page,
                    proposal,
                    assessment,
                    crop_original,
                    crop_rotated,
                ),
                json_schema=_extractor_schema(),
                force=force,
                request_metadata={"region_id": rid},
            )
            extractions.append(result)
            extractor_fingerprints[rid] = fp
            _add_usage(
                usage_totals,
                "table_extractor",
                usage,
                reused,
            )

        verifier_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "detector_fingerprint": detector_fp,
            "extractor_fingerprints": extractor_fingerprints,
            "detector": detector,
            "geometry_proposals": [
                {
                    "region_id": proposal["region_id"],
                    "geometry_method": proposal.get("geometry_method"),
                    "deterministic_row_count": int(
                        proposal.get("deterministic_row_count") or 0
                    ),
                    "deterministic_column_count": int(
                        proposal.get("deterministic_column_count") or 0
                    ),
                    "geometry_recovery": proposal.get("geometry_recovery") or {},
                }
                for proposal in proposals
            ],
            "extractions": extractions,
            "render_dpi": RENDER_DPI,
        }
        verifier, verifier_usage, verifier_reused, verifier_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_bom_page_verifier_v1",
            region_hash=_sha256_json(verifier_request),
            model=VERIFIER_MODEL,
            prompt_version=VERIFIER_PROMPT_VERSION,
            request_payload=verifier_request,
            messages=_verifier_messages(
                page,
                proposals,
                detector,
                extractions,
                page_original,
                page_rotated,
                region_images,
            ),
            json_schema=_verifier_schema(),
            force=force,
            request_metadata={
                "region_count": len(extractions),
                "detector_fingerprint": detector_fp,
                "extractor_fingerprints": extractor_fingerprints,
            },
        )
        _add_usage(
            usage_totals,
            "verifier",
            verifier_usage,
            verifier_reused,
        )

        _apply_overrides(
            extractions,
            verifier.get("field_overrides") or [],
        )
        page_passed, rows, issues = _validate_page(
            page=page,
            proposals=proposals,
            detector=detector,
            extractions=extractions,
            verifier=verifier,
            word_map=word_map,
        )

        _db_replace_page_issues(
            context=context,
            page_id=int(page["id"]),
            issues=issues,
        )
        if page_passed:
            _db_publish_page_rows(
                context=context,
                page=page,
                rows=rows,
                detector_fingerprint=detector_fp,
                extractor_fingerprints=extractor_fingerprints,
                verifier_fingerprint=verifier_fp,
            )

        blocking = sum(
            1
            for issue in issues
            if issue.get("severity") in {"high", "critical"}
        )
        warning = sum(
            1 for issue in issues if issue.get("severity") == "warning"
        )
        state = _db_update_version_state(
            context=context,
            page=page,
            page_passed=page_passed,
            published_rows=len(rows) if page_passed else 0,
            blocking_count=blocking,
        )

        assessment_map = {
            _clean_text(x.get("region_id"), 120): x
            for x in (detector.get("proposal_assessments") or [])
            if isinstance(x, dict)
            and x.get("visible")
            and x.get("distinct_table")
            and x.get("kind") == "bom_table"
        }
        extraction_map = {
            _clean_text(x.get("region_id"), 120): x
            for x in extractions
            if isinstance(x, dict)
        }
        region_stats: list[dict] = []
        for rid in sorted(assessment_map):
            assessment = assessment_map[rid]
            extraction = extraction_map.get(rid) or {}
            proposal = next(
                (p for p in proposals if p["region_id"] == rid),
                {},
            )
            ordered_rows = sorted(
                extraction.get("rows") or [],
                key=lambda x: int(x.get("visual_order") or 0),
            )
            region_stats.append(
                {
                    "region_id": rid,
                    "geometry_method": proposal.get("geometry_method"),
                    "deterministic_rows": int(
                        proposal.get("deterministic_row_count") or 0
                    ),
                    "deterministic_columns": int(
                        proposal.get("deterministic_column_count") or 0
                    ),
                    "geometry_recovery": proposal.get("geometry_recovery") or {},
                    "expected_header_rows": int(
                        assessment.get("expected_header_rows") or 0
                    ),
                    "expected_item_rows": int(
                        assessment.get("expected_item_rows") or 0
                    ),
                    "materialized_item_rows": len(ordered_rows),
                    "rows_with_manufacturer": sum(
                        1
                        for row in ordered_rows
                        if _clean_text(
                            row.get("manufacturer_normalized")
                            or row.get("manufacturer_original"),
                            1000,
                        )
                    ),
                    "blank_manufacturer_rows": sum(
                        1
                        for row in ordered_rows
                        if not _clean_text(
                            row.get("manufacturer_normalized")
                            or row.get("manufacturer_original"),
                            1000,
                        )
                    ),
                    "component_tag_sequence": [
                        _clean_text(
                            row.get("component_tag_original"),
                            300,
                        )
                        for row in ordered_rows
                    ],
                }
            )

        unresolved_duplicate_keys, resolved_duplicate_values = (
            _adjudicate_verifier_duplicate_physical_keys(
                reported_keys=verifier.get("duplicate_physical_keys") or [],
                rows=rows,
            )
        )

        return {
            "electrical_document_id": context["electrical_document_id"],
            "electrical_version_id": context["version_id"],
            "pdf_page_number": page["pdf_page_number"],
            "sheet_code": page["sheet_code"],
            "page_type": page["page_type"],
            "language": detector.get("language")
            or page.get("classification_language"),
            "page_passed": bool(page_passed),
            "proposals_count": len(proposals),
            "tables_extracted": len(extractions),
            "physical_item_rows_expected": sum(
                int(x.get("expected_item_rows") or 0)
                for x in assessment_map.values()
            ),
            "published_bom_rows": len(rows) if page_passed else 0,
            "blocking_issue_count_this_page": blocking,
            "warning_issue_count_this_page": warning,
            "duplicate_value_rows_preserved": bool(
                page_passed and not unresolved_duplicate_keys
            ),
            "duplicate_value_adjudication": resolved_duplicate_values,
            "region_stats": region_stats,
            "severity_counts": _severity_counts(issues),
            **state,
            **_db_ai_totals(context["version_id"]),
            **usage_totals,
        }
    finally:
        source_doc.close()
