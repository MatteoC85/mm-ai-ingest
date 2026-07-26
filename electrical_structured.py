import base64
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Optional

import fitz
import psycopg2
import requests

from electrical_source_store import download_electrical_source_pdf

# MachineMind Phase 2 V2
# Multimodal, geometry-first, page -> region -> row extraction with independent audit.
# The deterministic layer never classifies by fixed Italian/English keywords.


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
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

ELECTRICAL_STRUCTURED_ENABLED = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_ENABLED") or "0"
).strip() == "1"

DETECTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_DETECTOR_MODEL") or "gpt-5.4"
).strip()
EXTRACTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_EXTRACTOR_MODEL") or "gpt-5.4"
).strip()
VERIFIER_MODEL = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_VERIFIER_MODEL") or "gpt-5.4"
).strip()

DETECTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_DETECTOR_PROMPT_VERSION")
    or "mm-electrical-region-detector-v2"
).strip()
EXTRACTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_EXTRACTOR_PROMPT_VERSION")
    or "mm-electrical-table-extractor-v2.2"
).strip()
VERIFIER_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_VERIFIER_PROMPT_VERSION")
    or "mm-electrical-page-verifier-v2.4"
).strip()
MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_STRUCTURED_MATERIALIZER_VERSION")
    or "mm-electrical-structured-materializer-v2.5"
).strip()

OPENAI_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_STRUCTURED_TIMEOUT_SECONDS", 240, 30, 600
)
FETCH_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_STRUCTURED_FETCH_TIMEOUT_SECONDS", 60, 10, 300
)
RENDER_DPI = _env_int("MM_ELECTRICAL_STRUCTURED_RENDER_DPI", 220, 120, 320)
MAX_COMPLETION_TOKENS = _env_int(
    "MM_ELECTRICAL_STRUCTURED_MAX_COMPLETION_TOKENS", 16000, 1000, 64000
)
ROW_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_STRUCTURED_ROW_MIN_CONFIDENCE", 0.80, 0.0, 1.0
)
PAGE_PASS_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_STRUCTURED_PAGE_PASS_MIN_CONFIDENCE", 0.90, 0.0, 1.0
)
TEXT_RECONSTRUCTION_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_STRUCTURED_TEXT_RECONSTRUCTION_MIN_CONFIDENCE",
    0.90,
    0.0,
    1.0,
)
INPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_STRUCTURED_INPUT_USD_PER_MILLION", 0.0
)
OUTPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_STRUCTURED_OUTPUT_USD_PER_MILLION", 0.0
)
MAX_SOURCE_BYTES = _env_int(
    "MM_ELECTRICAL_STRUCTURED_MAX_SOURCE_BYTES", 100_000_000, 1_000_000, 500_000_000
)

PIPELINE_MARKER = "phase2-vision-v2.5-source-snapshot"
COLUMN_BINDING_ADJUDICATOR_VERSION = ("peer-equivalent-column-consensus-v1")
EXTRACTION_METHOD = "openai_vision_structured_v2"
PHASE_NAME = "structured_vision_v2"
IO_PAGE_TYPES = {"plc_io_table", "safety_io_table"}
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
ROW_ROLES = {
    "title",
    "column_header",
    "signal",
    "placeholder",
    "power",
    "connector_aux",
    "blank_unused",
    "other_data",
}
SEVERITIES = {"info", "warning", "high", "critical"}
VISUAL_TEXT_FIELDS = {"signal_name", "description", "expected_normal_state"}

COLUMN_BINDING_ROLES = {
    "channel_ref",
    "terminal_reference",
    "wire_reference",
    "plc_address",
    "connector_reference",
    "signal_name",
    "description",
    "expected_normal_state",
    "other_data",
}
REFERENCE_COLUMN_ROLES = {
    "channel_ref",
    "terminal_reference",
    "wire_reference",
    "plc_address",
    "connector_reference",
}
REFERENCE_ROLE_TO_RESULT_FIELD = {
    "channel_ref": "channel_ref_original",
    "terminal_reference": "terminal_reference_original",
    "wire_reference": "wire_reference_original",
    "plc_address": "plc_address_original",
    "connector_reference": "connector_ref_original",
}
REFERENCE_ROLE_MAX_LENGTH = {
    "channel_ref": 160,
    "terminal_reference": 300,
    "wire_reference": 300,
    "plc_address": 200,
    "connector_reference": 200,
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
    # Legacy keys remain so main.py can expose them without breaking startup.
    return {
        "enabled": bool(ELECTRICAL_STRUCTURED_ENABLED),
        "model": EXTRACTOR_MODEL,
        "prompt_version": EXTRACTOR_PROMPT_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "min_confidence": ROW_MIN_CONFIDENCE,
        "pipeline_marker": PIPELINE_MARKER,
        "detector_model": DETECTOR_MODEL,
        "extractor_model": EXTRACTOR_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "detector_prompt_version": DETECTOR_PROMPT_VERSION,
        "extractor_prompt_version": EXTRACTOR_PROMPT_VERSION,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
        "text_reconstruction_min_confidence": (
            TEXT_RECONSTRUCTION_MIN_CONFIDENCE
        ),
        "render_dpi": RENDER_DPI,
    }


def _clean_text(value: Any, max_len: int = 2000) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _semantic_character_signature(value: Any) -> str:
    """Return a language-independent character signature for safe re-spacing.

    The normalized display text may repair word boundaries and punctuation spacing,
    but it must preserve the original alphanumeric content and order. This blocks
    paraphrases, translations, spelling guesses, and invented technical terms.
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
        .replace("−", "-")
        .upper()
    )
    return "".join(ch for ch in text if ch.isalnum())


def _safe_normalized_display_text(
    *,
    original: Any,
    normalized: Any,
    max_len: int,
    field_name: str,
    region_id: str,
    row_id: str,
    fail,
) -> str:
    original_text = _clean_text(original, max_len)
    normalized_text = _clean_text(normalized, max_len)

    if not original_text:
        if normalized_text:
            fail(
                "normalized_text_without_source",
                f"Region {region_id} row {row_id} has normalized {field_name} "
                "without source text",
                region_id=region_id,
                row_ids=[row_id],
                field_name=field_name,
                normalized_text=normalized_text,
            )
        return ""

    if not normalized_text:
        fail(
            "missing_normalized_text",
            f"Region {region_id} row {row_id} is missing normalized "
            f"{field_name}",
            region_id=region_id,
            row_ids=[row_id],
            field_name=field_name,
            original_text=original_text,
        )
        return original_text

    original_sig = _semantic_character_signature(original_text)
    normalized_sig = _semantic_character_signature(normalized_text)
    if original_sig != normalized_sig:
        fail(
            "normalized_text_not_character_equivalent",
            f"Region {region_id} row {row_id} normalized {field_name} "
            "changes source alphanumeric content",
            region_id=region_id,
            row_ids=[row_id],
            field_name=field_name,
            original_text=original_text,
            normalized_text=normalized_text,
            original_signature=original_sig,
            normalized_signature=normalized_sig,
        )

    return normalized_text


def _canonical_key(value: Any, max_len: int = 120) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value or "none")[:max_len]


def _json_obj(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return value


def _clamp_conf(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except Exception:
        return 0.0


def _price(input_tokens: int, output_tokens: int) -> float:
    return round(
        max(0, int(input_tokens or 0)) / 1_000_000.0 * INPUT_USD_PER_MILLION
        + max(0, int(output_tokens or 0)) / 1_000_000.0 * OUTPUT_USD_PER_MILLION,
        6,
    )


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


def _parse_chat_content(data: dict) -> str:
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"OpenAI refused structured vision request: {str(refusal)[:800]}")
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
            "OpenAI electrical vision call failed: "
            f"{response.status_code} {response.text[:1800]}"
        )

    data = response.json()
    text = _parse_chat_content(data)
    if not text:
        raise RuntimeError("OpenAI electrical vision call returned empty content")
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            f"Electrical vision JSON parse failed: {exc}; raw={text[:1200]}"
        ) from exc

    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
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
    # AI artifacts depend on the model, prompt and exact request only.
    # Local publication/materializer changes must never force paid AI reruns.
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
                       reasoning_tokens, cost_usd, model, prompt_version, fingerprint
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
    """Backward-compatible cache lookup.

    Earlier Phase 2 V2 fingerprints included MATERIALIZER_VERSION. Reuse the
    already-paid AI response when task, model, prompt and exact request are the
    same, even if only local publication logic changed.
    """
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, response_json, input_tokens, output_tokens,
                       reasoning_tokens, cost_usd, model, prompt_version, fingerprint
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
                    str(task_type),
                    str(model),
                    str(prompt_version),
                    str(request_sha256),
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
                    response_json, input_tokens, output_tokens, reasoning_tokens,
                    cost_usd, status, error_message, created_at, completed_at
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
    page_types: Optional[list[str]],
    pdf_page_numbers: Optional[list[int]],
) -> dict:
    requested_types = sorted(set(page_types or IO_PAGE_TYPES))
    unsupported = set(requested_types) - IO_PAGE_TYPES
    if unsupported:
        raise ValueError(
            "Phase 2 V2 is intentionally limited to verified I/O pages. "
            "Unsupported page_types: " + ", ".join(sorted(unsupported))
        )

    page_numbers = sorted({int(x) for x in (pdf_page_numbers or []) if int(x) > 0})
    if len(page_numbers) != 1:
        raise ValueError(
            "Phase 2 V2 requires exactly one pdf_page_numbers value per request "
            "to keep extraction atomic and within the Cloud Run timeout."
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
                raise ValueError("Electrical version not found for supplied scope")

            cur.execute(
                """
                SELECT id, pdf_page_number, sheet_code, sheet_title, group_code,
                       page_type, page_width_pt, page_height_pt, page_sha256,
                       raw_text, text_spans_json, classification_language,
                       semantic_confidence, classification_metadata
                FROM public.electrical_pages
                WHERE version_id=%s
                  AND page_type=ANY(%s)
                  AND pdf_page_number=ANY(%s)
                ORDER BY pdf_page_number;
                """,
                (int(row[2]), requested_types, page_numbers),
            )
            pages: list[dict] = []
            for p in cur.fetchall():
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
                        "words": list(_json_obj(p[10], []) or []),
                        "classification_language": str(p[11] or "unknown"),
                        "semantic_confidence": float(p[12] or 0.0),
                        "classification_metadata": _json_obj(p[13], {}) or {},
                    }
                )
            if len(pages) != 1:
                raise ValueError(
                    "Requested page was not found among semantically classified I/O pages"
                )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_pages
                WHERE version_id=%s AND page_type=ANY(%s);
                """,
                (int(row[2]), sorted(IO_PAGE_TYPES)),
            )
            all_io_pages_total = int(cur.fetchone()[0] or 0)

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
                "declared_sheet_count": int(row[7]) if row[7] is not None else None,
                "source_sha256": str(row[8] or ""),
                "source_snapshot_uri": str(
                    source_snapshot.get("uri")
                    or version_metadata.get("source_snapshot_uri")
                    or ""
                ).strip(),
                "file_url": str(row[9] or ""),
                "pages": pages,
                "all_io_pages_total": all_io_pages_total,
                "requested_page_types": requested_types,
                "requested_page_numbers": page_numbers,
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
                "SOURCE_SNAPSHOT_READ_FAILED: the private persisted electrical PDF "
                f"could not be read: {str(exc)[:700]}"
            ) from exc
    else:
        # Legacy fallback only. New ingests persist an immutable private GCS snapshot
        # and never depend on an expiring Bubble/CDN signed URL after ingest.
        url = str(context.get("file_url") or "").strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url:
            raise ValueError(
                "SOURCE_SNAPSHOT_MISSING: this legacy electrical version has no private "
                "source snapshot and no usable fallback URL. Backfill the source snapshot "
                "once, then retry."
            )
        try:
            response = requests.get(
                url,
                timeout=FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status in {401, 403}:
                raise ValueError(
                    "SOURCE_SNAPSHOT_MISSING: the legacy Bubble/CDN signed URL is no "
                    "longer authorized. Backfill the immutable private source snapshot "
                    "once; do not keep refreshing signed URLs."
                ) from exc
            raise
        data = response.content

    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Electrical source PDF is empty or exceeds the configured size limit")
    actual_sha = _sha256_bytes(data)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            "Electrical source PDF SHA-256 does not match the indexed version; "
            "refusing to analyze a different file."
        )
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Electrical source PDF cannot be opened: {exc}") from exc
    if len(doc) != int(context.get("pdf_page_count") or len(doc)):
        doc.close()
        raise ValueError("Electrical source PDF page count differs from indexed version")
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


def _canonical_hardware_model_code(
    value: Any,
    *,
    module_tag: str,
    sheet_code: str,
) -> tuple[str, str]:
    """Return only a strongly code-like hardware model identifier.

    Table headers often contain the module tag plus a functional description
    (for example a channel count and I/O function). Those descriptions are not
    hardware model numbers and must not be written into module_model. The raw
    candidate and full header remain preserved in properties for audit.
    """
    raw = _clean_text(value, 240)
    if not raw:
        return "", "absent"

    excluded = {
        re.sub(r"[^a-z0-9]", "", str(module_tag or "").lower()),
        re.sub(r"[^a-z0-9]", "", str(sheet_code or "").lower()),
    }
    excluded.discard("")

    accepted: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/\-]{3,}", raw):
        normalized = re.sub(r"[^a-z0-9]", "", token.lower())
        if not normalized or normalized in excluded:
            continue
        letters = sum(ch.isalpha() for ch in normalized)
        digits = sum(ch.isdigit() for ch in normalized)
        # Require a sufficiently distinctive mixed alphanumeric code. This
        # rejects channel counts and prose while accepting identifiers such as
        # SDI101, STO081 and 6ES7131-6BF01-0BA0.
        if len(normalized) < 5 or letters < 2 or digits < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        accepted.append(token)

    if not accepted:
        return "", "not_visually_explicit"
    return " ".join(accepted)[:200], "strong_code"


def _rect_list(rect: fitz.Rect, digits: int = 2) -> list[float]:
    return [round(float(rect.x0), digits), round(float(rect.y0), digits),
            round(float(rect.x1), digits), round(float(rect.y1), digits)]


def _rect_from(value: Any) -> fitz.Rect:
    if isinstance(value, fitz.Rect):
        return value
    return fitz.Rect(*[float(x) for x in value])


def _word_center(word: dict) -> tuple[float, float]:
    return ((word["x0"] + word["x1"]) / 2.0, (word["y0"] + word["y1"]) / 2.0)


def _ids_in_rect(word_map: dict[int, dict], rect: fitz.Rect) -> list[int]:
    ids: list[int] = []
    for wid, word in word_map.items():
        cx, cy = _word_center(word)
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            ids.append(wid)
    return sorted(ids, key=lambda wid: (word_map[wid]["y0"], word_map[wid]["x0"], wid))


def _text_for_ids(ids: list[int], word_map: dict[int, dict], max_len: int = 5000) -> str:
    words = [word_map[i] for i in ids if i in word_map]
    words.sort(key=lambda w: (w["y0"], w["x0"], w["id"]))
    return _clean_text(" ".join(w["text"] for w in words), max_len)


def _bbox_for_ids(ids: list[int], word_map: dict[int, dict], fallback: fitz.Rect) -> fitz.Rect:
    words = [word_map[i] for i in ids if i in word_map]
    if not words:
        return fallback
    return fitz.Rect(
        min(w["x0"] for w in words),
        min(w["y0"] for w in words),
        max(w["x1"] for w in words),
        max(w["y1"] for w in words),
    )


def _table_is_outer_frame(table: Any, page_rect: fitz.Rect) -> bool:
    bbox = _rect_from(table.bbox)
    area_ratio = bbox.get_area() / max(1.0, page_rect.get_area())
    touches = (
        bbox.x0 <= page_rect.x0 + 10
        and bbox.y0 <= page_rect.y0 + 10
        and bbox.x1 >= page_rect.x1 - 10
        and bbox.y1 >= page_rect.y1 - 10
    )
    return area_ratio >= 0.62 or (touches and int(table.row_count or 0) <= 8)


def _detect_table_proposals(
    *,
    source_page: fitz.Page,
    inventory_page: dict,
    word_map: dict[int, dict],
) -> list[dict]:
    finder = source_page.find_tables()
    tables = [
        table
        for table in (finder.tables or [])
        if int(table.row_count or 0) >= 4
        and not _table_is_outer_frame(table, source_page.rect)
    ]
    tables.sort(key=lambda t: (_rect_from(t.bbox).y0, _rect_from(t.bbox).x0))
    if not tables:
        return []

    # Most I/O pages arrange the independently bordered tables horizontally.
    # Boundaries are based only on geometry and gaps, never on labels/language.
    bboxes = [_rect_from(t.bbox) for t in tables]
    proposals: list[dict] = []
    for index, (table, bbox) in enumerate(zip(tables, bboxes), start=1):
        prev_bbox = bboxes[index - 2] if index > 1 else None
        next_bbox = bboxes[index] if index < len(bboxes) else None
        left = max(source_page.rect.x0, bbox.x0 - 30.0)
        right = min(source_page.rect.x1, bbox.x1 + 30.0)
        if prev_bbox is not None and abs(prev_bbox.y0 - bbox.y0) < 80:
            left = max(source_page.rect.x0, (prev_bbox.x1 + bbox.x0) / 2.0)
        if next_bbox is not None and abs(next_bbox.y0 - bbox.y0) < 80:
            right = min(source_page.rect.x1, (bbox.x1 + next_bbox.x0) / 2.0)
        crop_rect = fitz.Rect(
            max(source_page.rect.x0, left),
            max(source_page.rect.y0, bbox.y0 - 22.0),
            min(source_page.rect.x1, right),
            min(source_page.rect.y1, bbox.y1 + 10.0),
        )

        row_candidates: list[dict] = []
        extracted_rows = table.extract() or []
        for row_index, table_row in enumerate(table.rows or [], start=1):
            row_bbox = _rect_from(table_row.bbox)
            expanded_row = fitz.Rect(crop_rect.x0, row_bbox.y0, crop_rect.x1, row_bbox.y1)
            ids = _ids_in_rect(word_map, expanded_row)
            cell_texts: list[str] = []
            if row_index - 1 < len(extracted_rows):
                cell_texts = [
                    _clean_text(cell, 800) if cell is not None else ""
                    for cell in (extracted_rows[row_index - 1] or [])
                ]
            row_candidates.append(
                {
                    "row_id": f"R{row_index:02d}",
                    "row_index": row_index,
                    "bbox_pt": _rect_list(expanded_row),
                    "word_ids": ids,
                    "word_text_original": _text_for_ids(ids, word_map, 3000),
                    "deterministic_cell_text_original": cell_texts,
                }
            )

        region_id = f"P{int(inventory_page['pdf_page_number'])}-T{index:02d}"
        proposal_core = {
            "region_id": region_id,
            "table_bbox_pt": _rect_list(bbox),
            "crop_bbox_pt": _rect_list(crop_rect),
            "deterministic_row_count": int(table.row_count or 0),
            "deterministic_column_count": int(table.col_count or 0),
            "row_candidates": row_candidates,
        }
        proposal_core["region_hash"] = _sha256_json(
            {
                "page_sha256": inventory_page.get("page_sha256"),
                **proposal_core,
            }
        )
        proposals.append(proposal_core)
    return proposals


def _render_page_with_regions(
    source_doc: fitz.Document,
    page_index: int,
    proposals: list[dict],
) -> bytes:
    source_page = source_doc[page_index]
    overlay = fitz.open()
    try:
        page = overlay.new_page(width=source_page.rect.width, height=source_page.rect.height)
        page.show_pdf_page(page.rect, source_doc, page_index)
        for proposal in proposals:
            rect = _rect_from(proposal["crop_bbox_pt"])
            page.draw_rect(rect, color=(1, 0, 0), width=0.8, overlay=True)
            page.insert_text(
                (rect.x0 + 2, max(rect.y0 + 8, 9)),
                proposal["region_id"],
                fontsize=6,
                color=(1, 0, 0),
                overlay=True,
            )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0),
            alpha=False,
        )
        return pix.tobytes("png")
    finally:
        overlay.close()


def _render_region_with_rows(
    source_doc: fitz.Document,
    page_index: int,
    proposal: dict,
) -> bytes:
    clip = _rect_from(proposal["crop_bbox_pt"])
    overlay = fitz.open()
    try:
        page = overlay.new_page(width=clip.width, height=clip.height)
        page.show_pdf_page(page.rect, source_doc, page_index, clip=clip)
        for row in proposal["row_candidates"]:
            original = _rect_from(row["bbox_pt"])
            local = fitz.Rect(
                0,
                max(0.0, original.y0 - clip.y0),
                clip.width,
                min(clip.height, original.y1 - clip.y0),
            )
            page.draw_rect(local, color=(1, 0, 0), width=0.35, overlay=True)
            page.insert_text(
                (2, min(clip.height - 2, max(6, local.y0 + 6))),
                row["row_id"],
                fontsize=5,
                color=(1, 0, 0),
                overlay=True,
            )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0),
            alpha=False,
        )
        return pix.tobytes("png")
    finally:
        overlay.close()


def _data_url_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _bbox_schema() -> dict:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
    }


def _issue_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issue_type": {"type": "string"},
            "severity": {"type": "string", "enum": sorted(SEVERITIES)},
            "message": {"type": "string"},
            "region_id": {"type": "string"},
            "row_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 100,
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


def _detector_schema() -> dict:
    return {
        "name": "electrical_io_region_detector_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "language": {"type": "string"},
                "proposal_assessments": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "visible": {"type": "boolean"},
                            "distinct_table": {"type": "boolean"},
                            "kind": {
                                "type": "string",
                                "enum": ["io_table", "other_table", "not_table"],
                            },
                            "visible_header_row_count": {"type": "integer"},
                            "visible_physical_row_count": {"type": "integer"},
                            "bbox_pt": _bbox_schema(),
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "visible",
                            "distinct_table",
                            "kind",
                            "visible_header_row_count",
                            "visible_physical_row_count",
                            "bbox_pt",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "missing_visible_io_tables": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "temporary_id": {"type": "string"},
                            "bbox_pt": _bbox_schema(),
                            "visible_physical_row_count": {"type": "integer"},
                            "reason": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "temporary_id",
                            "bbox_pt",
                            "visible_physical_row_count",
                            "reason",
                            "confidence",
                        ],
                    },
                },
                "all_visible_io_tables_accounted_for": {"type": "boolean"},
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 30,
                },
            },
            "required": [
                "page_id",
                "language",
                "proposal_assessments",
                "missing_visible_io_tables",
                "all_visible_io_tables_accounted_for",
                "confidence",
                "issues",
            ],
        },
    }


def _extractor_schema() -> dict:
    return {
        "name": "electrical_io_region_extractor_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "region_id": {"type": "string"},
                "language": {"type": "string"},
                "module_tag_original": {"type": "string"},
                "module_model_original": {"type": "string"},
                "table_label_original": {"type": "string"},
                "io_type": {"type": "string", "enum": sorted(IO_TYPES)},
                "is_safety": {"type": "boolean"},
                "module_header_source_word_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 120,
                },
                "visible_header_row_count": {"type": "integer"},
                "visible_physical_row_count": {"type": "integer"},
                "row_results": {
                    "type": "array",
                    "maxItems": 128,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "row_id": {"type": "string"},
                            "row_role": {"type": "string", "enum": sorted(ROW_ROLES)},
                            "include_in_io": {"type": "boolean"},
                            "channel_ref_original": {"type": "string"},
                            "connector_ref_original": {"type": "string"},
                            "plc_address_original": {"type": "string"},
                            "wire_reference_original": {"type": "string"},
                            "terminal_reference_original": {"type": "string"},
                            "signal_name_original": {"type": "string"},
                            "signal_name_normalized": {"type": "string"},
                            "description_original": {"type": "string"},
                            "description_normalized": {"type": "string"},
                            "expected_normal_state_original": {"type": "string"},
                            "expected_normal_state_normalized": {"type": "string"},
                            "text_reconstruction_confidence": {"type": "number"},
                            "text_reconstruction_note": {"type": "string"},
                            "is_placeholder": {"type": "boolean"},
                            "source_word_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 160,
                            },
                            "confidence": {"type": "number"},
                            "note": {"type": "string"},
                        },
                        "required": [
                            "row_id",
                            "row_role",
                            "include_in_io",
                            "channel_ref_original",
                            "connector_ref_original",
                            "plc_address_original",
                            "wire_reference_original",
                            "terminal_reference_original",
                            "signal_name_original",
                            "signal_name_normalized",
                            "description_original",
                            "description_normalized",
                            "expected_normal_state_original",
                            "expected_normal_state_normalized",
                            "text_reconstruction_confidence",
                            "text_reconstruction_note",
                            "is_placeholder",
                            "source_word_ids",
                            "confidence",
                            "note",
                        ],
                    },
                },
                "unaccounted_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 128,
                },
                "duplicate_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 128,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 30,
                },
            },
            "required": [
                "page_id",
                "region_id",
                "language",
                "module_tag_original",
                "module_model_original",
                "table_label_original",
                "io_type",
                "is_safety",
                "module_header_source_word_ids",
                "visible_header_row_count",
                "visible_physical_row_count",
                "row_results",
                "unaccounted_row_ids",
                "duplicate_row_ids",
                "confidence",
                "issues",
            ],
        },
    }


def _verifier_schema() -> dict:
    return {
        "name": "electrical_io_page_verifier_v2_4",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": ["pass", "block"]},
                "all_visible_tables_accounted_for": {"type": "boolean"},
                "all_visible_physical_rows_accounted_for": {"type": "boolean"},
                "all_module_tags_supported_by_own_headers": {"type": "boolean"},
                "all_published_text_visually_supported": {"type": "boolean"},
                "all_visible_text_resolved_or_blocked": {"type": "boolean"},
                "all_visible_columns_accounted_for": {"type": "boolean"},
                "all_reference_columns_materialized": {"type": "boolean"},
                "sheet_code_used_as_module_tag_without_visual_support": {
                    "type": "boolean"
                },
                "column_binding_decisions": {
                    "type": "array",
                    "maxItems": 96,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "source_column_index": {
                                "type": "integer",
                                "minimum": 0
                            },
                            "header_text_original": {"type": "string"},
                            "canonical_roles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": sorted(COLUMN_BINDING_ROLES),
                                },
                                "minItems": 1,
                                "maxItems": 6,
                            },
                            "use_for_missing_values": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "source_column_index",
                            "header_text_original",
                            "canonical_roles",
                            "use_for_missing_values",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "column_binding_checks": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "visible_column_count": {"type": "integer"},
                            "accounted_column_count": {"type": "integer"},
                            "pass": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "visible_column_count",
                            "accounted_column_count",
                            "pass",
                            "reason",
                        ],
                    },
                },
                "region_checks": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "module_tag_original": {"type": "string"},
                            "visible_physical_row_count": {"type": "integer"},
                            "accounted_physical_row_count": {"type": "integer"},
                            "visible_text_row_count": {"type": "integer"},
                            "supported_text_row_count": {"type": "integer"},
                            "pass": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "module_tag_original",
                            "visible_physical_row_count",
                            "accounted_physical_row_count",
                            "visible_text_row_count",
                            "supported_text_row_count",
                            "pass",
                            "reason",
                        ],
                    },
                },
                "text_overrides": {
                    "type": "array",
                    "maxItems": 240,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "row_id": {"type": "string"},
                            "field_name": {
                                "type": "string",
                                "enum": sorted(VISUAL_TEXT_FIELDS),
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
                "blocked_text_rows": {
                    "type": "array",
                    "maxItems": 240,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "row_id": {"type": "string"},
                            "field_name": {
                                "type": "string",
                                "enum": sorted(VISUAL_TEXT_FIELDS | {"all_free_text"}),
                            },
                            "reason": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "region_id",
                            "row_id",
                            "field_name",
                            "reason",
                            "confidence",
                        ],
                    },
                },
                "missing_region_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 24,
                },
                "missing_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 200,
                },
                "duplicate_row_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 200,
                },
                "duplicate_source_module_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 24,
                },
                "unaccounted_visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 30,
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
                "verdict",
                "all_visible_tables_accounted_for",
                "all_visible_physical_rows_accounted_for",
                "all_module_tags_supported_by_own_headers",
                "all_published_text_visually_supported",
                "all_visible_text_resolved_or_blocked",
                "all_visible_columns_accounted_for",
                "all_reference_columns_materialized",
                "sheet_code_used_as_module_tag_without_visual_support",
                "column_binding_decisions",
                "column_binding_checks",
                "region_checks",
                "text_overrides",
                "blocked_text_rows",
                "missing_region_ids",
                "missing_row_ids",
                "duplicate_row_keys",
                "duplicate_source_module_tags",
                "unaccounted_visual_evidence",
                "confidence",
                "issues",
            ],
        },
    }


def _image_message(text: str, png_bytes: bytes) -> list[dict]:
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(png_bytes),
                "detail": "original",
            },
        },
    ]


def _detector_messages(page: dict, proposals: list[dict], image: bytes) -> list[dict]:
    proposal_summary = [
        {
            "region_id": p["region_id"],
            "bbox_pt": p["crop_bbox_pt"],
            "deterministic_row_count": p["deterministic_row_count"],
            "deterministic_column_count": p["deterministic_column_count"],
        }
        for p in proposals
    ]
    system = (
        "You are the visual perception stage of an industrial electrical-schematic "
        "reader. Work from the complete page image and geometry proposals. The source "
        "may be Italian, English, mixed, or another language. Understand meaning "
        "semantically; do not depend on a vocabulary list. Identify every visually "
        "distinct I/O table. Never collapse adjacent tables. Count every physical body "
        "row, including signal rows, placeholders, unused rows, connector A/B rows, "
        "power rows, and blank-but-bordered rows. Title and column-header rows are "
        "headers, not physical body rows. A repeated module tag in two visible tables "
        "must be preserved, not corrected. Do not infer missing text."
    )
    user_text = (
        "Audit the red geometry proposals on this full page. Return one assessment for "
        "every proposal ID and list any visible I/O table not covered by a proposal.\n\n"
        + json.dumps(
            {
                "page_id": page["id"],
                "pdf_page_number": page["pdf_page_number"],
                "sheet_code_original": page.get("sheet_code"),
                "sheet_title_original": page.get("sheet_title"),
                "page_type": page.get("page_type"),
                "page_width_pt": page.get("page_width_pt"),
                "page_height_pt": page.get("page_height_pt"),
                "proposals": proposal_summary,
            },
            ensure_ascii=False,
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _image_message(user_text, image)},
    ]


def _extractor_messages(
    page: dict,
    proposal: dict,
    detector_assessment: dict,
    image: bytes,
) -> list[dict]:
    system = (
        "You are reading exactly one cropped I/O table from an industrial electrical "
        "schematic. Use the crop image and the exact vector-word row candidates. The "
        "document can be in any language. Reason semantically, not by matching a fixed "
        "Italian/English dictionary. Preserve original visible strings. The module tag "
        "must come from this table's own visible header; never substitute the page sheet "
        "code. Return exactly one row_result for every supplied row_id. Every physical "
        "body row must have include_in_io=true, including placeholders, unused rows, "
        "blank bordered rows, power rows, and connector/auxiliary A-B rows. Only title "
        "and column-header rows have include_in_io=false. Do not omit a row because its "
        "meaning is unclear. Use source_word_ids from that row; a truly blank bordered "
        "row may have an empty source list and must still be accounted for by row_id. "
        "For signal_name, description, and expected_normal_state return two versions: "
        "the *_original field preserves the exact source characters and order, including "
        "artificial PDF spacing; the *_normalized field repairs only word boundaries and "
        "spacing visible in the crop. Normalized text must stay in the original language, "
        "must not translate or paraphrase, and must not add, remove, reorder, or correct "
        "any alphanumeric character. Preserve numbers, codes, abbreviations, punctuation, "
        "and symbols. If the source is already readable, normalized equals original. If "
        "safe reconstruction is impossible, keep normalized equal to original and lower "
        "text_reconstruction_confidence. Do not invent addresses, wires, tags, or descriptions."
    )
    compact_rows = [
        {
            "row_id": row["row_id"],
            "bbox_pt": row["bbox_pt"],
            "word_ids": row["word_ids"],
            "word_text_original": row["word_text_original"],
            "deterministic_cell_text_original": row[
                "deterministic_cell_text_original"
            ],
        }
        for row in proposal["row_candidates"]
    ]
    user_text = (
        "Extract and account for this single red-row-annotated table.\n\n"
        + json.dumps(
            {
                "page_id": page["id"],
                "pdf_page_number": page["pdf_page_number"],
                "sheet_code_original": page.get("sheet_code"),
                "region_id": proposal["region_id"],
                "region_bbox_pt": proposal["crop_bbox_pt"],
                "detector_visible_header_row_count": detector_assessment.get(
                    "visible_header_row_count"
                ),
                "detector_visible_physical_row_count": detector_assessment.get(
                    "visible_physical_row_count"
                ),
                "row_candidates": compact_rows,
            },
            ensure_ascii=False,
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _image_message(user_text, image)},
    ]


def _verifier_messages(
    page: dict,
    detector: dict,
    proposals: list[dict],
    extractions: list[dict],
    column_binding_proposals: list[dict],
    page_image: bytes,
    region_images: dict[str, bytes],
) -> list[dict]:
    proposal_by_id = {p["region_id"]: p for p in proposals}
    extraction_summary = []
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        proposal = proposal_by_id.get(rid) or {}
        header_count = max(
            0,
            min(
                int(extraction.get("visible_header_row_count") or 0),
                len(proposal.get("row_candidates") or []),
            ),
        )
        header_source_text = " | ".join(
            _clean_text(row.get("word_text_original"), 800)
            for row in (proposal.get("row_candidates") or [])[:header_count]
            if _clean_text(row.get("word_text_original"), 800)
        )
        rows = extraction.get("row_results") or []
        extraction_summary.append(
            {
                "region_id": rid,
                "module_header_source_text": header_source_text,
                "module_tag_original": extraction.get("module_tag_original"),
                # This is only a hardware-model candidate. It is allowed to be
                # empty or incomplete and must never be compared with the full
                # visible header.
                "module_hardware_model_candidate": extraction.get(
                    "module_model_original"
                ),
                "table_label_original": extraction.get("table_label_original"),
                "io_type": extraction.get("io_type"),
                "visible_header_row_count": extraction.get(
                    "visible_header_row_count"
                ),
                "visible_physical_row_count": extraction.get(
                    "visible_physical_row_count"
                ),
                "included_rows": [
                    {
                        "row_id": row.get("row_id"),
                        "row_role": row.get("row_role"),
                        "channel_ref_original": row.get("channel_ref_original"),
                        "connector_ref_original": row.get("connector_ref_original"),
                        "plc_address_original": row.get("plc_address_original"),
                        "wire_reference_original": row.get("wire_reference_original"),
                        "terminal_reference_original": row.get("terminal_reference_original"),
                        "signal_name_original": row.get("signal_name_original"),
                        "signal_name_candidate": row.get("signal_name_normalized"),
                        "description_original": row.get("description_original"),
                        "description_candidate": row.get("description_normalized"),
                        "expected_normal_state_original": row.get(
                            "expected_normal_state_original"
                        ),
                        "expected_normal_state_candidate": row.get(
                            "expected_normal_state_normalized"
                        ),
                        "text_reconstruction_confidence": row.get(
                            "text_reconstruction_confidence"
                        ),
                        "text_reconstruction_note": row.get(
                            "text_reconstruction_note"
                        ),
                        "is_placeholder": row.get("is_placeholder"),
                    }
                    for row in rows
                    if row.get("include_in_io")
                ],
                "excluded_rows": [
                    row.get("row_id") for row in rows if not row.get("include_in_io")
                ],
            }
        )

    system = (
        "You are the independent visual adjudication stage of an industrial "
        "electrical-schematic reader. Re-read the complete page and every supplied "
        "high-resolution table crop. The source may be in any language. Do not trust "
        "OCR spacing or the previous extraction when the image contradicts it. Verify "
        "that every visible table, physical body row, and physical data column is "
        "accounted for and that each module tag comes from its own header. The field "
        "module_hardware_model_candidate is optional and is NOT the full header; do not "
        "fail a region merely because it is blank, contains only a functional label, or "
        "does not repeat words such as a generic device label. A hardware model is valid "
        "only when a distinct model/code is actually visible. "
        "Semantically bind every physical source column to one or more canonical roles "
        "without relying on a fixed Italian/English vocabulary. Return exactly one "
        "column_binding_decision for every source_column_index in every region. A single "
        "physical column may legitimately serve multiple roles only when its own visible "
        "header, body values, and table function support that interpretation. For "
        "example, a visible row/terminal index can also identify the module channel, but "
        "do not duplicate one field into another merely because their values happen to "
        "match. Use peer tables on the same page only as corroborating evidence; each "
        "decision must remain supported by the region's own crop. Set "
        "use_for_missing_values=true only when exact body-cell text can safely populate "
        "a missing short reference field (channel, terminal, wire, address, or connector). "
        "Every visible column must be assigned at least one role; use other_data only for "
        "a genuinely non-canonical column. Confirm that every visible reference column "
        "is materialized for every row, including placeholders. "
        "For each non-placeholder row with free text, compare the candidate against the "
        "high-resolution crop. In region_checks, visible_text_row_count and "
        "supported_text_row_count count only included non-placeholder rows that have "
        "at least one non-empty signal, description, or expected-state field. If the "
        "candidate is already an exact readable visual transcription, leave it unchanged. "
        "If OCR introduced or split characters and the crop clearly supports the exact "
        "printed text, return a text_override with the exact visible transcription. "
        "Return a separate override for every affected field, including both signal_name "
        "and description when they repeat the same visible text. Such an override may "
        "correct OCR characters, word boundaries, punctuation, or spacing, but must "
        "preserve the original language and must not translate, paraphrase, expand "
        "abbreviations, infer hidden meaning, or invent technical terms. If an extractor "
        "row has text_reconstruction_confidence below 0.90, return an override for every "
        "non-empty free-text field even when the approved transcription is unchanged; "
        "this records independent visual confirmation. If the crop or a column meaning "
        "is ambiguous, block rather than guess. Count every reviewed non-placeholder "
        "text row in each region. Return pass only when all visible content and columns "
        "are complete and every published value is visually supported."
    )
    user_text = (
        "Audit this page, bind every physical column semantically, and adjudicate "
        "exact visible text. The full page is followed by one high-resolution crop "
        "for every region. Return exactly one column-binding decision for each "
        "physical source column, and return text overrides only where a candidate "
        "needs correction.\n\n"
        + json.dumps(
            {
                "page_id": page["id"],
                "pdf_page_number": page["pdf_page_number"],
                "sheet_code_original": page.get("sheet_code"),
                "sheet_title_original": page.get("sheet_title"),
                "detector": detector,
                "proposal_region_ids": [p["region_id"] for p in proposals],
                "column_binding_proposals": column_binding_proposals,
                "extractions": extraction_summary,
            },
            ensure_ascii=False,
        )
    )
    content: list[dict] = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "FULL PAGE IMAGE"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(page_image),
                "detail": "original",
            },
        },
    ]
    for proposal in proposals:
        rid = proposal["region_id"]
        crop = region_images.get(rid)
        if not crop:
            continue
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"HIGH-RESOLUTION REGION CROP {rid}; red labels identify "
                        "deterministic row IDs."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url_png(crop),
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


def _add_usage(totals: dict, task: str, usage: dict, reused: bool) -> None:
    totals["calls"] += 1
    totals["reused_calls"] += 1 if reused else 0
    totals["task_call_counts"][task] = int(
        totals["task_call_counts"].get(task, 0)
    ) + 1
    if not reused:
        totals["new_input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["new_output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["new_reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
        totals["new_cost_usd"] += float(usage.get("cost_usd") or 0.0)


def _normalize_issue(raw: dict, *, source: str, page: dict, sequence: int) -> dict:
    severity = str(raw.get("severity") or "warning").strip().lower()
    if severity not in SEVERITIES:
        severity = "warning"
    issue_type = _canonical_key(raw.get("issue_type") or "unspecified")
    region_id = _clean_text(raw.get("region_id"), 120)
    row_ids = [
        _clean_text(x, 50)
        for x in (raw.get("row_ids") or [])
        if _clean_text(x, 50)
    ]
    return {
        "issue_key": (
            f"structured-v2:{int(page['pdf_page_number'])}:"
            f"{_canonical_key(source)}:{issue_type}:{sequence}"
        ),
        "issue_type": f"structured_v2_{issue_type}",
        "severity": severity,
        "message": _clean_text(raw.get("message") or issue_type, 1600),
        "candidates_json": [],
        "properties": {
            "phase": PHASE_NAME,
            "source_stage": source,
            "pdf_page_number": int(page["pdf_page_number"]),
            "page_type": page.get("page_type"),
            "region_id": region_id,
            "row_ids": row_ids,
            "confidence": _clamp_conf(raw.get("confidence")),
        },
    }


def _local_issue(
    *,
    page: dict,
    code: str,
    message: str,
    severity: str = "high",
    region_id: str = "",
    row_ids: Optional[list[str]] = None,
    sequence: int = 1,
    properties: Optional[dict] = None,
) -> dict:
    severity = severity if severity in SEVERITIES else "high"
    return {
        "issue_key": (
            f"structured-v2:{int(page['pdf_page_number'])}:"
            f"deterministic:{_canonical_key(code)}:{sequence}"
        ),
        "issue_type": f"structured_v2_{_canonical_key(code)}",
        "severity": severity,
        "message": _clean_text(message, 1600),
        "candidates_json": [],
        "properties": {
            "phase": PHASE_NAME,
            "source_stage": "deterministic_validator",
            "pdf_page_number": int(page["pdf_page_number"]),
            "page_type": page.get("page_type"),
            "region_id": region_id,
            "row_ids": row_ids or [],
            **(properties or {}),
        },
    }


def _assessment_map(detector: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in detector.get("proposal_assessments") or []:
        key = _clean_text(item.get("region_id"), 120)
        if key and key not in out:
            out[key] = item
    return out


def _reference_signature(value: Any) -> str:
    """Normalize a short technical reference without changing its characters."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
        .replace("−", "-")
        .upper()
    )
    return re.sub(r"\s+", "", text)


def _safe_bound_reference_cell_value(value: Any, role: str) -> str:
    """Return an exact atomic cell value safe for reference-field publication.

    The semantic role comes from the visual verifier. This local guard only
    prevents a malformed table boundary from copying an entire prose cell into
    a short technical-reference field.
    """
    max_len = int(REFERENCE_ROLE_MAX_LENGTH.get(role, 200))
    text = _clean_text(value, max_len)
    if not text:
        return ""
    if len(text) > min(max_len, 120):
        return ""
    tokens = [token for token in text.split(" ") if token]
    if len(tokens) > 6:
        return ""
    alnum = sum(ch.isalnum() for ch in text)
    if alnum == 0:
        return ""
    # A no-digit reference can still be valid (A, B, L+, PE, etc.), but a long
    # alphabetic phrase is not an atomic terminal/channel/address reference.
    if not any(ch.isdigit() for ch in text):
        letters = sum(ch.isalpha() for ch in text)
        if letters > 24:
            return ""
    return text


def _build_column_binding_proposals(
    *,
    proposals: list[dict],
    detector: dict,
    extractions: list[dict],
) -> list[dict]:
    """Build language-independent column-role evidence for the visual verifier.

    No header vocabulary is hardcoded. Evidence comes from:
      * physical column index and exact header/body cell text;
      * exact agreement between body cells and already extracted canonical
        reference fields;
      * full-page peer tables, which the verifier sees together.
    """
    assessments = _assessment_map(detector)
    extraction_by_region = {
        _clean_text(item.get("region_id"), 120): item
        for item in extractions
        if _clean_text(item.get("region_id"), 120)
    }
    output: list[dict] = []

    for proposal in proposals:
        rid = _clean_text(proposal.get("region_id"), 120)
        assessment = assessments.get(rid) or {}
        extraction = extraction_by_region.get(rid) or {}
        header_count = max(
            0,
            min(
                int(assessment.get("visible_header_row_count") or 0),
                len(proposal.get("row_candidates") or []),
            ),
        )
        column_count = int(proposal.get("deterministic_column_count") or 0)
        row_result_map = {
            _clean_text(row.get("row_id"), 50): row
            for row in (extraction.get("row_results") or [])
            if _clean_text(row.get("row_id"), 50)
        }
        row_candidates = proposal.get("row_candidates") or []
        body_candidates = row_candidates[header_count:]
        columns: list[dict] = []

        for column_index in range(column_count):
            header_cells: list[str] = []
            for row in row_candidates[:header_count]:
                cells = row.get("deterministic_cell_text_original") or []
                value = _clean_text(
                    cells[column_index] if column_index < len(cells) else "",
                    800,
                )
                if value and value not in header_cells:
                    header_cells.append(value)

            body_cell_values: list[dict] = []
            for row in body_candidates:
                row_id = _clean_text(row.get("row_id"), 50)
                cells = row.get("deterministic_cell_text_original") or []
                cell_text = _clean_text(
                    cells[column_index] if column_index < len(cells) else "",
                    1200,
                )
                body_cell_values.append(
                    {
                        "row_id": row_id,
                        "cell_text_original": cell_text,
                    }
                )

            role_evidence: dict[str, dict] = {}
            suggested_roles: list[str] = []
            for role, field_name in REFERENCE_ROLE_TO_RESULT_FIELD.items():
                visible_nonempty = 0
                extractor_nonempty = 0
                comparable = 0
                exact_matches = 0
                for body in body_cell_values:
                    row_id = body["row_id"]
                    cell_text = body["cell_text_original"]
                    extracted = _clean_text(
                        (row_result_map.get(row_id) or {}).get(field_name),
                        int(REFERENCE_ROLE_MAX_LENGTH.get(role, 200)),
                    )
                    if cell_text:
                        visible_nonempty += 1
                    if extracted:
                        extractor_nonempty += 1
                    if cell_text and extracted:
                        comparable += 1
                        if _reference_signature(cell_text) == _reference_signature(
                            extracted
                        ):
                            exact_matches += 1

                visible_coverage = (
                    round(exact_matches / visible_nonempty, 4)
                    if visible_nonempty
                    else 0.0
                )
                comparable_accuracy = (
                    round(exact_matches / comparable, 4)
                    if comparable
                    else 0.0
                )
                role_evidence[role] = {
                    "visible_nonempty_count": visible_nonempty,
                    "extractor_nonempty_count": extractor_nonempty,
                    "comparable_count": comparable,
                    "exact_match_count": exact_matches,
                    "visible_coverage_ratio": visible_coverage,
                    "comparable_accuracy_ratio": comparable_accuracy,
                }
                minimum_support = min(3, visible_nonempty)
                if (
                    visible_nonempty > 0
                    and exact_matches >= minimum_support
                    and visible_coverage >= 0.80
                    and comparable_accuracy >= 0.90
                ):
                    suggested_roles.append(role)

            nonempty_values = [
                item["cell_text_original"]
                for item in body_cell_values
                if item["cell_text_original"]
            ]
            atomic_values = [
                value
                for value in nonempty_values
                if any(
                    _safe_bound_reference_cell_value(value, role)
                    for role in REFERENCE_COLUMN_ROLES
                )
            ]
            integer_values: list[int] = []
            for value in nonempty_values:
                compact = _reference_signature(value)
                if compact.isdigit():
                    integer_values.append(int(compact))
            is_monotonic_integer_sequence = bool(integer_values) and integer_values == sorted(
                integer_values
            )

            columns.append(
                {
                    "source_column_index": column_index,
                    "header_cell_texts_original": header_cells,
                    "body_cell_values": body_cell_values,
                    "body_nonempty_count": len(nonempty_values),
                    "body_distinct_count": len(set(nonempty_values)),
                    "atomic_reference_value_ratio": (
                        round(len(atomic_values) / len(nonempty_values), 4)
                        if nonempty_values
                        else 0.0
                    ),
                    "is_monotonic_integer_sequence": (
                        is_monotonic_integer_sequence
                    ),
                    "local_reference_role_evidence": role_evidence,
                    "locally_suggested_roles": sorted(set(suggested_roles)),
                }
            )

        output.append(
            {
                "region_id": rid,
                "visible_column_count": column_count,
                "columns": columns,
            }
        )

    # Add corroborating, language-independent peer evidence. This does not
    # publish or copy values by itself; it helps the visual verifier recognize
    # equivalent columns across independently bordered tables on the same page.
    for region in output:
        for column in region.get("columns") or []:
            header_cells = column.get("header_cell_texts_original") or []
            own_header = _clean_text(
                header_cells[-1] if header_cells else "",
                800,
            )
            own_signature = _semantic_character_signature(own_header)
            peer_evidence: list[dict] = []
            consensus_roles = set(column.get("locally_suggested_roles") or [])

            for peer_region in output:
                if peer_region.get("region_id") == region.get("region_id"):
                    continue
                if int(peer_region.get("visible_column_count") or 0) != int(
                    region.get("visible_column_count") or 0
                ):
                    continue
                peer_column = next(
                    (
                        item
                        for item in (peer_region.get("columns") or [])
                        if int(item.get("source_column_index", -1))
                        == int(column.get("source_column_index", -2))
                    ),
                    None,
                )
                if not peer_column:
                    continue
                peer_headers = (
                    peer_column.get("header_cell_texts_original") or []
                )
                peer_header = _clean_text(
                    peer_headers[-1] if peer_headers else "",
                    800,
                )
                peer_signature = _semantic_character_signature(peer_header)
                if not own_signature or not peer_signature:
                    similarity = 0.0
                else:
                    similarity = difflib.SequenceMatcher(
                        None,
                        own_signature,
                        peer_signature,
                    ).ratio()
                    shorter = min(len(own_signature), len(peer_signature))
                    longer = max(len(own_signature), len(peer_signature))
                    if (
                        shorter > 0
                        and longer > 0
                        and (
                            own_signature in peer_signature
                            or peer_signature in own_signature
                        )
                        and shorter / longer >= 0.55
                    ):
                        similarity = max(similarity, 0.90)

                comparable_shape = (
                    bool(column.get("is_monotonic_integer_sequence"))
                    == bool(peer_column.get("is_monotonic_integer_sequence"))
                    and abs(
                        float(
                            column.get("atomic_reference_value_ratio") or 0.0
                        )
                        - float(
                            peer_column.get(
                                "atomic_reference_value_ratio"
                            )
                            or 0.0
                        )
                    )
                    <= 0.30
                )
                peer_roles = sorted(
                    set(peer_column.get("locally_suggested_roles") or [])
                )
                if similarity >= 0.72 and comparable_shape:
                    consensus_roles.update(peer_roles)
                peer_evidence.append(
                    {
                        "peer_region_id": peer_region.get("region_id"),
                        "peer_source_column_index": peer_column.get(
                            "source_column_index"
                        ),
                        "header_similarity": round(similarity, 4),
                        "comparable_value_shape": comparable_shape,
                        "peer_locally_suggested_roles": peer_roles,
                    }
                )

            column["peer_equivalent_column_evidence"] = peer_evidence
            column["page_consensus_suggested_roles"] = sorted(
                consensus_roles
            )
    return output



def _column_body_signature_sequence(column: dict) -> list[str]:
    """Return exact normalized body-cell signatures in physical row order."""
    return [
        _reference_signature(item.get("cell_text_original"))
        for item in (column.get("body_cell_values") or [])
        if _reference_signature(item.get("cell_text_original"))
    ]


def _adjudicate_omitted_consensus_reference_role(
    *,
    region_id: str,
    source_column_index: int,
    role: str,
    local_decision: dict,
    proposal_column: dict,
    binding_proposal_by_region: dict[str, dict],
    raw_decision_by_key: dict[tuple[str, int], dict],
) -> Optional[dict]:
    """Safely complete a verifier-omitted reference role from page consensus.

    This adjudicator is language-independent and never inspects fixed header
    words. It activates only when all of the following independently agree:

      * the local verifier approved the physical column for missing values;
      * the omitted role is present in deterministic page consensus;
      * the local role is actually missing from the extractor;
      * the column contains atomic short references for essentially every row;
      * one or more geometrically equivalent peer columns have the same role
        approved by the visual verifier at page-level confidence;
      * no equivalent peer or other local column contradicts that role.

    The function does not copy values from a peer table. It only authorizes the
    local physical cells already present in the current region to populate the
    omitted canonical field.
    """
    if role not in REFERENCE_COLUMN_ROLES:
        return None
    if not bool(local_decision.get("use_for_missing_values")):
        return None

    local_confidence = _clamp_conf(local_decision.get("confidence"))
    if local_confidence < PAGE_PASS_MIN_CONFIDENCE:
        return None

    consensus_roles = set(
        proposal_column.get("page_consensus_suggested_roles") or []
    )
    if role not in consensus_roles:
        return None

    local_role_evidence = (
        proposal_column.get("local_reference_role_evidence") or {}
    ).get(role) or {}
    if int(local_role_evidence.get("extractor_nonempty_count") or 0) != 0:
        return None

    body_nonempty_count = int(proposal_column.get("body_nonempty_count") or 0)
    atomic_ratio = float(
        proposal_column.get("atomic_reference_value_ratio") or 0.0
    )
    if body_nonempty_count <= 0 or atomic_ratio < 0.95:
        return None

    # A role already assigned to another physical column in the same region is
    # a real ambiguity and must never be duplicated by consensus.
    for (other_region_id, other_column_index), other_decision in (
        raw_decision_by_key.items()
    ):
        if other_region_id != region_id:
            continue
        if int(other_column_index) == int(source_column_index):
            continue
        if role in set(other_decision.get("canonical_roles") or []):
            return None

    local_sequence = _column_body_signature_sequence(proposal_column)
    supportive_peers: list[dict] = []
    contradictory_peers: list[dict] = []

    for peer in proposal_column.get("peer_equivalent_column_evidence") or []:
        similarity = float(peer.get("header_similarity") or 0.0)
        if similarity < 0.90 or not bool(peer.get("comparable_value_shape")):
            continue
        if role not in set(peer.get("peer_locally_suggested_roles") or []):
            continue

        peer_region_id = _clean_text(peer.get("peer_region_id"), 120)
        try:
            peer_column_index = int(peer.get("peer_source_column_index"))
        except Exception:
            continue

        peer_decision = raw_decision_by_key.get(
            (peer_region_id, peer_column_index)
        )
        if not peer_decision:
            continue

        peer_roles = set(peer_decision.get("canonical_roles") or [])
        peer_confidence = _clamp_conf(peer_decision.get("confidence"))
        peer_use_for_missing = bool(
            peer_decision.get("use_for_missing_values")
        )

        peer_region = binding_proposal_by_region.get(peer_region_id) or {}
        peer_column = next(
            (
                item
                for item in (peer_region.get("columns") or [])
                if int(item.get("source_column_index", -1))
                == peer_column_index
            ),
            None,
        )
        if not peer_column:
            continue

        peer_atomic_ratio = float(
            peer_column.get("atomic_reference_value_ratio") or 0.0
        )
        peer_body_nonempty_count = int(
            peer_column.get("body_nonempty_count") or 0
        )
        peer_sequence = _column_body_signature_sequence(peer_column)
        same_exact_sequence = bool(local_sequence) and (
            local_sequence == peer_sequence
        )
        same_monotonic_shape = (
            bool(proposal_column.get("is_monotonic_integer_sequence"))
            and bool(peer_column.get("is_monotonic_integer_sequence"))
            and body_nonempty_count == peer_body_nonempty_count
            and abs(atomic_ratio - peer_atomic_ratio) <= 0.05
        )
        strong_value_shape = same_exact_sequence or same_monotonic_shape

        evidence = {
            "peer_region_id": peer_region_id,
            "peer_source_column_index": peer_column_index,
            "header_similarity": round(similarity, 4),
            "peer_binding_confidence": peer_confidence,
            "same_exact_value_sequence": same_exact_sequence,
            "same_monotonic_value_shape": same_monotonic_shape,
            "peer_roles": sorted(peer_roles),
        }

        if (
            role in peer_roles
            and peer_use_for_missing
            and peer_confidence >= PAGE_PASS_MIN_CONFIDENCE
            and peer_atomic_ratio >= 0.95
            and strong_value_shape
        ):
            supportive_peers.append(evidence)
        else:
            contradictory_peers.append(evidence)

    if not supportive_peers or contradictory_peers:
        return None

    adjudication_confidence = min(
        [local_confidence]
        + [
            float(item.get("peer_binding_confidence") or 0.0)
            for item in supportive_peers
        ]
        + [
            float(item.get("header_similarity") or 0.0)
            for item in supportive_peers
        ]
    )
    if adjudication_confidence < PAGE_PASS_MIN_CONFIDENCE:
        return None

    return {
        "version": COLUMN_BINDING_ADJUDICATOR_VERSION,
        "canonical_role": role,
        "region_id": region_id,
        "source_column_index": int(source_column_index),
        "confidence": round(adjudication_confidence, 4),
        "local_verifier_roles": sorted(
            set(local_decision.get("canonical_roles") or [])
        ),
        "local_verifier_confidence": local_confidence,
        "local_body_nonempty_count": body_nonempty_count,
        "local_atomic_reference_value_ratio": round(atomic_ratio, 4),
        "local_extractor_nonempty_count": int(
            local_role_evidence.get("extractor_nonempty_count") or 0
        ),
        "supporting_peers": supportive_peers,
        "contradictory_peers": contradictory_peers,
        "reason": (
            "The local physical column is independently supported by exact "
            "geometry, atomic row values, a page-consensus role, and one or "
            "more verifier-approved equivalent peer columns. Values remain "
            "local to this region; no peer value is copied."
        ),
    }


def _validate_and_materialize(
    *,
    page: dict,
    word_map: dict[int, dict],
    proposals: list[dict],
    detector: dict,
    extractions: list[dict],
    verifier: dict,
    column_binding_proposals: list[dict],
    fingerprints: dict,
) -> tuple[bool, list[dict], list[dict], dict]:
    issues: list[dict] = []
    sequence = 0

    for source, response in [
        ("detector", detector),
        ("verifier", verifier),
    ]:
        for raw in response.get("issues") or []:
            sequence += 1
            issues.append(_normalize_issue(raw, source=source, page=page, sequence=sequence))
    for extraction in extractions:
        for raw in extraction.get("issues") or []:
            sequence += 1
            issues.append(
                _normalize_issue(raw, source="extractor", page=page, sequence=sequence)
            )

    def fail(code: str, message: str, region_id: str = "", row_ids: Optional[list[str]] = None, **props: Any) -> None:
        nonlocal sequence
        sequence += 1
        issues.append(
            _local_issue(
                page=page,
                code=code,
                message=message,
                severity="high",
                region_id=region_id,
                row_ids=row_ids,
                sequence=sequence,
                properties=props,
            )
        )

    def note(code: str, message: str, region_id: str = "", row_ids: Optional[list[str]] = None, **props: Any) -> None:
        nonlocal sequence
        sequence += 1
        issues.append(
            _local_issue(
                page=page,
                code=code,
                message=message,
                severity="info",
                region_id=region_id,
                row_ids=row_ids or [],
                sequence=sequence,
                properties=props,
            )
        )

    text_override_map: dict[tuple[str, str, str], dict] = {}
    used_text_override_keys: set[tuple[str, str, str]] = set()
    for item in verifier.get("text_overrides") or []:
        rid = _clean_text(item.get("region_id"), 120)
        row_id = _clean_text(item.get("row_id"), 50)
        field_name = _clean_text(item.get("field_name"), 80)
        key = (rid, row_id, field_name)
        if not rid or not row_id or field_name not in VISUAL_TEXT_FIELDS:
            fail(
                "invalid_verifier_text_override",
                "Verifier returned an invalid text override key",
                region_id=rid,
                row_ids=[row_id] if row_id else [],
                field_name=field_name,
            )
            continue
        if key in text_override_map:
            fail(
                "duplicate_verifier_text_override",
                "Verifier returned the same text override more than once",
                region_id=rid,
                row_ids=[row_id],
                field_name=field_name,
            )
            continue
        text_override_map[key] = item

    blocked_text_keys: set[tuple[str, str, str]] = set()
    for item in verifier.get("blocked_text_rows") or []:
        rid = _clean_text(item.get("region_id"), 120)
        row_id = _clean_text(item.get("row_id"), 50)
        field_name = _clean_text(item.get("field_name"), 80)
        if field_name not in VISUAL_TEXT_FIELDS | {"all_free_text"}:
            field_name = "all_free_text"
        blocked_text_keys.add((rid, row_id, field_name))
        fail(
            "verifier_blocked_text_row",
            f"Verifier could not safely transcribe {field_name} for region {rid} row {row_id}",
            region_id=rid,
            row_ids=[row_id] if row_id else [],
            field_name=field_name,
            verifier_reason=_clean_text(item.get("reason"), 1000),
            verifier_confidence=_clamp_conf(item.get("confidence")),
        )

    if int(detector.get("page_id") or 0) != int(page["id"]):
        fail("detector_page_id_mismatch", "Detector returned a different page_id")
    if not bool(detector.get("all_visible_io_tables_accounted_for")):
        fail("detector_missing_visible_table", "Detector reports unaccounted visible I/O tables")
    if detector.get("missing_visible_io_tables"):
        fail(
            "detector_missing_region",
            "Detector listed visible I/O table regions without deterministic proposals",
            missing_regions=detector.get("missing_visible_io_tables"),
        )

    assessments = _assessment_map(detector)
    proposal_ids = [p["region_id"] for p in proposals]
    if set(assessments) != set(proposal_ids):
        fail(
            "detector_region_accounting_mismatch",
            "Detector did not return exactly one assessment for every deterministic region",
            expected=proposal_ids,
            returned=sorted(assessments),
        )

    extraction_by_region: dict[str, dict] = {}
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        if rid in extraction_by_region:
            fail("duplicate_extraction_region", f"Region {rid} was extracted more than once", region_id=rid)
        elif rid:
            extraction_by_region[rid] = extraction
    if set(extraction_by_region) != set(proposal_ids):
        fail(
            "extractor_region_accounting_mismatch",
            "Extractor results do not exactly match deterministic regions",
            expected=proposal_ids,
            returned=sorted(extraction_by_region),
        )

    binding_proposal_by_region = {
        _clean_text(item.get("region_id"), 120): item
        for item in column_binding_proposals
        if _clean_text(item.get("region_id"), 120)
    }
    if set(binding_proposal_by_region) != set(proposal_ids):
        fail(
            "column_binding_proposal_accounting_mismatch",
            "Column-binding proposals do not exactly match deterministic regions",
            expected=proposal_ids,
            returned=sorted(binding_proposal_by_region),
        )

    expected_column_keys: set[tuple[str, int]] = set()
    for proposal in proposals:
        rid = proposal["region_id"]
        for column_index in range(
            int(proposal.get("deterministic_column_count") or 0)
        ):
            expected_column_keys.add((rid, column_index))

    raw_decision_by_key: dict[tuple[str, int], dict] = {}
    for raw_decision in verifier.get("column_binding_decisions") or []:
        raw_region_id = _clean_text(raw_decision.get("region_id"), 120)
        try:
            raw_column_index = int(raw_decision.get("source_column_index"))
        except Exception:
            raw_column_index = -1
        raw_decision_by_key[(raw_region_id, raw_column_index)] = raw_decision

    column_binding_map: dict[str, dict[int, dict]] = {}
    decision_keys: set[tuple[str, int]] = set()
    for decision in verifier.get("column_binding_decisions") or []:
        rid = _clean_text(decision.get("region_id"), 120)
        try:
            column_index = int(decision.get("source_column_index"))
        except Exception:
            column_index = -1
        key = (rid, column_index)
        if key in decision_keys:
            fail(
                "duplicate_column_binding_decision",
                "Verifier returned the same column-binding decision more than once",
                region_id=rid,
                source_column_index=column_index,
            )
            continue
        decision_keys.add(key)

        roles: list[str] = []
        for raw_role in decision.get("canonical_roles") or []:
            role = _clean_text(raw_role, 80)
            if role in COLUMN_BINDING_ROLES and role not in roles:
                roles.append(role)
        if not rid or key not in expected_column_keys:
            fail(
                "invalid_column_binding_key",
                "Verifier returned a column-binding decision outside the deterministic table geometry",
                region_id=rid,
                source_column_index=column_index,
            )
            continue
        if not roles:
            fail(
                "empty_column_binding_roles",
                "Verifier returned a source column without a canonical role",
                region_id=rid,
                source_column_index=column_index,
            )

        proposal_column = next(
            (
                item
                for item in (
                    (binding_proposal_by_region.get(rid) or {}).get(
                        "columns"
                    )
                    or []
                )
                if int(item.get("source_column_index", -1))
                == column_index
            ),
            None,
        )
        verifier_returned_roles = list(roles)
        consensus_reference_roles = set(
            (proposal_column or {}).get(
                "page_consensus_suggested_roles"
            )
            or []
        ).intersection(REFERENCE_COLUMN_ROLES)
        omitted_consensus_roles = sorted(
            consensus_reference_roles.difference(roles)
        )
        consensus_added_roles: list[str] = []
        consensus_adjudication_evidence: dict[str, dict] = {}
        for omitted_role in omitted_consensus_roles:
            adjudication = _adjudicate_omitted_consensus_reference_role(
                region_id=rid,
                source_column_index=column_index,
                role=omitted_role,
                local_decision=decision,
                proposal_column=proposal_column or {},
                binding_proposal_by_region=binding_proposal_by_region,
                raw_decision_by_key=raw_decision_by_key,
            )
            if adjudication:
                roles.append(omitted_role)
                consensus_added_roles.append(omitted_role)
                consensus_adjudication_evidence[omitted_role] = adjudication
                note(
                    "consensus_reference_role_adjudicated",
                    "A verifier-omitted reference role was safely completed "
                    "from independent peer-equivalent column consensus",
                    region_id=rid,
                    source_column_index=column_index,
                    canonical_role=omitted_role,
                    adjudication=adjudication,
                )
            else:
                fail(
                    "verifier_omitted_consensus_reference_role",
                    "Verifier omitted a strongly corroborated reference-column role",
                    region_id=rid,
                    source_column_index=column_index,
                    omitted_roles=[omitted_role],
                    returned_roles=roles,
                    peer_evidence=(proposal_column or {}).get(
                        "peer_equivalent_column_evidence"
                    )
                    or [],
                )

        if "other_data" in roles and len(roles) > 1:
            fail(
                "other_data_column_binding_conflict",
                "other_data cannot be combined with canonical data roles",
                region_id=rid,
                source_column_index=column_index,
                canonical_roles=roles,
            )
        confidence = _clamp_conf(decision.get("confidence"))
        if confidence < ROW_MIN_CONFIDENCE:
            fail(
                "column_binding_confidence_below_threshold",
                "Column semantic binding confidence is below the minimum threshold",
                region_id=rid,
                source_column_index=column_index,
                binding_confidence=confidence,
                threshold=ROW_MIN_CONFIDENCE,
            )
        use_for_missing = bool(decision.get("use_for_missing_values"))
        if use_for_missing and not set(roles).intersection(
            REFERENCE_COLUMN_ROLES
        ):
            fail(
                "column_binding_fill_without_reference_role",
                "A column can fill missing values only when it has a canonical reference role",
                region_id=rid,
                source_column_index=column_index,
                canonical_roles=roles,
            )
        if use_for_missing and confidence < PAGE_PASS_MIN_CONFIDENCE:
            fail(
                "column_binding_fill_confidence_below_threshold",
                "A column used to fill missing references requires page-level confidence",
                region_id=rid,
                source_column_index=column_index,
                binding_confidence=confidence,
                threshold=PAGE_PASS_MIN_CONFIDENCE,
            )

        role_sources = {
            role: (
                "deterministic_peer_consensus"
                if role in consensus_added_roles
                else "visual_verifier"
            )
            for role in roles
        }
        normalized_decision = {
            "region_id": rid,
            "source_column_index": column_index,
            "header_text_original": _clean_text(
                decision.get("header_text_original"), 1000
            ),
            "canonical_roles": roles,
            "verifier_returned_roles": verifier_returned_roles,
            "consensus_added_roles": consensus_added_roles,
            "role_sources": role_sources,
            "consensus_adjudicator_version": (
                COLUMN_BINDING_ADJUDICATOR_VERSION
                if consensus_added_roles
                else ""
            ),
            "consensus_adjudication_evidence": (
                consensus_adjudication_evidence
            ),
            "use_for_missing_values": use_for_missing,
            "confidence": confidence,
            "reason": _clean_text(decision.get("reason"), 1200),
        }
        column_binding_map.setdefault(rid, {})[
            column_index
        ] = normalized_decision

    if decision_keys != expected_column_keys:
        fail(
            "column_binding_decision_accounting_mismatch",
            "Verifier did not return exactly one semantic decision for every physical source column",
            expected=[list(item) for item in sorted(expected_column_keys)],
            returned=[list(item) for item in sorted(decision_keys)],
        )

    io_rows: list[dict] = []
    region_stats: list[dict] = []
    region_text_row_counts: dict[str, int] = {}
    for proposal in proposals:
        rid = proposal["region_id"]
        assessment = assessments.get(rid) or {}
        extraction = extraction_by_region.get(rid) or {}
        if not assessment:
            continue
        if not bool(assessment.get("visible")) or not bool(assessment.get("distinct_table")):
            fail("region_not_confirmed_visible", f"Region {rid} was not confirmed as a distinct visible table", region_id=rid)
        if str(assessment.get("kind") or "") != "io_table":
            fail("region_not_io_table", f"Region {rid} was not confirmed as an I/O table", region_id=rid)

        deterministic_total = len(proposal["row_candidates"])
        header_count = int(assessment.get("visible_header_row_count") or 0)
        physical_count = int(assessment.get("visible_physical_row_count") or 0)
        if header_count < 1 or header_count > 4 or physical_count <= 0:
            fail(
                "invalid_visual_row_counts",
                f"Region {rid} returned implausible header/body row counts",
                region_id=rid,
                header_count=header_count,
                physical_count=physical_count,
            )
        if header_count + physical_count != deterministic_total:
            fail(
                "row_count_accounting_mismatch",
                f"Region {rid}: visual header + physical rows do not equal deterministic rows",
                region_id=rid,
                deterministic_total=deterministic_total,
                header_count=header_count,
                physical_count=physical_count,
            )

        if int(extraction.get("page_id") or 0) != int(page["id"]):
            fail("extractor_page_id_mismatch", f"Region {rid} returned a different page_id", region_id=rid)
        if int(extraction.get("visible_header_row_count") or -1) != header_count:
            fail("extractor_header_count_mismatch", f"Region {rid} extractor/header count differs from detector", region_id=rid)
        if int(extraction.get("visible_physical_row_count") or -1) != physical_count:
            fail("extractor_physical_count_mismatch", f"Region {rid} extractor/body count differs from detector", region_id=rid)

        candidate_map = {row["row_id"]: row for row in proposal["row_candidates"]}
        row_results = extraction.get("row_results") or []
        result_ids = [_clean_text(row.get("row_id"), 50) for row in row_results]
        if len(result_ids) != len(set(result_ids)):
            fail("duplicate_row_result", f"Region {rid} contains duplicate row IDs", region_id=rid, row_ids=result_ids)
        if set(result_ids) != set(candidate_map):
            fail(
                "row_result_accounting_mismatch",
                f"Region {rid} did not return exactly one result for every deterministic row",
                region_id=rid,
                expected=sorted(candidate_map),
                returned=sorted(set(result_ids)),
            )
        if extraction.get("unaccounted_row_ids") or extraction.get("duplicate_row_ids"):
            fail(
                "extractor_declared_unaccounted_rows",
                f"Region {rid} declared unaccounted or duplicate rows",
                region_id=rid,
                unaccounted=extraction.get("unaccounted_row_ids"),
                duplicates=extraction.get("duplicate_row_ids"),
            )

        module_tag = _clean_text(extraction.get("module_tag_original"), 200)
        module_model_candidate = _clean_text(
            extraction.get("module_model_original"), 240
        )
        table_label_original = _clean_text(
            extraction.get("table_label_original"), 500
        )
        if not module_tag:
            fail("missing_module_tag", f"Region {rid} has no visually supported module tag", region_id=rid)
        header_ids = [
            int(x)
            for x in (extraction.get("module_header_source_word_ids") or [])
            if str(x).isdigit() and int(x) in word_map
        ]
        header_allowed_ids: set[int] = set()
        for row in proposal["row_candidates"][: max(1, header_count)]:
            header_allowed_ids.update(row.get("word_ids") or [])
        if not header_ids or not set(header_ids).issubset(header_allowed_ids):
            fail(
                "invalid_module_header_evidence",
                f"Region {rid} module header evidence is missing or outside header rows",
                region_id=rid,
                header_ids=header_ids,
            )
        header_text = _text_for_ids(header_ids, word_map, 1200)
        normalized_tag = re.sub(r"[^a-z0-9]", "", module_tag.lower())
        normalized_header = re.sub(r"[^a-z0-9]", "", header_text.lower())
        if normalized_tag and normalized_tag not in normalized_header:
            fail(
                "module_tag_not_in_header_evidence",
                f"Region {rid} module tag is not supported by its selected header words",
                region_id=rid,
                module_tag=module_tag,
                header_text=header_text,
            )

        # Publish a hardware model only when the extractor explicitly selected
        # a strong code-like candidate from the header. The full header remains
        # audit evidence, but generic labels, electrical ratings, and functional
        # descriptions are never promoted to module_model.
        module_model, module_model_evidence_quality = (
            _canonical_hardware_model_code(
                module_model_candidate,
                module_tag=module_tag,
                sheet_code=str(page.get("sheet_code") or ""),
            )
        )

        region_column_bindings = column_binding_map.get(rid) or {}
        expected_region_column_count = int(
            proposal.get("deterministic_column_count") or 0
        )
        if len(region_column_bindings) != expected_region_column_count:
            fail(
                "region_column_binding_count_mismatch",
                f"Region {rid} does not have one semantic binding per physical column",
                region_id=rid,
                expected_column_count=expected_region_column_count,
                returned_column_count=len(region_column_bindings),
            )
        region_reference_fill_count = 0
        region_binding_fingerprint = _sha256_json(
            [
                region_column_bindings[index]
                for index in sorted(region_column_bindings)
            ]
        )

        included_rows = [row for row in row_results if bool(row.get("include_in_io"))]
        if len(included_rows) != physical_count:
            fail(
                "physical_row_materialization_mismatch",
                f"Region {rid} included {len(included_rows)} rows but {physical_count} physical rows are visible",
                region_id=rid,
                included=len(included_rows),
                visible=physical_count,
            )

        table_conf = _clamp_conf(extraction.get("confidence"))
        for result in row_results:
            row_id = _clean_text(result.get("row_id"), 50)
            candidate = candidate_map.get(row_id)
            if not candidate:
                continue
            role = str(result.get("row_role") or "other_data")
            if role not in ROW_ROLES:
                role = "other_data"
            include = bool(result.get("include_in_io"))
            if not include:
                continue
            if role in {"title", "column_header"}:
                fail(
                    "header_included_as_io",
                    f"Region {rid} row {row_id} is marked as a header but included as I/O",
                    region_id=rid,
                    row_ids=[row_id],
                )
                continue

            row_conf = _clamp_conf(result.get("confidence"))
            if min(table_conf, row_conf) < ROW_MIN_CONFIDENCE:
                fail(
                    "row_confidence_below_threshold",
                    f"Region {rid} row {row_id} confidence is below the publication threshold",
                    region_id=rid,
                    row_ids=[row_id],
                    row_confidence=row_conf,
                    table_confidence=table_conf,
                    threshold=ROW_MIN_CONFIDENCE,
                )

            candidate_ids = set(candidate.get("word_ids") or [])
            source_ids = []
            for raw_id in result.get("source_word_ids") or []:
                try:
                    wid = int(raw_id)
                except Exception:
                    continue
                if wid in word_map and wid not in source_ids:
                    source_ids.append(wid)
            if source_ids and not set(source_ids).issubset(candidate_ids):
                fail(
                    "row_evidence_outside_candidate",
                    f"Region {rid} row {row_id} cites words outside its deterministic row band",
                    region_id=rid,
                    row_ids=[row_id],
                    source_word_ids=source_ids,
                    candidate_word_ids=sorted(candidate_ids),
                )
            if not source_ids and candidate_ids and role != "blank_unused":
                fail(
                    "missing_row_evidence",
                    f"Region {rid} row {row_id} has visible words but no source evidence",
                    region_id=rid,
                    row_ids=[row_id],
                )

            fallback_bbox = _rect_from(candidate["bbox_pt"])
            evidence_bbox = _bbox_for_ids(source_ids, word_map, fallback_bbox)
            source_text = _text_for_ids(source_ids, word_map, 5000)
            is_placeholder = bool(result.get("is_placeholder")) or role in {
                "placeholder",
                "blank_unused",
            }

            extractor_text_confidence = _clamp_conf(
                result.get("text_reconstruction_confidence")
            )
            original_text_values = {
                "signal_name": _clean_text(
                    result.get("signal_name_original"), 700
                ),
                "description": _clean_text(
                    result.get("description_original"), 1600
                ),
                "expected_normal_state": _clean_text(
                    result.get("expected_normal_state_original"), 500
                ),
            }
            extractor_candidates = {
                "signal_name": _clean_text(
                    result.get("signal_name_normalized"), 700
                ) or original_text_values["signal_name"],
                "description": _clean_text(
                    result.get("description_normalized"), 1600
                ) or original_text_values["description"],
                "expected_normal_state": _clean_text(
                    result.get("expected_normal_state_normalized"), 500
                ) or original_text_values["expected_normal_state"],
            }
            max_lengths = {
                "signal_name": 700,
                "description": 1600,
                "expected_normal_state": 500,
            }
            approved_text_values: dict[str, str] = {}
            approved_text_sources: dict[str, str] = {}
            approved_text_confidences: dict[str, float] = {}
            verifier_override_properties: dict[str, dict] = {}

            for field_name in sorted(VISUAL_TEXT_FIELDS):
                if (
                    (rid, row_id, "all_free_text") in blocked_text_keys
                    or (rid, row_id, field_name) in blocked_text_keys
                ):
                    approved_text_values[field_name] = ""
                    approved_text_sources[field_name] = "blocked"
                    approved_text_confidences[field_name] = 0.0
                    continue

                key = (rid, row_id, field_name)
                override = text_override_map.get(key)
                if override is not None:
                    approved = _clean_text(
                        override.get("approved_text"),
                        max_lengths[field_name],
                    )
                    confidence = _clamp_conf(override.get("confidence"))
                    if not approved and (
                        original_text_values[field_name]
                        or extractor_candidates[field_name]
                    ):
                        fail(
                            "empty_verifier_text_override",
                            f"Verifier returned an empty approved {field_name}",
                            region_id=rid,
                            row_ids=[row_id],
                            field_name=field_name,
                        )
                    if confidence < TEXT_RECONSTRUCTION_MIN_CONFIDENCE:
                        fail(
                            "verifier_text_override_confidence_below_threshold",
                            f"Verifier override confidence is too low for {field_name}",
                            region_id=rid,
                            row_ids=[row_id],
                            field_name=field_name,
                            verifier_confidence=confidence,
                            threshold=TEXT_RECONSTRUCTION_MIN_CONFIDENCE,
                        )
                    approved_text_values[field_name] = approved
                    approved_text_sources[field_name] = "visual_verifier_override"
                    approved_text_confidences[field_name] = confidence
                    verifier_override_properties[field_name] = {
                        "approved_text": approved,
                        "confidence": confidence,
                        "reason": _clean_text(override.get("reason"), 1000),
                    }
                    used_text_override_keys.add(key)
                else:
                    approved_text_values[field_name] = extractor_candidates[
                        field_name
                    ]
                    approved_text_sources[field_name] = "extractor_visual_candidate"
                    approved_text_confidences[field_name] = (
                        extractor_text_confidence
                    )

            has_free_text = any(original_text_values.values()) or any(
                approved_text_values.values()
            )
            if has_free_text and not is_placeholder:
                region_text_row_counts[rid] = int(
                    region_text_row_counts.get(rid, 0)
                ) + 1
                nonempty_confidences = [
                    approved_text_confidences[field_name]
                    for field_name in VISUAL_TEXT_FIELDS
                    if (
                        original_text_values[field_name]
                        or approved_text_values[field_name]
                    )
                ]
                final_text_confidence = min(nonempty_confidences or [0.0])
                if final_text_confidence < TEXT_RECONSTRUCTION_MIN_CONFIDENCE:
                    fail(
                        "visual_text_confidence_below_threshold",
                        f"Region {rid} row {row_id} exact visual text confidence "
                        "is below the publication threshold",
                        region_id=rid,
                        row_ids=[row_id],
                        text_confidence=final_text_confidence,
                        threshold=TEXT_RECONSTRUCTION_MIN_CONFIDENCE,
                    )
            else:
                final_text_confidence = extractor_text_confidence

            normalized_signal_name = approved_text_values["signal_name"]
            normalized_description = approved_text_values["description"]
            normalized_expected_normal_state = approved_text_values[
                "expected_normal_state"
            ]
            original_signal_name = original_text_values["signal_name"]
            original_description = original_text_values["description"]
            original_expected_normal_state = original_text_values[
                "expected_normal_state"
            ]

            extractor_reference_values = {
                "channel_ref": _clean_text(
                    result.get("channel_ref_original"), 160
                ),
                "terminal_reference": _clean_text(
                    result.get("terminal_reference_original"), 300
                ),
                "wire_reference": _clean_text(
                    result.get("wire_reference_original"), 300
                ),
                "plc_address": _clean_text(
                    result.get("plc_address_original"), 200
                ),
                "connector_reference": _clean_text(
                    result.get("connector_ref_original"), 200
                ),
            }
            published_reference_values = dict(extractor_reference_values)
            reference_value_sources = {
                role: ("extractor" if value else "")
                for role, value in extractor_reference_values.items()
            }
            reference_value_column_indexes: dict[str, list[int]] = {
                role: [] for role in REFERENCE_COLUMN_ROLES
            }
            bound_reference_candidates: dict[str, list[tuple[int, str, str]]] = {
                role: [] for role in REFERENCE_COLUMN_ROLES
            }
            deterministic_cells = (
                candidate.get("deterministic_cell_text_original") or []
            )

            for column_index, binding in sorted(
                region_column_bindings.items()
            ):
                if not bool(binding.get("use_for_missing_values")):
                    continue
                cell_text = _clean_text(
                    deterministic_cells[column_index]
                    if column_index < len(deterministic_cells)
                    else "",
                    1200,
                )
                if not cell_text:
                    continue
                for role in binding.get("canonical_roles") or []:
                    if role not in REFERENCE_COLUMN_ROLES:
                        continue
                    safe_value = _safe_bound_reference_cell_value(
                        cell_text,
                        role,
                    )
                    if not safe_value:
                        if not published_reference_values.get(role):
                            fail(
                                "bound_reference_cell_not_atomic",
                                f"Region {rid} row {row_id} column {column_index} "
                                f"cannot safely populate missing {role}",
                                region_id=rid,
                                row_ids=[row_id],
                                source_column_index=column_index,
                                canonical_role=role,
                                cell_text_original=cell_text,
                            )
                        continue
                    role_source = (
                        (binding.get("role_sources") or {}).get(role)
                        or "visual_verifier"
                    )
                    bound_reference_candidates[role].append(
                        (column_index, safe_value, role_source)
                    )

            for role in sorted(REFERENCE_COLUMN_ROLES):
                candidates_for_role = bound_reference_candidates.get(role) or []
                distinct_by_signature: dict[
                    str,
                    tuple[int, str, str],
                ] = {}
                for column_index, value, role_source in candidates_for_role:
                    signature = _reference_signature(value)
                    if signature and signature not in distinct_by_signature:
                        distinct_by_signature[signature] = (
                            column_index,
                            value,
                            role_source,
                        )
                existing = published_reference_values.get(role) or ""

                if existing:
                    if distinct_by_signature:
                        existing_signature = _reference_signature(existing)
                        if existing_signature in distinct_by_signature:
                            matching_sources = {
                                role_source
                                for column_index, value, role_source
                                in candidates_for_role
                                if _reference_signature(value)
                                == existing_signature
                            }
                            if "deterministic_peer_consensus" in matching_sources:
                                reference_value_sources[role] = (
                                    "extractor_verified_by_peer_consensus_column_binding"
                                )
                            else:
                                reference_value_sources[role] = (
                                    "extractor_verified_by_column_binding"
                                )
                            reference_value_column_indexes[role] = sorted(
                                {
                                    column_index
                                    for column_index, value, role_source
                                    in candidates_for_role
                                    if _reference_signature(value)
                                    == existing_signature
                                }
                            )
                        elif len(distinct_by_signature) == 1:
                            column_index, bound_value, role_source = next(
                                iter(distinct_by_signature.values())
                            )
                            fail(
                                "extractor_reference_conflicts_with_bound_column",
                                f"Region {rid} row {row_id} extractor {role} "
                                "conflicts with the approved source column",
                                region_id=rid,
                                row_ids=[row_id],
                                canonical_role=role,
                                extractor_value=existing,
                                bound_cell_value=bound_value,
                                source_column_index=column_index,
                                binding_role_source=role_source,
                            )
                        else:
                            fail(
                                "multiple_bound_reference_values",
                                f"Region {rid} row {row_id} has conflicting "
                                f"column-bound values for {role}",
                                region_id=rid,
                                row_ids=[row_id],
                                canonical_role=role,
                                candidates=[
                                    {
                                        "source_column_index": column_index,
                                        "value": value,
                                        "role_source": role_source,
                                    }
                                    for column_index, value, role_source
                                    in candidates_for_role
                                ],
                            )
                    continue

                if not distinct_by_signature:
                    continue
                if len(distinct_by_signature) > 1:
                    fail(
                        "multiple_bound_reference_values",
                        f"Region {rid} row {row_id} has conflicting "
                        f"column-bound values for missing {role}",
                        region_id=rid,
                        row_ids=[row_id],
                        canonical_role=role,
                        candidates=[
                            {
                                "source_column_index": column_index,
                                "value": value,
                                "role_source": role_source,
                            }
                            for column_index, value, role_source
                            in candidates_for_role
                        ],
                    )
                    continue

                column_index, bound_value, role_source = next(
                    iter(distinct_by_signature.values())
                )
                published_reference_values[role] = bound_value
                reference_value_sources[role] = (
                    "deterministic_peer_consensus_column_binding"
                    if role_source == "deterministic_peer_consensus"
                    else "verifier_approved_column_binding"
                )
                reference_value_column_indexes[role] = [column_index]
                region_reference_fill_count += 1

            io_type = str(extraction.get("io_type") or "other")
            if io_type not in IO_TYPES:
                io_type = "other"
            io_rows.append(
                {
                    "io_key": (
                        f"vision-v2:{int(page['pdf_page_number'])}:"
                        f"{_canonical_key(rid)}:{_canonical_key(row_id)}"
                    ),
                    "module_tag": module_tag,
                    "module_model": module_model,
                    "channel_ref": published_reference_values[
                        "channel_ref"
                    ],
                    "plc_address": published_reference_values[
                        "plc_address"
                    ],
                    "io_type": io_type,
                    "is_safety": bool(extraction.get("is_safety"))
                    or io_type.startswith("safety_"),
                    "signal_name": normalized_signal_name,
                    "description": normalized_description,
                    "expected_normal_state": (
                        normalized_expected_normal_state
                    ),
                    "wire_reference": published_reference_values[
                        "wire_reference"
                    ],
                    "terminal_reference": published_reference_values[
                        "terminal_reference"
                    ],
                    "bbox": evidence_bbox,
                    "source_text": source_text,
                    "confidence": round(
                        min(
                            table_conf or 1.0,
                            row_conf or 1.0,
                            _clamp_conf(verifier.get("confidence")) or 1.0,
                        ),
                        4,
                    ),
                    "properties": {
                        "phase": PHASE_NAME,
                        "pipeline_marker": PIPELINE_MARKER,
                        "materializer_version": MATERIALIZER_VERSION,
                        "region_id": rid,
                        "region_hash": proposal["region_hash"],
                        "row_id": row_id,
                        "row_index": candidate["row_index"],
                        "row_role": role,
                        "is_placeholder": is_placeholder,
                        "connector_ref_original": _clean_text(
                            result.get("connector_ref_original"), 200
                        ),
                        "source_word_ids": source_ids,
                        "row_bbox_pt": candidate["bbox_pt"],
                        "region_bbox_pt": proposal["crop_bbox_pt"],
                        "module_header_source_word_ids": header_ids,
                        "module_header_source_text": header_text,
                        "table_label_original": table_label_original,
                        "module_model_original_candidate": module_model_candidate,
                        "module_model_canonical": module_model,
                        "module_model_evidence_quality": (
                            module_model_evidence_quality
                        ),
                        "column_binding_version": "semantic-column-binding-v2",
                        "column_binding_adjudicator_version": (
                            COLUMN_BINDING_ADJUDICATOR_VERSION
                        ),
                        "column_binding_fingerprint": (
                            region_binding_fingerprint
                        ),
                        "column_binding_decisions": [
                            region_column_bindings[index]
                            for index in sorted(region_column_bindings)
                        ],
                        "published_reference_values": (
                            published_reference_values
                        ),
                        "reference_value_sources": reference_value_sources,
                        "reference_value_column_indexes": (
                            reference_value_column_indexes
                        ),
                        "published_connector_reference": (
                            published_reference_values[
                                "connector_reference"
                            ]
                        ),
                        "text_reconstruction_confidence": (
                            final_text_confidence
                        ),
                        "text_reconstruction_note": _clean_text(
                            result.get("text_reconstruction_note"), 800
                        ),
                        "vector_row_text_original": source_text,
                        "extractor_candidate": extractor_candidates,
                        "published_visual_text": approved_text_values,
                        "visual_text_source": approved_text_sources,
                        "visual_text_confidence": approved_text_confidences,
                        "verifier_text_overrides": (
                            verifier_override_properties
                        ),
                        # Backward-compatible alias used by existing SQL/UI.
                        "normalized": {
                            "signal_name": normalized_signal_name,
                            "description": normalized_description,
                            "expected_normal_state": (
                                normalized_expected_normal_state
                            ),
                        },
                        "detector_fingerprint": fingerprints.get("detector"),
                        "extractor_fingerprint": fingerprints.get(rid),
                        "verifier_fingerprint": fingerprints.get("verifier"),
                        "original": {
                            "channel_ref": _clean_text(
                                result.get("channel_ref_original"), 160
                            ),
                            "plc_address": _clean_text(
                                result.get("plc_address_original"), 200
                            ),
                            "wire_reference": _clean_text(
                                result.get("wire_reference_original"), 300
                            ),
                            "terminal_reference": _clean_text(
                                result.get("terminal_reference_original"), 300
                            ),
                            "signal_name": original_signal_name,
                            "description": original_description,
                            "expected_normal_state": (
                                original_expected_normal_state
                            ),
                        },
                    },
                }
            )

        region_stats.append(
            {
                "region_id": rid,
                "module_tag": module_tag,
                "deterministic_rows": deterministic_total,
                "header_rows": header_count,
                "physical_rows": physical_count,
                "materialized_rows": len(included_rows),
                "reference_values_filled": region_reference_fill_count,
                "column_bindings": [
                    region_column_bindings[index]
                    for index in sorted(region_column_bindings)
                ],
            }
        )

    if int(verifier.get("page_id") or 0) != int(page["id"]):
        fail("verifier_page_id_mismatch", "Verifier returned a different page_id")
    if str(verifier.get("verdict") or "") != "pass":
        fail("verifier_blocked_page", "Independent visual verifier did not pass the page")
    for field, code in [
        ("all_visible_tables_accounted_for", "verifier_missing_table"),
        ("all_visible_physical_rows_accounted_for", "verifier_missing_rows"),
        (
            "all_module_tags_supported_by_own_headers",
            "verifier_module_tag_header_inconsistency",
        ),
        (
            "all_published_text_visually_supported",
            "verifier_text_not_visually_supported",
        ),
        (
            "all_visible_text_resolved_or_blocked",
            "verifier_text_review_incomplete",
        ),
        (
            "all_visible_columns_accounted_for",
            "verifier_column_accounting_incomplete",
        ),
        (
            "all_reference_columns_materialized",
            "verifier_reference_materialization_incomplete",
        ),
    ]:
        if not bool(verifier.get(field)):
            fail(code, f"Independent verifier returned {field}=false")

    if bool(verifier.get("sheet_code_used_as_module_tag_without_visual_support")):
        fail(
            "sheet_code_confused_with_module_tag",
            "Verifier detected a page sheet code used as a module tag",
        )
    if verifier.get("missing_region_ids") or verifier.get("missing_row_ids"):
        fail(
            "verifier_declared_missing_content",
            "Verifier declared missing regions or rows",
            missing_regions=verifier.get("missing_region_ids"),
            missing_rows=verifier.get("missing_row_ids"),
        )
    if verifier.get("duplicate_row_keys"):
        fail(
            "verifier_duplicate_rows",
            "Verifier found duplicate row keys",
            duplicate_row_keys=verifier.get("duplicate_row_keys"),
        )
    if verifier.get("unaccounted_visual_evidence"):
        fail(
            "verifier_unaccounted_visual_evidence",
            "Verifier found unaccounted visual evidence",
            evidence=verifier.get("unaccounted_visual_evidence"),
        )
    if _clamp_conf(verifier.get("confidence")) < PAGE_PASS_MIN_CONFIDENCE:
        fail(
            "verifier_confidence_below_threshold",
            "Verifier confidence is below the page publication threshold",
            verifier_confidence=_clamp_conf(verifier.get("confidence")),
            threshold=PAGE_PASS_MIN_CONFIDENCE,
        )

    unused_override_keys = sorted(set(text_override_map) - used_text_override_keys)
    if unused_override_keys:
        fail(
            "verifier_override_not_applied",
            "Verifier returned text overrides that do not match published rows",
            override_keys=[list(x) for x in unused_override_keys],
        )

    checks = verifier.get("region_checks") or []
    check_map = {
        _clean_text(check.get("region_id"), 120): check
        for check in checks
        if _clean_text(check.get("region_id"), 120)
    }
    if set(check_map) != set(proposal_ids):
        fail(
            "verifier_region_check_mismatch",
            "Verifier did not return exactly one check per region",
            expected=proposal_ids,
            returned=sorted(check_map),
        )
    for rid, check in check_map.items():
        if not bool(check.get("pass")):
            fail("verifier_region_failed", f"Verifier failed region {rid}: {check.get('reason')}", region_id=rid)
        assessment = assessments.get(rid) or {}
        if int(check.get("visible_physical_row_count") or -1) != int(
            assessment.get("visible_physical_row_count") or -2
        ):
            fail("verifier_region_visible_count_mismatch", f"Verifier visible row count differs for {rid}", region_id=rid)
        if int(check.get("accounted_physical_row_count") or -1) != int(
            assessment.get("visible_physical_row_count") or -2
        ):
            fail("verifier_region_accounted_count_mismatch", f"Verifier accounted row count differs for {rid}", region_id=rid)
        expected_text_rows = int(region_text_row_counts.get(rid, 0))
        if int(check.get("visible_text_row_count") or 0) != expected_text_rows:
            fail(
                "verifier_region_text_count_mismatch",
                f"Verifier visible text-row count differs for {rid}",
                region_id=rid,
                expected_text_rows=expected_text_rows,
                verifier_text_rows=int(check.get("visible_text_row_count") or 0),
            )
        if int(check.get("supported_text_row_count") or 0) != expected_text_rows:
            fail(
                "verifier_region_text_support_incomplete",
                f"Verifier did not support every visible text row for {rid}",
                region_id=rid,
                expected_text_rows=expected_text_rows,
                supported_text_rows=int(check.get("supported_text_row_count") or 0),
            )

    binding_checks = verifier.get("column_binding_checks") or []
    binding_check_map = {
        _clean_text(check.get("region_id"), 120): check
        for check in binding_checks
        if _clean_text(check.get("region_id"), 120)
    }
    if set(binding_check_map) != set(proposal_ids):
        fail(
            "verifier_column_binding_check_mismatch",
            "Verifier did not return exactly one column-binding check per region",
            expected=proposal_ids,
            returned=sorted(binding_check_map),
        )
    proposal_by_id = {proposal["region_id"]: proposal for proposal in proposals}
    for rid, check in binding_check_map.items():
        expected_columns = int(
            (proposal_by_id.get(rid) or {}).get(
                "deterministic_column_count"
            )
            or 0
        )
        if not bool(check.get("pass")):
            fail(
                "verifier_column_binding_failed",
                f"Verifier failed semantic column binding for {rid}: "
                f"{check.get('reason')}",
                region_id=rid,
            )
        if int(check.get("visible_column_count") or -1) != expected_columns:
            fail(
                "verifier_visible_column_count_mismatch",
                f"Verifier visible column count differs for {rid}",
                region_id=rid,
                expected_column_count=expected_columns,
                verifier_column_count=int(
                    check.get("visible_column_count") or -1
                ),
            )
        if int(check.get("accounted_column_count") or -1) != expected_columns:
            fail(
                "verifier_accounted_column_count_mismatch",
                f"Verifier did not account for every source column in {rid}",
                region_id=rid,
                expected_column_count=expected_columns,
                accounted_column_count=int(
                    check.get("accounted_column_count") or -1
                ),
            )

    blocking = [i for i in issues if i["severity"] in {"high", "critical"}]
    passed = len(blocking) == 0
    audit = {
        "passed": passed,
        "region_count": len(proposals),
        "physical_rows_expected": sum(x["physical_rows"] for x in region_stats),
        "physical_rows_materialized": len(io_rows),
        "placeholder_rows": sum(
            1 for row in io_rows if bool(row["properties"].get("is_placeholder"))
        ),
        "region_stats": region_stats,
        "blocking_issue_count": len(blocking),
        "warning_issue_count": sum(1 for i in issues if i["severity"] == "warning"),
        "info_issue_count": sum(1 for i in issues if i["severity"] == "info"),
        "duplicate_source_module_tags": verifier.get("duplicate_source_module_tags") or [],
    }
    return passed, io_rows, issues, audit


def _insert_issue(cur: Any, context: dict, page: dict, issue: dict) -> None:
    cur.execute(
        """
        INSERT INTO public.electrical_review_issues(
            version_id, company_id, machine_id, bubble_document_id,
            page_id, entity_id, edge_id, issue_key, issue_type,
            severity, status, message, candidates_json, properties,
            created_at, updated_at
        ) VALUES (
            %s,%s,%s,%s,%s,NULL,NULL,%s,%s,%s,'open',%s,%s::jsonb,%s::jsonb,
            NOW(),NOW()
        )
        ON CONFLICT (version_id, issue_key) DO UPDATE SET
            issue_type=EXCLUDED.issue_type,
            severity=EXCLUDED.severity,
            status='open',
            message=EXCLUDED.message,
            candidates_json=EXCLUDED.candidates_json,
            properties=EXCLUDED.properties,
            resolved_by_user_id=NULL,
            resolution_note=NULL,
            resolved_at=NULL,
            updated_at=NOW();
        """,
        (
            int(context["version_id"]),
            context["company_id"],
            context["machine_id"],
            context["bubble_document_id"],
            int(page["id"]),
            issue["issue_key"],
            issue["issue_type"],
            issue["severity"],
            issue["message"],
            json.dumps(issue.get("candidates_json") or [], ensure_ascii=False),
            json.dumps(issue.get("properties") or {}, ensure_ascii=False),
        ),
    )


def _publish_page(
    *,
    context: dict,
    page: dict,
    passed: bool,
    io_rows: list[dict],
    issues: list[dict],
    audit: dict,
    usage_totals: dict,
    language: str,
) -> dict:
    version_id = int(context["version_id"])
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.electrical_review_issues
                WHERE version_id=%s AND page_id=%s
                  AND issue_key LIKE 'structured-v2:%%';
                """,
                (version_id, int(page["id"])),
            )

            if passed:
                cur.execute(
                    """
                    DELETE FROM public.electrical_io
                    WHERE version_id=%s AND page_id=%s
                      AND extraction_method=%s;
                    """,
                    (version_id, int(page["id"]), EXTRACTION_METHOD),
                )
                for row in io_rows:
                    bbox = row["bbox"]
                    cur.execute(
                        """
                        INSERT INTO public.electrical_io(
                            version_id, company_id, machine_id, bubble_document_id,
                            page_id, source_entity_id, io_key, module_tag, module_model,
                            rack_ref, slot_ref, channel_ref, plc_address, io_type,
                            is_safety, signal_name, description, expected_normal_state,
                            wire_reference, terminal_reference,
                            x0,y0,x1,y1,source_text,properties,confidence,
                            extraction_method,is_verified,created_at,updated_at
                        ) VALUES (
                            %s,%s,%s,%s,%s,NULL,%s,%s,%s,NULL,NULL,%s,%s,%s,%s,
                            %s,%s,%s,%s,%s,
                            %s,%s,%s,%s,%s,%s::jsonb,%s,%s,false,NOW(),NOW()
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
                            is_verified=false,
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
                            row["channel_ref"],
                            row["plc_address"],
                            row["io_type"],
                            row["is_safety"],
                            row["signal_name"],
                            row["description"],
                            row["expected_normal_state"],
                            row["wire_reference"],
                            row["terminal_reference"],
                            float(bbox.x0),
                            float(bbox.y0),
                            float(bbox.x1),
                            float(bbox.y1),
                            row["source_text"],
                            json.dumps(row["properties"], ensure_ascii=False),
                            row["confidence"],
                            EXTRACTION_METHOD,
                        ),
                    )

            for issue in issues:
                _insert_issue(cur, context, page, issue)

            # Only high/critical issues block the version. Informational warnings remain auditable.
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_review_issues
                WHERE version_id=%s AND status='open'
                  AND severity IN ('high','critical');
                """,
                (version_id,),
            )
            blocking_count = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_review_issues
                WHERE version_id=%s AND status='open';
                """,
                (version_id,),
            )
            all_issue_count = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(DISTINCT page_id), COUNT(*)
                FROM public.electrical_io
                WHERE version_id=%s AND extraction_method=%s
                  AND properties ->> 'phase' = %s;
                """,
                (version_id, EXTRACTION_METHOD, PHASE_NAME),
            )
            current_row = cur.fetchone()
            passed_pages = int(current_row[0] or 0)
            current_pipeline_io_count = int(current_row[1] or 0)
            cur.execute(
                "SELECT COUNT(*) FROM public.electrical_io WHERE version_id=%s;",
                (version_id,),
            )
            io_count = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COALESCE(SUM(input_tokens),0),
                       COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(cost_usd),0)
                FROM public.electrical_ai_artifacts
                WHERE version_id=%s;
                """,
                (version_id,),
            )
            token_row = cur.fetchone()
            total_input = int(token_row[0] or 0)
            total_output = int(token_row[1] or 0)
            total_cost = float(token_row[2] or 0.0)

            if blocking_count > 0:
                structured_status = "review_required"
                version_status = "review_required"
            elif passed_pages >= int(context.get("all_io_pages_total") or 0) > 0:
                structured_status = "io_ready"
                version_status = "queued"
            else:
                structured_status = "partial"
                version_status = "queued"

            page_statuses = dict(
                (context.get("metadata") or {}).get("structured_v2_page_statuses")
                or {}
            )
            page_statuses[str(page["pdf_page_number"])] = {
                "status": "passed" if passed else "blocked",
                "published_rows": len(io_rows) if passed else 0,
                "audit": audit,
                "updated_at": datetime.now().astimezone().isoformat(),
            }
            metadata_patch = {
                "structured_status": structured_status,
                "structured_v2_status": structured_status,
                "structured_v2_pipeline_marker": PIPELINE_MARKER,
                "structured_v2_materializer_version": MATERIALIZER_VERSION,
                "structured_v2_detector_model": DETECTOR_MODEL,
                "structured_v2_extractor_model": EXTRACTOR_MODEL,
                "structured_v2_verifier_model": VERIFIER_MODEL,
                "structured_v2_detector_prompt_version": DETECTOR_PROMPT_VERSION,
                "structured_v2_extractor_prompt_version": EXTRACTOR_PROMPT_VERSION,
                "structured_v2_verifier_prompt_version": VERIFIER_PROMPT_VERSION,
                "structured_v2_page_statuses": page_statuses,
                "structured_v2_passed_pages": passed_pages,
                "structured_v2_current_pipeline_io_count": current_pipeline_io_count,
                "structured_v2_total_io_pages": int(context.get("all_io_pages_total") or 0),
                "structured_v2_last_language": language,
                "structured_v2_last_run_at": datetime.now().astimezone().isoformat(),
            }
            cur.execute(
                """
                UPDATE public.electrical_versions
                SET status=%s,
                    deterministic_only=false,
                    openai_used=true,
                    io_count=%s,
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
                    version_status,
                    io_count,
                    all_issue_count,
                    total_input,
                    total_output,
                    total_cost,
                    json.dumps(metadata_patch, ensure_ascii=False),
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
                (version_status, int(context["electrical_document_id"])),
            )
        conn.commit()
        return {
            "status": version_status,
            "structured_status": structured_status,
            "io_count": io_count,
            "current_pipeline_io_count": current_pipeline_io_count,
            "review_issue_count": all_issue_count,
            "blocking_review_issue_count": blocking_count,
            "structured_v2_passed_pages": passed_pages,
            "structured_v2_total_io_pages": int(context.get("all_io_pages_total") or 0),
            "ai_input_tokens_total": total_input,
            "ai_output_tokens_total": total_output,
            "ai_cost_usd_total": round(total_cost, 6),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _mark_failed(context: dict, page: dict, message: str) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.electrical_versions
                SET status='review_required',
                    error_code='ELECTRICAL_STRUCTURED_V2_FAILED',
                    error_message=%s,
                    metadata=COALESCE(metadata,'{}'::jsonb) || %s::jsonb,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (
                    _clean_text(message, 2000),
                    json.dumps(
                        {
                            "structured_v2_status": "failed",
                            "structured_v2_failed_page": int(page["pdf_page_number"]),
                            "structured_v2_failed_at": datetime.now()
                            .astimezone()
                            .isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                    int(context["version_id"]),
                ),
            )
            cur.execute(
                """
                UPDATE public.electrical_documents
                SET index_status='review_required',
                    last_error_code='ELECTRICAL_STRUCTURED_V2_FAILED',
                    last_error_message=%s,
                    updated_at=NOW()
                WHERE id=%s;
                """,
                (_clean_text(message, 2000), int(context["electrical_document_id"])),
            )
        conn.commit()
    finally:
        conn.close()


def extract_electrical_structured_version(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int],
    page_types: Optional[list[str]],
    pdf_page_numbers: Optional[list[int]] = None,
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
        pdf_page_numbers=pdf_page_numbers,
    )
    context["company_id"] = str(company_id)
    context["machine_id"] = str(machine_id)
    context["bubble_document_id"] = str(bubble_document_id)
    page = context["pages"][0]

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

    totals = _usage_accumulator()
    source_doc: Optional[fitz.Document] = None
    try:
        _, source_doc = _fetch_source_pdf(context)
        page_index = int(page["pdf_page_number"]) - 1
        source_page = source_doc[page_index]
        word_map = _word_map(page)
        proposals = _detect_table_proposals(
            source_page=source_page,
            inventory_page=page,
            word_map=word_map,
        )
        if not proposals:
            issue = _local_issue(
                page=page,
                code="no_deterministic_table_regions",
                message=(
                    "Geometry-first detector found no independently bordered table regions; "
                    "the page was not sent for blind row extraction."
                ),
                severity="high",
            )
            applied = _publish_page(
                context=context,
                page=page,
                passed=False,
                io_rows=[],
                issues=[issue],
                audit={
                    "passed": False,
                    "region_count": 0,
                    "physical_rows_expected": 0,
                    "physical_rows_materialized": 0,
                    "blocking_issue_count": 1,
                },
                usage_totals=totals,
                language="unknown",
            )
            return {
                "electrical_document_id": int(context["electrical_document_id"]),
                "electrical_version_id": int(context["version_id"]),
                "pdf_page_number": int(page["pdf_page_number"]),
                "page_passed": False,
                "proposals_count": 0,
                "published_rows": 0,
                **applied,
                **totals,
                "new_cost_usd": 0.0,
            }

        page_image = _render_page_with_regions(source_doc, page_index, proposals)
        detector_request = {
            "phase": PHASE_NAME,
            "task": "full_page_region_detection",
            "page_sha256": page.get("page_sha256"),
            "source_sha256": context.get("source_sha256"),
            "page_image_sha256": _sha256_bytes(page_image),
            "page_id": int(page["id"]),
            "pdf_page_number": int(page["pdf_page_number"]),
            "proposals": [
                {
                    "region_id": p["region_id"],
                    "region_hash": p["region_hash"],
                    "bbox_pt": p["crop_bbox_pt"],
                    "row_count": p["deterministic_row_count"],
                    "column_count": p["deterministic_column_count"],
                }
                for p in proposals
            ],
        }
        detector, usage, reused, detector_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_io_region_detector_v2",
            region_hash=_sha256_json([p["region_hash"] for p in proposals]),
            model=DETECTOR_MODEL,
            prompt_version=DETECTOR_PROMPT_VERSION,
            request_payload=detector_request,
            messages=_detector_messages(page, proposals, page_image),
            json_schema=_detector_schema(),
            force=bool(force),
            request_metadata={"proposal_count": len(proposals)},
        )
        _add_usage(totals, "detector", usage, reused)
        assessments = _assessment_map(detector)

        # PyMuPDF documents are rendered sequentially before network concurrency.
        # This avoids sharing a fitz.Document across worker threads.
        crop_images = {
            proposal["region_id"]: _render_region_with_rows(
                source_doc,
                page_index,
                proposal,
            )
            for proposal in proposals
        }

        def run_region(proposal: dict) -> tuple[dict, dict, bool, str]:
            rid = proposal["region_id"]
            assessment = assessments.get(rid) or {
                "visible_header_row_count": 0,
                "visible_physical_row_count": 0,
            }
            crop_image = crop_images[rid]
            request_payload = {
                "phase": PHASE_NAME,
                "task": "single_region_row_extraction",
                "page_sha256": page.get("page_sha256"),
                "source_sha256": context.get("source_sha256"),
                "crop_image_sha256": _sha256_bytes(crop_image),
                "region_id": rid,
                "region_hash": proposal["region_hash"],
                "detector_assessment": assessment,
                "rows": [
                    {
                        "row_id": row["row_id"],
                        "bbox_pt": row["bbox_pt"],
                        "word_ids": row["word_ids"],
                        "word_text_original": row["word_text_original"],
                        "deterministic_cell_text_original": row[
                            "deterministic_cell_text_original"
                        ],
                    }
                    for row in proposal["row_candidates"]
                ],
            }
            result, region_usage, region_reused, fp = _cached_call(
                context=context,
                page=page,
                task_type="vision_io_region_extractor_v2",
                region_hash=proposal["region_hash"],
                model=EXTRACTOR_MODEL,
                prompt_version=EXTRACTOR_PROMPT_VERSION,
                request_payload=request_payload,
                messages=_extractor_messages(
                    page, proposal, assessment, crop_image
                ),
                json_schema=_extractor_schema(),
                force=bool(force),
                request_metadata={
                    "region_id": rid,
                    "deterministic_row_count": proposal[
                        "deterministic_row_count"
                    ],
                },
            )
            return result, region_usage, region_reused, fp

        extraction_results: list[dict] = []
        fingerprints: dict[str, str] = {"detector": detector_fp}
        max_workers = max(1, min(3, len(proposals)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_by_region = {
                pool.submit(run_region, proposal): proposal["region_id"]
                for proposal in proposals
            }
            for future in concurrent.futures.as_completed(future_by_region):
                rid = future_by_region[future]
                result, region_usage, region_reused, fp = future.result()
                extraction_results.append(result)
                fingerprints[rid] = fp
                _add_usage(totals, "region_extractor", region_usage, region_reused)
        extraction_results.sort(key=lambda x: str(x.get("region_id") or ""))

        column_binding_proposals = _build_column_binding_proposals(
            proposals=proposals,
            detector=detector,
            extractions=extraction_results,
        )

        verifier_request = {
            "phase": PHASE_NAME,
            "task": "independent_full_page_visual_adjudication",
            "page_sha256": page.get("page_sha256"),
            "source_sha256": context.get("source_sha256"),
            "page_image_sha256": _sha256_bytes(page_image),
            "region_image_sha256": {
                rid: _sha256_bytes(image)
                for rid, image in sorted(crop_images.items())
            },
            "detector_response_sha256": _sha256_json(detector),
            "extraction_response_sha256": _sha256_json(extraction_results),
            "column_binding_proposals_sha256": _sha256_json(
                column_binding_proposals
            ),
            "region_hashes": [p["region_hash"] for p in proposals],
        }
        verifier, usage, reused, verifier_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_io_page_verifier_v2",
            region_hash=_sha256_json([p["region_hash"] for p in proposals]),
            model=VERIFIER_MODEL,
            prompt_version=VERIFIER_PROMPT_VERSION,
            request_payload=verifier_request,
            messages=_verifier_messages(
                page,
                detector,
                proposals,
                extraction_results,
                column_binding_proposals,
                page_image,
                crop_images,
            ),
            json_schema=_verifier_schema(),
            force=bool(force),
            request_metadata={"proposal_count": len(proposals)},
        )
        fingerprints["verifier"] = verifier_fp
        _add_usage(totals, "verifier", usage, reused)

        passed, io_rows, issues, audit = _validate_and_materialize(
            page=page,
            word_map=word_map,
            proposals=proposals,
            detector=detector,
            extractions=extraction_results,
            verifier=verifier,
            column_binding_proposals=column_binding_proposals,
            fingerprints=fingerprints,
        )
        language = _clean_text(detector.get("language"), 50) or "unknown"
        applied = _publish_page(
            context=context,
            page=page,
            passed=passed,
            io_rows=io_rows if passed else [],
            issues=issues,
            audit=audit,
            usage_totals=totals,
            language=language,
        )
        return {
            "electrical_document_id": int(context["electrical_document_id"]),
            "electrical_version_id": int(context["version_id"]),
            "pdf_page_number": int(page["pdf_page_number"]),
            "sheet_code": page.get("sheet_code"),
            "page_type": page.get("page_type"),
            "language": language,
            "page_passed": passed,
            "proposals_count": len(proposals),
            "regions_extracted": len(extraction_results),
            "physical_rows_expected": int(audit.get("physical_rows_expected") or 0),
            "published_rows": len(io_rows) if passed else 0,
            "placeholder_rows": int(audit.get("placeholder_rows") or 0) if passed else 0,
            "blocking_issue_count_this_page": int(audit.get("blocking_issue_count") or 0),
            "warning_issue_count_this_page": int(audit.get("warning_issue_count") or 0),
            "duplicate_source_module_tags": audit.get("duplicate_source_module_tags") or [],
            "region_stats": audit.get("region_stats") or [],
            **applied,
            **totals,
            "new_cost_usd": round(float(totals["new_cost_usd"]), 6),
        }
    except Exception as exc:
        _mark_failed(context, page, str(exc))
        raise
    finally:
        if source_doc is not None:
            source_doc.close()


# Local deterministic preflight helper; it makes no network, DB, or OpenAI call.
def _preview_geometry_for_pdf(pdf_path: str, page_numbers: list[int]) -> dict:
    doc = fitz.open(pdf_path)
    try:
        result = {}
        for page_number in page_numbers:
            source_page = doc[page_number - 1]
            words = list(source_page.get_text("words", sort=True) or [])
            inventory_page = {
                "id": page_number,
                "pdf_page_number": page_number,
                "page_sha256": hashlib.sha256(
                    source_page.get_text("text", sort=True).encode("utf-8")
                ).hexdigest(),
                "words": words,
            }
            proposals = _detect_table_proposals(
                source_page=source_page,
                inventory_page=inventory_page,
                word_map=_word_map(inventory_page),
            )
            result[page_number] = [
                {
                    "region_id": p["region_id"],
                    "rows": p["deterministic_row_count"],
                    "columns": p["deterministic_column_count"],
                    "bbox": p["crop_bbox_pt"],
                }
                for p in proposals
            ]
        return result
    finally:
        doc.close()
