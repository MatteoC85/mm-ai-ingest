import base64
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

# MachineMind Phase 2T V1
# Isolated multimodal terminal-strip extraction.
# The engine is geometry-first and language-independent: no fixed Italian/
# English vocabulary is used by deterministic publication logic.


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

TERMINALS_ENABLED = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_ENABLED") or "0"
).strip() == "1"

DETECTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_DETECTOR_MODEL") or "gpt-5.4"
).strip()
EXTRACTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_EXTRACTOR_MODEL") or "gpt-5.4"
).strip()
VERIFIER_MODEL = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_VERIFIER_MODEL") or "gpt-5.4"
).strip()

DETECTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_DETECTOR_PROMPT_VERSION")
    or "mm-electrical-terminal-detector-v1"
).strip()
EXTRACTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_EXTRACTOR_PROMPT_VERSION")
    or "mm-electrical-terminal-strip-extractor-v1"
).strip()
VERIFIER_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_VERIFIER_PROMPT_VERSION")
    or "mm-electrical-terminal-page-verifier-v1.2"
).strip()
MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_TERMINALS_MATERIALIZER_VERSION")
    or "mm-electrical-terminal-materializer-v1.2"
).strip()

OPENAI_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_TERMINALS_TIMEOUT_SECONDS", 240, 30, 600
)
FETCH_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_TERMINALS_FETCH_TIMEOUT_SECONDS", 60, 10, 300
)
RENDER_DPI = _env_int(
    "MM_ELECTRICAL_TERMINALS_RENDER_DPI", 220, 120, 360
)
MAX_COMPLETION_TOKENS = _env_int(
    "MM_ELECTRICAL_TERMINALS_MAX_COMPLETION_TOKENS", 16000, 1000, 64000
)
ROW_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_TERMINALS_ROW_MIN_CONFIDENCE", 0.82, 0.0, 1.0
)
PAGE_PASS_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_TERMINALS_PAGE_PASS_MIN_CONFIDENCE", 0.90, 0.0, 1.0
)
INPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_TERMINALS_INPUT_USD_PER_MILLION", 0.0
)
OUTPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_TERMINALS_OUTPUT_USD_PER_MILLION", 0.0
)
MAX_SOURCE_BYTES = _env_int(
    "MM_ELECTRICAL_TERMINALS_MAX_SOURCE_BYTES",
    100_000_000,
    1_000_000,
    500_000_000,
)

PIPELINE_MARKER = "phase2-terminals-v1.2-source-snapshot"
EXTRACTION_METHOD = "openai_vision_terminals_v1"
PHASE_NAME = "terminal_vision_v1"
PAGE_TYPE = "terminal_table"
SEVERITIES = {"info", "warning", "high", "critical"}
TERMINAL_ROW_ROLES = {"terminal", "spare_terminal"}
BOUNDARY_ROW_ROLES = {
    "boundary_plate",
    "header",
    "footer",
    "annotation",
    "other_non_terminal",
}
OVERRIDABLE_FIELDS = {
    "strip_tag_original",
    "terminal_number_original",
    "level_ref_original",
    "side_a_origin_original",
    "side_b_destination_original",
    "wire_number_original",
    "cable_reference_original",
    "potential_original",
    "conductor_color_original",
    "conductor_cross_section_original",
    "side_a_description_original",
    "side_b_description_original",
}

FIELD_ASSIGNMENT_DECISION_VERSION = (
    "single-visual-evidence-field-arbitration-v1"
)

REGION_ADJUDICATION_VERSION = "visual-terminal-region-eligibility-v1"
VISUAL_REGION_KINDS = {
    "terminal_strip",
    "auxiliary_description_grid",
    "boundary_plate_row",
    "annotation_grid",
    "title_block",
    "other_non_terminal",
}
NON_DATA_REGION_KINDS = VISUAL_REGION_KINDS - {"terminal_strip"}

# Every visible word associated with a numbered terminal row must remain
# represented after verifier overrides. This prevents a positive source value
# from disappearing when an unsupported field assignment is cleared.
SOURCE_EVIDENCE_FIELDS = (
    "terminal_number_original",
    "level_ref_original",
    "side_a_origin_original",
    "side_b_destination_original",
    "wire_number_original",
    "cable_reference_original",
    "potential_original",
    "conductor_color_original",
    "conductor_cross_section_original",
    "side_a_description_original",
    "side_b_description_original",
)

# These groups contain canonical fields that may accidentally receive the
# same visible token from one physical cell. A repeated value is allowed only
# when an independent visual verifier explicitly supports the shared meaning
# or when separate visible occurrences exist.
FIELD_COLLISION_GROUPS = (
    (
        "wire_number_original",
        "cable_reference_original",
        "potential_original",
        "conductor_color_original",
        "conductor_cross_section_original",
    ),
    (
        "side_a_origin_original",
        "side_b_destination_original",
    ),
    (
        "side_a_description_original",
        "side_b_description_original",
    ),
)


def get_electrical_terminal_runtime_config() -> dict:
    return {
        "enabled": bool(TERMINALS_ENABLED),
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
            f"OpenAI refused terminal vision request: {str(refusal)[:800]}"
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
            "OpenAI terminal vision call failed: "
            f"{response.status_code} {response.text[:1800]}"
        )

    data = response.json()
    text = _parse_chat_content(data)
    if not text:
        raise RuntimeError("OpenAI terminal vision call returned empty content")
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            f"Terminal vision JSON parse failed: {exc}; raw={text[:1200]}"
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
            "Terminal extraction requires exactly one pdf_page_numbers value "
            "per request to keep publication atomic."
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
                    "Requested page was not found among classified "
                    "terminal_table pages"
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
                "all_terminal_pages_total": total_pages,
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
        bbox.x0 <= page_rect.x0 + 12
        and bbox.y0 <= page_rect.y0 + 12
        and bbox.x1 >= page_rect.x1 - 12
        and bbox.y1 >= page_rect.y1 - 12
    )
    return area_ratio >= 0.58 or touches


def _column_intervals(table: Any) -> list[tuple[float, float]]:
    xs: list[float] = []
    for cell in table.cells or []:
        if not cell:
            continue
        xs.extend([float(cell[0]), float(cell[2])])
    xs.sort()
    clustered: list[list[float]] = []
    for x in xs:
        if not clustered:
            clustered.append([x])
            continue
        mean = sum(clustered[-1]) / len(clustered[-1])
        if abs(x - mean) <= 0.7:
            clustered[-1].append(x)
        else:
            clustered.append([x])
    boundaries = [sum(c) / len(c) for c in clustered]
    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] >= 3.0
    ]


def _candidate_tables(source_page: fitz.Page) -> list[Any]:
    finder = source_page.find_tables()
    candidates: list[Any] = []
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
        if int(table.col_count or 0) < 4:
            continue
        if area_ratio > 0.36:
            continue
        if bbox.width < 45 or bbox.height < 55:
            continue
        candidates.append(table)
    candidates.sort(key=lambda t: (_rect_from(t.bbox).x0, _rect_from(t.bbox).y0))
    return candidates


def _detect_geometry_proposals(
    *,
    source_page: fitz.Page,
    inventory_page: dict,
    word_map: dict[int, dict],
) -> list[dict]:
    tables = _candidate_tables(source_page)
    if not tables:
        return []

    bboxes = [_rect_from(t.bbox) for t in tables]
    proposals: list[dict] = []
    for index, (table, bbox) in enumerate(zip(tables, bboxes), start=1):
        prev_bbox = bboxes[index - 2] if index > 1 else None
        next_bbox = bboxes[index] if index < len(bboxes) else None

        # Expand toward labels/descriptions but stop at neighboring strip midpoints.
        left = max(source_page.rect.x0, bbox.x0 - 78.0)
        right = min(source_page.rect.x1, bbox.x1 + 78.0)
        if prev_bbox is not None:
            left = max(left, (prev_bbox.x1 + bbox.x0) / 2.0)
        if next_bbox is not None:
            right = min(right, (bbox.x1 + next_bbox.x0) / 2.0)
        crop_rect = fitz.Rect(
            left,
            max(source_page.rect.y0, bbox.y0 - 115.0),
            right,
            min(source_page.rect.y1, bbox.y1 + 115.0),
        )

        extracted = table.extract() or []
        intervals = _column_intervals(table)
        slot_candidates: list[dict] = []
        for slot_index, (x0, x1) in enumerate(intervals, start=1):
            slot_rect = fitz.Rect(x0, bbox.y0, x1, bbox.y1)
            ids = _ids_in_rect(word_map, slot_rect)
            cell_texts: list[str] = []
            for row_index in range(int(table.row_count or 0)):
                value = ""
                if row_index < len(extracted):
                    row_values = extracted[row_index] or []
                    if slot_index - 1 < len(row_values):
                        value = _clean_text(
                            row_values[slot_index - 1],
                            1000,
                        )
                cell_texts.append(value)
            slot_candidates.append(
                {
                    "slot_id": f"S{slot_index:03d}",
                    "slot_index": slot_index,
                    "bbox_pt": _rect_list(slot_rect),
                    "word_ids": ids,
                    "word_text_original": _text_for_ids(
                        ids,
                        word_map,
                        3000,
                    ),
                    "deterministic_cell_text_original": cell_texts,
                }
            )

        region_id = (
            f"P{int(inventory_page['pdf_page_number'])}-TS{index:02d}"
        )
        core = {
            "region_id": region_id,
            "table_bbox_pt": _rect_list(bbox),
            "crop_bbox_pt": _rect_list(crop_rect),
            "deterministic_row_count": int(table.row_count or 0),
            "deterministic_column_count": int(table.col_count or 0),
            "geometry_slot_count": len(slot_candidates),
            "slot_candidates": slot_candidates,
        }
        core["region_hash"] = _sha256_json(
            {
                "page_sha256": inventory_page.get("page_sha256"),
                **core,
            }
        )
        proposals.append(core)
    return proposals


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
            page.draw_rect(rect, color=(1, 0, 0), width=0.6, overlay=True)
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
    source_page = source_doc[page_index]
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
        for slot in proposal.get("slot_candidates") or []:
            original = _rect_from(slot["bbox_pt"])
            local = fitz.Rect(
                original.x0 - crop.x0,
                original.y0 - crop.y0,
                original.x1 - crop.x0,
                original.y1 - crop.y0,
            )
            page.draw_rect(
                local,
                color=(1, 0, 0),
                width=0.35,
                overlay=True,
            )
            page.insert_text(
                (
                    max(1.0, local.x0 + 1.0),
                    max(6.0, local.y0 + 6.0),
                ),
                slot["slot_id"],
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
                "maxItems": 300,
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
        "name": "electrical_terminal_page_detector_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "all_visible_strips_accounted_for": {"type": "boolean"},
                "preferred_reading_rotation_degrees": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                },
                "regions": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "is_terminal_strip": {"type": "boolean"},
                            "strip_tag_original": {"type": "string"},
                            "expected_terminal_rows": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 400,
                            },
                            "expected_boundary_rows": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 40,
                            },
                            "visible_number_sequence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 400,
                            },
                            "confidence": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "is_terminal_strip",
                            "strip_tag_original",
                            "expected_terminal_rows",
                            "expected_boundary_rows",
                            "visible_number_sequence",
                            "confidence",
                            "notes",
                        ],
                    },
                },
                "missing_visible_strips": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
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
                "all_visible_strips_accounted_for",
                "preferred_reading_rotation_degrees",
                "regions",
                "missing_visible_strips",
                "confidence",
                "issues",
            ],
        },
    }


def _terminal_row_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "row_id": {"type": "string"},
            "visual_order": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
            },
            "row_role": {
                "type": "string",
                "enum": sorted(TERMINAL_ROW_ROLES),
            },
            "terminal_number_original": {"type": "string"},
            "level_ref_original": {"type": "string"},
            "side_a_origin_original": {"type": "string"},
            "side_b_destination_original": {"type": "string"},
            "wire_number_original": {"type": "string"},
            "cable_reference_original": {"type": "string"},
            "potential_original": {"type": "string"},
            "conductor_color_original": {"type": "string"},
            "conductor_cross_section_original": {"type": "string"},
            "side_a_description_original": {"type": "string"},
            "side_b_description_original": {"type": "string"},
            "source_slot_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 12,
            },
            "source_word_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 300,
            },
            "bbox_pt": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "confidence": {"type": "number"},
            "evidence_notes": {"type": "string"},
        },
        "required": [
            "row_id",
            "visual_order",
            "row_role",
            "terminal_number_original",
            "level_ref_original",
            "side_a_origin_original",
            "side_b_destination_original",
            "wire_number_original",
            "cable_reference_original",
            "potential_original",
            "conductor_color_original",
            "conductor_cross_section_original",
            "side_a_description_original",
            "side_b_description_original",
            "source_slot_ids",
            "source_word_ids",
            "bbox_pt",
            "confidence",
            "evidence_notes",
        ],
    }


def _extractor_schema() -> dict:
    return {
        "name": "electrical_terminal_strip_extractor_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "region_id": {"type": "string"},
                "strip_tag_original": {"type": "string"},
                "source_side_label_original": {"type": "string"},
                "destination_side_label_original": {"type": "string"},
                "boundary_rows": {
                    "type": "array",
                    "maxItems": 40,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "boundary_id": {"type": "string"},
                            "row_role": {
                                "type": "string",
                                "enum": sorted(BOUNDARY_ROW_ROLES),
                            },
                            "text_original": {"type": "string"},
                            "source_slot_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 12,
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "boundary_id",
                            "row_role",
                            "text_original",
                            "source_slot_ids",
                            "confidence",
                        ],
                    },
                },
                "terminals": {
                    "type": "array",
                    "maxItems": 400,
                    "items": _terminal_row_schema(),
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 40,
                },
            },
            "required": [
                "region_id",
                "strip_tag_original",
                "source_side_label_original",
                "destination_side_label_original",
                "boundary_rows",
                "terminals",
                "confidence",
                "issues",
            ],
        },
    }



def _verifier_schema() -> dict:
    region_adjudication_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "region_ref": {"type": "string"},
            "visual_region_kind": {
                "type": "string",
                "enum": sorted(VISUAL_REGION_KINDS),
            },
            "is_data_terminal_strip": {"type": "boolean"},
            "has_strip_tag": {"type": "boolean"},
            "has_terminal_number_axis": {"type": "boolean"},
            "has_numbered_terminal_rows": {"type": "boolean"},
            "has_connection_semantics": {"type": "boolean"},
            "accounted": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "region_ref",
            "visual_region_kind",
            "is_data_terminal_strip",
            "has_strip_tag",
            "has_terminal_number_axis",
            "has_numbered_terminal_rows",
            "has_connection_semantics",
            "accounted",
            "confidence",
            "reason",
        ],
    }
    return {
        "name": "electrical_terminal_page_verifier_v1_2",
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
                # Kept for backward-compatible audit. In V1.2 it means that
                # every actual data-bearing strip is accounted for after the
                # strip-like regions have been classified.
                "all_visible_strips_accounted_for": {"type": "boolean"},
                "all_strip_like_regions_classified": {"type": "boolean"},
                "all_data_terminal_strips_accounted_for": {
                    "type": "boolean"
                },
                "all_visible_terminal_rows_accounted_for": {
                    "type": "boolean"
                },
                "all_strip_tags_supported_by_headers": {"type": "boolean"},
                "all_terminal_numbers_visually_supported": {
                    "type": "boolean"
                },
                "all_published_fields_visually_supported": {
                    "type": "boolean"
                },
                "uncovered_region_adjudications": {
                    "type": "array",
                    "maxItems": 40,
                    "items": region_adjudication_schema,
                },
                "unaccounted_data_terminal_strip_regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "region_checks": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "strip_tag_original": {"type": "string"},
                            "expected_terminal_rows": {"type": "integer"},
                            "extracted_terminal_rows": {"type": "integer"},
                            "expected_terminal_number_sequence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 400,
                            },
                            "verified_terminal_number_sequence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 400,
                            },
                            "boundary_rows_accounted_for": {
                                "type": "boolean"
                            },
                            "pass": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "strip_tag_original",
                            "expected_terminal_rows",
                            "extracted_terminal_rows",
                            "expected_terminal_number_sequence",
                            "verified_terminal_number_sequence",
                            "boundary_rows_accounted_for",
                            "pass",
                            "confidence",
                            "notes",
                        ],
                    },
                },
                "field_support_decisions": {
                    "type": "array",
                    "maxItems": 300,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "row_id": {"type": "string"},
                            "source_text_original": {"type": "string"},
                            "source_slot_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 12,
                            },
                            "visual_occurrence_count": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 20,
                            },
                            "supported_fields": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": sorted(OVERRIDABLE_FIELDS),
                                },
                                "maxItems": 12,
                            },
                            "unsupported_fields": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": sorted(OVERRIDABLE_FIELDS),
                                },
                                "maxItems": 12,
                            },
                            "shared_semantics_explicitly_supported": {
                                "type": "boolean"
                            },
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "row_id",
                            "source_text_original",
                            "source_slot_ids",
                            "visual_occurrence_count",
                            "supported_fields",
                            "unsupported_fields",
                            "shared_semantics_explicitly_supported",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "field_overrides": {
                    "type": "array",
                    "maxItems": 300,
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
                "missing_terminal_row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 400,
                },
                "duplicate_physical_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 400,
                },
                "unaccounted_visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 40,
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
                "verdict",
                "all_visible_strips_accounted_for",
                "all_strip_like_regions_classified",
                "all_data_terminal_strips_accounted_for",
                "all_visible_terminal_rows_accounted_for",
                "all_strip_tags_supported_by_headers",
                "all_terminal_numbers_visually_supported",
                "all_published_fields_visually_supported",
                "uncovered_region_adjudications",
                "unaccounted_data_terminal_strip_regions",
                "region_checks",
                "field_support_decisions",
                "field_overrides",
                "missing_region_ids",
                "missing_terminal_row_ids",
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
            "table_bbox_pt": p["table_bbox_pt"],
            "crop_bbox_pt": p["crop_bbox_pt"],
            "deterministic_row_count": p["deterministic_row_count"],
            "deterministic_column_count": p[
                "deterministic_column_count"
            ],
            "geometry_slot_count": p["geometry_slot_count"],
        }
        for p in proposals
    ]
    system = (
        "You are the visual perception stage of an industrial electrical "
        "terminal-strip reader. The page can be in any language, font, "
        "orientation, CAD system, or drawing standard. Work semantically "
        "from the complete images and geometry proposals; never depend on a "
        "fixed vocabulary. Identify every distinct physical terminal strip. "
        "Do not confuse title blocks, empty description grids, symbols, or "
        "boundary/plate-terminal rows with numbered terminals. Preserve "
        "printed strip tags and terminal-number order exactly, including "
        "gaps, non-monotonic order, suffixes, levels, and repeated numbers. "
        "Never invent a missing number or correct the source. A page can "
        "contain multiple strips."
    )
    user_text = (
        "Audit every red geometry proposal. Return one region assessment for "
        "each proposal ID, identify any visible strip not covered, count "
        "actual numbered terminal rows separately from boundary/plate rows, "
        "and report the visible terminal-number sequence in visual order.\n\n"
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
        {
            "type": "text",
            "text": "PAGE ROTATED 90 DEGREES FOR READING",
        },
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
    detector_region: dict,
    crop_original: bytes,
    crop_rotated: bytes,
) -> list[dict]:
    system = (
        "You are the extraction stage of an industrial electrical "
        "terminal-strip reader. The crop may use any language, font, "
        "orientation, or drawing standard. Understand column/side meaning "
        "semantically from the image; do not use a fixed word list. Extract "
        "one record for every visible numbered physical terminal row and no "
        "record for plate-terminal boundaries, headers, footers, or empty "
        "description scaffolding. Preserve visual order and printed values "
        "exactly; never renumber, reorder, translate, normalize codes, or "
        "fill absent values. The source side and destination side can appear "
        "on either physical side of the strip. Map side_a_origin to the side "
        "semantically presented as source/origin and side_b_destination to "
        "the side semantically presented as destination/target. A terminal "
        "number is mandatory. A gap in numbering is valid. A visible value "
        "can be both a wire number and a potential only when the crop clearly "
        "supports both meanings. Keep side descriptions separate from tags. "
        "Use only source_slot_ids and source_word_ids supplied in the request."
    )
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "region_id": proposal["region_id"],
        "detector_region": detector_region,
        "geometry": {
            "table_bbox_pt": proposal["table_bbox_pt"],
            "crop_bbox_pt": proposal["crop_bbox_pt"],
            "slot_candidates": proposal["slot_candidates"],
        },
    }
    user_text = (
        "Extract the terminal strip completely. Red labels identify geometry "
        "slot candidates only; merged cells or boundary rows can make the "
        "geometry count differ from the true numbered-terminal count. Use the "
        "visual crop as source of truth and explain ambiguities as issues.\n\n"
        + json.dumps(request, ensure_ascii=False)
    )
    content = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "REGION IN SOURCE ORIENTATION"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(crop_original),
                "detail": "original",
            },
        },
        {
            "type": "text",
            "text": "REGION ROTATED 90 DEGREES FOR READING",
        },
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
    word_map: dict[int, dict],
) -> list[dict]:
    system = (
        "You are the independent visual verifier of an industrial electrical "
        "terminal-strip reader. Re-read the full page and every high-resolution "
        "strip crop. The source may use any language, font, orientation, or CAD "
        "system. Verify that every actual data-bearing terminal strip, numbered "
        "terminal, boundary row, side value, and visible row text is accounted "
        "for. A grid is not automatically a terminal strip. For every uncovered "
        "strip-like region reported by the preliminary detector, copy its "
        "description exactly into region_ref and classify it as one of: "
        "terminal_strip, auxiliary_description_grid, boundary_plate_row, "
        "annotation_grid, title_block, or other_non_terminal. A data-bearing "
        "terminal strip requires numbered terminal rows plus connection "
        "semantics and either a strip identity or a terminal-number axis. Empty "
        "layout/template grids without numbered terminal rows or electrical "
        "connection semantics are non-data regions: account for them in the "
        "audit, but do not require extraction or publication. Never dismiss a "
        "true data-bearing strip merely because it is empty in some columns. "
        "Compare exact terminal-number sequences, not just counts. Preserve "
        "source errors, gaps, non-monotonic ordering, and repeated tags. Do not "
        "infer missing values. A plate-terminal/boundary row is not a numbered "
        "terminal. A repeated strip tag is allowed only when two distinct "
        "physical regions are visible. Provide a field override only when the "
        "image supports one unambiguous exact transcription; otherwise block. "
        "Physical column/lane semantics are stronger evidence than the lexical "
        "shape of a value. For every row where one non-empty visible text is "
        "assigned to more than one canonical field, return a "
        "field_support_decision. When a visible text is absent from its correct "
        "physical field or is assigned to a wrong field, emit both sides of the "
        "reassignment: set the correct field to the exact visible text and clear "
        "the unsupported field. This also applies to spare/unused terminal rows: "
        "a textual label printed in a physical wire lane remains wire text even "
        "when it is not code-like. Do not translate, paraphrase, normalize, or "
        "drop visible text. Return pass only when every actual data strip and "
        "terminal row is covered, every strip-like region is classified, every "
        "visible row word is represented in an approved field, and every "
        "published field is visually supported."
    )
    source_evidence = _row_source_evidence(extractions, word_map)
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "detector": detector,
        "detector_missing_visible_strips": (
            detector.get("missing_visible_strips") or []
        ),
        "region_adjudication_version": REGION_ADJUDICATION_VERSION,
        "proposal_region_ids": [p["region_id"] for p in proposals],
        "extractions": extractions,
        "row_source_evidence": source_evidence,
        "same_text_multi_field_candidates": (
            _same_text_multi_field_candidates(extractions)
        ),
        "field_assignment_decision_version": (
            FIELD_ASSIGNMENT_DECISION_VERSION
        ),
        "row_min_confidence": ROW_MIN_CONFIDENCE,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
    }
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Audit this page and all extracted terminal rows. First "
                "classify every detector-reported uncovered strip-like region "
                "using its exact detector text as region_ref. Distinguish "
                "data-bearing terminal strips from empty auxiliary/layout "
                "grids by visual structure and electrical content, not by "
                "language or font. Then verify exact per-region sequences. "
                "Return a field_support_decision for every same-text "
                "multi-field assignment. For any visible row text assigned "
                "to a wrong field or missing from the physically supported "
                "field, return positive and negative field_overrides so no "
                "source evidence disappears.\n\n"
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
                {
                    "type": "text",
                    "text": f"REGION {rid} ROTATED 90 DEGREES",
                },
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


def _row_source_evidence(
    extractions: list[dict],
    word_map: dict[int, dict],
) -> list[dict]:
    evidence: list[dict] = []
    for extraction in extractions:
        region_id = _clean_text(extraction.get("region_id"), 120)
        for terminal in extraction.get("terminals") or []:
            word_ids = [
                int(x)
                for x in (terminal.get("source_word_ids") or [])
                if isinstance(x, int) or str(x).isdigit()
            ]
            evidence.append(
                {
                    "region_id": region_id,
                    "row_id": _clean_text(terminal.get("row_id"), 120),
                    "row_role": _clean_text(terminal.get("row_role"), 120),
                    "source_slot_ids": [
                        str(x)
                        for x in (terminal.get("source_slot_ids") or [])
                        if str(x)
                    ],
                    "source_word_ids": word_ids,
                    "source_text_original": _text_for_ids(
                        word_ids,
                        word_map,
                        4000,
                    ),
                    "current_fields": {
                        field: _clean_text(terminal.get(field), 1000)
                        for field in SOURCE_EVIDENCE_FIELDS
                    },
                }
            )
    return evidence


def _normalized_evidence_atom(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z]+", "", text)


def _unrepresented_source_evidence(
    terminal: dict,
    word_map: dict[int, dict],
) -> list[dict]:
    published_values = [
        _normalized_evidence_atom(terminal.get(field))
        for field in SOURCE_EVIDENCE_FIELDS
        if _normalized_evidence_atom(terminal.get(field))
    ]
    missing: list[dict] = []
    for raw_id in terminal.get("source_word_ids") or []:
        try:
            word_id = int(raw_id)
        except Exception:
            continue
        word = word_map.get(word_id)
        if not word:
            continue
        original = _clean_text(word.get("text"), 500)
        atom = _normalized_evidence_atom(original)
        if not atom:
            continue
        if any(atom in value for value in published_values):
            continue
        missing.append(
            {
                "word_id": word_id,
                "text_original": original,
                "normalized_atom": atom,
            }
        )
    return missing


def _adjudicate_uncovered_regions(
    *,
    detector: dict,
    verifier: dict,
) -> tuple[list[dict], list[dict]]:
    missing_refs = [
        _clean_text(x, 500)
        for x in (detector.get("missing_visible_strips") or [])
        if _clean_text(x, 500)
    ]
    adjudications = [
        x
        for x in (verifier.get("uncovered_region_adjudications") or [])
        if isinstance(x, dict)
    ]
    by_ref = {
        _clean_text(x.get("region_ref"), 500).casefold(): x
        for x in adjudications
        if _clean_text(x.get("region_ref"), 500)
    }

    audit: list[dict] = []
    issues: list[dict] = []

    if missing_refs and not verifier.get("all_strip_like_regions_classified"):
        issues.append(
            {
                "issue_type": "terminal-strip-like-regions-not-classified",
                "severity": "high",
                "message": (
                    "The verifier did not classify every uncovered "
                    "strip-like visual region."
                ),
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )

    for ref in missing_refs:
        adjudication = by_ref.get(ref.casefold())
        if not adjudication:
            issues.append(
                {
                    "issue_type": "terminal-uncovered-region-not-adjudicated",
                    "severity": "high",
                    "message": (
                        "Detector-reported uncovered visual region was not "
                        f"adjudicated by the verifier: {ref}"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
            continue

        kind = _clean_text(
            adjudication.get("visual_region_kind"),
            120,
        )
        confidence = max(
            0.0,
            min(1.0, float(adjudication.get("confidence") or 0.0)),
        )
        accounted = bool(adjudication.get("accounted"))
        has_strip_tag = bool(adjudication.get("has_strip_tag"))
        has_axis = bool(adjudication.get("has_terminal_number_axis"))
        has_rows = bool(adjudication.get("has_numbered_terminal_rows"))
        has_connections = bool(
            adjudication.get("has_connection_semantics")
        )
        declared_data = bool(adjudication.get("is_data_terminal_strip"))
        # Numbered physical rows plus either a strip identity or a
        # terminal-number axis are sufficient to make a region data-bearing.
        # Connection semantics strengthen the decision but are not mandatory:
        # a valid strip can consist entirely of spare/reserved terminals.
        computed_data = bool(
            has_rows
            and (has_strip_tag or has_axis)
        )

        audit_item = {
            "version": REGION_ADJUDICATION_VERSION,
            "region_ref": ref,
            "visual_region_kind": kind,
            "is_data_terminal_strip": declared_data,
            "computed_is_data_terminal_strip": computed_data,
            "has_strip_tag": has_strip_tag,
            "has_terminal_number_axis": has_axis,
            "has_numbered_terminal_rows": has_rows,
            "has_connection_semantics": has_connections,
            "accounted": accounted,
            "confidence": confidence,
            "reason": _clean_text(adjudication.get("reason"), 1600),
        }
        audit.append(audit_item)

        if kind not in VISUAL_REGION_KINDS:
            issues.append(
                {
                    "issue_type": "terminal-visual-region-kind-invalid",
                    "severity": "high",
                    "message": f"Invalid visual region kind for {ref}: {kind!r}",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": confidence,
                    "source_stage": "deterministic_validator",
                }
            )
            continue
        if confidence < PAGE_PASS_MIN_CONFIDENCE or not accounted:
            issues.append(
                {
                    "issue_type": "terminal-uncovered-region-low-confidence",
                    "severity": "high",
                    "message": (
                        "Uncovered visual region was not confidently "
                        f"accounted for: {ref}"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": confidence,
                    "source_stage": "deterministic_validator",
                }
            )
            continue
        if declared_data != computed_data:
            issues.append(
                {
                    "issue_type": "terminal-region-eligibility-inconsistent",
                    "severity": "high",
                    "message": (
                        "Verifier region eligibility is internally "
                        f"inconsistent for {ref}"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": confidence,
                    "source_stage": "deterministic_validator",
                }
            )
            continue
        if computed_data or kind == "terminal_strip":
            issues.append(
                {
                    "issue_type": "terminal-uncovered-data-strip",
                    "severity": "high",
                    "message": (
                        "A data-bearing terminal strip is visible but has no "
                        f"geometry/extraction region: {ref}"
                    ),
                    "region_id": "",
                    "row_ids": [],
                    "confidence": confidence,
                    "source_stage": "deterministic_validator",
                }
            )
            continue

        issues.append(
            {
                "issue_type": "terminal-non-data-region-adjudicated",
                "severity": "info",
                "message": (
                    f"Uncovered strip-like region classified as {kind}: {ref}"
                ),
                "region_id": "",
                "row_ids": [],
                "confidence": confidence,
                "source_stage": "deterministic_validator",
            }
        )

    unaccounted_data = [
        _clean_text(x, 500)
        for x in (
            verifier.get("unaccounted_data_terminal_strip_regions") or []
        )
        if _clean_text(x, 500)
    ]
    if unaccounted_data:
        issues.append(
            {
                "issue_type": "terminal-unaccounted-data-strips",
                "severity": "high",
                "message": (
                    "Verifier found unaccounted data-bearing terminal strips: "
                    + "; ".join(unaccounted_data)
                ),
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    if not verifier.get("all_data_terminal_strips_accounted_for"):
        issues.append(
            {
                "issue_type": "terminal-data-strip-coverage-failed",
                "severity": "high",
                "message": (
                    "Verifier returned "
                    "all_data_terminal_strips_accounted_for=false"
                ),
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    return audit, issues


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
        totals["new_input_tokens"] += int(
            usage.get("input_tokens") or 0
        )
        totals["new_output_tokens"] += int(
            usage.get("output_tokens") or 0
        )
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
            issue.get("message") or "Terminal extraction issue",
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
        ][:300],
        "confidence": max(
            0.0,
            min(1.0, float(issue.get("confidence") or 0.0)),
        ),
        "source_stage": source_stage,
    }


def _apply_overrides(
    extractions: list[dict],
    overrides: list[dict],
) -> None:
    lookup: dict[tuple[str, str], dict] = {}
    for extraction in extractions:
        rid = str(extraction.get("region_id") or "")
        for terminal in extraction.get("terminals") or []:
            lookup[(rid, str(terminal.get("row_id") or ""))] = terminal
    for override in overrides or []:
        rid = str(override.get("region_id") or "")
        row_id = str(override.get("row_id") or "")
        field = str(override.get("field_name") or "")
        approved = _clean_text(override.get("approved_text"), 1000)
        confidence = float(override.get("confidence") or 0.0)
        terminal = lookup.get((rid, row_id))
        if not terminal or field not in OVERRIDABLE_FIELDS:
            continue
        if confidence < PAGE_PASS_MIN_CONFIDENCE:
            continue
        terminal.setdefault("verifier_overrides", {})[field] = {
            "before": str(terminal.get(field) or ""),
            "after": approved,
            "confidence": confidence,
            "reason": _clean_text(override.get("reason"), 1000),
        }
        terminal[field] = approved



def _normalized_field_value(value: Any) -> str:
    return _clean_text(value, 1000).casefold()


def _field_collisions(terminal: dict) -> list[dict]:
    collisions: list[dict] = []
    for group in FIELD_COLLISION_GROUPS:
        by_value: dict[str, list[str]] = {}
        original_by_value: dict[str, str] = {}
        for field in group:
            original = _clean_text(terminal.get(field), 1000)
            normalized = _normalized_field_value(original)
            if not normalized:
                continue
            by_value.setdefault(normalized, []).append(field)
            original_by_value.setdefault(normalized, original)
        for normalized, fields in by_value.items():
            if len(fields) < 2:
                continue
            collisions.append(
                {
                    "normalized_value": normalized,
                    "source_text_original": original_by_value[normalized],
                    "fields": sorted(fields),
                }
            )
    return collisions


def _same_text_multi_field_candidates(
    extractions: list[dict],
) -> list[dict]:
    candidates: list[dict] = []
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        for terminal in extraction.get("terminals") or []:
            row_id = _clean_text(terminal.get("row_id"), 120)
            for collision in _field_collisions(terminal):
                candidates.append(
                    {
                        "region_id": rid,
                        "row_id": row_id,
                        "source_text_original": collision[
                            "source_text_original"
                        ],
                        "assigned_fields": collision["fields"],
                        "source_slot_ids": [
                            str(x)
                            for x in (
                                terminal.get("source_slot_ids") or []
                            )
                            if str(x)
                        ],
                    }
                )
    return candidates


def _decision_lookup(verifier: dict) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for raw in verifier.get("field_support_decisions") or []:
        if not isinstance(raw, dict):
            continue
        rid = _clean_text(raw.get("region_id"), 120)
        row_id = _clean_text(raw.get("row_id"), 120)
        value = _normalized_field_value(raw.get("source_text_original"))
        if not rid or not row_id or not value:
            continue
        lookup[(rid, row_id, value)] = raw
    return lookup


def _apply_field_support_decisions(
    extractions: list[dict],
    decisions: list[dict],
) -> None:
    lookup: dict[tuple[str, str], dict] = {}
    for extraction in extractions:
        rid = _clean_text(extraction.get("region_id"), 120)
        for terminal in extraction.get("terminals") or []:
            row_id = _clean_text(terminal.get("row_id"), 120)
            if rid and row_id:
                lookup[(rid, row_id)] = terminal

    for raw in decisions or []:
        if not isinstance(raw, dict):
            continue
        rid = _clean_text(raw.get("region_id"), 120)
        row_id = _clean_text(raw.get("row_id"), 120)
        terminal = lookup.get((rid, row_id))
        if not terminal:
            continue

        confidence = max(
            0.0,
            min(1.0, float(raw.get("confidence") or 0.0)),
        )
        source_text = _clean_text(raw.get("source_text_original"), 1000)
        normalized = _normalized_field_value(source_text)
        supported = {
            str(x)
            for x in (raw.get("supported_fields") or [])
            if str(x) in OVERRIDABLE_FIELDS
        }
        unsupported = {
            str(x)
            for x in (raw.get("unsupported_fields") or [])
            if str(x) in OVERRIDABLE_FIELDS
        }
        row_slots = {
            str(x)
            for x in (terminal.get("source_slot_ids") or [])
            if str(x)
        }
        decision_slots = {
            str(x)
            for x in (raw.get("source_slot_ids") or [])
            if str(x)
        }

        audit = {
            "version": FIELD_ASSIGNMENT_DECISION_VERSION,
            "source_text_original": source_text,
            "source_slot_ids": sorted(decision_slots),
            "visual_occurrence_count": int(
                raw.get("visual_occurrence_count") or 0
            ),
            "supported_fields": sorted(supported),
            "unsupported_fields": sorted(unsupported),
            "shared_semantics_explicitly_supported": bool(
                raw.get("shared_semantics_explicitly_supported")
            ),
            "confidence": confidence,
            "reason": _clean_text(raw.get("reason"), 1200),
            "applied": False,
        }

        terminal.setdefault(
            "verifier_field_support_decisions",
            [],
        ).append(audit)

        if confidence < PAGE_PASS_MIN_CONFIDENCE:
            continue
        if not normalized or not supported:
            continue
        if supported & unsupported:
            continue
        if decision_slots and not decision_slots.issubset(row_slots):
            continue

        matching_fields = {
            field
            for field in supported | unsupported
            if _normalized_field_value(terminal.get(field)) == normalized
        }
        if not supported.issubset(matching_fields):
            continue

        changed = False
        for field in unsupported:
            if _normalized_field_value(terminal.get(field)) != normalized:
                continue
            before = _clean_text(terminal.get(field), 1000)
            terminal.setdefault("verifier_overrides", {})[field] = {
                "before": before,
                "after": "",
                "confidence": confidence,
                "reason": (
                    "Cleared by "
                    f"{FIELD_ASSIGNMENT_DECISION_VERSION}: "
                    + _clean_text(raw.get("reason"), 900)
                ),
            }
            terminal[field] = ""
            changed = True

        audit["applied"] = bool(changed or not unsupported)


def _field_collision_issues(
    *,
    terminal: dict,
    region_id: str,
    row_id: str,
    verifier_decisions: dict[tuple[str, str, str], dict],
) -> list[dict]:
    issues: list[dict] = []
    for collision in _field_collisions(terminal):
        value = collision["normalized_value"]
        fields = set(collision["fields"])
        decision = verifier_decisions.get((region_id, row_id, value))
        approved = False
        if isinstance(decision, dict):
            confidence = float(decision.get("confidence") or 0.0)
            supported = {
                str(x)
                for x in (decision.get("supported_fields") or [])
                if str(x) in OVERRIDABLE_FIELDS
            }
            unsupported = {
                str(x)
                for x in (decision.get("unsupported_fields") or [])
                if str(x) in OVERRIDABLE_FIELDS
            }
            occurrence_count = int(
                decision.get("visual_occurrence_count") or 0
            )
            shared = bool(
                decision.get("shared_semantics_explicitly_supported")
            )
            approved = (
                confidence >= PAGE_PASS_MIN_CONFIDENCE
                and not unsupported
                and fields.issubset(supported)
                and (
                    occurrence_count >= len(fields)
                    or shared
                )
            )
        if approved:
            continue
        issues.append(
            {
                "issue_type": (
                    "terminal-single-visual-evidence-multi-field-unresolved"
                ),
                "severity": "high",
                "message": (
                    f"Row {region_id}/{row_id} publishes one visible value "
                    f"{collision['source_text_original']!r} in multiple "
                    f"canonical fields {sorted(fields)} without a verified "
                    "multi-role field decision."
                ),
                "region_id": region_id,
                "row_ids": [row_id],
                "confidence": float(terminal.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    return issues


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

    proposal_by_id = {p["region_id"]: p for p in proposals}
    detector_regions = {
        str(x.get("region_id") or ""): x
        for x in (detector.get("regions") or [])
        if isinstance(x, dict)
    }
    extraction_by_id = {
        str(x.get("region_id") or ""): x
        for x in extractions
        if isinstance(x, dict)
    }
    verifier_decisions = _decision_lookup(verifier)

    region_audit, region_coverage_issues = (
        _adjudicate_uncovered_regions(
            detector=detector,
            verifier=verifier,
        )
    )
    issues.extend(region_coverage_issues)

    for raw in detector.get("issues") or []:
        issues.append(
            _normalize_issue(
                raw,
                default_type="terminal-detector-issue",
                source_stage="detector",
            )
        )

    active_region_ids: list[str] = []
    for rid, dregion in detector_regions.items():
        if not dregion.get("is_terminal_strip"):
            continue
        active_region_ids.append(rid)
        proposal = proposal_by_id.get(rid)
        extraction = extraction_by_id.get(rid)
        if not proposal or not extraction:
            issues.append(
                {
                    "issue_type": "terminal-region-not-extracted",
                    "severity": "high",
                    "message": f"Terminal region {rid} is missing extraction",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
            continue

        detector_tag = _clean_text(
            dregion.get("strip_tag_original"),
            200,
        )
        extraction_tag = _clean_text(
            extraction.get("strip_tag_original"),
            200,
        )
        expected_count = int(dregion.get("expected_terminal_rows") or 0)
        terminals = extraction.get("terminals") or []
        if not detector_tag or not extraction_tag:
            issues.append(
                {
                    "issue_type": "terminal-strip-tag-missing",
                    "severity": "high",
                    "message": f"Strip tag is missing for region {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
        if detector_tag and extraction_tag and detector_tag != extraction_tag:
            issues.append(
                {
                    "issue_type": "terminal-strip-tag-mismatch",
                    "severity": "high",
                    "message": (
                        f"Detector/extractor strip-tag mismatch in {rid}: "
                        f"{detector_tag!r} vs {extraction_tag!r}"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": min(
                        float(dregion.get("confidence") or 0.0),
                        float(extraction.get("confidence") or 0.0),
                    ),
                    "source_stage": "deterministic_validator",
                }
            )
        expected_boundary_count = int(
            dregion.get("expected_boundary_rows") or 0
        )
        actual_boundary_count = len(extraction.get("boundary_rows") or [])
        if actual_boundary_count != expected_boundary_count:
            issues.append(
                {
                    "issue_type": "terminal-boundary-count-mismatch",
                    "severity": "high",
                    "message": (
                        f"Region {rid} expected {expected_boundary_count} "
                        f"boundary rows but extracted {actual_boundary_count}"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        if expected_count <= 0 or len(terminals) != expected_count:
            issues.append(
                {
                    "issue_type": "terminal-row-count-mismatch",
                    "severity": "high",
                    "message": (
                        f"Region {rid} expected {expected_count} terminal rows "
                        f"but extracted {len(terminals)}"
                    ),
                    "region_id": rid,
                    "row_ids": [
                        str(x.get("row_id") or "") for x in terminals
                    ],
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

        detector_sequence = [
            _clean_text(x, 300)
            for x in (dregion.get("visible_number_sequence") or [])
        ]
        extraction_sequence = [
            _clean_text(x.get("terminal_number_original"), 300)
            for x in sorted(
                terminals,
                key=lambda x: int(x.get("visual_order") or 0),
            )
        ]
        if detector_sequence != extraction_sequence:
            issues.append(
                {
                    "issue_type": "terminal-detector-extractor-sequence-mismatch",
                    "severity": "high",
                    "message": (
                        f"Detector/extractor terminal sequence mismatch in {rid}; "
                        f"detector={detector_sequence}, "
                        f"extractor={extraction_sequence}"
                    ),
                    "region_id": rid,
                    "row_ids": [
                        str(x.get("row_id") or "") for x in terminals
                    ],
                    "confidence": min(
                        float(dregion.get("confidence") or 0.0),
                        float(extraction.get("confidence") or 0.0),
                    ),
                    "source_stage": "deterministic_validator",
                }
            )

        valid_slot_ids = {
            str(x.get("slot_id") or "")
            for x in proposal.get("slot_candidates") or []
        }
        seen_row_ids: set[str] = set()
        seen_physical_keys: set[tuple[str, str, str, int]] = set()
        for item in terminals:
            row_id = _clean_text(item.get("row_id"), 120)
            visual_order = int(item.get("visual_order") or 0)
            terminal_number = _clean_text(
                item.get("terminal_number_original"),
                300,
            )
            level_ref = _clean_text(
                item.get("level_ref_original"),
                200,
            )
            confidence = float(item.get("confidence") or 0.0)
            row_role = str(item.get("row_role") or "")
            slot_ids = [
                str(x)
                for x in (item.get("source_slot_ids") or [])
                if str(x)
            ]
            source_word_ids = [
                int(x)
                for x in (item.get("source_word_ids") or [])
                if isinstance(x, int) or str(x).isdigit()
            ]

            if not row_id or row_id in seen_row_ids:
                issues.append(
                    {
                        "issue_type": "terminal-row-id-invalid",
                        "severity": "high",
                        "message": f"Missing or duplicate terminal row_id in {rid}",
                        "region_id": rid,
                        "row_ids": [row_id] if row_id else [],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            seen_row_ids.add(row_id)
            if not terminal_number:
                issues.append(
                    {
                        "issue_type": "terminal-number-missing",
                        "severity": "high",
                        "message": f"Terminal number missing in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            if visual_order < 1:
                issues.append(
                    {
                        "issue_type": "terminal-visual-order-invalid",
                        "severity": "high",
                        "message": f"Visual order invalid in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            if row_role not in TERMINAL_ROW_ROLES:
                issues.append(
                    {
                        "issue_type": "terminal-row-role-invalid",
                        "severity": "high",
                        "message": f"Invalid terminal row role in {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            if confidence < ROW_MIN_CONFIDENCE:
                issues.append(
                    {
                        "issue_type": "terminal-row-confidence-below-threshold",
                        "severity": "high",
                        "message": f"Low confidence terminal row {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            if not slot_ids or any(x not in valid_slot_ids for x in slot_ids):
                issues.append(
                    {
                        "issue_type": "terminal-slot-evidence-invalid",
                        "severity": "high",
                        "message": f"Invalid slot evidence for {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            invalid_word_ids = [
                wid for wid in source_word_ids if wid not in word_map
            ]
            if invalid_word_ids:
                issues.append(
                    {
                        "issue_type": "terminal-word-evidence-invalid",
                        "severity": "high",
                        "message": f"Invalid source word IDs for {rid}/{row_id}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            physical_key = (
                rid,
                terminal_number,
                level_ref,
                visual_order,
            )
            if physical_key in seen_physical_keys:
                issues.append(
                    {
                        "issue_type": "terminal-physical-key-duplicate",
                        "severity": "high",
                        "message": f"Duplicate physical terminal key {physical_key}",
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )
            seen_physical_keys.add(physical_key)

            issues.extend(
                _field_collision_issues(
                    terminal=item,
                    region_id=rid,
                    row_id=row_id,
                    verifier_decisions=verifier_decisions,
                )
            )

            missing_evidence = _unrepresented_source_evidence(
                item,
                word_map,
            )
            item["source_evidence_coverage"] = {
                "version": "terminal-row-source-evidence-v1",
                "missing": missing_evidence,
                "complete": not bool(missing_evidence),
            }
            if missing_evidence:
                issues.append(
                    {
                        "issue_type": (
                            "terminal-visible-source-evidence-unrepresented"
                        ),
                        "severity": "high",
                        "message": (
                            f"Visible source evidence is not represented after "
                            f"field adjudication for {rid}/{row_id}: "
                            + ", ".join(
                                repr(x["text_original"])
                                for x in missing_evidence
                            )
                        ),
                        "region_id": rid,
                        "row_ids": [row_id],
                        "confidence": confidence,
                        "source_stage": "deterministic_validator",
                    }
                )

            item["region_id"] = rid
            item["strip_tag_original"] = extraction_tag or detector_tag
            item["source_side_label_original"] = _clean_text(
                extraction.get("source_side_label_original"),
                200,
            )
            item["destination_side_label_original"] = _clean_text(
                extraction.get("destination_side_label_original"),
                200,
            )
            rows.append(item)

        for raw in extraction.get("issues") or []:
            issues.append(
                _normalize_issue(
                    raw,
                    default_type="terminal-extractor-issue",
                    default_region_id=rid,
                    source_stage="extractor",
                )
            )

    for raw in verifier.get("issues") or []:
        issues.append(
            _normalize_issue(
                raw,
                default_type="terminal-verifier-issue",
                source_stage="verifier",
            )
        )

    verifier_checks = {
        str(x.get("region_id") or ""): x
        for x in (verifier.get("region_checks") or [])
        if isinstance(x, dict)
    }
    for rid in active_region_ids:
        check = verifier_checks.get(rid)
        if not check:
            issues.append(
                {
                    "issue_type": "terminal-verifier-region-missing",
                    "severity": "high",
                    "message": f"Verifier did not return region {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "deterministic_validator",
                }
            )
            continue
        if not check.get("pass") or float(check.get("confidence") or 0.0) < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(
                {
                    "issue_type": "terminal-verifier-region-failed",
                    "severity": "high",
                    "message": _clean_text(
                        check.get("notes") or f"Verifier failed {rid}",
                        1600,
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(check.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
        expected_sequence = [
            _clean_text(x, 300)
            for x in (check.get("expected_terminal_number_sequence") or [])
        ]
        verified_sequence = [
            _clean_text(x, 300)
            for x in (check.get("verified_terminal_number_sequence") or [])
        ]
        actual_sequence = [
            _clean_text(x.get("terminal_number_original"), 300)
            for x in sorted(
                [r for r in rows if r.get("region_id") == rid],
                key=lambda x: int(x.get("visual_order") or 0),
            )
        ]
        if expected_sequence != verified_sequence or verified_sequence != actual_sequence:
            issues.append(
                {
                    "issue_type": "terminal-number-sequence-mismatch",
                    "severity": "high",
                    "message": (
                        f"Terminal-number sequence mismatch in {rid}; "
                        f"expected={expected_sequence}, verified={verified_sequence}, "
                        f"actual={actual_sequence}"
                    ),
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(check.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
        if not check.get("boundary_rows_accounted_for"):
            issues.append(
                {
                    "issue_type": "terminal-boundary-row-unaccounted",
                    "severity": "high",
                    "message": f"Boundary/plate rows not accounted for in {rid}",
                    "region_id": rid,
                    "row_ids": [],
                    "confidence": float(check.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )

    verifier_flags = [
        "all_strip_like_regions_classified",
        "all_data_terminal_strips_accounted_for",
        "all_visible_terminal_rows_accounted_for",
        "all_strip_tags_supported_by_headers",
        "all_terminal_numbers_visually_supported",
        "all_published_fields_visually_supported",
    ]
    for flag in verifier_flags:
        if not verifier.get(flag):
            issues.append(
                {
                    "issue_type": f"terminal-verifier-{flag}",
                    "severity": "high",
                    "message": f"Verifier returned {flag}=false",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": float(verifier.get("confidence") or 0.0),
                    "source_stage": "deterministic_validator",
                }
            )
    if str(verifier.get("verdict") or "") != "pass":
        issues.append(
            {
                "issue_type": "terminal-verifier-blocked-page",
                "severity": "high",
                "message": "Independent verifier did not pass terminal page",
                "region_id": "",
                "row_ids": [],
                "confidence": float(verifier.get("confidence") or 0.0),
                "source_stage": "deterministic_validator",
            }
        )
    if float(verifier.get("confidence") or 0.0) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(
            {
                "issue_type": "terminal-page-confidence-below-threshold",
                "severity": "high",
                "message": "Terminal page confidence below threshold",
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


def _terminal_key(
    *,
    context: dict,
    page_id: int,
    region_id: str,
    visual_order: int,
    terminal_number: str,
    level_ref: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            [
                str(context["version_id"]),
                str(page_id),
                region_id,
                str(visual_order),
                terminal_number,
                level_ref,
            ]
        ).encode("utf-8")
    ).hexdigest()


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
                DELETE FROM public.electrical_terminals
                WHERE version_id=%s AND page_id=%s;
                """,
                (int(context["version_id"]), int(page["id"])),
            )
            for item in rows:
                region_id = _clean_text(item.get("region_id"), 120)
                strip_tag = _clean_text(
                    item.get("strip_tag_original"),
                    300,
                )
                terminal_number = _clean_text(
                    item.get("terminal_number_original"),
                    300,
                )
                level_ref = _clean_text(
                    item.get("level_ref_original"),
                    200,
                )
                visual_order = int(item.get("visual_order") or 0)
                confidence = max(
                    0.0,
                    min(1.0, float(item.get("confidence") or 0.0)),
                )
                bbox = list(item.get("bbox_pt") or [0, 0, 0, 0])
                if len(bbox) != 4:
                    bbox = [0, 0, 0, 0]
                source_word_ids = [
                    int(x)
                    for x in (item.get("source_word_ids") or [])
                    if isinstance(x, int) or str(x).isdigit()
                ]
                source_text = _clean_text(
                    " | ".join(
                        str(item.get(field) or "")
                        for field in [
                            "terminal_number_original",
                            "level_ref_original",
                            "side_a_origin_original",
                            "side_b_destination_original",
                            "wire_number_original",
                            "cable_reference_original",
                            "potential_original",
                            "conductor_color_original",
                            "conductor_cross_section_original",
                            "side_a_description_original",
                            "side_b_description_original",
                        ]
                        if str(item.get(field) or "").strip()
                    ),
                    5000,
                )
                properties = {
                    "phase": PHASE_NAME,
                    "pipeline_marker": PIPELINE_MARKER,
                    "materializer_version": MATERIALIZER_VERSION,
                    "pdf_page_number": int(page["pdf_page_number"]),
                    "sheet_code": page.get("sheet_code"),
                    "region_id": region_id,
                    "row_id": item.get("row_id"),
                    "visual_order": visual_order,
                    "row_role": item.get("row_role"),
                    "source_side_label_original": item.get(
                        "source_side_label_original"
                    ),
                    "destination_side_label_original": item.get(
                        "destination_side_label_original"
                    ),
                    "side_a_description_original": item.get(
                        "side_a_description_original"
                    ),
                    "side_b_description_original": item.get(
                        "side_b_description_original"
                    ),
                    "source_slot_ids": item.get("source_slot_ids") or [],
                    "source_word_ids": source_word_ids,
                    "verifier_overrides": item.get("verifier_overrides") or {},
                    "verifier_field_support_decisions": item.get(
                        "verifier_field_support_decisions"
                    )
                    or [],
                    "source_evidence_coverage": item.get(
                        "source_evidence_coverage"
                    )
                    or {},
                    "field_assignment_decision_version": (
                        FIELD_ASSIGNMENT_DECISION_VERSION
                    ),
                    "evidence_notes": item.get("evidence_notes") or "",
                    "detector_fingerprint": detector_fingerprint,
                    "extractor_fingerprint": extractor_fingerprints.get(
                        region_id,
                        "",
                    ),
                    "verifier_fingerprint": verifier_fingerprint,
                    "page_passed": True,
                }
                key = _terminal_key(
                    context=context,
                    page_id=int(page["id"]),
                    region_id=region_id,
                    visual_order=visual_order,
                    terminal_number=terminal_number,
                    level_ref=level_ref,
                )
                cur.execute(
                    """
                    INSERT INTO public.electrical_terminals(
                        version_id, company_id, machine_id,
                        bubble_document_id, page_id, source_entity_id,
                        terminal_key, strip_tag, terminal_number,
                        level_ref, side_a_origin, side_b_destination,
                        wire_number, cable_reference, potential,
                        conductor_color, conductor_cross_section,
                        x0, y0, x1, y1, source_text, properties,
                        confidence, extraction_method, is_verified,
                        created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, terminal_key)
                    DO UPDATE SET
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
                        int(context["version_id"]),
                        context["company_id"],
                        context["machine_id"],
                        context["bubble_document_id"],
                        int(page["id"]),
                        key,
                        strip_tag,
                        terminal_number,
                        level_ref or None,
                        _clean_text(item.get("side_a_origin_original"), 1000)
                        or None,
                        _clean_text(
                            item.get("side_b_destination_original"),
                            1000,
                        )
                        or None,
                        _clean_text(item.get("wire_number_original"), 1000)
                        or None,
                        _clean_text(
                            item.get("cable_reference_original"),
                            1000,
                        )
                        or None,
                        _clean_text(item.get("potential_original"), 500)
                        or None,
                        _clean_text(
                            item.get("conductor_color_original"),
                            500,
                        )
                        or None,
                        _clean_text(
                            item.get("conductor_cross_section_original"),
                            500,
                        )
                        or None,
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        source_text,
                        json.dumps(properties, ensure_ascii=False),
                        confidence,
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
            page_results = metadata.get("terminal_page_results") or {}
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
            total_pages = int(context["all_terminal_pages_total"])
            terminal_status = (
                "terminals_ready"
                if passed_pages == total_pages and total_pages > 0
                else ("partial" if passed_pages > 0 else "review_required")
            )
            if not page_passed:
                terminal_status = "review_required"

            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_terminals
                WHERE version_id=%s
                  AND extraction_method=%s;
                """,
                (int(context["version_id"]), EXTRACTION_METHOD),
            )
            terminal_count = int(cur.fetchone()[0] or 0)

            metadata["terminal_page_results"] = page_results
            metadata["terminal_structured_status"] = terminal_status
            metadata["terminal_pipeline_marker"] = PIPELINE_MARKER
            metadata["terminal_materializer_version"] = MATERIALIZER_VERSION
            metadata["terminal_passed_pages"] = passed_pages
            metadata["terminal_total_pages"] = total_pages
            metadata["terminal_rows"] = terminal_count

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
                    None if page_passed else "TERMINAL_REVIEW_REQUIRED",
                    None
                    if page_passed
                    else "Terminal page requires review before publication",
                    int(context["version_id"]),
                ),
            )
        conn.commit()
        return {
            "status": version_status,
            "terminal_status": terminal_status,
            "terminal_count": terminal_count,
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


def extract_electrical_terminal_page(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int] = None,
    pdf_page_numbers: Optional[list[int]] = None,
    force: bool = False,
) -> dict:
    if not TERMINALS_ENABLED:
        raise ValueError("Electrical terminal extraction is disabled")

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
        if not proposals:
            issues = [
                {
                    "issue_type": "terminal-no-geometry-proposals",
                    "severity": "high",
                    "message": "No terminal-strip table geometry was detected",
                    "region_id": "",
                    "row_ids": [],
                    "confidence": 0.0,
                    "source_stage": "geometry",
                }
            ]
            _db_replace_page_issues(
                context=context,
                page_id=int(page["id"]),
                issues=issues,
            )
            state = _db_update_version_state(
                context=context,
                page=page,
                page_passed=False,
                published_rows=0,
                blocking_count=1,
            )
            return {
                "electrical_document_id": context["electrical_document_id"],
                "electrical_version_id": context["version_id"],
                "pdf_page_number": page["pdf_page_number"],
                "sheet_code": page["sheet_code"],
                "page_passed": False,
                "proposals_count": 0,
                "strips_extracted": 0,
                "physical_terminal_rows_expected": 0,
                "published_terminal_rows": 0,
                "blocking_issue_count_this_page": 1,
                **state,
                **_db_ai_totals(context["version_id"]),
                **usage_totals,
            }

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
                    "table_bbox_pt": p["table_bbox_pt"],
                    "crop_bbox_pt": p["crop_bbox_pt"],
                    "deterministic_row_count": p[
                        "deterministic_row_count"
                    ],
                    "deterministic_column_count": p[
                        "deterministic_column_count"
                    ],
                    "geometry_slot_count": p["geometry_slot_count"],
                }
                for p in proposals
            ],
            "render_dpi": RENDER_DPI,
        }
        detector, detector_usage, detector_reused, detector_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_terminal_region_detector_v1",
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

        detector_by_region = {
            str(x.get("region_id") or ""): x
            for x in (detector.get("regions") or [])
            if isinstance(x, dict) and x.get("is_terminal_strip")
        }
        extractions: list[dict] = []
        extractor_fingerprints: dict[str, str] = {}
        region_images: dict[str, tuple[bytes, bytes]] = {}
        for proposal in proposals:
            rid = proposal["region_id"]
            dregion = detector_by_region.get(rid)
            if not dregion:
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
                int(detector.get("preferred_reading_rotation_degrees") or 90),
            )
            region_images[rid] = (crop_original, crop_rotated)
            extractor_request = {
                "page_sha256": page.get("page_sha256"),
                "pdf_page_number": page["pdf_page_number"],
                "region_id": rid,
                "region_hash": proposal["region_hash"],
                "detector_region": dregion,
                "slot_candidates": proposal["slot_candidates"],
                "render_dpi": RENDER_DPI,
            }
            result, usage, reused, fp = _cached_call(
                context=context,
                page=page,
                task_type="vision_terminal_strip_extractor_v1",
                region_hash=proposal["region_hash"],
                model=EXTRACTOR_MODEL,
                prompt_version=EXTRACTOR_PROMPT_VERSION,
                request_payload=extractor_request,
                messages=_extractor_messages(
                    page,
                    proposal,
                    dregion,
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
                "strip_extractor",
                usage,
                reused,
            )

        verifier_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "detector_fingerprint": detector_fp,
            "extractor_fingerprints": extractor_fingerprints,
            "detector": detector,
            "extractions": extractions,
            "row_source_evidence": _row_source_evidence(
                extractions,
                word_map,
            ),
            "same_text_multi_field_candidates": (
                _same_text_multi_field_candidates(extractions)
            ),
            "region_adjudication_version": REGION_ADJUDICATION_VERSION,
            "field_assignment_decision_version": (
                FIELD_ASSIGNMENT_DECISION_VERSION
            ),
            "render_dpi": RENDER_DPI,
        }
        verifier, verifier_usage, verifier_reused, verifier_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_terminal_page_verifier_v1",
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
                word_map,
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

        region_audit, _ = _adjudicate_uncovered_regions(
            detector=detector,
            verifier=verifier,
        )

        _apply_field_support_decisions(
            extractions,
            verifier.get("field_support_decisions") or [],
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

        region_stats: list[dict] = []
        detector_map = {
            str(x.get("region_id") or ""): x
            for x in (detector.get("regions") or [])
            if isinstance(x, dict) and x.get("is_terminal_strip")
        }
        extraction_map = {
            str(x.get("region_id") or ""): x
            for x in extractions
            if isinstance(x, dict)
        }
        for rid in sorted(detector_map):
            dr = detector_map[rid]
            ex = extraction_map.get(rid) or {}
            region_stats.append(
                {
                    "region_id": rid,
                    "strip_tag": ex.get("strip_tag_original")
                    or dr.get("strip_tag_original"),
                    "geometry_slots": next(
                        (
                            p["geometry_slot_count"]
                            for p in proposals
                            if p["region_id"] == rid
                        ),
                        0,
                    ),
                    "expected_terminal_rows": int(
                        dr.get("expected_terminal_rows") or 0
                    ),
                    "expected_boundary_rows": int(
                        dr.get("expected_boundary_rows") or 0
                    ),
                    "materialized_terminal_rows": len(
                        ex.get("terminals") or []
                    ),
                    "terminal_number_sequence": [
                        str(x.get("terminal_number_original") or "")
                        for x in sorted(
                            ex.get("terminals") or [],
                            key=lambda x: int(x.get("visual_order") or 0),
                        )
                    ],
                }
            )

        severity_counts = _severity_counts(issues)
        return {
            "electrical_document_id": context["electrical_document_id"],
            "electrical_version_id": context["version_id"],
            "pdf_page_number": page["pdf_page_number"],
            "sheet_code": page["sheet_code"],
            "page_type": page["page_type"],
            "language": page.get("classification_language") or "unknown",
            "page_passed": bool(page_passed),
            "proposals_count": len(proposals),
            "strips_extracted": len(extractions),
            "physical_terminal_rows_expected": sum(
                int(x.get("expected_terminal_rows") or 0)
                for x in detector_map.values()
            ),
            "published_terminal_rows": len(rows) if page_passed else 0,
            "blocking_issue_count_this_page": blocking,
            "warning_issue_count_this_page": warning,
            "ignored_non_data_regions": [
                item
                for item in region_audit
                if not item.get("is_data_terminal_strip")
            ],
            "unaccounted_data_terminal_strip_regions": (
                verifier.get("unaccounted_data_terminal_strip_regions")
                or []
            ),
            "duplicate_source_strip_tags": sorted(
                {
                    tag
                    for tag in [
                        _clean_text(
                            x.get("strip_tag_original"),
                            300,
                        )
                        for x in extractions
                    ]
                    if tag
                    and sum(
                        1
                        for ex in extractions
                        if _clean_text(
                            ex.get("strip_tag_original"),
                            300,
                        )
                        == tag
                    )
                    > 1
                }
            ),
            "region_stats": region_stats,
            "severity_counts": severity_counts,
            **state,
            **_db_ai_totals(context["version_id"]),
            **usage_totals,
        }
    finally:
        source_doc.close()


# Pure helpers intentionally exposed for fail-safe Cloud Build tests.
def _terminal_geometry_preflight_from_pdf(
    pdf_bytes: bytes,
    pdf_page_number: int,
) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[int(pdf_page_number) - 1]
        tables = _candidate_tables(page)
        return {
            "pdf_page_number": int(pdf_page_number),
            "proposal_count": len(tables),
            "table_shapes": [
                [int(t.row_count or 0), int(t.col_count or 0)]
                for t in tables
            ],
        }
    finally:
        doc.close()
