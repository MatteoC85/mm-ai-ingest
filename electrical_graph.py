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

# MachineMind Phase 2G V1
# Page-atomic, multimodal electrical graph extraction.
# The deterministic layer is geometry/evidence based and contains no page,
# language, font, manufacturer, component-tag or drawing-template dictionary.


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

GRAPH_ENABLED = (
    os.environ.get("MM_ELECTRICAL_GRAPH_ENABLED") or "0"
).strip() == "1"

DETECTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_GRAPH_DETECTOR_MODEL") or "gpt-5.4"
).strip()
EXTRACTOR_MODEL = (
    os.environ.get("MM_ELECTRICAL_GRAPH_EXTRACTOR_MODEL") or "gpt-5.4"
).strip()
VERIFIER_MODEL = (
    os.environ.get("MM_ELECTRICAL_GRAPH_VERIFIER_MODEL") or "gpt-5.4"
).strip()

DETECTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_GRAPH_DETECTOR_PROMPT_VERSION")
    or "mm-electrical-graph-detector-v1"
).strip()
EXTRACTOR_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_GRAPH_EXTRACTOR_PROMPT_VERSION")
    or "mm-electrical-graph-page-extractor-v1"
).strip()
VERIFIER_PROMPT_VERSION = (
    os.environ.get("MM_ELECTRICAL_GRAPH_VERIFIER_PROMPT_VERSION")
    or "mm-electrical-graph-page-verifier-v1"
).strip()
MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_GRAPH_MATERIALIZER_VERSION")
    or "mm-electrical-graph-materializer-v1"
).strip()

OPENAI_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_GRAPH_TIMEOUT_SECONDS", 300, 30, 600
)
FETCH_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_GRAPH_FETCH_TIMEOUT_SECONDS", 60, 10, 300
)
RENDER_DPI = _env_int(
    "MM_ELECTRICAL_GRAPH_RENDER_DPI", 240, 120, 360
)
MAX_COMPLETION_TOKENS = _env_int(
    "MM_ELECTRICAL_GRAPH_MAX_COMPLETION_TOKENS", 24000, 1000, 64000
)
ENTITY_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_GRAPH_ENTITY_MIN_CONFIDENCE", 0.84, 0.0, 1.0
)
EDGE_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_GRAPH_EDGE_MIN_CONFIDENCE", 0.86, 0.0, 1.0
)
PAGE_PASS_MIN_CONFIDENCE = _env_float(
    "MM_ELECTRICAL_GRAPH_PAGE_PASS_MIN_CONFIDENCE", 0.90, 0.0, 1.0
)
INPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_GRAPH_INPUT_USD_PER_MILLION", 0.0
)
OUTPUT_USD_PER_MILLION = _env_float(
    "MM_ELECTRICAL_GRAPH_OUTPUT_USD_PER_MILLION", 0.0
)
MAX_SOURCE_BYTES = _env_int(
    "MM_ELECTRICAL_GRAPH_MAX_SOURCE_BYTES",
    100_000_000,
    1_000_000,
    500_000_000,
)
MAX_GLYPHS_IN_PROMPT = _env_int(
    "MM_ELECTRICAL_GRAPH_MAX_GLYPHS_IN_PROMPT", 6000, 500, 20000
)
MAX_DRAWINGS_IN_PROMPT = _env_int(
    "MM_ELECTRICAL_GRAPH_MAX_DRAWINGS_IN_PROMPT", 3000, 100, 10000
)

PIPELINE_MARKER = "phase2-graph-v1-page-topology-source-snapshot"
MATERIALIZATION_PHASE = "graph_vision_v1"
EXTRACTION_METHOD = "openai_vision_graph_v1"
PAGE_TYPE = "schematic"
SEVERITIES = {"info", "warning", "high", "critical"}

REGION_KINDS = {
    "power_chain",
    "control_chain",
    "safety_chain",
    "io_interface",
    "terminal_interface",
    "off_page_reference",
    "mixed_circuit",
    "other",
}
ENTITY_TYPES = {
    "component_occurrence",
    "contact",
    "coil",
    "switch",
    "sensor",
    "actuator",
    "protective_device",
    "connector",
    "junction",
    "potential",
    "io_reference",
    "terminal_reference",
    "page_reference",
    "conductor_endpoint",
    "other",
}
COMPONENT_ENTITY_TYPES = {
    "component_occurrence",
    "contact",
    "coil",
    "switch",
    "sensor",
    "actuator",
    "protective_device",
    "connector",
}
RELATION_TYPES = {
    "electrically_connected_to",
    "carries_potential",
    "contact_of",
    "coil_of",
    "controls",
    "feedback_of",
    "linked_to_component",
    "has_pin",
}
GEOMETRY_REQUIRED_RELATIONS = {
    "electrically_connected_to",
    "carries_potential",
    "controls",
    "feedback_of",
}


def get_electrical_graph_runtime_config() -> dict:
    return {
        "enabled": bool(GRAPH_ENABLED),
        "pipeline_marker": PIPELINE_MARKER,
        "detector_model": DETECTOR_MODEL,
        "extractor_model": EXTRACTOR_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "detector_prompt_version": DETECTOR_PROMPT_VERSION,
        "extractor_prompt_version": EXTRACTOR_PROMPT_VERSION,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "entity_min_confidence": ENTITY_MIN_CONFIDENCE,
        "edge_min_confidence": EDGE_MIN_CONFIDENCE,
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
    raw = "|".join(str(part or "") for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def _canonical_reference(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.sub(r"\s+", "", text)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_conf(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _price(input_tokens: int, output_tokens: int) -> float:
    return round(
        max(0, int(input_tokens or 0)) / 1_000_000.0
        * INPUT_USD_PER_MILLION
        + max(0, int(output_tokens or 0)) / 1_000_000.0
        * OUTPUT_USD_PER_MILLION,
        6,
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


def _bbox_valid(value: Any, page: dict) -> bool:
    try:
        rect = _rect_from(value)
    except Exception:
        return False
    if rect.x0 >= rect.x1 or rect.y0 >= rect.y1:
        return False
    width = float(page.get("page_width_pt") or 0.0)
    height = float(page.get("page_height_pt") or 0.0)
    if width > 0 and height > 0:
        return bool(
            rect.x0 >= -2.0
            and rect.y0 >= -2.0
            and rect.x1 <= width + 2.0
            and rect.y1 <= height + 2.0
        )
    return True


def _data_url_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _parse_chat_content(data: dict) -> str:
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(
            f"OpenAI refused electrical graph request: {str(refusal)[:800]}"
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
            "OpenAI electrical graph call failed: "
            f"{response.status_code} {response.text[:1800]}"
        )
    data = response.json()
    text = _parse_chat_content(data)
    if not text:
        raise RuntimeError("OpenAI electrical graph call returned empty content")
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            f"Electrical graph JSON parse failed: {exc}; raw={text[:1200]}"
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
            "phase": MATERIALIZATION_PHASE,
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
            "Electrical graph extraction requires exactly one "
            "pdf_page_numbers value per request to keep publication atomic."
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
                       text_spans_json, links_json,
                       classification_language, semantic_confidence,
                       classification_metadata
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
                    "Requested page was not found among classified schematic pages"
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
                "stored_links": list(_json_obj(p[11], []) or []),
                "classification_language": str(p[12] or "unknown"),
                "semantic_confidence": float(p[13] or 0.0),
                "classification_metadata": _json_obj(p[14], {}) or {},
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

            metadata = _json_obj(row[5], {}) or {}
            source_snapshot = metadata.get("source_snapshot") or {}
            if not isinstance(source_snapshot, dict):
                source_snapshot = {}
            return {
                "electrical_document_id": int(row[0]),
                "source_filename": str(row[1] or ""),
                "version_id": int(row[2]),
                "version_no": int(row[3]),
                "version_status": str(row[4] or ""),
                "metadata": metadata,
                "pdf_page_count": int(row[6] or 0),
                "declared_sheet_count": (
                    int(row[7]) if row[7] is not None else None
                ),
                "source_sha256": str(row[8] or ""),
                "source_snapshot_uri": str(
                    source_snapshot.get("uri")
                    or metadata.get("source_snapshot_uri")
                    or ""
                ).strip(),
                "file_url": str(row[9] or ""),
                "page": page,
                "all_graph_pages_total": total_pages,
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


def _load_reference_registry(context: dict, page_id: int) -> dict:
    version_id = int(context["version_id"])
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_id, component_tag, manufacturer,
                       part_number, description, confidence
                FROM public.electrical_bom
                WHERE version_id=%s
                  AND extraction_method='openai_vision_bom_v1'
                ORDER BY page_id, id;
                """,
                (version_id,),
            )
            bom = [
                {
                    "id": int(r[0]),
                    "page_id": int(r[1]),
                    "component_tag": str(r[2] or ""),
                    "manufacturer": str(r[3] or ""),
                    "part_number": str(r[4] or ""),
                    "description": str(r[5] or ""),
                    "confidence": float(r[6] or 0.0),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT id, page_id, module_tag, channel_ref, plc_address,
                       io_type, is_safety, signal_name, description,
                       wire_reference, terminal_reference, confidence
                FROM public.electrical_io
                WHERE version_id=%s
                  AND extraction_method='openai_vision_structured_v2'
                ORDER BY page_id, id;
                """,
                (version_id,),
            )
            io_rows = [
                {
                    "id": int(r[0]),
                    "page_id": int(r[1]),
                    "module_tag": str(r[2] or ""),
                    "channel_ref": str(r[3] or ""),
                    "plc_address": str(r[4] or ""),
                    "io_type": str(r[5] or ""),
                    "is_safety": bool(r[6]),
                    "signal_name": str(r[7] or ""),
                    "description": str(r[8] or ""),
                    "wire_reference": str(r[9] or ""),
                    "terminal_reference": str(r[10] or ""),
                    "confidence": float(r[11] or 0.0),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT id, page_id, strip_tag, terminal_number, level_ref,
                       side_a_origin, side_b_destination, wire_number,
                       cable_reference, potential, confidence
                FROM public.electrical_terminals
                WHERE version_id=%s
                  AND extraction_method='openai_vision_terminals_v1'
                ORDER BY page_id, id;
                """,
                (version_id,),
            )
            terminals = [
                {
                    "id": int(r[0]),
                    "page_id": int(r[1]),
                    "strip_tag": str(r[2] or ""),
                    "terminal_number": str(r[3] or ""),
                    "level_ref": str(r[4] or ""),
                    "side_a_origin": str(r[5] or ""),
                    "side_b_destination": str(r[6] or ""),
                    "wire_number": str(r[7] or ""),
                    "cable_reference": str(r[8] or ""),
                    "potential": str(r[9] or ""),
                    "confidence": float(r[10] or 0.0),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT id, pdf_page_number, sheet_code, sheet_title, page_type
                FROM public.electrical_pages
                WHERE version_id=%s
                ORDER BY pdf_page_number;
                """,
                (version_id,),
            )
            pages = [
                {
                    "id": int(r[0]),
                    "pdf_page_number": int(r[1]),
                    "sheet_code": str(r[2] or ""),
                    "sheet_title": str(r[3] or ""),
                    "page_type": str(r[4] or ""),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT id, target_page_id, target_sheet_code,
                       target_pdf_page_number, source_label,
                       source_x0, source_y0, source_x1, source_y1,
                       target_x, target_y, relation_type, confidence
                FROM public.electrical_cross_references
                WHERE version_id=%s AND source_page_id=%s
                ORDER BY id;
                """,
                (version_id, int(page_id)),
            )
            xrefs = [
                {
                    "id": int(r[0]),
                    "target_page_id": int(r[1]) if r[1] is not None else None,
                    "target_sheet_code": str(r[2] or ""),
                    "target_pdf_page_number": (
                        int(r[3]) if r[3] is not None else None
                    ),
                    "source_label": str(r[4] or ""),
                    "source_bbox_pt": [
                        _safe_float(r[5]),
                        _safe_float(r[6]),
                        _safe_float(r[7]),
                        _safe_float(r[8]),
                    ],
                    "target_x": _safe_float(r[9]),
                    "target_y": _safe_float(r[10]),
                    "relation_type": str(r[11] or ""),
                    "confidence": float(r[12] or 0.0),
                }
                for r in cur.fetchall()
            ]
        return {
            "bom": bom,
            "io": io_rows,
            "terminals": terminals,
            "pages": pages,
            "cross_references": xrefs,
        }
    finally:
        conn.close()


def _word_registry(page: dict) -> list[dict]:
    output: list[dict] = []
    for index, word in enumerate(page.get("words") or [], start=1):
        if not isinstance(word, (list, tuple)) or len(word) < 5:
            continue
        try:
            bbox = [round(float(word[i]), 2) for i in range(4)]
        except Exception:
            continue
        text = str(word[4] or "").replace("\x00", "")
        if not text.strip():
            continue
        output.append({
            "word_id": index,
            "bbox_pt": bbox,
            "text_original": text,
        })
    return output


def _glyph_registry(source_page: fitz.Page) -> list[dict]:
    raw = source_page.get_text("rawdict") or {}
    glyphs: list[dict] = []
    glyph_id = 0
    for block in raw.get("blocks") or []:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            direction = line.get("dir") or [1.0, 0.0]
            for span in line.get("spans") or []:
                font_name = str(span.get("font") or "")
                for char in span.get("chars") or []:
                    text = str(char.get("c") or "")
                    if not text or text == "\x00":
                        continue
                    try:
                        bbox = [round(float(x), 3) for x in char["bbox"]]
                    except Exception:
                        continue
                    origin = char.get("origin") or [bbox[0], bbox[3]]
                    glyph_id += 1
                    glyphs.append({
                        "glyph_id": glyph_id,
                        "text_original": text,
                        "bbox_pt": bbox,
                        "origin_pt": [
                            round(float(origin[0]), 3),
                            round(float(origin[1]), 3),
                        ],
                        "direction": [
                            round(float(direction[0]), 4),
                            round(float(direction[1]), 4),
                        ],
                        # Font is audit only and is never used for classification.
                        "font_audit": font_name,
                    })
    return glyphs


def _drawing_registry(source_page: fitz.Page) -> list[dict]:
    output: list[dict] = []
    for index, drawing in enumerate(source_page.get_drawings() or [], start=1):
        rect_value = drawing.get("rect")
        try:
            rect = _rect_from(rect_value)
        except Exception:
            continue
        item_types: dict[str, int] = {}
        for item in drawing.get("items") or []:
            if not item:
                continue
            item_type = str(item[0])
            item_types[item_type] = item_types.get(item_type, 0) + 1
        output.append({
            "drawing_id": index,
            "bbox_pt": _rect_list(rect, 3),
            "item_count": len(drawing.get("items") or []),
            "item_types": item_types,
            "width": round(_safe_float(drawing.get("width")), 3),
            "closed": bool(drawing.get("closePath")),
            "has_fill": drawing.get("fill") is not None,
        })
    return output


def _render_page(source_doc: fitz.Document, page_index: int, rotation: int) -> bytes:
    page = source_doc[page_index]
    pix = page.get_pixmap(
        matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0)
        .prerotate(rotation),
        alpha=False,
    )
    return pix.tobytes("png")


def _issue_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issue_type": {"type": "string"},
            "severity": {"type": "string", "enum": sorted(SEVERITIES)},
            "message": {"type": "string"},
            "entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 500,
            },
            "edge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 500,
            },
            "confidence": {"type": "number"},
        },
        "required": [
            "issue_type",
            "severity",
            "message",
            "entity_ids",
            "edge_ids",
            "confidence",
        ],
    }


def _bbox_schema() -> dict:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
    }


def _detector_schema() -> dict:
    return {
        "name": "electrical_graph_page_detector_v1",
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
                "all_visible_circuit_regions_accounted_for": {
                    "type": "boolean"
                },
                "regions": {
                    "type": "array",
                    "maxItems": 60,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region_id": {"type": "string"},
                            "region_kind": {
                                "type": "string",
                                "enum": sorted(REGION_KINDS),
                            },
                            "bbox_pt": _bbox_schema(),
                            "visible_component_count": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 1000,
                            },
                            "visible_connection_count": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 3000,
                            },
                            "confidence": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "region_id",
                            "region_kind",
                            "bbox_pt",
                            "visible_component_count",
                            "visible_connection_count",
                            "confidence",
                            "notes",
                        ],
                    },
                },
                "uncovered_visual_regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 50,
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
                "language",
                "preferred_reading_rotation_degrees",
                "all_visible_circuit_regions_accounted_for",
                "regions",
                "uncovered_visual_regions",
                "confidence",
                "issues",
            ],
        },
    }


def _entity_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "occurrence_id": {"type": "string"},
            "region_id": {"type": "string"},
            "entity_type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
            "subtype": {"type": "string"},
            "tag_original": {"type": "string"},
            "label_original": {"type": "string"},
            "description_original": {"type": "string"},
            "function_text_original": {"type": "string"},
            "symbol_code": {"type": "string"},
            "location_code": {"type": "string"},
            "reference_value_original": {"type": "string"},
            "reference_context_original": {"type": "string"},
            "bbox_pt": _bbox_schema(),
            "source_glyph_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 1000,
            },
            "source_word_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 500,
            },
            "confidence": {"type": "number"},
            "evidence_notes": {"type": "string"},
        },
        "required": [
            "occurrence_id",
            "region_id",
            "entity_type",
            "subtype",
            "tag_original",
            "label_original",
            "description_original",
            "function_text_original",
            "symbol_code",
            "location_code",
            "reference_value_original",
            "reference_context_original",
            "bbox_pt",
            "source_glyph_ids",
            "source_word_ids",
            "confidence",
            "evidence_notes",
        ],
    }


def _edge_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "edge_id": {"type": "string"},
            "source_occurrence_id": {"type": "string"},
            "target_occurrence_id": {"type": "string"},
            "relation_type": {
                "type": "string",
                "enum": sorted(RELATION_TYPES),
            },
            "is_directed": {"type": "boolean"},
            "potential_original": {"type": "string"},
            "wire_reference_original": {"type": "string"},
            "bbox_pt": _bbox_schema(),
            "source_glyph_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 1000,
            },
            "source_drawing_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 1000,
            },
            "source_link_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 300,
            },
            "confidence": {"type": "number"},
            "evidence_notes": {"type": "string"},
        },
        "required": [
            "edge_id",
            "source_occurrence_id",
            "target_occurrence_id",
            "relation_type",
            "is_directed",
            "potential_original",
            "wire_reference_original",
            "bbox_pt",
            "source_glyph_ids",
            "source_drawing_ids",
            "source_link_ids",
            "confidence",
            "evidence_notes",
        ],
    }


def _extractor_schema() -> dict:
    return {
        "name": "electrical_graph_page_extractor_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "entities": {
                    "type": "array",
                    "items": _entity_schema(),
                    "maxItems": 1500,
                },
                "edges": {
                    "type": "array",
                    "items": _edge_schema(),
                    "maxItems": 4000,
                },
                "unresolved_visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 200,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 100,
                },
            },
            "required": [
                "page_id",
                "entities",
                "edges",
                "unresolved_visual_evidence",
                "confidence",
                "issues",
            ],
        },
    }


def _verifier_schema() -> dict:
    return {
        "name": "electrical_graph_page_verifier_v1",
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
                "all_visible_entities_accounted_for": {"type": "boolean"},
                "all_visible_connections_accounted_for": {"type": "boolean"},
                "all_entity_text_visually_supported": {"type": "boolean"},
                "all_connection_geometry_supported": {"type": "boolean"},
                "all_references_resolved_or_explicitly_unresolved": {
                    "type": "boolean"
                },
                "duplicates_preserved": {"type": "boolean"},
                "verified_entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 1500,
                },
                "verified_edge_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4000,
                },
                "rejected_entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 500,
                },
                "rejected_edge_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 1000,
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _issue_schema(),
                    "maxItems": 100,
                },
            },
            "required": [
                "page_id",
                "verdict",
                "all_visible_entities_accounted_for",
                "all_visible_connections_accounted_for",
                "all_entity_text_visually_supported",
                "all_connection_geometry_supported",
                "all_references_resolved_or_explicitly_unresolved",
                "duplicates_preserved",
                "verified_entity_ids",
                "verified_edge_ids",
                "rejected_entity_ids",
                "rejected_edge_ids",
                "confidence",
                "issues",
            ],
        },
    }


def _candidate_reference_registry(page: dict, registry: dict) -> dict:
    page_signature = _canonical_reference(page.get("raw_text"))

    def appears(value: Any) -> bool:
        signature = _canonical_reference(value)
        return bool(signature and signature in page_signature)

    bom = [row for row in registry.get("bom") or [] if appears(row.get("component_tag"))]
    io_rows = [row for row in registry.get("io") or [] if appears(row.get("module_tag"))]
    terminals = [
        row
        for row in registry.get("terminals") or []
        if appears(row.get("strip_tag"))
    ]
    page_rows = [
        row
        for row in registry.get("pages") or []
        if appears(row.get("sheet_code"))
    ]
    return {
        "bom": bom,
        "io": io_rows,
        "terminals": terminals,
        "pages": page_rows,
        "cross_references": registry.get("cross_references") or [],
    }


def _detector_messages(
    page: dict,
    image_original: bytes,
    image_rotated: bytes,
    drawing_summary: dict,
    link_summary: dict,
) -> list[dict]:
    system = (
        "You are the visual perception stage of an industrial electrical "
        "schematic graph reader. The source can use any language, font, CAD "
        "system, orientation or drawing standard. Read the complete page as a "
        "human electrical engineer would. Partition the page into coherent "
        "electrical circuit regions without relying on keywords. Distinguish "
        "power, control, safety, I/O, terminal and off-page-reference areas. "
        "Count visible physical component/symbol occurrences and connection "
        "paths. Repeated tags can be valid separate occurrences. Do not infer "
        "hidden components and do not treat the title block or coordinate grid "
        "as circuit content."
    )
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "page_type": page.get("page_type"),
        "page_width_pt": page.get("page_width_pt"),
        "page_height_pt": page.get("page_height_pt"),
        "vector_word_count": len(page.get("words") or []),
        "drawing_summary": drawing_summary,
        "link_summary": link_summary,
    }
    content = [
        {
            "type": "text",
            "text": (
                "Identify all circuit regions on the page and report any "
                "visible circuit content that cannot be assigned to one of "
                "them.\n\n" + json.dumps(request, ensure_ascii=False)
            ),
        },
        {"type": "text", "text": "FULL PAGE ORIGINAL"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(image_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "FULL PAGE ROTATED 90 DEGREES"},
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
    *,
    page: dict,
    detector: dict,
    words: list[dict],
    glyphs: list[dict],
    drawings: list[dict],
    links: list[dict],
    reference_candidates: dict,
    image_original: bytes,
    image_rotated: bytes,
) -> list[dict]:
    system = (
        "You are the extraction stage of a page-atomic industrial electrical "
        "graph reader. Work semantically from the complete page images while "
        "using glyph, word, drawing and link registries as exact evidence. The "
        "source may use any language or font. Extract every visible physical "
        "electrical entity occurrence required to understand the circuit: "
        "components, contacts, coils, switches, sensors, actuators, protective "
        "devices, connectors, junctions, potentials, I/O references, terminal "
        "references and page references. Preserve printed tags, punctuation, "
        "pin/channel values and repeated occurrences exactly. Use source glyph "
        "and word IDs only when they truly support the entity text. "
        "Extract graph edges only when their endpoints and relation are visible. "
        "An electrically_connected_to, carries_potential, controls or feedback_of "
        "edge must cite visible drawing IDs or a PDF link ID; proximity alone is "
        "not electrical continuity. Do not invent invisible wire crossings as "
        "junctions. External certified BOM, I/O, terminal and page rows are "
        "reference candidates, not permission to invent a match. For a reference "
        "entity, place the module/strip/sheet tag in tag_original and the exact "
        "channel, wire, terminal or coordinate value in reference_value_original."
    )
    compact_glyphs = [
        {
            "glyph_id": item["glyph_id"],
            "text_original": item["text_original"],
            "bbox_pt": item["bbox_pt"],
            "origin_pt": item["origin_pt"],
            "direction": item["direction"],
        }
        for item in glyphs[:MAX_GLYPHS_IN_PROMPT]
    ]
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "page_width_pt": page.get("page_width_pt"),
        "page_height_pt": page.get("page_height_pt"),
        "detector": detector,
        "vector_words": words,
        "source_glyphs": compact_glyphs,
        "glyph_registry_complete": len(glyphs) <= MAX_GLYPHS_IN_PROMPT,
        "drawing_registry": drawings[:MAX_DRAWINGS_IN_PROMPT],
        "drawing_registry_complete": len(drawings) <= MAX_DRAWINGS_IN_PROMPT,
        "pdf_link_registry": links,
        "certified_reference_candidates": reference_candidates,
        "entity_min_confidence": ENTITY_MIN_CONFIDENCE,
        "edge_min_confidence": EDGE_MIN_CONFIDENCE,
    }
    content = [
        {
            "type": "text",
            "text": (
                "Build the complete page-local electrical graph. Every ID must "
                "be unique within this response. Account explicitly for any "
                "visual evidence that remains unresolved.\n\n"
                + json.dumps(request, ensure_ascii=False)
            ),
        },
        {"type": "text", "text": "FULL PAGE ORIGINAL"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(image_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "FULL PAGE ROTATED 90 DEGREES"},
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


def _verifier_messages(
    *,
    page: dict,
    detector: dict,
    extraction: dict,
    resolution: dict,
    drawing_count: int,
    link_count: int,
    image_original: bytes,
    image_rotated: bytes,
) -> list[dict]:
    system = (
        "You are the independent visual verifier of an industrial electrical "
        "page graph. Re-read the full page without trusting the extractor. "
        "Verify exact physical occurrence identity, printed tags, symbol roles, "
        "connection topology, wire/potential labels and off-page references. "
        "Repeated component tags must remain separate occurrences. A visible "
        "wire crossing is not a junction unless the drawing visibly supports a "
        "junction. Every geometry-dependent connection must be supported by a "
        "drawing or PDF-link evidence ID. Certified BOM, I/O, terminal and page "
        "matches must be exact and grounded in both the visible page and the "
        "provided certified registry resolution. Return pass only when the full "
        "post-resolution graph is safe to publish atomically."
    )
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "detector": detector,
        "candidate_graph": extraction,
        "deterministic_reference_resolution": resolution,
        "available_drawing_count": drawing_count,
        "available_link_count": link_count,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
    }
    content = [
        {
            "type": "text",
            "text": (
                "Audit every entity and edge ID. Echo exactly the IDs that are "
                "visually and deterministically supported. Reject rather than "
                "repair ambiguous topology.\n\n"
                + json.dumps(request, ensure_ascii=False)
            ),
        },
        {"type": "text", "text": "FULL PAGE ORIGINAL"},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url_png(image_original),
                "detail": "original",
            },
        },
        {"type": "text", "text": "FULL PAGE ROTATED 90 DEGREES"},
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


def _reference_value_signatures(value: Any, tag: Any = "") -> set[str]:
    signature = _canonical_reference(value)
    tag_signature = _canonical_reference(tag)
    candidates = {signature} if signature else set()
    if signature and tag_signature and signature.startswith(tag_signature):
        candidates.add(signature[len(tag_signature):])
    expanded = set(candidates)
    for candidate in candidates:
        stripped = candidate.strip("-:/.\\")
        if stripped:
            expanded.add(stripped)
    return {value for value in expanded if value}


def _reference_value_matches(
    value: Any,
    row: dict,
    fields: tuple[str, ...],
    tag: Any = "",
) -> bool:
    signatures = _reference_value_signatures(value, tag)
    if not signatures:
        return False
    return any(
        _canonical_reference(row.get(field)) in signatures
        for field in fields
        if _canonical_reference(row.get(field))
    )


def _resolve_references(extraction: dict, registry: dict) -> dict:
    bom_by_tag: dict[str, list[dict]] = {}
    for row in registry.get("bom") or []:
        key = _canonical_reference(row.get("component_tag"))
        if key:
            bom_by_tag.setdefault(key, []).append(row)

    io_by_module: dict[str, list[dict]] = {}
    for row in registry.get("io") or []:
        key = _canonical_reference(row.get("module_tag"))
        if key:
            io_by_module.setdefault(key, []).append(row)

    terminal_by_strip: dict[str, list[dict]] = {}
    for row in registry.get("terminals") or []:
        key = _canonical_reference(row.get("strip_tag"))
        if key:
            terminal_by_strip.setdefault(key, []).append(row)

    pages_by_sheet: dict[str, list[dict]] = {}
    for row in registry.get("pages") or []:
        key = _canonical_reference(row.get("sheet_code"))
        if key:
            pages_by_sheet.setdefault(key, []).append(row)

    entity_resolutions: list[dict] = []
    unresolved_reference_entity_ids: list[str] = []
    totals = {"bom": 0, "io": 0, "terminal": 0, "page": 0}

    for entity in extraction.get("entities") or []:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        tag = _clean_text(entity.get("tag_original"), 500)
        tag_key = _canonical_reference(tag)
        reference_value = _clean_text(
            entity.get("reference_value_original"), 500
        )
        record = {
            "occurrence_id": occurrence_id,
            "entity_type": entity_type,
            "tag_original": tag,
            "reference_value_original": reference_value,
            "bom_matches": [],
            "io_matches": [],
            "terminal_matches": [],
            "page_matches": [],
            "resolved": True,
            "reason": "",
        }

        if tag_key and entity_type in COMPONENT_ENTITY_TYPES:
            record["bom_matches"] = list(bom_by_tag.get(tag_key) or [])
            totals["bom"] += len(record["bom_matches"])

        if entity_type == "io_reference":
            candidates = list(io_by_module.get(tag_key) or [])
            if reference_value:
                candidates = [
                    row
                    for row in candidates
                    if _reference_value_matches(
                        reference_value,
                        row,
                        (
                            "channel_ref",
                            "wire_reference",
                            "terminal_reference",
                            "plc_address",
                        ),
                        tag,
                    )
                ]
            if len(candidates) == 1:
                record["io_matches"] = candidates
                totals["io"] += 1
            else:
                record["resolved"] = False
                record["reason"] = (
                    "I/O reference did not resolve to exactly one certified row"
                )

        elif entity_type == "terminal_reference":
            candidates = list(terminal_by_strip.get(tag_key) or [])
            if reference_value:
                candidates = [
                    row
                    for row in candidates
                    if _reference_value_matches(
                        reference_value,
                        row,
                        ("terminal_number", "wire_number", "potential"),
                        tag,
                    )
                ]
            if len(candidates) == 1:
                record["terminal_matches"] = candidates
                totals["terminal"] += 1
            else:
                record["resolved"] = False
                record["reason"] = (
                    "Terminal reference did not resolve to exactly one "
                    "certified terminal row"
                )

        elif entity_type == "page_reference":
            sheet_signatures = {tag_key} if tag_key else set()
            if "." in tag_key:
                sheet_signatures.add(tag_key.split(".", 1)[0])
            candidates = [
                row
                for signature in sheet_signatures
                for row in (pages_by_sheet.get(signature) or [])
            ]
            candidates = list({int(row["id"]): row for row in candidates}.values())
            if len(candidates) != 1:
                # PDF links are a second independent exact source. Match either
                # the target sheet or the complete printed source label.
                visible_reference_signatures = {
                    value for value in (
                        tag_key,
                        _canonical_reference(reference_value),
                        _canonical_reference(
                            entity.get("reference_context_original")
                        ),
                    ) if value
                }
                xref_candidates = [
                    row
                    for row in registry.get("cross_references") or []
                    if row.get("target_page_id") is not None
                    and (
                        _canonical_reference(row.get("target_sheet_code"))
                        in sheet_signatures
                        or _canonical_reference(row.get("source_label"))
                        in visible_reference_signatures
                    )
                ]
                page_ids = sorted(
                    {int(row["target_page_id"]) for row in xref_candidates}
                )
                if len(page_ids) == 1:
                    candidates = [
                        row
                        for row in registry.get("pages") or []
                        if int(row.get("id") or 0) == page_ids[0]
                    ]
            if len(candidates) == 1:
                record["page_matches"] = candidates
                totals["page"] += 1
            else:
                record["resolved"] = False
                record["reason"] = (
                    "Page reference did not resolve to exactly one indexed page"
                )

        if not record["resolved"]:
            unresolved_reference_entity_ids.append(occurrence_id)
        entity_resolutions.append(record)

    return {
        "version": "exact-certified-reference-resolution-v1",
        "entity_resolutions": entity_resolutions,
        "unresolved_reference_entity_ids": unresolved_reference_entity_ids,
        "match_counts": totals,
        "all_reference_entities_resolved": not bool(
            unresolved_reference_entity_ids
        ),
    }


def _normalize_issue(
    issue: Any,
    *,
    default_type: str,
    source_stage: str,
) -> dict:
    raw = issue if isinstance(issue, dict) else {}
    severity = str(raw.get("severity") or "warning").lower()
    if severity not in SEVERITIES:
        severity = "warning"
    return {
        "issue_type": _clean_text(
            raw.get("issue_type") or default_type,
            180,
        ),
        "severity": severity,
        "message": _clean_text(
            raw.get("message") or "Electrical graph extraction issue",
            1600,
        ),
        "entity_ids": [
            _clean_text(value, 160)
            for value in (raw.get("entity_ids") or [])
            if _clean_text(value, 160)
        ][:500],
        "edge_ids": [
            _clean_text(value, 160)
            for value in (raw.get("edge_ids") or [])
            if _clean_text(value, 160)
        ][:500],
        "confidence": _clamp_conf(raw.get("confidence")),
        "source_stage": source_stage,
    }


def _local_issue(
    *,
    issue_type: str,
    message: str,
    entity_ids: Optional[list[str]] = None,
    edge_ids: Optional[list[str]] = None,
    confidence: float = 0.0,
    severity: str = "high",
    source_stage: str = "deterministic_validator",
) -> dict:
    return {
        "issue_type": issue_type,
        "severity": severity if severity in SEVERITIES else "high",
        "message": _clean_text(message, 1600),
        "entity_ids": entity_ids or [],
        "edge_ids": edge_ids or [],
        "confidence": _clamp_conf(confidence),
        "source_stage": source_stage,
    }


def _validate_candidate_graph(
    *,
    page: dict,
    detector: dict,
    extraction: dict,
    verifier: dict,
    resolution: dict,
    glyphs: list[dict],
    words: list[dict],
    drawings: list[dict],
    links: list[dict],
) -> tuple[bool, list[dict], list[dict], list[dict]]:
    issues: list[dict] = []
    entities = [
        item for item in (extraction.get("entities") or [])
        if isinstance(item, dict)
    ]
    edges = [
        item for item in (extraction.get("edges") or [])
        if isinstance(item, dict)
    ]

    for raw in detector.get("issues") or []:
        issues.append(_normalize_issue(
            raw,
            default_type="graph-detector-issue",
            source_stage="detector",
        ))
    for raw in extraction.get("issues") or []:
        issues.append(_normalize_issue(
            raw,
            default_type="graph-extractor-issue",
            source_stage="extractor",
        ))
    for raw in verifier.get("issues") or []:
        issues.append(_normalize_issue(
            raw,
            default_type="graph-verifier-issue",
            source_stage="verifier",
        ))

    if int(detector.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-detector-page-id-mismatch",
            message="Detector returned a different page_id",
        ))
    if not detector.get("all_visible_circuit_regions_accounted_for"):
        issues.append(_local_issue(
            issue_type="graph-detector-region-coverage-failed",
            message="Detector reports uncovered visible circuit regions",
            confidence=detector.get("confidence") or 0.0,
        ))
    if detector.get("uncovered_visual_regions"):
        issues.append(_local_issue(
            issue_type="graph-detector-uncovered-visual-regions",
            message="Detector returned non-empty uncovered_visual_regions",
            confidence=detector.get("confidence") or 0.0,
        ))
    if _clamp_conf(detector.get("confidence")) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-detector-confidence-below-threshold",
            message="Detector confidence is below page threshold",
            confidence=detector.get("confidence") or 0.0,
        ))

    region_ids: list[str] = []
    for region in detector.get("regions") or []:
        if not isinstance(region, dict):
            continue
        rid = _clean_text(region.get("region_id"), 160)
        if not rid or rid in region_ids:
            issues.append(_local_issue(
                issue_type="graph-region-id-invalid",
                message="Missing or duplicate detector region_id",
            ))
        region_ids.append(rid)
        if _clean_text(region.get("region_kind"), 120) not in REGION_KINDS:
            issues.append(_local_issue(
                issue_type="graph-region-kind-invalid",
                message=f"Invalid graph region kind for {rid}",
            ))
        if not _bbox_valid(region.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-region-bbox-invalid",
                message=f"Invalid graph region bbox for {rid}",
            ))
        if _clamp_conf(region.get("confidence")) < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-region-confidence-below-threshold",
                message=f"Detector region confidence below threshold for {rid}",
                confidence=region.get("confidence") or 0.0,
            ))

    if int(extraction.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-extractor-page-id-mismatch",
            message="Extractor returned a different page_id",
        ))
    if _clamp_conf(extraction.get("confidence")) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-extractor-confidence-below-threshold",
            message="Extractor confidence is below page threshold",
            confidence=extraction.get("confidence") or 0.0,
        ))
    if extraction.get("unresolved_visual_evidence"):
        issues.append(_local_issue(
            issue_type="graph-unresolved-visual-evidence",
            message="Extractor returned unresolved circuit evidence",
            confidence=extraction.get("confidence") or 0.0,
        ))

    valid_glyph_ids = {int(item["glyph_id"]) for item in glyphs}
    valid_word_ids = {int(item["word_id"]) for item in words}
    valid_drawing_ids = {int(item["drawing_id"]) for item in drawings}
    valid_link_ids = {int(item.get("id") or 0) for item in links}

    occurrence_ids: list[str] = []
    entity_by_id: dict[str, dict] = {}
    for entity in entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        confidence = _clamp_conf(entity.get("confidence"))
        if not occurrence_id or occurrence_id in occurrence_ids:
            issues.append(_local_issue(
                issue_type="graph-entity-id-invalid",
                message="Missing or duplicate entity occurrence_id",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
        occurrence_ids.append(occurrence_id)
        entity_by_id[occurrence_id] = entity
        if entity_type not in ENTITY_TYPES:
            issues.append(_local_issue(
                issue_type="graph-entity-type-invalid",
                message=f"Invalid entity type in {occurrence_id}",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        if _clean_text(entity.get("region_id"), 160) not in set(region_ids):
            issues.append(_local_issue(
                issue_type="graph-entity-region-invalid",
                message=f"Entity {occurrence_id} references an unknown region",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        if not _bbox_valid(entity.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-entity-bbox-invalid",
                message=f"Invalid entity bbox in {occurrence_id}",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        if confidence < ENTITY_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-entity-confidence-below-threshold",
                message=f"Low-confidence entity {occurrence_id}",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        source_glyph_ids = {
            int(value)
            for value in (entity.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        source_word_ids = {
            int(value)
            for value in (entity.get("source_word_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if source_glyph_ids - valid_glyph_ids or source_word_ids - valid_word_ids:
            issues.append(_local_issue(
                issue_type="graph-entity-evidence-id-invalid",
                message=f"Entity {occurrence_id} cites invalid glyph/word IDs",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        substantive_text = any(_clean_text(entity.get(field), 1000) for field in (
            "tag_original", "label_original", "description_original",
            "function_text_original", "reference_value_original",
        ))
        if substantive_text and not (source_glyph_ids or source_word_ids):
            issues.append(_local_issue(
                issue_type="graph-entity-text-evidence-missing",
                message=f"Entity {occurrence_id} has text without source evidence",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))
        if entity_type in COMPONENT_ENTITY_TYPES and not _clean_text(
            entity.get("tag_original"), 500
        ):
            issues.append(_local_issue(
                issue_type="graph-component-tag-missing",
                message=f"Component-like entity {occurrence_id} has no visible tag",
                entity_ids=[occurrence_id],
                confidence=confidence,
            ))

    edge_ids: list[str] = []
    for edge in edges:
        edge_id = _clean_text(edge.get("edge_id"), 160)
        relation_type = _clean_text(edge.get("relation_type"), 120)
        source_id = _clean_text(edge.get("source_occurrence_id"), 160)
        target_id = _clean_text(edge.get("target_occurrence_id"), 160)
        confidence = _clamp_conf(edge.get("confidence"))
        if not edge_id or edge_id in edge_ids:
            issues.append(_local_issue(
                issue_type="graph-edge-id-invalid",
                message="Missing or duplicate edge_id",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
        edge_ids.append(edge_id)
        if relation_type not in RELATION_TYPES:
            issues.append(_local_issue(
                issue_type="graph-edge-relation-invalid",
                message=f"Invalid relation type in edge {edge_id}",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        if source_id not in entity_by_id or target_id not in entity_by_id:
            issues.append(_local_issue(
                issue_type="graph-edge-endpoint-missing",
                message=f"Edge {edge_id} references a missing entity",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        if source_id and source_id == target_id:
            issues.append(_local_issue(
                issue_type="graph-edge-self-reference",
                message=f"Edge {edge_id} connects an entity to itself",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        if not _bbox_valid(edge.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-edge-bbox-invalid",
                message=f"Invalid edge bbox in {edge_id}",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        if confidence < EDGE_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-edge-confidence-below-threshold",
                message=f"Low-confidence edge {edge_id}",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        drawing_ids = {
            int(value)
            for value in (edge.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        link_ids = {
            int(value)
            for value in (edge.get("source_link_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        glyph_ids = {
            int(value)
            for value in (edge.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if (
            drawing_ids - valid_drawing_ids
            or link_ids - valid_link_ids
            or glyph_ids - valid_glyph_ids
        ):
            issues.append(_local_issue(
                issue_type="graph-edge-evidence-id-invalid",
                message=f"Edge {edge_id} cites invalid evidence IDs",
                edge_ids=[edge_id],
                confidence=confidence,
            ))
        if (
            relation_type in GEOMETRY_REQUIRED_RELATIONS
            and not drawing_ids
            and not link_ids
        ):
            issues.append(_local_issue(
                issue_type="graph-edge-geometry-evidence-missing",
                message=(
                    f"Geometry-dependent edge {edge_id} has no drawing or "
                    "PDF-link evidence"
                ),
                edge_ids=[edge_id],
                confidence=confidence,
            ))

    if not resolution.get("all_reference_entities_resolved"):
        issues.append(_local_issue(
            issue_type="graph-reference-resolution-failed",
            message="One or more exact external references are unresolved",
            entity_ids=resolution.get("unresolved_reference_entity_ids") or [],
            confidence=0.0,
        ))

    if int(verifier.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-verifier-page-id-mismatch",
            message="Verifier returned a different page_id",
        ))
    verified_entity_ids = [
        _clean_text(value, 160)
        for value in (verifier.get("verified_entity_ids") or [])
    ]
    verified_edge_ids = [
        _clean_text(value, 160)
        for value in (verifier.get("verified_edge_ids") or [])
    ]
    if set(verified_entity_ids) != set(occurrence_ids) or len(
        verified_entity_ids
    ) != len(occurrence_ids):
        issues.append(_local_issue(
            issue_type="graph-verifier-entity-accounting-mismatch",
            message="Verifier did not verify exactly every extracted entity ID",
            confidence=verifier.get("confidence") or 0.0,
        ))
    if set(verified_edge_ids) != set(edge_ids) or len(verified_edge_ids) != len(
        edge_ids
    ):
        issues.append(_local_issue(
            issue_type="graph-verifier-edge-accounting-mismatch",
            message="Verifier did not verify exactly every extracted edge ID",
            confidence=verifier.get("confidence") or 0.0,
        ))
    if verifier.get("rejected_entity_ids") or verifier.get("rejected_edge_ids"):
        issues.append(_local_issue(
            issue_type="graph-verifier-rejected-candidates",
            message="Verifier rejected one or more graph candidates",
            entity_ids=verifier.get("rejected_entity_ids") or [],
            edge_ids=verifier.get("rejected_edge_ids") or [],
            confidence=verifier.get("confidence") or 0.0,
        ))
    for flag in (
        "all_visible_entities_accounted_for",
        "all_visible_connections_accounted_for",
        "all_entity_text_visually_supported",
        "all_connection_geometry_supported",
        "all_references_resolved_or_explicitly_unresolved",
        "duplicates_preserved",
    ):
        if not verifier.get(flag):
            issues.append(_local_issue(
                issue_type=f"graph-verifier-{flag}",
                message=f"Verifier returned {flag}=false",
                confidence=verifier.get("confidence") or 0.0,
            ))
    if str(verifier.get("verdict") or "") != "pass":
        issues.append(_local_issue(
            issue_type="graph-verifier-blocked-page",
            message="Independent verifier did not pass the graph page",
            confidence=verifier.get("confidence") or 0.0,
        ))
    if _clamp_conf(verifier.get("confidence")) < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-verifier-confidence-below-threshold",
            message="Verifier confidence is below page threshold",
            confidence=verifier.get("confidence") or 0.0,
        ))

    if not entities:
        issues.append(_local_issue(
            issue_type="graph-no-entities",
            message="No graph entities were extracted from the schematic page",
        ))
    if not edges:
        issues.append(_local_issue(
            issue_type="graph-no-edges",
            message="No graph edges were extracted from the schematic page",
        ))

    blocking = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    return not blocking and bool(entities) and bool(edges), entities, edges, issues


def _build_materialization_plan(
    *,
    context: dict,
    page: dict,
    entities: list[dict],
    edges: list[dict],
    resolution: dict,
    detector_fingerprint: str,
    extractor_fingerprint: str,
    verifier_fingerprint: str,
) -> dict:
    entity_specs: dict[str, dict] = {}
    edge_specs: dict[str, dict] = {}
    occurrence_key_by_id: dict[str, str] = {}
    resolution_by_id = {
        row["occurrence_id"]: row
        for row in resolution.get("entity_resolutions") or []
    }

    def add_entity(spec: dict) -> None:
        key = spec["entity_key"]
        if key not in entity_specs:
            entity_specs[key] = spec

    def add_edge(spec: dict) -> None:
        key = spec["edge_key"]
        if key not in edge_specs:
            edge_specs[key] = spec

    for entity in entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        tag = _clean_text(entity.get("tag_original"), 500)
        canonical_tag = _canonical_reference(tag)
        parent_key = None
        if entity_type in COMPONENT_ENTITY_TYPES and canonical_tag:
            parent_key = f"graph:canonical:component:{canonical_tag}"
            add_entity({
                "entity_key": parent_key,
                "page_id": None,
                "parent_key": None,
                "entity_type": "component",
                "subtype": entity_type,
                "tag": tag,
                "label": tag,
                "description": "",
                "function_text": "",
                "symbol_code": "",
                "location_code": "",
                "bbox_pt": [None, None, None, None],
                "source_text": tag,
                "confidence": _clamp_conf(entity.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "pipeline_marker": PIPELINE_MARKER,
                    "materializer_version": MATERIALIZER_VERSION,
                    "canonical_reference": canonical_tag,
                    "canonical_scope": "version_component_tag",
                },
            })

        occurrence_key = (
            f"graph:page:{int(page['id'])}:occurrence:{occurrence_id}"
        )
        occurrence_key_by_id[occurrence_id] = occurrence_key
        bbox = list(entity.get("bbox_pt") or [None, None, None, None])
        source_text = " | ".join(
            value for value in [
                tag,
                _clean_text(entity.get("label_original"), 1000),
                _clean_text(entity.get("description_original"), 2000),
                _clean_text(entity.get("function_text_original"), 2000),
                _clean_text(entity.get("reference_value_original"), 500),
            ] if value
        )
        add_entity({
            "entity_key": occurrence_key,
            "page_id": int(page["id"]),
            "parent_key": parent_key,
            "entity_type": entity_type,
            "subtype": _clean_text(entity.get("subtype"), 200) or None,
            "tag": tag or None,
            "label": _clean_text(entity.get("label_original"), 1000) or None,
            "description": _clean_text(
                entity.get("description_original"), 3000
            ) or None,
            "function_text": _clean_text(
                entity.get("function_text_original"), 3000
            ) or None,
            "symbol_code": _clean_text(entity.get("symbol_code"), 300) or None,
            "location_code": _clean_text(
                entity.get("location_code"), 300
            ) or None,
            "bbox_pt": bbox,
            "source_text": source_text or None,
            "confidence": _clamp_conf(entity.get("confidence")),
            "properties": {
                "phase": MATERIALIZATION_PHASE,
                "pipeline_marker": PIPELINE_MARKER,
                "materializer_version": MATERIALIZER_VERSION,
                "pdf_page_number": int(page["pdf_page_number"]),
                "sheet_code": page.get("sheet_code"),
                "occurrence_id": occurrence_id,
                "region_id": entity.get("region_id"),
                "reference_value_original": entity.get(
                    "reference_value_original"
                ) or "",
                "reference_context_original": entity.get(
                    "reference_context_original"
                ) or "",
                "source_glyph_ids": entity.get("source_glyph_ids") or [],
                "source_word_ids": entity.get("source_word_ids") or [],
                "evidence_notes": entity.get("evidence_notes") or "",
                "detector_fingerprint": detector_fingerprint,
                "extractor_fingerprint": extractor_fingerprint,
                "verifier_fingerprint": verifier_fingerprint,
                "page_passed": True,
            },
        })

    for occurrence_id, occurrence_key in occurrence_key_by_id.items():
        entity = next(
            item for item in entities
            if _clean_text(item.get("occurrence_id"), 160) == occurrence_id
        )
        tag_key = _canonical_reference(entity.get("tag_original"))
        if entity.get("entity_type") in COMPONENT_ENTITY_TYPES and tag_key:
            canonical_key = f"graph:canonical:component:{tag_key}"
            add_edge({
                "edge_key": (
                    f"graph:page:{page['id']}:occurrence_of:{occurrence_id}"
                ),
                "page_id": int(page["id"]),
                "source_key": occurrence_key,
                "target_key": canonical_key,
                "relation_type": "occurrence_of",
                "is_directed": True,
                "bbox_pt": entity.get("bbox_pt") or [None] * 4,
                "source_text": entity.get("tag_original") or "",
                "confidence": _clamp_conf(entity.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "source": "deterministic_component_identity",
                },
            })

        resolved = resolution_by_id.get(occurrence_id) or {}
        for bom_row in resolved.get("bom_matches") or []:
            target_key = f"graph:reference:bom:{int(bom_row['id'])}"
            add_entity({
                "entity_key": target_key,
                "page_id": None,
                "parent_key": None,
                "entity_type": "bom_reference",
                "subtype": "certified_bom_row",
                "tag": bom_row.get("component_tag") or None,
                "label": bom_row.get("part_number") or None,
                "description": bom_row.get("description") or None,
                "function_text": "",
                "symbol_code": "",
                "location_code": "",
                "bbox_pt": [None] * 4,
                "source_text": " | ".join(
                    str(value or "") for value in (
                        bom_row.get("component_tag"),
                        bom_row.get("part_number"),
                        bom_row.get("manufacturer"),
                        bom_row.get("description"),
                    ) if value
                ),
                "confidence": _clamp_conf(bom_row.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "reference_table": "electrical_bom",
                    "reference_id": int(bom_row["id"]),
                    "reference_page_id": int(bom_row["page_id"]),
                    "manufacturer": bom_row.get("manufacturer") or "",
                    "part_number": bom_row.get("part_number") or "",
                },
            })
            add_edge({
                "edge_key": (
                    f"graph:page:{page['id']}:matches_bom:"
                    f"{occurrence_id}:{int(bom_row['id'])}"
                ),
                "page_id": int(page["id"]),
                "source_key": occurrence_key,
                "target_key": target_key,
                "relation_type": "matches_bom",
                "is_directed": True,
                "bbox_pt": entity.get("bbox_pt") or [None] * 4,
                "source_text": entity.get("tag_original") or "",
                "confidence": min(
                    _clamp_conf(entity.get("confidence")),
                    _clamp_conf(bom_row.get("confidence")),
                ),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "source": "exact_certified_bom_tag_match",
                },
            })

        for io_row in resolved.get("io_matches") or []:
            target_key = f"graph:reference:io:{int(io_row['id'])}"
            add_entity({
                "entity_key": target_key,
                "page_id": None,
                "parent_key": None,
                "entity_type": "io_reference_record",
                "subtype": io_row.get("io_type") or None,
                "tag": io_row.get("module_tag") or None,
                "label": io_row.get("channel_ref") or None,
                "description": io_row.get("description") or None,
                "function_text": io_row.get("signal_name") or None,
                "symbol_code": "",
                "location_code": "",
                "bbox_pt": [None] * 4,
                "source_text": " | ".join(
                    str(value or "") for value in (
                        io_row.get("module_tag"),
                        io_row.get("channel_ref"),
                        io_row.get("wire_reference"),
                        io_row.get("signal_name"),
                    ) if value
                ),
                "confidence": _clamp_conf(io_row.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "reference_table": "electrical_io",
                    "reference_id": int(io_row["id"]),
                    "reference_page_id": int(io_row["page_id"]),
                    "channel_ref": io_row.get("channel_ref") or "",
                    "wire_reference": io_row.get("wire_reference") or "",
                    "plc_address": io_row.get("plc_address") or "",
                    "is_safety": bool(io_row.get("is_safety")),
                },
            })
            add_edge({
                "edge_key": (
                    f"graph:page:{page['id']}:maps_to_io:"
                    f"{occurrence_id}:{int(io_row['id'])}"
                ),
                "page_id": int(page["id"]),
                "source_key": occurrence_key,
                "target_key": target_key,
                "relation_type": "maps_to_io",
                "is_directed": True,
                "bbox_pt": entity.get("bbox_pt") or [None] * 4,
                "source_text": entity.get("reference_context_original") or "",
                "confidence": min(
                    _clamp_conf(entity.get("confidence")),
                    _clamp_conf(io_row.get("confidence")),
                ),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "source": "exact_certified_io_reference_match",
                },
            })

        for terminal_row in resolved.get("terminal_matches") or []:
            target_key = (
                f"graph:reference:terminal:{int(terminal_row['id'])}"
            )
            add_entity({
                "entity_key": target_key,
                "page_id": None,
                "parent_key": None,
                "entity_type": "terminal_reference_record",
                "subtype": "certified_terminal",
                "tag": terminal_row.get("strip_tag") or None,
                "label": terminal_row.get("terminal_number") or None,
                "description": "",
                "function_text": "",
                "symbol_code": "",
                "location_code": "",
                "bbox_pt": [None] * 4,
                "source_text": " | ".join(
                    str(value or "") for value in (
                        terminal_row.get("strip_tag"),
                        terminal_row.get("terminal_number"),
                        terminal_row.get("wire_number"),
                        terminal_row.get("potential"),
                    ) if value
                ),
                "confidence": _clamp_conf(terminal_row.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "reference_table": "electrical_terminals",
                    "reference_id": int(terminal_row["id"]),
                    "reference_page_id": int(terminal_row["page_id"]),
                    "wire_number": terminal_row.get("wire_number") or "",
                    "potential": terminal_row.get("potential") or "",
                },
            })
            add_edge({
                "edge_key": (
                    f"graph:page:{page['id']}:maps_to_terminal:"
                    f"{occurrence_id}:{int(terminal_row['id'])}"
                ),
                "page_id": int(page["id"]),
                "source_key": occurrence_key,
                "target_key": target_key,
                "relation_type": "maps_to_terminal",
                "is_directed": True,
                "bbox_pt": entity.get("bbox_pt") or [None] * 4,
                "source_text": entity.get("reference_context_original") or "",
                "confidence": min(
                    _clamp_conf(entity.get("confidence")),
                    _clamp_conf(terminal_row.get("confidence")),
                ),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "source": "exact_certified_terminal_reference_match",
                },
            })

        for page_row in resolved.get("page_matches") or []:
            target_key = f"graph:reference:page:{int(page_row['id'])}"
            add_entity({
                "entity_key": target_key,
                "page_id": None,
                "parent_key": None,
                "entity_type": "page_reference_record",
                "subtype": page_row.get("page_type") or None,
                "tag": page_row.get("sheet_code") or None,
                "label": page_row.get("sheet_title") or None,
                "description": "",
                "function_text": "",
                "symbol_code": "",
                "location_code": "",
                "bbox_pt": [None] * 4,
                "source_text": " | ".join(
                    str(value or "") for value in (
                        page_row.get("sheet_code"),
                        page_row.get("sheet_title"),
                    ) if value
                ),
                "confidence": 1.0,
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "reference_table": "electrical_pages",
                    "reference_id": int(page_row["id"]),
                    "pdf_page_number": int(page_row["pdf_page_number"]),
                },
            })
            add_edge({
                "edge_key": (
                    f"graph:page:{page['id']}:references_page:"
                    f"{occurrence_id}:{int(page_row['id'])}"
                ),
                "page_id": int(page["id"]),
                "source_key": occurrence_key,
                "target_key": target_key,
                "relation_type": "references_page",
                "is_directed": True,
                "bbox_pt": entity.get("bbox_pt") or [None] * 4,
                "source_text": entity.get("reference_context_original") or "",
                "confidence": _clamp_conf(entity.get("confidence")),
                "properties": {
                    "phase": MATERIALIZATION_PHASE,
                    "source": "exact_indexed_page_reference_match",
                },
            })

    for edge in edges:
        source_occurrence_id = _clean_text(
            edge.get("source_occurrence_id"), 160
        )
        target_occurrence_id = _clean_text(
            edge.get("target_occurrence_id"), 160
        )
        edge_id = _clean_text(edge.get("edge_id"), 160)
        add_edge({
            "edge_key": f"graph:page:{page['id']}:visual:{edge_id}",
            "page_id": int(page["id"]),
            "source_key": occurrence_key_by_id[source_occurrence_id],
            "target_key": occurrence_key_by_id[target_occurrence_id],
            "relation_type": _clean_text(edge.get("relation_type"), 120),
            "is_directed": bool(edge.get("is_directed")),
            "bbox_pt": edge.get("bbox_pt") or [None] * 4,
            "source_text": " | ".join(
                value for value in (
                    _clean_text(edge.get("potential_original"), 500),
                    _clean_text(edge.get("wire_reference_original"), 500),
                ) if value
            ),
            "confidence": _clamp_conf(edge.get("confidence")),
            "properties": {
                "phase": MATERIALIZATION_PHASE,
                "pipeline_marker": PIPELINE_MARKER,
                "materializer_version": MATERIALIZER_VERSION,
                "edge_id": edge_id,
                "potential_original": edge.get("potential_original") or "",
                "wire_reference_original": edge.get(
                    "wire_reference_original"
                ) or "",
                "source_glyph_ids": edge.get("source_glyph_ids") or [],
                "source_drawing_ids": edge.get("source_drawing_ids") or [],
                "source_link_ids": edge.get("source_link_ids") or [],
                "evidence_notes": edge.get("evidence_notes") or "",
                "detector_fingerprint": detector_fingerprint,
                "extractor_fingerprint": extractor_fingerprint,
                "verifier_fingerprint": verifier_fingerprint,
                "page_passed": True,
            },
        })

    return {
        "entities": list(entity_specs.values()),
        "edges": list(edge_specs.values()),
        "occurrence_key_by_id": occurrence_key_by_id,
    }


def _db_replace_page_issues(
    *,
    context: dict,
    page: dict,
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
                (
                    int(context["version_id"]),
                    int(page["id"]),
                    MATERIALIZATION_PHASE,
                ),
            )
            for index, issue in enumerate(issues, start=1):
                issue_key = hashlib.sha256(
                    "|".join([
                        str(context["version_id"]),
                        str(page["id"]),
                        MATERIALIZATION_PHASE,
                        str(index),
                        str(issue.get("issue_type") or ""),
                        str(issue.get("message") or ""),
                    ]).encode("utf-8")
                ).hexdigest()
                properties = {
                    "phase": MATERIALIZATION_PHASE,
                    "pipeline_marker": PIPELINE_MARKER,
                    "materializer_version": MATERIALIZER_VERSION,
                    "pdf_page_number": int(page["pdf_page_number"]),
                    "sheet_code": page.get("sheet_code"),
                    "entity_ids": issue.get("entity_ids") or [],
                    "edge_ids": issue.get("edge_ids") or [],
                    "confidence": _clamp_conf(issue.get("confidence")),
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
                        int(page["id"]),
                        issue_key,
                        _clean_text(issue.get("issue_type"), 180),
                        (
                            issue.get("severity")
                            if issue.get("severity") in SEVERITIES
                            else "warning"
                        ),
                        _clean_text(issue.get("message"), 1600),
                        json.dumps(properties, ensure_ascii=False),
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _db_publish_graph_plan(
    *,
    context: dict,
    page: dict,
    plan: dict,
) -> dict:
    version_id = int(context["version_id"])
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            # Page-owned rows are replaced only after full validation has passed.
            cur.execute(
                """
                DELETE FROM public.electrical_edges
                WHERE version_id=%s
                  AND page_id=%s
                  AND extraction_method=%s;
                """,
                (version_id, int(page["id"]), EXTRACTION_METHOD),
            )
            cur.execute(
                """
                DELETE FROM public.electrical_entities
                WHERE version_id=%s
                  AND page_id=%s
                  AND extraction_method=%s;
                """,
                (version_id, int(page["id"]), EXTRACTION_METHOD),
            )

            id_by_key: dict[str, int] = {}
            entities = sorted(
                plan.get("entities") or [],
                key=lambda item: (
                    1 if item.get("parent_key") else 0,
                    1 if item.get("page_id") is not None else 0,
                    item.get("entity_key") or "",
                ),
            )
            pending = list(entities)
            while pending:
                progressed = False
                next_pending: list[dict] = []
                for spec in pending:
                    parent_key = spec.get("parent_key")
                    if parent_key and parent_key not in id_by_key:
                        next_pending.append(spec)
                        continue
                    entity_key = str(spec["entity_key"])
                    entity_id = _stable_bigint_id(
                        "electrical_entities", version_id, entity_key
                    )
                    bbox = list(spec.get("bbox_pt") or [None] * 4)
                    if len(bbox) != 4:
                        bbox = [None] * 4
                    cur.execute(
                        """
                        INSERT INTO public.electrical_entities(
                            id, version_id, company_id, machine_id,
                            bubble_document_id, page_id, parent_entity_id,
                            entity_key, entity_type, subtype, tag, label,
                            description, function_text, symbol_code,
                            location_code, x0, y0, x1, y1, source_text,
                            properties, confidence, extraction_method,
                            is_verified, created_at, updated_at
                        ) VALUES (
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            %s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,false,
                            NOW(),NOW()
                        )
                        ON CONFLICT (version_id, entity_key)
                        DO UPDATE SET
                            page_id=EXCLUDED.page_id,
                            parent_entity_id=EXCLUDED.parent_entity_id,
                            entity_type=EXCLUDED.entity_type,
                            subtype=EXCLUDED.subtype,
                            tag=EXCLUDED.tag,
                            label=EXCLUDED.label,
                            description=EXCLUDED.description,
                            function_text=EXCLUDED.function_text,
                            symbol_code=EXCLUDED.symbol_code,
                            location_code=EXCLUDED.location_code,
                            x0=EXCLUDED.x0,
                            y0=EXCLUDED.y0,
                            x1=EXCLUDED.x1,
                            y1=EXCLUDED.y1,
                            source_text=EXCLUDED.source_text,
                            properties=EXCLUDED.properties,
                            confidence=EXCLUDED.confidence,
                            extraction_method=EXCLUDED.extraction_method,
                            updated_at=NOW()
                        RETURNING id;
                        """,
                        (
                            entity_id,
                            version_id,
                            context["company_id"],
                            context["machine_id"],
                            context["bubble_document_id"],
                            spec.get("page_id"),
                            id_by_key.get(parent_key) if parent_key else None,
                            entity_key,
                            spec.get("entity_type"),
                            spec.get("subtype"),
                            spec.get("tag"),
                            spec.get("label"),
                            spec.get("description"),
                            spec.get("function_text"),
                            spec.get("symbol_code"),
                            spec.get("location_code"),
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            spec.get("source_text"),
                            json.dumps(
                                spec.get("properties") or {},
                                ensure_ascii=False,
                            ),
                            _clamp_conf(spec.get("confidence")),
                            EXTRACTION_METHOD,
                        ),
                    )
                    id_by_key[entity_key] = int(cur.fetchone()[0])
                    progressed = True
                if not progressed and next_pending:
                    raise RuntimeError(
                        "Graph entity parent dependency could not be resolved"
                    )
                pending = next_pending

            for spec in plan.get("edges") or []:
                source_key = spec.get("source_key")
                target_key = spec.get("target_key")
                if source_key not in id_by_key or target_key not in id_by_key:
                    raise RuntimeError(
                        "Graph edge endpoint was not materialized"
                    )
                edge_key = str(spec["edge_key"])
                edge_id = _stable_bigint_id(
                    "electrical_edges", version_id, edge_key
                )
                bbox = list(spec.get("bbox_pt") or [None] * 4)
                if len(bbox) != 4:
                    bbox = [None] * 4
                cur.execute(
                    """
                    INSERT INTO public.electrical_edges(
                        id, version_id, company_id, machine_id,
                        bubble_document_id, page_id, edge_key,
                        source_entity_id, target_entity_id, relation_type,
                        is_directed, x0, y0, x1, y1, source_text,
                        properties, confidence, extraction_method,
                        is_verified, created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s::jsonb,%s,%s,false,NOW(),NOW()
                    )
                    ON CONFLICT (version_id, edge_key)
                    DO UPDATE SET
                        page_id=EXCLUDED.page_id,
                        source_entity_id=EXCLUDED.source_entity_id,
                        target_entity_id=EXCLUDED.target_entity_id,
                        relation_type=EXCLUDED.relation_type,
                        is_directed=EXCLUDED.is_directed,
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
                        edge_id,
                        version_id,
                        context["company_id"],
                        context["machine_id"],
                        context["bubble_document_id"],
                        spec.get("page_id"),
                        edge_key,
                        id_by_key[source_key],
                        id_by_key[target_key],
                        spec.get("relation_type"),
                        bool(spec.get("is_directed")),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        spec.get("source_text"),
                        json.dumps(
                            spec.get("properties") or {},
                            ensure_ascii=False,
                        ),
                        _clamp_conf(spec.get("confidence")),
                        EXTRACTION_METHOD,
                    ),
                )

        conn.commit()
        page_entity_count = sum(
            1 for item in plan.get("entities") or []
            if item.get("page_id") == int(page["id"])
        )
        page_edge_count = sum(
            1 for item in plan.get("edges") or []
            if item.get("page_id") == int(page["id"])
        )
        return {
            "published_page_entities": page_entity_count,
            "published_page_edges": page_edge_count,
            "materialized_entity_count_including_references": len(
                plan.get("entities") or []
            ),
            "materialized_edge_count_including_references": len(
                plan.get("edges") or []
            ),
        }
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
    published_entities: int,
    published_edges: int,
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
            page_results = metadata.get("graph_page_results") or {}
            if not isinstance(page_results, dict):
                page_results = {}
            page_results[str(page["pdf_page_number"])] = {
                "page_passed": bool(page_passed),
                "published_entities": int(published_entities),
                "published_edges": int(published_edges),
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
            total_pages = int(context["all_graph_pages_total"])
            graph_status = (
                "graph_ready"
                if passed_pages == total_pages and total_pages > 0
                else ("partial" if passed_pages > 0 else "review_required")
            )
            if not page_passed:
                graph_status = "review_required"

            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_entities
                WHERE version_id=%s AND extraction_method=%s;
                """,
                (int(context["version_id"]), EXTRACTION_METHOD),
            )
            entity_count = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.electrical_edges
                WHERE version_id=%s AND extraction_method=%s;
                """,
                (int(context["version_id"]), EXTRACTION_METHOD),
            )
            edge_count = int(cur.fetchone()[0] or 0)

            metadata["graph_page_results"] = page_results
            metadata["graph_structured_status"] = graph_status
            metadata["graph_pipeline_marker"] = PIPELINE_MARKER
            metadata["graph_materializer_version"] = MATERIALIZER_VERSION
            metadata["graph_passed_pages"] = passed_pages
            metadata["graph_total_pages"] = total_pages
            metadata["graph_entities"] = entity_count
            metadata["graph_edges"] = edge_count

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
                    None if page_passed else "GRAPH_REVIEW_REQUIRED",
                    (
                        None
                        if page_passed
                        else "Electrical graph page requires review before publication"
                    ),
                    int(context["version_id"]),
                ),
            )
        conn.commit()
        return {
            "status": version_status,
            "graph_status": graph_status,
            "graph_entity_count": entity_count,
            "graph_edge_count": edge_count,
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
        totals["new_reasoning_tokens"] += int(
            usage.get("reasoning_tokens") or 0
        )
        totals["new_cost_usd"] = round(
            float(totals["new_cost_usd"])
            + float(usage.get("cost_usd") or 0.0),
            6,
        )


def _severity_counts(issues: list[dict]) -> dict:
    return {
        severity: sum(
            1 for issue in issues if issue.get("severity") == severity
        )
        for severity in sorted(SEVERITIES)
    }


def extract_electrical_graph_page(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int] = None,
    pdf_page_numbers: Optional[list[int]] = None,
    force: bool = False,
) -> dict:
    if not GRAPH_ENABLED:
        raise ValueError("Electrical graph extraction is disabled")

    context = _load_context(
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        version_id=version_id,
        pdf_page_numbers=pdf_page_numbers,
    )
    page = context["page"]
    context["page"] = page
    _, source_doc = _fetch_source_pdf(context)
    usage_totals = _usage_accumulator()

    try:
        page_index = int(page["pdf_page_number"]) - 1
        source_page = source_doc[page_index]
        words = _word_registry(page)
        glyphs = _glyph_registry(source_page)
        drawings = _drawing_registry(source_page)
        registry = _load_reference_registry(context, int(page["id"]))
        reference_candidates = _candidate_reference_registry(page, registry)
        links = list(registry.get("cross_references") or [])

        if not glyphs:
            raise ValueError(
                "GRAPH_GLYPH_EVIDENCE_MISSING: the schematic page has no "
                "independent vector-character evidence."
            )
        if len(glyphs) > MAX_GLYPHS_IN_PROMPT:
            raise ValueError(
                "GRAPH_GLYPH_EVIDENCE_LIMIT_EXCEEDED: page glyph registry "
                "exceeds the configured complete-prompt limit."
            )
        if len(drawings) > MAX_DRAWINGS_IN_PROMPT:
            raise ValueError(
                "GRAPH_DRAWING_EVIDENCE_LIMIT_EXCEEDED: page drawing registry "
                "exceeds the configured complete-prompt limit."
            )

        page_original = _render_page(source_doc, page_index, 0)
        page_rotated = _render_page(source_doc, page_index, 90)

        drawing_summary = {
            "drawing_count": len(drawings),
            "item_count": sum(int(x.get("item_count") or 0) for x in drawings),
            "drawing_bbox_union": (
                _rect_list(
                    fitz.Rect(
                        min(x["bbox_pt"][0] for x in drawings),
                        min(x["bbox_pt"][1] for x in drawings),
                        max(x["bbox_pt"][2] for x in drawings),
                        max(x["bbox_pt"][3] for x in drawings),
                    )
                )
                if drawings
                else []
            ),
        }
        link_summary = {
            "link_count": len(links),
            "target_sheet_codes": sorted({
                str(x.get("target_sheet_code") or "")
                for x in links
                if str(x.get("target_sheet_code") or "")
            }),
        }
        detector_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "sheet_code": page.get("sheet_code"),
            "sheet_title": page.get("sheet_title"),
            "page_width_pt": page.get("page_width_pt"),
            "page_height_pt": page.get("page_height_pt"),
            "glyph_count": len(glyphs),
            "word_count": len(words),
            "drawing_summary": drawing_summary,
            "link_summary": link_summary,
            "render_dpi": RENDER_DPI,
        }
        detector, detector_usage, detector_reused, detector_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_graph_region_detector_v1",
            region_hash=_sha256_json(detector_request),
            model=DETECTOR_MODEL,
            prompt_version=DETECTOR_PROMPT_VERSION,
            request_payload=detector_request,
            messages=_detector_messages(
                page,
                page_original,
                page_rotated,
                drawing_summary,
                link_summary,
            ),
            json_schema=_detector_schema(),
            force=force,
            request_metadata={
                "glyph_count": len(glyphs),
                "drawing_count": len(drawings),
                "link_count": len(links),
            },
        )
        _add_usage(
            usage_totals,
            "detector",
            detector_usage,
            detector_reused,
        )

        extractor_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "detector_fingerprint": detector_fp,
            "detector": detector,
            "vector_words": words,
            "source_glyphs": [
                {
                    "glyph_id": x["glyph_id"],
                    "text_original": x["text_original"],
                    "bbox_pt": x["bbox_pt"],
                    "origin_pt": x["origin_pt"],
                    "direction": x["direction"],
                }
                for x in glyphs
            ],
            "drawing_registry": drawings,
            "pdf_link_registry": links,
            "certified_reference_candidates": reference_candidates,
            "render_dpi": RENDER_DPI,
        }
        extraction, extractor_usage, extractor_reused, extractor_fp = (
            _cached_call(
                context=context,
                page=page,
                task_type="vision_graph_page_extractor_v1",
                region_hash=_sha256_json(extractor_request),
                model=EXTRACTOR_MODEL,
                prompt_version=EXTRACTOR_PROMPT_VERSION,
                request_payload=extractor_request,
                messages=_extractor_messages(
                    page=page,
                    detector=detector,
                    words=words,
                    glyphs=glyphs,
                    drawings=drawings,
                    links=links,
                    reference_candidates=reference_candidates,
                    image_original=page_original,
                    image_rotated=page_rotated,
                ),
                json_schema=_extractor_schema(),
                force=force,
                request_metadata={
                    "detector_fingerprint": detector_fp,
                    "glyph_count": len(glyphs),
                    "drawing_count": len(drawings),
                    "link_count": len(links),
                },
            )
        )
        _add_usage(
            usage_totals,
            "extractor",
            extractor_usage,
            extractor_reused,
        )

        resolution = _resolve_references(extraction, registry)
        verifier_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "detector_fingerprint": detector_fp,
            "extractor_fingerprint": extractor_fp,
            "detector": detector,
            "extraction": extraction,
            "deterministic_reference_resolution": resolution,
            "drawing_count": len(drawings),
            "link_count": len(links),
            "render_dpi": RENDER_DPI,
        }
        verifier, verifier_usage, verifier_reused, verifier_fp = _cached_call(
            context=context,
            page=page,
            task_type="vision_graph_page_verifier_v1",
            region_hash=_sha256_json(verifier_request),
            model=VERIFIER_MODEL,
            prompt_version=VERIFIER_PROMPT_VERSION,
            request_payload=verifier_request,
            messages=_verifier_messages(
                page=page,
                detector=detector,
                extraction=extraction,
                resolution=resolution,
                drawing_count=len(drawings),
                link_count=len(links),
                image_original=page_original,
                image_rotated=page_rotated,
            ),
            json_schema=_verifier_schema(),
            force=force,
            request_metadata={
                "detector_fingerprint": detector_fp,
                "extractor_fingerprint": extractor_fp,
                "entity_count": len(extraction.get("entities") or []),
                "edge_count": len(extraction.get("edges") or []),
            },
        )
        _add_usage(
            usage_totals,
            "verifier",
            verifier_usage,
            verifier_reused,
        )

        page_passed, entities, edges, issues = _validate_candidate_graph(
            page=page,
            detector=detector,
            extraction=extraction,
            verifier=verifier,
            resolution=resolution,
            glyphs=glyphs,
            words=words,
            drawings=drawings,
            links=links,
        )
        _db_replace_page_issues(
            context=context,
            page=page,
            issues=issues,
        )

        publication = {
            "published_page_entities": 0,
            "published_page_edges": 0,
            "materialized_entity_count_including_references": 0,
            "materialized_edge_count_including_references": 0,
        }
        if page_passed:
            plan = _build_materialization_plan(
                context=context,
                page=page,
                entities=entities,
                edges=edges,
                resolution=resolution,
                detector_fingerprint=detector_fp,
                extractor_fingerprint=extractor_fp,
                verifier_fingerprint=verifier_fp,
            )
            publication = _db_publish_graph_plan(
                context=context,
                page=page,
                plan=plan,
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
            published_entities=publication["published_page_entities"],
            published_edges=publication["published_page_edges"],
            blocking_count=blocking,
        )

        entity_type_counts: dict[str, int] = {}
        for entity in entities:
            key = _clean_text(entity.get("entity_type"), 120)
            entity_type_counts[key] = entity_type_counts.get(key, 0) + 1
        relation_type_counts: dict[str, int] = {}
        for edge in edges:
            key = _clean_text(edge.get("relation_type"), 120)
            relation_type_counts[key] = relation_type_counts.get(key, 0) + 1

        return {
            "electrical_document_id": context["electrical_document_id"],
            "electrical_version_id": context["version_id"],
            "pdf_page_number": page["pdf_page_number"],
            "sheet_code": page["sheet_code"],
            "sheet_title": page["sheet_title"],
            "page_type": page["page_type"],
            "language": detector.get("language")
            or page.get("classification_language"),
            "page_passed": bool(page_passed),
            "detected_region_count": len(detector.get("regions") or []),
            "extracted_entity_count": len(entities),
            "extracted_edge_count": len(edges),
            **publication,
            "blocking_issue_count_this_page": blocking,
            "warning_issue_count_this_page": warning,
            "severity_counts": _severity_counts(issues),
            "entity_type_counts": entity_type_counts,
            "relation_type_counts": relation_type_counts,
            "reference_resolution": {
                "all_reference_entities_resolved": resolution.get(
                    "all_reference_entities_resolved"
                ),
                "unresolved_reference_entity_ids": resolution.get(
                    "unresolved_reference_entity_ids"
                ) or [],
                "match_counts": resolution.get("match_counts") or {},
            },
            "source_evidence": {
                "vector_word_count": len(words),
                "glyph_count": len(glyphs),
                "drawing_count": len(drawings),
                "pdf_link_count": len(links),
                "glyph_registry_complete": True,
                "drawing_registry_complete": True,
            },
            **state,
            **_db_ai_totals(context["version_id"]),
            **usage_totals,
        }
    finally:
        source_doc.close()
