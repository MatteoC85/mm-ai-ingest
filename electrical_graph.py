import base64
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Optional

import fitz
import psycopg2
import requests

from electrical_source_store import download_electrical_source_pdf

# MachineMind Phase 2G V3
# Page-atomic, multimodal electrical graph extraction with evidence ownership.
# The deterministic layer is geometry/evidence based and contains no page,
# language, font, manufacturer, component-tag or drawing-template dictionary.
# Batch orchestration remains outside the page transaction so every page is
# independently resumable and fail-closed.


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
    or "mm-electrical-graph-page-verifier-v3"
).strip()
MATERIALIZER_VERSION = (
    os.environ.get("MM_ELECTRICAL_GRAPH_MATERIALIZER_VERSION")
    or "mm-electrical-graph-materializer-v3.1"
).strip()

OPENAI_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_GRAPH_TIMEOUT_SECONDS", 600, 30, 600
)
FETCH_TIMEOUT_SECONDS = _env_int(
    "MM_ELECTRICAL_GRAPH_FETCH_TIMEOUT_SECONDS", 60, 10, 300
)
RENDER_DPI = _env_int(
    "MM_ELECTRICAL_GRAPH_RENDER_DPI", 240, 120, 360
)
MAX_COMPLETION_TOKENS = _env_int(
    "MM_ELECTRICAL_GRAPH_MAX_COMPLETION_TOKENS", 40000, 1000, 64000
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

PIPELINE_MARKER = "phase2-graph-v3.1-echo-tolerant-patch-plan-source-snapshot"
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
REFERENCE_ENTITY_TYPES = {
    "io_reference",
    "terminal_reference",
    "page_reference",
}
REFERENCE_ONLY_REJECTION_ISSUE_TYPES = {
    "certified_registry_mismatch",
    "unresolved_reference",
}
POST_VERIFIER_ADJUDICATION_VERSION = (
    "graph-post-verifier-candidate-adjudication-v1"
)
EXPLICIT_UNRESOLVED_REFERENCE_VERSION = (
    "graph-explicit-unresolved-reference-v1"
)
PAGE_REFERENCE_RESOLUTION_VERSION = (
    "graph-page-reference-sheet-and-link-v1"
)
EDGE_GEOMETRY_POLICY_VERSION = "graph-line-or-area-edge-geometry-v1"
REGION_BBOX_ADJUDICATION_VERSION = "graph-region-bbox-from-final-evidence-v1"
VERIFIER_EVIDENCE_RECOVERY_VERSION = "graph-verifier-evidence-recovery-v1"
VISUAL_EVIDENCE_ADJUDICATION_VERSION = "graph-visual-evidence-adjudication-v1"
RECOVERABLE_ENTITY_TYPES = ENTITY_TYPES - REFERENCE_ENTITY_TYPES
VISUAL_EVIDENCE_STATUSES = {
    "accounted_existing_graph",
    "accounted_non_materializable",
    "recovered_entity",
    "recovered_edge",
    "still_unresolved",
}

# Only these fields claim literal printed source text. Description/function
# fields are semantic annotations produced from the image and may describe a
# graphic-only symbol without pretending that those words are printed nearby.
SOURCE_VISIBLE_ENTITY_TEXT_FIELDS = (
    "tag_original",
    "label_original",
    "location_code",
    "reference_value_original",
    "reference_context_original",
)
SEMANTIC_ENTITY_ANNOTATION_FIELDS = (
    "description_original",
    "function_text_original",
    "subtype",
    "symbol_code",
)
ENTITY_BBOX_RECONCILIATION_VERSION = (
    "graph-entity-bbox-from-source-evidence-v2"
)
EDGE_BBOX_RECONCILIATION_VERSION = (
    "graph-edge-bbox-from-source-evidence-v2"
)
NONMATERIALIZABLE_CONTEXT_VERSION = (
    "graph-nonmaterializable-context-links-v2"
)
REVIEW_GROUPING_VERSION = "graph-review-signature-v3-causal-family"
GRAPH_PATCH_PLAN_VERSION = "graph-atomic-patch-plan-v1"
GRAPH_PATCH_APPLICATION_VERSION = "graph-atomic-patch-application-v1.1"
PATCH_RESULT_COMPATIBILITY_VERSION = (
    "graph-patch-result-echo-normalization-v1"
)
GRAPH_FINAL_VALIDATION_VERSION = "graph-final-projection-validation-v1"
ENTITY_PATCH_ACTIONS = {
    "KEEP_ENTITY",
    "REMOVE_ENTITY",
    "REPLACE_ENTITY",
    "SPLIT_ENTITY",
    "ADD_ENTITY",
}
EDGE_PATCH_ACTIONS = {
    "KEEP_EDGE",
    "REMOVE_EDGE",
    "REWIRE_EDGE",
    "ADD_EDGE",
}
PATCH_EVIDENCE_STATUSES = {
    "accounted_by_final_graph",
    "accounted_non_materializable",
    "still_unresolved",
}
VERIFIER_ISSUE_RESOLUTION_STATUSES = {
    "open",
    "resolved_by_patch_plan",
    "informational",
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



def _edge_bbox_valid(value: Any, page: dict) -> bool:
    """Validate a conductor/path bbox without requiring positive area.

    Electrical conductors are often represented by one vertical or horizontal
    segment, so x0 == x1 or y0 == y1 is valid. A point, reversed coordinates,
    non-finite values, or an out-of-page path remains invalid.
    """
    try:
        coords = [float(x) for x in value]
    except Exception:
        return False
    if len(coords) != 4 or not all(math.isfinite(x) for x in coords):
        return False
    x0, y0, x1, y1 = coords
    if x0 > x1 or y0 > y1:
        return False
    if max(x1 - x0, y1 - y0) <= 0.25:
        return False
    width = float(page.get("page_width_pt") or 0.0)
    height = float(page.get("page_height_pt") or 0.0)
    if width > 0.0 and height > 0.0:
        return bool(
            x0 >= -2.0
            and y0 >= -2.0
            and x1 <= width + 2.0
            and y1 <= height + 2.0
        )
    return True


def _rect_overlap_score(a: Any, b: Any) -> float:
    """Return a conservative overlap score for source-link adjudication."""
    try:
        ra = _rect_from(a)
        rb = _rect_from(b)
    except Exception:
        return 0.0
    inter = ra & rb
    inter_area = max(0.0, float(inter.get_area()))
    if inter_area > 0.0:
        denom = max(1.0, min(float(ra.get_area()), float(rb.get_area())))
        return min(1.0, inter_area / denom)
    # A zero-area source rectangle is possible for some CAD link annotations.
    ac = fitz.Point((ra.x0 + ra.x1) / 2.0, (ra.y0 + ra.y1) / 2.0)
    bc = fitz.Point((rb.x0 + rb.x1) / 2.0, (rb.y0 + rb.y1) / 2.0)
    if ra.contains(bc) or rb.contains(ac):
        return 1.0
    return 0.0


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



def list_electrical_graph_pages(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    version_id: Optional[int] = None,
    include_passed: bool = True,
) -> dict:
    """Return a read-only, resumable page plan for external batch runners."""
    company_id = _clean_text(company_id, 300)
    machine_id = _clean_text(machine_id, 300)
    bubble_document_id = _clean_text(bubble_document_id, 300)
    if not (company_id and machine_id and bubble_document_id):
        raise ValueError(
            "company_id, machine_id and bubble_document_id are required"
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
                SELECT v.id, v.version_no, v.status, v.metadata
                FROM public.electrical_versions v
                WHERE v.company_id=%s
                  AND v.machine_id=%s
                  AND v.bubble_document_id=%s
                  {version_clause}
                ORDER BY v.version_no DESC
                LIMIT 1;
                """,
                params,
            )
            version_row = cur.fetchone()
            if not version_row:
                raise ValueError(
                    "Electrical version not found for supplied scope"
                )
            resolved_version_id = int(version_row[0])
            metadata = _json_obj(version_row[3], {}) or {}
            page_results = metadata.get("graph_page_results") or {}
            if not isinstance(page_results, dict):
                page_results = {}

            cur.execute(
                """
                SELECT
                    p.id,
                    p.pdf_page_number,
                    p.sheet_code,
                    p.sheet_title,
                    p.group_code,
                    p.page_width_pt,
                    p.page_height_pt,
                    p.classification_language,
                    p.semantic_confidence,
                    LENGTH(COALESCE(p.raw_text, '')),
                    jsonb_array_length(
                        COALESCE(p.text_spans_json, '[]'::jsonb)
                    ),
                    jsonb_array_length(
                        COALESCE(p.links_json, '[]'::jsonb)
                    ),
                    (
                        SELECT COUNT(*)
                        FROM public.electrical_entities e
                        WHERE e.version_id=p.version_id
                          AND e.page_id=p.id
                          AND e.properties ->> 'phase'=%s
                    ) AS published_entity_rows,
                    (
                        SELECT COUNT(*)
                        FROM public.electrical_edges ed
                        WHERE ed.version_id=p.version_id
                          AND ed.page_id=p.id
                          AND ed.properties ->> 'phase'=%s
                    ) AS published_edge_rows
                FROM public.electrical_pages p
                WHERE p.version_id=%s
                  AND p.page_type=%s
                ORDER BY p.pdf_page_number;
                """,
                (
                    MATERIALIZATION_PHASE,
                    MATERIALIZATION_PHASE,
                    resolved_version_id,
                    PAGE_TYPE,
                ),
            )
            pages: list[dict] = []
            for row in cur.fetchall():
                pdf_page_number = int(row[1])
                result = page_results.get(str(pdf_page_number)) or {}
                if not isinstance(result, dict):
                    result = {}
                page_passed = bool(result.get("page_passed"))
                status = (
                    "passed"
                    if page_passed
                    else (
                        "review_required"
                        if str(pdf_page_number) in page_results
                        else "not_started"
                    )
                )
                if status == "passed" and not include_passed:
                    continue
                pages.append({
                    "page_id": int(row[0]),
                    "pdf_page_number": pdf_page_number,
                    "sheet_code": str(row[2] or ""),
                    "sheet_title": str(row[3] or ""),
                    "group_code": str(row[4] or ""),
                    "page_width_pt": float(row[5] or 0.0),
                    "page_height_pt": float(row[6] or 0.0),
                    "language": str(row[7] or "unknown"),
                    "semantic_confidence": float(row[8] or 0.0),
                    "raw_text_chars": int(row[9] or 0),
                    "vector_word_count": int(row[10] or 0),
                    "stored_link_count": int(row[11] or 0),
                    "published_entity_rows": int(row[12] or 0),
                    "published_edge_rows": int(row[13] or 0),
                    "status": status,
                    "page_passed": page_passed,
                    "result_pipeline_marker": str(
                        result.get("pipeline_marker") or ""
                    ),
                    "result_materializer_version": str(
                        result.get("materializer_version") or ""
                    ),
                    "blocking_issue_count": int(
                        result.get("blocking_issue_count") or 0
                    ),
                    "blocking_issue_type_counts": result.get(
                        "blocking_issue_type_counts"
                    ) or {},
                    "review_cause_family_counts": result.get(
                        "review_cause_family_counts"
                    ) or {},
                    "patch_plan_validated": bool(
                        result.get("patch_plan_validated")
                    ),
                    "review_signature": str(
                        result.get("review_signature") or ""
                    ),
                    "updated_at": str(result.get("updated_at") or ""),
                })

            counts = {
                "total": len(pages) if include_passed else int(
                    sum(1 for _ in pages)
                ),
                "passed": sum(1 for page in pages if page["status"] == "passed"),
                "review_required": sum(
                    1 for page in pages
                    if page["status"] == "review_required"
                ),
                "not_started": sum(
                    1 for page in pages if page["status"] == "not_started"
                ),
            }
            return {
                "electrical_version_id": resolved_version_id,
                "electrical_version_no": int(version_row[1]),
                "version_status": str(version_row[2] or ""),
                "graph_status": str(
                    metadata.get("graph_structured_status") or "not_started"
                ),
                "graph_pipeline_marker": PIPELINE_MARKER,
                "graph_materializer_version": MATERIALIZER_VERSION,
                "include_passed": bool(include_passed),
                "counts": counts,
                "pages": pages,
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



def _registry_bbox_map(items: list[dict], id_field: str) -> dict[int, list[float]]:
    output: dict[int, list[float]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get(id_field))
            bbox = [float(value) for value in (item.get("bbox_pt") or [])]
        except Exception:
            continue
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        if not all(math.isfinite(value) for value in bbox):
            continue
        output[item_id] = [x0, y0, x1, y1]
    return output


def _evidence_bbox(
    *,
    ids: set[int],
    bbox_map: dict[int, list[float]],
    page: dict,
    minimum_extent: float = 1.0,
    margin: float = 1.25,
) -> Optional[list[float]]:
    rects: list[fitz.Rect] = []
    for item_id in sorted(ids):
        bbox = bbox_map.get(int(item_id))
        if not bbox:
            continue
        try:
            rect = fitz.Rect(*bbox)
        except Exception:
            continue
        # A line drawing can legitimately have zero width or height. Keep it
        # as evidence and give the final ownership rectangle a tiny extent.
        if rect.x1 < rect.x0:
            rect.x0, rect.x1 = rect.x1, rect.x0
        if rect.y1 < rect.y0:
            rect.y0, rect.y1 = rect.y1, rect.y0
        rects.append(rect)
    if not rects:
        return None

    union = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union.include_rect(rect)

    if union.width < minimum_extent:
        center = (union.x0 + union.x1) / 2.0
        union.x0 = center - minimum_extent / 2.0
        union.x1 = center + minimum_extent / 2.0
    if union.height < minimum_extent:
        center = (union.y0 + union.y1) / 2.0
        union.y0 = center - minimum_extent / 2.0
        union.y1 = center + minimum_extent / 2.0

    union.x0 -= margin
    union.y0 -= margin
    union.x1 += margin
    union.y1 += margin

    page_width = float(page.get("page_width_pt") or 0.0)
    page_height = float(page.get("page_height_pt") or 0.0)
    if page_width > 0.0:
        union.x0 = max(0.0, min(page_width, union.x0))
        union.x1 = max(0.0, min(page_width, union.x1))
    if page_height > 0.0:
        union.y0 = max(0.0, min(page_height, union.y0))
        union.y1 = max(0.0, min(page_height, union.y1))
    if union.x1 <= union.x0 or union.y1 <= union.y0:
        return None
    return _rect_list(union, 3)


def _reconcile_graph_geometry_from_evidence(
    *,
    page: dict,
    entities: list[dict],
    edges: list[dict],
    glyphs: list[dict],
    words: list[dict],
    drawings: list[dict],
) -> tuple[dict, list[dict]]:
    """Repair only invalid candidate geometry from exact cited source evidence.

    AI bboxes are advisory. Ownership is established by the source registries:
    glyph/word rectangles for text-bearing occurrences and drawing rectangles
    for symbols/conductors. Valid original-PDF bboxes are never changed.
    """
    issues: list[dict] = []
    glyph_map = _registry_bbox_map(glyphs, "glyph_id")
    word_map = _registry_bbox_map(words, "word_id")
    drawing_map = _registry_bbox_map(drawings, "drawing_id")
    entity_audit: list[dict] = []
    edge_audit: list[dict] = []

    for entity in entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        original = list(entity.get("bbox_pt") or [])
        if _bbox_valid(original, page):
            entity_audit.append({
                "occurrence_id": occurrence_id,
                "reason": "original_pdf_point_bbox_valid",
                "original_bbox_pt": original,
                "final_bbox_pt": original,
                "validated": True,
            })
            continue

        glyph_ids = {
            int(value) for value in (entity.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        word_ids = {
            int(value) for value in (entity.get("source_word_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        drawing_ids = {
            int(value) for value in (entity.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        source_rects: list[list[float]] = []
        for ids, bbox_map in (
            (glyph_ids, glyph_map),
            (word_ids, word_map),
            (drawing_ids, drawing_map),
        ):
            candidate = _evidence_bbox(
                ids=ids,
                bbox_map=bbox_map,
                page=page,
            )
            if candidate:
                source_rects.append(candidate)
        final_bbox = None
        if source_rects:
            rect = fitz.Rect(*source_rects[0])
            for candidate in source_rects[1:]:
                rect.include_rect(fitz.Rect(*candidate))
            final_bbox = _rect_list(rect, 3)
        validated = bool(final_bbox and _bbox_valid(final_bbox, page))
        if validated:
            entity["bbox_pt"] = final_bbox
            entity["bbox_reconciliation"] = {
                "version": ENTITY_BBOX_RECONCILIATION_VERSION,
                "original_bbox_pt": original,
                "final_bbox_pt": final_bbox,
                "source_glyph_ids": sorted(glyph_ids),
                "source_word_ids": sorted(word_ids),
                "source_drawing_ids": sorted(drawing_ids),
                "validated": True,
            }
            issues.append(_local_issue(
                issue_type="graph-entity-bbox-reconciled-from-evidence",
                message=(
                    "Invalid candidate entity geometry was replaced by the "
                    "union of its exact source evidence in original PDF points"
                ),
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=entity.get("confidence") or 0.0,
                severity="info",
                source_stage="source_evidence_geometry_adjudicator",
            ))
        entity_audit.append({
            "occurrence_id": occurrence_id,
            "reason": (
                "invalid_candidate_rebuilt_from_source_evidence"
                if validated else "invalid_candidate_without_repairable_evidence"
            ),
            "original_bbox_pt": original,
            "final_bbox_pt": final_bbox or original,
            "validated": validated,
        })

    for edge in edges:
        edge_id = _clean_text(edge.get("edge_id"), 160)
        original = list(edge.get("bbox_pt") or [])
        if _edge_bbox_valid(original, page):
            edge_audit.append({
                "edge_id": edge_id,
                "reason": "original_pdf_point_bbox_valid",
                "original_bbox_pt": original,
                "final_bbox_pt": original,
                "validated": True,
            })
            continue
        drawing_ids = {
            int(value) for value in (edge.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        glyph_ids = {
            int(value) for value in (edge.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        candidates: list[list[float]] = []
        for ids, bbox_map in (
            (drawing_ids, drawing_map),
            (glyph_ids, glyph_map),
        ):
            candidate = _evidence_bbox(
                ids=ids,
                bbox_map=bbox_map,
                page=page,
                minimum_extent=0.5,
                margin=0.5,
            )
            if candidate:
                candidates.append(candidate)
        final_bbox = None
        if candidates:
            rect = fitz.Rect(*candidates[0])
            for candidate in candidates[1:]:
                rect.include_rect(fitz.Rect(*candidate))
            final_bbox = _rect_list(rect, 3)
        validated = bool(final_bbox and _edge_bbox_valid(final_bbox, page))
        if validated:
            edge["bbox_pt"] = final_bbox
            edge["bbox_reconciliation"] = {
                "version": EDGE_BBOX_RECONCILIATION_VERSION,
                "original_bbox_pt": original,
                "final_bbox_pt": final_bbox,
                "source_glyph_ids": sorted(glyph_ids),
                "source_drawing_ids": sorted(drawing_ids),
                "validated": True,
            }
            issues.append(_local_issue(
                issue_type="graph-edge-bbox-reconciled-from-evidence",
                message=(
                    "Invalid candidate edge geometry was replaced by the union "
                    "of exact local drawing evidence"
                ),
                edge_ids=[edge_id] if edge_id else [],
                confidence=edge.get("confidence") or 0.0,
                severity="info",
                source_stage="source_evidence_geometry_adjudicator",
            ))
        edge_audit.append({
            "edge_id": edge_id,
            "reason": (
                "invalid_candidate_rebuilt_from_source_evidence"
                if validated else "invalid_candidate_without_repairable_evidence"
            ),
            "original_bbox_pt": original,
            "final_bbox_pt": final_bbox or original,
            "validated": validated,
        })

    return {
        "version": "graph-source-evidence-geometry-reconciliation-v2",
        "entity_reconciliations": entity_audit,
        "edge_reconciliations": edge_audit,
        "reconciled_entity_count": sum(
            1 for item in entity_audit
            if item.get("reason") == "invalid_candidate_rebuilt_from_source_evidence"
            and item.get("validated")
        ),
        "reconciled_edge_count": sum(
            1 for item in edge_audit
            if item.get("reason") == "invalid_candidate_rebuilt_from_source_evidence"
            and item.get("validated")
        ),
        "validated": all(
            item.get("validated") for item in entity_audit + edge_audit
        ),
    }, issues


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


def _recovery_entity_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "occurrence_id": {"type": "string"},
            "region_id": {"type": "string"},
            "entity_type": {
                "type": "string",
                "enum": sorted(RECOVERABLE_ENTITY_TYPES),
            },
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
            "source_drawing_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 1000,
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
            "source_drawing_ids",
            "confidence",
            "evidence_notes",
        ],
    }


def _visual_evidence_adjudication_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_index": {"type": "integer", "minimum": 0},
            "evidence_text_original": {"type": "string"},
            "status": {
                "type": "string",
                "enum": sorted(VISUAL_EVIDENCE_STATUSES),
            },
            "related_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 100,
            },
            "related_edge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 100,
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "evidence_index",
            "evidence_text_original",
            "status",
            "related_entity_ids",
            "related_edge_ids",
            "confidence",
            "reason",
        ],
    }



def _patch_entity_schema() -> dict:
    """Canonical final-entity schema used by the verifier patch plan.

    Extractor V1 responses remain cache-compatible. The verifier can normalize,
    replace, split or add entities into this canonical final schema, which adds
    exact drawing evidence for graphic-only occurrences.
    """
    schema = json.loads(json.dumps(_entity_schema()))
    schema["properties"]["source_drawing_ids"] = {
        "type": "array",
        "items": {"type": "integer"},
        "maxItems": 1000,
    }
    schema["required"].append("source_drawing_ids")
    return schema


def _entity_patch_operation_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": sorted(ENTITY_PATCH_ACTIONS),
            },
            "source_entity_id": {"type": "string"},
            "result_entities": {
                "type": "array",
                "items": _patch_entity_schema(),
                "maxItems": 12,
            },
            "evidence_indexes": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": 200,
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "operation_id",
            "action",
            "source_entity_id",
            "result_entities",
            "evidence_indexes",
            "confidence",
            "reason",
        ],
    }


def _edge_patch_operation_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": sorted(EDGE_PATCH_ACTIONS),
            },
            "source_edge_id": {"type": "string"},
            "result_edges": {
                "type": "array",
                "items": _edge_schema(),
                "maxItems": 24,
            },
            "evidence_indexes": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": 200,
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "operation_id",
            "action",
            "source_edge_id",
            "result_edges",
            "evidence_indexes",
            "confidence",
            "reason",
        ],
    }


def _patch_evidence_adjudication_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_index": {"type": "integer", "minimum": 0},
            "evidence_text_original": {"type": "string"},
            "status": {
                "type": "string",
                "enum": sorted(PATCH_EVIDENCE_STATUSES),
            },
            "raw_context_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 200,
            },
            "raw_context_edge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 400,
            },
            "final_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 200,
            },
            "final_edge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 400,
            },
            "related_operation_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 200,
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": [
            "evidence_index",
            "evidence_text_original",
            "status",
            "raw_context_entity_ids",
            "raw_context_edge_ids",
            "final_entity_ids",
            "final_edge_ids",
            "related_operation_ids",
            "confidence",
            "reason",
        ],
    }


def _verifier_patch_issue_schema() -> dict:
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
            "resolution_status": {
                "type": "string",
                "enum": sorted(VERIFIER_ISSUE_RESOLUTION_STATUSES),
            },
            "related_operation_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 300,
            },
        },
        "required": [
            "issue_type",
            "severity",
            "message",
            "entity_ids",
            "edge_ids",
            "confidence",
            "resolution_status",
            "related_operation_ids",
        ],
    }


def _verifier_schema() -> dict:
    return {
        "name": "electrical_graph_page_verifier_v3_atomic_patch_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_id": {"type": "integer"},
                "verdict": {
                    "type": "string",
                    "enum": ["apply_patch", "review_required"],
                },
                "patch_plan_version": {
                    "type": "string",
                    "enum": [GRAPH_PATCH_PLAN_VERSION],
                },
                "entity_operations": {
                    "type": "array",
                    "items": _entity_patch_operation_schema(),
                    "maxItems": 2500,
                },
                "edge_operations": {
                    "type": "array",
                    "items": _edge_patch_operation_schema(),
                    "maxItems": 6000,
                },
                "evidence_adjudications": {
                    "type": "array",
                    "items": _patch_evidence_adjudication_schema(),
                    "maxItems": 500,
                },
                "final_assertions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "all_raw_entities_decided": {"type": "boolean"},
                        "all_raw_edges_decided": {"type": "boolean"},
                        "all_visible_entities_accounted_for": {"type": "boolean"},
                        "all_visible_connections_accounted_for": {"type": "boolean"},
                        "all_entity_text_visually_supported": {"type": "boolean"},
                        "all_connection_geometry_supported": {"type": "boolean"},
                        "all_references_resolved_or_explicitly_unresolved": {
                            "type": "boolean"
                        },
                        "duplicates_preserved": {"type": "boolean"},
                        "patch_plan_safe_to_apply": {"type": "boolean"},
                    },
                    "required": [
                        "all_raw_entities_decided",
                        "all_raw_edges_decided",
                        "all_visible_entities_accounted_for",
                        "all_visible_connections_accounted_for",
                        "all_entity_text_visually_supported",
                        "all_connection_geometry_supported",
                        "all_references_resolved_or_explicitly_unresolved",
                        "duplicates_preserved",
                        "patch_plan_safe_to_apply",
                    ],
                },
                "confidence": {"type": "number"},
                "issues": {
                    "type": "array",
                    "items": _verifier_patch_issue_schema(),
                    "maxItems": 300,
                },
            },
            "required": [
                "page_id",
                "verdict",
                "patch_plan_version",
                "entity_operations",
                "edge_operations",
                "evidence_adjudications",
                "final_assertions",
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
    words: list[dict],
    glyphs: list[dict],
    drawings: list[dict],
    links: list[dict],
    image_original: bytes,
    image_rotated: bytes,
) -> list[dict]:
    system = (
        "You are the independent final graph editor for an industrial electrical "
        "schematic. Re-read the full page and produce an atomic patch plan over "
        "the extractor graph. The source may use any language, font, CAD system, "
        "orientation or drawing standard. The canonical entity types listed in "
        "the request are ALL valid, including io_reference, terminal_reference "
        "and page_reference. Never reject those types merely because they are "
        "references. A reference can remain explicitly unresolved when the visible "
        "page is real but no exact certified registry match exists; never invent a "
        "mapping. PDF links are navigation evidence only and can never substitute "
        "for local conductor drawing evidence. "
        "Return exactly one entity operation for every raw entity and exactly one "
        "edge operation for every raw edge. KEEP means the raw object is already "
        "correct. REMOVE deletes a false candidate. REPLACE normalizes or substitutes "
        "one raw entity. SPLIT separates merged repeated physical occurrences. ADD "
        "creates one omitted visible occurrence. If an entity is removed, replaced "
        "with a different ID, or split, every incident raw edge must be removed or "
        "rewired so no final edge points to a deleted ID. REWIRE may replace one raw "
        "edge with one or more final edges. "
        "Every final entity/edge must cite exact glyph, word and/or drawing IDs from "
        "the supplied registries. Literal printed text requires glyph or word evidence. "
        "A graphic-only entity requires drawing evidence. Geometry-dependent edges "
        "require local drawing IDs and two visible final endpoints. Repeated printed "
        "occurrences must remain separate. A wire crossing is not a junction unless "
        "the drawing supports it. "
        "Every numbered extractor unresolved-evidence item must be adjudicated exactly "
        "once. accounted_non_materializable may cite RAW context IDs even when those "
        "raw candidates are later removed or replaced; those links are audit context, "
        "not graph objects. Patch operations do not need an unresolved-evidence index "
        "when they correct an already extracted candidate. "
        "Mark a verifier issue resolved_by_patch_plan only when the listed operation "
        "IDs completely resolve it. Keep resolution_status=open for anything still "
        "unsafe. Return apply_patch only when the final projection is safe after all "
        "operations, rewiring and explicit-unresolved reference handling."
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
    unresolved_items = [
        {
            "evidence_index": index,
            "evidence_text_original": _clean_text(value, 4000),
        }
        for index, value in enumerate(
            extraction.get("unresolved_visual_evidence") or []
        )
    ]
    request = {
        "page_id": page["id"],
        "pdf_page_number": page["pdf_page_number"],
        "sheet_code_original": page.get("sheet_code"),
        "sheet_title_original": page.get("sheet_title"),
        "page_width_pt": page.get("page_width_pt"),
        "page_height_pt": page.get("page_height_pt"),
        "canonical_entity_types": sorted(ENTITY_TYPES),
        "canonical_relation_types": sorted(RELATION_TYPES),
        "entity_patch_actions": sorted(ENTITY_PATCH_ACTIONS),
        "edge_patch_actions": sorted(EDGE_PATCH_ACTIONS),
        "patch_plan_version": GRAPH_PATCH_PLAN_VERSION,
        "detector": detector,
        "raw_candidate_graph": extraction,
        "raw_reference_resolution": resolution,
        "numbered_unresolved_visual_evidence": unresolved_items,
        "vector_words": words,
        "source_glyphs": compact_glyphs,
        "glyph_registry_complete": len(glyphs) <= MAX_GLYPHS_IN_PROMPT,
        "drawing_registry": drawings[:MAX_DRAWINGS_IN_PROMPT],
        "drawing_registry_complete": len(drawings) <= MAX_DRAWINGS_IN_PROMPT,
        "pdf_link_registry": links,
        "entity_min_confidence": ENTITY_MIN_CONFIDENCE,
        "edge_min_confidence": EDGE_MIN_CONFIDENCE,
        "page_pass_min_confidence": PAGE_PASS_MIN_CONFIDENCE,
    }
    content = [
        {
            "type": "text",
            "text": (
                "Produce the complete atomic graph patch plan. Cover every raw "
                "entity and edge exactly once, reconnect all surviving topology, "
                "and adjudicate every numbered unresolved evidence item.\n\n"
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


def _resolve_references_for_verifier_v1(extraction: dict, registry: dict) -> dict:
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




def _sheet_reference_match(reference: Any, sheet_code: Any) -> bool:
    """Match a printed sheet/grid reference to an indexed sheet code.

    A reference such as ``205.3`` means sheet ``205`` plus a grid coordinate.
    The rule uses indexed sheet codes and visible separators; it contains no
    page-specific number, language token, or drawing template.
    """
    ref = _canonical_reference(reference)
    sheet = _canonical_reference(sheet_code)
    if not ref or not sheet:
        return False
    if ref == sheet:
        return True
    if not ref.startswith(sheet) or len(ref) <= len(sheet):
        return False
    return ref[len(sheet)] in {".", ":", "/", "-", "\\"}


def _resolve_page_reference(entity: dict, registry: dict) -> tuple[list[dict], str]:
    pages = list(registry.get("pages") or [])
    visible_values = [
        entity.get("reference_value_original"),
        entity.get("reference_context_original"),
        entity.get("label_original"),
        entity.get("tag_original"),
    ]
    text_matches = [
        page
        for page in pages
        if any(
            _sheet_reference_match(value, page.get("sheet_code"))
            for value in visible_values
            if _clean_text(value, 1000)
        )
    ]
    text_matches = list({int(row["id"]): row for row in text_matches}.values())

    entity_bbox = entity.get("bbox_pt") or []
    xref_page_ids: set[int] = set()
    for xref in registry.get("cross_references") or []:
        target_page_id = xref.get("target_page_id")
        if target_page_id is None:
            continue
        if _rect_overlap_score(
            entity_bbox,
            xref.get("source_bbox_pt") or [],
        ) >= 0.35:
            xref_page_ids.add(int(target_page_id))
    xref_matches = [
        row for row in pages if int(row.get("id") or 0) in xref_page_ids
    ]

    text_ids = {int(row["id"]) for row in text_matches}
    xref_ids = {int(row["id"]) for row in xref_matches}
    # Printed sheet/grid text is the primary exact evidence. A PDF link can
    # corroborate it, or act as fallback when the visible text is incomplete.
    if len(text_ids) == 1:
        if len(xref_ids) == 1 and text_ids == xref_ids:
            return text_matches, "sheet_code_and_pdf_link_geometry"
        return text_matches, "sheet_code_prefix"
    if len(text_ids) > 1:
        return text_matches, "ambiguous_sheet_code_candidates"
    if len(xref_ids) == 1:
        return xref_matches, "pdf_link_geometry"
    if len(xref_ids) > 1:
        return xref_matches, "ambiguous_pdf_link_candidates"
    return [], "no_page_candidate"


def _resolve_references(extraction: dict, registry: dict) -> dict:
    """Resolve exact certified references or preserve them explicitly.

    A visible reference is never discarded merely because the current
    certified registries do not contain it. In that case it remains a source-
    supported occurrence marked explicitly unresolved, and no false mapping
    edge is created.
    """
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

    entity_resolutions: list[dict] = []
    unresolved_reference_entity_ids: list[str] = []
    ambiguous_reference_entity_ids: list[str] = []
    invalid_reference_entity_ids: list[str] = []
    totals = {"bom": 0, "io": 0, "terminal": 0, "page": 0}
    status_counts: dict[str, int] = {}

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
            "candidate_count": 0,
            "resolved": True,
            "explicitly_unresolved": False,
            "accounted": True,
            "resolution_status": "not_applicable",
            "resolution_source": "",
            "reason": "",
        }

        if tag_key and entity_type in COMPONENT_ENTITY_TYPES:
            record["bom_matches"] = list(bom_by_tag.get(tag_key) or [])
            totals["bom"] += len(record["bom_matches"])

        candidates: list[dict] = []
        resolution_source = ""
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
            resolution_source = "certified_io_registry"
            if len(candidates) == 1:
                record["io_matches"] = candidates
                totals["io"] += 1

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
            resolution_source = "certified_terminal_registry"
            if len(candidates) == 1:
                record["terminal_matches"] = candidates
                totals["terminal"] += 1

        elif entity_type == "page_reference":
            candidates, resolution_source = _resolve_page_reference(
                entity,
                registry,
            )
            if len(candidates) == 1:
                record["page_matches"] = candidates
                totals["page"] += 1

        if entity_type in REFERENCE_ENTITY_TYPES:
            record["candidate_count"] = len(candidates)
            record["resolution_source"] = resolution_source
            has_visible_identity = bool(
                tag_key
                or _canonical_reference(reference_value)
                or _canonical_reference(entity.get("label_original"))
            )
            if not occurrence_id or not has_visible_identity:
                record["resolved"] = False
                record["accounted"] = False
                record["resolution_status"] = "invalid_visible_reference"
                record["reason"] = "Reference has no stable visible identity"
                invalid_reference_entity_ids.append(occurrence_id)
            elif len(candidates) == 1:
                record["resolution_status"] = "resolved_exact"
            else:
                record["resolved"] = False
                record["explicitly_unresolved"] = True
                if len(candidates) > 1:
                    record["resolution_status"] = "unresolved_ambiguous"
                    record["reason"] = (
                        "Visible reference matches more than one certified record"
                    )
                    ambiguous_reference_entity_ids.append(occurrence_id)
                else:
                    record["resolution_status"] = "unresolved_no_match"
                    record["reason"] = (
                        "Visible reference has no exact certified registry match"
                    )
                unresolved_reference_entity_ids.append(occurrence_id)

        status = str(record["resolution_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        entity_resolutions.append(record)

    all_accounted = not bool(invalid_reference_entity_ids)
    return {
        "version": "exact-or-explicit-unresolved-reference-resolution-v1",
        "page_reference_resolution_version": PAGE_REFERENCE_RESOLUTION_VERSION,
        "explicit_unresolved_reference_version": (
            EXPLICIT_UNRESOLVED_REFERENCE_VERSION
        ),
        "entity_resolutions": entity_resolutions,
        "unresolved_reference_entity_ids": unresolved_reference_entity_ids,
        "ambiguous_reference_entity_ids": ambiguous_reference_entity_ids,
        "invalid_reference_entity_ids": invalid_reference_entity_ids,
        "resolution_status_counts": status_counts,
        "match_counts": totals,
        "all_reference_entities_resolved": not bool(
            unresolved_reference_entity_ids or invalid_reference_entity_ids
        ),
        "all_reference_entities_accounted_for": all_accounted,
        "all_reference_entities_resolved_or_explicitly_unresolved": (
            all_accounted
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


def _post_verifier_candidate_adjudication(
    *,
    extraction: dict,
    verifier: dict,
    resolution: dict,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """Build a fail-closed final candidate projection after verification.

    Verifier-rejected false edges are removed. A rejected visible reference may
    remain only when it is source-supported, explicitly unresolved by the
    deterministic registry resolver, rejected solely for registry mismatch,
    and every surviving incident edge is independently verified.
    """
    entities = [
        item for item in (extraction.get("entities") or [])
        if isinstance(item, dict)
    ]
    edges = [
        item for item in (extraction.get("edges") or [])
        if isinstance(item, dict)
    ]
    entity_by_id = {
        _clean_text(item.get("occurrence_id"), 160): item
        for item in entities
        if _clean_text(item.get("occurrence_id"), 160)
    }
    edge_by_id = {
        _clean_text(item.get("edge_id"), 160): item
        for item in edges
        if _clean_text(item.get("edge_id"), 160)
    }
    resolution_by_id = {
        str(item.get("occurrence_id") or ""): item
        for item in (resolution.get("entity_resolutions") or [])
    }

    verified_entities = {
        _clean_text(value, 160)
        for value in (verifier.get("verified_entity_ids") or [])
        if _clean_text(value, 160)
    }
    verified_edges = {
        _clean_text(value, 160)
        for value in (verifier.get("verified_edge_ids") or [])
        if _clean_text(value, 160)
    }
    rejected_entities = {
        _clean_text(value, 160)
        for value in (verifier.get("rejected_entity_ids") or [])
        if _clean_text(value, 160)
    }
    rejected_edges = {
        _clean_text(value, 160)
        for value in (verifier.get("rejected_edge_ids") or [])
        if _clean_text(value, 160)
    }

    issues: list[dict] = []
    raw_entity_ids = set(entity_by_id)
    raw_edge_ids = set(edge_by_id)
    unknown_verified_entities = sorted(verified_entities - raw_entity_ids)
    unknown_verified_edges = sorted(verified_edges - raw_edge_ids)
    unknown_rejected_entities = sorted(rejected_entities - raw_entity_ids)
    unknown_rejected_edges = sorted(rejected_edges - raw_edge_ids)
    if unknown_verified_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-verified-unknown-entities",
            message="Verifier verified entity IDs not present in extraction",
            entity_ids=unknown_verified_entities,
        ))
    if unknown_verified_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-verified-unknown-edges",
            message="Verifier verified edge IDs not present in extraction",
            edge_ids=unknown_verified_edges,
        ))
    if unknown_rejected_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-rejected-unknown-entities",
            message="Verifier rejected entity IDs not present in extraction",
            entity_ids=unknown_rejected_entities,
        ))
    if unknown_rejected_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-rejected-unknown-edges",
            message="Verifier rejected edge IDs not present in extraction",
            edge_ids=unknown_rejected_edges,
        ))
    if verified_entities & rejected_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-entity-decision-conflict",
            message="Verifier both verified and rejected the same entity",
            entity_ids=sorted(verified_entities & rejected_entities),
        ))
    if verified_edges & rejected_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-edge-decision-conflict",
            message="Verifier both verified and rejected the same edge",
            edge_ids=sorted(verified_edges & rejected_edges),
        ))

    verifier_issues = [
        item for item in (verifier.get("issues") or [])
        if isinstance(item, dict)
    ]
    issue_entity_ids = {
        _clean_text(value, 160)
        for raw in verifier_issues
        for value in (raw.get("entity_ids") or [])
        if _clean_text(value, 160)
    }
    issue_edge_ids = {
        _clean_text(value, 160)
        for raw in verifier_issues
        for value in (raw.get("edge_ids") or [])
        if _clean_text(value, 160)
    }
    undocumented_edges = sorted(rejected_edges - issue_edge_ids)
    undocumented_entities = sorted(rejected_entities - issue_entity_ids)
    if undocumented_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-rejected-edge-without-audit",
            message="Rejected edges lack a structured verifier issue",
            edge_ids=undocumented_edges,
        ))
    if undocumented_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-rejected-entity-without-audit",
            message="Rejected entities lack a structured verifier issue",
            entity_ids=undocumented_entities,
        ))

    preserved_unresolved: set[str] = set()
    removed_entities: set[str] = set()
    for occurrence_id in rejected_entities:
        entity = entity_by_id.get(occurrence_id) or {}
        resolved = resolution_by_id.get(occurrence_id) or {}
        linked_issues = [
            raw for raw in verifier_issues
            if occurrence_id in {
                _clean_text(value, 160)
                for value in (raw.get("entity_ids") or [])
            }
        ]
        issue_types = {
            _clean_text(raw.get("issue_type"), 180)
            for raw in linked_issues
        }
        surviving_incident_edges = {
            edge_id
            for edge_id, edge in edge_by_id.items()
            if edge_id not in rejected_edges
            and occurrence_id in {
                _clean_text(edge.get("source_occurrence_id"), 160),
                _clean_text(edge.get("target_occurrence_id"), 160),
            }
        }
        has_source_evidence = bool(
            entity.get("source_glyph_ids") or entity.get("source_word_ids")
        )
        can_preserve = bool(
            entity.get("entity_type") in REFERENCE_ENTITY_TYPES
            and resolved.get("explicitly_unresolved")
            and resolved.get("accounted")
            and has_source_evidence
            and linked_issues
            and issue_types.issubset(REFERENCE_ONLY_REJECTION_ISSUE_TYPES)
            and surviving_incident_edges.issubset(verified_edges)
            and verifier.get(
                "all_references_resolved_or_explicitly_unresolved"
            )
        )
        if can_preserve:
            preserved_unresolved.add(occurrence_id)
            continue

        dependent_verified_edges = surviving_incident_edges & verified_edges
        if dependent_verified_edges:
            issues.append(_local_issue(
                issue_type="graph-rejected-entity-required-by-verified-edge",
                message=(
                    "A rejected entity is required by a surviving verified edge"
                ),
                entity_ids=[occurrence_id],
                edge_ids=sorted(dependent_verified_edges),
            ))
        removed_entities.add(occurrence_id)

    removed_edges = set(rejected_edges)
    for edge_id, edge in edge_by_id.items():
        if (
            _clean_text(edge.get("source_occurrence_id"), 160)
            in removed_entities
            or _clean_text(edge.get("target_occurrence_id"), 160)
            in removed_entities
        ):
            removed_edges.add(edge_id)

    final_entities = [
        item for item in entities
        if _clean_text(item.get("occurrence_id"), 160) not in removed_entities
    ]
    final_entity_ids = {
        _clean_text(item.get("occurrence_id"), 160)
        for item in final_entities
    }
    final_edges = [
        item for item in edges
        if _clean_text(item.get("edge_id"), 160) not in removed_edges
        and _clean_text(item.get("source_occurrence_id"), 160)
        in final_entity_ids
        and _clean_text(item.get("target_occurrence_id"), 160)
        in final_entity_ids
    ]
    final_edge_ids = {
        _clean_text(item.get("edge_id"), 160) for item in final_edges
    }

    unverified_final_entities = sorted(
        final_entity_ids - verified_entities - preserved_unresolved
    )
    unverified_final_edges = sorted(final_edge_ids - verified_edges)
    if unverified_final_entities:
        issues.append(_local_issue(
            issue_type="graph-post-verifier-unverified-final-entities",
            message="Final entities are neither verified nor safely preserved",
            entity_ids=unverified_final_entities,
        ))
    if unverified_final_edges:
        issues.append(_local_issue(
            issue_type="graph-post-verifier-unverified-final-edges",
            message="Final edges were not independently verified",
            edge_ids=unverified_final_edges,
        ))

    if removed_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-entity-rejections-applied",
            message="Verifier-rejected false entity candidates were removed",
            entity_ids=sorted(removed_entities),
            confidence=verifier.get("confidence") or 0.0,
            severity="info",
            source_stage="verifier_post_adjudication",
        ))
    if removed_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-edge-rejections-applied",
            message="Verifier-rejected false edge candidates were removed",
            edge_ids=sorted(removed_edges),
            confidence=verifier.get("confidence") or 0.0,
            severity="info",
            source_stage="verifier_post_adjudication",
        ))
    if preserved_unresolved:
        issues.append(_local_issue(
            issue_type="graph-explicit-unresolved-references-preserved",
            message=(
                "Visible reference entities rejected only for missing registry "
                "matches were preserved without fabricating mapping edges"
            ),
            entity_ids=sorted(preserved_unresolved),
            confidence=verifier.get("confidence") or 0.0,
            severity="info",
            source_stage="reference_resolution_adjudicator",
        ))

    audit = {
        "version": POST_VERIFIER_ADJUDICATION_VERSION,
        "raw_entity_count": len(entities),
        "raw_edge_count": len(edges),
        "final_entity_count": len(final_entities),
        "final_edge_count": len(final_edges),
        "verified_entity_ids": sorted(verified_entities),
        "verified_edge_ids": sorted(verified_edges),
        "removed_entity_ids": sorted(removed_entities),
        "removed_edge_ids": sorted(removed_edges),
        "preserved_unresolved_reference_ids": sorted(preserved_unresolved),
        "validated": not any(
            issue.get("severity") in {"high", "critical"}
            for issue in issues
        ),
    }
    return final_entities, final_edges, audit, issues


def _augment_resolution_for_recovered_entities(
    resolution: dict,
    recovered_entities: list[dict],
) -> None:
    existing = {
        str(item.get("occurrence_id") or "")
        for item in (resolution.get("entity_resolutions") or [])
    }
    added = 0
    for entity in recovered_entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        if not occurrence_id or occurrence_id in existing:
            continue
        resolution.setdefault("entity_resolutions", []).append({
            "occurrence_id": occurrence_id,
            "entity_type": _clean_text(entity.get("entity_type"), 120),
            "tag_original": _clean_text(entity.get("tag_original"), 500),
            "reference_value_original": _clean_text(
                entity.get("reference_value_original"), 500
            ),
            "bom_matches": [],
            "io_matches": [],
            "terminal_matches": [],
            "page_matches": [],
            "candidate_count": 0,
            "resolved": True,
            "explicitly_unresolved": False,
            "accounted": True,
            "resolution_status": "not_applicable",
            "resolution_source": "verifier_evidence_recovery",
            "reason": "Recovered local visual entity; no external registry resolution applies",
        })
        existing.add(occurrence_id)
        added += 1
    if added:
        counts = resolution.setdefault("resolution_status_counts", {})
        counts["not_applicable"] = int(counts.get("not_applicable") or 0) + added


def _apply_verifier_evidence_recovery(
    *,
    page: dict,
    detector: dict,
    extraction: dict,
    verifier: dict,
    resolution: dict,
    base_entities: list[dict],
    base_edges: list[dict],
    glyphs: list[dict],
    words: list[dict],
    drawings: list[dict],
    links: list[dict],
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """Validate and merge verifier-proposed omitted visual evidence.

    Recovery is intentionally narrower than extraction. It may add only local,
    directly visible entities or edges with exact PDF-point geometry and source
    evidence. It can never add external reference entities or registry mappings.
    Every extractor unresolved-evidence note must be adjudicated exactly once.
    """
    issues: list[dict] = []
    region_ids = {
        _clean_text(item.get("region_id"), 160)
        for item in (detector.get("regions") or [])
        if isinstance(item, dict) and _clean_text(item.get("region_id"), 160)
    }
    valid_glyph_ids = {int(item["glyph_id"]) for item in glyphs}
    valid_word_ids = {int(item["word_id"]) for item in words}
    valid_drawing_ids = {int(item["drawing_id"]) for item in drawings}
    valid_link_ids = {int(item["id"]) for item in links}

    entities = [dict(item) for item in base_entities]
    edges = [dict(item) for item in base_edges]
    existing_entity_ids = {
        _clean_text(item.get("occurrence_id"), 160) for item in entities
    }
    existing_edge_ids = {
        _clean_text(item.get("edge_id"), 160) for item in edges
    }
    recovered_entities: list[dict] = []
    recovered_edges: list[dict] = []
    claimed_recovery_drawing_ids: dict[int, str] = {}

    for raw in verifier.get("recovery_entities") or []:
        if not isinstance(raw, dict):
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-invalid",
                message="Verifier returned a non-object recovery entity",
            ))
            continue
        entity = dict(raw)
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        region_id = _clean_text(entity.get("region_id"), 160)
        confidence = _clamp_conf(entity.get("confidence"))
        entity_problem = False
        if not occurrence_id or occurrence_id in existing_entity_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-id-invalid",
                message="Recovery entity ID is missing or collides with an existing entity",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if entity_type not in RECOVERABLE_ENTITY_TYPES:
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-type-invalid",
                message="Recovery entity type is not permitted for local evidence recovery",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if region_id not in region_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-region-invalid",
                message="Recovery entity references an unknown detector region",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if not _bbox_valid(entity.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-bbox-invalid",
                message="Recovery entity bbox is not valid original-PDF point geometry",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if confidence + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-confidence-below-threshold",
                message="Recovery entity confidence is below the page threshold",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True

        glyph_ids = {
            int(value)
            for value in (entity.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        word_ids = {
            int(value)
            for value in (entity.get("source_word_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        drawing_ids = {
            int(value)
            for value in (entity.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if (
            glyph_ids - valid_glyph_ids
            or word_ids - valid_word_ids
            or drawing_ids - valid_drawing_ids
        ):
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-evidence-id-invalid",
                message="Recovery entity cites evidence IDs absent from the source registries",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        visible_source_text_fields = [
            field for field in SOURCE_VISIBLE_ENTITY_TEXT_FIELDS
            if _clean_text(entity.get(field), 1000)
        ]
        semantic_annotation_fields = [
            field for field in SEMANTIC_ENTITY_ANNOTATION_FIELDS
            if _clean_text(entity.get(field), 1000)
        ]
        if visible_source_text_fields and not (glyph_ids or word_ids):
            issues.append(_local_issue(
                issue_type="graph-recovery-entity-text-evidence-missing",
                message=(
                    "Recovery entity claims literal printed text without exact "
                    "glyph or word evidence"
                ),
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if not visible_source_text_fields and not drawing_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-graphic-evidence-missing",
                message=(
                    "Graphic-only recovery entity has no exact source drawing "
                    "evidence"
                ),
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        if entity_type in COMPONENT_ENTITY_TYPES and not _clean_text(
            entity.get("tag_original"), 500
        ):
            issues.append(_local_issue(
                issue_type="graph-recovery-component-tag-missing",
                message="Recovered component-like entity has no visible component tag",
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
            entity_problem = True
        for drawing_id in drawing_ids:
            owner = claimed_recovery_drawing_ids.get(drawing_id)
            if owner and owner != occurrence_id:
                issues.append(_local_issue(
                    issue_type="graph-recovery-drawing-evidence-ambiguous",
                    message="One drawing ID was claimed by multiple recovered entities",
                    entity_ids=sorted({owner, occurrence_id}),
                    confidence=confidence,
                ))
                entity_problem = True
            else:
                claimed_recovery_drawing_ids[drawing_id] = occurrence_id
        if entity_problem:
            continue
        entity["source_glyph_ids"] = sorted(glyph_ids)
        entity["source_word_ids"] = sorted(word_ids)
        entity["source_drawing_ids"] = sorted(drawing_ids)
        entity["recovery_evidence"] = {
            "version": VERIFIER_EVIDENCE_RECOVERY_VERSION,
            "source": "independent_verifier",
            "visible_source_text_fields": visible_source_text_fields,
            "semantic_annotation_fields": semantic_annotation_fields,
            "text_evidence_mode": (
                "literal_source_text"
                if visible_source_text_fields
                else "graphic_with_semantic_annotation"
            ),
            "validated": True,
        }
        recovered_entities.append(entity)
        existing_entity_ids.add(occurrence_id)

    merged_entity_ids = existing_entity_ids
    for raw in verifier.get("recovery_edges") or []:
        if not isinstance(raw, dict):
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-invalid",
                message="Verifier returned a non-object recovery edge",
            ))
            continue
        edge = dict(raw)
        edge_id = _clean_text(edge.get("edge_id"), 160)
        source_id = _clean_text(edge.get("source_occurrence_id"), 160)
        target_id = _clean_text(edge.get("target_occurrence_id"), 160)
        relation_type = _clean_text(edge.get("relation_type"), 120)
        confidence = _clamp_conf(edge.get("confidence"))
        edge_problem = False
        if not edge_id or edge_id in existing_edge_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-id-invalid",
                message="Recovery edge ID is missing or collides with an existing edge",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if relation_type not in RELATION_TYPES:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-relation-invalid",
                message="Recovery edge relation type is invalid",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if source_id not in merged_entity_ids or target_id not in merged_entity_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-endpoint-missing",
                message="Recovery edge does not have two existing visible endpoints",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if source_id and source_id == target_id:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-self-reference",
                message="Recovery edge connects an entity to itself",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if not _edge_bbox_valid(edge.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-bbox-invalid",
                message="Recovery edge bbox is not valid original-PDF point geometry",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if confidence + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-confidence-below-threshold",
                message="Recovery edge confidence is below the page threshold",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        drawing_ids = {
            int(value)
            for value in (edge.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        glyph_ids = {
            int(value)
            for value in (edge.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        link_ids = {
            int(value)
            for value in (edge.get("source_link_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if (
            drawing_ids - valid_drawing_ids
            or glyph_ids - valid_glyph_ids
            or link_ids - valid_link_ids
        ):
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-evidence-id-invalid",
                message="Recovery edge cites evidence IDs absent from the source registries",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if relation_type in GEOMETRY_REQUIRED_RELATIONS and not drawing_ids:
            issues.append(_local_issue(
                issue_type="graph-recovery-edge-geometry-evidence-missing",
                message="Recovery edge has no exact local drawing evidence",
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
            edge_problem = True
        if edge_problem:
            continue
        edge["source_drawing_ids"] = sorted(drawing_ids)
        edge["source_glyph_ids"] = sorted(glyph_ids)
        edge["source_link_ids"] = sorted(link_ids)
        edge["recovery_evidence"] = {
            "version": VERIFIER_EVIDENCE_RECOVERY_VERSION,
            "source": "independent_verifier",
            "validated": True,
        }
        recovered_edges.append(edge)
        existing_edge_ids.add(edge_id)

    recovered_entity_ids = {
        _clean_text(item.get("occurrence_id"), 160)
        for item in recovered_entities
    }
    recovered_edge_ids = {
        _clean_text(item.get("edge_id"), 160) for item in recovered_edges
    }
    unresolved_items = [
        _clean_text(value, 4000)
        for value in (extraction.get("unresolved_visual_evidence") or [])
    ]
    adjudication_rows = [
        item for item in (verifier.get("visual_evidence_adjudications") or [])
        if isinstance(item, dict)
    ]
    seen_indexes: set[int] = set()
    referenced_recovery_entities: set[str] = set()
    referenced_recovery_edges: set[str] = set()
    normalized_adjudications: list[dict] = []
    for item in adjudication_rows:
        try:
            index = int(item.get("evidence_index"))
        except Exception:
            index = -1
        text = _clean_text(item.get("evidence_text_original"), 4000)
        status = _clean_text(item.get("status"), 120)
        confidence = _clamp_conf(item.get("confidence"))
        entity_ids = {
            _clean_text(value, 160)
            for value in (item.get("related_entity_ids") or [])
            if _clean_text(value, 160)
        }
        edge_ids = {
            _clean_text(value, 160)
            for value in (item.get("related_edge_ids") or [])
            if _clean_text(value, 160)
        }
        row_problem = False
        if index < 0 or index >= len(unresolved_items) or index in seen_indexes:
            issues.append(_local_issue(
                issue_type="graph-visual-evidence-adjudication-index-invalid",
                message="Visual evidence adjudication index is missing, duplicate or out of range",
                confidence=confidence,
            ))
            row_problem = True
        else:
            seen_indexes.add(index)
            if text != unresolved_items[index]:
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-adjudication-text-canonicalized",
                    message=(
                        "The evidence index was valid; the audit text was "
                        "canonicalized from the extractor ledger"
                    ),
                    confidence=confidence,
                    severity="info",
                    source_stage="verifier_evidence_recovery",
                ))
                text = unresolved_items[index]
        if status not in VISUAL_EVIDENCE_STATUSES:
            issues.append(_local_issue(
                issue_type="graph-visual-evidence-adjudication-status-invalid",
                message="Visual evidence adjudication returned an invalid status",
                confidence=confidence,
            ))
            row_problem = True
        if confidence + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-visual-evidence-adjudication-confidence-low",
                message="Visual evidence adjudication confidence is below threshold",
                confidence=confidence,
            ))
            row_problem = True
        if status == "accounted_existing_graph":
            if not (entity_ids or edge_ids):
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-existing-link-missing",
                    message="Existing-graph adjudication must identify at least one supporting entity or edge",
                    confidence=confidence,
                ))
                row_problem = True
            if not entity_ids.issubset(merged_entity_ids):
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-existing-entity-link-invalid",
                    message="Evidence adjudication cites an entity absent from the final graph",
                    entity_ids=sorted(entity_ids),
                    confidence=confidence,
                ))
                row_problem = True
            if not edge_ids.issubset(existing_edge_ids):
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-existing-edge-link-invalid",
                    message="Evidence adjudication cites an edge absent from the final graph",
                    edge_ids=sorted(edge_ids),
                    confidence=confidence,
                ))
                row_problem = True
            referenced_recovery_entities.update(entity_ids & recovered_entity_ids)
            referenced_recovery_edges.update(edge_ids & recovered_edge_ids)
        elif status == "recovered_entity":
            if not entity_ids or not entity_ids.issubset(recovered_entity_ids):
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-recovery-entity-link-invalid",
                    message="Evidence adjudication does not point to valid recovered entities",
                    entity_ids=sorted(entity_ids),
                    confidence=confidence,
                ))
                row_problem = True
            referenced_recovery_entities.update(entity_ids)
        elif status == "recovered_edge":
            if not edge_ids or not edge_ids.issubset(recovered_edge_ids):
                issues.append(_local_issue(
                    issue_type="graph-visual-evidence-recovery-edge-link-invalid",
                    message="Evidence adjudication does not point to valid recovered edges",
                    edge_ids=sorted(edge_ids),
                    confidence=confidence,
                ))
                row_problem = True
            referenced_recovery_edges.update(edge_ids)
        elif status == "accounted_non_materializable":
            # Context links are audit pointers only: they explain where an
            # annotation/boundary belongs without creating a new entity/edge.
            # They are valid only when every cited ID already exists in the
            # final graph.
            invalid_context_entities = entity_ids - merged_entity_ids
            invalid_context_edges = edge_ids - existing_edge_ids
            if invalid_context_entities or invalid_context_edges:
                issues.append(_local_issue(
                    issue_type=(
                        "graph-visual-evidence-nonmaterializable-context-invalid"
                    ),
                    message=(
                        "Non-materializable evidence cites context IDs absent "
                        "from the final graph"
                    ),
                    entity_ids=sorted(invalid_context_entities),
                    edge_ids=sorted(invalid_context_edges),
                    confidence=confidence,
                ))
                row_problem = True
        elif status == "still_unresolved":
            issues.append(_local_issue(
                issue_type="graph-visual-evidence-still-unresolved",
                message="A visible evidence item remains unresolved after independent recovery",
                entity_ids=sorted(entity_ids),
                edge_ids=sorted(edge_ids),
                confidence=confidence,
            ))
            row_problem = True
        normalized_adjudications.append({
            "evidence_index": index,
            "evidence_text_original": text,
            "status": status,
            "related_entity_ids": sorted(entity_ids),
            "related_edge_ids": sorted(edge_ids),
            "confidence": confidence,
            "reason": _clean_text(item.get("reason"), 1600),
            "context_link_policy": (
                {
                    "version": NONMATERIALIZABLE_CONTEXT_VERSION,
                    "materializes_new_graph_items": False,
                    "validated_existing_context_only": not row_problem,
                }
                if status == "accounted_non_materializable"
                else {}
            ),
            "validated": not row_problem,
        })

    expected_indexes = set(range(len(unresolved_items)))
    if seen_indexes != expected_indexes:
        issues.append(_local_issue(
            issue_type="graph-visual-evidence-adjudication-accounting-mismatch",
            message="Verifier did not adjudicate every extractor unresolved-evidence item exactly once",
            confidence=verifier.get("confidence") or 0.0,
        ))
    unlinked_entities = recovered_entity_ids - referenced_recovery_entities
    unlinked_edges = recovered_edge_ids - referenced_recovery_edges
    if unlinked_entities:
        issues.append(_local_issue(
            issue_type="graph-recovery-entities-not-linked-to-gap",
            message="Recovered entities are not linked to any numbered visual evidence gap",
            entity_ids=sorted(unlinked_entities),
            confidence=verifier.get("confidence") or 0.0,
        ))
    if unlinked_edges:
        issues.append(_local_issue(
            issue_type="graph-recovery-edges-not-linked-to-gap",
            message="Recovered edges are not linked to any numbered visual evidence gap",
            edge_ids=sorted(unlinked_edges),
            confidence=verifier.get("confidence") or 0.0,
        ))

    entities.extend(recovered_entities)
    edges.extend(recovered_edges)
    _augment_resolution_for_recovered_entities(resolution, recovered_entities)
    blocking = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    audit = {
        "version": VERIFIER_EVIDENCE_RECOVERY_VERSION,
        "raw_unresolved_visual_evidence_count": len(unresolved_items),
        "recovered_entity_ids": sorted(recovered_entity_ids),
        "recovered_edge_ids": sorted(recovered_edge_ids),
        "recovered_entity_count": len(recovered_entities),
        "recovered_edge_count": len(recovered_edges),
        "visual_evidence_adjudications": normalized_adjudications,
        "all_visual_evidence_adjudicated": seen_indexes == expected_indexes,
        "all_recovery_candidates_linked": not (
            unlinked_entities or unlinked_edges
        ),
        "validated": not blocking,
    }
    if recovered_entities or recovered_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-evidence-recovery-applied",
            message="Independent verifier recovery candidates were validated and merged",
            entity_ids=sorted(recovered_entity_ids),
            edge_ids=sorted(recovered_edge_ids),
            confidence=verifier.get("confidence") or 0.0,
            severity="info",
            source_stage="verifier_evidence_recovery",
        ))
    if unresolved_items and audit["validated"]:
        issues.append(_local_issue(
            issue_type="graph-extractor-evidence-fully-adjudicated",
            message="Every extractor unresolved-evidence note was independently and deterministically adjudicated",
            confidence=verifier.get("confidence") or 0.0,
            severity="info",
            source_stage="verifier_evidence_recovery",
        ))
    return entities, edges, audit, issues


def _adjudicate_detector_region_bboxes(
    *,
    page: dict,
    detector: dict,
    entities: list[dict],
    edges: list[dict],
    final_assertions: Optional[dict] = None,
) -> tuple[dict, list[dict]]:
    """Reconcile detector coordinate frames against final PDF-point evidence.

    Detector models can occasionally report rendered-image pixel coordinates
    even though the schema requests PDF points. A region is accepted as-is only
    when its bbox is in page bounds and covers the majority of final entity
    centers assigned to that region. Otherwise a conservative bbox is derived
    from the union of final entity geometry (or incident edge geometry when the
    region contains no entity). No semantic classification depends on this box.
    """
    issues: list[dict] = []
    audits: list[dict] = []
    entity_by_id = {
        _clean_text(item.get("occurrence_id"), 160): item
        for item in entities
        if _clean_text(item.get("occurrence_id"), 160)
    }
    page_width = float(page.get("page_width_pt") or 0.0)
    page_height = float(page.get("page_height_pt") or 0.0)

    def rect_for_edge(edge: dict) -> Optional[fitz.Rect]:
        if not _edge_bbox_valid(edge.get("bbox_pt"), page):
            return None
        x0, y0, x1, y1 = [float(x) for x in edge.get("bbox_pt")]
        pad = 1.5
        if x0 == x1:
            x0 -= pad
            x1 += pad
        if y0 == y1:
            y0 -= pad
            y1 += pad
        return fitz.Rect(x0, y0, x1, y1)

    for region in detector.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_id = _clean_text(region.get("region_id"), 160)
        original = list(region.get("bbox_pt") or [])
        entity_rects: list[fitz.Rect] = []
        entity_centers: list[tuple[float, float]] = []
        region_entity_ids: list[str] = []
        for entity in entities:
            if _clean_text(entity.get("region_id"), 160) != region_id:
                continue
            if not _bbox_valid(entity.get("bbox_pt"), page):
                continue
            rect = _rect_from(entity.get("bbox_pt"))
            entity_rects.append(rect)
            entity_centers.append(
                ((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
            )
            region_entity_ids.append(
                _clean_text(entity.get("occurrence_id"), 160)
            )
        edge_rects: list[fitz.Rect] = []
        region_edge_ids: list[str] = []
        for edge in edges:
            source = entity_by_id.get(
                _clean_text(edge.get("source_occurrence_id"), 160)
            ) or {}
            target = entity_by_id.get(
                _clean_text(edge.get("target_occurrence_id"), 160)
            ) or {}
            if region_id not in {
                _clean_text(source.get("region_id"), 160),
                _clean_text(target.get("region_id"), 160),
            }:
                continue
            edge_rect = rect_for_edge(edge)
            if edge_rect is not None:
                edge_rects.append(edge_rect)
                region_edge_ids.append(_clean_text(edge.get("edge_id"), 160))

        original_valid = _bbox_valid(original, page)
        coverage_ratio = 1.0
        if original_valid and entity_centers:
            raw_rect = _rect_from(original)
            covered = sum(
                1 for x, y in entity_centers
                if raw_rect.x0 <= x <= raw_rect.x1
                and raw_rect.y0 <= y <= raw_rect.y1
            )
            coverage_ratio = covered / max(1, len(entity_centers))
        needs_recovery = bool(
            not original_valid
            or (entity_centers and coverage_ratio < 0.60)
        )
        if not needs_recovery:
            audits.append({
                "region_id": region_id,
                "original_bbox_pt": original,
                "final_bbox_pt": original,
                "reason": "original_pdf_point_bbox_valid",
                "entity_center_coverage_ratio": round(coverage_ratio, 4),
                "source_entity_ids": region_entity_ids,
                "source_edge_ids": region_edge_ids,
                "validated": True,
            })
            continue

        source_rects = entity_rects or edge_rects
        if not source_rects:
            expected_components = int(
                region.get("visible_component_count") or 0
            )
            expected_connections = int(
                region.get("visible_connection_count") or 0
            )
            assertions = final_assertions or {}
            final_coverage_asserted = bool(
                assertions.get("all_visible_entities_accounted_for")
                and assertions.get("all_visible_connections_accounted_for")
            )
            detector_low_confidence = bool(
                _clamp_conf(region.get("confidence")) + 1e-9
                < PAGE_PASS_MIN_CONFIDENCE
            )
            empty_region_is_preliminary_false_positive = bool(
                (expected_components == 0 and expected_connections == 0)
                or (final_coverage_asserted and detector_low_confidence)
            )
            if empty_region_is_preliminary_false_positive:
                issues.append(_local_issue(
                    issue_type="graph-empty-detector-region-superseded",
                    message=(
                        "A preliminary detector region has no final graph "
                        "objects and was superseded by complete final-projection "
                        "coverage validation"
                    ),
                    confidence=region.get("confidence") or 0.0,
                    severity="warning" if (
                        expected_components or expected_connections
                    ) else "info",
                    source_stage="detector_preliminary_audit",
                ))
                audits.append({
                    "region_id": region_id,
                    "original_bbox_pt": original,
                    "final_bbox_pt": [],
                    "reason": "empty_preliminary_region_superseded",
                    "entity_center_coverage_ratio": round(coverage_ratio, 4),
                    "source_entity_ids": [],
                    "source_edge_ids": [],
                    "detector_visible_component_count": expected_components,
                    "detector_visible_connection_count": expected_connections,
                    "validated": True,
                })
                continue
            issues.append(_local_issue(
                issue_type="graph-region-bbox-unrecoverable",
                message=(
                    "Detector region bbox is invalid or inconsistent and no "
                    "final PDF-point entity/edge evidence can recover it"
                ),
                confidence=region.get("confidence") or 0.0,
            ))
            audits.append({
                "region_id": region_id,
                "original_bbox_pt": original,
                "final_bbox_pt": [],
                "reason": "no_final_geometry_evidence",
                "entity_center_coverage_ratio": round(coverage_ratio, 4),
                "source_entity_ids": region_entity_ids,
                "source_edge_ids": region_edge_ids,
                "detector_visible_component_count": expected_components,
                "detector_visible_connection_count": expected_connections,
                "validated": False,
            })
            continue
        union = fitz.Rect(source_rects[0])
        for rect in source_rects[1:]:
            union.include_rect(rect)
        margin = 6.0
        union = fitz.Rect(
            max(0.0, union.x0 - margin),
            max(0.0, union.y0 - margin),
            min(page_width, union.x1 + margin) if page_width > 0 else union.x1 + margin,
            min(page_height, union.y1 + margin) if page_height > 0 else union.y1 + margin,
        )
        final_bbox = _rect_list(union)
        if not _bbox_valid(final_bbox, page):
            issues.append(_local_issue(
                issue_type="graph-region-bbox-recovery-invalid",
                message="Derived detector region bbox is not valid PDF-point geometry",
                confidence=region.get("confidence") or 0.0,
            ))
            validated = False
        else:
            region["bbox_pt"] = final_bbox
            region["bbox_adjudication"] = {
                "version": REGION_BBOX_ADJUDICATION_VERSION,
                "original_bbox_pt": original,
                "final_bbox_pt": final_bbox,
                "original_bbox_valid": original_valid,
                "original_entity_center_coverage_ratio": round(
                    coverage_ratio, 4
                ),
                "source_entity_ids": region_entity_ids,
                "source_edge_ids": region_edge_ids,
                "validated": True,
            }
            issues.append(_local_issue(
                issue_type="graph-region-bbox-adjudicated",
                message=(
                    "Detector region coordinates were reconciled to the final "
                    "original-PDF point evidence frame"
                ),
                entity_ids=region_entity_ids,
                edge_ids=region_edge_ids,
                confidence=region.get("confidence") or 0.0,
                severity="info",
                source_stage="region_bbox_adjudicator",
            ))
            validated = True
        audits.append({
            "region_id": region_id,
            "original_bbox_pt": original,
            "final_bbox_pt": final_bbox,
            "reason": (
                "original_out_of_page"
                if not original_valid
                else "original_bbox_did_not_cover_final_region_entities"
            ),
            "entity_center_coverage_ratio": round(coverage_ratio, 4),
            "source_entity_ids": region_entity_ids,
            "source_edge_ids": region_edge_ids,
            "validated": validated,
        })

    return {
        "version": REGION_BBOX_ADJUDICATION_VERSION,
        "regions": audits,
        "adjudicated_region_count": sum(
            1 for item in audits
            if item.get("reason") != "original_pdf_point_bbox_valid"
            and item.get("validated")
        ),
        "validated": not any(
            issue.get("severity") in {"high", "critical"}
            for issue in issues
        ),
    }, issues


def _normalize_verifier_issue_after_adjudication(
    raw: Any,
    *,
    adjudication: dict,
    resolution: dict,
    recovery_audit: Optional[dict] = None,
) -> dict:
    issue = _normalize_issue(
        raw,
        default_type="graph-verifier-issue",
        source_stage="verifier",
    )
    issue_type = _clean_text(issue.get("issue_type"), 180)
    entity_ids = set(issue.get("entity_ids") or [])
    edge_ids = set(issue.get("edge_ids") or [])
    removed_edges = set(adjudication.get("removed_edge_ids") or [])
    preserved_unresolved = set(
        adjudication.get("preserved_unresolved_reference_ids") or []
    )
    recovery_audit = recovery_audit or {}
    recovered_entities = set(recovery_audit.get("recovered_entity_ids") or [])
    recovered_edges = set(recovery_audit.get("recovered_edge_ids") or [])
    resolution_by_id = {
        str(item.get("occurrence_id") or ""): item
        for item in (resolution.get("entity_resolutions") or [])
    }

    if edge_ids and edge_ids.issubset(removed_edges):
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_post_adjudication"
        issue["message"] = (
            issue["message"]
            + " [Resolved by removing the rejected candidate edge.]"
        )[:1600]
        issue["adjudication"] = {
            "validated": True,
            "action": "removed_rejected_edges",
        }
        return issue

    if entity_ids and entity_ids.issubset(recovered_entities):
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_evidence_recovery"
        issue["message"] = (
            issue["message"]
            + " [Resolved by a validated verifier recovery entity.]"
        )[:1600]
        issue["adjudication"] = {
            "validated": True,
            "action": "added_recovery_entities",
        }
        return issue
    if edge_ids and edge_ids.issubset(recovered_edges):
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_evidence_recovery"
        issue["message"] = (
            issue["message"]
            + " [Resolved by a validated verifier recovery edge.]"
        )[:1600]
        issue["adjudication"] = {
            "validated": True,
            "action": "added_recovery_edges",
        }
        return issue
    missing_issue_recovered = bool(
        (
            issue_type == "missing_visible_entity"
            and int(recovery_audit.get("recovered_entity_count") or 0) > 0
        )
        or (
            issue_type == "missing_visible_connections"
            and recovery_audit.get("all_visual_evidence_adjudicated")
        )
    )
    if (
        not entity_ids
        and not edge_ids
        and issue_type in {"missing_visible_entity", "missing_visible_connections"}
        and recovery_audit.get("validated")
        and missing_issue_recovered
    ):
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_evidence_recovery"
        issue["message"] = (
            issue["message"]
            + " [Resolved by complete structured visual-evidence adjudication.]"
        )[:1600]
        issue["adjudication"] = {
            "validated": True,
            "action": "structured_visual_evidence_recovery",
        }
        return issue

    if (
        entity_ids
        and entity_ids.issubset(preserved_unresolved)
        and issue_type in REFERENCE_ONLY_REJECTION_ISSUE_TYPES
        and all(
            bool((resolution_by_id.get(entity_id) or {}).get(
                "explicitly_unresolved"
            ))
            for entity_id in entity_ids
        )
    ):
        issue["severity"] = "warning"
        issue["source_stage"] = "reference_resolution_adjudicator"
        issue["message"] = (
            issue["message"]
            + " [Preserved as an explicit unresolved visible reference; "
            "no certified mapping edge was fabricated.]"
        )[:1600]
        issue["adjudication"] = {
            "validated": True,
            "action": "preserved_explicit_unresolved_reference",
        }
        return issue

    if issue.get("severity") == "warning" and entity_ids:
        if all(
            bool((resolution_by_id.get(entity_id) or {}).get("resolved"))
            for entity_id in entity_ids
        ):
            issue["severity"] = "info"
            issue["source_stage"] = "reference_resolution_adjudicator"
            issue["message"] = (
                issue["message"]
                + " [Superseded by deterministic exact reference resolution.]"
            )[:1600]
            issue["adjudication"] = {
                "validated": True,
                "action": "resolved_after_reference_reconciliation",
            }
    return issue


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

    entities, edges, adjudication, adjudication_issues = (
        _post_verifier_candidate_adjudication(
            extraction=extraction,
            verifier=verifier,
            resolution=resolution,
        )
    )
    issues.extend(adjudication_issues)
    pre_recovery_entity_count = len(entities)
    pre_recovery_edge_count = len(edges)
    entities, edges, recovery_audit, recovery_issues = (
        _apply_verifier_evidence_recovery(
            page=page,
            detector=detector,
            extraction=extraction,
            verifier=verifier,
            resolution=resolution,
            base_entities=entities,
            base_edges=edges,
            glyphs=glyphs,
            words=words,
            drawings=drawings,
            links=links,
        )
    )
    issues.extend(recovery_issues)
    adjudication["pre_recovery_final_entity_count"] = (
        pre_recovery_entity_count
    )
    adjudication["pre_recovery_final_edge_count"] = pre_recovery_edge_count
    adjudication["final_entity_count"] = len(entities)
    adjudication["final_edge_count"] = len(edges)
    adjudication["recovered_entity_ids"] = recovery_audit.get(
        "recovered_entity_ids"
    ) or []
    adjudication["recovered_edge_ids"] = recovery_audit.get(
        "recovered_edge_ids"
    ) or []
    adjudication["evidence_recovery_validated"] = bool(
        recovery_audit.get("validated")
    )
    adjudication["validated"] = bool(
        adjudication.get("validated") and recovery_audit.get("validated")
    )
    adjudication["verifier_evidence_recovery"] = recovery_audit
    extraction["verifier_evidence_recovery"] = recovery_audit

    geometry_audit, geometry_issues = _reconcile_graph_geometry_from_evidence(
        page=page,
        entities=entities,
        edges=edges,
        glyphs=glyphs,
        words=words,
        drawings=drawings,
    )
    issues.extend(geometry_issues)
    adjudication["source_evidence_geometry_reconciliation"] = geometry_audit
    extraction["source_evidence_geometry_reconciliation"] = geometry_audit
    extraction["post_verifier_adjudication"] = adjudication

    region_bbox_audit, region_bbox_issues = (
        _adjudicate_detector_region_bboxes(
            page=page,
            detector=detector,
            entities=entities,
            edges=edges,
        )
    )
    adjudication["region_bbox_adjudication"] = region_bbox_audit
    extraction["region_bbox_adjudication"] = region_bbox_audit
    issues.extend(region_bbox_issues)

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
        issues.append(_normalize_verifier_issue_after_adjudication(
            raw,
            adjudication=adjudication,
            resolution=resolution,
            recovery_audit=recovery_audit,
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
    if _clamp_conf(detector.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
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
        region_kind = _clean_text(region.get("region_kind"), 120)
        if not rid or rid in region_ids:
            issues.append(_local_issue(
                issue_type="graph-region-id-invalid",
                message="Missing or duplicate detector region_id",
            ))
        region_ids.append(rid)
        if region_kind not in REGION_KINDS:
            issues.append(_local_issue(
                issue_type="graph-region-kind-invalid",
                message=f"Invalid graph region kind for {rid}",
            ))
        if not _bbox_valid(region.get("bbox_pt"), page):
            issues.append(_local_issue(
                issue_type="graph-region-bbox-invalid",
                message=f"Invalid graph region bbox for {rid}",
            ))
        region_threshold = (
            ENTITY_MIN_CONFIDENCE
            if region_kind == "off_page_reference"
            and int(region.get("visible_connection_count") or 0) == 0
            else PAGE_PASS_MIN_CONFIDENCE
        )
        if _clamp_conf(region.get("confidence")) + 1e-9 < region_threshold:
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
    if _clamp_conf(extraction.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-extractor-confidence-below-threshold",
            message="Extractor confidence is below page threshold",
            confidence=extraction.get("confidence") or 0.0,
        ))
    if extraction.get("unresolved_visual_evidence"):
        if (
            recovery_audit.get("validated")
            and recovery_audit.get("all_visual_evidence_adjudicated")
            and recovery_audit.get("all_recovery_candidates_linked")
        ):
            issues.append(_local_issue(
                issue_type="graph-extractor-evidence-explicitly-accounted",
                message=(
                    "Every extractor evidence note is accounted for by the "
                    "structured verifier recovery/adjudication ledger"
                ),
                confidence=extraction.get("confidence") or 0.0,
                severity="info",
                source_stage="verifier_evidence_recovery",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-unresolved-visual-evidence",
                message="Extractor returned visual evidence that remains unresolved",
                confidence=extraction.get("confidence") or 0.0,
            ))

    valid_glyph_ids = {int(item["glyph_id"]) for item in glyphs}
    valid_word_ids = {int(item["word_id"]) for item in words}
    valid_drawing_ids = {int(item["drawing_id"]) for item in drawings}
    valid_link_ids = {int(item["id"]) for item in links}
    region_id_set = set(region_ids)

    occurrence_ids: list[str] = []
    entity_by_id: dict[str, dict] = {}
    entity_validation_failed: set[str] = set()
    resolution_by_id = {
        str(item.get("occurrence_id") or ""): item
        for item in (resolution.get("entity_resolutions") or [])
    }

    def entity_fail(
        issue_type: str,
        message: str,
        occurrence_id: str,
        confidence: float,
    ) -> None:
        entity_validation_failed.add(occurrence_id)
        issues.append(_local_issue(
            issue_type=issue_type,
            message=message,
            entity_ids=[occurrence_id] if occurrence_id else [],
            confidence=confidence,
        ))

    for entity in entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        confidence = _clamp_conf(entity.get("confidence"))
        if not occurrence_id or occurrence_id in occurrence_ids:
            entity_fail(
                "graph-entity-id-invalid",
                "Missing or duplicate entity occurrence_id",
                occurrence_id,
                confidence,
            )
        occurrence_ids.append(occurrence_id)
        entity_by_id[occurrence_id] = entity
        if entity_type not in ENTITY_TYPES:
            entity_fail(
                "graph-entity-type-invalid",
                f"Invalid entity type in {occurrence_id}",
                occurrence_id,
                confidence,
            )
        if _clean_text(entity.get("region_id"), 160) not in region_id_set:
            entity_fail(
                "graph-entity-region-invalid",
                f"Entity {occurrence_id} references an unknown region",
                occurrence_id,
                confidence,
            )
        if not _bbox_valid(entity.get("bbox_pt"), page):
            entity_fail(
                "graph-entity-bbox-invalid",
                f"Invalid entity bbox in {occurrence_id}",
                occurrence_id,
                confidence,
            )
        if confidence + 1e-9 < ENTITY_MIN_CONFIDENCE:
            entity_fail(
                "graph-entity-confidence-below-threshold",
                f"Low-confidence entity {occurrence_id}",
                occurrence_id,
                confidence,
            )
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
        source_drawing_ids = {
            int(value)
            for value in (entity.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if (
            source_glyph_ids - valid_glyph_ids
            or source_word_ids - valid_word_ids
            or source_drawing_ids - valid_drawing_ids
        ):
            entity_fail(
                "graph-entity-evidence-id-invalid",
                f"Entity {occurrence_id} cites invalid glyph/word/drawing IDs",
                occurrence_id,
                confidence,
            )
        visible_source_text_fields = [
            field for field in SOURCE_VISIBLE_ENTITY_TEXT_FIELDS
            if _clean_text(entity.get(field), 1000)
        ]
        if visible_source_text_fields and not (
            source_glyph_ids or source_word_ids
        ):
            entity_fail(
                "graph-entity-text-evidence-missing",
                (
                    f"Entity {occurrence_id} claims literal printed text "
                    "without source glyph/word evidence"
                ),
                occurrence_id,
                confidence,
            )
        entity["source_text_evidence_policy"] = {
            "version": "graph-source-visible-text-fields-v2",
            "visible_source_text_fields": visible_source_text_fields,
            "semantic_annotation_fields": [
                field for field in SEMANTIC_ENTITY_ANNOTATION_FIELDS
                if _clean_text(entity.get(field), 1000)
            ],
            "validated": bool(
                not visible_source_text_fields
                or source_glyph_ids
                or source_word_ids
            ),
        }
        if entity_type in COMPONENT_ENTITY_TYPES and not _clean_text(
            entity.get("tag_original"), 500
        ):
            entity_fail(
                "graph-component-tag-missing",
                f"Component-like entity {occurrence_id} has no visible tag",
                occurrence_id,
                confidence,
            )
        if entity_type in REFERENCE_ENTITY_TYPES:
            resolved = resolution_by_id.get(occurrence_id) or {}
            if not resolved.get("accounted"):
                entity_fail(
                    "graph-reference-entity-not-accounted",
                    (
                        f"Reference entity {occurrence_id} is neither resolved "
                        "nor safely explicit"
                    ),
                    occurrence_id,
                    confidence,
                )
            entity["reference_resolution"] = {
                key: resolved.get(key)
                for key in (
                    "resolved",
                    "explicitly_unresolved",
                    "resolution_status",
                    "resolution_source",
                    "reason",
                )
            }

    edge_ids: list[str] = []
    edge_validation_failed: set[str] = set()

    def edge_fail(
        issue_type: str,
        message: str,
        edge_id: str,
        confidence: float,
    ) -> None:
        edge_validation_failed.add(edge_id)
        issues.append(_local_issue(
            issue_type=issue_type,
            message=message,
            edge_ids=[edge_id] if edge_id else [],
            confidence=confidence,
        ))

    for edge in edges:
        edge_id = _clean_text(edge.get("edge_id"), 160)
        relation_type = _clean_text(edge.get("relation_type"), 120)
        source_id = _clean_text(edge.get("source_occurrence_id"), 160)
        target_id = _clean_text(edge.get("target_occurrence_id"), 160)
        confidence = _clamp_conf(edge.get("confidence"))
        if not edge_id or edge_id in edge_ids:
            edge_fail(
                "graph-edge-id-invalid",
                "Missing or duplicate edge_id",
                edge_id,
                confidence,
            )
        edge_ids.append(edge_id)
        if relation_type not in RELATION_TYPES:
            edge_fail(
                "graph-edge-relation-invalid",
                f"Invalid relation type in edge {edge_id}",
                edge_id,
                confidence,
            )
        if source_id not in entity_by_id or target_id not in entity_by_id:
            edge_fail(
                "graph-edge-endpoint-missing",
                f"Edge {edge_id} references a missing entity",
                edge_id,
                confidence,
            )
        if source_id and source_id == target_id:
            edge_fail(
                "graph-edge-self-reference",
                f"Edge {edge_id} connects an entity to itself",
                edge_id,
                confidence,
            )
        if not _edge_bbox_valid(edge.get("bbox_pt"), page):
            edge_fail(
                "graph-edge-bbox-invalid",
                f"Invalid edge path bbox in {edge_id}",
                edge_id,
                confidence,
            )
        if confidence + 1e-9 < EDGE_MIN_CONFIDENCE:
            edge_fail(
                "graph-edge-confidence-below-threshold",
                f"Low-confidence edge {edge_id}",
                edge_id,
                confidence,
            )
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
            edge_fail(
                "graph-edge-evidence-id-invalid",
                f"Edge {edge_id} cites invalid evidence IDs",
                edge_id,
                confidence,
            )
        if relation_type in GEOMETRY_REQUIRED_RELATIONS and not drawing_ids:
            edge_fail(
                "graph-edge-geometry-evidence-missing",
                (
                    f"Geometry-dependent edge {edge_id} has no local drawing "
                    "evidence; PDF links are not electrical conductors"
                ),
                edge_id,
                confidence,
            )

    if not resolution.get("all_reference_entities_accounted_for"):
        issues.append(_local_issue(
            issue_type="graph-reference-accounting-failed",
            message=(
                "One or more external references are neither exactly resolved "
                "nor explicitly preserved as unresolved"
            ),
            entity_ids=resolution.get("invalid_reference_entity_ids") or [],
            confidence=0.0,
        ))
    unresolved_ids = resolution.get("unresolved_reference_entity_ids") or []
    if unresolved_ids:
        issues.append(_local_issue(
            issue_type="graph-unresolved-references-preserved",
            message=(
                "Visible unresolved references were preserved without creating "
                "fabricated certified-registry mapping edges"
            ),
            entity_ids=unresolved_ids,
            confidence=verifier.get("confidence") or 0.0,
            severity="warning",
            source_stage="reference_resolution_adjudicator",
        ))

    if int(verifier.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-verifier-page-id-mismatch",
            message="Verifier returned a different page_id",
        ))

    verified_entity_ids = {
        _clean_text(value, 160)
        for value in (verifier.get("verified_entity_ids") or [])
        if _clean_text(value, 160)
    }
    verified_edge_ids = {
        _clean_text(value, 160)
        for value in (verifier.get("verified_edge_ids") or [])
        if _clean_text(value, 160)
    }
    preserved_unresolved = set(
        adjudication.get("preserved_unresolved_reference_ids") or []
    )
    recovered_entity_ids = set(
        recovery_audit.get("recovered_entity_ids") or []
    )
    recovered_edge_ids = set(
        recovery_audit.get("recovered_edge_ids") or []
    )
    final_entity_ids = set(occurrence_ids)
    final_edge_ids = set(edge_ids)
    unaccounted_entities = (
        final_entity_ids
        - verified_entity_ids
        - preserved_unresolved
        - recovered_entity_ids
    )
    if unaccounted_entities:
        issues.append(_local_issue(
            issue_type="graph-verifier-entity-accounting-mismatch",
            message=(
                "Final graph contains entities that were not independently "
                "verified or safely preserved as explicit references"
            ),
            entity_ids=sorted(unaccounted_entities),
            confidence=verifier.get("confidence") or 0.0,
        ))
    unaccounted_edges = final_edge_ids - verified_edge_ids - recovered_edge_ids
    if unaccounted_edges:
        issues.append(_local_issue(
            issue_type="graph-verifier-edge-accounting-mismatch",
            message="Final graph contains edges not independently verified",
            edge_ids=sorted(unaccounted_edges),
            confidence=verifier.get("confidence") or 0.0,
        ))

    verifier_issue_types = {
        _clean_text(item.get("issue_type"), 180)
        for item in (verifier.get("issues") or [])
        if isinstance(item, dict)
    }
    missing_entity_recovery_required = bool(
        "missing_visible_entity" in verifier_issue_types
    )
    entity_coverage_recovered = bool(
        recovery_audit.get("validated")
        and recovery_audit.get("all_visual_evidence_adjudicated")
        and (
            not missing_entity_recovery_required
            or int(recovery_audit.get("recovered_entity_count") or 0) > 0
        )
    )
    if not verifier.get("all_visible_entities_accounted_for"):
        if (
            entity_coverage_recovered
            and not entity_validation_failed
        ):
            issues.append(_local_issue(
                issue_type=(
                    "graph-verifier-visible-entity-flag-"
                    "superseded-post-recovery"
                ),
                message=(
                    "The verifier raw-candidate entity coverage flag was false, "
                    "but all missing visual entity evidence was recovered or "
                    "explicitly adjudicated and every final entity is valid"
                ),
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="verifier_evidence_recovery",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-all_visible_entities_accounted_for",
                message=(
                    "Verifier returned all_visible_entities_accounted_for=false"
                ),
                confidence=verifier.get("confidence") or 0.0,
            ))

    if not verifier.get("all_visible_connections_accounted_for"):
        if (
            recovery_audit.get("validated")
            and recovery_audit.get("all_visual_evidence_adjudicated")
            and not edge_validation_failed
        ):
            issues.append(_local_issue(
                issue_type=(
                    "graph-verifier-visible-connection-flag-"
                    "superseded-post-recovery"
                ),
                message=(
                    "The verifier raw-candidate connection coverage flag was "
                    "false, but every visual gap was recovered or explicitly "
                    "adjudicated and every final edge is valid"
                ),
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="verifier_evidence_recovery",
            ))
        else:
            issues.append(_local_issue(
                issue_type=(
                    "graph-verifier-all_visible_connections_accounted_for"
                ),
                message=(
                    "Verifier returned all_visible_connections_accounted_for=false"
                ),
                confidence=verifier.get("confidence") or 0.0,
            ))

    for flag in (
        "all_references_resolved_or_explicitly_unresolved",
        "duplicates_preserved",
    ):
        if not verifier.get(flag):
            issues.append(_local_issue(
                issue_type=f"graph-verifier-{flag}",
                message=f"Verifier returned {flag}=false",
                confidence=verifier.get("confidence") or 0.0,
            ))

    if not verifier.get("all_entity_text_visually_supported"):
        if not entity_validation_failed:
            issues.append(_local_issue(
                issue_type=(
                    "graph-verifier-entity-text-flag-superseded-post-adjudication"
                ),
                message=(
                    "The pre-adjudication verifier flag was false, but every "
                    "final entity has valid source text evidence"
                ),
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="post_verifier_adjudication",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-all_entity_text_visually_supported",
                message=(
                    "Verifier returned all_entity_text_visually_supported=false"
                ),
                confidence=verifier.get("confidence") or 0.0,
            ))

    if not verifier.get("all_connection_geometry_supported"):
        if not edge_validation_failed:
            issues.append(_local_issue(
                issue_type=(
                    "graph-verifier-geometry-flag-superseded-post-adjudication"
                ),
                message=(
                    "Rejected non-geometric edges were removed and every final "
                    "edge has valid local drawing evidence"
                ),
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="post_verifier_adjudication",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-all_connection_geometry_supported",
                message=(
                    "Verifier returned all_connection_geometry_supported=false"
                ),
                confidence=verifier.get("confidence") or 0.0,
            ))

    if _clamp_conf(verifier.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-verifier-confidence-below-threshold",
            message="Verifier confidence is below page threshold",
            confidence=verifier.get("confidence") or 0.0,
        ))

    if not entities:
        issues.append(_local_issue(
            issue_type="graph-no-entities",
            message="No graph entities remain after adjudication",
        ))
    if not edges:
        issues.append(_local_issue(
            issue_type="graph-no-edges",
            message="No graph edges remain after adjudication",
        ))

    blocking_before_verdict = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    if str(verifier.get("verdict") or "") != "pass":
        if not blocking_before_verdict:
            issues.append(_local_issue(
                issue_type="graph-verifier-verdict-superseded-post-adjudication",
                message=(
                    "The verifier review verdict applied to raw candidates; "
                    "the pruned and reference-accounted final graph passes all "
                    "deterministic checks"
                ),
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="post_verifier_adjudication",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-blocked-page",
                message="Independent verifier did not pass the graph page",
                confidence=verifier.get("confidence") or 0.0,
            ))

    blocking = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    return not blocking and bool(entities) and bool(edges), entities, edges, issues



def _patch_entity_result_normalized(raw: Any) -> dict:
    entity = dict(raw) if isinstance(raw, dict) else {}
    entity.setdefault("source_drawing_ids", [])
    for field in (
        "source_glyph_ids", "source_word_ids", "source_drawing_ids"
    ):
        entity[field] = sorted({
            int(value) for value in (entity.get(field) or [])
            if isinstance(value, int) or str(value).isdigit()
        })
    return entity


def _patch_edge_result_normalized(raw: Any) -> dict:
    edge = dict(raw) if isinstance(raw, dict) else {}
    for field in (
        "source_glyph_ids", "source_drawing_ids", "source_link_ids"
    ):
        edge[field] = sorted({
            int(value) for value in (edge.get(field) or [])
            if isinstance(value, int) or str(value).isdigit()
        })
    return edge


def _patch_bbox_signature(value: Any) -> tuple[float, ...]:
    values = list(value or [])
    if len(values) != 4:
        return ()
    try:
        return tuple(round(float(item), 3) for item in values)
    except Exception:
        return ()


def _patch_int_signature(value: Any) -> tuple[int, ...]:
    return tuple(sorted({
        int(item)
        for item in (value or [])
        if isinstance(item, int) or str(item).isdigit()
    }))


def _entity_strict_echo_signature(value: Any) -> tuple[Any, ...]:
    """Signature for recognizing a redundant copy of the raw entity.

    Confidence, notes and source_drawing_ids are intentionally excluded. The
    extractor V1 entity contract has no drawing-id field, while the canonical
    verifier entity contract does. All identity, literal source text, geometry
    and extractor-owned source evidence must otherwise be unchanged.
    """
    entity = _patch_entity_result_normalized(value)
    text_fields = (
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
    )
    return (
        *(_clean_text(entity.get(field), 4000) for field in text_fields),
        _patch_bbox_signature(entity.get("bbox_pt")),
        _patch_int_signature(entity.get("source_glyph_ids")),
        _patch_int_signature(entity.get("source_word_ids")),
    )


def _edge_strict_echo_signature(value: Any) -> tuple[Any, ...]:
    """Signature for recognizing a redundant copy of the raw edge."""
    edge = _patch_edge_result_normalized(value)
    return (
        _clean_text(edge.get("edge_id"), 160),
        _clean_text(edge.get("source_occurrence_id"), 160),
        _clean_text(edge.get("target_occurrence_id"), 160),
        _clean_text(edge.get("relation_type"), 120),
        bool(edge.get("is_directed")),
        _clean_text(edge.get("potential_original"), 2000),
        _clean_text(edge.get("wire_reference_original"), 2000),
        _patch_bbox_signature(edge.get("bbox_pt")),
        _patch_int_signature(edge.get("source_glyph_ids")),
        _patch_int_signature(edge.get("source_drawing_ids")),
        _patch_int_signature(edge.get("source_link_ids")),
    )


def _entity_keep_projection_compatible(source: dict, result: dict) -> bool:
    """Allow a verifier to echo the final KEEP projection safely.

    The identity, canonical type and region must remain the same. All other
    fields are still validated later against the complete source registries,
    so this does not bypass text, geometry or evidence ownership checks.
    """
    return bool(
        _clean_text(result.get("occurrence_id"), 160)
        == _clean_text(source.get("occurrence_id"), 160)
        and _clean_text(result.get("entity_type"), 120)
        == _clean_text(source.get("entity_type"), 120)
        and _clean_text(result.get("region_id"), 120)
        == _clean_text(source.get("region_id"), 120)
    )


def _edge_keep_projection_compatible(source: dict, result: dict) -> bool:
    """Allow a verifier to echo a KEEP edge without silently rewiring it."""
    return bool(
        _clean_text(result.get("edge_id"), 160)
        == _clean_text(source.get("edge_id"), 160)
        and _clean_text(result.get("source_occurrence_id"), 160)
        == _clean_text(source.get("source_occurrence_id"), 160)
        and _clean_text(result.get("target_occurrence_id"), 160)
        == _clean_text(source.get("target_occurrence_id"), 160)
        and _clean_text(result.get("relation_type"), 120)
        == _clean_text(source.get("relation_type"), 120)
        and bool(result.get("is_directed"))
        == bool(source.get("is_directed"))
    )


def _normalize_entity_patch_results(
    *,
    action: str,
    source_id: str,
    source: Optional[dict],
    results: list[dict],
) -> tuple[list[dict], str, Optional[str]]:
    """Normalize schema-valid verifier result echoes without weakening safety.

    Strict JSON Schema requires ``result_entities`` on every operation. Models
    may therefore echo the source object for KEEP/REMOVE, or include that echo
    alongside REPLACE/SPLIT results. The operation action remains authoritative:
    redundant echoes are accepted only when their source identity is provable.
    """
    normalized = [_patch_entity_result_normalized(item) for item in results]
    if action == "ADD_ENTITY":
        if source_id or len(normalized) != 1:
            return [], "invalid_add_contract", "ADD_ENTITY requires one result and no source ID"
        return normalized, "direct_add", None

    if source is None:
        return [], "invalid_source", "Entity operation source does not exist"
    normalized_source = _patch_entity_result_normalized(source)

    if action == "KEEP_ENTITY":
        if not normalized:
            return [normalized_source], "implicit_raw_keep", None
        if len(normalized) == 1 and _entity_keep_projection_compatible(
            normalized_source, normalized[0]
        ):
            return normalized, "echoed_keep_projection", None
        return [], "invalid_keep_contract", (
            "KEEP_ENTITY accepts no result or one same-ID/type/region final projection"
        )

    if action == "REMOVE_ENTITY":
        if not normalized:
            return [], "direct_remove", None
        if (
            len(normalized) == 1
            and _clean_text(normalized[0].get("occurrence_id"), 160)
            == source_id
        ):
            return [], "echoed_remove_projection_discarded", None
        return [], "invalid_remove_contract", (
            "REMOVE_ENTITY accepts no result or one redundant same-ID source echo"
        )

    effective = list(normalized)
    strict_source_signature = _entity_strict_echo_signature(normalized_source)
    strict_echo_indexes = [
        index
        for index, item in enumerate(effective)
        if _entity_strict_echo_signature(item) == strict_source_signature
    ]
    compatibility_mode = "direct_results"
    if len(effective) > 1 and len(strict_echo_indexes) == 1:
        effective.pop(strict_echo_indexes[0])
        compatibility_mode = "redundant_source_echo_discarded"

    if action == "REPLACE_ENTITY":
        if len(effective) != 1:
            return [], "invalid_replace_contract", (
                "REPLACE_ENTITY requires exactly one effective final entity"
            )
        return effective, compatibility_mode, None
    if action == "SPLIT_ENTITY":
        if not (2 <= len(effective) <= 12):
            return [], "invalid_split_contract", (
                "SPLIT_ENTITY requires two to twelve effective final entities"
            )
        return effective, compatibility_mode, None
    return [], "invalid_action", "Unsupported entity patch action"


def _normalize_edge_patch_results(
    *,
    action: str,
    source_id: str,
    source: Optional[dict],
    results: list[dict],
) -> tuple[list[dict], str, Optional[str]]:
    """Normalize redundant edge result echoes while preserving topology rules."""
    normalized = [_patch_edge_result_normalized(item) for item in results]
    if action == "ADD_EDGE":
        if source_id or len(normalized) != 1:
            return [], "invalid_add_contract", "ADD_EDGE requires one result and no source ID"
        return normalized, "direct_add", None

    if source is None:
        return [], "invalid_source", "Edge operation source does not exist"
    normalized_source = _patch_edge_result_normalized(source)

    if action == "KEEP_EDGE":
        if not normalized:
            return [normalized_source], "implicit_raw_keep", None
        if len(normalized) == 1 and _edge_keep_projection_compatible(
            normalized_source, normalized[0]
        ):
            return normalized, "echoed_keep_projection", None
        return [], "invalid_keep_contract", (
            "KEEP_EDGE cannot change endpoints, relation, direction or edge ID"
        )

    if action == "REMOVE_EDGE":
        if not normalized:
            return [], "direct_remove", None
        if (
            len(normalized) == 1
            and _clean_text(normalized[0].get("edge_id"), 160) == source_id
        ):
            return [], "echoed_remove_projection_discarded", None
        return [], "invalid_remove_contract", (
            "REMOVE_EDGE accepts no result or one redundant same-ID source echo"
        )

    effective = list(normalized)
    strict_source_signature = _edge_strict_echo_signature(normalized_source)
    strict_echo_indexes = [
        index
        for index, item in enumerate(effective)
        if _edge_strict_echo_signature(item) == strict_source_signature
    ]
    compatibility_mode = "direct_results"
    if len(effective) > 1 and len(strict_echo_indexes) == 1:
        effective.pop(strict_echo_indexes[0])
        compatibility_mode = "redundant_source_echo_discarded"

    if action == "REWIRE_EDGE":
        if not (1 <= len(effective) <= 24):
            return [], "invalid_rewire_contract", (
                "REWIRE_EDGE requires one to twenty-four effective final edges"
            )
        return effective, compatibility_mode, None
    return [], "invalid_action", "Unsupported edge patch action"


def _apply_graph_patch_plan(
    *,
    page: dict,
    extraction: dict,
    verifier: dict,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """Apply the verifier patch plan entirely in memory and fail closed.

    Every raw entity and edge must be decided exactly once. Replacement/split
    lineage is explicit, and any edge that would retain a deleted endpoint must
    be rewired or removed. This function does not trust verifier global flags;
    it validates operation cardinality, coverage and final endpoint integrity.
    """
    issues: list[dict] = []
    raw_entities = [
        dict(item) for item in (extraction.get("entities") or [])
        if isinstance(item, dict)
    ]
    raw_edges = [
        dict(item) for item in (extraction.get("edges") or [])
        if isinstance(item, dict)
    ]
    raw_entity_by_id = {
        _clean_text(item.get("occurrence_id"), 160): item
        for item in raw_entities
        if _clean_text(item.get("occurrence_id"), 160)
    }
    raw_edge_by_id = {
        _clean_text(item.get("edge_id"), 160): item
        for item in raw_edges
        if _clean_text(item.get("edge_id"), 160)
    }
    raw_entity_ids = set(raw_entity_by_id)
    raw_edge_ids = set(raw_edge_by_id)
    unresolved_items = [
        _clean_text(value, 4000)
        for value in (extraction.get("unresolved_visual_evidence") or [])
    ]

    if verifier.get("patch_plan_version") != GRAPH_PATCH_PLAN_VERSION:
        issues.append(_local_issue(
            issue_type="graph-patch-plan-version-mismatch",
            message="Verifier returned an unsupported graph patch-plan version",
            confidence=verifier.get("confidence") or 0.0,
        ))

    operation_ids: set[str] = set()
    applied_operation_ids: set[str] = set()
    raw_entity_claims: dict[str, list[str]] = {}
    raw_edge_claims: dict[str, list[str]] = {}
    entity_lineage: dict[str, list[str]] = {}
    edge_lineage: dict[str, list[str]] = {}
    final_entities: list[dict] = []
    final_edges: list[dict] = []
    entity_operation_audit: list[dict] = []
    edge_operation_audit: list[dict] = []

    def register_operation_id(operation_id: str, *, kind: str) -> bool:
        if not operation_id or operation_id in operation_ids:
            issues.append(_local_issue(
                issue_type="graph-patch-operation-id-invalid",
                message=(
                    f"Missing or duplicate {kind} patch operation_id"
                ),
                confidence=verifier.get("confidence") or 0.0,
            ))
            return False
        operation_ids.add(operation_id)
        return True

    for raw in verifier.get("entity_operations") or []:
        op = raw if isinstance(raw, dict) else {}
        operation_id = _clean_text(op.get("operation_id"), 160)
        action = _clean_text(op.get("action"), 80)
        source_id = _clean_text(op.get("source_entity_id"), 160)
        confidence = _clamp_conf(op.get("confidence"))
        results = [
            _patch_entity_result_normalized(item)
            for item in (op.get("result_entities") or [])
            if isinstance(item, dict)
        ]
        evidence_indexes = sorted({
            int(value) for value in (op.get("evidence_indexes") or [])
            if isinstance(value, int) or str(value).isdigit()
        })
        problem = not register_operation_id(operation_id, kind="entity")
        if action not in ENTITY_PATCH_ACTIONS:
            issues.append(_local_issue(
                issue_type="graph-entity-patch-action-invalid",
                message="Verifier returned an invalid entity patch action",
                entity_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True
        if confidence + 1e-9 < ENTITY_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-entity-patch-confidence-below-threshold",
                message="Entity patch operation confidence is below threshold",
                entity_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True
        if any(index < 0 or index >= len(unresolved_items) for index in evidence_indexes):
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-index-invalid",
                message="Entity patch operation cites an invalid evidence index",
                entity_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True

        source_entity = raw_entity_by_id.get(source_id)
        if action != "ADD_ENTITY":
            if source_id not in raw_entity_ids:
                issues.append(_local_issue(
                    issue_type="graph-entity-patch-source-invalid",
                    message="Entity patch operation references an unknown raw entity",
                    entity_ids=[source_id] if source_id else [],
                    confidence=confidence,
                ))
                problem = True
            else:
                raw_entity_claims.setdefault(source_id, []).append(operation_id)

        produced, compatibility_mode, contract_error = (
            _normalize_entity_patch_results(
                action=action,
                source_id=source_id,
                source=source_entity,
                results=results,
            )
        )
        if contract_error:
            problem = True

        if problem:
            issues.append(_local_issue(
                issue_type="graph-entity-patch-cardinality-invalid",
                message=(
                    "Entity patch operation violates its source/result contract"
                    + (f": {contract_error}" if contract_error else "")
                ),
                entity_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            entity_operation_audit.append({
                "operation_id": operation_id,
                "action": action,
                "source_entity_id": source_id,
                "input_result_entity_ids": [
                    _clean_text(item.get("occurrence_id"), 160)
                    for item in results
                ],
                "result_entity_ids": [
                    _clean_text(item.get("occurrence_id"), 160)
                    for item in produced
                ],
                "input_result_count": len(results),
                "effective_result_count": len(produced),
                "compatibility_mode": compatibility_mode,
                "contract_error": contract_error or "",
                "validated": False,
            })
            continue

        produced_ids = [
            _clean_text(item.get("occurrence_id"), 160) for item in produced
        ]
        if any(not value for value in produced_ids) or len(produced_ids) != len(set(produced_ids)):
            issues.append(_local_issue(
                issue_type="graph-entity-patch-result-id-invalid",
                message="Entity patch operation produced missing or duplicate final IDs",
                entity_ids=[value for value in produced_ids if value],
                confidence=confidence,
            ))
            entity_operation_audit.append({
                "operation_id": operation_id,
                "action": action,
                "source_entity_id": source_id,
                "result_entity_ids": produced_ids,
                "validated": False,
            })
            continue

        for entity in produced:
            entity.setdefault("patch_provenance", {})
            entity["patch_provenance"] = {
                "version": GRAPH_PATCH_APPLICATION_VERSION,
                "operation_id": operation_id,
                "action": action,
                "source_entity_id": source_id,
                "evidence_indexes": evidence_indexes,
                "confidence": confidence,
                "reason": _clean_text(op.get("reason"), 1600),
            }
            final_entities.append(entity)
        if source_id:
            entity_lineage[source_id] = produced_ids
        applied_operation_ids.add(operation_id)
        entity_operation_audit.append({
            "operation_id": operation_id,
            "action": action,
            "source_entity_id": source_id,
            "input_result_entity_ids": [
                _clean_text(item.get("occurrence_id"), 160)
                for item in results
            ],
            "result_entity_ids": produced_ids,
            "input_result_count": len(results),
            "effective_result_count": len(produced),
            "compatibility_mode": compatibility_mode,
            "evidence_indexes": evidence_indexes,
            "confidence": confidence,
            "validated": True,
        })

    missing_entity_decisions = sorted(raw_entity_ids - set(raw_entity_claims))
    duplicate_entity_decisions = sorted(
        source_id for source_id, claims in raw_entity_claims.items()
        if len(claims) != 1
    )
    if missing_entity_decisions or duplicate_entity_decisions:
        issues.append(_local_issue(
            issue_type="graph-entity-patch-coverage-failed",
            message="Every raw entity must be decided exactly once",
            entity_ids=sorted(set(missing_entity_decisions + duplicate_entity_decisions)),
            confidence=verifier.get("confidence") or 0.0,
        ))

    final_entity_ids_list = [
        _clean_text(item.get("occurrence_id"), 160) for item in final_entities
    ]
    duplicate_final_entity_ids = sorted({
        value for value in final_entity_ids_list
        if value and final_entity_ids_list.count(value) > 1
    })
    if duplicate_final_entity_ids:
        issues.append(_local_issue(
            issue_type="graph-final-entity-id-collision",
            message="Multiple patch operations produced the same final entity ID",
            entity_ids=duplicate_final_entity_ids,
            confidence=verifier.get("confidence") or 0.0,
        ))
    final_entity_ids = set(final_entity_ids_list)

    for raw in verifier.get("edge_operations") or []:
        op = raw if isinstance(raw, dict) else {}
        operation_id = _clean_text(op.get("operation_id"), 160)
        action = _clean_text(op.get("action"), 80)
        source_id = _clean_text(op.get("source_edge_id"), 160)
        confidence = _clamp_conf(op.get("confidence"))
        results = [
            _patch_edge_result_normalized(item)
            for item in (op.get("result_edges") or [])
            if isinstance(item, dict)
        ]
        evidence_indexes = sorted({
            int(value) for value in (op.get("evidence_indexes") or [])
            if isinstance(value, int) or str(value).isdigit()
        })
        problem = not register_operation_id(operation_id, kind="edge")
        if action not in EDGE_PATCH_ACTIONS:
            issues.append(_local_issue(
                issue_type="graph-edge-patch-action-invalid",
                message="Verifier returned an invalid edge patch action",
                edge_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True
        if confidence + 1e-9 < EDGE_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-edge-patch-confidence-below-threshold",
                message="Edge patch operation confidence is below threshold",
                edge_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True
        if any(index < 0 or index >= len(unresolved_items) for index in evidence_indexes):
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-index-invalid",
                message="Edge patch operation cites an invalid evidence index",
                edge_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            problem = True

        source_edge = raw_edge_by_id.get(source_id)
        if action != "ADD_EDGE":
            if source_id not in raw_edge_ids:
                issues.append(_local_issue(
                    issue_type="graph-edge-patch-source-invalid",
                    message="Edge patch operation references an unknown raw edge",
                    edge_ids=[source_id] if source_id else [],
                    confidence=confidence,
                ))
                problem = True
            else:
                raw_edge_claims.setdefault(source_id, []).append(operation_id)

        produced, compatibility_mode, contract_error = (
            _normalize_edge_patch_results(
                action=action,
                source_id=source_id,
                source=source_edge,
                results=results,
            )
        )
        if contract_error:
            problem = True

        if problem:
            issues.append(_local_issue(
                issue_type="graph-edge-patch-cardinality-invalid",
                message=(
                    "Edge patch operation violates its source/result contract"
                    + (f": {contract_error}" if contract_error else "")
                ),
                edge_ids=[source_id] if source_id else [],
                confidence=confidence,
            ))
            edge_operation_audit.append({
                "operation_id": operation_id,
                "action": action,
                "source_edge_id": source_id,
                "input_result_edge_ids": [
                    _clean_text(item.get("edge_id"), 160) for item in results
                ],
                "result_edge_ids": [
                    _clean_text(item.get("edge_id"), 160) for item in produced
                ],
                "input_result_count": len(results),
                "effective_result_count": len(produced),
                "compatibility_mode": compatibility_mode,
                "contract_error": contract_error or "",
                "validated": False,
            })
            continue
        produced_ids = [
            _clean_text(item.get("edge_id"), 160) for item in produced
        ]
        if any(not value for value in produced_ids) or len(produced_ids) != len(set(produced_ids)):
            issues.append(_local_issue(
                issue_type="graph-edge-patch-result-id-invalid",
                message="Edge patch operation produced missing or duplicate final IDs",
                edge_ids=[value for value in produced_ids if value],
                confidence=confidence,
            ))
            edge_operation_audit.append({
                "operation_id": operation_id,
                "action": action,
                "source_edge_id": source_id,
                "result_edge_ids": produced_ids,
                "validated": False,
            })
            continue
        for edge in produced:
            edge.setdefault("patch_provenance", {})
            edge["patch_provenance"] = {
                "version": GRAPH_PATCH_APPLICATION_VERSION,
                "operation_id": operation_id,
                "action": action,
                "source_edge_id": source_id,
                "evidence_indexes": evidence_indexes,
                "confidence": confidence,
                "reason": _clean_text(op.get("reason"), 1600),
            }
            final_edges.append(edge)
        if source_id:
            edge_lineage[source_id] = produced_ids
        applied_operation_ids.add(operation_id)
        edge_operation_audit.append({
            "operation_id": operation_id,
            "action": action,
            "source_edge_id": source_id,
            "input_result_edge_ids": [
                _clean_text(item.get("edge_id"), 160) for item in results
            ],
            "result_edge_ids": produced_ids,
            "input_result_count": len(results),
            "effective_result_count": len(produced),
            "compatibility_mode": compatibility_mode,
            "evidence_indexes": evidence_indexes,
            "confidence": confidence,
            "validated": True,
        })

    missing_edge_decisions = sorted(raw_edge_ids - set(raw_edge_claims))
    duplicate_edge_decisions = sorted(
        source_id for source_id, claims in raw_edge_claims.items()
        if len(claims) != 1
    )
    if missing_edge_decisions or duplicate_edge_decisions:
        issues.append(_local_issue(
            issue_type="graph-edge-patch-coverage-failed",
            message="Every raw edge must be decided exactly once",
            edge_ids=sorted(set(missing_edge_decisions + duplicate_edge_decisions)),
            confidence=verifier.get("confidence") or 0.0,
        ))

    final_edge_ids_list = [
        _clean_text(item.get("edge_id"), 160) for item in final_edges
    ]
    duplicate_final_edge_ids = sorted({
        value for value in final_edge_ids_list
        if value and final_edge_ids_list.count(value) > 1
    })
    if duplicate_final_edge_ids:
        issues.append(_local_issue(
            issue_type="graph-final-edge-id-collision",
            message="Multiple patch operations produced the same final edge ID",
            edge_ids=duplicate_final_edge_ids,
            confidence=verifier.get("confidence") or 0.0,
        ))
    final_edge_ids = set(final_edge_ids_list)

    invalid_endpoint_edges = []
    for edge in final_edges:
        source_id = _clean_text(edge.get("source_occurrence_id"), 160)
        target_id = _clean_text(edge.get("target_occurrence_id"), 160)
        if source_id not in final_entity_ids or target_id not in final_entity_ids:
            invalid_endpoint_edges.append(_clean_text(edge.get("edge_id"), 160))
    if invalid_endpoint_edges:
        issues.append(_local_issue(
            issue_type="graph-patch-final-edge-endpoint-missing",
            message="Final patched edges reference deleted or absent final entities",
            edge_ids=sorted(invalid_endpoint_edges),
            confidence=verifier.get("confidence") or 0.0,
        ))

    adjudications = [
        item for item in (verifier.get("evidence_adjudications") or [])
        if isinstance(item, dict)
    ]
    seen_evidence_indexes: set[int] = set()
    normalized_adjudications: list[dict] = []
    for item in adjudications:
        try:
            evidence_index = int(item.get("evidence_index"))
        except Exception:
            evidence_index = -1
        confidence = _clamp_conf(item.get("confidence"))
        status = _clean_text(item.get("status"), 120)
        raw_context_entity_ids = {
            _clean_text(value, 160)
            for value in (item.get("raw_context_entity_ids") or [])
            if _clean_text(value, 160)
        }
        raw_context_edge_ids = {
            _clean_text(value, 160)
            for value in (item.get("raw_context_edge_ids") or [])
            if _clean_text(value, 160)
        }
        cited_final_entity_ids = {
            _clean_text(value, 160)
            for value in (item.get("final_entity_ids") or [])
            if _clean_text(value, 160)
        }
        cited_final_edge_ids = {
            _clean_text(value, 160)
            for value in (item.get("final_edge_ids") or [])
            if _clean_text(value, 160)
        }
        related_operation_ids = {
            _clean_text(value, 160)
            for value in (item.get("related_operation_ids") or [])
            if _clean_text(value, 160)
        }
        row_problem = False
        if (
            evidence_index < 0
            or evidence_index >= len(unresolved_items)
            or evidence_index in seen_evidence_indexes
        ):
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-adjudication-index-invalid",
                message="Evidence adjudication index is missing, duplicate or out of range",
                confidence=confidence,
            ))
            row_problem = True
        else:
            seen_evidence_indexes.add(evidence_index)
        if status not in PATCH_EVIDENCE_STATUSES:
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-status-invalid",
                message="Evidence adjudication returned an invalid status",
                confidence=confidence,
            ))
            row_problem = True
        if confidence + 1e-9 < ENTITY_MIN_CONFIDENCE:
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-confidence-below-threshold",
                message="Evidence adjudication confidence is below threshold",
                confidence=confidence,
            ))
            row_problem = True
        if raw_context_entity_ids - raw_entity_ids or raw_context_edge_ids - raw_edge_ids:
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-raw-context-invalid",
                message="Evidence adjudication cites raw context IDs absent from extraction",
                entity_ids=sorted(raw_context_entity_ids - raw_entity_ids),
                edge_ids=sorted(raw_context_edge_ids - raw_edge_ids),
                confidence=confidence,
            ))
            row_problem = True
        if cited_final_entity_ids - final_entity_ids or cited_final_edge_ids - final_edge_ids:
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-final-context-invalid",
                message="Evidence adjudication cites IDs absent from the final graph",
                entity_ids=sorted(cited_final_entity_ids - final_entity_ids),
                edge_ids=sorted(cited_final_edge_ids - final_edge_ids),
                confidence=confidence,
            ))
            row_problem = True
        if related_operation_ids - applied_operation_ids:
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-operation-link-invalid",
                message="Evidence adjudication cites unapplied patch operations",
                confidence=confidence,
            ))
            row_problem = True
        if status == "accounted_by_final_graph" and not (
            cited_final_entity_ids or cited_final_edge_ids
        ):
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-final-link-missing",
                message="Final-graph evidence adjudication must cite a final entity or edge",
                confidence=confidence,
            ))
            row_problem = True
        if status == "still_unresolved":
            issues.append(_local_issue(
                issue_type="graph-patch-evidence-still-unresolved",
                message="A visible evidence item remains unresolved after patch planning",
                entity_ids=sorted(cited_final_entity_ids),
                edge_ids=sorted(cited_final_edge_ids),
                confidence=confidence,
            ))
            row_problem = True
        canonical_text = (
            unresolved_items[evidence_index]
            if 0 <= evidence_index < len(unresolved_items)
            else _clean_text(item.get("evidence_text_original"), 4000)
        )
        normalized_adjudications.append({
            "evidence_index": evidence_index,
            "evidence_text_original": canonical_text,
            "status": status,
            "raw_context_entity_ids": sorted(raw_context_entity_ids),
            "raw_context_edge_ids": sorted(raw_context_edge_ids),
            "final_entity_ids": sorted(cited_final_entity_ids),
            "final_edge_ids": sorted(cited_final_edge_ids),
            "related_operation_ids": sorted(related_operation_ids),
            "confidence": confidence,
            "reason": _clean_text(item.get("reason"), 1600),
            "validated": not row_problem,
        })

    expected_evidence_indexes = set(range(len(unresolved_items)))
    if seen_evidence_indexes != expected_evidence_indexes:
        issues.append(_local_issue(
            issue_type="graph-patch-evidence-accounting-mismatch",
            message="Every extractor unresolved-evidence item must be adjudicated exactly once",
            confidence=verifier.get("confidence") or 0.0,
        ))

    blocking = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    compatibility_modes: dict[str, int] = {}
    for row in entity_operation_audit + edge_operation_audit:
        mode = _clean_text(row.get("compatibility_mode"), 160)
        if mode:
            compatibility_modes[mode] = compatibility_modes.get(mode, 0) + 1

    audit = {
        "version": GRAPH_PATCH_APPLICATION_VERSION,
        "patch_plan_version": verifier.get("patch_plan_version") or "",
        "patch_result_compatibility": {
            "version": PATCH_RESULT_COMPATIBILITY_VERSION,
            "normalized_operation_count": sum(
                count
                for mode, count in compatibility_modes.items()
                if mode not in {"direct_add", "direct_remove", "direct_results"}
            ),
            "mode_counts": dict(sorted(compatibility_modes.items())),
        },
        "raw_entity_count": len(raw_entities),
        "raw_edge_count": len(raw_edges),
        "final_entity_count": len(final_entities),
        "final_edge_count": len(final_edges),
        "entity_operations": entity_operation_audit,
        "edge_operations": edge_operation_audit,
        "applied_operation_ids": sorted(applied_operation_ids),
        "entity_lineage": entity_lineage,
        "edge_lineage": edge_lineage,
        "removed_entity_ids": sorted(
            source_id for source_id, targets in entity_lineage.items()
            if not targets
        ),
        "removed_edge_ids": sorted(
            source_id for source_id, targets in edge_lineage.items()
            if not targets
        ),
        "added_entity_ids": sorted(
            _clean_text(item.get("occurrence_id"), 160)
            for item in final_entities
            if (item.get("patch_provenance") or {}).get("action") == "ADD_ENTITY"
        ),
        "added_edge_ids": sorted(
            _clean_text(item.get("edge_id"), 160)
            for item in final_edges
            if (item.get("patch_provenance") or {}).get("action") == "ADD_EDGE"
        ),
        "evidence_adjudications": normalized_adjudications,
        "all_raw_entities_decided": not (
            missing_entity_decisions or duplicate_entity_decisions
        ),
        "all_raw_edges_decided": not (
            missing_edge_decisions or duplicate_edge_decisions
        ),
        "all_unresolved_evidence_adjudicated": (
            seen_evidence_indexes == expected_evidence_indexes
        ),
        "validated": not blocking,
    }
    return final_entities, final_edges, audit, issues


def _normalize_verifier_patch_issue(
    raw: Any,
    *,
    patch_audit: dict,
) -> dict:
    issue = _normalize_issue(
        raw,
        default_type="graph-verifier-patch-issue",
        source_stage="verifier",
    )
    source = raw if isinstance(raw, dict) else {}
    status = _clean_text(source.get("resolution_status"), 80) or "open"
    related_operation_ids = {
        _clean_text(value, 160)
        for value in (source.get("related_operation_ids") or [])
        if _clean_text(value, 160)
    }
    applied = set(patch_audit.get("applied_operation_ids") or [])
    resolution_validated = bool(
        status == "resolved_by_patch_plan"
        and related_operation_ids
        and related_operation_ids.issubset(applied)
    )
    if status == "informational":
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_patch_audit"
    elif resolution_validated:
        issue["severity"] = "info"
        issue["source_stage"] = "verifier_patch_plan_resolved"
    elif status == "resolved_by_patch_plan":
        issue["severity"] = "high"
        issue["source_stage"] = "deterministic_patch_validator"
        issue["message"] = (
            issue["message"]
            + " [Invalid resolution claim: one or more related patch operations were not applied.]"
        )[:1600]
    issue["patch_resolution"] = {
        "status": status,
        "related_operation_ids": sorted(related_operation_ids),
        "validated": resolution_validated,
    }
    return issue


def _normalize_raw_issue_after_patch(
    raw: Any,
    *,
    default_type: str,
    source_stage: str,
    patch_audit: dict,
) -> dict:
    issue = _normalize_issue(
        raw,
        default_type=default_type,
        source_stage=source_stage,
    )
    entity_ids = set(issue.get("entity_ids") or [])
    edge_ids = set(issue.get("edge_ids") or [])
    entity_lineage = patch_audit.get("entity_lineage") or {}
    edge_lineage = patch_audit.get("edge_lineage") or {}
    transformed_entities = {
        source_id for source_id, targets in entity_lineage.items()
        if targets != [source_id]
    }
    transformed_edges = {
        source_id for source_id, targets in edge_lineage.items()
        if targets != [source_id]
    }
    if (
        issue.get("severity") in {"high", "critical"}
        and (entity_ids or edge_ids)
        and entity_ids.issubset(transformed_entities)
        and edge_ids.issubset(transformed_edges)
    ):
        issue["severity"] = "info"
        issue["source_stage"] = "raw_issue_superseded_by_patch"
        issue["message"] = (
            issue["message"]
            + " [Superseded by validated entity/edge patch operations.]"
        )[:1600]
    return issue


def _validate_patched_graph(
    *,
    page: dict,
    detector: dict,
    extraction: dict,
    verifier: dict,
    resolution: dict,
    entities: list[dict],
    edges: list[dict],
    patch_audit: dict,
    patch_issues: list[dict],
    glyphs: list[dict],
    words: list[dict],
    drawings: list[dict],
    links: list[dict],
) -> tuple[bool, list[dict], list[dict], list[dict]]:
    """Validate only the final patched projection, never the pre-patch graph."""
    issues = list(patch_issues)

    geometry_audit, geometry_issues = _reconcile_graph_geometry_from_evidence(
        page=page,
        entities=entities,
        edges=edges,
        glyphs=glyphs,
        words=words,
        drawings=drawings,
    )
    issues.extend(geometry_issues)
    final_assertions = verifier.get("final_assertions") or {}
    region_bbox_audit, region_bbox_issues = _adjudicate_detector_region_bboxes(
        page=page,
        detector=detector,
        entities=entities,
        edges=edges,
        final_assertions=final_assertions,
    )
    issues.extend(region_bbox_issues)

    for raw in detector.get("issues") or []:
        issue = _normalize_raw_issue_after_patch(
            raw,
            default_type="graph-detector-issue",
            source_stage="detector",
            patch_audit=patch_audit,
        )
        if issue.get("severity") in {"high", "critical"} and region_bbox_audit.get("validated"):
            issue["severity"] = "warning"
            issue["source_stage"] = "detector_preliminary_audit"
        issues.append(issue)
    for raw in extraction.get("issues") or []:
        issues.append(_normalize_raw_issue_after_patch(
            raw,
            default_type="graph-extractor-issue",
            source_stage="extractor",
            patch_audit=patch_audit,
        ))
    for raw in verifier.get("issues") or []:
        issues.append(_normalize_verifier_patch_issue(
            raw,
            patch_audit=patch_audit,
        ))

    if int(detector.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-detector-page-id-mismatch",
            message="Detector returned a different page_id",
        ))
    if int(extraction.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-extractor-page-id-mismatch",
            message="Extractor returned a different page_id",
        ))
    if int(verifier.get("page_id") or 0) != int(page["id"]):
        issues.append(_local_issue(
            issue_type="graph-verifier-page-id-mismatch",
            message="Verifier returned a different page_id",
        ))

    preliminary_coverage_false = bool(
        not detector.get("all_visible_circuit_regions_accounted_for")
        or detector.get("uncovered_visual_regions")
    )
    if preliminary_coverage_false:
        if (
            patch_audit.get("validated")
            and patch_audit.get("all_unresolved_evidence_adjudicated")
            and final_assertions.get("all_visible_entities_accounted_for")
            and final_assertions.get("all_visible_connections_accounted_for")
        ):
            issues.append(_local_issue(
                issue_type="graph-detector-coverage-superseded-by-final-patch",
                message="Preliminary detector coverage was superseded by the complete final patch projection",
                confidence=detector.get("confidence") or 0.0,
                severity="info",
                source_stage="final_patch_validation",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-detector-region-coverage-failed",
                message="Detector coverage remains unresolved by the final patch plan",
                confidence=detector.get("confidence") or 0.0,
            ))

    if _clamp_conf(detector.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-detector-confidence-low-audit",
            message="Detector confidence was low but is not authoritative after final evidence validation",
            confidence=detector.get("confidence") or 0.0,
            severity="warning",
            source_stage="detector_preliminary_audit",
        ))
    if _clamp_conf(extraction.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
        issues.append(_local_issue(
            issue_type="graph-extractor-confidence-low-audit",
            message="Extractor confidence was low; final patched objects remain subject to deterministic validation",
            confidence=extraction.get("confidence") or 0.0,
            severity="warning",
            source_stage="extractor_preliminary_audit",
        ))

    if extraction.get("unresolved_visual_evidence"):
        if patch_audit.get("all_unresolved_evidence_adjudicated") and patch_audit.get("validated"):
            issues.append(_local_issue(
                issue_type="graph-extractor-evidence-fully-adjudicated",
                message="Every extractor unresolved-evidence item was adjudicated by the atomic patch plan",
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="final_patch_validation",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-unresolved-visual-evidence",
                message="Extractor visual evidence remains unresolved after patch application",
                confidence=extraction.get("confidence") or 0.0,
            ))

    valid_glyph_ids = {int(item["glyph_id"]) for item in glyphs}
    valid_word_ids = {int(item["word_id"]) for item in words}
    valid_drawing_ids = {int(item["drawing_id"]) for item in drawings}
    valid_link_ids = {int(item["id"]) for item in links}
    region_ids = {
        _clean_text(item.get("region_id"), 160)
        for item in (detector.get("regions") or [])
        if isinstance(item, dict) and _clean_text(item.get("region_id"), 160)
    }

    occurrence_ids: list[str] = []
    entity_by_id: dict[str, dict] = {}
    entity_validation_failed: set[str] = set()
    resolution_by_id = {
        str(item.get("occurrence_id") or ""): item
        for item in (resolution.get("entity_resolutions") or [])
    }
    for entity in entities:
        occurrence_id = _clean_text(entity.get("occurrence_id"), 160)
        entity_type = _clean_text(entity.get("entity_type"), 120)
        confidence = _clamp_conf(entity.get("confidence"))
        def fail(issue_type: str, message: str) -> None:
            entity_validation_failed.add(occurrence_id)
            issues.append(_local_issue(
                issue_type=issue_type,
                message=message,
                entity_ids=[occurrence_id] if occurrence_id else [],
                confidence=confidence,
            ))
        if not occurrence_id or occurrence_id in occurrence_ids:
            fail("graph-entity-id-invalid", "Missing or duplicate final entity occurrence_id")
        occurrence_ids.append(occurrence_id)
        entity_by_id[occurrence_id] = entity
        if entity_type not in ENTITY_TYPES:
            fail("graph-entity-type-invalid", "Final entity type is outside the canonical schema")
        if _clean_text(entity.get("region_id"), 160) not in region_ids:
            fail("graph-entity-region-invalid", "Final entity references an unknown detector region")
        if not _bbox_valid(entity.get("bbox_pt"), page):
            fail("graph-entity-bbox-invalid", "Final entity bbox is invalid after evidence reconciliation")
        if confidence + 1e-9 < ENTITY_MIN_CONFIDENCE:
            fail("graph-entity-confidence-below-threshold", "Final entity confidence is below threshold")
        glyph_ids = {
            int(value) for value in (entity.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        word_ids = {
            int(value) for value in (entity.get("source_word_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        drawing_ids = {
            int(value) for value in (entity.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if glyph_ids - valid_glyph_ids or word_ids - valid_word_ids or drawing_ids - valid_drawing_ids:
            fail("graph-entity-evidence-id-invalid", "Final entity cites source evidence IDs absent from registries")
        visible_fields = [
            field for field in SOURCE_VISIBLE_ENTITY_TEXT_FIELDS
            if _clean_text(entity.get(field), 1000)
        ]
        if visible_fields and not (glyph_ids or word_ids):
            fail("graph-entity-text-evidence-missing", "Final entity claims literal text without glyph/word evidence")
        if not visible_fields and not drawing_ids:
            fail("graph-entity-graphic-evidence-missing", "Graphic-only final entity lacks drawing evidence")
        if entity_type in COMPONENT_ENTITY_TYPES and not _clean_text(entity.get("tag_original"), 500):
            fail("graph-component-tag-missing", "Component-like final entity has no visible tag; it must be retyped or replaced")
        entity["source_text_evidence_policy"] = {
            "version": "graph-source-visible-text-fields-v3",
            "visible_source_text_fields": visible_fields,
            "semantic_annotation_fields": [
                field for field in SEMANTIC_ENTITY_ANNOTATION_FIELDS
                if _clean_text(entity.get(field), 1000)
            ],
            "validated": bool((not visible_fields and drawing_ids) or (visible_fields and (glyph_ids or word_ids))),
        }
        if entity_type in REFERENCE_ENTITY_TYPES:
            resolved = resolution_by_id.get(occurrence_id) or {}
            if not resolved.get("accounted"):
                fail("graph-reference-entity-not-accounted", "Final reference is neither exactly resolved nor explicitly unresolved")
            entity["reference_resolution"] = {
                key: resolved.get(key) for key in (
                    "resolved", "explicitly_unresolved", "accounted",
                    "resolution_status", "resolution_source", "reason",
                )
            }

    edge_ids: list[str] = []
    edge_validation_failed: set[str] = set()
    for edge in edges:
        edge_id = _clean_text(edge.get("edge_id"), 160)
        relation_type = _clean_text(edge.get("relation_type"), 120)
        source_id = _clean_text(edge.get("source_occurrence_id"), 160)
        target_id = _clean_text(edge.get("target_occurrence_id"), 160)
        confidence = _clamp_conf(edge.get("confidence"))
        def fail(issue_type: str, message: str) -> None:
            edge_validation_failed.add(edge_id)
            issues.append(_local_issue(
                issue_type=issue_type,
                message=message,
                edge_ids=[edge_id] if edge_id else [],
                confidence=confidence,
            ))
        if not edge_id or edge_id in edge_ids:
            fail("graph-edge-id-invalid", "Missing or duplicate final edge_id")
        edge_ids.append(edge_id)
        if relation_type not in RELATION_TYPES:
            fail("graph-edge-relation-invalid", "Final edge relation type is outside the canonical schema")
        if source_id not in entity_by_id or target_id not in entity_by_id:
            fail("graph-edge-endpoint-missing", "Final edge references a missing final entity")
        if source_id and source_id == target_id:
            fail("graph-edge-self-reference", "Final edge connects an entity to itself")
        if not _edge_bbox_valid(edge.get("bbox_pt"), page):
            fail("graph-edge-bbox-invalid", "Final edge bbox is invalid after evidence reconciliation")
        if confidence + 1e-9 < EDGE_MIN_CONFIDENCE:
            fail("graph-edge-confidence-below-threshold", "Final edge confidence is below threshold")
        drawing_ids = {
            int(value) for value in (edge.get("source_drawing_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        glyph_ids = {
            int(value) for value in (edge.get("source_glyph_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        link_ids = {
            int(value) for value in (edge.get("source_link_ids") or [])
            if isinstance(value, int) or str(value).isdigit()
        }
        if drawing_ids - valid_drawing_ids or glyph_ids - valid_glyph_ids or link_ids - valid_link_ids:
            fail("graph-edge-evidence-id-invalid", "Final edge cites source evidence IDs absent from registries")
        if relation_type in GEOMETRY_REQUIRED_RELATIONS and not drawing_ids:
            fail("graph-edge-geometry-evidence-missing", "Geometry-dependent final edge lacks local drawing evidence; PDF links are not conductors")

    if not resolution.get("all_reference_entities_accounted_for"):
        issues.append(_local_issue(
            issue_type="graph-reference-accounting-failed",
            message="Final external references are neither exactly resolved nor explicitly unresolved",
            entity_ids=resolution.get("invalid_reference_entity_ids") or [],
        ))
    unresolved_ids = resolution.get("unresolved_reference_entity_ids") or []
    if unresolved_ids:
        issues.append(_local_issue(
            issue_type="graph-unresolved-references-preserved",
            message="Visible unresolved references are preserved without fabricated registry mappings",
            entity_ids=unresolved_ids,
            confidence=verifier.get("confidence") or 0.0,
            severity="warning",
            source_stage="reference_resolution_adjudicator",
        ))

    deterministic_blockers = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    assertions_false = sorted(
        key for key, value in final_assertions.items() if value is not True
    )
    if assertions_false:
        if not deterministic_blockers:
            issues.append(_local_issue(
                issue_type="graph-verifier-final-assertions-superseded",
                message="Conservative verifier assertions were superseded by complete deterministic final-projection validation",
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="final_patch_validation",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-final-assertions-failed",
                message="Verifier final assertions remain false and deterministic blockers remain",
                confidence=verifier.get("confidence") or 0.0,
            ))

    if _clamp_conf(verifier.get("confidence")) + 1e-9 < PAGE_PASS_MIN_CONFIDENCE:
        if deterministic_blockers:
            issues.append(_local_issue(
                issue_type="graph-verifier-confidence-below-threshold",
                message="Verifier confidence is below page threshold and final blockers remain",
                confidence=verifier.get("confidence") or 0.0,
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-confidence-low-audit",
                message="Global verifier confidence is conservative; every applied operation and final object passes its own threshold",
                confidence=verifier.get("confidence") or 0.0,
                severity="warning",
                source_stage="final_patch_validation",
            ))

    if not entities:
        issues.append(_local_issue(
            issue_type="graph-no-entities",
            message="No graph entities remain after patch application",
        ))
    if not edges:
        issues.append(_local_issue(
            issue_type="graph-no-edges",
            message="No graph edges remain after patch application",
        ))

    blockers_before_verdict = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    if verifier.get("verdict") != "apply_patch":
        if not blockers_before_verdict:
            issues.append(_local_issue(
                issue_type="graph-verifier-verdict-superseded-post-patch",
                message="Verifier review verdict applied before deterministic patch validation; the final projection is safe",
                confidence=verifier.get("confidence") or 0.0,
                severity="info",
                source_stage="final_patch_validation",
            ))
        else:
            issues.append(_local_issue(
                issue_type="graph-verifier-blocked-page",
                message="Verifier requested review and final deterministic blockers remain",
                confidence=verifier.get("confidence") or 0.0,
            ))

    patch_audit["source_evidence_geometry_reconciliation"] = geometry_audit
    patch_audit["region_bbox_adjudication"] = region_bbox_audit
    patch_audit["final_validation_version"] = GRAPH_FINAL_VALIDATION_VERSION
    patch_audit["final_entity_count"] = len(entities)
    patch_audit["final_edge_count"] = len(edges)
    patch_audit["final_assertions"] = final_assertions
    patch_audit["final_reference_resolution"] = {
        "unresolved_reference_entity_ids": resolution.get("unresolved_reference_entity_ids") or [],
        "resolution_status_counts": resolution.get("resolution_status_counts") or {},
        "match_counts": resolution.get("match_counts") or {},
        "all_reference_entities_accounted_for": resolution.get("all_reference_entities_accounted_for"),
    }
    blocking = [
        issue for issue in issues
        if issue.get("severity") in {"high", "critical"}
    ]
    patch_audit["validated"] = not blocking
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
                "source_drawing_ids": entity.get("source_drawing_ids") or [],
                "recovery_evidence": entity.get("recovery_evidence") or {},
                "bbox_reconciliation": entity.get("bbox_reconciliation") or {},
                "source_text_evidence_policy": entity.get(
                    "source_text_evidence_policy"
                ) or {},
                "patch_provenance": entity.get("patch_provenance") or {},
                "evidence_notes": entity.get("evidence_notes") or "",
                "reference_resolution": {
                    key: (resolution_by_id.get(occurrence_id) or {}).get(key)
                    for key in (
                        "resolved",
                        "explicitly_unresolved",
                        "accounted",
                        "resolution_status",
                        "resolution_source",
                        "reason",
                    )
                },
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
                "recovery_evidence": edge.get("recovery_evidence") or {},
                "bbox_reconciliation": edge.get("bbox_reconciliation") or {},
                "patch_provenance": edge.get("patch_provenance") or {},
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
                    "adjudication": issue.get("adjudication") or {},
                    "patch_resolution": issue.get("patch_resolution") or {},
                    "review_cause_family": _review_cause_family(
                        issue.get("issue_type"), issue.get("source_stage")
                    ),
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
    post_verifier_adjudication: Optional[dict] = None,
    reference_resolution: Optional[dict] = None,
    issue_type_counts: Optional[dict] = None,
    blocking_issue_type_counts: Optional[dict] = None,
    review_cause_family_counts: Optional[dict] = None,
    review_signature: str = "",
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
            adjudication = post_verifier_adjudication or {}
            resolution = reference_resolution or {}
            page_results[str(page["pdf_page_number"])] = {
                "page_passed": bool(page_passed),
                "published_entities": int(published_entities),
                "published_edges": int(published_edges),
                "blocking_issue_count": int(blocking_count),
                "raw_extracted_entities": int(
                    adjudication.get("raw_entity_count") or 0
                ),
                "raw_extracted_edges": int(
                    adjudication.get("raw_edge_count") or 0
                ),
                "final_visual_entities": int(
                    adjudication.get("final_entity_count") or 0
                ),
                "final_visual_edges": int(
                    adjudication.get("final_edge_count") or 0
                ),
                "removed_entity_ids": adjudication.get(
                    "removed_entity_ids"
                ) or [],
                "removed_edge_ids": adjudication.get(
                    "removed_edge_ids"
                ) or [],
                "patch_plan_version": adjudication.get(
                    "patch_plan_version"
                ) or GRAPH_PATCH_PLAN_VERSION,
                "patch_application_version": adjudication.get(
                    "version"
                ) or GRAPH_PATCH_APPLICATION_VERSION,
                "final_validation_version": adjudication.get(
                    "final_validation_version"
                ) or GRAPH_FINAL_VALIDATION_VERSION,
                "patch_plan_validated": bool(adjudication.get("validated")),
                "patch_result_compatibility": adjudication.get(
                    "patch_result_compatibility"
                ) or {},
                "applied_patch_operation_count": len(
                    adjudication.get("applied_operation_ids") or []
                ),
                "entity_patch_operation_count": len(
                    adjudication.get("entity_operations") or []
                ),
                "edge_patch_operation_count": len(
                    adjudication.get("edge_operations") or []
                ),
                "preserved_unresolved_reference_ids": resolution.get(
                    "unresolved_reference_entity_ids"
                ) or [],
                "recovered_entity_ids": adjudication.get(
                    "added_entity_ids"
                ) or [],
                "recovered_edge_ids": adjudication.get(
                    "added_edge_ids"
                ) or [],
                "evidence_recovery_validated": bool(
                    adjudication.get("validated")
                ),
                "visual_evidence_adjudication_count": len(
                    adjudication.get("evidence_adjudications") or []
                ),
                "adjudicated_region_bbox_count": int(
                    (
                        adjudication.get("region_bbox_adjudication")
                        or {}
                    ).get("adjudicated_region_count")
                    or 0
                ),
                "source_evidence_geometry_validated": bool(
                    (
                        adjudication.get(
                            "source_evidence_geometry_reconciliation"
                        )
                        or {}
                    ).get("validated")
                ),
                "reconciled_entity_bbox_count": int(
                    (
                        adjudication.get(
                            "source_evidence_geometry_reconciliation"
                        )
                        or {}
                    ).get("reconciled_entity_count")
                    or 0
                ),
                "reconciled_edge_bbox_count": int(
                    (
                        adjudication.get(
                            "source_evidence_geometry_reconciliation"
                        )
                        or {}
                    ).get("reconciled_edge_count")
                    or 0
                ),
                "unresolved_reference_entity_ids": resolution.get(
                    "unresolved_reference_entity_ids"
                ) or [],
                "resolution_status_counts": resolution.get(
                    "resolution_status_counts"
                ) or {},
                "reference_match_counts": resolution.get(
                    "match_counts"
                ) or {},
                "issue_type_counts": issue_type_counts or {},
                "blocking_issue_type_counts": (
                    blocking_issue_type_counts or {}
                ),
                "review_cause_family_counts": (
                    review_cause_family_counts or {}
                ),
                "review_signature_version": REVIEW_GROUPING_VERSION,
                "review_signature": review_signature or "",
                "pipeline_marker": PIPELINE_MARKER,
                "materializer_version": MATERIALIZER_VERSION,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            passed_pages = sum(
                1
                for value in page_results.values()
                if isinstance(value, dict) and value.get("page_passed")
            )
            review_pages = sum(
                1
                for value in page_results.values()
                if isinstance(value, dict) and not value.get("page_passed")
            )
            total_pages = int(context["all_graph_pages_total"])
            if passed_pages == total_pages and total_pages > 0:
                graph_status = "graph_ready"
            elif review_pages > 0:
                graph_status = "review_required"
            elif passed_pages > 0:
                graph_status = "partial"
            else:
                graph_status = "not_started"

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

            version_status = (
                "review_required"
                if graph_status == "review_required"
                else "queued"
            )
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



def _issue_type_counts(
    issues: list[dict],
    *,
    blocking_only: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues or []:
        if blocking_only and issue.get("severity") not in {"high", "critical"}:
            continue
        issue_type = _clean_text(issue.get("issue_type"), 180) or "unknown"
        counts[issue_type] = counts.get(issue_type, 0) + 1
    return dict(sorted(counts.items()))


def _blocking_issue_summary(issues: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for issue in issues or []:
        if issue.get("severity") not in {"high", "critical"}:
            continue
        issue_type = _clean_text(issue.get("issue_type"), 180) or "unknown"
        source_stage = _clean_text(issue.get("source_stage"), 120) or "unknown"
        key = (issue_type, source_stage)
        row = grouped.setdefault(key, {
            "issue_type": issue_type,
            "source_stage": source_stage,
            "severity": issue.get("severity") or "high",
            "count": 0,
            "entity_ids": set(),
            "edge_ids": set(),
            "sample_message": _clean_text(issue.get("message"), 500),
        })
        row["count"] += 1
        row["entity_ids"].update(issue.get("entity_ids") or [])
        row["edge_ids"].update(issue.get("edge_ids") or [])
        if issue.get("severity") == "critical":
            row["severity"] = "critical"
    output: list[dict] = []
    for key in sorted(grouped):
        row = grouped[key]
        output.append({
            **{k: v for k, v in row.items() if k not in {"entity_ids", "edge_ids"}},
            "entity_ids": sorted(row["entity_ids"])[:60],
            "edge_ids": sorted(row["edge_ids"])[:60],
        })
    return output



def _review_cause_family(issue_type: Any, source_stage: Any = "") -> str:
    value = _clean_text(issue_type, 180).casefold()
    stage = _clean_text(source_stage, 120).casefold()
    if any(token in value for token in (
        "rewire", "edge-endpoint", "edge-geometry", "connection_geometry"
    )):
        return "edge_rewire_or_geometry"
    if any(token in value for token in (
        "split", "merged", "duplicate", "under_materialized",
        "false_candidate", "entity-bbox", "component-tag"
    )):
        return "entity_identity_or_replacement"
    if any(token in value for token in (
        "reference", "registry", "explicitly-unresolved"
    )):
        return "reference_resolution"
    if any(token in value for token in (
        "evidence", "glyph", "drawing"
    )):
        return "source_evidence_accounting"
    if any(token in value for token in (
        "patch", "schema", "type-invalid", "operation"
    )):
        return "canonical_patch_contract"
    if "confidence" in value or stage.endswith("preliminary_audit"):
        return "preliminary_confidence"
    if (
        value.startswith("graph-verifier-all_")
        or any(token in value for token in (
            "verdict", "publish", "blocked-page", "final-assertions"
        ))
    ):
        return "publish_gate_cascade"
    return "other"


def _review_cause_family_counts(issues: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues or []:
        if issue.get("severity") not in {"high", "critical"}:
            continue
        family = _review_cause_family(
            issue.get("issue_type"), issue.get("source_stage")
        )
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _review_signature(issues: list[dict]) -> str:
    families = _review_cause_family_counts(issues)
    if not families:
        return "pass"
    # Gate-cascade issues do not create a distinct technical family when a
    # concrete upstream cause is present.
    causal = {
        key: value for key, value in families.items()
        if key != "publish_gate_cascade"
    } or families
    raw = json.dumps(
        sorted(causal),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


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

        # Detector and extractor requests remain byte-for-byte compatible with
        # the certified V1 stages. The V3 verifier receives complete PDF-point
        # evidence and returns one canonical atomic patch plan.
        verifier_resolution = _resolve_references_for_verifier_v1(
            extraction,
            registry,
        )
        verifier_request = {
            "page_sha256": page.get("page_sha256"),
            "pdf_page_number": page["pdf_page_number"],
            "detector_fingerprint": detector_fp,
            "extractor_fingerprint": extractor_fp,
            "detector": detector,
            "extraction": extraction,
            "deterministic_reference_resolution": verifier_resolution,
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
            "patch_plan_version": GRAPH_PATCH_PLAN_VERSION,
            "entity_patch_actions": sorted(ENTITY_PATCH_ACTIONS),
            "edge_patch_actions": sorted(EDGE_PATCH_ACTIONS),
            "canonical_entity_types": sorted(ENTITY_TYPES),
            "canonical_relation_types": sorted(RELATION_TYPES),
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
                resolution=verifier_resolution,
                words=words,
                glyphs=glyphs,
                drawings=drawings,
                links=links,
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
                "patch_plan_version": GRAPH_PATCH_PLAN_VERSION,
            },
        )
        _add_usage(
            usage_totals,
            "verifier",
            verifier_usage,
            verifier_reused,
        )

        # Graph V3 applies the complete verifier patch plan in memory, then
        # resolves references and validates only the final projection. Raw
        # candidates and pre-patch global flags are audit evidence, never the
        # publication authority.
        entities, edges, patch_audit, patch_issues = _apply_graph_patch_plan(
            page=page,
            extraction=extraction,
            verifier=verifier,
        )
        patched_extraction = dict(extraction)
        patched_extraction["entities"] = entities
        patched_extraction["edges"] = edges
        patched_extraction["graph_patch_plan"] = patch_audit
        resolution = _resolve_references(patched_extraction, registry)

        page_passed, entities, edges, issues = _validate_patched_graph(
            page=page,
            detector=detector,
            extraction=extraction,
            verifier=verifier,
            resolution=resolution,
            entities=entities,
            edges=edges,
            patch_audit=patch_audit,
            patch_issues=patch_issues,
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
        issue_type_counts = _issue_type_counts(issues)
        blocking_issue_type_counts = _issue_type_counts(
            issues,
            blocking_only=True,
        )
        review_signature = _review_signature(issues)
        review_cause_family_counts = _review_cause_family_counts(issues)
        blocking_issue_summary = _blocking_issue_summary(issues)
        state = _db_update_version_state(
            context=context,
            page=page,
            page_passed=page_passed,
            published_entities=publication["published_page_entities"],
            published_edges=publication["published_page_edges"],
            blocking_count=blocking,
            post_verifier_adjudication=patch_audit,
            reference_resolution=resolution,
            issue_type_counts=issue_type_counts,
            blocking_issue_type_counts=blocking_issue_type_counts,
            review_cause_family_counts=review_cause_family_counts,
            review_signature=review_signature,
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
            "graph_pipeline_marker": PIPELINE_MARKER,
            "graph_materializer_version": MATERIALIZER_VERSION,
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
            "issue_type_counts": issue_type_counts,
            "blocking_issue_type_counts": blocking_issue_type_counts,
            "blocking_issue_summary": blocking_issue_summary,
            "review_signature_version": REVIEW_GROUPING_VERSION,
            "review_signature": review_signature,
            "review_cause_family_counts": review_cause_family_counts,
            "entity_type_counts": entity_type_counts,
            "relation_type_counts": relation_type_counts,
            "reference_resolution": {
                "all_reference_entities_resolved": resolution.get(
                    "all_reference_entities_resolved"
                ),
                "all_reference_entities_accounted_for": resolution.get(
                    "all_reference_entities_accounted_for"
                ),
                "all_reference_entities_resolved_or_explicitly_unresolved": (
                    resolution.get(
                        "all_reference_entities_resolved_or_explicitly_unresolved"
                    )
                ),
                "unresolved_reference_entity_ids": resolution.get(
                    "unresolved_reference_entity_ids"
                ) or [],
                "ambiguous_reference_entity_ids": resolution.get(
                    "ambiguous_reference_entity_ids"
                ) or [],
                "invalid_reference_entity_ids": resolution.get(
                    "invalid_reference_entity_ids"
                ) or [],
                "resolution_status_counts": resolution.get(
                    "resolution_status_counts"
                ) or {},
                "match_counts": resolution.get("match_counts") or {},
            },
            "graph_patch_plan": patch_audit,
            # Compatibility aliases for existing audit consumers. They now
            # point to the final V3 patch/geometry audits rather than the old
            # keep/reject/recovery protocol.
            "post_verifier_adjudication": patch_audit,
            "region_bbox_adjudication": patch_audit.get(
                "region_bbox_adjudication"
            ) or {},
            "source_evidence_geometry_reconciliation": patch_audit.get(
                "source_evidence_geometry_reconciliation"
            ) or {},
            "raw_extracted_entity_count": len(
                extraction.get("entities") or []
            ),
            "raw_extracted_edge_count": len(
                extraction.get("edges") or []
            ),
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
