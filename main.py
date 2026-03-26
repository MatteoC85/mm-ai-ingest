import os
import re
import math
import json
import unicodedata
from typing import Optional, List, Any, Union

import requests
import psycopg2
import fitz  # PyMuPDF
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from google.cloud import tasks_v2
from urllib.parse import urlparse, unquote

app = FastAPI()

AI_INTERNAL_SECRET = (os.environ.get("AI_INTERNAL_SECRET") or "").strip()
FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

# DB
DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

# Indicizzabilità
MIN_TEXT_CHARS = int(os.environ.get("MM_MIN_TEXT_CHARS", "2000"))
MIN_TEXT_CHARS_SHORT = int(os.environ.get("MM_MIN_TEXT_CHARS_SHORT", "800"))
MIN_PAGE_CHARS = int(os.environ.get("MM_MIN_PAGE_CHARS", "30"))
MIN_PAGES_WITH_TEXT_ABS = int(os.environ.get("MM_MIN_PAGES_WITH_TEXT_ABS", "2"))
MIN_PAGES_WITH_TEXT_PCT = float(os.environ.get("MM_MIN_PAGES_WITH_TEXT_PCT", "0.20"))

# Chunking
CHUNK_TARGET_CHARS = int(os.environ.get("MM_CHUNK_TARGET_CHARS", "3800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("MM_CHUNK_OVERLAP_CHARS", "800"))
CHUNK_MIN_CHARS = int(os.environ.get("MM_CHUNK_MIN_CHARS", "200"))

# OpenAI
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_EMBED_MODEL = (os.environ.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
OPENAI_EMBED_URL = (os.environ.get("OPENAI_EMBED_URL") or "https://api.openai.com/v1/embeddings").strip()

# OpenAI Chat
OPENAI_CHAT_MODEL = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4.1-mini").strip()
OPENAI_CHAT_URL = (os.environ.get("OPENAI_CHAT_URL") or "https://api.openai.com/v1/chat/completions").strip()

# OpenAI Citation Reranker (on-demand)
OPENAI_RERANK_MODEL = (os.environ.get("OPENAI_RERANK_MODEL") or "gpt-4.1-nano").strip()
RERANK_MAX_CANDIDATES = int(os.environ.get("MM_RERANK_MAX_CANDIDATES", "18"))
RERANK_SNIPPET_CHARS = int(os.environ.get("MM_RERANK_SNIPPET_CHARS", "320"))
RERANK_TIMEOUT = int(os.environ.get("MM_RERANK_TIMEOUT_SECONDS", "30"))
RERANK_ENABLED = (os.environ.get("MM_RERANK_ENABLED") or "0").strip() == "1"
RERANK_MIN_SIM_MAX = float(os.environ.get("MM_RERANK_MIN_SIM_MAX", "0.38"))
RERANK_MAX_SIM_MAX = float(os.environ.get("MM_RERANK_MAX_SIM_MAX", "0.72"))
RERANK_MAX_SPREAD = float(os.environ.get("MM_RERANK_MAX_SPREAD", "0.10"))
RERANK_MIN_CANDIDATES = int(os.environ.get("MM_RERANK_MIN_CANDIDATES", "4"))

ASK_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SIM_THRESHOLD", "0.35"))
ASK_MAX_TOP_K = int(os.environ.get("MM_ASK_MAX_TOP_K", "8"))
ASK_SNIPPET_CHARS = int(os.environ.get("MM_ASK_SNIPPET_CHARS", "700"))
ASK_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_MAX_CONTEXT_CHARS", "9000"))

# -----------------------------
# Entity fallback (URL / email / phone)
# -----------------------------
URL_REGEX = re.compile(r"(https?://[^\s\)\]\}]+|www\.[^\s\)\]\}]+)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

# -----------------------------
# Code token fallback (SENTINEL / part-number / codici)
# -----------------------------
CODE_TOKEN_REGEX = re.compile(r"\b[A-Z0-9_]{6,}\b")

URL_HINTS = ["sito", "website", "web site", "url", "link", "pagina", "dominio", "www"]
EMAIL_HINTS = ["email", "e-mail", "mail", "posta"]
PHONE_HINTS = ["telefono", "cell", "cellulare", "tel", "contatto", "chiamare", "numero"]

STRUCTURED_SOURCE_TYPES = {
    "procedure",
    "step",
    "ps",
    "md_photo",
    "md_video",
}

class IngestRequest(BaseModel):
    file_url: str
    company_id: str
    machine_id: str
    bubble_document_id: str
    plan_embed_chars_limit_total: Optional[int] = None
    plan_index_storage_limit_bytes: Optional[int] = None
    embed_chars_used_total: Optional[int] = None
    index_storage_used_total: Optional[int] = None
    doc_prev_embed_chars: Optional[int] = None
    doc_prev_index_storage_bytes: Optional[int] = None


class IndexDocumentRequest(BaseModel):
    company_id: str
    machine_id: str
    bubble_document_id: str
    trace_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    company_id: str
    bubble_document_id: Optional[str] = None
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    top_k: int = 5
    debug: Optional[bool] = False


class RootCauseRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    top_k: int = 8
    max_causes: int = 3
    debug: Optional[bool] = False

class DraftPSOptions(BaseModel):
    top_k: int = 8
    max_causes: int = 3


class DraftPSRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    language: Optional[str] = "it"
    options: Optional[DraftPSOptions] = None
    debug: Optional[bool] = False

class DeleteDocumentRequest(BaseModel):
    company_id: str
    bubble_document_id: str

class StructuredSourceIngestRequest(BaseModel):
    company_id: str
    machine_id: str
    source_type: str
    source_id: str
    source_url: Optional[str] = None

    title: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = None
    procedure_type: Optional[str] = None
    step_number: Optional[int] = None
    category: Optional[str] = None
    solution: Optional[str] = None
    notes: Optional[str] = None

    plan_embed_chars_limit_total: Optional[int] = None
    plan_index_storage_limit_bytes: Optional[int] = None
    embed_chars_used_total: Optional[int] = None
    index_storage_used_total: Optional[int] = None
    doc_prev_embed_chars: Optional[int] = None
    doc_prev_index_storage_bytes: Optional[int] = None

def _normalize_document_ids(value: Optional[Union[List[str], str]]) -> Optional[list[str]]:
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]

    if isinstance(value, list):
        value = [str(x).strip() for x in value if str(x).strip()]
        return value or None

    return None

def _resolve_query_scope(
    company_id: str,
    machine_id: Optional[str],
    bubble_document_id: Optional[str] = None,
    document_ids: Optional[Union[List[str], str]] = None,
) -> dict:
    company_id = (company_id or "").strip()
    if not company_id:
        raise HTTPException(status_code=400, detail="Missing company_id")

    machine_id = (machine_id or "").strip()
    if not machine_id:
        raise HTTPException(status_code=400, detail="Missing machine_id")

    bubble_document_id = (bubble_document_id or "").strip() or None
    doc_ids = _normalize_document_ids(document_ids)

    return {
        "company_id": company_id,
        "machine_id": machine_id,
        "bubble_document_id": bubble_document_id,
        "document_ids": doc_ids,
    }

def _fetch_dense_chunk_candidates(
    *,
    company_id: str,
    machine_id: str,
    q_vec_lit: str,
    candidate_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
) -> tuple[Optional[int], list[tuple]]:
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
                           embedding
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = ANY(%s)
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, doc_ids, q_vec_lit, candidate_k),
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
                          AND (machine_id=%s OR machine_id IS NULL);
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
                           embedding
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, bdid, machine_id, q_vec_lit, candidate_k),
                )

            else:
                if debug:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM public.document_chunks
                        WHERE company_id=%s
                          AND embedding IS NOT NULL
                          AND (machine_id=%s OR machine_id IS NULL);
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
                           embedding
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, machine_id, q_vec_lit, candidate_k),
                )

            raw_rows = cur.fetchall()
            return chunks_matching_filter, raw_rows
    finally:
        conn.close()

def _raw_rows_to_dense_candidates(
    raw_rows: list[tuple],
    *,
    query_used: Optional[str] = None,
) -> list[dict]:
    candidates: list[dict] = []

    for (bdid, chunk_index, page_from, page_to, snippet, chunk_full, similarity, embedding) in raw_rows:
        if embedding is None:
            emb_list = None
        elif isinstance(embedding, list):
            emb_list = embedding
        else:
            s = str(embedding).strip()
            s = s.strip("[]")
            emb_list = [float(x) for x in s.split(",") if x.strip()]

        item = {
            "citation_id": f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}",
            "bubble_document_id": str(bdid),
            "page_from": int(page_from),
            "page_to": int(page_to),
            "snippet": (snippet or "").strip(),
            "chunk_full": (chunk_full or "").strip(),
            "similarity": float(similarity),
            "embedding_list": emb_list or [],
        }

        if query_used is not None:
            item["query_used"] = query_used

        candidates.append(item)

    return candidates

def _db_conn():
    if not (DB_HOST and DB_USER and DB_PASSWORD):
        raise HTTPException(status_code=500, detail="DB env missing")
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _get_table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s;
        """,
        (table_name,),
    )
    return {r[0] for r in cur.fetchall()}


def _db_upsert_document_file(company_id: str, bubble_document_id: str, file_url: str) -> None:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    file_url = (file_url or "").strip()
    if not (company_id and bubble_document_id and file_url):
        return

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.document_files(company_id, bubble_document_id, file_url, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (company_id, bubble_document_id)
                DO UPDATE SET file_url = EXCLUDED.file_url, updated_at = NOW();
                """,
                (company_id, bubble_document_id, file_url),
            )
        conn.commit()
    finally:
        conn.close()


def _db_upsert_cleaning_meta(
    company_id: str,
    bubble_document_id: str,
    header_norm: set[str],
    footer_norm: set[str],
) -> None:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    if not (company_id and bubble_document_id):
        return

    header_list = sorted([x for x in (header_norm or set()) if x])
    footer_list = sorted([x for x in (footer_norm or set()) if x])

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.document_cleaning_meta(company_id, bubble_document_id, header_norm, footer_norm, updated_at)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (company_id, bubble_document_id)
                DO UPDATE SET header_norm = EXCLUDED.header_norm,
                              footer_norm = EXCLUDED.footer_norm,
                              updated_at = NOW();
                """,
                (company_id, bubble_document_id, json.dumps(header_list), json.dumps(footer_list)),
            )
        conn.commit()
    finally:
        conn.close()


def _db_get_cleaning_meta(company_id: str, bubble_document_id: str) -> tuple[set[str], set[str]]:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    if not (company_id and bubble_document_id):
        return set(), set()

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT header_norm, footer_norm
                FROM public.document_cleaning_meta
                WHERE company_id=%s AND bubble_document_id=%s
                LIMIT 1;
                """,
                (company_id, bubble_document_id),
            )
            row = cur.fetchone()
            if not row:
                return set(), set()

            h, f = row
            if isinstance(h, str):
                h = json.loads(h or "[]")
            if isinstance(f, str):
                f = json.loads(f or "[]")

            header_set = {str(x) for x in (h or []) if x}
            footer_set = {str(x) for x in (f or []) if x}
            return header_set, footer_set
    finally:
        conn.close()

def _db_get_index_usage(company_id: str, bubble_document_id: Optional[str] = None) -> dict:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip() if bubble_document_id else None

    if not company_id:
        return {
            "text_chars": 0,
            "chunk_count": 0,
            "est_storage_bytes": 0,
        }

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if bubble_document_id:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(text_chars), 0)
                    FROM public.document_pages
                    WHERE company_id=%s
                      AND bubble_document_id=%s;
                    """,
                    (company_id, bubble_document_id),
                )
                text_chars = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.document_chunks
                    WHERE company_id=%s
                      AND bubble_document_id=%s;
                    """,
                    (company_id, bubble_document_id),
                )
                chunk_count = int(cur.fetchone()[0] or 0)
            else:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(text_chars), 0)
                    FROM public.document_pages
                    WHERE company_id=%s;
                    """,
                    (company_id,),
                )
                text_chars = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.document_chunks
                    WHERE company_id=%s;
                    """,
                    (company_id,),
                )
                chunk_count = int(cur.fetchone()[0] or 0)

        est_storage_bytes = int(text_chars * 3 + chunk_count * 2000)

        return {
            "text_chars": text_chars,
            "chunk_count": chunk_count,
            "est_storage_bytes": est_storage_bytes,
        }
    finally:
        conn.close()

def _build_rg_links(company_id: str, citations: list[dict]) -> list[dict]:
    if not citations:
        return []

    doc_ids = sorted(
        {
            str(c.get("bubble_document_id") or "").strip()
            for c in citations
            if c.get("bubble_document_id")
        }
    )
    if not doc_ids:
        return []

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bubble_document_id, file_url
                FROM public.document_files
                WHERE company_id = %s
                  AND bubble_document_id = ANY(%s);
                """,
                (company_id, doc_ids),
            )
            rows = cur.fetchall()
            file_map = {str(bdid): (url or "").strip() for (bdid, url) in rows if bdid and url}
    finally:
        conn.close()

    out: list[dict] = []
    for c in citations:
        bdid = str(c.get("bubble_document_id") or "").strip()
        if not bdid:
            continue

        file_url = file_map.get(bdid)
        if not file_url:
            continue

        base = file_url.split("#", 1)[0]
        page_from = int(c.get("page_from") or 1)
        if page_from < 1:
            page_from = 1

        is_structured = _is_structured_source_key(bdid)
        final_url = file_url if is_structured else f"{base}#page={page_from}"

        out.append(
            {
                "citation_id": c.get("citation_id"),
                "bubble_document_id": bdid,
                "page_from": int(c.get("page_from") or page_from),
                "page_to": int(c.get("page_to") or page_from),
                "url": final_url,
            }
        )

    return out

def _normalize_structured_source_type(source_type: str) -> str:
    s = re.sub(r"[\s\-]+", "_", str(source_type or "").strip().lower())

    aliases = {
        "procedure": "procedure",
        "step": "step",
        "ps": "ps",
        "problemsolution": "ps",
        "problem_solution": "ps",
        "problem_solution_item": "ps",
        "md_photo": "md_photo",
        "machine_detail_photo": "md_photo",
        "photo_machine_detail": "md_photo",
        "md_video": "md_video",
        "machine_detail_video": "md_video",
        "video_machine_detail": "md_video",
    }

    s = aliases.get(s, s)
    if s not in STRUCTURED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported source_type: {source_type}")

    return s


def _build_structured_source_key(source_type: str, source_id: str) -> str:
    st = _normalize_structured_source_type(source_type)
    sid = str(source_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing source_id")
    return f"{st}:{sid}"

def _is_structured_source_key(value: str) -> bool:
    v = str(value or "").strip().lower()
    if ":" not in v:
        return False

    prefix = v.split(":", 1)[0].strip()
    return prefix in STRUCTURED_SOURCE_TYPES

def _clean_structured_text_value(value: Any) -> str:
    s = _normalize_unicode_advanced(str(value or ""))
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for ln in s.split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)

    return "\n".join(lines).strip()


def _append_structured_field(lines: list[str], label: str, value: Any) -> None:
    v = _clean_structured_text_value(value)
    if v:
        lines.append(f"{label}: {v}")


def _compose_structured_source_text(payload: StructuredSourceIngestRequest) -> str:
    st = _normalize_structured_source_type(payload.source_type)
    lines: list[str] = []

    if st == "procedure":
        lines.append("SOURCE_TYPE: procedure")
        _append_structured_field(lines, "TITLE", payload.title)
        _append_structured_field(lines, "PROCEDURE_TYPE", payload.procedure_type)
        _append_structured_field(lines, "SHORT_DESCRIPTION", payload.short_description)

    elif st == "step":
        lines.append("SOURCE_TYPE: step")
        if payload.step_number is not None:
            lines.append(f"STEP_NUMBER: {int(payload.step_number)}")
        _append_structured_field(lines, "TITLE", payload.title)
        _append_structured_field(lines, "DESCRIPTION", payload.description)

    elif st == "ps":
        lines.append("SOURCE_TYPE: problem_solution")
        _append_structured_field(lines, "TITLE", payload.title)
        _append_structured_field(lines, "CATEGORY", payload.category)
        _append_structured_field(lines, "DESCRIPTION", payload.description)
        _append_structured_field(lines, "SOLUTION", payload.solution)
        _append_structured_field(lines, "NOTES", payload.notes)

    elif st == "md_photo":
        lines.append("SOURCE_TYPE: machine_detail_photo")
        _append_structured_field(lines, "TITLE", payload.title)
        _append_structured_field(lines, "DESCRIPTION", payload.description)

    elif st == "md_video":
        lines.append("SOURCE_TYPE: machine_detail_video")
        _append_structured_field(lines, "TITLE", payload.title)
        _append_structured_field(lines, "DESCRIPTION", payload.description)

    text = "\n".join(lines).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Structured source text is empty")

    return text


def _estimate_index_storage_bytes_for_text(text_chars: int) -> int:
    effective_step = max(1, CHUNK_TARGET_CHARS - min(CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS - 1))
    est_chunks = int(math.ceil(max(1, int(text_chars or 0)) / effective_step))

    bytes_per_char = 3
    bytes_per_chunk = 2000

    return int(int(text_chars or 0) * bytes_per_char + est_chunks * bytes_per_chunk)

def _collapse_structured_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    page_from = min(int(c.get("page_from") or 1) for c in chunks)
    page_to = max(int(c.get("page_to") or 1) for c in chunks)

    lines: list[str] = []
    seen = set()

    for c in chunks:
        txt = (c.get("chunk_text") or "").strip()
        if not txt:
            continue

        for ln in txt.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            if ln in seen:
                continue
            seen.add(ln)
            lines.append(ln)

    merged = "\n".join(lines).strip()
    if not merged:
        return []

    return [
        {
            "chunk_index": 1,
            "page_from": page_from,
            "page_to": page_to,
            "chunk_text": merged,
        }
    ]

def _extract_code_tokens(q: str) -> list[str]:
    q = (q or "").strip().upper()
    if not q:
        return []

    toks = CODE_TOKEN_REGEX.findall(q)
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:5]


def _infer_machine_components(q: str) -> list[str]:
    if not q:
        return []

    schema = {
        "name": "component_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "components": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["components"]
        }
    }

    system_msg = (
        "You are an industrial machine expert. "
        "Given a machine symptom, list machine components likely involved. "
        "Return short component names only."
    )

    user_msg = f"Symptom:\n{q}"

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4.1-nano",
            json_schema=schema,
            timeout=20,
        )

        comps = parsed.get("components") or []

        out = []
        seen = set()

        for c in comps:
            c = str(c).strip().lower()
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)

        return out[:6]

    except Exception:
        return []

def _build_diagnostic_queries(q: str, inferred_components: list[str]) -> list[str]:
    q = re.sub(r"\s+", " ", (q or "").strip())
    comps = [re.sub(r"\s+", " ", str(x).strip()) for x in (inferred_components or []) if str(x).strip()]
    if not q:
        return []

    q_low = _normalize_unicode_advanced(q).lower()

    out: list[str] = []
    seen = set()

    def add(s: str):
        s = re.sub(r"\s+", " ", (s or "").strip())
        if not s:
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    def has_any(stems: list[str]) -> bool:
        return any(st in q_low for st in stems)

    symptom_aliases: list[str] = []

    if has_any(["vibr", "oscillat", "oscillaz"]):
        symptom_aliases.extend(["vibration", "vibrazione"])
    if has_any(["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"]):
        symptom_aliases.extend(["noise", "rumore"])
    if has_any(["overheat", "surriscal", "temperat", "hot", "cald"]):
        symptom_aliases.extend(["overheating", "surriscaldamento"])
    if has_any(["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"]):
        symptom_aliases.extend(["jam", "blocco"])
    if has_any(["lubric", "lubrif", "oil", "olio", "grease", "grasso"]):
        symptom_aliases.extend(["lubrication", "lubrificazione"])
    if has_any(["feed", "advance", "avanz", "wire", "filo", "material"]):
        symptom_aliases.extend(["feed", "avanzamento"])
    if has_any(["bend", "bending", "pieg", "forming", "formatura"]):
        symptom_aliases.extend(["bending", "piegatura"])

    add(q)

    if comps:
        add(q + " " + " ".join(comps[:3]))

    add(f"root cause {q}")
    add(f"causa {q}")

    if symptom_aliases:
        add(f"{q} {symptom_aliases[0]}")
        if len(symptom_aliases) > 1:
            add(f"{q} {symptom_aliases[1]}")

    if comps:
        add(f"{q} {comps[0]}")
        add(f"root cause {comps[0]} {q}")

    add(f"diagnosi {q}")

    return out[:8]


def _rrf_merge_candidates(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    best_item: dict[str, dict] = {}

    for ranked in ranked_lists:
        for idx, item in enumerate(ranked):
            cid = str(item.get("citation_id") or "").strip()
            if not cid:
                continue

            score = 1.0 / float(k + idx + 1)
            scores[cid] = scores.get(cid, 0.0) + score

            prev = best_item.get(cid)
            if prev is None or float(item.get("similarity", 0.0)) > float(prev.get("similarity", 0.0)):
                best_item[cid] = item

    out = []
    for cid, item in best_item.items():
        merged = dict(item)
        merged["rrf_score"] = scores.get(cid, 0.0)
        out.append(merged)

    out.sort(
        key=lambda x: (
            float(x.get("rrf_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )
    return out


def _collect_candidate_keywords(q: str, inferred_components: list[str]) -> list[str]:
    q_norm = _normalize_unicode_advanced(q or "").lower()
    comps = [str(x).strip().lower() for x in (inferred_components or []) if str(x).strip()]

    out = []
    seen = set()

    def add(x: str):
        x = re.sub(r"\s+", " ", (x or "").strip().lower())
        if not x or x in seen:
            return
        seen.add(x)
        out.append(x)

    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from",
        "machine", "problem", "issue", "fault", "cause", "possible", "probable",
        "abnormal", "anomalous", "anomaly", "diagnosis",
        "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "dei", "delle",
        "con", "per", "tra", "fra", "sul", "sulla", "macchina", "problema",
        "guasto", "causa", "possibile", "probabile", "anomalo", "anomala",
        "anomali", "anomale", "durante", "quando", "mentre", "dopo", "prima",
        "nel", "nella", "su"
    }

    for c in comps[:6]:
        add(c)

    for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", q_norm):
        if tok not in stopwords:
            add(tok)

    alias_groups = [
        (["vibr", "oscillat", "oscillaz"], ["vibration", "vibrazione", "oscillation", "oscillazione"]),
        (["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"], ["noise", "rumore", "rattle", "stridore"]),
        (["overheat", "surriscal", "temperat", "hot", "cald"], ["overheating", "surriscaldamento", "temperature", "temperatura"]),
        (["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"], ["jam", "blocco", "stoppage", "arresto"]),
        (["lubric", "lubrif", "oil", "olio", "grease", "grasso"], ["lubrication", "lubrificazione", "oil", "olio", "grease", "grasso"]),
        (["feed", "advance", "avanz", "wire", "filo", "material"], ["feed", "avanzamento", "wire", "filo", "materiale"]),
        (["bend", "bending", "pieg", "forming", "formatura"], ["bending", "piegatura", "forming", "formatura"]),
    ]

    for stems, aliases in alias_groups:
        if any(st in q_norm for st in stems):
            for alias in aliases:
                add(alias)

    return out[:12]


def _expand_with_neighbor_chunks(
    company_id: str,
    bubble_document_id: str,
    citation_ids: list[str],
    *,
    radius: int = 1,
) -> list[dict]:
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
            cur.execute(
                """
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       left(chunk_text, 2000) AS chunk_full
                FROM public.document_chunks
                WHERE company_id=%s
                  AND bubble_document_id=%s
                  AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index;
                """,
                (
                    ASK_SNIPPET_CHARS,
                    company_id,
                    bubble_document_id,
                    min_idx,
                    max_idx,
                ),
            )
            rows = cur.fetchall()
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

def _root_cause_chunk_signal_summary(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
) -> dict:
    txt = _normalize_unicode_advanced(chunk_text or "").lower()
    if not txt:
        return {}

    section = _normalize_unicode_advanced(_extract_section_from_text(chunk_text) or "").lower()
    q_low = _normalize_unicode_advanced(q or "").lower()

    diag_terms = []
    seen_diag = set()
    for x in diagnostic_keywords or []:
        x = re.sub(r"\s+", " ", str(x).strip().lower())
        if len(x) < 3 or x in seen_diag:
            continue
        seen_diag.add(x)
        diag_terms.append(x)

    overview_section_markers = [
        "overview",
        "general features",
        "general description",
        "caratteristiche generali",
        "descrizione generale",
        "technical data",
        "specifications",
        "dati tecnici",
        "caratteristiche tecniche",
        "intended use",
        "destinazione d'uso",
    ]

    description_section_markers = [
        "descrizione della macchina",
        "machine description",
        "description of the machine",
        "descrizione macchina",
    ]

    boilerplate_section_markers = overview_section_markers + [
        "safety",
        "warning",
        "warnings",
        "sicurezza",
        "avvertenze",
        "installation",
        "installazione",
        "electrical connections",
        "collegamenti elettrici",
        "wiring",
        "transport",
        "trasporto",
        "storage",
        "stoccaggio",
        "commissioning",
        "messa in servizio",
        "start-up",
        "startup",
        "prima accensione",
        "messa in moto",
        "foundation",
        "fondazione",
        "positioning",
        "posizionamento",
    ]

    startup_install_markers = [
        "before starting",
        "before start-up",
        "before startup",
        "prima di avviare",
        "prima dell'avviamento",
        "prima della messa in moto",
        "before commissioning",
        "messa in servizio",
        "messa in moto",
        "start the machine",
        "avviare la macchina",
        "control panel",
        "quadro di comando",
        "network voltage",
        "tensione di rete",
        "power supply",
        "alimentazione",
        "minimum level",
        "livello minimo",
        "installation",
        "installazione",
        "transport",
        "trasporto",
        "storage",
        "stoccaggio",
        "foundation",
        "fondazione",
        "positioning",
        "posizionamento",
    ]

    positioning_markers = [
        "positioning",
        "posizionamento",
        "foundation",
        "fondazione",
        "planarity",
        "planarità",
        "level",
        "livella",
        "levelling",
        "leveling",
        "livellamento",
        "support surface",
        "piano di appoggio",
        "rubber shims",
        "spessori di gomma",
        "threaded holes",
        "fori filettati",
        "mounting holes",
        "fori di fondazione",
        "near a wall",
        "vicino ad un muro",
    ]

    safety_access_markers = [
        "work area",
        "zona di lavoro",
        "access door",
        "porta di accesso",
        "porte protette",
        "safety door",
        "safety doors",
        "micro switch",
        "micro switches",
        "microswitch",
        "microinterruttor",
        "automatic cycle",
        "ciclo automatico",
        "protective guard",
        "protective guards",
        "riparo",
        "ripari",
    ]

    acoustic_protection_markers = [
        "acoustic",
        "noise emission",
        "noise emissions",
        "noise level",
        "sound pressure",
        "sound insulation",
        "soundproof",
        "soundproofing",
        "emissioni sonore",
        "livello di rumore",
        "pressione sonora",
        "isolamento acustico",
        "fonoassorb",
        "rumoros",
        "protective panel",
        "protective panels",
        "pannelli di protezione",
    ]

    lube_control_markers = [
        "lubrication circuit",
        "circuito di lubrificazione",
        "automatic lubrication",
        "lubrificazione automatica",
        "pressure switch",
        "pressostato",
        "oil level",
        "livello olio",
        "minimum level",
        "livello minimo",
        "oil tank",
        "serbatoio",
        "pressure drop",
        "cali pressione",
    ]

    strong_component_markers = [
        "bearing",
        "cuscinet",
        "gear",
        "ingran",
        "gearbox",
        "ridutt",
        "shaft",
        "albero",
        "belt",
        "cinghia",
        "chain",
        "catena",
        "roller",
        "rullo",
        "guide",
        "guida",
        "motor",
        "motore",
        "sensor",
        "sensore",
        "encoder",
        "valve",
        "valvol",
        "cylinder",
        "cilindr",
        "pump",
        "pompa",
        "brake",
        "freno",
        "alignment",
        "alline",
        "clearance",
        "gioco",
        "backlash",
        "friction",
        "attrit",
        "slitta",
        "slide",
        "die",
        "stampo",
    ]

    process_markers = [
        "feed",
        "advance",
        "avanz",
        "wire",
        "filo",
        "bend",
        "bending",
        "pieg",
        "forming",
        "formatura",
        "straighten",
        "straightening",
        "raddrizz",
    ]

    symptom_groups = [
        (["vibr", "oscillat", "oscillaz"], ["vibrat", "vibraz", "oscillat", "oscill"]),
        (["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"], ["noise", "rumor", "rumore", "rumoros", "rattle", "squeal", "strid", "sfreg"]),
        (["overheat", "surriscal", "temperat", "hot", "cald"], ["overheat", "surriscal", "temperat", "hot", "cald"]),
        (["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"], ["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"]),
        (["lubric", "lubrif", "oil", "olio", "grease", "grasso"], ["lubric", "lubrif", "oil", "olio", "grease", "grasso"]),
        (["feed", "advance", "avanz", "wire", "filo", "material"], ["feed", "advance", "avanz", "wire", "filo", "material"]),
        (["bend", "bending", "pieg", "forming", "formatura"], ["bend", "bending", "pieg", "forming", "formatura"]),
    ]

    symptom_markers: list[str] = []
    for query_stems, chunk_stems in symptom_groups:
        if any(st in q_low for st in query_stems):
            symptom_markers.extend(chunk_stems)

    def count_hits(markers: list[str], hay: str) -> int:
        return sum(1 for m in markers if m and m in hay)

    spec_markers = [
        "noise level",
        "livello di rumore",
        "sound pressure",
        "pressione sonora",
        "dimensions",
        "dimensioni",
        "weight",
        "peso",
        "voltage",
        "tensione",
        "frequency",
        "frequenza",
    ]

    query_install_related = any(
        st in q_low
        for st in [
            "install",
            "startup",
            "start-up",
            "start",
            "avvi",
            "messa in servizio",
            "messa in moto",
            "commission",
            "elettric",
            "power",
            "alimentaz",
            "tension",
            "posizion",
            "livell",
            "fondaz",
            "planarit",
            "piano di appoggio",
            "support surface",
            "setup",
            "mount",
        ]
    )
    query_safety_related = any(
        st in q_low
        for st in [
            "sicur",
            "safety",
            "ripar",
            "guard",
            "porta",
            "door",
            "microinter",
            "interlock",
            "arrest",
            "stop",
            "emerg",
            "zona di lavoro",
            "work area",
        ]
    )

    query_lube_related = any(
        st in q_low
        for st in [
            "lubric",
            "lubrif",
            "oil",
            "olio",
            "grease",
            "grasso",
            "pressost",
            "pressure switch",
        ]
    )

    return {
        "diag_hits": count_hits(diag_terms[:12], txt),
        "section_diag_hits": count_hits(diag_terms[:12], section),
        "boilerplate_section_hit": any(m in section for m in boilerplate_section_markers),
        "overview_section_hit": any(m in section for m in overview_section_markers),
        "description_section_hit": any(m in section for m in description_section_markers),
        "startup_install_hits": count_hits(startup_install_markers, txt),
        "positioning_hits": count_hits(positioning_markers, txt),
        "safety_access_hits": count_hits(safety_access_markers, txt),
        "acoustic_protection_hits": count_hits(acoustic_protection_markers, txt),
        "lube_control_hits": count_hits(lube_control_markers, txt),
        "strong_component_hits": count_hits(strong_component_markers, txt),
        "process_hits": count_hits(process_markers, txt),
        "symptom_hits": count_hits(list(dict.fromkeys(symptom_markers)), txt),
        "spec_hits": count_hits(spec_markers, txt),
        "query_install_related": query_install_related,
        "query_safety_related": query_safety_related,
        "query_lube_related": query_lube_related,
    }

def _should_downrank_generic_root_cause_chunk(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
) -> bool:
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return False

    if (
        sig["description_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (
        sig["safety_access_hits"] >= 2
        and not sig["query_safety_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (sig["strong_component_hits"] >= 2 or sig["process_hits"] >= 1) and (
        sig["diag_hits"] >= 1 or sig["section_diag_hits"] >= 1
    ):
        return False

    if sig["symptom_hits"] >= 1 and sig["strong_component_hits"] >= 2:
        return False

    if sig["overview_section_hit"] and (
        sig["acoustic_protection_hits"] >= 1 or sig["safety_access_hits"] >= 2
    ):
        return True

    if (
        sig["overview_section_hit"]
        and sig["symptom_hits"] >= 1
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["boilerplate_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["startup_install_hits"] >= 3
        and not sig["query_install_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
    ):
        return True

    if (
        sig["positioning_hits"] >= 2
        and not sig["query_install_related"]
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["lube_control_hits"] >= 2
        and not sig["query_lube_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if sig["spec_hits"] >= 2 and sig["diag_hits"] == 0 and sig["symptom_hits"] == 0:
        return True

    return False

def _should_hard_exclude_root_cause_chunk(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
) -> bool:
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return False

    if (
        sig["description_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (
        sig["safety_access_hits"] >= 2
        and not sig["query_safety_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
        and sig["process_hits"] == 0
    ):
        return True

    if (sig["strong_component_hits"] >= 2 or sig["process_hits"] >= 1) and (
        sig["diag_hits"] >= 1 or sig["section_diag_hits"] >= 1
    ):
        return False

    if sig["symptom_hits"] >= 1 and sig["strong_component_hits"] >= 2:
        return False

    if sig["overview_section_hit"] and (
        sig["acoustic_protection_hits"] >= 1 or sig["safety_access_hits"] >= 2
    ):
        return True

    if (
        sig["overview_section_hit"]
        and sig["symptom_hits"] >= 1
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["startup_install_hits"] >= 4
        and not sig["query_install_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
        and sig["process_hits"] == 0
    ):
        return True

    if (
        sig["positioning_hits"] >= 3
        and not sig["query_install_related"]
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["acoustic_protection_hits"] >= 2
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
    ):
        return True

    if (
        sig["spec_hits"] >= 3
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    return False

def _score_root_cause_chunk_semantic(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
) -> dict:
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return {
            "semantic_score": -0.30,
            "semantic_band": "weak",
        }

    score = 0.0

    # segnali positivi
    score += min(0.20, 0.05 * float(sig.get("diag_hits", 0)))
    score += min(0.18, 0.08 * float(sig.get("section_diag_hits", 0)))
    score += min(0.16, 0.06 * float(sig.get("symptom_hits", 0)))
    score += min(0.22, 0.05 * float(sig.get("strong_component_hits", 0)))
    score += min(0.18, 0.07 * float(sig.get("process_hits", 0)))

    # sinergie utili
    if sig.get("symptom_hits", 0) >= 1 and sig.get("strong_component_hits", 0) >= 1:
        score += 0.08
    if sig.get("diag_hits", 0) >= 1 and (
        sig.get("strong_component_hits", 0) >= 1 or sig.get("process_hits", 0) >= 1
    ):
        score += 0.08
    if sig.get("section_diag_hits", 0) >= 1 and sig.get("process_hits", 0) >= 1:
        score += 0.06

    # penalità generali
    if sig.get("description_section_hit"):
        score -= 0.28
    if sig.get("overview_section_hit"):
        score -= 0.14
    if sig.get("boilerplate_section_hit"):
        score -= 0.10

    if sig.get("startup_install_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= min(0.18, 0.04 * float(sig.get("startup_install_hits", 0)))

    if sig.get("positioning_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= min(0.22, 0.05 * float(sig.get("positioning_hits", 0)))

    if sig.get("safety_access_hits", 0) >= 2 and not sig.get("query_safety_related"):
        score -= min(0.22, 0.05 * float(sig.get("safety_access_hits", 0)))

    if sig.get("acoustic_protection_hits", 0) >= 1 and not sig.get("query_safety_related"):
        score -= min(0.18, 0.06 * float(sig.get("acoustic_protection_hits", 0)))

    if (
        sig.get("lube_control_hits", 0) >= 2
        and not sig.get("query_lube_related")
        and sig.get("process_hits", 0) == 0
    ):
        score -= min(0.14, 0.04 * float(sig.get("lube_control_hits", 0)))

    if sig.get("spec_hits", 0) >= 2:
        score -= min(0.14, 0.04 * float(sig.get("spec_hits", 0)))

    score = max(-0.45, min(0.45, score))

    if score >= 0.18:
        band = "strong"
    elif score >= 0.02:
        band = "medium"
    else:
        band = "weak"

    return {
        "semantic_score": score,
        "semantic_band": band,
    }

def _q_has_any(q: str, hints: list[str]) -> bool:
    qq = (q or "").lower()
    return any(h in qq for h in hints)


def _clean_tail(s: str) -> str:
    return (s or "").rstrip(".,;:!?)\"]}")


def _extract_first(regex: re.Pattern, text: str) -> Optional[str]:
    if not text:
        return None
    m = regex.search(text)
    if not m:
        return None
    return _clean_tail(m.group(1))


def _pick_entity_from_citations(q: str, citations: list[dict]) -> Optional[tuple[str, dict]]:
    wants_url = _q_has_any(q, URL_HINTS)
    wants_email = _q_has_any(q, EMAIL_HINTS)
    wants_phone = _q_has_any(q, PHONE_HINTS)

    if not (wants_url or wants_email or wants_phone):
        return None

    for c in citations:
        snip = c.get("snippet", "") or ""

        if wants_url:
            u = _extract_first(URL_REGEX, snip)
            if u:
                return (u, c)

        if wants_email:
            e = _extract_first(EMAIL_REGEX, snip)
            if e:
                return (e, c)

        if wants_phone:
            p = _extract_first(PHONE_REGEX, snip)
            if p:
                return (p, c)

    return None


def _db_find_token_chunk(
    company_id: str,
    machine_id: str,
    token: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> Optional[dict]:
    token = (token or "").strip()
    if not token:
        return None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            if doc_ids:
                where.append("bubble_document_id = ANY(%s)")
                params.append(doc_ids)
            elif bubble_document_id:
                where.append("bubble_document_id = %s")
                params.append(bubble_document_id)
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)
            else:
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)

            where_sql = " AND ".join(where)

            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet
                FROM public.document_chunks
                WHERE {where_sql}
                  AND chunk_text ILIKE %s
                ORDER BY bubble_document_id, page_from, chunk_index
                LIMIT 1;
                """,
                [ASK_SNIPPET_CHARS, *params, f"%{token}%"],
            )
            row = cur.fetchone()
            if not row:
                return None

            bdid, chunk_index, page_from, page_to, snippet = row
            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            return {
                "citation_id": citation_id,
                "bubble_document_id": str(bdid),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": (snippet or "").strip(),
                "similarity": 0.0,
            }
    finally:
        conn.close()


def _db_find_entity_chunk(
    company_id: str,
    machine_id: str,
    kind: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> Optional[dict]:
    if kind == "url":
        pattern = r"(https?://|www\.)"
        rx = URL_REGEX
    elif kind == "email":
        pattern = r"@[A-Z0-9.-]+\.[A-Z]{2,}"
        rx = EMAIL_REGEX
    elif kind == "phone":
        pattern = r"\+?\d[\d\s().-]{7,}\d"
        rx = PHONE_REGEX
    else:
        return None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            if doc_ids:
                where.append("bubble_document_id = ANY(%s)")
                params.append(doc_ids)
            elif bubble_document_id:
                where.append("bubble_document_id = %s")
                params.append(bubble_document_id)
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)
            else:
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)

            where_sql = " AND ".join(where)

            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet
                FROM public.document_chunks
                WHERE {where_sql}
                  AND chunk_text ~* %s
                ORDER BY bubble_document_id, page_from, chunk_index
                LIMIT 1;
                """,
                [ASK_SNIPPET_CHARS, *params, pattern],
            )
            row = cur.fetchone()
            if not row:
                return None

            bdid, chunk_index, page_from, page_to, snippet = row
            snippet = (snippet or "").strip()
            value = _extract_first(rx, snippet)
            if not value:
                return None

            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            return {
                "citation_id": citation_id,
                "bubble_document_id": str(bdid),
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": snippet,
                "value": value,
            }
    finally:
        conn.close()


def _dedup_citations_by_snippet(citations: list[dict], max_items: int) -> list[dict]:
    def norm(s: str) -> str:
        s = _normalize_unicode_advanced(s or "")
        s = re.sub(r"^SECTION:\s*[^\n]+\n?", "", s, flags=re.IGNORECASE).strip()

        lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
        cleaned = []
        seen_lines = set()

        for ln in lines:
            ln_low = re.sub(r"\s+", " ", ln.lower()).strip()

            if re.fullmatch(r"\d+", ln_low):
                continue

            if ln_low in seen_lines:
                continue

            seen_lines.add(ln_low)
            cleaned.append(ln_low)

        s = " ".join(cleaned)
        s = re.sub(r"\s+", " ", s).strip()
        return s[:500]

    best = {}
    for c in citations:
        k = norm(c.get("snippet", ""))
        if k:
            k = (
                f"{str(c.get('bubble_document_id') or '').strip()}"
                f"|{int(c.get('page_from') or 0)}"
                f"|{int(c.get('page_to') or 0)}"
                f"|{k[:220]}"
            )
        else:
            k = str(c.get("citation_id") or "").strip()

        prev = best.get(k)
        if (prev is None) or (float(c.get("similarity", 0.0)) > float(prev.get("similarity", 0.0))):
            best[k] = c

    out = list(best.values())
    out.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    return out[:max_items]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0

    for i in range(min(len(a), len(b))):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb

    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _mmr_select(
    q_vec: list[float],
    candidates: list[dict],
    top_k: int,
    lambda_mult: float = 0.85,
) -> list[dict]:
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = candidates[:]

    remaining.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_score = -1e9

        for i, cand in enumerate(remaining):
            sim_q = float(cand.get("similarity", 0.0))

            max_sim_sel = 0.0
            ce = cand.get("embedding_list") or []
            for s in selected:
                se = s.get("embedding_list") or []
                max_sim_sel = max(max_sim_sel, _cosine_sim(ce, se))

            score = lambda_mult * sim_q - (1.0 - lambda_mult) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _fts_search_chunks(
    company_id: str,
    machine_id: str,
    q: str,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> list[dict]:
    q = (q or "").strip()
    if not q:
        return []

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            if doc_ids:
                where.append("bubble_document_id = ANY(%s)")
                params.append(doc_ids)
            elif bubble_document_id:
                where.append("bubble_document_id = %s")
                params.append(bubble_document_id)
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)
            else:
                where.append("(machine_id = %s OR machine_id IS NULL)")
                params.append(machine_id)

            where_sql = " AND ".join(where)

            cur.execute(
                f"""
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet,
                       ts_rank_cd(
                           to_tsvector('simple', chunk_text),
                           plainto_tsquery('simple', %s)
                       ) AS rank
                FROM public.document_chunks
                WHERE {where_sql}
                  AND to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s;
                """,
                [ASK_SNIPPET_CHARS, q, *params, q, top_k],
            )
            rows = cur.fetchall()

            out: list[dict] = []
            for (bdid, chunk_index, page_from, page_to, snippet, _rank) in rows:
                citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
                out.append(
                    {
                        "citation_id": citation_id,
                        "bubble_document_id": bdid,
                        "page_from": int(page_from),
                        "page_to": int(page_to),
                        "snippet": (snippet or "").strip(),
                        "similarity": 0.0,
                    }
                )
            return out
    finally:
        conn.close()


def _normalize_unicode_advanced(s: str) -> str:
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ﬁ", "fi")
    s = s.replace("ﬂ", "fl")
    s = s.replace("ﬀ", "ff")
    s = s.replace("ﬃ", "ffi")
    s = s.replace("ﬄ", "ffl")
    s = s.replace("–", "-").replace("—", "-").replace("‐", "-")
    s = s.replace("\u00A0", " ")
    s = s.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    return s


def _dehyphenate_lines_keep_newlines(s: str) -> str:
    if not s:
        return ""

    lines = s.split("\n")
    out: list[str] = []
    i = 0

    end_word_hyphen = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-$")
    next_starts_lower = re.compile(r"^[a-zà-öø-ÿ]")
    avoid_prev = re.compile(r"[0-9_]\-$")
    avoid_next = re.compile(r"^[0-9_]+")

    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            cur_stripped = cur.rstrip()
            nxt_stripped = nxt.lstrip()

            if nxt_stripped.startswith(("-", "•", "·", "*")):
                out.append(cur)
                i += 1
                continue

            if avoid_prev.search(cur_stripped) or avoid_next.search(nxt_stripped):
                out.append(cur)
                i += 1
                continue

            m = end_word_hyphen.search(cur_stripped)
            if m and next_starts_lower.search(nxt_stripped):
                merged = cur_stripped[:-1] + nxt_stripped
                out.append(merged)
                i += 2
                continue

        out.append(cur)
        i += 1

    return "\n".join(out)


def _normalize_text_keep_lines(s: str) -> str:
    if not s:
        return ""
    s = _normalize_unicode_advanced(s)
    s = _dehyphenate_lines_keep_newlines(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")

    out = []
    prev_space = False
    for ch in s:
        if ch == " ":
            if prev_space:
                continue
            prev_space = True
            out.append(" ")
        else:
            prev_space = False
            out.append(ch)

    return "".join(out).strip()


def _pymupdf_page_to_text_blocks(page: "fitz.Page") -> str:
    blocks = page.get_text("blocks") or []
    blocks_sorted = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

    parts: list[str] = []
    for b in blocks_sorted:
        txt = (b[4] or "").strip()
        if not txt:
            continue
        parts.append(txt)

    return "\n\n".join(parts).strip()


def _extract_pages_with_layout_blocks(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out: list[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            t = _pymupdf_page_to_text_blocks(page)
            t = _normalize_text_keep_lines(t)
            out.append(t)
        return out
    finally:
        doc.close()


def _hf_norm_line(s: str) -> str:
    s = _normalize_text_keep_lines(s)
    s = s.lower()
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_top_bottom_lines(page_text: str, top_n: int = 4, bottom_n: int = 4) -> tuple[list[str], list[str]]:
    lines = [ln.strip() for ln in (page_text or "").split("\n")]
    lines = [ln for ln in lines if ln]
    top = lines[:top_n] if top_n > 0 else []
    bottom = lines[-bottom_n:] if bottom_n > 0 else []
    return top, bottom


def _detect_repeated_headers_footers(
    pages_text: list[str],
    top_n: int = 4,
    bottom_n: int = 4,
    min_ratio: float = 0.7,
    min_len: int = 8,
    max_len: int = 140,
) -> tuple[set[str], set[str]]:
    if not pages_text:
        return set(), set()

    top_counts: dict[str, int] = {}
    bot_counts: dict[str, int] = {}

    for txt in pages_text:
        top, bottom = _extract_top_bottom_lines(txt, top_n=top_n, bottom_n=bottom_n)
        for ln in top:
            k = _hf_norm_line(ln)
            if min_len <= len(k) <= max_len:
                top_counts[k] = top_counts.get(k, 0) + 1
        for ln in bottom:
            k = _hf_norm_line(ln)
            if min_len <= len(k) <= max_len:
                bot_counts[k] = bot_counts.get(k, 0) + 1

    n_pages = len(pages_text)
    thr = int(math.ceil(n_pages * min_ratio))

    top_keep = {k for (k, c) in top_counts.items() if c >= thr}
    bot_keep = {k for (k, c) in bot_counts.items() if c >= thr}
    return top_keep, bot_keep


_PAGE_NOISE_RX = re.compile(
    r"^(?:"
    r"(?:page|pagina)\s*#?(?:\s*of\s*#?)?"
    r"|#\s*/\s*#"
    r")$",
    re.IGNORECASE,
)


def _is_page_noise_line(line: str) -> bool:
    k = _hf_norm_line(line)
    if not k:
        return False
    if len(k) > 40:
        return False
    return _PAGE_NOISE_RX.match(k) is not None


def _strip_page_noise_prefix(line: str) -> str:
    if not line:
        return line

    line = _normalize_unicode_advanced(line)
    line = re.sub(
        r"\b(?:page|pagina)\s*\d+\s*(?:of|di)?\s*\d*\b",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def _remove_headers_footers_from_page(
    page_text: str,
    header_norm: set[str],
    footer_norm: set[str],
    top_n: int = 4,
    bottom_n: int = 4,
) -> str:
    lines = [ln.strip() for ln in (page_text or "").split("\n")]

    scan_n = min(12, len(lines))
    for k in range(scan_n):
        lines[k] = _strip_page_noise_prefix(lines[k])
        if _is_page_noise_line(lines[k]):
            lines[k] = ""

    for i in range(min(top_n, len(lines))):
        lines[i] = _strip_page_noise_prefix(lines[i])
        if (_hf_norm_line(lines[i]) in header_norm) or _is_page_noise_line(lines[i]):
            lines[i] = ""

    for j in range(len(lines) - min(bottom_n, len(lines)), len(lines)):
        if 0 <= j < len(lines):
            lines[j] = _strip_page_noise_prefix(lines[j])
        if 0 <= j < len(lines) and ((_hf_norm_line(lines[j]) in footer_norm) or _is_page_noise_line(lines[j])):
            lines[j] = ""

    out_lines = []
    prev_empty = False
    for ln in lines:
        ln = ln.strip()
        if not ln:
            if prev_empty:
                continue
            prev_empty = True
            out_lines.append("")
        else:
            prev_empty = False
            out_lines.append(ln)

    return "\n".join(out_lines).strip()


def _strip_hf_from_chunk_text(chunk_text: str, header_norm: set[str], footer_norm: set[str]) -> str:
    if not chunk_text:
        return ""

    lines = [ln.strip() for ln in chunk_text.split("\n")]
    cleaned = []
    for ln in lines:
        if not ln:
            continue

        k = _hf_norm_line(ln)
        if (k in header_norm) or (k in footer_norm):
            continue

        ln2 = _strip_page_noise_prefix(ln)
        if _is_page_noise_line(ln2):
            continue

        cleaned.append(ln2)

    return "\n".join(cleaned).strip()


def _looks_like_bullet(line: str) -> bool:
    s = (line or "").lstrip()
    if not s:
        return False
    if s.startswith(("•", "·", "*", "-")):
        return True
    if re.match(r"^\(?\d+\)?[.)]\s+", s):
        return True
    if re.match(r"^[a-zA-Z][.)]\s+", s):
        return True
    return False


def _looks_like_table(line: str) -> bool:
    if not line:
        return False

    s = line.rstrip("\n")

    if "|" in s and re.search(r"\S+\s*\|\s*\S+", s):
        return True
    if len(re.findall(r"\s{3,}", s)) >= 2:
        return True
    if re.search(r"\.{4,}", s):
        return True

    tokens = re.split(r"\s+", s.strip())
    if len(tokens) >= 6:
        if re.search(r"\d", s) and (re.search(r"\b(rpm|bar|mm|cm|kg|°c|v|a|hz)\b", s, re.IGNORECASE) is not None):
            return True

    if len(tokens) >= 4 and len(re.findall(r"\s{2,}", s)) >= 3:
        return True

    return False


def _looks_like_title(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", "", s)
    if letters and letters.isupper() and len(s) <= 80:
        return True
    return False


_SECTION_ENUM_RX = re.compile(r"^\s*\d+(?:\.\d+){0,3}\s+[A-Za-zÀ-ÖØ-öø-ÿ]")
_SECTION_ALLCAPS_RX = re.compile(r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9\s\-]{3,}$")


def _looks_like_section_header(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    if _SECTION_ENUM_RX.match(s) and len(s) <= 120:
        return True
    if _SECTION_ALLCAPS_RX.match(s) and len(s.split()) <= 6:
        return True
    return False


def _reflow_paragraphs_conservative(page_text: str) -> str:
    if not page_text:
        return ""

    lines = [ln.rstrip() for ln in page_text.split("\n")]
    out: list[str] = []
    i = 0

    while i < len(lines):
        cur = (lines[i] or "").rstrip()
        if not cur:
            out.append("")
            i += 1
            continue

        if _looks_like_bullet(cur) or _looks_like_table(cur) or _looks_like_title(cur):
            out.append(cur)
            i += 1
            continue

        j = i
        buf = cur

        while j + 1 < len(lines):
            nxt = (lines[j + 1] or "").strip()
            if not nxt:
                break

            if _looks_like_table(buf) or _looks_like_table(nxt):
                break
            if _looks_like_bullet(nxt) or _looks_like_table(nxt) or _looks_like_title(nxt):
                break
            if re.search(r"[.;:!?]$", buf):
                break
            if not re.match(r"^[a-zà-öø-ÿ]", nxt):
                break

            buf = buf + " " + nxt
            j += 1

        out.append(buf)
        i = j + 1

    cleaned = []
    prev_empty = False
    for ln in out:
        ln = ln.strip()
        if not ln:
            if prev_empty:
                continue
            prev_empty = True
            cleaned.append("")
        else:
            prev_empty = False
            cleaned.append(ln)

    return "\n".join(cleaned).strip()


_TOC_TITLE_RX = re.compile(r"\b(?:contents|content|indice|index|table of contents)\b", re.IGNORECASE)


def _looks_like_toc_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    if re.search(r"\.{4,}", s):
        return True
    if re.search(r"\b\d+(?:\.\d+){1,}\b", s) and re.search(r"\s\d{1,4}\s*$", s):
        return True
    if re.search(r"\s\d{1,4}\s*$", s) and len(s) <= 140 and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s):
        return True
    if len(re.findall(r"\s{3,}", s)) >= 2 and re.search(r"\d{1,4}\s*$", s):
        return True

    return False


def _strip_toc_lines(page_text: str) -> str:
    if not page_text:
        return ""

    lines = [ln.rstrip() for ln in page_text.split("\n")]
    kept: list[str] = []
    removed = 0

    for ln in lines:
        if _looks_like_toc_line(ln):
            removed += 1
            continue
        kept.append(ln)

    out = "\n".join(kept).strip()
    if removed >= 6 and len(out) < 200:
        return ""
    return out


def _maybe_remove_toc(page_text: str) -> str:
    if not page_text:
        return ""

    if _TOC_TITLE_RX.search(page_text[:800] or ""):
        return _strip_toc_lines(page_text)

    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    if len(lines) < 8:
        return page_text

    toc_hits = sum(1 for ln in lines[:40] if _looks_like_toc_line(ln))
    ratio = toc_hits / max(1, min(len(lines), 40))
    if toc_hits >= 6 and ratio >= 0.30:
        return _strip_toc_lines(page_text)

    return page_text


_SENT_SPLIT_RX = re.compile(r"(?<=[\.\!\?])\s+")


def _split_sentences_conservative(text: str) -> list[str]:
    if not text:
        return []

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out: list[str] = []

    for ln in lines:
        if ln.startswith(("•", "·", "*", "-")) or re.match(r"^\(?\d+\)?[.)]\s+", ln):
            out.append(ln)
            continue

        parts = [p.strip() for p in _SENT_SPLIT_RX.split(ln) if p.strip()]
        out.extend(parts)

    return out


def _chunk_sentences_with_pages(
    pages: list[tuple[int, str]],
    target_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[dict]:
    seq: list[tuple[int, str, Optional[str]]] = []
    current_section: Optional[str] = None

    for pn, txt in pages:
        for s in _split_sentences_conservative(txt or ""):
            s_clean = s.strip()
            if _looks_like_section_header(s_clean):
                current_section = s_clean
            seq.append((int(pn), s_clean, current_section))

    if not seq:
        return []

    chunks: list[dict] = []
    i = 0
    chunk_index = 1

    while i < len(seq):
        buf: list[str] = []
        pages_in_chunk: list[int] = []

        total = 0
        j = i
        chunk_section: Optional[str] = seq[i][2]

        while j < len(seq):
            pn, s, sent_section = seq[j]
            add = s + " "

            # Non attraversare il cambio di sezione
            if buf and sent_section != chunk_section:
                break

            if total + len(add) > target_chars and total >= min_chars:
                break

            buf.append(s)
            pages_in_chunk.append(pn)
            total += len(add)

            if chunk_section is None and sent_section is not None:
                chunk_section = sent_section

            j += 1

        if not buf:
            pn, s, sent_section = seq[i]
            buf = [s]
            pages_in_chunk = [pn]
            chunk_section = sent_section
            j = i + 1

        page_from = min(pages_in_chunk)
        page_to = max(pages_in_chunk)
        chunk_text = "\n".join(buf).strip()

        if chunk_section:
            chunk_text = f"SECTION: {chunk_section}\n" + chunk_text

        chunks.append(
            {
                "chunk_index": chunk_index,
                "page_from": page_from,
                "page_to": page_to,
                "chunk_text": chunk_text,
            }
        )
        chunk_index += 1

        if overlap_chars > 0:
            carry = []
            carry_len = 0
            k = j - 1
            while k >= i and carry_len < overlap_chars:
                pn, s, sent_section = seq[k]
                if sent_section != chunk_section:
                    break
                carry.insert(0, (pn, s))
                carry_len += len(s) + 1
                k -= 1
            i = max(i + 1, j - len(carry))
        else:
            i = j

    return [c for c in chunks if c.get("chunk_text")]


def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    payload = {"model": OPENAI_EMBED_MODEL, "input": texts}
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_EMBED_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"OpenAI embeddings failed: {r.status_code} {r.text}")

    data = r.json()
    out = [None] * len(texts)
    for item in data.get("data", []):
        out[int(item["index"])] = item["embedding"]

    if any(v is None for v in out):
        raise Exception("OpenAI embeddings response missing some items")

    return out


def _openai_chat(messages: list[dict]) -> str:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    payload = {"model": OPENAI_CHAT_MODEL, "messages": messages, "temperature": 0.2}
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"OpenAI chat failed: {r.status_code} {r.text}")

    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


def _extract_section_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"^SECTION:\s*(.+)$", text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "")[:120]


def _extract_citation_ids_from_answer(answer: str) -> list[str]:
    answer = (answer or "").strip()
    if not answer:
        return []

    ids = re.findall(r"\[([^\]]+)\]", answer)
    out: list[str] = []
    seen = set()

    for cid in ids:
        cid = (cid or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)

    return out


def _ground_citations_to_answer(answer: str, citations: list[dict]) -> list[dict]:
    if not answer or not citations:
        return citations

    used_ids = set(_extract_citation_ids_from_answer(answer))

    if not used_ids:
        return citations

    grounded = [c for c in citations if str(c.get("citation_id") or "").strip() in used_ids]

    if not grounded:
        return citations

    return grounded


def _openai_chat_json(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    json_schema: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    payload = {
        "model": (model or OPENAI_CHAT_MODEL),
        "messages": messages,
        "temperature": 0,
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise Exception(f"OpenAI chat JSON failed: {r.status_code} {r.text}")

    data = r.json()
    msg = (data.get("choices", [{}])[0].get("message", {}) or {})
    content = msg.get("content", "")

    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = str(content or "").strip()

    if not text:
        raise Exception("OpenAI chat JSON empty response")

    try:
        return json.loads(text)
    except Exception as e:
        raise Exception(f"OpenAI chat JSON parse failed: {str(e)} | raw={text[:500]}")


def _llm_rerank_citations(
    q: str,
    candidates: list[dict],
    top_k: int,
    diagnostic_mode: bool = False,
) -> list[str]:
    q = (q or "").strip()
    if not q or not candidates:
        return []

    requested_k = max(1, min(int(top_k or 1), ASK_MAX_TOP_K))
    max_candidates = max(1, min(int(RERANK_MAX_CANDIDATES), len(candidates)))

    items = []
    seen = set()

    for c in candidates[:max_candidates]:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)

        full_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        section = _extract_section_from_text(full_text)

        snippet = (c.get("snippet") or "").strip()
        snippet = re.sub(r"^SECTION:\s*[^\n]+\n?", "", snippet).strip()

        items.append(
            {
                "citation_id": cid,
                "section": section,
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "similarity": round(float(c.get("similarity", 0.0)), 4),
                "snippet": snippet[:RERANK_SNIPPET_CHARS],
            }
        )

    if not items:
        return []

    schema = {
        "name": "citation_rerank",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["selected_ids"],
        },
    }

    if diagnostic_mode:
        system_msg = (
            "Selezioni le citazioni più utili per diagnosticare un problema tecnico su una macchina industriale. "
            "Regole obbligatorie: "
            "1) tieni solo fonti che parlano del fenomeno o dei componenti coinvolti; "
            "2) scarta fonti generiche di manutenzione, sicurezza, installazione o lubrificazione se non sono direttamente legate al sintomo; "
            "3) preferisci fonti che descrivono componenti, regolazioni, giochi meccanici, allineamenti o anomalie operative; "
            "4) overview, caratteristiche generali, safety, acoustic, installation, start-up e sezioni simili sono bassa priorità, salvo che il sintomo riguardi esplicitamente quei temi; "
            "5) se una fonte è soprattutto boilerplate e solo marginalmente correlata, scartala; "
            "6) se due fonti sono simili, tieni la più specifica; "
            "7) non collassare tutto su una sola fonte se 2-3 fonti specifiche coprono aree causali diverse; "
            "8) restituisci il minor numero possibile di citation_id davvero utili."
        )
        
    else:
        system_msg = (
            "Selezioni le citazioni minime e più precise per rispondere a una domanda tecnica industriale. "
            "Obiettivo: tenere solo le fonti strettamente necessarie e scartare quelle solo vagamente correlate. "
            "Regole obbligatorie: "
            "1) seleziona il minor numero possibile di citation_id utili; "
            "2) preferisci chunk che contengono direttamente la risposta; "
            "3) scarta chunk generici di manutenzione o contesto se non aggiungono informazione utile; "
            "4) se due chunk sono simili, tieni solo il più specifico."
        )

    user_msg = (
        f"DOMANDA:\n{q}\n\n"
        f"TOP_K_DESIDERATO: {requested_k}\n\n"
        "CANDIDATI_JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Restituisci JSON valido con questa forma:\n"
        '{"selected_ids":["id1","id2"]}\n'
        "Ordina selected_ids dal migliore al meno rilevante. "
        "Non includere più di TOP_K_DESIDERATO elementi."
    )

    parsed = _openai_chat_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=OPENAI_RERANK_MODEL,
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    selected = parsed.get("selected_ids") or []
    if not isinstance(selected, list):
        return []

    allowed = {item["citation_id"] for item in items}
    out = []
    used = set()

    for cid in selected:
        cid = str(cid or "").strip()
        if not cid or cid not in allowed or cid in used:
            continue
        used.add(cid)
        out.append(cid)
        if len(out) >= requested_k:
            break

    return out


def _llm_filter_diagnostic_chunks(
    q: str,
    candidates: list[dict],
    max_keep: int,
) -> list[str]:
    if not q or not candidates:
        return []

    items = []

    for c in candidates[:18]:
        cid = str(c.get("citation_id") or "").strip()
        snippet = (c.get("snippet") or "").strip()
        section = _extract_section_from_text(c.get("chunk_full") or c.get("snippet") or "")

        items.append({
            "citation_id": cid,
            "section": section[:120],
            "evidence_family": _root_cause_evidence_family_key(c),
            "matched_subsystems": c.get("matched_subsystems") or [],
            "subsystem_score": round(float(c.get("subsystem_score", 0.0)), 4),
            "generic_downranked": bool(c.get("generic_downranked")),
            "snippet": snippet[:300]
        })

    schema = {
        "name": "diagnostic_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["selected_ids"]
        }
    }

    system_msg = (
        "Selezioni solo le fonti realmente utili per diagnosticare un problema tecnico su una macchina industriale.\n"
        "Regole:\n"
        "1) Tieni solo fonti che parlano del fenomeno o dei componenti coinvolti.\n"
        "2) Scarta fonti generiche di manutenzione, sicurezza, installazione o lubrificazione se non sono direttamente legate al sintomo.\n"
        "3) Se una fonte parla solo di controlli generici o procedure standard, scartala.\n"
        "4) Mantieni poche fonti ma molto pertinenti.\n"
        "5) Non collassare tutto su una sola fonte se esistono 2-3 aree causali diverse ben supportate.\n"
        "6) Le fonti con generic_downranked=true sono bassa priorità e vanno tenute solo se il sintomo coincide in modo diretto.\n"
        "7) Preferisci sezioni operative o di componente rispetto a overview, safety, installation, start-up o caratteristiche generali.\n"
        "8) Evita di selezionare più citation_id della stessa evidence_family se una sola fonte rappresenta già bene quell'area.\n"
        "9) Seleziona fonti che coprono aree causali diverse quando sono ben supportate.\n"
        "10) Preferisci fonti allineate ai sottosistemi dominanti implicati dal sintomo.\n"
        "11) Se esistono fonti di sottosistemi secondari, mantienile solo se spiegano una causa davvero plausibile e non indiretta.\n"
    )

    user_msg = (
        f"PROBLEMA:\n{q}\n\n"
        f"CANDIDATI:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        f"Restituisci JSON con gli id delle fonti più utili alla diagnosi."
    )

    parsed = _openai_chat_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=OPENAI_RERANK_MODEL,
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    selected = parsed.get("selected_ids") or []

    out = []
    used = set()

    for cid in selected:
        cid = str(cid).strip()
        if cid and cid not in used:
            used.add(cid)
            out.append(cid)
        if len(out) >= max_keep:
            break

    return out

def _llm_build_diagnostic_evidence_matrix(
    q: str,
    citations: list[dict],
    max_causes: int,
) -> dict:
    if not q or not citations:
        return {}

    max_causes = max(1, min(int(max_causes or 1), 3))

    items = []
    seen = set()

    for c in citations[:10]:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)

        snippet = (c.get("chunk_full") or c.get("snippet") or "").strip()
        snippet = re.sub(r"^SECTION:\s*[^\n]+\n?", "", snippet).strip()

        section = _extract_section_from_text(c.get("chunk_full") or c.get("snippet") or "")

        items.append(
            {
                "citation_id": cid,
                "section": section[:120],
                "evidence_family": _root_cause_evidence_family_key(c),
                "matched_subsystems": c.get("matched_subsystems") or [],
                "subsystem_score": round(float(c.get("subsystem_score", 0.0)), 4),
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "generic_downranked": bool(c.get("generic_downranked")),
                "snippet": snippet[:360],
            }
        )

    if not items:
        return {}

    schema = {
        "name": "diagnostic_evidence_matrix",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "discard_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cause_hypotheses": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cause": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "check_focus": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["cause", "evidence_ids", "check_focus"],
                    },
                },
            },
            "required": ["keep_ids", "discard_ids", "cause_hypotheses"],
        },
    }

    system_msg = (
        "Selezioni e organizzi le evidenze per una root cause analysis industriale.\n"
        "Obiettivo: tenere solo le fonti davvero utili e raggrupparle per area causale.\n"
        "Regole obbligatorie:\n"
        "1) keep_ids = solo citazioni utili alla diagnosi.\n"
        "2) discard_ids = citazioni generiche, ripetitive, di solo contesto o sicurezza.\n"
        "3) cause_hypotheses = massimo poche ipotesi distinte; non duplicare varianti della stessa causa.\n"
        "4) Ogni ipotesi deve usare solo citation_id presenti nei candidati.\n"
        "5) check_focus = verifiche pratiche brevi, non frasi lunghe.\n"
        "6) Non collassare tutto su una sola causa se le citazioni supportano aree causali diverse.\n"
        "7) keep_ids deve mantenere copertura delle aree causali utili, non solo il numero minimo di fonti.\n"
        "8) Le fonti con generic_downranked=true sono bassa priorità e non vanno usate come evidenza centrale se esistono fonti più specifiche.\n"
        "9) Preferisci sezioni operative o di componente rispetto a overview, safety, installation, start-up o caratteristiche generali.\n"
        "10) Evita di mantenere più citation_id della stessa evidence_family se una sola fonte è già rappresentativa.\n"
        "11) keep_ids e cause_hypotheses devono massimizzare la copertura di aree causali diverse, non la ripetizione della stessa area.\n"
        "12) Preferisci ipotesi coerenti con i sottosistemi dominanti implicati dal sintomo.\n"
        "13) Le evidenze di sottosistemi secondari vanno tenute solo se supportano una causa distinta e plausibile.\n"
    )

    user_msg = (
        f"SINTOMO/PROBLEMA:\n{q}\n\n"
        "CITAZIONI_CANDIDATE_JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Restituisci JSON valido."
    )

    parsed = _openai_chat_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=OPENAI_RERANK_MODEL,
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    return parsed if isinstance(parsed, dict) else {}

def _should_use_reranker(
    q: str,
    candidates: list[dict],
    sim_max: float,
    top_k: int,
) -> bool:
    if not RERANK_ENABLED:
        return False
    if not q or not candidates:
        return False
    if len(candidates) < max(2, RERANK_MIN_CANDIDATES):
        return False
    if sim_max is None:
        return False
    if sim_max < RERANK_MIN_SIM_MAX:
        return False
    if sim_max > RERANK_MAX_SIM_MAX:
        return False

    ordered = sorted(
        candidates,
        key=lambda x: float(x.get("similarity", 0.0)),
        reverse=True,
    )
    if len(ordered) < 2:
        return False

    spread = float(ordered[0].get("similarity", 0.0)) - float(
        ordered[min(len(ordered) - 1, top_k - 1)].get("similarity", 0.0)
    )
    if spread > RERANK_MAX_SPREAD:
        return False

    return True


def _unique_non_empty_strings(items: list[Any], limit: Optional[int] = None) -> list[str]:
    out: list[str] = []
    seen = set()

    for item in items or []:
        s = str(item or "").strip()
        if not s:
            continue

        k = s.lower()
        if k in seen:
            continue

        seen.add(k)
        out.append(s)

        if limit is not None and len(out) >= limit:
            break

    return out


def _extract_citation_ids_from_root_cause_json(result: dict) -> list[str]:
    if not isinstance(result, dict):
        return []

    out: list[str] = []
    seen = set()

    for cause in result.get("possible_causes") or []:
        if not isinstance(cause, dict):
            continue

        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if not cid or cid in seen:
                continue

            seen.add(cid)
            out.append(cid)

    return out


def _ground_root_cause_result(
    result: dict,
    citations: list[dict],
    max_causes: int,
) -> tuple[dict, list[dict]]:
    result = result if isinstance(result, dict) else {}
    max_causes = max(1, min(int(max_causes or 1), 5))

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }

    grounded_causes: list[dict] = []

    for cause in result.get("possible_causes") or []:
        if not isinstance(cause, dict):
            continue

        cause_text = str(cause.get("cause") or "").strip()
        why_text = str(cause.get("why") or "").strip()
        checks = _unique_non_empty_strings(cause.get("checks") or [], limit=4)

        used_ids: list[str] = []
        seen_ids = set()

        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if not cid or cid not in by_id or cid in seen_ids:
                continue

            seen_ids.add(cid)
            used_ids.append(cid)
            if len(used_ids) >= 3:
                break

        if not cause_text or not why_text or not used_ids:
            continue

        grounded_causes.append(
            {
                "rank": len(grounded_causes) + 1,
                "cause": cause_text,
                "why": why_text,
                "checks": checks,
                "citations": used_ids,
            }
        )

        if len(grounded_causes) >= max_causes:
            break

    problem_summary = str(result.get("problem_summary") or "").strip()

    recommended_next_checks = _unique_non_empty_strings(
        result.get("recommended_next_checks") or [],
        limit=6,
    )

    if not recommended_next_checks:
        flattened_checks = []
        for cause in grounded_causes:
            flattened_checks.extend(cause.get("checks") or [])
        recommended_next_checks = _unique_non_empty_strings(flattened_checks, limit=6)

    grounded = {
        "problem_summary": problem_summary,
        "possible_causes": grounded_causes,
        "recommended_next_checks": recommended_next_checks,
    }

    grounded_ids = _extract_citation_ids_from_root_cause_json(grounded)
    grounded_citations = [by_id[cid] for cid in grounded_ids if cid in by_id]

    return grounded, grounded_citations

def _sanitize_citations_for_response(citations: list[dict]) -> list[dict]:
    out: list[dict] = []

    for c in citations or []:
        cid = str(c.get("citation_id") or "").strip()
        bdid = str(c.get("bubble_document_id") or "").strip()

        if not cid or not bdid:
            continue

        out.append(
            {
                "citation_id": cid,
                "bubble_document_id": bdid,
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "snippet": (c.get("snippet") or "").strip(),
            }
        )

    return out

def _build_sources_block_from_citations(
    citations: list[dict],
    *,
    max_context_chars: int = ASK_MAX_CONTEXT_CHARS,
) -> str:
    ctx_parts: List[str] = []
    total_chars = 0

    for c in citations or []:
        part = (
            f"[{c['citation_id']}] "
            f"(doc={c['bubble_document_id']}, p{c['page_from']}-{c['page_to']})\n"
            f"{(c.get('snippet') or '').strip()}\n"
        )

        if total_chars + len(part) > max_context_chars:
            break

        ctx_parts.append(part)
        total_chars += len(part)

    return "\n".join(ctx_parts).strip()

def _build_grounded_context_for_query(
    *,
    query: str,
    company_id: str,
    machine_id: str,
    bubble_document_id: Optional[str],
    document_ids: Optional[Union[List[str], str]],
    top_k: int,
    debug: bool = False,
) -> dict:

def _reorder_citations_by_priority_ids(
    citations: list[dict],
    priority_ids: list[str],
    max_items: int,
) -> list[dict]:
    if not citations:
        return []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }

    out: list[dict] = []
    used = set()

    for cid in priority_ids or []:
        cid = str(cid or "").strip()
        if not cid or cid in used or cid not in by_id:
            continue
        used.add(cid)
        out.append(by_id[cid])

    for c in citations:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        out.append(c)

    return out[:max_items]

def _score_root_cause_causal_strength(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
) -> dict:
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    txt = _normalize_unicode_advanced(chunk_text or "").lower()

    if not sig or not txt:
        return {
            "causal_strength_score": -0.25,
            "causal_strength_band": "weak",
        }

    score = 0.0

    # segnali di evidenza diretta sul gruppo/processo coinvolto
    if sig.get("strong_component_hits", 0) >= 2:
        score += 0.20
    elif sig.get("strong_component_hits", 0) == 1:
        score += 0.08

    if sig.get("process_hits", 0) >= 1:
        score += 0.16

    if sig.get("symptom_hits", 0) >= 1:
        score += 0.12

    if sig.get("section_diag_hits", 0) >= 1:
        score += 0.10
    elif sig.get("diag_hits", 0) >= 1:
        score += 0.06

    # sinergie: componente + processo/sintomo = evidenza forte
    if sig.get("strong_component_hits", 0) >= 1 and sig.get("process_hits", 0) >= 1:
        score += 0.10
    if sig.get("strong_component_hits", 0) >= 1 and sig.get("symptom_hits", 0) >= 1:
        score += 0.08

    # segnali di evidenza indiretta/collaterale
    if sig.get("lube_control_hits", 0) >= 2 and not sig.get("query_lube_related"):
        score -= 0.10

    if sig.get("startup_install_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= 0.12

    if sig.get("description_section_hit"):
        score -= 0.18

    if sig.get("overview_section_hit"):
        score -= 0.08

    # penalizza chunk molto parziali / istruzioni isolate
    partial_instruction_markers = [
        "invertire le fasi",
        "invert phases",
        "contattare immediatamente",
        "contact immediately",
        "prima accensione",
        "first start-up",
        "first startup",
        "alla prima accensione",
        "freccia",
        "arrow",
    ]
    partial_hits = sum(1 for m in partial_instruction_markers if m in txt)
    if partial_hits >= 1 and sig.get("strong_component_hits", 0) == 0:
        score -= 0.10
    if partial_hits >= 2:
        score -= 0.08

    # penalizza evidenze senza vero legame col processo o col sintomo
    if (
        sig.get("strong_component_hits", 0) == 0
        and sig.get("process_hits", 0) == 0
        and sig.get("symptom_hits", 0) == 0
    ):
        score -= 0.14

    score = max(-0.40, min(0.40, score))

    if score >= 0.18:
        band = "direct"
    elif score >= 0.02:
        band = "indirect"
    else:
        band = "collateral"

    return {
        "causal_strength_score": score,
        "causal_strength_band": band,
    }

def _root_cause_target_subsystems(
    q: str,
    inferred_components: list[str],
) -> list[str]:
    q_low = _normalize_unicode_advanced(q or "").lower()
    comps_low = " ".join(
        _normalize_unicode_advanced(str(x) or "").lower()
        for x in (inferred_components or [])
    )

    out: list[str] = []
    seen = set()

    def add(name: str):
        if not name or name in seen:
            return
        seen.add(name)
        out.append(name)

    if any(t in q_low for t in ["feed", "advance", "avanz", "wire", "filo", "strip", "nastro", "material"]):
        add("material_feed")

    if any(t in q_low for t in ["bend", "bending", "pieg", "forming", "formatura", "press", "pressa", "tool", "die", "stampo"]):
        add("forming")

    if any(t in q_low for t in ["straight", "straighten", "raddrizz"]):
        add("straightening")

    if any(t in q_low or t in comps_low for t in [
        "motor", "motore", "bearing", "cuscinet", "shaft", "albero",
        "gear", "ingran", "gearbox", "ridutt", "transmission", "trasmission",
        "brushless", "cam", "coupling", "giunto", "belt", "cinghia", "chain", "catena"
    ]):
        add("drive_train")

    if any(t in q_low for t in ["lubric", "lubrif", "oil", "olio", "grease", "grasso", "pump", "pompa"]):
        add("lubrication")

    if any(t in q_low for t in ["pneumat", "hydraul", "idraulic", "air", "aria", "pressure", "pression", "valv", "valvol", "cylind", "cilindr"]):
        add("fluid_power")

    if any(t in q_low or t in comps_low for t in ["encoder", "sensor", "sensore", "plc", "electr", "elettric", "inverter", "control", "controllo"]):
        add("electrical_control")

    if not out:
        if any(t in q_low for t in ["vibr", "oscillat", "rumor", "rumore", "noise", "rattle", "strid"]):
            add("drive_train")

    return out[:4]


def _score_root_cause_subsystem_alignment(
    q: str,
    chunk_text: str,
    target_subsystems: list[str],
) -> dict:
    txt = _normalize_unicode_advanced(chunk_text or "").lower()
    section = _normalize_unicode_advanced(_extract_section_from_text(chunk_text) or "").lower()

    if not txt:
        return {
            "subsystem_score": -0.25,
            "matched_subsystems": [],
        }

    subsystem_markers = {
        "material_feed": [
            "feed", "advance", "avanz", "trascin", "infeed", "material",
            "wire", "filo", "strip", "nastro", "roller", "rullo", "roll"
        ],
        "forming": [
            "bend", "bending", "pieg", "forming", "formatura",
            "press", "pressa", "tool", "die", "stampo", "punch", "punzone", "matrice"
        ],
        "straightening": [
            "straighten", "straightening", "raddrizz", "flatten"
        ],
        "drive_train": [
            "motor", "motore", "brushless", "shaft", "albero", "bearing", "cuscinet",
            "gear", "ingran", "gearbox", "ridutt", "transmission", "trasmission",
            "cam", "coupling", "giunto", "belt", "cinghia", "chain", "catena"
        ],
        "lubrication": [
            "lubric", "lubrif", "oil", "olio", "grease", "grasso",
            "pump", "pompa", "pressostato", "pressure switch"
        ],
        "fluid_power": [
            "pneumat", "hydraul", "idraulic", "aria", "air",
            "pressure", "pression", "valv", "valvol", "cylind", "cilindr"
        ],
        "electrical_control": [
            "encoder", "sensor", "sensore", "plc", "electr", "elettric",
            "inverter", "control", "controllo", "drive"
        ],
        "safety_installation": [
            "safety", "sicurezza", "riparo", "guard", "door", "porta",
            "microinter", "installation", "installazione", "startup", "start-up",
            "commission", "messa in servizio", "foundation", "fondazione",
            "positioning", "posizionamento", "livellamento", "piano di appoggio"
        ],
    }

    matched_subsystems: list[str] = []
    for name, markers in subsystem_markers.items():
        if any(m in txt or m in section for m in markers):
            matched_subsystems.append(name)

    target_set = {str(x).strip() for x in (target_subsystems or []) if str(x).strip()}
    matched_set = set(matched_subsystems)
    overlap = target_set & matched_set

    score = 0.0

    if overlap:
        score += min(0.34, 0.18 * len(overlap))

        if any(ts in section for ts in [
            "feed", "advance", "avanz", "bend", "pieg",
            "straight", "raddrizz", "trasmission", "transmission"
        ]):
            score += 0.06

    if matched_set and not overlap:
        support_only = matched_set <= {"lubrication", "fluid_power", "electrical_control", "safety_installation"}
        if support_only:
            score -= 0.16
        else:
            score -= 0.08

    if matched_set == {"lubrication"} and "lubrication" not in target_set:
        score -= 0.08

    if "safety_installation" in matched_set and not overlap:
        score -= 0.10

    score = max(-0.30, min(0.40, score))

    return {
        "subsystem_score": score,
        "matched_subsystems": matched_subsystems[:4],
    }

def _dedup_root_cause_candidates_semantic(
    citations: list[dict],
    max_items: int,
) -> list[dict]:
    def _sig(c: dict) -> str:
        txt = _normalize_unicode_advanced(c.get("snippet", "") or "")
        txt = re.sub(r"^SECTION:\s*[^\n]+\n?", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\s+", " ", txt).lower()
        txt = txt[:220]

        return (
            f"{str(c.get('bubble_document_id') or '').strip()}|"
            f"{int(c.get('page_from') or 0)}|"
            f"{int(c.get('page_to') or 0)}|"
            f"{txt}"
        )

    best = {}
    for c in citations or []:
        k = _sig(c)
        prev = best.get(k)

        if prev is None:
            best[k] = c
            continue

        prev_tuple = (
            float(prev.get("causal_strength_score", 0.0)),
            float(prev.get("semantic_score", 0.0)),
            float(prev.get("similarity", 0.0)),
        )
        cur_tuple = (
            float(c.get("causal_strength_score", 0.0)),
            float(c.get("semantic_score", 0.0)),
            float(c.get("similarity", 0.0)),
        )

        if cur_tuple > prev_tuple:
            best[k] = c

    out = list(best.values())
    out.sort(
        key=lambda x: (
            float(x.get("causal_strength_score", 0.0)),
            float(x.get("semantic_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )
    return out[:max_items]

def _root_cause_evidence_family_key(c: dict) -> str:
    doc_id = str(c.get("bubble_document_id") or "").strip()
    page_from = int(c.get("page_from") or 0)

    section = _extract_section_from_text(c.get("chunk_full") or c.get("snippet") or "")
    section = _normalize_unicode_advanced(section or "")
    section = re.sub(r"\s+", " ", section).strip().lower()[:120]

    return f"{doc_id}|p{page_from}|sec:{section}"


def _prioritize_root_cause_coverage(
    citations: list[dict],
    max_items: int,
) -> list[dict]:
    if not citations:
        return []

    ordered = sorted(
        citations,
        key=lambda x: (
            float(x.get("causal_strength_score", 0.0)),
            float(x.get("semantic_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )

    out: list[dict] = []
    used_ids = set()
    used_families = set()

    # primo pass: massimizza copertura famiglie diverse
    for c in ordered:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used_ids:
            continue

        fam = _root_cause_evidence_family_key(c)
        if fam in used_families:
            continue

        used_ids.add(cid)
        used_families.add(fam)
        out.append(c)

        if len(out) >= max_items:
            return out[:max_items]

    # secondo pass: riempi eventuali slot rimanenti
    for c in ordered:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used_ids:
            continue

        used_ids.add(cid)
        out.append(c)

        if len(out) >= max_items:
            break

    return out[:max_items]


def _compact_root_cause_result_citations_by_family(
    result: dict,
    citations: list[dict],
    max_per_cause: int = 2,
) -> tuple[dict, list[dict]]:
    result = dict(result or {})
    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in (citations or [])
        if c.get("citation_id")
    }

    compact_causes: list[dict] = []
    final_ids: list[str] = []
    seen_final_ids = set()

    for cause in result.get("possible_causes") or []:
        if not isinstance(cause, dict):
            continue

        kept_ids: list[str] = []
        used_families = set()

        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if not cid or cid not in by_id:
                continue

            fam = _root_cause_evidence_family_key(by_id[cid])
            if fam in used_families:
                continue

            used_families.add(fam)
            kept_ids.append(cid)

            if len(kept_ids) >= max_per_cause:
                break

        if not kept_ids:
            continue

        new_cause = dict(cause)
        new_cause["citations"] = kept_ids
        compact_causes.append(new_cause)

        for cid in kept_ids:
            if cid not in seen_final_ids:
                seen_final_ids.add(cid)
                final_ids.append(cid)

    for i, cause in enumerate(compact_causes, start=1):
        cause["rank"] = i

    result["possible_causes"] = compact_causes
    compact_citations = [by_id[cid] for cid in final_ids if cid in by_id]

    return result, compact_citations

def _root_cause_response_schema(max_causes: int) -> dict:
    max_causes = max(1, min(int(max_causes or 1), 3))

    return {
        "name": "root_cause_finder_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "problem_summary": {"type": "string"},
                "possible_causes": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rank": {"type": "integer"},
                            "cause": {"type": "string"},
                            "why": {"type": "string"},
                            "checks": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "citations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["rank", "cause", "why", "checks", "citations"],
                    },
                },
                "recommended_next_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["problem_summary", "possible_causes", "recommended_next_checks"],
        },
    }

def _draft_ps_response_schema(max_causes: int) -> dict:
    max_causes = max(1, min(int(max_causes or 1), 5))

    return {
        "name": "draft_ps_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "problem_summary": {"type": "string"},
                "possible_causes": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rank": {"type": "integer"},
                            "cause": {"type": "string"},
                            "why": {"type": "string"},
                            "checks": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "citations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["rank", "cause", "why", "checks", "citations"],
                    },
                },
            },
            "required": ["title", "problem_summary", "possible_causes"],
        },
    }

@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/version")
def version():
    return {
        "ok": True,
        "service": os.environ.get("K_SERVICE"),
        "revision": os.environ.get("K_REVISION"),
        "commit_sha": os.environ.get("COMMIT_SHA"),
    }


@app.post("/v1/ai/ingest/document")
def ingest_document(
    payload: IngestRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()
    if not (company_id and machine_id and bubble_document_id):
        raise HTTPException(status_code=400, detail="Missing company_id/machine_id/bubble_document_id")

    url = payload.file_url.strip()
    if url.startswith("//"):
        url = "https:" + url

    _db_upsert_document_file(company_id, bubble_document_id, url)

    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        data = r.content
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Fetch failed")

    content_type = (r.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    content_disposition = (r.headers.get("Content-Disposition") or "").strip()

    url_path = unquote(urlparse(url).path or "")
    detected_filename = os.path.basename(url_path) or ""
    detected_extension = os.path.splitext(detected_filename)[1].lower()

    if "filename=" in content_disposition:
        cd_filename = content_disposition.split("filename=", 1)[1].strip().strip('"').strip("'")
        if cd_filename:
            detected_filename = cd_filename
            detected_extension = os.path.splitext(detected_filename)[1].lower()

    pdf_magic = b"%PDF" in data[:1024]
    looks_like_pdf = (
        pdf_magic
        or detected_extension == ".pdf"
        or content_type == "application/pdf"
    )

    if not looks_like_pdf:
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

    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large")

    try:
        raw_pages = _extract_pages_with_layout_blocks(data)
        pages_total = len(raw_pages)

        header_norm, footer_norm = _detect_repeated_headers_footers(raw_pages)

        try:
            _db_upsert_cleaning_meta(company_id, bubble_document_id, header_norm, footer_norm)
        except Exception as e:
            print("CLEANING_META_UPSERT_FAIL", str(e))

        pages_text: list[str] = []
        pages_with_text = 0
        text_chars = 0

        for t in raw_pages:
            cleaned = _remove_headers_footers_from_page(t, header_norm, footer_norm)
            cleaned = _reflow_paragraphs_conservative(cleaned)
            cleaned = _maybe_remove_toc(cleaned)
            pages_text.append(cleaned)
            text_chars += len(cleaned)
            if len(cleaned) >= MIN_PAGE_CHARS:
                pages_with_text += 1
    except Exception:
        raise HTTPException(status_code=422, detail="PDF parse failed")

    if pages_total <= 2:
        if pages_with_text < 1 or text_chars < MIN_TEXT_CHARS_SHORT:
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
            MIN_PAGES_WITH_TEXT_ABS,
            int(math.ceil(pages_total * MIN_PAGES_WITH_TEXT_PCT)),
        )
        if text_chars < MIN_TEXT_CHARS or pages_with_text < min_pages_required:
            reason = "LOW_TEXT_CHARS" if text_chars < MIN_TEXT_CHARS else "LOW_TEXT_COVERAGE"
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

    effective_step = max(1, CHUNK_TARGET_CHARS - min(CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS - 1))
    est_chunks = int(math.ceil(max(1, text_chars) / effective_step))

    BYTES_PER_CHAR = 3
    BYTES_PER_CHUNK = 2000
    est_storage_bytes = int(text_chars * BYTES_PER_CHAR + est_chunks * BYTES_PER_CHUNK)

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
    new_total_storage = used_storage_total - prev_doc_storage + int(est_storage_bytes)

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

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_pages WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            for i, t in enumerate(pages_text, start=1):
                cur.execute(
                    """
                    INSERT INTO document_pages(company_id, machine_id, bubble_document_id, page_number, text, text_chars)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (company_id, machine_id, bubble_document_id, i, t, len(t)),
                )
        conn.commit()
    finally:
        conn.close()

    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "machinemind-ai-2a"
        location = "europe-west1"
        queue = "mm-ai-index-dev"
        service_url = os.environ.get("K_SERVICE_URL") or os.environ.get("SERVICE_URL")
        if not service_url:
            service_url = "https://mm-ai-ingest-fixed-pvgxe6eo5q-ew.a.run.app"

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(project, location, queue)

        task_payload = {
            "company_id": company_id,
            "machine_id": machine_id,
            "bubble_document_id": bubble_document_id,
            "trace_id": "ingest_auto",
        }

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{service_url}/v1/ai/index/document",
                "headers": {
                    "Content-Type": "application/json",
                    "X-AI-Internal-Secret": AI_INTERNAL_SECRET,
                },
                "body": json.dumps(task_payload).encode(),
            }
        }

        client.create_task(request={"parent": parent, "task": task})
    except Exception:
        pass

    return {
        "ok": True,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_detected": pages_total,
        "text_chars": text_chars,
        "est_storage_bytes": est_storage_bytes,
    }

@app.post("/v1/ai/ingest/source")
def ingest_structured_source(
    payload: StructuredSourceIngestRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    source_id = (payload.source_id or "").strip()

    if not (company_id and machine_id and source_id):
        raise HTTPException(status_code=400, detail="Missing company_id/machine_id/source_id")

    source_type = _normalize_structured_source_type(payload.source_type)
    source_key = _build_structured_source_key(source_type, source_id)

    text = _compose_structured_source_text(payload)
    text_chars = len(text)

    if text_chars <= 0:
        raise HTTPException(status_code=400, detail="Structured source text is empty")

    source_url = (payload.source_url or "").strip()
    if source_url.startswith("//"):
        source_url = "https:" + source_url

    est_storage_bytes = _estimate_index_storage_bytes_for_text(text_chars)

    plan_chars_limit = int(payload.plan_embed_chars_limit_total or 0)
    plan_storage_limit = int(payload.plan_index_storage_limit_bytes or 0)

    prev_usage = _db_get_index_usage(company_id=company_id, bubble_document_id=source_key)
    company_usage = _db_get_index_usage(company_id=company_id)

    new_total_chars = int(company_usage["text_chars"]) - int(prev_usage["text_chars"]) + int(text_chars)
    new_total_storage = int(company_usage["est_storage_bytes"]) - int(prev_usage["est_storage_bytes"]) + int(est_storage_bytes)

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

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.document_pages
                WHERE company_id=%s AND bubble_document_id=%s;
                """,
                (company_id, source_key),
            )

            cur.execute(
                """
                INSERT INTO public.document_pages(
                    company_id,
                    machine_id,
                    bubble_document_id,
                    page_number,
                    text,
                    text_chars
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (company_id, machine_id, source_key, 1, text, text_chars),
            )
        conn.commit()
    finally:
        conn.close()

    if source_url:
        _db_upsert_document_file(company_id, source_key, source_url)

    index_result = index_document(
        IndexDocumentRequest(
            company_id=company_id,
            machine_id=machine_id,
            bubble_document_id=source_key,
            trace_id="structured_ingest",
        ),
        x_ai_internal_secret=AI_INTERNAL_SECRET,
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
    }

@app.post("/v1/ai/index/document")
def index_document(
    payload: IndexDocumentRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()
    trace_id = (payload.trace_id or "").strip() or None

    if not (company_id and machine_id and bubble_document_id):
        raise HTTPException(status_code=400, detail="Missing company_id/machine_id/bubble_document_id")

    header_norm, footer_norm = _db_get_cleaning_meta(company_id, bubble_document_id)

    conn = _db_conn()
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

            pages = [(int(pn), txt or "") for (pn, txt) in page_rows]
            chunks = _chunk_sentences_with_pages(
                pages=pages,
                target_chars=CHUNK_TARGET_CHARS,
                overlap_chars=CHUNK_OVERLAP_CHARS,
                min_chars=CHUNK_MIN_CHARS,
            )

            source_is_structured = _is_structured_source_key(bubble_document_id)
            if source_is_structured:
                chunks = _collapse_structured_chunks(chunks)

            filtered_chunks = []
            for c in chunks:
                txt = (c.get("chunk_text") or "").strip()

                if source_is_structured:
                    if len(txt) < 20:
                        continue
                else:
                    if len(txt) < 120:
                        continue

                if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", txt):
                    continue
                if len(txt.split()) <= 4 and txt.isupper():
                    continue

                filtered_chunks.append(c)

            chunks = filtered_chunks

            for c in chunks:
                c["chunk_text"] = _strip_hf_from_chunk_text(c.get("chunk_text", ""), header_norm, footer_norm)

            chunks = [c for c in chunks if (c.get("chunk_text") or "").strip()]

            chunk_cols = _get_table_columns(cur, "document_chunks")
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
                raise HTTPException(
                    status_code=500,
                    detail=f"document_chunks missing columns: {missing}. Found: {sorted(chunk_cols)}",
                )

            cur.execute(
                "DELETE FROM document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

            insert_q = """
                INSERT INTO document_chunks(
                    company_id, machine_id, bubble_document_id,
                    chunk_index, page_from, page_to,
                    chunk_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """

            for c in chunks:
                cur.execute(
                    insert_q,
                    (
                        company_id,
                        machine_id,
                        bubble_document_id,
                        int(c["chunk_index"]),
                        int(c["page_from"]),
                        int(c["page_to"]),
                        c["chunk_text"],
                    ),
                )

            BATCH_SIZE = 32
            chunk_texts = [c["chunk_text"] for c in chunks]

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch_texts = chunk_texts[batch_start: batch_start + BATCH_SIZE]

                try:
                    vectors = _openai_embed_texts(batch_texts)
                except Exception as e:
                    for j in range(len(batch_texts)):
                        idx = batch_start + j
                        cur.execute(
                            """
                            UPDATE document_chunks
                            SET embedding_error=%s, embedding_model=%s, embedded_at=NOW()
                            WHERE company_id=%s AND bubble_document_id=%s AND chunk_index=%s;
                            """,
                            (
                                str(e),
                                OPENAI_EMBED_MODEL,
                                company_id,
                                bubble_document_id,
                                int(chunks[idx]["chunk_index"]),
                            ),
                        )
                    continue

                for j, vec in enumerate(vectors):
                    idx = batch_start + j
                    chunk_idx = int(chunks[idx]["chunk_index"])
                    vec_literal = _vector_literal(vec)

                    cur.execute(
                        """
                        UPDATE document_chunks
                        SET embedding = %s::vector,
                            embedding_model = %s,
                            embedded_at = NOW(),
                            embedding_error = NULL
                        WHERE company_id=%s AND bubble_document_id=%s AND chunk_index=%s;
                        """,
                        (vec_literal, OPENAI_EMBED_MODEL, company_id, bubble_document_id, chunk_idx),
                    )

        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "status": "indexed",
        "company_id": company_id,
        "machine_id": machine_id,
        "bubble_document_id": bubble_document_id,
        "trace_id": trace_id,
        "chunks_written": len(chunks),
        "pages_detected": len(pages),
        "chunk_target_chars": CHUNK_TARGET_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
    }


@app.post("/v1/ai/search")
def search_chunks(
    payload: SearchRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")

    company_id = (payload.company_id or "").strip()
    if not company_id:
        raise HTTPException(status_code=400, detail="Missing company_id")

    top_k = int(payload.top_k or 5)
    top_k = max(1, min(top_k, 20))

    q_vec = _openai_embed_texts([q])[0]
    q_vec_lit = _vector_literal(q_vec)

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if payload.bubble_document_id:
                bubble_document_id = payload.bubble_document_id.strip()
                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to, left(chunk_text, 400) AS preview,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
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
                ORDER BY embedding <=> %s::vector
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


@app.post("/v1/ai/ask")
def ask_v1(
    payload: AskRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")

    scope = _resolve_query_scope(
        company_id=payload.company_id,
        machine_id=payload.machine_id,
        bubble_document_id=payload.bubble_document_id,
        document_ids=payload.document_ids,
    )
    company_id = scope["company_id"]
    machine_id = scope["machine_id"]
    bubble_document_id = scope["bubble_document_id"]
    doc_ids = scope["document_ids"]

    top_k = int(payload.top_k or 5)
    top_k = max(1, min(top_k, ASK_MAX_TOP_K))
    candidate_k = max(top_k, min(40, top_k * 6))

    q_vec = _openai_embed_texts([q])[0]
    q_vec_lit = _vector_literal(q_vec)

    rows = []
    chunks_matching_filter = None
    sim_max: Optional[float] = None
    fts_used = False
    rerank_used = False
    rerank_error: Optional[str] = None

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "chunks_matching_filter": chunks_matching_filter,
                "similarity_max": sim_max,
                "fts_used": fts_used,
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": rerank_used,
                "rerank_error": rerank_error[:300] if rerank_error else None,
            }
        return resp

    chunks_matching_filter, rows = _fetch_dense_chunk_candidates(
        company_id=company_id,
        machine_id=machine_id,
        q_vec_lit=q_vec_lit,
        candidate_k=candidate_k,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=bubble_document_id,
        debug=payload.debug,
    )

    if not rows:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": "Non trovo informazioni nei documenti indicizzati per rispondere.",
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": None,
            }
        )

    sim_max = max((float(r[6]) for r in rows), default=-1.0)
    candidates = _raw_rows_to_dense_candidates(rows)

    q_low = q.lower()
    key_terms = []
    if "lubrif" in q_low or "olio" in q_low or "ingrass" in q_low or "cuscinet" in q_low or "riduttor" in q_low:
        key_terms = ["lubrif", "olio", "ingrass", "cuscinet", "riduttor"]

    if key_terms:
        gated = []
        for c in candidates:
            hay = ((c.get("chunk_full") or c.get("snippet") or "") + " " + (c.get("citation_id") or "")).lower()
            if any(t in hay for t in key_terms):
                gated.append(c)

        if len(gated) >= 2:
            candidates = gated

    cutoff_delta = 0.08 if key_terms else 0.12
    cut_candidates = [c for c in candidates if (sim_max - float(c.get("similarity", 0.0))) <= cutoff_delta]
    cut_candidates.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)

    if len(cut_candidates) < min(2, len(candidates)):
        cut_candidates = sorted(
            candidates,
            key=lambda x: float(x.get("similarity", 0.0)),
            reverse=True,
        )[: min(2, len(candidates))]

    citations = _mmr_select(q_vec, cut_candidates, top_k=top_k, lambda_mult=0.85)

    if _should_use_reranker(q=q, candidates=cut_candidates, sim_max=sim_max, top_k=top_k):
        try:
            reranked_ids = _llm_rerank_citations(q=q, candidates=cut_candidates, top_k=top_k)
            if reranked_ids:
                by_id = {str(c.get("citation_id")): c for c in cut_candidates}
                reranked = [by_id[cid] for cid in reranked_ids if cid in by_id]
                if reranked:
                    citations = reranked
                    rerank_used = True
        except Exception as e:
            rerank_error = str(e)

    for c in citations:
        c.pop("embedding_list", None)
        c.pop("chunk_full", None)

    citations = _dedup_citations_by_snippet(citations, max_items=top_k)

    if sim_max < ASK_SIM_THRESHOLD:
        fts = _fts_search_chunks(
            company_id=company_id,
            machine_id=machine_id,
            q=q,
            top_k=top_k,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
        )
        if fts:
            fts_used = True
            citations = _dedup_citations_by_snippet(citations + fts, max_items=top_k)

    if sim_max < ASK_SIM_THRESHOLD:
        picked = _pick_entity_from_citations(q, citations)
        if picked:
            value, c = picked
            answer = f"Nel documento compare questo dato: {value} [{c['citation_id']}]"

            rg_links = []
            try:
                rg_links = _build_rg_links(company_id, citations)
            except Exception as e:
                print("RG_LINKS_FAIL", str(e))
                rg_links = []

            return _finalize(
                {
                    "ok": True,
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "rg_links": rg_links,
                    "top_k": top_k,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                }
            )

        kind = None
        if _q_has_any(q, URL_HINTS):
            kind = "url"
        elif _q_has_any(q, EMAIL_HINTS):
            kind = "email"
        elif _q_has_any(q, PHONE_HINTS):
            kind = "phone"

        if kind:
            hit = _db_find_entity_chunk(
                company_id=company_id,
                machine_id=machine_id,
                kind=kind,
                doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                bubble_document_id=bubble_document_id,
            )
            if hit:
                answer = f"Nel documento compare questo dato: {hit['value']} [{hit['citation_id']}]"
                cit_list = [
                    {
                        "citation_id": hit["citation_id"],
                        "bubble_document_id": hit["bubble_document_id"],
                        "page_from": hit["page_from"],
                        "page_to": hit["page_to"],
                        "snippet": hit["snippet"],
                        "similarity": sim_max,
                    }
                ]
                rg_links = _build_rg_links(company_id, cit_list)
                return _finalize(
                    {
                        "ok": True,
                        "status": "answered",
                        "answer": answer,
                        "citations": cit_list,
                        "rg_links": rg_links,
                        "top_k": top_k,
                        "similarity_max": sim_max,
                        "chat_model": OPENAI_CHAT_MODEL,
                    }
                )

        code_tokens = _extract_code_tokens(q)
        if code_tokens:
            hit = None
            matched_token = None

            for tok in code_tokens:
                hit = _db_find_token_chunk(
                    company_id=company_id,
                    machine_id=machine_id,
                    token=tok,
                    doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                    bubble_document_id=bubble_document_id,
                )
                if hit:
                    matched_token = tok
                    break

            if hit:
                cit_list = [hit]
                rg_links = []
                try:
                    rg_links = _build_rg_links(company_id, cit_list)
                except Exception as e:
                    print("RG_LINKS_FAIL", str(e))
                    rg_links = []

                return _finalize(
                    {
                        "ok": True,
                        "status": "answered",
                        "answer": f"Nel documento compare questa stringa: {matched_token} [{hit['citation_id']}]",
                        "citations": cit_list,
                        "rg_links": rg_links,
                        "top_k": top_k,
                        "similarity_max": sim_max,
                        "chat_model": OPENAI_CHAT_MODEL,
                    }
                )

        if not (fts_used and citations):
            return _finalize(
                {
                    "ok": True,
                    "status": "no_sources",
                    "answer": "Non trovo informazioni nei documenti indicizzati per rispondere.",
                    "citations": [],
                    "rg_links": [],
                    "top_k": top_k,
                    "similarity_max": sim_max,
                }
            )

    ctx_parts: List[str] = []
    total_chars = 0
    for c in citations:
        part = f"[{c['citation_id']}] (p{c['page_from']}-{c['page_to']})\n{c['snippet']}\n"
        if total_chars + len(part) > ASK_MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(part)
        total_chars += len(part)
    sources_block = "\n".join(ctx_parts).strip()

    system_msg = (
        "Sei un assistente tecnico per aziende industriali. "
        "Devi rispondere SOLO usando le FONTI fornite. "
        "Regole obbligatorie:\n"
        "1) Se le fonti NON contengono la risposta, rispondi ESATTAMENTE: "
        "\"Non trovo informazioni nei documenti indicizzati per rispondere.\" e basta.\n"
        "2) Quando affermi qualcosa, aggiungi sempre la citazione tra parentesi quadre usando il citation_id, es: [DOCID:p1-2:c3].\n"
        "3) Non inventare, non usare conoscenza esterna.\n"
        "4) Rispondi in italiano, chiaro e conciso.\n"
    )

    user_msg = (
        f"DOMANDA:\n{q}\n\n"
        f"FONTI:\n{sources_block}\n\n"
        "ISTRUZIONE:\nRispondi alla domanda usando SOLO le fonti e inserendo citazioni [citation_id]."
    )

    try:
        answer = _openai_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        ).strip()

        answer = answer.replace("[DOC:", "[").replace("[doc:", "[")

        m = re.search(r"\[([^\]:]+):([^\]]+);([^\]]+)\]", answer)
        if m:
            docid = m.group(1).strip()
            first = m.group(2).strip()
            rest = m.group(3).strip()
            parts = [p.strip() for p in rest.split(";") if p.strip()]
            expanded = [f"[{docid}:{first}]"] + [f"[{docid}:{p}]" for p in parts]
            answer = re.sub(r"\[[^\]]+\]", " ".join(expanded), answer, count=1)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    if not answer:
        answer = "Non trovo informazioni nei documenti indicizzati per rispondere."

    if answer == "Non trovo informazioni nei documenti indicizzati per rispondere.":
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": answer,
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
            }
        )

    citations = _ground_citations_to_answer(answer, citations)

    rg_links = []
    try:
        rg_links = _build_rg_links(company_id, citations)
    except Exception as e:
        print("RG_LINKS_FAIL", str(e))
        rg_links = []

    return _finalize(
        {
            "ok": True,
            "status": "answered",
            "answer": answer,
            "citations": citations,
            "rg_links": rg_links,
            "top_k": top_k,
            "similarity_max": sim_max,
            "chat_model": OPENAI_CHAT_MODEL,
        }
    )

@app.post("/v1/ai/draft_ps")
def draft_ps_v1(
    payload: DraftPSRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")

    scope = _resolve_query_scope(
        company_id=payload.company_id,
        machine_id=payload.machine_id,
        bubble_document_id=payload.bubble_document_id,
        document_ids=payload.document_ids,
    )
    company_id = scope["company_id"]
    machine_id = scope["machine_id"]
    bubble_document_id = scope["bubble_document_id"]
    doc_ids = scope["document_ids"]

    language = (payload.language or "it").strip().lower()
    if language not in {"it", "en"}:
        language = "it"

    options = payload.options or DraftPSOptions()

    top_k = int(options.top_k or 8)
    top_k = max(3, min(top_k, 12))

    max_causes = int(options.max_causes or 3)
    max_causes = max(1, min(max_causes, 5))

    candidate_k = max(top_k, min(40, top_k * 6))

    rows = []
    chunks_matching_filter = None
    sim_max: Optional[float] = None
    fts_used = False
    rerank_used = False
    rerank_error: Optional[str] = None

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "chunks_matching_filter": chunks_matching_filter,
                "similarity_max": sim_max,
                "fts_used": fts_used,
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": rerank_used,
                "rerank_error": rerank_error[:300] if rerank_error else None,
            }
        return resp

    q_vec = _openai_embed_texts([q])[0]
    q_vec_lit = _vector_literal(q_vec)

    chunks_matching_filter, rows = _fetch_dense_chunk_candidates(
        company_id=company_id,
        machine_id=machine_id,
        q_vec_lit=q_vec_lit,
        candidate_k=candidate_k,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=bubble_document_id,
        debug=payload.debug,
    )

    if not rows:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "title": "",
                "problem_summary": "",
                "possible_causes": [],
                "citations": [],
                "rg_links": [],
                "meta": {
                    "top_k": top_k,
                    "max_causes": max_causes,
                    "similarity_max": None,
                    "language": language,
                },
            }
        )

    sim_max = max((float(r[6]) for r in rows), default=-1.0)
    candidates = _raw_rows_to_dense_candidates(rows)

    q_low = q.lower()
    key_terms = []
    if (
        "lubrif" in q_low
        or "olio" in q_low
        or "ingrass" in q_low
        or "cuscinet" in q_low
        or "riduttor" in q_low
    ):
        key_terms = ["lubrif", "olio", "ingrass", "cuscinet", "riduttor"]

    if key_terms:
        gated = []
        for c in candidates:
            hay = (
                ((c.get("chunk_full") or c.get("snippet") or "") + " " + (c.get("citation_id") or ""))
                .lower()
            )
            if any(t in hay for t in key_terms):
                gated.append(c)

        if len(gated) >= 2:
            candidates = gated

    cutoff_delta = 0.08 if key_terms else 0.12
    cut_candidates = [
        c for c in candidates
        if (sim_max - float(c.get("similarity", 0.0))) <= cutoff_delta
    ]
    cut_candidates.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)

    if len(cut_candidates) < min(2, len(candidates)):
        cut_candidates = sorted(
            candidates,
            key=lambda x: float(x.get("similarity", 0.0)),
            reverse=True,
        )[: min(2, len(candidates))]

    citations = _mmr_select(q_vec, cut_candidates, top_k=top_k, lambda_mult=0.85)

    if _should_use_reranker(q=q, candidates=cut_candidates, sim_max=sim_max, top_k=min(top_k, ASK_MAX_TOP_K)):
        try:
            reranked_ids = _llm_rerank_citations(
                q=q,
                candidates=cut_candidates,
                top_k=min(top_k, ASK_MAX_TOP_K),
            )
            if reranked_ids:
                by_id = {str(c.get("citation_id")): c for c in cut_candidates}
                reranked = [by_id[cid] for cid in reranked_ids if cid in by_id]
                if reranked:
                    citations = reranked
                    rerank_used = True
        except Exception as e:
            rerank_error = str(e)

    for c in citations:
        c.pop("embedding_list", None)
        c.pop("chunk_full", None)

    citations = _dedup_citations_by_snippet(citations, max_items=top_k)

    if sim_max < ASK_SIM_THRESHOLD:
        fts = _fts_search_chunks(
            company_id=company_id,
            machine_id=machine_id,
            q=q,
            top_k=top_k,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
        )
        if fts:
            fts_used = True
            citations = _dedup_citations_by_snippet(citations + fts, max_items=top_k)

    if sim_max < ASK_SIM_THRESHOLD and not (fts_used and citations):
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "title": "",
                "problem_summary": "",
                "possible_causes": [],
                "citations": [],
                "rg_links": [],
                "meta": {
                    "top_k": top_k,
                    "max_causes": max_causes,
                    "similarity_max": sim_max,
                    "language": language,
                },
            }
        )

    prompt_citations = _sanitize_citations_for_response(citations)
    sources_block = _build_sources_block_from_citations(prompt_citations)

    schema = _draft_ps_response_schema(max_causes=max_causes)

    if language == "en":
        system_msg = (
            "You are an industrial technical assistant specialized in drafting Problem & Solution entries. "
            "You must use ONLY the provided sources. "
            "Do not use outside knowledge. "
            "Each possible cause must have at least one real citation_id from the sources. "
            "If the sources are not sufficient, return possible_causes as an empty array. "
            "Write clearly, technically, and concretely."
        )

        user_msg = (
            f"SYMPTOM / PROBLEM:\n{q}\n\n"
            f"SOURCES:\n{sources_block}\n\n"
            "Return valid JSON for a grounded Problem & Solution draft. "
            "Use only citation_id values that appear in the sources."
        )
    else:
        system_msg = (
            "Sei un assistente tecnico industriale specializzato nella bozza di Problem & Solution. "
            "Devi usare SOLO le fonti fornite. "
            "Non usare conoscenza esterna. "
            "Ogni possibile causa deve avere almeno una citation_id reale presa dalle fonti. "
            "Se le fonti non bastano, restituisci possible_causes come array vuoto. "
            "Scrivi in modo tecnico, chiaro e concreto."
        )

        user_msg = (
            f"PROBLEMA / SINTOMO:\n{q}\n\n"
            f"FONTI:\n{sources_block}\n\n"
            "Restituisci JSON valido per una bozza grounded di Problem & Solution. "
            "Nelle citations usa solo citation_id presenti nelle fonti."
        )

    try:
        result_json = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=OPENAI_CHAT_MODEL,
            json_schema=schema,
            timeout=60,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    grounded_result, grounded_citations = _ground_root_cause_result(
        result=result_json,
        citations=citations,
        max_causes=max_causes,
    )

    grounded_result, grounded_citations = _compact_root_cause_result_citations_by_family(
        result=grounded_result,
        citations=grounded_citations,
        max_per_cause=2,
    )

    final_title = str((result_json or {}).get("title") or "").strip()
    if not final_title:
        final_title = f"P&S draft — {q[:80]}"

    final_problem_summary = (
        grounded_result.get("problem_summary")
        or str((result_json or {}).get("problem_summary") or "").strip()
        or q
    )

    if not grounded_result.get("possible_causes"):
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "title": "",
                "problem_summary": final_problem_summary,
                "possible_causes": [],
                "citations": [],
                "rg_links": [],
                "meta": {
                    "top_k": top_k,
                    "max_causes": max_causes,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                    "language": language,
                },
            }
        )

    response_citations = _sanitize_citations_for_response(grounded_citations)

    rg_links = []
    try:
        rg_links = _build_rg_links(company_id, response_citations)
    except Exception as e:
        print("RG_LINKS_FAIL", str(e))
        rg_links = []

    return _finalize(
        {
            "ok": True,
            "status": "drafted",
            "title": final_title,
            "problem_summary": final_problem_summary,
            "possible_causes": grounded_result.get("possible_causes") or [],
            "citations": response_citations,
            "rg_links": rg_links,
            "meta": {
                "top_k": top_k,
                "max_causes": max_causes,
                "similarity_max": sim_max,
                "chat_model": OPENAI_CHAT_MODEL,
                "language": language,
            },
        }
    )

@app.post("/v1/ai/root-cause")
def root_cause_v1(
    payload: RootCauseRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")

    scope = _resolve_query_scope(
        company_id=payload.company_id,
        machine_id=payload.machine_id,
        bubble_document_id=payload.bubble_document_id,
        document_ids=payload.document_ids,
    )
    company_id = scope["company_id"]
    machine_id = scope["machine_id"]
    bubble_document_id = scope["bubble_document_id"]
    doc_ids = scope["document_ids"]

    top_k = int(payload.top_k or 8)
    top_k = max(1, min(top_k, ASK_MAX_TOP_K))
    max_causes = max(1, min(int(payload.max_causes or 3), 3))
    candidate_k = max(top_k, min(80, top_k * 10))

    inferred_components = _infer_machine_components(q)
    diagnostic_queries = _build_diagnostic_queries(q, inferred_components)
    diagnostic_keywords = _collect_candidate_keywords(q, inferred_components)
    target_subsystems = _root_cause_target_subsystems(q, inferred_components)

    query_vectors: dict[str, list[float]] = {}
    query_texts = diagnostic_queries[:] if diagnostic_queries else [q]

    try:
        batch_vectors = _openai_embed_texts(query_texts)
        query_vectors = {
            text: vec
            for text, vec in zip(query_texts, batch_vectors)
            if text and vec
        }
    except Exception:
        query_vectors = {q: _openai_embed_texts([q])[0]}

    if q not in query_vectors:
        query_vectors[q] = _openai_embed_texts([q])[0]

    rows = []
    chunks_matching_filter = None
    sim_max: Optional[float] = None
    fts_used = False
    rerank_used = False
    rerank_error: Optional[str] = None
    generic_downranked_count = 0
    hard_excluded_count = 0
    evidence_matrix_used = False
    evidence_matrix: dict = {}
    semantic_score_max: Optional[float] = None
    semantic_score_avg: Optional[float] = None
    causal_strength_max: Optional[float] = None
    causal_strength_avg: Optional[float] = None
    subsystem_score_max: Optional[float] = None
    subsystem_score_avg: Optional[float] = None

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "chunks_matching_filter": chunks_matching_filter,
                "similarity_max": sim_max,
                "fts_used": fts_used,
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": rerank_used,
                "rerank_error": rerank_error[:300] if rerank_error else None,
                "diagnostic_queries": diagnostic_queries,
                "inferred_components": inferred_components,
                "diagnostic_keywords": diagnostic_keywords,
                "target_subsystems": target_subsystems,
                "generic_downranked_count": generic_downranked_count,
                "hard_excluded_count": hard_excluded_count,
                "semantic_score_max": semantic_score_max,
                "semantic_score_avg": semantic_score_avg,
                "causal_strength_max": causal_strength_max,
                "causal_strength_avg": causal_strength_avg,
                "subsystem_score_max": subsystem_score_max,
                "subsystem_score_avg": subsystem_score_avg,
                "evidence_matrix_used": evidence_matrix_used,
                "evidence_matrix_hypotheses": len(evidence_matrix.get("cause_hypotheses") or []) if isinstance(evidence_matrix, dict) else 0,
            }
        return resp

    dense_ranked_lists: list[list[dict]] = []

    for dq, dq_vec in query_vectors.items():
        q_vec_lit = _vector_literal(dq_vec)

        current_chunks_matching_filter, raw_rows = _fetch_dense_chunk_candidates(
            company_id=company_id,
            machine_id=machine_id,
            q_vec_lit=q_vec_lit,
            candidate_k=candidate_k,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
            debug=payload.debug,
        )

        if chunks_matching_filter is None:
            chunks_matching_filter = current_chunks_matching_filter

        ranked = _raw_rows_to_dense_candidates(raw_rows, query_used=dq)

        if ranked:
            dense_ranked_lists.append(ranked)

    candidates = _rrf_merge_candidates(dense_ranked_lists, k=60)

    generic_downranked_count = 0
    hard_excluded_count = 0
    rescored_candidates = []

    for c in candidates:
        chunk_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        if not chunk_text:
            continue

        if _should_hard_exclude_root_cause_chunk(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
        ):
            hard_excluded_count += 1
            continue

        cc = dict(c)

        is_generic = _should_downrank_generic_root_cause_chunk(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
        )
        cc["generic_downranked"] = bool(is_generic)

        if is_generic:
            generic_downranked_count += 1
            cc["similarity"] = max(0.0, float(cc.get("similarity", 0.0)) - 0.10)
            cc["rrf_score"] = max(0.0, float(cc.get("rrf_score", 0.0)) - 0.015)

        keyword_hits = sum(
            1
            for t in diagnostic_keywords[:12]
            if t and t in chunk_text.lower()
        )
        if keyword_hits >= 2:
            cc["similarity"] = min(1.0, float(cc.get("similarity", 0.0)) + 0.03)
            cc["rrf_score"] = float(cc.get("rrf_score", 0.0)) + 0.005

        rescored_candidates.append(cc)

    semantic_candidates = []
    semantic_scores = []
    causal_scores = []
    subsystem_scores = []

    for c in rescored_candidates:
        cc = dict(c)
        chunk_text = (cc.get("chunk_full") or cc.get("snippet") or "").strip()

        semantic = _score_root_cause_chunk_semantic(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
        )
        causal = _score_root_cause_causal_strength(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
        )
        subsystem = _score_root_cause_subsystem_alignment(
            q=q,
            chunk_text=chunk_text,
            target_subsystems=target_subsystems,
        )

        cc.update(semantic)
        cc.update(causal)
        cc.update(subsystem)

        semantic_candidates.append(cc)
        semantic_scores.append(float(cc.get("semantic_score", 0.0)))
        causal_scores.append(float(cc.get("causal_strength_score", 0.0)))
        subsystem_scores.append(float(cc.get("subsystem_score", 0.0)))

    candidates = semantic_candidates

    if semantic_scores:
        semantic_score_max = max(semantic_scores)
        semantic_score_avg = sum(semantic_scores) / len(semantic_scores)

    if causal_scores:
        causal_strength_max = max(causal_scores)
        causal_strength_avg = sum(causal_scores) / len(causal_scores)

    if subsystem_scores:
        subsystem_score_max = max(subsystem_scores)
        subsystem_score_avg = sum(subsystem_scores) / len(subsystem_scores)

    if not candidates:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "symptom": q,
                "problem_summary": "",
                "possible_causes": [],
                "recommended_next_checks": [],
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": None,
            }
        )

    sim_max = max(float(c.get("similarity", 0.0)) for c in candidates) if candidates else None

    q_low = q.lower()
    tight_cutoff = any(
        stem in q_low
        for stem in [
            "lubrif", "olio", "ingrass", "oil", "grease",
            "bearing", "cuscinet", "riduttor", "gearbox",
        ]
    )

    cutoff_delta = 0.08 if tight_cutoff else 0.12
    cut_candidates = [
        c for c in candidates
        if (float(sim_max or 0.0) - float(c.get("similarity", 0.0))) <= cutoff_delta
    ]

    sem_good = [c for c in cut_candidates if float(c.get("semantic_score", 0.0)) >= 0.02]
    if len(sem_good) >= 2:
        cut_candidates = sem_good

    causal_good = [c for c in cut_candidates if float(c.get("causal_strength_score", 0.0)) >= 0.02]
    if len(causal_good) >= 2:
        cut_candidates = causal_good

    subsystem_good = [c for c in cut_candidates if float(c.get("subsystem_score", 0.0)) >= 0.02]
    if len(subsystem_good) >= 2:
        cut_candidates = subsystem_good

    cut_candidates.sort(
        key=lambda x: (
            float(x.get("subsystem_score", 0.0)),
            float(x.get("causal_strength_score", 0.0)),
            float(x.get("semantic_score", 0.0)),
            float(x.get("rrf_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )

    if len(cut_candidates) < min(2, len(candidates)):
        cut_candidates = sorted(
            candidates,
            key=lambda x: (
                float(x.get("subsystem_score", 0.0)),
                float(x.get("causal_strength_score", 0.0)),
                float(x.get("semantic_score", 0.0)),
                float(x.get("rrf_score", 0.0)),
                float(x.get("similarity", 0.0)),
            ),
            reverse=True,
        )[: min(2, len(candidates))]

    base_q_vec = query_vectors.get(q)
    if base_q_vec is None:
        try:
            base_q_vec = _openai_embed_texts([q])[0]
        except Exception:
            base_q_vec = next(iter(query_vectors.values()))

    max_rrf = max(float(c.get("rrf_score", 0.0)) for c in cut_candidates) if cut_candidates else 0.0
    mmr_candidates = []

    for c in cut_candidates:
        cc = dict(c)
        raw_sim = float(c.get("similarity", 0.0))
        rrf = float(c.get("rrf_score", 0.0))
        rrf_norm = (rrf / max_rrf) if max_rrf > 0 else 0.0

        semantic_raw = float(c.get("semantic_score", 0.0))
        semantic_norm = max(0.0, min(1.0, (semantic_raw + 0.45) / 0.90))

        causal_raw = float(c.get("causal_strength_score", 0.0))
        causal_norm = max(0.0, min(1.0, (causal_raw + 0.40) / 0.80))

        subsystem_raw = float(c.get("subsystem_score", 0.0))
        subsystem_norm = max(0.0, min(1.0, (subsystem_raw + 0.30) / 0.70))

        cc["raw_similarity"] = raw_sim
        cc["similarity"] = (
            0.34 * raw_sim
            + 0.16 * rrf_norm
            + 0.16 * semantic_norm
            + 0.16 * causal_norm
            + 0.18 * subsystem_norm
        )
        mmr_candidates.append(cc)

    citations = _mmr_select(base_q_vec, mmr_candidates, top_k=top_k, lambda_mult=0.85)

    if _should_use_reranker(q=q, candidates=cut_candidates, sim_max=float(sim_max or 0.0), top_k=top_k):
        try:
            reranked_ids = _llm_rerank_citations(
                q=q,
                candidates=cut_candidates,
                top_k=top_k,
                diagnostic_mode=True,
            )
            if reranked_ids:
                by_id = {str(c.get("citation_id")): c for c in cut_candidates}
                reranked = [by_id[cid] for cid in reranked_ids if cid in by_id]
                if reranked:
                    citations = reranked
                    rerank_used = True
        except Exception as e:
            rerank_error = str(e)

    for c in citations:
        c.pop("embedding_list", None)
        c.pop("query_used", None)

    citations = _dedup_citations_by_snippet(citations, max_items=top_k)
    citations = _dedup_root_cause_candidates_semantic(citations, max_items=top_k)
    citations = _prioritize_root_cause_coverage(citations, max_items=top_k)

    try:
        selected_diag_ids = _llm_filter_diagnostic_chunks(
            q=q,
            candidates=cut_candidates[:18],
            max_keep=top_k,
        )
        if selected_diag_ids:
            nongeneric_ids = {
                str(c.get("citation_id") or "").strip()
                for c in citations
                if c.get("citation_id") and not bool(c.get("generic_downranked"))
            }

            priority_ids = [cid for cid in selected_diag_ids if cid in nongeneric_ids]
            if not priority_ids:
                priority_ids = selected_diag_ids

            citations = _reorder_citations_by_priority_ids(
                citations=citations,
                priority_ids=priority_ids,
                max_items=top_k,
            )
    except Exception as e:
        rerank_error = str(e) if not rerank_error else rerank_error

    citations = _prioritize_root_cause_coverage(citations, max_items=top_k)

    if float(sim_max or 0.0) < ASK_SIM_THRESHOLD:
        fts_queries = [q]
        for dq in diagnostic_queries:
            dq_low = dq.lower()
            if dq_low == q.lower():
                continue
            if dq_low.startswith(("root cause ", "causa ", "diagnosi ")):
                continue
            fts_queries.append(dq)
            if len(fts_queries) >= 4:
                break

        fts_merged: list[dict] = []

        for fts_q in fts_queries:
            fts_hits = _fts_search_chunks(
                company_id=company_id,
                machine_id=machine_id,
                q=fts_q,
                top_k=top_k,
                doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                bubble_document_id=bubble_document_id,
            )

            for c in fts_hits:
                chunk_text = (c.get("snippet") or "").strip()
                if not chunk_text:
                    continue

                if _should_hard_exclude_root_cause_chunk(
                    q=q,
                    chunk_text=chunk_text,
                    diagnostic_keywords=diagnostic_keywords,
                ):
                    continue

                if _should_downrank_generic_root_cause_chunk(
                    q=q,
                    chunk_text=chunk_text,
                    diagnostic_keywords=diagnostic_keywords,
                ):
                    c["similarity"] = max(0.0, float(c.get("similarity", 0.0)) - 0.05)

                fts_merged.append(c)

        if fts_merged:
            fts_used = True
            citations = _dedup_citations_by_snippet(citations + fts_merged, max_items=top_k)

    try:
        evidence_matrix = _llm_build_diagnostic_evidence_matrix(
            q=q,
            citations=citations,
            max_causes=max_causes,
        )

        keep_ids = [
            str(x).strip()
            for x in (evidence_matrix.get("keep_ids") or [])
            if str(x).strip()
        ]
        cause_hypotheses = evidence_matrix.get("cause_hypotheses") or []

        priority_ids: list[str] = []
        seen_priority = set()

        for cid in keep_ids:
            if cid not in seen_priority:
                seen_priority.add(cid)
                priority_ids.append(cid)

        for hyp in cause_hypotheses:
            if not isinstance(hyp, dict):
                continue
            for cid in hyp.get("evidence_ids") or []:
                cid = str(cid).strip()
                if cid and cid not in seen_priority:
                    seen_priority.add(cid)
                    priority_ids.append(cid)

        if priority_ids:
            nongeneric_ids = {
                str(c.get("citation_id") or "").strip()
                for c in citations
                if c.get("citation_id") and not bool(c.get("generic_downranked"))
            }

            filtered_priority_ids = [cid for cid in priority_ids if cid in nongeneric_ids]
            if not filtered_priority_ids:
                filtered_priority_ids = priority_ids

            citations = _reorder_citations_by_priority_ids(
                citations=citations,
                priority_ids=filtered_priority_ids,
                max_items=top_k,
            )

        evidence_matrix_used = bool(cause_hypotheses or keep_ids)
    except Exception as e:
        rerank_error = str(e) if not rerank_error else rerank_error
        evidence_matrix = {}
        evidence_matrix_used = False

    nongeneric_citations = [
        c for c in citations
        if not bool(c.get("generic_downranked"))
    ]

    if len(nongeneric_citations) >= 2:
        citations = nongeneric_citations[:top_k]

    citations = _prioritize_root_cause_coverage(citations, max_items=top_k)
    citations = _dedup_root_cause_candidates_semantic(citations, max_items=top_k)

    sem_strong = [
        c for c in citations
        if float(c.get("semantic_score", 0.0)) >= 0.02
    ]
    if len(sem_strong) >= 2:
        citations = sem_strong[:top_k]

    if float(sim_max or 0.0) < ASK_SIM_THRESHOLD and not (fts_used and citations):
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "symptom": q,
                "problem_summary": "",
                "possible_causes": [],
                "recommended_next_checks": [],
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
            }
        )

    enriched_citations: list[dict] = []
    seen_enriched = set()

    for c in citations:
        cid = str(c.get("citation_id") or "").strip()
        if not cid:
            continue

        central_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        if not central_text:
            continue

        merged = dict(c)
        merged["evidence_pack"] = f"[{cid}] {central_text[:2200]}"

        if cid not in seen_enriched:
            seen_enriched.add(cid)
            enriched_citations.append(merged)

    if enriched_citations:
        citations = enriched_citations

    ctx_parts: List[str] = []
    total_chars = 0
    for c in citations:
        evidence_text = (c.get("evidence_pack") or c.get("chunk_full") or c.get("snippet") or "").strip()
        part = (
            f"[{c['citation_id']}] "
            f"(doc={c['bubble_document_id']}, p{c['page_from']}-{c['page_to']})\n"
            f"{evidence_text}\n"
        )
        if total_chars + len(part) > ASK_MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(part)
        total_chars += len(part)

    sources_block = "\n\n".join(ctx_parts).strip()

    evidence_matrix_block = ""
    if evidence_matrix_used:
        lines = []
        for i, hyp in enumerate(evidence_matrix.get("cause_hypotheses") or [], start=1):
            cause = str(hyp.get("cause") or "").strip()
            ev_ids = [str(x).strip() for x in (hyp.get("evidence_ids") or []) if str(x).strip()]
            checks = [str(x).strip() for x in (hyp.get("check_focus") or []) if str(x).strip()]

            if not cause:
                continue

            lines.append(f"IPOTESI_{i}: {cause}")
            if ev_ids:
                lines.append("EVIDENZE: " + ", ".join(ev_ids[:3]))
            if checks:
                lines.append("VERIFICHE_FOCUS: " + " | ".join(checks[:3]))

        evidence_matrix_block = "\n".join(lines).strip()

    subsystem_focus_block = ", ".join(target_subsystems) if target_subsystems else "(non identificato)"

    schema = _root_cause_response_schema(max_causes=max_causes)

    system_msg = (
        "Sei un assistente tecnico industriale specializzato in root cause analysis. "
        "Devi ragionare come un tecnico esperto che distingue tra sintomo, possibile causa, evidenza e verifica pratica. "
        "Devi usare SOLO le FONTI fornite. "
        "Usa la MATRICE_EVIDENZE come pre-sintesi utile, ma verifica sempre coerenza con le fonti. "
        "Se più evidenze puntano alla stessa area causale, uniscile in un'unica causa e non duplicarle. "
        "Obiettivo: proporre poche cause plausibili e verifiche pratiche, senza inventare. "
        f"Regole obbligatorie:\n"
        f"1) restituisci massimo {max_causes} possibili cause;\n"
        "2) ogni causa deve avere almeno 1 citation_id preso ESATTAMENTE dalle fonti;\n"
        "3) non usare conoscenza esterna;\n"
        "4) se le fonti non bastano, restituisci possible_causes=[] e recommended_next_checks=[];\n"
        "5) le verifiche devono essere controlli operativi concreti e brevi;\n"
        "6) privilegia cause coerenti con il sintomo osservato, non manutenzione generica;\n"
        "7) non duplicare cause quasi uguali;\n"
        "8) non citare fonti inesistenti.\n"
        "9) preferisci evidenze provenienti da sezioni operative o di componente; overview, safety, installation, start-up, caratteristiche generali o acoustic sections valgono solo se il sintomo le riguarda direttamente.\n"
        "10) non proporre cause basate principalmente su sezioni generali di manutenzione, lubrificazione macchina, pneumatica, safety o overview se le fonti non collegano esplicitamente quella causa al gruppo coinvolto dal sintomo.\n"
        "11) se una fonte parla del sintomo in modo generico ma non del gruppo/processo corretto, non usarla come evidenza centrale.\n"
        "12) non proporre cause di installazione, posizionamento, fondazione, livellamento o piano di appoggio salvo che la query riguardi esplicitamente setup, installazione, messa in servizio o spostamento macchina.\n"
        "13) non usare come evidenza centrale sezioni di descrizione generale della macchina o sezioni safety/ripari/porte se non collegano esplicitamente il problema al componente o gruppo coinvolto.\n"
        "14) evita di usare più citations della stessa famiglia di evidenza per la stessa causa, salvo che siano realmente complementari.\n"
        "15) privilegia cause diverse ben supportate, non variazioni della stessa causa basate sulla stessa area di testo.\n"
        "16) privilegia cause supportate da evidenze dirette sul componente o sul processo coinvolto; le evidenze collaterali o indirette valgono meno se esistono fonti più dirette.\n"
        "17) non promuovere come causa principale istruzioni isolate di avviamento, inversione fasi, verso motore pompa o controlli secondari se non sono il supporto più diretto disponibile.\n"
        "18) privilegia cause appartenenti ai sottosistemi dominanti implicati dal sintomo.\n"
        "19) usa sottosistemi secondari solo se la causa è davvero plausibile e ben supportata da fonti specifiche.\n"
    )

    user_msg = (
        f"SINTOMO/PROBLEMA:\n{q}\n\n"
        f"SOTTOSISTEMI_PRIORITARI:\n{subsystem_focus_block}\n\n"
        f"MATRICE_EVIDENZE:\n{evidence_matrix_block or '(non disponibile)'}\n\n"
        f"FONTI:\n{sources_block}\n\n"
        "Restituisci JSON valido. Nelle citations usa solo citation_id presenti nelle fonti."
    )

    try:
        result_json = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=OPENAI_CHAT_MODEL,
            json_schema=schema,
            timeout=60,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    grounded_result, grounded_citations = _ground_root_cause_result(
        result=result_json,
        citations=citations,
        max_causes=max_causes,
    )

    grounded_result, grounded_citations = _compact_root_cause_result_citations_by_family(
        result=grounded_result,
        citations=grounded_citations,
        max_per_cause=2,
    )

    if not grounded_result.get("problem_summary"):
        grounded_result["problem_summary"] = q

    if not grounded_result.get("possible_causes"):
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "symptom": q,
                "problem_summary": grounded_result.get("problem_summary") or "",
                "possible_causes": [],
                "recommended_next_checks": [],
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
                "chat_model": OPENAI_CHAT_MODEL,
            }
        )

    response_citations = _sanitize_citations_for_response(grounded_citations)

    rg_links = []
    try:
        rg_links = _build_rg_links(company_id, response_citations)
    except Exception as e:
        print("RG_LINKS_FAIL", str(e))
        rg_links = []

    return _finalize(
        {
            "ok": True,
            "status": "answered",
            "symptom": q,
            "problem_summary": grounded_result.get("problem_summary") or q,
            "possible_causes": grounded_result.get("possible_causes") or [],
            "recommended_next_checks": grounded_result.get("recommended_next_checks") or [],
            "citations": response_citations,
            "rg_links": rg_links,
            "top_k": top_k,
            "similarity_max": sim_max,
            "chat_model": OPENAI_CHAT_MODEL,
        }
    )

@app.post("/v1/ai/delete/document")
def delete_document_v1(
    payload: DeleteDocumentRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    company_id = (payload.company_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()
    if not (company_id and bubble_document_id):
        raise HTTPException(status_code=400, detail="Missing company_id/bubble_document_id")

    deleted_chunks = 0
    deleted_pages = 0
    deleted_files = 0

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            deleted_chunks = cur.rowcount or 0

            cur.execute(
                "DELETE FROM public.document_pages WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            deleted_pages = cur.rowcount or 0

            cur.execute(
                "DELETE FROM public.document_files WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            deleted_files = cur.rowcount or 0

            cur.execute(
                "DELETE FROM public.document_cleaning_meta WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "status": "deleted",
        "company_id": company_id,
        "bubble_document_id": bubble_document_id,
        "deleted": {
            "document_chunks": int(deleted_chunks),
            "document_pages": int(deleted_pages),
            "document_files": int(deleted_files),
        },
    }