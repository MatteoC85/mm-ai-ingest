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
OPENAI_CHAT_MODEL = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-5.4-mini").strip()
OPENAI_CHAT_URL = (os.environ.get("OPENAI_CHAT_URL") or "https://api.openai.com/v1/chat/completions").strip()

# OpenAI Citation Reranker (on-demand)
OPENAI_RERANK_MODEL = (os.environ.get("OPENAI_RERANK_MODEL") or "gpt-5.4-nano").strip()
RERANK_MAX_CANDIDATES = int(os.environ.get("MM_RERANK_MAX_CANDIDATES", "18"))
RERANK_SNIPPET_CHARS = int(os.environ.get("MM_RERANK_SNIPPET_CHARS", "320"))
RERANK_TIMEOUT = int(os.environ.get("MM_RERANK_TIMEOUT_SECONDS", "30"))
RERANK_ENABLED = (os.environ.get("MM_RERANK_ENABLED") or "1").strip() == "1"
RERANK_MIN_SIM_MAX = float(os.environ.get("MM_RERANK_MIN_SIM_MAX", "0.38"))
RERANK_MAX_SIM_MAX = float(os.environ.get("MM_RERANK_MAX_SIM_MAX", "0.72"))
RERANK_MAX_SPREAD = float(os.environ.get("MM_RERANK_MAX_SPREAD", "0.10"))
RERANK_MIN_CANDIDATES = int(os.environ.get("MM_RERANK_MIN_CANDIDATES", "4"))

FINAL_CITATION_LOCK_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_DELTA", "0.028"))
FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA", "0.042"))
FINAL_CITATION_LOCK_SET_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_SET_DELTA", "0.014"))
FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA", "0.020"))
FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA", "0.016"))
ROOT_CAUSE_SET_LOCK_DELTA = float(os.environ.get("MM_ROOT_CAUSE_SET_LOCK_DELTA", "0.028"))

ASK_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SIM_THRESHOLD", "0.35"))
ASK_SHORT_QUERY_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SHORT_QUERY_SIM_THRESHOLD", "0.28"))
ASK_MAX_TOP_K = int(os.environ.get("MM_ASK_MAX_TOP_K", "8"))
ASK_SNIPPET_CHARS = int(os.environ.get("MM_ASK_SNIPPET_CHARS", "700"))
ASK_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_MAX_CONTEXT_CHARS", "9000"))
DRAFT_PS_SIM_THRESHOLD = float(os.environ.get("MM_DRAFT_PS_SIM_THRESHOLD", "0.42"))

# Shared semantic retrieval planner
SEMANTIC_QUERY_PLANNER_MODEL = (os.environ.get("MM_SEMANTIC_QUERY_PLANNER_MODEL") or "gpt-5.4-mini").strip()
SEMANTIC_QUERY_PLANNER_TIMEOUT = int(os.environ.get("MM_SEMANTIC_QUERY_PLANNER_TIMEOUT_SECONDS", "20"))
SEMANTIC_MAX_DENSE_QUERIES = int(os.environ.get("MM_SEMANTIC_MAX_DENSE_QUERIES", "5"))
SEMANTIC_MAX_LEXICAL_QUERIES = int(os.environ.get("MM_SEMANTIC_MAX_LEXICAL_QUERIES", "5"))

# Root-cause semantic intent gate
ROOT_CAUSE_INTENT_MODEL = (os.environ.get("MM_ROOT_CAUSE_INTENT_MODEL") or "gpt-5.4-mini").strip()
ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE", "0.33"))
ROOT_CAUSE_GATE_MIN_MARGIN = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_MARGIN", "0.05"))
ROOT_CAUSE_GATE_MIN_PRELIM_SIM = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_PRELIM_SIM", "0.36"))
ROOT_CAUSE_GATE_MIN_PRELIM_HITS = int(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_PRELIM_HITS", "2"))
ROOT_CAUSE_GATE_PRELIM_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_GATE_PRELIM_TOP_K", "6"))
DIAGNOSTIC_PIPELINE_ENABLED = (os.environ.get("MM_DIAGNOSTIC_PIPELINE_ENABLED") or "1").strip() == "1"
DIAGNOSTIC_EVIDENCE_MODEL = (os.environ.get("MM_DIAGNOSTIC_EVIDENCE_MODEL") or "gpt-5.4-mini").strip()
ROOT_CAUSE_RESPONSE_MODEL = (os.environ.get("MM_ROOT_CAUSE_RESPONSE_MODEL") or "gpt-5.4").strip()
ROOT_CAUSE_EXTRA_CANDIDATE_K = int(os.environ.get("MM_ROOT_CAUSE_EXTRA_CANDIDATE_K", "60"))
ROOT_CAUSE_MAX_EVIDENCE_POOL = int(os.environ.get("MM_ROOT_CAUSE_MAX_EVIDENCE_POOL", "10"))
ROOT_CAUSE_MAX_PROMPT_CITATIONS = int(os.environ.get("MM_ROOT_CAUSE_MAX_PROMPT_CITATIONS", "7"))
ROOT_CAUSE_DIRECT_SIGNAL_BONUS = float(os.environ.get("MM_ROOT_CAUSE_DIRECT_SIGNAL_BONUS", "0.12"))
ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY", "0.14"))
ROOT_CAUSE_HARD_EXCLUDE_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_HARD_EXCLUDE_PENALTY", "0.30"))
ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY", "0.22"))
ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES = int(os.environ.get("MM_ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES", "2"))
ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA = int(os.environ.get("MM_ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA", "2"))
ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL = (os.environ.get("MM_ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL") or "1").strip() != "0"

RESPONSE_ARB_ENABLED = (os.environ.get("MM_RESPONSE_ARB_ENABLED") or "1").strip() != "0"
ASK_CANDIDATE_ENABLED = (os.environ.get("MM_ASK_CANDIDATE_ENABLED") or "1").strip() != "0"
ROOT_CAUSE_CANDIDATE_ENABLED = (os.environ.get("MM_ROOT_CAUSE_CANDIDATE_ENABLED") or "1").strip() != "0"
RESPONSE_ARB_KEEP_BASELINE_ON_TIE = (os.environ.get("MM_RESPONSE_ARB_KEEP_BASELINE_ON_TIE") or "1").strip() != "0"
ASK_ARB_MIN_DELTA = float(os.environ.get("MM_ASK_ARB_MIN_DELTA", "0.035"))
ROOT_CAUSE_ARB_MIN_DELTA = float(os.environ.get("MM_ROOT_CAUSE_ARB_MIN_DELTA", "0.040"))
ROOT_CAUSE_CANDIDATE_CORE_PROMOTION = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_CORE_PROMOTION", "0.08"))
ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY", "0.16"))
ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY", "0.20"))
ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY", "0.16"))
ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY", "0.14"))
ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K", "10"))
ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K", "7"))
ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX = (os.environ.get("MM_ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX") or "1").strip() != "0"
ASK_CANDIDATE_MATRIX_TOP_K = int(os.environ.get("MM_ASK_CANDIDATE_MATRIX_TOP_K", "7"))
ASK_CANDIDATE_PROMPT_TOP_K = int(os.environ.get("MM_ASK_CANDIDATE_PROMPT_TOP_K", "6"))


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
    language: Optional[str] = None
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
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
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
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
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
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
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
            "chunk_index": int(chunk_index),
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
    q = _normalize_unicode_advanced(q or "")
    if not q.strip():
        return []

    raw = re.findall(r"\b[A-Za-z0-9_\-/]{4,}\b", q)
    out = []
    seen = set()

    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue

        has_digit = any(ch.isdigit() for ch in tok)
        has_sep = ("_" in tok) or ("-" in tok) or ("/" in tok)
        has_upper = any(ch.isupper() for ch in tok)

        if not (has_digit or has_sep or (has_upper and len(tok) >= 6)):
            continue

        key = tok.upper()
        if key in seen:
            continue

        seen.add(key)
        out.append(tok)

    return out[:5]

def _llm_classify_root_cause_query_intent(q: str) -> dict:
    schema = {
        "name": "root_cause_intent_classifier",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent_class": {
                    "type": "string",
                    "enum": [
                        "technical_fault_symptom",
                        "technical_information_question",
                        "non_technical_or_nonsense",
                        "ambiguous",
                    ],
                },
                "confidence": {
                    "type": "number",
                },
                "rationale": {
                    "type": "string",
                },
            },
            "required": ["intent_class", "confidence", "rationale"],
        },
    }

    system_msg = (
        "You classify a user query in an industrial machinery context. "
        "Work semantically, not by keyword matching. "
        "The query may be in Italian, English, or mixed language. "
        "The machinery type is unknown and can be any industrial machine. "
        "Classes:\n"
        "- technical_fault_symptom: the query expresses a fault symptom, anomaly, missing condition, malfunction, abnormal behavior, or a concise diagnostic complaint that could justify root cause analysis.\n"
        "- technical_information_question: the query is technical and relevant to machinery, but it is explanatory/informational rather than a fault symptom.\n"
        "- non_technical_or_nonsense: the query is outside the technical machinery domain, casual, or nonsense.\n"
        "- ambiguous: not enough certainty.\n"
        "Very short phrases can still be technical_fault_symptom if they express a real machine condition."
    )

    user_msg = f"QUERY:\n{q}"

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_INTENT_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=20,
        )
        if not isinstance(parsed, dict):
            return {
                "intent_class": "ambiguous",
                "confidence": 0.0,
                "rationale": "invalid classifier response",
            }

        parsed["intent_class"] = str(parsed.get("intent_class") or "ambiguous").strip()
        parsed["confidence"] = float(parsed.get("confidence") or 0.0)
        parsed["rationale"] = str(parsed.get("rationale") or "").strip()
        return parsed

    except Exception as e:
        return {
            "intent_class": "ambiguous",
            "confidence": 0.0,
            "rationale": f"classifier_error: {str(e)[:160]}",
        }

def _root_cause_preliminary_retrieval_signal(
    *,
    company_id: str,
    machine_id: str,
    q_vec: list[float],
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
) -> dict:
    if not q_vec:
        return {
            "chunks_matching_filter": None,
            "rows_found": 0,
            "similarity_max": None,
            "hits_over_prelim_threshold": 0,
            "hits_over_ask_threshold": 0,
        }

    q_vec_lit = _vector_literal(q_vec)

    chunks_matching_filter, raw_rows = _fetch_dense_chunk_candidates(
        company_id=company_id,
        machine_id=machine_id,
        q_vec_lit=q_vec_lit,
        candidate_k=max(1, ROOT_CAUSE_GATE_PRELIM_TOP_K),
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    sims = [float(r[6]) for r in raw_rows] if raw_rows else []
    sim_max = max(sims) if sims else None

    return {
        "chunks_matching_filter": chunks_matching_filter,
        "rows_found": len(raw_rows),
        "similarity_max": sim_max,
        "hits_over_prelim_threshold": sum(1 for s in sims if s >= ROOT_CAUSE_GATE_MIN_PRELIM_SIM),
        "hits_over_ask_threshold": sum(1 for s in sims if s >= ASK_SIM_THRESHOLD),
    }


def _root_cause_query_signal_summary(
    q: str,
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: Optional[str] = None,
    doc_ids: Optional[list[str]] = None,
    debug: bool = False,
) -> dict:
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    q_low = q_norm.lower()

    tokens = re.findall(r"[a-zà-öø-ÿ0-9]{2,}", q_low)
    code_hits = len(_extract_code_tokens(q_norm))

    classifier_used = True
    classified = _llm_classify_root_cause_query_intent(q_norm)
    intent_class = str(classified.get("intent_class") or "ambiguous").strip()
    intent_confidence = float(classified.get("confidence") or 0.0)
    intent_rationale = str(classified.get("rationale") or "").strip()

    q_vec = _openai_embed_texts([q_norm])[0] if q_norm else []

    preliminary = _root_cause_preliminary_retrieval_signal(
        company_id=company_id,
        machine_id=machine_id,
        q_vec=q_vec,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    return {
        "query_norm": q_norm,
        "token_count": len(tokens),
        "code_hits": code_hits,
        "query_vector": q_vec,
        "preliminary_retrieval": preliminary,
        "intent_class": intent_class,
        "intent_confidence": intent_confidence,
        "intent_rationale": intent_rationale,
        "classifier_used": classifier_used,
    }

def _should_fail_closed_root_cause_query(signal_summary: dict) -> bool:
    if not signal_summary:
        return True

    token_count = int(signal_summary.get("token_count", 0) or 0)
    intent_class = str(signal_summary.get("intent_class") or "ambiguous").strip()
    intent_confidence = float(signal_summary.get("intent_confidence", 0.0) or 0.0)

    prelim = signal_summary.get("preliminary_retrieval") or {}
    prelim_sim_max = prelim.get("similarity_max")
    prelim_hits = int(prelim.get("hits_over_prelim_threshold", 0) or 0)

    if token_count <= 0:
        return True

    if intent_class == "technical_fault_symptom":
        return False

    if intent_class == "non_technical_or_nonsense":
        return True

    strong_preliminary_signal = (
        prelim_sim_max is not None
        and float(prelim_sim_max) >= ROOT_CAUSE_GATE_MIN_PRELIM_SIM + 0.04
        and prelim_hits >= max(1, ROOT_CAUSE_GATE_MIN_PRELIM_HITS)
    )

    if intent_class == "technical_information_question":
        return not strong_preliminary_signal

    # ambiguous
    if intent_confidence < 0.80 and strong_preliminary_signal:
        return False

    return True

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
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[DIAGNOSTIC_EVIDENCE_MODEL, ROOT_CAUSE_INTENT_MODEL, OPENAI_CHAT_MODEL],
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
            -float(x.get("rrf_score", 0.0)),
            -float(x.get("similarity", 0.0)),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        ),
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


def _dedup_text_values(values: list[str], limit: Optional[int] = None) -> list[str]:
    out: list[str] = []
    seen = set()

    for value in values or []:
        s = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(value or ""))).strip()
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(s)

        if limit is not None and len(out) >= limit:
            break

    return out


def _count_query_tokens(q: str) -> int:
    return len(re.findall(r"[a-zà-öø-ÿ0-9]{2,}", _normalize_unicode_advanced(q or "").lower()))


def _simple_query_language(q: str) -> str:
    toks = re.findall(r"[a-zà-öø-ÿ']{2,}", _normalize_unicode_advanced(q or "").lower())
    if not toks:
        return "it"

    it_markers = {
        "il", "lo", "la", "gli", "le", "di", "del", "della", "dei", "delle", "con",
        "per", "quando", "durante", "mentre", "dopo", "prima", "non", "si", "una", "un",
    }
    en_markers = {
        "the", "with", "for", "when", "during", "while", "after", "before", "not",
        "does", "is", "are", "can", "cannot", "won't", "will", "a", "an",
    }

    it_hits = sum(1 for t in toks if t in it_markers)
    en_hits = sum(1 for t in toks if t in en_markers)

    if en_hits > it_hits:
        return "en"
    return "it"


def _select_response_language(
    q: str,
    planner: Optional[dict] = None,
    preferred: Optional[str] = None,
) -> str:
    pref = str(preferred or "").strip().lower()
    if pref in {"it", "en"}:
        return pref

    if isinstance(planner, dict):
        lang = str(planner.get("query_language") or "").strip().lower()
        if lang in {"it", "en"}:
            return lang

    return _simple_query_language(q)


def _localized_no_sources(language: str) -> str:
    return (
        "I cannot find enough information in the indexed documents to answer."
        if str(language or "").lower() == "en"
        else "Non trovo informazioni sufficienti nei documenti indicizzati per rispondere."
    )


def _localized_value_answer(language: str, value: str, citation_id: str) -> str:
    if str(language or "").lower() == "en":
        return f"The document contains this value: {value} [{citation_id}]"
    return f"Nel documento compare questo dato: {value} [{citation_id}]"


def _localized_token_answer(language: str, token: str, citation_id: str) -> str:
    if str(language or "").lower() == "en":
        return f"The document contains this string: {token} [{citation_id}]"
    return f"Nel documento compare questa stringa: {token} [{citation_id}]"



def _ask_response_schema() -> dict:
    return {
        "name": "ask_grounded_answer_v3",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_status": {
                    "type": "string",
                    "enum": ["answered", "no_sources"],
                },
                "grounded_points": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            },
                        },
                        "required": ["text", "citation_ids"],
                    },
                },
            },
            "required": ["answer_status", "grounded_points"],
        },
    }

def _render_grounded_answer_points(
    grounded_points: list[dict],
    citations: list[dict],
    *,
    max_points: int = 3,
) -> tuple[str, list[dict]]:
    if not grounded_points:
        return "", []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations or []
        if c.get("citation_id")
    }

    parts: list[str] = []
    used_ids: list[str] = []
    seen_ids = set()

    for point in grounded_points[:max_points]:
        if not isinstance(point, dict):
            continue

        text = re.sub(r"\s+", " ", str(point.get("text") or "")).strip()
        if not text:
            continue

        cids = []
        for cid in point.get("citation_ids") or []:
            cid = str(cid or "").strip()
            if not cid or cid not in by_id:
                continue
            cids.append(cid)
            if cid not in seen_ids:
                seen_ids.add(cid)
                used_ids.append(cid)

        if not cids:
            continue

        text = re.sub(r"\s*\[[^\]]+\]\s*$", "", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        parts.append(f"{text} {' '.join(f'[{cid}]' for cid in cids)}")

    final_citations = [by_id[cid] for cid in used_ids if cid in by_id]
    return " ".join(parts).strip(), final_citations


def _language_marker_score(text: str, language: str) -> int:
    toks = re.findall(r"[a-zà-öø-ÿ']{2,}", _normalize_unicode_advanced(text or "").lower())
    if not toks:
        return 0

    markers = {
        "it": {
            "il", "lo", "la", "gli", "le", "di", "del", "della", "dei", "delle",
            "con", "per", "quando", "durante", "mentre", "dopo", "prima", "non", "si",
            "una", "un", "può", "puo", "quindi", "documenti",
        },
        "en": {
            "the", "with", "for", "when", "during", "while", "after", "before", "not",
            "does", "is", "are", "can", "cannot", "will", "would", "should", "documents",
            "this", "that", "these", "those", "mode",
        },
    }
    target = markers.get(str(language or "").lower(), set())
    return sum(1 for t in toks if t in target)


def _looks_like_target_language(text: str, target_language: str) -> bool:
    target_language = str(target_language or "").lower()
    if target_language not in {"it", "en"}:
        return True

    other = "en" if target_language == "it" else "it"
    target_score = _language_marker_score(text, target_language)
    other_score = _language_marker_score(text, other)

    if target_score == 0 and other_score == 0:
        return True
    return target_score >= other_score


def _translation_response_schema() -> dict:
    return {
        "name": "translation_preserving_citations_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    }


def _translate_text_preserving_citations(text: str, target_language: str) -> str:
    text = str(text or "").strip()
    target_language = str(target_language or "").strip().lower()
    if not text or target_language not in {"it", "en"}:
        return text

    system_msg = (
        "Translate the text into the requested target language while preserving every citation token "
        "like [DOC:p1-2:c3] exactly as-is. Do not add or remove content."
    )
    user_msg = (
        f"TARGET_LANGUAGE: {target_language}\n\n"
        f"TEXT:\n{text}"
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=OPENAI_CHAT_MODEL,
            json_schema=_translation_response_schema(),
            timeout=30,
        )
        translated = re.sub(r"\s+", " ", str((parsed or {}).get("text") or "")).strip()
        return translated or text
    except Exception:
        return text




def _query_translation_schema() -> dict:
    return {
        "name": "query_translation_for_retrieval_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    }


def _translate_query_for_retrieval(text: str, target_language: str) -> str:
    text = re.sub(r"\s+", " ", _normalize_unicode_advanced(text or "")).strip()
    target_language = str(target_language or "").strip().lower()
    if not text or target_language not in {"it", "en"}:
        return text

    system_msg = (
        "Translate the user's technical query for document retrieval. "
        "Preserve meaning exactly, keep it short, do not add explanations, "
        "do not assume a domain, and preserve codes, identifiers, and proper names exactly. "
        "Prefer natural industrial wording. For symptoms, use physically plausible verbs like "
        "'vibrates', 'makes noise', 'jams', 'stops', 'does not start', 'automatic mode', "
        "'vibra', 'fa rumore', 'si blocca', 'si inceppa', 'non parte', 'in automatico'. "
        "Avoid software-like or colloquial mistranslations such as 'freezes' for a machine stop."
    )
    user_msg = (
        f"TARGET_LANGUAGE: {target_language}\n\n"
        f"QUERY:\n{text}"
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=SEMANTIC_QUERY_PLANNER_MODEL,
            json_schema=_query_translation_schema(),
            timeout=20,
        )
        translated = re.sub(r"\s+", " ", str((parsed or {}).get("text") or "")).strip()
        return translated or text
    except Exception:
        return text


def _query_symptom_profile(q: str) -> dict:
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip().lower()
    tokens = re.findall(r"[a-zà-öø-ÿ0-9]{2,}", q_norm)

    classes: list[str] = []
    if any(st in q_norm for st in ["vibr", "oscillat", "oscillaz"]):
        classes.append("vibration")
    if any(st in q_norm for st in ["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"]):
        classes.append("noise")
    if any(st in q_norm for st in ["jam", "block", "stuck", "bloc", "blocc", "ferma", "arrest", "incepp", "impunt"]):
        classes.append("jam")
    if (
        any(st in q_norm for st in ["non parte", "non si avvia", "does not start", "won't start", "will not start", "cannot start", "can't start", "doesnt start"])
        or ("start" in q_norm and any(st in q_norm for st in [" non ", " not ", "won't", "will not", "cannot", "can't"]))
    ):
        classes.append("no_start")

    has_bending_anchor = any(st in q_norm for st in ["bend", "bending", "pieg", "forming", "formatura", "press", "pressa", "tool", "die", "stampo"])
    has_feed_anchor = any(st in q_norm for st in ["feed", "advance", "advancement", "avanz", "wire", "filo", "strip", "nastro", "material"])
    has_process_anchor = has_bending_anchor or has_feed_anchor or any(st in q_norm for st in ["cut", "cutting", "taglio", "drill", "fora", "weld", "sald", "straighten", "raddrizz"])
    automatic_mode = any(st in q_norm for st in ["automatic", "automatico", "ciclo automatico", "automatic cycle", "auto mode", "modalità automatica"])

    has_support_anchor = any(
        st in q_norm
        for st in [
            "lubric", "lubrif", "oil", "olio", "grease", "grasso",
            "electr", "elettric", "phase", "fasi", "voltage", "tensione",
            "pneumat", "hydraul", "idraulic", "pressure", "pression",
            "safety", "sicurezza", "door", "porta", "interlock", "microinter",
            "plc", "encoder", "sensor", "sensore", "panel", "quadro"
        ]
    )

    generic_symptom = bool(classes) and not has_process_anchor and not has_support_anchor and len(tokens) <= 6
    if "no_start" in classes and automatic_mode:
        generic_symptom = False

    return {
        "classes": _dedup_text_values(classes, limit=4),
        "has_bending_anchor": has_bending_anchor,
        "has_feed_anchor": has_feed_anchor,
        "has_process_anchor": has_process_anchor,
        "has_support_anchor": has_support_anchor,
        "automatic_mode": automatic_mode,
        "generic_symptom": generic_symptom,
    }


def _symptom_crosslingual_expansions(q: str, source_language: str) -> list[str]:
    if not ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL:
        return []

    source_language = str(source_language or "").strip().lower()
    if source_language not in {"it", "en"}:
        return []

    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])
    out: list[str] = []

    def add(x: str):
        x = re.sub(r"\s+", " ", str(x or "").strip())
        if x:
            out.append(x)

    if source_language == "it":
        if "vibration" in classes:
            if profile.get("has_bending_anchor"):
                add("the machine vibrates during bending")
            elif profile.get("has_feed_anchor"):
                add("the machine vibrates during feed")
            else:
                add("the machine vibrates")
        if "noise" in classes:
            if profile.get("has_bending_anchor"):
                add("the machine makes noise during bending")
            elif profile.get("has_feed_anchor"):
                add("the machine makes noise during feed")
            else:
                add("the machine makes noise")
        if "jam" in classes:
            if profile.get("has_feed_anchor"):
                add("the machine jams during feed")
                add("the machine stops during feed")
            else:
                add("the machine jams")
                add("the machine stops unexpectedly")
        if "no_start" in classes:
            if profile.get("automatic_mode"):
                add("the machine does not start in automatic mode")
                add("the machine does not start in automatic cycle")
            else:
                add("the machine does not start")
    else:
        if "vibration" in classes:
            if profile.get("has_bending_anchor"):
                add("la macchina vibra durante la piegatura")
            elif profile.get("has_feed_anchor"):
                add("la macchina vibra durante l'avanzamento")
            else:
                add("la macchina vibra")
        if "noise" in classes:
            if profile.get("has_bending_anchor"):
                add("la macchina fa rumore durante la piegatura")
            elif profile.get("has_feed_anchor"):
                add("la macchina fa rumore durante l'avanzamento")
            else:
                add("la macchina fa rumore")
        if "jam" in classes:
            if profile.get("has_feed_anchor"):
                add("la macchina si inceppa durante l'avanzamento")
                add("la macchina si blocca durante l'avanzamento")
            else:
                add("la macchina si blocca")
                add("la macchina si inceppa")
        if "no_start" in classes:
            if profile.get("automatic_mode"):
                add("la macchina non parte in automatico")
                add("la macchina non si avvia in ciclo automatico")
            else:
                add("la macchina non parte")
                add("la macchina non si avvia")

    return _dedup_text_values(out, limit=3)


def _augment_crosslingual_query_plan(q: str, planner: Optional[dict]) -> dict:
    planner = dict(planner or {})
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    query_language = str(planner.get("query_language") or _simple_query_language(q_norm)).strip().lower()

    planner["crosslingual_dense_queries"] = []
    planner["crosslingual_lexical_queries"] = []

    if not q_norm or query_language not in {"it", "en"}:
        return planner

    target_language = "en" if query_language == "it" else "it"
    base_text = re.sub(r"\s+", " ", str(planner.get("normalized_query") or q_norm)).strip() or q_norm
    translated = _translate_query_for_retrieval(base_text, target_language)
    deterministic = _symptom_crosslingual_expansions(base_text, query_language)

    cross_texts = []
    if translated and translated.lower() != base_text.lower():
        cross_texts.append(translated)
    cross_texts.extend(deterministic)

    planner["crosslingual_dense_queries"] = _dedup_text_values(cross_texts, limit=3)
    planner["crosslingual_lexical_queries"] = _dedup_text_values(cross_texts, limit=3)

    return planner


def _ask_rescue_response_schema() -> dict:
    return {
        "name": "ask_grounded_answer_rescue_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grounded_points": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            },
                        },
                        "required": ["text", "citation_ids"],
                    },
                },
            },
            "required": ["grounded_points"],
        },
    }


def _extractive_fallback_answer(
    citations: list[dict],
    response_language: str,
    *,
    max_points: int = 2,
) -> tuple[str, list[dict]]:
    if not citations:
        return "", []

    parts: list[str] = []
    used: list[dict] = []

    for c in citations[:max_points]:
        body = re.sub(r"^SECTION:\s*[^\n]+\n?", "", (c.get("chunk_full") or c.get("snippet") or ""), flags=re.IGNORECASE).strip()
        if not body:
            continue

        sentence = ""
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        if lines:
            sentence = lines[0]
        if not sentence:
            match = re.split(r"(?<=[\.!?])\s+", body)
            sentence = (match[0] if match else body).strip()

        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue

        if sentence[-1] not in ".!?":
            sentence += "."
        parts.append(f"{sentence} [{c['citation_id']}]")
        used.append(c)

    answer = " ".join(parts).strip()
    if answer and not _looks_like_target_language(answer, response_language):
        answer = _translate_text_preserving_citations(answer, response_language)

    return answer, used


def _enrich_ask_prompt_citations(
    company_id: str,
    citations: list[dict],
    *,
    max_manual_expansions: int = 2,
    radius: int = 1,
) -> list[dict]:
    out: list[dict] = []
    manual_done = 0

    for c in citations or []:
        cc = dict(c)
        source_type = _source_type_from_document_id(cc.get("bubble_document_id") or "")
        if source_type == "manual" and manual_done < max_manual_expansions:
            try:
                neighbors = _expand_with_neighbor_chunks(
                    company_id=company_id,
                    bubble_document_id=str(cc.get("bubble_document_id") or ""),
                    citation_ids=[str(cc.get("citation_id") or "")],
                    radius=radius,
                )
            except Exception:
                neighbors = []

            texts: list[str] = []
            seen = set()
            for n in neighbors:
                txt = re.sub(r"\s+", " ", (n.get("chunk_full") or n.get("snippet") or "").strip())
                if not txt:
                    continue
                if txt in seen:
                    continue
                seen.add(txt)
                texts.append(txt)

            if texts:
                cc["chunk_full"] = "\n".join(texts)[:2200]
                if not cc.get("snippet"):
                    cc["snippet"] = texts[0][:ASK_SNIPPET_CHARS]

            manual_done += 1

        out.append(cc)

    return out


def _generate_ask_grounded_points(
    *,
    q: str,
    planner: dict,
    response_language: str,
    company_id: str,
    citations: list[dict],
    allow_no_sources: bool,
) -> tuple[str, list[dict]]:
    if not citations:
        return "no_sources", []

    prompt_citations = _enrich_ask_prompt_citations(
        company_id=company_id,
        citations=citations,
        max_manual_expansions=2,
        radius=1,
    )
    sources_block = _build_sources_block_from_citations(
        prompt_citations,
        max_context_chars=ASK_MAX_CONTEXT_CHARS,
        prefer_chunk_full=True,
    )

    if allow_no_sources:
        system_msg = (
            "You are a technical documentation assistant for machinery and industrial equipment. "
            "Use ONLY the provided sources. "
            "Procedures and problem-solution entries are valid evidence when directly relevant, "
            "but a generic procedure or generic P&S must not outweigh a more specific manual passage. "
            "Prefer the most specific grounded evidence available. "
            "When multiple evidence sets are close in quality, prefer the dominant grounded evidence family rather than mixing weak alternatives. "
            "Never use outside knowledge. "
            "Always reply in the requested response language. "
            "If the sources do not directly answer the question but they still contain closely relevant evidence, "
            "return answered with cautious grounded points that explicitly say the documents do or do not state something directly. "
            "Use no_sources only when the sources are genuinely not helpful. "
            "Return 1 to 3 short grounded points only. "
            "Each point must be directly supported by its citation_ids."
        )
        schema = _ask_response_schema()
    else:
        system_msg = (
            "You are a technical documentation assistant for machinery and industrial equipment. "
            "Use ONLY the provided sources. "
            "Always reply in the requested response language. "
            "Return 1 to 3 very short grounded points. "
            "You MUST return grounded_points, even if the evidence is partial. "
            "If the documents do not state the requested thing directly, say that explicitly and then report the closest grounded evidence. "
            "Do not return no_sources. "
            "Each point must be directly supported by citation_ids from the sources."
        )
        schema = _ask_rescue_response_schema()

    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"NORMALIZED_QUESTION:\n{planner.get('normalized_query') or q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        "Return valid JSON. Use only citation ids present in the sources."
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=OPENAI_CHAT_MODEL,
            json_schema=schema,
            timeout=60,
        )
    except Exception:
        return "no_sources", []

    if allow_no_sources:
        answer_status = str((parsed or {}).get("answer_status") or "").strip().lower()
        grounded_points = list((parsed or {}).get("grounded_points") or [])
        return (answer_status or "no_sources"), grounded_points

    grounded_points = list((parsed or {}).get("grounded_points") or [])
    return ("answered" if grounded_points else "no_sources"), grounded_points

def _ground_citations_to_ids(citation_ids: list[str], citations: list[dict]) -> list[dict]:
    if not citation_ids or not citations:
        return []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }

    out: list[dict] = []
    seen = set()

    for cid in citation_ids:
        cid = str(cid or "").strip()
        if not cid or cid in seen or cid not in by_id:
            continue

        seen.add(cid)
        out.append(by_id[cid])

    return out


def _semantic_query_plan(q: str, *, mode: str = "ask") -> dict:
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    fallback_style = "telegraphic" if _count_query_tokens(q_norm) <= 6 else "natural"
    fallback = {
        "normalized_query": q_norm,
        "dense_queries": [q_norm] if q_norm else [],
        "lexical_queries": [q_norm] if q_norm else [],
        "query_style": fallback_style,
        "query_language": _simple_query_language(q_norm),
    }

    if not q_norm:
        return fallback

    schema = {
        "name": "semantic_query_plan_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "normalized_query": {"type": "string"},
                "dense_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(1, SEMANTIC_MAX_DENSE_QUERIES),
                },
                "lexical_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(1, SEMANTIC_MAX_LEXICAL_QUERIES),
                },
                "query_style": {
                    "type": "string",
                    "enum": [
                        "telegraphic",
                        "natural",
                        "identifier_lookup",
                        "contact_lookup",
                    ],
                },
                "query_language": {
                    "type": "string",
                    "enum": ["it", "en", "mixed", "other"],
                },
            },
            "required": [
                "normalized_query",
                "dense_queries",
                "lexical_queries",
                "query_style",
                "query_language",
            ],
        },
    }

    system_msg = (
        "You prepare retrieval plans for a technical documentation assistant that must work across different machine sectors. "
        "Queries and documents may be in Italian or English. "
        "Work semantically and domain-agnostically. "
        "Preserve the user's meaning exactly. "
        "Do not inject unsupported components, causes, sectors, or jargon. "
        "Produce a small set of retrieval-ready rewrites: "
        "dense_queries for semantic embedding recall, lexical_queries for keyword/FTS rescue. "
        "You may include one careful translation between Italian and English if it improves mixed-language recall, "
        "but do not broaden the meaning. "
        "query_style should describe the surface form of the query, not the machine domain."
    )

    user_msg = (
        f"MODE: {mode}\n"
        f"QUERY:\n{q_norm}"
    )

    try:
        parsed = _openai_chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=SEMANTIC_QUERY_PLANNER_MODEL,
            json_schema=schema,
            timeout=SEMANTIC_QUERY_PLANNER_TIMEOUT,
        )
        if not isinstance(parsed, dict):
            return fallback

        normalized_query = re.sub(r"\s+", " ", str(parsed.get("normalized_query") or q_norm)).strip() or q_norm
        dense_queries = _dedup_text_values(
            [q_norm, normalized_query] + list(parsed.get("dense_queries") or []),
            limit=max(2, SEMANTIC_MAX_DENSE_QUERIES + 1),
        )
        lexical_queries = _dedup_text_values(
            [q_norm, normalized_query] + list(parsed.get("lexical_queries") or []),
            limit=max(2, SEMANTIC_MAX_LEXICAL_QUERIES + 1),
        )
        query_style = str(parsed.get("query_style") or fallback_style).strip().lower()
        if query_style not in {"telegraphic", "natural", "identifier_lookup", "contact_lookup"}:
            query_style = fallback_style

        query_language = str(parsed.get("query_language") or fallback["query_language"]).strip().lower()
        if query_language not in {"it", "en", "mixed", "other"}:
            query_language = fallback["query_language"]

        return {
            "normalized_query": normalized_query,
            "dense_queries": dense_queries or [q_norm],
            "lexical_queries": lexical_queries or [q_norm],
            "query_style": query_style,
            "query_language": query_language,
        }
    except Exception:
        return fallback


def _effective_similarity_threshold(
    q: str,
    *,
    planner: Optional[dict] = None,
    base_threshold: float = ASK_SIM_THRESHOLD,
) -> float:
    token_count = _count_query_tokens(q)
    style = str((planner or {}).get("query_style") or "").strip().lower()

    if style in {"telegraphic", "identifier_lookup", "contact_lookup"} or token_count <= 5:
        return min(base_threshold, ASK_SHORT_QUERY_SIM_THRESHOLD)

    return base_threshold


def _build_prefix_tsquery_from_texts(texts: list[str], limit: int = 10) -> Optional[str]:
    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from",
        "this", "that", "these", "those", "into", "onto", "about", "question",
        "machine", "system", "document", "documents", "manual", "answer", "issue", "problem",
        "il", "lo", "la", "i", "gli", "le", "con", "per", "quando", "durante", "dopo", "prima",
        "questo", "questa", "questi", "queste", "domanda", "documento", "documenti",
        "macchina", "sistema", "problema", "guasto", "risposta",
    }

    toks: list[str] = []
    seen = set()

    for text in texts or []:
        for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", _normalize_unicode_advanced(text or "").lower()):
            if tok in stopwords:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            toks.append(tok)
            if len(toks) >= limit:
                break
        if len(toks) >= limit:
            break

    if not toks:
        return None

    return " | ".join(f"{tok}:*" for tok in toks)


def _fts_search_chunks_prefix(
    company_id: str,
    machine_id: str,
    texts: list[str],
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> list[dict]:
    ts_query = _build_prefix_tsquery_from_texts(texts, limit=10)
    if not ts_query:
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
                           to_tsquery('simple', %s)
                       ) AS rank
                FROM public.document_chunks
                WHERE {where_sql}
                  AND to_tsvector('simple', chunk_text) @@ to_tsquery('simple', %s)
                ORDER BY rank DESC, bubble_document_id, page_from, chunk_index
                LIMIT %s;
                """,
                [ASK_SNIPPET_CHARS, ts_query, *params, ts_query, top_k],
            )
            rows = cur.fetchall()

        out: list[dict] = []
        for (bdid, chunk_index, page_from, page_to, snippet, _rank) in rows:
            citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
            out.append(
                {
                    "citation_id": citation_id,
                    "bubble_document_id": str(bdid),
                    "chunk_index": int(chunk_index),
                    "page_from": int(page_from),
                    "page_to": int(page_to),
                    "snippet": (snippet or "").strip(),
                    "similarity": 0.0,
                }
            )
        return out
    finally:
        conn.close()


def _fts_search_chunks_multi(
    company_id: str,
    machine_id: str,
    queries: list[str],
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> list[dict]:
    merged: list[dict] = []

    for q in _dedup_text_values(queries, limit=max(1, SEMANTIC_MAX_LEXICAL_QUERIES + 1)):
        merged.extend(
            _fts_search_chunks(
                company_id=company_id,
                machine_id=machine_id,
                q=q,
                top_k=top_k,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
            )
        )

    return _dedup_citations_by_snippet(merged, max_items=top_k)


def _dense_candidates_multi_query(
    *,
    query_texts: list[str],
    company_id: str,
    machine_id: str,
    candidate_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
) -> tuple[Optional[int], list[dict], dict[str, list[float]]]:
    cleaned_queries = _dedup_text_values(query_texts, limit=max(1, SEMANTIC_MAX_DENSE_QUERIES + 2))
    if not cleaned_queries:
        return None, [], {}

    vectors = _openai_embed_texts(cleaned_queries)
    query_vectors: dict[str, list[float]] = {}
    dense_ranked_lists: list[list[dict]] = []
    chunks_matching_filter = None

    for qq, vec in zip(cleaned_queries, vectors):
        query_vectors[qq] = vec
        q_vec_lit = _vector_literal(vec)

        current_chunks_matching_filter, raw_rows = _fetch_dense_chunk_candidates(
            company_id=company_id,
            machine_id=machine_id,
            q_vec_lit=q_vec_lit,
            candidate_k=candidate_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
            debug=debug,
        )

        if chunks_matching_filter is None:
            chunks_matching_filter = current_chunks_matching_filter

        ranked = _raw_rows_to_dense_candidates(raw_rows, query_used=qq)
        if ranked:
            dense_ranked_lists.append(ranked)

    merged = _rrf_merge_candidates(dense_ranked_lists, k=60)
    return chunks_matching_filter, merged, query_vectors


def _candidate_order_key(item: dict) -> tuple:
    return (
        -float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0),
        -float(item.get("similarity", 0.0) or 0.0),
        -float(item.get("rrf_score", 0.0) or 0.0),
        str(item.get("bubble_document_id") or ""),
        int(item.get("page_from") or 0),
        int(item.get("page_to") or 0),
        int(item.get("chunk_index") or 0),
        str(item.get("citation_id") or ""),
    )


def _source_type_from_document_id(value: str) -> str:
    v = str(value or "").strip().lower()
    if not v or ":" not in v:
        return "manual"
    prefix = v.split(":", 1)[0].strip()
    if prefix in STRUCTURED_SOURCE_TYPES:
        return prefix
    return "manual"


def _content_term_set(text: str, limit: int = 80) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from", "into", "onto",
        "this", "that", "these", "those", "question", "answer", "issue", "problem", "machine", "system",
        "document", "documents", "manual", "procedure", "step", "solution",
        "il", "lo", "la", "i", "gli", "le", "con", "per", "quando", "durante", "mentre", "dopo", "prima",
        "questo", "questa", "questi", "queste", "domanda", "risposta", "problema", "macchina", "sistema",
        "procedura", "step", "soluzione", "documenti", "documento",
    }
    out = []
    seen = set()
    for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", _normalize_unicode_advanced(text or "").lower()):
        if tok in stopwords:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return set(out)


def _planner_query_term_set(q: str, planner: Optional[dict]) -> set[str]:
    texts = [q]
    if isinstance(planner, dict):
        texts.append(str(planner.get("normalized_query") or ""))
        texts.extend(list(planner.get("lexical_queries") or []))
    return _content_term_set("\n".join(texts), limit=60)


def _term_overlap_score(query_terms: set[str], text_terms: set[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    inter = len(query_terms & text_terms)
    if inter <= 0:
        return 0.0
    return inter / math.sqrt(float(len(query_terms)) * float(len(text_terms)))


def _candidate_specificity_score(item: dict) -> float:
    text = (item.get("chunk_full") or item.get("snippet") or "").strip()
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    text_terms = _content_term_set(text, limit=100)
    char_len = len(text)
    line_count = len([ln for ln in text.split("\n") if ln.strip()])

    score = 0.0
    score += min(0.07, 0.0018 * len(text_terms))

    if 160 <= char_len <= 1800:
        score += 0.04
    elif char_len < 90:
        score -= 0.08
    elif char_len < 160:
        score -= 0.04

    if line_count >= 2:
        score += 0.02

    if source_type == "ps" and len(text_terms) < 20:
        score -= 0.08
    elif source_type in {"procedure", "step"} and len(text_terms) < 18:
        score -= 0.05

    return max(-0.14, min(0.14, score))




def _stable_evidence_family_key(item: dict) -> str:
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    doc_id = str(item.get("bubble_document_id") or "").strip()
    page_from = int(item.get("page_from") or 0)

    section = _extract_section_from_text(item.get("chunk_full") or item.get("snippet") or "")
    section = _normalize_unicode_advanced(section or "")
    section = re.sub(r"\s+", " ", section).strip().lower()[:120]

    if section:
        return f"{source_type}|{doc_id}|sec:{section}"

    if source_type == "manual":
        return f"{source_type}|{doc_id}|p{page_from}"

    return f"{source_type}|{doc_id}|p{page_from}"


def _stable_evidence_set_key(item: dict) -> str:
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    doc_id = str(item.get("bubble_document_id") or "").strip()

    section = _extract_section_from_text(item.get("chunk_full") or item.get("snippet") or "")
    section = _normalize_unicode_advanced(section or "")
    section = re.sub(r"\s+", " ", section).strip().lower()[:80]

    if source_type in {"ps", "procedure", "step"}:
        return f"{source_type}|{doc_id}"

    if section:
        return f"{source_type}|{doc_id}|sec:{section}"

    return f"{source_type}|{doc_id}"


def _locked_family_score(bundle: dict) -> float:
    members = list(bundle.get("members") or [])
    if not members:
        return -1.0

    best = float(members[0].get("retrieval_score", members[0].get("similarity", 0.0)) or 0.0)
    second = float(members[1].get("retrieval_score", members[1].get("similarity", 0.0)) or 0.0) if len(members) >= 2 else 0.0
    max_overlap = max((float(m.get("overlap_score", 0.0)) for m in members), default=0.0)
    max_specificity = max((float(m.get("specificity_score", 0.0)) for m in members), default=0.0)
    selected_count = int(bundle.get("selected_count") or 0)
    source_type = str(bundle.get("source_type") or "manual")
    member_count = len(members)

    score = (
        0.55 * best
        + 0.10 * second
        + 0.14 * max_overlap
        + 0.11 * max_specificity
        + 0.04 * min(selected_count, 2)
        + 0.03 * min(member_count, 3)
    )

    if source_type == "manual" and (max_overlap >= 0.08 or max_specificity >= 0.04):
        score += 0.04

    if source_type in {"ps", "procedure", "step"} and max_overlap < 0.22 and max_specificity < 0.05:
        score -= 0.08

    return score


def _locked_member_order_key(item: dict, selected_ids: set[str]) -> tuple:
    cid = str(item.get("citation_id") or "").strip()
    return (
        0 if cid in selected_ids else 1,
        -float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0),
        -float(item.get("overlap_score", 0.0) or 0.0),
        -float(item.get("specificity_score", 0.0) or 0.0),
        -float(item.get("similarity", 0.0) or 0.0),
        str(item.get("bubble_document_id") or ""),
        int(item.get("page_from") or 0),
        int(item.get("page_to") or 0),
        int(item.get("chunk_index") or 0),
        cid,
    )


def _locked_set_score(set_row: dict) -> float:
    families = list(set_row.get("families") or [])
    if not families:
        return -1.0

    top = float(families[0].get("family_score", 0.0))
    second = float(families[1].get("family_score", 0.0)) if len(families) >= 2 else 0.0
    manual_bonus = 0.05 if any(str(f.get("source_type") or "") == "manual" for f in families[:2]) else 0.0
    overlap_bonus = max((float(f.get("max_overlap", 0.0)) for f in families), default=0.0) * 0.10
    specificity_bonus = max((float(f.get("max_specificity", 0.0)) for f in families), default=0.0) * 0.08
    generic_penalty = 0.07 if all(bool(f.get("generic_structured")) for f in families[:2]) else 0.0

    return (
        0.74 * top
        + 0.18 * second
        + 0.03 * min(len(families), 3)
        + manual_bonus
        + overlap_bonus
        + specificity_bonus
        - generic_penalty
    )


def _family_row_sort_key(row: dict) -> tuple:
    return (
        -float(row.get("family_score", 0.0)),
        -float(row.get("best_score", 0.0)),
        0 if str(row.get("source_type") or "") == "manual" else 1,
        str(row.get("family_key") or ""),
    )


def _set_row_sort_key(row: dict) -> tuple:
    return (
        -float(row.get("set_score", 0.0)),
        0 if bool(row.get("has_manual")) else 1,
        -float(row.get("best_family_score", 0.0)),
        str(row.get("set_key") or ""),
    )


def _lock_final_citations(
    *,
    selected_citations: list[dict],
    ranked_candidates: list[dict],
    top_k: int,
    diagnostic_mode: bool = False,
    query_token_count: int = 0,
) -> list[dict]:
    selected_citations = list(selected_citations or [])
    ranked_candidates = list(ranked_candidates or [])

    if not selected_citations and not ranked_candidates:
        return []

    selected_ids = {
        str(c.get("citation_id") or "").strip()
        for c in selected_citations
        if c.get("citation_id")
    }

    ordered_ranked = sorted(ranked_candidates, key=_candidate_order_key)

    by_id: dict[str, dict] = {}
    for item in ordered_ranked + selected_citations:
        cid = str(item.get("citation_id") or "").strip()
        if not cid:
            continue
        prev = by_id.get(cid)
        if prev is None:
            by_id[cid] = dict(item)
            continue
        prev_score = float(prev.get("retrieval_score", prev.get("similarity", 0.0)) or 0.0)
        cur_score = float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0)
        if cur_score > prev_score:
            merged = dict(prev)
            merged.update(item)
            by_id[cid] = merged

    families: dict[str, dict] = {}
    ordered_items = [by_id[str(c.get("citation_id") or "").strip()] for c in selected_citations if str(c.get("citation_id") or "").strip() in by_id]
    ordered_items += [by_id[cid] for cid in by_id if cid not in selected_ids]

    for item in ordered_items:
        cid = str(item.get("citation_id") or "").strip()
        if not cid:
            continue
        fam = _stable_evidence_family_key(item)
        bundle = families.setdefault(
            fam,
            {
                "family_key": fam,
                "set_key": _stable_evidence_set_key(item),
                "source_type": _source_type_from_document_id(item.get("bubble_document_id") or ""),
                "members": [],
                "selected_count": 0,
            },
        )
        bundle["members"].append(item)
        if cid in selected_ids:
            bundle["selected_count"] += 1

    if not families:
        return _dedup_citations_by_snippet(selected_citations, max_items=top_k)

    rows = []
    for fam, bundle in families.items():
        members = sorted(bundle["members"], key=lambda x: _locked_member_order_key(x, selected_ids))
        bundle["members"] = members
        row = dict(bundle)
        row["family_score"] = _locked_family_score(bundle)
        row["best_score"] = float(members[0].get("retrieval_score", members[0].get("similarity", 0.0)) or 0.0)
        row["max_overlap"] = max((float(m.get("overlap_score", 0.0)) for m in members), default=0.0)
        row["max_specificity"] = max((float(m.get("specificity_score", 0.0)) for m in members), default=0.0)
        row["generic_structured"] = (
            row["source_type"] in {"ps", "procedure", "step"}
            and row["max_overlap"] < 0.22
            and row["max_specificity"] < 0.05
        )
        rows.append(row)

    rows.sort(key=_family_row_sort_key)

    set_rows_map: dict[str, dict] = {}
    for row in rows:
        set_key = str(row.get("set_key") or "")
        srow = set_rows_map.setdefault(
            set_key,
            {
                "set_key": set_key,
                "families": [],
                "has_manual": False,
            },
        )
        srow["families"].append(row)
        if str(row.get("source_type") or "") == "manual":
            srow["has_manual"] = True

    set_rows = []
    for set_key, srow in set_rows_map.items():
        srow["families"] = sorted(srow["families"], key=_family_row_sort_key)
        srow["best_family_score"] = float(srow["families"][0].get("family_score", 0.0))
        srow["set_score"] = _locked_set_score(srow)
        set_rows.append(srow)

    set_rows.sort(key=_set_row_sort_key)
    if not set_rows:
        return _dedup_citations_by_snippet(selected_citations, max_items=top_k)

    keep_sets = [set_rows[0]]
    set_delta = FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA if diagnostic_mode else FINAL_CITATION_LOCK_SET_DELTA

    if len(set_rows) >= 2:
        top_set = set_rows[0]
        second_set = set_rows[1]
        gap = float(top_set.get("set_score", 0.0)) - float(second_set.get("set_score", 0.0))
        top_generic = all(bool(f.get("generic_structured")) for f in top_set.get("families")[:2])

        if top_generic and gap <= (set_delta + 0.012):
            keep_sets = [second_set]
            if diagnostic_mode and len(set_rows) >= 3:
                third_set = set_rows[2]
                if float(second_set.get("set_score", 0.0)) - float(third_set.get("set_score", 0.0)) <= set_delta:
                    keep_sets.append(third_set)
        elif diagnostic_mode and gap <= set_delta and not all(bool(f.get("generic_structured")) for f in second_set.get("families")[:2]):
            keep_sets.append(second_set)

    dominant_rows: list[dict] = []
    family_delta = FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA

    for set_row in keep_sets:
        fams = list(set_row.get("families") or [])
        if not fams:
            continue

        dominant_rows.append(fams[0])

        if diagnostic_mode:
            for fam in fams[1:]:
                gap = float(fams[0].get("family_score", 0.0)) - float(fam.get("family_score", 0.0))
                same_set_count = len([r for r in dominant_rows if r.get("set_key") == set_row.get("set_key")])
                if gap <= family_delta and same_set_count < 2:
                    dominant_rows.append(fam)

    dominant_rows = sorted(
        {str(r.get("family_key") or ""): r for r in dominant_rows if r.get("family_key")}.values(),
        key=_family_row_sort_key,
    )

    quotas = [2, 1, 1] if diagnostic_mode else [2]
    if dominant_rows and str(dominant_rows[0].get("source_type") or "") == "manual" and query_token_count >= 4:
        quotas[0] = min(3, top_k)

    locked: list[dict] = []
    used_ids: set[str] = set()

    for idx, row in enumerate(dominant_rows):
        quota = quotas[min(idx, len(quotas) - 1)]
        family_added = 0
        for member in row.get("members") or []:
            cid = str(member.get("citation_id") or "").strip()
            if not cid or cid in used_ids:
                continue
            locked.append(member)
            used_ids.add(cid)
            family_added += 1
            if family_added >= quota:
                break
        if len(locked) >= top_k:
            break

    if len(locked) < top_k:
        for row in dominant_rows:
            for member in row.get("members") or []:
                cid = str(member.get("citation_id") or "").strip()
                if not cid or cid in used_ids:
                    continue
                locked.append(member)
                used_ids.add(cid)
                if len(locked) >= top_k:
                    break
            if len(locked) >= top_k:
                break

    if len(locked) < top_k:
        top_set_key = str(keep_sets[0].get("set_key") or "") if keep_sets else ""
        for row in rows:
            if top_set_key and str(row.get("set_key") or "") != top_set_key:
                continue
            for member in row.get("members") or []:
                cid = str(member.get("citation_id") or "").strip()
                if not cid or cid in used_ids:
                    continue
                locked.append(member)
                used_ids.add(cid)
                if len(locked) >= top_k:
                    break
            if len(locked) >= top_k:
                break

    return _dedup_citations_by_snippet(locked, max_items=top_k)


def _candidate_source_bias(
    item: dict,
    query_terms: set[str],
    *,
    query_style: str = "",
    query_token_count: int = 0,
) -> tuple[float, dict]:
    text = (item.get("chunk_full") or item.get("snippet") or "").strip()
    text_terms = _content_term_set(text, limit=100)
    overlap = _term_overlap_score(query_terms, text_terms)
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")

    bias = 0.0

    if source_type == "manual":
        if overlap >= 0.10 and len(text_terms) >= 16:
            bias += 0.04
        if query_token_count >= 4 and overlap >= 0.08 and len(text_terms) >= 14:
            bias += 0.03

    elif source_type == "ps":
        if query_token_count >= 4:
            if len(text_terms) <= 26 and overlap < 0.24:
                bias -= 0.16
            elif len(text_terms) <= 36 and overlap < 0.20:
                bias -= 0.10
        else:
            if len(text_terms) <= 18 and overlap < 0.22:
                bias -= 0.14
            elif len(text_terms) <= 28 and overlap < 0.18:
                bias -= 0.08
        if overlap >= 0.28 and len(text_terms) >= 16:
            bias += 0.02

    elif source_type in {"procedure", "step"}:
        if query_token_count >= 4:
            if len(text_terms) <= 26 and overlap < 0.22:
                bias -= 0.11
            elif len(text_terms) <= 34 and overlap < 0.18:
                bias -= 0.07
        else:
            if len(text_terms) <= 18 and overlap < 0.18:
                bias -= 0.08
        if overlap >= 0.26:
            bias += 0.03

    if query_style == "natural" and source_type in {"ps", "procedure", "step"} and len(text_terms) <= 24:
        bias -= 0.04

    return max(-0.18, min(0.10, bias)), {
        "source_type": source_type,
        "overlap_score": overlap,
        "content_term_count": len(text_terms),
    }


def _rebalance_selected_citations(
    selected_citations: list[dict],
    ranked_candidates: list[dict],
    top_k: int,
    *,
    query_style: str = "",
    query_token_count: int = 0,
) -> list[dict]:
    if not selected_citations:
        return []

    ordered_ranked = sorted(ranked_candidates or [], key=_candidate_order_key)
    out = _dedup_citations_by_snippet(selected_citations, max_items=top_k)
    selected_ids = {str(c.get("citation_id") or "").strip() for c in out if c.get("citation_id")}

    def source_type(c: dict) -> str:
        return str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or ""))

    top_score = float(out[0].get("retrieval_score", out[0].get("similarity", 0.0)) or 0.0)
    manual_selected = any(source_type(c) == "manual" for c in out)

    if query_token_count >= 4 and not manual_selected:
        for cand in ordered_ranked:
            cid = str(cand.get("citation_id") or "").strip()
            if not cid or cid in selected_ids:
                continue
            if source_type(cand) != "manual":
                continue
            cand_score = float(cand.get("retrieval_score", cand.get("similarity", 0.0)) or 0.0)
            if cand_score >= top_score - 0.14:
                out = [cand] + out
                out = _dedup_citations_by_snippet(out, max_items=top_k)
                selected_ids = {str(c.get("citation_id") or "").strip() for c in out if c.get("citation_id")}
                break

    generic_structured = [
        c for c in out
        if source_type(c) in {"ps", "procedure", "step"}
        and float(c.get("overlap_score", 0.0)) < 0.20
        and float(c.get("specificity_score", 0.0)) < 0.04
    ]

    if query_token_count >= 4 and len(generic_structured) > 1:
        keep_first = True
        replacements: list[dict] = []
        for c in out:
            if c not in generic_structured:
                replacements.append(c)
                continue
            if keep_first:
                replacements.append(c)
                keep_first = False

        existing_ids = {str(c.get("citation_id") or "").strip() for c in replacements if c.get("citation_id")}
        for cand in ordered_ranked:
            cid = str(cand.get("citation_id") or "").strip()
            if not cid or cid in existing_ids:
                continue
            if source_type(cand) != "manual":
                continue
            cand_score = float(cand.get("retrieval_score", cand.get("similarity", 0.0)) or 0.0)
            if cand_score >= top_score - 0.16:
                replacements.append(cand)
                existing_ids.add(cid)
            if len(replacements) >= top_k:
                break

        out = _dedup_citations_by_snippet(replacements, max_items=top_k)

    if query_style == "natural" and query_token_count >= 4:
        reordered: list[dict] = []
        manuals = [c for c in out if source_type(c) == "manual"]
        others = [c for c in out if source_type(c) != "manual"]
        if manuals:
            reordered.extend(manuals[: max(1, min(len(manuals), top_k))])
            for c in others:
                if len(reordered) >= top_k:
                    break
                reordered.append(c)
            out = reordered[:top_k]

    return out[:top_k]



def _retrieval_quality_score(retrieval: dict) -> float:
    citations = list((retrieval or {}).get("citations") or [])
    if not citations:
        return -1.0

    sim_max = float((retrieval or {}).get("similarity_max") or 0.0)
    manual_count = sum(1 for c in citations if _source_type_from_document_id(c.get("bubble_document_id") or "") == "manual")
    max_overlap = max((float(c.get("overlap_score", 0.0)) for c in citations), default=0.0)
    max_specificity = max((float(c.get("specificity_score", 0.0)) for c in citations), default=0.0)
    generic_structured = sum(
        1
        for c in citations
        if _source_type_from_document_id(c.get("bubble_document_id") or "") in {"ps", "procedure", "step"}
        and float(c.get("overlap_score", 0.0)) < 0.20
        and float(c.get("specificity_score", 0.0)) < 0.04
    )

    return (
        0.34 * sim_max
        + (0.22 if manual_count > 0 else 0.0)
        + 0.24 * max_overlap
        + 0.16 * max_specificity
        - 0.07 * generic_structured
    )


def _shared_semantic_retrieval(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "ask",
    base_threshold: float = ASK_SIM_THRESHOLD,
    diagnostic_mode: bool = False,
) -> dict:
    planner = _semantic_query_plan(q, mode=planner_mode)
    planner = _augment_crosslingual_query_plan(q, planner)

    dense_queries = _dedup_text_values(
        [q, planner.get("normalized_query")]
        + list(planner.get("dense_queries") or [])
        + list(planner.get("crosslingual_dense_queries") or []),
        limit=max(3, SEMANTIC_MAX_DENSE_QUERIES + 2),
    )
    lexical_queries = _dedup_text_values(
        [q, planner.get("normalized_query")]
        + list(planner.get("lexical_queries") or [])
        + list(planner.get("crosslingual_lexical_queries") or []),
        limit=max(3, SEMANTIC_MAX_LEXICAL_QUERIES + 2),
    )

    chunks_matching_filter, candidates, query_vectors = _dense_candidates_multi_query(
        query_texts=dense_queries,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    sim_max = max((float(c.get("similarity", 0.0)) for c in candidates), default=None) if candidates else None
    effective_threshold = _effective_similarity_threshold(
        q,
        planner=planner,
        base_threshold=base_threshold,
    )

    rerank_used = False
    rerank_error: Optional[str] = None
    fts_used = False
    prefix_fts_used = False
    exact_fts_used = False

    selected_citations: list[dict] = []
    cut_candidates: list[dict] = []
    query_terms = _planner_query_term_set(q, planner)
    query_style = str((planner or {}).get("query_style") or "").strip().lower()
    query_token_count = _count_query_tokens(q)

    if candidates:
        max_rrf = max(float(c.get("rrf_score", 0.0)) for c in candidates) if candidates else 0.0
        prepared: list[dict] = []

        for c in candidates:
            cc = dict(c)
            rrf_norm = (float(cc.get("rrf_score", 0.0)) / max_rrf) if max_rrf > 0 else 0.0
            source_bias, source_meta = _candidate_source_bias(
                cc,
                query_terms,
                query_style=query_style,
                query_token_count=query_token_count,
            )
            specificity_score = _candidate_specificity_score(cc)
            overlap_score = float(source_meta.get("overlap_score", 0.0))

            cc.update(source_meta)
            cc["specificity_score"] = specificity_score
            cc["source_bias"] = source_bias
            cc["retrieval_score"] = (
                0.58 * float(cc.get("similarity", 0.0))
                + 0.17 * rrf_norm
                + 0.13 * overlap_score
                + specificity_score
                + source_bias
            )
            prepared.append(cc)

        prepared.sort(key=_candidate_order_key)

        sim_delta = 0.15
        if sim_max is not None:
            if sim_max >= 0.55:
                sim_delta = 0.10
            elif sim_max >= 0.40:
                sim_delta = 0.12

        cut_candidates = []
        for c in prepared:
            if sim_max is None:
                cut_candidates.append(c)
                continue
            if (float(sim_max) - float(c.get("similarity", 0.0))) <= sim_delta:
                cut_candidates.append(c)

        min_keep = min(len(prepared), max(4, top_k))
        if len(cut_candidates) < min_keep:
            cut_candidates = prepared[:min(len(prepared), max(top_k * 3, 12))]
        else:
            cut_candidates = cut_candidates[:min(len(cut_candidates), max(top_k * 3, 14))]

        q_vec = (
            query_vectors.get(q)
            or query_vectors.get(str(planner.get("normalized_query") or ""))
            or next(iter(query_vectors.values()), [])
        )

        selected_citations = _mmr_select(
            q_vec,
            cut_candidates,
            top_k=top_k,
            lambda_mult=0.88 if diagnostic_mode else 0.86,
        )

        if sim_max is not None and _should_use_reranker(q=q, candidates=cut_candidates, sim_max=float(sim_max), top_k=top_k):
            try:
                reranked_ids = _llm_rerank_citations(
                    q=q,
                    candidates=cut_candidates,
                    top_k=top_k,
                    diagnostic_mode=diagnostic_mode,
                )
                if reranked_ids:
                    by_id = {str(c.get("citation_id") or "").strip(): c for c in cut_candidates}
                    reranked = [by_id[cid] for cid in reranked_ids if cid in by_id]
                    if reranked:
                        selected_citations = reranked
                        rerank_used = True
            except Exception as e:
                rerank_error = str(e)

        selected_citations = _dedup_citations_by_snippet(selected_citations, max_items=top_k)
        selected_citations = _rebalance_selected_citations(
            selected_citations=selected_citations,
            ranked_candidates=cut_candidates,
            top_k=top_k,
            query_style=query_style,
            query_token_count=query_token_count,
        )
        selected_citations = _lock_final_citations(
            selected_citations=selected_citations,
            ranked_candidates=cut_candidates,
            top_k=top_k,
            diagnostic_mode=diagnostic_mode,
            query_token_count=query_token_count,
        )

    if (sim_max is None or float(sim_max) < effective_threshold) or not selected_citations:
        prefix_hits = _fts_search_chunks_prefix(
            company_id=company_id,
            machine_id=machine_id,
            texts=lexical_queries,
            top_k=top_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if prefix_hits:
            fts_used = True
            prefix_fts_used = True
            selected_citations = _dedup_citations_by_snippet(selected_citations + prefix_hits, max_items=top_k)

        exact_hits = _fts_search_chunks_multi(
            company_id=company_id,
            machine_id=machine_id,
            queries=lexical_queries,
            top_k=top_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if exact_hits:
            fts_used = True
            exact_fts_used = True
            selected_citations = _dedup_citations_by_snippet(selected_citations + exact_hits, max_items=top_k)

    if selected_citations:
        selected_citations = _lock_final_citations(
            selected_citations=selected_citations,
            ranked_candidates=list(cut_candidates or []) + list(selected_citations or []),
            top_k=top_k,
            diagnostic_mode=diagnostic_mode,
            query_token_count=query_token_count,
        )

    return {
        "planner": planner,
        "dense_queries": dense_queries,
        "lexical_queries": lexical_queries,
        "chunks_matching_filter": chunks_matching_filter,
        "similarity_max": sim_max,
        "effective_threshold": effective_threshold,
        "candidates": cut_candidates,
        "citations": selected_citations,
        "fts_used": fts_used,
        "prefix_fts_used": prefix_fts_used,
        "exact_fts_used": exact_fts_used,
        "rerank_used": rerank_used,
        "rerank_error": rerank_error[:300] if rerank_error else None,
    }

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
    out.sort(
        key=lambda x: (
            -float(x.get("similarity", 0.0)),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )
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
                ORDER BY rank DESC, bubble_document_id, page_from, chunk_index
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
                        "chunk_index": int(chunk_index),
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


def _openai_chat(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    payload = {
        "model": (model or OPENAI_CHAT_MODEL),
        "messages": messages,
        "temperature": temperature,
    }
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


def _normalize_model_candidates(models: Optional[list[str]]) -> list[str]:
    out: list[str] = []
    seen = set()

    for model_name in models or []:
        model_name = str(model_name or "").strip()
        if not model_name:
            continue
        if model_name in seen:
            continue
        seen.add(model_name)
        out.append(model_name)

    return out


def _openai_chat_json_models(
    messages: list[dict],
    *,
    models: Optional[list[str]] = None,
    json_schema: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    tried = _normalize_model_candidates(models) or [OPENAI_CHAT_MODEL]
    last_error: Optional[Exception] = None

    for model_name in tried:
        try:
            return _openai_chat_json(
                messages,
                model=model_name,
                json_schema=json_schema,
                timeout=timeout,
            )
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error

    raise Exception("No model candidates available for JSON chat call")


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
            "causal_strength_score": round(float(c.get("causal_strength_score", 0.0)), 4),
            "semantic_score": round(float(c.get("semantic_score", 0.0)), 4),
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
        "12) Per sintomi generici come vibrazione, rumore o blocco, non privilegiare lubrificazione, start-up, installazione o sicurezza se il testo non collega esplicitamente quel sottosistema al sintomo.\n"
        "13) Per il mancato avvio, i blocchi elettrici, interlock e consensi sono più forti di una nota generica di lubrificazione.\n"
        "14) Favorisci le evidenze con causal_strength_score e semantic_score più alti.\n"
    )

    user_msg = (
        f"PROBLEMA:\n{q}\n\n"
        f"CANDIDATI:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        f"Restituisci JSON con gli id delle fonti più utili alla diagnosi."
    )

    parsed = _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
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
                "causal_strength_score": round(float(c.get("causal_strength_score", 0.0)), 4),
                "semantic_score": round(float(c.get("semantic_score", 0.0)), 4),
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
        "13) Per sintomi generici come vibrazione, rumore o blocco, le fonti di lubrificazione, start-up, installazione o sicurezza non devono diventare ipotesi centrali senza un legame esplicito col sintomo.\n"
        "14) Per il mancato avvio, preferisci cause elettriche/interlock/consensi rispetto a note generiche di lubrificazione.\n"
        "15) Le evidenze con causal_strength_score e semantic_score più alti hanno priorità.\n"
    )

    user_msg = (
        f"SINTOMO/PROBLEMA:\n{q}\n\n"
        "CITAZIONI_CANDIDATE_JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Restituisci JSON valido."
    )

    parsed = _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
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



def _root_cause_label_canonicalization_schema(max_causes: int) -> dict:
    max_causes = max(1, min(int(max_causes or 1), 5))
    return {
        "name": "root_cause_label_canonicalization_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "labels": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rank": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["rank", "label"],
                    },
                },
            },
            "required": ["labels"],
        },
    }


def _canonicalize_root_cause_labels(
    result: dict,
    citations: list[dict],
    *,
    language: str,
) -> dict:
    result = dict(result or {})
    possible_causes = list(result.get("possible_causes") or [])
    if not possible_causes:
        return result

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in (citations or [])
        if c.get("citation_id")
    }

    items = []
    for cause in possible_causes:
        if not isinstance(cause, dict):
            continue
        evidence = []
        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if not cid or cid not in by_id:
                continue
            c = by_id[cid]
            snippet = re.sub(r"\s+", " ", (c.get("snippet") or c.get("chunk_full") or "")).strip()
            evidence.append({
                "citation_id": cid,
                "snippet": snippet[:220],
            })
            if len(evidence) >= 2:
                break

        items.append(
            {
                "rank": int(cause.get("rank") or 0),
                "current_label": str(cause.get("cause") or "").strip(),
                "why": str(cause.get("why") or "").strip(),
                "evidence": evidence,
            }
        )

    if not items:
        return result

    system_msg = (
        "Normalize root-cause labels into short canonical technical labels. "
        "Reuse source terminology whenever possible. "
        "Do not broaden or narrow the meaning. "
        "Keep each label to about 3 to 8 words, noun-phrase style, with no trailing period. "
        "Use the requested language."
    )
    user_msg = (
        f"LANGUAGE: {language}\n\n"
        f"CAUSE_ITEMS_JSON:\n{json.dumps(items, ensure_ascii=False)}"
    )

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_RESPONSE_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_root_cause_label_canonicalization_schema(len(items)),
            timeout=40,
        )
    except Exception:
        return result

    labels = {
        int(x.get("rank") or 0): re.sub(r"\s+", " ", str(x.get("label") or "")).strip().rstrip(".")
        for x in (parsed or {}).get("labels") or []
        if isinstance(x, dict)
    }

    out_causes = []
    for cause in possible_causes:
        rank = int(cause.get("rank") or 0)
        label = labels.get(rank)
        new_cause = dict(cause)
        if label:
            new_cause["cause"] = label
        out_causes.append(new_cause)

    result["possible_causes"] = out_causes
    return result

def _normalized_cause_label_key(label: str) -> str:
    s = _normalize_unicode_advanced(label or "").lower()
    s = re.sub(r"[^a-zà-öø-ÿ0-9\s\-_/]", " ", s)
    s = re.sub(r"\b(?:the|a|an|il|lo|la|i|gli|le|di|del|della|dei|delle|of|for)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _lock_root_cause_result(
    result: dict,
    retrieval_citations: list[dict],
    *,
    max_causes: int,
) -> tuple[dict, list[dict]]:
    result = dict(result or {})
    possible_causes = list(result.get("possible_causes") or [])
    retrieval_citations = list(retrieval_citations or [])

    if not possible_causes or not retrieval_citations:
        return result, retrieval_citations

    locked_pool = _lock_final_citations(
        selected_citations=retrieval_citations,
        ranked_candidates=retrieval_citations,
        top_k=max(len(retrieval_citations), max_causes * 2),
        diagnostic_mode=True,
        query_token_count=6,
    )

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in retrieval_citations
        if c.get("citation_id")
    }

    family_score_map: dict[str, float] = {}
    family_rank_map: dict[str, int] = {}
    set_rank_map: dict[str, int] = {}

    seen_fams = set()
    seen_sets = set()
    for idx, c in enumerate(locked_pool):
        fam = _stable_evidence_family_key(c)
        set_key = _stable_evidence_set_key(c)
        if fam not in seen_fams:
            seen_fams.add(fam)
            family_rank_map[fam] = idx
        if set_key not in seen_sets:
            seen_sets.add(set_key)
            set_rank_map[set_key] = idx

    for c in retrieval_citations:
        fam = _stable_evidence_family_key(c)
        set_key = _stable_evidence_set_key(c)
        score = (
            float(c.get("retrieval_score", c.get("similarity", 0.0)) or 0.0)
            + 0.10 * float(c.get("overlap_score", 0.0) or 0.0)
            + 0.08 * float(c.get("specificity_score", 0.0) or 0.0)
        )
        family_score_map[fam] = max(score, family_score_map.get(fam, -1.0))
        family_rank_map.setdefault(fam, len(family_rank_map) + 100)
        set_rank_map.setdefault(set_key, len(set_rank_map) + 100)

    rows = []
    for cause in possible_causes:
        if not isinstance(cause, dict):
            continue

        raw_ids = []
        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if cid and cid in by_id:
                raw_ids.append(cid)

        if not raw_ids:
            continue

        family_best: dict[str, dict] = {}
        for cid in raw_ids:
            item = by_id[cid]
            fam = _stable_evidence_family_key(item)
            prev = family_best.get(fam)
            if prev is None or _candidate_order_key(item) < _candidate_order_key(prev):
                family_best[fam] = item

        ordered_fams = sorted(
            family_best.keys(),
            key=lambda fam: (
                family_rank_map.get(fam, 9999),
                -family_score_map.get(fam, 0.0),
                fam,
            ),
        )
        if not ordered_fams:
            continue

        dominant_family = ordered_fams[0]
        dominant_set = _stable_evidence_set_key(family_best[dominant_family])

        same_set_fams = [fam for fam in ordered_fams if _stable_evidence_set_key(family_best[fam]) == dominant_set]
        kept_ids = [str(family_best[fam].get("citation_id") or "").strip() for fam in same_set_fams[:2] if family_best.get(fam)]
        kept_ids = [cid for cid in kept_ids if cid]

        if not kept_ids:
            continue

        label_key = _normalized_cause_label_key(str(cause.get("cause") or ""))
        dominant_item = family_best.get(dominant_family) or {}
        score = (
            family_score_map.get(dominant_family, 0.0)
            + 0.04 * len(kept_ids)
            + 0.26 * float(dominant_item.get("causal_strength_score", 0.0) or 0.0)
            + 0.22 * float(dominant_item.get("semantic_score", 0.0) or 0.0)
            + 0.18 * float(dominant_item.get("subsystem_score", 0.0) or 0.0)
            - (0.05 if bool(dominant_item.get("generic_downranked")) else 0.0)
            - 0.01 * max(0, set_rank_map.get(dominant_set, 9999))
        )

        row = dict(cause)
        row["citations"] = kept_ids
        rows.append(
            {
                "cause": row,
                "label_key": label_key,
                "dominant_family": dominant_family,
                "dominant_set": dominant_set,
                "score": score,
            }
        )

    if not rows:
        return result, retrieval_citations

    best_by_label: dict[str, dict] = {}
    for row in rows:
        key = row["label_key"] or str(row["cause"].get("cause") or "").strip().lower()
        prev = best_by_label.get(key)
        if prev is None or row["score"] > prev["score"] or (
            row["score"] == prev["score"]
            and str(row["cause"].get("cause") or "") < str(prev["cause"].get("cause") or "")
        ):
            best_by_label[key] = row

    deduped = sorted(
        best_by_label.values(),
        key=lambda x: (
            set_rank_map.get(x.get("dominant_set") or "", 9999),
            -float(x.get("score", 0.0)),
            family_rank_map.get(x.get("dominant_family") or "", 9999),
            str((x.get("cause") or {}).get("cause") or ""),
        ),
    )

    final_rows: list[dict] = []
    per_set_counts: dict[str, int] = {}

    for row in deduped:
        set_key = str(row.get("dominant_set") or "")
        count = per_set_counts.get(set_key, 0)
        if count == 0:
            final_rows.append(row)
            per_set_counts[set_key] = 1
            continue

        if count >= 2:
            continue

        first = next((r for r in final_rows if str(r.get("dominant_set") or "") == set_key), None)
        if first is None:
            final_rows.append(row)
            per_set_counts[set_key] = 1
            continue

        gap = float(first.get("score", 0.0)) - float(row.get("score", 0.0))
        if gap <= ROOT_CAUSE_SET_LOCK_DELTA and str(row.get("dominant_family") or "") != str(first.get("dominant_family") or ""):
            final_rows.append(row)
            per_set_counts[set_key] = count + 1

    final_rows = sorted(
        final_rows,
        key=lambda x: (
            set_rank_map.get(x.get("dominant_set") or "", 9999),
            -float(x.get("score", 0.0)),
            family_rank_map.get(x.get("dominant_family") or "", 9999),
            str((x.get("cause") or {}).get("cause") or ""),
        ),
    )[:max_causes]

    final_causes: list[dict] = []
    final_citation_ids: list[str] = []
    seen_citation_ids = set()

    for idx, row in enumerate(final_rows, start=1):
        cause = dict(row["cause"])
        cause["rank"] = idx
        final_causes.append(cause)
        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if cid and cid not in seen_citation_ids:
                seen_citation_ids.add(cid)
                final_citation_ids.append(cid)

    final_citations = [by_id[cid] for cid in final_citation_ids if cid in by_id]

    result["possible_causes"] = final_causes
    if not result.get("recommended_next_checks"):
        flattened_checks = []
        for cause in final_causes:
            flattened_checks.extend(cause.get("checks") or [])
        result["recommended_next_checks"] = _unique_non_empty_strings(flattened_checks, limit=6)

    return result, final_citations


def _ensure_candidate_retrieval_fields(
    citations: list[dict],
    *,
    query_terms: set[str],
    query_style: str = "",
    query_token_count: int = 0,
) -> list[dict]:
    out: list[dict] = []
    max_rrf = max((float((c or {}).get("rrf_score", 0.0) or 0.0) for c in (citations or [])), default=0.0)

    for c in citations or []:
        if not isinstance(c, dict):
            continue

        cc = dict(c)
        cc["snippet"] = (cc.get("snippet") or cc.get("chunk_full") or "").strip()
        cc["chunk_full"] = (cc.get("chunk_full") or cc.get("snippet") or "").strip()
        source_bias, source_meta = _candidate_source_bias(
            cc,
            query_terms,
            query_style=query_style,
            query_token_count=query_token_count,
        )
        specificity_score = _candidate_specificity_score(cc)
        overlap_score = float(source_meta.get("overlap_score", 0.0))
        rrf_norm = (float(cc.get("rrf_score", 0.0) or 0.0) / max_rrf) if max_rrf > 0 else 0.0
        base_retrieval_score = (
            0.58 * float(cc.get("similarity", 0.0) or 0.0)
            + 0.17 * rrf_norm
            + 0.13 * overlap_score
            + specificity_score
            + source_bias
        )

        cc.update(source_meta)
        cc["specificity_score"] = float(cc.get("specificity_score", specificity_score) or 0.0)
        cc["source_bias"] = float(cc.get("source_bias", source_bias) or 0.0)
        cc["overlap_score"] = float(cc.get("overlap_score", overlap_score) or 0.0)
        cc["retrieval_score"] = float(cc.get("retrieval_score", base_retrieval_score) or 0.0)
        out.append(cc)

    return out


def _fallback_root_cause_result_from_matrix(
    *,
    q: str,
    matrix: dict,
    citations: list[dict],
    max_causes: int,
    response_language: str,
) -> dict:
    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations or []
        if c.get("citation_id")
    }

    hypotheses = []
    for idx, row in enumerate((matrix or {}).get("cause_hypotheses") or [], start=1):
        if not isinstance(row, dict):
            continue

        cause = re.sub(r"\s+", " ", str(row.get("cause") or "")).strip()
        checks = _unique_non_empty_strings(row.get("check_focus") or [], limit=4)
        evidence_ids = [
            str(cid or "").strip()
            for cid in (row.get("evidence_ids") or [])
            if str(cid or "").strip() in by_id
        ]

        if not cause or not evidence_ids:
            continue

        why_text = (
            "Best-supported hypothesis from the retrieved evidence matrix."
            if response_language == "en"
            else "Ipotesi meglio supportata dalla matrice di evidenze recuperate."
        )

        hypotheses.append(
            {
                "rank": idx,
                "cause": cause,
                "why": why_text,
                "checks": checks,
                "citations": evidence_ids[:3],
            }
        )

        if len(hypotheses) >= max_causes:
            break

    return {
        "problem_summary": q,
        "possible_causes": hypotheses,
        "recommended_next_checks": _unique_non_empty_strings(
            [chk for row in hypotheses for chk in (row.get("checks") or [])],
            limit=6,
        ),
    }


def _diagnostic_evidence_pipeline(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    max_causes: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "root_cause",
    base_threshold: float = ASK_SIM_THRESHOLD,
) -> dict:
    top_k = max(1, int(top_k or 1))
    max_causes = max(1, min(int(max_causes or 1), 5))
    candidate_k = max(int(candidate_k or 0), max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 8))
    retrieval_top_k = max(top_k, min(ROOT_CAUSE_MAX_EVIDENCE_POOL, max(top_k + 2, 8)))

    base_retrieval = _shared_semantic_retrieval(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=retrieval_top_k,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
        planner_mode=planner_mode,
        base_threshold=base_threshold,
        diagnostic_mode=True,
    )

    if not DIAGNOSTIC_PIPELINE_ENABLED:
        base_retrieval["citations"] = _lock_final_citations(
            selected_citations=list(base_retrieval.get("citations") or []),
            ranked_candidates=list(base_retrieval.get("candidates") or []) + list(base_retrieval.get("citations") or []),
            top_k=top_k,
            diagnostic_mode=True,
            query_token_count=_count_query_tokens(q),
        )
        return base_retrieval

    planner = base_retrieval.get("planner") or {}
    query_style = str(planner.get("query_style") or "").strip().lower()
    query_token_count = _count_query_tokens(q)
    query_terms = _planner_query_term_set(q, planner)

    inferred_components = _infer_machine_components(q)
    diagnostic_queries = _build_diagnostic_queries(q, inferred_components)
    diagnostic_keywords = _collect_candidate_keywords(q, inferred_components)
    target_subsystems = _root_cause_target_subsystems(q, inferred_components)
    symptom_profile = _query_symptom_profile(q)

    base_candidates = _ensure_candidate_retrieval_fields(
        list(base_retrieval.get("candidates") or []) + list(base_retrieval.get("citations") or []),
        query_terms=query_terms,
        query_style=query_style,
        query_token_count=query_token_count,
    )

    extra_dense_queries = _dedup_text_values(
        diagnostic_queries + [planner.get("normalized_query") or q],
        limit=max(4, SEMANTIC_MAX_DENSE_QUERIES + 3),
    )

    extra_candidates: list[dict] = []
    if extra_dense_queries:
        _, dense_candidates, _ = _dense_candidates_multi_query(
            query_texts=extra_dense_queries,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=max(candidate_k, ROOT_CAUSE_EXTRA_CANDIDATE_K),
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
            debug=debug,
        )
        extra_candidates = _ensure_candidate_retrieval_fields(
            dense_candidates,
            query_terms=query_terms,
            query_style=query_style,
            query_token_count=query_token_count,
        )

    merged_pool = _dedup_citations_by_snippet(
        base_candidates + extra_candidates,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 3, top_k * 4, 18),
    )

    rescored_pool: list[dict] = []
    for item in merged_pool:
        cc = dict(item)
        chunk_text = (cc.get("chunk_full") or cc.get("snippet") or "").strip()

        semantic = _score_root_cause_chunk_semantic(q, chunk_text, diagnostic_keywords)
        causal = _score_root_cause_causal_strength(q, chunk_text, diagnostic_keywords)
        subsystem = _score_root_cause_subsystem_alignment(q, chunk_text, target_subsystems)
        context_fit = _score_root_cause_context_fit(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
            symptom_profile=symptom_profile,
            matched_subsystems=subsystem.get("matched_subsystems") or [],
        )
        generic_downranked = _should_downrank_generic_root_cause_chunk(q, chunk_text, diagnostic_keywords)
        hard_excluded = _should_hard_exclude_root_cause_chunk(q, chunk_text, diagnostic_keywords)

        diagnostic_score = float(cc.get("retrieval_score", cc.get("similarity", 0.0)) or 0.0)
        diagnostic_score += float(semantic.get("semantic_score", 0.0) or 0.0)
        diagnostic_score += float(causal.get("causal_strength_score", 0.0) or 0.0)
        diagnostic_score += float(subsystem.get("subsystem_score", 0.0) or 0.0)
        diagnostic_score += float(context_fit.get("context_fit_score", 0.0) or 0.0)

        if str(causal.get("causal_strength_band") or "") == "direct":
            diagnostic_score += ROOT_CAUSE_DIRECT_SIGNAL_BONUS
        if str(cc.get("source_type") or "") == "manual" and str(causal.get("causal_strength_band") or "") in {"direct", "indirect"}:
            diagnostic_score += 0.04
        if bool(context_fit.get("direct_mechanism_supported")) and bool(symptom_profile.get("generic_symptom")):
            diagnostic_score += 0.04
        if generic_downranked:
            diagnostic_score -= ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY
        if bool(context_fit.get("support_only_penalized")):
            diagnostic_score -= 0.06
        if hard_excluded:
            diagnostic_score -= ROOT_CAUSE_HARD_EXCLUDE_PENALTY

        cc.update(semantic)
        cc.update(causal)
        cc.update(subsystem)
        cc.update(context_fit)
        cc["generic_downranked"] = bool(generic_downranked)
        cc["hard_excluded"] = bool(hard_excluded)
        cc["base_retrieval_score"] = float(cc.get("retrieval_score", 0.0) or 0.0)
        cc["diagnostic_score"] = diagnostic_score
        cc["retrieval_score"] = diagnostic_score
        rescored_pool.append(cc)

    rescored_pool.sort(
        key=lambda x: (
            -float(x.get("diagnostic_score", x.get("retrieval_score", x.get("similarity", 0.0))) or 0.0),
            -float(x.get("similarity", 0.0) or 0.0),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )

    non_excluded = [c for c in rescored_pool if not bool(c.get("hard_excluded"))]
    working_pool = non_excluded if len(non_excluded) >= max(top_k, 4) else rescored_pool
    working_pool = _dedup_root_cause_candidates_semantic(
        working_pool,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 2, top_k * 3, 14),
    )
    working_pool = _prioritize_root_cause_coverage(
        working_pool,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
    )

    llm_priority_ids: list[str] = []
    if len(working_pool) >= 4:
        try:
            llm_priority_ids = _llm_filter_diagnostic_chunks(
                q=q,
                candidates=working_pool,
                max_keep=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
            )
        except Exception:
            llm_priority_ids = []

    if llm_priority_ids:
        working_pool = _reorder_citations_by_priority_ids(
            working_pool,
            llm_priority_ids,
            max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
        )

    diagnostic_matrix: dict = {}
    try:
        diagnostic_matrix = _llm_build_diagnostic_evidence_matrix(
            q=q,
            citations=working_pool[: max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2)],
            max_causes=max_causes,
        )
    except Exception:
        diagnostic_matrix = {}

    matrix_keep_ids = _unique_non_empty_strings((diagnostic_matrix or {}).get("keep_ids") or [], limit=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2))
    if matrix_keep_ids:
        working_pool = _reorder_citations_by_priority_ids(
            working_pool,
            matrix_keep_ids,
            max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
        )

    prompt_citations = _select_prompt_citations_from_matrix(
        rescored_candidates=working_pool,
        diagnostic_matrix=diagnostic_matrix or {},
        max_prompt=max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k),
    )
    if not prompt_citations:
        prompt_citations = _lock_final_citations(
            selected_citations=working_pool[: max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k)],
            ranked_candidates=working_pool + rescored_pool,
            top_k=max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k),
            diagnostic_mode=True,
            query_token_count=query_token_count,
        )

    final_citations = _lock_final_citations(
        selected_citations=prompt_citations,
        ranked_candidates=working_pool + rescored_pool,
        top_k=top_k,
        diagnostic_mode=True,
        query_token_count=query_token_count,
    )

    result = dict(base_retrieval)
    result["citations"] = _dedup_citations_by_snippet(
        list(prompt_citations or []) + list(final_citations or []),
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 3),
    )
    result["prompt_citations"] = prompt_citations
    result["candidates"] = rescored_pool
    result["candidate_pool"] = working_pool
    result["diagnostic_matrix"] = diagnostic_matrix or {}
    result["diagnostic_queries"] = diagnostic_queries
    result["diagnostic_keywords"] = diagnostic_keywords
    result["inferred_components"] = inferred_components
    result["target_subsystems"] = target_subsystems
    result["llm_priority_ids"] = llm_priority_ids
    result["extra_dense_queries"] = extra_dense_queries
    result["symptom_profile"] = symptom_profile
    return result


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
    prefer_chunk_full: bool = False,
) -> str:
    ctx_parts: List[str] = []
    total_chars = 0

    for c in citations or []:
        body = ""
        if prefer_chunk_full:
            body = (c.get("chunk_full") or c.get("snippet") or "").strip()
        else:
            body = (c.get("snippet") or c.get("chunk_full") or "").strip()

        part = (
            f"[{c['citation_id']}] "
            f"(doc={c['bubble_document_id']}, p{c['page_from']}-{c['page_to']})\n"
            f"{body}\n"
        )

        if total_chars + len(part) > max_context_chars:
            break

        ctx_parts.append(part)
        total_chars += len(part)

    return "\n".join(ctx_parts).strip()

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


def _score_root_cause_context_fit(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    symptom_profile: dict,
    matched_subsystems: list[str],
) -> dict:
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )

    classes = set(symptom_profile.get("classes") or [])
    matched_set = {str(x).strip() for x in (matched_subsystems or []) if str(x).strip()}
    direct_subsystems = {"drive_train", "material_feed", "forming", "straightening"}
    support_subsystems = {"lubrication", "fluid_power", "electrical_control", "safety_installation"}

    support_only = bool(matched_set) and not (matched_set & direct_subsystems) and (matched_set <= support_subsystems)
    direct_mechanism_supported = bool(matched_set & direct_subsystems)

    has_support_anchor = bool(symptom_profile.get("has_support_anchor"))
    automatic_mode = bool(symptom_profile.get("automatic_mode"))
    generic_symptom = bool(symptom_profile.get("generic_symptom"))

    score = 0.0

    if classes & {"vibration", "noise"}:
        if direct_mechanism_supported and (
            int(sig.get("strong_component_hits", 0) or 0) >= 1
            or int(sig.get("process_hits", 0) or 0) >= 1
            or int(sig.get("symptom_hits", 0) or 0) >= 1
        ):
            score += 0.14

        if support_only and not has_support_anchor:
            score -= ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY

        if int(sig.get("lube_control_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.12
        if int(sig.get("startup_install_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.12
        if int(sig.get("safety_access_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.10

    if "jam" in classes:
        if (matched_set & {"material_feed", "straightening", "forming", "drive_train"}) and (
            int(sig.get("process_hits", 0) or 0) >= 1
            or int(sig.get("strong_component_hits", 0) or 0) >= 1
        ):
            score += 0.14

        if support_only and not has_support_anchor:
            score -= ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY

        if int(sig.get("lube_control_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.10

    if "no_start" in classes:
        if "electrical_control" in matched_set:
            score += 0.14
        if "safety_installation" in matched_set:
            score += 0.10
        if automatic_mode and ({"electrical_control", "safety_installation"} & matched_set):
            score += 0.06
        if matched_set == {"lubrication"} and not has_support_anchor:
            score -= 0.16
        if int(sig.get("startup_install_hits", 0) or 0) >= 2 and not ({"electrical_control", "safety_installation"} & matched_set):
            score -= 0.08

    if generic_symptom and support_only:
        score -= 0.08

    if generic_symptom and direct_mechanism_supported:
        score += 0.08

    score = max(-0.45, min(0.30, score))
    return {
        "context_fit_score": score,
        "support_only_penalized": bool(score < 0 and support_only and (classes & {"vibration", "noise", "jam"})),
        "direct_mechanism_supported": direct_mechanism_supported,
    }


def _select_prompt_citations_from_matrix(
    rescored_candidates: list[dict],
    diagnostic_matrix: dict,
    *,
    max_prompt: int,
) -> list[dict]:
    if not rescored_candidates:
        return []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in rescored_candidates
        if c.get("citation_id")
    }

    out: list[dict] = []
    used_ids = set()
    used_families = set()

    def try_add(cid: str, prefer_new_family: bool = True) -> bool:
        cid = str(cid or "").strip()
        if not cid or cid not in by_id or cid in used_ids:
            return False

        item = by_id[cid]
        fam = _root_cause_evidence_family_key(item)

        if prefer_new_family and fam in used_families:
            return False

        used_ids.add(cid)
        used_families.add(fam)
        out.append(item)
        return True

    for row in (diagnostic_matrix or {}).get("cause_hypotheses") or []:
        per_cause = 0
        for cid in row.get("evidence_ids") or []:
            added = try_add(cid, prefer_new_family=True)
            if not added:
                added = try_add(cid, prefer_new_family=False)
            if added:
                per_cause += 1
            if per_cause >= max(1, ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA):
                break
        if len(out) >= max_prompt:
            return out[:max_prompt]

    for cid in (diagnostic_matrix or {}).get("keep_ids") or []:
        if try_add(cid, prefer_new_family=True) or try_add(cid, prefer_new_family=False):
            if len(out) >= max_prompt:
                return out[:max_prompt]

    for item in rescored_candidates:
        cid = str(item.get("citation_id") or "").strip()
        if try_add(cid, prefer_new_family=True) or try_add(cid, prefer_new_family=False):
            if len(out) >= max_prompt:
                break

    return out[:max_prompt]


def _merge_matrix_supported_causes(
    *,
    result: dict,
    citations: list[dict],
    matrix: dict,
    max_causes: int,
    response_language: str,
) -> tuple[dict, list[dict]]:
    result = dict(result or {})
    possible_causes = list(result.get("possible_causes") or [])
    if len(possible_causes) >= max_causes:
        return result, citations

    fallback = _fallback_root_cause_result_from_matrix(
        q=str(result.get("problem_summary") or "").strip(),
        matrix=matrix,
        citations=citations,
        max_causes=max_causes,
        response_language=response_language,
    )
    if not (fallback or {}).get("possible_causes"):
        return result, citations

    grounded_fallback, grounded_fallback_citations = _ground_root_cause_result(
        result=fallback,
        citations=citations,
        max_causes=max_causes,
    )
    if not grounded_fallback.get("possible_causes"):
        return result, citations

    existing_labels = {
        _normalized_cause_label_key(str(c.get("cause") or ""))
        for c in possible_causes
        if isinstance(c, dict)
    }

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }
    existing_families = set()
    for cause in possible_causes:
        for cid in (cause.get("citations") or [])[:1]:
            item = by_id.get(str(cid or "").strip())
            if item:
                existing_families.add(_root_cause_evidence_family_key(item))

    merged_causes = list(possible_causes)

    for cause in grounded_fallback.get("possible_causes") or []:
        if not isinstance(cause, dict):
            continue

        label_key = _normalized_cause_label_key(str(cause.get("cause") or ""))
        if label_key and label_key in existing_labels:
            continue

        cause_family = None
        for cid in cause.get("citations") or []:
            item = by_id.get(str(cid or "").strip())
            if item:
                cause_family = _root_cause_evidence_family_key(item)
                break

        if cause_family and cause_family in existing_families:
            continue

        merged_causes.append(dict(cause))
        if label_key:
            existing_labels.add(label_key)
        if cause_family:
            existing_families.add(cause_family)

        if len(merged_causes) >= max_causes:
            break

    if len(merged_causes) == len(possible_causes):
        return result, citations

    for idx, cause in enumerate(merged_causes, start=1):
        cause["rank"] = idx

    recommended = _unique_non_empty_strings(
        [chk for row in merged_causes for chk in (row.get("checks") or [])],
        limit=6,
    )

    merged_result = dict(result)
    merged_result["possible_causes"] = merged_causes
    merged_result["recommended_next_checks"] = recommended or list(result.get("recommended_next_checks") or [])

    merged_citations = _dedup_citations_by_snippet(
        list(citations or []) + list(grounded_fallback_citations or []),
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, len(citations) + len(grounded_fallback_citations)),
    )
    return merged_result, merged_citations


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
    return _stable_evidence_family_key(c)


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
                    ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
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
                ORDER BY embedding <=> %s::vector, bubble_document_id, page_from, chunk_index
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




def _should_route_ask_through_root_cause(q: str) -> bool:
    if _is_lookup_or_identifier_query(q):
        return False
    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])
    if "no_start" in classes:
        return True
    if classes & {"vibration", "noise", "jam"}:
        return True
    return False


def _build_ask_answer_from_root_cause_response(q: str, root_response: dict, *, max_causes: int = 2) -> tuple[str, list[dict]]:
    response = dict(root_response or {})
    causes = [c for c in (response.get("possible_causes") or []) if isinstance(c, dict)][: max(1, max_causes)]
    citations = list(response.get("citations") or [])
    if not causes or not citations:
        return "", []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }

    used_ids: list[str] = []
    seen_ids = set()

    def take_ids(cause: dict, limit: int = 2) -> list[str]:
        out: list[str] = []
        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if cid and cid in by_id and cid not in out:
                out.append(cid)
            if len(out) >= limit:
                break
        return out

    primary = causes[0]
    primary_ids = take_ids(primary, limit=2)
    secondary = causes[1] if len(causes) >= 2 else None
    secondary_ids = take_ids(secondary, limit=2) if secondary else []

    checks = _unique_non_empty_strings(
        (primary.get("checks") or []) + ((secondary.get("checks") or []) if secondary else []),
        limit=3,
    )

    def register(ids: list[str]) -> str:
        local = []
        for cid in ids:
            if cid not in seen_ids:
                seen_ids.add(cid)
                used_ids.append(cid)
            local.append(f"[{cid}]")
        return " ".join(local)

    lang = _select_response_language(q)
    primary_label = re.sub(r"\s+", " ", str(primary.get("cause") or "")).strip().rstrip(".")
    secondary_label = re.sub(r"\s+", " ", str((secondary or {}).get("cause") or "")).strip().rstrip(".")

    parts: list[str] = []
    if lang == "en":
        if primary_label:
            cite = register(primary_ids[:1] or primary_ids)
            parts.append(f"The strongest evidence points to {primary_label}. {cite}".strip())
        if secondary_label:
            cite = register(secondary_ids[:1] or secondary_ids)
            parts.append(f"A secondary possibility is {secondary_label}. {cite}".strip())
        if checks:
            check_text = "; ".join(checks)
            cite_ids = primary_ids[:1] + [cid for cid in secondary_ids[:1] if cid not in primary_ids[:1]]
            cite = register(cite_ids)
            parts.append(f"Recommended checks: {check_text}. {cite}".strip())
    else:
        if primary_label:
            cite = register(primary_ids[:1] or primary_ids)
            parts.append(f"Le evidenze puntano soprattutto a {primary_label}. {cite}".strip())
        if secondary_label:
            cite = register(secondary_ids[:1] or secondary_ids)
            parts.append(f"In seconda battuta è plausibile {secondary_label}. {cite}".strip())
        if checks:
            check_text = "; ".join(checks)
            cite_ids = primary_ids[:1] + [cid for cid in secondary_ids[:1] if cid not in primary_ids[:1]]
            cite = register(cite_ids)
            parts.append(f"Verifiche consigliate: {check_text}. {cite}".strip())

    answer = re.sub(r"\s+", " ", " ".join(parts)).strip()
    final_citations = [by_id[cid] for cid in used_ids if cid in by_id]
    return answer, final_citations


def _build_ask_response_from_root_cause_bridge(
    *,
    q: str,
    company_id: str,
    top_k: int,
    root_response: dict,
    debug: bool = False,
) -> Optional[dict]:
    if str((root_response or {}).get("status") or "").strip().lower() != "answered":
        return None

    answer, final_citations = _build_ask_answer_from_root_cause_response(q, root_response, max_causes=2)
    if not answer or not final_citations:
        return None

    response_citations = _sanitize_citations_for_response(final_citations)
    rg_links = []
    try:
        rg_links = _build_rg_links(company_id, response_citations)
    except Exception as e:
        print("RG_LINKS_FAIL", str(e))
        rg_links = []

    resp = {
        "ok": True,
        "status": "answered",
        "answer": answer,
        "citations": response_citations,
        "rg_links": rg_links,
        "top_k": top_k,
        "similarity_max": (root_response or {}).get("similarity_max"),
        "chat_model": (root_response or {}).get("chat_model") or ROOT_CAUSE_RESPONSE_MODEL,
    }
    if debug:
        resp["debug"] = {
            "root_cause_bridge": True,
            "root_cause_status": str((root_response or {}).get("status") or ""),
            "root_cause_similarity_max": (root_response or {}).get("similarity_max"),
            "root_cause_debug": (root_response or {}).get("debug") or {},
        }
    return resp

def _ask_v1_baseline_impl(
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
    candidate_k = max(top_k, min(48, top_k * 7))

    if _should_route_ask_through_root_cause(q):
        try:
            root_payload = RootCauseRequest(
                query=q,
                company_id=company_id,
                machine_id=machine_id,
                bubble_document_id=bubble_document_id,
                document_ids=doc_ids,
                top_k=max(6, top_k),
                max_causes=2,
                debug=payload.debug,
            )
            root_response = root_cause_v1(root_payload, x_ai_internal_secret)
            bridged = _build_ask_response_from_root_cause_bridge(
                q=q,
                company_id=company_id,
                top_k=top_k,
                root_response=root_response,
                debug=bool(payload.debug),
            )
            if bridged:
                return bridged
        except Exception as e:
            if payload.debug:
                print("ASK_ROOT_CAUSE_BRIDGE_FAIL", str(e))

    primary_retrieval = _shared_semantic_retrieval(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=top_k,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=bubble_document_id,
        debug=payload.debug,
        planner_mode="ask",
        base_threshold=ASK_SIM_THRESHOLD,
        diagnostic_mode=False,
    )

    planner = primary_retrieval.get("planner") or {}
    response_language = _select_response_language(q, planner=planner)
    no_sources_text = _localized_no_sources(response_language)

    retrieval = primary_retrieval
    citations = list(retrieval.get("citations") or [])
    sim_max = retrieval.get("similarity_max")
    query_token_count = _count_query_tokens(q)
    symptom_profile = _query_symptom_profile(q)
    ask_arbiter_debug: dict = {}

    rescue_retrieval: Optional[dict] = None
    need_rescue_retrieval = (
        query_token_count >= 4
        and (
            not citations
            or response_language == "en"
            or all(_source_type_from_document_id(c.get("bubble_document_id") or "") in {"ps", "procedure", "step"} for c in citations)
            or float(sim_max or 0.0) < float(retrieval.get("effective_threshold") or ASK_SIM_THRESHOLD) + 0.05
        )
    )

    if need_rescue_retrieval:
        rescue_retrieval = _shared_semantic_retrieval(
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=max(candidate_k, top_k * 10),
            top_k=max(top_k, 5),
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
            debug=payload.debug,
            planner_mode="root_cause",
            base_threshold=min(ASK_SIM_THRESHOLD, ASK_SHORT_QUERY_SIM_THRESHOLD),
            diagnostic_mode=True,
        )

        primary_score = _retrieval_quality_score(primary_retrieval)
        rescue_score = _retrieval_quality_score(rescue_retrieval)

        if rescue_score > primary_score + 0.04:
            retrieval = rescue_retrieval
        elif rescue_retrieval.get("citations"):
            merged = _dedup_citations_by_snippet(
                list(primary_retrieval.get("citations") or []) + list(rescue_retrieval.get("citations") or []),
                max_items=max(top_k, 6),
            )
            merged = _lock_final_citations(
                selected_citations=merged,
                ranked_candidates=list(primary_retrieval.get("candidates") or []) + list(rescue_retrieval.get("candidates") or []) + list(merged or []),
                top_k=top_k,
                diagnostic_mode=False,
                query_token_count=query_token_count,
            )
            retrieval = dict(primary_retrieval)
            retrieval["citations"] = merged
            retrieval["similarity_max"] = max(
                float(primary_retrieval.get("similarity_max") or 0.0),
                float(rescue_retrieval.get("similarity_max") or 0.0),
            )
            retrieval["fts_used"] = bool(primary_retrieval.get("fts_used")) or bool(rescue_retrieval.get("fts_used"))
            retrieval["prefix_fts_used"] = bool(primary_retrieval.get("prefix_fts_used")) or bool(rescue_retrieval.get("prefix_fts_used"))
            retrieval["exact_fts_used"] = bool(primary_retrieval.get("exact_fts_used")) or bool(rescue_retrieval.get("exact_fts_used"))

    planner = retrieval.get("planner") or planner
    citations = list(retrieval.get("citations") or [])
    if citations:
        citations = _lock_final_citations(
            selected_citations=citations,
            ranked_candidates=list(retrieval.get("candidates") or []) + list(citations or []),
            top_k=top_k,
            diagnostic_mode=False,
            query_token_count=query_token_count,
        )
        retrieval["citations"] = citations
    sim_max = retrieval.get("similarity_max")

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "query_plan": planner,
                "dense_queries": retrieval.get("dense_queries") or [],
                "lexical_queries": retrieval.get("lexical_queries") or [],
                "chunks_matching_filter": retrieval.get("chunks_matching_filter"),
                "similarity_max": sim_max,
                "effective_ask_threshold": retrieval.get("effective_threshold"),
                "fts_used": bool(retrieval.get("fts_used")),
                "prefix_fts_used": bool(retrieval.get("prefix_fts_used")),
                "exact_fts_used": bool(retrieval.get("exact_fts_used")),
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": bool(retrieval.get("rerank_used")),
                "rerank_error": retrieval.get("rerank_error"),
                "rescue_retrieval_used": rescue_retrieval is not None,
                "rescue_retrieval_quality": _retrieval_quality_score(rescue_retrieval or {}),
                "primary_retrieval_quality": _retrieval_quality_score(primary_retrieval or {}),
            }
        return resp

    effective_threshold = float(retrieval.get("effective_threshold") or ASK_SIM_THRESHOLD)

    if (sim_max is None or float(sim_max) < effective_threshold) and citations:
        picked = _pick_entity_from_citations(q, citations)
        if picked:
            value, c = picked

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
                    "answer": _localized_value_answer(response_language, value, c["citation_id"]),
                    "citations": _sanitize_citations_for_response(citations),
                    "rg_links": rg_links,
                    "top_k": top_k,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                }
            )

    if (sim_max is None or float(sim_max) < effective_threshold):
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
                cit_list = [
                    {
                        "citation_id": hit["citation_id"],
                        "bubble_document_id": hit["bubble_document_id"],
                        "chunk_index": int(hit.get("chunk_index") or 0),
                        "page_from": hit["page_from"],
                        "page_to": hit["page_to"],
                        "snippet": hit["snippet"],
                        "similarity": sim_max or 0.0,
                    }
                ]
                rg_links = _build_rg_links(company_id, cit_list)
                return _finalize(
                    {
                        "ok": True,
                        "status": "answered",
                        "answer": _localized_value_answer(response_language, hit["value"], hit["citation_id"]),
                        "citations": _sanitize_citations_for_response(cit_list),
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

            if hit and matched_token:
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
                        "answer": _localized_token_answer(response_language, matched_token, hit["citation_id"]),
                        "citations": _sanitize_citations_for_response(cit_list),
                        "rg_links": rg_links,
                        "top_k": top_k,
                        "similarity_max": sim_max,
                        "chat_model": OPENAI_CHAT_MODEL,
                    }
                )

    if not citations:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": no_sources_text,
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
            }
        )

    answer_status, grounded_points = _generate_ask_grounded_points(
        q=q,
        planner=planner,
        response_language=response_language,
        company_id=company_id,
        citations=citations,
        allow_no_sources=True,
    )

    if answer_status == "no_sources" or not grounded_points:
        answer_status, grounded_points = _generate_ask_grounded_points(
            q=q,
            planner=planner,
            response_language=response_language,
            company_id=company_id,
            citations=citations,
            allow_no_sources=False,
        )

    if answer_status == "no_sources" or not grounded_points:
        answer, final_citations = _extractive_fallback_answer(
            citations=citations,
            response_language=response_language,
            max_points=min(2, top_k),
        )
    else:
        answer, final_citations = _render_grounded_answer_points(
            grounded_points=grounded_points,
            citations=citations,
            max_points=min(3, top_k),
        )

    if not answer or not final_citations:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": no_sources_text,
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
            }
        )

    relocked_final_citations = _lock_final_citations(
        selected_citations=final_citations,
        ranked_candidates=list(retrieval.get("candidates") or []) + list(citations or []) + list(final_citations or []),
        top_k=min(top_k, max(1, len(final_citations))),
        diagnostic_mode=False,
        query_token_count=query_token_count,
    )

    if relocked_final_citations:
        original_ids = [str(c.get("citation_id") or "").strip() for c in final_citations if c.get("citation_id")]
        relocked_ids = [str(c.get("citation_id") or "").strip() for c in relocked_final_citations if c.get("citation_id")]

        if relocked_ids != original_ids or query_token_count >= 4:
            stable_answer, stable_citations = _extractive_fallback_answer(
                relocked_final_citations,
                response_language=response_language,
                max_points=min(2, top_k),
            )
            if stable_answer and stable_citations:
                answer = stable_answer
                final_citations = stable_citations
            else:
                final_citations = relocked_final_citations

    if not _looks_like_target_language(answer, response_language):
        answer = _translate_text_preserving_citations(answer, response_language)

    response_citations = _sanitize_citations_for_response(final_citations)

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
            "answer": answer,
            "citations": response_citations,
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

    options = payload.options or DraftPSOptions()

    top_k = int(options.top_k or 8)
    top_k = max(3, min(top_k, 12))

    max_causes = int(options.max_causes or 3)
    max_causes = max(1, min(max_causes, 5))

    candidate_k = max(top_k, min(80, max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 10)))

    retrieval = _diagnostic_evidence_pipeline(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=top_k,
        max_causes=max_causes,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=bubble_document_id,
        debug=payload.debug,
        planner_mode="draft_ps",
        base_threshold=DRAFT_PS_SIM_THRESHOLD,
    )

    planner = retrieval.get("planner") or {}
    language = _select_response_language(q, planner=planner, preferred=payload.language)
    sim_max = retrieval.get("similarity_max")
    citations = list(retrieval.get("citations") or [])

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "query_plan": planner,
                "dense_queries": retrieval.get("dense_queries") or [],
                "lexical_queries": retrieval.get("lexical_queries") or [],
                "extra_dense_queries": retrieval.get("extra_dense_queries") or [],
                "diagnostic_queries": retrieval.get("diagnostic_queries") or [],
                "diagnostic_keywords": retrieval.get("diagnostic_keywords") or [],
                "inferred_components": retrieval.get("inferred_components") or [],
                "target_subsystems": retrieval.get("target_subsystems") or [],
                "diagnostic_matrix": retrieval.get("diagnostic_matrix") or {},
                "llm_priority_ids": retrieval.get("llm_priority_ids") or [],
                "symptom_profile": retrieval.get("symptom_profile") or {},
                "chunks_matching_filter": retrieval.get("chunks_matching_filter"),
                "similarity_max": sim_max,
                "effective_threshold": retrieval.get("effective_threshold"),
                "fts_used": bool(retrieval.get("fts_used")),
                "prefix_fts_used": bool(retrieval.get("prefix_fts_used")),
                "exact_fts_used": bool(retrieval.get("exact_fts_used")),
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": bool(retrieval.get("rerank_used")),
                "rerank_error": retrieval.get("rerank_error"),
            }
        return resp

    if not citations:
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
                    "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
                },
            }
        )

    prompt_citations = []
    for c in (retrieval.get("prompt_citations") or citations):
        cc = dict(c)
        cc["chunk_full"] = (cc.get("chunk_full") or cc.get("snippet") or "").strip()[:1800]
        cc["snippet"] = (cc.get("snippet") or cc.get("chunk_full") or "").strip()
        prompt_citations.append(cc)

    prompt_citations = prompt_citations[: max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k)]
    sources_block = _build_sources_block_from_citations(
        prompt_citations,
        max_context_chars=ASK_MAX_CONTEXT_CHARS,
        prefer_chunk_full=True,
    )

    matrix = retrieval.get("diagnostic_matrix") or {}
    matrix_json = json.dumps(matrix, ensure_ascii=False)
    inferred_components = json.dumps(retrieval.get("inferred_components") or [], ensure_ascii=False)
    target_subsystems = json.dumps(retrieval.get("target_subsystems") or [], ensure_ascii=False)

    schema = _draft_ps_response_schema(max_causes=max_causes)

    if language == "en":
        system_msg = (
            "You draft grounded Problem & Solution entries for technical equipment. "
            "Use ONLY the provided sources and the evidence matrix. "
            "Work domain-agnostically: do not assume a sector, machine family, or failure taxonomy unless the sources support it. "
            "Treat the evidence matrix as the preferred structure for candidate causes and checks. "
            "It is acceptable to make cautious multi-source inferences, but every proposed cause must stay tightly grounded. "
            "Each cause label should be compact, canonical, and technically precise. "
            "Prefer direct component/process evidence over generic maintenance or safety content. For generic symptoms like vibration, noise, or jams, do not center lubrication, startup, installation, or safety unless the sources explicitly connect them to the symptom. For no-start cases, prefer electrical supply, interlock, consent, mode-selection, and control evidence over generic lubrication notes. "
            "Merge near-duplicate causes instead of listing paraphrases. If the evidence matrix contains two distinct hypotheses with separate evidence families, preserve more than one cause instead of collapsing to one. Do not narrow to a single component unless the sources support that narrowing directly. "
            "Use only citation_id values present in the sources."
        )

        user_msg = (
            f"USER_PROBLEM:\n{q}\n\n"
            f"NORMALIZED_PROBLEM:\n{planner.get('normalized_query') or q}\n\n"
            f"INFERRED_COMPONENTS_JSON:\n{inferred_components}\n\n"
            f"TARGET_SUBSYSTEMS_JSON:\n{target_subsystems}\n\n"
            f"DIAGNOSTIC_EVIDENCE_MATRIX_JSON:\n{matrix_json}\n\n"
            f"SOURCES:\n{sources_block}\n\n"
            "Return valid JSON for a grounded Problem & Solution draft."
        )
    else:
        system_msg = (
            "Redigi bozze grounded di Problem & Solution per apparecchiature tecniche. "
            "Usa SOLO le fonti fornite e la matrice di evidenze. "
            "Lavora in modo domain-agnostic: non assumere settore, famiglia macchina o tassonomia guasti se le fonti non lo supportano. "
            "Tratta la matrice di evidenze come struttura preferita per possibili cause e verifiche. "
            "Sono ammesse inferenze caute multi-fonte, ma ogni possibile causa deve restare strettamente grounded. "
            "Ogni causa deve avere un'etichetta compatta, canonica e tecnicamente precisa. "
            "Preferisci evidenza diretta di componente/processo rispetto a contenuti generici di manutenzione o sicurezza. Per sintomi generici come vibrazione, rumore o blocco, non centrare lubrificazione, start-up, installazione o sicurezza se le fonti non le collegano esplicitamente al sintomo. Per il mancato avvio, preferisci evidenze di alimentazione elettrica, interlock, consensi, selezione modalità e controllo rispetto a note generiche di lubrificazione. "
            "Unisci cause quasi duplicate invece di elencare parafrasi. Se la matrice di evidenze contiene due ipotesi distinte con famiglie di evidenza separate, conserva più di una causa invece di collassare tutto in una sola. Non restringere a un singolo componente se le fonti non supportano direttamente quel restringimento. "
            "Usa solo citation_id presenti nelle fonti."
        )

        user_msg = (
            f"PROBLEMA_UTENTE:\n{q}\n\n"
            f"PROBLEMA_NORMALIZZATO:\n{planner.get('normalized_query') or q}\n\n"
            f"COMPONENTI_INFERITI_JSON:\n{inferred_components}\n\n"
            f"SOTTOSISTEMI_TARGET_JSON:\n{target_subsystems}\n\n"
            f"MATRICE_EVIDENZE_DIAGNOSTICHE_JSON:\n{matrix_json}\n\n"
            f"FONTI:\n{sources_block}\n\n"
            "Restituisci JSON valido per una bozza grounded di Problem & Solution."
        )

    try:
        result_json = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_RESPONSE_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=90,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    if not (result_json or {}).get("possible_causes") and matrix:
        fallback_json = _fallback_root_cause_result_from_matrix(
            q=planner.get("normalized_query") or q,
            matrix=matrix,
            citations=citations,
            max_causes=max_causes,
            response_language=language,
        )
        result_json = {
            "title": str((result_json or {}).get("title") or "").strip(),
            "problem_summary": str((result_json or {}).get("problem_summary") or "").strip() or q,
            "possible_causes": fallback_json.get("possible_causes") or [],
        }

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
    grounded_result = _canonicalize_root_cause_labels(
        grounded_result,
        grounded_citations,
        language=language,
    )
    grounded_result, grounded_citations = _lock_root_cause_result(
        grounded_result,
        grounded_citations,
        max_causes=max_causes,
    )
    if matrix and len(matrix.get("cause_hypotheses") or []) >= ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES:
        grounded_result, grounded_citations = _merge_matrix_supported_causes(
            result=grounded_result,
            citations=grounded_citations,
            matrix=matrix,
            max_causes=max_causes,
            response_language=language,
        )
        grounded_result = _canonicalize_root_cause_labels(
            grounded_result,
            grounded_citations,
            language=language,
        )

    if not grounded_result.get("possible_causes") and matrix:
        fallback_grounded = _fallback_root_cause_result_from_matrix(
            q=planner.get("normalized_query") or q,
            matrix=matrix,
            citations=citations,
            max_causes=max_causes,
            response_language=language,
        )
        grounded_result, grounded_citations = _ground_root_cause_result(
            result=fallback_grounded,
            citations=citations,
            max_causes=max_causes,
        )
        grounded_result, grounded_citations = _compact_root_cause_result_citations_by_family(
            result=grounded_result,
            citations=grounded_citations,
            max_per_cause=2,
        )
        grounded_result, grounded_citations = _lock_root_cause_result(
            grounded_result,
            grounded_citations,
            max_causes=max_causes,
        )

    final_title = str((result_json or {}).get("title") or "").strip()
    if not final_title:
        final_title = (f"P&S draft — {q[:80]}" if language == "en" else f"Bozza P&S — {q[:80]}")

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
                    "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
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
                "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
                "language": language,
            },
        }
    )


def _root_cause_v1_baseline_impl(
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
    candidate_k = max(top_k, min(80, max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 10)))

    query_signal_summary = _root_cause_query_signal_summary(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        debug=payload.debug,
    )
    query_fail_closed = _should_fail_closed_root_cause_query(query_signal_summary)

    prelim = query_signal_summary.get("preliminary_retrieval") or {}
    preliminary_chunks_matching_filter = prelim.get("chunks_matching_filter")
    preliminary_similarity_max = prelim.get("similarity_max")

    retrieval = {
        "planner": {},
        "dense_queries": [],
        "lexical_queries": [],
        "extra_dense_queries": [],
        "diagnostic_queries": [],
        "diagnostic_keywords": [],
        "inferred_components": [],
        "target_subsystems": [],
        "diagnostic_matrix": {},
        "llm_priority_ids": [],
        "symptom_profile": {},
        "chunks_matching_filter": None,
        "similarity_max": None,
        "effective_threshold": ASK_SIM_THRESHOLD,
        "citations": [],
        "prompt_citations": [],
        "candidate_pool": [],
        "fts_used": False,
        "prefix_fts_used": False,
        "exact_fts_used": False,
        "rerank_used": False,
        "rerank_error": None,
    }

    if not query_fail_closed:
        retrieval = _diagnostic_evidence_pipeline(
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=candidate_k,
            top_k=top_k,
            max_causes=max_causes,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
            debug=payload.debug,
            planner_mode="root_cause",
            base_threshold=ASK_SIM_THRESHOLD,
        )

    planner = retrieval.get("planner") or {}
    sim_max = retrieval.get("similarity_max")

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "query_signal_summary": query_signal_summary,
                "query_fail_closed": query_fail_closed,
                "preliminary_chunks_matching_filter": preliminary_chunks_matching_filter,
                "preliminary_similarity_max": preliminary_similarity_max,
                "query_plan": planner,
                "dense_queries": retrieval.get("dense_queries") or [],
                "lexical_queries": retrieval.get("lexical_queries") or [],
                "extra_dense_queries": retrieval.get("extra_dense_queries") or [],
                "diagnostic_queries": retrieval.get("diagnostic_queries") or [],
                "diagnostic_keywords": retrieval.get("diagnostic_keywords") or [],
                "inferred_components": retrieval.get("inferred_components") or [],
                "target_subsystems": retrieval.get("target_subsystems") or [],
                "diagnostic_matrix": retrieval.get("diagnostic_matrix") or {},
                "llm_priority_ids": retrieval.get("llm_priority_ids") or [],
                "symptom_profile": retrieval.get("symptom_profile") or {},
                "chunks_matching_filter": retrieval.get("chunks_matching_filter"),
                "similarity_max": sim_max,
                "effective_threshold": retrieval.get("effective_threshold"),
                "fts_used": bool(retrieval.get("fts_used")),
                "prefix_fts_used": bool(retrieval.get("prefix_fts_used")),
                "exact_fts_used": bool(retrieval.get("exact_fts_used")),
                "rerank_enabled": RERANK_ENABLED,
                "rerank_used": bool(retrieval.get("rerank_used")),
                "rerank_error": retrieval.get("rerank_error"),
            }
        return resp

    if query_fail_closed:
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
                "similarity_max": preliminary_similarity_max,
            }
        )

    citations = list(retrieval.get("citations") or [])
    if not citations:
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

    prompt_citations = []
    for c in (retrieval.get("prompt_citations") or citations):
        cc = dict(c)
        cc["chunk_full"] = (cc.get("chunk_full") or cc.get("snippet") or "").strip()[:1800]
        cc["snippet"] = (cc.get("snippet") or cc.get("chunk_full") or "").strip()
        prompt_citations.append(cc)

    prompt_citations = prompt_citations[: max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k)]

    sources_block = _build_sources_block_from_citations(
        prompt_citations,
        max_context_chars=ASK_MAX_CONTEXT_CHARS,
        prefer_chunk_full=True,
    )

    schema = _root_cause_response_schema(max_causes=max_causes)
    response_language = _select_response_language(q, planner=planner)

    matrix = retrieval.get("diagnostic_matrix") or {}
    matrix_json = json.dumps(matrix, ensure_ascii=False)
    inferred_components = json.dumps(retrieval.get("inferred_components") or [], ensure_ascii=False)
    target_subsystems = json.dumps(retrieval.get("target_subsystems") or [], ensure_ascii=False)

    system_msg = (
        "You are a root-cause assistant for technical equipment and machine documentation. "
        "Use ONLY the provided sources and evidence matrix. "
        "Work domain-agnostically: do not assume a sector, machine family, subsystem taxonomy, or standard failure mode unless the sources support it. "
        "Treat the evidence matrix as the preferred structure for candidate causes and checks. "
        "Procedures and problem-solution entries are valid evidence when directly relevant, but a generic procedure or generic P&S must not outweigh a more specific manual passage. "
        "You may make cautious multi-source inferences, but every proposed cause must remain tightly grounded. "
        "Order causes by groundedness, causal support, and specificity, not by creativity. "
        "Each 'cause' must be a short canonical label of about 3 to 10 words, preferably a noun phrase, with no trailing period. "
        "Reuse source terminology when possible and avoid switching between near-synonymous paraphrases across runs. "
        "If the evidence is narrow, return fewer causes rather than broad generic ones. "
        "Avoid generic boilerplate causes unless the sources clearly support them. "
        "Merge near-duplicate causes instead of listing paraphrases. If the evidence matrix contains two distinct hypotheses with separate evidence families, preserve more than one cause instead of collapsing to one. For generic symptoms like vibration, noise, or jams, do not center lubrication, startup, installation, or safety unless the sources explicitly connect them to the symptom. For no-start cases, prefer electrical supply, interlock, consent, mode-selection, and control evidence over generic lubrication notes. Do not narrow to a single component unless the sources support that narrowing directly. "
        "Always reply in the same language as the user query."
    )

    user_msg = (
        f"USER_PROBLEM:\n{q}\n\n"
        f"NORMALIZED_PROBLEM:\n{planner.get('normalized_query') or q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"INFERRED_COMPONENTS_JSON:\n{inferred_components}\n\n"
        f"TARGET_SUBSYSTEMS_JSON:\n{target_subsystems}\n\n"
        f"DIAGNOSTIC_EVIDENCE_MATRIX_JSON:\n{matrix_json}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        "Return valid JSON. Use only citation_id values present in the sources."
    )

    try:
        result_json = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_RESPONSE_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=90,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    if not (result_json or {}).get("possible_causes") and matrix:
        result_json = _fallback_root_cause_result_from_matrix(
            q=planner.get("normalized_query") or q,
            matrix=matrix,
            citations=citations,
            max_causes=max_causes,
            response_language=response_language,
        )

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
    grounded_result = _canonicalize_root_cause_labels(
        grounded_result,
        grounded_citations,
        language=response_language,
    )
    grounded_result, grounded_citations = _lock_root_cause_result(
        grounded_result,
        grounded_citations,
        max_causes=max_causes,
    )
    if matrix and len(matrix.get("cause_hypotheses") or []) >= ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES:
        grounded_result, grounded_citations = _merge_matrix_supported_causes(
            result=grounded_result,
            citations=grounded_citations,
            matrix=matrix,
            max_causes=max_causes,
            response_language=response_language,
        )
        grounded_result = _canonicalize_root_cause_labels(
            grounded_result,
            grounded_citations,
            language=response_language,
        )

    if not grounded_result.get("possible_causes") and matrix:
        fallback_grounded = _fallback_root_cause_result_from_matrix(
            q=planner.get("normalized_query") or q,
            matrix=matrix,
            citations=citations,
            max_causes=max_causes,
            response_language=response_language,
        )
        grounded_result, grounded_citations = _ground_root_cause_result(
            result=fallback_grounded,
            citations=citations,
            max_causes=max_causes,
        )
        grounded_result, grounded_citations = _compact_root_cause_result_citations_by_family(
            result=grounded_result,
            citations=grounded_citations,
            max_per_cause=2,
        )
        grounded_result, grounded_citations = _lock_root_cause_result(
            grounded_result,
            grounded_citations,
            max_causes=max_causes,
        )

    if not grounded_result.get("problem_summary"):
        grounded_result["problem_summary"] = planner.get("normalized_query") or q

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
                "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
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
            "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
        }
    )




def _strip_internal_response_artifacts(resp: dict) -> dict:
    if not isinstance(resp, dict):
        return resp

    out = {}
    for k, v in resp.items():
        if str(k).startswith("_arb_"):
            continue
        out[k] = v
    return out


def _cause_label_specificity_score(label: str) -> float:
    txt = re.sub(r"\s+", " ", _normalize_unicode_advanced(label or "")).strip().lower()
    if not txt:
        return 0.0

    generic_markers = {
        "problem", "issue", "fault", "anomaly", "generic anomaly", "possible cause",
        "problema", "guasto", "anomalia", "possibile causa", "mancato avviamento intenzionale",
        "not starting", "does not start", "machine issue", "machine problem", "machine fault",
    }
    technical_markers = [
        "play", "wear", "backlash", "misalignment", "alignment", "transmission", "gear", "gearbox",
        "bearing", "roller", "slide", "guide", "feed", "straighten", "press", "eccentric", "cam",
        "drive", "motor", "phase", "power", "interlock", "selector", "panel", "pressure", "hydraulic",
        "pneumatic", "lubric", "oil", "ridutt", "ingran", "cuscinet", "rullo", "slitta", "guida",
        "avanz", "raddrizz", "pressa", "eccentric", "camme", "fasi", "aliment", "interblocco",
        "selettore", "quadro", "pression", "idraulic", "pneumat", "lubr", "olio",
    ]
    score = 0.35
    words = re.findall(r"[a-zà-öø-ÿ0-9]+", txt)
    if 2 <= len(words) <= 8:
        score += 0.15
    if any(m in txt for m in technical_markers):
        score += 0.35
    if txt in generic_markers or any(txt == g for g in generic_markers):
        score -= 0.35
    if any(g in txt for g in ["generic", "generico", "issue", "problema", "fault", "anomalia"]):
        score -= 0.12
    return max(0.0, min(1.0, score))




def _looks_like_installation_positioning_false_positive(text: str) -> bool:
    txt = re.sub(r"\s+", " ", _normalize_unicode_advanced(text or "")).strip().lower()
    if not txt:
        return False

    hard_phrases = [
        "posizionamento della macchina",
        "machine positioning",
        "piano di appoggio",
        "support surface",
        "spessori di gomma",
        "rubber shims",
        "attutire le vibrazioni",
        "attenuate the vibrations",
        "flatness of the support surface",
        "planarità del piano di appoggio",
        "fori di fondazione",
        "foundation holes",
        "livellamento della macchina",
        "machine leveling",
        "spirit level",
        "livella",
    ]
    if any(p in txt for p in hard_phrases):
        return True

    soft_phrases = [
        "fondazione",
        "foundation",
        "posizionamento",
        "positioning",
        "livellamento",
        "leveling",
        "planarità",
        "planarity",
    ]
    hits = sum(1 for p in soft_phrases if p in txt)
    return hits >= 2

def _classify_diagnostic_role_from_text(
    q: str,
    chunk_text: str,
    symptom_profile: dict,
    diagnostic_keywords: list[str],
    target_subsystems: list[str],
) -> dict:
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return {
            "role_class": "collateral",
            "role_group": "collateral",
            "role_adjustment": -0.06,
            "matched_subsystems": [],
            "role_reason": "empty_chunk",
        }

    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    subsystem = _score_root_cause_subsystem_alignment(
        q=q,
        chunk_text=chunk_text,
        target_subsystems=target_subsystems,
    )

    classes = set(symptom_profile.get("classes") or [])
    matched_set = {str(x).strip() for x in (subsystem.get("matched_subsystems") or []) if str(x).strip()}
    direct_subsystems = {"drive_train", "material_feed", "forming", "straightening"}
    support_subsystems = {"lubrication", "fluid_power", "electrical_control", "safety_installation"}
    has_support_anchor = bool(symptom_profile.get("has_support_anchor"))
    automatic_mode = bool(symptom_profile.get("automatic_mode"))

    strong_component_hits = int(sig.get("strong_component_hits", 0) or 0)
    process_hits = int(sig.get("process_hits", 0) or 0)
    symptom_hits = int(sig.get("symptom_hits", 0) or 0)
    lube_hits = int(sig.get("lube_control_hits", 0) or 0)
    startup_hits = int(sig.get("startup_install_hits", 0) or 0) + int(sig.get("positioning_hits", 0) or 0)
    safety_hits = int(sig.get("safety_access_hits", 0) or 0) + int(sig.get("acoustic_protection_hits", 0) or 0)

    direct_mechanism_supported = bool(matched_set & direct_subsystems) and (
        strong_component_hits >= 1 or process_hits >= 1 or symptom_hits >= 1 or float(subsystem.get("subsystem_score", 0.0) or 0.0) > 0.0
    )

    if "no_start" in classes:
        if "electrical_control" in matched_set:
            return {
                "role_class": "support_electrical_interlock",
                "role_group": "support",
                "role_adjustment": 0.16 if automatic_mode else 0.12,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_electrical_control",
            }
        if "safety_installation" in matched_set:
            return {
                "role_class": "support_safety",
                "role_group": "support",
                "role_adjustment": 0.10 if automatic_mode else 0.04,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_safety_interlock",
            }
        if "lubrication" in matched_set or lube_hits >= 2:
            return {
                "role_class": "support_lubrication",
                "role_group": "support",
                "role_adjustment": -0.18 if not has_support_anchor else -0.05,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_lubrication_secondary",
            }

    if direct_mechanism_supported:
        if (matched_set & {"forming", "material_feed", "straightening"}) or process_hits >= 1:
            return {
                "role_class": "core_process",
                "role_group": "core",
                "role_adjustment": 0.14,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "direct_process_mechanism",
            }
        return {
            "role_class": "core_mechanical",
            "role_group": "core",
            "role_adjustment": 0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "direct_mechanical_mechanism",
        }

    if matched_set & {"fluid_power"}:
        return {
            "role_class": "support_fluid_power",
            "role_group": "support",
            "role_adjustment": 0.06 if ("jam" in classes or has_support_anchor) else -0.04,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "fluid_power_support",
        }

    if matched_set & {"electrical_control"}:
        return {
            "role_class": "support_electrical_interlock",
            "role_group": "support",
            "role_adjustment": 0.08 if ("no_start" in classes or has_support_anchor) else -0.03,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "electrical_or_control_support",
        }

    if matched_set & {"lubrication"} or lube_hits >= 2:
        return {
            "role_class": "support_lubrication",
            "role_group": "support",
            "role_adjustment": 0.04 if has_support_anchor else -0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "lubrication_support",
        }

    if startup_hits >= 2 or bool(sig.get("overview_section_hit")) or bool(sig.get("description_section_hit")):
        return {
            "role_class": "support_startup_install",
            "role_group": "support",
            "role_adjustment": -0.16 if not has_support_anchor else -0.03,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "startup_install_or_overview",
        }

    if safety_hits >= 2:
        return {
            "role_class": "support_safety",
            "role_group": "support",
            "role_adjustment": 0.03 if ("no_start" in classes and automatic_mode) else -0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "safety_support",
        }

    if strong_component_hits >= 1 or process_hits >= 1:
        return {
            "role_class": "core_mechanical",
            "role_group": "core",
            "role_adjustment": 0.06,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "component_or_process_anchor_without_subsystem",
        }

    return {
        "role_class": "collateral",
        "role_group": "collateral",
        "role_adjustment": -0.08,
        "matched_subsystems": sorted(matched_set),
        "role_reason": "collateral_or_weak",
    }


def _summarize_evidence_roles_for_prompt(
    q: str,
    citations: list[dict],
    *,
    symptom_profile: Optional[dict] = None,
    diagnostic_keywords: Optional[list[str]] = None,
    target_subsystems: Optional[list[str]] = None,
    max_items: int = 8,
) -> list[dict]:
    symptom_profile = dict(symptom_profile or _query_symptom_profile(q))
    inferred_components = _infer_machine_components(q)
    diagnostic_keywords = list(diagnostic_keywords or _collect_candidate_keywords(q, inferred_components))
    target_subsystems = list(target_subsystems or _root_cause_target_subsystems(q, inferred_components))

    out = []
    used = set()
    for c in citations or []:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        chunk_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=chunk_text,
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        out.append(
            {
                "citation_id": cid,
                "role_class": str(c.get("role_class") or role.get("role_class") or "collateral"),
                "role_group": str(c.get("role_group") or role.get("role_group") or "collateral"),
                "role_adjustment": round(float(c.get("role_adjustment", role.get("role_adjustment", 0.0)) or 0.0), 4),
                "matched_subsystems": list(c.get("matched_subsystems") or role.get("matched_subsystems") or []),
                "source_type": str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")),
                "diagnostic_score": round(float(c.get("candidate_score", c.get("diagnostic_score", c.get("retrieval_score", c.get("similarity", 0.0)))) or 0.0), 4),
                "snippet": re.sub(r"\s+", " ", (c.get("snippet") or chunk_text or "").strip())[:260],
            }
        )
        if len(out) >= max_items:
            break
    return out


def _llm_build_role_aware_diagnostic_evidence_matrix(
    q: str,
    citations: list[dict],
    max_causes: int,
) -> dict:
    if not q or not citations:
        return {}

    max_causes = max(1, min(int(max_causes or 1), 3))
    items = []
    used = set()
    for c in citations[: max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, max_causes + 5)]:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        items.append(
            {
                "citation_id": cid,
                "role_class": str(c.get("role_class") or "collateral"),
                "role_group": str(c.get("role_group") or "collateral"),
                "role_adjustment": round(float(c.get("role_adjustment", 0.0) or 0.0), 4),
                "matched_subsystems": list(c.get("matched_subsystems") or []),
                "evidence_family": _root_cause_evidence_family_key(c),
                "diagnostic_score": round(float(c.get("candidate_score", c.get("diagnostic_score", c.get("retrieval_score", c.get("similarity", 0.0)))) or 0.0), 4),
                "source_type": str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")),
                "snippet": re.sub(r"\s+", " ", (c.get("chunk_full") or c.get("snippet") or "").strip())[:360],
            }
        )

    if not items:
        return {}

    schema = {
        "name": "role_aware_diagnostic_evidence_matrix",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep_ids": {"type": "array", "items": {"type": "string"}},
                "discard_ids": {"type": "array", "items": {"type": "string"}},
                "cause_hypotheses": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cause": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "check_focus": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["cause", "evidence_ids", "check_focus"],
                    },
                },
            },
            "required": ["keep_ids", "discard_ids", "cause_hypotheses"],
        },
    }

    system_msg = (
        "You organize evidence for industrial root-cause diagnosis. "
        "Use the role metadata strictly. core roles are preferred for generic symptoms such as vibration, noise, or jams. "
        "support roles (lubrication, startup/install, safety, electrical/interlock, fluid power) may stay only when the symptom explicitly anchors them or when no stronger core evidence is available. "
        "For no-start and automatic-mode failures, support_electrical_interlock can be primary, but support_lubrication should remain secondary unless directly anchored. "
        "Maximize distinct causal families and avoid duplicate paraphrases. Keep only the most diagnostic evidence."
    )
    user_msg = (
        f"PROBLEM:\n{q}\n\n"
        f"ROLE_AWARE_CANDIDATES_JSON:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Return valid JSON."
    )

    return _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, ROOT_CAUSE_RESPONSE_MODEL, OPENAI_CHAT_MODEL],
        json_schema=schema,
        timeout=70,
    )


def _diagnostic_evidence_candidate_pipeline(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    max_causes: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "root_cause",
    base_threshold: float = ASK_SIM_THRESHOLD,
) -> dict:
    result = _diagnostic_evidence_pipeline(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=top_k,
        max_causes=max_causes,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
        planner_mode=planner_mode,
        base_threshold=base_threshold,
    )

    symptom_profile = dict(result.get("symptom_profile") or _query_symptom_profile(q))
    inferred_components = list(result.get("inferred_components") or _infer_machine_components(q))
    diagnostic_keywords = list(result.get("diagnostic_keywords") or _collect_candidate_keywords(q, inferred_components))
    target_subsystems = list(result.get("target_subsystems") or _root_cause_target_subsystems(q, inferred_components))
    classes = set(symptom_profile.get("classes") or [])

    raw_pool = list(result.get("candidate_pool") or [])
    if not raw_pool:
        raw_pool = list(result.get("candidates") or []) + list(result.get("citations") or [])
    raw_pool = _ensure_candidate_retrieval_fields(
        raw_pool,
        query_terms=_planner_query_term_set(q, result.get("planner") or {}),
        query_style=str((result.get("planner") or {}).get("query_style") or "").strip().lower(),
        query_token_count=_count_query_tokens(q),
    )
    if not raw_pool:
        result["role_summary"] = []
        return result

    rescored = []
    for item in raw_pool:
        cc = dict(item)
        chunk_text = (cc.get("chunk_full") or cc.get("snippet") or "").strip()
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=chunk_text,
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        candidate_score = float(cc.get("diagnostic_score", cc.get("retrieval_score", cc.get("similarity", 0.0))) or 0.0)
        candidate_score += float(role.get("role_adjustment", 0.0) or 0.0)

        if symptom_profile.get("generic_symptom") and role.get("role_group") == "core":
            candidate_score += ROOT_CAUSE_CANDIDATE_CORE_PROMOTION
        if symptom_profile.get("generic_symptom") and role.get("role_group") == "support" and not symptom_profile.get("has_support_anchor") and "no_start" not in classes:
            candidate_score -= ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY * 0.45
        if role.get("role_class") == "support_lubrication" and classes & {"vibration", "noise", "jam"} and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY
        if role.get("role_class") == "support_lubrication" and "no_start" in classes and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY
        if role.get("role_class") == "support_startup_install" and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY
        if role.get("role_class") == "support_safety" and not ("no_start" in classes and symptom_profile.get("automatic_mode")) and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY
        if role.get("role_class") == "collateral":
            candidate_score -= 0.04

        cc.update(role)
        cc["candidate_score"] = candidate_score
        cc["retrieval_score"] = candidate_score
        rescored.append(cc)

    rescored.sort(
        key=lambda x: (
            -float(x.get("candidate_score", x.get("diagnostic_score", x.get("retrieval_score", x.get("similarity", 0.0)))) or 0.0),
            -float(x.get("similarity", 0.0) or 0.0),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )

    working_pool = _dedup_root_cause_candidates_semantic(
        rescored,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 2, top_k * 3, ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K + 4),
    )
    working_pool = _prioritize_root_cause_coverage(
        working_pool,
        max_items=max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, top_k + 3),
    )

    matrix = {}
    if ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX:
        try:
            matrix = _llm_build_role_aware_diagnostic_evidence_matrix(
                q=q,
                citations=working_pool[: max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, top_k + 2)],
                max_causes=max_causes,
            )
        except Exception:
            matrix = {}

    if not matrix:
        matrix = dict(result.get("diagnostic_matrix") or {})

    prompt_citations = _select_prompt_citations_from_matrix(
        working_pool,
        matrix,
        max_prompt=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
    )
    if not prompt_citations:
        prompt_citations = _lock_final_citations(
            selected_citations=working_pool[: max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k)],
            ranked_candidates=working_pool + rescored,
            top_k=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
            diagnostic_mode=True,
            query_token_count=_count_query_tokens(q),
        )

    final_citations = _lock_final_citations(
        selected_citations=prompt_citations,
        ranked_candidates=working_pool + rescored,
        top_k=max(top_k, min(len(prompt_citations) or top_k, top_k + 2)),
        diagnostic_mode=True,
        query_token_count=_count_query_tokens(q),
    )

    result["candidates"] = rescored
    result["candidate_pool"] = working_pool
    result["prompt_citations"] = prompt_citations
    result["citations"] = _dedup_citations_by_snippet(
        list(prompt_citations or []) + list(final_citations or []),
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 3),
    )
    result["diagnostic_matrix"] = matrix or {}
    result["role_summary"] = _summarize_evidence_roles_for_prompt(
        q=q,
        citations=prompt_citations or working_pool,
        symptom_profile=symptom_profile,
        diagnostic_keywords=diagnostic_keywords,
        target_subsystems=target_subsystems,
        max_items=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
    )
    return result


def _cause_role_from_response_cause(
    q: str,
    cause: dict,
    by_id: dict[str, dict],
    *,
    symptom_profile: Optional[dict] = None,
    diagnostic_keywords: Optional[list[str]] = None,
    target_subsystems: Optional[list[str]] = None,
) -> tuple[str, str]:
    symptom_profile = dict(symptom_profile or _query_symptom_profile(q))
    inferred_components = _infer_machine_components(q)
    diagnostic_keywords = list(diagnostic_keywords or _collect_candidate_keywords(q, inferred_components))
    target_subsystems = list(target_subsystems or _root_cause_target_subsystems(q, inferred_components))

    roles = []
    for cid in cause.get("citations") or []:
        item = by_id.get(str(cid or "").strip())
        if not item:
            continue
        if item.get("role_class") and item.get("role_group"):
            roles.append((str(item.get("role_class") or "collateral"), str(item.get("role_group") or "collateral")))
            continue
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=(item.get("chunk_full") or item.get("snippet") or ""),
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        roles.append((str(role.get("role_class") or "collateral"), str(role.get("role_group") or "collateral")))

    if not roles:
        txt = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(cause.get("cause") or ""))).strip().lower()
        if any(m in txt for m in ["lubric", "olio", "grease", "lubr"]):
            return "support_lubrication", "support"
        if any(m in txt for m in ["interlock", "sicur", "safety", "door", "guard", "selector", "mode", "fasi", "power", "phase", "voltage", "panel", "quadro"]):
            return "support_electrical_interlock", "support"
        if any(m in txt for m in ["startup", "install", "commission", "avviamento", "messa in servizio"]):
            return "support_startup_install", "support"
        if any(m in txt for m in ["gear", "gearbox", "bearing", "roller", "guide", "slide", "press", "eccentric", "cam", "transmission", "motor", "ridutt", "cuscinet", "rullo", "slitta", "guida", "pressa", "camme", "trasmission"]):
            return "core_mechanical", "core"
        return "collateral", "collateral"

    if any(group == "core" for _, group in roles):
        first = next((r for r in roles if r[1] == "core"), roles[0])
        return first
    if any(role == "support_electrical_interlock" for role, _ in roles):
        return "support_electrical_interlock", "support"
    return roles[0]


def _enforce_diverse_root_cause_hypotheses(
    *,
    q: str,
    result: dict,
    citations: list[dict],
    retrieval: dict,
    max_causes: int,
    response_language: str,
) -> tuple[dict, list[dict]]:
    merged_result, merged_citations = _merge_matrix_supported_causes(
        result=result,
        citations=citations,
        matrix=retrieval.get("diagnostic_matrix") or {},
        max_causes=max_causes,
        response_language=response_language,
    )

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in (retrieval.get("candidate_pool") or []) + list(merged_citations or [])
        if c.get("citation_id")
    }
    symptom_profile = retrieval.get("symptom_profile") or _query_symptom_profile(q)
    classes = set(symptom_profile.get("classes") or [])
    has_support_anchor = bool(symptom_profile.get("has_support_anchor"))
    automatic_mode = bool(symptom_profile.get("automatic_mode"))

    rows = []
    for cause in merged_result.get("possible_causes") or []:
        if not isinstance(cause, dict):
            continue
        role_class, role_group = _cause_role_from_response_cause(
            q=q,
            cause=cause,
            by_id=by_id,
            symptom_profile=symptom_profile,
            diagnostic_keywords=retrieval.get("diagnostic_keywords") or [],
            target_subsystems=retrieval.get("target_subsystems") or [],
        )
        score = 0.30
        if role_group == "core":
            score += 0.60
        elif role_class == "support_electrical_interlock" and "no_start" in classes:
            score += 0.48
        elif role_group == "support":
            score += 0.18
        else:
            score += 0.05

        label_specificity = _cause_label_specificity_score(str(cause.get("cause") or ""))
        score += 0.10 * label_specificity
        score += min(0.10, 0.03 * len(cause.get("citations") or []))

        if symptom_profile.get("generic_symptom") and classes & {"vibration", "noise", "jam"} and role_group == "support" and not has_support_anchor:
            score -= 0.32
        if "no_start" in classes and role_class == "support_lubrication" and not has_support_anchor:
            score -= 0.34
        if "no_start" in classes and automatic_mode and role_class == "support_safety":
            score += 0.06

        rows.append(
            {
                "cause": dict(cause),
                "role_class": role_class,
                "role_group": role_group,
                "score": score,
                "label_key": _normalized_cause_label_key(str(cause.get("cause") or "")),
            }
        )

    dedup = {}
    for row in rows:
        key = row["label_key"] or str(row["cause"].get("cause") or "").strip().lower()
        prev = dedup.get(key)
        if prev is None or float(row["score"]) > float(prev["score"]):
            dedup[key] = row

    ordered = sorted(
        dedup.values(),
        key=lambda x: (
            -float(x.get("score", 0.0)),
            0 if str(x.get("role_group") or "") == "core" else 1,
            str((x.get("cause") or {}).get("cause") or ""),
        ),
    )

    final_causes = []
    final_citation_ids = []
    used_roles = set()
    used_ids = set()

    for row in ordered:
        role_key = (str(row.get("role_group") or ""), str(row.get("role_class") or ""))
        if role_key in used_roles and len(final_causes) >= 1:
            continue
        final_causes.append(dict(row["cause"]))
        used_roles.add(role_key)
        for cid in row["cause"].get("citations") or []:
            cid = str(cid or "").strip()
            if cid and cid not in used_ids:
                used_ids.add(cid)
                final_citation_ids.append(cid)
        if len(final_causes) >= max_causes:
            break

    if not final_causes:
        return merged_result, merged_citations

    for idx, cause in enumerate(final_causes, start=1):
        cause["rank"] = idx

    by_id_grounded = {str(c.get("citation_id") or "").strip(): c for c in merged_citations if c.get("citation_id")}
    final_citations = [by_id_grounded[cid] for cid in final_citation_ids if cid in by_id_grounded]

    final_result = dict(merged_result)
    final_result["possible_causes"] = final_causes[:max_causes]
    final_result["recommended_next_checks"] = _unique_non_empty_strings(
        [chk for row in final_causes for chk in (row.get("checks") or [])],
        limit=6,
    )
    return final_result, final_citations or merged_citations


def _infer_response_citation_roles(q: str, citations: list[dict]) -> list[dict]:
    symptom_profile = _query_symptom_profile(q)
    inferred_components = _infer_machine_components(q)
    diagnostic_keywords = _collect_candidate_keywords(q, inferred_components)
    target_subsystems = _root_cause_target_subsystems(q, inferred_components)
    out = []
    for c in citations or []:
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=(c.get("chunk_full") or c.get("snippet") or ""),
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        row = dict(c)
        row.update(role)
        out.append(row)
    return out


def _root_cause_response_proxy_score(q: str, response: dict) -> dict:
    response = dict(response or {})
    status = str(response.get("status") or "").strip().lower()
    causes = [c for c in (response.get("possible_causes") or []) if isinstance(c, dict)]
    citations = list(response.get("citations") or [])
    language = _simple_query_language(q)
    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])

    score = 0.0
    hard_fail = False
    notes: list[str] = []

    if status == "answered":
        score += 0.55
    elif status == "no_sources":
        return {"score": 0.0, "hard_fail": False, "top_role_class": "none", "top_role_group": "none", "notes": ["no_sources"]}
    else:
        return {"score": 0.0, "hard_fail": True, "top_role_class": "none", "top_role_group": "none", "notes": ["invalid_status"]}

    if not causes:
        hard_fail = True
        notes.append("no_causes")
        score -= 0.35

    if citations:
        score += min(0.12, 0.05 * len(citations))
    else:
        hard_fail = True
        notes.append("no_citations")
        score -= 0.18

    problem_summary = str(response.get("problem_summary") or "")
    cause_text = " | ".join(str(c.get("cause") or "") for c in causes)
    if _looks_like_target_language(problem_summary + " " + cause_text, language):
        score += 0.10
    else:
        score -= 0.20
        hard_fail = True
        notes.append("language_mismatch")

    label_scores = [_cause_label_specificity_score(str(c.get("cause") or "")) for c in causes[:3]]
    if label_scores:
        score += 0.12 * (sum(label_scores) / len(label_scores))

    role_citations = _infer_response_citation_roles(q, citations)
    role_map = {str(c.get("citation_id") or "").strip(): c for c in role_citations if c.get("citation_id")}
    top_role_class = "collateral"
    top_role_group = "collateral"
    if causes:
        top_role_class, top_role_group = _cause_role_from_response_cause(
            q=q,
            cause=causes[0],
            by_id=role_map,
            symptom_profile=profile,
        )
    elif role_citations:
        top_role_class = str(role_citations[0].get("role_class") or "collateral")
        top_role_group = str(role_citations[0].get("role_group") or "collateral")

    if profile.get("generic_symptom") and classes & {"vibration", "noise", "jam"}:
        if top_role_group == "core":
            score += 0.22
        elif top_role_group == "support" and not profile.get("has_support_anchor"):
            score -= 0.28
            notes.append("support_dominance_generic_symptom")

    if "no_start" in classes:
        if top_role_class in {"support_electrical_interlock", "support_safety"} or top_role_group == "core":
            score += 0.18
        elif top_role_class == "support_lubrication" and not profile.get("has_support_anchor"):
            score -= 0.24
            notes.append("no_start_lubrication_dominance")

    if len(causes) >= 2:
        distinct = len({_normalized_cause_label_key(str(c.get("cause") or "")) for c in causes})
        if distinct >= 2:
            score += 0.08

    top_cause_txt = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(causes[0].get("cause") or ""))).strip().lower() if causes else ""
    if any(bad in top_cause_txt for bad in ["mancato avviamento intenzionale", "generic problem", "possible cause", "anomalia generica"]):
        score -= 0.24
        notes.append("generic_or_bad_top_cause")

    return {
        "score": max(0.0, min(1.25, score)),
        "hard_fail": hard_fail,
        "top_role_class": top_role_class,
        "top_role_group": top_role_group,
        "notes": notes,
    }


def _ask_response_proxy_score(q: str, response: dict) -> dict:
    response = dict(response or {})
    status = str(response.get("status") or "").strip().lower()
    answer = str(response.get("answer") or "")
    citations = list(response.get("citations") or [])
    language = _simple_query_language(q)
    profile = _query_symptom_profile(q)
    classes = set(profile.get("classes") or [])

    score = 0.0
    hard_fail = False
    notes: list[str] = []

    if status == "answered":
        score += 0.52
    elif status == "no_sources":
        return {"score": 0.0, "hard_fail": False, "top_role_class": "none", "top_role_group": "none", "notes": ["no_sources"]}
    else:
        return {"score": 0.0, "hard_fail": True, "top_role_class": "none", "top_role_group": "none", "notes": ["invalid_status"]}

    if not answer:
        score -= 0.28
        hard_fail = True
        notes.append("empty_answer")
    else:
        if 30 <= len(answer) <= 700:
            score += 0.08

    if citations:
        score += min(0.12, 0.05 * len(citations))
    else:
        score -= 0.18
        hard_fail = True
        notes.append("no_citations")

    if _looks_like_target_language(answer, language):
        score += 0.10
    else:
        score -= 0.18
        hard_fail = True
        notes.append("language_mismatch")

    role_citations = _infer_response_citation_roles(q, citations)
    top_role_class = str(role_citations[0].get("role_class") or "collateral") if role_citations else "none"
    top_role_group = str(role_citations[0].get("role_group") or "collateral") if role_citations else "none"

    installation_false_positive = False
    if profile.get("generic_symptom") and classes & {"vibration", "noise", "jam"}:
        installation_false_positive = _looks_like_installation_positioning_false_positive(answer) or any(
            _looks_like_installation_positioning_false_positive((c.get("snippet") or "") + "\n" + (c.get("chunk_full") or ""))
            for c in (role_citations or citations)
        )
        if installation_false_positive and not profile.get("has_support_anchor"):
            score -= 0.42
            hard_fail = True
            notes.append("installation_false_positive_generic_symptom")
        elif top_role_group == "core":
            score += 0.18
        elif top_role_group == "support" and not profile.get("has_support_anchor"):
            score -= 0.24
            notes.append("support_dominance_generic_symptom")

    if "no_start" in classes:
        if top_role_class in {"support_electrical_interlock", "support_safety"} or top_role_group == "core":
            score += 0.16
        elif top_role_class == "support_lubrication" and not profile.get("has_support_anchor"):
            score -= 0.22
            notes.append("no_start_lubrication_dominance")

    if any(bad in answer.lower() for bad in ["i cannot find enough information", "non trovo informazioni sufficienti"]):
        score -= 0.10

    return {
        "score": max(0.0, min(1.20, score)),
        "hard_fail": hard_fail,
        "top_role_class": top_role_class,
        "top_role_group": top_role_group,
        "notes": notes,
    }


def _should_attempt_root_cause_candidate(q: str, baseline_response: dict) -> bool:
    if not (RESPONSE_ARB_ENABLED and ROOT_CAUSE_CANDIDATE_ENABLED):
        return False
    profile = _query_symptom_profile(q)
    language = _simple_query_language(q)
    baseline_eval = _root_cause_response_proxy_score(q, baseline_response)

    if str((baseline_response or {}).get("status") or "").strip().lower() != "answered":
        return True
    if baseline_eval.get("hard_fail"):
        return True
    if language == "en":
        return True
    if profile.get("generic_symptom") or profile.get("automatic_mode"):
        return True
    if float(baseline_eval.get("score", 0.0) or 0.0) < 0.84:
        return True
    if str(baseline_eval.get("top_role_group") or "") == "support" and not profile.get("has_support_anchor"):
        return True
    return False


def _is_lookup_or_identifier_query(q: str) -> bool:
    if _q_has_any(q, URL_HINTS) or _q_has_any(q, EMAIL_HINTS) or _q_has_any(q, PHONE_HINTS):
        return True
    if _extract_code_tokens(q):
        return True
    return False


def _should_attempt_ask_candidate(q: str, baseline_response: dict) -> bool:
    if not (RESPONSE_ARB_ENABLED and ASK_CANDIDATE_ENABLED):
        return False
    if _should_route_ask_through_root_cause(q):
        return False
    if _is_lookup_or_identifier_query(q):
        return False
    profile = _query_symptom_profile(q)
    language = _simple_query_language(q)
    baseline_eval = _ask_response_proxy_score(q, baseline_response)

    if str((baseline_response or {}).get("status") or "").strip().lower() != "answered":
        return True
    if baseline_eval.get("hard_fail"):
        return True
    if language == "en":
        return True
    if profile.get("generic_symptom") or profile.get("automatic_mode") or bool(profile.get("classes")):
        return True
    if float(baseline_eval.get("score", 0.0) or 0.0) < 0.82:
        return True
    if str(baseline_eval.get("top_role_group") or "") == "support" and not profile.get("has_support_anchor"):
        return True
    return False


def _attach_arbiter_debug(
    resp: dict,
    *,
    baseline_eval: dict,
    candidate_eval: Optional[dict],
    chosen: str,
    candidate_attempted: bool,
    candidate_error: Optional[str] = None,
) -> dict:
    if not isinstance(resp, dict):
        return resp
    debug = dict(resp.get("debug") or {})
    debug["arbiter"] = {
        "candidate_attempted": bool(candidate_attempted),
        "chosen": chosen,
        "baseline_proxy": baseline_eval,
        "candidate_proxy": candidate_eval,
        "candidate_error": candidate_error,
    }
    resp["debug"] = debug
    return resp


def _choose_root_cause_response(q: str, baseline_response: dict, candidate_response: Optional[dict], *, debug: bool) -> dict:
    baseline_eval = _root_cause_response_proxy_score(q, baseline_response)
    candidate_eval = _root_cause_response_proxy_score(q, candidate_response or {}) if candidate_response else None

    chosen = baseline_response
    chosen_name = "baseline"
    if candidate_response and candidate_eval and not candidate_eval.get("hard_fail"):
        baseline_score = float(baseline_eval.get("score", 0.0) or 0.0)
        candidate_score = float(candidate_eval.get("score", 0.0) or 0.0)
        if candidate_score > baseline_score + ROOT_CAUSE_ARB_MIN_DELTA:
            chosen = candidate_response
            chosen_name = "candidate"
        elif not RESPONSE_ARB_KEEP_BASELINE_ON_TIE and candidate_score >= baseline_score:
            chosen = candidate_response
            chosen_name = "candidate"

    if debug:
        chosen = _attach_arbiter_debug(
            chosen,
            baseline_eval=baseline_eval,
            candidate_eval=candidate_eval,
            chosen=chosen_name,
            candidate_attempted=bool(candidate_response),
        )
    return chosen


def _choose_ask_response(q: str, baseline_response: dict, candidate_response: Optional[dict], *, debug: bool) -> dict:
    baseline_eval = _ask_response_proxy_score(q, baseline_response)
    candidate_eval = _ask_response_proxy_score(q, candidate_response or {}) if candidate_response else None

    chosen = baseline_response
    chosen_name = "baseline"
    if candidate_response and candidate_eval and not candidate_eval.get("hard_fail"):
        baseline_score = float(baseline_eval.get("score", 0.0) or 0.0)
        candidate_score = float(candidate_eval.get("score", 0.0) or 0.0)
        if candidate_score > baseline_score + ASK_ARB_MIN_DELTA:
            chosen = candidate_response
            chosen_name = "candidate"
        elif not RESPONSE_ARB_KEEP_BASELINE_ON_TIE and candidate_score >= baseline_score:
            chosen = candidate_response
            chosen_name = "candidate"

    if debug:
        chosen = _attach_arbiter_debug(
            chosen,
            baseline_eval=baseline_eval,
            candidate_eval=candidate_eval,
            chosen=chosen_name,
            candidate_attempted=bool(candidate_response),
        )
    return chosen


def _generate_ask_candidate_grounded_points(
    *,
    q: str,
    planner: dict,
    response_language: str,
    company_id: str,
    citations: list[dict],
    role_summary: list[dict],
    diagnostic_matrix: dict,
    allow_no_sources: bool,
) -> tuple[str, list[dict]]:
    if not citations:
        return "no_sources", []

    prompt_citations = _enrich_ask_prompt_citations(
        company_id=company_id,
        citations=citations,
        max_manual_expansions=2,
        radius=1,
    )
    sources_block = _build_sources_block_from_citations(
        prompt_citations,
        max_context_chars=ASK_MAX_CONTEXT_CHARS,
        prefer_chunk_full=True,
    )
    roles_json = json.dumps(role_summary or [], ensure_ascii=False)
    matrix_json = json.dumps(diagnostic_matrix or {}, ensure_ascii=False)

    if allow_no_sources:
        system_msg = (
            "You are a technical documentation assistant for machinery and industrial equipment. "
            "Use ONLY the provided sources, evidence-role summary, and evidence matrix. "
            "Always answer the user's question directly. For generic symptoms or technical issues, prefer core mechanical or core process evidence before support-only evidence. "
            "Do not center lubrication, startup/install, or safety unless the question anchors them or the evidence matrix explicitly shows they are primary. "
            "If the documents do not state the requested thing directly, say that explicitly and then report the closest grounded evidence. "
            "Always reply in the requested response language. Return 1 to 3 short grounded points only."
        )
        schema = _ask_response_schema()
    else:
        system_msg = (
            "You are a technical documentation assistant for machinery and industrial equipment. "
            "Use ONLY the provided sources, evidence-role summary, and evidence matrix. "
            "Always reply in the requested response language. Return 1 to 3 very short grounded points. "
            "If the requested thing is not stated directly, say that explicitly and then report the closest grounded evidence."
        )
        schema = _ask_rescue_response_schema()

    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"NORMALIZED_QUESTION:\n{planner.get('normalized_query') or q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"ROLE_AWARE_EVIDENCE_JSON:\n{roles_json}\n\n"
        f"DIAGNOSTIC_EVIDENCE_MATRIX_JSON:\n{matrix_json}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        "Return valid JSON. Use only citation ids present in the sources."
    )

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=70,
        )
    except Exception:
        return "no_sources", []

    if allow_no_sources:
        answer_status = str((parsed or {}).get("answer_status") or "").strip().lower()
        grounded_points = list((parsed or {}).get("grounded_points") or [])
        return (answer_status or "no_sources"), grounded_points

    grounded_points = list((parsed or {}).get("grounded_points") or [])
    return ("answered" if grounded_points else "no_sources"), grounded_points


def _ask_v1_candidate_impl(
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
    candidate_k = max(top_k, min(80, max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 10)))

    response_language = _select_response_language(q)
    no_sources_text = _localized_no_sources(response_language)
    symptom_profile = _query_symptom_profile(q)
    technical_candidate = bool(symptom_profile.get("classes")) or response_language == "en" or _count_query_tokens(q) >= 4

    if technical_candidate:
        retrieval = _diagnostic_evidence_candidate_pipeline(
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=candidate_k,
            top_k=max(top_k, min(ASK_CANDIDATE_MATRIX_TOP_K, top_k + 1)),
            max_causes=2,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
            debug=payload.debug,
            planner_mode="ask_candidate",
            base_threshold=min(ASK_SIM_THRESHOLD, ASK_SHORT_QUERY_SIM_THRESHOLD),
        )
    else:
        retrieval = _shared_semantic_retrieval(
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=candidate_k,
            top_k=top_k,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=bubble_document_id,
            debug=payload.debug,
            planner_mode="ask",
            base_threshold=ASK_SIM_THRESHOLD,
            diagnostic_mode=False,
        )
        retrieval["role_summary"] = _summarize_evidence_roles_for_prompt(
            q=q,
            citations=list(retrieval.get("citations") or []),
            max_items=max(ASK_CANDIDATE_PROMPT_TOP_K, top_k),
        )

    planner = retrieval.get("planner") or {}
    response_language = _select_response_language(q, planner=planner)
    citations = list(retrieval.get("prompt_citations") or retrieval.get("citations") or [])
    sim_max = retrieval.get("similarity_max")

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "query_plan": planner,
                "similarity_max": sim_max,
                "effective_ask_threshold": retrieval.get("effective_threshold"),
                "candidate_mode": technical_candidate,
                "role_summary": retrieval.get("role_summary") or [],
                "diagnostic_matrix": retrieval.get("diagnostic_matrix") or {},
            }
        return resp

    if not citations:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": no_sources_text,
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
                "chat_model": DIAGNOSTIC_EVIDENCE_MODEL,
            }
        )

    answer_status, grounded_points = _generate_ask_candidate_grounded_points(
        q=q,
        planner=planner,
        response_language=response_language,
        company_id=company_id,
        citations=citations,
        role_summary=list(retrieval.get("role_summary") or []),
        diagnostic_matrix=dict(retrieval.get("diagnostic_matrix") or {}),
        allow_no_sources=True,
    )

    if answer_status == "no_sources" or not grounded_points:
        answer_status, grounded_points = _generate_ask_candidate_grounded_points(
            q=q,
            planner=planner,
            response_language=response_language,
            company_id=company_id,
            citations=citations,
            role_summary=list(retrieval.get("role_summary") or []),
            diagnostic_matrix=dict(retrieval.get("diagnostic_matrix") or {}),
            allow_no_sources=False,
        )

    if answer_status == "no_sources" or not grounded_points:
        answer, final_citations = _extractive_fallback_answer(
            citations=citations,
            response_language=response_language,
            max_points=min(2, top_k),
        )
    else:
        answer, final_citations = _render_grounded_answer_points(
            grounded_points=grounded_points,
            citations=citations,
            max_points=min(3, top_k),
        )

    if not answer or not final_citations:
        return _finalize(
            {
                "ok": True,
                "status": "no_sources",
                "answer": no_sources_text,
                "citations": [],
                "rg_links": [],
                "top_k": top_k,
                "similarity_max": sim_max,
                "chat_model": DIAGNOSTIC_EVIDENCE_MODEL,
            }
        )

    if not _looks_like_target_language(answer, response_language):
        answer = _translate_text_preserving_citations(answer, response_language)

    response_citations = _sanitize_citations_for_response(final_citations)
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
            "answer": answer,
            "citations": response_citations,
            "rg_links": rg_links,
            "top_k": top_k,
            "similarity_max": sim_max,
            "chat_model": DIAGNOSTIC_EVIDENCE_MODEL,
        }
    )


def _root_cause_v1_candidate_impl(
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
    candidate_k = max(top_k, min(90, max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 11)))

    query_signal_summary = _root_cause_query_signal_summary(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        bubble_document_id=bubble_document_id,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        debug=payload.debug,
    )
    query_fail_closed = _should_fail_closed_root_cause_query(query_signal_summary)
    prelim = query_signal_summary.get("preliminary_retrieval") or {}
    preliminary_similarity_max = prelim.get("similarity_max")

    if query_fail_closed:
        resp = {
            "ok": True,
            "status": "no_sources",
            "symptom": q,
            "problem_summary": "",
            "possible_causes": [],
            "recommended_next_checks": [],
            "citations": [],
            "rg_links": [],
            "top_k": top_k,
            "similarity_max": preliminary_similarity_max,
        }
        if payload.debug:
            resp["debug"] = {
                "query_signal_summary": query_signal_summary,
                "query_fail_closed": query_fail_closed,
                "candidate_mode": True,
            }
        return resp

    retrieval = _diagnostic_evidence_candidate_pipeline(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=top_k,
        max_causes=max_causes,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=bubble_document_id,
        debug=payload.debug,
        planner_mode="root_cause_candidate",
        base_threshold=ASK_SIM_THRESHOLD,
    )

    planner = retrieval.get("planner") or {}
    sim_max = retrieval.get("similarity_max")
    citations = list(retrieval.get("citations") or [])
    response_language = _select_response_language(q, planner=planner)

    def _finalize(resp: dict) -> dict:
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": bubble_document_id,
                "document_ids": doc_ids,
                "query_signal_summary": query_signal_summary,
                "query_fail_closed": query_fail_closed,
                "query_plan": planner,
                "similarity_max": sim_max,
                "role_summary": retrieval.get("role_summary") or [],
                "diagnostic_matrix": retrieval.get("diagnostic_matrix") or {},
                "candidate_mode": True,
            }
        return resp

    if not citations:
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

    prompt_citations = []
    for c in (retrieval.get("prompt_citations") or citations):
        cc = dict(c)
        cc["chunk_full"] = (cc.get("chunk_full") or cc.get("snippet") or "").strip()[:1800]
        cc["snippet"] = (cc.get("snippet") or cc.get("chunk_full") or "").strip()
        prompt_citations.append(cc)
    prompt_citations = prompt_citations[: max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k)]

    if not prompt_citations:
        prompt_citations = list(citations[: max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k)])

    sources_block = _build_sources_block_from_citations(
        prompt_citations,
        max_context_chars=ASK_MAX_CONTEXT_CHARS,
        prefer_chunk_full=True,
    )

    matrix = retrieval.get("diagnostic_matrix") or {}
    role_summary = retrieval.get("role_summary") or []

    system_msg = (
        "You are a root-cause assistant for technical equipment and machine documentation. "
        "Use ONLY the provided sources, evidence-role summary, and evidence matrix. "
        "Work domain-agnostically: do not assume a sector, machine family, subsystem taxonomy, or standard failure mode unless the sources support it. "
        "core_process and core_mechanical evidence outrank support-only evidence for generic symptoms such as vibration, noise, or jams. "
        "support_lubrication, support_startup_install, and support_safety must not become rank-1 causes for generic symptoms unless the evidence matrix explicitly shows that they are primary and no stronger core hypothesis exists. "
        "For no-start and automatic-mode failures, support_electrical_interlock may be primary; support_lubrication should remain secondary unless directly anchored by the symptom. "
        "Preserve more than one cause when the evidence matrix contains distinct, separately supported hypotheses. "
        "Each cause must be a short canonical technical label, 3 to 10 words, noun-phrase style, with no trailing period. Always reply in the same language as the user query."
    )
    user_msg = (
        f"USER_PROBLEM:\n{q}\n\n"
        f"NORMALIZED_PROBLEM:\n{planner.get('normalized_query') or q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"ROLE_AWARE_EVIDENCE_JSON:\n{json.dumps(role_summary, ensure_ascii=False)}\n\n"
        f"DIAGNOSTIC_EVIDENCE_MATRIX_JSON:\n{json.dumps(matrix, ensure_ascii=False)}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        "Return valid JSON. Use only citation_id values present in the sources."
    )

    try:
        result_json = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ROOT_CAUSE_RESPONSE_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_root_cause_response_schema(max_causes=max_causes),
            timeout=90,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")

    if not (result_json or {}).get("possible_causes") and matrix:
        result_json = _fallback_root_cause_result_from_matrix(
            q=planner.get("normalized_query") or q,
            matrix=matrix,
            citations=citations,
            max_causes=max_causes,
            response_language=response_language,
        )

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
    grounded_result = _canonicalize_root_cause_labels(
        grounded_result,
        grounded_citations,
        language=response_language,
    )
    grounded_result, grounded_citations = _lock_root_cause_result(
        grounded_result,
        grounded_citations,
        max_causes=max_causes,
    )
    grounded_result, grounded_citations = _enforce_diverse_root_cause_hypotheses(
        q=q,
        result=grounded_result,
        citations=grounded_citations,
        retrieval=retrieval,
        max_causes=max_causes,
        response_language=response_language,
    )
    grounded_result = _canonicalize_root_cause_labels(
        grounded_result,
        grounded_citations,
        language=response_language,
    )

    if not grounded_result.get("problem_summary"):
        grounded_result["problem_summary"] = planner.get("normalized_query") or q

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
                "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
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
            "chat_model": ROOT_CAUSE_RESPONSE_MODEL,
        }
    )


@app.post("/v1/ai/ask")
def ask_v1(
    payload: AskRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    baseline_response = _ask_v1_baseline_impl(payload, x_ai_internal_secret)

    if not _should_attempt_ask_candidate(payload.query or "", baseline_response):
        return _strip_internal_response_artifacts(baseline_response)

    candidate_response = None
    candidate_error = None
    try:
        candidate_response = _ask_v1_candidate_impl(payload, x_ai_internal_secret)
    except Exception as e:
        candidate_error = str(e)
        candidate_response = None

    chosen = _choose_ask_response(
        payload.query or "",
        baseline_response,
        candidate_response,
        debug=bool(payload.debug),
    )

    if payload.debug and candidate_error:
        chosen = _attach_arbiter_debug(
            chosen,
            baseline_eval=_ask_response_proxy_score(payload.query or "", baseline_response),
            candidate_eval=_ask_response_proxy_score(payload.query or "", candidate_response or {}) if candidate_response else None,
            chosen=(chosen.get("debug") or {}).get("arbiter", {}).get("chosen", "baseline"),
            candidate_attempted=True,
            candidate_error=candidate_error,
        )

    return _strip_internal_response_artifacts(chosen)


@app.post("/v1/ai/root-cause")
def root_cause_v1(
    payload: RootCauseRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    baseline_response = _root_cause_v1_baseline_impl(payload, x_ai_internal_secret)

    if not _should_attempt_root_cause_candidate(payload.query or "", baseline_response):
        return _strip_internal_response_artifacts(baseline_response)

    candidate_response = None
    candidate_error = None
    try:
        candidate_response = _root_cause_v1_candidate_impl(payload, x_ai_internal_secret)
    except Exception as e:
        candidate_error = str(e)
        candidate_response = None

    chosen = _choose_root_cause_response(
        payload.query or "",
        baseline_response,
        candidate_response,
        debug=bool(payload.debug),
    )

    if payload.debug and candidate_error:
        chosen = _attach_arbiter_debug(
            chosen,
            baseline_eval=_root_cause_response_proxy_score(payload.query or "", baseline_response),
            candidate_eval=_root_cause_response_proxy_score(payload.query or "", candidate_response or {}) if candidate_response else None,
            chosen=(chosen.get("debug") or {}).get("arbiter", {}).get("chosen", "baseline"),
            candidate_attempted=True,
            candidate_error=candidate_error,
        )

    return _strip_internal_response_artifacts(chosen)


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

# Commit: feat(v8): preserve v4 hot paths and add opt-in shadow reasoning endpoints
V8_SHADOW_REASONING_ENABLED = (os.environ.get("MM_V8_SHADOW_REASONING_ENABLED") or "1").strip() != "0"
V8_SHADOW_ASK_ENABLED = (os.environ.get("MM_V8_SHADOW_ASK_ENABLED") or "1").strip() != "0"
V8_SHADOW_ROOT_CAUSE_ENABLED = (os.environ.get("MM_V8_SHADOW_ROOT_CAUSE_ENABLED") or "1").strip() != "0"
V8_SHADOW_MODEL = (os.environ.get("MM_V8_SHADOW_MODEL") or DIAGNOSTIC_EVIDENCE_MODEL or OPENAI_CHAT_MODEL).strip()
V8_SHADOW_TIMEOUT = int(os.environ.get("MM_V8_SHADOW_TIMEOUT_SECONDS", "20"))
V8_SHADOW_MAX_CITATIONS = int(os.environ.get("MM_V8_SHADOW_MAX_CITATIONS", "4"))
V8_SHADOW_MIN_ASK_PROXY = float(os.environ.get("MM_V8_SHADOW_MIN_ASK_PROXY", "0.78"))
V8_SHADOW_MIN_ROOT_PROXY = float(os.environ.get("MM_V8_SHADOW_MIN_ROOT_PROXY", "0.86"))
V8_SHADOW_MAX_CAUSES = int(os.environ.get("MM_V8_SHADOW_MAX_CAUSES", "3"))


def _v8_shadow_compact_citations(citations: list[dict], *, max_items: int = 4, snippet_chars: int = 220) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for c in citations or []:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "citation_id": cid,
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "snippet": re.sub(r"\s+", " ", str(c.get("snippet") or c.get("chunk_full") or "").strip())[:snippet_chars],
            }
        )
        if len(out) >= max_items:
            break
    return out


def _v8_shadow_ask_schema() -> dict:
    return {
        "name": "mm_v8_shadow_ask",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "grounded_brief": {"type": "string"},
                "evidence_map": {
                    "type": "array",
                    "maxItems": V8_SHADOW_MAX_CITATIONS,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "citation_id": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["citation_id", "why_it_matters"],
                    },
                },
            },
            "required": ["confidence", "grounded_brief", "evidence_map"],
        },
    }


def _v8_shadow_root_cause_schema() -> dict:
    return {
        "name": "mm_v8_shadow_root_cause",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "diagnostic_brief": {"type": "string"},
                "cause_notes": {
                    "type": "array",
                    "maxItems": V8_SHADOW_MAX_CAUSES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cause": {"type": "string"},
                            "why_short": {"type": "string"},
                            "checks_focus": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 3,
                            },
                        },
                        "required": ["cause", "why_short", "checks_focus"],
                    },
                },
                "evidence_map": {
                    "type": "array",
                    "maxItems": V8_SHADOW_MAX_CITATIONS,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "citation_id": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["citation_id", "why_it_matters"],
                    },
                },
            },
            "required": ["confidence", "diagnostic_brief", "cause_notes", "evidence_map"],
        },
    }


def _v8_shadow_build_ask_overlay(q: str, response: dict) -> Optional[dict]:
    response = dict(response or {})
    citations = list(response.get("citations") or [])
    if str(response.get("status") or "").strip().lower() != "answered":
        return None
    if not citations:
        return None

    proxy = _ask_response_proxy_score(q, response)
    if float(proxy.get("score", 0.0) or 0.0) < V8_SHADOW_MIN_ASK_PROXY:
        return None

    response_language = _select_response_language(q)
    compact_response = {
        "status": str(response.get("status") or ""),
        "answer": re.sub(r"\s+", " ", str(response.get("answer") or "").strip())[:1200],
        "citations": [str(c.get("citation_id") or "").strip() for c in citations if c.get("citation_id")][:V8_SHADOW_MAX_CITATIONS],
    }
    compact_citations = _v8_shadow_compact_citations(citations, max_items=V8_SHADOW_MAX_CITATIONS)

    system_msg = (
        "You are producing a shadow reasoning overlay for an already accepted technical answer. "
        "Do NOT change the answer, do NOT add or remove claims, and do NOT introduce new citations. "
        "Explain only why the existing answer is grounded in the provided citations. "
        "Keep it concise, useful to an operator, and in the requested response language."
    )
    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"ACCEPTED_RESPONSE_JSON:\n{json.dumps(compact_response, ensure_ascii=False)}\n\n"
        f"CITATIONS_JSON:\n{json.dumps(compact_citations, ensure_ascii=False)}\n\n"
        "Return valid JSON. Every evidence_map citation_id must be one of the provided citations."
    )

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[V8_SHADOW_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_v8_shadow_ask_schema(),
            timeout=V8_SHADOW_TIMEOUT,
        )
    except Exception:
        return None

    parsed = dict(parsed or {})
    parsed["mode"] = "shadow"
    parsed["base_proxy_score"] = round(float(proxy.get("score", 0.0) or 0.0), 4)
    parsed["model"] = V8_SHADOW_MODEL
    return parsed


def _v8_shadow_build_root_cause_overlay(q: str, response: dict) -> Optional[dict]:
    response = dict(response or {})
    citations = list(response.get("citations") or [])
    possible_causes = [c for c in (response.get("possible_causes") or []) if isinstance(c, dict)]
    if str(response.get("status") or "").strip().lower() != "answered":
        return None
    if not citations or not possible_causes:
        return None

    proxy = _root_cause_response_proxy_score(q, response)
    if float(proxy.get("score", 0.0) or 0.0) < V8_SHADOW_MIN_ROOT_PROXY:
        return None

    response_language = _select_response_language(q)
    compact_response = {
        "status": str(response.get("status") or ""),
        "problem_summary": re.sub(r"\s+", " ", str(response.get("problem_summary") or "").strip())[:500],
        "possible_causes": [
            {
                "cause": re.sub(r"\s+", " ", str(c.get("cause") or "").strip())[:120],
                "checks": [re.sub(r"\s+", " ", str(x or "").strip())[:140] for x in (c.get("checks") or [])[:3]],
                "citations": [str(x or "").strip() for x in (c.get("citations") or [])[:3]],
            }
            for c in possible_causes[:V8_SHADOW_MAX_CAUSES]
        ],
        "recommended_next_checks": [re.sub(r"\s+", " ", str(x or "").strip())[:140] for x in (response.get("recommended_next_checks") or [])[:4]],
    }
    compact_citations = _v8_shadow_compact_citations(citations, max_items=V8_SHADOW_MAX_CITATIONS)

    system_msg = (
        "You are producing a shadow reasoning overlay for an already accepted industrial root-cause response. "
        "Do NOT change, add, remove, or reorder causes. Do NOT introduce new citations. "
        "Explain only why the accepted causes are grounded and what the existing checks are trying to discriminate. "
        "Keep the output concise, operator-facing, and in the requested response language."
    )
    user_msg = (
        f"PROBLEM:\n{q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"ACCEPTED_RESPONSE_JSON:\n{json.dumps(compact_response, ensure_ascii=False)}\n\n"
        f"CITATIONS_JSON:\n{json.dumps(compact_citations, ensure_ascii=False)}\n\n"
        "Return valid JSON. cause_notes.cause must reuse the accepted cause labels exactly."
    )

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[V8_SHADOW_MODEL, DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_v8_shadow_root_cause_schema(),
            timeout=V8_SHADOW_TIMEOUT,
        )
    except Exception:
        return None

    parsed = dict(parsed or {})
    parsed["mode"] = "shadow"
    parsed["base_proxy_score"] = round(float(proxy.get("score", 0.0) or 0.0), 4)
    parsed["model"] = V8_SHADOW_MODEL
    return parsed


def _v8_attach_shadow_overlay(mode: str, q: str, response: dict) -> dict:
    response = dict(response or {})
    if not V8_SHADOW_REASONING_ENABLED:
        return response
    if mode == "ask" and not V8_SHADOW_ASK_ENABLED:
        return response
    if mode == "root_cause" and not V8_SHADOW_ROOT_CAUSE_ENABLED:
        return response

    overlay = None
    if mode == "ask":
        overlay = _v8_shadow_build_ask_overlay(q, response)
    elif mode == "root_cause":
        overlay = _v8_shadow_build_root_cause_overlay(q, response)

    if overlay:
        response["shadow_reasoning"] = overlay
    return response


@app.post("/v1/ai/ask-shadow")
def ask_v1_shadow(
    payload: AskRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    base_response = ask_v1(payload, x_ai_internal_secret)
    return _v8_attach_shadow_overlay("ask", payload.query or "", dict(base_response or {}))


@app.post("/v1/ai/root-cause-shadow")
def root_cause_v1_shadow(
    payload: RootCauseRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    base_response = root_cause_v1(payload, x_ai_internal_secret)
    return _v8_attach_shadow_overlay("root_cause", payload.query or "", dict(base_response or {}))


# Commit: feat(v8.4): polish root-cause checks and stabilize root-cause language conservatively
V84_ROOT_CAUSE_TEXT_POLISH_ENABLED = (os.environ.get("MM_V84_ROOT_CAUSE_TEXT_POLISH_ENABLED") or "1").strip() != "0"
V84_ROOT_CAUSE_TEXT_POLISH_TIMEOUT = int(os.environ.get("MM_V84_ROOT_CAUSE_TEXT_POLISH_TIMEOUT_SECONDS", "45"))
V84_ROOT_CAUSE_TEXT_POLISH_MAX_CITATIONS = int(os.environ.get("MM_V84_ROOT_CAUSE_TEXT_POLISH_MAX_CITATIONS", "5"))
V84_ROOT_CAUSE_TEXT_POLISH_MAX_CHECKS = int(os.environ.get("MM_V84_ROOT_CAUSE_TEXT_POLISH_MAX_CHECKS", "3"))


_V84_PREV_STRIP_INTERNAL_RESPONSE_ARTIFACTS = _strip_internal_response_artifacts


def _v84_is_root_cause_response(resp: dict) -> bool:
    if not isinstance(resp, dict):
        return False
    if str(resp.get("status") or "").strip().lower() != "answered":
        return False
    if not isinstance(resp.get("possible_causes"), list):
        return False
    return True


def _v84_clean_text_line(text: str, *, max_chars: int = 220) -> str:
    s = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(text or ""))).strip()
    s = re.sub(r"^[\-•*]+\s*", "", s)
    s = s.rstrip(" .;,:-")
    if not s:
        return ""
    return s[:max_chars]


def _v84_clean_text_list(items: list[Any], *, max_items: int = 6, max_chars: int = 180) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items or []:
        s = _v84_clean_text_line(str(item or ""), max_chars=max_chars)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _v84_check_needs_polish(text: str, language: str) -> bool:
    s = _v84_clean_text_line(text, max_chars=260)
    if not s:
        return True
    if len(s) > 120:
        return True

    generic_markers_it = [
        "controllare la macchina",
        "verificare la macchina",
        "eseguire controlli",
        "controllo visivo generale",
        "verificare se",
        "controllare se",
        "ispezione generale",
    ]
    generic_markers_en = [
        "check the machine",
        "verify the machine",
        "perform checks",
        "general visual inspection",
        "check if",
        "verify if",
        "general inspection",
    ]
    low = s.lower()
    markers = generic_markers_en if str(language or "").lower() == "en" else generic_markers_it
    if any(m in low for m in markers):
        return True
    return False


def _v84_root_cause_needs_polish(query: str, resp: dict) -> dict:
    query = str(query or resp.get("symptom") or resp.get("problem_summary") or "").strip()
    target_language = _simple_query_language(query)
    causes = [c for c in (resp.get("possible_causes") or []) if isinstance(c, dict)]
    recommended = list(resp.get("recommended_next_checks") or [])

    texts = []
    if resp.get("problem_summary"):
        texts.append(str(resp.get("problem_summary") or ""))
    for cause in causes[:3]:
        texts.append(str(cause.get("cause") or ""))
        texts.append(str(cause.get("why") or ""))
        texts.extend([str(x or "") for x in (cause.get("checks") or [])[:3]])
    texts.extend([str(x or "") for x in recommended[:5]])

    needs_language = any(
        txt and not _looks_like_target_language(txt, target_language)
        for txt in texts
    )

    needs_checks = False
    if len(recommended) < 2:
        needs_checks = True
    for cause in causes:
        checks = list(cause.get("checks") or [])
        if not checks:
            needs_checks = True
            break
        if any(_v84_check_needs_polish(chk, target_language) for chk in checks[:3]):
            needs_checks = True
            break

    return {
        "target_language": target_language,
        "needs_language": needs_language,
        "needs_checks": needs_checks,
        "should_run": bool(needs_language or needs_checks),
    }


def _v84_root_cause_polish_schema(max_causes: int) -> dict:
    max_causes = max(1, min(int(max_causes or 1), 5))
    return {
        "name": "mm_v84_root_cause_text_polish",
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
                                "maxItems": V84_ROOT_CAUSE_TEXT_POLISH_MAX_CHECKS,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["rank", "cause", "why", "checks"],
                    },
                },
                "recommended_next_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 6,
                },
            },
            "required": ["problem_summary", "possible_causes", "recommended_next_checks"],
        },
    }


def _v84_build_root_cause_polish_payload(resp: dict) -> dict:
    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in (resp.get("citations") or [])
        if c.get("citation_id")
    }
    compact_citations = _v8_shadow_compact_citations(
        list(resp.get("citations") or []),
        max_items=V84_ROOT_CAUSE_TEXT_POLISH_MAX_CITATIONS,
        snippet_chars=220,
    )
    cause_rows = []
    for cause in (resp.get("possible_causes") or [])[:5]:
        if not isinstance(cause, dict):
            continue
        evidence = []
        for cid in cause.get("citations") or []:
            cid = str(cid or "").strip()
            if not cid or cid not in by_id:
                continue
            c = by_id[cid]
            evidence.append(
                {
                    "citation_id": cid,
                    "snippet": re.sub(r"\s+", " ", str(c.get("snippet") or c.get("chunk_full") or "").strip())[:220],
                }
            )
            if len(evidence) >= 2:
                break
        cause_rows.append(
            {
                "rank": int(cause.get("rank") or 0),
                "cause": str(cause.get("cause") or "").strip(),
                "why": str(cause.get("why") or "").strip(),
                "checks": _v84_clean_text_list(list(cause.get("checks") or []), max_items=4, max_chars=180),
                "citations": [str(cid or "").strip() for cid in (cause.get("citations") or []) if str(cid or "").strip()],
                "evidence": evidence,
            }
        )
    return {
        "problem_summary": str(resp.get("problem_summary") or resp.get("symptom") or "").strip(),
        "possible_causes": cause_rows,
        "recommended_next_checks": _v84_clean_text_list(list(resp.get("recommended_next_checks") or []), max_items=6, max_chars=180),
        "compact_citations": compact_citations,
    }


def _v84_should_accept_new_cause_label(original_label: str, new_label: str, *, allow_language_override: bool) -> bool:
    new_label = _v84_clean_text_line(new_label, max_chars=120)
    if not new_label:
        return False
    if allow_language_override:
        return True
    orig_score = _cause_label_specificity_score(original_label)
    new_score = _cause_label_specificity_score(new_label)
    return new_score >= max(0.20, orig_score - 0.08)



def _v84_merge_root_cause_polish(original: dict, polished: dict, *, target_language: str, allow_language_override: bool) -> dict:
    original = dict(original or {})
    polished = dict(polished or {})

    orig_causes = [c for c in (original.get("possible_causes") or []) if isinstance(c, dict)]
    polished_by_rank = {
        int(c.get("rank") or 0): c
        for c in (polished.get("possible_causes") or [])
        if isinstance(c, dict)
    }

    merged_causes: list[dict] = []
    for cause in orig_causes:
        rank = int(cause.get("rank") or 0)
        row = polished_by_rank.get(rank) or {}
        merged = dict(cause)

        new_cause = _v84_clean_text_line(str(row.get("cause") or ""), max_chars=120)
        if _v84_should_accept_new_cause_label(str(cause.get("cause") or ""), new_cause, allow_language_override=allow_language_override):
            merged["cause"] = new_cause

        new_why = _v84_clean_text_line(str(row.get("why") or ""), max_chars=220)
        if new_why:
            merged["why"] = new_why

        new_checks = _v84_clean_text_list(list(row.get("checks") or []), max_items=V84_ROOT_CAUSE_TEXT_POLISH_MAX_CHECKS, max_chars=160)
        if new_checks:
            merged["checks"] = new_checks
        else:
            merged["checks"] = _v84_clean_text_list(list(cause.get("checks") or []), max_items=V84_ROOT_CAUSE_TEXT_POLISH_MAX_CHECKS, max_chars=160)

        merged_causes.append(merged)

    merged = dict(original)
    merged["possible_causes"] = merged_causes

    new_problem_summary = _v84_clean_text_line(str(polished.get("problem_summary") or ""), max_chars=220)
    if new_problem_summary:
        merged["problem_summary"] = new_problem_summary

    recs = _v84_clean_text_list(list(polished.get("recommended_next_checks") or []), max_items=6, max_chars=160)
    if not recs:
        recs = _v84_clean_text_list(
            [chk for cause in merged_causes for chk in (cause.get("checks") or [])],
            max_items=6,
            max_chars=160,
        )
    merged["recommended_next_checks"] = recs
    return merged


def _v84_postprocess_root_cause_response(resp: dict) -> dict:
    if not V84_ROOT_CAUSE_TEXT_POLISH_ENABLED:
        return resp
    if not _v84_is_root_cause_response(resp):
        return resp

    q = str(resp.get("symptom") or resp.get("problem_summary") or "").strip()
    need = _v84_root_cause_needs_polish(q, resp)
    if not need.get("should_run"):
        if isinstance(resp.get("debug"), dict):
            dbg = dict(resp.get("debug") or {})
            dbg["v84_root_cause_polish"] = {"applied": False, **need}
            resp["debug"] = dbg
        return resp

    payload_json = _v84_build_root_cause_polish_payload(resp)
    max_causes = len([c for c in (resp.get("possible_causes") or []) if isinstance(c, dict)])

    system_msg = (
        "You are polishing an already accepted industrial root-cause response. "
        "Do NOT change the number of causes, their rank order, or their causal meaning. "
        "You may only improve wording quality and target-language consistency. "
        "Task: rewrite problem_summary, why text, checks, and recommended_next_checks in the requested target language. "
        "Cause labels may be translated or lightly normalized only if needed, but must preserve the original meaning and stay compact, technical, and noun-phrase style. "
        "Checks must be short, operational, and technician-friendly. Prefer concrete actions over generic advice. "
        "Do not invent new components, causes, or evidence beyond the current response and provided snippets. "
        "Keep output concise and structured."
    )
    user_msg = (
        f"TARGET_LANGUAGE: {need.get('target_language') or 'it'}\n\n"
        f"CURRENT_ROOT_CAUSE_JSON:\n{json.dumps(payload_json, ensure_ascii=False)}\n\n"
        "Return valid JSON with the same ranks and the same number of causes."
    )

    try:
        polished = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_v84_root_cause_polish_schema(max_causes=max_causes),
            timeout=V84_ROOT_CAUSE_TEXT_POLISH_TIMEOUT,
        )
    except Exception as e:
        if isinstance(resp.get("debug"), dict):
            dbg = dict(resp.get("debug") or {})
            dbg["v84_root_cause_polish"] = {"applied": False, **need, "error": str(e)[:240]}
            resp["debug"] = dbg
        return resp

    merged = _v84_merge_root_cause_polish(
        resp,
        polished,
        target_language=str(need.get("target_language") or "it"),
        allow_language_override=bool(need.get("needs_language")),
    )

    if isinstance(merged.get("debug"), dict):
        dbg = dict(merged.get("debug") or {})
        dbg["v84_root_cause_polish"] = {"applied": True, **need}
        merged["debug"] = dbg
    elif isinstance(resp.get("debug"), dict):
        dbg = dict(resp.get("debug") or {})
        dbg["v84_root_cause_polish"] = {"applied": True, **need}
        merged["debug"] = dbg

    return merged



def _strip_internal_response_artifacts(resp: dict) -> dict:
    cleaned = _V8_PREV_STRIP_INTERNAL_RESPONSE_ARTIFACTS(resp)
    if not isinstance(cleaned, dict):
        return cleaned
    if _v84_is_root_cause_response(cleaned):
        return _v84_postprocess_root_cause_response(cleaned)
    return cleaned
