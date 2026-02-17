# trigger test 2026-02-17

import os
import io
import re
import math
import json
from typing import Optional, List

import requests
import psycopg2
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from google.cloud import tasks_v2

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
PAGE_JOIN_SEPARATOR = "\n\n"

# OpenAI
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_EMBED_MODEL = (os.environ.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
OPENAI_EMBED_URL = (os.environ.get("OPENAI_EMBED_URL") or "https://api.openai.com/v1/embeddings").strip()

# OpenAI Chat
OPENAI_CHAT_MODEL = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4.1-mini").strip()
OPENAI_CHAT_URL = (os.environ.get("OPENAI_CHAT_URL") or "https://api.openai.com/v1/chat/completions").strip()

ASK_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SIM_THRESHOLD", "0.35"))
ASK_MAX_TOP_K = int(os.environ.get("MM_ASK_MAX_TOP_K", "8"))
ASK_SNIPPET_CHARS = int(os.environ.get("MM_ASK_SNIPPET_CHARS", "700"))
ASK_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_MAX_CONTEXT_CHARS", "6000"))

# -----------------------------
# Entity fallback (URL / email / phone)
# -----------------------------
URL_REGEX = re.compile(r"(https?://[^\s\)\]\}]+|www\.[^\s\)\]\}]+)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

URL_HINTS = ["sito", "website", "web site", "url", "link", "pagina", "dominio", "www"]
EMAIL_HINTS = ["email", "e-mail", "mail", "posta"]
PHONE_HINTS = ["telefono", "cell", "cellulare", "tel", "contatto", "chiamare", "numero"]


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

def _db_find_entity_chunk(company_id: str, machine_id: str, kind: str) -> Optional[dict]:
    """
    Cerca nei chunk (scope: company_id AND (machine_id = M OR NULL)) un chunk che contenga un'entità.
    kind: "url" | "email" | "phone"
    Ritorna {citation_id, bubble_document_id, page_from, page_to, snippet} oppure None.
    """
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
            cur.execute(
                """
                SELECT bubble_document_id, chunk_index, page_from, page_to,
                       left(chunk_text, %s) AS snippet
                FROM public.document_chunks
                WHERE company_id = %s
                  AND (machine_id = %s OR machine_id IS NULL)
                  AND chunk_text ~* %s
                ORDER BY bubble_document_id, page_from, chunk_index
                LIMIT 1;
                """,
                (ASK_SNIPPET_CHARS, company_id, machine_id, pattern),
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
                "bubble_document_id": bdid,
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": snippet,
                "value": value,
            }
    finally:
        conn.close()

@app.get("/ping")
def ping():
    return {"ok": True}


class IngestRequest(BaseModel):
    file_url: str
    company_id: str
    machine_id: str
    bubble_document_id: str


def _db_conn():
    if not (DB_HOST and DB_USER and DB_PASSWORD):
        raise HTTPException(status_code=500, detail="DB env missing")
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


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


def _normalize_text_keep_lines(s: str) -> str:
    if not s:
        return ""
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


def _build_global_text_and_page_spans(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0

    for (page_number, text) in pages:
        t = _normalize_text_keep_lines(text or "")
        start = pos
        parts.append(t)
        pos += len(t)
        end = pos
        spans.append((start, end, int(page_number)))

        parts.append(PAGE_JOIN_SEPARATOR)
        pos += len(PAGE_JOIN_SEPARATOR)

    if parts:
        global_text = "".join(parts)
        global_text = global_text[: -len(PAGE_JOIN_SEPARATOR)]
        return global_text, spans

    return "", []


def _pages_for_slice(spans: list[tuple[int, int, int]], slice_start: int, slice_end: int) -> tuple[int, int]:
    page_from = None
    page_to = None

    for (ps, pe, pn) in spans:
        if ps < slice_end and pe > slice_start:
            if page_from is None:
                page_from = pn
            page_to = pn

    if page_from is None:
        page_from = spans[0][2] if spans else 1
    if page_to is None:
        page_to = page_from

    return int(page_from), int(page_to)


def _chunk_text(global_text: str, spans: list[tuple[int, int, int]]) -> list[dict]:
    text_len = len(global_text)
    if text_len == 0:
        return []

    target = max(500, CHUNK_TARGET_CHARS)
    overlap = min(max(0, CHUNK_OVERLAP_CHARS), target - 1) if target > 1 else 0
    step = max(1, target - overlap)

    chunks: list[dict] = []
    start = 0
    chunk_index = 1

    while start < text_len:
        end = min(start + target, text_len)
        chunk_raw = global_text[start:end]
        chunk_clean = chunk_raw.strip()

        if chunks and len(chunk_clean) < CHUNK_MIN_CHARS:
            prev = chunks[-1]
            glued = (prev["chunk_text"] + "\n" + chunk_clean).strip()
            prev["chunk_text"] = glued
            pf, pt = _pages_for_slice(spans, start, end)
            prev["page_to"] = max(prev["page_to"], pt)
            break

        pf, pt = _pages_for_slice(spans, start, end)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "page_from": pf,
                "page_to": pt,
                "chunk_text": chunk_clean,
            }
        )

        chunk_index += 1
        start += step

    chunks = [c for c in chunks if c["chunk_text"]]
    return chunks


def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    payload = {"model": OPENAI_EMBED_MODEL, "input": texts}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

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
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"OpenAI chat failed: {r.status_code} {r.text}")

    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


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
    top_k: int = 5
    debug: Optional[bool] = False


class Citation(BaseModel):
    citation_id: str
    bubble_document_id: str
    page_from: int
    page_to: int
    snippet: str
    similarity: float


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


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

    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        data = r.content
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF too large")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Fetch failed")

    try:
        reader = PdfReader(io.BytesIO(data))
        pages_total = len(reader.pages)

        pages_text: list[str] = []
        pages_with_text = 0
        text_chars = 0

        for p in reader.pages:
            t = (p.extract_text() or "").strip()
            pages_text.append(t)
            text_chars += len(t)
            if len(t) >= MIN_PAGE_CHARS:
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

        resp = client.create_task(request={"parent": parent, "task": task})
        print("ENQUEUE_OK", resp.name)

    except Exception as e:
        print("ENQUEUE_FAIL", str(e))

    return {
        "ok": True,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_detected": pages_total,
        "text_chars": text_chars,
    }


class IndexDocumentRequest(BaseModel):
    company_id: str
    machine_id: str
    bubble_document_id: str
    trace_id: Optional[str] = None


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
            global_text, spans = _build_global_text_and_page_spans(pages)
            chunks = _chunk_text(global_text, spans)

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
                            (str(e), OPENAI_EMBED_MODEL, company_id, bubble_document_id, int(chunks[idx]["chunk_index"])),
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

    company_id = (payload.company_id or "").strip()
    if not company_id:
        raise HTTPException(status_code=400, detail="Missing company_id")

    # ✅ machine_id obbligatorio: scope = macchina corrente + generici (machine_id IS NULL)
    machine_id = (payload.machine_id or "").strip()
    if not machine_id:
        raise HTTPException(status_code=400, detail="Missing machine_id")

    top_k = int(payload.top_k or 5)
    top_k = max(1, min(top_k, ASK_MAX_TOP_K))

    q_vec = _openai_embed_texts([q])[0]
    q_vec_lit = _vector_literal(q_vec)

    rows = []
    chunks_matching_filter = None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if payload.bubble_document_id:
                bdid = payload.bubble_document_id.strip()

                if payload.debug:
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
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND bubble_document_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, bdid, machine_id, q_vec_lit, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT bubble_document_id, chunk_index, page_from, page_to,
                           left(chunk_text, %s) AS snippet,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM public.document_chunks
                    WHERE company_id = %s
                      AND embedding IS NOT NULL
                      AND (machine_id = %s OR machine_id IS NULL)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, machine_id, q_vec_lit, top_k),
                )

            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        resp = {
            "ok": True,
            "status": "no_sources",
            "answer": "Non trovo informazioni nei documenti indicizzati per rispondere.",
            "citations": [],
            "top_k": top_k,
            "similarity_max": None,
        }
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": payload.bubble_document_id,
                "chunks_matching_filter": chunks_matching_filter,
            }
        return resp

    citations: List[dict] = []
    sim_max = -1.0

    for (bdid, chunk_index, page_from, page_to, snippet, similarity) in rows:
        sim = float(similarity)
        sim_max = max(sim_max, sim)
        citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
        citations.append(
            {
                "citation_id": citation_id,
                "bubble_document_id": bdid,
                "page_from": int(page_from),
                "page_to": int(page_to),
                "snippet": (snippet or "").strip(),
                "similarity": sim,
            }
        )

    # ✅ Entity fallback: se la domanda chiede sito/email/telefono e lo troviamo negli snippet, rispondiamo anche sotto soglia
    if sim_max < ASK_SIM_THRESHOLD:
        picked = _pick_entity_from_citations(q, citations)
        if picked:
            value, c = picked
            answer = f"Nel documento compare questo dato: {value} [{c['citation_id']}]"
            return {
                "ok": True,
                "status": "answered",
                "answer": answer,
                "citations": citations,
                "top_k": top_k,
                "similarity_max": sim_max,
                "chat_model": OPENAI_CHAT_MODEL,
            }
        
        # 2nd-level fallback: se gli snippet top_k non contengono l'entità,
        # prova a cercarla direttamente nel DB nello scope macchina + generici.
        kind = None
        if _q_has_any(q, URL_HINTS):
            kind = "url"
        elif _q_has_any(q, EMAIL_HINTS):
            kind = "email"
        elif _q_has_any(q, PHONE_HINTS):
            kind = "phone"

        if kind:
            hit = _db_find_entity_chunk(company_id=company_id, machine_id=machine_id, kind=kind)
            if hit:
                answer = f"Nel documento compare questo dato: {hit['value']} [{hit['citation_id']}]"
                return {
                    "ok": True,
                    "status": "answered",
                    "answer": answer,
                    "citations": [
                        {
                            "citation_id": hit["citation_id"],
                            "bubble_document_id": hit["bubble_document_id"],
                            "page_from": hit["page_from"],
                            "page_to": hit["page_to"],
                            "snippet": hit["snippet"],
                            "similarity": sim_max,
                        }
                    ],
                    "top_k": top_k,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                }

        resp = {
            "ok": True,
            "status": "no_sources",
            "answer": "Non trovo informazioni nei documenti indicizzati per rispondere.",
            "citations": [],
            "top_k": top_k,
            "similarity_max": sim_max,
        }
        if payload.debug:
            resp["debug"] = {
                "company_id": company_id,
                "machine_id": machine_id,
                "bubble_document_id": payload.bubble_document_id,
                "chunks_matching_filter": chunks_matching_filter,
            }
        return resp

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
        return {
            "ok": True,
            "status": "no_sources",
            "answer": answer,
            "citations": [],
            "top_k": top_k,
            "similarity_max": sim_max,
        }

    return {
        "ok": True,
        "status": "answered",
        "answer": answer,
        "citations": citations,
        "top_k": top_k,
        "similarity_max": sim_max,
        "chat_model": OPENAI_CHAT_MODEL,
    }
