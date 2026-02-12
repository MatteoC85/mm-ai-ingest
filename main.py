import os
import io
import math
from typing import Optional

import requests
import psycopg2
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from psycopg2 import sql

from google.cloud import tasks_v2
import json

app = FastAPI()

AI_INTERNAL_SECRET = (os.environ.get("AI_INTERNAL_SECRET") or "").strip()
FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

# DB (Cloud SQL via unix socket /cloudsql/...)
DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

# Indicizzabilità (PRO) — configurabile via env
MIN_TEXT_CHARS = int(os.environ.get("MM_MIN_TEXT_CHARS", "2000"))
MIN_TEXT_CHARS_SHORT = int(os.environ.get("MM_MIN_TEXT_CHARS_SHORT", "800"))
MIN_PAGE_CHARS = int(os.environ.get("MM_MIN_PAGE_CHARS", "30"))
MIN_PAGES_WITH_TEXT_ABS = int(os.environ.get("MM_MIN_PAGES_WITH_TEXT_ABS", "2"))
MIN_PAGES_WITH_TEXT_PCT = float(os.environ.get("MM_MIN_PAGES_WITH_TEXT_PCT", "0.20"))


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
        host=DB_HOST,      # /cloudsql/PROJECT:REGION:INSTANCE
        dbname=DB_NAME,    # postgres
        user=DB_USER,      # mm_ai_app
        password=DB_PASSWORD,
    )


# -------------------------------
# Helpers: schema-adaptive chunks
# -------------------------------
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


def _pick_column(colset: set[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in colset:
            return c
    return None


@app.post("/v1/ai/ingest/document")
def ingest_document(
    payload: IngestRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None),
):
    # auth
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # metadati necessari per salvare pages
    company_id = (payload.company_id or "").strip()
    machine_id = (payload.machine_id or "").strip()
    bubble_document_id = (payload.bubble_document_id or "").strip()
    if not (company_id and machine_id and bubble_document_id):
        raise HTTPException(status_code=400, detail="Missing company_id/machine_id/bubble_document_id")

    url = payload.file_url.strip()
    if url.startswith("//"):
        url = "https:" + url

    # download
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

    # parse + estrazione testo per pagina
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

    # Indicizzabilità PRO (se non indicizzabile, NON salviamo niente su DB)
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
                "pages_detected": pages_total,  # compat
                "text_chars": text_chars,
                "thresholds": {
                    "min_text_chars_short": MIN_TEXT_CHARS_SHORT,
                    "min_page_chars": MIN_PAGE_CHARS,
                    "min_pages_with_text_abs": 1,
                },
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
                "pages_detected": pages_total,  # compat
                "text_chars": text_chars,
                "thresholds": {
                    "min_text_chars": MIN_TEXT_CHARS,
                    "min_page_chars": MIN_PAGE_CHARS,
                    "min_pages_with_text_abs": MIN_PAGES_WITH_TEXT_ABS,
                    "min_pages_with_text_pct": MIN_PAGES_WITH_TEXT_PCT,
                    "min_pages_required": min_pages_required,
                },
            }

    # ✅ Indicizzabile -> salva document_pages (replace completo)
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            # pulizia eventuale re-ingest
            cur.execute(
                "DELETE FROM document_pages WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

            # insert pagine (page_number 1-based)
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

    # ===============================
    # Enqueue async index job (Cloud Tasks)
    # ===============================
    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        location = "europe-west1"
        queue = "mm-ai-index-dev"
        service_url = os.environ.get("K_SERVICE_URL") or os.environ.get("SERVICE_URL")

        if not service_url:
            # fallback hardcoded (DEV only)
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

    except Exception as e:
        # non blocchiamo ingest se enqueue fallisce
        print("Cloud Tasks enqueue failed:", str(e))

    return {
        "ok": True,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_detected": pages_total,  # compat
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
    # auth (stessa dell’ingest)
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
            # 1) Leggi pagine estratte
            cur.execute(
                """
                SELECT page_number, text, text_chars
                FROM document_pages
                WHERE company_id=%s AND bubble_document_id=%s
                ORDER BY page_number;
                """,
                (company_id, bubble_document_id),
            )
            pages = cur.fetchall()

            if not pages:
                raise HTTPException(
                    status_code=404,
                    detail="No document_pages found for company_id + bubble_document_id",
                )

            # 2) Scopri lo schema reale di document_chunks (mappa nomi colonna)
            chunk_cols = _get_table_columns(cur, "document_chunks")

            company_col = "company_id" if "company_id" in chunk_cols else None
            machine_col = "machine_id" if "machine_id" in chunk_cols else None
            doc_col = _pick_column(chunk_cols, ["bubble_document_id", "document_id", "doc_id"])
            page_col = _pick_column(chunk_cols, ["page_number", "page"])
            chunk_col = _pick_column(chunk_cols, ["chunk_number", "chunk_index", "chunk_no"])
            text_col = _pick_column(chunk_cols, ["text", "chunk_text", "content"])
            chars_col = _pick_column(chunk_cols, ["text_chars", "chunk_chars", "chars"])
            embedding_col = _pick_column(chunk_cols, ["embedding", "vector"])

            missing = []
            if not company_col:
                missing.append("company_id")
            if not doc_col:
                missing.append("bubble_document_id/document_id/doc_id")
            if not text_col:
                missing.append("text/chunk_text/content")

            if missing:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"document_chunks schema missing required columns: {missing}. "
                        f"Found columns: {sorted(chunk_cols)}"
                    ),
                )

            # 3) Replace completo (idempotente per retry Cloud Tasks)
            delete_q = sql.SQL(
                "DELETE FROM document_chunks WHERE {company_col}=%s AND {doc_col}=%s;"
            ).format(
                company_col=sql.Identifier(company_col),
                doc_col=sql.Identifier(doc_col),
            )
            cur.execute(delete_q, (company_id, bubble_document_id))

            # 4) Insert dinamico (solo colonne esistenti)
            insert_columns: list[str] = []
            insert_columns.append(company_col)
            if machine_col:
                insert_columns.append(machine_col)
            insert_columns.append(doc_col)

            # v1: 1 chunk = 1 pagina
            if chunk_col:
                insert_columns.append(chunk_col)
            if page_col:
                insert_columns.append(page_col)

            insert_columns.append(text_col)
            if chars_col:
                insert_columns.append(chars_col)
            if embedding_col:
                insert_columns.append(embedding_col)

            insert_q = sql.SQL("INSERT INTO document_chunks ({cols}) VALUES ({vals});").format(
                cols=sql.SQL(", ").join([sql.Identifier(c) for c in insert_columns]),
                vals=sql.SQL(", ").join([sql.Placeholder() for _ in insert_columns]),
            )

            chunks_written = 0
            total_chars = 0

            for (page_number, text, text_chars) in pages:
                page_number_i = int(page_number or 0)
                text_s = text or ""
                chars_i = int(text_chars or len(text_s))

                values: list[object] = []
                values.append(company_id)
                if machine_col:
                    values.append(machine_id)
                values.append(bubble_document_id)

                if chunk_col:
                    values.append(page_number_i)  # chunk == page number (v1)
                if page_col:
                    values.append(page_number_i)

                values.append(text_s)
                if chars_col:
                    values.append(chars_i)
                if embedding_col:
                    values.append(None)  # embeddings non implementati ancora

                cur.execute(insert_q, values)

                chunks_written += 1
                total_chars += chars_i

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
        "chunks_written": chunks_written,
        "text_chars": total_chars,
        "pages_detected": len(pages),
    }
