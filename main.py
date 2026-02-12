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
GCP_PROJECT_ID = (os.environ.get("MM_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "").strip()

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
        if not GCP_PROJECT_ID:
            raise Exception("Missing project id (set MM_GCP_PROJECT on Cloud Run)")

        location = "europe-west1"
        queue = "mm-ai-index-dev"

        # usa sempre questa env se disponibile (Cloud Run standard)
        service_url = (os.environ.get("K_SERVICE") and os.environ.get("K_REVISION"))  # solo per check
        base_url = (os.environ.get("SERVICE_URL") or os.environ.get("K_SERVICE_URL") or "").strip()

        # FALLBACK: meglio hardcodata che vuota (DEV only)
        if not base_url:
            base_url = "https://mm-ai-ingest-fixed-pvgxe6eo5q-ew.a.run.app"

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(GCP_PROJECT_ID, location, queue)

        task_payload = {
            "company_id": company_id,
            "machine_id": machine_id,
            "bubble_document_id": bubble_document_id,
            "trace_id": "ingest_auto",
        }

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{base_url}/v1/ai/index/document",
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

    # DEBUG: stampa dove stai scrivendo (senza password)
    print("INDEX_START", {
        "company_id": company_id,
        "machine_id": machine_id,
        "bubble_document_id": bubble_document_id,
        "trace_id": trace_id,
        "db_host": DB_HOST,
        "db_name": DB_NAME,
        "db_user": DB_USER,
    })

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
            print("INDEX_PAGES", {"count": len(pages)})

            if not pages:
                # IMPORTANTISSIMO: lo vediamo in log
                print("INDEX_ABORT_NO_PAGES", {"company_id": company_id, "bubble_document_id": bubble_document_id})
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

            # 2) Verifica schema document_chunks
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
                print("INDEX_ABORT_SCHEMA_MISSING", {"missing": missing, "found": sorted(chunk_cols)})
                raise HTTPException(
                    status_code=500,
                    detail=f"document_chunks missing columns: {missing}. Found: {sorted(chunk_cols)}",
                )

            # 3) Replace completo (idempotente)
            cur.execute(
                "DELETE FROM document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

            # 4) Insert: v1 = 1 chunk per pagina
            insert_q = """
                INSERT INTO document_chunks(
                    company_id, machine_id, bubble_document_id,
                    chunk_index, page_from, page_to,
                    chunk_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """

            chunks_written = 0
            for (page_number, text, text_chars) in pages:
                page_number_i = int(page_number or 0)
                text_s = (text or "").strip()

                cur.execute(
                    insert_q,
                    (
                        company_id,
                        machine_id,
                        bubble_document_id,
                        page_number_i,  # chunk_index
                        page_number_i,  # page_from
                        page_number_i,  # page_to
                        text_s,         # chunk_text
                    ),
                )
                chunks_written += 1

            conn.commit()

            # DEBUG: conta righe appena inserite
            cur.execute(
                "SELECT count(*) FROM document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            inserted_count = int(cur.fetchone()[0])
            print("INDEX_DONE", {"chunks_written": chunks_written, "inserted_count": inserted_count})

    except Exception as e:
        # IMPORTANTISSIMO: se qualcosa va storto, lo vediamo
        print("INDEX_ERROR", str(e))
        raise
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
        "pages_detected": len(pages),
    }
