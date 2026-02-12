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

# Chunking (Solution A) — robust defaults (char-based, overlap)
# Nota: char != token, ma su testo tecnico è una proxy buona e stabile.
CHUNK_TARGET_CHARS = int(os.environ.get("MM_CHUNK_TARGET_CHARS", "3800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("MM_CHUNK_OVERLAP_CHARS", "800"))
CHUNK_MIN_CHARS = int(os.environ.get("MM_CHUNK_MIN_CHARS", "200"))  # evita chunk microscopici
PAGE_JOIN_SEPARATOR = "\n\n"  # conserva struttura (utile per tabelle/testo a blocchi)

# OpenAI
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_EMBED_MODEL = (os.environ.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
OPENAI_EMBED_URL = (os.environ.get("OPENAI_EMBED_URL") or "https://api.openai.com/v1/embeddings").strip()

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


def _normalize_text_keep_lines(s: str) -> str:
    """
    Normalizzazione 'safe' per manuali/tabelle:
    - normalizza newline
    - rimuove spazi e TAB ripetuti MA non schiaccia i newline
    - strip finale
    """
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # riduci TAB a spazio (tabelle estratte spesso hanno tab)
    s = s.replace("\t", " ")
    # comprimi spazi multipli, ma preserva newline
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
    """
    pages: list of (page_number, text)
    Ritorna:
    - global_text: concatenazione pagine con separatore
    - spans: lista di (start_idx, end_idx, page_number) in global_text
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0

    for (page_number, text) in pages:
        t = _normalize_text_keep_lines(text or "")
        start = pos
        parts.append(t)
        pos += len(t)
        end = pos  # end is exclusive
        spans.append((start, end, int(page_number)))

        # aggiungi separatore tra pagine (non dopo l’ultima)
        parts.append(PAGE_JOIN_SEPARATOR)
        pos += len(PAGE_JOIN_SEPARATOR)

    if parts:
        # rimuovi l'ultimo separatore aggiunto
        global_text = "".join(parts)
        global_text = global_text[: -len(PAGE_JOIN_SEPARATOR)]
        # correggi l'ultima span end (perché abbiamo tagliato il separatore finale)
        last_start, last_end, last_page = spans[-1]
        # last_end non cambia: il separatore era fuori dalla span (aggiunto dopo)
        return global_text, spans

    return "", []


def _pages_for_slice(spans: list[tuple[int, int, int]], slice_start: int, slice_end: int) -> tuple[int, int]:
    """
    Dato un intervallo [slice_start, slice_end) su global_text,
    ritorna (page_from, page_to) basandosi sulle spans.
    """
    page_from = None
    page_to = None

    for (ps, pe, pn) in spans:
        # overlap se:
        # ps < slice_end AND pe > slice_start
        if ps < slice_end and pe > slice_start:
            if page_from is None:
                page_from = pn
            page_to = pn

    # fallback safety
    if page_from is None:
        page_from = spans[0][2] if spans else 1
    if page_to is None:
        page_to = page_from

    return int(page_from), int(page_to)


def _chunk_text(global_text: str, spans: list[tuple[int, int, int]]) -> list[dict]:
    """
    Chunking solution A:
    - step = target - overlap
    - chunk_end = min(start + target, len(text))
    - trim whitespace ai bordi ma NON toccare i newline interni
    - scarta chunk troppo piccoli (tranne se è l’unico)
    Ritorna lista di dict:
      {chunk_index, page_from, page_to, chunk_text}
    """
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

        # trim bordi (non tocca i newline interni)
        chunk_clean = chunk_raw.strip()

        # se l’ultimo chunk viene minuscolo, prova ad attaccarlo al precedente (PRO robustness)
        if chunks and len(chunk_clean) < CHUNK_MIN_CHARS:
            # append al chunk precedente mantenendo separatore
            prev = chunks[-1]
            prev_text = prev["chunk_text"]
            glued = (prev_text + "\n" + chunk_clean).strip()
            prev["chunk_text"] = glued
            # aggiorna page_to col nuovo slice
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

    # se per qualche motivo restasse vuoto (testo quasi tutto whitespace)
    chunks = [c for c in chunks if c["chunk_text"]]
    return chunks

def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY missing")

    # Privacy: inviamo SOLO i testi, niente metadati.
    payload = {
        "model": OPENAI_EMBED_MODEL,
        "input": texts,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_EMBED_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"OpenAI embeddings failed: {r.status_code} {r.text}")

    data = r.json()
    # data["data"] = [{embedding: [...], index: i, ...}, ...]
    out = [None] * len(texts)
    for item in data.get("data", []):
        out[int(item["index"])] = item["embedding"]
    if any(v is None for v in out):
        raise Exception("OpenAI embeddings response missing some items")
    return out


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
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "machinemind-ai-2a"
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

        resp = client.create_task(request={"parent": parent, "task": task})
        print("ENQUEUE_OK", resp.name)

    except Exception as e:
        # non blocchiamo ingest se enqueue fallisce
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
    # auth
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

            # 2) Costruisci testo globale + mappa posizioni->pagine
            pages = [(int(pn), txt or "") for (pn, txt) in page_rows]
            global_text, spans = _build_global_text_and_page_spans(pages)

            # 3) Chunking PRO (char-based + overlap)
            chunks = _chunk_text(global_text, spans)

            # 4) Verifica schema document_chunks (il tuo schema è fisso)
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

            # 5) Replace completo (idempotente)
            cur.execute(
                "DELETE FROM document_chunks WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )

            # 6) Insert chunks
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

            # 7) Embeddings (batch) + update su DB
            # Batch piccolo per DEV robusto
            BATCH_SIZE = 32

            # Prepara i testi (privacy: solo chunk_text)
            chunk_texts = [c["chunk_text"] for c in chunks]

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch_texts = chunk_texts[batch_start: batch_start + BATCH_SIZE]

                try:
                    vectors = _openai_embed_texts(batch_texts)
                except Exception as e:
                    # segna errore su tutti i chunk del batch
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

                    # pgvector: passiamo stringa "[...]" e castiamo a vector
                    vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

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
