import os
import io
import math
import json
import requests
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pypdf import PdfReader

AI_INTERNAL_SECRET = os.environ.get("AI_INTERNAL_SECRET", "").strip()

MIN_TOTAL_CHARS = int(os.environ.get("MM_MIN_TOTAL_CHARS", "2000"))
MIN_PAGES_WITH_TEXT_ABS = int(os.environ.get("MM_MIN_PAGES_WITH_TEXT_ABS", "2"))
MIN_PAGES_WITH_TEXT_PCT = float(os.environ.get("MM_MIN_PAGES_WITH_TEXT_PCT", "0.20"))

FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

app = FastAPI()


def error(code: str, message: str, **extra):
    payload = {
        "ok": False,
        "status": "error",
        "error": {"code": code, "message": message},
    }
    payload.update(extra)
    return JSONResponse(payload, status_code=200)


@app.get("/healthz")
def healthz():
    return {"ok": True, "status": "healthy"}


@app.post("/v1/ai/ingest/document")
async def ingest_document(request: Request):
    body: Dict[str, Any] = await request.json()

    token = (body.get("auth_token") or "").strip()
    if not AI_INTERNAL_SECRET or token != AI_INTERNAL_SECRET:
        return error("UNAUTHORIZED", "Invalid auth token")

    file_url = (body.get("file_url") or "").strip()
    if not file_url:
        return error("BAD_REQUEST", "file_url is required")

    if file_url.startswith("//"):
        file_url = "https:" + file_url

    try:
        r = requests.get(file_url, timeout=FETCH_TIMEOUT)
    except Exception as e:
        return error("PDF_FETCH_FAILED", str(e))

    if not r.ok:
        return error("PDF_FETCH_FAILED", f"HTTP {r.status_code}")

    pdf_bytes = r.content
    if len(pdf_bytes) > MAX_PDF_BYTES:
        return error("PDF_TOO_LARGE", "PDF exceeds max allowed size")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        return error("PDF_PARSE_FAILED", str(e))

    pages_detected = len(reader.pages)
    pages_with_text = 0
    total_chars = 0

    for p in reader.pages:
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if txt:
            pages_with_text += 1
            total_chars += len(txt)

    if total_chars == 0:
        return error(
            "PDF_NO_TEXT_LAYER",
            "PDF has no extractable text layer",
            pages_detected=pages_detected,
            pages_with_text=0,
            text_chars=0,
        )

    min_pages_required = max(
        MIN_PAGES_WITH_TEXT_ABS,
        math.ceil(pages_detected * MIN_PAGES_WITH_TEXT_PCT),
    )

    if pages_with_text < min_pages_required or total_chars < MIN_TOTAL_CHARS:
        return error(
            "NOT_INDEXABLE",
            "PDF text content below deterministic thresholds",
            pages_detected=pages_detected,
            pages_with_text=pages_with_text,
            text_chars=total_chars,
        )

    bubble_doc_id = (
        body.get("bubble_document_id")
        or body.get("bubble_document")
        or body.get("source", {}).get("bubble_document_id")
        or "unknown"
    )

    return JSONResponse(
        {
            "ok": True,
            "status": "indexed",
            "source_id": f"mm:doc:{bubble_doc_id}",
            "pages_detected": pages_detected,
            "pages_with_text": pages_with_text,
            "text_chars": total_chars,
        },
        status_code=200,
    )
