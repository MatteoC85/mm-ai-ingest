
import os
import re
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI()

AI_INTERNAL_SECRET = (os.environ.get("AI_INTERNAL_SECRET") or "").strip()

FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))  # 50MB default


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "status": "healthy",
        "build": os.environ.get("BUILD_ID", "no-build-id"),
    }


@app.get("/ping")
def ping():
    return {
        "ok": True,
        "status": "ok",
        "build": os.environ.get("BUILD_ID", "no-build-id"),
    }


class IngestRequest(BaseModel):
    # URL temporaneo (Bubble signed URL / CDN)
    file_url: str
    # opzionali (per logging/debug)
    document_id: Optional[str] = None
    filename: Optional[str] = None


def _require_secret(x_ai_internal_secret: Optional[str]) -> None:
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured: AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_url(u: str) -> str:
    # Bubble/CF a volte danno //cdn...
    if u.startswith("//"):
        return "https:" + u
    return u


def _safe_filename(name: Optional[str]) -> str:
    if not name:
        return "document.pdf"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "document.pdf"


@app.post("/v1/ai/ingest/document")
def ingest_document(
    payload: IngestRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None, convert_underscores=False),
):
    _require_secret(x_ai_internal_secret)

    url = _normalize_url(payload.file_url.strip())
    if not url.lower().startswith("http"):
        raise HTTPException(status_code=400, detail="file_url must be http(s)")

    # download con limite dimensione
    try:
        with requests.get(url, stream=True, timeout=FETCH_TIMEOUT) as r:
            r.raise_for_status()

            content_length = r.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > MAX_PDF_BYTES:
                        raise HTTPException(status_code=413, detail="PDF too large")
                except ValueError:
                    pass

            data = bytearray()
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise HTTPException(status_code=413, detail="PDF too large")

    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {type(e).__name__}")

    # parse PDF
    try:
        reader = PdfReader(bytes(data))
        pages_total = len(reader.pages)
        pages_with_text = 0
        text_chars = 0

        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages_with_text += 1
                text_chars += len(t)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF parse failed: {type(e).__name__}")

    return {
        "ok": True,
        "build": os.environ.get("BUILD_ID", "no-build-id"),
        "document_id": payload.document_id,
        "filename": _safe_filename(payload.filename),
        "source_id": payload.document_id or "unknown",
        "pages_detected": pages_total,
        "pages_with_text": pages_with_text,
        "text_chars": text_chars,
    }

%), col  2/10 ( 20%), char    2/3839 ( 0%) ]
^G Help          ^O Write Out     ^W Where Is      ^K
