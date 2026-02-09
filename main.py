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
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

@app.get("/ping")
def ping():
    return {"ok": True}

class IngestRequest(BaseModel):
    file_url: str

@app.post("/v1/ai/ingest/document")
def ingest_document(
    payload: IngestRequest,
    x_ai_internal_secret: Optional[str] = Header(default=None, convert_underscores=False),
):
    # auth
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
    except Exception as e:
        raise HTTPException(status_code=502, detail="Fetch failed")

    # parse
    try:
        reader = PdfReader(bytes(data))
        pages_total = len(reader.pages)
        text_chars = 0
        for p in reader.pages:
            t = p.extract_text() or ""
            text_chars += len(t)
    except Exception:
        raise HTTPException(status_code=422, detail="PDF parse failed")

    return {"ok": True, "pages_detected": pages_total, "text_chars": text_chars}
