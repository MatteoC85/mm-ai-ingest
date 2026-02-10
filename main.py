import os
import re
import io
from typing import Optional
import math

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI()

AI_INTERNAL_SECRET = (os.environ.get("AI_INTERNAL_SECRET") or "").strip()
FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

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
        reader = PdfReader(io.BytesIO(data))
        pages_total = len(reader.pages)
        text_chars = 0
        pages_with_text = 0

        for p in reader.pages:
            t = (p.extract_text() or "").strip()
            text_chars += len(t)
            if len(t) >= MIN_PAGE_CHARS:
                pages_with_text += 1
    except Exception:
        raise HTTPException(status_code=422, detail="PDF parse failed")

    # Indicizzabilità PRO
    # Regola short per documenti 1–2 pagine (evita falsi negativi su schede brevi)
    if pages_total <= 2:
        if pages_with_text < 1 or text_chars < MIN_TEXT_CHARS_SHORT:
            reason = "LOW_TEXT_COVERAGE" if pages_with_text < 1 else "LOW_TEXT_CHARS"
            return {
                "ok": False,
                "error": {
                    "code": "NOT_INDEXABLE",
                    "message": (
                        f"Documento non indicizzabile: troppo poco testo per {pages_total} pagina/e."
                    ),
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
                    "message": (
                        "Documento non indicizzabile: testo insufficiente o troppo poco distribuito sulle pagine."
                    ),
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

        # NOT_INDEXABLE (V1)
    if text_chars < MIN_TEXT_CHARS:
        return {
            "ok": False,
            "error": {
                "code": "NOT_INDEXABLE",
                "message": f"Documento non indicizzabile: testo troppo corto (< {MIN_TEXT_CHARS} caratteri). Probabile PDF scannerizzato o vuoto.",
            },
            "pages_detected": pages_total,
            "text_chars": text_chars,
        }

        return {
        "ok": True,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "pages_detected": pages_total,  # compat
        "text_chars": text_chars,
    }

