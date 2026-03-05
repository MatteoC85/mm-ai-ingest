import os
import io
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
# -----------------------------
# Code token fallback (SENTINEL / part-number / codici)
# -----------------------------
CODE_TOKEN_REGEX = re.compile(r"\b[A-Z0-9_]{6,}\b")


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
    return out[:5]  # cap low-cost


def _db_find_token_chunk(
    company_id: str,
    machine_id: str,
    token: str,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Fallback deterministico per codici (SENTINEL, part-number, ecc.)
    Cerca ILIKE '%TOKEN%' nello stesso scope dell'ask.
    Ritorna citation-like: {citation_id, bubble_document_id, page_from, page_to, snippet, similarity}
    """
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


def _dedup_citations_by_snippet(citations: list[dict], max_items: int) -> list[dict]:
    """
    Dedup semplice: normalizza snippet e tiene la citation con similarity più alta per snippet.
    """
    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s[:400]  # basta per dedup

    best = {}
    for c in citations:
        k = norm(c.get("snippet", ""))
        if not k:
            k = c.get("citation_id")
        prev = best.get(k)
        if (prev is None) or (float(c.get("similarity", 0)) > float(prev.get("similarity", 0))):
            best[k] = c

    out = list(best.values())
    out.sort(key=lambda x: float(x.get("similarity", 0)), reverse=True)
    return out[:max_items]


def _fts_search_chunks(
    company_id: str,
    machine_id: str,
    q: str,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
) -> list[dict]:
    """
    Lexical retrieval (Postgres FTS) nello stesso scope del dense retrieval.
    Ritorna citations-like: {citation_id, bubble_document_id, page_from, page_to, snippet, similarity}
    similarity per FTS = 0.0 (non confrontabile con cosine).
    """
    q = (q or "").strip()
    if not q:
        return []

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            where = ["company_id = %s"]
            params: list[Any] = [company_id]

            # scope coerente con ask_v1
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
            for (bdid, chunk_index, page_from, page_to, snippet, rank) in rows:
                citation_id = f"{bdid}:p{int(page_from)}-{int(page_to)}:c{int(chunk_index)}"
                out.append(
                    {
                        "citation_id": citation_id,
                        "bubble_document_id": bdid,
                        "page_from": int(page_from),
                        "page_to": int(page_to),
                        "snippet": (snippet or "").strip(),
                        "similarity": 0.0,  # FTS non è cosine
                    }
                )
            return out
    finally:
        conn.close()


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


def _db_conn():
    if not (DB_HOST and DB_USER and DB_PASSWORD):
        raise HTTPException(status_code=500, detail="DB env missing")
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# -----------------------------
# rg_links support
# -----------------------------
def _db_upsert_document_file(company_id: str, bubble_document_id: str, file_url: str) -> None:
    """
    Salva (best-effort) file_url per bubble_document_id, per costruire rg_links in /ask.
    """
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


def _build_rg_links(company_id: str, citations: list[dict]) -> list[dict]:
    """
    Ritorna lista di link reali: file_url#page=page_from per ogni citation.
    """
    if not citations:
        return []

    doc_ids = sorted({str(c.get("bubble_document_id") or "").strip() for c in citations if c.get("bubble_document_id")})
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

        out.append(
            {
                "citation_id": c.get("citation_id"),
                "bubble_document_id": bdid,
                "page_from": int(c.get("page_from") or page_from),
                "page_to": int(c.get("page_to") or page_from),
                "url": f"{base}#page={page_from}",
            }
        )

    return out


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

def _normalize_unicode_advanced(s: str) -> str:
    if not s:
        return ""

    # Unicode compatibility normalization
    s = unicodedata.normalize("NFKC", s)

    # Ligatures (extra safety)
    s = s.replace("ﬁ", "fi")
    s = s.replace("ﬂ", "fl")
    s = s.replace("ﬀ", "ff")
    s = s.replace("ﬃ", "ffi")
    s = s.replace("ﬄ", "ffl")

    # Normalize dashes
    s = s.replace("–", "-").replace("—", "-").replace("‐", "-")

    # Replace non-breaking space
    s = s.replace("\u00A0", " ")

    # Remove zero-width chars
    s = s.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")

    return s

def _dehyphenate_lines_keep_newlines(s: str) -> str:
    """
    Unisce solo casi sicuri:
    - riga termina con '-' e prima c'è una parola (>=2 lettere)
    - riga successiva inizia con lettera minuscola
    - evita casi tipo ISO-9001 / codici con numeri o underscore vicino al '-'
    """
    if not s:
        return ""

    lines = s.split("\n")
    out: list[str] = []
    i = 0

    # pattern conservativi
    end_word_hyphen = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-$")
    next_starts_lower = re.compile(r"^[a-zà-öø-ÿ]")
    avoid_prev = re.compile(r"[0-9_]\-$")     # es: ISO_-, X9-
    avoid_next = re.compile(r"^[0-9_]+")      # es: -123, _ABC

    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]

            cur_stripped = cur.rstrip()
            nxt_stripped = nxt.lstrip()

            # evita liste/bullet
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
                # unisci: "parola-" + "continua" => "parolacontinua"
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
    """
    Estrae testo usando blocks (x0,y0,x1,y1,text,...), ordina per lettura (y,x).
    Più robusto di get_text("text") su bullet/layout.
    """
    blocks = page.get_text("blocks") or []
    # block: (x0, y0, x1, y1, "text", block_no, block_type)
    blocks_sorted = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

    parts: list[str] = []
    for b in blocks_sorted:
        txt = (b[4] or "").strip()
        if not txt:
            continue
        parts.append(txt)

    # separa i blocchi con doppio newline (preserva paragrafi)
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
    # numeri -> #
    s = re.sub(r"\d+", "#", s)
    # spazi multipli già compressi, ma ripuliamo ancora
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_top_bottom_lines(page_text: str, top_n: int = 4, bottom_n: int = 4) -> tuple[list[str], list[str]]:
    lines = [ln.strip() for ln in (page_text or "").split("\n")]
    lines = [ln for ln in lines if ln]  # drop empty

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
    """
    Ritorna set di righe NORMALIZZATE (hf_norm_line) da rimuovere in header e footer.
    """
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
    r"(?:page|pagina)\s*#?(?:\s*of\s*#?)?"     # "page 3" / "pagina 3" / "page 3 of 20"
    r"|#\s*/\s*#"                               # "3/20" normalizzato a "#/#"
    r")$",
    re.IGNORECASE,
)

def _is_page_noise_line(line: str) -> bool:
    k = _hf_norm_line(line)  # lowercase + digits-># + spaces normalized
    if not k:
        return False
    if len(k) > 40:
        return False
    return _PAGE_NOISE_RX.match(k) is not None

_PAGE_PREFIX_STRIP_RX = re.compile(
    r"^\s*(?:page|pagina)\s*\d+(?:\s*(?:of|di)\s*\d+)?\s*",
    re.IGNORECASE,
)

def _strip_page_noise_prefix(line: str) -> str:
    if not line:
        return line

    line = _normalize_unicode_advanced(line)

    # rimuove "Page X of Y" o "Pagina X di Y" ovunque nella riga
    line = re.sub(
        r"\b(?:page|pagina)\s*\d+\s*(?:of|di)?\s*\d*\b",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # pulizia spazi
    line = re.sub(r"\s{2,}", " ", line)

    return line.strip()

def _remove_headers_footers_from_page(
    page_text: str,
    header_norm: set[str],
    footer_norm: set[str],
    top_n: int = 4,
    bottom_n: int = 4,
    debug: bool = False,
    debug_page_number: Optional[int] = None,
) -> str:

    lines = [ln.strip() for ln in (page_text or "").split("\n")]

    if debug and debug_page_number and debug_page_number <= 2:
    for idx in range(min(3, len(lines))):
        before = lines[idx]
        after = _strip_page_noise_prefix(before)

        print(f"MM_DEBUG_CLEAN p{debug_page_number} line{idx}_before={repr(before)}")
        print(f"MM_DEBUG_CLEAN p{debug_page_number} line{idx}_after={repr(after)}")
        
    # remove header candidates only in the top area
    for i in range(min(top_n, len(lines))):
        lines[i] = _strip_page_noise_prefix(lines[i]) 
        if (_hf_norm_line(lines[i]) in header_norm) or _is_page_noise_line(lines[i]):
            lines[i] = ""

    # remove footer candidates only in the bottom area
    for j in range(len(lines) - min(bottom_n, len(lines)), len(lines)):
        if 0 <= j < len(lines):
            lines[j] = _strip_page_noise_prefix(lines[j])  

        if 0 <= j < len(lines) and ((_hf_norm_line(lines[j]) in footer_norm) or _is_page_noise_line(lines[j])):
            lines[j] = ""

    # cleanup: collapse multiple empty lines
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

def _looks_like_bullet(line: str) -> bool:
    s = (line or "").lstrip()
    if not s:
        return False
    if s.startswith(("•", "·", "*", "-")):
        return True
    if re.match(r"^\(?\d+\)?[.)]\s+", s):  # "1) " "1. "
        return True
    if re.match(r"^[a-zA-Z][.)]\s+", s):  # "a) " "A. "
        return True
    return False


def _looks_like_table(line: str) -> bool:
    """
    Euristiche conservative per righe tabellari/colonnate (senza coordinate).
    Se True: NON reflow.
    """
    if not line:
        return False

    s = line.rstrip("\n")

    # separatori tabella
    if "|" in s and re.search(r"\S+\s*\|\s*\S+", s):
        return True

    # molti spazi come separatori di colonne (almeno 2 gap larghi)
    if len(re.findall(r"\s{3,}", s)) >= 2:
        return True

    # presenza di "leader dots" tipo "...."
    if re.search(r"\.{4,}", s):
        return True

    # riga con pattern "chiave  valore  valore" (molte colonne)
    tokens = re.split(r"\s+", s.strip())
    if len(tokens) >= 6:
        # se contiene numeri/unità è più probabile tabella
        if re.search(r"\d", s) and (re.search(r"\b(rpm|bar|mm|cm|kg|°c|v|a|hz)\b", s, re.IGNORECASE) is not None):
            return True

    # header tipico tabella: tante parole + spaziatura ampia
    if len(tokens) >= 4 and len(re.findall(r"\s{2,}", s)) >= 3:
        return True

    return False


def _looks_like_title(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    # titolo tipico: tutto maiuscolo e corto
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", "", s)
    if letters and letters.isupper() and len(s) <= 80:
        return True
    return False

def _reflow_paragraphs_conservative(page_text: str) -> str:
    """
    Unisce righe in paragrafi con regole conservative:
    - NON unisce se riga è bullet, tabella, titolo
    - unisce se la riga successiva sembra continuazione (inizia minuscola)
    - unisce se la riga corrente non finisce con punteggiatura forte
    """
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

        # blocchi che non tocchiamo
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

            # non unire se next è bullet/table/title
            if _looks_like_bullet(nxt) or _looks_like_table(nxt) or _looks_like_title(nxt):
                break

            # non unire se current finisce con stop forte
            if re.search(r"[.;:!?]$", buf):
                break

            # unisci solo se next sembra continuazione (inizia minuscola)
            if not re.match(r"^[a-zà-öø-ÿ]", nxt):
                break

            # ok: merge con spazio
            buf = buf + " " + nxt
            j += 1

        out.append(buf)
        i = j + 1

    # collassa righe vuote multiple
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

_SENT_SPLIT_RX = re.compile(r"(?<=[\.\!\?])\s+")

def _split_sentences_conservative(text: str) -> list[str]:
    """
    Split conservativo:
    - preserva righe bullet come "frasi" singole
    - spezza su . ! ? + spazi
    - mantiene newline come separatore forte
    """
    if not text:
        return []

    # prima separa per newline per preservare bullet/list
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out: list[str] = []

    for ln in lines:
        # bullet / enumerazioni: keep as-is
        if ln.startswith(("•", "·", "*", "-")) or re.match(r"^\(?\d+\)?[.)]\s+", ln):
            out.append(ln)
            continue

        # split per punteggiatura forte
        parts = [p.strip() for p in _SENT_SPLIT_RX.split(ln) if p.strip()]
        out.extend(parts)

    return out


def _chunk_sentences_with_pages(
    pages: list[tuple[int, str]],
    target_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[dict]:
    """
    Crea chunk concatenando frasi, mantenendo page_from/page_to.
    Overlap implementato come "carry" delle ultime frasi fino a overlap_chars.
    """
    # 1) espandi in lista di (page_number, sentence)
    seq: list[tuple[int, str]] = []
    for pn, txt in pages:
        for s in _split_sentences_conservative(txt or ""):
            seq.append((int(pn), s))

    if not seq:
        return []

    chunks: list[dict] = []
    i = 0
    chunk_index = 1

    while i < len(seq):
        # build one chunk
        buf: list[str] = []
        pages_in_chunk: list[int] = []

        total = 0
        j = i
        while j < len(seq):
            pn, s = seq[j]
            add = (s + " ")
            if total + len(add) > target_chars and total >= min_chars:
                break
            buf.append(s)
            pages_in_chunk.append(pn)
            total += len(add)
            j += 1

        if not buf:
            # forced add one sentence
            pn, s = seq[i]
            buf = [s]
            pages_in_chunk = [pn]
            j = i + 1

        page_from = min(pages_in_chunk)
        page_to = max(pages_in_chunk)
        chunk_text = "\n".join(buf).strip()

        chunks.append({
            "chunk_index": chunk_index,
            "page_from": page_from,
            "page_to": page_to,
            "chunk_text": chunk_text,
        })
        chunk_index += 1

        # overlap: step avanti ma porta dietro ultime frasi fino a overlap_chars
        if overlap_chars > 0:
            carry = []
            carry_len = 0
            k = j - 1
            while k >= i and carry_len < overlap_chars:
                pn, s = seq[k]
                carry.insert(0, (pn, s))
                carry_len += len(s) + 1
                k -= 1
            # prossimo i = k+1 (cioè riparti includendo carry)
            i = max(i + 1, j - len(carry))
        else:
            i = j

    return [c for c in chunks if c.get("chunk_text")]

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
    document_ids: Optional[Union[List[str], str]] = None  # ✅ robusto
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

    # ✅ rg_links: salva file_url per poter costruire link reali (doc#page)
    _db_upsert_document_file(company_id, bubble_document_id, url)

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
        raw_pages = _extract_pages_with_layout_blocks(data)
        pages_total = len(raw_pages)

        # Detect repeated header/footer lines (normalized)
        header_norm, footer_norm = _detect_repeated_headers_footers(raw_pages)

        pages_text: list[str] = []
        pages_with_text = 0
        text_chars = 0

        debug_clean = (os.environ.get("MM_DEBUG_CLEAN") or "").strip() == "1"

        for page_number, t in enumerate(raw_pages, start=1):
            cleaned = _remove_headers_footers_from_page(
                t,
                header_norm,
                footer_norm,
                debug=debug_clean,
                debug_page_number=page_number,
            )
            cleaned = _reflow_paragraphs_conservative(cleaned)
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

    # ✅ Plan limits (enforcement dopo parsing, prima di scrivere su DB)
    plan_chars_limit = int(payload.plan_embed_chars_limit_total or 0)
    plan_storage_limit = int(payload.plan_index_storage_limit_bytes or 0)

    # Calcolo stima storage per questo documento (semplice/robusto)
    # chunks stimati: basati su testo totale e parametri chunking
    effective_step = max(1, CHUNK_TARGET_CHARS - min(CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS - 1))
    est_chunks = int(math.ceil(max(1, text_chars) / effective_step))

    BYTES_PER_CHAR = 3          # stima conservativa
    BYTES_PER_CHUNK = 2000      # overhead embedding+metadata
    est_storage_bytes = int(text_chars * BYTES_PER_CHAR + est_chunks * BYTES_PER_CHUNK)

    # NOTE: per enforcement "attuale" serve anche il current usage (lo faremo dopo via Bubble counters)
    # Qui facciamo enforcement "per singolo documento" (hard-stop se documento troppo grande per il piano)
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

    # ✅ Total limits (company usage): used - prev + new <= limit
    used_chars_total = int(payload.embed_chars_used_total or 0)
    used_storage_total = int(payload.index_storage_used_total or 0)

    prev_doc_chars = int(payload.doc_prev_embed_chars or 0)
    prev_doc_storage = int(payload.doc_prev_index_storage_bytes or 0)

    # Delta-correct total
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
        "est_storage_bytes": est_storage_bytes,
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
            chunks = _chunk_sentences_with_pages(
                pages=pages,
                target_chars=CHUNK_TARGET_CHARS,
                overlap_chars=CHUNK_OVERLAP_CHARS,
                min_chars=CHUNK_MIN_CHARS,
            )

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

    doc_ids = payload.document_ids

    # ✅ supporta: lista vera oppure stringa "id1, id2, id3"
    if isinstance(doc_ids, str):
        doc_ids = [x.strip() for x in doc_ids.split(",") if x.strip()]

    if isinstance(doc_ids, list):
        doc_ids = [str(x).strip() for x in doc_ids if str(x).strip()]
        if not doc_ids:
            doc_ids = None
    else:
        doc_ids = None

    top_k = int(payload.top_k or 5)
    top_k = max(1, min(top_k, ASK_MAX_TOP_K))

    q_vec = _openai_embed_texts([q])[0]
    q_vec_lit = _vector_literal(q_vec)

    rows = []
    chunks_matching_filter = None

    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            # ✅ 1) Se document_ids presenti, lo scope è ESATTAMENTE quello (company + doc_ids). Niente filtro macchina.
            if doc_ids:
                if payload.debug:
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
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM public.document_chunks
                    WHERE company_id = %s
                    AND bubble_document_id = ANY(%s)
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (ASK_SNIPPET_CHARS, q_vec_lit, company_id, doc_ids, q_vec_lit, top_k),
                )

            # ✅ 2) Backward compat: singolo bubble_document_id (qui mantiene filtro macchina + generici)
            elif payload.bubble_document_id:
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
            # ✅ 3) Default: tutti i doc nello scope macchina + generici
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
            "rg_links": [],  # ✅
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

    # ✅ dedup: evita doppioni (stesso snippet) quando document_ids include doc simili/duplicati
    citations = _dedup_citations_by_snippet(citations, max_items=top_k)

    fts_used = False

    # ✅ Hybrid retrieval: se dense è debole, aggiungi anche risultati FTS nello stesso scope
    if sim_max < ASK_SIM_THRESHOLD:
        fts = _fts_search_chunks(
            company_id=company_id,
            machine_id=machine_id,
            q=q,
            top_k=top_k,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=payload.bubble_document_id.strip() if payload.bubble_document_id else None,
        )
        if fts:
            fts_used = True
            citations = _dedup_citations_by_snippet(citations + fts, max_items=top_k)

    # ✅ Entity fallback: se la domanda chiede sito/email/telefono e lo troviamo negli snippet, rispondiamo anche sotto soglia
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

            return {
                "ok": True,
                "status": "answered",
                "answer": answer,
                "citations": citations,
                "rg_links": rg_links,  # ✅
                "top_k": top_k,
                "similarity_max": sim_max,
                "chat_model": OPENAI_CHAT_MODEL,
            }

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
                return {
                    "ok": True,
                    "status": "answered",
                    "answer": answer,
                    "citations": cit_list,
                    "rg_links": rg_links,  # ✅
                    "top_k": top_k,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                }
        # ✅ Fallback deterministico per "codici" (SENTINEL, part-number, ecc.)
        code_tokens = _extract_code_tokens(q)
        if code_tokens:
            hit = None
            for tok in code_tokens:
                hit = _db_find_token_chunk(
                    company_id=company_id,
                    machine_id=machine_id,
                    token=tok,
                    doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                    bubble_document_id=payload.bubble_document_id.strip() if payload.bubble_document_id else None,
                )
                if hit:
                    break

            if hit:
                cit_list = [hit]
                rg_links = []
                try:
                    rg_links = _build_rg_links(company_id, cit_list)
                except Exception as e:
                    print("RG_LINKS_FAIL", str(e))
                    rg_links = []

                return {
                    "ok": True,
                    "status": "answered",
                    "answer": f"Nel documento compare questa stringa: {code_tokens[0]} [{hit['citation_id']}]",
                    "citations": cit_list,
                    "rg_links": rg_links,
                    "top_k": top_k,
                    "similarity_max": sim_max,
                    "chat_model": OPENAI_CHAT_MODEL,
                }
                
        if not (fts_used and citations):
            resp = {
                "ok": True,
                "status": "no_sources",
                "answer": "Non trovo informazioni nei documenti indicizzati per rispondere.",
                "citations": [],
                "rg_links": [],  # ✅
                "top_k": top_k,
                "similarity_max": sim_max,
            }
            if payload.debug:
                resp["debug"] = {
                    "company_id": company_id,
                    "machine_id": machine_id,
                    "bubble_document_id": payload.bubble_document_id,
                    "chunks_matching_filter": chunks_matching_filter,
                    "fts_used": fts_used,
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
            "rg_links": [],  # ✅
            "top_k": top_k,
            "similarity_max": sim_max,
        }

    rg_links = []
    try:
        rg_links = _build_rg_links(company_id, citations)
    except Exception as e:
        print("RG_LINKS_FAIL", str(e))
        rg_links = []


    return {
        "ok": True,
        "status": "answered",
        "answer": answer,
        "citations": citations,
        "rg_links": rg_links,  # ✅
        "top_k": top_k,
        "similarity_max": sim_max,
        "chat_model": OPENAI_CHAT_MODEL,
    }
# -----------------------------
# Delete PRO (hard delete RAG data)
# -----------------------------
class DeleteDocumentRequest(BaseModel):
    company_id: str
    bubble_document_id: str


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
            # Ordine: chunks -> pages -> files (idempotente)
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
