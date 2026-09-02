"""Shared text normalization and PDF page-extraction primitives.

This module is intentionally limited to deterministic text/PDF mechanics.  It does
not clean document noise, detect headers/footers, chunk text, access storage, call
external services, or know anything about tenant scope.  Product-specific runtime
objects are passed in explicitly so the historical composition-root monkeypatch
surface remains intact.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any


NormalizeText = Callable[[str], str]
PageToTextBlocks = Callable[[Any], str]


def normalize_unicode_advanced(s: str) -> str:
    """Apply the production Unicode/ligature/dash normalization exactly."""
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


def dehyphenate_lines_keep_newlines(s: str) -> str:
    """Join only conservative alphabetic line-break hyphenations."""
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


def normalize_text_keep_lines(
    s: str,
    *,
    normalize_unicode_fn: NormalizeText = normalize_unicode_advanced,
    dehyphenate_fn: NormalizeText = dehyphenate_lines_keep_newlines,
) -> str:
    """Normalize extracted text while retaining meaningful line boundaries."""
    if not s:
        return ""
    s = normalize_unicode_fn(s)
    s = dehyphenate_fn(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")

    out: list[str] = []
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


def pymupdf_page_to_text_blocks(page: Any) -> str:
    """Read PyMuPDF blocks in stable top-to-bottom, left-to-right order."""
    blocks = page.get_text("blocks") or []
    blocks_sorted = sorted(blocks, key=lambda b: (float(b[1]), float(b[0])))

    parts: list[str] = []
    for block in blocks_sorted:
        text = (block[4] or "").strip()
        if not text:
            continue
        parts.append(text)

    return "\n\n".join(parts).strip()


def extract_pages_with_layout_blocks(
    pdf_bytes: bytes,
    *,
    fitz_module: Any,
    page_to_text_blocks_fn: PageToTextBlocks = pymupdf_page_to_text_blocks,
    normalize_text_keep_lines_fn: NormalizeText = normalize_text_keep_lines,
) -> list[str]:
    """Return one normalized string per physical PDF page.

    The fitz module and helper callbacks are injected on every call.  That preserves
    the historical late-bound behavior of ``main.fitz`` and the composition-root
    helper functions used by existing tests and operational probes.
    """
    doc = fitz_module.open(stream=pdf_bytes, filetype="pdf")
    try:
        out: list[str] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page_to_text_blocks_fn(page)
            text = normalize_text_keep_lines_fn(text)
            out.append(text)
        return out
    finally:
        doc.close()
