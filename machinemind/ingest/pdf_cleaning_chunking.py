"""Deterministic PDF cleaning and sentence-aware chunking primitives.

This module owns the behavior applied after low-level page extraction and before
embedding/index persistence: repeated header/footer detection, page-noise
removal, conservative paragraph reflow, table-of-contents suppression,
sentence splitting and section-aware overlapping chunks.

Product composition remains in :mod:`main`.  Every helper dependency that was
historically late-bound in ``main`` can be injected explicitly, so the existing
adapter names and monkeypatch surface remain stable.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Optional


TextTransform = Callable[[str], str]
LinePredicate = Callable[[str], bool]
LineTransform = Callable[[str], str]
TopBottomExtractor = Callable[[str, int, int], tuple[list[str], list[str]]]
SentenceSplitter = Callable[[str], list[str]]


PAGE_NOISE_RX = re.compile(
    r"^(?:"
    r"(?:page|pagina)\s*#?(?:\s*of\s*#?)?"
    r"|#\s*/\s*#"
    r")$",
    re.IGNORECASE,
)

SECTION_ENUM_RX = re.compile(r"^\s*\d+(?:\.\d+){0,3}\s+[A-Za-zÀ-ÖØ-öø-ÿ]")
SECTION_ALLCAPS_RX = re.compile(r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9\s\-]{3,}$")
TOC_TITLE_RX = re.compile(r"\b(?:contents|content|indice|index|table of contents)\b", re.IGNORECASE)
SENT_SPLIT_RX = re.compile(r"(?<=[\.\!\?])\s+")


def hf_norm_line(value: str, *, normalize_text_keep_lines: TextTransform) -> str:
    """Normalize a candidate header/footer line for repetition matching."""
    value = normalize_text_keep_lines(value)
    value = value.lower()
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_top_bottom_lines(
    page_text: str,
    top_n: int = 4,
    bottom_n: int = 4,
) -> tuple[list[str], list[str]]:
    """Return non-empty top and bottom lines from one physical page."""
    lines = [line.strip() for line in (page_text or "").split("\n")]
    lines = [line for line in lines if line]
    top = lines[:top_n] if top_n > 0 else []
    bottom = lines[-bottom_n:] if bottom_n > 0 else []
    return top, bottom


def detect_repeated_headers_footers(
    pages_text: list[str],
    top_n: int = 4,
    bottom_n: int = 4,
    min_ratio: float = 0.7,
    min_len: int = 8,
    max_len: int = 140,
    *,
    extract_top_bottom_lines_fn: TopBottomExtractor,
    hf_norm_line_fn: TextTransform,
) -> tuple[set[str], set[str]]:
    """Detect normalized top/bottom lines repeated across enough pages."""
    if not pages_text:
        return set(), set()

    top_counts: dict[str, int] = {}
    bottom_counts: dict[str, int] = {}

    for text in pages_text:
        top, bottom = extract_top_bottom_lines_fn(text, top_n, bottom_n)
        for line in top:
            key = hf_norm_line_fn(line)
            if min_len <= len(key) <= max_len:
                top_counts[key] = top_counts.get(key, 0) + 1
        for line in bottom:
            key = hf_norm_line_fn(line)
            if min_len <= len(key) <= max_len:
                bottom_counts[key] = bottom_counts.get(key, 0) + 1

    threshold = int(math.ceil(len(pages_text) * min_ratio))
    top_keep = {key for key, count in top_counts.items() if count >= threshold}
    bottom_keep = {key for key, count in bottom_counts.items() if count >= threshold}
    return top_keep, bottom_keep


def is_page_noise_line(
    line: str,
    *,
    hf_norm_line_fn: TextTransform,
    page_noise_rx: re.Pattern[str] = PAGE_NOISE_RX,
) -> bool:
    """Return whether a line is only a page-number marker."""
    key = hf_norm_line_fn(line)
    if not key:
        return False
    if len(key) > 40:
        return False
    return page_noise_rx.match(key) is not None


def strip_page_noise_prefix(
    line: str,
    *,
    normalize_unicode_advanced: TextTransform,
) -> str:
    """Remove embedded ``Page X of Y``/``Pagina X di Y`` prefixes."""
    if not line:
        return line

    line = normalize_unicode_advanced(line)
    line = re.sub(
        r"\b(?:page|pagina)\s*\d+\s*(?:of|di)?\s*\d*\b",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def remove_headers_footers_from_page(
    page_text: str,
    header_norm: set[str],
    footer_norm: set[str],
    top_n: int = 4,
    bottom_n: int = 4,
    *,
    strip_page_noise_prefix_fn: LineTransform,
    is_page_noise_line_fn: LinePredicate,
    hf_norm_line_fn: TextTransform,
) -> str:
    """Remove detected repeated headers/footers and page-number noise."""
    lines = [line.strip() for line in (page_text or "").split("\n")]

    scan_n = min(12, len(lines))
    for index in range(scan_n):
        lines[index] = strip_page_noise_prefix_fn(lines[index])
        if is_page_noise_line_fn(lines[index]):
            lines[index] = ""

    for index in range(min(top_n, len(lines))):
        lines[index] = strip_page_noise_prefix_fn(lines[index])
        if hf_norm_line_fn(lines[index]) in header_norm or is_page_noise_line_fn(lines[index]):
            lines[index] = ""

    start = len(lines) - min(bottom_n, len(lines))
    for index in range(start, len(lines)):
        if 0 <= index < len(lines):
            lines[index] = strip_page_noise_prefix_fn(lines[index])
        if 0 <= index < len(lines) and (
            hf_norm_line_fn(lines[index]) in footer_norm
            or is_page_noise_line_fn(lines[index])
        ):
            lines[index] = ""

    output: list[str] = []
    previous_empty = False
    for line in lines:
        line = line.strip()
        if not line:
            if previous_empty:
                continue
            previous_empty = True
            output.append("")
        else:
            previous_empty = False
            output.append(line)

    return "\n".join(output).strip()


def strip_hf_from_chunk_text(
    chunk_text: str,
    header_norm: set[str],
    footer_norm: set[str],
    *,
    hf_norm_line_fn: TextTransform,
    strip_page_noise_prefix_fn: LineTransform,
    is_page_noise_line_fn: LinePredicate,
) -> str:
    """Re-apply header/footer cleanup after chunk overlap/section prefixes."""
    if not chunk_text:
        return ""

    lines = [line.strip() for line in chunk_text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            continue

        key = hf_norm_line_fn(line)
        if key in header_norm or key in footer_norm:
            continue

        stripped = strip_page_noise_prefix_fn(line)
        if is_page_noise_line_fn(stripped):
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def looks_like_bullet(line: str) -> bool:
    """Recognize bullet/list lines that must not be merged into paragraphs."""
    value = (line or "").lstrip()
    if not value:
        return False
    if value.startswith(("•", "·", "*", "-")):
        return True
    if re.match(r"^\(?\d+\)?[.)]\s+", value):
        return True
    if re.match(r"^[a-zA-Z][.)]\s+", value):
        return True
    return False


def looks_like_table(line: str) -> bool:
    """Recognize table-like rows conservatively."""
    if not line:
        return False

    value = line.rstrip("\n")

    if "|" in value and re.search(r"\S+\s*\|\s*\S+", value):
        return True
    if len(re.findall(r"\s{3,}", value)) >= 2:
        return True
    if re.search(r"\.{4,}", value):
        return True

    tokens = re.split(r"\s+", value.strip())
    if len(tokens) >= 6:
        if re.search(r"\d", value) and re.search(
            r"\b(rpm|bar|mm|cm|kg|°c|v|a|hz)\b",
            value,
            re.IGNORECASE,
        ) is not None:
            return True

    if len(tokens) >= 4 and len(re.findall(r"\s{2,}", value)) >= 3:
        return True

    return False


def looks_like_title(line: str) -> bool:
    """Recognize short all-uppercase title rows."""
    value = (line or "").strip()
    if not value:
        return False
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]+", "", value)
    if letters and letters.isupper() and len(value) <= 80:
        return True
    return False


def looks_like_section_header(
    line: str,
    *,
    section_enum_rx: re.Pattern[str] = SECTION_ENUM_RX,
    section_allcaps_rx: re.Pattern[str] = SECTION_ALLCAPS_RX,
) -> bool:
    """Recognize numbered or compact all-uppercase section headers."""
    value = (line or "").strip()
    if not value:
        return False

    if section_enum_rx.match(value) and len(value) <= 120:
        return True
    if section_allcaps_rx.match(value) and len(value.split()) <= 6:
        return True
    return False


def reflow_paragraphs_conservative(
    page_text: str,
    *,
    looks_like_bullet_fn: LinePredicate,
    looks_like_table_fn: LinePredicate,
    looks_like_title_fn: LinePredicate,
) -> str:
    """Join only obvious lowercase continuations, preserving structural rows."""
    if not page_text:
        return ""

    lines = [line.rstrip() for line in page_text.split("\n")]
    output: list[str] = []
    index = 0

    while index < len(lines):
        current = (lines[index] or "").rstrip()
        if not current:
            output.append("")
            index += 1
            continue

        if looks_like_bullet_fn(current) or looks_like_table_fn(current) or looks_like_title_fn(current):
            output.append(current)
            index += 1
            continue

        cursor = index
        buffer = current

        while cursor + 1 < len(lines):
            following = (lines[cursor + 1] or "").strip()
            if not following:
                break

            if looks_like_table_fn(buffer) or looks_like_table_fn(following):
                break
            if looks_like_bullet_fn(following) or looks_like_table_fn(following) or looks_like_title_fn(following):
                break
            if re.search(r"[.;:!?]$", buffer):
                break
            if not re.match(r"^[a-zà-öø-ÿ]", following):
                break

            buffer = buffer + " " + following
            cursor += 1

        output.append(buffer)
        index = cursor + 1

    cleaned: list[str] = []
    previous_empty = False
    for line in output:
        line = line.strip()
        if not line:
            if previous_empty:
                continue
            previous_empty = True
            cleaned.append("")
        else:
            previous_empty = False
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def looks_like_toc_line(line: str) -> bool:
    """Recognize an individual table-of-contents line."""
    value = (line or "").strip()
    if not value:
        return False

    if re.search(r"\.{4,}", value):
        return True
    if re.search(r"\b\d+(?:\.\d+){1,}\b", value) and re.search(r"\s\d{1,4}\s*$", value):
        return True
    if re.search(r"\s\d{1,4}\s*$", value) and len(value) <= 140 and re.search(
        r"[A-Za-zÀ-ÖØ-öø-ÿ]", value
    ):
        return True
    if len(re.findall(r"\s{3,}", value)) >= 2 and re.search(r"\d{1,4}\s*$", value):
        return True

    return False


def strip_toc_lines(
    page_text: str,
    *,
    looks_like_toc_line_fn: LinePredicate,
) -> str:
    """Remove TOC-like rows and discard a page left with only a tiny residue."""
    if not page_text:
        return ""

    lines = [line.rstrip() for line in page_text.split("\n")]
    kept: list[str] = []
    removed = 0

    for line in lines:
        if looks_like_toc_line_fn(line):
            removed += 1
            continue
        kept.append(line)

    output = "\n".join(kept).strip()
    if removed >= 6 and len(output) < 200:
        return ""
    return output


def maybe_remove_toc(
    page_text: str,
    *,
    toc_title_rx: re.Pattern[str] = TOC_TITLE_RX,
    looks_like_toc_line_fn: LinePredicate,
    strip_toc_lines_fn: TextTransform,
) -> str:
    """Suppress a TOC page only when title/density heuristics are strong enough."""
    if not page_text:
        return ""

    if toc_title_rx.search(page_text[:800] or ""):
        return strip_toc_lines_fn(page_text)

    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    if len(lines) < 8:
        return page_text

    toc_hits = sum(1 for line in lines[:40] if looks_like_toc_line_fn(line))
    ratio = toc_hits / max(1, min(len(lines), 40))
    if toc_hits >= 6 and ratio >= 0.30:
        return strip_toc_lines_fn(page_text)

    return page_text


def split_sentences_conservative(
    text: str,
    *,
    sent_split_rx: re.Pattern[str] = SENT_SPLIT_RX,
) -> list[str]:
    """Split prose on sentence punctuation while preserving list rows."""
    if not text:
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    output: list[str] = []

    for line in lines:
        if line.startswith(("•", "·", "*", "-")) or re.match(r"^\(?\d+\)?[.)]\s+", line):
            output.append(line)
            continue

        parts = [part.strip() for part in sent_split_rx.split(line) if part.strip()]
        output.extend(parts)

    return output


def chunk_sentences_with_pages(
    pages: list[tuple[int, str]],
    target_chars: int,
    overlap_chars: int,
    min_chars: int,
    *,
    split_sentences_fn: SentenceSplitter,
    looks_like_section_header_fn: LinePredicate,
) -> list[dict]:
    """Build page-aware, section-bounded chunks with deterministic overlap."""
    sequence: list[tuple[int, str, Optional[str]]] = []
    current_section: Optional[str] = None

    for page_number, text in pages:
        for sentence in split_sentences_fn(text or ""):
            clean_sentence = sentence.strip()
            if looks_like_section_header_fn(clean_sentence):
                current_section = clean_sentence
            sequence.append((int(page_number), clean_sentence, current_section))

    if not sequence:
        return []

    chunks: list[dict] = []
    index = 0
    chunk_index = 1

    while index < len(sequence):
        buffer: list[str] = []
        pages_in_chunk: list[int] = []

        total = 0
        cursor = index
        chunk_section: Optional[str] = sequence[index][2]

        while cursor < len(sequence):
            page_number, sentence, sentence_section = sequence[cursor]
            addition = sentence + " "

            if buffer and sentence_section != chunk_section:
                break

            if total + len(addition) > target_chars and total >= min_chars:
                break

            buffer.append(sentence)
            pages_in_chunk.append(page_number)
            total += len(addition)

            if chunk_section is None and sentence_section is not None:
                chunk_section = sentence_section

            cursor += 1

        if not buffer:
            page_number, sentence, sentence_section = sequence[index]
            buffer = [sentence]
            pages_in_chunk = [page_number]
            chunk_section = sentence_section
            cursor = index + 1

        page_from = min(pages_in_chunk)
        page_to = max(pages_in_chunk)
        chunk_text = "\n".join(buffer).strip()

        if chunk_section:
            chunk_text = f"SECTION: {chunk_section}\n" + chunk_text

        chunks.append(
            {
                "chunk_index": chunk_index,
                "page_from": page_from,
                "page_to": page_to,
                "chunk_text": chunk_text,
            }
        )
        chunk_index += 1

        if overlap_chars > 0:
            carry: list[tuple[int, str]] = []
            carry_len = 0
            carry_cursor = cursor - 1
            while carry_cursor >= index and carry_len < overlap_chars:
                page_number, sentence, sentence_section = sequence[carry_cursor]
                if sentence_section != chunk_section:
                    break
                carry.insert(0, (page_number, sentence))
                carry_len += len(sentence) + 1
                carry_cursor -= 1
            index = max(index + 1, cursor - len(carry))
        else:
            index = cursor

    return [chunk for chunk in chunks if chunk.get("chunk_text")]
