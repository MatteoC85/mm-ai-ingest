"""Citation labels, snippets, prompt blocks and resource links.

This is a behavior-preserving extraction from the certified production monolith.
It receives product-specific helpers from the composition root so source selection,
grounding, tenant scope and historical late-bound monkeypatch points remain outside
this module and unchanged.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Collection
from typing import Any, Optional


CleanText = Callable[..., str]
SafeInt = Callable[..., int]
SourceType = Callable[[str], str]
StructuredPredicate = Callable[[str], bool]
ParseFields = Callable[[str], dict[str, str]]
TitleFromUrl = Callable[[str], str]
FetchFileMap = Callable[[str, list[str]], dict[str, str]]
SourceMeta = Callable[..., dict]
StructuredSnippet = Callable[..., str]
XlsxPredicate = Callable[[str], bool]
XlsxSnippet = Callable[..., str]
CompactManualSnippet = Callable[..., str]


def clean_display_text(value: Any, max_len: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" -–—:;,.\t\n")
    if max_len and len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def title_from_file_url(
    file_url: str,
    *,
    urlparse_fn: Callable[[str], Any],
    unquote_fn: Callable[[str], str],
    clean_text_fn: CleanText,
) -> str:
    file_url = str(file_url or "").strip()
    if not file_url:
        return ""
    try:
        parsed = urlparse_fn(file_url.split("#", 1)[0].split("?", 1)[0])
        name = unquote_fn((parsed.path or "").rstrip("/").split("/")[-1])
    except Exception:
        name = ""
    name = re.sub(r"[_-]+", " ", name or "")
    name = re.sub(
        r"\.(pdf|docx?|xlsx?|pptx?|txt|png|jpe?g|webp|mp4|mov|avi)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return clean_text_fn(name, max_len=90)


def parse_structured_source_fields(
    text: str,
    *,
    clean_text_fn: CleanText,
) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in str(text or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = re.sub(r"\s+", "_", key.strip().lower())
        value = clean_text_fn(value, max_len=240)
        if key and value and key not in fields:
            fields[key] = value
    return fields


def source_display_meta_for_citation(
    citation: dict,
    file_url: str = "",
    *,
    source_type_fn: SourceType,
    structured_key_fn: StructuredPredicate,
    structured_source_types: Collection[str],
    parse_fields_fn: ParseFields,
    clean_text_fn: CleanText,
    title_from_url_fn: TitleFromUrl,
    safe_int_fn: SafeInt,
) -> dict:
    citation = citation or {}
    document_id = str(citation.get("bubble_document_id") or "").strip()
    source_kind = source_type_fn(document_id)
    raw_prefix = (
        document_id.split(":", 1)[0].strip().lower()
        if ":" in document_id
        else ""
    )
    if source_kind == "manual" and raw_prefix in {
        "problem_solution",
        "problemsolution",
    }:
        source_kind = "ps"
    source_id = (
        document_id.split(":", 1)[1].strip()
        if ":" in document_id
        else document_id
    )
    snippet = (
        citation.get("chunk_full")
        or citation.get("snippet")
        or citation.get("snippet_clean")
        or ""
    )
    fields = parse_fields_fn(snippet)
    page_from = safe_int_fn(citation.get("page_from"), 0)
    page_to = safe_int_fn(citation.get("page_to"), page_from)

    display_title = ""
    display_location = ""
    display_label = ""

    if source_kind == "procedure":
        title = clean_text_fn(
            fields.get("title")
            or fields.get("short_description")
            or "Procedura",
            max_len=90,
        )
        display_title = title
        display_label = f"Procedura: {title}" if title else "Procedura"
    elif source_kind == "step":
        title = clean_text_fn(
            fields.get("title") or fields.get("description") or "Step",
            max_len=90,
        )
        step_no = clean_text_fn(fields.get("step_number") or "", max_len=20)
        display_title = title
        display_location = f"Step {step_no}" if step_no else "Step"
        if step_no and title and title.lower() != "step":
            display_label = f"Step {step_no}: {title}"
        elif title and title.lower() != "step":
            display_label = f"Step: {title}"
        else:
            display_label = display_location
    elif source_kind == "ps":
        title = clean_text_fn(
            fields.get("title")
            or fields.get("category")
            or fields.get("description")
            or "P&S",
            max_len=90,
        )
        category = clean_text_fn(fields.get("category") or "", max_len=60)
        display_title = title
        if category and category.lower() not in title.lower():
            display_label = f"P&S: {title} — Categoria: {category}"
        else:
            display_label = f"P&S: {title}" if title else "P&S"
    elif source_kind == "md_photo":
        title = clean_text_fn(
            fields.get("title") or fields.get("description") or "Foto",
            max_len=90,
        )
        display_title = title
        display_label = (
            f"Foto: {title}" if title and title.lower() != "foto" else "Foto"
        )
    elif source_kind == "md_video":
        title = clean_text_fn(
            fields.get("title") or fields.get("description") or "Video",
            max_len=90,
        )
        display_title = title
        display_label = (
            f"Video: {title}" if title and title.lower() != "video" else "Video"
        )
    else:
        source_kind = "document"
        title = (
            clean_text_fn(str(citation.get("display_title") or ""), max_len=90)
            or title_from_url_fn(file_url)
            or "Documento"
        )
        display_title = title
        if page_from > 0 and page_to > page_from:
            display_location = f"pag. {page_from}/{page_to}"
        elif page_from > 0:
            display_location = f"pag. {page_from}"
        else:
            display_location = ""
        display_label = (
            f"{display_title} - {display_location}"
            if display_location
            else display_title
        )

    display_title = clean_text_fn(display_title, max_len=100)
    display_location = clean_text_fn(display_location, max_len=60)
    display_label = clean_text_fn(
        display_label or display_title or document_id,
        max_len=160,
    )
    return {
        "source_type": source_kind,
        "source_id": source_id,
        "is_structured_source": bool(
            structured_key_fn(document_id)
            or source_kind in structured_source_types
        ),
        "display_title": display_title,
        "display_location": display_location,
        "display_label": display_label,
    }


def structured_source_snippet_for_display(
    citation: dict,
    *,
    max_len: int = 520,
    source_type_fn: SourceType,
    parse_fields_fn: ParseFields,
    clean_text_fn: CleanText,
) -> str:
    citation = citation or {}
    document_id = str(citation.get("bubble_document_id") or "").strip()
    source_kind = source_type_fn(document_id)
    raw_text = str(
        citation.get("chunk_full")
        or citation.get("snippet")
        or citation.get("snippet_clean")
        or ""
    )
    fields = parse_fields_fn(raw_text)

    def value(*keys: str, limit: int = 240) -> str:
        for key in keys:
            current = clean_text_fn(fields.get(key) or "", max_len=limit)
            if current:
                return current
        return ""

    lines: list[str] = []
    if source_kind == "procedure":
        title = value("title", limit=90) or "Procedura"
        procedure_type = value("procedure_type", limit=80)
        description = value("short_description", "description", limit=260)
        lines.append(f"Procedura: {title}")
        if procedure_type:
            lines.append(f"Tipo: {procedure_type}")
        if description:
            lines.append(f"Descrizione: {description}")
    elif source_kind == "step":
        step_no = value("step_number", limit=20)
        title = value("title", limit=90) or "Step"
        description = value("description", limit=320)
        prefix = f"Step {step_no}:" if step_no else "Step:"
        lines.append(f"{prefix} {title}")
        if description and description.lower() not in title.lower():
            lines.append(description)
    elif source_kind == "ps":
        title = value("title", limit=100) or "Problema/Soluzione"
        category = value("category", limit=70)
        description = value("description", limit=300)
        solution = value("solution", limit=300)
        notes = value("notes", limit=220)
        lines.append(f"P&S: {title}")
        if category:
            lines.append(f"Categoria: {category}")
        if description:
            lines.append(f"Problema: {description}")
        if solution:
            lines.append(f"Soluzione: {solution}")
        if notes:
            lines.append(f"Note: {notes}")
    elif source_kind == "md_photo":
        title = value("title", limit=100) or "Foto"
        description = value("description", limit=320)
        lines.append(f"Foto: {title}")
        if description and description.lower() not in title.lower():
            lines.append(description)
    elif source_kind == "md_video":
        title = value("title", limit=100) or "Video"
        description = value("description", limit=320)
        lines.append(f"Video: {title}")
        if description and description.lower() not in title.lower():
            lines.append(description)
    else:
        return ""

    text = " — ".join([line for line in lines if line]).strip()
    text = re.sub(
        r"\bSOURCE_TYPE\s*:\s*[^—]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" -–—")
    return clean_text_fn(text, max_len=max_len)


def format_citation_note_lines(
    citations: list[dict],
    *,
    language: str = "it",
    max_items: int = 6,
    clean_text_fn: CleanText,
) -> str:
    citations = [item for item in (citations or []) if isinstance(item, dict)]
    if not citations:
        return ""
    header = (
        "Evidence used:"
        if str(language or "").lower().startswith("en")
        else "Fonti utilizzate:"
    )
    lines = [header]
    seen = set()
    for citation in citations:
        label = clean_text_fn(
            citation.get("display_label")
            or citation.get("citation_id")
            or "Fonte",
            max_len=160,
        )
        if not label or label in seen:
            continue
        seen.add(label)
        snippet = clean_text_fn(
            citation.get("snippet_clean") or citation.get("snippet") or "",
            max_len=260,
        )
        lines.append(f"- {label} — {snippet}" if snippet else f"- {label}")
        if len(lines) - 1 >= max_items:
            break
    return "\n".join(lines).strip()


def build_rg_links(
    company_id: str,
    citations: list[dict],
    *,
    fetch_file_map_fn: FetchFileMap,
    safe_int_fn: SafeInt,
    source_meta_fn: SourceMeta,
) -> list[dict]:
    if not citations:
        return []
    document_ids = sorted(
        {
            str(item.get("bubble_document_id") or "").strip()
            for item in citations
            if isinstance(item, dict) and item.get("bubble_document_id")
        }
    )
    if not document_ids:
        return []
    file_map = fetch_file_map_fn(company_id, document_ids)
    out: list[dict] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        document_id = str(citation.get("bubble_document_id") or "").strip()
        if not document_id:
            continue
        file_url = file_map.get(document_id)
        if not file_url:
            continue
        base_url = file_url.split("#", 1)[0]
        page_from = safe_int_fn(citation.get("page_from"), 1)
        if page_from < 1:
            page_from = 1
        meta = source_meta_fn(citation, file_url=file_url)
        final_url = (
            file_url
            if bool(meta.get("is_structured_source"))
            else f"{base_url}#page={page_from}"
        )
        source_type = str(
            meta.get("source_type")
            or citation.get("source_type")
            or "document"
        ).strip().lower()
        if bool(citation.get("ask_structured_manual_support")):
            evidence_role = "manual_support"
        elif bool(citation.get("ask_structured_direct")):
            evidence_role = source_type or "structured"
        else:
            evidence_role = str(
                citation.get("evidence_role") or source_type or "document"
            ).strip().lower()
        out.append(
            {
                "citation_id": citation.get("citation_id"),
                "bubble_document_id": document_id,
                "page_from": safe_int_fn(citation.get("page_from"), page_from),
                "page_to": safe_int_fn(citation.get("page_to"), page_from),
                "url": final_url,
                "evidence_role": evidence_role,
                "ask_structured_manual_support": bool(
                    citation.get("ask_structured_manual_support")
                ),
                "ask_manual_support_kind": str(
                    citation.get("ask_manual_support_kind") or ""
                ),
                **meta,
            }
        )
    return out


def looks_like_xlsx_indexed_text(text: str) -> bool:
    value = str(text or "")
    return (
        "DOCUMENT_FILE_TYPE: XLSX" in value
        or "EXTRACTION_MODE: XLSX" in value
        or "DOCUMENT_KIND: Excel file" in value
    )


def clean_xlsx_snippet_for_display(
    text: str,
    *,
    max_len: int = 520,
    clean_text_fn: CleanText,
) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.match(
            r"^(?:DOCUMENT_FILE_TYPE|DOCUMENT_KIND|DOCUMENT_FORMAT_HINTS|EXTRACTION_MODE|SHEET_INDEX|SHEET_NAME|SHEET_PART|DETECTED_HEADER_ROW)\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        match = re.match(
            r"^DOCUMENT_TITLE\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            title = clean_text_fn(match.group(1), max_len=120)
            if title:
                lines.append(f"Documento Excel: {title}")
            continue
        match = re.match(r"^SHEET\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            sheet = clean_text_fn(match.group(1), max_len=90)
            if sheet:
                lines.append(f"Foglio: {sheet}")
            continue
        match = re.match(
            r"^HEADER ROW\s+\d+\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            header = clean_text_fn(match.group(1), max_len=260)
            if header:
                lines.append(f"Intestazioni: {header}")
            continue
        lines.append(line)
    clean = " — ".join(lines)
    clean = re.sub(r"\s+", " ", clean).strip(" -–—")
    if len(clean) > max_len:
        clean = clean[: max_len - 1].rsplit(" ", 1)[0].strip() + "…"
    return clean


def sanitize_citations_for_response(
    citations: list[dict],
    company_id: Optional[str] = None,
    *,
    fetch_file_map_fn: FetchFileMap,
    safe_int_fn: SafeInt,
    source_meta_fn: SourceMeta,
    structured_snippet_fn: StructuredSnippet,
    xlsx_predicate_fn: XlsxPredicate,
    xlsx_snippet_fn: XlsxSnippet,
    compact_manual_snippet_fn: CompactManualSnippet,
    clean_text_fn: CleanText,
    max_snippet_clean_chars: int,
    manual_support_snippet_chars: int,
    log_fn: Callable[..., Any] = print,
) -> list[dict]:
    out: list[dict] = []
    document_ids = sorted(
        {
            str(item.get("bubble_document_id") or "").strip()
            for item in citations or []
            if isinstance(item, dict) and item.get("bubble_document_id")
        }
    )
    file_map: dict[str, str] = {}
    if company_id and document_ids:
        try:
            file_map = fetch_file_map_fn(company_id, document_ids)
        except Exception as exc:
            log_fn("CITATION_FILE_MAP_FAIL", str(exc))
            file_map = {}

    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        citation_id = str(citation.get("citation_id") or "").strip()
        document_id = str(citation.get("bubble_document_id") or "").strip()
        if not citation_id or not document_id:
            continue
        raw_snippet = (
            citation.get("snippet") or citation.get("chunk_full") or ""
        ).strip()
        base_for_meta = {
            **citation,
            "citation_id": citation_id,
            "bubble_document_id": document_id,
            "page_from": safe_int_fn(citation.get("page_from"), 0),
            "page_to": safe_int_fn(citation.get("page_to"), 0),
            "snippet": raw_snippet,
        }
        meta = source_meta_fn(
            base_for_meta,
            file_url=file_map.get(document_id, ""),
        )
        if bool(meta.get("is_structured_source")):
            clean_snippet = structured_snippet_fn(
                base_for_meta,
                max_len=int(max_snippet_clean_chars or 520),
            )
            if not clean_snippet:
                clean_snippet = re.sub(
                    r"\b(?:SOURCE_TYPE|TITLE|STEP_NUMBER|PROCEDURE_TYPE|SHORT_DESCRIPTION|DESCRIPTION|SOLUTION|NOTES|CATEGORY)\s*:\s*",
                    "",
                    raw_snippet,
                    flags=re.IGNORECASE,
                )
                clean_snippet = re.sub(r"\s*\n\s*", " — ", clean_snippet)
                clean_snippet = re.sub(r"\s+", " ", clean_snippet).strip(" -–—")
                clean_snippet = clean_text_fn(
                    clean_snippet,
                    max_len=int(max_snippet_clean_chars or 520),
                )
        else:
            if xlsx_predicate_fn(raw_snippet):
                clean_snippet = xlsx_snippet_fn(
                    raw_snippet,
                    max_len=int(max_snippet_clean_chars or 520),
                )
            else:
                clean_snippet = re.sub(
                    r"^SECTION:\s*[^\n]+\n?",
                    "",
                    raw_snippet,
                    flags=re.IGNORECASE,
                ).strip()
                clean_snippet = re.sub(r"\s*\n\s*", " ", clean_snippet)
                clean_snippet = re.sub(r"\s+", " ", clean_snippet).strip()
                if bool(citation.get("ask_structured_manual_support")):
                    clean_snippet = compact_manual_snippet_fn(
                        clean_snippet,
                        max_len=max(
                            180,
                            int(manual_support_snippet_chars or 260),
                        ),
                    )
        base = {
            "citation_id": citation_id,
            "bubble_document_id": document_id,
            "page_from": safe_int_fn(citation.get("page_from"), 0),
            "page_to": safe_int_fn(citation.get("page_to"), 0),
            "snippet": raw_snippet,
            "snippet_clean": clean_snippet,
            "similarity": float(
                citation.get("similarity")
                or citation.get("retrieval_score")
                or 0.0
            ),
            "ask_structured_manual_support": bool(
                citation.get("ask_structured_manual_support")
            ),
            "ask_structured_direct": bool(citation.get("ask_structured_direct")),
        }
        base.update(meta)
        out.append(base)
    return out


def build_sources_block_from_citations(
    citations: list[dict],
    *,
    max_context_chars: int,
    prefer_chunk_full: bool = False,
) -> str:
    parts: list[str] = []
    total_chars = 0
    for citation in citations or []:
        if prefer_chunk_full:
            body = (
                citation.get("chunk_full") or citation.get("snippet") or ""
            ).strip()
        else:
            body = (
                citation.get("snippet") or citation.get("chunk_full") or ""
            ).strip()
        part = (
            f"[{citation['citation_id']}] "
            f"(doc={citation['bubble_document_id']}, "
            f"p{citation['page_from']}-{citation['page_to']})\n"
            f"{body}\n"
        )
        if total_chars + len(part) > max_context_chars:
            break
        parts.append(part)
        total_chars += len(part)
    return "\n".join(parts).strip()
