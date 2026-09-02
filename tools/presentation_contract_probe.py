#!/usr/bin/env python3
"""Deterministic characterization of citation presentation and resource links."""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
import types
from pathlib import Path
from typing import Any


def install_stubs() -> None:
    psycopg2 = types.ModuleType("psycopg2")
    psycopg2.connect = lambda **kwargs: None
    sys.modules["psycopg2"] = psycopg2

    google = sys.modules.get("google") or types.ModuleType("google")
    if not hasattr(google, "__path__"):
        google.__path__ = []
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    tasks = types.ModuleType("google.cloud.tasks_v2")

    class CloudTasksClient:
        pass

    class HttpMethod:
        POST = "POST"

    tasks.CloudTasksClient = CloudTasksClient
    tasks.HttpMethod = HttpMethod
    cloud.tasks_v2 = tasks
    google.cloud = cloud
    sys.modules["google"] = google
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.tasks_v2"] = tasks


install_stubs()
sys.path.insert(0, str(Path.cwd()))
import main  # noqa: E402


def normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(normalize(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def capture(fn, *args, **kwargs) -> dict:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            value = fn(*args, **kwargs)
        error = None
    except Exception as exc:
        value = None
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "status_code": getattr(exc, "status_code", None),
            "detail": normalize(getattr(exc, "detail", None)),
        }
    return {
        "value": normalize(value),
        "error": error,
        "stdout": [line for line in stream.getvalue().splitlines() if line.strip()],
    }


class Patch:
    def __init__(self, **values: Any):
        self.values = values
        self.originals: dict[str, Any] = {}
        self.missing: set[str] = set()

    def __enter__(self):
        for name, value in self.values.items():
            if hasattr(main, name):
                self.originals[name] = getattr(main, name)
            else:
                self.missing.add(name)
            setattr(main, name, value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name in self.values:
            if name in self.missing:
                try:
                    delattr(main, name)
                except AttributeError:
                    pass
            else:
                setattr(main, name, self.originals[name])
        return False


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"


FUNCTIONS = [
    "_clean_display_text",
    "_title_from_file_url",
    "_parse_structured_source_fields",
    "_source_display_meta_for_citation",
    "_structured_source_snippet_for_display",
    "_format_citation_note_lines",
    "_build_rg_links",
    "_looks_like_xlsx_indexed_text",
    "_clean_xlsx_snippet_for_display",
    "_sanitize_citations_for_response",
    "_build_sources_block_from_citations",
]

out: dict[str, Any] = {
    "signatures": {
        name: {
            "signature": safe_signature(getattr(main, name)),
            "module": getattr(getattr(main, name), "__module__", ""),
        }
        for name in FUNCTIONS
    }
}

# ---------------------------------------------------------------------------
# Primitive text and structured-field presentation
# ---------------------------------------------------------------------------
for name, value, max_len in [
    ("none", None, 140),
    ("whitespace", "  A\n\t  B   C  ", 140),
    ("edge_punctuation", " --:; Test value.,— ", 140),
    ("only_punctuation", " --:;,.— ", 140),
    ("numeric", 12345, 140),
    ("no_limit", "abcdef", 0),
    ("one_char_limit", "abcdef", 1),
    ("five_char_limit", "abcdef gh", 5),
    ("unicode", "  Pressione ≥ 5 bar  ", 140),
]:
    out[f"clean_{name}"] = capture(main._clean_display_text, value, max_len=max_len)

for name, url in {
    "empty": "",
    "pdf_query_fragment": "https://host.test/folder/My_Manual-v2.PDF?token=x#page=9",
    "xlsx_encoded": "https://host.test/files/Capacity%20Table.xlsx",
    "trailing_slash": "https://host.test/folder/",
    "image": "https://host.test/a/product-photo.JPEG",
    "no_extension": "https://host.test/a/drive_unit",
}.items():
    out[f"title_{name}"] = capture(main._title_from_file_url, url)

with Patch(
    urlparse=lambda _value: types.SimpleNamespace(path="/patched/Encoded%20Name.PDF"),
    unquote=lambda value: value.replace("%20", "_"),
    _clean_display_text=lambda value, max_len=140: f"CLEAN<{value}>:{max_len}",
):
    out["title_late_bound_helpers"] = capture(main._title_from_file_url, "ignored")

structured_text = (
    "SOURCE_TYPE: step\r\n"
    " TITLE : First title \r\n"
    "TITLE: second title\n"
    "DESCRIPTION: Open valve: slowly\n"
    "NO_COLON\n"
    "EMPTY:   \n"
    " STEP NUMBER :  7  "
)
out["parse_structured_fields"] = capture(main._parse_structured_source_fields, structured_text)
with Patch(_clean_display_text=lambda value, max_len=140: f"FIELD<{str(value).strip()}>:{max_len}"):
    out["parse_fields_late_bound_clean"] = capture(
        main._parse_structured_source_fields,
        "TITLE: Pump\nDESCRIPTION: Starts line",
    )

# ---------------------------------------------------------------------------
# Source display metadata and human-readable snippets
# ---------------------------------------------------------------------------
meta_cases = {
    "document_page": {
        "citation_id": "D1:p3",
        "bubble_document_id": "D1",
        "page_from": 3,
        "page_to": 3,
        "display_title": " Existing title ",
        "snippet": "text",
    },
    "document_range_from_url": {
        "citation_id": "D2:p4-6",
        "bubble_document_id": "D2",
        "page_from": "4",
        "page_to": "6",
        "snippet": "text",
    },
    "procedure": {
        "citation_id": "P1",
        "bubble_document_id": "procedure:proc-1",
        "chunk_full": "SOURCE_TYPE: procedure\nTITLE: Cambio formato\nSHORT_DESCRIPTION: Sequenza completa",
    },
    "step": {
        "citation_id": "S1",
        "bubble_document_id": "step:step-1",
        "chunk_full": "SOURCE_TYPE: step\nSTEP_NUMBER: 2\nTITLE: Bloccare energia\nDESCRIPTION: Applicare LOTO",
    },
    "step_no_number": {
        "citation_id": "S2",
        "bubble_document_id": "step:step-2",
        "chunk_full": "SOURCE_TYPE: step\nTITLE: Verifica finale",
    },
    "ps": {
        "citation_id": "PS1",
        "bubble_document_id": "ps:ps-1",
        "chunk_full": "SOURCE_TYPE: ps\nTITLE: Asse non parte\nCATEGORY: Assi\nDESCRIPTION: Nessun movimento",
    },
    "ps_alias_problem_solution": {
        "citation_id": "PS2",
        "bubble_document_id": "problem_solution:ps-2",
        "chunk_full": "TITLE: Pressa ferma\nCATEGORY: Sicurezze",
    },
    "photo": {
        "citation_id": "F1",
        "bubble_document_id": "md_photo:photo-1",
        "chunk_full": "SOURCE_TYPE: md_photo\nTITLE: Sensore ingresso\nDESCRIPTION: Posizione del sensore",
    },
    "video": {
        "citation_id": "V1",
        "bubble_document_id": "md_video:video-1",
        "chunk_full": "SOURCE_TYPE: md_video\nDESCRIPTION: Regolazione guidata",
    },
    "empty_document": {
        "citation_id": "E1",
        "bubble_document_id": "",
        "page_from": None,
        "page_to": None,
    },
}
meta_urls = {
    "document_range_from_url": "https://host.test/docs/Capacity_Table.pdf",
    "empty_document": "",
}
for name, citation in meta_cases.items():
    out[f"meta_{name}"] = capture(
        main._source_display_meta_for_citation,
        citation,
        file_url=meta_urls.get(name, ""),
    )

with Patch(
    _source_type_from_document_id=lambda _value: "procedure",
    _is_structured_source_key=lambda _value: False,
    STRUCTURED_SOURCE_TYPES={"procedure"},
    _parse_structured_source_fields=lambda _text: {"title": "Patched title"},
    _clean_display_text=lambda value, max_len=140: f"META<{value}>:{max_len}",
    _title_from_file_url=lambda _url: "Patched URL title",
    _safe_int=lambda value, default=0: 99 if value is not None else default,
):
    out["meta_late_bound_helpers"] = capture(
        main._source_display_meta_for_citation,
        {"bubble_document_id": "x", "page_from": 1, "page_to": 1},
        file_url="https://ignored",
    )

snippet_cases = {
    "procedure": meta_cases["procedure"],
    "step": meta_cases["step"],
    "ps": {
        "bubble_document_id": "ps:ps-1",
        "chunk_full": "SOURCE_TYPE: ps\nTITLE: Asse non parte\nCATEGORY: Assi\nDESCRIPTION: Nessun movimento\nSOLUTION: Ripristinare consenso\nNOTES: Verificare cablaggio",
    },
    "photo": meta_cases["photo"],
    "video": meta_cases["video"],
    "manual": {"bubble_document_id": "D1", "chunk_full": "ordinary text"},
    "long": {
        "bubble_document_id": "step:long",
        "chunk_full": "STEP_NUMBER: 10\nTITLE: " + ("Long title " * 30),
    },
}
for name, citation in snippet_cases.items():
    kwargs = {"max_len": 80} if name == "long" else {}
    out[f"structured_snippet_{name}"] = capture(
        main._structured_source_snippet_for_display,
        citation,
        **kwargs,
    )

with Patch(
    _source_type_from_document_id=lambda _value: "step",
    _parse_structured_source_fields=lambda _text: {
        "step_number": "8",
        "title": "Patched",
        "description": "Detail",
    },
    _clean_display_text=lambda value, max_len=140: f"SNIP<{value}>:{max_len}",
):
    out["structured_snippet_late_bound_helpers"] = capture(
        main._structured_source_snippet_for_display,
        {"bubble_document_id": "anything", "chunk_full": "ignored"},
        max_len=77,
    )

note_citations = [
    {"display_label": "Manuale - pag. 2", "snippet_clean": "Primo dettaglio"},
    {"display_label": "Manuale - pag. 2", "snippet_clean": "Duplicato"},
    {"citation_id": "C3", "snippet": "Secondo dettaglio"},
    "invalid",
    {"display_label": "Procedura", "snippet_clean": ""},
]
out["citation_notes_it"] = capture(main._format_citation_note_lines, note_citations)
out["citation_notes_en_limit2"] = capture(
    main._format_citation_note_lines,
    note_citations,
    language="en-US",
    max_items=2,
)
out["citation_notes_empty"] = capture(main._format_citation_note_lines, [])
with Patch(_clean_display_text=lambda value, max_len=140: f"NOTE<{value}>:{max_len}"):
    out["citation_notes_late_bound_clean"] = capture(
        main._format_citation_note_lines,
        [{"display_label": "A", "snippet_clean": "B"}],
    )

# ---------------------------------------------------------------------------
# Link building and late-bound repository/meta helpers
# ---------------------------------------------------------------------------
file_map = {
    "D1": "https://files.test/manual.pdf#old",
    "procedure:proc-1": "https://app.test/procedure/1",
    "step:step-1": "https://app.test/step/1",
    "ps:ps-1": "https://app.test/ps/1",
}
link_citations = [
    {
        "citation_id": "D1:p3",
        "bubble_document_id": "D1",
        "page_from": 3,
        "page_to": 4,
        "display_title": "Manuale",
    },
    {
        "citation_id": "P1",
        "bubble_document_id": "procedure:proc-1",
        "page_from": 0,
        "page_to": 0,
        "chunk_full": "TITLE: Cambio formato",
        "ask_structured_direct": True,
    },
    {
        "citation_id": "S1",
        "bubble_document_id": "step:step-1",
        "page_from": -2,
        "page_to": None,
        "chunk_full": "STEP_NUMBER: 1\nTITLE: Sicurezza",
        "ask_structured_manual_support": True,
        "ask_manual_support_kind": "safety",
    },
    {
        "citation_id": "PS1",
        "bubble_document_id": "ps:ps-1",
        "chunk_full": "TITLE: Allarme",
        "evidence_role": "custom",
    },
    {"citation_id": "MISS", "bubble_document_id": "missing", "page_from": 2},
    {"citation_id": "EMPTY", "bubble_document_id": ""},
    "invalid",
]
with Patch(_fetch_document_file_map=lambda company_id, doc_ids: dict(file_map)):
    out["rg_links"] = capture(main._build_rg_links, "company-1", link_citations)

with Patch(
    _fetch_document_file_map=lambda company_id, doc_ids: {"D": "https://x.test/d.pdf"},
    _safe_int=lambda value, default=0: 7,
    _source_display_meta_for_citation=lambda citation, file_url="": {
        "source_type": "patched",
        "source_id": "id",
        "is_structured_source": False,
        "display_title": "title",
        "display_location": "location",
        "display_label": "label",
    },
):
    out["rg_links_late_bound_helpers"] = capture(
        main._build_rg_links,
        "company",
        [{"citation_id": "C", "bubble_document_id": "D", "page_from": 1}],
    )

with Patch(_fetch_document_file_map=lambda company_id, doc_ids: (_ for _ in ()).throw(RuntimeError("file map failed"))):
    out["rg_links_file_map_error"] = capture(
        main._build_rg_links,
        "company",
        [{"citation_id": "C", "bubble_document_id": "D"}],
    )

# ---------------------------------------------------------------------------
# XLSX display, citation sanitization and prompt source block
# ---------------------------------------------------------------------------
for name, value in {
    "file_type": "DOCUMENT_FILE_TYPE: XLSX\nROW 1: A",
    "extraction_mode": "EXTRACTION_MODE: XLSX",
    "kind": "DOCUMENT_KIND: Excel file",
    "lowercase": "document_file_type: xlsx",
    "ordinary": "XLSX appears casually",
    "empty": "",
}.items():
    out[f"looks_xlsx_{name}"] = capture(main._looks_like_xlsx_indexed_text, value)

xlsx_text = (
    "DOCUMENT_FILE_TYPE: XLSX\n"
    "DOCUMENT_KIND: Excel file\n"
    "DOCUMENT_FORMAT_HINTS: x\n"
    "EXTRACTION_MODE: XLSX\n"
    "DOCUMENT_TITLE: Capacity register\n"
    "SHEET_INDEX: 1\n"
    "SHEET_NAME: Data\n"
    "SHEET: Manutenzione\n"
    "SHEET_PART: 1\n"
    "DETECTED_HEADER_ROW: 2\n"
    "HEADER ROW 2: Code | Interval | Action\n"
    "ROW 3: CAL-003 | 500 h | Check filter"
)
out["xlsx_clean"] = capture(main._clean_xlsx_snippet_for_display, xlsx_text)
out["xlsx_clean_short"] = capture(main._clean_xlsx_snippet_for_display, xlsx_text, max_len=75)
with Patch(_clean_display_text=lambda value, max_len=140: f"XLSX<{value}>:{max_len}"):
    out["xlsx_clean_late_bound_clean"] = capture(
        main._clean_xlsx_snippet_for_display,
        "DOCUMENT_TITLE: T\nSHEET: S\nHEADER ROW 1: H\nROW 2: V",
    )

sanitize_input = [
    {
        "citation_id": "D1:p2",
        "bubble_document_id": "D1",
        "page_from": "2",
        "page_to": "2",
        "snippet": "SECTION: Maintenance\nCheck oil level every 500 h.",
        "similarity": 0.88,
    },
    {
        "citation_id": "X1:p1",
        "bubble_document_id": "X1",
        "page_from": 1,
        "page_to": 1,
        "snippet": xlsx_text,
        "retrieval_score": 0.77,
    },
    {
        "citation_id": "P1",
        "bubble_document_id": "procedure:proc-1",
        "snippet": "SOURCE_TYPE: procedure\nTITLE: Cambio formato\nSHORT_DESCRIPTION: Sequenza completa",
        "ask_structured_direct": True,
    },
    {
        "citation_id": "S1",
        "bubble_document_id": "step:step-1",
        "snippet": "SOURCE_TYPE: step\nSTEP_NUMBER: 1\nTITLE: Isolare energia\nDESCRIPTION: Applicare LOTO",
    },
    {
        "citation_id": "M1",
        "bubble_document_id": "M1",
        "snippet": "SECTION: Safety\nLong manual support text for a structured answer.",
        "ask_structured_manual_support": True,
    },
    {"citation_id": "NO_DOC", "bubble_document_id": "", "snippet": "bad"},
    {"citation_id": "", "bubble_document_id": "D2", "snippet": "bad"},
    "invalid",
]
sanitize_file_map = {
    "D1": "https://files.test/Maintenance_Manual.pdf",
    "X1": "https://files.test/Capacity.xlsx",
    "procedure:proc-1": "https://app.test/procedure/1",
    "step:step-1": "https://app.test/step/1",
    "M1": "https://files.test/Support.pdf",
}
with Patch(
    _fetch_document_file_map=lambda company_id, doc_ids: dict(sanitize_file_map),
    _compact_manual_support_snippet_for_display=lambda text, max_len=260: f"COMPACT<{text}>:{max_len}",
):
    out["sanitize_citations"] = capture(
        main._sanitize_citations_for_response,
        sanitize_input,
        company_id="company-1",
    )

with Patch(
    _fetch_document_file_map=lambda company_id, doc_ids: (_ for _ in ()).throw(RuntimeError("db unavailable")),
):
    out["sanitize_file_map_fail_open"] = capture(
        main._sanitize_citations_for_response,
        [sanitize_input[0]],
        company_id="company-1",
    )

with Patch(
    _fetch_document_file_map=lambda company_id, doc_ids: {"D": "https://x.test/d.pdf"},
    _safe_int=lambda value, default=0: 8,
    _source_display_meta_for_citation=lambda citation, file_url="": {
        "source_type": "document",
        "source_id": "patched",
        "is_structured_source": False,
        "display_title": "patched",
        "display_location": "patched",
        "display_label": "patched",
    },
    _looks_like_xlsx_indexed_text=lambda text: False,
    _compact_manual_support_snippet_for_display=lambda text, max_len=260: f"MANUAL<{text}>:{max_len}",
    _clean_display_text=lambda value, max_len=140: f"DISPLAY<{value}>:{max_len}",
):
    out["sanitize_late_bound_helpers"] = capture(
        main._sanitize_citations_for_response,
        [{
            "citation_id": "C",
            "bubble_document_id": "D",
            "snippet": "SECTION: X\nBody",
            "ask_structured_manual_support": True,
        }],
        company_id="company",
    )

source_citations = [
    {
        "citation_id": "C1",
        "bubble_document_id": "D1",
        "page_from": 1,
        "page_to": 2,
        "snippet": "short one",
        "chunk_full": "full one",
    },
    {
        "citation_id": "C2",
        "bubble_document_id": "D2",
        "page_from": 3,
        "page_to": 3,
        "snippet": "short two",
        "chunk_full": "full two",
    },
]
out["sources_block_default"] = capture(
    main._build_sources_block_from_citations,
    source_citations,
)
out["sources_block_full"] = capture(
    main._build_sources_block_from_citations,
    source_citations,
    prefer_chunk_full=True,
)
first_part = "[C1] (doc=D1, p1-2)\nshort one\n"
out["sources_block_exact_first_limit"] = capture(
    main._build_sources_block_from_citations,
    source_citations,
    max_context_chars=len(first_part),
)
out["sources_block_below_first_limit"] = capture(
    main._build_sources_block_from_citations,
    source_citations,
    max_context_chars=len(first_part) - 1,
)
out["sources_block_empty"] = capture(main._build_sources_block_from_citations, [])
out["sources_block_missing_key"] = capture(
    main._build_sources_block_from_citations,
    [{"citation_id": "C", "bubble_document_id": "D", "page_from": 1}],
)

print(json.dumps(normalize(out), ensure_ascii=False, sort_keys=True, indent=2))
