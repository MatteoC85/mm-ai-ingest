#!/usr/bin/env python3
"""Deterministic characterization of response finalization and UI rendering."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import re
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


def html_summary(value: str) -> dict[str, Any]:
    html = str(value or "")
    visible = main._assistant_ui_visible_text_from_html(html) if html else ""
    return {
        "length": len(html),
        "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "visible_text": visible,
        "article_kinds": re.findall(r'data-mm-answer="([^"]+)"', html),
        "render_versions": re.findall(r'data-mm-render="([^"]+)"', html),
        "li_values": [int(x) for x in re.findall(r'<li value="(\d+)"', html)],
        "ordered_list_count": len(re.findall(r"<ol\b", html)),
        "unordered_list_count": len(re.findall(r"<ul\b", html)),
        "contains_raw_script": "<script" in html.lower(),
        "contains_escaped_script": "&lt;script" in html.lower(),
        "contains_sources_heading": bool(re.search(r"\b(?:fonti|sources|link)\b", visible, flags=re.I)),
    }


def response_summary(value: dict) -> dict:
    out = normalize(dict(value or {}))
    raw_html = str((value or {}).get("answer_html") or "")
    if raw_html:
        out["answer_html"] = html_summary(raw_html)
    return out


FUNCTIONS = [
    "_build_structured_procedure_ui_model",
    "_procedure_ui_model_to_text",
    "_assistant_ui_escape",
    "_procedure_ui_model_to_html",
    "_assistant_ui_inline_markup",
    "_assistant_ui_section_kind",
    "_assistant_ui_extract_labeled_line",
    "_assistant_ui_split_inline_numbered",
    "_assistant_ui_render_numbered_cards",
    "_assistant_ui_sentence_has_any",
    "_assistant_ui_promote_unlabelled_sections",
    "_assistant_ui_normalize_markdown_tables",
    "_assistant_ui_root_cause_text",
    "_assistant_ui_generic_html",
    "_assistant_ui_root_cause_html",
    "_assistant_ui_normalize_url_for_key",
    "_assistant_ui_dedupe_links",
    "_assistant_ui_dedupe_citations",
    "_assistant_ui_visible_text_from_html",
    "_assistant_ui_canonical_tokens",
    "_assistant_ui_token_coverage",
    "_assistant_ui_lossless_html",
    "_assistant_ui_finalize_response",
    "_format_structured_procedure_answer_for_ui",
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

# Safe inline rendering and text helpers.
out["escape_untrusted"] = capture(main._assistant_ui_escape, '<script>alert("x")</script> & "')
out["inline_markup"] = capture(main._assistant_ui_inline_markup, "**Bold** and `CODE-7` <img src=x>")
out["section_kinds"] = {
    label: capture(main._assistant_ui_section_kind, label)
    for label in [
        "Causa più probabile",
        "Controlli consigliati",
        "Soluzione",
        "Nota tecnica",
        "Problema",
        "Attenzione",
        "In sintesi",
        "Unknown",
    ]
}
out["labeled_lines"] = {
    line: capture(main._assistant_ui_extract_labeled_line, line)
    for line in [
        "**Controlli consigliati:** 1. Verifica A 2. Verifica B",
        "2. Soluzione: Ripristinare il consenso",
        "Testo libero",
    ]
}
out["split_inline_numbered"] = capture(
    main._assistant_ui_split_inline_numbered,
    "1. Primo controllo; 2. Secondo controllo 3) Terzo controllo",
)
out["normalize_markdown_table"] = capture(
    main._assistant_ui_normalize_markdown_tables,
    "| Voce | Valore |\n|---|---|\n| Pressione | 5 bar |\n| Corsa | 30 mm |",
)

# Generic renderer: explicit numbers must survive separate <ol> fragments.
numbered_answer = (
    "PROC-001 — Avviamento sicuro\n"
    "1. Verificare area e ripari\n"
    "Azione operativa: controllare l'area.\n"
    "Sicurezza: non avviare con anomalie.\n"
    "2. Controllare le utenze\n"
    "Azione operativa: verificare aria e alimentazione.\n"
    "Sicurezza: personale qualificato.\n"
    "3. Inserire l'alimentazione generale"
)
generic_numbered = main._assistant_ui_generic_html(
    numbered_answer,
    links=[],
    citations=[],
    response_language="it",
)
out["generic_numbered_restart"] = html_summary(generic_numbered)
out["generic_untrusted"] = html_summary(
    main._assistant_ui_generic_html(
        "Risposta\n<script>alert(1)</script>\n- **Valore:** `A-7`",
        links=[],
        citations=[],
        response_language="it",
    )
)
out["generic_no_sources"] = html_summary(
    main._assistant_ui_generic_html(
        "Non trovo informazioni sufficienti.",
        links=[],
        citations=[],
        response_language="it",
        status="no_sources",
    )
)

# Lossless fallback preserves original list values too.
lossless_answer = "1. Primo\n\n**Nota**\n2. Secondo\n\n3. Terzo"
out["lossless_numbered_restart"] = html_summary(
    main._assistant_ui_lossless_html(
        lossless_answer,
        response_language="it",
    )
)

# Procedure model and renderer.
procedure_model = {
    "kind": "procedure",
    "language": "it",
    "title": "PROC-001 — Avviamento sicuro",
    "summary": "Sequenza validata di avvio.",
    "personnel": "Operatore qualificato",
    "safety_level": "ALTO",
    "before": [],
    "steps": [
        {
            "display_number": 1,
            "title": "Verificare area e ripari",
            "instruction": "Controllare che l'area sia libera.",
            "safety": "Non avviare in presenza di anomalie.",
        },
        {
            "display_number": 2,
            "title": "Controllare le utenze",
            "instruction": "Verificare collegamento elettrico e pressione 5-6 bar.",
            "safety": "Collegamenti riservati a personale qualificato.",
        },
        {
            "display_number": 3,
            "title": "Inserire l'alimentazione generale",
            "instruction": "Attendere l'avvio di HMI e PLC.",
            "safety": "",
        },
    ],
    "final_checks": [],
    "manual_notes": ["Il manuale richiede un'azione volontaria."],
}
procedure_text = main._procedure_ui_model_to_text(procedure_model, response_language="it")
out["procedure_text"] = procedure_text
out["procedure_html"] = html_summary(
    main._procedure_ui_model_to_html(
        procedure_model,
        links=[{"url": "https://ignored.test"}],
        response_language="it",
    )
)

# Dedupe behavior remains stable.
links = [
    {"bubble_document_id": "D1", "source_type": "document", "url": "https://x.test/a.pdf#page=3", "page_from": 3},
    {"bubble_document_id": "D1", "source_type": "document", "url": "https://x.test/a.pdf#page=3", "page_from": 3},
    {"bubble_document_id": "D1", "source_type": "document", "url": "https://x.test/a.pdf#page=4", "page_from": 4},
    {"bubble_document_id": "step:S1", "source_type": "step", "url": "https://app.test/step/1", "page_from": 0},
    {"bubble_document_id": "step:S1", "source_type": "step", "url": "https://app.test/step/1?x=1", "page_from": 0},
]
citations = [
    {"citation_id": "C1", "bubble_document_id": "D1", "source_type": "document", "page_from": 3, "page_to": 3},
    {"citation_id": "C1b", "bubble_document_id": "D1", "source_type": "document", "page_from": 3, "page_to": 3},
    {"citation_id": "C2", "bubble_document_id": "D1", "source_type": "document", "page_from": 4, "page_to": 4},
    {"citation_id": "S1", "bubble_document_id": "step:S1", "source_type": "step", "page_from": 0, "page_to": 0},
    {"citation_id": "S1b", "bubble_document_id": "step:S1", "source_type": "step", "page_from": 9, "page_to": 9},
]
out["dedupe_links"] = capture(main._assistant_ui_dedupe_links, links, max_items=10)
out["dedupe_citations"] = capture(main._assistant_ui_dedupe_citations, citations, max_items=10)

# Final public envelopes.
procedure_response = {
    "ok": True,
    "status": "answered",
    "answer": procedure_text,
    "language": "it",
    "citations": citations,
    "rg_links": links,
    "_assistant_ui_model": procedure_model,
    "meta": {"cacheable": True},
}
out["finalize_procedure"] = response_summary(
    main._assistant_ui_finalize_response(procedure_response, language="it")
)

generic_response = {
    "ok": True,
    "status": "answered",
    "answer": numbered_answer,
    "language": "it",
    "citations": citations,
    "rg_links": links,
    "meta": {"cacheable": True},
}
out["finalize_generic"] = response_summary(
    main._assistant_ui_finalize_response(generic_response, language="it")
)

root_response = {
    "ok": True,
    "status": "answered",
    "effective_mode": "root_cause",
    "problem_summary": "La macchina non si abilita.",
    "possible_causes": [
        {
            "rank": 1,
            "cause": "Ripristino zone incompleto",
            "why": "Ogni zona richiede il proprio reset.",
            "checks": ["Premere i due reset anteriori", "Premere il reset posteriore"],
        },
        {
            "rank": 2,
            "cause": "Emergenza ancora attiva",
            "why": "L'emergenza ha priorità.",
            "checks": ["Controllare tutti i funghi"],
        },
    ],
    "recommended_next_checks": ["Premere i due reset anteriori", "Verificare HMI"],
    "citations": citations,
    "rg_links": links,
}
out["root_text"] = main._assistant_ui_root_cause_text(root_response, response_language="it")
out["finalize_root_cause"] = response_summary(
    main._assistant_ui_finalize_response(root_response, language="it")
)

# Canonicality and fail-safe behavior.
out["token_coverage"] = {
    "exact": capture(main._assistant_ui_token_coverage, "Valore 5 bar", "Valore 5 bar"),
    "partial": capture(main._assistant_ui_token_coverage, "Valore 5 bar", "Valore"),
    "list_markers_ignored": capture(main._assistant_ui_token_coverage, "1. Primo\n2. Secondo", "Primo Secondo"),
}
with Patch(ASSISTANT_UI_MAX_HTML_CHARS=40):
    out["finalize_html_limit_fallback"] = response_summary(
        main._assistant_ui_finalize_response(procedure_response, language="it")
    )

# Late-bound callbacks remain live through the composition root.
with Patch(
    _safe_int=lambda value, default=0: 77,
    _source_type_from_document_id=lambda value: "step",
    STRUCTURED_SOURCE_TYPES={"step"},
):
    out["late_bound_helpers"] = {
        "procedure_html": html_summary(
            main._procedure_ui_model_to_html(
                {
                    "kind": "procedure",
                    "title": "Patched",
                    "steps": [{"display_number": 2, "title": "X", "instruction": "Y"}],
                },
                links=[],
                response_language="it",
            )
        ),
        "dedupe": capture(
            main._assistant_ui_dedupe_links,
            [
                {"bubble_document_id": "anything", "url": "https://x.test/a", "page_from": 1},
                {"bubble_document_id": "anything", "url": "https://x.test/b", "page_from": 2},
            ],
            max_items=10,
        ),
    }

print(json.dumps(normalize(out), ensure_ascii=False, sort_keys=True, indent=2))
