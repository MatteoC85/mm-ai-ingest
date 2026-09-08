"""P4-C9: extracted source parsing implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class NormalizeStructuredSourceTypeRuntime:
    HTTPException: Any
    STRUCTURED_SOURCE_TYPES: Any
    re: Any


def normalize_structured_source_type(source_type: str, *, runtime: NormalizeStructuredSourceTypeRuntime) -> str:
    HTTPException = runtime.HTTPException
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    re = runtime.re
    s = re.sub(r"[\s\-]+", "_", str(source_type or "").strip().lower())

    aliases = {
        "procedure": "procedure",
        "step": "step",
        "ps": "ps",
        "problemsolution": "ps",
        "problem_solution": "ps",
        "problem_solution_item": "ps",
        "md_photo": "md_photo",
        "machine_detail_photo": "md_photo",
        "photo_machine_detail": "md_photo",
        "md_video": "md_video",
        "machine_detail_video": "md_video",
        "video_machine_detail": "md_video",
    }

    s = aliases.get(s, s)
    if s not in STRUCTURED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported source_type: {source_type}")

    return s


@dataclass(frozen=True)
class NormalizeStructuredSourceKeyRuntime:
    _normalize_structured_source_type: Callable[..., Any]


def normalize_structured_source_key(source_type: str, source_id_or_key: Any, *, runtime: NormalizeStructuredSourceKeyRuntime) -> str:
    _normalize_structured_source_type = runtime._normalize_structured_source_type
    raw = str(source_id_or_key or "").strip()
    if not raw:
        return ""
    prefix = f"{_normalize_structured_source_type(source_type)}:"
    return raw if raw.lower().startswith(prefix.lower()) else prefix + raw


@dataclass(frozen=True)
class IsStructuredSourceKeyRuntime:
    STRUCTURED_SOURCE_TYPES: Any


def is_structured_source_key(value: str, *, runtime: IsStructuredSourceKeyRuntime) -> bool:
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    v = str(value or "").strip().lower()
    if ":" not in v:
        return False

    prefix = v.split(":", 1)[0].strip()
    return prefix in STRUCTURED_SOURCE_TYPES


@dataclass(frozen=True)
class ProcedureUiRawTextRuntime:
    pass


def procedure_ui_raw_text(citation: dict, *, runtime: ProcedureUiRawTextRuntime) -> str:
    return str(
        (citation or {}).get("chunk_full")
        or (citation or {}).get("snippet")
        or (citation or {}).get("snippet_clean")
        or ""
    ).replace("\r\n", "\n").replace("\r", "\n").strip()


@dataclass(frozen=True)
class ProcedureUiFieldsRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    _procedure_ui_raw_text: Callable[..., Any]
    re: Any


def procedure_ui_fields(citation: dict, *, runtime: ProcedureUiFieldsRuntime) -> dict[str, str]:
    """Read complete structured fields, including multiline descriptions."""
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    _procedure_ui_raw_text = runtime._procedure_ui_raw_text
    re = runtime.re
    raw = _procedure_ui_raw_text(citation)
    if not raw:
        return {}

    known = {
        "source_type", "title", "procedure_type", "short_description",
        "step_number", "description", "category", "solution", "notes",
        "procedure", "procedura", "parent_procedure", "parent_procedura",
        "procedure_id", "procedura_id", "procedure_code", "codice_procedura",
        "procedure_title", "titolo_procedura", "related_procedure",
        "procedura_collegata",
    }
    values: dict[str, list[str]] = {}
    current = ""
    for raw_line in raw.split("\n"):
        line = re.sub(r"[ \t]+", " ", str(raw_line or "")).strip()
        if not line:
            continue
        key = ""
        value = ""
        if ":" in line:
            key_part, value_part = line.split(":", 1)
            normalized = re.sub(
                r"[^a-z0-9à-öø-ÿ]+",
                "_",
                _normalize_unicode_advanced(key_part).lower(),
            ).strip("_")
            if normalized in known:
                key = normalized
                value = value_part.strip()
        if key:
            current = key
            values.setdefault(current, [])
            if value:
                values[current].append(value)
        elif current:
            values.setdefault(current, []).append(line)

    # Compatibility with old one-line records. Prefer the complete values above.
    out = dict(_parse_structured_source_fields(raw))
    for key, chunks in values.items():
        value = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        if value:
            out[key] = value
    return out


@dataclass(frozen=True)
class ProcedureUiCleanRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def procedure_ui_clean(value: Any, *, finish_sentence: bool = False, runtime: ProcedureUiCleanRuntime) -> str:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    text = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(value or ""))).strip()
    if not text:
        return ""
    text = re.sub(
        r"(?i)\b(?:codice\s+interno|internal\s+code)\s+[A-Z0-9._/-]+\s*[.;,:–—-]?\s*",
        "",
        text,
    )
    text = re.sub(r"(?i)\s*[—–-]\s*(?:PROCEDURA|PROCEDURE)\s*:.*$", "", text)
    text = re.sub(r"(?i)\b(?:MEDIA\s+CORRELATI|RELATED\s+MEDIA)\b.*$", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—\t\n")
    if finish_sentence and text and text[-1] not in ".!?":
        text += "."
    return text


@dataclass(frozen=True)
class ProcedureUiSectionsRuntime:
    _procedure_ui_clean: Callable[..., Any]
    re: Any


def procedure_ui_sections(value: str, *, runtime: ProcedureUiSectionsRuntime) -> dict[str, str]:
    """Split the labels used in Procedure/Step descriptions, in IT and EN."""
    _procedure_ui_clean = runtime._procedure_ui_clean
    re = runtime.re
    text = str(value or "").replace("\r", "\n").strip()
    if not text:
        return {}
    labels = [
        ("instruction", "AZIONE OPERATIVA"),
        ("instruction", "OPERATIONAL ACTION"),
        ("instruction", "ISTRUZIONE OPERATIVA"),
        ("instruction", "OPERATIONAL INSTRUCTION"),
        ("instruction", "OPERATING INSTRUCTION"),
        ("safety", "NOTA DI SICUREZZA"),
        ("safety", "SAFETY NOTE"),
        ("media", "MEDIA CORRELATI"),
        ("media", "RELATED MEDIA"),
        ("duration", "DURATA INDICATIVA"),
        ("duration", "INDICATIVE DURATION"),
        ("safety_level", "LIVELLO DI SICUREZZA"),
        ("safety_level", "SAFETY LEVEL"),
        ("technical_sources", "RIFERIMENTI TECNICI"),
        ("technical_sources", "TECHNICAL REFERENCES"),
        ("technical_sources", "FONTI TECNICHE"),
        ("technical_sources", "TECHNICAL SOURCES"),
        ("recipients", "DESTINATARI"),
        ("recipients", "RECIPIENTS"),
        ("purpose", "SCOPO"),
        ("purpose", "PURPOSE"),
        ("purpose", "OBJECTIVE"),
    ]
    labels.sort(key=lambda item: len(item[1]), reverse=True)
    key_by_label = {label.lower(): key for key, label in labels}
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(label) for _, label in labels) + r")\b\s*[:：–—-]?\s*",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {"body": _procedure_ui_clean(text)}

    out: dict[str, str] = {}
    prefix = _procedure_ui_clean(text[:matches[0].start()])
    if prefix:
        out["body"] = prefix
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = _procedure_ui_clean(text[match.end():end])
        key = key_by_label.get(str(match.group(1) or "").lower(), "")
        if key and body:
            out[key] = (out.get(key, "") + " " + body).strip()
    return out


@dataclass(frozen=True)
class ProcedureUiIsSafetySetupRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def procedure_ui_is_safety_setup(text: str, *, runtime: ProcedureUiIsSafetySetupRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    normalized = _normalize_unicode_advanced(text or "").lower()
    return bool(re.search(
        r"\b(?:sicurezza|sicure|arrestare|isolamento|isolare|emergenza|ripari|"
        r"safe|safety|stop|isolation|isolate|emergency|guard|guards|lockout)\b",
        normalized,
    ))


@dataclass(frozen=True)
class V12StepContractTextRuntime:
    _procedure_ui_fields: Callable[..., Any]
    _procedure_ui_sections: Callable[..., Any]


def v12_step_contract_text(step: dict, *, runtime: V12StepContractTextRuntime) -> str:
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_sections = runtime._procedure_ui_sections
    fields = _procedure_ui_fields(step)
    sections = _procedure_ui_sections(fields.get("description") or "")
    return " ".join(
        [
            str(fields.get("title") or ""),
            str(sections.get("instruction") or sections.get("body") or ""),
            str(sections.get("safety") or ""),
        ]
    ).strip()


