"""P4-C9: extracted retrieval primitives implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class SafeIntRuntime:
    pass


def safe_int(value: Any, default: int = 0, *, runtime: SafeIntRuntime) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


@dataclass(frozen=True)
class ExtractCodeTokensRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def extract_code_tokens(q: str, *, runtime: ExtractCodeTokensRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q = _normalize_unicode_advanced(q or "")
    if not q.strip():
        return []

    raw = re.findall(r"\b[A-Za-z0-9_\-/]{4,}\b", q)
    out = []
    seen = set()

    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue

        has_digit = any(ch.isdigit() for ch in tok)
        has_sep = ("_" in tok) or ("-" in tok) or ("/" in tok)
        has_upper = any(ch.isupper() for ch in tok)

        if not (has_digit or has_sep or (has_upper and len(tok) >= 6)):
            continue

        key = tok.upper()
        if key in seen:
            continue

        seen.add(key)
        out.append(tok)

    return out[:5]


@dataclass(frozen=True)
class DedupTextValuesRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def dedup_text_values(values: list[str], limit: Optional[int] = None, *, runtime: DedupTextValuesRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    out: list[str] = []
    seen = set()

    for value in values or []:
        s = re.sub(r"\s+", " ", _normalize_unicode_advanced(str(value or ""))).strip()
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(s)

        if limit is not None and len(out) >= limit:
            break

    return out


@dataclass(frozen=True)
class QHasAnyRuntime:
    pass


def q_has_any(q: str, hints: list[str], *, runtime: QHasAnyRuntime) -> bool:
    qq = (q or "").lower()
    return any(h in qq for h in hints)


@dataclass(frozen=True)
class CleanTailRuntime:
    pass


def clean_tail(s: str, *, runtime: CleanTailRuntime) -> str:
    return (s or "").rstrip(".,;:!?)\"]}")


@dataclass(frozen=True)
class ExtractFirstRuntime:
    _clean_tail: Callable[..., Any]


def extract_first(regex: re.Pattern, text: str, *, runtime: ExtractFirstRuntime) -> Optional[str]:
    _clean_tail = runtime._clean_tail
    if not text:
        return None
    m = regex.search(text)
    if not m:
        return None
    return _clean_tail(m.group(1))


@dataclass(frozen=True)
class CosineSimRuntime:
    pass


def cosine_sim(a: list[float], b: list[float], *, runtime: CosineSimRuntime) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0

    for i in range(min(len(a), len(b))):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb

    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


@dataclass(frozen=True)
class ExtractSectionFromTextRuntime:
    re: Any


def extract_section_from_text(text: str, *, runtime: ExtractSectionFromTextRuntime) -> str:
    re = runtime.re
    if not text:
        return ""
    m = re.search(r"^SECTION:\s*(.+)$", text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "")[:120]


@dataclass(frozen=True)
class UniqueNonEmptyStringsRuntime:
    pass


def unique_non_empty_strings(items: list[Any], limit: Optional[int] = None, *, runtime: UniqueNonEmptyStringsRuntime) -> list[str]:
    out: list[str] = []
    seen = set()

    for item in items or []:
        s = str(item or "").strip()
        if not s:
            continue

        k = s.lower()
        if k in seen:
            continue

        seen.add(k)
        out.append(s)

        if limit is not None and len(out) >= limit:
            break

    return out


@dataclass(frozen=True)
class AskStructuredFieldValueRuntime:
    _clean_display_text: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]


def ask_structured_field_value(c: dict, *keys: str, limit: int = 240, runtime: AskStructuredFieldValueRuntime) -> str:
    _clean_display_text = runtime._clean_display_text
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    text = str((c or {}).get("chunk_full") or (c or {}).get("snippet") or (c or {}).get("snippet_clean") or "")
    fields = _parse_structured_source_fields(text)
    for k in keys:
        v = _clean_display_text(fields.get(k) or "", max_len=limit)
        if v:
            return v
    return ""


@dataclass(frozen=True)
class AssistantCoreCandidateEvidenceTextRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def assistant_core_candidate_evidence_text(candidate: dict, *, runtime: AssistantCoreCandidateEvidenceTextRuntime) -> str:
    """Text that is legitimately visible to synthesis and grounding checks.

    A source title/display label is part of the indexed source metadata and may
    contain the exact machine/component designation even when a selected page
    does not repeat it. Internal citation ids are intentionally excluded.
    """
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _v13_candidate_text = runtime._v13_candidate_text
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (
        candidate.get("display_title"),
        candidate.get("display_label"),
        _v13_candidate_text(candidate),
    ):
        value = str(raw or "").strip()
        key = _normalize_unicode_advanced(value).casefold()
        if value and key not in seen:
            parts.append(value)
            seen.add(key)
    return "\n".join(parts)


@dataclass(frozen=True)
class ReorderCitationsByPriorityIdsRuntime:
    pass


def reorder_citations_by_priority_ids(
    citations: list[dict],
    priority_ids: list[str],
    max_items: int,
    *, runtime: ReorderCitationsByPriorityIdsRuntime,
) -> list[dict]:
    if not citations:
        return []

    by_id = {
        str(c.get("citation_id") or "").strip(): c
        for c in citations
        if c.get("citation_id")
    }

    out: list[dict] = []
    used = set()

    for cid in priority_ids or []:
        cid = str(cid or "").strip()
        if not cid or cid in used or cid not in by_id:
            continue
        used.add(cid)
        out.append(by_id[cid])

    for c in citations:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        out.append(c)

    return out[:max_items]


