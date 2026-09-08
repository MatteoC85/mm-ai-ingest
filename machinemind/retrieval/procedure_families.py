"""P4-C8: extracted procedure families implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class V12CodeKeysRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def v12_code_keys(value: str, *, runtime: V12CodeKeysRuntime) -> set[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    raw = _normalize_unicode_advanced(str(value or "")).upper()
    out: set[str] = set()
    pattern = (
        r"\b[A-Z]{1,12}[A-Z0-9]*(?:[-_/.][A-Z0-9]{1,16})+\b"
        r"|\b[A-Z]{2,12}[-_ ]?\d{1,8}[A-Z0-9.,/_-]*\b"
    )
    for token in re.findall(pattern, raw):
        key = re.sub(r"[^A-Z0-9]", "", token)
        if len(key) >= 4 and any(ch.isdigit() for ch in key):
            out.add(key)
    return out


@dataclass(frozen=True)
class V12IdentityTokensRuntime:
    _ask_structured_direct_stopwords: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def v12_identity_tokens(value: str, *, runtime: V12IdentityTokensRuntime) -> set[str]:
    _ask_structured_direct_stopwords = runtime._ask_structured_direct_stopwords
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    text = _normalize_unicode_advanced(str(value or "")).lower()
    stop = _ask_structured_direct_stopwords() | {
        "procedure", "procedura", "step", "passaggio", "fase", "title", "titolo",
        "description", "descrizione", "source", "type", "tipo", "codice", "code",
    }
    return {
        tok for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", text)
        if tok not in stop
    }


@dataclass(frozen=True)
class V12StructuredParentValuesRuntime:
    resolve_procedure_fields: Callable[[], Any]
    _clean_display_text: Callable[..., Any]
    _normalize_structured_source_key: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    re: Any


def v12_structured_parent_values(c: dict, *, runtime: V12StructuredParentValuesRuntime) -> list[str]:
    """Read an explicit parent relation, including legacy DESCRIPTION prefixes."""
    _clean_display_text = runtime._clean_display_text
    _normalize_structured_source_key = runtime._normalize_structured_source_key
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    re = runtime.re
    raw = str((c or {}).get("chunk_full") or (c or {}).get("snippet") or "")
    fields = _parse_structured_source_fields(raw)
    complete_parser = runtime.resolve_procedure_fields()
    if callable(complete_parser):
        try:
            fields.update(complete_parser(c) or {})
        except Exception:
            pass

    keys = (
        "parent_source_key",
        "procedure", "procedura", "parent_procedure", "parent_procedura",
        "parent_procedure_id", "procedure_id", "procedura_id",
        "parent_procedure_code", "procedure_code", "codice_procedura",
        "parent_procedure_title", "procedure_title", "titolo_procedura",
        "related_procedure", "procedura_collegata",
    )
    vals: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        clean = _clean_display_text(value or "", max_len=320)
        norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(clean).lower()).strip()
        if norm and norm not in seen:
            seen.add(norm)
            vals.append(clean)

    for key in keys:
        add(fields.get(key))

    parent_id = str(
        fields.get("parent_procedure_id")
        or fields.get("procedure_id")
        or fields.get("procedura_id")
        or ""
    ).strip()
    if parent_id:
        add(parent_id)
        add(_normalize_structured_source_key("procedure", parent_id))

    # New canonical fields, if present in the raw source.
    for match in re.finditer(
        r"(?im)^\s*(?:PARENT_SOURCE_KEY|PARENT_PROCEDURE_ID|PARENT_PROCEDURE_CODE|"
        r"PARENT_PROCEDURE_TITLE|PROCEDURE_ID|PROCEDURE_CODE)\s*:\s*([^\n]+)",
        raw,
    ):
        add(match.group(1))

    # Legacy indexed Steps already contain:
    # DESCRIPTION: PROCEDURA: PROC-002 — <title>
    # This recovery path is language-neutral at the code level and supports IT/EN labels.
    for match in re.finditer(
        r"(?im)^\s*(?:DESCRIPTION\s*:\s*)?(?:PROCEDURA|PROCEDURE)\s*:\s*([^\n]+)",
        raw,
    ):
        add(match.group(1))

    description = str(fields.get("description") or "")
    for match in re.finditer(
        r"(?i)\b(?:PROCEDURA|PROCEDURE)\s*:\s*([A-Z]{2,12}[-_]?\d{1,8}[A-Z0-9._/-]*)",
        description,
    ):
        add(match.group(1))

    return vals


@dataclass(frozen=True)
class V12ProcedureIdentityTextRuntime:
    resolve_procedure_fields: Callable[[], Any]
    _parse_structured_source_fields: Callable[..., Any]


def v12_procedure_identity_text(c: dict, *, runtime: V12ProcedureIdentityTextRuntime) -> str:
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    raw = str((c or {}).get("chunk_full") or (c or {}).get("snippet") or "")
    fields = _parse_structured_source_fields(raw)
    complete_parser = runtime.resolve_procedure_fields()
    if callable(complete_parser):
        try:
            fields.update(complete_parser(c) or {})
        except Exception:
            pass
    parts = [
        str(c.get("bubble_document_id") or ""),
        raw,
        fields.get("title") or "",
        fields.get("short_description") or "",
        fields.get("description") or "",
        fields.get("procedure_code") or "",
        fields.get("codice") or "",
        fields.get("code") or "",
    ]
    return " ".join(str(x or "") for x in parts)


@dataclass(frozen=True)
class V12StepMatchesProcedureRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    _v12_code_keys: Callable[..., Any]
    _v12_identity_tokens: Callable[..., Any]
    _v12_procedure_identity_text: Callable[..., Any]
    _v12_structured_parent_values: Callable[..., Any]
    re: Any


def v12_step_matches_procedure(step: dict, procedure: dict, *, runtime: V12StepMatchesProcedureRuntime) -> Optional[bool]:
    """True/False when the Step declares a parent; None when no parent is declared."""
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _v12_code_keys = runtime._v12_code_keys
    _v12_identity_tokens = runtime._v12_identity_tokens
    _v12_procedure_identity_text = runtime._v12_procedure_identity_text
    _v12_structured_parent_values = runtime._v12_structured_parent_values
    re = runtime.re
    parents = _v12_structured_parent_values(step)
    if not parents:
        return None

    proc_bdid = str((procedure or {}).get("bubble_document_id") or "").strip()
    proc_id = proc_bdid.split(":", 1)[1].strip() if proc_bdid.lower().startswith("procedure:") else proc_bdid
    proc_text = _v12_procedure_identity_text(procedure)
    proc_norm = re.sub(r"[^a-zà-öø-ÿ0-9]+", " ", _normalize_unicode_advanced(proc_text).lower()).strip()
    proc_codes = _v12_code_keys(proc_text)
    proc_tokens = _v12_identity_tokens(proc_text)

    for parent in parents:
        parent_raw = str(parent or "").strip()
        parent_key = parent_raw if parent_raw.lower().startswith("procedure:") else ""
        if parent_key and parent_key.lower() == proc_bdid.lower():
            return True
        if parent_raw and proc_id and parent_raw.lower() == proc_id.lower():
            return True

        parent_norm = re.sub(
            r"[^a-zà-öø-ÿ0-9]+",
            " ",
            _normalize_unicode_advanced(parent_raw).lower(),
        ).strip()
        parent_codes = _v12_code_keys(parent_raw)
        if parent_codes and proc_codes and (parent_codes & proc_codes):
            return True
        if len(parent_norm) >= 5 and parent_norm in proc_norm:
            return True
        parent_tokens = _v12_identity_tokens(parent_raw)
        if parent_tokens and proc_tokens:
            overlap = len(parent_tokens & proc_tokens)
            ratio = overlap / max(1, min(len(parent_tokens), len(proc_tokens)))
            if overlap >= 2 and ratio >= 0.50:
                return True
    return False


@dataclass(frozen=True)
class V12ChoosePrimaryProcedureRuntime:
    _v12_evidence_role: Callable[..., Any]
    _v12_structured_rank: Callable[..., Any]


def v12_choose_primary_procedure(citations: list[dict], model_used: list[dict], *, runtime: V12ChoosePrimaryProcedureRuntime) -> Optional[dict]:
    _v12_evidence_role = runtime._v12_evidence_role
    _v12_structured_rank = runtime._v12_structured_rank
    procedures = [
        c for c in citations or []
        if isinstance(c, dict) and _v12_evidence_role(c) == "procedure"
    ]
    if not procedures:
        return None
    used_ids = {str(c.get("citation_id") or "") for c in model_used or [] if isinstance(c, dict)}
    return sorted(procedures, key=lambda c: _v12_structured_rank(c, used_ids))[0]


@dataclass(frozen=True)
class V12StepSortKeyRuntime:
    _ask_structured_field_value: Callable[..., Any]
    _safe_int: Callable[..., Any]


def v12_step_sort_key(c: dict, *, runtime: V12StepSortKeyRuntime) -> tuple[int, str]:
    _ask_structured_field_value = runtime._ask_structured_field_value
    _safe_int = runtime._safe_int
    raw = _ask_structured_field_value(c, "step_number", limit=20)
    return (_safe_int(raw, 9999), str(c.get("bubble_document_id") or ""))


@dataclass(frozen=True)
class V12ChoosePrimaryProcedureFamilyRuntime:
    ASK_STRUCTURED_DIRECT_TEXT_CHARS: Any
    _db_fetch_parent_procedure_pages_for_steps: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _safe_int: Callable[..., Any]
    _v12_dedupe_family_steps: Callable[..., Any]
    _v12_evidence_role: Callable[..., Any]
    _v12_expand_primary_procedure_steps: Callable[..., Any]
    _v12_family_score: Callable[..., Any]
    _v12_relation_procedure_candidate: Callable[..., Any]
    _v12_step_matches_procedure: Callable[..., Any]
    _v12_step_sort_key: Callable[..., Any]
    _v12_structured_parent_values: Callable[..., Any]
    _v12_structured_rank: Callable[..., Any]


def v12_choose_primary_procedure_family(
    *,
    company_id: str,
    machine_id: str,
    q: str,
    planner: Optional[dict],
    citations: list[dict],
    model_used: Optional[list[dict]] = None,
    runtime: V12ChoosePrimaryProcedureFamilyRuntime,
) -> tuple[Optional[dict], list[dict], dict]:
    """Recover and rank Procedure families from the Step children first.

    Semantic ranking may omit the parent Procedure or include a nearby parent from
    another family. The canonical relation table is therefore authoritative. The
    family that best covers the router facets and has the strongest admitted Step
    support wins; a Procedure title alone cannot outvote its own children.
    """
    ASK_STRUCTURED_DIRECT_TEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_TEXT_CHARS
    _db_fetch_parent_procedure_pages_for_steps = runtime._db_fetch_parent_procedure_pages_for_steps
    _dedup_text_values = runtime._dedup_text_values
    _safe_int = runtime._safe_int
    _v12_dedupe_family_steps = runtime._v12_dedupe_family_steps
    _v12_evidence_role = runtime._v12_evidence_role
    _v12_expand_primary_procedure_steps = runtime._v12_expand_primary_procedure_steps
    _v12_family_score = runtime._v12_family_score
    _v12_relation_procedure_candidate = runtime._v12_relation_procedure_candidate
    _v12_step_matches_procedure = runtime._v12_step_matches_procedure
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v12_structured_parent_values = runtime._v12_structured_parent_values
    _v12_structured_rank = runtime._v12_structured_rank
    citations = [dict(c) for c in (citations or []) if isinstance(c, dict)]
    model_used = [dict(c) for c in (model_used or []) if isinstance(c, dict)]
    raw_procedures = [c for c in citations if _v12_evidence_role(c) == "procedure"]
    raw_steps = [c for c in citations if _v12_evidence_role(c) == "step"]
    if not raw_steps and not raw_procedures:
        return None, [], {"reason": "no_procedure_or_step_sources"}

    procedure_by_key: dict[str, dict] = {}
    raw_procedure_keys: set[str] = set()
    for procedure in raw_procedures:
        key = str(procedure.get("bubble_document_id") or "").strip()
        if not key:
            continue
        raw_procedure_keys.add(key)
        current = procedure_by_key.get(key)
        if current is None or _v12_structured_rank(procedure, set()) < _v12_structured_rank(current, set()):
            procedure_by_key[key] = dict(procedure)

    child_keys = _dedup_text_values(
        [str(step.get("bubble_document_id") or "").strip() for step in raw_steps],
        limit=500,
    )
    relation_rows = _db_fetch_parent_procedure_pages_for_steps(
        company_id=company_id,
        machine_id=machine_id,
        child_source_keys=child_keys,
        text_chars=max(800, int(ASK_STRUCTURED_DIRECT_TEXT_CHARS or 5000)),
    )
    parent_by_child: dict[str, dict] = {}
    children_by_parent: dict[str, list[dict]] = {}
    for row in relation_rows:
        child = str(row.get("child_source_key") or "").strip()
        parent = str(row.get("parent_source_key") or "").strip()
        if not child or not parent:
            continue
        parent_by_child[child] = dict(row)
        fallback_title = ""
        for step in raw_steps:
            if str(step.get("bubble_document_id") or "").strip() != child:
                continue
            values = _v12_structured_parent_values(step)
            if values:
                fallback_title = str(values[0] or "")
            break
        if parent not in procedure_by_key:
            procedure_by_key[parent] = _v12_relation_procedure_candidate(
                parent_source_key=parent,
                machine_id=str(row.get("machine_id") or machine_id),
                page_number=_safe_int(row.get("page_number"), 1),
                parent_text=str(row.get("parent_text") or ""),
                fallback_title=fallback_title,
            )
        children_by_parent.setdefault(parent, [])

    unresolved_steps: list[dict] = []
    for step in raw_steps:
        child = str(step.get("bubble_document_id") or "").strip()
        relation = parent_by_child.get(child)
        if relation:
            parent = str(relation.get("parent_source_key") or "").strip()
            cc = dict(step)
            cc["_v10_5_parent_source_key"] = parent
            if relation.get("ordinal") is not None:
                cc["structured_relation_ordinal"] = _safe_int(relation.get("ordinal"), 0)
            cc.setdefault("structured_relation_source", "structured_source_relations_seed")
            children_by_parent.setdefault(parent, []).append(cc)
        else:
            unresolved_steps.append(dict(step))

    # Compatibility for legacy data or a staged migration: assign a Step only when
    # its indexed parent metadata identifies exactly one available Procedure.
    for step in unresolved_steps:
        matching = [
            key for key, procedure in procedure_by_key.items()
            if _v12_step_matches_procedure(step, procedure) is True
        ]
        if len(matching) == 1:
            parent = matching[0]
            cc = dict(step)
            cc["_v10_5_parent_source_key"] = parent
            cc.setdefault("structured_relation_source", "legacy_parent_text_seed")
            children_by_parent.setdefault(parent, []).append(cc)

    # Include Procedure-only families as low-priority fallbacks, but never let them
    # beat a related Step family solely on semantic similarity.
    for key in procedure_by_key:
        children_by_parent.setdefault(key, [])

    families: list[dict] = []
    for parent_key, seed_steps in children_by_parent.items():
        procedure = procedure_by_key.get(parent_key)
        if not isinstance(procedure, dict):
            continue
        procedure = dict(procedure)
        procedure["evidence_role"] = "procedure"
        procedure["ask_structured_direct"] = True
        complete_steps = _v12_expand_primary_procedure_steps(
            company_id=company_id,
            machine_id=machine_id,
            procedure=procedure,
            existing_steps=list(raw_steps),
        )
        complete_steps = _v12_dedupe_family_steps(list(complete_steps) + list(seed_steps))
        if not complete_steps and seed_steps:
            complete_steps = _v12_dedupe_family_steps(seed_steps)
        metrics = _v12_family_score(
            q=q,
            planner=planner,
            procedure=procedure,
            seed_steps=seed_steps,
            complete_steps=complete_steps,
            raw_procedure_present=parent_key in raw_procedure_keys,
        )
        families.append(
            {
                "parent_source_key": parent_key,
                "procedure": procedure,
                "seed_steps": list(seed_steps),
                "complete_steps": complete_steps,
                "metrics": metrics,
            }
        )

    if not families:
        return None, [], {
            "reason": "no_resolved_procedure_family",
            "raw_step_count": len(raw_steps),
            "relation_row_count": len(relation_rows),
        }

    families.sort(
        key=lambda row: (
            -float((row.get("metrics") or {}).get("score") or 0.0),
            -float((row.get("metrics") or {}).get("facet_coverage") or 0.0),
            -float((row.get("metrics") or {}).get("seed_facet_coverage") or 0.0),
            -int((row.get("metrics") or {}).get("seed_count") or 0),
            str(row.get("parent_source_key") or ""),
        )
    )
    winner = families[0]
    runner_up_score = float((families[1].get("metrics") or {}).get("score") or 0.0) if len(families) > 1 else None
    winner_score = float((winner.get("metrics") or {}).get("score") or 0.0)
    debug = {
        "reason": "family_voting",
        "selected_parent_source_key": str(winner.get("parent_source_key") or ""),
        "selected_score": round(winner_score, 6),
        "runner_up_score": round(runner_up_score, 6) if runner_up_score is not None else None,
        "relation_row_count": len(relation_rows),
        "families": [
            {
                "parent_source_key": str(row.get("parent_source_key") or ""),
                **{
                    key: value
                    for key, value in dict(row.get("metrics") or {}).items()
                    if key not in {"covered_facets", "missing_facets"}
                },
                "covered_facets": list((row.get("metrics") or {}).get("covered_facets") or []),
                "step_numbers": [
                    _v12_step_sort_key(step)[0]
                    for step in (row.get("complete_steps") or [])
                    if 0 < _v12_step_sort_key(step)[0] < 9999
                ],
            }
            for row in families[:6]
        ],
    }
    procedure = dict(winner.get("procedure") or {})
    procedure["_v10_5_family_debug"] = debug
    return procedure, list(winner.get("complete_steps") or []), debug


@dataclass(frozen=True)
class V12CurateStructuredSourcesRuntime:
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS: Any
    ASK_UI_STRUCTURED_MAX_CITATIONS: Any
    INFO_PROCEDURE_FULL: Any
    INFO_PROCEDURE_SEGMENT: Any
    _ask_structured_direct_intent: Callable[..., Any]
    _dedup_citations_preserve_order: Callable[..., Any]
    _v12_choose_primary_procedure_family: Callable[..., Any]
    _v12_evidence_role: Callable[..., Any]
    _v12_step_matches_procedure: Callable[..., Any]
    _v12_step_sort_key: Callable[..., Any]
    _v12_structured_rank: Callable[..., Any]


def v12_curate_structured_sources(
    *,
    company_id: str,
    machine_id: str,
    q: str,
    planner: Optional[dict],
    citations: list[dict],
    model_used: Optional[list[dict]] = None,
    runtime: V12CurateStructuredSourcesRuntime,
) -> list[dict]:
    """Keep one coherent procedure family and remove unrelated P&S/steps."""
    ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS = runtime.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS
    ASK_UI_STRUCTURED_MAX_CITATIONS = runtime.ASK_UI_STRUCTURED_MAX_CITATIONS
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    _ask_structured_direct_intent = runtime._ask_structured_direct_intent
    _dedup_citations_preserve_order = runtime._dedup_citations_preserve_order
    _v12_choose_primary_procedure_family = runtime._v12_choose_primary_procedure_family
    _v12_evidence_role = runtime._v12_evidence_role
    _v12_step_matches_procedure = runtime._v12_step_matches_procedure
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v12_structured_rank = runtime._v12_structured_rank
    citations = [dict(c) for c in citations or [] if isinstance(c, dict)]
    if not citations:
        return []
    intent = _ask_structured_direct_intent(q, planner=planner)
    planner_task = str((planner or {}).get("information_task") or "").strip().lower()
    operational = bool(
        planner_task in {INFO_PROCEDURE_FULL, INFO_PROCEDURE_SEGMENT}
        or (
            intent.get("enabled")
            and not intent.get("broad_overview")
            and any(p in {"procedure", "step"} for p in (intent.get("prefixes") or []))
        )
    )
    model_used = [dict(c) for c in model_used or [] if isinstance(c, dict)]
    if not operational:
        # Before synthesis model_used is empty and the full admitted pack is preserved.
        # After synthesis, non-procedural links must follow the sources the model
        # actually cited; free UI slots are not a reason to expose nearby media/P&S.
        if bool(intent.get("broad_overview")) or not model_used:
            return citations
        used_ids = {str(c.get("citation_id") or "").strip() for c in model_used}
        kept = [
            c for c in citations
            if str(c.get("citation_id") or "").strip() in used_ids
        ]
        return kept or citations[:1]

    primary, expanded_steps, family_debug = _v12_choose_primary_procedure_family(
        company_id=company_id,
        machine_id=machine_id,
        q=q,
        planner=planner,
        citations=citations,
        model_used=model_used,
    )
    if primary is None:
        used_ids = {str(c.get("citation_id") or "") for c in model_used}
        if used_ids:
            kept = [c for c in citations if str(c.get("citation_id") or "") in used_ids]
            return kept or citations
        # Do not manufacture INCOMPLETE_PROCEDURE_BUNDLE here. Returning the
        # admitted evidence lets the caller fall back to a grounded multi-source
        # procedural synthesis when no single family is objectively dominant.
        return citations

    primary = dict(primary)
    primary["evidence_role"] = "procedure"
    primary["ask_structured_direct"] = True
    primary["_v10_5_family_debug"] = dict(family_debug or {})

    existing_steps = [c for c in citations if _v12_evidence_role(c) == "step"]
    used_ids = {str(c.get("citation_id") or "") for c in model_used}
    for c in existing_steps:
        relation = _v12_step_matches_procedure(c, primary)
        if relation is None and str(c.get("citation_id") or "") in used_ids:
            cc = dict(c)
            cc["evidence_role"] = "step"
            expanded_steps.append(cc)

    best_steps: dict[str, dict] = {}
    for c in expanded_steps:
        bdid = str(c.get("bubble_document_id") or "").strip()
        if not bdid:
            continue
        cc = dict(c)
        cc["evidence_role"] = "step"
        cc["ask_structured_direct"] = True
        prev = best_steps.get(bdid)
        if prev is None or _v12_structured_rank(cc, used_ids) < _v12_structured_rank(prev, used_ids):
            best_steps[bdid] = cc
    steps = sorted(best_steps.values(), key=_v12_step_sort_key)

    # Keep non-procedure records only when the answer model explicitly used them.
    extras: list[dict] = []
    for c in model_used:
        role = _v12_evidence_role(c)
        if role in {"ps", "md_photo", "md_video"}:
            cc = dict(c)
            cc["evidence_role"] = role
            extras.append(cc)

    max_structured = max(12, int(ASK_UI_STRUCTURED_MAX_CITATIONS or 14) - max(1, int(ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS or 2)))
    return _dedup_citations_preserve_order([primary] + steps + extras, max_items=max_structured)


@dataclass(frozen=True)
class V12MarkManualSupportRuntime:
    pass


def v12_mark_manual_support(citations: list[dict], *, runtime: V12MarkManualSupportRuntime) -> list[dict]:
    out: list[dict] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        cc["ask_structured_manual_support"] = True
        cc["evidence_role"] = "manual_support"
        out.append(cc)
    return out


@dataclass(frozen=True)
class V12FilterLinkableManualSupportRuntime:
    _dedup_text_values: Callable[..., Any]
    _fetch_document_file_map: Callable[..., Any]


def v12_filter_linkable_manual_support(company_id: str, citations: list[dict], *, runtime: V12FilterLinkableManualSupportRuntime) -> list[dict]:
    """A manual claim may be shown only when Bubble can expose its source link."""
    _dedup_text_values = runtime._dedup_text_values
    _fetch_document_file_map = runtime._fetch_document_file_map
    citations = [dict(c) for c in (citations or []) if isinstance(c, dict)]
    doc_ids = _dedup_text_values(
        [str(c.get("bubble_document_id") or "").strip() for c in citations],
        limit=100,
    )
    if not doc_ids:
        return []
    try:
        file_map = _fetch_document_file_map(company_id, doc_ids)
    except Exception as exc:
        print("ASK_V12_MANUAL_LINK_PREFLIGHT_FAIL", str(exc)[:500])
        return []
    return [
        c for c in citations
        if str(file_map.get(str(c.get("bubble_document_id") or "").strip()) or "").strip()
    ]


@dataclass(frozen=True)
class V12FilterManualSupportToSelectedBundleRuntime:
    _content_term_set: Callable[..., Any]
    _procedure_ui_fields: Callable[..., Any]
    _procedure_ui_is_safety_setup: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v12_evidence_role: Callable[..., Any]


def v12_filter_manual_support_to_selected_bundle(
    *,
    q: str,
    structured_citations: list[dict],
    manual_support_citations: list[dict],
    runtime: V12FilterManualSupportToSelectedBundleRuntime,
) -> list[dict]:
    """Keep only manual pages that still match the final selected Step span.

    The strict LLM selector initially sees the complete Procedure family so it can
    help the answer model. After a partial Procedure has been narrowed, this cheap
    deterministic guard removes pages that supported an earlier setup Step but no
    longer support the final answer. It changes neither retrieval nor embeddings.
    """
    _content_term_set = runtime._content_term_set
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_is_safety_setup = runtime._procedure_ui_is_safety_setup
    _term_overlap_score = runtime._term_overlap_score
    _v12_evidence_role = runtime._v12_evidence_role
    manual_rows = [dict(c) for c in (manual_support_citations or []) if isinstance(c, dict)]
    if not manual_rows:
        return []

    operational_texts: list[str] = []
    all_selected_texts: list[str] = []
    for citation in structured_citations or []:
        if not isinstance(citation, dict):
            continue
        role = _v12_evidence_role(citation)
        if role not in {"procedure", "step"}:
            continue
        fields = _procedure_ui_fields(citation)
        text = " ".join(
            [
                str(fields.get("title") or ""),
                str(fields.get("short_description") or fields.get("description") or ""),
            ]
        ).strip()
        if not text:
            continue
        all_selected_texts.append(text)
        if role == "step" and not _procedure_ui_is_safety_setup(text):
            operational_texts.append(text)

    query_terms = _content_term_set(q, limit=80)
    operational_terms = _content_term_set(" ".join(operational_texts), limit=220)
    all_selected_terms = _content_term_set(" ".join(all_selected_texts), limit=260)

    kept: list[dict] = []
    for citation in manual_rows:
        manual_text = str(citation.get("chunk_full") or citation.get("snippet") or "")
        manual_terms = _content_term_set(manual_text, limit=260)
        if not manual_terms:
            continue
        kind = str(citation.get("ask_manual_support_kind") or "operation").strip().lower()
        selected_terms = all_selected_terms if kind == "safety" else operational_terms
        shared_query = query_terms & manual_terms
        shared_selected = selected_terms & manual_terms
        strong_shared = {term for term in shared_selected if len(term) >= 8}
        query_overlap = _term_overlap_score(query_terms, manual_terms) if query_terms else 0.0
        selected_overlap = _term_overlap_score(selected_terms, manual_terms) if selected_terms else 0.0

        keep = bool(
            query_overlap >= 0.045
            or selected_overlap >= 0.045
            or len(shared_query) >= 2
            or len(shared_selected) >= 2
            or strong_shared
        )
        if keep:
            citation["selected_bundle_query_overlap"] = float(query_overlap)
            citation["selected_bundle_step_overlap"] = float(selected_overlap)
            kept.append(citation)
    return kept


@dataclass(frozen=True)
class V12MarkStructuredRolesRuntime:
    _v12_evidence_role: Callable[..., Any]


def v12_mark_structured_roles(citations: list[dict], *, runtime: V12MarkStructuredRolesRuntime) -> list[dict]:
    _v12_evidence_role = runtime._v12_evidence_role
    out: list[dict] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        role = _v12_evidence_role(cc)
        cc["evidence_role"] = role
        cc["ask_structured_direct"] = True
        out.append(cc)
    return out


