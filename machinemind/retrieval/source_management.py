"""P4-C3: unchanged source identity, title matching, locking and source selection behind explicit runtime dependencies.

This is an extraction, not a new evidence policy. The composition root supplies
retrieval, metadata, policy and presentation callbacks at call time. This module
has no application, database, provider or web-framework import. Legacy decisions,
ordering, exception handling and budget checks are deliberately preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class CandidateOrderKeyRuntime:
    pass


def candidate_order_key(item: dict, *, runtime: CandidateOrderKeyRuntime) -> tuple:
    return (
        -float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0),
        -float(item.get("similarity", 0.0) or 0.0),
        -float(item.get("rrf_score", 0.0) or 0.0),
        str(item.get("bubble_document_id") or ""),
        int(item.get("page_from") or 0),
        int(item.get("page_to") or 0),
        int(item.get("chunk_index") or 0),
        str(item.get("citation_id") or ""),
    )


@dataclass(frozen=True)
class SourceTypeFromDocumentIdRuntime:
    STRUCTURED_SOURCE_TYPES: Any


def source_type_from_document_id(value: str, *, runtime: SourceTypeFromDocumentIdRuntime) -> str:
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    v = str(value or "").strip().lower()
    if not v or ":" not in v:
        return "manual"
    prefix = v.split(":", 1)[0].strip()
    if prefix in STRUCTURED_SOURCE_TYPES:
        return prefix
    return "manual"


@dataclass(frozen=True)
class StableEvidenceFamilyKeyRuntime:
    _extract_section_from_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    re: Any


def stable_evidence_family_key(item: dict, *, runtime: StableEvidenceFamilyKeyRuntime) -> str:
    _extract_section_from_text = runtime._extract_section_from_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _source_type_from_document_id = runtime._source_type_from_document_id
    re = runtime.re
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    doc_id = str(item.get("bubble_document_id") or "").strip()
    page_from = int(item.get("page_from") or 0)

    section = _extract_section_from_text(item.get("chunk_full") or item.get("snippet") or "")
    section = _normalize_unicode_advanced(section or "")
    section = re.sub(r"\s+", " ", section).strip().lower()[:120]

    if section:
        return f"{source_type}|{doc_id}|sec:{section}"

    if source_type == "manual":
        return f"{source_type}|{doc_id}|p{page_from}"

    return f"{source_type}|{doc_id}|p{page_from}"


@dataclass(frozen=True)
class StableEvidenceSetKeyRuntime:
    _extract_section_from_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    re: Any


def stable_evidence_set_key(item: dict, *, runtime: StableEvidenceSetKeyRuntime) -> str:
    _extract_section_from_text = runtime._extract_section_from_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _source_type_from_document_id = runtime._source_type_from_document_id
    re = runtime.re
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    doc_id = str(item.get("bubble_document_id") or "").strip()

    section = _extract_section_from_text(item.get("chunk_full") or item.get("snippet") or "")
    section = _normalize_unicode_advanced(section or "")
    section = re.sub(r"\s+", " ", section).strip().lower()[:80]

    if source_type in {"ps", "procedure", "step"}:
        return f"{source_type}|{doc_id}"

    if section:
        return f"{source_type}|{doc_id}|sec:{section}"

    return f"{source_type}|{doc_id}"


@dataclass(frozen=True)
class LockedFamilyScoreRuntime:
    pass


def locked_family_score(bundle: dict, *, runtime: LockedFamilyScoreRuntime) -> float:
    members = list(bundle.get("members") or [])
    if not members:
        return -1.0

    best = float(members[0].get("retrieval_score", members[0].get("similarity", 0.0)) or 0.0)
    second = float(members[1].get("retrieval_score", members[1].get("similarity", 0.0)) or 0.0) if len(members) >= 2 else 0.0
    max_overlap = max((float(m.get("overlap_score", 0.0)) for m in members), default=0.0)
    max_specificity = max((float(m.get("specificity_score", 0.0)) for m in members), default=0.0)
    selected_count = int(bundle.get("selected_count") or 0)
    source_type = str(bundle.get("source_type") or "manual")
    member_count = len(members)

    score = (
        0.55 * best
        + 0.10 * second
        + 0.14 * max_overlap
        + 0.11 * max_specificity
        + 0.04 * min(selected_count, 2)
        + 0.03 * min(member_count, 3)
    )

    if source_type == "manual" and (max_overlap >= 0.08 or max_specificity >= 0.04):
        score += 0.04

    if source_type in {"ps", "procedure", "step"} and max_overlap < 0.22 and max_specificity < 0.05:
        score -= 0.08

    return score


@dataclass(frozen=True)
class LockedMemberOrderKeyRuntime:
    pass


def locked_member_order_key(item: dict, selected_ids: set[str], *, runtime: LockedMemberOrderKeyRuntime) -> tuple:
    cid = str(item.get("citation_id") or "").strip()
    return (
        0 if cid in selected_ids else 1,
        -float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0),
        -float(item.get("overlap_score", 0.0) or 0.0),
        -float(item.get("specificity_score", 0.0) or 0.0),
        -float(item.get("similarity", 0.0) or 0.0),
        str(item.get("bubble_document_id") or ""),
        int(item.get("page_from") or 0),
        int(item.get("page_to") or 0),
        int(item.get("chunk_index") or 0),
        cid,
    )


@dataclass(frozen=True)
class LockedSetScoreRuntime:
    pass


def locked_set_score(set_row: dict, *, runtime: LockedSetScoreRuntime) -> float:
    families = list(set_row.get("families") or [])
    if not families:
        return -1.0

    top = float(families[0].get("family_score", 0.0))
    second = float(families[1].get("family_score", 0.0)) if len(families) >= 2 else 0.0
    manual_bonus = 0.05 if any(str(f.get("source_type") or "") == "manual" for f in families[:2]) else 0.0
    overlap_bonus = max((float(f.get("max_overlap", 0.0)) for f in families), default=0.0) * 0.10
    specificity_bonus = max((float(f.get("max_specificity", 0.0)) for f in families), default=0.0) * 0.08
    generic_penalty = 0.07 if all(bool(f.get("generic_structured")) for f in families[:2]) else 0.0

    return (
        0.74 * top
        + 0.18 * second
        + 0.03 * min(len(families), 3)
        + manual_bonus
        + overlap_bonus
        + specificity_bonus
        - generic_penalty
    )


@dataclass(frozen=True)
class FamilyRowSortKeyRuntime:
    pass


def family_row_sort_key(row: dict, *, runtime: FamilyRowSortKeyRuntime) -> tuple:
    return (
        -float(row.get("family_score", 0.0)),
        -float(row.get("best_score", 0.0)),
        0 if str(row.get("source_type") or "") == "manual" else 1,
        str(row.get("family_key") or ""),
    )


@dataclass(frozen=True)
class SetRowSortKeyRuntime:
    pass


def set_row_sort_key(row: dict, *, runtime: SetRowSortKeyRuntime) -> tuple:
    return (
        -float(row.get("set_score", 0.0)),
        0 if bool(row.get("has_manual")) else 1,
        -float(row.get("best_family_score", 0.0)),
        str(row.get("set_key") or ""),
    )


@dataclass(frozen=True)
class LockFinalCitationsRuntime:
    FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA: Any
    FINAL_CITATION_LOCK_SET_DELTA: Any
    FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA: Any
    _candidate_order_key: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _family_row_sort_key: Callable[..., Any]
    _locked_family_score: Callable[..., Any]
    _locked_member_order_key: Callable[..., Any]
    _locked_set_score: Callable[..., Any]
    _set_row_sort_key: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _stable_evidence_family_key: Callable[..., Any]
    _stable_evidence_set_key: Callable[..., Any]


def lock_final_citations(*, selected_citations: list[dict], ranked_candidates: list[dict], top_k: int, diagnostic_mode: bool=False, query_token_count: int=0, runtime: LockFinalCitationsRuntime) -> list[dict]:
    FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA = runtime.FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA
    FINAL_CITATION_LOCK_SET_DELTA = runtime.FINAL_CITATION_LOCK_SET_DELTA
    FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA = runtime.FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA
    _candidate_order_key = runtime._candidate_order_key
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _family_row_sort_key = runtime._family_row_sort_key
    _locked_family_score = runtime._locked_family_score
    _locked_member_order_key = runtime._locked_member_order_key
    _locked_set_score = runtime._locked_set_score
    _set_row_sort_key = runtime._set_row_sort_key
    _source_type_from_document_id = runtime._source_type_from_document_id
    _stable_evidence_family_key = runtime._stable_evidence_family_key
    _stable_evidence_set_key = runtime._stable_evidence_set_key
    selected_citations = list(selected_citations or [])
    ranked_candidates = list(ranked_candidates or [])

    if not selected_citations and not ranked_candidates:
        return []

    selected_ids = {
        str(c.get("citation_id") or "").strip()
        for c in selected_citations
        if c.get("citation_id")
    }

    ordered_ranked = sorted(ranked_candidates, key=_candidate_order_key)

    by_id: dict[str, dict] = {}
    for item in ordered_ranked + selected_citations:
        cid = str(item.get("citation_id") or "").strip()
        if not cid:
            continue
        prev = by_id.get(cid)
        if prev is None:
            by_id[cid] = dict(item)
            continue
        prev_score = float(prev.get("retrieval_score", prev.get("similarity", 0.0)) or 0.0)
        cur_score = float(item.get("retrieval_score", item.get("similarity", 0.0)) or 0.0)
        if cur_score > prev_score:
            merged = dict(prev)
            merged.update(item)
            by_id[cid] = merged

    families: dict[str, dict] = {}
    ordered_items = [by_id[str(c.get("citation_id") or "").strip()] for c in selected_citations if str(c.get("citation_id") or "").strip() in by_id]
    ordered_items += [by_id[cid] for cid in by_id if cid not in selected_ids]

    for item in ordered_items:
        cid = str(item.get("citation_id") or "").strip()
        if not cid:
            continue
        fam = _stable_evidence_family_key(item)
        bundle = families.setdefault(
            fam,
            {
                "family_key": fam,
                "set_key": _stable_evidence_set_key(item),
                "source_type": _source_type_from_document_id(item.get("bubble_document_id") or ""),
                "members": [],
                "selected_count": 0,
            },
        )
        bundle["members"].append(item)
        if cid in selected_ids:
            bundle["selected_count"] += 1

    if not families:
        return _dedup_citations_by_snippet(selected_citations, max_items=top_k)

    rows = []
    for fam, bundle in families.items():
        members = sorted(bundle["members"], key=lambda x: _locked_member_order_key(x, selected_ids))
        bundle["members"] = members
        row = dict(bundle)
        row["family_score"] = _locked_family_score(bundle)
        row["best_score"] = float(members[0].get("retrieval_score", members[0].get("similarity", 0.0)) or 0.0)
        row["max_overlap"] = max((float(m.get("overlap_score", 0.0)) for m in members), default=0.0)
        row["max_specificity"] = max((float(m.get("specificity_score", 0.0)) for m in members), default=0.0)
        row["generic_structured"] = (
            row["source_type"] in {"ps", "procedure", "step"}
            and row["max_overlap"] < 0.22
            and row["max_specificity"] < 0.05
        )
        rows.append(row)

    rows.sort(key=_family_row_sort_key)

    set_rows_map: dict[str, dict] = {}
    for row in rows:
        set_key = str(row.get("set_key") or "")
        srow = set_rows_map.setdefault(
            set_key,
            {
                "set_key": set_key,
                "families": [],
                "has_manual": False,
            },
        )
        srow["families"].append(row)
        if str(row.get("source_type") or "") == "manual":
            srow["has_manual"] = True

    set_rows = []
    for set_key, srow in set_rows_map.items():
        srow["families"] = sorted(srow["families"], key=_family_row_sort_key)
        srow["best_family_score"] = float(srow["families"][0].get("family_score", 0.0))
        srow["set_score"] = _locked_set_score(srow)
        set_rows.append(srow)

    set_rows.sort(key=_set_row_sort_key)
    if not set_rows:
        return _dedup_citations_by_snippet(selected_citations, max_items=top_k)

    keep_sets = [set_rows[0]]
    set_delta = FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA if diagnostic_mode else FINAL_CITATION_LOCK_SET_DELTA

    if len(set_rows) >= 2:
        top_set = set_rows[0]
        second_set = set_rows[1]
        gap = float(top_set.get("set_score", 0.0)) - float(second_set.get("set_score", 0.0))
        top_generic = all(bool(f.get("generic_structured")) for f in top_set.get("families")[:2])

        if top_generic and gap <= (set_delta + 0.012):
            keep_sets = [second_set]
            if diagnostic_mode and len(set_rows) >= 3:
                third_set = set_rows[2]
                if float(second_set.get("set_score", 0.0)) - float(third_set.get("set_score", 0.0)) <= set_delta:
                    keep_sets.append(third_set)
        elif diagnostic_mode and gap <= set_delta and not all(bool(f.get("generic_structured")) for f in second_set.get("families")[:2]):
            keep_sets.append(second_set)

    dominant_rows: list[dict] = []
    family_delta = FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA

    for set_row in keep_sets:
        fams = list(set_row.get("families") or [])
        if not fams:
            continue

        dominant_rows.append(fams[0])

        if diagnostic_mode:
            for fam in fams[1:]:
                gap = float(fams[0].get("family_score", 0.0)) - float(fam.get("family_score", 0.0))
                same_set_count = len([r for r in dominant_rows if r.get("set_key") == set_row.get("set_key")])
                if gap <= family_delta and same_set_count < 2:
                    dominant_rows.append(fam)

    dominant_rows = sorted(
        {str(r.get("family_key") or ""): r for r in dominant_rows if r.get("family_key")}.values(),
        key=_family_row_sort_key,
    )

    quotas = [2, 1, 1] if diagnostic_mode else [2]
    if dominant_rows and str(dominant_rows[0].get("source_type") or "") == "manual" and query_token_count >= 4:
        quotas[0] = min(3, top_k)

    locked: list[dict] = []
    used_ids: set[str] = set()

    for idx, row in enumerate(dominant_rows):
        quota = quotas[min(idx, len(quotas) - 1)]
        family_added = 0
        for member in row.get("members") or []:
            cid = str(member.get("citation_id") or "").strip()
            if not cid or cid in used_ids:
                continue
            locked.append(member)
            used_ids.add(cid)
            family_added += 1
            if family_added >= quota:
                break
        if len(locked) >= top_k:
            break

    if len(locked) < top_k:
        for row in dominant_rows:
            for member in row.get("members") or []:
                cid = str(member.get("citation_id") or "").strip()
                if not cid or cid in used_ids:
                    continue
                locked.append(member)
                used_ids.add(cid)
                if len(locked) >= top_k:
                    break
            if len(locked) >= top_k:
                break

    if len(locked) < top_k:
        top_set_key = str(keep_sets[0].get("set_key") or "") if keep_sets else ""
        for row in rows:
            if top_set_key and str(row.get("set_key") or "") != top_set_key:
                continue
            for member in row.get("members") or []:
                cid = str(member.get("citation_id") or "").strip()
                if not cid or cid in used_ids:
                    continue
                locked.append(member)
                used_ids.add(cid)
                if len(locked) >= top_k:
                    break
            if len(locked) >= top_k:
                break

    return _dedup_citations_by_snippet(locked, max_items=top_k)


@dataclass(frozen=True)
class CandidateSourceBiasRuntime:
    SEMANTIC_EXACT_MACHINE_BONUS: Any
    _content_term_set: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]


def candidate_source_bias(item: dict, query_terms: set[str], *, query_style: str='', query_token_count: int=0, runtime: CandidateSourceBiasRuntime) -> tuple[float, dict]:
    SEMANTIC_EXACT_MACHINE_BONUS = runtime.SEMANTIC_EXACT_MACHINE_BONUS
    _content_term_set = runtime._content_term_set
    _source_type_from_document_id = runtime._source_type_from_document_id
    _term_overlap_score = runtime._term_overlap_score
    text = (item.get("chunk_full") or item.get("snippet") or "").strip()
    text_terms = _content_term_set(text, limit=100)
    overlap = _term_overlap_score(query_terms, text_terms)
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")

    bias = 0.0

    # Exact machine knowledge must outrank company-general knowledge when semantic
    # relevance is close. This is a soft bonus, never a hard exclusion.
    exact_machine_scope = bool(item.get("exact_machine_scope"))
    if exact_machine_scope:
        bias += float(SEMANTIC_EXACT_MACHINE_BONUS or 0.0)

    if source_type == "manual":
        if overlap >= 0.10 and len(text_terms) >= 16:
            bias += 0.04
        if query_token_count >= 4 and overlap >= 0.08 and len(text_terms) >= 14:
            bias += 0.03

    elif source_type == "ps":
        if query_token_count >= 4:
            if len(text_terms) <= 26 and overlap < 0.24:
                bias -= 0.16
            elif len(text_terms) <= 36 and overlap < 0.20:
                bias -= 0.10
        else:
            if len(text_terms) <= 18 and overlap < 0.22:
                bias -= 0.14
            elif len(text_terms) <= 28 and overlap < 0.18:
                bias -= 0.08
        if overlap >= 0.28 and len(text_terms) >= 16:
            bias += 0.02

    elif source_type in {"procedure", "step"}:
        if query_token_count >= 4:
            if len(text_terms) <= 26 and overlap < 0.22:
                bias -= 0.11
            elif len(text_terms) <= 34 and overlap < 0.18:
                bias -= 0.07
        else:
            if len(text_terms) <= 18 and overlap < 0.18:
                bias -= 0.08
        if overlap >= 0.26:
            bias += 0.03

    if query_style == "natural" and source_type in {"ps", "procedure", "step"} and len(text_terms) <= 24:
        bias -= 0.04

    return max(-0.18, min(0.16, bias)), {
        "source_type": source_type,
        "overlap_score": overlap,
        "content_term_count": len(text_terms),
        "exact_machine_scope": exact_machine_scope,
        "scope_bonus": float(SEMANTIC_EXACT_MACHINE_BONUS or 0.0) if exact_machine_scope else 0.0,
    }


@dataclass(frozen=True)
class RebalanceSelectedCitationsRuntime:
    _candidate_order_key: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]


def rebalance_selected_citations(selected_citations: list[dict], ranked_candidates: list[dict], top_k: int, *, query_style: str='', query_token_count: int=0, runtime: RebalanceSelectedCitationsRuntime) -> list[dict]:
    _candidate_order_key = runtime._candidate_order_key
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _source_type_from_document_id = runtime._source_type_from_document_id
    if not selected_citations:
        return []

    ordered_ranked = sorted(ranked_candidates or [], key=_candidate_order_key)
    out = _dedup_citations_by_snippet(selected_citations, max_items=top_k)
    selected_ids = {str(c.get("citation_id") or "").strip() for c in out if c.get("citation_id")}

    def source_type(c: dict) -> str:
        return str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or ""))

    top_score = float(out[0].get("retrieval_score", out[0].get("similarity", 0.0)) or 0.0)
    manual_selected = any(source_type(c) == "manual" for c in out)

    if query_token_count >= 4 and not manual_selected:
        for cand in ordered_ranked:
            cid = str(cand.get("citation_id") or "").strip()
            if not cid or cid in selected_ids:
                continue
            if source_type(cand) != "manual":
                continue
            cand_score = float(cand.get("retrieval_score", cand.get("similarity", 0.0)) or 0.0)
            if cand_score >= top_score - 0.14:
                out = [cand] + out
                out = _dedup_citations_by_snippet(out, max_items=top_k)
                selected_ids = {str(c.get("citation_id") or "").strip() for c in out if c.get("citation_id")}
                break

    generic_structured = [
        c for c in out
        if source_type(c) in {"ps", "procedure", "step"}
        and float(c.get("overlap_score", 0.0)) < 0.20
        and float(c.get("specificity_score", 0.0)) < 0.04
    ]

    if query_token_count >= 4 and len(generic_structured) > 1:
        keep_first = True
        replacements: list[dict] = []
        for c in out:
            if c not in generic_structured:
                replacements.append(c)
                continue
            if keep_first:
                replacements.append(c)
                keep_first = False

        existing_ids = {str(c.get("citation_id") or "").strip() for c in replacements if c.get("citation_id")}
        for cand in ordered_ranked:
            cid = str(cand.get("citation_id") or "").strip()
            if not cid or cid in existing_ids:
                continue
            if source_type(cand) != "manual":
                continue
            cand_score = float(cand.get("retrieval_score", cand.get("similarity", 0.0)) or 0.0)
            if cand_score >= top_score - 0.16:
                replacements.append(cand)
                existing_ids.add(cid)
            if len(replacements) >= top_k:
                break

        out = _dedup_citations_by_snippet(replacements, max_items=top_k)

    if query_style == "natural" and query_token_count >= 4:
        reordered: list[dict] = []
        manuals = [c for c in out if source_type(c) == "manual"]
        others = [c for c in out if source_type(c) != "manual"]
        if manuals:
            reordered.extend(manuals[: max(1, min(len(manuals), top_k))])
            for c in others:
                if len(reordered) >= top_k:
                    break
                reordered.append(c)
            out = reordered[:top_k]

    return out[:top_k]


@dataclass(frozen=True)
class V12EvidenceRoleRuntime:
    _source_type_from_document_id: Callable[..., Any]


def v12_evidence_role(c: dict, *, runtime: V12EvidenceRoleRuntime) -> str:
    _source_type_from_document_id = runtime._source_type_from_document_id
    if not isinstance(c, dict):
        return ""
    if bool(c.get("ask_structured_manual_support")):
        return "manual_support"
    explicit = str(c.get("evidence_role") or "").strip().lower()
    if explicit:
        return explicit
    st = str(c.get("source_type") or "").strip().lower()
    if not st:
        st = _source_type_from_document_id(str(c.get("bubble_document_id") or ""))
    return st or "document"


@dataclass(frozen=True)
class DedupRootCauseCandidatesSemanticRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def dedup_root_cause_candidates_semantic(citations: list[dict], max_items: int, *, runtime: DedupRootCauseCandidatesSemanticRuntime) -> list[dict]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    def _sig(c: dict) -> str:
        txt = _normalize_unicode_advanced(c.get("snippet", "") or "")
        txt = re.sub(r"^SECTION:\s*[^\n]+\n?", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\s+", " ", txt).lower()
        txt = txt[:220]

        return (
            f"{str(c.get('bubble_document_id') or '').strip()}|"
            f"{int(c.get('page_from') or 0)}|"
            f"{int(c.get('page_to') or 0)}|"
            f"{txt}"
        )

    best = {}
    for c in citations or []:
        k = _sig(c)
        prev = best.get(k)

        if prev is None:
            best[k] = c
            continue

        prev_tuple = (
            float(prev.get("causal_strength_score", 0.0)),
            float(prev.get("semantic_score", 0.0)),
            float(prev.get("similarity", 0.0)),
        )
        cur_tuple = (
            float(c.get("causal_strength_score", 0.0)),
            float(c.get("semantic_score", 0.0)),
            float(c.get("similarity", 0.0)),
        )

        if cur_tuple > prev_tuple:
            best[k] = c

    out = list(best.values())
    out.sort(
        key=lambda x: (
            float(x.get("causal_strength_score", 0.0)),
            float(x.get("semantic_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )
    return out[:max_items]


@dataclass(frozen=True)
class RootCauseEvidenceFamilyKeyRuntime:
    _stable_evidence_family_key: Callable[..., Any]


def root_cause_evidence_family_key(c: dict, *, runtime: RootCauseEvidenceFamilyKeyRuntime) -> str:
    _stable_evidence_family_key = runtime._stable_evidence_family_key
    return _stable_evidence_family_key(c)


@dataclass(frozen=True)
class PrioritizeRootCauseCoverageRuntime:
    _root_cause_evidence_family_key: Callable[..., Any]


def prioritize_root_cause_coverage(citations: list[dict], max_items: int, *, runtime: PrioritizeRootCauseCoverageRuntime) -> list[dict]:
    _root_cause_evidence_family_key = runtime._root_cause_evidence_family_key
    if not citations:
        return []

    ordered = sorted(
        citations,
        key=lambda x: (
            float(x.get("causal_strength_score", 0.0)),
            float(x.get("semantic_score", 0.0)),
            float(x.get("similarity", 0.0)),
        ),
        reverse=True,
    )

    out: list[dict] = []
    used_ids = set()
    used_families = set()

    # primo pass: massimizza copertura famiglie diverse
    for c in ordered:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used_ids:
            continue

        fam = _root_cause_evidence_family_key(c)
        if fam in used_families:
            continue

        used_ids.add(cid)
        used_families.add(fam)
        out.append(c)

        if len(out) >= max_items:
            return out[:max_items]

    # secondo pass: riempi eventuali slot rimanenti
    for c in ordered:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used_ids:
            continue

        used_ids.add(cid)
        out.append(c)

        if len(out) >= max_items:
            break

    return out[:max_items]


@dataclass(frozen=True)
class V13CandidateTextRuntime:
    pass


def v13_candidate_text(c: dict, *, runtime: V13CandidateTextRuntime) -> str:
    return str(c.get("chunk_full") or c.get("snippet") or c.get("snippet_clean") or "").strip()


@dataclass(frozen=True)
class V13SourceTitleTokensCachedRuntime:
    _V13_SOURCE_TITLE_STOPWORDS: Any
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def v13_source_title_tokens_cached(value: str, limit: int, *, runtime: V13SourceTitleTokensCachedRuntime) -> tuple[str, ...]:
    _V13_SOURCE_TITLE_STOPWORDS = runtime._V13_SOURCE_TITLE_STOPWORDS
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    text = _normalize_unicode_advanced(str(value or "")).lower()
    # Human titles often use hyphens/slashes as word separators (for example
    # ``uscita-pezzo`` or ``aspo-macchina``), while technical identifiers use the
    # same characters inside mixed letter/digit codes (PROC-009, BBX-300/40T). Split
    # only separators surrounded by alphabetic characters, preserving code tokens.
    text = re.sub(r"(?<=[a-zà-öø-ÿ])[-_/](?=[a-zà-öø-ÿ])", " ", text)
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zà-öø-ÿ0-9][a-zà-öø-ÿ0-9_.\-/]{1,}", text):
        token = token.strip("._-/ ")
        if len(token) < 3 or token in _V13_SOURCE_TITLE_STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max(1, int(limit or 48)):
            break
    return tuple(out)


@dataclass(frozen=True)
class V13SourceTitleTokensRuntime:
    _v13_source_title_tokens_cached: Callable[..., Any]


def v13_source_title_tokens(value: Any, *, limit: int=48, runtime: V13SourceTitleTokensRuntime) -> list[str]:
    _v13_source_title_tokens_cached = runtime._v13_source_title_tokens_cached
    return list(_v13_source_title_tokens_cached(str(value or ""), max(1, int(limit or 48))))


@dataclass(frozen=True)
class V13SourceTokenEditDistanceRuntime:
    pass


def v13_source_token_edit_distance(left: str, right: str, *, max_distance: int=3, runtime: V13SourceTokenEditDistanceRuntime) -> int:
    """Small bounded Levenshtein distance used only for source-title tokens."""
    a = str(left or "")
    b = str(right or "")
    if a == b:
        return 0
    if not a:
        return min(len(b), max_distance + 1)
    if not b:
        return min(len(a), max_distance + 1)
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (0 if ca == cb else 1),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class V13SourceInflectionStemsRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def v13_source_inflection_stems(token: str, *, runtime: V13SourceInflectionStemsRuntime) -> set[str]:
    """Return conservative language-generic inflection stems, never semantic roots."""
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    value = _normalize_unicode_advanced(str(token or "")).lower().strip()
    out = {value} if value else set()
    if len(value) >= 5 and value[-1:] in {"a", "e", "i", "o"}:
        out.add(value[:-1])
    if len(value) >= 5 and value.endswith("s"):
        out.add(value[:-1])
    if len(value) >= 6 and value.endswith("es"):
        out.add(value[:-2])
    if len(value) >= 6 and value.endswith("ies"):
        out.add(value[:-3] + "y")
    return {x for x in out if len(x) >= 4}


@dataclass(frozen=True)
class V13SourceTitleTokenSimilarityRuntime:
    SequenceMatcher: Any
    _normalize_unicode_advanced: Callable[..., Any]
    _v13_source_inflection_stems: Callable[..., Any]
    _v13_source_token_edit_distance: Callable[..., Any]


def v13_source_title_token_similarity(left: str, right: str, *, runtime: V13SourceTitleTokenSimilarityRuntime) -> float:
    """Conservative fuzzy token match for titles.

    It accepts exact matches, ordinary singular/plural inflections and small typos.
    Shared prefixes alone never establish a match, avoiding pairs such as
    pressa/pressione, stampo/stampaggio or ciclo/cicloturismo.
    """
    SequenceMatcher = runtime.SequenceMatcher
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _v13_source_inflection_stems = runtime._v13_source_inflection_stems
    _v13_source_token_edit_distance = runtime._v13_source_token_edit_distance
    a = _normalize_unicode_advanced(str(left or "")).lower().strip()
    b = _normalize_unicode_advanced(str(right or "")).lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if any(ch.isdigit() for ch in a + b):
        return 0.0

    stems_a = _v13_source_inflection_stems(a)
    stems_b = _v13_source_inflection_stems(b)
    if stems_a & stems_b:
        return 0.96

    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    if min_len < 4:
        return 0.0
    allowed_edits = 1 if max_len <= 7 else 2
    distance = _v13_source_token_edit_distance(a, b, max_distance=allowed_edits)
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1
    suffix = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        suffix += 1
    boundary_coverage = min(min_len, prefix + suffix)
    if distance <= allowed_edits and prefix >= 2 and boundary_coverage >= min_len:
        normalized = 1.0 - (distance / max_len)
        return max(0.84, min(0.94, normalized))

    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.92 and prefix >= 3 and boundary_coverage >= min_len - 2:
        return float(ratio)
    return 0.0


@dataclass(frozen=True)
class V13SourceTitleMatchMetricsRuntime:
    SequenceMatcher: Any
    _v13_source_title_token_similarity: Callable[..., Any]
    _v13_source_title_tokens: Callable[..., Any]


def v13_source_title_match_metrics(q: str, title: str, description: str='', *, runtime: V13SourceTitleMatchMetricsRuntime) -> dict:
    SequenceMatcher = runtime.SequenceMatcher
    _v13_source_title_token_similarity = runtime._v13_source_title_token_similarity
    _v13_source_title_tokens = runtime._v13_source_title_tokens
    query_tokens = _v13_source_title_tokens(q, limit=64)
    title_tokens = _v13_source_title_tokens(title, limit=32)
    description_tokens = _v13_source_title_tokens(description, limit=96)
    if not query_tokens or not title_tokens:
        return {
            "score": 0.0, "title_coverage": 0.0, "query_coverage": 0.0,
            "strict_coverage": 0.0, "strict_query_coverage": 0.0,
            "matched_title_terms": 0, "matched_query_terms": 0,
            "title_term_count": len(title_tokens), "query_term_count": len(query_tokens),
            "description_support": 0.0,
        }

    title_best = [
        max((_v13_source_title_token_similarity(title_token, query_token) for query_token in query_tokens), default=0.0)
        for title_token in title_tokens
    ]
    query_best = [
        max((_v13_source_title_token_similarity(query_token, title_token) for title_token in title_tokens), default=0.0)
        for query_token in query_tokens
    ]
    matched_title = sum(1 for score in title_best if score >= 0.82)
    matched_query = sum(1 for score in query_best if score >= 0.82)
    title_coverage = sum(title_best) / max(1, len(title_best))
    strict_coverage = matched_title / max(1, len(title_tokens))

    # Ignore request framing by measuring the strongest query terms, bounded by the
    # title size. This remains content-based and is independent of source-type words.
    query_keep = max(1, min(len(title_tokens) + 1, len(query_best)))
    query_coverage = sum(sorted(query_best, reverse=True)[:query_keep]) / query_keep
    strict_query_coverage = min(1.0, matched_query / max(1, min(len(query_tokens), len(title_tokens))))

    title_norm = " ".join(title_tokens)
    query_norm = " ".join(query_tokens)
    sequence_score = SequenceMatcher(None, title_norm, query_norm).ratio() if title_norm and query_norm else 0.0

    if description_tokens:
        description_matches = [
            max((_v13_source_title_token_similarity(query_token, desc_token) for desc_token in description_tokens), default=0.0)
            for query_token in query_tokens
        ]
        keep = max(1, min(len(title_tokens) + 1, len(description_matches)))
        description_support = sum(sorted(description_matches, reverse=True)[:keep]) / keep
    else:
        description_support = 0.0

    score = (
        0.52 * title_coverage
        + 0.16 * strict_coverage
        + 0.14 * query_coverage
        + 0.06 * strict_query_coverage
        + 0.06 * sequence_score
        + 0.06 * description_support
    )

    # Two or more strongly matching title terms are meaningful evidence even when the
    # user omits one qualifier from a short title. This is generic partial-title recall,
    # not a domain-specific phrase rule.
    if matched_title >= 2:
        if len(title_tokens) <= 3 and strict_coverage >= (2.0 / 3.0):
            score = max(score, 0.76 + 0.14 * max(0.0, strict_coverage - (2.0 / 3.0)) / (1.0 / 3.0))
        elif len(title_tokens) <= 5 and strict_coverage >= 0.60:
            score = max(score, 0.72)
    if strict_coverage >= 0.999 and len(title_tokens) >= 2:
        score = max(score, 0.92)
    if len(title_tokens) == 1:
        score = max(score, 0.90) if matched_title == 1 else score * 0.45

    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "title_coverage": round(max(0.0, min(1.0, title_coverage)), 6),
        "query_coverage": round(max(0.0, min(1.0, query_coverage)), 6),
        "strict_coverage": round(max(0.0, min(1.0, strict_coverage)), 6),
        "strict_query_coverage": round(max(0.0, min(1.0, strict_query_coverage)), 6),
        "matched_title_terms": int(matched_title),
        "matched_query_terms": int(matched_query),
        "title_term_count": int(len(title_tokens)),
        "query_term_count": int(len(query_tokens)),
        "description_support": round(max(0.0, min(1.0, description_support)), 6),
    }


@dataclass(frozen=True)
class V13SourceSqlMatchPatternsRuntime:
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _v13_source_inflection_stems: Callable[..., Any]


def v13_source_sql_match_patterns(token: str, *, runtime: V13SourceSqlMatchPatternsRuntime) -> list[str]:
    """Bounded SQL patterns for exact and ordinary inflection variants."""
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _v13_source_inflection_stems = runtime._v13_source_inflection_stems
    value = _normalize_unicode_advanced(str(token or "")).lower().strip()
    if not value:
        return []
    variants = [value]
    for stem in sorted(_v13_source_inflection_stems(value), key=lambda x: (-len(x), x)):
        if stem != value and len(stem) >= 4:
            variants.append(stem)
    return _dedup_text_values(variants, limit=3)


@dataclass(frozen=True)
class V13PromoteExistingSourceCandidatesRuntime:
    STRUCTURED_SOURCE_TYPES: Any
    V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE: Any
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES: Any
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE: Any
    _dedup_citations_by_snippet: Callable[..., Any]
    _fetch_document_file_map: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _v13_real_semantic_similarity: Callable[..., Any]
    _v13_source_candidate_title: Callable[..., Any]
    _v13_source_title_match_metrics: Callable[..., Any]


def v13_promote_existing_source_candidates(q: str, retrieval: dict, *, company_id: str, runtime: V13PromoteExistingSourceCandidatesRuntime) -> list[dict]:
    """Expose strong title or dense-semantic candidates already present in retrieval.

    This closes the lexical-prefilter gap without another embedding or reasoning call.
    It annotates copies only; the baseline retrieval order and evidence pack remain
    unchanged unless the semantic gate later selects the item for a direct source task.
    """
    STRUCTURED_SOURCE_TYPES = runtime.STRUCTURED_SOURCE_TYPES
    V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE = runtime.V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES = runtime.V13_SOURCE_RETRIEVAL_MAX_CANDIDATES
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE = runtime.V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _fetch_document_file_map = runtime._fetch_document_file_map
    _source_type_from_document_id = runtime._source_type_from_document_id
    _v13_real_semantic_similarity = runtime._v13_real_semantic_similarity
    _v13_source_candidate_title = runtime._v13_source_candidate_title
    _v13_source_title_match_metrics = runtime._v13_source_title_match_metrics
    raw = [
        dict(c) for c in ((retrieval or {}).get("candidates") or [])[:32]
        if isinstance(c, dict)
    ]
    if not raw:
        return []

    document_ids = [
        str(c.get("bubble_document_id") or "").strip()
        for c in raw
        if str(
            c.get("source_type")
            or _source_type_from_document_id(c.get("bubble_document_id") or "")
        ).strip().lower() == "document"
        and str(c.get("bubble_document_id") or "").strip()
        and _v13_real_semantic_similarity(c) >= 0.45
    ]
    file_map: dict[str, str] = {}
    if document_ids:
        try:
            file_map = _fetch_document_file_map(company_id, document_ids)
        except Exception as exc:
            print("V13_SOURCE_EXISTING_FILE_TITLE_FAIL", str(exc)[:400])
            file_map = {}

    out: list[dict] = []
    allowed_types = set(STRUCTURED_SOURCE_TYPES) | {"document"}
    for c in raw:
        source_type = str(
            c.get("source_type")
            or _source_type_from_document_id(c.get("bubble_document_id") or "")
        ).strip().lower()
        if source_type not in allowed_types:
            continue
        bdid = str(c.get("bubble_document_id") or "").strip()
        title, description = _v13_source_candidate_title(
            c,
            file_url=str(file_map.get(bdid) or ""),
        )
        if not title:
            continue
        metrics = _v13_source_title_match_metrics(q, title, description)
        title_score = float(metrics.get("score") or 0.0)
        semantic_score = _v13_real_semantic_similarity(c)
        if (
            title_score < V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE
            and semantic_score < V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE
        ):
            continue
        cc = dict(c)
        cc["source_type"] = source_type
        cc["structured_title_match"] = bool(title_score >= V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE)
        cc["structured_title_match_score"] = title_score
        cc["structured_title_coverage"] = float(metrics.get("title_coverage") or 0.0)
        cc["structured_title_query_coverage"] = float(metrics.get("query_coverage") or 0.0)
        cc["structured_title_strict_coverage"] = float(metrics.get("strict_coverage") or 0.0)
        cc["structured_title_strict_query_coverage"] = float(metrics.get("strict_query_coverage") or 0.0)
        cc["structured_title_matched_terms"] = int(metrics.get("matched_title_terms") or 0)
        cc["structured_title_term_count"] = int(metrics.get("title_term_count") or 0)
        cc["structured_title"] = title
        cc["structured_description"] = description
        cc["source_retrieval_existing_candidate"] = True
        cc["source_retrieval_probe_score"] = max(title_score, 0.92 * semantic_score)
        out.append(cc)

    out.sort(
        key=lambda c: (
            -float(c.get("source_retrieval_probe_score") or 0.0),
            -float(c.get("structured_title_match_score") or 0.0),
            -_v13_real_semantic_similarity(c),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
        )
    )
    return _dedup_citations_by_snippet(out, max_items=V13_SOURCE_RETRIEVAL_MAX_CANDIDATES)


@dataclass(frozen=True)
class V13ShouldForceSourceTaskGateRuntime:
    V13_SOURCE_RETRIEVAL_ENABLED: Any
    V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE: Any
    V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE: Any
    V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS: Any
    _count_query_tokens: Callable[..., Any]
    _v13_real_semantic_similarity: Callable[..., Any]


def v13_should_force_source_task_gate(q: str, title_candidates: list[dict], *, runtime: V13ShouldForceSourceTaskGateRuntime) -> bool:
    V13_SOURCE_RETRIEVAL_ENABLED = runtime.V13_SOURCE_RETRIEVAL_ENABLED
    V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE = runtime.V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE
    V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE = runtime.V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE
    V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS = runtime.V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS
    _count_query_tokens = runtime._count_query_tokens
    _v13_real_semantic_similarity = runtime._v13_real_semantic_similarity
    if not V13_SOURCE_RETRIEVAL_ENABLED or not title_candidates:
        return False
    if _count_query_tokens(q) > V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS:
        return False
    top = title_candidates[0]
    title_score = float(top.get("structured_title_match_score") or 0.0)
    title_coverage = float(top.get("structured_title_coverage") or 0.0)
    query_coverage = float(top.get("structured_title_query_coverage") or 0.0)
    matched_terms = int(top.get("structured_title_matched_terms") or 0)
    semantic_score = _v13_real_semantic_similarity(top)
    strong_title = bool(
        title_score >= V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE
        and matched_terms >= 2
        and (title_coverage >= 0.60 or query_coverage >= 0.60)
    )
    strong_semantic = bool(
        semantic_score >= V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE
        and str(top.get("structured_title") or top.get("display_title") or "").strip()
    )
    return bool(strong_title or strong_semantic)


@dataclass(frozen=True)
class V13SourceRetrievalResultLimitRuntime:
    V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW: Any
    V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY: Any


def v13_source_retrieval_result_limit(cardinality: str, top_k: int, *, runtime: V13SourceRetrievalResultLimitRuntime) -> int:
    V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW = runtime.V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW
    V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY = runtime.V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY
    value = str(cardinality or "few").strip().lower()
    if value == "one":
        return 1
    if value == "many":
        return max(1, min(int(top_k or 5), V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY))
    return max(1, min(int(top_k or 5), V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW))


@dataclass(frozen=True)
class V13SourceCandidateTitleRuntime:
    _clean_display_text: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    _title_from_file_url: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def v13_source_candidate_title(candidate: dict, *, file_url: str='', runtime: V13SourceCandidateTitleRuntime) -> tuple[str, str]:
    _clean_display_text = runtime._clean_display_text
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    _title_from_file_url = runtime._title_from_file_url
    _v13_candidate_text = runtime._v13_candidate_text
    c = candidate if isinstance(candidate, dict) else {}
    body = _v13_candidate_text(c)
    fields = _parse_structured_source_fields(body)
    title = _clean_display_text(
        c.get("structured_title")
        or c.get("display_title")
        or fields.get("title")
        or fields.get("short_description")
        or "",
        max_len=180,
    )
    description = _clean_display_text(
        c.get("structured_description")
        or fields.get("description")
        or fields.get("solution")
        or "",
        max_len=700,
    )
    if not title:
        title = _clean_display_text(_title_from_file_url(file_url or c.get("file_url") or ""), max_len=180)
    return title, description


@dataclass(frozen=True)
class V13SourceRetrievalCandidateMetricsRuntime:
    _v13_real_semantic_similarity: Callable[..., Any]
    _v13_source_candidate_title: Callable[..., Any]
    _v13_source_title_match_metrics: Callable[..., Any]


def v13_source_retrieval_candidate_metrics(q: str, candidate: dict, *, task_focus: str='', file_url: str='', runtime: V13SourceRetrievalCandidateMetricsRuntime) -> dict:
    _v13_real_semantic_similarity = runtime._v13_real_semantic_similarity
    _v13_source_candidate_title = runtime._v13_source_candidate_title
    _v13_source_title_match_metrics = runtime._v13_source_title_match_metrics
    c = candidate if isinstance(candidate, dict) else {}
    title, description = _v13_source_candidate_title(c, file_url=file_url)
    query_metrics = _v13_source_title_match_metrics(q, title, description) if title else {
        "score": 0.0, "title_coverage": 0.0, "strict_coverage": 0.0,
        "matched_title_terms": 0, "title_term_count": 0, "description_support": 0.0,
    }
    focus_metrics = _v13_source_title_match_metrics(task_focus, title, description) if task_focus and title else {
        "score": 0.0, "title_coverage": 0.0, "strict_coverage": 0.0,
        "matched_title_terms": 0, "title_term_count": 0, "description_support": 0.0,
    }
    stored_score = float(c.get("structured_title_match_score") or 0.0)
    title_score = max(stored_score, float(query_metrics.get("score") or 0.0))
    focus_score = float(focus_metrics.get("score") or 0.0)
    semantic_score = _v13_real_semantic_similarity(c)
    # Exact/fuzzy title evidence is strongest. Semantic similarity is retained as a
    # conservative supplement, not as a way to turn a generic source into a title hit.
    effective_score = max(title_score, focus_score, 0.92 * semantic_score)
    return {
        "title": title,
        "description": description,
        "title_score": max(0.0, min(1.0, title_score)),
        "focus_score": max(0.0, min(1.0, focus_score)),
        "semantic_score": max(0.0, min(1.0, semantic_score)),
        "effective_score": max(0.0, min(1.0, effective_score)),
        "title_coverage": max(
            float(c.get("structured_title_coverage") or 0.0),
            float(query_metrics.get("title_coverage") or 0.0),
        ),
        "strict_coverage": max(
            float(c.get("structured_title_strict_coverage") or 0.0),
            float(query_metrics.get("strict_coverage") or 0.0),
        ),
        "matched_title_terms": max(
            int(c.get("structured_title_matched_terms") or 0),
            int(query_metrics.get("matched_title_terms") or 0),
        ),
    }


@dataclass(frozen=True)
class V13SourcesBlockRuntime:
    _source_type_from_document_id: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def v13_sources_block(citations: list[dict], *, max_context_chars: int, runtime: V13SourcesBlockRuntime) -> str:
    _source_type_from_document_id = runtime._source_type_from_document_id
    _v13_candidate_text = runtime._v13_candidate_text
    parts: list[str] = []
    total = 0
    for c in citations or []:
        cid = str(c.get("citation_id") or "").strip()
        body = _v13_candidate_text(c)
        if not cid or not body:
            continue
        source_type = str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or ""))
        scope_role = "exact_machine" if bool(c.get("exact_machine_scope")) else "company_or_general"
        role = str(c.get("role_class") or c.get("evidence_role") or source_type)
        score = round(float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0), 4)
        title_match = float(c.get("structured_title_match_score") or 0.0)
        title_meta = f"; title_match={title_match:.4f}" if title_match > 0.0 else ""
        part = (
            f"[{cid}] source_type={source_type}; scope={scope_role}; role={role}; score={score}{title_meta}; "
            f"doc={c.get('bubble_document_id')}; p={c.get('page_from')}-{c.get('page_to')}\n"
            f"{body}\n"
        )
        if total + len(part) > max_context_chars:
            if not parts:
                parts.append(part[:max_context_chars])
            break
        parts.append(part)
        total += len(part)
    return "\n".join(parts).strip()


@dataclass(frozen=True)
class V13ExactIdentifierCandidatesRuntime:
    _db_find_token_chunk: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _extract_code_tokens: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]


def v13_exact_identifier_candidates(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], runtime: V13ExactIdentifierCandidatesRuntime) -> list[dict]:
    """Retrieve exact identifier evidence without answering before the shared gate."""
    _db_find_token_chunk = runtime._db_find_token_chunk
    _dedup_text_values = runtime._dedup_text_values
    _extract_code_tokens = runtime._extract_code_tokens
    _source_type_from_document_id = runtime._source_type_from_document_id
    out: list[dict] = []
    seen: set[str] = set()
    for token in _dedup_text_values(_extract_code_tokens(q), limit=8):
        hit = _db_find_token_chunk(
            company_id=company_id,
            machine_id=machine_id,
            token=token,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if not hit:
            continue
        item = dict(hit)
        cid = str(item.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        item["chunk_full"] = str(item.get("snippet") or "")
        item["snippet_clean"] = str(item.get("snippet") or "")
        item["source_type"] = _source_type_from_document_id(item.get("bubble_document_id") or "")
        item["exact_code_hit"] = True
        item["exact_identifier_token"] = str(token)
        item["identifier_retrieval_v13"] = True
        # Do not fake semantic similarity. Exact identity is recorded separately and
        # contextual requests are validated by the semantic sufficiency gate.
        item["similarity"] = float(item.get("similarity") or 0.0)
        item["retrieval_score"] = max(float(item.get("retrieval_score") or 0.0), 0.25)
        out.append(item)
    return out


@dataclass(frozen=True)
class AssistantCoreCandidateSourceTypeRuntime:
    _source_type_from_document_id: Callable[..., Any]


def assistant_core_candidate_source_type(candidate: dict, *, runtime: AssistantCoreCandidateSourceTypeRuntime) -> str:
    _source_type_from_document_id = runtime._source_type_from_document_id
    raw = str(
        (candidate or {}).get("source_type")
        or _source_type_from_document_id((candidate or {}).get("bubble_document_id") or "")
        or "document"
    ).strip().lower()
    return "document" if raw == "manual" else raw


@dataclass(frozen=True)
class AssistantCoreCandidateStableKeyRuntime:
    _safe_int: Callable[..., Any]


def assistant_core_candidate_stable_key(candidate: dict, *, runtime: AssistantCoreCandidateStableKeyRuntime) -> str:
    _safe_int = runtime._safe_int
    cid = str(candidate.get("citation_id") or "").strip()
    if cid:
        return cid
    bdid = str(candidate.get("bubble_document_id") or "").strip()
    p1 = _safe_int(candidate.get("page_from"), 0)
    p2 = _safe_int(candidate.get("page_to"), p1)
    return f"{bdid}:p{p1}-{p2}:{str(candidate.get('source_type') or '')}"


@dataclass(frozen=True)
class AssistantCoreSourceDiversityPoolRuntime:
    _assistant_core_candidate_source_type: Callable[..., Any]


def assistant_core_source_diversity_pool(candidates: list[dict], *, per_type: int=2, runtime: AssistantCoreSourceDiversityPoolRuntime) -> list[dict]:
    """Preserve a few strong sources from every available family for overviews."""
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    grouped: dict[str, list[dict]] = {}
    for c in candidates or []:
        st = _assistant_core_candidate_source_type(c) or "unknown"
        grouped.setdefault(st, []).append(c)
    out: list[dict] = []
    for st in sorted(grouped):
        rows = sorted(grouped[st], key=lambda c: -float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0))
        out.extend(rows[:per_type])
    return out


@dataclass(frozen=True)
class AssistantCoreMachineCatalogDigestRuntime:
    _assistant_core_candidate_source_type: Callable[..., Any]
    _clean_display_text: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def assistant_core_machine_catalog_digest(candidates: list[dict], *, max_chars: int=18000, runtime: AssistantCoreMachineCatalogDigestRuntime) -> str:
    """Compact inventory of all authorised machine-level structured/media records."""
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _clean_display_text = runtime._clean_display_text
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    _v13_candidate_text = runtime._v13_candidate_text
    rows: list[str] = []
    used = 0
    for candidate in candidates or []:
        if not isinstance(candidate, dict) or not bool(candidate.get("assistant_core_catalog_candidate")):
            continue
        cid = str(candidate.get("citation_id") or "").strip()
        source_type = _assistant_core_candidate_source_type(candidate) or "source"
        fields = _parse_structured_source_fields(_v13_candidate_text(candidate))
        title = _clean_display_text(
            fields.get("title") or candidate.get("display_title") or candidate.get("bubble_document_id") or source_type,
            max_len=180,
        )
        description = _clean_display_text(
            fields.get("short_description")
            or fields.get("description")
            or fields.get("notes")
            or _v13_candidate_text(candidate),
            max_len=700,
        )
        row = f"- [{cid}] TYPE={source_type}; TITLE={title}; DESCRIPTION={description}"
        if used + len(row) > max_chars:
            break
        rows.append(row)
        used += len(row) + 1
    return "\n".join(rows)


@dataclass(frozen=True)
class AssistantCoreOverviewCatalogCandidatesRuntime:
    _assistant_core_candidate_source_type: Callable[..., Any]


def assistant_core_overview_catalog_candidates(candidates: list[dict], *, runtime: AssistantCoreOverviewCatalogCandidatesRuntime) -> list[dict]:
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    catalog = [
        dict(c) for c in candidates or []
        if isinstance(c, dict) and bool(c.get("assistant_core_catalog_candidate"))
    ]
    # Media describes physical layout; procedures expose auxiliary systems and
    # assemblies that may be absent from a single overview page. Preserve both.
    catalog.sort(
        key=lambda c: (
            0 if _assistant_core_candidate_source_type(c) in {"md_photo", "md_video", "photo", "video"} else 1,
            str(c.get("bubble_document_id") or ""),
        )
    )
    return catalog


