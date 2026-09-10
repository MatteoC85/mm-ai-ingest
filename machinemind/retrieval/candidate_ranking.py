"""P4-B: existing retrieval behavior behind explicit dependencies.

Pure extraction from the P4-A baseline. Existing source selection heuristics,
SQL, ranking weights, tie-breaking and failure behavior are preserved, including
legacy limitations. No application, database or provider client is imported.
Dependencies are resolved by the composition root at call time.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
import re
import math
import json
from difflib import SequenceMatcher

@dataclass(frozen=True)
class LlmRerankCitationsRuntime:
    ASK_MAX_TOP_K: int
    OPENAI_RERANK_MODEL: str
    RERANK_MAX_CANDIDATES: int
    RERANK_SNIPPET_CHARS: int
    RERANK_TIMEOUT: float
    _extract_section_from_text: Callable[..., Any]
    _openai_chat_json: Callable[..., Any]


def llm_rerank_citations(q: str, candidates: list[dict], top_k: int, diagnostic_mode: bool=False, *, runtime: LlmRerankCitationsRuntime, lineage: Optional[Callable[..., None]]=None) -> list[str]:
    ASK_MAX_TOP_K = runtime.ASK_MAX_TOP_K
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    RERANK_MAX_CANDIDATES = runtime.RERANK_MAX_CANDIDATES
    RERANK_SNIPPET_CHARS = runtime.RERANK_SNIPPET_CHARS
    RERANK_TIMEOUT = runtime.RERANK_TIMEOUT
    _extract_section_from_text = runtime._extract_section_from_text
    _openai_chat_json = runtime._openai_chat_json
    q = (q or "").strip()
    if not q or not candidates:
        if lineage is not None:
            lineage(())
        return []

    requested_k = max(1, min(int(top_k or 1), ASK_MAX_TOP_K))
    max_candidates = max(1, min(int(RERANK_MAX_CANDIDATES), len(candidates)))

    items = []
    seen = set()

    if lineage is not None:
        _positions = {}
        _input_index = -1
    for c in candidates[:max_candidates]:
        if lineage is not None:
            _input_index += 1
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if lineage is not None:
            _positions[cid] = _input_index

        full_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        section = _extract_section_from_text(full_text)

        snippet = (c.get("snippet") or "").strip()
        snippet = re.sub(r"^SECTION:\s*[^\n]+\n?", "", snippet).strip()

        items.append(
            {
                "citation_id": cid,
                "section": section,
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "similarity": round(float(c.get("similarity", 0.0)), 4),
                "snippet": snippet[:RERANK_SNIPPET_CHARS],
            }
        )

    if not items:
        if lineage is not None:
            lineage(())
        return []

    schema = {
        "name": "citation_rerank",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["selected_ids"],
        },
    }

    if diagnostic_mode:
        system_msg = (
            "Selezioni le citazioni più utili per diagnosticare un problema tecnico su una macchina industriale. "
            "Regole obbligatorie: "
            "1) tieni solo fonti che parlano del fenomeno o dei componenti coinvolti; "
            "2) scarta fonti generiche di manutenzione, sicurezza, installazione o lubrificazione se non sono direttamente legate al sintomo; "
            "3) preferisci fonti che descrivono componenti, regolazioni, giochi meccanici, allineamenti o anomalie operative; "
            "4) overview, caratteristiche generali, safety, acoustic, installation, start-up e sezioni simili sono bassa priorità, salvo che il sintomo riguardi esplicitamente quei temi; "
            "5) se una fonte è soprattutto boilerplate e solo marginalmente correlata, scartala; "
            "6) se due fonti sono simili, tieni la più specifica; "
            "7) non collassare tutto su una sola fonte se 2-3 fonti specifiche coprono aree causali diverse; "
            "8) restituisci il minor numero possibile di citation_id davvero utili."
        )
        
    else:
        system_msg = (
            "Selezioni le citazioni minime e più precise per rispondere a una domanda tecnica industriale. "
            "Obiettivo: tenere solo le fonti strettamente necessarie e scartare quelle solo vagamente correlate. "
            "Regole obbligatorie: "
            "1) seleziona il minor numero possibile di citation_id utili; "
            "2) preferisci chunk che contengono direttamente la risposta; "
            "3) scarta chunk generici di manutenzione o contesto se non aggiungono informazione utile; "
            "4) se due chunk sono simili, tieni solo il più specifico."
        )

    user_msg = (
        f"DOMANDA:\n{q}\n\n"
        f"TOP_K_DESIDERATO: {requested_k}\n\n"
        "CANDIDATI_JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Restituisci JSON valido con questa forma:\n"
        '{"selected_ids":["id1","id2"]}\n'
        "Ordina selected_ids dal migliore al meno rilevante. "
        "Non includere più di TOP_K_DESIDERATO elementi."
    )

    parsed = _openai_chat_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=OPENAI_RERANK_MODEL,
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    selected = parsed.get("selected_ids") or []
    if not isinstance(selected, list):
        if lineage is not None:
            lineage(())
        return []

    allowed = {item["citation_id"] for item in items}
    out = []
    used = set()

    for cid in selected:
        cid = str(cid or "").strip()
        if not cid or cid not in allowed or cid in used:
            continue
        used.add(cid)
        out.append(cid)
        if len(out) >= requested_k:
            break

    if lineage is not None:
        lineage(tuple(((0, _positions[cid]),) for cid in out))
    return out



@dataclass(frozen=True)
class ShouldUseRerankerRuntime:
    RERANK_ENABLED: bool
    RERANK_MAX_SIM_MAX: float
    RERANK_MAX_SPREAD: float
    RERANK_MIN_CANDIDATES: int
    RERANK_MIN_SIM_MAX: float


def should_use_reranker(q: str, candidates: list[dict], sim_max: float, top_k: int, *, runtime: ShouldUseRerankerRuntime) -> bool:
    RERANK_ENABLED = runtime.RERANK_ENABLED
    RERANK_MAX_SIM_MAX = runtime.RERANK_MAX_SIM_MAX
    RERANK_MAX_SPREAD = runtime.RERANK_MAX_SPREAD
    RERANK_MIN_CANDIDATES = runtime.RERANK_MIN_CANDIDATES
    RERANK_MIN_SIM_MAX = runtime.RERANK_MIN_SIM_MAX
    if not RERANK_ENABLED:
        return False
    if not q or not candidates:
        return False
    if len(candidates) < max(2, RERANK_MIN_CANDIDATES):
        return False
    if sim_max is None:
        return False
    if sim_max < RERANK_MIN_SIM_MAX:
        return False
    if sim_max > RERANK_MAX_SIM_MAX:
        return False

    ordered = sorted(
        candidates,
        key=lambda x: float(x.get("similarity", 0.0)),
        reverse=True,
    )
    if len(ordered) < 2:
        return False

    spread = float(ordered[0].get("similarity", 0.0)) - float(
        ordered[min(len(ordered) - 1, top_k - 1)].get("similarity", 0.0)
    )
    if spread > RERANK_MAX_SPREAD:
        return False

    return True


def rrf_merge_candidates(ranked_lists: list[list[dict]], k: int=60, *, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    scores: dict[str, float] = {}
    best_item: dict[str, dict] = {}

    origins = {} if lineage is not None else None
    chosen = {} if lineage is not None else None
    emitted = {} if lineage is not None else None
    for group_index, ranked in enumerate(ranked_lists):
        for idx, item in enumerate(ranked):
            cid = str(item.get("citation_id") or "").strip()
            if not cid:
                continue

            if lineage is not None:
                origins.setdefault(cid, []).append((group_index, idx))
            score = 1.0 / float(k + idx + 1)
            scores[cid] = scores.get(cid, 0.0) + score

            prev = best_item.get(cid)
            if prev is None or float(item.get("similarity", 0.0)) > float(prev.get("similarity", 0.0)):
                best_item[cid] = item
                if lineage is not None:
                    chosen[cid] = (group_index, idx)

    out = []
    for cid, item in best_item.items():
        merged = dict(item)
        merged["rrf_score"] = scores.get(cid, 0.0)
        out.append(merged)
        if lineage is not None:
            # Direct birth-site association, not a later value/ID lookup.
            emitted[id(merged)] = (chosen[cid],) + tuple(
                pos for pos in origins[cid] if pos != chosen[cid])

    out.sort(
        key=lambda x: (
            -float(x.get("rrf_score", 0.0)),
            -float(x.get("similarity", 0.0)),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        ),
    )
    if lineage is not None:
        lineage(tuple(emitted[id(item)] for item in out))
    return out


@dataclass(frozen=True)
class PromoteStructuredRescueHitsRuntime:
    STRUCTURED_RESCUE_MAX_HITS: int
    _dedup_citations_by_snippet: Callable[..., Any]


def promote_structured_rescue_hits(selected_citations: list[dict], structured_hits: list[dict], top_k: int, *, runtime: PromoteStructuredRescueHitsRuntime, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    STRUCTURED_RESCUE_MAX_HITS = runtime.STRUCTURED_RESCUE_MAX_HITS
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    if not structured_hits:
        if lineage is not None:
            lineage(tuple(((0, _index),) for _index in range(len(selected_citations or []))))
        return selected_citations or []

    out: list[dict] = []
    used: set[str] = set()

    if lineage is not None:
        _parents = []
        _input_index = -1
    for h in structured_hits:
        if lineage is not None:
            _input_index += 1
        cid = str(h.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        out.append(h)
        if lineage is not None:
            _parents.append(((1, _input_index),))
        used.add(cid)
        if len(out) >= min(STRUCTURED_RESCUE_MAX_HITS, top_k):
            break

    if lineage is not None:
        _input_index = -1
    for c in selected_citations or []:
        if lineage is not None:
            _input_index += 1
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        out.append(c)
        if lineage is not None:
            _parents.append(((0, _input_index),))
        used.add(cid)
        if len(out) >= top_k:
            break

    if lineage is not None:
        _before_selection = tuple(out)
        _result = _dedup_citations_by_snippet(out, max_items=top_k)
        lineage(_selected_occurrence_lineage(_before_selection, tuple(_parents), _result))
        return _result
    return _dedup_citations_by_snippet(out, max_items=top_k)



@dataclass(frozen=True)
class DedupCitationsBySnippetRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def dedup_citations_by_snippet(citations: list[dict], max_items: int, *, runtime: DedupCitationsBySnippetRuntime, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    def norm(s: str) -> str:
        s = _normalize_unicode_advanced(s or "")
        s = re.sub(r"^SECTION:\s*[^\n]+\n?", "", s, flags=re.IGNORECASE).strip()

        lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
        cleaned = []
        seen_lines = set()

        for ln in lines:
            ln_low = re.sub(r"\s+", " ", ln.lower()).strip()

            if re.fullmatch(r"\d+", ln_low):
                continue

            if ln_low in seen_lines:
                continue

            seen_lines.add(ln_low)
            cleaned.append(ln_low)

        s = " ".join(cleaned)
        s = re.sub(r"\s+", " ", s).strip()
        return s[:500]

    def priority(c: dict) -> tuple[float, float, float]:
        return (
            float(c.get("retrieval_score", c.get("similarity", 0.0)) or 0.0),
            float(c.get("similarity", 0.0) or 0.0),
            float(c.get("rrf_score", 0.0) or 0.0),
        )

    best = {}
    chosen = {} if lineage is not None else None
    for item_index, c in enumerate(citations):
        k = norm(c.get("snippet", ""))
        if k:
            k = (
                f"{str(c.get('bubble_document_id') or '').strip()}"
                f"|{int(c.get('page_from') or 0)}"
                f"|{int(c.get('page_to') or 0)}"
                f"|{k[:220]}"
            )
        else:
            k = str(c.get("citation_id") or "").strip()

        prev = best.get(k)
        if prev is None or priority(c) > priority(prev):
            best[k] = c
            if lineage is not None:
                chosen[k] = item_index

    # Selected occurrence only; discarded duplicates do not contribute content.
    emitted = {} if lineage is not None else None
    if lineage is not None:
        for key, pos in chosen.items():
            emitted.setdefault(id(best[key]), []).append(((0, pos),))
    out = list(best.values())
    out.sort(
        key=lambda x: (
            -priority(x)[0],
            -priority(x)[1],
            -priority(x)[2],
            0 if bool(x.get("exact_machine_scope")) else 1,
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )
    if lineage is not None:
        lineage(tuple(emitted[id(item)].pop(0) for item in out[:max_items]))
    return out[:max_items]


def dedup_citations_preserve_order(citations: list[dict], max_items: int, *, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    """Deduplicate citations while preserving supplied priority order.

    Used for structured answers where procedure/step records must remain before
    secondary manual support, regardless of similarity/debug score.
    """
    out: list[dict] = []
    seen: set[tuple[str, int, int, str]] = set()
    origins = [] if lineage is not None else None
    for item_index, c in enumerate(citations or []):
        if not isinstance(c, dict):
            continue
        bdid = str(c.get("bubble_document_id") or "").strip()
        pf = int(c.get("page_from") or 0)
        pt = int(c.get("page_to") or 0)
        cid = str(c.get("citation_id") or "").strip()
        key = (bdid, pf, pt, cid or str(c.get("snippet") or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if lineage is not None:
            origins.append(((0, item_index),))
        if len(out) >= max_items:
            break
    if lineage is not None:
        lineage(tuple(origins))
    return out


@dataclass(frozen=True)
class MmrSelectRuntime:
    _cosine_sim: Callable[..., Any]


def mmr_select(q_vec: list[float], candidates: list[dict], top_k: int, lambda_mult: float=0.85, *, runtime: MmrSelectRuntime, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    _cosine_sim = runtime._cosine_sim
    if not candidates:
        if lineage is not None:
            lineage(())
        return []

    selected: list[dict] = []
    remaining = candidates[:]

    if lineage is not None:
        _positions = {}
        for _index, _item in enumerate(remaining):
            _positions.setdefault(id(_item), []).append(_index)
    remaining.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    if lineage is not None:
        _remaining_positions = [_positions[id(_item)].pop(0) for _item in remaining]
        _selected_positions = [((0, _remaining_positions.pop(0)),)]
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_score = -1e9

        for i, cand in enumerate(remaining):
            sim_q = float(cand.get("similarity", 0.0))

            max_sim_sel = 0.0
            ce = cand.get("embedding_list") or []
            for s in selected:
                se = s.get("embedding_list") or []
                max_sim_sel = max(max_sim_sel, _cosine_sim(ce, se))

            score = lambda_mult * sim_q - (1.0 - lambda_mult) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))
        if lineage is not None:
            _selected_positions.append(((0, _remaining_positions.pop(best_idx)),))

    if lineage is not None:
        lineage(tuple(_selected_positions))
    return selected



def v12_structured_rank(c: dict, used_ids: set[str]) -> tuple:
    cid = str(c.get("citation_id") or "")
    return (
        0 if cid in used_ids else 1,
        0 if bool(c.get("exact_machine_scope")) else 1,
        -float(c.get("structured_direct_score") or c.get("retrieval_score") or c.get("similarity") or 0.0),
        str(c.get("bubble_document_id") or ""),
    )


@dataclass(frozen=True)
class V12MergeCandidateMetadataRuntime:
    _dedup_text_values: Callable[..., Any]


def v12_merge_candidate_metadata(preferred: dict, secondary: dict, *, runtime: V12MergeCandidateMetadataRuntime) -> dict:
    """Preserve semantic/facet annotations while keeping the better source body."""
    _dedup_text_values = runtime._dedup_text_values
    out = dict(secondary or {})
    out.update(dict(preferred or {}))
    for key in (
        "assistant_core_facet_hits",
        "assistant_core_covered_facets",
        "matched_subsystems",
    ):
        merged = _dedup_text_values(
            list((preferred or {}).get(key) or []) + list((secondary or {}).get(key) or []),
            limit=24,
        )
        if merged:
            out[key] = merged
    for key in (
        "v13_score",
        "retrieval_score",
        "similarity",
        "semantic_similarity",
        "assistant_core_facet_coverage",
    ):
        values = []
        for source in (preferred or {}, secondary or {}):
            try:
                values.append(float(source.get(key) or 0.0))
            except Exception:
                pass
        if values:
            out[key] = max(values)
    return out


@dataclass(frozen=True)
class V12DedupeFamilyStepsRuntime:
    _v12_merge_candidate_metadata: Callable[..., Any]
    _v12_step_sort_key: Callable[..., Any]
    _v12_structured_rank: Callable[..., Any]


def v12_dedupe_family_steps(steps: list[dict], *, runtime: V12DedupeFamilyStepsRuntime) -> list[dict]:
    """Deduplicate Step representations and conflicting duplicate ordinals."""
    _v12_merge_candidate_metadata = runtime._v12_merge_candidate_metadata
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v12_structured_rank = runtime._v12_structured_rank
    best_by_doc: dict[str, dict] = {}
    for candidate in steps or []:
        if not isinstance(candidate, dict):
            continue
        bdid = str(candidate.get("bubble_document_id") or "").strip()
        if not bdid:
            continue
        current = best_by_doc.get(bdid)
        if current is None:
            best_by_doc[bdid] = dict(candidate)
            continue
        if _v12_structured_rank(candidate, set()) < _v12_structured_rank(current, set()):
            best_by_doc[bdid] = _v12_merge_candidate_metadata(candidate, current)
        else:
            best_by_doc[bdid] = _v12_merge_candidate_metadata(current, candidate)

    best_by_number: dict[tuple[int, str], dict] = {}
    for candidate in best_by_doc.values():
        number = _v12_step_sort_key(candidate)[0]
        key = (number, "") if 0 < number < 9999 else (number, str(candidate.get("bubble_document_id") or ""))
        current = best_by_number.get(key)
        if current is None:
            best_by_number[key] = candidate
            continue
        current_relation = str(current.get("structured_relation_source") or "")
        new_relation = str(candidate.get("structured_relation_source") or "")
        current_priority = 0 if current_relation == "structured_source_relations" else 1
        new_priority = 0 if new_relation == "structured_source_relations" else 1
        if (new_priority, _v12_structured_rank(candidate, set())) < (
            current_priority,
            _v12_structured_rank(current, set()),
        ):
            best_by_number[key] = _v12_merge_candidate_metadata(candidate, current)
        else:
            best_by_number[key] = _v12_merge_candidate_metadata(current, candidate)
    return sorted(best_by_number.values(), key=_v12_step_sort_key)


def v13_merge_candidates(candidate_lists: list[list[dict]], *, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    by_id: dict[str, dict] = {}
    origins = {} if lineage is not None else None
    for group_index, candidates in enumerate(candidate_lists or []):
        for item_index, raw in enumerate(candidates or []):
            if not isinstance(raw, dict):
                continue
            c = dict(raw)
            cid = str(c.get("citation_id") or "").strip()
            if not cid:
                continue
            if lineage is not None:
                origins.setdefault(cid, []).append((group_index, item_index))
            prev = by_id.get(cid)
            if prev is None:
                by_id[cid] = c
                continue

            merged = dict(prev)
            for key, value in c.items():
                if value not in (None, "", [], {}):
                    if key in {
                        "similarity", "semantic_similarity", "retrieval_score", "v13_score",
                        "ask_evidence_score", "structured_direct_score",
                        "structured_title_match_score", "structured_title_coverage",
                        "structured_title_strict_coverage", "structured_title_description_support",
                        "structured_title_matched_terms", "structured_title_term_count",
                    }:
                        try:
                            merged[key] = max(float(merged.get(key) or 0.0), float(value or 0.0))
                        except Exception:
                            merged[key] = value
                    elif key in {"chunk_full", "snippet", "snippet_clean"}:
                        if len(str(value or "")) > len(str(merged.get(key) or "")):
                            merged[key] = value
                    else:
                        merged[key] = value
            by_id[cid] = merged
    if lineage is not None:
        lineage(tuple(tuple(origins[cid]) for cid in by_id))
    return list(by_id.values())


@dataclass(frozen=True)
class V13ScoreCandidatesRuntime:
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE: float
    _candidate_source_bias: Callable[..., Any]
    _candidate_specificity_score: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _extract_code_tokens: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]
    _v13_real_semantic_similarity: Callable[..., Any]


def v13_score_candidates(q: str, candidates: list[dict], *, runtime: V13ScoreCandidatesRuntime, lineage: Optional[Callable[..., None]]=None) -> list[dict]:
    V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE = runtime.V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE
    _candidate_source_bias = runtime._candidate_source_bias
    _candidate_specificity_score = runtime._candidate_specificity_score
    _content_term_set = runtime._content_term_set
    _count_query_tokens = runtime._count_query_tokens
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _extract_code_tokens = runtime._extract_code_tokens
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _source_type_from_document_id = runtime._source_type_from_document_id
    _term_overlap_score = runtime._term_overlap_score
    _v13_candidate_text = runtime._v13_candidate_text
    _v13_real_semantic_similarity = runtime._v13_real_semantic_similarity
    query_terms = _content_term_set(q, limit=70)
    query_style = "telegraphic" if _count_query_tokens(q) <= 6 else "natural"
    token_count = _count_query_tokens(q)
    codes = {str(x).lower() for x in _extract_code_tokens(q)}

    out: list[dict] = []
    if lineage is not None:
        _origins = {}
        _input_index = -1
    for raw in candidates or []:
        if lineage is not None:
            _input_index += 1
        if not isinstance(raw, dict):
            continue
        c = dict(raw)
        text = _v13_candidate_text(c)
        text_terms = _content_term_set(text, limit=120)
        overlap = _term_overlap_score(query_terms, text_terms)
        source_bias, source_meta = _candidate_source_bias(
            c,
            query_terms,
            query_style=query_style,
            query_token_count=token_count,
        )
        specificity = _candidate_specificity_score(c)
        routing_similarity = max(0.0, float(c.get("similarity") or 0.0))
        semantic_similarity = _v13_real_semantic_similarity(c)
        source_type = str(
            c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")
        )

        normalized_candidate_text = _normalize_unicode_advanced(text).lower()
        exact_code_hit = any(code and code in normalized_candidate_text for code in codes)

        # Start from true cosine similarity only. Synthetic page/structured scores may
        # help order candidates *after* an independent lexical or identifier signal,
        # but they can never create relevance by themselves.
        base = semantic_similarity
        title_match_score = max(0.0, min(1.0, float(c.get("structured_title_match_score") or 0.0)))
        title_match_terms = int(c.get("structured_title_matched_terms") or 0)
        title_support = bool(
            c.get("structured_title_match")
            and title_match_terms >= 2
            and title_match_score >= V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE
        )
        has_real_support_signal = bool(
            exact_code_hit
            or semantic_similarity >= 0.18
            or overlap >= 0.02
            or title_support
            or (bool(c.get("fts_v13")) and overlap > 0.0)
        )
        if has_real_support_signal and (exact_code_hit or overlap > 0.0):
            base = max(base, min(0.70, routing_similarity))
        if c.get("ask_evidence_score") is not None and has_real_support_signal:
            base = max(base, min(0.88, 0.44 + float(c.get("ask_evidence_score") or 0.0) / 180.0))
        if bool(c.get("ask_structured_direct")) and has_real_support_signal:
            base = max(base, min(0.82, 0.46 + 0.018 * float(c.get("structured_direct_score") or 0.0)))
        if title_support:
            # A strong title/description match is direct source evidence, not a generic
            # source-type bonus. It may rank a source candidate but still cannot bypass
            # the semantic ASK task/evidence gate.
            base = max(base, min(0.90, title_match_score))
        if bool(c.get("fts_v13")) and overlap > 0.0:
            base = max(base, 0.42 + min(0.18, 0.50 * overlap))

        if exact_code_hit:
            base += 0.12

        v13_score = (
            base
            + 0.16 * overlap
            + 0.55 * specificity
            + 0.55 * source_bias
            + (0.04 if bool(c.get("exact_machine_scope")) else 0.0)
        )
        c.update(source_meta)
        c["source_type"] = source_type
        c["semantic_similarity"] = semantic_similarity
        c["routing_similarity"] = routing_similarity
        c["specificity_score"] = specificity
        c["overlap_score"] = overlap
        c["exact_code_hit"] = exact_code_hit
        c["structured_title_support"] = title_support
        c["v13_score"] = float(v13_score)
        c["retrieval_score"] = float(v13_score)
        out.append(c)
        if lineage is not None:
            _origins[id(c)] = ((0, _input_index),)

    out.sort(
        key=lambda c: (
            -float(c.get("v13_score") or 0.0),
            -float(c.get("semantic_similarity") or 0.0),
            -float(c.get("overlap_score") or 0.0),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
            int(c.get("chunk_index") or 0),
        )
    )
    if lineage is not None:
        _parents = tuple(_origins[id(c)] for c in out)
        _before_selection = tuple(out)
        _result = _dedup_citations_by_snippet(out, max_items=max(20, len(out)))
        lineage(_selected_occurrence_lineage(_before_selection, _parents, _result))
        return _result
    return _dedup_citations_by_snippet(out, max_items=max(20, len(out)))



@dataclass(frozen=True)
class V13RescoreRootCandidatesRuntime:
    ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY: float
    ROOT_CAUSE_HARD_EXCLUDE_PENALTY: float
    _classify_diagnostic_role_from_text: Callable[..., Any]
    _collect_candidate_keywords: Callable[..., Any]
    _dedup_root_cause_candidates_semantic: Callable[..., Any]
    _prioritize_root_cause_coverage: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]
    _root_cause_target_subsystems: Callable[..., Any]
    _score_root_cause_causal_strength: Callable[..., Any]
    _score_root_cause_chunk_semantic: Callable[..., Any]
    _score_root_cause_context_fit: Callable[..., Any]
    _score_root_cause_subsystem_alignment: Callable[..., Any]
    _should_downrank_generic_root_cause_chunk: Callable[..., Any]
    _should_hard_exclude_root_cause_chunk: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]


def v13_rescore_root_candidates(q: str, candidates: list[dict], *, runtime: V13RescoreRootCandidatesRuntime) -> list[dict]:
    ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY = runtime.ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY
    ROOT_CAUSE_HARD_EXCLUDE_PENALTY = runtime.ROOT_CAUSE_HARD_EXCLUDE_PENALTY
    _classify_diagnostic_role_from_text = runtime._classify_diagnostic_role_from_text
    _collect_candidate_keywords = runtime._collect_candidate_keywords
    _dedup_root_cause_candidates_semantic = runtime._dedup_root_cause_candidates_semantic
    _prioritize_root_cause_coverage = runtime._prioritize_root_cause_coverage
    _query_symptom_profile = runtime._query_symptom_profile
    _root_cause_target_subsystems = runtime._root_cause_target_subsystems
    _score_root_cause_causal_strength = runtime._score_root_cause_causal_strength
    _score_root_cause_chunk_semantic = runtime._score_root_cause_chunk_semantic
    _score_root_cause_context_fit = runtime._score_root_cause_context_fit
    _score_root_cause_subsystem_alignment = runtime._score_root_cause_subsystem_alignment
    _should_downrank_generic_root_cause_chunk = runtime._should_downrank_generic_root_cause_chunk
    _should_hard_exclude_root_cause_chunk = runtime._should_hard_exclude_root_cause_chunk
    _v13_candidate_text = runtime._v13_candidate_text
    diagnostic_keywords = _collect_candidate_keywords(q, [])
    target_subsystems = _root_cause_target_subsystems(q, [])
    symptom_profile = _query_symptom_profile(q)

    rescored: list[dict] = []
    for raw in candidates or []:
        c = dict(raw)
        text = _v13_candidate_text(c)
        semantic = _score_root_cause_chunk_semantic(q, text, diagnostic_keywords)
        causal = _score_root_cause_causal_strength(q, text, diagnostic_keywords)
        subsystem = _score_root_cause_subsystem_alignment(q, text, target_subsystems)
        context_fit = _score_root_cause_context_fit(
            q=q,
            chunk_text=text,
            diagnostic_keywords=diagnostic_keywords,
            symptom_profile=symptom_profile,
            matched_subsystems=subsystem.get("matched_subsystems") or [],
        )
        generic_downranked = _should_downrank_generic_root_cause_chunk(q, text, diagnostic_keywords)
        hard_excluded = _should_hard_exclude_root_cause_chunk(q, text, diagnostic_keywords)
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=text,
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )

        score = float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0)
        score += float(semantic.get("semantic_score") or 0.0)
        score += float(causal.get("causal_strength_score") or 0.0)
        score += float(subsystem.get("subsystem_score") or 0.0)
        score += float(context_fit.get("context_fit_score") or 0.0)
        score += float(role.get("role_adjustment") or 0.0)
        if generic_downranked:
            score -= ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY
        if hard_excluded:
            score -= ROOT_CAUSE_HARD_EXCLUDE_PENALTY

        c.update(semantic)
        c.update(causal)
        c.update(subsystem)
        c.update(context_fit)
        c.update(role)
        c["generic_downranked"] = bool(generic_downranked)
        c["hard_excluded"] = bool(hard_excluded)
        c["v13_score"] = float(score)
        c["retrieval_score"] = float(score)
        rescored.append(c)

    rescored.sort(
        key=lambda c: (
            1 if bool(c.get("hard_excluded")) else 0,
            0 if str(c.get("role_group") or "") == "core" else 1,
            -float(c.get("v13_score") or 0.0),
            -float(c.get("similarity") or 0.0),
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
        )
    )
    non_excluded = [c for c in rescored if not bool(c.get("hard_excluded"))]
    pool = non_excluded if len(non_excluded) >= 4 else rescored
    pool = _dedup_root_cause_candidates_semantic(pool, max_items=24)
    pool = _prioritize_root_cause_coverage(pool, max_items=20)
    return pool


@dataclass(frozen=True)
class V13MergeSourceTitleCandidatesRuntime:
    V13_MAX_EVIDENCE_ITEMS_ASK: int
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES: int
    _v13_evidence_metrics: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]
    _v13_score_candidates: Callable[..., Any]


def v13_merge_source_title_candidates(q: str, retrieval: dict, title_candidates: list[dict], *, runtime: V13MergeSourceTitleCandidatesRuntime) -> dict:
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES = runtime.V13_SOURCE_RETRIEVAL_MAX_CANDIDATES
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_score_candidates = runtime._v13_score_candidates
    if not title_candidates:
        return dict(retrieval or {})
    original = dict(retrieval or {})
    merged = _v13_merge_candidates(
        [list(original.get("candidates") or []), list(title_candidates or [])]
    )
    scored = _v13_score_candidates(q, merged)
    out = dict(original)
    out["candidates"] = scored
    out["citations"] = scored[:V13_MAX_EVIDENCE_ITEMS_ASK]
    out["metrics"] = _v13_evidence_metrics(scored)
    out["source_title_candidates"] = [
        {
            "citation_id": str(c.get("citation_id") or ""),
            "bubble_document_id": str(c.get("bubble_document_id") or ""),
            "source_type": str(c.get("source_type") or ""),
            "title": str(c.get("structured_title") or ""),
            "score": round(float(c.get("structured_title_match_score") or 0.0), 6),
        }
        for c in title_candidates[:V13_SOURCE_RETRIEVAL_MAX_CANDIDATES]
    ]
    return out


@dataclass(frozen=True)
class V13MergeSourceProbeCandidatesRuntime:
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES: int
    _v13_real_semantic_similarity: Callable[..., Any]


def v13_merge_source_probe_candidates(*groups: list[dict], runtime: V13MergeSourceProbeCandidatesRuntime) -> list[dict]:
    V13_SOURCE_RETRIEVAL_MAX_CANDIDATES = runtime.V13_SOURCE_RETRIEVAL_MAX_CANDIDATES
    _v13_real_semantic_similarity = runtime._v13_real_semantic_similarity
    by_doc: dict[str, dict] = {}
    for group in groups:
        for raw in group or []:
            if not isinstance(raw, dict):
                continue
            c = dict(raw)
            bdid = str(c.get("bubble_document_id") or "").strip()
            if not bdid:
                continue
            c.setdefault(
                "source_retrieval_probe_score",
                max(
                    float(c.get("structured_title_match_score") or 0.0),
                    0.92 * _v13_real_semantic_similarity(c),
                ),
            )
            previous = by_doc.get(bdid)
            if previous is None or float(c.get("source_retrieval_probe_score") or 0.0) > float(previous.get("source_retrieval_probe_score") or 0.0):
                by_doc[bdid] = c
    ordered = sorted(
        by_doc.values(),
        key=lambda c: (
            -float(c.get("source_retrieval_probe_score") or 0.0),
            -float(c.get("structured_title_match_score") or 0.0),
            -_v13_real_semantic_similarity(c),
            str(c.get("bubble_document_id") or ""),
        ),
    )
    return ordered[:V13_SOURCE_RETRIEVAL_MAX_CANDIDATES]


@dataclass(frozen=True)
class AssistantCoreMergeFacetCandidatesRuntime:
    _assistant_core_candidate_stable_key: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]


def assistant_core_merge_facet_candidates(candidate_lists: list[list[dict]], *, runtime: AssistantCoreMergeFacetCandidatesRuntime, lineage: Optional[Callable[..., Any]]=None) -> list[dict]:
    """Merge candidates while preserving all facet annotations across searches."""
    _assistant_core_candidate_stable_key = runtime._assistant_core_candidate_stable_key
    _dedup_text_values = runtime._dedup_text_values
    _v13_merge_candidates = runtime._v13_merge_candidates
    merged = _v13_merge_candidates(candidate_lists)
    annotations: dict[str, dict] = {}
    if lineage is not None:
        _lineage_annotation_inputs = {}
    for candidates in candidate_lists or []:
        for raw in candidates or []:
            if not isinstance(raw, dict):
                continue
            key = _assistant_core_candidate_stable_key(raw)
            if not key:
                continue
            item = annotations.setdefault(
                key,
                {
                    "facets": [],
                    "types": [],
                    "preferred": [],
                    "must_cover": [],
                    "score": 0.0,
                    "score_map": {},
                },
            )
            if lineage is not None:
                _lineage_annotation_inputs.setdefault(id(item), (item, []))[1].append(raw)
            item["facets"].extend(raw.get("assistant_core_facet_hits") or [])
            item["types"].extend(raw.get("assistant_core_facet_answer_types") or [])
            item["preferred"].extend(raw.get("assistant_core_facet_preferred_source_types") or [])
            item["must_cover"].extend(raw.get("assistant_core_facet_must_cover") or [])
            item["score"] = max(
                float(item.get("score") or 0.0),
                float(raw.get("assistant_core_facet_retrieval_score") or 0.0),
            )
            for facet_name, facet_score in dict(raw.get("assistant_core_facet_score_map") or {}).items():
                name = str(facet_name or "").strip()
                if not name:
                    continue
                item["score_map"][name] = max(
                    float(item["score_map"].get(name) or 0.0), float(facet_score or 0.0)
                )

    out: list[dict] = []
    for raw in merged:
        c = dict(raw)
        item = annotations.get(_assistant_core_candidate_stable_key(c)) or {}
        c["assistant_core_facet_hits"] = _dedup_text_values(item.get("facets") or [], limit=12)
        c["assistant_core_facet_answer_types"] = _dedup_text_values(item.get("types") or [], limit=10)
        c["assistant_core_facet_preferred_source_types"] = _dedup_text_values(item.get("preferred") or [], limit=8)
        c["assistant_core_facet_must_cover"] = _dedup_text_values(item.get("must_cover") or [], limit=12)
        c["assistant_core_facet_retrieval_score"] = float(item.get("score") or 0.0)
        c["assistant_core_facet_score_map"] = dict(item.get("score_map") or {})
        if lineage is not None:
            lineage("facet_merge", raw, c, tuple(_lineage_annotation_inputs.get(id(item), (None, ()))[1]))
        out.append(c)
    return out




def _selected_occurrence_lineage(inputs, parents, selected):
    """Trace a selecting collaborator's actual objects within this operation.

    Positions were captured when the operator made each occurrence. Identity is
    local to these explicit inputs, not source authorization or a content/ID join.
    A collaborator returning copies/new records needs its own derivation producer;
    it is deliberately rejected rather than matched back by text or citation ID.
    Existing snippet dedup returns its selected input objects. Repeated aliases
    consume their captured positions in order; no persistent registry is created.
    """
    if type(selected) is not list or len(inputs) != len(parents):
        raise ValueError("invalid selecting collaborator output")
    available = {}
    for item, origin in zip(inputs, parents):
        available.setdefault(id(item), []).append((item, origin))
    result = []
    for item in selected:
        occurrences = available.get(id(item))
        if not occurrences or occurrences[0][0] is not item:
            raise ValueError("selecting collaborator returned an untracked occurrence")
        result.append(occurrences.pop(0)[1])
    return tuple(result)
