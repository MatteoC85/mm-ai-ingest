"""P4-C5: unchanged legacy retrieval coordinators behind explicit runtime dependencies.

Shared semantic retrieval, diagnostic evidence expansion and role-aware ranking
are retained for legacy ASK, Root Cause, draft and Smart consumers. The caller
supplies existing planning, retrieval, scoring and matrix callbacks; this module
performs no application initialization, provider imports or direct DB access.
This extraction deliberately preserves thresholds, call order and error paths.
No claim is made that the inherited semantic policies are universally correct.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from assistant_core_v2 import AssistantCoreRequest

@dataclass(frozen=True)
class RetrievalQualityScoreRuntime:
    _source_type_from_document_id: Callable[..., Any]


def retrieval_quality_score(retrieval: dict, *, runtime: RetrievalQualityScoreRuntime) -> float:
    _source_type_from_document_id = runtime._source_type_from_document_id
    citations = list((retrieval or {}).get("citations") or [])
    if not citations:
        return -1.0

    sim_max = float((retrieval or {}).get("similarity_max") or 0.0)
    manual_count = sum(1 for c in citations if _source_type_from_document_id(c.get("bubble_document_id") or "") == "manual")
    max_overlap = max((float(c.get("overlap_score", 0.0)) for c in citations), default=0.0)
    max_specificity = max((float(c.get("specificity_score", 0.0)) for c in citations), default=0.0)
    generic_structured = sum(
        1
        for c in citations
        if _source_type_from_document_id(c.get("bubble_document_id") or "") in {"ps", "procedure", "step"}
        and float(c.get("overlap_score", 0.0)) < 0.20
        and float(c.get("specificity_score", 0.0)) < 0.04
    )

    return (
        0.34 * sim_max
        + (0.22 if manual_count > 0 else 0.0)
        + 0.24 * max_overlap
        + 0.16 * max_specificity
        - 0.07 * generic_structured
    )


@dataclass(frozen=True)
class SharedSemanticRetrievalRuntime:
    SEMANTIC_MAX_DENSE_QUERIES: Any
    SEMANTIC_MAX_LEXICAL_QUERIES: Any
    _augment_crosslingual_query_plan: Callable[..., Any]
    _candidate_order_key: Callable[..., Any]
    _candidate_source_bias: Callable[..., Any]
    _candidate_specificity_score: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _dense_candidates_multi_query: Callable[..., Any]
    _effective_similarity_threshold: Callable[..., Any]
    _fetch_structured_rescue_candidates: Callable[..., Any]
    _fts_search_chunks_multi: Callable[..., Any]
    _fts_search_chunks_prefix: Callable[..., Any]
    _llm_rerank_citations: Callable[..., Any]
    _lock_final_citations: Callable[..., Any]
    _mmr_select: Callable[..., Any]
    _planner_query_term_set: Callable[..., Any]
    _promote_structured_rescue_hits: Callable[..., Any]
    _rebalance_selected_citations: Callable[..., Any]
    _semantic_query_plan: Callable[..., Any]
    _should_use_reranker: Callable[..., Any]


def shared_semantic_retrieval(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "ask",
    base_threshold: float,
    diagnostic_mode: bool = False,
    runtime: SharedSemanticRetrievalRuntime,
) -> dict:
    SEMANTIC_MAX_DENSE_QUERIES = runtime.SEMANTIC_MAX_DENSE_QUERIES
    SEMANTIC_MAX_LEXICAL_QUERIES = runtime.SEMANTIC_MAX_LEXICAL_QUERIES
    _augment_crosslingual_query_plan = runtime._augment_crosslingual_query_plan
    _candidate_order_key = runtime._candidate_order_key
    _candidate_source_bias = runtime._candidate_source_bias
    _candidate_specificity_score = runtime._candidate_specificity_score
    _count_query_tokens = runtime._count_query_tokens
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _dedup_text_values = runtime._dedup_text_values
    _dense_candidates_multi_query = runtime._dense_candidates_multi_query
    _effective_similarity_threshold = runtime._effective_similarity_threshold
    _fetch_structured_rescue_candidates = runtime._fetch_structured_rescue_candidates
    _fts_search_chunks_multi = runtime._fts_search_chunks_multi
    _fts_search_chunks_prefix = runtime._fts_search_chunks_prefix
    _llm_rerank_citations = runtime._llm_rerank_citations
    _lock_final_citations = runtime._lock_final_citations
    _mmr_select = runtime._mmr_select
    _planner_query_term_set = runtime._planner_query_term_set
    _promote_structured_rescue_hits = runtime._promote_structured_rescue_hits
    _rebalance_selected_citations = runtime._rebalance_selected_citations
    _semantic_query_plan = runtime._semantic_query_plan
    _should_use_reranker = runtime._should_use_reranker
    planner = _semantic_query_plan(q, mode=planner_mode)
    planner = _augment_crosslingual_query_plan(q, planner)

    dense_queries = _dedup_text_values(
        [q, planner.get("normalized_query")]
        + list(planner.get("dense_queries") or [])
        + list(planner.get("crosslingual_dense_queries") or []),
        limit=max(3, SEMANTIC_MAX_DENSE_QUERIES + 2),
    )
    lexical_queries = _dedup_text_values(
        [q, planner.get("normalized_query")]
        + list(planner.get("lexical_queries") or [])
        + list(planner.get("crosslingual_lexical_queries") or []),
        limit=max(3, SEMANTIC_MAX_LEXICAL_QUERIES + 2),
    )

    chunks_matching_filter, candidates, query_vectors = _dense_candidates_multi_query(
        query_texts=dense_queries,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
    )

    sim_max = max((float(c.get("similarity", 0.0)) for c in candidates), default=None) if candidates else None
    effective_threshold = _effective_similarity_threshold(
        q,
        planner=planner,
        base_threshold=base_threshold,
    )

    rerank_used = False
    rerank_error: Optional[str] = None
    fts_used = False
    prefix_fts_used = False
    exact_fts_used = False

    selected_citations: list[dict] = []
    cut_candidates: list[dict] = []
    query_terms = _planner_query_term_set(q, planner)
    query_style = str((planner or {}).get("query_style") or "").strip().lower()
    query_token_count = _count_query_tokens(q)

    structured_rescue_hits: list[dict] = []
    # Keep root-cause diagnostic retrieval untouched: this rescue is for Ask/how-to
    # and draft support, where exact structured records must not be hidden by manuals.
    if not diagnostic_mode and str(planner_mode or "").strip().lower() != "root_cause":
        structured_rescue_hits = _fetch_structured_rescue_candidates(
            company_id=company_id,
            machine_id=machine_id,
            q=q,
            planner=planner,
            top_k=top_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )

    if candidates:
        max_rrf = max(float(c.get("rrf_score", 0.0)) for c in candidates) if candidates else 0.0
        prepared: list[dict] = []

        for c in candidates:
            cc = dict(c)
            rrf_norm = (float(cc.get("rrf_score", 0.0)) / max_rrf) if max_rrf > 0 else 0.0
            source_bias, source_meta = _candidate_source_bias(
                cc,
                query_terms,
                query_style=query_style,
                query_token_count=query_token_count,
            )
            specificity_score = _candidate_specificity_score(cc)
            overlap_score = float(source_meta.get("overlap_score", 0.0))

            cc.update(source_meta)
            cc["specificity_score"] = specificity_score
            cc["source_bias"] = source_bias
            cc["retrieval_score"] = (
                0.58 * float(cc.get("similarity", 0.0))
                + 0.17 * rrf_norm
                + 0.13 * overlap_score
                + specificity_score
                + source_bias
            )
            prepared.append(cc)

        prepared.sort(key=_candidate_order_key)

        sim_delta = 0.15
        if sim_max is not None:
            if sim_max >= 0.55:
                sim_delta = 0.10
            elif sim_max >= 0.40:
                sim_delta = 0.12

        cut_candidates = []
        for c in prepared:
            if sim_max is None:
                cut_candidates.append(c)
                continue
            if (float(sim_max) - float(c.get("similarity", 0.0))) <= sim_delta:
                cut_candidates.append(c)

        min_keep = min(len(prepared), max(4, top_k))
        if len(cut_candidates) < min_keep:
            cut_candidates = prepared[:min(len(prepared), max(top_k * 3, 12))]
        else:
            cut_candidates = cut_candidates[:min(len(cut_candidates), max(top_k * 3, 14))]

        q_vec = (
            query_vectors.get(q)
            or query_vectors.get(str(planner.get("normalized_query") or ""))
            or next(iter(query_vectors.values()), [])
        )

        selected_citations = _mmr_select(
            q_vec,
            cut_candidates,
            top_k=top_k,
            lambda_mult=0.88 if diagnostic_mode else 0.86,
        )

        if sim_max is not None and _should_use_reranker(q=q, candidates=cut_candidates, sim_max=float(sim_max), top_k=top_k):
            try:
                reranked_ids = _llm_rerank_citations(
                    q=q,
                    candidates=cut_candidates,
                    top_k=top_k,
                    diagnostic_mode=diagnostic_mode,
                )
                if reranked_ids:
                    by_id = {str(c.get("citation_id") or "").strip(): c for c in cut_candidates}
                    reranked = [by_id[cid] for cid in reranked_ids if cid in by_id]
                    if reranked:
                        selected_citations = reranked
                        rerank_used = True
            except Exception as e:
                rerank_error = str(e)

        selected_citations = _dedup_citations_by_snippet(selected_citations, max_items=top_k)
        selected_citations = _rebalance_selected_citations(
            selected_citations=selected_citations,
            ranked_candidates=cut_candidates,
            top_k=top_k,
            query_style=query_style,
            query_token_count=query_token_count,
        )
        selected_citations = _lock_final_citations(
            selected_citations=selected_citations,
            ranked_candidates=cut_candidates,
            top_k=top_k,
            diagnostic_mode=diagnostic_mode,
            query_token_count=query_token_count,
        )

    if (sim_max is None or float(sim_max) < effective_threshold) or not selected_citations:
        prefix_hits = _fts_search_chunks_prefix(
            company_id=company_id,
            machine_id=machine_id,
            texts=lexical_queries,
            top_k=top_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if prefix_hits:
            fts_used = True
            prefix_fts_used = True
            selected_citations = _dedup_citations_by_snippet(selected_citations + prefix_hits, max_items=top_k)

        exact_hits = _fts_search_chunks_multi(
            company_id=company_id,
            machine_id=machine_id,
            queries=lexical_queries,
            top_k=top_k,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if exact_hits:
            fts_used = True
            exact_fts_used = True
            selected_citations = _dedup_citations_by_snippet(selected_citations + exact_hits, max_items=top_k)

    if selected_citations:
        selected_citations = _lock_final_citations(
            selected_citations=selected_citations,
            ranked_candidates=list(cut_candidates or []) + list(selected_citations or []),
            top_k=top_k,
            diagnostic_mode=diagnostic_mode,
            query_token_count=query_token_count,
        )

    if structured_rescue_hits:
        selected_citations = _promote_structured_rescue_hits(
            selected_citations=selected_citations,
            structured_hits=structured_rescue_hits,
            top_k=top_k,
        )

    return {
        "planner": planner,
        "dense_queries": dense_queries,
        "lexical_queries": lexical_queries,
        "chunks_matching_filter": chunks_matching_filter,
        "similarity_max": sim_max,
        "effective_threshold": effective_threshold,
        "candidates": cut_candidates,
        "citations": selected_citations,
        "fts_used": fts_used,
        "prefix_fts_used": prefix_fts_used,
        "exact_fts_used": exact_fts_used,
        "rerank_used": rerank_used,
        "rerank_error": rerank_error[:300] if rerank_error else None,
    }


@dataclass(frozen=True)
class EnsureCandidateRetrievalFieldsRuntime:
    _candidate_source_bias: Callable[..., Any]
    _candidate_specificity_score: Callable[..., Any]


def ensure_candidate_retrieval_fields(
    citations: list[dict],
    *,
    query_terms: set[str],
    query_style: str = "",
    query_token_count: int = 0,
    runtime: EnsureCandidateRetrievalFieldsRuntime,
) -> list[dict]:
    _candidate_source_bias = runtime._candidate_source_bias
    _candidate_specificity_score = runtime._candidate_specificity_score
    out: list[dict] = []
    max_rrf = max((float((c or {}).get("rrf_score", 0.0) or 0.0) for c in (citations or [])), default=0.0)

    for c in citations or []:
        if not isinstance(c, dict):
            continue

        cc = dict(c)
        cc["snippet"] = (cc.get("snippet") or cc.get("chunk_full") or "").strip()
        cc["chunk_full"] = (cc.get("chunk_full") or cc.get("snippet") or "").strip()
        source_bias, source_meta = _candidate_source_bias(
            cc,
            query_terms,
            query_style=query_style,
            query_token_count=query_token_count,
        )
        specificity_score = _candidate_specificity_score(cc)
        overlap_score = float(source_meta.get("overlap_score", 0.0))
        rrf_norm = (float(cc.get("rrf_score", 0.0) or 0.0) / max_rrf) if max_rrf > 0 else 0.0
        base_retrieval_score = (
            0.58 * float(cc.get("similarity", 0.0) or 0.0)
            + 0.17 * rrf_norm
            + 0.13 * overlap_score
            + specificity_score
            + source_bias
        )

        cc.update(source_meta)
        cc["specificity_score"] = float(cc.get("specificity_score", specificity_score) or 0.0)
        cc["source_bias"] = float(cc.get("source_bias", source_bias) or 0.0)
        cc["overlap_score"] = float(cc.get("overlap_score", overlap_score) or 0.0)
        cc["retrieval_score"] = float(cc.get("retrieval_score", base_retrieval_score) or 0.0)
        out.append(cc)

    return out


@dataclass(frozen=True)
class DiagnosticEvidencePipelineRuntime:
    DIAGNOSTIC_PIPELINE_ENABLED: Any
    ROOT_CAUSE_DIRECT_SIGNAL_BONUS: Any
    ROOT_CAUSE_EXTRA_CANDIDATE_K: Any
    ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY: Any
    ROOT_CAUSE_HARD_EXCLUDE_PENALTY: Any
    ROOT_CAUSE_MAX_EVIDENCE_POOL: Any
    ROOT_CAUSE_MAX_PROMPT_CITATIONS: Any
    SEMANTIC_MAX_DENSE_QUERIES: Any
    _build_diagnostic_queries: Callable[..., Any]
    _collect_candidate_keywords: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _dedup_root_cause_candidates_semantic: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _dense_candidates_multi_query: Callable[..., Any]
    _ensure_candidate_retrieval_fields: Callable[..., Any]
    _infer_machine_components: Callable[..., Any]
    _llm_build_diagnostic_evidence_matrix: Callable[..., Any]
    _llm_filter_diagnostic_chunks: Callable[..., Any]
    _lock_final_citations: Callable[..., Any]
    _planner_query_term_set: Callable[..., Any]
    _prioritize_root_cause_coverage: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]
    _reorder_citations_by_priority_ids: Callable[..., Any]
    _root_cause_target_subsystems: Callable[..., Any]
    _score_root_cause_causal_strength: Callable[..., Any]
    _score_root_cause_chunk_semantic: Callable[..., Any]
    _score_root_cause_context_fit: Callable[..., Any]
    _score_root_cause_subsystem_alignment: Callable[..., Any]
    _select_prompt_citations_from_matrix: Callable[..., Any]
    _shared_semantic_retrieval: Callable[..., Any]
    _should_downrank_generic_root_cause_chunk: Callable[..., Any]
    _should_hard_exclude_root_cause_chunk: Callable[..., Any]
    _unique_non_empty_strings: Callable[..., Any]


def diagnostic_evidence_pipeline(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    max_causes: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "root_cause",
    base_threshold: float,
    runtime: DiagnosticEvidencePipelineRuntime,
) -> dict:
    DIAGNOSTIC_PIPELINE_ENABLED = runtime.DIAGNOSTIC_PIPELINE_ENABLED
    ROOT_CAUSE_DIRECT_SIGNAL_BONUS = runtime.ROOT_CAUSE_DIRECT_SIGNAL_BONUS
    ROOT_CAUSE_EXTRA_CANDIDATE_K = runtime.ROOT_CAUSE_EXTRA_CANDIDATE_K
    ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY = runtime.ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY
    ROOT_CAUSE_HARD_EXCLUDE_PENALTY = runtime.ROOT_CAUSE_HARD_EXCLUDE_PENALTY
    ROOT_CAUSE_MAX_EVIDENCE_POOL = runtime.ROOT_CAUSE_MAX_EVIDENCE_POOL
    ROOT_CAUSE_MAX_PROMPT_CITATIONS = runtime.ROOT_CAUSE_MAX_PROMPT_CITATIONS
    SEMANTIC_MAX_DENSE_QUERIES = runtime.SEMANTIC_MAX_DENSE_QUERIES
    _build_diagnostic_queries = runtime._build_diagnostic_queries
    _collect_candidate_keywords = runtime._collect_candidate_keywords
    _count_query_tokens = runtime._count_query_tokens
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _dedup_root_cause_candidates_semantic = runtime._dedup_root_cause_candidates_semantic
    _dedup_text_values = runtime._dedup_text_values
    _dense_candidates_multi_query = runtime._dense_candidates_multi_query
    _ensure_candidate_retrieval_fields = runtime._ensure_candidate_retrieval_fields
    _infer_machine_components = runtime._infer_machine_components
    _llm_build_diagnostic_evidence_matrix = runtime._llm_build_diagnostic_evidence_matrix
    _llm_filter_diagnostic_chunks = runtime._llm_filter_diagnostic_chunks
    _lock_final_citations = runtime._lock_final_citations
    _planner_query_term_set = runtime._planner_query_term_set
    _prioritize_root_cause_coverage = runtime._prioritize_root_cause_coverage
    _query_symptom_profile = runtime._query_symptom_profile
    _reorder_citations_by_priority_ids = runtime._reorder_citations_by_priority_ids
    _root_cause_target_subsystems = runtime._root_cause_target_subsystems
    _score_root_cause_causal_strength = runtime._score_root_cause_causal_strength
    _score_root_cause_chunk_semantic = runtime._score_root_cause_chunk_semantic
    _score_root_cause_context_fit = runtime._score_root_cause_context_fit
    _score_root_cause_subsystem_alignment = runtime._score_root_cause_subsystem_alignment
    _select_prompt_citations_from_matrix = runtime._select_prompt_citations_from_matrix
    _shared_semantic_retrieval = runtime._shared_semantic_retrieval
    _should_downrank_generic_root_cause_chunk = runtime._should_downrank_generic_root_cause_chunk
    _should_hard_exclude_root_cause_chunk = runtime._should_hard_exclude_root_cause_chunk
    _unique_non_empty_strings = runtime._unique_non_empty_strings
    top_k = max(1, int(top_k or 1))
    max_causes = max(1, min(int(max_causes or 1), 5))
    candidate_k = max(int(candidate_k or 0), max(ROOT_CAUSE_EXTRA_CANDIDATE_K, top_k * 8))
    retrieval_top_k = max(top_k, min(ROOT_CAUSE_MAX_EVIDENCE_POOL, max(top_k + 2, 8)))

    base_retrieval = _shared_semantic_retrieval(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=retrieval_top_k,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
        planner_mode=planner_mode,
        base_threshold=base_threshold,
        diagnostic_mode=True,
    )

    if not DIAGNOSTIC_PIPELINE_ENABLED:
        base_retrieval["citations"] = _lock_final_citations(
            selected_citations=list(base_retrieval.get("citations") or []),
            ranked_candidates=list(base_retrieval.get("candidates") or []) + list(base_retrieval.get("citations") or []),
            top_k=top_k,
            diagnostic_mode=True,
            query_token_count=_count_query_tokens(q),
        )
        return base_retrieval

    planner = base_retrieval.get("planner") or {}
    query_style = str(planner.get("query_style") or "").strip().lower()
    query_token_count = _count_query_tokens(q)
    query_terms = _planner_query_term_set(q, planner)

    inferred_components = _infer_machine_components(q)
    diagnostic_queries = _build_diagnostic_queries(q, inferred_components)
    diagnostic_keywords = _collect_candidate_keywords(q, inferred_components)
    target_subsystems = _root_cause_target_subsystems(q, inferred_components)
    symptom_profile = _query_symptom_profile(q)

    base_candidates = _ensure_candidate_retrieval_fields(
        list(base_retrieval.get("candidates") or []) + list(base_retrieval.get("citations") or []),
        query_terms=query_terms,
        query_style=query_style,
        query_token_count=query_token_count,
    )

    extra_dense_queries = _dedup_text_values(
        diagnostic_queries + [planner.get("normalized_query") or q],
        limit=max(4, SEMANTIC_MAX_DENSE_QUERIES + 3),
    )

    extra_candidates: list[dict] = []
    if extra_dense_queries:
        _, dense_candidates, _ = _dense_candidates_multi_query(
            query_texts=extra_dense_queries,
            company_id=company_id,
            machine_id=machine_id,
            candidate_k=max(candidate_k, ROOT_CAUSE_EXTRA_CANDIDATE_K),
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
            debug=debug,
        )
        extra_candidates = _ensure_candidate_retrieval_fields(
            dense_candidates,
            query_terms=query_terms,
            query_style=query_style,
            query_token_count=query_token_count,
        )

    merged_pool = _dedup_citations_by_snippet(
        base_candidates + extra_candidates,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 3, top_k * 4, 18),
    )

    rescored_pool: list[dict] = []
    for item in merged_pool:
        cc = dict(item)
        chunk_text = (cc.get("chunk_full") or cc.get("snippet") or "").strip()

        semantic = _score_root_cause_chunk_semantic(q, chunk_text, diagnostic_keywords)
        causal = _score_root_cause_causal_strength(q, chunk_text, diagnostic_keywords)
        subsystem = _score_root_cause_subsystem_alignment(q, chunk_text, target_subsystems)
        context_fit = _score_root_cause_context_fit(
            q=q,
            chunk_text=chunk_text,
            diagnostic_keywords=diagnostic_keywords,
            symptom_profile=symptom_profile,
            matched_subsystems=subsystem.get("matched_subsystems") or [],
        )
        generic_downranked = _should_downrank_generic_root_cause_chunk(q, chunk_text, diagnostic_keywords)
        hard_excluded = _should_hard_exclude_root_cause_chunk(q, chunk_text, diagnostic_keywords)

        diagnostic_score = float(cc.get("retrieval_score", cc.get("similarity", 0.0)) or 0.0)
        diagnostic_score += float(semantic.get("semantic_score", 0.0) or 0.0)
        diagnostic_score += float(causal.get("causal_strength_score", 0.0) or 0.0)
        diagnostic_score += float(subsystem.get("subsystem_score", 0.0) or 0.0)
        diagnostic_score += float(context_fit.get("context_fit_score", 0.0) or 0.0)

        if str(causal.get("causal_strength_band") or "") == "direct":
            diagnostic_score += ROOT_CAUSE_DIRECT_SIGNAL_BONUS
        if str(cc.get("source_type") or "") == "manual" and str(causal.get("causal_strength_band") or "") in {"direct", "indirect"}:
            diagnostic_score += 0.04
        if bool(context_fit.get("direct_mechanism_supported")) and bool(symptom_profile.get("generic_symptom")):
            diagnostic_score += 0.04
        if generic_downranked:
            diagnostic_score -= ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY
        if bool(context_fit.get("support_only_penalized")):
            diagnostic_score -= 0.06
        if hard_excluded:
            diagnostic_score -= ROOT_CAUSE_HARD_EXCLUDE_PENALTY

        cc.update(semantic)
        cc.update(causal)
        cc.update(subsystem)
        cc.update(context_fit)
        cc["generic_downranked"] = bool(generic_downranked)
        cc["hard_excluded"] = bool(hard_excluded)
        cc["base_retrieval_score"] = float(cc.get("retrieval_score", 0.0) or 0.0)
        cc["diagnostic_score"] = diagnostic_score
        cc["retrieval_score"] = diagnostic_score
        rescored_pool.append(cc)

    rescored_pool.sort(
        key=lambda x: (
            -float(x.get("diagnostic_score", x.get("retrieval_score", x.get("similarity", 0.0))) or 0.0),
            -float(x.get("similarity", 0.0) or 0.0),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )

    non_excluded = [c for c in rescored_pool if not bool(c.get("hard_excluded"))]
    working_pool = non_excluded if len(non_excluded) >= max(top_k, 4) else rescored_pool
    working_pool = _dedup_root_cause_candidates_semantic(
        working_pool,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 2, top_k * 3, 14),
    )
    working_pool = _prioritize_root_cause_coverage(
        working_pool,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
    )

    llm_priority_ids: list[str] = []
    if len(working_pool) >= 4:
        try:
            llm_priority_ids = _llm_filter_diagnostic_chunks(
                q=q,
                candidates=working_pool,
                max_keep=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
            )
        except Exception:
            llm_priority_ids = []

    if llm_priority_ids:
        working_pool = _reorder_citations_by_priority_ids(
            working_pool,
            llm_priority_ids,
            max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
        )

    diagnostic_matrix: dict = {}
    try:
        diagnostic_matrix = _llm_build_diagnostic_evidence_matrix(
            q=q,
            citations=working_pool[: max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2)],
            max_causes=max_causes,
        )
    except Exception:
        diagnostic_matrix = {}

    matrix_keep_ids = _unique_non_empty_strings((diagnostic_matrix or {}).get("keep_ids") or [], limit=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2))
    if matrix_keep_ids:
        working_pool = _reorder_citations_by_priority_ids(
            working_pool,
            matrix_keep_ids,
            max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 2),
        )

    prompt_citations = _select_prompt_citations_from_matrix(
        rescored_candidates=working_pool,
        diagnostic_matrix=diagnostic_matrix or {},
        max_prompt=max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k),
    )
    if not prompt_citations:
        prompt_citations = _lock_final_citations(
            selected_citations=working_pool[: max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k)],
            ranked_candidates=working_pool + rescored_pool,
            top_k=max(ROOT_CAUSE_MAX_PROMPT_CITATIONS, top_k),
            diagnostic_mode=True,
            query_token_count=query_token_count,
        )

    final_citations = _lock_final_citations(
        selected_citations=prompt_citations,
        ranked_candidates=working_pool + rescored_pool,
        top_k=top_k,
        diagnostic_mode=True,
        query_token_count=query_token_count,
    )

    result = dict(base_retrieval)
    result["citations"] = _dedup_citations_by_snippet(
        list(prompt_citations or []) + list(final_citations or []),
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 3),
    )
    result["prompt_citations"] = prompt_citations
    result["candidates"] = rescored_pool
    result["candidate_pool"] = working_pool
    result["diagnostic_matrix"] = diagnostic_matrix or {}
    result["diagnostic_queries"] = diagnostic_queries
    result["diagnostic_keywords"] = diagnostic_keywords
    result["inferred_components"] = inferred_components
    result["target_subsystems"] = target_subsystems
    result["llm_priority_ids"] = llm_priority_ids
    result["extra_dense_queries"] = extra_dense_queries
    result["symptom_profile"] = symptom_profile
    return result


@dataclass(frozen=True)
class DiagnosticEvidenceCandidatePipelineRuntime:
    ROOT_CAUSE_CANDIDATE_CORE_PROMOTION: Any
    ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX: Any
    ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K: Any
    ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY: Any
    ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K: Any
    ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY: Any
    ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY: Any
    ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY: Any
    ROOT_CAUSE_MAX_EVIDENCE_POOL: Any
    _classify_diagnostic_role_from_text: Callable[..., Any]
    _collect_candidate_keywords: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _dedup_root_cause_candidates_semantic: Callable[..., Any]
    _diagnostic_evidence_pipeline: Callable[..., Any]
    _ensure_candidate_retrieval_fields: Callable[..., Any]
    _infer_machine_components: Callable[..., Any]
    _llm_build_role_aware_diagnostic_evidence_matrix: Callable[..., Any]
    _lock_final_citations: Callable[..., Any]
    _planner_query_term_set: Callable[..., Any]
    _prioritize_root_cause_coverage: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]
    _root_cause_target_subsystems: Callable[..., Any]
    _select_prompt_citations_from_matrix: Callable[..., Any]
    _summarize_evidence_roles_for_prompt: Callable[..., Any]


def diagnostic_evidence_candidate_pipeline(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    candidate_k: int,
    top_k: int,
    max_causes: int,
    doc_ids: Optional[list[str]] = None,
    bubble_document_id: Optional[str] = None,
    debug: bool = False,
    planner_mode: str = "root_cause",
    base_threshold: float,
    runtime: DiagnosticEvidenceCandidatePipelineRuntime,
) -> dict:
    ROOT_CAUSE_CANDIDATE_CORE_PROMOTION = runtime.ROOT_CAUSE_CANDIDATE_CORE_PROMOTION
    ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX = runtime.ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX
    ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K = runtime.ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K
    ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY = runtime.ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY
    ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K = runtime.ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K
    ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY = runtime.ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY
    ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY = runtime.ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY
    ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY = runtime.ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY
    ROOT_CAUSE_MAX_EVIDENCE_POOL = runtime.ROOT_CAUSE_MAX_EVIDENCE_POOL
    _classify_diagnostic_role_from_text = runtime._classify_diagnostic_role_from_text
    _collect_candidate_keywords = runtime._collect_candidate_keywords
    _count_query_tokens = runtime._count_query_tokens
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _dedup_root_cause_candidates_semantic = runtime._dedup_root_cause_candidates_semantic
    _diagnostic_evidence_pipeline = runtime._diagnostic_evidence_pipeline
    _ensure_candidate_retrieval_fields = runtime._ensure_candidate_retrieval_fields
    _infer_machine_components = runtime._infer_machine_components
    _llm_build_role_aware_diagnostic_evidence_matrix = runtime._llm_build_role_aware_diagnostic_evidence_matrix
    _lock_final_citations = runtime._lock_final_citations
    _planner_query_term_set = runtime._planner_query_term_set
    _prioritize_root_cause_coverage = runtime._prioritize_root_cause_coverage
    _query_symptom_profile = runtime._query_symptom_profile
    _root_cause_target_subsystems = runtime._root_cause_target_subsystems
    _select_prompt_citations_from_matrix = runtime._select_prompt_citations_from_matrix
    _summarize_evidence_roles_for_prompt = runtime._summarize_evidence_roles_for_prompt
    result = _diagnostic_evidence_pipeline(
        q=q,
        company_id=company_id,
        machine_id=machine_id,
        candidate_k=candidate_k,
        top_k=top_k,
        max_causes=max_causes,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
        debug=debug,
        planner_mode=planner_mode,
        base_threshold=base_threshold,
    )

    symptom_profile = dict(result.get("symptom_profile") or _query_symptom_profile(q))
    inferred_components = list(result.get("inferred_components") or _infer_machine_components(q))
    diagnostic_keywords = list(result.get("diagnostic_keywords") or _collect_candidate_keywords(q, inferred_components))
    target_subsystems = list(result.get("target_subsystems") or _root_cause_target_subsystems(q, inferred_components))
    classes = set(symptom_profile.get("classes") or [])

    raw_pool = list(result.get("candidate_pool") or [])
    if not raw_pool:
        raw_pool = list(result.get("candidates") or []) + list(result.get("citations") or [])
    raw_pool = _ensure_candidate_retrieval_fields(
        raw_pool,
        query_terms=_planner_query_term_set(q, result.get("planner") or {}),
        query_style=str((result.get("planner") or {}).get("query_style") or "").strip().lower(),
        query_token_count=_count_query_tokens(q),
    )
    if not raw_pool:
        result["role_summary"] = []
        return result

    rescored = []
    for item in raw_pool:
        cc = dict(item)
        chunk_text = (cc.get("chunk_full") or cc.get("snippet") or "").strip()
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=chunk_text,
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        candidate_score = float(cc.get("diagnostic_score", cc.get("retrieval_score", cc.get("similarity", 0.0))) or 0.0)
        candidate_score += float(role.get("role_adjustment", 0.0) or 0.0)

        if symptom_profile.get("generic_symptom") and role.get("role_group") == "core":
            candidate_score += ROOT_CAUSE_CANDIDATE_CORE_PROMOTION
        if symptom_profile.get("generic_symptom") and role.get("role_group") == "support" and not symptom_profile.get("has_support_anchor") and "no_start" not in classes:
            candidate_score -= ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY * 0.45
        if role.get("role_class") == "support_lubrication" and classes & {"vibration", "noise", "jam"} and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY
        if role.get("role_class") == "support_lubrication" and "no_start" in classes and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY
        if role.get("role_class") == "support_startup_install" and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY
        if role.get("role_class") == "support_safety" and not ("no_start" in classes and symptom_profile.get("automatic_mode")) and not symptom_profile.get("has_support_anchor"):
            candidate_score -= ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY
        if role.get("role_class") == "collateral":
            candidate_score -= 0.04

        cc.update(role)
        cc["candidate_score"] = candidate_score
        cc["retrieval_score"] = candidate_score
        rescored.append(cc)

    rescored.sort(
        key=lambda x: (
            -float(x.get("candidate_score", x.get("diagnostic_score", x.get("retrieval_score", x.get("similarity", 0.0)))) or 0.0),
            -float(x.get("similarity", 0.0) or 0.0),
            str(x.get("bubble_document_id") or ""),
            int(x.get("page_from") or 0),
            int(x.get("page_to") or 0),
            int(x.get("chunk_index") or 0),
            str(x.get("citation_id") or ""),
        )
    )

    working_pool = _dedup_root_cause_candidates_semantic(
        rescored,
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL * 2, top_k * 3, ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K + 4),
    )
    working_pool = _prioritize_root_cause_coverage(
        working_pool,
        max_items=max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, top_k + 3),
    )

    matrix = {}
    if ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX:
        try:
            matrix = _llm_build_role_aware_diagnostic_evidence_matrix(
                q=q,
                citations=working_pool[: max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, top_k + 2)],
                max_causes=max_causes,
            )
        except Exception:
            matrix = {}

    if not matrix:
        matrix = dict(result.get("diagnostic_matrix") or {})

    prompt_citations = _select_prompt_citations_from_matrix(
        working_pool,
        matrix,
        max_prompt=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
    )
    if not prompt_citations:
        prompt_citations = _lock_final_citations(
            selected_citations=working_pool[: max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k)],
            ranked_candidates=working_pool + rescored,
            top_k=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
            diagnostic_mode=True,
            query_token_count=_count_query_tokens(q),
        )

    final_citations = _lock_final_citations(
        selected_citations=prompt_citations,
        ranked_candidates=working_pool + rescored,
        top_k=max(top_k, min(len(prompt_citations) or top_k, top_k + 2)),
        diagnostic_mode=True,
        query_token_count=_count_query_tokens(q),
    )

    result["candidates"] = rescored
    result["candidate_pool"] = working_pool
    result["prompt_citations"] = prompt_citations
    result["citations"] = _dedup_citations_by_snippet(
        list(prompt_citations or []) + list(final_citations or []),
        max_items=max(ROOT_CAUSE_MAX_EVIDENCE_POOL, top_k + 3),
    )
    result["diagnostic_matrix"] = matrix or {}
    result["role_summary"] = _summarize_evidence_roles_for_prompt(
        q=q,
        citations=prompt_citations or working_pool,
        symptom_profile=symptom_profile,
        diagnostic_keywords=diagnostic_keywords,
        target_subsystems=target_subsystems,
        max_items=max(ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K, top_k),
    )
    return result


