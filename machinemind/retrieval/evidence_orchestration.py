"""P4-C1: unchanged evidence orchestration behind explicit runtime dependencies.

This is an extraction, not a new evidence policy. The composition root supplies
retrieval, policy, budget and presentation callbacks at call time. This module
has no application, database, provider or web-framework import. Legacy decisions,
ordering, exception handling and budget checks are deliberately preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class V13PlanRetrievalRuntime:
    V13_DENSE_QUERY_LIMIT: Any
    V13_LEXICAL_QUERY_LIMIT: Any
    V13_MIN_SECONDS_FOR_REFINEMENT: Any
    V13_PLANNER_MAX_OUTPUT_TOKENS: Any
    V13_PLANNER_MODEL: Any
    V13_PLANNER_TIMEOUT_SECONDS: Any
    _dedup_text_values: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    _v13_fallback_plan: Callable[..., Any]
    _v13_json_models: Callable[..., Any]
    _v13_query_plan_schema: Callable[..., Any]
    re: Any


def v13_plan_retrieval(*, q: str, mode: str, company_id: str, runtime: V13PlanRetrievalRuntime) -> dict:
    V13_DENSE_QUERY_LIMIT = runtime.V13_DENSE_QUERY_LIMIT
    V13_LEXICAL_QUERY_LIMIT = runtime.V13_LEXICAL_QUERY_LIMIT
    V13_MIN_SECONDS_FOR_REFINEMENT = runtime.V13_MIN_SECONDS_FOR_REFINEMENT
    V13_PLANNER_MAX_OUTPUT_TOKENS = runtime.V13_PLANNER_MAX_OUTPUT_TOKENS
    V13_PLANNER_MODEL = runtime.V13_PLANNER_MODEL
    V13_PLANNER_TIMEOUT_SECONDS = runtime.V13_PLANNER_TIMEOUT_SECONDS
    _dedup_text_values = runtime._dedup_text_values
    _v13_current_budget = runtime._v13_current_budget
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_json_models = runtime._v13_json_models
    _v13_query_plan_schema = runtime._v13_query_plan_schema
    re = runtime.re
    budget = _v13_current_budget()
    if budget is None or budget.remaining() < V13_MIN_SECONDS_FOR_REFINEMENT:
        return _v13_fallback_plan(q)

    system_msg = (
        "Prepare one compact retrieval plan for an industrial evidence assistant. "
        "Do not answer and do not propose causes. Preserve the exact meaning, codes, numbers, negations, timing, process state and requested source type. "
        "Generate only a few high-value semantic and lexical rewrites in Italian and English when useful. "
        "Do not inject unsupported components, sectors, alarms or failure modes."
    )
    user_msg = f"MODE: {mode}\nQUERY:\n{q}\n\nReturn a retrieval plan only."

    try:
        parsed, _model = _v13_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[V13_PLANNER_MODEL],
            json_schema=_v13_query_plan_schema(),
            effort="low",
            reasoning_mode="",
            timeout=V13_PLANNER_TIMEOUT_SECONDS,
            max_output_tokens=V13_PLANNER_MAX_OUTPUT_TOKENS,
            company_id=company_id,
            purpose=f"{mode}_retrieval_refinement",
        )
    except Exception as exc:
        print("V13_RETRIEVAL_PLAN_FAIL", str(exc)[:700])
        return _v13_fallback_plan(q)

    fallback = _v13_fallback_plan(q)
    parsed = dict(parsed or {})
    parsed["normalized_query"] = re.sub(
        r"\s+", " ", str(parsed.get("normalized_query") or fallback["normalized_query"])
    ).strip()
    parsed["dense_queries"] = _dedup_text_values(
        [q, parsed.get("normalized_query")] + list(parsed.get("dense_queries") or []),
        limit=V13_DENSE_QUERY_LIMIT + 1,
    )
    parsed["lexical_queries"] = _dedup_text_values(
        [q, parsed.get("normalized_query")] + list(parsed.get("lexical_queries") or []),
        limit=V13_LEXICAL_QUERY_LIMIT + 1,
    )
    parsed["exact_terms"] = _dedup_text_values(
        list(parsed.get("exact_terms") or []) + fallback["exact_terms"],
        limit=18,
    )
    parsed["required_facets"] = _dedup_text_values(
        list(parsed.get("required_facets") or []) + fallback["required_facets"],
        limit=12,
    )
    if budget is not None:
        budget.refinement_used = True
    return parsed


@dataclass(frozen=True)
class V13InitialRetrievalRuntime:
    V13_DENSE_QUERY_LIMIT: Any
    V13_LEXICAL_QUERY_LIMIT: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE: Any
    _ask_source_preference_profile: Callable[..., Any]
    _ask_structured_direct_fetch_sources: Callable[..., Any]
    _count_query_tokens: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _fetch_dense_chunk_candidates: Callable[..., Any]
    _fts_search_chunks_multi: Callable[..., Any]
    _fts_search_chunks_prefix: Callable[..., Any]
    _openai_embed_texts: Callable[..., Any]
    _raw_rows_to_dense_candidates: Callable[..., Any]
    _rrf_merge_candidates: Callable[..., Any]
    _structured_rescue_query_intent: Callable[..., Any]
    _v13_build_profile_from_plan: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    _v13_evidence_metrics: Callable[..., Any]
    _v13_exact_identifier_candidates: Callable[..., Any]
    _v13_fallback_plan: Callable[..., Any]
    _v13_fetch_preferred_source_pages: Callable[..., Any]
    _v13_fetch_scored_pages: Callable[..., Any]
    _v13_fetch_structured_dense_candidates: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]
    _v13_rescore_root_candidates: Callable[..., Any]
    _v13_score_candidates: Callable[..., Any]
    _vector_literal: Callable[..., Any]


def v13_initial_retrieval(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], ai_scope: str, response_language: str, mode: str, plan: Optional[dict]=None, runtime: V13InitialRetrievalRuntime, lineage: Optional[Callable[..., None]]=None) -> dict:
    V13_DENSE_QUERY_LIMIT = runtime.V13_DENSE_QUERY_LIMIT
    V13_LEXICAL_QUERY_LIMIT = runtime.V13_LEXICAL_QUERY_LIMIT
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
    _ask_source_preference_profile = runtime._ask_source_preference_profile
    _ask_structured_direct_fetch_sources = runtime._ask_structured_direct_fetch_sources
    _count_query_tokens = runtime._count_query_tokens
    _dedup_text_values = runtime._dedup_text_values
    _fetch_dense_chunk_candidates = runtime._fetch_dense_chunk_candidates
    _fts_search_chunks_multi = runtime._fts_search_chunks_multi
    _fts_search_chunks_prefix = runtime._fts_search_chunks_prefix
    _openai_embed_texts = runtime._openai_embed_texts
    _raw_rows_to_dense_candidates = runtime._raw_rows_to_dense_candidates
    _rrf_merge_candidates = runtime._rrf_merge_candidates
    _structured_rescue_query_intent = runtime._structured_rescue_query_intent
    _v13_build_profile_from_plan = runtime._v13_build_profile_from_plan
    _v13_current_budget = runtime._v13_current_budget
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_exact_identifier_candidates = runtime._v13_exact_identifier_candidates
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_fetch_preferred_source_pages = runtime._v13_fetch_preferred_source_pages
    _v13_fetch_scored_pages = runtime._v13_fetch_scored_pages
    _v13_fetch_structured_dense_candidates = runtime._v13_fetch_structured_dense_candidates
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_rescore_root_candidates = runtime._v13_rescore_root_candidates
    _v13_score_candidates = runtime._v13_score_candidates
    _vector_literal = runtime._vector_literal
    budget = _v13_current_budget()
    if budget is not None:
        budget.ensure_time(8.0)

    plan = dict(plan or _v13_fallback_plan(q))
    dense_queries = _dedup_text_values(
        [q, plan.get("normalized_query")] + list(plan.get("dense_queries") or []),
        limit=V13_DENSE_QUERY_LIMIT if plan.get("dense_queries") else 2,
    )
    lexical_queries = _dedup_text_values(
        [q, plan.get("normalized_query")] + list(plan.get("lexical_queries") or []),
        limit=V13_LEXICAL_QUERY_LIMIT if plan.get("lexical_queries") else 2,
    )

    candidate_k = 52 if mode in {"root_cause", "neutral"} else 34
    dense_lists: list[list[dict]] = []
    chunks_matching_filter = None

    try:
        vectors = _openai_embed_texts(dense_queries, timeout=12)
    except Exception as exc:
        print("V13_DENSE_EMBED_FAIL", str(exc)[:600])
        vectors = []

    for query_text, vector in zip(dense_queries, vectors):
        try:
            current_count, raw_rows = _fetch_dense_chunk_candidates(
                company_id=company_id,
                machine_id=machine_id,
                q_vec_lit=_vector_literal(vector),
                candidate_k=candidate_k,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
                debug=False,
            )
            if chunks_matching_filter is None:
                chunks_matching_filter = current_count
            dense_lists.append(_raw_rows_to_dense_candidates(raw_rows, query_used=query_text))
        except Exception as exc:
            print("V13_DENSE_FETCH_FAIL", str(exc)[:500])

    dense_candidates = _rrf_merge_candidates(dense_lists, k=60) if dense_lists else []

    prefix_hits: list[dict] = []
    exact_hits: list[dict] = []
    try:
        prefix_hits = _fts_search_chunks_prefix(
            company_id=company_id,
            machine_id=machine_id,
            texts=lexical_queries,
            top_k=14,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if lineage is not None:
            lineage("prefix_before", tuple(prefix_hits))
        for c in prefix_hits:
            c["fts_v13"] = True
        if lineage is not None:
            lineage("prefix_after", tuple(prefix_hits))
    except Exception as exc:
        print("V13_PREFIX_FTS_FAIL", str(exc)[:500])

    try:
        exact_hits = _fts_search_chunks_multi(
            company_id=company_id,
            machine_id=machine_id,
            queries=lexical_queries,
            top_k=14,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
        )
        if lineage is not None:
            lineage("lexical_before", tuple(exact_hits))
        for c in exact_hits:
            c["fts_v13"] = True
        if lineage is not None:
            lineage("lexical_after", tuple(exact_hits))
    except Exception as exc:
        print("V13_EXACT_FTS_FAIL", str(exc)[:500])

    identifier_hits: list[dict] = []
    try:
        identifier_hits = _v13_exact_identifier_candidates(
            q=q, company_id=company_id, machine_id=machine_id,
            doc_ids=doc_ids, bubble_document_id=bubble_document_id,
        )
    except Exception as exc:
        print("V13_IDENTIFIER_RETRIEVAL_FAIL", str(exc)[:500])

    page_hits: list[dict] = []
    source_profile = _ask_source_preference_profile(q)
    should_scan_pages = bool(
        doc_ids
        or bubble_document_id
        or str(source_profile.get("strength") or "none") != "none"
        or _count_query_tokens(q) >= 5
        or mode in {"root_cause", "neutral"}
    )
    if should_scan_pages:
        try:
            profile = _v13_build_profile_from_plan(q, response_language, plan)
            page_hits = _v13_fetch_scored_pages(
                q=q,
                profile=profile,
                company_id=company_id,
                machine_id=machine_id,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
                top_pages=10 if mode in {"root_cause", "neutral"} else 8,
            )
        except Exception as exc:
            print("V13_PAGE_SCAN_FAIL", str(exc)[:600])

    preferred_hits: list[dict] = []
    preferred = str(source_profile.get("preferred_source") or "").strip().lower()
    strength = str(source_profile.get("strength") or "none").strip().lower()
    if mode in {"ask", "neutral"} and preferred in {"manual", "xlsx"} and strength in {"prefer", "hard"}:
        try:
            preferred_hits = _v13_fetch_preferred_source_pages(
                q=q,
                company_id=company_id,
                machine_id=machine_id,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
                response_language=response_language,
                top_k=8,
                plan=plan,
                source_kind=preferred,
            )
        except Exception as exc:
            print("V13_PREFERRED_SOURCE_FETCH_FAIL", str(exc)[:500])

    structured_semantic_hits: list[dict] = []
    structured_direct_hits: list[dict] = []
    if mode in {"ask", "neutral"} and ai_scope == "machine_all" and not doc_ids and not bubble_document_id:
        try:
            structured_semantic_hits = _v13_fetch_structured_dense_candidates(
                company_id=company_id,
                machine_id=machine_id,
                query_vectors=list(zip(dense_queries, vectors)),
                top_k=18,
            )
        except Exception as exc:
            print("V13_STRUCTURED_SEMANTIC_RETRIEVAL_FAIL", str(exc)[:600])

        # Source/listing wording is only a supplement. Normal procedure/P&S routing is
        # decided later from semantic evidence, not from operation-specific keywords.
        if _structured_rescue_query_intent(q, plan):
            try:
                structured_direct_hits = _ask_structured_direct_fetch_sources(
                    company_id=company_id,
                    machine_id=machine_id,
                    q=q,
                    planner=plan,
                    top_k=12,
                )
            except Exception as exc:
                print("V13_STRUCTURED_DIRECT_RETRIEVAL_FAIL", str(exc)[:600])

    merged = _v13_merge_candidates(
        [identifier_hits, preferred_hits, structured_direct_hits, structured_semantic_hits, page_hits, dense_candidates, prefix_hits, exact_hits]
    )
    scored = _v13_score_candidates(q, merged)
    if mode == "root_cause":
        scored = _v13_rescore_root_candidates(q, scored)

    metrics = _v13_evidence_metrics(scored)
    if lineage is not None:
        lineage("complete", tuple(scored))
    return {
        "plan": plan,
        "candidates": scored,
        "citations": scored[: (
            V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE if mode == "root_cause" else V13_MAX_EVIDENCE_ITEMS_ASK
        )],
        "metrics": metrics,
        "chunks_matching_filter": chunks_matching_filter,
        "source_profile": source_profile,
        "dense_queries": dense_queries,
        "lexical_queries": lexical_queries,
    }


@dataclass(frozen=True)
class V13ResolveEvidenceSupportRuntime:
    V13_EVIDENCE_GATE_MODEL: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE: Any
    _v13_current_budget: Callable[..., Any]
    _v13_deterministic_evidence_state: Callable[..., Any]
    _v13_evidence_metrics: Callable[..., Any]
    _v13_fallback_plan: Callable[..., Any]
    _v13_filter_retrieval_candidates: Callable[..., Any]
    _v13_initial_retrieval: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]
    _v13_plan_from_evidence_gate: Callable[..., Any]
    _v13_pre_admission_retrieval_assurance: Callable[..., Any]
    _v13_rescore_root_candidates: Callable[..., Any]
    _v13_score_candidates: Callable[..., Any]
    _v13_semantic_evidence_gate: Callable[..., Any]


def v13_resolve_evidence_support(*, q: str, company_id: str, machine_id: str, doc_ids: Optional[list[str]], bubble_document_id: Optional[str], ai_scope: str, response_language: str, mode: str, narrow_scope: bool, initial_retrieval: dict, force_semantic_gate: bool=False, request_task_contract: bool=False, runtime: V13ResolveEvidenceSupportRuntime) -> tuple[bool, dict, dict]:
    V13_EVIDENCE_GATE_MODEL = runtime.V13_EVIDENCE_GATE_MODEL
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
    _v13_current_budget = runtime._v13_current_budget
    _v13_deterministic_evidence_state = runtime._v13_deterministic_evidence_state
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_filter_retrieval_candidates = runtime._v13_filter_retrieval_candidates
    _v13_initial_retrieval = runtime._v13_initial_retrieval
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_plan_from_evidence_gate = runtime._v13_plan_from_evidence_gate
    _v13_pre_admission_retrieval_assurance = runtime._v13_pre_admission_retrieval_assurance
    _v13_rescore_root_candidates = runtime._v13_rescore_root_candidates
    _v13_score_candidates = runtime._v13_score_candidates
    _v13_semantic_evidence_gate = runtime._v13_semantic_evidence_gate
    budget = _v13_current_budget()
    candidates = list((initial_retrieval or {}).get("candidates") or [])
    state, initial_signals = _v13_deterministic_evidence_state(q, candidates, mode=mode, narrow_scope=narrow_scope)
    initial_plan = dict((initial_retrieval or {}).get("plan") or _v13_fallback_plan(q))
    gate_meta = {
        "initial_state": state, "semantic_gate_used": False, "refinement_used": False,
        "decision": "unsupported", "reason_code": "evidence_irrelevant",
        "confidence": 1.0 if state != "borderline" else 0.0,
        "initial_top_similarity": round(float(initial_signals.get("top_similarity") or 0.0), 6),
        "initial_top_overlap": round(float(initial_signals.get("top_overlap") or 0.0), 6),
        "dense_queries": list(initial_plan.get("dense_queries") or []),
        "lexical_queries": list(initial_plan.get("lexical_queries") or []),
        "exact_terms": list(initial_plan.get("exact_terms") or []),
        "required_facets": list(initial_plan.get("required_facets") or []),
        "missing_information": [],
        "relevant_evidence_ids": [],
        "task_mode": "other",
        "task_confidence": 0.0,
        "result_cardinality": "few",
        "requires_explanation": True,
        "preferred_source_types": [],
        "source_type_policy": "none",
        "task_focus": "",
        "forced_semantic_gate": bool(force_semantic_gate),
        "task_contract_requested": bool(request_task_contract),
    }
    must_use_semantic_gate = bool(force_semantic_gate)
    if state == "unsupported" and not (must_use_semantic_gate and candidates):
        recovered, pre_meta = _v13_pre_admission_retrieval_assurance(
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            doc_ids=doc_ids,
            bubble_document_id=bubble_document_id,
            ai_scope=ai_scope,
            response_language=response_language,
            mode=mode,
            narrow_scope=narrow_scope,
            retrieval=initial_retrieval,
            signals=initial_signals,
        )
        gate_meta["pre_admission_assurance"] = dict(pre_meta or {})
        if bool((pre_meta or {}).get("adopted")):
            initial_retrieval = recovered
            candidates = list((recovered or {}).get("candidates") or [])
            state, rescued_signals = _v13_deterministic_evidence_state(
                q, candidates, mode=mode, narrow_scope=narrow_scope
            )
            gate_meta.update(
                {
                    "post_pre_assurance_state": state,
                    "post_pre_assurance_top_similarity": round(float(rescued_signals.get("top_similarity") or 0.0), 6),
                    "post_pre_assurance_top_overlap": round(float(rescued_signals.get("top_overlap") or 0.0), 6),
                }
            )
            # Deterministic rescue only recovered candidates; it never grants permission
            # to answer. A semantic gate is mandatory before synthesis.
            must_use_semantic_gate = True
            if budget is not None:
                budget.retrieval_assurance = {"pre_admission": dict(pre_meta or {})}
        else:
            if budget is not None:
                budget.evidence_gate = dict(gate_meta)
            return False, dict(initial_retrieval or {}), gate_meta
    elif state == "unsupported" and must_use_semantic_gate and candidates:
        # A high-confidence structured-title match is real source evidence even when
        # generic cosine/overlap bands remain low. It still requires the semantic gate;
        # this branch grants no permission to answer by itself.
        gate_meta["forced_gate_on_structured_title_match"] = True

    if state == "supported" and mode == "ask" and not must_use_semantic_gate:
        admitted = _v13_filter_retrieval_candidates(q, initial_retrieval, mode=mode)
        gate_meta.update({"decision": "supported", "reason_code": "evidence_sufficient", "confidence": 1.0, "selected_count": len(admitted.get("citations") or [])})
        if budget is not None:
            budget.evidence_gate = dict(gate_meta)
        return bool(admitted.get("citations")), admitted, gate_meta
    if state == "supported" and mode == "root_cause":
        # A close document match does not by itself prove that the input describes an
        # abnormal condition suitable for causal analysis. Root Cause therefore always
        # buys one semantic mode-fit gate before its single synthesis call.
        gate_meta["deterministic_support_requires_semantic_mode_fit"] = True
    try:
        semantic = _v13_semantic_evidence_gate(
            q=q, mode=mode, response_language=response_language, company_id=company_id,
            candidates=candidates, narrow_scope=narrow_scope,
            include_task_contract=bool(mode == "ask" and request_task_contract),
        )
    except Exception as exc:
        print("V13_EVIDENCE_GATE_FAIL_CLOSED", mode, str(exc)[:700])
        gate_meta.update({"semantic_gate_used": True, "decision": "unsupported", "reason_code": "evidence_incomplete", "confidence": 0.0})
        if budget is not None:
            budget.evidence_gate = dict(gate_meta)
        return False, dict(initial_retrieval or {}), gate_meta
    decision = str(semantic.get("decision") or "unsupported")
    gate_meta.update({
        "semantic_gate_used": True, "decision": decision,
        "reason_code": str(semantic.get("reason_code") or "evidence_irrelevant"),
        "confidence": round(float(semantic.get("confidence") or 0.0), 4),
        "model": str(semantic.get("model") or V13_EVIDENCE_GATE_MODEL),
        "relevant_evidence_count": len(semantic.get("relevant_evidence_ids") or []),
        "relevant_evidence_ids": list(semantic.get("relevant_evidence_ids") or []),
        "dense_queries": list(semantic.get("dense_queries") or []),
        "lexical_queries": list(semantic.get("lexical_queries") or []),
        "exact_terms": list(semantic.get("exact_terms") or []),
        "required_facets": list(semantic.get("required_facets") or []),
        "missing_information": list(semantic.get("missing_information") or []),
        "task_mode": str(semantic.get("task_mode") or "other"),
        "task_confidence": round(float(semantic.get("task_confidence") or 0.0), 4),
        "result_cardinality": str(semantic.get("result_cardinality") or "few"),
        "requires_explanation": bool(semantic.get("requires_explanation", True)),
        "preferred_source_types": list(semantic.get("preferred_source_types") or []),
        "source_type_policy": str(semantic.get("source_type_policy") or "none"),
        "task_focus": str(semantic.get("task_focus") or ""),
    })
    if decision == "supported":
        admitted = _v13_filter_retrieval_candidates(q, initial_retrieval, relevant_ids=list(semantic.get("relevant_evidence_ids") or []), mode=mode)
        gate_meta["selected_count"] = len(admitted.get("citations") or [])
        if budget is not None:
            budget.evidence_gate = dict(gate_meta)
        return bool(admitted.get("citations")), admitted, gate_meta
    if decision != "refine":
        if budget is not None:
            budget.evidence_gate = dict(gate_meta)
        return False, dict(initial_retrieval or {}), gate_meta
    if budget is not None:
        budget.refinement_used = True
    gate_meta["refinement_used"] = True
    refined_plan = _v13_plan_from_evidence_gate(q, semantic, dict((initial_retrieval or {}).get("plan") or _v13_fallback_plan(q)))
    refined = _v13_initial_retrieval(
        q=q, company_id=company_id, machine_id=machine_id, doc_ids=doc_ids,
        bubble_document_id=bubble_document_id, ai_scope=ai_scope,
        response_language=response_language, mode=mode, plan=refined_plan,
    )
    merged_candidates = _v13_merge_candidates([list((initial_retrieval or {}).get("candidates") or []), list((refined or {}).get("candidates") or [])])
    merged_candidates = _v13_score_candidates(q, merged_candidates)
    if mode == "root_cause":
        merged_candidates = _v13_rescore_root_candidates(q, merged_candidates)
    combined = {
        **dict(initial_retrieval or {}), **dict(refined or {}), "plan": refined_plan,
        "candidates": merged_candidates,
        "citations": merged_candidates[:(V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE if mode == "root_cause" else V13_MAX_EVIDENCE_ITEMS_ASK)],
        "metrics": _v13_evidence_metrics(merged_candidates),
    }
    post_state, post_signals = _v13_deterministic_evidence_state(q, merged_candidates, mode=mode, narrow_scope=narrow_scope)
    gate_meta.update({
        "post_refinement_state": post_state,
        "post_top_similarity": round(float(post_signals.get("top_similarity") or 0.0), 6),
        "post_top_overlap": round(float(post_signals.get("top_overlap") or 0.0), 6),
    })
    # The one semantic call requested a retrieval rewrite, not permission to answer.
    # After refinement the evidence must cross a clear support band; a still-borderline
    # result fails closed rather than being promoted automatically.
    if post_state != "supported":
        gate_meta["decision"] = "unsupported"
        gate_meta["reason_code"] = "evidence_incomplete"
        if budget is not None:
            budget.evidence_gate = dict(gate_meta)
        return False, combined, gate_meta
    admitted = _v13_filter_retrieval_candidates(q, combined, mode=mode)
    gate_meta.update({"decision": "supported_after_refinement", "reason_code": "evidence_sufficient", "selected_count": len(admitted.get("citations") or [])})
    if budget is not None:
        budget.evidence_gate = dict(gate_meta)
    return bool(admitted.get("citations")), admitted, gate_meta


@dataclass(frozen=True)
class AssistantCoreRetrieveNeutralRuntime:
    MODE_ROOT_CAUSE: Any
    _assistant_core_retrieval_query: Callable[..., Any]
    _assistant_core_scope_value: Callable[..., Any]
    _retrieval_diagnostic_query: Any
    _v13_fallback_plan: Callable[..., Any]
    _v13_fetch_structured_title_candidates: Callable[..., Any]
    _v13_initial_retrieval: Callable[..., Any]
    _v13_merge_source_title_candidates: Callable[..., Any]


def assistant_core_retrieve_neutral(request: AssistantCoreRequest, *, runtime: AssistantCoreRetrieveNeutralRuntime, lineage: Optional[Callable[..., Any]]=None) -> dict:
    MODE_ROOT_CAUSE = runtime.MODE_ROOT_CAUSE
    _assistant_core_retrieval_query = runtime._assistant_core_retrieval_query
    _assistant_core_scope_value = runtime._assistant_core_scope_value
    _retrieval_diagnostic_query = runtime._retrieval_diagnostic_query
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_fetch_structured_title_candidates = runtime._v13_fetch_structured_title_candidates
    _v13_initial_retrieval = runtime._v13_initial_retrieval
    _v13_merge_source_title_candidates = runtime._v13_merge_source_title_candidates
    is_root_cause = (
        str(request.requested_mode or "").strip().lower() == MODE_ROOT_CAUSE
    )
    profile = (
        _retrieval_diagnostic_query.analyze_diagnostic_query(
            request.query, response_language=request.response_language
        )
        if is_root_cause
        else None
    )
    retrieval_query = request.query if is_root_cause else _assistant_core_retrieval_query(request)
    doc_ids = _assistant_core_scope_value(request, "document_ids")
    bubble_document_id = _assistant_core_scope_value(request, "bubble_document_id")
    plan = _v13_fallback_plan(retrieval_query)
    retrieval = _v13_initial_retrieval(
        q=retrieval_query,
        company_id=request.company_id,
        machine_id=request.machine_id,
        doc_ids=doc_ids if isinstance(doc_ids, list) else None,
        bubble_document_id=str(bubble_document_id or "").strip() or None,
        ai_scope=request.ai_scope,
        response_language=request.response_language,
        mode="neutral",
        plan=plan,
    )
    # A bounded title/description probe improves source discovery without deciding
    # the task or replacing neutral evidence.
    try:
        title_candidates = _v13_fetch_structured_title_candidates(
            q=retrieval_query,
            company_id=request.company_id,
            machine_id=request.machine_id,
            ai_scope=request.ai_scope,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=str(bubble_document_id or "").strip() or None,
        )
        if title_candidates:
            retrieval = _v13_merge_source_title_candidates(
                retrieval_query, retrieval, title_candidates
            )
    except Exception as exc:
        print("ASSISTANT_CORE_TITLE_PROBE_FAIL", str(exc)[:500])
    if profile is not None:
        retrieval = {
            **dict(retrieval or {}),
            "assistant_core_discovery_query_state": profile.public_summary(),
        }
    if lineage is not None:
        lineage("complete", tuple(retrieval.get("candidates") or []), tuple(retrieval.get("citations") or []))
    return retrieval


@dataclass(frozen=True)
class AssistantCoreRefineRetrievalRuntime:
    ASSISTANT_CORE_MAX_FACETS: Any
    V13_DENSE_QUERY_LIMIT: Any
    V13_LEXICAL_QUERY_LIMIT: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE: Any
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_facet_candidate_confidence: Callable[..., Any]
    _assistant_core_merge_facet_candidates: Callable[..., Any]
    _assistant_core_retrieval_query: Callable[..., Any]
    _assistant_core_scope_value: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    _v13_evidence_metrics: Callable[..., Any]
    _v13_fallback_plan: Callable[..., Any]
    _v13_initial_retrieval: Callable[..., Any]
    _v13_score_candidates: Callable[..., Any]


def assistant_core_refine_retrieval(request: AssistantCoreRequest, retrieval: dict, decision: AssistantCoreDecision, *, runtime: AssistantCoreRefineRetrievalRuntime, lineage: Optional[Callable[..., Any]]=None) -> dict:
    ASSISTANT_CORE_MAX_FACETS = runtime.ASSISTANT_CORE_MAX_FACETS
    V13_DENSE_QUERY_LIMIT = runtime.V13_DENSE_QUERY_LIMIT
    V13_LEXICAL_QUERY_LIMIT = runtime.V13_LEXICAL_QUERY_LIMIT
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_facet_candidate_confidence = runtime._assistant_core_facet_candidate_confidence
    _assistant_core_merge_facet_candidates = runtime._assistant_core_merge_facet_candidates
    _assistant_core_retrieval_query = runtime._assistant_core_retrieval_query
    _assistant_core_scope_value = runtime._assistant_core_scope_value
    _dedup_text_values = runtime._dedup_text_values
    _v13_current_budget = runtime._v13_current_budget
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_initial_retrieval = runtime._v13_initial_retrieval
    _v13_score_candidates = runtime._v13_score_candidates
    retrieval_query = _assistant_core_retrieval_query(request)
    budget = _v13_current_budget()
    if budget is not None and budget.remaining() < 18.0:
        if lineage is not None:
            lineage("unchanged", tuple(retrieval.get("candidates") or []), tuple(retrieval.get("citations") or []))
        return retrieval
    if not (
        decision.dense_queries
        or decision.lexical_queries
        or decision.exact_terms
        or decision.required_facets
        or decision.facet_queries
    ):
        if lineage is not None:
            lineage("unchanged", tuple(retrieval.get("candidates") or []), tuple(retrieval.get("citations") or []))
        return retrieval

    doc_ids = _assistant_core_scope_value(request, "document_ids")
    bubble_document_id = _assistant_core_scope_value(request, "bubble_document_id")
    base_plan = dict(retrieval.get("plan") or _v13_fallback_plan(retrieval_query))
    candidate_lists: list[list[dict]] = [list(retrieval.get("candidates") or [])]
    facet_runs: list[dict] = []

    facet_queries = list(decision.facet_queries or [])[:ASSISTANT_CORE_MAX_FACETS]
    if facet_queries:
        for index, facet_query in enumerate(facet_queries, start=1):
            if budget is not None and budget.remaining() < 13.0:
                facet_runs.append({
                    "facet": facet_query.facet,
                    "status": "skipped_budget",
                })
                break
            facet = str(facet_query.facet or "").strip()
            if not facet:
                continue
            dense = _dedup_text_values(
                [facet] + list(facet_query.dense_queries or []),
                limit=min(V13_DENSE_QUERY_LIMIT, 4),
            )
            lexical = _dedup_text_values(
                [facet] + list(facet_query.lexical_queries or []),
                limit=min(V13_LEXICAL_QUERY_LIMIT, 6),
            )
            exact = _dedup_text_values(
                list(facet_query.exact_terms or []), limit=12
            )
            query_for_search = str((dense or lexical or [facet])[0]).strip() or facet
            facet_plan = {
                **_v13_fallback_plan(query_for_search),
                "intent": decision.request_kind,
                "information_task": decision.information_task,
                "required_answer_types": [facet_query.answer_type],
                "normalized_query": query_for_search,
                "dense_queries": list(dense),
                "lexical_queries": list(lexical),
                "exact_terms": list(exact),
                "required_facets": [facet],
                "ambiguities": [],
            }
            try:
                current = _v13_initial_retrieval(
                    q=query_for_search,
                    company_id=request.company_id,
                    machine_id=request.machine_id,
                    doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                    bubble_document_id=str(bubble_document_id or "").strip() or None,
                    ai_scope=request.ai_scope,
                    response_language=request.response_language,
                    mode="neutral",
                    plan=facet_plan,
                )
                current_candidates = _v13_score_candidates(
                    query_for_search,
                    [(dict(c) if lineage is None else lineage("copy", c, dict(c))) for c in (current.get("candidates") or []) if isinstance(c, dict)],
                )
                annotated: list[dict] = []
                for rank, raw in enumerate(current_candidates, start=1):
                    c = dict(raw)
                    if lineage is not None:
                        lineage("annotation_before", raw, c)
                    support = _assistant_core_facet_candidate_confidence(
                        candidate=c,
                        facet=facet,
                        answer_type=facet_query.answer_type,
                        dense_queries=dense,
                        lexical_queries=lexical,
                        exact_terms=exact,
                        preferred_source_types=facet_query.preferred_source_types,
                        rank=rank,
                    )
                    credible_hit = bool(support.get("credible"))
                    facet_score = float(support.get("score") or 0.0)
                    c["assistant_core_facet_hits"] = [facet] if credible_hit else []
                    c["assistant_core_facet_answer_types"] = [facet_query.answer_type] if credible_hit else []
                    c["assistant_core_facet_preferred_source_types"] = list(
                        facet_query.preferred_source_types or []
                    ) if credible_hit else []
                    c["assistant_core_facet_must_cover"] = [facet] if (facet_query.must_cover and credible_hit) else []
                    c["assistant_core_facet_retrieval_score"] = facet_score
                    c["assistant_core_facet_score_map"] = {facet: facet_score}
                    c["assistant_core_facet_support"] = dict(support)
                    if lineage is not None:
                        lineage("annotation_after", c)
                    annotated.append(c)
                candidate_lists.append(annotated)
                facet_runs.append(
                    {
                        "facet": facet,
                        "answer_type": facet_query.answer_type,
                        "must_cover": bool(facet_query.must_cover),
                        "dense_queries": list(dense),
                        "lexical_queries": list(lexical),
                        "exact_terms": list(exact),
                        "candidate_count": len(annotated),
                        "credible_candidate_count": sum(1 for c in annotated if c.get("assistant_core_facet_hits")),
                        "status": "completed",
                    }
                )
            except Exception as exc:
                print("ASSISTANT_CORE_FACET_RETRIEVAL_FAIL", facet[:180], str(exc)[:500])
                facet_runs.append(
                    {
                        "facet": facet,
                        "answer_type": facet_query.answer_type,
                        "must_cover": bool(facet_query.must_cover),
                        "status": "error",
                        "error": str(exc)[:300],
                    }
                )
    else:
        plan = {
            **base_plan,
            "intent": decision.request_kind,
            "information_task": decision.information_task,
            "required_answer_types": list(decision.required_answer_types),
            "dense_queries": _dedup_text_values(
                [retrieval_query] + list(decision.dense_queries) + list(decision.required_facets),
                limit=V13_DENSE_QUERY_LIMIT + 4,
            ),
            "lexical_queries": _dedup_text_values(
                [retrieval_query] + list(decision.lexical_queries) + list(decision.required_facets),
                limit=V13_LEXICAL_QUERY_LIMIT + 4,
            ),
            "exact_terms": _dedup_text_values(
                list(base_plan.get("exact_terms") or []) + list(decision.exact_terms), limit=18
            ),
            "required_facets": _dedup_text_values(list(decision.required_facets), limit=12),
            "ambiguities": _dedup_text_values(list(decision.missing_information), limit=8),
        }
        refined = _v13_initial_retrieval(
            q=retrieval_query,
            company_id=request.company_id,
            machine_id=request.machine_id,
            doc_ids=doc_ids if isinstance(doc_ids, list) else None,
            bubble_document_id=str(bubble_document_id or "").strip() or None,
            ai_scope=request.ai_scope,
            response_language=request.response_language,
            mode="neutral",
            plan=plan,
        )
        candidate_lists.append(list(refined.get("candidates") or []))
        facet_runs.append({"facet": "__combined__", "status": "completed"})

    merged = _assistant_core_merge_facet_candidates(candidate_lists)
    merged = _v13_score_candidates(retrieval_query, merged)
    if lineage is not None:
        lineage("bonus_before", tuple(merged))
    for c in merged:
        source_type = _assistant_core_candidate_source_type(c)
        preferred_for_facets = {
            str(x or "").strip().lower()
            for x in (c.get("assistant_core_facet_preferred_source_types") or [])
            if str(x or "").strip()
        }
        bonus = min(
            0.34,
            0.10 * len(c.get("assistant_core_facet_hits") or [])
            + 0.12 * float(c.get("assistant_core_facet_retrieval_score") or 0.0)
            + (0.08 if source_type in preferred_for_facets else 0.0),
        )
        c["assistant_core_facet_retrieval_bonus"] = bonus
        c["v13_score"] = float(c.get("v13_score") or 0.0) + bonus
        c["retrieval_score"] = max(
            float(c.get("retrieval_score") or 0.0), float(c.get("v13_score") or 0.0)
        )
    if lineage is not None:
        lineage("bonus_after", tuple(merged))
    merged.sort(
        key=lambda c: (
            -len(c.get("assistant_core_facet_must_cover") or []),
            -len(c.get("assistant_core_facet_hits") or []),
            -float(c.get("v13_score", c.get("retrieval_score", 0.0)) or 0.0),
            str(c.get("citation_id") or ""),
        )
    )
    plan = {
        **base_plan,
        "intent": decision.request_kind,
        "information_task": decision.information_task,
        "required_answer_types": list(decision.required_answer_types),
        "required_facets": list(decision.required_facets),
        "facet_queries": [
            {
                "facet": item.facet,
                "answer_type": item.answer_type,
                "must_cover": item.must_cover,
                "dense_queries": list(item.dense_queries),
                "lexical_queries": list(item.lexical_queries),
                "exact_terms": list(item.exact_terms),
                "preferred_source_types": list(item.preferred_source_types),
            }
            for item in decision.facet_queries
        ],
    }
    if lineage is not None:
        lineage("complete", tuple(merged), tuple(merged[: max(16, V13_MAX_EVIDENCE_ITEMS_ASK, V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE)]))
    return {
        **dict(retrieval or {}),
        "plan": plan,
        "candidates": merged,
        "citations": merged[: max(16, V13_MAX_EVIDENCE_ITEMS_ASK, V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE)],
        "metrics": _v13_evidence_metrics(merged),
        "assistant_core_refined": True,
        "assistant_core_facet_retrieval": facet_runs,
    }


@dataclass(frozen=True)
class AssistantCoreRootDiagnosticEvidenceAssuranceRuntime:
    ASSISTANT_CORE_MAX_FACETS: Any
    MODE_ROOT_CAUSE: Any
    V13_DENSE_QUERY_LIMIT: Any
    V13_LEXICAL_QUERY_LIMIT: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE: Any
    _assistant_core_candidate_stable_key: Callable[..., Any]
    _assistant_core_merge_facet_candidates: Callable[..., Any]
    _assistant_core_retrieval_query: Callable[..., Any]
    _assistant_core_scope_value: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    _v13_evidence_metrics: Callable[..., Any]
    _v13_fallback_plan: Callable[..., Any]
    _v13_fetch_structured_title_candidates: Callable[..., Any]
    _v13_initial_retrieval: Callable[..., Any]
    _v13_score_candidates: Callable[..., Any]


def assistant_core_root_diagnostic_evidence_assurance(request: AssistantCoreRequest, retrieval: dict, decision: AssistantCoreDecision, *, runtime: AssistantCoreRootDiagnosticEvidenceAssuranceRuntime) -> dict:
    """Monotonically supplement Root Cause evidence from explicit diagnostic clues.

    The semantic router may correctly classify a Root Cause request as already
    supported by broad manual pages. In that case the normal refinement stage can be
    skipped even though a component-specific datasheet, P&S, Procedure or Step would
    be much more discriminative. This bounded retrieval never removes the baseline
    pack and never decides the cause. It only runs another neutral cross-source search
    using the router's own discriminants, operating conditions, observables,
    subsystems and required facets, then merges any additional evidence before the
    existing Root Cause viability/ranking logic.
    """
    ASSISTANT_CORE_MAX_FACETS = runtime.ASSISTANT_CORE_MAX_FACETS
    MODE_ROOT_CAUSE = runtime.MODE_ROOT_CAUSE
    V13_DENSE_QUERY_LIMIT = runtime.V13_DENSE_QUERY_LIMIT
    V13_LEXICAL_QUERY_LIMIT = runtime.V13_LEXICAL_QUERY_LIMIT
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
    _assistant_core_candidate_stable_key = runtime._assistant_core_candidate_stable_key
    _assistant_core_merge_facet_candidates = runtime._assistant_core_merge_facet_candidates
    _assistant_core_retrieval_query = runtime._assistant_core_retrieval_query
    _assistant_core_scope_value = runtime._assistant_core_scope_value
    _dedup_text_values = runtime._dedup_text_values
    _v13_current_budget = runtime._v13_current_budget
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_fallback_plan = runtime._v13_fallback_plan
    _v13_fetch_structured_title_candidates = runtime._v13_fetch_structured_title_candidates
    _v13_initial_retrieval = runtime._v13_initial_retrieval
    _v13_score_candidates = runtime._v13_score_candidates
    if str(decision.effective_mode or "").strip().lower() != MODE_ROOT_CAUSE:
        return retrieval

    retrieval_query = _assistant_core_retrieval_query(request)

    budget = _v13_current_budget()
    if budget is not None and budget.remaining() < 12.0:
        return {
            **dict(retrieval or {}),
            "assistant_core_root_diagnostic_assurance": {
                "enabled": True,
                "status": "skipped_budget",
                "remaining_seconds": round(float(budget.remaining()), 3),
            },
        }

    facet_queries = list(decision.facet_queries or [])[:ASSISTANT_CORE_MAX_FACETS]
    facet_names = [
        str(item.facet or "").strip()
        for item in facet_queries
        if str(item.facet or "").strip()
    ]
    facet_dense = [
        str(query or "").strip()
        for item in facet_queries
        for query in list(item.dense_queries or [])[:2]
        if str(query or "").strip()
    ]
    facet_lexical = [
        str(query or "").strip()
        for item in facet_queries
        for query in list(item.lexical_queries or [])[:3]
        if str(query or "").strip()
    ]
    facet_exact = [
        str(term or "").strip()
        for item in facet_queries
        for term in list(item.exact_terms or [])[:5]
        if str(term or "").strip()
    ]

    diagnostic_clues = _dedup_text_values(
        list(decision.diagnostic_discriminants or [])
        + list(decision.diagnostic_operating_conditions or [])
        + list(decision.diagnostic_observables or [])
        + list(decision.diagnostic_subsystems or []),
        limit=16,
    )
    required_facets = _dedup_text_values(
        facet_names
        + list(decision.required_facets or [])
        + list(decision.diagnostic_discriminants or []),
        limit=16,
    )
    dense_queries = _dedup_text_values(
        [retrieval_query]
        + facet_dense
        + list(decision.dense_queries or [])
        + diagnostic_clues
        + required_facets,
        limit=max(V13_DENSE_QUERY_LIMIT, 8),
    )
    lexical_queries = _dedup_text_values(
        [retrieval_query]
        + facet_lexical
        + list(decision.lexical_queries or [])
        + diagnostic_clues
        + required_facets,
        limit=max(V13_LEXICAL_QUERY_LIMIT, 10),
    )
    exact_terms = _dedup_text_values(
        facet_exact + list(decision.exact_terms or []),
        limit=24,
    )

    if not dense_queries and not lexical_queries:
        return retrieval

    base_plan = dict(
        (retrieval or {}).get("plan") or _v13_fallback_plan(retrieval_query)
    )
    plan = {
        **base_plan,
        "intent": decision.request_kind,
        "information_task": decision.information_task,
        "required_answer_types": list(decision.required_answer_types or []),
        "normalized_query": retrieval_query,
        "dense_queries": dense_queries,
        "lexical_queries": lexical_queries,
        "exact_terms": exact_terms,
        "required_facets": required_facets,
        "ambiguities": _dedup_text_values(
            list(decision.missing_information or []),
            limit=8,
        ),
    }

    scoped_doc_ids = _assistant_core_scope_value(request, "document_ids")
    doc_ids = (
        list(scoped_doc_ids or [])
        if isinstance(scoped_doc_ids, list)
        else []
    )
    bubble_document_id = str(
        _assistant_core_scope_value(request, "bubble_document_id") or ""
    ).strip() or None

    try:
        assured = _v13_initial_retrieval(
            q=retrieval_query,
            company_id=request.company_id,
            machine_id=request.machine_id,
            doc_ids=doc_ids if doc_ids else None,
            bubble_document_id=bubble_document_id,
            ai_scope=request.ai_scope,
            response_language=request.response_language,
            mode="neutral",
            plan=plan,
        )
        assured_candidates = [
            dict(candidate)
            for candidate in (assured.get("candidates") or [])
            if isinstance(candidate, dict)
        ]

        # A title/description probe is especially useful for component-specific
        # datasheets and structured maintenance records whose body text may be short.
        try:
            title_query = " ".join(
                _dedup_text_values(
                    [retrieval_query] + diagnostic_clues + required_facets,
                    limit=12,
                )
            )[:1800]
            title_candidates = _v13_fetch_structured_title_candidates(
                q=title_query or retrieval_query,
                company_id=request.company_id,
                machine_id=request.machine_id,
                ai_scope=request.ai_scope,
                doc_ids=doc_ids if doc_ids else None,
                bubble_document_id=bubble_document_id,
            )
            assured_candidates = _assistant_core_merge_facet_candidates(
                [assured_candidates, list(title_candidates or [])]
            )
        except Exception as exc:
            print(
                "ASSISTANT_CORE_ROOT_ASSURANCE_TITLE_FAIL",
                str(exc)[:500],
            )

        assured_keys = {
            _assistant_core_candidate_stable_key(candidate)
            for candidate in assured_candidates
            if _assistant_core_candidate_stable_key(candidate)
        }
        baseline_candidates = [
            dict(candidate)
            for candidate in (
                (retrieval or {}).get("candidates")
                or (retrieval or {}).get("citations")
                or []
            )
            if isinstance(candidate, dict)
        ]
        merged = _assistant_core_merge_facet_candidates(
            [baseline_candidates, assured_candidates]
        )
        merged = _v13_score_candidates(retrieval_query, merged)
        for candidate in merged:
            if _assistant_core_candidate_stable_key(candidate) in assured_keys:
                candidate["assistant_core_diagnostic_assurance"] = True
                candidate[
                    "assistant_core_diagnostic_assurance_query_count"
                ] = len(dense_queries) + len(lexical_queries)

        return {
            **dict(retrieval or {}),
            "plan": plan,
            "candidates": merged,
            "citations": merged[
                : max(
                    18,
                    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE,
                    V13_MAX_EVIDENCE_ITEMS_ASK,
                )
            ],
            "metrics": _v13_evidence_metrics(merged),
            "assistant_core_root_diagnostic_assurance": {
                "enabled": True,
                "status": "completed",
                "baseline_candidate_count": len(baseline_candidates),
                "assurance_candidate_count": len(assured_candidates),
                "merged_candidate_count": len(merged),
                "dense_queries": dense_queries,
                "lexical_queries": lexical_queries,
                "exact_terms": exact_terms,
                "required_facets": required_facets,
            },
        }
    except Exception as exc:
        print(
            "ASSISTANT_CORE_ROOT_DIAGNOSTIC_ASSURANCE_FAIL",
            str(exc)[:700],
        )
        return {
            **dict(retrieval or {}),
            "assistant_core_root_diagnostic_assurance": {
                "enabled": True,
                "status": "error",
                "error": str(exc)[:300],
            },
        }


@dataclass(frozen=True)
class AssistantCorePrepareEvidenceRuntime:
    EVIDENCE_PARTIAL: Any
    EVIDENCE_REFINE: Any
    EVIDENCE_SUPPORTED: Any
    INFO_FAULT_DIAGNOSTIC: Any
    INFO_INTERFACE_NAVIGATION: Any
    INFO_NUMERIC_SPECIFICATION: Any
    INFO_PROCEDURE_FULL: Any
    INFO_PROCEDURE_SEGMENT: Any
    INFO_SEQUENCE_SYNCHRONIZATION: Any
    KIND_FAULT_DIAGNOSTIC: Any
    KIND_GENERAL_TECHNICAL: Any
    KIND_GUIDED_DIAGNOSTIC: Any
    KIND_PROCEDURE: Any
    MODE_ASK: Any
    MODE_ROOT_CAUSE: Any
    MODE_SMART_DIAGNOSTIC: Any
    POLICY_GENERAL_ALLOWED: Any
    REQ_DIAGNOSTIC_CAUSES: Any
    REQ_INTERFACE_LOCATIONS: Any
    REQ_NUMERIC_VALUE: Any
    REQ_ORDERED_ACTIONS: Any
    REQ_STATE_SEQUENCE: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE: Any
    _assistant_core_candidate_facet_metrics: Callable[..., Any]
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_diagnostic_priority_metrics: Callable[..., Any]
    _assistant_core_enumeration_metrics: Callable[..., Any]
    _assistant_core_enumeration_requested: Callable[..., Any]
    _assistant_core_expand_enumeration_sections: Callable[..., Any]
    _assistant_core_facet_balanced_pool: Callable[..., Any]
    _assistant_core_interface_navigation_signal: Callable[..., Any]
    _assistant_core_machine_catalog_candidates: Callable[..., Any]
    _assistant_core_machine_catalog_digest: Callable[..., Any]
    _assistant_core_numeric_signal: Callable[..., Any]
    _assistant_core_overview_catalog_candidates: Callable[..., Any]
    _assistant_core_ps_is_substantive: Callable[..., Any]
    _assistant_core_retrieval_query: Callable[..., Any]
    _assistant_core_root_candidate_viable: Callable[..., Any]
    _assistant_core_root_diagnostic_evidence_assurance: Callable[..., Any]
    _assistant_core_root_source_selection: Callable[..., Any]
    _assistant_core_sequence_signal: Callable[..., Any]
    _assistant_core_source_bonus: Callable[..., Any]
    _assistant_core_source_diversity_pool: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _dedup_citations_by_snippet: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _v13_assurance_fetch_neighbor_pages: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    _v13_deterministic_evidence_state: Callable[..., Any]
    _v13_evidence_metrics: Callable[..., Any]
    _v13_merge_candidates: Callable[..., Any]
    _v13_rescore_root_candidates: Callable[..., Any]
    re: Any
    time_module: Any


def assistant_core_prepare_evidence(request: AssistantCoreRequest, retrieval: dict, decision: AssistantCoreDecision, *, runtime: AssistantCorePrepareEvidenceRuntime, lineage: Optional[Callable[..., None]]=None) -> dict:
    EVIDENCE_PARTIAL = runtime.EVIDENCE_PARTIAL
    EVIDENCE_REFINE = runtime.EVIDENCE_REFINE
    EVIDENCE_SUPPORTED = runtime.EVIDENCE_SUPPORTED
    INFO_FAULT_DIAGNOSTIC = runtime.INFO_FAULT_DIAGNOSTIC
    INFO_INTERFACE_NAVIGATION = runtime.INFO_INTERFACE_NAVIGATION
    INFO_NUMERIC_SPECIFICATION = runtime.INFO_NUMERIC_SPECIFICATION
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    INFO_SEQUENCE_SYNCHRONIZATION = runtime.INFO_SEQUENCE_SYNCHRONIZATION
    KIND_FAULT_DIAGNOSTIC = runtime.KIND_FAULT_DIAGNOSTIC
    KIND_GENERAL_TECHNICAL = runtime.KIND_GENERAL_TECHNICAL
    KIND_GUIDED_DIAGNOSTIC = runtime.KIND_GUIDED_DIAGNOSTIC
    KIND_PROCEDURE = runtime.KIND_PROCEDURE
    MODE_ASK = runtime.MODE_ASK
    MODE_ROOT_CAUSE = runtime.MODE_ROOT_CAUSE
    MODE_SMART_DIAGNOSTIC = runtime.MODE_SMART_DIAGNOSTIC
    POLICY_GENERAL_ALLOWED = runtime.POLICY_GENERAL_ALLOWED
    REQ_DIAGNOSTIC_CAUSES = runtime.REQ_DIAGNOSTIC_CAUSES
    REQ_INTERFACE_LOCATIONS = runtime.REQ_INTERFACE_LOCATIONS
    REQ_NUMERIC_VALUE = runtime.REQ_NUMERIC_VALUE
    REQ_ORDERED_ACTIONS = runtime.REQ_ORDERED_ACTIONS
    REQ_STATE_SEQUENCE = runtime.REQ_STATE_SEQUENCE
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
    _assistant_core_candidate_facet_metrics = runtime._assistant_core_candidate_facet_metrics
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_diagnostic_priority_metrics = runtime._assistant_core_diagnostic_priority_metrics
    _assistant_core_enumeration_metrics = runtime._assistant_core_enumeration_metrics
    _assistant_core_enumeration_requested = runtime._assistant_core_enumeration_requested
    _assistant_core_expand_enumeration_sections = runtime._assistant_core_expand_enumeration_sections
    _assistant_core_facet_balanced_pool = runtime._assistant_core_facet_balanced_pool
    _assistant_core_interface_navigation_signal = runtime._assistant_core_interface_navigation_signal
    _assistant_core_machine_catalog_candidates = runtime._assistant_core_machine_catalog_candidates
    _assistant_core_machine_catalog_digest = runtime._assistant_core_machine_catalog_digest
    _assistant_core_numeric_signal = runtime._assistant_core_numeric_signal
    _assistant_core_overview_catalog_candidates = runtime._assistant_core_overview_catalog_candidates
    _assistant_core_ps_is_substantive = runtime._assistant_core_ps_is_substantive
    _assistant_core_retrieval_query = runtime._assistant_core_retrieval_query
    _assistant_core_root_candidate_viable = runtime._assistant_core_root_candidate_viable
    _assistant_core_root_diagnostic_evidence_assurance = runtime._assistant_core_root_diagnostic_evidence_assurance
    _assistant_core_root_source_selection = runtime._assistant_core_root_source_selection
    _assistant_core_sequence_signal = runtime._assistant_core_sequence_signal
    _assistant_core_source_bonus = runtime._assistant_core_source_bonus
    _assistant_core_source_diversity_pool = runtime._assistant_core_source_diversity_pool
    _content_term_set = runtime._content_term_set
    _dedup_citations_by_snippet = runtime._dedup_citations_by_snippet
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _term_overlap_score = runtime._term_overlap_score
    _v13_assurance_fetch_neighbor_pages = runtime._v13_assurance_fetch_neighbor_pages
    _v13_candidate_text = runtime._v13_candidate_text
    _v13_current_budget = runtime._v13_current_budget
    _v13_deterministic_evidence_state = runtime._v13_deterministic_evidence_state
    _v13_evidence_metrics = runtime._v13_evidence_metrics
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_rescore_root_candidates = runtime._v13_rescore_root_candidates
    re = runtime.re
    time_module = runtime.time_module
    retrieval_query = _assistant_core_retrieval_query(request)
    retrieval = _assistant_core_root_diagnostic_evidence_assurance(
        request, retrieval, decision
    )
    candidates = [
        dict(c)
        for c in (retrieval.get("candidates") or retrieval.get("citations") or [])
        if isinstance(c, dict)
    ]
    if lineage is not None:
        lineage("copy", tuple(zip(
            (c for c in (retrieval.get("candidates") or retrieval.get("citations") or [])
             if isinstance(c, dict)), candidates)))
    if not candidates:
        return {
            "supported": False,
            "retrieval": {**dict(retrieval or {}), "citations": [], "candidates": []},
        }

    relevant_ids = {
        str(x or "").strip()
        for x in decision.relevant_evidence_ids
        if str(x or "").strip()
    }
    preferred = {
        str(x or "").strip().lower()
        for x in decision.preferred_source_types
        if str(x or "").strip()
    }
    required_answer_types = {
        str(x or "").strip().lower()
        for x in decision.required_answer_types
        if str(x or "").strip()
    }
    must_cover_facets = _dedup_text_values(
        [item.facet for item in decision.facet_queries if bool(item.must_cover)]
        or list(decision.required_facets),
        limit=12,
    )
    enumeration_requested = _assistant_core_enumeration_requested(request, decision)
    query_low_for_catalog = _normalize_unicode_advanced(retrieval_query or "").casefold()
    overview_catalog_requested = bool(
        enumeration_requested
        and re.search(r"\b(?:gruppi|componenti|elementi|parti principali|groups?|components?|main parts?|main elements?)\b", query_low_for_catalog)
    )
    overview_catalog_candidates: list[dict] = []
    if overview_catalog_requested:
        catalog = _assistant_core_machine_catalog_candidates(request)
        if catalog:
            overview_catalog_candidates = _assistant_core_overview_catalog_candidates(catalog)
            candidates = _v13_merge_candidates([
                overview_catalog_candidates,
                _assistant_core_source_diversity_pool(candidates, per_type=2),
                candidates,
            ])
    enumeration_neighbor_count = 0
    if enumeration_requested and request.machine_id:
        budget_for_neighbors = _v13_current_budget()
        neighbor_deadline = (
            float(budget_for_neighbors.deadline_monotonic)
            if budget_for_neighbors is not None
            else time_module.monotonic() + 5.0
        )
        try:
            enumeration_neighbors = _v13_assurance_fetch_neighbor_pages(
                q=retrieval_query,
                company_id=request.company_id,
                machine_id=request.machine_id,
                candidates=candidates[:16],
                retrieval=retrieval,
                response_language=request.response_language,
                deadline_monotonic=neighbor_deadline,
            )
            if enumeration_neighbors:
                enumeration_neighbor_count = len(enumeration_neighbors)
                candidates = _v13_merge_candidates([candidates, enumeration_neighbors])
            section_pages = _assistant_core_expand_enumeration_sections(
                request=request,
                retrieval=retrieval,
                candidates=candidates[:18],
            )
            if section_pages:
                enumeration_neighbor_count += len(section_pages)
                candidates = _v13_merge_candidates([
                    _assistant_core_source_diversity_pool(candidates, per_type=2),
                    candidates,
                    section_pages,
                ])
        except Exception as exc:
            print("ASSISTANT_CORE_ENUMERATION_NEIGHBOR_FAIL", str(exc)[:500])
    needs_numeric = (
        decision.information_task == INFO_NUMERIC_SPECIFICATION
        or REQ_NUMERIC_VALUE in required_answer_types
    )
    needs_interface = (
        decision.information_task == INFO_INTERFACE_NAVIGATION
        or REQ_INTERFACE_LOCATIONS in required_answer_types
    )
    needs_sequence = (
        decision.information_task == INFO_SEQUENCE_SYNCHRONIZATION
        or REQ_STATE_SEQUENCE in required_answer_types
    )
    needs_ordered_actions = (
        decision.information_task in {INFO_PROCEDURE_FULL, INFO_PROCEDURE_SEGMENT}
        or decision.request_kind == KIND_PROCEDURE
        or REQ_ORDERED_ACTIONS in required_answer_types
    )
    needs_diagnostic = (
        decision.request_kind in {KIND_FAULT_DIAGNOSTIC, KIND_GUIDED_DIAGNOSTIC}
        or decision.information_task == INFO_FAULT_DIAGNOSTIC
        or REQ_DIAGNOSTIC_CAUSES in required_answer_types
    )
    query_terms = _content_term_set(retrieval_query, limit=80)

    scored: list[dict] = []
    preferred_viable = False
    for c in candidates:
        cc = dict(c)
        cid = str(cc.get("citation_id") or "").strip()
        source_type = _assistant_core_candidate_source_type(cc)
        cc["source_type"] = source_type
        text = _v13_candidate_text(cc)
        overlap = _term_overlap_score(query_terms, _content_term_set(text, limit=140)) if query_terms else 0.0
        semantic = float(
            cc.get("semantic_similarity", cc.get("gate_similarity", cc.get("similarity", 0.0)))
            or 0.0
        )
        base_score = float(
            cc.get("v13_score", cc.get("retrieval_score", semantic)) or 0.0
        )
        source_bonus = _assistant_core_source_bonus(
            source_type, decision.request_kind, preferred, decision.information_task
        )
        facet_metrics = _assistant_core_candidate_facet_metrics(cc, must_cover_facets)
        facet_coverage = float(facet_metrics.get("coverage") or 0.0)
        diagnostic_priority = _assistant_core_diagnostic_priority_metrics(cc, decision)
        clue_coverage = float(diagnostic_priority.get("score") or 0.0)
        numeric_signal = _assistant_core_numeric_signal(text)
        interface_signal = _assistant_core_interface_navigation_signal(text)
        sequence_signal = _assistant_core_sequence_signal(text)
        substantive_ps = _assistant_core_ps_is_substantive(cc)
        enumeration_metrics = _assistant_core_enumeration_metrics(text)
        enumeration_bonus = 0.0
        if enumeration_requested and (
            facet_coverage >= 0.18
            or overlap >= 0.02
            or float(cc.get("assistant_core_router_id_bonus") or 0.0) > 0.0
            or cid in relevant_ids
        ):
            enumeration_bonus = min(
                0.34,
                0.035 * float(enumeration_metrics.get("item_count") or 0)
                + 0.025 * float(enumeration_metrics.get("heading_count") or 0),
            )
        task_contract_bonus = 0.0
        # Composite contracts are cumulative: a value-plus-checklist request must
        # not lose the numeric evidence merely because ordered actions are also
        # requested.
        if needs_numeric:
            task_contract_bonus += 0.18 if numeric_signal.get("has_number_with_unit") else 0.08 if numeric_signal.get("has_number") else -0.16
            task_contract_bonus += min(0.12, facet_coverage * 0.12)
        if needs_interface:
            task_contract_bonus += 0.16 if interface_signal else -0.12
            task_contract_bonus += min(0.14, facet_coverage * 0.14)
        if needs_sequence:
            task_contract_bonus += 0.15 if sequence_signal else -0.10
            task_contract_bonus += min(0.14, facet_coverage * 0.14)
        if needs_ordered_actions:
            task_contract_bonus += min(0.12, facet_coverage * 0.12)
        if source_type == "ps" and not substantive_ps:
            task_contract_bonus -= 0.35
        router_id_bonus = 0.16 if cid and cid in relevant_ids else 0.0
        exact_bonus = 0.18 if bool(
            cc.get("exact_code_hit")
            or cc.get("gate_exact_code_hit")
            or cc.get("exact_identifier_token")
        ) else 0.0
        title_bonus = min(0.16, max(0.0, float(cc.get("structured_title_match_score") or 0.0)) * 0.20)
        lexical_bonus = min(0.16, max(0.0, overlap) * 0.24)
        facet_retrieval_bonus = min(
            0.24,
            0.08 * len(cc.get("assistant_core_facet_hits") or [])
            + 0.08 * float(cc.get("assistant_core_facet_retrieval_score") or 0.0),
        )
        diagnostic_clue_bonus = min(0.20, 0.20 * clue_coverage) if needs_diagnostic else 0.0
        score = (
            base_score
            + source_bonus
            + router_id_bonus
            + exact_bonus
            + title_bonus
            + lexical_bonus
            + task_contract_bonus
            + facet_retrieval_bonus
            + diagnostic_clue_bonus
            + enumeration_bonus
        )
        if decision.source_type_policy == "require" and preferred and source_type not in preferred:
            score -= 0.12
        elif decision.source_type_policy == "prefer" and preferred and source_type not in preferred:
            score -= 0.04
        cc["assistant_core_selected"] = True
        cc["evidence_gate_selected"] = True
        cc["assistant_core_router_id_bonus"] = router_id_bonus
        cc["assistant_core_source_bonus"] = source_bonus
        cc["assistant_core_query_overlap"] = overlap
        cc["assistant_core_information_task"] = decision.information_task
        cc["assistant_core_facet_coverage"] = facet_coverage
        cc["assistant_core_covered_facets"] = list(facet_metrics.get("covered") or [])
        cc["assistant_core_missing_facets"] = list(facet_metrics.get("missing") or [])
        cc["assistant_core_tagged_facets"] = list(facet_metrics.get("tagged") or [])
        cc["assistant_core_diagnostic_clue_coverage"] = clue_coverage
        cc["assistant_core_covered_diagnostic_clues"] = list(diagnostic_priority.get("covered") or [])
        cc["assistant_core_missing_diagnostic_clues"] = list(diagnostic_priority.get("missing") or [])
        cc["assistant_core_diagnostic_priority"] = dict(diagnostic_priority)
        cc["assistant_core_numeric_signal"] = dict(numeric_signal)
        cc["assistant_core_interface_signal"] = bool(interface_signal)
        cc["assistant_core_sequence_signal"] = bool(sequence_signal)
        cc["assistant_core_ps_substantive"] = bool(substantive_ps)
        cc["assistant_core_task_contract_bonus"] = task_contract_bonus
        cc["assistant_core_facet_retrieval_bonus"] = facet_retrieval_bonus
        cc["assistant_core_diagnostic_clue_bonus"] = diagnostic_clue_bonus
        cc["assistant_core_enumeration_requested"] = bool(enumeration_requested)
        cc["assistant_core_enumeration_metrics"] = dict(enumeration_metrics)
        cc["assistant_core_enumeration_bonus"] = float(enumeration_bonus)
        cc["v13_score"] = score
        cc["retrieval_score"] = max(float(cc.get("retrieval_score") or 0.0), score)
        if source_type in preferred and (semantic >= 0.28 or overlap >= 0.04 or title_bonus > 0.0 or exact_bonus > 0.0):
            preferred_viable = True
        scored.append(cc)
        if lineage is not None:
            lineage("score", ((c, cc),))

    if lineage is not None:
        lineage("scored", tuple(scored))

    # An explicit "only manual/photo/..." request may constrain source type, but a
    # router mistake must never erase all otherwise relevant evidence. Enforce the
    # type only when at least one preferred candidate is independently viable.
    if decision.source_type_policy == "require" and preferred and preferred_viable:
        filtered = [c for c in scored if str(c.get("source_type") or "") in preferred]
        if filtered:
            scored = filtered

    scored.sort(
        key=lambda c: (
            -float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0),
            0 if bool(c.get("exact_machine_scope")) else 1,
            str(c.get("bubble_document_id") or ""),
            int(c.get("page_from") or 0),
        )
    )
    scored = _dedup_citations_by_snippet(scored, max_items=24)

    diagnostic_mode = decision.effective_mode in {MODE_ROOT_CAUSE, MODE_SMART_DIAGNOSTIC}
    if diagnostic_mode:
        scored = _v13_rescore_root_candidates(retrieval_query, scored)
        for candidate in scored:
            diagnostic_priority = _assistant_core_diagnostic_priority_metrics(candidate, decision)
            candidate["assistant_core_diagnostic_priority"] = dict(diagnostic_priority)
            candidate["assistant_core_diagnostic_clue_coverage"] = float(
                diagnostic_priority.get("score") or 0.0
            )
            priority_bonus = min(0.36, 0.36 * float(diagnostic_priority.get("score") or 0.0))
            candidate["assistant_core_diagnostic_priority_bonus"] = priority_bonus
            candidate["v13_score"] = float(
                candidate.get("v13_score", candidate.get("retrieval_score", candidate.get("similarity", 0.0)))
                or 0.0
            ) + priority_bonus
            candidate["assistant_core_root_viable"] = _assistant_core_root_candidate_viable(
                request, decision, candidate
            )
        scored.sort(
            key=lambda c: (
                0 if bool(c.get("assistant_core_root_viable")) else 1,
                -float((c.get("assistant_core_diagnostic_priority") or {}).get("score") or 0.0),
                -float(c.get("v13_score", c.get("retrieval_score", c.get("similarity", 0.0))) or 0.0),
                str(c.get("citation_id") or ""),
            )
        )

    deterministic_mode = "root_cause" if diagnostic_mode else "ask"
    deterministic_state, deterministic_signals = _v13_deterministic_evidence_state(
        retrieval_query,
        scored,
        mode=deterministic_mode,
        narrow_scope=request.narrow_scope,
    )

    top_candidates = scored[:12]
    source_types = [str(c.get("source_type") or "") for c in top_candidates]
    has_procedure_sources = any(st in {"procedure", "step"} for st in source_types)
    top_similarity = float(deterministic_signals.get("top_similarity") or 0.0)
    top_overlap = float(deterministic_signals.get("top_overlap") or 0.0)
    fts_hits = int(deterministic_signals.get("fts_overlap_count") or 0)
    exact_hits = int(deterministic_signals.get("exact_code_count") or 0)

    def _contract_semantic(candidate: dict) -> float:
        return float(
            candidate.get(
                "semantic_similarity",
                candidate.get("gate_similarity", candidate.get("similarity", 0.0)),
            )
            or 0.0
        )

    numeric_viable = [
        c for c in top_candidates
        if bool((c.get("assistant_core_numeric_signal") or {}).get("has_number"))
        and (
            float(c.get("assistant_core_facet_coverage") or 0.0) >= 0.20
            or float(c.get("assistant_core_router_id_bonus") or 0.0) > 0.0
            or float(c.get("assistant_core_query_overlap") or 0.0) >= 0.025
            # Query and source may be in different languages. A value with a real
            # engineering unit plus independent semantic similarity is a valid
            # cross-language contract signal; naked page/menu numbers require a
            # materially stronger similarity and cannot win merely by being numeric.
            or (
                bool((c.get("assistant_core_numeric_signal") or {}).get("has_number_with_unit"))
                and _contract_semantic(c) >= 0.30
            )
            or _contract_semantic(c) >= 0.46
        )
    ]
    interface_viable = [
        c for c in top_candidates
        if bool(c.get("assistant_core_interface_signal"))
        and (
            float(c.get("assistant_core_facet_coverage") or 0.0) >= 0.20
            or float(c.get("assistant_core_router_id_bonus") or 0.0) > 0.0
            or _contract_semantic(c) >= 0.40
        )
    ]
    sequence_viable = [
        c for c in top_candidates
        if bool(c.get("assistant_core_sequence_signal"))
        and (
            float(c.get("assistant_core_facet_coverage") or 0.0) >= 0.20
            or float(c.get("assistant_core_router_id_bonus") or 0.0) > 0.0
            or float(c.get("assistant_core_query_overlap") or 0.0) >= 0.035
            or _contract_semantic(c) >= 0.42
        )
    ]
    root_viable = [c for c in scored if bool(c.get("assistant_core_root_viable"))]

    covered_facets: set[str] = set()
    for candidate in top_candidates:
        covered_facets.update(
            str(x or "").strip().casefold()
            for x in (candidate.get("assistant_core_covered_facets") or [])
            if str(x or "").strip()
        )
        covered_facets.update(
            str(x or "").strip().casefold()
            for x in (candidate.get("assistant_core_facet_hits") or [])
            if str(x or "").strip()
        )
    required_facet_keys = {
        str(x or "").strip().casefold()
        for x in must_cover_facets
        if str(x or "").strip()
    }
    facet_evidence_coverage = (
        len(required_facet_keys & covered_facets) / max(1, len(required_facet_keys))
        if required_facet_keys
        else 1.0
    )
    covered_answer_types: set[str] = set()
    for candidate in top_candidates:
        covered_answer_types.update(
            str(x or "").strip().lower()
            for x in (candidate.get("assistant_core_facet_answer_types") or [])
            if str(x or "").strip()
        )

    # Router evidence IDs/state are advisory, but a failed router is not permission
    # to synthesize from arbitrary nearby text. Successful semantic routing can be
    # corroborated by deterministic retrieval signals.
    independent_signal = bool(
        deterministic_state == "supported"
        or top_similarity >= 0.34
        or top_overlap >= 0.035
        or fts_hits > 0
        or exact_hits > 0
        or (decision.request_kind == KIND_PROCEDURE and has_procedure_sources)
    )
    supported = False
    if not decision.degraded:
        if deterministic_state == "supported":
            supported = True
        elif decision.evidence_state in {EVIDENCE_SUPPORTED, EVIDENCE_PARTIAL, EVIDENCE_REFINE} and independent_signal:
            supported = True

    # Every mandatory output shape must have independently viable evidence.
    if needs_ordered_actions:
        supported = supported and (
            has_procedure_sources
            or top_similarity >= 0.40
            or (top_similarity >= 0.32 and top_overlap >= 0.05)
        )
    if needs_numeric:
        supported = supported and bool(numeric_viable)
    if needs_interface:
        supported = supported and bool(interface_viable)
    if needs_sequence:
        supported = supported and bool(sequence_viable)
    if needs_diagnostic or diagnostic_mode:
        # A P&S record is not sufficient merely because it belongs to the machine.
        # At least one candidate must match the symptom/subsystem/facets and carry a
        # plausible causal or discriminating-check signal.
        supported = supported and bool(root_viable)

    precision_requirements = required_answer_types & {
        REQ_NUMERIC_VALUE,
        REQ_INTERFACE_LOCATIONS,
        REQ_STATE_SEQUENCE,
        REQ_DIAGNOSTIC_CAUSES,
    }
    minimum_facet_coverage: Optional[float] = None
    missing_precision_types: set[str] = set()
    facet_threshold_pass = True
    precision_type_annotations_pass = True
    if must_cover_facets and precision_requirements:
        # The strict multi-facet gate remains the preferred path. It is useful for
        # ranking and for avoiding a synthesis call on obviously incomplete packs,
        # but its lexical/annotation coverage is not ground truth: cross-language
        # wording and distributed procedure evidence can under-report coverage.
        minimum_facet_coverage = 0.70 if len(must_cover_facets) >= 4 else 0.80
        facet_threshold_pass = facet_evidence_coverage >= minimum_facet_coverage
        supported = supported and facet_threshold_pass
        if covered_answer_types:
            missing_precision_types = precision_requirements - covered_answer_types
            # Independent contract signals can satisfy precision types even when a
            # router/facet annotation is absent.
            if REQ_NUMERIC_VALUE in missing_precision_types and numeric_viable:
                missing_precision_types.remove(REQ_NUMERIC_VALUE)
            if REQ_INTERFACE_LOCATIONS in missing_precision_types and interface_viable:
                missing_precision_types.remove(REQ_INTERFACE_LOCATIONS)
            if REQ_STATE_SEQUENCE in missing_precision_types and sequence_viable:
                missing_precision_types.remove(REQ_STATE_SEQUENCE)
            if REQ_DIAGNOSTIC_CAUSES in missing_precision_types and root_viable:
                missing_precision_types.remove(REQ_DIAGNOSTIC_CAUSES)
            precision_type_annotations_pass = not missing_precision_types
            supported = supported and precision_type_annotations_pass

    strict_supported = bool(supported)
    baseline_fallback_used = False
    baseline_fallback_reason = ""

    # Monotonic ASK admission: V10.2's facet annotations may add evidence and
    # improve ranking, but they must not erase an independently viable baseline
    # evidence pack. When the strict gate alone rejects an ASK pack, admit the
    # baseline to synthesis and let the existing grounded answer-contract + one
    # bounded repair call decide whether the final answer is complete. Hard
    # numeric/interface/sequence signals remain mandatory, and diagnostics never
    # use this fallback.
    if (
        not supported
        and decision.effective_mode == MODE_ASK
        and not decision.degraded
        and decision.request_kind not in {KIND_FAULT_DIAGNOSTIC, KIND_GUIDED_DIAGNOSTIC}
        and decision.evidence_state in {EVIDENCE_SUPPORTED, EVIDENCE_PARTIAL, EVIDENCE_REFINE}
    ):
        baseline_strength = bool(
            top_candidates
            and independent_signal
            and (
                deterministic_state == "supported"
                or top_similarity >= 0.38
                or (
                    has_procedure_sources
                    and (
                        top_similarity >= 0.28
                        or facet_evidence_coverage > 0.0
                        or top_overlap >= 0.025
                    )
                )
                or (fts_hits > 0 and top_overlap >= 0.03)
                or exact_hits > 0
            )
        )
        baseline_contract_viable = bool(
            (not needs_numeric or numeric_viable)
            and (not needs_interface or interface_viable)
            and (not needs_sequence or sequence_viable)
            and (
                not needs_ordered_actions
                or has_procedure_sources
                or top_similarity >= 0.40
                or (top_similarity >= 0.32 and top_overlap >= 0.05)
            )
        )
        baseline_scope_viable = bool(
            decision.source_type_policy != "require"
            or not preferred
            or preferred_viable
        )
        if baseline_strength and baseline_contract_viable and baseline_scope_viable:
            supported = True
            baseline_fallback_used = True
            baseline_fallback_reason = (
                "strict_facet_or_annotation_gate_rejected_"
                "independently_viable_ask_evidence"
            )

    if decision.request_kind == KIND_GENERAL_TECHNICAL and decision.evidence_policy == POLICY_GENERAL_ALLOWED:
        supported = deterministic_state == "supported" and not decision.degraded
        strict_supported = bool(supported)
        baseline_fallback_used = False
        baseline_fallback_reason = ""

    root_source_selection_summary: dict = {}
    if diagnostic_mode:
        limit = V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE
        selected_pool = root_viable
        if decision.effective_mode == MODE_ROOT_CAUSE:
            root_source_selection = _assistant_core_root_source_selection(
                root_viable,
                limit=limit,
            )
            selected_pool = list(root_source_selection.candidates)
            root_source_selection_summary = dict(root_source_selection.summary)
    else:
        limit = max(
            V13_MAX_EVIDENCE_ITEMS_ASK,
            12 if needs_ordered_actions else V13_MAX_EVIDENCE_ITEMS_ASK,
            24 if overview_catalog_requested else (16 if enumeration_requested else V13_MAX_EVIDENCE_ITEMS_ASK),
            min(24 if overview_catalog_requested else 18, len(must_cover_facets) + 8) if must_cover_facets else 0,
        )
        priority_pool: list[dict] = []
        if needs_numeric:
            priority_pool.extend(numeric_viable)
        if needs_interface:
            priority_pool.extend(interface_viable)
        if needs_sequence:
            priority_pool.extend(sequence_viable)
        if needs_ordered_actions:
            priority_pool.extend(
                c for c in scored
                if str(c.get("source_type") or "") in {"procedure", "step"}
            )
        if enumeration_requested:
            priority_pool.extend(
                sorted(
                    scored,
                    key=lambda c: (
                        -float(c.get("assistant_core_enumeration_bonus") or 0.0),
                        -float(c.get("v13_score", c.get("retrieval_score", 0.0)) or 0.0),
                    ),
                )[:10]
            )
        selected_pool = _v13_merge_candidates(
            [priority_pool, scored]
        ) if priority_pool else scored

    if supported and must_cover_facets:
        selected = _assistant_core_facet_balanced_pool(
            selected_pool, must_cover_facets, limit=limit
        )
    else:
        selected = selected_pool[:limit] if supported else []
    if supported and overview_catalog_requested and overview_catalog_candidates:
        # Enumeration is a recall task: low lexical similarity must not erase an
        # explicitly indexed assembly or auxiliary system from the machine catalog.
        selected = _v13_merge_candidates([
            overview_catalog_candidates,
            selected,
            selected_pool,
        ])[:limit]
    machine_catalog_digest = _assistant_core_machine_catalog_digest(
        overview_catalog_candidates if overview_catalog_requested else []
    )
    out_retrieval = {
        **dict(retrieval or {}),
        "candidates": selected,
        "citations": selected,
        "metrics": _v13_evidence_metrics(selected),
        "assistant_core_contract": {
            "information_task": decision.information_task,
            "required_answer_types": list(decision.required_answer_types),
            "required_facets": list(decision.required_facets),
            "must_cover_facets": list(must_cover_facets),
            "facet_queries": [
                {
                    "facet": item.facet,
                    "answer_type": item.answer_type,
                    "must_cover": item.must_cover,
                    "dense_queries": list(item.dense_queries),
                    "lexical_queries": list(item.lexical_queries),
                    "exact_terms": list(item.exact_terms),
                    "preferred_source_types": list(item.preferred_source_types),
                }
                for item in decision.facet_queries
            ],
            "diagnostic_clues": list(decision.diagnostic_clues),
            "diagnostic_exclusions": list(decision.diagnostic_exclusions),
            "missing_information": list(decision.missing_information),
            "fail_closed": True,
            "overview_catalog_requested": bool(overview_catalog_requested),
            "machine_catalog_digest": machine_catalog_digest,
            "machine_catalog_source_ids": [
                str(c.get("citation_id") or "")
                for c in overview_catalog_candidates
                if str(c.get("citation_id") or "").strip()
            ],
        },
        "assistant_core_decision": {
            "request_kind": decision.request_kind,
            "information_task": decision.information_task,
            "required_answer_types": list(decision.required_answer_types),
            "must_cover_facets": list(must_cover_facets),
            "effective_mode": decision.effective_mode,
            "router_evidence_state": decision.evidence_state,
            "preferred_source_types": list(decision.preferred_source_types),
            "source_type_policy": decision.source_type_policy,
            "router_relevant_evidence_ids": list(decision.relevant_evidence_ids),
            "deterministic_state": deterministic_state,
            "deterministic_signals": deterministic_signals,
            "independent_signal": independent_signal,
            "numeric_viable_count": len(numeric_viable),
            "interface_viable_count": len(interface_viable),
            "sequence_viable_count": len(sequence_viable),
            "root_viable_count": len(root_viable),
            "root_diversified_pool_count": (
                len(selected_pool)
                if decision.effective_mode == MODE_ROOT_CAUSE
                else len(root_viable)
            ),
            "root_source_selection": root_source_selection_summary,
            "facet_evidence_coverage": facet_evidence_coverage,
            "minimum_facet_coverage": minimum_facet_coverage,
            "facet_threshold_pass": facet_threshold_pass,
            "covered_facets": sorted(covered_facets),
            "covered_answer_types": sorted(covered_answer_types),
            "missing_precision_types": sorted(missing_precision_types),
            "precision_type_annotations_pass": precision_type_annotations_pass,
            "strict_supported": strict_supported,
            "baseline_fallback_used": baseline_fallback_used,
            "baseline_fallback_reason": baseline_fallback_reason,
            "enumeration_requested": bool(enumeration_requested),
            "overview_catalog_requested": bool(overview_catalog_requested),
            "overview_catalog_count": len(overview_catalog_candidates),
            "overview_catalog_digest_chars": len(machine_catalog_digest),
            "enumeration_neighbor_count": int(enumeration_neighbor_count),
            "diagnostic_clues": list(decision.diagnostic_clues),
            "diagnostic_exclusions": list(decision.diagnostic_exclusions),
            "supported": supported,
        },
    }
    return {"supported": bool(supported and selected), "retrieval": out_retrieval}


