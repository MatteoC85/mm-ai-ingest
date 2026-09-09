"""P6-B4d: request flow and last-resort scalar recovery with explicit dependencies.

This is a behavior-preserving extraction, NOT canonical activation or a new Core.
The existing scope, budget, cache policies, direct exits, fallback conditions,
exception handling and UI formatters are unchanged. No transport is created here.

The composition root supplies run_core explicitly; it may eventually bind the
EXISTING Core to the request-owned evidence session instead of mutating shared
hooks. In this checkpoint main supplies the current Core.run unchanged. Cache
hits still precede Core execution and scalar rescue still follows its result.
Those boundaries must be migrated before an end-to-end canonical acceptance.

This module does NOT turn cached payloads, URLs or scalar candidates into authority.
No session, authorization inference, linguistic rule or extra provider call is
introduced. Nested recovery is an explicit callback, not an implicit new engine.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True, slots=True, repr=False)
class RequestFlowRuntime:
    """Per-invocation dependency snapshot; repr never prints the internal secret.

    Capturing collaborators does not invoke them. Allocation is CPU/memory, not
    an LLM/DB call, and is not claimed free. Mutable collaborator internals keep
    their existing lifecycle; this class does not certify their isolation.
    """
    AI_INTERNAL_SECRET: Any
    ASK_MAX_TOP_K: Any
    AssistantCoreRequest: Any
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    EVIDENCE_SUPPORTED: Any
    HTTPException: Any
    INFO_NUMERIC_SPECIFICATION: Any
    KIND_FACTUAL: Any
    MODE_ASK: Any
    MODE_ROOT_CAUSE: Any
    POLICY_MACHINE_REQUIRED: Any
    REQ_NUMERIC_VALUE: Any
    _PRECISION_FACT_RUNTIME: Callable[..., Any]
    _ROOT_DIAGNOSTIC_QUERY_PROFILE_KEY: Any
    _V13BudgetExceeded: Any
    _V13_BUDGET_CTX: Any
    _assistant_core_attach_runtime_meta: Callable[..., Any]
    _assistant_core_budget_response: Callable[..., Any]
    _assistant_core_clear_unsupported_sources: Callable[..., Any]
    _assistant_core_diagnostic_query_profile: Callable[..., Any]
    _assistant_core_new_budget: Callable[..., Any]
    _assistant_core_precision_fact_rescue: Callable[..., Any]
    _assistant_core_technical_error: Callable[..., Any]
    _assistant_ui_finalize_response: Callable[..., Any]
    _build_rg_links: Callable[..., Any]
    _resolve_query_scope: Callable[..., Any]
    _retrieval_diagnostic_query: Any
    _retrieval_precision_facts: Any
    _retrieval_review_packet: Any
    _retrieval_review_references: Any
    _root_cause_response_language: Callable[..., Any]
    _sanitize_citations_for_response: Callable[..., Any]
    _select_response_language: Callable[..., Any]
    _v13_cache_lookup: Callable[..., Any]
    _v13_cache_store: Callable[..., Any]
    response_has_rejected_answer: Callable[..., Any]
    run_core: Callable[..., Any]


def precision_fact_rescue(
    *, q: str, company_id: str, machine_id: str,
    doc_ids: Optional[list[str]], bubble_document_id: Optional[str],
    response_language: str, top_k: int, answer_contract: Optional[dict] = None,
    runtime: RequestFlowRuntime,
) -> Optional[dict]:
    """Recover one unambiguous scalar fact after ASK has already failed closed.

    Safety, scope and semantic routing run first in Assistant Core.  This rescue is
    deliberately unavailable to Root Cause/Smart Diagnostic and never replaces an
    existing successful answer.  It reads only authorized ``document_pages`` and
    requires a same-label numeric value with an engineering unit.
    """
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    EVIDENCE_SUPPORTED = runtime.EVIDENCE_SUPPORTED
    INFO_NUMERIC_SPECIFICATION = runtime.INFO_NUMERIC_SPECIFICATION
    KIND_FACTUAL = runtime.KIND_FACTUAL
    MODE_ASK = runtime.MODE_ASK
    POLICY_MACHINE_REQUIRED = runtime.POLICY_MACHINE_REQUIRED
    REQ_NUMERIC_VALUE = runtime.REQ_NUMERIC_VALUE
    _PRECISION_FACT_RUNTIME = runtime._PRECISION_FACT_RUNTIME
    _build_rg_links = runtime._build_rg_links
    _retrieval_precision_facts = runtime._retrieval_precision_facts
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    if (
        (not str(machine_id or "").strip()
         or str(machine_id or "").strip() == COMPANY_GENERAL_MACHINE_SENTINEL)
        and not doc_ids
        and not str(bubble_document_id or "").strip()
    ):
        return None
    try:
        resolution = _retrieval_precision_facts.resolve_precision_fact(
            query=q,
            company_id=company_id,
            machine_id=machine_id,
            doc_ids=doc_ids,
            bubble_document_id=str(bubble_document_id or "").strip() or None,
            runtime=_PRECISION_FACT_RUNTIME(),
            answer_contract=answer_contract,
        )
    except Exception as exc:
        print("PRECISION_FACT_RESCUE_FAIL", type(exc).__name__, str(exc)[:500])
        return None
    if resolution is None:
        return None

    raw_citation = _retrieval_precision_facts.resolution_to_candidate(resolution)
    try:
        citations = _sanitize_citations_for_response(
            [raw_citation], company_id=company_id
        )
    except Exception as exc:
        print("PRECISION_FACT_CITATION_FAIL", type(exc).__name__, str(exc)[:500])
        return None
    if not citations:
        return None
    try:
        rg_links = _build_rg_links(company_id, citations)
    except Exception as exc:
        print("PRECISION_FACT_LINK_FAIL", type(exc).__name__, str(exc)[:500])
        return None
    if not rg_links:
        return None

    is_en = str(response_language or "").lower().startswith("en")
    answer = (
        f"The requested value is {resolution.value}."
        if is_en
        else f"{resolution.label}: {resolution.value}."
    )
    return {
        "ok": True,
        "status": "answered",
        "result_code": "ANSWERED",
        "requested_mode": MODE_ASK,
        "effective_mode": MODE_ASK,
        "routed": False,
        "request_kind": KIND_FACTUAL,
        "information_task": INFO_NUMERIC_SPECIFICATION,
        "required_answer_types": [REQ_NUMERIC_VALUE],
        "evidence_state": EVIDENCE_SUPPORTED,
        "evidence_policy": POLICY_MACHINE_REQUIRED,
        "grounding": "indexed_machine_sources",
        "answer": answer,
        "language": response_language,
        "citations": citations,
        "rg_links": rg_links,
        "top_k": max(1, int(top_k or 1)),
        "similarity_max": None,
        "chat_model": "deterministic_precision_fact_rescue_v1",
        "meta": {
            "cacheable": True,
            "semantic_cacheable": False,
            "precision_fact_rescue": {
                "version": "precision-fact-v1",
                "page": int(resolution.page_number),
                "supporting_pages": list(resolution.supporting_pages),
                "property_terms": list(resolution.property_terms),
                "score": round(float(resolution.score), 6),
            },
        },
    }


def run_sync(payload: Any, x_ai_internal_secret: Optional[str], *,
             requested_mode: str, runtime: RequestFlowRuntime) -> dict:
    AI_INTERNAL_SECRET = runtime.AI_INTERNAL_SECRET
    ASK_MAX_TOP_K = runtime.ASK_MAX_TOP_K
    AssistantCoreRequest = runtime.AssistantCoreRequest
    HTTPException = runtime.HTTPException
    MODE_ASK = runtime.MODE_ASK
    MODE_ROOT_CAUSE = runtime.MODE_ROOT_CAUSE
    run_core = runtime.run_core
    _ROOT_DIAGNOSTIC_QUERY_PROFILE_KEY = runtime._ROOT_DIAGNOSTIC_QUERY_PROFILE_KEY
    _V13BudgetExceeded = runtime._V13BudgetExceeded
    _V13_BUDGET_CTX = runtime._V13_BUDGET_CTX
    _assistant_core_attach_runtime_meta = runtime._assistant_core_attach_runtime_meta
    _assistant_core_budget_response = runtime._assistant_core_budget_response
    _assistant_core_clear_unsupported_sources = runtime._assistant_core_clear_unsupported_sources
    _assistant_core_diagnostic_query_profile = runtime._assistant_core_diagnostic_query_profile
    _assistant_core_new_budget = runtime._assistant_core_new_budget
    _assistant_core_precision_fact_rescue = runtime._assistant_core_precision_fact_rescue
    _assistant_core_technical_error = runtime._assistant_core_technical_error
    _assistant_ui_finalize_response = runtime._assistant_ui_finalize_response
    _resolve_query_scope = runtime._resolve_query_scope
    _retrieval_diagnostic_query = runtime._retrieval_diagnostic_query
    _retrieval_precision_facts = runtime._retrieval_precision_facts
    _retrieval_review_packet = runtime._retrieval_review_packet
    _retrieval_review_references = runtime._retrieval_review_references
    _root_cause_response_language = runtime._root_cause_response_language
    _select_response_language = runtime._select_response_language
    _v13_cache_lookup = runtime._v13_cache_lookup
    _v13_cache_store = runtime._v13_cache_store
    response_has_rejected_answer = runtime.response_has_rejected_answer
    if not AI_INTERNAL_SECRET:
        raise HTTPException(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != AI_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    q = str(payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")

    response_language = (
        _root_cause_response_language(q, preferred=payload.language)
        if requested_mode == MODE_ROOT_CAUSE
        else _select_response_language(q, preferred=payload.language)
    )
    root_query_profile = (
        _retrieval_diagnostic_query.analyze_diagnostic_query(
            q, response_language=response_language
        )
        if requested_mode == MODE_ROOT_CAUSE
        else None
    )
    top_default = 8 if requested_mode == MODE_ROOT_CAUSE else 5
    top_k = max(1, min(int(payload.top_k or top_default), ASK_MAX_TOP_K))
    max_causes = max(1, min(int(getattr(payload, "max_causes", 3) or 3), 3))
    budget = _assistant_core_new_budget(requested_mode, company_id=str(payload.company_id or ""))
    token = _V13_BUDGET_CTX.set(budget)
    try:
        scope = _resolve_query_scope(
            company_id=payload.company_id,
            machine_id=payload.machine_id,
            bubble_document_id=payload.bubble_document_id,
            document_ids=payload.document_ids,
            ai_scope=payload.ai_scope,
        )
        company_id = scope["company_id"]
        machine_id = scope["machine_id"]
        budget.company_id = company_id
        doc_ids = scope.get("document_ids") if isinstance(scope.get("document_ids"), list) else None
        bubble_document_id = scope.get("bubble_document_id")
        ai_scope = str(scope.get("ai_scope") or "machine_all")
        narrow_scope = bool(doc_ids or bubble_document_id or ai_scope == "document_ids")
        cache_scope = {**scope, "_v13_top_k": top_k, "_v13_max_causes": max_causes}
        if requested_mode == MODE_ROOT_CAUSE:
            cache_scope["_root_observation_policy"] = _retrieval_diagnostic_query.DIAGNOSTIC_BASIS_POLICY
            cache_scope["_root_review_packet_policy"] = _retrieval_review_packet.POLICY_VERSION
            cache_scope["_root_review_decision_policy"] = _retrieval_review_references.POLICY_VERSION

        # Reuse only an exact request with a complete current-policy interpretation.
        cached = _v13_cache_lookup(
            mode=requested_mode, q=q, company_id=company_id, machine_id=machine_id,
            scope=cache_scope, language=response_language, debug=bool(payload.debug),
        )
        if cached is not None and requested_mode == MODE_ROOT_CAUSE:
            cached_basis = (cached.get("meta") or {}).get(_retrieval_diagnostic_query.BASIS_METADATA_KEY) or {}
            # A semantic neighbour cannot transfer its current-observation proof.
            if (
                cached_basis.get("policy_version") != _retrieval_diagnostic_query.DIAGNOSTIC_BASIS_POLICY
                or cached_basis.get("query_sha256") != _retrieval_diagnostic_query.query_fingerprint(q)
                or cached_basis.get("coverage_complete") is not True
                or cached_basis.get("interpretation_status") != "valid"
                or (cached.get("possible_causes") and
                    ((cached.get("meta") or {}).get("root_review_packet") or {}).get("policy_version")
                    != _retrieval_review_packet.POLICY_VERSION)
                or (requested_mode == MODE_ROOT_CAUSE and cached.get("possible_causes") and
                    ((cached.get("meta") or {}).get("root_review_decisions") or {}).get("policy_version")
                    != _retrieval_review_references.POLICY_VERSION)
            ):
                cached = None
                budget.semantic_cache = "bypass_unvalidated_observation_basis"
        if (
            cached is not None
            and requested_mode == MODE_ASK
            and response_has_rejected_answer(cached)
        ):
            # Reject old cache entries containing a draft vetoed by validation.
            # This does not flush valid entries, alter Worker cache, or change cost caps.
            cached = None
            budget.semantic_cache = "bypass_rejected_answer_contract"
        if cached is not None and str(cached.get("status") or "").strip().lower() == "no_sources":
            cached = None
            budget.semantic_cache = "bypass_negative_answer"
        if cached is not None:
            if (
                requested_mode == MODE_ASK
                and str(cached.get("status") or "").strip().lower() == "no_sources"
            ):
                rescued = _assistant_core_precision_fact_rescue(
                    q=q,
                    company_id=company_id,
                    machine_id=machine_id,
                    doc_ids=doc_ids,
                    bubble_document_id=bubble_document_id,
                    response_language=response_language,
                    top_k=top_k,
                    answer_contract=(cached.get("meta") or {}).get("assistant_core_v2"),
                )
                if rescued is not None:
                    budget.route = "assistant_core_precision_fact_cache_rescue"
                    rescued = _assistant_ui_finalize_response(
                        rescued, language=response_language
                    )
                    return _assistant_core_attach_runtime_meta(
                        rescued, budget, debug=bool(payload.debug)
                    )
            budget.route = "assistant_core_semantic_cache"
            cached = _assistant_ui_finalize_response(cached, language=response_language)
            return _assistant_core_attach_runtime_meta(cached, budget, debug=bool(payload.debug))

        request = AssistantCoreRequest(
            query=q,
            requested_mode=requested_mode,
            response_language=response_language,
            company_id=company_id,
            machine_id=machine_id,
            ai_scope=ai_scope,
            top_k=top_k,
            max_causes=max_causes,
            narrow_scope=narrow_scope,
            allowed_effective_modes=((MODE_ASK,) if requested_mode == MODE_ASK else (MODE_ROOT_CAUSE, MODE_ASK)),
            debug=bool(payload.debug),
            metadata={
                "document_ids": doc_ids,
                "bubble_document_id": bubble_document_id,
                **(
                    {_ROOT_DIAGNOSTIC_QUERY_PROFILE_KEY: root_query_profile.to_dict()}
                    if root_query_profile is not None
                    else {}
                ),
            },
        )
        precision_rescued = False
        final = run_core(request)
        if requested_mode == MODE_ROOT_CAUSE:
            basis_summary = request.metadata.get(_retrieval_diagnostic_query.BASIS_METADATA_KEY)
            if isinstance(basis_summary, dict):
                final = {**dict(final), "meta": {
                    **dict(final.get("meta") or {}),
                    _retrieval_diagnostic_query.BASIS_METADATA_KEY: dict(basis_summary),
                    "assistant_core_diagnostic_query_state": _assistant_core_diagnostic_query_profile(request).public_summary(),
                }}
            router_execution = request.metadata.get(_retrieval_diagnostic_query.ROUTER_EXECUTION_KEY)
            if isinstance(router_execution, dict):
                final_meta = dict(final.get("meta") or {})
                final_meta[_retrieval_diagnostic_query.ROUTER_EXECUTION_KEY] = dict(router_execution)
                core_meta = dict(final_meta.get("assistant_core_v2") or {})
                if router_execution.get("outcome") != "completed":
                    core_meta["router_degraded"] = True
                    core_meta["router_degraded_reason"] = str(router_execution.get("outcome") or "router_unavailable")
                final_meta["assistant_core_v2"] = core_meta
                final = {**dict(final), "meta": final_meta}
        if requested_mode == MODE_ASK:
            scalar_target = request.metadata.get(_retrieval_precision_facts.SCALAR_TARGET_KEY)
            if isinstance(scalar_target, dict):
                final = {**dict(final), "meta": {**dict(final.get("meta") or {}),
                    _retrieval_precision_facts.SCALAR_TARGET_KEY: dict(scalar_target)}}
        if (
            requested_mode == MODE_ASK
            and str(final.get("status") or "").strip().lower() == "no_sources"
        ):
            rescued = _assistant_core_precision_fact_rescue(
                q=q,
                company_id=company_id,
                machine_id=machine_id,
                doc_ids=doc_ids,
                bubble_document_id=bubble_document_id,
                response_language=response_language,
                top_k=top_k,
                answer_contract={**dict((final.get("meta") or {}).get("assistant_core_v2") or {}),
                    _retrieval_precision_facts.SCALAR_TARGET_KEY:
                        (final.get("meta") or {}).get(_retrieval_precision_facts.SCALAR_TARGET_KEY)},
            )
            if rescued is not None:
                target = (final.get("meta") or {}).get(_retrieval_precision_facts.SCALAR_TARGET_KEY)
                if isinstance(target, dict):
                    rescued["meta"] = {**dict(rescued.get("meta") or {}),
                        _retrieval_precision_facts.SCALAR_TARGET_KEY: dict(target)}
                final = rescued
                precision_rescued = True
                budget.route = "assistant_core_precision_fact_rescue"
        final = _assistant_core_clear_unsupported_sources(final)
        final = _assistant_ui_finalize_response(final, language=response_language)
        effective_mode = str(final.get("effective_mode") or requested_mode)
        routed = bool(final.get("routed"))
        if not precision_rescued:
            budget.route = (
                f"assistant_core_{requested_mode}_to_{effective_mode}"
                if routed else f"assistant_core_{effective_mode}"
            )
        final = _assistant_core_attach_runtime_meta(final, budget, debug=bool(payload.debug))
        _v13_cache_store(
            mode=requested_mode,
            q=q,
            company_id=company_id,
            machine_id=machine_id,
            scope=cache_scope,
            language=response_language,
            response=final,
            debug=bool(payload.debug),
        )
        return final
    except _V13BudgetExceeded as exc:
        return _assistant_core_budget_response(
            requested_mode=requested_mode,
            q=q,
            language=response_language,
            top_k=top_k,
            budget=budget,
            exc=exc,
        )
    except HTTPException:
        raise
    except Exception as exc:
        print("ASSISTANT_CORE_REQUEST_FAIL", requested_mode, str(exc)[:1200])
        return _assistant_core_technical_error(
            requested_mode=requested_mode,
            q=q,
            language=response_language,
            budget=budget,
            exc=exc,
        )
    finally:
        _V13_BUDGET_CTX.reset(token)


