"""Existing structured + overview synthesis, with optional request-owned lineage.
No additional semantic pipeline or provider. The extracted default algorithms and
prompts are preserved. Evidence binding is opt-in, supplied only by the owned Core.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Any, Callable, Optional
from .generation import GenericGenerationEvidence, GenerationEvidenceError
from ..evidence.ask_input import _same_value, ask_request_key
from ..evidence.contracts import EvidenceContractError, SourceType
from ..retrieval.ask_composition import AskSelection
from ..retrieval.residual_producers import derive_page_candidates
from ..retrieval import procedure_families as families, structured, candidate_ranking as ranking
from ..retrieval.supplemental_evidence import storage_key

TASK_GENERATION_VERSION = "ask-task-synthesis-evidence-p6b4o-v1"

@dataclass(frozen=True, slots=True, repr=False)
class TaskSynthesisRuntime:
    ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS: Any
    ASK_UI_MAX_POINTS: Any
    ASK_UI_STRUCTURED_MAX_CITATIONS: Any
    INFO_OTHER: Any
    INFO_PROCEDURE_FULL: Any
    INFO_PROCEDURE_SEGMENT: Any
    V13_FAST_EFFORT: Any
    V13_FAST_MAX_OUTPUT_TOKENS: Any
    V13_FAST_MODEL: Any
    V13_FAST_TIMEOUT_SECONDS: Any
    V13_HEAVY_CONTEXT_CHARS: Any
    V13_PLANNER_MODEL: Any
    _V13BudgetExceeded: Any
    _ask_evidence_answer_schema: Any
    _ask_structured_direct_fetch_sources: Any
    _assistant_core_build_machine_overview_answer: Any
    _assistant_core_candidate_source_type: Any
    _assistant_core_machine_overview_schema: Any
    _assistant_core_overview_attach_record: Any
    _assistant_core_overview_fallback_function: Any
    _assistant_core_overview_inventory_records: Any
    _assistant_core_overview_item_text: Any
    _assistant_core_overview_merge_duplicate_items: Any
    _assistant_core_overview_records_block: Any
    _assistant_core_redact_internal_text: Any
    _build_rg_links: Any
    _build_structured_procedure_ui_model: Any
    _clean_display_text: Any
    _content_term_set: Any
    _dedup_text_values: Any
    _finalize_ask_response_for_ui: Any
    _looks_like_target_language: Any
    _procedure_ui_fields: Any
    _procedure_ui_is_safety_setup: Any
    _procedure_ui_merge_sources: Any
    _procedure_ui_model_to_text: Any
    _procedure_ui_order_citations: Any
    _render_grounded_answer_points: Any
    _sanitize_citations_for_response: Any
    _source_type_from_document_id: Any
    _structured_rescue_query_intent: Any
    _term_overlap_score: Any
    _v12_choose_primary_procedure: Any
    _v12_curate_response_items_for_ui: Any
    _v12_curate_structured_sources: Any
    _v12_evidence_role: Any
    _v12_filter_manual_support_to_selected_bundle: Any
    _v12_mark_structured_roles: Any
    _v12_procedure_selection_mode: Any
    _v12_select_response_steps: Any
    _v12_step_sort_key: Any
    _v13_assurance_prompt_block: Any
    _v13_fetch_manual_support_deterministic: Any
    _v13_json_models: Any
    _v13_merge_candidates: Any
    _v13_sources_block: Any
    family: families.V12ChoosePrimaryProcedureFamilyRuntime
    curation: families.V12CurateStructuredSourcesRuntime
    expansion: structured.V12ExpandPrimaryProcedureStepsRuntime
    step_dedupe: ranking.V12DedupeFamilyStepsRuntime
    metadata_merge: ranking.V12MergeCandidateMetadataRuntime
    manual_filter: families.V12FilterManualSupportToSelectedBundleRuntime
    roles: families.V12MarkStructuredRolesRuntime
    copy_fn: Any = None

def structured_ask(
    *,
    q: str,
    company_id: str,
    machine_id: str,
    response_language: str,
    top_k: int,
    planner: dict,
    seed_citations: Optional[list[dict]] = None,
    assurance_meta: Optional[dict] = None,
    debug: bool,
    runtime: TaskSynthesisRuntime,
) -> Optional[dict]:
    ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS
    ASK_UI_MAX_POINTS = runtime.ASK_UI_MAX_POINTS
    ASK_UI_STRUCTURED_MAX_CITATIONS = runtime.ASK_UI_STRUCTURED_MAX_CITATIONS
    INFO_OTHER = runtime.INFO_OTHER
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    V13_FAST_EFFORT = runtime.V13_FAST_EFFORT
    V13_FAST_MAX_OUTPUT_TOKENS = runtime.V13_FAST_MAX_OUTPUT_TOKENS
    V13_FAST_MODEL = runtime.V13_FAST_MODEL
    V13_FAST_TIMEOUT_SECONDS = runtime.V13_FAST_TIMEOUT_SECONDS
    V13_HEAVY_CONTEXT_CHARS = runtime.V13_HEAVY_CONTEXT_CHARS
    V13_PLANNER_MODEL = runtime.V13_PLANNER_MODEL
    _V13BudgetExceeded = runtime._V13BudgetExceeded
    _ask_evidence_answer_schema = runtime._ask_evidence_answer_schema
    _ask_structured_direct_fetch_sources = runtime._ask_structured_direct_fetch_sources
    _assistant_core_build_machine_overview_answer = runtime._assistant_core_build_machine_overview_answer
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_machine_overview_schema = runtime._assistant_core_machine_overview_schema
    _assistant_core_overview_attach_record = runtime._assistant_core_overview_attach_record
    _assistant_core_overview_fallback_function = runtime._assistant_core_overview_fallback_function
    _assistant_core_overview_inventory_records = runtime._assistant_core_overview_inventory_records
    _assistant_core_overview_item_text = runtime._assistant_core_overview_item_text
    _assistant_core_overview_merge_duplicate_items = runtime._assistant_core_overview_merge_duplicate_items
    _assistant_core_overview_records_block = runtime._assistant_core_overview_records_block
    _assistant_core_redact_internal_text = runtime._assistant_core_redact_internal_text
    _build_rg_links = runtime._build_rg_links
    _build_structured_procedure_ui_model = runtime._build_structured_procedure_ui_model
    _clean_display_text = runtime._clean_display_text
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _finalize_ask_response_for_ui = runtime._finalize_ask_response_for_ui
    _looks_like_target_language = runtime._looks_like_target_language
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_is_safety_setup = runtime._procedure_ui_is_safety_setup
    _procedure_ui_merge_sources = runtime._procedure_ui_merge_sources
    _procedure_ui_model_to_text = runtime._procedure_ui_model_to_text
    _procedure_ui_order_citations = runtime._procedure_ui_order_citations
    _render_grounded_answer_points = runtime._render_grounded_answer_points
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    _source_type_from_document_id = runtime._source_type_from_document_id
    _structured_rescue_query_intent = runtime._structured_rescue_query_intent
    _term_overlap_score = runtime._term_overlap_score
    _v12_choose_primary_procedure = runtime._v12_choose_primary_procedure
    _v12_curate_response_items_for_ui = runtime._v12_curate_response_items_for_ui
    _v12_curate_structured_sources = runtime._v12_curate_structured_sources
    _v12_evidence_role = runtime._v12_evidence_role
    _v12_filter_manual_support_to_selected_bundle = runtime._v12_filter_manual_support_to_selected_bundle
    _v12_mark_structured_roles = runtime._v12_mark_structured_roles
    _v12_procedure_selection_mode = runtime._v12_procedure_selection_mode
    _v12_select_response_steps = runtime._v12_select_response_steps
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v13_assurance_prompt_block = runtime._v13_assurance_prompt_block
    _v13_fetch_manual_support_deterministic = runtime._v13_fetch_manual_support_deterministic
    _v13_json_models = runtime._v13_json_models
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_sources_block = runtime._v13_sources_block
    _copy_candidate = dict if runtime.copy_fn is None else runtime.copy_fn
    structured_types = {"procedure", "step", "ps", "md_photo", "md_video"}
    raw = [
        _copy_candidate(c) for c in (seed_citations or [])
        if isinstance(c, dict)
        and str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")) in structured_types
    ]

    # Explicit listing/source requests may require records that are not semantically
    # similar to a single operation. Merge the bounded deterministic structured scan.
    if _structured_rescue_query_intent(q, planner):
        try:
            direct = _ask_structured_direct_fetch_sources(
                company_id=company_id,
                machine_id=machine_id,
                q=q,
                planner=planner,
                top_k=max(10, top_k),
            )
            raw = _v13_merge_candidates([raw, direct])
        except Exception as exc:
            print("V13_STRUCTURED_FETCH_FAIL", str(exc)[:600])

    if not raw:
        return None

    structured = _v12_curate_structured_sources(
        company_id=company_id,
        machine_id=machine_id,
        q=q,
        planner=planner,
        citations=raw,
        model_used=[],
    )
    structured = _v12_mark_structured_roles(structured)
    if not structured:
        return None

    information_task = str((planner or {}).get("information_task") or INFO_OTHER).strip().lower()
    procedure_sequence_mode = information_task in {
        INFO_PROCEDURE_FULL,
        INFO_PROCEDURE_SEGMENT,
    }
    structured_scope = list(structured)
    selected_step_numbers: list[int] = []
    expanded_step_numbers: list[int] = []

    if procedure_sequence_mode:
        primary = _v12_choose_primary_procedure(structured, [])
        complete_steps = sorted(
            [_copy_candidate(c) for c in structured if _v12_evidence_role(c) == "step"],
            key=_v12_step_sort_key,
        )
        expanded_step_numbers = [
            _v12_step_sort_key(c)[0] for c in complete_steps
            if 0 < _v12_step_sort_key(c)[0] < 9999
        ]
        if primary is None or not complete_steps:
            # A Procedure title/parent may be absent from semantic top-k. Family
            # recovery above normally restores it from Step relations; if no single
            # family can be proven, return None so the caller can perform the normal
            # grounded multi-source procedural synthesis instead of a false bundle
            # error.
            return None

        if information_task == INFO_PROCEDURE_FULL:
            selected_steps = complete_steps
        else:
            selected_steps = _v12_select_response_steps(
                all_steps=complete_steps,
                selected_step_ids=[],
                model_used_citations=[],
                q=q,
                planner=planner,
            )
        if not selected_steps:
            return None

        selected_step_numbers = [
            _v12_step_sort_key(c)[0] for c in selected_steps
            if 0 < _v12_step_sort_key(c)[0] < 9999
        ]
        selected_number_set = set(selected_step_numbers)
        first_selected = min(selected_number_set or {9999})
        safety_prerequisites: list[dict] = []
        for candidate in complete_steps:
            number = _v12_step_sort_key(candidate)[0]
            fields = _procedure_ui_fields(candidate)
            safety_text = " ".join([
                str(fields.get("title") or ""),
                str(fields.get("description") or ""),
            ])
            if (
                number < first_selected
                and number not in selected_number_set
                and _procedure_ui_is_safety_setup(safety_text)
            ):
                safety_prerequisites = [_copy_candidate(candidate)]
                break

        extras = [
            _copy_candidate(c) for c in structured
            if _v12_evidence_role(c) not in {"procedure", "step"}
        ]
        structured_scope = _v12_mark_structured_roles(
            [_copy_candidate(primary)] + safety_prerequisites + list(selected_steps) + extras
        )

    manual_support = _v13_fetch_manual_support_deterministic(
        company_id=company_id,
        machine_id=machine_id,
        q=q,
        planner=planner,
        structured_citations=structured_scope,
    )
    if procedure_sequence_mode:
        manual_support = _v12_filter_manual_support_to_selected_bundle(
            q=q,
            structured_citations=structured_scope,
            manual_support_citations=manual_support,
        )
    all_evidence = list(structured_scope) + list(manual_support)
    sources_block = _v13_sources_block(
        all_evidence,
        max_context_chars=min(V13_HEAVY_CONTEXT_CHARS, max(16000, ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS)),
    )
    if not sources_block:
        return None

    system_msg = (
        "You are MachineMind ASK. Use only the supplied evidence. Structured procedures and their explicitly related ordered steps are primary. "
        "Never mix steps from different procedure families. The supplied Step set is authoritative: it may be a contiguous operation span or a sparse ordered checklist of conditions. Preserve its order, do not invent missing intermediate Steps, and include all supplied conditions and warnings needed to answer the request. "
        "Manual-support sources are secondary: use them only for directly relevant operating detail, prerequisite or safety context, and keep their citations so the manual appears among the links. "
        "For P&S records, report problem, solution and notes. For photo/video records, use title/description metadata only and never claim visual or audio inspection. "
        "Every visible point must be grounded in citation_ids from SOURCES. Do not expose raw ids in text. Reply in the requested language."
    )
    assurance_block = _v13_assurance_prompt_block({"retrieval_assurance": dict(assurance_meta or {})})
    user_msg = (
        f"QUESTION:\n{q}\n\nRESPONSE_LANGUAGE: {response_language}\n\nSOURCES:\n{sources_block}\n\n"
        + (f"{assurance_block}\n\n" if assurance_block else "")
        + "Return JSON only. Answer from the selected structured procedure/steps first, then add a brief directly applicable manual support or safety note when present."
    )

    model = V13_FAST_MODEL
    effort = V13_FAST_EFFORT
    mode = ""
    timeout = V13_FAST_TIMEOUT_SECONDS
    output_tokens = V13_FAST_MAX_OUTPUT_TOKENS

    parsed: dict = {}
    model_used = model
    try:
        parsed, model_used = _v13_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[model],
            json_schema=_ask_evidence_answer_schema(),
            effort=effort,
            reasoning_mode=mode,
            timeout=timeout,
            max_output_tokens=output_tokens,
            company_id=company_id,
            purpose="ask_structured_synthesis",
        )
    except _V13BudgetExceeded:
        raise
    except Exception as exc:
        print("V13_STRUCTURED_SYNTHESIS_FAIL", str(exc)[:700])
        parsed = {"answer_status": "no_sources", "grounded_points": []}

    grounded_points = list(parsed.get("grounded_points") or [])
    model_answer, model_citations = _render_grounded_answer_points(
        grounded_points=grounded_points,
        citations=all_evidence,
        max_points=max(1, int(ASK_UI_MAX_POINTS or 5)),
        q=q,
    )

    if procedure_sequence_mode:
        # The deterministic Procedure span is authoritative. Do not call the
        # generic curator again here, because it expands the parent Procedure back
        # to every Step and would undo a valid partial selection.
        final_structured = _v12_mark_structured_roles(structured_scope)
    else:
        final_structured = _v12_curate_structured_sources(
            company_id=company_id,
            machine_id=machine_id,
            q=q,
            planner=planner,
            citations=structured_scope,
            model_used=model_citations,
        )
        final_structured = _v12_mark_structured_roles(final_structured)
    has_procedure_context = any(
        _v12_evidence_role(c) in {"procedure", "step"}
        for c in final_structured
        if isinstance(c, dict)
    )
    if not has_procedure_context:
        used_ids = {
            str(c.get("citation_id") or "").strip()
            for c in (model_citations or [])
            if isinstance(c, dict) and str(c.get("citation_id") or "").strip()
        }
        manual_support = [
            c for c in manual_support
            if str(c.get("citation_id") or "").strip() in used_ids
        ]
    ui_structured = _procedure_ui_merge_sources(
        structured_scope,
        final_structured,
        model_citations,
    )
    answer_ui_model = _build_structured_procedure_ui_model(
        structured_citations=ui_structured,
        manual_support_citations=manual_support,
        grounded_points=grounded_points,
        response_language=response_language,
        q=q,
    )
    sectioned_answer = _procedure_ui_model_to_text(
        answer_ui_model,
        response_language=response_language,
    )
    synthesis_grounded = bool(model_answer and model_citations)
    answer = sectioned_answer or model_answer
    if not answer:
        return None

    # Keep one complete ordered Procedure/Step family, then secondary evidence.
    model_extras = [
        c for c in (model_citations or [])
        if isinstance(c, dict) and _v12_evidence_role(c) not in {"procedure", "step"}
    ]
    final_citations = _v12_curate_response_items_for_ui(
        _procedure_ui_order_citations(
            list(ui_structured) + list(manual_support) + model_extras
        ),
        max_items=max(1, int(ASK_UI_STRUCTURED_MAX_CITATIONS or 14)),
    )
    if not final_citations:
        return None

    response_citations = _sanitize_citations_for_response(final_citations, company_id=company_id)
    role_by_id = {
        str(c.get("citation_id") or ""): {
            "evidence_role": _v12_evidence_role(c),
            "ask_structured_direct": bool(c.get("ask_structured_direct")),
            "ask_structured_manual_support": bool(c.get("ask_structured_manual_support")),
            "ask_manual_support_kind": str(c.get("ask_manual_support_kind") or ""),
            "exact_machine_scope": bool(c.get("exact_machine_scope")),
        }
        for c in final_citations
    }
    for c in response_citations:
        c.update(role_by_id.get(str(c.get("citation_id") or ""), {}))
    response_citations = _procedure_ui_order_citations(response_citations)

    try:
        rg_links = _procedure_ui_order_citations(
            _build_rg_links(company_id, response_citations)
        )
    except Exception as exc:
        print("RG_LINKS_FAIL", str(exc)[:500])
        rg_links = []

    resp = {
        "ok": True,
        "status": "answered",
        "answer": answer,
        "language": response_language,
        "citations": response_citations,
        "rg_links": rg_links,
        "top_k": top_k,
        "similarity_max": max([float(c.get("similarity") or 0.0) for c in all_evidence], default=None),
        "chat_model": model_used if synthesis_grounded else "v13_deterministic_structured_fallback",
        "_assistant_ui_model": answer_ui_model,
        # Internal, trusted evidence manifest. These citations are loaded and
        # curated deterministically by the backend (including Procedure->Step
        # expansion) and must survive Assistant Core validation even when they
        # were not part of the original semantic retrieval top-k.
        "_assistant_core_validation_evidence": [_copy_candidate(c) for c in final_citations],
        # Sparse procedural checklists combine non-contiguous conditions. Their
        # semantic completeness cannot be judged reliably by phrase overlap alone,
        # so the existing bounded third-call verifier is enabled only for this mode.
        "_assistant_core_force_semantic_verify": bool(
            procedure_sequence_mode
            and _v12_procedure_selection_mode(q, planner) == "sparse_ordered_steps"
        ),
        "meta": (
            {"cacheable": True, "semantic_cacheable": True}
            if synthesis_grounded
            else {
                "cacheable": False,
                "semantic_cacheable": False,
                "degraded": True,
                "degraded_reason": "structured_deterministic_fallback",
            }
        ),
    }
    if debug:
        resp["debug"] = {
            "v13_structured": {
                "raw_sources": len(raw),
                "structured_sources": len(final_structured),
                "manual_support_sources": len(manual_support),
                "manual_support_links": sum(1 for x in rg_links if str(x.get("evidence_role") or "") == "manual_support"),
                "procedure_sequence_mode": bool(procedure_sequence_mode),
                "information_task": information_task,
                "procedure_selection_mode": _v12_procedure_selection_mode(q, planner),
                "procedure_family": dict((primary or {}).get("_v10_5_family_debug") or {}) if procedure_sequence_mode else {},
                "expanded_step_numbers": expanded_step_numbers,
                "selected_step_numbers": selected_step_numbers,
            }
        }
    return _finalize_ask_response_for_ui(resp, language=response_language)

def synthesize_machine_overview(
    request: AssistantCoreRequest,
    retrieval: dict,
    decision: AssistantCoreDecision,
    *, runtime: TaskSynthesisRuntime,
) -> dict | None:
    """Exhaustive overview path with deterministic source accounting.

    This path replaces free-form overview generation only when the evidence stage
    explicitly requested the machine catalog. The model may merge and translate
    source records, but deterministic post-processing appends any record it failed
    to represent. Therefore a low lexical score cannot silently remove a documented
    assembly or auxiliary system.
    """
    ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS = runtime.ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS
    ASK_UI_MAX_POINTS = runtime.ASK_UI_MAX_POINTS
    ASK_UI_STRUCTURED_MAX_CITATIONS = runtime.ASK_UI_STRUCTURED_MAX_CITATIONS
    INFO_OTHER = runtime.INFO_OTHER
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    V13_FAST_EFFORT = runtime.V13_FAST_EFFORT
    V13_FAST_MAX_OUTPUT_TOKENS = runtime.V13_FAST_MAX_OUTPUT_TOKENS
    V13_FAST_MODEL = runtime.V13_FAST_MODEL
    V13_FAST_TIMEOUT_SECONDS = runtime.V13_FAST_TIMEOUT_SECONDS
    V13_HEAVY_CONTEXT_CHARS = runtime.V13_HEAVY_CONTEXT_CHARS
    V13_PLANNER_MODEL = runtime.V13_PLANNER_MODEL
    _V13BudgetExceeded = runtime._V13BudgetExceeded
    _ask_evidence_answer_schema = runtime._ask_evidence_answer_schema
    _ask_structured_direct_fetch_sources = runtime._ask_structured_direct_fetch_sources
    _assistant_core_build_machine_overview_answer = runtime._assistant_core_build_machine_overview_answer
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_machine_overview_schema = runtime._assistant_core_machine_overview_schema
    _assistant_core_overview_attach_record = runtime._assistant_core_overview_attach_record
    _assistant_core_overview_fallback_function = runtime._assistant_core_overview_fallback_function
    _assistant_core_overview_inventory_records = runtime._assistant_core_overview_inventory_records
    _assistant_core_overview_item_text = runtime._assistant_core_overview_item_text
    _assistant_core_overview_merge_duplicate_items = runtime._assistant_core_overview_merge_duplicate_items
    _assistant_core_overview_records_block = runtime._assistant_core_overview_records_block
    _assistant_core_redact_internal_text = runtime._assistant_core_redact_internal_text
    _build_rg_links = runtime._build_rg_links
    _build_structured_procedure_ui_model = runtime._build_structured_procedure_ui_model
    _clean_display_text = runtime._clean_display_text
    _content_term_set = runtime._content_term_set
    _dedup_text_values = runtime._dedup_text_values
    _finalize_ask_response_for_ui = runtime._finalize_ask_response_for_ui
    _looks_like_target_language = runtime._looks_like_target_language
    _procedure_ui_fields = runtime._procedure_ui_fields
    _procedure_ui_is_safety_setup = runtime._procedure_ui_is_safety_setup
    _procedure_ui_merge_sources = runtime._procedure_ui_merge_sources
    _procedure_ui_model_to_text = runtime._procedure_ui_model_to_text
    _procedure_ui_order_citations = runtime._procedure_ui_order_citations
    _render_grounded_answer_points = runtime._render_grounded_answer_points
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    _source_type_from_document_id = runtime._source_type_from_document_id
    _structured_rescue_query_intent = runtime._structured_rescue_query_intent
    _term_overlap_score = runtime._term_overlap_score
    _v12_choose_primary_procedure = runtime._v12_choose_primary_procedure
    _v12_curate_response_items_for_ui = runtime._v12_curate_response_items_for_ui
    _v12_curate_structured_sources = runtime._v12_curate_structured_sources
    _v12_evidence_role = runtime._v12_evidence_role
    _v12_filter_manual_support_to_selected_bundle = runtime._v12_filter_manual_support_to_selected_bundle
    _v12_mark_structured_roles = runtime._v12_mark_structured_roles
    _v12_procedure_selection_mode = runtime._v12_procedure_selection_mode
    _v12_select_response_steps = runtime._v12_select_response_steps
    _v12_step_sort_key = runtime._v12_step_sort_key
    _v13_assurance_prompt_block = runtime._v13_assurance_prompt_block
    _v13_fetch_manual_support_deterministic = runtime._v13_fetch_manual_support_deterministic
    _v13_json_models = runtime._v13_json_models
    _v13_merge_candidates = runtime._v13_merge_candidates
    _v13_sources_block = runtime._v13_sources_block
    _copy_candidate = dict if runtime.copy_fn is None else runtime.copy_fn
    records = _assistant_core_overview_inventory_records(retrieval)
    catalog_records = [record for record in records if bool(record.get("must_account"))]
    manual_records = [record for record in records if str(record.get("source_type") or "") == "document"]
    if not catalog_records:
        return None

    record_by_inventory = {str(record.get("inventory_id") or ""): record for record in records}
    candidate_by_citation = {
        str(record.get("citation_id") or ""): _copy_candidate(record.get("candidate") or {})
        for record in records
        if str(record.get("citation_id") or "").strip()
    }
    block = _assistant_core_overview_records_block(records)
    parsed: dict = {}
    model_used = "deterministic_overview_fallback"
    if block:
        system_msg = (
            "You are MachineMind's machine-overview inventory compiler. Use only SOURCE_INVENTORY. "
            "Return an exhaustive but concise machine overview in RESPONSE_LANGUAGE. Merge synonyms and duplicate views into distinct user-facing assemblies or systems. "
            "Every row marked MUST_ACCOUNT=yes must be represented by at least one overview item through its INVENTORY_ID. This does not require one item per source: multiple source rows may support one item. "
            "Do not discard an auxiliary system merely because its wording scores weakly against the question. Include physical assemblies, material-flow groups, clamping/feed functions, tooling, HMI/control interfaces, lubrication/pneumatic/ventilation or other explicitly documented auxiliary systems, protections and outfeed when present. "
            "Procedure records are evidence about the underlying assembly or system, not instructions to expose internal procedure codes. Cite only supplied CITATION_ID values. Do not claim direct visual inspection of media."
        )
        user_msg = (
            f"QUESTION:\n{request.query}\n\n"
            f"RESPONSE_LANGUAGE: {request.response_language}\n\n"
            f"SOURCE_INVENTORY:\n{block}\n\n"
            "Return JSON only. The function summary must be grounded in at least one technical document when available. Every overview item must have valid INVENTORY_ID and CITATION_ID values."
        )
        try:
            parsed, model_used = _v13_json_models(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                models=[V13_FAST_MODEL, V13_PLANNER_MODEL],
                json_schema=_assistant_core_machine_overview_schema(),
                effort=V13_FAST_EFFORT,
                reasoning_mode="",
                timeout=min(28, V13_FAST_TIMEOUT_SECONDS),
                max_output_tokens=min(5200, V13_FAST_MAX_OUTPUT_TOKENS),
                company_id=request.company_id,
                purpose="assistant_core_machine_overview_inventory",
            )
        except _V13BudgetExceeded:
            parsed = {}
        except Exception as exc:
            print("ASSISTANT_CORE_OVERVIEW_INVENTORY_FAIL", str(exc)[:700])
            parsed = {}

    valid_inventory_ids = set(record_by_inventory)
    valid_citation_ids = set(candidate_by_citation)
    items: list[dict] = []
    used_inventory_ids: set[str] = set()
    used_citation_ids: list[str] = []
    rejected_unsupported_model_items = 0

    for raw in (parsed.get("overview_items") or []):
        if not isinstance(raw, dict):
            continue
        inventory_ids = [
            str(iid or "").strip()
            for iid in (raw.get("inventory_ids") or [])
            if str(iid or "").strip() in valid_inventory_ids
        ]
        citation_ids = [
            str(cid or "").strip()
            for cid in (raw.get("citation_ids") or [])
            if str(cid or "").strip() in valid_citation_ids
        ]
        # Derive citations from assigned inventory rows when the model omitted or
        # mistyped the long citation id but correctly identified the source row.
        for iid in inventory_ids:
            cid = str(record_by_inventory[iid].get("citation_id") or "").strip()
            if cid and cid not in citation_ids:
                citation_ids.append(cid)
        label = _clean_display_text(raw.get("label") or "", max_len=160)
        description = _clean_display_text(raw.get("description") or "", max_len=620)
        if not label or not citation_ids or not inventory_ids:
            # Every model item must identify at least one admitted inventory row.
            # This prevents unsupported decorative groups from competing with the
            # deterministic source-accounted items.
            rejected_unsupported_model_items += 1
            continue
        assigned_text = " ".join(
            " ".join(
                [
                    str(record_by_inventory[iid].get("title") or ""),
                    str(record_by_inventory[iid].get("description") or ""),
                ]
            )
            for iid in inventory_ids
            if iid in record_by_inventory
        ).strip()
        item_text_for_support = " ".join([label, description]).strip()
        same_language_support = bool(
            _looks_like_target_language(assigned_text, request.response_language)
            and _looks_like_target_language(item_text_for_support, request.response_language)
        )
        lexical_support = _term_overlap_score(
            _content_term_set(assigned_text, limit=220),
            _content_term_set(item_text_for_support, limit=180),
        )
        if same_language_support and lexical_support < 0.035:
            # The item points to real source ids but its visible claim is unrelated
            # to those sources. Drop it; deterministic accounting will append the
            # clean admitted records instead. Cross-language translations are not
            # rejected by this lexical guard.
            rejected_unsupported_model_items += 1
            continue
        item = {
            "label": label,
            "description": description,
            "kind": str(raw.get("kind") or "other"),
            "inventory_ids": _dedup_text_values(inventory_ids, limit=24),
            "citation_ids": _dedup_text_values(citation_ids, limit=16),
        }
        items.append(item)
        used_inventory_ids.update(item["inventory_ids"])
        for cid in item["citation_ids"]:
            if cid not in used_citation_ids:
                used_citation_ids.append(cid)

    # Build document-frequency statistics for source terminology. They let the
    # deterministic accounting check a few distinctive system terms instead of
    # requiring lexical overlap with every administrative word in a Procedure.
    from collections import Counter as _OverviewCounter
    catalog_term_df = _OverviewCounter()
    catalog_terms_by_id: dict[str, set[str]] = {}
    for source_record in catalog_records:
        source_text = " ".join(
            [
                str(source_record.get("title") or ""),
                str(source_record.get("description") or ""),
            ]
        ).strip()
        terms = set(_content_term_set(source_text, limit=120))
        catalog_terms_by_id[str(source_record.get("inventory_id") or "")] = terms
        catalog_term_df.update(terms)

    # The model is never allowed to be the sole completeness gate. Attach every
    # catalog record that it did not account for, and append the source wording
    # when an assigned item failed to carry the record's distinctive terminology.
    for record in catalog_records:
        iid = str(record.get("inventory_id") or "")
        if iid not in used_inventory_ids:
            _assistant_core_overview_attach_record(items, record)
        else:
            # Even an accounted source may have been attached to a semantically
            # empty item. Preserve its distinctive source wording in the same item.
            matched = [item for item in items if iid in (item.get("inventory_ids") or [])]
            if matched:
                record_text = " ".join(
                    [str(record.get("title") or ""), str(record.get("description") or "")]
                ).strip()
                record_terms = _content_term_set(record_text, limit=120)
                combined_item_text = " ".join(
                    _assistant_core_overview_item_text(item) for item in items
                )
                combined_terms = set(
                    _content_term_set(combined_item_text, limit=420)
                )
                iid_terms = catalog_terms_by_id.get(iid, record_terms)
                # Rare terms are the most useful evidence that the underlying
                # system/function really appears in the overview. Prioritize title
                # terms, then globally rare terms from the short description.
                title_terms = set(
                    _content_term_set(str(record.get("title") or ""), limit=50)
                )
                distinctive = sorted(
                    [
                        term
                        for term in iid_terms
                        if len(term) >= 5 and catalog_term_df.get(term, 0) <= 2
                    ],
                    key=lambda term: (
                        0 if term in title_terms else 1,
                        catalog_term_df.get(term, 99),
                        -len(term),
                        term,
                    ),
                )[:7]
                same_language = _looks_like_target_language(
                    record_text, request.response_language
                )
                if same_language and distinctive:
                    hits = sum(1 for term in distinctive if term in combined_terms)
                    needed = 1 if len(distinctive) <= 2 else 2
                    represented = hits >= needed
                else:
                    representation = _term_overlap_score(
                        record_terms, combined_terms
                    )
                    represented = representation >= 0.18

                if not represented:
                    # Preserve the missing source wording in the best matching
                    # model item when it fits without truncation. If one broad item
                    # claimed many inventory ids, fall back to a separate source
                    # item before any later record can be erased by a length cap.
                    target = max(
                        matched,
                        key=lambda item: _term_overlap_score(
                            record_terms,
                            _content_term_set(
                                _assistant_core_overview_item_text(item), limit=180
                            ),
                        ),
                    )
                    support = _clean_display_text(
                        record.get("description") or record.get("title") or "",
                        max_len=520,
                    )
                    current = str(target.get("description") or "").strip()
                    proposed = (current + "; " + support).strip(" ;")
                    claimed_ids = len(target.get("inventory_ids") or [])
                    if support and claimed_ids <= 4 and len(proposed) <= 560:
                        target["description"] = proposed
                    else:
                        _assistant_core_overview_attach_record(items, record)

    items = _assistant_core_overview_merge_duplicate_items(items)

    function_summary = _assistant_core_redact_internal_text(
        parsed.get("function_summary") or ""
    )
    function_citation_ids = [
        str(cid or "").strip()
        for cid in (parsed.get("function_citation_ids") or [])
        if str(cid or "").strip() in valid_citation_ids
    ]
    fallback_summary, fallback_cids = _assistant_core_overview_fallback_function(
        records,
        query=request.query,
        language=request.response_language,
    )
    if not function_summary:
        function_summary = fallback_summary
    if not function_citation_ids:
        function_citation_ids = list(fallback_cids)

    # A machine overview should retain one technical document whenever the admitted
    # pack contains one. This is a source-diversity requirement, not a keyword rule.
    if manual_records and not any(
        str(record.get("citation_id") or "") in function_citation_ids
        for record in manual_records
    ):
        top_manual_id = str(manual_records[0].get("citation_id") or "").strip()
        if top_manual_id:
            function_citation_ids.insert(0, top_manual_id)

    for cid in function_citation_ids:
        if cid and cid not in used_citation_ids:
            used_citation_ids.insert(0, cid)
    for item in items:
        for cid in item.get("citation_ids") or []:
            if cid and cid not in used_citation_ids:
                used_citation_ids.append(cid)

    # Preserve at least one media and one structured/system source when present.
    for family in (
        {"md_photo", "md_video", "photo", "video"},
        {"procedure", "step", "ps"},
    ):
        if any(
            _assistant_core_candidate_source_type(candidate_by_citation.get(cid, {})) in family
            for cid in used_citation_ids
        ):
            continue
        for record in records:
            if str(record.get("source_type") or "") in family:
                cid = str(record.get("citation_id") or "").strip()
                if cid and cid not in used_citation_ids:
                    used_citation_ids.append(cid)
                break

    if not function_summary:
        return None
    answer = _assistant_core_build_machine_overview_answer(
        function_summary=function_summary,
        items=items,
        language=request.response_language,
    )
    if not answer or not items:
        return None

    used_candidates: list[dict] = []
    for cid in used_citation_ids:
        candidate = candidate_by_citation.get(cid)
        if not candidate:
            continue
        cc = _copy_candidate(candidate)
        st = _assistant_core_candidate_source_type(cc)
        if st == "document":
            cc.setdefault("evidence_role", "manual_support")
            cc.setdefault("ask_structured_manual_support", True)
        else:
            cc.setdefault("evidence_role", st)
            cc.setdefault("ask_structured_direct", True)
        used_candidates.append(cc)

    expected_items = [str(item.get("label") or "").strip() for item in items if str(item.get("label") or "").strip()]
    semantic_contract = {
        "outcome": "pass",
        "answer": answer,
        "covered_facets": list(decision.required_facets),
        "missing_facets": [],
        "covered_answer_types": list(decision.required_answer_types),
        "missing_answer_types": [],
        "enumeration_requested": True,
        "expected_list_items": expected_items,
        "covered_list_items": expected_items,
        "missing_list_items": [],
        "citation_ids": [str(c.get("citation_id") or "") for c in used_candidates],
        "reason": "deterministic_machine_overview_inventory_complete",
        "model": model_used,
    }
    return {
        "ok": True,
        "status": "answered",
        "answer": answer,
        "language": request.response_language,
        "citations": used_candidates,
        "rg_links": [],
        "top_k": request.top_k,
        "similarity_max": (retrieval.get("metrics") or {}).get("top_similarity"),
        "chat_model": model_used,
        "information_task": decision.information_task,
        "_assistant_core_semantic_verified": semantic_contract,
        "_assistant_core_validation_evidence": used_candidates,
        "meta": {
            "cacheable": True,
            "semantic_cacheable": True,
            "machine_overview_inventory": {
                "enabled": True,
                "version": "machine-overview-inventory-v1",
                "record_count": len(records),
                "catalog_record_count": len(catalog_records),
                "manual_record_count": len(manual_records),
                "item_count": len(items),
                "all_catalog_records_accounted": all(
                    str(record.get("inventory_id") or "")
                    in {
                        iid
                        for item in items
                        for iid in (item.get("inventory_ids") or [])
                    }
                    for record in catalog_records
                ),
                "model": model_used,
                "rejected_unsupported_model_items": int(
                    rejected_unsupported_model_items
                ),
            },
        },
    }

class TaskGenerationEvidence(GenericGenerationEvidence):
    """Structured and overview consumers on the SAME caller-owned session.

    Candidate-copy sites explicitly declare their parent object. Metadata-only
    drafts are sealed before another consumer/read/model or final output. Source
    identity, locator and evidence text cannot be rewritten through draft copies.
    This journal is a trusted-code provenance contract, not a Python sandbox.
    """
    _BODY = ("citation_id", "bubble_document_id", "company_id", "machine_id",
             "page_from", "page_to", "page_number", "chunk_index", "snippet",
             "snippet_clean", "chunk_full", "text", "source_type")

    def __init__(self, *, tasks, **kwargs):
        super().__init__(**kwargs)
        if type(tasks) is not TaskSynthesisRuntime:
            raise GenerationEvidenceError("explicit task synthesis runtime required")
        self.tasks = tasks
        self.drafts, self.raw_refs, self.inventory_refs = {}, {}, {}
        self.all_handles, self.validation_handles = (), ()
        self.model_handles = None
        self.kind = None
        self.trace_bytes = 0

    def _bounded(self, value):
        import json
        self.trace_bytes += len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        if (len(self.refs) + len(self.drafts) + len(self.raw_refs) >= 16384
                or self.trace_bytes > 64 * 1024 * 1024):
            raise GenerationEvidenceError("bounded task synthesis journal exceeded")

    def _remember(self, row, handle):
        self._bounded(row)
        self._bind(row, handle)
        self.all_handles += (handle,)
        return row

    def _draft(self, row, handle, *, body):
        self._bounded(row)
        if id(row) in self.refs or id(row) in self.drafts:
            raise GenerationEvidenceError("duplicate task copy identity")
        self.drafts[id(row)] = (row, body, handle)
        return row

    def copy(self, row):
        self.check()
        link = self.link_refs.get(id(row))
        if link is not None:
            if link[0] is not row or not _same_value(row, link[1]):
                raise GenerationEvidenceError("modified task link")
            view = deepcopy(row)
            self._bounded(view)
            self.link_refs[id(view)] = (view, deepcopy(view))
            return view
        parent = self.handles([row])[0]
        view = deepcopy(row)
        return self._draft(view, parent, body={k: deepcopy(row[k]) for k in self._BODY if k in row})

    def handles(self, rows):
        self.check()
        if type(rows) is not list:
            raise GenerationEvidenceError("explicit task record list required")
        for row in rows:
            draft = self.drafts.get(id(row))
            if draft is not None:
                if draft[0] is not row or not _same_value(
                        {k: row[k] for k in self._BODY if k in row}, draft[1]):
                    raise GenerationEvidenceError("task copy changed source body or locator")
                handle = self.session.derive_batch(request=self.request,
                    views=(((draft[2],), deepcopy(row)),),
                    operation=TASK_GENERATION_VERSION+":metadata-copy",
                    layout="retrieval_candidate", current_allowed_sources=self.current())[0]
                self.drafts.pop(id(row))
                self._remember(row, handle)
        return super().handles(rows)

    def _materialize(self, handles):
        result = []
        for handle, item in zip(handles, self.records(handles)):
            if item.layout != "retrieval_candidate":
                raise GenerationEvidenceError("raw task page needs explicit conversion")
            result.append(self._remember(deepcopy(dict(item.record)), handle))
        return result

    def _scope_parameters(self, parameters):
        parameters = dict(parameters)
        for key, expected in (("company_id", self.request.company_id),
                              ("machine_id", self.request.machine_id)):
            if parameters.pop(key, None) != expected:
                raise GenerationEvidenceError("task reader company/machine drift")
        scope, _ = self.session.read_contract(request=self.request, current_allowed_sources=self.current())
        if scope.ai_scope != "machine_all":
            raise GenerationEvidenceError("family acquisition requires machine_all scope")
        return parameters

    def _relation_rows(self, registered, kind):
        receipt = self.session.inspect_read(request=self.request, handle=registered.read,
                                           current_allowed_sources=self.current())
        pages = self.records(registered.records)
        self.all_handles += registered.records
        out, seen = [], set()
        for relation in receipt.relations:
            index = relation.page_observation_index
            if index is None:
                # A LEFT JOIN without a page is not a manufactured Procedure.
                raise GenerationEvidenceError("parent relation has no observed page")
            item, handle = pages[index], registered.records[index]
            raw = dict(item.record)
            if kind == "parents":
                key = (relation.child_source_key, relation.parent_source_key)
                if key in seen:
                    continue
                seen.add(key)
                row = dict(child_source_key=relation.child_source_key,
                    parent_source_key=relation.parent_source_key, ordinal=relation.ordinal,
                    machine_id=str(raw.get("machine_id") or "").strip(),
                    page_number=raw["page_number"], parent_text=str(raw["text"]).strip())
            else:
                row = (raw["bubble_document_id"], relation.ordinal, raw["machine_id"],
                       raw["page_number"], raw["text"])
            self._bounded(row)
            self.raw_refs[id(row)] = (row, deepcopy(row), handle)
            out.append(row)
        return out

    def parent_rows(self, steps, **parameters):
        def work():
            args = self._scope_parameters(parameters)
            handles = self.handles(steps)
            keys = []
            for item in self.records(handles):
                key = storage_key(item.context.source)
                if key not in keys: keys.append(key)
            if args.get("child_source_keys") != keys or set(args) != {"child_source_keys", "text_chars"}:
                raise GenerationEvidenceError("parent read anchor drift")
            if not handles: return []
            registered = self.sources.parent_procedure_pages(step_handles=handles, text_chars=args["text_chars"])
            return self._relation_rows(registered, "parents")
        return self.invoke(work, self.request)

    def related_rows(self, procedure, **parameters):
        def work():
            args = self._scope_parameters(parameters)
            handle = self.handles([procedure])[0]
            source = self.records((handle,))[0].context.source
            if args.get("parent_source_key") != storage_key(source) or set(args) != {"parent_source_key", "text_chars"}:
                raise GenerationEvidenceError("related step anchor drift")
            return self._relation_rows(self.sources.related_step_pages(
                procedure_handle=handle, text_chars=args["text_chars"]), "steps")
        return self.invoke(work, self.request)

    def fallback_rows(self, **parameters):
        def work():
            args = self._scope_parameters(parameters)
            expected = max(800, int(self.tasks.expansion.ASK_STRUCTURED_DIRECT_TEXT_CHARS or 5000))
            if args != {"text_chars": expected}:
                raise GenerationEvidenceError("step fallback projection drift")
            registered = self.sources.step_fallback_pages()
            result = []
            for handle, item in zip(registered.records, self.records(registered.records)):
                raw = item.record
                row = (raw["bubble_document_id"], raw["machine_id"], raw["page_number"], raw["text"])
                self._bounded(row)
                self.raw_refs[id(row)] = (row, deepcopy(row), handle)
                result.append(row)
            self.all_handles += registered.records
            return result
        return self.invoke(work, self.request)

    def page_view(self, row, view):
        def work():
            entry = self.raw_refs.get(id(row))
            if entry is None or entry[0] is not row or not _same_value(row, entry[1]):
                raise GenerationEvidenceError("unregistered raw page conversion")
            # Legacy builders invent chunk_index=1 for pages. No chunk was read.
            view.pop("chunk_index", None)
            handle = derive_page_candidates(request=self.request, session=self.session,
                page_handles=(entry[2],), converter=lambda records: ((deepcopy(view), (0,)),),
                operation=TASK_GENERATION_VERSION+":page-view", authorize=self.authorize,
                invoke=self.invoke)[0]
            self._remember(view, handle)
        return self.invoke(work, self.request)

    def merge_candidates(self, groups):
        def work():
            origins = [self.handles(rows) for rows in groups]
            events = []
            result = ranking.v13_merge_candidates(groups, lineage=events.append)
            if len(events) != 1 or len(events[0]) != len(result):
                raise GenerationEvidenceError("candidate merge lineage missing")
            for row, positions in zip(result, events[0]):
                parents = tuple(origins[g][i] for g, i in positions)
                handle = self.session.derive_batch(request=self.request,
                    views=((parents, deepcopy(row)),), operation=TASK_GENERATION_VERSION+":merge",
                    current_allowed_sources=self.current())[0]
                self._remember(row, handle)
            return result
        return self.invoke(work, self.request)

    def dedupe_steps(self, rows):
        def merge(preferred, secondary):
            handles = self.handles([preferred, secondary])
            result = ranking.v12_merge_candidate_metadata(preferred, secondary, runtime=self.tasks.metadata_merge)
            # The original merge policy keeps preferred evidence text and merges
            # ranking metadata. Record the chosen body at that exact branch.
            if not _same_value({k:result[k] for k in self._BODY if k in result},
                               {k:preferred[k] for k in self._BODY if k in preferred}):
                raise GenerationEvidenceError("family metadata merge changed chosen body")
            handle = self.session.derive_batch(request=self.request,
                views=(((handles[0],), deepcopy(result)),), operation=TASK_GENERATION_VERSION+":preferred-body",
                current_allowed_sources=self.current())[0]
            return self._remember(result, handle)
        runtime = replace(self.tasks.step_dedupe, _v12_merge_candidate_metadata=merge)
        return ranking.v12_dedupe_family_steps(rows, runtime=runtime, trace=self)

    def _family(self, **parameters):
        runtime = replace(self.tasks.family, _v12_dedupe_family_steps=self.dedupe_steps,
            _v12_expand_primary_procedure_steps=lambda **kw: structured.v12_expand_primary_procedure_steps(
                **kw, runtime=self.tasks.expansion, trace=self))
        return families.v12_choose_primary_procedure_family(**parameters, runtime=runtime, trace=self)

    def _curate(self, **parameters):
        runtime = replace(self.tasks.curation, _v12_choose_primary_procedure_family=self._family)
        return families.v12_curate_structured_sources(**parameters, runtime=runtime, trace=self)

    def _direct(self, **parameters):
        def work():
            args = self._scope_parameters(parameters)
            if args.get("q") != self.request.query:
                raise GenerationEvidenceError("structured direct query drift")
            return self._materialize(self.sources.structured_direct(**args))
        return self.invoke(work, self.request)

    def _manual(self, **parameters):
        def work():
            args = self._scope_parameters(parameters)
            rows = args.pop("structured_citations")
            handles = self.handles(rows)
            if args.get("q") != self.request.query:
                raise GenerationEvidenceError("manual support query drift")
            result = self._materialize(self.sources.deterministic_manual_support(
                **args, structured_handles=handles))
            # The protected reader defers its old hidden file-map filter here.
            # Selection/top-k are unchanged; the file read is now receipted.
            selected = self.handles(result)
            files = self.sources.document_file_map(selected)
            return [row for row in result if str(files.get(str(row.get("bubble_document_id") or "").strip()) or "").strip()]
        return self.invoke(work, self.request)

    def _sources_block(self, rows, **parameters):
        def work():
            self.model_handles = self.handles(rows)
            value = self.tasks._v13_sources_block(rows, **parameters)
            self.handles(rows)
            if not isinstance(value, str):raise GenerationEvidenceError("task sources must be text")
            return value
        return self.invoke(work, self.request)

    def _inventory(self, retrieval):
        def work():
            if not _same_value(retrieval, self.retrieval_snapshot):
                raise GenerationEvidenceError("overview changed prepared retrieval")
            records = self.tasks._assistant_core_overview_inventory_records(retrieval, trace=self)
            for record in records:
                self.handles([record["candidate"]])
                self.inventory_refs[id(record)] = (record, deepcopy(record))
                self._bounded(record)
            return records
        return self.invoke(work, self.request)

    def _inventory_block(self, records):
        def work():
            for record in records:
                ref = self.inventory_refs.get(id(record))
                if ref is None or ref[0] is not record or not _same_value(record,ref[1]):
                    raise GenerationEvidenceError("untracked overview inventory")
            self.model_handles = self.handles([r["candidate"] for r in records])
            result = self.tasks._assistant_core_overview_records_block(records)
            for record, snapshot in self.inventory_refs.values():
                if not _same_value(record, snapshot):raise GenerationEvidenceError("inventory mutated during rendering")
            return result
        return self.invoke(work, self.request)

    def _model(self, *args, **parameters):
        def work():
            purpose = "ask_structured_synthesis" if self.kind=="structured" else "assistant_core_machine_overview_inventory"
            if (not self.model_handles or parameters.get("company_id")!=self.request.company_id
                    or parameters.get("purpose")!=purpose):
                raise GenerationEvidenceError("task model scope or input missing")
            self.records(self.all_handles)
            if not _same_value(self.retrieval,self.retrieval_snapshot):
                raise GenerationEvidenceError("task input changed before model")
            result = self.tasks._v13_json_models(*args, **parameters)
            self.records(self.all_handles)
            return result
        return self.invoke(work, self.request)

    def _render(self, **parameters):
        def work():
            rows = parameters["citations"]
            if self.handles(rows) != self.model_handles or parameters.get("q")!=self.request.query:
                raise GenerationEvidenceError("task renderer source drift")
            answer, selected = self.tasks._render_grounded_answer_points(**parameters)
            self.handles(rows);self.selected = self.handles(selected)
            return answer, selected
        return self.invoke(work,self.request)

    def _file_map_for(self, handles, company_id, keys):
        if company_id != self.request.company_id:
            raise GenerationEvidenceError("task file-map company drift")
        values = self.records(handles)
        all_keys = sorted({storage_key(item.context.source) for item in values})
        if keys != all_keys:
            raise GenerationEvidenceError("task file-map selection drift")
        return self._source_file_map(handles)

    def _sanitize(self, rows, *, company_id):
        def work():
            handles = self.handles(rows)
            result = self.runtime._sanitize_citations_for_response(rows,company_id=company_id,
                file_map_fn=lambda cid,keys:self._file_map_for(handles,cid,keys))
            self.handles(rows)
            if type(result) is not list or len(result)!=len(rows):
                raise GenerationEvidenceError("task citation projection count drift")
            for parent,view,handle in zip(rows,result,handles):
                if (type(view) is not dict or view.get("citation_id")!=str(parent.get("citation_id") or "").strip()
                        or view.get("bubble_document_id")!=str(parent.get("bubble_document_id") or "").strip()):
                    raise GenerationEvidenceError("task citation projection identity drift")
                projected = self.session.derive_batch(request=self.request,
                    views=(((handle,),deepcopy(view)),),operation=TASK_GENERATION_VERSION+":sanitize",
                    current_allowed_sources=self.current())[0]
                self._draft(view,projected,body={k:deepcopy(view[k]) for k in self._BODY if k in view})
            return result
        return self.invoke(work,self.request)

    def _links(self,company_id,rows):
        def work():
            handles=self.handles(rows)
            result=self.runtime._build_rg_links(company_id,rows,
                file_map_fn=lambda cid,keys:self._file_map_for(handles,cid,keys))
            self.handles(rows)
            if type(result) is not list or any(type(row) is not dict for row in result):
                raise GenerationEvidenceError("task link output invalid")
            for row in result:self.link_refs[id(row)]=(row,deepcopy(row))
            return result
        return self.invoke(work,self.request)

    def _finalize(self,response,*,language):
        def work():
            if language!=self.request.response_language:raise GenerationEvidenceError("task UI language drift")
            self.rendered=self.handles(response["citations"])
            result=self.runtime._finalize_ask_response_for_ui(response,language=language,
                citation_copy_fn=self._copy_citation)
            self.handles(response["citations"])
            self.rendered=self.handles(result["citations"])
            for row in result.get("rg_links",[]):
                ref=self.link_refs.get(id(row))
                if ref is None or ref[0] is not row or not _same_value(row,ref[1]):
                    raise GenerationEvidenceError("unregistered final task link")
            return result
        return self.invoke(work,self.request)

    def _bound_runtime(self):
        t=self.tasks
        return replace(t,copy_fn=self.copy,_v13_merge_candidates=self.merge_candidates,
            _v12_curate_structured_sources=lambda **kw:self.invoke(self._curate,self.request,**kw),
            _v12_mark_structured_roles=lambda rows:families.v12_mark_structured_roles(rows,runtime=t.roles,trace=self),
            _v12_select_response_steps=lambda **kw:t._v12_select_response_steps(**kw,trace=self),
            _v12_filter_manual_support_to_selected_bundle=lambda **kw:families.v12_filter_manual_support_to_selected_bundle(**kw,runtime=t.manual_filter,trace=self),
            _ask_structured_direct_fetch_sources=self._direct,_v13_fetch_manual_support_deterministic=self._manual,
            _v13_sources_block=self._sources_block,_v13_json_models=self._model,
            _render_grounded_answer_points=self._render,_sanitize_citations_for_response=self._sanitize,
            _build_rg_links=self._links,_finalize_ask_response_for_ui=self._finalize,
            _procedure_ui_merge_sources=lambda *groups:t._procedure_ui_merge_sources(*groups,trace=self),
            _procedure_ui_order_citations=lambda rows:t._procedure_ui_order_citations(rows,trace=self),
            _assistant_core_overview_inventory_records=self._inventory,
            _assistant_core_overview_records_block=self._inventory_block)

    def run_task(self,kind,retrieval,selections,decision):
        if self.used or kind not in {"structured","overview"}:
            raise GenerationEvidenceError("single explicit task synthesis operation required")
        self.used,self.active,self.kind=True,True,kind
        try:
            def work():
                self.session.admission(request=self.request,retrieval=retrieval,selections=selections,
                    current_allowed_sources=self.current())
                self.input_handles=tuple(h for s in selections for h in s.records)
                self.all_handles=self.input_handles
                for sel in selections:
                    for row,handle in zip(retrieval[sel.name],sel.records):self._bind(row,handle)
                self.retrieval,self.retrieval_snapshot=retrieval,deepcopy(retrieval)
                contract=retrieval.get("assistant_core_contract") or {}
                if contract.get("fail_closed") is not True:
                    raise GenerationEvidenceError("task requires fail-closed contract")
                r=self.request;runtime=self._bound_runtime()
                if kind=="overview":
                    result=synthesize_machine_overview(r,retrieval,decision,runtime=runtime)
                else:
                    result=structured_ask(q=r.query,company_id=r.company_id,machine_id=r.machine_id,
                        response_language=r.response_language,top_k=r.top_k,planner=retrieval.get("plan") or {},
                        seed_citations=retrieval.get("citations") or retrieval.get("candidates") or [],
                        assurance_meta=retrieval.get("retrieval_assurance") or {},debug=r.debug,runtime=runtime)
                self.records(self.all_handles)
                if not _same_value(retrieval,self.retrieval_snapshot):
                    raise GenerationEvidenceError("task changed admitted input")
                if result is None:return None,(),()
                output=self.handles(result.get("citations",[]))
                validation=self.handles(result.get("_assistant_core_validation_evidence",[]))
                outsel=(AskSelection("citations",output),)
                self.session.admission(request=r,retrieval={"citations":result.get("citations",[])},
                    selections=outsel,current_allowed_sources=self.current())
                self.session.admission(request=r,retrieval={"candidates":result.get("_assistant_core_validation_evidence",[])},
                    selections=(AskSelection("candidates",validation),),current_allowed_sources=self.current())
                return result,outsel,(AskSelection("candidates",validation),)
            return self.invoke(work,self.request)
        finally:
            self.active=False
            self.refs.clear();self.link_refs.clear();self.drafts.clear();self.raw_refs.clear();self.inventory_refs.clear()
            self.all_handles=self.input_handles=self.validation_handles=()
            self.retrieval=self.retrieval_snapshot=None
