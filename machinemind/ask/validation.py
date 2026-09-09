"""P6-B4c: response validation and citation reconstruction with explicit dependencies.

Main uses the behavior-preserving legacy delegates. Optional binding reuses the
EXISTING B4b execution runtime and therefore the same B4a session admission; it
never creates an authority, a second registry or a source search. It protects the
collections at this boundary, not answer semantics, URL ownership, internal
callback acquisitions or all request paths. Canonical main activation is pending.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, TYPE_CHECKING

from . import execution
from ..evidence.ask_input import apply_ask_evidence_input, ask_request_key
from ..evidence.contracts import EvidenceContractError

if TYPE_CHECKING:
    from assistant_core_v2 import AssistantCoreRequest, AssistantCoreDecision

ASK_VALIDATION_VERSION = "ask-validation-p6b4c-v1"


@dataclass(frozen=True, slots=True)
class AskValidationRuntime:
    """Legacy callbacks/constants; canonical checks require an explicit binding."""
    EVIDENCE_PARTIAL: Any
    KIND_PROCEDURE: Any
    MODE_ASK: Any
    MODE_ROOT_CAUSE: Any
    REQ_CHECKLIST: Any
    REQ_INTERFACE_LOCATIONS: Any
    REQ_NUMERIC_VALUE: Any
    REQ_ORDERED_ACTIONS: Any
    REQ_SAFETY_CONDITIONS: Any
    REQ_STATE_SEQUENCE: Any
    RESULT_INCOMPLETE_ANSWER_CONTRACT: Any
    RESULT_NO_MACHINE_EVIDENCE: Any
    _assistant_core_answer_contract_check: Callable[..., Any]
    _assistant_core_build_no_evidence: Callable[..., Any]
    _assistant_core_candidate_evidence_text: Callable[..., Any]
    _assistant_core_candidate_source_type: Callable[..., Any]
    _assistant_core_claim_supported: Callable[..., Any]
    _assistant_core_claims: Callable[..., Any]
    _assistant_core_explicit_partial_answer_allowed: Callable[..., Any]
    _assistant_core_filter_unsupported_claim_sentences: Callable[..., Any]
    _assistant_core_media_metadata_only: Callable[..., Any]
    _assistant_core_recover_citations: Callable[..., Any]
    _assistant_core_redact_internal_text: Callable[..., Any]
    _assistant_core_response_claim_text: Callable[..., Any]
    _assistant_core_root_candidate_viable: Callable[..., Any]
    _assistant_core_should_semantic_verify_answer: Callable[..., Any]
    _assistant_core_source_bonus: Callable[..., Any]
    _assistant_core_verify_or_repair_answer: Callable[..., Any]
    _build_rg_links: Callable[..., Any]
    _content_term_set: Callable[..., Any]
    _dedup_text_values: Callable[..., Any]
    _procedure_ui_order_citations: Callable[..., Any]
    _sanitize_citations_for_response: Callable[..., Any]
    _term_overlap_score: Callable[..., Any]
    _unique_non_empty_strings: Callable[..., Any]
    _v12_evidence_role: Callable[..., Any]
    _v13_current_budget: Callable[..., Any]
    execution_runtime: execution.AskExecutionRuntime | None = None


def _guarded(runtime: AskValidationRuntime, request: Any, decision: Any) -> bool:
    return (runtime.execution_runtime is not None
            and execution._guarded(runtime.execution_runtime, request, decision))


def _admit(request: Any, data: dict, decision: Any, *,
           runtime: AskValidationRuntime, stage: str) -> dict:
    """Keep a pre-callback snapshot; admission cannot rewrite its own reference.

    select/authorize are trusted application callbacks bound by B4b. Missing
    handles, revoked grants or mutated records raise a technical contract error.
    No matching by citation id, score, title or text is an authorization here.
    Collections must already be bounded by the producer; copies are not free.
    """
    if not _guarded(runtime, request, decision):
        return data
    key = ask_request_key(request)
    snapshot = deepcopy(data)
    observed = deepcopy(snapshot)
    admission = runtime.execution_runtime.evidence_admission(request, observed, decision, stage)
    if ask_request_key(request) != key:
        raise EvidenceContractError("ASK request changed during validation admission")
    return apply_ask_evidence_input(snapshot, request_key=key, admission=admission)


def _collection(request: Any, items: Any, decision: Any, *,
                runtime: AskValidationRuntime, stage: str) -> Any:
    return _admit(request, {"candidates": items}, decision,
                  runtime=runtime, stage=stage)["candidates"]


def _inputs(response: dict, request: Any, retrieval: dict, decision: Any, *,
            runtime: AskValidationRuntime, stage: str) -> tuple[dict, dict]:
    if not _guarded(runtime, request, decision):
        return response, retrieval
    checked = _admit(request, retrieval, decision, runtime=runtime, stage=stage+".retrieval")
    out = dict(response or {})
    if "_assistant_core_validation_evidence" in out:
        out["_assistant_core_validation_evidence"] = _collection(request,
            out["_assistant_core_validation_evidence"], decision,
            runtime=runtime, stage=stage+".private")
    return out, checked


def _output(response: dict, request: Any, decision: Any, *,
            runtime: AskValidationRuntime) -> dict:
    if not _guarded(runtime, request, decision):
        return response
    out = dict(response)
    if "citations" in out:
        out["citations"] = _collection(request, out["citations"], decision,
            runtime=runtime, stage="validation.output_citations")
    if "_assistant_core_validation_evidence" in out:
        out["_assistant_core_validation_evidence"] = _collection(request,
            out["_assistant_core_validation_evidence"], decision,
            runtime=runtime, stage="validation.output_private")
    return out


def bind_ask_validation(runtime: AskValidationRuntime, *,
                        execution_runtime: execution.AskExecutionRuntime) -> AskValidationRuntime:
    """Reuse an already-bound EXECUTION runtime; do not construct another session.

    The owner must pass the result of bind_ask_execution for this request. It
    carries the current authority and exact occurrence selection. The nested
    verifier and citation recovery are rebound too; otherwise main's global
    callbacks would silently drop admission. Other requested/effective modes
    retain their original callbacks and policies. This is not an engine factory
    or a complete lifecycle activation; no global engine is mutated.
    """
    if type(runtime) is not AskValidationRuntime or runtime.execution_runtime is not None:
        raise EvidenceContractError("one unbound validation runtime required")
    if (type(execution_runtime) is not execution.AskExecutionRuntime
            or not callable(execution_runtime.evidence_admission)):
        raise EvidenceContractError("the existing request-bound EXECUTION runtime is required")

    def recover(response: dict, *, request: Any, retrieval: dict, decision: Any):
        if not execution._ask_path(request, decision):
            return runtime._assistant_core_recover_citations(response, request=request,
                retrieval=retrieval, decision=decision)
        return recover_citations(response, request=request, retrieval=retrieval,
            decision=decision, runtime=bound)

    def verify(*, request: Any, decision: Any, answer: str, candidates: list[dict], **kwargs):
        if not execution._ask_path(request, decision):
            return runtime._assistant_core_verify_or_repair_answer(request=request,
                decision=decision, answer=answer, candidates=candidates, **kwargs)
        return execution.verify_or_repair_answer(request=request, decision=decision,
            answer=answer, candidates=candidates, runtime=execution_runtime, **kwargs)

    bound = replace(runtime, execution_runtime=execution_runtime,
        _assistant_core_recover_citations=recover,
        _assistant_core_verify_or_repair_answer=verify)
    return bound


def recover_citations(
    response: dict,
    *,
    request: AssistantCoreRequest,
    retrieval: dict,
    decision: AssistantCoreDecision,
    runtime: AskValidationRuntime,
) -> tuple[list[dict], list[dict]]:
    KIND_PROCEDURE = runtime.KIND_PROCEDURE
    _assistant_core_candidate_evidence_text = runtime._assistant_core_candidate_evidence_text
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _assistant_core_claim_supported = runtime._assistant_core_claim_supported
    _assistant_core_claims = runtime._assistant_core_claims
    _assistant_core_response_claim_text = runtime._assistant_core_response_claim_text
    _assistant_core_source_bonus = runtime._assistant_core_source_bonus
    _build_rg_links = runtime._build_rg_links
    _content_term_set = runtime._content_term_set
    _procedure_ui_order_citations = runtime._procedure_ui_order_citations
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    _term_overlap_score = runtime._term_overlap_score
    _v12_evidence_role = runtime._v12_evidence_role
    response, retrieval = _inputs(response, request, retrieval, decision, runtime=runtime, stage="citation_recovery")
    retrieval_selected = [
        dict(c)
        for c in (retrieval.get("citations") or retrieval.get("candidates") or [])
        if isinstance(c, dict) and str(c.get("citation_id") or "").strip()
    ]
    trusted_manifest = [
        dict(c)
        for c in (response.get("_assistant_core_validation_evidence") or [])
        if isinstance(c, dict)
        and str(c.get("citation_id") or "").strip()
        and (
            bool(c.get("ask_structured_direct"))
            or bool(c.get("ask_structured_manual_support"))
            or _v12_evidence_role(c) in {"procedure", "step", "ps", "md_photo", "md_video", "manual_support"}
        )
    ]

    # Merge the original retrieval pack with the deterministic structured
    # manifest. The latter contains the complete ordered Procedure/Step family
    # loaded from structured_source_relations, which semantic top-k may omit.
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for candidate in retrieval_selected + trusted_manifest:
        cid = str(candidate.get("citation_id") or "").strip()
        if not cid or cid in selected_ids:
            continue
        selected_ids.add(cid)
        selected.append(candidate)

    selected = _collection(request, selected, decision, runtime=runtime, stage="citation_recovery.selected")
    by_id = {str(c.get("citation_id") or "").strip(): c for c in selected}
    existing = [
        dict(c)
        for c in (response.get("citations") or [])
        if isinstance(c, dict) and str(c.get("citation_id") or "").strip() in by_id
    ]

    # For a deterministic structured procedure response, preserve the exact
    # manifest used to build answer/answer_html. Do not re-rank it against the
    # original semantic retrieval and do not apply the generic 8-source cap.
    if trusted_manifest and decision.request_kind == KIND_PROCEDURE:
        manifest_ids = {
            str(c.get("citation_id") or "").strip()
            for c in trusted_manifest
            if str(c.get("citation_id") or "").strip()
        }
        ordered = [
            dict(c) for c in (response.get("citations") or [])
            if isinstance(c, dict)
            and str(c.get("citation_id") or "").strip() in manifest_ids
        ]
        ordered_ids = {str(c.get("citation_id") or "").strip() for c in ordered}
        ordered.extend(
            dict(c) for c in trusted_manifest
            if str(c.get("citation_id") or "").strip() not in ordered_ids
        )
        ordered = _procedure_ui_order_citations(ordered)
        ordered = _collection(request, ordered, decision, runtime=runtime, stage="citation_recovery.procedure_before_sanitize")
        try:
            sanitized = _sanitize_citations_for_response(ordered, company_id=request.company_id)
        except Exception:
            if _guarded(runtime, request, decision):
                raise
            sanitized = ordered
        sanitized = _collection(request, sanitized, decision, runtime=runtime, stage="citation_recovery.procedure_after_sanitize")
        sanitized = _procedure_ui_order_citations(sanitized)
        sanitized = _collection(request, sanitized, decision, runtime=runtime, stage="citation_recovery.procedure_final")
        try:
            links = _procedure_ui_order_citations(
                _build_rg_links(request.company_id, sanitized)
            )
        except Exception:
            if _guarded(runtime, request, decision):
                raise
            links = [
                dict(link) for link in (response.get("rg_links") or [])
                if isinstance(link, dict)
                and str(link.get("citation_id") or "").strip()
                in {str(c.get("citation_id") or "").strip() for c in sanitized}
            ]
        return sanitized, links
    existing_ids = {str(c.get("citation_id") or "").strip() for c in existing}
    answer_text = _assistant_core_response_claim_text(response, decision.effective_mode)
    answer_claims = _assistant_core_claims(answer_text)

    cited_text = "\n".join(_assistant_core_candidate_evidence_text(by_id[cid]) for cid in existing_ids if cid in by_id)
    cited_claims = _assistant_core_claims(cited_text)
    missing_claims = [
        claim
        for claim in answer_claims
        if not _assistant_core_claim_supported(claim, cited_text, cited_claims)
    ]

    query_terms = _content_term_set(request.query + " " + answer_text, limit=160)
    ranked: list[tuple[float, dict]] = []
    for candidate in selected:
        cid = str(candidate.get("citation_id") or "").strip()
        if cid in existing_ids:
            continue
        candidate_text = _assistant_core_candidate_evidence_text(candidate)
        candidate_claims = _assistant_core_claims(candidate_text)
        support_count = sum(
            1
            for claim in missing_claims
            if _assistant_core_claim_supported(claim, candidate_text, candidate_claims)
        )
        overlap = _term_overlap_score(
            query_terms,
            _content_term_set(candidate_text, limit=220),
        ) if query_terms else 0.0
        source_type = _assistant_core_candidate_source_type(candidate)
        task_bonus = _assistant_core_source_bonus(
            source_type,
            decision.request_kind,
            set(decision.preferred_source_types),
            decision.information_task,
        )
        score = support_count * 2.0 + overlap + task_bonus
        if support_count > 0 or (not existing and overlap >= 0.025):
            ranked.append((score, candidate))

    ranked.sort(
        key=lambda item: (
            -item[0],
            -float(item[1].get("v13_score", item[1].get("retrieval_score", item[1].get("similarity", 0.0))) or 0.0),
            str(item[1].get("citation_id") or ""),
        )
    )
    recovered = list(existing)
    for _, candidate in ranked:
        cid = str(candidate.get("citation_id") or "").strip()
        if cid in existing_ids:
            continue
        recovered.append(candidate)
        existing_ids.add(cid)
        if len(recovered) >= 8:
            break

    # Procedure answers without numerical claims still need an operational source
    # manifest. Select the highest-ranked procedure/steps when synthesis omitted it.
    if not recovered and decision.request_kind == KIND_PROCEDURE:
        for candidate in selected:
            if _assistant_core_candidate_source_type(candidate) not in {"procedure", "step", "document"}:
                continue
            recovered.append(candidate)
            if len(recovered) >= 6:
                break

    recovered = _collection(request, recovered, decision, runtime=runtime, stage="citation_recovery.generic_before_sanitize")
    try:
        sanitized = _sanitize_citations_for_response(recovered, company_id=request.company_id)
    except Exception:
        if _guarded(runtime, request, decision):
            raise
        sanitized = recovered
    sanitized = _collection(request, sanitized, decision, runtime=runtime, stage="citation_recovery.generic_after_sanitize")
    valid_ids = {str(c.get("citation_id") or "").strip() for c in sanitized}
    links = [
        dict(link)
        for link in (response.get("rg_links") or [])
        if isinstance(link, dict)
        and str(link.get("citation_id") or "").strip() in valid_ids
    ]
    if sanitized and len({str(x.get("citation_id") or "") for x in links}) < len(valid_ids):
        try:
            built = _build_rg_links(request.company_id, sanitized)
            if built:
                links = built
        except Exception:
            if _guarded(runtime, request, decision):
                raise
            pass
    return sanitized, links


def validate_response(
    response: dict,
    request: AssistantCoreRequest,
    retrieval: dict,
    decision: AssistantCoreDecision,
    *,
    runtime: AskValidationRuntime,
) -> dict:
    EVIDENCE_PARTIAL = runtime.EVIDENCE_PARTIAL
    MODE_ASK = runtime.MODE_ASK
    MODE_ROOT_CAUSE = runtime.MODE_ROOT_CAUSE
    REQ_CHECKLIST = runtime.REQ_CHECKLIST
    REQ_INTERFACE_LOCATIONS = runtime.REQ_INTERFACE_LOCATIONS
    REQ_NUMERIC_VALUE = runtime.REQ_NUMERIC_VALUE
    REQ_ORDERED_ACTIONS = runtime.REQ_ORDERED_ACTIONS
    REQ_SAFETY_CONDITIONS = runtime.REQ_SAFETY_CONDITIONS
    REQ_STATE_SEQUENCE = runtime.REQ_STATE_SEQUENCE
    RESULT_INCOMPLETE_ANSWER_CONTRACT = runtime.RESULT_INCOMPLETE_ANSWER_CONTRACT
    RESULT_NO_MACHINE_EVIDENCE = runtime.RESULT_NO_MACHINE_EVIDENCE
    _assistant_core_answer_contract_check = runtime._assistant_core_answer_contract_check
    _assistant_core_build_no_evidence = runtime._assistant_core_build_no_evidence
    _assistant_core_candidate_evidence_text = runtime._assistant_core_candidate_evidence_text
    _assistant_core_explicit_partial_answer_allowed = runtime._assistant_core_explicit_partial_answer_allowed
    _assistant_core_filter_unsupported_claim_sentences = runtime._assistant_core_filter_unsupported_claim_sentences
    _assistant_core_media_metadata_only = runtime._assistant_core_media_metadata_only
    _assistant_core_recover_citations = runtime._assistant_core_recover_citations
    _assistant_core_redact_internal_text = runtime._assistant_core_redact_internal_text
    _assistant_core_root_candidate_viable = runtime._assistant_core_root_candidate_viable
    _assistant_core_should_semantic_verify_answer = runtime._assistant_core_should_semantic_verify_answer
    _assistant_core_verify_or_repair_answer = runtime._assistant_core_verify_or_repair_answer
    _build_rg_links = runtime._build_rg_links
    _dedup_text_values = runtime._dedup_text_values
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    _unique_non_empty_strings = runtime._unique_non_empty_strings
    _v13_current_budget = runtime._v13_current_budget
    response, retrieval = _inputs(response, request, retrieval, decision, runtime=runtime, stage="validation")
    out = dict(response or {})
    allowed_candidates = [
        dict(c)
        for c in (retrieval.get("citations") or retrieval.get("candidates") or [])
        if isinstance(c, dict) and str(c.get("citation_id") or "").strip()
    ]
    allowed_candidates.extend(
        dict(c)
        for c in (out.get("_assistant_core_validation_evidence") or [])
        if isinstance(c, dict) and str(c.get("citation_id") or "").strip()
    )
    allowed = {
        str(c.get("citation_id") or "").strip(): c
        for c in allowed_candidates
        if str(c.get("citation_id") or "").strip()
    }
    preserve_empty_evidence = bool(
        str(out.get("status") or "").strip().lower() == "no_sources"
        and str(out.get("result_code") or "").strip().upper()
        == RESULT_INCOMPLETE_ANSWER_CONTRACT
    )
    if preserve_empty_evidence or (
        request.requested_mode in {MODE_ASK, MODE_ROOT_CAUSE}
        and str(out.get("status") or "").strip().lower() == "no_sources"
    ):
        citations, links = [], []
    elif request.requested_mode == MODE_ROOT_CAUSE and (out.get("meta") or {}).get("root_causal_applicability"):
        # Rebuild exclusively from the accepted cause IDs below. Generic citation
        # recovery must not reintroduce an unvalidated source, including on error.
        citations, links = [], []
    else:
        citations, links = _assistant_core_recover_citations(
            out,
            request=request,
            retrieval=retrieval,
            decision=decision,
        )
    out["citations"] = citations
    out["rg_links"] = links
    used_ids = {str(c.get("citation_id") or "").strip() for c in citations}

    # Validate against the complete evidence pack actually supplied to synthesis,
    # not merely against the subset the model happened to repeat as citations.
    source_text = "\n".join(
        _assistant_core_candidate_evidence_text(candidate)
        for candidate in allowed.values()
    )
    removed_claims: list[str] = []
    answer_contract_result: dict = {"passed": True, "reason": "not_applicable"}
    if preserve_empty_evidence:
        repair_meta_for_validation = dict(
            (out.get("meta") or {}).get("assistant_core_repair") or {}
        )
        first_contract = dict(
            repair_meta_for_validation.get("first_answer_contract") or {}
        )
        answer_contract_result = {
            **first_contract,
            "passed": False,
            "reason": "repair_failed_after_one_attempt",
            "repair": repair_meta_for_validation,
        }

    if decision.effective_mode == MODE_ROOT_CAUSE:
        cleaned_causes: list[dict] = []
        for cause in out.get("possible_causes") or []:
            if not isinstance(cause, dict):
                continue
            cc = dict(cause)
            cause_source_text = source_text
            if request.requested_mode == MODE_ROOT_CAUSE and (out.get("meta") or {}).get("root_causal_applicability"):
                cause_source_text = "\n".join(
                    _assistant_core_candidate_evidence_text(allowed[cid])
                    for cid in (cc.get("citations") or []) if cid in allowed
                )
            cc["cause"], removed = _assistant_core_filter_unsupported_claim_sentences(
                _assistant_core_redact_internal_text(cc.get("cause") or ""), cause_source_text
            )
            removed_claims.extend(removed)
            cc["why"], removed = _assistant_core_filter_unsupported_claim_sentences(
                _assistant_core_redact_internal_text(cc.get("why") or ""), cause_source_text
            )
            removed_claims.extend(removed)
            checks: list[str] = []
            for check in cc.get("checks") or []:
                cleaned, removed = _assistant_core_filter_unsupported_claim_sentences(
                    _assistant_core_redact_internal_text(check), cause_source_text
                )
                removed_claims.extend(removed)
                if cleaned:
                    checks.append(cleaned)
            cc["checks"] = checks
            cause_ids = []
            for raw_id in (cc.get("citations") or []):
                cid = str(raw_id or "").strip()
                candidate = allowed.get(cid)
                if not cid or not isinstance(candidate, dict):
                    continue
                if not _assistant_core_root_candidate_viable(request, decision, candidate):
                    continue
                cause_ids.append(cid)
                if len(cause_ids) >= 4:
                    break
            # Never launder an unsupported cause by attaching the nearest citation
            # after synthesis. A cause without its own compatible evidence is dropped.
            cc["citations"] = cause_ids
            if cause_ids and cc.get("cause") and (cc.get("why") or checks):
                cleaned_causes.append(cc)
        for idx, cause in enumerate(cleaned_causes, start=1):
            cause["rank"] = idx
        out["possible_causes"] = cleaned_causes[: max(1, request.max_causes)]
        final_cause_ids = _dedup_text_values(
            [cid for cause in out["possible_causes"] for cid in (cause.get("citations") or [])],
            limit=16,
        )
        if final_cause_ids:
            final_raw_citations = [allowed[cid] for cid in final_cause_ids if cid in allowed]
            try:
                out["citations"] = _sanitize_citations_for_response(
                    final_raw_citations, company_id=request.company_id
                )
            except Exception:
                if _guarded(runtime, request, decision):
                    raise
                out["citations"] = final_raw_citations
            try:
                out["rg_links"] = _build_rg_links(request.company_id, out["citations"])
            except Exception:
                if _guarded(runtime, request, decision):
                    raise
                out["rg_links"] = []
        out["problem_summary"] = _assistant_core_redact_internal_text(out.get("problem_summary") or "")
        out["recommended_next_checks"] = _unique_non_empty_strings(
            [check for c in out["possible_causes"] for check in (c.get("checks") or [])],
            limit=8,
        )
        if str(out.get("status") or "").lower() == "answered" and not out["possible_causes"]:
            out.update(
                {
                    "status": "no_sources",
                    "result_code": RESULT_NO_MACHINE_EVIDENCE,
                    "problem_summary": _assistant_core_build_no_evidence(request, decision, retrieval).get("problem_summary", ""),
                    "citations": [],
                    "rg_links": [],
                }
            )
    else:
        answer = _assistant_core_redact_internal_text(out.get("answer") or "")
        if citations and source_text:
            answer, removed_claims = _assistant_core_filter_unsupported_claim_sentences(answer, source_text)
        answer = _assistant_core_media_metadata_only(answer, citations, request.response_language)
        out["answer"] = answer

        semantic_contract: dict = dict(out.pop("_assistant_core_semantic_verified", {}) or {})
        semantic_contract_pass = bool(
            semantic_contract
            and str(semantic_contract.get("outcome") or "").strip().lower() in {"pass", "rewrite"}
            and not list(semantic_contract.get("missing_facets") or [])
            and not list(semantic_contract.get("missing_answer_types") or [])
            and not list(semantic_contract.get("missing_list_items") or [])
        )
        semantic_contract_partial = False
        if (
            not semantic_contract_pass
            and str(out.get("status") or "").lower() == "answered"
            and answer
            and (
                _assistant_core_should_semantic_verify_answer(decision)
                or bool(out.get("_assistant_core_force_semantic_verify"))
            )
        ):
            semantic_contract = _assistant_core_verify_or_repair_answer(
                request=request,
                decision=decision,
                answer=answer,
                candidates=list(allowed.values()),
            )
            outcome = str(semantic_contract.get("outcome") or "").strip().lower()
            repaired_answer = _assistant_core_redact_internal_text(semantic_contract.get("answer") or "")
            hard_requirements = {
                str(x or "").strip().lower()
                for x in decision.required_answer_types
                if str(x or "").strip()
            } & {
                REQ_NUMERIC_VALUE,
                REQ_INTERFACE_LOCATIONS,
                REQ_STATE_SEQUENCE,
                REQ_CHECKLIST,
                REQ_SAFETY_CONDITIONS,
                REQ_ORDERED_ACTIONS,
            }

            if outcome in {"pass", "rewrite"} and repaired_answer:
                answer = repaired_answer
                semantic_contract_pass = (
                    not bool(semantic_contract.get("missing_facets"))
                    and not bool(semantic_contract.get("missing_answer_types"))
                    and not bool(semantic_contract.get("missing_list_items"))
                )
            elif (
                outcome == "partial"
                and repaired_answer
                and (
                    not hard_requirements
                    or _assistant_core_explicit_partial_answer_allowed(
                        answer=repaired_answer,
                        decision=decision,
                        semantic_contract=semantic_contract,
                    )
                )
            ):
                answer = repaired_answer
                semantic_contract_partial = True
            elif outcome in {"partial", "no_sources"}:
                evidence_admission_supported = bool(
                    allowed
                    and (retrieval.get("assistant_core_decision") or {}).get("supported", True)
                )
                if evidence_admission_supported and decision.effective_mode == MODE_ASK:
                    # The verifier may be stricter than the first synthesis. Preserve
                    # the grounded first answer and route it through the single bounded
                    # repair cycle instead of mislabelling available evidence as absent.
                    out["_assistant_core_semantic_rejected_with_evidence"] = True
                    out["_assistant_core_semantic_rejection"] = dict(semantic_contract)
                else:
                    no_evidence = _assistant_core_build_no_evidence(request, decision, retrieval)
                    out.update(no_evidence)
                    out.pop("answer_html", None)
                    out.pop("_assistant_ui_model", None)
                    out["citations"] = []
                    out["rg_links"] = []
                    answer = str(out.get("answer") or "")

            verifier_ids = [
                str(cid or "").strip()
                for cid in (semantic_contract.get("citation_ids") or [])
                if str(cid or "").strip() in allowed
            ]
            if (semantic_contract_pass or semantic_contract_partial) and verifier_ids:
                raw_final = [allowed[cid] for cid in verifier_ids if cid in allowed]
                raw_final = _collection(request, raw_final, decision, runtime=runtime, stage="validation.semantic_citations")
                try:
                    out["citations"] = _sanitize_citations_for_response(
                        raw_final, company_id=request.company_id
                    )
                except Exception:
                    if _guarded(runtime, request, decision):
                        raise
                    out["citations"] = raw_final
                try:
                    out["rg_links"] = _build_rg_links(request.company_id, out["citations"])
                except Exception:
                    if _guarded(runtime, request, decision):
                        raise
                    out["rg_links"] = []
                citations = list(out.get("citations") or [])

            if (semantic_contract_pass or semantic_contract_partial) and source_text:
                answer, removed_after_repair = _assistant_core_filter_unsupported_claim_sentences(
                    answer, source_text
                )
                removed_claims.extend(removed_after_repair)
                answer = _assistant_core_media_metadata_only(
                    answer, citations, request.response_language
                )
                out["answer"] = answer

        if str(out.get("status") or "").lower() == "answered" and answer:
            deterministic_contract = _assistant_core_answer_contract_check(
                answer=answer,
                evidence_text=source_text,
                decision=decision,
            )
            if semantic_contract_pass:
                answer_contract_result = {
                    **dict(deterministic_contract),
                    "passed": True,
                    "reason": "semantic_contract_complete",
                    "semantic_verifier": dict(semantic_contract),
                }
            elif semantic_contract_partial:
                answer_contract_result = {
                    **dict(deterministic_contract),
                    "passed": True,
                    "reason": "semantic_contract_partial_explicit",
                    "semantic_verifier": dict(semantic_contract),
                }
                out["evidence_state"] = EVIDENCE_PARTIAL
            else:
                answer_contract_result = {
                    **dict(deterministic_contract),
                    "semantic_verifier": dict(semantic_contract),
                }
            semantic_rejected_with_evidence = bool(
                out.pop("_assistant_core_semantic_rejected_with_evidence", False)
            )
            semantic_missing_facets = _dedup_text_values(
                semantic_contract.get("missing_facets") or [], limit=12
            )
            semantic_missing_types = _dedup_text_values(
                semantic_contract.get("missing_answer_types") or [], limit=10
            )
            semantic_missing_list = _dedup_text_values(
                semantic_contract.get("missing_list_items") or [], limit=40
            )
            semantic_contract_incomplete = bool(
                semantic_contract
                and not semantic_contract_pass
                and not semantic_contract_partial
                and (
                    semantic_missing_facets
                    or semantic_missing_types
                    or semantic_missing_list
                )
            )
            if semantic_rejected_with_evidence or semantic_contract_incomplete:
                answer_contract_result = {
                    **dict(answer_contract_result),
                    "passed": False,
                    "reason": (
                        "semantic_contract_rejected_grounded_answer"
                        if semantic_rejected_with_evidence
                        else "semantic_contract_incomplete_grounded_answer"
                    ),
                    "missing_answer_facets": semantic_missing_facets
                    or list(answer_contract_result.get("missing_answer_facets") or []),
                    "semantic_missing_answer_types": semantic_missing_types,
                    "missing_list_items": semantic_missing_list,
                    "semantic_verifier": dict(semantic_contract),
                }
            if not bool(answer_contract_result.get("passed")):
                repair_attempted = bool(out.get("_assistant_core_repair_attempted"))
                missing_evidence_facets = list(
                    answer_contract_result.get("missing_evidence_facets") or []
                )
                admission_meta = dict(retrieval.get("assistant_core_decision") or {})
                try:
                    admission_facet_coverage = float(
                        admission_meta.get("facet_evidence_coverage") or 0.0
                    )
                except Exception:
                    admission_facet_coverage = 0.0
                admission_supported = bool(admission_meta.get("supported", True))
                # The admission stage evaluates the full balanced evidence pool and
                # may prove facet coverage that the lightweight final string matcher
                # cannot recognise because wording differs. A strict/sufficient
                # admission must therefore be allowed one bounded grounded repair;
                # otherwise valid evidence is mislabelled NO_MACHINE_EVIDENCE.
                admission_proves_evidence = bool(
                    admission_supported
                    and (
                        bool(admission_meta.get("strict_supported"))
                        or admission_facet_coverage >= float(
                            admission_meta.get("minimum_facet_coverage") or 0.70
                        )
                        or not missing_evidence_facets
                    )
                )
                evidence_complete = bool(allowed and admission_proves_evidence)
                failed_answer_types = [
                    str(item.get("requirement") or "").strip().lower()
                    for item in (answer_contract_result.get("requirement_checks") or [])
                    if isinstance(item, dict)
                    and not bool(item.get("passed"))
                    and str(item.get("requirement") or "").strip()
                ]

                if (
                    decision.effective_mode == MODE_ASK
                    and evidence_complete
                    and not repair_attempted
                ):
                    budget_for_repair = _v13_current_budget()
                    if (
                        budget_for_repair is not None
                        and budget_for_repair.llm_calls >= budget_for_repair.max_llm_calls
                        and budget_for_repair.remaining() >= 12.0
                        and budget_for_repair.estimated_cost_usd < budget_for_repair.max_estimated_cost_usd
                    ):
                        budget_for_repair.grant_retry_allowance(
                            failed_attempts=1,
                            reason="grounded_answer_contract_repair_stage",
                        )
                    # Keep the first grounded response and full evidence manifest.
                    # AssistantCoreV2 will invoke exactly one repair hook, using the
                    # already-budgeted third call, then validate the replacement once.
                    out["_assistant_core_repair_needed"] = True
                    out["_assistant_core_repair_context"] = {
                        "first_answer": answer,
                        "first_answer_contract": dict(answer_contract_result),
                        "missing_answer_facets": list(
                            answer_contract_result.get("missing_answer_facets") or []
                        ),
                        "missing_answer_types": _dedup_text_values(
                            list(failed_answer_types)
                            + list(answer_contract_result.get("semantic_missing_answer_types") or []),
                            limit=10,
                        ),
                        "missing_list_items": _dedup_text_values(
                            answer_contract_result.get("missing_list_items") or [],
                            limit=40,
                        ),
                        "evidence_facet_coverage": answer_contract_result.get(
                            "evidence_facet_coverage"
                        ),
                    }
                else:
                    no_evidence = _assistant_core_build_no_evidence(
                        request, decision, retrieval
                    )
                    out.update(no_evidence)
                    if evidence_complete:
                        out["result_code"] = RESULT_INCOMPLETE_ANSWER_CONTRACT
                    out.pop("answer_html", None)
                    out.pop("_assistant_ui_model", None)
                    out["citations"] = []
                    out["rg_links"] = []
        if str(out.get("status") or "").lower() == "answered" and not answer:
            no_evidence = _assistant_core_build_no_evidence(request, decision, retrieval)
            out.update(no_evidence)

    if str(out.get("status") or "").lower() != "answered" and "answer" not in out:
        out.pop("answer_html", None)
        out.pop("_assistant_ui_model", None)

    meta = dict(out.get("meta") or {})
    meta["assistant_core_validation"] = {
        "valid_citation_count": len(out.get("citations") or []),
        "valid_link_count": len(out.get("rg_links") or []),
        "unsupported_numeric_or_code_claims_removed": sorted(set(removed_claims))[:20],
        "answer_contract": dict(answer_contract_result),
    }
    if request.debug:
        meta["assistant_core_evidence_admission"] = dict(
            retrieval.get("assistant_core_decision") or {}
        )
    out["meta"] = meta
    # Internal-only data survives only between the first validation and the
    # single bounded repair hook. It is never exposed in the final API payload.
    repair_pending = bool(out.get("_assistant_core_repair_needed")) and not bool(
        out.get("_assistant_core_repair_attempted")
    )
    if not repair_pending:
        out.pop("_assistant_core_validation_evidence", None)
        out.pop("_assistant_core_force_semantic_verify", None)
        out.pop("_assistant_core_repair_context", None)
        out.pop("_assistant_core_repair_needed", None)
        out.pop("_assistant_core_repair_attempted", None)
        out.pop("_assistant_core_semantic_rejection", None)
    out = _output(out, request, decision, runtime=runtime)
    return out


