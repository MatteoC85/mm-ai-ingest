"""Protected ASK completion lifetime: one ledger through final publication.

Only the trusted ASK composition root activates this owner. OFF/RC do not use it.
The optional cache write has a child time window; expiry never authorizes content
or a cache entry. Final publication still performs its own current authority
check after any store. All child workers are joined before that check.

Cancellation is cooperative: blocking transports/OS scheduling are not preempted
by Python. A late result is vetoed, never labelled ANSWERED within the deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any
from . import phase_trace

def _monotonic() -> float:
    # The completion observer can be imported by pure request-bound Core tests.
    # Load infrastructure only when a protected lifetime actually asks for time.
    from ..infrastructure.request_budget import _monotonic as clock
    return clock()

VERSION = "ask-completion-through-publication-v1"
PHASE6_CANDIDATE_VERSION = "phase6-closure-2026-09-29-v1"
_ACTIVE = ContextVar("mm_ask_request_completion", default=None)
_CACHE_WINDOW = ContextVar("mm_ask_optional_cache_window", default=None)
_REVIEW_RESERVE = ContextVar("mm_ask_synthesis_review_reserve", default=None)
_INDEPENDENT_REVIEW_RESERVE = ContextVar("mm_ask_independent_review_reserve", default=None)
_SYNTHESIS_PURPOSES = frozenset({"ask_structured_synthesis", "ask_final_synthesis",
    "assistant_core_machine_overview_inventory"})
_REVIEW_PURPOSE = "assistant_core_answer_contract_verifier"
_KNOWN_PURPOSES = _SYNTHESIS_PURPOSES | {_REVIEW_PURPOSE,
    "assistant_core_v2_semantic_router", "assistant_core_general_technical_answer", "embedding"}


class _CacheWindowExpired(BaseException):
    """Internal cancellation, NOT a failed permission check.

    Must cross legacy broad Exception handlers without becoming a sticky
    authority-provider failure or a successful cache write. Caught only by the
    request owner's optional_cache boundary after all workers have joined.
    """


@dataclass(frozen=True, slots=True)
class _Window:
    owner: Any
    deadline: float


def current():
    owner = _ACTIVE.get()
    return owner if type(owner) is RequestCompletion and owner.active else None


def check_io_time() -> None:
    owner = current()
    if owner is None:
        return
    window = _CACHE_WINDOW.get()
    if type(window) is _Window and window.owner is owner:
        if _monotonic() >= window.deadline:
            raise _CacheWindowExpired("optional cache deadline reached")
    owner.budget.ensure_time(0.0)


def transport_timeout(configured: float) -> float:
    """Keep existing transport/accounting ceilings; only clamp to live time."""
    owner = current()
    if owner is None:
        return configured
    check_io_time()
    remaining = owner.budget.remaining()
    window = _CACHE_WINDOW.get()
    if type(window) is _Window and window.owner is owner:
        remaining = min(remaining, window.deadline - _monotonic())
    value = min(float(configured), remaining)
    if not math.isfinite(value) or value <= 0:
        check_io_time()
        from ..infrastructure.request_budget import _V13BudgetExceeded
        raise _V13BudgetExceeded("protected request deadline before transport")
    return value


def record_fence(seconds: float) -> None:
    owner = current()
    if owner is not None and type(seconds) in (int, float) and math.isfinite(seconds):
        owner.largest_fence_seconds = max(owner.largest_fence_seconds, float(seconds), 0.0)


@contextmanager
def synthesis_review_reserve(*, required: bool):
    """Reserve the EXISTING nine-second review admission threshold, not a SLA.

    Applied only to synthesis dispatch purposes. A nested recovery verifier can
    use its reserved window without inheriting a synthesis child deadline.
    """
    owner = current()
    if owner is None:
        yield
        return
    token = _REVIEW_RESERVE.set((owner, bool(required)))
    try:
        yield
    finally:
        _REVIEW_RESERVE.reset(token)


def synthesize_observed(callback, review_required):
    """Injected observer; querying the review policy is protected-request-only."""
    if current() is None:
        return callback()
    with synthesis_review_reserve(required=bool(review_required())):
        return callback()


@contextmanager
def independent_review_reserve(*, required: bool):
    """Keep 25 seconds for a later check within the SAME protected deadline.

    The composition root activates this only around the initial source-first
    factual producer. Every verifier transport attempt, including fallback,
    leaves this window plus the existing publication allowance. A nested scope
    cannot disable a reserve already required by the same live owner.
    """
    owner = current()
    if owner is None:
        yield
        return
    previous = _INDEPENDENT_REVIEW_RESERVE.get()
    inherited = previous is not None and previous[0] is owner and previous[1]
    token = _INDEPENDENT_REVIEW_RESERVE.set((owner, bool(required) or bool(inherited)))
    try:
        yield
    finally:
        _INDEPENDENT_REVIEW_RESERVE.reset(token)


def _decision_observation(diagnostics: dict) -> dict:
    """Only fixed diagnostics may survive a terminal without a fresh fence."""
    decision = diagnostics.get('independent_review_decision')
    if type(decision) is not dict:
        return {}
    checks = decision.get('checks')
    result = {'independent_review_checks': {
        key: checks[key] for key in ('source_fidelity', 'scope_applicability', 'completeness',
                                    'safety', 'language', 'provenance')
        if type(checks) is dict and type(checks.get(key)) is bool}}
    reason = decision.get('reason')
    allowed = {'accepted', 'unsupported_fact', 'missing_available_information',
        'unavailable_requested_details', 'missing_prerequisite', 'incorrect_scope',
        'incorrect_language', 'citation_problem', 'inconsistent_sources',
        'uneditable_candidate', 'other'}
    result['independent_review_reason_code'] = reason if type(reason) is str and reason in allowed else 'other'
    return result


def record_contract_observation(stage: str, contract: dict, *, diagnostics: dict | None = None) -> None:
    """Bounded enums/counts survive errors; never retain answer or source text."""
    owner = current()
    if owner is None or not isinstance(contract, dict) or len(owner.contract_checks) >= 8:
        return
    reasons = {
        "operation_boundary_incomplete", "operation_boundary_quote_or_order_invalid",
        "manual_procedure_semantic_review_incomplete", "source_step_sequence_or_notes_incomplete",
        "semantic_contract_rejected_grounded_answer", "semantic_contract_incomplete_grounded_answer",
        "missing_supported_list_items", "procedure_review_coverage_not_complete",
        "source_step_note_binding_incomplete",
        "semantic_answer_changed_after_review", "semantic_contract_incomplete",
        "semantic_contract_complete", "semantic_contract_partial_explicit",
        "source_owned_content_missing_or_changed", "reviewed_draft_binding_invalid",
        "deterministic_response_accepted_with_unresolved_facets",
        "source_step_structure_unverifiable", "procedure_family_semantic_review_incomplete",
        "supported_list_inventory_missing",
        "factual_inventory_template_invalid", "source_table_inventory_incomplete",
        "source_table_reference_invalid",
        "factual_context_review_invalid", "factual_context_review_unavailable",
        "factual_context_review_rejected", "factual_context_review_incomplete",
        "factual_context_review_completed",
    }
    outcome = contract.get("outcome")
    if type(outcome) is not str or outcome not in {"pass", "rewrite", "partial", "no_sources", "unavailable"}:
        outcome = "pass" if contract.get("passed") is True else "failed" if contract.get("passed") is False else "unknown"
    reason = contract.get("reason")
    row = {"stage": stage if type(stage) is str and stage in {"semantic_review", "final_contract"} else "other",
           "outcome": outcome, "reason": reason if type(reason) is str and reason in reasons else "other",
           "at_seconds": round(owner.budget.elapsed(), 6)}
    for key in ("semantic_answer_bound", "complete"):
        if type(contract.get(key)) is bool:
            row[key] = contract[key]
    source_preservation = contract.get("source_preservation")
    if type(source_preservation) is dict and type(source_preservation.get("all_present")) is bool:
        row["source_notes_all_present"] = source_preservation["all_present"]
    if type(diagnostics) is dict:
        for key in ("admitted_source_count", "final_citation_count", "post_review_removed_claims_count"):
            value = diagnostics.get(key)
            if type(value) is int and value >= 0:
                row[key] = min(value, 100)
        if type(diagnostics.get("post_review_grounding_applied")) is bool:
            row["post_review_grounding_applied"] = diagnostics["post_review_grounding_applied"]
    # Hashes are debug-only and computed/validated here. Never copy arbitrary
    # provider fields, response/source prose, identifiers or exception messages.
    if getattr(owner.payload, "debug", False) is True:
        if stage == "semantic_review" and type(diagnostics) is dict:
            row.update(_decision_observation(diagnostics))
            for key, values in {
                'inventory_model_outcome': {'pass', 'rewrite', 'partial', 'no_sources', 'unavailable'},
                'inventory_answer_form': {'empty', 'template', 'full', 'invalid'},
                'inventory_template_gate': {'rejected', 'accepted_or_unused'},
                'table_coverage_gate': {'rejected', 'accepted_or_unused'},
                'source_table_gate': {'unused', 'accepted', 'rewrite_required', 'inventory_shape',
                    'template_invalid', 'marker_outside_inventory', 'citation_shape',
                    'citation_not_admitted', 'source_scope_invalid', 'coverage_shape',
                    'coverage_duplicate', 'item_shape', 'reference_unknown', 'reference_duplicate',
                    'selected_record_conflict', 'required_reference_missing', 'expanded_size'},
                'independent_review_outcome': {'pass', 'rewrite', 'no_sources', 'unavailable', 'invalid'},
                'independent_review_gate': {'accepted', 'frame_mismatch', 'reply_shape',
                    'source_frame', 'provider_unavailable', 'model_rejected', 'checks_false',
                    'citation_alias', 'citation_set', 'table_count', 'table_shape', 'table_target',
                    'documentary_limits_unchecked', 'quote_shape', 'quote_source', 'quote_not_literal',
                    'table_owner', 'table_marker', 'pass_changed', 'edit_target', 'template', 'assembled_size'},
            }.items():
                value = diagnostics.get(key)
                row[key] = value if type(value) is str and value in values else 'other'
            for key in ('inventory_wire_count', 'inventory_canonical_count', 'table_expected_count', 'table_wire_count',
                        'source_table_selected_count', 'source_table_bound_record_count', 'source_table_bound_citation_count',
                        'independent_review_edit_count'):
                value = diagnostics.get(key)
                if type(value) is int and value >= 0:
                    row[key] = min(value, 100)
        if stage == "final_contract" and type(diagnostics) is dict:
            # Only digests survive a later terminal response, which may skip
            # a fresh authority fence. Never retain raw claims or source text.
            # Bound both hashing work and the request-local telemetry size.
            source = diagnostics.get("source_text")
            if type(source) is str and len(source) <= 262144:
                row["source_packet_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            initial = diagnostics.get("initial_grounding")
            if type(initial) is dict:
                for key in ("input", "redacted", "grounded", "media"):
                    value = initial.get(key)
                    if type(value) is str and len(value) <= 262144:
                        row["initial_" + key + "_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
                count = initial.get("removed_claims_count")
                if type(count) is int and count >= 0:
                    row["initial_removed_claims_count"] = min(count, 100)
            removed = diagnostics.get("post_review_removed_claims")
            if type(removed) in (list, tuple):
                hashes = []
                for value in removed[:16]:
                    if type(value) is str and 0 < len(value) <= 512:
                        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                        if digest not in hashes:
                            hashes.append(digest)
                row["removed_claim_sha256"] = hashes
        semantic = contract.get("semantic_verifier")
        if type(semantic) is dict and type(semantic.get("answer")) is str:
            row["reviewed_answer_sha256"] = hashlib.sha256(semantic["answer"].strip().encode("utf-8")).hexdigest()
        digest = contract.get("answer_sha256")
        if type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            row["final_answer_sha256"] = digest
    for key in ("covered_facets", "missing_facets", "missing_answer_facets", "missing_evidence_facets",
                "covered_answer_types", "missing_answer_types", "semantic_missing_answer_types", "missing_list_items"):
        value = contract.get(key)
        row[key + "_count"] = min(len(value), 100) if isinstance(value, (list, tuple)) else None
    boundary = contract.get("operation_boundary_validation")
    row["operation_boundary_complete"] = (boundary.get("complete") is True) if isinstance(boundary, dict) else None
    structure = contract.get("procedure_structure")
    if isinstance(structure, dict):
        for key in ("source_sequence_required", "sequence_complete", "source_notes_present"):
            row[key] = structure.get(key) if type(structure.get(key)) is bool else None
        for key in ("visible_numbers", "expected_numbers"):
            numbers = structure.get(key)
            row[key + "_count"] = min(len(numbers), 100) if isinstance(numbers, (list, tuple)) else None
    review = contract.get("procedure_review")
    if isinstance(review, dict):
        mode = review.get("reply_mode")
        row["review_reply_mode"] = mode if type(mode) is str and mode in {"retain", "edit", "replace", "reject"} else "other"
    owner.contract_checks.append(row)
    if (getattr(owner.payload, 'debug', False) is True and row['stage'] == 'semantic_review'
            and type(diagnostics) is dict and 'source_table_gate' in diagnostics):
        event = {'event': 'source_table_binding', 'outcome': row['outcome'],
                 'gate': row['source_table_gate'], 'reason': row['reason']}
        for key in ('source_table_selected_count', 'source_table_bound_record_count', 'source_table_bound_citation_count'):
            if key in row:
                event[key] = row[key]
        request_id = getattr(owner.budget, 'request_id', None)
        if (type(request_id) is str and len(request_id) == 32
                and all(char in '0123456789abcdef' for char in request_id)):
            event['request_id'] = request_id
        try:
            print('ASK_SOURCE_TABLE_BINDING', json.dumps(event, separators=(',', ':')), flush=True)
        except Exception:
            pass  # Diagnostics never change the binding or its rejection.
    if (getattr(owner.payload, "debug", False) is True and row["stage"] == "semantic_review"
            and type(diagnostics) is dict and "independent_review_gate" in diagnostics):
        # Log only the already sanitized, content-free terminal observation.
        # This remains recoverable if the response transport loses its body.
        event = {"event": "independent_review", "outcome": row["independent_review_outcome"],
                 "gate": row["independent_review_gate"], "reason": row["reason"]}
        request_id = getattr(owner.budget, "request_id", None)
        if (type(request_id) is str and len(request_id) == 32
                and all(char in "0123456789abcdef" for char in request_id)):
            event["request_id"] = request_id
        if "independent_review_edit_count" in row:
            event["edit_count"] = row["independent_review_edit_count"]
        if 'independent_review_checks' in row:
            event['failed_checks'] = [key for key, passed in row['independent_review_checks'].items() if not passed]
        if 'independent_review_reason_code' in row:
            event['independent_review_reason_code'] = row['independent_review_reason_code']
        try:
            print("ASK_INDEPENDENT_REVIEW", json.dumps(event, separators=(",", ":")), flush=True)
        except Exception:
            pass  # Observability failure never changes the answer or its gates.
    if (getattr(owner.payload, "debug", False) is True and row["stage"] == "final_contract"
            and "initial_removed_claims_count" in row):
        event = {"event": "initial_grounding", "outcome": row["outcome"], "reason": row["reason"]}
        for key in ("initial_removed_claims_count", "initial_input_sha256", "initial_redacted_sha256",
                    "initial_grounded_sha256", "initial_media_sha256", "source_packet_sha256"):
            if key in row:
                event[key] = row[key]
        request_id = getattr(owner.budget, "request_id", None)
        if (type(request_id) is str and len(request_id) == 32
                and all(char in "0123456789abcdef" for char in request_id)):
            event["request_id"] = request_id
        try:
            print("ASK_INITIAL_GROUNDING", json.dumps(event, separators=(",", ":")), flush=True)
        except Exception:
            pass


def provider_timeout(purpose: str, configured: int) -> int:
    owner = current()
    return configured if owner is None else owner.provider_timeout(purpose, configured)


class RequestCompletion:
    """One explicit outer owner, never selected by client request metadata."""
    def __init__(self, *, runtime, payload, started: float):
        from .request_flow import RequestFlowRuntime
        if type(runtime) is not RequestFlowRuntime or type(started) not in (int, float) or not math.isfinite(started):
            raise TypeError("typed runtime and finite protected entry time required")
        if current() is not None:
            raise RuntimeError("ASK_COMPLETION_ALREADY_ACTIVE")
        self.runtime, self.payload = runtime, payload
        self.budget = runtime._assistant_core_new_budget("ask", company_id=str(payload.company_id or ""))
        # The same original duration, now starting at protected entry. Never
        # extend a shorter budget supplied by the application or outer control.
        start = min(float(started), self.budget.started_monotonic)
        self.budget.started_monotonic = start
        self.budget.deadline_monotonic = min(self.budget.deadline_monotonic,
                                             start + float(self.budget.deadline_seconds))
        self.started = start
        self.active = True
        self._budget_token = runtime._V13_BUDGET_CTX.set(self.budget)
        self._owner_token = _ACTIVE.set(self)
        self.trace = phase_trace.PhaseTrace(started=start, clock=_monotonic)
        self._trace_token = phase_trace.activate(self.trace)
        self._terminal = None
        self.largest_fence_seconds = 0.0
        self.core_finished = None
        self.publication_started = self.publication_finished = None
        self.cache_state = "not_attempted"
        self.cache_seconds = 0.0
        self.publication_reserve = 0.0
        self.cache_allowance = 0.0
        self.cache_committed = False
        self.provider_schedules = []
        self.contract_checks = []

    def measure(self, name, callback, /, *args, **kwargs):
        if not self.active or current() is not self:
            raise RuntimeError("ASK_COMPLETION_EXPIRED")
        return phase_trace.call(name, callback, *args, **kwargs)

    def provider_timeout(self, purpose, configured):
        self.budget.ensure_time(0.0)
        # Observed request-local fence wall time plus existing local DB/admission
        # allowance. The floor reserves three seconds before the first fence.
        # This is a scheduling estimate; fresh final authority still decides.
        publication = max(1.5, self.largest_fence_seconds) + 1.5
        review = 0.0
        planned = _REVIEW_RESERVE.get()
        if (purpose in _SYNTHESIS_PURPOSES and planned is not None
                and planned[0] is self and planned[1]):
            review = 9.0 + max(1.5, self.largest_fence_seconds)
        independent = _INDEPENDENT_REVIEW_RESERVE.get()
        if (purpose == _REVIEW_PURPOSE and current() is self
                and independent is not None and independent[0] is self and independent[1]):
            review = 25.0
        available = self.budget.remaining() - publication - review
        minimum = 9.0 if purpose == _REVIEW_PURPOSE else 1.0
        timeout = min(int(configured), math.floor(available))
        row = {"purpose": purpose if purpose in _KNOWN_PURPOSES else "other",
               "at_seconds": round(self.budget.elapsed(), 6),
               "publication_reserve_seconds": round(publication, 6),
               "review_reserve_seconds": round(review, 6),
               "remaining_seconds": round(self.budget.remaining(), 6),
               "timeout_seconds": max(0, timeout), "admitted": timeout >= minimum}
        # At most six LLM calls and bounded embeddings; don't let diagnostics
        # create an unbounded payload if a caller violates those assumptions.
        if len(self.provider_schedules) < 64:
            self.provider_schedules.append(row)
        self.publication_reserve = max(self.publication_reserve, publication)
        if timeout < minimum:
            from ..infrastructure.request_budget import _V13BudgetExceeded
            raise _V13BudgetExceeded("protected ASK deadline lacks time for provider and required finalization")
        return timeout

    def budget_for(self, payload, runtime):
        if (not self.active or current() is not self or payload is not self.payload
                or runtime._V13_BUDGET_CTX is not self.runtime._V13_BUDGET_CTX
                or runtime._V13_BUDGET_CTX.get() is not self.budget):
            raise RuntimeError("ASK_COMPLETION_OWNERSHIP_INVALID")
        return self.budget

    def core_done(self):
        self.budget.ensure_time(0.0)
        self.core_finished = self.budget.elapsed()

    def mark_terminal(self, response):
        if (type(response) is not dict or response.get("result_code") not in {"TIMEOUT", "BUDGET_EXCEEDED"}
                or response.get("citations") or response.get("rg_links")):
            raise RuntimeError("ASK_COMPLETION_TERMINAL_INVALID")
        self._terminal = deepcopy(response)
        return response

    def timeout_response(self, exc):
        p = self.payload
        result = self.runtime._assistant_core_budget_response(
            requested_mode="ask", q=str(p.query or "").strip(),
            language=self.runtime._select_response_language(str(p.query or "").strip(), preferred=p.language),
            top_k=max(1, min(int(p.top_k or 5), self.runtime.ASK_MAX_TOP_K)),
            budget=self.budget, exc=exc)
        return self.mark_terminal(result)

    def publish(self, response, guard):
        if self._terminal is not None and response == self._terminal:
            # Only the original server-produced content-free terminal envelope
            # can avoid new source I/O after cancellation, never an arbitrary
            # success/body with a client-controlled timeout flag.
            return deepcopy(response)
        if type(response) is dict and response.get("ok") is False and response.get("status") == "error":
            return guard(response)  # original error sanitization, no content
        self.budget.ensure_time(0.0)
        self.publication_started = self.budget.elapsed()
        with phase_trace.span("flow.publication"):
            result = guard(response)
        self.budget.ensure_time(0.0)
        self.publication_finished = self.budget.elapsed()
        return result

    @contextmanager
    def optional_cache(self, *, authority_timeout: float):
        from ..infrastructure.request_budget import _REQUEST_CONTROL_CTX, _RequestControl
        self.budget.ensure_time(0.0)
        if _CACHE_WINDOW.get() is not None:
            raise RuntimeError("ASK_COMPLETION_NESTED_CACHE")
        if type(authority_timeout) not in (int, float) or not math.isfinite(authority_timeout) or authority_timeout <= 0:
            raise TypeError("configured authority timeout required")
        # Reserve the larger of the existing GET ceiling and the largest fresh
        # fence already observed in THIS request, plus the existing DB-connect
        # minimum. This is scheduling, not a grant or a statistical SLA. The final
        # guard is always checked and a late answer is still vetoed.
        reserve = max(float(authority_timeout), self.largest_fence_seconds) + 1.5
        allowance = max(0.0, self.budget.remaining() - reserve)
        self.publication_reserve, self.cache_allowance = reserve, allowance
        if allowance < float(authority_timeout):
            self.cache_state = "skipped_time_reserved_for_publication"
            yield False
            return
        began = _monotonic()
        window = _Window(self, began + allowance)
        wt = _CACHE_WINDOW.set(window)
        control = _RequestControl(allowance, allow_llm=False, parent=_REQUEST_CONTROL_CTX.get())
        ct = _REQUEST_CONTROL_CTX.set(control)
        self.cache_state = "attempted"
        try:
            yield True
            check_io_time()
            self.cache_state = "committed" if self.cache_committed else "not_stored"
        except _CacheWindowExpired:
            self.cache_state = "committed_at_window_edge" if self.cache_committed else "skipped_cache_window_expired"
        finally:
            control.cancel()
            _REQUEST_CONTROL_CTX.reset(ct)
            _CACHE_WINDOW.reset(wt)
            self.cache_seconds += max(0.0, _monotonic() - began)
        self.budget.ensure_time(0.0)

    def cache_connection(self, connect, *, entry_write=True):
        check_io_time()
        conn = connect()
        try:
            check_io_time()
            return _CacheConnection(conn, self, entry_write=entry_write)
        except BaseException:
            conn.close()
            raise

    def finish(self, response):
        if type(response) is not dict:
            raise TypeError("response mapping required")
        if str(response.get("status") or "").lower() == "answered":
            self.budget.ensure_time(0.0)
        result = self.runtime._assistant_core_attach_runtime_meta(
            response, self.budget, debug=bool(getattr(self.payload, "debug", False)))
        commit = os.environ.get("COMMIT_SHA", "").lower()
        commit = commit if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) else None
        result["meta"] = {**dict(result.get("meta") or {}), "request_completion": {
            "version": VERSION, "timing_scope": "protected_entry_through_publication",
            "elapsed_seconds": round(self.budget.elapsed(), 6),
            "deadline_seconds": float(self.budget.deadline_seconds),
            "remaining_seconds": round(self.budget.remaining(), 6),
            "core_finished_seconds": self.core_finished,
            "publication_started_seconds": self.publication_started,
            "publication_finished_seconds": self.publication_finished,
            "cache_write": self.cache_state, "cache_write_seconds": round(self.cache_seconds, 6),
            "publication_reserve_seconds": round(self.publication_reserve, 6),
            "cache_allowance_seconds": round(self.cache_allowance, 6),
            "cache_commit_observed": self.cache_committed,
            "late_success_forbidden": True,
            "provider_scheduling": list(self.provider_schedules),
            "contract_checks": list(self.contract_checks),
            "reserve_basis": "observed_fence_wall_plus_local_allowance_not_SLA",
        }, "ask_phase_trace": self.trace.summary(), "runtime_commit_sha": commit,
            "phase6_candidate_version": PHASE6_CANDIDATE_VERSION}
        # Read-only diagnostics must survive timeout/revocation. Never trigger
        # fresh authority I/O here or turn a fault snapshot into scope approval.
        from ..authority.request_admission import current_summary
        authority = current_summary()
        if authority is not None:
            existing = result["meta"].get("request_authority")
            result["meta"]["request_authority"] = {
                **authority, **(existing if type(existing) is dict else {})}
        if str(result.get("status") or "").lower() == "answered":
            # Metadata/trace serialization belongs to this request too. A
            # callback or scheduler pause after the entry check cannot release
            # an answer that the same clock already knows is late.
            self.budget.ensure_time(0.0)
        return result

    def close(self):
        if not self.active:
            return
        self.active = False
        try:
            self.runtime._V13_BUDGET_CTX.reset(self._budget_token)
        finally:
            try:
                phase_trace.deactivate(self._trace_token)
            finally:
                _ACTIVE.reset(self._owner_token)
                self.contract_checks.clear()
                self._terminal = self.payload = self.runtime = None


class _CacheConnection:
    """Keep the original SQL/transaction; check the child window between steps."""
    def __init__(self, conn, owner, *, entry_write=True):
        self._conn, self._owner = conn, owner
        self._entry_write = entry_write
        self._committed = False
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def cursor(self, *args, **kwargs):
        check_io_time()
        return _CacheCursor(self._conn.cursor(*args, **kwargs))
    def commit(self):
        check_io_time()
        self._conn.commit()
        self._committed = True
        if self._entry_write:
            self._owner.cache_committed = True
        check_io_time()
    def rollback(self):
        return self._conn.rollback()
    def close(self):
        try:
            if not self._committed:
                self._conn.rollback()
        finally:
            self._conn.close()


class _CacheCursor:
    def __init__(self, cursor):
        self._cursor = cursor
    def __getattr__(self, name):
        return getattr(self._cursor, name)
    def __enter__(self):
        self._cursor.__enter__()
        return self
    def __exit__(self, *args):
        return self._cursor.__exit__(*args)
    def execute(self, *args, **kwargs):
        check_io_time()
        value = self._cursor.execute(*args, **kwargs)
        check_io_time()
        return value
