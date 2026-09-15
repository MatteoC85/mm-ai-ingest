"""Request-owned validation, recovery and one repair cycle on existing evidence.

This observer invokes the existing EXECUTION/VALIDATION algorithms exactly once.
No model, retrieval, authority provider, Core or session is duplicated. Every
candidate copy names its original object at the copy site; selection by a model
citation ID remains policy, never a way of discovering an occurrence handle.

The immutable Core's final presentation is checked against its own pure helpers.
The chosen validated envelope is explicit (first result or one bounded repair),
then citation deduplication records input positions. No text/value search grants
permission or chooses an origin. The OFF/default functions have no observer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from assistant_core_v2 import (
    _choose_monotonic_response, _finish_ask_validation, _clean_text,
    response_has_rejected_answer, decorate_response,
)
from .task_generation import TaskGenerationEvidence
from .generation import GenerationEvidenceError
from ..evidence.ask_input import _same_value, apply_ask_evidence_input
from ..retrieval.ask_composition import AskSelection
from ..retrieval import source_management

RESPONSE_EVIDENCE_VERSION = "ask-response-evidence-p6b4o-v1"


class ResponseEvidenceError(GenerationEvidenceError):
    """Technical failure on the owned request, never successful source absence."""


class ResponseEvidenceFlow(TaskGenerationEvidence):
    """A bounded consumer journal on the SAME RequestEvidenceOwner session.

    The inherited journal already registers typed file reads, projections, and
    explicit copies used by structured/overview. This class extends its lifetime
    across validation/repair; it never manufactures a new evidence registry.
    """
    _STAGES = frozenset({
        "synthesis.input", "synthesis.overview", "synthesis.structured", "synthesis.generate",
        "recovery.input", "verifier.input", "verifier.provider", "repair.retrieval",
        "repair.validation", "validation.retrieval", "validation.private",
        "citation_recovery.retrieval", "citation_recovery.private",
        "citation_recovery.selected", "citation_recovery.procedure_before_sanitize",
        "citation_recovery.procedure_after_sanitize", "citation_recovery.procedure_final",
        "citation_recovery.generic_before_sanitize", "citation_recovery.generic_after_sanitize",
        "validation.semantic_citations", "validation.output_citations", "validation.output_private",
    })

    def __init__(self, owner):
        super().__init__(request=owner.request, session=owner.session, readers=owner.readers,
            authorize=owner.authorize, invoke=owner.invoke, runtime=owner.generation,
            tasks=owner.tasks, allow_structured=True)
        self.owner = owner
        self.active = True
        self.pending = None
        self.stack = []
        self.decision = None
        self.execution = self.validation = None
        self.synthesis = self.repaired = None
        self.validated = []
        self.last_produced = self.last_empty = None
        self.completed = False
        self.final_data, self.final_handles = None, None
        self.model_inputs = None
        self.verifier_calls = 0
        self.synthesis_calls = self.repair_calls = 0

    def _seed(self, data, selections):
        self.session.admission(request=self.request, retrieval=data, selections=selections,
                               current_allowed_sources=self.current())
        for sel in selections:
            if len(data[sel.name]) != len(sel.records):
                raise ResponseEvidenceError("consumer boundary count drift")
            for row, handle in zip(data[sel.name], sel.records):
                old = self.refs.get(id(row))
                if old is not None:
                    if old[0] is not row or old[2] != handle or not _same_value(row, old[1]):
                        raise ResponseEvidenceError("consumer boundary occurrence changed")
                else:
                    self._remember(row, handle)

    def _selections(self, data):
        return tuple(AskSelection(name, self.handles(data[name]))
                     for name in ("candidates", "citations") if name in data)

    def _envelope(self, response):
        if type(response) is not dict:
            raise ResponseEvidenceError("consumer response must be a dictionary")
        self.handles(response.get("citations", []))
        self.handles(response.get("_assistant_core_validation_evidence", []))
        for link in response.get("rg_links", []):
            ref = self.link_refs.get(id(link))
            if ref is None or ref[0] is not link or not _same_value(link, ref[1]):
                raise ResponseEvidenceError("consumer response has an unregistered link")
        if response.get("candidates"):
            raise ResponseEvidenceError("uncomposed response candidate collection")
        self.records(self.all_handles)

    def _snapshot(self, response):
        self._envelope(response)
        return (deepcopy(response), self.handles(response.get("citations", [])),
                self.handles(response.get("_assistant_core_validation_evidence", [])))

    def generated(self, response, selections, private):
        """Explicit handoff from an existing generation producer, not a search."""
        def work():
            self._seed({"citations": response.get("citations", [])}, selections)
            self._seed({"candidates": response.get("_assistant_core_validation_evidence", [])},
                       private or (AskSelection("candidates", ()),))
            for row in response.get("rg_links", []):
                if type(row) is not dict:
                    raise ResponseEvidenceError("generated link shape changed")
                self._bounded(row)
                self.link_refs[id(row)] = (row, deepcopy(row))
            self.last_produced = self._snapshot(response)
        return self.invoke(work, self.request)

    def empty(self, callback, request, decision, retrieval):
        def work():
            if request is not self.request:
                raise ResponseEvidenceError("empty response belongs to another request")
            result = callback(request, decision, retrieval)
            if type(result) is not dict or any(result.get(k) for k in
                    ("candidates", "citations", "rg_links", "_assistant_core_validation_evidence")):
                raise ResponseEvidenceError("no-evidence builder emitted source records")
            self.last_empty = deepcopy(result)
            self.last_produced = self._snapshot(result)
            return result
        return self.invoke(work, request)

    def admit(self, request, data, decision, stage, admission):
        def work():
            if request is not self.request or stage not in self._STAGES or not self.stack:
                raise ResponseEvidenceError("uncomposed consumer admission stage")
            if decision != self.decision or self.pending is not None:
                raise ResponseEvidenceError("consumer decision/admission re-entry drift")
            original = deepcopy(data)
            selections = self._selections(data)
            # Synthesis metadata transform is independently checked by the existing
            # intake contract before delegating this boundary's exact selections.
            if stage == "synthesis.input":
                if not _same_value(data, self.owner._prepared_retrieval):
                    raise ResponseEvidenceError("synthesis changed prepared retrieval")
                self.owner._synthesis_decision = deepcopy(decision)
            elif stage in {"synthesis.overview", "synthesis.structured", "synthesis.generate"}:
                if not _same_value(data, self.owner._synthesis_contract(decision)):
                    raise ResponseEvidenceError("synthesis changed prepared contract")
                if stage == "synthesis.generate":
                    self.owner._generation_input = deepcopy(data)
                else:
                    self.owner._task_input[stage] = deepcopy(data)
            self.pending = {"stage": stage, "data": original, "selections": selections,
                            "decision": deepcopy(decision), "used": False}
            try:
                admitted = admission(data)
                if not self.pending["used"] or not _same_value(data, original):
                    raise ResponseEvidenceError("consumer admission callback changed input")
                result = apply_ask_evidence_input(original, request_key=self.key, admission=admitted)
                self._seed(result, selections)
                return result
            finally:
                self.pending = None
        return self.invoke(work, request)

    def select_pending(self, request, data, decision, stage):
        self.check()
        p = self.pending
        if (p is None or p["used"] or request is not self.request or stage != p["stage"]
                or decision != p["decision"] or not _same_value(data, p["data"])):
            raise ResponseEvidenceError("consumer selection lacks exact pending boundary")
        p["used"] = True
        return p["selections"]

    def consume(self, name, parameters, callback):
        def work():
            request, decision, runtime = (parameters["request"], parameters["decision"], parameters["runtime"])
            er = getattr(runtime, "execution_runtime", None) or runtime
            if (request is not self.request or not callable(getattr(er, "evidence_admission", None))
                    or getattr(runtime, "evidence_observer", None) is not self
                    or getattr(er, "evidence_observer", None) is not self
                    or decision.effective_mode != "ask"):
                raise ResponseEvidenceError("observer requires the existing request-bound ASK runtime")
            if self.decision is None:
                self.decision = deepcopy(decision)
            if decision != self.decision:
                raise ResponseEvidenceError("consumer changed the routing decision")
            parent = self.stack[-1] if self.stack else None
            allowed = {
                "synthesize_ask": {None}, "validate_response": {None},
                "repair_response": {None}, "recover_citations": {"validate_response"},
                "recover_ask_from_evidence": {"synthesize_ask"},
                "verify_or_repair_answer": {"recover_ask_from_evidence", "validate_response", "repair_response"},
            }
            if name not in allowed or parent not in allowed[name] or self.completed:
                raise ResponseEvidenceError("invalid response consumer transition")
            if name == "synthesize_ask":
                if self.synthesis_calls or not _same_value(parameters["retrieval"], self.owner._prepared_retrieval):
                    raise ResponseEvidenceError("synthesis must consume its one preparation")
                self.synthesis_calls += 1
                self._seed(parameters["retrieval"], self.owner._selection)
            elif name in {"validate_response", "repair_response"}:
                if not _same_value(parameters["retrieval"], self.owner._prepared_retrieval):
                    raise ResponseEvidenceError("validation/repair changed prepared retrieval")
                self._seed(parameters["retrieval"], self.owner._selection)
                if name == "validate_response":
                    if self.synthesis is None:
                        raise ResponseEvidenceError("validation must follow observed synthesis")
                    if len(self.validated) >= 2 or (self.validated and self.repaired is None):
                        raise ResponseEvidenceError("at most one validation then one revalidation")
                    expected = deepcopy((self.repaired if self.validated else self.synthesis)[0])
                    if self.validated:
                        expected["_assistant_core_repair_attempted"] = True
                else:
                    if self.repair_calls or len(self.validated) != 1:
                        raise ResponseEvidenceError("one repair after first validation required")
                    expected = deepcopy(self.validated[0][0])
                    if not expected.get("_assistant_core_repair_needed") or expected.get("_assistant_core_repair_attempted"):
                        raise ResponseEvidenceError("repair requires an incomplete validated answer")
                    self.repair_calls += 1
                if not _same_value(parameters["response"], expected):
                    raise ResponseEvidenceError("consumer response differs from the preceding envelope")
                self._envelope(parameters["response"])
            elif name == "verify_or_repair_answer":
                self.handles(parameters["candidates"])
                self.model_inputs = None
            self.stack.append(name)
            try:
                result = callback()
                if name == "synthesize_ask":
                    if self.last_produced is None:
                        raise ResponseEvidenceError("synthesis returned without a registered producer")
                    expected = deepcopy(self.last_produced[0])
                    if "information_task" in result:
                        expected.setdefault("information_task", decision.information_task)
                    if not _same_value(result, expected):
                        raise ResponseEvidenceError("synthesis changed its producer response")
                    self.synthesis = self._snapshot(result)
                elif name == "validate_response":
                    self.validated.append(self._snapshot(result))
                elif name == "repair_response":
                    if "_assistant_core_repair_retrieval" in result:
                        raise ResponseEvidenceError("repair may not inject unregistered retrieval")
                    self.repaired = self._snapshot(result)
                elif name == "recover_ask_from_evidence" and result is not None:
                    self.last_produced = self._snapshot(result)
                elif name == "recover_citations":
                    citations, links = result
                    self._envelope({"citations": citations, "rg_links": links})
                elif name == "verify_or_repair_answer":
                    if not isinstance(result, dict):
                        raise ResponseEvidenceError("verifier response shape invalid")
                    if str(result.get("outcome") or "").lower() not in {"pass", "rewrite", "partial", "no_sources", "unavailable"}:
                        raise ResponseEvidenceError("canonical verifier outcome invalid")
                    if str(result.get("outcome") or "").lower() == "unavailable":
                        raise ResponseEvidenceError("canonical verifier unavailable: " + str(result.get("reason") or "unknown"))
                self.records(self.all_handles)
                return result
            finally:
                self.stack.pop()
        return self.invoke(work, parameters["request"])

    def _catalog(self, rows):
        def work():
            parents, events = self.handles(rows), []
            result = source_management.assistant_core_overview_catalog_candidates(rows,
                runtime=source_management.AssistantCoreOverviewCatalogCandidatesRuntime(
                    self.execution._assistant_core_candidate_source_type), lineage=events.append)
            if len(events) != 1 or len(events[0]) != len(result):
                raise ResponseEvidenceError("verifier catalog selection missing lineage")
            for row, positions in zip(result, events[0]):
                handles = tuple(parents[index] for group, index in positions if group == 0)
                if len(handles) != len(positions):
                    raise ResponseEvidenceError("catalog parent group invalid")
                h = self.session.derive_batch(request=self.request, views=((handles, deepcopy(row)),),
                    operation=RESPONSE_EVIDENCE_VERSION+":catalog", current_allowed_sources=self.current())[0]
                self._remember(row, h)
            return result
        return self.invoke(work, self.request)

    def _digest(self, rows):
        def work():
            self.handles(rows)
            result = self.execution._assistant_core_machine_catalog_digest(rows)
            self.handles(rows)
            return result
        return self.invoke(work, self.request)

    def _verification_sources(self, rows, **parameters):
        def work():
            if not self.stack or self.stack[-1] != "verify_or_repair_answer" or self.model_inputs is not None:
                raise ResponseEvidenceError("one verifier sources block per call required")
            self.model_inputs = self.handles(rows)
            result = self.execution._v13_sources_block(rows, **parameters)
            self.handles(rows)
            return result
        return self.invoke(work, self.request)

    def _verification_model(self, *args, **parameters):
        def work():
            if (not self.model_inputs or not self.stack or self.stack[-1] != "verify_or_repair_answer"
                    or parameters.get("company_id") != self.request.company_id
                    or parameters.get("purpose") != "assistant_core_answer_contract_verifier"):
                raise ResponseEvidenceError("verifier model lacks its scoped source occurrences")
            self.records(self.all_handles)
            self.verifier_calls += 1
            result = self.execution._v13_json_models(*args, **parameters)
            self.records(self.all_handles)
            return result
        return self.invoke(work, self.request)

    def execution_runtime(self, runtime):
        if runtime.evidence_observer is not None:
            raise ResponseEvidenceError("fresh execution observer required")
        self.execution = runtime
        return replace(runtime, evidence_observer=self,
            _sanitize_citations_for_response=self._sanitize, _build_rg_links=self._links,
            _v13_json_models=self._verification_model, _v13_sources_block=self._verification_sources,
            _v13_merge_candidates=self.merge_candidates,
            _assistant_core_overview_catalog_candidates=self._catalog,
            _assistant_core_machine_catalog_digest=self._digest,
            _assistant_core_build_no_evidence=lambda req, dec, data:
                self.empty(runtime._assistant_core_build_no_evidence, req, dec, data))

    def validation_runtime(self, runtime):
        if runtime.evidence_observer is not None:
            raise ResponseEvidenceError("fresh validation observer required")
        self.validation = runtime
        return replace(runtime, evidence_observer=self,
            _sanitize_citations_for_response=self._sanitize, _build_rg_links=self._links,
            _procedure_ui_order_citations=lambda rows:
                self.tasks._procedure_ui_order_citations(rows, trace=self),
            _assistant_core_build_no_evidence=lambda req, dec, data:
                self.empty(runtime._assistant_core_build_no_evidence, req, dec, data))

    def finish(self, request, result):
        def work():
            if self.completed or self.stack or request is not self.request:
                raise ResponseEvidenceError("final response boundary re-entry")
            collections = {k: result[k] for k in ("candidates", "citations") if k in result}
            if not self.synthesis_calls:
                if any(collections.values()) or result.get("rg_links"):
                    raise ResponseEvidenceError("early Core exit contains uncomposed sources")
                selection = tuple(AskSelection(k, ()) for k in collections)
            else:
                if not self.validated:
                    raise ResponseEvidenceError("source response bypassed validation")
                chosen = self.validated[0]
                out = deepcopy(chosen[0])
                if len(self.validated) == 2:
                    out = _choose_monotonic_response(self.validated[0][0], self.validated[1][0])
                    selection_name = out["meta"]["monotonic_repair"]["selected"]
                    chosen = (self.validated[1] if selection_name == "candidate" else
                              self.validated[0] if selection_name == "baseline" else ({}, (), ()))
                if (not _clean_text(out.get("status"), 40) or
                    (_clean_text(out.get("status"), 40).lower() == "answered" and response_has_rejected_answer(out))):
                    if self.last_empty is None:
                        raise ResponseEvidenceError("final rejected draft lacks observed no-evidence builder")
                    selection_meta = dict(out.get("meta") or {})
                    out = deepcopy(self.last_empty)
                    from assistant_core_v2 import REPAIR_SELECTION_POLICY
                    out["meta"] = {**selection_meta, **dict(out.get("meta") or {}),
                        "answer_acceptance_guard": {"policy_version": REPAIR_SELECTION_POLICY,
                                                     "reason": "no_accepted_answer_after_validation"}}
                    chosen = (out, (), ())
                out = _finish_ask_validation(out)
                expected = decorate_response(out, self.request, self.decision)
                if not _same_value(result, expected):
                    raise ResponseEvidenceError("Core result differs from its validated choice/presentation")
                # Exactly the positional dedup in Core._align_evidence_manifest.
                # IDs decide presentation duplicates, NEVER authority/parent lookup.
                rows, handles, seen = [], [], set()
                incoming = out.get("citations", [])
                parents = chosen[1] if incoming else ()
                if len(incoming) != len(parents):
                    raise ResponseEvidenceError("final citation count lacks declared parents")
                for row, handle in zip(incoming, parents):
                    cid = _clean_text(row.get("citation_id"), 260)
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    rows.append(dict(row)); handles.append(handle)
                if not _same_value(result.get("citations", []), rows):
                    raise ResponseEvidenceError("final citation projection changed")
                selection = tuple(AskSelection(k, tuple(handles) if k == "citations" else ()) for k in collections)
            self.session.admission(request=request, retrieval=collections, selections=selection,
                                   current_allowed_sources=self.current())
            self.final_data, self.final_handles = deepcopy(collections), selection
            self.completed = True
        return self.invoke(work, request)

    def final_selection(self, data):
        self.check()
        if not self.completed or not _same_value(data, self.final_data):
            raise ResponseEvidenceError("unverified final source projection")
        return self.final_handles

    def close(self):
        self.active = False
        self.refs.clear(); self.link_refs.clear(); self.drafts.clear()
        self.raw_refs.clear(); self.inventory_refs.clear()
        self.all_handles = self.input_handles = ()
        self.pending = None
        self.last_produced = self.last_empty = self.synthesis = self.repaired = None
        self.validated.clear(); self.stack.clear()
        self.final_data = self.final_handles = None
        self.model_inputs = None
