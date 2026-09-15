"""B4o request-owned initial/neutral/refinement intake for the EXISTING Core.

No new semantic pipeline, authority provider, session, global monkey patch or
client flag. The existing B4j/B4k producers acquire through B4l's real readers and
retain exact selections. The Core input port consumes those selections, not an
ID/text/score search over the results. Default OFF remains in main.

Intake and optional P4 preparation share this lifetime. Preparation uses the
existing producer and three real scoped readers with explicit seed occurrences.
Generic Document/XLSX generation is optional on this same lifetime.
Structured/overview synthesis is optional on this same lifetime; validation/repair remain gated;
no legacy source-acquiring callback is used as a fallback.
The same owner remains alive for the existing scalar rescue on genuine no-source
or refusal results; technical errors cannot be converted to successful absence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable

from assistant_core_v2 import AssistantCoreV2, AssistantCoreHooks
from .request_binding import AskRequestEvidence, AskRuntimeFactories, run_core_request
from .execution import AskExecutionRuntime
from .generation import AskGenerationRuntime, GenericGenerationEvidence
from .task_generation import TaskSynthesisRuntime, TaskGenerationEvidence
from ..evidence.contracts import EvidenceContractError
from ..evidence.ask_input import _same_value, ask_request_key
from ..retrieval.ask_composition import AskEvidenceSession, AskSelection, RecordHandle
from ..retrieval.production_adapters import ProductionReaderAdapters
from ..retrieval import candidate_ranking as ranking, evidence_orchestration as orchestration
from ..retrieval import receipt_producers as producers, source_management, lexical

CORE_INTAKE_VERSION = "ask-core-intake-evidence-p6b4o-v1"


class CoreIntakeError(EvidenceContractError):
    """Technical composition failure, never a successful no_sources response."""


@dataclass(frozen=True, slots=True, repr=False)
class CoreIntakeRuntime:
    """One explicit snapshot of the existing P4 collaborators; no I/O here."""
    initial: orchestration.V13InitialRetrievalRuntime
    neutral: orchestration.AssistantCoreRetrieveNeutralRuntime
    refine: orchestration.AssistantCoreRefineRetrievalRuntime
    score: ranking.V13ScoreCandidatesRuntime
    titles: ranking.V13MergeSourceTitleCandidatesRuntime
    facets: ranking.AssistantCoreMergeFacetCandidatesRuntime
    lexical_multi: lexical.LexicalMultiQueryRuntime
    snippet: ranking.DedupCitationsBySnippetRuntime
    identifier: source_management.V13ExactIdentifierCandidatesRuntime
    prefix_query: Callable
    max_trace_records: int = 16384
    max_trace_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        types = (
            ("initial", orchestration.V13InitialRetrievalRuntime),
            ("neutral", orchestration.AssistantCoreRetrieveNeutralRuntime),
            ("refine", orchestration.AssistantCoreRefineRetrievalRuntime),
            ("score", ranking.V13ScoreCandidatesRuntime),
            ("titles", ranking.V13MergeSourceTitleCandidatesRuntime),
            ("facets", ranking.AssistantCoreMergeFacetCandidatesRuntime),
            ("lexical_multi", lexical.LexicalMultiQueryRuntime),
            ("snippet", ranking.DedupCitationsBySnippetRuntime),
            ("identifier", source_management.V13ExactIdentifierCandidatesRuntime),
        )
        if any(type(getattr(self, name)) is not cls for name, cls in types):
            raise CoreIntakeError("explicit existing intake runtimes required")
        if (not callable(self.prefix_query) or type(self.max_trace_records) is not int
                or not 1 <= self.max_trace_records <= 32768
                or type(self.max_trace_bytes) is not int
                or not 1 <= self.max_trace_bytes <= 128 * 1024 * 1024):
            raise CoreIntakeError("explicit bounded intake trace required")
        if (type(self.lexical_multi.max_lexical_queries) is not int
                or not 0 <= self.lexical_multi.max_lexical_queries <= 32):
            raise CoreIntakeError("bounded lexical query count required")


class _CoreIntake:
    """One operation using the caller's lifetime, not a second evidence owner.

    Only a producer can replace _selection. Copies supplied by the Core are
    verified against this exact previously declared selection. Stage names label
    code boundaries, not machine/language/relevance rules. No evidence/grants are
    serialized into response metadata. Retained callbacks expire at Core return.
    """
    def __init__(self, *, request, session, readers, authorize, invoke,
                 core, runtimes, runtime, preparation=None, generation=None, tasks=None):
        if (type(session) is not AskEvidenceSession
                or type(readers) is not ProductionReaderAdapters
                or type(core) is not AssistantCoreV2
                or type(core.hooks) is not AssistantCoreHooks
                or type(runtimes) is not AskRuntimeFactories
                or type(runtime) is not CoreIntakeRuntime
                or not callable(authorize) or not callable(invoke)):
            raise CoreIntakeError("existing Core/session/readers and owner required")
        if (preparation is not None
                and type(preparation) is not orchestration.AssistantCorePrepareEvidenceRuntime):
            raise CoreIntakeError("explicit existing preparation runtime required")
        if generation is not None and type(generation) is not AskGenerationRuntime:
            raise CoreIntakeError("explicit existing generation runtime required")
        if tasks is not None and (type(tasks) is not TaskSynthesisRuntime or generation is None):
            raise CoreIntakeError("task synthesis requires existing generation runtime")
        self.tasks = tasks
        self._task_input = {}
        self._task_used = set()
        self._generated_validation_selection = None
        self.generation = generation
        self._execution_snapshot = None
        self._prepared_retrieval = self._synthesis_decision = self._generation_input = None
        self._generation_used = False
        self._generated_response = self._generated_selections = None
        self.preparation, self._prepared = preparation, False
        self.request, self.session, self.readers = request, session, readers
        self.authorize, self.owner_invoke = authorize, invoke
        self.core, self.runtimes, self.runtime = core, runtimes, runtime
        self.key = ask_request_key(request)
        self.active, self.used, self.fault = False, False, None
        self._selection = None

    def _check(self, request=None):
        if self.fault is not None:
            raise self.fault
        if (not self.active or (request is not None and request is not self.request)
                or ask_request_key(self.request) != self.key
                or self.request.allowed_effective_modes != ("ask",)):
            raise CoreIntakeError("intake request/lifetime changed")

    def invoke(self, callback, request, /, *args, **kwargs):
        def work():
            try:
                self._check(request)
                result = callback(*args, **kwargs)
                self._check(request)
                return result
            except Exception as exc:
                if self.fault is None:
                    self.fault = exc
                raise self.fault
        return self.owner_invoke(work, request)

    def _current(self):
        return self.invoke(self.authorize, self.request, self.request)

    def _scope(self):
        return self.session.read_contract(request=self.request,
            current_allowed_sources=self._current())[0]

    def _parameters(self, values, *, ai_scope=False, language=False, selectors=True):
        """Reject drift BEFORE a read; returned query parameters grant nothing."""
        values = dict(values)
        scope = self._scope()
        expected = (scope.sql_selectors() if selectors else
                    {"company_id": scope.company_id, "machine_id": scope.machine_id})
        if ai_scope:
            expected = {**expected, "ai_scope": scope.ai_scope}
        if language:
            expected = {**expected, "response_language": self.request.response_language}
        if any(name not in values or not _same_value(values[name], val)
               for name, val in expected.items()):
            raise CoreIntakeError("intake callback scope differs from owned request")
        for name in expected:
            values.pop(name)
        return values

    def _values(self, handles):
        return [deepcopy(dict(item.record)) for item in self.session.records(
            request=self.request, handles=handles,
            current_allowed_sources=self._current())]

    def _read(self, reader, values, **options):
        return self.readers.candidate_handles(reader,
            **self._parameters(values, **options))

    def _dense(self, **values):
        return self.readers.dense(**self._parameters(values))

    def _prefix(self, **values):
        return self.readers.candidate_handles("read_prefix_chunk_evidence",
            **self._parameters(values), build_prefix_query=self.runtime.prefix_query)

    def _lexical(self, **values):
        """Run the original multi-query loop; dedup uses direct input positions."""
        params = self._parameters(values)
        entries, consumed, selected = {}, [], None
        def search(**kw):
            handles = self._read("read_fts_chunk_evidence", kw)
            rows = self._values(handles)
            for row, handle in zip(rows, handles):
                entries[id(row)] = (row, deepcopy(row), handle)
            consumed.extend(handles)
            return rows
        def dedup(rows, max_items):
            nonlocal selected
            if selected is not None:
                raise CoreIntakeError("duplicate lexical selection boundary")
            handles = []
            for row in rows:
                entry = entries.get(id(row))
                if entry is None or row is not entry[0] or not _same_value(row, entry[1]):
                    raise CoreIntakeError("lexical input occurrence was replaced or mutated")
                handles.append(entry[2])
            selected = producers.rank_records(request=self.request, session=self.session,
                groups=(tuple(handles),), operation="dedup_snippet", max_items=max_items,
                snippet_runtime=self.runtime.snippet, authorize=self.authorize, invoke=self.invoke)
            return self._values(selected)
        rt = replace(self.runtime.lexical_multi, search_chunks=search, dedup_citations=dedup)
        out = lexical.fts_search_chunks_multi(**self._scope().sql_selectors(),
                                              **params, runtime=rt)
        if selected is None or not _same_value(out, self._values(selected)):
            raise CoreIntakeError("lexical result differs from its declared selection")
        self.session.records(request=self.request, handles=tuple(consumed),
                             current_allowed_sources=self._current())
        return selected

    def _identifier(self, **values):
        """Keep the original exact-code algorithm and its existing dedup policy.

        The classification callback is invoked exactly once WHERE that algorithm
        has accepted/copied the current token hit (before appending its result).
        We capture that hit's handle at this call boundary. Returned IDs/text are
        never searched to discover its origin. This is a pinned trusted algorithm
        contract, not permission derived from a source-type string.
        """
        params = self._parameters(values)
        current = None
        origins, consumed = [], []
        def find(**kw):
            nonlocal current
            handles = self._read("read_token_chunk_evidence", kw)
            if len(handles) > 1:
                raise CoreIntakeError("token reader returned multiple selected occurrences")
            current = handles[0] if handles else None
            consumed.extend(handles)
            return self._values(handles)[0] if handles else None
        def classify(doc_id):
            if current is None:
                raise CoreIntakeError("identifier creation lacks current token occurrence")
            record = self._values((current,))[0]
            if not _same_value(doc_id, record.get("bubble_document_id") or ""):
                raise CoreIntakeError("identifier classification scope drift")
            origins.append(current)
            return self.runtime.identifier._source_type_from_document_id(doc_id)
        rt = replace(self.runtime.identifier, _db_find_token_chunk=find,
                     _source_type_from_document_id=classify)
        out = source_management.v13_exact_identifier_candidates(
            **self._scope().sql_selectors(), **params, runtime=rt)
        if type(out) is not list or len(out) != len(origins):
            raise CoreIntakeError("identifier creation events incomplete")
        after = self._current()
        self.session.records(request=self.request, handles=tuple(consumed),
                             current_allowed_sources=after)
        return self.session.derive_batch(request=self.request,
            views=tuple(((h,), deepcopy(row)) for h, row in zip(origins, out)),
            operation=CORE_INTAKE_VERSION + ":identifier", current_allowed_sources=after)

    def _titles(self, **values):
        params = self._parameters(values, ai_scope=True)
        # Same structural eligibility as the existing title source: this reader
        # is machine-wide only. An inapplicable probe is NOT a DB read/receipt.
        if self._scope().ai_scope != "machine_all":
            return ()
        return self.readers.candidate_handles("read_structured_title_page_evidence", **params)

    def _initial(self, **values):
        params = self._parameters(values, ai_scope=True, language=True)
        rt = replace(self.runtime.initial,
            _rrf_merge_candidates=ranking.rrf_merge_candidates,
            _v13_merge_candidates=ranking.v13_merge_candidates,
            _v13_score_candidates=self._score)
        callbacks = {
            "dense": self._dense,
            "prefix": self._prefix,
            "lexical": self._lexical,
            "identifier": self._identifier,
            "pages": lambda **kw: self._read("read_scored_page_evidence", kw),
            "preferred": lambda **kw: self._read("read_v13_preferred_page_evidence", kw),
            "structured_dense": lambda **kw: self._read("read_structured_dense_chunk_evidence", kw, selectors=False),
            "structured_direct": lambda **kw: self._read("read_structured_direct_page_evidence", kw, selectors=False),
        }
        return producers.retrieve_initial_records(request=self.request, session=self.session,
            runtime=rt, source_callbacks=callbacks, authorize=self.authorize,
            invoke=self.invoke, max_trace_records=self.runtime.max_trace_records,
            max_trace_bytes=self.runtime.max_trace_bytes, **params)

    def _score(self, q, rows, *, lineage=None):
        return ranking.v13_score_candidates(q, rows, runtime=self.runtime.score,
                                            lineage=lineage)

    def _remember(self, result, selections):
        self.session.admission(request=self.request, retrieval=result,
            selections=selections, current_allowed_sources=self._current())
        self._selection = selections  # producer supplied; never reconstructed
        return result

    def neutral(self, request):
        def work():
            result, selections = producers.retrieve_neutral_records(
                request=request, session=self.session, runtime=self.runtime.neutral,
                title_runtime=replace(self.runtime.titles,
                    _v13_merge_candidates=ranking.v13_merge_candidates,
                    _v13_score_candidates=self._score),
                initial_adapter=self._initial,
                title_adapter=self._titles,
                authorize=self.authorize, invoke=self.invoke,
                max_trace_records=self.runtime.max_trace_records,
                max_trace_bytes=self.runtime.max_trace_bytes)
            return self._remember(result, selections)
        return self.invoke(work, request)

    def refine(self, request, data, decision):
        def work():
            self.select(request, data, decision, "core.refine.input")
            result, selections = producers.refine_records(request=request,
                session=self.session, retrieval=data, selections=self._selection,
                decision=decision, runtime=replace(self.runtime.refine,
                    _v13_score_candidates=self._score),
                facet_runtime=replace(self.runtime.facets,
                    _v13_merge_candidates=ranking.v13_merge_candidates),
                initial_adapter=self._initial, authorize=self.authorize,
                invoke=self.invoke, max_trace_records=self.runtime.max_trace_records,
                max_trace_bytes=self.runtime.max_trace_bytes)
            return self._remember(result, selections)
        return self.invoke(work, request)

    def select(self, request, data, decision, stage):
        def work():
            allowed = {"core.retrieve.output", "core.route.input", "core.refine.input",
                       "core.refine.output", "core.prepare.input"}
            if stage == "core.output.collections":
                # Refusal/clarification exits create empty collections; they do
                # not assert that a source was searched or that evidence is absent.
                if any(type(rows) is not list or rows for rows in data.values()):
                    raise CoreIntakeError("output source derivation not composed")
                return tuple(AskSelection(name, ()) for name in data)
            if self._prepared:
                allowed.update({"core.prepare.output", "core.prepare_ask.input"})
            if self._prepared and self.generation is not None:
                allowed.update({"synthesis.input", "synthesis.generate"})
                if self.tasks is not None:
                    allowed.update({"synthesis.structured", "synthesis.overview"})
                if stage == "synthesis.input":
                    if not _same_value(data, self._prepared_retrieval):
                        raise CoreIntakeError("synthesis changed prepared metadata")
                    self._synthesis_decision = deepcopy(decision)
                elif stage in {"synthesis.generate", "synthesis.structured", "synthesis.overview"}:
                    if (self._synthesis_decision is None or decision != self._synthesis_decision
                            or not _same_value(data, self._synthesis_contract(decision))):
                        raise CoreIntakeError("generation changed prepared contract")
                    if stage == "synthesis.generate":
                        self._generation_input = deepcopy(data)
                    else:
                        self._task_input[stage] = deepcopy(data)
            if stage not in allowed or self._selection is None:
                raise CoreIntakeError("canonical consumer composition pending: " + str(stage))
            self.session.admission(request=request, retrieval=data,
                selections=self._selection, current_allowed_sources=self._current())
            return self._selection
        return self.invoke(work, request)

    def prepare(self, request, data, decision):
        """Same P4 policy, now with exact input and source-creation lineage.

        Catalog is a machine-wide read, inapplicable to narrower scopes. The
        anchored neighbors/sections callbacks receive handles created by the
        preparation journal, never reconstructed from text or citation IDs.
        The next consumer boundary still fails closed until it is composed.
        """
        def work():
            if self.preparation is None:
                raise CoreIntakeError("canonical preparation/consumer composition pending")
            if self._prepared:
                raise CoreIntakeError("duplicate preparation boundary")
            self.select(request, data, decision, "core.prepare.input")

            def catalog(req, **params):
                self._check(req)
                scope = self._scope()
                if scope.ai_scope != "machine_all":
                    return ()
                return self.readers.candidate_handles("read_machine_catalog_page_evidence", **params)

            def neighbors(**params):
                return self.readers.candidate_handles("read_assurance_neighbor_page_evidence",
                    **self._parameters(params, selectors=False, language=True),
                    response_language=request.response_language)

            def sections(*, request, **params):
                self._check(request)
                return self.readers.candidate_handles("read_enumeration_page_evidence", **params)

            rt = replace(self.preparation,
                _v13_merge_candidates=ranking.v13_merge_candidates,
                _assistant_core_overview_catalog_candidates=lambda rows, **kw:
                    source_management.assistant_core_overview_catalog_candidates(rows,
                        runtime=source_management.AssistantCoreOverviewCatalogCandidatesRuntime(
                            self.preparation._assistant_core_candidate_source_type), **kw))
            result, selections = producers.prepare_records(request=request,
                session=self.session, retrieval=data, selections=self._selection,
                decision=decision, runtime=rt, source_callbacks={"catalog": catalog},
                anchored_source_callbacks={"neighbors": neighbors, "sections": sections},
                authorize=self.authorize, invoke=self.invoke,
                max_trace_records=self.runtime.max_trace_records,
                max_trace_bytes=self.runtime.max_trace_bytes)
            self._remember(result["retrieval"], selections)
            self._prepared = True
            self._prepared_retrieval = deepcopy(result["retrieval"])
            return result
        return self.invoke(work, request)

    def _execution_factory(self):
        runtime = self.runtimes.execution()
        if type(runtime) is not AskExecutionRuntime or runtime.evidence_admission is not None:
            raise CoreIntakeError("fresh existing execution runtime required")
        self._execution_snapshot = runtime
        changes = {"_v13_generate_ask_response": self.generate}
        if self.tasks is not None:
            changes.update(_v13_structured_ask=self.structured_generate,
                           _assistant_core_synthesize_machine_overview=self.overview_generate)
        return replace(runtime, **changes)

    def _synthesis_contract(self, decision):
        """Verify the existing execution transform, including metadata outside lists.

        These are exactly the contract/planner assignments in synthesize_ask.
        No new semantic routing, retrieval, inference or model is performed.
        """
        if self._execution_snapshot is None or self._prepared_retrieval is None:
            raise CoreIntakeError("generation requires prepared execution snapshot")
        data = deepcopy(self._prepared_retrieval)
        contract = {
            **dict(data.get("assistant_core_contract") or {}),
            "information_task": decision.information_task,
            "required_answer_types": list(decision.required_answer_types),
            "required_facets": list(decision.required_facets),
            "facet_queries": [
                {"facet": item.facet, "answer_type": item.answer_type,
                 "must_cover": item.must_cover, "dense_queries": list(item.dense_queries),
                 "lexical_queries": list(item.lexical_queries), "exact_terms": list(item.exact_terms),
                 "preferred_source_types": list(item.preferred_source_types)}
                for item in decision.facet_queries],
            "missing_information": list(decision.missing_information), "fail_closed": True,
        }
        planner = dict(data.get("plan") or self._execution_snapshot._v13_fallback_plan(self.request.query))
        planner["information_task"] = decision.information_task
        planner["required_answer_types"] = list(decision.required_answer_types)
        planner["required_facets"] = self._execution_snapshot._dedup_text_values(
            list(planner.get("required_facets") or []) + list(decision.required_facets), limit=14)
        planner["facet_queries"] = list(contract.get("facet_queries") or [])
        planner["request_kind"] = decision.request_kind
        data.update(assistant_core_contract=contract, plan=planner)
        return data

    def generate(self, **parameters):
        def work():
            if (self.generation is None or self._generation_input is None
                    or self._generation_used):
                raise CoreIntakeError("generation requires its one prepared input")
            expected = dict(q=self.request.query, company_id=self.request.company_id,
                response_language=self.request.response_language, top_k=self.request.top_k,
                retrieval=self._generation_input, narrow_scope=self.request.narrow_scope,
                debug=self.request.debug)
            if not _same_value(parameters, expected):
                raise CoreIntakeError("generation request/scope/metadata drift")
            self._generation_used = True
            operation = GenericGenerationEvidence(request=self.request, session=self.session,
                readers=self.readers, authorize=self.authorize, invoke=self.invoke,
                runtime=self.generation, allow_structured=self.tasks is not None)
            result, selections = operation.run(parameters["retrieval"], self._selection)
            self._generated_response, self._generated_selections = deepcopy(result), selections
            return result
        return self.invoke(work, self.request)

    def _run_task(self, kind, data, decision):
        stage = "synthesis." + kind
        if (self.tasks is None or kind in self._task_used or stage not in self._task_input
                or decision != self._synthesis_decision
                or not _same_value(data, self._task_input[stage])):
            raise CoreIntakeError("task synthesis requires its exact prepared boundary")
        self._task_used.add(kind)
        operation = TaskGenerationEvidence(request=self.request, session=self.session,
            readers=self.readers, authorize=self.authorize, invoke=self.invoke,
            runtime=self.generation, tasks=self.tasks)
        response, selections, validation = operation.run_task(kind, data, self._selection, decision)
        if response is not None:
            self._generated_response = deepcopy(response)
            self._generated_selections = selections
            self._generated_validation_selection = validation
        return response

    def overview_generate(self, request, retrieval, decision):
        def work():
            if request is not self.request:
                raise CoreIntakeError("overview belongs to another request")
            return self._run_task("overview", retrieval, decision)
        return self.invoke(work, request)

    def structured_generate(self, **parameters):
        def work():
            data = self._task_input.get("synthesis.structured")
            if data is None:
                raise CoreIntakeError("structured synthesis before admission")
            r = self.request
            expected = dict(q=r.query, company_id=r.company_id, machine_id=r.machine_id,
                response_language=r.response_language, top_k=r.top_k,
                planner=data.get("plan") or {}, seed_citations=data.get("citations") or data.get("candidates") or [],
                assurance_meta=data.get("retrieval_assurance") or {}, debug=r.debug)
            if not _same_value(parameters, expected):
                raise CoreIntakeError("structured synthesis request or scope drift")
            return self._run_task("structured", data, self._synthesis_decision)
        return self.invoke(work, self.request)

    def run(self, request):
        def work():
            if self.used or request is not self.request:
                raise CoreIntakeError("core intake requires the one original request")
            self.used, self.active = True, True
            try:
                self._check(request)
                local_hooks = replace(self.core.hooks, retrieve_neutral=self.neutral,
                    refine_retrieval=self.refine, prepare_evidence=self.prepare)
                # Same implementation, ONE run inside request_binding. This
                # instance is an immutable per-request hooks container only.
                local_core = AssistantCoreV2(local_hooks)
                binding = AskRequestEvidence(session=self.session,
                    authorize=self.authorize, select=self.select)
                return run_core_request(request, core=local_core,
                    runtimes=(self.runtimes if self.generation is None else
                        AskRuntimeFactories(execution=self._execution_factory,
                                            validation=self.runtimes.validation)),
                    evidence=binding, acquisition=None)
            finally:
                self.active = False
                self._selection = None
                self._prepared = False
                self._execution_snapshot = None
                self._prepared_retrieval = self._synthesis_decision = self._generation_input = None
                self._generated_response = self._generated_selections = None
                self._generated_validation_selection = None
                self._task_input.clear()
                self._task_used.clear()
        return self.owner_invoke(work, request)


def bind_core_intake(**kwargs) -> Callable:
    """Internal factory for RequestEvidenceOwner's original session."""
    return _CoreIntake(**kwargs).run
