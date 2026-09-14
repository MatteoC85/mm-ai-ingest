"""Existing generic ASK synthesis with explicit call-time collaborators.

The algorithm and prompts are extracted unchanged from main. The optional
request binding below consumes existing admitted occurrences and registered file
references; it does not invent authority or create another Core/session. Dedicated
structured/overview synthesis, validation/repair and cache remain separate gates.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable
from ..evidence.ask_input import _same_value, ask_request_key
from ..evidence.contracts import EvidenceContractError, SourceType
from ..retrieval.ask_composition import AskEvidenceSession, AskSelection
from ..retrieval.production_adapters import ProductionReaderAdapters
from ..retrieval.residual_producers import ResidualSourceAdapters
from ..retrieval.supplemental_evidence import storage_key

GENERATION_VERSION = "ask-generic-generation-evidence-p6b4o-v1"

@dataclass(frozen=True, slots=True, repr=False)
class AskGenerationRuntime:
    """One existing dependency snapshot; no I/O at construction."""
    ASK_UI_MAX_POINTS: Any
    INFO_INTERFACE_NAVIGATION: Any
    INFO_NUMERIC_SPECIFICATION: Any
    INFO_OTHER: Any
    INFO_PROCEDURE_FULL: Any
    INFO_PROCEDURE_SEGMENT: Any
    INFO_SEQUENCE_SYNCHRONIZATION: Any
    REQ_CHECKLIST: Any
    REQ_INTERFACE_LOCATIONS: Any
    REQ_NUMERIC_VALUE: Any
    REQ_ORDERED_ACTIONS: Any
    REQ_SAFETY_CONDITIONS: Any
    REQ_STATE_SEQUENCE: Any
    V13_FAST_CONTEXT_CHARS: Any
    V13_FAST_MAX_OUTPUT_TOKENS: Any
    V13_FAST_MODEL: Any
    V13_FAST_TIMEOUT_SECONDS: Any
    V13_HEAVY_CONTEXT_CHARS: Any
    V13_HEAVY_MAX_OUTPUT_TOKENS: Any
    V13_HEAVY_MODEL: Any
    V13_HEAVY_TIMEOUT_SECONDS: Any
    V13_MAX_EVIDENCE_ITEMS_ASK: Any
    _V13BudgetExceeded: Any
    _ask_evidence_answer_schema: Callable
    _build_rg_links: Callable
    _dedup_text_values: Callable
    _finalize_ask_response_for_ui: Callable
    _localized_no_sources: Callable
    _render_grounded_answer_points: Callable
    _sanitize_citations_for_response: Callable
    _v13_assurance_prompt_block: Callable
    _v13_choose_ask_model: Callable
    _v13_extractive_fallback_answer: Callable
    _v13_json_models: Callable
    _v13_sources_block: Callable
    json: Any


def generate_ask_response(
    *,
    q: str,
    company_id: str,
    response_language: str,
    top_k: int,
    retrieval: dict,
    narrow_scope: bool,
    debug: bool,
    runtime: AskGenerationRuntime,
) -> dict:
    ASK_UI_MAX_POINTS = runtime.ASK_UI_MAX_POINTS
    INFO_INTERFACE_NAVIGATION = runtime.INFO_INTERFACE_NAVIGATION
    INFO_NUMERIC_SPECIFICATION = runtime.INFO_NUMERIC_SPECIFICATION
    INFO_OTHER = runtime.INFO_OTHER
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    INFO_SEQUENCE_SYNCHRONIZATION = runtime.INFO_SEQUENCE_SYNCHRONIZATION
    REQ_CHECKLIST = runtime.REQ_CHECKLIST
    REQ_INTERFACE_LOCATIONS = runtime.REQ_INTERFACE_LOCATIONS
    REQ_NUMERIC_VALUE = runtime.REQ_NUMERIC_VALUE
    REQ_ORDERED_ACTIONS = runtime.REQ_ORDERED_ACTIONS
    REQ_SAFETY_CONDITIONS = runtime.REQ_SAFETY_CONDITIONS
    REQ_STATE_SEQUENCE = runtime.REQ_STATE_SEQUENCE
    V13_FAST_CONTEXT_CHARS = runtime.V13_FAST_CONTEXT_CHARS
    V13_FAST_MAX_OUTPUT_TOKENS = runtime.V13_FAST_MAX_OUTPUT_TOKENS
    V13_FAST_MODEL = runtime.V13_FAST_MODEL
    V13_FAST_TIMEOUT_SECONDS = runtime.V13_FAST_TIMEOUT_SECONDS
    V13_HEAVY_CONTEXT_CHARS = runtime.V13_HEAVY_CONTEXT_CHARS
    V13_HEAVY_MAX_OUTPUT_TOKENS = runtime.V13_HEAVY_MAX_OUTPUT_TOKENS
    V13_HEAVY_MODEL = runtime.V13_HEAVY_MODEL
    V13_HEAVY_TIMEOUT_SECONDS = runtime.V13_HEAVY_TIMEOUT_SECONDS
    V13_MAX_EVIDENCE_ITEMS_ASK = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
    _V13BudgetExceeded = runtime._V13BudgetExceeded
    _ask_evidence_answer_schema = runtime._ask_evidence_answer_schema
    _build_rg_links = runtime._build_rg_links
    _dedup_text_values = runtime._dedup_text_values
    _finalize_ask_response_for_ui = runtime._finalize_ask_response_for_ui
    _localized_no_sources = runtime._localized_no_sources
    _render_grounded_answer_points = runtime._render_grounded_answer_points
    _sanitize_citations_for_response = runtime._sanitize_citations_for_response
    _v13_assurance_prompt_block = runtime._v13_assurance_prompt_block
    _v13_choose_ask_model = runtime._v13_choose_ask_model
    _v13_extractive_fallback_answer = runtime._v13_extractive_fallback_answer
    _v13_json_models = runtime._v13_json_models
    _v13_sources_block = runtime._v13_sources_block
    json = runtime.json
    contract = dict(retrieval.get("assistant_core_contract") or {})
    overview_catalog_requested = bool(contract.get("overview_catalog_requested"))
    machine_catalog_digest = str(contract.get("machine_catalog_digest") or "").strip()
    candidates = list(retrieval.get("citations") or retrieval.get("candidates") or [])
    evidence_limit = 24 if overview_catalog_requested else V13_MAX_EVIDENCE_ITEMS_ASK
    candidates = candidates[:evidence_limit]
    information_task = str(contract.get("information_task") or INFO_OTHER).strip().lower()
    required_answer_types = {
        str(x or "").strip().lower()
        for x in (contract.get("required_answer_types") or [])
        if str(x or "").strip()
    }
    if information_task == INFO_NUMERIC_SPECIFICATION:
        required_answer_types.add(REQ_NUMERIC_VALUE)
    elif information_task == INFO_INTERFACE_NAVIGATION:
        required_answer_types.add(REQ_INTERFACE_LOCATIONS)
    elif information_task == INFO_SEQUENCE_SYNCHRONIZATION:
        required_answer_types.add(REQ_STATE_SEQUENCE)
    elif information_task in {INFO_PROCEDURE_FULL, INFO_PROCEDURE_SEGMENT}:
        required_answer_types.add(REQ_ORDERED_ACTIONS)
    required_facets = _dedup_text_values(contract.get("required_facets") or [], limit=12)
    fail_closed = bool(contract.get("fail_closed"))
    if not candidates:
        return {
            "ok": True,
            "status": "no_sources",
            "answer": _localized_no_sources(response_language),
            "language": response_language,
            "citations": [],
            "rg_links": [],
            "top_k": top_k,
            "similarity_max": None,
        }

    model, effort, reasoning_mode = _v13_choose_ask_model(
        q,
        retrieval,
        narrow_scope=narrow_scope,
    )
    context_chars = V13_FAST_CONTEXT_CHARS if model == V13_FAST_MODEL else V13_HEAVY_CONTEXT_CHARS
    if overview_catalog_requested:
        context_chars = max(context_chars, 28000)
    sources_block = _v13_sources_block(candidates, max_context_chars=context_chars)
    if not sources_block:
        return {
            "ok": True,
            "status": "no_sources",
            "answer": _localized_no_sources(response_language),
            "language": response_language,
            "citations": [],
            "rg_links": [],
            "top_k": top_k,
            "similarity_max": None,
        }

    system_msg = (
        "You are MachineMind ASK, an evidence-grounded industrial documentation assistant. Use only SOURCES. "
        "Answer the exact user question, not a nearby topic. Prefer exact-machine evidence over company-general evidence when relevance is comparable. "
        "For procedures, return the actual ordered operations and conditions; for lists/tables, preserve all relevant items, labels, codes, values and units; for comparisons, keep the compared facts aligned. "
        "Structured procedure/step/P&S records are first-class evidence. Manual text may support them, but generic legal, overview, installation or safety text cannot replace a specific answer. "
        "For photo/video records, use metadata only and never claim visual/audio inspection. Do not expose citation ids or internal Bubble ids in visible text. "
        "If SOURCES do not support the answer, return no_sources. Reply in the requested language."
    )
    if REQ_NUMERIC_VALUE in required_answer_types:
        system_msg += " The answer must state the requested value with its unit/context; a nearby qualitative statement or another unrelated number is insufficient."
    if REQ_INTERFACE_LOCATIONS in required_answer_types:
        system_msg += " Name every requested screen/page/menu/location distinctly; mentioning the HMI or feature generically is insufficient."
    if REQ_STATE_SEQUENCE in required_answer_types:
        system_msg += " State the participating functions and their temporal/state order explicitly (what opens/closes/moves and when)."
    if REQ_CHECKLIST in required_answer_types:
        system_msg += " Include the requested practical checks as a compact checklist grounded in the sources."
    if REQ_SAFETY_CONDITIONS in required_answer_types:
        system_msg += " Include directly applicable authorization or safety conditions without replacing the requested technical answer."
    if overview_catalog_requested:
        system_msg += (
            " For a machine overview, MACHINE_CATALOG is the authoritative recall inventory. "
            "Inspect every catalog item before answering. Merge synonyms, but include every distinct "
            "physical assembly and explicitly documented auxiliary system relevant to the machine "
            "(including systems that score weakly against the wording of the question). Do not stop "
            "after the first overview page or image, and do not present procedure names as physical "
            "groups unless their descriptions identify the underlying assembly/system."
        )
    assurance_block = _v13_assurance_prompt_block(retrieval)
    contract_block = (
        f"INFORMATION_TASK: {information_task}\n"
        f"REQUIRED_ANSWER_TYPES: {json.dumps(sorted(required_answer_types), ensure_ascii=False)}\n"
        f"REQUIRED_FACETS: {json.dumps(required_facets, ensure_ascii=False)}\n"
    )
    catalog_block = (f"MACHINE_CATALOG:\n{machine_catalog_digest}\n\n" if machine_catalog_digest else "")
    user_msg = (
        f"QUESTION:\n{q}\n\nRESPONSE_LANGUAGE: {response_language}\n\n{contract_block}\n"
        + catalog_block
        + f"SOURCES:\n{sources_block}\n\n"
        + (f"{assurance_block}\n\n" if assurance_block else "")
        + "Return JSON only. Produce a concise but operationally complete answer, satisfy every supported required facet, and cite every point using citation_ids from SOURCES."
    )

    parsed: dict = {}
    model_used = model
    try:
        parsed, model_used = _v13_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[model, V13_FAST_MODEL] if model != V13_FAST_MODEL else [model],
            json_schema=_ask_evidence_answer_schema(),
            effort=effort,
            reasoning_mode=reasoning_mode,
            timeout=V13_HEAVY_TIMEOUT_SECONDS if model == V13_HEAVY_MODEL else V13_FAST_TIMEOUT_SECONDS,
            max_output_tokens=V13_HEAVY_MAX_OUTPUT_TOKENS if model == V13_HEAVY_MODEL else V13_FAST_MAX_OUTPUT_TOKENS,
            company_id=company_id,
            purpose="ask_final_synthesis",
        )
    except _V13BudgetExceeded:
        raise
    except Exception as exc:
        print("V13_ASK_SYNTHESIS_FAIL", str(exc)[:800])
        parsed = {"answer_status": "no_sources", "grounded_points": []}

    answer = ""
    final_citations: list[dict] = []
    synthesis_grounded = False
    if str(parsed.get("answer_status") or "").strip().lower() == "answered":
        dynamic_max_points = max(
            1,
            min(
                8,
                max(
                    int(ASK_UI_MAX_POINTS or 5),
                    len(required_facets) + (1 if len(required_answer_types) > 1 else 0),
                ),
            ),
        )
        answer, final_citations = _render_grounded_answer_points(
            grounded_points=list(parsed.get("grounded_points") or []),
            citations=candidates,
            max_points=dynamic_max_points,
            q=q,
        )
        synthesis_grounded = bool(answer and final_citations)

    if (not answer or not final_citations) and fail_closed:
        return {
            "ok": True,
            "status": "no_sources",
            "answer": _localized_no_sources(response_language),
            "language": response_language,
            "citations": [],
            "rg_links": [],
            "top_k": top_k,
            "similarity_max": (retrieval.get("metrics") or {}).get("top_similarity"),
            "chat_model": model_used,
            "meta": {
                "cacheable": False,
                "semantic_cacheable": False,
                "degraded": True,
                "degraded_reason": "assistant_core_synthesis_fail_closed",
            },
        }

    if not answer or not final_citations:
        answer, final_citations = _v13_extractive_fallback_answer(
            candidates,
            response_language=response_language,
            max_points=min(2, top_k),
            q=q,
        )

    if not answer or not final_citations:
        return {
            "ok": True,
            "status": "no_sources",
            "answer": _localized_no_sources(response_language),
            "language": response_language,
            "citations": [],
            "rg_links": [],
            "top_k": top_k,
            "similarity_max": (retrieval.get("metrics") or {}).get("top_similarity"),
            "chat_model": model_used,
            "meta": {
                "cacheable": False,
                "semantic_cacheable": False,
                "degraded": True,
                "degraded_reason": "ask_synthesis_unavailable",
            },
        }

    response_citations = _sanitize_citations_for_response(final_citations, company_id=company_id)
    try:
        rg_links = _build_rg_links(company_id, response_citations)
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
        "similarity_max": (retrieval.get("metrics") or {}).get("top_similarity"),
        "chat_model": model_used if synthesis_grounded else "v13_extractive_fallback",
        "meta": (
            {"cacheable": True, "semantic_cacheable": True}
            if synthesis_grounded
            else {
                "cacheable": False,
                "semantic_cacheable": False,
                "degraded": True,
                "degraded_reason": "ask_extractive_fallback",
            }
        ),
    }
    if debug:
        resp["debug"] = {
            "v13_ask": {
                "metrics": retrieval.get("metrics") or {},
                "plan": retrieval.get("plan") or {},
                "candidate_count": len(retrieval.get("candidates") or []),
                "evidence_ids": [c.get("citation_id") for c in candidates],
            }
        }
    return _finalize_ask_response_for_ui(resp, language=response_language)

class GenerationEvidenceError(EvidenceContractError):
    """Technical failure; never absence of evidence or a new permission."""


class GenericGenerationEvidence:
    """One source-consuming operation on the caller's existing session.

    Identity tables keep strong references to objects at their creation boundary.
    They are not indexes by citation ID, text, title, score or source key. The
    renderer's policy can choose citation IDs, but the resulting actual input
    objects retain the original occurrence handles. Copy events are explicit.
    This is a trusted-code contract, not a sandbox against malicious Python.

    This lot covers the generic Document/XLSX synthesis path. Dedicated
    structured/overview generation and downstream validation/repair remain
    pending, and main does not enable authority or cache.
    """
    def __init__(self, *, request, session, readers, authorize, invoke, runtime):
        if (type(session) is not AskEvidenceSession
                or type(readers) is not ProductionReaderAdapters
                or type(runtime) is not AskGenerationRuntime
                or not callable(authorize) or not callable(invoke)):
            raise GenerationEvidenceError("existing generation dependencies required")
        self.request, self.session, self.runtime = request, session, runtime
        self.authorize, self.owner_invoke = authorize, invoke
        self.key = ask_request_key(request)
        self.active, self.used, self.fault = False, False, None
        self.refs, self.link_refs = {}, {}
        self.selected, self.rendered = (), ()
        self.input_handles = ()
        self.provider_handles = None
        self.retrieval = self.retrieval_snapshot = None
        self.sources = ResidualSourceAdapters(request=request, session=session,
            readers=readers, authorize=authorize, invoke=self.invoke)

    def check(self):
        if self.fault is not None:
            raise self.fault
        if (not self.active or ask_request_key(self.request) != self.key
                or self.request.allowed_effective_modes != ("ask",)):
            raise GenerationEvidenceError("generation request/lifetime changed")

    def invoke(self, callback, request, /, *args, **kwargs):
        def work():
            try:
                self.check()
                if request is not self.request:
                    raise GenerationEvidenceError("generation belongs to another request")
                result = callback(*args, **kwargs)
                self.check()
                return result
            except Exception as exc:
                if self.fault is None:
                    self.fault = exc
                raise self.fault
        return self.owner_invoke(work, request)

    def current(self):
        return self.invoke(self.authorize, self.request, self.request)

    def records(self, handles):
        return self.session.records(request=self.request, handles=handles,
            current_allowed_sources=self.current())

    def _bind(self, row, handle):
        if type(row) is not dict:
            raise GenerationEvidenceError("generation needs exact record dictionaries")
        old = self.refs.get(id(row))
        if old is not None and (old[0] is not row or old[2] != handle):
            raise GenerationEvidenceError("ambiguous generation occurrence identity")
        self.refs[id(row)] = (row, deepcopy(row), handle)

    def handles(self, rows):
        if type(rows) is not list:
            raise GenerationEvidenceError("explicit generation collection required")
        handles = []
        for row in rows:
            item = self.refs.get(id(row))
            if item is None or item[0] is not row or not _same_value(row, item[1]):
                raise GenerationEvidenceError("unregistered or mutated generation occurrence")
            handles.append(item[2])
        values = self.records(tuple(handles))
        if any(not _same_value(dict(item.record), row) for item, row in zip(values, rows)):
            raise GenerationEvidenceError("generation reference differs from receipt")
        return tuple(handles)

    def _source_block(self, rows, **parameters):
        def work():
            handles = self.handles(rows)
            if self.provider_handles is not None:
                raise GenerationEvidenceError("duplicate generation source block")
            # The next lot owns structured source rendering/family expansion.
            # Do not silently treat a non-document as a document file reference.
            if any(item.context.source.source_type != SourceType.DOCUMENT
                   for item in self.records(handles)):
                raise GenerationEvidenceError("canonical structured synthesis composition pending")
            self.provider_handles = handles
            result = self.runtime._v13_sources_block(rows, **parameters)
            self.handles(rows)
            if not isinstance(result, str):
                raise GenerationEvidenceError("generation source block must be text")
            return result
        return self.invoke(work, self.request)

    def _model(self, *args, **parameters):
        def work():
            if self.provider_handles is None or not self.provider_handles:
                raise GenerationEvidenceError("model called without explicit source occurrences")
            if (parameters.get("company_id") != self.request.company_id
                    or parameters.get("purpose") != "ask_final_synthesis"):
                raise GenerationEvidenceError("generation model scope drift")
            if not _same_value(self.retrieval, self.retrieval_snapshot):
                raise GenerationEvidenceError("generation input mutated before model")
            self.records(self.input_handles)
            result = self.runtime._v13_json_models(*args, **parameters)
            self.records(self.input_handles)
            return result
        return self.invoke(work, self.request)

    def _render(self, *, grounded_points, citations, max_points, q):
        def work():
            if q != self.request.query or self.handles(citations) != self.provider_handles:
                raise GenerationEvidenceError("renderer differs from model source occurrences")
            answer, selected = self.runtime._render_grounded_answer_points(
                grounded_points=grounded_points, citations=citations,
                max_points=max_points, q=q)
            self.handles(citations)
            self.selected = self.handles(selected)
            if not isinstance(answer, str):
                raise GenerationEvidenceError("generation answer must be text")
            return answer, selected
        return self.invoke(work, self.request)

    def _file_map(self, company_id, keys):
        def work():
            if company_id != self.request.company_id:
                raise GenerationEvidenceError("generation file company drift")
            values = self.records(self.selected)
            expected = sorted({storage_key(item.context.source) for item in values})
            if not _same_value(keys, expected):
                raise GenerationEvidenceError("generation file selection drift")
            return self.sources.document_file_map(self.selected)
        return self.invoke(work, self.request)

    def _sanitize(self, rows, *, company_id):
        def work():
            if company_id != self.request.company_id or self.handles(rows) != self.selected:
                raise GenerationEvidenceError("generation sanitize selection drift")
            result = self.runtime._sanitize_citations_for_response(rows,
                company_id=company_id, file_map_fn=self._file_map)
            self.handles(rows)
            # The existing sanitizer preserves order and emits one item per
            # valid input. These inputs have already passed the reader contract.
            if type(result) is not list or len(result) != len(rows):
                raise GenerationEvidenceError("citation projection count changed")
            for parent, view in zip(rows, result):
                if (type(view) is not dict
                        or view.get("citation_id") != str(parent.get("citation_id") or "").strip()
                        or view.get("bubble_document_id") != str(parent.get("bubble_document_id") or "").strip()
                        or view.get("snippet") != (str(parent.get("snippet") or "").strip()
                            or str(parent.get("chunk_full") or "").strip())):
                    raise GenerationEvidenceError("citation projection reordered or changed source")
            self.rendered = self.session.derive_batch(request=self.request,
                views=tuple(((handle,), deepcopy(row)) for handle, row in zip(self.selected, result)),
                operation=GENERATION_VERSION+":sanitize", layout="retrieval_candidate",
                current_allowed_sources=self.current())
            for row, handle in zip(result, self.rendered):
                self._bind(row, handle)
            return result
        return self.invoke(work, self.request)

    def _links(self, company_id, rows):
        def work():
            if company_id != self.request.company_id or self.handles(rows) != self.rendered:
                raise GenerationEvidenceError("generation links selection drift")
            result = self.runtime._build_rg_links(company_id, rows, file_map_fn=self._file_map)
            self.handles(rows)
            if type(result) is not list or any(type(row) is not dict for row in result):
                raise GenerationEvidenceError("generation links invalid")
            self.link_refs = {id(row): (row, deepcopy(row)) for row in result}
            return result
        return self.invoke(work, self.request)

    def _copy_citation(self, parent, view):
        def work():
            handles = self.handles([parent])
            if handles[0] not in self.rendered:
                raise GenerationEvidenceError("UI copy does not belong to generated citations")
            handle = self.session.derive_batch(request=self.request,
                views=((handles, deepcopy(view)),), operation=GENERATION_VERSION+":ui-copy",
                layout="retrieval_candidate", current_allowed_sources=self.current())[0]
            self._bind(view, handle)
        return self.invoke(work, self.request)

    def _finalize(self, response, *, language):
        def work():
            if language != self.request.response_language:
                raise GenerationEvidenceError("generation language drift")
            if self.handles(response["citations"]) != self.rendered:
                raise GenerationEvidenceError("generation UI input drift")
            result = self.runtime._finalize_ask_response_for_ui(response,
                language=language, citation_copy_fn=self._copy_citation)
            self.handles(response["citations"])
            self.rendered = self.handles(result["citations"])
            for row in result.get("rg_links", []):
                entry = self.link_refs.get(id(row))
                if entry is None or entry[0] is not row or not _same_value(row, entry[1]):
                    raise GenerationEvidenceError("UI output contains unregistered link")
            return result
        return self.invoke(work, self.request)

    def _no_extractive_fallback(self, *args, **kwargs):
        raise GenerationEvidenceError("canonical synthesis cannot bypass fail_closed")

    def run(self, retrieval, selections):
        if self.used:
            raise GenerationEvidenceError("generation operation is single use")
        self.used, self.active = True, True
        try:
            def work():
                self.session.admission(request=self.request, retrieval=retrieval,
                    selections=selections, current_allowed_sources=self.current())
                contract = retrieval.get("assistant_core_contract") or {}
                if contract.get("fail_closed") is not True or contract.get("overview_catalog_requested"):
                    raise GenerationEvidenceError("generic fail-closed synthesis contract required")
                self.input_handles = tuple(h for selection in selections for h in selection.records)
                for selection in selections:
                    for row, handle in zip(retrieval[selection.name], selection.records):
                        self._bind(row, handle)
                original = deepcopy(retrieval)
                self.retrieval, self.retrieval_snapshot = retrieval, original
                bound = replace(self.runtime, _v13_sources_block=self._source_block,
                    _v13_json_models=self._model, _render_grounded_answer_points=self._render,
                    _sanitize_citations_for_response=self._sanitize, _build_rg_links=self._links,
                    _finalize_ask_response_for_ui=self._finalize,
                    _v13_extractive_fallback_answer=self._no_extractive_fallback)
                result = generate_ask_response(q=self.request.query,
                    company_id=self.request.company_id, response_language=self.request.response_language,
                    top_k=self.request.top_k, retrieval=retrieval,
                    narrow_scope=self.request.narrow_scope, debug=self.request.debug, runtime=bound)
                if not _same_value(retrieval, original):
                    raise GenerationEvidenceError("synthesis mutated admitted retrieval")
                self.records(self.input_handles)
                if result.get("citations"):
                    output = self.handles(result["citations"])
                    if output != self.rendered:
                        raise GenerationEvidenceError("generation output changed after finalization")
                else:
                    output = ()
                selection = (AskSelection("citations", output),)
                self.session.admission(request=self.request,
                    retrieval={"citations":result.get("citations",[])}, selections=selection,
                    current_allowed_sources=self.current())
                return result, selection
            return self.invoke(work, self.request)
        finally:
            self.active = False
            self.refs.clear(); self.link_refs.clear()
            self.input_handles = self.selected = self.rendered = ()
            self.provider_handles = None
            self.retrieval = self.retrieval_snapshot = None
