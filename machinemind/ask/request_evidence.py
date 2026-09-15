"""B4o: one request-owned B4a/B4l lifetime and the real B4m scalar producer.

This is the scalar-rescue integration lot, not complete canonical ASK. Main
installs it only in the guarded HTTP path (authority still OFF by default).
The existing request_flow scalar response/policy is reused without copying it.
An optional internal Core factory receives THIS session and guarded readers.
No extra lifetime is created. Complete consumers/cache/activation remain separate
integration work; public metadata continues to report canonical=False.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from .request_flow import (RequestFlowRuntime, RequestFlowEvidenceBinding,
                           precision_fact_rescue)
from ..authority.policy import RequestAuthority
from ..evidence.adapter_types import AdapterLimits
from ..evidence.assembly import AssemblyLimits
from ..evidence.ask_input import ask_request_key, _same_value
from ..evidence.contracts import EvidenceContractError
from ..evidence.legacy_compatibility import LegacyLimits
from ..evidence.manifest import ManifestLimits
from ..retrieval.ask_composition import AskEvidenceSession, AskSelection, AskSessionLimits
from ..retrieval.chunk_evidence import ChunkEvidenceLimits, ChunkReadScope
from ..retrieval.production_adapters import ProductionReaderAdapters
from ..retrieval.residual_producers import (ResidualSourceAdapters,
    resolve_precision_fact_evidence)
from ..retrieval import precision_facts
from ..retrieval.precision_facts import PrecisionFactRuntime
from ..retrieval.supplemental_evidence import storage_key

REQUEST_EVIDENCE_VERSION = "ask-request-scalar-evidence-p6b4o-v1"


def precision_session_limits(runtime: PrecisionFactRuntime) -> AskSessionLimits:
    """Explicit allocation caps, never a heap estimate, price or truncation rule.

    A read can retain 1,024 occurrences (current main scan limit is <=900).
    The lifetime retains at most 32 receipts, 8,192 views and 64 MiB serialized.
    A capacity error fails technically. Current authority's transport meter and
    the existing request/model budget remain independent and unchanged.
    """
    if (type(runtime) is not PrecisionFactRuntime
            or type(runtime.page_text_chars) is not int or runtime.page_text_chars < 1
            or type(runtime.page_scan_limit) is not int
            or not 1 <= runtime.page_scan_limit <= 1024):
        raise EvidenceContractError("explicit bounded precision runtime required")
    chars = max(12000, runtime.page_text_chars)
    return AskSessionLimits(
        evidence=ChunkEvidenceLimits(
            adapter=AdapterLimits(chars, 128, max(65536, chars * 4)),
            assembly=AssemblyLimits(ManifestLimits(1024, 32 * 1024 * 1024),
                                    1024, 48 * 1024 * 1024),
            legacy=LegacyLimits(24, 262144, 64 * 1024 * 1024)),
        max_reads=32, max_records=8192, max_bytes=64 * 1024 * 1024)


class RequestEvidenceOwner:
    """Own one original request/session, not a grant reconstructed from sources.

    Reader factory must return the existing RequestAuthority and all concrete
    ProductionReaderAdapters using THIS session and invocation guard. All errors
    in invoked callbacks latch, including errors swallowed by legacy scalar
    rendering. No source text, URL, grant or handle is exported in public metadata.
    Retained callbacks expire on close; failed construction closes partial state.
    The normal HTTP worker executes this owner synchronously, not reentrantly.
    """
    def __init__(self, *, request: Any, scope: ChunkReadScope,
                 limits: AskSessionLimits, readers_factory: Callable,
                 flow_runtime: RequestFlowRuntime,
                 precision_runtime: PrecisionFactRuntime,
                 core_factory: Callable | None = None,
                 response_observer: Callable | None = None):
        if (type(scope) is not ChunkReadScope or type(limits) is not AskSessionLimits
                or not callable(readers_factory)
                or type(flow_runtime) is not RequestFlowRuntime
                or type(precision_runtime) is not PrecisionFactRuntime
                or (core_factory is not None and not callable(core_factory))
                or (response_observer is not None and not callable(response_observer))
                or getattr(request, "requested_mode", None) != "ask"
                or getattr(request, "allowed_effective_modes", None) != ("ask",)):
            raise EvidenceContractError("typed ASK request evidence dependencies required")
        self._request, self._scope = request, scope
        self._key = ask_request_key(request)
        self._active, self._fault = True, None
        self._authority = self._readers = self._sources = None
        self._flow = flow_runtime
        self._precision = precision_runtime
        self._selected = None
        self._rendered = ()
        self._busy = False
        self._run_core = None
        self._response_observer = response_observer
        self.session = AskEvidenceSession(request=request, scope=scope, limits=limits)
        try:
            pair = readers_factory(request=request, session=self.session, invoke=self.invoke)
            if type(pair) is not tuple or len(pair) != 2:
                raise EvidenceContractError("concrete authority and reader pair required")
            self._authority, self._readers = pair
            if (type(self._authority) is not RequestAuthority
                    or type(self._readers) is not ProductionReaderAdapters):
                raise EvidenceContractError("existing production authority/readers required")
            self._sources = ResidualSourceAdapters(request=request, session=self.session,
                readers=self._readers, authorize=self.authorize, invoke=self.invoke)
            if core_factory is not None:
                self._run_core = self.invoke(core_factory, request,
                    request=request, session=self.session, readers=self._readers,
                    authorize=self.authorize, invoke=self.invoke)
                if not callable(self._run_core):
                    raise EvidenceContractError("request-owned Core factory must return callable")
            self.check()
        except BaseException:
            self.close()
            raise

    def _observe_response(self, result, scalar_target=None):
        if self._response_observer is not None:
            self.check()
            observed = deepcopy(result)
            # run_sync copies this already-established target after the Core or
            # scalar result. Observe that exact presentation projection too.
            target = (self._request.metadata or {}).get(
                precision_facts.SCALAR_TARGET_KEY) if scalar_target is None else scalar_target
            if isinstance(target, dict):
                observed["meta"] = {**dict(observed.get("meta") or {}),
                    precision_facts.SCALAR_TARGET_KEY: deepcopy(target)}
            self._response_observer(observed,
                self.session.cache_dependencies(request=self._request))
            self.check()
        return result

    def _observed_core(self, request):
        return self.invoke(lambda: self._observe_response(self._run_core(request)), request)

    def check(self) -> None:
        if self._fault is not None:
            raise self._fault
        try:
            if (not self._active or ask_request_key(self._request) != self._key
                    or self._request.allowed_effective_modes != ("ask",)):
                raise EvidenceContractError("request evidence lifetime expired or request changed")
            if self.session.summary()["closed"]:
                raise EvidenceContractError("request evidence session was closed")
        except Exception as exc:
            self._fault = exc
            raise

    def invoke(self, callback, request, /, *args, **kwargs):
        self.check()
        try:
            if request is not self._request or not callable(callback):
                raise EvidenceContractError("callback belongs to another request")
            result = callback(*args, **kwargs)
            self.check()
            return result
        except Exception as exc:
            if self._fault is None:
                self._fault = exc
            raise self._fault

    def authorize(self, request):
        current = self.invoke(self._authority, request, request)
        self.session.read_contract(request=request, current_allowed_sources=current)
        return current

    def resolve_precision_fact(self, *, query, company_id, machine_id, doc_ids,
                               bubble_document_id, runtime, answer_contract=None):
        """Internal dependency adapter for the UNCHANGED scalar response builder."""
        def work():
            if (not self._busy or query != self._request.query
                    or company_id != self._scope.company_id
                    or machine_id != self._scope.machine_id
                    or tuple(doc_ids or ()) != self._scope.document_ids
                    or (bubble_document_id or None) != self._scope.bubble_document_id
                    or runtime is not self._precision):
                raise EvidenceContractError("precision callback scope/runtime drift")
            self._selected = resolve_precision_fact_evidence(
                request=self._request, session=self.session, sources=self._sources,
                query=query, target_machine_id=machine_id,
                page_scan_limit=self._precision.page_scan_limit,
                answer_contract=deepcopy(answer_contract),
                authorize=self.authorize, invoke=self.invoke)
            return None if self._selected is None else self._selected.resolution
        return self.invoke(work, self._request)

    def resolution_to_candidate(self, resolution):
        def work():
            if (not self._busy or self._selected is None
                    or resolution is not self._selected.resolution):
                raise EvidenceContractError("unregistered scalar resolution")
            current = self.authorize(self._request)
            values = self.session.records(request=self._request,
                handles=(self._selected.candidate,), current_allowed_sources=current)
            return deepcopy(dict(values[0].record))
        return self.invoke(work, self._request)

    def file_map(self, company_id, document_ids):
        """Existing presentation file lookup, bound to the selected occurrence.

        Requested keys are only checked against already-selected handles. They
        cannot supply authority or choose another document from a cached URL.
        """
        def work():
            if not self._busy or self._selected is None or company_id != self._scope.company_id:
                raise EvidenceContractError("scalar file-map scope/lifetime drift")
            current = self.authorize(self._request)
            values = self.session.records(request=self._request,
                handles=(self._selected.candidate,), current_allowed_sources=current)
            expected = [storage_key(values[0].context.source)]
            if not _same_value(document_ids, expected):
                raise EvidenceContractError("scalar file-map selection drift")
            return self._sources.document_file_map((self._selected.candidate,))
        return self.invoke(work, self._request)

    def _sanitize(self, citations, *, company_id):
        def work():
            if not self._busy or company_id != self._scope.company_id:
                raise EvidenceContractError("scalar presentation scope drift")
            candidate = self.resolution_to_candidate(self._selected.resolution)
            if not _same_value(citations, [candidate]):
                raise EvidenceContractError("scalar renderer input differs from selected occurrence")
            result = self._flow._sanitize_citations_for_response(
                deepcopy(citations), company_id=company_id, file_map_fn=self.file_map)
            if type(result) is not list or len(result) != 1 or type(result[0]) is not dict:
                raise EvidenceContractError("scalar citation rendering failed")
            current = self.authorize(self._request)
            # The renderer has exactly ONE input. Parent is passed explicitly at
            # creation, never recovered by a citation/source ID or text match.
            self._rendered = self.session.derive_batch(request=self._request,
                views=(((self._selected.candidate,), deepcopy(result[0])),),
                operation=REQUEST_EVIDENCE_VERSION + ":presentation",
                layout="retrieval_candidate", current_allowed_sources=current)
            return result
        return self.invoke(work, self._request)

    def _links(self, company_id, citations):
        def work():
            if not self._busy or company_id != self._scope.company_id or not self._rendered:
                raise EvidenceContractError("scalar link scope/lifetime drift")
            current = self.authorize(self._request)
            self.session.admission(request=self._request,
                retrieval={"citations": citations},
                selections=(AskSelection("citations", self._rendered),),
                current_allowed_sources=current)
            saved = deepcopy(citations)
            result = self._flow._build_rg_links(company_id, citations, file_map_fn=self.file_map)
            if not _same_value(citations, saved):
                raise EvidenceContractError("scalar link builder mutated selected citations")
            if type(result) is not list or not result or any(type(x) is not dict for x in result):
                raise EvidenceContractError("scalar link rendering failed")
            return result
        return self.invoke(work, self._request)

    def precision_rescue(self, **parameters):
        def work():
            if self._busy:
                raise EvidenceContractError("reentrant precision evidence use")
            expected = dict(q=self._request.query, company_id=self._scope.company_id,
                machine_id=self._scope.machine_id, doc_ids=list(self._scope.document_ids) or None,
                bubble_document_id=self._scope.bubble_document_id,
                response_language=self._request.response_language, top_k=self._request.top_k)
            if set(parameters) not in (set(expected), set(expected) | {"answer_contract"}):
                raise EvidenceContractError("unexpected precision rescue arguments")
            actual = {k: parameters[k] for k in expected}
            if not _same_value(actual, expected):
                raise EvidenceContractError("precision rescue differs from original request")
            self._busy = True
            self._selected, self._rendered = None, ()
            try:
                bound = replace(self._flow, _retrieval_precision_facts=self,
                    _PRECISION_FACT_RUNTIME=lambda: self._precision,
                    _sanitize_citations_for_response=self._sanitize,
                    _build_rg_links=self._links)
                # Same eligibility, scalar choice, IT/EN answer and response
                # shape. Only source/resolution/presentation dependencies change.
                result = precision_fact_rescue(runtime=bound, **parameters)
                self.check()  # catches failures hidden by legacy broad catches
                if result is not None:
                    current = self.authorize(self._request)
                    self.session.admission(request=self._request,
                        retrieval={"citations": result["citations"]},
                        selections=(AskSelection("citations", self._rendered),),
                        current_allowed_sources=current)
                    result["meta"]["precision_fact_rescue"]["evidence_lineage_version"] = REQUEST_EVIDENCE_VERSION
                    self._observe_response(result,
                        (parameters.get("answer_contract") or {}).get(precision_facts.SCALAR_TARGET_KEY))
                return result
            finally:
                self._busy = False
        return self.invoke(work, self._request)

    def binding(self) -> RequestFlowEvidenceBinding:
        self.check()
        return RequestFlowEvidenceBinding(self.check, self.precision_rescue, self.close,
                                          (self._observed_core if self._run_core is not None
                                           and self._response_observer is not None else self._run_core))

    def close(self) -> None:
        # Idempotent even after partially failed construction; always release the
        # retained observations and do not mask the original error on cleanup.
        self._active = False
        try:
            if type(self._authority) is RequestAuthority:
                self._authority.close()
        finally:
            self.session.close()
            self._authority = self._readers = self._sources = None
            self._request = self._key = self._selected = self._flow = self._precision = None
            self._rendered = ()
            self._fault = None
            self._run_core = None
            self._response_observer = None

    def __repr__(self):
        return "RequestEvidenceOwner(<request-owned>)"


def bind_request_evidence(**kwargs) -> RequestFlowEvidenceBinding:
    owner = RequestEvidenceOwner(**kwargs)
    try:
        return owner.binding()
    except BaseException:
        owner.close()
        raise


def complete_session_limits(runtime: PrecisionFactRuntime, *, read_caps: tuple[int, ...]) -> AskSessionLimits:
    """Allocate lineage for the existing full-ASK SQL limits, not new retrieval.

    The scalar-only envelope (1,024 occurrences) cannot admit the pre-existing
    structured scan (1,200 by default). Derive the occurrence envelope from the
    configured readers while retaining the lifetime's 8,192-record/64-MiB caps.
    No SQL LIMIT, top_k, prompt/context, HTTP meter or model budget is changed.
    """
    base = precision_session_limits(runtime)
    if (type(read_caps) is not tuple or not read_caps
            or any(type(x) is not int or x < 0 or x > base.max_records for x in read_caps)):
        raise EvidenceContractError("explicit bounded full-request read caps required")
    count = max(base.evidence.assembly.max_occurrences, *read_caps)
    assembly = replace(base.evidence.assembly, max_occurrences=count,
        manifest_limits=replace(base.evidence.assembly.manifest_limits, max_records=count))
    return replace(base, evidence=replace(base.evidence, assembly=assembly))
