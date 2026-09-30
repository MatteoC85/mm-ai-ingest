"""B4l concrete adapters for the 27 EXISTING typed repository readers.

Readers execute their real implementations/SQL. The caller supplies the already
owned B4a session, current application authority and the SAME B4e invocation
guard. No permission is obtained from receipt.read_sources, candidates or IDs.

This module does not convert document_page into retrieval_candidate (B4m), does
not fabricate lineage for hidden callbacks, and does not activate main's Core.
Arguments/anchor handles belong to trusted application composition, not JSON.
"""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

from . import dense, lexical, document_readers, structured, context_expansion, evidence_assurance, precision_facts
from .ask_composition import AskEvidenceSession, RecordHandle, RegisteredRead
from .receipt_producers import acquire_read
from ..evidence.contracts import EvidenceContractError, canonical_json

PRODUCTION_ADAPTER_VERSION = "production-readers-p6b4l-v1"


@dataclass(frozen=True, slots=True)
class _Spec:
    reader: Callable[..., Any]
    runtime_type: type
    current: bool
    request: bool


_READERS = {
    "read_dense_chunk_evidence": _Spec(dense.read_dense_chunk_evidence, dense.DenseRuntime, False, False),
    "read_prefix_chunk_evidence": _Spec(lexical.read_prefix_chunk_evidence, lexical.LexicalRuntime, False, False),
    "read_fts_chunk_evidence": _Spec(lexical.read_fts_chunk_evidence, lexical.LexicalRuntime, False, False),
    "read_parent_procedure_page_evidence": _Spec(document_readers.read_parent_procedure_page_evidence, document_readers.DbFetchParentProcedurePagesForStepsRuntime, True, False),
    "read_ask_page_evidence": _Spec(document_readers.read_ask_page_evidence, document_readers.AskEvidenceFetchPagesRuntime, False, False),
    "read_full_context_page_evidence": _Spec(document_readers.read_full_context_page_evidence, document_readers.AskFullContextFetchPagesRuntime, False, False),
    "read_scored_page_evidence": _Spec(document_readers.read_scored_page_evidence, document_readers.V13FetchScoredPagesRuntime, False, False),
    "read_token_chunk_evidence": _Spec(document_readers.read_token_chunk_evidence, document_readers.DbFindTokenChunkRuntime, False, False),
    "read_entity_chunk_evidence": _Spec(document_readers.read_entity_chunk_evidence, document_readers.DbFindEntityChunkRuntime, False, False),
    "read_preferred_page_evidence": _Spec(document_readers.read_preferred_page_evidence, document_readers.AskFetchPreferredSourcePagesRuntime, False, False),
    "read_v13_preferred_page_evidence": _Spec(document_readers.read_v13_preferred_page_evidence, document_readers.V13FetchPreferredSourcePagesRuntime, False, False),
    "read_maintenance_page_evidence": _Spec(document_readers.read_maintenance_page_evidence, document_readers.AskFetchManualMaintenanceTargetPagesRuntime, False, False),
    "read_machine_catalog_page_evidence": _Spec(document_readers.read_machine_catalog_page_evidence, document_readers.AssistantCoreMachineCatalogCandidatesRuntime, False, True),
    "read_semantic_manual_support_page_evidence": _Spec(document_readers.read_semantic_manual_support_page_evidence, document_readers.AskStructuredDirectFetchManualSupportRuntime, True, False),
    "read_deterministic_manual_support_page_evidence": _Spec(document_readers.read_deterministic_manual_support_page_evidence, document_readers.V13FetchManualSupportDeterministicRuntime, True, False),
    "read_document_file_references": _Spec(document_readers.read_document_file_references, document_readers.FetchDocumentFileMapRuntime, True, False),
    "read_related_step_page_evidence": _Spec(structured.read_related_step_page_evidence, structured.DbFetchRelatedStepPagesRuntime, True, False),
    "read_structured_direct_page_evidence": _Spec(structured.read_structured_direct_page_evidence, structured.AskStructuredDirectFetchSourcesRuntime, False, False),
    "read_structured_title_page_evidence": _Spec(structured.read_structured_title_page_evidence, structured.V13FetchStructuredTitleCandidatesRuntime, False, False),
    "read_structured_rescue_chunk_evidence": _Spec(structured.read_structured_rescue_chunk_evidence, structured.FetchStructuredRescueCandidatesRuntime, False, False),
    "read_structured_dense_chunk_evidence": _Spec(structured.read_structured_dense_chunk_evidence, structured.V13FetchStructuredDenseCandidatesRuntime, False, False),
    "read_step_fallback_page_evidence": _Spec(structured.read_step_fallback_page_evidence, structured.V12ExpandPrimaryProcedureStepsRuntime, False, False),
    "read_neighbor_chunk_evidence": _Spec(context_expansion.read_neighbor_chunk_evidence, context_expansion.ExpandWithNeighborChunksRuntime, True, False),
    "read_enumeration_page_evidence": _Spec(context_expansion.read_enumeration_page_evidence, context_expansion.AssistantCoreExpandEnumerationSectionsRuntime, True, True),
    "read_assurance_neighbor_page_evidence": _Spec(evidence_assurance.read_assurance_neighbor_page_evidence, evidence_assurance.V13AssuranceFetchNeighborPagesRuntime, True, False),
    "read_assurance_parent_page_evidence": _Spec(evidence_assurance.read_assurance_parent_page_evidence, evidence_assurance.V13AssuranceExpandStructuredRelationsRuntime, False, False),
    "read_precision_page_evidence": _Spec(precision_facts.read_precision_page_evidence, precision_facts.PrecisionFactRuntime, False, False),
}
READER_NAMES = tuple(sorted(_READERS))
_RESERVED = frozenset({"scope", "limits", "runtime", "current_allowed_sources", "request",
                        "seed_inputs", "candidate_inputs", "structured_inputs"})
_HANDLE_INPUTS = {"seed_handles": "seed_inputs", "candidate_handles": "candidate_inputs",
                  "structured_handles": "structured_inputs"}

# Exact-query reuse is limited to immutable DB search snapshots. In particular
# file URLs, relation expansion and readers consuming occurrence handles are
# never memoized here. Authorization must be checked on every hit.
_MEMO_READERS = frozenset({"read_dense_chunk_evidence", "read_fts_chunk_evidence",
    "read_token_chunk_evidence", "read_structured_dense_chunk_evidence"})


def _memo_value(value, depth=0):
    """Type/order-sensitive bounded key; unsupported inputs disable reuse."""
    if depth > 16:
        raise ValueError("memo key nesting")
    typ = type(value)
    if value is None or typ in {str, int, float, bool}:
        return (typ.__name__, value)
    if typ in {list, tuple}:
        return (typ.__name__, tuple(_memo_value(x, depth + 1) for x in value))
    if typ is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, _memo_value(v, depth + 1)) for k, v in value.items()))
    raise ValueError("unsupported memo key")


class ProductionReaderAdapters:
    """No new owner/session and no mutable shared runtime patching."""
    def __init__(self, *, request: Any, session: AskEvidenceSession,
                 authorize: Callable, invoke: Callable, runtimes: dict[str, Any],
                 memo_epoch: Callable[[], int | None] | None = None):
        if (type(session) is not AskEvidenceSession or not callable(authorize)
                or not callable(invoke) or type(runtimes) is not dict
                or set(runtimes) != set(READER_NAMES)):
            raise EvidenceContractError("complete explicit production reader dependencies required")
        for name, spec in _READERS.items():
            if type(runtimes[name]) is not spec.runtime_type:
                raise EvidenceContractError("wrong production reader runtime: " + name)
        self._request, self._session = request, session
        self._authorize, self._invoke, self._runtimes = authorize, invoke, dict(runtimes)
        if memo_epoch is not None and not callable(memo_epoch):
            raise EvidenceContractError("explicit request snapshot epoch required")
        self._memo_epoch = memo_epoch
        self._memo = {}
        self._epoch = None
        self._memo_hits = self._memo_misses = 0

    def memo_summary(self):
        return {"hits": self._memo_hits, "misses": self._memo_misses,
                "retained_reads": len(self._memo)}

    def _memo_key(self, name, parameters):
        if self._memo_epoch is None or name not in _MEMO_READERS:
            return None
        try:
            key = canonical_json(_memo_value(parameters))
        except (ValueError, TypeError, OverflowError):
            return None
        if len(key) > 262144:
            return None
        current = self._invoke(self._authorize, self._request, self._request)
        self._session.read_contract(request=self._request, current_allowed_sources=current)
        epoch = self._invoke(self._memo_epoch, self._request)
        if epoch is None:
            return None
        if type(epoch) is not int or epoch < 0:
            raise EvidenceContractError("invalid request snapshot epoch")
        if epoch != self._epoch:
            self._memo.clear()
            self._epoch = epoch
        return (name, epoch, key), current

    def read(self, name: str, **parameters) -> RegisteredRead:
        # Fixed names and integer counters only; no query, source or tenant data.
        # An inactive ASK trace performs no output, network or clock reads here.
        from ..ask.phase_trace import count, span

        def work():
            if type(name) is not str or name not in _READERS or _RESERVED.intersection(parameters):
                raise EvidenceContractError("unknown reader or caller-supplied authority input")
            spec = _READERS[name]
            supplied = dict(parameters)
            handles = {k: supplied.pop(k) for k in _HANDLE_INPUTS if k in supplied}
            for value in handles.values():
                if type(value) is not tuple or any(type(h) is not RecordHandle for h in value):
                    raise EvidenceContractError("explicit same-session occurrence handles required")

            memo = self._memo_key(name, parameters) if not handles else None
            if memo is not None:
                memo_key, current = memo
                cached = self._memo.get(memo_key)
                if cached is not None:
                    # Revalidate ALL original read dependencies, not just the
                    # selected candidates. A revoked read never becomes a hit.
                    self._session.inspect_read(request=self._request, handle=cached.read,
                        current_allowed_sources=current)
                    self._session.validate_records(request=self._request, handles=cached.records,
                        current_allowed_sources=current)
                    self._memo_hits += 1
                    count("retrieval.memo_hits")
                    return cached
                self._memo_misses += 1
                count("retrieval.memo_misses")

            def reader(*, scope, limits, current_allowed_sources):
                kwargs = dict(supplied, scope=scope, limits=limits, runtime=self._runtimes[name])
                if spec.current:
                    kwargs["current_allowed_sources"] = current_allowed_sources
                if spec.request:
                    kwargs["request"] = self._request
                for alias, refs in handles.items():
                    kwargs[_HANDLE_INPUTS[alias]] = self._session.records(request=self._request,
                        handles=refs, current_allowed_sources=current_allowed_sources)
                # Reject accidental scope/argument drift before a database read.
                try:
                    signature(spec.reader).bind(**kwargs)
                except TypeError:
                    raise EvidenceContractError("reader adapter arguments do not match its contract") from None
                return spec.reader(**kwargs)

            result = acquire_read(request=self._request, session=self._session, reader=reader,
                authorize=self._authorize, invoke=self._invoke)
            if (memo is not None and len(self._memo) < 128
                    and self._invoke(self._memo_epoch, self._request) == memo_key[1]):
                self._memo[memo_key] = result
            return result

        def measured_work():
            with span("retrieval.read"):
                count("retrieval.read_calls")
                result = work()
                count("retrieval.records", len(result.records))
                return result

        return self._invoke(measured_work, self._request)

    def candidate_handles(self, name: str, **parameters) -> tuple[RecordHandle, ...]:
        """For actual candidate-layout reads only; raw page conversion stays explicit."""
        def work():
            registered = self.read(name, **parameters)
            current = self._invoke(self._authorize, self._request, self._request)
            records = self._session.records(request=self._request, handles=registered.records,
                                             current_allowed_sources=current)
            if any(r.layout != "retrieval_candidate" for r in records):
                raise EvidenceContractError("raw page/file output requires its explicit consumer adapter")
            # A file receipt has no records; it must not impersonate an empty
            # candidate read. Inspect the trusted receipt type/layout as well.
            receipt = self._session.inspect_read(request=self._request, handle=registered.read,
                                                 current_allowed_sources=current)
            if getattr(receipt, "layout", None) == "document_page" or name == "read_document_file_references":
                raise EvidenceContractError("non-candidate reader cannot supply candidate handles")
            return registered.records
        return self._invoke(work, self._request)

    def dense(self, **parameters) -> tuple[int | None, tuple[RecordHandle, ...]]:
        """Exact B4j dense adapter contract: debug count and registered occurrences."""
        def work():
            registered = self.read("read_dense_chunk_evidence", **parameters)
            current = self._invoke(self._authorize, self._request, self._request)
            receipt = self._session.inspect_read(request=self._request, handle=registered.read,
                                                 current_allowed_sources=current)
            return receipt.chunks_matching_filter, registered.records
        return self._invoke(work, self._request)
