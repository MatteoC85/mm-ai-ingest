"""P6-B4a: request-owned composition of B1/B2/B3 receipts into the P6-A port.

No reader, query, ACL lookup, cache, model, linguistic rule or global registry is
implemented here. Trusted composition code supplies CURRENT authorization at
EACH use and explicit occurrence handles, never a join inferred from candidates.
Original SQL observations stay separate from legacy views and their derivations.
This is internal Python application code, NOT a client/JSON/cached credential API.

Production main does not install this session. Main wiring, all synthesis/repair/
rescue exits and both caches still require the remaining B4 activation gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

from .chunk_evidence import ChunkEvidenceRead, ChunkEvidenceLimits, ChunkReadScope
from .page_evidence import PageEvidenceRead
from .relation_evidence import RelationEvidenceRead
from .supplemental_evidence import FileReferenceRead, SupplementalChunkRead, storage_key
from ..evidence.adapter_types import AdapterContext
from ..evidence.ask_input import (
    AskCollectionInput, AskEvidenceAdmission, AskRequestKey,
    apply_ask_evidence_input, ask_request_key, build_ask_evidence_input,
)
from ..evidence.contracts import (
    EvidenceContractError, EvidenceRecord, Provenance, SourceIdentity,
    SourceLocator, canonical_json,
)
from ..evidence.legacy_compatibility import (
    LegacyEvidenceBundle, LegacyRecordInput, build_legacy_bundle,
    restore_legacy_records,
)

COMPOSITION_VERSION = "ask-request-evidence-composition-p6b4a-v1"
_RECEIPT_TYPES = (ChunkEvidenceRead, PageEvidenceRead, RelationEvidenceRead,
                  SupplementalChunkRead, FileReferenceRead)


class AskCompositionError(EvidenceContractError):
    """Technical binding/allocation/lifecycle failure; never a no_sources result."""


@dataclass(frozen=True, slots=True)
class AskSessionLimits:
    """Explicit allocation bounds, not semantic thresholds or API/dollar budgets.

    max_bytes bounds retained serialized provider snapshots + node lineage, and
    that state plus ONE returned admission. It is not a process-heap estimator.
    Limits have no production defaults; activation must choose/review them.
    """
    evidence: ChunkEvidenceLimits
    max_reads: int
    max_records: int
    max_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ChunkEvidenceLimits):
            raise AskCompositionError("explicit P5/provider allocation limits required")
        for name in ("max_reads", "max_records", "max_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise AskCompositionError(name + ": positive integer required")


@dataclass(frozen=True, slots=True)
class ReadHandle:
    _owner: object = field(repr=False)
    index: int


@dataclass(frozen=True, slots=True)
class RecordHandle:
    _owner: object = field(repr=False)
    index: int


@dataclass(frozen=True, slots=True)
class RegisteredRead:
    read: ReadHandle
    records: tuple[RecordHandle, ...]


@dataclass(frozen=True, slots=True)
class AskSelection:
    """Explicit collection occurrence order; duplicates are intentionally legal."""
    name: str
    records: tuple[RecordHandle, ...]
    container_kind: str = "list"

    def __post_init__(self) -> None:
        if self.name not in {"candidates", "citations"} or self.container_kind not in {"list", "tuple"}:
            raise AskCompositionError("unsupported ASK collection or container")
        if type(self.records) is not tuple or any(type(h) is not RecordHandle for h in self.records):
            raise AskCompositionError("explicit immutable occurrence handles required")


@dataclass(frozen=True, slots=True)
class Derivation:
    """An explicit trusted-code view, not a new observation from the machine.

    The named operation and ordered parents document lineage, not a claim that
    the text transformation is semantically correct. That needs consumer tests.
    """
    operation: str
    parent_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecordInspection:
    record: LegacyRecordInput = field(repr=False)
    original_observations: tuple[Any, ...] = field(repr=False)
    derivation: Derivation | None


@dataclass(frozen=True, slots=True)
class _Node:
    bundle: LegacyEvidenceBundle = field(repr=False)
    offset: int
    layout: str
    source: SourceIdentity
    provenance: Provenance
    link_target: Any
    relations: tuple
    # Each reference identifies an ACTUAL read occurrence, not a content hash.
    roots: tuple[tuple[int, int], ...]
    derivation: Derivation | None = None

    def accounting_bytes(self) -> int:
        return len(canonical_json((self.offset, self.layout, self.source,
            self.provenance, self.link_target, self.relations, self.roots,
            self.derivation)).encode("utf-8"))

    def evidence(self) -> EvidenceRecord:
        return self.bundle.assembly.occurrence_entries()[self.offset].evidence


def _receipt_scope(receipt: Any) -> ChunkReadScope:
    return receipt.pages.scope if type(receipt) is RelationEvidenceRead else receipt.scope


def _page_part(receipt: Any) -> Any:
    return receipt.pages if type(receipt) is RelationEvidenceRead else receipt


def _required_sources(receipt: Any) -> frozenset[SourceIdentity]:
    """Dependencies to CHECK against caller allowance, never an inferred grant."""
    part = _page_part(receipt)
    sources = {o.source for o in part.observations}
    sources.update(getattr(receipt, "anchors", ()))
    if type(receipt) is not FileReferenceRead:
        for entry in part.bundle.assembly.occurrence_entries():
            sources.add(entry.evidence.source)
            sources.update(r.target for r in entry.evidence.relations)
    return frozenset(sources)


def _covers_interval(start: int | None, stop: int | None,
                     intervals: tuple[tuple[int | None, int | None], ...]) -> bool:
    """Integer interval union without enumerating page/row numbers."""
    if start is None:
        return True
    end = start if stop is None else stop
    frontier = start
    for a, b in sorted((a, a if b is None else b) for a, b in intervals if a is not None):
        if b < frontier:
            continue
        if a > frontier:
            return False
        frontier = max(frontier, b + 1)
        if frontier > end:
            return True
    return False


def _check_derived_locator(locator: SourceLocator, parents: tuple[EvidenceRecord, ...]) -> None:
    """Do not manufacture page/chunk/sheet/row/field coordinates in a view."""
    locators = tuple(p.locator for p in parents)
    # Keep non-default qualifiers together: never combine the sheet of one
    # occurrence with the row/page/field of another merely because each exists.
    qualifiers = ("page_basis", "page_label", "sheet", "field", "section_path")
    default = SourceLocator()
    eligible = tuple(l for l in locators if all(
        getattr(locator, name) == getattr(default, name) or
        getattr(locator, name) == getattr(l, name) for name in qualifiers))
    if not eligible:
        raise AskCompositionError("derived view invents locator metadata")
    if locator.chunk_index is not None:
        eligible = tuple(l for l in eligible if l.chunk_index == locator.chunk_index)
        if not eligible:
            raise AskCompositionError("derived view claims an unobserved chunk")
    if not _covers_interval(locator.page_from, locator.page_to,
            tuple((l.page_from, l.page_to) for l in eligible)):
        raise AskCompositionError("derived view claims unobserved page coordinates")
    # When row and page are both declared, they must agree on the same parent
    # projections. No text or language parsing is used to fill missing metadata.
    row_parents = tuple(l for l in eligible if l.sheet == locator.sheet and
        (locator.page_from is None or (l.page_from is not None and
         l.page_from <= (locator.page_to or locator.page_from) and
         (l.page_to or l.page_from) >= locator.page_from)))
    if not _covers_interval(locator.row_from, locator.row_to,
            tuple((l.row_from, l.row_to) for l in row_parents)):
        raise AskCompositionError("derived view claims unobserved worksheet rows")



class AskEvidenceSession:
    """One explicit request lifetime; no ambient/context-global authorization.

    Handles are private object-identity capabilities, not serializable IDs and
    not protection against malicious Python code in the same process. A fresh
    request object, even with identical text, cannot reuse this session. Every
    data operation validates the original request key AND the caller's current
    allowance. Neither receipt.read_sources nor a manifest grants authorization.
    """

    def __init__(self, *, request: Any, scope: ChunkReadScope,
                 limits: AskSessionLimits) -> None:
        key = ask_request_key(request)
        if not isinstance(scope, ChunkReadScope) or not isinstance(limits, AskSessionLimits):
            raise AskCompositionError("typed scope and explicit session limits required")
        if (key.company_id, key.machine_id, key.ai_scope, key.document_ids,
                key.bubble_document_id) != (scope.company_id, scope.machine_id,
                scope.ai_scope, scope.document_ids, scope.bubble_document_id):
            raise AskCompositionError("original ASK request and resolved scope differ")
        self._request = request
        self._key = key
        self._scope = scope
        self._limits = limits
        self._owner = object()
        self._lock = RLock()
        self._closed = False
        self._reads: list[Any] = []
        self._nodes: list[_Node] = []
        self._bytes = 0

    def __enter__(self) -> AskEvidenceSession:
        with self._lock:
            self._check_request(self._request)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent; release retained observations, views and request reference."""
        with self._lock:
            self._closed = True
            self._reads.clear()
            self._nodes.clear()
            self._bytes = 0
            self._request = None
            self._key = None

    def summary(self) -> dict[str, Any]:
        """Counts only: no query, tenant/source IDs, text, URLs or permissions."""
        with self._lock:
            return {"version": COMPOSITION_VERSION, "closed": self._closed,
                    "registered_reads": len(self._reads), "record_views": len(self._nodes),
                    "retained_serialized_bytes": self._bytes,
                    "scope": "request_local_composition"}

    def _check_request(self, request: Any) -> None:
        if self._closed:
            raise AskCompositionError("ASK evidence session is closed")
        if request is not self._request or ask_request_key(request) != self._key:
            raise AskCompositionError("original request identity or selectors changed")

    def _check_allowance(self, current: frozenset[SourceIdentity]) -> None:
        if type(current) is not frozenset:
            raise AskCompositionError("explicit CURRENT immutable allowance required")
        for source in current:
            if not isinstance(source, SourceIdentity) or not self._scope.permits(
                    source.scope.company_id, source.scope.machine_id, storage_key(source)):
                raise AskCompositionError("current allowance outside resolved ASK scope")

    def _read(self, handle: ReadHandle, current: frozenset[SourceIdentity]) -> Any:
        if type(handle) is not ReadHandle or handle._owner is not self._owner or type(handle.index) is not int or not 0 <= handle.index < len(self._reads):
            raise AskCompositionError("foreign or invalid read handle")
        receipt = self._reads[handle.index]
        if not _required_sources(receipt).issubset(current):
            raise AskCompositionError("read dependency authorization missing or revoked")
        return receipt

    def _node(self, handle: RecordHandle, current: frozenset[SourceIdentity]) -> _Node:
        if type(handle) is not RecordHandle or handle._owner is not self._owner or type(handle.index) is not int or not 0 <= handle.index < len(self._nodes):
            raise AskCompositionError("foreign or invalid evidence occurrence handle")
        node = self._nodes[handle.index]
        if node.source not in current or any(r.target not in current for r in node.relations):
            raise AskCompositionError("record or relation authorization missing or revoked")
        # Selected-record access does not expose unrelated scanned rows.
        # Their revocation must not veto this still-authorized occurrence.
        # Full read inspection separately requires every snapshot dependency.
        return node

    def _inputs(self, handles: tuple[RecordHandle, ...], current: frozenset[SourceIdentity]) -> tuple[LegacyRecordInput, ...]:
        if type(handles) is not tuple or len(handles) > self._limits.evidence.assembly.max_occurrences:
            raise AskCompositionError("bounded immutable occurrence selection required")
        result = []
        for handle in handles:
            node = self._node(handle, current)
            # Each node owns a single-record P5 snapshot. Restore reauthorizes
            # only the selected occurrence (and relations), not unrelated rows.
            # Repeated handles produce independent nested mappings, not aliases.
            view = restore_legacy_records(node.bundle, company_id=self._scope.company_id,
                allowed_sources=current, adapter_limits=self._limits.evidence.adapter)[node.offset]
            result.append(LegacyRecordInput(node.layout, view.record,
                AdapterContext(node.source, node.provenance, current,
                    link_target=node.link_target, relations=node.relations), view.indexed_text))
        return tuple(result)

    def register(self, *, request: Any, receipt: Any,
                 current_allowed_sources: frozenset[SourceIdentity]) -> RegisteredRead:
        """Atomically retain an authentic internal typed receipt; no source lookup.

        Scanned-but-unselected observations and unresolved relations are retained
        but are NOT turned into selectable evidence handles. File metadata has
        no content handles at all. Empty reads are valid, still bounded reads.
        """
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            if type(receipt) not in _RECEIPT_TYPES:
                raise AskCompositionError("B1/B2/B3 typed provider receipt required")
            if _receipt_scope(receipt) != self._scope:
                raise AskCompositionError("provider receipt scope differs from ASK scope")
            receipt.__post_init__()
            required = _required_sources(receipt)
            if not required.issubset(current_allowed_sources):
                raise AskCompositionError("current provider authorization missing")
            if len(self._reads) >= self._limits.max_reads:
                raise AskCompositionError("aggregate read count exceeded")
            part = _page_part(receipt)
            if type(receipt) is FileReferenceRead:
                receipt.as_file_map(scope=self._scope, current_allowed_sources=current_allowed_sources)
                inputs, indices = (), ()
            else:
                inputs = receipt.as_legacy_inputs(scope=self._scope,
                    current_allowed_sources=current_allowed_sources,
                    adapter_limits=self._limits.evidence.adapter)
                indices = (tuple(range(len(part.observations))) if type(receipt) is ChunkEvidenceRead
                           else part.selected_observation_indices)
            read_index = len(self._reads)
            bounds = self._limits.evidence
            nodes = []
            for pos, value in enumerate(inputs):
                single = build_legacy_bundle((value,), company_id=self._scope.company_id,
                    allowed_sources=current_allowed_sources, adapter_limits=bounds.adapter,
                    assembly_limits=bounds.assembly, legacy_limits=bounds.legacy)
                nodes.append(_Node(single, 0, value.layout, value.context.source,
                    value.context.provenance, value.context.link_target, value.context.relations,
                    ((read_index, indices[pos]),)))
            extra_bytes = receipt.size_bytes + sum(n.bundle.size_bytes + n.accounting_bytes() for n in nodes)
            self._check_capacity(len(nodes), extra_bytes)
            # No registry mutation occurs until all authorization/data/budget checks pass.
            start = len(self._nodes)
            self._reads.append(receipt)
            self._nodes.extend(nodes)
            self._bytes += extra_bytes
            return RegisteredRead(ReadHandle(self._owner, read_index),
                tuple(RecordHandle(self._owner, start + i) for i in range(len(nodes))))

    def _check_capacity(self, records: int, size: int) -> None:
        if len(self._nodes) + records > self._limits.max_records:
            raise AskCompositionError("aggregate record count exceeded")
        if self._bytes + size > self._limits.max_bytes:
            raise AskCompositionError("aggregate retained/admission byte budget exceeded")

    def records(self, *, request: Any, handles: tuple[RecordHandle, ...],
                current_allowed_sources: frozenset[SourceIdentity]) -> tuple[LegacyRecordInput, ...]:
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            return self._inputs(handles, current_allowed_sources)

    def inspect_record(self, *, request: Any, handle: RecordHandle,
                       current_allowed_sources: frozenset[SourceIdentity]) -> RecordInspection:
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            node = self._node(handle, current_allowed_sources)
            originals = tuple(_page_part(self._reads[r]).observations[i] for r, i in node.roots)
            return RecordInspection(self._inputs((handle,), current_allowed_sources)[0], originals,
                                    node.derivation)

    def inspect_read(self, *, request: Any, handle: ReadHandle,
                     current_allowed_sources: frozenset[SourceIdentity]) -> Any:
        """Authorized INTERNAL immutable snapshot, not a public response serializer."""
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            return self._read(handle, current_allowed_sources)

    def file_map(self, *, request: Any, handle: ReadHandle,
                 current_allowed_sources: frozenset[SourceIdentity]) -> dict[str, str]:
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            receipt = self._read(handle, current_allowed_sources)
            if type(receipt) is not FileReferenceRead:
                raise AskCompositionError("explicit file-reference read required")
            return receipt.as_file_map(scope=self._scope,
                current_allowed_sources=current_allowed_sources)

    def derive(self, *, request: Any, parents: tuple[RecordHandle, ...],
               record: dict[str, Any], operation: str,
               current_allowed_sources: frozenset[SourceIdentity],
               layout: str = "retrieval_candidate") -> RecordHandle:
        """Register an EXPLICIT view computed by existing trusted Python consumers.

        No matching by citation ID/text/score. Parents supply identity, original
        observations and relationships, never the proposed record. Cross-source
        composites must remain multiple records, not one fabricated document.
        P5 checks metadata consistency; a view cannot claim new coordinates.
        Changed text is labeled as a derivation and does NOT replace SQL text.
        Semantic equivalence of the named legacy transformation is not proven
        here; activating each consumer requires separate before/after tests.
        """
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            if type(parents) is not tuple or not parents or len(parents) > self._limits.evidence.assembly.max_occurrences:
                raise AskCompositionError("bounded explicit derivation parents required")
            if type(record) is not dict or layout not in {"document_page", "retrieval_candidate"}:
                raise AskCompositionError("supported explicit legacy view required")
            if type(operation) is not str or not operation.strip() or len(operation) > self._limits.evidence.adapter.max_aux_chars:
                raise AskCompositionError("bounded named trusted transformation required")
            nodes = tuple(self._node(h, current_allowed_sources) for h in parents)
            source = nodes[0].source
            if any(n.source != source for n in nodes):
                raise AskCompositionError("one derived record cannot merge different sources")
            links = {n.link_target for n in nodes if n.link_target is not None}
            if len(links) > 1:
                raise AskCompositionError("conflicting parent application references")
            relations = tuple(dict.fromkeys(r for n in nodes for r in n.relations))
            roots = tuple(dict.fromkeys(root for n in nodes for root in n.roots))
            derivation = Derivation(operation, tuple(h.index for h in parents))
            # Reference immutable parent evidence IDs, not mutable dictionary metadata.
            provenance = Provenance(COMPOSITION_VERSION + ":view",
                canonical_json((derivation, tuple(n.evidence().evidence_id for n in nodes))))
            context = AdapterContext(source, provenance, current_allowed_sources,
                link_target=next(iter(links), None), relations=relations)
            bounds = self._limits.evidence
            bundle = build_legacy_bundle((LegacyRecordInput(layout, record, context),),
                company_id=self._scope.company_id, allowed_sources=current_allowed_sources,
                adapter_limits=bounds.adapter, assembly_limits=bounds.assembly,
                legacy_limits=bounds.legacy)
            evidence = bundle.assembly.occurrence_entries()[0].evidence
            _check_derived_locator(evidence.locator, tuple(n.evidence() for n in nodes))
            node = _Node(bundle, 0, layout, source, provenance, context.link_target,
                         relations, roots, derivation)
            self._check_capacity(1, bundle.size_bytes + node.accounting_bytes())
            handle = RecordHandle(self._owner, len(self._nodes))
            self._nodes.append(node)
            self._bytes += bundle.size_bytes + node.accounting_bytes()
            return handle

    def admission(self, *, request: Any, retrieval: dict[str, Any],
                  selections: tuple[AskSelection, ...],
                  current_allowed_sources: frozenset[SourceIdentity]) -> AskEvidenceAdmission:
        """Bind exactly the current collection order/content to the original request.

        The P6-A port itself compares the frozen round-trip with retrieval here;
        the real Core checks it again immediately before initial ASK synthesis.
        No unselected scanned row or opaque file URL can become an ASK candidate.
        """
        with self._lock:
            self._check_request(request)
            self._check_allowance(current_allowed_sources)
            if type(selections) is not tuple or len(selections) > 2 or any(type(s) is not AskSelection for s in selections):
                raise AskCompositionError("explicit bounded ASK collection selections required")
            if sum(len(s.records) for s in selections) > self._limits.evidence.assembly.max_occurrences:
                raise AskCompositionError("aggregate ASK selection count exceeded")
            collections = tuple(AskCollectionInput(s.name,
                self._inputs(s.records, current_allowed_sources), s.container_kind) for s in selections)
            bounds = self._limits.evidence
            envelope = build_ask_evidence_input(request_key=self._key, collections=collections,
                allowed_sources=current_allowed_sources, adapter_limits=bounds.adapter,
                assembly_limits=bounds.assembly, legacy_limits=bounds.legacy)
            result = AskEvidenceAdmission(envelope, current_allowed_sources, bounds.adapter)
            self._check_capacity(0, envelope.bundle.size_bytes)
            apply_ask_evidence_input(retrieval, request_key=self._key, admission=result)
            return result

    def admission_hook(self, *,
                       authorize: Callable[[Any], frozenset[SourceIdentity]],
                       select: Callable[[Any, dict, Any], tuple[AskSelection, ...]]) -> Callable:
        """P6-A hook adapter; policy and selection are explicit composition inputs.

        authorize runs afresh on each invocation. This hook is NOT sufficient
        for production B4 activation: repair/direct/cache paths must be wired and
        proved separately. It does not intercept other modes or replace the Core.
        """
        if not callable(authorize) or not callable(select):
            raise AskCompositionError("explicit current-authority and selection callbacks required")
        def prepare(request: Any, retrieval: dict, decision: Any) -> AskEvidenceAdmission:
            with self._lock:
                self._check_request(request)
            if getattr(decision, "effective_mode", None) != "ask":
                raise AskCompositionError("ASK/ASK admission only")
            current = authorize(request)
            selections = select(request, retrieval, decision)
            return self.admission(request=request, retrieval=retrieval, selections=selections,
                                  current_allowed_sources=current)
        return prepare
