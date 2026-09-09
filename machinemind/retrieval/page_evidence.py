"""P6-B2: request-local evidence for existing scoped page reads.

No SQL, I/O, model, parsing of human language or public authorization endpoint.
Scope/limits reuse P6-B1; canonical data/compatibility reuse P5. Only repository
readers create the capture. Columns from their SAME SELECT establish the source
binding before any ranking/deduplication. Candidate fields never grant access.

All SQL text projections are retained literally, separately from the historical
trimmed/ranked candidates. No sheet, row, PDF page basis, file format, revision,
URL, title, field or relationship is invented when absent from the SQL record.
A structured media page is indexed metadata, never a visual analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunk_evidence import (ChunkReadScope, ChunkEvidenceLimits,
                             _source_from_columns, validate_read)
from ..evidence.adapter_types import AdapterContext, AdapterLimits
from ..evidence.contracts import (EvidenceContractError, EvidenceRelation,
    Provenance, SourceIdentity, SourceLocator, canonical_json)
from ..evidence.legacy_compatibility import (LegacyEvidenceBundle, LegacyRecordInput,
    build_legacy_bundle, restore_legacy_records)

PROVIDER_VERSION = "scoped-page-reader-p6b2-v1"
PAGE_BINDING_COLUMNS = ", company_id AS evidence_company_id, machine_id AS evidence_machine_id"
PAGE_READ_KINDS = frozenset({"ask_pages", "scored_pages", "full_context",
                            "structured_direct", "structured_title"})


class PageReadBindingError(EvidenceContractError):
    """Technical read/binding failure; never converted into no_sources."""


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise PageReadBindingError(name + ": invalid integer")


def _source(scope: ChunkReadScope, company: Any, machine: Any, key: Any,
            limits: ChunkEvidenceLimits) -> SourceIdentity:
    for value in (company, machine, key):
        if type(value) is str and len(value) > limits.adapter.max_aux_chars:
            raise PageReadBindingError("source identity exceeds allocation budget")
    result = _source_from_columns(company, machine, key)
    if not scope.permits(company, machine, key):
        raise PageReadBindingError("SQL page outside resolved request scope")
    return result


@dataclass(frozen=True, slots=True)
class PageReadObservation:
    source: SourceIdentity
    provenance: Provenance
    stored_company_id: str
    stored_machine_id: str | None
    storage_document_id: str
    page_number: int
    text_projection: str = field(repr=False)
    projection_chars: int
    batch: int
    row: int
    relations: tuple[EvidenceRelation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.provenance, Provenance):
            raise PageReadBindingError("typed page source and provenance required")
        expected = _source_from_columns(self.stored_company_id, self.stored_machine_id,
                                        self.storage_document_id)
        if expected != self.source:
            raise PageReadBindingError("page source differs from SQL association")
        _integer(self.page_number, "page_number", 1)
        _integer(self.projection_chars, "projection_chars")
        _integer(self.batch, "batch")
        _integer(self.row, "row")
        if type(self.text_projection) is not str or len(self.text_projection) > self.projection_chars:
            raise PageReadBindingError("SQL page text exceeds its declared projection")
        if type(self.relations) is not tuple or any(not isinstance(r, EvidenceRelation) for r in self.relations):
            raise PageReadBindingError("immutable typed page relationships required")

    @property
    def locator(self) -> SourceLocator:
        # A page_number alone does not establish PDF physical page numbering.
        return SourceLocator(page_from=self.page_number, page_to=self.page_number)


def observe_page(*, scope: ChunkReadScope, company: Any, machine: Any, key: Any,
                 page: Any, text: Any, projection_chars: int, kind: str,
                 batch: int, row: int, limits: ChunkEvidenceLimits,
                 relations: tuple[EvidenceRelation, ...] = ()) -> PageReadObservation:
    source = _source(scope, company, machine, key, limits)
    if type(text) is not str or len(text) > limits.adapter.max_text_chars:
        raise PageReadBindingError("indexed SQL text must fit the explicit allocation budget")
    provenance = Provenance(PROVIDER_VERSION + ":" + kind,
        canonical_json(("public.document_pages", company, machine, key, page,
                        projection_chars, batch, row)))
    return PageReadObservation(source, provenance, company, machine, key, page, text,
                               projection_chars, batch, row, relations)


@dataclass(frozen=True, slots=True)
class PageEvidenceRead:
    """Fresh read, not a transferable credential or permission cache.

    read_sources is acquisition evidence only. Consumers explicitly supply their
    current allowance and identical resolved scope at reuse. Raw observations
    include scanned-but-unselected rows; the bundle has only selected records.
    """
    scope: ChunkReadScope
    kind: str
    observations: tuple[PageReadObservation, ...] = field(repr=False)
    selected_observation_indices: tuple[int, ...]
    bundle: LegacyEvidenceBundle = field(repr=False)
    size_bytes: int
    query_count: int
    layout: str = "retrieval_candidate"

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ChunkReadScope) or not isinstance(self.bundle, LegacyEvidenceBundle):
            raise PageReadBindingError("typed scoped page read required")
        if self.kind not in PAGE_READ_KINDS | {"related_steps", "parent_procedures"}:
            raise PageReadBindingError("unknown repository page read")
        if self.layout not in {"retrieval_candidate", "document_page"}:
            raise PageReadBindingError("explicit page layout required")
        if type(self.observations) is not tuple or any(not isinstance(o, PageReadObservation) for o in self.observations):
            raise PageReadBindingError("immutable observations required")
        if type(self.selected_observation_indices) is not tuple:
            raise PageReadBindingError("immutable occurrence selection required")
        entries = self.bundle.assembly.occurrence_entries()
        if len(entries) != len(self.selected_observation_indices) or self.bundle.assembly.manifest.company_id != self.scope.company_id:
            raise PageReadBindingError("page selection/company mismatch")
        for o in self.observations:
            if not self.scope.permits(o.stored_company_id, o.stored_machine_id, o.storage_document_id):
                raise PageReadBindingError("observed page outside scope")
        for idx, entry in zip(self.selected_observation_indices, entries):
            _integer(idx, "selected observation")
            if idx >= len(self.observations):
                raise PageReadBindingError("selected observation missing")
            o = self.observations[idx]
            if entry.evidence.source != o.source or entry.evidence.provenance != o.provenance or entry.evidence.relations != o.relations:
                raise PageReadBindingError("selected source/provenance/relationships altered")
        _integer(self.query_count, "query_count")
        size = self.bundle.size_bytes + sum(len(canonical_json(o).encode("utf-8")) for o in self.observations)
        if type(self.size_bytes) is not int or self.size_bytes != size or size > self.bundle.limits.max_bytes:
            raise PageReadBindingError("aggregate page snapshot budget exceeded")

    @property
    def read_sources(self) -> frozenset[SourceIdentity]:
        return frozenset(o.source for o in self.observations)

    def as_legacy_inputs(self, *, scope: ChunkReadScope,
                         current_allowed_sources: frozenset[SourceIdentity],
                         adapter_limits: AdapterLimits) -> tuple[LegacyRecordInput, ...]:
        if not isinstance(scope, ChunkReadScope) or scope != self.scope:
            raise PageReadBindingError("request scope changed since page acquisition")
        restored = restore_legacy_records(self.bundle, company_id=scope.company_id,
            allowed_sources=current_allowed_sources, adapter_limits=adapter_limits)
        return tuple(LegacyRecordInput(self.layout, item.record,
            AdapterContext(o.source, o.provenance, current_allowed_sources, relations=o.relations))
            for item, o in zip(restored, (self.observations[i] for i in self.selected_observation_indices)))


def assemble_page_read(*, scope: ChunkReadScope, kind: str,
                       observations: tuple[PageReadObservation, ...], indices: tuple[int, ...],
                       records: list[dict], limits: ChunkEvidenceLimits, query_count: int,
                       layout: str = "retrieval_candidate",
                       relation_allowance: frozenset[SourceIdentity] = frozenset()) -> PageEvidenceRead:
    """Internal builder. Authorize from observed columns, never candidate dicts."""
    validate_read(scope, len(observations), limits)
    if len(records) != len(indices):
        raise PageReadBindingError("page record/occurrence count mismatch")
    allowed = frozenset(o.source for o in observations) | relation_allowance
    inputs = [LegacyRecordInput(layout, record, AdapterContext(observations[idx].source,
        observations[idx].provenance, allowed, relations=observations[idx].relations))
        for idx, record in zip(indices, records)]
    bundle = build_legacy_bundle(inputs, company_id=scope.company_id, allowed_sources=allowed,
        adapter_limits=limits.adapter, assembly_limits=limits.assembly, legacy_limits=limits.legacy)
    size = bundle.size_bytes + sum(len(canonical_json(o).encode("utf-8")) for o in observations)
    return PageEvidenceRead(scope, kind, observations, indices, bundle, size, query_count, layout)


class _PageReadCapture:
    """Single-call repository collector, never a global or request payload field."""
    def __init__(self, scope: ChunkReadScope, limits: ChunkEvidenceLimits, kind: str,
                 *, snippet_chars: int, candidate_chars: int | None = None) -> None:
        validate_read(scope, 0, limits)
        if kind not in PAGE_READ_KINDS:
            raise PageReadBindingError("unsupported page capture")
        _integer(snippet_chars, "snippet_chars")
        if candidate_chars is not None:
            _integer(candidate_chars, "candidate_chars")
        self.scope, self.limits, self.kind = scope, limits, kind
        self.snippet_chars, self.candidate_chars = snippet_chars, candidate_chars
        self._observations: list[PageReadObservation] = []
        self._pending: tuple[int, int] | None = None
        self._queries = 0
        self._bytes = 0
        self._finished = False

    def expect(self, row_limit: int, text_chars: int) -> None:
        if self._finished or self._pending is not None:
            raise PageReadBindingError("capture lifecycle violation")
        validate_read(self.scope, row_limit, self.limits)
        _integer(text_chars, "text_chars")
        if text_chars > self.limits.adapter.max_text_chars:
            raise PageReadBindingError("SQL projection exceeds explicit allocation limit")
        max_queries = 1 if self.kind in {"full_context", "structured_title"} else 2
        if self._queries >= max_queries:
            raise PageReadBindingError("unexpected additional page query")
        self._pending = (row_limit, text_chars)

    def capture(self, rows: list[tuple]) -> list[tuple]:
        if self._finished or self._pending is None:
            raise PageReadBindingError("page capture without bounded read")
        row_limit, chars = self._pending
        if type(rows) is not list or len(rows) > row_limit:
            raise PageReadBindingError("SQL page row limit/shape violated")
        if len(self._observations) + len(rows) > self.limits.assembly.max_occurrences:
            raise PageReadBindingError("aggregate SQL observation limit exceeded")
        out = []
        for idx, raw in enumerate(rows):
            if type(raw) is not tuple or len(raw) != 6:
                raise PageReadBindingError("expected four legacy columns plus two binding columns")
            key, mid, page, text, company, stored_mid = raw
            if type(mid) is not type(stored_mid) or mid != stored_mid:
                raise PageReadBindingError("projected machine columns disagree")
            o = observe_page(scope=self.scope, company=company, machine=stored_mid,
                key=key, page=page, text=text, projection_chars=chars, kind=self.kind,
                batch=self._queries, row=idx, limits=self.limits)
            self._bytes += len(canonical_json(o).encode("utf-8"))
            if self._bytes > self.limits.legacy.max_bytes:
                raise PageReadBindingError("SQL page observation byte budget exceeded")
            self._observations.append(o)
            out.append(raw[:4])
        self._queries += 1
        self._pending = None
        return out

    def finish(self, candidates: list[dict]) -> PageEvidenceRead:
        if self._finished or self._pending is not None:
            raise PageReadBindingError("unfinished or reused page capture")
        self._finished = True
        if type(candidates) is not list:
            raise PageReadBindingError("candidate list required")
        indices = []
        used: set[int] = set()
        for candidate in candidates:
            if type(candidate) is not dict:
                raise PageReadBindingError("candidate mapping required")
            matches = []
            for idx, o in enumerate(self._observations):
                text = o.text_projection.strip()
                if self.candidate_chars is not None:
                    text = text[:self.candidate_chars]
                expected = {"bubble_document_id": o.storage_document_id,
                    "page_from": o.page_number, "page_to": o.page_number,
                    "chunk_full": text, "snippet": text[:self.snippet_chars]}
                if all(type(candidate.get(k)) is type(v) and candidate[k] == v for k, v in expected.items()):
                    matches.append(idx)
            if not matches:
                raise PageReadBindingError("candidate is not the exact legacy projection of an observed page")
            # Do not resolve a source collision using relevance/exact_machine flags.
            bindings = {(self._observations[i].source, self._observations[i].page_number,
                         self._observations[i].text_projection) for i in matches}
            if len(bindings) != 1:
                raise PageReadBindingError("ambiguous page binding; no score-based authorization")
            available = [i for i in matches if i not in used]
            if not available:
                raise PageReadBindingError("more candidate occurrences than observed pages")
            idx = available[0]
            if "snippet_clean" in candidate and candidate["snippet_clean"] != candidate["snippet"]:
                raise PageReadBindingError("candidate clean snippet not the observed projection")
            used.add(idx); indices.append(idx)
        return assemble_page_read(scope=self.scope, kind=self.kind,
            observations=tuple(self._observations), indices=tuple(indices), records=candidates,
            limits=self.limits, query_count=self._queries)
