"""P6-B3: scoped projections for auxiliary chunk reads and file references.

Pure repository boundary, not a second Core or an ACL service. It uses P5
contracts/snapshots and the resolved scope/limits from P6-B1. Source bindings
come from columns of the same scoped SELECT, BEFORE selectors or ranking.
File URLs remain private, unverified metadata attached to CURRENT authorized
anchors; document_files has no machine column and cannot grant text access.
No I/O, SQL text rewriting, language interpretation or environment lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunk_evidence import (ChunkReadScope, ChunkEvidenceLimits,
    _source_from_columns, validate_read)
from ..evidence.adapter_types import AdapterContext, AdapterLimits
from ..evidence.contracts import (EvidenceContractError, LinkTarget, Provenance,
    RetrievalScore, SourceIdentity, SourceLocator, SourceType, canonical_json)
from ..evidence.legacy_compatibility import (LegacyEvidenceBundle, LegacyRecordInput,
    build_legacy_bundle, restore_legacy_records)

PROVIDER_VERSION = "scoped-supplemental-readers-p6b3-v1"
CHUNK_BINDING_COLUMNS = ", company_id AS evidence_company_id, machine_id AS evidence_machine_id"
FILE_BINDING_COLUMNS = ", company_id AS evidence_company_id"
_CHUNK_KINDS = frozenset({"token_chunk", "entity_chunk", "structured_rescue", "structured_dense", "neighbor_chunks"})


class SupplementalBindingError(EvidenceContractError):
    """Technical failure, never an empty successful result or implicit retry."""


def require_machine_scope(scope: ChunkReadScope) -> None:
    if not isinstance(scope, ChunkReadScope) or scope.ai_scope != "machine_all":
        raise SupplementalBindingError("this legacy reader requires resolved machine_all scope")


def storage_key(source: SourceIdentity) -> str:
    if not isinstance(source, SourceIdentity):
        raise SupplementalBindingError("typed source required")
    return source.source_id if source.source_type == SourceType.DOCUMENT else source.source_type.value + ":" + source.source_id


def validate_anchors(*, scope: ChunkReadScope, anchors: tuple[SourceIdentity, ...],
                     current_allowed_sources: frozenset[SourceIdentity],
                     limits: ChunkEvidenceLimits) -> None:
    if type(anchors) is not tuple or type(current_allowed_sources) is not frozenset:
        raise SupplementalBindingError("immutable anchors and current allowance required")
    validate_read(scope, len(anchors), limits)
    if any(not isinstance(s, SourceIdentity) for s in current_allowed_sources):
        raise SupplementalBindingError("typed current allowance required")
    for s in anchors:
        if not isinstance(s, SourceIdentity) or s not in current_allowed_sources:
            raise SupplementalBindingError("anchor missing from current allowance")
        if not scope.permits(s.scope.company_id, s.scope.machine_id, storage_key(s)):
            raise SupplementalBindingError("anchor outside request scope")
    by_key: dict[str, SourceIdentity] = {}
    for s in anchors:
        key = storage_key(s)
        if key in by_key and by_key[key] != s:
            raise SupplementalBindingError("ambiguous current source for storage key")
        by_key[key] = s


def checked_input_records(*, scope: ChunkReadScope, records: tuple[LegacyRecordInput, ...],
                          current_allowed_sources: frozenset[SourceIdentity],
                          limits: ChunkEvidenceLimits) -> list[dict]:
    """Revalidate provider-bound input records without deriving grants from dicts."""
    if type(records) is not tuple or any(not isinstance(r, LegacyRecordInput) for r in records):
        raise SupplementalBindingError("provider-bound immutable inputs required")
    validate_anchors(scope=scope, anchors=tuple(r.context.source for r in records),
                     current_allowed_sources=current_allowed_sources, limits=limits)
    bundle = build_legacy_bundle(records, company_id=scope.company_id,
        allowed_sources=current_allowed_sources, adapter_limits=limits.adapter,
        assembly_limits=limits.assembly, legacy_limits=limits.legacy)
    return [v.record for v in restore_legacy_records(bundle, company_id=scope.company_id,
        allowed_sources=current_allowed_sources, adapter_limits=limits.adapter)]


@dataclass(frozen=True, slots=True)
class SupplementalChunkObservation:
    source: SourceIdentity
    provenance: Provenance
    locator: SourceLocator
    stored_company_id: str
    stored_machine_id: str | None
    storage_document_id: str
    snippet_projection: str | None = field(repr=False)
    chunk_projection: str | None = field(repr=False)
    snippet_chars: int
    chunk_chars: int | None
    batch: int
    row: int
    query_text: str | None = field(repr=False)
    raw_score: RetrievalScore | None = None

    def __post_init__(self) -> None:
        if self.source != _source_from_columns(self.stored_company_id, self.stored_machine_id, self.storage_document_id):
            raise SupplementalBindingError("chunk association differs from SQL columns")
        if not isinstance(self.provenance, Provenance) or not isinstance(self.locator, SourceLocator):
            raise SupplementalBindingError("typed chunk provenance and locator required")
        for value in (self.snippet_chars, self.batch, self.row):
            if type(value) is not int or value < 0:
                raise SupplementalBindingError("invalid chunk projection bound or position")
        if self.chunk_chars is not None and (type(self.chunk_chars) is not int or self.chunk_chars < 0):
            raise SupplementalBindingError("invalid chunk projection bound")
        for text, chars in ((self.snippet_projection,self.snippet_chars), (self.chunk_projection,self.chunk_chars)):
            if text is not None and (type(text) is not str or chars is None or len(text)>chars):
                raise SupplementalBindingError("SQL chunk projection must remain bounded text or NULL")
        if self.query_text is not None and type(self.query_text) is not str:
            raise SupplementalBindingError("query text must remain literal text")
        if self.raw_score is not None and not isinstance(self.raw_score, RetrievalScore):
            raise SupplementalBindingError("typed raw retrieval score required")


@dataclass(frozen=True, slots=True)
class SupplementalChunkRead:
    scope: ChunkReadScope
    kind: str
    observations: tuple[SupplementalChunkObservation, ...] = field(repr=False)
    selected_observation_indices: tuple[int, ...]
    bundle: LegacyEvidenceBundle = field(repr=False)
    size_bytes: int
    query_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ChunkReadScope) or self.kind not in _CHUNK_KINDS or not isinstance(self.bundle, LegacyEvidenceBundle):
            raise SupplementalBindingError("typed supplemental read required")
        if type(self.observations) is not tuple or any(not isinstance(o,SupplementalChunkObservation) for o in self.observations):
            raise SupplementalBindingError("immutable SQL observations required")
        if type(self.selected_observation_indices) is not tuple:
            raise SupplementalBindingError("immutable occurrence selection required")
        if type(self.query_count) is not int or self.query_count < 0:
            raise SupplementalBindingError("invalid query count")
        if self.bundle.assembly.manifest.company_id != self.scope.company_id:
            raise SupplementalBindingError("bundle company differs from request")
        entries=self.bundle.assembly.occurrence_entries()
        if len(entries)!=len(self.selected_observation_indices):
            raise SupplementalBindingError("selected occurrence count differs")
        for o in self.observations:
            if not self.scope.permits(o.stored_company_id,o.stored_machine_id,o.storage_document_id):
                raise SupplementalBindingError("SQL chunk outside request scope")
        for idx,entry in zip(self.selected_observation_indices,entries):
            if type(idx) is not int or idx<0 or idx>=len(self.observations):
                raise SupplementalBindingError("missing selected observation")
            o=self.observations[idx]
            if entry.evidence.source!=o.source or entry.evidence.provenance!=o.provenance:
                raise SupplementalBindingError("selected binding/provenance differs")
        size=self.bundle.size_bytes+sum(len(canonical_json(o).encode('utf-8')) for o in self.observations)
        if type(self.size_bytes) is not int or self.size_bytes!=size or size>self.bundle.limits.max_bytes:
            raise SupplementalBindingError("aggregate supplemental byte budget exceeded")

    @property
    def read_sources(self) -> frozenset[SourceIdentity]:
        """Fresh acquisition only, NOT an allowance for a future cached request."""
        return frozenset(o.source for o in self.observations)

    def as_legacy_inputs(self, *, scope: ChunkReadScope,
                         current_allowed_sources: frozenset[SourceIdentity],
                         adapter_limits: AdapterLimits) -> tuple[LegacyRecordInput,...]:
        if not isinstance(scope,ChunkReadScope) or scope!=self.scope:
            raise SupplementalBindingError("scope changed since acquisition")
        restored=restore_legacy_records(self.bundle,company_id=scope.company_id,
            allowed_sources=current_allowed_sources,adapter_limits=adapter_limits)
        return tuple(LegacyRecordInput("retrieval_candidate",r.record,
            AdapterContext(self.observations[i].source,self.observations[i].provenance,current_allowed_sources))
            for r,i in zip(restored,self.selected_observation_indices))


class _ChunkSelectionCapture:
    """One request-local collector; explicit known layouts, never row autodetection."""
    def __init__(self, scope: ChunkReadScope, limits: ChunkEvidenceLimits, kind: str,
                 *, snippet_chars: int, max_queries: int = 1) -> None:
        validate_read(scope,0,limits)
        if kind not in _CHUNK_KINDS or type(max_queries) is not int or max_queries<1 or max_queries>limits.assembly.max_occurrences:
            raise SupplementalBindingError("invalid bounded chunk read configuration")
        if type(snippet_chars) is not int or snippet_chars<0 or snippet_chars>limits.adapter.max_text_chars:
            raise SupplementalBindingError("invalid snippet bound")
        self.scope,self.limits,self.kind=scope,limits,kind
        self.snippet_chars,self.max_queries=snippet_chars,max_queries
        self.observations=[];self.queries=0;self.byte_count=0;self.pending=None;self.finished=False

    def expect(self, row_limit: int, *, query_text: str | None = None) -> None:
        if self.finished or self.pending is not None or self.queries>=self.max_queries:
            raise SupplementalBindingError("chunk capture lifecycle violation")
        validate_read(self.scope,row_limit,self.limits)
        if query_text is not None and (type(query_text) is not str or len(query_text)>self.limits.adapter.max_aux_chars):
            raise SupplementalBindingError("invalid literal query metadata")
        full_chars=None if self.kind in {"token_chunk","entity_chunk"} else 2400 if self.kind=="structured_dense" else 2000
        if full_chars is not None and full_chars>self.limits.adapter.max_text_chars:
            raise SupplementalBindingError("chunk projection exceeds explicit budget")
        self.pending=(row_limit,query_text,full_chars)

    def capture(self, rows: list[tuple]) -> list[tuple]:
        if self.finished or self.pending is None:
            raise SupplementalBindingError("chunk capture without a bounded read")
        limit,query,full_chars=self.pending
        width=5 if self.kind in {"token_chunk","entity_chunk"} else 9 if self.kind=="structured_dense" else 6
        if type(rows) is not list or len(rows)>limit or len(rows)+len(self.observations)>self.limits.assembly.max_occurrences:
            raise SupplementalBindingError("SQL chunk occurrence budget exceeded")
        out=[]
        for idx,raw in enumerate(rows):
            if type(raw) is not tuple or len(raw)!=width+2:
                raise SupplementalBindingError("unexpected bound chunk SQL layout")
            key,chunk,p1,p2,snippet=raw[:5];company,mid=raw[-2:]
            source=_source_from_columns(company,mid,key)
            if any(type(v) is str and len(v)>self.limits.adapter.max_aux_chars for v in (company,mid,key)):
                raise SupplementalBindingError("SQL identity exceeds explicit budget")
            if not self.scope.permits(company,mid,key):
                raise SupplementalBindingError("SQL chunk outside resolved scope")
            locator=SourceLocator(page_from=p1,page_to=p2,chunk_index=chunk)
            if p1 is None or p2 is None or chunk is None:
                raise SupplementalBindingError("missing SQL chunk coordinates")
            full=raw[5] if width>=6 else None
            score=RetrievalScore("dense_similarity",raw[6],PROVIDER_VERSION+":structured_dense") if width==9 else None
            provenance=Provenance(PROVIDER_VERSION+":"+self.kind,
                canonical_json(("public.document_chunks",company,mid,key,chunk,p1,p2,self.queries,idx)))
            o=SupplementalChunkObservation(source,provenance,locator,company,mid,key,snippet,full,
                self.snippet_chars,full_chars,self.queries,idx,query,score)
            self.byte_count+=len(canonical_json(o).encode('utf-8'))
            if self.byte_count>self.limits.legacy.max_bytes:
                raise SupplementalBindingError("SQL observations exceed byte budget")
            self.observations.append(o);out.append(raw[:width])
        self.queries+=1;self.pending=None
        return out

    def finish(self, candidates: list[dict]) -> SupplementalChunkRead:
        if self.finished or self.pending is not None or type(candidates) is not list:
            raise SupplementalBindingError("unfinished or reused chunk capture")
        self.finished=True;indices=[];used=set()
        for c in candidates:
            if type(c) is not dict:raise SupplementalBindingError("candidate mapping required")
            matches=[]
            for i,o in enumerate(self.observations):
                expected={"bubble_document_id":o.storage_document_id,"page_from":o.locator.page_from,
                    "page_to":o.locator.page_to,"snippet":(o.snippet_projection or "").strip()}
                if o.chunk_chars is not None:expected['chunk_full']=(o.chunk_projection or '').strip()
                if 'chunk_index' in c:expected['chunk_index']=o.locator.chunk_index
                if 'query_used' in c:expected['query_used']=o.query_text
                if all(type(c.get(k)) is type(v) and c[k]==v for k,v in expected.items()):matches.append(i)
            if not matches:raise SupplementalBindingError("candidate not a literal SQL projection")
            bindings={(self.observations[i].source,self.observations[i].locator,
                self.observations[i].snippet_projection,self.observations[i].chunk_projection) for i in matches}
            if len(bindings)!=1:raise SupplementalBindingError("ambiguous chunk association")
            available=[i for i in matches if i not in used]
            if not available:raise SupplementalBindingError("candidate occurrence not observed")
            if 'snippet_clean' in c and c['snippet_clean']!=c['snippet']:
                raise SupplementalBindingError("clean snippet differs from observed projection")
            idx=available[0];used.add(idx);indices.append(idx)
        allowed=frozenset(o.source for o in self.observations)
        inputs=[LegacyRecordInput('retrieval_candidate',c,AdapterContext(self.observations[i].source,
            self.observations[i].provenance,allowed)) for i,c in zip(indices,candidates)]
        bundle=build_legacy_bundle(inputs,company_id=self.scope.company_id,allowed_sources=allowed,
            adapter_limits=self.limits.adapter,assembly_limits=self.limits.assembly,legacy_limits=self.limits.legacy)
        size=bundle.size_bytes+self.byte_count
        return SupplementalChunkRead(self.scope,self.kind,tuple(self.observations),tuple(indices),bundle,size,self.queries)


@dataclass(frozen=True, slots=True)
class FileReferenceObservation:
    """A company-scoped URL row tied to a separately authorized document anchor."""
    source: SourceIdentity
    provenance: Provenance
    link_target: LinkTarget
    stored_company_id: str
    storage_document_id: str
    file_url: str | None = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source,SourceIdentity) or self.source.source_type!=SourceType.DOCUMENT:
            raise SupplementalBindingError("document anchor required for file metadata")
        if self.source.scope.company_id!=self.stored_company_id or storage_key(self.source)!=self.storage_document_id:
            raise SupplementalBindingError("file company/key differs from authorized anchor")
        if not isinstance(self.provenance,Provenance) or self.link_target!=LinkTarget('document_files',self.storage_document_id):
            raise SupplementalBindingError("invalid opaque file reference")
        if self.file_url is not None and type(self.file_url) is not str:
            raise SupplementalBindingError("file URL must remain text or NULL")


@dataclass(frozen=True, slots=True)
class FileReferenceRead:
    scope: ChunkReadScope
    anchors: tuple[SourceIdentity,...]
    observations: tuple[FileReferenceObservation,...] = field(repr=False)
    limits: ChunkEvidenceLimits = field(repr=False)
    size_bytes: int
    query_count: int

    def __post_init__(self) -> None:
        validate_read(self.scope,len(self.observations),self.limits)
        validate_anchors(scope=self.scope,anchors=self.anchors,current_allowed_sources=frozenset(self.anchors),limits=self.limits)
        if type(self.observations) is not tuple or any(not isinstance(o,FileReferenceObservation) or o.source not in self.anchors for o in self.observations):
            raise SupplementalBindingError("file observations do not match anchors")
        if type(self.query_count) is not int or self.query_count not in (0,1) or (self.observations and not self.query_count):
            raise SupplementalBindingError("invalid file query count")
        size=len(canonical_json((self.anchors,self.observations)).encode('utf-8'))
        if type(self.size_bytes) is not int or self.size_bytes!=size or size>self.limits.legacy.max_bytes:
            raise SupplementalBindingError("file reference allocation budget exceeded")

    def as_file_map(self, *, scope: ChunkReadScope,
                    current_allowed_sources: frozenset[SourceIdentity]) -> dict[str,str]:
        if not isinstance(scope,ChunkReadScope) or scope!=self.scope:
            raise SupplementalBindingError("scope changed before file-reference reuse")
        validate_anchors(scope=scope,anchors=self.anchors,current_allowed_sources=current_allowed_sources,limits=self.limits)
        # Same ordered last-row-wins projection as the legacy reader. Raw rows
        # remain observable; this is not URL validation or a claim of usability.
        return {o.storage_document_id:o.file_url.strip() for o in self.observations if o.file_url}


def build_file_reference_read(*, scope: ChunkReadScope, anchors: tuple[SourceIdentity,...],
                              current_allowed_sources: frozenset[SourceIdentity],
                              rows: list[tuple], limits: ChunkEvidenceLimits,
                              query_count: int) -> FileReferenceRead:
    validate_anchors(scope=scope,anchors=anchors,current_allowed_sources=current_allowed_sources,limits=limits)
    if any(s.source_type!=SourceType.DOCUMENT for s in anchors):
        raise SupplementalBindingError("file-map anchors must be documents")
    if type(rows) is not list:raise SupplementalBindingError("file rows required")
    validate_read(scope,len(rows),limits)
    by_key={storage_key(s):s for s in anchors};observations=[];size=0
    for idx,row in enumerate(rows):
        if type(row) is not tuple or len(row)!=3:raise SupplementalBindingError("unexpected file row layout")
        key,url,company=row
        if company!=scope.company_id or key not in by_key:
            raise SupplementalBindingError("file reference outside current authorized anchors")
        if type(url) is str and len(url)>limits.adapter.max_aux_chars:
            raise SupplementalBindingError("file URL exceeds explicit allocation bound")
        o=FileReferenceObservation(by_key[key],Provenance(PROVIDER_VERSION+':document_files',
            canonical_json(('public.document_files',company,key,idx))),LinkTarget('document_files',key),company,key,url)
        size+=len(canonical_json(o).encode('utf-8'))
        if size>limits.legacy.max_bytes:raise SupplementalBindingError("file observation budget exceeded")
        observations.append(o)
    obs=tuple(observations)
    return FileReferenceRead(scope,anchors,obs,limits,len(canonical_json((anchors,obs)).encode('utf-8')),query_count)


def require_request_scope(scope: ChunkReadScope, request: Any) -> None:
    """Compare already-resolved selectors, never derive authorization from metadata."""
    if not isinstance(scope, ChunkReadScope):
        raise SupplementalBindingError("resolved scope required")
    if (request.company_id, request.machine_id, request.ai_scope) != (scope.company_id, scope.machine_id, scope.ai_scope):
        raise SupplementalBindingError("request/scope mismatch")
    raw_ids=request.metadata.get("document_ids")
    ids=tuple(raw_ids) if type(raw_ids) in (list,tuple) else () if raw_ids is None else None
    single=request.metadata.get("bubble_document_id")
    if ids!=scope.document_ids or single!=scope.bubble_document_id:
        raise SupplementalBindingError("request/document selectors mismatch")
