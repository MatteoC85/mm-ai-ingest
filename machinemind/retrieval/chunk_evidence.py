"""P6-B1: canonical views of fresh, scoped document_chunks reads.

Only trusted repository readers call ``build_chunk_read``. Authorization is
bound from columns returned by their scoped SQL, BEFORE looking at candidates.
This is not an ACL service, a client deserializer, a cache, or a live ASK switch.
No I/O, environment, globals, LLMs or additional source lookup exist here.

Both views are retained: the exact indexed SQL text projections and raw read
score, plus the unchanged legacy candidate frozen by the existing P5 bundle.
A LEFT(text, N) projection is not claimed to be the whole document/chunk. File
format, physical-page basis, sheet/row, revision, links and relationships absent
from this read are NOT inferred from text, a title, an ID, a URL or a question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..evidence.adapter_types import AdapterContext, AdapterLimits
from ..evidence.assembly import AssemblyLimits
from ..evidence.contracts import (
    EvidenceContractError, Provenance, RetrievalScore, ScopeLevel, SourceFormat,
    SourceIdentity, SourceLocator, SourceScope, SourceType, canonical_json,
)
from ..evidence.legacy_compatibility import (
    LegacyEvidenceBundle, LegacyLimits, LegacyRecordInput, build_legacy_bundle,
    restore_legacy_records,
)

PROVIDER_VERSION = "scoped-chunk-reader-p6b1-v1"
_READ_KINDS = frozenset({"dense", "fts", "fts_prefix"})
# Storage prefixes written by the existing ingest contract, not domain keywords.
_STRUCTURED_PREFIXES = {s.value: s for s in SourceType if s != SourceType.DOCUMENT}


class ChunkReadBindingError(EvidenceContractError):
    """A read cannot be bound without guessing, broadening scope or losing data."""


def _identifier(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or not value.strip():
        raise ChunkReadBindingError(name + ": nonblank string required")


@dataclass(frozen=True, slots=True)
class ChunkReadScope:
    """An INTERNAL, already resolved scope; no public aliases or normalization.

    document_ids takes precedence over the single-document selector exactly as
    in the current SQL. An explicit document list selects those IDs in the
    company, including another machine's document when explicitly selected. A
    single-document selector still applies the machine/company predicate.
    Company-general reads admit only NULL/empty stored machine associations.
    ``machine_id`` retains the resolved SQL selector (including a sentinel); it
    is NEVER copied to the source's actual association.
    """
    company_id: str
    machine_id: str | None
    ai_scope: str
    document_ids: tuple[str, ...] = ()
    bubble_document_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.company_id, "company_id")
        _identifier(self.machine_id, "machine_id", optional=True)
        _identifier(self.bubble_document_id, "bubble_document_id", optional=True)
        if self.ai_scope not in {"machine_all", "company_general", "document_ids"}:
            raise ChunkReadBindingError("resolved ai_scope required")
        if type(self.document_ids) is not tuple:
            raise ChunkReadBindingError("immutable document selectors required")
        for value in self.document_ids:
            _identifier(value, "document_ids item")
        if self.ai_scope == "document_ids":
            if not self.document_ids and self.bubble_document_id is None:
                raise ChunkReadBindingError("document_ids scope requires an explicit selector")
        elif self.document_ids or self.bubble_document_id is not None:
            raise ChunkReadBindingError("unresolved document selectors outside document scope")
        if self.ai_scope == "machine_all" and self.machine_id is None:
            raise ChunkReadBindingError("machine_all requires a machine selector")

    def permits(self, company_id: str, machine_id: str | None, document_id: str) -> bool:
        """Check DB columns, not a title, score or exact_machine_scope flag."""
        if company_id != self.company_id:
            return False
        if self.ai_scope == "company_general":
            return machine_id is None or machine_id == ""
        if self.document_ids:
            return document_id in self.document_ids
        if self.bubble_document_id is not None and document_id != self.bubble_document_id:
            return False
        return machine_id is None or machine_id == "" or machine_id == self.machine_id

    def sql_selectors(self) -> dict[str, Any]:
        """Fresh arguments for the pre-existing reader; no mutable scope alias."""
        return {"company_id": self.company_id, "machine_id": self.machine_id,
                "doc_ids": list(self.document_ids) if self.document_ids else None,
                "bubble_document_id": self.bubble_document_id}


@dataclass(frozen=True, slots=True)
class ChunkEvidenceLimits:
    """Reuse P5 allocation budgets; no new model, token, query or dollar limit.

    legacy.max_bytes also bounds the sum of the legacy bundle and the exact SQL
    observations retained here. Overflows fail atomically, never truncate.
    """
    adapter: AdapterLimits
    assembly: AssemblyLimits
    legacy: LegacyLimits

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, AdapterLimits) or not isinstance(self.assembly, AssemblyLimits) or not isinstance(self.legacy, LegacyLimits):
            raise ChunkReadBindingError("explicit typed P5 allocation limits required")


def validate_read(scope: ChunkReadScope, limit: int, limits: ChunkEvidenceLimits) -> None:
    """Called before opening the DB, including on empty-query paths."""
    if not isinstance(scope, ChunkReadScope) or not isinstance(limits, ChunkEvidenceLimits):
        raise ChunkReadBindingError("typed scope and allocation limits required")
    if type(limit) is not int or limit < 0 or limit > limits.assembly.max_occurrences:
        raise ChunkReadBindingError("read LIMIT outside the explicit occurrence budget")


@dataclass(frozen=True, slots=True)
class ChunkReadObservation:
    """One immutable DB-row observation. No embedding or temporary URL."""
    source: SourceIdentity
    provenance: Provenance
    locator: SourceLocator
    stored_company_id: str
    stored_machine_id: str | None
    storage_document_id: str
    snippet_projection: str | None = field(repr=False)
    chunk_projection: str | None = field(repr=False)
    raw_score: RetrievalScore

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.provenance, Provenance) or not isinstance(self.locator, SourceLocator) or not isinstance(self.raw_score, RetrievalScore):
            raise ChunkReadBindingError("typed read observation required")
        _identifier(self.stored_company_id, "stored company")
        _identifier(self.storage_document_id, "stored document")
        if self.stored_machine_id is not None and type(self.stored_machine_id) is not str:
            raise ChunkReadBindingError("stored machine must be text, empty or NULL")
        if self.stored_machine_id not in (None, "") and not self.stored_machine_id.strip():
            raise ChunkReadBindingError("whitespace-only stored machine is not company scope")
        if self.source.scope.company_id != self.stored_company_id or self.source.scope.machine_id != (self.stored_machine_id or None):
            raise ChunkReadBindingError("source binding differs from actual DB association")
        expected_key = self.source.source_id if self.source.source_type == SourceType.DOCUMENT else self.source.source_type.value + ":" + self.source.source_id
        if expected_key != self.storage_document_id:
            raise ChunkReadBindingError("storage key differs from typed source identity")
        for value in (self.snippet_projection, self.chunk_projection):
            if value is not None and type(value) is not str:
                raise ChunkReadBindingError("SQL text projection must remain text or NULL")


def _source_from_columns(company_id: str, machine_id: str | None, key: str) -> SourceIdentity:
    _identifier(company_id, "stored company")
    _identifier(key, "stored document")
    if machine_id is not None and type(machine_id) is not str:
        raise ChunkReadBindingError("stored machine must be text or NULL")
    if machine_id not in (None, ""):
        _identifier(machine_id, "stored machine")
    scope = SourceScope(company_id, ScopeLevel.COMPANY if machine_id in (None, "") else ScopeLevel.MACHINE,
                        None if machine_id in (None, "") else machine_id)
    prefix, separator, identifier = key.partition(":")
    if separator and prefix in _STRUCTURED_PREFIXES:
        _identifier(identifier, "structured storage identifier")
        return SourceIdentity(scope, _STRUCTURED_PREFIXES[prefix], identifier, SourceFormat.STRUCTURED)
    # Do not silently turn a noncanonical structured key into a Document. P5
    # currently requires the canonical lowercase prefix for a lossless binding.
    if separator and prefix.lower() in _STRUCTURED_PREFIXES:
        raise ChunkReadBindingError("noncanonical structured storage prefix")
    return SourceIdentity(scope, SourceType.DOCUMENT, key, SourceFormat.UNKNOWN)


def strip_binding_columns(rows: list[tuple], *, width: int) -> list[tuple]:
    """Remove only the two repository-projected columns, never infer a binding."""
    if type(rows) is not list or any(type(r) is not tuple or len(r) != width + 2 for r in rows):
        raise ChunkReadBindingError("unexpected bound SQL row shape")
    return [row[:width] for row in rows]


@dataclass(frozen=True, slots=True)
class ChunkEvidenceRead:
    """Fresh provider result; NOT serializable proof of current authorization.

    ``read_sources`` records what this scoped DB read admitted at acquisition.
    A later consumer must supply its CURRENT scope and allowance explicitly.
    Do not recreate that allowance from candidates, a cache or this bundle.
    Canonical evidence and legacy snapshots use the existing P5 contracts.
    """
    scope: ChunkReadScope
    kind: str
    observations: tuple[ChunkReadObservation, ...] = field(repr=False)
    bundle: LegacyEvidenceBundle = field(repr=False)
    size_bytes: int
    chunks_matching_filter: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ChunkReadScope) or self.kind not in _READ_KINDS or not isinstance(self.bundle, LegacyEvidenceBundle):
            raise ChunkReadBindingError("invalid scoped chunk read")
        if type(self.observations) is not tuple or any(not isinstance(x, ChunkReadObservation) for x in self.observations):
            raise ChunkReadBindingError("immutable DB observations required")
        entries = self.bundle.assembly.occurrence_entries()
        if len(entries) != len(self.observations) or self.bundle.assembly.manifest.company_id != self.scope.company_id:
            raise ChunkReadBindingError("read/bundle occurrence or company mismatch")
        for observation, entry in zip(self.observations, entries):
            if not self.scope.permits(observation.stored_company_id, observation.stored_machine_id, observation.storage_document_id):
                raise ChunkReadBindingError("DB source outside resolved query scope")
            if entry.evidence.source != observation.source or entry.evidence.provenance != observation.provenance:
                raise ChunkReadBindingError("canonical source differs from DB observation")
        size = self.bundle.size_bytes + sum(len(canonical_json(o).encode("utf-8")) for o in self.observations)
        if type(self.size_bytes) is not int or self.size_bytes != size or size > self.bundle.limits.max_bytes:
            raise ChunkReadBindingError("aggregate read snapshot byte budget exceeded")
        if self.chunks_matching_filter is not None and (type(self.chunks_matching_filter) is not int or self.chunks_matching_filter < 0):
            raise ChunkReadBindingError("invalid debug count")

    @property
    def read_sources(self) -> frozenset[SourceIdentity]:
        return frozenset(o.source for o in self.observations)

    def as_legacy_inputs(self, *, scope: ChunkReadScope,
                         current_allowed_sources: frozenset[SourceIdentity],
                         adapter_limits: AdapterLimits) -> tuple[LegacyRecordInput, ...]:
        """Fresh restored records for P6-A/common consumers; no implicit grants."""
        if not isinstance(scope, ChunkReadScope) or scope != self.scope:
            raise ChunkReadBindingError("query scope changed since source acquisition")
        restored = restore_legacy_records(self.bundle, company_id=scope.company_id,
                                           allowed_sources=current_allowed_sources,
                                           adapter_limits=adapter_limits)
        return tuple(LegacyRecordInput("retrieval_candidate", item.record,
                     AdapterContext(o.source, o.provenance, current_allowed_sources))
                     for item, o in zip(restored, self.observations))


def build_chunk_read(*, scope: ChunkReadScope, kind: str, rows: list[tuple],
                     candidates: list[dict], limits: ChunkEvidenceLimits,
                     chunks_matching_filter: int | None = None) -> ChunkEvidenceRead:
    """Repository-only constructor from the same authorized SQL result.

    Dense row layout is the historical nine columns followed by company/machine;
    FTS is its historical six followed by company/machine. These are explicit
    layouts, not detection by candidate fields or text. Every occurrence remains
    in the order read. Scores never establish source authorization.
    """
    if kind not in _READ_KINDS:
        raise ChunkReadBindingError("unsupported explicit read kind")
    if type(rows) is not list:
        raise ChunkReadBindingError("explicit SQL row list required")
    validate_read(scope, len(rows), limits)
    width = 9 if kind == "dense" else 6
    legacy_rows = strip_binding_columns(rows, width=width)
    if type(candidates) is not list or len(candidates) != len(legacy_rows):
        raise ChunkReadBindingError("one legacy candidate per observed SQL row required")
    observations = []
    byte_count = 0
    for row in rows:
        key, chunk, page_from, page_to = row[:4]
        company, machine = row[-2:]
        for value in (key, company, machine):
            if type(value) is str and len(value) > limits.adapter.max_aux_chars:
                raise ChunkReadBindingError("read identity exceeds auxiliary allocation budget")
        for value in (row[4], row[5] if kind == "dense" else None):
            if value is not None and (type(value) is not str or len(value) > limits.adapter.max_text_chars):
                raise ChunkReadBindingError("exact SQL text projection exceeds allocation budget")
        # Scope is bound from DB columns before inspecting the candidate.
        source = _source_from_columns(company, machine, key)
        if not scope.permits(company, machine, key):
            raise ChunkReadBindingError("DB source outside resolved query scope")
        locator = SourceLocator(page_from=page_from, page_to=page_to, chunk_index=chunk)
        provenance = Provenance(PROVIDER_VERSION + ":" + kind,
                                 canonical_json(("public.document_chunks", company, machine, key, chunk, page_from, page_to)))
        observation = ChunkReadObservation(
            source, provenance, locator, company, machine, key, row[4],
            row[5] if kind == "dense" else None,
            RetrievalScore("semantic_similarity" if kind == "dense" else "fts_rank",
                           row[6] if kind == "dense" else row[5], PROVIDER_VERSION + ":" + kind))
        byte_count += len(canonical_json(observation).encode("utf-8"))
        if byte_count > limits.legacy.max_bytes:
            raise ChunkReadBindingError("read observation budget exceeded")
        observations.append(observation)
    allowance = frozenset(o.source for o in observations)
    inputs = []
    for candidate, row, observation in zip(candidates, legacy_rows, observations):
        if type(candidate) is not dict:
            raise ChunkReadBindingError("legacy candidate mapping required")
        expected = {"bubble_document_id": row[0], "chunk_index": row[1],
                    "page_from": row[2], "page_to": row[3],
                    "citation_id": f"{row[0]}:p{row[2]}-{row[3]}:c{row[1]}",
                    "snippet": (row[4] or "").strip(),
                    "similarity": float(row[6]) if kind == "dense" else 0.0}
        if kind == "dense":
            expected["chunk_full"] = (row[5] or "").strip()
            expected["semantic_similarity"] = float(row[6])
        elif "chunk_full" in candidate or "semantic_similarity" in candidate:
            raise ChunkReadBindingError("FTS candidate invents unobserved chunk/semantic data")
        if any(type(candidate.get(k)) is not type(v) or candidate.get(k) != v
               or (type(v) is float and candidate[k].hex() != v.hex())
               for k, v in expected.items()):
            raise ChunkReadBindingError("candidate differs from the observed SQL row")
        inputs.append(LegacyRecordInput("retrieval_candidate", candidate,
                      AdapterContext(observation.source, observation.provenance, allowance)))
    bundle = build_legacy_bundle(inputs, company_id=scope.company_id, allowed_sources=allowance,
                                 adapter_limits=limits.adapter, assembly_limits=limits.assembly,
                                 legacy_limits=limits.legacy)
    return ChunkEvidenceRead(scope, kind, tuple(observations), bundle, bundle.size_bytes + byte_count,
                             chunks_matching_filter)
