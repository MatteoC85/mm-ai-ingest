"""P5-C: bounded, ordered assembly of already-authorized adaptation results.

No retrieval, ranking, authorization lookup, inferred relation or text cleaning.
The manifest stores each EXACT immutable record once, in first-occurrence order.
Every input occurrence retains its own scores and adaptation trace. Duplicates
are not semantically merged and conflicting records are never resolved here.

Allowances must be supplied by a trusted provider, not reconstructed from input
records or from a client-supplied serialized manifest. No deserializer is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .adapter_types import AdaptationResult, AdaptationTrace
from .contracts import EvidenceContractError, RetrievalScore, SourceIdentity, to_primitive
from .manifest import EvidenceEntry, EvidenceManifest, ManifestLimits, build_manifest

ASSEMBLY_VERSION = "canonical-evidence-assembly-v1"


class EvidenceAssemblyError(EvidenceContractError):
    """Assembly would lose information, violate scope, or exceed its bounds."""


def _int(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise EvidenceAssemblyError(name + ": invalid integer bound")


def _allowance(company_id: str, allowed_sources: frozenset[SourceIdentity]) -> None:
    if type(company_id) is not str or not company_id.strip():
        raise EvidenceAssemblyError("explicit company required")
    if type(allowed_sources) is not frozenset or any(
        not isinstance(s, SourceIdentity) or s.scope.company_id != company_id
        for s in allowed_sources
    ):
        raise EvidenceAssemblyError("explicit same-company provider allowance required")


def _trace(trace: AdaptationTrace) -> None:
    if not isinstance(trace, AdaptationTrace):
        raise EvidenceAssemblyError("adaptation trace required")
    for value in (trace.layout, trace.text_field, trace.adapter_version):
        if type(value) is not str or not value.strip():
            raise EvidenceAssemblyError("invalid adaptation trace")
    if trace.legacy_citation_id is not None and (
        type(trace.legacy_citation_id) is not str or not trace.legacy_citation_id.strip()
    ):
        raise EvidenceAssemblyError("invalid legacy citation association")
    groups = (trace.consumed_fields, trace.unmapped_fields, trace.excluded_transport_fields)
    for group in groups:
        if type(group) is not tuple or any(type(s) is not str or not s for s in group):
            raise EvidenceAssemblyError("trace fields must be immutable strings")
        if len(set(group)) != len(group):
            raise EvidenceAssemblyError("duplicate trace field")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise EvidenceAssemblyError("overlapping adaptation trace groups")


@dataclass(frozen=True, slots=True)
class AssemblyLimits:
    """Explicit allocation bounds, not token limits or economic budgets.

    max_occurrences includes repeated records. manifest_limits counts unique
    records. max_serialized_bytes includes the manifest AND occurrence traces.
    Exceeding any limit raises; there is no silent truncation or partial result.
    """
    manifest_limits: ManifestLimits
    max_occurrences: int
    max_serialized_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_limits, ManifestLimits):
            raise EvidenceAssemblyError("explicit manifest limits required")
        _int(self.max_occurrences, "max_occurrences")
        _int(self.max_serialized_bytes, "max_serialized_bytes", 1)


@dataclass(frozen=True, slots=True)
class EvidenceOccurrence:
    position: int  # Zero-based INPUT position; not a rank or source ordinal.
    evidence_id: str
    scores: tuple[RetrievalScore, ...]
    trace: AdaptationTrace

    def __post_init__(self) -> None:
        _int(self.position, "occurrence position")
        if type(self.evidence_id) is not str or not self.evidence_id.startswith("ev1_"):
            raise EvidenceAssemblyError("invalid occurrence evidence reference")
        if type(self.scores) is not tuple or any(not isinstance(s, RetrievalScore) for s in self.scores):
            raise EvidenceAssemblyError("immutable scores required")
        if len({(s.producer, s.name) for s in self.scores}) != len(self.scores):
            raise EvidenceAssemblyError("duplicate score from a producer")
        _trace(self.trace)


@dataclass(frozen=True, slots=True)
class EvidenceAssembly:
    manifest: EvidenceManifest
    occurrences: tuple[EvidenceOccurrence, ...]
    limits: AssemblyLimits

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, EvidenceManifest) or not isinstance(self.limits, AssemblyLimits):
            raise EvidenceAssemblyError("manifest and assembly limits required")
        if self.manifest.limits != self.limits.manifest_limits:
            raise EvidenceAssemblyError("manifest limits differ from assembly limits")
        if type(self.occurrences) is not tuple or any(
            not isinstance(o, EvidenceOccurrence) for o in self.occurrences
        ):
            raise EvidenceAssemblyError("immutable occurrence sequence required")
        if len(self.occurrences) > self.limits.max_occurrences:
            raise EvidenceAssemblyError("occurrence budget exceeded")
        by_id = {e.evidence.evidence_id: e for e in self.manifest.entries}
        first_seen = []
        seen = set()
        for entry in self.manifest.entries:
            if entry.scores:
                raise EvidenceAssemblyError("scores belong to occurrences, not unique records")
            if any(r.target not in self.manifest.allowed_sources for r in entry.evidence.relations):
                raise EvidenceAssemblyError("relation target outside assembly allowance")
        for index, occurrence in enumerate(self.occurrences):
            if occurrence.position != index or occurrence.evidence_id not in by_id:
                raise EvidenceAssemblyError("invalid occurrence order or reference")
            if occurrence.evidence_id not in seen:
                seen.add(occurrence.evidence_id)
                first_seen.append(occurrence.evidence_id)
        if first_seen != list(by_id):
            raise EvidenceAssemblyError("manifest must follow first occurrences with no unreferenced records")
        if len(self.to_json().encode("utf-8")) > self.limits.max_serialized_bytes:
            raise EvidenceAssemblyError("assembly serialized budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        """Canonical data only. Never serialize the provider's source allowance."""
        return {"assembly_version": ASSEMBLY_VERSION, "manifest": self.manifest.to_dict(),
                "occurrences": [to_primitive(o) for o in self.occurrences]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @property
    def assembly_id(self) -> str:
        return "ea1_" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def occurrence_entries(self) -> tuple[EvidenceEntry, ...]:
        """Recover input order, multiplicity and scores, without reranking."""
        records = {e.evidence.evidence_id: e.evidence for e in self.manifest.entries}
        return tuple(EvidenceEntry(records[o.evidence_id], o.scores) for o in self.occurrences)

    def citation_occurrences(self, legacy_citation_id: str) -> tuple[int, ...]:
        """An ambiguous old citation returns ALL matches, never an arbitrary one."""
        if type(legacy_citation_id) is not str or not legacy_citation_id.strip():
            raise EvidenceAssemblyError("citation association must be nonblank text")
        return tuple(o.position for o in self.occurrences
                     if o.trace.legacy_citation_id == legacy_citation_id)


def assemble_evidence(results: Iterable[AdaptationResult], *, company_id: str,
                      allowed_sources: frozenset[SourceIdentity],
                      limits: AssemblyLimits) -> EvidenceAssembly:
    """Build atomically; exact duplicates are indexed, not silently discarded.

    Relations can name allowed sources not included in this batch. They neither
    fetch nor authorize those sources. No semantic/recency conflict is resolved.
    """
    _allowance(company_id, allowed_sources)
    if not isinstance(limits, AssemblyLimits):
        raise EvidenceAssemblyError("explicit assembly limits required")
    entries: list[EvidenceEntry] = []
    occurrences: list[EvidenceOccurrence] = []
    records = {}
    for result in results:
        if len(occurrences) >= limits.max_occurrences:
            raise EvidenceAssemblyError("occurrence budget exceeded")
        if not isinstance(result, AdaptationResult):
            raise EvidenceAssemblyError("expected an adaptation result")
        record = result.entry.evidence
        if record.source.scope.company_id != company_id or record.source not in allowed_sources:
            raise EvidenceAssemblyError("record outside assembly allowance")
        if any(r.target not in allowed_sources for r in record.relations):
            raise EvidenceAssemblyError("relation target outside assembly allowance")
        eid = record.evidence_id
        if eid in records and records[eid] != record:
            raise EvidenceAssemblyError("conflicting data under one evidence identifier")
        if eid not in records:
            if len(entries) >= limits.manifest_limits.max_records:
                raise EvidenceAssemblyError("unique record budget exceeded")
            records[eid] = record
            entries.append(EvidenceEntry(record))
        occurrences.append(EvidenceOccurrence(len(occurrences), eid, result.entry.scores, result.trace))
    manifest = build_manifest(company_id=company_id, entries=entries,
                              allowed_sources=allowed_sources, limits=limits.manifest_limits)
    return EvidenceAssembly(manifest, tuple(occurrences), limits)
