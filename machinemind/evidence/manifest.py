"""Bounded ordered manifests built from an explicit trusted source allowance.

The allowance MUST be supplied by the already-authorized provider. This module
cannot authenticate users, verify ACLs or reconstruct missing machine associations. It is
an additional consistency boundary, NOT a replacement for SQL/tenant filters.
An unresolved machine association stays unresolved; an external provider may
explicitly authorize that source through document/company rules. Missing metadata
does not create permission and does not erase a verified document allowance.
It neither follows relations nor resolves application links.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Any

from .contracts import (SCHEMA_VERSION, EvidenceContractError, EvidenceRecord,
                        RetrievalScore, SourceIdentity, ScopeLevel,
                        _text, _number, _tuple_of, to_primitive)


@dataclass(frozen=True, slots=True)
class ManifestLimits:
    max_records: int
    max_serialized_bytes: int

    def __post_init__(self) -> None:
        _number(self.max_records, "max_records", 0)
        _number(self.max_serialized_bytes, "max_serialized_bytes", 1)


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    evidence: EvidenceRecord
    scores: tuple[RetrievalScore, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceRecord):
            raise EvidenceContractError("entry requires an EvidenceRecord")
        _tuple_of(self.scores, RetrievalScore, "scores")
        if len({(s.producer, s.name) for s in self.scores}) != len(self.scores):
            raise EvidenceContractError("duplicate score from the same producer")

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence.to_dict(), "scores": to_primitive(self.scores)}


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    company_id: str
    entries: tuple[EvidenceEntry, ...]
    # This allowance is not serialized: do not trust one round-tripped in a client payload.
    allowed_sources: frozenset[SourceIdentity]
    limits: ManifestLimits

    def __post_init__(self) -> None:
        _text(self.company_id, "manifest company", nonblank=True)
        _tuple_of(self.entries, EvidenceEntry, "manifest entries")
        if type(self.allowed_sources) is not frozenset or any(
            not isinstance(s, SourceIdentity) for s in self.allowed_sources
        ):
            raise EvidenceContractError("explicit immutable source allowance required")
        if not isinstance(self.limits, ManifestLimits):
            raise EvidenceContractError("explicit manifest limits required")
        if any(s.scope.company_id != self.company_id for s in self.allowed_sources):
            raise EvidenceContractError("invalid tenant in source allowance")
        if len(self.entries) > self.limits.max_records:
            raise EvidenceContractError("manifest record budget exceeded")
        seen: set[str] = set()
        for entry in self.entries:
            record = entry.evidence
            if record.source.scope.company_id != self.company_id or record.source not in self.allowed_sources:
                raise EvidenceContractError("evidence not in the explicit source allowance")
            if record.evidence_id in seen:
                raise EvidenceContractError("duplicate evidence; no implicit merge or reranking")
            seen.add(record.evidence_id)
        if len(self.to_json().encode("utf-8")) > self.limits.max_serialized_bytes:
            raise EvidenceContractError("manifest serialized budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "company_id": self.company_id,
                "entries": [entry.to_dict() for entry in self.entries]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @property
    def manifest_id(self) -> str:
        return "mf1_" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def build_manifest(*, company_id: str, entries: Iterable[EvidenceEntry],
                   allowed_sources: frozenset[SourceIdentity],
                   limits: ManifestLimits) -> EvidenceManifest:
    """Preserve caller order; never silently discard, expand or change sources."""
    if not isinstance(limits, ManifestLimits):
        raise EvidenceContractError("explicit manifest limits required")
    if type(allowed_sources) is not frozenset:
        raise EvidenceContractError("trusted source allowance must be a frozenset")
    collected: list[EvidenceEntry] = []
    for entry in entries:
        if len(collected) >= limits.max_records:
            raise EvidenceContractError("manifest record budget exceeded")
        collected.append(entry)
    return EvidenceManifest(company_id, tuple(collected), allowed_sources, limits)
