"""P5-C: request-local compatibility snapshots for existing record layouts.

This is NOT a public response renderer, link resolver, cache format or client
payload deserializer. The canonical assembly excludes opaque legacy metadata.
An explicit restore returns a fresh copy of the original record, including old
transport fields/URLs: these are NOT validated links or evidence, and MUST NOT
be passed to a prompt or returned to a client by this module's caller.

Supported values are finite JSON scalars, mappings with string keys, lists and
(additionally) tuples. Unsupported Python objects are rejected, not stringified.
No pickle/eval, I/O, retrieval, ranking, text normalization or source inference.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable

from .adapter_types import AdapterContext, AdapterLimits, AdaptationResult
from .assembly import (AssemblyLimits, EvidenceAssembly, EvidenceAssemblyError,
                       _allowance, _int, assemble_evidence)
from .contracts import SourceIdentity
from .record_adapters import adapt_candidate, adapt_chunk, adapt_page
from .structured_adapters import adapt_structured_source

COMPATIBILITY_VERSION = "canonical-evidence-legacy-compatibility-v1"
LAYOUTS = frozenset({"document_page", "document_chunk", "retrieval_candidate",
                     "structured_source_snapshot"})


@dataclass(frozen=True, slots=True)
class LegacyLimits:
    """Aggregate sidecar bounds; max_bytes covers assembly plus all snapshots."""
    max_depth: int
    max_nodes: int
    max_bytes: int

    def __post_init__(self) -> None:
        _int(self.max_depth, "max_depth", 1)
        if self.max_depth > 64:
            raise EvidenceAssemblyError("max_depth must be <= 64")
        _int(self.max_nodes, "max_nodes", 1)
        _int(self.max_bytes, "max_bytes", 1)


@dataclass(frozen=True, slots=True)
class LegacyRecordInput:
    layout: str
    record: Mapping[str, Any] = field(repr=False)
    context: AdapterContext = field(repr=False)
    indexed_text: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.layout) is not str or self.layout not in LAYOUTS:
            raise EvidenceAssemblyError("unsupported explicit record layout")
        if not isinstance(self.record, Mapping) or not isinstance(self.context, AdapterContext):
            raise EvidenceAssemblyError("record and trusted provider binding required")
        if self.layout == "structured_source_snapshot":
            if type(self.indexed_text) is not str:
                raise EvidenceAssemblyError("explicit indexed snapshot required")
        elif self.indexed_text is not None:
            raise EvidenceAssemblyError("indexed_text is only for structured snapshots")


@dataclass(frozen=True, slots=True)
class LegacyRecordView:
    """A NEW mutable mapping, for an explicitly authorized internal old consumer."""
    layout: str
    record: dict[str, Any] = field(repr=False)
    indexed_text: str | None = field(default=None, repr=False)


# Immutable tagged tree; dict insertion order and tuple/list types are preserved.
@dataclass(frozen=True, slots=True)
class _Frozen:
    kind: str
    value: Any = field(repr=False)


def _wire(node: _Frozen) -> Any:
    if node.kind == "map":
        return ["map", [[k, _wire(v)] for k, v in node.value]]
    if node.kind in ("list", "tuple"):
        return [node.kind, [_wire(v) for v in node.value]]
    return [node.kind, node.value]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class _Freezer:
    def __init__(self, limits: LegacyLimits) -> None:
        self.limits = limits
        self.nodes = 0
        self.scalar_bytes = 0
        self.ancestors: set[int] = set()

    def freeze(self, value: Any, depth: int = 0) -> _Frozen:
        if depth > self.limits.max_depth:
            raise EvidenceAssemblyError("legacy depth budget exceeded")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise EvidenceAssemblyError("legacy node budget exceeded")
        typ = type(value)
        if value is None or typ in (str, bool, int, float):
            if typ is float and not math.isfinite(value):
                raise EvidenceAssemblyError("non-finite legacy value")
            kind = "null" if value is None else {str:"str", bool:"bool", int:"int", float:"float"}[typ]
            # A cheap lower bound precedes JSON escaping and the aggregate check.
            if typ is str and len(value) > self.limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            try:
                self.scalar_bytes += len(_json([kind, value]).encode("utf-8"))
            except (ValueError, OverflowError) as exc:
                raise EvidenceAssemblyError("unserializable legacy scalar") from exc
            if self.scalar_bytes > self.limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            return _Frozen(kind, value)
        if not isinstance(value, Mapping) and typ not in (list, tuple):
            raise EvidenceAssemblyError("unsupported legacy value; no implicit conversion")
        if id(value) in self.ancestors:
            raise EvidenceAssemblyError("cyclic legacy data")
        if len(value) > self.limits.max_nodes - self.nodes:
            raise EvidenceAssemblyError("legacy node budget exceeded")
        self.ancestors.add(id(value))
        try:
            if isinstance(value, Mapping):
                pairs = []
                for key, item in value.items():
                    if type(key) is not str:
                        raise EvidenceAssemblyError("legacy mapping keys must be strings")
                    self.freeze(key, depth + 1)  # Account for keys as well as values.
                    pairs.append((key, self.freeze(item, depth + 1)))
                return _Frozen("map", tuple(pairs))
            return _Frozen("list" if typ is list else "tuple",
                           tuple(self.freeze(item, depth + 1) for item in value))
        finally:
            self.ancestors.remove(id(value))


def _thaw(node: _Frozen) -> Any:
    if node.kind == "map":
        return {k: _thaw(v) for k, v in node.value}
    if node.kind == "list":
        return [_thaw(v) for v in node.value]
    if node.kind == "tuple":
        return tuple(_thaw(v) for v in node.value)
    return node.value


@dataclass(frozen=True, slots=True)
class _Snapshot:
    layout: str
    source: SourceIdentity
    evidence_id: str
    payload: _Frozen = field(repr=False)
    digest: str


def _digest(payload: _Frozen) -> str:
    return hashlib.sha256(_json(_wire(payload)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LegacyEvidenceBundle:
    """Canonical assembly + private, immutable, request-local sidecars.

    No serialization of sidecars or of permissions is supplied. Building this
    object does not authenticate a caller. Never accept it from a client.
    """
    assembly: EvidenceAssembly
    _snapshots: tuple[_Snapshot, ...] = field(repr=False)
    limits: LegacyLimits
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, EvidenceAssembly) or not isinstance(self.limits, LegacyLimits):
            raise EvidenceAssemblyError("invalid compatibility bundle")
        if type(self._snapshots) is not tuple or len(self._snapshots) != len(self.assembly.occurrences):
            raise EvidenceAssemblyError("one snapshot per input occurrence required")
        entries = self.assembly.occurrence_entries()
        for snapshot, entry in zip(self._snapshots, entries):
            if not isinstance(snapshot, _Snapshot) or snapshot.layout not in LAYOUTS:
                raise EvidenceAssemblyError("invalid internal legacy snapshot")
            if snapshot.source != entry.evidence.source or snapshot.evidence_id != entry.evidence.evidence_id:
                raise EvidenceAssemblyError("snapshot/canonical binding mismatch")
            if _digest(snapshot.payload) != snapshot.digest:
                raise EvidenceAssemblyError("snapshot integrity mismatch")
        actual = len(self.assembly.to_json().encode("utf-8")) + sum(
            len(_json(_wire(s.payload)).encode("utf-8")) for s in self._snapshots)
        _int(self.size_bytes, "bundle size")
        if actual != self.size_bytes or actual > self.limits.max_bytes:
            raise EvidenceAssemblyError("compatibility byte budget exceeded or invalid size")


def _adapt(layout: str, raw: dict[str, Any], *, indexed_text: str | None,
           context: AdapterContext, limits: AdapterLimits) -> AdaptationResult:
    if layout == "document_page":
        return adapt_page(raw, context=context, limits=limits)
    if layout == "document_chunk":
        return adapt_chunk(raw, context=context, limits=limits)
    if layout == "retrieval_candidate":
        return adapt_candidate(raw, context=context, limits=limits)
    if layout == "structured_source_snapshot":
        return adapt_structured_source(raw, indexed_text=indexed_text, context=context, limits=limits)
    raise EvidenceAssemblyError("unsupported explicit record layout")


def build_legacy_bundle(records: Iterable[LegacyRecordInput], *, company_id: str,
                        allowed_sources: frozenset[SourceIdentity],
                        adapter_limits: AdapterLimits, assembly_limits: AssemblyLimits,
                        legacy_limits: LegacyLimits) -> LegacyEvidenceBundle:
    """Convert snapshots atomically. Do not infer authorization from contexts.

    Raw unconsumed and transport fields stay ONLY in private compatibility data.
    Assembly traces name them explicitly. Their values never become evidence.
    """
    _allowance(company_id, allowed_sources)
    if not isinstance(adapter_limits, AdapterLimits) or not isinstance(assembly_limits, AssemblyLimits):
        raise EvidenceAssemblyError("explicit adapter and assembly limits required")
    if not isinstance(legacy_limits, LegacyLimits):
        raise EvidenceAssemblyError("explicit legacy bounds required")
    freezer = _Freezer(legacy_limits)
    results, snapshots = [], []
    snapshot_bytes = 0
    for item in records:
        if len(results) >= assembly_limits.max_occurrences:
            raise EvidenceAssemblyError("occurrence budget exceeded")
        if not isinstance(item, LegacyRecordInput):
            raise EvidenceAssemblyError("explicit legacy record input required")
        if item.context.source not in allowed_sources or any(
            r.target not in allowed_sources for r in item.context.relations
        ):
            raise EvidenceAssemblyError("provider binding outside assembly allowance")
        frozen = freezer.freeze(item.record)
        snapshot_bytes += len(_json(_wire(frozen)).encode("utf-8"))
        if snapshot_bytes > legacy_limits.max_bytes:
            raise EvidenceAssemblyError("legacy byte budget exceeded")
        raw = _thaw(frozen)
        result = _adapt(item.layout, raw, indexed_text=item.indexed_text,
                        context=item.context, limits=adapter_limits)
        results.append(result)
        snapshots.append(_Snapshot(item.layout, item.context.source, result.entry.evidence.evidence_id,
                                   frozen, _digest(frozen)))
    assembly = assemble_evidence(results, company_id=company_id, allowed_sources=allowed_sources,
                                 limits=assembly_limits)
    size = snapshot_bytes + len(assembly.to_json().encode("utf-8"))
    return LegacyEvidenceBundle(assembly, tuple(snapshots), legacy_limits, size)


def restore_legacy_records(bundle: LegacyEvidenceBundle, *, company_id: str,
                           allowed_sources: frozenset[SourceIdentity],
                           adapter_limits: AdapterLimits) -> tuple[LegacyRecordView, ...]:
    """Explicit internal restore with FRESH provider allowance and fresh copies.

    Re-adaptation validates correspondence to canonical records, scores and traces.
    The original field names, values, order, duplicates and types are preserved;
    missing fields stay missing. Signed URLs in raw transport stay unverified.
    """
    _allowance(company_id, allowed_sources)
    if not isinstance(bundle, LegacyEvidenceBundle) or not isinstance(adapter_limits, AdapterLimits):
        raise EvidenceAssemblyError("bundle and explicit adapter limits required")
    if bundle.assembly.manifest.company_id != company_id:
        raise EvidenceAssemblyError("restore company differs from assembly")
    output = []
    for snapshot, occurrence, entry in zip(bundle._snapshots, bundle.assembly.occurrences,
                                            bundle.assembly.occurrence_entries()):
        evidence = entry.evidence
        if evidence.source not in allowed_sources or any(r.target not in allowed_sources for r in evidence.relations):
            raise EvidenceAssemblyError("restore outside current provider allowance")
        raw = _thaw(snapshot.payload)
        indexed = evidence.text if snapshot.layout == "structured_source_snapshot" else None
        context = AdapterContext(evidence.source, evidence.provenance, allowed_sources,
                                 evidence.link_target, evidence.relations)
        result = _adapt(snapshot.layout, raw, indexed_text=indexed, context=context, limits=adapter_limits)
        if result.entry != entry or result.trace != occurrence.trace:
            raise EvidenceAssemblyError("legacy/canonical conversion mismatch")
        output.append(LegacyRecordView(snapshot.layout, raw, indexed))
    return tuple(output)
