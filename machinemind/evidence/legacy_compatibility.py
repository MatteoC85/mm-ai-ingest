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
from array import array
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable

from .adapter_types import AdapterContext, AdapterLimits, AdaptationResult
from .assembly import (AssemblyLimits, EvidenceAssembly, EvidenceAssemblyError,
                       _allowance, _int, assemble_evidence)
from .contracts import SourceIdentity, _PackedFloats
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


class _PayloadPool:
    """Request-owned interning of immutable numeric VALUES, never permissions.

    Every lookup checks the actual incoming component types and IEEE-754 bytes.
    No source ID, hash-only provenance join, mutable object-ID cache, dimension,
    language or model-specific rule is involved. The index is bounded; overflow
    merely uses an unshared immutable value, never drops or truncates evidence.
    """
    def __init__(self, max_bytes: int):
        _int(max_bytes, "payload pool bound", 1)
        self.max_bytes = max_bytes
        self._lock = RLock()
        self._values = {}
        self.retained_bytes = 0
        self.hits = 0
        self.misses = 0
        self.closed = False

    def pack(self, values):
        with self._lock:
            if self.closed:
                raise EvidenceAssemblyError("legacy payload pool is closed")
            # Called only after the freezer's exact homogeneous-float type check.
            key = array("d", values).tobytes()
            found = self._values.get(key)
            if found is not None:
                self.hits += 1
                return found
            packed = _PackedFloats.from_values(values)
            self.misses += 1
            # Includes every retained encoded representation, not just the key.
            size = _numeric_retained_bytes(packed)
            if self.retained_bytes + size <= self.max_bytes:
                self._values[packed.binary] = packed
                self.retained_bytes += size
            return packed

    def summary(self):
        with self._lock:
            return {"unique_numeric_payloads": len(self._values),
                    "numeric_payload_pool_bytes": self.retained_bytes,
                    "numeric_payload_pool_hits": self.hits,
                    "numeric_payload_pool_misses": self.misses}

    def close(self):
        with self._lock:
            self._values.clear()
            self.retained_bytes = 0
            self.closed = True


# Context-local ONLY while the owning ASK invocation is executing. This carries
# no source, principal, grant or authority result and defaults to no sharing.
_CURRENT_PAYLOAD_POOL = ContextVar("mm_ask_immutable_payload_pool", default=None)


@contextmanager
def _payload_scope(pool):
    if type(pool) is not _PayloadPool or pool.closed:
        raise EvidenceAssemblyError("live request-owned payload pool required")
    if _CURRENT_PAYLOAD_POOL.get() is pool:
        yield
        return
    token = _CURRENT_PAYLOAD_POOL.set(pool)
    try:
        yield
    finally:
        _CURRENT_PAYLOAD_POOL.reset(token)


def _pack_floats(values):
    pool = _CURRENT_PAYLOAD_POOL.get()
    return pool.pack(values) if pool is not None else _PackedFloats.from_values(values)


# Immutable tagged tree; dict insertion order and tuple/list types are preserved.
@dataclass(frozen=True, slots=True)
class _Frozen:
    kind: str
    value: Any = field(repr=False)


def _wire(node: _Frozen) -> Any:
    if node.kind == "map":
        return ["map", [[k, _wire(v)] for k, v in node.value]]
    if node.kind in ("list", "tuple"):
        if type(node.value) is _PackedFloats:
            return [node.kind, [["float", x] for x in node.value.values]]
        return [node.kind, [_wire(v) for v in node.value]]
    return [node.kind, node.value]


def _wire_json(node: _Frozen) -> str:
    """Exact former _json(_wire(node)), without building scalar wrapper trees."""
    if node.kind == "map":
        return '["map",[' + ','.join('[' + _json(k) + ',' + _wire_json(v) + ']'
                                      for k, v in node.value) + ']]'
    if node.kind in ("list", "tuple"):
        children = (node.value.wire_children() if type(node.value) is _PackedFloats
                    else '[' + ','.join(_wire_json(v) for v in node.value) + ']')
        return '[' + _json(node.kind) + ',' + children + ']'
    return _json([node.kind, node.value])


# Stateless encoder: reuses only configuration, never content, grants or IDs.
_JSON_ENCODER = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _json(value: Any) -> str:
    return _JSON_ENCODER.encode(value)


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
                # ensure_ascii guarantees ASCII; same wire length as [kind,value].
                self.scalar_bytes += len(_json(value)) + len(kind) + 5
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
        if typ in (list, tuple) and value and set(map(type, value)) == {float}:
            if depth + 1 > self.limits.max_depth:
                raise EvidenceAssemblyError("legacy depth budget exceeded")
            try:
                packed = _pack_floats(value)
            except (ValueError, OverflowError) as exc:
                raise EvidenceAssemblyError("non-finite legacy value") from exc
            self.nodes += len(value)
            self.scalar_bytes += packed.scalar_bytes
            if self.scalar_bytes > self.limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            return _Frozen("list" if typ is list else "tuple", packed)
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


def _frozen_canonical_size(node: _Frozen) -> int:
    """Exact ASCII byte count of canonical_json(node), without a JSON object tree.

    Only accepts the private tree produced by _Freezer. Bounds, cycles and
    non-finite values have already been checked by that traversal. This is NOT
    the compatibility wire format (_wire), and it changes no allocation limit.
    No snapshot data is cached; scalar escaping uses the same JSON encoder.
    """
    if node.kind == "map":
        return (len('{"kind":"map","value":[]}')
                + max(0, len(node.value) - 1)
                + sum(len(_json(k)) + _frozen_canonical_size(v) + 3
                      for k, v in node.value))
    if node.kind in ("list", "tuple"):
        if type(node.value) is _PackedFloats:
            return len('{"kind":"' + node.kind + '","value":[]}') + node.value.canonical_children_bytes()
        return (len('{"kind":"' + node.kind + '","value":[]}')
                + max(0, len(node.value) - 1)
                + sum(_frozen_canonical_size(v) for v in node.value))
    return len('{"kind":"' + node.kind + '","value":}') + len(_json(node.value))


def _legacy_canonical_size(value: Any, limits: LegacyLimits) -> int:
    """Validate/count exactly as freeze + frozen_canonical_size, without a tree.

    Used only where the temporary frozen object was discarded after counting.
    Limits, scalar escaping, mapping order and cycle rejection are unchanged.
    Real retained snapshots still use _Freezer and retain their immutable tree.
    """
    nodes = 0
    scalar_bytes = 0
    ancestors = set()

    def measure(item: Any, depth: int) -> int:
        nonlocal nodes, scalar_bytes
        if depth > limits.max_depth:
            raise EvidenceAssemblyError("legacy depth budget exceeded")
        nodes += 1
        if nodes > limits.max_nodes:
            raise EvidenceAssemblyError("legacy node budget exceeded")
        typ = type(item)
        if item is None or typ in (str, bool, int, float):
            if typ is float and not math.isfinite(item):
                raise EvidenceAssemblyError("non-finite legacy value")
            kind = "null" if item is None else {str:"str", bool:"bool", int:"int", float:"float"}[typ]
            if typ is str and len(item) > limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            try:
                encoded_length = len(_json(item))
                scalar_bytes += encoded_length + len(kind) + 5
            except (ValueError, OverflowError) as exc:
                raise EvidenceAssemblyError("unserializable legacy scalar") from exc
            if scalar_bytes > limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            return len('{"kind":"' + kind + '","value":}') + encoded_length
        if not isinstance(item, Mapping) and typ not in (list, tuple):
            raise EvidenceAssemblyError("unsupported legacy value; no implicit conversion")
        if id(item) in ancestors:
            raise EvidenceAssemblyError("cyclic legacy data")
        if len(item) > limits.max_nodes - nodes:
            raise EvidenceAssemblyError("legacy node budget exceeded")
        if typ in (list, tuple) and item and set(map(type, item)) == {float}:
            if depth + 1 > limits.max_depth:
                raise EvidenceAssemblyError("legacy depth budget exceeded")
            try:
                packed = _pack_floats(item)
            except (ValueError, OverflowError) as exc:
                raise EvidenceAssemblyError("non-finite legacy value") from exc
            nodes += len(item)
            scalar_bytes += packed.scalar_bytes
            if scalar_bytes > limits.max_bytes:
                raise EvidenceAssemblyError("legacy byte budget exceeded")
            kind = "list" if typ is list else "tuple"
            return len('{"kind":"' + kind + '","value":[]}') + packed.canonical_children_bytes()
        ancestors.add(id(item))
        try:
            if isinstance(item, Mapping):
                size = len('{"kind":"map","value":[]}')
                count = 0
                for key, child in item.items():
                    if type(key) is not str:
                        raise EvidenceAssemblyError("legacy mapping keys must be strings")
                    key_size = measure(key, depth + 1)
                    # Map keys occupy their escaped string, NOT a tagged node.
                    key_length = key_size - len('{"kind":"str","value":}')
                    size += key_length + measure(child, depth + 1) + 3
                    count += 1
                return size + max(0, count - 1)
            kind = "list" if typ is list else "tuple"
            size = len('{"kind":"' + kind + '","value":[]}')
            count = 0
            for child in item:
                size += measure(child, depth + 1)
                count += 1
            return size + max(0, count - 1)
        finally:
            ancestors.remove(id(item))

    return measure(value, 0)


def _thaw(node: _Frozen) -> Any:
    if node.kind == "map":
        return {k: _thaw(v) for k, v in node.value}
    if node.kind == "list":
        if type(node.value) is _PackedFloats:
            return list(node.value.values)
        return [_thaw(v) for v in node.value]
    if node.kind == "tuple":
        if type(node.value) is _PackedFloats:
            return tuple(node.value.values)
        return tuple(_thaw(v) for v in node.value)
    return node.value


def _matches_frozen(value: Any, node: _Frozen) -> bool:
    """Exact mutable-input comparison to a private immutable witness.

    Component values are checked again at EVERY use. Cached serialization never
    stands in for checking a mutable list, including type changes and -0.0.
    """
    typ = type(value)
    if node.kind == "map":
        return (typ is dict and tuple(value) == tuple(k for k, _ in node.value)
                and all(_matches_frozen(v, n) for v, (_, n) in zip(value.values(), node.value)))
    if node.kind in ("list", "tuple"):
        if typ is not (list if node.kind == "list" else tuple):
            return False
        if type(node.value) is _PackedFloats:
            return (len(value) == len(node.value.values)
                    and set(map(type, value)) == {float}
                    and array("d", value).tobytes() == node.value.binary)
        return len(value) == len(node.value) and all(
            _matches_frozen(v, n) for v, n in zip(value, node.value))
    expected = {"null": type(None), "str": str, "bool": bool, "int": int, "float": float}[node.kind]
    if typ is not expected:
        return False
    if typ is float:
        return value is node.value or value.hex() == node.value.hex()
    return value is node.value or value == node.value


@dataclass(frozen=True, slots=True)
class _LegacyWitness(Mapping):
    """Private immutable journal snapshot; materialize only at a real boundary.

    The Mapping interface returns fresh values for existing metadata adapters.
    It is NEVER handed to a legacy collaborator or to the public ASK port.
    """
    _tree: _Frozen = field(repr=False)
    size_bytes: int
    retained_metadata_bytes: int = field(init=False, repr=False)
    numeric_payloads: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        size, payloads = _retained_witness_profile(self._tree)
        object.__setattr__(self, "retained_metadata_bytes", size)
        object.__setattr__(self, "numeric_payloads", payloads)

    @classmethod
    def capture(cls, value, limits):
        tree = _Freezer(limits).freeze(value)
        return cls(tree, _frozen_canonical_size(tree))

    def matches(self, value):
        return _matches_frozen(value, self._tree)

    def detached(self):
        return _thaw(self._tree)

    def __iter__(self):
        if self._tree.kind != "map":
            raise TypeError("mapping witness required")
        return (k for k, _ in self._tree.value)

    def __len__(self):
        return len(self._tree.value)

    def __getitem__(self, key):
        if self._tree.kind != "map":
            raise TypeError("mapping witness required")
        for k, node in self._tree.value:
            if k == key:
                return _thaw(node)
        raise KeyError(key)

    def __deepcopy__(self, memo):
        return self  # Only immutable primitives/tree nodes are retained.



def _numeric_retained_bytes(payload):
    # Actual immutable numeric buffers held by _PackedFloats. The numeric tuple
    # has the same binary payload, not an additional serialized JSON document.
    # Python object/header/allocator overhead is measured separately by RSS.
    return (len(payload.binary) + len(payload.encoded) + len(payload.wire_children())
            + len(payload.reference_id))


def _retained_witness_profile(root):
    """Internal reference-form footprint, NOT a change to compatibility wire JSON.

    A journal witness really retains references to shared _PackedFloats objects.
    Charge each occurrence's reference metadata, then each OWNED numeric buffer
    once in _WitnessBudget. Canonical round-trip/digest/type/node/depth checks
    still inspect the original full payload; no component is omitted.
    """
    payloads = {}
    def size(node):
        if node.kind == "map":
            return (len('{"kind":"map","value":[]}') + max(0, len(node.value)-1)
                    + sum(len(_json(k)) + size(v) + 3 for k,v in node.value))
        if node.kind in ("list", "tuple"):
            if type(node.value) is _PackedFloats:
                payloads[id(node.value)] = node.value
                return len(_json({"kind":node.kind,"value":{
                    "numeric_payload_ref":node.value.reference_id,
                    "components":len(node.value.values)}}))
            return (len('{"kind":"'+node.kind+'","value":[]}')
                    + max(0,len(node.value)-1) + sum(size(v) for v in node.value))
        return len('{"kind":"'+node.kind+'","value":}') + len(_json(node.value))
    return size(root), tuple(payloads.values())


class _WitnessBudget:
    """Bounded live journal accounting for the representation actually retained.

    Deduplication is by strong OBJECT reference to privately immutable buffers,
    never by source ID, string equality, checksum-only permission or allowance.
    Full legacy snapshots supplied by old internal callers remain fully charged.
    Every mutable record is still checked against its witness on every entry().
    """
    def __init__(self, limits):
        self.limits = limits
        self.metadata_bytes = 0
        self.numeric_bytes = 0
        self.logical_bytes = 0
        self.payloads = {}

    @property
    def size(self):
        return self.metadata_bytes + self.numeric_bytes

    def _parts(self, witness):
        if type(witness) is _LegacyWitness:
            return witness.retained_metadata_bytes, witness.numeric_payloads, witness.size_bytes
        full = _legacy_canonical_size(witness, self.limits)
        return full, (), full

    def delta(self, witness):
        metadata, payloads, _ = self._parts(witness)
        return metadata + sum(_numeric_retained_bytes(p) for p in payloads
                              if id(p) not in self.payloads)

    def add(self, witness):
        metadata, payloads, logical = self._parts(witness)
        self.metadata_bytes += metadata
        self.logical_bytes += logical
        for p in payloads:
            old = self.payloads.get(id(p))
            if old is None:
                self.payloads[id(p)] = [p, 1]
                self.numeric_bytes += _numeric_retained_bytes(p)
            else:
                if old[0] is not p:
                    raise EvidenceAssemblyError("numeric witness identity changed")
                old[1] += 1

    def remove(self, witness):
        metadata, payloads, logical = self._parts(witness)
        if self.metadata_bytes < metadata or self.logical_bytes < logical:
            raise EvidenceAssemblyError("witness budget underflow")
        for p in payloads:
            old = self.payloads.get(id(p))
            if old is None or old[0] is not p or old[1] < 1:
                raise EvidenceAssemblyError("unowned numeric witness")
        self.metadata_bytes -= metadata
        self.logical_bytes -= logical
        for p in payloads:
            old = self.payloads[id(p)]
            old[1] -= 1
            if not old[1]:
                self.numeric_bytes -= _numeric_retained_bytes(p)
                del self.payloads[id(p)]

    @classmethod
    def from_witnesses(cls, witnesses, limits):
        budget = cls(limits)
        for witness in witnesses:
            budget.add(witness)
        return budget

    def clear(self):
        self.payloads.clear()
        self.metadata_bytes = self.numeric_bytes = self.logical_bytes = 0


def _witness_size(value, limits):
    return value.size_bytes if type(value) is _LegacyWitness else _legacy_canonical_size(value, limits)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    layout: str
    source: SourceIdentity
    evidence_id: str
    payload: _Frozen = field(repr=False)
    digest: str


def _digest(payload: _Frozen) -> str:
    return hashlib.sha256(_wire_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ConversionProof:
    """Private proof of a PURE conversion, never a stored permission grant.

    Issued only by build_legacy_bundle for the exact immutable assembly and
    snapshot tuple it constructed. Substitution of either invalidates reuse.
    """
    assembly: EvidenceAssembly = field(repr=False)
    snapshots: tuple[_Snapshot, ...] = field(repr=False)
    adapter_limits: AdapterLimits
    entries: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "entries", self.assembly.occurrence_entries())


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
    _conversion_proof: _ConversionProof | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, EvidenceAssembly) or not isinstance(self.limits, LegacyLimits):
            raise EvidenceAssemblyError("invalid compatibility bundle")
        if type(self._snapshots) is not tuple or len(self._snapshots) != len(self.assembly.occurrences):
            raise EvidenceAssemblyError("one snapshot per input occurrence required")
        entries = self.assembly.occurrence_entries()
        snapshot_bytes = 0
        for snapshot, entry in zip(self._snapshots, entries):
            if not isinstance(snapshot, _Snapshot) or snapshot.layout not in LAYOUTS:
                raise EvidenceAssemblyError("invalid internal legacy snapshot")
            if snapshot.source != entry.evidence.source or snapshot.evidence_id != entry.evidence.evidence_id:
                raise EvidenceAssemblyError("snapshot/canonical binding mismatch")
            serialized = _wire_json(snapshot.payload).encode("utf-8")
            if hashlib.sha256(serialized).hexdigest() != snapshot.digest:
                raise EvidenceAssemblyError("snapshot integrity mismatch")
            snapshot_bytes += len(serialized)
        actual = len(self.assembly.to_json().encode("utf-8")) + snapshot_bytes
        _int(self.size_bytes, "bundle size")
        if actual != self.size_bytes or actual > self.limits.max_bytes:
            raise EvidenceAssemblyError("compatibility byte budget exceeded or invalid size")


def _has_validated_conversion(bundle: LegacyEvidenceBundle,
                              adapter_limits: AdapterLimits) -> bool:
    """Check binding of the creation proof; NEVER infer current authorization."""
    if type(bundle) is not LegacyEvidenceBundle or type(adapter_limits) is not AdapterLimits:
        return False
    proof = bundle._conversion_proof
    return (type(proof) is _ConversionProof and proof.assembly is bundle.assembly
            and proof.snapshots is bundle._snapshots
            and proof.adapter_limits == adapter_limits)


def _occurrence_entries(bundle):
    proof = bundle._conversion_proof
    if (type(proof) is _ConversionProof and proof.assembly is bundle.assembly
            and proof.snapshots is bundle._snapshots):
        return proof.entries
    return bundle.assembly.occurrence_entries()


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
        serialized = _wire_json(frozen).encode("utf-8")
        snapshot_bytes += len(serialized)
        if snapshot_bytes > legacy_limits.max_bytes:
            raise EvidenceAssemblyError("legacy byte budget exceeded")
        raw = _thaw(frozen)
        result = _adapt(item.layout, raw, indexed_text=item.indexed_text,
                        context=item.context, limits=adapter_limits)
        results.append(result)
        snapshots.append(_Snapshot(item.layout, item.context.source, result.entry.evidence.evidence_id,
                                   frozen, hashlib.sha256(serialized).hexdigest()))
    assembly = assemble_evidence(results, company_id=company_id, allowed_sources=allowed_sources,
                                 limits=assembly_limits)
    size = snapshot_bytes + len(assembly.to_json().encode("utf-8"))
    snapshots = tuple(snapshots)
    # Both the adaptation and assembly were just computed from these immutable
    # snapshots. The constructor still performs all original integrity checks.
    proof = _ConversionProof(assembly, snapshots, adapter_limits)
    return LegacyEvidenceBundle(assembly, snapshots, legacy_limits, size, proof)


def restore_legacy_records(bundle: LegacyEvidenceBundle, *, company_id: str,
                           allowed_sources: frozenset[SourceIdentity],
                           adapter_limits: AdapterLimits,
                           positions: tuple[int, ...] | None = None) -> tuple[LegacyRecordView, ...]:
    """Explicit internal restore with FRESH provider allowance and fresh copies.

    Reuse the pure conversion only for the exact privately validated immutable
    graph and unchanged adapter limits; otherwise re-adapt against records,
    scores and traces. CURRENT source/relation grants are checked at every use.
    The original field names, values, order, duplicates and types are preserved;
    missing fields stay missing. Signed URLs in raw transport stay unverified.
    """
    _allowance(company_id, allowed_sources)
    if not isinstance(bundle, LegacyEvidenceBundle) or not isinstance(adapter_limits, AdapterLimits):
        raise EvidenceAssemblyError("bundle and explicit adapter limits required")
    if bundle.assembly.manifest.company_id != company_id:
        raise EvidenceAssemblyError("restore company differs from assembly")
    conversion_valid = _has_validated_conversion(bundle, adapter_limits)
    assembly = bundle.assembly
    entries = _occurrence_entries(bundle)
    if positions is None:
        positions = tuple(range(len(bundle._snapshots)))
    elif (type(positions) is not tuple or len(positions) > assembly.limits.max_occurrences
          or any(type(p) is not int or not 0 <= p < len(bundle._snapshots) for p in positions)):
        raise EvidenceAssemblyError("bounded explicit legacy occurrence positions required")
    output = []
    for position in positions:
        snapshot = bundle._snapshots[position]
        occurrence = assembly.occurrences[position]
        entry = entries[position]
        evidence = entry.evidence
        if evidence.source not in allowed_sources or any(r.target not in allowed_sources for r in evidence.relations):
            raise EvidenceAssemblyError("restore outside current provider allowance")
        raw = _thaw(snapshot.payload)
        indexed = evidence.text if snapshot.layout == "structured_source_snapshot" else None
        if not conversion_valid:
            # Hand-built/replaced bundles and changed adapter limits retain the
            # original full re-adaptation. Current authorization above is ALWAYS
            # checked, also for bundles whose pure conversion can be reused.
            context = AdapterContext(evidence.source, evidence.provenance, allowed_sources,
                                     evidence.link_target, evidence.relations)
            result = _adapt(snapshot.layout, raw, indexed_text=indexed, context=context, limits=adapter_limits)
            if result.entry != entry or result.trace != occurrence.trace:
                raise EvidenceAssemblyError("legacy/canonical conversion mismatch")
        output.append(LegacyRecordView(snapshot.layout, raw, indexed))
    return tuple(output)
