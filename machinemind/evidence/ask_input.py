"""P6-A: explicit, request-local input port from canonical evidence to ASK.

This module is NOT an authorization provider, retriever, prompt builder or cache.
The application must supply already-authorized LegacyRecordInput bindings and a
current allowance independently of retrieved titles, scores or user text. It must
not reconstruct permissions from this object. Nothing here interprets language.

The optional core hook is deliberately unconfigured in this delivery. A later
change must bind the live provider and cover repair/rescue/cache before enabling
it. With a configured hook, this first compatibility port passes only losslessly
restored candidate collections to the existing ASK consumer. It never silently
falls back to unvalidated raw records on conversion or permission failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from .adapter_types import AdapterLimits
from .assembly import AssemblyLimits, EvidenceAssemblyError
from .contracts import SourceIdentity
from .legacy_compatibility import (
    LegacyEvidenceBundle, LegacyLimits, LegacyRecordInput,
    build_legacy_bundle, restore_legacy_records,
)

ASK_INPUT_VERSION = "canonical-ask-input-port-v1"
COLLECTIONS = frozenset({"candidates", "citations"})


class AskEvidenceInputError(EvidenceAssemblyError):
    """Technical input-contract failure, NOT evidence absence or a diagnosis."""


@dataclass(frozen=True, slots=True)
class AskRequestKey:
    """Exact request association; NOT an ACL or a proof of record ownership.

    Document order/duplicates and the original query are retained. Machine-level
    vs explicit-document permissions remain the provider's responsibility.
    """
    query: str = field(repr=False)
    company_id: str = field(repr=False)
    machine_id: str = field(repr=False)
    ai_scope: str
    response_language: str
    top_k: int
    narrow_scope: bool
    document_ids: tuple[str, ...] = field(repr=False)
    bubble_document_id: str | None = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("query", "company_id", "ai_scope", "response_language"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise AskEvidenceInputError("invalid request field: " + name)
        if type(self.machine_id) is not str:
            raise AskEvidenceInputError("machine_id must be text, possibly empty")
        if type(self.top_k) is not int or self.top_k < 1 or type(self.narrow_scope) is not bool:
            raise AskEvidenceInputError("invalid request selection bounds")
        if type(self.document_ids) is not tuple or any(type(v) is not str or not v.strip() for v in self.document_ids):
            raise AskEvidenceInputError("explicit immutable document identifiers required")
        if self.bubble_document_id is not None and type(self.bubble_document_id) is not str:
            raise AskEvidenceInputError("invalid single-document selector")


def ask_request_key(request: Any) -> AskRequestKey:
    """Read the current internal AssistantCoreRequest without importing the core.

    This is intentionally not a client deserializer. Missing/invalid request
    attributes fail instead of inventing scope. Other request metadata stay with
    the existing pipeline and are not serialized into the canonical manifest.
    """
    if getattr(request, "requested_mode", None) != "ask":
        raise AskEvidenceInputError("only an explicitly requested ASK may use this port")
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise AskEvidenceInputError("request metadata mapping required")
    doc_ids = metadata.get("document_ids")
    if doc_ids is None:
        doc_ids = ()
    if type(doc_ids) not in (list, tuple):
        raise AskEvidenceInputError("document_ids must be a list or tuple")
    try:
        return AskRequestKey(request.query, request.company_id, request.machine_id,
                             request.ai_scope, request.response_language, request.top_k,
                             request.narrow_scope, tuple(doc_ids), metadata.get("bubble_document_id"))
    except AttributeError as exc:
        raise AskEvidenceInputError("missing internal request binding") from exc


@dataclass(frozen=True, slots=True)
class AskCollectionInput:
    """One explicit candidate collection already bound by the authorized provider."""
    name: str
    records: tuple[LegacyRecordInput, ...] = field(repr=False)
    container_kind: str = "list"

    def __post_init__(self) -> None:
        if self.name not in COLLECTIONS or self.container_kind not in {"list", "tuple"}:
            raise AskEvidenceInputError("unsupported ASK collection or container")
        if type(self.records) is not tuple or any(not isinstance(v, LegacyRecordInput) for v in self.records):
            raise AskEvidenceInputError("explicit immutable provider-bound records required")
        if any(v.layout != "retrieval_candidate" for v in self.records):
            raise AskEvidenceInputError("ASK port requires retrieved candidates, not inferred layouts")


@dataclass(frozen=True, slots=True)
class AskCollectionSpan:
    name: str
    start: int
    stop: int
    container_kind: str

    def __post_init__(self) -> None:
        if self.name not in COLLECTIONS or self.container_kind not in {"list", "tuple"}:
            raise AskEvidenceInputError("invalid ASK collection span")
        if type(self.start) is not int or type(self.stop) is not int or self.start < 0 or self.stop < self.start:
            raise AskEvidenceInputError("invalid ASK occurrence interval")


@dataclass(frozen=True, slots=True)
class AskEvidenceInput:
    """One aggregate canonical bundle, preserving both collection boundaries.

    There is one shared memory/occurrence budget across collections, not a new
    budget per collection. The original legacy data remain private to the bundle.
    No public-response or authorization serializer is provided.
    """
    request_key: AskRequestKey = field(repr=False)
    bundle: LegacyEvidenceBundle = field(repr=False)
    collections: tuple[AskCollectionSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, AskRequestKey) or not isinstance(self.bundle, LegacyEvidenceBundle):
            raise AskEvidenceInputError("request key and canonical bundle required")
        if type(self.collections) is not tuple or len(self.collections) > len(COLLECTIONS):
            raise AskEvidenceInputError("invalid ASK collection bounds")
        seen, position = set(), 0
        for span in self.collections:
            if not isinstance(span, AskCollectionSpan) or span.name in seen or span.start != position:
                raise AskEvidenceInputError("duplicate or noncontiguous ASK collection")
            seen.add(span.name)
            position = span.stop
        if position != len(self.bundle.assembly.occurrences):
            raise AskEvidenceInputError("unbound or missing evidence occurrences")
        if self.bundle.assembly.manifest.company_id != self.request_key.company_id:
            raise AskEvidenceInputError("bundle tenant differs from ASK request")


def build_ask_evidence_input(*, request_key: AskRequestKey,
                             collections: tuple[AskCollectionInput, ...],
                             allowed_sources: frozenset[SourceIdentity],
                             adapter_limits: AdapterLimits,
                             assembly_limits: AssemblyLimits,
                             legacy_limits: LegacyLimits) -> AskEvidenceInput:
    """Build atomically from provider bindings; no source lookup or auto-grant."""
    if not isinstance(request_key, AskRequestKey):
        raise AskEvidenceInputError("explicit request key required")
    if type(collections) is not tuple or len(collections) > len(COLLECTIONS):
        raise AskEvidenceInputError("bounded immutable collections required")
    if not isinstance(assembly_limits, AssemblyLimits):
        raise AskEvidenceInputError("aggregate assembly limits required")
    spans, records, seen = [], [], set()
    for item in collections:
        if not isinstance(item, AskCollectionInput) or item.name in seen:
            raise AskEvidenceInputError("duplicate or invalid ASK input collection")
        seen.add(item.name)
        if len(records) + len(item.records) > assembly_limits.max_occurrences:
            raise AskEvidenceInputError("aggregate ASK occurrence limit exceeded")
        start = len(records)
        records.extend(item.records)
        spans.append(AskCollectionSpan(item.name, start, len(records), item.container_kind))
    bundle = build_legacy_bundle(records, company_id=request_key.company_id,
                                 allowed_sources=allowed_sources, adapter_limits=adapter_limits,
                                 assembly_limits=assembly_limits, legacy_limits=legacy_limits)
    return AskEvidenceInput(request_key, bundle, tuple(spans))


@dataclass(frozen=True, slots=True)
class AskEvidenceAdmission:
    """Internal hook result with the CURRENT allowance from the trusted provider.

    Do not deserialize from clients/cache. Do not fill current_allowed_sources
    from bundle.manifest.allowed_sources; authorization must precede conversion.
    """
    evidence_input: AskEvidenceInput = field(repr=False)
    current_allowed_sources: frozenset[SourceIdentity] = field(repr=False)
    adapter_limits: AdapterLimits

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_input, AskEvidenceInput) or not isinstance(self.adapter_limits, AdapterLimits):
            raise AskEvidenceInputError("typed ASK input and adapter limits required")
        if type(self.current_allowed_sources) is not frozenset or any(
            not isinstance(s, SourceIdentity) or s.scope.company_id != self.evidence_input.request_key.company_id
            for s in self.current_allowed_sources
        ):
            raise AskEvidenceInputError("current same-company provider allowance required")


def _same_value(left: Any, right: Any) -> bool:
    """Exact supported legacy shape, including insertion order and scalar types."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(_same_value(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return left.hex() == right.hex()
    return (left is None or type(left) in (str, int, bool)) and left == right


def apply_ask_evidence_input(retrieval: dict, *, request_key: AskRequestKey,
                              admission: AskEvidenceAdmission) -> dict:
    """Return fresh canonical round-tripped collections for the old ASK consumer.

    The callback is trusted application code, not an untrusted plugin. This port
    rejects a different selection, reordering, truncation or changed record: its
    job is a compatibility transition, NOT reranking or repair. Non-evidence
    fields remain exactly on the existing path. Failure propagates as a technical
    exception; it is never converted here to no_sources or bypassed.
    """
    if type(retrieval) is not dict or not isinstance(admission, AskEvidenceAdmission):
        raise AskEvidenceInputError("prepared retrieval and typed admission required")
    envelope = admission.evidence_input
    if envelope.request_key != request_key:
        raise AskEvidenceInputError("ASK request binding changed")
    names = tuple(k for k in retrieval if k in COLLECTIONS and retrieval[k] is not None)
    if names != tuple(span.name for span in envelope.collections):
        raise AskEvidenceInputError("ASK collections missing, reordered or unexpectedly added")
    restored = restore_legacy_records(envelope.bundle, company_id=request_key.company_id,
                                      allowed_sources=admission.current_allowed_sources,
                                      adapter_limits=admission.adapter_limits)
    result = dict(retrieval)
    for span in envelope.collections:
        original = retrieval[span.name]
        expected_type = list if span.container_kind == "list" else tuple
        if type(original) is not expected_type:
            raise AskEvidenceInputError("ASK collection type changed")
        records = expected_type(v.record for v in restored[span.start:span.stop])
        if not _same_value(original, records):
            raise AskEvidenceInputError("canonical/legacy ASK selection differs")
        result[span.name] = records
    return result
