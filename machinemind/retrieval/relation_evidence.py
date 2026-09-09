"""P6-B2: explicit relation observations from existing two-direction SQL reads.

A relation row does NOT authorize its referenced page. The source endpoint must
come from p.company_id/p.machine_id/p.bubble_document_id of the joined page; the
other endpoint must be an independently authorized anchor supplied by the caller.
A missing LEFT-JOIN page stays an unresolved reference, never a fabricated page,
source, text or scope. Existing P5 relations and bundles are reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

from .chunk_evidence import ChunkReadScope, ChunkEvidenceLimits, validate_read
from .page_evidence import (PageEvidenceRead, PageReadBindingError, _integer,
                             observe_page, assemble_page_read)
from ..evidence.adapter_types import AdapterLimits
from ..evidence.contracts import (EvidenceRelation, SourceIdentity, SourceType,
                                  SourceFormat, canonical_json)
from ..evidence.legacy_compatibility import LegacyRecordInput

RELATION_BINDING_COLUMNS = (", r.company_id AS evidence_relation_company_id"
    ", r.machine_id AS evidence_relation_machine_id"
    ", r.parent_source_key AS evidence_parent_key, r.child_source_key AS evidence_child_key"
    ", r.relation_type AS evidence_relation_type"
    ", r.parent_source_type AS evidence_parent_type, r.child_source_type AS evidence_child_type"
    ", r.ordinal AS evidence_ordinal, r.relation_source AS evidence_relation_source"
    ", r.metadata AS evidence_relation_metadata"
    ", p.company_id AS evidence_page_company_id, p.bubble_document_id AS evidence_page_key")
RELATION_PROVIDER_VERSION = "scoped-relation-reader-p6b2-v1"


def _key(source: SourceIdentity) -> str:
    return source.source_type.value + ":" + source.source_id


def validate_relation_request(*, scope: ChunkReadScope, kind: str,
                              anchors: tuple[SourceIdentity, ...],
                              current_allowed_sources: frozenset[SourceIdentity],
                              text_chars: int, limits: ChunkEvidenceLimits) -> None:
    validate_read(scope, 0, limits)
    if kind not in {"related_steps", "parent_procedures"} or scope.ai_scope != "machine_all":
        raise PageReadBindingError("relation expansion requires explicit machine_all scope")
    if type(anchors) is not tuple or not anchors or len(anchors) > limits.assembly.max_occurrences:
        raise PageReadBindingError("explicit bounded anchor tuple required")
    if type(current_allowed_sources) is not frozenset or any(not isinstance(s, SourceIdentity)
        or s.scope.company_id != scope.company_id for s in current_allowed_sources):
        raise PageReadBindingError("explicit current company allowance required")
    expected_type = SourceType.PROCEDURE if kind == "related_steps" else SourceType.STEP
    if kind == "related_steps" and len(anchors) != 1:
        raise PageReadBindingError("one parent anchor required")
    bindings: dict[str, SourceIdentity] = {}
    for anchor in anchors:
        if (not isinstance(anchor, SourceIdentity) or anchor not in current_allowed_sources
            or anchor.source_type != expected_type or anchor.source_format != SourceFormat.STRUCTURED
            or not scope.permits(anchor.scope.company_id, anchor.scope.machine_id, _key(anchor))):
            raise PageReadBindingError("anchor not currently authorized in this scope")
        previous = bindings.get(_key(anchor))
        if previous is not None and previous != anchor:
            raise PageReadBindingError("ambiguous anchor association")
        bindings[_key(anchor)] = anchor
    _integer(text_chars, "text_chars")
    if text_chars > limits.adapter.max_text_chars:
        raise PageReadBindingError("relation page projection exceeds allocation limit")
    # The sentinel read is bounded before DB access; it fails on overflow.
    if limits.assembly.max_occurrences < 1:
        raise PageReadBindingError("positive relation row allocation required")


def _metadata_json(value: Any, limits: ChunkEvidenceLimits) -> str:
    """Bounded immutable snapshot of a JSONB value; no textual interpretation."""
    if value is not None and type(value) is not dict:
        raise PageReadBindingError("relation metadata must be a JSON object or NULL")
    nodes, chars = 0, 0
    ancestors: set[int] = set()
    def walk(item: Any, depth: int) -> None:
        nonlocal nodes, chars
        nodes += 1
        if nodes > limits.legacy.max_nodes or depth > limits.legacy.max_depth:
            raise PageReadBindingError("relation metadata allocation exceeded")
        typ = type(item)
        if item is None or typ in (str, bool, int, float):
            if typ is str:
                chars += len(item)
            if typ is float and not math.isfinite(item):
                raise PageReadBindingError("non-finite relation metadata")
            if chars > limits.legacy.max_bytes:
                raise PageReadBindingError("relation metadata byte budget exceeded")
            return
        if typ not in (dict, list) or id(item) in ancestors:
            raise PageReadBindingError("invalid/cyclic JSON relation metadata")
        if len(item) > limits.legacy.max_nodes - nodes:
            raise PageReadBindingError("relation metadata node budget exceeded")
        ancestors.add(id(item))
        try:
            if typ is dict:
                for key, entry in item.items():
                    if type(key) is not str:
                        raise PageReadBindingError("JSON metadata keys must be strings")
                    walk(key, depth + 1); walk(entry, depth + 1)
            else:
                for entry in item:
                    walk(entry, depth + 1)
        finally:
            ancestors.remove(id(item))
    walk(value, 0)
    result = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode("utf-8")) > limits.legacy.max_bytes:
        raise PageReadBindingError("relation JSON snapshot byte budget exceeded")
    return result


@dataclass(frozen=True, slots=True)
class RelationObservation:
    company_id: str
    machine_id: str
    parent_source_key: str
    child_source_key: str
    relation_type: str
    parent_source_type: str
    child_source_type: str
    ordinal: int | None
    relation_source: str | None
    metadata_json: str = field(repr=False)
    page_observation_index: int | None
    original_reference: str

    def __post_init__(self) -> None:
        for value in (self.company_id, self.machine_id, self.parent_source_key,
                      self.child_source_key, self.relation_type, self.original_reference):
            if type(value) is not str or not value.strip():
                raise PageReadBindingError("missing relation identity")
        if self.parent_source_type != "procedure" or self.child_source_type != "step":
            raise PageReadBindingError("relation endpoint types contradict Procedure/Step contract")
        if self.ordinal is not None and type(self.ordinal) is not int:
            raise PageReadBindingError("ordinal must be the stored integer or NULL")
        if self.relation_source is not None and type(self.relation_source) is not str:
            raise PageReadBindingError("relation_source must be stored text or NULL")
        if type(self.metadata_json) is not str:
            raise PageReadBindingError("immutable JSON metadata snapshot required")
        if self.page_observation_index is not None:
            _integer(self.page_observation_index, "page observation index")


@dataclass(frozen=True, slots=True)
class RelationEvidenceRead:
    """Missing-page relations are observations, NOT authorized evidence records."""
    pages: PageEvidenceRead = field(repr=False)
    anchors: tuple[SourceIdentity, ...]
    relations: tuple[RelationObservation, ...] = field(repr=False)
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.pages, PageEvidenceRead) or type(self.anchors) is not tuple or any(not isinstance(a, SourceIdentity) for a in self.anchors):
            raise PageReadBindingError("typed relation read required")
        if type(self.relations) is not tuple or any(not isinstance(r, RelationObservation) for r in self.relations):
            raise PageReadBindingError("immutable relation observations required")
        for r in self.relations:
            if r.company_id != self.pages.scope.company_id or r.machine_id != self.pages.scope.machine_id:
                raise PageReadBindingError("relation outside request scope")
            idx = r.page_observation_index
            if idx is not None:
                if idx >= len(self.pages.observations):
                    raise PageReadBindingError("referenced page observation missing")
                o = self.pages.observations[idx]
                expected_key = r.child_source_key if self.pages.kind == "related_steps" else r.parent_source_key
                if o.storage_document_id != expected_key:
                    raise PageReadBindingError("relation/page key mismatch")
        size = self.pages.size_bytes + len(canonical_json(self.relations).encode("utf-8")) + len(canonical_json(self.anchors).encode("utf-8"))
        if type(self.size_bytes) is not int or self.size_bytes != size or size > self.pages.bundle.limits.max_bytes:
            raise PageReadBindingError("aggregate relation snapshot byte budget exceeded")

    @property
    def unresolved_page_references(self) -> tuple[RelationObservation, ...]:
        return tuple(r for r in self.relations if r.page_observation_index is None)

    @property
    def read_sources(self) -> frozenset[SourceIdentity]:
        return self.pages.read_sources

    def as_legacy_inputs(self, *, scope: ChunkReadScope,
                         current_allowed_sources: frozenset[SourceIdentity],
                         adapter_limits: AdapterLimits) -> tuple[LegacyRecordInput, ...]:
        if type(current_allowed_sources) is not frozenset or any(a not in current_allowed_sources for a in self.anchors):
            raise PageReadBindingError("relation anchor authorization revoked")
        return self.pages.as_legacy_inputs(scope=scope,
            current_allowed_sources=current_allowed_sources, adapter_limits=adapter_limits)


def build_relation_read(*, scope: ChunkReadScope, kind: str, anchors: tuple[SourceIdentity, ...],
                        current_allowed_sources: frozenset[SourceIdentity], text_chars: int,
                        relation_type: str, rows: list[tuple], limits: ChunkEvidenceLimits) -> RelationEvidenceRead:
    validate_relation_request(scope=scope, kind=kind, anchors=anchors,
        current_allowed_sources=current_allowed_sources, text_chars=text_chars, limits=limits)
    if type(relation_type) is not str or not relation_type.strip():
        raise PageReadBindingError("explicit relation type required")
    if type(rows) is not list or len(rows) > limits.assembly.max_occurrences:
        raise PageReadBindingError("relation read overflow: no partial family admitted")
    width = 5 if kind == "related_steps" else 6
    pages, records, observations = [], [], []
    anchor_map = {_key(a): a for a in anchors}
    used_bytes = 0
    for row_index, row in enumerate(rows):
        if type(row) is not tuple or len(row) != width + 12:
            raise PageReadBindingError("unexpected relation SQL shape")
        (company, machine, parent, child, typ, parent_type, child_type, ordinal,
         relation_source, metadata, pcompany, pkey) = row[width:]
        if kind == "related_steps":
            raw_child, raw_ordinal, pmachine, page_no, text = row[:width]
            raw_parent = parent
        else:
            raw_child, raw_parent, raw_ordinal, pmachine, page_no, text = row[:width]
        if company != scope.company_id or machine != scope.machine_id or typ != relation_type:
            raise PageReadBindingError("relation row outside exact SQL scope/type")
        if raw_child != child or raw_parent != parent or type(raw_ordinal) is not type(ordinal) or raw_ordinal != ordinal:
            raise PageReadBindingError("relation projected columns disagree")
        # Storage key types are fixed by ingest, not identified from human text.
        if type(parent) is not str or not parent.startswith("procedure:") or not parent[len("procedure:"):].strip():
            raise PageReadBindingError("invalid explicit parent storage key")
        if type(child) is not str or not child.startswith("step:") or not child[len("step:"):].strip():
            raise PageReadBindingError("invalid explicit child storage key")
        for value in (company, machine, parent, child, typ, parent_type, child_type, relation_source):
            if type(value) is str and len(value) > limits.adapter.max_aux_chars:
                raise PageReadBindingError("relation auxiliary allocation exceeded")
        anchor_key = parent if kind == "related_steps" else child
        if anchor_key not in anchor_map:
            raise PageReadBindingError("relation not returned for an authorized anchor")
        snapshot = _metadata_json(metadata, limits)
        reference = canonical_json(("public.structured_source_relations", company, machine,
                                     child, parent, typ, ordinal, relation_source))
        page_index = None
        if pkey is None:
            # Only the parent LEFT JOIN can yield a missing page.
            if kind != "parent_procedures" or pcompany is not None or pmachine is not None or page_no is not None or text != "":
                raise PageReadBindingError("inconsistent missing joined page")
        else:
            expected_key = child if kind == "related_steps" else parent
            if pkey != expected_key or pcompany != company:
                raise PageReadBindingError("joined page does not match the relation endpoint")
            relation = EvidenceRelation("parent_procedure" if kind == "related_steps" else "child_step",
                                        anchor_map[anchor_key], reference)
            page_index = len(pages)
            o = observe_page(scope=scope, company=pcompany, machine=pmachine, key=pkey,
                page=page_no, text=text, projection_chars=text_chars, kind=kind,
                batch=0, row=row_index, limits=limits, relations=(relation,))
            pages.append(o)
            # No fake chunk_index, default page=1, parsed metadata or parent text.
            records.append({"bubble_document_id": pkey, "company_id": pcompany,
                "machine_id": pmachine, "page_number": page_no, "text": text})
        obs = RelationObservation(company, machine, parent, child, typ, parent_type,
            child_type, ordinal, relation_source, snapshot, page_index, reference)
        used_bytes += len(canonical_json(obs).encode("utf-8"))
        if used_bytes > limits.legacy.max_bytes:
            raise PageReadBindingError("relation observation byte budget exceeded")
        observations.append(obs)
    result = assemble_page_read(scope=scope, kind=kind, observations=tuple(pages),
        indices=tuple(range(len(pages))), records=records, limits=limits, query_count=1,
        layout="document_page", relation_allowance=frozenset(anchors))
    size = result.size_bytes + len(canonical_json(tuple(observations)).encode("utf-8")) + len(canonical_json(anchors).encode("utf-8"))
    return RelationEvidenceRead(result, anchors, tuple(observations), size)
