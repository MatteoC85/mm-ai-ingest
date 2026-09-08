"""Adapters for structured source snapshots and explicit Step->Procedure edges.

An indexed snapshot is REQUIRED. The original payload and the already-indexed
text are distinct views; this adapter does not silently regenerate one from the
other, parse free prose as database columns, or create an edge from a title/code.
The caller must supply metadata from the same source/revision. These adapters do
not read, update or authorize the underlying stored records.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adapter_types import AdapterContext, AdapterLimits, AdaptationResult, EvidenceAdapterError
from .contracts import EvidenceRelation, SourceType
from .record_adapters import _View, _adapt


def adapt_structured_source(record: Mapping[str, Any], *, indexed_text: str,
                            context: AdapterContext, limits: AdapterLimits) -> AdaptationResult:
    """Existing ingest payload shape plus a provider-supplied indexed snapshot.

    Known payload columns (including multilingual/multiline description, solution,
    notes) are copied exactly to named fields. An inline title never becomes a
    scoped identifier. Media have metadata semantics even when text describes a
    visual observation. The actual image/video bytes are never read here.
    """
    if not isinstance(context, AdapterContext) or context.source.source_type == SourceType.DOCUMENT:
        raise EvidenceAdapterError("structured provider binding required")
    if not isinstance(record, Mapping) or not isinstance(limits, AdapterLimits):
        raise EvidenceAdapterError("record mapping and explicit limits required")
    if len(record) > limits.max_fields:
        raise EvidenceAdapterError("payload field limit exceeded")
    if "indexed_text" in record and record["indexed_text"] != indexed_text:
        raise EvidenceAdapterError("different indexed snapshots supplied")
    combined = dict(record)
    combined["indexed_text"] = indexed_text
    return _adapt(combined, context=context, limits=limits, layout="structured_source_snapshot",
                  text_field="indexed_text", payload=True)


def adapt_step_parent_relation(record: Mapping[str, Any], *, child: AdapterContext,
                               parent: AdapterContext, original_reference: str,
                               limits: AdapterLimits) -> EvidenceRelation:
    """Convert a structured_source_relations row, never a fuzzy title match.

    The relation's direction is child (Step) -> parent (Procedure). Source type,
    both keys, company, machine and the stored relation_type are checked. Neither
    the ordinal nor the edge grants access to a source or triggers a lookup.
    The ordinal stays in the record/Step fields; it is not changed into a row or
    chunk index. Parent inclusion/manifest assembly is a later P5-C operation.
    """
    if not isinstance(child, AdapterContext) or not isinstance(parent, AdapterContext):
        raise EvidenceAdapterError("two explicit provider bindings required")
    if child.source.source_type != SourceType.STEP or parent.source.source_type != SourceType.PROCEDURE:
        raise EvidenceAdapterError("expected Step -> Procedure relation")
    if child.source.scope != parent.source.scope:
        raise EvidenceAdapterError("relation scope differs between Step and Procedure")
    if parent.source not in child.allowed_sources or child.source not in parent.allowed_sources:
        raise EvidenceAdapterError("relation endpoints not allowed by both bindings")
    if type(original_reference) is not str or not original_reference.strip():
        raise EvidenceAdapterError("stored relation provenance required")
    view = _View(record, limits)
    expected = {
        "company_id": child.source.scope.company_id,
        "child_source_key": child.legacy_document_id,
        "parent_source_key": parent.legacy_document_id,
        "relation_type": "procedure_step",
        "child_source_type": "step",
        "parent_source_type": "procedure",
    }
    for key, value in expected.items():
        if view.string(key, nonblank=True, metadata=False) != value:
            raise EvidenceAdapterError("inconsistent relation column: " + key)
    # The stored relation table is machine scoped. Missing association is not
    # guessed from a request; it must match the endpoint binding.
    if view.string("machine_id", nonblank=True, metadata=False) != child.source.scope.machine_id:
        raise EvidenceAdapterError("relation machine differs from binding")
    ordinal = view.get("ordinal")
    if ordinal is not None and type(ordinal) is not int:
        raise EvidenceAdapterError("relation ordinal must be an integer or null")
    view.aux_chars += len(original_reference)
    view._bound_aux()
    return EvidenceRelation("parent_procedure", parent.source, original_reference)
