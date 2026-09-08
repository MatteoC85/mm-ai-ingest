"""Pure adapters of authorized document_pages/chunks and retrieval dictionaries.

Text is a snapshot of the selected INDEXED field, not an original PDF, workbook,
image or video. No cleaning, translation, re-chunking, ranking, I/O or inference.
A legacy candidate's chunk_full is often LEFT(..., N) in the current retrieval;
its name does not establish completeness. Textual envelopes are retained opaque:
sheet/row/field metadata is mapped ONLY if provided as explicit record metadata.
Unmapped metadata is reported, not silently made into canonical facts.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adapter_types import (AdapterContext, AdapterLimits, AdaptationResult,
                            AdaptationTrace, EvidenceAdapterError)
from .contracts import (ContentKind, EvidenceField, EvidenceRecord, PageBasis,
                        RetrievalScore, ScopeLevel, SourceFormat, SourceLocator,
                        SourceType)
from .manifest import EvidenceEntry

DEFAULT_SCORE_FIELDS = (
    "similarity", "semantic_similarity", "retrieval_score", "rerank_score",
    "fts_rank", "rrf_score", "structured_rescue_score", "structured_direct_score",
    "overlap_score", "specificity_score",
)
# Transport data never becomes evidence text, confidence or a stable link.
TRANSPORT_FIELDS = frozenset({"embedding", "embedding_list", "file_url", "source_url", "url",
                             "rg_links", "exact_machine_scope", "query_used"})
SOURCE_TYPE_ALIASES = {
    "document": SourceType.DOCUMENT,
    "procedure": SourceType.PROCEDURE,
    "step": SourceType.STEP,
    "ps": SourceType.PROBLEM_SOLUTION,
    "problem_solution": SourceType.PROBLEM_SOLUTION,
    "md_photo": SourceType.PHOTO,
    "machine_detail_photo": SourceType.PHOTO,
    "md_video": SourceType.VIDEO,
    "machine_detail_video": SourceType.VIDEO,
}
# These are serialized field names of the existing ingest API, not domain words.
STRUCTURED_FIELDS = {
    SourceType.PROCEDURE: ("title", "procedure_type", "short_description"),
    SourceType.STEP: ("title", "step_number", "parent_procedure_id",
                      "parent_procedure_code", "parent_procedure_title", "description"),
    SourceType.PROBLEM_SOLUTION: ("title", "category", "description", "solution", "notes"),
    SourceType.PHOTO: ("title", "description"),
    SourceType.VIDEO: ("title", "description"),
}


class _View:
    """Per-call bookkeeping. No globals or mutation of the caller's mapping."""
    def __init__(self, raw: Mapping[str, Any], limits: AdapterLimits) -> None:
        if not isinstance(raw, Mapping) or not isinstance(limits, AdapterLimits):
            raise EvidenceAdapterError("record mapping and explicit limits required")
        if len(raw) > limits.max_fields or any(type(k) is not str for k in raw):
            raise EvidenceAdapterError("record field limit or invalid key")
        self.raw = dict(raw)
        meta = self.raw.get("metadata")
        if meta is None:
            meta = {}
        if not isinstance(meta, Mapping) or any(type(k) is not str for k in meta):
            raise EvidenceAdapterError("metadata must be a mapping with text keys")
        if len(raw) + len(meta) > limits.max_fields:
            raise EvidenceAdapterError("record and metadata field limit exceeded")
        self.meta = dict(meta)
        self.used: set[str] = set()
        self.limits = limits
        self.aux_chars = sum(len(k) for k in self.raw) + sum(len(k) for k in self.meta)
        self._bound_aux()

    def _bound_aux(self) -> None:
        if self.aux_chars > self.limits.max_aux_chars:
            raise EvidenceAdapterError("auxiliary data budget exceeded")

    def get(self, *keys: str, metadata: bool = True) -> Any:
        found = []
        for key in keys:
            if key in self.raw:
                self.used.add(key)
                if self.raw[key] is not None:
                    found.append(self.raw[key])
            if metadata and key in self.meta:
                self.used.add("metadata." + key)
                if self.meta[key] is not None:
                    found.append(self.meta[key])
        if not found:
            return None
        if any(type(v) is not type(found[0]) or v != found[0] for v in found[1:]):
            raise EvidenceAdapterError("conflicting explicit metadata: " + keys[0])
        return found[0]

    def string(self, *keys: str, nonblank: bool = False, metadata: bool = True) -> str | None:
        value = self.get(*keys, metadata=metadata)
        if value is None:
            return None
        if type(value) is not str or (nonblank and not value.strip()):
            raise EvidenceAdapterError("expected text metadata: " + keys[0])
        self.aux_chars += len(value)
        self._bound_aux()
        return value

    def integer(self, *keys: str, minimum: int) -> int | None:
        value = self.get(*keys)
        if value is not None and (type(value) is not int or value < minimum):
            raise EvidenceAdapterError("invalid ordinal: " + keys[0])
        return value

    def flag(self, *keys: str) -> bool | None:
        value = self.get(*keys)
        if value is not None and type(value) is not bool:
            raise EvidenceAdapterError("invalid boolean: " + keys[0])
        return value

    def trace(self, layout: str, text_field: str, citation: str | None) -> AdaptationTrace:
        all_fields = (set(self.raw) - {"metadata"}) | {"metadata." + k for k in self.meta}
        excluded = {k for k in all_fields if k.rsplit(".", 1)[-1] in TRANSPORT_FIELDS}
        # Do not hide a used key merely because a caller supplied a transport alias.
        excluded -= self.used
        return AdaptationTrace(layout, text_field, citation, tuple(sorted(self.used)),
                               tuple(sorted(all_fields - self.used - excluded)),
                               tuple(sorted(excluded)))


def _binding(view: _View, context: AdapterContext, *, payload: bool) -> None:
    if not isinstance(context, AdapterContext):
        raise EvidenceAdapterError("explicit provider context required")
    key = view.string("bubble_document_id", nonblank=True)
    if not payload and key is None:
        raise EvidenceAdapterError("storage source key missing")
    if key is not None and key != context.legacy_document_id:
        raise EvidenceAdapterError("record identity differs from provider binding")
    source_id = view.string("source_id", nonblank=True)
    if payload and source_id is None:
        raise EvidenceAdapterError("structured source_id missing")
    if source_id is not None and source_id != context.source.source_id:
        raise EvidenceAdapterError("source_id differs from provider binding")
    typ = view.string("source_type", nonblank=True)
    if payload and typ is None:
        raise EvidenceAdapterError("structured source_type missing")
    if typ is not None and SOURCE_TYPE_ALIASES.get(typ) != context.source.source_type:
        raise EvidenceAdapterError("source type differs or is unsupported")
    company = view.string("company_id", nonblank=True)
    if company is not None and company != context.source.scope.company_id:
        raise EvidenceAdapterError("tenant differs from provider binding")
    # A projected candidate can lack these columns. Their absence does NOT create
    # permission; the provider binding is still mandatory. Explicit NULL/empty
    # association is not rewritten into the request's selected machine.
    for location in (view.raw, view.meta):
        if "company_id" in location and location["company_id"] != context.source.scope.company_id:
            raise EvidenceAdapterError("explicit tenant column differs from provider binding")
        if "machine_id" in location:
            explicit = location["machine_id"]
            if explicit is not None and type(explicit) is not str:
                raise EvidenceAdapterError("machine_id must be text or null")
            if (explicit or None) != context.source.scope.machine_id:
                raise EvidenceAdapterError("explicit machine column differs from provider binding")
    machine = view.get("machine_id")
    if machine is not None and type(machine) is not str:
        raise EvidenceAdapterError("machine_id must be text or null")
    has_machine = "machine_id" in view.raw or "machine_id" in view.meta
    if has_machine and (machine or None) != context.source.scope.machine_id:
        raise EvidenceAdapterError("machine association differs from provider binding")
    fmt = view.string("source_format", "source_file_type", nonblank=True)
    if fmt is not None and fmt != context.source.source_format.value:
        raise EvidenceAdapterError("format differs from provider binding")
    revision = view.string("source_revision", nonblank=True)
    if revision is not None and revision != context.provenance.source_revision:
        raise EvidenceAdapterError("revision differs from provider binding")


def _locator(view: _View, context: AdapterContext, reasons: list[str]) -> SourceLocator:
    start = view.integer("page_from", "page_number", minimum=1)
    end = view.integer("page_to", minimum=1)
    chunk = view.integer("chunk_index", minimum=0)
    label = view.string("page_label", nonblank=True)
    sheet = view.string("sheet", "sheet_name", nonblank=True)
    row = view.integer("row_from", "row_number", minimum=1)
    row_to = view.integer("row_to", minimum=1)
    field = view.string("field", nonblank=True)
    section_path = view.get("section_path")
    if section_path is None:
        section_path = ()
    if type(section_path) not in (tuple, list) or any(type(s) is not str or not s.strip() for s in section_path):
        raise EvidenceAdapterError("section_path must be explicit ordered text components")
    view.aux_chars += sum(map(len, section_path))
    view._bound_aux()
    if len(section_path) > view.limits.max_fields:
        raise EvidenceAdapterError("section path limit exceeded")
    fmt = context.source.source_format
    if start is None:
        basis = PageBasis.UNSPECIFIED
        reasons.append("page_locator_unavailable")
    elif fmt == SourceFormat.PDF:
        basis = PageBasis.PDF_PHYSICAL
    elif fmt in (SourceFormat.XLSX, SourceFormat.STRUCTURED):
        basis = PageBasis.VIRTUAL
    else:
        basis = PageBasis.UNSPECIFIED
    raw_basis = view.string("page_basis", nonblank=True)
    if raw_basis is not None and raw_basis != basis.value:
        raise EvidenceAdapterError("page basis contradicts supplied source format")
    if (sheet is not None or row is not None or row_to is not None) and (
        context.source.source_type != SourceType.DOCUMENT or fmt == SourceFormat.PDF
    ):
        raise EvidenceAdapterError("spreadsheet locator on a non-spreadsheet source")
    if fmt == SourceFormat.XLSX:
        if sheet is None:
            reasons.append("xlsx_sheet_metadata_unavailable")
        if row is None:
            reasons.append("xlsx_row_metadata_unavailable")
    return SourceLocator(start, end, basis, label, chunk, sheet, row, row_to, field,
                         tuple(section_path))


def _fields(view: _View, context: AdapterContext) -> tuple[EvidenceField, ...]:
    result = []
    for key in STRUCTURED_FIELDS.get(context.source.source_type, ()):
        if key == "step_number":
            value = view.get(key)
            if value is not None:
                # The existing API accepts any integer step_number. Preserve it
                # as a field; it is NOT a canonical page, row or chunk index.
                if type(value) is not int:
                    raise EvidenceAdapterError("step_number must be an integer")
                result.append(EvidenceField(key, str(value)))
        else:
            value = view.string(key)
            if value is not None:
                result.append(EvidenceField(key, value))
    if len(result) > view.limits.max_fields:
        raise EvidenceAdapterError("canonical field limit exceeded")
    return tuple(result)


def _adapt(record: Mapping[str, Any], *, context: AdapterContext, limits: AdapterLimits,
           layout: str, text_field: str, payload: bool = False,
           score_fields: tuple[str, ...] = DEFAULT_SCORE_FIELDS) -> AdaptationResult:
    view = _View(record, limits)
    _binding(view, context, payload=payload)
    if type(score_fields) is not tuple or any(type(k) is not str or not k for k in score_fields):
        raise EvidenceAdapterError("score field names must be an immutable tuple")
    if len(score_fields) != len(set(score_fields)) or len(score_fields) > limits.max_fields:
        raise EvidenceAdapterError("duplicate or excessive score fields")
    if any(k not in DEFAULT_SCORE_FIELDS for k in score_fields):
        raise EvidenceAdapterError("unsupported score field; no implicit confidence conversion")
    text = view.get(text_field, metadata=False)
    if type(text) is not str:
        raise EvidenceAdapterError("indexed text missing or invalid: " + text_field)
    if len(text) > limits.max_text_chars:
        raise EvidenceAdapterError("indexed text budget exceeded; no truncation")
    reasons: list[str] = []
    if not text:
        reasons.append("empty_indexed_text")
    if layout == "retrieval_candidate":
        reasons.append("retrieved_text_extent_unverified")
        if text_field == "snippet":
            reasons.append("snippet_only")
    if view.flag("truncated", "is_truncated"):
        reasons.append("provider_declared_truncation")
    if context.source.scope.level == ScopeLevel.UNRESOLVED:
        reasons.append("machine_association_unresolved")
    if context.source.source_format == SourceFormat.UNKNOWN:
        reasons.append("source_format_unavailable")
    if context.provenance.source_revision is None:
        reasons.append("source_revision_unavailable")
    if context.link_target is None:
        reasons.append("application_link_unavailable")
    locator = _locator(view, context, reasons)
    title = view.string("title", "source_title")
    language = view.string("language", "source_language", nonblank=True)
    fields = _fields(view, context)
    if context.source.source_type == SourceType.STEP:
        parent_id = next((f.value for f in fields if f.name == "parent_procedure_id" and f.value), None)
        parent_edges = [r for r in context.relations if r.kind == "parent_procedure"]
        if any(r.target.source_type != SourceType.PROCEDURE for r in parent_edges):
            raise EvidenceAdapterError("parent_procedure edge must target a Procedure")
        if len(parent_edges) > 1:
            raise EvidenceAdapterError("multiple explicit parent procedures")
        if parent_id is not None and parent_edges:
            expected_parent_ids = {parent_edges[0].target.source_id,
                                   "procedure:" + parent_edges[0].target.source_id}
            if parent_id not in expected_parent_ids:
                raise EvidenceAdapterError("parent metadata conflicts with bound relation")
        if parent_id is not None and not parent_edges:
            reasons.append("parent_relationship_not_bound")
    if context.source.source_type not in (SourceType.DOCUMENT, SourceType.PHOTO, SourceType.VIDEO) and not fields:
        reasons.append("structured_fields_unavailable_in_indexed_view")
    # Limit all trusted side data too; a provider binding cannot bypass data bounds.
    auxiliary = (context.legacy_document_id, context.source.scope.company_id,
                 context.source.scope.machine_id or "", context.provenance.provider,
                 context.provenance.original_reference, context.provenance.source_revision or "")
    view.aux_chars += sum(map(len, auxiliary))
    if len(context.relations) > limits.max_fields:
        raise EvidenceAdapterError("relation count budget exceeded")
    for rel in context.relations:
        view.aux_chars += len(rel.kind) + len(rel.original_reference) + len(rel.target.source_id)
    if context.link_target is not None:
        view.aux_chars += len(context.link_target.collection) + len(context.link_target.record_id)
    view._bound_aux()
    scores = []
    for key in score_fields:
        value = view.get(key)
        if value is not None:
            # RetrievalScore rejects bool/NaN/inf and preserves signed scores >1.
            scores.append(RetrievalScore(key, value, context.provenance.provider))
    source_type = context.source.source_type
    kind = (ContentKind.INDEXED_TEXT if source_type == SourceType.DOCUMENT else
            ContentKind.MEDIA_METADATA if source_type in (SourceType.PHOTO, SourceType.VIDEO)
            else ContentKind.STRUCTURED_TEXT)
    evidence = EvidenceRecord(context.source, locator, context.provenance, kind, text,
                              title, language, fields, context.relations, context.link_target,
                              tuple(dict.fromkeys(reasons)))
    citation = view.string("citation_id", nonblank=True)
    return AdaptationResult(EvidenceEntry(evidence, tuple(scores)),
                            view.trace(layout, text_field, citation))


def adapt_page(record: Mapping[str, Any], *, context: AdapterContext,
               limits: AdapterLimits) -> AdaptationResult:
    """document_pages-style dictionary (text, page_number, source key)."""
    return _adapt(record, context=context, limits=limits, layout="document_page", text_field="text")


def adapt_chunk(record: Mapping[str, Any], *, context: AdapterContext,
                limits: AdapterLimits) -> AdaptationResult:
    """document_chunks-style dictionary; chunk_index is copied, not renumbered."""
    return _adapt(record, context=context, limits=limits, layout="document_chunk", text_field="chunk_text")


def adapt_candidate(record: Mapping[str, Any], *, context: AdapterContext,
                    limits: AdapterLimits) -> AdaptationResult:
    """Use chunk_full if supplied, otherwise snippet with an explicit warning.

    Empty chunk_full stays empty. An invalid non-text value is not hidden by a
    fallback. Neither field is represented as the full original source.
    """
    if not isinstance(record, Mapping):
        raise EvidenceAdapterError("candidate mapping required")
    field = "chunk_full" if record.get("chunk_full") is not None else "snippet"
    return _adapt(record, context=context, limits=limits, layout="retrieval_candidate", text_field=field)
