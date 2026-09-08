"""Lossless, immutable internal evidence contracts; not an authorization service.

The provider supplies identity, scope and provenance from authorized records.
No field is inferred from a question, title, language, URL or source name.
Page/row numbers are one-based; chunk indexes and text spans are zero-based.
Offsets refer to the exact indexed text, NOT to bytes or the original PDF.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

SCHEMA_VERSION = "canonical-evidence-v1"


class EvidenceContractError(ValueError):
    """Malformed, inconsistent or unsupported canonical data."""


class SourceType(str, Enum):
    DOCUMENT = "document"
    PROCEDURE = "procedure"
    STEP = "step"
    PROBLEM_SOLUTION = "ps"
    PHOTO = "md_photo"
    VIDEO = "md_video"


class SourceFormat(str, Enum):
    PDF = "pdf"
    XLSX = "xlsx"
    STRUCTURED = "structured"
    UNKNOWN = "unknown"


class ScopeLevel(str, Enum):
    MACHINE = "machine"
    COMPANY = "company"
    # Machine association unavailable; not itself an authorization verdict.
    UNRESOLVED = "unresolved"


class ContentKind(str, Enum):
    INDEXED_TEXT = "indexed_text"
    STRUCTURED_TEXT = "structured_text"
    MEDIA_METADATA = "media_metadata"


class PageBasis(str, Enum):
    PDF_PHYSICAL = "pdf_physical"
    VIRTUAL = "virtual"
    UNSPECIFIED = "unspecified"


def _text(value: Any, name: str, *, optional: bool = False, nonblank: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise EvidenceContractError(f"{name}: expected {'nonblank ' if nonblank else ''}text")


def _number(value: Any, name: str, minimum: int, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not int or value < minimum:
        raise EvidenceContractError(f"{name}: expected integer >= {minimum}")


def _enum(value: Any, cls: type[Enum], name: str) -> None:
    if not isinstance(value, cls):
        raise EvidenceContractError(f"{name}: expected {cls.__name__}")


def _tuple_of(value: Any, cls: type, name: str) -> None:
    # Copying a mutable sequence implicitly could hide later source mutations.
    if type(value) is not tuple or any(not isinstance(x, cls) for x in value):
        raise EvidenceContractError(f"{name}: expected immutable tuple of {cls.__name__}")


def to_primitive(value: Any) -> Any:
    """A fresh JSON-compatible value; no references to mutable caller objects."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_primitive(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [to_primitive(x) for x in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise EvidenceContractError("unsupported JSON value")


def canonical_json(value: Any) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class SourceScope:
    company_id: str
    level: ScopeLevel
    machine_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.company_id, "company_id", nonblank=True)
        _enum(self.level, ScopeLevel, "scope level")
        _text(self.machine_id, "machine_id", optional=True, nonblank=True)
        if (self.level == ScopeLevel.MACHINE) != (self.machine_id is not None):
            raise EvidenceContractError("machine scope requires a machine_id; other levels do not")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    scope: SourceScope
    source_type: SourceType
    source_id: str
    source_format: SourceFormat = SourceFormat.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.scope, SourceScope):
            raise EvidenceContractError("source scope is required")
        _enum(self.source_type, SourceType, "source_type")
        _enum(self.source_format, SourceFormat, "source_format")
        _text(self.source_id, "source_id", nonblank=True)
        if self.source_type != SourceType.DOCUMENT and self.source_format in (SourceFormat.PDF, SourceFormat.XLSX):
            raise EvidenceContractError("PDF/XLSX are document formats, not structured source types")


@dataclass(frozen=True, slots=True)
class SourceLocator:
    page_from: int | None = None
    page_to: int | None = None
    page_basis: PageBasis = PageBasis.UNSPECIFIED
    page_label: str | None = None
    chunk_index: int | None = None
    sheet: str | None = None
    row_from: int | None = None
    row_to: int | None = None
    field: str | None = None
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("page_from", "page_to", "row_from", "row_to"):
            _number(getattr(self, name), name, 1, optional=True)
        _number(self.chunk_index, "chunk_index", 0, optional=True)
        _enum(self.page_basis, PageBasis, "page_basis")
        for start, end in ((self.page_from, self.page_to), (self.row_from, self.row_to)):
            if end is not None and (start is None or end < start):
                raise EvidenceContractError("range end requires a start and cannot precede it")
        if self.page_from is None and self.page_basis != PageBasis.UNSPECIFIED:
            raise EvidenceContractError("page basis without a page")
        for name in ("page_label", "sheet", "field"):
            _text(getattr(self, name), name, optional=True, nonblank=True)
        if self.row_from is not None and self.sheet is None:
            raise EvidenceContractError("a spreadsheet row requires a sheet name")
        _tuple_of(self.section_path, str, "section_path")
        for item in self.section_path:
            _text(item, "section", nonblank=True)


@dataclass(frozen=True, slots=True)
class Provenance:
    provider: str
    original_reference: str
    source_revision: str | None = None

    def __post_init__(self) -> None:
        _text(self.provider, "provider", nonblank=True)
        _text(self.original_reference, "original_reference", nonblank=True)
        _text(self.source_revision, "source_revision", optional=True, nonblank=True)


@dataclass(frozen=True, slots=True)
class EvidenceField:
    name: str
    value: str

    def __post_init__(self) -> None:
        _text(self.name, "field name", nonblank=True)
        _text(self.value, "field value")


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    """A provider-declared relationship, never an inferred authorization/causality."""
    kind: str
    target: SourceIdentity
    original_reference: str

    def __post_init__(self) -> None:
        _text(self.kind, "relation kind", nonblank=True)
        _text(self.original_reference, "relation provenance", nonblank=True)
        if not isinstance(self.target, SourceIdentity):
            raise EvidenceContractError("relation target must be scoped")


@dataclass(frozen=True, slots=True)
class LinkTarget:
    """Opaque application reference; no stored signed URL or network resolution."""
    collection: str
    record_id: str

    def __post_init__(self) -> None:
        _text(self.collection, "link collection", nonblank=True)
        _text(self.record_id, "link record_id", nonblank=True)
        if "://" in self.collection or "://" in self.record_id:
            raise EvidenceContractError("link target must contain IDs, not URLs")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    source: SourceIdentity
    locator: SourceLocator
    provenance: Provenance
    content_kind: ContentKind
    text: str
    title: str | None = None
    language: str | None = None
    fields: tuple[EvidenceField, ...] = ()
    relations: tuple[EvidenceRelation, ...] = ()
    link_target: LinkTarget | None = None
    incomplete_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, cls in ((self.source, SourceIdentity), (self.locator, SourceLocator),
                           (self.provenance, Provenance)):
            if not isinstance(value, cls):
                raise EvidenceContractError(f"expected {cls.__name__}")
        _enum(self.content_kind, ContentKind, "content_kind")
        _text(self.text, "indexed text")
        _text(self.title, "title", optional=True)
        _text(self.language, "language", optional=True, nonblank=True)
        _tuple_of(self.fields, EvidenceField, "fields")
        _tuple_of(self.relations, EvidenceRelation, "relations")
        _tuple_of(self.incomplete_reasons, str, "incomplete_reasons")
        if len({f.name for f in self.fields}) != len(self.fields):
            raise EvidenceContractError("duplicate field name")
        for reason in self.incomplete_reasons:
            _text(reason, "incomplete reason", nonblank=True)
        if self.link_target is not None and not isinstance(self.link_target, LinkTarget):
            raise EvidenceContractError("invalid link target")
        if any(r.target.scope.company_id != self.source.scope.company_id for r in self.relations):
            raise EvidenceContractError("cross-company relation")
        if self.source.source_type in (SourceType.PHOTO, SourceType.VIDEO):
            if self.content_kind != ContentKind.MEDIA_METADATA:
                raise EvidenceContractError("photo/video currently support metadata only")
        elif self.content_kind == ContentKind.MEDIA_METADATA:
            raise EvidenceContractError("media metadata requires a photo/video source")
        if self.source.source_format == SourceFormat.XLSX and self.locator.page_basis == PageBasis.PDF_PHYSICAL:
            raise EvidenceContractError("XLSX pages cannot be declared physical PDF pages")

    @property
    def evidence_id(self) -> str:
        """Versioned digest of ALL declared immutable data, not a retrieval score.

        IDs change when declared content/provenance changes. Missing values are
        retained as missing. Text is never normalized, truncated or lowercased.
        """
        digest = hashlib.sha256((SCHEMA_VERSION + "\n" + canonical_json(self)).encode("utf-8")).hexdigest()
        return "ev1_" + digest

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "evidence_id": self.evidence_id,
                **to_primitive(self)}


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        _number(self.start, "span start", 0)
        _number(self.end, "span end", 0)
        if self.end <= self.start:
            raise EvidenceContractError("span must be non-empty")

    def extract(self, evidence: EvidenceRecord) -> str:
        if not isinstance(evidence, EvidenceRecord) or self.end > len(evidence.text):
            raise EvidenceContractError("span outside indexed text")
        return evidence.text[self.start:self.end]


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Named raw score; never silently interpreted as a probability or clamped."""
    name: str
    value: float
    producer: str

    def __post_init__(self) -> None:
        _text(self.name, "score name", nonblank=True)
        _text(self.producer, "score producer", nonblank=True)
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise EvidenceContractError("score must be finite")
