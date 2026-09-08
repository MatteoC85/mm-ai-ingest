"""P5-B: explicit, immutable inputs/outputs for conversion only.

An AdapterContext is supplied by a provider which has ALREADY authorized and
bound the record. It is not derived from the user's question, current machine,
source title, URL, or ``exact_machine_scope`` retrieval flag. No ACL is evaluated
here, and constructing this object does not authenticate its caller.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import (EvidenceContractError, EvidenceRelation, LinkTarget,
                        Provenance, SourceIdentity, SourceType)
from .manifest import EvidenceEntry

ADAPTER_VERSION = "canonical-record-adapters-v1"


class EvidenceAdapterError(EvidenceContractError):
    """The input cannot be converted without guessing or violating a binding."""


@dataclass(frozen=True, slots=True)
class AdapterLimits:
    """Data allocation limits, not LLM token limits or a dollar budget.

    The composing provider must choose them explicitly. Exceeding a bound raises;
    it NEVER truncates text, fields, scores, or relations to fit.
    """
    max_text_chars: int
    max_fields: int
    max_aux_chars: int

    def __post_init__(self) -> None:
        for name in ("max_text_chars", "max_fields", "max_aux_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise EvidenceAdapterError(f"{name}: expected nonnegative integer")


@dataclass(frozen=True, slots=True)
class AdapterContext:
    source: SourceIdentity
    provenance: Provenance
    allowed_sources: frozenset[SourceIdentity]
    link_target: LinkTarget | None = None
    relations: tuple[EvidenceRelation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.provenance, Provenance):
            raise EvidenceAdapterError("provider source binding and provenance required")
        if type(self.allowed_sources) is not frozenset or any(
            not isinstance(s, SourceIdentity) for s in self.allowed_sources
        ):
            raise EvidenceAdapterError("explicit immutable provider allowance required")
        if self.source not in self.allowed_sources or any(
            s.scope.company_id != self.source.scope.company_id for s in self.allowed_sources
        ):
            raise EvidenceAdapterError("source binding outside provider allowance")
        if self.link_target is not None:
            if not isinstance(self.link_target, LinkTarget):
                raise EvidenceAdapterError("invalid application link reference")
            if self.link_target.record_id != self.source.source_id:
                raise EvidenceAdapterError("application link identity differs from source")
        if type(self.relations) is not tuple or any(
            not isinstance(r, EvidenceRelation) for r in self.relations
        ):
            raise EvidenceAdapterError("immutable provider relations required")
        if any(r.target not in self.allowed_sources for r in self.relations):
            raise EvidenceAdapterError("relation target outside provider allowance")
        if len(set(self.relations)) != len(self.relations):
            raise EvidenceAdapterError("duplicate relation; no implicit merge")

    @property
    def legacy_document_id(self) -> str:
        """Storage-key convention, not a title-based source-type classifier.

        Canonical source_id is the application's unprefixed record identifier.
        The type and the identifier come from the provider, not from this parser.
        """
        if self.source.source_type == SourceType.DOCUMENT:
            return self.source.source_id
        return self.source.source_type.value + ":" + self.source.source_id


@dataclass(frozen=True, slots=True)
class AdaptationTrace:
    layout: str
    text_field: str
    # This is an association only, NOT an exact-text locator or evidence of scope.
    legacy_citation_id: str | None
    consumed_fields: tuple[str, ...]
    unmapped_fields: tuple[str, ...]
    excluded_transport_fields: tuple[str, ...]
    adapter_version: str = ADAPTER_VERSION


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    entry: EvidenceEntry
    trace: AdaptationTrace

    def __post_init__(self) -> None:
        if not isinstance(self.entry, EvidenceEntry) or not isinstance(self.trace, AdaptationTrace):
            raise EvidenceAdapterError("invalid adaptation result")
