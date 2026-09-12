"""P6-B4m: explicit lineage for residual ASK source-producing paths.

This module does not activate canonical ASK, change cache policy, build final
citations/links or introduce new readers.  It closes three preparation gaps:

* raw ``document_page`` occurrences can be converted to candidate views only
  through an explicit trusted converter that returns parent occurrence positions;
* structured/procedural callback reads are exposed as request-owned registered
  reads instead of being hidden inside synthesis callbacks;
* precision/scalar rescue can read pages/file metadata through the same evidence
  session and retain the exact selected page occurrence for its candidate view.

Every I/O operation is still one of the existing B1/B2/B3 production readers.
Current authorization is supplied before/after use by the existing B4 owner.
There is no ID/text/score join used to reconstruct provenance after a transform.
Main composition, cache provenance and final citation/link provenance remain
B4o/B4n responsibilities respectively.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .ask_composition import AskEvidenceSession, RecordHandle, RegisteredRead
from .production_adapters import ProductionReaderAdapters
from . import precision_facts
from ..evidence.ask_input import _same_value
from ..evidence.contracts import EvidenceContractError, SourceIdentity, SourceType

B4M_VERSION = "ask-residual-lineage-p6b4m-v1"
PAGE_CONVERSION_VERSION = "document-page-to-candidate-p6b4m-v1"
PRECISION_RESCUE_VERSION = "precision-fact-lineage-p6b4m-v1"


class ResidualEvidenceError(EvidenceContractError):
    """Technical residual-evidence failure; never a successful no_sources result."""


def _callbacks(session: AskEvidenceSession, authorize: Callable, invoke: Callable) -> None:
    if type(session) is not AskEvidenceSession or not callable(authorize) or not callable(invoke):
        raise ResidualEvidenceError("request-owned session and owner callbacks required")


def derive_page_candidates(*, request: Any, session: AskEvidenceSession,
        page_handles: tuple[RecordHandle, ...], converter: Callable[..., Any],
        operation: str, authorize: Callable[[Any], frozenset[SourceIdentity]],
        invoke: Callable[..., Any]) -> tuple[RecordHandle, ...]:
    """Convert authorized raw pages using direct positional lineage.

    ``converter(records)`` receives fresh mutable copies of the ordered raw page
    mappings and must return an immutable tuple of ``(record, parent_positions)``.
    ``parent_positions`` are zero-based positions in that exact input tuple.
    The converter may select, merge or transform pages, but one derived record
    may still only combine occurrences from one source as enforced by B4a.

    The contract deliberately does not accept citation IDs, source IDs, snippets
    or scores as lineage.  Position capture happens where the trusted converter
    creates each result.  Raw input mutation is rejected.
    """
    _callbacks(session, authorize, invoke)
    if type(page_handles) is not tuple or any(type(h) is not RecordHandle for h in page_handles):
        raise ResidualEvidenceError("immutable raw page occurrence handles required")
    if not callable(converter) or type(operation) is not str or not operation.strip():
        raise ResidualEvidenceError("explicit trusted page converter and operation required")

    def work():
        before = invoke(authorize, request, request)
        _, limits = session.read_contract(request=request, current_allowed_sources=before)
        if len(page_handles) > limits.assembly.max_occurrences:
            raise ResidualEvidenceError("raw page conversion input count exceeded")
        inputs = session.records(request=request, handles=page_handles,
                                 current_allowed_sources=before)
        if any(item.layout != "document_page" for item in inputs):
            raise ResidualEvidenceError("document_page inputs required for page conversion")
        records = tuple(deepcopy(dict(item.record)) for item in inputs)
        snapshots = deepcopy(records)
        try:
            converted = invoke(converter, request, records)
        except EvidenceContractError:
            raise
        except Exception:
            raise ResidualEvidenceError("page conversion callback failed") from None
        if not _same_value(records, snapshots):
            raise ResidualEvidenceError("page converter mutated its raw page inputs")
        if type(converted) is not tuple or len(converted) > limits.assembly.max_occurrences:
            raise ResidualEvidenceError("bounded immutable page conversion output required")

        views = []
        for item in converted:
            if type(item) is not tuple or len(item) != 2:
                raise ResidualEvidenceError("page conversion must return record and parent positions")
            record, positions = item
            if type(record) is not dict or type(positions) is not tuple or not positions:
                raise ResidualEvidenceError("page conversion output requires explicit parents")
            parents = []
            for pos in positions:
                if type(pos) is not int or not 0 <= pos < len(page_handles):
                    raise ResidualEvidenceError("page conversion lineage references absent input")
                parents.append(page_handles[pos])
            views.append((tuple(parents), deepcopy(record)))

        # Reauthorize every source after arbitrary trusted conversion work and
        # before retaining any derived view in the request-owned session.
        after = invoke(authorize, request, request)
        session.records(request=request, handles=page_handles,
                        current_allowed_sources=after)
        return session.derive_batch(request=request, views=tuple(views),
            operation=operation, layout="retrieval_candidate",
            current_allowed_sources=after)

    return invoke(work, request)


class ResidualSourceAdapters:
    """Named B4m adapters for reads historically hidden in ASK callbacks.

    Methods return occurrence handles/registered reads, not legacy unbound lists.
    They neither run structured synthesis nor infer which procedure family wins.
    That keeps acquisition explicit while B4o remains responsible for composing
    these sources into the existing algorithms.
    """

    def __init__(self, *, request: Any, session: AskEvidenceSession,
                 readers: ProductionReaderAdapters, authorize: Callable,
                 invoke: Callable):
        _callbacks(session, authorize, invoke)
        if type(readers) is not ProductionReaderAdapters:
            raise ResidualEvidenceError("production reader adapters required")
        self._request = request
        self._session = session
        self._readers = readers
        self._authorize = authorize
        self._invoke = invoke

    def _current(self) -> frozenset[SourceIdentity]:
        current = self._invoke(self._authorize, self._request, self._request)
        self._session.read_contract(request=self._request,
                                    current_allowed_sources=current)
        return current

    def _sources(self, handles: tuple[RecordHandle, ...],
                 expected: SourceType | None = None) -> tuple[SourceIdentity, ...]:
        if type(handles) is not tuple or any(type(h) is not RecordHandle for h in handles):
            raise ResidualEvidenceError("same-session occurrence handles required")
        current = self._current()
        inputs = self._session.records(request=self._request, handles=handles,
                                       current_allowed_sources=current)
        sources = []
        for item in inputs:
            source = item.context.source
            if expected is not None and source.source_type != expected:
                raise ResidualEvidenceError("unexpected source type for residual reader")
            if source not in sources:
                sources.append(source)
        return tuple(sources)

    def structured_direct(self, *, q: str, planner: Optional[dict], top_k: int
                          ) -> tuple[RecordHandle, ...]:
        return self._readers.candidate_handles("read_structured_direct_page_evidence",
                                               q=q, planner=planner, top_k=top_k)

    def deterministic_manual_support(self, *, q: str, planner: Optional[dict],
                                     structured_handles: tuple[RecordHandle, ...]
                                     ) -> tuple[RecordHandle, ...]:
        return self._readers.candidate_handles(
            "read_deterministic_manual_support_page_evidence",
            q=q, planner=planner, structured_handles=structured_handles)

    def parent_procedure_pages(self, *, step_handles: tuple[RecordHandle, ...],
                               text_chars: int) -> RegisteredRead:
        children = self._sources(step_handles, SourceType.STEP)
        if not children:
            raise ResidualEvidenceError("parent procedure read requires Step occurrences")
        return self._readers.read("read_parent_procedure_page_evidence",
                                  children=children, text_chars=text_chars)

    def related_step_pages(self, *, procedure_handle: RecordHandle,
                           text_chars: int) -> RegisteredRead:
        parents = self._sources((procedure_handle,), SourceType.PROCEDURE)
        if len(parents) != 1:
            raise ResidualEvidenceError("related Step read requires one Procedure occurrence")
        return self._readers.read("read_related_step_page_evidence",
                                  parent=parents[0], text_chars=text_chars)

    def step_fallback_pages(self) -> RegisteredRead:
        return self._readers.read("read_step_fallback_page_evidence")

    def precision_pages(self, property_query: precision_facts.PropertyQuery) -> RegisteredRead:
        if type(property_query) is not precision_facts.PropertyQuery:
            raise ResidualEvidenceError("typed precision property query required")
        return self._readers.read("read_precision_page_evidence",
                                  property_query=property_query)

    def document_file_map(self, page_handles: tuple[RecordHandle, ...]) -> dict[str, str]:
        sources = self._sources(page_handles, SourceType.DOCUMENT)
        if not sources:
            return {}
        registered = self._readers.read("read_document_file_references", sources=sources)
        current = self._current()
        # File metadata never grants access; session.file_map rechecks anchors.
        return self._session.file_map(request=self._request, handle=registered.read,
                                      current_allowed_sources=current)


@dataclass(frozen=True, slots=True)
class PrecisionFactEvidence:
    """One legacy resolution plus its canonical request-owned candidate view."""
    resolution: precision_facts.PrecisionFactResolution
    candidate: RecordHandle


def _precision_candidate_view(resolution: precision_facts.PrecisionFactResolution) -> dict[str, Any]:
    candidate = precision_facts.resolution_to_candidate(resolution)
    # document_page observations do not establish a chunk coordinate.  The old
    # rescue used synthetic chunk_index=0 for UI compatibility; canonical lineage
    # must not claim an unobserved chunk.  Legacy path stays unchanged until B4o.
    candidate.pop("chunk_index", None)
    return candidate


def resolve_precision_fact_evidence(*, request: Any, session: AskEvidenceSession,
        sources: ResidualSourceAdapters, query: str, target_machine_id: str,
        page_scan_limit: int, answer_contract: Optional[Mapping[str, Any]],
        authorize: Callable[[Any], frozenset[SourceIdentity]],
        invoke: Callable[..., Any]) -> Optional[PrecisionFactEvidence]:
    """Run the existing scalar selection over registered page/file evidence.

    This mirrors ``resolve_precision_fact`` ordering and ambiguity rules but the
    page provider and optional file-map provider are explicit request-owned B4m
    adapters.  The selected page handle comes from positional lineage emitted by
    ``precision_facts`` during scalar selection, never a post-hoc ID/text match.
    Final citation/link construction is intentionally left to B4n.
    """
    _callbacks(session, authorize, invoke)
    if type(sources) is not ResidualSourceAdapters:
        raise ResidualEvidenceError("typed residual source adapters required")
    if type(query) is not str or type(target_machine_id) is not str:
        raise ResidualEvidenceError("literal query and resolved machine selector required")
    if type(page_scan_limit) is not int or page_scan_limit < 1:
        raise ResidualEvidenceError("positive precision page scan limit required")

    def read_one(prop: precision_facts.PropertyQuery):
        registered = sources.precision_pages(prop)
        current = invoke(authorize, request, request)
        inputs = session.records(request=request, handles=registered.records,
                                 current_allowed_sources=current)
        if any(item.layout != "document_page" for item in inputs):
            raise ResidualEvidenceError("precision reader must return document_page evidence")
        pages = [deepcopy(dict(item.record)) for item in inputs]
        handles = list(registered.records)
        # Same fail-closed truncation rule as legacy resolve_precision_fact.
        if len(pages) >= max(20, page_scan_limit):
            return "truncated", None, None

        if precision_facts._strong_query_codes(prop):
            file_map = sources.document_file_map(tuple(handles))
            filtered_pages, filtered_handles = [], []
            for page, handle in zip(pages, handles):
                key = str(page.get("bubble_document_id") or "")
                if precision_facts._source_matches_query_codes(
                    query=prop,
                    source_hint=str(file_map.get(key) or ""),
                    page_text=str(page.get("text") or page.get("page_text") or ""),
                ):
                    filtered_pages.append(page)
                    filtered_handles.append(handle)
            pages, handles = filtered_pages, filtered_handles

        pairs = precision_facts.matching_scoped_fact_pairs_with_lineage(
            pages, prop, target_machine_id
        )
        if not pairs:
            return "empty", None, None
        chosen = precision_facts.choose_unambiguous_fact_with_lineage(pairs, prop)
        if chosen is None:
            return "ambiguous", None, None
        resolution, page_position = chosen
        if not 0 <= page_position < len(handles):
            raise ResidualEvidenceError("precision lineage references absent page occurrence")
        return "resolved", resolution, handles[page_position]

    def derive(resolution, parent):
        after = invoke(authorize, request, request)
        session.records(request=request, handles=(parent,), current_allowed_sources=after)
        handle = session.derive(request=request, parents=(parent,),
            record=_precision_candidate_view(resolution), operation=PRECISION_RESCUE_VERSION,
            layout="retrieval_candidate", current_allowed_sources=after)
        return PrecisionFactEvidence(resolution, handle)

    def work():
        seen: set[tuple[str, ...]] = set()
        first = (
            precision_facts.extract_property_query(query),
            precision_facts.property_query_from_answer_contract(query, answer_contract),
        )
        for prop in first:
            if prop is None or prop.property_terms in seen:
                continue
            seen.add(prop.property_terms)
            state, resolution, parent = read_one(prop)
            if state == "truncated" or state == "ambiguous":
                return None
            if state == "resolved":
                return derive(resolution, parent)

        resolutions = []
        for prop in precision_facts.property_queries_from_scalar_target(query, answer_contract):
            if prop.property_terms in seen:
                continue
            seen.add(prop.property_terms)
            state, resolution, parent = read_one(prop)
            if state == "truncated" or state == "ambiguous":
                return None
            if state == "resolved":
                resolutions.append((resolution, parent))
        if not resolutions:
            return None
        values = {(r.canonical_number, r.canonical_unit) for r, _ in resolutions}
        if len(values) != 1:
            return None
        resolution, parent = sorted(resolutions,
            key=lambda item: (-item[0].score, item[0].bubble_document_id,
                              item[0].page_number))[0]
        return derive(resolution, parent)

    return invoke(work, request)
