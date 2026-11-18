"""Exact one-field document proposals. A proposal always needs independent ASK review."""
from __future__ import annotations

import re
import unicodedata

from ..evidence.contracts import SourceType
from ..retrieval import precision_facts
from ..retrieval.page_evidence import PageReadObservation

MODEL = "deterministic_document_scalar_draft_v1"
_QUESTION = {
    "en": r"What is the (?P<label>[^?]+)\?",
    "it": r"Qual (?:è|e') (?:il |lo |la |l['’])?(?P<label>[^?]+)\?",
}


def _text(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def target(request, contract, plan):
    """Use the router's bound scalar intent and consume the complete exact-label question."""
    if (request.requested_mode != "ask" or request.allowed_effective_modes != ("ask",)
            or request.narrow_scope is not True or plan.get("request_kind") != "factual"
            or contract.get("information_task") != "numeric_specification"
            or contract.get("required_answer_types") != ["numeric_value"]
            or contract.get("procedure_scope") not in (None, "", "other")
            or contract.get("procedure_phase") or contract.get("missing_information")):
        return None
    facets = contract.get("facet_queries")
    if (type(facets) is not list or len(facets) != 1 or type(facets[0]) is not dict
            or facets[0].get("must_cover") is not True
            or facets[0].get("answer_type") != "numeric_value"
            or len(contract.get("required_facets") or ()) != 1):
        return None
    pattern = _QUESTION.get(request.response_language)
    query = request.query.strip()
    match = re.fullmatch(pattern, query, flags=re.IGNORECASE) if pattern and len(query) <= 125 else None
    if match is None:
        return None
    label = _text(match["label"])
    if not re.fullmatch(r"[^\W\d_]+(?:[ '’\-][^\W\d_]+){1,7}", label):
        return None
    scalar = request.metadata.get(precision_facts.SCALAR_TARGET_KEY)
    validated = precision_facts.property_queries_from_scalar_target(query, {
        **contract, "effective_mode": "ask", "evidence_policy": "machine_sources_required",
        precision_facts.SCALAR_TARGET_KEY: scalar,
    })
    property_query = precision_facts.extract_property_query(label)
    if not validated or property_query is None or not any(
        item.property_terms == property_query.property_terms for item in validated
    ):
        return None
    if not any(_text(item).casefold() == label.casefold() for item in scalar["property_labels"]):
        return None
    return label, property_query


def proposal(*, marker, document_id, candidate_pages, emitted_message):
    """Select a literal, unqualified numeric label/value line from typed page receipts."""
    if not 1 <= len(candidate_pages) <= 16:
        return None
    label, property_query = marker
    emitted_text = _text(emitted_message)
    pages = []
    for row, observation in candidate_pages:
        if (type(observation) is not PageReadObservation or observation.source.source_type != SourceType.DOCUMENT
                or observation.storage_document_id != document_id
                or not 0 < len(observation.text_projection) < observation.projection_chars
                or _text(observation.text_projection) not in emitted_text
                or row.get("bubble_document_id") != document_id
                or row.get("page_from") != observation.page_number
                or row.get("page_to") != observation.page_number
                or not row.get("citation_id")):
            return None
        pages.append({"bubble_document_id": document_id, "machine_id": observation.stored_machine_id,
                      "page_number": observation.page_number, "text": observation.text_projection})
    matches = precision_facts.matching_scoped_fact_pairs_with_lineage(
        pages, property_query, pages[0]["machine_id"])
    if len({(pair.canonical_number, pair.canonical_unit) for pair, _ in matches}) != 1:
        return None
    emitted_lines = {_text(line) for line in emitted_message.splitlines()}
    for pair, index in matches:
        if not pair.same_line or _text(pair.label).casefold() != label.casefold():
            continue
        line = f"{pair.label}: {pair.value}"
        row, observation = candidate_pages[index]
        normalized = _text(line)
        if normalized in emitted_lines and normalized in {
                _text(value) for value in observation.text_projection.splitlines()}:
            return {"answer_status": "answered", "grounded_points": [
                {"text": line + ".", "citation_ids": [row["citation_id"]]},
            ]}
    return None
