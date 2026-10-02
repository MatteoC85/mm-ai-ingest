"""Bounded, source-fair context for Root Cause synthesis; no retrieval or inference.

Only already-selected source records enter. A long record cannot remove later
records from the prompt. Literal prefixes and explicit completeness markers retain
source boundaries. Contained duplicates merge only within identical supplied
company/machine/source/page identity; the longer source keeps its own citation.
This is context allocation, never an evidence-admission or causal-truth decision.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from typing import Any

POLICY_VERSION = "root-synthesis-source-budget-v1"


class SourcePacketError(ValueError):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _identity(candidate: Mapping[str, Any], source_type: str) -> tuple | None:
    # Missing scope metadata must not become equality across unknown owners.
    company = candidate.get("company_id")
    document = candidate.get("bubble_document_id")
    first, last = candidate.get("page_from"), candidate.get("page_to")
    if (not isinstance(company, str) or not company or not isinstance(document, str) or not document
            or "machine_id" not in candidate or type(first) is not int or type(last) is not int
            or first <= 0 or last < first):
        return None
    return company, candidate.get("machine_id"), source_type, document, first, last


def _prefix(body: str, cap: int) -> str:
    if len(body) <= cap:
        return body
    # A contiguous literal prefix ending at an existing word boundary. Explicit
    # text_complete=false means that later conditions may remain undisplayed.
    end = max(body.rfind(" ", 0, cap), body.rfind("\n", 0, cap), body.rfind("\t", 0, cap)) + 1
    return body[:end if end else cap]


def build(*, citations: Sequence[Mapping[str, Any]], max_chars: int,
          candidate_text: Callable[[Mapping[str, Any]], str],
          source_type_from_document_id: Callable[[str], str]) -> dict[str, Any]:
    if type(max_chars) is not int or max_chars <= 0:
        raise SourcePacketError("invalid_root_source_budget")
    prepared = []
    missing = []
    input_count = 0
    for candidate in citations or []:
        input_count += 1
        if not isinstance(candidate, Mapping):
            raise SourcePacketError("malformed_root_source")
        cid = str(candidate.get("citation_id") or "").strip()
        body = candidate_text(candidate)
        if not cid or not isinstance(body, str) or not body.strip():
            missing.append(cid)
            continue
        kind = str(candidate.get("source_type") or source_type_from_document_id(candidate.get("bubble_document_id") or ""))
        record = {"citation_id": cid, "source_type": kind,
                  "document_id": candidate.get("bubble_document_id"),
                  "page_from": candidate.get("page_from"), "page_to": candidate.get("page_to"),
                  "exact_machine_scope": (candidate.get("exact_machine_scope")
                      if type(candidate.get("exact_machine_scope")) is bool else None)}
        # Normalize whitespace only to compare containment; displayed text always
        # remains the chosen source's literal original. Case and units are intact.
        normalized = " ".join(body.split())
        scope = _identity(candidate, kind)
        equivalent = None
        if scope is not None:
            for row in prepared:
                if row["scope"] == scope and (normalized in row["normalized"] or row["normalized"] in normalized):
                    equivalent = row
                    break
        if equivalent is None:
            prepared.append({"record": record, "body": body, "scope": scope,
                             "normalized": normalized, "represented_ids": [cid]})
        else:
            equivalent["represented_ids"].append(cid)
            if len(normalized) > len(equivalent["normalized"]):
                equivalent.update(record=record, body=body, normalized=normalized)
    if not prepared:
        return {"sources_block": "", "summary": {"policy_version": POLICY_VERSION,
                "input_source_count": input_count, "emitted_source_count": 0,
                "evidence_payload_chars": 0, "evidence_payload_limit": max_chars,
                "sources": [], "missing_source_ids": missing}}

    def render(cap: int):
        sources = [{**row["record"], "text": _prefix(row["body"], cap),
                    "text_complete": len(row["body"]) <= cap} for row in prepared]
        packet = {"policy_version": POLICY_VERSION, "sources": sources}
        return packet, _dump(packet)

    largest = max(len(row["body"]) for row in prepared)
    packet, encoded = render(largest)
    if len(encoded) > max_chars:
        packet, encoded = render(1)
        if len(encoded) > max_chars:
            raise SourcePacketError("root_source_metadata_exceeds_budget")
        low, high = 1, largest
        # One shared prefix cap gives each retained record equal access. Short
        # complete sources consume only their actual length; unused room is
        # redistributed to longer records without any query/page-specific rule.
        while low < high:
            middle = (low + high + 1) // 2
            trial, trial_encoded = render(middle)
            if len(trial_encoded) <= max_chars:
                low, packet, encoded = middle, trial, trial_encoded
            else:
                high = middle - 1
    sources = packet["sources"]
    aliases = [{"removed_citation_id": cid, "represented_by_citation_id": row["record"]["citation_id"],
                "reason": "same_scope_source_page_literal_containment"}
               for row in prepared for cid in row["represented_ids"] if cid != row["record"]["citation_id"]]
    return {"sources_block": encoded, "summary": {
        "policy_version": POLICY_VERSION, "input_source_count": input_count,
        "emitted_source_count": len(sources), "deduplicated_source_count": len(aliases),
        "evidence_payload_chars": len(encoded), "evidence_payload_limit": max_chars,
        "truncated_source_count": sum(not row["text_complete"] for row in sources),
        "missing_source_ids": missing, "deduplicated_sources": aliases,
        "sources": [{"citation_id": source["citation_id"],
                     "input_chars": len(row["body"]), "emitted_chars": len(source["text"]),
                     "text_complete": source["text_complete"],
                     "input_sha256": hashlib.sha256(row["body"].encode("utf-8")).hexdigest(),
                     "emitted_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest()}
                    for source, row in zip(sources, prepared)],
    }}
