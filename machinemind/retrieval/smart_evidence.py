"""Lossless, bounded Smart evidence carried inside the already signed state.

Only the server's admitted raw retrieval producer may call build/enrichment.
Fingerprinting detects mutation; it is NOT authorization or a replacement for
state HMAC verification. Legacy display snippets cannot reconstruct this packet.
No I/O, model calls, relevance scoring, title-derived evidence or text clipping.
Selection precedes display projection and can coalesce only identical full bodies.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

VERSION = "smart-grounding-packet-v1"
MAX_CHARS = 22000
MAX_SOURCES = 14
MAX_NEW_SOURCES = 3
MAX_SELECTED_SOURCES = 8
_SOURCE_KEYS = {
    "citation_id", "bubble_document_id", "source_id", "source_type",
    "page_from", "page_to", "machine_relation", "body_id", "start", "end",
}
_PACKET_KEYS = {"version", "scope", "max_chars", "sources", "bodies", "fingerprint"}


class SmartEvidenceError(ValueError):
    """Fixed diagnostic code; caller must fail closed or explicitly defer enrichment."""


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise SmartEvidenceError("evidence_not_serializable") from None


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SmartEvidenceError(code)


def _identifier(value: Any, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    _require(isinstance(value, str) and len(value) <= 512
             and value == value.strip() and all(ord(c) >= 32 for c in value)
             and (optional or bool(value)), "evidence_identity_invalid")
    return value


def _scope(value: Any) -> dict:
    _require(isinstance(value, dict), "evidence_scope_invalid")
    result = {k: _identifier(value.get(k)) for k in ("company_id", "machine_id", "ai_scope")}
    _require(result["ai_scope"] == "machine_all", "evidence_scope_not_machine")
    return result


def _limit(value: Any) -> int:
    _require(type(value) is int and 0 < value <= MAX_CHARS, "evidence_limit_invalid")
    return value


def _page(value: Any) -> int:
    if value is None:
        return 0
    _require(type(value) is int and 0 <= value <= 1000000, "evidence_page_invalid")
    return value


def _normalize(row: Any, scope: dict, allow_raw_snippet: bool) -> tuple[dict, str]:
    _require(isinstance(row, dict), "evidence_row_invalid")
    company = _identifier(row.get("company_id"), optional=True)
    machine = _identifier(row.get("machine_id"), optional=True)
    _require(not company or company == scope["company_id"], "evidence_company_mismatch")
    _require(not machine or machine == scope["machine_id"], "evidence_machine_mismatch")
    keys = ("chunk_full", "text", "snippet") if allow_raw_snippet else ("chunk_full", "text")
    body = next((row[k] for k in keys if isinstance(row.get(k), str) and row[k].strip()), None)
    _require(body is not None, "full_admitted_text_required")
    # No .strip(), whitespace normalization, prefix/suffix window or body limit.
    relation = ("selected_machine" if machine == scope["machine_id"]
                or row.get("exact_machine_scope") is True else
                "company_general" if "machine_id" in row and not machine else "not_established")
    source = {
        "citation_id": _identifier(row.get("citation_id")),
        "bubble_document_id": _identifier(row.get("bubble_document_id"), optional=True),
        "source_id": _identifier(row.get("source_id"), optional=True),
        "source_type": _identifier(row.get("source_type"), optional=True),
        "page_from": _page(row.get("page_from")), "page_to": _page(row.get("page_to")),
        "machine_relation": relation,
    }
    _require(bool(source["bubble_document_id"] or source["source_id"]), "evidence_owner_missing")
    _require(source["page_to"] >= source["page_from"], "evidence_page_range_invalid")
    return source, body


def _owner(source: dict) -> str:
    return canonical({k: v for k, v in source.items()
                      if k not in {"citation_id", "body_id", "start", "end"}})


def select_complete_sources(raw_rows: list[dict], *, scope: dict, max_items: int,
                            allow_raw_snippet: bool = False, relevant_ids=()) -> list[dict]:
    """Select ordered raw representatives without comparing display projections.

    Only a server-admitted producer can supply these rows and opt into snippet
    as its full-record field, exactly as for build(). A shared title, first 520
    characters, normalized text or containment never makes records equivalent.
    Exact aliases with identical provenance/body keep the first original ID;
    selection occurs before planner/review references exist. Incoming signed
    state's old IDs must instead remain intact through update_enrichment().

    Validate the complete input before the bounded selection: conflicting uses
    of one ID or a foreign row cannot hide after an alias or the selected cap.
    The existing serialized/expanded 22k limit remains enforced by build().
    """
    bound = _scope(scope)
    _require(type(max_items) is int and 0 <= max_items <= MAX_SELECTED_SOURCES,
             "evidence_selection_limit_invalid")
    _require(type(allow_raw_snippet) is bool, "evidence_raw_mode_invalid")
    _require(isinstance(raw_rows, list), "evidence_row_invalid")
    _require(isinstance(relevant_ids, (list, tuple, set, frozenset)), "evidence_priority_invalid")
    priority = {_identifier(cid) for cid in relevant_ids}
    by_id: dict[str, tuple[str, str]] = {}
    validated = []
    for row in raw_rows:
        source, body = _normalize(row, bound, allow_raw_snippet)
        cid = source["citation_id"]
        complete = (_owner(source), body)
        if cid in by_id:
            _require(by_id[cid] == complete, "evidence_duplicate_citation_conflict")
        by_id[cid] = complete
        validated.append((row, cid, complete))
    # Trusted router IDs affect priority only, never admission. Stable order
    # within both groups preserves the retrieval producer's existing ranking.
    validated.sort(key=lambda item: item[1] not in priority)
    seen_complete: set[tuple[str, str]] = set()
    selected: list[dict] = []
    for row, cid, complete in validated:
        if complete in seen_complete:
            continue
        seen_complete.add(complete)
        if len(selected) < max_items:
            selected.append(deepcopy(row))
    return selected


def _text(packet: dict, source: dict) -> str:
    body = packet["bodies"][source["body_id"]]["text"]
    return body[source["start"]:source["end"]]


def _model(packet: dict) -> dict:
    scope = packet["scope"]
    return {
        "policy_version": VERSION,
        "request_context": {
            "origin": "application_request_scope", "company_id": scope["company_id"],
            "ai_scope": "machine_all", "selected_machine_present": True,
            "selected_machine_id": scope["machine_id"], "document_ids": [],
            "bubble_document_id": None, "component_identity_asserted": False,
        },
        "sources": [{
            "citation_id": s["citation_id"], "document_id": s["bubble_document_id"],
            "source_id": s["source_id"], "source_type": s["source_type"],
            "page_from": s["page_from"], "page_to": s["page_to"],
            "machine_relation": s["machine_relation"], "text": _text(packet, s),
            "text_complete": True, "context_ids": [], "context_status": "admitted_record_only",
        } for s in packet["sources"]],
        "document_contexts": [],
    }


def _check_capacity(packet: dict) -> None:
    limit = packet["max_chars"]
    _require(len(canonical(packet)) <= limit, "evidence_packet_capacity_exceeded")
    # Expanding aliases for the existing review registry is also bounded, not
    # an unaccounted larger prompt hidden behind compressed state storage.
    _require(len(canonical(_model(packet))) <= limit, "evidence_review_capacity_exceeded")


def build(raw_citations: list[dict], *, scope: dict, max_chars: int = MAX_CHARS,
          allow_raw_snippet: bool = False) -> dict:
    """Freeze ALL caller-admitted bodies/IDs, or raise; never select/drop a row.

    allow_raw_snippet is a SERVER call-site assertion that a raw producer uses
    the field name snippet for its full admitted body. Never derive it from
    payload metadata or use it on sanitized/state/UI citation projections.
    Completeness means the full admitted record, not an entire manual/document.
    """
    bound = _scope(scope)
    limit = _limit(max_chars)
    _require(type(allow_raw_snippet) is bool, "evidence_raw_mode_invalid")
    _require(isinstance(raw_citations, list) and 1 <= len(raw_citations) <= MAX_SOURCES,
             "evidence_source_count_invalid")
    rows = [_normalize(row, bound, allow_raw_snippet) for row in raw_citations]
    ids = [source["citation_id"] for source, _ in rows]
    _require(len(ids) == len(set(ids)), "evidence_duplicate_citation_id")
    # Exact containment only within identical provenance. Shared storage removes
    # no source text: each original citation retains exact contiguous offsets.
    bodies: list[dict] = []
    assignments: dict[int, tuple[int, int, int]] = {}
    for index in sorted(range(len(rows)), key=lambda i: (-len(rows[i][1]), i)):
        source, text = rows[index]
        owner = _owner(source)
        match = next(((i, b["text"].find(text)) for i, b in enumerate(bodies)
                      if b["owner"] == owner and text in b["text"]), None)
        if match is None:
            match = (len(bodies), 0)
            bodies.append({"owner": owner, "text": text})
        assignments[index] = (match[0], match[1], match[1] + len(text))
    sources = [{**source, "body_id": assignments[i][0], "start": assignments[i][1],
                "end": assignments[i][2]} for i, (source, _) in enumerate(rows)]
    packet = {"version": VERSION, "scope": bound, "max_chars": limit,
              "sources": sources, "bodies": bodies}
    packet["fingerprint"] = _digest(packet)
    _check_capacity(packet)
    return packet


def validate(packet: Any, *, scope: dict, allowed_ids) -> dict:
    """Call only after verifying the containing state HMAC and request binding."""
    _require(isinstance(packet, dict) and packet.get("version") == VERSION,
             "grounding_packet_restart_required")
    _require(set(packet) == _PACKET_KEYS, "evidence_packet_schema_invalid")
    bound = _scope(scope)
    _require(packet.get("scope") == bound, "evidence_scope_mismatch")
    _limit(packet.get("max_chars"))
    _require(isinstance(allowed_ids, (list, tuple, set, frozenset)), "evidence_allowed_ids_invalid")
    admitted = [_identifier(v) for v in allowed_ids]
    _require(len(admitted) == len(set(admitted)), "evidence_allowed_ids_invalid")
    sources, bodies = packet.get("sources"), packet.get("bodies")
    _require(isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES
             and isinstance(bodies, list) and 1 <= len(bodies) <= len(sources),
             "evidence_source_count_invalid")
    seen, used = set(), set()
    for body in bodies:
        _require(isinstance(body, dict) and set(body) == {"owner", "text"}
                 and isinstance(body["owner"], str) and isinstance(body["text"], str)
                 and bool(body["text"].strip()), "evidence_body_invalid")
    for source in sources:
        _require(isinstance(source, dict) and set(source) == _SOURCE_KEYS, "evidence_source_schema_invalid")
        cid = _identifier(source["citation_id"])
        _require(cid not in seen, "evidence_duplicate_citation_id")
        seen.add(cid)
        for key in ("bubble_document_id", "source_id", "source_type"):
            _identifier(source[key], optional=True)
        _require(bool(source["bubble_document_id"] or source["source_id"]), "evidence_owner_missing")
        _require(_page(source["page_to"]) >= _page(source["page_from"]), "evidence_page_range_invalid")
        _require(source["machine_relation"] in {"selected_machine", "company_general", "not_established"},
                 "evidence_relation_invalid")
        index, start, end = source["body_id"], source["start"], source["end"]
        _require(type(index) is int and 0 <= index < len(bodies), "evidence_body_reference_invalid")
        _require(type(start) is int and type(end) is int and 0 <= start < end <= len(bodies[index]["text"]),
                 "evidence_offsets_invalid")
        _require(bodies[index]["owner"] == _owner(source), "evidence_owner_mismatch")
        used.add(index)
    _require(seen == set(admitted), "evidence_allowed_ids_mismatch")
    _require(used == set(range(len(bodies))), "evidence_orphan_body")
    _check_capacity(packet)
    _require(isinstance(packet.get("fingerprint"), str)
             and packet["fingerprint"] == _digest({k: v for k, v in packet.items() if k != "fingerprint"}),
             "evidence_packet_altered")
    return deepcopy(packet)


def _validated(packet: dict) -> dict:
    _require(isinstance(packet, dict), "grounding_packet_restart_required")
    sources = packet.get("sources")
    _require(isinstance(sources, list), "grounding_packet_restart_required")
    ids = [s.get("citation_id") for s in sources if isinstance(s, dict)]
    return validate(packet, scope=packet.get("scope"), allowed_ids=ids)


def review_packet(packet: dict) -> dict:
    """Return the exact full source projection accepted by review_references.prepare."""
    valid = _validated(packet)
    model = _model(valid)
    records = [{
        "citation_id": s["citation_id"], "source_type": s["source_type"],
        "page_from": s["page_from"], "page_to": s["page_to"], "text": s["text"],
        "ownership_fragments": [], "ownership_context": "",
        "context_status": "admitted_record_only", "machine_relation": s["machine_relation"],
    } for s in model["sources"]]
    return {"model_packet": model, "model_json": canonical(model), "validator_records": records,
            "summary": {"policy_version": VERSION, "source_count": len(records),
                "unique_body_count": len(valid["bodies"]), "packet_chars": len(canonical(valid)),
                "evidence_payload_chars": len(canonical(model)), "evidence_payload_limit": valid["max_chars"],
                "packet_fingerprint": valid["fingerprint"], "complete_admitted_records": True,
                "full_document_hydrated": False, "semantic_coverage_proven": False}}


def evidence_block(packet: dict) -> str:
    """One representation for START, ANSWER and FINALIZE; no display titles/snippets."""
    return review_packet(packet)["model_json"]


def update_enrichment(old_packet: dict, admitted_newrows: list[dict], *, scope: dict,
                      allowed_ids=None, allow_raw_snippet: bool = False,
                      max_new_sources: int = MAX_NEW_SOURCES) -> dict:
    """Append authorized new records atomically; retain every old ID and full body."""
    _require(type(max_new_sources) is int and 0 <= max_new_sources <= MAX_NEW_SOURCES,
             "evidence_enrichment_limit_invalid")
    old = _validated(old_packet)
    _require(old["scope"] == _scope(scope), "evidence_scope_mismatch")
    _require(isinstance(admitted_newrows, list), "evidence_row_invalid")
    reconstructed = []
    old_by_id = {}
    for source in old["sources"]:
        row = {k: source[k] for k in ("citation_id", "bubble_document_id", "source_id",
                                      "source_type", "page_from", "page_to")}
        row["chunk_full"] = _text(old, source)
        if source["machine_relation"] == "selected_machine":
            row["machine_id"] = old["scope"]["machine_id"]
        elif source["machine_relation"] == "company_general":
            row["machine_id"] = ""
        reconstructed.append(row)
        old_by_id[source["citation_id"]] = (_owner(source), row["chunk_full"])
    additions = []
    seen = set(old_by_id)
    for row in admitted_newrows:
        normalized, body = _normalize(row, old["scope"], allow_raw_snippet)
        cid = normalized["citation_id"]
        if cid in old_by_id:
            _require(old_by_id[cid] == (_owner(normalized), body), "evidence_existing_source_changed")
            continue
        _require(cid not in seen, "evidence_duplicate_citation_id")
        seen.add(cid)
        additions.append(row)
    _require(len(additions) <= max_new_sources, "evidence_enrichment_source_limit")
    if allowed_ids is not None:
        _require(isinstance(allowed_ids, (list, tuple, set, frozenset)), "evidence_allowed_ids_invalid")
        _require(set(old_by_id) <= set(allowed_ids), "evidence_selection_lost_old_sources")
        _require(set(allowed_ids) == seen, "evidence_allowed_ids_mismatch")
    updated = build(reconstructed + additions, scope=scope, max_chars=old["max_chars"],
                    allow_raw_snippet=allow_raw_snippet)
    return validate(updated, scope=scope, allowed_ids=seen)
