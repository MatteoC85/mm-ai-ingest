"""A bounded source packet for independent review of an admitted procedure.

This module selects/renders already admitted records. It neither grants access
nor discovers evidence. The owning ResponseEvidenceFlow keeps occurrence handles
and rechecks their current authority before/after each model boundary. A complete
packet is a representation invariant, NEVER a semantic coverage verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
from typing import Callable

from ..evidence.contracts import EvidenceContractError
from ..infrastructure.request_budget import _V13BudgetExceeded

VERSION = "ask-procedure-review-evidence-v1"


class ReviewEvidenceError(EvidenceContractError):
    """Invalid/incomplete representation; must not be relabelled source absence."""


def complete_limits(primary, *, render, max_records, max_context_chars, max_bytes):
    """Preserve selected source bodies within canonical bytes and provider cost caps."""
    required, _ = _unique(primary, [])
    chars = size = 0
    for row in required:
        part = render([row], max_context_chars=sys.maxsize)
        if not isinstance(part, str) or not part:
            raise ReviewEvidenceError("producer review record has no rendered body")
        if chars:
            chars += 2
            size += 2
        chars += len(part)
        size += len(part.encode("utf-8"))
        if size > max_bytes:
            raise _V13BudgetExceeded("procedure_review_evidence_capacity")
    return max(max_records, len(required)), max(max_context_chars, chars)


@dataclass(frozen=True, slots=True)
class ReviewEvidencePacket:
    # Records are occurrence-bound references, not frozen/authorizing dicts. The
    # flow owns their witnesses; any mutation is rejected before reuse.
    rows: tuple[dict, ...]
    sources: str
    required_count: int
    available_count: int
    unique_count: int
    max_records: int
    max_context_chars: int

    def diagnostic(self, *, reused: bool) -> dict:
        return {
            "version": VERSION,
            "basis": "admitted_synthesis_manifest",
            "source_sha256": hashlib.sha256(self.sources.encode("utf-8")).hexdigest(),
            "source_characters": len(self.sources),
            "required_records": self.required_count,
            "required_citation_ids": [str(row["citation_id"]) for row in self.rows[:self.required_count]],
            "emitted_citation_ids": [str(row["citation_id"]) for row in self.rows],
            "emitted_required_records": self.required_count,
            "available_occurrences": self.available_count,
            "unique_citation_views": self.unique_count,
            "emitted_records": len(self.rows),
            "deferred_citation_views": self.unique_count - len(self.rows),
            "max_records": self.max_records,
            "max_context_chars": self.max_context_chars,
            "reused_request_packet": reused,
            "complete_primary_representation": True,
            "semantic_coverage_proven": False,
        }


def _unique(primary: list[dict], extension: list[dict]) -> tuple[list[dict], list[dict]]:
    """Prefer the producer's full view, not a retrieval/citation projection.

    Citation IDs decide display duplicates ONLY. The caller has already checked
    the explicit handles (including the discarded display alternatives). A
    duplicate does not acquire permissions or let a different source own a body.
    """
    if type(primary) is not list or type(extension) is not list:
        raise ReviewEvidenceError("review packet requires explicit record lists")
    order: list[str] = []
    views: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for row in extension + primary:
        if type(row) is not dict:
            raise ReviewEvidenceError("review packet record must be a dictionary")
        cid = str(row.get("citation_id") or "").strip()
        if not cid:
            raise ReviewEvidenceError("review packet record lacks citation identity")
        owner = str(row.get("bubble_document_id") or "").strip()
        if cid in owners and owners[cid] != owner:
            raise ReviewEvidenceError("review citation aliases different source records")
        owners[cid] = owner
        if cid not in views:
            order.append(cid)
        views[cid] = row
    primary_ids = list(dict.fromkeys(str(r["citation_id"]).strip() for r in primary))
    needed = set(primary_ids)
    return [views[c] for c in primary_ids], [views[c] for c in order if c not in needed]


def compile_packet(*, primary: list[dict], extension: list[dict],
                   render: Callable[..., str], max_records: int,
                   max_context_chars: int) -> ReviewEvidencePacket:
    """Reserve complete producer records before the existing optional prefix.

    The old renderer can silently cut a first oversized record or omit the tail
    of a list. Individual rendering below is measurement only; the original
    model-context/count limits are enforced on the *entire* resulting packet.
    No Step/body is shortened to make a required record fit. Extensions not in
    this bounded display stay in the owner's original evidence/session.
    """
    if type(max_records) is not int or type(max_context_chars) is not int or min(max_records, max_context_chars) < 1:
        raise ReviewEvidenceError("invalid review packet limits")
    required, optional = _unique(primary, extension)
    if not required:
        raise ReviewEvidenceError("review packet has no producer manifest")
    if len(required) > max_records:
        raise ReviewEvidenceError("complete review manifest exceeds record capacity")
    rows: list[dict] = []
    parts: list[str] = []
    for required_record, group in ((True, required), (False, optional)):
        for row in group:
            if len(rows) >= max_records:
                break
            # sys.maxsize applies ONLY to length measurement, not to a prompt.
            part = render([row], max_context_chars=sys.maxsize)
            if not isinstance(part, str):
                raise ReviewEvidenceError("review renderer returned non-text")
            if not part:
                if required_record:
                    raise ReviewEvidenceError("producer review record has no rendered body")
                continue
            if len("\n\n".join(parts + [part])) > max_context_chars:
                if required_record:
                    raise ReviewEvidenceError("complete review manifest exceeds context capacity")
                break  # preserve the existing optional-prefix policy
            parts.append(part)
            rows.append(row)
    expected = "\n\n".join(parts)
    rendered = render(rows, max_context_chars=max_context_chars)
    # Detect any secondary prefix/truncation or incompatible rendering instead
    # of issuing a prompt whose advertised source records were never visible.
    if not expected or rendered != expected or len(rendered) > max_context_chars:
        raise ReviewEvidenceError("review source serialization changed selected records")
    return ReviewEvidencePacket(tuple(rows), rendered, len(required),
        len(primary) + len(extension), len(required) + len(optional),
        max_records, max_context_chars)
