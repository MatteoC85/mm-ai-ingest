"""Request-local final ASK verdicts and lossless source-owned safety supplements.

This module neither grants authority nor retrieves evidence. Callers supply only
already-admitted occurrences. The semantic verdict applies to the exact reviewed
answer; deterministic diagnostics stay separate. Source-owned notes are quoted,
not generated or silently translated, and carry their original citation identity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import re
from typing import Callable, Any

from ..evidence.contracts import EvidenceContractError

VERSION = "ask-final-contract-v1"


def answer_digest(answer: str) -> str:
    return hashlib.sha256(str(answer or "").strip().encode("utf-8")).hexdigest()



@dataclass(frozen=True, slots=True)
class ReviewedDraft:
    """Server-owned input snapshot, never a model/client-supplied hash or grant.

    Strings and tuples are immutable. The request key is the existing immutable
    ASK key and is checked again after the provider. This object lives only on
    the verifier's stack; it is not exported, cached, or reused by another ASK.
    """
    answer: str = field(repr=False)
    request_key: Any = field(repr=False)
    required_facets: tuple[str, ...]
    required_types: tuple[str, ...]


def capture_reviewed_draft(answer: str, *, request_key: Any,
                           required_facets: tuple[str, ...],
                           required_types: tuple[str, ...]) -> ReviewedDraft:
    # Recovery-from-evidence intentionally starts with an empty CURRENT_ANSWER.
    # Capture it too: a nonempty rewrite remains legal, but there is no existing
    # body to approve via an empty PASS (checked at resolution, not pre-call).
    if type(answer) is not str:
        raise EvidenceContractError("reviewed_draft_requires_text")
    for values in (required_facets, required_types):
        if type(values) is not tuple or any(type(x) is not str for x in values):
            raise EvidenceContractError("reviewed_draft_requires_immutable_requirements")
    return ReviewedDraft(answer, request_key, required_facets, required_types)


def resolve_unchanged_pass(parsed: dict, *, draft: ReviewedDraft,
                           request_key: Any) -> dict:
    """Resolve only an explicit, fully covered PASS with no replacement text.

    A rewrite must still contain a replacement. An incomplete or malformed PASS
    is not promoted by copying whatever answer happens to be current. The only
    permitted body comes from the exact immutable input captured at the call.
    Enumeration is checked again by the existing verifier after resolution;
    grounding, authority, the final text binding and cache accounting are not
    replaced by this local protocol adapter.
    """
    if type(draft) is not ReviewedDraft or draft.request_key != request_key:
        raise EvidenceContractError("reviewed_draft_request_changed")
    out = deepcopy(parsed)
    # A provider cannot manufacture the server's binding metadata.
    out.pop("reviewed_draft_binding", None)
    outcome = str(out.get("outcome") or "").strip().lower()
    if outcome not in {"pass", "rewrite"}:
        return out
    body = out.get("answer")
    if type(body) is not str:
        raise EvidenceContractError("reviewed_draft_success_text_invalid")
    if body.strip():
        return out  # Existing full-answer and compact-procedure paths unchanged.
    if outcome == "rewrite":
        raise EvidenceContractError("reviewed_draft_empty_rewrite")
    if not draft.answer.strip():
        raise EvidenceContractError("reviewed_draft_empty_pass_without_draft")
    required_lists = ("covered_facets", "covered_answer_types", "missing_facets",
                      "missing_answer_types", "expected_list_items",
                      "covered_list_items", "missing_list_items")
    for key in required_lists:
        values = out.get(key)
        if type(values) is not list or any(type(x) is not str for x in values):
            raise EvidenceContractError("reviewed_draft_coverage_shape_invalid")
    if any(out[key] for key in ("missing_facets", "missing_answer_types", "missing_list_items")):
        raise EvidenceContractError("reviewed_draft_success_has_missing_requirements")
    if (not set(draft.required_facets).issubset(out["covered_facets"])
            or not set(draft.required_types).issubset(
                str(x).strip().lower() for x in out["covered_answer_types"])
            or not set(out["expected_list_items"]).issubset(out["covered_list_items"])):
        raise EvidenceContractError("reviewed_draft_coverage_not_complete")
    out["answer"] = draft.answer
    out["reviewed_draft_binding"] = {
        "version": "ask-reviewed-draft-v1",
        "basis": "server_snapshot_at_verifier_call",
        "wire_answer_empty": True,
        "input_answer_sha256": hashlib.sha256(draft.answer.encode("utf-8")).hexdigest(),
        "input_answer_utf8_bytes": len(draft.answer.encode("utf-8")),
        "resolved_answer_sha256": answer_digest(draft.answer),
        "coverage_reported_complete": True,
    }
    return out


def bind_source_fields(runtime: Any, *, fields: Callable, sections: Callable):
    if not callable(fields) or not callable(sections):
        raise TypeError("structured source readers required")
    return replace(runtime, final_contract_enabled=True, source_fields=fields, source_sections=sections)


def source_safety_notes(candidates: list[dict], *, fields: Callable,
                        sections: Callable, source_type: Callable) -> tuple[dict, ...]:
    """Project notes from admitted structured fields, never user text or a model.

    No new parsers, keyword heuristics, source IDs or fixed step counts. The
    existing ingestion/presentation field readers define the source schema.
    Notes are not excerpted or limited by presentation point/character caps.
    """
    out = []
    seen = set()
    for candidate in candidates:
        if source_type(candidate) != "step":
            continue
        cid = str(candidate.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        data = fields(candidate)
        parts = sections(str(data.get("description") or ""))
        safety = str(parts.get("safety") or "").strip()
        # A structured notes field also belongs to this Step; it is not a new
        # action and must not be discarded by a prose-format preference.
        notes = str(data.get("notes") or "").strip()
        contents = tuple(dict.fromkeys(x for x in (safety, notes) if x))
        if not contents:
            continue
        ordinal = str(data.get("step_number") or "").strip()
        out.append({"citation_id": cid, "source_number": ordinal,
                    "source_title": str(data.get("title") or "").strip(),
                    "texts": contents})
        seen.add(cid)
    return tuple(out)


def source_note_present(answer: str, note: str) -> bool:
    """Literal note retention, allowing only prose sentence-initial casing.

    Do NOT casefold technical text: mA/MA, mW/MW and identifiers are distinct.
    A lowercase initial prose word may match its title-cased source spelling;
    the remainder, negation, numbers, punctuation and units stay exact.
    """
    body, text = " ".join(str(answer).split()), " ".join(str(note).split())
    if text in body:
        return True
    word = re.match(r"[^\W\d_]{3,}\b", text, flags=re.UNICODE)
    if not word or not word.group().istitle():
        return False
    variant = text[0].lower() + text[1:]
    # A substring inside a different word is not a retained source note.
    return re.search(r"(?<!\w)" + re.escape(variant), body) is not None


def preserve_source_notes(answer: str, notes: tuple[dict, ...], *,
                          language: str, sentence_initial_equivalence: bool = False) -> tuple[str, dict]:
    """Retain exact source wording if a reviewer omitted or translated a note.

    The separate, labelled source quotation cannot be mistaken for a model
    translation. Whitespace-only differences are equivalent; no fuzzy semantic
    matcher or model permission may erase a source warning.
    """
    answer = str(answer or "").strip()
    original_answer = answer
    normal = " ".join(answer.split())
    present = (source_note_present if sentence_initial_equivalence else
               lambda a, t: " ".join(str(t).split()) in " ".join(str(a).split()))
    missing = []
    for note in notes:
        texts = [str(x) for x in note["texts"] if not present(answer, str(x))]
        if texts:
            missing.append((note, texts))
    english = str(language).lower().startswith("en")
    blocks = []
    for note, texts in missing:
        label = ("Step " if english else "Passo ") + (note["source_number"] or note["source_title"])
        # Markdown quotations prevent source formatting from inventing new
        # top-level actions. The existing HTML renderer escapes source content.
        blocks.append(label + "\n" + "\n".join("> " + line for text in texts for line in text.splitlines()))
    if blocks:
        heading = ("Step notes and safety precautions — original wording" if english
                   else "Note e precauzioni degli Step — testo originale")
        answer += "\n\n" + heading + "\n\n" + "\n\n".join(blocks)
    proof = {"basis": "admitted_source_fields",
             "input_answer_sha256": answer_digest(original_answer),
             "output_answer_sha256": answer_digest(answer), "notes": len(notes),
             "quoted_notes": len(missing), "source_units": [
                 {"citation_id": n["citation_id"], "source_number": n["source_number"],
                  "text_sha256": [answer_digest(t) for t in n["texts"]]} for n in notes],
             "all_present": all(present(answer, t)
                                for n in notes for t in n["texts"])}
    return answer, proof


def final_contract(*, deterministic: dict, semantic: dict, reviewed_answer: str,
                   answer: str, semantic_complete: bool, semantic_partial: bool,
                   source_proof: dict | None = None, final_answer: str | None = None,
                   grounding_proof: dict | None = None) -> dict:
    """Resolve one final verdict without laundering old missing-facet lists.

    `answer` is the grounded/filtered model text before source-only supplements.
    `reviewed_answer` is the same redaction projection of the verifier's output.
    A changed answer cannot inherit an earlier semantic PASS except for the exact
    existing grounding projection with zero claims removed. Unresolved or
    partial coverage remains explicit and never obtains a complete cache proof.
    """
    deterministic = deepcopy(deterministic)
    semantic = deepcopy(semantic)
    projected_binding = bool(grounding_proof
        and grounding_proof.get("basis") == "existing_grounding_projection"
        and grounding_proof.get("removed_claims") == 0
        and grounding_proof.get("input_answer_sha256") == answer_digest(reviewed_answer)
        and grounding_proof.get("output_answer_sha256") == answer_digest(answer))
    bound = answer_digest(answer) == answer_digest(reviewed_answer) or projected_binding
    semantic_missing = any(semantic.get(k) for k in
        ("missing_facets", "missing_answer_types", "missing_list_items"))
    outcome = str(semantic.get("outcome") or "").strip().lower()
    valid_complete = bool(semantic_complete and outcome in {"pass", "rewrite"} and not semantic_missing)
    valid_partial = bool(semantic_partial and outcome == "partial")
    if valid_complete and bound:
        result = {k: deepcopy(v) for k, v in deterministic.items()
                  if k not in {"passed", "reason", "missing_answer_facets", "missing_evidence_facets",
                               "missing_list_items", "requirement_checks", "answer_facet_coverage",
                               "evidence_facet_coverage"}}
        result.update(passed=True, reason="semantic_contract_complete",
                      missing_answer_facets=[], missing_evidence_facets=[], missing_list_items=[],
                      coverage_basis="semantic_verifier",
                      covered_facets=list(semantic.get("covered_facets") or []),
                      covered_answer_types=list(semantic.get("covered_answer_types") or []))
    elif valid_partial and bound:
        result = dict(deterministic)
        result.update(passed=True, reason="semantic_contract_partial_explicit",
                      missing_answer_facets=list(semantic.get("missing_facets") or []),
                      missing_list_items=list(semantic.get("missing_list_items") or []))
    elif (semantic_complete or semantic_partial) and not bound:
        result = dict(deterministic)
        result.update(passed=False, reason="semantic_answer_changed_after_review")
    elif semantic_complete or semantic_partial or semantic_missing or outcome == "no_sources":
        result = dict(deterministic)
        result.update(passed=False, reason="semantic_contract_incomplete",
                      missing_answer_facets=list(semantic.get("missing_facets") or []),
                      missing_list_items=list(semantic.get("missing_list_items") or []),
                      semantic_missing_answer_types=list(semantic.get("missing_answer_types") or []))
    else:
        result = dict(deterministic)
    result.update(version=VERSION, deterministic_diagnostics=deterministic,
                  semantic_verifier=semantic, semantic_answer_bound=bound if semantic else None,
                  answer_sha256=answer_digest(answer if final_answer is None else final_answer),
                  source_preservation=deepcopy(source_proof or {}),
                  grounding_projection=deepcopy(grounding_proof or {}))
    final_text = answer if final_answer is None else final_answer
    unchanged_final = answer_digest(final_text) == answer_digest(answer)
    valid_supplement = bool(source_proof and source_proof.get("all_present")
        and source_proof.get("input_answer_sha256") == answer_digest(answer)
        and source_proof.get("output_answer_sha256") == answer_digest(final_text))
    if (source_proof and not valid_supplement) or (not unchanged_final and not valid_supplement):
        result.update(passed=False, reason="source_owned_content_missing_or_changed")
    # Response acceptance under existing coverage thresholds is not a proof of
    # complete coverage. Keep unresolved diagnostics and deny complete caching.
    if (not semantic and result.get("passed")
            and any(result.get(k) for k in ("missing_answer_facets", "missing_evidence_facets", "missing_list_items"))):
        result["reason"] = "deterministic_response_accepted_with_unresolved_facets"
        result["coverage_basis"] = "existing_deterministic_thresholds"
    draft_binding = semantic.get("reviewed_draft_binding")
    if draft_binding is not None:
        binding_valid = bool(type(draft_binding) is dict
            and draft_binding.get("version") == "ask-reviewed-draft-v1"
            and draft_binding.get("basis") == "server_snapshot_at_verifier_call"
            and draft_binding.get("wire_answer_empty") is True
            and draft_binding.get("coverage_reported_complete") is True
            and outcome == "pass" and not semantic_missing
            and bool(reviewed_answer.strip())
            and draft_binding.get("resolved_answer_sha256") == answer_digest(reviewed_answer)
            and answer_digest(semantic.get("answer") or "") == answer_digest(reviewed_answer))
        if not binding_valid:
            result.update(passed=False, reason="reviewed_draft_binding_invalid")
    result["complete"] = bool(result.get("passed") and not semantic_partial
        and not any(result.get(k) for k in ("missing_answer_facets", "missing_evidence_facets", "missing_list_items")))
    return result


def response_contract_bound(response: dict) -> bool:
    """Additional check for this version; legacy/scalar policy stays unchanged.

    This is not an authenticity proof. ProtectedCacheOwner still requires its
    request-owned observation, HMAC and fresh current-authority checks.
    """
    contract = ((response.get("meta") or {}).get("assistant_core_validation") or {}).get("answer_contract") or {}
    if contract.get("version") != VERSION:
        return True
    return bool(contract.get("complete") and contract.get("passed")
                and contract.get("answer_sha256") == answer_digest(response.get("answer"))
                and (not contract.get("source_preservation") or contract["source_preservation"].get("all_present")))
