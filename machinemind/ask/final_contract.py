"""Request-local final ASK verdicts and lossless source-owned safety supplements.

This module neither grants authority nor retrieves evidence. Callers supply only
already-admitted occurrences. The semantic verdict applies to the exact reviewed
answer; deterministic diagnostics stay separate. Source-owned notes are quoted,
not generated or silently translated, and carry their original citation identity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from typing import Callable, Any

VERSION = "ask-final-contract-v1"


def answer_digest(answer: str) -> str:
    return hashlib.sha256(str(answer or "").strip().encode("utf-8")).hexdigest()


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


def preserve_source_notes(answer: str, notes: tuple[dict, ...], *,
                          language: str) -> tuple[str, dict]:
    """Retain exact source wording if a reviewer omitted or translated a note.

    The separate, labelled source quotation cannot be mistaken for a model
    translation. Whitespace-only differences are equivalent; no fuzzy semantic
    matcher or model permission may erase a source warning.
    """
    answer = str(answer or "").strip()
    original_answer = answer
    normal = " ".join(answer.split())
    missing = []
    for note in notes:
        texts = [str(x) for x in note["texts"] if " ".join(str(x).split()) not in normal]
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
             "all_present": all(" ".join(t.split()) in " ".join(answer.split())
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
