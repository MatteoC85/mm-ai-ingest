"""Root-specific draft guidance, immutable ranking and truthful review failures.

The shared evidence reviewer remains the only authority that accepts a proposal.
This module adds no evidence, repairs, provider calls or semantic code inference.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import review_references

POLICY_VERSION = "root-response-quality-v2"

SYNTHESIS_INSTRUCTION = """
A root-cause answer is a diagnostic decision aid, not a repair procedure.
Before choosing hypotheses, compare the COMPLETE abnormal observation against the
conditional consequences documented in EVERY source. Distinguish a source that
explicitly says a particular condition produces the observed combination of
behaviors from a source that merely describes normal operation or a routine setup
check. Prioritize the former when the condition is possible and not ruled out.
A component that moves normally while its intended downstream effect is absent
is discriminating evidence: preserve both observations when comparing mechanisms.
Use this general comparison for any subsystem; do not substitute keyword overlap,
retrieval rank or a generic operating sequence for the documented causal link.

An unknown enabling condition does NOT disqualify a documented mechanism: state
it as a conditional hypothesis and explicitly say that condition is unverified.
Do not turn unknown information into a positive or negative finding. Preserve
distinct supported mechanisms that can explain the observations; do not collapse
one into a different mechanism simply because both concern the same subsystem.
A broad sequence/configuration hypothesis must not crowd out an explicit failure
mechanism with a closer documented consequence. Return only supported causes,
within the existing cause limit; do not fill slots with weaker unrelated cases.

For each hypothesis, explain the documented mechanism, the reported observation
it could explain, and what remains unverified. Prefer one or two high-information,
independently understandable checks;
the check limit is a ceiling, not a quota. Each check must discriminate this
hypothesis using an observable state or relationship actually documented in its
cited source. State what result would support or weaken the hypothesis when that
relationship is documented. Do not append a generic corrective-action checklist.

Start by asking for relevant observations, recorded states or readings ALREADY
available. When the user expressly says a discriminating condition has not been
checked, acknowledge that it remains unknown; if helpful, ask whether an existing
observation or record can resolve it, without asking for a new cycle, reach into
the machine, touch test or physical intervention. Say that it can remain unknown
when no such observation is available. A retrospective question must concern the
specific source-documented condition, not an invented measurement or reference.
It is not necessary to produce an executable test for every hypothesis: checks=[]
is preferable to an operation whose prerequisites cannot be established.

Do not prescribe a reset, restoration, adjustment, replacement, bypass, disassembly
or new machine cycle merely because a source describes a possible fault or repair.
If a physical action or powered observation is genuinely necessary, include all
applicable source-documented operating mode, personnel, protective and isolation
conditions in that same check; do not move prerequisites to another check or a
different hypothesis. If those conditions are missing or uncertain, omit that
action rather than inventing them. A safe retrospective question about an already
observed state can be useful without asking the operator to reproduce a hazard.
Do not invent an approved setup, reference position, target value or tolerance.
A comparison needs an identified, applicable reference actually in SOURCES.
It is valid to return a supported hypothesis with no actionable checks when the
sources establish its mechanism but provide no safe discriminating operation.

Order hypotheses by their fit to the actual discriminating observations and
recent changes. In 'why', describe each hypothesis' own evidence and limitations;
do not call an individual hypothesis 'the most likely', 'the most coherent',
'first choice' or an equivalent comparative winner. The rank field alone conveys
the relative priority, which the independent review will verify. Do not infer a
probability or a confirmed fault from retrieval similarity or source ordering.
""".strip()

RANKING_INSTRUCTION = """
After deciding acceptance under ALL the evidence and safety rules above, return
ranked_proposal_indices: every accepted original proposal_index exactly once, in
descending diagnostic priority; use [] when none was accepted. Exclude all
rejected proposals. Rank by how well each documented mechanism explains the
actual observations, discriminating facts and recent changes, not by retrieval
order or general frequency. Give priority to an explicit source-documented
conditional consequence matching the abnormal combination of observations over a
general normal-operation sequence, unless the condition is contradicted. An
unverified condition remains unknown and may support a qualified hypothesis; it
must not be treated as an observed fact or as automatically excluded. Preserve
uncertainty and the original immutable wording. If a proposal's explanation asserts a comparative priority incompatible
with the resulting order, reject that inconsistent proposal rather than silently
rewriting its explanation. Ranking never grants permission to accept an unsafe
check or to remove a prerequisite needed by a retained action.
""".strip()


CHECK_SELECTION_INSTRUCTION = """
Review the COMPLETE original draft before selecting any check. Each decision also
includes selected_check_indices and safe_check_subset. For an accepted cause,
selected_check_indices contains the exact original indices of the safe checks to
retain, in their original order. blocking_checks contains every omitted index;
the two lists must be disjoint and cover ALL original checks. Use a short factual
note explaining omissions; use an empty note when nothing is omitted. An omitted
unsafe/unsupported action must never appear in retained checks, cause or why.

Set safe_check_subset=true ONLY when the cause and explanation are supported and
every retained check is independently safe and fully supported with its necessary
conditions. Explicitly consider all omitted checks: none may be a prerequisite
for a retained action. A safety/isolation condition in an omitted check cannot
justify a retained intervention. Retain such prerequisites together, or omit the
dependent action too; if the remaining wording is unsafe, reject the proposal.
No rewriting, merging, paraphrasing or invented safety conditions is permitted.
A supported cause may have selected_check_indices=[] when no check qualifies;
its mechanism and explanation still require causal proofs. For a rejected cause,
selected_check_indices=[], safe_check_subset=false and proofs=[]; normal rejection
rules apply. For acceptance, provide proofs for EVERY retained check, using its
ORIGINAL check index. No proof may cover a discarded check. A cause without
causal proof must be rejected even if one of its checks is generally sensible.
A retrospective clarification can be supported by the source's documented
condition/consequence even if that source does not phrase it as a question: it
must ask only for an already available observation of that specific condition,
without inventing a reading, interface or method, or requiring a new operation.
Do not accept a powered observation merely because it is in manual mode or at low
speed: any applicable personnel, protection and operating prerequisites must be
explicit and supported. Otherwise omit that action and retain a supported causal
hypothesis without checks when justified; never invent missing prerequisites.
""".strip()


def review_instruction() -> str:
    # Preserve the shared evidence semantics while replacing only the original
    # whole-proposal selection contract with an explicitly reviewed subset.
    replacements = {
        "Accept only when its cause, explanation and EVERY check are supported and safe.":
            "Accept only when its cause, explanation and EVERY retained check are supported and safe.",
        "Each check must have supporting evidence with applicable target context.":
            "Each retained check must have supporting evidence with applicable target context.",
        "Retain a whole supported\nproposal or reject it; NEVER drop a safety prerequisite and keep its dependent\noperation.":
            "Select only independently safe, supported immutable checks or reject the\nproposal; NEVER drop a safety prerequisite and keep its dependent operation.",
        "Accepted decisions have reason=supported, blocking_checks=[]\nand note=\"\".":
            "Accepted decisions have reason=supported and explicitly partition retained\nand omitted original check indices as specified below.",
        "ALL checks need coverage.": "ALL retained checks need coverage.",
    }
    instruction = review_references.INSTRUCTION
    for old, new in replacements.items():
        if instruction.count(old) != 1:
            raise review_references.ReferenceError("review_instruction_contract_changed")
        instruction = instruction.replace(old, new)
    return instruction + "\n\n" + CHECK_SELECTION_INSTRUCTION + "\n\n" + RANKING_INSTRUCTION


def review_schema(frozen: dict[str, Any]) -> dict[str, Any]:
    """Extend the exact source-bound schema; never widen evidence authorization."""
    result = deepcopy(review_references.schema(frozen))
    result["name"] = "machinemind_root_quality_review_v2"
    authorized = review_references.authorized_references(frozen)
    ids = [item["proposal_index"] for item in authorized["proposals"]]
    for branch, proposal in zip(result["schema"]["properties"]["decisions"]["items"]["anyOf"],
                                authorized["proposals"]):
        checks = proposal["check_indices"]
        selected = {"type": "array", "maxItems": len(checks),
                    "items": {"type": "integer", **({"enum": checks} if checks else {})}}
        branch["properties"].update(selected_check_indices=selected,
                                     safe_check_subset={"type": "boolean"})
        branch["required"].extend(["selected_check_indices", "safe_check_subset"])
    result["schema"]["properties"]["ranked_proposal_indices"] = {
        "type": "array", "maxItems": len(ids),
        "items": {"type": "integer", "enum": ids},
    }
    result["schema"]["required"].append("ranked_proposal_indices")
    return result


def _indices(value: Any, allowed: set[int], *, ordered: bool = False) -> list[int]:
    if (not isinstance(value, list) or any(type(i) is not int or i not in allowed for i in value)
            or len(value) != len(set(value)) or (ordered and value != sorted(value))):
        raise review_references.ReferenceError("invalid_check_selection_indices")
    return value


def validate_review(*, parsed: dict[str, Any], frozen: dict[str, Any],
                    records: list[dict[str, Any]], observed_query: str) -> dict[str, Any]:
    """Validate selected immutable text with the unchanged shared proof validator.

The provider sees the original frozen manifest in full. Code only selects exact
check strings expressly approved by that reviewer; it cannot repair an action,
forge a proof or approve a mechanism. Original indices remain in the audit.
"""
    if not isinstance(parsed, dict) or set(parsed) != {"decisions", "ranked_proposal_indices"}:
        raise review_references.ReferenceError("invalid_ranked_review_envelope")
    review_references.authorized_references(frozen)  # Verify original fingerprint first.
    original = frozen["proposals"]
    decisions = parsed["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(original):
        raise review_references.ReferenceError("decision_coverage_incomplete")
    derived = deepcopy(frozen)
    selected_decisions = []
    selection_audit = []
    maps = {}
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise review_references.ReferenceError("decision_keys")
        pi = decision.get("proposal_index")
        if type(pi) is not int or pi not in range(len(original)) or pi in seen:
            raise review_references.ReferenceError("unknown_or_duplicate_proposal")
        seen.add(pi)
        if "selected_check_indices" not in decision or type(decision.get("safe_check_subset")) is not bool:
            raise review_references.ReferenceError("missing_check_selection")
        allowed = set(range(len(original[pi]["checks"])))
        selected = _indices(decision["selected_check_indices"], allowed, ordered=True)
        blocked = _indices(decision.get("blocking_checks"), allowed)
        normalized = deepcopy(decision)
        normalized.pop("selected_check_indices")
        normalized.pop("safe_check_subset")
        if decision.get("verdict") == "accept":
            if set(selected) & set(blocked) or set(selected) | set(blocked) != allowed:
                raise review_references.ReferenceError("incomplete_check_partition")
            if decision["safe_check_subset"] is not True:
                raise review_references.ReferenceError("check_subset_not_safe")
            note = decision.get("note")
            if not isinstance(note, str) or len(note) > 220 or (bool(blocked) != bool(note.strip())):
                raise review_references.ReferenceError("invalid_selection_note")
            mapping = {index: position for position, index in enumerate(selected)}
            maps[pi] = {position: index for index, position in mapping.items()}
            derived["proposals"][pi]["checks"] = [
                {**deepcopy(original[pi]["checks"][index]), "check_index": position}
                for position, index in enumerate(selected)]
            proofs = normalized.get("proofs")
            if not isinstance(proofs, list):
                raise review_references.ReferenceError("proof_count")
            for proof in proofs:
                if not isinstance(proof, dict):
                    raise review_references.ReferenceError("proof_keys")
                indices = _indices(proof.get("check_indices"), set(selected))
                proof["check_indices"] = [mapping[index] for index in indices]
            normalized.update(blocking_checks=[], note="")
        elif selected or decision["safe_check_subset"] is not False:
            raise review_references.ReferenceError("rejected_proposal_keeps_checks")
        selected_decisions.append(normalized)
        selection_audit.append({"proposal_index": pi, "accepted": decision.get("verdict") == "accept",
            "selected_check_indices": list(selected), "blocking_checks": list(blocked),
            "safe_check_subset": decision["safe_check_subset"], "note": decision.get("note"),
            "input_check_count": len(allowed)})
    derived["fingerprint"] = review_references.digest(
        {key: value for key, value in derived.items() if key != "fingerprint"})
    result = review_references.validate(parsed={"decisions": selected_decisions},
        frozen=derived, records=records, observed_query=observed_query)
    approved = [verdict["input_index"] for verdict in result["summary"]["verdicts"]
                if verdict["accepted"]]
    order = parsed["ranked_proposal_indices"]
    if (not isinstance(order, list) or any(type(index) is not int for index in order)
            or len(order) != len(approved) or len(set(order)) != len(order)
            or set(order) != set(approved)):
        raise review_references.ReferenceError("accepted_ranking_mismatch")
    by_index = dict(zip(approved, result["causes"]))
    result["causes"] = [{**by_index[index], "rank": rank}
                        for rank, index in enumerate(order, 1)]
    result["citation_ids"] = list(dict.fromkeys(
        cid for cause in result["causes"] for cid in cause["citations"]))
    for proof in result["summary"]["support_proofs"]:
        mapping = maps[proof["proposal_index"]]
        proof["decision"]["check_indices"] = [mapping[i] for i in proof["decision"]["check_indices"]]
    subset_used = any(row["accepted"] and row["blocking_checks"] for row in selection_audit)
    result["summary"].update(quality_policy_version=POLICY_VERSION,
        ranking_validated=True, ranked_input_indices=list(order),
        immutable_proposals=not subset_used, immutable_proposal_text=True,
        immutable_retained_checks=True, all_input_checks_decided=True,
        all_retained_checks_covered=True, proof_check_index_namespace="original_proposal",
        check_selection=sorted(selection_audit, key=lambda row: row["proposal_index"]),
        original_registry_fingerprint=frozen["fingerprint"],
        selected_registry_fingerprint=derived["fingerprint"])
    return result


def rejected_response(*, response: dict[str, Any], meta: dict[str, Any],
                      language: str, result_code: str) -> dict[str, Any]:
    """A rejected generated proposal is not evidence that sources are absent.

No draft action or citation survives. The existing public technical-error code
is supplied by the caller to preserve transport/UI compatibility.
"""
    message = (
        "Sources were found, but the proposed causes or checks did not pass the "
        "evidence and safety review. I cannot provide a validated diagnosis from "
        "this attempt. You can report machine states or readings already available, "
        "without performing new interventions."
        if str(language).lower().startswith("en") else
        "Ho trovato fonti, ma le cause o i controlli proposti non hanno superato la "
        "verifica delle evidenze e della sicurezza. Questo tentativo non fornisce "
        "una diagnosi validata. Puoi riportare stati o valori della macchina già "
        "disponibili, senza eseguire nuovi interventi."
    )
    out = dict(response)
    out.update(ok=False, status="error", result_code=result_code,
        problem_summary=message, possible_causes=[], recommended_next_checks=[],
        citations=[], rg_links=[])
    out["meta"] = {**meta, "cacheable": False, "semantic_cacheable": False,
        "root_response_quality": {"policy_version": POLICY_VERSION,
            "outcome": "review_rejected_proposals", "sources_present": True,
            "validated_diagnosis": False},
        "assistant_core_root_adjudication": {
            **dict(meta.get("assistant_core_root_adjudication") or {}),
            "outcome": "rejected", "reason": "review_rejected_proposals",
            "accepted_count": 0}}
    return out
