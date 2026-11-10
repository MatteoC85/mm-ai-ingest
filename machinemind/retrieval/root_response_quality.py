"""Root-specific draft guidance, immutable ranking and truthful review failures.

The shared evidence reviewer remains the only authority that accepts a proposal.
This module adds no evidence, repairs, provider calls or semantic code inference.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import re
from typing import Any

from . import review_references

POLICY_VERSION = "root-response-quality-v8"

PRODUCER_CONTRACT_VERSION = "root-producer-applicability-v1"

PRODUCER_CONTRACT_INSTRUCTION = """
The private producer contract separates two different uncertainties. applicability
states whether the cited configuration actually applies to the reported equipment;
the existing why separately preserves whether the causal enabling state was observed.
An attached/routed source is not proof of installed configuration. When that
relationship is unverified, choose source_scoped_unverified and write an explicit,
self-contained statement that the explanation concerns the configuration documented
by the cited source and that its applicability to the reported equipment is unverified.
Do not substitute unknown component position/state for that applicability statement.
Use established_by_sources_and_observations only when literal source content and
reported facts establish the relevant relationship, never from an identifier or
routing alone. known_incompatible cannot become a hypothesis through conditional words.

Classify each original check by what it actually requests: an observation or record
ALREADY available, a NEW observation, or an intervention. With unverified applicability,
return only already_available_observation_or_record checks. If no such record exists,
leave the state unknown; do not append an authorized/manual/slow-cycle fallback.
A classification is not evidence or safety authorization. The independent reviewer
will judge the complete original rendered statement, mechanism and retained check text.
Return recommended_next_checks=[]; the server derives that list from retained checks.
""".strip()


class ProducerContractError(ValueError):
    """The opted-in writer did not return its private typed proposal contract."""


def producer_schema(base_schema: dict[str, Any]) -> dict[str, Any]:
    """Clone only this writer's provider schema; the public/default schema is untouched."""
    result = deepcopy(base_schema)
    result["name"] = "root_cause_producer_applicability_v1"
    schema = result["schema"]
    cause = schema["properties"]["possible_causes"]["items"]
    cause["properties"]["applicability"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": [
                "source_scoped_unverified", "established_by_sources_and_observations",
                "known_incompatible"]},
            "statement": {"type": "string"},
        },
        "required": ["status", "statement"],
    }
    cause["required"].append("applicability")
    cause["properties"]["checks"]["items"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "enum": [
                "already_available_observation_or_record", "new_observation", "intervention"]},
            "text": {"type": "string"},
        },
        "required": ["intent", "text"],
    }
    schema["properties"]["recommended_next_checks"]["maxItems"] = 0
    return result


def normalize_producer_result(parsed: Any, *, max_causes: int) -> dict[str, Any]:
    """Prepare original public-shaped text BEFORE grounding and immutable review.

    Model statuses/intent labels confer no target, applicability or safety proof.
    Misclassified retained text must still pass the unchanged independent reviewer.
    """
    def fail(code: str):
        raise ProducerContractError(code)

    def obj(value: Any, fields: set[str], code: str) -> dict[str, Any]:
        if type(value) is not dict or set(value) != fields:
            fail(code)
        return value

    def text(value: Any, code: str) -> str:
        if type(value) is not str or not value.strip():
            fail(code)
        return value

    top = obj(parsed, {"problem_summary", "possible_causes", "recommended_next_checks"},
              "invalid_producer_top_level")
    summary = text(top["problem_summary"], "invalid_producer_problem_summary")
    rows = top["possible_causes"]
    limit = max(1, min(int(max_causes or 1), 3))
    if type(rows) is not list or len(rows) > limit:
        fail("invalid_producer_causes")
    if type(top["recommended_next_checks"]) is not list or top["recommended_next_checks"]:
        fail("untyped_producer_next_checks")
    causes = []
    next_checks = []
    for row in rows:
        row = obj(row, {"rank", "cause", "why", "checks", "citations", "applicability"},
                  "invalid_producer_cause")
        if type(row["rank"]) is not int or row["rank"] < 1:
            fail("invalid_producer_rank")
        cause = text(row["cause"], "invalid_producer_cause_text")
        why = text(row["why"], "invalid_producer_why")
        app = obj(row["applicability"], {"status", "statement"},
                  "invalid_producer_applicability")
        status = app["status"]
        if type(status) is not str or status not in {
                "source_scoped_unverified", "established_by_sources_and_observations",
                "known_incompatible"}:
            fail("invalid_producer_applicability_status")
        statement = text(app["statement"], "invalid_producer_applicability_statement")
        if type(row["citations"]) is not list or not row["citations"]:
            fail("invalid_producer_citations")
        for cid in row["citations"]:
            text(cid, "invalid_producer_citation")
        if type(row["checks"]) is not list:
            fail("invalid_producer_checks")
        retained = []
        for check in row["checks"]:
            check = obj(check, {"intent", "text"}, "invalid_producer_check")
            intent = check["intent"]
            if type(intent) is not str or intent not in {
                    "already_available_observation_or_record", "new_observation", "intervention"}:
                fail("invalid_producer_check_intent")
            check_text = text(check["text"], "invalid_producer_check_text")
            if status == "source_scoped_unverified" and intent != "already_available_observation_or_record":
                continue
            retained.append(check_text)
        if status == "known_incompatible":
            continue
        causes.append({"rank": row["rank"], "cause": cause,
            "why": statement + "\n\n" + why, "checks": retained,
            "citations": list(row["citations"])})
        next_checks.extend(retained)
    if not causes:
        fail("no_admissible_producer_proposals")
    return {"problem_summary": summary, "possible_causes": causes,
            "recommended_next_checks": list(dict.fromkeys(next_checks))}


MAX_WHY_UNITS = 8
# These are punctuation ambiguities, not evidence or diagnostic classifications.
_ABBREVIATIONS = frozenset({"approx", "prof", "dott", "sign", "resp", "incl", "dept", "assn", "misc"})


def why_units(why: str) -> list[dict[str, Any]]:
    """Lossless complete-sentence spans, or one atomic original on ambiguity.

    This mechanical partition grants no support to any sentence. In particular,
    it cannot decide that an omitted qualifier is dispensable: the independent
    reviewer must judge the complete retained explanation in its original context.
    """
    if not isinstance(why, str) or not why.strip():
        raise review_references.ReferenceError("invalid_why_text")

    def atomic():
        return [{"why_index": 0, "start": 0, "end": len(why), "text": why}]

    if not why.rstrip().endswith((".", "?", "!")):
        return atomic()
    stack = []
    closing = {"(": ")", "[": "]", "{": "}", "«": "»", "“": "”", "‘": "’"}
    boundaries = []
    for pos, char in enumerate(why):
        # Apostrophes within words are not quotation marks. An unmatched quote
        # (including a possessive apostrophe) conservatively keeps WHY atomic.
        word_apostrophe = (char in "'’" and 0 < pos < len(why) - 1
                           and why[pos - 1].isalnum() and why[pos + 1].isalnum())
        if word_apostrophe:
            continue
        if char in "\"'":
            if pos and why[pos - 1] == "\\":
                return atomic()
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        elif char in closing:
            stack.append(closing[char])
        elif char in closing.values():
            if not stack or stack.pop() != char:
                return atomic()
        if char not in ".?!" or stack:
            continue
        # Only punctuation followed by whitespace and a new capitalized sentence
        # is separable. Internal decimals/codes are never split.
        end = pos + 1
        if end >= len(why) or not why[end].isspace():
            continue
        while end < len(why) and why[end].isspace():
            end += 1
        if end == len(why):
            continue
        if not why[end].isupper() or (pos and why[pos - 1] in ".?!"):
            return atomic()
        if char == ".":
            word = re.search(r"([^\W\d_]+)$", why[:pos])
            if (not word or len(word[1]) < 4 or word[1].isupper()
                    or word[1].lower() in _ABBREVIATIONS):
                return atomic()
        boundaries.append(end)
    if stack or len(boundaries) >= MAX_WHY_UNITS:
        return atomic()
    offsets = [0, *boundaries, len(why)]
    return [{"why_index": index, "start": start, "end": end, "text": why[start:end]}
            for index, (start, end) in enumerate(zip(offsets, offsets[1:]))]


def why_selection_packet(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Root-only proposal spans; these indices are never source-evidence IDs."""
    review_references.authorized_references(frozen)
    return [{"proposal_index": proposal["proposal_index"],
             "why_units": why_units(proposal["why"])} for proposal in frozen["proposals"]]

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

SOURCES metadata such as exact_machine_scope, document IDs and citation IDs
describes retrieval provenance only. A source attached or routed to the selected
machine does not by itself establish its installed model, configuration, physical
applicability, or the manual's correctness/currentness. Prefer exact-machine
evidence only when literal source content and explicit user facts establish the
relevant relationship; never infer that relationship from routing metadata or a
name/code match alone.

When a documented mechanism explains the reported observation but its installed
configuration or applicability remains unverified, scope the original hypothesis
and explanation explicitly to the configuration documented in that source and
state that applicability is unverified. Keep this distinct from whether the
mechanism's enabling condition was observed. Do not describe the source as the
exact/current manual for the installed equipment without supporting evidence.
Preserve known incompatibilities; conditional wording cannot transfer a mechanism
between unrelated targets. With unverified applicability, ask only for relevant
observations or records already available, or return no checks; do not prescribe
a new operation or intervention. Omit a hypothesis if the source does not support
the mechanism-to-symptom relationship, or if a known target incompatibility applies.

For each hypothesis, explain the documented mechanism, the reported observation
it could explain, and what remains unverified. Prefer one or two high-information,
independently understandable checks;
the check limit is a ceiling, not a quota. Each check must discriminate this
hypothesis using an observable state or relationship actually documented in its
cited source. State what result would support or weaken the hypothesis when that
relationship is documented. Do not append a generic corrective-action checklist.
For support/weakening statements, prefer naming the affected cause, especially
when comparing alternatives. A pronoun is acceptable when its reference remains
unambiguous and refers to the same cause. Do not silently switch the referent of
"it", "this hypothesis" or a similar pronoun between clauses; name the cause when
needed to keep the comparison clear. The direction of
each inference must follow the documented relationship: the same observation
must not both support and weaken the same cause under unchanged conditions.
A concise clear comparison is sufficient; do not add unsupported alternatives
or recite all manual details to satisfy this requirement.
Keep the condition at the time of the failure distinct from its state after a
correction or other change. An alarm disappearing after a suspected condition
was corrected does not by itself weaken that condition as the initiating cause;
it may be compatible with that cause. A later normal state does not establish
that the earlier state was normal. Name the relevant time/conditions whenever
the distinction affects the inference. Do not invent a causal prediction merely
to complete a check: asking for the documented observation already available is
sufficient when its diagnostic direction is not established.

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
uncertainty and the original immutable wording. If the retained explanation asserts a comparative priority incompatible
with the resulting order, reject that inconsistent proposal rather than silently
rewriting its explanation. Ranking never grants permission to accept an unsafe
check or to remove a prerequisite needed by a retained action.
""".strip()


CONDITIONAL_APPLICABILITY_INSTRUCTION = """
For Root hypotheses, distinguish the documented target/mechanism from whether its
enabling condition or optional configuration was actually present. Do not reject
solely as unknown_target because that condition remains unverified, but ONLY when
the original proposal itself explicitly preserves that uncertainty, the source
and reported observation establish the same relevant target and symptom, and the
documented mechanism explains that observation conditionally. A name/code match
alone cannot override a conflicting model or establish a causal dependency.
Distinguish an UNKNOWN installed model from a KNOWN incompatible model. An empty
machine-model field is not itself evidence of incompatibility. Judge the type of
claim, not whether the request uses a documentary or hypothetical label. A real
machine question may also receive an explicitly source-scoped conditional
explanation while applicability remains unverified; that explanation does not
assert the documented configuration is installed. The source's actual content
and governing context must independently connect the relevant target, reported
observations and causal mechanism; a matching message, document title or
selected-machine routing alone does not establish that connection. Use authorized
target units to substantiate this bounded relationship, not to assert installed
equipment identity. For an explicitly documentary case, assess the observations
within that stated case without turning them into a real incident.

This permits only the original wording that explicitly limits the mechanism to
the documented configuration and leaves its applicability unverified. Do not
require an extra model label when the retained wording already preserves that
material limitation. A useful conditional hypothesis is not a confirmed fault,
installed-model identification or authorization to apply the manual's procedures.
If the proposal asserts installation/applicability without support, contradicts a
known model, or fails to preserve a necessary limitation, reject it; do not add a
qualification or assume compatibility. A merely hypothetical phrasing never
licenses transferring a mechanism between unrelated targets. A claim that the
mechanism actually applies to installed equipment requires applicable
identity/dependency evidence as usual, even in a purported documentary case.

With unverified configuration, retained checks may ask only for already available
observations or records of the specific source-documented condition, or be empty.
They must not invent a sensor, interface, reading or dependency, require a new
operation, or prescribe an intervention whose applicability/prerequisites remain
unknown. The proposed cause and explanation must themselves respect these limits.
The reviewer must still independently establish applicable target evidence and
causal support with the existing authorized proofs; uncertainty is not a proof.
Selected-machine routing alone remains insufficient. Reject wrong-model claims,
unsupported dependencies/actions and wording that presents an unknown condition
as established. Never repair a proposal or convert a rejection into acceptance
merely because it uses conditional words.
""".strip()


CHECK_SELECTION_INSTRUCTION = """
Review the COMPLETE original draft before selecting any check. Each decision also
includes selected_check_indices and safe_check_subset. For an accepted cause,
selected_check_indices contains the exact original indices of the safe checks to
retain, in their original order. blocking_checks contains every omitted index;
the two lists must be disjoint and cover ALL original checks. Use a short factual
note explaining check or WHY omissions; use an empty note when neither is omitted. An omitted
unsafe/unsupported action must never appear in retained checks, cause or why.

Set safe_check_subset=true ONLY when the cause and selected explanation are supported and
every retained check is independently safe and fully supported with its necessary
conditions. Explicitly consider all omitted checks: none may be a prerequisite
for a retained action. A safety/isolation condition in an omitted check cannot
justify a retained intervention. Retain such prerequisites together, or omit the
dependent action too; if the remaining wording is unsafe, reject the proposal.
No rewriting, merging, paraphrasing or invented safety conditions is permitted.
A supported cause may have selected_check_indices=[] when no check qualifies;
its mechanism and selected explanation still require causal proofs. For a rejected cause,
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

Review the causal meaning of EACH retained check, not just whether a similar
test appears in a source. Prefer explicit cause names in support/weakening
statements, especially when alternatives are compared, but accept clear pronouns
that consistently refer to the same cause. Do not reject a check merely because
it uses a pronoun or omits a repeated label. Reject only a genuinely ambiguous
causal referent or a claimed inference that is contradictory or unsupported by
its cited condition/consequence. A pronoun must not silently change referent.
Do not guess the intended cause, rewrite the check, or invent the missing inference.
If that defect is confined to a check and the cause/selected explanation remain supported,
put only that check in blocking_checks and retain the other independently safe,
supported checks through the existing subset contract. It is valid to retain
no checks. Do not reject an otherwise supported useful cause merely to demand
every manual detail or a longer list of alternatives. Complete clear comparisons
between supported alternatives are allowed; do not require every check to discuss
only its parent cause. All existing proof and safety-prerequisite rules still apply.

Check the timing and conditions of every causal inference. Distinguish a state
observed during the failure from a later state after correction or another
change. Disappearance after correction cannot by itself refute that condition
as the initiating cause, and a later normal observation cannot be projected
backward as proof of an earlier normal state. If a check makes that unsupported
inference, omit that check through the existing explicit subset decision; do not
discard a separately supported cause or independent useful check. No inferred
outcome is required: a supported question about an existing observation may be
retained without claiming what it proves. Do not repair the original wording,
infer an intervention occurred, or add a new test to resolve missing history.
""".strip()


WHY_SELECTION_INSTRUCTION = """
Review the COMPLETE original cause, WHY and checks before choosing any WHY unit.
WHY_SELECTION contains server-defined complete original sentence spans, not source
evidence. For EVERY decision include selected_why_indices, omitted_why_indices and
safe_why_subset. The selected and omitted indices must be disjoint and together
cover every original WHY unit. Keep selected indices in their original order.
An accepted cause must retain at least one unit, set safe_why_subset=true and have
the existing authorized causal proofs for its cause and COMPLETE retained WHY.
Read the retained units as one explanation, with the original omitted context in
view: each remaining claim and the combined meaning must be supported and safe.
Do not project a document's consequence into an observed fact that the user did
not report. A false observational sentence may be omitted only when the remaining
original explanation independently supports the hypothesis without that claim.

Preserve every qualification, negation, uncertainty, optional-model limitation,
safety prerequisite and dependency necessary to the retained cause, explanation
or checks. Never make a hypothesis more certain or broadly applicable by dropping
such wording. If a removed sentence is needed to interpret a retained reference,
condition, contrast or action safely and truthfully, retain the dependency too
only if supported; otherwise reject the proposal or omit the dependent material.
Do not select a fragment, rewrite, paraphrase, merge claims or invent replacement
text. Atomic units cannot be split further. An unsafe clause inside a retained
unit still disqualifies that unit. Selection does not turn an existing rejection
into acceptance: independently assess the entire selected result and its proofs.
For a rejected cause select no WHY units, omit all of them, set safe_why_subset=false
and provide no proofs. Explain omissions in note when accepting with any WHY unit
or check omitted; note is empty only when both original lists are fully retained.
safe_check_subset=true must assess checks together with the selected explanation,
including the prerequisites and limitations that remain. No extra facts or checks
are authorized by this selection contract.
""".strip()


def review_instruction() -> str:
    # Preserve the shared evidence semantics while replacing only the original
    # whole-proposal selection contract with an explicitly reviewed subset.
    replacements = {
        "Accept only when its cause, explanation and EVERY check are supported and safe.":
            "Accept only when its cause, selected explanation and EVERY retained check are supported and safe.",
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
    return (instruction + "\n\n" + CONDITIONAL_APPLICABILITY_INSTRUCTION
            + "\n\n" + CHECK_SELECTION_INSTRUCTION + "\n\n" + WHY_SELECTION_INSTRUCTION
            + "\n\n" + RANKING_INSTRUCTION)


def review_schema(frozen: dict[str, Any]) -> dict[str, Any]:
    """Extend the exact source-bound schema; never widen evidence authorization."""
    result = deepcopy(review_references.schema(frozen))
    result["name"] = "machinemind_root_quality_review_v3"
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
        why_indices = [unit["why_index"] for unit in why_units(
            frozen["proposals"][proposal["proposal_index"]]["why"])]
        why_array = {"type": "array", "maxItems": len(why_indices),
                     "items": {"type": "integer", "enum": why_indices}}
        branch["properties"].update(selected_why_indices=deepcopy(why_array),
            omitted_why_indices=deepcopy(why_array), safe_why_subset={"type": "boolean"})
        branch["required"].extend(["selected_why_indices", "omitted_why_indices", "safe_why_subset"])
    result["schema"]["properties"]["ranked_proposal_indices"] = {
        "type": "array", "maxItems": len(ids),
        "items": {"type": "integer", "enum": ids},
    }
    result["schema"]["required"].append("ranked_proposal_indices")
    return result


def _indices(value: Any, allowed: set[int], *, ordered: bool = False,
             error: str = "invalid_check_selection_indices") -> list[int]:
    if (not isinstance(value, list) or any(type(i) is not int or i not in allowed for i in value)
            or len(value) != len(set(value)) or (ordered and value != sorted(value))):
        raise review_references.ReferenceError(error)
    return value


def validate_review(*, parsed: dict[str, Any], frozen: dict[str, Any],
                    records: list[dict[str, Any]], observed_query: str) -> dict[str, Any]:
    """Validate selected immutable text with the unchanged shared proof validator.

The provider sees the original frozen manifest in full. Code only selects exact
check strings and complete WHY spans expressly approved by that reviewer; it cannot repair an action,
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
    why_audit = []
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
        units = why_units(original[pi]["why"])
        why_allowed = set(range(len(units)))
        if ("selected_why_indices" not in decision or "omitted_why_indices" not in decision
                or type(decision.get("safe_why_subset")) is not bool):
            raise review_references.ReferenceError("missing_why_selection")
        why_selected = _indices(decision["selected_why_indices"], why_allowed, ordered=True,
                                error="invalid_why_selection_indices")
        why_omitted = _indices(decision["omitted_why_indices"], why_allowed, ordered=True,
                               error="invalid_why_selection_indices")
        if (set(why_selected) & set(why_omitted)
                or set(why_selected) | set(why_omitted) != why_allowed):
            raise review_references.ReferenceError("incomplete_why_partition")
        normalized = deepcopy(decision)
        normalized.pop("selected_check_indices")
        normalized.pop("safe_check_subset")
        normalized.pop("selected_why_indices")
        normalized.pop("omitted_why_indices")
        normalized.pop("safe_why_subset")
        if decision.get("verdict") == "accept":
            if not why_selected:
                raise review_references.ReferenceError("empty_selected_why")
            if decision["safe_why_subset"] is not True:
                raise review_references.ReferenceError("why_subset_not_safe")
            if set(selected) & set(blocked) or set(selected) | set(blocked) != allowed:
                raise review_references.ReferenceError("incomplete_check_partition")
            if decision["safe_check_subset"] is not True:
                raise review_references.ReferenceError("check_subset_not_safe")
            note = decision.get("note")
            if (not isinstance(note, str) or len(note) > 220
                    or (bool(blocked or why_omitted) != bool(note.strip()))):
                raise review_references.ReferenceError("invalid_selection_note")
            derived["proposals"][pi]["why"] = "".join(units[index]["text"] for index in why_selected)
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
        elif why_selected or decision["safe_why_subset"] is not False:
            raise review_references.ReferenceError("rejected_proposal_keeps_why")
        selected_decisions.append(normalized)
        selection_audit.append({"proposal_index": pi, "accepted": decision.get("verdict") == "accept",
            "selected_check_indices": list(selected), "blocking_checks": list(blocked),
            "safe_check_subset": decision["safe_check_subset"], "note": decision.get("note"),
            "input_check_count": len(allowed)})
        why_audit.append({"proposal_index": pi, "accepted": decision.get("verdict") == "accept",
            "selected_why_indices": list(why_selected), "omitted_why_indices": list(why_omitted),
            "safe_why_subset": decision["safe_why_subset"], "note": decision.get("note"),
            "input_why_unit_count": len(units), "units": [
                {key: unit[key] for key in ("why_index", "start", "end")}
                | {"text_sha256": sha256(unit["text"].encode("utf-8")).hexdigest()}
                for unit in units]})
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
    why_subset_used = any(row["accepted"] and row["omitted_why_indices"] for row in why_audit)
    result["summary"].update(quality_policy_version=POLICY_VERSION,
        ranking_validated=True, ranked_input_indices=list(order),
        immutable_proposals=not (subset_used or why_subset_used),
        immutable_proposal_text=not why_subset_used, immutable_retained_why_units=True,
        all_input_why_units_decided=True, why_selection=sorted(why_audit, key=lambda row: row["proposal_index"]),
        immutable_retained_checks=True, all_input_checks_decided=True,
        all_retained_checks_covered=True, proof_check_index_namespace="original_proposal",
        check_selection=sorted(selection_audit, key=lambda row: row["proposal_index"]),
        original_registry_fingerprint=frozen["fingerprint"],
        selected_registry_fingerprint=derived["fingerprint"])
    return result


def response_review_summary(summary: dict[str, Any], *, debug: bool) -> dict[str, Any]:
    """Keep unreviewed free-form omission notes in the private/debug audit only.

    A reviewer can quote rejected WHY or checks when explaining an omission.
    Publishing that note would put unapproved claims back into normal metadata.
    Hashes retain audit correlation without interpreting or rewriting the note.
    """
    result = deepcopy(summary)
    if not debug:
        for section in ("verdicts", "check_selection", "why_selection"):
            for row in result.get(section, []):
                if "note" in row:
                    note = row.pop("note")
                    row["note_sha256"] = sha256(note.encode("utf-8")).hexdigest()
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
