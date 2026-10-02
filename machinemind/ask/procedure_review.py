"""Compact independent review of a source-ordered ASK procedure.

This module does not grant authority, infer source identity, cache permissions,
score evidence, or call a provider. Its inputs are projected only AFTER request
admission. The semantic reviewer still evaluates every requirement and all
claims; a structural observation is never a semantic PASS.

The response protocol can retain an already-correct draft byte for byte, edit
specific blocks, or provide a full replacement when genuinely necessary. Old
full-answer replies remain readable, without acquiring a structural proof.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, Callable

VERSION = "ask-procedure-review-v2"
_NUMBERED = re.compile(r"(?m)^(?P<number>[1-9][0-9]*)[.)][ \t]+")
# Reviewers can reproduce a heading with Markdown, indentation, or a line
# break after its number. Recognize those forms before accepting an edit so
# formatting cannot accidentally become a new or changed procedure step.
_EDIT_NUMBERED = re.compile(
    r"(?m)^[ \t]*(?:\*\*|__)?(?P<number>[1-9][0-9]*)[.)]"
    r"(?:\*\*|__)?(?:[ \t]+|\r?\n|$)")


def digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _numbers(text: str) -> list[int]:
    return [int(m.group("number")) for m in _NUMBERED.finditer(str(text))]


def observe_structure(answer: str, occurrences: list[dict], *, fields: Callable,
                      notes: tuple[dict, ...], notes_present: Callable,
                      source_ordered: bool = True) -> dict:
    """Observe admitted Steps without conflating them with synthesized actions.

    A malformed/ambiguous selection never becomes a guessed 1..N range. Sparse
    selections keep their actual source numbers. The digest binds the text
    inspected, but is not an authorization or an authenticity token.
    Only an explicit source-ordered producer requires visible source ordinals.
    A manual/mixed answer needs independent semantic order/coverage review;
    retrieved Step ordinals are diagnostic data, never a guessed action list.
    """
    units = []
    seen = set()
    for occurrence in occurrences:
        data = fields(occurrence)
        cid = str(occurrence.get("citation_id") or "").strip()
        raw = str(data.get("step_number") or "").strip()
        if not cid or cid in seen or not re.fullmatch(r"[1-9][0-9]*", raw):
            if source_ordered:
                return {"version": VERSION, "usable": False, "reason": "ambiguous_step_selection"}
            continue
        seen.add(cid)
        units.append({"citation_id": cid, "number": int(raw),
                      "description_sha256": digest(str(data.get("description") or "")),
                      "title_sha256": digest(str(data.get("title") or ""))})
    expected = [u["number"] for u in units]
    usable = bool(expected and expected == sorted(set(expected))) if source_ordered else bool(answer.strip())
    visible = _numbers(answer)
    present = bool(all(notes_present(answer, t) for n in notes for t in n["texts"]))
    return {"version": VERSION, "usable": usable,
            "basis": ("admitted_step_occurrences_not_facet_word_overlap" if source_ordered
                      else "answer_local_numbering_requires_semantic_order_review"),
            "source_sequence_required": source_ordered,
            "answer_sha256": digest(answer), "units": units,
            "source_numbers": expected,
            "expected_numbers": expected if source_ordered else [], "visible_numbers": visible,
            "sequence_complete": bool(usable and visible == expected) if source_ordered else None,
            "source_notes": len(notes), "source_notes_present": present,
            "semantic_coverage_proven": False}


def block_layout(answer: str) -> dict:
    """Lossless blocks: even separators remain owned by their original block."""
    answer = str(answer)
    matches = list(_NUMBERED.finditer(answer))
    blocks = []
    start = matches[0].start() if matches else len(answer)
    if start:
        blocks.append({"block_id": "preamble", "text": answer[:start]})
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        blocks.append({"block_id": f"step_{i + 1}", "text": answer[match.start():end]})
    if not blocks:
        blocks = [{"block_id": "preamble", "text": answer}]
    assert "".join(x["text"] for x in blocks) == answer
    return {"version": VERSION, "answer_sha256": digest(answer), "blocks": blocks}


def _block_parts(text: str) -> tuple[str, str, str]:
    """Keep the source step header and inter-block whitespace server-owned."""
    heading = _NUMBERED.match(text)
    prefix = heading.group(0) if heading else ""
    content = text[len(prefix):]
    body = content.rstrip()
    return prefix, body, content[len(body):]


def review_blocks(layout: dict) -> list[dict]:
    """Unambiguous edit targets; ordinals in block IDs are not step numbers."""
    result = []
    for block in layout["blocks"]:
        prefix, body, _ = _block_parts(block["text"])
        result.append({"block_id": block["block_id"],
            "step_number": _numbers(prefix)[0] if prefix else None,
            "body_first_line": body.splitlines()[0] if body else ""})
    return result


def _edited_block(original: str, value: str) -> str:
    prefix, _, separator = _block_parts(original)
    body = value.strip()
    # Accept a repeated heading only when it identifies the same original
    # step. Otherwise the wire contract is the complete corrected body alone.
    heading = _EDIT_NUMBERED.match(body)
    if heading:
        if not prefix or int(heading.group("number")) != _numbers(prefix)[0]:
            raise ValueError("procedure_review_edit_changes_step_sequence")
        body = body[heading.end():].strip()
    if not body or _EDIT_NUMBERED.search(body):
        raise ValueError("procedure_review_edit_changes_step_sequence")
    return prefix + body + separator


def review_schema(legacy: dict, layout: dict, facets: list[str], types: list[str], *,
                  operation_boundary: bool = False,
                  operation_source_ids: list[str] | None = None) -> dict:
    result = deepcopy(legacy)
    result["name"] = legacy["name"]  # backwards-readable verifier envelope
    schema = result["schema"]
    # Required facets and must-cover retrieval plans may be distinct. Their
    # union must fit in the verdict; never make coverage impossible or drop an
    # obligation merely because the old wire format allowed only twelve keys.
    for key in ("covered_facets", "missing_facets"):
        field = schema["properties"].get(key)
        if field is not None:
            field["maxItems"] = max(int(field.get("maxItems", 0)), len(set(facets)))
    schema["properties"]["reply_mode"] = {"type": "string", "enum": ["replace", "retain", "edit"]}
    schema["properties"]["edits"] = {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "block_id": {"type": "string", "enum": [b["block_id"] for b in layout["blocks"]]},
            "text": {"type": "string", "description":
                "Complete corrected body of this block only; omit the step number and trailing "
                "block separators, which the server preserves. Keep every source instruction, "
                "precaution, condition, numeric value, and citation required in this block."}},
        "required": ["block_id", "text"]}, "maxItems": len(layout["blocks"])}
    schema["required"] = list(schema["required"]) + ["reply_mode", "edits"]
    if operation_boundary:
        if not operation_source_ids:
            raise ValueError("operation_boundary_requires_admitted_sources")
        block_ids = [b["block_id"] for b in layout["blocks"]]
        fields = {key: {"type": "string"} for key in (
            "action_quote", "closing_source_citation_id", "closing_source_quote", "closing_answer_quote")}
        fields.update(block_id={"type": "string", "enum": block_ids},
                      resolution={"type": "string", "enum": ["restored", "documented_terminal_state", "unresolved"]})
        fields["closing_source_citation_id"]["enum"] = list(dict.fromkeys(operation_source_ids))
        schema["properties"]["operation_boundary"] = {
            "type": "object", "additionalProperties": False,
            "properties": {"complete": {"type": "boolean"},
                "checked_blocks": {"type": "array", "items": {"type": "string", "enum": block_ids},
                                   "maxItems": len(block_ids)},
                "changes": {"type": "array", "maxItems": 16, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": fields, "required": list(fields)}}},
            "required": ["complete", "checked_blocks", "changes"]}
        schema["required"].append("operation_boundary")
    return result


OPERATION_BOUNDARY_INSTRUCTIONS = (
    "\nOPERATION BOUNDARY CHECK (mandatory for this manual/mixed procedure): Check every PROCEDURE_BLOCK "
    "against the complete source operation, including its continuation beyond router facets. Return "
    "operation_boundary.checked_blocks containing every block_id once. For every temporary state change "
    "introduced by the answer (for example loosening, opening, releasing pressure, disabling, changing "
    "mode or adjustment), add a changes entry. block_id identifies its original draft block; action_quote "
    "is an exact phrase in your final resolved answer. Resolve it with the source-documented restoration "
    "or documented safe terminal/hold state, not an invented restart. closing_source_citation_id and "
    "closing_source_quote must identify an exact quote in SOURCES; closing_answer_quote is an exact "
    "phrase of the final answer that carries out that closure. A restoration must follow action_quote; "
    "a documented terminal hold may be stated in the same instruction. Use resolution=restored "
    "or documented_terminal_state only when supported. If a closing action is missing but documented, "
    "rewrite to include it now, including its prerequisites, before claiming completion/readiness. "
    "Use replace if adding steps changes the block structure. If a necessary closing state is unknown, "
    "use unresolved, complete=false and outcome=no_sources. Complete=true requires every introduced "
    "temporary change resolved. Do not force unrelated operations into a segment. An empty changes list "
    "is correct only after checking every block and finding no introduced temporary state. "
    "Standing safety prerequisites such as isolation, keeping STOP pressed and wearing PPE are invariants "
    "to maintain, not temporary operational changes requiring reversal. Do not invent re-energization, "
    "restart or release of a safety prerequisite. Respect explicit before-only boundaries and documented "
    "safe hold states; do not perform an operation that the user explicitly excluded."
)


def _quote_text(value: str) -> str:
    # Typography/line wrapping only; keep case, negations, numbers and units.
    return " ".join(value.translate(str.maketrans({"\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"'})).split())


def verify_operation_boundary(parsed: dict, *, layout: dict, source_texts: dict[str, str]) -> dict:
    """Validate a same-call semantic closure audit, never infer physical safety."""
    out = deepcopy(parsed)
    audit = out.get("operation_boundary")
    expected = [b["block_id"] for b in layout["blocks"]]
    answer = str(out.get("answer") or "")
    body = _quote_text(answer)
    reason = "operation_boundary_incomplete"
    valid = (type(audit) is dict and audit.get("complete") is True
        and type(audit.get("checked_blocks")) is list
        and all(type(x) is str for x in audit["checked_blocks"])
        and len(audit["checked_blocks"]) == len(expected)
        and set(audit["checked_blocks"]) == set(expected)
        and type(audit.get("changes")) is list and len(audit["changes"]) <= 16)
    if valid:
        for change in audit["changes"]:
            if (type(change) is not dict or change.get("block_id") not in expected
                    or change.get("resolution") not in {"restored", "documented_terminal_state"}
                    or any(type(change.get(k)) is not str or not change[k].strip() for k in
                           ("action_quote", "closing_source_citation_id", "closing_source_quote", "closing_answer_quote"))):
                valid = False
                break
            source = source_texts.get(change["closing_source_citation_id"])
            action = _quote_text(change["action_quote"])
            closure = _quote_text(change["closing_answer_quote"])
            action_at = body.find(action)
            closure_start = (action_at if change["resolution"] == "documented_terminal_state"
                             else action_at + len(action))
            closure_at = body.find(closure, closure_start) if action_at >= 0 else -1
            if (source is None or _quote_text(change["closing_source_quote"]) not in _quote_text(source)
                    or action_at < 0 or closure_at < 0):
                valid = False
                reason = "operation_boundary_quote_or_order_invalid"
                break
    out["operation_boundary_validation"] = {"version": "ask-operation-boundary-v1",
        "complete": bool(valid), "answer_sha256": digest(answer.strip()),
        "checked_blocks": len(expected), "temporary_changes": len(audit["changes"]) if valid else None}
    if valid:
        # The reviewer explicitly cited these admitted records in its closure
        # audit. Carry those same references through normal allowed-citation
        # reconstruction; do not require it to repeat the IDs in a second list.
        out["citation_ids"] = list(dict.fromkeys(list(out.get("citation_ids") or []) + [
            change["closing_source_citation_id"] for change in audit["changes"]]))
    if not valid and out.get("outcome") in {"pass", "rewrite"}:
        out["outcome"] = "no_sources"
        out["reason"] = reason
    return out


PROTOCOL_INSTRUCTIONS = (
    "\nPROCEDURE REVIEW OUTPUT PROTOCOL: You remain the independent semantic and safety verifier. "
    "Check every mandatory facet, source instruction, precaution, numeric value, condition, and translation. "
    "PROCEDURE_STRUCTURE describes only source-occurrence numbers and literal source-note retention; "
    "it does not prove relevance, translation, or semantic coverage. Verify those against SOURCES. "
    "When source_sequence_required=false, CURRENT_ANSWER is a manual or mixed-source synthesis: its "
    "numbers are local presentation order, not the numbers of retrieved Step citations. Independently "
    "check that every requested operation is present in the correct source-supported operational order, "
    "with all prerequisites and precautions; sparse retrieved Step numbers do not define that sequence. "
    "A requirement is an obligation, not text that must be echoed: an ordered source sequence can "
    "satisfy 'all steps in order' without repeating that phrase. Never rewrite merely to echo a requirement. "
    "If CURRENT_ANSWER is fully correct, set outcome=pass, reply_mode=retain, answer='', edits=[]. "
    "Do not re-emit an unchanged answer. For this protocol, a complete replacement means the complete "
    "result after applying the selected mode, not mandatory transmission of unchanged blocks. "
    "For a local correction, use outcome=rewrite, reply_mode=edit, "
    "answer='', and only changed blocks from PROCEDURE_BLOCKS. Each edit.text must contain the complete "
    "corrected body of that block, WITHOUT its numbered heading or trailing block separators; the server "
    "preserves the original heading and separators. block_id is an opaque edit target; step_number gives "
    "the actual source step number. Preserve every instruction, source precaution, condition, numeric "
    "value, and citation. Never exchange instructions between Steps or add numbered steps inside a body. "
    "Use reply_mode=replace with a full answer and edits=[] when a safe correction needs a different "
    "block structure, missing steps, or extensive rewriting. For partial/no_sources use replace and "
    "report the unresolved requirements honestly. Keep all covered/missing contract keys verbatim. "
    "Every successful reply must cover every mandatory requirement. All original evidence, scope, "
    "safety, and no-invention rules above remain in force."
)


def resolve_reply(parsed: dict, *, answer: str, layout: dict,
                  required_facets: list[str], required_types: list[str]) -> dict:
    """Apply a complete review to exactly its input; no provider retry on errors.

    Absent reply_mode is a legacy full-answer response. It follows the existing
    validator and receives no compact-review proof. This is backwards-readable
    data, not a bypass of an invalid new-protocol response.
    """
    out = deepcopy(parsed)
    if "reply_mode" not in out:
        return out
    if layout.get("answer_sha256") != digest(answer) or "".join(
            b["text"] for b in layout["blocks"]) != answer:
        raise ValueError("procedure_review_input_changed")
    mode = out.get("reply_mode")
    outcome = out.get("outcome")
    edits = out.get("edits")
    if mode not in {"retain", "edit", "replace"} or type(edits) is not list:
        raise ValueError("procedure_review_protocol_invalid")
    if outcome not in {"pass", "rewrite", "partial", "no_sources"}:
        raise ValueError("procedure_review_outcome_invalid")
    complete = outcome in {"pass", "rewrite"}
    if complete and mode != "replace":
        if any(out.get(k) for k in ("missing_facets", "missing_answer_types", "missing_list_items")):
            raise ValueError("procedure_review_success_has_missing_requirements")
        for required, key in ((required_facets, "covered_facets"),
                              (required_types, "covered_answer_types")):
            values = out.get(key)
            if type(values) is not list or any(type(v) is not str for v in values):
                raise ValueError("procedure_review_coverage_shape_invalid")
            if not set(required).issubset(set(values)):
                raise ValueError("procedure_review_coverage_not_complete")
    if mode == "retain":
        if outcome != "pass" or edits or out.get("answer") != "":
            raise ValueError("procedure_review_retain_invalid")
        reviewed = answer
    elif mode == "edit":
        if outcome != "rewrite" or not edits or out.get("answer") != "":
            raise ValueError("procedure_review_edit_invalid")
        originals = {b["block_id"]: b["text"] for b in layout["blocks"]}
        replacements = {}
        for edit in edits:
            if type(edit) is not dict or set(edit) != {"block_id", "text"}:
                raise ValueError("procedure_review_edit_shape_invalid")
            key, value = edit["block_id"], edit["text"]
            if type(key) is not str or key not in originals or key in replacements or type(value) is not str:
                raise ValueError("procedure_review_edit_identity_invalid")
            replacements[key] = _edited_block(originals[key], value)
        reviewed = "".join(replacements.get(b["block_id"], b["text"]) for b in layout["blocks"])
        if _numbers(reviewed) != _numbers(answer) or not reviewed.strip():
            raise ValueError("procedure_review_edit_changes_layout")
    else:
        if edits or type(out.get("answer")) is not str:
            raise ValueError("procedure_review_replace_invalid")
        reviewed = out["answer"]
        if complete and not reviewed.strip():
            raise ValueError("procedure_review_empty_success")
        if outcome == "pass" and reviewed != answer:
            raise ValueError("procedure_review_pass_changes_answer")
    out["answer"] = reviewed
    out["procedure_review"] = {"version": VERSION, "reply_mode": mode,
        "input_answer_sha256": digest(answer), "output_answer_sha256": digest(reviewed),
        "edited_blocks": len(edits), "independent_semantic_review": True}
    return out
