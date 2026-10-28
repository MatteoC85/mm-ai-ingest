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

from .final_contract import source_note_present

VERSION = "ask-procedure-review-v2"
_NUMBERED = re.compile(r"(?m)^(?P<number>[1-9][0-9]*)[.)][ \t]+")
# Reviewers can reproduce a heading with Markdown, indentation, or a line
# break after its number. Recognize those forms before accepting an edit so
# formatting cannot accidentally become a new or changed procedure step.
_EDIT_NUMBERED = re.compile(
    r"(?m)^[ \t]*(?:\*\*|__)?(?P<number>[1-9][0-9]*)[.)]"
    r"(?:\*\*|__)?(?:[ \t]+|\r?\n|$)")


def repair_contract_feedback(contract: dict) -> dict:
    """Project diagnostic feedback, never sources or the reviewed answer.

    CURRENT_ANSWER, SOURCES, required facets/types and procedure structure have
    their own complete prompt fields. Serializing the whole prior verdict here
    duplicates those bodies and can consume the repair's reserved output budget.
    This projection is only model guidance; final validation still consumes the
    original unmodified contract and all of its source/answer bindings.
    """
    if not isinstance(contract, dict):
        return {}
    fields = ("passed", "complete", "reason", "missing_answer_facets",
              "missing_evidence_facets", "missing_answer_types", "semantic_missing_answer_types",
              "missing_list_items", "evidence_facet_coverage", "minimum_facet_coverage")
    out = {key: deepcopy(contract[key]) for key in fields if key in contract}
    if isinstance(contract.get("requirement_checks"), list):
        out["requirement_checks"] = [{key: deepcopy(check[key])
            for key in ("requirement", "passed", "reason") if key in check}
            for check in contract["requirement_checks"] if isinstance(check, dict)]
    prior = contract.get("semantic_verifier")
    if isinstance(prior, dict):
        out["semantic_verifier"] = {key: deepcopy(prior[key]) for key in (
            "outcome", "reason", "covered_facets", "missing_facets", "covered_answer_types",
            "missing_answer_types", "expected_list_items", "covered_list_items", "missing_list_items")
            if key in prior}
        boundary = prior.get("operation_boundary_validation")
        if isinstance(boundary, dict):
            out["semantic_verifier"]["operation_boundary_validation"] = {
                key: deepcopy(boundary[key]) for key in ("complete", "reason") if key in boundary}
    return out


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
                  operation_boundary: bool = False, source_ordered: bool = False,
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
    # A source-authored Step sequence owns its headings. The reviewer may
    # rewrite every body, but cannot replace that sequence with a free-text
    # numbered list. Unsupported or incomplete evidence remains a rejection.
    schema["properties"]["reply_mode"] = {"type": "string", "enum": (
        ["retain", "edit", "reject"] if source_ordered else ["replace", "retain", "edit"])}
    if source_ordered:
        schema["properties"]["answer"] = {"type": "string", "enum": [""]}
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
        if source_ordered:
            # Compact source-backed partial-order edges, not copies of the draft
            # or source body. IDs target the immutable resolved block layout.
            edge_fields = {key: {"type": "string", "maxLength": 400} for key in
                           ("before_answer_quote", "after_answer_quote", "source_quote")}
            edge_fields.update(before_block_id={"type": "string", "enum": block_ids},
                               after_block_id={"type": "string", "enum": block_ids},
                               source_citation_id={"type": "string", "enum": list(dict.fromkeys(operation_source_ids))})
            boundary = schema["properties"]["operation_boundary"]
            boundary["properties"]["precedence"] = {"type": "array", "maxItems": 16,
                "items": {"type": "object", "additionalProperties": False,
                          "properties": edge_fields, "required": list(edge_fields)}}
            boundary["required"].append("precedence")
            # The same action/closure/source quote already exists in an edge.
            # Send only its index; reconstruct the legacy canonical change below
            # before running the unchanged source/closure validation.
            boundary["properties"].pop("changes")
            boundary["required"].remove("changes")
            boundary["properties"]["change_edges"] = {"type": "array", "maxItems": 16,
                "items": {"type": "object", "additionalProperties": False,
                    "properties": {"edge_index": {"type": "integer", "minimum": 0, "maximum": 15},
                        "resolution": {"type": "string", "enum": ["restored", "documented_terminal_state", "unresolved"]}},
                    "required": ["edge_index", "resolution"]}}
            boundary["required"].append("change_edges")

        schema["required"].append("operation_boundary")
    return result


OPERATION_BOUNDARY_INSTRUCTIONS = (
    "\nOPERATION BOUNDARY CHECK (mandatory for this reviewed procedure): Check every PROCEDURE_BLOCK "
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
    "For a manual/mixed draft only, use replace if adding steps changes the block structure. For a source-ordered draft preserve its immutable Step structure; if it cannot hold the required safe sequence, reject. If a necessary closing state is unknown, "
    "use unresolved, complete=false and outcome=no_sources. Complete=true requires every introduced "
    "temporary change resolved. Do not force unrelated operations into a segment. An empty changes list "
    "is correct only after checking every block and finding no introduced temporary state. "
    "Standing safety prerequisites such as isolation, keeping STOP pressed and wearing PPE are invariants "
    "to maintain, not temporary operational changes requiring reversal. Do not invent re-energization, "
    "restart or release of a safety prerequisite. Respect explicit before-only boundaries and documented "
    "safe hold states; do not perform an operation that the user explicitly excluded."
)


SOURCE_ORDERED_BOUNDARY_INSTRUCTIONS = (
    "\nSOURCE-ORDERED OPERATION AUDIT: Check every FINAL resolved block against the complete source "
    "operation, including continuation beyond router facets. Return checked_blocks with every "
    "PROCEDURE_BLOCK block_id exactly once. For each introduced/changed operation, include its source "
    "preparation, positioning, mode/energy conditions, locking, intermediate stages and terminal/restoration "
    "conditions. Re-establish prerequisites invalidated by intervening actions. Reduced speed is not proof "
    "of safety. Restore required protections BEFORE EACH dependent movement/test, not just final restart. "
    "operation_boundary.precedence lists every required prerequisite-to-action, temporary-change-to-closure "
    "and restoration-to-dependent-operation relation. Each edge names before_block_id, after_block_id, "
    "exact before_answer_quote/after_answer_quote from those FINAL blocks and source_citation_id/source_quote "
    "proving the dependency. Use short, unique-in-block spans (maximum400characters), never full answers "
    "or sources. At most16edges: reject if necessary proof cannot fit. Every materially edited block must "
    "participate in a proved edge. For each temporary change (opening, releasing, adjustment, disabling, "
    "mode change, etc.), change_edges references the zero-based precedence edge_index connecting its action "
    "to documented closure, with resolution=restored or documented_terminal_state. No repeated quotes in "
    "change_edges. A documented terminal hold may have identical before/after quotes in one block; otherwise "
    "the action must precede its closure. Unknown closure: complete=false, outcome=no_sources. Complete=true "
    "means every dependency and temporary change has been audited/resolved; empty lists are allowed only "
    "when no corresponding obligation exists, and no substantive edit may have empty precedence. "
    "Standing isolation/STOP/PPE prerequisites are invariants, not changes requiring reversal. Never invent "
    "re-energization/restart or safety release. Respect before-only requests and documented safe holds. Keep "
    "source Step structure and original notes; add a required prerequisite earlier without moving its note. "
    "Reject if this layout cannot hold a complete safe sequence. Do not introduce an extra corrective "
    "intervention merely because a manual describes it alongside the requested check. If necessary, include "
    "its complete documented subprocedure. For a requested path/process, state every necessary documented "
    "intermediate stage in its prescribed order; generic categories cannot replace the sequence. Never "
    "invent an order absent from SOURCES."
)


def _quote_text(value: str) -> str:
    # Typography/line wrapping only; keep case, negations, numbers and units.
    return " ".join(value.translate(str.maketrans({"\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"'})).split())


def _canonical_boundary(audit: Any) -> dict | None:
    """Expand source-ordered wire references; indices add no source authority."""
    if type(audit) is not dict:
        return None
    if "change_edges" not in audit:
        return deepcopy(audit)  # internal canonical/legacy representation
    refs, edges = audit.get("change_edges"), audit.get("precedence")
    if ("changes" in audit or type(refs) is not list or len(refs) > 16 or type(edges) is not list):
        return None
    changes, seen = [], set()
    for ref in refs:
        if (type(ref) is not dict or set(ref) != {"edge_index", "resolution"}
                or type(ref.get("edge_index")) is not int or not 0 <= ref["edge_index"] < len(edges)
                or type(ref.get("resolution")) is not str
                or ref["resolution"] not in {"restored", "documented_terminal_state", "unresolved"}
                or ref["edge_index"] in seen):
            return None
        edge = edges[ref["edge_index"]]
        keys = ("before_block_id", "before_answer_quote", "after_block_id", "after_answer_quote",
                "source_citation_id", "source_quote")
        if type(edge) is not dict or any(type(edge.get(k)) is not str for k in keys):
            return None
        seen.add(ref["edge_index"])
        changes.append({"block_id": edge["before_block_id"], "action_quote": edge["before_answer_quote"],
            "closing_source_citation_id": edge["source_citation_id"], "closing_source_quote": edge["source_quote"],
            "closing_answer_quote": edge["after_answer_quote"], "resolution": ref["resolution"]})
    result = {k: deepcopy(v) for k, v in audit.items() if k != "change_edges"}
    result["changes"] = changes
    return result


def verify_operation_boundary(parsed: dict, *, layout: dict, source_texts: dict[str, str],
                              source_ordered: bool = False) -> dict:
    """Validate a same-call semantic closure audit, never infer physical safety."""
    out = deepcopy(parsed)
    audit = _canonical_boundary(out.get("operation_boundary")) if source_ordered else out.get("operation_boundary")
    if source_ordered and audit is not None:
        out["operation_boundary"] = audit
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
    edge_sources = []
    edge_count = None
    if valid and source_ordered:
        # This proves exact quoted spans and their declared order. It does not
        # infer semantic completeness or machine state: that remains the same
        # independent reviewer's obligation against all admitted sources.
        edges = audit.get("precedence")
        final_layout = block_layout(answer)
        resolved_ids = [b["block_id"] for b in final_layout["blocks"]]
        valid = (type(edges) is list and len(edges) <= 16 and resolved_ids == expected)
        spans, position = {}, 0
        for block in final_layout["blocks"]:
            text = _quote_text(block["text"])
            spans[block["block_id"]] = (position, text)
            position += len(text) + 1
        touched = set()
        if valid:
            for edge in edges:
                keys = ("before_block_id", "after_block_id", "before_answer_quote",
                        "after_answer_quote", "source_citation_id", "source_quote")
                if (type(edge) is not dict or any(type(edge.get(k)) is not str or not edge[k].strip() for k in keys)
                        or edge["before_block_id"] not in spans or edge["after_block_id"] not in spans
                        or any(len(edge[k]) > 400 for k in ("before_answer_quote", "after_answer_quote", "source_quote"))):
                    valid = False
                    break
                before_base, before_body = spans[edge["before_block_id"]]
                after_base, after_body = spans[edge["after_block_id"]]
                before = _quote_text(edge["before_answer_quote"])
                after = _quote_text(edge["after_answer_quote"])
                source = source_texts.get(edge["source_citation_id"])
                same_terminal_hold = bool(edge["before_block_id"] == edge["after_block_id"]
                    and before == after and any(change["resolution"] == "documented_terminal_state"
                        and change["block_id"] == edge["before_block_id"]
                        and _quote_text(change["action_quote"]) == before
                        and _quote_text(change["closing_answer_quote"]) == after for change in audit["changes"]))
                if (source is None or _quote_text(edge["source_quote"]) not in _quote_text(source)
                        or before_body.count(before) != 1 or after_body.count(after) != 1
                        or (not same_terminal_hold and before_base + before_body.find(before) + len(before)
                            > after_base + after_body.find(after))):
                    valid = False
                    break
                touched.update((edge["before_block_id"], edge["after_block_id"]))
                edge_sources.append(edge["source_citation_id"])
        if valid:
            for change in audit["changes"]:
                action, closure = _quote_text(change["action_quote"]), _quote_text(change["closing_answer_quote"])
                original_block = spans.get(change["block_id"])
                same_hold = bool(change["resolution"] == "documented_terminal_state"
                    and action == closure and original_block is not None
                    and original_block[1].count(action) == 1)
                proved_edge = any(edge["before_block_id"] == change["block_id"]
                    and _quote_text(edge["before_answer_quote"]) == action
                    and _quote_text(edge["after_answer_quote"]) == closure for edge in edges)
                if not (same_hold or proved_edge):
                    valid = False
                    break
        if valid:
            originals = {b["block_id"]: _quote_text(b["text"]) for b in layout["blocks"]}
            changed = {bid for bid, (_, text) in spans.items() if text != originals[bid]}
            valid = changed.issubset(touched)
        if not valid:
            reason = "operation_precedence_missing_ambiguous_or_invalid"
        else:
            edge_count = len(edges)
    out["operation_boundary_validation"] = {"version": "ask-operation-boundary-v1",
        "complete": bool(valid), "answer_sha256": digest(answer.strip()),
        "checked_blocks": len(expected), "temporary_changes": len(audit["changes"]) if valid else None,
        **({"precedence_checked": bool(valid), "precedence_edges": edge_count}
           if source_ordered else {})}
    if valid:
        # The reviewer explicitly cited these admitted records in its closure
        # audit. Carry those same references through normal allowed-citation
        # reconstruction; do not require it to repeat the IDs in a second list.
        out["citation_ids"] = list(dict.fromkeys(list(out.get("citation_ids") or []) + [
            change["closing_source_citation_id"] for change in audit["changes"]] + edge_sources))
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
    "When source_sequence_required=false, use reply_mode=replace with a full answer and edits=[] "
    "when a safe correction needs a different block structure, missing steps, or extensive rewriting. "
    "For partial/no_sources in that manual/mixed mode, use replace and report the unresolved requirements. "
    "When source_sequence_required=true, the schema permits only retain, edit or reject and answer must "
    "always be empty. Source-authored Step numbers and order are server-owned. Even an extensive "
    "correction must use edit with all affected blocks, preserving every source instruction and safety "
    "condition in its own Step. Source-authored notes and warnings must remain verbatim in their original "
    "Step body; never omit, paraphrase, or move them even if your coverage verdict is complete. "
    "Never renumber sparse source Steps or turn a Step into a separate heading. "
    "If the admitted Step set cannot satisfy the request without changing its structure, return "
    "reply_mode=reject, outcome=no_sources (or partial only where otherwise permitted), answer='', "
    "edits=[], and the unresolved requirements. Never claim successful coverage of an omitted Step. "
    "Keep all covered/missing contract keys verbatim. "
    "Every successful reply must cover every mandatory requirement. All original evidence, scope, "
    "safety, and no-invention rules above remain in force."
)



def _source_notes_preserved(notes: tuple[dict, ...], *, layout: dict, reviewed: str) -> bool:
    """Check already-admitted, source-authored notes in their original Step.

    Coverage assertions from the model cannot delete a warning or move it to a
    different operation. This function only rejects; it never edits reviewed
    text or invents a substitute source note.
    """
    original_blocks = {str(_numbers(b["text"])[0]): b["text"] for b in layout["blocks"]
                       if _NUMBERED.match(b["text"])}
    final_blocks = {str(_numbers(b["text"])[0]): b["text"] for b in block_layout(reviewed)["blocks"]
                    if _NUMBERED.match(b["text"])}
    for note in notes:
        number = str(note.get("source_number") or "")
        texts = note.get("texts")
        if (number not in original_blocks or number not in final_blocks
                or not isinstance(texts, (list, tuple)) or not texts
                or any(type(text) is not str or not text for text in texts)):
            return False
        for text in texts:
            if (not source_note_present(original_blocks[number], text)
                    or not source_note_present(final_blocks[number], text)):
                return False
            for other, block in final_blocks.items():
                if (other != number and source_note_present(block, text)
                        and not source_note_present(original_blocks.get(other, ""), text)):
                    return False
    return True


def resolve_reply(parsed: dict, *, answer: str, layout: dict,
                  required_facets: list[str], required_types: list[str],
                  source_ordered: bool = False, source_notes: tuple[dict, ...] = ()) -> dict:
    """Apply a complete review to exactly its input; no provider retry on errors.

    Absent reply_mode is a legacy full-answer response. It follows the existing
    validator and receives no compact-review proof. This is backwards-readable
    data, not a bypass of an invalid new-protocol response.
    """
    out = deepcopy(parsed)
    if "reply_mode" not in out:
        if source_ordered:
            raise ValueError("procedure_review_source_sequence_protocol_required")
        return out
    if layout.get("answer_sha256") != digest(answer) or "".join(
            b["text"] for b in layout["blocks"]) != answer:
        raise ValueError("procedure_review_input_changed")
    mode = out.get("reply_mode")
    outcome = out.get("outcome")
    edits = out.get("edits")
    modes = {"retain", "edit", "reject"} if source_ordered else {"retain", "edit", "replace"}
    if mode not in modes or type(edits) is not list:
        raise ValueError("procedure_review_protocol_invalid")
    if outcome not in {"pass", "rewrite", "partial", "no_sources"}:
        raise ValueError("procedure_review_outcome_invalid")
    if source_ordered and out.get("answer") != "":
        raise ValueError("procedure_review_source_sequence_answer_not_empty")
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
    if mode == "reject":
        if outcome not in {"partial", "no_sources"} or edits or out.get("answer") != "":
            raise ValueError("procedure_review_reject_invalid")
        reviewed = ""
    elif mode == "retain":
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
    notes_preserved = bool(not source_ordered or not complete or _source_notes_preserved(
        source_notes, layout=layout, reviewed=reviewed))
    if not notes_preserved:
        out.update(outcome="no_sources", reason="source_step_note_binding_incomplete")
        reviewed = ""
    out["answer"] = reviewed
    out["procedure_review"] = {"version": VERSION, "reply_mode": mode,
        "input_answer_sha256": digest(answer), "output_answer_sha256": digest(reviewed),
        "edited_blocks": len(edits), "independent_semantic_review": True,
        "source_sequence_immutable": source_ordered,
        "source_note_bindings_complete": notes_preserved if source_ordered else None}
    return out
