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
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable

from .final_contract import source_note_present

VERSION = "ask-procedure-review-v2"

# Only these fixed parser labels may leave the review boundary. Never include
# provider text, source excerpts, exception arguments, or the rejected draft.
_PROTOCOL_FAILURE_CODES = frozenset({
    "procedure_reference_catalog_invalid",
    "procedure_reference_source_missing",
    "procedure_reference_shape_invalid",
    "procedure_reference_range_invalid",
    "procedure_reference_deleted_unit",
    "procedure_reference_edit_invalid",
    "procedure_reference_answer_changed",
    "procedure_review_wire_alias_unknown",
    "procedure_review_wire_citations_invalid",
    "procedure_review_wire_edits_invalid",
    "procedure_review_micro_edit_invalid",
    "procedure_review_micro_edit_anchor_ambiguous_or_missing",
    "procedure_review_wire_boundary_invalid",
    "procedure_review_wire_edge_invalid",
    "procedure_review_source_sequence_protocol_required",
    "procedure_review_input_changed",
    "procedure_review_protocol_invalid",
    "procedure_review_outcome_invalid",
    "procedure_review_source_sequence_answer_not_empty",
    "procedure_review_success_has_missing_requirements",
    "procedure_review_coverage_shape_invalid",
    "procedure_review_coverage_not_complete",
    "procedure_review_reject_invalid",
    "procedure_review_retain_invalid",
    "procedure_review_edit_invalid",
    "procedure_review_edit_shape_invalid",
    "procedure_review_edit_identity_invalid",
    "procedure_review_edit_changes_step_sequence",
    "procedure_review_edit_changes_layout",
    "procedure_review_replace_invalid",
    "procedure_review_empty_success",
    "procedure_review_pass_changes_answer",
    "procedure_review_translation_incomplete",
    "procedure_review_judge_reply_invalid",
})


def protocol_failure_code(exc: ValueError) -> str:
    code = str(exc)
    return code if code in _PROTOCOL_FAILURE_CODES else "procedure_review_protocol_invalid"


class _ReferenceFailure(ValueError):
    """Fixed parser category and bounded numbers, never wire text or identities."""
    def __init__(self, code: str, *, stage: str, reason: str, value=None, count=0):
        super().__init__(code)
        value = value if type(value) is dict else {}
        def number(raw):
            return max(-1000000, min(1000000, raw)) if type(raw) is int else None
        self.diagnostic = {"version": "ask-procedure-reference-diagnostic-v1",
            "stage": stage, "reason": reason, "unit_count": number(count),
            "first": number(value.get("first")), "last": number(value.get("last"))}


def protocol_failure_diagnostic(exc: ValueError) -> dict:
    """Do not project arbitrary exception attributes or provider-controlled text."""
    if type(exc) is not _ReferenceFailure:
        return {}
    value = exc.diagnostic
    if (value.get("stage") not in {"edit_original", "action", "next", "proof"}
            or value.get("reason") not in {"shape", "reversed", "out_of_bounds", "empty_span",
                                           "replacement_missing", "kind_invalid"}):
        return {}
    return {"version": "ask-procedure-reference-diagnostic-v1",
            "stage": value["stage"], "reason": value["reason"],
            **{key: max(-1000000, min(1000000, value[key])) if type(value.get(key)) is int else None
               for key in ("unit_count", "first", "last")}}


def _manual_attachment_removed(bindings, original_answer: str, reviewed_answer: str,
                               edits: list[dict]) -> bool:
    """Recognize exact appendix deletion inside its original owning block.

    Its operational prefix cannot change except whitespace. Other blocks may
    contain explicit edits from the same already-verified review; never infer,
    normalize or approve their meaning here. All global review guards still run.
    """
    original_blocks = block_layout(original_answer)["blocks"]
    final_blocks = block_layout(reviewed_answer)["blocks"]
    if ([b["block_id"] for b in original_blocks] != [b["block_id"] for b in final_blocks]
            or type(edits) is not list):
        return False
    originals = {b["block_id"]: b["text"] for b in original_blocks}
    replacements = {}
    for edit in edits:
        if (type(edit) is not dict or set(edit) != {"block_id", "text"}
                or type(edit.get("block_id")) is not str or edit["block_id"] not in originals
                or edit["block_id"] in replacements or type(edit.get("text")) is not str):
            return False
        try:
            replacements[edit["block_id"]] = _edited_block(originals[edit["block_id"]], edit["text"])
        except ValueError:
            return False
    for before, after in zip(original_blocks, final_blocks):
        expected = replacements.get(before["block_id"], before["text"])
        if " ".join(expected.split()) != " ".join(after["text"].split()):
            return False
    original = " ".join(original_blocks[-1]["text"].split())
    reviewed = " ".join(final_blocks[-1]["text"].split())
    notes = " ".join("- " + " ".join(binding["text"].split()) for binding in bindings)
    for heading in ("Manual support", "Supporto dal manuale"):
        suffix = " " + heading + " " + notes
        if (original.endswith(suffix)
                and " ".join(original_answer.split()).count(suffix) == 1):
            return bool(reviewed) and original[:-len(suffix)] == reviewed
    return False


def manual_note_projection(bindings, *, original_answer: str, reviewed_answer: str,
                           review: dict, admitted_ids: set[str],
                           final_citation_ids: set[str] | None = None) -> tuple[list[str], dict]:
    """Keep references for unchanged server-bound manual excerpts, not authority.

    Only an independently reviewed, byte-identical retain can restore a source
    reference omitted by the reviewer. An edited/translated note needs the
    independently verified block edit and its original references. Only a pure,
    independently reviewed removal of the complete server-bound appendix can
    omit its references, independently of explicit reviewed edits to other blocks.
    Original admission is distinct from final display: this
    function never attributes new prose to a source or grants a semantic PASS.
    """
    if not bindings:
        return [], {}
    if type(bindings) is not list:
        raise ValueError("manual_note_binding_invalid")
    required = []
    for binding in bindings:
        if type(binding) is not dict:
            raise ValueError("manual_note_binding_invalid")
        text, ids = binding.get("text"), binding.get("citation_ids")
        if (type(text) is not str or not text or type(ids) is not list or not ids
                or any(type(cid) is not str or cid not in admitted_ids for cid in ids)
                or not source_note_present(original_answer, text)):
            raise ValueError("manual_note_binding_invalid")
        required.extend(ids)
    required = list(dict.fromkeys(required))
    audit = review.get("procedure_review") or {}
    mode = audit.get("reply_mode")
    if (mode == "retain" and original_answer == reviewed_answer
            and audit.get("independent_semantic_review") is True
            and audit.get("input_answer_sha256") == digest(original_answer)
            and audit.get("output_answer_sha256") == digest(reviewed_answer)):
        if final_citation_ids is not None and not set(required).issubset(final_citation_ids):
            raise ValueError("manual_note_binding_invalid")
        return required, {"version": "ask-manual-note-binding-v1", "complete": True,
                          "basis": "unchanged_reviewed_source_excerpt", "notes": len(bindings),
                          "citation_ids": required}
    if mode != "edit":
        raise ValueError("manual_note_binding_invalid")
    boundary = review.get("operation_boundary_validation") or {}
    if (audit.get("independent_semantic_review") is not True
            or audit.get("source_sequence_immutable") is not True
            or audit.get("input_answer_sha256") != digest(original_answer)
            or audit.get("output_answer_sha256") != digest(reviewed_answer)
            or boundary.get("complete") is not True
            or boundary.get("precedence_checked") is not True
            or boundary.get("answer_sha256") != digest(reviewed_answer)):
        raise ValueError("manual_note_binding_invalid")
    changed_blocks = {edit.get("block_id") for edit in review.get("edits") or [] if type(edit) is dict}
    original_blocks = block_layout(original_answer)["blocks"]
    if (review.get("outcome") == "rewrite"
            and audit.get("source_note_bindings_complete") is True
            and original_blocks[-1]["block_id"] in changed_blocks
            and _manual_attachment_removed(bindings, original_answer, reviewed_answer,
                                           review.get("edits"))):
        return [], {"version": "ask-manual-note-binding-v1", "complete": True,
                    "basis": "reviewed_exact_manual_attachment_removal", "notes": len(bindings),
                    "removed_notes": len(bindings), "citation_ids": []}
    if (not set(required).issubset(set(review.get("citation_ids") or []))
            or final_citation_ids is not None and not set(required).issubset(final_citation_ids)):
        raise ValueError("manual_note_binding_invalid")
    for binding in bindings:
        if source_note_present(reviewed_answer, binding["text"]):
            continue
        owners = [block["block_id"] for block in original_blocks
                  if source_note_present(block["text"], binding["text"])]
        if len(owners) != 1 or owners[0] not in changed_blocks:
            raise ValueError("manual_note_binding_invalid")
    return [], {"version": "ask-manual-note-binding-v1", "complete": True,
                "basis": "reviewed_edit_with_original_source_refs", "notes": len(bindings),
                "citation_ids": required}


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


def review_wire_aliases(layout: dict, source_ids: list[str]) -> dict:
    """Request-local names encode existing identities; they grant no authority."""
    sources = list(dict.fromkeys(source_ids))
    blocks = [b["block_id"] for b in layout["blocks"]]
    if (not sources or any(type(cid) is not str or not cid for cid in sources)
            or len(set(blocks)) != len(blocks)):
        raise ValueError("procedure_review_alias_catalog_invalid")
    return {"sources": {f"s{i+1}": cid for i, cid in enumerate(sources)},
            "blocks": {f"b{i+1}": bid for i, bid in enumerate(blocks)}}


def reference_units(text: str) -> tuple[tuple[int, int, str], ...]:
    """Lossless punctuation/line units, bounded without semantic classification."""
    cuts = [m.end() for m in re.finditer(r"(?:[.!?;:,](?=\s)|[。！？；：]|\n)[ \t\r\n]*", text)]
    cuts.append(len(text))
    units, start = [], 0
    for end in cuts:
        while end - start > 160:
            at = text.rfind(" ", start + 1, start + 160)
            stop = at + 1 if at > start else start + 160
            units.append((start, stop, text[start:stop])); start = stop
        if end > start:
            units.append((start, end, text[start:end])); start = end
    assert "".join(unit[2] for unit in units) == text
    return tuple(units)


def reference_catalog(layout: dict, aliases: dict, sources_block: str,
                      source_texts: dict[str, str]) -> dict:
    """Bind only complete emitted sources and the immutable original block bodies."""
    sources, blocks = {}, {}
    ids = list(aliases["sources"].values())
    headers = ["[" + cid + "] " for cid in ids]
    positions = []
    for header in headers:
        matches = list(re.finditer(r"(?m)^" + re.escape(header), sources_block))
        if len(matches) != 1:
            raise ValueError("procedure_reference_catalog_invalid")
        positions.append(matches[0].start())
    if positions != sorted(positions) or not positions or sources_block[:positions[0]].strip():
        raise ValueError("procedure_reference_catalog_invalid")
    for index, (alias, cid) in enumerate(aliases["sources"].items()):
        segment = sources_block[positions[index]:positions[index + 1] if index + 1 < len(ids) else len(sources_block)]
        header, separator, body = segment.partition("\n")
        body = body.strip()
        if not separator or not body or cid not in source_texts or body not in source_texts[cid]:
            raise ValueError("procedure_reference_source_missing")
        sources[alias] = {"citation_id": cid, "header": header, "text": body,
                          "units": reference_units(body), "grounding_sha256": digest(source_texts[cid])}
    original_blocks = {block["block_id"]: block["text"] for block in layout["blocks"]}
    for alias, bid in aliases["blocks"].items():
        prefix, body, suffix = _block_parts(original_blocks[bid])
        if not body.strip():
            # This is inside the caller's catalog preflight catch, before the
            # provider/schema boundary. Empty owners cannot define a valid span.
            raise ValueError("procedure_reference_catalog_invalid")
        blocks[alias] = {"block_id": bid, "prefix": prefix, "text": body, "suffix": suffix,
                         "units": reference_units(body)}
    return {"version": "ask-procedure-references-v2", "answer_sha256": layout["answer_sha256"],
            "sources": sources, "blocks": blocks}


def reference_views(catalog: dict) -> tuple[str, list[dict]]:
    """An alternative lossless presentation, never an additional source selection."""
    sources = []
    for alias, source in catalog["sources"].items():
        sources.append(source["header"] + "\nSOURCE_UNITS " + alias + ":" + json.dumps(
            [[i, unit[2]] for i, unit in enumerate(source["units"])], ensure_ascii=False, separators=(",", ":")))
    blocks = [{"block_id": alias, "heading": block["prefix"], "separator": block["suffix"],
               "units": [[i, unit[2]] for i, unit in enumerate(block["units"])]}
              for alias, block in catalog["blocks"].items()]
    return "\n\n".join(sources), blocks


def _reference_range(value: Any, units: tuple, *, stage: str) -> tuple[int, int, str]:
    if (type(value) is not dict or set(value) != {"first", "last"}
            or type(value.get("first")) is not int or type(value.get("last")) is not int):
        raise _ReferenceFailure("procedure_reference_shape_invalid", stage=stage,
                                reason="shape", value=value, count=len(units))
    first, last = value["first"], value["last"]
    if first > last:
        raise _ReferenceFailure("procedure_reference_range_invalid", stage=stage,
                                reason="reversed", value=value, count=len(units))
    if not 0 <= first <= last < len(units):
        raise _ReferenceFailure("procedure_reference_range_invalid", stage=stage,
                                reason="out_of_bounds", value=value, count=len(units))
    selected = units[first:last + 1]
    text = "".join(unit[2] for unit in selected)
    if not text.strip():
        raise _ReferenceFailure("procedure_reference_range_invalid", stage=stage,
                                reason="empty_span", value=value, count=len(units))
    return selected[0][0], selected[-1][1], text


def _answer_projection(text: str) -> str:
    # Match only the existing final redactor's innocuous whitespace projection.
    # Citation/identifier removal or content masking is deliberately NOT copied.
    return re.sub(r"[ \t]+", " ", text).strip()


def _project_offset(text: str, offset: int) -> int:
    collapsed = re.sub(r"[ \t]+", " ", text)
    leading = len(collapsed) - len(collapsed.lstrip())
    return len(re.sub(r"[ \t]+", " ", text[:offset])) - leading


@dataclass(frozen=True)
class ReferenceProof:
    original_sha256: str
    raw_answer_sha256: str
    answer_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    # block IDs, final absolute start/end offsets, source ID and literal span.
    edges: tuple[tuple, ...]


def decode_reference_reply(parsed: dict, aliases: dict, *, layout: dict,
                           catalog: dict) -> tuple[dict, ReferenceProof]:
    """Resolve this single review's explicit spans; never infer action semantics."""
    if catalog["answer_sha256"] != layout["answer_sha256"]:
        raise ValueError("procedure_review_input_changed")
    out = deepcopy(parsed)
    def owner(alias, kind):
        if type(alias) is not str or alias not in catalog[kind]:
            raise ValueError("procedure_review_wire_alias_unknown")
        return catalog[kind][alias]
    citations = out.get("citation_ids")
    if type(citations) is not list:
        raise ValueError("procedure_review_wire_citations_invalid")
    out["citation_ids"] = [owner(alias, "sources")["citation_id"] for alias in citations]
    edits = out.get("edits")
    if type(edits) is not list:
        raise ValueError("procedure_reference_edit_invalid")
    changes, canonical_edits = {}, []
    for edit in edits:
        if type(edit) is not dict or set(edit) != {"block_id", "original", "replacement_parts"}:
            raise ValueError("procedure_reference_edit_invalid")
        alias = edit["block_id"]; block = owner(alias, "blocks")
        start, end, _ = _reference_range(edit["original"], block["units"], stage="edit_original")
        parts = edit["replacement_parts"]
        if (alias in changes or type(parts) is not list or not 1 <= len(parts) <= 16
                or any(type(part) is not str or not part.strip() for part in parts)):
            raise ValueError("procedure_reference_edit_invalid")
        replacement = "".join(parts)
        body = block["text"][:start] + replacement + block["text"][end:]
        original = block["prefix"] + block["text"] + block["suffix"]
        if _edited_block(original, body) != block["prefix"] + body.strip() + block["suffix"]:
            raise ValueError("procedure_reference_edit_invalid")
        changes[alias] = (start, end, tuple(parts), body)
        canonical_edits.append({"block_id": block["block_id"], "text": body})
    final, positions, cursor = [], {}, 0
    for alias, block in catalog["blocks"].items():
        raw_body = changes[alias][3] if alias in changes else block["text"]
        body = raw_body.strip() if alias in changes else raw_body
        text = block["prefix"] + body + block["suffix"]
        leading = len(raw_body) - len(raw_body.lstrip()) if alias in changes else 0
        positions[alias] = cursor + len(block["prefix"]) - leading
        final.append(text); cursor += len(text)
    answer = "".join(final)
    def answer_span(ref, stage):
        if type(ref) is not dict or set(ref) != {"block_id", "kind", "range"}:
            raise _ReferenceFailure("procedure_reference_shape_invalid", stage=stage, reason="shape")
        alias = ref["block_id"]; block = owner(alias, "blocks")
        interval = ref["range"]
        if ref["kind"] == "original":
            start, end, text = _reference_range(interval, block["units"], stage=stage)
            if alias in changes:
                old_start, old_end, parts, _ = changes[alias]
                if start < old_end and end > old_start:
                    raise ValueError("procedure_reference_deleted_unit")
                if start >= old_end:
                    shift = len("".join(parts)) - (old_end - old_start)
                    start += shift; end += shift
        elif ref["kind"] == "replacement" and alias in changes:
            old_start, _, parts, _ = changes[alias]
            units, at = [], old_start
            for part in parts:
                units.append((at, at + len(part), part)); at += len(part)
            start, end, text = _reference_range(interval, tuple(units), stage=stage)
        else:
            raise _ReferenceFailure("procedure_reference_range_invalid", stage=stage,
                reason="replacement_missing" if ref["kind"] == "replacement" else "kind_invalid",
                value=interval)
        # Whitespace is retained in the answer, but not part of action boundaries.
        leading = len(text) - len(text.lstrip()); trailing = len(text) - len(text.rstrip())
        start += leading; end -= trailing; text = text.strip()
        start += positions[alias]; end += positions[alias]
        if answer[start:end] != text:
            raise ValueError("procedure_reference_answer_changed")
        return block["block_id"], start, end, text
    audit = out.get("operation_boundary")
    if (type(audit) is not dict or set(audit) != {"complete", "checked_blocks", "precedence", "change_edges"}
            or type(audit.get("checked_blocks")) is not list or type(audit.get("precedence")) is not list
            or len(audit["precedence"]) > 16):
        raise ValueError("procedure_review_wire_boundary_invalid")
    audit["checked_blocks"] = [owner(alias,"blocks")["block_id"] for alias in audit["checked_blocks"]]
    edges, proof_edges = [], []
    for edge in audit["precedence"]:
        if type(edge) is not dict or set(edge) != {"action", "next", "proof"}:
            raise ValueError("procedure_review_wire_edge_invalid")
        before = answer_span(edge["action"], "action")
        after = answer_span(edge["next"], "next")
        proof = edge["proof"]
        if type(proof) is not dict or set(proof) != {"source", "range"}:
            raise _ReferenceFailure("procedure_reference_shape_invalid", stage="proof", reason="shape")
        source = owner(proof["source"], "sources")
        _, _, quote = _reference_range(proof["range"], source["units"], stage="proof")
        edges.append({"before_block_id":before[0], "before_answer_quote":before[3],
                      "after_block_id":after[0], "after_answer_quote":after[3],
                      "source_citation_id":source["citation_id"], "source_quote":quote})
        proof_edges.append((before[0], _project_offset(answer,before[1]), _project_offset(answer,before[2]),
                            after[0], _project_offset(answer,after[1]), _project_offset(answer,after[2]),
                            source["citation_id"], quote))
    out["edits"] = canonical_edits; audit["precedence"] = edges
    return out, ReferenceProof(layout["answer_sha256"], digest(answer), digest(_answer_projection(answer)),
        tuple((source["citation_id"], source["grounding_sha256"]) for source in catalog["sources"].values()),
        tuple(proof_edges))


_WIRE_EDGE_FIELDS = {"from": "before_block_id", "action": "before_answer_quote",
    "to": "after_block_id", "next": "after_answer_quote", "source": "source_citation_id",
    "proof": "source_quote"}


def decode_review_wire_aliases(parsed: dict, aliases: dict, *, layout: dict | None = None) -> dict:
    """Lossless expansion before the existing layout/source/order validators.

    Only explicit identity fields are decoded. Source quotes and answer text are
    never rewritten, and an unknown name cannot become a new citation or block.
    """
    out = deepcopy(parsed)
    sources, blocks = aliases["sources"], aliases["blocks"]
    def identity(value, catalog):
        if type(value) is not str or value not in catalog:
            raise ValueError("procedure_review_wire_alias_unknown")
        return catalog[value]
    citations = out.get("citation_ids", [])
    if type(citations) is not list:
        raise ValueError("procedure_review_wire_citations_invalid")
    out["citation_ids"] = [identity(cid, sources) for cid in citations]
    edits = out.get("edits", [])
    if type(edits) is not list or any(type(edit) is not dict for edit in edits):
        raise ValueError("procedure_review_wire_edits_invalid")
    original_blocks = {b["block_id"]: b["text"] for b in (layout or {}).get("blocks", [])}
    reconstructed, edited_blocks = [], set()
    for edit in edits:
        bid = identity(edit.get("block_id"), blocks)
        if (set(edit) != {"block_id", "old_text", "new_text"} or bid not in original_blocks
                or bid in edited_blocks or type(edit.get("old_text")) is not str
                or not edit["old_text"].strip() or type(edit.get("new_text")) is not str):
            raise ValueError("procedure_review_micro_edit_invalid")
        # Each patch targets the ORIGINAL block body, not another edit's output.
        # Header/separators are never part of a provider-controlled replacement.
        _, body, _ = _block_parts(original_blocks[bid])
        old, replacement = edit["old_text"], edit["new_text"]
        at = body.find(old)
        if at < 0 or body.find(old, at + 1) >= 0:
            raise ValueError("procedure_review_micro_edit_anchor_ambiguous_or_missing")
        reconstructed.append({"block_id": bid,
            "text": body[:at] + replacement + body[at + len(old):]})
        edited_blocks.add(bid)
    if "edits" in out:
        out["edits"] = reconstructed
    audit = out.get("operation_boundary")
    if type(audit) is not dict:
        raise ValueError("procedure_review_wire_boundary_invalid")
    if type(audit) is dict:
        checked, edges = audit.get("checked_blocks"), audit.get("precedence")
        if type(checked) is not list or type(edges) is not list:
            raise ValueError("procedure_review_wire_boundary_invalid")
        audit["checked_blocks"] = [identity(bid, blocks) for bid in checked]
        decoded = []
        for edge in edges:
            if type(edge) is not dict or set(edge) != set(_WIRE_EDGE_FIELDS):
                raise ValueError("procedure_review_wire_edge_invalid")
            canonical = {_WIRE_EDGE_FIELDS[key]: deepcopy(value) for key, value in edge.items()}
            canonical["before_block_id"] = identity(edge["from"], blocks)
            canonical["after_block_id"] = identity(edge["to"], blocks)
            canonical["source_citation_id"] = identity(edge["source"], sources)
            decoded.append(canonical)
        audit["precedence"] = decoded
    return out


def review_schema(legacy: dict, layout: dict, facets: list[str], types: list[str], *,
                  operation_boundary: bool = False, source_ordered: bool = False,
                  operation_source_ids: list[str] | None = None, wire_aliases: dict | None = None,
                  reference_catalogue: dict | None = None, translation_language: str = "",
                  judge_only: bool = False) -> dict:
    if judge_only and (not source_ordered or not operation_boundary
                       or wire_aliases is None or reference_catalogue is None or translation_language):
        raise ValueError("procedure_review_judge_reply_invalid")
    result = deepcopy(legacy)
    result["name"] = legacy["name"]  # backwards-readable verifier envelope
    schema = result["schema"]
    if translation_language:
        if not source_ordered or translation_language not in {"it", "en"}:
            raise ValueError("procedure_review_translation_incomplete")
        schema["properties"]["translation_review"] = {
            "type": "object", "additionalProperties": False,
            "properties": {"language": {"type": "string", "enum": [translation_language]},
                "body_and_safety_complete": {"type": "boolean"}},
            "required": ["language", "body_and_safety_complete"]}
        schema["required"] = list(schema["required"]) + ["translation_review"]
    # Required facets and must-cover retrieval plans may be distinct. Their
    # union must fit in the verdict; never make coverage impossible or drop an
    # obligation merely because the old wire format allowed only twelve keys.
    for key in ("covered_facets", "missing_facets"):
        field = schema["properties"].get(key)
        if field is not None:
            field["maxItems"] = max(int(field.get("maxItems", 0)), len(set(facets)))
    # Coverage keys identify the request's obligations, not prose to translate.
    # Missing facets remain open text so the independent whole-question review
    # can report an omitted category that the router did not plan.
    if facets and "covered_facets" in schema["properties"]:
        schema["properties"]["covered_facets"]["items"] = {
            "type": "string", "enum": list(dict.fromkeys(facets))}
    # A source-authored Step sequence owns its headings. The reviewer may
    # rewrite every body, but cannot replace that sequence with a free-text
    # numbered list. Unsupported or incomplete evidence remains a rejection.
    schema["properties"]["reply_mode"] = {"type": "string", "enum": (
        ["retain", "reject"] if judge_only else
        ["retain", "edit", "reject"] if source_ordered else ["replace", "retain", "edit"])}
    if judge_only:
        schema["properties"]["outcome"] = {"type": "string", "enum": ["pass", "no_sources"]}
        if "outcome" not in schema["required"]:
            schema["required"].append("outcome")
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
        if not source_ordered:
            fields["action_quote"]["description"] = (
                "Exact quote of a temporary state change introduced in the final answer. "
                "A requested restoration of an earlier completed operation is not itself a new "
                "temporary opening. Do not include the later closure in this action quote.")
            fields["closing_answer_quote"]["description"] = (
                "Exact final-answer quote of the documented closure/terminal state. With restored, "
                "this quote must start strictly after the end of action_quote: identical, "
                "overlapping or prefix-containing self-closure is invalid.")
            fields["resolution"]["description"] = (
                "restored: an introduced temporary state is closed by a distinct later action. "
                "documented_terminal_state: the source explicitly supports the final safe state; "
                "the same instruction may state it. unresolved: required closure is unavailable. "
                "Never relabel an unsupported opening as a terminal state to satisfy the protocol.")
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
    if wire_aliases is not None:
        if not (source_ordered and operation_boundary):
            raise ValueError("procedure_review_aliases_require_source_ordered_audit")
        source_names, block_names = list(wire_aliases["sources"]), list(wire_aliases["blocks"])
        if (list(wire_aliases["sources"].values()) != list(dict.fromkeys(operation_source_ids))
                or list(wire_aliases["blocks"].values()) != [b["block_id"] for b in layout["blocks"]]):
            raise ValueError("procedure_review_alias_catalog_drift")
        schema["properties"]["citation_ids"]["items"] = {"type": "string", "enum": source_names}
        schema["properties"]["edits"]["items"] = {"type": "object", "additionalProperties": False,
            "properties": {"block_id": {"type": "string", "enum": block_names},
                "old_text": {"type": "string", "minLength": 1,
                    "description": "One exact unique non-empty span of the original block body, without its heading."},
                "new_text": {"type": "string",
                    "description": "Replacement for only that span. All other original body text is preserved by the server."}},
            "required": ["block_id", "old_text", "new_text"]}
        boundary = schema["properties"]["operation_boundary"]
        boundary["properties"]["checked_blocks"]["items"]["enum"] = block_names
        edge = boundary["properties"]["precedence"]["items"]
        fields = edge["properties"]
        fields["before_block_id"]["enum"] = block_names
        fields["after_block_id"]["enum"] = block_names
        fields["source_citation_id"]["enum"] = source_names
        edge["properties"] = {wire: fields[canonical] for wire, canonical in _WIRE_EDGE_FIELDS.items()}
        edge["required"] = list(_WIRE_EDGE_FIELDS)
    if reference_catalogue is not None:
        if wire_aliases is None or reference_catalogue["answer_sha256"] != layout["answer_sha256"]:
            raise ValueError("procedure_reference_catalog_invalid")
        # Bind an owner and its bounds in the SAME alternative. Group owners
        # with equal unit counts and share definitions: no owner cross-product,
        # raw source strings, or quadratic enumeration of every possible span.
        definitions = {}
        def obj(fields):
            return {"type":"object", "additionalProperties":False,
                    "properties":fields, "required":list(fields)}
        def interval(count):
            if count < 1:
                raise ValueError("procedure_reference_catalog_invalid")
            name = "procedure_range_" + str(count)
            definitions.setdefault(name, obj({key:{"type":"integer", "minimum":0, "maximum":count-1}
                                               for key in ("first", "last")}))
            return {"$ref":"#/$defs/" + name}
        def groups(kind):
            grouped = {}
            for alias, item in reference_catalogue[kind].items():
                grouped.setdefault(len(item["units"]), []).append(alias)
            return grouped.items()
        endpoints, edits, proofs = [], [], []
        for count, owners in groups("blocks"):
            owned = {"type":"string", "enum":owners}
            endpoints.append(obj({"block_id":owned, "kind":{"type":"string", "enum":["original"]},
                                  "range":interval(count)}))
            edits.append(obj({"block_id":owned, "original":interval(count),
                "replacement_parts":{"type":"array", "minItems":1, "maxItems":16,
                                     "items":{"type":"string", "minLength":1}}}))
        if not judge_only:
            endpoints.append(obj({"block_id":{"type":"string", "enum":list(wire_aliases["blocks"])},
                                  "kind":{"type":"string", "enum":["replacement"]}, "range":interval(16)}))
        for count, owners in groups("sources"):
            proofs.append(obj({"source":{"type":"string", "enum":owners}, "range":interval(count)}))
        definitions["procedure_answer_reference"] = {"anyOf":endpoints}
        definitions["procedure_source_reference"] = {"anyOf":proofs}
        if set(definitions).intersection(schema.get("$defs", {})):
            raise ValueError("procedure_reference_catalog_invalid")
        schema.setdefault("$defs", {}).update(definitions)
        schema["properties"]["edits"]["items"] = {"anyOf":edits}
        edge = schema["properties"]["operation_boundary"]["properties"]["precedence"]["items"]
        edge["properties"] = {"action":{"$ref":"#/$defs/procedure_answer_reference"},
                              "next":{"$ref":"#/$defs/procedure_answer_reference"},
                              "proof":{"$ref":"#/$defs/procedure_source_reference"}}
        edge["required"] = list(edge["properties"])
    if judge_only:
        schema["properties"]["edits"]["maxItems"] = 0
    return result


def validate_judge_reply(parsed: dict) -> None:
    """Reject an unsolicited rewrite; never silently retain or drop its proof.

    This is only a restricted producer profile. Existing source, coverage,
    answer binding, note, closure and final-contract checks still follow it.
    """
    if (parsed.get("answer") != "" or parsed.get("edits") != []
            or (parsed.get("reply_mode"), parsed.get("outcome")) not in {
                ("retain", "pass"), ("reject", "no_sources")}):
        raise ValueError("procedure_review_judge_reply_invalid")
    audit = parsed.get("operation_boundary")
    if type(audit) is dict and type(audit.get("precedence")) is list:
        for edge in audit["precedence"]:
            if type(edge) is dict:
                for name in ("action", "next"):
                    ref = edge.get(name)
                    if type(ref) is dict and ref.get("kind") != "original":
                        raise ValueError("procedure_review_judge_reply_invalid")


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
    "safe hold states; do not perform an operation that the user explicitly excluded. "
    "Distinguish restoring a state changed in an already completed phase from introducing a NEW "
    "temporary change now. A requested closure such as replacing a cover after cleaning does not "
    "require another opening/closure cycle. If no new temporary change is introduced, changes=[] "
    "is valid only after full source-based review of every requested restoration and its prerequisites. "
    "Completing an earlier phase does not cancel documented safety invariants. Carry forward the "
    "applicable prerequisites from the whole operation family into every proposed physical restoration "
    "without repeating the earlier operation. For example, when the source requires isolation or "
    "trained personnel, merely saying equipment is not running does not replace those conditions. "
    "State the required conditions before the affected actions; retain a documented safe hold until "
    "the separately requested or authorized next phase. Do not infer missing conditions. "
    "When an entry proves an explicitly documented terminal/restored state in the same instruction, "
    "use documented_terminal_state with exact source support, never a self-paired restored entry. "
    "With restored, closing_answer_quote must start after the END of action_quote in the final answer: "
    "identical, overlapping or prefix-containing quotes cannot prove later restoration. Quote only "
    "the opening/change as action_quote and the distinct later closure as closing_answer_quote. "
    "Do not omit an actually introduced temporary change or relabel an unsupported opening merely "
    "to pass this check. block_id and checked_blocks still refer to the ORIGINAL supplied draft "
    "blocks, including with replace; final-answer quotes must occur in the replacement actually returned."
)


SOURCE_ORDERED_BOUNDARY_INSTRUCTIONS = (
    "\nSOURCE-ORDERED FULL REVIEW / DELTA PROOF: Independently verify the ENTIRE final resolved procedure "
    "against ALL supplied sources, all obligations, every Step, safety condition and dependency, including "
    "unchanged blocks. Source numbering and literal note retention are NOT semantic/safety proof. Return "
    "checked_blocks containing every block alias once. Reject any unsafe, unsupported or incomplete part; "
    "retain/edits=[] never bypasses full review. Use PROCEDURE_SOURCE_ALIASES for citation_ids/edge source, "
    "and PROCEDURE_BLOCKS aliases for edits/checked_blocks/edge endpoints. They encode existing identities. "
    "MICRO-EDITS in this source-ordered wire format: "
    "each edit has block_id, old_text and new_text only. old_text is one exact, non-empty, unique span of "
    "that ORIGINAL block's BODY (no numbered heading); new_text replaces only that span. Choose the shortest "
    "distinctive span covering the change; retain its anchor in new_text when inserting beside it. At most "
    "one patch per block. All patches apply atomically to originals, never sequentially to another patch. "
    "The server preserves everything else, including headings, separators and original notes. Never "
    "re-emit unchanged full bodies/notes or fabricate an anchor. If safe correction cannot fit, reject. "
    "For each introduced/changed operation, include its complete source-documented preparation, position, "
    "mode/energy conditions, locking, intermediate actions and terminal/restoration conditions. Re-establish "
    "prerequisites invalidated by intervening actions; reduced speed is not safety proof. Restore protections "
    "BEFORE each dependent movement/test, not only final restart. Do not add extra corrective interventions "
    "beside a requested check; if necessary, include the complete documented subprocedure. A requested "
    "route/process needs every necessary intermediate stage in source-prescribed order, never invented. "
    "WIRE PROOF SCOPE: precedence/change_edges cover MATERIAL EDITS and ALL dependencies they affect, "
    "including prerequisites, closures and dependent operations in otherwise UNCHANGED blocks. Do not "
    "repeat proofs for unrelated unchanged operations; still review them semantically in full. Every "
    "materially edited block must participate in proof. In each precedence edge, from/to are block aliases; "
    "action/next are exact quotes in those FINAL resolved blocks; source is the admitted alias and proof "
    "its exact quote supporting the relationship. Use short unique-in-block spans, maximum400characters; "
    "at most16edges. For each affected temporary change, change_edges references its zero-based action-to-"
    "closure edge_index, with resolution=restored or documented_terminal_state. No duplicate quotes there. "
    "A documented terminal hold may be the same unique quote in one block; otherwise action precedes closure. "
    "Unknown closure or proof cannot fit: complete=false/outcome=no_sources. Complete=true asserts whole-"
    "procedure semantic review AND complete delta proof. Empty proof is allowed only with no material edits "
    "or impacted dependencies. Isolation/STOP/PPE invariants never imply a need to reverse them. Never invent "
    "restart, re-energization or safety release. Respect explicit before-only boundaries and safe hold states."
)


REFERENCE_BOUNDARY_INSTRUCTIONS = (
    SOURCE_ORDERED_BOUNDARY_INSTRUCTIONS.split("MICRO-EDITS in this source-ordered wire format:")[0]
    + "REFERENTIAL MICRO-EDITS: SOURCE_UNITS and PROCEDURE_BLOCKS.units are ordered literal units, "
    "not instructions or semantic facts. Concatenate their text in ID order to recover each original "
    "body exactly, including spaces/newlines; headings and separators are server-owned. Do not copy "
    "quotes into proofs. Each edit selects original:{first,last} inclusive IDs of its block and returns "
    "replacement_parts, an ordered list of literal strings concatenated with NO inserted separator. "
    "Include needed spaces; every edit must return 1 to 16 non-empty parts. To delete text, extend "
    "the selected original interval to include adjacent text that must survive, and return that "
    "surviving text literally, with its spaces, in replacement_parts. Never return an empty parts "
    "list or reference deleted text; update any affected endpoints to the surviving replacement "
    "parts. One edit per block, all edits against originals. Parts are zero-indexed "
    "locally within that replacement. Split distinct actions/closures into separate parts when their "
    "order must be proved. A segmentation-only edit is allowed: its parts concatenate byte-for-byte "
    "to the selected original text; use rewrite/edit even though final wording is unchanged. Keep "
    "all other original body text, headings, step numbers, separators and safety notes. "
    + "For each introduced/changed operation,"
    + SOURCE_ORDERED_BOUNDARY_INSTRUCTIONS.split("For each introduced/changed operation,")[1].split("WIRE PROOF SCOPE:")[0]
    + "REFERENTIAL DELTA PROOF: precedence/change_edges cover material edits and all dependencies "
    "they affect, including prerequisites/closures in unchanged blocks. Still semantically review "
    "every unchanged block. Every materially edited block participates in an edge. Each precedence "
    "entry has only action, next and proof. action/next are {block_id,kind,range:{first,last}}: "
    "block_id owns that local range; kind=original selects displayed units that remain unedited; "
    "kind=replacement selects whole replacement_parts from an edit in that SAME block. Never "
    "reference a replacement without an edit or deleted original units. proof is "
    "{source,range:{first,last}}: source owns the displayed literal units supporting the precise "
    "relationship, not merely similar words. Every range is inclusive, zero-based and must satisfy "
    "0 <= first <= last < that owner's unit count (or that edit's replacement_parts count). "
    "Original and replacement indices are separate namespaces. References select "
    "positions, not semantic approval. The server verifies actual resolved offsets. With restored, "
    "the opening ends before the closure starts. Identical/overlapping parts cannot prove restoration. "
    "If a single original unit contains both actions, use a segmentation-only edit rather than "
    "claiming an overlapping edge. A source-documented terminal hold may reference the same part. "
    "change_edges references the zero-based precedence edge_index and its resolution. No separate "
    "quote copy there. At most16 edges. Unknown closure, unsafe order or unsupported relationship: "
    "complete=false/outcome=no_sources. Never use a reference merely to make validation pass. "
    "Complete=true still asserts whole-procedure semantic review and complete delta proof. Empty "
    "proof only when no material edits or impacted dependencies. Isolation/STOP/PPE invariants do "
    "not require reversal. Never invent restart/re-energization; preserve before-only boundaries "
    "and safe hold states."
)


def _quote_text(value: str) -> str:
    # Typography/line wrapping only; keep case, negations, numbers and units.
    return " ".join(value.translate(str.maketrans({"\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"'})).split())


def _source_quote_present(quote: str, source: str) -> bool:
    """Match a literal source span, tolerating one enclosing quotation pair.

    A reviewer can delimit a verbatim citation with double quotes that are not
    printed in the source. Only that balanced outer pair is presentation; the
    entire interior must still match this same admitted source. Inner quotes,
    negations, case, numbers and units are never removed or rewritten.
    """
    quoted, original = _quote_text(quote), _quote_text(source)
    if not quoted:
        return False
    if quoted in original:
        return True
    if len(quoted) > 2 and quoted[0] == quoted[-1] == '"':
        interior = quoted[1:-1].strip()
        return bool(interior) and interior in original
    return False


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
                              source_ordered: bool = False,
                              reference_proof: ReferenceProof | None = None) -> dict:
    """Validate a same-call semantic closure audit, never infer physical safety."""
    if reference_proof is not None:
        return _verify_reference_boundary(parsed, layout=layout, source_texts=source_texts,
                                          proof=reference_proof, source_ordered=source_ordered)
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
            if (source is None or not _source_quote_present(change["closing_source_quote"], source)
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
                if (source is None or not _source_quote_present(edge["source_quote"], source)
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
        "reason": "complete" if valid else reason,
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


def _verify_reference_boundary(parsed: dict, *, layout: dict, source_texts: dict[str, str],
                               proof: ReferenceProof, source_ordered: bool) -> dict:
    """Exact same-call identity/order proof over server-resolved positions."""
    out = deepcopy(parsed)
    answer = str(out.get("answer") or "")
    projected_answer = _answer_projection(answer)
    raw_audit = out.get("operation_boundary")
    audit = _canonical_boundary(out.get("operation_boundary"))
    if audit is not None:
        out["operation_boundary"] = audit
    expected = [block["block_id"] for block in layout["blocks"]]
    reason = "operation_reference_boundary_incomplete"
    valid = (source_ordered and type(proof) is ReferenceProof
        and type(audit) is dict and audit.get("complete") is True
        and type(audit.get("checked_blocks")) is list
        and all(type(bid) is str for bid in audit["checked_blocks"])
        and len(audit["checked_blocks"]) == len(expected)
        and set(audit["checked_blocks"]) == set(expected)
        and type(audit.get("changes")) is list and len(audit["changes"]) <= 16
        and type(audit.get("precedence")) is list and len(audit["precedence"]) == len(proof.edges))
    if valid and (proof.original_sha256 != layout["answer_sha256"]
                  or proof.answer_sha256 != digest(projected_answer)):
        valid = False; reason = "operation_reference_answer_binding_changed"
    review_audit = out.get("procedure_review")
    if valid and (type(review_audit) is not dict
                  or review_audit.get("output_answer_sha256") != proof.raw_answer_sha256):
        valid = False; reason = "operation_reference_review_binding_changed"
    if valid and (set(cid for cid,_ in proof.source_hashes) != set(source_texts)
                  or any(digest(source_texts[cid]) != expected_hash for cid,expected_hash in proof.source_hashes)):
        valid = False; reason = "operation_reference_source_binding_changed"
    final_layout = block_layout(projected_answer)
    if valid and [block["block_id"] for block in final_layout["blocks"]] != expected:
        valid = False; reason = "operation_reference_block_layout_changed"
    block_spans, at = {}, 0
    for block in final_layout["blocks"]:
        block_spans[block["block_id"]] = (at, at + len(block["text"]))
        at += len(block["text"])
    touched, edge_sources = set(), []
    if valid:
        terminal_edges = {item["edge_index"] for item in raw_audit.get("change_edges", [])
                          if item["resolution"] == "documented_terminal_state"}
        for edge_index, (edge, bound) in enumerate(zip(audit["precedence"], proof.edges)):
            before_bid, start, end, after_bid, close_start, close_end, cid, quote = bound
            before_span, after_span = block_spans.get(before_bid), block_spans.get(after_bid)
            if (not before_span or not before_span[0] <= start < end <= before_span[1]
                    or projected_answer[start:end] != _answer_projection(str(edge.get("before_answer_quote") or ""))
                    or edge.get("before_block_id") != before_bid):
                valid = False; reason = "operation_reference_action_absent"; break
            if (not after_span or not after_span[0] <= close_start < close_end <= after_span[1]
                    or projected_answer[close_start:close_end] != _answer_projection(str(edge.get("after_answer_quote") or ""))
                    or edge.get("after_block_id") != after_bid):
                valid = False; reason = "operation_reference_closure_absent"; break
            if cid not in source_texts or edge.get("source_citation_id") != cid or edge.get("source_quote") != quote or quote not in source_texts[cid]:
                valid = False; reason = "operation_reference_source_absent"; break
            same_hold = (before_bid == after_bid and start == close_start and end == close_end
                         and edge_index in terminal_edges)
            if not same_hold and end > close_start:
                valid = False; reason = "operation_reference_closure_order_invalid"; break
            touched.update((before_bid,after_bid)); edge_sources.append(cid)
    if valid:
        for change in audit["changes"]:
            if (type(change) is not dict or change.get("resolution") not in {"restored","documented_terminal_state"}
                    or not any(edge.get("before_block_id") == change.get("block_id")
                        and edge.get("before_answer_quote") == change.get("action_quote")
                        and edge.get("after_answer_quote") == change.get("closing_answer_quote")
                        and edge.get("source_citation_id") == change.get("closing_source_citation_id")
                        and edge.get("source_quote") == change.get("closing_source_quote") for edge in audit["precedence"])):
                valid = False; reason = "operation_reference_unresolved_change"; break
    if valid:
        originals = {block["block_id"]:_quote_text(block["text"]) for block in layout["blocks"]}
        changed = {block["block_id"] for block in final_layout["blocks"]
                   if _quote_text(block["text"]) != originals[block["block_id"]]}
        if not changed.issubset(touched):
            valid = False; reason = "operation_reference_edited_dependency_missing"
    out["operation_boundary_validation"] = {
        # Reference units change the private wire, not the canonical safety
        # proof consumed by the final procedure-family contract.
        "version":"ask-operation-boundary-v1", "reference_protocol":"ask-procedure-references-v2",
        "complete":bool(valid),
        "answer_sha256":digest(answer.strip()), "reason":"complete" if valid else reason,
        "checked_blocks":len(expected), "temporary_changes":len(audit["changes"]) if valid else None,
        "precedence_checked":bool(valid), "precedence_edges":len(proof.edges) if valid else None}
    if valid:
        # Only the verified, same-call whitespace projection may rebind the
        # canonical review audit. Content removal/masking already failed above.
        review_audit["output_answer_sha256"] = digest(answer)
        out["citation_ids"] = list(dict.fromkeys(list(out.get("citation_ids") or []) + edge_sources))
    elif out.get("outcome") in {"pass","rewrite"}:
        out.update(outcome="no_sources",reason=reason)
    return out


MICRO_PROTOCOL_INSTRUCTIONS = (
    "\nPROCEDURE REVIEW PROTOCOL: You are the independent semantic and safety verifier. Check every "
    "mandatory facet, source instruction, precaution, numeric value, condition and translation against "
    "SOURCES. PROCEDURE_STRUCTURE proves only source-occurrence numbers and literal note retention, "
    "never relevance, semantic completeness or safety. An obligation need not be echoed verbatim: do "
    "not rewrite merely to repeat a requirement already satisfied. If the complete draft is correct, "
    "use outcome=pass, reply_mode=retain, answer='', edits=[]. For required corrections use "
    "outcome=rewrite, reply_mode=edit and the source-ordered micro-edit format below. The complete "
    "replacement is the result after server patch application, not retransmission of unchanged text. "
    "Source Step numbers, headings and separators are immutable. Preserve each source instruction and "
    "verbatim note/warning in its original Step: never move, omit, paraphrase a note, renumber, exchange "
    "instructions or add numbered Steps inside a body. If this admitted sequence cannot satisfy the "
    "request safely and completely, use reply_mode=reject, outcome=no_sources (partial only if otherwise "
    "allowed), answer='', edits=[] and unresolved keys. Successful coverage must never conceal an "
    "omitted Step. Keep covered/missing contract keys verbatim. All evidence, authority, no-invention "
    "and safety requirements remain mandatory."
)


JUDGE_PROTOCOL_INSTRUCTIONS = (
    "\nCOMPLETE LITERAL DRAFT REVIEW: You are its independent semantic and safety judge, not "
    "its editor. In this profile, any general permission above to rewrite or edit does not apply. "
    "The server has preserved the admitted source occurrences, numbers and notes in the current "
    "draft; that is representation only, never proof of relevance, coverage, accuracy or safety. "
    "Check the complete QUESTION, every mandatory facet and answer type, numeric value and unit, "
    "source relationship, phase scope, prerequisite, closure and every block against SOURCES. "
    "Supported equivalent wording need not repeat a requirement verbatim. If the existing complete "
    "draft is useful, fully supported and safe, return outcome=pass, reply_mode=retain, answer='', "
    "edits=[]. If any actual unsupported claim, missing mandatory instruction, unsafe order, "
    "unavailable prerequisite or required correction remains, return outcome=no_sources, "
    "reply_mode=reject, answer='', edits=[] and the unresolved coverage keys. Never force approval "
    "because the source representation is complete. Do not repair, add prose, reformat, generate "
    "replacement parts, or reference output that you would have written. Source headings, action "
    "order, notes and all source/authority/no-invention requirements remain mandatory."
)


JUDGE_REFERENCE_INSTRUCTIONS = (
    "\nLITERAL DRAFT AUDIT: SOURCES and PROCEDURE_BLOCKS are complete, lossless original units, "
    "not semantic approval. Review every unchanged block and its complete source operation, "
    "including prerequisites and any required safe closure or terminal hold. Keep standing "
    "isolation/STOP/PPE invariants; never invent restart or re-energization. Preserve before-only "
    "boundaries. Set operation_boundary.complete=true only after this whole-procedure semantic "
    "and safety review. checked_blocks lists every supplied block alias exactly once. No edit is "
    "permitted, so no newly generated operation or replacement-part namespace exists. Empty "
    "precedence/change_edges is valid only after independently finding the unchanged draft safe "
    "and no material edit or affected dependency to prove. If you use an explicit source-backed "
    "precedence edge, action and next are {block_id,kind:'original',range:{first,last}}; proof is "
    "{source,range:{first,last}}. All indices are inclusive and zero-based in that supplied owner, "
    "0 <= first <= last < its original unit count. They never refer to replacement text. Select "
    "the exact original relationship, not merely similar words. change_edges uses the zero-based "
    "precedence edge_index and resolution. A restored edge ends the opening before the closure; "
    "a documented_terminal_state needs exact source support, never an invented restoration. "
    "A genuinely missing condition, unsafe order or unavailable closure requires rejection, not "
    "an edit, clamped reference, guessed proof, or approval."
)


TRANSLATION_INSTRUCTIONS = (
    "\nCROSS-LANGUAGE PROCEDURE (mandatory): the canonical draft may contain untranslated source "
    "text. English/Italian headings or labels alone are not a translated answer. In this SAME review, "
    "use edit for every affected block, including the preamble and titles: the primary explanation, "
    "every action, condition and ALL precautions must be complete in RESPONSE_LANGUAGE. Translate "
    "technical meaning faithfully, preserving negation, quantities, identifiers, operating modes, "
    "STOP versus isolation, sequence and closure. Source step NUMBERS and separators remain immutable; "
    "the textual title inside each block may be translated. Keep each original source warning verbatim "
    "in its own Step as an explicitly labelled supporting quotation AFTER its complete translated "
    "precaution. That original quotation is not a substitute for an understandable translated warning. "
    "Prefer editing only the affected prefix/range while leaving existing original-warning units "
    "untouched: place the complete translated precaution and quotation label before those units. "
    "Do not re-emit unchanged source quotations merely to translate adjacent text. "
    "Manual support remains in SOURCES even when not appended to the draft: inspect it for essential "
    "detail, prerequisites, warnings and closure, integrate everything applicable in the main translated "
    "instructions and retain supporting citation IDs. Never treat absent appendices as absent evidence "
    "or silently omit a mandatory detail. Check your final resolved translation against every admitted "
    "source and requirement, using the same operation-boundary proof. Set translation_review.language "
    "to RESPONSE_LANGUAGE and body_and_safety_complete=true ONLY after verifying the entire resolved "
    "body and every precaution in that language. If you cannot provide and verify it within this reply, "
    "set that field false and reject; never approve an untranslated draft. No scope, safety, source-note, "
    "citation, number, or operation-order requirement is waived."
)


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
                  source_ordered: bool = False, source_notes: tuple[dict, ...] = (),
                  translation_language: str = "") -> dict:
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
    if translation_language and complete:
        language_review = out.get("translation_review")
        if (not source_ordered or translation_language not in {"it", "en"}
                or type(language_review) is not dict
                or set(language_review) != {"language", "body_and_safety_complete"}
                or language_review.get("language") != translation_language
                or language_review.get("body_and_safety_complete") is not True):
            raise ValueError("procedure_review_translation_incomplete")
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
    if translation_language:
        out["procedure_review"]["translation_review"] = {
            "language": translation_language,
            "body_and_safety_complete": bool(complete and notes_preserved
                and (out.get("translation_review") or {}).get("body_and_safety_complete") is True)}
    return out
