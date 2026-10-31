"""Source-first inventory for factual ASK list reviews; no extra provider call.

The inventory is still a semantic judgement, not deterministic entailment proof.
Existing evidence admission, text binding and final list checks remain required.
"""
from copy import deepcopy
import re
from . import table_units as _tables

INVENTORY_MARKER = '[[FACTUAL_INVENTORY]]'
INVENTORY_PREFIX = '[[FACTUAL_INVENTORY'
ANSWER_INSTRUCTIONS = (
    " For outcome=rewrite, avoid copying the inventory twice: put [[FACTUAL_INVENTORY]] "
    "exactly once on its own line in answer. The server replaces that line with one bullet "
    "per expected_list_items string, verbatim and in order. Put headings, limitations or "
    "necessary context outside that line. Each inventory string must therefore be a complete "
    "reader-facing obligation in RESPONSE_LANGUAGE with its supported conditions. Review "
    "the resulting complete answer and cite all its sources. Never put the marker in an "
    "inventory item or another field. For pass/no_sources do not use the marker."
)


def applies(request, decision, *, protected: bool, enumeration: bool) -> bool:
    return bool(protected and enumeration
        and getattr(request, "requested_mode", "") == "ask"
        and getattr(decision, "effective_mode", "") == "ask"
        and getattr(decision, "request_kind", "") == "factual"
        and getattr(decision, "evidence_state", "") in {"supported", "partial"}
        and (getattr(decision, "information_task", "") in {
            "document_explanation", "comparison", "numeric_specification"}
            or (getattr(decision, "information_task", "") == "other"
                and "checklist" in (getattr(decision, "required_answer_types", ()) or ()))))


INSTRUCTIONS = (
    " FACTUAL_LIST_REVIEW: build expected_list_items from the complete QUESTION and SOURCES "
    "before judging CURRENT_ANSWER. Inspect every relevant table row and continuation section "
    "for each independently requested category. The draft is not the inventory. Each item must "
    "state a concrete supported fact or explicitly requested intervention, preserving its component/assembly, action, "
    "interval or value, and applicability condition; a component name alone is insufficient "
    "when the question also requests an action or interval. Distinguish inspection from "
    "intervention and unconditional schedules from conditional actions. Keep separately "
    "qualified row and note relationships as separate items. Write these items in "
    "RESPONSE_LANGUAGE, then use the same concrete wording in the replacement where needed. "
    "Do not invent an item to fill an empty list: if no requested item is supported, return "
    "no_sources. Every pass or rewrite needs a nonempty supported inventory. Verify all "
    "inventory items against the final answer. For informational requests, preserve the requested "
    "facts and qualifications but omit unrequested access, setup, state-change or restart instructions "
    "from editable prose. If the user requests an intervention, retain every applicable source "
    "prerequisite and operating condition; generic safety advice cannot replace a specific condition. "
    "Preserve exact identifiers and literal source conditions, including in documentary table quotations. "
    "A request joining categories asks for their union: a record relevant to one requested "
    "category must not be excluded merely because it is unrelated to another. In a table, "
    "a material/product column does not itself prescribe an intervention. Use the explicit "
    "action/detail cell with its interval; preserve separately qualified actions in notes. "
    "Do not turn a scheduled inspection into scheduled intervention by combining its interval "
    "with a material column. covered_list_items must be []: the server computes coverage "
    "from expected_list_items and the final answer; do not repeat the inventory there."
)

SOURCE_FIRST_INSTRUCTIONS = INSTRUCTIONS.replace(
    'Inspect every relevant table row and continuation section for each independently requested category.',
    'Inspect source rows and continuations for every requested category; select sufficient reliable '
    'detail for a useful answer, not every available manual fact. State material limits; never '
    'present an incomplete list as exhaustive.') + (
    ' Explicitly requested exhaustiveness remains a requirement or a clearly stated limitation. '
    'Every published fact or action must retain all qualifications and prerequisites affecting '
    'its meaning or safety. A coverage reason may truthfully identify a relevant record not included '
    'in the explicitly limited answer; do not falsely label it unrelated.')


def useful_verifier_instructions(text: str) -> str:
    """Scope useful, explicitly limited inventories without conflicting completeness rules."""
    replacements = (
        ("Verify coverage of every independently requested category and its relevant source rows, including continuation sections and items absent from the facet plan.",
         "Verify useful coverage of every independently requested category, considering its source rows, continuation sections and items absent from the facet plan."),
        ("If the answer omits a requested category or supported relevant rows, rewrite it to include them; never approve it solely because every planned facet is covered.",
         "If the answer omits a requested category or a material qualification of a published fact or action, rewrite it; never approve it solely because every planned facet is covered. A useful inventory may be explicitly limited without repeating every available row."),
        ("If ENUMERATION_REQUESTED=true, identify the complete set of directly supported labels, types, modes, options, actions or settings that answer the requested category.",
         "If ENUMERATION_REQUESTED=true, identify sufficient directly supported labels, types, modes, options, actions or settings to answer every requested category usefully. State any material limits; explicitly requested exhaustiveness remains required or must be clearly qualified as unavailable."),
        ("Preserve all supported values, units, labels, modes, control types and list items.",
         "Preserve the exact values, units, labels, modes, control types and material qualifications of every retained fact or action."),
        ("When MACHINE_CATALOG is present, inspect every catalog row before deriving expected_list_items; low ranking or different wording is not permission to omit an explicitly documented machine assembly or auxiliary system.",
         "When MACHINE_CATALOG is present, inspect it for requested categories and relevant applicability conditions. Low ranking or different wording does not make a source irrelevant; an explicitly limited useful answer need not quote every assembly or auxiliary system."),
    )
    for before, after in replacements:
        text = text.replace(before, after)
    return text


RECORD_INSTRUCTIONS = (
    " TABLE_RECORD markers identify unchanged literal PDF record occurrences within "
    "their original source context, not new evidence or assembly equivalence. "
    "Account for every ID exactly once in table_record_coverage using compact strings: "
    "T1=0,2 means zero-based expected_list_items indices covering all its requested "
    "actions, intervals and conditions; T2!reason excludes it with a brief reason for the complete QUESTION, "
    "not only its first category. IDs and coverage declarations do not prove entailment: "
    "check the full cells, detail rows and notes, including continuation records. "
    "Keep these IDs out of the visible answer."
)


def annotate_table_records(sources: str) -> tuple[str, tuple[str, ...]]:
    """Annotate validated occurrences; record lines stay exactly deduplicable.

    The input is the already admitted source block. IDs identify literal record
    occurrences only; they neither replace citation identities nor certify meaning.
    Invalid/unsupported layouts remain untouched, never assigned fake cells.
    """
    if (type(sources) is not str or len(sources) > 30000
            or '[TABLE_RECORD ' in sources):
        return sources, ()
    rows, identities = [], []
    for raw in sources.splitlines(keepends=True):
        line = raw.rstrip('\r\n')
        match = _tables._LINE.fullmatch(line)
        if match and match[1] == 'RECORD':
            first, last, columns = int(match[2]), int(match[3] or match[2]), int(match[4])
            try:
                cells = (_tables._cells(match[5], 0, first, last, columns)
                    if 1 <= first <= last <= 100000 and 1 <= columns <= 256 else None)
            except (ValueError, TypeError):
                cells = None
            if cells:
                if len(identities) >= 64:
                    return sources, ()
                identity = 'T' + str(len(identities) + 1)
                identities.append(identity)
                raw = '[TABLE_RECORD ' + identity + ']\n' + raw
        rows.append(raw)
    return ''.join(rows), tuple(identities)


def source_first_schema(schema: dict, *, record_ids: tuple[str, ...] = (), useful_response: bool = False) -> dict:
    out = deepcopy(schema)
    properties = out["schema"]["properties"]
    inventory = properties["expected_list_items"]
    inventory["description"] = (
        "Complete supported inventory derived from QUESTION and SOURCES independently of "
        "CURRENT_ANSWER, in RESPONSE_LANGUAGE. Each item preserves the requested relationship "
        "between component, action, interval/value and condition. Include all relevant rows "
        "and continuation sections; do not use broad category names as concrete items.")
    if useful_response is True:
        inventory['description'] = (
            'Useful supported inventory addressing every requested category in RESPONSE_LANGUAGE, '
            'with truthful limits and no unsupported claim of exhaustiveness. Preserve every '
            'published component/action/value relationship and its applicable qualifications and '
            'prerequisites; broad category names alone are not concrete facts.')
    out["schema"]["properties"] = {
        "expected_list_items": inventory,
        **{key: value for key, value in properties.items() if key != "expected_list_items"},
    }
    properties = out["schema"]["properties"]
    properties['covered_list_items'] = {
        **properties['covered_list_items'], 'maxItems': 0,
        'description': 'Return []; the server computes coverage from the inventory and answer.'}
    if record_ids:
        properties['table_record_coverage'] = {
            'type': 'array', 'minItems': len(record_ids), 'maxItems': len(record_ids),
            'items': {'type': 'string', 'maxLength': 512},
            'description': 'One string per TABLE_RECORD occurrence: T1=0,2 for inventory indices or T2!brief outside-request reason. No missing or duplicate IDs.'}
        out['schema']['required'] = [*out['schema']['required'], 'table_record_coverage']
    return out


def inventory_positions_match(wire, canonical, *, normalize_item=None) -> bool:
    """Prove the existing normalization preserves every wire index exactly.

    The visible wire strings are not rewritten. Collisions, dropped items and
    reordered positions cannot be justified by a matching set of strings.
    """
    if (type(wire) is not list or type(canonical) is not list
            or not 0 < len(wire) == len(canonical) <= 40
            or any(type(item) is not str or not item.strip() for item in wire + canonical)
            or len({item.lower() for item in canonical}) != len(canonical)):
        return False
    return all((normalize_item(item) if normalize_item is not None else [item]) == [canonical[index]]
               for index, item in enumerate(wire))


def validate_table_coverage(result: dict, record_ids: tuple[str, ...], *, wire_inventory=None, normalize_item=None) -> dict:
    """Require complete bounded bookkeeping, not semantic entailment by ID."""
    if not record_ids or str(result.get('outcome') or '').lower() not in {'pass', 'rewrite'}:
        return result
    expected = result.get('expected_list_items')
    rows = result.get('table_record_coverage')
    valid = (inventory_positions_match(wire_inventory, expected, normalize_item=normalize_item)
        and type(rows) is list and len(rows) == len(record_ids))
    seen = set()
    decoded = []
    if valid:
        for row in rows:
            match = (re.fullmatch(r'(T[1-9][0-9]*)(?:=([0-9]+(?:,[0-9]+)*)|!([^\r\n]+))', row)
                if type(row) is str and len(row) <= 512 else None)
            if not match:
                valid = False; break
            identity = match[1]
            tokens = match[2].split(',') if match[2] is not None else []
            # Canonical integers only; no coercion of booleans, signs or aliases.
            if any(len(token) > 2 or str(int(token)) != token for token in tokens):
                valid = False; break
            indices, reason = [int(token) for token in tokens], match[3] or ''
            if (type(identity) is not str or identity not in record_ids or identity in seen
                    or type(indices) is not list or len(indices) > 40
                    or any(type(i) is not int or not 0 <= i < len(expected) for i in indices)
                    or len(set(indices)) != len(indices) or type(reason) is not str
                    or bool(indices) == bool(reason.strip())):
                valid = False; break
            seen.add(identity)
            decoded.append({'record_id':identity,'inventory_items':indices,'outside_request_reason':reason})
    if not valid or seen != set(record_ids):
        return {**result, 'outcome': 'no_sources', 'reason': 'source_table_inventory_incomplete',
            '_terminal_review_failure': True}
    return {**result, 'table_record_coverage':decoded}


def resolve_inventory_answer(result: dict, *, canonical_inventory: list[str], allowed: bool, normalize_item=None) -> dict:
    """Expand an explicit rewrite template before any binding/grounding checks.

    Only model-returned inventory strings supply content. The original request,
    sources and draft cannot activate this representation or contribute text.
    Full-answer legacy replies are unchanged; this is never a semantic PASS.
    """
    body = result.get('answer')
    if type(body) is not str or INVENTORY_PREFIX not in body:
        return result
    items = result.get('expected_list_items')
    lines = body.splitlines(keepends=True)
    marker_lines = [i for i,line in enumerate(lines) if line.rstrip('\r\n') == INVENTORY_MARKER]
    valid = (allowed and result.get('outcome') == 'rewrite' and body.count(INVENTORY_PREFIX) == 1
        and len(marker_lines) == 1 and type(items) is list and bool(items)
        and inventory_positions_match(items, canonical_inventory, normalize_item=normalize_item)
        and all(type(item) is str and item.strip() and INVENTORY_PREFIX not in item for item in items))
    if not valid:
        return {**result, 'outcome':'no_sources','answer':'',
            'reason':'factual_inventory_template_invalid','_terminal_review_failure':True}
    index = marker_lines[0]
    ending = '\r\n' if lines[index].endswith('\r\n') else '\n' if lines[index].endswith('\n') else ''
    lines[index] = '\n'.join('- '+item for item in items) + ending
    return {**result, 'answer':''.join(lines)}


def reject_empty_inventory(result: dict) -> dict:
    """An empty inventory cannot certify a successful factual enumeration."""
    if (str(result.get("outcome") or "").strip().lower() in {"pass", "rewrite"}
            and not result.get("expected_list_items")):
        result = {**result, "outcome": "no_sources",
                  "reason": "supported_list_inventory_missing"}
    return result
