"""Source-first inventory for factual ASK list reviews; no extra provider call.

The inventory is still a semantic judgement, not deterministic entailment proof.
Existing evidence admission, text binding and final list checks remain required.
"""
from copy import deepcopy
from . import table_units as _tables


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
    "state a concrete supported obligation, preserving its component/assembly, action, "
    "interval or value, and applicability condition; a component name alone is insufficient "
    "when the question also requests an action or interval. Distinguish inspection from "
    "intervention and unconditional schedules from conditional actions. Keep separately "
    "qualified row and note relationships as separate items. Write these items in "
    "RESPONSE_LANGUAGE, then use the same concrete wording in the replacement where needed. "
    "Do not invent an item to fill an empty list: if no requested item is supported, return "
    "no_sources. Every pass or rewrite needs a nonempty supported inventory. Verify all "
    "inventory items against the final answer, including required access conditions for any "
    "physical instruction retained. Remove unnecessary physical instructions rather than "
    "adding an unrequested procedure. Preserve exact identifiers and literal source conditions. "
    "A request joining categories asks for their union: a record relevant to one requested "
    "category must not be excluded merely because it is unrelated to another. In a table, "
    "a material/product column does not itself prescribe an intervention. Use the explicit "
    "action/detail cell with its interval; preserve separately qualified actions in notes. "
    "Do not turn a scheduled inspection into scheduled intervention by combining its interval "
    "with a material column. covered_list_items must be []: the server computes coverage "
    "from expected_list_items and the final answer; do not repeat the inventory there."
)


RECORD_INSTRUCTIONS = (
    " TABLE_RECORD markers identify unchanged literal PDF record occurrences within "
    "their original source context, not new evidence or assembly equivalence. "
    "Account for every ID exactly once in table_record_coverage: inventory_items are "
    "zero-based expected_list_items indices covering all its requested actions, intervals "
    "and conditions; otherwise state outside_request_reason for the complete QUESTION, "
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


def source_first_schema(schema: dict, *, record_ids: tuple[str, ...] = ()) -> dict:
    out = deepcopy(schema)
    properties = out["schema"]["properties"]
    inventory = properties["expected_list_items"]
    inventory["description"] = (
        "Complete supported inventory derived from QUESTION and SOURCES independently of "
        "CURRENT_ANSWER, in RESPONSE_LANGUAGE. Each item preserves the requested relationship "
        "between component, action, interval/value and condition. Include all relevant rows "
        "and continuation sections; do not use broad category names as concrete items.")
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
            'items': {'type': 'object', 'additionalProperties': False,
                'properties': {
                    'record_id': {'type': 'string', 'enum': list(record_ids)},
                    'inventory_items': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 39}, 'maxItems': 40},
                    'outside_request_reason': {'type': 'string'}},
                'required': ['record_id', 'inventory_items', 'outside_request_reason']}}
        out['schema']['required'] = [*out['schema']['required'], 'table_record_coverage']
    return out


def validate_table_coverage(result: dict, record_ids: tuple[str, ...], *, wire_inventory=None) -> dict:
    """Require complete bounded bookkeeping, not semantic entailment by ID."""
    if not record_ids or str(result.get('outcome') or '').lower() not in {'pass', 'rewrite'}:
        return result
    expected = result.get('expected_list_items')
    rows = result.get('table_record_coverage')
    valid = (type(expected) is list and type(wire_inventory) is list
        and expected == wire_inventory and type(rows) is list and len(rows) == len(record_ids))
    seen = set()
    if valid:
        for row in rows:
            if type(row) is not dict:
                valid = False; break
            identity, indices, reason = row.get('record_id'), row.get('inventory_items'), row.get('outside_request_reason')
            if (type(identity) is not str or identity not in record_ids or identity in seen
                    or type(indices) is not list or len(indices) > 40
                    or any(type(i) is not int or not 0 <= i < len(expected) for i in indices)
                    or len(set(indices)) != len(indices) or type(reason) is not str
                    or bool(indices) == bool(reason.strip())):
                valid = False; break
            seen.add(identity)
    if not valid or seen != set(record_ids):
        return {**result, 'outcome': 'no_sources', 'reason': 'source_table_inventory_incomplete',
            '_terminal_review_failure': True}
    return result


def reject_empty_inventory(result: dict) -> dict:
    """An empty inventory cannot certify a successful factual enumeration."""
    if (str(result.get("outcome") or "").strip().lower() in {"pass", "rewrite"}
            and not result.get("expected_list_items")):
        result = {**result, "outcome": "no_sources",
                  "reason": "supported_list_inventory_missing"}
    return result
