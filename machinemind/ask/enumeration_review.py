"""Source-first inventory for factual ASK list reviews; no extra provider call.

The inventory is still a semantic judgement, not deterministic entailment proof.
Existing evidence admission, text binding and final list checks remain required.
"""
from copy import deepcopy


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
    "adding an unrequested procedure. Preserve exact identifiers and literal source conditions."
)


def source_first_schema(schema: dict) -> dict:
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
    return out


def reject_empty_inventory(result: dict) -> dict:
    """An empty inventory cannot certify a successful factual enumeration."""
    if (str(result.get("outcome") or "").strip().lower() in {"pass", "rewrite"}
            and not result.get("expected_list_items")):
        result = {**result, "outcome": "no_sources",
                  "reason": "supported_list_inventory_missing"}
    return result
