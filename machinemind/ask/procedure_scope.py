"""Semantic procedure intent in the existing ASK router; no text heuristics."""
from copy import deepcopy


SCOPES = ("conditions_only", "execution_steps", "other")
INSTRUCTION = (
    "For ASK, set procedure_scope from what USER_REQUEST asks the answer to do, not from operations mentioned in the sources. "
    "Use conditions_only when the user asks only which prerequisites, checks or safety conditions must hold before an operation. "
    "For this conditions_only procedural request use procedure_segment with checklist and/or safety_conditions; do not add ordered_actions merely because an operation is named. "
    "Each mandatory facet must describe the requested checks or conditions, not an unrequested execution sequence. "
    "For conditions_only, derive mandatory facets from USER_REQUEST and its named operation, not from details of other operations found in INDEXED_EVIDENCE. "
    "Do not turn a neighboring procedure's controls, values, resets or readiness states into mandatory prerequisites for the requested phase. "
    "Preserve specifics actually requested by the user; otherwise describe the required dimensions and operation, and use source-specific search terms only when their applicability to that operation and phase is established. "
    "Use execution_steps when the user asks how to carry out all or part of an operation, including a requested Step range, a material path, restoration actions, or a combined request for execution and checks. "
    "Preserve ordered_actions/state_sequence and all operational facets for those requests, even when safety or prerequisite checks are also required. "
    "Use other for non-procedural information or when no such intent is established. "
    "This distinction is semantic in the user's language: the word before, a safety reference, or checklist formatting alone does not establish conditions_only. "
)


def router_schema(schema):
    out = deepcopy(schema)
    obj = out["schema"]
    obj["properties"]["procedure_scope"] = {"type": "string", "enum": list(SCOPES)}
    if "procedure_scope" not in obj["required"]:
        obj["required"].append("procedure_scope")
    return out
