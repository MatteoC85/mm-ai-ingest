"""Semantic procedure intent in the existing ASK router; no text heuristics."""
from copy import deepcopy
from collections.abc import Mapping


SCOPES = ("conditions_only", "bounded_phase", "execution_steps", "other")
PHASE_FIELDS = ("completed_context", "requested_actions", "later_context")
INSTRUCTION = (
    "For ASK, set procedure_scope from what USER_REQUEST asks the answer to do, not from operations mentioned in the sources. "
    "Use conditions_only when the user asks only which prerequisites, checks or safety conditions must hold before an operation. "
    "For this conditions_only procedural request use procedure_segment with checklist and/or safety_conditions; do not add ordered_actions merely because an operation is named. "
    "Each mandatory facet must describe the requested checks or conditions, not an unrequested execution sequence. "
    "For conditions_only, derive mandatory facets from USER_REQUEST and its named operation, not from details of other operations found in INDEXED_EVIDENCE. "
    "Do not turn a neighboring procedure's controls, values, resets or readiness states into mandatory prerequisites for the requested phase. "
    "Preserve specifics actually requested by the user; otherwise describe the required dimensions and operation, and use source-specific search terms only when their applicability to that operation and phase is established. "
    "Use bounded_phase with procedure_segment and ordered_actions when the user asks for actions within an explicitly bounded phase: an operation already completed and/or a later operation that must not yet be executed. "
    "Distinguish that completed context, the actions actually requested, and the later context. Do not repeat the completed operation or start the later one. "
    "For bounded_phase, procedure_phase must contain exact, unmodified excerpts from USER_REQUEST: completed_context, requested_actions, later_context. "
    "Each nonempty excerpt must include the relevant action or state, not only a temporal connective; use at most 240 characters per excerpt. "
    "requested_actions must be nonempty and at least one boundary excerpt must be nonempty; leave absent boundaries empty and never invent them from INDEXED_EVIDENCE. "
    "The excerpts scope the answer; they are not proof that equipment is safe. Preserve all source-backed precautions and closures needed for actions actually proposed. "
    "For all scopes other than bounded_phase, set all three procedure_phase strings to empty. "
    "Use execution_steps for execution requests without this phase boundary, including a requested Step range, a material path, a complete operation, or a combined request for execution and checks. "
    "Preserve ordered_actions/state_sequence and all operational facets for those requests, even when safety or prerequisite checks are also required. "
    "Use other for non-procedural information or when no such intent is established. "
    "This distinction is semantic in the user's language: the word before, a safety reference, or checklist formatting alone does not establish conditions_only. "
)


def router_schema(schema):
    out = deepcopy(schema)
    obj = out["schema"]
    obj["properties"]["procedure_scope"] = {"type": "string", "enum": list(SCOPES)}
    obj["properties"]["procedure_phase"] = {"type": "object", "additionalProperties": False,
        "properties": {name: {"type": "string"} for name in PHASE_FIELDS},
        "required": list(PHASE_FIELDS)}
    if "procedure_scope" not in obj["required"]:
        obj["required"].append("procedure_scope")
    if "procedure_phase" not in obj["required"]:
        obj["required"].append("procedure_phase")
    return out


def validated_phase(raw, question):
    """Retain literal user excerpts only; this does not validate their semantics."""
    if not isinstance(raw, Mapping) or set(raw) != set(PHASE_FIELDS) or type(question) is not str:
        return ()
    values = tuple(raw[name] for name in PHASE_FIELDS)
    if any(type(value) is not str or len(value) > 240 for value in values):
        return ()
    completed, actions, later = values
    present = [value for value in values if value]
    if (not actions or not (completed or later)
            or any(not value.strip() or value not in question for value in present)
            or len(set(present)) != len(present)):
        return ()
    return values
