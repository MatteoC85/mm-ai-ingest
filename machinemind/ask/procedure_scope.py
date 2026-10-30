"""Semantic procedure intent in the existing ASK router; no text heuristics."""
from copy import deepcopy
from collections.abc import Mapping


SCOPES = ("conditions_only", "bounded_phase", "execution_steps", "other")
PHASE_FIELDS = ("completed_context", "requested_actions", "later_context")
INSTRUCTION = (
    "For ASK, set procedure_scope from what USER_REQUEST asks the answer to do, not from operations mentioned in the sources. "
    "First extract procedure_phase independently of the scope label for every request to execute a procedure segment. "
    "Identify what the user says is already completed, which actions are requested now, and which later operation is not yet requested. "
    "Return exact, unmodified USER_REQUEST excerpts in completed_context, requested_actions and later_context, even if your preliminary scope label is execution_steps. "
    "Do not erase a stated boundary merely because the broad request is procedural or because the sources describe the whole operation. "
    "Only after this extraction choose the scope. A valid action excerpt and at least one temporal boundary select bounded_phase rather than generic execution_steps. "
    "Use conditions_only when the user asks only which prerequisites, checks or safety conditions must hold before an operation. "
    "For this conditions_only procedural request use procedure_segment with checklist and/or safety_conditions; do not add ordered_actions merely because an operation is named. "
    "Each mandatory facet must describe the requested checks or conditions, not an unrequested execution sequence. "
    "For conditions_only, derive mandatory facets from USER_REQUEST and its named operation, not from details of other operations found in INDEXED_EVIDENCE. "
    "Do not turn a neighboring procedure's controls, values, resets or readiness states into mandatory prerequisites for the requested phase. "
    "Preserve specifics actually requested by the user; otherwise describe the required dimensions and operation, and use source-specific search terms only when their applicability to that operation and phase is established. "
    "Use bounded_phase with procedure_segment and ordered_actions when the user asks for actions within an explicitly bounded phase: an operation already completed and/or a later operation that must not yet be executed. "
    "Distinguish that completed context, the actions actually requested, and the later context. Do not repeat the completed operation or start the later one. "
    "Each nonempty excerpt must include the relevant action or state, not only a temporal connective; use at most 240 characters per excerpt. "
    "requested_actions must be nonempty and at least one boundary excerpt must be nonempty; leave absent boundaries empty and never invent them from INDEXED_EVIDENCE. "
    "The excerpts scope the answer; they are not proof that equipment is safe. Preserve all source-backed precautions and closures needed for actions actually proposed. "
    "Do not infer that an operation is completed from a requested physical Step range, material path or full procedure: those ask for execution, not for what follows it. "
    "If no completed or excluded later phase is expressed, leave both boundary strings empty. For conditions_only, full procedures and non-execution requests leave all three strings empty. "
    "Use execution_steps for execution requests without this phase boundary, including a requested Step range, a material path, a complete operation, or a combined request for execution and checks. "
    "Preserve ordered_actions/state_sequence and all operational facets for those requests, even when safety or prerequisite checks are also required. "
    "Use other for non-procedural information or when no such intent is established. "
    "This distinction is semantic in the user's language: the word before, a safety reference, or checklist formatting alone does not establish conditions_only. "
)


def router_schema(schema):
    out = deepcopy(schema)
    obj = out["schema"]
    phase_schema = {"type": "object", "additionalProperties": False,
        "properties": {name: {"type": "string"} for name in PHASE_FIELDS},
        "required": list(PHASE_FIELDS)}
    # Put the user excerpts before classification in the same structured reply.
    # Shape/limits are unchanged; no extra router or provider request is added.
    obj["properties"] = {"procedure_phase": phase_schema,
        **{key: value for key, value in obj["properties"].items()
           if key not in {"procedure_phase", "procedure_scope"}},
        "procedure_scope": {"type": "string", "enum": list(SCOPES)}}
    obj["required"] = ["procedure_phase", *[key for key in obj["required"]
        if key not in {"procedure_phase", "procedure_scope"}], "procedure_scope"]
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
