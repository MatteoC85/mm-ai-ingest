"""Conservative interpretation of the router's explicit procedural scope.

This selects an answer producer, never grants evidence or proves coverage. An
absent, contradictory or legacy scope keeps the existing execution path.
"""
from collections.abc import Mapping
from dataclasses import replace


QUESTION_CONDITIONS_REQUIREMENT = (
    "All requested checks and conditions in the complete QUESTION, including every "
    "explicit detail and all documented precautions applicable to its operation and phase"
)


def conditions_only_contract(*, procedure_scope, information_task,
                             required_answer_types, facet_queries) -> bool:
    if procedure_scope != "conditions_only" or information_task != "procedure_segment":
        return False
    requirements = set(required_answer_types or ())
    checks = {"checklist", "safety_conditions"}
    if not requirements.intersection(checks) or not requirements.issubset(checks | {"explanation"}):
        return False
    mandatory = []
    for facet in facet_queries or ():
        field = facet.get if isinstance(facet, Mapping) else lambda key, default=None: getattr(facet, key, default)
        if field("must_cover") is True:
            mandatory.append(field("answer_type"))
    return bool(mandatory) and all(kind in checks for kind in mandatory)


def bounded_phase_contract(*, procedure_scope, information_task,
                           required_answer_types, facet_queries,
                           procedure_phase=(), question="") -> bool:
    """A source-backed phase answer, never lexical guesses about temporal intent.

    The same router supplies exact QUESTION excerpts for completed context,
    requested actions and later context. They identify scope, not source facts.
    Inconsistent/legacy metadata keeps the original execution producer.
    """
    if procedure_scope != "bounded_phase" or information_task != "procedure_segment":
        return False
    if type(procedure_phase) not in {tuple, list} or len(procedure_phase) != 3:
        return False
    before, actions, after = procedure_phase
    values = [value for value in procedure_phase if value]
    if (not actions or not (before or after)
            or any(type(value) is not str or len(value) > 240 or not value.strip() for value in values)
            or any(type(value) is not str for value in procedure_phase)):
        return False
    question_text = str(question or "")
    if (len(set(values)) != len(values)
            or any(value not in question_text for value in values)):
        return False
    requirements = set(required_answer_types or ())
    allowed = {"ordered_actions", "state_sequence", "safety_conditions", "checklist", "explanation"}
    if "ordered_actions" not in requirements or not requirements.issubset(allowed):
        return False
    mandatory = []
    for facet in facet_queries or ():
        field = facet.get if isinstance(facet, Mapping) else lambda key, default=None: getattr(facet, key, default)
        if field("must_cover") is True:
            mandatory.append(field("answer_type"))
    return bool(mandatory) and "ordered_actions" in mandatory and all(kind in allowed for kind in mandatory)


def family_context_contract(*, procedure_scope, information_task,
                            required_answer_types, facet_queries,
                            procedure_phase=(), question="") -> bool:
    return conditions_only_contract(procedure_scope=procedure_scope,
        information_task=information_task, required_answer_types=required_answer_types,
        facet_queries=facet_queries) or bounded_phase_contract(
        procedure_scope=procedure_scope, information_task=information_task,
        required_answer_types=required_answer_types, facet_queries=facet_queries,
        procedure_phase=procedure_phase, question=question)


def bind_conditions_requirements_to_question(decision):
    """Separate router search hypotheses from the user's answer obligation.

    QUESTION remains the original request, not a copied/truncated facet value.
    Reuse one existing facet slot and every query array: no additional retrieval
    queries, source authority or semantic coverage is created by this projection.
    """
    if decision.effective_mode != "ask" or decision.request_kind in {"ambiguous", "unsafe_request", "out_of_scope"}:
        return decision
    if not conditions_only_contract(procedure_scope=decision.procedure_scope,
            information_task=decision.information_task,
            required_answer_types=decision.required_answer_types,
            facet_queries=decision.facet_queries):
        return decision
    anchor = next(i for i, facet in enumerate(decision.facet_queries) if facet.must_cover)
    facets = tuple(replace(facet,
        facet=QUESTION_CONDITIONS_REQUIREMENT if i == anchor else facet.facet,
        must_cover=i == anchor) for i, facet in enumerate(decision.facet_queries))
    return replace(decision, required_facets=(QUESTION_CONDITIONS_REQUIREMENT,), facet_queries=facets)
