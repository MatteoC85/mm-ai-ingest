"""Conservative interpretation of the router's explicit procedural scope.

This selects an answer producer, never grants evidence or proves coverage. An
absent, contradictory or legacy scope keeps the existing execution path.
"""
from collections.abc import Mapping


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
