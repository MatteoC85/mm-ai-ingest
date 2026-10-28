"""Semantic ASK readiness within the existing router call; no lexical case rules."""
from copy import deepcopy

INSTRUCTION = (
    "For ASK, classify request_readiness before selecting evidence. Use answerable when the requested information or concrete symptom is identifiable, even if no indexed source supplies the answer. "
    "Use underspecified when the request only reports a generic failure or asks what to do without a concrete observable symptom or operation; do not invent a generic troubleshooting task to make it answerable. "
    "Use contradictory when the same relevant condition is asserted both true and false for the same time/context and choosing one would change the answer. Different times or physical state versus sensor indication are not contradictions. "
    "Use uninterpretable for text with no identifiable technical question. Hostile wording alone is neither unsafe nor out of scope: ask a calm, concise clarification if its technical request is underspecified. "
    "For these three clarification states set evidence_state=clarify and write one short clarification_question in RESPONSE_LANGUAGE that resolves the decisive missing observation or conflict. "
    "Ask only for already available information, not a new physical operation or test, and do not ask for the machine identity already supplied by SCOPE. "
    "Do not treat unavailable measurements as evidence of either outcome. Do not clarify merely because sources are absent, an exact undocumented value was requested, or some optional detail is unknown. "
    "Safety refusal and out-of-scope classification take precedence over readiness; keep those request kinds. "
)
STATES = ("answerable", "underspecified", "contradictory", "uninterpretable")


def router_schema(schema):
    out = deepcopy(schema)
    obj = out["schema"]
    obj["properties"]["request_readiness"] = {"type": "string", "enum": list(STATES)}
    obj["required"].append("request_readiness")
    return out


def apply_readiness(payload, *, language="it"):
    out = dict(payload)
    if out.get("request_kind") in {"unsafe_request", "out_of_scope"}:
        return out
    state = out.get("request_readiness")
    # Legacy responses/fallbacks do not claim a semantic classification they lack.
    if state not in STATES[1:]:
        return out
    out["evidence_state"] = "clarify"
    if state == "uninterpretable":
        out["request_kind"] = "ambiguous"
    if not str(out.get("clarification_question") or "").strip():
        en = str(language).lower().startswith("en")
        out["clarification_question"] = (
            ("Which of the conflicting conditions did you observe, and when?" if en else "Quale delle condizioni in contrasto hai osservato, e in quale momento?")
            if state == "contradictory" else
            ("What exactly happens on the machine, and what operation are you trying to perform?" if en else "Che cosa succede esattamente sulla macchina e quale operazione stai cercando di eseguire?")
        )
    # Clarification is terminal for this request: do not retrieve invented facets.
    for key in ("required_facets", "facet_queries", "dense_queries", "lexical_queries", "exact_terms", "relevant_evidence_ids"):
        out[key] = []
    return out
