"""Smart source selection, before causal-only filtering or display deduplication.

Retrieval annotations locate candidates; they are never source text. A guided
diagnosis needs both mechanisms and observation instructions. In particular an
HMI legend or an individual inspection Step need not itself assert a cause.
This selector does not certify a hypothesis/check: independent review still does.
"""
from collections import Counter
from math import log
import re

from . import smart_evidence


POLICY = "smart-body-relevance-v1"


def select(rows, *, request, decision, term_set, candidate_text, max_items=8):
    scope = {"company_id": request.company_id, "machine_id": request.machine_id,
             "ai_scope": request.ai_scope}
    # Validate ALL received producer IDs, owners and bodies, even beyond the cap.
    # SQL-scoped producer merges/projections are an existing upstream boundary;
    # this is not a claim to have observed or validated every raw database row.
    # No snippet equivalence, metadata-based exemption or best-effort dropping.
    smart_evidence.select_complete_sources(rows, scope=scope, max_items=0,
                                           allow_raw_snippet=True)
    # Use exactly the body which will enter the grounding packet. A UI snippet
    # cannot substitute for a producer's complete `text` field.
    bodies = [smart_evidence._normalize(row, scope, True)[1] for row in rows]
    text_terms = [term_set(body, limit=4096) for body in bodies]
    source_terms = {}
    for row, terms in zip(rows, text_terms):
        owner = str(row.get("bubble_document_id") or row.get("source_id"))
        source_terms.setdefault(owner, set()).update(terms)
    frequency = Counter(term for terms in source_terms.values() for term in terms)
    source_count = len(source_terms)

    # Use actual bilingual search phrases, not their assigned facet-hit labels.
    phrases = [request.query]
    facets = list(decision.facet_queries or [])
    for facet in facets:
        phrases.extend([facet.facet, *facet.dense_queries, *facet.lexical_queries])
    phrases.extend(decision.dense_queries or [])
    phrases.extend(decision.lexical_queries or [])
    query_terms = term_set("\n".join(str(p) for p in phrases if p), limit=4096)
    subsystem_terms = term_set("\n".join(decision.diagnostic_subsystems or []), limit=4096)
    identifiers = [term_set(value, limit=100) for value in re.findall(
        r"\b[\w]+(?:[-/][\w]+)+\b", "\n".join(str(p) for p in phrases if p))
        if any(c.isdigit() for c in value)]

    def independent_count(matches):
        # A composite identifier is one clue, even if tokenization splits it.
        # This applies equally to model names and fault/component identifiers.
        remaining, count = set(matches), 0
        for group in identifiers:
            if len(group) > 1 and group <= remaining:
                remaining -= group
                count += 1
        return count + len(remaining)
    # Common model/name/boilerplate terms carry little ranking weight. Keep them:
    # a genuinely focused corpus can repeat every symptom term, even when a
    # reported condition such as 'after service' is absent from all sources.
    ubiquitous = {term for term, count in frequency.items()
                  if source_count >= 3 and count / source_count >= 0.60}
    informative = query_terms | subsystem_terms
    weights = {term: (1.0 + log((1 + source_count) / (1 + frequency[term])))
               * (2.0 if term in subsystem_terms else 0.1 if term in ubiquitous else 1.0)
               for term in informative}
    denominator = sum(weights.values()) or 1.0
    relevant_ids = set(decision.relevant_evidence_ids or [])
    ranked = []
    for ordinal, (row, terms) in enumerate(zip(rows, text_terms)):
        matches = terms & informative
        semantic = max(0.0, min(1.0, float(row.get("semantic_similarity",
            row.get("gate_similarity", row.get("similarity", 0.0))) or 0.0)))
        # A router ID, high retrieval score or facet tag alone cannot admit a
        # cover/contents page. Two real terms or one corroborated term are needed.
        corroborated = semantic >= 0.42 or row.get("citation_id") in relevant_ids
        exact_query = bool(matches and independent_count(query_terms) == 1)
        eligible = independent_count(matches) >= 2 or exact_query or bool(matches & subsystem_terms and corroborated)
        if not eligible or row.get("hard_excluded"):
            continue
        body_score = sum(weights[t] for t in matches) / denominator
        # Text relevance precedes advisory semantic/router signals. No old
        # v13_score, synthetic exact-code bonus or facet support is used here.
        rank = (body_score, len(matches & subsystem_terms), semantic,
                row.get("citation_id") in relevant_ids, -ordinal)
        ranked.append((rank, row))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    # First expose up to two distinct records per source; fill any remaining
    # capacity afterwards. Full-body aliases are coalesced by the existing owner
    # and body validator, while different pages/units/conditions remain distinct.
    first, remainder, per_source = [], [], Counter()
    for _, row in ranked:
        owner = str(row.get("bubble_document_id") or row.get("source_id"))
        (first if per_source[owner] < 2 else remainder).append(row)
        per_source[owner] += 1
    selected = smart_evidence.select_complete_sources(first + remainder,
        scope=scope, max_items=max_items, allow_raw_snippet=True)
    return selected, {"policy": POLICY, "input_count": len(rows),
        "eligible_count": len(ranked), "selected_count": len(selected),
        "input_source_types": dict(Counter(str(row.get("source_type") or "unknown") for row in rows)),
        "selected_ids": [row["citation_id"] for row in selected],
        "selected_source_types": dict(Counter(str(row.get("source_type") or "unknown") for row in selected)),
        "added_reads": 0, "added_model_calls": 0,
        "source_truth_verified": False}
