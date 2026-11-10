"""Request-bound Root retrieval hints; never observations or causal proof.

Only the server can place the frozen carrier in a Decision. The selector callback
recomputes matches from that carrier and the current source body on every call;
candidate annotations and serialized dictionaries cannot manufacture support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import hashlib
import json
import re

from . import diagnostic_query as dq
from .retrieval_policy import content_term_set, ContentTermSetRuntime
from ..ingest.text_pdf import normalize_unicode_advanced

POLICY_VERSION = "root-bound-clue-support-v1"
WIRE_KEY = "_root_clue_support"
_SEAL = object()
_TERMS_RUNTIME = ContentTermSetRuntime(normalize_unicode_advanced, re)
_WEIGHTS = (("discriminants", .40), ("operating_conditions", .24),
            ("observables", .22), ("subsystems", .14))


@dataclass(frozen=True)
class RootClueSupport:
    original_query: str
    query_sha256: str
    company_id: str
    machine_id: str
    parts: tuple[dq.RequestPart, ...]
    _binding_sha256: str = field(repr=False)
    _seal: object = field(repr=False, compare=False)
    router_source_hashes: tuple[tuple[str, str], ...] = ()

    def __deepcopy__(self, memo: dict) -> "RootClueSupport":
        # Every field is immutable; preserve the server-only provenance marker.
        return self


def _binding(query: str, company: str, machine: str, parts: tuple,
             router_sources: tuple = ()) -> str:
    wire = [query, company, machine, [p.to_dict() for p in parts], router_sources]
    return hashlib.sha256(json.dumps(wire, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _valid(value: Any) -> bool:
    if not isinstance(value, RootClueSupport) or value._seal is not _SEAL:
        return False
    if value.query_sha256 != dq.query_fingerprint(value.original_query):
        return False
    if (not isinstance(value.original_query, str) or not isinstance(value.parts, tuple)
            or not all(isinstance(p, dq.RequestPart) for p in value.parts)):
        return False
    if not _valid_router_sources(value.router_source_hashes):
        return False
    if value._binding_sha256 != _binding(value.original_query, value.company_id, value.machine_id,
                                        value.parts, value.router_source_hashes):
        return False
    checked, error = dq._partition([p.to_dict() for p in value.parts], value.original_query)
    return not error and checked == value.parts


def _valid_router_sources(value: Any) -> bool:
    return bool(isinstance(value, tuple) and len(value) <= 64
                and all(isinstance(row, tuple) and len(row) == 2
                        and isinstance(row[0], str) and 0 < len(row[0]) <= 512
                        and isinstance(row[1], str) and re.fullmatch(r"[0-9a-f]{64}", row[1])
                        for row in value)
                and len({row[0] for row in value}) == len(value))


def snapshot_router_sources(sources: Any) -> tuple[tuple[str, str], ...]:
    """Snapshot source identity BEFORE routing; ambiguous duplicate IDs get no lane."""
    if not isinstance(sources, (tuple, list)) or len(sources) > 64:
        return ()
    hashes, ambiguous = {}, set()
    for row in sources:
        if (not isinstance(row, (tuple, list)) or len(row) != 2
                or not isinstance(row[0], str) or not 0 < len(row[0]) <= 512
                or not isinstance(row[1], str) or not row[1].strip()):
            continue
        cid, text = row
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cid in hashes and hashes[cid] != digest:
            ambiguous.add(cid)
        hashes[cid] = digest
    return tuple(sorted((cid, digest) for cid, digest in hashes.items() if cid not in ambiguous))


def from_profile(profile: dq.DiagnosticQueryProfile, *, company_id: str,
                 machine_id: str, router_sources: tuple = ()) -> RootClueSupport | None:
    if profile.interpretation_status != "valid" or not profile.diagnostic_projection:
        return None
    support, gaps, error = dq._basis_parts(profile)
    if error or gaps or not support:
        return None
    bindings = router_sources if _valid_router_sources(router_sources) else ()
    value = RootClueSupport(profile.original_query, dq.query_fingerprint(profile.original_query),
                            str(company_id), str(machine_id), profile.parts,
                            _binding(profile.original_query, str(company_id), str(machine_id), profile.parts,
                                     bindings), _SEAL, bindings)
    return value if _valid(value) else None


def for_request(value: Any, *, query: str, company_id: str,
                machine_id: str) -> RootClueSupport | None:
    if (not _valid(value) or dq._space(value.original_query) != dq._space(str(query or ""))
            or value.company_id != str(company_id) or value.machine_id != str(machine_id)):
        return None
    return value


def _terms(value: str, limit: int) -> set[str]:
    return content_term_set(value, limit=limit, runtime=_TERMS_RUNTIME)


def _matches(text: str, clue: str, exact_terms: tuple[str, ...],
             metrics: Callable[[str, list[str]], dict]) -> bool:
    source_terms, clue_terms = _terms(text, 260), _terms(clue, 30)
    literals: set[str] = set()
    for exact in (*exact_terms, *dq.protected_literals(clue)):
        tokens = _terms(exact, 30)
        if tokens and tokens <= clue_terms:
            literals.update(tokens)
    remainder = clue_terms - literals
    # A model/header alone cannot satisfy an additional descriptive observation.
    if literals and remainder and not (remainder & source_terms):
        return False
    # Unlike broad word overlap, a different code or measured value is not a
    # match. Whitespace within a numeric-unit literal is the only equivalence.
    if not dq.protected_literals_present(clue, text):
        return False
    return bool(metrics(text, [clue]).get("covered"))


def _coverage(text: str, facets: list[str]) -> dict:
    normalized = re.sub(r"\s+", " ", normalize_unicode_advanced(text).lower()).strip()
    source_terms = _terms(normalized, 260)
    covered = []
    for facet in facets:
        phrase = re.sub(r"\s+", " ", normalize_unicode_advanced(facet).lower()).strip()
        terms = _terms(phrase, 30)
        if phrase and (phrase in normalized or (terms and len(terms & source_terms) / len(terms) >= .50)):
            covered.append(facet)
    return {"covered": covered}


def _exact_terms(decision: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*decision.exact_terms,
        *(term for facet in decision.facet_queries for term in facet.exact_terms)]))


def _part_matches(part: dq.RequestPart, text: str, exact_terms: tuple[str, ...],
                  metrics: Callable[[str, list[str]], dict]) -> bool:
    return any(_matches(text, variant, exact_terms, metrics)
               for variant in (part.quote, *part.retrieval_aliases))


def support_match(decision: Any, text: str, *, source_id: str = "") -> dict:
    """Compute new matches; neither a candidate flag nor a cached verdict is input."""
    value = getattr(decision, "root_clue_support", None)
    if getattr(decision, "effective_mode", None) != "root_cause" or not _valid(value):
        return {}
    matched = [f"{p.start}:{p.end}:{p.role}" for p in value.parts
               if p.role in dq.FACT_ROLES and p.basis_use == "support"
               and _part_matches(p, text, _exact_terms(decision), _coverage)]
    source_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    relevant = getattr(decision, "relevant_evidence_ids", ())
    router_context = bool(isinstance(source_id, str) and source_id
                          and isinstance(relevant, tuple) and source_id in relevant
                          and dict(value.router_source_hashes).get(source_id) == source_digest)
    return {"matched_support_part_ids": tuple(dict.fromkeys(matched)),
            "query_sha256": value.query_sha256,
            "source_text_sha256": source_digest,
            "router_context_selected": router_context,
            "router_context_source_id": source_id if router_context else "",
            "policy_version": POLICY_VERSION}


def priority_metrics(decision: Any, text: str, *, source_type: str,
                     substantive_ps: bool,
                     metrics: Callable[[str, list[str]], dict]) -> dict:
    """Keep existing weights, count the same original part once at its max share."""
    value = getattr(decision, "root_clue_support", None)
    # Decision clues use _clean_text(..., 240) and case-insensitive uniqueness.
    # Resolve that same bounded key only when it names one original request
    # part. Two longer parts sharing a prefix must not borrow either alias.
    parts: dict[str, dq.RequestPart | None] = {}
    if _valid(value):
        for part in value.parts:
            if part.role not in dq.FACT_ROLES:
                continue
            key = " ".join(part.quote.split()).strip()[:240].casefold()
            parts[key] = None if key in parts else part
    exact_terms = _exact_terms(decision)
    events: dict[Any, dict] = {}
    for group, weight in _WEIGHTS:
        clues = tuple(dict.fromkeys(getattr(decision, "diagnostic_" + group)))
        if not clues:
            continue
        share = weight / len(clues)
        for clue in clues:
            part = parts.get(" ".join(clue.split()).strip()[:240].casefold())
            identity = (part.start, part.end, part.role) if part else ("literal", clue)
            if identity not in events or share > events[identity]["weight"]:
                events[identity] = {"clue": clue, "part": part, "group": group, "weight": share}
    for event in events.values():
        part = event["part"]
        event["covered"] = (_part_matches(part, text, exact_terms, metrics) if part else
                            _matches(text, event["clue"], exact_terms, metrics))
    total = sum(e["weight"] for e in events.values())
    score = sum(e["weight"] for e in events.values() if e["covered"]) / total if total else 0.0
    groups = {}
    for name, _ in _WEIGHTS:
        items = [e for e in events.values() if e["group"] == name]
        denominator = sum(e["weight"] for e in items)
        groups[name] = sum(e["weight"] for e in items if e["covered"]) / denominator if denominator else 0.0
    bonus = 0.0
    if source_type == "ps" and substantive_ps:
        if groups["discriminants"] >= .34:
            bonus = .20
        elif groups["observables"] >= .34 and groups["subsystems"] > 0:
            bonus = .12
    return {"score": min(1.0, score + bonus), "base_score": score, "exact_case_bonus": bonus,
            "covered": list(dict.fromkeys(e["clue"] for e in events.values() if e["covered"])),
            "missing": list(dict.fromkeys(e["clue"] for e in events.values() if not e["covered"])),
            "groups": groups}
