"""Relevance-gated evidence compaction for MachineMind Root Cause.

This module operates *after* retrieval, scope enforcement, diagnostic rescoring
and the existing viability gate.  Its purpose is deliberately narrow: build a
small final evidence pack without allowing source diversity by itself to admit a
weakly related document or structured record.

The selector does not inspect the user's query, does not retrieve data, does not
infer causes and does not know machine-specific vocabulary.  It consumes only
bounded, generic signals already attached to each candidate by the composition
root (diagnostic clue coverage, facets, causal/subsystem/context support and
semantic similarity).  Source diversity remains useful as a saturation guard,
but it is never an admission criterion on its own.

The boundary is Root Cause only.  ASK and Smart Diagnostic keep their existing
selection paths.

A separate pure claim validator binds adjudicator outputs to literal observation
and source spans. It does not replace semantic judgement with lexical ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import math
import re
import unicodedata


_POLICY_VERSION = "root-diagnostic-source-selection-v1"
_GROUP_THRESHOLD = 0.20


@dataclass(frozen=True)
class DiagnosticSourceRuntime:
    """Composition-root callbacks needed to preserve legacy candidate identity."""

    source_type: Callable[[dict], str]
    stable_key: Callable[[dict], str]
    family_key: Callable[[dict], str]


@dataclass(frozen=True)
class DiagnosticSelectionPolicy:
    """Bounded admission policy for a final Root Cause evidence pack."""

    version: str = _POLICY_VERSION
    per_source_cap: int = 3
    per_family_cap: int = 2
    absolute_quality_floor: float = 0.30
    relative_peer_margin: float = 0.16
    novel_signal_margin: float = 0.24
    minimum_peer_axes: int = 2
    minimum_novel_axes: int = 3


@dataclass(frozen=True)
class DiagnosticCandidateAssessment:
    candidate: dict
    input_rank: int
    stable_key: str
    source_key: str
    family_key: str
    page_key: tuple[str, int, int]
    source_type: str
    quality: float
    diagnostic_priority: float
    base_priority: float
    semantic_similarity: float
    causal_support: float
    subsystem_support: float
    context_support: float
    facet_coverage: float
    strong_group_count: int
    independent_axis_count: int
    strict_documented_case: bool
    generic_downranked: bool
    role_group: str
    evidence_tokens: frozenset[str]


@dataclass(frozen=True)
class DiagnosticSelectionResult:
    candidates: tuple[dict, ...]
    summary: dict


def _bounded(value: Any, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def _positive_scaled(value: Any, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return _bounded(max(0.0, _bounded(value, low=-10.0, high=10.0)) / scale)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _normalize_token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]


def _source_key(candidate: dict, runtime: DiagnosticSourceRuntime) -> str:
    source_type = str(runtime.source_type(candidate) or "document").strip().lower()
    document_id = str(candidate.get("bubble_document_id") or "").strip()
    fallback = str(
        candidate.get("citation_id") or runtime.stable_key(candidate) or ""
    ).strip()
    return f"{source_type}|{document_id or fallback}"


def _semantic_similarity(candidate: Mapping[str, Any]) -> float:
    for key in ("semantic_similarity", "gate_similarity", "similarity"):
        if key in candidate and candidate.get(key) is not None:
            return _bounded(candidate.get(key))
    return 0.0


def _group_scores(candidate: Mapping[str, Any]) -> dict[str, float]:
    diagnostic = candidate.get("assistant_core_diagnostic_priority") or {}
    if not isinstance(diagnostic, Mapping):
        return {}
    raw = diagnostic.get("groups") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name or "").strip().lower(): _bounded(value)
        for name, value in raw.items()
        if str(name or "").strip()
    }


def _diagnostic_priority(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    diagnostic = candidate.get("assistant_core_diagnostic_priority") or {}
    if not isinstance(diagnostic, Mapping):
        diagnostic = {}
    score = _bounded(diagnostic.get("score"))
    exact_bonus = _bounded(diagnostic.get("exact_case_bonus"), high=0.40)
    base = _bounded(diagnostic.get("base_score"))
    if base <= 0.0 and score > 0.0:
        base = _bounded(score - exact_bonus)
    return score, base, exact_bonus


def _strict_documented_case(
    *,
    source_type: str,
    base_priority: float,
    exact_bonus: float,
    groups: Mapping[str, float],
) -> bool:
    """Accept an exact P&S bonus only when it is corroborated by another clue.

    The historical viability gate may grant a P&S bonus from one high-value
    discriminant.  That is useful for recall, but too permissive for the final
    source pack: a generic operating-state word can otherwise make an unrelated
    record look like an exact documented case.  The final selector therefore
    requires multi-signal support without knowing any symptom vocabulary.
    """
    if source_type != "ps" or exact_bonus <= 0.0:
        return False

    strong = [name for name, value in groups.items() if value >= _GROUP_THRESHOLD]
    discriminants = _bounded(groups.get("discriminants"))
    observables = _bounded(groups.get("observables"))
    subsystems = _bounded(groups.get("subsystems"))
    operating = _bounded(groups.get("operating_conditions"))
    corroborated_discriminant = bool(
        discriminants >= 0.34
        and max(observables, subsystems, operating) >= _GROUP_THRESHOLD
    )
    corroborated_observable = bool(
        observables >= 0.34 and subsystems >= 0.10
    )
    return bool(
        base_priority >= 0.22
        and (
            len(strong) >= 2
            or corroborated_discriminant
            or corroborated_observable
        )
    )


def _evidence_tokens(
    candidate: Mapping[str, Any],
    *,
    groups: Mapping[str, float],
) -> frozenset[str]:
    tokens: set[str] = set()
    for name, value in groups.items():
        if value >= _GROUP_THRESHOLD:
            tokens.add(f"group:{name}")

    for key in ("assistant_core_facet_hits", "assistant_core_covered_facets"):
        raw = candidate.get(key) or []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            continue
        for value in raw:
            normalized = _normalize_token(value)
            if normalized:
                tokens.add(f"facet:{normalized}")

    raw_subsystems = candidate.get("matched_subsystems") or []
    if isinstance(raw_subsystems, Sequence) and not isinstance(
        raw_subsystems, (str, bytes, bytearray)
    ):
        for value in raw_subsystems:
            normalized = _normalize_token(value)
            if normalized:
                tokens.add(f"subsystem:{normalized}")
    return frozenset(tokens)


def assess_root_cause_candidate(
    candidate: dict,
    *,
    input_rank: int,
    runtime: DiagnosticSourceRuntime,
) -> DiagnosticCandidateAssessment:
    item = dict(candidate or {})
    source_type = str(runtime.source_type(item) or "document").strip().lower()
    stable_key = str(runtime.stable_key(item) or "").strip()
    source_key = _source_key(item, runtime)
    family_key = str(runtime.family_key(item) or source_key).strip() or source_key
    page_from = _safe_int(item.get("page_from"), 0)
    page_to = _safe_int(item.get("page_to"), page_from)

    groups = _group_scores(item)
    priority, base_priority, exact_bonus = _diagnostic_priority(item)
    semantic = _semantic_similarity(item)
    causal_raw = _bounded(item.get("causal_strength_score"), low=-1.0, high=1.0)
    subsystem_raw = _bounded(item.get("subsystem_score"), low=-1.0, high=1.0)
    context_raw = _bounded(item.get("context_fit_score"), low=-1.0, high=1.0)
    causal = _positive_scaled(causal_raw, 0.40)
    subsystem = _positive_scaled(subsystem_raw, 0.40)
    context = _positive_scaled(context_raw, 0.30)
    facet = _bounded(item.get("assistant_core_facet_coverage"))
    role_group = str(item.get("role_group") or "").strip().lower()
    strong_group_count = sum(
        1 for value in groups.values() if value >= _GROUP_THRESHOLD
    )
    strict_case = _strict_documented_case(
        source_type=source_type,
        base_priority=base_priority,
        exact_bonus=exact_bonus,
        groups=groups,
    )
    generic = bool(item.get("generic_downranked"))

    # Independent support axes.  A candidate may be cross-language and therefore
    # weak on lexical facets, but it still needs corroboration beyond one broad
    # clue before entering the final Root Cause pack.
    axis_count = sum(
        (
            priority >= 0.20,
            facet >= 0.25,
            causal_raw >= 0.06,
            subsystem_raw > 0.0 or context_raw > 0.0,
            semantic >= 0.45,
            strong_group_count >= 2,
        )
    )

    role_bonus = (
        1.0
        if role_group == "core"
        else 0.55
        if role_group == "support"
        else 0.0
    )
    quality = (
        0.26 * priority
        + 0.12 * base_priority
        + 0.14 * facet
        + 0.12 * semantic
        + 0.14 * causal
        + 0.09 * subsystem
        + 0.06 * context
        + 0.03 * role_bonus
    )
    if strong_group_count >= 2:
        quality += 0.04
    if axis_count >= 3:
        quality += 0.04
    if strict_case:
        quality += 0.06
    if generic:
        quality -= 0.10
    if role_group == "collateral":
        quality -= 0.07
    quality = _bounded(quality)

    return DiagnosticCandidateAssessment(
        candidate=item,
        input_rank=int(input_rank),
        stable_key=stable_key,
        source_key=source_key,
        family_key=family_key,
        page_key=(source_key, page_from, page_to),
        source_type=source_type,
        quality=quality,
        diagnostic_priority=priority,
        base_priority=base_priority,
        semantic_similarity=semantic,
        causal_support=causal_raw,
        subsystem_support=subsystem_raw,
        context_support=context_raw,
        facet_coverage=facet,
        strong_group_count=strong_group_count,
        independent_axis_count=axis_count,
        strict_documented_case=strict_case,
        generic_downranked=generic,
        role_group=role_group,
        evidence_tokens=_evidence_tokens(item, groups=groups),
    )


def _annotated_candidate(
    assessment: DiagnosticCandidateAssessment,
    *,
    selected_rank: int,
    reason: str,
    policy: DiagnosticSelectionPolicy,
) -> dict:
    item = dict(assessment.candidate)
    item["assistant_core_diagnostic_source_selection"] = {
        "policy_version": policy.version,
        "selected_rank": int(selected_rank),
        "reason": str(reason),
        "quality": round(float(assessment.quality), 6),
        "independent_axis_count": int(assessment.independent_axis_count),
        "strong_group_count": int(assessment.strong_group_count),
        "strict_documented_case": bool(assessment.strict_documented_case),
    }
    return item


def select_root_cause_candidates(
    candidates: Iterable[dict],
    *,
    limit: int,
    runtime: DiagnosticSourceRuntime,
    policy: DiagnosticSelectionPolicy | None = None,
) -> DiagnosticSelectionResult:
    """Return a relevance-gated, source-capped Root Cause evidence pack.

    Candidates are expected to be pre-sorted by the existing retrieval/ranking
    path and to have passed ``assistant_core_root_viable``.  No candidate is
    retained merely because it is ranked first: the initial evidence must satisfy
    an absolute diagnostic admission contract.  Later candidates may be a
    close-quality peer only after a valid pack has been seeded, or may enter as a
    corroborated documented case / genuinely new supported diagnostic signal.
    A new source alone is never enough.
    """
    policy = policy or DiagnosticSelectionPolicy()
    limit = max(0, int(limit or 0))
    raw_items = [
        dict(item) for item in (candidates or []) if isinstance(item, dict)
    ]
    if limit <= 0 or not raw_items:
        return DiagnosticSelectionResult(
            candidates=(),
            summary={
                "policy_version": policy.version,
                "input_count": len(raw_items),
                "unique_count": 0,
                "selected_count": 0,
                "pruned_count": len(raw_items),
                "selected_source_count": 0,
                "rejection_counts": {},
            },
        )

    assessments: list[DiagnosticCandidateAssessment] = []
    seen_ids: set[str] = set()
    seen_pages: set[tuple[str, int, int]] = set()
    duplicate_count = 0
    hard_excluded_count = 0
    for rank, raw in enumerate(raw_items):
        if bool(raw.get("hard_excluded")):
            hard_excluded_count += 1
            continue
        assessed = assess_root_cause_candidate(raw, input_rank=rank, runtime=runtime)
        if not assessed.stable_key:
            duplicate_count += 1
            continue
        if assessed.stable_key in seen_ids or assessed.page_key in seen_pages:
            duplicate_count += 1
            continue
        seen_ids.add(assessed.stable_key)
        seen_pages.add(assessed.page_key)
        assessments.append(assessed)

    if not assessments:
        return DiagnosticSelectionResult(
            candidates=(),
            summary={
                "policy_version": policy.version,
                "input_count": len(raw_items),
                "unique_count": 0,
                "selected_count": 0,
                "pruned_count": len(raw_items),
                "selected_source_count": 0,
                "rejection_counts": {
                    "duplicate_or_missing_identity": duplicate_count,
                    "hard_excluded": hard_excluded_count,
                },
            },
        )

    top_quality = max(item.quality for item in assessments)
    peer_floor = max(
        policy.absolute_quality_floor,
        top_quality - policy.relative_peer_margin,
    )
    novel_floor = max(
        policy.absolute_quality_floor - 0.04,
        top_quality - policy.novel_signal_margin,
    )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_pages: set[tuple[str, int, int]] = set()
    source_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    covered_tokens: set[str] = set()
    rejection_reasons: dict[str, str] = {}

    def capacity_reason(assessed: DiagnosticCandidateAssessment) -> str:
        if len(selected) >= limit:
            return "limit"
        if assessed.stable_key in selected_ids or assessed.page_key in selected_pages:
            return "duplicate"
        if source_counts.get(assessed.source_key, 0) >= max(
            1, policy.per_source_cap
        ):
            return "source_cap"
        if family_counts.get(assessed.family_key, 0) >= max(1, policy.per_family_cap):
            return "family_cap"
        return ""

    def add(assessed: DiagnosticCandidateAssessment, reason: str) -> bool:
        # Relative/fallback evidence may enrich a valid pack, but it must never
        # manufacture the first cause.  This allows an intentionally empty pack
        # when every retrieved candidate is too weak or ambiguous.
        strong_seed_reasons = {
            "corroborated_documented_case",
            "new_supported_signal",
            "core_mechanism",
        }
        if not selected and reason not in strong_seed_reasons:
            rejection_reasons.setdefault(assessed.stable_key, "insufficient_absolute_evidence")
            return False
        blocked = capacity_reason(assessed)
        if blocked:
            rejection_reasons.setdefault(assessed.stable_key, blocked)
            return False
        if not selected and assessed is assessments[0]:
            reason = "anchor"
        selected_rank = len(selected) + 1
        selected.append(
            _annotated_candidate(
                assessed,
                selected_rank=selected_rank,
                reason=reason,
                policy=policy,
            )
        )
        selected_ids.add(assessed.stable_key)
        selected_pages.add(assessed.page_key)
        source_counts[assessed.source_key] = source_counts.get(assessed.source_key, 0) + 1
        family_counts[assessed.family_key] = family_counts.get(assessed.family_key, 0) + 1
        covered_tokens.update(assessed.evidence_tokens)
        rejection_reasons.pop(assessed.stable_key, None)
        return True

    def contract_flags(
        assessed: DiagnosticCandidateAssessment,
    ) -> tuple[bool, bool, bool, bool]:
        new_tokens = set(assessed.evidence_tokens) - covered_tokens
        close_peer = bool(
            assessed.quality >= peer_floor
            and assessed.independent_axis_count >= policy.minimum_peer_axes
            and (
                assessed.strong_group_count >= 2
                or assessed.strict_documented_case
                or assessed.causal_support >= 0.06
                or (
                    assessed.subsystem_support > 0.0
                    and assessed.context_support > 0.0
                )
                or assessed.facet_coverage >= 0.50
            )
        )
        documented_case = bool(
            assessed.strict_documented_case
            and assessed.quality >= novel_floor
            and assessed.independent_axis_count >= policy.minimum_peer_axes
        )
        novel_supported = bool(
            new_tokens
            and assessed.quality >= novel_floor
            and assessed.independent_axis_count >= policy.minimum_novel_axes
            and (
                assessed.strong_group_count >= 2
                or (
                    assessed.causal_support >= 0.08
                    and (
                        assessed.subsystem_support > 0.0
                        or assessed.context_support > 0.0
                    )
                )
                or (
                    assessed.subsystem_support > 0.0
                    and assessed.context_support > 0.0
                    and assessed.semantic_similarity >= 0.45
                )
            )
        )
        core_mechanism = bool(
            assessed.role_group == "core"
            and assessed.quality >= novel_floor
            and assessed.independent_axis_count >= policy.minimum_novel_axes
            and (assessed.causal_support >= 0.08 or assessed.subsystem_support > 0.0)
        )
        return documented_case, novel_supported, core_mechanism, close_peer

    # Coverage pass: every candidate, including rank 1, must pass an absolute
    # diagnostic admission contract.  The first admitted rank-1 candidate keeps
    # the public ``anchor`` label, but ranking alone never grants admission.
    # This also permits a zero-evidence result for underspecified symptoms.
    for assessed in assessments:
        if len(selected) >= limit:
            break
        (
            documented_case,
            novel_supported,
            core_mechanism,
            _close_peer,
        ) = contract_flags(assessed)
        if documented_case:
            add(assessed, "corroborated_documented_case")
        elif novel_supported:
            add(assessed, "new_supported_signal")
        elif core_mechanism:
            add(assessed, "core_mechanism")

    # Relevance-fill pass: add only close-quality peers.  Different source identity
    # is intentionally absent from this admission decision.
    for assessed in assessments:
        if assessed.stable_key in selected_ids:
            continue
        if not selected:
            rejection_reasons.setdefault(
                assessed.stable_key, "insufficient_absolute_evidence"
            )
            continue
        if len(selected) >= limit:
            rejection_reasons.setdefault(assessed.stable_key, "limit")
            continue
        (
            _documented_case,
            _novel_supported,
            _core_mechanism,
            close_peer,
        ) = contract_flags(assessed)
        if close_peer:
            add(assessed, "close_quality_peer")
            continue
        if assessed.strong_group_count <= 1 and not assessed.strict_documented_case:
            rejection_reasons.setdefault(
                assessed.stable_key, "single_clue_without_corroboration"
            )
        else:
            rejection_reasons.setdefault(assessed.stable_key, "below_relevance_floor")

    # Assign a stable reason to candidates that were skipped after the pack became
    # full during the coverage pass.
    for assessed in assessments:
        if assessed.stable_key not in selected_ids:
            rejection_reasons.setdefault(
                assessed.stable_key,
                "limit" if len(selected) >= limit else "below_relevance_floor",
            )

    rejection_counts: dict[str, int] = {}
    for reason in rejection_reasons.values():
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    summary = {
        "policy_version": policy.version,
        "input_count": len(raw_items),
        "unique_count": len(assessments),
        "selected_count": len(selected),
        "pruned_count": max(0, len(raw_items) - len(selected)),
        "selected_source_count": len(source_counts),
        "top_quality": round(float(top_quality), 6),
        "peer_quality_floor": round(float(peer_floor), 6),
        "novel_quality_floor": round(float(novel_floor), 6),
        "rejection_counts": {
            **({"duplicate_or_missing_identity": duplicate_count} if duplicate_count else {}),
            **({"hard_excluded": hard_excluded_count} if hard_excluded_count else {}),
            **dict(sorted(rejection_counts.items())),
        },
        "selected": [
            {
                "citation_id": str(item.get("citation_id") or ""),
                **dict(item.get("assistant_core_diagnostic_source_selection") or {}),
            }
            for item in selected
        ],
    }
    return DiagnosticSelectionResult(candidates=tuple(selected), summary=summary)

# Claim-level evidence admission is separate from candidate ranking. A highly
# ranked excerpt is not proof that it describes the component in the question.
CAUSAL_GROUNDING_POLICY = "root-causal-applicability-v1"
CAUSAL_GROUNDING_INSTRUCTION = (
    " You must validate every retained hypothesis against both the CURRENT "
    "observations and the applicability of its cited passage. Retrieval rank, "
    "score, shared vocabulary and membership in the same machine are NOT proof. "
    "SOURCES is a JSON list of records. text is the selected excerpt. "
    "ownership_context contains ordered neighbouring pages from the SAME "
    "authorized document, only to determine the governing heading, model or "
    "component. Follow the nearest governing heading at the excerpt's position; "
    "a different heading earlier/later on the page must not change its owner. "
    "Context is not a new selectable citation and must not supply extra claims. "
    "For each cause return support proofs with verbatim observation_quote, "
    "source_quote, and target_quote. The observation quote must come only from "
    "OBSERVED_SYMPTOM; source_quote from that citation's text; target_quote from "
    "its text or ownership_context and establish why it applies to the affected "
    "component. Do not infer the owner of a fragment when it is not established. "
    "A checklist mentioning multiple utilities or components proves only that "
    "they are to be checked; it does NOT prove that one utility controls another "
    "component's ready/permissive signal. Cross-component causes need an explicit "
    "documented dependency, not juxtaposition. Unknown/not measured variables "
    "are NOT symptoms. A missing measurement alone cannot promote a cause. "
    "A general instruction to record comparable measurements is not a competing "
    "physical cause. Unsupported hypotheses, target transfers, and checklist-only "
    "associations must be omitted, even if labelled conditional or cautious. "
    "Keep a valid leading hypothesis and other independently supported mechanisms; "
    "do not fill the maximum number of slots. A bounded inference is allowed only "
    "from an applicable mechanism/check plus the actual observations and must "
    "state its uncertainty. Return cause-specific citations equal to the proof "
    "citation IDs. Never invent a literal quote to satisfy the schema. "
    "All strings except verbatim quotes/citation IDs must use RESPONSE_LANGUAGE."
)


def causal_grounding_schema(base_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Extend an internal LLM schema without changing public response schemas."""
    import copy
    schema = copy.deepcopy(dict(base_schema))
    schema["name"] = "machinemind_root_causal_applicability_v1"
    item = schema["schema"]["properties"]["possible_causes"]["items"]
    item["properties"]["support"] = {
        "type": "array", "minItems": 1, "maxItems": 3,
        "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "citation_id": {"type": "string"},
                "observation_quote": {"type": "string"},
                "source_quote": {"type": "string"},
                "target_quote": {"type": "string"},
                "applicability": {"type": "string", "enum": [
                    "same_target", "documented_dependency", "unknown", "different_target"]},
                "support_type": {"type": "string", "enum": [
                    "documented_mechanism", "bounded_inference", "checklist_only"]},
            },
            "required": ["citation_id", "observation_quote", "source_quote",
                         "target_quote", "applicability", "support_type"],
        },
    }
    item["required"] = list(item["required"]) + ["support"]
    return schema


def _quote_normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip().casefold()


def _literal_quote_in(quote: Any, body: Any, minimum: int = 8) -> bool:
    value = _quote_normalized(quote)
    return bool(minimum <= len(value) <= 1000 and value in _quote_normalized(body))


def validate_causal_grounding(
    *, parsed: Mapping[str, Any], observed_query: str,
    records: Sequence[Mapping[str, Any]], max_causes: int = 3,
) -> dict[str, Any]:
    """Bind model assessments to actual source/query spans; fail closed on error.

    Literal checks establish provenance, NOT entailment. Applicability judgement
    remains semantic and must be evaluated on live model outputs as well.
    """
    by_id = {str(r.get("citation_id") or ""): r for r in records
             if isinstance(r, Mapping) and r.get("citation_id")}
    raw_causes = parsed.get("possible_causes")
    if not isinstance(raw_causes, list):
        raw_causes = []
    accepted: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    limit = max(1, min(3, int(max_causes)))
    for index, cause in enumerate(raw_causes[:limit]):
        reason = "accepted"
        proofs = cause.get("support") if isinstance(cause, Mapping) else None
        valid_ids: list[str] = []
        if not isinstance(cause, Mapping) or not isinstance(proofs, list) or not 1 <= len(proofs) <= 3:
            reason = "missing_cause_support"
        else:
            for proof in proofs:
                if not isinstance(proof, Mapping):
                    reason = "malformed_support"; break
                cid = str(proof.get("citation_id") or "")
                record = by_id.get(cid)
                if record is None:
                    reason = "unknown_citation"; break
                if proof.get("applicability") not in {"same_target", "documented_dependency"}:
                    reason = "target_applicability_not_established"; break
                if proof.get("support_type") not in {"documented_mechanism", "bounded_inference"}:
                    reason = "checklist_is_not_causal_evidence"; break
                if not _literal_quote_in(proof.get("observation_quote"), observed_query):
                    reason = "observation_not_in_current_request"; break
                if not _literal_quote_in(proof.get("source_quote"), record.get("text"), 16):
                    reason = "source_quote_not_in_cited_excerpt"; break
                owner_text = str(record.get("text") or "") + "\n" + str(record.get("ownership_context") or "")
                if not _literal_quote_in(proof.get("target_quote"), owner_text):
                    reason = "target_quote_not_in_cited_context"; break
                if cid not in valid_ids:
                    valid_ids.append(cid)
            raw_ids = cause.get("citations")
            if reason == "accepted" and (
                not isinstance(raw_ids, list) or set(map(str, raw_ids)) != set(valid_ids)
            ):
                reason = "citations_not_bound_to_support"
            if reason == "accepted" and (not str(cause.get("cause") or "").strip()
                                        or not str(cause.get("why") or "").strip()
                                        or not isinstance(cause.get("checks"), list)):
                reason = "incomplete_cause"
        verdicts.append({"input_index": index, "accepted": reason == "accepted", "reason": reason})
        if reason == "accepted":
            accepted.append({
                "rank": len(accepted) + 1, "cause": str(cause["cause"]),
                "why": str(cause["why"]),
                "checks": [str(c) for c in cause["checks"][:5] if str(c).strip()],
                "citations": valid_ids,
            })
    used = list(dict.fromkeys(cid for cause in accepted for cid in cause["citations"]))
    return {
        "causes": accepted, "citation_ids": used,
        "summary": {"policy_version": CAUSAL_GROUNDING_POLICY,
                    "input_causes": len(raw_causes), "accepted_causes": len(accepted),
                    "rejected_causes": len(verdicts) - len(accepted), "verdicts": verdicts,
                    "support_proofs": [dict(p) for c in raw_causes[:limit]
                                       if isinstance(c, Mapping) for p in (c.get("support") or [])
                                       if isinstance(p, Mapping)][:9]},
    }
