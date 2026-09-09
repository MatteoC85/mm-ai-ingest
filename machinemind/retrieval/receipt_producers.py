"""P6-B4g: real receipt acquisition and explicit merge/dedup occurrence lineage.

These producers use the EXISTING scoped readers, B4a session, P4 ranking and the
B4e invocation guard. No session, ACL registry, alternate algorithm, model call,
embedding, SQL statement or language rule is introduced. Main remains OFF.

A trusted reader receives the session's exact immutable scope and allocation
limits, returns its actual B1/B2/B3 receipt, and is registered only against the
caller's CURRENT authorization after the read. read_sources is not a grant.
Merges record parent positions WHERE the algorithm creates each result; they do
not join authorization by citation ID, text, score or flags after the fact.

RRF/V13 merge, snippet/order dedup, V13 scoring, MMR, model reranking and
structured promotion now retain explicit occurrence lineage. P4 prepare, other
callback-internal acquisition, facts/rescue, cache and output/link transformations
still need producers before main enables canonical ASK. This is not full intake.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Callable

from . import candidate_ranking as ranking
from .ask_composition import AskEvidenceSession, RecordHandle, RegisteredRead
from ..evidence.contracts import EvidenceContractError, SourceIdentity
from ..evidence.ask_input import _same_value
from ..evidence.record_adapters import adapt_candidate

PRODUCER_VERSION = "ask-receipt-lineage-p6b4g-v1"


class EvidenceProductionError(EvidenceContractError):
    """Technical acquisition/lineage failure, never an empty evidence result."""


def _callbacks(session, authorize, invoke):
    if (type(session) is not AskEvidenceSession or not callable(authorize)
            or not callable(invoke)):
        raise EvidenceProductionError("existing session and current owner callbacks required")


def acquire_read(*, request: Any, session: AskEvidenceSession,
                 reader: Callable[..., Any],
                 authorize: Callable[[Any], frozenset[SourceIdentity]],
                 invoke: Callable[..., Any]) -> RegisteredRead:
    """Acquire once and register a typed receipt in the already-owned session.

    reader(scope=..., limits=..., current_allowed_sources=...) is a trusted
    repository adapter to ONE existing reader operation. It may perform that
    operation's existing multiple SQL reads but this helper adds none/retries none.
    It must return the authentic receipt, not a legacy list or fabricated source.
    invoke is the SAME B4e guard; request identity/lifetime/faults are not global.
    Authority runs before and after I/O; its own cost is not assumed to be zero.
    Original observations, unselected scans, unresolved relations and file-only
    metadata are retained by B4a exactly as its existing receipt contract requires.
    """
    _callbacks(session, authorize, invoke)
    if not callable(reader):
        raise EvidenceProductionError("explicit trusted receipt reader required")

    def work():
        before = invoke(authorize, request, request)
        scope, limits = session.read_contract(request=request, current_allowed_sources=before)
        try:
            receipt = invoke(reader, request, scope=scope, limits=limits,
                             current_allowed_sources=before)
        except EvidenceContractError:
            raise
        except Exception:
            # No DB/client secret or human-language dependent error matching.
            raise EvidenceProductionError("receipt provider failed") from None
        after = invoke(authorize, request, request)
        return session.register(request=request, receipt=receipt,
                                current_allowed_sources=after)

    # Latch registration/provider/authority faults through the existing owner.
    return invoke(work, request)


def rank_records(*, request: Any, session: AskEvidenceSession,
                 groups: tuple[tuple[RecordHandle, ...], ...], operation: str,
                 authorize: Callable[[Any], frozenset[SourceIdentity]],
                 invoke: Callable[..., Any], k: int = 60,
                 max_items: int | None = None,
                 snippet_runtime: ranking.DedupCitationsBySnippetRuntime | None = None
                 ) -> tuple[RecordHandle, ...]:
    """Execute one existing merge/selection with direct positional lineage.

    Fresh records are restored from explicit handles before computation, not
    accepted from arbitrary candidate dictionaries. All inputs are authorized
    again after callbacks, even if a revoked item would be dropped by dedup.
    Merges derive an atomic batch using B4a's source/locator/relationship checks.
    Selections retain the exact chosen handles (and create no new evidence).
    RRF records the chosen representative first, then rank contributors. A merge
    collision between DIFFERENT sources is rejected, not granted by legacy CID.
    V13 field merging also requires equal P5 locators: it must not attach longer
    text from one location to a different location selected by a CID collision.
    Original SQL text remains distinct from derived/ranking metadata.

    This preserves the four legacy algorithms and their tie-breaking; it does not
    remove their ID/snippet heuristics, certify model meaning, or implement score
    provenance. Temporary record copies have bounded inputs and CPU/heap overhead
    still to measure. No claim of atomic ACL revocation across remote calls.
    """
    _callbacks(session, authorize, invoke)

    def work():
        if type(operation) is not str or operation not in {"rrf", "merge", "dedup_snippet", "dedup_order"}:
            raise EvidenceProductionError("unsupported existing ranking operation")
        if type(groups) is not tuple or any(type(g) is not tuple for g in groups):
            raise EvidenceProductionError("immutable ordered handle groups required")
        if operation.startswith("dedup_") and len(groups) != 1:
            raise EvidenceProductionError("selection requires one explicit group")
        before = invoke(authorize, request, request)
        _, limits = session.read_contract(request=request, current_allowed_sources=before)
        if (len(groups) > limits.assembly.max_occurrences
                or sum(len(g) for g in groups) > limits.assembly.max_occurrences):
            raise EvidenceProductionError("too many groups/input occurrences")
        flat = tuple(h for g in groups for h in g)
        # Aggregate, not a separate limit per group; session also validates handles.
        inputs = session.records(request=request, handles=flat, current_allowed_sources=before)
        if any(i.layout != "retrieval_candidate" for i in inputs):
            raise EvidenceProductionError("page-to-candidate conversion must be explicit upstream")
        by_handle = {h: i for h, i in zip(flat, inputs)}
        records = [i.record for i in inputs]
        values, offset = [], 0
        for group in groups:
            values.append(records[offset:offset + len(group)])
            offset += len(group)
        traces = []

        def trace(parents):
            traces.append(parents)

        if operation == "rrf":
            if type(k) is not int or k < 0:
                raise EvidenceProductionError("nonnegative integer RRF parameter required")
            result = invoke(ranking.rrf_merge_candidates, request, deepcopy(values), k=k, lineage=trace)
        elif operation == "merge":
            result = invoke(ranking.v13_merge_candidates, request, deepcopy(values), lineage=trace)
        else:
            if (type(max_items) is not int or max_items < 1
                    or max_items > limits.assembly.max_occurrences):
                raise EvidenceProductionError("positive bounded selection count required")
            if operation == "dedup_snippet":
                if type(snippet_runtime) is not ranking.DedupCitationsBySnippetRuntime:
                    raise EvidenceProductionError("existing snippet runtime required")
                result = invoke(ranking.dedup_citations_by_snippet, request, deepcopy(values[0]),
                                max_items=max_items, runtime=snippet_runtime, lineage=trace)
            else:
                result = invoke(ranking.dedup_citations_preserve_order, request,
                                deepcopy(values[0]), max_items=max_items, lineage=trace)
        if (type(result) is not list or len(result) > limits.assembly.max_occurrences
                or len(traces) != 1 or type(traces[0]) is not tuple
                or len(traces[0]) != len(result)):
            raise EvidenceProductionError("missing/inconsistent direct operation lineage")
        views, selected = [], []
        for record, parents in zip(result, traces[0]):
            if (type(record) is not dict or type(parents) is not tuple or not parents
                    or len(parents) > limits.assembly.max_occurrences):
                raise EvidenceProductionError("explicit result and input occurrences required")
            parent_handles = []
            for pos in parents:
                if (type(pos) is not tuple or len(pos) != 2
                        or any(type(x) is not int for x in pos)
                        or not 0 <= pos[0] < len(groups)
                        or not 0 <= pos[1] < len(groups[pos[0]])):
                    raise EvidenceProductionError("lineage references an absent input occurrence")
                parent_handles.append(groups[pos[0]][pos[1]])
            if operation.startswith("dedup_"):
                if len(parents) != 1 or not _same_value(record, values[parents[0][0]][parents[0][1]]):
                    raise EvidenceProductionError("selection changed a record without a derivation")
                selected.append(parent_handles[0])
            else:
                if operation == "merge" and len(parent_handles) > 1:
                    locators = tuple(adapt_candidate(by_handle[h].record,
                        context=by_handle[h].context, limits=limits.adapter).entry.evidence.locator
                        for h in parent_handles)
                    if any(loc != locators[0] for loc in locators[1:]):
                        raise EvidenceProductionError("legacy merge combined incompatible locators")
                views.append((tuple(parent_handles), record))
        after = invoke(authorize, request, request)
        session.records(request=request, handles=flat, current_allowed_sources=after)
        if operation.startswith("dedup_"):
            return tuple(selected)
        return session.derive_batch(request=request, views=tuple(views),
            operation=PRODUCER_VERSION + ":" + operation,
            current_allowed_sources=after)

    return invoke(work, request)


TRANSFORM_VERSION = "ask-score-selection-lineage-p6b4h-v1"


def transform_records(*, request: Any, session: AskEvidenceSession,
                      groups: tuple[tuple[RecordHandle, ...], ...], operation: str,
                      runtime: Any,
                      authorize: Callable[[Any], frozenset[SourceIdentity]],
                      invoke: Callable[..., Any], q: str = "",
                      q_vec: list[float] | None = None, top_k: int | None = None,
                      lambda_mult: float = 0.85, diagnostic_mode: bool = False
                      ) -> tuple[RecordHandle, ...]:
    """Run the EXISTING score/MMR/reranker/structured-promotion algorithms.

    No thresholds, prompts, model budget or selection policy are changed. The
    reranker performs its one existing model callback only when legacy requires
    it; this producer adds none and never retries. Caller-supplied runtimes must
    contain the same trusted dependencies as the corresponding legacy operation.

    Inputs are restored from explicitly owned handles. All consumed inputs are
    authorized again after computation, including ones dropped by selection or
    not sent to the model because of the existing candidate cap. Score changes
    become atomic, single-parent derivations; selection returns ORIGINAL handles.
    A model's returned ID is only an algorithmic choice amongst its supplied
    items. Its parent position is captured in that same item-building loop; an
    ID, score, snippet or selection flag never becomes an authorization grant.

    For score/promotion the nested snippet dedup must SELECT the exact objects it
    receives (as the existing implementation does). Copying or rewriting that
    callback requires a separate producer, never an equality-based provenance
    guess. Immutable snapshots protect selection verification from mutations.

    No new SQL, session, ACL service or engine. Current-authority callback cost,
    temporary copies, CPU/heap and model usage must still be measured upstream.
    This does not implement all P4 prepare transformations, raw-page conversion,
    hidden callback reads, cache, URL verification or live canonical activation.
    Trusted collaborators must not mutate the session/request or hide their own
    contract errors; no claim of atomic remote revocation or Python sandboxing.
    """
    _callbacks(session, authorize, invoke)

    def work():
        expected_runtime = {
            "score": ranking.V13ScoreCandidatesRuntime,
            "mmr": ranking.MmrSelectRuntime,
            "rerank": ranking.LlmRerankCitationsRuntime,
            "promote_structured": ranking.PromoteStructuredRescueHitsRuntime,
        }
        if type(operation) is not str or operation not in expected_runtime:
            raise EvidenceProductionError("unsupported existing transform")
        if type(runtime) is not expected_runtime[operation]:
            raise EvidenceProductionError("explicit existing transform runtime required")
        expected_groups = 2 if operation == "promote_structured" else 1
        if (type(groups) is not tuple or len(groups) != expected_groups
                or any(type(group) is not tuple for group in groups)):
            raise EvidenceProductionError("explicit ordered handle groups required")
        if type(q) is not str or type(diagnostic_mode) is not bool:
            raise EvidenceProductionError("explicit query and boolean mode required")
        before = invoke(authorize, request, request)
        _, limits = session.read_contract(request=request, current_allowed_sources=before)
        if sum(len(group) for group in groups) > limits.assembly.max_occurrences:
            raise EvidenceProductionError("too many transform input occurrences")
        flat = tuple(handle for group in groups for handle in group)
        if operation != "score" and (type(top_k) is not int or not 1 <= top_k <= limits.assembly.max_occurrences):
            raise EvidenceProductionError("positive bounded selection count required")
        if operation == "mmr":
            if (type(lambda_mult) not in (int, float) or not isfinite(lambda_mult)
                    or not 0 <= lambda_mult <= 1):
                raise EvidenceProductionError("finite MMR blend in [0, 1] required")
            if q_vec is not None and (type(q_vec) is not list or any(
                    type(v) not in (int, float) or not isfinite(v) for v in q_vec)):
                raise EvidenceProductionError("finite explicit query vector required")
        inputs = session.records(request=request, handles=flat, current_allowed_sources=before)
        if any(item.layout != "retrieval_candidate" for item in inputs):
            raise EvidenceProductionError("page-to-candidate conversion must be explicit upstream")
        values, offset = [], 0
        for group in groups:
            values.append([item.record for item in inputs[offset:offset + len(group)]])
            offset += len(group)
        # Never let a collaborator rewrite the expected records used below.
        supplied = deepcopy(values)
        traces = []
        try:
            if operation == "score":
                result = invoke(ranking.v13_score_candidates, request, q, supplied[0],
                                runtime=runtime, lineage=traces.append)
            elif operation == "mmr":
                result = invoke(ranking.mmr_select, request, deepcopy(q_vec or []), supplied[0],
                                top_k, lambda_mult, runtime=runtime, lineage=traces.append)
            elif operation == "rerank":
                result = invoke(ranking.llm_rerank_citations, request, q, supplied[0], top_k,
                                diagnostic_mode, runtime=runtime, lineage=traces.append)
            else:
                result = invoke(ranking.promote_structured_rescue_hits, request,
                                supplied[0], supplied[1], top_k, runtime=runtime,
                                lineage=traces.append)
        except EvidenceContractError:
            raise
        except Exception:
            raise EvidenceProductionError("evidence transform failed") from None
        if (type(result) is not list or len(result) > limits.assembly.max_occurrences
                or len(traces) != 1 or type(traces[0]) is not tuple
                or len(traces[0]) != len(result)):
            raise EvidenceProductionError("missing/inconsistent direct transform lineage")
        selected, views = [], []
        for record, parents in zip(result, traces[0]):
            if (type(parents) is not tuple or len(parents) != 1
                    or type(parents[0]) is not tuple or len(parents[0]) != 2
                    or any(type(index) is not int for index in parents[0])):
                raise EvidenceProductionError("one exact transform input occurrence required")
            group, index = parents[0]
            if not 0 <= group < len(groups) or not 0 <= index < len(groups[group]):
                raise EvidenceProductionError("transform lineage references an absent input")
            handle = groups[group][index]
            original = values[group][index]
            if operation == "score":
                if type(record) is not dict:
                    raise EvidenceProductionError("score view must be an explicit candidate")
                views.append(((handle,), record))
            else:
                expected = (str(original.get("citation_id") or "").strip()
                            if operation == "rerank" else original)
                if not _same_value(record, expected):
                    raise EvidenceProductionError("selection changed an input without derivation")
                selected.append(handle)
        after = invoke(authorize, request, request)
        session.records(request=request, handles=flat, current_allowed_sources=after)
        if operation != "score":
            return tuple(selected)
        return session.derive_batch(request=request, views=tuple(views),
                                    operation=TRANSFORM_VERSION + ":score",
                                    current_allowed_sources=after)

    return invoke(work, request)
