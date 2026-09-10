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
from dataclasses import replace
from math import isfinite
from typing import Any, Callable

from . import candidate_ranking as ranking
from .ask_composition import AskEvidenceSession, RecordHandle, RegisteredRead, AskSelection, _check_derived_locator
from . import evidence_orchestration as orchestration
from ..evidence.legacy_compatibility import _Freezer
from ..evidence.contracts import canonical_json
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


PREPARATION_VERSION = "ask-prepare-lineage-p6b4i-v1"


class _PreparationTrace:
    """Bounded, operation-local journal, NOT another evidence session or ACL.

    Identity maps only objects materialized from explicit session handles or
    copies reported at their creation sites. Strong references prevent ID reuse.
    Snapshots are independent of collaborator-owned mutable dictionaries.
    Only terminal views are committed to the existing session, in one batch.
    """

    def __init__(self, *, limits, max_records, max_bytes):
        if (type(max_records) is not int or max_records < 1
                or type(max_bytes) is not int or max_bytes < 1):
            raise EvidenceProductionError("explicit positive preparation trace limits required")
        self.limits = limits
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.entries = {}
        self.roots = {}
        self.bytes = 0
        self.active = True
        self.fault = None
        self.copy_count = None
        self.scored = []
        self.scoring_complete = False

    def check(self):
        if not self.active:
            raise EvidenceProductionError("preparation callback expired")
        if self.fault is not None:
            raise self.fault

    def run(self, callback, *args, **kwargs):
        self.check()
        try:
            return callback(*args, **kwargs)
        except EvidenceContractError as exc:
            self.fault = exc
            raise
        except Exception:
            self.fault = EvidenceProductionError("evidence preparation collaborator failed")
            raise self.fault from None
        finally:
            self.check()

    def entry(self, record):
        self.check()
        item = self.entries.get(id(record))
        if item is None or record is not item[0] or not _same_value(record, item[1]):
            raise EvidenceProductionError("untracked or altered preparation occurrence")
        return item

    def add(self, record, parents, *, context=None):
        self.check()
        if type(record) is not dict or type(parents) is not tuple or not parents:
            raise EvidenceProductionError("explicit preparation record and origins required")
        if len(parents) > self.limits.assembly.max_occurrences:
            raise EvidenceProductionError("too many preparation origins")
        if id(record) in self.entries:
            raise EvidenceProductionError("preparation occurrence already registered")
        if len(self.entries) >= self.max_records:
            raise EvidenceProductionError("preparation trace record limit exceeded")
        size = len(canonical_json(_Freezer(self.limits.legacy).freeze(record)).encode("utf-8"))
        if self.bytes + size > self.max_bytes:
            raise EvidenceProductionError("preparation trace byte limit exceeded")
        contexts = tuple(self.roots[h] for h in parents)
        source = contexts[0].context.source
        if any(value.context.source != source for value in contexts):
            raise EvidenceProductionError("preparation cannot merge different sources")
        ctx = context or contexts[0].context
        evidence = adapt_candidate(record, context=ctx, limits=self.limits.adapter).entry.evidence
        parent_evidence = tuple(adapt_candidate(value.record, context=value.context,
            limits=self.limits.adapter).entry.evidence for value in contexts)
        _check_derived_locator(evidence.locator, parent_evidence)
        self.entries[id(record)] = (record, deepcopy(record), parents)
        self.bytes += size

    def materialize(self, handles, inputs):
        if len(handles) != len(inputs):
            raise EvidenceProductionError("preparation source length mismatch")
        if len(set(self.roots).union(handles)) > self.limits.assembly.max_occurrences:
            raise EvidenceProductionError("too many consumed preparation handles")
        out = []
        for h, item in zip(handles, inputs):
            if item.layout != "retrieval_candidate":
                raise EvidenceProductionError("page conversion must precede preparation")
            self.roots[h] = item
            value = deepcopy(item.record)
            self.add(value, (h,), context=item.context)
            out.append(value)
        return out

    def copies(self, stage, pairs):
        self.check()
        if stage not in {"copy", "score", "scored"} or type(pairs) is not tuple:
            raise EvidenceProductionError("unknown preparation creation event")
        if stage == "scored":
            if (self.copy_count is None or self.scoring_complete
                    or len(pairs) != len(self.scored)
                    or any(a is not b for a, b in zip(pairs, self.scored))):
                raise EvidenceProductionError("incomplete preparation scoring lineage")
            for value in pairs:
                self.entry(value)
            self.scoring_complete = True
            return
        if stage == "copy":
            if self.copy_count is not None:
                raise EvidenceProductionError("duplicate preparation input event")
            self.copy_count = len(pairs)
        elif self.copy_count is None or self.scoring_complete:
            raise EvidenceProductionError("out-of-order preparation score event")
        for pair in pairs:
            if type(pair) is not tuple or len(pair) != 2:
                raise EvidenceProductionError("explicit source/view pair required")
            source, view = pair
            entry = self.entry(source)
            if stage == "copy" and not _same_value(source, view):
                raise EvidenceProductionError("copy changed evidence without transformation")
            self.add(view, entry[2])
            if stage == "score":
                self.scored.append(view)

    def traced(self, callback, groups, *args, identical=False, **kwargs):
        # Only operator-reported input POSITIONS can give a new object origins.
        originals = [[self.entry(record) for record in group] for group in groups]
        traces = []
        result = callback(*args, lineage=traces.append, **kwargs)
        if (type(result) is not list or len(traces) != 1
                or type(traces[0]) is not tuple or len(traces[0]) != len(result)):
            raise EvidenceProductionError("missing preparation operator lineage")
        for record, positions in zip(result, traces[0]):
            if type(positions) is not tuple or not positions:
                raise EvidenceProductionError("explicit preparation parent positions required")
            parents = []
            for pos in positions:
                if (type(pos) is not tuple or len(pos) != 2
                        or any(type(v) is not int for v in pos)
                        or not 0 <= pos[0] < len(groups)
                        or not 0 <= pos[1] < len(groups[pos[0]])):
                    raise EvidenceProductionError("invalid preparation parent position")
                entry = originals[pos[0]][pos[1]]
                self.entry(entry[0])
                if len(parents) + len(entry[2]) > self.limits.assembly.max_occurrences:
                    raise EvidenceProductionError("too many preparation merge origins")
                parents.extend(entry[2])
            if identical and (len(positions) != 1 or not _same_value(record, entry[1])):
                raise EvidenceProductionError("catalog projection changed record")
            self.add(record, tuple(parents))
        return result

    def selected(self, callback, records, *args, **kwargs):
        entries = [self.entry(record) for record in records]
        result = callback(records, *args, **kwargs)
        if type(result) is not list:
            raise EvidenceProductionError("preparation selector must return a list")
        admitted = {id(item[0]) for item in entries}
        for record in result:
            self.entry(record)
            if id(record) not in admitted:
                raise EvidenceProductionError("selector introduced a different occurrence")
        for item in entries:
            self.entry(item[0])
        return result

    def verify(self):
        if self.copy_count is None or (self.copy_count and not self.scoring_complete):
            raise EvidenceProductionError("preparation creation trace incomplete")
        for entry in self.entries.values():
            self.entry(entry[0])


def prepare_records(*, request: Any, session: AskEvidenceSession,
                    retrieval: dict, selections: tuple[AskSelection, ...], decision: Any,
                    runtime: orchestration.AssistantCorePrepareEvidenceRuntime,
                    source_callbacks: dict[str, Callable[..., tuple[RecordHandle, ...]]],
                    authorize: Callable[[Any], frozenset[SourceIdentity]],
                    invoke: Callable[..., Any], max_trace_records: int,
                    max_trace_bytes: int) -> tuple[dict, tuple[AskSelection, ...]]:
    """Run the REAL P4 ASK preparation, retaining its original policy/heuristics.

    The caller supplies the existing request session/owner, CURRENT authority,
    exact input collection handles, explicit trace capacities, and P4 runtime.
    This is not a new router/Core/ACL service. No model, SQL or embedding is added.

    Three optional source_callbacks use the corresponding original signatures:
    catalog, neighbors, sections. If a reached branch lacks its adapter, it fails
    technically, never silently skips a source. Each adapter must perform its
    authentic existing read/conversion and return explicit session handles, not
    raw records or grants inferred from IDs/text. Earlier acquired reads remain
    in the session on failure; no terminal derived views are committed then.

    Runtime merge and catalog-projection callbacks MUST accept `lineage` and
    bind the existing traced v13_merge_candidates/overview_catalog_candidates
    operators. The three selectors must return selected original objects. Metric
    callbacks must not mutate evidence. Fresh copies require creation-site trace.
    Main's legacy wrappers do NOT satisfy this binding automatically; caller must
    explicitly compose these dependencies before activation. No fallback runtime.

    Metadata outside candidates/citations is preserved, NOT made authorized text
    by this operation. Catalog digest derives from separately authorized inputs,
    but output semantics and links still need validation. Only ASK->ASK is
    supported; RC/Smart remain in their unchanged legacy path. This does not wire
    production reads, page conversion, current ACL, other callbacks, direct/facts,
    post-Core rescue, cache or main. No claim of full canonical ASK activation.

    Trace capacities bound records/serialized snapshots, not heap, model tokens
    or dollars. Authority callbacks have real costs to measure. Collaborators are
    trusted, non-reentrant for this operation; no Python sandbox or atomic remote
    revocation is promised. Caller owns/ends the same session lifetime.
    """
    _callbacks(session, authorize, invoke)

    def work():
        if (getattr(request, "requested_mode", None) != "ask"
                or getattr(decision, "effective_mode", None) != "ask"):
            raise EvidenceProductionError("preparation lineage supports ASK/ASK only")
        if type(runtime) is not orchestration.AssistantCorePrepareEvidenceRuntime:
            raise EvidenceProductionError("existing explicit P4 preparation runtime required")
        if (type(source_callbacks) is not dict
                or any(k not in {"catalog", "neighbors", "sections"}
                       or not callable(v) for k, v in source_callbacks.items())):
            raise EvidenceProductionError("explicit preparation source adapters required")
        before = invoke(authorize, request, request)
        _, limits = session.read_contract(request=request, current_allowed_sources=before)
        journal = _PreparationTrace(limits=limits, max_records=max_trace_records,
                                    max_bytes=max_trace_bytes)
        # P6-A exact collection check BEFORE providing data to a collaborator.
        session.admission(request=request, retrieval=retrieval, selections=selections,
                          current_allowed_sources=before)
        supplied = deepcopy(retrieval)
        try:
            for selection in selections:
                inputs = session.records(request=request, handles=selection.records,
                                         current_allowed_sources=before)
                values = journal.materialize(selection.records, inputs)
                supplied[selection.name] = values if selection.container_kind == "list" else tuple(values)

            def call(fn, *args, **kwargs):
                return journal.run(invoke, fn, request, *args, **kwargs)

            def source(name, *args, **kwargs):
                def read():
                    if name not in source_callbacks:
                        raise EvidenceProductionError("required preparation source adapter missing")
                    current = invoke(authorize, request, request)
                    session.read_contract(request=request, current_allowed_sources=current)
                    session.records(request=request, handles=tuple(journal.roots),
                                    current_allowed_sources=current)
                    handles = invoke(source_callbacks[name], request, *args, **kwargs)
                    if (type(handles) is not tuple or len(handles) > limits.assembly.max_occurrences
                            or any(type(h) is not RecordHandle for h in handles)):
                        raise EvidenceProductionError("source adapter must return bounded explicit handles")
                    after = invoke(authorize, request, request)
                    inputs = session.records(request=request, handles=handles,
                                             current_allowed_sources=after)
                    return journal.materialize(handles, inputs)
                return journal.run(read)

            def assurance(req, data, route):
                original = {name: tuple(data.get(name) or ()) for name in ("candidates", "citations")}
                result = call(runtime._assistant_core_root_diagnostic_evidence_assurance,
                              req, data, route)
                # ASK assurance must not silently introduce unregistered copies.
                for name in ("candidates", "citations"):
                    records = result.get(name) or []
                    if len(records) != len(original[name]) or any(
                            a is not b for a, b in zip(records, original[name])):
                        raise EvidenceProductionError("ASK assurance changed its registered collections")
                    for record in records:
                        journal.entry(record)
                return result

            def merge(groups):
                return journal.run(journal.traced,
                    lambda *a, **kw: call(runtime._v13_merge_candidates, *a, **kw),
                    groups, groups)

            def overview(records):
                return journal.run(journal.traced,
                    lambda *a, **kw: call(runtime._assistant_core_overview_catalog_candidates, *a, **kw),
                    [records], records, identical=True)

            def selector(fn):
                def selected(records, *args, **kwargs):
                    return journal.run(journal.selected,
                        lambda *a, **kw: call(fn, *a, **kw), records, *args, **kwargs)
                return selected

            def readonly(fn):
                def observed(*args, **kwargs):
                    snapshots = [(value, deepcopy(value)) for value in (*args, *kwargs.values())
                                 if type(value) in (dict, list, tuple)]
                    result = call(fn, *args, **kwargs)
                    if any(not _same_value(value, expected) for value, expected in snapshots):
                        raise EvidenceProductionError("read-only preparation callback changed its input")
                    return result
                return observed

            read_only_names = (
                "_assistant_core_candidate_source_type", "_v13_candidate_text",
                "_assistant_core_candidate_facet_metrics", "_assistant_core_diagnostic_priority_metrics",
                "_assistant_core_ps_is_substantive", "_v13_evidence_metrics",
                "_assistant_core_machine_catalog_digest", "_v13_deterministic_evidence_state")
            bound = replace(runtime,
                **{name: readonly(getattr(runtime, name)) for name in read_only_names},
                _assistant_core_root_diagnostic_evidence_assurance=assurance,
                _assistant_core_machine_catalog_candidates=lambda *a, **kw: source("catalog", *a, **kw),
                _v13_assurance_fetch_neighbor_pages=lambda *a, **kw: source("neighbors", *a, **kw),
                _assistant_core_expand_enumeration_sections=lambda *a, **kw: source("sections", *a, **kw),
                _v13_merge_candidates=merge,
                _assistant_core_overview_catalog_candidates=overview,
                _dedup_citations_by_snippet=selector(runtime._dedup_citations_by_snippet),
                _assistant_core_facet_balanced_pool=selector(runtime._assistant_core_facet_balanced_pool),
                _assistant_core_source_diversity_pool=selector(runtime._assistant_core_source_diversity_pool))
            result = journal.run(invoke, orchestration.assistant_core_prepare_evidence,
                request, request, supplied, decision, runtime=bound,
                lineage=lambda *a: journal.run(journal.copies, *a))
            journal.verify()  # Includes consumed records no longer in the result.
            after = invoke(authorize, request, request)
            roots = tuple(journal.roots)
            if len(roots) > limits.assembly.max_occurrences:
                raise EvidenceProductionError("too many consumed preparation handles")
            session.records(request=request, handles=roots, current_allowed_sources=after)
            output = result.get("retrieval") if type(result) is dict else None
            if (type(output) is not dict or type(result.get("supported")) is not bool
                    or type(output.get("candidates")) is not list
                    or type(output.get("citations")) is not list):
                raise EvidenceProductionError("P4 preparation output contract changed")
            values = output["candidates"]
            if (not _same_value(values, output["citations"])
                    or any(a is not b for a, b in zip(values, output["citations"]))
                    or 2 * len(values) > limits.assembly.max_occurrences):
                raise EvidenceProductionError("prepared collections lack one exact selection")
            views = tuple((journal.entry(record)[2], deepcopy(record)) for record in values)
            # All final source/locator/relationship checks remain in the SAME session.
            handles = session.derive_batch(request=request, views=views,
                operation=PREPARATION_VERSION, current_allowed_sources=after)
            return result, (AskSelection("candidates", handles), AskSelection("citations", handles))
        finally:
            journal.active = False
            journal.entries.clear()
            journal.roots.clear()
            journal.scored.clear()

    return invoke(work, request)


INITIAL_LINEAGE_VERSION = "ask-initial-retrieval-lineage-p6b4j-v1"


def retrieve_initial_records(*, request: Any, session: AskEvidenceSession,
        q: str, mode: str, runtime: orchestration.V13InitialRetrievalRuntime,
        source_callbacks: dict[str, Callable[..., Any]],
        authorize: Callable[[Any], frozenset[SourceIdentity]],
        invoke: Callable[..., Any], max_trace_records: int, max_trace_bytes: int,
        plan: dict | None = None) -> tuple[dict, tuple[AskSelection, ...]]:
    """Run the EXISTING initial P4 policy with explicit registered source adapters.

    Only an ASK-owned request is accepted; P4 mode may be ask or neutral. Scope
    comes from that request's session, never from candidates or query text.
    Eight source adapters have the legacy callback signatures: dense, prefix,
    lexical, identifier, pages, preferred, structured_dense, structured_direct.
    Dense returns (debug_count_or_None, tuple_of_handles); the others return a
    tuple of handles. They must actually acquire/register the corresponding
    authorized receipts, using acquire_read and the SAME owner/session. Raw rows,
    guessed source identities or read_sources used as grants are not adapters.

    The dense two-callback boundary uses a private single-use token carrying
    those explicit occurrences, not a fabricated SQL tuple. Its historical
    query_used annotation is a recorded view. FTS annotations are observed at
    their mutation site in P4. RRF/merge/score callbacks must forward lineage to
    the existing traced operators; nested selectors must preserve object origin.
    Read-only callbacks cannot silently mutate input evidence or a query plan.

    The operation-local bounded journal from B4i is reused, not another evidence
    session, ACL service, router or algorithm. Output candidates are committed
    in one existing derive_batch; citations retain their exact prefix positions.
    Already acquired receipts survive a later failure; caller ends the session.
    Every consumed handle is authorized again before export, including dropped
    records. Errors in called collaborators latch despite legacy broad catches;
    they are technical failures rather than successful empty retrievals.

    Main remains unconfigured. This does not automatically implement adapter
    bodies, current production authority, raw-page conversion, neutral/refine
    outer transformations, hidden reads, direct facts/rescue, cache or links.
    Metadata outside evidence collections is preserved, not authorized text.
    No new query/model/embedding is introduced; authority callback I/O and trace
    CPU/heap overhead must be measured. Bounds are serialized state, not heap.
    Collaborators are trusted/non-reentrant for this operation: no Python sandbox
    or atomic remote revocation guarantee is claimed.
    """
    _callbacks(session, authorize, invoke)

    def work():
        names = {"dense", "prefix", "lexical", "identifier", "pages", "preferred",
                 "structured_dense", "structured_direct"}
        if (getattr(request, "requested_mode", None) != "ask"
                or type(mode) is not str or mode not in {"ask", "neutral"}):
            raise EvidenceProductionError("initial lineage supports ASK-owned ask/neutral only")
        if type(q) is not str or not q.strip() or (plan is not None and type(plan) is not dict):
            raise EvidenceProductionError("explicit initial query and optional plan required")
        if type(runtime) is not orchestration.V13InitialRetrievalRuntime:
            raise EvidenceProductionError("existing initial retrieval runtime required")
        if (type(source_callbacks) is not dict or any(
                name not in names or not callable(fn) for name, fn in source_callbacks.items())):
            raise EvidenceProductionError("explicit initial source adapters required")
        before = invoke(authorize, request, request)
        scope, limits = session.read_contract(request=request, current_allowed_sources=before)
        cap = runtime.V13_MAX_EVIDENCE_ITEMS_ASK
        if type(cap) is not int or not 1 <= cap <= limits.assembly.max_occurrences:
            raise EvidenceProductionError("bounded initial citation capacity required")
        journal = _PreparationTrace(limits=limits, max_records=max_trace_records,
                                    max_bytes=max_trace_bytes)
        dense_tokens = {}
        flags = {}
        score_result = None
        complete = False
        supplied_plan = deepcopy(plan)
        try:
            def call(fn, *args, **kwargs):
                return journal.run(invoke, fn, request, *args, **kwargs)

            def readonly(fn):
                def observed(*args, **kwargs):
                    snapshots = [(value, deepcopy(value)) for value in (*args, *kwargs.values())
                                 if type(value) in (dict, list, tuple)]
                    result = call(fn, *args, **kwargs)
                    if any(not _same_value(value, expected) for value, expected in snapshots):
                        raise EvidenceProductionError("read-only initial callback changed its input")
                    return result
                return lambda *a, **kw: journal.run(observed, *a, **kw)

            def source(name, **kwargs):
                def read():
                    if name not in source_callbacks:
                        raise EvidenceProductionError("required initial source adapter missing")
                    current = invoke(authorize, request, request)
                    session.read_contract(request=request, current_allowed_sources=current)
                    session.records(request=request, handles=tuple(journal.roots),
                                    current_allowed_sources=current)
                    result = readonly(source_callbacks[name])(**kwargs)
                    count = None
                    if name == "dense":
                        if (type(result) is not tuple or len(result) != 2
                                or (result[0] is not None and
                                    (type(result[0]) is not int or result[0] < 0))):
                            raise EvidenceProductionError("dense adapter requires count and handles")
                        count, result = result
                    if (type(result) is not tuple or len(result) > limits.assembly.max_occurrences
                            or any(type(h) is not RecordHandle for h in result)):
                        raise EvidenceProductionError("initial source adapter requires bounded explicit handles")
                    after = invoke(authorize, request, request)
                    session.records(request=request, handles=tuple(journal.roots),
                                    current_allowed_sources=after)
                    inputs = session.records(request=request, handles=result,
                                             current_allowed_sources=after)
                    records = journal.materialize(result, inputs)
                    if name == "dense":
                        token = object()
                        dense_tokens[token] = records
                        return count, token
                    return records
                return journal.run(read)

            def dense_candidates(token, *, query_used=None):
                def convert():
                    if token not in dense_tokens or type(query_used) is not str:
                        raise EvidenceProductionError("dense conversion requires this call's one-use token")
                    originals = dense_tokens.pop(token)
                    result = []
                    for original in originals:
                        entry = journal.entry(original)
                        value = dict(original)
                        value["query_used"] = query_used
                        journal.add(value, entry[2])
                        result.append(value)
                    return result
                return journal.run(convert)

            def traced(fn, groups, *args, same_locator=False, **kwargs):
                def op(*a, **kw):
                    emit = kw["lineage"]
                    def trace(positions):
                        if type(positions) is not tuple:
                            raise EvidenceProductionError("explicit initial operator positions required")
                        for contributors in positions:
                            if (type(contributors) is not tuple or not contributors
                                    or any(type(pos) is not tuple or len(pos) != 2
                                        or type(pos[0]) is not int or type(pos[1]) is not int
                                        or not 0 <= pos[0] < len(groups)
                                        or not 0 <= pos[1] < len(groups[pos[0]])
                                        for pos in contributors)):
                                raise EvidenceProductionError("invalid initial operator positions")
                        if same_locator:
                            # V13 field-wise merge cannot take text from another
                            # location merely because its legacy CID collides.
                            for contributors in positions:
                                locators = []
                                for group, index in contributors:
                                    entry = journal.entry(groups[group][index])
                                    context = journal.roots[entry[2][0]].context
                                    locators.append(adapt_candidate(entry[1], context=context,
                                        limits=limits.adapter).entry.evidence.locator)
                                if locators and any(loc != locators[0] for loc in locators[1:]):
                                    raise EvidenceProductionError("initial merge combines different locators")
                        emit(positions)
                    kw["lineage"] = trace
                    return call(fn, *a, **kw)
                return journal.run(journal.traced, op, groups, *args, **kwargs)

            def scored(query, records):
                nonlocal score_result
                if score_result is not None:
                    raise EvidenceProductionError("duplicate initial scoring boundary")
                result = traced(runtime._v13_score_candidates, [records], query, records)
                score_result = tuple(result)
                return result

            def creation(stage, records):
                nonlocal complete
                def record_event():
                    nonlocal complete
                    if type(records) is not tuple or len(records) > limits.assembly.max_occurrences:
                        raise EvidenceProductionError("bounded initial creation event required")
                    if stage == "complete":
                        if (complete or score_result is None or dense_tokens
                                or set(flags) != {"prefix_done", "lexical_done"}
                                or len(records) != len(score_result)
                                or any(a is not b for a, b in zip(records, score_result))):
                            raise EvidenceProductionError("initial creation trace incomplete")
                        for record in records:
                            journal.entry(record)
                        complete = True
                        return
                    if stage not in {"prefix_before", "prefix_after", "lexical_before", "lexical_after"}:
                        raise EvidenceProductionError("unknown initial creation event")
                    name, moment = stage.rsplit("_", 1)
                    if moment == "before":
                        if name in flags or name + "_done" in flags:
                            raise EvidenceProductionError("duplicate initial flag input event")
                        flags[name] = tuple(journal.entry(record) for record in records)
                    else:
                        if name not in flags:
                            raise EvidenceProductionError("initial flag output without origin")
                        entries = flags.pop(name)
                        if len(entries) != len(records):
                            raise EvidenceProductionError("initial flag occurrence count changed")
                        for entry, record in zip(entries, records):
                            expected = dict(entry[1]); expected["fts_v13"] = True
                            if record is not entry[0] or not _same_value(record, expected):
                                raise EvidenceProductionError("initial flag changed unrecorded fields")
                            # Update only this operation-local view. The session's
                            # immutable SQL observation and original view are intact.
                            del journal.entries[id(record)]
                            journal.bytes -= len(canonical_json(
                                _Freezer(limits.legacy).freeze(entry[1])).encode("utf-8"))
                            journal.add(record, entry[2])
                        flags[name + "_done"] = True
                return journal.run(record_event)

            reads = {"_fetch_dense_chunk_candidates": "dense", "_fts_search_chunks_prefix": "prefix",
                     "_fts_search_chunks_multi": "lexical", "_v13_exact_identifier_candidates": "identifier",
                     "_v13_fetch_scored_pages": "pages", "_v13_fetch_preferred_source_pages": "preferred",
                     "_v13_fetch_structured_dense_candidates": "structured_dense",
                     "_ask_structured_direct_fetch_sources": "structured_direct"}
            replacements = {name: (lambda key: lambda **kw: source(key, **kw))(key)
                            for name, key in reads.items()}
            for name in ("_v13_current_budget", "_v13_fallback_plan", "_dedup_text_values",
                         "_openai_embed_texts", "_vector_literal", "_ask_source_preference_profile",
                         "_count_query_tokens", "_v13_build_profile_from_plan",
                         "_structured_rescue_query_intent", "_v13_evidence_metrics"):
                replacements[name] = readonly(getattr(runtime, name))
            bound = replace(runtime, **replacements,
                _raw_rows_to_dense_candidates=dense_candidates,
                _rrf_merge_candidates=lambda groups, **kw: traced(
                    runtime._rrf_merge_candidates, groups, groups, **kw),
                _v13_merge_candidates=lambda groups: traced(
                    runtime._v13_merge_candidates, groups, groups, same_locator=True),
                _v13_score_candidates=lambda *a: journal.run(scored, *a))
            result = journal.run(invoke, orchestration.v13_initial_retrieval, request,
                q=q, mode=mode, response_language=request.response_language,
                ai_scope=scope.ai_scope, **scope.sql_selectors(), plan=supplied_plan,
                runtime=bound, lineage=creation)
            journal.check()
            if not complete:
                raise EvidenceProductionError("initial retrieval did not close its creation trace")
            for entry in journal.entries.values():
                journal.entry(entry[0])
            if (type(result) is not dict or type(result.get("candidates")) is not list
                    or type(result.get("citations")) is not list
                    or len(result["candidates"]) != len(score_result)
                    or any(a is not b for a, b in zip(result["candidates"], score_result))
                    or len(result["citations"]) != len(score_result[:cap])
                    or any(a is not b for a, b in zip(result["citations"], score_result[:cap]))):
                raise EvidenceProductionError("initial output differs from exact scored selection")
            if len(score_result) + len(result["citations"]) > limits.assembly.max_occurrences:
                raise EvidenceProductionError("aggregate initial output count exceeded")
            after = invoke(authorize, request, request)
            session.records(request=request, handles=tuple(journal.roots), current_allowed_sources=after)
            views = tuple((journal.entry(record)[2], deepcopy(record)) for record in score_result)
            result_handles = session.derive_batch(request=request, views=views,
                operation=INITIAL_LINEAGE_VERSION, current_allowed_sources=after)
            return result, (AskSelection("candidates", result_handles),
                            AskSelection("citations", result_handles[:cap]))
        finally:
            journal.active = False
            journal.entries.clear(); journal.roots.clear(); dense_tokens.clear(); flags.clear()

    return invoke(work, request)


OUTER_RETRIEVAL_LINEAGE_VERSION = "ask-neutral-refinement-lineage-p6b4k-v1"


class _OuterRetrievalTrace(_PreparationTrace):
    """Reuse the bounded B4i journal for ONE neutral/refinement operation.

    This is not a second evidence session. Only explicit input handles and
    creation-site events populate object identity; keys/scores never grant scope.
    Collaborators are trusted; callbacks expire even when legacy catches errors.
    """

    def __init__(self, *, request, session, authorize, invoke, max_records, max_bytes):
        self.request, self.session = request, session
        self.authorize, self.invoke = authorize, invoke
        allowed = invoke(authorize, request, request)
        self.scope, limits = session.read_contract(request=request, current_allowed_sources=allowed)
        super().__init__(limits=limits, max_records=max_records, max_bytes=max_bytes)
        self.pending = {}
        self.completed = None
        self.bonus_done = False
        self.initial_calls = 0
        self.title_calls = 0

    def call(self, fn, *args, **kwargs):
        return self.run(self.invoke, fn, self.request, *args, **kwargs)

    def current(self):
        allowed = self.call(self.authorize, self.request)
        self.session.read_contract(request=self.request, current_allowed_sources=allowed)
        self.session.records(request=self.request, handles=tuple(self.roots), current_allowed_sources=allowed)
        return allowed

    def snapshot(self, value):
        size = len(canonical_json(_Freezer(self.limits.legacy).freeze(value)).encode("utf-8"))
        if size > self.max_bytes:
            raise EvidenceProductionError("outer retrieval snapshot byte limit exceeded")
        return deepcopy(value)

    def readonly(self, fn):
        def read(*args, **kwargs):
            copies = [(v, self.snapshot(v)) for v in (*args, *kwargs.values())
                      if type(v) in (dict, list, tuple)]
            result = self.call(fn, *args, **kwargs)
            if any(not _same_value(v, saved) for v, saved in copies):
                raise EvidenceProductionError("read-only outer retrieval callback changed input")
            return result
        return lambda *a, **kw: self.run(read, *a, **kw)

    def import_pack(self, result, selections):
        allowed = self.current()
        self.session.admission(request=self.request, retrieval=result, selections=selections,
                               current_allowed_sources=allowed)
        if (type(result) is not dict or type(selections) is not tuple
                or len(selections) != 2 or {s.name for s in selections} != {"candidates", "citations"}
                or any(s.container_kind != "list" for s in selections)):
            raise EvidenceProductionError("explicit list candidate/citation selections required")
        if sum(len(s.records) for s in selections) > self.limits.assembly.max_occurrences:
            raise EvidenceProductionError("outer retrieval input occurrence limit exceeded")
        supplied = self.snapshot(result)
        # Reuse only an explicitly selected HANDLE, never a content/ID match.
        objects = {}
        for selection in selections:
            values = self.session.records(request=self.request, handles=selection.records,
                                          current_allowed_sources=allowed)
            items = []
            for handle, value in zip(selection.records, values):
                if handle not in objects:
                    objects[handle] = self.materialize((handle,), (value,))[0]
                items.append(objects[handle])
            supplied[selection.name] = items
        return supplied

    def selectors(self, kwargs):
        expected = {**self.scope.sql_selectors(), "ai_scope": self.scope.ai_scope}
        if any(k not in kwargs or not _same_value(kwargs[k], v) for k, v in expected.items()):
            raise EvidenceProductionError("outer source adapter scope selectors differ")
        if "response_language" in kwargs and kwargs["response_language"] != self.request.response_language:
            raise EvidenceProductionError("outer initial response language differs")

    def initial(self, adapter, **kwargs):
        def read():
            self.current()
            self.selectors(kwargs)
            if not callable(adapter):
                raise EvidenceProductionError("required initial retrieval adapter missing")
            self.initial_calls += 1
            pair = self.readonly(adapter)(**kwargs)
            if type(pair) is not tuple or len(pair) != 2:
                raise EvidenceProductionError("initial adapter must return retrieval and selections")
            return self.import_pack(*pair)
        return self.run(read)

    def titles(self, adapter, **kwargs):
        def read():
            self.current()
            self.selectors(kwargs)
            if not callable(adapter):
                raise EvidenceProductionError("required title receipt adapter missing")
            self.title_calls += 1
            hs = self.readonly(adapter)(**kwargs)
            if (type(hs) is not tuple or len(hs) > self.limits.assembly.max_occurrences
                    or any(type(h) is not RecordHandle for h in hs)):
                raise EvidenceProductionError("title adapter requires bounded explicit handles")
            allowed = self.current()
            values = self.session.records(request=self.request, handles=hs, current_allowed_sources=allowed)
            return self.materialize(hs, values)
        return self.run(read)

    def operator(self, fn, groups, *args, same_locator=False, **kwargs):
        def operation(*a, **kw):
            emit = kw["lineage"]
            def traced(positions):
                if type(positions) is not tuple:
                    raise EvidenceProductionError("explicit outer operator positions required")
                for parents in positions:
                    if (type(parents) is not tuple or not parents
                            or any(type(p) is not tuple or len(p) != 2
                                or type(p[0]) is not int or type(p[1]) is not int
                                or not 0 <= p[0] < len(groups)
                                or not 0 <= p[1] < len(groups[p[0]]) for p in parents)):
                        raise EvidenceProductionError("invalid outer operator position")
                    if same_locator:
                        locations = []
                        for group, index in parents:
                            item = self.entry(groups[group][index])
                            context = self.roots[item[2][0]].context
                            locations.append(adapt_candidate(item[1], context=context,
                                limits=self.limits.adapter).entry.evidence.locator)
                        if any(loc != locations[0] for loc in locations[1:]):
                            raise EvidenceProductionError("outer merge combines different locators")
                emit(positions)
            kw["lineage"] = traced
            return self.call(fn, *a, **kw)
        return self.run(self.traced, operation, groups, *args, **kwargs)

    def mutation_before(self, tag, values):
        if tag in self.pending or type(values) is not tuple:
            raise EvidenceProductionError("duplicate or invalid mutation boundary")
        self.pending[tag] = tuple(self.entry(v) for v in values)

    def mutation_after(self, tag, values, allowed_fields):
        if tag not in self.pending or type(values) is not tuple:
            raise EvidenceProductionError("mutation without input boundary")
        entries = self.pending.pop(tag)
        if len(entries) != len(values):
            raise EvidenceProductionError("mutation occurrence count changed")
        for item, record in zip(entries, values):
            if record is not item[0] or not allowed_fields.issubset(record):
                raise EvidenceProductionError("mutation lost its exact occurrence or fields")
            expected = dict(item[1])
            for key in record:
                if key in allowed_fields:
                    expected[key] = record[key]
            if not _same_value(record, expected):
                raise EvidenceProductionError("mutation changed unrecorded evidence fields")
            del self.entries[id(record)]
            self.bytes -= len(canonical_json(_Freezer(self.limits.legacy).freeze(item[1])).encode("utf-8"))
            self.add(record, item[2])

    def event(self, stage, *args):
        def accept():
            if self.completed is not None:
                raise EvidenceProductionError("event after outer retrieval completion")
            if stage == "copy":
                source, value = args
                entry = self.entry(source)
                if not _same_value(source, value):
                    raise EvidenceProductionError("outer copy changed input")
                self.add(value, entry[2])
                return value
            if stage == "annotation_before":
                source, value = args
                self.event("copy", source, value)
                self.mutation_before(("annotation", id(value)), (value,))
            elif stage == "annotation_after":
                value, = args
                self.mutation_after(("annotation", id(value)), (value,), frozenset({
                    "assistant_core_facet_hits", "assistant_core_facet_answer_types",
                    "assistant_core_facet_preferred_source_types", "assistant_core_facet_must_cover",
                    "assistant_core_facet_retrieval_score", "assistant_core_facet_score_map",
                    "assistant_core_facet_support"}))
            elif stage == "facet_merge":
                source, value, contributors = args
                entry = self.entry(source)
                if type(contributors) is not tuple:
                    raise EvidenceProductionError("explicit facet annotation contributors required")
                parents = list(entry[2])
                for origin in contributors:
                    parents.extend(self.entry(origin)[2])
                    if len(parents) > self.limits.assembly.max_occurrences:
                        raise EvidenceProductionError("too many facet annotation origins")
                self.add(value, tuple(parents))
            elif stage == "bonus_before":
                values, = args
                if self.bonus_done:
                    raise EvidenceProductionError("repeated bonus phase")
                self.mutation_before("bonus", values)
            elif stage == "bonus_after":
                values, = args
                self.mutation_after("bonus", values, frozenset({
                    "assistant_core_facet_retrieval_bonus", "v13_score", "retrieval_score"}))
                self.bonus_done = True
            elif stage in {"complete", "unchanged"}:
                candidates, citations = args
                if (self.pending or type(candidates) is not tuple or type(citations) is not tuple
                        or len(candidates) + len(citations) > self.limits.assembly.max_occurrences):
                    raise EvidenceProductionError("incomplete or oversized outer retrieval trace")
                for value in candidates + citations:
                    self.entry(value)
                self.completed = (stage, candidates, citations)
            else:
                raise EvidenceProductionError("unknown outer retrieval lineage event")
        return self.run(accept)

    def finish(self, result, *, preserve=None):
        self.check()
        if self.completed is None or self.pending or type(result) is not dict:
            raise EvidenceProductionError("outer retrieval did not close its trace")
        for name, expected in zip(("candidates", "citations"), self.completed[1:]):
            values = result.get(name)
            if (type(values) is not list or len(values) != len(expected)
                    or any(v is not e for v, e in zip(values, expected))):
                raise EvidenceProductionError("outer output differs from exact traced selection")
        for entry in self.entries.values():
            self.entry(entry[0])
        allowed = self.current()  # Includes inputs no longer selected.
        candidates, citations = self.completed[1:]
        positions = {}
        for index, value in enumerate(candidates):
            positions.setdefault(id(value), []).append(index)
        citation_positions = []
        for value in citations:
            slots = positions.get(id(value), [])
            if not slots:
                raise EvidenceProductionError("citation is not a selected candidate occurrence")
            citation_positions.append(slots.pop(0))
        if preserve is not None:
            self.session.admission(request=self.request, retrieval=result, selections=preserve,
                                   current_allowed_sources=allowed)
            return result, preserve
        views = tuple((self.entry(value)[2], deepcopy(value)) for value in candidates)
        out = self.session.derive_batch(request=self.request, views=views,
            operation=OUTER_RETRIEVAL_LINEAGE_VERSION, current_allowed_sources=allowed)
        return result, (AskSelection("candidates", out),
                        AskSelection("citations", tuple(out[index] for index in citation_positions)))

    def dispose(self):
        self.active = False
        self.entries.clear(); self.roots.clear(); self.pending.clear(); self.completed = None


def retrieve_neutral_records(*, request: Any, session: AskEvidenceSession,
        runtime: orchestration.AssistantCoreRetrieveNeutralRuntime,
        title_runtime: ranking.V13MergeSourceTitleCandidatesRuntime,
        initial_adapter: Callable[..., Any], title_adapter: Callable[..., tuple[RecordHandle, ...]],
        authorize: Callable[[Any], frozenset[SourceIdentity]], invoke: Callable[..., Any],
        max_trace_records: int, max_trace_bytes: int) -> tuple[dict, tuple[AskSelection, ...]]:
    """Execute the existing neutral ASK wrapper with explicit initial/title origins.

    initial_adapter(**legacy_kwargs) must call the bound B4j producer (or an
    equivalent explicit producer) and return (retrieval, AskSelection tuple).
    title_adapter returns authentic session handles, never rows/IDs as grants.
    title_runtime binds the existing title merge's collaborators; merge/scoring
    must forward lineage. Policies/model/SQL calls are those of the supplied P4
    implementations, not added by this helper. Authority I/O remains measurable.

    No production main wiring, page conversion or current authority is inferred.
    This is ASK-owned neutral acquisition, not RC/Smart migration. Noncollection
    metadata remains metadata, not authorized answer text. See refine_records
    for operation lifetime, rollback and limits shared by these producers.
    """
    _callbacks(session, authorize, invoke)
    def work():
        if (getattr(request, "requested_mode", None) != "ask"
                or type(runtime) is not orchestration.AssistantCoreRetrieveNeutralRuntime
                or type(title_runtime) is not ranking.V13MergeSourceTitleCandidatesRuntime):
            raise EvidenceProductionError("ASK neutral runtime and title runtime required")
        trace = _OuterRetrievalTrace(request=request, session=session, authorize=authorize,
            invoke=invoke, max_records=max_trace_records, max_bytes=max_trace_bytes)
        try:
            for cap in (title_runtime.V13_MAX_EVIDENCE_ITEMS_ASK,
                        title_runtime.V13_SOURCE_RETRIEVAL_MAX_CANDIDATES):
                if type(cap) is not int or not 1 <= cap <= trace.limits.assembly.max_occurrences:
                    raise EvidenceProductionError("bounded title retrieval capacities required")
            title_bound = replace(title_runtime,
                _v13_merge_candidates=lambda groups: trace.operator(
                    title_runtime._v13_merge_candidates, groups, groups, same_locator=True),
                _v13_score_candidates=lambda q, values: trace.operator(
                    title_runtime._v13_score_candidates, [values], q, values),
                _v13_evidence_metrics=trace.readonly(title_runtime._v13_evidence_metrics))
            bound = replace(runtime,
                _assistant_core_retrieval_query=trace.readonly(runtime._assistant_core_retrieval_query),
                _assistant_core_scope_value=trace.readonly(runtime._assistant_core_scope_value),
                _v13_fallback_plan=trace.readonly(runtime._v13_fallback_plan),
                _v13_initial_retrieval=lambda **kw: trace.initial(initial_adapter, **kw),
                _v13_fetch_structured_title_candidates=lambda **kw: trace.titles(title_adapter, **kw),
                _v13_merge_source_title_candidates=lambda q, pack, titles: trace.run(
                    ranking.v13_merge_source_title_candidates, q, pack, titles, runtime=title_bound))
            result = trace.call(orchestration.assistant_core_retrieve_neutral,
                                request, runtime=bound, lineage=trace.event)
            if trace.initial_calls != 1 or trace.title_calls != 1:
                raise EvidenceProductionError("neutral acquisition trace incomplete")
            return trace.finish(result)
        finally:
            trace.dispose()
    return invoke(work, request)


def refine_records(*, request: Any, session: AskEvidenceSession, retrieval: dict,
        selections: tuple[AskSelection, ...], decision: Any,
        runtime: orchestration.AssistantCoreRefineRetrievalRuntime,
        facet_runtime: ranking.AssistantCoreMergeFacetCandidatesRuntime,
        initial_adapter: Callable[..., Any],
        authorize: Callable[[Any], frozenset[SourceIdentity]], invoke: Callable[..., Any],
        max_trace_records: int, max_trace_bytes: int) -> tuple[dict, tuple[AskSelection, ...]]:
    """Execute the existing whole ASK refinement: combined or bounded facet runs.

    Same original budget checks, plans, facet confidence/bonus/order and caps;
    no new keywords, thresholds, planner/model/embedding or SQL. Confidence and
    metrics are read-only collaborators, not an authorization authority. Copies,
    annotations, merges and bonus changes report origins at their creation site.
    Semantic facet annotations are derived views, never observed measurements.

    initial_adapter follows retrieve_neutral_records's explicit output contract.
    Reached missing adapters and evidence errors are latched despite P4's broad
    facet catch. Scope and all consumed handles are rechecked with CURRENT grants.
    Only terminal views are committed in one atomic derive_batch; receipts and
    views produced by earlier nested calls survive a later failure. The caller
    owns/closes the SAME session, including on error. Local callbacks then expire.

    Limits bound serialized snapshots, not heap/tokens/cost; request input must
    be bounded upstream. No hidden callback reads, live ACL adapter installation,
    main activation, direct/final rescue, cache or link/semantic certification.
    Trusted collaborators, no concurrent/reentrant use of this journal or hostile
    Python sandbox and no atomic external revocation during model execution.
    """
    _callbacks(session, authorize, invoke)
    def work():
        if (getattr(request, "requested_mode", None) != "ask"
                or getattr(decision, "effective_mode", None) != "ask"
                or type(runtime) is not orchestration.AssistantCoreRefineRetrievalRuntime
                or type(facet_runtime) is not ranking.AssistantCoreMergeFacetCandidatesRuntime):
            raise EvidenceProductionError("ASK/ASK refinement and facet runtimes required")
        trace = _OuterRetrievalTrace(request=request, session=session, authorize=authorize,
            invoke=invoke, max_records=max_trace_records, max_bytes=max_trace_bytes)
        try:
            for cap in (runtime.ASSISTANT_CORE_MAX_FACETS, runtime.V13_DENSE_QUERY_LIMIT,
                        runtime.V13_LEXICAL_QUERY_LIMIT, runtime.V13_MAX_EVIDENCE_ITEMS_ASK,
                        runtime.V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE):
                if type(cap) is not int or not 1 <= cap <= trace.limits.assembly.max_occurrences:
                    raise EvidenceProductionError("bounded refinement capacities required")
            supplied = trace.import_pack(retrieval, selections)
            facet_bound = replace(facet_runtime,
                _assistant_core_candidate_stable_key=trace.readonly(facet_runtime._assistant_core_candidate_stable_key),
                _dedup_text_values=trace.readonly(facet_runtime._dedup_text_values),
                _v13_merge_candidates=lambda groups: trace.operator(
                    facet_runtime._v13_merge_candidates, groups, groups, same_locator=True))
            names = ("_assistant_core_retrieval_query", "_assistant_core_scope_value",
                     "_v13_current_budget", "_v13_fallback_plan", "_dedup_text_values",
                     "_assistant_core_facet_candidate_confidence", "_assistant_core_candidate_source_type",
                     "_v13_evidence_metrics")
            bound = replace(runtime, **{n: trace.readonly(getattr(runtime, n)) for n in names},
                _v13_initial_retrieval=lambda **kw: trace.initial(initial_adapter, **kw),
                _v13_score_candidates=lambda q, values: trace.operator(
                    runtime._v13_score_candidates, [values], q, values),
                _assistant_core_merge_facet_candidates=lambda groups: trace.run(
                    ranking.assistant_core_merge_facet_candidates, groups,
                    runtime=facet_bound, lineage=trace.event))
            result = trace.call(orchestration.assistant_core_refine_retrieval,
                                request, supplied, decision, runtime=bound, lineage=trace.event)
            if trace.completed is None:
                raise EvidenceProductionError("refinement completion missing")
            if trace.completed[0] == "unchanged":
                if result is not supplied or trace.initial_calls or trace.bonus_done:
                    raise EvidenceProductionError("unchanged refinement altered its path")
            elif not trace.bonus_done:
                raise EvidenceProductionError("refinement bonus phase incomplete")
            return trace.finish(result, preserve=selections if trace.completed[0] == "unchanged" else None)
        finally:
            trace.dispose()
    return invoke(work, request)
