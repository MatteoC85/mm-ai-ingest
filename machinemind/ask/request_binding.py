"""P6-B4e: request-local hooks for the EXISTING AssistantCoreV2.

Main dispatches cache-miss ASK through these local consumers. Non-ASK requests
retain the original engine. No shared hooks/runtime are mutated and no alternate
Core implementation is introduced. The local Core is only a hooks container.

The optional evidence binding reuses the caller's B4a session, current authority,
and explicit occurrence selection through B4b EXECUTION and B4c VALIDATION_V1.
Acquisition/derivation callbacks must already register their own receipts. This
module never invents grants or reconstructs handles from IDs/text/scores.

Production main supplies NO evidence binding yet. This is request composition,
not final canonical activation: callback internals, metadata outside the two
collections, scalar rescue, cache, link validity and answer semantics remain
separate integration gates. No transport or new retrieval/model call lives here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from assistant_core_v2 import AssistantCoreHooks, AssistantCoreV2
from . import execution, validation
from ..evidence.ask_input import apply_ask_evidence_input, ask_request_key
from ..evidence.contracts import EvidenceContractError, SourceIdentity
from ..retrieval.ask_composition import AskEvidenceSession, AskSelection

REQUEST_BINDING_VERSION = "ask-request-binding-p6b4e-v1"


@dataclass(frozen=True, slots=True, repr=False)
class AskRuntimeFactories:
    """Call-time dependency factories; never evaluated on a non-ASK dispatch."""
    execution: Callable[[], execution.AskExecutionRuntime]
    validation: Callable[[], validation.AskValidationRuntime]

    def __post_init__(self) -> None:
        if not callable(self.execution) or not callable(self.validation):
            raise EvidenceContractError("explicit ASK runtime factories required")


@dataclass(frozen=True, slots=True, repr=False)
class AskRequestEvidence:
    """Already-owned session plus trusted CURRENT-authority and handle selectors.

    No client/cache deserialization, provider search or allowance caching. select
    may receive decision=None before semantic routing. Its stage is a code
    boundary label, not an instruction about a machine, language or relevance.
    The caller owns/cleans up the session; this module creates no second session.
    """
    session: AskEvidenceSession = field(repr=False)
    authorize: Callable[[Any], frozenset[SourceIdentity]] = field(repr=False)
    select: Callable[[Any, dict, Any, str], tuple[AskSelection, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.session, AskEvidenceSession)
                or not callable(self.authorize) or not callable(self.select)):
            raise EvidenceContractError("existing session and trusted authority/selection required")


def _local_legacy_consumers(er: execution.AskExecutionRuntime,
                            vr: validation.AskValidationRuntime):
    """Bind nested legacy calls to the same per-request dependency snapshots.

    This mirrors main's delegates under a stable configuration. It deliberately
    freezes these two dependency sets once per ASK core run, rather than reading
    main globals again during a nested verifier/recovery call. Collaborator
    internals are not made immutable/thread-safe by a frozen runtime object.
    """
    def verify(*, request, decision, answer, candidates, repair_context=None):
        if not execution._ask_path(request, decision):
            return er._assistant_core_verify_or_repair_answer(request=request,
                decision=decision, answer=answer, candidates=candidates,
                repair_context=repair_context)
        return execution.verify_or_repair_answer(request=request, decision=decision,
            answer=answer, candidates=candidates, repair_context=repair_context,
            runtime=local_execution)

    def recover(*, request, decision, retrieval, reason):
        if not execution._ask_path(request, decision):
            return er._assistant_core_recover_ask_from_evidence(request=request,
                decision=decision, retrieval=retrieval, reason=reason)
        return execution.recover_ask_from_evidence(request=request, decision=decision,
            retrieval=retrieval, reason=reason, runtime=local_execution)

    def citations(response, *, request, retrieval, decision):
        if not execution._ask_path(request, decision):
            return vr._assistant_core_recover_citations(response, request=request,
                retrieval=retrieval, decision=decision)
        return validation.recover_citations(response, request=request,
            retrieval=retrieval, decision=decision, runtime=local_validation)

    local_execution = replace(er,
        _assistant_core_verify_or_repair_answer=verify,
        _assistant_core_recover_ask_from_evidence=recover)
    local_validation = replace(vr, execution_runtime=local_execution,
        _assistant_core_verify_or_repair_answer=verify,
        _assistant_core_recover_citations=citations)
    return local_execution, local_validation


def run_core_request(request: Any, *, core: AssistantCoreV2,
                     runtimes: AskRuntimeFactories,
                     evidence: AskRequestEvidence | None = None) -> dict:
    """Execute once using request-owned hooks, never patch the shared engine.

    Non-ASK (including RC routed to ASK) goes straight to the existing engine and
    does not construct ASK runtimes. Main keeps evidence=None in this release.
    Guarded execution validates collection snapshots before routing/preparation
    and before returning, as well as B4b/B4c consumer boundaries. This does not
    authorize evidence hidden in callback internals or outside the collections.

    A retained callback expires at return/error and cannot be called with a clone
    of this request. A binding fault is terminal even if a legacy router catches
    its exception. This is not a sandbox against malicious Python or a promise of
    atomic ACL revocation during a remote call. Upstream must bound collections;
    snapshots/factories have CPU/heap costs that need runtime measurement.
    """
    if type(core) is not AssistantCoreV2 or type(core.hooks) is not AssistantCoreHooks:
        raise EvidenceContractError("existing AssistantCoreV2 with typed hooks required")
    if type(runtimes) is not AskRuntimeFactories:
        raise EvidenceContractError("typed ASK runtime factories required")
    if evidence is not None and type(evidence) is not AskRequestEvidence:
        raise EvidenceContractError("typed request evidence binding required")
    if getattr(request, "requested_mode", None) != "ask":
        if evidence is not None:
            raise EvidenceContractError("canonical request binding is ASK-only")
        return core.run(request)

    original_request = request
    original_key = ask_request_key(request) if evidence is not None else None
    if evidence is not None and getattr(request, "allowed_effective_modes", ()) != ("ask",):
        raise EvidenceContractError("canonical ASK request must remain on ASK")
    hooks = core.hooks
    if hooks.prepare_ask_evidence is not None:
        raise EvidenceContractError("existing canonical hook must not be overwritten")
    active = True
    fault: Exception | None = None

    def fail(message: str):
        nonlocal fault
        fault = EvidenceContractError(message)
        raise fault

    def check(req):
        nonlocal fault
        if not active:
            fail("request callbacks have expired")
        if fault is not None:
            raise fault
        if req is not original_request:
            fail("callback belongs to another request")
        if evidence is not None:
            try:
                if (ask_request_key(req) != original_key
                        or req.allowed_effective_modes != ("ask",)):
                    fail("original ASK request scope/selection changed")
            except Exception as exc:
                fault = exc
                raise

    def invoke(callback, req, *args, **kwargs):
        nonlocal fault
        check(req)
        try:
            return callback(*args, **kwargs)
        except EvidenceContractError as exc:
            if evidence is not None:
                fault = exc
            raise
        finally:
            check(req)

    def authorize(req):
        nonlocal fault
        check(req)
        try:
            current = evidence.authorize(req)
            check(req)
            return current
        except Exception as exc:
            fault = exc
            raise

    def select(req, data, decision, stage):
        nonlocal fault
        check(req)
        try:
            # The selector cannot rewrite the reference that admission checks.
            result = evidence.select(req, deepcopy(data), decision, stage)
            check(req)
            return result
        except Exception as exc:
            fault = exc
            raise

    def admit(req, data, decision, stage):
        nonlocal fault
        check(req)
        if evidence is None:
            return data
        try:
            snapshot = deepcopy(data)
            current = authorize(req)
            selected = select(req, snapshot, decision, stage)
            admitted = evidence.session.admission(request=req, retrieval=snapshot,
                selections=selected, current_allowed_sources=current)
            check(req)
            return apply_ask_evidence_input(snapshot, request_key=original_key,
                                            admission=admitted)
        except Exception as exc:
            fault = exc
            raise

    try:
        if evidence is not None:
            # Validate ownership/lifecycle before invoking any acquisition callback.
            evidence.session.records(request=request, handles=(),
                                     current_allowed_sources=authorize(request))
        er = invoke(runtimes.execution, request)
        vr = invoke(runtimes.validation, request)
        if (type(er) is not execution.AskExecutionRuntime
                or er.evidence_admission is not None
                or type(vr) is not validation.AskValidationRuntime
                or vr.execution_runtime is not None):
            fail("fresh unbound EXECUTION and VALIDATION runtimes required")
        if evidence is None:
            er, vr = _local_legacy_consumers(er, vr)
        else:
            er = execution.bind_ask_execution(er, session=evidence.session,
                                             authorize=authorize, select=select)
            vr = validation.bind_ask_validation(vr, execution_runtime=er)

        def neutral(req):
            return admit(req, invoke(hooks.retrieve_neutral, req, req), None,
                         "core.retrieve.output")

        def route(req, data):
            data = admit(req, data, None, "core.route.input")
            return invoke(hooks.route_semantically, req, req, data)

        def refine(req, data, dec):
            data = admit(req, data, dec, "core.refine.input")
            return admit(req, invoke(hooks.refine_retrieval, req, req, data, dec),
                         dec, "core.refine.output")

        def prepare(req, data, dec):
            data = admit(req, data, dec, "core.prepare.input")
            out = invoke(hooks.prepare_evidence, req, req, data, dec)
            if evidence is not None:
                out = dict(out or {})
                if "retrieval" in out and out["retrieval"] is not None:
                    out["retrieval"] = admit(req, dict(out["retrieval"]), dec,
                                             "core.prepare.output")
            return out

        def prepare_ask(req, data, dec):
            check(req)
            result = er.evidence_admission(req, data, dec, "core.prepare_ask.input")
            check(req)
            return result

        def synthesize(req, data, dec):
            if not execution._ask_path(req, dec):
                return invoke(hooks.synthesize_ask, req, req, data, dec)
            return invoke(execution.synthesize_ask, req, req, data, dec, runtime=er)

        def validate(response, req, data, dec):
            if not execution._ask_path(req, dec):
                return invoke(hooks.validate_response, req, response, req, data, dec)
            return invoke(validation.validate_response, req, response, req, data, dec, runtime=vr)

        def repair(response, req, data, dec):
            if not execution._ask_path(req, dec):
                return invoke(hooks.repair_response, req, response, req, data, dec)
            return invoke(execution.repair_response, req, response, req, data, dec, runtime=er)

        def simple(callback):
            def call(req, *args):
                return invoke(callback, req, req, *args)
            return call

        local_hooks = replace(hooks,
            retrieve_neutral=neutral, route_semantically=route,
            refine_retrieval=refine if hooks.refine_retrieval is not None else None,
            prepare_evidence=prepare, synthesize_ask=synthesize,
            validate_response=validate if hooks.validate_response is not None else None,
            repair_response=repair if hooks.repair_response is not None else None,
            prepare_ask_evidence=prepare_ask if evidence is not None else None,
            synthesize_root_cause=simple(hooks.synthesize_root_cause),
            synthesize_general=simple(hooks.synthesize_general),
            synthesize_smart_start=(simple(hooks.synthesize_smart_start)
                if hooks.synthesize_smart_start is not None else None),
            build_no_evidence=simple(hooks.build_no_evidence),
            build_clarification=simple(hooks.build_clarification),
            build_out_of_scope=simple(hooks.build_out_of_scope),
            build_safety_refusal=simple(hooks.build_safety_refusal))
        # Same implementation, one run; a new hooks container, not a new Core design.
        result = AssistantCoreV2(local_hooks).run(request)
        check(request)
        if evidence is not None:
            out = dict(result)
            collections = {k:out[k] for k in ("candidates", "citations") if k in out}
            checked = admit(request, collections, None, "core.output.collections")
            out.update(checked)
            return out
        return result
    finally:
        active = False
