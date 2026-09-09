"""P6-B4f: request-local acquisition dependencies for the existing P4 algorithms.

No retrieval algorithm is copied. The original evidence_orchestration functions
are executed with four explicit snapshots (initial, neutral, refinement, prepare).
Nested initial retrieval uses the SAME snapshot, not a main/global factory.

The owner is request_binding: its invoke closure checks request identity, lifetime
and latched canonical failures before/after callbacks. This module allocates no
session, grants, cache or transport. It does not recover provenance from text/IDs.
Future receipt-producing callbacks must retain/register their own occurrences.
Main provides legacy readers in this release; this is not live canonical intake.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable

from ..evidence.contracts import EvidenceContractError
from ..retrieval import evidence_orchestration as orchestration

ACQUISITION_VERSION = "ask-acquisition-runtime-p6b4f-v1"


@dataclass(frozen=True, slots=True, repr=False)
class AskAcquisitionFactories:
    """Four call-time factories; invoked once, only for an ASK Core cache miss.

    The snapshot captures references, not the mutable internals of collaborators.
    Upstream must bound records and supply current authorization when canonical
    readers are configured. This class is not an ACL provider or cost budget.
    """
    initial: Callable[[], orchestration.V13InitialRetrievalRuntime]
    neutral: Callable[[], orchestration.AssistantCoreRetrieveNeutralRuntime]
    refine: Callable[[], orchestration.AssistantCoreRefineRetrievalRuntime]
    prepare: Callable[[], orchestration.AssistantCorePrepareEvidenceRuntime]

    def __post_init__(self) -> None:
        if not all(callable(getattr(self, name)) for name in
                   ("initial", "neutral", "refine", "prepare")):
            raise EvidenceContractError("four explicit acquisition factories required")


@dataclass(frozen=True, slots=True, repr=False)
class AskAcquisitionCallbacks:
    """Callbacks owned by the caller's request lifetime, not portable credentials."""
    neutral: Callable[..., dict]
    refine: Callable[..., dict]
    prepare: Callable[..., dict]


def bind_acquisition(request: Any, *, factories: AskAcquisitionFactories,
                     invoke: Callable[..., Any]) -> AskAcquisitionCallbacks:
    """Snapshot and bind existing orchestration to the caller's invocation guard.

    `invoke(callback, req, *args, **kwargs)` is the existing B4e guard, not a new
    validator. It latches EvidenceContractError on the opt-in canonical path so
    a catch inside P4 cannot disguise a direct dependency's binding failure as an
    empty retrieval. Off-mode preserves legacy catch/fallback behavior.

    This guards callbacks directly present in these four runtimes. It cannot
    detect exceptions already swallowed inside an opaque collaborator, authorize
    its hidden reads, infer lineage of a transformed record, or make ACL revocation
    atomic across remote calls. Those producers must use the same owner explicitly.
    """
    if getattr(request, "requested_mode", None) != "ask":
        raise EvidenceContractError("acquisition composition is ASK-only")
    if type(factories) is not AskAcquisitionFactories or not callable(invoke):
        raise EvidenceContractError("typed factories and request invocation required")

    snapshots = []
    for name, expected in (
        ("initial", orchestration.V13InitialRetrievalRuntime),
        ("neutral", orchestration.AssistantCoreRetrieveNeutralRuntime),
        ("refine", orchestration.AssistantCoreRefineRetrievalRuntime),
        ("prepare", orchestration.AssistantCorePrepareEvidenceRuntime),
    ):
        value = invoke(getattr(factories, name), request)
        if type(value) is not expected:
            raise EvidenceContractError("unexpected acquisition runtime: " + name)
        snapshots.append(value)
    initial_runtime, neutral_runtime, refine_runtime, prepare_runtime = snapshots

    def guarded(callback):
        def call(*args, **kwargs):
            return invoke(callback, request, *args, **kwargs)
        return call

    def guard_runtime(runtime):
        # Only actual callable dependency values are wrapped. Configuration and
        # module references retain object identity. No signature/argument coercion.
        return replace(runtime, **{f.name: guarded(getattr(runtime, f.name))
            for f in fields(runtime) if callable(getattr(runtime, f.name))})

    local_initial = guard_runtime(initial_runtime)

    def initial(*, q, company_id, machine_id, doc_ids, bubble_document_id,
                ai_scope, response_language, mode, plan=None):
        return invoke(orchestration.v13_initial_retrieval, request,
            q=q, company_id=company_id, machine_id=machine_id,
            doc_ids=doc_ids, bubble_document_id=bubble_document_id,
            ai_scope=ai_scope, response_language=response_language,
            mode=mode, plan=plan, runtime=local_initial)

    local_neutral = guard_runtime(replace(neutral_runtime,
                                         _v13_initial_retrieval=initial))
    local_refine = guard_runtime(replace(refine_runtime,
                                        _v13_initial_retrieval=initial))
    local_prepare = guard_runtime(prepare_runtime)

    def neutral(req):
        return invoke(orchestration.assistant_core_retrieve_neutral, req,
                      req, runtime=local_neutral)

    def refine(req, data, decision):
        return invoke(orchestration.assistant_core_refine_retrieval, req,
                      req, data, decision, runtime=local_refine)

    def prepare(req, data, decision):
        return invoke(orchestration.assistant_core_prepare_evidence, req,
                      req, data, decision, runtime=local_prepare)

    return AskAcquisitionCallbacks(neutral, refine, prepare)
