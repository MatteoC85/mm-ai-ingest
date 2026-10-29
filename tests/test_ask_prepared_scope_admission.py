"""Real owned admission and generic generation, starting at prepared evidence.

Only SQL observations/current allowance and provider output are fixtures. No
admission hook, observer, session, source compiler or generation adapter is
patched. Retrieval/routing/preparation quality is outside this boundary test.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from unittest.mock import patch
from machinemind.ask import core_intake, execution, response_evidence
from machinemind.evidence.contracts import (EvidenceContractError, SourceIdentity, SourceScope,
    SourceType, SourceFormat, ScopeLevel)
from machinemind.retrieval.ask_composition import AskEvidenceSession, AskSessionLimits, AskSelection
from machinemind.retrieval.chunk_evidence import ChunkReadScope
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.production_adapters import ProductionReaderAdapters
from test_lineage_copy_reuse import fixture as limits_fixture
import test_ask_conditions_pipeline as condition_fixtures


def deny(*a, **kw):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class StructuredBoundaryReached(Exception):
    pass


class AskPreparedScopeAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {"MM_USAGE_ENFORCEMENT": "off",
            "AI_INTERNAL_SECRET": "offline-only", "OPENAI_API_KEY": "offline-unused",
            "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, deny))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def harness(self, scope="conditions_only", *, structured=False, observe=True, tamper=None, revoke=False):
        m = self.m
        base = condition_fixtures.AskConditionsPipelineTests()
        base.m = m
        request, decision = base.request_decision()
        request = replace(request, allowed_effective_modes=("ask",))
        decision = replace(decision, procedure_scope=scope)
        if scope != "conditions_only":
            decision = replace(decision, required_answer_types=("ordered_actions",),
                facet_queries=(m.AssistantCoreFacetQuery(facet="safe insertion", answer_type="ordered_actions", must_cover=True),),
                required_facets=("safe insertion",))
        key = "step:fixture" if structured else "fixture-manual"
        source = SourceIdentity(SourceScope(request.company_id, ScopeLevel.MACHINE, request.machine_id),
            SourceType.STEP if structured else SourceType.DOCUMENT,
            "fixture" if structured else key, SourceFormat.STRUCTURED if structured else SourceFormat.UNKNOWN)
        allowance = [frozenset({source})]
        def authorize(req):
            self.assertIs(req, request)
            return allowance[0]
        def invoke(callback, req, *args, **kwargs):
            self.assertIs(req, request)
            return callback(*args, **kwargs)
        limits = limits_fixture(1, 1)[0].limits
        read_scope = ChunkReadScope(request.company_id, request.machine_id, request.ai_scope)
        session = AskEvidenceSession(request=request, scope=read_scope,
            limits=AskSessionLimits(limits, max_reads=20, max_records=400, max_bytes=128_000_000))
        body = ("SOURCE_TYPE: step\nSTEP_NUMBER: 1\nTITLE: Safe insertion\nDESCRIPTION: " if structured else "")
        body += "Before insertion, verify zero energy and a clear guide. Keep STOP pressed."
        capture = _PageReadCapture(read_scope, limits, "structured_direct" if structured else "ask_pages", snippet_chars=800)
        capture.expect(1, 2000)
        capture.capture([(key, request.machine_id, 1, body, request.company_id, request.machine_id)])
        row = {"bubble_document_id": key, "page_from": 1, "page_to": 1,
            "citation_id": key + ":p1", "source_type": "step" if structured else "document",
            "chunk_full": body, "snippet": body, "similarity": .8}
        receipt = capture.finish([row])
        registration = session.register(request=request, receipt=receipt, current_allowed_sources=authorize(request))
        rows = [dict(item.record) for item in session.records(request=request, handles=registration.records,
            current_allowed_sources=authorize(request))]
        readers = ProductionReaderAdapters(request=request, session=session, authorize=authorize,
            invoke=invoke, runtimes=m._assistant_core_authority_reader_runtimes())
        owner = m._assistant_core_intake_factory(request=request, session=session, readers=readers,
            authorize=authorize, invoke=invoke).__self__
        owner.active = True
        owner._prepared = True
        owner._selection = (AskSelection("candidates", registration.records), AskSelection("citations", registration.records))
        prepared = {"candidates": rows, "citations": rows, "plan": {},
            "assistant_core_contract": {"fail_closed": True}, "retrieval_assurance": {}}
        owner._prepared_retrieval = deepcopy(prepared)
        calls = []
        def provider(messages, **kwargs):
            calls.append((messages, kwargs))
            return {"answer_status": "answered", "grounded_points": [{"text":
                "Verify zero energy and a clear guide before insertion. Keep STOP pressed.",
                "citation_ids": [row["citation_id"]]}]}, "offline-model"
        owner.generation = replace(owner.generation,
            _v13_json_models=provider, _v13_choose_ask_model=lambda *a, **kw:(m.V13_FAST_MODEL, "low", ""),
            _sanitize_citations_for_response=lambda items, **kw:[dict(item) for item in items],
            _build_rg_links=lambda *a, **kw:[])
        # Bind the existing factory, observer, admission hook and generic adapter.
        # The initialized prepared state above stands in only for earlier SQL/P4.
        owner.consumers = observe
        runtime = owner._execution_factory()
        runtime = execution.bind_ask_execution(runtime, session=session, authorize=authorize, select=owner.select)
        real_admission = runtime.evidence_admission
        stages = []
        def recording_admission(req, data, dec, stage):
            stages.append(stage)
            if stage in {"synthesis.generate", "synthesis.structured"} and revoke:
                allowance[0] = frozenset()
            if stage == "synthesis.generate" and tamper is not None and not observe:
                target, value = tamper
                data = {**data, target: {**data[target], "procedure_scope": value}}
            return real_admission(req, data, dec, stage)
        runtime = replace(runtime, evidence_admission=recording_admission)
        if structured and scope != "conditions_only":
            def reached(**kw):
                # This sentinel executes strictly AFTER the real owned structured
                # admission. It prevents irrelevant DB/family selection in this test.
                raise StructuredBoundaryReached()
            runtime = replace(runtime, _v13_structured_ask=reached)
        if tamper is not None and observe:
            real_observer = runtime.evidence_observer
            original_admit = real_observer.admit
            def mutate_at_boundary(req, data, dec, stage, admission):
                if stage == "synthesis.generate":
                    target, value = tamper
                    data = {**data, target: {**data[target], "procedure_scope": value}}
                return original_admit(req, data, dec, stage, admission)
            real_observer.admit = mutate_at_boundary
        return request, decision, owner, prepared, runtime, session, calls, stages

    def finish(self, owner, session):
        if owner._response_flow is not None:
            owner._response_flow.close()
        owner.active = False
        session.close()

    def test_all_scope_values_reach_real_generic_generation_with_owned_admission(self):
        for scope in ("conditions_only", "execution_steps", "other", ""):
            with self.subTest(scope=scope):
                request, decision, owner, data, runtime, session, calls, stages = self.harness(scope)
                try:
                    result = execution.synthesize_ask(request, data, decision, runtime=runtime)
                    self.assertEqual(result["status"], "answered")
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(stages, ["synthesis.input", "synthesis.generate"])
                    self.assertEqual(owner._generation_input["assistant_core_contract"]["procedure_scope"], scope)
                    self.assertEqual(owner._generation_input["plan"]["procedure_scope"], scope)
                    self.assertTrue(result["meta"]["generation_evidence"]["complete_primary_representation"])
                    self.assertIsNotNone(owner._response_flow.synthesis)
                finally:
                    self.finish(owner, session)

    def test_checks_with_structured_evidence_use_real_generic_adapter(self):
        request, decision, owner, data, runtime, session, calls, stages = self.harness(structured=True)
        try:
            result = execution.synthesize_ask(request, data, decision, runtime=runtime)
            self.assertEqual(result["status"], "answered")
            self.assertEqual(len(calls), 1)
            self.assertEqual(stages, ["synthesis.input", "synthesis.generate"])
            self.assertEqual(result["citations"][0]["bubble_document_id"], "step:fixture")
        finally:
            self.finish(owner, session)

    def test_execution_and_legacy_structured_admission_remain_valid(self):
        for scope in ("execution_steps", "other", ""):
            with self.subTest(scope=scope):
                request, decision, owner, data, runtime, session, calls, stages = self.harness(scope, structured=True)
                try:
                    with self.assertRaises(StructuredBoundaryReached):
                        execution.synthesize_ask(request, data, decision, runtime=runtime)
                    self.assertEqual(stages, ["synthesis.input", "synthesis.structured"])
                    self.assertEqual(calls, [])
                finally:
                    self.finish(owner, session)

    def test_contract_or_plan_scope_tampering_is_rejected_before_provider(self):
        for observe in (True, False):
            for target in ("assistant_core_contract", "plan"):
                with self.subTest(target=target, observer=observe):
                    request, decision, owner, data, runtime, session, calls, stages = self.harness(
                        tamper=(target, "execution_steps"), observe=observe)
                    try:
                        error = response_evidence.ResponseEvidenceError if observe else core_intake.CoreIntakeError
                        message = "synthesis changed prepared contract" if observe else "generation changed prepared contract"
                        with self.assertRaisesRegex(error, message):
                            execution.synthesize_ask(request, data, decision, runtime=runtime)
                        self.assertEqual(calls, [])
                        self.assertIsNotNone(owner.fault)
                    finally:
                        self.finish(owner, session)

    def test_scope_field_does_not_bypass_current_source_revocation(self):
        request, decision, owner, data, runtime, session, calls, stages = self.harness(revoke=True)
        try:
            with self.assertRaises(EvidenceContractError):
                execution.synthesize_ask(request, data, decision, runtime=runtime)
            self.assertEqual(calls, [])
        finally:
            self.finish(owner, session)

    def test_intake_admission_without_response_observer_uses_same_pinned_transform(self):
        request, decision, owner, data, runtime, session, calls, stages = self.harness(observe=False)
        try:
            result = execution.synthesize_ask(request, data, decision, runtime=runtime)
            self.assertEqual(result["status"], "answered")
            self.assertEqual(len(calls), 1)
            self.assertEqual(stages, ["synthesis.input", "synthesis.generate"])
        finally:
            self.finish(owner, session)


if __name__ == "__main__":
    unittest.main(verbosity=2)
