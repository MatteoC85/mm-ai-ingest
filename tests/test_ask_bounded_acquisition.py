"""Offline regressions for bounded ASK acquisition and complete review windows.

All provider/database/network calls are denied. These assert retrieval work,
scope forwarding, facet confidence and scheduling, not live answer quality.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields, replace
import importlib
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_core_v2 import AssistantCoreDecision, AssistantCoreFacetQuery, AssistantCoreHooks, AssistantCoreRequest, AssistantCoreV2
from machinemind.ask import core_intake, execution, request_binding, request_completion
from machinemind.retrieval import evidence_orchestration as orchestration


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class AskBoundedAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        cls.guards.enter_context(patch.object(socket, "create_connection", denied))
        cls.guards.enter_context(patch.object(socket.socket, "connect", denied))
        cls.guards.enter_context(patch("requests.sessions.Session.request", denied))
        cls.guards.enter_context(patch("psycopg2.connect", denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def fixture(self, mode="ask"):
        facets = ("mechanical alignment", "pressure isolation", "sensor confirmation", "lubrication interval")
        request = AssistantCoreRequest(query="Describe preparation and maintenance checks.",
            requested_mode=mode, response_language="en", company_id="fixture-company",
            machine_id="fixture-machine", ai_scope="selected_docs", top_k=8,
            metadata={"document_ids": ["fixture-document"], "bubble_document_id": "fixture-document"})
        decision = AssistantCoreDecision(request_kind="procedure", effective_mode=mode,
            confidence=.99, requested_mode_fit=True, evidence_state="supported",
            evidence_policy="evidence_required", information_task="procedure_full",
            required_answer_types=("ordered_actions",), required_facets=facets,
            facet_queries=tuple(AssistantCoreFacetQuery(facet=f, answer_type="ordered_actions",
                dense_queries=(f + " manual passage",), lexical_queries=(f + " exact check",),
                exact_terms=("identifier-" + str(i),), preferred_source_types=("document",))
                for i, f in enumerate(facets)))
        rows = [{"citation_id": "fixture:p1:c" + str(i), "bubble_document_id": "fixture-document",
                 "page_from": 1, "page_to": 1, "chunk_index": i, "source_type": "document",
                 "chunk_full": "Documented " + facet + ": carry out the prescribed check.",
                 "snippet": "Documented " + facet + ": carry out the prescribed check.",
                 "similarity": .8, "semantic_similarity": .8, "retrieval_score": .8}
                for i, facet in enumerate(facets[:3])]
        return request, decision, rows

    def refine(self, *, mode="ask", fail=False, remaining=70):
        request, decision, rows = self.fixture(mode)
        before = deepcopy(rows)
        calls, supports = [], []
        runtime = self.m._assistant_core_refine_retrieval_runtime()

        def acquire(**kwargs):
            calls.append(deepcopy(kwargs))
            if fail:
                raise RuntimeError("offline provider unavailable")
            return {"candidates": deepcopy(rows)}

        def confidence(**kwargs):
            supports.append((kwargs["facet"], kwargs["candidate"]["citation_id"]))
            return runtime._assistant_core_facet_candidate_confidence(**kwargs)

        local = replace(runtime, _v13_current_budget=lambda: SimpleNamespace(remaining=lambda: remaining),
            _v13_initial_retrieval=acquire, _assistant_core_facet_candidate_confidence=confidence)
        result = orchestration.assistant_core_refine_retrieval(request,
            {"candidates": rows, "citations": rows, "plan": {}}, decision, runtime=local)
        self.assertEqual(rows, before)
        return request, decision, result, calls, supports

    def test_four_ask_facets_share_one_acquisition_but_keep_independent_confidence(self):
        request, decision, result, calls, supports = self.refine()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "neutral")
        self.assertTrue(calls[0]["plan"]["_ask_facet_batch"])
        self.assertEqual(calls[0]["plan"]["dense_queries"], list(decision.required_facets))
        self.assertEqual({facet for facet, _ in supports}, set(decision.required_facets))
        self.assertEqual(len(result["assistant_core_facet_retrieval"]), 4)
        missing = next(r for r in result["assistant_core_facet_retrieval"]
                       if r["facet"] == "lubrication interval")
        self.assertEqual(missing["credible_candidate_count"], 0)
        self.assertFalse(any("lubrication interval" in r.get("assistant_core_facet_hits", [])
                             for r in result["candidates"]))

    def test_batch_similarity_and_title_score_cannot_claim_an_unrelated_facet(self):
        request, decision, _ = self.fixture()
        row = {"citation_id": "fixture:p1:c0", "bubble_document_id": "fixture-document",
               "page_from": 1, "page_to": 1, "chunk_index": 0, "source_type": "step",
               "chunk_full": "Open the enclosure. Close the door.",
               "semantic_similarity": .99, "similarity": .99,
               "structured_title_match": True, "structured_title_match_score": .99}
        local = replace(self.m._assistant_core_refine_retrieval_runtime(),
            _v13_current_budget=lambda: None,
            _v13_initial_retrieval=lambda **kw: {"candidates": [row]})
        result = orchestration.assistant_core_refine_retrieval(request,
            {"candidates": [], "citations": []}, decision, runtime=local)
        self.assertTrue(result["candidates"])
        self.assertTrue(all(r["credible_candidate_count"] == 0 for r in result["assistant_core_facet_retrieval"]))
        self.assertTrue(all(not r["assistant_core_facet_hits"] for r in result["candidates"]))

    def test_scope_document_selection_and_language_are_forwarded_unchanged(self):
        request, _, result, calls, _ = self.refine()
        for field in ("company_id", "machine_id", "ai_scope", "response_language"):
            self.assertEqual(calls[0][field], getattr(request, field))
        self.assertEqual(calls[0]["doc_ids"], ["fixture-document"])
        self.assertEqual(calls[0]["bubble_document_id"], "fixture-document")
        self.assertTrue(all(r["bubble_document_id"] == "fixture-document" for r in result["candidates"]))

    def test_failed_batch_never_retries_per_facet_or_discards_existing_evidence(self):
        _, _, result, calls, supports = self.refine(fail=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertTrue(supports)
        self.assertTrue(all(r["acquisition_failed"] for r in result["assistant_core_facet_retrieval"]))

    def test_root_and_smart_keep_their_original_acquisition_paths(self):
        for mode in ("root_cause", "smart_diagnostic"):
            with self.subTest(mode=mode):
                _, _, _, calls, _ = self.refine(mode=mode)
                self.assertEqual(len(calls), 4)
                self.assertTrue(all("_ask_facet_batch" not in c["plan"] for c in calls))
                self.assertTrue(all(c["mode"] == ("smart_neutral" if mode == "smart_diagnostic" else "neutral") for c in calls))

    def test_low_budget_still_skips_all_remote_refinement(self):
        _, _, result, calls, supports = self.refine(remaining=17)
        self.assertEqual(calls, [])
        self.assertEqual(supports, [])
        self.assertNotIn("assistant_core_refined", result)

    def test_initial_batch_preserves_all_query_slots_and_keeps_original_question_for_scoring(self):
        runtime = self.m._assistant_core_initial_retrieval_runtime()
        embedded, scored, selectors = [], [], []
        rows = self.fixture()[2]
        def embed(queries, **kwargs):
            embedded.extend(queries)
            return [[1.0] for _ in queries]
        def dense(**kwargs):
            selectors.append(kwargs)
            return 0, []
        local = replace(runtime, _v13_current_budget=lambda: None,
            _openai_embed_texts=embed, _fetch_dense_chunk_candidates=dense,
            _raw_rows_to_dense_candidates=lambda *a, **k: [],
            _rrf_merge_candidates=lambda *a, **k: [], _fts_search_chunks_prefix=lambda **k: [],
            _fts_search_chunks_multi=lambda **k: [], _v13_exact_identifier_candidates=lambda **k: [],
            _v13_fetch_scored_pages=lambda **k: [], _v13_fetch_preferred_source_pages=lambda **k: [],
            _v13_fetch_structured_dense_candidates=lambda **k: [],
            _ask_structured_direct_fetch_sources=lambda **k: [],
            _v13_merge_candidates=lambda groups: rows,
            _v13_score_candidates=lambda q, values: (scored.append(q) or values))
        request, decision, _ = self.fixture()
        result = orchestration.v13_initial_retrieval(q=request.query,
            company_id=request.company_id, machine_id=request.machine_id,
            doc_ids=["fixture-document"], bubble_document_id="fixture-document",
            ai_scope=request.ai_scope, response_language="en", mode="neutral",
            plan={"_ask_facet_batch": True, "dense_queries": list(decision.required_facets),
                  "lexical_queries": list(decision.required_facets)}, runtime=local)
        self.assertEqual(embedded, list(decision.required_facets))
        self.assertEqual(scored, [request.query])
        self.assertEqual(len(selectors), 4)
        self.assertTrue(all(s["company_id"] == request.company_id and s["machine_id"] == request.machine_id
                            and s["doc_ids"] == ["fixture-document"] for s in selectors))

    def test_shared_pool_transforms_keep_real_lineage_and_reject_unregistered_records(self):
        # Exercise the real local lineage journal/operators. Authentication and
        # SQL are tested separately: these inputs are already registered receipts.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_lineage_copy_reuse import fixture as lineage_fixture
        from machinemind.retrieval import receipt_producers as production, candidate_ranking as ranking
        for forged in (False, True):
            with self.subTest(forged=forged):
                base, rows = lineage_fixture(count=3, vector_size=4)
                trace = production._OuterRetrievalTrace.__new__(production._OuterRetrievalTrace)
                trace.__dict__.update(base.__dict__)
                trace.pending, trace.completed, trace.bonus_done = {}, None, False
                trace._live_outer_records = {}
                trace.invoke = lambda fn, req, *a, **kw: fn(*a, **kw)
                request, decision, _ = self.fixture()
                trace.request = request
                source_rows = [dict(r) for r in rows] if forged else rows
                score_runtime = ranking.V13ScoreCandidatesRuntime(**{
                    field.name: getattr(self.m, field.name) for field in fields(ranking.V13ScoreCandidatesRuntime)})
                def score(q, values, **kwargs):
                    return ranking.v13_score_candidates(q, values, runtime=score_runtime, **kwargs)
                facets = ranking.AssistantCoreMergeFacetCandidatesRuntime(
                    _assistant_core_candidate_stable_key=self.m._assistant_core_candidate_stable_key,
                    _dedup_text_values=self.m._dedup_text_values,
                    _v13_merge_candidates=lambda groups: trace.operator(
                        ranking.v13_merge_candidates, groups, groups, same_locator=True))
                local = replace(self.m._assistant_core_refine_retrieval_runtime(),
                    _v13_current_budget=lambda: None,
                    _v13_initial_retrieval=lambda **kw: {"candidates": source_rows},
                    _v13_score_candidates=lambda q, values: trace.operator(score, [values], q, values),
                    _assistant_core_merge_facet_candidates=lambda groups: ranking.assistant_core_merge_facet_candidates(
                        groups, runtime=facets, lineage=trace.event))
                def run():
                    return orchestration.assistant_core_refine_retrieval(request,
                        {"candidates": rows, "citations": rows}, decision, runtime=local, lineage=trace.event)
                if forged:
                    with self.assertRaises(production.EvidenceProductionError):
                        run()
                else:
                    result = run()
                    self.assertEqual(trace.completed[0], "complete")
                    self.assertTrue(trace.bonus_done)
                    self.assertTrue(result["candidates"])
                    # None of these synthetic source passages supports a facet.
                    # They stay available, without one derived view per non-hit.
                    self.assertEqual(len(result["candidates"]), len(rows))
                    self.assertEqual(trace._live_outer_records, {})
                    self.assertTrue(all(not row["assistant_core_facet_hits"] for row in result["candidates"]))
                    for row in result["candidates"]:
                        origins = trace.entry(row)[2]
                        self.assertTrue(origins)
                        self.assertTrue(all(origin in trace.roots for origin in origins))

    def test_real_intake_rebinding_preserves_synthesis_and_deadline_observers(self):
        request, _, _ = self.fixture()
        hooks = AssistantCoreHooks(retrieve_neutral=denied, route_semantically=denied,
            prepare_evidence=denied, synthesize_ask=denied, synthesize_root_cause=denied,
            synthesize_general=denied, build_no_evidence=denied, build_clarification=denied,
            build_out_of_scope=denied, build_safety_refusal=denied)
        check = lambda: None
        runtimes = request_binding.AskRuntimeFactories(
            execution=self.m._assistant_core_ask_execution_runtime,
            validation=self.m._assistant_core_ask_validation_runtime,
            check=check, synthesis=request_completion.synthesize_observed)
        intake = core_intake._CoreIntake.__new__(core_intake._CoreIntake)
        intake.used, intake.active, intake.request = False, False, request
        intake.core, intake.runtimes, intake.generation = AssistantCoreV2(hooks), runtimes, object()
        intake.session, intake.consumers, intake._response_flow = object(), False, None
        intake._task_input, intake._task_used = {}, set()
        intake._check = lambda req: None
        intake.authorize = lambda req: frozenset()
        intake.owner_invoke = lambda fn, req: fn()
        captured = []
        def bound_run(req, *, core, runtimes, evidence, acquisition):
            captured.append(runtimes)
            runtimes.synthesis(lambda: request_completion.provider_timeout('ask_structured_synthesis', 60), lambda: True)
            return {'ok': True}
        owner = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=request.company_id), started=time.monotonic())
        try:
            with patch.object(core_intake, 'AskRequestEvidence', lambda **kw: SimpleNamespace(**kw)), \
                 patch.object(core_intake, 'run_core_request', bound_run):
                self.assertEqual(intake.run(request), {'ok': True})
            self.assertIs(captured[0].check, check)
            self.assertIs(captured[0].synthesis, runtimes.synthesis)
            self.assertEqual(owner.provider_schedules[0]['review_reserve_seconds'], 10.5)
            self.assertFalse(intake.active)
        finally:
            owner.close()

    def test_protected_binding_reserves_review_even_when_initial_procedure_policy_says_false(self):
        m = self.m
        request, decision, _ = self.fixture()
        er = replace(m._assistant_core_ask_execution_runtime(),
                     _assistant_core_should_semantic_verify_answer=lambda dec: False)
        vr = m._assistant_core_ask_validation_runtime()
        # Explicit required hooks avoid using the actual Core state machine in a
        # lifecycle test: its real local synthesis hook is exercised below.
        hooks = AssistantCoreHooks(retrieve_neutral=denied, route_semantically=denied,
            prepare_evidence=denied, synthesize_ask=denied, synthesize_root_cause=denied,
            synthesize_general=denied, build_no_evidence=denied, build_clarification=denied,
            build_out_of_scope=denied, build_safety_refusal=denied)
        owner = request_completion.RequestCompletion(runtime=m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=request.company_id), started=time.monotonic())
        try:
            def synthesize(*args, **kwargs):
                request_completion.provider_timeout("ask_structured_synthesis", 60)
                return {"ok": True, "status": "answered"}
            def run(instance, req):
                return instance.hooks.synthesize_ask(req, {}, decision)
            with patch.object(AssistantCoreV2, "run", run), patch.object(execution, "synthesize_ask", synthesize):
                result = request_binding.run_core_request(request, core=AssistantCoreV2(hooks),
                    runtimes=request_binding.AskRuntimeFactories(execution=lambda: er, validation=lambda: vr,
                        synthesis=request_completion.synthesize_observed))
            self.assertEqual(result["status"], "answered")
            self.assertEqual(owner.provider_schedules[0]["review_reserve_seconds"], 10.5)
            self.assertEqual(owner.budget.llm_calls, 0)
            request_completion.provider_timeout("ask_final_synthesis", 60)
            self.assertEqual(owner.provider_schedules[-1]["review_reserve_seconds"], 0)
        finally:
            owner.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
