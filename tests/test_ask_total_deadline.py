"""Offline total ASK clock and evidence-parity regressions; no provider I/O."""
import ast
from contextlib import ExitStack
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, request_completion, review_evidence
from machinemind.infrastructure import request_budget


def denied(*a, **kw): raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


def configured_clocks(env):
    source = Path(__file__).resolve().parents[1] / "machinemind/config/assistant_runtime.py"
    names = {"ASSISTANT_CORE_HARD_TIMEOUT_SECONDS", "ASSISTANT_CORE_ASK_DEADLINE_SECONDS"}
    assignments = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    assert len(assignments) == 2
    scope = {"os": SimpleNamespace(environ=env)}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec"), scope)
    return scope["ASSISTANT_CORE_ASK_DEADLINE_SECONDS"], scope["ASSISTANT_CORE_HARD_TIMEOUT_SECONDS"]


class AskTotalDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {"MM_USAGE_ENFORCEMENT":"off", "AI_INTERNAL_SECRET":"offline-only",
            "OPENAI_API_KEY":"offline-unused", "MM_INGEST_LEDGER_AUTO_DDL":"0"}))
        for target in ("socket.create_connection", "socket.socket.connect", "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guards.enter_context(patch(target, denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls): cls.guards.close()

    def test_default_clock_and_shorter_overrides_keep_outer_margin(self):
        self.assertEqual(configured_clocks({}), (100,115))
        for requested in (50,65,85,100,105,999):
            for hard in (95,105,115,120):
                with self.subTest(requested=requested, hard=hard):
                    ask,outer=configured_clocks({"MM_ASSISTANT_CORE_ASK_DEADLINE_SECONDS":str(requested),
                        "MM_ASSISTANT_CORE_HARD_TIMEOUT_SECONDS":str(hard)})
                    self.assertEqual(ask,min(requested,100,hard-15))
                    self.assertLessEqual(ask,outer-15)
                    self.assertLessEqual(ask,100)

    def test_recorded_r9_window_gains_exactly_fifteen_seconds_without_new_call_or_cost(self):
        now=[1000.0]
        with patch.object(request_completion,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(request_budget,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(self.m,"ASSISTANT_CORE_ASK_DEADLINE_SECONDS",100):
            owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id="offline-company"),started=now[0])
            try:
                owner.largest_fence_seconds=1.691911
                budget=owner.budget;budget.llm_calls=2;budget.max_llm_calls=3
                initial_cost=budget.estimated_cost_usd
                now[0]+=48.225568
                rt=SimpleNamespace(V13_FAST_TIMEOUT_SECONDS=26,review_uses_remaining_window=True)
                requested=execution._review_timeout(rt,budget,protected=True,compact=True)
                scheduled=owner.provider_timeout("assistant_core_answer_contract_verifier",requested)
                self.assertEqual(requested,51)
                self.assertEqual(scheduled,48)  # recorded 33 + exactly 15
                self.assertLessEqual(scheduled+owner.publication_reserve,budget.remaining())
                self.assertEqual(budget.llm_calls,2)
                self.assertEqual(budget.estimated_cost_usd,initial_cost)
                self.assertEqual(budget.max_estimated_cost_usd,.25)
                self.assertEqual(budget.max_llm_calls,3)
                self.assertEqual(budget.deadline_monotonic,1100.0)
            finally: owner.close()

    def test_clock_starts_at_original_entry_and_late_success_is_still_forbidden(self):
        now=[1000.0]
        with patch.object(request_completion,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(request_budget,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(self.m,"ASSISTANT_CORE_ASK_DEADLINE_SECONDS",100):
            owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id="offline-company"),started=990.0)
            try:
                self.assertEqual(owner.budget.deadline_monotonic,1090.0)
                now[0]=1089.0
                with self.assertRaises(request_budget._V13BudgetExceeded):
                    owner.provider_timeout("assistant_core_answer_contract_verifier",100)
                self.assertEqual(owner.budget.llm_calls,0)
                now[0]=1090.0
                with self.assertRaises(request_budget._V13BudgetExceeded):
                    owner.finish({"ok":True,"status":"answered","answer":"late"})
            finally: owner.close()

    def test_publication_metadata_cannot_release_a_late_answer(self):
        now=[1000.0]
        def attach(response,budget,**kwargs):
            now[0]=1100.01
            return dict(response)
        with patch.object(request_completion,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(request_budget,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(self.m,"ASSISTANT_CORE_ASK_DEADLINE_SECONDS",100):
            runtime=replace(self.m._assistant_core_request_flow_runtime(),_assistant_core_attach_runtime_meta=attach)
            owner=request_completion.RequestCompletion(runtime=runtime,
                payload=SimpleNamespace(company_id="offline-company"),started=now[0])
            try:
                now[0]=1099.9
                with self.assertRaises(request_budget._V13BudgetExceeded):
                    owner.finish({"ok":True,"status":"answered","answer":"too late after metadata"})
            finally: owner.close()

    def test_parent_control_and_explicit_provider_limits_remain_stricter(self):
        now=[1000.0]
        with patch.object(request_completion,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(request_budget,"_monotonic",side_effect=lambda:now[0]), \
             patch.object(self.m,"ASSISTANT_CORE_ASK_DEADLINE_SECONDS",100):
            control=request_budget._RequestControl(60)
            token=request_budget._REQUEST_CONTROL_CTX.set(control)
            owner=None
            try:
                owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                    payload=SimpleNamespace(company_id="offline-company"),started=now[0])
                now[0]+=48
                self.assertEqual(owner.budget.remaining(),12)
                runtime=SimpleNamespace(V13_FAST_TIMEOUT_SECONDS=15,review_uses_remaining_window=False)
                self.assertEqual(execution._review_timeout(runtime,owner.budget,protected=True,compact=True),15)
                self.assertEqual(owner.provider_timeout("assistant_core_answer_contract_verifier",15),9)
            finally:
                if owner: owner.close()
                request_budget._REQUEST_CONTROL_CTX.reset(token)

    def test_same_complete_family_does_not_prove_equivalence_of_review_extensions(self):
        # Concrete counterexample to skipping broad refine merely because the
        # family and deterministic manual support are unchanged. A scoped extra
        # page can carry a distinct condition for the independent reviewer.
        primary=[{"citation_id":"procedure:p1","bubble_document_id":"procedure:p1","chunk_full":"Install the cartridge."},
                 {"citation_id":"step:s1","bubble_document_id":"step:s1","chunk_full":"Secure the housing."}]
        condition={"citation_id":"manual:p9","bubble_document_id":"manual","chunk_full":"Do not start movement until the guard latch is confirmed."}
        render=lambda rows,**kw:"\n\n".join(row["chunk_full"] for row in rows)
        fast=review_evidence.compile_packet(primary=primary,extension=[],render=render,max_records=18,max_context_chars=22000)
        broad=review_evidence.compile_packet(primary=primary,extension=[condition],render=render,max_records=18,max_context_chars=22000)
        self.assertEqual(fast.required_count,broad.required_count)
        self.assertEqual(fast.rows,tuple(primary))
        self.assertIn(condition,broad.rows)
        self.assertNotEqual(fast.sources,broad.sources)
        self.assertFalse(broad.diagnostic(reused=False)["semantic_coverage_proven"])


if __name__ == "__main__": unittest.main(verbosity=2)
