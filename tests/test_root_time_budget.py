"""Root scheduling with real orchestration/native budgets and no provider I/O."""
from copy import deepcopy
from contextvars import Context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import request_budget as native, openai_transport
from machinemind.retrieval import evidence_orchestration as orchestration
from machinemind.retrieval import root_time_budget as policy


class RootTimeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(native, "_monotonic", lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.budget = native._V13RequestBudget("root_cause")
        self.budget.deadline_seconds = 92
        self.budget.deadline_monotonic = 92.0
        self.request = SimpleNamespace(requested_mode="root_cause", company_id="fixture-company",
            machine_id="fixture-machine", ai_scope="machine", response_language="en", query="A fixture fault")
        facets = [SimpleNamespace(facet=f"facet-{i}", answer_type="explanation", must_cover=True,
            dense_queries=(f"query-{i}",), lexical_queries=(), exact_terms=(), preferred_source_types=())
            for i in range(4)]
        self.decision = SimpleNamespace(effective_mode="root_cause", dense_queries=(), lexical_queries=(),
            exact_terms=(), required_facets=tuple(f.facet for f in facets), facet_queries=facets,
            request_kind="fault_diagnostic", information_task="fault_diagnostic",
            required_answer_types=("explanation",), missing_information=())
        self.baseline = {"candidates": [{"citation_id": "neutral-original", "text": "Complete original evidence"}],
                         "citations": [{"citation_id": "neutral-original", "text": "Complete original evidence"}],
                         "plan": {}}

    def optional(self, run, stage="facet_refinement", retrieval=None):
        return policy.optional_retrieval(request=self.request, decision=self.decision,
            retrieval=self.baseline if retrieval is None else retrieval, budget=self.budget,
            producer_timeout=26, review_timeout=20, stage=stage, run=run)

    def refine(self, durations):
        calls = []
        def retrieve(**kwargs):
            index = len(calls)
            calls.append(kwargs)
            self.now += durations[index]
            return {"candidates": [{"citation_id": f"facet-source-{index}", "text": "Whole supported excerpt"}]}
        def merge(groups):
            rows = {}
            for group in groups:
                for row in group:
                    rows.setdefault(row["citation_id"], deepcopy(row))
            return list(rows.values())
        runtime = SimpleNamespace(ASSISTANT_CORE_MAX_FACETS=4, V13_DENSE_QUERY_LIMIT=4,
            V13_LEXICAL_QUERY_LIMIT=6, V13_MAX_EVIDENCE_ITEMS_ASK=10, V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE=10,
            _assistant_core_candidate_source_type=lambda row: "manual",
            _assistant_core_facet_candidate_confidence=lambda **kwargs: {"credible": True, "score": 0.8},
            _assistant_core_merge_facet_candidates=merge,
            _assistant_core_retrieval_query=lambda request: request.query,
            _assistant_core_scope_value=lambda request, key: None,
            _dedup_text_values=lambda values, limit: list(dict.fromkeys(values))[:limit],
            _v13_current_budget=lambda: self.budget, _v13_evidence_metrics=lambda rows: {},
            _v13_fallback_plan=lambda query: {}, _v13_initial_retrieval=retrieve,
            _v13_score_candidates=lambda query, rows, **kwargs: rows)
        def run(retrieval):
            return orchestration.assistant_core_refine_retrieval(
                self.request, retrieval, self.decision, runtime=runtime)
        return run, calls

    def test_measured_four_pass_timing_starves_old_path_but_preserves_new_final_window(self):
        durations = [17.703, 14.750, 13.922, 15.219]
        self.now = 22.328
        old, old_calls = self.refine(durations)
        old(deepcopy(self.baseline))
        self.assertEqual(len(old_calls), 4)
        self.assertLess(self.budget.remaining(), 10)
        self.now = 22.328
        run, calls = self.refine(durations)
        result = self.optional(run)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(self.budget.remaining(), 50)
        self.assertEqual({c["citation_id"] for c in result["candidates"]},
                         {"neutral-original", "facet-source-0"})
        self.assertEqual(self.baseline["candidates"][0]["text"], "Complete original evidence")
        runs = result["assistant_core_facet_retrieval"]
        self.assertEqual([r["status"] for r in runs], ["completed", "skipped_budget"])
        self.assertEqual(result["plan"]["required_facets"], list(self.decision.required_facets))
        for call in calls:
            self.assertEqual((call["company_id"], call["machine_id"], call["ai_scope"]),
                             (self.request.company_id, self.request.machine_id, "machine"))
        self.now += 1.344  # Observed evidence preparation; no extra sources invented.
        with policy.producer_window(request=self.request, decision=self.decision,
                budget=self.budget, review_timeout=20):
            self.assertGreaterEqual(self.budget._timeout(26), 24)
            self.now += self.budget._timeout(26)
        self.assertGreaterEqual(self.budget.remaining(), 24)
        self.assertIsNone(policy.current_window())

    def test_assurance_cannot_spend_the_reserve_after_completed_refinement(self):
        self.now = 40.031
        calls = []
        result = self.optional(lambda rows: calls.append(rows), stage="diagnostic_assurance")
        self.assertEqual(calls, [])
        self.assertEqual(result["candidates"], self.baseline["candidates"])
        self.assertEqual(result["assistant_core_root_time_budget"]["stages"][0]["status"],
                         "skipped_reserved_time")

    def test_expired_optional_work_cannot_mutate_baseline_or_refresh_its_deadline(self):
        self.now = 20
        original = deepcopy(self.baseline)
        observed_deadlines = []
        def run(rows):
            window = policy.current_window()
            observed_deadlines.append(window.deadline)
            rows["candidates"][0]["text"] = "Partial unreturned work"
            self.now += 23
            observed_deadlines.append(window.deadline)
            window.check()
        result = self.optional(run)
        self.assertEqual(observed_deadlines, [42.0, 42.0])
        self.assertEqual(self.baseline, original)
        self.assertEqual(result["candidates"], original["candidates"])
        self.assertEqual(result["assistant_core_root_time_budget"]["stages"][0]["status"],
                         "optional_deadline_reached")
        self.assertIsNone(policy.current_window())
        self.assertEqual(self.budget.deadline_monotonic, 92)

    def test_non_time_failure_keeps_identity_and_restores_context(self):
        error = native._V13BudgetExceeded("committed cost budget exhausted")
        def run(rows):
            raise error
        with self.assertRaises(native._V13BudgetExceeded) as caught:
            self.optional(run)
        self.assertIs(caught.exception, error)
        self.assertIsNone(policy.current_window())
        self.assertEqual(self.budget.remaining(), 92)

    def test_copy_time_is_inside_fixed_optional_deadline_and_expiry_keeps_baseline(self):
        original = deepcopy(self.baseline)
        for started, copy_seconds, expected_runs in ((20, 2, 1), (23, 20, 0)):
            with self.subTest(copy_seconds=copy_seconds):
                self.now = started
                deadlines, calls = [], []
                def slow_copy(rows):
                    window = policy.current_window()
                    self.assertIsNotNone(window)
                    deadlines.append(window.deadline)
                    copied = deepcopy(rows)
                    copied["candidates"][0]["text"] = "Private copied evidence"
                    self.now += copy_seconds
                    deadlines.append(window.deadline)
                    return copied
                def run(rows):
                    calls.append(rows)
                    deadlines.append(policy.current_window().deadline)
                    return rows
                with patch.object(policy, "deepcopy", slow_copy):
                    result = self.optional(run)
                self.assertEqual(len(calls), expected_runs)
                self.assertTrue(all(deadline == 42.0 for deadline in deadlines))
                self.assertEqual(self.baseline, original)
                if not expected_runs:
                    self.assertEqual(result["candidates"], original["candidates"])
                    self.assertEqual(result["assistant_core_root_time_budget"]["stages"][0]["status"],
                                     "optional_deadline_reached")
                self.assertIsNone(policy.current_window())
                self.assertEqual(self.budget.remaining(), 92 - self.now)
                self.assertEqual(self.budget.llm_calls, 0)
                self.assertEqual(self.budget.embedding_calls, 0)

    def test_ask_smart_and_rerouted_requests_are_exact_pass_through(self):
        for requested, effective in (("ask", "ask"), ("smart_diagnostic", "smart_diagnostic"),
                                     ("root_cause", "ask"), ("ask", "root_cause")):
            with self.subTest(requested=requested, effective=effective):
                self.request.requested_mode = requested
                self.decision.effective_mode = effective
                seen, result = [], object()
                def run(rows):
                    seen.append(rows)
                    self.assertIsNone(policy.current_window())
                    return result
                self.assertIs(self.optional(run), result)
                self.assertIs(seen[0], self.baseline)
                with policy.producer_window(request=self.request, decision=self.decision,
                        budget=self.budget, review_timeout=20):
                    self.assertIsNone(policy.current_window())

    def test_actual_refinement_behavior_for_ask_and_smart_is_unchanged(self):
        for mode, expected_calls in (("ask", 1), ("smart_diagnostic", 4)):
            with self.subTest(mode=mode):
                self.request.requested_mode = self.decision.effective_mode = mode
                self.now = 22.328
                baseline_run, baseline_calls = self.refine([1, 1, 1, 1])
                expected = baseline_run(deepcopy(self.baseline))
                self.now = 22.328
                run, calls = self.refine([1, 1, 1, 1])
                actual = self.optional(run, retrieval=deepcopy(self.baseline))
                self.assertEqual(actual, expected)
                self.assertEqual(calls, baseline_calls)
                self.assertEqual(len(calls), expected_calls)

    def test_low_time_producer_is_refused_before_any_dispatch(self):
        self.now = 86.781
        calls = []
        with self.assertRaisesRegex(native._V13BudgetExceeded, "root_producer_deadline_reserve"):
            with policy.producer_window(request=self.request, decision=self.decision,
                    budget=self.budget, review_timeout=20):
                calls.append("provider")
        self.assertEqual(calls, [])
        self.assertEqual(self.budget.llm_calls, 0)
        self.assertEqual(self.budget.committed_cost_usd, 0)

    def test_reservation_follows_configured_timeouts_and_parent_deadline(self):
        self.assertEqual(policy.reserved_final_seconds(producer_timeout=35, review_timeout=20), 59)
        parent, token = native._v13_push_operation_limits(seconds=60)
        try:
            with policy._window(self.budget, 10, optional=True):
                self.assertLessEqual(self.budget.remaining(), 10)
            self.assertEqual(self.budget.remaining(), 60)
        finally:
            native._v13_pop_operation_limits(token)
        self.assertEqual(self.budget.remaining(), 92)

    def test_window_does_not_cross_independent_request_contexts(self):
        with policy._window(self.budget, 20, optional=True) as window:
            self.assertIs(policy.current_window(), window)
            other_request = Context()
            self.assertIsNone(other_request.run(policy.current_window))
            self.assertEqual(other_request.run(policy.optional_transport_timeout, 12), 12)
        self.assertIsNone(policy.current_window())

    def test_real_transport_clips_optional_connect_read_and_never_retries(self):
        calls = []
        class Response:
            status_code = 200
            def json(self):
                return {"ok": True}
        def post(*args, **kwargs):
            calls.append(kwargs)
            return Response()
        def send():
            return openai_transport._send_json_attempt(url="https://fixture.invalid", headers={}, payload={},
                timeout=12, post_fn=post, budget=None, call_index=None, embedding=True,
                cache_hits=0, purpose="embedding")
        with policy._window(self.budget, 10, optional=True):
            send()
            self.assertEqual(calls[-1]["timeout"], 4)
            self.now = 8
            with self.assertRaises(policy.RootWindowExpired):
                send()
        self.assertEqual(len(calls), 1)
        send()
        self.assertEqual(calls[-1]["timeout"], 12)

    def test_optional_expiry_after_reservation_releases_only_unsent_liability(self):
        calls = []
        with policy._window(self.budget, 10, optional=True):
            timeout, index = self.budget.reserve_embedding(model="text-embedding-3-small",
                texts=["fixture input"], requested_timeout=12)
            self.now = 8
            with self.assertRaises(policy.RootWindowExpired):
                openai_transport._send_json_attempt(url="https://fixture.invalid", headers={}, payload={},
                    timeout=timeout, post_fn=lambda *a, **k: calls.append(1), budget=self.budget,
                    call_index=index, embedding=True, cache_hits=0, purpose="embedding")
        self.assertEqual(calls, [])
        self.assertEqual(self.budget.embedding_log[0]["accounting_state"], "not_sent")
        self.assertFalse(self.budget.embedding_log[0]["dispatched"])
        self.assertEqual(self.budget.committed_cost_usd, 0)

    def test_optional_provider_timeout_keeps_uncertain_liability_without_retry(self):
        from requests.exceptions import ReadTimeout
        calls, error = [], ReadTimeout("offline transport fixture")
        def post(*args, **kwargs):
            calls.append(kwargs["timeout"])
            raise error
        with policy._window(self.budget, 10, optional=True):
            timeout, index = self.budget.reserve_embedding(model="text-embedding-3-small",
                texts=["fixture input"], requested_timeout=12)
            with self.assertRaises(ReadTimeout) as caught:
                openai_transport._send_json_attempt(url="https://fixture.invalid", headers={}, payload={},
                    timeout=timeout, post_fn=post, budget=self.budget, call_index=index,
                    embedding=True, cache_hits=0, purpose="embedding")
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [4])
        self.assertEqual(self.budget.embedding_log[0]["accounting_state"], "uncertain")
        self.assertTrue(self.budget.embedding_log[0]["dispatched"])
        self.assertGreater(self.budget.committed_cost_usd, 0)

    def test_actual_main_producer_caller_guards_before_model_and_restores_for_review(self):
        path = Path(__file__).resolve().parents[1] / "main.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_assistant_core_synthesize_root_cause")
        self.request.top_k = 8
        self.request.max_causes = 3
        self.request.debug = False
        self.decision.diagnostic_clues = self.decision.diagnostic_exclusions = []
        calls = []
        def producer(**kwargs):
            calls.append("producer")
            self.assertIsNotNone(policy.current_window())
            self.now += 20
            return {"marker": "unchanged"}
        def review(**kwargs):
            calls.append("review")
            self.assertIsNone(policy.current_window())
            self.assertGreaterEqual(self.budget.remaining(), 24)
            return kwargs["response"]
        namespace = {"AssistantCoreRequest": object, "AssistantCoreDecision": object,
            "_assistant_core_retrieval_query": lambda request: request.query,
            "_v13_rescore_root_candidates": lambda query, rows: rows,
            "_v13_evidence_metrics": lambda rows: {}, "V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE": 10,
            "MODE_ROOT_CAUSE": "root_cause", "_assistant_core_diagnostic_query_profile": lambda request: {},
            "_retrieval_diagnostic_query": SimpleNamespace(reasoning_packet=lambda profile: {}),
            "_v13_current_budget": lambda: self.budget, "V13_FAST_TIMEOUT_SECONDS": 26,
            "_v13_generate_root_cause_response": producer, "_assistant_core_adjudicate_root_cause": review}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        run = namespace[node.name]
        self.now = 86.781
        with self.assertRaises(native._V13BudgetExceeded):
            run(self.request, deepcopy(self.baseline), self.decision)
        self.assertEqual(calls, [])
        self.now = 40
        self.assertEqual(run(self.request, deepcopy(self.baseline), self.decision), {"marker": "unchanged"})
        self.assertEqual(calls, ["producer", "review"])

    def test_actual_main_optional_hooks_install_root_window_and_leave_other_routes_unchanged(self):
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = ("_assistant_core_refine_retrieval", "_assistant_core_root_diagnostic_evidence_assurance")
        for hook_name in names:
            with self.subTest(hook=hook_name):
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == hook_name)
                seen = []
                def underlying(request, retrieval, decision, *, runtime):
                    seen.append((request, retrieval, decision, policy.current_window()))
                    return retrieval
                namespace = {"AssistantCoreRequest": object, "AssistantCoreDecision": object,
                    "V13_FAST_TIMEOUT_SECONDS": 26, "_v13_current_budget": lambda: self.budget,
                    "_assistant_core_refine_retrieval_runtime": lambda: object(),
                    "_retrieval_evidence_orchestration": SimpleNamespace(
                        assistant_core_refine_retrieval=underlying,
                        assistant_core_root_diagnostic_evidence_assurance=underlying,
                        AssistantCoreRootDiagnosticEvidenceAssuranceRuntime=lambda **kwargs: kwargs)}
                for field in orchestration.AssistantCoreRootDiagnosticEvidenceAssuranceRuntime.__dataclass_fields__:
                    namespace.setdefault(field, object())
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                run = namespace[hook_name]
                self.now = 18
                self.request.requested_mode = self.decision.effective_mode = "root_cause"
                result = run(self.request, self.baseline, self.decision)
                self.assertTrue(seen[-1][3].optional)
                self.assertIsNot(seen[-1][1], self.baseline)
                # The real caller reserves the configured 26-second reviewer,
                # not the former artificial 20-second sublimit.
                self.assertEqual(seen[-1][3].deadline, 36)
                self.assertEqual(result["candidates"], self.baseline["candidates"])
                self.assertIsNone(policy.current_window())
                for requested, effective in (("ask", "ask"), ("smart_diagnostic", "smart_diagnostic"),
                                             ("ask", "root_cause"), ("root_cause", "ask")):
                    self.request.requested_mode, self.decision.effective_mode = requested, effective
                    result = run(self.request, self.baseline, self.decision)
                    self.assertIs(result, self.baseline)
                    self.assertIsNone(seen[-1][3])

    def test_review_has_configured_time_and_keeps_native_deadline_and_cost_caps(self):
        self.now = 40
        original = (self.budget.deadline_monotonic, self.budget.max_llm_calls,
                    self.budget.max_estimated_cost_usd)
        with policy.reviewer_window(budget=self.budget):
            self.assertEqual(self.budget._timeout(policy.review_timeout_seconds(26)), 26)
            self.now += 21  # The old 20-second transport cap would have failed.
            self.assertGreater(self.budget.remaining(), 0)
        self.assertEqual(original, (self.budget.deadline_monotonic,
            self.budget.max_llm_calls, self.budget.max_estimated_cost_usd))
        self.assertEqual(self.budget.llm_calls, 0)
        self.assertEqual(self.budget.committed_cost_usd, 0)
        self.assertIsNone(policy.current_window())

    def test_late_review_is_shortened_and_finalization_stays_reserved(self):
        self.now = 80
        with policy.reviewer_window(budget=self.budget):
            self.assertEqual(self.budget._timeout(policy.review_timeout_seconds(26)), 7)
            self.now += 7
        self.assertEqual(self.budget.remaining(), 5)
        self.assertEqual(self.budget.deadline_monotonic, 92)

    def test_no_review_dispatch_when_only_finalization_time_remains(self):
        self.now = 88
        dispatched = []
        with self.assertRaises(native._V13BudgetExceeded):
            with policy.reviewer_window(budget=self.budget):
                dispatched.append(True)
        self.assertEqual(dispatched, [])
        self.assertIsNone(policy.current_window())

    def test_configured_shorter_review_is_respected_and_runtime_ceiling_retained(self):
        self.assertEqual(policy.review_timeout_seconds(12), 12)
        self.assertEqual(policy.review_timeout_seconds(20), 20)
        self.assertEqual(policy.review_timeout_seconds(26), 26)
        self.assertEqual(policy.review_timeout_seconds(100), 35)
        for value in (True, None, 0, -1, 20.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.review_timeout_seconds(value)


    def cap(self, explicit=False, configured=26):
        return policy.review_dispatch_timeout_seconds(configured, use_remaining_window=not explicit)

    def test_default_uses_only_existing_ceiling_explicit_fast_remains_cap(self):
        self.assertEqual(self.cap(), 35)
        for configured, expected in ((12, 12), (20, 20), (26, 26), (100, 35)):
            self.assertEqual(self.cap(explicit=True, configured=configured), expected)
        self.assertEqual(policy.review_dispatch_timeout_seconds(26), 26)
        for invalid in (None, 1, "true"):
            with self.assertRaises(ValueError): policy.review_dispatch_timeout_seconds(26, use_remaining_window=invalid)

    def test_native_remaining_floor_bounds_late_review(self):
        for elapsed, expected in ((38.828, 35), (60.0, 27), (80.0, 7), (80.3, 6)):
            self.now = elapsed
            with self.subTest(elapsed=elapsed), policy.reviewer_window(budget=self.budget):
                self.assertEqual(self.budget._timeout(self.cap()), expected)
            self.assertEqual(self.budget.deadline_monotonic, 92)

    def test_outer_child_deadline_is_not_bypassed(self):
        self.now = 40
        _, token = native._v13_push_operation_limits(seconds=20)
        try:
            with policy.reviewer_window(budget=self.budget):
                self.assertEqual(self.budget._timeout(self.cap()), 15)
        finally: native._v13_pop_operation_limits(token)
        self.assertEqual(self.budget.remaining(), 52)

    def test_shortage_fails_before_any_call_or_liability(self):
        for elapsed in (87.5, 88, 91.9):
            self.now = elapsed; calls = []
            with self.subTest(elapsed=elapsed), self.assertRaises(native._V13BudgetExceeded):
                with policy.reviewer_window(budget=self.budget):
                    self.budget.reserve_call(model="gpt-5.6-terra", purpose="assistant_core_root_cause_adjudicator",
                        requested_timeout=self.cap(), max_output_tokens=1000, messages=[{"role": "user", "content": "fixture"}])
                    calls.append(True)
            self.assertEqual(calls, []); self.assertEqual(self.budget.llm_calls, 0)
            self.assertEqual(self.budget.committed_cost_usd, 0)
            self.assertIsNone(policy.current_window())

    def test_upstream_reservations_and_planner_allowance_unchanged(self):
        self.assertEqual(policy.review_timeout_seconds(26), 26)
        self.assertEqual(policy.reserved_final_seconds(producer_timeout=26, review_timeout=26), 56)
        self.now = 18; seen = []
        def run(retrieval):
            seen.append(policy.current_window().deadline)
            return retrieval
        result = policy.optional_retrieval(request=self.request, decision=self.decision, retrieval={"citations": []},
            budget=self.budget, producer_timeout=26, review_timeout=26, stage="facet_refinement", run=run)
        self.assertEqual(seen, [36.0])
        self.assertEqual(result["assistant_core_root_time_budget"]["stages"][0]["reserved_final_seconds"], 56)
        self.assertEqual(self.budget.llm_calls, 0)

    def test_simulated_28s_review_can_use_idle_time_without_deadline_extension(self):
        from requests.exceptions import ReadTimeout
        for explicit, succeeds in ((True, False), (False, True)):
            self.now = 38.828
            original = (self.budget.deadline_monotonic, self.budget.max_llm_calls, self.budget.max_estimated_cost_usd)
            def simulate():
                with policy.reviewer_window(budget=self.budget):
                    timeout = self.budget._timeout(self.cap(explicit=explicit))
                    self.now += min(28, timeout)
                    if timeout < 28: raise ReadTimeout("fake clock only")
            if succeeds: simulate()
            else:
                with self.assertRaises(ReadTimeout): simulate()
            self.assertEqual(original, (self.budget.deadline_monotonic, self.budget.max_llm_calls, self.budget.max_estimated_cost_usd))
            self.assertGreater(self.budget.remaining(), 4)

    def test_single_fake_timeout_retains_uncertain_liability_no_retry(self):
        from requests.exceptions import ReadTimeout
        self.now = 38.828; calls = []
        def post(*args, **kwargs):
            calls.append(kwargs["timeout"]); self.now += kwargs["timeout"]
            raise ReadTimeout("one fake provider attempt")
        with policy.reviewer_window(budget=self.budget):
            timeout, tokens, index = self.budget.reserve_call(model="gpt-5.6-terra", purpose="assistant_core_root_cause_adjudicator",
                requested_timeout=self.cap(), max_output_tokens=1000, messages=[{"role": "user", "content": "fixture"}])
            with self.assertRaises(ReadTimeout):
                openai_transport._send_json_attempt(url="https://fixture.invalid", headers={}, payload={"model": "gpt-5.6-terra"},
                    timeout=timeout, post_fn=post, budget=self.budget, call_index=index, embedding=False,
                    cache_hits=0, purpose="assistant_core_root_cause_adjudicator")
        self.assertEqual(calls, [35]); self.assertEqual(self.budget.llm_calls, 1)
        self.assertEqual(self.budget.call_log[0]["accounting_state"], "uncertain")
        self.assertGreater(self.budget.committed_cost_usd, 0)
        self.assertEqual(self.budget.deadline_monotonic, 92)

    def test_time_choice_does_not_change_model_token_or_cost_reservation(self):
        outcomes = []
        for configured, explicit in ((26, True), (26, False)):
            self.now = 0
            budget = native._V13RequestBudget("root_cause"); budget.deadline_monotonic = 92
            self.now = 38.828
            with policy.reviewer_window(budget=budget):
                timeout, tokens, index = budget.reserve_call(model="gpt-5.6-terra", purpose="assistant_core_root_cause_adjudicator",
                    requested_timeout=self.cap(explicit=explicit), max_output_tokens=1000,
                    messages=[{"role": "user", "content": "same source protocol"}])
            row = budget.call_log[0]
            outcomes.append((timeout, tokens, row["reserved_cost_usd"], row["model"], budget.max_llm_calls, budget.max_estimated_cost_usd))
        self.assertEqual(outcomes[0][0], 26); self.assertEqual(outcomes[1][0], 35)
        self.assertEqual(outcomes[0][1:], outcomes[1][1:])

    def test_actual_proposed_main_uses_environment_presence_not_value(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_assistant_core_adjudicate_root_cause_grounded")
        assignment = next(n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "review_dispatch_timeout" for t in n.targets))
        for environ, configured, expected in (({}, 26, 35), ({"MM_V13_FAST_TIMEOUT_SECONDS": "26"}, 26, 26), ({"MM_V13_FAST_TIMEOUT_SECONDS": "12"}, 12, 12)):
            namespace = {"root_time_budget": policy, "V13_FAST_TIMEOUT_SECONDS": configured, "os": SimpleNamespace(environ=environ)}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "proposed-root-allocation", "exec"), namespace)
            self.assertEqual(namespace["review_dispatch_timeout"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
