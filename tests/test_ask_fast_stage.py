"""Run the real staged model dispatch, token proof and native ledger offline."""
from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.infrastructure import ask_fast_stage as stage, ask_input_count as counter
from machinemind.infrastructure import request_budget as budgets
import test_ask_exact_input_count as exact


class AskFastStageTests(unittest.TestCase):
    setUpClass = classmethod(exact.ExactInputCountTests.setUpClass.__func__)
    tearDownClass = classmethod(exact.ExactInputCountTests.tearDownClass.__func__)
    activate = exact.ExactInputCountTests.activate

    def setUp(self):
        exact.ExactInputCountTests.setUp(self)
        self.stack.enter_context(patch.dict("os.environ", {stage.FLAG: stage.VERSION}))
        self.tier = "fast"
        self.usage_input = 100
        self.usage_output = 40
        self.reported_model = None
        self.count_error = False

    def post(self, url, **kw):
        if url == counter.ENDPOINT:
            if self.count_error:
                self.posts.append((url, deepcopy(kw)))
                return SimpleNamespace(status_code=400, close=lambda: None)
            return exact.ExactInputCountTests.post(self, url, **kw)
        self.assertEqual(url, counter.CREATE_ENDPOINT)
        self.posts.append((url, deepcopy(kw)))
        if self.on_create:
            self.on_create(kw)
        sent = kw["json"]
        value = {"status": "completed", "model": self.reported_model or sent["model"],
                 "output_text": '{"answer":"supported"}',
                 "usage": {"input_tokens": self.usage_input, "output_tokens": self.usage_output,
                           "input_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 8}}}
        if self.tier is not None:
            value["service_tier"] = self.tier
        return SimpleNamespace(status_code=200, json=lambda: value)

    def call(self, *, purpose=stage.REVIEW, permit=True, models=("gpt-5.6-terra",), size=100,
             router_schema=False, output=3200, full=True):
        schema = deepcopy(exact.SCHEMA)
        if router_schema:
            schema["name"] = "machinemind_ask_compact_router_v1"
        with stage.allow(purpose, eligible=permit):
            return self.m._v13_json_models([{"role": "user", "content": "é" * size}], models=list(models),
                json_schema=schema, effort="none" if purpose == stage.WRITER else "low", reasoning_mode="",
                timeout=25, max_output_tokens=output, company_id="offline-company", purpose=purpose,
                require_full_output=full)

    def test_three_actual_roles_keep_models_effort_shape_cost_limits_and_settlement(self):
        budget = self.activate()
        limits = (budget.max_estimated_cost_usd, budget.max_llm_calls, budget.deadline_monotonic)
        self.assertEqual(limits[0], .25)
        rows = ((stage.ROUTER, ("gpt-5.6-luna",), "gpt-6-luna", 3000, False, "fast", (1, 6)),
                (stage.WRITER, ("gpt-5.6-terra",), "gpt-6.1-terra", 6000, False, "priority", (2.5, 15)),
                (stage.REVIEW, ("gpt-5.6-terra", "gpt-5.6-luna"), "gpt-6.1-terra", 3200, True, "default", (2.5, 15)))
        for purpose, chosen, model, cap, full, tier, rates in rows:
            with self.subTest(purpose=purpose):
                self.tier = tier
                self.assertEqual(self.call(purpose=purpose, models=chosen, output=cap,
                    router_schema=purpose == stage.ROUTER, full=full), ({"answer": "supported"}, model))
                payload = self.posts[-1][1]["json"]
                receipt = budget.call_log[-1]
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["max_output_tokens"], cap)
                self.assertEqual(payload["reasoning"]["effort"], "none" if purpose == stage.WRITER else "low")
                self.assertEqual(payload["text"]["format"]["schema"], exact.SCHEMA["schema"])
                self.assertEqual(payload["service_tier"], "fast")
                self.assertEqual(payload["prompt_cache_options"], {"ttl": "30m"})
                self.assertIs(payload["store"], False)
                self.assertEqual(receipt["configured_model"], chosen[0])
                self.assertEqual(receipt["provider_model"], model)
                self.assertEqual(receipt["provider_service_tier"], tier)
                self.assertEqual((receipt["input_rate"], receipt["output_rate"]), rates)
                self.assertAlmostEqual(receipt["estimated_cost_usd"],
                    ((80 + 12 * .10 + 8 * 1.25) * rates[0] + 40 * rates[1]) / 1_000_000)
        self.assertEqual(len(self.posts), 3)
        self.assertTrue(budget.public_meta()["accounting_complete"])
        self.assertEqual(limits, (budget.max_estimated_cost_usd, budget.max_llm_calls, budget.deadline_monotonic))

    def test_stage_flag_scope_unknown_configuration_and_router_contract_are_exact(self):
        self.activate()
        for settings in (
            {"purpose": stage.WRITER, "permit": False},
            {"purpose": stage.REVIEW, "permit": False},
            {"purpose": stage.ROUTER, "router_schema": False, "models": ("gpt-5.6-luna",)},
            {"purpose": stage.ROUTER, "router_schema": True, "models": ("gpt-5.6-luna", "gpt-5.6-terra")},
            {"purpose": "assistant_core_root_cause_adjudicator"},
            {"purpose": "smart_diagnostic_independent_review"},
            {"purpose": "legacy_chat_json"},
            {"models": ("gpt-5.6-terra", "gpt-5.6-unverified")},
        ):
            with self.subTest(settings=settings):
                self.budget.llm_calls = 0  # independent dispatch fixtures
                self.budget.call_log.clear()
                self.budget.estimated_cost_usd = 0
                self.call(**settings)
                self.assertEqual(self.posts[-1][1]["json"]["service_tier"], "default")
                self.assertNotIn("prompt_cache_options", self.posts[-1][1]["json"])
        with patch.dict("os.environ", {stage.FLAG: "unrecognized"}):
            self.call()
        self.assertEqual(self.posts[-1][1]["json"]["service_tier"], "default")

    def test_stage_setup_cannot_create_protected_owner_or_bypass_standard_envelope(self):
        self.activate(protected=False)
        self.call()
        self.assertEqual(self.posts[-1][1]["json"]["model"], "gpt-5.6-terra")
        self.assertEqual(self.posts[-1][1]["json"]["service_tier"], "default")

    def test_actual_count_proof_binds_mapped_model_tier_and_cache_ttl_to_single_paid_call(self):
        budget = self.activate()
        budget.estimated_cost_usd = .12
        self.assertEqual(self.call(size=40000)[1], "gpt-6.1-terra")
        self.assertEqual([url for url, _ in self.posts], [counter.ENDPOINT, counter.CREATE_ENDPOINT])
        counted = json.loads(self.posts[0][1]["data"])
        sent = self.posts[1][1]["json"]
        self.assertEqual(counted["model"], "gpt-6.1-terra")
        self.assertNotIn("service_tier", counted)
        self.assertNotIn("prompt_cache_options", counted)
        self.assertEqual(sent["max_output_tokens"], 3200)
        row = budget.call_log[0]
        self.assertAlmostEqual(row["reserved_cost_usd"], 7000 * 2.5 * 1.25 / 1_000_000 + .048)
        self.assertEqual(row["input_count_sent_create_sha256"], counter._sha(sent))
        for changed in ({**sent, "service_tier": "default"}, {**sent, "prompt_cache_options": {"ttl": "12h"}}):
            self.assertNotEqual(row["input_count_sent_create_sha256"], counter._sha(changed))
            with self.assertRaises(ValueError):
                counter._projection(changed)

    def test_unsupported_exact_count_is_unsent_under_the_same_native_cap(self):
        budget = self.activate()
        budget.estimated_cost_usd = .12
        self.count_error = True
        with self.assertRaises(budgets._V13BudgetExceeded):
            self.call(size=40000)
        self.assertEqual([url for url, _ in self.posts], [counter.ENDPOINT])
        self.assertEqual(budget.llm_calls, 0)
        self.assertEqual(budget.committed_cost_usd, .12)

    def test_exact_count_does_not_force_a_full_review_over_the_existing_cap(self):
        budget = self.activate()
        budget.estimated_cost_usd = .17
        self.count_tokens = 11000   # .034375 input + .048 full output > remaining .08
        with self.assertRaises(budgets._V13BudgetExceeded):
            self.call(size=40000)
        self.assertEqual([url for url, _ in self.posts], [counter.ENDPOINT])
        self.assertEqual(budget.llm_calls, 0)
        self.assertEqual(budget.committed_cost_usd, .17)

    def test_input_outside_the_short_context_tariff_stays_standard(self):
        budget = self.activate()
        self.tier = "default"
        self.assertEqual(self.call(size=101000)[1], "gpt-5.6-terra")
        self.assertEqual([url for url, _ in self.posts], [counter.ENDPOINT, counter.CREATE_ENDPOINT])
        self.assertNotIn("prompt_cache_options", self.posts[-1][1]["json"])
        self.assertEqual(budget.call_log[0]["pricing_basis"], "configured_model_rates")

    def test_provider_usage_over_reservation_records_and_rejects_overrun(self):
        budget = self.activate()
        self.usage_output = 3300
        with self.assertRaisesRegex(budgets._V13BudgetExceeded, "exceeded the reservation"):
            self.call(models=("gpt-5.6-terra", "gpt-5.6-luna"))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(budget.call_log[0]["output_tokens"], 3300)
        self.assertIn("provider_tokens_exceeded_limit", budget.accounting_anomalies)
        self.assertFalse(budget.public_meta()["accounting_complete"])

    def test_missing_or_unsupported_actual_tier_retains_uncertain_amount_without_fallback(self):
        self.activate()
        for tier in (None, "scale", "unrecognized"):
            with self.subTest(tier=tier):
                self.tier = tier
                before = len(self.posts)
                with self.assertRaisesRegex(budgets._V13BudgetExceeded, "accounting is uncertain"):
                    self.call(models=("gpt-5.6-terra", "gpt-5.6-luna"))
                row = self.budget.call_log[-1]
                self.assertEqual(len(self.posts), before + 1)
                self.assertEqual(row["accounting_state"], "uncertain")
                self.assertGreater(row["reserved_cost_usd"], 0)
        self.assertFalse(self.budget.public_meta()["accounting_complete"])
        self.assertEqual(self.budget.retry_allowance_calls, 0)

    def test_timeout_stops_after_one_paid_attempt_and_keeps_reservation(self):
        budget = self.activate()
        def timeout(_):
            raise TimeoutError("offline")
        self.on_create = timeout
        with self.assertRaisesRegex(budgets._V13BudgetExceeded, "accounting is uncertain"):
            self.call(models=("gpt-5.6-terra", "gpt-5.6-luna"))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(budget.committed_cost_usd, budget.call_log[0]["reserved_cost_usd"])
        self.assertEqual(budget.call_log[0]["accounting_state"], "uncertain")

    def test_unknown_provider_model_does_not_spend_with_fallback(self):
        budget = self.activate()
        self.reported_model = "different-model"
        with self.assertRaises(budgets._V13BudgetExceeded):
            self.call(models=("gpt-5.6-terra", "gpt-5.6-luna"))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(budget.call_log[0]["accounting_state"], "uncertain")
        self.assertNotIn("provider_model", budget.call_log[0])

    def test_real_owned_core_permits_finite_writer_and_first_review_only(self):
        import test_ask_router_family_context as route
        import test_ask_procedure_context_recovery as replay

        observed = []
        original_core = route.RouterFixture.run_core

        def run(fixture):
            phases = []
            original_reply, original_reviewer = fixture.reply, fixture.reviewer

            def reply(value, model, messages, kwargs):
                if kwargs["purpose"] == stage.WRITER:
                    phases.append((stage.WRITER, stage._PERMIT.get()))
                return original_reply(value, model, messages, kwargs)

            def reviewer(messages, **kwargs):
                phases.append((stage.REVIEW, stage._PERMIT.get()))
                return original_reviewer(messages, **kwargs)

            fixture.reply, fixture.reviewer = reply, reviewer
            result = original_core(fixture)
            observed.append((fixture.manual is not None, result["status"], phases))
            self.assertIsNone(stage._PERMIT.get())
            return result

        case = replay.AskProcedureContextRecoveryTests()
        case.m = self.m
        with patch.object(route.RouterFixture, "run_core", run):
            case.test_actual_r7_required_parent_type_keeps_complete_family_and_manual_support()
        self.assertEqual(observed, [
            (False, "answered", [(stage.WRITER, stage.WRITER), (stage.REVIEW, stage.REVIEW)]),
            (True, "answered", [(stage.WRITER, None), (stage.REVIEW, stage.REVIEW)]),
            (False, "no_sources", [(stage.WRITER, stage.WRITER), (stage.REVIEW, stage.REVIEW)]),
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
