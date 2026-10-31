"""Observe real transport failure branches without changing admission or billing."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import routing_runtime
from machinemind.infrastructure import openai_transport as transport
from machinemind.infrastructure import provider_failures as failures
from machinemind.infrastructure import request_budget as budgets


def deny(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class Response:
    def __init__(self, data=None, status=200, error=None):
        self.data, self.status_code, self.error = data, status, error

    def json(self):
        if self.error is not None:
            raise self.error
        return self.data


class ProviderFailureObservationsTests(unittest.TestCase):
    def setUp(self):
        self.guard = ExitStack()
        self.addCleanup(self.guard.close)
        for target in ("socket.create_connection", "socket.socket.connect", "requests.sessions.Session.request"):
            self.guard.enter_context(patch(target, deny))
        self.guard.enter_context(patch.object(budgets, "_monotonic", return_value=1000.0))
        self.guard.enter_context(patch.object(budgets, "_RUNTIME_GLOBALS", {
            "V13_MAX_ESTIMATED_COST_ASK_USD": .25,
            "V13_MAX_LLM_CALLS_ASK": 4,
            "V13_ASK_DEADLINE_SECONDS": 100,
            "V13_PRICE_TERRA_INPUT": 2.5, "V13_PRICE_TERRA_OUTPUT": 15,
            "V13_PRICE_LUNA_INPUT": 1, "V13_PRICE_LUNA_OUTPUT": 6,
        }))
        self.budget = budgets._V13RequestBudget("ask")
        self.payloads = []
        self.schema = {"name": "observed_reply", "strict": True,
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                       "required": ["ok"], "additionalProperties": False}}

    def dispatch(self, response, model="gpt-5.6-terra", **kwargs):
        def post(url, **request):
            self.assertFalse(request["allow_redirects"])
            self.payloads.append(request["json"])
            if isinstance(response, BaseException):
                raise response
            return response
        return transport.responses_json(
            [{"role": "user", "content": "offline input"}], model=model,
            json_schema=kwargs.get("json_schema", self.schema), effort="low",
            timeout=30, max_output_tokens=1000, company_id="offline-company",
            purpose="assistant_core_answer_contract_verifier", api_key="offline-unused",
            url="https://offline.invalid/responses", post_fn=post,
            current_budget_fn=lambda: self.budget, response_text_fn=transport.response_text,
            safety_identifier_fn=transport.safety_identifier)

    @staticmethod
    def completed(text='{"ok":true}', output=40):
        return Response({"status": "completed", "output_text": text,
            "usage": {"input_tokens": 100, "output_tokens": output}})

    def assert_failure(self, response, code, state="settled", **kwargs):
        with self.assertRaises(Exception) as caught:
            self.dispatch(response, **kwargs)
        row = self.budget.call_log[-1]
        self.assertEqual(row["provider_failure_code"], code)
        self.assertEqual(row["accounting_state"], state)
        self.assertTrue(row["dispatched"])
        self.assertTrue(row["failed"])
        self.assertNotIn("raw-private-provider-text", json.dumps(self.budget.public_meta()))
        self.assertEqual(len(self.payloads), self.budget.llm_calls)
        return caught.exception, row

    def test_observed_output_limit_survives_successful_model_fallback(self):
        reply = Response({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 100, "output_tokens": 1000,
                      "output_tokens_details": {"reasoning_tokens": 700}}})
        def delegate(messages, model, **kwargs):
            return self.dispatch(reply if model.endswith("terra") else self.completed(), model=model)
        limits = (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd)
        parsed, model = transport.json_models([], models=["gpt-5.6-terra", "gpt-5.6-luna"],
            json_schema=self.schema, effort="low", reasoning_mode="", timeout=30,
            max_output_tokens=1000, company_id="offline-company", purpose="review",
            default_model="gpt-5.6-terra", normalize_models_fn=transport.normalize_model_candidates,
            current_budget_fn=lambda: self.budget, responses_json_fn=delegate, chat_json_fn=deny,
            budget_exceeded_type=budgets._V13BudgetExceeded)
        self.assertEqual((parsed, model), ({"ok": True}, "gpt-5.6-luna"))
        self.assertEqual(self.budget.llm_calls, 2)
        self.assertEqual(limits, (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd))
        self.assertEqual(self.budget.output_tokens, 1040)
        self.assertAlmostEqual(self.budget.estimated_cost_usd, .01525 + .00034)
        rows = routing_runtime._provider_attempts(self.budget.public_meta())
        self.assertEqual(rows[0]["provider_failure_code"], "output_limit")
        self.assertEqual(rows[0]["accounting_state"], "settled")
        self.assertNotIn("provider_failure_code", rows[1])
        self.assertTrue(self.budget.public_meta()["accounting_complete"])

    def test_equal_output_cap_is_not_evidence_of_truncation(self):
        self.assertEqual(self.dispatch(self.completed(output=1000)), {"ok": True})
        self.assertNotIn("provider_failure_code", self.budget.call_log[0])
        self.assertEqual(self.payloads[0]["max_output_tokens"], 1000)
        self.assertEqual(self.payloads[0]["reasoning"]["effort"], "low")

    def test_explicit_statuses_and_refusals_have_bounded_codes(self):
        for status, reason, expected in (("incomplete", "content_filter", "content_filter"),
                ("incomplete", "raw-private-provider-text", "response_incomplete"),
                ("failed", None, "response_failed"), ("cancelled", None, "response_cancelled")):
            with self.subTest(status=status, reason=reason):
                data = self.completed().data
                data.update(status=status, incomplete_details={"reason": reason})
                error, _ = self.assert_failure(Response(data), expected)
                self.assertNotIn("raw-private-provider-text", str(error))

    def test_refusal_and_empty_output_settle_usage_without_provider_text(self):
        data = self.completed().data
        data.pop("output_text")
        data["output"] = [{"type": "message", "content": [{"type": "refusal", "refusal": "raw-private-provider-text"}]}]
        error, _ = self.assert_failure(Response(data), "refusal")
        self.assertNotIn("raw-private-provider-text", str(error))
        data["output"] = []
        self.assert_failure(Response(data), "empty_output")

    def test_invalid_output_json_shape_and_conflicting_duplicate_preserve_usage(self):
        self.assert_failure(self.completed("{"), "invalid_json")
        self.assert_failure(self.completed("[]"), "invalid_shape")
        self.assert_failure(self.completed('{"ok":true,"ok":false}'), "duplicate_key",
                            json_schema={**self.schema, "reject_duplicate_keys": True})
        self.assertEqual(self.budget.output_tokens, 120)

    def test_http_errors_with_and_without_json_keep_unknown_reservation(self):
        for status, response in ((429, Response({"error": "raw-private-provider-text"}, 429)),
                (502, Response(status=502, error=ValueError("raw-private-provider-text")))):
            with self.subTest(status=status):
                error, row = self.assert_failure(response, "http_4xx" if status == 429 else "http_5xx", "uncertain")
                self.assertNotIn("raw-private-provider-text", str(error))
                self.assertGreater(row["reserved_cost_usd"], 0)
        self.assertEqual(self.budget.estimated_cost_usd, 0)
        self.assertAlmostEqual(self.budget.committed_cost_usd,
                               sum(row["reserved_cost_usd"] for row in self.budget.call_log))

    def test_timeout_then_success_cannot_settle_previous_unknown(self):
        from requests.exceptions import ReadTimeout
        self.assert_failure(ReadTimeout("raw-private-provider-text"), "timeout", "uncertain")
        reserve = self.budget.call_log[0]["reserved_cost_usd"]
        self.dispatch(self.completed())
        self.assertEqual(self.budget.call_log[0]["accounting_state"], "uncertain")
        self.assertAlmostEqual(self.budget.committed_cost_usd, reserve + self.budget.estimated_cost_usd)
        self.assertFalse(self.budget.public_meta()["accounting_complete"])

    def test_invalid_outer_json_and_shape_remain_unknown(self):
        self.assert_failure(Response(error=json.JSONDecodeError("bad", "raw-private-provider-text", 0)),
                            "invalid_json", "uncertain")
        self.assert_failure(Response([]), "invalid_shape", "uncertain")

    def test_safe_projection_and_unknown_error_never_copy_text(self):
        self.assertEqual(failures.failure_code(RuntimeError("raw-private-provider-text")), "unclassified")
        projection = routing_runtime._provider_attempts({"calls": [
            {"failed": True, "provider_failure_code": "raw-private-provider-text", "error": "private"},
            {"failed": False, "provider_failure_code": "output_limit"},
            {"failed": True, "provider_failure_code": "timeout", "error": "private"}]})
        self.assertNotIn("provider_failure_code", projection[0])
        self.assertNotIn("provider_failure_code", projection[1])
        self.assertEqual(projection[2]["provider_failure_code"], "timeout")
        self.assertNotIn("private", json.dumps(projection))

    def test_chat_finish_reason_is_observed_not_guessed(self):
        for finish, code in (("length", "output_limit"), ("content_filter", "content_filter")):
            with self.subTest(finish=finish), self.assertRaises(failures.ProviderResponseError) as caught:
                transport._chat_content({"choices": [{"finish_reason": finish, "message": {"content": "partial"}}]})
            self.assertEqual(caught.exception.failure_code, code)
        self.assertEqual(transport._chat_content({"choices": [{"finish_reason": "stop",
                         "message": {"content": "valid"}}]}), "valid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
