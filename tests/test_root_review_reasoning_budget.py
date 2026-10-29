"""Root review transport uses bounded reasoning without changing its proof gate.

The source-bound main function and evidence proof validator are real. The source
packet is the existing synthetic fixture; HTTP replies/time are deterministic.
This tests scheduling and protocol, not model quality or real-world latency.
"""
import ast
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from requests.exceptions import ReadTimeout
from machinemind.infrastructure import openai_transport, request_budget
from machinemind.retrieval import root_response_quality
import test_root_response_quality as existing


def deny(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class RootReviewReasoningBudgetTests(unittest.TestCase):
    def dispatch(self, *, timeout=26, output_cap=3000, failure=None, exhausted=False):
        harness = existing.RootReviewMainIntegrationTests('runTest')
        harness.setUp()
        now = [1000.0]
        posts, calls = [], []
        with ExitStack() as stack:
            for name in ('socket.create_connection', 'socket.socket.connect',
                         'requests.sessions.Session.request'):
                stack.enter_context(patch(name, deny))
            stack.enter_context(patch.object(request_budget, '_RUNTIME_GLOBALS', {}))
            stack.enter_context(patch.object(request_budget, '_monotonic', side_effect=lambda: now[0]))
            budget = request_budget._V13RequestBudget('root_cause')
            if exhausted:
                # Real reserve_call must reject before HTTP; no call credit added.
                budget.max_estimated_cost_usd = 0.000001
            before = (budget.deadline_monotonic, budget.max_llm_calls,
                      budget.max_estimated_cost_usd, budget.base_max_llm_calls)
            harness.budget = budget
            if failure == 'proof':
                harness.parsed['decisions'][0]['proofs'][0]['check_indices'] = [0, 1, 2]

            def post(url, **kwargs):
                posts.append(deepcopy(kwargs))
                if failure == 'timeout':
                    now[0] += kwargs['timeout'] + 0.025
                    raise ReadTimeout('offline bounded timeout')
                body = {'status': 'completed', 'output_text': json.dumps(harness.parsed),
                        'usage': {'input_tokens': 100, 'output_tokens': 100,
                                  'output_tokens_details': {'reasoning_tokens': 20}}}
                return SimpleNamespace(status_code=200, json=lambda: body)

            def responses(messages, **kwargs):
                return openai_transport.responses_json(messages, **kwargs,
                    api_key='offline-unused', url='https://offline.invalid/responses',
                    post_fn=post, current_budget_fn=lambda: budget,
                    response_text_fn=openai_transport.response_text,
                    safety_identifier_fn=openai_transport.safety_identifier)

            def models(messages, **kwargs):
                calls.append(deepcopy(kwargs))
                return openai_transport.json_models(messages, **kwargs,
                    default_model='gpt-5.6-terra',
                    normalize_models_fn=openai_transport.normalize_model_candidates,
                    current_budget_fn=lambda: budget, responses_json_fn=responses,
                    chat_json_fn=deny, budget_exceeded_type=request_budget._V13BudgetExceeded)

            harness.run_review.__globals__.update(
                _v13_json_models=models, _V13BudgetExceeded=request_budget._V13BudgetExceeded,
                V13_FAST_MODEL='gpt-5.6-terra', V13_FAST_EFFORT='medium',
                V13_FAST_TIMEOUT_SECONDS=timeout, V13_FAST_MAX_OUTPUT_TOKENS=output_cap,
                time_module=SimpleNamespace(monotonic=lambda: now[0]))
            out = harness.run_response()
            self.assertEqual((budget.deadline_monotonic, budget.max_llm_calls,
                              budget.max_estimated_cost_usd, budget.base_max_llm_calls), before)
            self.assertEqual(budget.retry_allowance_calls, 0)
            return harness, out, posts, calls, budget

    def test_actual_transport_low_preserves_schema_sources_limits_and_proof(self):
        h, out, posts, calls, budget = self.dispatch()
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(calls), 1)
        payload = posts[0]['json']
        self.assertEqual(payload['reasoning'], {'effort': 'low', 'context': 'current_turn'})
        self.assertEqual(payload['model'], 'gpt-5.6-terra')
        self.assertEqual(payload['max_output_tokens'], 3000)
        self.assertEqual(posts[0]['timeout'], 20)
        self.assertFalse(posts[0]['allow_redirects'])
        self.assertFalse(payload['store'])
        schema = root_response_quality.review_schema(h.frozen)
        self.assertEqual(payload['text']['format']['schema'], schema['schema'])
        self.assertEqual(payload['text']['format']['name'], schema['name'])
        self.assertTrue(payload['text']['format']['strict'])
        self.assertEqual(payload['input'][0]['content'], root_response_quality.review_instruction())
        for row in h.records:
            self.assertIn(row['text'], payload['input'][1]['content'])
        self.assertIn('Adjust the clamp under load with guards open.', payload['input'][1]['content'])
        self.assertNotIn('guards open', str(out['possible_causes']))
        self.assertEqual(budget.llm_calls, 1)
        self.assertEqual(budget.call_log[0]['accounting_state'], 'settled')

    def test_configured_smaller_timeout_and_existing_output_ceiling_are_preserved(self):
        _, out, posts, _, budget = self.dispatch(timeout=12, output_cap=4200)
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(posts[0]['timeout'], 12)
        self.assertEqual(posts[0]['json']['max_output_tokens'], 3400)
        self.assertEqual(budget.call_log[0]['timeout_seconds'], 12)

    def test_read_timeout_keeps_uncertain_liability_and_drops_unreviewed_draft_without_retry(self):
        _, out, posts, calls, budget = self.dispatch(failure='timeout')
        self.assertEqual(out['result_code'], 'TIMEOUT')
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(out['recommended_next_checks'], [])
        self.assertEqual(out['citations'], [])
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(calls), 1)
        review = out['meta']['root_review_execution']
        self.assertEqual(review['provider_error'], 'ReadTimeout')
        self.assertEqual(review['elapsed_seconds'], 20.025)
        self.assertFalse(review['provider_response_received'])
        self.assertEqual(review['provider_attempts'], 1)
        row = budget.call_log[0]
        self.assertEqual(row['accounting_state'], 'uncertain')
        self.assertTrue(row['dispatched'])
        self.assertGreater(row['reserved_cost_usd'], 0)
        self.assertEqual(budget.committed_cost_usd, row['reserved_cost_usd'])

    def test_exhausted_real_cost_admission_prevents_dispatch(self):
        _, out, posts, calls, budget = self.dispatch(exhausted=True)
        self.assertEqual(out['result_code'], 'TIMEOUT')
        self.assertEqual(out['meta']['root_review_execution']['validation_error'], 'validation_budget_exceeded')
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(posts, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.llm_calls, 0)
        self.assertEqual(budget.call_log, [])

    def test_invalid_proof_still_rejects_after_settled_response_without_retry(self):
        _, out, posts, _, budget = self.dispatch(failure='proof')
        self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(out['possible_causes'], [])
        self.assertIn('invalid_check_selection_indices', out['meta']['root_review_execution']['validation_error'])
        self.assertEqual(len(posts), 1)
        self.assertEqual(budget.call_log[0]['accounting_state'], 'settled')

    def test_other_root_adjudicator_path_keeps_configured_effort(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_assistant_core_adjudicate_root_cause')
        calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == '_v13_json_models']
        self.assertEqual(len(calls), 1)
        effort = next(kw.value for kw in calls[0].keywords if kw.arg == 'effort')
        self.assertIsInstance(effort, ast.Name)
        self.assertEqual(effort.id, 'V13_FAST_EFFORT')


if __name__ == '__main__':
    unittest.main(verbosity=2)
