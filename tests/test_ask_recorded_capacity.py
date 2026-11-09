"""Replay observed accounting states through the real protected ASK review boundary.

The historical rejected payloads were not retained. The source/prompt fixture is
therefore synthetic, not a claim that the exact N01 or A02 request now completes.
The captured liabilities, elapsed time and distinction between an unknown router
timeout and a settled semantic rejection are real sanitized observations.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, request_completion
from machinemind.ask.numeric_premise_contract import NUMERIC_PREMISE_CORRECTION_INSTRUCTION
from machinemind.infrastructure import openai_transport as transport, request_budget as budgets


def deny(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


RECORDED = {
    'UI-N-ASK-01': {
        'known': .01319358, 'elapsed': 53.26623, 'calls': 2,
        'uncertain': .151259375,
        'unknown_row': {'model': 'gpt-5.6-terra',
            'purpose': 'assistant_core_v2_semantic_router',
            'timeout_seconds': 15, 'max_output_tokens': 3000,
            'reserved_input_tokens': 34003, 'reserved_cost_usd': .151259375,
            'accounting_state': 'uncertain', 'dispatched': True, 'failed': True},
    },
    'UI-A-ASK-02': {
        'known': .13060482, 'elapsed': 64.406, 'calls': 3, 'uncertain': 0.,
        # Three completed attempts: router, writer and the first semantic judge.
        # That judge reported three missing facets/two types; this state is not
        # an uncertain transport failure and may not be labelled as one.
        'first_review_missing_facets': 3, 'first_review_missing_types': 2,
    },
}


class AskRecordedCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {
            'MM_USAGE_ENFORCEMENT': 'off', 'AI_INTERNAL_SECRET': 'offline-only',
            'OPENAI_API_KEY': 'offline-unused', 'MM_INGEST_LEDGER_AUTO_DDL': '0'}))
        for target in ('socket.create_connection', 'socket.socket.connect',
                'requests.sessions.Session.request', 'psycopg2.connect'):
            cls.guard.enter_context(patch(target, deny))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def replay(self, case, *, outcome='pass', budget_limit=.25,
            guarded=True, active=True, mode='ask', exhausted=None, source_first=False,
            malformed_reply=False):
        m = self.m
        recorded = RECORDED[case]
        now = [1000.]
        with patch.object(budgets, '_monotonic', side_effect=lambda: now[0]), \
                patch.object(request_completion, '_monotonic', side_effect=lambda: now[0]):
            request = m.AssistantCoreRequest(
                query='Which safety condition applies before insertion?',
                requested_mode=mode, response_language='en', company_id='offline-company',
                machine_id='offline-machine', ai_scope='machine_all', top_k=6)
            decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,
                effective_mode=mode, confidence=.99, requested_mode_fit=True,
                evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy='machine_sources_required',
                information_task='procedure_segment', required_facets=('isolation check',),
                required_answer_types=(m.REQ_SAFETY_CONDITIONS,))
            answer = 'Verify that the supply is isolated before insertion.'
            rows = [{'citation_id': 'manual:safety', 'source_type': 'document', 'chunk_full': answer}]
            owner = (request_completion.RequestCompletion(runtime=m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id=request.company_id), started=1000.) if active else None)
            budget = owner.budget if owner else m._assistant_core_new_budget('ask')
            budget.max_estimated_cost_usd = budget_limit
            budget.estimated_cost_usd = recorded['known']
            budget.llm_calls = recorded['calls']
            for number in range(recorded['calls']):
                budget.call_log.append({'call': number + 1, 'accounting_state': 'settled',
                    'dispatched': True, 'reserved_cost_usd': 0.})
            if recorded.get('unknown_row'):
                budget.call_log[0].update(deepcopy(recorded['unknown_row']))
            before_rows = deepcopy(budget.call_log)
            limits = (budget.deadline_monotonic, budget.max_estimated_cost_usd, budget.max_llm_calls)
            now[0] += recorded['elapsed']
            if exhausted == 'deadline':
                now[0] = budget.deadline_monotonic
            elif exhausted == 'calls':
                budget.llm_calls = budget.max_llm_calls
            elif exhausted == 'anomaly':
                budget.accounting_anomalies.append('offline-fixture-anomaly')
            posts, attempted, kwargs_seen = [], [], []
            verdict = {'outcome': outcome, 'answer': answer if outcome == 'pass' else '',
                'reason': 'offline fixture only', 'citation_ids': ['manual:safety'],
                'covered_facets': ['isolation check'] if outcome == 'pass' else [],
                'missing_facets': [] if outcome == 'pass' else ['isolation check'],
                'covered_answer_types': [m.REQ_SAFETY_CONDITIONS] if outcome == 'pass' else [],
                'missing_answer_types': [] if outcome == 'pass' else [m.REQ_SAFETY_CONDITIONS],
                'missing_list_items': []}
            def provider(messages, **kwargs):
                kwargs_seen.append(kwargs)
                protected = guarded and active and mode == 'ask'
                self.assertEqual(NUMERIC_PREMISE_CORRECTION_INSTRUCTION in messages[0]['content'], protected)
                self.assertEqual('require_full_output' in kwargs, protected)
                def delegate(messages, model, **parameters):
                    attempted.append(model)
                    def post(url, **request):
                        posts.append(request['json'])
                        return SimpleNamespace(status_code=200, json=lambda: {
                            'status': 'completed', 'output_text': '{' if malformed_reply else json.dumps(verdict),
                            'usage': {'input_tokens': 100, 'output_tokens': 100}})
                    return transport.responses_json(messages, model=model,
                        api_key='offline-unused', url='https://offline.invalid/responses',
                        post_fn=post, current_budget_fn=lambda: budget,
                        response_text_fn=transport.response_text,
                        safety_identifier_fn=transport.safety_identifier, **parameters)
                return transport.json_models(messages, default_model='gpt-5.6-terra',
                    normalize_models_fn=transport.normalize_model_candidates,
                    current_budget_fn=lambda: budget, responses_json_fn=delegate, chat_json_fn=deny,
                    budget_exceeded_type=budgets._V13BudgetExceeded, **kwargs)
            runtime = replace(m._assistant_core_ask_execution_runtime(),
                evidence_admission=deny if guarded else None, _v13_current_budget=lambda: budget,
                _v13_json_models=provider,
                # The actual rejected historical prompts do not exist. This
                # deterministic documentary fixture exercises the real bound;
                # no tokenizer, provider, price or reservation is mocked.
                _v13_sources_block=lambda *args, **kw: (answer + '\n') * 350)
            result, error = None, None
            try:
                with ExitStack() as scope:
                    scope.enter_context(patch.object(execution, '_admit_input',
                        side_effect=lambda q, data, d, **kw: data))
                    if source_first:
                        scope.enter_context(patch.object(execution._enumeration_review, 'applies', return_value=True))
                    try:
                        result = execution.verify_or_repair_answer(request=request, decision=decision,
                            answer='' if source_first else answer, candidates=rows, runtime=runtime)
                    except budgets._V13BudgetExceeded as exc:
                        error = exc
                self.assertEqual(budget.call_log[:len(before_rows)], before_rows)
                self.assertEqual((budget.deadline_monotonic, budget.max_estimated_cost_usd,
                    budget.max_llm_calls), limits)
                self.assertEqual((budget.retry_allowance_calls, budget.retry_events), (0, []))
                self.assertAlmostEqual(budget.public_meta()['uncertain_cost_usd'], recorded['uncertain'], places=8)
                self.assertLessEqual(budget.committed_cost_usd, budget.max_estimated_cost_usd)
                return result, error, posts, attempted, kwargs_seen, budget.public_meta()
            finally:
                if owner:
                    owner.close()

    def test_neronis_unknown_router_debt_is_retained_when_cheaper_review_fits(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-N-ASK-01')
        self.assertIsNone(error)
        self.assertEqual(attempted, ['gpt-5.6-terra', 'gpt-5.6-luna'])
        self.assertEqual([p['model'] for p in posts], ['gpt-5.6-luna'])
        self.assertEqual(posts[0]['max_output_tokens'], 3200)
        self.assertIs(kwargs[0]['require_full_output'], True)
        self.assertEqual(result['outcome'], 'pass')
        self.assertEqual(meta['llm_calls'], 3)
        self.assertFalse(meta['accounting_complete'])

    def test_aurelis_settled_repair_state_selects_only_affordable_full_model(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-A-ASK-02')
        self.assertIsNone(error)
        self.assertEqual(attempted, ['gpt-5.6-terra', 'gpt-5.6-luna'])
        self.assertEqual([p['model'] for p in posts], ['gpt-5.6-luna'])
        self.assertEqual(result['outcome'], 'pass')
        self.assertEqual(meta['llm_calls'], 4)
        self.assertTrue(meta['accounting_complete'])

    def test_affordable_review_cannot_approve_a_missing_mandatory_source(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-A-ASK-02', outcome='no_sources')
        self.assertIsNone(error)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['missing_facets'], ['isolation check'])
        self.assertEqual(result['missing_answer_types'], [self.m.REQ_SAFETY_CONDITIONS])
        self.assertEqual(len(posts), 1)  # no repeated semantic-approval attempt

    def test_unaffordable_models_never_dispatch_or_erase_uncertain_debt(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-N-ASK-01', budget_limit=.18)
        self.assertIsInstance(error, budgets._V13OutputCapacityInsufficient)
        self.assertEqual(attempted, ['gpt-5.6-terra', 'gpt-5.6-luna'])
        self.assertEqual(posts, [])
        self.assertEqual(meta['llm_calls'], 2)
        self.assertFalse(meta['accounting_complete'])

    def test_malformed_affordable_reply_is_not_an_approval_or_a_free_retry(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-N-ASK-01', malformed_reply=True)
        self.assertIsNone(error)
        self.assertEqual(result['outcome'], 'unavailable')
        self.assertEqual(attempted, ['gpt-5.6-terra', 'gpt-5.6-luna'])
        self.assertEqual([p['model'] for p in posts], ['gpt-5.6-luna'])
        self.assertEqual(meta['llm_calls'], 3)
        self.assertEqual(meta['calls'][-1]['accounting_state'], 'settled')
        self.assertGreater(meta['estimated_cost_usd'], RECORDED['UI-N-ASK-01']['known'])
        self.assertFalse(meta['accounting_complete'])  # original unknown hold

    def test_deadline_calls_and_accounting_anomaly_do_not_trigger_capacity_fallback(self):
        for exhausted in ('deadline', 'calls', 'anomaly'):
            with self.subTest(exhausted=exhausted):
                result, error, posts, attempted, kwargs, meta = self.replay('UI-A-ASK-02', exhausted=exhausted)
                self.assertIsInstance(error, budgets._V13BudgetExceeded)
                self.assertNotIsInstance(error, budgets._V13OutputCapacityInsufficient)
                self.assertEqual(posts, [])
                self.assertNotIn('gpt-5.6-luna', attempted)

    def test_factual_producer_cannot_consume_the_only_slot_needed_by_its_independent_judge(self):
        result, error, posts, attempted, kwargs, meta = self.replay('UI-A-ASK-02', source_first=True)
        self.assertIsInstance(error, budgets._V13BudgetExceeded)
        self.assertIn('mandatory review call slot', str(error))
        self.assertEqual((posts, attempted, kwargs), ([], [], []))
        self.assertEqual(meta['llm_calls'], 3)

    def test_off_legacy_root_and_smart_do_not_acquire_the_new_capacity_policy_or_prompt(self):
        for settings in ({'guarded': False}, {'active': False},
                {'mode': 'root_cause'}, {'mode': 'smart_diagnostic'}):
            with self.subTest(settings=settings):
                result, error, posts, attempted, kwargs, meta = self.replay('UI-A-ASK-02', **settings)
                self.assertNotIn('require_full_output', kwargs[0])
                self.assertEqual(attempted, ['gpt-5.6-terra'])

    def test_observed_cheap_router_cost_leaves_two_full_reviews_with_fixture_receipts(self):
        # Planning counterfactual: N01's completed Luna router cost, then twice
        # A02's observed first-review envelope and aggregate tokens, with two
        # explicit uncached/no-write fixture receipts. Cache detail was not
        # retained, so these are not historical review-cost reproductions or a
        # claim that every future producer output is affordable.
        with patch.object(budgets, '_monotonic', return_value=1000.), \
                patch.object(budgets, '_text_input_bound', return_value=(32124, 7552)):
            budget = self.m._assistant_core_new_budget('ask')
            budget.max_estimated_cost_usd = .25
            budget.estimated_cost_usd = .0131925
            budget.llm_calls = 1
            limits = (budget.deadline_monotonic, budget.max_llm_calls, budget.max_estimated_cost_usd)
            for _ in range(2):
                _, cap, index = budget.reserve_call(model='gpt-5.6-terra', purpose='offline-review',
                    requested_timeout=20, max_output_tokens=3200, messages=[], require_full_output=True)
                self.assertEqual(cap, 3200)
                self.assertLessEqual(budget.committed_cost_usd, .25)
                budget.mark_dispatched(index)
                budget.record_usage(index, 'gpt-5.6-terra', {'input_tokens': 7552, 'output_tokens': 952})
            self.assertEqual(budget.llm_calls, 3)
            self.assertTrue(budget.public_meta()['accounting_complete'])
            input_rate, output_rate = budgets._v13_model_rates('gpt-5.6-terra')
            receipt_cost = (7552 * input_rate + 952 * output_rate) / 1_000_000
            self.assertAlmostEqual(budget.estimated_cost_usd, .0131925 + 2 * receipt_cost, places=8)
            self.assertEqual((budget.deadline_monotonic, budget.max_llm_calls, budget.max_estimated_cost_usd), limits)


if __name__ == '__main__':
    unittest.main(verbosity=2)
