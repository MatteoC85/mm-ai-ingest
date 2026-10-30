"""Offline owner/transport proof for a second check inside the existing deadline."""
from contextlib import ExitStack
from contextvars import copy_context
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, request_completion as completion
from machinemind.infrastructure import openai_transport, request_budget

PURPOSE = 'assistant_core_answer_contract_verifier'


def deny(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class IndependentReviewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {
            'MM_USAGE_ENFORCEMENT': 'off', 'AI_INTERNAL_SECRET': 'offline-only',
            'OPENAI_API_KEY': 'offline-unused', 'MM_INGEST_LEDGER_AUTO_DDL': '0'}))
        for target in ('socket.create_connection', 'socket.socket.connect',
                       'requests.sessions.Session.request', 'psycopg2.connect'):
            cls.guards.enter_context(patch(target, deny))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def setUp(self):
        self.now = [1000.0]
        self.clocks = ExitStack()
        self.clocks.enter_context(patch.object(completion, '_monotonic', side_effect=lambda: self.now[0]))
        self.clocks.enter_context(patch.object(request_budget, '_monotonic', side_effect=lambda: self.now[0]))
        self.addCleanup(self.clocks.close)

    def owner(self):
        value = completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'), started=self.now[0])
        self.addCleanup(value.close)
        return value

    def dispatch(self, owner, post, models=('gpt-5.6-terra', 'gpt-5.6-luna')):
        def responses(messages, **kwargs):
            return openai_transport.responses_json(messages, **kwargs,
                api_key='offline-unused', url='https://offline.invalid/responses', post_fn=post,
                current_budget_fn=lambda: owner.budget, response_text_fn=openai_transport.response_text,
                safety_identifier_fn=openai_transport.safety_identifier)
        return openai_transport.json_models([{'role': 'user', 'content': 'Offline source fixture.'}],
            models=list(models), json_schema={'name': 'fixture', 'strict': True,
                'schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
            effort='low', reasoning_mode='', timeout=100, max_output_tokens=3200,
            company_id='offline-company', purpose=PURPOSE, default_model='gpt-5.6-terra',
            normalize_models_fn=openai_transport.normalize_model_candidates,
            current_budget_fn=lambda: owner.budget, responses_json_fn=responses,
            chat_json_fn=deny, budget_exceeded_type=request_budget._V13BudgetExceeded)

    def response(self, *, completed):
        body = {'status': 'completed' if completed else 'incomplete', 'output_text': '{}',
            'usage': {'input_tokens': 100, 'output_tokens': 100}}
        return SimpleNamespace(status_code=200, json=lambda: body)

    def test_default_and_unowned_context_leave_existing_scheduling_unchanged(self):
        with completion.independent_review_reserve(required=True):
            self.assertEqual(completion.provider_timeout(PURPOSE, 77), 77)
            owner = self.owner()  # No owner existed when the scope was entered.
            self.assertEqual(completion.provider_timeout(PURPOSE, 100), 97)
        with completion.independent_review_reserve(required=False):
            self.assertEqual(completion.provider_timeout(PURPOSE, 100), 97)
        with completion.independent_review_reserve(required=True):
            for purpose in ('embedding', 'ask_final_synthesis', 'root_cause_review', 'smart_diagnostic_review'):
                self.assertEqual(completion.provider_timeout(purpose, 100), 97)
            with completion.synthesis_review_reserve(required=True):
                self.assertEqual(completion.provider_timeout('ask_final_synthesis', 100), 86)
                self.assertEqual(completion.provider_timeout(PURPOSE, 100), 72)
        self.assertEqual(owner.budget.llm_calls, 0)

    def test_nested_same_owner_cannot_disable_reserve_and_finally_clears_it(self):
        owner = self.owner()
        with self.assertRaisesRegex(ValueError, 'fixture'):
            with completion.independent_review_reserve(required=True):
                with completion.independent_review_reserve(required=False):
                    self.assertEqual(completion.provider_timeout(PURPOSE, 100), 72)
                with completion.independent_review_reserve(required=True):
                    self.assertEqual(completion.provider_timeout(PURPOSE, 100), 72)
                raise ValueError('fixture')
        self.assertEqual(completion.provider_timeout(PURPOSE, 100), 97)
        self.assertEqual(owner.provider_schedules[0]['review_reserve_seconds'], 25)
        self.assertEqual(owner.provider_schedules[-1]['review_reserve_seconds'], 0)

    def test_closed_owner_and_copied_context_do_not_reserve_a_new_request(self):
        old = self.owner()
        with completion.independent_review_reserve(required=True):
            copied = copy_context()
            old.close()
            self.assertEqual(completion.provider_timeout(PURPOSE, 77), 77)
            new = self.owner()
            try:
                self.assertEqual(completion.provider_timeout(PURPOSE, 100), 97)
                with completion.independent_review_reserve(required=True):
                    self.assertEqual(completion.provider_timeout(PURPOSE, 100), 72)
                self.assertEqual(completion.provider_timeout(PURPOSE, 100), 97)
            finally:
                new.close()
        def inactive_in_stale_context():
            self.assertIsNone(completion.current())
            self.assertEqual(completion.provider_timeout(PURPOSE, 77), 77)
        copied.run(inactive_in_stale_context)

    def test_real_fallback_attempts_retain_window_and_next_review_uses_remainder(self):
        owner = self.owner(); budget = owner.budget; calls = []
        limits = (budget.deadline_monotonic, budget.max_estimated_cost_usd)
        def post(url, **kwargs):
            calls.append((kwargs['timeout'], budget.remaining(), kwargs['json']['max_output_tokens']))
            self.now[0] += 12 if len(calls) == 1 else 7
            return self.response(completed=len(calls) == 2)
        with completion.independent_review_reserve(required=True):
            result, model = self.dispatch(owner, post)
        self.assertEqual(result, {}); self.assertEqual(model, 'gpt-5.6-luna')
        self.assertEqual([r[0] for r in calls], [72, 60])
        for timeout, remaining, cap in calls:
            self.assertLessEqual(timeout, remaining - 25 - 3)
            self.assertEqual(cap, 3200)
        self.assertEqual(completion.provider_timeout(PURPOSE, 100), 78)
        self.assertEqual((budget.deadline_monotonic, budget.max_estimated_cost_usd), limits)
        self.assertEqual(budget.llm_calls, 2)
        self.assertTrue(all(r['accounting_state'] == 'settled' for r in budget.call_log))

    def test_late_fallback_is_denied_before_http_and_leaves_second_window(self):
        owner = self.owner(); calls = []
        def post(url, **kwargs):
            calls.append(kwargs['timeout'])
            self.now[0] += 43 if len(calls) == 1 else 29
            return self.response(completed=False)
        with completion.independent_review_reserve(required=True):
            with self.assertRaises(request_budget._V13BudgetExceeded):
                self.dispatch(owner, post, models=('gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.6-sol'))
        self.assertEqual(calls, [72, 29])
        self.assertEqual(owner.budget.llm_calls, 2)
        self.assertEqual(owner.provider_schedules[-1]['admitted'], False)
        self.assertEqual(completion.provider_timeout(PURPOSE, 100), 25)
        self.assertEqual(owner.budget.deadline_monotonic, 1100)

    def test_insufficient_time_and_explicit_lower_caps_never_borrow_reserved_time(self):
        owner = self.owner(); owner.largest_fence_seconds = 4
        with completion.independent_review_reserve(required=True):
            self.assertEqual(completion.provider_timeout(PURPOSE, 15), 15)
            self.now[0] += 61  # 39 remaining - 5.5 publication - 25 independent = 8.5.
            with self.assertRaises(request_budget._V13BudgetExceeded):
                self.dispatch(owner, deny)
        self.assertEqual(owner.budget.llm_calls, 0)
        self.assertEqual(owner.provider_schedules[-1]['timeout_seconds'], 8)
        self.assertEqual(completion.provider_timeout(PURPOSE, 100), 33)
        self.now[0] = 1100.01
        with completion.independent_review_reserve(required=True):
            with self.assertRaises(request_budget._V13BudgetExceeded):
                completion.provider_timeout(PURPOSE, 100)

    def test_transport_recheck_after_reservation_cannot_cross_second_window(self):
        owner = self.owner(); reserve = owner.budget.reserve_call
        def delayed_reservation(**kwargs):
            result = reserve(**kwargs)
            self.now[0] += 65
            return result
        with completion.independent_review_reserve(required=True), \
                patch.object(owner.budget, 'reserve_call', side_effect=delayed_reservation):
            with self.assertRaises(request_budget._V13BudgetExceeded):
                self.dispatch(owner, deny)
        self.assertEqual(owner.budget.llm_calls, 1)
        self.assertEqual(owner.budget.call_log[0]['accounting_state'], 'not_sent')
        self.assertFalse(owner.budget.call_log[0]['dispatched'])
        self.assertEqual(completion.provider_timeout(PURPOSE, 100), 32)
        self.assertEqual(owner.budget.deadline_monotonic, 1100)

    def test_recorded_second_review_bound_uses_remaining_window_and_existing_cost_ceiling(self):
        owner = self.owner(); budget = owner.budget
        limits = (budget.deadline_monotonic,budget.max_estimated_cost_usd,budget.max_llm_calls)
        self.assertEqual(limits,(1100,.25,4))
        # T41's settled router + producer + two embeddings, before the second
        # review. This is an accounting-state replay, not the unretained prompt.
        budget.llm_calls = 2
        budget.estimated_cost_usd = .094844275
        self.now[0] += 52.417
        runtime = SimpleNamespace(V13_FAST_TIMEOUT_SECONDS=20,review_uses_remaining_window=True)
        requested_timeout = execution._review_timeout(runtime,budget,protected=True,compact=False)
        self.assertEqual(requested_timeout,47)
        messages = [{'role':'user','content':''}]
        schema = {'name':'recorded_bound_fixture','strict':True,
                  'schema':{'type':'object','properties':{},'additionalProperties':False}}
        payload = {'model':'gpt-5.6-terra','input':messages,'store':False,
            'reasoning':{'effort':'low','context':'current_turn'},
            'text':{'format':{'type':'json_schema',**schema}},
            'safety_identifier':openai_transport.safety_identifier('offline-company')}
        base = request_budget._text_input_bound(messages,payload)[0]
        messages[0]['content'] = 'x' * (36055-base)
        self.assertEqual(request_budget._text_input_bound(messages,payload)[0],36055)
        seen = []
        def post(url, **kwargs):
            seen.append({'timeout':kwargs['timeout'], 'cap':kwargs['json']['max_output_tokens'],
                         'committed':budget.committed_cost_usd,'row':dict(budget.call_log[-1])})
            return self.response(completed=True)
        result = openai_transport.responses_json(messages,model='gpt-5.6-terra',json_schema=schema,
            effort='low',reasoning_mode='',timeout=requested_timeout,max_output_tokens=3200,
            company_id='offline-company',purpose=PURPOSE,api_key='offline-unused',
            url='https://offline.invalid/responses',post_fn=post,current_budget_fn=lambda:budget,
            response_text_fn=openai_transport.response_text,
            safety_identifier_fn=openai_transport.safety_identifier)
        self.assertEqual(result,{})
        self.assertEqual(len(seen),1)
        self.assertEqual((seen[0]['timeout'],seen[0]['cap'],seen[0]['row']['reserved_input_tokens']),
                         (44,2832,36055))
        self.assertAlmostEqual(seen[0]['committed'],.24999615,places=12)
        self.assertLessEqual(seen[0]['committed'],.25)
        self.assertEqual(budget.llm_calls,3)
        self.assertEqual((budget.deadline_monotonic,budget.max_estimated_cost_usd,budget.max_llm_calls),limits)
        self.assertTrue(all(s['review_reserve_seconds']==0 for s in owner.provider_schedules))
        self.assertTrue(all(s['publication_reserve_seconds']==3 for s in owner.provider_schedules))
        # The larger output request cannot borrow from publication or reopen an
        # expired review window, even after the previous call settled cheaply.
        self.now[0] = 1091.01
        with self.assertRaises(request_budget._V13BudgetExceeded):
            self.dispatch(owner,deny,models=('gpt-5.6-terra',))
        self.assertEqual(budget.llm_calls,3)
        self.assertEqual(budget.deadline_monotonic,1100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
