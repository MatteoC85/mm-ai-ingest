"""Offline request-tier determinism; no model, account setting or budget change."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import openai_transport as transport
from machinemind.infrastructure import request_budget as budgets


def deny(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class Response:
    status_code = 200
    def __init__(self, data): self.data = data
    def json(self): return self.data


class ServiceTierTests(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        for name in ('socket.create_connection', 'socket.socket.connect', 'requests.sessions.Session.request'):
            self.guards.enter_context(patch(name, deny))
        self.guards.enter_context(patch.object(budgets, '_monotonic', return_value=1000.0))
        self.guards.enter_context(patch.object(budgets, '_RUNTIME_GLOBALS', {
            'V13_MAX_ESTIMATED_COST_ASK_USD': .25, 'V13_MAX_LLM_CALLS_ASK': 4,
            'V13_ASK_DEADLINE_SECONDS': 100, 'V13_PRICE_TERRA_INPUT': 2.5,
            'V13_PRICE_TERRA_OUTPUT': 15, 'V13_PRICE_EMBED_INPUT': .02}))
        self.budget = budgets._V13RequestBudget('ask')
        self.calls = []
        self.messages = [{'role': 'user', 'content': 'Offline synthetic request.'}]
        self.schema = {'name': 'offline', 'strict': True, 'schema': {'type': 'object',
            'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}}

    def post(self, data):
        def send(url, **kwargs):
            self.calls.append((url, deepcopy(kwargs)))
            self.assertFalse(kwargs['allow_redirects'])
            self.assertEqual(self.budget.llm_calls + self.budget.embedding_calls, len(self.calls))
            return Response(data)
        return send

    def responses(self, *, url='https://api.openai.com/v1/responses', tier='default'):
        data = {'status': 'completed', 'output_text': '{"ok":true}',
            'usage': {'input_tokens': 100, 'output_tokens': 40}, 'service_tier': tier}
        return transport.responses_json(self.messages, model='gpt-5.6-terra', json_schema=self.schema,
            effort='low', timeout=30, max_output_tokens=1000, company_id='offline-company',
            purpose='offline-tier', api_key='offline-unused', url=url, post_fn=self.post(data),
            current_budget_fn=lambda: self.budget, response_text_fn=transport.response_text,
            safety_identifier_fn=transport.safety_identifier)

    def chat(self, *, url='https://api.openai.com/v1/chat/completions', tier='default', budget=True):
        data = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 40}, 'service_tier': tier}
        post = self.post(data) if budget else lambda url, **kwargs: (self.calls.append((url, kwargs)) or Response(data))
        return transport.chat_json(self.messages, model='gpt-5.6-terra', json_schema=self.schema,
            timeout=30, api_key='offline-unused', default_model='gpt-5.6-terra', url=url, post_fn=post,
            current_budget_fn=(lambda: self.budget) if budget else None, max_output_tokens=1000)

    def test_official_responses_and_chat_pin_standard_before_budget_reservation(self):
        before_messages, before_schema = deepcopy(self.messages), deepcopy(self.schema)
        limits = (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd, self.budget.max_llm_calls)
        original = self.budget.reserve_call
        seen = []
        def reserve(**kwargs):
            seen.append(kwargs['request_payload']['service_tier'])
            return original(**kwargs)
        with patch.object(self.budget, 'reserve_call', side_effect=reserve):
            self.assertEqual(self.responses(), {'ok': True})
            self.assertEqual(self.chat(), {'ok': True})
        self.assertEqual(seen, ['default', 'default'])
        for _, request in self.calls:
            self.assertEqual(request['json']['service_tier'], 'default')
            self.assertEqual(request['json']['model'], 'gpt-5.6-terra')
            self.assertEqual(request['timeout'], 30)
        self.assertEqual(self.calls[0][1]['json']['max_output_tokens'], 1000)
        self.assertEqual(self.calls[1][1]['json']['max_completion_tokens'], 1000)
        self.assertEqual(self.messages, before_messages)
        self.assertEqual(self.schema, before_schema)
        self.assertEqual(limits, (self.budget.deadline_monotonic, self.budget.max_estimated_cost_usd, self.budget.max_llm_calls))
        self.assertEqual(self.budget.llm_calls, 2)
        self.assertAlmostEqual(self.budget.estimated_cost_usd, 2 * .00085)
        self.assertTrue(self.budget.public_meta()['accounting_complete'])

    def test_custom_or_lookalike_origins_do_not_receive_openai_specific_tier(self):
        for url in ('https://custom.invalid/v1/responses', 'https://api.openai.com.evil.invalid/v1/responses',
                    'https://api.openai.com/v1/responses?proxy=1'):
            with self.subTest(url=url):
                self.responses(url=url)
                self.assertNotIn('service_tier', self.calls[-1][1]['json'])
        self.chat(url='https://custom.invalid/v1/chat/completions')
        self.assertNotIn('service_tier', self.calls[-1][1]['json'])

    def test_unbudgeted_legacy_chat_also_pins_standard_without_new_budget(self):
        self.assertEqual(self.chat(budget=False), {'ok': True})
        self.assertEqual(self.calls[-1][1]['json']['service_tier'], 'default')
        self.assertEqual(self.budget.llm_calls, 0)

    def test_embeddings_payload_and_metering_are_unchanged(self):
        data = {'data': [{'index': 0, 'embedding': [.1, .2]}], 'usage': {'prompt_tokens': 5}}
        vectors = transport.embed_texts(['offline embedding'], api_key='offline-unused', model='text-embedding-3-small',
            url='https://api.openai.com/v1/embeddings', post_fn=self.post(data),
            current_budget_fn=lambda: self.budget, current_ingest_meter_fn=lambda: None)
        self.assertEqual(vectors, [[.1, .2]])
        self.assertEqual(self.calls[0][1]['json'], {'model': 'text-embedding-3-small', 'input': ['offline embedding']})
        self.assertNotIn('requested_service_tier', self.budget.embedding_log[0])
        self.assertNotIn('provider_service_tier', self.budget.embedding_log[0])

    def test_actual_returned_tier_is_preserved_without_repricing_or_assuming_requested_tier(self):
        self.responses(tier='priority')
        row = self.budget.public_meta()['calls'][0]
        self.assertEqual(row['requested_service_tier'], 'default')
        self.assertEqual(row['provider_service_tier'], 'priority')
        self.assertAlmostEqual(self.budget.estimated_cost_usd, .00085)
        self.assertEqual((row['input_rate'], row['output_rate']), (2.5, 15))

    def test_untrusted_tier_text_is_allowlisted_and_never_copies_provider_content(self):
        self.responses(tier='PRIVATE_PROVIDER_TEXT')
        self.assertEqual(self.budget.call_log[0]['provider_service_tier'], 'unknown')
        self.assertNotIn('PRIVATE_PROVIDER_TEXT', json.dumps(self.budget.public_meta()))

    def test_transport_failure_keeps_requested_tier_and_existing_uncertain_reservation_no_retry(self):
        def fail(url, **kwargs):
            self.calls.append((url, kwargs))
            raise TimeoutError('offline timeout')
        with patch.object(self, 'post', return_value=fail), self.assertRaises(TimeoutError):
            self.responses()
        self.assertEqual(len(self.calls), 1)
        row = self.budget.call_log[0]
        self.assertEqual(row['requested_service_tier'], 'default')
        self.assertNotIn('provider_service_tier', row)
        self.assertEqual(row['accounting_state'], 'uncertain')
        self.assertEqual(self.budget.committed_cost_usd, row['reserved_cost_usd'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
