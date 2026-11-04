"""Protected owner isolation and existing factual execution observation, offline."""
from contextlib import redirect_stdout
from contextvars import Context, ContextVar
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from machinemind.ask import request_completion as c, private_review_diagnostics as p
from machinemind.ask.request_flow import RequestFlowRuntime
from test_private_review_diagnostics import body, SECRET


class Budget:
    def __init__(self):
        self.started_monotonic = time.monotonic()
        self.deadline_seconds = 100
        self.deadline_monotonic = self.started_monotonic + 100
        self.request_id = 'b' * 32
        self.max_estimated_cost_usd = .25
        self.llm_calls = 0
    def elapsed(self): return time.monotonic() - self.started_monotonic
    def remaining(self): return self.deadline_monotonic - time.monotonic()
    def ensure_time(self, seconds):
        if self.remaining() < seconds: raise RuntimeError('offline deadline')


def runtime(secret=SECRET):
    values = {f.name: None for f in fields(RequestFlowRuntime)}
    values.update(AI_INTERNAL_SECRET=secret, _V13_BUDGET_CTX=ContextVar('offline-budget', default=None),
        _assistant_core_new_budget=lambda *a, **k: Budget(),
        _assistant_core_attach_runtime_meta=lambda response, budget, **kwargs: deepcopy(response))
    return RequestFlowRuntime(**values)


def request():
    return SimpleNamespace(company_id='offline-company', machine_id='offline-machine', query='question', debug=True, requested_mode='ask')


def publish_allowed(response):
    return {**response, 'meta': {**response.get('meta', {}), 'request_authority': {'scope_authorized': True}}}


class PrivateCaptureOwnerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'COMMIT_SHA': 'a' * 40}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)

    def owner(self, *, debug=True, secret=SECRET):
        payload = request(); payload.debug = debug
        owner = c.RequestCompletion(runtime=runtime(secret), payload=payload, started=time.monotonic())
        self.addCleanup(owner.close)
        return owner

    def record(self, **kwargs):
        data = body(); data.pop('question')
        c.record_private_factual_review(request=kwargs.pop('request', request()), **data, **kwargs)

    def finish(self, owner, *, allowed=True):
        value = {'ok': True, 'status': 'no_sources', 'result_code': 'NO_MACHINE_EVIDENCE',
                 'answer': 'Supported evidence was not released.', 'meta': {}}
        if allowed: value = owner.publish(value, publish_allowed)
        return owner.finish(value)

    def test_only_ciphertext_is_attached_after_allowed_publication_with_same_answer_and_budget(self):
        owner = self.owner(); limits = owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd
        self.record(); before = {'ok': True, 'status': 'no_sources', 'answer': 'unchanged', 'meta': {}}
        seen = []
        def guard(value):
            seen.append(deepcopy(value)); self.assertNotIn('private_factual_review_diagnostics', value.get('meta', {}))
            return publish_allowed(value)
        final = owner.finish(owner.publish(before, guard))
        self.assertEqual(final['answer'], before['answer'])
        self.assertEqual(seen, [before])
        envelope = final['meta']['private_factual_review_diagnostics']
        decoded = p.open_envelope(envelope, secret=SECRET, expected_binding=owner._private_review_binding())
        self.assertEqual(decoded, p._plain(body()))
        self.assertNotIn('Literal answer', json.dumps(final))
        self.assertEqual(limits, (owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd))
        self.assertEqual(owner.budget.llm_calls, 0)

    def test_no_debug_short_secret_foreign_request_and_duplicate_capture_are_suppressed(self):
        for debug, secret, foreign, twice in [(False, SECRET, False, False), (True, 'short', False, False),
                                              (True, SECRET, True, False), (True, SECRET, False, True)]:
            with self.subTest(debug=debug, short=len(secret)<32, foreign=foreign, twice=twice):
                owner = self.owner(debug=debug, secret=secret)
                req = request()
                if foreign: req.company_id = 'foreign'
                self.record(request=req)
                if twice: self.record()
                final = self.finish(owner)
                self.assertNotIn('private_factual_review_diagnostics', final['meta'])
                owner.close()

    def test_unfinished_publication_denial_timeout_and_scope_not_authorized_never_emit_capture(self):
        for kind in ('unfinished', 'denied', 'timeout', 'scope_false'):
            with self.subTest(kind=kind):
                owner = self.owner(); self.record()
                value = {'ok': True, 'status': 'no_sources', 'answer': 'not a draft', 'meta': {}}
                if kind == 'denied':
                    with self.assertRaises(RuntimeError):
                        owner.publish(value, lambda value: (_ for _ in ()).throw(RuntimeError('offline denied')))
                    value = {'ok': False, 'status': 'error', 'result_code': 'AUTHORITY_ERROR', 'meta': {}}
                elif kind == 'timeout':
                    value = owner.mark_terminal({'ok': False, 'status': 'error', 'result_code': 'TIMEOUT', 'citations': [], 'rg_links': [], 'meta': {}})
                elif kind == 'scope_false':
                    value = owner.publish(value, lambda value: {**value, 'meta': {'request_authority': {'scope_authorized': False}}})
                final = owner.finish(value)
                self.assertNotIn('private_factual_review_diagnostics', final['meta'])
                owner.close()

    def test_crypto_failure_never_logs_plaintext_or_changes_existing_response(self):
        owner = self.owner(); output = io.StringIO()
        with patch.object(p, 'seal', side_effect=RuntimeError('UNREVIEWED_PRIVATE_DRAFT ' + SECRET)), redirect_stdout(output):
            self.record(); final = self.finish(owner)
        self.assertEqual(output.getvalue(), '')
        self.assertNotIn(SECRET, json.dumps(final)); self.assertNotIn('UNREVIEWED_PRIVATE_DRAFT', json.dumps(final))
        self.assertNotIn('private_factual_review_diagnostics', final['meta'])

    def test_new_or_foreign_context_cannot_inherit_capture_and_close_clears_owner(self):
        owner = self.owner()
        Context().run(self.record)
        self.assertIsNone(owner._private_review_envelope)
        self.record(); self.assertIsInstance(owner._private_review_envelope, dict)
        owner.close(); self.assertIsNone(owner._private_review_envelope); self.assertIsNone(c.current())
        other = self.owner(); final = self.finish(other)
        self.assertNotIn('private_factual_review_diagnostics', final['meta'])

    def test_changed_query_or_runtime_identity_before_finish_suppresses_envelope(self):
        owner = self.owner(); self.record()
        owner.payload.query = 'changed question'
        self.assertNotIn('private_factual_review_diagnostics', self.finish(owner)['meta'])
        owner.close()
        owner = self.owner(); self.record()
        os.environ['COMMIT_SHA'] = 'e' * 40
        self.assertNotIn('private_factual_review_diagnostics', self.finish(owner)['meta'])

    def test_close_while_observing_cannot_restore_cleared_capture(self):
        owner = self.owner()
        def closing_seal(**kwargs):
            owner.close()
            return {'ciphertext_b64': 'offline-cipher-only'}
        with patch.object(p, 'seal', side_effect=closing_seal): self.record()
        self.assertIsNone(owner._private_review_envelope)
        self.assertIsNone(c.current())

    def test_root_and_smart_do_not_acquire_ask_capture_from_an_enclosing_owner(self):
        for mode in ('root_cause', 'smart_diagnostic'):
            owner = self.owner(); req = request(); req.requested_mode = mode
            self.record(request=req)
            self.assertNotIn('private_factual_review_diagnostics', self.finish(owner)['meta'])
            owner.close()

    def test_foreign_or_cached_envelope_is_removed_on_every_nonowned_exit(self):
        for kind in ('no_debug', 'unfinished', 'unauthorized', 'terminal'):
            owner = self.owner(debug=kind != 'no_debug')
            value = {'ok': True, 'status': 'no_sources', 'answer': 'safe unchanged',
                     'meta': {'private_factual_review_diagnostics': {'FOREIGN_CACHED_ENVELOPE': 'must disappear'}}}
            before = deepcopy(value)
            if kind == 'unauthorized':
                value = owner.publish(value, lambda v: {**v, 'meta': {**v['meta'], 'request_authority': {'scope_authorized': False}}})
            elif kind == 'terminal':
                value = owner.mark_terminal({**value, 'ok': False, 'status': 'error', 'result_code': 'TIMEOUT', 'citations': [], 'rg_links': []})
            else:
                if kind != 'unfinished': value = owner.publish(value, publish_allowed)
            final = owner.finish(value)
            self.assertNotIn('private_factual_review_diagnostics', final['meta'])
            self.assertNotIn('FOREIGN_CACHED_ENVELOPE', json.dumps(final))
            self.assertIn('private_factual_review_diagnostics', before['meta'])
            owner.close()

    def test_reentrant_second_record_permanently_invalidates_first_inflight_capture(self):
        owner = self.owner(); original_seal = p.seal
        def reentrant_seal(**kwargs):
            self.record()  # Nested observer invalidates, never begins encryption.
            return original_seal(**kwargs)
        with patch.object(p, 'seal', side_effect=reentrant_seal) as mocked:
            self.record()
        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(owner._private_review_invalidated)
        self.assertIsNone(owner._private_review_envelope)
        self.assertNotIn('private_factual_review_diagnostics', self.finish(owner)['meta'])


class PrivateCaptureExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_factual_context_owned import FactualContextOwnedTests
        FactualContextOwnedTests.setUpClass()
        cls.m = FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        from test_factual_context_owned import FactualContextOwnedTests
        FactualContextOwnedTests.tearDownClass()

    def test_actual_rejected_factual_execution_is_captured_without_extra_provider_or_approval(self):
        from test_factual_context_owned import ContextFixture
        from machinemind.ask import execution
        query = 'Which maintenance intervals and conditions are documented?'
        payload = SimpleNamespace(company_id='offline-company', machine_id='offline-machine', query=query, debug=True)
        with patch.dict(os.environ, {'COMMIT_SHA': 'a' * 40}, clear=True):
            owner = c.RequestCompletion(runtime=replace(self.m._assistant_core_request_flow_runtime(), AI_INTERNAL_SECRET=SECRET),
                                        payload=payload, started=time.monotonic())
            self.addCleanup(owner.close)
            fixture = ContextFixture(self, query=query, second=lambda reply: reply.update(outcome='no_sources'))
            self.addCleanup(fixture.close)
            with patch.object(execution, '_record_private_factual_review', wraps=c.record_private_factual_review) as observed:
                result = fixture.run()
            self.assertEqual(result['status'], 'no_sources')
            self.assertEqual(len(fixture.post_calls), 2)
            self.assertEqual(observed.call_count, 1)
            actual = observed.call_args.kwargs
            self.assertEqual(actual['first_wire_reply'], fixture.first_provider_wire_reply)
            self.assertEqual(actual['actual_review_reply']['outcome'], 'no_sources')
            envelope = owner._private_review_envelope
            self.assertIsInstance(envelope, dict)
            decoded = p.open_envelope(envelope, secret=SECRET, expected_binding=owner._private_review_binding())
            self.assertEqual(decoded['first_wire_reply'], actual['first_wire_reply'])
            self.assertEqual(decoded['expanded_candidate'], p._plain(actual['expanded_candidate']))
            self.assertEqual(decoded['review_frame'], p._plain(actual['review_frame']))
            self.assertEqual(decoded['actual_review_reply'], actual['actual_review_reply'])
            self.assertEqual(decoded['sources_block'], actual['sources_block'])
            self.assertEqual(decoded['review_frame']['answer'], fixture.first_candidate)
            self.assertFalse(envelope['aad']['redaction_changed'])
            final = owner.finish(owner.publish(result, publish_allowed))
            self.assertEqual(final['status'], 'no_sources')
            self.assertEqual(final.get('citations') or [], [])
            self.assertIn('private_factual_review_diagnostics', final['meta'])
            self.assertEqual(len(fixture.post_calls), 2)
            self.assertEqual(owner.budget.llm_calls, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
