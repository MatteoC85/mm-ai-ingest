"""Offline direct endpoint tests (no ASGI loopback required on Windows sandbox)."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smart_review import SmartReviewEndpointTests, parsed_review, wire_review, review, denied


def direct_post(self, kind, state=None, **kwargs):
    """Call the real decorated handler with validated request models and auth."""
    body = dict(company_id='fixture-company', machine_id='fixture-machine',
                session_id='fixture-session', language='en')
    if state is not None:
        body['state_json'] = state
    body.update(kwargs)
    model = {'start': self.m.SmartDiagnosticStartRequest, 'answer': self.m.SmartDiagnosticAnswerRequest,
             'finalize': self.m.SmartDiagnosticFinalizeRequest}[kind]
    handler = getattr(self.m, 'smart_diagnostic_' + kind + '_v1')
    try:
        result = handler(model(**body), x_ai_internal_secret='offline-smart-review-only')
        status = 200
    except self.m.HTTPException as exc:
        result, status = {'detail': exc.detail}, exc.status_code
    return SimpleNamespace(status_code=status, json=lambda: result, text=json.dumps(result))


class SmartContinuityEndpointTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    setUp = SmartReviewEndpointTests.setUp
    post = direct_post
    start = SmartReviewEndpointTests.start

    def rejecting_question(self, *args, **kwargs):
        return wire_review(parsed_review(self.prepared[-1], rejected=(1,), reject_question=True)), 'offline-review'

    def test_recovered_start_finalizes_and_repeats_without_question_source_forgery(self):
        with patch.object(self.m, '_v13_json_models', side_effect=self.rejecting_question):
            start = self.start()
        state = json.loads(start['session_state_json'])
        self.assertEqual(start['question']['application_reporting_version'], review.continuity.REPORTING_VERSION)
        self.assertFalse(start['meta']['smart_review']['question_validated'])
        self.assertTrue(start['meta']['smart_review']['rejected_question_discarded'])
        self.assertEqual(start['h1_probability_band'], 'unknown')
        self.assertEqual(start['h1_probability_pct'], 50)
        self.m._sd_validate_state_binding(state, company_id='fixture-company', machine_id='fixture-machine',
                                          session_id='fixture-session', question_id='Q1')
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            final = self.post('finalize', start['session_state_json'])
            self.assertEqual(final.status_code, 200, final.text)
            self.assertTrue(final.json()['final_ready'])
            self.assertEqual(final.json()['final_probability_band'], 'unknown')
            self.assertIn('diagnosis is not confirmed', final.json()['final_summary_text'])
            repeated = self.post('finalize', final.json()['session_state_json'])
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(final.json()['final_result'], repeated.json()['final_result'])

    def test_rejected_answer_then_unknown_preserves_history_and_finishes_free(self):
        start = self.start()
        next_step = deepcopy(self.raw)
        next_step['question']['question_text'] = 'An unsupported next technical question?'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step), \
             patch.object(self.m, '_v13_json_models', side_effect=self.rejecting_question):
            response = self.post('answer', start['session_state_json'], question_id='Q1',
                answer=dict(value='unknown', api_value='unknown', label='Unknown', free_text='Unverified note'))
        self.assertEqual(response.status_code, 200, response.text)
        second = response.json()
        state = json.loads(second['session_state_json'])
        self.assertEqual(len(state['history']), 1)
        self.assertEqual(state['history'][0]['answer']['free_text'], 'Unverified note')
        self.assertEqual(second['question_id'], 'Q2')
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_llm_step_answer', side_effect=denied), \
             patch.object(self.m, '_sd_enrich_state_evidence_from_answer', side_effect=denied):
            response = self.post('answer', second['session_state_json'], question_id='Q2',
                answer=dict(value='unknown', api_value='unknown', label='Unknown', free_text='Another unverified note'))
        self.assertEqual(response.status_code, 200, response.text)
        final = response.json()
        self.assertTrue(final['ok'])
        self.assertTrue(final['final_ready'])
        self.assertEqual(final['meta']['smart_review']['finalization_provider_calls'], 0)
        final_state = json.loads(final['session_state_json'])
        self.assertEqual(len(final_state['history']), 2)
        self.assertEqual(final_state['history'][-1]['answer']['free_text'], 'Another unverified note')
        self.assertEqual(final['final_probability_band'], 'unknown')
        self.assertEqual(final['final_recommended_checks_json'], '[]')
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            repeated = self.post('finalize', final['session_state_json'])
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()['final_result'], final['final_result'])

    def test_all_rejected_preserves_signed_history_without_stale_claims(self):
        start = self.start()
        next_step = deepcopy(self.raw)
        next_step['question']['question_text'] = 'Another unsupported question?'
        def reject_all(*args, **kwargs):
            return wire_review(parsed_review(self.prepared[-1], rejected=(0, 1, 2, 3), reject_question=True)), 'offline-review'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step), \
             patch.object(self.m, '_v13_json_models', side_effect=reject_all):
            response = self.post('answer', start['session_state_json'], question_id='Q1',
                answer=dict(value='unknown', api_value='unknown', label='Unknown', free_text='Retain this note'))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_code'], 'SMART_DIAGNOSTIC_REVIEW_REJECTED')
        state = json.loads(result['session_state_json'])
        self.assertEqual(len(state['history']), 1)
        self.assertEqual(result['history'], state['history'])
        self.assertNotIn('grounding_review', state)
        self.assertEqual(state['hypotheses'], [])
        self.assertEqual(state['current_question'], {})
        self.assertTrue(result['meta']['session_history_preserved'])
        self.m._sd_validate_state_binding(state, company_id='fixture-company', machine_id='fixture-machine',
                                          session_id='fixture-session')

    def test_recovery_cannot_be_used_for_another_tenant(self):
        with patch.object(self.m, '_v13_json_models', side_effect=self.rejecting_question):
            start = self.start()
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            result = self.post('answer', start['session_state_json'], question_id='Q1', company_id='other-company',
                answer=dict(value='unknown', api_value='unknown'))
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.json()['detail']['code'], 'SMART_DIAGNOSTIC_STATE_SCOPE_MISMATCH')


    def test_generation_receives_unknown_without_unverified_note_but_state_keeps_it(self):
        start = self.start()
        state = json.loads(start['session_state_json'])
        answer = dict(value='unknown', api_value='unknown', free_text='Unverified contradictory sensor claim')
        state['history'] = [dict(question=state['current_question'], answer=answer)]
        with patch.object(self.m, '_assistant_core_sd_json_models', return_value=deepcopy(self.raw)) as generator:
            self.m._sd_llm_step_answer(state=state, answer=answer, language='en', max_hypotheses=4)
        messages = generator.call_args.args[0]
        self.assertNotIn(answer['free_text'], json.dumps(messages))
        self.assertEqual(state['history'][0]['answer']['free_text'], answer['free_text'])
        self.assertIn('Unknown', messages[0]['content'])


del SmartReviewEndpointTests

if __name__ == '__main__':
    unittest.main()
