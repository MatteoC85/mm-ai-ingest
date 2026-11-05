"""Zero-weight reviewed possibilities stay useful without a manufactured winner."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_smart_review import SmartReviewEndpointTests, denied, review
from test_smart_continuity_endpoints import direct_post


class UnweightedEndpointTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    post = direct_post
    start = SmartReviewEndpointTests.start
    setup_fixture = SmartReviewEndpointTests.setUp

    def setUp(self):
        self.setup_fixture()
        self.raw['hypotheses'] = self.raw['hypotheses'][:2]
        for h in self.raw['hypotheses']:
            h.update(probability_pct=0.0, probability_band='unknown', score_raw=0.0,
                     status='open', checks=[])
        self.raw['question']['target_hypotheses'] = [h['id'] for h in self.raw['hypotheses']]

    def assert_unknown(self, body):
        self.assertTrue(body['ok'], body)
        self.assertTrue(body['hypotheses'])
        for h in body['hypotheses']:
            self.assertEqual((h['status'], h['probability_pct'], h['score_raw'], h['probability_band']),
                             ('open', 0.0, 0.0, 'unknown'))

    def test_start_unknown_answer_final_and_repeat_keep_no_cause_and_reviewed_question(self):
        start = self.start()
        self.assert_unknown(start)
        self.assertFalse(start['final_ready'])
        self.assertTrue(start['meta']['smart_review']['question_validated'])
        answer_step = deepcopy(self.raw)
        answer_step['question']['question_text'] = 'Which signal was already recorded before the stop?'
        note = 'Unverified assertion: the sensor has definitely failed.'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=answer_step):
            response = self.post('answer', start['session_state_json'], question_id='Q1',
                answer=dict(value='unknown', api_value='unknown', free_text=note))
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assert_unknown(answer)
        state = json.loads(answer['session_state_json'])
        self.assertEqual(len(state['history']), 1)
        self.assertEqual(state['history'][0]['answer']['free_text'], note)
        self.assertNotIn(note, review.observation_text(state['symptom_text'], state['history']))
        self.m._sd_validate_state_binding(state, company_id='fixture-company', machine_id='fixture-machine',
                                         session_id='fixture-session', question_id='Q2')
        before = self.provider.call_count
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_llm_step_answer', side_effect=denied):
            final = self.post('finalize', answer['session_state_json'])
            self.assertEqual(final.status_code, 200, final.text)
            out = final.json()
            self.assert_unknown(out)
            self.assertTrue(out['final_ready'])
            self.assertEqual(out['final_result']['most_likely_hypothesis_id'], '')
            self.assertEqual(out['final_result']['most_likely_label'], 'No cause identified')
            self.assertEqual(out['final_probability_pct'], 0)
            self.assertEqual(out['final_probability_band'], 'unknown')
            self.assertIn('inconclusive', out['final_summary_text'])
            self.assertIn(answer_step['question']['question_text'], out['final_summary_text'])
            self.assertNotIn('Most supported hypothesis:', out['final_summary_text'])
            self.assertNotIn(note, out['final_summary_text'])
            self.assertEqual(out['final_recommended_checks_json'], '[]')
            repeated = self.post('finalize', out['session_state_json'])
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(repeated.json()['final_result'], out['final_result'])
            self.assertEqual(json.loads(out['session_state_json'])['history'], state['history'])
        self.assertEqual(self.provider.call_count, before)

    def test_zero_state_does_not_relax_hmac_or_tenant_binding(self):
        start = self.start()
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            foreign = self.post('finalize', start['session_state_json'], company_id='other-company')
            self.assertEqual(foreign.status_code, 400)
            self.assertEqual(foreign.json()['detail']['code'], 'SMART_DIAGNOSTIC_STATE_SCOPE_MISMATCH')
            state = json.loads(start['session_state_json'])
            state['hypotheses'][0]['probability_pct'] = 100
            tampered = self.post('finalize', json.dumps(state))
            self.assertEqual(tampered.status_code, 400)
            self.assertEqual(tampered.json()['detail']['code'], 'SMART_DIAGNOSTIC_STATE_TAMPERED')

    def test_finalizer_ignores_unreviewed_winner_and_preserves_reviewed_reasons_only(self):
        invented = 'Confirmed pump failure. Bypass the guard and restart.'
        final = self.m._sd_canonicalize_final_result(
            dict(most_likely_hypothesis_id='H2', summary=invented,
                 recommended_checks=['Bypass the guard and restart.']), self.raw['hypotheses'], 'it')
        self.assertEqual(final['most_likely_hypothesis_id'], '')
        self.assertEqual(final['probability_pct'], 0)
        self.assertEqual(final['probability_band'], 'unknown')
        self.assertIn('inconclusiva', final['summary'])
        self.assertNotIn('pump failure', json.dumps(final))
        self.assertNotIn('Bypass the guard', json.dumps(final))
        self.assertEqual(final['recommended_checks'], [])

    def test_unreviewed_draft_final_and_operator_claims_never_reach_public_final(self):
        invented = 'Confirmed pump failure. Bypass the guard and restart.'
        self.raw['operator_summary'] = invented
        self.raw['final_result'] = dict(summary=invented, most_likely_hypothesis_id='H2',
            most_likely_label='Failed pump', probability_pct=100.0, probability_band='high',
            recommended_checks=['Bypass the guard and restart.'])
        start = self.start()
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            response = self.post('finalize', start['session_state_json'])
        self.assertEqual(response.status_code, 200, response.text)
        final = response.json()
        self.assertEqual(final['final_result']['most_likely_hypothesis_id'], '')
        self.assertEqual(final['final_probability_pct'], 0)
        for body in (start, final):
            public = json.dumps({k: v for k, v in body.items() if k != 'session_state_json'})
            self.assertNotIn('Confirmed pump failure', public)
            self.assertNotIn('Bypass the guard', public)


del SmartReviewEndpointTests

if __name__ == '__main__':
    unittest.main(verbosity=2)
