"""Offline contracts for a reviewed terminal with no surviving hypothesis.

Provider decisions are explicit fixtures, not claims of model semantic accuracy.
The real endpoint, source-proof validator, signing and finalize replay execute.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_smart_review import SmartReviewEndpointTests, parsed_review, wire_review, review, denied
from test_smart_continuity_endpoints import direct_post


class SmartInconclusiveTerminalTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    setUp = SmartReviewEndpointTests.setUp
    post = direct_post
    start = SmartReviewEndpointTests.start

    def terminal(self, *, count=1):
        result = deepcopy(self.raw)
        result.update(status='completed', final_ready=True, question={},
                      operator_summary='UNREVIEWED GENERATOR SUMMARY', final_result={
                          'summary': 'UNREVIEWED GENERATOR SUMMARY',
                          'most_likely_hypothesis_id': 'H1', 'most_likely_label': 'Invented confirmed cause',
                          'probability_pct': 99, 'recommended_checks': ['UNREVIEWED OPERATION']})
        result['hypotheses'] = result['hypotheses'][:count]
        for h in result['hypotheses']:
            h.update(status='excluded', probability_pct=0, probability_band='very_low', checks=[],
                     why='The reported setup record rules out the displacement required by this documented mechanism. '
                         'The unavailable signal trace remains unknown; it does not restore that prerequisite.')
        return result

    def normalize(self, value, ids=None):
        return self.m._sd_normalize_step(value, language='en', question_number_default=3,
            max_hypotheses=4, allowed_evidence_ids={'manual:p1', 'manual:p2'} if ids is None else ids)

    def accepted_terminal(self):
        self.raw['hypotheses'] = self.raw['hypotheses'][:1]
        self.raw['hypotheses'][0]['checks'] = []
        self.raw['question']['target_hypotheses'] = ['H1']
        start = self.start()
        next_step = deepcopy(self.raw)
        next_step['question'].update(
            question_text='According to setup information already available, was the fixture displaced?',
            why_asked='The documented mechanism requires displacement; an unavailable signal is not evidence of it.',
            options=[{'id': 'yes', 'label_en': 'Yes', 'label_it': 'Sì'},
                     {'id': 'no', 'label_en': 'No', 'label_it': 'No'},
                     {'id': 'unknown', 'label_en': "I don't know", 'label_it': 'Non so'}])
        with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step):
            first = self.post('answer', start['session_state_json'], question_id='Q1',
                answer=dict(value='unknown', api_value='unknown', label='Unknown', free_text='Unverified note'))
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['question_id'], 'Q2')
        calls_before = self.provider.call_count
        with patch.object(self.m, '_sd_llm_step_answer', return_value=self.terminal()):
            second = self.post('answer', first.json()['session_state_json'], question_id='Q2',
                answer=dict(value='no', api_value='no', label='No', free_text=''))
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(self.provider.call_count, calls_before + 1, 'terminal exclusion must be reviewed')
        return second.json()

    def test_normalization_preserves_grounded_negative_terminal_without_raw_final_claims(self):
        step = self.normalize(self.terminal(count=2))
        self.assertEqual(step['status'], 'completed')
        self.assertTrue(step['final_ready'])
        self.assertEqual([h['status'] for h in step['hypotheses']], ['excluded', 'excluded'])
        self.assertEqual(step['final_result']['most_likely_hypothesis_id'], '')
        self.assertEqual(step['final_result']['probability_pct'], 0)
        self.assertEqual(step['final_result']['probability_band'], 'unknown')
        self.assertIn('inconclusive', step['final_result']['summary'])
        self.assertNotIn('UNREVIEWED', json.dumps(step))
        self.assertEqual(step['final_result']['recommended_checks'], [])

    def test_no_sources_unfinished_positive_weight_and_ungrounded_do_not_gain_terminal_status(self):
        for mutation in ('no_sources', 'unfinished', 'positive_weight', 'ungrounded', 'empty'):
            with self.subTest(mutation=mutation):
                raw = self.terminal()
                if mutation == 'no_sources':
                    raw.update(status='no_sources', final_ready=False)
                elif mutation == 'unfinished':
                    raw.update(status='in_progress', final_ready=False)
                elif mutation == 'positive_weight':
                    raw['hypotheses'][0]['probability_pct'] = 60
                elif mutation == 'ungrounded':
                    raw['hypotheses'][0]['evidence_ids'] = ['foreign-document:p8']
                else:
                    raw['hypotheses'] = []
                step = self.normalize(raw)
                self.assertEqual(step['status'], 'no_sources')
                self.assertFalse(step['final_ready'])

    def test_unknown_then_no_preserves_signed_history_and_finalizes_without_new_calls(self):
        result = self.accepted_terminal()
        self.assertTrue(result['ok'])
        self.assertTrue(result['final_ready'])
        self.assertEqual(result['final_result']['most_likely_hypothesis_id'], '')
        self.assertEqual(result['final_probability_pct'], 0)
        self.assertEqual(result['final_probability_band'], 'unknown')
        self.assertTrue(result['citations'])
        self.assertTrue(result['meta']['source_manifest']['reviewer_proof_bound'])
        state = json.loads(result['session_state_json'])
        self.assertEqual([row['answer']['api_value'] for row in state['history']], ['unknown', 'no'])
        self.assertEqual(state['history'][0]['answer']['free_text'], 'Unverified note')
        self.assertEqual(state['hypotheses'][0]['status'], 'excluded')
        self.assertIn('grounding_review', state)
        self.m._sd_validate_state_binding(state, company_id='fixture-company', machine_id='fixture-machine', session_id='fixture-session')
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_llm_step_answer', side_effect=denied), \
             patch.object(self.m, '_sd_enrich_state_evidence_from_answer', side_effect=denied):
            for _ in range(2):
                final = self.post('finalize', result['session_state_json'])
                self.assertEqual(final.status_code, 200, final.text)
                self.assertTrue(final.json()['ok'])
                self.assertEqual(final.json()['final_result'], result['final_result'])
                self.assertEqual(final.json()['meta']['smart_review']['finalization_provider_calls'], 0)
                result = final.json()

    def test_rejected_exclusion_is_not_relabelled_as_successful_inconclusive_result(self):
        start = self.start()
        def reject(*args, **kwargs):
            parsed = parsed_review(self.prepared[-1], rejected=(0,))
            parsed['decisions'][0].update(reason='contradicted', note='Unknown does not justify excluding this mechanism.')
            return wire_review(parsed), 'offline-review'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=self.terminal()), \
             patch.object(self.m, '_v13_json_models', side_effect=reject):
            response = self.post('answer', start['session_state_json'], question_id='Q1',
                answer=dict(value='unknown', api_value='unknown', label='Unknown', free_text='Not verified'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['ok'])
        self.assertFalse(response.json()['final_ready'])
        self.assertEqual(response.json()['error_code'], 'SMART_DIAGNOSTIC_REVIEW_REJECTED')
        state = json.loads(response.json()['session_state_json'])
        self.assertEqual(len(state['history']), 1)
        self.assertNotIn('grounding_review', state)

    def test_partial_rejection_cannot_claim_all_candidates_excluded(self):
        step = self.normalize(self.terminal(count=2))
        packet = self.m._smart_evidence.review_packet(self.m._smart_evidence.build(
            self.sources, scope=self.m._sd_grounding_scope('fixture-company', 'fixture-machine'),
            max_chars=22000, allow_raw_snippet=True))
        prepared = review.prepare(step=step, packet=packet, symptom_text='Reported stop; displacement ruled out.', history=[])
        parsed = parsed_review(prepared, rejected=(1,))
        parsed['decisions'][1].update(reason='unsupported', note='This exclusion lacks supporting observations.')
        with self.assertRaisesRegex(review.SmartReviewError, 'no_supported_hypotheses'):
            review.resolve(prepared=prepared, parsed=parsed,
                probability_band=self.m._sd_probability_band, continuity_policy=review.continuity.POLICY_VERSION)

    def test_signed_terminal_still_rejects_foreign_tenant_and_tampered_claims_or_review(self):
        result = self.accepted_terminal()
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            foreign = self.post('finalize', result['session_state_json'], company_id='other-company')
            self.assertEqual(foreign.status_code, 400)
            self.assertEqual(foreign.json()['detail']['code'], 'SMART_DIAGNOSTIC_STATE_SCOPE_MISMATCH')
            for mutation in ('final_claim', 'hypothesis', 'review', 'packet'):
                with self.subTest(mutation=mutation):
                    state = json.loads(result['session_state_json'])
                    if mutation == 'final_claim':
                        state['final_result']['summary'] = 'Invented confirmed diagnosis'
                    elif mutation == 'hypothesis':
                        state['hypotheses'][0]['why'] = 'Invented exclusion'
                    elif mutation == 'review':
                        state.pop('grounding_review')
                    else:
                        state['grounding_packet']['bodies'][0]['text'] += ' Forged source'
                    # Re-sign solely to exercise source/claim replay beyond HMAC;
                    # no endpoint or client is given the server signing key.
                    state = self.m._sd_sign_state(state)
                    final = self.post('finalize', json.dumps(state))
                    self.assertEqual(final.status_code, 409, final.text)

    def test_mixed_active_and_excluded_keeps_the_supported_active_conclusion(self):
        raw = self.terminal(count=2)
        raw['hypotheses'][1].update(status='open', probability_pct=40, why='This separate supported mechanism remains conditional.')
        step = self.normalize(raw)
        self.assertTrue(step['final_ready'])
        self.assertEqual(step['final_result']['most_likely_hypothesis_id'], 'H2')
        self.assertEqual(step['hypotheses'][0]['status'], 'excluded')


del SmartReviewEndpointTests

if __name__ == '__main__':
    unittest.main()
