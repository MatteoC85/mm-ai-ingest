"""Public Smart wording must not rank a tie or turn relative 100% into certainty."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_smart_review import SmartReviewEndpointTests, denied
from test_smart_continuity_endpoints import direct_post


class SmartSummaryPresentationTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)

    def hypotheses(self, weights=(50, 50), status='open'):
        return [{'id': 'H' + str(i + 1), 'rank': i + 1, 'label': 'Documented mechanism ' + str(i + 1),
                 'status': status, 'probability_pct': weight, 'probability_band': 'unknown',
                 'why': 'The observation is unknown; the guard must remain closed.',
                 'checks': ['Read only the existing record.']} for i, weight in enumerate(weights)]

    def display(self, language, hypotheses):
        final = self.m._sd_canonicalize_final_result({}, hypotheses, language)
        before = deepcopy((final, hypotheses))
        displayed = self.m._sd_final_result_for_display(final, hypotheses, language)
        self.assertEqual((final, hypotheses), before)
        active = [h for h in hypotheses if h.get('status') != 'excluded']
        weights = [round(float(h.get('probability_pct') or 0), 1) for h in active]
        tied = bool(weights and max(weights) > 0 and weights.count(max(weights)) > 1
                    and not any(h.get('status') == 'confirmed' for h in active))
        projected = {'summary', 'most_likely_hypothesis_id', 'most_likely_label',
                     'probability_pct', 'probability_band'} if tied else {'summary'}
        self.assertEqual({k: v for k, v in displayed.items() if k not in projected},
                         {k: v for k, v in final.items() if k not in projected})
        old_lead = ('Most supported hypothesis: ' if language == 'en' else 'Ipotesi più supportata: ')
        if final['summary'].startswith(old_lead):
            tail = final['summary'][len(old_lead + final['most_likely_label'] + '.'):]
            self.assertTrue(displayed['summary'].endswith(tail))
        return final, displayed

    def test_it_en_ties_lead_with_inconclusive_without_selecting_a_new_winner(self):
        for language, phrase in [('it', 'alla pari'), ('en', 'are equal')]:
            with self.subTest(language=language):
                final, displayed = self.display(language, self.hypotheses())
                self.assertIn('inconclus', displayed['summary'].split('\n')[0])
                self.assertIn(phrase, displayed['summary'])
                self.assertNotIn('Most supported hypothesis:', displayed['summary'])
                self.assertNotIn('Ipotesi più supportata:', displayed['summary'])
                self.assertEqual(displayed['most_likely_hypothesis_id'], '')
                self.assertIn('prevalente' if language == 'it' else 'leading', displayed['most_likely_label'])
                self.assertEqual(displayed['probability_pct'], 0)
                self.assertEqual(displayed['probability_band'], 'unknown')

    def test_it_en_single_100_is_explicitly_not_confirmation(self):
        for language, phrase in [('it', '100% non conferma'), ('en', '100% does not confirm')]:
            with self.subTest(language=language):
                _, displayed = self.display(language, self.hypotheses((100,)))
                self.assertIn(phrase, displayed['summary'])
                self.assertIn('inconclus', displayed['summary'])
                self.assertEqual(displayed['probability_pct'], 100)
                self.assertEqual(displayed['probability_band'], 'unknown')

    def test_unequal_weights_do_not_gain_a_tie_or_new_machine_claim(self):
        _, displayed = self.display('en', self.hypotheses((70, 30)))
        self.assertIn('inconclusive', displayed['summary'])
        self.assertNotIn('are equal', displayed['summary'])
        self.assertNotIn('100%', displayed['summary'])
        self.assertIn('The observation is unknown; the guard must remain closed.', displayed['summary'])

    def test_confirmed_hypothesis_or_confirmed_tie_keeps_existing_conclusion(self):
        for weights in ((100,), (50, 50)):
            for language in ('it', 'en'):
                with self.subTest(weights=weights, language=language):
                    final, displayed = self.display(language, self.hypotheses(weights, status='confirmed'))
                    self.assertEqual(displayed, final)

    def test_excluded_empty_and_unweighted_conclusions_are_unchanged(self):
        for hypotheses in ([], self.hypotheses(status='excluded'), self.hypotheses((0, 0))):
            for language in ('it', 'en'):
                final = self.m._sd_canonicalize_final_result({}, hypotheses, language)
                self.assertEqual(self.m._sd_final_result_for_display(final, hypotheses, language), final)

    def test_noncanonical_text_and_unknown_selected_identity_are_not_rewritten(self):
        hypotheses = self.hypotheses()
        final = self.m._sd_canonicalize_final_result({}, hypotheses, 'en')
        for changed in ({**final, 'summary': 'A separately reviewed inconclusive conclusion.'},
                        {**final, 'most_likely_hypothesis_id': 'foreign'}):
            self.assertEqual(self.m._sd_final_result_for_display(changed, hypotheses, 'en'), changed)


class SmartSummaryReplayTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    setup_fixture = SmartReviewEndpointTests.setUp
    post = direct_post
    start = SmartReviewEndpointTests.start

    def setUp(self):
        self.setup_fixture()
        self.raw['hypotheses'] = self.raw['hypotheses'][:2]
        for h in self.raw['hypotheses']:
            h.update(status='open', probability_pct=50.0, probability_band='unknown', score_raw=.5, checks=[])
        self.raw['question']['target_hypotheses'] = [h['id'] for h in self.raw['hypotheses']]

    def test_old_signed_final_replays_with_new_public_wording_and_unchanged_state(self):
        start = self.start()
        answer_step = deepcopy(self.raw)
        answer_step['question']['question_text'] = 'Which information was already recorded before the stop?'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=answer_step):
            answered = self.post('answer', start['session_state_json'], question_id='Q1',
                answer={'value': 'unknown', 'api_value': 'unknown'}).json()
        before = self.provider.call_count
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_final_result_for_display', side_effect=lambda final, hypotheses, language: dict(final)):
            old = self.post('finalize', answered['session_state_json'])
        self.assertEqual(old.status_code, 200, old.text)
        old = old.json(); signed = json.loads(old['session_state_json'])
        self.assertIn('Most supported hypothesis:', old['final_summary_text'])
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            response = self.post('finalize', old['session_state_json'])
            self.assertEqual(response.status_code, 200, response.text)
            current = response.json()
            again = self.post('finalize', current['session_state_json'])
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(self.provider.call_count, before)
        self.assertTrue(current['final_summary_text'].startswith('The diagnosis is inconclusive:'))
        self.assertIn('are equal', current['final_summary_text'])
        self.assertEqual(current['operator_summary'], current['final_summary_text'])
        self.assertEqual(current['final_result']['summary'], current['final_summary_text'])
        self.assertEqual(json.loads(current['final_result_json']), current['final_result'])
        self.assertEqual(again.json()['final_result'], current['final_result'])
        current_state = json.loads(current['session_state_json'])
        for key in ('hypotheses', 'history', 'grounding_review', 'final_result'):
            self.assertEqual(current_state[key], signed[key])
        self.assertEqual(current['hypotheses'], old['hypotheses'])
        self.assertEqual(current['final_recommended_checks_json'], old['final_recommended_checks_json'])
        self.assertIn(answer_step['question']['question_text'], current['final_summary_text'])
        self.assertIn('Unknown', current['final_summary_text'])
        self.m._sd_validate_state_binding(current_state, company_id='fixture-company',
            machine_id='fixture-machine', session_id='fixture-session')

    def test_distinct_operator_context_is_not_erased_by_display_projection(self):
        start = self.start()
        terminal = self.m._sd_terminal_reviewed_step
        extra = 'Keep the recorded isolation condition until the approved review.'
        def with_extra(*args, **kwargs):
            step = terminal(*args, **kwargs)
            step['operator_summary'] += '\n' + extra
            return step
        before = self.provider.call_count
        with patch.object(self.m, '_sd_terminal_reviewed_step', side_effect=with_extra), \
             patch.object(self.m, '_v13_json_models', side_effect=denied):
            response = self.post('finalize', start['session_state_json'])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body['operator_summary'], body['final_summary_text'] + '\n' + extra)
        self.assertIn('Most supported hypothesis:', body['final_summary_text'])
        self.assertEqual(self.provider.call_count, before)


del SmartReviewEndpointTests

if __name__ == '__main__':
    unittest.main(verbosity=2)
