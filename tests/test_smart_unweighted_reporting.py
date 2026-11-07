"""Reviewed but unweighted causes retain a safe, nontechnical reporting path."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import smart_review as review, smart_continuity as continuity
from test_smart_review import fixture, parsed_review, smart_evidence, refs


class UnweightedReportingTests(unittest.TestCase):
    def setUp(self):
        sources, self.step = fixture()
        self.packet = smart_evidence.review_packet(smart_evidence.build(sources, scope={
            'company_id': 'fixture-company', 'machine_id': 'fixture-machine', 'ai_scope': 'machine_all'}))
        for h in self.step['hypotheses']:
            h.update(probability_pct=0, probability_band='unknown', score_raw=0)
        self.history = [{'question': {'question_text': 'Which configuration was already recorded?'},
                         'answer': {'value': 'unknown', 'free_text': 'Unverified failure assertion.'}}]
        self.band = lambda pct: 'low'

    def prepared(self, history=None):
        return review.prepare(step=self.step, packet=self.packet, symptom_text='Reported stop.',
                              history=self.history if history is None else history, language='en')

    def resolve(self, *, rejected=(1, 2, 3), reject_question=True, policy=continuity.POLICY_VERSION, history=None):
        prepared = self.prepared(history)
        return review.resolve(prepared=prepared, parsed=parsed_review(prepared,
            rejected=rejected, reject_question=reject_question), probability_band=self.band,
            continuity_policy=policy)

    def state(self, out, seal, history=None):
        return dict(grounding_review=seal, status=out['status'], symptom_text='Reported stop.',
                    language='en', history=deepcopy(self.history if history is None else history),
                    hypotheses=out['hypotheses'], current_question=out['question'])

    def test_rejected_technical_question_is_discarded_without_increasing_zero_weight(self):
        rejected_text = 'Start a cycle to obtain the unknown signal.'
        self.step['question']['question_text'] = rejected_text
        out, seal, meta = self.resolve()
        self.assertEqual([h['id'] for h in out['hypotheses']], ['H1'])
        h = out['hypotheses'][0]
        self.assertEqual((h['status'], h['probability_pct'], h['score_raw'], h['probability_band']),
                         ('open', 0.0, 0.0, 'unknown'))
        self.assertEqual(h['checks'], [])
        self.assertEqual(h['checks_json'], '[]')
        self.assertEqual(out['question']['application_reporting_version'], continuity.REPORTING_VERSION)
        self.assertEqual(out['question']['target_hypotheses'], ['H1'])
        self.assertNotIn(rejected_text, refs.canonical(out))
        self.assertIn(rejected_text, refs.canonical(seal['input_step']))
        self.assertTrue(meta['rejected_question_discarded'])
        self.assertFalse(meta['question_validated'])
        self.assertFalse(out['final_ready'])
        self.assertEqual(out['final_result'], {})

    def test_accepted_question_with_no_surviving_target_uses_same_reporting_path(self):
        self.step['question']['target_hypotheses'] = ['H2']
        out, _, meta = self.resolve(reject_question=False)
        self.assertTrue(meta['question_validated'])
        self.assertFalse(meta['rejected_question_discarded'])
        self.assertEqual(out['question']['target_hypotheses'], ['H1'])
        self.assertEqual(meta['application_question'], continuity.REPORTING_VERSION)

    def test_valid_unweighted_clarification_is_preserved(self):
        self.step['question']['target_hypotheses'] = ['H1']
        out, _, meta = self.resolve(reject_question=False)
        self.assertIsNone(meta['application_question'])
        self.assertEqual(out['question']['question_text'], self.step['question']['question_text'])
        self.assertEqual(out['hypotheses'][0]['checks'], self.step['hypotheses'][0]['checks'])

    def test_no_supported_cause_remains_a_rejection(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'no_supported_hypotheses'):
            self.resolve(rejected=(0, 1, 2, 3))

    def test_invalid_current_proof_cannot_enter_reporting(self):
        prepared = self.prepared()
        parsed = parsed_review(prepared, rejected=(1, 2, 3), reject_question=True)
        parsed['decisions'][0]['proofs'][0]['source_units'] = [999999]
        with self.assertRaises(refs.ReferenceError):
            review.resolve(prepared=prepared, parsed=parsed, probability_band=self.band,
                           continuity_policy=continuity.POLICY_VERSION)

    def test_zero_weight_non_unknown_or_non_open_states_do_not_gain_recovery(self):
        for change in ({'status': 'likely'}, {'probability_band': 'high'}):
            with self.subTest(change=change):
                original = deepcopy(self.step)
                self.step['hypotheses'][0].update(change)
                with self.assertRaisesRegex(review.SmartReviewError, 'no_positive_supported_hypothesis'):
                    self.resolve()
                self.step = original

    def test_cancelled_nonzero_weights_do_not_qualify_as_unweighted(self):
        self.step['hypotheses'][0]['probability_pct'] = -10
        self.step['hypotheses'][1]['probability_pct'] = 10
        with self.assertRaisesRegex(review.SmartReviewError, 'no_positive_supported_hypothesis'):
            self.resolve(rejected=(2, 3))

    def test_terminal_or_inconsistent_turn_cannot_gain_reporting(self):
        for status, final_ready in (('completed', False), ('in_progress', True), ('completed', True)):
            with self.subTest(status=status, final_ready=final_ready):
                self.step.update(status=status, final_ready=final_ready)
                with self.assertRaisesRegex(review.SmartReviewError, 'no_positive_supported_hypothesis'):
                    self.resolve()

    def test_legacy_policy_keeps_rejected_question_closed(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'question_not_supported'):
            self.resolve(policy=None)

    def test_reporting_unknown_replays_identical_causes_without_new_observations(self):
        out, seal, _ = self.resolve()
        state = self.state(out, seal)
        replayed, _ = review.replay(state=state, packet=self.packet, probability_band=self.band)
        self.assertEqual(review.claims_digest(out), review.claims_digest(replayed))
        final, next_seal, history, _ = review.complete_reporting_unknown(state=state,
            packet=self.packet, answer={'value': 'unknown', 'free_text': 'An unverified new diagnosis.'},
            probability_band=self.band)
        self.assertTrue(final['final_ready'])
        self.assertEqual(final['question'], {})
        self.assertEqual(final['hypotheses'], out['hypotheses'])
        self.assertEqual(seal['reference_fingerprint'], next_seal['reference_fingerprint'])
        self.assertEqual(len(history), len(self.history) + 1)
        self.assertNotIn('unverified', review.observation_text('Reported stop.', history).lower())

    def test_tampered_or_foreign_reporting_state_fails_replay(self):
        out, seal, _ = self.resolve()
        state = self.state(out, seal)
        state['current_question']['question_text'] = 'Inspect the moving mechanism.'
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_claims_changed'):
            review.replay(state=state, packet=self.packet, probability_band=self.band)
        foreign = deepcopy(self.packet)
        foreign['validator_records'][0]['company_id'] = 'foreign-company'
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_packet_changed'):
            review.replay(state=self.state(out, seal), packet=foreign, probability_band=self.band)


if __name__ == '__main__':
    unittest.main()
