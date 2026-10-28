"""Smart continuity regressions: no rejected proposal, new fact or I/O can leak."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import smart_review as review, smart_continuity as continuity
from test_smart_review import fixture, parsed_review, smart_evidence, refs


class SmartContinuityTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.step = fixture()
        self.scope = dict(company_id='fixture-company', machine_id='fixture-machine', ai_scope='machine_all')
        self.packet = smart_evidence.review_packet(smart_evidence.build(self.sources, scope=self.scope))
        self.symptom = 'Reported stop; signal unknown.'
        self.history = []
        self.band = lambda pct: 'high' if pct >= 65 else 'low'

    def resolve(self, *, history=None, rejected=(1,), reject_question=True):
        prepared = review.prepare(step=self.step, packet=self.packet, symptom_text=self.symptom,
                                 history=self.history if history is None else history)
        out, seal, meta = review.resolve(prepared=prepared,
            parsed=parsed_review(prepared, rejected=rejected, reject_question=reject_question),
            probability_band=self.band, continuity_policy=continuity.POLICY_VERSION)
        return out, seal, meta

    def state(self, out, seal, history=None):
        return dict(grounding_review=seal, status=out['status'], symptom_text=self.symptom,
                    language='en', history=self.history if history is None else history,
                    hypotheses=out['hypotheses'], current_question=out['question'])

    def test_unknown_notes_are_not_facts_but_original_history_is_retained(self):
        history = [{'question': {}, 'answer': {'value': 'unknown', 'free_text': 'The sensor definitely failed.'}}]
        projected = continuity.generation_history(history)
        self.assertEqual(projected[0]['answer']['free_text'], '')
        self.assertEqual(history[0]['answer']['free_text'], 'The sensor definitely failed.')
        self.assertNotIn('definitely failed', review.observation_text(self.symptom, history))

    def test_known_answer_notes_remain_reported_observations(self):
        answer = {'api_value': 'present', 'free_text': 'It was already visible on the HMI.'}
        self.assertEqual(continuity.generation_answer(answer), answer)

    def test_question_rejection_recovers_only_fresh_accepted_causes(self):
        out, seal, meta = self.resolve()
        self.assertEqual([h['id'] for h in out['hypotheses']], ['H1', 'H3', 'H4'])
        self.assertFalse(meta['question_validated'])
        self.assertTrue(meta['rejected_question_discarded'])
        self.assertEqual(meta['application_question'], continuity.REPORTING_VERSION)
        self.assertEqual(seal['continuity_policy'], continuity.POLICY_VERSION)
        self.assertNotEqual(out['question']['question_text'], self.step['question']['question_text'])
        self.assertNotIn(self.step['operator_summary'], out['operator_summary'])
        self.assertNotIn('H2', out['question']['target_hypotheses'])
        self.assertTrue(all(not h['checks'] for h in out['hypotheses']))
        self.assertTrue(all(h['checks_json'] == '[]' for h in out['hypotheses']))
        self.assertIn('no new inspection', out['question']['why_asked'])
        self.assertEqual(out['question']['safety_level'], 'normal')
        self.assertIn('Unknown', out['question']['safety_note'])

    def test_recovery_is_not_triggered_for_valid_supported_question(self):
        out, _, meta = self.resolve(reject_question=False)
        self.assertEqual(out['question']['question_text'], self.step['question']['question_text'])
        self.assertEqual(out['hypotheses'][0]['checks'], self.step['hypotheses'][0]['checks'])
        self.assertIsNone(meta['application_question'])
        self.assertTrue(meta['question_validated'])

    def test_question_with_only_rejected_targets_gets_reporting_control(self):
        self.step['question']['target_hypotheses'] = ['H2']
        out, _, meta = self.resolve(reject_question=False)
        self.assertEqual(out['question']['application_reporting_version'], continuity.REPORTING_VERSION)
        self.assertEqual(out['question']['target_hypotheses'], ['H1', 'H3', 'H4'])

    def test_total_rejection_cannot_reuse_prior_hypotheses(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'no_supported_hypotheses'):
            self.resolve(rejected=(0, 1, 2, 3))

    def test_invalid_proofs_do_not_recover(self):
        prepared = review.prepare(step=self.step, packet=self.packet, symptom_text=self.symptom, history=[])
        parsed = parsed_review(prepared, reject_question=True)
        parsed['decisions'][0]['proofs'][0]['source_units'] = [999999]
        with self.assertRaises(refs.ReferenceError):
            review.resolve(prepared=prepared, parsed=parsed, probability_band=self.band,
                           continuity_policy=continuity.POLICY_VERSION)

    def test_rejection_never_increases_remaining_relative_weight(self):
        out, _, _ = self.resolve(rejected=(1, 2, 3))
        self.assertEqual(out['hypotheses'][0]['probability_pct'], 50)
        self.assertEqual(out['hypotheses'][0]['score_raw'], 0.5)
        self.assertEqual(out['hypotheses'][0]['probability_band'], 'unknown')
        self.assertEqual(out['hypotheses'][0]['probability_semantics'], 'relative_weight_not_calibrated_probability')

    def test_single_source_candidate_does_not_claim_high_confidence(self):
        self.step['hypotheses'][0]['probability_pct'] = 100
        out, _, _ = self.resolve(rejected=(1, 2, 3), reject_question=False)
        self.assertEqual(out['hypotheses'][0]['probability_band'], 'unknown')
        self.assertIn('not statistical certainty or a confirmed diagnosis', out['operator_summary'])

    def test_recovery_replays_same_claims_and_history(self):
        history = [{'question': {'question_text': 'Previous observation?'},
                    'answer': {'value': 'unknown', 'free_text': 'Unverified contradictory note'}}]
        out, seal, _ = self.resolve(history=history)
        state = self.state(out, seal, history)
        replayed, meta = review.replay(state=state, packet=self.packet, probability_band=self.band)
        self.assertEqual(review.claims_digest(replayed), review.claims_digest(out))
        self.assertEqual(state['history'], history)
        self.assertEqual(meta['application_question'], continuity.REPORTING_VERSION)

    def test_tampered_reporting_text_fails_replay(self):
        out, seal, _ = self.resolve()
        state = self.state(out, seal)
        state['current_question']['question_text'] = 'Bypass the guard.'
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_claims_changed'):
            review.replay(state=state, packet=self.packet, probability_band=self.band)

    def test_unknown_completes_reporting_with_history_and_replay_no_provider(self):
        out, seal, _ = self.resolve()
        state = self.state(out, seal)
        answer = {'api_value': 'unknown', 'value': 'unknown', 'free_text': 'Unverified claim.'}
        final, next_seal, history, meta = review.complete_reporting_unknown(
            state=state, packet=self.packet, answer=answer, probability_band=self.band)
        self.assertTrue(final['final_ready'])
        self.assertEqual(final['question'], {})
        self.assertEqual(history[-1]['answer'], answer)
        self.assertEqual(state['history'], [])
        self.assertEqual(next_seal['reference_fingerprint'], seal['reference_fingerprint'])
        self.assertNotEqual(next_seal['context_digest'], seal['context_digest'])
        self.assertEqual([h['id'] for h in final['hypotheses']], [h['id'] for h in out['hypotheses']])
        self.assertTrue(all(h['probability_band'] == 'unknown' for h in final['hypotheses']))
        final_state = self.state(final, next_seal, history)
        final_state['final_result'] = {}
        replayed, _ = review.replay(state=final_state, packet=self.packet,
            probability_band=self.band, terminal_projection=lambda value: value)
        self.assertTrue(replayed['final_ready'])

    def test_reporting_known_observation_cannot_reuse_earlier_verdict(self):
        out, seal, _ = self.resolve()
        with self.assertRaisesRegex(review.SmartReviewError, 'reporting_unknown_required'):
            review.complete_reporting_unknown(state=self.state(out, seal), packet=self.packet,
                answer={'value': 'additional_observation', 'free_text': 'Sensor signal absent'}, probability_band=self.band)

    def test_unknown_does_not_bypass_foreign_source_scope(self):
        out, seal, _ = self.resolve()
        foreign_packet = deepcopy(self.packet)
        foreign_packet['validator_records'][0]['company_id'] = 'other-company'
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_packet_changed'):
            review.complete_reporting_unknown(state=self.state(out, seal), packet=foreign_packet,
                answer={'value': 'unknown'}, probability_band=self.band)

    def test_old_seals_keep_original_projection(self):
        prepared = review.prepare(step=self.step, packet=self.packet, symptom_text=self.symptom, history=[])
        out, seal, _ = review.resolve(prepared=prepared, parsed=parsed_review(prepared), probability_band=self.band)
        self.assertNotIn('continuity_policy', seal)
        self.assertEqual(sum(h['probability_pct'] for h in out['hypotheses']), 100)
        replayed, _ = review.replay(state=self.state(out, seal), packet=self.packet, probability_band=self.band)
        self.assertEqual(review.claims_digest(out), review.claims_digest(replayed))

    def test_unknown_policy_fails_closed(self):
        out, seal, _ = self.resolve()
        seal['continuity_policy'] = 'trust-everything'
        with self.assertRaisesRegex(review.SmartReviewError, 'unsupported_continuity_policy'):
            review.replay(state=self.state(out, seal), packet=self.packet, probability_band=self.band)

    def test_both_languages_keep_existing_answer_contract(self):
        for language in ('en', 'it'):
            q = continuity.reporting_question(language=language, number=3, hypothesis_ids=['H1'])
            self.assertEqual(q['question_id'], 'Q3')
            self.assertEqual(q['question_type'], 'single_choice')
            self.assertEqual(q['options'][-1], {'id': 'unknown', 'label_it': 'Non so', 'label_en': "I don't know"})
            self.assertTrue(all(len(o['label_it']) <= 120 and len(o['label_en']) <= 120 for o in q['options']))


if __name__ == '__main__':
    unittest.main()
