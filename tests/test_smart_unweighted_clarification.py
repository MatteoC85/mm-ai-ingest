"""Reviewed questions can clarify unknown causes without inventing probabilities."""
from copy import deepcopy
from pathlib import Path
import sys, unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.retrieval import smart_review as review, smart_continuity as continuity
from test_smart_review import fixture, parsed_review, smart_evidence, refs


class UnweightedClarificationTests(unittest.TestCase):
    def setUp(self):
        sources, self.step = fixture()
        self.packet = smart_evidence.review_packet(smart_evidence.build(sources,
            scope=dict(company_id='fixture-company', machine_id='fixture-machine', ai_scope='machine_all')))
        self.symptom = 'Reported symptom; physical circuit and measured state remain unknown.'
        for h in self.step['hypotheses']:
            h.update(probability_pct=0.0, probability_band='unknown', score_raw=0.0, status='open')
        self.band = lambda value: 'unknown' if value == 0 else 'high'

    def resolve(self, *, rejected=(), reject_question=False, policy=continuity.POLICY_VERSION, history=None):
        prepared = review.prepare(step=self.step, packet=self.packet, symptom_text=self.symptom,
                                  history=history or [])
        parsed = parsed_review(prepared, rejected=rejected, reject_question=reject_question)
        return review.resolve(prepared=prepared, parsed=parsed, probability_band=self.band,
                              continuity_policy=policy)

    def state(self, out, seal, history=None):
        return dict(grounding_review=seal, status=out['status'], symptom_text=self.symptom,
                    language='en', history=history or [], hypotheses=out['hypotheses'],
                    current_question=out['question'])

    def test_fully_reviewed_question_retains_zero_unknown_weights_and_claims(self):
        out, seal, summary = self.resolve()
        self.assertFalse(out['final_ready'])
        self.assertEqual(out['question']['question_text'], self.step['question']['question_text'])
        self.assertTrue(summary['question_validated'])
        self.assertIsNone(summary['application_question'])
        self.assertEqual([h['label'] for h in out['hypotheses']], [h['label'] for h in self.step['hypotheses']])
        for h in out['hypotheses']:
            self.assertEqual((h['status'], h['probability_pct'], h['score_raw'], h['probability_band']),
                             ('open', 0.0, 0.0, 'unknown'))
        replayed, _ = review.replay(state=self.state(out, seal), packet=self.packet, probability_band=self.band)
        self.assertEqual(review.claims_digest(out), review.claims_digest(replayed))

    def test_rejected_question_cannot_use_zero_weight_clarification(self):
        out, _, summary = self.resolve(reject_question=True)
        self.assertFalse(summary['question_validated'])
        self.assertTrue(summary['rejected_question_discarded'])
        self.assertEqual(summary['application_question'], continuity.REPORTING_VERSION)
        self.assertNotEqual(out['question']['question_text'], self.step['question']['question_text'])
        self.assertTrue(all(h['checks'] == [] for h in out['hypotheses']))

    def test_rejected_or_foreign_target_cannot_use_zero_weight_clarification(self):
        for target in ('H2', 'foreign-target'):
            self.step['question']['target_hypotheses'] = [target]
            out, _, summary = self.resolve(rejected=(1,))
            self.assertEqual(summary['application_question'], continuity.REPORTING_VERSION)
            self.assertEqual(out['question']['target_hypotheses'], ['H1', 'H3', 'H4'])
            self.assertNotIn(target, out['question']['target_hypotheses'])
            self.assertTrue(all(h['probability_pct'] == 0 for h in out['hypotheses']))

    def test_all_rejected_cannot_publish_prior_or_unreviewed_causes(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'no_supported_hypotheses'):
            self.resolve(rejected=(0, 1, 2, 3))

    def test_confirmed_nonunknown_and_terminal_shapes_are_not_admitted(self):
        original = deepcopy(self.step)
        for change in (lambda s: s['hypotheses'][0].update(status='confirmed'),
                       lambda s: s['hypotheses'][0].update(probability_band='high'),
                       lambda s: s.update(status='completed', final_ready=True)):
            self.step = deepcopy(original); change(self.step)
            with self.assertRaisesRegex(review.SmartReviewError, 'no_positive_supported_hypothesis'):
                self.resolve()

    def test_legacy_policy_cannot_divide_zero_or_change_old_seals(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'no_positive_supported_hypothesis'):
            self.resolve(policy=None)

    def test_bad_source_proof_still_rejected_before_probability_projection(self):
        prepared = review.prepare(step=self.step, packet=self.packet, symptom_text=self.symptom, history=[])
        parsed = parsed_review(prepared)
        parsed['decisions'][0]['proofs'][0]['source_units'] = [999999]
        with self.assertRaises(refs.ReferenceError):
            review.resolve(prepared=prepared, parsed=parsed, probability_band=self.band,
                           continuity_policy=continuity.POLICY_VERSION)

    def test_unknown_history_is_not_new_evidence_and_replay_scope_is_preserved(self):
        history = [{'question': {'question_text': 'What was already observed?'},
                    'answer': {'value': 'unknown', 'free_text': 'The pump has certainly failed.'}}]
        out, seal, _ = self.resolve(history=history)
        self.assertNotIn('certainly failed', review.observation_text(self.symptom, history))
        self.assertTrue(all(h['probability_pct'] == 0 and h['probability_band'] == 'unknown' for h in out['hypotheses']))
        state = self.state(out, seal, history)
        changed = deepcopy(self.packet)
        changed['validator_records'][0]['company_id'] = 'foreign-company'
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_packet_changed'):
            review.replay(state=state, packet=changed, probability_band=self.band)
        state['hypotheses'][0]['probability_pct'] = 100
        with self.assertRaisesRegex(review.SmartReviewError, 'reviewed_claims_changed'):
            review.replay(state=state, packet=self.packet, probability_band=self.band)


if __name__ == '__main__':
    unittest.main(verbosity=2)
