"""Offline Smart review search hints; hints never supply semantic acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import smart_review as review, smart_evidence, review_references as refs


class SmartReviewSourceHintTests(unittest.TestCase):
    def setUp(self):
        self.scope = {'company_id': 'fixture-company', 'machine_id': 'fixture-machine', 'ai_scope': 'machine_all'}
        texts = [
            'The delivery message can be a notice or a trip. Record the circuit and message already observed; the cause is unknown.',
            'A different subsystem has its own diagnostic procedure. Its limits do not apply to the delivery circuit.',
            'Message: Delivery rate. Type: Notice. Description: The delivery rate is in the caution range. Use only the existing recorded message.',
            'Message: Delivery rate. Type: Trip. Description: The delivery rate is outside the specified range. Use only the existing recorded message.']
        self.sources = [{'citation_id': f'fixture-source-{i}', 'bubble_document_id': f'fixture-doc-{i}',
            'company_id': self.scope['company_id'], 'machine_id': self.scope['machine_id'],
            'source_type': 'document', 'page_from': i + 1, 'page_to': i + 1, 'chunk_full': text}
            for i, text in enumerate(texts)]
        self.step = {'status': 'in_progress', 'final_ready': False, 'operator_summary': '', 'final_result': {},
            'hypotheses': [{'id': 'H1', 'rank': 1, 'label': 'Possible delivery condition outside its normal range',
                'description': 'A notice marks a caution range; a trip marks a value outside the specified range.',
                'why': 'The message is reported, but its type and actual reading remain unknown.',
                'status': 'open', 'probability_pct': 0, 'probability_band': 'unknown', 'checks': [],
                'evidence_ids': ['fixture-source-3', 'fixture-source-0', 'fixture-source-2']}],
            'question': {'question_id': 'Q1', 'question_number': 1, 'question_type': 'single_choice',
                'question_text': 'From information already recorded, was the message a notice or a trip? Do not inspect now.',
                'why_asked': 'Its documented message type distinguishes the two ranges; the type is unknown.',
                'safety_level': 'normal', 'safety_note': '', 'target_hypotheses': ['H1'],
                'options': [{'id': 'notice', 'label_en': 'Notice', 'label_it': 'Avviso'},
                            {'id': 'trip', 'label_en': 'Trip', 'label_it': 'Arresto'},
                            {'id': 'unknown', 'label_en': 'Unknown / cannot check safely', 'label_it': 'Non verificabile in sicurezza'}]}}
        self.symptom = 'A delivery message was reported; its type and reading are not known.'
        self.packet = smart_evidence.review_packet(smart_evidence.build(self.sources, scope=self.scope))
        self.prepared = self.prepare()

    def prepare(self, step=None, packet=None):
        return review.prepare(step=step or self.step, packet=packet or self.packet,
            symptom_text=self.symptom, history=[], language='en')

    def decisions(self, *, rejected=False):
        frozen = self.prepared['references']['frozen']
        rows = []
        for proposal in frozen['proposals']:
            question = proposal['kind'] == 'diagnostic_question'
            rows.append({'proposal_index': proposal['proposal_index'],
                'verdict': 'reject' if rejected else 'accept',
                'reason': ('wrong_target' if question else 'unsupported') if rejected else 'supported',
                'blocking_checks': [], 'note': 'Unsupported proposed distinction.' if rejected else '',
                'proofs': [] if rejected else [{
                    'source_index': index, 'supports_cause': not question,
                    'observation_units': [] if question else frozen['observed_ids'],
                    'source_units': frozen['source_sets'][index], 'target_units': frozen['target_sets'][index],
                    'applicability': 'same_target', 'support_type': 'documented_check' if question else 'bounded_inference',
                    'check_indices': [c['check_index'] for c in proposal['checks']]}
                    for index in (2, 3)]})
        return {'decisions': rows}

    def test_exact_admitted_draft_ids_guide_cause_and_declared_question(self):
        self.assertEqual(review.proposal_source_hints(self.prepared), [
            {'proposal_index': 0, 'source_indices': [0, 2, 3]},
            {'proposal_index': 1, 'source_indices': [0, 2, 3]}])
        messages = review.messages(self.prepared, language='en', symptom_text=self.symptom, history=[])
        transmitted = json.loads(messages[1]['content'].split('PROPOSAL_SOURCE_HINTS: ', 1)[1].split('\n', 1)[0])
        self.assertEqual(transmitted, review.proposal_source_hints(self.prepared))
        for text in (self.sources[2]['chunk_full'], self.sources[3]['chunk_full']):
            self.assertIn(text, messages[1]['content'])

    def test_unknown_aliases_and_private_ids_are_not_emitted_or_authorized(self):
        step = deepcopy(self.step)
        step['hypotheses'][0]['evidence_ids'] = ['fixture-source-2', 'fixture-source-2',
            'fixture-source-2 ', 'other-tenant-private-source', 'https://private.invalid/token', 3, True]
        step['hypotheses'][0]['source_indices'] = [1]
        step['question']['target_hypotheses'] += ['not-a-hypothesis']
        prepared = self.prepare(step)
        before = deepcopy(prepared['references']['frozen'])
        hints = review.proposal_source_hints(prepared)
        self.assertEqual([row['source_indices'] for row in hints], [[2], [2]])
        self.assertNotIn('private', json.dumps(hints))
        self.assertEqual(prepared['references']['frozen'], before)

    def test_missing_or_malformed_ids_are_empty_hints_not_a_rejection(self):
        for value in (None, [], 'fixture-source-2', {'fixture-source-2': True}):
            with self.subTest(value=value):
                step = deepcopy(self.step)
                step['hypotheses'][0]['evidence_ids'] = value
                prepared = self.prepare(step)
                self.assertEqual(prepared['references']['frozen']['fingerprint'], self.prepared['references']['frozen']['fingerprint'])
                self.assertEqual([r['source_indices'] for r in review.proposal_source_hints(prepared)], [[], []])
                # Semantic decisions remain independent of hints, including valid rebinding.
                out, _, _ = review.resolve(prepared=prepared, parsed=self.decisions(),
                    probability_band=lambda p: 'unknown', continuity_policy=review.continuity.POLICY_VERSION)
                self.assertEqual(out['hypotheses'][0]['evidence_ids'], ['fixture-source-2', 'fixture-source-3'])
                self.assertEqual(out['hypotheses'][0]['probability_pct'], 0)

    def test_packet_reordering_rebinds_existing_indices_without_cross_source_units(self):
        packet = smart_evidence.review_packet(smart_evidence.build(list(reversed(self.sources)), scope=self.scope))
        prepared = self.prepare(packet=packet)
        hints = review.proposal_source_hints(prepared)[0]['source_indices']
        manifest = prepared['references']['frozen']['source_manifest']
        self.assertEqual({manifest[index]['citation_id'] for index in hints}, set(self.step['hypotheses'][0]['evidence_ids']))
        self.assertEqual(len(hints), 3)

    def test_prompt_keeps_complete_packet_claims_authorizations_and_seal_unchanged(self):
        before = deepcopy(self.prepared)
        parsed = self.decisions()
        old_out, old_seal, _ = review.resolve(prepared=self.prepared, parsed=parsed,
            probability_band=lambda p: 'unknown', continuity_policy=review.continuity.POLICY_VERSION)
        schema = review.wire_schema(self.prepared)
        messages = review.messages(self.prepared, language='en', symptom_text=self.symptom, history=[])
        table = json.loads(messages[1]['content'].split('REVIEW_PACKET: ', 1)[1].rsplit('\nReturn decisions only.', 1)[0])
        decoded = {k: v for k, v in table.items() if k != 'SOURCE_TABLE'}
        decoded['sources'] = [dict(zip(table['SOURCE_TABLE']['columns'], row)) for row in table['SOURCE_TABLE']['rows']]
        self.assertEqual(decoded, before['references']['model_packet'])
        self.assertEqual(self.prepared, before)
        self.assertEqual(review.wire_schema(self.prepared), schema)
        out, seal, _ = review.resolve(prepared=self.prepared, parsed=parsed,
            probability_band=lambda p: 'unknown', continuity_policy=review.continuity.POLICY_VERSION)
        self.assertEqual((out, seal), (old_out, old_seal))
        self.assertIn('NOT evidence, proof', messages[0]['content'])
        self.assertIn('Question hints come from its declared target hypotheses, not accepted targets', messages[0]['content'])

    def test_actual_semantic_rejection_is_preserved_despite_matching_hints(self):
        review.messages(self.prepared, language='en', symptom_text=self.symptom, history=[])
        with self.assertRaisesRegex(review.SmartReviewError, 'no_supported_hypotheses') as caught:
            review.resolve(prepared=self.prepared, parsed=self.decisions(rejected=True),
                probability_band=lambda p: 'unknown', continuity_policy=review.continuity.POLICY_VERSION)
        self.assertEqual(caught.exception.validated_summary['accepted_causes'], 0)

    def test_hinted_source_cannot_replace_missing_or_borrowed_proof(self):
        for mutation in ('no_proof', 'borrowed_units', 'question_without_proof'):
            with self.subTest(mutation=mutation):
                parsed = self.decisions()
                if mutation == 'no_proof':
                    parsed['decisions'][0]['proofs'] = []
                elif mutation == 'question_without_proof':
                    parsed['decisions'][1]['proofs'] = []
                else:
                    parsed['decisions'][0]['proofs'][0]['source_units'] = self.prepared['references']['frozen']['source_sets'][1]
                with self.assertRaises(refs.ReferenceError):
                    review.resolve(prepared=self.prepared, parsed=parsed, probability_band=lambda p: 'unknown',
                        continuity_policy=review.continuity.POLICY_VERSION)

    def test_scope_mismatch_is_rejected_before_any_hint_exists(self):
        for key in ('company_id', 'machine_id'):
            with self.subTest(key=key):
                sources = deepcopy(self.sources)
                sources[0][key] = 'another-scope'
                with self.assertRaises(smart_evidence.SmartEvidenceError):
                    smart_evidence.build(sources, scope=self.scope)


if __name__ == '__main__':
    unittest.main(verbosity=2)
