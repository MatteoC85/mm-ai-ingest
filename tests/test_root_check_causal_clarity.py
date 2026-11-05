"""Offline immutable check selection for ambiguous causal discriminators.

Reviewer verdicts here are explicit fixtures, not simulated proof that a model
will always follow the instruction. No lexical auto-rewrite or extra calls.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality
import test_root_response_quality as fixture_module

SAFE = ('If a pressure indication is already recorded, report it. '
        'Recorded pressure loss would support the loss-of-clamping-pressure hypothesis; '
        'normal recorded clamping pressure would weaken the loss-of-clamping-pressure hypothesis.')
AMBIGUOUS = ('Normal clamping pressure would weaken pressure loss; '
             'a high filter pressure difference would support a restricted filter and weaken it.')
CLEAR_ALTERNATIVES = ('Normal recorded clamping pressure would weaken loss of clamping pressure. '
                      'A recorded high filter pressure difference would support a restricted filter.')

def fixture(check=AMBIGUOUS, selected=(0,)):
    observed = 'Model Z feed slips while its motor continues running; pressure readings are unknown.'
    records = [{'citation_id': 'model-z-manual', 'ownership_fragments': [],
        'text': ('Model Z: loss of clamping pressure allows the feed to slip while the motor runs. '
                 'Normal clamping pressure argues against pressure loss. '
                 'A restricted filter reduces flow and causes a high filter pressure difference. '
                 'Use pressure indications already recorded; absent readings remain unknown.')}]
    proposal = {'cause': 'Possible loss of clamping pressure',
        'why': 'The documented mechanism could explain the slip; no pressure reading is available.',
        'checks': [SAFE, check]}
    manifest = review_decisions.manifest([proposal], records, max_causes=3)
    packet = {'sources': [{'citation_id': row['citation_id'], 'text': row['text'], 'context_ids': []}
                          for row in records], 'document_contexts': []}
    frozen = refs.prepare(packet=packet, proposal_manifest=manifest, records=records,
        original_query=observed, observed_query=observed)['frozen']
    selected = list(selected)
    blocked = [index for index in range(2) if index not in selected]
    parsed = {'decisions': [{'proposal_index': 0, 'verdict': 'accept', 'reason': 'supported',
        'selected_check_indices': selected, 'blocking_checks': blocked, 'safe_check_subset': True,
        'selected_why_indices': [u['why_index'] for u in quality.why_units(proposal['why'])],
        'omitted_why_indices': [], 'safe_why_subset': True,
        'note': 'Causal referent in the omitted check is ambiguous.' if blocked else '',
        'proofs': [{'source_index': 0, 'supports_cause': True, 'observation_units': frozen['observed_ids'],
            'source_units': frozen['source_sets'][0], 'target_units': frozen['target_sets'][0],
            'applicability': 'same_target', 'support_type': 'bounded_inference', 'check_indices': selected}]}],
        'ranked_proposal_indices': [0]}
    return frozen, records, observed, parsed, proposal

class CausalCheckSelectionTests(unittest.TestCase):
    def test_reviewer_omits_ambiguous_check_only_and_preserves_useful_cause_and_check(self):
        frozen, records, observed, parsed, proposal = fixture()
        before = deepcopy(frozen)
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(len(result['causes']), 1)
        cause = result['causes'][0]
        self.assertEqual(cause['cause'], proposal['cause'])
        self.assertEqual(cause['why'], proposal['why'])
        self.assertEqual(cause['checks'], [SAFE])
        self.assertEqual(cause['citations'], ['model-z-manual'])
        self.assertNotIn(AMBIGUOUS, str(result['causes']))
        self.assertEqual(result['summary']['check_selection'][0]['blocking_checks'], [1])
        self.assertEqual(result['summary']['support_proofs'][0]['decision']['check_indices'], [0])
        self.assertFalse(result['summary']['semantic_truth_verified_by_code'])
        self.assertEqual(frozen, before)

    def test_clear_named_alternative_comparison_survives_without_rewriting(self):
        frozen, records, observed, parsed, proposal = fixture(CLEAR_ALTERNATIVES, (0, 1))
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['checks'], proposal['checks'])
        self.assertEqual(result['summary']['check_selection'][0]['blocking_checks'], [])

    def test_clear_same_cause_pronoun_is_retained_without_repeating_its_label(self):
        clear_pronoun = ('Recorded low clamping pressure would support loss of clamping pressure. '
                         'Normal recorded clamping pressure would weaken it.')
        frozen, records, observed, parsed, proposal = fixture(clear_pronoun, (0, 1))
        before = deepcopy(frozen)
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['checks'], proposal['checks'])
        self.assertEqual(result['causes'][0]['checks'][1], clear_pronoun)
        self.assertEqual(result['summary']['check_selection'][0]['blocking_checks'], [])
        self.assertEqual(frozen, before)

    def test_contradictory_check_is_omitted_only_by_explicit_review_decision(self):
        contradictory = ('Normal clamping pressure would weaken loss of clamping pressure; '
                         'the same normal reading also supports loss of clamping pressure.')
        frozen, records, observed, parsed, _ = fixture(contradictory)
        parsed['decisions'][0]['note'] = 'The inference contradicts the cited relationship.'
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['checks'], [SAFE])
        self.assertNotIn(contradictory, str(result['causes']))

    def test_success_after_correction_does_not_launder_a_false_historical_exclusion(self):
        temporal_error = ('If an existing record shows pressure was restored and the feed then ran '
                          'without slip, this weakens loss of clamping pressure as the original cause.')
        frozen, records, observed, parsed, proposal = fixture(temporal_error)
        parsed['decisions'][0]['note'] = ('A result after pressure restoration cannot exclude pressure '
                                         'loss during the earlier failure; independent first check retained.')
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['cause'], proposal['cause'])
        self.assertEqual(result['causes'][0]['why'], proposal['why'])
        self.assertEqual(result['causes'][0]['checks'], [SAFE])
        self.assertEqual(result['summary']['check_selection'][0]['blocking_checks'], [1])
        self.assertNotIn(temporal_error, str(result['causes']))
        self.assertEqual(frozen['proposals'][0]['checks'][1]['text'], temporal_error)

    def test_existing_time_specific_observation_can_survive_without_causal_prediction(self):
        observation_only = ('Is a clamping-pressure reading already recorded for the time of the slip, '
                            'rather than only after a later change? If not, the earlier pressure remains '
                            'unknown; do not run a new cycle to obtain it.')
        frozen, records, observed, parsed, proposal = fixture(observation_only, (0, 1))
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['checks'], proposal['checks'])
        self.assertEqual(result['causes'][0]['checks'][1], observation_only)
        self.assertEqual(result['summary']['check_selection'][0]['blocking_checks'], [])
        self.assertEqual(result['summary']['support_proofs'][0]['decision']['check_indices'], [0, 1])

    def test_missing_decision_or_proof_for_discarded_check_fails_closed(self):
        frozen, records, observed, parsed, _ = fixture()
        bad = deepcopy(parsed)
        bad['decisions'][0]['blocking_checks'] = []
        with self.assertRaisesRegex(refs.ReferenceError, 'incomplete_check_partition'):
            quality.validate_review(parsed=bad, frozen=frozen, records=records, observed_query=observed)
        bad = deepcopy(parsed)
        bad['decisions'][0]['proofs'][0]['check_indices'] = [0, 1]
        with self.assertRaisesRegex(refs.ReferenceError, 'invalid_check_selection_indices'):
            quality.validate_review(parsed=bad, frozen=frozen, records=records, observed_query=observed)

    def test_supported_cause_can_survive_without_checks_but_not_without_causal_proof(self):
        frozen, records, observed, parsed, _ = fixture(selected=())
        result = quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)
        self.assertEqual(result['causes'][0]['checks'], [])
        parsed['decisions'][0]['proofs'][0].update(supports_cause=False, observation_units=[], support_type='documented_check')
        with self.assertRaises(refs.ReferenceError):
            quality.validate_review(parsed=parsed, frozen=frozen, records=records, observed_query=observed)

    def test_actual_root_stage_keeps_good_check_and_has_no_auto_rewrite_or_retry(self):
        helper = fixture_module.RootReviewMainIntegrationTests()
        helper.setUp()
        helper.draft['possible_causes'] = helper.draft['possible_causes'][:1]
        helper.draft['possible_causes'][0]['checks'] = [SAFE, AMBIGUOUS]
        helper.parsed['decisions'] = helper.parsed['decisions'][:1]
        helper.parsed['ranked_proposal_indices'] = [0]
        decision = helper.parsed['decisions'][0]
        decision.update(selected_check_indices=[0], blocking_checks=[1],
                        note='Ambiguous causal referent in second check; first check stands alone.')
        decision['proofs'][0]['check_indices'] = [0]
        original = deepcopy(helper.draft)
        out = helper.run_response()
        self.assertTrue(out['ok'])
        self.assertEqual(len(helper.calls), 1)
        self.assertEqual(helper.calls[0][0][0]['content'], quality.review_instruction())
        self.assertIn(AMBIGUOUS, helper.calls[0][0][1]['content'])
        self.assertEqual(out['possible_causes'][0]['checks'], [SAFE])
        self.assertEqual(out['recommended_next_checks'], [SAFE])
        self.assertEqual(out['possible_causes'][0]['cause'], original['possible_causes'][0]['cause'])
        self.assertEqual(out['possible_causes'][0]['why'], original['possible_causes'][0]['why'])
        self.assertEqual(helper.draft, original)
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertEqual(helper.no_evidence_calls, [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
