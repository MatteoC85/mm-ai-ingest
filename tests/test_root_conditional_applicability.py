"""Offline conditional Root proofs and rejection boundaries; no model calls.

These tests exercise immutable proof validation and the real Root review caller.
They do not claim to prove a model will interpret the clarification correctly.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality
import test_root_response_quality as root_test_fixture


def conditional_fixture():
    observed = 'In the Model Q documentary case, WAIT appeared in VERIFY and ordinary acknowledgement did not clear it.'
    query = observed + ' The relay option and contact state are unverified; what is a documented conditional explanation?'
    record = {'citation_id': 'model-q-manual', 'ownership_fragments': [],
        'text': ('Model Q: with the relay option installed, an open secondary contact '
                 'during VERIFY produces WAIT; ordinary acknowledgement does not clear it. '
                 'The relevant observation is the secondary contact state at that transition.')}
    proposal = {
        'cause': 'Possible open secondary contact, only if the relay option is installed',
        'why': ('The documented conditional consequence matches WAIT during VERIFY '
                'without ordinary acknowledgement clearing it. The relay option and '
                'contact state remain unverified; this is not a confirmed fault.'),
        'checks': ['If an observation of the contact state at that transition is already available, '
                   'report it. Otherwise leave the condition unknown; do not run a new cycle.'],
    }
    records = [record]
    manifest = review_decisions.manifest([proposal], records, max_causes=3)
    packet = {'sources': [{'citation_id': record['citation_id'], 'text': record['text'], 'context_ids': []}],
              'document_contexts': []}
    frozen = refs.prepare(packet=packet, proposal_manifest=manifest, records=records,
                          original_query=query, observed_query=observed)['frozen']
    parsed = {'decisions': [{'proposal_index': 0, 'verdict': 'accept', 'reason': 'supported',
        'selected_check_indices': [0], 'safe_check_subset': True, 'blocking_checks': [], 'note': '',
        'selected_why_indices': [u['why_index'] for u in quality.why_units(proposal['why'])],
        'omitted_why_indices': [], 'safe_why_subset': True,
        'proofs': [{'source_index': 0, 'supports_cause': True,
            'observation_units': frozen['observed_ids'], 'source_units': frozen['source_sets'][0],
            'target_units': frozen['target_sets'][0], 'applicability': 'same_target',
            'support_type': 'bounded_inference', 'check_indices': [0]}]}],
        'ranked_proposal_indices': [0]}
    return frozen, records, observed, parsed, proposal


class ConditionalRootProofTests(unittest.TestCase):
    def setUp(self):
        self.frozen, self.records, self.observed, self.parsed, self.proposal = conditional_fixture()

    def validate(self):
        return quality.validate_review(parsed=self.parsed, frozen=self.frozen,
            records=self.records, observed_query=self.observed)

    def test_reviewed_conditional_hypothesis_preserves_unknown_wording_and_bound_proofs(self):
        before = deepcopy(self.frozen)
        result = self.validate()
        cause = result['causes'][0]
        for field in ('cause', 'why', 'checks'):
            self.assertEqual(cause[field], self.proposal[field])
        self.assertEqual(cause['citations'], ['model-q-manual'])
        self.assertEqual(result['summary']['support_proofs'][0]['decision']['observation_units'],
                         self.frozen['observed_ids'])
        self.assertFalse(result['summary']['semantic_truth_verified_by_code'])
        self.assertEqual(self.frozen, before)

    def test_conditional_language_never_overrides_a_reviewer_rejection(self):
        for reason in ('unknown_target', 'wrong_target', 'contradicted', 'unsupported_check'):
            with self.subTest(reason=reason):
                self.parsed['decisions'][0].update(verdict='reject', reason=reason,
                    selected_check_indices=[], safe_check_subset=False, proofs=[],
                    selected_why_indices=[], safe_why_subset=False,
                    omitted_why_indices=[u['why_index'] for u in quality.why_units(self.proposal['why'])],
                    blocking_checks=[0], note='Reviewer found applicability or support insufficient.')
                self.parsed['ranked_proposal_indices'] = []
                result = self.validate()
                self.assertEqual(result['causes'], [])
                self.assertFalse(result['summary']['verdicts'][0]['accepted'])
                self.assertEqual(result['summary']['verdicts'][0]['reason'], reason)

    def test_unknown_condition_cannot_replace_required_observation_or_causal_support(self):
        proof = self.parsed['decisions'][0]['proofs'][0]
        proof.update(supports_cause=False, support_type='documented_check', observation_units=[])
        with self.assertRaisesRegex(refs.ReferenceError, 'mechanism_not_supported'):
            self.validate()

    def test_conditional_draft_does_not_authorize_unknown_or_borrowed_source_units(self):
        self.parsed['decisions'][0]['proofs'][0]['source_units'] = [999999]
        with self.assertRaises(refs.ReferenceError):
            self.validate()

    def test_retrospective_check_still_requires_its_own_support(self):
        self.parsed['decisions'][0]['proofs'][0]['check_indices'] = []
        with self.assertRaisesRegex(refs.ReferenceError, 'not_all_checks_supported'):
            self.validate()

    def test_no_automatic_qualification_or_repair_of_original_draft(self):
        self.frozen['proposals'][0]['why'] = 'The relay option is certainly installed.'
        with self.assertRaisesRegex(refs.ReferenceError, 'manifest_altered'):
            self.validate()


class RootOnlyInstructionIntegrationTests(unittest.TestCase):
    def test_actual_root_review_receives_clarification_once_without_shared_mutation(self):
        helper = root_test_fixture.RootReviewMainIntegrationTests()
        helper.setUp()
        shared_before = refs.INSTRUCTION
        out = helper.run_response()
        self.assertTrue(out['ok'])
        self.assertEqual(len(helper.calls), 1)
        messages, kwargs = helper.calls[0]
        self.assertEqual(messages[0]['content'], quality.review_instruction())
        self.assertEqual(refs.INSTRUCTION, shared_before)
        self.assertEqual(kwargs['purpose'], 'assistant_core_root_cause_adjudicator')
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')

    def test_actual_root_review_does_not_reinterpret_unknown_target_as_accept(self):
        helper = root_test_fixture.RootReviewMainIntegrationTests()
        helper.setUp()
        for decision in helper.parsed['decisions']:
            decision.update(verdict='reject', reason='unknown_target', selected_check_indices=[],
                safe_check_subset=False, proofs=[], note='Target genuinely not established.',
                selected_why_indices=[], safe_why_subset=False,
                omitted_why_indices=[u['why_index'] for u in quality.why_units(
                    helper.frozen['proposals'][decision['proposal_index']]['why'])])
        helper.parsed['ranked_proposal_indices'] = []
        out = helper.run_response()
        self.assertFalse(out['ok'])
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(len(helper.calls), 1)
        self.assertEqual(helper.no_evidence_calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
