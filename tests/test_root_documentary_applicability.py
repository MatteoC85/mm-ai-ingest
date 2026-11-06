"""Documentary versus installed-target review contracts, without provider calls.

Semantic verdicts below are explicit fixtures, not a simulated model judgment.
The real Root caller must transmit the distinction and honor either verdict;
only the reviewer can establish that a particular mechanism fits the case.
"""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality
import test_root_response_quality as root_fixture


def documentary_fixture(*, installed_claim=False, incompatible=False, actual_case=False):
    observed = ('In a documentary press case, HOLD appears on entering VERIFY '
                'and ordinary acknowledgement does not clear it.')
    query = observed + ' The installed model and relay option are unknown.'
    if actual_case:
        observed = ('My installed press shows HOLD on entering VERIFY and '
                    'ordinary acknowledgement does not clear the alarm.')
        query = observed + ' Its model and configuration are unknown.'
    if incompatible:
        query = observed = ('The installed press is Model R without a relay circuit. '
                            'HOLD appears on entering VERIFY.')
    record = {'citation_id': 'press-case', 'ownership_fragments': [],
        'text': ('Documentary press case for Model Q: with its relay option, an open '
                 'contact during VERIFY causes HOLD, prevents drive enable and '
                 'cannot be cleared by ordinary acknowledgement. The documented '
                 'discriminant is the contact state at that transition. This case '
                 'does not establish the installed model or authorize operation.')}
    unrelated = {'citation_id': 'unrelated-pump', 'ownership_fragments': [],
        'text': 'Pump Model R: HOLD concerns a flow switch, not the press relay circuit.'}
    proposal = {'cause': 'Possible open contact during verification',
        'why': ('For the documented relay-equipped press, an open contact explains '
                'HOLD during verification without ordinary acknowledgement clearing the alarm. '
                'Drive enable was also observed to be unavailable. '
                'Whether this documented configuration applies and whether the contact '
                'was open remain unknown; this is a conditional hypothesis.'),
        'checks': ['If the contact state at that transition was already recorded, '
                   'report it; otherwise leave it unknown.',
                   'Have qualified personnel run a new verification cycle.']}
    if installed_claim:
        proposal['why'] = ('This machine is Model Q with the relay option installed. '
                           'Its open contact caused HOLD.')
    records = [record, unrelated]
    packet = {'sources': [{'citation_id': r['citation_id'], 'text': r['text'],
                           'context_ids': []} for r in records], 'document_contexts': []}
    frozen = refs.prepare(packet=packet,
        proposal_manifest=review_decisions.manifest([proposal], records, max_causes=3),
        records=records, original_query=query, observed_query=observed)['frozen']
    parsed = {'ranked_proposal_indices': [0], 'decisions': [{
        'proposal_index': 0, 'verdict': 'accept', 'reason': 'supported',
        'selected_why_indices': [0, 2], 'omitted_why_indices': [1], 'safe_why_subset': True,
        'selected_check_indices': [0], 'blocking_checks': [1], 'safe_check_subset': True,
        'note': 'Omitted an unreported consequence and operation with unknown applicability.',
        'proofs': [{'source_index': 0, 'supports_cause': True,
                    'observation_units': frozen['observed_ids'],
                    'source_units': frozen['source_sets'][0],
                    'target_units': frozen['target_sets'][0], 'applicability': 'same_target',
                    'support_type': 'bounded_inference', 'check_indices': [0]}]}]}
    return frozen, records, observed, query, packet, proposal, parsed


def reject(parsed, frozen, reason):
    parsed['ranked_proposal_indices'] = []
    parsed['decisions'][0].update(verdict='reject', reason=reason, proofs=[],
        selected_why_indices=[],
        omitted_why_indices=list(range(len(quality.why_units(frozen['proposals'][0]['why'])))),
        safe_why_subset=False, selected_check_indices=[], blocking_checks=[0, 1],
        safe_check_subset=False, note='Reviewer found the target assertion unsupported.')


def caller_for(data):
    frozen, records, observed, query, packet, proposal, parsed = data
    caller = root_fixture.RootReviewMainIntegrationTests()
    caller.setUp()
    caller.frozen, caller.records, caller.raw_records = frozen, records, records
    caller.observed, caller.parsed = observed, parsed
    caller.request.query = query
    caller.decision.missing_information = ['Installed model and relay configuration are unknown.']
    caller.draft['possible_causes'] = [deepcopy(proposal)]
    caller.retrieval = {'citations': [{'citation_id': r['citation_id']} for r in records]}
    caller.run_review.__globals__['_retrieval_review_packet'] = SimpleNamespace(
        POLICY_VERSION='offline-documentary-fixture', ReviewPacketError=ValueError,
        build_review_packet=lambda **kwargs: {'validator_records': records,
            'model_packet': packet, 'summary': {}})
    return caller


class DocumentaryRootReviewTests(unittest.TestCase):
    def test_real_caller_preserves_reviewed_conditional_subset_and_full_review_packet(self):
        data = documentary_fixture()
        frozen, _, _, _, _, proposal, _ = data
        before = deepcopy(frozen)
        shared_instruction = refs.INSTRUCTION
        caller = caller_for(data)
        out = caller.run_response()
        self.assertTrue(out['ok'])
        self.assertEqual(len(caller.calls), 1)
        messages, kwargs = caller.calls[0]
        self.assertEqual(messages[0]['content'], quality.review_instruction())
        self.assertIn('UNKNOWN installed model from a KNOWN incompatible model', messages[0]['content'])
        self.assertIn('mechanism actually applies to installed equipment requires applicable', messages[0]['content'])
        self.assertIn('Pump Model R', messages[1]['content'])
        self.assertIn(proposal['checks'][1], messages[1]['content'])
        self.assertEqual(kwargs['purpose'], 'assistant_core_root_cause_adjudicator')
        self.assertEqual(kwargs['timeout'], 20)
        cause = out['possible_causes'][0]
        self.assertEqual(cause['cause'], proposal['cause'])
        self.assertEqual(cause['why'], ''.join(u['text'] for u in quality.why_units(proposal['why'])
                                             if u['why_index'] in [0, 2]))
        self.assertIn('applies', cause['why'])
        self.assertIn('remain unknown', cause['why'])
        self.assertEqual(cause['checks'], [proposal['checks'][0]])
        self.assertEqual(cause['citations'], ['press-case'])
        self.assertNotIn('Drive enable was also observed', str(out['possible_causes']))
        self.assertNotIn('new verification cycle', str(out['possible_causes']))
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertEqual(out['meta']['root_review_decisions']['quality_policy_version'], quality.POLICY_VERSION)
        self.assertFalse(out['meta']['root_review_decisions']['semantic_truth_verified_by_code'])
        self.assertEqual(refs.INSTRUCTION, shared_instruction)
        self.assertEqual(frozen, before)

    def test_known_incompatible_model_rejection_remains_rejection(self):
        data = documentary_fixture(incompatible=True)
        reject(data[-1], data[0], 'wrong_target')
        self.assert_rejection(caller_for(data), 'wrong_target', 'Model R without a relay circuit')

    def test_unqualified_installed_model_assertion_is_not_repaired_after_rejection(self):
        data = documentary_fixture(installed_claim=True)
        reject(data[-1], data[0], 'unknown_target')
        self.assert_rejection(caller_for(data), 'unknown_target', 'This machine is Model Q')

    def test_plain_actual_question_can_retain_reviewed_source_scoped_hypothesis(self):
        data = documentary_fixture(actual_case=True)
        caller = caller_for(data)
        out = caller.run_response()
        self.assertTrue(out['ok'])
        self.assertIn('documented relay-equipped press', out['possible_causes'][0]['why'])
        self.assertIn('Whether this documented configuration applies', out['possible_causes'][0]['why'])
        self.assertIn('remain unknown', out['possible_causes'][0]['why'])
        self.assertEqual(out['possible_causes'][0]['checks'], [data[5]['checks'][0]])
        self.assertEqual(len(caller.calls), 1)
        messages, _ = caller.calls[0]
        self.assertIn('claim, not whether the request uses a documentary or hypothetical label.', messages[0]['content'])
        self.assertIn('My installed press', messages[1]['content'])
        self.assertNotIn('In a documentary press case', messages[1]['content'])

    def test_actual_case_does_not_override_a_reviewed_identity_gap(self):
        data = documentary_fixture(actual_case=True)
        reject(data[-1], data[0], 'unknown_target')
        self.assert_rejection(caller_for(data), 'unknown_target', 'My installed press')

    def test_conditional_wording_does_not_override_unknown_or_contradicted_review(self):
        for reason in ('unknown_target', 'contradicted', 'unsafe'):
            with self.subTest(reason=reason):
                data = documentary_fixture()
                reject(data[-1], data[0], reason)
                self.assert_rejection(caller_for(data), reason, 'conditional hypothesis')

    def assert_rejection(self, caller, reason, original_text):
        out = caller.run_response()
        self.assertFalse(out['ok'])
        self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(out['possible_causes'], [])
        self.assertFalse(out['meta']['cacheable'])
        self.assertEqual(out['meta']['root_review_decisions']['verdicts'][0]['reason'], reason)
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertEqual(len(caller.calls), 1)
        self.assertIn(original_text, caller.calls[0][0][1]['content'])
        self.assertEqual(caller.no_evidence_calls, [])

    def test_unrelated_target_units_cannot_prove_documentary_applicability(self):
        data = documentary_fixture()
        data[-1]['decisions'][0]['proofs'][0]['target_units'] = data[0]['target_sets'][1]
        self.assert_contract_failure(data)

    def test_conditional_review_still_requires_causal_proof(self):
        data = documentary_fixture()
        data[-1]['decisions'][0]['proofs'] = []
        self.assert_contract_failure(data)

    def test_missing_observations_cannot_be_substituted_for_evidence(self):
        data = documentary_fixture()
        data[-1]['decisions'][0]['proofs'][0]['observation_units'] = []
        self.assert_contract_failure(data)

    def test_omission_still_needs_explicit_complete_why_partition(self):
        data = documentary_fixture()
        data[-1]['decisions'][0]['omitted_why_indices'] = []
        self.assert_contract_failure(data)

    def assert_contract_failure(self, data):
        caller = caller_for(data)
        out = caller.run_response()
        self.assertFalse(out['ok'])
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(len(caller.calls), 1)
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertIn('review_decision_', out['meta']['root_review_execution']['validation_error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
