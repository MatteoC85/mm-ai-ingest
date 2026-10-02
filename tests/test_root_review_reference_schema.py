"""Request-bound provider reference enums and unchanged fail-closed validation.

No live calls. The recorded incident is a rejected out-of-set ID; the exact ID
was not captured. These fixtures reproduce every reference namespace failure.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import review_decisions, review_references as refs


def fixture():
    # The repeated literal deliberately does NOT authorize exchanging unit IDs.
    records = [{'citation_id': 'source-a', 'text': 'Joint mechanism. ' * 55,
                'ownership_fragments': ['Model A context and safety prerequisites.']},
               {'citation_id': 'source-b', 'text': 'Filter mechanism. ' * 55,
                'ownership_fragments': ['Model B context and safety prerequisites.']}]
    packet = {'sources': [{'citation_id': row['citation_id'], 'text': row['text'], 'context_ids': [f'context-{i}']}
                           for i, row in enumerate(records)],
              'document_contexts': [{'id': f'context-{i}', 'fragments': [{'start': 0, 'end': len(row['ownership_fragments'][0]),
                                     'text': row['ownership_fragments'][0]}]} for i, row in enumerate(records)]}
    proposals = [{'cause': 'Possible leaking joint', 'why': 'Admits air; not yet verified.',
                  'checks': ['Check applicable joint.', 'Observe applicable indication.']},
                 {'cause': 'Possible blocked filter', 'why': 'Restricts delivery; not yet verified.', 'checks': []}]
    manifest = review_decisions.manifest(proposals, records, max_causes=3)
    observed = 'Delivery falls while the motor runs.'
    prepared = refs.prepare(packet=packet, proposal_manifest=manifest, records=records,
                            original_query=observed, observed_query=observed)
    frozen = prepared['frozen']
    parsed = {'decisions': [{'proposal_index': i, 'verdict': 'accept', 'reason': 'supported',
                            'blocking_checks': [], 'note': '', 'proofs': [{
                            'source_index': i, 'supports_cause': True,
                            'observation_units': frozen['observed_ids'],
                            'source_units': frozen['source_sets'][i],
                            'target_units': frozen['target_sets'][i],
                            'applicability': 'same_target', 'support_type': 'bounded_inference',
                            'check_indices': list(range(len(proposals[i]['checks'])))}]}
                            for i in range(len(proposals))]}
    return prepared, records, observed, parsed


def proof_branch(schema, proposal_index, source_index):
    decision = schema['schema']['properties']['decisions']['items']['anyOf'][proposal_index]
    assert decision['properties']['proposal_index']['enum'] == [proposal_index]
    ref = decision['properties']['proofs']['items']['$ref']
    branch = schema['schema']['$defs'][ref.rsplit('/', 1)[1]]['anyOf'][source_index]
    assert branch['properties']['source_index']['enum'] == [source_index]
    return branch['properties']


class RequestBoundReferenceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.prepared, self.records, self.observed, self.parsed = fixture()
        self.frozen = self.prepared['frozen']
        self.schema = refs.schema(self.frozen)

    def validate(self, parsed):
        return refs.validate(parsed=parsed, frozen=self.frozen, records=self.records, observed_query=self.observed)

    def test_valid_supported_decisions_are_not_changed(self):
        before = deepcopy(self.frozen)
        result = self.validate(self.parsed)
        self.assertEqual(result['citation_ids'], ['source-a', 'source-b'])
        self.assertEqual(len(result['causes']), 2)
        self.assertEqual(self.frozen, before)
        self.assertFalse(result['summary']['semantic_truth_verified_by_code'])

    def test_every_proposal_and_source_has_exact_authorized_enums(self):
        decisions = self.schema['schema']['properties']['decisions']
        self.assertEqual(decisions['maxItems'], 2)
        self.assertEqual(len(decisions['items']['anyOf']), 2)
        for pi, proposal in enumerate(self.frozen['proposals']):
            for si in range(len(self.records)):
                props = proof_branch(self.schema, pi, si)
                for field, expected in (('source_units', self.frozen['source_sets'][si]),
                                        ('target_units', self.frozen['target_sets'][si]),
                                        ('observation_units', self.frozen['observed_ids'])):
                    self.assertEqual(props[field]['items']['enum'], expected)
                if proposal['checks']:
                    self.assertEqual(props['check_indices']['items']['enum'], list(range(len(proposal['checks']))))
                else:
                    self.assertEqual(props['check_indices']['maxItems'], 0)

    def test_old_global_range_permitted_unassigned_id_new_schema_does_not(self):
        old = refs.schema()['schema']['properties']['decisions']['items']['properties']['proofs']['items']['properties']
        nonexistent_id = len(self.frozen['units']) + 10
        self.assertLess(nonexistent_id, old['source_units']['items']['maximum'])
        for si in range(len(self.records)):
            self.assertNotIn(nonexistent_id, proof_branch(self.schema, 0, si)['source_units']['items']['enum'])

    def test_source_context_observation_and_check_ids_cannot_be_borrowed(self):
        illegal = {
            'source_units': self.frozen['source_sets'][1],
            'target_units': [uid for uid in self.frozen['target_sets'][1] if uid not in self.frozen['source_sets'][1]],
            'observation_units': self.frozen['source_sets'][0],
            'check_indices': [2],
        }
        allowed = proof_branch(self.schema, 0, 0)
        for field, ids in illegal.items():
            with self.subTest(field=field):
                self.assertTrue(set(ids) - set(allowed[field]['items']['enum']))
                bad = deepcopy(self.parsed)
                bad['decisions'][0]['proofs'][0][field] = ids
                with self.assertRaises(refs.ReferenceError) as captured:
                    self.validate(bad)
                self.assertEqual(str(captured.exception), 'reference_outside_authorized_set')
                detail = captured.exception.reference_failure
                self.assertEqual(detail['field'], 'proposal_0.proof_0.source_0.' + field)
                self.assertEqual(detail['submitted_integer_ids'], ids)
                self.assertEqual(detail['allowed_ids'], allowed[field]['items']['enum'])

    def test_context_unit_cannot_be_used_as_cause_excerpt(self):
        context_id = next(uid for uid in self.frozen['target_sets'][0] if uid not in self.frozen['source_sets'][0])
        props = proof_branch(self.schema, 0, 0)
        self.assertIn(context_id, props['target_units']['items']['enum'])
        self.assertNotIn(context_id, props['source_units']['items']['enum'])
        bad = deepcopy(self.parsed)
        bad['decisions'][0]['proofs'][0]['source_units'] = [context_id]
        with self.assertRaisesRegex(refs.ReferenceError, 'reference_outside_authorized_set'):
            self.validate(bad)

    def test_source_local_zero_is_never_remapped_to_global_id(self):
        self.assertNotIn(0, self.frozen['source_sets'][1])
        bad = deepcopy(self.parsed)
        bad['decisions'][1]['proofs'][0]['source_units'] = [0]
        with self.assertRaisesRegex(refs.ReferenceError, 'reference_outside_authorized_set'):
            self.validate(bad)

    def test_rejection_with_zero_checks_has_no_fabricated_check_sentinel(self):
        decision = self.schema['schema']['properties']['decisions']['items']['anyOf'][1]['properties']
        self.assertEqual(decision['blocking_checks']['maxItems'], 0)
        parsed = deepcopy(self.parsed)
        parsed['decisions'][1].update(verdict='reject', reason='unsupported', note='No established applicable mechanism.', proofs=[])
        result = self.validate(parsed)
        self.assertEqual(len(result['causes']), 1)

    def test_wrong_manifest_cannot_widen_schema_or_prompt(self):
        altered = deepcopy(self.frozen)
        altered['source_sets'][0].append(511)
        for function in (refs.schema, refs.authorized_references):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(refs.ReferenceError, 'manifest_altered'):
                function(altered)

    def test_prompt_reference_index_matches_registry_and_has_no_source_text(self):
        index = refs.authorized_references(self.frozen)
        self.assertEqual(index['observation_units'], self.frozen['observed_ids'])
        self.assertEqual(index['sources'][1]['source_units'], self.frozen['source_sets'][1])
        self.assertEqual(index['proposals'][1]['check_indices'], [])
        serialized = refs.canonical(index)
        self.assertNotIn(self.observed, serialized)
        self.assertNotIn('Joint mechanism', serialized)

    def test_validation_telemetry_cannot_echo_malformed_reference_text(self):
        bad = deepcopy(self.parsed)
        bad['decisions'][0]['proofs'][0]['source_units'] = ['private source text or credentials']
        with self.assertRaises(refs.ReferenceError) as captured:
            self.validate(bad)
        self.assertEqual(captured.exception.reference_failure['non_integer_count'], 1)
        self.assertNotIn('private', refs.canonical(captured.exception.reference_failure))


if __name__ == '__main__':
    unittest.main()
