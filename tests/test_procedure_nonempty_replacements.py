"""Nonempty compact edit domains; exact deletion encoding, no provider IO."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.ask import procedure_review as r
import test_procedure_reference_owners as owners
import test_procedure_reference_review as refs


class NonemptyReplacementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = owners.OwnerReferenceTests()
        self.fixture.setUp()

    def wire(self, parts):
        value = deepcopy(self.fixture.wire)
        value.update(outcome='rewrite', reply_mode='edit')
        value['edits'] = [{'block_id': 'b2', 'original': owners.span(1),
                           'replacement_parts': parts}]
        value['operation_boundary']['precedence'][0]['next'] = owners.endpoint(
            'b2', kind='replacement')
        return value

    def test_empty_replacement_is_excluded_for_every_edit_owner(self):
        fixture = self.fixture
        item = fixture.schema['properties']['edits']['items']
        for owner in fixture.catalog['blocks']:
            for count in (0, 1, 16, 17):
                edit = {'block_id': owner, 'original': owners.span(),
                        'replacement_parts': ['Literal text.'] * count}
                self.assertEqual(owners.accepts(edit, item, fixture.schema), 1 <= count <= 16)

    def test_previous_empty_domain_wire_is_rejected_without_conversion(self):
        # Same observed failure class: a deletion and an endpoint at replacement 0.
        # The live model wire/owner was not retained; this is a synthetic counterexample.
        value = self.wire([])
        before = deepcopy(value)
        self.assertFalse(owners.accepts(value['edits'][0],
            self.fixture.schema['properties']['edits']['items'], self.fixture.schema))
        with self.assertRaisesRegex(ValueError, 'procedure_reference_edit_invalid'):
            self.fixture.decoded(value)
        self.assertEqual(value, before)

    def test_decoder_rejects_empty_parts_even_without_a_replacement_endpoint(self):
        for parts in ([], [''], [' \t\n'], ['literal'] * 17):
            value = self.wire(parts)
            value['operation_boundary']['precedence'][0]['next'] = owners.endpoint('b2', 2)
            with self.subTest(parts=parts), self.assertRaisesRegex(
                    ValueError, 'procedure_reference_edit_invalid'):
                self.fixture.decoded(value)

    def test_deletion_outputs_remain_exactly_expressible_with_a_survivor(self):
        # Enumerate every contiguous deletion in varied blocks, including preamble,
        # Unicode, newlines and number/unit text. This checks representability, not
        # semantic approval; proof endpoints must be reviewed for the new interval.
        examples = (
            '1. Keep STOP. Duplicate context. Check 6.5 bar. Secure guide.',
            'Preamble. Extra context.\n\n1. Non aprire. Nota duplicata. Tenere STOP.',
            '1. 保持停止。重複説明。ガイドを固定。',
            '1. First.\n\n\nMiddle.\nLast.',
        )
        exercised = 0
        for answer in examples:
            layout = r.block_layout(answer)
            aliases = r.review_wire_aliases(layout, ['manual:synthetic'])
            source = 'Keep the documented safe state.'
            catalogue = r.reference_catalog(layout, aliases,
                '[manual:synthetic] source_type=document\n' + source, {'manual:synthetic': source})
            before = deepcopy(catalogue)
            for alias, block in catalogue['blocks'].items():
                units = block['units']
                for first in range(len(units)):
                    for last in range(first, len(units)):
                        start, end = units[first][0], units[last][1]
                        old_body = block['text'][:start] + block['text'][end:]
                        if not old_body.strip():
                            continue  # The existing _edited_block already rejects this.
                        left = next((i for i in range(first - 1, -1, -1) if units[i][2].strip()), None)
                        right = next((i for i in range(last + 1, len(units)) if units[i][2].strip()), None)
                        low, high = (left, last) if left is not None else (first, right)
                        survivor = block['text'][units[low][0]:start] + block['text'][end:units[high][1]]
                        self.assertTrue(survivor.strip())
                        wire = {'citation_ids': ['s1'], 'edits': [{'block_id': alias,
                            'original': owners.span(low, high), 'replacement_parts': [survivor]}],
                            'operation_boundary': {'complete': True, 'checked_blocks': list(catalogue['blocks']),
                                                   'precedence': [], 'change_edges': []}}
                        decoded, binding = r.decode_reference_reply(wire, aliases, layout=layout, catalog=catalogue)
                        self.assertEqual(decoded['edits'][0]['text'].encode(), old_body.encode())
                        expected = ''.join(r._edited_block(b['text'], old_body)
                            if b['block_id'] == block['block_id'] else b['text'] for b in layout['blocks'])
                        self.assertEqual(binding.raw_answer_sha256, r.digest(expected))
                        self.assertEqual(catalogue, before)
                        exercised += 1
        self.assertGreater(exercised, 25)

    def test_reviewed_deletion_preserves_exact_answer_sources_notes_and_order(self):
        helper = refs.ReferenceTests()
        answer = helper.ANSWER.replace('Keep the supply isolated.', 'Duplicate context. Keep the supply isolated.')
        prepared = helper.prepare(answer=answer)
        value = helper.reply()
        value.update(outcome='rewrite', reply_mode='edit', edits=[{'block_id': 'b1',
            'original': owners.span(0, 1), 'replacement_parts': ['Release the guide. ']}])
        value['operation_boundary']['precedence'][0]['action'] = owners.endpoint('b1', kind='replacement')
        original, layout, aliases, catalogue, sources = prepared
        before = deepcopy(sources)
        decoded, binding = r.decode_reference_reply(value, aliases, layout=layout, catalog=catalogue)
        resolved = r.resolve_reply(decoded, answer=original, layout=layout,
            required_facets=['feed'], required_types=['ordered_actions'], source_ordered=True,
            source_notes=({'citation_id': 'step:synthetic', 'source_number': '1',
                           'texts': ('Keep the supply isolated.',)},))
        result = r.verify_operation_boundary(resolved, layout=layout, source_texts=sources,
            source_ordered=True, reference_proof=binding)
        self.assertEqual(result['answer'].encode(), helper.ANSWER.encode())
        self.assertEqual(result['outcome'], 'rewrite')
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertEqual(result['citation_ids'], ['manual:one'])
        self.assertEqual(sources, before)

    def test_nonempty_edits_still_reject_dynamic_out_of_bounds_for_both_endpoints(self):
        for stage in ('action', 'next'):
            for index in (1, 15):
                value = self.wire(['Feed strip. '])
                value['operation_boundary']['precedence'][0][stage] = owners.endpoint('b2', index, kind='replacement')
                with self.subTest(stage=stage, index=index), self.assertRaises(ValueError) as caught:
                    self.fixture.decoded(value)
                diagnostic = r.protocol_failure_diagnostic(caught.exception)
                self.assertEqual((diagnostic['stage'], diagnostic['reason'], diagnostic['unit_count']),
                                 (stage, 'out_of_bounds', 1))

    def test_no_new_source_or_deleted_original_authority(self):
        for mutation in ('source', 'deleted_original'):
            value = self.wire(['Feed strip. '])
            edge = value['operation_boundary']['precedence'][0]
            if mutation == 'source':
                edge['proof']['source'] = 'unadmitted-source'
                expected = 'procedure_review_wire_alias_unknown'
            else:
                edge['next'] = owners.endpoint('b2', 1)
                expected = 'procedure_reference_deleted_unit'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, expected):
                self.fixture.decoded(value)

    def test_retain_and_semantic_reject_are_unchanged(self):
        helper = refs.ReferenceTests()
        retained = helper.execute(helper.reply())
        self.assertEqual(retained['answer'], helper.ANSWER)
        self.assertEqual(retained['outcome'], 'pass')
        rejected = helper.reply()
        rejected.update(outcome='no_sources', reply_mode='reject', reason='unsafe_or_unsupported')
        result = helper.execute(rejected)
        self.assertEqual((result['outcome'], result['reason']), ('no_sources', 'unsafe_or_unsupported'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
