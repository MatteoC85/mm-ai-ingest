"""Offline proof-delta safety boundaries; no claim of model semantic quality.

The independent reviewer still owns semantic completeness across ALL sources.
These fixtures check that its rejection and affected dependencies in untouched
blocks survive compact wire handling and deterministic final validation.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review


class ProcedureDeltaSafetyTests(unittest.TestCase):
    DRAFT = ('1. Stop and isolate the drive.\n\n'
             '4. Adjust the guide.\nKeep hands clear.\n\n'
             '9. Close and latch the cover.\nJog the drive.')
    SOURCE = ('Stop and isolate the drive before unlatching the guide cover. '
              'After unlatching the guide cover, close and latch the cover. '
              'Close and latch the cover before jogging the drive. Keep hands clear.')
    BODY = 'Unlatch the guide cover.\nAdjust the guide.\nKeep hands clear.'

    def reply(self):
        def edge(left, action, right, next_action, source):
            return {'before_block_id': left, 'before_answer_quote': action,
                'after_block_id': right, 'after_answer_quote': next_action,
                'source_citation_id': 'manual:one', 'source_quote': source}
        return {'outcome': 'rewrite', 'reply_mode': 'edit', 'answer': '',
            'edits': [{'block_id': 'step_2', 'text': self.BODY}],
            'covered_facets': ['adjustment'], 'missing_facets': [],
            'covered_answer_types': ['ordered_actions'], 'missing_answer_types': [],
            'missing_list_items': [], 'citation_ids': [],
            'operation_boundary': {'complete': True, 'checked_blocks': ['step_1', 'step_2', 'step_3'],
                'change_edges': [{'edge_index': 1, 'resolution': 'restored'}],
                'precedence': [
                    edge('step_1', 'Stop and isolate the drive.', 'step_2', 'Unlatch the guide cover.',
                        'Stop and isolate the drive before unlatching the guide cover.'),
                    edge('step_2', 'Unlatch the guide cover.', 'step_3', 'Close and latch the cover.',
                        'After unlatching the guide cover, close and latch the cover.'),
                    edge('step_3', 'Close and latch the cover.', 'step_3', 'Jog the drive.',
                        'Close and latch the cover before jogging the drive.')]}}

    def micro_reply(self, draft=None):
        draft = self.DRAFT if draft is None else draft
        layout = review.block_layout(draft)
        aliases = review.review_wire_aliases(layout, ['manual:one'])
        source_names = {cid: alias for alias, cid in aliases['sources'].items()}
        block_names = {bid: alias for alias, bid in aliases['blocks'].items()}
        wire = self.reply()
        wire['edits'] = [{'block_id': block_names['step_2'], 'old_text': 'Adjust the guide.',
                          'new_text': 'Unlatch the guide cover.\nAdjust the guide.'}]
        audit = wire['operation_boundary']
        audit['checked_blocks'] = [block_names[bid] for bid in audit['checked_blocks']]
        audit['precedence'] = [{'from': block_names[e['before_block_id']],
            'to': block_names[e['after_block_id']], 'action': e['before_answer_quote'],
            'next': e['after_answer_quote'], 'source': source_names[e['source_citation_id']],
            'proof': e['source_quote']} for e in audit['precedence']]
        return wire, aliases, layout

    def resolve(self, value, draft=None):
        draft = self.DRAFT if draft is None else draft
        return review.resolve_reply(value, answer=draft, layout=review.block_layout(draft),
            required_facets=['adjustment'], required_types=['ordered_actions'], source_ordered=True,
            source_notes=({'citation_id': 'step:4', 'source_number': '4', 'texts': ('Keep hands clear.',)},))

    def verify(self, value, draft=None, sources=None):
        draft = self.DRAFT if draft is None else draft
        resolved = self.resolve(value, draft)
        return review.verify_operation_boundary(resolved, layout=review.block_layout(draft),
            source_texts={'manual:one': self.SOURCE} if sources is None else sources, source_ordered=True)

    def test_micro_edit_expands_to_identical_full_body_proof_and_digest(self):
        wire, aliases, layout = self.micro_reply()
        original = deepcopy(wire)
        decoded = review.decode_review_wire_aliases(wire, aliases, layout=layout)
        self.assertEqual(wire, original)
        self.assertEqual(decoded, self.reply())
        self.assertEqual(self.verify(decoded), self.verify(self.reply()))

    def test_micro_edit_cannot_hide_unsafe_untouched_dependency_or_delete_note(self):
        unsafe = self.DRAFT.replace('Close and latch the cover.\nJog the drive.',
                                   'Jog the drive.\nClose and latch the cover.')
        wire, aliases, layout = self.micro_reply(unsafe)
        decoded = review.decode_review_wire_aliases(wire, aliases, layout=layout)
        self.assertEqual(self.verify(decoded, unsafe)['outcome'], 'no_sources')
        wire, aliases, layout = self.micro_reply()
        wire['edits'][0].update(old_text='Keep hands clear.', new_text='')
        decoded = review.decode_review_wire_aliases(wire, aliases, layout=layout)
        result = self.verify(decoded)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertFalse(result['procedure_review']['source_note_bindings_complete'])

    def test_affected_dependencies_in_unedited_blocks_remain_verified(self):
        result = self.verify(self.reply())
        self.assertEqual(result['outcome'], 'rewrite')
        audit = result['operation_boundary_validation']
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['checked_blocks'], 3)
        self.assertEqual(audit['precedence_edges'], 3)
        self.assertEqual(audit['answer_sha256'], review.digest(result['answer']))
        self.assertEqual(result['procedure_review']['edited_blocks'], 1)
        self.assertIn('manual:one', result['citation_ids'])
        self.assertTrue(result['answer'].endswith('9. Close and latch the cover.\nJog the drive.'))

    def test_unsafe_dependency_in_untouched_block_is_not_exempt_from_order_validation(self):
        unsafe = self.DRAFT.replace('Close and latch the cover.\nJog the drive.',
                                   'Jog the drive.\nClose and latch the cover.')
        result = self.verify(self.reply(), unsafe)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertFalse(result['operation_boundary_validation']['complete'])
        self.assertFalse(result['operation_boundary_validation']['precedence_checked'])

    def test_declared_rejection_of_unchanged_canonical_draft_is_never_promoted(self):
        # A canonical sequence is not semantic/safety proof. Here the model has
        # explicitly rejected an unsafe ORIGINAL block; no edit is necessary to
        # carry that rejection through the deterministic publication checks.
        value = self.reply()
        value.update(outcome='no_sources', reply_mode='reject', edits=[],
                     reason='unchanged_source_sequence_unsafe', missing_facets=['adjustment'])
        value['operation_boundary'].update(complete=False, change_edges=[], precedence=[])
        result = self.verify(value)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertEqual(result['answer'], '')
        self.assertFalse(result['operation_boundary_validation']['complete'])

    def test_unchanged_draft_still_requires_full_declared_semantic_coverage(self):
        value = self.reply()
        value.update(outcome='pass', reply_mode='retain', edits=[], covered_facets=[])
        value['operation_boundary'].update(change_edges=[], precedence=[])
        with self.assertRaisesRegex(ValueError, 'coverage_not_complete'):
            self.verify(value)

    def test_unedited_blocks_cannot_be_omitted_from_review_inventory(self):
        value = self.reply(); value['operation_boundary']['checked_blocks'] = ['step_2']
        result = self.verify(value)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertFalse(result['operation_boundary_validation']['complete'])

    def test_changed_block_without_dependency_proof_remains_rejected(self):
        value = self.reply(); value['operation_boundary'].update(precedence=[], change_edges=[])
        result = self.verify(value)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertFalse(result['operation_boundary_validation']['complete'])

    def test_original_note_is_immutable_even_when_added_operation_is_reviewed(self):
        value = self.reply(); value['edits'][0]['text'] = self.BODY.replace('Keep hands clear.', '')
        result = self.verify(value)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertEqual(result['answer'], '')
        self.assertFalse(result['procedure_review']['source_note_bindings_complete'])

    def test_affected_dependency_cannot_introduce_unadmitted_source(self):
        value = self.reply(); value['operation_boundary']['precedence'][2]['source_citation_id'] = 'foreign'
        result = self.verify(value)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertFalse(result['operation_boundary_validation']['complete'])


if __name__ == '__main__': unittest.main()
