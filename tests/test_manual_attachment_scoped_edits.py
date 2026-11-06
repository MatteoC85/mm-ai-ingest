"""Appendix ownership must not veto separately reviewed source-block corrections."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import procedure_review as review
import test_generated_manual_note_transport as transport


class ScopedManualAttachmentTests(unittest.TestCase):
    def fixture(self):
        core = ('1. Prepare\nStop cycle\nApply isolation.\nKeep the supply isolated.\n\n'
                '2. Insert cartridge.\n\n3. Restore\nSecure housing.\nKeep the guard closed.')
        notes = [{'text': 'Insert through the guide and restore it.', 'citation_ids': ['manual:a']}]
        original = core + '\n\nManual support\n- ' + notes[0]['text']
        corrected = core.replace('Stop cycle\n', 'Stop cycle.')
        return original, corrected, notes

    def verdict(self, original, answer, *, preserve_notes=True):
        layout = review.block_layout(original)
        final = review.block_layout(answer)
        edits = [{'block_id': new['block_id'], 'text': new['text']}
                 for old, new in zip(layout['blocks'], final['blocks']) if old['text'] != new['text']]
        value = {'outcome': 'rewrite', 'reply_mode': 'edit', 'answer': '', 'edits': edits,
                 'citation_ids': ['steps'], 'covered_facets': ['sequence'], 'missing_facets': [],
                 'covered_answer_types': ['ordered_actions'], 'missing_answer_types': [], 'missing_list_items': [],
                 'operation_boundary': {'complete': True,
                     'checked_blocks': [b['block_id'] for b in layout['blocks']], 'changes': [],
                     'precedence': [
                         {'before_block_id': 'step_1', 'after_block_id': 'step_1',
                          'before_answer_quote': 'Stop cycle.', 'after_answer_quote': 'Apply isolation.',
                          'source_citation_id': 'steps', 'source_quote': 'Stop cycle. Apply isolation.'},
                         {'before_block_id': 'step_3', 'after_block_id': 'step_3',
                          'before_answer_quote': 'Secure housing.', 'after_answer_quote': 'Keep the guard closed.',
                          'source_citation_id': 'steps', 'source_quote': 'Secure housing. Keep the guard closed.'}]}}
        notes = ({'source_number': 1, 'texts': ['Keep the supply isolated.']},
                 {'source_number': 3, 'texts': ['Keep the guard closed.']}) if preserve_notes else ()
        resolved = review.resolve_reply(value, answer=original, layout=layout,
            required_facets=['sequence'], required_types=['ordered_actions'], source_ordered=True,
            source_notes=notes)
        return review.verify_operation_boundary(resolved, layout=layout,
            source_texts={'steps': 'Stop cycle. Apply isolation. Secure housing. Keep the guard closed.'},
            source_ordered=True)

    def project(self, original, verdict, notes):
        return review.manual_note_projection(notes, original_answer=original,
            reviewed_answer=verdict['answer'], review=verdict,
            admitted_ids={'steps', 'manual:a'}, final_citation_ids={'steps'})

    def test_reviewed_other_block_punctuation_and_exact_appendix_deletion_are_compatible(self):
        original, answer, notes = self.fixture()
        verdict = self.verdict(original, answer)
        self.assertTrue(verdict['operation_boundary_validation']['complete'], verdict)
        ids, proof = self.project(original, verdict, notes)
        self.assertEqual(ids, [])
        self.assertEqual(proof['removed_notes'], 1)
        self.assertEqual(verdict['answer'], answer)

    def test_unreported_or_mismatched_other_block_edit_rejects(self):
        original, answer, notes = self.fixture()
        verdict = self.verdict(original, answer)
        for variant in ('absent', 'mismatch', 'duplicate', 'foreign'):
            value = deepcopy(verdict)
            if variant == 'absent': value['edits'] = value['edits'][1:]
            elif variant == 'mismatch': value['edits'][0]['text'] += ' A new instruction.'
            elif variant == 'duplicate': value['edits'].append(deepcopy(value['edits'][0]))
            else: value['edits'][0]['block_id'] = 'foreign'
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.project(original, value, notes)

    def test_source_warning_removal_cannot_use_an_approved_appendix_deletion(self):
        original, answer, notes = self.fixture()
        value = self.verdict(original, answer.replace('Keep the supply isolated.', ''))
        self.assertEqual(value['outcome'], 'no_sources')
        self.assertFalse(value['procedure_review']['source_note_bindings_complete'])
        with self.assertRaises(ValueError): self.project(original, value, notes)

    def test_attachment_owner_operational_prefix_cannot_change(self):
        original, answer, notes = self.fixture()
        value = self.verdict(original, answer)
        changed = value['answer'].replace('Secure housing.', 'Secure housing. Apply excess pressure.')
        value['answer'] = changed
        value['procedure_review']['output_answer_sha256'] = review.digest(changed)
        value['operation_boundary_validation']['answer_sha256'] = review.digest(changed)
        value['edits'][-1]['text'] = value['edits'][-1]['text'].replace('Secure housing.', 'Secure housing. Apply excess pressure.')
        with self.assertRaises(ValueError): self.project(original, value, notes)

    def test_stale_hash_or_boundary_still_rejects(self):
        original, answer, notes = self.fixture()
        for field in ('hash', 'boundary', 'sequence'):
            value = self.verdict(original, answer)
            if field == 'hash': value['procedure_review']['output_answer_sha256'] = '0' * 64
            elif field == 'boundary': value['operation_boundary_validation']['complete'] = False
            else: value['procedure_review']['source_sequence_immutable'] = False
            with self.subTest(field=field), self.assertRaises(ValueError): self.project(original, value, notes)


class GeneratedScopedAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transport.GeneratedManualNoteTransportTests.setUpClass()
        cls.fixture = transport.GeneratedManualNoteTransportTests('run')

    @classmethod
    def tearDownClass(cls): transport.GeneratedManualNoteTransportTests.tearDownClass()

    def validated(self, *, remove_warning=False, foreign=False):
        h, generated, note, fixture = self.fixture.generated(aggregate=True, foreign=foreign)
        calls = []
        def verifier(**kw):
            calls.append(1)
            answer = kw['answer']; layout = review.block_layout(answer)
            first = next(b for b in layout['blocks'] if b['block_id'] == 'step_1')
            last = layout['blocks'][-1]
            changed_first = first['text'].replace('Isolate the supply.', 'Isolate the supply', 1)
            if remove_warning: changed_first = changed_first.replace('Keep the supply isolated.', '')
            retained = last['text'].split('Manual support')[0].rstrip()
            sources = [c for c in kw['candidates'] if not c['citation_id'].startswith('manual:')]
            step_sources = [c for c in sources if c.get('source_type') == 'step']
            values = []
            for block, text, source in ((first, changed_first, step_sources[0]), (last, retained, step_sources[-1])):
                body = review._block_parts(text)[1]
                action = body.split('Keep the supply isolated.')[0].rstrip()
                values.append({'before_block_id': block['block_id'], 'after_block_id': block['block_id'],
                    'before_answer_quote': action, 'after_answer_quote': 'Keep the supply isolated.',
                    'source_citation_id': source['citation_id'],
                    'source_quote': self.fixture.m._assistant_core_candidate_evidence_text(source)})
            value = {'outcome': 'rewrite', 'reply_mode': 'edit', 'answer': '',
                'edits': [{'block_id': first['block_id'], 'text': changed_first}, {'block_id': last['block_id'], 'text': retained}],
                'citation_ids': [c['citation_id'] for c in sources],
                'covered_facets': ['cartridge installation'], 'missing_facets': [],
                'covered_answer_types': ['ordered_actions'], 'missing_answer_types': [], 'missing_list_items': [],
                'operation_boundary': {'complete': True, 'checked_blocks': [b['block_id'] for b in layout['blocks']],
                    'changes': [], 'precedence': values}}
            out = review.resolve_reply(value, answer=answer, layout=layout,
                required_facets=['cartridge installation'], required_types=['ordered_actions'], source_ordered=True,
                source_notes=kw['repair_context']['procedure_notes'])
            return review.verify_operation_boundary(out, layout=layout,
                source_texts={c['citation_id']: self.fixture.m._assistant_core_candidate_evidence_text(c) for c in kw['candidates']},
                source_ordered=True)
        result = fixture.validate_bundle(h, generated, verifier)
        self.assertEqual(len(calls), 1)
        self.fixture.assert_public(result)
        return result

    def test_real_generation_review_and_final_validation_accept_two_independent_edits(self):
        result = self.validated()
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'], 'answered', contract)
        self.assertTrue(contract['passed'], contract)
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(contract['manual_note_bindings']['removed_notes'], 1)
        self.assertNotIn('Manual support', result['answer'])
        self.assertIn('Keep the supply isolated.', result['answer'])
        self.assertFalse(any(c['citation_id'].startswith('manual:') for c in result['citations']))

    def test_real_warning_or_foreign_carrier_still_fails_closed(self):
        for args in ({'remove_warning': True}, {'foreign': True}):
            with self.subTest(args=args):
                result = self.validated(**args)
                contract = result['meta']['assistant_core_validation']['answer_contract']
                self.assertFalse(contract['passed'])
                if args.get('foreign'):
                    self.assertEqual(result['status'], 'no_sources')
                    self.assertTrue(contract['manual_note_binding_invalid'])
                else:
                    # Validation retains the safe original draft for its existing
                    # bounded repair path; the unsafe reviewed edit is rejected.
                    self.assertEqual(contract['semantic_verifier']['outcome'], 'no_sources')
                    self.assertFalse(contract['semantic_verifier']['procedure_review']['source_note_bindings_complete'])
                    self.assertTrue(result['_assistant_core_repair_needed'])
                    self.assertIn('Keep the supply isolated.', result['answer'])


if __name__ == '__main__': unittest.main(verbosity=2)
