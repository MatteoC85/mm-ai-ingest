"""Reviewed pure deletion of a server-owned appendix; no provider or customer data."""
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import procedure_review as review
import test_generated_manual_note_transport as transport


class ManualAttachmentRemovalTests(unittest.TestCase):
    def setUp(self):
        self.core = ('1. Isolate supply.\nKeep the guard closed.\n\n'
                     '2. Insert cartridge.\n\n3. Secure housing.')
        self.notes = [{'text': 'Insert cartridge through the guide.', 'citation_ids': ['manual:a']},
                      {'text': 'Restore the guide afterwards.', 'citation_ids': ['manual:b']}]
        self.admitted = {'steps', 'manual:a', 'manual:b'}

    def original(self, heading='Manual support'):
        return self.core + '\n\n' + heading + '\n' + '\n'.join('- ' + n['text'] for n in self.notes)

    def verdict(self, original, answer, *, ids=('steps',), reject=False):
        layout = review.block_layout(original)
        final = review.block_layout(answer)
        value = {'outcome': 'no_sources' if reject else 'rewrite',
                 'reply_mode': 'reject' if reject else 'edit', 'answer': '',
                 'edits': [] if reject else [{'block_id': b['block_id'], 'text': b['text']}
                     for b, old in zip(final['blocks'], layout['blocks']) if b['text'] != old['text']],
                 'citation_ids': list(ids), 'covered_facets': ['sequence'],
                 'missing_facets': ['sequence'] if reject else [],
                 'covered_answer_types': ['ordered_actions'], 'missing_answer_types': [],
                 'missing_list_items': [], 'operation_boundary': {'complete': not reject,
                     'checked_blocks': [b['block_id'] for b in layout['blocks']], 'changes': [],
                     'precedence': [{'before_block_id': 'step_2', 'after_block_id': 'step_3',
                         'before_answer_quote': 'Insert cartridge.', 'after_answer_quote': 'Secure housing.',
                         'source_citation_id': 'steps', 'source_quote': 'Insert cartridge. Secure housing.'}]}}
        resolved = review.resolve_reply(value, answer=original, layout=layout,
            required_facets=['sequence'], required_types=['ordered_actions'], source_ordered=True,
            source_notes=({'source_number': 1, 'texts': ['Keep the guard closed.']},))
        return review.verify_operation_boundary(resolved, layout=layout,
            source_texts={'steps': 'Insert cartridge. Secure housing.'}, source_ordered=True)

    def project(self, *, original=None, answer=None, ids=('steps',), admitted=None, final=None, tamper=None):
        original = self.original() if original is None else original
        answer = self.core if answer is None else answer
        verdict = self.verdict(original, answer, ids=ids)
        if tamper:
            tamper(verdict)
        kwargs = {} if final is None else {'final_citation_ids': final}
        return review.manual_note_projection(self.notes, original_answer=original,
            reviewed_answer=verdict['answer'], review=verdict,
            admitted_ids=self.admitted if admitted is None else admitted, **kwargs)

    def test_reviewed_whole_appendix_removal_does_not_restore_its_citations(self):
        for heading in ('Manual support', 'Supporto dal manuale'):
            with self.subTest(heading=heading):
                ids, proof = self.project(original=self.original(heading))
                self.assertEqual(ids, [])
                self.assertTrue(proof['complete'])
                self.assertEqual(proof['citation_ids'], [])
                self.assertEqual(proof['removed_notes'], 2)

    def test_final_citations_can_omit_only_deleted_appendix_sources(self):
        _, proof = self.project(final={'steps'})
        self.assertEqual(proof['basis'], 'reviewed_exact_manual_attachment_removal')

    def test_whitespace_projection_does_not_change_removed_attachment_content(self):
        ids, _ = self.project(original=self.original().replace('\n- ', '\n\t-   '))
        self.assertEqual(ids, [])

    def test_missing_or_forged_header_cannot_authorize_removal(self):
        for heading in ('', 'Operator advice', 'Manual support with new instructions'):
            with self.subTest(heading=heading), self.assertRaises(ValueError):
                self.project(original=self.original(heading))

    def test_partial_removal_or_new_unbound_replacement_stays_rejected(self):
        for answer in (self.core + '\nManual support\n- ' + self.notes[0]['text'],
                       self.core + '\nUse a different guide.', self.core + '\nDo not secure housing.'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                self.project(answer=answer)

    def test_removed_core_warning_or_extra_unbound_appendix_text_is_rejected(self):
        for original, answer in ((self.original(), self.core.replace('Keep the guard closed.', '')),
                                 (self.original() + '\nNever bypass the interlock.', self.core)):
            with self.subTest(original=original), self.assertRaises(ValueError):
                self.project(original=original, answer=answer)

    def test_incomplete_review_stale_hash_and_missing_original_admission_reject(self):
        changes = [lambda v: v['procedure_review'].update(input_answer_sha256='0'*64),
                   lambda v: v['procedure_review'].update(output_answer_sha256='0'*64),
                   lambda v: v['procedure_review'].update(independent_semantic_review=False),
                   lambda v: v['operation_boundary_validation'].update(complete=False),
                   lambda v: v['procedure_review'].update(source_sequence_immutable=False),
                   lambda v: v['procedure_review'].update(source_note_bindings_complete=False),
                   lambda v: v.update(outcome='no_sources'),
                   lambda v: v.update(edits=[])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.project(tamper=change)
        with self.assertRaises(ValueError):
            self.project(admitted={'steps', 'manual:a'})

    def test_surviving_rewrite_requires_original_and_final_references(self):
        translated = self.original().replace('Restore the guide afterwards.', 'Restore the guide at the end.')
        with self.assertRaises(ValueError):
            self.project(answer=translated)
        with self.assertRaises(ValueError):
            self.project(answer=translated, ids=tuple(self.admitted), final={'steps', 'manual:a'})

    def test_unchanged_note_still_requires_restored_reference_in_final_citations(self):
        original = self.original()
        verdict = {'citation_ids': ['steps'], 'procedure_review': {'reply_mode': 'retain',
            'independent_semantic_review': True, 'input_answer_sha256': review.digest(original),
            'output_answer_sha256': review.digest(original)}}
        ids, _ = review.manual_note_projection(self.notes, original_answer=original,
            reviewed_answer=original, review=verdict, admitted_ids=self.admitted)
        self.assertEqual(set(ids), {'manual:a', 'manual:b'})
        with self.assertRaises(ValueError):
            review.manual_note_projection(self.notes, original_answer=original,
                reviewed_answer=original, review=verdict, admitted_ids=self.admitted,
                final_citation_ids={'steps', 'manual:a'})


class ActualGeneratedAttachmentRemovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transport.GeneratedManualNoteTransportTests.setUpClass()
        cls.fixture = transport.GeneratedManualNoteTransportTests('run')

    @classmethod
    def tearDownClass(cls):
        transport.GeneratedManualNoteTransportTests.tearDownClass()

    def validated_deletion(self, *, foreign=False):
        h, generated, note, fixture = self.fixture.generated(aggregate=True, foreign=foreign)
        calls = []
        def verifier(**kw):
            calls.append(kw)
            answer = kw['answer']; layout = review.block_layout(answer)
            last = layout['blocks'][-1]
            retained = last['text'].split('Manual support')[0].rstrip()
            ids = [c['citation_id'] for c in kw['candidates'] if not c['citation_id'].startswith('manual:')]
            source = next(c for c in kw['candidates'] if c['citation_id'] == ids[-1])
            value = {'outcome': 'rewrite', 'reply_mode': 'edit', 'answer': '',
                'edits': [{'block_id': last['block_id'], 'text': retained}], 'citation_ids': ids,
                'covered_facets': ['cartridge installation'], 'missing_facets': [],
                'covered_answer_types': ['ordered_actions'], 'missing_answer_types': [], 'missing_list_items': [],
                'operation_boundary': {'complete': True, 'checked_blocks': [b['block_id'] for b in layout['blocks']],
                    'changes': [], 'precedence': [{'before_block_id': last['block_id'],
                        'after_block_id': last['block_id'],
                        'before_answer_quote': 'Secure the housing. Secure the housing.',
                        'after_answer_quote': 'Keep the supply isolated.',
                        'source_citation_id': source['citation_id'],
                        'source_quote': self.fixture.m._assistant_core_candidate_evidence_text(source)}]}}
            out = review.resolve_reply(value, answer=answer, layout=layout,
                required_facets=['cartridge installation'], required_types=['ordered_actions'],
                source_ordered=True, source_notes=kw['repair_context']['procedure_notes'])
            out = review.verify_operation_boundary(out, layout=layout,
                source_texts={c['citation_id']: self.fixture.m._assistant_core_candidate_evidence_text(c)
                              for c in kw['candidates']}, source_ordered=True)
            self.assertTrue(out['operation_boundary_validation']['complete'], out)
            return out
        result = fixture.validate_bundle(h, generated, verifier)
        self.assertEqual(len(calls), 1)
        self.fixture.assert_public(result)
        return result

    def test_real_generated_family_redactor_and_final_validator_accept_only_reviewed_deletion(self):
        result = self.validated_deletion()
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'], 'answered', contract)
        self.assertTrue(contract['passed'], contract)
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(contract['manual_note_bindings']['removed_notes'], 1)
        self.assertNotIn('Manual support', result['answer'])
        self.assertIn('Keep the supply isolated.', result['answer'])
        self.assertFalse(any(c['citation_id'].startswith('manual:') for c in result['citations']))

    def test_prior_foreign_binding_failure_is_not_cleared_by_approved_deletion(self):
        result = self.validated_deletion(foreign=True)
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'], 'no_sources')
        self.assertFalse(contract['passed'])
        self.assertTrue(contract['manual_note_binding_invalid'])
        self.assertFalse(result.get('_assistant_core_repair_needed'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
