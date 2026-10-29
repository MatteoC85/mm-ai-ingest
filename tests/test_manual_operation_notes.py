"""Complete manual support, page bindings and reviewed citation projection offline."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.ask import procedure_review as review
from machinemind.ask import validation
from machinemind.presentation import responses
from test_compound_procedure_selection import step
import test_source_ordered_review as wired_fixtures


def deny(*a, **kw): raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class ManualOperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guard.enter_context(patch(name, deny))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls): cls.guard.close()

    def setUp(self):
        self.rows = [step(3, 'Insert ribbon', 'Insert ribbon through guides to the clamp.', 'Keep the supply isolated.'),
                     step(4, 'Restore guide', 'Close the guide lever.', 'Do not apply excessive pressure.')]
        self.note = ('To insert ribbon, release the guide lever and open the inlet guide only with the supply isolated. '
            'Pass the ribbon through the horizontal guides and vertical guide wheels to the carriage clamp, keeping '
            'its two edges inside the illustrated channel and observing the marked orientation without reaching '
            'into moving parts or changing the independent pressure screws. '
            'Then close the guide lever and restore the inlet guide before leaving the preparation area.')
        self.manual = [{'citation_id':'manual-route', 'source_type':'document', 'ask_structured_manual_support':True,
                        'page_from':12, 'chunk_full':self.note},
                       {'citation_id':'manual-other', 'source_type':'document', 'ask_structured_manual_support':True,
                        'page_from':18, 'chunk_full':'For adjustment of a separate drive, the qualified operator must open the protective guard under safety maintenance conditions.'}]

    def model(self, **kwargs):
        return responses.build_structured_procedure_ui_model(structured_citations=self.rows,
            manual_support_citations=self.manual, grounded_points=[], response_language='en',
            q='How do I insert ribbon?', _runtime=self.m._RESPONSE_PRESENTATION_RUNTIME(), preserve_complete=True, **kwargs)

    def test_complete_operating_note_survives_steps_text_and_html_with_closure(self):
        before = deepcopy((self.rows, self.manual))
        model = self.model()
        self.assertGreater(len(self.note), 360)
        self.assertEqual(model['manual_notes'], [self.note])
        self.assertEqual(model['manual_note_sources'], [{'text':self.note,'citation_ids':['manual-route']}])
        text = self.m._procedure_ui_model_to_text(model, response_language='en', preserve_complete=True)
        body = self.m._procedure_ui_model_to_html(model, links=[], response_language='en')
        for phrase in ('horizontal guides', 'vertical guide wheels', 'carriage clamp', 'Then close the guide lever'):
            self.assertIn(phrase, text)
            self.assertIn(phrase, body)
        self.assertNotIn('separate drive', text)
        self.assertNotIn('…', text)
        self.assertIn('Keep the supply isolated.', text)
        self.assertIn('Do not apply excessive pressure.', text)
        self.assertEqual((self.rows, self.manual), before)

    def test_adjacent_page_never_becomes_continuation_or_safety_note(self):
        first = 'To insert ribbon, release the guide lever. Then pass ribbon through the guide to the clamp.'
        self.manual[0]['chunk_full'] = first
        self.manual[1]['chunk_full'] = 'Then adjust ribbon clamp pressure by turning the separate calibration screw. Qualified operator safety protection applies.'
        with patch.object(self.m, '_ask_structured_manual_support_terms', return_value=['insert', 'ribbon', 'release', 'guide', 'lever']):
            note, safety = self.m._manual_operation_and_safety_notes_from_support_citations(
                self.manual, q='insert ribbon', structured_citations=self.rows, language='en')
        self.assertEqual(note, first)
        self.assertNotIn('calibration', note + safety)

    def test_grounded_fallback_requires_only_retained_manual_ids(self):
        empty = {'operation_note':'','safety_note':'','operation_citation_ids':[],'safety_citation_ids':[]}
        runtime = replace(self.m._RESPONSE_PRESENTATION_RUNTIME(),
            manual_operation_and_safety_notes_from_support_citations=lambda *a, **k: empty)
        points = [{'text':'Safety: forbidden external source note.','citation_ids':['not-retained']},
                  {'text':'Safety: mixed source note.','citation_ids':['manual-route','not-retained']},
                  {'text':'Safety: use the documented protective gloves.','citation_ids':['manual-route']}]
        model = responses.build_structured_procedure_ui_model(structured_citations=self.rows,
            manual_support_citations=self.manual, grounded_points=points, response_language='en',
            q='insert ribbon', _runtime=runtime, preserve_complete=True)
        self.assertNotIn('external', ' '.join(model['manual_notes']))
        self.assertNotIn('mixed', ' '.join(model['manual_notes']))
        self.assertIn('protective gloves', ' '.join(model['manual_notes']))
        self.assertEqual(model['manual_note_sources'][0]['citation_ids'], ['manual-route'])

    def test_semantic_aggregate_note_keeps_all_selected_source_ids(self):
        for record in self.manual:
            record['llm_operation_note'] = self.note
            record['llm_operation_citation_ids'] = ['manual-route','manual-other']
        result = self.m._manual_operation_and_safety_notes_from_support_citations(
            self.manual, q='insert ribbon', structured_citations=self.rows, language='en', with_bindings=True)
        self.assertEqual(result['operation_citation_ids'], ['manual-route','manual-other'])

    def test_filtered_aggregate_source_cannot_be_rebound_to_remaining_page(self):
        aggregate = 'First follow the route; then apply a separate adjustment from the second page.'
        self.manual[0]['llm_operation_note'] = aggregate
        self.manual[0]['llm_operation_citation_ids'] = ['manual-route','manual-other']
        result = self.m._manual_operation_and_safety_notes_from_support_citations(
            self.manual[:1], q='insert ribbon', structured_citations=self.rows, language='en', with_bindings=True)
        self.assertNotEqual(result['operation_note'], aggregate)
        self.assertNotIn('separate adjustment', result['operation_note'])
        self.assertEqual(result['operation_citation_ids'], ['manual-route'])

    def test_oversize_or_incomplete_note_is_not_cut_into_an_instruction(self):
        self.manual[0]['chunk_full'] = 'Insert ribbon ' + ('through the guide ' * 200) + 'only while isolated.'
        self.manual = self.manual[:1]
        note, _ = self.m._manual_operation_and_safety_notes_from_support_citations(
            self.manual, q='insert ribbon', structured_citations=self.rows, language='en')
        self.assertEqual(note, '')

    def test_semantically_selected_safety_only_page_keeps_its_binding(self):
        self.manual[1]['llm_safety_note'] = 'Use the specified protective equipment.'
        self.manual[1]['llm_safety_citation_ids'] = ['manual-other']
        with patch.object(self.m, '_ask_structured_manual_support_terms', return_value=[]):
            result = self.m._manual_operation_and_safety_notes_from_support_citations(
                self.manual, q='required protective equipment', structured_citations=[], language='en', with_bindings=True)
        self.assertEqual(result['operation_note'], '')
        self.assertEqual(result['safety_note'], self.manual[1]['llm_safety_note'])
        self.assertEqual(result['safety_citation_ids'], ['manual-other'])

    def wired_review(self, *, foreign=False, translate=False, unsupported=False):
        real_validate = validation.validate_response
        note = 'Passare il nastro attraverso la guida. Ripristinare la guida al termine.' if translate else self.note
        manual = {'citation_id':'manual-bound', 'source_type':'document', 'chunk_full':note,
                  'ask_structured_manual_support':True}
        semantic_calls = []
        def inject(response, request, retrieval, decision, *, runtime):
            response = deepcopy(response)
            response['answer'] += '\nManual support\n- ' + note
            response['_assistant_ui_model'] = {'manual_note_sources':[{'text':note,
                'citation_ids':['manual-foreign' if foreign else 'manual-bound']}]}
            response['_assistant_core_validation_evidence'] = [manual]
            def verifier(**kwargs):
                semantic_calls.append(kwargs)
                answer = kwargs['answer']; layout = review.block_layout(answer)
                value = {'outcome':'pass', 'reply_mode':'retain', 'answer':'', 'edits':[],
                    'citation_ids':[c['citation_id'] for c in retrieval['citations']],
                    'covered_facets':['sequence','safety'], 'missing_facets':[],
                    'covered_answer_types':['ordered_actions','safety_conditions'],
                    'missing_answer_types':[], 'missing_list_items':[],
                    'operation_boundary':{'complete':True, 'checked_blocks':[b['block_id'] for b in layout['blocks']],
                                          'precedence':[], 'changes':[]}}
                if translate:
                    projected, removed = runtime._assistant_core_filter_unsupported_claim_sentences(note, note)
                    self.assertEqual(removed, [])
                    last = layout['blocks'][-1]
                    translated = 'Pass the strip through the guide. Restore the guide afterwards.'
                    if unsupported:
                        translated += ' Set temperature to 99999 °C.'
                    value.update(outcome='rewrite', reply_mode='edit',
                        citation_ids=value['citation_ids']+['manual-bound'],
                        edits=[{'block_id':last['block_id'], 'text':last['text'].replace(projected, translated)}])
                    value['operation_boundary']['precedence'] = [{
                        'before_block_id':last['block_id'],'after_block_id':last['block_id'],
                        'before_answer_quote':'Pass the strip through the guide.',
                        'after_answer_quote':'Restore the guide afterwards.',
                        'source_citation_id':'manual-bound','source_quote':note}]
                result = review.resolve_reply(value, answer=answer, layout=layout,
                    required_facets=['sequence','safety'], required_types=['ordered_actions','safety_conditions'],
                    source_ordered=True, source_notes=kwargs['repair_context']['procedure_notes'])
                return review.verify_operation_boundary(result, layout=layout,
                    source_texts={c['citation_id']:c['chunk_full'] for c in retrieval['citations']+[manual]},
                    source_ordered=True)
            runtime = replace(runtime, _assistant_core_verify_or_repair_answer=verifier)
            return real_validate(response, request, retrieval, decision, runtime=runtime)
        fixture = wired_fixtures.WiredSourceOrderedReviewTests('run')
        fixture.m = self.m
        with patch.object(validation, 'validate_response', side_effect=inject):
            result, _, _ = fixture.run_review()
        self.assertEqual(len(semantic_calls), 1)
        return result

    def test_final_validator_restores_admitted_manual_link_for_unchanged_note(self):
        result = self.wired_review()
        self.assertEqual(result['status'], 'answered')
        self.assertIn('manual-bound', {c['citation_id'] for c in result['citations']})
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['manual_note_bindings']['complete'])

    def test_final_validator_rejects_foreign_binding_without_repair(self):
        result = self.wired_review(foreign=True)
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertTrue(contract['manual_note_binding_invalid'])
        self.assertEqual(result['status'], 'no_sources')
        self.assertFalse(result.get('_assistant_core_repair_needed'))

    def test_final_validator_accepts_reviewed_translation_through_zero_removal_projection(self):
        result = self.wired_review(translate=True)
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(contract['manual_note_bindings']['complete'])
        self.assertEqual(contract['grounding_projection']['basis'], 'existing_grounding_projection')
        self.assertEqual(contract['grounding_projection']['removed_claims'], 0)
        self.assertIn('Pass the strip through the guide', result['answer'])
        self.assertIn('Restore the guide afterwards', result['answer'])

    def test_final_validator_does_not_treat_removed_manual_claim_as_formatting(self):
        result = self.wired_review(translate=True, unsupported=True)
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertFalse(contract['semantic_answer_bound'])
        self.assertTrue(contract['manual_note_binding_invalid'])
        self.assertEqual(result['status'], 'no_sources')
        self.assertFalse(result.get('_assistant_core_repair_needed'))


class ManualBindingTests(unittest.TestCase):
    def setUp(self):
        self.note = 'Passare attraverso la guida verticale.'
        self.answer = '2. Feed material.\nManual support\n- ' + self.note
        self.bindings = [{'text':self.note, 'citation_ids':['manual-one']}]
        self.audit = {'reply_mode':'retain', 'input_answer_sha256':review.digest(self.answer),
            'output_answer_sha256':review.digest(self.answer), 'independent_semantic_review':True,
            'source_sequence_immutable':True}
        self.verdict = {'citation_ids':['step-two'], 'procedure_review':self.audit}

    def project(self, reviewed=None, verdict=None, admitted=None):
        return review.manual_note_projection(self.bindings, original_answer=self.answer,
            reviewed_answer=reviewed or self.answer, review=verdict or self.verdict,
            admitted_ids=admitted if admitted is not None else {'step-two','manual-one'})

    def test_retain_restores_only_exact_note_admitted_reference(self):
        ids, proof = self.project()
        self.assertEqual(ids, ['manual-one'])
        self.assertTrue(proof['complete'])
        self.assertEqual(self.verdict['citation_ids'], ['step-two'])

    def test_changed_retain_or_unknown_source_cannot_rebind(self):
        with self.assertRaises(ValueError): self.project(reviewed=self.answer+' Additional action.')
        with self.assertRaises(ValueError): self.project(admitted={'step-two'})
        changed = deepcopy(self.verdict)
        changed['procedure_review']['input_answer_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.project(verdict=changed)

    def test_english_edit_needs_source_reference_and_verified_changed_block(self):
        translated = self.answer.replace(self.note, 'Pass through the vertical guide.')
        audit = {**self.audit, 'reply_mode':'edit', 'output_answer_sha256':review.digest(translated)}
        verdict = {'citation_ids':['step-two','manual-one'], 'procedure_review':audit,
            'edits':[{'block_id':'step_1','text':'Feed material.\nManual support\n- Pass through the vertical guide.'}],
            'operation_boundary_validation':{'complete':True,'precedence_checked':True,'answer_sha256':review.digest(translated)}}
        ids, proof = self.project(reviewed=translated, verdict=verdict)
        self.assertEqual(ids, [])
        self.assertTrue(proof['complete'])
        for key in ('citation_ids','edits','operation_boundary_validation'):
            broken = deepcopy(verdict); broken[key] = [] if key != 'operation_boundary_validation' else {}
            with self.subTest(key=key), self.assertRaises(ValueError): self.project(reviewed=translated, verdict=broken)


if __name__ == '__main__': unittest.main(verbosity=2)
