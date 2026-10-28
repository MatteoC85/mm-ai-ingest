"""Exact source-backed ordering proofs for independently reviewed procedure edits.

Semantic completeness remains the reviewer's responsibility. These offline
fixtures exercise deterministic quote identity, block binding and ordering.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review


class SourceOrderedBoundaryTests(unittest.TestCase):
    DRAFT = '1. Prepare the device.\n\n4. Inspect the transfer guide.\nKeep hands clear.'
    SAFE = ('Stop and isolate the drive.\nAdjust the guide.\nLock the carriage.\n'
            'Close the cover.\nJog the transfer.\nKeep hands clear.')
    SOURCE = ('Stop and isolate the drive before adjusting the guide. '
              'After adjusting the guide, lock the carriage before jogging the transfer. '
              'Close the cover before jogging the transfer. Keep hands clear.')

    def edge(self, before, after, source):
        return {'before_block_id':'step_2','before_answer_quote':before,
                'after_block_id':'step_2','after_answer_quote':after,
                'source_citation_id':'manual:one','source_quote':source}

    def reply(self, body=None):
        value = {'outcome':'rewrite','reply_mode':'edit','answer':'',
            'edits':[{'block_id':'step_2','text':self.SAFE if body is None else body}],
            'covered_facets':['operation'],'missing_facets':[],
            'covered_answer_types':['ordered_actions'],'missing_answer_types':[],
            'missing_list_items':[], 'citation_ids':['step:four'],
            'operation_boundary':{'complete':True,'checked_blocks':['step_1','step_2'],
                'changes':[{'block_id':'step_2','action_quote':'Adjust the guide.',
                    'closing_source_citation_id':'manual:one',
                    'closing_source_quote':'After adjusting the guide, lock the carriage before jogging the transfer.',
                    'closing_answer_quote':'Lock the carriage.','resolution':'restored'}],
                'precedence':[
                    self.edge('Stop and isolate the drive.','Adjust the guide.',
                              'Stop and isolate the drive before adjusting the guide.'),
                    self.edge('Adjust the guide.','Lock the carriage.',
                              'After adjusting the guide, lock the carriage before jogging the transfer.'),
                    self.edge('Lock the carriage.','Jog the transfer.',
                              'After adjusting the guide, lock the carriage before jogging the transfer.'),
                    self.edge('Close the cover.','Jog the transfer.',
                              'Close the cover before jogging the transfer.')]}}
        return review.resolve_reply(value,answer=self.DRAFT,layout=review.block_layout(self.DRAFT),
            required_facets=['operation'],required_types=['ordered_actions'],source_ordered=True,
            source_notes=({'citation_id':'step:four','source_number':'4','texts':('Keep hands clear.',)},))

    def verify(self,value,sources=None):
        return review.verify_operation_boundary(value,layout=review.block_layout(self.DRAFT),
            source_texts={'manual:one':self.SOURCE} if sources is None else sources,source_ordered=True)

    def assert_rejected(self,value):
        result=self.verify(value)
        self.assertEqual(result['outcome'],'no_sources')
        self.assertFalse(result['operation_boundary_validation']['complete'])
        self.assertFalse(result['operation_boundary_validation']['precedence_checked'])

    def test_valid_edit_has_bound_audit_and_keeps_original_step_and_note(self):
        result=self.verify(self.reply())
        self.assertEqual(result['outcome'],'rewrite')
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertTrue(result['operation_boundary_validation']['precedence_checked'])
        self.assertEqual(result['operation_boundary_validation']['precedence_edges'],4)
        self.assertEqual(result['operation_boundary_validation']['answer_sha256'],review.digest(result['answer']))
        self.assertIn('\n\n4. ',result['answer'])
        self.assertTrue(result['answer'].endswith('Keep hands clear.'))
        self.assertIn('manual:one',result['citation_ids'])

    def test_missing_prerequisite_or_preparation_after_action_fails_closed(self):
        for body in (self.SAFE.replace('Stop and isolate the drive.\n',''),
                     self.SAFE.replace('Stop and isolate the drive.\nAdjust the guide.',
                                       'Adjust the guide.\nStop and isolate the drive.')):
            with self.subTest(body=body):self.assert_rejected(self.reply(body))

    def test_restoration_after_motion_is_rejected_even_when_before_later_restart(self):
        body=self.SAFE.replace('Close the cover.\nJog the transfer.',
                               'Jog the transfer.\nClose the cover.\nResume automatic operation.')
        self.assert_rejected(self.reply(body))

    def test_changed_block_cannot_claim_complete_without_precedence_proof(self):
        for mode in ('missing','empty','unchecked'):
            with self.subTest(mode=mode):
                value=self.reply()
                if mode=='missing':value['operation_boundary'].pop('precedence')
                if mode=='empty':value['operation_boundary'].update(precedence=[],changes=[])
                if mode=='unchecked':value['operation_boundary']['checked_blocks']=['step_1']
                self.assert_rejected(value)

    def test_repeated_quotes_or_wrong_final_block_are_not_ambiguous_proof(self):
        value=self.reply(self.SAFE.replace('Jog the transfer.','Jog the transfer.\nJog the transfer.'))
        self.assert_rejected(value)
        value=self.reply();value['operation_boundary']['precedence'][0]['before_block_id']='step_1'
        self.assert_rejected(value)

    def test_unknown_source_changed_negation_or_numeric_quote_is_rejected(self):
        for mutate in (lambda edge:edge.update(source_citation_id='foreign:one'),
                       lambda edge:edge.update(source_quote='Do not isolate the drive before adjusting the guide.'),
                       lambda edge:edge.update(source_quote='Set pressure to 25 bar.')):
            value=self.reply();mutate(value['operation_boundary']['precedence'][0]);self.assert_rejected(value)

    def test_declared_temporary_change_cannot_use_another_occurrence_for_closure(self):
        value=self.reply();value['operation_boundary']['changes'][0]['block_id']='step_1'
        self.assert_rejected(value)
        value=self.reply();value['operation_boundary']['precedence'].pop(1)
        self.assert_rejected(value)

    def test_unchanged_draft_can_report_no_new_dependency_without_fabrication(self):
        value={'outcome':'pass','answer':self.DRAFT,'operation_boundary':{
            'complete':True,'checked_blocks':['step_1','step_2'],'changes':[],'precedence':[]}}
        result=self.verify(value)
        self.assertEqual(result['outcome'],'pass')
        self.assertTrue(result['operation_boundary_validation']['precedence_checked'])
        # A missing proof field is not equivalent to a reviewed empty list.
        value['operation_boundary'].pop('precedence');self.assert_rejected(value)

    def test_schema_requires_compact_precedence_only_for_source_ordered_review(self):
        legacy={'name':'test','schema':{'properties':{},'required':[]}}
        layout=review.block_layout(self.DRAFT)
        wire=review.review_schema(legacy,layout,[],[],source_ordered=True,
            operation_boundary=True,operation_source_ids=['manual:one'])['schema']
        boundary=wire['properties']['operation_boundary']
        self.assertIn('precedence',boundary['required'])
        edges=boundary['properties']['precedence']
        self.assertEqual(edges['maxItems'],16)
        self.assertEqual(edges['items']['properties']['source_quote']['maxLength'],400)
        self.assertEqual(wire['properties']['reply_mode']['enum'],['retain','edit','reject'])
        manual=review.review_schema(legacy,layout,[],[],operation_boundary=True,
            operation_source_ids=['manual:one'])['schema']['properties']['operation_boundary']
        self.assertNotIn('precedence',manual['properties'])


if __name__=='__main__':unittest.main(verbosity=2)
