"""Atomic exact-span patches preserve original blocks and downstream validators."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review


class ReviewMicroEditTests(unittest.TestCase):
    DRAFT='2. Inspect the inlet.\nKeep hands clear.\n\n7. Inspect the outlet.\nKeep the supply isolated.'

    def decode(self,edits,draft=None):
        draft=self.DRAFT if draft is None else draft
        layout=review.block_layout(draft)
        aliases=review.review_wire_aliases(layout,['manual:one'])
        value={'outcome':'rewrite','reply_mode':'edit','answer':'','citation_ids':['s1'],'edits':edits,
            'covered_facets':['inspection'],'missing_facets':[],
            'covered_answer_types':['ordered_actions'],'missing_answer_types':[],'missing_list_items':[],
            'operation_boundary':{'complete':True,'checked_blocks':list(aliases['blocks']),
                                  'precedence':[],'change_edges':[]}}
        original=deepcopy(value)
        result=review.decode_review_wire_aliases(value,aliases,layout=layout)
        self.assertEqual(value,original)
        return result,layout

    def resolve(self,result,layout):
        return review.resolve_reply(result,answer=''.join(b['text']for b in layout['blocks']),layout=layout,
            required_facets=['inspection'],required_types=['ordered_actions'],source_ordered=True,
            source_notes=({'citation_id':'step:2','source_number':'2','texts':('Keep hands clear.',)},
                          {'citation_id':'step:7','source_number':'7','texts':('Keep the supply isolated.',)}))

    def test_minimal_anchor_preserves_every_unedited_character_and_source_note(self):
        result,layout=self.decode([{'block_id':'b1','old_text':'the inlet','new_text':'the marked inlet'}])
        self.assertEqual(result['edits'],[{'block_id':'step_1','text':'Inspect the marked inlet.\nKeep hands clear.'}])
        answer=self.resolve(result,layout)['answer']
        self.assertEqual(answer,self.DRAFT.replace('the inlet','the marked inlet'))

    def test_missing_empty_whitespace_and_overlapping_duplicate_anchors_reject(self):
        for draft,old in ((self.DRAFT,'not present'),(self.DRAFT,''),(self.DRAFT,' '),
                          ('2. aaa\nKeep hands clear.','aa')):
            with self.subTest(old=old),self.assertRaises(ValueError):
                self.decode([{'block_id':'b1','old_text':old,'new_text':'replacement'}],draft)
        with self.assertRaises(ValueError):
            self.decode([{'block_id':'b1','old_text':'inspect','new_text':'replacement'}],
                        '2. inspect and inspect again.')

    def test_invalid_block_and_duplicate_targets_cannot_chain(self):
        for edits in ([{'block_id':'b9','old_text':'Inspect','new_text':'Check'}],
                      [{'block_id':'b1','old_text':'Inspect','new_text':'Check'},
                       {'block_id':'b1','old_text':'Check','new_text':'Skip'}]):
            with self.subTest(edits=edits),self.assertRaises(ValueError):self.decode(edits)

    def test_multiple_blocks_use_their_own_original_body_atomically(self):
        result,layout=self.decode([{'block_id':'b1','old_text':'the inlet','new_text':'the labelled inlet'},
                                   {'block_id':'b2','old_text':'the outlet','new_text':'the labelled outlet'}])
        self.assertEqual(self.resolve(result,layout)['answer'],self.DRAFT.replace('the inlet','the labelled inlet').replace('the outlet','the labelled outlet'))
        with self.assertRaises(ValueError):
            self.decode([{'block_id':'b1','old_text':'the inlet','new_text':'the labelled inlet'},
                         {'block_id':'b2','old_text':'the labelled inlet','new_text':'wrong cross-block anchor'}])

    def test_heading_is_not_an_anchor_and_new_numbered_step_still_rejects(self):
        with self.assertRaises(ValueError):self.decode([{'block_id':'b1','old_text':'2.','new_text':'3.'}])
        result,layout=self.decode([{'block_id':'b1','old_text':'Inspect the inlet.',
                                    'new_text':'99. Skip the inlet.'}])
        with self.assertRaises(ValueError):self.resolve(result,layout)

    def test_moving_a_note_between_patches_cannot_satisfy_step_binding(self):
        result,layout=self.decode([{'block_id':'b1','old_text':'Keep hands clear.','new_text':''},
            {'block_id':'b2','old_text':'Inspect the outlet.','new_text':'Inspect the outlet.\nKeep hands clear.'}])
        resolved=self.resolve(result,layout)
        self.assertEqual(resolved['outcome'],'no_sources')
        self.assertFalse(resolved['procedure_review']['source_note_bindings_complete'])


if __name__=='__main__':unittest.main(verbosity=2)
