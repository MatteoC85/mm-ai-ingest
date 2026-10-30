"""Independent verdict representation; model quality/authority are not simulated."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import factual_context_review as review
from machinemind.ask import enumeration_review, source_table_review


class FactualContextReviewTests(unittest.TestCase):
    def test_normalization_proof_does_not_change_editable_wire_strings(self):
        raw = 'Verificare ３００\u00a0ore;  usare  il DPI indicato.'
        normalize = lambda value: [' '.join(unicodedata.normalize('NFKC', value).split())]
        wire = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]',
            expected_list_items=[raw], citation_ids=['manual:p1'],
            missing_facets=[], missing_answer_types=[], missing_list_items=[])
        expanded = enumeration_review.resolve_inventory_answer(wire,
            canonical_inventory=normalize(raw), allowed=True, normalize_item=normalize)
        frame = review.prepare_review(wire, expanded,
            catalog=source_table_review.TableCatalog(), admitted_citation_ids=['manual:p1'])
        result = review.apply_review(expanded, self.verdict(), frame)
        self.assertEqual(result['answer'], '- ' + raw)
        self.assertEqual(result['expected_list_items'], [raw])
        self.assertEqual(frame.items, (raw,))

    def candidate(self, *, table=True, template=True):
        literal = 'Original quotation.\n  - Action: «Check cleanliness»; Interval: «Daily»; Note: «Intervene only if needed»'
        entry = source_table_review.TableEntry('[[SOURCE_TABLE:G1]]', literal, 'manual:p1', ('T1',), '0'*64)
        catalog = source_table_review.TableCatalog((entry,) if table else (), '')
        items = ['[[SOURCE_TABLE:G1]]', 'Inspect the level and top up if necessary.'] if table else ['Inspect the level and top up if necessary.']
        body = 'Documented facts:\n[[FACTUAL_INVENTORY]]\nKeep the stated scope.' if template else items[0]
        wire = dict(outcome='rewrite', answer=body, expected_list_items=items,
            covered_list_items=[], missing_list_items=[], citation_ids=['manual:p1'],
            covered_facets=['documented maintenance'], missing_facets=[],
            covered_answer_types=['checklist'], missing_answer_types=[],
            reason='PRIVATE_FIRST_VERDICT', table_record_coverage=['T1=0'] if table else [])
        expanded = source_table_review.resolve_reply(wire, catalog)
        expanded = enumeration_review.resolve_inventory_answer(expanded,
            canonical_inventory=expanded['expected_list_items'], allowed=True)
        frame = review.prepare_review(wire, expanded, catalog=catalog,
            admitted_citation_ids=['manual:p1', 'manual:p2'])
        return wire, expanded, frame, catalog

    def verdict(self, *, outcome='pass', edits=(), citations=('manual:p1',), **checks):
        return dict(outcome=outcome, edits=list(edits), citation_ids=list(citations),
                    checks={key: checks.get(key, True) for key in review.CHECKS}, reason='PRIVATE_SECOND_REASON')

    def test_pass_keeps_generated_rewrite_and_exact_table_inventory(self):
        _, first, frame, _ = self.candidate()
        out = review.apply_review(first, self.verdict(), frame)
        self.assertEqual(out['outcome'], 'rewrite')
        for field in ('answer','expected_list_items','table_record_coverage','citation_ids','missing_facets','missing_list_items'):
            self.assertEqual(out[field], first[field])
        self.assertEqual(out['independent_context_review'], {'outcome':'pass','edit_count':0,
            'answer_sha256':hashlib.sha256(out['answer'].encode()).hexdigest()})
        self.assertNotIn('PRIVATE_SECOND_REASON', str(out))

    def test_supported_safety_completion_changes_only_the_same_non_table_slot(self):
        _, first, frame, _ = self.candidate()
        text = 'Inspect the level. Before topping up, use the documented personal protective equipment.'
        out = review.apply_review(first, self.verdict(outcome='rewrite', edits=[{'target':'item:1','text':text}],
            citations=['manual:p1','manual:p2']), frame)
        self.assertEqual(out['expected_list_items'], [first['expected_list_items'][0], text])
        self.assertEqual(out['table_record_coverage'], first['table_record_coverage'])
        self.assertEqual(out['covered_facets'], first['covered_facets'])
        self.assertIn(first['expected_list_items'][0], out['answer'])
        self.assertIn(text, out['answer'])
        self.assertEqual(out['independent_context_review']['edit_count'], 1)
        self.assertEqual(first['expected_list_items'][1], 'Inspect the level and top up if necessary.')

    def test_template_edit_preserves_exact_inventory_and_hashes_assembled_text(self):
        _, first, frame, _ = self.candidate()
        replacement='Documentary facts, not additional actions:\n[[FACTUAL_INVENTORY]]\nThe stated conditions apply.'
        out=review.apply_review(first,self.verdict(outcome='rewrite',edits=[
            {'target':'template-answer','text':replacement}]),frame)
        self.assertEqual(out['expected_list_items'], first['expected_list_items'])
        self.assertNotIn('[[FACTUAL_INVENTORY',out['answer'])
        for item in first['expected_list_items']:self.assertIn(item,out['answer'])
        self.assertEqual(out['independent_context_review']['answer_sha256'],hashlib.sha256(out['answer'].encode()).hexdigest())

    def test_all_six_independent_checks_are_required_and_missing_first_obligations_stay_missing(self):
        _, first, frame, _ = self.candidate()
        for check in review.CHECKS:
            out=review.apply_review(first,self.verdict(**{check:False}),frame)
            self.assertEqual(out['outcome'],'no_sources')
            self.assertEqual(out['answer'],'')
            self.assertTrue(out['_terminal_review_failure'])
        wire, first, _, catalog=self.candidate()
        for field in ('missing_facets','missing_answer_types','missing_list_items'):
            bad=deepcopy(first);bad[field]=['Not provided']
            with self.assertRaisesRegex(ValueError,'^factual_context_review_invalid$'):
                review.prepare_review(wire,bad,catalog=catalog,admitted_citation_ids=['manual:p1'])

    def test_foreign_ids_table_edits_duplicate_targets_and_malformed_templates_are_rejected(self):
        _,first,frame,_=self.candidate()
        variants=[
            self.verdict(citations=['foreign:p1']), self.verdict(citations=[]),
            self.verdict(outcome='rewrite',edits=[{'target':'item:0','text':'Changed table'}]),
            self.verdict(outcome='rewrite',edits=[{'target':'item:9','text':'New slot'}]),
            self.verdict(outcome='rewrite',edits=[{'target':'item:1','text':'Changed'}, {'target':'item:1','text':'Changed again'}]),
            self.verdict(outcome='rewrite',edits=[{'target':'item:1','text':'[[SOURCE_TABLE:G1]]'}]),
            self.verdict(outcome='rewrite',edits=[{'target':'item:1','text':'[[FACTUAL_INVENTORY]]'}]),
            self.verdict(edits=[{'target':'item:1','text':'Changed on pass'}]),
        ]
        for body in ('No inventory','inline [[FACTUAL_INVENTORY]]','[[FACTUAL_INVENTORY]]\n[[FACTUAL_INVENTORY]]'):
            variants.append(self.verdict(outcome='rewrite',edits=[{'target':'template-answer','text':body}]))
        for v in variants:
            out=review.apply_review(first,v,frame)
            self.assertEqual(out['outcome'],'no_sources')
            self.assertEqual(out['reason'],'factual_context_review_invalid')
            self.assertEqual(out['answer'],'')

    def test_mutation_between_frame_and_apply_cannot_be_approved(self):
        _,first,frame,_=self.candidate()
        for key,value in [('answer','Changed'),('expected_list_items',['Changed']),('citation_ids',['manual:p2']),
                          ('table_record_coverage',['T1!excluded']),('missing_facets',['Missing'])]:
            changed=deepcopy(first);changed[key]=value
            self.assertEqual(review.apply_review(changed,self.verdict(),frame)['outcome'],'no_sources')

    def test_sources_are_byte_identical_and_prior_judgement_is_not_in_second_prompt(self):
        _,_,frame,_=self.candidate()
        sources='[manual:p1] source_type=document; p=1\nExact\nsource\n\n[manual:p2] source_type=document; p=2\nPPE.'
        messages=review.build_messages(frame,question='Which maintenance is documented?',response_language='en',sources_block=sources)
        actual=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(actual,sources)
        self.assertIn(frame.answer,messages[1]['content'])
        self.assertNotIn('PRIVATE_FIRST_VERDICT',str(messages))
        self.assertNotIn('SOURCE_TABLE_PREVIEWS',str(messages))
        self.assertEqual(review.schema(frame)['name'],'factual_context_review')

    def test_plain_inventory_is_supported_and_legacy_no_marker_is_read_only(self):
        for template in (True,False):
            _,first,frame,_=self.candidate(table=False,template=template)
            self.assertEqual(frame.immutable_indices,())
            self.assertEqual(review.apply_review(first,self.verdict(),frame)['answer'],first['answer'])
            if template:
                out=review.apply_review(first,self.verdict(outcome='rewrite',edits=[{'target':'item:0','text':'Use the documented prerequisites before intervention.'}]),frame)
                self.assertEqual(out['outcome'],'rewrite')
            else:
                self.assertEqual(frame.editable_targets,())
                s=review.schema(frame)['schema']['properties']['edits']
                self.assertEqual(s['maxItems'],0)
                self.assertNotIn('enum',s['items']['properties']['target'])
                self.assertEqual(review.apply_review(first,self.verdict(outcome='rewrite'),frame)['outcome'],'no_sources')


if __name__=='__main__':unittest.main(verbosity=2)
