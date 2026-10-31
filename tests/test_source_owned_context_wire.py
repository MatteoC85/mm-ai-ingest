"""Atomic citations and exact candidate reuse; no simulated semantic quality."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review
from context_review_wire_fixture import encode_context_reply, original_context_sources, assembled_context_candidate
import test_factual_context_review as context_fixture
import test_source_quote_references as quote_fixture


def candidate_view(messages):
    return json.loads(messages[1]['content'].split('\n\nCOMPLETE_CANDIDATE:\n',1)[1].split('\n\n',1)[0])


class SourceOwnedContextWireTests(unittest.TestCase):
    def test_atomic_selection_preserves_current_and_only_explicit_support_citations(self):
        helper=context_fixture.FactualContextReviewTests()
        _,first,frame,_=helper.candidate()
        messages=review.build_messages(frame,question='Maintenance?',response_language='en',sources_block=helper.sources)
        verdict=helper.verdict()
        verdict['table_contexts'][0]['source_quotes'].append({
            'citation_id':'manual:p2','quote':'Use the documented personal protective equipment.'})
        wire=encode_context_reply(verdict,messages,source_ids=frame.admitted_citation_ids)
        wire['additional_citation_ids']=[]
        self.assertNotIn('citation_ids',wire)
        self.assertIsInstance(wire['table_contexts']['item:0']['owning_source_quote'],str)
        out=review.apply_wire_review(first,wire,frame)
        canonical=deepcopy(verdict);canonical['citation_ids']=['manual:p1','manual:p2']
        self.assertEqual(out,review.apply_review(first,canonical,frame))
        self.assertEqual(out['citation_ids'],['manual:p1','manual:p2'])
        self.assertEqual(out['independent_context_review']['answer_sha256'],hashlib.sha256(out['answer'].encode()).hexdigest())
        # The direct canonical gate still refuses an uncited quoted source.
        self.assertEqual(review.apply_review(first,verdict,frame)['independent_context_review']['gate'],'quote_source')
        wire['table_contexts']['item:0']['supplementary_source_quotes'].pop()
        self.assertEqual(review.apply_wire_review(first,wire,frame)['citation_ids'],['manual:p1'])
        wire['additional_citation_ids']=['S1']
        self.assertEqual(review.apply_wire_review(first,wire,frame)['citation_ids'],['manual:p1','manual:p2'])

    def test_immutable_table_occurrence_is_reconstructed_exactly_from_emitted_source(self):
        helper=quote_fixture.SourceQuoteReferencesTests()
        _,_,sources,_,first,frame=helper.table_frame()
        messages=review.build_messages(frame,question='Which component?',response_language='en',sources_block=sources)
        view=candidate_view(messages)
        self.assertIn('source_table',view['inventory'][0])
        self.assertNotIn('text',view['inventory'][0])
        self.assertEqual(assembled_context_candidate(messages),first['answer'])
        displayed=messages[1]['content'].split('\n\nSOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(displayed,sources),sources)
        self.assertEqual(frame.items,tuple(first['expected_list_items']))
        self.assertNotIn('CITATION_ALIASES',messages[1]['content'])
        # Candidate-only conditions must remain visible, never disappear into a reference.
        for suffix in ('\nKeep the separately documented qualification.', '\n  - Additional cell: «unaltered»'):
            changed=frame.items[0]+suffix
            extra=replace(frame,items=(changed,),answer=review._expand(frame.template,[changed]))
            fallback=review.build_messages(extra,question='q',response_language='en',sources_block=sources)
            self.assertEqual(candidate_view(fallback)['inventory'][0]['text'],changed)
            self.assertNotIn('source_table',candidate_view(fallback)['inventory'][0])
            self.assertEqual(assembled_context_candidate(fallback),extra.answer)
        # Equal text from another source does not merge the owning occurrence.
        self.assertEqual(dict(frame.source_bodies)['manual:p1'],dict(frame.source_bodies)['other:p1'])
        self.assertEqual(view['inventory'][0]['source_table'],'S0')

    def test_unstructured_immutable_item_and_non_table_items_stay_full_text(self):
        helper=context_fixture.FactualContextReviewTests()
        _,first,frame,_=helper.candidate()
        messages=review.build_messages(frame,question='q',response_language='en',sources_block=helper.sources)
        view=candidate_view(messages)
        self.assertEqual([item['text'] for item in view['inventory']],list(frame.items))
        self.assertEqual(assembled_context_candidate(messages),first['answer'])

    def test_enum_capacity_fallback_preserves_all_units_sources_and_previous_protocol(self):
        helper=quote_fixture.SourceQuoteReferencesTests()
        _,value,sources,_,first,frame=helper.table_frame('\n'.join('Distinct literal item '+str(i) for i in range(600)))
        self.assertGreater(sum(len(owners) for _,_,owners in frame.quote_units),1000)
        self.assertIsNone(review._atomic_references(frame))
        schema=review.schema(frame)['schema']['properties']
        self.assertIn('citation_ids',schema)
        self.assertNotIn('additional_citation_ids',schema)
        self.assertIn('quote_ref',schema['table_contexts']['items']['properties']['source_quotes']['items']['properties'])
        messages=review.build_messages(frame,question='q',response_language='en',sources_block=sources)
        self.assertIn('CITATION_ALIASES',messages[1]['content'])
        self.assertEqual(assembled_context_candidate(messages),first['answer'])
        displayed=messages[1]['content'].split('\n\nSOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(displayed),sources)
        verdict=context_fixture.FactualContextReviewTests().verdict(citations=['manual:p1','other:p1'])
        verdict['table_contexts'][0]['source_quotes']=[{'citation_id':'manual:p1','quote':value}]
        out=review.apply_wire_review(first,encode_context_reply(verdict,messages),frame)
        self.assertEqual(out,review.apply_review(first,verdict,frame))

    def test_atomic_schema_total_limits_and_foreign_pair_do_not_change_canonical_gates(self):
        helper=context_fixture.FactualContextReviewTests()
        _,first,frame,_=helper.candidate()
        schema=review.schema(frame)
        counts=[]
        def visit(node):
            if isinstance(node,dict):
                if 'enum' in node:
                    counts.append(len(node['enum']))
                    if len(node['enum'])>250:self.assertLessEqual(sum(map(len,node['enum'])),15000)
                for child in node.values():visit(child)
            elif isinstance(node,list):
                for child in node:visit(child)
        visit(schema)
        self.assertLessEqual(sum(counts),1000)
        messages=review.build_messages(frame,question='q',response_language='en',sources_block=helper.sources)
        good=encode_context_reply(helper.verdict(),messages,source_ids=frame.admitted_citation_ids)
        for ref in ('999:1','0:0','0:01',True,{'source':'S0','quote':'literal'}):
            bad=deepcopy(good);bad['table_contexts']['item:0']['owning_source_quote']=ref
            result=review.apply_wire_review(first,bad,frame)
            self.assertEqual(result['outcome'],'no_sources')
            self.assertEqual(result['answer'],'')
        invalid=deepcopy(good);invalid['checks']['safety']=False
        self.assertEqual(review.apply_wire_review(first,invalid,frame)['independent_context_review']['gate'],'checks_false')


if __name__=='__main__':unittest.main(verbosity=2)
