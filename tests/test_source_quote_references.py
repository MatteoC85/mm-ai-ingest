"""Lossless quotation references; authority and semantic verdicts are not inferred."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review, review_source_wire, source_table_review, enumeration_review
from context_review_wire_fixture import encode_context_reply, original_context_sources
import test_factual_context_review as fixture


class SourceQuoteReferencesTests(unittest.TestCase):
    def fixture(self):
        helper = fixture.FactualContextReviewTests()
        wire, first, frame, catalog = helper.candidate()
        messages = review.build_messages(frame, question='Which maintenance?', response_language='en', sources_block=helper.sources)
        return helper, wire, first, frame, catalog, messages

    def table_frame(self, extra=''):
        qualifier = 'È richiesta la protezione «A»; preserve <script>alert("x")</script> and \\ literally.'
        header = '- [PDF_TABLE_HEADER rows=1 columns=2] c1 r1="Item" | c2 r1="Qualification"'
        value = 'Only when required, inspect the stated component.'
        row = '- [PDF_TABLE_RECORD rows=2 columns=2] c1 "Item" r2="Assembly" | c2 "Qualification" r2=' + json.dumps(value)
        body = qualifier + '\n' + header + '\n' + row + ('\n' + extra if extra else '')
        blocks = []
        views = {}
        for index,cid in enumerate(('manual:p1','other:p1'),1):
            annotated = body.replace(row, '[TABLE_RECORD T' + str(index) + ']\n' + row)
            blocks.append('[' + cid + '] source_type=document; p=1\n' + annotated)
            views[cid] = dict(body=body,page_from=1,page_to=1)
        original = '\n\n'.join(blocks) + '\n'
        catalog = source_table_review.compile_catalog(original,views,'en')
        self.assertEqual(len(catalog.entries),2)
        sources = review_source_wire.compact_sources_block(original)
        wire = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]', expected_list_items=['[[SOURCE_TABLE:G1]]'],
                    citation_ids=['manual:p1','other:p1'], missing_facets=[], missing_answer_types=[], missing_list_items=[],
                    table_record_coverage=['T1=0','T2!Separate source occurrence'])
        first = source_table_review.resolve_reply(wire,catalog)
        first = enumeration_review.resolve_inventory_answer(first,canonical_inventory=first['expected_list_items'],allowed=True)
        frame = review.prepare_review(wire,first,catalog=catalog,admitted_citation_ids=list(views),sources_block=sources)
        return qualifier,value,sources,original,first,frame

    def test_reference_decodes_before_unchanged_literal_and_final_answer_binding(self):
        helper,_,first,frame,_,messages = self.fixture()
        schema = review.schema(frame)['schema']
        reference = schema['properties']['table_contexts']['properties']['item:0']['anyOf'][0]['properties']['supplementary_source_quotes']['items']
        self.assertEqual(reference, {'$ref': '#/$defs/source_quote_reference'})
        properties = schema['$defs']['source_quote_reference']
        self.assertEqual(properties['type'],'string')
        self.assertEqual(set(properties['enum']),set(review._atomic_references(frame)))
        canonical = helper.verdict()
        wire = encode_context_reply(canonical,messages,source_ids=frame.admitted_citation_ids)
        self.assertNotIn('quote',wire['table_contexts']['item:0']['owning_source_quote'])
        out = review.apply_wire_review(first,wire,frame)
        self.assertEqual(out,review.apply_review(first,canonical,frame))
        self.assertEqual(out['independent_context_review']['answer_sha256'],hashlib.sha256(out['answer'].encode()).hexdigest())
        self.assertIn(fixture.CONTEXT,out['answer'])
        self.assertNotIn('[/Q]',out['answer'])
        self.assertEqual(out['expected_list_items'],first['expected_list_items'])
        canonical['table_contexts'][0]['source_quotes'][0]['quote']='A paraphrase absent from the source.'
        self.assertEqual(review.apply_review(first,canonical,frame)['independent_context_review']['gate'],'quote_not_literal')

    def test_compacted_source_headers_geometry_and_all_bytes_roundtrip_without_public_metadata(self):
        qualifier,value,sources,original,first,frame = self.table_frame('long ' * 230)
        self.assertTrue(sources.startswith('LOSSLESS_SOURCE_LINES_V1\n'))
        self.assertTrue(hasattr(frame,'quote_view'))
        self.assertEqual(original_context_sources(frame.quote_view),sources)
        self.assertEqual(review_source_wire.restore_sources_block(original_context_sources(frame.quote_view)),original)
        self.assertGreater(len(frame.quote_units),3)
        self.assertTrue(all(0<len(text)<=480 for _,text,_ in frame.quote_units))
        self.assertTrue(all(text in dict(frame.source_bodies)[cid] for _,text,owners in frame.quote_units for cid in owners))
        self.assertTrue(all('[PDF_TABLE_' not in text and 'c2 "Qualification" r2=' not in text for _,text,_ in frame.quote_units))
        self.assertTrue(any(text==value for _,text,_ in frame.quote_units))
        self.assertTrue(any(text==qualifier for _,text,_ in frame.quote_units))
        shared = next(owners for _,text,owners in frame.quote_units if text==value)
        self.assertEqual(set(shared),{'manual:p1','other:p1'})
        messages=review.build_messages(frame,question='Which component?',response_language='en',sources_block=sources)
        verdict=fixture.FactualContextReviewTests().verdict(citations=['manual:p1','other:p1'])
        verdict['table_contexts'][0]['source_quotes']=[{'citation_id':'manual:p1','quote':value},
            {'citation_id':'manual:p1','quote':qualifier}]
        out=review.apply_wire_review(first,encode_context_reply(verdict,messages,source_ids=frame.admitted_citation_ids),frame)
        self.assertEqual(out['outcome'],'rewrite')
        self.assertIn('> '+value,out['answer'])
        self.assertNotIn('c2 "Qualification" r2=',out['answer'])

    def test_unknown_noncanonical_cross_source_and_free_text_references_fail_closed(self):
        helper,_,first,frame,_,messages=self.fixture()
        good=encode_context_reply(helper.verdict(citations=['manual:p1','manual:p2']),messages,source_ids=frame.admitted_citation_ids)
        ref=good['table_contexts']['item:0']['owning_source_quote']
        cases=[('0:99999','quote_source'),('00:'+ref.split(':')[1],'quote_source'),
               (ref+' 0:2','quote_source'),(ref+'\n','quote_source'),
               ({'quote':'Invented free text'},'quote_shape'),('1:'+ref.split(':')[1],'quote_source'),
               (True,'quote_shape')]
        for change,gate in cases:
            bad=deepcopy(good);bad['table_contexts']['item:0']['owning_source_quote']=change
            with self.subTest(change=change):
                result=review.apply_wire_review(first,bad,frame)
                self.assertEqual(result['outcome'],'no_sources');self.assertEqual(result['answer'],'')
                self.assertEqual(result['independent_context_review']['gate'],gate)
        canonical=helper.verdict();canonical['citation_ids']=['S0']
        canonical['table_contexts'][0]['source_quotes'][0]['citation_id']='S0'
        self.assertEqual(review.apply_wire_review(first,canonical,frame)['independent_context_review']['gate'],'reply_shape')

    def test_shared_text_never_substitutes_the_table_owner_and_tampered_text_hits_literal_gate(self):
        _,value,sources,_,first,frame=self.table_frame()
        messages=review.build_messages(frame,question='Which component?',response_language='en',sources_block=sources)
        verdict=fixture.FactualContextReviewTests().verdict(citations=['manual:p1','other:p1'])
        verdict['table_contexts'][0]['source_quotes']=[{'citation_id':'other:p1','quote':value}]
        self.assertEqual(review.apply_wire_review(first,encode_context_reply(verdict,messages,source_ids=frame.admitted_citation_ids),frame)['independent_context_review']['gate'],'table_owner')
        helper,_,first,frame,_,messages=self.fixture()
        wire=encode_context_reply(helper.verdict(),messages,source_ids=frame.admitted_citation_ids)
        ref='Q'+wire['table_contexts']['item:0']['owning_source_quote'].split(':')[1]
        damaged=replace(frame,quote_units=tuple((q,'Invented text' if q==ref else text,owners)for q,text,owners in frame.quote_units))
        self.assertEqual(review.apply_wire_review(first,wire,damaged)['independent_context_review']['gate'],'quote_not_literal')
        changed=deepcopy(first);changed['answer']='Changed first response'
        self.assertEqual(review.apply_wire_review(changed,wire,frame)['independent_context_review']['gate'],'frame_mismatch')

    def test_source_marker_collisions_and_changed_source_packet_cannot_create_references(self):
        helper,wire,first,frame,catalog,_=self.fixture()
        for marker in ('[Q1]','[Q0]','[Q01]','[ Q 01 ]','[/Q]','[ / q ]'):
            with self.subTest(marker=marker),self.assertRaisesRegex(ValueError,'^factual_context_review_invalid$'):
                review.prepare_review(wire,first,catalog=catalog,admitted_citation_ids=['manual:p1','manual:p2'],sources_block=helper.sources+'\n'+marker)
        with self.assertRaisesRegex(ValueError,'^factual_context_review_invalid$'):
            review.build_messages(frame,question='q',response_language='en',sources_block=helper.sources+' changed')
        with self.assertRaisesRegex(ValueError,'^factual_context_review_invalid$'):
            review.build_messages(replace(frame,quote_view=frame.quote_view+' changed'),question='q',response_language='en',sources_block=helper.sources)

    def test_reference_markers_cannot_leak_into_context_edits_or_first_candidate(self):
        helper,raw,first,frame,catalog,messages=self.fixture()
        good=encode_context_reply(helper.verdict(),messages,source_ids=frame.admitted_citation_ids)
        for marker in ('[Q1]','[/Q]','[ Q 01 ]'):
            bad=deepcopy(good);bad['table_contexts']['item:0']['context']+=' '+marker
            self.assertEqual(review.apply_wire_review(first,bad,frame)['independent_context_review']['gate'],'table_marker')
            bad=deepcopy(good);bad['edits']=[{'target':'item:1','text':'Keep '+marker}]
            self.assertEqual(review.apply_wire_review(first,bad,frame)['independent_context_review']['gate'],'edit_target')
            bad=deepcopy(good);bad['outcome']='no_sources';bad['table_contexts']['item:0']['context']+=marker
            result=review.apply_wire_review(first,bad,frame)
            self.assertEqual(result['independent_context_review']['gate'],'model_rejected')
            self.assertEqual(result['answer'],'')
            wire=deepcopy(raw);expanded=deepcopy(first)
            wire['expected_list_items'][1]+=marker;expanded['expected_list_items'][1]+=marker
            expanded['answer']=review._expand(wire['answer'],expanded['expected_list_items'])
            with self.assertRaisesRegex(ValueError,'^factual_context_review_invalid$'):
                review.prepare_review(wire,expanded,catalog=catalog,admitted_citation_ids=['manual:p1','manual:p2'],sources_block=helper.sources)
        ordinary=deepcopy(good);ordinary['table_contexts']['item:0']['context']+=' Identifier Q1 stays a normal identifier.'
        self.assertEqual(review.apply_wire_review(first,ordinary,frame)['outcome'],'rewrite')

    def test_malformed_technical_markers_stay_context_and_plain_no_table_path_stays_unchanged(self):
        for text in ('- [PDF_TABLE_RECORD broken c1 r9="value"','[[SOURCE_TABLE:broken','[TABLE_RECORD broken'):
            sources='[manual:p1] source_type=document; p=1\n'+text
            view,units=review._source_quote_view(sources,{'manual:p1':text})
            self.assertEqual(original_context_sources(view),sources)
            self.assertEqual(units,())
        helper=fixture.FactualContextReviewTests();_,first,frame,_=helper.candidate(table=False)
        messages=review.build_messages(frame,question='q',response_language='en',sources_block=helper.sources)
        actual=messages[1]['content'].split('\n\nSOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(actual,helper.sources)
        self.assertEqual(frame.quote_units,())
        wire=encode_context_reply(helper.verdict(table=False),messages)
        self.assertEqual(review.apply_wire_review(first,wire,frame)['answer'],first['answer'])


if __name__=='__main__':unittest.main(verbosity=2)
