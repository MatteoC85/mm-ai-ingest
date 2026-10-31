"""Generated section labels cannot witness a documentary qualification.

Source observations/model decisions are fixtures; no semantic model claim.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_duplicate_table_context as fx
import test_source_table_owned as owned
from context_review_wire_fixture import encode_context_reply,original_context_sources
from machinemind.ask import factual_context_review as review
from machinemind.ask.response_evidence import ResponseEvidenceError

LABEL='SECTION: DOCUMENTED ASSEMBLY'


class ContextProseReferencesTests(unittest.TestCase):
    def candidate(self):
        return fx.candidate({'doc:p1':LABEL+'\n'+fx.SCOPE+'\n'+fx.CELLS,
            'doc:p2':LABEL+'\n'+fx.SCOPE+'\n'+fx.QUALIFIER+'\n'+fx.CELLS})

    def test_section_only_audit_rejects_and_qualifier_refs_preserve_exact_sources(self):
        first,frame,sources,_=self.candidate();good=fx.verdict(frame)
        bad=deepcopy(good)
        for audit in bad['table_contexts']:
            audit['context']='The source concerns the documented section.'
            audit['source_quotes'][0]['quote']=LABEL
        result=review.apply_review(first,bad,frame)
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(result['independent_context_review']['gate'],'table_owner')
        messages=review.build_messages(frame,question='Which intervals?',response_language='en',sources_block=sources)
        targets=json.loads(messages[1]['content'].split('TABLE_CONTEXT_TARGETS:\n',1)[1].split('\n\n',1)[0])
        refs=review._atomic_references(frame)
        for target in targets:
            self.assertTrue(target['owner_prose_refs'])
            self.assertFalse(any(refs[r][1]==LABEL for r in target['owner_prose_refs']))
        self.assertTrue(any(refs[r][1]==fx.QUALIFIER for r in targets[1]['owner_prose_refs']))
        result=review.apply_wire_review(first,encode_context_reply(good,messages,original_sources=sources),frame)
        self.assertEqual(result['outcome'],'rewrite');self.assertIn(fx.QUALIFIER,result['answer'])
        self.assertEqual(result['expected_list_items'],first['expected_list_items'])
        self.assertEqual(result['table_record_coverage'],first['table_record_coverage'])
        emitted=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(emitted,sources),sources)

    def test_only_initial_generated_label_is_separated_and_table_only_still_works(self):
        later='SECTION: This later sentence is part of the document.'
        body=LABEL+'\n'+fx.SCOPE+'\n'+fx.CELLS+'\n'+later
        prose=review._documentary_prose(body)
        self.assertNotIn(LABEL,prose);self.assertIn(fx.SCOPE,prose);self.assertIn(later,prose)
        self.assertIn('Inline '+LABEL,review._documentary_prose('Inline '+LABEL+'\n'+fx.CELLS))
        for body in (fx.CELLS,LABEL+'\n'+fx.CELLS):
            first,frame,sources,_=fx.candidate({'doc:p1':body})
            self.assertIsNone(review._owner_prose_refs(frame,'doc:p1',review._atomic_references(frame)))
            verdict=dict(outcome='rewrite',edits=[],citation_ids=['doc:p1'],checks={k:True for k in review.CHECKS},reason='accepted',
                table_contexts=[dict(target='item:0',context='Only the quoted table is available.',documentary_limits_checked=True,
                    source_quotes=[dict(citation_id='doc:p1',quote='Oil temperature')])])
            self.assertEqual(review.apply_review(first,verdict,frame)['outcome'],'rewrite')

    def test_owner_membership_and_complete_fallback_are_unchanged(self):
        first,frame,sources,_=self.candidate();good=fx.verdict(frame)
        bad=deepcopy(good);bad['table_contexts'][1]['source_quotes']=[{'citation_id':'doc:p1','quote':fx.SCOPE}]
        self.assertEqual(review.apply_review(first,bad,frame)['outcome'],'no_sources')
        # Exhaust only representation capacity; do not truncate the source body.
        units=tuple(('Q'+str(i),fx.QUALIFIER,('doc:p2',))for i in range(1,1100))
        fallback=replace(frame,quote_units=units)
        self.assertIsNone(review._atomic_references(fallback))
        eligible=review._owner_prose_refs(fallback,'doc:p2',None)
        self.assertEqual(len(eligible),len(units))
        messages=review.build_messages(fallback,question='Which intervals?',response_language='en',sources_block=sources)
        emitted=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(emitted,sources),sources)


class ContextProseOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):owned.SourceTableOwnedTests.setUpClass();cls.m=owned.SourceTableOwnedTests.m
    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()
    case=owned.SourceTableOwnedTests.case

    def test_real_owner_rejects_metadata_quote_and_keeps_supported_context_to_html(self):
        for metadata_only in (True,False):
            with self.subTest(metadata_only=metadata_only):
                with patch.object(owned,'TABLE_BODY',LABEL+'\n'+owned.TABLE_BODY):owner,f=self.case()
                original=f.owner._response_flow.execution._v13_json_models
                def provider(messages,**kw):
                    result,model=original(messages,**kw)
                    if kw['json_schema']['name']=='factual_context_review':
                        targets=json.loads(messages[1]['content'].split('TABLE_CONTEXT_TARGETS:\n',1)[1].split('\n\n',1)[0])
                        self.assertTrue(targets[0]['owner_prose_refs'])
                        canonical=dict(outcome='rewrite',edits=[],citation_ids=['inspection:p1','inspection:p2'],
                            checks={k:True for k in review.CHECKS},reason='accepted',table_contexts=owned.table_contexts())
                        if metadata_only:canonical['table_contexts'][0]['source_quotes'][0]['quote']=LABEL
                        first_sources=f.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                        result=encode_context_reply(canonical,messages,original_sources=first_sources)
                        self.assertEqual(original_context_sources(messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0],first_sources),first_sources)
                    return result,model
                f.owner._response_flow.execution=replace(f.owner._response_flow.execution,_v13_json_models=provider)
                if metadata_only:
                    with self.assertRaises(ResponseEvidenceError):f.run()
                else:
                    result=f.run();self.assertEqual(result['status'],'answered')
                    for text in (owned.TABLE_QUALIFIER.rstrip('.'),owned.CONDITION.rstrip('.'),'Monthly'):
                        self.assertIn(text,result['answer']);self.assertIn(text,result['answer_html'])
                    contract=result['meta']['assistant_core_validation']['answer_contract']
                    self.assertTrue(contract['semantic_answer_bound'])
                    self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
                self.assertEqual(len(f.posts),2);self.assertEqual(f.writer_calls,[])
                self.assertEqual(f.budget_ids,[id(owner.budget)]*2)
                f.close();owner.close()


if __name__=='__main__':unittest.main(verbosity=2)
