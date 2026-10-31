"""Owned representation tests; model choices are fixtures, never quality proof."""
from pathlib import Path
import hashlib
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_factual_context_owned as owned
from context_review_wire_fixture import original_context_sources
from machinemind.ask import request_completion

FACT='The documented inspection interval is monthly.'
UNREQUESTED=FACT+' Open the access guards, follow safety rules, then restart.'
PRECONDITIONS='Before opening the access guards, select maintenance mode and isolate the energy. Only trained personnel may perform the intervention.'
SOURCE=FACT+' '+PRECONDITIONS+' Close the guards after the intervention before restarting.'


class InformationalFactualScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.FactualContextOwnedTests.setUpClass()
        cls.m=owned.FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.FactualContextOwnedTests.tearDownClass()

    def case(self, *, query, correction=None, rejected=False):
        owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=False),started=time.monotonic())
        self.addCleanup(owner.close)
        def first_change(reply):reply['expected_list_items'][1]=UNREQUESTED
        def second(reply):
            if rejected:reply['checks']['safety']=False
            elif correction is not None:reply['edits']=[{'target':'item:1','text':correction}]
        with patch.object(owned,'SAFE_ITEM',SOURCE):
            fixture=owned.ContextFixture(self,query=query,first_change=first_change,second=second)
        self.addCleanup(fixture.close)
        return owner,fixture

    def assert_bound(self,result,f):
        self.assertEqual(result['status'],'answered')
        contract=f.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed']);self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
        self.assertEqual(len(f.post_calls),2);self.assertEqual(f.writer_calls,[])
        first=owned.source_packet(f.calls[0]['messages'][1]['content'],'\n\nCURRENT_ANSWER:')
        second=owned.source_packet(f.calls[1]['messages'][1]['content'],'\n\nCOMPLETE_CANDIDATE:')
        self.assertEqual(first,original_context_sources(second,first))
        self.assertIn(SOURCE,first)
        self.assertEqual(f.session.summary()['registered_reads'],f.initial_reads)
        self.assertEqual({c['citation_id']for c in result['citations']},{'inspection:p1','inspection:p2'})
        return contract['semantic_verifier']

    def test_informational_edit_removes_access_instructions_without_losing_requested_facts_or_tables(self):
        query='Which inspections and intervals are documented? State any missing data.'
        owner,f=self.case(query=query,correction=FACT)
        result=f.run();verifier=self.assert_bound(result,f)
        self.assertIn(UNREQUESTED,f.first_candidate)
        self.assertEqual(verifier['expected_list_items'][1],FACT)
        self.assertIn(FACT,result['answer']);self.assertIn(FACT,result['answer_html'])
        for text in ('Open the access guards','follow safety rules','then restart'):
            self.assertNotIn(text,result['answer']);self.assertNotIn(text,result['answer_html'])
        self.assertEqual(verifier['table_record_coverage'][0]['inventory_items'],[0])
        self.assertIn('Inspect wear every 360 hours and grease only if necessary',result['answer'])
        self.assertIn(owned.table_fixture.TABLE_QUALIFIER,result['answer'])
        self.assertEqual({c['budget_id']for c in f.calls},{id(owner.budget)})
        self.assertEqual(f.request.query,query)

    def test_explicit_intervention_retains_full_source_conditions_in_the_same_editable_slot(self):
        query='Which inspection interval is documented, and what conditions are required before opening the access guards?'
        owner,f=self.case(query=query,correction=FACT+' '+PRECONDITIONS)
        result=f.run();verifier=self.assert_bound(result,f)
        self.assertEqual(verifier['expected_list_items'][1],FACT+' '+PRECONDITIONS)
        for text in ('maintenance mode','isolate the energy','Only trained personnel'):
            self.assertIn(text,result['answer']);self.assertIn(text,result['answer_html'])
        self.assertNotIn('then restart',result['answer'])
        self.assertEqual(f.request.query,query)
        self.assertEqual(verifier['independent_context_review']['edit_count'],1)
        self.assertEqual({c['budget_id']for c in f.calls},{id(owner.budget)})

    def test_rejected_incomplete_conditions_do_not_publish_the_first_draft(self):
        _,f=self.case(query='Which conditions are required before opening the access guards?',rejected=True)
        result=f.run()
        self.assertEqual(result['status'],'no_sources')
        self.assertFalse(result.get('citations'))
        self.assertNotIn(UNREQUESTED,str(result))
        self.assertEqual(len(f.post_calls),2);self.assertEqual(f.writer_calls,[])


if __name__=='__main__':unittest.main(verbosity=2)
