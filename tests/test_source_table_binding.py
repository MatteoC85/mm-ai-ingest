"""Selected-token bindings, unchanged complete coverage and real owner fences.

Model replies and database observations are fixtures, not semantic quality proof.
The original failed live wire was unavailable; these are general regressions.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_source_table_review as pure
import test_source_table_owned as owned
from machinemind.ask import source_table_review as st, enumeration_review as er, request_completion as completion


class SourceTableBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass()
        cls.m = owned.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls): owned.SourceTableOwnedTests.tearDownClass()

    case = owned.SourceTableOwnedTests.case

    def test_selected_token_binds_only_its_owner_and_records_through_publication(self):
        for changes in ({'citation_ids':['inspection:p2']}, {'table_record_coverage':[]},
                        {'citation_ids':['inspection:p2'],'table_record_coverage':[]}):
            with self.subTest(changes=changes):
                owner, f = self.case(change=lambda r:r.update(deepcopy(changes)))
                owner.payload.debug = True
                limits=(owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd)
                with patch('sys.stdout',new=io.StringIO()): result=f.run()
                self.assertEqual(result['status'],'answered')
                for key,value in changes.items():self.assertEqual(f.wire_reply[key],value)
                self.assertEqual({c['citation_id']for c in result['citations']},{'inspection:p1','inspection:p2'})
                contract=f.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['semantic_answer_bound'])
                self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
                for literal in ('Inspect wiring integrity', 'Monthly', owned.CONDITION.rstrip('.')):
                    self.assertIn(literal,result['answer'])
                    self.assertIn(literal,self.m._assistant_ui_visible_text_from_html(result['answer_html']))
                schema=f.reviewer_calls[0][1]['json_schema']['schema']['properties']['table_record_coverage']
                self.assertEqual(schema['type'],'object')
                self.assertFalse(schema['additionalProperties'])
                self.assertEqual(set(schema['properties']),{'T1','T2'})
                self.assertEqual(set(schema['required']),{'T1','T2'})
                self.assertEqual(f.provider_wire_reply['table_record_coverage'],
                    {'T1':None,'T2':None} if 'table_record_coverage'in changes else {'T1':'=0','T2':'=0'})
                observation=next(r for r in owner.contract_checks if r['stage']=='semantic_review')
                self.assertEqual(observation['source_table_gate'],'accepted')
                self.assertEqual(observation['source_table_bound_record_count'],2 if 'table_record_coverage'in changes else 0)
                self.assertEqual(observation['source_table_bound_citation_count'],int('citation_ids'in changes))
                self.assertEqual(len(f.posts),2);self.assertEqual(f.writer_calls,[])
                self.assertEqual(f.budget_ids,[id(owner.budget)]*2)
                self.assertEqual(limits,(owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd))
                self.assertTrue(f.owner._response_flow.completed)
                f.close();owner.close()

    def test_explicit_rejections_and_forged_diagnostic_never_turn_into_bindings(self):
        cases=[
            ({'citation_ids':['foreign:p1']},'citation_not_admitted'),
            ({'expected_list_items':['[[SOURCE_TABLE:G999]]',owned.CONDITION]},'reference_unknown'),
            ({'table_record_coverage':['T1=0','T2!outside request']},'selected_record_conflict'),
            ({'table_record_coverage':['T1=0','T2=1']},'selected_record_conflict'),
            ({'expected_list_items':['[[SOURCE_TABLE:G1]]']*2},'reference_duplicate'),
            ({'expected_list_items':['Grease every day.',owned.CONDITION]},'required_reference_missing'),
        ]
        for changes,gate in cases:
            with self.subTest(gate=gate):
                def change(r):
                    r.update(deepcopy(changes));r['source_table_gate']='accepted'
                    r['source_table_bound_record_count']=999
                owner,f=self.case(change=change);owner.payload.debug=True
                with patch('sys.stdout',new=io.StringIO())as logs:result=f.run()
                self.assertEqual(result['status'],'no_sources');self.assertEqual(len(f.posts),1)
                observation=next(r for r in owner.contract_checks if r['stage']=='semantic_review')
                self.assertEqual(observation['source_table_gate'],gate)
                self.assertNotIn('source_table_bound_record_count',observation)
                self.assertNotIn('foreign:p1',logs.getvalue())
                f.close();owner.close()
        # A duplicate object key must be tested as raw JSON before parsing
        # (the scoped transport suite does so). The unchanged canonical decoder
        # still rejects duplicate rows rather than silently coalescing them.
        helper=pure.SourceTableReviewTests();catalog=helper.catalog();reply=helper.result(catalog)
        reply['table_record_coverage']=['T1=0','T1=0']
        diagnostic={}
        result=st.resolve_reply(reply,catalog,admitted_citation_ids={pure.CID},diagnostics=diagnostic)
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(diagnostic['source_table_gate'],'coverage_duplicate')
        self.assertNotIn('source_table_bound_record_count',diagnostic)

    def test_unselected_occurrences_still_require_complete_canonical_classification(self):
        helper=pure.SourceTableReviewTests()
        catalog=helper.catalog([pure.row(),pure.row('document:beta:p3-4:c1',page=3)])
        reply=helper.result(catalog);reply.update(citation_ids=[],table_record_coverage=[])
        before=deepcopy(reply)
        out=st.resolve_reply(reply,catalog,admitted_citation_ids={e.citation_id for e in catalog.entries})
        self.assertEqual(reply,before)
        self.assertEqual(out['citation_ids'],[catalog.entries[0].citation_id])
        self.assertEqual(out['table_record_coverage'],['T1=0','T2=0'])
        ids=tuple(r for e in catalog.entries for r in e.record_ids)
        failed=er.validate_table_coverage(out,ids,wire_inventory=out['expected_list_items'])
        self.assertEqual(failed['reason'],'source_table_inventory_incomplete')
        reply['table_record_coverage']=['T3!different occurrence outside request','T4!different occurrence outside request']
        out=st.resolve_reply(reply,catalog,admitted_citation_ids={e.citation_id for e in catalog.entries})
        checked=er.validate_table_coverage(out,ids,wire_inventory=out['expected_list_items'])
        self.assertEqual(checked['outcome'],'rewrite')
        self.assertEqual({r['record_id']for r in checked['table_record_coverage']},set(ids))
        self.assertEqual(checked['citation_ids'],[catalog.entries[0].citation_id])

    def test_schema_omission_is_catalog_conditional_and_other_constraints_unchanged(self):
        helper=pure.SourceTableReviewTests();catalog=helper.catalog()
        schema=er.source_first_schema(self.m._assistant_core_contract_verifier_schema(),record_ids=('T1','T2'))
        self.assertIs(st.reference_schema(schema,st.TableCatalog()),schema)
        new=st.reference_schema(schema,catalog);restored=deepcopy(new)
        restored['schema']['properties']['table_record_coverage']=deepcopy(schema['schema']['properties']['table_record_coverage'])
        self.assertEqual(restored,schema)
        self.assertEqual(schema['schema']['properties']['table_record_coverage']['minItems'],2)
        self.assertEqual(new['schema']['properties']['table_record_coverage']['maxItems'],2)

    def test_selected_owner_still_requires_current_scope_and_explicit_rows_stay_literal(self):
        helper=pure.SourceTableReviewTests();catalog=helper.catalog();reply=helper.result(catalog)
        reply['citation_ids']=[]
        diag={};out=st.resolve_reply(reply,catalog,admitted_citation_ids={'other:p1'},diagnostics=diag)
        self.assertEqual(out['outcome'],'no_sources');self.assertEqual(diag['source_table_gate'],'source_scope_invalid')
        reply['table_record_coverage']=['T1=00','T2=0']
        out=st.resolve_reply(reply,catalog,admitted_citation_ids={pure.CID})
        self.assertEqual(out['table_record_coverage'],reply['table_record_coverage'])
        checked=er.validate_table_coverage(out,('T1','T2'),wire_inventory=out['expected_list_items'])
        self.assertEqual(checked['outcome'],'no_sources')

    def test_diagnostics_are_static_debug_only_and_cannot_copy_provider_text(self):
        for debug in (True,False,'true',1):
            owner,f=self.case();owner.payload.debug=debug
            diagnostic={'source_table_gate':'private arbitrary source text','source_table_selected_count':1000,
                        'source_table_bound_record_count':True,'answer':'PRIVATE_SOURCE','citation_ids':['private:cid']}
            with patch('sys.stdout',new=io.StringIO())as logs:
                completion.record_contract_observation('semantic_review',{'outcome':'no_sources',
                    'reason':'source_table_reference_invalid'},diagnostics=diagnostic)
            observed=owner.contract_checks[-1]
            terminal=owner.finish({'ok':False,'status':'error','result_code':'TECHNICAL_ERROR','meta':{}})
            encoded=json.dumps(terminal)+logs.getvalue()
            for text in ('private arbitrary source text','PRIVATE_SOURCE','private:cid'):self.assertNotIn(text,encoded)
            if debug is True:
                self.assertEqual(observed['source_table_gate'],'other')
                self.assertEqual(observed['source_table_selected_count'],100)
                self.assertNotIn('source_table_bound_record_count',observed)
            else:
                self.assertNotIn('source_table_gate',observed);self.assertEqual(logs.getvalue(),'')
            f.close();owner.close();self.assertEqual(owner.contract_checks,[])


if __name__=='__main__':unittest.main(verbosity=2)
