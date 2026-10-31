"""Explanatory mentions are data, never selections or publication instructions.

Database/model fixtures exercise real owned gates. The failed live field was
not retained, so these regressions do not establish its precise cause.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_source_table_review as pure
import test_source_table_owned as owned
from machinemind.ask import source_table_review as st,enumeration_review as er


MENTION='Explanation only: [[SOURCE_TABLE:G1]] and [[SOURCE_TABLE:G999]] are mentioned, not selected.'


class SourceTableMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass();cls.m=owned.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()

    case=owned.SourceTableOwnedTests.case

    def test_reason_and_exclusion_mentions_survive_owned_binding_without_rendering_or_authority(self):
        condition=owned.CONDITION
        second='- [PDF_TABLE_RECORD rows=1 columns=1] c1 r1="Separate assembly outside the requested inspection scope."'
        for metadata in ('reason','exclusion','both'):
            with self.subTest(metadata=metadata):
                def change(reply):
                    reply['expected_list_items']=['[[SOURCE_TABLE:G1]]',condition]
                    reply['table_record_coverage']=['T1=0','T2=0','T3!'+(MENTION if metadata!='reason' else 'Separate assembly')]
                    if metadata!='exclusion':reply['reason']=MENTION
                with patch.object(owned,'TABLE_BODY',owned.TABLE_BODY+'\n'+condition),patch.object(owned,'CONDITION',second):
                    owner,f=self.case(change=change)
                owner.payload.debug=True
                with patch('sys.stdout',new=io.StringIO())as logs:result=f.run()
                self.assertEqual(result['status'],'answered');self.assertEqual(len(f.posts),2)
                self.assertEqual(f.writer_calls,[])
                self.assertNotIn('[[SOURCE_TABLE:',result['answer']);self.assertNotIn('[[SOURCE_TABLE:',result['answer_html'])
                self.assertNotIn(MENTION,logs.getvalue())
                self.assertNotIn(MENTION,f.reviewer_calls[1][0][1]['content'])
                contract=f.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['semantic_answer_bound'])
                self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
                coverage=contract['semantic_verifier']['table_record_coverage']
                self.assertEqual([r['inventory_items']for r in coverage],[[0],[0],[]])
                self.assertEqual(coverage[-1]['outside_request_reason'],MENTION if metadata!='reason'else'Separate assembly')
                for text in (condition.rstrip('.'),'Inspect wiring integrity','Monthly'):
                    self.assertIn(text,result['answer'])
                    self.assertIn(text,self.m._assistant_ui_visible_text_from_html(result['answer_html']))
                self.assertEqual({c['citation_id']for c in result['citations']},{'inspection:p1','inspection:p2'})
                self.assertTrue(f.owner._response_flow.completed)
                f.close();owner.close()

    def test_metadata_only_mentions_do_not_activate_or_select_a_catalog_entry(self):
        helper=pure.SourceTableReviewTests();catalog=helper.catalog()
        for outcome in ('pass','no_sources'):
            result={'outcome':outcome,'answer':'Ordinary unchanged answer'if outcome=='pass'else'',
                    'reason':MENTION,'citation_ids':[], 'expected_list_items':['Ordinary item'],
                    'table_record_coverage':['T1!'+MENTION,'T2!Not requested']}
            before=deepcopy(result);diag={}
            self.assertIs(st.resolve_reply(result,catalog,admitted_citation_ids={pure.CID},diagnostics=diag),result)
            self.assertEqual(result,before);self.assertEqual(diag,{'source_table_gate':'unused'})
            self.assertEqual(result['citation_ids'],[])

    def test_operational_and_unknown_fields_still_reject_markers_before_second_provider(self):
        mutations=[{'answer':'[[FACTUAL_INVENTORY]]\n'+MENTION},
                   {'citation_ids':[MENTION]}, {'covered_facets':[MENTION]},
                   {'missing_facets':[MENTION]}, {'covered_list_items':[MENTION]},
                   {'missing_list_items':[MENTION]}, {'covered_answer_types':[MENTION]},
                   {'unknown':{'reason':MENTION}}, {'reason':[MENTION]}]
        for mutation in mutations:
            with self.subTest(field=next(iter(mutation))):
                owner,f=self.case(change=lambda r:r.update(deepcopy(mutation)));owner.payload.debug=True
                with patch('sys.stdout',new=io.StringIO()):result=f.run()
                self.assertEqual(result['status'],'no_sources');self.assertEqual(len(f.posts),1)
                row=next(x for x in owner.contract_checks if x['stage']=='semantic_review')
                self.assertEqual(row['source_table_gate'],'marker_outside_inventory')
                f.close();owner.close()

    def test_exclusions_keep_strict_grammar_and_selected_record_conflicts(self):
        helper=pure.SourceTableReviewTests();catalog=helper.catalog();valid=helper.result(catalog)
        for row in ('T1!'+MENTION,'T1=0!'+MENTION,'T1!\n'+MENTION,'T01!'+MENTION,'T1!'+MENTION+'x'*512):
            result={**deepcopy(valid),'table_record_coverage':[row,'T2=0']};diag={}
            out=st.resolve_reply(result,catalog,diagnostics=diag)
            self.assertEqual(out['outcome'],'no_sources')
            self.assertEqual(diag['source_table_gate'],'selected_record_conflict'if row=='T1!'+MENTION else'marker_outside_inventory')
        # Explanatory exemption is not a replacement for complete classification.
        for rows in (['T1!'+MENTION],['T1!'+MENTION,'T1!'+MENTION],['T1!'+MENTION,'T99!'+MENTION]):
            result={'outcome':'rewrite','answer':'Documented fact','expected_list_items':['Documented fact'],
                    'citation_ids':[pure.CID],'table_record_coverage':rows,'reason':MENTION}
            decoded=st.resolve_reply(result,catalog)
            checked=er.validate_table_coverage(decoded,('T1','T2'),wire_inventory=decoded['expected_list_items'])
            self.assertEqual(checked['outcome'],'no_sources')
            self.assertEqual(checked['reason'],'source_table_inventory_incomplete')


if __name__=='__main__':unittest.main(verbosity=2)
