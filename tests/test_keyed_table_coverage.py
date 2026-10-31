"""Keyed classifications preserve canonical coverage and owned publication.

Source observations/model replies are fixtures, not semantic model quality.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_source_table_owned as owned
import test_source_table_review as pure
import test_duplicate_table_context as context
from machinemind.ask import source_table_review as st,enumeration_review as er
from machinemind.evidence.contracts import EvidenceContractError


class KeyedTableCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):owned.SourceTableOwnedTests.setUpClass();cls.m=owned.SourceTableOwnedTests.m
    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()
    case=owned.SourceTableOwnedTests.case

    def catalog(self,different=False):
        return pure.SourceTableReviewTests().catalog([
            pure.row('doc:p1',context.SCOPE+'\n'+context.CELLS),
            pure.row('doc:p2',context.QUALIFIER+'\n'+(context.CELLS.replace('Periodic','Annual')if different else context.CELLS))])

    def decode(self,keyed,*,different=False,**changes):
        catalog=self.catalog(different)
        raw=dict(outcome='rewrite',answer='[[FACTUAL_INVENTORY]]',expected_list_items=[catalog.entries[0].token],
            citation_ids=['doc:p1'],table_record_coverage=keyed)
        raw.update(changes);snapshot=deepcopy(raw);diagnostic={}
        out=st.resolve_reply(raw,catalog,admitted_citation_ids={'doc:p1','doc:p2'},derive_duplicate_records=True,
            keyed_coverage=True,diagnostics=diagnostic)
        self.assertEqual(raw,snapshot)
        checked=er.validate_table_coverage(out,('T1','T2'),wire_inventory=out['expected_list_items'])
        return out,checked,diagnostic

    def test_key_order_is_irrelevant_and_every_occurrence_is_explicit(self):
        first,checked,_=self.decode({'T1':'=0','T2':'!Exact duplicate table; distinct context reviewed'})
        second,other,_=self.decode({'T2':'!Exact duplicate table; distinct context reviewed','T1':'=0'})
        self.assertEqual(first,second);self.assertEqual(checked,other);self.assertEqual(checked['outcome'],'rewrite')
        self.assertEqual(first['citation_ids'],['doc:p1'])
        self.assertEqual([row['inventory_items']for row in checked['table_record_coverage']],[[0],[]])
        self.assertEqual(self.decode({'T1':None,'T2':None})[1]['outcome'],'no_sources')
        self.assertEqual(self.decode({'T1':None,'T2':None},different=True)[1]['outcome'],'no_sources')
        self.assertEqual(self.decode({'T1':'=0','T2':'!Different documented table'},different=True)[1]['outcome'],'rewrite')
        limit='!'+'x'*(512-len('T2')-1)
        self.assertEqual(self.decode({'T1':'=0','T2':limit},different=True)[1]['outcome'],'rewrite')
        self.assertEqual(self.decode({'T1':'=0','T2':limit+'x'},different=True)[1]['outcome'],'no_sources')

    def test_shape_and_explicit_contradictions_are_never_treated_as_omissions(self):
        cases=[{}, {'T1':None}, {'T1':None,'T2':None,'T99':None},
            {'T1':False,'T2':None}, {'T1':[0],'T2':None}, ['T1=0','T2!outside'],
            {'T1':'=0','T2':None}, {'T1':None,'T2':'!other'},
            {'T1':'=00','T2':'!other'}, {'T1':'=0,0','T2':'!other'}, {'T1':'=1','T2':'!other'},
            {'T1':'!Excluded selected table','T2':'!other'}, {'T1':'=0','T2':'=0'},
            {'T1':'=0','T2':'!   '}, {'T1':'=0','T2':'!reason\nextra'},
            {'T1':'=0','T2':''}, {'T1':'=0','T2':'!'}, {'T1':'=0','T2':'='}]
        for value in cases:
            with self.subTest(value=value):self.assertEqual(self.decode(value)[1]['outcome'],'no_sources')

    def test_inert_reason_markers_do_not_select_but_operational_markers_still_fail(self):
        mention='Other occurrence [[SOURCE_TABLE:G2]] has its own context'
        out,checked,_=self.decode({'T1':'=0','T2':'!'+mention},reason=mention)
        self.assertEqual(checked['outcome'],'rewrite');self.assertNotIn(mention,out['expected_list_items'][0])
        self.assertEqual(out['citation_ids'],['doc:p1'])
        self.assertEqual(self.decode({'T1':'=0','T2':'!Other occurrence'},answer='[[FACTUAL_INVENTORY]]\n[[SOURCE_TABLE:G2]]')[2]['source_table_gate'],'marker_outside_inventory')

    def test_schema_is_scoped_and_all_keys_required_without_new_limits(self):
        base=er.source_first_schema(self.m._assistant_core_contract_verifier_schema(),record_ids=('T1','T2'))
        catalog=self.catalog();old=st.reference_schema(base,catalog,derive_duplicate_records=True)
        new=st.reference_schema(base,catalog,derive_duplicate_records=True,keyed_coverage=True)
        field=new['schema']['properties']['table_record_coverage']
        self.assertEqual(field['required'],['T1','T2']);self.assertFalse(field['additionalProperties'])
        self.assertEqual(set(field['properties']),{'T1','T2'});self.assertIs(new['reject_duplicate_keys'],True)
        for rid,value in field['properties'].items():
            self.assertEqual(value['maxLength'],512-len(rid));self.assertEqual(value['type'],'string')
        self.assertEqual(old['schema']['properties']['table_record_coverage']['type'],'array')
        self.assertNotIn('reject_duplicate_keys',old)
        restored=deepcopy(new);restored.pop('reject_duplicate_keys')
        restored['schema']['properties']['table_record_coverage']=old['schema']['properties']['table_record_coverage']
        self.assertEqual(restored,old)
        self.assertEqual(st.reference_schema(base,st.TableCatalog(),keyed_coverage=True),base)

    def test_owned_keyed_wire_reaches_two_reviews_real_hash_and_html(self):
        owner,f=self.case(change=lambda reply:reply.update(table_record_coverage={'T1':'=0','T2':'=0'}))
        try:
            result=f.run();self.assertEqual(result['status'],'answered')
            contract=f.validated['meta']['assistant_core_validation']['answer_contract']
            self.assertTrue(contract['semantic_answer_bound'])
            self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
            self.assertIn(owned.TABLE_QUALIFIER.rstrip('.'),result['answer_html'])
            self.assertEqual(len(f.posts),2);self.assertEqual(f.writer_calls,[])
            self.assertEqual(f.budget_ids,[id(owner.budget)]*2)
            self.assertEqual(f.wire_reply['table_record_coverage'],{'T1':'=0','T2':'=0'})
            self.assertNotIn('reject_duplicate_keys',str(f.posts))
            emitted=f.posts[0]['text']['format']
            self.assertIs(emitted['strict'],True)
            coverage=emitted['schema']['properties']['table_record_coverage']
            self.assertEqual(coverage['type'],'object');self.assertFalse(coverage['additionalProperties'])
            self.assertEqual(set(coverage['required']),{'T1','T2'})
            # Assert the schema actually sent to the provider forbids null;
            # server-side rejection alone would not prevent the R55 failure.
            self.assertEqual({rid:field['type']for rid,field in coverage['properties'].items()},
                             {'T1':'string','T2':'string'})
            for rid,field in coverage['properties'].items():
                self.assertEqual(field['minLength'],2)
                self.assertEqual(field['maxLength'],512-len(rid))
                self.assertEqual(field['pattern'],r'^(?:=[0-9]+(?:,[0-9]+)*|![^\r\n]+)$')
            self.assertIn('Explicitly map every selected record',f.posts[0]['input'][0]['content'])
            self.assertIn('non-null suffix string for each key',f.posts[0]['input'][0]['content'])
            self.assertNotIn('Set these selected record values to null',f.posts[0]['input'][0]['content'])
            self.assertEqual(f.posts[0]['input'],f.reviewer_calls[0][0])
        finally:f.close();owner.close()

    def test_owned_rejection_and_source_revocation_cannot_skip_final_authority(self):
        for change in (
            lambda reply:reply.update(table_record_coverage={'T1':None}),
            lambda reply:reply.update(table_record_coverage={'T1':None,'T2':None}),
            lambda reply:reply.update(table_record_coverage={'T1':'=0','T2':None}),
            lambda reply:reply.update(citation_ids=['foreign:p1']),
            lambda reply:reply.update(outcome='no_sources',answer='',table_record_coverage={'T1':None,'T2':None})):
            owner,f=self.case(change=change)
            try:
                result=f.run();self.assertEqual(result['status'],'no_sources');self.assertEqual(result.get('citations')or[],[])
                self.assertEqual(len(f.posts),1)
            finally:f.close();owner.close()
        owner,f=self.case(revoke=True,change=lambda reply:reply.update(table_record_coverage={'T1':'=0','T2':'=0'}))
        try:
            with self.assertRaises(EvidenceContractError):f.run()
            self.assertEqual(len(f.posts),1)
        finally:f.close();owner.close()

    def test_useful_limited_owned_answer_still_requires_independent_acceptance_and_no_declared_gaps(self):
        limitation='The supplied excerpts support the following information; this is not an exhaustive list of the entire manual.'
        for missing in (False,True):
            def change(reply):
                reply.update(answer=limitation+'\n[[FACTUAL_INVENTORY]]',table_record_coverage={'T1':'=0','T2':'=0'})
                if missing:reply['missing_facets']=['documented obligations']
            owner,f=self.case(change=change)
            try:
                result=f.run()
                if missing:
                    self.assertEqual(result['status'],'no_sources');self.assertEqual(len(f.posts),1)
                else:
                    self.assertEqual(result['status'],'answered');self.assertEqual(len(f.posts),2)
                    self.assertIn(limitation.rstrip('.'),result['answer'])
                    self.assertTrue(f.validated['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
                    prompt=f.reviewer_calls[0][0][0]['content']
                    self.assertIn('A useful inventory may be explicitly limited',prompt)
                    self.assertNotIn('identify the complete set',prompt)
                    self.assertNotIn('or supported relevant rows, rewrite',prompt)
                    self.assertNotIn('Preserve all supported values',prompt)
                    self.assertNotIn('not permission to omit an explicitly documented',prompt)
                    self.assertIn('Verify each complete source relationship',prompt)
                    self.assertIn('PPE or closing guards afterwards does not replace an access prerequisite',prompt)
                    self.assertIn('Every FACET_CONTRACT marked must_cover=true is mandatory',prompt)
            finally:f.close();owner.close()


if __name__=='__main__':unittest.main(verbosity=2)
