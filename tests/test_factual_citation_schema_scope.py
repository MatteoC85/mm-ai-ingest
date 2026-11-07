"""Source-first wire constraints; fixture replies are not model-quality claims."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_useful_table_selection as useful
import test_source_ordered_review as procedure
from machinemind.evidence.contracts import EvidenceContractError

class FactualCitationScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        useful.UsefulTableOwnedTests.setUpClass()
    @classmethod
    def tearDownClass(cls):
        useful.UsefulTableOwnedTests.tearDownClass()

    def test_actual_source_first_schema_has_exact_admitted_ids_including_plain_context(self):
        helper=useful.UsefulTableOwnedTests()
        with ExitStack() as cleanup:
            completion,f=helper.fixture(cleanup)
            before=deepcopy(f.prepared)
            deadline=completion.budget.deadline_monotonic
            cap=completion.budget.max_estimated_cost_usd
            result=f.run()
            self.assertEqual(result['status'],'answered')
            self.assertEqual(len(f.posts),2)  # Producer plus mandatory independent review.
            first=f.posts[0]['text']['format']['schema']['properties']['citation_ids']
            self.assertEqual(first,{'type':'array','items':{'type':'string','enum':['inspection:p1','inspection:p2']},'maxItems':18})
            self.assertEqual(f.prepared,before)
            self.assertIn(useful.owned.TABLE_QUALIFIER,result['answer'])
            self.assertIn(useful.owned.CONDITION,result['answer'])
            self.assertEqual((completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd),(deadline,cap))
            self.assertEqual(f.reviewer_calls[0][1]['max_output_tokens'],3200)
            self.assertEqual(f.writer_calls,[])
            second=f.posts[1]['text']['format']['schema']['properties']
            self.assertNotIn('citation_ids',second)
            self.assertIn('additional_citation_ids',second)
            self.assertNotIn('enum',helper.m._assistant_core_contract_verifier_schema()['schema']['properties']['citation_ids']['items'])

    def test_request_scopes_do_not_share_enums_or_acquire_foreign_ids(self):
        helper=useful.UsefulTableOwnedTests(); enums=[]
        for company,machine,doc in [('offline-cobalt','mill','manual-z'),('offline-ochre','pump','manual-a')]:
            with ExitStack() as cleanup:
                _,f=helper.fixture(cleanup,company_id=company,machine_id=machine,document_id=doc)
                self.assertEqual(f.run()['status'],'answered')
                field=f.posts[0]['text']['format']['schema']['properties']['citation_ids']['items']
                self.assertEqual(field['enum'],[doc+':p1',doc+':p2'])
                enums.append(field['enum'])
        self.assertFalse(set(enums[0])&set(enums[1]))

    def test_malicious_nonconforming_provider_reply_is_still_rejected_not_filtered(self):
        helper=useful.UsefulTableOwnedTests()
        for bad in ['foreign-company:p1','S0','inspection','inspection:p1-extra']:
            with self.subTest(bad=bad),ExitStack() as cleanup:
                _,f=helper.fixture(cleanup,change=lambda reply:reply.update(citation_ids=['inspection:p1',bad]))
                result=f.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(len(f.posts),1)
                self.assertIn(bad,f.provider_wire_reply['citation_ids'])
                self.assertEqual(result.get('citations') or [],[])
                self.assertFalse(f.reviewer_calls[0][1]['json_schema']['schema']['properties']['citation_ids']['items']['enum'].count(bad))

    def test_schema_membership_is_not_semantic_approval(self):
        helper=useful.UsefulTableOwnedTests()
        for check in ('source_fidelity','scope_applicability','provenance','completeness','safety','language'):
            with self.subTest(check=check),ExitStack() as cleanup:
                _,f=helper.fixture(cleanup,second_change=lambda reply:reply['checks'].update({check:False}))
                result=f.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(len(f.posts),2)
                self.assertEqual(result.get('citations') or [],[])
        for change in (lambda reply:reply.update(outcome='no_sources'),lambda reply:reply.update(missing_facets=['necessary condition'])):
            with ExitStack() as cleanup:
                _,f=helper.fixture(cleanup,change=change)
                self.assertEqual(f.run()['status'],'no_sources')
                self.assertEqual(len(f.posts),1)

    def test_revoke_or_source_change_cannot_reuse_admitted_schema_as_authority(self):
        helper=useful.UsefulTableOwnedTests()
        for args in ({'revoke':True},{'mutate':True}):
            with self.subTest(args=args),ExitStack() as cleanup:
                _,f=helper.fixture(cleanup,**args)
                with self.assertRaises(EvidenceContractError):f.run()
                self.assertFalse(f.owner._response_flow.completed)

class ProcedureUnchangedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):procedure.WiredSourceOrderedReviewTests.setUpClass()
    @classmethod
    def tearDownClass(cls):procedure.WiredSourceOrderedReviewTests.tearDownClass()
    def test_source_ordered_procedure_uses_unchanged_reference_schema_and_one_review(self):
        result,calls,_=procedure.WiredSourceOrderedReviewTests().run_review(family_required=True,reply_mode='retain')
        self.assertEqual(len(calls),1)
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])
        self.assertEqual(calls[0][1]['max_output_tokens'],3200)
        self.assertEqual(calls[0][1]['json_schema']['schema']['properties']['citation_ids'],
            {'type':'array','items':{'type':'string','enum':['s1','s2','s3']},'maxItems':18})

if __name__=='__main__':unittest.main(verbosity=2)
