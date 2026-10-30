"""Bounded failure telemetry through real validation; no external I/O.

The shared fixture supplies writer/reviewer replies and isolates source authority.
The claim filter, final contract, validator and completion observer are real.
These tests certify diagnostics and unchanged rejection, not model quality.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_ask_source_identifiers as source_fixture
from machinemind.ask import request_completion


class FinalContractDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_fixture.AskSourceIdentifierTests.setUpClass()
        cls.m=source_fixture.AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        source_fixture.AskSourceIdentifierTests.tearDownClass()

    def owner(self,debug=False):
        return request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=debug),started=time.monotonic())

    def pipeline(self,text,*,debug=False,terminal_error=False):
        owner=self.owner(debug)
        try:
            result,calls,_=source_fixture.AskSourceIdentifierTests().run_pipeline(text)
            original=deepcopy(result)
            terminal=({'ok':False,'status':'budget_exceeded','result_code':'BUDGET_EXCEEDED','meta':{}}
                if terminal_error else result)
            finished=owner.finish(terminal)
            observations=finished['meta']['request_completion']['contract_checks']
            return original,finished,observations,calls
        finally:
            owner.close()

    def test_removed_claim_explains_real_final_rejection_without_exposing_claim(self):
        text="Il documento indica il grado QRT-7.\nLa frequenza documentata e' di 600 ore."
        result,_,observations,calls=self.pipeline(text,debug=True)
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'],'no_sources')
        self.assertEqual(len(calls),2)
        self.assertFalse(contract['passed'])
        final=observations[-1]
        self.assertEqual(final['reason'],'semantic_answer_changed_after_review')
        self.assertFalse(final['semantic_answer_bound'])
        self.assertFalse(final['complete'])
        self.assertTrue(final['post_review_grounding_applied'])
        self.assertEqual(final['post_review_removed_claims_count'],1)
        self.assertEqual(final['admitted_source_count'],1)
        self.assertEqual(final['final_citation_count'],1)
        self.assertNotEqual(final['reviewed_answer_sha256'],final['final_answer_sha256'])
        self.assertNotIn('QRT-7',json.dumps(observations))
        self.assertNotIn('QTR-7',json.dumps(observations))
        self.assertNotIn('Il documento',json.dumps(observations))

    def test_supported_rewrite_keeps_pass_and_zero_removal(self):
        text="Il documento indica il grado QTR-7.\nLa frequenza documentata e' di 600 ore."
        result,_,observations,_=self.pipeline(text,debug=True)
        self.assertEqual(result['status'],'answered')
        final=observations[-1]
        self.assertEqual(final['reason'],'semantic_contract_complete')
        self.assertTrue(final['semantic_answer_bound'])
        self.assertTrue(final['complete'])
        self.assertEqual(final['post_review_removed_claims_count'],0)
        self.assertEqual(final['reviewed_answer_sha256'],final['final_answer_sha256'])

    def test_prior_diagnostics_survive_later_budget_error_without_rejected_draft(self):
        result,finished,observations,_=self.pipeline(
            "Il documento indica il grado QRT-7.\nLa frequenza documentata e' di 600 ore.",
            debug=True,terminal_error=True)
        self.assertEqual(result['status'],'no_sources')
        self.assertEqual(finished['result_code'],'BUDGET_EXCEEDED')
        self.assertEqual(observations[-1]['reason'],'semantic_answer_changed_after_review')
        self.assertNotIn('answer',finished)
        self.assertNotIn('QRT-7',json.dumps(finished))

    def test_nondebug_malformed_and_untrusted_fields_remain_bounded(self):
        owner=self.owner(False)
        try:
            contract={'passed':False,'reason':'PRIVATE_REASON','semantic_answer_bound':'PRIVATE_BOOL',
                'complete':1,'answer_sha256':'a'*64,'semantic_verifier':{'answer':'PRIVATE_ANSWER'},
                'source_preservation':{'all_present':'PRIVATE_BOOL'}}
            diagnostics={'admitted_source_count':True,'final_citation_count':-1,
                'post_review_removed_claims_count':10000,'post_review_grounding_applied':'PRIVATE_BOOL',
                'answer':'PRIVATE_ANSWER','source':'PRIVATE_SOURCE'}
            before=deepcopy((contract,diagnostics))
            request_completion.record_contract_observation('final_contract',contract,diagnostics=diagnostics)
            for _ in range(20):
                request_completion.record_contract_observation('final_contract',{'passed':False})
            finished=owner.finish({'ok':False,'status':'error','meta':{}})
            rows=finished['meta']['request_completion']['contract_checks']
            self.assertEqual(len(rows),8)
            row=rows[0]
            self.assertEqual(row['reason'],'other')
            self.assertEqual(row.get('post_review_removed_claims_count'),100)
            for key in ('admitted_source_count','final_citation_count','post_review_grounding_applied',
                    'semantic_answer_bound','complete','source_notes_all_present',
                    'reviewed_answer_sha256','final_answer_sha256'):
                self.assertNotIn(key,row)
            self.assertNotIn('PRIVATE_',json.dumps(rows))
            self.assertEqual((contract,diagnostics),before)
        finally:
            owner.close()

    def test_known_final_reasons_and_only_valid_debug_hashes_are_preserved(self):
        owner=self.owner(True)
        try:
            reasons=('semantic_answer_changed_after_review','semantic_contract_incomplete',
                'source_owned_content_missing_or_changed','reviewed_draft_binding_invalid',
                'source_step_structure_unverifiable','supported_list_inventory_missing')
            for reason in reasons:
                request_completion.record_contract_observation('final_contract',{
                    'passed':False,'reason':reason,'answer_sha256':'PRIVATE_INVALID_HASH',
                    'semantic_verifier':{'answer':'PRIVATE_ANSWER'},
                    'source_preservation':{'all_present':False}})
            rows=owner.finish({'ok':False,'status':'error','meta':{}})['meta']['request_completion']['contract_checks']
            self.assertEqual([row['reason'] for row in rows],list(reasons))
            for row in rows:
                self.assertFalse(row['source_notes_all_present'])
                self.assertNotIn('final_answer_sha256',row)
                self.assertEqual(row['reviewed_answer_sha256'],hashlib.sha256(b'PRIVATE_ANSWER').hexdigest())
            self.assertNotIn('PRIVATE_',json.dumps(rows))
        finally:
            owner.close()


if __name__=='__main__':
    unittest.main(verbosity=2)
