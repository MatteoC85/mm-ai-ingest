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
    def test_independent_gate_is_debug_only_bounded_and_content_free_after_terminal_failure(self):
        for debug in (True,False,'true'):
            owner=self.owner(debug)
            try:
                request_completion.record_contract_observation('semantic_review',
                    {'outcome':'unavailable','reason':'factual_context_review_invalid'},diagnostics={
                        'independent_review_outcome':'invalid','independent_review_gate':'quote_not_literal',
                        'independent_review_edit_count':1000,'raw_quote':'PRIVATE SOURCE PHRASE',
                        'answer':'PRIVATE CANDIDATE'})
                terminal=owner.finish({'ok':False,'status':'error','result_code':'TECHNICAL_ERROR','meta':{}})
                row=terminal['meta']['request_completion']['contract_checks'][-1]
                self.assertEqual(row['reason'],'factual_context_review_invalid')
                self.assertEqual(row['outcome'],'unavailable')
                if debug is True:
                    self.assertEqual(row['independent_review_gate'],'quote_not_literal')
                    self.assertEqual(row['independent_review_edit_count'],100)
                else:
                    self.assertNotIn('independent_review_gate',row)
                    self.assertNotIn('independent_review_outcome',row)
                self.assertNotIn('PRIVATE',json.dumps(terminal))
            finally:owner.close()
        owner=self.owner(True)
        try:
            self.assertEqual(owner.contract_checks,[])
            request_completion.record_contract_observation('semantic_review',{'outcome':'no_sources',
                'reason':'factual_context_review_rejected'},diagnostics={
                    'independent_review_outcome':'PRIVATE SOURCE PHRASE',
                    'independent_review_gate':'PRIVATE MODEL ERROR'})
            self.assertEqual(owner.contract_checks[-1]['independent_review_gate'],'other')
            self.assertEqual(owner.contract_checks[-1]['independent_review_outcome'],'other')
            self.assertNotIn('PRIVATE',json.dumps(owner.contract_checks))
        finally:owner.close()

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
        self.assertEqual(final.get('removed_claim_sha256'), [hashlib.sha256(b'QRT-7').hexdigest()])
        expected_source = ('Documento del costruttore: grado QTR-7. '
            'Frequenza documentata: ogni 600 ore. Non sono elencate alternative equivalenti.')
        self.assertEqual(final.get('source_packet_sha256'), hashlib.sha256(expected_source.encode()).hexdigest())

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
        self.assertEqual(observations[-1].get('removed_claim_sha256'), [hashlib.sha256(b'QRT-7').hexdigest()])
        self.assertRegex(observations[-1]['source_packet_sha256'], r'^[0-9a-f]{64}$')

    def test_nondebug_malformed_and_untrusted_fields_remain_bounded(self):
        owner=self.owner(False)
        try:
            contract={'passed':False,'reason':'PRIVATE_REASON','semantic_answer_bound':'PRIVATE_BOOL',
                'complete':1,'answer_sha256':'a'*64,'semantic_verifier':{'answer':'PRIVATE_ANSWER'},
                'source_preservation':{'all_present':'PRIVATE_BOOL'}}
            diagnostics={'admitted_source_count':True,'final_citation_count':-1,
                'post_review_removed_claims_count':10000,'post_review_grounding_applied':'PRIVATE_BOOL',
                'answer':'PRIVATE_ANSWER','source':'PRIVATE_SOURCE',
                'source_text':'PRIVATE_SOURCE', 'post_review_removed_claims':['PRIVATE_CLAIM']}
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
                    'reviewed_answer_sha256','final_answer_sha256',
                    'source_packet_sha256','removed_claim_sha256'):
                self.assertNotIn(key,row)
            self.assertNotIn('PRIVATE_',json.dumps(rows))
            self.assertEqual((contract,diagnostics),before)
        finally:
            owner.close()

    def test_debug_hashes_are_bounded_typed_and_request_local(self):
        owner=self.owner(True)
        source='PRIVATE_SOURCE_' + 'é'
        claims=['PRIVATE_CLAIM_0', 'PRIVATE_CLAIM_0', None, 17, True, {'claim':'PRIVATE'},
                'X'*513, '', *[f'PRIVATE_CLAIM_{i}' for i in range(1,30)]]
        before=deepcopy(claims)
        try:
            request_completion.record_contract_observation('final_contract',
                {'passed':False}, diagnostics={'source_text':source,'post_review_removed_claims':claims})
            row=owner.contract_checks[-1]
            expected=list(dict.fromkeys(hashlib.sha256(s.encode()).hexdigest()
                for s in claims[:16] if type(s) is str and 0<len(s)<=512))
            self.assertEqual(row.get('removed_claim_sha256'),expected)
            self.assertEqual(row['source_packet_sha256'],hashlib.sha256(source.encode()).hexdigest())
            self.assertNotIn('PRIVATE',json.dumps(owner.contract_checks))
            self.assertEqual(claims,before)
            request_completion.record_contract_observation('final_contract',{'passed':False},
                diagnostics={'source_text':'S'*262145,'post_review_removed_claims':'PRIVATE_STRING'})
            self.assertNotIn('source_packet_sha256',owner.contract_checks[-1])
            self.assertNotIn('removed_claim_sha256',owner.contract_checks[-1])
            published=owner.finish({'ok':False,'status':'budget_exceeded','meta':{}})
        finally:
            owner.close()
        self.assertEqual(owner.contract_checks,[])
        self.assertIsNone(request_completion.current())
        # Clearing owner RAM does not erase the already returned content-free receipt.
        self.assertEqual(published['meta']['request_completion']['contract_checks'][0],row)
        next_owner=self.owner(True)
        try:
            self.assertEqual(next_owner.contract_checks,[])
            request_completion.record_contract_observation('final_contract',{'passed':False},
                diagnostics={'source_text':'NEXT_SOURCE','post_review_removed_claims':['NEXT_CLAIM']})
            next_row=next_owner.contract_checks[-1]
            self.assertNotEqual(next_row['source_packet_sha256'],row['source_packet_sha256'])
            self.assertEqual(next_row['removed_claim_sha256'],[hashlib.sha256(b'NEXT_CLAIM').hexdigest()])
        finally:
            next_owner.close()

    def test_debug_requires_literal_true_and_no_active_request_captures_nothing(self):
        self.assertIsNone(request_completion.current())
        request_completion.record_contract_observation('final_contract',{'passed':False},
            diagnostics={'source_text':'PRIVATE_SOURCE','post_review_removed_claims':['PRIVATE_CLAIM']})
        for value in (False, None, 1, 'true'):
            owner=self.owner(value)
            try:
                request_completion.record_contract_observation('final_contract',{'passed':False},
                    diagnostics={'source_text':'PRIVATE_SOURCE','post_review_removed_claims':['PRIVATE_CLAIM']})
                row=owner.contract_checks[-1]
                self.assertNotIn('source_packet_sha256',row)
                self.assertNotIn('removed_claim_sha256',row)
                self.assertNotIn('PRIVATE',json.dumps(row))
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
