"""Actual Smart handlers: rejected updates close without adopting their claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_smart_review import SmartReviewEndpointTests, parsed_review, wire_review, denied
from test_smart_continuity_endpoints import direct_post


class SmartProtocolInconclusiveTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    post = direct_post
    start = SmartReviewEndpointTests.start
    _fixture_setup = SmartReviewEndpointTests.setUp

    def setUp(self):
        self._fixture_setup()
        self.raw['question'].update(question_type='yes_no', options=[
            {'id': 'yes', 'label_en': 'Yes', 'label_it': 'Sì'},
            {'id': 'no', 'label_en': 'No', 'label_it': 'No'},
            {'id': 'unknown', 'label_en': "I don't know", 'label_it': 'Non so'}])

    def reject_all(self, *args, **kwargs):
        return wire_review(parsed_review(self.prepared[-1], rejected=(0, 1, 2, 3), reject_question=True)), 'offline-review'

    def rejected_update(self, start, *, value='no', debug=False):
        proposal = deepcopy(self.raw)
        proposal['question']['question_text'] = 'UNSUPPORTED_NEW_OPERATION_MOVE_MACHINE?'
        proposal['operator_summary'] = 'UNSUPPORTED_RAW_SUMMARY'
        proposal['hypotheses'][0]['label'] = 'UNSUPPORTED_NEW_CAUSE'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=proposal), \
             patch.object(self.m, '_v13_json_models', side_effect=self.reject_all):
            return self.post('answer', start['session_state_json'], question_id='Q1', debug=debug,
                answer=dict(value=value, api_value=value, label='Ignored client label',
                            free_text='UNVERIFIED_FREE_TEXT_MUST_NOT_BECOME_A_CLAIM'))

    def assert_neutral(self, result, prior, value):
        self.assertTrue(result['ok'])
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['final_ready'])
        self.assertEqual(result['hypotheses'], [])
        self.assertEqual(result['final_most_likely_label'], '')
        self.assertEqual(result['final_result']['most_likely_hypothesis_id'], '')
        self.assertEqual(result['final_probability_pct'], 0)
        self.assertEqual(result['final_probability_band'], 'unknown')
        self.assertEqual(json.loads(result['final_recommended_checks_json']), [])
        self.assertEqual(result['question_text'], '')
        self.assertTrue(all(result['h' + str(index) + '_id'] == '' for index in range(1, 5)))
        self.assertIn('diagnosis remains inconclusive', result['final_summary_text'])
        self.assertIn(prior['question_text'], result['final_summary_text'])
        self.assertIn('not an instruction', result['final_summary_text'])
        self.assertIn('does not confirm or exclude', result['final_summary_text'])
        self.assertNotIn('UNSUPPORTED_', json.dumps(result))
        self.assertNotIn('UNVERIFIED_FREE_TEXT', result['final_summary_text'])
        audit = result['meta']['smart_review']
        self.assertEqual(audit['outcome'], 'rejected')
        self.assertTrue(audit['decision_validated'])
        self.assertEqual(audit['failure_reason'], 'no_supported_hypotheses')
        self.assertFalse(result['meta']['protocol_terminal']['current_proposal_accepted'])
        self.assertFalse(result['meta']['protocol_terminal']['exclusions_certified'])
        self.assertTrue(result['meta']['source_manifest']['protocol_context_only'])
        self.assertFalse(result['meta']['source_manifest']['reviewer_proof_bound'])
        self.assertTrue(result['meta']['source_manifest']['question_proof_bound'])
        self.assertTrue(result['citations'])
        state = json.loads(result['session_state_json'])
        self.assertNotIn('grounding_review', state)
        self.assertIn('prior_review', state['protocol_inconclusive'])
        self.assertEqual(state['history'][-1]['answer']['api_value'], value)
        self.m._sd_validate_state_binding(state, company_id='fixture-company', machine_id='fixture-machine',
                                          session_id='fixture-session')
        return state

    def test_valid_no_returns_claimless_protocol_terminal_without_leaking_rejected_draft(self):
        start = self.start()
        response = self.rejected_update(start, debug=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_neutral(response.json(), start, 'no')

    def test_policy_is_generic_to_closed_yes_or_unknown_not_a_no_specific_rule(self):
        for value in ('yes', 'unknown'):
            with self.subTest(value=value):
                start = self.start()
                response = self.rejected_update(start, value=value)
                self.assertEqual(response.status_code, 200, response.text)
                self.assert_neutral(response.json(), start, value)

    def test_finalize_replays_exact_protocol_result_without_generation_enrichment_or_provider(self):
        start = self.start()
        result = self.rejected_update(start).json()
        self.assert_neutral(result, start, 'no')
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_llm_step_answer', side_effect=denied), \
             patch.object(self.m, '_sd_enrich_state_evidence_from_answer', side_effect=denied):
            for _ in range(2):
                response = self.post('finalize', result['session_state_json'])
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['final_result'], result['final_result'])
                self.assertEqual(response.json()['citations'], result['citations'])
                self.assertEqual(response.json()['meta']['protocol_terminal']['finalization_provider_calls'], 0)
                result = response.json()

    def test_missing_or_changed_prior_review_cannot_grant_protocol_terminal(self):
        for mutation in ('missing', 'observations', 'claims'):
            with self.subTest(mutation=mutation):
                start = self.start()
                state = json.loads(start['session_state_json'])
                if mutation == 'missing':
                    state.pop('grounding_review')
                elif mutation == 'observations':
                    state['history'].append({'question': {}, 'answer': {'value': 'unknown'}})
                else:
                    state['hypotheses'][0]['why'] += ' Altered prior claim.'
                start['session_state_json'] = json.dumps(self.m._sd_sign_state(state))
                response = self.rejected_update(start)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(response.json()['ok'])
                self.assertEqual(response.json()['status'], 'no_sources')
                self.assertNotIn('protocol_terminal', response.json()['meta'])

    def test_finalize_language_override_preserves_replayable_signed_record(self):
        start = self.start()
        result = self.rejected_update(start).json()
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            response = self.post('finalize', result['session_state_json'], language='it')
            self.assertEqual(response.status_code, 200, response.text)
            final = response.json()
            self.assertEqual(final['language'], result['language'])
            self.assertEqual(final['final_result'], result['final_result'])
            repeated = self.post('finalize', final['session_state_json'])
            self.assertEqual(repeated.status_code, 200, repeated.text)

    def test_legitimate_packet_change_after_prior_snapshot_cannot_use_old_review(self):
        start = self.start()
        changed = deepcopy(self.sources)
        changed[0]['chunk_full'] += '\nNewly admitted documentary context.'
        def enrich(**kwargs):
            state = deepcopy(kwargs['state'])
            state['grounding_packet'] = self.m._smart_evidence.build(changed,
                scope=self.m._sd_grounding_scope('fixture-company', 'fixture-machine'),
                max_chars=22000, allow_raw_snippet=True)
            return state
        with patch.object(self.m, '_sd_enrich_state_evidence_from_answer', side_effect=enrich):
            response = self.rejected_update(start)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['ok'])
        self.assertNotIn('protocol_terminal', response.json()['meta'])

    def test_foreign_tenant_bad_signature_and_invalid_option_fail_before_provider(self):
        start = self.start()
        with patch.object(self.m, '_v13_json_models', side_effect=denied), \
             patch.object(self.m, '_sd_llm_step_answer', side_effect=denied):
            foreign = self.post('answer', start['session_state_json'], company_id='other-company',
                question_id='Q1', answer={'value': 'no', 'api_value': 'no'})
            self.assertEqual(foreign.status_code, 400)
            invalid = self.post('answer', start['session_state_json'], question_id='Q1',
                answer={'value': 'invented', 'api_value': 'invented'})
            self.assertEqual(invalid.status_code, 400)
            state = json.loads(start['session_state_json'])
            state['symptom_text'] = 'Tampered symptom'
            tampered = self.post('answer', json.dumps(state), question_id='Q1',
                answer={'value': 'no', 'api_value': 'no'})
            self.assertEqual(tampered.status_code, 400)

    def test_finalize_rejects_resigned_claim_history_prior_proof_or_packet_tampering(self):
        start = self.start()
        result = self.rejected_update(start).json()
        for mutation in ('final', 'history', 'question', 'prior_claim', 'prior_proof', 'packet', 'reject_audit'):
            with self.subTest(mutation=mutation):
                state = json.loads(result['session_state_json'])
                if mutation == 'final':
                    state['final_result']['summary'] = 'Invented conclusion'
                elif mutation == 'history':
                    state['history'][-1]['answer']['api_value'] = 'yes'
                elif mutation == 'question':
                    state['history'][-1]['question']['question_text'] = 'Invented operation'
                elif mutation == 'prior_claim':
                    state['protocol_inconclusive']['prior_hypotheses'][0]['why'] = 'Invented prior claim'
                elif mutation == 'prior_proof':
                    state['protocol_inconclusive']['prior_review'].pop('decisions')
                elif mutation == 'packet':
                    state['grounding_packet']['bodies'][0]['text'] += ' Altered source'
                else:
                    state['grounding_review_meta']['failure_reason'] = 'transport_error'
                state = self.m._sd_sign_state(state)
                with patch.object(self.m, '_v13_json_models', side_effect=denied):
                    response = self.post('finalize', json.dumps(state))
                self.assertIn(response.status_code, (409, 502), response.text)

    def test_transport_or_malformed_review_does_not_gain_protocol_success(self):
        start = self.start()
        proposal = deepcopy(self.raw)
        proposal['question']['question_text'] = 'Another question?'
        for fault in (RuntimeError('offline transport failure'), ({'invalid': True}, 'offline-review')):
            with self.subTest(fault=type(fault).__name__):
                provider = patch.object(self.m, '_v13_json_models', side_effect=fault) if isinstance(fault, Exception) else patch.object(self.m, '_v13_json_models', return_value=fault)
                with patch.object(self.m, '_sd_llm_step_answer', return_value=proposal), provider:
                    response = self.post('answer', start['session_state_json'], question_id='Q1',
                        answer={'value': 'no', 'api_value': 'no'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(response.json()['ok'])
                self.assertEqual(response.json()['status'], 'error')
                self.assertFalse(response.json()['final_ready'])
                self.assertNotIn('protocol_terminal', response.json()['meta'])


del SmartReviewEndpointTests

if __name__ == '__main__':
    unittest.main()
