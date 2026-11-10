"""Public confidence projection with real signed endpoint/reviewer replay.

Provider responses are controlled offline decisions, not new live accuracy proof.
The sanitized actual R12 receipt reproduces the public 100/unknown mismatch.
"""
import ast
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
WORK = HERE.parent
REPO = WORK / 'mm-ai-ingest'
if not REPO.exists():
    REPO = HERE
MAIN = HERE / 'source/main.py'
if not MAIN.exists():
    MAIN = REPO / 'main.py'
MODULE = HERE / 'source/machinemind/retrieval/smart_presentation.py'
if not MODULE.exists():
    MODULE = REPO / 'machinemind/retrieval/smart_presentation.py'
spec = importlib.util.spec_from_file_location('_smart_public_projection_under_test', MODULE)
projection = importlib.util.module_from_spec(spec); spec.loader.exec_module(projection)
sys.path[:0] = [str(HERE / 'tests'), str(REPO), str(REPO / 'tests')]
from test_smart_review import SmartReviewEndpointTests, denied
from test_smart_continuity_endpoints import direct_post


class PublicProjectionTests(unittest.TestCase):
    def hypothesis(self, status='open', pct=100, band='unknown'):
        return dict(id='H1', rank=1, label='Documented conditional mechanism', status=status,
            why='The actual observation remains unknown.', probability_pct=pct,
            probability_band=band, score_raw=pct / 100, checks=['Keep the approved guard condition.'],
            evidence_ids=['manual:p1'], probability_semantics='relative_weight_not_calibrated_probability')

    def test_actual_R12_final_unknown_100_has_neutral_public_fields_without_new_claims(self):
        receipt = WORK / 'phase6-ask-correction-r12-20261110/live-client/receipts/R12-N-SMART-UNKNOWN-FINAL.json'
        self.assertEqual(hashlib.sha256(receipt.read_bytes()).hexdigest(), '70329a26e2a320322068c07db45f7e3adc30f919e7143de79560256b64de6c8d')
        raw = json.loads(receipt.read_text(encoding='utf-8'))['response']
        self.assertEqual(raw['final_probability_pct'], 100)
        self.assertEqual(raw['final_probability_band'], 'unknown')
        before = deepcopy(raw)
        final, hyps, outcome = projection.project(final_result=raw['final_result'], hypotheses=raw['hypotheses'],
            final_ready=raw['final_ready'], status=raw['status'], language=raw['language'])
        self.assertEqual((final['most_likely_hypothesis_id'], final['most_likely_label'],
            final['probability_pct'], final['probability_band']), ('', 'No cause identified', 0, 'unknown'))
        self.assertEqual(outcome, 'inconclusive')
        self.assertEqual(final['summary'], raw['final_result']['summary'])
        self.assertEqual(final['recommended_checks'], raw['final_result']['recommended_checks'])
        self.assertEqual(hyps[0]['probability_pct'], 0)
        self.assertEqual(hyps[0]['relative_weight_pct'], 100)
        for key in ('id', 'rank', 'label', 'description', 'why', 'checks', 'evidence_ids', 'score_raw', 'status'):
            self.assertEqual(hyps[0][key], raw['hypotheses'][0][key])
        self.assertEqual(raw, before)

    def test_cross_language_empty_excluded_and_no_question_do_not_invent_a_cause(self):
        for language, label in [('en', 'No cause identified'), ('it', 'Nessuna causa identificata')]:
            for hypotheses in ([], [self.hypothesis()], [self.hypothesis('excluded', 0)]):
                final = {'summary': 'Literal reviewed explanation.', 'recommended_checks': []}
                old = deepcopy((final, hypotheses))
                public, visible, outcome = projection.project(final_result=final, hypotheses=hypotheses,
                    final_ready=True, status='completed', language=language)
                self.assertEqual(public['most_likely_label'], label)
                self.assertEqual(public['probability_pct'], 0)
                self.assertEqual(public['probability_band'], 'unknown')
                self.assertEqual(public['summary'], final['summary'])
                self.assertEqual(outcome, 'inconclusive')
                self.assertEqual((final, hypotheses), old)

    def test_confirmed_supported_case_and_no_sources_keep_original_fields(self):
        for status, hyps in [('completed', [self.hypothesis('confirmed')]),
                             ('no_sources', [])]:
            final = dict(summary='Literal reviewed result.', most_likely_hypothesis_id='H1',
                most_likely_label='Documented conditional mechanism', probability_pct=100,
                probability_band='high', recommended_checks=['Reviewed check.'])
            original = deepcopy((final, hyps))
            result = projection.project(final_result=final, hypotheses=hyps, final_ready=True,
                status=status, language='en')
            self.assertEqual(result, (*original, None))

    def test_only_selected_confirmed_hypothesis_can_keep_final_confirmation(self):
        hypotheses = [self.hypothesis(), {**self.hypothesis('confirmed', 70, 'high'), 'id':'H2', 'rank':2}]
        for selected, expected_outcome in [('H1', 'inconclusive'), ('H2', None), ('foreign', 'inconclusive')]:
            final = dict(summary='Literal reviewed explanation.', most_likely_hypothesis_id=selected,
                most_likely_label='Reviewed selected hypothesis', probability_pct=70, probability_band='high', recommended_checks=[])
            original = deepcopy((final, hypotheses))
            result, visible, outcome = projection.project(final_result=final, hypotheses=hypotheses,
                final_ready=True, status='completed', language='en')
            self.assertEqual(outcome, expected_outcome)
            self.assertEqual(visible[0]['probability_pct'], 0)
            self.assertEqual(visible[0]['score_raw'], 1)
            self.assertEqual(visible[1], hypotheses[1])
            if selected == 'H2':
                self.assertEqual(result, final)
            else:
                self.assertEqual(result['most_likely_hypothesis_id'], '')
                self.assertEqual(result['probability_pct'], 0)
                self.assertEqual(result['probability_band'], 'unknown')
            self.assertEqual((final, hypotheses), original)

    def test_in_progress_unknown_confidence_uses_separate_relative_weight_and_preserves_ranking(self):
        raw = [self.hypothesis(pct=70), {**self.hypothesis(pct=30), 'id':'H2', 'rank':2}]
        final, hyps, outcome = projection.project(final_result={}, hypotheses=raw, final_ready=False,
            status='in_progress', language='en')
        self.assertEqual(final, {})
        self.assertIsNone(outcome)
        self.assertEqual([h['probability_pct'] for h in hyps], [0, 0])
        self.assertEqual([h['relative_weight_pct'] for h in hyps], [70, 30])
        self.assertEqual([h['score_raw'] for h in hyps], [.7, .3])
        self.assertEqual([h['rank'] for h in hyps], [1, 2])
        self.assertEqual([h['probability_pct'] for h in raw], [70, 30])


class SignedEndpointProjectionTests(unittest.TestCase):
    setUpClass_original = SmartReviewEndpointTests.setUpClass.__func__
    tearDownClass_original = SmartReviewEndpointTests.tearDownClass.__func__
    setup_fixture = SmartReviewEndpointTests.setUp
    start = SmartReviewEndpointTests.start
    post = direct_post

    @classmethod
    def setUpClass(cls):
        cls.setUpClass_original(cls)
        names = {'_sd_response_from_step', '_sd_final_result_for_display'}
        nodes = [n for n in ast.parse(MAIN.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        for node in nodes:
            cls.guards.callback(setattr, cls.m, node.name, getattr(cls.m, node.name))
        cls.guards.enter_context(patch.object(cls.m, '_smart_presentation', projection, create=True))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN), 'exec'), cls.m.__dict__)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownClass_original(cls)

    def setUp(self):
        self.setup_fixture()
        self.raw['hypotheses'] = self.raw['hypotheses'][:1]
        self.raw['hypotheses'][0].update(status='open', probability_pct=100.0, probability_band='unknown', score_raw=1.0, checks=[])
        self.raw['question']['target_hypotheses'] = [self.raw['hypotheses'][0]['id']]

    def assert_public(self, body):
        self.assertEqual(body['final_result'], json.loads(body['final_result_json']))
        self.assertEqual(body['final_most_likely_label'], 'No cause identified')
        self.assertEqual(body['final_result']['most_likely_hypothesis_id'], '')
        self.assertEqual(body['final_probability_pct'], 0)
        self.assertEqual(body['final_probability_band'], 'unknown')
        self.assertEqual(body['meta']['diagnostic_outcome'], 'inconclusive')
        self.assertEqual(body['hypotheses'], json.loads(body['hypotheses_json']))
        self.assertEqual(body['hypotheses'][0]['probability_pct'], body['h1_probability_pct'])
        self.assertEqual(body['h1_probability_pct'], 0)
        self.assertEqual(body['hypotheses'][0]['relative_weight_pct'], 100)
        self.assertEqual(body['h1_score_raw'], 1)
        self.assertEqual(body['operator_summary'], body['final_summary_text'])
        self.assertEqual(body['final_summary_text'], body['final_result']['summary'])

    def test_real_signed_unknown_chain_neutralizes_every_public_view_but_preserves_internal_state(self):
        start = self.start()
        original = json.loads(start['session_state_json'])
        self.assertEqual(start['h1_probability_pct'], 0)
        self.assertEqual(original['hypotheses'][0]['probability_pct'], 100)
        next_step = deepcopy(self.raw)
        next_step['question']['question_text'] = 'What was already recorded before the stop?'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step):
            answered = self.post('answer', start['session_state_json'], question_id='Q1',
                answer={'value':'unknown', 'api_value':'unknown', 'free_text':'Unverified note stays unknown.'})
        self.assertEqual(answered.status_code, 200, answered.text)
        answered = answered.json(); before = json.loads(answered['session_state_json'])
        self.assertEqual(before['history'][-1]['answer']['api_value'], 'unknown')
        calls = self.provider.call_count
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            result = self.post('finalize', answered['session_state_json'])
            self.assertEqual(result.status_code, 200, result.text)
            body = result.json(); repeated = self.post('finalize', body['session_state_json'])
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assert_public(body)
        self.assertEqual(body['final_result'], repeated.json()['final_result'])
        signed = json.loads(body['session_state_json'])
        for key in ('hypotheses', 'history', 'grounding_review', 'grounding_packet'):
            self.assertEqual(signed[key], before[key])
        self.assertEqual(signed['final_result']['probability_pct'], 100)
        self.assertEqual(signed['hypotheses'][0]['probability_pct'], 100)
        self.m._sd_validate_state_binding(signed, company_id='fixture-company', machine_id='fixture-machine', session_id='fixture-session')
        self.assertEqual(body['citations'], repeated.json()['citations'])
        self.assertEqual(body['rg_links'], repeated.json()['rg_links'])
        self.assertEqual(self.provider.call_count, calls)

    def test_Unknown_No_and_Yes_stay_distinct_and_do_not_autoconfirm_open_hypotheses(self):
        self.raw['question']['options'] = [
            {'id':'yes', 'label_it':'Sì', 'label_en':'Yes'},
            {'id':'no', 'label_it':'No', 'label_en':'No'},
            {'id':'unknown', 'label_it':'Non so', 'label_en':'Unknown'}]
        for value in ('unknown', 'no', 'yes'):
            with self.subTest(value=value):
                start = self.start(); next_step = deepcopy(self.raw)
                next_step['question']['question_text'] = 'Which information was already available?'
                with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step):
                    result = self.post('answer', start['session_state_json'], question_id='Q1',
                        answer={'value':value, 'api_value':value})
                self.assertEqual(result.status_code, 200, result.text)
                state = json.loads(result.json()['session_state_json'])
                self.assertEqual(state['history'][-1]['answer']['api_value'], value)
                self.assertEqual(state['hypotheses'][0]['status'], 'open')
                with patch.object(self.m, '_v13_json_models', side_effect=denied):
                    final = self.post('finalize', result.json()['session_state_json'])
                self.assertEqual(final.status_code, 200, final.text)
                self.assert_public(final.json())

    def test_mixed_reviewed_status_does_not_let_unselected_confirmation_mask_open_lead(self):
        raw = deepcopy(self.raw['hypotheses'])
        raw.append({**deepcopy(raw[0]), 'id':'confirmed-alternative', 'rank':2, 'status':'confirmed', 'probability_pct':20})
        final = self.m._sd_canonicalize_final_result({}, raw, 'en')
        self.assertEqual(final['most_likely_hypothesis_id'], raw[0]['id'])
        displayed = self.m._sd_final_result_for_display(final, raw, 'en')
        self.assertIn('diagnosis is inconclusive', displayed['summary'])
        public, visible, outcome = projection.project(final_result=displayed, hypotheses=raw,
            final_ready=True, status='completed', language='en')
        self.assertEqual(outcome, 'inconclusive')
        self.assertEqual(public['probability_pct'], 0)
        self.assertEqual(visible[1]['status'], 'confirmed')
        self.assertEqual(visible[1]['probability_pct'], 20)

    def test_reviewed_selected_confirmation_after_positive_observation_keeps_existing_endpoint_flow(self):
        start = self.start(); next_step = deepcopy(self.raw)
        next_step['hypotheses'][0].update(status='confirmed', probability_pct=100)
        next_step['question']['question_text'] = 'Which existing observation identifies the recorded condition?'
        with patch.object(self.m, '_sd_llm_step_answer', return_value=next_step):
            answered = self.post('answer', start['session_state_json'], question_id='Q1',
                answer={'value':'present', 'api_value':'present'})
        self.assertEqual(answered.status_code, 200, answered.text)
        state = json.loads(answered.json()['session_state_json'])
        self.assertEqual(state['history'][-1]['answer']['api_value'], 'present')
        self.assertEqual(state['hypotheses'][0]['status'], 'confirmed')
        with patch.object(self.m, '_v13_json_models', side_effect=denied):
            result = self.post('finalize', answered.json()['session_state_json'])
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json(); signed = json.loads(body['session_state_json'])
        self.assertEqual(body['final_result'], signed['final_result'])
        self.assertEqual(body['final_most_likely_label'], state['hypotheses'][0]['label'])
        self.assertEqual(body['final_probability_pct'], 100)
        self.assertEqual(body['hypotheses'], signed['hypotheses'])
        self.assertNotIn('diagnostic_outcome', body['meta'])
        self.m._sd_validate_state_binding(signed, company_id='fixture-company', machine_id='fixture-machine', session_id='fixture-session')

    def test_public_probability_cannot_be_substituted_into_signed_state(self):
        start = self.start(); tampered = json.loads(start['session_state_json'])
        tampered['hypotheses'] = start['hypotheses']
        calls = self.provider.call_count
        response = self.post('finalize', json.dumps(tampered))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['detail']['code'], 'SMART_DIAGNOSTIC_STATE_TAMPERED')
        self.assertEqual(self.provider.call_count, calls)


del SmartReviewEndpointTests
if __name__ == '__main__':
    unittest.main(verbosity=2)
