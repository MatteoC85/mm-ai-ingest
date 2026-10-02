"""Offline source-review contracts and emitted signed Smart state lifecycle."""
import copy
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import smart_review as review, smart_evidence, review_references as refs


def denied(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


def fixture():
    sources = [{'citation_id': f'manual:p{i}', 'bubble_document_id': 'manual', 'source_type': 'document',
                'company_id': 'fixture-company', 'machine_id': 'fixture-machine',
                'page_from': i, 'page_to': i, 'display_title': 'Fixture manual',
                'display_label': f'Fixture manual page{i}', 'snippet_clean': 'Display title only',
                'chunk_full': ('Observe the material and sensor from outside guards. Material absence can cause a missing signal. '
                               'Sensor displacement or cable damage can prevent detection. Compare the actual signal with its control window. '
                               'Keep isolation for physical inspection; do not bypass guards or change protected parameters.')} for i in range(1, 3)]
    hyps = [{'id': f'H{i}', 'rank': i, 'label': f'Qualified hypothesis {i}', 'description': 'A documented conditional mechanism.',
             'why': 'The reported stop is compatible; the signal is not yet verified.', 'probability_pct': pct,
             'probability_band': 'low', 'status': 'open', 'checks': ['Observe the HMI from outside guards.'],
             'evidence_ids': ['manual:p2' if i == 2 else 'manual:p1']} for i, pct in enumerate([50, 25, 15, 10], 1)]
    question = {'question_id': 'Q1', 'question_number': 1, 'question_type': 'single_choice',
                'question_text': 'Is material visibly present from a safe observation position?',
                'why_asked': 'Separate physical material presence from an unverified detected signal.',
                'safety_level': 'caution', 'safety_note': 'Do not enter or bypass the guarded area.',
                'options': [{'id': 'present', 'label_en': 'Present', 'label_it': 'Presente'},
                            {'id': 'unknown', 'label_en': 'Unknown / cannot check safely', 'label_it': 'Non verificabile in sicurezza'}],
                'target_hypotheses': [h['id'] for h in hyps]}
    return sources, {'status': 'in_progress', 'final_ready': False, 'operator_summary': 'Unreviewed summary mentioning H2.',
                     'question': question, 'hypotheses': hyps, 'final_result': {}}


def parsed_review(prepared, *, rejected=(1,), reject_question=False):
    frozen = prepared['references']['frozen']
    decisions = []
    for proposal in frozen['proposals']:
        index = proposal['proposal_index']
        question = proposal.get('kind') == 'diagnostic_question'
        reject = index in rejected or (question and reject_question)
        if reject:
            decisions.append({'proposal_index': index, 'verdict': 'reject', 'reason': 'unsupported_check',
                              'blocking_checks': [0] if proposal['checks'] else [], 'note': 'The cited source does not support this check.', 'proofs': []})
            continue
        decisions.append({'proposal_index': index, 'verdict': 'accept', 'reason': 'supported', 'blocking_checks': [], 'note': '',
                          'proofs': [{'source_index': 0, 'supports_cause': not question,
                                      'observation_units': [] if question else frozen['observed_ids'][:1],
                                      'source_units': frozen['source_sets'][0][:1], 'target_units': frozen['target_sets'][0][:1],
                                      'applicability': 'same_target', 'support_type': 'documented_check' if question else 'bounded_inference',
                                      'check_indices': list(range(len(proposal['checks'])))}]})
    return {'decisions': decisions}


def wire_review(parsed):
    """Test-only inverse, with no evidence synthesis or defaults."""
    kinds = {'documented_mechanism': 'm', 'bounded_inference': 'i', 'documented_check': 'c'}
    applicability = {'same_target': 's', 'documented_dependency': 'd'}
    return {'decisions': [{'p': d['proposal_index'], 'r': d['reason'], 'b': d['blocking_checks'], 'n': d['note'],
        'e': [{'c': p['check_indices'], 'v': {'s': p['source_index'], 'k': kinds[p['support_type']], 'o': p['observation_units'],
               'u': p['source_units'], 't': p['target_units'], 'a': applicability[p['applicability']],
               }} for p in d['proofs']]} for d in parsed['decisions']]}


class SmartReviewTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.step = fixture()
        self.scope = {'company_id': 'fixture-company', 'machine_id': 'fixture-machine', 'ai_scope': 'machine_all'}
        self.packet = smart_evidence.review_packet(smart_evidence.build(self.sources, scope=self.scope))
        self.prepared = review.prepare(step=self.step, packet=self.packet, symptom_text='Reported stop; signal unknown.', history=[])
        self.band = lambda p: 'high' if p >= 60 else 'medium' if p >= 25 else 'low' if p >= 10 else 'very_low'

    def resolve(self, **kw):
        return review.resolve(prepared=self.prepared, parsed=parsed_review(self.prepared, **kw), probability_band=self.band)

    def test_four_to_three_retains_independently_supported_question(self):
        out, seal, meta = self.resolve()
        self.assertEqual([h['id'] for h in out['hypotheses']], ['H1', 'H3', 'H4'])
        self.assertEqual(out['question']['target_hypotheses'], ['H1', 'H3', 'H4'])
        self.assertEqual(sum(h['probability_pct'] for h in out['hypotheses']), 100)
        self.assertEqual(out['hypotheses'][0]['probability_band'], 'high')
        self.assertNotIn('H2', out['operator_summary'])
        self.assertIn('not statistical certainty', out['operator_summary'])
        self.assertTrue(meta['question_validated'])

    def test_question_rejection_or_only_rejected_targets_fails(self):
        with self.assertRaisesRegex(review.SmartReviewError, 'question_not_supported'):
            self.resolve(reject_question=True)
        self.step['question']['target_hypotheses'] = ['H2']
        self.prepared = review.prepare(step=self.step, packet=self.packet, symptom_text='Stop', history=[])
        with self.assertRaisesRegex(review.SmartReviewError, 'question_has_no_supported_target'):
            self.resolve()

    def test_single_remaining_weight_is_not_diagnosis_certainty(self):
        out, _, _ = self.resolve(rejected=(1, 2, 3))
        self.assertEqual(out['hypotheses'][0]['probability_pct'], 100)
        self.assertIn('not statistical certainty or a confirmed diagnosis', out['operator_summary'])

    def test_root_default_rejects_smart_question_and_excess_proposals(self):
        manifest = {'proposals': self.prepared['references']['frozen']['proposals'],
                    'sources': self.prepared['references']['frozen']['source_manifest']}
        with self.assertRaisesRegex(refs.ReferenceError, 'invalid_proposal_count'):
            refs.prepare(packet=self.packet['model_packet'], proposal_manifest=manifest, records=self.packet['validator_records'],
                         original_query='Stop', observed_query='Stop')
        manifest['proposals'] = [copy.deepcopy(manifest['proposals'][-1])]
        manifest['proposals'][0]['proposal_index'] = 0
        with self.assertRaisesRegex(refs.ReferenceError, 'invalid_proposal_kind'):
            refs.prepare(packet=self.packet['model_packet'], proposal_manifest=manifest, records=self.packet['validator_records'],
                         original_query='Stop', observed_query='Stop')

    def test_cause_cannot_use_check_only_proof_or_foreign_unit(self):
        parsed = parsed_review(self.prepared)
        proof = parsed['decisions'][0]['proofs'][0]
        proof.update(supports_cause=False, support_type='documented_check', observation_units=[])
        with self.assertRaisesRegex(refs.ReferenceError, 'mechanism_not_supported'):
            review.resolve(prepared=self.prepared, parsed=parsed, probability_band=self.band)
        parsed = parsed_review(self.prepared)
        parsed['decisions'][0]['proofs'][0]['source_units'] = self.prepared['references']['frozen']['source_sets'][1]
        with self.assertRaisesRegex(refs.ReferenceError, 'reference_outside_authorized_set'):
            review.resolve(prepared=self.prepared, parsed=parsed, probability_band=self.band)

    def test_sixth_check_not_silently_truncated_and_unknown_not_observation(self):
        self.step['hypotheses'][0]['checks'] = ['Observe safely.'] * 5 + ['Unsupported sixth check.']
        with self.assertRaisesRegex(review.SmartReviewError, 'invalid_check_count'):
            review.prepare(step=self.step, packet=self.packet, symptom_text='Stop', history=[])
        text = review.observation_text('Stop', [{'question': {'question_text': 'Signal?'},
                                               'answer': {'api_value': 'unknown', 'free_text': 'NOT_AN_OBSERVATION'}}])
        self.assertNotIn('NOT_AN_OBSERVATION', text)

    def test_question_fields_and_options_are_immutable_covered_checks(self):
        question = self.prepared['references']['frozen']['proposals'][-1]
        full = ''.join(c['text'] for c in question['checks'])
        self.assertIn('Non verificabile in sicurezza', full)
        self.assertIn(self.step['question']['safety_note'], full)
        parsed = parsed_review(self.prepared)
        parsed['decisions'][-1]['proofs'][0]['check_indices'] = [0, 1]
        with self.assertRaisesRegex(refs.ReferenceError, 'not_all_checks_supported'):
            review.resolve(prepared=self.prepared, parsed=parsed, probability_band=self.band)

    def test_wire_roundtrip_preserves_accept_reject_rebinding_and_replay(self):
        parsed = parsed_review(self.prepared)
        decoded = review.decode_wire(wire_review(parsed))
        self.assertEqual(decoded, parsed)
        expected = review.resolve(prepared=self.prepared, parsed=parsed, probability_band=self.band)
        self.assertEqual(review.resolve(prepared=self.prepared, parsed=decoded, probability_band=self.band), expected)
        # A valid rebind to another ADMITTED source survives exactly, including its
        # source-local unit ownership. This is not semantic acceptance by this test.
        proof = parsed['decisions'][0]['proofs'][0]
        proof.update(source_index=1, source_units=self.prepared['references']['frozen']['source_sets'][1][:1],
                     target_units=self.prepared['references']['frozen']['target_sets'][1][:1])
        self.assertEqual(review.decode_wire(wire_review(parsed)), parsed)

    def test_wire_rejects_invalid_fields_codes_and_reference_locality(self):
        wire = wire_review(parsed_review(self.prepared))
        for mutate in (lambda w: w.update(extra=True),
                       lambda w: w['decisions'][0].pop('n'),
                       lambda w: w['decisions'][0]['e'][0]['v'].update(k='unknown')):
            bad = copy.deepcopy(wire)
            mutate(bad)
            with self.assertRaises(review.SmartReviewError):
                review.decode_wire(bad)
        for mutate in (lambda w: w['decisions'][0].update(p=True),
                       lambda w: w['decisions'][0].update(p=99),
                       lambda w: w['decisions'][0]['e'][0]['v'].update(s=True),
                       lambda w: w['decisions'][0]['e'][0]['v'].update(u=[999]),
                       lambda w: w['decisions'][0]['e'][0]['v'].update(u=self.prepared['references']['frozen']['source_sets'][1]),
                       lambda w: w['decisions'][0]['e'][0].update(c=[4]),
                       lambda w: w['decisions'][0]['e'][0].update(c=[0, 0]),
                       lambda w: w['decisions'].__setitem__(1, copy.deepcopy(w['decisions'][0]))):
            bad = copy.deepcopy(wire)
            mutate(bad)
            with self.assertRaises((review.SmartReviewError, refs.ReferenceError)):
                review.resolve(prepared=self.prepared, parsed=review.decode_wire(bad), probability_band=self.band)
        # The provider schema itself still constrains each source to its own units.
        shape = review.wire_schema(self.prepared)['schema']
        branches = shape['$defs']['proof']['anyOf']
        self.assertEqual(branches[0]['properties']['u']['items']['enum'],
                         self.prepared['references']['frozen']['source_sets'][0])
        decisions = shape['properties']['decisions']['items']['anyOf']
        self.assertEqual(decisions[0]['properties']['e']['items']['properties']['c']['items']['enum'], [0])

    def test_eight_source_wire_reduces_schema_and_output_without_dropping_material(self):
        sources = []
        for i in range(8):
            source = copy.deepcopy(self.sources[i % 2])
            source.update(citation_id=f'fixture:{i}', page_from=i + 1, page_to=i + 1,
                          chunk_full=(source['chunk_full'] + '\n') * 5)
            sources.append(source)
        step = copy.deepcopy(self.step)
        for hypothesis, count in zip(step['hypotheses'], [3, 2, 3, 2]):
            hypothesis['checks'] = [f'Check {j}: Observe safely from the authorized position.' for j in range(count)]
        packet = smart_evidence.review_packet(smart_evidence.build(sources, scope=self.scope))
        prepared = review.prepare(step=step, packet=packet, symptom_text='Reported stop; signal unknown.', history=[])
        old_schema = refs.schema(prepared['references']['frozen'])
        new_schema = review.wire_schema(prepared)
        parsed = parsed_review(prepared, rejected=())
        self.assertLess(len(refs.canonical(new_schema)), len(refs.canonical(old_schema)) * .5)
        self.assertLess(len(refs.canonical(wire_review(parsed))), len(refs.canonical(parsed)) * .6)
        self.assertEqual(review.decode_wire(wire_review(parsed)), parsed)
        self.assertEqual(len(packet['validator_records']), 8)
        self.assertEqual([r['text'] for r in packet['validator_records']], [s['chunk_full'] for s in sources])
        self.assertEqual(len(prepared['references']['frozen']['proposals']), 5)
        self.assertTrue(all(len(p['checks']) == n for p, n in zip(prepared['references']['frozen']['proposals'], [3, 2, 3, 2, 3])))

    def test_failure_diagnostic_never_exposes_error_body_or_source_text(self):
        detail = review.failure_diagnostic(RuntimeError('OpenAI provider returned HTTP 429 PRIVATE_TOKEN_RAW_BODY'),
            call_rows=[{'purpose': 'smart_diagnostic_independent_review', 'error': 'RuntimeError',
                        'accounting_state': 'uncertain', 'raw_body': 'PRIVATE_SOURCE'}], elapsed_seconds=1.25)
        self.assertEqual(detail['category'], 'provider_http')
        self.assertEqual(detail['http_status'], 429)
        self.assertEqual(set(detail), {'category', 'error_class', 'elapsed_seconds', 'accounting_state', 'http_status'})
        self.assertNotIn('PRIVATE', refs.canonical(detail))


class SmartReviewEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT': 'off', 'AI_INTERNAL_SECRET': 'offline-smart-review-only',
                                                       'OPENAI_API_KEY': 'offline-unused', 'MM_INGEST_LEDGER_AUTO_DDL': '0'}))
        original = socket.socket.connect
        def guarded(sock, address):
            if isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}:
                return original(sock, address)
            return denied()
        cls.guards.enter_context(patch.object(socket.socket, 'connect', guarded))
        cls.guards.enter_context(patch.object(socket, 'create_connection', denied))
        cls.guards.enter_context(patch('requests.sessions.Session.request', denied))
        cls.guards.enter_context(patch('psycopg2.connect', denied))
        cls.m = importlib.import_module('main')
        from fastapi.testclient import TestClient
        cls.client = TestClient(cls.m.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.guards.close()

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for name, value in [('AI_INTERNAL_SECRET', 'offline-smart-review-only'), ('SMART_DIAGNOSTIC_ENABLED', True), ('ASSISTANT_CORE_V2_ENABLED', True)]:
            self.patches.enter_context(patch.object(self.m, name, value))
        self.patches.enter_context(patch.object(self.m, '_build_rg_links', return_value=[]))
        self.patches.enter_context(patch.object(self.m, '_sd_enrich_state_evidence_from_answer', side_effect=lambda **kw: kw['state']))
        self.sources, self.raw = fixture()
        self.patches.enter_context(patch.object(self.m, '_sd_llm_step_start', side_effect=lambda **kw: copy.deepcopy(self.raw)))
        def run(request):
            decision = SimpleNamespace(evidence_state='supported', confidence=1, router_model='offline', relevant_evidence_ids=['manual:p1'], request_kind='fault_diagnostic')
            response = self.m._assistant_core_synthesize_smart_start(request, {'citations': self.sources}, decision)
            response.update(effective_mode='smart_diagnostic')
            return response
        self.patches.enter_context(patch.object(self.m._ASSISTANT_CORE_SMART_ENGINE, 'run', side_effect=run))
        original_prepare = review.prepare
        self.prepared = []
        self.real_json_models = self.m._v13_json_models
        def prepare(**kw):
            result = original_prepare(**kw)
            self.prepared.append(result)
            return result
        self.patches.enter_context(patch.object(review, 'prepare', side_effect=prepare))
        def provider(messages, **kw):
            self.assertEqual(kw['models'], [self.m.V13_FAST_MODEL])
            self.assertLessEqual(kw['timeout'], 20)
            rejected = (1,) if len(self.prepared[-1]['step']['hypotheses']) == 4 else ()
            return wire_review(parsed_review(self.prepared[-1], rejected=rejected)), 'offline-review'
        self.provider = self.patches.enter_context(patch.object(self.m, '_v13_json_models', side_effect=provider))
        self.patches.enter_context(patch.object(self.m, '_sd_llm_finalize', side_effect=denied))

    def post(self, kind, state=None, **kw):
        body = {'company_id': 'fixture-company', 'machine_id': 'fixture-machine', 'session_id': 'fixture-session', 'language': 'en'}
        if state is not None:
            body['state_json'] = state
        body.update(kw)
        return self.client.post('/v1/ai/smart-diagnostic/' + kind, json=body,
                                headers={'x-ai-internal-secret': 'offline-smart-review-only'})

    def start(self):
        r = self.post('start', symptom_text='Reported stop; actual signal remains unknown.')
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['hypotheses'], r.text)
        return r.json()

    def test_emitted_signed_start_answer_finalize_and_repeated_finalize(self):
        start = self.start()
        self.assertEqual(len(start['hypotheses']), 3)
        answer_step = copy.deepcopy(self.raw)
        answer_step['hypotheses'] = start['hypotheses']
        answer_step['question']['question_text'] = 'Does the HMI show the material detection signal?'
        answer_step['question']['target_hypotheses'] = [h['id'] for h in start['hypotheses']]
        with patch.object(self.m, '_sd_llm_step_answer', return_value=answer_step):
            r = self.post('answer', start['session_state_json'], question_id='Q1',
                          answer={'value': 'unknown', 'api_value': 'unknown', 'label': 'Unknown', 'free_text': ''})
        self.assertEqual(r.status_code, 200, r.text)
        answer = r.json()
        self.assertEqual(answer['question_number'], 2)
        before = self.provider.call_count
        final = self.post('finalize', answer['session_state_json'])
        self.assertEqual(final.status_code, 200, final.text)
        self.assertTrue(final.json()['final_ready'], final.text)
        self.assertEqual(self.provider.call_count, before)
        again = self.post('finalize', final.json()['session_state_json'])
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(again.json()['final_result'], final.json()['final_result'])
        self.assertEqual(self.provider.call_count, before)

    def test_internal_claim_change_or_missing_proof_rejected_before_provider(self):
        start = self.start()
        for change in ('check', 'proof'):
            state = json.loads(start['session_state_json'])
            if change == 'check':
                state['hypotheses'][0]['checks'][0] = 'Undocumented changed instruction.'
            else:
                state.pop('grounding_review')
            signed = self.m._sd_sign_state(state)
            before = self.provider.call_count
            r = self.post('finalize', signed)
            self.assertEqual(r.status_code, 409, r.text)
            self.assertEqual(self.provider.call_count, before)

    def test_old_title_only_state_requires_restart_before_generation(self):
        start = self.start()
        state = json.loads(start['session_state_json'])
        state.pop('grounding_packet')
        with patch.object(self.m, '_sd_llm_step_answer', side_effect=denied):
            r = self.post('answer', self.m._sd_sign_state(state), question_id='Q1',
                          answer={'value': 'unknown', 'api_value': 'unknown', 'label': 'Unknown'})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(r.json()['detail']['code'], 'SMART_DIAGNOSTIC_SESSION_RESTART_REQUIRED')

    def test_long_safety_check_survives_normalization_and_finalization(self):
        check = 'Inspect only from the authorized observation position. ' + ('Keep the setup condition. ' * 8) + 'Never bypass the guard.'
        self.raw['hypotheses'][0]['checks'] = [check]
        start = self.start()
        final = self.post('finalize', start['session_state_json'])
        self.assertEqual(final.status_code, 200, final.text)
        self.assertIn(check, final.json()['final_result']['recommended_checks'])

    def test_review_failure_has_one_real_ledger_reservation_no_retry(self):
        import requests
        for mode in ('malformed', 'timeout'):
            with self.subTest(mode=mode):
                def http(*args, **kwargs):
                    if mode == 'timeout':
                        raise requests.ReadTimeout('offline timeout')
                    return SimpleNamespace(status_code=200, json=lambda: {'status': 'completed',
                        'usage': {'input_tokens': 100, 'output_tokens': 50}, 'output_text': '{"decisions":[]}'})
                with patch.object(self.m, '_v13_json_models', self.real_json_models), \
                     patch('machinemind.authority.request_admission.before_egress', return_value=None), \
                     patch.object(self.m.requests, 'post', side_effect=http) as post:
                    response = self.post('start', symptom_text='Reported stop; signal unknown.')
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body['result_code'], 'TECHNICAL_ERROR')
                self.assertEqual(post.call_count, 1)
                self.assertEqual(body['meta']['v13_llm_calls'], 1)
                self.assertGreater(body['meta']['v13_committed_cost_usd'], 0)
                self.assertEqual(body['meta']['v13_accounting_complete'], mode == 'malformed')
                if mode == 'timeout':
                    self.assertGreater(body['meta']['v13_uncertain_cost_usd'], 0)
                    diagnostic = body['meta']['smart_review']['transport']
                    self.assertEqual(diagnostic['category'], 'timeout')
                    self.assertEqual(diagnostic['error_class'], 'ReadTimeout')
                    self.assertEqual(diagnostic['accounting_state'], 'uncertain')

    def test_each_planner_fallback_keeps_review_time_and_call_slot(self):
        budget = self.m._assistant_core_new_budget('smart_diagnostic', company_id='fixture-company')
        token = self.m._V13_BUDGET_CTX.set(budget)
        try:
            with patch.object(budget, 'remaining', return_value=45.0), \
                 patch.object(self.m, '_v13_json_models', side_effect=[RuntimeError('offline first model failure'), ({}, 'second')]) as provider:
                result = self.m._assistant_core_sd_json_models([], json_schema={'name': 'fixture', 'schema': {}}, timeout=70, phase='start')
            self.assertEqual(result, {})
            self.assertEqual(provider.call_count, 2)
            self.assertTrue(all(call.kwargs['timeout'] <= 24 for call in provider.call_args_list))
        finally:
            self.m._V13_BUDGET_CTX.reset(token)

    def test_final_public_citations_keep_seven_proof_selected_ids(self):
        citations = [{'citation_id': f'source{i}:p1', 'bubble_document_id': f'source{i}', 'source_type': 'document',
                      'display_title': f'Source {i}', 'display_label': f'Source {i} page1', 'page_from': 1, 'page_to': 1,
                      'snippet': f'Distinct support record number {i}.'} for i in range(7)]
        hyps = copy.deepcopy(self.raw['hypotheses'])
        for h, source_range in zip(hyps, ((0, 1), (2, 3), (4, 5), (6,))):
            h['evidence_ids'] = [citations[i]['citation_id'] for i in source_range]
        step = self.m._sd_terminal_reviewed_step({'hypotheses': hyps}, language='en', question_number=1)
        response = self.m._sd_response_from_step(session_id='fixture-session', company_id='fixture-company',
            machine_id='fixture-machine', symptom_text='Reported stop.', language='en',
            state={'grounding_review': {'policy_version': review.POLICY_VERSION}, 'evidence': citations},
            step=step, citations=citations, rg_links=[])
        public_ids = {c['citation_id'] for c in response['citations']}
        self.assertEqual(len(public_ids), 7)
        self.assertTrue(all(set(h['evidence_ids']) <= public_ids for h in response['hypotheses']))

    def test_legacy_start_uses_complete_packet_review_and_deterministic_finalize(self):
        with patch.object(self.m, 'ASSISTANT_CORE_V2_ENABLED', False), \
             patch.object(self.m, '_diagnostic_evidence_pipeline', return_value={'citations': self.sources}), \
             patch.object(self.m, '_v13_deterministic_evidence_state', return_value=('supported', {})), \
             patch.object(self.m, '_sd_semantic_evidence_gate', return_value={'accepted': True, 'decision': 'supported'}), \
             patch.object(self.m, '_sd_run_retrieval_assurance', return_value=(True, self.sources, {})):
            start = self.start()
            before = self.provider.call_count
            final = self.post('finalize', start['session_state_json'])
        self.assertEqual(final.status_code, 200, final.text)
        self.assertTrue(final.json()['final_ready'])
        self.assertEqual(self.provider.call_count, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
