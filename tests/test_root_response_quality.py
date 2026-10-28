"""Offline Root review selection/ranking regressions. No model or network calls."""
import ast
from copy import deepcopy
import json
import time
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality


def fixture():
    records = [
        {'citation_id': 'manual-feed', 'text': 'Model Z: pressure loss allows the clamp to slip. Read the existing pressure indication. Qualified personnel must isolate before accessing the clamp.', 'ownership_fragments': []},
        {'citation_id': 'manual-filter', 'text': 'Model Z: a restricted filter reduces flow. Record the existing differential pressure indication.', 'ownership_fragments': []},
    ]
    packet = {'sources': [{'citation_id': row['citation_id'], 'text': row['text'], 'context_ids': []}
                          for row in records], 'document_contexts': []}
    draft = [
        {'cause': 'Possible loss of clamping pressure', 'why': 'Loss of pressure could explain observed slip; pressure is unknown.',
         'checks': ['Record the pressure indication already available.', 'Qualified personnel: isolate before accessing the clamp.', 'Adjust the clamp under load with guards open.']},
        {'cause': 'Possible restricted filter', 'why': 'Restriction can reduce flow; the pressure difference is unknown.',
         'checks': ['Record the differential pressure indication already available.']},
    ]
    observed = 'The feed slips intermittently while the motor keeps running.'
    manifest = review_decisions.manifest(draft, records, max_causes=3)
    frozen = refs.prepare(packet=packet, proposal_manifest=manifest, records=records,
        original_query=observed, observed_query=observed)['frozen']
    parsed = {'decisions': [], 'ranked_proposal_indices': [0, 1]}
    for i in range(2):
        selected = [0, 1] if i == 0 else [0]
        parsed['decisions'].append({'proposal_index': i, 'verdict': 'accept', 'reason': 'supported',
            'selected_check_indices': selected, 'safe_check_subset': True,
            'blocking_checks': [2] if i == 0 else [],
            'note': 'Unsafe operation omitted; retained checks require no removed prerequisite.' if i == 0 else '',
            'proofs': [{'source_index': i, 'supports_cause': True,
                'observation_units': frozen['observed_ids'], 'source_units': frozen['source_sets'][i],
                'target_units': frozen['target_sets'][i], 'applicability': 'same_target',
                'support_type': 'bounded_inference', 'check_indices': selected}]})
    return frozen, records, observed, parsed


class RootResponseQualityTests(unittest.TestCase):
    def setUp(self):
        self.frozen, self.records, self.observed, self.parsed = fixture()

    def validate(self, parsed=None, frozen=None):
        return quality.validate_review(parsed=self.parsed if parsed is None else parsed,
            frozen=self.frozen if frozen is None else frozen, records=self.records,
            observed_query=self.observed)

    def test_explicit_review_selection_removes_only_rejected_action(self):
        frozen_before = deepcopy(self.frozen)
        parsed_before = deepcopy(self.parsed)
        result = self.validate()
        self.assertEqual(result['causes'][0]['checks'], [c['text'] for c in self.frozen['proposals'][0]['checks'][:2]])
        self.assertEqual(result['causes'][0]['cause'], self.frozen['proposals'][0]['cause'])
        self.assertEqual(result['causes'][0]['why'], self.frozen['proposals'][0]['why'])
        self.assertNotIn('guards open', str(result['causes']))
        self.assertEqual(self.frozen, frozen_before)
        self.assertEqual(self.parsed, parsed_before)
        audit = result['summary']
        self.assertFalse(audit['immutable_proposals'])
        self.assertTrue(audit['immutable_retained_checks'])
        self.assertTrue(audit['all_input_checks_decided'])
        self.assertTrue(audit['all_retained_checks_covered'])
        self.assertFalse(audit['semantic_truth_verified_by_code'])
        self.assertEqual(audit['original_registry_fingerprint'], frozen_before['fingerprint'])
        self.assertNotEqual(audit['selected_registry_fingerprint'], frozen_before['fingerprint'])
        self.assertEqual(audit['check_selection'][0]['blocking_checks'], [2])

    def test_reviewer_ranking_orders_approved_causes_without_rewriting(self):
        self.parsed['ranked_proposal_indices'] = [1, 0]
        result = self.validate()
        self.assertEqual([c['cause'] for c in result['causes']], [self.frozen['proposals'][i]['cause'] for i in [1, 0]])
        self.assertEqual([c['rank'] for c in result['causes']], [1, 2])
        self.assertEqual(result['citation_ids'], ['manual-filter', 'manual-feed'])

    def test_noncontiguous_selection_proof_audit_uses_original_indices(self):
        # A separate fixture where only the second immutable observation is approved.
        decision = self.parsed['decisions'][0]
        decision.update(selected_check_indices=[1], blocking_checks=[0, 2])
        decision['proofs'][0]['check_indices'] = [1]
        result = self.validate()
        self.assertEqual(result['causes'][0]['checks'], [self.frozen['proposals'][0]['checks'][1]['text']])
        self.assertEqual(result['summary']['support_proofs'][0]['decision']['check_indices'], [1])

    def test_check_partition_cannot_hide_an_unreviewed_check(self):
        for selected, blocked in [([0], [2]), ([0, 1], [1, 2])]:
            bad = deepcopy(self.parsed)
            bad['decisions'][0].update(selected_check_indices=selected, blocking_checks=blocked)
            with self.subTest(selected=selected), self.assertRaisesRegex(refs.ReferenceError, 'incomplete_check_partition'):
                self.validate(bad)

    def test_check_indices_cannot_duplicate_reorder_or_include_boolean(self):
        for selected in ([1, 0], [0, 0, 1], [0, True], [0, 3]):
            bad = deepcopy(self.parsed)
            bad['decisions'][0]['selected_check_indices'] = selected
            with self.subTest(selected=selected), self.assertRaisesRegex(refs.ReferenceError, 'invalid_check_selection_indices'):
                self.validate(bad)

    def test_omitted_safety_prerequisite_prevents_acceptance_when_reviewer_flags_subset(self):
        self.parsed['decisions'][0]['safe_check_subset'] = False
        with self.assertRaisesRegex(refs.ReferenceError, 'check_subset_not_safe'):
            self.validate()

    def test_no_proof_may_launder_a_discarded_check(self):
        self.parsed['decisions'][0]['proofs'][0]['check_indices'] = [0, 1, 2]
        with self.assertRaisesRegex(refs.ReferenceError, 'invalid_check_selection_indices'):
            self.validate()

    def test_retained_checks_still_need_complete_evidence_coverage(self):
        self.parsed['decisions'][0]['proofs'][0]['check_indices'] = [0]
        with self.assertRaisesRegex(refs.ReferenceError, 'not_all_checks_supported'):
            self.validate()

    def test_sensible_check_cannot_replace_missing_causal_proof(self):
        proof = self.parsed['decisions'][0]['proofs'][0]
        proof.update(supports_cause=False, observation_units=[], support_type='documented_check')
        with self.assertRaisesRegex(refs.ReferenceError, 'mechanism_not_supported'):
            self.validate()

    def test_proofs_still_cannot_borrow_another_source(self):
        self.parsed['decisions'][0]['proofs'][0]['source_units'] = self.frozen['source_sets'][1]
        with self.assertRaisesRegex(refs.ReferenceError, 'reference_outside_authorized_set'):
            self.validate()

    def test_altered_original_draft_is_detected_before_derived_manifest(self):
        changed = deepcopy(self.frozen)
        changed['proposals'][0]['checks'][0]['text'] = 'Tampered action'
        with self.assertRaisesRegex(refs.ReferenceError, 'manifest_altered'):
            self.validate(frozen=changed)

    def test_cause_without_safe_checks_is_only_allowed_with_causal_proof(self):
        decision = self.parsed['decisions'][0]
        decision.update(selected_check_indices=[], blocking_checks=[0, 1, 2])
        decision['proofs'][0]['check_indices'] = []
        result = self.validate()
        self.assertEqual(result['causes'][0]['checks'], [])
        self.assertTrue(result['causes'][0]['citations'])

    def test_rejected_cause_never_returns_even_if_in_rank_list(self):
        decision = self.parsed['decisions'][0]
        decision.update(verdict='reject', reason='unsupported', selected_check_indices=[],
            safe_check_subset=False, proofs=[], note='No applicable mechanism.')
        with self.assertRaisesRegex(refs.ReferenceError, 'accepted_ranking_mismatch'):
            self.validate()
        self.parsed['ranked_proposal_indices'] = [1]
        result = self.validate()
        self.assertEqual(len(result['causes']), 1)
        self.assertEqual(result['causes'][0]['citations'], ['manual-filter'])

    def test_rank_must_be_exact_unique_approved_index_permutation(self):
        for order in ([0], [0, 0], [0, 2], [False, 1], [0, 1, 2]):
            bad = deepcopy(self.parsed)
            bad['ranked_proposal_indices'] = order
            with self.subTest(order=order), self.assertRaisesRegex(refs.ReferenceError, 'accepted_ranking_mismatch'):
                self.validate(bad)

    def test_schema_does_not_widen_reference_sets_or_modify_shared_schema(self):
        before = refs.schema(self.frozen)
        after = quality.review_schema(self.frozen)
        self.assertEqual(before, refs.schema(self.frozen))
        self.assertEqual(before['schema']['$defs'], after['schema']['$defs'])
        branch = after['schema']['properties']['decisions']['items']['anyOf'][0]
        self.assertEqual(branch['properties']['selected_check_indices']['items']['enum'], [0, 1, 2])
        self.assertIn('safe_check_subset', branch['required'])
        self.assertEqual(after['schema']['properties']['ranked_proposal_indices']['items']['enum'], [0, 1])

    def test_instructions_do_not_contradict_subset_safety_contract(self):
        instruction = quality.review_instruction()
        self.assertIn('COMPLETE original draft', instruction)
        self.assertIn('NEVER drop a safety prerequisite', instruction)
        self.assertNotIn('Retain a whole supported', instruction)
        self.assertNotIn('EVERY check are supported', instruction)
        self.assertNotIn('check still rejects its', instruction)
        self.assertIn('untrusted data', instruction)

    def test_rejected_proposal_failure_is_truthful_and_contains_no_actions(self):
        for language in ('it', 'en'):
            with self.subTest(language=language):
                out = quality.rejected_response(response={'possible_causes': [{'checks': ['unsafe action']}], 'status': 'answered'},
                    meta={'root_review_decisions': {'accepted_causes': 0}}, language=language, result_code='TECHNICAL_ERROR')
                self.assertFalse(out['ok'])
                self.assertEqual(out['status'], 'error')
                self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
                self.assertEqual(out['possible_causes'], [])
                self.assertEqual(out['citations'], [])
                self.assertFalse(out['meta']['cacheable'])
                self.assertFalse(out['meta']['root_response_quality']['validated_diagnosis'])
                self.assertNotIn('unsafe action', str(out))



class RootReviewMainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frozen, self.records, self.observed, self.parsed = fixture()
        self.calls = []
        self.no_evidence_calls = []
        self.raw_records = self.records
        self.budget = SimpleNamespace(llm_calls=0, max_llm_calls=8, call_log=[], remaining=lambda: 50.0)
        packet = {'sources': [{'citation_id': r['citation_id'], 'text': r['text'], 'context_ids': []}
                              for r in self.records], 'document_contexts': []}
        self.request = SimpleNamespace(company_id='fixture-company', machine_id='fixture-machine',
            ai_scope='machine', query=self.observed, response_language='en', max_causes=3,
            debug=False)
        self.decision = SimpleNamespace(missing_information=['Pressure has not been read.'])
        self.draft = {'ok': True, 'status': 'answered', 'possible_causes': [
            {'cause': p['cause'], 'why': p['why'], 'checks': [c['text'] for c in p['checks']]}
            for p in self.frozen['proposals']], 'meta': {'accounting_marker': 'preserved'}}
        self.retrieval = {'citations': [{'citation_id': r['citation_id']} for r in self.records]}

        def model(messages, **kwargs):
            self.calls.append((messages, kwargs))
            return deepcopy(self.parsed), 'offline-fixture-model'

        def no_evidence(*args):
            self.no_evidence_calls.append(args)
            return {'ok': True, 'status': 'no_sources', 'result_code': 'NO_MACHINE_EVIDENCE'}

        path = Path(__file__).resolve().parents[1] / 'main.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_assistant_core_adjudicate_root_cause_grounded')
        namespace = {
            'AssistantCoreRequest': object, 'AssistantCoreDecision': object,
            '_v13_current_budget': lambda: self.budget,
            '_retrieval_review_packet': SimpleNamespace(POLICY_VERSION='fixture-packet',
                ReviewPacketError=ValueError, build_review_packet=lambda **kwargs: {
                    'validator_records': self.records, 'model_packet': packet,
                    'summary': {'policy_version': 'fixture-packet'}}),
            '_retrieval_review_references': refs, '_retrieval_review_decisions': review_decisions,
            '_retrieval_diagnostic_query': SimpleNamespace(reasoning_packet=lambda profile: {}),
            '_retrieval_diagnostic_sources': SimpleNamespace(CAUSAL_GROUNDING_POLICY='fixture-causal-policy'),
            '_assistant_core_diagnostic_query_profile': lambda request: None,
            '_assistant_core_root_applicability_records': lambda request, candidates: self.raw_records,
            '_assistant_core_retrieval_query': lambda request: self.observed,
            '_assistant_core_scope_value': lambda *args: None,
            '_assistant_core_build_no_evidence': no_evidence,
            '_v13_json_models': model, '_V13BudgetExceeded': type('BudgetExceeded', (Exception,), {}),
            '_sanitize_citations_for_response': lambda citations, **kwargs: citations,
            '_build_rg_links': lambda *args: [],
            '_unique_non_empty_strings': lambda values, limit: list(dict.fromkeys(values))[:limit],
            'COMPANY_GENERAL_MACHINE_SENTINEL': 'GENERAL', 'V13_FAST_TIMEOUT_SECONDS': 20,
            'V13_FAST_MODEL': 'fixture-model', 'V13_FAST_EFFORT': 'low',
            'V13_FAST_MAX_OUTPUT_TOKENS': 3400, 'RESULT_TIMEOUT': 'TIMEOUT',
            'RESULT_TECHNICAL_ERROR': 'TECHNICAL_ERROR', 'json': json, 'time_module': time,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        self.run_review = namespace[function.name]

    def run_response(self):
        return self.run_review(request=self.request, decision=self.decision,
            retrieval=self.retrieval, response=self.draft)

    def test_real_root_stage_uses_one_call_and_keeps_full_original_draft_in_review(self):
        out = self.run_response()
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(len(self.calls), 1)
        messages, arguments = self.calls[0]
        self.assertIn('Adjust the clamp under load with guards open.', messages[1]['content'])
        self.assertIn('safe_check_subset', arguments['json_schema']['schema']['properties']['decisions']['items']['anyOf'][0]['properties'])
        self.assertEqual(arguments['purpose'], 'assistant_core_root_cause_adjudicator')
        self.assertNotIn('guards open', str(out['possible_causes']))
        self.assertNotIn('guards open', str(out['recommended_next_checks']))
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertEqual(self.no_evidence_calls, [])

    def test_rejected_draft_is_not_reported_as_missing_sources(self):
        for d in self.parsed['decisions']:
            d.update(verdict='reject', reason='unsupported', selected_check_indices=[],
                safe_check_subset=False, proofs=[], note='Unsupported explanation.')
        self.parsed['ranked_proposal_indices'] = []
        out = self.run_response()
        self.assertFalse(out['ok'])
        self.assertEqual(out['status'], 'error')
        self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(out['meta']['root_response_quality']['outcome'], 'review_rejected_proposals')
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(self.no_evidence_calls, [])
        self.assertEqual(len(self.calls), 1)

    def test_genuinely_absent_evidence_preserves_no_sources_without_model_call(self):
        self.raw_records = []
        out = self.run_response()
        self.assertEqual(out['status'], 'no_sources')
        self.assertEqual(out['result_code'], 'NO_MACHINE_EVIDENCE')
        self.assertEqual(len(self.no_evidence_calls), 1)
        self.assertEqual(self.calls, [])

    def test_no_draft_does_not_create_a_new_review_or_claim_sources_exist(self):
        self.draft = {'ok': True, 'status': 'no_sources', 'possible_causes': []}
        out = self.run_response()
        self.assertEqual(out, self.draft)
        self.assertEqual(self.calls, [])

    def test_invalid_selection_fails_closed_with_accounting_and_no_retry(self):
        self.parsed['decisions'][0]['proofs'][0]['check_indices'] = [0, 1, 2]
        out = self.run_response()
        self.assertFalse(out['ok'])
        self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(out['meta']['accounting_marker'], 'preserved')
        self.assertEqual(len(self.calls), 1)
        self.assertIn('invalid_check_selection_indices', out['meta']['root_review_execution']['validation_error'])

    def test_real_root_stage_preserves_supported_hypothesis_without_executable_checks(self):
        decision = self.parsed['decisions'][0]
        decision.update(selected_check_indices=[], blocking_checks=[0, 1, 2],
            note='Only mechanism supported; no independently safe operation established.')
        decision['proofs'][0]['check_indices'] = []
        self.parsed['ranked_proposal_indices'] = [0, 1]
        out = self.run_response()
        self.assertTrue(out['ok'])
        self.assertEqual(out['possible_causes'][0]['cause'], self.draft['possible_causes'][0]['cause'])
        self.assertEqual(out['possible_causes'][0]['why'], self.draft['possible_causes'][0]['why'])
        self.assertEqual(out['possible_causes'][0]['checks'], [])
        self.assertTrue(out['possible_causes'][0]['citations'])
        self.assertEqual(out['recommended_next_checks'], self.draft['possible_causes'][1]['checks'])
        self.assertEqual(len(self.calls), 1)

    def test_retrospective_unknown_question_is_preserved_without_synthesized_action(self):
        clarification = ('If a pressure reading is already recorded, what was it? '
                         'Otherwise leave the pressure unknown; do not start a new cycle to obtain it.')
        self.draft['possible_causes'][0]['checks'] = [clarification]
        decision = self.parsed['decisions'][0]
        decision.update(selected_check_indices=[0], blocking_checks=[], note='')
        decision['proofs'][0]['check_indices'] = [0]
        out = self.run_response()
        self.assertTrue(out['ok'])
        self.assertEqual(out['possible_causes'][0]['checks'], [clarification])
        self.assertEqual(out['possible_causes'][0]['why'], self.draft['possible_causes'][0]['why'])
        self.assertIn('pressure is unknown', out['possible_causes'][0]['why'])
        self.assertEqual(len(self.calls), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
