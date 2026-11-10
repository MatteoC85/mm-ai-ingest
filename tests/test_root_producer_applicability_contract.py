"""Private Root writer contract and immutable actual-R13 failure replay, offline.

Controlled reviewer replies exercise source-bound validation, not model truth or
future routing quality. The original capture has no full provider reply; its
retained verdict is reconstructed without changing the original proposal/frame.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality

WORK = next((p for p in (P.parent, P.parent.parent)
             if (p / 'phase6-final-candidate-r13-20261110').is_dir()), None)
RECEIPT = (WORK / 'phase6-final-candidate-r13-20261110/live-client/receipts/R13-A-ROOT-01.json'
           if WORK is not None else None)
RECEIPT_SHA = 'a5a952953c8c3d01eaf46da21ca8cea59732dc04eb86b4a1ac6ded2d63971bcc'
SOURCE = (P / 'main.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def functions(names, namespace):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(P / 'main.py'), 'exec'), namespace)
    return namespace


def unique(values, limit):
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))[:limit]


def captured():
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA
    return json.loads(raw)


def rejection(frozen, reason='unknown_target', note='Retained reviewer rejection.'):
    return {'decisions': [{'proposal_index': i, 'verdict': 'reject', 'reason': reason,
        'selected_check_indices': [], 'safe_check_subset': False,
        'blocking_checks': [c['check_index'] for c in p['checks']],
        'selected_why_indices': [], 'omitted_why_indices': [
            u['why_index'] for u in quality.why_units(p['why'])],
        'safe_why_subset': False, 'proofs': [], 'note': note}
        for i, p in enumerate(frozen['proposals'])], 'ranked_proposal_indices': []}


def packet_for(records):
    # Every original ownership fragment is preserved, with local exact offsets.
    contexts = []
    by_text = {}
    sources = []
    for row in records:
        links = []
        for body in row.get('ownership_fragments', []):
            if body not in by_text:
                cid = 'ctx' + str(len(contexts) + 1)
                by_text[body] = cid
                contexts.append({'id': cid, 'fragments': [
                    {'start': 0, 'end': len(body), 'text': body}]})
            links.append(by_text[body])
        sources.append({'citation_id': row['citation_id'], 'text': row['text'],
                        'context_ids': links})
    return {'sources': sources, 'document_contexts': contexts}


def freeze(draft, records, query, observed):
    manifest = review_decisions.manifest(draft['possible_causes'], records, max_causes=3)
    return refs.prepare(packet=packet_for(records), proposal_manifest=manifest,
        records=records, original_query=query, observed_query=observed)['frozen']


def accepted(frozen):
    # Controlled semantic adjudication: it claims only bounded conditional support.
    p = frozen['proposals'][0]
    return {'decisions': [{'proposal_index': 0, 'verdict': 'accept', 'reason': 'supported',
        'selected_check_indices': list(range(len(p['checks']))), 'safe_check_subset': True,
        'blocking_checks': [], 'selected_why_indices': [
            u['why_index'] for u in quality.why_units(p['why'])],
        'omitted_why_indices': [], 'safe_why_subset': True, 'note': '',
        'proofs': [{'source_index': 0, 'supports_cause': True,
            'observation_units': frozen['observed_ids'],
            'source_units': frozen['source_sets'][0], 'target_units': frozen['source_sets'][0],
            'applicability': 'same_target', 'support_type': 'bounded_inference',
            'check_indices': list(range(len(p['checks'])))}]}],
        'ranked_proposal_indices': [0]}


@unittest.skipUnless(RECEIPT is not None and RECEIPT.is_file(),
                     "private immutable R13 replay receipt is unavailable")
class ProducerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = captured()
        cls.meta = cls.receipt['response']['meta']
        cls.fixture = cls.meta['root_review_capture']['fixture']
        cls.original = cls.meta['root_review_decisions']['proposals'][0]
        cls.records = cls.fixture['validator_records']

    def setUp(self):
        self.socket_guard = patch.object(socket, 'socket', side_effect=AssertionError('NETWORK_FORBIDDEN'))
        self.socket_guard.start()
        self.addCleanup(self.socket_guard.stop)
        p = self.original
        self.typed = {'problem_summary': self.fixture['observed_query'],
            'possible_causes': [{'rank': 1, 'cause': p['cause'], 'why': p['why'],
                'citations': [self.records[0]['citation_id']],
                'applicability': {'status': 'source_scoped_unverified',
                    'statement': ('Questa ipotesi riguarda la configurazione documentata nel manuale citato; '
                                  'la sua applicabilità alla macchina attuale non è verificata.')},
                'checks': [{'intent': 'already_available_observation_or_record',
                            'text': p['checks'][0]['text']},
                           {'intent': 'new_observation', 'text': p['checks'][1]['text']}]}],
            'recommended_next_checks': []}

    def normalize(self, value=None):
        return quality.normalize_producer_result(self.typed if value is None else value, max_causes=3)

    def writer(self, result, opt_in=True):
        calls = []
        grounded = []
        fallbacks = []
        ns = functions({'_root_cause_response_schema', '_ground_root_cause_result'}, {
            '_unique_non_empty_strings': unique,
            '_extract_citation_ids_from_root_cause_json': lambda r: unique(
                [c for row in r.get('possible_causes', []) for c in row.get('citations', [])], 20),
        })
        real_ground = ns['_ground_root_cause_result']
        def ground(**kwargs):
            grounded.append(deepcopy(kwargs['result']))
            return real_ground(**kwargs)
        def model(messages, **kwargs):
            calls.append((deepcopy(messages), deepcopy(kwargs)))
            return deepcopy(result), 'gpt-5.6-terra'
        packet = {'sources_block': '\n\n'.join(row['text'] for row in self.records),
                  'summary': {'source_count': len(self.records)}}
        ns.update(_v13_root_cause_model=lambda *a: ('gpt-5.6-terra', 'none', 'standard'),
            V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE=8, V13_FAST_MODEL='gpt-5.6-terra',
            V13_HEAVY_MODEL='gpt-5.6-sol', V13_FAST_CONTEXT_CHARS=24000,
            V13_HEAVY_CONTEXT_CHARS=38000, V13_FAST_TIMEOUT_SECONDS=35,
            V13_HEAVY_TIMEOUT_SECONDS=45, V13_FAST_MAX_OUTPUT_TOKENS=3200,
            V13_HEAVY_MAX_OUTPUT_TOKENS=6000,
            _dedup_text_values=lambda xs, limit: unique(xs, limit),
            _retrieval_source_management=SimpleNamespace(
                V13SourcesBlockRuntime=lambda **kw: kw,
                v13_root_sources_packet=lambda *a, **kw: deepcopy(packet)),
            _source_type_from_document_id=lambda x: 'document', _v13_candidate_text=lambda x: x['text'],
            _retrieval_diagnostic_query=SimpleNamespace(POLICY_VERSION='fixture'),
            _v13_assurance_prompt_block=lambda r: '', _v13_json_models=model,
            _V13BudgetExceeded=type('BudgetExceeded', (Exception,), {}),
            _ground_root_cause_result=ground,
            _compact_root_cause_result_citations_by_family=lambda **kw: (kw['result'], kw['citations']),
            _lock_root_cause_result=lambda result, citations, **kw: (result, citations),
            _v13_root_fallback_from_evidence=lambda **kw: fallbacks.append(kw) or ({}, []),
            _sanitize_citations_for_response=lambda citations, **kw: citations,
            _build_rg_links=lambda *a: [], json=json)
        functions({'_v13_generate_root_cause_response'}, ns)
        out = ns['_v13_generate_root_cause_response'](q=self.fixture['original_query'],
            company_id=self.fixture['request_scope']['company_id'] if 'request_scope' in self.fixture else 'fixture-company',
            response_language='it', top_k=6, max_causes=3,
            retrieval={'citations': deepcopy(self.records)}, debug=False,
            root_producer_applicability_contract=opt_in)
        return out, calls, grounded, fallbacks

    def test_schema_clone_leaves_default_public_and_smart_schema_untouched(self):
        ns = functions({'_root_cause_response_schema'}, {})
        original = ns['_root_cause_response_schema'](3)
        before = deepcopy(original)
        clone = quality.producer_schema(original)
        self.assertEqual(original, before)
        self.assertEqual(original['schema']['properties']['possible_causes']['items']['properties']['checks']['items'],
                         {'type': 'string'})
        self.assertNotIn('applicability', original['schema']['properties']['possible_causes']['items']['properties'])
        self.assertEqual(clone['schema']['properties']['recommended_next_checks']['maxItems'], 0)
        # The opt-in is at the owned actual Root caller, never a request metadata flag.
        caller = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and
                      n.name == '_assistant_core_synthesize_root_cause')
        opted = [k.value for n in ast.walk(caller) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == '_v13_generate_root_cause_response'
                 for k in n.keywords if k.arg == 'root_producer_applicability_contract']
        self.assertEqual(len(opted), 1)
        expression = compile(ast.Expression(opted[0]), '<owned-root-mode-guard>', 'eval')
        for requested, effective, expected in [('root_cause', 'root_cause', True),
                ('ask', 'root_cause', False), ('smart_diagnostic', 'root_cause', False),
                ('root_cause', 'ask', False)]:
            self.assertEqual(eval(expression, {'request': SimpleNamespace(requested_mode=requested),
                'decision': SimpleNamespace(effective_mode=effective), 'MODE_ROOT_CAUSE': 'root_cause'}), expected)

    def test_actual_writer_renders_boundary_before_real_grounding_and_one_existing_model_call(self):
        before = deepcopy(self.typed)
        out, calls, grounded, fallbacks = self.writer(self.typed)
        self.assertEqual(self.typed, before)
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['purpose'], 'root_cause_final_reasoning')
        self.assertEqual(calls[0][1]['max_output_tokens'], 3200)
        self.assertEqual(calls[0][1]['models'], ['gpt-5.6-terra'])
        self.assertEqual(calls[0][1]['effort'], 'none')
        self.assertEqual(calls[0][1]['timeout'], 35)
        self.assertIn(self.records[0]['text'], calls[0][0][1]['content'])
        normalized = self.normalize()
        self.assertEqual(grounded, [normalized])
        self.assertEqual(out['possible_causes'][0]['why'], normalized['possible_causes'][0]['why'])
        self.assertEqual(out['possible_causes'][0]['checks'], [self.original['checks'][0]['text']])
        self.assertEqual(out['recommended_next_checks'], [self.original['checks'][0]['text']])
        self.assertNotIn('applicability', out['possible_causes'][0])
        self.assertNotIn(self.original['checks'][1]['text'], str(out))
        self.assertEqual(fallbacks, [])

    def test_missing_or_malformed_opted_in_contract_stops_before_grounding_or_fallback(self):
        cases = []
        legacy = self.normalize()
        cases.append(legacy)
        for change in ('missing_statement', 'boolean_status', 'extra_field', 'untyped_check', 'untyped_next', 'bool_rank'):
            bad = deepcopy(self.typed)
            row = bad['possible_causes'][0]
            if change == 'missing_statement': del row['applicability']['statement']
            elif change == 'boolean_status': row['applicability']['status'] = True
            elif change == 'extra_field': row['applicability']['verified'] = True
            elif change == 'untyped_check': row['checks'][0] = row['checks'][0]['text']
            elif change == 'untyped_next': bad['recommended_next_checks'] = ['Start a cycle']
            elif change == 'bool_rank': row['rank'] = True
            cases.append(bad)
        for bad in cases:
            with self.subTest(shape=repr(bad)[:70]):
                out, calls, grounded, fallbacks = self.writer(bad)
                self.assertFalse(out['ok'])
                self.assertEqual(out['result_code'], 'TECHNICAL_ERROR')
                self.assertEqual(out['meta']['root_producer_contract']['outcome'], 'invalid_producer_contract')
                self.assertEqual(len(calls), 1)
                self.assertEqual(grounded, [])
                self.assertEqual(fallbacks, [])

    def test_default_legacy_writer_still_accepts_original_public_shape(self):
        legacy = self.normalize()
        out, calls, grounded, fallbacks = self.writer(legacy, opt_in=False)
        self.assertEqual(out['status'], 'answered')
        self.assertEqual(calls[0][1]['json_schema']['name'], 'root_cause_finder_v1')
        self.assertEqual(grounded, [legacy])
        self.assertEqual(fallbacks, [])

    def test_actual_recorded_r13_proposal_and_rejection_stay_rejected_unchanged(self):
        frozen = self.fixture['frozen']
        before = deepcopy(frozen)
        verdict = self.meta['root_review_decisions']['verdicts'][0]
        # Raw provider reply was not captured; reconstruct ONLY its retained verdict.
        self.assertFalse(self.meta['root_review_capture']['complete'])
        parsed = rejection(frozen, verdict['reason'], verdict['note'])
        out = quality.validate_review(parsed=parsed, frozen=frozen, records=self.records,
                                      observed_query=self.fixture['observed_query'])
        self.assertEqual(out['causes'], [])
        self.assertEqual(out['summary']['verdicts'][0]['reason'], 'unknown_target')
        self.assertEqual(frozen, before)
        self.assertEqual(frozen['proposals'][0]['why'], self.original['why'])

    def test_qualified_conditional_alternative_record_only_survives_controlled_source_bound_review(self):
        out, calls, _, _ = self.writer(self.typed)
        frozen = freeze(out, self.records, self.fixture['original_query'], self.fixture['observed_query'])
        reviewed = quality.validate_review(parsed=accepted(frozen), frozen=frozen,
            records=self.records, observed_query=self.fixture['observed_query'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(reviewed['causes']), 1)
        self.assertEqual(reviewed['causes'][0]['why'], out['possible_causes'][0]['why'])
        self.assertEqual(reviewed['causes'][0]['checks'], [self.original['checks'][0]['text']])
        self.assertFalse(reviewed['summary']['semantic_truth_verified_by_code'])
        self.assertIn('applicabilità', reviewed['causes'][0]['why'])
        self.assertIn('Rimane da verificare', reviewed['causes'][0]['why'])

    def test_unsafe_check_mislabel_and_forged_established_status_never_bypass_review(self):
        for forged_status in (False, True):
            bad = deepcopy(self.typed)
            row = bad['possible_causes'][0]
            row['checks'][1]['intent'] = 'already_available_observation_or_record'
            if forged_status:
                row['applicability'] = {'status': 'established_by_sources_and_observations',
                                       'statement': 'Questa configurazione è certamente installata.'}
            draft = self.normalize(bad)
            self.assertEqual(len(draft['possible_causes'][0]['checks']), 2)
            frozen = freeze(draft, self.records, self.fixture['original_query'], self.fixture['observed_query'])
            reason = 'unknown_target' if forged_status else 'unsupported_check'
            reviewed = quality.validate_review(parsed=rejection(frozen, reason), frozen=frozen,
                records=self.records, observed_query=self.fixture['observed_query'])
            self.assertEqual(reviewed['causes'], [])
            self.assertEqual(reviewed['summary']['verdicts'][0]['reason'], reason)

    def test_wrong_model_foreign_or_changed_source_never_gets_proof_from_status(self):
        draft = self.normalize()
        frozen = freeze(draft, self.records, self.fixture['original_query'], self.fixture['observed_query'])
        wrong = quality.validate_review(parsed=rejection(frozen, 'wrong_target'), frozen=frozen,
            records=self.records, observed_query=self.fixture['observed_query'])
        self.assertEqual(wrong['causes'], [])
        bad = accepted(frozen)
        bad['decisions'][0]['proofs'][0]['source_units'] = [999999]
        with self.assertRaises(refs.ReferenceError):
            quality.validate_review(parsed=bad, frozen=frozen, records=self.records,
                                    observed_query=self.fixture['observed_query'])
        records = deepcopy(self.records)
        records[0]['text'] += '\nDifferent installed model.'
        with self.assertRaises(refs.ReferenceError):
            quality.validate_review(parsed=accepted(frozen), frozen=frozen, records=records,
                                    observed_query=self.fixture['observed_query'])

    def test_foreign_citation_cannot_fall_back_around_the_opted_in_contract(self):
        bad = deepcopy(self.typed)
        bad['possible_causes'][0]['citations'] = ['foreign-company-document']
        out, calls, grounded, fallbacks = self.writer(bad)
        self.assertFalse(out['ok'])
        self.assertEqual(out['meta']['root_producer_contract']['reason'], 'no_grounded_producer_proposals')
        self.assertEqual(out['possible_causes'], [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(grounded), 1)
        self.assertEqual(fallbacks, [])

    def test_known_incompatibility_does_not_become_a_conditional_cause_or_legacy_fallback(self):
        bad = deepcopy(self.typed)
        bad['possible_causes'][0]['applicability']['status'] = 'known_incompatible'
        out, calls, grounded, fallbacks = self.writer(bad)
        self.assertFalse(out['ok'])
        self.assertEqual(out['meta']['root_producer_contract']['reason'], 'no_admissible_producer_proposals')
        self.assertEqual(len(calls), 1)
        self.assertEqual(grounded, [])
        self.assertEqual(fallbacks, [])

    def test_established_status_does_not_remove_documented_checks_and_postreview_change_fails(self):
        raw = deepcopy(self.typed)
        raw['possible_causes'][0]['applicability']['status'] = 'established_by_sources_and_observations'
        draft = self.normalize(raw)
        self.assertEqual(len(draft['possible_causes'][0]['checks']), 2)
        frozen = freeze(draft, self.records, self.fixture['original_query'], self.fixture['observed_query'])
        parsed = accepted(frozen)
        frozen['proposals'][0]['why'] += ' Appended after review.'
        with self.assertRaisesRegex(refs.ReferenceError, 'manifest_altered'):
            quality.validate_review(parsed=parsed, frozen=frozen, records=self.records,
                                    observed_query=self.fixture['observed_query'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
