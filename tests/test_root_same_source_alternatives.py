"""Distinct proposals may share a source; the Root reviewer still decides support."""
import ast
from copy import deepcopy
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import test_root_fallback_fields as integration_fixture
import test_response_visual_hierarchy as visual_fixture
from machinemind.presentation import responses
from machinemind.retrieval import source_management


def actual_functions(namespace):
    names = {'_lock_root_cause_result', '_normalized_cause_label_key',
             '_ground_root_cause_result', '_compact_root_cause_result_citations_by_family',
             '_extract_citation_ids_from_root_cause_json'}
    nodes = [n for n in ast.parse((HERE.parent / 'main.py').read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    rt = SimpleNamespace(_extract_section_from_text=lambda text: '',
        _normalize_unicode_advanced=str,
        _source_type_from_document_id=lambda identifier: identifier.split(':')[0], re=re)
    namespace.update(re=re, _normalize_unicode_advanced=str,
        _stable_evidence_family_key=lambda row: source_management.stable_evidence_family_key(row, runtime=rt),
        _stable_evidence_set_key=lambda row: source_management.stable_evidence_set_key(row, runtime=rt),
        _root_cause_evidence_family_key=lambda row: source_management.stable_evidence_family_key(row, runtime=rt),
        _lock_final_citations=lambda **kw: kw['selected_citations'],
        _candidate_order_key=lambda row: row['citation_id'], ROOT_CAUSE_SET_LOCK_DELTA=.2)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-root-grounding-and-lock>', 'exec'), namespace)
    return namespace


class SameSourceAlternativesTests(unittest.TestCase):
    def setUp(self):
        self.fixture = integration_fixture.RootFallbackIntegrationTests()
        self.fixture.setUp()
        self.ns = actual_functions(self.fixture.ns)
        self.source = {'citation_id': 'ps:synthetic:p1', 'bubble_document_id': 'ps:synthetic',
            'source_type': 'ps', 'page_from': 1, 'similarity': .8,
            'assistant_core_root_viable': True,
            'chunk_full': 'A diagnostic alarm can result from an interrupted sensor output or an open safety feedback circuit. Neither is confirmed by the alarm alone.'}
        self.causes = [self.cause('Interrupted sensor output'), self.cause('Open safety feedback circuit')]

    def cause(self, label):
        return {'rank': 1, 'cause': label, 'why': 'A documented possible mechanism; its current state is unknown.',
                'checks': [], 'citations': [self.source['citation_id']]}

    def lock(self, causes=None, **kwargs):
        result = {'possible_causes': deepcopy(self.causes if causes is None else causes)}
        return self.ns['_lock_root_cause_result'](result, [deepcopy(self.source)], max_causes=kwargs.pop('max_causes', 3), **kwargs)

    def test_two_distinct_mechanisms_from_one_record_reach_review(self):
        result, sources = self.lock(preserve_source_alternatives=True)
        self.assertEqual({c['cause'] for c in result['possible_causes']}, {c['cause'] for c in self.causes})
        self.assertEqual(sources, [self.source])
        self.assertTrue(all(c['citations'] == [self.source['citation_id']] for c in result['possible_causes']))

    def test_default_and_false_keep_legacy_selection_exact(self):
        self.assertEqual(self.lock(), self.lock(preserve_source_alternatives=False))
        self.assertEqual(len(self.lock()[0]['possible_causes']), 1)

    def test_opt_in_is_literal_true_and_global_cause_cap_is_unchanged(self):
        causes = self.causes + [self.cause('Documented configuration mismatch')]
        self.assertEqual(len(self.lock(causes, preserve_source_alternatives=True)[0]['possible_causes']), 3)
        self.assertEqual(len(self.lock(causes, preserve_source_alternatives=True, max_causes=2)[0]['possible_causes']), 2)
        for invalid in ('true', 1, None):
            self.assertEqual(self.lock(preserve_source_alternatives=invalid), self.lock())

    def test_duplicate_label_still_deduplicates_without_inventing_causes(self):
        result, _ = self.lock([self.cause('The open circuit'), self.cause('open circuit')], preserve_source_alternatives=True)
        self.assertEqual(len(result['possible_causes']), 1)
        self.assertEqual(self.lock([], preserve_source_alternatives=True)[0]['possible_causes'], [])

    def test_invalid_citations_are_still_removed_before_lock(self):
        invalid = self.cause('Unbound alternative')
        invalid['citations'] = ['ps:foreign:p1']
        grounded, citations = self.ns['_ground_root_cause_result'](
            result={'possible_causes': self.causes + [invalid]}, citations=[self.source], max_causes=3)
        result, _ = self.ns['_lock_root_cause_result'](grounded, citations, max_causes=3, preserve_source_alternatives=True)
        self.assertEqual(len(result['possible_causes']), 2)
        self.assertNotIn('Unbound alternative', [c['cause'] for c in result['possible_causes']])

    def run_core(self, requested, effective):
        self.fixture.calls.clear()
        self.fixture.reviews.clear()
        def model(messages, **kwargs):
            self.fixture.calls.append((messages, kwargs))
            return {'problem_summary': 'Diagnostic alarm, causes unconfirmed.', 'possible_causes': deepcopy(self.causes)}, 'offline-model'
        self.ns['_v13_json_models'] = model
        request = SimpleNamespace(query='Which documented mechanisms could explain this alarm?',
            company_id='synthetic', response_language='en', requested_mode=requested,
            top_k=6, max_causes=3, debug=False)
        decision = SimpleNamespace(effective_mode=effective, diagnostic_clues=[], diagnostic_exclusions=[], missing_information=[])
        return self.ns['_assistant_core_synthesize_root_cause'](request, {'citations': [deepcopy(self.source)]}, decision)

    def test_real_producer_grounding_lock_and_core_review_handoff(self):
        result = self.run_core('root_cause', 'root_cause')
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(len(self.fixture.reviews), 1)
        proposed = self.fixture.reviews[0]['response']['possible_causes']
        self.assertEqual(len(proposed), 2)
        self.assertEqual(result['possible_causes'], [])
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['review_required'])
        self.assertEqual(self.fixture.calls[0][1]['max_output_tokens'], 3400)
        self.assertEqual(self.fixture.calls[0][1]['timeout'], 26)

    def test_requested_and_effective_root_are_both_required(self):
        for requested, effective in [('ask', 'root_cause'), ('root_cause', 'ask'), ('smart', 'root_cause'), ('root_cause', 'smart')]:
            with self.subTest(requested=requested, effective=effective):
                self.run_core(requested, effective)
                self.assertEqual(len(self.fixture.reviews[0]['response']['possible_causes']), 1)

    def test_rank_label_and_title_are_readable_without_changing_diagnostic_text(self):
        response = {'status': 'answered', 'problem_summary': 'Unconfirmed.', 'possible_causes': deepcopy(self.causes)}
        response['possible_causes'][1]['rank'] = 2
        original = deepcopy(response)
        for language, labels in [('en', ['Most likely cause', 'Possible cause 2']), ('it', ['Causa pi\u00f9 probabile', 'Causa possibile 2'])]:
            rendered = responses.assistant_ui_root_cause_html(response, response_language=language, _runtime=visual_fixture.runtime())
            tree = visual_fixture.Tree(rendered).root
            for label in labels:
                node = next(n for n in tree.find() if n.tag == 'p' and n.text() == label)
                self.assertIn('font-size:14px', node.attrs['style'])
            titles = [n for n in tree.find() if n.tag == 'h3']
            self.assertEqual(len(titles), 2)
            self.assertTrue(all('font-size:16px' in n.attrs['style'] for n in titles))
            for cause in self.causes:
                self.assertIn(cause['why'], tree.text())
        self.assertEqual(response, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
