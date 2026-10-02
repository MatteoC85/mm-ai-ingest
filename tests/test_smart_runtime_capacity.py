"""Real config and START admission seam with typed offline relation fixtures."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from machinemind.retrieval import smart_evidence
import test_smart_context as fixtures


def configured_limit(value=None):
    with patch.dict(os.environ):
        if value is None:
            os.environ.pop('MM_SMART_DIAGNOSTIC_MAX_EVIDENCE_IN_STATE', None)
        else:
            os.environ['MM_SMART_DIAGNOSTIC_MAX_EVIDENCE_IN_STATE'] = value
        spec = importlib.util.spec_from_file_location('_smart_capacity_fixture',
            ROOT / 'machinemind/config/smart_diagnostic_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SMART_DIAGNOSTIC_MAX_EVIDENCE_IN_STATE


class HTTPException(Exception):
    pass


class Budget:
    def remaining(self): return 100
    def ensure_time(self, seconds): pass


class RuntimeCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8-sig'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == '_sd_prepare_start_grounding')
        cls.code = compile(ast.Module(body=[function], type_ignores=[]), 'main.py:START-admission', 'exec')

    def setUp(self):
        self.context = fixtures.ContextTests()
        self.context.setUp()
        self.raw = [fixtures.raw(3), fixtures.raw(6)] + [{
            **fixtures.raw(index), 'citation_id': 'manual:page-%d' % index,
            'bubble_document_id': 'manual-%d' % index, 'source_type': 'manual',
            'source_id': 'manual-%d' % index,
            'chunk_full': 'Complete independent manual record %d, including its precautions.' % index
        } for index in range(10, 16)]
        self.original = deepcopy(self.raw)
        self.context_transform = lambda rows: rows
        self.pop_count = 0

    def run_start(self, limit=None):
        def select(rows, **kwargs):
            try:
                return smart_evidence.select_complete_sources(rows, scope=fixtures.SCOPE,
                    max_items=kwargs['max_items'], allow_raw_snippet=True,
                    relevant_ids=kwargs.get('relevant_ids', ()))
            except ValueError as exc:
                raise HTTPException('EVIDENCE_PACKET_INVALID') from exc

        def build(rows, citations, **kwargs):
            by_id = {row['citation_id']: row for row in rows}
            return smart_evidence.build([by_id[row['citation_id']] for row in citations],
                scope=fixtures.SCOPE, allow_raw_snippet=True)

        def acquire(selected, **kwargs):
            self.context.selected = selected
            rows, metadata = self.context.acquire()
            return self.context_transform(rows), metadata

        def pop(_token): self.pop_count += 1

        env = {
            '_sd_select_complete_citations': select, '_sd_build_grounding_packet': build,
            '_v13_current_budget': lambda: Budget(),
            '_smart_review': types.SimpleNamespace(MAX_REVIEW_SECONDS=30, FINALIZATION_RESERVE_SECONDS=2),
            'V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ROOT_CAUSE': 32,
            'SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_ENABLED': True,
            'SMART_DIAGNOSTIC_MAX_EVIDENCE_IN_STATE': configured_limit() if limit is None else limit,
            '_assistant_core_authority_reader_runtimes': lambda: {},
            '_v13_push_operation_limits': lambda **kwargs: (None, None),
            '_v13_pop_operation_limits': pop,
            '_smart_context': types.SimpleNamespace(acquire=acquire),
            '_sd_grounding_scope': lambda company, machine: fixtures.SCOPE,
            '_smart_evidence': smart_evidence, 'HTTPException': HTTPException,
        }
        exec(self.code, env)
        return env['_sd_prepare_start_grounding'](self.raw,
            company_id=fixtures.SCOPE['company_id'], machine_id=fixtures.SCOPE['machine_id'],
            max_items=smart_evidence.MAX_SELECTED_SOURCES)

    def assert_base_preserved(self, packet):
        expected = {row['citation_id']: row['chunk_full'] for row in self.original}
        actual = {row['citation_id']: row['text'] for row in smart_evidence.review_packet(packet)['validator_records']}
        self.assertEqual({key: actual[key] for key in expected}, expected)
        self.assertEqual(self.raw, self.original)
        self.assertEqual(self.pop_count, 1)

    def test_default_state_admits_context_after_all_eight_base_sources(self):
        citations, packet, meta = self.run_start()
        self.assertEqual(configured_limit(), smart_evidence.MAX_SOURCES)
        self.assertEqual(smart_evidence.MAX_SELECTED_SOURCES, 8)
        self.assertEqual(smart_evidence.MAX_NEW_SOURCES, 3)
        self.assertEqual(meta['reason'], 'context_admitted')
        self.assertEqual((len(citations), len(packet['sources']), len(self.context.calls)), (11, 11, 2))
        self.assertEqual(meta['added_ids'], ['procedure:p:p1-1:smart-context',
            'step:s2:p1-1:smart-context', 'step:s5:p1-1:smart-context'])
        self.assertLessEqual(len(smart_evidence.canonical(packet)), 22000)
        self.assertLessEqual(len(smart_evidence.evidence_block(packet)), 22000)
        self.assert_base_preserved(packet)

    def test_explicit_eight_retains_atomic_omission_without_evicting_base(self):
        citations, packet, meta = self.run_start(configured_limit('8'))
        self.assertEqual((len(citations), len(packet['sources'])), (8, 8))
        self.assertEqual(meta['reason'], 'context_packet_not_admitted')
        self.assertEqual(meta['added_ids'], [])
        self.assertEqual(len(meta['omitted_ids']), 3)
        self.assert_base_preserved(packet)

    def test_override_is_bounded_by_existing_packet_hard_limit(self):
        self.assertEqual(configured_limit('99'), smart_evidence.MAX_SOURCES)
        self.assertEqual(configured_limit('9'), 9)
        self.assertEqual(configured_limit('1'), 1)
        self.assertEqual(configured_limit('0'), 1)
        self.assertEqual(configured_limit('-5'), 1)
        with self.assertRaises(ValueError): configured_limit('not-a-number')

    def test_text_capacity_still_rejects_all_additions_without_clipping(self):
        self.context_transform = lambda rows: [{**row, 'chunk_full': 'Z' * 22000} for row in rows]
        citations, packet, meta = self.run_start()
        self.assertEqual((len(citations), len(packet['sources'])), (8, 8))
        self.assertEqual(meta['reason'], 'context_packet_not_admitted')
        self.assertEqual(meta['added_ids'], [])
        self.assert_base_preserved(packet)

    def test_foreign_context_is_atomically_rejected_before_state_admission(self):
        self.context_transform = lambda rows: [*rows[:-1], {**rows[-1], 'company_id': 'another-company'}]
        citations, packet, meta = self.run_start()
        self.assertEqual((len(citations), len(packet['sources'])), (8, 8))
        self.assertEqual(meta['reason'], 'context_packet_not_admitted')
        self.assertEqual(meta['added_ids'], [])
        self.assert_base_preserved(packet)

    def test_foreign_database_page_is_rejected_by_actual_typed_reader(self):
        self.context.steps[-1] = fixtures.sql_row(8, company='foreign-company')
        citations, packet, meta = self.run_start()
        self.assertEqual((len(citations), len(packet['sources'])), (8, 8))
        self.assertEqual(meta['reason'], 'context_read_unavailable')
        self.assertEqual(meta['added_ids'], [])
        self.assert_base_preserved(packet)


if __name__ == '__main__':
    unittest.main(verbosity=2)
