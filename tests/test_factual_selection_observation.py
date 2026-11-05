"""Request-owned table mode observations; fixture verdicts are not live quality proof."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_source_table_owned as owned
from machinemind.ask import request_completion as completion, source_table_review as tables

NOTE = 'These documented facts answer the requested categories; this is not a complete inventory.'
BOOL_FIELDS = ('source_first_factual_review', 'useful_factual_selection', 'inventory_template_allowed',
               'source_table_catalog_nonempty', 'source_table_catalog_complete', 'selected_records_only')


class FactualSelectionObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass()
        cls.m = owned.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.SourceTableOwnedTests.tearDownClass()

    def fixture(self, cleanup, *, intent='useful', forged=False, query='When are the slide and wiring inspected?'):
        owner = completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company', debug=True), started=time.monotonic())
        cleanup.callback(owner.close)
        def reply_change(reply):
            reply.update(table_record_coverage=[], selection_scope_note=NOTE)
            if forged:
                reply.update(factual_coverage_intent='useful', selected_records_only=True,
                             source_table_catalog_complete=True)
        f = owned.OwnedTableFixture(self, change=reply_change, query=query)
        cleanup.callback(f.close)
        f.decision = replace(f.decision, information_task='numeric_specification', factual_coverage=intent)
        return owner, f

    def observation(self, owner):
        return next(r for r in owner.contract_checks if r['stage'] == 'semantic_review')

    def test_finite_factual_selection_emits_actual_mode_and_preserves_owned_checks(self):
        for question in ('When are the slide and wiring inspected?',
                         'Ogni quanto si controllano la slitta e il cablaggio?'):
            with self.subTest(question=question), ExitStack() as cleanup:
                owner, f = self.fixture(cleanup, query=question)
                limits = (owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd)
                result = f.run()
                self.assertEqual(result['status'], 'answered')
                row = self.observation(owner)
                self.assertEqual(row['factual_coverage_intent'], 'useful')
                self.assertTrue(all(row[key] is True for key in BOOL_FIELDS))
                schema = f.posts[0]['text']['format']['schema']['properties']['table_record_coverage']
                self.assertEqual((schema['type'], schema['maxItems']), ('array', 0))
                self.assertEqual(len(f.posts), 2)
                self.assertEqual(f.budget_ids, [id(owner.budget)] * 2)
                self.assertEqual(limits, (owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd))
                self.assertIn('Monthly', result['answer'])
                self.assertIn('only if necessary', result['answer'])
                self.assertIn(owned.TABLE_QUALIFIER, result['answer'])
                contract = f.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['semantic_answer_bound'])

    def test_exhaustive_unspecified_and_missing_intent_stay_strict_despite_forged_reply_flags(self):
        for intent in ('exhaustive', 'unspecified', None):
            with self.subTest(intent=intent), ExitStack() as cleanup:
                owner, f = self.fixture(cleanup, intent=intent, forged=True)
                result = f.run()
                self.assertEqual(result['status'], 'no_sources')
                row = self.observation(owner)
                self.assertEqual(row['factual_coverage_intent'], intent if intent is not None else 'other')
                self.assertTrue(row['source_first_factual_review'])
                self.assertFalse(row['useful_factual_selection'])
                self.assertFalse(row['selected_records_only'])
                self.assertTrue(row['source_table_catalog_complete'])
                schema = f.posts[0]['text']['format']['schema']['properties']['table_record_coverage']
                self.assertEqual(schema['type'], 'object')
                self.assertEqual(set(schema['required']), {'T1', 'T2'})
                self.assertEqual(len(f.posts), 1)

    def test_missing_catalog_cannot_enable_selection_even_with_useful_intent(self):
        with ExitStack() as cleanup:
            owner, f = self.fixture(cleanup)
            with patch.object(tables, 'compile_catalog', return_value=tables.TableCatalog()):
                result = f.run()
            self.assertEqual(result['status'], 'no_sources')
            row = self.observation(owner)
            self.assertEqual(row['factual_coverage_intent'], 'useful')
            self.assertTrue(row['useful_factual_selection'])
            self.assertTrue(row['inventory_template_allowed'])
            self.assertFalse(row['source_table_catalog_nonempty'])
            self.assertFalse(row['source_table_catalog_complete'])
            self.assertFalse(row['selected_records_only'])
            self.assertEqual(len(f.posts), 1)

    def test_debug_observation_filters_unknown_enum_nonbool_and_source_text(self):
        with ExitStack() as cleanup:
            owner, _ = self.fixture(cleanup)
            private = 'UNTRUSTED_PRIVATE_SOURCE_CONTENT'
            completion.record_contract_observation('semantic_review', {'outcome': 'no_sources'}, diagnostics={
                'factual_coverage_intent': private, 'selected_records_only': 1,
                'source_table_catalog_complete': 'true', 'source_first_factual_review': True,
                'source_table_catalog_nonempty': False, 'answer': private, 'sources_block': private})
            row = self.observation(owner)
            self.assertEqual(row['factual_coverage_intent'], 'other')
            self.assertNotIn('selected_records_only', row)
            self.assertNotIn('source_table_catalog_complete', row)
            self.assertIs(row['source_first_factual_review'], True)
            self.assertIs(row['source_table_catalog_nonempty'], False)
            self.assertNotIn(private, repr(row))
            owner.payload.debug = False
            completion.record_contract_observation('semantic_review', {'outcome': 'no_sources'}, diagnostics={
                'factual_coverage_intent': 'useful', **{key: True for key in BOOL_FIELDS}})
            plain = owner.contract_checks[-1]
            self.assertNotIn('factual_coverage_intent', plain)
            self.assertFalse(set(BOOL_FIELDS) & set(plain))


if __name__ == '__main__':
    unittest.main(verbosity=2)
