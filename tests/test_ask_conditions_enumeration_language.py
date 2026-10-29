"""Checklist enumeration uses response-language labels without relaxing proofs.

The owned pipeline and final matcher are real; provider replies are explicit
fixtures. This verifies the supplied contract, not live model compliance.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution
from test_ask_conditions_family_admission import OwnedFamilyFixture, catalog
import test_ask_prepared_scope_admission as prepared_fixtures
import test_ask_conditions_pipeline as condition_fixtures


class LanguageLabelsFixture(OwnedFamilyFixture):
    def reply(self, reply, model, messages, kwargs):
        if kwargs['purpose'] == 'assistant_core_answer_contract_verifier':
            self.review_contract = (messages[0]['content'], messages[1]['content'])
            labels = (['gloves', 'goggles', 'keep STOP pressed'] if self.request.response_language == 'en'
                      else ['guanti', 'occhiali', 'mantenere STOP premuto'])
            reply = {**reply, 'expected_list_items': labels, 'covered_list_items': labels,
                     'missing_list_items': [], 'enumeration_requested': True}
        return super().reply(reply, model, messages, kwargs)


class AskConditionsEnumerationLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared_fixtures.AskPreparedScopeAdmissionTests.setUpClass()
        cls.m = prepared_fixtures.AskPreparedScopeAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        prepared_fixtures.AskPreparedScopeAdmissionTests.tearDownClass()

    def test_owned_review_contract_preserves_localized_items_and_complete_sources(self):
        for language in ('en', 'it'):
            with self.subTest(language=language):
                rows = deepcopy(catalog())
                rows[0]['short_description'] += '\n- Durata indicativa: 35 minuti'
                query = ('Which checks and safety controls must I verify before inserting a cartridge?'
                         if language == 'en' else 'Quali controlli di sicurezza verifico prima di inserire la cartuccia?')
                h = LanguageLabelsFixture(self, language=language, rows=rows, query=query)
                try:
                    result = h.run()
                    self.assertEqual([c[0] for c in h.calls], ['selector', 'writer', 'reviewer'])
                    self.assertEqual(result['status'], 'answered')
                    contract = result['meta']['assistant_core_validation']['answer_contract']
                    self.assertTrue(contract['passed'])
                    verifier = contract['semantic_verifier']
                    self.assertEqual(verifier['missing_list_items'], [])
                    self.assertEqual(verifier['covered_list_items'], verifier['expected_list_items'])
                    self.assertEqual(len(verifier['covered_list_items']), 3)
                    self.assertTrue(verifier['operation_boundary_validation']['complete'])
                    system, user = h.review_contract
                    self.assertIn('not a request to enumerate every Step', system)
                    self.assertIn('only supported checks and precautions applicable to the requested phase', system)
                    self.assertIn('covered_list_items and missing_list_items in RESPONSE_LANGUAGE', system)
                    self.assertIn('short exact phrase from the final resolved answer', system)
                    self.assertIn('Source quotes and operation-boundary evidence remain verbatim', system)
                    self.assertIn('ENUMERATION_REQUESTED: true', user)
                    hints = json.loads(user.split('DETERMINISTIC_LIST_CANDIDATES: ', 1)[1].splitlines()[0])
                    self.assertEqual(hints, [])
                    self.assertIn('Durata indicativa: 35 minuti', user)
                    self.assertIn('Verificare il consenso Ready dopo il ripristino.', user)
                finally:
                    h.close()

    def test_cross_language_or_invented_labels_are_not_accepted_by_the_lexical_gate(self):
        answer = 'Wear cut-resistant gloves and safety glasses; keep STOP pressed.'
        matcher = self.m._assistant_core_list_item_in_answer
        for label in ('cut-resistant gloves', 'safety glasses', 'keep STOP pressed'):
            self.assertTrue(matcher(label, answer))
        for label in ('guanti antitaglio', 'occhiali', 'pulsante di stop premuto', 'release STOP'):
            self.assertFalse(matcher(label, answer))

    def test_conditions_omit_metadata_hints_without_dropping_the_source(self):
        rows = deepcopy(catalog())
        rows[0]['short_description'] += '\n- Durata indicativa: 35 minuti'
        h = OwnedFamilyFixture(self, rows=rows,
            query='Which checks and safety controls must I verify before inserting a cartridge?')
        try:
            result = h.run()
            self.assertEqual(result['status'], 'answered')
            prompt = h.review_packets[0]
            self.assertIn('ENUMERATION_REQUESTED: true', prompt)
            self.assertIn('Durata indicativa: 35 minuti', prompt)
            hints = json.loads(prompt.split('DETERMINISTIC_LIST_CANDIDATES: ', 1)[1].splitlines()[0])
            self.assertEqual(hints, [])
        finally:
            h.close()

    def test_legacy_and_full_review_keep_enumeration_hints_and_existing_contract(self):
        base = condition_fixtures.AskConditionsPipelineTests(); base.m = self.m
        request, decision = base.request_decision()
        request = replace(request, query='Which control modes are available?')
        answer = 'The documented control modes are Fast and Slow.'
        rows = [{'citation_id': 'manual:modes', 'source_type': 'document',
                 'chunk_full': 'Control modes:\n- Fast\n- Slow'}]
        for changes in ({'procedure_scope': ''}, {'information_task': 'procedure_full'}):
            with self.subTest(changes=changes):
                seen = []
                d = replace(decision, **changes)
                def reviewer(messages, **kwargs):
                    seen.append(messages)
                    return {'outcome': 'pass', 'answer': answer, 'citation_ids': ['manual:modes'],
                        'covered_facets': list(d.required_facets), 'missing_facets': [],
                        'covered_answer_types': list(d.required_answer_types), 'missing_answer_types': [],
                        'expected_list_items': ['Fast', 'Slow'], 'covered_list_items': ['Fast', 'Slow'],
                        'missing_list_items': [], 'reason': 'offline fixture'}, 'offline-model'
                # Legacy review has no owned source adapter. The owned conditions
                # case above separately checks real admission and final handoff.
                runtime = replace(self.m._assistant_core_ask_execution_runtime(), evidence_admission=None,
                    _v13_json_models=reviewer, _v13_current_budget=lambda:self.m._assistant_core_new_budget('ask'))
                result = execution.verify_or_repair_answer(request=request, decision=d, answer=answer,
                    candidates=rows, runtime=runtime)
                self.assertEqual(result['outcome'], 'pass')
                self.assertEqual(result['missing_list_items'], [])
                self.assertEqual(len(seen), 1)
                self.assertNotIn('short exact phrase from the final resolved answer', seen[0][0]['content'])
                hints = json.loads(seen[0][1]['content'].split('DETERMINISTIC_LIST_CANDIDATES: ', 1)[1].splitlines()[0])
                self.assertIn('Fast', hints)
                self.assertIn('Slow', hints)


if __name__ == '__main__':
    unittest.main(verbosity=2)
