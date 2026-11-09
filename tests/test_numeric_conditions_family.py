"""Mixed numeric preconditions retain the real owned-family pipeline, offline.

Provider responses and SQL records are fixtures; this proves the source flow and
strict contracts, not live model quality. Public fixtures use an unrelated
cartridge operation, not a company/machine/document-specific product rule.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import request_scope
from assistant_core_v2 import normalize_decision
import test_ask_conditions_family_admission as owned


def raw_router():
    rows = [('maximum permitted cartridge mass and unit', 'numeric_value'),
            ('initial checks', 'checklist'), ('safe access conditions', 'safety_conditions')]
    return {'request_kind': 'procedure', 'effective_mode': 'ask', 'confidence': .98,
        'requested_mode_fit': True, 'evidence_state': 'partial', 'evidence_policy': 'machine_sources_required',
        'information_task': 'numeric_specification', 'procedure_scope': 'conditions_only',
        'required_answer_types': ['numeric_value', 'checklist', 'safety_conditions'],
        'required_facets': [name for name, _ in rows], 'preferred_source_types': ['procedure', 'step', 'document'],
        'source_type_policy': 'require', 'facet_queries': [dict(facet=name, answer_type=kind, must_cover=True,
            dense_queries=[name], lexical_queries=[name], exact_terms=[],
            preferred_source_types=['procedure', 'step', 'document']) for name, kind in rows]}


class NumericFamilyFixture(owned.OwnedFamilyFixture):
    def __init__(self, test, **kwargs):
        self.mass = kwargs.pop('mass', '40 kg')
        self.writer_points = kwargs.pop('writer_points', None)
        super().__init__(test, **kwargs)

    def provider(self, messages, **kwargs):
        if 'selected_procedure_citation_id' in kwargs['json_schema']['schema']['properties']:
            return super().provider(messages, **kwargs)
        self.calls.append(('writer', deepcopy(messages), deepcopy(kwargs)))
        block = messages[1]['content'].split('SOURCES:\n', 1)[1]
        self.writer_packets.append(block)
        for key, _, _, body in self.rows:
            if key == 'procedure:' + self.chosen or any(key == row[0] for row in self.children['procedure:' + self.chosen]):
                self.test.assertIn(body, block)
            elif key.startswith('step:'):
                self.test.assertNotIn(body, block)
        cids = [next(cid for cid in re.findall(r'\[([^\]\n]+)\]', block) if cid.startswith(key + ':'))
                for key, *_ in self.children['procedure:' + self.chosen]]
        points = self.writer_points or [
            (0, 'The maximum permitted cartridge mass is ' + self.mass + '.'),
            (0, 'Before insertion, verify the cycle is stopped and energy isolated. Use trained personnel.'),
            (1, 'Wear gloves and goggles and keep STOP pressed during insertion.'),
            (2, 'Verify straight unobstructed entry. Do not enter with the feeder enabled.'),
            (3, 'The low-speed test belongs after restoration, not to the initial checks.')]
        return self.reply({'answer_status': 'answered', 'grounded_points': [
            {'text': text, 'citation_ids': [cids[index]]} for index, text in points]},
            'offline-writer', messages, kwargs)


class NumericConditionsFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m = owned.AskConditionsFamilyAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        owned.AskConditionsFamilyAdmissionTests.tearDownClass()

    def request(self, mode='ask'):
        return self.m.AssistantCoreRequest(
            query='Before installing a cartridge, what is its maximum permitted mass and which checks and safety conditions must I verify? Keep the later low-speed test separate.',
            requested_mode=mode, response_language='en', company_id='offline-company',
            machine_id='offline-machine', ai_scope='machine_all', top_k=6)

    def test_observed_mixed_shape_is_canonicalized_without_losing_numeric_or_query_obligations(self):
        raw = raw_router()
        decision = normalize_decision(self.request(), raw)
        self.assertEqual(decision.information_task, 'procedure_segment')
        self.assertEqual(set(decision.required_answer_types), set(raw['required_answer_types']))
        self.assertTrue(request_scope.conditions_only_contract(procedure_scope=decision.procedure_scope,
            information_task=decision.information_task, required_answer_types=decision.required_answer_types,
            facet_queries=decision.facet_queries))
        self.assertEqual(len(decision.facet_queries), len(raw['facet_queries']))
        for before, after in zip(raw['facet_queries'], decision.facet_queries):
            self.assertEqual(tuple(before['dense_queries']), after.dense_queries)
            self.assertEqual(tuple(before['lexical_queries']), after.lexical_queries)
        numeric = decision.facet_queries[0]
        self.assertTrue(numeric.must_cover)
        self.assertEqual(numeric.facet, raw['facet_queries'][0]['facet'])
        self.assertEqual(decision.facet_queries[1].facet, request_scope.QUESTION_CONDITIONS_REQUIREMENT)
        self.assertTrue(decision.facet_queries[1].must_cover)
        self.assertEqual(request_scope.bind_conditions_requirements_to_question(decision), decision)
        self.assertEqual(raw['information_task'], 'numeric_specification')

    def test_pure_numeric_nonprocedure_other_phases_and_non_ask_cannot_acquire_this_path(self):
        variants = [dict(procedure_scope=value) for value in ('', 'execution_steps', 'bounded_phase')]
        variants += [dict(request_kind=value) for value in ('factual', 'general_technical', 'ambiguous', 'unsafe_request', 'out_of_scope')]
        variants += [dict(required_answer_types=['numeric_value'], facet_queries=[raw_router()['facet_queries'][0]]),
                     dict(required_answer_types=['numeric_value', 'checklist', 'ordered_actions']),
                     dict(information_task='comparison')]
        for updates in variants:
            with self.subTest(updates=updates):
                raw = {**raw_router(), **updates}
                decision = normalize_decision(self.request(), raw)
                expected = {'unsafe_request': 'other', 'out_of_scope': 'out_of_scope'}.get(
                    updates.get('request_kind'), raw['information_task'])
                self.assertEqual(decision.information_task, expected)
        for mode in ('root_cause', 'smart_diagnostic'):
            with self.subTest(mode=mode):
                decision = normalize_decision(self.request(mode), raw_router())
                self.assertEqual(decision.information_task, 'numeric_specification')
                self.assertEqual(decision.procedure_scope, '')

    def test_untrusted_request_metadata_does_not_supply_router_phase_or_numeric_contract(self):
        request = replace(self.request(), metadata={'procedure_scope': 'conditions_only',
            'information_task': 'procedure_segment', 'required_answer_types': ['numeric_value', 'checklist']})
        raw = {**raw_router(), 'procedure_scope': ''}
        decision = normalize_decision(request, raw)
        self.assertEqual(decision.information_task, 'numeric_specification')
        self.assertEqual(decision.procedure_scope, '')

    def test_no_numeric_or_no_check_mandatory_facet_does_not_canonicalize(self):
        for answer_type in ('numeric_value', 'checklist'):
            raw = raw_router()
            for facet in raw['facet_queries']:
                if (facet['answer_type'] == 'numeric_value') == (answer_type == 'numeric_value'):
                    facet['must_cover'] = False
            decision = normalize_decision(self.request(), raw)
            self.assertEqual(decision.information_task, 'numeric_specification')

    def test_owned_complete_family_writer_review_final_binding_and_phase_separation(self):
        decision = normalize_decision(self.request(), raw_router())
        rows = owned.catalog()
        rows[0]['steps'][0]['description'] += '\nLimite documentato di massa della cartuccia: 40 kg.'
        fixture = NumericFamilyFixture(self, decision=decision, kind='procedure', rows=rows,
            query=self.request().query)
        try:
            result = fixture.run()
            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual([row[0] for row in fixture.calls], ['selector', 'writer', 'reviewer'])
            self.assertEqual(len(fixture.generated['_assistant_core_validation_evidence']), 5)
            self.assertIn('40 kg', result['answer'])
            self.assertIn('STOP', result['answer'])
            self.assertIn('feeder enabled', result['answer'])
            self.assertIn('not to the initial checks', result['answer'])
            self.assertNotIn('Avviamento automatico', fixture.writer_packets[0])
            self.assertTrue(fixture.owner._response_flow.completed)
            numeric = next(f for f in fixture.decision.facet_queries if f.answer_type == 'numeric_value')
            self.assertTrue(numeric.must_cover)
            self.assertIn(numeric.facet, fixture.review_packets[0])
        finally:
            fixture.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
