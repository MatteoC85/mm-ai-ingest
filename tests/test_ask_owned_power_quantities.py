"""Actual failed ASK table shape plus dimensional and review-binding negatives.

These are offline parser/contract tests, not live semantic quality evidence.
The real D33 page excerpt keeps its footnote verbatim. Separate examples use
different values so the implementation cannot depend on that document.
"""
import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution, final_contract, unit_grounding as u, validation
import test_ask_source_identifiers as identifier_fixture


def row(label, value, unit, *, index=2):
    cells = [f'c{i} r{index}=' + json.dumps(part, ensure_ascii=False)
             for i, part in enumerate((label, value, unit), 1)]
    return f'- [PDF_TABLE_RECORD rows={index} columns=3] ' + ' | '.join(cells)


D33_PAGE_EXCERPT = '\n\n'.join((
    '- [PDF_TABLE_RECORD rows=1 columns=3] c1-3 r1="Process Head - Optical Specification"',
    row('Power Handling MM (water cooled)', '12000', 'W', index=2),
    row('Power Handling SM (water cooled)', '3000', 'W', index=3),
    row('Collimating Lens', '100 / 120 / 140 / 160*', 'mm', index=7),
    '- [PDF_TABLE_RECORD rows=20 columns=3] c1-3 r20="* Maximum power should not exceed 2kW with multi-mode laser."',
))
D33_REVIEWED_ANSWER = (
    'The head power handling is 12 kW (12,000 W) for multi-mode operation.\n'
    'Collimating lens options include 100 mm, 120 mm, 140 mm and 160 mm.\n'
    'For the 160 mm option marked with an asterisk, maximum power should not '
    'exceed 2 kW with a multi-mode laser.')


class OwnedPowerQuantityTests(unittest.TestCase):
    def assert_supported(self, answer, source):
        self.assertEqual(u._assistant_core_filter_unsupported_claim_sentences(answer, source), (answer, []))

    def assert_rejected(self, answer, source):
        kept, removed = u._assistant_core_filter_unsupported_claim_sentences(answer, source)
        self.assertEqual(kept, '', (source, answer, kept))
        self.assertTrue(removed, (source, answer))

    def test_actual_failed_row_shape_preserves_reviewed_answer_and_footnote(self):
        before = D33_PAGE_EXCERPT.encode()
        self.assert_supported(D33_REVIEWED_ANSWER, D33_PAGE_EXCERPT)
        claims = u._assistant_core_claims(D33_PAGE_EXCERPT)
        self.assertIn(('12000', 'w'), {(c['raw'], c['unit']) for c in claims})
        self.assertIn(('160', 'mm'), {(c['raw'], c['unit']) for c in claims})
        self.assertEqual(D33_PAGE_EXCERPT.encode(), before)
        self.assertIn('* Maximum power should not exceed 2kW', D33_PAGE_EXCERPT)

    def test_general_row_values_ranges_and_numeric_footnote_alternatives(self):
        source = '\n'.join((row('Power rating', '7500', 'W'),
            row('Lens options', '85 / 105†', 'mm', index=3),
            row('Permitted mass', '15 to 19', 'kg', index=4),
            '- [PDF_TABLE_RECORD rows=5 columns=3] c1-3 r5="† Restricted to 1.5kW with the specified cooling."'))
        for answer in ('Power rating: 7.5 kW.', 'Power rating: 7,5 kW.',
                       'Lens: 85 mm or 105 mm.', 'Permitted mass: 15 to 19 kg.',
                       'Marked option: 105 mm, restricted to 1500 W.'):
            self.assert_supported(answer, source)
        for answer in ('Power rating: 75 kW.', 'Power rating: 7.5 W.',
                       'Lens: 7500 mm.', 'Permitted mass: 105 kg.',
                       'Lens: 106 mm.', 'Restricted power: 1.5 mm.'):
            self.assert_rejected(answer, source)

    def test_unit_ownership_requires_complete_exact_row_and_pure_unit(self):
        negatives = [
            '- [PDF_TABLE_RECORD rows=2 columns=2] c1 r2="7500" | c2 r2="W"',
            row('Power rating', '7500', 'Other cell: W'),
            row('Power rating', '7500', 'W').replace('c2 r2=', 'c2 r3='),
            row('Power rating', '7500', 'W').replace('c2 r2=', 'c2-3 r2='),
            row('Power rating', '7500', 'W').replace('c3 r2=', 'c3 r2-3=').replace('rows=2 ', 'rows=2-3 '),
            row('Power rating', '7500', 'W').replace('columns=3', 'columns=4'),
            row('Power rating', '7500', '')
                + '\nSource: another document\n' + row('Unit declaration', '', 'W'),
            row('Power rating', '7500', '') + '\n' + row('Other property', '', 'W', index=3),
        ]
        for source in negatives:
            with self.subTest(source=source):
                self.assert_rejected('Power rating: 7.5 kW.', source)

    def test_unit_conflicts_cannot_be_retyped_as_inline_units_or_bare_numbers(self):
        header = '- [PDF_TABLE_HEADER rows=1 columns=3] c1 r1="Property" | c2 r1="Temperature °C" | c3 r1="Unit"'
        for source in (header + '\n' + row('Power rating', '7500 W', 'W'),
                       row('Power rating [kg]', '7500 W', 'W'),
                       row('Temperature', '7500 W', '°C')):
            for answer in ('Power rating: 7.5 kW.', 'Power rating: 7500 W.', 'Temperature: 7500 °C.'):
                with self.subTest(source=source, answer=answer):
                    self.assert_rejected(answer, source)
        self.assert_rejected('Torque: 160 N m.', row('Optical wavelength', '160', 'nm'))

    def test_partial_options_arithmetic_unknown_suffixes_and_multiple_stars_fail_closed(self):
        for value in ('160/2', '160*2', '160**', '160foo', '1e3',
                      '100 / 160**', '100 / 160foo', '100 / 500WF',
                      '100 / / 160', '100 / 160 /', '100 / 160*2'):
            with self.subTest(value=value):
                self.assert_rejected('Lens: 160 mm.', row('Lens options', value, 'mm'))

    def test_power_conversion_is_scalar_exact_and_never_unit_only(self):
        for answer, source in (
                ('Power: 12 kW.', 'Power: 12000 W.'),
                ('Power: 12000 W.', 'Power: 12 kW.'),
                ('Power: 7.5 kW.', 'Power: 7500 W.'),
                ('Power: 7500 W.', 'Power: 7,5 kW.'),
                ('Power: 12.3456 kW.', 'Power: 12345.6 W.'),
                ('Power: 50 W.', 'Power: .05 kW.')):
            with self.subTest(answer=answer, source=source):
                self.assert_supported(answer, source)
        for answer, source in (
                ('Power: 12 kW.', 'Power: 12 W.'),
                ('Power: 12 W.', 'Power: 12 kW.'),
                ('Power: 7.5 kW.', 'Voltage: 7500 V.'),
                ('Power: 13 kW.', row('Power rating', '7500', 'W') + '\n' + row('Width', '13', 'mm', index=3)),
                ('Power: 12 kW.', 'Ambiguous power: 12,000 W.'),
                ('Power: 12 kW.', 'Ambiguous power: 12.000 W.'),
                ('Power: 5000 W.', 'Power: 50.00 W.'),
                ('Power: 1000 W.', 'Power: 1.0000000000000000000000000000000001 kW.')):
            with self.subTest(answer=answer, source=source):
                self.assert_rejected(answer, source)
        self.assertFalse(u._assistant_core_units_equivalent('W', 'kW'))
        self.assertEqual(u._exact_power_quantity('12,345.6', 'W'),
                         u._exact_power_quantity('12.3456', 'kW'))

    def test_unsupported_user_numeric_premise_is_not_made_source_evidence_by_negation(self):
        source = 'Maximum coil mass: 1000 kg. STOP does not isolate energy.'
        for answer in ('No, a 2500 kg coil is not allowed.',
                       'No: una matassa da 2.500 kg non è ammessa.',
                       'The source does not say the limit is 2500 kg.',
                       'The unsupported example of 2500 kg is illustrative.'):
            self.assert_rejected(answer, source)
        for answer in ('The proposed coil exceeds the documented maximum of 1000 kg.',
                       'La massa proposta supera il limite documentato di 1.000 kg.'):
            self.assert_supported(answer, source)

    def test_final_semantic_binding_does_not_allow_footnote_or_negation_changes(self):
        projected, removed = u._assistant_core_filter_unsupported_claim_sentences(D33_REVIEWED_ANSWER, D33_PAGE_EXCERPT)
        semantic = dict(outcome='rewrite', answer=D33_REVIEWED_ANSWER,
                        missing_facets=[], missing_answer_types=[], missing_list_items=[])
        unchanged = final_contract.final_contract(deterministic={'passed': True}, semantic=semantic,
            reviewed_answer=D33_REVIEWED_ANSWER, answer=projected,
            semantic_complete=True, semantic_partial=False)
        self.assertEqual(removed, [])
        self.assertTrue(unchanged['passed'], unchanged)
        for changed in (D33_REVIEWED_ANSWER.replace('not exceed 2 kW', 'not exceed 12 kW'),
                        D33_REVIEWED_ANSWER.split('\n')[:2],
                        D33_REVIEWED_ANSWER.replace('not exceed', 'exceed')):
            if isinstance(changed, list):
                changed = '\n'.join(changed)
            result = final_contract.final_contract(deterministic={'passed': True}, semantic=semantic,
                reviewed_answer=D33_REVIEWED_ANSWER, answer=changed,
                semantic_complete=True, semantic_partial=False)
            self.assertFalse(result['passed'], result)
            self.assertFalse(result['semantic_answer_bound'], result)
            self.assertEqual(result['reason'], 'semantic_answer_changed_after_review')


class ScopedPowerQuantityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identifier_fixture.AskSourceIdentifierTests.setUpClass()
        cls.m = identifier_fixture.AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        identifier_fixture.AskSourceIdentifierTests.tearDownClass()

    def test_only_ask_factory_binds_new_projection_and_conversion(self):
        answer = 'Head power: 12 kW.\nLens option: 160 mm.'
        runtime = self.m._assistant_core_scoped_ask_validation_runtime()
        self.assertEqual(runtime._assistant_core_filter_unsupported_claim_sentences(answer, D33_PAGE_EXCERPT),
                         (answer, []))
        kept, removed = self.m._assistant_core_filter_unsupported_claim_sentences(answer, D33_PAGE_EXCERPT)
        self.assertEqual(kept, '')
        self.assertIn('12', removed)
        self.assertIn('160', removed)

    def test_actual_failed_representation_survives_independent_review_and_final_gate(self):
        m = self.m
        row_data = {'citation_id': 'offline:p3:c0', 'source_type': 'document',
                    'chunk_full': D33_PAGE_EXCERPT, 'snippet': D33_PAGE_EXCERPT}
        request = m.AssistantCoreRequest(query='Explain the documented head power and collimating lens specifications.',
            requested_mode='ask', response_language='en', company_id='offline-company',
            machine_id='offline-machine', ai_scope='machine_all', top_k=6)
        decision = m.AssistantCoreDecision(request_kind='factual', effective_mode='ask',
            confidence=.9, requested_mode_fit=True, evidence_state='supported',
            evidence_policy='machine_sources_required', information_task='document_explanation',
            required_answer_types=('explanation',), required_facets=('power', 'lens'))
        for reviewed, expected in ((D33_REVIEWED_ANSWER, True),
                (D33_REVIEWED_ANSWER.replace('12 kW (12,000 W)', '13 kW'), False)):
            calls = []
            def reviewer(messages, **kwargs):
                calls.append(messages)
                return dict(outcome='rewrite', answer=reviewed, reason='offline fixture',
                    covered_facets=['power', 'lens'], missing_facets=[],
                    covered_answer_types=['explanation'], missing_answer_types=[],
                    expected_list_items=[], covered_list_items=[], missing_list_items=[],
                    citation_ids=['offline:p3:c0']), 'offline-reviewer'
            eruntime = replace(m._assistant_core_ask_execution_runtime(),
                evidence_admission=lambda *a, **kw: None, _v13_json_models=reviewer,
                _v13_current_budget=lambda: m._assistant_core_new_budget('ask'))
            runtime = replace(m._assistant_core_scoped_ask_validation_runtime(),
                execution_runtime=eruntime,
                _assistant_core_recover_citations=lambda response, **kw: (response['citations'], []),
                _assistant_core_verify_or_repair_answer=lambda **kw: execution.verify_or_repair_answer(**kw, runtime=eruntime),
                _sanitize_citations_for_response=lambda rows, **kw: list(rows),
                _build_rg_links=lambda *a, **kw: [], _v13_current_budget=lambda: None)
            generated = dict(answer=D33_REVIEWED_ANSWER, status='answered', citations=[row_data],
                rg_links=[], meta={}, _assistant_core_repair_attempted=True)
            retrieval = {'citations': [row_data], 'candidates': [row_data]}
            with patch.object(execution, '_admit_input', side_effect=lambda q, d, s, **kw: d), \
                 patch.object(validation, '_inputs', side_effect=lambda r, q, t, d, **kw: (r, t)), \
                 patch.object(validation, '_collection', side_effect=lambda q, items, d, **kw: items), \
                 patch.object(validation, '_output', side_effect=lambda r, *a, **kw: r):
                result = validation.validate_response(generated, request, retrieval, decision, runtime=runtime)
            contract = result['meta']['assistant_core_validation']['answer_contract']
            self.assertEqual(len(calls), 1)
            self.assertEqual(contract['passed'], expected, contract)
            self.assertEqual(contract['semantic_answer_bound'], expected, contract)
            if expected:
                self.assertEqual(result['answer'], D33_REVIEWED_ANSWER)
            else:
                self.assertEqual(contract['reason'], 'semantic_answer_changed_after_review')


if __name__ == '__main__':
    unittest.main(verbosity=2)
