"""Same-column dimensions; no provider/network/database calls.

The final gate fixture uses the real protected ASK runtime factory and real
claim/binding gates. Authority adapters and model replies are explicitly isolated.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution, unit_grounding as u, validation
import test_ask_source_identifiers as identifier_fixture


def record(row, values, *, role='RECORD', columns=2, labels=None):
    cells = []
    for index, value in enumerate(values, 1):
        label = ' ' + json.dumps(labels[index - 1]) if labels else ''
        cells.append(f'c{index}{label} r{row}=' + json.dumps(value, ensure_ascii=False))
    return f'- [PDF_TABLE_{role} rows={row} columns={columns}] ' + ' | '.join(cells)


TABLE = '\n'.join((record(1, ['Oil temperature °C', 'Lubrication interval (h) - Synthetic oil']),
                   record(2, ['Minore di 65', '25000']),
                   record(3, ['Da 68 a 80', '18000']),
                   record(4, ['Da 80 a 95', '12500'])))


class AskTableUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identifier_fixture.AskSourceIdentifierTests.setUpClass()
        cls.m = identifier_fixture.AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        identifier_fixture.AskSourceIdentifierTests.tearDownClass()

    def supported(self, text, source):
        kept, removed = u._assistant_core_filter_unsupported_claim_sentences(text, source)
        self.assertEqual(removed, [], (text, source, removed))
        self.assertEqual(kept, text)

    def rejected(self, text, source):
        kept, removed = u._assistant_core_filter_unsupported_claim_sentences(text, source)
        self.assertEqual(kept, '', (text, source, kept))
        self.assertTrue(removed)

    def test_explicit_header_units_bind_bare_range_endpoints_and_interval_column(self):
        for temperature in (65, 68, 80, 95):
            self.supported(f'Temperature: {temperature} °C.', TABLE)
        self.supported('Temperature: 68 to 80 °C.', TABLE)
        self.supported('Interval: 25000 h.', TABLE)
        self.supported('Interval: 18000 h.', TABLE)
        self.supported('Interval: 12500 h.', TABLE)
        for text in ('Temperature: 66 °C.', 'Pressure: 65 bar.', 'Interval: 65 h.',
                     'Current: 68 A.', 'Current: 80 A.',
                     'Temperature: 25000 °C.'):
            self.rejected(text, TABLE)

    def test_same_number_in_different_column_does_not_inherit_other_dimension(self):
        source = '\n'.join((record(1, ['Temperature °C', 'Interval (h)'], role='HEADER'),
                            record(2, ['41', '65'])))
        self.supported('Interval: 65 h.', source)
        self.supported('Temperature: 41 °C.', source)
        self.rejected('Temperature: 65 °C.', source)
        self.rejected('Interval: 41 h.', source)

    def test_table_scope_ends_at_prose_restart_column_change_gap_or_ambiguity(self):
        header = record(1, ['Temperature °C', 'Interval (h)'])
        for middle, tail in (
                ('SECTION: Another table', record(2, ['65', '25000'])),
                (record(1, ['Component', 'Reference']), record(2, ['65', '25000'])),
                ('', record(2, ['65', '25000', 'X'], columns=3)),
                ('', record(4, ['65', '25000'])),
                ('- [PDF_TABLE_UNASSIGNED relationship=ambiguous] "unassigned"', record(2, ['65', '25000'])),
                ('- [PDF_TABLE_RECORD malformed]', record(2, ['65', '25000'])),
                ('SOURCE: foreign document', record(2, ['65', '25000']))):
            with self.subTest(middle=middle, tail=tail):
                self.rejected('Temperature: 65 °C.', '\n'.join((header, middle, tail)))
        self.rejected('Temperature: 65 °C.', record(2, ['65', '25000']))
        self.rejected('Temperature: 65 °C.', header + '\nUnrelated value: 65.')

    def test_repeated_quoted_column_label_is_self_contained_with_rowspan(self):
        source = record('7-8', ['68 to 80', '18000'],
                        labels=['Oil "inlet" temperature [°C]', 'Interval (h)'])
        self.supported('Temperature: 68 to 80 °C.', source)
        self.supported('Interval: 18000 h.', source)
        self.rejected('Temperature: 18000 °C.', source)

    def test_explicit_cell_unit_must_agree_and_conflicts_cannot_be_retyped(self):
        header = record(1, ['Temperature °C', 'Interval (h)'])
        for cell in ('65 °C', '65°C', '[°C] 65'):
            self.supported('Temperature: 65 °C.', header + '\n' + record(2, [cell, '25000 h']))
        conflict = header + '\n' + record(2, ['65 bar', '25000 h'])
        self.rejected('Temperature: 65 °C.', conflict)
        self.rejected('Pressure: 65 bar.', conflict)
        self.rejected('Temperature: 65 °C.', header + '\n' + record(
            2, ['65', '25000'], labels=['Pressure (bar)', 'Interval (h)']))
        for unit in ('bar', 'MPa'):
            source = header + '\n' + record(2, ['65' + unit, '25000'])
            self.rejected('Temperature: 65 °C.', source)
            self.rejected(f'Pressure: 65 {unit}.', source)
            self.rejected(f'Pressure: 65{unit}.', source)
            self.supported(f'Pressure: 65{unit}.', source + f'\nIndependent pressure: 65{unit}.')
            self.rejected(f'Pressure: 65{unit}.', 'Unicode prefix: ㎜ café.\n' + source)
        self.supported('Pressure: 65MPa.',
            record(1, ['Pressure MPa', 'Interval (h)']) + '\n' + record(2, ['65MPa', '25000']))
        self.supported('Pressure: 65bar.',
            record(1, ['Temperature °C', 'Pressure bar']) + '\n' + record(2, ['65bar', '65bar']))
        # Conflicting labels do not erase unrelated nonnumeric identifiers.
        self.supported('Designation: QTR-7.', header + '\n' + record(
            2, ['QTR-7', '25000'], labels=['Pressure bar', 'Interval (h)']))

    def test_ambiguous_headings_or_layout_and_mixed_prose_do_not_inherit(self):
        for source in (
            record(1, ['Temperature °C / bar', 'Interval (h)']) + '\n' + record(2, ['65', '25000']),
            record(1, ['Temperature 41 °C', 'Interval (h)']) + '\n' + record(2, ['65', '25000']),
            '- [PDF_TABLE_HEADER rows=1 columns=2] c1-2 r1="Temperature °C"\n' + record(2, ['65', '25000']),
            '- [PDF_TABLE_HEADER rows=1 columns=2] c1 r1="Temperature °C" | c1 r1="Interval (h)"\n' + record(2, ['65', '25000']),
            record(1, ['Temperature °C', 'Interval (h)']) + '\n' + record(2, ['See drawing 65', '25000']),
            record(1, ['Temperature °C', 'Interval (h)']) + '\n' + record(2, ['65 hertz_unknown', '25000'])):
            with self.subTest(source=source):
                self.rejected('Temperature: 65 °C.', source)

    def test_protected_factory_uses_table_projection_and_root_parser_is_unchanged(self):
        runtime = self.m._assistant_core_scoped_ask_validation_runtime()
        text = 'Temperature: 65 °C.'
        self.assertEqual(runtime._assistant_core_filter_unsupported_claim_sentences(text, TABLE), (text, []))
        old_text, old_removed = self.m._assistant_core_filter_unsupported_claim_sentences(text, TABLE)
        self.assertEqual(old_text, '')
        self.assertEqual(old_removed, ['65'])

    def test_final_review_binding_keeps_supported_temperature_and_rejects_wrong_value(self):
        m = self.m
        row = {'citation_id': 'offline:p1:c0', 'source_type': 'document',
               'chunk_full': TABLE, 'snippet': TABLE}
        request = m.AssistantCoreRequest(query='What temperature is documented?', requested_mode='ask',
            response_language='en', company_id='offline-company', machine_id='offline-machine',
            ai_scope='machine_all', top_k=6)
        decision = m.AssistantCoreDecision(request_kind='factual', effective_mode='ask',
            confidence=.9, requested_mode_fit=True, evidence_state='supported',
            evidence_policy='machine_sources_required', information_task='document_explanation',
            required_answer_types=('explanation',), required_facets=('temperature',))
        for answer, expected in (('Temperature: 65 °C.', True), ('Temperature: 66 °C.', False)):
            answer = 'Documented operating conditions.\n' + answer
            calls = []
            def reviewer(messages, **kwargs):
                calls.append(messages)
                return {'outcome': 'rewrite', 'answer': answer, 'reason': 'fixture',
                    'covered_facets': ['temperature'], 'missing_facets': [],
                    'covered_answer_types': ['explanation'], 'missing_answer_types': [],
                    'expected_list_items': [], 'covered_list_items': [], 'missing_list_items': [],
                    'citation_ids': ['offline:p1:c0']}, 'offline-reviewer'
            eruntime = replace(m._assistant_core_ask_execution_runtime(),
                evidence_admission=lambda *a, **kw: None, _v13_json_models=reviewer,
                _v13_current_budget=lambda: m._assistant_core_new_budget('ask'))
            runtime = replace(m._assistant_core_scoped_ask_validation_runtime(),
                execution_runtime=eruntime,
                _assistant_core_recover_citations=lambda response, **kw: (response['citations'], []),
                _assistant_core_verify_or_repair_answer=lambda **kw: execution.verify_or_repair_answer(**kw, runtime=eruntime),
                _sanitize_citations_for_response=lambda rows, **kw: list(rows),
                _build_rg_links=lambda *a, **kw: [], _v13_current_budget=lambda: None)
            generated = {'answer': 'Documented operating conditions.\nTemperature: 65 °C.', 'status': 'answered',
                'citations': [row], 'rg_links': [], 'meta': {}, '_assistant_core_repair_attempted': True}
            retrieval = {'citations': [row], 'candidates': [row]}
            with patch.object(execution, '_admit_input', side_effect=lambda q, d, s, **kw: d), \
                 patch.object(validation, '_inputs', side_effect=lambda r, q, t, d, **kw: (r, t)), \
                 patch.object(validation, '_collection', side_effect=lambda q, items, d, **kw: items), \
                 patch.object(validation, '_output', side_effect=lambda r, *a, **kw: r):
                result = validation.validate_response(generated, request, retrieval, decision, runtime=runtime)
            contract = result['meta']['assistant_core_validation']['answer_contract']
            self.assertEqual(len(calls), 1)
            self.assertEqual(contract['passed'], expected, contract)
            self.assertEqual(contract['semantic_answer_bound'], expected, contract)
            if not expected:
                self.assertEqual(contract['reason'], 'semantic_answer_changed_after_review')


if __name__ == '__main__':
    unittest.main(verbosity=2)
