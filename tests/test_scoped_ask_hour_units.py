"""Hours in the actual protected ASK runtime; no external I/O or model calls."""
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_source_identifiers as fixture
from test_ask_table_units import TABLE


class ScopedAskHourUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AskSourceIdentifierTests.setUpClass()
        cls.m = fixture.AskSourceIdentifierTests.m
        cls.runtime = cls.m._assistant_core_scoped_ask_validation_runtime()

    @classmethod
    def tearDownClass(cls):
        fixture.AskSourceIdentifierTests.tearDownClass()

    def supported(self, answer, source):
        kept, removed = self.runtime._assistant_core_filter_unsupported_claim_sentences(answer, source)
        self.assertEqual((kept, removed), (answer, []))

    def rejected(self, answer, source):
        kept, removed = self.runtime._assistant_core_filter_unsupported_claim_sentences(answer, source)
        self.assertEqual(kept, '')
        self.assertTrue(removed)

    def test_actual_runtime_recognizes_same_value_hours_bidirectionally(self):
        self.assertEqual(self.runtime._assistant_core_claims.__module__, 'machinemind.ask.unit_grounding')
        for unit in ('ora', 'ore', 'hour', 'hours', 'hr', 'hrs', 'HOURS'):
            with self.subTest(unit=unit):
                self.supported('Interval: 600 h.', f'Intervallo: 600 {unit}.')
                self.supported(f'Interval: 600 {unit}.', 'Intervallo: 600 h.')
                self.supported('Interval: 2 h.', f'Intervallo: 2 {unit}.')

    def test_changed_values_untyped_numbers_and_other_dimensions_do_not_support_hours(self):
        for answer, source in (
            ('Interval: 601 h.', 'Intervallo: 600 ore.'),
            ('Interval: 600 ore.', 'Intervallo: 601 h.'),
            ('Interval: 600 h.', 'Value: 600.'),
            ('Interval: 2 hours.', 'Value: 2.'),
            ('Interval: 600 h.', 'Length: 600 mm.'),
            ('Length: 600 mm.', 'Intervallo: 600 ore.'),
            ('Interval: 600 ore.', 'Pressure: 600 bar.'),
        ):
            with self.subTest(answer=answer, source=source):
                self.rejected(answer, source)

    def test_hour_tokens_do_not_match_prose_prefixes_or_suffixes(self):
        for word in ('orecchie', 'orario', 'hourglass', 'hours_extra', 'hrsvalue', 'xhours'):
            with self.subTest(word=word):
                source = f'Entry: 600 {word}.'
                self.rejected('Interval: 600 h.', source)
                self.assertFalse(any(c.get('unit') == 'h'
                                     for c in self.runtime._assistant_core_claims(source)))

    def test_quoted_cells_range_and_labeled_value_preserve_hour_dimension(self):
        source = '- [PDF_TABLE_RECORD rows=2 columns=2] c1 r2="Frame" | c2 r2="Ogni 1000 ore"'
        self.supported('Interval: 1000 h.', source)
        self.supported('Intervallo: 1000 ore.', source)
        for left, right in (("\"", "\""), ("'", "'"), ('“', '”'), ('‘', '’')):
            with self.subTest(quote=right):
                self.supported('Interval: 600 h.', f'Interval: {left}600 hours{right}.')
                self.supported(f'Interval: {left}600 h{right}.', 'Intervallo: 600 ore.')
                self.rejected('Interval: 600 h.', f'Entry: {left}600 hourglass{right}.')
        self.supported('Interval: 200 h.', 'Range: "200 to 300 ore"')
        self.supported('Interval: 300 h.', 'Range: "200 to 300 ore"')
        self.supported('Interval: 600 h.', 'Record: "Value: 600 | Unit: hrs"')
        self.supported('Interval: 600 h.', 'Interval: 600 [ore]')
        self.supported('Interval: 600 h.', '[hours] 600')
        self.rejected('Interval: 600 h.', 'Interval: 600 [ore')

    def test_pressure_units_precision_and_dimensions_remain_strict(self):
        self.supported('Pressure: 20 MPa.', 'Pressure: 20 MPa.')
        self.supported('Pressure: 25 MPa.', 'Pressure: 25 MPa.')
        for answer, source in (
            ('Pressure: 25 MPa.', 'Pressure: 20 MPa.'),
            ('Pressure: 20 MPa.', 'Pressure: 25 MPa.'),
            ('Pressure: 20 bar.', 'Pressure: 20 MPa.'),
            ('Pressure: 20 MPa.', 'Interval: 20 hours.'),
            ('Length: 5000 mm.', 'Length: 50.00 mm.'),
        ):
            with self.subTest(answer=answer, source=source):
                self.rejected(answer, source)

    def test_table_column_dimension_support_is_preserved(self):
        self.supported('Temperature: 65 °C.', TABLE)
        self.supported('Interval: 25000 hours.', TABLE)
        self.rejected('Temperature: 66 °C.', TABLE)
        self.rejected('Interval: 65 hours.', TABLE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
