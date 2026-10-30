"""Hour aliases preserve dimensions and the final reviewed-answer binding.

No provider or external I/O is allowed. The integration fixture drives the real
writer/render/review/final validator with explicitly simulated model replies.
"""
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_ask_source_identifiers import AskSourceIdentifierTests


class AskHourUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AskSourceIdentifierTests.setUpClass()
        cls.m = AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        AskSourceIdentifierTests.tearDownClass()

    def assert_supported(self, answer, source):
        text, removed = self.m._assistant_core_filter_unsupported_claim_sentences(answer, source)
        self.assertEqual(removed, [], (answer, source, removed))
        self.assertEqual(text, answer)

    def assert_rejected(self, answer, source):
        text, removed = self.m._assistant_core_filter_unsupported_claim_sentences(answer, source)
        self.assertEqual(text, "", (answer, source, text))
        self.assertTrue(removed, (answer, source))

    def test_hour_aliases_are_bidirectional_without_numeric_conversion(self):
        for unit in ("ora", "ore", "hour", "hours", "hr", "hrs", "HOURS"):
            with self.subTest(unit=unit):
                self.assert_supported("Intervallo: 300 h.", f"Intervallo: 300 {unit}.")
                self.assert_supported(f"Intervallo: 300 {unit}.", "Intervallo: 300 h.")

    def test_different_quantities_dimensions_and_untyped_numbers_stay_rejected(self):
        for answer, source in (
            ("Intervallo: 301 h.", "Intervallo: 300 ore."),
            ("Intervallo: 300 ore.", "Intervallo: 301 h."),
            ("Intervallo: 300 h.", "Misura: 300 mm."),
            ("Intervallo: 300 ore.", "Misura: 300 mm."),
            ("Misura: 300 mm.", "Intervallo: 300 ore."),
            ("Intervallo: 1000 h.", "Valore: 1000."),
            ("Intervallo: 1000 hours.", "Valore: 1000."),
        ):
            with self.subTest(answer=answer, source=source):
                self.assert_rejected(answer, source)

    def test_unit_words_require_whole_token_boundaries(self):
        for word in ("orecchie", "orario", "hourglass", "hours_extra", "hrsvalue", "xhours"):
            with self.subTest(word=word):
                source = f"Voce: 300 {word}."
                self.assert_rejected("Intervallo: 300 h.", source)
                claims = self.m._assistant_core_claims(source)
                self.assertFalse(any(c.get("unit") == "h" for c in claims), claims)

    def test_hour_units_work_in_existing_range_and_labeled_table_grammar(self):
        self.assert_supported("Intervallo: 300 h.", "Intervallo: da 200 a 300 ore.")
        self.assert_supported("Intervallo: 200 h.", "Intervallo: da 200 a 300 ore.")
        self.assert_supported("Intervallo: 300 h.", "[hours] 300")
        self.assert_supported("Intervallo: 300 h.", "Valore: 300 | Unità: ore")
        self.assert_supported("Value: 300 | Unit: hrs", "Intervallo: 300 ore.")

    def test_small_hour_values_keep_their_dimension(self):
        self.assert_supported("Interval: 2 h.", "Intervallo: 2 ore.")
        self.assert_rejected("Interval: 2 hours.", "Intervallo: 2 mm.")
        self.assert_rejected("Interval: 2 hours.", "Valore: 2.")

    def test_quoted_table_cells_and_output_keep_exact_hour_dimension(self):
        source = '- [PDF_TABLE_RECORD rows=2 columns=2] c1 r2="Component" | c2 r2="Ogni 1000 ore"'
        self.assert_supported("Intervallo: 1000 ore.", source)
        self.assert_supported("Interval: 1000 h.", source)
        for left, right in (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")):
            with self.subTest(quote=right):
                self.assert_supported(f"Interval: {left}1000 h{right}.", source)
                self.assert_supported("Interval: 1000 h.", f"Interval: {left}1000 hours{right}.")
                self.assert_rejected("Interval: 1000 h.", f"Record: {left}1000 hourglass{right}.")
                self.assert_rejected(f"Interval: {left}1000 h{right}.", 'Value: "1000 mm"')
        self.assert_supported("Interval: 200 h.", 'Range: "200 to 300 hours"')
        self.assert_supported("Interval: 300 h.", 'Record: "Value: 300 | Unit: hrs"')
        self.assert_rejected("Interval: 1000 h.", 'Record: "1000"')

    def test_reviewed_hour_abbreviation_survives_final_contract_without_projection(self):
        fixture = AskSourceIdentifierTests()
        result, calls, _ = fixture.run_pipeline(
            "The documented designation is QTR-7.\nThe documented interval is 600 h.", language="en")
        self.assertEqual([c[0] for c in calls], ["writer", "reviewer"])
        self.assertEqual(result["status"], "answered", result)
        meta = result["meta"]["assistant_core_validation"]
        self.assertEqual(meta.get("unsupported_numeric_or_code_claims_removed", []), [])
        self.assertTrue(meta["answer_contract"]["passed"])
        self.assertTrue(meta["answer_contract"]["semantic_answer_bound"])

    def test_reviewed_wrong_dimension_still_invalidates_final_contract(self):
        fixture = AskSourceIdentifierTests()
        result, _, _ = fixture.run_pipeline(
            "The documented designation is QTR-7.\nThe documented interval is 600 mm.", language="en")
        self.assertEqual(result["status"], "no_sources", result)
        contract = result["meta"]["assistant_core_validation"]["answer_contract"]
        self.assertFalse(contract["semantic_answer_bound"])
        self.assertEqual(contract["reason"], "semantic_answer_changed_after_review")


if __name__ == "__main__":
    unittest.main(verbosity=2, defaultTest="AskHourUnitTests")
