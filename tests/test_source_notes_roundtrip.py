"""Offline checks: literal source notes survive only the source-owned wrapper."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask.final_contract import preserve_source_notes, source_note_present


class SourceNotesRoundtripTests(unittest.TestCase):
    def notes(self, *texts):
        return tuple({"citation_id": f"step:s{i}:c0", "source_number": str(i),
                      "source_title": "Observed source", "texts": (text,)}
                     for i, text in enumerate(texts, 1))

    def test_multiline_note_roundtrip_is_complete_and_idempotent(self):
        notes = self.notes("Non entrare nella zona di lavoro\ncon l'aspo abilitato.",
                           "Keep STOP pressed.\nDo not exceed 5 mA.\nUse guard A3.")
        for language in ("it", "en"):
            for initial in (False, True):
                with self.subTest(language=language, initial=initial):
                    answer, proof = preserve_source_notes("1. Follow the source.", notes,
                        language=language, sentence_initial_equivalence=initial)
                    self.assertTrue(proof["all_present"])
                    self.assertEqual(proof["quoted_notes"], 2)
                    self.assertTrue(all(source_note_present(answer, t)
                        for note in notes for t in note["texts"]))
                    again, new_proof = preserve_source_notes(answer, notes,
                        language=language, sentence_initial_equivalence=initial)
                    self.assertEqual(again, answer)
                    self.assertEqual(new_proof["quoted_notes"], 0)
                    self.assertTrue(new_proof["all_present"])
                    self.assertEqual(proof["source_units"], new_proof["source_units"])

    def test_warning_numbers_case_and_identifiers_remain_exact(self):
        note = "Do not exceed 5 mA.\nKeep guard A3 closed."
        for mutation in ("Do exceed 5 mA.", "Do not exceed 6 mA.",
                         "Do not exceed 5 MA."):
            answer, _ = preserve_source_notes("1. Wait.", self.notes(note), language="en")
            answer = answer.replace("Do not exceed 5 mA.", mutation)
            self.assertFalse(source_note_present(answer, note))
        answer, _ = preserve_source_notes("1. Wait.", self.notes(note), language="en")
        self.assertFalse(source_note_present(answer.replace("A3", "A8"), note))

    def test_only_one_outer_quote_marker_is_removed(self):
        note = "Threshold > 5 mA.\n> Keep source symbol intact."
        answer, proof = preserve_source_notes("1. Wait.", self.notes(note), language="en")
        self.assertTrue(proof["all_present"])
        self.assertTrue(source_note_present(answer, note))
        altered = answer.replace("> > Keep source symbol", "> Keep source symbol")
        self.assertFalse(source_note_present(altered, note))

    def test_no_general_markdown_cleanup_or_cross_step_match(self):
        note = "Do not enter.\nKeep guard A3 closed."
        for answer in (
            "> Do not enter.\n> Keep guard A3 closed.",
            "Do **not** enter.\nKeep guard A3 closed.",
            "1. Do not enter.\n2. Keep guard A3 closed.",
            "Step notes and safety precautions — original wording\n\n"
            "Step 1\n> Do not enter.\n\nStep 2\n> Keep guard A3 closed.",
        ):
            self.assertFalse(source_note_present(answer, note))

    def test_malformed_supplement_is_not_unwrapped(self):
        note = "Do not enter.\nKeep guard A3 closed."
        answer, _ = preserve_source_notes("1. Wait.", self.notes(note), language="en")
        for suffix in ("\nExtra unquoted text", "\n\nStep 2", "\n\nBogus label\n> Text"):
            self.assertFalse(source_note_present(answer + suffix, note))
        for changed in (answer.replace("Step 1\n", "Unlabelled\n"),
                        answer.replace("original wording\n\n", "original wording\n")):
            self.assertFalse(source_note_present(changed, note))

    def test_crlf_blank_source_lines_and_sentence_initial_equivalence(self):
        note = "Keep guard A3 closed.\r\n\r\nDo not enter."
        answer, proof = preserve_source_notes("1. Wait.", self.notes(note), language="en")
        self.assertTrue(proof["all_present"])
        self.assertTrue(source_note_present(answer.replace("Keep guard", "keep guard"), note))
        self.assertFalse(source_note_present(answer.replace("Do not", "do not"), note))


if __name__ == "__main__":
    unittest.main()
