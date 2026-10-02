"""Offline protocol regressions; no provider, source admission, or semantic proof."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review


class ProcedureReviewEditTests(unittest.TestCase):
    ANSWER = ("Before starting, isolate the supply.\n\n"
              "3. Release the lever at 2.5 bar. [source:3]\n\n"
              "8) Close the guide. [source:8]\n\n"
              "Caution: keep hands clear.\n")

    def verdict(self, text="Release the lever only after isolating the supply. [source:3]",
                block="step_1"):
        return {"outcome": "rewrite", "reply_mode": "edit", "answer": "",
                "edits": [{"block_id": block, "text": text}],
                "covered_facets": ["ordered_actions", "safety"], "missing_facets": [],
                "covered_answer_types": ["procedure"], "missing_answer_types": [],
                "missing_list_items": []}

    def resolve(self, parsed, *, answer=None, layout=None):
        answer = self.ANSWER if answer is None else answer
        return review.resolve_reply(parsed, answer=answer,
            layout=layout or review.block_layout(answer),
            required_facets=["ordered_actions", "safety"], required_types=["procedure"])

    def test_body_only_edit_keeps_original_number_and_separators(self):
        # This legitimate compact edit raised changes_step_sequence in v1:
        # its body has no number while the original block has [3].
        verdict = self.verdict()
        original = deepcopy(verdict)
        result = self.resolve(verdict)
        self.assertEqual(result["answer"], self.ANSWER.replace(
            "Release the lever at 2.5 bar. [source:3]", verdict["edits"][0]["text"]))
        self.assertEqual(verdict, original)
        self.assertEqual(result["procedure_review"]["edited_blocks"], 1)
        self.assertEqual(result["procedure_review"]["output_answer_sha256"],
                         review.digest(result["answer"]))

    def test_generic_manual_draft_preserves_all_actions_and_source_precautions(self):
        # A generic manual answer has no structured Step source records. This
        # fixture exercises only edit resolution, never claims authority or
        # replaces the independent source/safety review with a structural check.
        actions = ["Isolate the supply and verify zero energy.",
                   "Release the lever; keep hands clear.",
                   "Open the guide only after isolation.",
                   "Insert the material without forcing it.",
                   "Align both rollers at a maximum of 2.5 bar.",
                   "Close the guide before restoring power.",
                   "Lock the lever and verify the final position."]
        draft = "\n\n".join(f"{i}. {text} [manual:p{i}]" for i, text in enumerate(actions, 1))
        correction = "Align both rollers, never exceeding 2.5 bar. [manual:p5]"
        for text in (correction, "5. " + correction):
            with self.subTest(text=text):
                result = self.resolve(self.verdict(text, "step_5"), answer=draft)
                expected = draft.replace(actions[4], correction.removesuffix(" [manual:p5]"))
                self.assertEqual(result["answer"], expected)
                for action in actions[:4] + actions[5:]:
                    self.assertIn(action, result["answer"])
                self.assertEqual(len(review.block_layout(result["answer"])["blocks"]), 7)

    def test_same_heading_newline_or_markdown_is_only_formatting(self):
        body = "Release the lever at 2.5 bar after isolation. [source:3]"
        for prefix in ("3. ", "3) ", "3.\n", "3)\r\n", "**3.** ", "  __3)__ "):
            with self.subTest(prefix=prefix):
                result = self.resolve(self.verdict(prefix + body))
                self.assertEqual(result["answer"], self.ANSWER.replace(
                    "Release the lever at 2.5 bar. [source:3]", body))

    def test_block_ids_are_not_source_numbers(self):
        self.assertEqual(review.review_blocks(review.block_layout(self.ANSWER)), [
            {"block_id": "preamble", "step_number": None,
             "body_first_line": "Before starting, isolate the supply."},
            {"block_id": "step_1", "step_number": 3,
             "body_first_line": "Release the lever at 2.5 bar. [source:3]"},
            {"block_id": "step_2", "step_number": 8,
             "body_first_line": "Close the guide. [source:8]"}])

    def test_reordering_or_inserting_steps_is_rejected(self):
        for text in ("8. Close the guide.", "1. Release the lever.",
                     "3. Release the lever.\n4. Skip the interlock.",
                     "Release the lever.\n  4) Skip the interlock.",
                     "Release the lever.\n**8.** Close the guide.",
                     "Release the lever.\n8.\nClose the guide."):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "changes_step_sequence"):
                    self.resolve(self.verdict(text))

    def test_empty_body_and_number_only_edits_are_rejected(self):
        for text in ("", " \n", "3. ", "3.\n"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "changes_step_sequence"):
                    self.resolve(self.verdict(text))

    def test_preamble_cannot_add_step_but_can_keep_safety_warning(self):
        result = self.resolve(self.verdict("Isolate and lock the supply.\nKeep hands clear.", "preamble"))
        self.assertTrue(result["answer"].startswith(
            "Isolate and lock the supply.\nKeep hands clear.\n\n3. "))
        with self.assertRaisesRegex(ValueError, "changes_step_sequence"):
            self.resolve(self.verdict("1. Skip isolation.", "preamble"))

    def test_multiline_body_and_decimal_values_preserve_boundaries(self):
        body = "2.5 bar is the limit. [source:3]\n- Isolate first.\n- Keep hands clear."
        result = self.resolve(self.verdict(body + "\n\n\n"))
        self.assertIn("3. " + body + "\n\n8) Close", result["answer"])
        self.assertTrue(result["answer"].endswith("Caution: keep hands clear.\n"))

    def test_missing_safety_or_semantic_requirements_never_becomes_success(self):
        for mutation in ({"missing_facets": ["safety"]},
                         {"covered_facets": ["ordered_actions"]},
                         {"missing_answer_types": ["procedure"]},
                         {"covered_answer_types": []},
                         {"outcome": "partial"}):
            with self.subTest(mutation=mutation):
                verdict = self.verdict()
                verdict.update(mutation)
                with self.assertRaises(ValueError):
                    self.resolve(verdict)

    def test_unknown_duplicate_targets_and_changed_input_stay_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity_invalid"):
            self.resolve(self.verdict(block="step_3"))
        verdict = self.verdict()
        verdict["edits"] *= 2
        with self.assertRaisesRegex(ValueError, "identity_invalid"):
            self.resolve(verdict)
        with self.assertRaisesRegex(ValueError, "input_changed"):
            self.resolve(self.verdict(), answer=self.ANSWER + "changed",
                         layout=review.block_layout(self.ANSWER))

    def test_retain_is_byte_exact_and_still_requires_independent_pass(self):
        verdict = self.verdict()
        verdict.update(outcome="pass", reply_mode="retain", edits=[])
        self.assertEqual(self.resolve(verdict)["answer"], self.ANSWER)
        verdict["outcome"] = "rewrite"
        with self.assertRaisesRegex(ValueError, "retain_invalid"):
            self.resolve(verdict)

    def test_schema_and_prompt_specify_body_only_edits(self):
        legacy = {"name": "review", "schema": {"properties": {}, "required": []}}
        schema = review.review_schema(legacy, review.block_layout(self.ANSWER), [], [])
        text = schema["schema"]["properties"]["edits"]["items"]["properties"]["text"]
        self.assertIn("body", text["description"])
        self.assertIn("WITHOUT its numbered heading", review.PROTOCOL_INSTRUCTIONS)
        self.assertEqual(legacy["schema"]["properties"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
