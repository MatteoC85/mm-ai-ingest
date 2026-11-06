"""Source-quote delimiters do not weaken exact support or ordered safety checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review


class QuotedSourceBoundaryTests(unittest.TestCase):
    OPEN = "Release the guide."
    FEED = "Feed material to the carriage."
    CLOSE = "Secure the guide before movement."
    NOTE = "Keep the supply isolated."
    DRAFT = f"1. {OPEN}\n{NOTE}\n2. {FEED}\n3. {CLOSE}"
    SOURCE = f"{OPEN} {FEED} {CLOSE} {NOTE}"

    def reply(self, quote=None):
        return {"outcome": "rewrite", "answer": self.DRAFT, "citation_ids": ["manual:one"],
            "operation_boundary": {"complete": True,
                "checked_blocks": ["step_1", "step_2", "step_3"],
                "changes": [{"block_id": "step_1", "action_quote": self.OPEN,
                    "closing_source_citation_id": "manual:one",
                    "closing_source_quote": self.CLOSE if quote is None else quote,
                    "closing_answer_quote": self.CLOSE, "resolution": "restored"}],
                "precedence": [{"before_block_id": "step_1", "before_answer_quote": self.OPEN,
                    "after_block_id": "step_3", "after_answer_quote": self.CLOSE,
                    "source_citation_id": "manual:one", "source_quote": self.CLOSE if quote is None else quote}]}}

    def verify(self, value, *, sources=None, source_ordered=True, layout=None):
        original = deepcopy(value)
        result = review.verify_operation_boundary(value,
            layout=layout or review.block_layout(self.DRAFT),
            source_texts={"manual:one": self.SOURCE} if sources is None else sources,
            source_ordered=source_ordered)
        self.assertEqual(value, original)
        return result

    def test_exact_source_and_one_balanced_outer_pair_preserve_all_answer_bytes(self):
        for ordered in (False, True):
            for quote in (self.CLOSE, '"' + self.CLOSE + '"', '“' + self.CLOSE + '”'):
                with self.subTest(ordered=ordered, quote=quote):
                    result = self.verify(self.reply(quote), source_ordered=ordered)
                    self.assertEqual(result["outcome"], "rewrite")
                    self.assertTrue(result["operation_boundary_validation"]["complete"])
                    self.assertEqual(result["answer"], self.DRAFT)
                    self.assertEqual(result["citation_ids"], ["manual:one"])
                    self.assertEqual(result["operation_boundary_validation"]["answer_sha256"], review.digest(self.DRAFT))

    def test_whitespace_typography_inside_delimiter_keeps_literal_words(self):
        phrase = 'Secure the operator’s "guide" before movement.'
        value = self.reply('“Secure the operator\'s "guide"\n before movement.”')
        value["answer"] = self.DRAFT.replace(self.CLOSE, phrase)
        change = value["operation_boundary"]["changes"][0]
        change["closing_answer_quote"] = phrase
        value["operation_boundary"]["precedence"][0]["after_answer_quote"] = phrase
        result = self.verify(value, sources={"manual:one": phrase})
        self.assertTrue(result["operation_boundary_validation"]["complete"])

    def test_inner_quotation_marks_are_not_erased(self):
        for quote in ('"Secure the "guide" before movement."', 'Secure the "guide" before movement.'):
            with self.subTest(quote=quote):
                self.assertEqual(self.verify(self.reply(quote))["outcome"], "no_sources")
        self.assertFalse(review._source_quote_present('"Use 2 inches."', 'Use 2".'))

    def test_negation_case_quantity_and_unit_changes_still_reject(self):
        cases = [('"Close the guide."', "Do not close the guide."),
                 ('"Secure the Guide before movement."', self.SOURCE),
                 ('"Set 25 bar."', "Set 2.5 bar."),
                 ('"Set 2.5 psi."', "Set 2.5 bar.")]
        for quoted, source in cases:
            with self.subTest(quoted=quoted):
                self.assertEqual(self.verify(self.reply(quoted), sources={"manual:one": source})["outcome"], "no_sources")

    def test_wrong_or_missing_source_identity_cannot_borrow_another_source(self):
        for sources in ({}, {"manual:two": self.SOURCE},
                        {"manual:one": "Unrelated instruction.", "manual:two": self.SOURCE}):
            with self.subTest(sources=sources):
                self.assertEqual(self.verify(self.reply('"' + self.CLOSE + '"'), sources=sources)["outcome"], "no_sources")

    def test_closure_before_opening_and_missing_closure_still_reject(self):
        value = self.reply('"' + self.CLOSE + '"')
        for answer in (f"1. {self.CLOSE}\n2. {self.FEED}\n3. {self.OPEN}",
                       f"1. {self.OPEN}\n2. {self.FEED}\n3. Await instructions."):
            with self.subTest(answer=answer):
                candidate = deepcopy(value)
                candidate["answer"] = answer
                self.assertFalse(self.verify(candidate)["operation_boundary_validation"]["complete"])

    def test_unbalanced_empty_nested_and_single_delimiters_not_repaired(self):
        for quote in ('"' + self.CLOSE, self.CLOSE + '"', '""', '"  "',
                      '""' + self.CLOSE + '""', "'" + self.CLOSE + "'"):
            with self.subTest(quote=quote):
                self.assertEqual(self.verify(self.reply(quote))["outcome"], "no_sources")

    def test_source_ordered_wrong_block_and_ambiguous_span_still_reject(self):
        wrong = self.reply('"' + self.CLOSE + '"')
        wrong["operation_boundary"]["precedence"][0]["after_block_id"] = "step_2"
        self.assertEqual(self.verify(wrong)["outcome"], "no_sources")
        duplicated = self.reply('"' + self.CLOSE + '"')
        duplicated["answer"] += "\n" + self.CLOSE
        self.assertEqual(self.verify(duplicated)["outcome"], "no_sources")

    def test_missing_edited_dependency_and_unresolved_state_still_reject(self):
        value = self.reply('"' + self.CLOSE + '"')
        changed = deepcopy(value)
        changed["answer"] = changed["answer"].replace(self.FEED, "Feed material slowly to the carriage.")
        self.assertEqual(self.verify(changed)["outcome"], "no_sources")
        value["operation_boundary"]["changes"][0]["resolution"] = "unresolved"
        self.assertEqual(self.verify(value)["outcome"], "no_sources")

    def test_independent_reviewer_rejection_never_becomes_acceptance(self):
        value = self.reply('"' + self.CLOSE + '"')
        value.update(outcome="no_sources", reason="unsupported_operation")
        result = self.verify(value)
        self.assertEqual(result["outcome"], "no_sources")
        self.assertEqual(result["reason"], "unsupported_operation")

    def test_real_micro_edit_resolution_and_boundary_keep_notes_and_identity(self):
        layout = review.block_layout(self.DRAFT)
        aliases = review.review_wire_aliases(layout, ["manual:one"])
        value = {"outcome": "rewrite", "reply_mode": "edit", "answer": "", "citation_ids": ["s1"],
            "covered_facets": ["feed"], "missing_facets": [],
            "covered_answer_types": ["ordered_actions"], "missing_answer_types": [], "missing_list_items": [],
            "edits": [{"block_id": "b1", "old_text": self.OPEN, "new_text": "Release the marked guide."}],
            "operation_boundary": {"complete": True, "checked_blocks": ["b1", "b2", "b3"],
                "precedence": [{"from": "b1", "action": "Release the marked guide.", "to": "b3",
                    "next": self.CLOSE, "source": "s1", "proof": '"' + self.CLOSE + '"'}],
                "change_edges": [{"edge_index": 0, "resolution": "restored"}]}}
        decoded = review.decode_review_wire_aliases(value, aliases, layout=layout)
        resolved = review.resolve_reply(decoded, answer=self.DRAFT, layout=layout,
            required_facets=["feed"], required_types=["ordered_actions"], source_ordered=True,
            source_notes=({"citation_id": "step:1", "source_number": "1", "texts": (self.NOTE,)},))
        result = self.verify(resolved, layout=layout)
        self.assertEqual(result["outcome"], "rewrite")
        self.assertTrue(result["operation_boundary_validation"]["precedence_checked"])
        self.assertTrue(result["procedure_review"]["source_note_bindings_complete"])
        self.assertEqual(result["answer"], self.DRAFT.replace(self.OPEN, "Release the marked guide."))
        self.assertEqual(result["citation_ids"], ["manual:one"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
