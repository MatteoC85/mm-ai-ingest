"""Synthetic regression for an operation that stops before its documented closure."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as review, review_evidence
from machinemind.infrastructure.request_budget import _V13BudgetExceeded


class OperationBoundaryTests(unittest.TestCase):
    DRAFT = "1. Release the clamp and open the guide.\n2. Feed material to the carriage."
    CLOSURE = "Close the clamp and adjust the guide before feeding."
    SOURCE = DRAFT + "\nAfter reaching the carriage, " + CLOSURE

    def reply(self, *, complete_answer=True, resolution="restored"):
        answer = self.DRAFT + ("\n3. " + self.CLOSURE if complete_answer else "")
        return {"outcome": "rewrite", "answer": answer, "operation_boundary": {
            "complete": True, "checked_blocks": ["step_1", "step_2"], "changes": [{
                "block_id": "step_1", "action_quote": "Release the clamp and open the guide.",
                "closing_source_citation_id": "manual:one", "closing_source_quote": self.CLOSURE,
                "closing_answer_quote": self.CLOSURE, "resolution": resolution}]}}

    def verify(self, reply, sources=None):
        return review.verify_operation_boundary(reply, layout=review.block_layout(self.DRAFT),
            source_texts={"manual:one": self.SOURCE} if sources is None else sources)

    def test_physical_destination_without_restoration_cannot_pass(self):
        result = self.verify(self.reply(complete_answer=False))
        self.assertEqual(result["outcome"], "no_sources")
        self.assertFalse(result["operation_boundary_validation"]["complete"])

    def test_same_reviewer_can_repair_closure_without_extra_call(self):
        result = self.verify(self.reply())
        self.assertEqual(result["outcome"], "rewrite")
        self.assertTrue(result["operation_boundary_validation"]["complete"])
        self.assertEqual(result["operation_boundary_validation"]["answer_sha256"],
                         review.digest(result["answer"]))
        self.assertIn("manual:one", result["citation_ids"])

    def test_closure_before_opening_or_from_unknown_source_is_rejected(self):
        reversed_reply = self.reply()
        reversed_reply["answer"] = self.CLOSURE + "\n" + self.DRAFT
        for reply, sources in ((reversed_reply, None), (self.reply(), {})):
            with self.subTest(reply=reply, sources=sources):
                self.assertEqual(self.verify(reply, sources)["outcome"], "no_sources")

    def test_source_quote_cannot_change_verbatim_words_or_numbers(self):
        for source in ("Do not close the clamp.", "Restore pressure to 2.5 bar."):
            reply = self.reply()
            reply["operation_boundary"]["changes"][0]["closing_source_quote"] = (
                "Do close the clamp." if "not" in source else "Restore pressure to 25 bar.")
            self.assertEqual(self.verify(reply, {"manual:one": source})["outcome"], "no_sources")

    def test_whitespace_and_typographic_quotes_do_not_invalidate_exact_quote(self):
        reply = self.reply()
        phrase = 'Close the "guide" and secure the operator\'s clamp.'
        reply["answer"] = self.DRAFT + "\n3. " + phrase
        change = reply["operation_boundary"]["changes"][0]
        change["closing_source_quote"] = phrase
        change["closing_answer_quote"] = phrase
        source = "Close the \u201cguide\u201d\n and secure the operator\u2019s  clamp."
        self.assertTrue(self.verify(reply, {"manual:one": source})["operation_boundary_validation"]["complete"])

    def test_documented_safe_hold_is_valid_without_inventing_restart(self):
        reply = self.reply(resolution="documented_terminal_state")
        terminal = "Leave the unit locked out until the specialist completes the repair."
        reply["answer"] = self.DRAFT + "\n3. " + terminal
        change = reply["operation_boundary"]["changes"][0]
        change.update(closing_source_quote=terminal, closing_answer_quote=terminal)
        self.assertTrue(self.verify(reply, {"manual:one": terminal})["operation_boundary_validation"]["complete"])

    def test_standing_safety_prerequisites_do_not_require_reversal(self):
        answer = "1. Keep the supply isolated and wear PPE.\n2. Keep STOP pressed during the visual check."
        result = review.verify_operation_boundary({"outcome": "pass", "answer": answer,
            "operation_boundary": {"complete": True, "checked_blocks": ["step_1", "step_2"], "changes": []}},
            layout=review.block_layout(answer), source_texts={"manual:one": answer})
        self.assertTrue(result["operation_boundary_validation"]["complete"])
        self.assertEqual(result["answer"], answer)
        self.assertNotIn("restart", result["answer"])

    def test_documented_terminal_hold_can_be_the_same_instruction(self):
        instruction = "Keep the supply isolated throughout inspection."
        reply = self.reply(resolution="documented_terminal_state")
        reply["answer"] = instruction
        change = reply["operation_boundary"]["changes"][0]
        change.update(action_quote=instruction, closing_source_quote=instruction,
                      closing_answer_quote=instruction)
        self.assertTrue(self.verify(reply, {"manual:one": instruction})["operation_boundary_validation"]["complete"])
        change["resolution"] = "restored"
        self.assertFalse(self.verify(reply, {"manual:one": instruction})["operation_boundary_validation"]["complete"])

    def test_review_wire_contracts_remain_compatible(self):
        def without_descriptions(value):
            if isinstance(value, dict):
                return {key: without_descriptions(item) for key, item in value.items()
                        if key != 'description'}
            if isinstance(value, list):
                return [without_descriptions(item) for item in value]
            return value

        # Golden wire contracts from caf4b7a. Generic prose may change; the
        # source-ordered schema, including its descriptions, must not change.
        expected = {
            False: '3918a95c4864ce7cca5741c35493008cc24b7ed019934e4a0f0daa695c4eb001',
            True: '6d777bb37980b26510b5bcc802daf19f146d74d87e3d248267eb69d49f9d0dea',
        }
        for source_ordered, baseline_digest in expected.items():
            with self.subTest(source_ordered=source_ordered):
                schema = review.review_schema(
                    {'name': 'verifier', 'schema': {'properties': {}, 'required': []}},
                    review.block_layout(self.DRAFT), [], [], operation_boundary=True,
                    source_ordered=source_ordered, operation_source_ids=['manual:one'])
                contract = schema if source_ordered else without_descriptions(schema)
                encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), baseline_digest)

    def test_requested_restoration_does_not_need_an_invented_second_cycle(self):
        original='1. Replace the cover and secure its latch.\n2. Keep the supply isolated until authorized restart.'
        replacement='After cleaning:\n1. Replace the cover.\n2. Secure its latch.\nKeep the supply isolated until authorized restart.'
        result=review.verify_operation_boundary({'outcome':'rewrite','answer':replacement,
            'operation_boundary':{'complete':True,'checked_blocks':['step_1','step_2'],'changes':[]}},
            layout=review.block_layout(original),source_texts={'manual:one':original})
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertEqual(result['answer'],replacement)

    def test_terminal_restoration_exact_support_and_self_restored_negative(self):
        terminal='Secure the cover in its closed position.'
        reply=self.reply(resolution='documented_terminal_state')
        reply['answer']='1. '+terminal
        change=reply['operation_boundary']['changes'][0]
        change.update(action_quote=terminal,closing_source_quote=terminal,closing_answer_quote=terminal)
        result=self.verify(reply,{'manual:one':terminal})
        self.assertTrue(result['operation_boundary_validation']['complete'])
        for sources,resolution in (({},'documented_terminal_state'),
                                   ({'manual:one':'Do not secure the cover.'},'documented_terminal_state'),
                                   ({'manual:one':terminal},'restored')):
            candidate=deepcopy(reply);candidate['operation_boundary']['changes'][0]['resolution']=resolution
            self.assertEqual(self.verify(candidate,sources)['outcome'],'no_sources')

    def test_overlapping_self_closure_is_not_a_later_action(self):
        phrase='Adjust the guide and verify free movement.'
        reply=self.reply();reply['answer']='1. '+phrase
        change=reply['operation_boundary']['changes'][0]
        change.update(action_quote='Adjust the guide',closing_source_quote=phrase,closing_answer_quote=phrase)
        self.assertEqual(self.verify(reply,{'manual:one':phrase})['reason'],
                         'operation_boundary_quote_or_order_invalid')

    def test_unknown_closure_missing_audit_or_unchecked_block_cannot_pass(self):
        replies = []
        for mutate in (lambda r: r.pop("operation_boundary"),
                       lambda r: r["operation_boundary"].update(checked_blocks=["step_1"]),
                       lambda r: r["operation_boundary"].update(checked_blocks=["step_1", "step_1"]),
                       lambda r: r["operation_boundary"].update(complete=False),
                       lambda r: r["operation_boundary"]["changes"][0].update(resolution="unresolved")):
            reply = self.reply()
            mutate(reply)
            replies.append(reply)
        for reply in replies:
            self.assertEqual(self.verify(reply)["outcome"], "no_sources")

    def test_operation_schema_only_opted_into_manual_mixed_review(self):
        legacy = {"name": "verifier", "schema": {"properties": {}, "required": []}}
        layout = review.block_layout(self.DRAFT)
        normal = review.review_schema(legacy, layout, [], [])
        mixed = review.review_schema(legacy, layout, [], [], operation_boundary=True,
                                     operation_source_ids=["manual:one"])
        self.assertNotIn("operation_boundary", normal["schema"]["properties"])
        self.assertIn("operation_boundary", mixed["schema"]["required"])
        self.assertEqual(mixed["schema"]["properties"]["operation_boundary"]["properties"]
                         ["checked_blocks"]["items"]["enum"], ["step_1", "step_2"])
        self.assertEqual(mixed["schema"]["properties"]["operation_boundary"]["properties"]
                         ["changes"]["items"]["properties"]["closing_source_citation_id"]["enum"], ["manual:one"])

    def test_full_selected_packet_keeps_closing_tail_and_obeys_byte_limit(self):
        rows = [{"citation_id": f"manual:{i}", "bubble_document_id": "manual",
                 "text": f"Source operation {i}."} for i in range(12)]
        rows[-1]["text"] = self.CLOSURE
        original = deepcopy(rows)
        def render(items, *, max_context_chars):
            return "\n\n".join(f'[{r["citation_id"]}] {r["text"]}' for r in items)[:max_context_chars]
        count, chars = review_evidence.complete_limits(rows, render=render, max_records=8,
            max_context_chars=50, max_bytes=4096)
        packet = review_evidence.compile_packet(primary=rows, extension=[], render=render,
            max_records=count, max_context_chars=chars)
        self.assertEqual(len(packet.rows), 12)
        self.assertIn(self.CLOSURE, packet.sources)
        self.assertEqual(rows, original)
        with self.assertRaises(_V13BudgetExceeded):
            review_evidence.complete_limits(rows, render=render, max_records=8,
                max_context_chars=50, max_bytes=50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
