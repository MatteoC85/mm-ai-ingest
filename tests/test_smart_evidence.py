"""Offline representation/provenance tests; synthetic text only, no model or I/O."""
from copy import deepcopy
import ast
import hashlib
import hmac
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import review_references
from machinemind.retrieval import smart_evidence as evidence

SCOPE = {"company_id": "company-synthetic", "machine_id": "machine-synthetic", "ai_scope": "machine_all"}


def row(cid="ps:synthetic:p1-1:c1", text=None, **overrides):
    return {
        "citation_id": cid, "bubble_document_id": "ps:synthetic", "source_id": "synthetic",
        "source_type": "ps", "page_from": 1, "page_to": 1,
        "company_id": SCOPE["company_id"], "machine_id": SCOPE["machine_id"],
        "chunk_full": text or "Cause: a documented obstruction.\nCheck with the guard closed.\n",
        "snippet": "Display summary only", "snippet_clean": "Short UI title",
        "display_title": "A deliberately incomplete display title", **overrides,
    }


def ids(packet):
    return [source["citation_id"] for source in packet["sources"]]


class SmartEvidenceTests(unittest.TestCase):
    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaisesRegex(evidence.SmartEvidenceError, "^" + code + "$"):
            function(*args, **kwargs)

    def test_full_body_tail_and_whitespace_survive_all_turn_representations(self):
        text = " \n" + "Middle sentence.\n" * 100 + "SAFETY: keep isolation; restore the guide before use.\n "
        raw = row(text=text)
        before = deepcopy(raw)
        packet = evidence.build([raw], scope=SCOPE)
        loaded = json.loads(evidence.canonical(packet))
        verified = evidence.validate(loaded, scope=SCOPE, allowed_ids=ids(packet))
        for simulated_turn in ("start", "answer", "finalize"):
            block = evidence.evidence_block(verified)
            self.assertEqual(json.loads(block)["sources"][0]["text"], text, simulated_turn)
        self.assertEqual(evidence.review_packet(packet)["validator_records"][0]["text"], text)
        self.assertEqual(raw, before)
        self.assertNotIn("Short UI title", block)
        self.assertNotIn("deliberately incomplete display title", block)

    def test_source_text_fallback_is_exact(self):
        raw = row()
        raw.pop("chunk_full")
        raw["text"] = "\nNative source text, including final warning.\n"
        packet = evidence.build([raw], scope=SCOPE)
        self.assertEqual(evidence.review_packet(packet)["validator_records"][0]["text"], raw["text"])

    def test_display_snippet_only_requires_explicit_raw_producer_contract(self):
        raw = row()
        raw.pop("chunk_full")
        raw["snippet"] = "Full raw record " * 90 + "Final raw warning."
        self.assert_error("full_admitted_text_required", evidence.build, [raw], scope=SCOPE)
        packet = evidence.build([raw], scope=SCOPE, allow_raw_snippet=True)
        self.assertEqual(evidence.review_packet(packet)["validator_records"][0]["text"], raw["snippet"])

    def test_row_metadata_cannot_enable_raw_snippet_mode(self):
        raw = row(allow_raw_snippet=True)
        raw.pop("chunk_full")
        self.assert_error("full_admitted_text_required", evidence.build, [raw], scope=SCOPE)

    def test_ui_titles_are_never_a_body_even_in_raw_snippet_mode(self):
        raw = row()
        raw.pop("chunk_full")
        raw.pop("snippet")
        self.assert_error("full_admitted_text_required", evidence.build, [raw],
                          scope=SCOPE, allow_raw_snippet=True)

    def test_identical_aliases_retain_ids_with_one_lossless_body(self):
        raw = row()
        other = {**raw, "citation_id": "ps:synthetic:p1-1:v13page"}
        packet = evidence.build([raw, other], scope=SCOPE)
        self.assertEqual(len(packet["bodies"]), 1)
        self.assertEqual(ids(packet), [raw["citation_id"], other["citation_id"]])
        records = evidence.review_packet(packet)["validator_records"]
        self.assertEqual([r["text"] for r in records], [raw["chunk_full"]] * 2)

    def test_contiguous_alias_containment_retains_exact_original_offsets(self):
        small = row(text="middle warning")
        large = row("ps:synthetic:p1-1:v13page", "Heading\nmiddle warning\nclosing safety")
        packet = evidence.build([small, large], scope=SCOPE)
        self.assertEqual(len(packet["bodies"]), 1)
        self.assertGreater(packet["sources"][0]["start"], 0)
        self.assertEqual([r["text"] for r in evidence.review_packet(packet)["validator_records"]],
                         [small["chunk_full"], large["chunk_full"]])

    def test_same_text_different_owner_is_never_aliased(self):
        first = row()
        second = row("other:p1-1:c1", first["chunk_full"], bubble_document_id="other", source_id="other")
        packet = evidence.build([first, second], scope=SCOPE)
        self.assertEqual(len(packet["bodies"]), 2)

    def test_page_range_is_part_of_alias_provenance(self):
        first = row()
        second = row("ps:synthetic:p2-2:c1", first["chunk_full"], page_from=2, page_to=2)
        self.assertEqual(len(evidence.build([first, second], scope=SCOPE)["bodies"]), 2)

    def test_duplicate_citation_identity_is_rejected(self):
        self.assert_error("evidence_duplicate_citation_id", evidence.build, [row(), row()], scope=SCOPE)

    def test_cross_tenant_or_machine_source_rejected(self):
        self.assert_error("evidence_company_mismatch", evidence.build, [row(company_id="foreign")], scope=SCOPE)
        self.assert_error("evidence_machine_mismatch", evidence.build, [row(machine_id="foreign")], scope=SCOPE)

    def test_full_serialized_packet_cap_never_clips_text(self):
        raw = row(text="x" * evidence.MAX_CHARS + "WARNING_AT_END")
        original = deepcopy(raw)
        self.assert_error("evidence_packet_capacity_exceeded", evidence.build, [raw], scope=SCOPE)
        self.assertEqual(raw, original)

    def test_expanded_review_aliases_have_their_own_same_cap(self):
        rows = [row("ps:synthetic:p1-1:c" + str(i), "x" * 3000) for i in range(8)]
        self.assert_error("evidence_review_capacity_exceeded", evidence.build, rows, scope=SCOPE)

    def test_eight_full_sources_not_only_small_snippets_fit_with_safety_tails(self):
        rows = [row("doc" + str(i) + ":p1-1:c1",
                    ("Paragraph %d full sentence.\n" % i) * 45 + "LAST SAFETY NOTE %d" % i,
                    bubble_document_id="doc" + str(i), source_id="doc" + str(i)) for i in range(8)]
        packet = evidence.build(rows, scope=SCOPE)
        report = evidence.review_packet(packet)
        self.assertEqual(len(report["validator_records"]), 8)
        self.assertTrue(all(len(r["text"]) > 900 for r in report["validator_records"]))
        self.assertEqual([r["text"] for r in report["validator_records"]], [r["chunk_full"] for r in rows])
        self.assertLessEqual(report["summary"]["packet_chars"], 22000)
        self.assertLessEqual(report["summary"]["evidence_payload_chars"], 22000)

    def test_legacy_state_requires_restart_instead_of_rebuilding_ui_text(self):
        for legacy in (None, {}, {"evidence": [{"citation_id": "x", "snippet": "Display"}]}):
            self.assert_error("grounding_packet_restart_required", evidence.validate, legacy,
                              scope=SCOPE, allowed_ids=["x"])

    def test_wrong_scope_and_missing_allowed_id_fail_closed(self):
        packet = evidence.build([row()], scope=SCOPE)
        self.assert_error("evidence_scope_mismatch", evidence.validate, packet,
                          scope={**SCOPE, "company_id": "other"}, allowed_ids=ids(packet))
        self.assert_error("evidence_allowed_ids_mismatch", evidence.validate, packet,
                          scope=SCOPE, allowed_ids=["made-up-source"])

    def test_text_tamper_and_offset_tamper_rejected(self):
        packet = evidence.build([row()], scope=SCOPE)
        altered = deepcopy(packet)
        altered["bodies"][0]["text"] = altered["bodies"][0]["text"].replace("closed", "opened")
        self.assert_error("evidence_packet_altered", evidence.validate, altered, scope=SCOPE, allowed_ids=ids(packet))
        altered = deepcopy(packet)
        altered["sources"][0]["end"] += 1
        self.assert_error("evidence_offsets_invalid", evidence.validate, altered, scope=SCOPE, allowed_ids=ids(packet))

    def test_existing_state_hmac_covers_packet_even_if_attacker_rehashes_it(self):
        # Execute only the two pure production signing functions, not main imports.
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {"_sd_state_signature", "_sd_sign_state"}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        namespace = {"AI_INTERNAL_SECRET": "synthetic-unit-test-signing-key", "json": json,
                     "hmac": hmac, "hashlib": hashlib}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "signing_functions", "exec"), namespace)
        packet = evidence.build([row()], scope=SCOPE)
        signed = namespace["_sd_sign_state"]({**SCOPE, "grounding_packet": packet})
        altered = deepcopy(signed)
        altered["grounding_packet"] = evidence.build([row(text="Forged source body")], scope=SCOPE)
        # A locally valid fingerprint is not a server authorization witness.
        evidence.validate(altered["grounding_packet"], scope=SCOPE, allowed_ids=ids(packet))
        self.assertNotEqual(namespace["_sd_state_signature"](altered), signed["state_signature"])

    def test_validator_and_existing_reference_registry_share_literal_texts(self):
        packet = evidence.build([row(text="Machine A\n" + "Full source line.\n" * 70)], scope=SCOPE)
        review = evidence.review_packet(packet)
        proposals = {"sources": [{"citation_id": ids(packet)[0]}],
                     "proposals": [{"proposal_index": 0, "cause": "Obstruction",
                                    "why": "A qualified hypothesis", "checks": [
                                        {"check_index": 0, "text": "Check with guard closed"}]}]}
        prepared = review_references.prepare(
            packet=review["model_packet"], proposal_manifest=proposals,
            records=review["validator_records"], original_query="Symptom observed",
            observed_query="Symptom observed")
        units = prepared["frozen"]["units"]
        full = "".join(units[i]["text"] for i in prepared["frozen"]["source_sets"][0])
        self.assertEqual(full, review["validator_records"][0]["text"])
        self.assertTrue(prepared["summary"]["material_preserved"])

    def test_enrichment_appends_new_full_source_without_mutating_old(self):
        packet = evidence.build([row()], scope=SCOPE)
        original = deepcopy(packet)
        addition = row("new:p1-1:c1", "New observation source " * 60 + "Final safety.",
                       bubble_document_id="new", source_id="new")
        updated = evidence.update_enrichment(packet, [addition], scope=SCOPE,
                    allowed_ids=ids(packet) + [addition["citation_id"]])
        self.assertEqual(packet, original)
        before = evidence.review_packet(packet)["validator_records"][0]["text"]
        self.assertEqual(evidence.review_packet(updated)["validator_records"][0]["text"], before)
        self.assertNotEqual(updated["fingerprint"], packet["fingerprint"])

    def test_enrichment_cannot_replace_same_id_body_or_drop_old_identity(self):
        packet = evidence.build([row()], scope=SCOPE)
        self.assert_error("evidence_existing_source_changed", evidence.update_enrichment, packet,
                          [row(text="Rewritten lower-detail source")], scope=SCOPE)
        addition = row("new:p1-1:c1", bubble_document_id="new", source_id="new")
        self.assert_error("evidence_selection_lost_old_sources", evidence.update_enrichment, packet,
                          [addition], scope=SCOPE, allowed_ids=[addition["citation_id"]])

    def test_enrichment_max_three_new_sources_and_capacity_is_atomic(self):
        packet = evidence.build([row()], scope=SCOPE)
        original = deepcopy(packet)
        additions = [row("n%d:p1-1:c1" % i, bubble_document_id="n%d" % i, source_id="n%d" % i) for i in range(4)]
        self.assert_error("evidence_enrichment_source_limit", evidence.update_enrichment,
                          packet, additions, scope=SCOPE)
        huge = row("new:p1-1:c1", "x" * 22000, bubble_document_id="new", source_id="new")
        self.assert_error("evidence_packet_capacity_exceeded", evidence.update_enrichment,
                          packet, [huge], scope=SCOPE)
        self.assertEqual(packet, original)

    def test_enrichment_rejects_foreign_source(self):
        packet = evidence.build([row()], scope=SCOPE)
        self.assert_error("evidence_machine_mismatch", evidence.update_enrichment, packet,
                          [row("new", machine_id="other")], scope=SCOPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
