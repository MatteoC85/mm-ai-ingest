"""Root documentary review must bind one exact displayed span per source claim."""
import json
import unittest

from machinemind.retrieval import root_documentary as documentary
from machinemind.retrieval import root_documentary_diagnostics as diagnostics
from machinemind.retrieval import review_references


def packed_excerpts(excerpts):
    sources = [{"citation_id": cid, "text": text, "text_complete": True} for cid, text in excerpts.items()]
    return {"model_packet": {"sources": sources, "document_contexts": []}, "validator_records": sources}


class DocumentaryProofSlots(unittest.TestCase):
    def test_strict_schema_distinguishes_claims_using_the_same_source(self):
        claims = (("Independent document facts", ("a", "b")), ("A condition", ("a",)))
        slots = documentary.proof_slots(claims)
        self.assertEqual([(s["claim_index"], s["citation_id"]) for s in slots], [(0, "a"), (0, "b"), (1, "a")])
        evidence = documentary.review_evidence(packed_excerpts({"a": "Source A", "b": "Source B"}), claims)
        schema = documentary.review_schema(claims, evidence)["schema"]
        proof = schema["properties"]["proofs"]
        self.assertFalse(proof["additionalProperties"])
        self.assertEqual(proof["required"], [s["slot"] for s in slots])
        fields = list(proof["properties"].values())
        self.assertEqual(fields[0], fields[2])
        self.assertNotEqual(fields[0], fields[1])
        self.assertEqual([definition["enum"] for definition in schema["$defs"].values()], [["", "1"], ["", "2"]])

    def test_exact_binding_blocks_missing_extra_and_wrong_source_references(self):
        claims = (("Independent document facts", ("a", "b")), ("A condition", ("a",)))
        excerpts = {"a": "Source A excerpt, condition A", "b": "Source B excerpt"}
        evidence = documentary.review_evidence(packed_excerpts(excerpts), claims)
        first, second, third = (row["slot"] for row in documentary.proof_slots(claims))
        proofs = {first: "1", second: "2", third: "1"}
        def verdict(proofs):
            return documentary.review_outcome({"verdict": "approve", "reason": "supported", "proofs": proofs},
                                              claims, evidence)
        self.assertEqual(verdict(proofs), "verified")
        for invalid in ({first: proofs[first], second: proofs[second]}, {**proofs, "other": "1"}):
            self.assertEqual(verdict(invalid), "missing_or_excess_proof")
        for invalid in ({"1", "1-2", "Source B excerpt", " 2", "02", "2-2", "2-1", 2, None, ""}):
            with self.subTest(invalid=invalid):
                self.assertEqual(verdict({**proofs, second: invalid}), "proof_reference_mismatch")
        self.assertEqual(documentary.review_outcome(
            {"verdict": "reject", "reason": "wrong_source", "proofs": dict.fromkeys(proofs, "")}, claims, evidence
        ), "review_rejected")

    def test_parts_are_lossless_and_only_cited_excerpt_windows_are_selectable(self):
        originals = {"a": "A" * 149 + "\n" + "é\r\n  Ω " * 41, "b": "uncited text\r\n", "c": "C" * 6500}
        packed = packed_excerpts(originals)
        packed["model_packet"]["document_contexts"] = [{"id": "ctx", "fragments": [{"text": "context"}]}]
        claims = (("first", ("a",)), ("second", ("c",)))
        evidence = documentary.review_evidence(packed, claims)
        rendered = json.loads(evidence.model_json)
        self.assertEqual(rendered["document_contexts"], packed["model_packet"]["document_contexts"])
        last = 0
        for displayed, original in zip(rendered["sources"], packed["model_packet"]["sources"]):
            cid = displayed["citation_id"]
            if cid == "b":
                self.assertEqual(displayed, original)
                self.assertNotIn(cid, evidence.windows)
                continue
            parts = dict(displayed["proof_parts"])
            self.assertNotIn("text", displayed)
            self.assertEqual("".join(parts.values()), original["text"])
            self.assertEqual(list(parts), list(range(last + 1, last + 1 + len(parts))))
            last += len(parts)
            self.assertLessEqual(max(map(len, parts.values())), documentary.MAX_PROOF_PART_CHARS)
            for window in evidence.windows[cid]:
                start, _, end = window.partition("-")
                text = "".join(parts[i] for i in range(int(start), int(end or start) + 1))
                self.assertTrue(text.strip())
                self.assertLessEqual(len(text), documentary.MAX_QUOTE_CHARS)
        self.assertNotIn("1-3", evidence.windows["a"])
        self.assertIn("1-2", evidence.windows["a"])

    def test_exact_validator_binding_and_schema_size(self):
        excerpts = {str(i): ("λ" * 120 + "\n") * 17 for i in range(12)}
        claims = tuple(("claim", tuple(str(i) for i in range(start, start + 3))) for start in range(0, 12, 3))
        packed = packed_excerpts(excerpts)
        evidence = documentary.review_evidence(packed, claims)
        self.assertLessEqual(len(evidence.model_json), review_references.MAX_REFERENCE_PACKET_CHARS)
        schema = documentary.review_schema(claims, evidence)["schema"]
        self.assertLessEqual(sum(len(value["enum"]) for value in schema["$defs"].values()), 480)
        packed["validator_records"] = [{"citation_id": cid, "text": text + " "} for cid, text in excerpts.items()]
        self.assertIsNone(documentary.review_evidence(packed, claims))

    def test_shared_root_reference_slicing_retains_default_boundaries(self):
        body = "a" * 500 + "\n" + "b" * 700
        self.assertEqual(review_references.slices(body), [(0, 501), (501, 1141), (1141, 1201)])

    def test_rejected_focus_is_diagnostic_only_and_bound_to_one_exact_source(self):
        claims = (("First precise claim", ("a",)), ("Second precise claim", ("a", "b")))
        evidence = documentary.review_evidence(packed_excerpts({"a": "Allowed source A", "b": "Allowed source B"}), claims)
        expected = {"claim_1_citation_1": "1", "claim_2_citation_1": "", "claim_2_citation_2": "2"}
        raw = {"verdict": "reject", "reason": "other", "proofs": expected}
        ordinary = documentary.review_schema(claims, evidence)["schema"]["properties"]
        debug = documentary.review_schema(claims, evidence, rejection_focus=True)["schema"]["properties"]
        self.assertEqual({key: value for key, value in debug.items() if key != "rejection_focus"}, ordinary)
        located = documentary.rejected_focus(raw, claims, evidence, "claim_2_citation_1")
        self.assertEqual(located, {"focus": "claim_2_citation_1", "reason": "other", "claim_index": 1,
                                   "claim_text": "Second precise claim", "citation_id": "a",
                                   "proof_reference": "", "proof_parts": [[1, "Allowed source A"]]})
        foreign = {**raw, "proofs": {**expected, "claim_2_citation_1": "2"}}
        self.assertIsNone(documentary.rejected_focus(foreign, claims, evidence, "claim_2_citation_1"))
        self.assertIsNone(documentary.rejected_focus(raw, claims, evidence, "claim_9_citation_1"))
        self.assertEqual(documentary.rejected_focus(raw, claims, evidence, "coverage"),
                         {"focus": "coverage", "reason": "other"})
        self.assertEqual(documentary.review_outcome(raw, claims, evidence), "review_rejected")

    def test_private_rejection_rejects_a_different_scope_and_redacts_credentials(self):
        record = {"focus": "claim_1_citation_1", "reason": "other", "claim_index": 0,
                  "claim_origin": "generated", "citation_id": "private-citation", "proof_reference": "1",
                  "claim_text": "Exact rejected sentence. CHECK_TOKEN=synthetic-test-value",
                  "proof_parts": [[1, "Exact source unit. synthetic-test-value"]]}
        binding = {"commit_sha": "a" * 40, "request_id": "b" * 32, "query_sha256": "c" * 64,
                   "company_id": "private-company", "machine_id": "private-machine"}
        secret = "synthetic-secret-key-with-32-or-more-bytes"
        sealed = diagnostics.seal(record=record, binding=binding, secret=secret,
                                  environment={"CHECK_TOKEN": "synthetic-test-value"})
        self.assertNotIn("Exact rejected sentence", json.dumps(sealed))
        self.assertNotIn("private-company", json.dumps(sealed))
        with self.assertRaises(ValueError):
            diagnostics.open_envelope(sealed, secret=secret, expected_binding={**binding, "machine_id": "different"})
        opened = diagnostics.open_envelope(sealed, secret=secret, expected_binding=binding)
        self.assertTrue(opened["manifest"]["redaction_changed"])
        self.assertEqual(opened["record"]["claim_text"], "Exact rejected sentence. CHECK_TOKEN=[REDACTED]")
        self.assertEqual(opened["record"]["proof_parts"], [[1, "Exact source unit. [REDACTED]"]])


if __name__ == "__main__":
    unittest.main()
