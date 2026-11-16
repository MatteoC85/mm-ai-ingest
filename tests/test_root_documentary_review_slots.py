"""Root documentary review must bind one exact displayed span per source claim."""
import json
import unittest

from machinemind.retrieval import root_documentary as documentary
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


if __name__ == "__main__":
    unittest.main()
