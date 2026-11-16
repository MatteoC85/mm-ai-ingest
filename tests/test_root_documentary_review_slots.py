"""Root documentary review must bind one literal proof to each source claim."""
import unittest

from machinemind.retrieval import root_documentary as documentary


class DocumentaryProofSlots(unittest.TestCase):
    def test_strict_schema_distinguishes_claims_using_the_same_source(self):
        claims = (("Independent document facts", ("a", "b")), ("A condition", ("a",)))
        slots = documentary.proof_slots(claims)
        self.assertEqual([(s["claim_index"], s["citation_id"]) for s in slots], [(0, "a"), (0, "b"), (1, "a")])
        proof = documentary.review_schema(claims)["schema"]["properties"]["proofs"]
        self.assertFalse(proof["additionalProperties"])
        self.assertEqual(proof["required"], [s["slot"] for s in slots])

    def test_exact_binding_blocks_missing_extra_and_wrong_source_quotes(self):
        claims = (("Independent document facts", ("a", "b")), ("A condition", ("a",)))
        excerpts = {"a": "Source A excerpt, condition A", "b": "Source B excerpt"}
        first, second, third = (row["slot"] for row in documentary.proof_slots(claims))
        proofs = {first: "Source A excerpt", second: "Source B excerpt", third: "condition A"}
        self.assertTrue(documentary.approved({"verdict": "approve", "reason": "supported", "proofs": proofs}, claims, excerpts))
        for invalid in ({first: proofs[first], second: proofs[second]}, {**proofs, "other": "Source A excerpt"},
                        {**proofs, second: "Source A excerpt"}, {**proofs, third: ""}):
            with self.subTest(invalid=invalid):
                self.assertFalse(documentary.approved({"verdict": "approve", "reason": "supported", "proofs": invalid}, claims, excerpts))
        self.assertFalse(documentary.approved({"verdict": "reject", "reason": "wrong_source", "proofs": proofs}, claims, excerpts))


if __name__ == "__main__":
    unittest.main()
