"""Root WHY selection mechanics; explicit offline verdicts, no semantic oracle."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import review_decisions, review_references as refs
from machinemind.retrieval import root_response_quality as quality
import test_root_response_quality as existing


WHY = ("Loss of pressure can explain clamp slipping. "
       "Low pressure was observed during this failure. "
       "The actual pressure remains unknown.")


def fixture():
    records = [{"citation_id": "manual-clamp", "text":
        "Model X: low clamp pressure can cause slipping. The existing pressure record can distinguish this condition.",
        "ownership_fragments": []}]
    packet = {"sources": [{"citation_id": records[0]["citation_id"],
                           "text": records[0]["text"], "context_ids": []}], "document_contexts": []}
    draft = [{"cause": "Possible loss of clamp pressure", "why": WHY,
              "checks": ["What pressure was recorded, if that record already exists?"]}]
    query = "The clamp slips. Its pressure has not been read."
    frozen = refs.prepare(packet=packet, proposal_manifest=review_decisions.manifest(
        draft, records, max_causes=3), records=records, original_query=query, observed_query=query)["frozen"]
    decision = {"proposal_index": 0, "verdict": "accept", "reason": "supported",
        "selected_check_indices": [0], "blocking_checks": [], "safe_check_subset": True,
        "selected_why_indices": [0, 2], "omitted_why_indices": [1], "safe_why_subset": True,
        "note": "Unreported pressure observation omitted; mechanism and uncertainty retained.",
        "proofs": [{"source_index": 0, "supports_cause": True,
            "observation_units": frozen["observed_ids"], "source_units": frozen["source_sets"][0],
            "target_units": frozen["target_sets"][0], "applicability": "same_target",
            "support_type": "bounded_inference", "check_indices": [0]}]}
    return frozen, records, query, {"decisions": [decision], "ranked_proposal_indices": [0]}


class RootWhyUnitsTests(unittest.TestCase):
    def test_missing_or_nontext_why_cannot_create_an_empty_unit(self):
        for value in (None, "", "  \t\n", [], 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(refs.ReferenceError, "invalid_why_text"):
                quality.why_units(value)

    def test_offsets_and_whitespace_roundtrip_without_normalization(self):
        text = "  Mechanism can explain slipping.\r\n\tObservation remains uncertain.  "
        units = quality.why_units(text)
        self.assertEqual(len(units), 2)
        self.assertEqual("".join(unit["text"] for unit in units), text)
        self.assertEqual([u["why_index"] for u in units], [0, 1])
        self.assertEqual(units[0]["start"], 0)
        self.assertEqual(units[-1]["end"], len(text))
        for unit in units:
            self.assertEqual(unit["text"], text[unit["start"]:unit["end"]])

    def test_decimals_codes_and_quoted_sentences_are_not_split_inside(self):
        for first in ("Pressure 0.75 bar can explain slipping.",
                      "Code X.21 indicates slipping.",
                      "The manual says «Slipping. Configuration uncertain» about slipping.",
                      "The manual says ‘Slipping. Configuration uncertain’ about slipping.",
                      "The manual says 'Slipping. Configuration uncertain' about slipping.",
                      'The manual says "Slipping. Configuration uncertain" about slipping.'):
            with self.subTest(first=first):
                text = first + " Configuration remains unknown."
                units = quality.why_units(text)
                self.assertEqual(len(units), 2)
                self.assertEqual(units[0]["text"], first + " ")

    def test_ambiguous_boundaries_or_incomplete_tail_remain_atomic(self):
        examples = ["See Fig. Configuration remains unknown.",
            "See approx. Configuration remains unknown.", "Use P.S. Configuration remains unknown.",
            "The state was OFF. Configuration remains unknown.",
            "Mechanism can explain slipping. but applicability remains unknown.",
            "Mechanism can explain slipping... Configuration remains unknown.",
            "Mechanism (can explain slipping. Configuration remains unknown.",
            "Mechanism can explain slipping. Configuration remains unknown",
            "The manual says ‘Mechanism can explain slipping. Configuration remains unknown.",
            "The manual says 'Mechanism can explain slipping. Configuration remains unknown.",
            'The manual says "Mechanism can explain slipping. Configuration remains unknown.']
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(quality.why_units(text),
                    [{"why_index": 0, "start": 0, "end": len(text), "text": text}])

    def test_apostrophes_inside_words_do_not_hide_sentence_boundaries(self):
        for text in ("The operator's record explains slipping. Configuration remains unknown.",
                     "L’osservazione può spiegare il fenomeno. La configurazione resta ignota."):
            with self.subTest(text=text):
                self.assertEqual(len(quality.why_units(text)), 2)
                self.assertEqual("".join(u["text"] for u in quality.why_units(text)), text)

    def test_unit_limit_never_truncates_or_creates_a_partial_sentence(self):
        sentence = "Configuration remains unknown. "
        self.assertEqual(len(quality.why_units(sentence * 8)), 8)
        too_many = sentence * 9
        self.assertEqual(len(quality.why_units(too_many)), 1)
        self.assertEqual(quality.why_units(too_many)[0]["text"], too_many)


class RootWhySelectionTests(unittest.TestCase):
    def setUp(self):
        self.frozen, self.records, self.query, self.parsed = fixture()

    def validate(self, parsed=None, frozen=None):
        return quality.validate_review(parsed=self.parsed if parsed is None else parsed,
            frozen=self.frozen if frozen is None else frozen, records=self.records, observed_query=self.query)

    def test_explicit_subset_preserves_exact_mechanism_caveat_checks_and_proofs(self):
        original, parsed_before = deepcopy(self.frozen), deepcopy(self.parsed)
        result = self.validate()
        units = quality.why_units(WHY)
        self.assertEqual(result["causes"][0]["why"], units[0]["text"] + units[2]["text"])
        self.assertIn("remains unknown", result["causes"][0]["why"])
        self.assertNotIn("was observed", result["causes"][0]["why"])
        self.assertEqual(result["causes"][0]["checks"], [self.frozen["proposals"][0]["checks"][0]["text"]])
        self.assertEqual(result["causes"][0]["cause"], self.frozen["proposals"][0]["cause"])
        self.assertEqual(self.frozen, original)
        self.assertEqual(self.parsed, parsed_before)
        summary = result["summary"]
        self.assertEqual(summary["support_proofs"][0]["decision"], self.parsed["decisions"][0]["proofs"][0])
        self.assertFalse(summary["semantic_truth_verified_by_code"])
        self.assertFalse(summary["immutable_proposal_text"])
        self.assertTrue(summary["immutable_retained_why_units"])
        self.assertTrue(summary["all_input_why_units_decided"])
        self.assertEqual(summary["why_selection"][0]["units"], [
            {key: unit[key] for key in ("why_index", "start", "end")}
            | {"text_sha256": sha256(unit["text"].encode("utf-8")).hexdigest()} for unit in units])
        self.assertNotIn("was observed during this failure", str(summary))
        self.assertEqual(summary["original_registry_fingerprint"], original["fingerprint"])
        self.assertNotEqual(summary["selected_registry_fingerprint"], original["fingerprint"])

    def test_retaining_all_units_is_identical_and_requires_no_omission_note(self):
        decision = self.parsed["decisions"][0]
        decision.update(selected_why_indices=[0, 1, 2], omitted_why_indices=[], note="")
        # This synthetic verdict checks the projection only, not semantic truth.
        result = self.validate()
        self.assertEqual(result["causes"][0]["why"], WHY)
        self.assertTrue(result["summary"]["immutable_proposal_text"])
        self.assertEqual(result["summary"]["selected_registry_fingerprint"], self.frozen["fingerprint"])

    def test_required_selection_fields_have_no_legacy_or_rejection_fallback(self):
        for verdict in ("accept", "reject"):
            for field in ("selected_why_indices", "omitted_why_indices", "safe_why_subset"):
                bad = deepcopy(self.parsed)
                bad["decisions"][0]["verdict"] = verdict
                bad["decisions"][0].pop(field)
                with self.subTest(verdict=verdict, field=field), self.assertRaisesRegex(
                        refs.ReferenceError, "missing_why_selection"):
                    self.validate(bad)

    def test_nonboolean_safety_and_nonarray_indices_fail_closed(self):
        for field, value, error in [
            ("safe_why_subset", 1, "missing_why_selection"),
            ("safe_why_subset", "true", "missing_why_selection"),
            ("selected_why_indices", (0, 2), "invalid_why_selection_indices"),
            ("selected_why_indices", [0, 2.0], "invalid_why_selection_indices"),
            ("omitted_why_indices", None, "invalid_why_selection_indices"),
            ("omitted_why_indices", [True], "invalid_why_selection_indices")]:
            bad = deepcopy(self.parsed)
            bad["decisions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(refs.ReferenceError, error):
                self.validate(bad)

    def test_partition_and_original_order_are_strict(self):
        for selected, omitted, error in [([0], [1], "incomplete_why_partition"),
            ([0, 2], [1, 2], "incomplete_why_partition"), ([2, 0], [1], "invalid_why_selection_indices"),
            ([0, 0, 2], [1], "invalid_why_selection_indices"), ([False, 2], [1], "invalid_why_selection_indices"),
            ([0, 3], [1, 2], "invalid_why_selection_indices"), ([0], [2, 1], "invalid_why_selection_indices")]:
            bad = deepcopy(self.parsed)
            bad["decisions"][0].update(selected_why_indices=selected, omitted_why_indices=omitted)
            with self.subTest(selected=selected, omitted=omitted), self.assertRaisesRegex(refs.ReferenceError, error):
                self.validate(bad)

    def test_empty_why_and_unsafe_caveat_dependency_verdict_fail_closed(self):
        for fields, error in [({"selected_why_indices": [], "omitted_why_indices": [0, 1, 2]}, "empty_selected_why"),
            ({"safe_why_subset": False}, "why_subset_not_safe"),
            ({"safe_check_subset": False}, "check_subset_not_safe")]:
            bad = deepcopy(self.parsed)
            bad["decisions"][0].update(fields)
            with self.subTest(fields=fields), self.assertRaisesRegex(refs.ReferenceError, error):
                self.validate(bad)

    def test_why_only_omission_requires_a_note_without_omitting_any_check(self):
        self.parsed["decisions"][0]["note"] = ""
        with self.assertRaisesRegex(refs.ReferenceError, "invalid_selection_note"):
            self.validate()

    def test_rejection_remains_rejection_with_no_reconstructed_acceptance(self):
        decision = self.parsed["decisions"][0]
        decision.update(verdict="reject", reason="unsupported", selected_why_indices=[],
            omitted_why_indices=[0, 1, 2], safe_why_subset=False, selected_check_indices=[],
            safe_check_subset=False, blocking_checks=[], proofs=[], note="Required caveat cannot be retained safely.")
        self.parsed["ranked_proposal_indices"] = []
        self.assertEqual(self.validate()["causes"], [])
        decision.update(selected_why_indices=[0], omitted_why_indices=[1, 2])
        with self.assertRaisesRegex(refs.ReferenceError, "rejected_proposal_keeps_why"):
            self.validate()

    def test_atomic_why_cannot_drop_an_unsupported_internal_clause(self):
        # A semicolon does not authorize clause-level removal, even when an
        # explicit offline reviewer rejects only the second claim's truth.
        atomic = "Pressure may explain slipping; low pressure was observed during this failure."
        frozen = deepcopy(self.frozen)
        frozen["proposals"][0]["why"] = atomic
        frozen["fingerprint"] = refs.digest({k: v for k, v in frozen.items() if k != "fingerprint"})
        bad = deepcopy(self.parsed)
        bad["decisions"][0].update(selected_why_indices=[0], omitted_why_indices=[1])
        with self.assertRaisesRegex(refs.ReferenceError, "invalid_why_selection_indices"):
            self.validate(bad, frozen)
        bad["decisions"][0].update(verdict="reject", reason="unsupported", selected_why_indices=[],
            omitted_why_indices=[0], safe_why_subset=False, selected_check_indices=[],
            safe_check_subset=False, blocking_checks=[], proofs=[],
            note="The unsupported observation is inside the indivisible mechanism sentence.")
        bad["ranked_proposal_indices"] = []
        self.assertEqual(self.validate(bad, frozen)["causes"], [])

    def test_explicit_reviewer_rejects_selection_that_discards_essential_context(self):
        for why in (
            "This mechanism applies only to the optional clamp configuration. "
            "That configuration could explain slipping. Installation remains unverified.",
            "Low pressure has not been established. It could explain slipping. "
            "The existing record does not confirm the condition.",
        ):
            frozen = deepcopy(self.frozen)
            frozen["proposals"][0]["why"] = why
            frozen["fingerprint"] = refs.digest({k: v for k, v in frozen.items() if k != "fingerprint"})
            bad = deepcopy(self.parsed)
            # This fixture is the explicit reviewer verdict, not a semantic
            # assertion by the validator: the antecedent/limitation is needed.
            bad["decisions"][0].update(selected_why_indices=[1], omitted_why_indices=[0, 2],
                safe_why_subset=False, note="The retained sentence needs the omitted condition and uncertainty.")
            with self.subTest(why=why), self.assertRaisesRegex(refs.ReferenceError, "why_subset_not_safe"):
                self.validate(bad, frozen)
            bad["decisions"][0].update(selected_why_indices=[0, 1, 2], omitted_why_indices=[],
                safe_why_subset=True, note="")
            self.assertEqual(self.validate(bad, frozen)["causes"][0]["why"], why)

    def test_selection_cannot_replace_causal_proofs_or_authorize_foreign_evidence(self):
        for mutate, error in [
            (lambda proof: proof.update(supports_cause=False, observation_units=[], support_type="documented_check"), "mechanism_not_supported"),
            (lambda proof: proof.update(source_units=["invented-source-unit"]), "reference_outside_authorized_set")]:
            bad = deepcopy(self.parsed)
            mutate(bad["decisions"][0]["proofs"][0])
            with self.subTest(error=error), self.assertRaisesRegex(refs.ReferenceError, error):
                self.validate(bad)

    def test_tampering_original_why_or_packet_manifest_is_rejected(self):
        changed = deepcopy(self.frozen)
        changed["proposals"][0]["why"] = "A replacement claim."
        with self.assertRaisesRegex(refs.ReferenceError, "manifest_altered"):
            self.validate(frozen=changed)
        with self.assertRaisesRegex(refs.ReferenceError, "manifest_altered"):
            quality.why_selection_packet(changed)

    def test_schema_packet_preserves_authority_and_exposes_all_original_units(self):
        original_schema = refs.schema(self.frozen)
        schema = quality.review_schema(self.frozen)
        branch = schema["schema"]["properties"]["decisions"]["items"]["anyOf"][0]
        for name in ("selected_why_indices", "omitted_why_indices"):
            self.assertIn(name, branch["required"])
            self.assertEqual(branch["properties"][name]["items"]["enum"], [0, 1, 2])
        self.assertEqual(schema["schema"]["$defs"], original_schema["schema"]["$defs"])
        self.assertEqual(refs.schema(self.frozen), original_schema)
        packet = quality.why_selection_packet(self.frozen)
        self.assertEqual("".join(u["text"] for u in packet[0]["why_units"]), WHY)
        self.assertIn("optional-model limitation", quality.review_instruction())
        self.assertIn("COMPLETE retained WHY", quality.review_instruction())
        self.assertIn("negation", quality.WHY_SELECTION_INSTRUCTION)
        self.assertIn("combined meaning", quality.WHY_SELECTION_INSTRUCTION)
        self.assertIn("Atomic units cannot be split", quality.WHY_SELECTION_INSTRUCTION)
        self.assertIn("selected explanation", quality.CHECK_SELECTION_INSTRUCTION)

    def test_actual_root_caller_uses_one_review_and_public_string_projection(self):
        case = existing.RootReviewMainIntegrationTests()
        case.setUp()
        case.draft["possible_causes"][0]["why"] = WHY
        decision = case.parsed["decisions"][0]
        decision.update(selected_why_indices=[0, 2], omitted_why_indices=[1],
            note="Unreported observation and unsafe check omitted; uncertainty retained.")
        out = case.run_response()
        self.assertTrue(out["ok"])
        self.assertEqual(len(case.calls), 1)
        self.assertIn(WHY, case.calls[0][0][1]["content"])
        self.assertIn("WHY_SELECTION", case.calls[0][0][1]["content"])
        self.assertNotIn("was observed", out["possible_causes"][0]["why"])
        self.assertIsInstance(out["possible_causes"][0]["why"], str)
        self.assertIn("remains unknown", out["possible_causes"][0]["why"])
        self.assertEqual(out["meta"]["accounting_marker"], "preserved")
        self.assertEqual(case.calls[0][1]["purpose"], "assistant_core_root_cause_adjudicator")

    def test_nondebug_caller_does_not_reflect_omitted_text_through_review_notes(self):
        case = existing.RootReviewMainIntegrationTests()
        case.setUp()
        case.draft["possible_causes"][0]["why"] = WHY
        omitted_text = quality.why_units(WHY)[1]["text"].strip()
        note = "Omitted unsupported assertion: " + omitted_text
        case.parsed["decisions"][0].update(selected_why_indices=[0, 2], omitted_why_indices=[1], note=note)
        out = case.run_response()
        self.assertTrue(out["ok"])
        self.assertNotIn(omitted_text, str(out))
        self.assertNotIn("root_review_capture", out["meta"])
        for key in ("root_review_decisions", "root_causal_applicability"):
            for section in ("check_selection", "why_selection"):
                row = out["meta"][key][section][0]
                self.assertNotIn("note", row)
                self.assertEqual(row["note_sha256"], sha256(note.encode("utf-8")).hexdigest())
        self.assertIn(WHY, case.calls[0][0][1]["content"])

    def test_nondebug_rejection_notes_cannot_reintroduce_rejected_claims(self):
        case = existing.RootReviewMainIntegrationTests()
        case.setUp()
        rejected_text = case.draft["possible_causes"][0]["why"]
        for decision in case.parsed["decisions"]:
            pi = decision["proposal_index"]
            decision.update(verdict="reject", reason="unsupported", selected_check_indices=[],
                safe_check_subset=False, blocking_checks=[], proofs=[], selected_why_indices=[],
                omitted_why_indices=[u["why_index"] for u in quality.why_units(
                    case.draft["possible_causes"][pi]["why"])], safe_why_subset=False,
                note="Unsupported claim: " + rejected_text)
        case.parsed["ranked_proposal_indices"] = []
        out = case.run_response()
        self.assertFalse(out["ok"])
        self.assertNotIn(rejected_text, str(out))
        self.assertEqual(len(case.calls), 1)
        for row in out["meta"]["root_review_decisions"]["verdicts"]:
            self.assertNotIn("note", row)
            self.assertIn("note_sha256", row)

    def test_debug_preserves_full_original_why_and_notes_for_private_review(self):
        case = existing.RootReviewMainIntegrationTests()
        case.setUp()
        case.request.debug = True
        case.draft["possible_causes"][0]["why"] = WHY
        note = "Omitted unsupported assertion: " + quality.why_units(WHY)[1]["text"].strip()
        case.parsed["decisions"][0].update(selected_why_indices=[0, 2], omitted_why_indices=[1], note=note)
        out = case.run_response()
        self.assertTrue(out["ok"])
        self.assertNotIn("was observed", out["possible_causes"][0]["why"])
        summary = out["meta"]["root_review_decisions"]
        self.assertEqual(summary["proposals"][0]["why"], WHY)
        self.assertEqual(summary["why_selection"][0]["note"], note)
        self.assertEqual(summary["check_selection"][0]["note"], note)
        self.assertIn("root_review_capture", out["meta"])

    def test_response_summary_projection_never_mutates_private_audit(self):
        summary = self.validate()["summary"]
        before = deepcopy(summary)
        public = quality.response_review_summary(summary, debug=False)
        self.assertEqual(summary, before)
        self.assertNotIn("note", public["why_selection"][0])
        self.assertEqual(quality.response_review_summary(summary, debug=True), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
