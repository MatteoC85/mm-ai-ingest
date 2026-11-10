"""Literal multiline Root fallback; no private fixtures or provider calls."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.presentation import citations
from machinemind.retrieval import root_fallback, root_time_budget


SOURCE = (
    "SOURCE_TYPE: problem_solution\nTITLE: Delivery pressure loss\nCATEGORY: hydraulic\n"
    "DESCRIPTION: REPORTED CONDITION\nPump runs but delivery stops.\n"
    "SECTION: OBSERVATIONS\nPressure: -6 bar; status E42.2 was recorded.\n"
    "SOLUTION: Qualified personnel must isolate the machine.\n"
    "Inspect the inlet only after stored energy is discharged.\n"
    "NOTES: Prior condition remains unverified.\nKeep the recorded evidence."
)


def unique(values, limit):
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))[:limit]


def load_main_functions(namespace):
    path = Path(__file__).resolve().parents[1] / "main.py"
    names = {"_v13_root_fallback_from_evidence", "_v13_generate_root_cause_response",
             "_assistant_core_synthesize_root_cause", "_root_cause_response_schema",
             "_ground_root_cause_result", "_extract_citation_ids_from_root_cause_json"}
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class LiteralPsFieldTests(unittest.TestCase):
    def test_complete_fields_keep_internal_newlines_and_colon_sections(self):
        fields = root_fallback.ps_fields(SOURCE)
        self.assertEqual(fields["description"],
            "REPORTED CONDITION\nPump runs but delivery stops.\n"
            "SECTION: OBSERVATIONS\nPressure: -6 bar; status E42.2 was recorded.")
        self.assertEqual(fields["solution"],
            "Qualified personnel must isolate the machine.\n"
            "Inspect the inlet only after stored energy is discharged.")
        self.assertNotIn("NOTES:", fields["solution"])
        for value in fields.values():
            self.assertIn(value, SOURCE)

    def test_crlf_and_cr_are_preserved_inside_literal_values(self):
        for newline in ("\r\n", "\r"):
            source = SOURCE.replace("\n", newline)
            fields = root_fallback.ps_fields(source)
            self.assertIn("machine." + newline + "Inspect", fields["solution"])
            self.assertIn(fields["description"], source)

    def test_empty_known_field_closes_preceding_field_without_bleeding(self):
        fields = root_fallback.ps_fields(
            "DESCRIPTION: Reported symptom\nContinuation.\nSOLUTION:\nNOTES: Unknown.")
        self.assertEqual(fields, {"description": "Reported symptom\nContinuation.",
                                  "solution": "", "notes": "Unknown."})

    def test_duplicate_canonical_fields_fail_closed_even_with_same_value(self):
        for duplicate in ("DESCRIPTION: REPORTED CONDITION", "description:",
                          "SOURCE_TYPE: problem_solution"):
            with self.subTest(duplicate=duplicate):
                self.assertEqual(root_fallback.ps_fields(SOURCE + "\n" + duplicate), {})

    def test_unknown_colon_and_inline_known_labels_do_not_create_fields(self):
        source = "DESCRIPTION: Recorded event\nVoltage: 24 V\nCompare SOLUTION: in the log."
        self.assertEqual(root_fallback.ps_fields(source),
                         {"description": source[len("DESCRIPTION: "):]})
        self.assertEqual(root_fallback.ps_fields(None), {})
        self.assertEqual(root_fallback.ps_fields("ordinary unstructured text"), {})

    def test_shared_presentation_parser_keeps_legacy_behavior(self):
        fields = citations.parse_structured_source_fields(SOURCE,
            clean_text_fn=citations.clean_display_text)
        self.assertEqual(fields["description"], "REPORTED CONDITION")
        self.assertEqual(fields["solution"], "Qualified personnel must isolate the machine")
        self.assertEqual(fields["section"], "OBSERVATIONS")


class RootFallbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.reviews = []
        self.source = {"citation_id": "ps:offline:p1:c1", "source_type": "ps",
                       "chunk_full": SOURCE, "assistant_core_root_viable": True}

        def model(messages, **kwargs):
            self.calls.append((deepcopy(messages), deepcopy(kwargs)))
            if kwargs["json_schema"]["name"] == "root_cause_producer_applicability_v1":
                # The owned Root writer now requires a private typed proposal.
                # Empty output still exercises legacy/default fallback below.
                fields = root_fallback.ps_fields(SOURCE)
                return {"problem_summary": "Unconfirmed documentary case.",
                    "possible_causes": [{"rank": 1, "cause": fields["title"],
                        "why": fields["description"], "checks": [],
                        "citations": [self.source["citation_id"]],
                        "applicability": {"status": "source_scoped_unverified",
                            "statement": "This explanation concerns the documented case; its applicability to current equipment is unverified."}}],
                    "recommended_next_checks": []}, "offline-model"
            return {}, "offline-model"

        def review(**kwargs):
            self.reviews.append(deepcopy(kwargs))
            return {"status": "error", "possible_causes": [], "review_required": True}

        namespace = {
            "AssistantCoreRequest": SimpleNamespace, "AssistantCoreDecision": SimpleNamespace,
            "_source_type_from_document_id": lambda doc: "ps",
            "_assistant_core_ps_is_substantive": lambda row: True,
            "_v13_candidate_text": lambda row: row["chunk_full"],
            "_parse_structured_source_fields": lambda text: citations.parse_structured_source_fields(
                text, clean_text_fn=citations.clean_display_text),
            "_clean_display_text": citations.clean_display_text,
            "_unique_non_empty_strings": unique, "_dedup_text_values": unique,
            "_assistant_core_retrieval_query": lambda request: request.query,
            "_v13_rescore_root_candidates": lambda query, rows: deepcopy(rows),
            "_v13_evidence_metrics": lambda rows: {}, "_v13_current_budget": lambda: None,
            "_assistant_core_adjudicate_root_cause": review,
            "_retrieval_diagnostic_query": SimpleNamespace(POLICY_VERSION="offline",
                reasoning_packet=lambda profile: {}),
            "_assistant_core_diagnostic_query_profile": lambda request: None,
            "_v13_root_cause_model": lambda query, retrieval: ("offline-model", "low", ""),
            "_retrieval_source_management": SimpleNamespace(
                V13SourcesBlockRuntime=lambda **kwargs: None,
                v13_root_sources_packet=lambda rows, **kwargs: {
                    "sources_block": "\n".join(row["chunk_full"] for row in rows), "summary": {}}),
            "_v13_assurance_prompt_block": lambda retrieval: "",
            "_v13_json_models": model,
            "_compact_root_cause_result_citations_by_family": lambda **kwargs: (kwargs["result"], kwargs["citations"]),
            "_lock_root_cause_result": lambda result, rows, **kwargs: (result, rows),
            "_sanitize_citations_for_response": lambda rows, **kwargs: deepcopy(rows),
            "_build_rg_links": lambda *args: [],
            "_V13BudgetExceeded": type("BudgetExceeded", (Exception,), {}),
            "MODE_ROOT_CAUSE": "root_cause", "V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE": 7,
            "V13_FAST_MODEL": "offline-model", "V13_HEAVY_MODEL": "offline-heavy",
            "V13_FAST_CONTEXT_CHARS": 18000, "V13_HEAVY_CONTEXT_CHARS": 18000,
            "V13_FAST_TIMEOUT_SECONDS": 26, "V13_HEAVY_TIMEOUT_SECONDS": 26,
            "V13_FAST_MAX_OUTPUT_TOKENS": 3400, "V13_HEAVY_MAX_OUTPUT_TOKENS": 3400,
            "json": json,
        }
        self.ns = load_main_functions(namespace)

    def fallback(self, source=None, **options):
        return self.ns["_v13_root_fallback_from_evidence"](
            q="Pump delivery stopped.", citations=[self.source if source is None else source],
            max_causes=3, response_language="en", **options)

    def test_flagged_fallback_preserves_multiline_draft_and_original_citation(self):
        before = deepcopy(self.source)
        result, selected = self.fallback(root_literal_fallback_fields=True)
        cause = result["possible_causes"][0]
        self.assertIn("Pump runs but delivery stops", cause["why"])
        self.assertIn("Pressure: -6 bar; status E42.2", cause["why"])
        self.assertIn("only after stored energy is discharged", cause["checks"][0])
        self.assertEqual(cause["citations"], [self.source["citation_id"]])
        self.assertEqual(selected, [before])
        self.assertEqual(self.source, before)
        self.assertEqual(self.calls, [])

    def test_default_and_explicit_false_keep_exact_legacy_fallback(self):
        default = self.fallback()
        self.assertEqual(default, self.fallback(root_literal_fallback_fields=False))
        cause = default[0]["possible_causes"][0]
        self.assertEqual(cause["why"], "REPORTED CONDITION")
        self.assertEqual(cause["checks"], ["Qualified personnel must isolate the machine",
            "Verify that the observed condition matches the cited problem/solution record."])

    def test_overlong_solution_is_omitted_instead_of_clipped_into_a_repair_check(self):
        solution = "Inspect the assembly. " * 20 + "Only after isolating all stored energy."
        source = {**self.source, "chunk_full": "SOURCE_TYPE: problem_solution\nTITLE: Recorded fault\n"
                  "DESCRIPTION: Delivery stopped.\nSOLUTION: " + solution}
        result, selected = self.fallback(source, root_literal_fallback_fields=True)
        self.assertEqual(result["possible_causes"][0]["checks"],
            ["Verify that the observed condition matches the cited problem/solution record."])
        self.assertIn(solution, selected[0]["chunk_full"])
        legacy, _ = self.fallback(source)
        self.assertTrue(legacy["possible_causes"][0]["checks"][0].endswith("…"))

    def test_duplicate_record_does_not_produce_a_fallback_draft(self):
        source = {**self.source, "chunk_full": SOURCE + "\nDESCRIPTION: Different event."}
        result, selected = self.fallback(source, root_literal_fallback_fields=True)
        self.assertEqual(result["possible_causes"], [])
        self.assertEqual(selected, [])

    def test_existing_root_field_caps_and_complete_solution_boundary(self):
        for length in (320, 321):
            source = {**self.source, "chunk_full": "SOURCE_TYPE: problem_solution\nTITLE: " + "T" * 120
                      + "\nDESCRIPTION: " + "D" * 400 + "\nSOLUTION: " + "S" * length}
            result, _ = self.fallback(source, root_literal_fallback_fields=True)
            cause = result["possible_causes"][0]
            self.assertEqual(len(cause["cause"]), 100)
            self.assertEqual(len(cause["why"]), 320)
            self.assertEqual(len(cause["checks"]), 2 if length == 320 else 1)
            self.assertTrue(all(len(check) <= 320 for check in cause["checks"]))

    def test_generator_default_keeps_legacy_fields_without_extra_provider_call(self):
        result = self.ns["_v13_generate_root_cause_response"](q="Pump delivery stopped.",
            company_id="offline-company", response_language="en", top_k=7, max_causes=3,
            retrieval={"citations": [self.source]}, debug=False)
        self.assertEqual(result["possible_causes"][0]["why"], "REPORTED CONDITION")
        self.assertEqual(len(self.calls), 1)

    def test_core_flag_requires_both_root_modes_and_always_passes_draft_to_reviewer(self):
        for requested, effective in (("root_cause", "root_cause"), ("ask", "root_cause"),
                                     ("smart_diagnostic", "root_cause"), ("root_cause", "ask")):
            self.calls.clear()
            self.reviews.clear()
            request = SimpleNamespace(query="Pump delivery stopped.", company_id="offline-company",
                requested_mode=requested, response_language="en", top_k=7, max_causes=3, debug=False)
            decision = SimpleNamespace(effective_mode=effective, diagnostic_clues=(),
                diagnostic_exclusions=(), missing_information=())
            before = deepcopy(self.source)
            with self.subTest(requested=requested, effective=effective), patch.object(
                    root_time_budget, "producer_window", return_value=nullcontext()):
                result = self.ns["_assistant_core_synthesize_root_cause"](
                    request, {"citations": [self.source]}, decision)
                self.assertEqual(result["possible_causes"], [])
                self.assertTrue(result["review_required"])
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(len(self.reviews), 1)
                draft = self.reviews[0]["response"]["possible_causes"][0]
                if requested == effective == "root_cause":
                    self.assertIn("Pump runs but delivery stops", draft["why"])
                else:
                    self.assertEqual(draft["why"], "REPORTED CONDITION")
                self.assertEqual(self.calls[0][1]["max_output_tokens"], 3400)
                expected_schema = ("root_cause_producer_applicability_v1"
                    if requested == effective == "root_cause" else "root_cause_finder_v1")
                self.assertEqual(self.calls[0][1]["json_schema"]["name"], expected_schema)
                self.assertEqual(self.source, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
