"""Offline synthesis-context regressions; no model, credentials or database."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import root_source_packet, source_management


RUNTIME = source_management.V13SourcesBlockRuntime(
    _source_type_from_document_id=lambda document: "document",
    _v13_candidate_text=lambda candidate: candidate.get("chunk_full", ""))


def candidate(index, text, **extra):
    return {"citation_id": f"manual-{index}:p1", "bubble_document_id": f"manual-{index}",
            "company_id": "company", "machine_id": "machine", "source_type": "document",
            "page_from": 1, "page_to": 1, "exact_machine_scope": True,
            "chunk_full": text, **extra}


def pack(rows, limit=18000):
    return source_management.v13_root_sources_packet(rows, max_context_chars=limit, runtime=RUNTIME)


class RootSourcePacketTests(unittest.TestCase):
    def test_long_early_sources_no_longer_hide_later_documented_mechanism(self):
        rows = [candidate(i, ("Existing record provides detailed machine context and conditional checks. " * 60)) for i in range(6)]
        last = candidate(6, "During setup the isolation latch can hold the drive coupling open. "
                         "The actuator then moves without transmitting travel to the carriage. "
                         "This is a possible operating state, not a confirmed cause.\n" + "Observe the applicable operating conditions. " * 12)
        rows.append(last)
        old = source_management.v13_sources_block(rows, max_context_chars=18000, runtime=RUNTIME)
        self.assertNotIn(last["citation_id"], old)  # Reproduce first-fit starvation.
        result = pack(rows)
        sources = json.loads(result["sources_block"])["sources"]
        self.assertEqual([row["citation_id"] for row in sources], [row["citation_id"] for row in rows])
        self.assertEqual(sources[-1]["text"], last["chunk_full"])
        self.assertTrue(sources[-1]["text_complete"])
        self.assertLessEqual(len(result["sources_block"]), 18000)
        self.assertEqual(result["summary"]["emitted_source_count"], 7)

    def test_short_sources_return_unused_budget_to_longer_sources(self):
        rows = [candidate(1, "Short complete conditional mechanism."), candidate(2, "Literal longer source context. " * 500)]
        result = pack(rows, 4000)
        sources = json.loads(result["sources_block"])["sources"]
        self.assertEqual(sources[0]["text"], rows[0]["chunk_full"])
        self.assertGreater(len(sources[1]["text"]), 2800)
        self.assertFalse(sources[1]["text_complete"])

    def test_long_sources_share_budget_without_rank_starvation(self):
        rows = [candidate(i, f"Source {i} retains its own conditions. " * 600) for i in range(8)]
        for budget in (8000, 18000, 38000):
            with self.subTest(budget=budget):
                result = pack(rows, budget)
                sources = json.loads(result["sources_block"])["sources"]
                sizes = [len(row["text"]) for row in sources]
                self.assertEqual(len(sources), 8)
                self.assertGreater(min(sizes), 200)
                self.assertLess(max(sizes) - min(sizes), 40)
                self.assertLessEqual(len(result["sources_block"]), budget)

    def test_bound_includes_json_escaping_and_unicode(self):
        rows = [candidate(i, 'Pressure "P" at 12 MPa: sì, è documentata.\n' * 300) for i in range(8)]
        result = pack(rows, 8000)
        self.assertLessEqual(len(result["sources_block"]), 8000)
        for source, original in zip(json.loads(result["sources_block"])["sources"], rows):
            self.assertTrue(original["chunk_full"].startswith(source["text"]))
            self.assertFalse(source["text_complete"])

    def test_literal_containment_dedup_uses_longer_sources_own_citation(self):
        short = candidate(1, "A worn coupling permits relative shaft motion.", source_type="ps")
        long = {**short, "citation_id": "ps-expanded", "chunk_full": "Service case: " + short["chunk_full"] + " Compare the shaft positions."}
        result = pack([short, long])
        sources = json.loads(result["sources_block"])["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["citation_id"], "ps-expanded")
        self.assertEqual(sources[0]["text"], long["chunk_full"])
        self.assertEqual(result["summary"]["deduplicated_sources"][0]["removed_citation_id"], short["citation_id"])

    def test_distinct_same_page_mechanisms_are_not_deduplicated(self):
        first = candidate(1, "A leaking joint can admit air and reduce delivery.")
        second = {**first, "citation_id": "another-mechanism", "chunk_full": "A clogged filter can restrict flow and reduce delivery."}
        result = pack([first, second])
        self.assertEqual(result["summary"]["emitted_source_count"], 2)

    def test_never_dedup_across_scope_source_page_or_case_sensitive_units(self):
        first = candidate(1, "Compare the recorded threshold of 12 MPa.")
        for changed in ({"company_id": "another"}, {"machine_id": "another"}, {"bubble_document_id": "another"},
                        {"source_type": "ps"}, {"page_from": 2, "page_to": 2},
                        {"chunk_full": "Compare the recorded threshold of 12 mPa."}):
            with self.subTest(changed=changed):
                second = {**first, "citation_id": "second", **changed}
                self.assertEqual(pack([first, second])["summary"]["emitted_source_count"], 2)

    def test_unknown_scope_does_not_become_duplicate_provenance(self):
        first = candidate(1, "A conditional mechanism from the selected excerpt.")
        first.pop("company_id")
        second = {**first, "citation_id": "second"}
        self.assertEqual(pack([first, second])["summary"]["emitted_source_count"], 2)

    def test_no_input_rewriting_and_summary_contains_no_source_text_or_url(self):
        rows = [candidate(1, "Literal source includes https://example.invalid/reference for documentation.")]
        before = deepcopy(rows)
        result = pack(rows)
        self.assertEqual(rows, before)
        self.assertNotIn("https://", json.dumps(result["summary"]))
        self.assertNotIn("Literal source", json.dumps(result["summary"]))
        self.assertEqual(result["summary"]["sources"][0]["emitted_chars"], len(rows[0]["chunk_full"]))

    def test_empty_source_and_impossibly_small_budget_remain_explicit(self):
        self.assertEqual(pack([candidate(1, "")])["sources_block"], "")
        with self.assertRaisesRegex(root_source_packet.SourcePacketError, "root_source_metadata_exceeds_budget"):
            pack([candidate(1, "A documented mechanism.")], 5)

    def test_actual_root_synthesis_uses_fair_packet_and_emits_safe_telemetry(self):
        # Compile ONLY the production synthesis function into a controlled runtime,
        # avoiding main.py startup, clients, migrations and provider dependencies.
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_v13_generate_root_cause_response")
        calls = []
        rows = [candidate(i, "Detailed existing machine context. " * 150) for i in range(7)]
        rows[-1]["chunk_full"] = "A documented selected-source mechanism explains the observed transmission loss."

        def model(messages, **kwargs):
            calls.append((messages, kwargs))
            return {"problem_summary": "Qualified hypothesis", "possible_causes": [{"cause": "Documented hypothesis",
                    "why": "The documented mechanism can explain the observation.", "checks": [],
                    "citations": [rows[-1]["citation_id"]]}]}, "fixture-model"

        namespace = {"_retrieval_source_management": source_management,
            "_source_type_from_document_id": RUNTIME._source_type_from_document_id,
            "_v13_candidate_text": RUNTIME._v13_candidate_text,
            "_dedup_text_values": lambda values, **kwargs: values,
            "_v13_root_cause_model": lambda q, retrieval: ("fixture-model", "low", ""),
            "_v13_json_models": model, "_root_cause_response_schema": lambda **kwargs: {},
            "_v13_assurance_prompt_block": lambda retrieval: "",
            "_ground_root_cause_result": lambda **kwargs: (kwargs["result"], kwargs["citations"]),
            "_compact_root_cause_result_citations_by_family": lambda **kwargs: (kwargs["result"], kwargs["citations"]),
            "_lock_root_cause_result": lambda result, citations, **kwargs: (result, citations),
            "_sanitize_citations_for_response": lambda citations, **kwargs: citations,
            "_build_rg_links": lambda company, citations: [],
            "V13_FAST_MODEL": "fixture-model", "V13_HEAVY_MODEL": "fixture-heavy",
            "V13_FAST_CONTEXT_CHARS": 18000, "V13_HEAVY_CONTEXT_CHARS": 38000,
            "V13_FAST_TIMEOUT_SECONDS": 20, "V13_HEAVY_TIMEOUT_SECONDS": 30,
            "V13_FAST_MAX_OUTPUT_TOKENS": 3000, "V13_HEAVY_MAX_OUTPUT_TOKENS": 4000,
            "V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE": 14, "json": json}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        response = namespace[fn.name](q="Observed loss", company_id="company", response_language="en",
                                      top_k=8, max_causes=3, retrieval={"citations": rows}, debug=False)
        self.assertEqual(len(calls), 1)
        prompt = calls[0][0][1]["content"]
        payload = json.loads(prompt.split("SOURCES:\n", 1)[1].split("\n\nReturn JSON", 1)[0])
        self.assertIn(rows[-1]["citation_id"], [row["citation_id"] for row in payload["sources"]])
        self.assertIn("return one cause", calls[0][0][0]["content"])
        self.assertEqual(response["meta"]["root_synthesis_packet"]["emitted_source_count"], 7)
        self.assertLessEqual(response["meta"]["root_synthesis_packet"]["evidence_payload_chars"], 18000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
