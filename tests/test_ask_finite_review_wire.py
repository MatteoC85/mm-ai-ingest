"""Offline replay of finite-condition verdicts and the real owned source flow."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import finite_review_wire as wire
from machinemind.ask import execution, review_source_wire
from machinemind.infrastructure import request_budget
import test_ask_procedure_context_recovery as actual
import test_ask_router_family_context as route


def encode_verdict(reply, frame):
    """Model-frame fixture. The production code only decodes verdicts."""
    out = deepcopy(reply)
    for names, pairs in (
        (("citation_ids",), frame.source_labels),
        (("covered_facets", "missing_facets"), frame.facet_labels),
        (("covered_answer_types", "missing_answer_types"), frame.type_labels),
    ):
        labels = dict(pairs)
        for name in names:
            out[name] = [labels.get(value, value) for value in out[name]]
    for change in out.get("operation_boundary", {}).get("changes", []):
        field = "closing_source_citation_id"
        change[field] = dict(frame.source_labels)[change[field]]
    if frame.checklist_positions:
        for name in ("covered_list_items", "missing_list_items"):
            out[name] = [out["expected_list_items"].index(value) + 1 for value in out[name]]
    return out


def payload_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


class FiniteReviewWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        actual.AskProcedureContextRecoveryTests.setUpClass()
        cls.m = actual.AskProcedureContextRecoveryTests.m

    @classmethod
    def tearDownClass(cls):
        actual.AskProcedureContextRecoveryTests.tearDownClass()

    def test_archived_verdicts_round_trip_with_unchanged_full_answer_and_list_coverage(self):
        work = Path(__file__).resolve().parents[2]
        for revision, date in (("r5", 9), ("r9", 9), ("r11", 10), ("r12", 10)):
            with self.subTest(revision=revision):
                receipt = work / f"phase6-ask-correction-{revision}-202611{date:02}"
                response = json.loads((receipt / "live-client" / "receipts" /
                    f"{revision.upper()}-A-ASK-02.json").read_text(encoding="utf-8-sig"))["response"]
                meta = response["meta"]
                contract = meta["assistant_core_validation"]["answer_contract"]
                # Scoped normalized fields retained from the recorded verifier.
                # Some older receipts lack the original review source packet.
                reply = contract["semantic_verifier"]
                reply = {key: value for key, value in reply.items() if key != "review_evidence"}
                ids = tuple(reply["citation_ids"])
                frame = wire.prepare("\n\n".join(f"[{cid}] source_type=step\nLiteral fixture." for cid in ids),
                    citation_ids=ids, facets=tuple(meta["assistant_core_v2"]["required_facets"]),
                    types=tuple(contract["required_answer_types"]), question="Recorded private ASK",
                    answer=response["answer"])
                encoded = encode_verdict(reply, frame)
                self.assertEqual(frame.decode(encoded), reply)
                self.assertEqual(encoded["answer"], reply["answer"])
                self.assertLess(payload_bytes(encoded), payload_bytes(reply))

    def test_unknown_foreign_displaced_and_malformed_labels_are_rejected(self):
        cid = "step:valid:local:p1-1"
        frame = wire.prepare(f"[{cid}] source_type=step\nClose the guard.",
            citation_ids=(cid,), facets=("closing precaution",), types=("checklist",),
            question="Which checks apply?", answer="Close the guard.")
        reply = {"outcome": "rewrite", "answer": "Close the guard.", "reason": "real correction",
            "citation_ids": ["@s1"], "covered_facets": ["@f1"], "missing_facets": ["unplanned obligation"],
            "covered_answer_types": ["@t1"], "missing_answer_types": [],
            "operation_boundary": {"complete": True, "checked_blocks": ["b1"],
                "changes": [{"closing_source_citation_id": "@s1", "closing_source_quote": "Close the guard."}]}}
        resolved = frame.decode(reply)
        self.assertEqual(resolved["missing_facets"], ["unplanned obligation"])
        self.assertEqual(resolved["operation_boundary"]["changes"][0], {
            "closing_source_citation_id": cid, "closing_source_quote": "Close the guard."})
        for field, bad in (("citation_ids", ["@s2"]), ("citation_ids", ["foreign"]),
                           ("citation_ids", ["@f1"]), ("covered_facets", ["@f2"]),
                           ("covered_answer_types", ["@f1"]), ("missing_facets", [1]),
                           ("answer", "Close the guard @s1."), ("reason", "unknown @t2")):
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                frame.decode({**reply, field: bad})
        reason = frame.decode({**reply, "reason": "Verified @f1 and @t1 in @s1."})
        self.assertEqual(reason["reason"], "Verified closing precaution and checklist in cited source.")
        corrupt = deepcopy(reply)
        corrupt["operation_boundary"]["changes"][0]["closing_source_citation_id"] = "@s2"
        with self.assertRaises(ValueError):
            frame.decode(corrupt)

    def test_ambiguous_or_colliding_source_headers_keep_legacy_input(self):
        cid = "step:local:p1-1"
        header = f"[{cid}] source_type=step"
        common = dict(citation_ids=(cid,), facets=("mass",), types=("numeric_value",), question="Draft?", answer="10 kg.")
        for sources in (f"{header}\n{header}\nBody", f"[foreign] source_type=step\nBody",
                        f"{header}\nManual shows literal @s1", f"Unbound preface\n{header}\nBody"):
            with self.subTest(sources=sources):
                self.assertIsNone(wire.prepare(sources, **common))
        self.assertIsNone(wire.prepare(f"{header}\nBody", **{**common, "answer": "Literal @f1"}))

    def test_checklist_positions_preserve_explicit_coverage_and_reject_invalid_indices(self):
        cid = "step:document:p1-1"
        frame = wire.prepare(f"[{cid}] source_type=step\nSource notes.", citation_ids=(cid,),
            facets=("checks",), types=("checklist",), question="What checks apply?",
            answer="Check the guard.", checklist_positions=True)
        reply = {
            "outcome": "rewrite", "answer": "Check the guard.", "citation_ids": [cid],
            "covered_facets": [], "missing_facets": ["checks"],
            "covered_answer_types": [], "missing_answer_types": ["checklist"],
            "expected_list_items": ["Check the guard", "Verify the recorded shutdown"],
            "covered_list_items": ["Check the guard"], "missing_list_items": ["Verify the recorded shutdown"],
        }
        encoded = encode_verdict(reply, frame)
        self.assertEqual(frame.decode(encoded), reply)
        self.assertLess(payload_bytes(encoded), payload_bytes(reply))
        for field, invalid in (("covered_list_items", [True]), ("covered_list_items", [0]),
                               ("missing_list_items", [3]), ("missing_list_items", [1, 2]),
                               ("missing_list_items", ["Verify the recorded shutdown"])):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                frame.decode({**encoded, field: invalid})

    def test_required_checklist_enumeration_keeps_independent_list_coverage(self):
        m = self.m
        request = m.AssistantCoreRequest(query="Which checks apply before loading and what mass is allowed?",
            requested_mode="ask", response_language="en", company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=6)
        types = ("numeric_value", "checklist")
        facets = ("maximum documented load mass including unit and applicable configuration",
                  "complete required initial checks with distinct labels before the stated operation")
        decision = m.AssistantCoreDecision(request_kind="factual", effective_mode="ask", confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy="evidence_required",
            information_task="procedure_segment", procedure_scope="conditions_only",
            required_answer_types=types, required_facets=facets,
            facet_queries=tuple(m.AssistantCoreFacetQuery(facet=label, answer_type=kind, must_cover=True)
                                for label, kind in zip(facets, types, strict=True)))
        ids = [f"manual-{chr(97 + index) * 120}:p2-2:c0" for index in range(4)]
        body = "Check cable integrity. Use eye protection. Verify protective earth continuity. Maximum mass: 1000 kg."
        rows = [{"citation_id": cid, "bubble_document_id": cid.split(":")[0], "chunk_full": body,
                 "source_type": "document", "page_from": 2, "page_to": 2} for cid in ids]
        sources = m._v13_sources_block(rows, max_context_chars=22000)
        diagnostic = {"complete_primary_representation": True, "reused_request_packet": False,
                      "required_records": len(rows), "emitted_citation_ids": ids}
        observer = SimpleNamespace(consume=lambda _method, _parameters, call: call(),
            procedure_review_sources=lambda _rows, **_kwargs: (rows, sources, diagnostic))
        draft = "Check cable integrity. Use eye protection. The maximum mass is 1000 kg."

        def review(items, *, alternate=decision, enable_aliases=True, catalog=""):
            def provider(messages, **options):
                content = messages[1]["content"]
                self.assertIn("ENUMERATION_REQUESTED: true", content)
                has_aliases = "ANSWER_TYPE_LABELS:" in content
                self.assertEqual(has_aliases, enable_aliases)
                if has_aliases:
                    item_schema = options["json_schema"]["schema"]["properties"]["covered_list_items"]["items"]
                    self.assertEqual(item_schema, {"type": "integer", "minimum": 1, "maximum": 40})
                    frame = wire.prepare(sources, citation_ids=tuple(ids), facets=facets, types=types,
                                         question=request.query, answer=draft, checklist_positions=True)
                    source = content.split("\n\nSOURCES:\n", 1)[1].split("\n\nReturn only the required JSON.", 1)[0]
                    self.assertTrue(review_source_wire.restore_sources_block(source) == frame.sources)
                reply = {"outcome": "pass", "reason": "Offline source review", "answer": "",
                         "citation_ids": [ids[0]], "covered_facets": list(facets), "missing_facets": [],
                         "covered_answer_types": list(types), "missing_answer_types": [],
                         "enumeration_requested": True, "expected_list_items": items,
                         "covered_list_items": items, "missing_list_items": []}
                return (encode_verdict(reply, frame) if has_aliases else reply), "offline-reviewer"

            runtime = replace(m._assistant_core_ask_execution_runtime(), evidence_admission=lambda *args: None,
                evidence_observer=observer, _procedure_context_available=lambda: True,
                _v13_current_budget=lambda: m._assistant_core_new_budget("ask"), _v13_json_models=provider,
                _assistant_core_enumeration_requested=lambda *_: True,
                _assistant_core_machine_catalog_digest=lambda _: catalog)
            with (patch.object(execution, "_admit_input", side_effect=lambda _req, data, _dec, **_kwargs: data),
                  patch.object(execution, "_current_completion", return_value=object())):
                return execution.verify_or_repair_answer(request=request, decision=alternate, answer=draft,
                                                         candidates=rows, runtime=runtime)

        listed = ["Check cable integrity", "Use eye protection"]
        accepted = review(listed)
        self.assertEqual(accepted["expected_list_items"], listed)
        self.assertEqual(accepted["outcome"], "pass")
        missing = "Verify protective earth continuity"
        verdict = review([*listed, missing])
        self.assertEqual(verdict["missing_list_items"], [missing])
        self.assertEqual(verdict["reason"], "missing_supported_list_items")
        self.assertNotEqual(verdict["outcome"], "pass")
        self.assertEqual(review(listed, enable_aliases=False, catalog="admitted machine inventory")["outcome"], "pass")
        self.assertEqual(review(listed, enable_aliases=False,
            alternate=replace(decision, procedure_scope=""))["outcome"], "pass")

    def test_line_dictionary_keeps_each_source_header_and_restores_literal_text(self):
        ids = tuple(f"manual-{index}:p{index}-{index}:c0" for index in range(1, 4))
        shared = "Before work, verify the recorded isolation for this configuration. " * 9
        original = "\n\n".join(
            f"[{cid}] source_type=document; doc=manual-{index}; p={index}-{index}\n"
            f"Applicable configuration {index}.\n{shared}\nPreserve this location."
            for index, cid in enumerate(ids, 1))
        frame = wire.prepare(original, citation_ids=ids, facets=("initial checks",), types=("checklist",),
                             question="Which checks apply?", answer="Review the existing record.")
        compressed = review_source_wire.compact_sources_block(frame.sources)
        self.assertLess(payload_bytes(compressed), payload_bytes(frame.sources))
        self.assertEqual(review_source_wire.restore_sources_block(compressed), frame.sources)
        self.assertEqual(re.findall(r"^\[(@s[0-9]+)\] source_type=", compressed, flags=re.MULTILINE),
                         [label for _, label in frame.source_labels])

    def test_actual_owned_core_first_review_uses_shorter_input_and_same_semantic_guards(self):
        captures = {False: [], True: []}
        original = route.RouterFixture.run_core
        prepare_wire = wire.prepare
        now = request_budget._monotonic()
        for enabled in captures:
            def wrapped(fixture):
                reviewer = fixture.reviewer
                def provider(messages, **options):
                    reply, model = reviewer(messages, **options)
                    packet = fixture.owner._response_flow._review_packet
                    sources = messages[1]["content"].partition("\n\nSOURCES:\n")[2]
                    if "FACET_LABELS:" in messages[1]["content"]:
                        labels = {line.partition(": ")[0]: json.loads(line.partition(": ")[2])
                            for line in messages[1]["content"].splitlines()
                            if line.startswith(("FACET_LABELS: ", "ANSWER_TYPE_LABELS: "))}
                        source_ids = tuple(row["citation_id"] for row in packet.rows)
                        frame = prepare_wire(packet.sources, citation_ids=source_ids,
                            facets=tuple(labels["FACET_LABELS"].values()),
                            types=tuple(labels["ANSWER_TYPE_LABELS"].values()), question="", answer="",
                            checklist_positions=options["json_schema"]["schema"]["properties"]
                                ["covered_list_items"]["items"]["type"] == "integer")
                        source_wire = sources.partition("\n\nReturn only the required JSON.")[0]
                        self.assertEqual(review_source_wire.restore_sources_block(source_wire), frame.sources)
                        restored = re.sub(r"^\[(@s[0-9]+)\] source_type=",
                            lambda match: "[" + dict((alias, cid) for cid, alias in frame.source_labels)[match[1]]
                            + "] source_type=", frame.sources, flags=re.MULTILINE)
                        self.assertEqual(restored, packet.sources)
                        reply = encode_verdict(reply, frame)
                    captures[enabled].append((messages, options, packet.sources))
                    return reply, model
                fixture.reviewer = provider
                return original(fixture)
            case = actual.AskProcedureContextRecoveryTests()
            # Compare the same remaining budget in both presentation arms.
            # CPU scheduling must not change their integer reviewer timeout.
            with (patch.object(route.RouterFixture, "run_core", wrapped),
                  patch.object(wire, "prepare", prepare_wire if enabled else lambda *args, **kwargs: None),
                  patch.object(request_budget, "_monotonic", return_value=now)):
                case.test_actual_r7_required_parent_type_keeps_complete_family_and_manual_support()
        for index, expected in enumerate((True, False, True)):
            old, new = (captures[enabled][index] for enabled in (False, True))
            self.assertEqual(old[2], new[2])
            self.assertEqual("FACET_LABELS:" in new[0][1]["content"], expected)
            # Independent model, effort, timeout, purpose and complete-output cap.
            for key in ("models", "effort", "timeout", "purpose", "max_output_tokens", "require_full_output"):
                self.assertEqual(old[1].get(key), new[1].get(key))
            if expected:
                self.assertLess(payload_bytes((new[0], new[1]["json_schema"])),
                                payload_bytes((old[0], old[1]["json_schema"])))
            else:
                self.assertEqual(old[:2], new[:2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
