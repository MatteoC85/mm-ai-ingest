"""Citation recovery keeps full private context separate from published sources.

Offline boundary tests with the real citation recovery and claim helpers. The
protected selector/admission pipeline is exercised by its separate integration
suite; these tests isolate the publication policy for conditions-only answers.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import validation


def deny(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class ConditionsCitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, deny))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def fixture(self, kind="procedure", scope="conditions_only", task="procedure_segment"):
        m = self.m
        request = m.AssistantCoreRequest(query="Which precautions apply before insertion?",
            requested_mode="ask", response_language="en", company_id="offline-company",
            machine_id="offline-machine", ai_scope="machine_all", top_k=6)
        decision = m.AssistantCoreDecision(request_kind=kind, effective_mode="ask", confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED,
            evidence_policy="machine_sources_required", information_task=task,
            procedure_scope=scope, required_answer_types=("checklist", "safety_conditions"),
            required_facets=("precautions",), facet_queries=(m.AssistantCoreFacetQuery(
                facet="precautions", answer_type="safety_conditions", must_cover=True),))
        texts = ["Isolate the machine before insertion.", "Wear protective gloves before insertion.",
                 "Transport the pallet using the lifting equipment.", "After insertion, enable the slow trial cycle."]
        rows = [dict(citation_id="step:"+str(i), bubble_document_id="step:"+str(i),
                     source_type="step", ask_structured_direct=True, chunk_full=text,
                     title="Instruction "+str(i), step_number=i)
                for i, text in enumerate(texts, 1)]
        runtime = replace(m._assistant_core_ask_validation_runtime(),
            _sanitize_citations_for_response=lambda items, **kw: list(items),
            _procedure_ui_order_citations=lambda items: list(items),
            _build_rg_links=lambda company, items: [dict(citation_id=r["citation_id"]) for r in items])
        response = dict(status="answered", answer="Isolate the machine and wear protective gloves before insertion.",
                        citations=deepcopy(rows[:2]), _assistant_core_validation_evidence=deepcopy(rows))
        return request, decision, rows, runtime, response

    def recover(self, fixture):
        request, decision, rows, runtime, response = fixture
        return validation.recover_citations(response, request=request, decision=decision,
            retrieval={"candidates": rows}, runtime=runtime)

    def test_conditions_publish_only_supporting_citations_for_procedure_and_factual(self):
        for kind in ("procedure", "general_technical"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                original = deepcopy(fixture[-1])
                sources, links = self.recover(fixture)
                self.assertEqual([r["citation_id"] for r in sources], ["step:1", "step:2"])
                self.assertEqual([r["citation_id"] for r in links], ["step:1", "step:2"])
                self.assertEqual(fixture[-1], original)
                self.assertEqual(len(fixture[-1]["_assistant_core_validation_evidence"]), 4)

    def test_full_execution_and_legacy_preserve_existing_manifest_policy(self):
        for scope, task in (("execution_steps", "procedure_segment"), ("", "procedure_segment"),
                            ("conditions_only", "procedure_full")):
            with self.subTest(scope=scope, task=task):
                sources, _ = self.recover(self.fixture(scope=scope, task=task))
                self.assertEqual([r["citation_id"] for r in sources], ["step:1", "step:2", "step:3", "step:4"])

    def test_conditions_do_not_fill_empty_citations_with_unrelated_family(self):
        fixture = self.fixture()
        request, decision, rows, runtime, response = fixture
        response["answer"] = "No applicable precautions were established."
        response["citations"] = []
        request = replace(request, query="Unresolved precautions")
        sources, links = self.recover((request, decision, rows, runtime, response))
        self.assertEqual(sources, [])
        self.assertEqual(links, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
