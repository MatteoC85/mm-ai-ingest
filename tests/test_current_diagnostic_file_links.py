"""Offline current-file refresh: scope, privacy, page identity and no stale fallback."""
from copy import deepcopy
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.authority import service_files
from machinemind.authority.contracts import AuthorityError
from machinemind.evidence.contracts import SourceType


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


class DirectoryFixture:
    def __init__(self, records):
        self.records, self.calls, self.closed = records, [], False

    def get(self, typename, uid):
        self.calls.append((typename, uid))
        value = self.records.get((typename, uid))
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    def close(self):
        self.closed = True


class DiagnosticFileLinksTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, "connect", denied)
        self.network.start()
        self.addCleanup(self.network.stop)
        sources = [{"source_type": kind.value, "typename": kind.value,
            "company_field": "company", "machine_field": "machine", "deleted_field": "deleted",
            **({"parent_field": "procedure"} if kind == SourceType.STEP else {})}
            for kind in SourceType]
        schema = {"company_type": "company", "machine_type": "machine",
            "machine_company_field": "company", "deleted_false_or_blank": False,
            "sources": sources}
        self.env = {"MM_ASK_REQUEST_AUTHORITY": "required",
            "MM_BUBBLE_AUTHORITY_BASE_URL": "https://app.example.test/api/1.1/obj",
            "MM_BUBBLE_AUTHORITY_HOST": "app.example.test",
            "MM_BUBBLE_AUTHORITY_TOKEN": "offline-unused",
            "MM_BUBBLE_AUTHORITY_SCHEMA_JSON": json.dumps(schema),
            "MM_AUTHORITY_LIMITS_JSON": json.dumps({"max_http_calls": 100,
                "max_response_bytes": 100000, "max_total_bytes": 1000000,
                "max_records": 100, "page_size": 100, "timeout_seconds": 10,
                "total_seconds": 60})}
        self.native = "https://app.example.test/version-live/fileupload/file-a/manual.pdf"
        self.records = {("company", "company-a"): {"_id": "company-a"},
            ("machine", "machine-a"): {"_id": "machine-a", "company": "company-a"},
            ("document", "doc-a"): {"_id": "doc-a", "company": "company-a",
                "machine": "machine-a", "deleted": False, "file": self.native}}
        self.citations = [{"citation_id": "doc-a:p54-54:v13page", "bubble_document_id": "doc-a",
            "source_type": "document", "page_from": 54, "page_to": 54, "snippet": "Supported mechanism"},
            {"citation_id": "ps:problem-a:p1", "bubble_document_id": "ps:problem-a",
             "source_type": "ps", "page_from": 1, "page_to": 1}]
        self.links = [{**self.citations[0], "url": "https://cdn.example.test/old.pdf?Signature=expired#page=54"},
            {**self.citations[1], "url": "https://app.example.test/problem/problem-a"}]
        self.directory = DirectoryFixture(self.records)
        self.transport_limits = None

    def factory(self, *, connection, meter):
        self.transport_limits = meter.limits
        return self.directory

    def refresh(self):
        return service_files.refresh_diagnostic_links(company_id="company-a", machine_id="machine-a",
            citations=self.citations, rg_links=self.links, env=self.env, directory_factory=self.factory)

    def test_refreshes_exact_current_native_url_and_pdf_page_without_mutating_inputs(self):
        before = deepcopy((self.citations, self.links))
        result = self.refresh()
        self.assertEqual(result[0]["url"], self.native + "#page=54")
        self.assertEqual(result[1], self.links[1])
        self.assertEqual({k: v for k, v in result[0].items() if k != "url"},
                         {k: v for k, v in self.links[0].items() if k != "url"})
        self.assertEqual((self.citations, self.links), before)
        self.assertEqual(self.directory.calls, [("company", "company-a"),
            ("machine", "machine-a"), ("document", "doc-a")])
        self.assertTrue(self.directory.closed)
        self.assertEqual(self.transport_limits.max_http_calls, 3)
        self.assertEqual(self.transport_limits.timeout_seconds, 3)
        self.assertEqual(self.transport_limits.total_seconds, 8)

    def test_multiple_pages_read_document_once_and_preserve_each_page(self):
        citation = {**self.citations[0], "citation_id": "doc-a:p50", "page_from": 50, "page_to": 50}
        self.citations.append(citation)
        self.links.append({**citation, "url": "https://cdn.example.test/expired"})
        result = self.refresh()
        self.assertEqual(result[2]["url"], self.native + "#page=50")
        self.assertEqual(self.directory.calls.count(("document", "doc-a")), 1)

    def test_foreign_tenant_foreign_machine_deleted_and_missing_document_fail_closed(self):
        for changed in ({"company": "other-company"}, {"machine": "other-machine"},
                        {"deleted": True}, {"_id": "different-id"}):
            with self.subTest(changed=changed):
                saved = deepcopy(self.records[("document", "doc-a")])
                self.records[("document", "doc-a")].update(changed)
                with self.assertRaises(AuthorityError):
                    self.refresh()
                self.records[("document", "doc-a")] = saved
        self.records.pop(("document", "doc-a"))
        with self.assertRaises(AuthorityError):
            self.refresh()

    def test_machine_must_currently_belong_to_requested_company(self):
        self.records[("machine", "machine-a")]["company"] = "other-company"
        with self.assertRaises(AuthorityError):
            self.refresh()
        self.assertNotIn(("document", "doc-a"), self.directory.calls)

    def test_company_general_source_retains_same_tenant_boundary(self):
        self.records[("document", "doc-a")]["machine"] = None
        self.assertEqual(self.refresh()[0]["url"], self.native + "#page=54")

    def test_never_strips_signatures_or_accepts_foreign_or_credentialed_url(self):
        for bad in ("https://cdn.example.test/manual.pdf?Signature=fresh",
                    self.native + "?Signature=unexpected", self.native + "#page=1",
                    "https://user:password@app.example.test/fileupload/manual.pdf", None):
            with self.subTest(kind="invalid_reference"):
                self.records[("document", "doc-a")]["file"] = bad
                with self.assertRaises(AuthorityError):
                    self.refresh()

    def test_unselected_document_or_page_is_denied_before_any_read(self):
        for changed in ({"bubble_document_id": "unselected"}, {"page_from": 99}):
            with self.subTest(changed=changed):
                saved = deepcopy(self.links[0])
                self.links[0].update(changed)
                with self.assertRaises(AuthorityError):
                    self.refresh()
                self.assertEqual(self.directory.calls, [])
                self.links[0] = saved

    def test_provider_failure_has_no_stale_url_fallback(self):
        self.records[("document", "doc-a")] = AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
        with self.assertRaises(AuthorityError):
            self.refresh()
        self.assertTrue(self.directory.closed)

    def test_off_mode_and_structured_only_do_not_read_directory(self):
        self.env = {"MM_ASK_REQUEST_AUTHORITY": "off"}
        self.assertIs(self.refresh(), self.links)
        self.assertEqual(self.directory.calls, [])
        self.env = {"MM_ASK_REQUEST_AUTHORITY": "required"}
        self.citations, self.links = self.citations[1:], self.links[1:]
        self.assertIs(self.refresh(), self.links)
        self.assertEqual(self.directory.calls, [])

    def application(self):
        guards = ExitStack()
        self.addCleanup(guards.close)
        guards.enter_context(patch.dict(os.environ, {"AI_INTERNAL_SECRET": "offline-files-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_USAGE_ENFORCEMENT": "off",
            "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        guards.enter_context(patch("requests.sessions.Session.request", denied))
        guards.enter_context(patch("psycopg2.connect", denied))
        module = importlib.import_module("main")
        guards.enter_context(patch.object(module, "AI_INTERNAL_SECRET", "offline-files-fixture-only"))
        guards.enter_context(patch.dict(os.environ, self.env))
        guards.enter_context(patch.object(service_files, "BubbleDirectory", side_effect=self.factory))
        return module

    def test_smart_refresh_precedes_flat_fields_json_and_signed_next_state(self):
        m = self.application()
        question = {"question_id": "Q1", "question_number": 1, "question_type": "yes_no",
            "question_text": "Does the documented indicator change?", "options": m._sd_default_yes_no_options("en")}
        response = m._sd_response_from_step(session_id="session-a", company_id="company-a",
            machine_id="machine-a", symptom_text="Observed stop", language="en",
            state={"history": [], "evidence": deepcopy(self.citations)},
            step={"status": "in_progress", "question": question, "hypotheses": []},
            citations=deepcopy(self.citations), rg_links=deepcopy(self.links))
        self.assertEqual(response["c1_url"], self.native + "#page=54")
        self.assertEqual(json.loads(response["rg_links_json"]), response["rg_links"])
        state = json.loads(response["session_state_json"])
        self.assertEqual(state["rg_links"], response["rg_links"])
        self.assertEqual(state["evidence"], self.citations)
        m._sd_validate_state_binding(state, company_id="company-a", machine_id="machine-a",
            session_id="session-a", question_id="Q1")
        self.assertNotIn("Signature=expired", response["session_state_json"])

    def test_root_cause_refresh_preserves_result_and_failure_keeps_known_accounting(self):
        m = self.application()
        payload = m.RootCauseRequest(company_id="company-a", machine_id="machine-a", query="Observed stop")
        original = {"ok": True, "status": "answered", "problem_summary": "Supported mechanism",
            "citations": deepcopy(self.citations), "rg_links": deepcopy(self.links),
            "meta": {"v13_elapsed_seconds": 10, "v13_llm_calls": 2,
                "v13_estimated_cost_usd": .04, "v13_committed_cost_usd": .04,
                "v13_uncertain_cost_usd": 0, "v13_accounting_complete": True}}
        with patch.object(m, "_assistant_core_sync", return_value=original):
            response = m._assistant_core_root_cause_sync(payload, "offline-files-fixture-only")
        self.assertEqual(response["rg_links"][0]["url"], self.native + "#page=54")
        self.assertEqual(response["meta"], original["meta"])
        self.assertEqual(response["citations"], original["citations"])
        self.records[("document", "doc-a")]["company"] = "other-company"
        with patch.object(m, "_assistant_core_sync", return_value=original):
            response = m._assistant_core_root_cause_sync(payload, "offline-files-fixture-only")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["result_code"], "AUTHORITY_FILE_SCOPE_CHANGED")
        self.assertEqual(response["citations"], [])
        self.assertEqual(response["rg_links"], [])
        self.assertEqual(response["meta"]["v13_committed_cost_usd"], .04)
        self.assertTrue(response["meta"]["v13_accounting_complete"])

    def test_smart_turn_file_failure_returns_empty_error_with_runtime_accounting(self):
        m = self.application()
        self.records[("document", "doc-a")]["deleted"] = True
        def work(payload, secret):
            return m._sd_response_from_step(session_id="session-a", company_id="company-a",
                machine_id="machine-a", symptom_text="Observed stop", language="en", state={},
                step={"status": "in_progress", "question": {}, "hypotheses": []},
                citations=deepcopy(self.citations), rg_links=deepcopy(self.links))
        payload = SimpleNamespace(company_id="company-a", language="en", debug=False)
        with patch.object(m, "ASSISTANT_CORE_V2_ENABLED", True), patch.object(
                m, "_assistant_core_run_smart_with_hard_timeout",
                side_effect=lambda func, payload, secret, **kw: func(payload, secret)):
            response = m._assistant_core_budgeted_sd_turn("answer")(work)(payload, "offline-files-fixture-only")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["result_code"], "AUTHORITY_FILE_REVOKED")
        self.assertFalse(response["final_ready"])
        self.assertEqual(response["session_state_json"], "")
        self.assertEqual(response["citations"], [])
        self.assertEqual(response["c1_url"], "")
        self.assertEqual(response["meta"]["v13_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
