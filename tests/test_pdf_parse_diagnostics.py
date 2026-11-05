"""Safe failure context without exposing documents or starting indexing."""
from dataclasses import fields
from decimal import Decimal
from hashlib import sha256
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ingest.orchestration import DocumentIngestRuntime, ingest_document
from machinemind.ingest.text_pdf import PdfPageExtractionError, extract_pages_with_layout_blocks


class PublicError(Exception):
    def __init__(self, *, status_code, detail):
        self.status_code, self.detail = status_code, detail


class PdfParseDiagnosticsTests(unittest.TestCase):
    def test_page_stage_and_close_on_failure_without_content_in_message(self):
        for failing_stage in ("load_page", "extract_layout", "normalize_text"):
            with self.subTest(stage=failing_stage):
                private = "private-document-and-url-content"
                def fail(*args):
                    raise ValueError(private)
                doc = SimpleNamespace(page_count=2, close=Mock(), load_page=lambda i: i)
                extract = lambda page: "readable text"
                normalize = lambda text: text
                if failing_stage == "load_page":
                    doc.load_page = fail
                elif failing_stage == "extract_layout":
                    extract = fail
                else:
                    normalize = fail
                with self.assertRaises(PdfPageExtractionError) as caught:
                    extract_pages_with_layout_blocks(b"pdf", fitz_module=SimpleNamespace(open=lambda **kw: doc),
                        page_to_text_blocks_fn=extract, normalize_text_keep_lines_fn=normalize)
                self.assertEqual((caught.exception.page_number, caught.exception.stage), (1, failing_stage))
                self.assertNotIn(private, str(caught.exception))
                doc.close.assert_called_once_with()

    def make_runtime(self, error, *, logger_fails=False):
        callbacks = {f.name: Mock() for f in fields(DocumentIngestRuntime)}
        callbacks.update(ai_internal_secret="local-test-only", http_exception_cls=PublicError,
            normalize_ai_scope=lambda x: x, normalize_month_key=lambda x: x,
            decimal_value=lambda x: Decimal(str(x or 0)), sha256_fn=sha256,
            effective_request_key=lambda **kw: "request", build_usage_event_id=lambda *args: "event",
            load_document_file=Mock(return_value=dict(data=b"pdf", url="", content_type="application/pdf",
                content_disposition="", detected_filename="private.pdf", detected_extension="pdf", source_mode="inline")),
            detect_source_file_type=lambda *a, **kw: "pdf", max_pdf_bytes=100,
            extract_pdf_pages=Mock(side_effect=error), log_fn=Mock(side_effect=RuntimeError("logger") if logger_fails else None))
        return DocumentIngestRuntime(**callbacks)

    def payload(self):
        return SimpleNamespace(company_id="company", machine_id="machine", bubble_document_id="doc",
            ai_scope="machine_all", ingest_month_key="2026-11", ingest_credits_limit_month=0,
            ingest_credits_enforced=False, ingest_request_already_admitted=False, ingest_request_key="request")

    def test_parse_failure_keeps_public_contract_and_no_index_metering_side_effect(self):
        for error in (ValueError("private-url-token-and-document"), PdfPageExtractionError(32, "extract_layout")):
            with self.subTest(error=type(error).__name__):
                runtime = self.make_runtime(error)
                with self.assertRaises(PublicError) as caught:
                    ingest_document(self.payload(), "local-test-only", runtime=runtime)
                self.assertEqual((caught.exception.status_code, caught.exception.detail), (422, "PDF parse failed"))
                runtime.log_fn.assert_called_once()
                args = runtime.log_fn.call_args.args
                self.assertEqual(args[0], "PDF_PARSE_FAILED")
                self.assertEqual(set(args[1]), {"stage", "page_number", "pages_detected"})
                self.assertNotIn("private", repr(args))
                self.assertEqual(args[1]["page_number"], 32 if isinstance(error, PdfPageExtractionError) else None)
                for name in ("replace_document_pages", "prepare_event", "enqueue_index_task", "invalidate_company_knowledge"):
                    getattr(runtime, name).assert_not_called()

    def test_logger_failure_does_not_replace_parse_error(self):
        runtime = self.make_runtime(ValueError("private"), logger_fails=True)
        with self.assertRaises(PublicError) as caught:
            ingest_document(self.payload(), "local-test-only", runtime=runtime)
        self.assertEqual(caught.exception.detail, "PDF parse failed")
        runtime.enqueue_index_task.assert_not_called()

    def admitted_fixture(self):
        from dataclasses import replace
        runtime = self.make_runtime(None)
        runtime = replace(runtime, extract_pdf_pages=lambda data: ["Readable page content " * 100] * 3,
            detect_repeated_headers_footers=lambda pages: (set(), set()),
            remove_headers_footers_from_page=lambda text, h, f: text,
            reflow_paragraphs=lambda text: text, maybe_remove_toc=lambda text: text,
            min_page_chars=10, min_text_chars=100, min_pages_with_text_abs=1,
            min_pages_with_text_pct=.1, ceil_fn=ceil, chunk_target_chars=3800,
            chunk_overlap_chars=800, month_usage=lambda *a: {}, pricing_version="test", metering_version="test")
        payload = self.payload()
        for field in ("plan_embed_chars_limit_total", "plan_index_storage_limit_bytes", "embed_chars_used_total",
                      "index_storage_used_total", "doc_prev_embed_chars", "doc_prev_index_storage_bytes"):
            setattr(payload, field, 0)
        return runtime, payload

    def test_plan_rejection_after_parse_has_no_metadata_pages_or_index_writes(self):
        for limit in ("plan_embed_chars_limit_total", "plan_index_storage_limit_bytes"):
            with self.subTest(limit=limit):
                runtime, payload = self.admitted_fixture()
                setattr(payload, limit, 1)
                result = ingest_document(payload, "local-test-only", runtime=runtime)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "LIMIT_EXCEEDED")
                for name in ("upsert_document_file", "upsert_cleaning_meta", "replace_document_pages",
                             "prepare_event", "enqueue_index_task", "invalidate_company_knowledge"):
                    getattr(runtime, name).assert_not_called()

    def test_success_persists_cleaning_before_pages_and_queues_once(self):
        runtime, payload = self.admitted_fixture()
        calls = Mock()
        for name in ("upsert_cleaning_meta", "replace_document_pages", "prepare_event", "enqueue_index_task"):
            calls.attach_mock(getattr(runtime, name), name)
        result = ingest_document(payload, "local-test-only", runtime=runtime)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages_total"], 3)
        self.assertEqual([call[0] for call in calls.mock_calls],
            ["upsert_cleaning_meta", "replace_document_pages", "prepare_event", "enqueue_index_task"])

    def test_metadata_failure_remains_nonblocking_and_content_free(self):
        runtime, payload = self.admitted_fixture()
        runtime.upsert_cleaning_meta.side_effect = ValueError("private-database-detail")
        result = ingest_document(payload, "local-test-only", runtime=runtime)
        self.assertTrue(result["ok"])
        self.assertEqual(runtime.log_fn.call_args.args,
            ("CLEANING_META_UPSERT_FAIL", "metadata_write_failed"))
        runtime.enqueue_index_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
