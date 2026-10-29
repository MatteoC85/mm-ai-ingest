"""Offline retrieval regressions for manual detail of selected procedure segments.

The fixture pages are synthetic. No model verdict is mocked as quality evidence:
these tests exercise the real bounded manual reader and source vocabulary only.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_EXTERNAL_IO_DENIED")


def source(kind, key, title, description, number=None):
    text = f"SOURCE_TYPE: {kind}\nTITLE: {title}\n"
    if kind == "procedure":
        text += "PROCEDURE_TYPE: setup\nSHORT_DESCRIPTION: " + description
    else:
        text += f"STEP_NUMBER: {number}\nDESCRIPTION: {description}"
    return {"citation_id": key + ":source", "bubble_document_id": kind + ":" + key,
            "source_type": kind, "chunk_full": text}


def selected_segment():
    return [source("procedure", "cartridge-install", "Cartridge commissioning",
                   "Planning logistics warehouse receipt docket serial register inventory allocation "
                   "dispatch packaging capacity delivery date supervisor signoff. " * 5),
            source("step", "isolate", "Isolate the supply", "OPERATIONAL ACTION\nIsolate the supply.\n"
                   "SAFETY NOTE\nUse qualified personnel and maintain isolation.", 1),
            source("step", "housing", "Open the housing", "OPERATIONAL ACTION\nOpen the housing with the supply isolated.\n"
                   "SAFETY NOTE\nKeep hands away from pinch points.", 5),
            source("step", "route", "Route the tube", "OPERATIONAL ACTION\nPass the tube through the spiral "
                   "channel then the elbow and exit collar.\nSAFETY NOTE\nDo not force the tube.", 6),
            source("step", "close", "Close the housing", "OPERATIONAL ACTION\nClose and secure the housing.\n"
                   "SAFETY NOTE\nDo not energize the supply.", 7)]


class ManualSupportSegmentRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {
            "MM_USAGE_ENFORCEMENT": "off", "AI_INTERNAL_SECRET": "offline-fixture-only",
            "OPENAI_API_KEY": "offline-unused", "MM_INGEST_LEDGER_AUTO_DDL": "0"}))
        for name in ("socket.create_connection", "socket.socket.connect",
                     "requests.sessions.Session.request", "psycopg2.connect"):
            cls.guard.enter_context(patch(name, denied))
        cls.m = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.guard.close()

    def read_support(self, rows):
        queries = []
        pages = [("synthetic-manual", "offline-machine", 2,
                  "Planning logistics warehouse receipt docket serial register inventory allocation "
                  "dispatch packaging capacity delivery date supervisor signoff."),
                 ("synthetic-manual", "offline-machine", 7,
                  "For cartridge insertion isolate the supply and open the housing. Route the tube through "
                  "the spiral channel first, then through the elbow, finally through the exit collar. "
                  "Close and secure the housing. Do not energize the supply."),
                 ("synthetic-manual", "offline-machine", 9,
                  "Inventory logistics delivery date and packaging capacity require a signed receipt. "
                  "Check warehouse register serial numbers before dispatch.")]
        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, params): queries.append((sql, params))
            def fetchall(self): return deepcopy(pages)
        class Connection:
            def cursor(self): return Cursor()
            def close(self): pass
        with patch.object(self.m, "_db_conn", lambda: Connection()), \
             patch.object(self.m, "_v12_filter_linkable_manual_support", lambda company, rows: rows):
            result = self.m._v13_fetch_manual_support_deterministic(
                company_id="offline-company", machine_id="offline-machine", q="How do I insert it?",
                planner={"information_task": "procedure_segment"}, structured_citations=rows)
        return result, queries

    def test_short_query_reaches_manual_route_despite_long_parent_metadata(self):
        rows = selected_segment()
        before = deepcopy(rows)
        result, queries = self.read_support(rows)
        self.assertEqual(result[0]["page_from"], 7)
        self.assertIn("spiral channel", result[0]["chunk_full"])
        self.assertEqual(rows, before)
        self.assertEqual(len(queries), 1)
        self.assertLessEqual(len(result), self.m.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS)

    def test_scoped_reader_limits_and_selected_bundle_filter_are_preserved(self):
        rows = selected_segment()
        result, queries = self.read_support(rows)
        sql, params = queries[0]
        self.assertIn("WHERE company_id=%s", sql)
        self.assertIn("(machine_id=%s OR machine_id IS NULL OR machine_id='')", sql)
        self.assertEqual(params[1:4], ("offline-company", "offline-machine", "offline-machine"))
        self.assertLessEqual(params[-1], 300)
        filtered = self.m._v12_filter_manual_support_to_selected_bundle(
            q="How do I insert it?", structured_citations=rows, manual_support_citations=result)
        self.assertIn(7, [row["page_from"] for row in filtered])

    def test_late_selected_step_survives_long_earlier_step(self):
        rows = selected_segment()
        rows[1] = source("step", "long-first", "Prepare the workspace",
                         "OPERATIONAL ACTION\n" + " ".join("preparation" + str(i) for i in range(80)), 1)
        terms = self.m._ask_structured_manual_support_terms("How do I insert it?", {}, rows)
        self.assertIn("spiral", terms)
        self.assertIn("elbow", terms)
        self.assertLessEqual(len(terms), 44)
        self.assertEqual(len(terms), len(set(terms)))

    def test_envelope_keys_and_safety_only_detail_do_not_replace_action_vocabulary(self):
        rows = selected_segment()
        rows[2]["chunk_full"] += "\nSAFETY NOTE\nSentinelwarning must be observed."
        terms = self.m._ask_structured_manual_support_terms("How do I insert it?", {}, rows)
        self.assertIn("tube", terms)
        self.assertIn("collar", terms)
        for word in ("source_type", "step_number", "short_description", "sentinelwarning"):
            self.assertNotIn(word, terms)
        self.assertIn("Sentinelwarning", rows[2]["chunk_full"])

    def test_same_records_have_stable_terms_and_do_not_grow_from_unselected_siblings(self):
        rows = selected_segment()
        first = self.m._ask_structured_manual_support_terms("How do I insert it?", {}, rows)
        for _ in range(3):
            self.assertEqual(first, self.m._ask_structured_manual_support_terms("How do I insert it?", {}, deepcopy(rows)))
        self.assertNotIn("calibration", first)
        self.assertEqual([self.m._procedure_ui_fields(r).get("step_number") for r in rows[1:]], ["1", "5", "6", "7"])

    def test_actual_safety_setup_is_not_reclassified_as_operational_support(self):
        rows = [source("step", "safe-only", "Isolate the electrical supply",
                       "OPERATIONAL ACTION\nIsolate the electrical supply and lock the breaker.\n"
                       "SAFETY NOTE\nKeep the breaker isolated.", 1)]
        page = {"citation_id": "manual:safety", "bubble_document_id": "manual", "source_type": "document",
                "chunk_full": "Isolate the electrical supply and lock the breaker. Keep the breaker isolated.",
                "ask_manual_support_kind": "operation"}
        before = deepcopy(rows)
        result = self.m._v12_filter_manual_support_to_selected_bundle(q="outline", structured_citations=rows,
                     manual_support_citations=[page])
        self.assertEqual(result, [])
        page["ask_manual_support_kind"] = "safety"
        result = self.m._v12_filter_manual_support_to_selected_bundle(q="outline", structured_citations=rows,
                     manual_support_citations=[page])
        self.assertEqual(len(result), 1)
        self.assertEqual(rows, before)

    def test_parent_only_source_remains_searchable_without_exposing_envelope(self):
        parent = source("procedure", "only-parent", "Replace the cartridge",
                        "Pass the cartridge through the insertion guide.")
        terms = self.m._ask_structured_manual_support_terms("How is this done?", {}, [parent])
        self.assertIn("cartridge", terms)
        self.assertIn("guide", terms)
        self.assertNotIn("short_description", terms)


if __name__ == "__main__":
    unittest.main()
