"""Offline connection-window tests, including the actual typed ASK facet batch."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import request_completion
from machinemind.infrastructure import ask_read_connection as reuse, database
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_prepared_scope_admission as prepared


class FixtureConnection:
    def __init__(self, factory):
        self.factory = factory
        self.closed = False
        self.session_settings = None
        self.state = 0

    def cursor(self):
        assert not self.closed
        return FixtureCursor(self)

    def set_session(self, **settings):
        self.session_settings = settings

    def get_transaction_status(self):
        return self.state

    def close(self):
        assert not self.closed
        self.closed = True


class FixtureCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, values):
        factory = self.connection.factory
        factory.queries.append((query, tuple(values)))
        if factory.failure:
            raise factory.failure
        if "document_chunks" not in query:
            return
        if "embedding <=>" in query:
            self.rows = [("fixture-manual", 0, 1, 1, factory.body, factory.body, .55,
                          "[0.5,0.25]", True, factory.company, factory.machine)]
        elif "ts_rank_cd" in query:
            self.rows = [("fixture-manual", 0, 1, 1, factory.body, .3,
                          factory.company, factory.machine)]

    def fetchall(self):
        return self.rows


class FixtureFactory:
    def __init__(self, company="", machine=""):
        self.body = "Maximum mass is 40 kg. Use eye protection and isolate energy before insertion."
        self.company, self.machine = company, machine
        self.kwargs, self.connections, self.queries = [], [], []
        self.failure = None

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        connection = FixtureConnection(self)
        self.connections.append(connection)
        return connection


class AskReadConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared.AskPreparedScopeAdmissionTests.setUpClass()
        cls.m = prepared.AskPreparedScopeAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        prepared.AskPreparedScopeAdmissionTests.tearDownClass()

    def test_actual_ask_batch_sql_packet_and_receipts_match_with_fewer_connections(self):
        case = prepared.AskPreparedScopeAdmissionTests()
        case.m = self.m

        def measure(flag):
            req, _, owner, _, _, session, _, _ = case.harness()
            factory = FixtureFactory(req.company_id, req.machine_id)
            original_authorize = owner.readers._authorize
            authorizations = []

            def check(request):
                authorizations.append(request)
                return original_authorize(request)

            owner.readers._authorize = check
            owner.runtime = replace(owner.runtime, initial=replace(owner.runtime.initial,
                _openai_embed_texts=lambda texts, **kwargs: [[.5, .25] for _ in texts]))
            original_read = owner._read
            # Empty optional pages/relations in both arms. All dense and lexical
            # retrieval, attribution, source selection and ordering are real.
            owner._read = lambda name, *args, **kwargs: (original_read(name, *args, **kwargs)
                if name in {"read_fts_chunk_evidence", "read_dense_chunk_evidence"} else ())
            completion = request_completion.RequestCompletion(
                runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id=req.company_id), started=time.monotonic())
            try:
                plan = self.m._v13_fallback_plan(req.query)
                plan.update(_ask_facet_batch=True, dense_queries=["mass limit", "eye protection", "isolating energy"],
                            lexical_queries=["mass limit", "eye protection"], exact_terms=[])
                with patch.dict(os.environ, {reuse.FLAG: flag}), patch.object(
                        self.m.psycopg2, "connect", factory), patch.multiple(
                        self.m, DB_HOST="offline", DB_NAME="offline", DB_USER="offline", DB_PASSWORD="offline"):
                    data, selections = owner._initial(q=req.query, company_id=req.company_id,
                        machine_id=req.machine_id, doc_ids=None, bubble_document_id=None,
                        ai_scope=req.ai_scope, response_language=req.response_language, mode="neutral", plan=plan)
                sources = self.m._v13_sources_block(data["citations"], max_context_chars=100000)
                self.assertTrue(sources)
                self.assertTrue(all(conn.closed for conn in factory.connections))
                return factory, data, sources, tuple((s.name, len(s.records)) for s in selections), len(authorizations)
            finally:
                completion.close()
                case.finish(owner, session)

        before, after = measure(""), measure(reuse.VERSION)
        baseline, reused = before[0], after[0]
        self.assertEqual(baseline.queries, reused.queries)
        self.assertEqual(before[1:], after[1:])
        self.assertEqual(len(baseline.queries), 8)
        self.assertEqual(len(baseline.kwargs), len(baseline.queries))
        self.assertEqual(len(reused.kwargs), 1)
        self.assertEqual(reused.connections[0].session_settings, {"readonly": True, "autocommit": True})
        self.assertIsNone(baseline.connections[0].session_settings)

    def test_scope_mismatch_tighter_timeout_and_unsafe_session_are_not_reused(self):
        factory = FixtureFactory()
        budget = SimpleNamespace(mode="assistant_core_ask", time_left=20)
        budget.ensure_time = lambda minimum: self.assertGreaterEqual(budget.time_left, minimum)
        budget.remaining = lambda: budget.time_left
        owner, stranger = object(), object()

        def connect():
            return database.connect_database(db_host="offline", db_name="offline", db_user="offline",
                db_password="offline", runtime_globals={"_v13_current_budget": lambda: budget},
                connect_fn=factory, monotonic_fn=lambda: 0)

        with patch.dict(os.environ, {reuse.FLAG: reuse.VERSION}), reuse.batch(owner):
            with reuse.reader(owner, "read_dense_chunk_evidence"):
                connect().close()
            retained = factory.connections[-1]
            with reuse.reader(stranger, "read_fts_chunk_evidence"):
                connect().close()
            self.assertFalse(retained.closed)
            self.assertTrue(factory.connections[-1].closed)
            budget.time_left = 6
            with reuse.reader(owner, "read_fts_chunk_evidence"):
                connect().close()
            self.assertTrue(retained.closed)
            self.assertIn("statement_timeout=5500", factory.kwargs[-1]["options"])
            non_idle = factory.connections[-1]
            non_idle.state = 3
            with reuse.reader(owner, "read_fts_chunk_evidence"):
                connect().close()
            self.assertTrue(non_idle.closed)
            with reuse.reader(owner, "read_structured_dense_chunk_evidence"):
                self.assertTrue(factory.connections[-1].closed)
                connect().close()
            self.assertIsNone(factory.connections[-1].session_settings)
        self.assertTrue(all(conn.closed for conn in factory.connections))

    def test_reader_failure_discards_and_late_leases_are_unusable(self):
        factory = FixtureFactory()
        owner = object()
        budget = SimpleNamespace(mode="assistant_core_ask")
        with patch.dict(os.environ, {reuse.FLAG: reuse.VERSION}), reuse.batch(owner):
            with self.assertRaisesRegex(OSError, "offline failure"):
                with reuse.reader(owner, "read_fts_chunk_evidence"):
                    lease = reuse.connect(factory, {}, budget)
                    factory.failure = OSError("offline failure")
                    try:
                        with lease.cursor() as cursor:
                            cursor.execute("SELECT fixture", ())
                    finally:
                        lease.close()
            reuse.discard(owner)  # same sticky cleanup boundary as acquire_read
            self.assertTrue(factory.connections[0].closed)
            self.assertEqual(len(factory.queries), 1)
            with self.assertRaisesRegex(RuntimeError, "Expired ASK SQL read lease"):
                lease.cursor()
        self.assertEqual(len(factory.connections), 1)

    def test_actual_reader_rechecks_authority_before_reusing_native_connection(self):
        case = prepared.AskPreparedScopeAdmissionTests()
        case.m = self.m
        req, _, owner, _, _, session, _, _ = case.harness()
        factory = FixtureFactory(req.company_id, req.machine_id)
        completion = request_completion.RequestCompletion(
            runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=req.company_id), started=time.monotonic())
        try:
            with patch.dict(os.environ, {reuse.FLAG: reuse.VERSION}), patch.object(
                    self.m.psycopg2, "connect", factory), patch.multiple(
                    self.m, DB_HOST="offline", DB_NAME="offline", DB_USER="offline", DB_PASSWORD="offline"
                ), reuse.batch(owner.readers):
                owner.readers.read("read_fts_chunk_evidence", q="initial check", top_k=3)
                self.assertFalse(factory.connections[0].closed)

                def denied(request):
                    raise EvidenceContractError("fixture source authority revoked")

                owner.readers._authorize = denied
                with self.assertRaisesRegex(EvidenceContractError, "fixture source authority revoked"):
                    owner.readers.read("read_fts_chunk_evidence", q="second check", top_k=3)
                self.assertTrue(factory.connections[0].closed)
        finally:
            completion.close()
            case.finish(owner, session)
        self.assertEqual(len(factory.queries), 1)
        self.assertEqual(len(factory.connections), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
