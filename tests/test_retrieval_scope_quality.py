"""Real local PostgreSQL/pgvector retrieval checks with deterministic vectors.

Requires the disposable p6_validation database on 127.0.0.1:55436. This test
never starts a database or contacts an embedding/provider API. Every fixture
connection rolls back, including fixture DDL, before closing.
"""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from psycopg2.extensions import parse_dsn
from psycopg2.extras import execute_values

from machinemind.evidence.adapter_types import AdapterLimits
from machinemind.evidence.assembly import AssemblyLimits
from machinemind.evidence.contracts import EvidenceContractError
from machinemind.evidence.legacy_compatibility import LegacyLimits
from machinemind.evidence.manifest import ManifestLimits
from machinemind.retrieval.chunk_evidence import ChunkEvidenceLimits, ChunkReadScope

from machinemind.retrieval.dense import (
    DenseRuntime, fetch_dense_chunk_candidates, raw_rows_to_dense_candidates,
    read_dense_chunk_evidence,
)
from machinemind.retrieval.lexical import (
    LexicalRuntime, fts_search_chunks, read_fts_chunk_evidence,
    read_prefix_chunk_evidence, build_prefix_tsquery_from_texts,
)

COMPANY_A = "p6-quality-company-a"
COMPANY_B = "p6-quality-company-b"
MACHINE_A = "p6-quality-machine-a"
MACHINE_B = "p6-quality-machine-b"
DOC_A = "p6-quality-doc-a"
DOC_OTHER_MACHINE = "p6-quality-doc-other-machine"
DOC_PRIVATE = "p6-quality-doc-private"
DOC_GENERAL_NULL = "p6-quality-doc-general-null"
DOC_GENERAL_EMPTY = "p6-quality-doc-general-empty"
DOC_NO_VECTOR = "p6-quality-doc-no-vector"
SNIPPET_CHARS = 48
ROWS = (
    (COMPANY_A, MACHINE_A, DOC_A, 3, 7, 8, "  scopefixture actuator pressure exact quotation for machine A. Do not infer missing steps.  ", (1, 0, 0)),
    (COMPANY_A, MACHINE_B, DOC_OTHER_MACHINE, 4, 2, 2, "scopefixture actuator pressure belonging to machine B only.", (1, 0, 0)),
    (COMPANY_B, MACHINE_A, DOC_PRIVATE, 5, 4, 4, "scopefixture actuator pressure private to company B.", (1, 0, 0)),
    (COMPANY_A, None, DOC_GENERAL_NULL, 6, 1, 1, "scopefixture actuator pressure general instruction, NULL machine association.", (0.8, 0.6, 0)),
    (COMPANY_A, "", DOC_GENERAL_EMPTY, 7, 9, 9, "scopefixture actuator pressure general instruction, empty machine association.", (0, 1, 0)),
    (COMPANY_A, MACHINE_A, DOC_NO_VECTOR, 8, 12, 12, "scopefixture actuator pressure indexed lexical text without an embedding.", None),
)


class RollbackConnection:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def cursor(self):
        return self.connection.cursor()

    def close(self):
        try:
            self.connection.rollback()
        finally:
            self.connection.close()
            self.closed = True


class RetrievalScopeQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dsn = os.environ.get("P6_NATIVE_TEST_DSN", "")
        if not dsn:
            raise unittest.SkipTest("P6_NATIVE_TEST_DSN required for real local PostgreSQL tests")
        parsed = parse_dsn(dsn)
        if parsed.get("host") != "127.0.0.1" or str(parsed.get("port")) != "55436" or parsed.get("dbname") != "p6_validation":
            raise RuntimeError("Retrieval tests require the disposable loopback p6_validation database on port 55436")
        if parsed.get("hostaddr") not in (None, "", "127.0.0.1") or parsed.get("service"):
            raise RuntimeError("Remote or indirect database addressing is forbidden in retrieval tests")
        cls.dsn = dsn

    def setUp(self):
        self.connections = []
        self.addCleanup(self.close_connections)
        self.dimension = 3
        # Discover the actual test table's vector typmod without assumptions about
        # the model dimension. Fixture DDL and inserts are rolled back here too.
        probe = self.fixture_connection()
        probe.close()
        self.dense_runtime = DenseRuntime(self.fixture_connection, SNIPPET_CHARS)
        self.lexical_runtime = LexicalRuntime(self.fixture_connection, SNIPPET_CHARS)
        self.limits = ChunkEvidenceLimits(
            adapter=AdapterLimits(max_text_chars=10000, max_fields=40, max_aux_chars=10000),
            assembly=AssemblyLimits(ManifestLimits(20, 1000000), 20, 2000000),
            legacy=LegacyLimits(32, 100000, 4000000),
        )

    def vector(self, values=(1, 0, 0)):
        if self.dimension < len(values):
            raise RuntimeError("The offline fixture requires a vector dimension of at least three")
        return "[" + ",".join(str(v) for v in (*values, *([0] * (self.dimension - len(values))))) + "]"

    def fixture_connection(self):
        raw = psycopg2.connect(self.dsn, connect_timeout=3)
        connection = RollbackConnection(raw)
        self.connections.append(connection)
        try:
            with raw.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '10s'")
                cur.execute("SET LOCAL lock_timeout = '2s'")
                cur.execute("SELECT current_database(), host(inet_server_addr()), inet_server_port()")
                if cur.fetchone() != ("p6_validation", "127.0.0.1", 55436):
                    raise RuntimeError("Database server is outside the disposable offline fixture")
                cur.execute("SELECT to_regclass('public.document_chunks')")
                if cur.fetchone()[0] is None:
                    cur.execute("""CREATE TABLE public.document_chunks (
                        company_id text NOT NULL, machine_id text,
                        bubble_document_id text NOT NULL, chunk_index integer NOT NULL,
                        page_from integer NOT NULL, page_to integer NOT NULL,
                        chunk_text text NOT NULL, embedding vector(3)
                    )""")
                cur.execute("""SELECT a.atttypmod, t.typname FROM pg_attribute a
                    JOIN pg_type t ON t.oid=a.atttypid
                    WHERE a.attrelid='public.document_chunks'::regclass
                    AND a.attname='embedding' AND NOT a.attisdropped""")
                dimension, column_type = cur.fetchone()
                if column_type != "vector":
                    raise RuntimeError("Fixture embedding column must use real pgvector")
                self.dimension = int(dimension) if int(dimension) > 0 else 3
                cur.execute("SELECT count(*) FROM public.document_chunks WHERE company_id=ANY(%s)", ([COMPANY_A, COMPANY_B],))
                if cur.fetchone()[0]:
                    raise RuntimeError("Offline fixture identifiers already exist; refusing to overwrite rows")
                rows = [(*row[:7], self.vector(row[7]) if row[7] else None) for row in ROWS]
                execute_values(cur, """INSERT INTO public.document_chunks
                    (company_id,machine_id,bubble_document_id,chunk_index,page_from,page_to,chunk_text,embedding)
                    VALUES %s""", rows, template="(%s,%s,%s,%s,%s,%s,%s,%s::vector)")
            return connection
        except BaseException:
            connection.close()
            raise

    def dense(self, *, company=COMPANY_A, machine=MACHINE_A, documents=None, single=None):
        count, rows = fetch_dense_chunk_candidates(
            company_id=company, machine_id=machine, q_vec_lit=self.vector(),
            candidate_k=20, doc_ids=documents, bubble_document_id=single,
            debug=True, runtime=self.dense_runtime,
        )
        self.assertEqual(count, len(rows))
        self.assertTrue(self.connections[-1].closed)
        return raw_rows_to_dense_candidates(rows, query_used="scopefixture")

    def scope(self, *, company=COMPANY_A, machine=MACHINE_A, documents=(), single=None, kind="machine_all"):
        return ChunkReadScope(company, machine, kind, tuple(documents), single)

    def bound_dense(self, scope):
        return read_dense_chunk_evidence(scope=scope, q_vec_lit=self.vector(), candidate_k=20,
            limits=self.limits, runtime=self.dense_runtime, query_used="scopefixture", debug=True)

    def assert_bound_text_and_scope(self, read):
        expected = {row[2]: row for row in ROWS}
        restored = read.as_legacy_inputs(scope=read.scope,
            current_allowed_sources=read.read_sources, adapter_limits=self.limits.adapter)
        self.assertEqual(len(restored), len(read.observations))
        for observation, legacy in zip(read.observations, restored):
            row = expected[observation.storage_document_id]
            self.assertEqual(observation.stored_company_id, row[0])
            self.assertEqual(observation.stored_machine_id, row[1])
            self.assertEqual(observation.source.scope.company_id, row[0])
            self.assertEqual(observation.source.scope.machine_id, row[1] or None)
            self.assertEqual(observation.source.source_id, row[2])
            self.assertEqual(observation.snippet_projection, row[6][:SNIPPET_CHARS])
            self.assertEqual(legacy.record["snippet"], row[6][:SNIPPET_CHARS].strip())
            self.assertEqual(legacy.record["citation_id"], f"{row[2]}:p{row[4]}-{row[5]}:c{row[3]}")
            if read.kind == "dense":
                self.assertEqual(observation.chunk_projection, row[6][:2000])
                self.assertEqual(legacy.record["chunk_full"], row[6][:2000].strip())
                self.assertEqual(legacy.record["similarity"], legacy.record["semantic_similarity"])
            else:
                self.assertIsNone(observation.chunk_projection)
                self.assertEqual(legacy.record["similarity"], 0.0)
                self.assertNotIn("semantic_similarity", legacy.record)

    def test_dense_machine_scope_excludes_other_tenant_and_machine(self):
        candidates = self.dense()
        self.assertEqual([x["bubble_document_id"] for x in candidates], [DOC_A, DOC_GENERAL_NULL, DOC_GENERAL_EMPTY])
        self.assertEqual([x["exact_machine_scope"] for x in candidates], [True, False, False])
        self.assertAlmostEqual(candidates[0]["semantic_similarity"], 1.0)
        self.assertAlmostEqual(candidates[-1]["semantic_similarity"], 0.0)

    def test_bound_dense_preserves_source_binding_citations_and_quotes(self):
        read = self.bound_dense(self.scope())
        self.assertEqual(read.chunks_matching_filter, 3)
        self.assert_bound_text_and_scope(read)
        self.assertTrue(all(o.stored_company_id == COMPANY_A for o in read.observations))

    def test_explicit_authorized_document_list_keeps_company_fence(self):
        # The existing contract permits explicitly selected same-company docs
        # of another machine; it does not authorize another company's documents.
        candidates = self.dense(documents=[DOC_OTHER_MACHINE, DOC_PRIVATE], single=DOC_A)
        self.assertEqual([x["bubble_document_id"] for x in candidates], [DOC_OTHER_MACHINE])
        read = self.bound_dense(self.scope(documents=[DOC_OTHER_MACHINE, DOC_PRIVATE], single=DOC_A, kind="document_ids"))
        self.assertEqual({s.scope.machine_id for s in read.read_sources}, {MACHINE_B})
        self.assert_bound_text_and_scope(read)

    def test_single_document_keeps_machine_fence_and_absent_is_empty(self):
        self.assertEqual(self.dense(single=DOC_OTHER_MACHINE), [])
        self.assertEqual(self.dense(documents=["p6-quality-absent"]), [])
        self.assertEqual(self.dense(company="p6-quality-no-company"), [])
        empty = self.bound_dense(self.scope(documents=["p6-quality-absent"], kind="document_ids"))
        self.assertEqual(empty.observations, ())
        self.assertEqual(empty.read_sources, frozenset())

    def test_company_general_keeps_only_null_or_empty_machine_associations(self):
        read = self.bound_dense(self.scope(machine=None, kind="company_general"))
        self.assertEqual({o.storage_document_id for o in read.observations}, {DOC_GENERAL_NULL, DOC_GENERAL_EMPTY})
        self.assert_bound_text_and_scope(read)

    def test_lexical_real_sql_keeps_scope_and_does_not_fabricate_cosine(self):
        read = read_fts_chunk_evidence(scope=self.scope(), q="scopefixture actuator", top_k=20,
            runtime=self.lexical_runtime, limits=self.limits)
        self.assertEqual({o.storage_document_id for o in read.observations}, {DOC_A, DOC_GENERAL_NULL, DOC_GENERAL_EMPTY, DOC_NO_VECTOR})
        self.assert_bound_text_and_scope(read)

    def test_lexical_document_selector_precedence_and_empty_query(self):
        candidates = fts_search_chunks(COMPANY_A, MACHINE_A, "scopefixture", 20,
            doc_ids=[DOC_OTHER_MACHINE, DOC_PRIVATE], bubble_document_id=DOC_A, runtime=self.lexical_runtime)
        self.assertEqual([x["bubble_document_id"] for x in candidates], [DOC_OTHER_MACHINE])
        self.assertEqual(fts_search_chunks(COMPANY_A, MACHINE_A, "scopefixture", 20,
            bubble_document_id=DOC_OTHER_MACHINE, runtime=self.lexical_runtime), [])
        before = len(self.connections)
        self.assertEqual(fts_search_chunks(COMPANY_A, MACHINE_A, "", 20, runtime=self.lexical_runtime), [])
        self.assertEqual(len(self.connections), before)
        self.assertEqual(fts_search_chunks(COMPANY_A, MACHINE_A, "definitelyabsenttoken", 20, runtime=self.lexical_runtime), [])

    def test_prefix_real_sql_preserves_same_source_and_quote_contract(self):
        read = read_prefix_chunk_evidence(scope=self.scope(), texts=["scopefixt"], top_k=20,
            runtime=self.lexical_runtime, limits=self.limits,
            build_prefix_query=lambda texts, limit: build_prefix_tsquery_from_texts(texts, limit, normalize_unicode=lambda s: s))
        self.assertEqual({o.storage_document_id for o in read.observations}, {DOC_A, DOC_GENERAL_NULL, DOC_GENERAL_EMPTY, DOC_NO_VECTOR})
        self.assert_bound_text_and_scope(read)

    def test_read_snapshot_does_not_grant_revoked_or_other_tenant_access(self):
        read = self.bound_dense(self.scope())
        with self.assertRaises(EvidenceContractError):
            read.as_legacy_inputs(scope=self.scope(), current_allowed_sources=frozenset(), adapter_limits=self.limits.adapter)
        with self.assertRaises(EvidenceContractError):
            read.as_legacy_inputs(scope=self.scope(company=COMPANY_B), current_allowed_sources=read.read_sources, adapter_limits=self.limits.adapter)

    def close_connections(self):
        for connection in self.connections:
            if not connection.closed:
                connection.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
