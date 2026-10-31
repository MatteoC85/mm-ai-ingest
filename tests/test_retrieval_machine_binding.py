"""DB binding survives lexical ranking/merge; synthetic, offline contract tests."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import lexical, dense, candidate_ranking as ranking
from machinemind.retrieval import source_management, smart_evidence
from machinemind.retrieval.chunk_evidence import ChunkReadBindingError
from test_smart_source_selection import scoring_runtime, terms


class Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, query, params): self.calls.append((query, params))
    def fetchall(self): return deepcopy(self.rows)


class Connection:
    def __init__(self, rows):
        self.cur = Cursor(rows)
        self.closed = False
    def cursor(self): return self.cur
    def close(self): self.closed = True


def source_row(company, machine, *, document='ps:fixture-source'):
    return (document, 1, 1, 1, 'Pressure can drop after the documented seal failure.', .75, company, machine)


def read(rows, company='tenant-A', machine='machine-A', *, prefix=False, canonical=False, **selectors):
    conn = Connection(rows)
    runtime = lexical.LexicalRuntime(lambda: conn, 700)
    if prefix:
        result = lexical._read_fts_search_chunks_prefix(company, machine, ['pressure'], 8,
            runtime=runtime, build_prefix_query=lambda *a, **k: 'pressure:*',
            include_binding_columns=canonical, **selectors)
    else:
        result = lexical._read_fts_search_chunks(company, machine, 'pressure', 8,
            runtime=runtime, include_binding_columns=canonical, **selectors)
    return result, conn


def score(rows):
    bias = source_management.CandidateSourceBiasRuntime(.05, terms, lambda key: 'ps',
        lambda a, b: len(a & b) / max(1, len(a)))
    runtime = replace(scoring_runtime(lambda rows, **kw: rows),
        _candidate_source_bias=lambda row, query, **kw: source_management.candidate_source_bias(
            row, query, runtime=bias, **kw),
        _v13_candidate_text=lambda row: row.get('chunk_full') or row.get('snippet') or '')
    return ranking.v13_score_candidates('pressure seal', rows, runtime=runtime, preserve_complete=True)


def packet(rows, company='tenant-A', machine='machine-A'):
    return smart_evidence.build(rows, scope=dict(company_id=company, machine_id=machine, ai_scope='machine_all'),
        allow_raw_snippet=True)


class RetrievalMachineBindingTests(unittest.TestCase):
    def test_selected_db_binding_survives_both_readers_ranking_and_both_merge_orders(self):
        for company, machine in [('tenant-A', 'machine-A'), ('tenant-B', 'unseen-machine-C')]:
            for prefix in (False, True):
                with self.subTest(company=company, machine=machine, prefix=prefix):
                    row = source_row(company, machine)
                    (fts, bound_rows), conn = read([row], company, machine, prefix=prefix)
                    self.assertTrue(conn.closed)
                    self.assertEqual(len(conn.cur.calls), 1)
                    self.assertIn('company_id AS evidence_company_id', conn.cur.calls[0][0])
                    self.assertIn('machine_id AS evidence_machine_id', conn.cur.calls[0][0])
                    self.assertEqual(bound_rows, [])
                    self.assertEqual(fts[0]['company_id'], company)
                    self.assertEqual(fts[0]['machine_id'], machine)
                    self.assertIs(fts[0]['exact_machine_scope'], True)
                    lexical_scored = score(fts)
                    dense_rows = dense.raw_rows_to_dense_candidates([
                        (*row[:5], row[4] + ' Complete retained passage.', .5, [1.0], True)])
                    for groups in ([dense_rows, lexical_scored], [lexical_scored, dense_rows]):
                        merged = ranking.v13_merge_candidates(groups)
                        self.assertEqual(len(merged), 1)
                        self.assertEqual(merged[0]['chunk_full'], dense_rows[0]['chunk_full'])
                        evidence = packet(merged, company, machine)
                        self.assertEqual(evidence['sources'][0]['machine_relation'], 'selected_machine')
                        self.assertEqual(evidence['bodies'][0]['text'], dense_rows[0]['chunk_full'])

    def test_unknown_ranking_cannot_invent_false_or_erase_explicit_binding(self):
        unknown = dict(citation_id='source:p1-1:c1', bubble_document_id='source',
            page_from=1, page_to=1, snippet='Pressure seal observation.')
        unknown_scored = score([unknown])
        self.assertNotIn('exact_machine_scope', unknown_scored[0])
        self.assertEqual(packet(unknown_scored)['sources'][0]['machine_relation'], 'not_established')
        explicit = {**unknown, 'exact_machine_scope': True, 'chunk_full': unknown['snippet']}
        for groups in ([score([explicit]), unknown_scored], [unknown_scored, score([explicit])]):
            self.assertEqual(packet(ranking.v13_merge_candidates(groups))['sources'][0]['machine_relation'], 'selected_machine')
        # An actual explicit false remains false. No OR across conflicting flags is introduced.
        denied = {**unknown, 'exact_machine_scope': False}
        self.assertIs(score([denied])[0]['exact_machine_scope'], False)
        self.assertIs(ranking.v13_merge_candidates([[explicit], [denied]])[0]['exact_machine_scope'], False)

    def test_company_general_null_and_empty_are_not_selected_machine(self):
        for stored_machine in (None, ''):
            for prefix in (False, True):
                with self.subTest(stored_machine=stored_machine, prefix=prefix):
                    (rows, _), _conn = read([source_row('tenant-A', stored_machine)], prefix=prefix)
                    self.assertEqual(rows[0]['machine_id'], stored_machine)
                    self.assertIs(rows[0]['exact_machine_scope'], False)
                    self.assertEqual(packet(score(rows))['sources'][0]['machine_relation'], 'company_general')

    def test_foreign_company_machine_or_missing_db_binding_fail_closed(self):
        for prefix in (False, True):
            for row in (source_row('foreign-tenant', 'machine-A'), source_row('tenant-A', 'foreign-machine'),
                        source_row('', 'machine-A'), source_row('tenant-A', '  '), source_row('tenant-A', 42),
                        source_row('tenant-A', 'machine-A')[:6]):
                with self.subTest(prefix=prefix, row=row):
                    with self.assertRaises((ChunkReadBindingError, ValueError)):
                        read([row], prefix=prefix)

    def test_empty_and_sentinel_request_selectors_preserve_company_general_db_rows(self):
        for selector in (None, '', '__COMPANY_GENERAL__'):
            for stored_machine in (None, ''):
                for prefix in (False, True):
                    with self.subTest(selector=selector, stored_machine=stored_machine, prefix=prefix):
                        (rows, _), _conn = read([source_row('tenant-A', stored_machine)],
                            machine=selector, prefix=prefix)
                        self.assertEqual(rows[0]['machine_id'], stored_machine)
                        self.assertIs(rows[0]['exact_machine_scope'], False)
                        self.assertEqual(packet(score(rows))['sources'][0]['machine_relation'], 'company_general')
                        (single_rows, _), _single_conn = read([source_row('tenant-A', stored_machine)],
                            machine=selector, prefix=prefix, bubble_document_id='ps:fixture-source')
                        self.assertEqual(single_rows, rows)
                        with self.assertRaises(ChunkReadBindingError):
                            read([source_row('tenant-A', 'other-machine')], machine=selector, prefix=prefix)
            # Document-list scope retains its documented precedence, even without a machine selector.
            (rows, _), _conn = read([source_row('tenant-A', 'other-machine', document='explicit-document')],
                machine=selector, doc_ids=['explicit-document'])
            self.assertEqual(rows[0]['machine_id'], 'other-machine')
            self.assertIs(rows[0]['exact_machine_scope'], False)

    def test_explicit_document_scope_preserves_actual_foreign_machine_without_promotion(self):
        row = source_row('tenant-A', 'other-machine', document='explicit-document')
        for prefix in (False, True):
            (rows, _), _conn = read([row], prefix=prefix, doc_ids=['explicit-document'])
            self.assertEqual(rows[0]['machine_id'], 'other-machine')
            self.assertIs(rows[0]['exact_machine_scope'], False)
            with self.assertRaisesRegex(smart_evidence.SmartEvidenceError, 'evidence_machine_mismatch'):
                packet(rows)
            with self.assertRaises(ChunkReadBindingError):
                read([row], prefix=prefix, doc_ids=['different-document'])
            with self.assertRaises(ChunkReadBindingError):
                read([source_row('foreign-tenant', 'other-machine', document='explicit-document')],
                    prefix=prefix, doc_ids=['explicit-document'])

    def test_canonical_view_and_sql_filters_order_projection_remain_identical(self):
        original = source_row('tenant-A', 'machine-A')
        for prefix in (False, True):
            (legacy, legacy_bound), legacy_conn = read([original], prefix=prefix)
            (canonical, canonical_bound), canonical_conn = read([original], prefix=prefix, canonical=True)
            self.assertEqual(legacy_conn.cur.calls, canonical_conn.cur.calls)
            self.assertEqual(legacy_bound, [])
            self.assertEqual(canonical_bound, [original])
            self.assertEqual({k:v for k,v in legacy[0].items() if k not in {'company_id','machine_id','exact_machine_scope'}}, canonical[0])
            self.assertEqual(canonical[0]['snippet'], original[4])
            self.assertNotIn('machine_id', canonical[0])

    def test_empty_single_selector_and_document_list_precedence_match_sql(self):
        for prefix in (False, True):
            for requested_machine, stored_machine in [('machine-A', 'machine-A'), (None, None)]:
                rows = [source_row('tenant-A', stored_machine)]
                actual, actual_conn = read(rows, machine=requested_machine, prefix=prefix, bubble_document_id='')
                expected, expected_conn = read(rows, machine=requested_machine, prefix=prefix)
                self.assertEqual(actual, expected)
                self.assertEqual(actual_conn.cur.calls, expected_conn.cur.calls)
            with self.assertRaises(ChunkReadBindingError):
                read([source_row('tenant-A', 'foreign-machine')], machine=None,
                    prefix=prefix, bubble_document_id='')
            with self.assertRaises(ChunkReadBindingError):
                read([source_row('foreign-tenant', None)], machine=None,
                    prefix=prefix, bubble_document_id='')
            for ignored_single in (None, '', 'ignored-document'):
                selectors = dict(machine=None, prefix=prefix, doc_ids=['explicit-document'],
                    bubble_document_id=ignored_single)
                (rows, _), _conn = read([source_row('tenant-A', 'other-machine', document='explicit-document')], **selectors)
                self.assertEqual(rows[0]['machine_id'], 'other-machine')
                self.assertIs(rows[0]['exact_machine_scope'], False)
                with self.assertRaises(ChunkReadBindingError):
                    read([source_row('tenant-A', 'other-machine', document='not-allowed')], **selectors)
                with self.assertRaises(ChunkReadBindingError):
                    read([source_row('foreign-tenant', 'other-machine', document='explicit-document')], **selectors)


if __name__ == '__main__': unittest.main()
