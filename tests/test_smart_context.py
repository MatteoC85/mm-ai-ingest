"""Synthetic SQL-column fixtures, real typed readers, no external I/O or model."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import smart_context as context, smart_evidence
from machinemind.retrieval.document_readers import (read_parent_procedure_page_evidence,
    DbFetchParentProcedurePagesForStepsRuntime)
from machinemind.retrieval.structured import (read_related_step_page_evidence,
    DbFetchRelatedStepPagesRuntime)

SCOPE = {'company_id': 'test-company', 'machine_id': 'test-machine', 'ai_scope': 'machine_all'}
REL = 'procedure_step'


def raw(number):
    return {'citation_id': f'step:s{number}:p1-1:c1', 'bubble_document_id': f'step:s{number}',
        'source_type': 'step', 'source_id': f's{number}', 'company_id': SCOPE['company_id'],
        'machine_id': SCOPE['machine_id'], 'page_from': 1, 'page_to': 1,
        'chunk_full': f'Complete selected step {number}; retain all source precautions.'}


def sql_row(number, *, parent=False, family='procedure:p', text=None, machine='test-machine', page=1,
            ordinal=None, company='test-company'):
    child, ordinal = f'step:s{number}', number if ordinal is None else ordinal
    key = family if parent else child
    body = text if text is not None else ('Procedure header: trained operators; isolate before access.' if parent
                else f'Step {number}: documented complete check. Preserve isolation and source safety tail.')
    head = (child, family, ordinal, machine, page, body) if parent else (child, ordinal, machine, page, body)
    return head + ('test-company', 'test-machine', family, child, REL, 'procedure', 'step', ordinal,
                   'canonical_ingest', {}, company, key)


class Cursor:
    def __init__(self, rows, calls): self.rows, self.calls = rows, calls
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, query, params): self.calls.append((query, params))
    def fetchall(self): return self.rows


class Connection:
    def __init__(self, rows, calls): self.rows, self.calls = rows, calls
    def cursor(self): return Cursor(self.rows, self.calls)
    def close(self): pass


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.selected = [raw(3), raw(6)]
        self.parents = [sql_row(3, parent=True), sql_row(6, parent=True)]
        self.steps = [sql_row(i) for i in range(1, 9)]
        self.calls = []
        self.ensure = Mock()

    def acquire(self):
        p = DbFetchParentProcedurePagesForStepsRuntime(REL,
            lambda: Connection(self.parents, self.calls), lambda rows, **kw: list(dict.fromkeys(rows)),
            lambda v, default=0: int(v or default))
        s = DbFetchRelatedStepPagesRuntime(REL, lambda: Connection(self.steps, self.calls))
        return context.acquire(self.selected, scope=SCOPE,
            parent_reader=lambda **kw: read_parent_procedure_page_evidence(**kw, runtime=p),
            step_reader=lambda **kw: read_related_step_page_evidence(**kw, runtime=s), ensure_time=self.ensure)

    def test_exact_parent_and_direct_predecessors_full_text_two_bounded_reads(self):
        before = deepcopy(self.selected)
        additions, meta = self.acquire()
        self.assertEqual([r['bubble_document_id'] for r in additions], ['procedure:p', 'step:s2', 'step:s5'])
        self.assertEqual(meta['reason'], 'context_admitted')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.ensure.call_count, 2)
        self.assertEqual(self.selected, before)
        for query, params in self.calls:
            self.assertIn('LIMIT %s', query)
            self.assertEqual(params[-1], context.MAX_READ_ROWS + 1)
            self.assertEqual(params[0], context.PROJECTION_CHARS)
        self.assertEqual(additions[1]['chunk_full'], self.steps[1][4])
        packet = smart_evidence.build(self.selected, scope=SCOPE)
        updated = smart_evidence.update_enrichment(packet, additions, scope=SCOPE,
            allowed_ids=[r['citation_id'] for r in self.selected + additions])
        self.assertEqual(len(updated['sources']), 5)
        self.assertEqual(smart_evidence.review_packet(updated)['validator_records'][-1]['text'], self.steps[4][4])

    def test_no_step_anchor_no_read(self):
        self.selected = [{**raw(1), 'bubble_document_id': 'document'}]
        additions, meta = self.acquire()
        self.assertEqual((additions, len(self.calls)), ([], 0))
        self.assertEqual(meta['reason'], 'no_scoped_step_anchor')

    def test_unestablished_machine_is_not_invented(self):
        self.selected = [{k: v for k, v in raw(3).items() if k != 'machine_id'}]
        self.assertEqual(self.acquire()[0], [])
        self.assertEqual(self.calls, [])

    def test_parent_ambiguity_fail_before_second_read(self):
        self.parents.append(sql_row(3, parent=True, family='procedure:other'))
        additions, meta = self.acquire()
        self.assertEqual(additions, [])
        self.assertEqual(meta['reason'], 'context_parent_ambiguous')
        self.assertEqual(len(self.calls), 1)

    def test_only_first_canonical_family_is_read_even_with_multiple_families(self):
        self.parents[1] = sql_row(6, parent=True, family='procedure:second')
        additions, _ = self.acquire()
        self.assertEqual([r['bubble_document_id'] for r in additions], ['procedure:p', 'step:s2'])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1][1][3], 'procedure:p')

    def test_missing_parent_never_fabricates_header(self):
        row = list(self.parents[0])
        row[3:6] = [None, None, '']
        row[-2:] = [None, None]
        self.parents[0] = tuple(row)
        self.assertEqual(self.acquire()[1]['reason'], 'context_parent_page_missing')
        self.assertEqual(len(self.calls), 1)

    def test_foreign_joined_page_rejected_even_outside_predecessors(self):
        for field in ('machine', 'company'):
            with self.subTest(field=field):
                self.steps[-1] = sql_row(8, **{field: 'foreign'})
                additions, meta = self.acquire()
                self.assertEqual(additions, [])
                self.assertEqual(meta['reason'], 'context_read_unavailable')

    def test_oversized_or_truncated_unselected_page_rejects_entire_context(self):
        self.steps[-1] = sql_row(8, text='X' * context.PROJECTION_CHARS)
        self.assertEqual(self.acquire()[1]['reason'], 'context_page_incomplete')

    def test_duplicate_ordinal_or_child_order_is_ambiguous(self):
        for extra, reason in ((sql_row(99, ordinal=2), 'context_predecessor_ambiguous'),
                              (sql_row(2, ordinal=9), 'context_child_ordinal_ambiguous')):
            with self.subTest(reason=reason):
                self.steps = [sql_row(i) for i in range(1, 9)] + [extra]
                self.assertEqual(self.acquire()[1]['reason'], reason)

    def test_missing_predecessor_or_changed_anchor_fail_closed(self):
        self.steps = [sql_row(i) for i in range(1, 9) if i != 2]
        self.assertEqual(self.acquire()[1]['reason'], 'context_predecessor_missing')
        self.steps = [sql_row(i) for i in range(1, 9) if i != 3]
        self.assertEqual(self.acquire()[1]['reason'], 'context_anchor_relation_changed')

    def test_family_row_overflow_fails_no_partial_context(self):
        self.steps = [sql_row(i) for i in range(1, context.MAX_READ_ROWS + 2)]
        self.assertEqual(self.acquire()[0], [])
        self.assertEqual(len(self.calls), 2)

    def test_company_general_parent_is_not_promoted_to_machine(self):
        self.parents = [sql_row(i, parent=True, machine=None) for i in (3, 6)]
        additions, _ = self.acquire()
        self.assertEqual(additions[0]['machine_id'], '')
        self.assertFalse(additions[0]['exact_machine_scope'])
        packet = smart_evidence.build(additions, scope=SCOPE)
        self.assertEqual(packet['sources'][0]['machine_relation'], 'company_general')

    def test_time_exhausted_stops_before_any_sql(self):
        self.ensure.side_effect = RuntimeError('deadline')
        self.assertEqual(self.acquire()[0], [])
        self.assertEqual(self.calls, [])

    def test_first_read_consumes_shared_time_and_prevents_second_read(self):
        self.ensure.side_effect = [None, RuntimeError('shared deadline')]
        additions, meta = self.acquire()
        self.assertEqual(additions, [])
        self.assertEqual(meta['read_calls'], 1)
        self.assertEqual(len(self.calls), 1)

    def test_parent_multi_page_or_nonpositive_ordinal_rejected(self):
        self.parents.append(sql_row(3, parent=True, page=2))
        self.assertEqual(self.acquire()[1]['reason'], 'context_parent_page_ambiguous')
        self.parents = [sql_row(3, parent=True, ordinal=0)]
        self.assertEqual(self.acquire()[1]['reason'], 'context_relation_ordinal_invalid')

    def test_more_than_three_context_records_report_omission_not_full_family(self):
        self.selected = [raw(3), raw(6), raw(8)]
        self.parents.append(sql_row(8, parent=True))
        additions, meta = self.acquire()
        self.assertEqual(len(additions), 3)
        self.assertEqual(meta['omitted_ids'], ['step:s7:p1-1:smart-context'])


if __name__ == '__main__':
    unittest.main()
