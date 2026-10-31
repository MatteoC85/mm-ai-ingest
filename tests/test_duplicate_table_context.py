"""Exact duplicate cells do not discard distinct admitted source context.

Model responses and SQL rows are fixtures; no claim of model semantic quality.
"""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review, source_table_review as tables, enumeration_review
from machinemind.ask import request_completion
from machinemind.ask.response_evidence import ResponseEvidenceError
from machinemind.infrastructure import request_budget, openai_transport
from context_review_wire_fixture import encode_context_reply, original_context_sources, encode_table_coverage_reply
import test_source_table_owned as owned

CELLS = ('- [PDF_TABLE_HEADER rows=1 columns=2] c1 r1="Oil temperature" | c2 r1="Interval"\n'
         '- [PDF_TABLE_RECORD rows=2 columns=2] c1 "Oil temperature" r2="Low" | c2 "Interval" r2="Periodic"')
SCOPE = 'This table concerns the documented sliding assembly.'
QUALIFIER = 'For this assembly, reduce the stated intervals under heavy loading.'
OTHER = 'A separate rotary assembly uses these values only while stationary.'


def candidate(bodies=None):
    bodies = bodies or {'doc:p1': SCOPE + '\n' + CELLS,
                        'doc:p2': SCOPE + '\n' + QUALIFIER + '\n' + CELLS}
    block = '\n\n'.join('[' + cid + '] source_type=document; p=' + str(i + 1) + '\n' + body
                        for i, (cid, body) in enumerate(bodies.items()))
    annotated, ids = enumeration_review.annotate_table_records(block)
    catalog = tables.compile_catalog(annotated, {cid: {'body': body, 'page_from': i + 1, 'page_to': i + 1}
                                               for i, (cid, body) in enumerate(bodies.items())}, 'en')
    assert len(catalog.entries) == len(bodies)
    selected = catalog.entries[0]
    raw = dict(outcome='rewrite', answer='Documented information:\n[[FACTUAL_INVENTORY]]',
               expected_list_items=[selected.token], citation_ids=[selected.citation_id],
               covered_list_items=[], missing_list_items=[], missing_facets=[], missing_answer_types=[],
               table_record_coverage=[rid + ('=0' if rid in selected.record_ids else '!same cells, distinct source') for rid in ids])
    expanded = tables.resolve_reply(raw, catalog)
    expanded = enumeration_review.resolve_inventory_answer(expanded, canonical_inventory=expanded['expected_list_items'], allowed=True)
    frame = review.prepare_review(raw, expanded, catalog=catalog, admitted_citation_ids=list(bodies), sources_block=annotated)
    return expanded, frame, annotated, catalog


def verdict(frame, *, omit_additional=False):
    audits = []
    for target, index, cid, _ in review._context_targets(frame):
        if omit_additional and ':context:' in target:
            continue
        body = dict(frame.source_bodies)[cid]
        quote = OTHER if OTHER in body else QUALIFIER if QUALIFIER in body else SCOPE
        audits.append(dict(target=target, context=quote, documentary_limits_checked=True,
                           source_quotes=[dict(citation_id=cid, quote=quote)]))
    return dict(outcome='rewrite', edits=[], citation_ids=list(frame.admitted_citation_ids), table_contexts=audits,
                checks={key: True for key in review.CHECKS}, reason='fixture')


class DuplicateTableContextTests(unittest.TestCase):
    def test_selected_owner_alone_cannot_approve_distinct_context(self):
        first, frame, sources, _ = candidate()
        old = verdict(frame, omit_additional=True)
        result = review.apply_review(first, old, frame)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertEqual(result['independent_context_review']['gate'], 'table_count')
        self.assertEqual(len(review._context_targets(frame)), 2)
        good = verdict(frame)
        messages = review.build_messages(frame, question='Which intervals are documented?', response_language='en', sources_block=sources)
        wire = encode_context_reply(good, messages, original_sources=sources, json_schema=review.schema(frame))
        result = review.apply_wire_review(first, wire, frame)
        self.assertEqual(result['outcome'], 'rewrite')
        self.assertEqual(result['expected_list_items'], first['expected_list_items'])
        self.assertEqual(result['table_record_coverage'], first['table_record_coverage'])
        self.assertEqual(result['answer'].count(first['expected_list_items'][0]), 1)
        self.assertIn(QUALIFIER, result['answer'])
        self.assertEqual(result['independent_context_review']['answer_sha256'], hashlib.sha256(result['answer'].encode()).hexdigest())
        emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
        self.assertEqual(original_context_sources(emitted, sources), sources)

    def test_header_quote_cannot_stand_in_for_distinct_prose(self):
        first, frame, _, _ = candidate()
        bad = verdict(frame)
        bad['table_contexts'][1]['source_quotes'][0]['quote'] = 'Oil temperature'
        result = review.apply_review(first, bad, frame)
        self.assertEqual(result['outcome'], 'no_sources')
        self.assertEqual(result['independent_context_review']['gate'], 'table_owner')

    def test_distinct_assembly_context_requires_its_own_source_without_transfer(self):
        bodies = {'doc:p1': SCOPE + '\n' + CELLS, 'other:p1': OTHER + '\n' + CELLS}
        first, frame, _, _ = candidate(bodies)
        good = verdict(frame)
        result = review.apply_review(first, good, frame)
        self.assertIn(OTHER, result['answer'])
        bad = deepcopy(good)
        bad['table_contexts'][1]['source_quotes'] = [{'citation_id': 'doc:p1', 'quote': SCOPE}]
        self.assertEqual(review.apply_review(first, bad, frame)['outcome'], 'no_sources')
        self.assertEqual(len(frame.table_owners), 1)

    def test_only_byte_identical_contexts_deduplicate_with_all_owners_visible(self):
        bodies = {'doc:p1': SCOPE + '\n' + CELLS, 'doc:p2': SCOPE + '\n' + QUALIFIER + '\n' + CELLS,
                  'doc:p3': SCOPE + '\n' + QUALIFIER + '\n' + CELLS}
        first, frame, sources, _ = candidate(bodies)
        targets = review._context_targets(frame)
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[1][3], ('doc:p2', 'doc:p3'))
        self.assertEqual(review.apply_review(first, verdict(frame), frame)['outcome'], 'rewrite')
        # Even whitespace differences remain separate; no fuzzy/subsequence equivalence.
        bodies['doc:p3'] = SCOPE + '\n ' + QUALIFIER + '\n' + CELLS
        self.assertEqual(len(review._context_targets(candidate(bodies)[1])), 3)

    def test_changed_cells_do_not_create_context_equivalence_and_source_mutation_rejects(self):
        bodies = {'doc:p1': SCOPE + '\n' + CELLS, 'doc:p2': QUALIFIER + '\n' + CELLS.replace('Periodic', 'Annual')}
        first, frame, _, _ = candidate(bodies)
        self.assertEqual(len(review._context_targets(frame)), 1)
        self.assertEqual(review.apply_review(first, verdict(frame), frame)['outcome'], 'rewrite')
        bad = deepcopy(first); bad['answer'] += ' extra'
        self.assertEqual(review.apply_review(bad, verdict(frame), frame)['outcome'], 'no_sources')

    def test_duplicate_with_no_prose_does_not_require_an_impossible_prose_quote(self):
        for empty in ('', '\n  \n'):
            first, frame, _, _ = candidate({'doc:p1': SCOPE + '\n' + CELLS, 'doc:p2': CELLS + empty})
            self.assertEqual(len(review._context_targets(frame)), 1)
            self.assertEqual(review.apply_review(first, verdict(frame), frame)['outcome'], 'rewrite')

    def test_moving_the_same_prose_across_the_table_is_a_distinct_context(self):
        bodies = {'doc:p1': SCOPE + '\n' + CELLS, 'doc:p2': CELLS + '\n' + SCOPE}
        first, frame, _, _ = candidate(bodies)
        self.assertEqual(len(review._context_targets(frame)), 2)
        self.assertEqual(review.apply_review(first, verdict(frame, omit_additional=True), frame)['outcome'], 'no_sources')


class DuplicateContextOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass(); cls.m = owned.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.SourceTableOwnedTests.tearDownClass()

    def test_real_source_first_admission_binding_and_public_html_preserve_alternate_context(self):
        for omit in (False, True):
            with self.subTest(missing_context=omit), ExitStack() as cleanup:
                completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                    payload=SimpleNamespace(company_id='offline-company', debug=True), started=time.monotonic())
                cleanup.callback(completion.close)
                with patch.object(owned, 'TABLE_BODY', SCOPE + '\n' + CELLS), patch.object(owned, 'CONDITION', SCOPE + '\n' + QUALIFIER + '\n' + CELLS):
                    f = owned.OwnedTableFixture(self)
                cleanup.callback(f.close)
                calls = []
                def provider(messages, **kw):
                    calls.append(deepcopy(messages))
                    if kw['json_schema']['name'] == 'factual_context_review':
                        targets = json.loads(messages[1]['content'].split('TABLE_CONTEXT_TARGETS:\n', 1)[1].split('\n\n', 1)[0])
                        self.assertEqual(len(targets), 2)
                        audits = [dict(target='item:0', context=SCOPE, documentary_limits_checked=True,
                            source_quotes=[dict(citation_id='inspection:p1', quote=SCOPE)])]
                        if not omit:
                            audits.append(dict(target='item:0:context:1', context=QUALIFIER, documentary_limits_checked=True,
                                source_quotes=[dict(citation_id='inspection:p2', quote=QUALIFIER)]))
                        reply = dict(outcome='rewrite', edits=[], citation_ids=['inspection:p1','inspection:p2'],
                            table_contexts=audits, checks={key:True for key in review.CHECKS}, reason='fixture')
                        sources = calls[0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                        reply = encode_context_reply(reply, messages, original_sources=sources, json_schema=kw['json_schema'])
                    else:
                        reply = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]', reason='fixture',
                            citation_ids=['inspection:p1'], covered_facets=['documented obligations'], missing_facets=[],
                            covered_answer_types=['checklist'], missing_answer_types=[], expected_list_items=['[[SOURCE_TABLE:G1]]'],
                            covered_list_items=[], missing_list_items=[], table_record_coverage=['T1=0','T2!same cells'])
                        reply = encode_table_coverage_reply(reply, kw['json_schema'])
                    class Response:
                        status_code = 200
                        def json(self): return dict(status='completed', output_text=json.dumps(reply), usage=dict(input_tokens=100,output_tokens=100))
                    budget = request_budget._v13_current_budget()
                    parsed = openai_transport.responses_json(messages, model=self.m.V13_FAST_MODEL,
                        json_schema=kw['json_schema'], effort=kw['effort'], reasoning_mode=kw['reasoning_mode'],
                        timeout=kw['timeout'], max_output_tokens=kw['max_output_tokens'], company_id=f.request.company_id,
                        purpose=kw['purpose'], api_key='offline-unused', url='https://offline.invalid/responses', post_fn=lambda *a,**k:Response(),
                        current_budget_fn=lambda:budget, response_text_fn=openai_transport.response_text,
                        safety_identifier_fn=openai_transport.safety_identifier)
                    return parsed, 'offline-reviewer'
                f.owner._response_flow.execution = replace(f.owner._response_flow.execution, _v13_json_models=provider)
                if omit:
                    with self.assertRaises(ResponseEvidenceError): f.run()
                else:
                    result = f.run()
                    self.assertEqual(result['status'], 'answered')
                    self.assertIn(QUALIFIER.rstrip('.'), result['answer'])
                    self.assertIn(QUALIFIER.rstrip('.'), result['answer_html'])
                    contract = result['meta']['assistant_core_validation']['answer_contract']
                    self.assertTrue(contract['semantic_answer_bound'])
                    self.assertEqual(contract['answer_sha256'], hashlib.sha256(result['answer'].encode()).hexdigest())
                    self.assertEqual([r['inventory_items'] for r in contract['semantic_verifier']['table_record_coverage']], [[0], []])
                self.assertEqual(len(calls), 2)
                self.assertEqual(f.writer_calls, [])


if __name__ == '__main__': unittest.main(verbosity=2)
