"""Citation ownership survives every review representation, not model quality."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review, request_completion
from context_review_wire_fixture import encode_context_reply, original_context_sources
import test_factual_context_review as plain_fixture
import test_duplicate_table_context as table_fixture
import test_source_table_owned as owned


class CitationRoundtripTests(unittest.TestCase):
    def plain(self, template=True):
        helper = plain_fixture.FactualContextReviewTests()
        _, first, frame, _ = helper.candidate(table=False, template=template)
        messages = review.build_messages(frame, question='What inspections are documented?',
            response_language='en', sources_block=helper.sources)
        canonical = helper.verdict(table=False)
        canonical['reason'] = 'accepted'
        wire = encode_context_reply(canonical, messages)
        wire['additional_citation_ids'] = []
        return first, frame, helper.sources, messages, canonical, wire

    def tables(self, mode):
        count = 20 if mode == 'unbound_atomic' else 2
        bodies = {f'synthetic-{i}:p1': table_fixture.SCOPE + '\n'
            + f'This document applies to assembly {i}.\n' + table_fixture.CELLS
            for i in range(count)}
        if mode == 'non_atomic':
            bodies['synthetic-0:p1'] += '\n' + '\n'.join(
                f'Qualification {i}.' for i in range(1001))
        first, frame, sources, _ = table_fixture.candidate(bodies)
        atomic = review._atomic_references(frame)
        if mode == 'non_atomic':
            self.assertIsNone(atomic)
        else:
            self.assertIsNotNone(atomic)
            self.assertEqual(review._bound_context_references(frame, atomic) is not None, mode == 'bound')
        messages = review.build_messages(frame, question='Which intervals are documented?',
            response_language='en', sources_block=sources)
        canonical = table_fixture.verdict(frame)
        canonical['reason'] = 'accepted'
        canonical['citation_ids'] = list(dict.fromkeys([*first['citation_ids'],
            *(quote['citation_id'] for context in canonical['table_contexts'] for quote in context['source_quotes'])]))
        wire = encode_context_reply(canonical, messages, original_sources=sources)
        wire['additional_citation_ids'] = []
        return first, frame, sources, messages, canonical, wire

    def assert_uniform_schema(self, frame):
        schema = review.schema(frame)['schema']
        self.assertIn('additional_citation_ids', schema['required'])
        self.assertIn('additional_citation_ids', schema['properties'])
        self.assertNotIn('citation_ids', schema['properties'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['additional_citation_ids']['items']['enum'],
            list(review._citation_aliases(frame)))

    def test_plain_inventory_and_read_only_pass_preserve_current_citations_without_repetition(self):
        for template in (True, False):
            with self.subTest(template=template):
                first, frame, sources, messages, canonical, wire = self.plain(template)
                self.assertIsNone(review._atomic_references(frame))
                self.assert_uniform_schema(frame)
                result = review.apply_wire_review(first, wire, frame)
                self.assertEqual(result, review.apply_review(first, canonical, frame))
                self.assertEqual(result['answer'], first['answer'])
                self.assertEqual(result['citation_ids'], first['citation_ids'])
                emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
                self.assertEqual(emitted, sources)

    def test_plain_rewrite_adds_only_explicit_support_and_cannot_drop_prior_citations(self):
        first, frame, _, _, canonical, wire = self.plain()
        canonical.update(outcome='rewrite', citation_ids=['manual:p1', 'manual:p2'],
            edits=[{'target':'item:0', 'text':'Use the documented personal protective equipment before topping up.'}])
        wire.update(outcome='rewrite', additional_citation_ids=['S1'], edits=canonical['edits'])
        result = review.apply_wire_review(first, wire, frame)
        self.assertEqual(result, review.apply_review(first, canonical, frame))
        self.assertEqual(result['citation_ids'], ['manual:p1', 'manual:p2'])
        self.assertEqual(first['citation_ids'], ['manual:p1'])
        self.assertEqual(result['independent_context_review']['answer_sha256'],
            hashlib.sha256(result['answer'].encode()).hexdigest())

    def test_all_table_representations_preserve_exact_canonical_result_and_selected_quote_owners(self):
        for mode in ('bound', 'unbound_atomic', 'non_atomic'):
            with self.subTest(mode=mode):
                first, frame, sources, messages, canonical, wire = self.tables(mode)
                self.assert_uniform_schema(frame)
                # Additional IDs are empty: every new owner is selected by an
                # actual quote; the server must not infer one from table text.
                self.assertEqual(wire['additional_citation_ids'], [])
                result = review.apply_wire_review(first, wire, frame)
                self.assertEqual(result, review.apply_review(first, canonical, frame))
                self.assertEqual(result['outcome'], 'rewrite')
                self.assertEqual(set(result['citation_ids']), set(frame.admitted_citation_ids))
                self.assertEqual(result['expected_list_items'], first['expected_list_items'])
                self.assertEqual(result['table_record_coverage'], first['table_record_coverage'])
                emitted = messages[1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
                self.assertEqual(original_context_sources(emitted, sources), sources)

    def test_legacy_replacement_or_mixed_citation_fields_fail_in_every_wire_mode(self):
        for mode in ('plain', 'bound', 'non_atomic'):
            first, frame, _, _, _, wire = self.plain() if mode == 'plain' else self.tables(mode)
            for keep_additional in (True, False):
                bad = deepcopy(wire); bad['citation_ids'] = ['S0']
                if not keep_additional: bad.pop('additional_citation_ids')
                result = review.apply_wire_review(first, bad, frame)
                self.assertEqual(result['independent_context_review']['gate'], 'reply_shape')
                self.assertEqual(result['answer'], '')

    def test_no_sources_false_checks_foreign_additions_and_frame_change_cannot_approve(self):
        for mode in ('plain', 'bound', 'non_atomic'):
            first, frame, _, _, _, wire = self.plain() if mode == 'plain' else self.tables(mode)
            bad = deepcopy(wire); bad['outcome'] = 'no_sources'
            self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'model_rejected')
            for check in review.CHECKS:
                bad = deepcopy(wire); bad['checks'][check] = False
                self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'checks_false')
            for alias in ('S99999', 'foreign-tenant:p1', True):
                bad = deepcopy(wire); bad['additional_citation_ids'] = [alias]
                self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'citation_alias')
            changed = deepcopy(first); changed['citation_ids'] = []
            self.assertEqual(review.apply_wire_review(changed, wire, frame)['independent_context_review']['gate'], 'frame_mismatch')

    def test_fallback_quotes_still_require_literal_membership_owning_prose_and_complete_targets(self):
        first, frame, sources, _, _, wire = self.tables('non_atomic')
        first_quote = wire['table_contexts'][0]['source_quotes'][0]
        bad = deepcopy(wire); bad['table_contexts'][0]['source_quotes'][0]['quote_ref'] = 'Q99999'
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'quote_source')
        bad = deepcopy(wire); bad['table_contexts'][0]['source_quotes'][0]['citation_id'] = 'foreign-tenant:p1'
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'citation_alias')
        bad = deepcopy(wire); bad['table_contexts'][0]['source_quotes'][0] = {'quote': 'Fabricated literal'}
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'quote_shape')
        cell_ref = next(ref for ref, literal, _ in frame.quote_units if literal == 'Oil temperature')
        bad = deepcopy(wire); bad['table_contexts'][0]['source_quotes'][0]['quote_ref'] = cell_ref
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'table_owner')
        bad = deepcopy(wire); bad['table_contexts'].pop()
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'table_count')
        bad = deepcopy(wire); bad['table_contexts'][0]['documentary_limits_checked'] = False
        self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'documentary_limits_unchecked')
        damaged = replace(frame, quote_units=tuple((ref, 'Fabricated literal' if ref == first_quote['quote_ref'] else literal, owners)
            for ref, literal, owners in frame.quote_units))
        self.assertEqual(review.apply_wire_review(first, wire, damaged)['independent_context_review']['gate'], 'quote_not_literal')
        with self.assertRaisesRegex(ValueError, '^factual_context_review_invalid$'):
            review.build_messages(frame, question='q', response_language='en', sources_block=sources + ' changed')

    def test_pass_cannot_add_support_or_change_prose_and_read_only_cannot_rewrite(self):
        for template in (True, False):
            first, frame, _, _, _, wire = self.plain(template)
            bad = deepcopy(wire); bad['additional_citation_ids'] = ['S1']
            self.assertEqual(review.apply_wire_review(first, bad, frame)['independent_context_review']['gate'], 'pass_changed')
            bad = deepcopy(wire); bad['edits'] = [{'target':'item:0', 'text':'Changed'}]
            self.assertEqual(review.apply_wire_review(first, bad, frame)['outcome'], 'no_sources')
            if not template:
                bad['outcome'] = 'rewrite'
                self.assertEqual(review.apply_wire_review(first, bad, frame)['outcome'], 'no_sources')


class CitationRoundtripOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass(); cls.m = owned.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.SourceTableOwnedTests.tearDownClass()

    def test_plain_selected_inventory_roundtrip_through_owner_transport_hash_html_and_tenant_scopes(self):
        note = 'These documented facts address the request without listing every manual fact.'
        for company, machine, document in (('tenant-cobalt', 'unseen-mill', 'mill-guide'),
                                            ('tenant-ochre', 'unseen-valve', 'valve-guide')):
            with self.subTest(machine=machine), ExitStack() as cleanup:
                completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                    payload=SimpleNamespace(company_id=company), started=time.monotonic())
                cleanup.callback(completion.close)
                def first(reply):
                    reply.update(expected_list_items=[owned.CONDITION], table_record_coverage=[], selection_scope_note=note)
                def second(reply):
                    reply.update(outcome='pass', edits=[], table_contexts=[], citation_ids=[])
                f = owned.OwnedTableFixture(self, company_id=company, machine_id=machine,
                    document_id=document, change=first, second_change=second)
                cleanup.callback(f.close)
                f.decision = replace(f.decision, factual_coverage='useful')
                before = deepcopy(f.prepared); reads = f.session.summary()['registered_reads']
                limits = (completion.budget.deadline_monotonic, completion.budget.max_estimated_cost_usd)
                result = f.run()
                self.assertEqual(result['status'], 'answered')
                self.assertEqual(len(f.posts), 2); self.assertEqual(f.writer_calls, [])
                self.assertEqual(f.budget_ids, [id(completion.budget)] * 2)
                self.assertEqual(f.prepared, before); self.assertEqual(f.session.summary()['registered_reads'], reads)
                self.assertEqual((completion.budget.deadline_monotonic, completion.budget.max_estimated_cost_usd), limits)
                emitted_schema = f.posts[1]['text']['format']['schema']
                self.assertIn('additional_citation_ids', emitted_schema['required'])
                self.assertNotIn('citation_ids', emitted_schema['properties'])
                self.assertEqual(emitted_schema['properties']['table_contexts']['maxItems'], 0)
                contract = f.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['passed']); self.assertTrue(contract['semantic_answer_bound'])
                self.assertEqual(contract['answer_sha256'], hashlib.sha256(result['answer'].encode()).hexdigest())
                self.assertIn(owned.CONDITION, result['answer']); self.assertIn('energy isolated', result['answer_html'])
                self.assertEqual({c['citation_id'] for c in result['citations']}, {document+':p1', document+':p2'})
                self.assertEqual(contract['semantic_verifier']['table_record_coverage'], [])
                first_sources = f.reviewer_calls[0][0][1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCURRENT_ANSWER:', 1)[0]
                second_sources = f.reviewer_calls[1][0][1]['content'].split('SOURCES:\n', 1)[1].split('\n\nCOMPLETE_CANDIDATE:', 1)[0]
                self.assertEqual(second_sources, first_sources)


if __name__ == '__main__': unittest.main(verbosity=2)
