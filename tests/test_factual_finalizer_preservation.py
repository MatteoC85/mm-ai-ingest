"""Explicit preservation is independent of source type; default UI stays compact.

Owned-source/model fixtures test representation and handoffs, not model quality.
The public request-flow renderer is also checked separately from the generation
finalizer: production source-first uses that already-lossless renderer at exit.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_source_first_inventory as shared
from machinemind.ask import request_completion


class FactualFinalizerPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared.AskSourceFirstInventoryTests.setUpClass()
        cls.m = shared.AskSourceFirstInventoryTests.m

    @classmethod
    def tearDownClass(cls):
        shared.AskSourceFirstInventoryTests.tearDownClass()

    def source(self, index, role='document'):
        return dict(citation_id=f'document:p{index}', bubble_document_id='document',
                    page_from=index, page_to=index, evidence_role=role)

    def test_owned_factual_body_keeps_final_condition_in_text_and_html(self):
        completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'), started=time.monotonic())
        fixture = shared.SourceFirstFixture(self, count=8)
        self.addCleanup(completion.close)
        self.addCleanup(fixture.close)
        bodies = [c['chunk_full'] for c in fixture.prepared['citations']]
        # All source observations existed before selection. Put the conditional
        # maintenance sentence after five lines, where legacy compaction lost it.
        fixture.items = ['\n'.join(bodies[2:] + bodies[:2])]
        result = fixture.run()
        contract = fixture.final['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(result['answer'], fixture.final['answer'])
        self.assertEqual(len(fixture.posts), 2)
        self.assertEqual(fixture.reviewer_calls[1][1]['json_schema']['name'], 'factual_context_review')
        self.assertEqual(contract['semantic_verifier']['independent_context_review']['outcome'], 'pass')
        self.assertEqual(fixture.writer_calls, [])
        visible = self.m._assistant_ui_visible_text_from_html(result['answer_html'])
        for body in bodies:
            self.assertIn(body.rstrip('.'), result['answer'])
            self.assertIn(body.rstrip('.'), visible)
        # This is the actual request-flow finalizer, already lossless before the
        # change. No claim is made that production exit used the compacting one.
        public = completion.runtime._assistant_ui_finalize_response(
            deepcopy(result), language='en', preserve_complete=True)
        self.assertEqual(public['answer'], fixture.final['answer'])
        self.assertEqual(hashlib.sha256(public['answer'].encode()).hexdigest(),
                         hashlib.sha256(fixture.final['answer'].encode()).hexdigest())
        self.assertIn('grease only if necessary',
                      self.m._assistant_ui_visible_text_from_html(public['answer_html']))

    def test_explicit_preservation_keeps_long_document_text_without_raising_source_caps(self):
        text = ('- Inspect the documented surface without changing its settings.\n' * 180
                + '- Keep the documented isolation condition until inspection is complete.')
        rows = [self.source(i) for i in range(1, 11)]
        response = dict(status='answered', answer=text, citations=rows, rg_links=[])
        preserved = self.m._finalize_ask_response_for_ui(response, language='en', preserve_complete=True)
        compact = self.m._finalize_ask_response_for_ui(response, language='en')
        self.assertEqual(preserved['answer'], text)
        self.assertIn('until inspection is complete',
                      self.m._assistant_ui_visible_text_from_html(preserved['answer_html']))
        self.assertEqual(preserved['citations'], compact['citations'])
        self.assertLess(len(preserved['citations']), len(rows))

    def test_default_and_explicit_false_remain_compact(self):
        text = '\n'.join(f'Inspect documented component {letter}.' for letter in 'ABCDEFGH')
        response = dict(status='answered', answer=text, citations=[self.source(1)])
        implicit = self.m._finalize_ask_response_for_ui(response, language='en')
        explicit = self.m._finalize_ask_response_for_ui(response, language='en', preserve_complete=False)
        self.assertEqual(implicit, explicit)
        self.assertNotIn('component H', implicit['answer'])

    def test_procedure_preservation_and_sanitization_are_unchanged(self):
        text = '\n'.join(f'Inspect documented component {letter}.' for letter in 'ABCDEFGH')
        result = self.m._finalize_ask_response_for_ui(
            dict(status='answered', answer=text, citations=[self.source(1, 'procedure')]),
            language='en', preserve_complete=True)
        self.assertEqual(result['answer'], text)
        unsafe = self.m._finalize_ask_response_for_ui(
            dict(status='answered', answer='Reference <script>alert(1)</script>. password: synthetic-only',
                 citations=[self.source(1)]), language='en', preserve_complete=True)
        self.assertNotIn('synthetic-only', unsafe['answer'])
        self.assertNotIn('<script>', unsafe['answer_html'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
