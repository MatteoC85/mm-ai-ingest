"""Owned family bodies are required; retrieval support remains bounded.

The SQL/model replies are fixtures. Real Core ownership, preparation, generation,
review and optional real Responses transport run with external I/O denied.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import generation
from machinemind.evidence.contracts import SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.infrastructure import openai_transport, request_budget
from machinemind.retrieval.ask_composition import AskSelection
from machinemind.retrieval.page_evidence import _PageReadCapture
import test_ask_conditions_family_admission as owned
import test_ask_router_family_context as router


class SupportFixture(router.RouterFixture):
    def manuals(self, count=36, body_chars=1000):
        limits = owned.limits_fixture(1, 1)[0].limits
        capture = _PageReadCapture(self.scope, limits, 'structured_direct', snippet_chars=800)
        capture.expect(count, 6000)
        pages, rows, sources = [], [], []
        for i in range(count):
            key = 'optional-' + str(i)
            # Distinct synthetic support bodies, not a claimed live source pool.
            text = 'Maintenance reference ' + str(i) + ': ' + chr(65 + i % 26) * body_chars
            pages.append((key, self.request.machine_id, 1, text, self.request.company_id, self.request.machine_id))
            rows.append(dict(bubble_document_id=key, citation_id=key+':p1', page_from=1, page_to=1,
                source_type='document', snippet=text[:800], chunk_full=text, similarity=.22))
            sources.append(SourceIdentity(SourceScope(self.request.company_id, ScopeLevel.MACHINE,
                self.request.machine_id), SourceType.DOCUMENT, key, SourceFormat.UNKNOWN))
        self.allowance |= frozenset(sources)
        capture.capture(pages)
        registered = self.session.register(request=self.request, receipt=capture.finish(rows), current_allowed_sources=self.allowance)
        material = [dict(item.record) for item in self.session.records(request=self.request,
            handles=registered.records, current_allowed_sources=self.allowance)]
        handles = self.owner._selection[0].records + registered.records
        values = self.prepared['candidates'] + material
        self.prepared = {**self.prepared, 'candidates':values, 'citations':values}
        self.owner._selection = (AskSelection('candidates', handles), AskSelection('citations', handles))


class TransportFixture(SupportFixture):
    """Actual R4 settled prior spend/call state, fake HTTP Responses replies."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.transport_posts, self.transport_budget_ids = [], []

    def reply(self, reply, model, messages, kwargs):
        budget = request_budget._v13_current_budget()
        if not self.transport_posts:
            budget.estimated_cost_usd = .01876867
            budget.llm_calls = 1
            budget.call_log.append(dict(call=1, model='gpt-5.6-luna', purpose='observed_settled_router',
                dispatched=True, accounting_state='settled', reserved_cost_usd=0.))
            self.original_limits = (budget.max_estimated_cost_usd, budget.max_llm_calls, budget.deadline_monotonic)
            self.prior_row = deepcopy(budget.call_log[0])
        self.transport_budget_ids.append(id(budget))
        selected_model = kwargs.get('models', [self.m.V13_FAST_MODEL])[0]
        def post(url, **request):
            self.transport_posts.append(request['json'])
            return SimpleNamespace(status_code=200, json=lambda:dict(status='completed',
                output_text=json.dumps(reply), usage=dict(input_tokens=100, output_tokens=100)))
        parsed = openai_transport.responses_json(messages, model=selected_model,
            json_schema=kwargs['json_schema'], effort=kwargs['effort'], reasoning_mode=kwargs.get('reasoning_mode', ''),
            timeout=kwargs['timeout'], max_output_tokens=kwargs['max_output_tokens'], company_id=self.request.company_id,
            purpose=kwargs['purpose'], api_key='offline-unused', url='https://offline.invalid/responses', post_fn=post,
            current_budget_fn=lambda:budget, response_text_fn=openai_transport.response_text,
            safety_identifier_fn=openai_transport.safety_identifier,
            **({'require_full_output':True} if kwargs.get('require_full_output') else {}))
        self.test.assertEqual(self.original_limits,
            (budget.max_estimated_cost_usd, budget.max_llm_calls, budget.deadline_monotonic))
        self.test.assertEqual(self.prior_row, budget.call_log[0])
        self.final_budget = budget
        return parsed, selected_model


class AskBoundedProcedureSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.AskConditionsFamilyAdmissionTests.setUpClass(); cls.m = owned.AskConditionsFamilyAdmissionTests.m
    @classmethod
    def tearDownClass(cls): owned.AskConditionsFamilyAdmissionTests.tearDownClass()

    def assert_same_pack(self, h):
        writer = h.writer_packets[0]
        generation_packet = h.owner._generated_response['meta']['generation_evidence']
        review_packet = h.owner._response_flow._review_packet
        self.assertEqual(generation_packet['source_sha256'], review_packet.diagnostic(reused=False)['source_sha256'])
        self.assertEqual(generation_packet['emitted_citation_ids'], [r['citation_id'] for r in review_packet.rows])
        for key, _, _, body in h.rows:
            if key == 'procedure:install' or any(key == r[0] for r in h.children['procedure:install']):
                self.assertIn(body, writer)

    def test_optional_search_pool_is_bounded_and_review_reuses_exact_writer_pack(self):
        h = SupportFixture(self)
        try:
            h.manuals()
            result = h.run_core()
            self.assertEqual(result['status'], 'answered')
            self.assert_same_pack(h)
            admitted = h.owner._generated_response['_assistant_core_validation_evidence']
            self.assertLess(len(admitted), 36)
            self.assertEqual(len([r for r in admitted if r['source_type'] in {'procedure','step'}]), 5)
            self.assertLessEqual(len(h.owner._prepared_retrieval['candidates']), 5 + 12)
            self.assertEqual([x[0] for x in h.calls], ['router','writer','reviewer'])
        finally: h.close()

    def test_optional_body_over_context_bound_is_deferred_not_clipped_or_reintroduced(self):
        h = SupportFixture(self)
        try:
            h.manuals(count=1, body_chars=4000)
            h.owner.generation = replace(h.owner.generation, V13_FAST_CONTEXT_CHARS=1, V13_HEAVY_CONTEXT_CHARS=1)
            result = h.run_core()
            self.assertEqual(result['status'], 'answered')
            self.assert_same_pack(h)
            self.assertNotIn('Maintenance reference', h.writer_packets[0])
            self.assertTrue(all(r['bubble_document_id'] != 'optional-0'
                for r in h.owner._generated_response['_assistant_core_validation_evidence']))
        finally: h.close()

    def test_writer_cannot_cite_an_optional_record_that_was_not_emitted(self):
        h = SupportFixture(self, reject_review=True)
        try:
            h.manuals(count=1, body_chars=4000)
            h.owner.generation = replace(h.owner.generation, V13_FAST_CONTEXT_CHARS=1, V13_HEAVY_CONTEXT_CHARS=1)
            original = h.provider
            def provider(messages, **kw):
                reply, model = original(messages, **kw)
                reply['grounded_points'] = [dict(text='Unseen maintenance claim.', citation_ids=['optional-0:p1'])]
                return reply, model
            h.owner.generation = replace(h.owner.generation, _v13_json_models=provider)
            result = h.run_core()
            self.assertEqual(result['status'], 'no_sources')
            self.assertEqual(result['citations'], [])
        finally: h.close()

    def test_ordinary_source_bounds_fit_real_transport_after_observed_r4_router_spend(self):
        h = TransportFixture(self)
        try:
            h.manuals()
            result = h.run_core()
            self.assertEqual(result['status'], 'answered')
            self.assert_same_pack(h)
            self.assertEqual(len(h.transport_posts), 2)
            self.assertEqual(len(set(h.transport_budget_ids)), 1)
            self.assertEqual(h.final_budget.llm_calls, 3)
            self.assertEqual(h.final_budget.public_meta()['uncertain_cost_usd'], 0.)
            self.assertLess(h.final_budget.estimated_cost_usd, .25)
        finally: h.close()

    def test_large_required_family_does_not_erase_short_mandatory_manual_support(self):
        rows = router.catalog()
        for i in range(5, 23):
            rows[0]['steps'].append(dict(step_id='install'+str(i), step_number=i,
                title='Documented preparation condition '+str(i), description='Maintain the documented isolation and STOP condition.'))
        raw = router.context.router()
        raw['required_answer_types'].append('interface_locations')
        raw['facet_queries'].append(dict(facet='calibration voltage and calibration display menu path', answer_type='interface_locations',
            must_cover=True, dense_queries=['calibration voltage display menu path'], lexical_queries=['calibration voltage display menu path'],
            exact_terms=[], preferred_source_types=['document']))
        raw['required_facets'].append(raw['facet_queries'][-1]['facet'])
        h = SupportFixture(self, rows=rows, raw=raw,
            manual='Calibration voltage: 24 V. Calibration display menu path: Service > Calibration > Voltage.')
        try:
            result = h.run_core()
            self.assertEqual(result['status'], 'answered')
            self.assert_same_pack(h)
            self.assertIn('24 V', result['answer'])
            self.assertIn(h.manual, h.writer_packets[0])
            self.assertEqual(len([r for r in h.owner._generated_response['_assistant_core_validation_evidence']
                if r['source_type'] in {'procedure','step'}]), 23)
        finally: h.close()

    def test_required_family_alone_exceeds_real_cost_capacity_without_body_clipping(self):
        rows = router.catalog()
        for i in range(5, 13):
            rows[0]['steps'].append(dict(step_id='install'+str(i), step_number=i,
                title='Complete documented condition '+str(i), description='Documented isolation condition '+str(i)+'.'))
        h = TransportFixture(self, rows=rows)
        try:
            # The short search seed is an exact observed projection. Larger
            # complete bodies arrive through the canonical family page reader.
            h.rows = [(key, mid, page, body + ('X'*5000 if 'Complete documented condition' in body else ''))
                for key, mid, page, body in h.rows]
            h.children['procedure:install'] = [(key, ordinal, mid, page,
                next(r[3] for r in h.rows if r[0] == key))
                for key, ordinal, mid, page, _ in h.children['procedure:install']]
            with self.assertRaises(request_budget._V13BudgetExceeded): h.run_core()
            self.assertEqual(h.transport_posts, [])
            self.assertEqual([x[0] for x in h.calls], ['router','writer'])
            for _, _, _, body in h.rows:
                if 'Complete documented condition' in body: self.assertIn(body, h.writer_packets[0])
        finally: h.close()


if __name__ == '__main__': unittest.main()
