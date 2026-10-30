"""Owned preparation and real writer handoff for mixed-scale page expansion.

SQL observations and the model reply are fixtures. The source receipts, current
authority, canonical scorer, preparation journal, selection and writer compiler
are real. These tests do not certify live model quality.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_source_identifiers as base_fixture
from test_lineage_copy_reuse import fixture as limits_fixture
from machinemind.ask import execution
from machinemind.evidence.contracts import (EvidenceContractError, SourceIdentity,
    SourceScope, ScopeLevel, SourceType, SourceFormat)
from machinemind.retrieval.ask_composition import AskEvidenceSession, AskSelection, AskSessionLimits
from machinemind.retrieval.chunk_evidence import ChunkReadScope
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.production_adapters import ProductionReaderAdapters
from machinemind.retrieval import candidate_ranking as ranking, evidence_orchestration as orchestration


class EnumerationSelectionFixture:
    def __init__(self, m):
        self.m = m
        self.query = 'Which lubrication points and maintenance intervals are documented?'
        self.request = m.AssistantCoreRequest(query=self.query, requested_mode='ask',
            response_language='en', company_id='offline-company', machine_id='offline-machine',
            ai_scope='machine_all', top_k=6, debug=True, allowed_effective_modes=('ask',))
        facets = ('manual lubrication points', 'component maintenance intervals')
        self.decision = m.AssistantCoreDecision(request_kind='factual', effective_mode='ask',
            confidence=.99, requested_mode_fit=True, evidence_state='partial',
            evidence_policy='machine_sources_required', information_task='document_explanation',
            procedure_scope='other', required_answer_types=('checklist', 'explanation'),
            required_facets=facets, preferred_source_types=('document',), source_type_policy='require',
            facet_queries=tuple(m.AssistantCoreFacetQuery(facet=f, answer_type='checklist',
                must_cover=True, dense_queries=(f,), lexical_queries=(f,)) for f in facets))
        self.source = SourceIdentity(SourceScope(self.request.company_id, ScopeLevel.MACHINE,
            self.request.machine_id), SourceType.DOCUMENT, 'offline-document', SourceFormat.UNKNOWN)
        self.allowance = frozenset({self.source})
        self.authorizations = 0
        def authorize(req):
            assert req is self.request
            self.authorizations += 1
            return self.allowance
        self.authorize = authorize
        def invoke(fn, req, *args, **kwargs):
            assert req is self.request
            return fn(*args, **kwargs)
        self.invoke = invoke
        self.limits = limits_fixture(1, 1)[0].limits
        self.scope = ChunkReadScope(self.request.company_id, self.request.machine_id, 'machine_all')
        self.session = AskEvidenceSession(request=self.request, scope=self.scope,
            limits=AskSessionLimits(self.limits, max_reads=30, max_records=500, max_bytes=128_000_000))
        self.readers = ProductionReaderAdapters(request=self.request, session=self.session,
            authorize=authorize, invoke=invoke, runtimes=m._assistant_core_authority_reader_runtimes())
        self.owner = m._assistant_core_intake_factory(request=self.request, session=self.session,
            readers=self.readers, authorize=authorize, invoke=invoke).__self__
        self.owner.active = True
        self.calls = []; self.score_calls = []; self.read_calls = []
        self.base = [self.row(i + 1, text, semantic=.74, score=1.5 - .015 * i,
            facets=facets if i != 1 else facets[:1]) for i, text in enumerate((
                'Manual lubrication points: oil the designated carriage fittings daily. Preserve guards.',
                'Spindle guide lubrication: inspect every 240 hours; apply grease only if needed.',
                'Gear reducer inspection annually; this sealed reducer has lifetime lubricant.',
                'Automatic lubrication reservoir: check the oil level daily before automatic operation.'))]
        self.neighbors = [self.row(30 + i, '\n'.join(
            f'{j}. General maintenance responsibility for handbook item {i}-{j}.' for j in range(1, 12)),
            semantic=0.0, score=35.0 + i, neighbor=True) for i in range(12)]
        self.sections = [self.row(70, 'Electrical maintenance intervals: inspect cabinet condition monthly. '
            + 'Documented descriptive context. ' * 95 + 'COMPLETE_SECTION_END',
            semantic=0.0, score=80.0, neighbor=True)]

    def row(self, page, text, *, semantic, score, facets=(), neighbor=False):
        return {'citation_id': f'offline-document:p{page}:fixture',
            'bubble_document_id': 'offline-document', 'source_type': 'document',
            'page_from': page, 'page_to': page, 'chunk_index': page,
            'chunk_full': text, 'snippet': text[:800], 'similarity': .8,
            'semantic_similarity': semantic, 'exact_machine_scope': True,
            **({'retrieval_score': score, 'ask_evidence_score': score,
                'retrieval_assurance_kind': 'page_neighbor'} if neighbor
               else {'v13_score': score, 'retrieval_score': score,
                     'assistant_core_facet_hits': list(facets)})}

    def register(self, rows):
        capture = _PageReadCapture(self.scope, self.limits, 'ask_pages', snippet_chars=800)
        capture.expect(len(rows), 7000)
        capture.capture([(r['bubble_document_id'], self.request.machine_id,
            r['page_from'], r['chunk_full'], self.request.company_id, self.request.machine_id) for r in rows])
        receipt = capture.finish(deepcopy(rows))
        return self.session.register(request=self.request, receipt=receipt,
            current_allowed_sources=self.authorize(self.request)).records

    def run(self, *, tamper_score=False, revoke_after_score=False, mutate_after_prepare=False, source_first=False):
        initial = self.register(self.base)
        values = [dict(x.record) for x in self.session.records(request=self.request,
            handles=initial, current_allowed_sources=self.authorize(self.request))]
        data = {'candidates': values, 'citations': values, 'plan': {}, 'retrieval_assurance': {}}
        self.owner._remember(data, (AskSelection('candidates', initial), AskSelection('citations', initial)))
        original_score = ranking.v13_score_candidates
        def score(q, rows, **kwargs):
            self.score_calls.append({'ids': [r['citation_id'] for r in rows],
                'preserve_complete': kwargs.get('preserve_complete'),
                'before': deepcopy(rows)})
            result = original_score(q, rows, **kwargs)
            if tamper_score:
                result[0]['page_from'] = result[0]['page_to'] = 999
            if revoke_after_score:
                self.allowance = frozenset()
            return result
        def read(method, **kwargs):
            self.read_calls.append(method)
            assert kwargs.get('candidate_handles'), 'Owned source anchors required'
            if method == 'read_assurance_neighbor_page_evidence':
                return self.register(self.neighbors)
            if method == 'read_enumeration_page_evidence':
                return self.register(self.sections)
            raise AssertionError('Unexpected reader: ' + method)
        with patch.object(self.readers, 'candidate_handles', side_effect=read), \
             patch.object(ranking, 'v13_score_candidates', side_effect=score):
            prepared = self.owner.prepare(self.request, data, self.decision)
        self.prepared = prepared
        assert prepared['supported']
        def writer(messages, **kwargs):
            self.calls.append((deepcopy(messages), deepcopy(kwargs)))
            first = self.base[0]
            return {'answer_status': 'answered', 'grounded_points': [
                {'text': first['chunk_full'], 'citation_ids': [first['citation_id']]}]}, 'offline-model'
        self.owner.generation = replace(self.owner.generation, _v13_json_models=writer,
            _v13_choose_ask_model=lambda *a, **kw:(self.m.V13_FAST_MODEL, 'low', ''),
            _sanitize_citations_for_response=lambda rows, **kw:[dict(x) for x in rows],
            _build_rg_links=lambda *a, **kw:[])
        # Keep exercising the legacy eight-source writer after the same owned
        # preparation. Production's observed factual source-first path and its
        # full sixteen-source review are covered by test_ask_source_first_inventory.
        self.owner.consumers = source_first
        runtime = self.owner._execution_factory()
        if source_first:
            def reviewer(messages, **kwargs):
                self.calls.append((deepcopy(messages), deepcopy(kwargs)))
                first = self.base[0]
                if kwargs['json_schema']['name'] == 'factual_context_review':
                    from machinemind.ask.factual_context_review import CHECKS
                    return {'outcome':'pass','edits':[], 'citation_ids':[first['citation_id']],
                        'checks':{name:True for name in CHECKS},'reason':'offline independent fixture'}, 'offline-reviewer'
                return {'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]',
                    'expected_list_items':[first['chunk_full']], 'covered_list_items':[], 'missing_list_items':[],
                    'covered_facets':list(self.decision.required_facets), 'missing_facets':[],
                    'covered_answer_types':list(self.decision.required_answer_types), 'missing_answer_types':[],
                    'citation_ids':[first['citation_id']], 'reason':'offline fixture'}, 'offline-reviewer'
            self.owner._response_flow.execution = replace(self.owner._response_flow.execution,
                _v13_json_models=reviewer)
        runtime = execution.bind_ask_execution(runtime, session=self.session,
            authorize=self.authorize, select=self.owner.select)
        payload = prepared['retrieval']
        if mutate_after_prepare:
            payload = deepcopy(payload)
            payload['citations'][0]['chunk_full'] = 'Injected foreign unobserved text.'
        result = execution.synthesize_ask(self.request, payload, self.decision, runtime=runtime)
        return result

    def close(self):
        if self.owner._response_flow is not None:
            self.owner._response_flow.close()
        self.owner.active = False
        self.session.close()


class AskEnumerationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_fixture.AskSourceIdentifierTests.setUpClass()
        cls.m = base_fixture.AskSourceIdentifierTests.m

    @classmethod
    def tearDownClass(cls):
        base_fixture.AskSourceIdentifierTests.tearDownClass()

    def test_owned_selection_retains_complementary_sources_within_same_writer_cap(self):
        f = EnumerationSelectionFixture(self.m)
        before = deepcopy((f.base, f.neighbors, f.sections))
        try:
            result = f.run()
            self.assertEqual(result['status'], 'answered')
            self.assertEqual(len(f.calls), 1)
            prompt = f.calls[0][0][1]['content']
            for row in f.base:
                self.assertIn(row['chunk_full'], prompt)
            self.assertIn(f.sections[0]['chunk_full'], prompt)
            self.assertIn('COMPLETE_SECTION_END', prompt)
            self.assertLessEqual(len(f.prepared['retrieval']['citations']), 16)
            self.assertLessEqual(len(result['debug']['v13_ask']['evidence_ids']), 8)
            self.assertEqual(len(f.score_calls), 2)
            self.assertTrue(all(c['preserve_complete'] is True for c in f.score_calls))
            scored_ids = {cid for c in f.score_calls for cid in c['ids']}
            self.assertFalse(scored_ids.intersection(r['citation_id'] for r in f.base))
            self.assertEqual(scored_ids, {r['citation_id'] for r in f.neighbors + f.sections})
            self.assertEqual((f.base, f.neighbors, f.sections), before)
            self.assertGreater(f.authorizations, 6)
        finally:
            f.close()

    def test_observed_source_first_review_receives_owned_ranked_complementary_records(self):
        from types import SimpleNamespace
        import time
        from machinemind.ask import request_completion, review_source_wire
        completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'), started=time.monotonic())
        f=EnumerationSelectionFixture(self.m)
        try:
            result=f.run(source_first=True)
            self.assertEqual(result['status'],'answered')
            self.assertEqual(len(f.calls),2)
            messages,kwargs=f.calls[0]
            self.assertEqual(kwargs['purpose'],'assistant_core_answer_contract_verifier')
            first_wire=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
            second_messages,second_kwargs=f.calls[1]
            self.assertEqual(second_kwargs['json_schema']['name'],'factual_context_review')
            self.assertEqual(second_kwargs['purpose'],'assistant_core_answer_contract_verifier')
            second_wire=second_messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
            self.assertEqual(first_wire,second_wire)
            block=review_source_wire.restore_sources_block(first_wire)
            for row in f.base + f.sections:self.assertIn(row['chunk_full'],block)
            self.assertIn('COMPLETE_SECTION_END',block)
            self.assertLessEqual(len(f.prepared['retrieval']['citations']),16)
            self.assertEqual(len(f.score_calls),2)
            self.assertTrue(all(c['preserve_complete'] is True for c in f.score_calls))
            self.assertIsNotNone(f.owner._response_flow.synthesis)
        finally:f.close();completion.close()

    def test_newcomer_routing_scores_do_not_become_semantic_similarity(self):
        f = EnumerationSelectionFixture(self.m)
        try:
            f.run()
            rows = f.prepared['retrieval']['candidates']
            new = [r for r in rows if r.get('retrieval_assurance_kind')]
            self.assertTrue(new)
            for row in new:
                self.assertEqual(row['semantic_similarity'], 0.0)
                self.assertLess(row['v13_score'], 5.0)
                self.assertEqual(row['chunk_full'], next(x['chunk_full'] for x in f.neighbors + f.sections
                    if x['citation_id'] == row['citation_id']))
        finally:
            f.close()

    def test_scorer_cannot_rebind_page_location(self):
        f = EnumerationSelectionFixture(self.m)
        try:
            with self.assertRaises(EvidenceContractError):
                f.run(tamper_score=True)
            self.assertEqual(f.calls, [])
        finally:
            f.close()

    def test_revoked_source_stops_before_writer(self):
        f = EnumerationSelectionFixture(self.m)
        try:
            with self.assertRaises(EvidenceContractError):
                f.run(revoke_after_score=True)
            self.assertEqual(f.calls, [])
        finally:
            f.close()

    def test_prepared_text_tampering_stops_before_writer(self):
        f = EnumerationSelectionFixture(self.m)
        try:
            with self.assertRaises(EvidenceContractError):
                f.run(mutate_after_prepare=True)
            self.assertEqual(f.calls, [])
        finally:
            f.close()

    def test_root_smart_and_unbound_ask_do_not_use_new_selection_branch(self):
        f = EnumerationSelectionFixture(self.m)
        try:
            for mode in ('root_cause', 'smart_diagnostic', 'ask'):
                with self.subTest(mode=mode):
                    def forbidden_score(*args, **kwargs):
                        self.fail('Noncanonical mode entered ASK page normalization')
                    runtime = replace(self.m._assistant_core_prepare_evidence_runtime(),
                        _v13_score_candidates=forbidden_score,
                        _assistant_core_root_diagnostic_evidence_assurance=lambda req, data, dec:data,
                        _assistant_core_enumeration_requested=lambda *args:True,
                        _v13_assurance_fetch_neighbor_pages=lambda **kw:deepcopy(f.neighbors),
                        _assistant_core_expand_enumeration_sections=lambda **kw:deepcopy(f.sections))
                    rows = deepcopy(f.base)
                    data = {'candidates': rows, 'citations': rows, 'plan': {}}
                    original = deepcopy(data)
                    result = orchestration.assistant_core_prepare_evidence(
                        replace(f.request, requested_mode=mode), data,
                        replace(f.decision, effective_mode=mode), runtime=runtime)
                    self.assertIn('supported', result)
                    self.assertEqual(data, original)
        finally:
            f.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
