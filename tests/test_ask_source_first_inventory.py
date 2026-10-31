"""Owned source-first ASK boundary; SQL authority and model replies are fixtures.

Admission, source compilation, transport accounting, validation and final handoff
are real. This proves orchestration/guards, not the fixture model's judgement.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_prepared_scope_admission as shared
from context_review_wire_fixture import encode_context_reply, original_context_sources
from test_lineage_copy_reuse import fixture as limits_fixture
from machinemind.ask import execution, validation, response_evidence, review_source_wire, factual_context_review
from machinemind.evidence.contracts import EvidenceContractError, SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.retrieval.ask_composition import AskEvidenceSession, AskSessionLimits, AskSelection
from machinemind.retrieval.chunk_evidence import ChunkReadScope
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.production_adapters import ProductionReaderAdapters
from machinemind.ask import request_completion
from machinemind.infrastructure import request_budget, openai_transport


class SourceFirstFixture:
    def __init__(self, test, *, outcome='rewrite', empty_pass=False, missing=False,
                 unsupported=False, revoke=False, mutate=False, count=2, foreign=False):
        self.test, self.m = test, test.m
        m = self.m
        self.request = request = m.AssistantCoreRequest(query='Which inspections and maintenance intervals are documented?',
            requested_mode='ask', response_language='en', company_id='offline-company',
            machine_id='offline-machine', ai_scope='machine_all', top_k=6,
            allowed_effective_modes=('ask',))
        self.decision = m.AssistantCoreDecision(request_kind='factual', effective_mode='ask', confidence=.9,
            requested_mode_fit=True, evidence_state='supported', evidence_policy='evidence_required',
            information_task='document_explanation', required_facets=('documented obligations',),
            required_answer_types=('checklist',), facet_queries=(m.AssistantCoreFacetQuery(
                facet='documented obligations', answer_type='checklist', must_cover=True),))
        identity = SourceIdentity(SourceScope(request.company_id, ScopeLevel.MACHINE, request.machine_id),
            SourceType.DOCUMENT, 'maintenance', SourceFormat.UNKNOWN)
        self.allowance = frozenset({identity})
        self.authorize = lambda req:self.allowance
        def invoke(callback, req, *args, **kwargs):
            test.assertIs(req, request)
            return callback(*args, **kwargs)
        limits = limits_fixture(1, 1)[0].limits
        scope = ChunkReadScope(request.company_id, request.machine_id, request.ai_scope)
        self.session = AskEvidenceSession(request=request, scope=scope,
            limits=AskSessionLimits(limits, max_reads=20, max_records=400, max_bytes=128_000_000))
        self.items = ['Check slide cleanliness daily', 'Inspect slide wear every 360 hours and grease only if necessary']
        bodies = [item+'.' for item in self.items] + ['Supplemental component note '+chr(65+i)+'.' for i in range(count-2)]
        capture = _PageReadCapture(scope, limits, 'ask_pages', snippet_chars=800)
        capture.expect(count, 2000)
        capture.capture([('maintenance', request.machine_id, i+1, text, request.company_id, request.machine_id)
                         for i,text in enumerate(bodies)])
        rows = [dict(bubble_document_id='maintenance', page_from=i+1, page_to=i+1,
                     citation_id='maintenance:p'+str(i+1), source_type='document',
                     chunk_full=text, snippet=text, similarity=.8) for i,text in enumerate(bodies)]
        reg = self.session.register(request=request, receipt=capture.finish(rows), current_allowed_sources=self.allowance)
        rows = [dict(x.record) for x in self.session.records(request=request, handles=reg.records,
                                                          current_allowed_sources=self.allowance)]
        readers = ProductionReaderAdapters(request=request, session=self.session, authorize=self.authorize,
            invoke=invoke, runtimes=m._assistant_core_authority_reader_runtimes())
        self.owner = owner = m._assistant_core_intake_factory(request=request, session=self.session,
            readers=readers, authorize=self.authorize, invoke=invoke).__self__
        owner.active = owner._prepared = True
        owner._selection = (AskSelection('candidates', reg.records), AskSelection('citations', reg.records))
        self.prepared = dict(candidates=rows, citations=rows, plan={},
                             assistant_core_contract={'fail_closed':True}, retrieval_assurance={})
        owner._prepared_retrieval = deepcopy(self.prepared)
        self.writer_calls, self.reviewer_calls, self.posts, self.budget_ids = [], [], [], []
        def forbidden_writer(*args, **kwargs):
            self.writer_calls.append(True)
            raise AssertionError('PRELIMINARY_WRITER_MUST_NOT_RUN')
        owner.generation = replace(owner.generation, _v13_json_models=forbidden_writer,
            _sanitize_citations_for_response=lambda rows, **kw:[dict(x) for x in rows], _build_rg_links=lambda *a,**kw:[])
        owner.consumers = True
        def provider(messages, **kwargs):
            self.reviewer_calls.append((deepcopy(messages), deepcopy(kwargs)))
            budget = request_budget._v13_current_budget()
            self.budget_ids.append(id(budget))
            values = self.items+(['Foreign unsupported interval 999 hours'] if unsupported else [])
            answer = '' if empty_pass else ('Only '+self.items[0]+'.' if missing else '[[FACTUAL_INVENTORY]]')
            reply = dict(outcome=outcome, answer=answer, reason='offline fixture',
                citation_ids=['maintenance:p17'] if foreign else ['maintenance:p1','maintenance:p2'],
                covered_facets=['documented obligations'], missing_facets=[], covered_answer_types=['checklist'],
                missing_answer_types=[], expected_list_items=values, covered_list_items=[], missing_list_items=[])
            if kwargs['json_schema']['name'] == 'factual_context_review':
                reply = dict(outcome='pass', edits=[], citation_ids=['maintenance:p1','maintenance:p2'],
                    table_contexts=[],
                    checks={name:True for name in factual_context_review.CHECKS}, reason='offline independent fixture')
                first_sources = self.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                reply = encode_context_reply(reply, messages, original_sources=first_sources, json_schema=kwargs['json_schema'])
            class Response:
                status_code=200
                def json(self): return {'status':'completed', 'output_text':json.dumps(reply),
                    'usage':{'input_tokens':100,'output_tokens':100,'output_tokens_details':{'reasoning_tokens':20}}}
            def post(url, **kw):
                self.posts.append(kw['json'])
                if revoke:self.allowance=frozenset()
                if mutate:self.prepared['citations'][0]['chunk_full'] += '\nUNTRUSTED CHANGE'
                return Response()
            parsed = openai_transport.responses_json(messages, model=m.V13_FAST_MODEL,
                json_schema=kwargs['json_schema'], effort=kwargs['effort'], reasoning_mode=kwargs['reasoning_mode'],
                timeout=kwargs['timeout'], max_output_tokens=kwargs['max_output_tokens'], company_id=request.company_id,
                purpose=kwargs['purpose'], api_key='offline-unused', url='https://offline.invalid/responses', post_fn=post,
                current_budget_fn=lambda:budget, response_text_fn=openai_transport.response_text,
                safety_identifier_fn=openai_transport.safety_identifier)
            return parsed, 'offline-reviewer'
        eruntime=owner._execution_factory()
        owner._response_flow.execution=replace(owner._response_flow.execution, _v13_json_models=provider)
        self.runtime=execution.bind_ask_execution(eruntime, session=self.session, authorize=self.authorize, select=owner.select)

    def run(self):
        self.generated=execution.synthesize_ask(self.request,self.prepared,self.decision,runtime=self.runtime)
        vruntime=validation.bind_ask_validation(self.owner._validation_factory(), execution_runtime=self.runtime)
        self.final=validation.validate_response(self.generated,self.request,self.prepared,self.decision,runtime=vruntime)
        from assistant_core_v2 import _finish_ask_validation, decorate_response
        projected=decorate_response(_finish_ask_validation(self.final),self.request,self.decision)
        self.owner._finalize_core(self.request,projected)
        self.test.assertTrue(self.owner._response_flow.completed)
        return self.m._finalize_ask_response_for_ui(projected,language='en',preserve_complete=True)

    def close(self):
        if self.owner._response_flow is not None:self.owner._response_flow.close()
        self.owner.active=False
        self.session.close()


class AskSourceFirstInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared.AskPreparedScopeAdmissionTests.setUpClass()
        cls.m=shared.AskPreparedScopeAdmissionTests.m
    @classmethod
    def tearDownClass(cls):shared.AskPreparedScopeAdmissionTests.tearDownClass()

    def run_case(self, **kwargs):
        owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'), started=time.monotonic())
        h=SourceFirstFixture(self, **kwargs)
        self.addCleanup(owner.close);self.addCleanup(h.close)
        return owner,h

    def test_real_owned_recovery_has_generation_and_independent_review_then_finish(self):
        owner,h=self.run_case()
        limits=(owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd)
        result=h.run()
        self.assertEqual(result['status'],'answered')
        self.assertEqual(h.writer_calls,[])
        self.assertEqual(len(h.reviewer_calls),2)
        self.assertEqual(len(h.posts),2)
        self.assertEqual(h.budget_ids,[id(owner.budget)]*2)
        self.assertEqual(owner.budget.llm_calls,2)
        self.assertEqual((owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd),limits)
        self.assertEqual(h.generated['meta']['assistant_core_recovered_from'],'source_first_factual_inventory')
        for item in h.items:self.assertIn(item,result['answer'])
        self.assertNotIn('[[FACTUAL_INVENTORY',result['answer'])
        self.assertEqual({c['citation_id'] for c in result['citations']},{'maintenance:p1','maintenance:p2'})
        self.assertFalse(any(k.startswith('_assistant') for k in result))
        self.assertTrue(h.final['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])

    def test_rejected_or_incomplete_inventory_has_no_writer_or_second_review(self):
        for options in ({'outcome':'no_sources'},{'missing':True}):
            with self.subTest(options=options):
                owner,h=self.run_case(**options)
                result=h.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(h.writer_calls,[])
                self.assertEqual(len(h.posts),1)
                self.assertEqual(len(h.reviewer_calls),1)
                h.close();owner.close()

    def test_pass_without_a_current_draft_fails_closed(self):
        owner,h=self.run_case(outcome='pass',empty_pass=True)
        with self.assertRaisesRegex(EvidenceContractError,'reviewed_draft_empty_pass_without_draft'):h.run()
        self.assertEqual(h.writer_calls,[])
        self.assertEqual(len(h.posts),1)

    def test_source_revocation_or_mutation_prevents_publication(self):
        for options in ({'revoke':True},{'mutate':True}):
            with self.subTest(options=options):
                owner,h=self.run_case(**options)
                with self.assertRaises(EvidenceContractError):h.run()
                self.assertEqual(h.writer_calls,[])
                # Prepared-mirror mutation does not alter the already admitted
                # packet reused by the second verifier; final handoff rejects
                # that mirror. Revocation instead stops before another call.
                self.assertEqual(len(h.posts),1 if options.get('revoke') else 2)
                self.assertNotIn('UNTRUSTED CHANGE',json.dumps(h.reviewer_calls))
                if options.get('mutate'):
                    first=h.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                    second=h.reviewer_calls[1][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
                    self.assertEqual(first,original_context_sources(second,first))
                self.assertFalse(h.owner._response_flow.completed)
                h.close();owner.close()

    def test_source_beyond_recovery_cap_cannot_be_cited(self):
        owner,h=self.run_case(count=17,foreign=True)
        with self.assertRaisesRegex(response_evidence.ResponseEvidenceError,
                'canonical verifier unavailable: factual_context_review_unavailable'):
            h.run()
        self.assertFalse(h.owner._response_flow.completed)
        block=review_source_wire.restore_sources_block(h.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0])
        self.assertNotIn('[maintenance:p17]',block)
        self.assertEqual(len(h.posts),1)
        self.assertEqual(h.writer_calls,[])

    def test_real_grounding_still_rejects_unsupported_quantity_after_recovery(self):
        owner,h=self.run_case(unsupported=True)
        result=h.run()
        self.assertEqual(result['status'],'no_sources')
        self.assertFalse(h.final['meta']['assistant_core_validation']['answer_contract']['passed'])
        self.assertEqual(h.writer_calls,[])


if __name__=='__main__':unittest.main(verbosity=2)
