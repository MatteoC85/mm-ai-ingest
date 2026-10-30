"""Literal table references through the actual protected source-first ASK path.

SQL observations and model replies are fixtures. Evidence admission, source
compilation, provider accounting, validation, final binding and UI are real.
These tests establish representation/authority invariants, not model judgement.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_source_first_inventory as shared
from context_review_wire_fixture import encode_context_reply
from machinemind.ask import execution, validation, request_completion, factual_context_review
from machinemind.ask.response_evidence import ResponseEvidenceError
from machinemind.evidence.contracts import EvidenceContractError, SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.retrieval.ask_composition import AskEvidenceSession, AskSessionLimits, AskSelection
from machinemind.retrieval.chunk_evidence import ChunkReadScope
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.production_adapters import ProductionReaderAdapters
from machinemind.infrastructure import request_budget, openai_transport


TABLE_QUALIFIER = 'The sliding-assembly intervals apply without contamination; halve them under heavy loading.'
TABLE_BODY = (TABLE_QUALIFIER+'\n'+
    '- [PDF_TABLE_HEADER rows=1 columns=4] c1 r1="Assembly/action" | c2 r1="Interval" | c3 r1="Material" | c4 r1="Notes"\n'
    '- [PDF_TABLE_RECORD rows=2-3 columns=4] c1 "Assembly/action" r2="Slide inspection" | '
    'c2 "Interval" r2-3="Daily" | c3 "Material" r2-3="Grease ZTR-4" | '
    'c4 "Notes" r2-3="Inspect wear every 360 hours and grease only if necessary." | '
    'c1 "Assembly/action" r3="Visually check cleanliness and contamination."\n'
    'Continuation in the same provided source excerpt.\n'
    '- [PDF_TABLE_RECORD rows=1 columns=2] c1 r1="Inspect wiring integrity." | c2 r1="Monthly"')
CONDITION = 'Only trained personnel may perform the documented inspections with energy isolated.'


def table_contexts():
    return [{'target':'item:0','context':TABLE_QUALIFIER,'documentary_limits_checked':True,
             'source_quotes':[{'citation_id':'inspection:p1','quote':TABLE_QUALIFIER}]}]


class OwnedTableFixture:
    def __init__(self, test, *, change=None, revoke=False, mutate=False, narrow=False):
        self.test, self.m = test, test.m
        m = self.m
        query = ('Which slide inspections are documented?' if narrow else
                 'Which inspections and maintenance intervals are documented?')
        self.request = request = m.AssistantCoreRequest(query=query, requested_mode='ask',
            response_language='en', company_id='offline-company', machine_id='offline-machine',
            ai_scope='machine_all', top_k=6, allowed_effective_modes=('ask',))
        self.decision = m.AssistantCoreDecision(request_kind='factual', effective_mode='ask', confidence=.9,
            requested_mode_fit=True, evidence_state='supported', evidence_policy='evidence_required',
            information_task='document_explanation', required_facets=('documented obligations',),
            required_answer_types=('checklist',), facet_queries=(m.AssistantCoreFacetQuery(
                facet='documented obligations', answer_type='checklist', must_cover=True),))
        identity = SourceIdentity(SourceScope(request.company_id, ScopeLevel.MACHINE, request.machine_id),
            SourceType.DOCUMENT, 'inspection', SourceFormat.UNKNOWN)
        self.allowance = frozenset({identity})
        self.authorize = lambda req:self.allowance
        def invoke(callback, req, *args, **kwargs):
            test.assertIs(req, request)
            return callback(*args, **kwargs)
        limits = shared.limits_fixture(1, 1)[0].limits
        scope = ChunkReadScope(request.company_id, request.machine_id, request.ai_scope)
        self.session = AskEvidenceSession(request=request, scope=scope,
            limits=AskSessionLimits(limits, max_reads=20, max_records=400, max_bytes=128_000_000))
        bodies = [TABLE_BODY, CONDITION]
        capture = _PageReadCapture(scope, limits, 'ask_pages', snippet_chars=2000)
        capture.expect(2, 3000)
        capture.capture([('inspection', request.machine_id, i+1, body, request.company_id, request.machine_id)
                         for i, body in enumerate(bodies)])
        rows = [dict(bubble_document_id='inspection', page_from=i+1, page_to=i+1,
                     citation_id='inspection:p'+str(i+1), source_type='document',
                     chunk_full=body, snippet=body, similarity=.8) for i,body in enumerate(bodies)]
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
            if kwargs['json_schema']['name'] == 'factual_context_review':
                reply = dict(outcome='rewrite', edits=[], citation_ids=['inspection:p1','inspection:p2'],
                    table_contexts=table_contexts(),
                    checks={name:True for name in factual_context_review.CHECKS}, reason='offline independent fixture')
                reply = encode_context_reply(reply, messages)
            else:
                reply = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]', reason='offline fixture',
                    citation_ids=['inspection:p1', 'inspection:p2'], covered_facets=['documented obligations'],
                    missing_facets=[], covered_answer_types=['checklist'], missing_answer_types=[],
                    expected_list_items=['[[SOURCE_TABLE:G1]]', CONDITION], covered_list_items=[], missing_list_items=[],
                    table_record_coverage=['T1=0', 'T2=0'])
                if change is not None:
                    change(reply)
                self.wire_reply = deepcopy(reply)
            class Response:
                status_code = 200
                def json(self):
                    return {'status':'completed', 'output_text':json.dumps(reply),
                        'usage':{'input_tokens':100, 'output_tokens':100, 'output_tokens_details':{'reasoning_tokens':20}}}
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
        eruntime = owner._execution_factory()
        # Keep ResponseEvidenceFlow._verification_model around this callback.
        owner._response_flow.execution = replace(owner._response_flow.execution, _v13_json_models=provider)
        self.runtime = execution.bind_ask_execution(eruntime, session=self.session,
                                                    authorize=self.authorize, select=owner.select)

    def run(self):
        self.generated = execution.synthesize_ask(self.request, self.prepared, self.decision, runtime=self.runtime)
        vruntime = validation.bind_ask_validation(self.owner._validation_factory(), execution_runtime=self.runtime)
        self.validated = validation.validate_response(self.generated, self.request, self.prepared, self.decision, runtime=vruntime)
        from assistant_core_v2 import _finish_ask_validation, decorate_response
        projected = decorate_response(_finish_ask_validation(self.validated), self.request, self.decision)
        self.owner._finalize_core(self.request, projected)
        # Exact public request-flow renderer, not the separate generation compactor.
        return self.m._assistant_core_request_flow_runtime()._assistant_ui_finalize_response(
            projected, language='en', preserve_complete=True)

    def close(self):
        if self.owner._response_flow is not None:self.owner._response_flow.close()
        self.owner.active = False
        self.session.close()


class SourceTableOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared.AskSourceFirstInventoryTests.setUpClass()
        cls.m = shared.AskSourceFirstInventoryTests.m

    @classmethod
    def tearDownClass(cls):
        shared.AskSourceFirstInventoryTests.tearDownClass()

    def case(self, **kwargs):
        completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'), started=time.monotonic())
        fixture = OwnedTableFixture(self, **kwargs)
        self.addCleanup(completion.close)
        self.addCleanup(fixture.close)
        return completion, fixture

    def test_complete_cells_continuation_and_non_table_conditions_reach_public_ui(self):
        for narrow in (False, True):
            with self.subTest(narrow=narrow):
                completion, f = self.case(narrow=narrow)
                result = f.run()
                self.assertEqual(result['status'], 'answered')
                self.assertEqual(result['answer'], f.validated['answer'])
                contract = f.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['semantic_answer_bound'])
                self.assertEqual(hashlib.sha256(result['answer'].encode()).hexdigest(), contract['answer_sha256'])
                visible = self.m._assistant_ui_visible_text_from_html(result['answer_html'])
                for literal in ('Slide inspection', 'Daily', 'Grease ZTR-4',
                        'Inspect wear every 360 hours and grease only if necessary',
                        'Visually check cleanliness and contamination', 'Inspect wiring integrity', 'Monthly', CONDITION.rstrip('.')):
                    self.assertIn(literal, result['answer'])
                    self.assertIn(literal, visible)
                self.assertIn('source context', result['answer'])
                for marker in ('[[SOURCE_TABLE:', '[[FACTUAL_INVENTORY', 'PDF_TABLE_', '[TABLE_RECORD'):
                    self.assertNotIn(marker, result['answer'])
                self.assertEqual({c['citation_id'] for c in result['citations']}, {'inspection:p1','inspection:p2'})
                self.assertEqual(f.writer_calls, [])
                self.assertEqual(len(f.posts), 2)
                self.assertEqual(f.budget_ids, [id(completion.budget)] * 2)
                self.assertIn('[[SOURCE_TABLE:G1]] owner=inspection:p1', f.reviewer_calls[0][0][1]['content'])
                self.assertTrue(f.owner._response_flow.completed)
                f.close(); completion.close()

    def test_foreign_missing_owner_and_contradictory_coverage_fail_before_publication(self):
        changes = [
            lambda r:r.update(expected_list_items=['[[SOURCE_TABLE:G999]]', CONDITION]),
            lambda r:r.update(citation_ids=['inspection:p2']),
            lambda r:r.update(citation_ids=['foreign-tenant:p1', 'inspection:p2']),
            lambda r:r.update(table_record_coverage=['T1=0','T2!outside request']),
            lambda r:r.update(table_record_coverage=['T1=0','T2=1']),
            lambda r:r.update(table_record_coverage=['T1=0,1','T2=0']),
            lambda r:r.update(expected_list_items=['Grease the slide daily.', CONDITION]),
            lambda r:r.update(expected_list_items=['prefix [[SOURCE_TABLE:G1]]', CONDITION]),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                completion, f = self.case(change=change)
                result = f.run()
                self.assertEqual(result['status'], 'no_sources')
                self.assertEqual(result.get('citations') or [], [])
                self.assertNotIn('[[SOURCE_TABLE:', result.get('answer') or '')
                self.assertEqual(len(f.posts), 1)
                self.assertEqual(f.writer_calls, [])
                f.close(); completion.close()

    def test_no_sources_or_pass_cannot_publish_table_reference(self):
        for outcome in ('no_sources', 'pass'):
            completion, f = self.case(change=lambda r:r.update(outcome=outcome))
            self.assertEqual(f.run()['status'], 'no_sources')
            self.assertEqual(len(f.posts), 1)
            f.close(); completion.close()

    def test_source_revocation_or_mutation_cannot_use_precomputed_catalog(self):
        for options in ({'revoke':True}, {'mutate':True}):
            completion, f = self.case(**options)
            with self.assertRaises(EvidenceContractError):
                f.run()
            # The second verifier reuses a separately admitted immutable source
            # snapshot. A changed prepared mirror is rejected at final handoff;
            # it cannot enter either provider packet. Revocation stops earlier.
            self.assertEqual(len(f.posts), 1 if options.get('revoke') else 2)
            self.assertNotIn('UNTRUSTED CHANGE',json.dumps(f.reviewer_calls))
            if options.get('mutate'):
                first=f.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                second=f.reviewer_calls[1][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
                self.assertEqual(first,second)
            self.assertFalse(f.owner._response_flow.completed)
            f.close(); completion.close()

    def test_free_prose_still_requires_real_numeric_grounding(self):
        completion, f = self.case(change=lambda r:r.update(expected_list_items=[
            '[[SOURCE_TABLE:G1]]', 'Inspect the assembly every 999 hours.']))
        # The small fixture deliberately does not execute Core's bounded repair
        # loop. A failed validation cannot be finished as a successful response.
        with self.assertRaisesRegex(ResponseEvidenceError, 'final rejected draft lacks observed no-evidence builder'):
            f.run()
        contract = f.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertFalse(contract['semantic_answer_bound'])
        self.assertNotIn('999 hours', f.validated.get('answer') or '')
        self.assertEqual(len(f.posts), 2)
        self.assertFalse(f.owner._response_flow.completed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
