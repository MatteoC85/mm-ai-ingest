"""Conditions producer through actual owned SQL receipts and both model boundaries.

Only SQL rows, current allowance and provider replies are fixtures. The family
reader/relations, session, adapters, consumer flow, compiler and UI are real.
Provider quality is not asserted by replaying these predetermined replies.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution, validation, procedure_review, request_completion
from machinemind.infrastructure import request_budget, openai_transport
from machinemind.evidence.contracts import EvidenceContractError, SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.retrieval.ask_composition import AskEvidenceSession, AskSessionLimits, AskSelection
from machinemind.retrieval.chunk_evidence import ChunkReadScope
from machinemind.retrieval.page_evidence import _PageReadCapture
from machinemind.retrieval.production_adapters import ProductionReaderAdapters
import test_ask_prepared_scope_admission as prepared_fixtures
import test_ask_conditions_pipeline as condition_fixtures
from test_lineage_copy_reuse import fixture as limits_fixture


def catalog():
    def step(n, title, action, safety):
        return dict(step_id='install'+str(n), step_number=n, title=title,
            description='AZIONE OPERATIVA\n'+action+'\n\nNOTA DI SICUREZZA\n'+safety)
    return [dict(procedure_id='install', title='Inserimento cartuccia',
        short_description='Inserire la cartuccia nel filtro e ripristinare la custodia.', steps=[
            step(1, 'Preparazione sicura', 'Arrestare il ciclo e isolare le energie.', 'Solo personale addestrato.'),
            step(2, 'Inserire la cartuccia', 'Inserire la cartuccia nella guida libera.', 'Indossare guanti e occhiali; mantenere STOP premuto.'),
            step(3, 'Controllare ingresso', 'Verificare che la cartuccia entri diritta senza ostacoli.', 'Non entrare con alimentatore abilitato.'),
            step(4, 'Ripristino e prova', 'Dopo inserimento chiudere la custodia e provare a bassa velocita.', 'Verificare il consenso Ready dopo il ripristino.')]),
        dict(procedure_id='startup', title='Avviamento automatico', short_description='Mettere in ciclo automatico.', steps=[
            dict(step_id='startup4', step_number=4, title='Ripristinare sicurezza',
                description='AZIONE OPERATIVA\nRilasciare emergenze e premere reset prima dell avviamento.\n\nNOTA DI SICUREZZA\nAvvio solo volontario.')])]


class OwnedFamilyFixture:
    def __init__(self, test, *, language='en', kind='factual', revoke=False, tamper=False,
                 rows=None, chosen='install', query=None, decision=None,
                 eight_points=False, no_sources=False, unknown_parent=False, transport=False, clock=None):
        self.test, self.m = test, test.m
        m = self.m
        base = condition_fixtures.AskConditionsPipelineTests(); base.m = m
        request, default = base.request_decision(language)
        self.request = request = replace(request, allowed_effective_modes=('ask',), query=query or request.query)
        self.decision = decision = replace(decision or default, request_kind=kind)
        self.catalog = rows or catalog(); self.chosen = chosen
        self.calls, self.sql, self.writer_packets, self.review_packets = [], [], [], []
        self.revoke, self.tamper = revoke, tamper
        self.eight_points, self.no_sources, self.unknown_parent = eight_points, no_sources, unknown_parent
        self.transport, self.clock, self.posts, self.budget_ids = transport, clock, [], []
        self.rows, self.children = [], {}
        for parent in self.catalog:
            pk = 'procedure:'+parent['procedure_id']
            text = 'SOURCE_TYPE: procedure\nTITLE: '+parent['title']+'\nSHORT_DESCRIPTION: '+parent['short_description']
            self.rows.append((pk, request.machine_id, 1, text))
            self.children[pk] = []
            for step in parent['steps']:
                key = 'step:'+step['step_id']
                body = 'SOURCE_TYPE: step\nSTEP_NUMBER: '+str(step['step_number'])+'\nTITLE: '+step['title']+'\nDESCRIPTION: '+step['description']
                self.rows.append((key, request.machine_id, 1, body))
                self.children[pk].append((key, step['step_number'], request.machine_id, 1, body))
        sources = frozenset(SourceIdentity(SourceScope(request.company_id, ScopeLevel.MACHINE, request.machine_id),
            SourceType.PROCEDURE if key.startswith('procedure:') else SourceType.STEP,
            key.split(':', 1)[1], SourceFormat.STRUCTURED) for key, _, _, _ in self.rows)
        self.allowance = sources
        def authorize(req):
            test.assertIs(req, request)
            return self.allowance
        def invoke(callback, req, *args, **kwargs):
            test.assertIs(req, request)
            return callback(*args, **kwargs)
        limits = limits_fixture(1, 1)[0].limits
        self.scope = ChunkReadScope(request.company_id, request.machine_id, request.ai_scope)
        self.session = AskEvidenceSession(request=request, scope=self.scope,
            limits=AskSessionLimits(limits, max_reads=60, max_records=2000, max_bytes=128_000_000))
        # The initial search deliberately only sees the wrong startup Step. The
        # real catalog read and canonical relation must recover later precautions.
        seed = self.rows[-1]
        capture = _PageReadCapture(self.scope, limits, 'structured_direct', snippet_chars=800)
        capture.expect(1, 6000)
        capture.capture([(*seed, request.company_id, request.machine_id)])
        row = dict(bubble_document_id=seed[0], page_from=1, page_to=1, citation_id=seed[0]+':seed',
            source_type='step', chunk_full=seed[3], snippet=seed[3], similarity=.8)
        reg = self.session.register(request=request, receipt=capture.finish([row]), current_allowed_sources=sources)
        seeds = [dict(item.record) for item in self.session.records(request=request, handles=reg.records, current_allowed_sources=sources)]
        runtimes = {name: (replace(rt, _db_conn=lambda name=name:self.connection(name)) if hasattr(rt, '_db_conn') else rt)
                    for name, rt in m._assistant_core_authority_reader_runtimes().items()}
        readers = ProductionReaderAdapters(request=request, session=self.session, authorize=authorize, invoke=invoke, runtimes=runtimes)
        self.owner = owner = m._assistant_core_intake_factory(request=request, session=self.session,
            readers=readers, authorize=authorize, invoke=invoke).__self__
        owner.active = owner._prepared = True
        owner._selection = (AskSelection('candidates', reg.records), AskSelection('citations', reg.records))
        self.prepared = dict(candidates=seeds, citations=seeds, plan={}, assistant_core_contract={'fail_closed':True}, retrieval_assurance={})
        owner._prepared_retrieval = deepcopy(self.prepared)
        owner.tasks = replace(owner.tasks, _v13_json_models=self.provider)
        owner.generation = replace(owner.generation,
            _sanitize_citations_for_response=lambda items, **kw:[dict(item) for item in items],
            _build_rg_links=lambda *a, **kw:[])
        owner.consumers = True
        eruntime = replace(owner._execution_factory(), _v13_json_models=self.reviewer,
            _v13_current_budget=lambda:request_budget._v13_current_budget() or m._assistant_core_new_budget('ask'))
        self.runtime = execution.bind_ask_execution(eruntime, session=self.session, authorize=authorize, select=owner.select)

    def reply(self, reply, model, messages, kwargs):
        if not self.transport:return reply, model
        budget = request_budget._v13_current_budget()
        self.budget_ids.append(id(budget))
        fixture = self
        class Response:
            status_code = 200
            def json(self):return {'status':'completed', 'output_text':json.dumps(reply),
                'usage':{'input_tokens':100, 'output_tokens':100, 'output_tokens_details':{'reasoning_tokens':20}}}
        def post(url, **kw):
            fixture.posts.append(kw['json'])
            if fixture.clock is not None and len(fixture.posts) == 1:fixture.clock[0] += 88
            return Response()
        parsed = openai_transport.responses_json(messages, model=self.m.V13_FAST_MODEL,
            json_schema=kwargs['json_schema'], effort=kwargs['effort'], reasoning_mode=kwargs['reasoning_mode'],
            timeout=kwargs['timeout'], max_output_tokens=kwargs['max_output_tokens'], company_id=self.request.company_id,
            purpose=kwargs['purpose'], api_key='offline-unused', url='https://offline.invalid/responses', post_fn=post,
            current_budget_fn=lambda:budget, response_text_fn=openai_transport.response_text,
            safety_identifier_fn=openai_transport.safety_identifier)
        return parsed, model

    def connection(self, reader):
        owner = self
        class Cursor:
            def __enter__(self):return self
            def __exit__(self, *a):pass
            def execute(self, sql, params):
                owner.sql.append((reader, sql, params)); self.params=params; self.sql=sql
                owner.test.assertIn(owner.request.company_id, params)
            def fetchall(self):
                r, c, mid = owner.request, owner.request.company_id, owner.request.machine_id
                if reader == 'read_structured_direct_page_evidence':
                    return [(*row, c, mid) for row in owner.rows if row[0].startswith('procedure:')]
                if reader == 'read_related_step_page_evidence':
                    parent = self.params[3]; relation = self.params[4]
                    return [(*row, c, mid, parent, row[0], relation, 'procedure', 'step', row[1], 'bubble', {}, c, row[0])
                            for row in owner.children[parent]]
                if reader == 'read_parent_procedure_page_evidence':
                    relation = self.params[4]
                    found = []
                    for parent, children in owner.children.items():
                        parent_row = next(row for row in owner.rows if row[0] == parent)
                        for row in children:
                            if row[0] in self.params[3]:
                                found.append((row[0], parent, row[1], mid, 1, parent_row[3],
                                    c, mid, parent, row[0], relation, 'procedure', 'step', row[1], 'bubble', {}, c, parent))
                    return found
                if reader in {'read_deterministic_manual_support_page_evidence', 'read_document_file_references'}:
                    return []
                raise AssertionError('UNEXPECTED_FIXTURE_READER: '+reader)
        class Conn:
            def cursor(self):return Cursor()
            def close(self):pass
        return Conn()

    def provider(self, messages, **kwargs):
        props = kwargs['json_schema']['schema']['properties']
        is_selector = 'selected_procedure_citation_id' in props
        self.calls.append(('selector' if is_selector else 'writer', deepcopy(messages), deepcopy(kwargs)))
        if is_selector:
            chosen = next(cid for cid in props['selected_procedure_citation_id']['enum'] if cid.startswith('procedure:'+self.chosen+':'))
            if self.revoke:
                self.allowance = frozenset(s for s in self.allowance if s.source_id != self.chosen)
            if self.tamper:
                self.prepared['citations'][0]['chunk_full'] += '\nUNTRUSTED MUTATION'
                # The owner holds its admitted input snapshot separately; mutate
                # the actual in-flight structured boundary after selector dispatch.
                self.owner._task_input['synthesis.structured']['citations'][0]['chunk_full'] += '\nUNTRUSTED MUTATION'
            return self.reply({'selected_procedure_citation_id':'unadmitted-parent' if self.unknown_parent else chosen},
                'offline-selector', messages, kwargs)
        block = messages[1]['content'].split('SOURCES:\n', 1)[1]
        self.writer_packets.append(block)
        for key, _, _, body in self.rows:
            if key == 'procedure:'+self.chosen or any(key == row[0] for row in self.children['procedure:'+self.chosen]):
                self.test.assertIn(body, block)
            elif key.startswith('step:'):
                self.test.assertNotIn(body, block)
        citations = {key: next(cid for cid in re.findall(r'\[([^\]\n]+)\]', block) if cid.startswith(key+':'))
                     for key, *_ in self.children['procedure:'+self.chosen]}
        bodies = (["Before insertion, verify the cycle is stopped and energy isolated. Use trained personnel.",
                   "Wear gloves and goggles and keep STOP pressed during insertion.",
                   "Verify straight, unobstructed entry. Do not enter with the feeder enabled."] if self.request.response_language == 'en' else
                  ["Prima dell'inserimento verificare ciclo arrestato ed energie isolate. Solo personale addestrato.",
                   "Indossare guanti e occhiali e mantenere STOP premuto durante l'inserimento.",
                   "Verificare ingresso diritto e libero. Non entrare con alimentatore abilitato."])
        points = [{'text':text, 'citation_ids':[cid]} for text, cid in zip(bodies, citations.values())]
        if self.eight_points:
            # Eight source-grounded details, with the access prohibition last.
            texts = ['Verify the cycle is stopped.', 'Verify energy is isolated.', 'Use trained personnel.',
                'Wear gloves.', 'Wear goggles.', 'Keep STOP pressed.', 'Verify straight unobstructed entry.',
                'Do not enter with the feeder enabled.']
            cids = list(citations.values())
            points = [{'text':text,'citation_ids':[cids[index]]} for text,index in zip(texts,[0,0,0,1,1,1,2,2])]
        return self.reply({'answer_status':'no_sources' if self.no_sources else 'answered',
            'grounded_points':[] if self.no_sources else points}, 'offline-writer', messages, kwargs)

    def reviewer(self, messages, **kwargs):
        self.calls.append(('reviewer', deepcopy(messages), deepcopy(kwargs)))
        self.review_packets.append(messages[1]['content'])
        current = messages[1]['content'].split('CURRENT_ANSWER:\n', 1)[1].split('\n\nSOURCES:', 1)[0]
        ids = [row['citation_id'] for row in self.generated['citations']]
        reply = {'outcome':'pass', 'reply_mode':'retain', 'answer':'', 'edits':[], 'reason':'offline fixture',
            'covered_facets':list(self.decision.required_facets), 'missing_facets':[],
            'covered_answer_types':list(self.decision.required_answer_types), 'missing_answer_types':[],
            'expected_list_items':[], 'covered_list_items':[], 'missing_list_items':[], 'enumeration_requested':False,
            'citation_ids':ids, 'operation_boundary':{'complete':True,
                'checked_blocks':[b['block_id'] for b in procedure_review.block_layout(current)['blocks']], 'changes':[]}}
        return self.reply(reply, 'offline-reviewer', messages, kwargs)

    def run(self, *, validate=True):
        self.generated = execution.synthesize_ask(self.request, self.prepared, self.decision, runtime=self.runtime)
        if not validate:return self.generated
        vruntime = replace(self.owner._validation_factory(), execution_runtime=self.runtime,
            _assistant_core_verify_or_repair_answer=lambda **kw:execution.verify_or_repair_answer(**kw, runtime=self.runtime),
            _v13_current_budget=lambda:None)
        self.final = validation.validate_response(self.generated, self.request, self.prepared, self.decision, runtime=vruntime)
        from assistant_core_v2 import _finish_ask_validation, decorate_response
        projected = decorate_response(_finish_ask_validation(self.final), self.request, self.decision)
        self.owner._finalize_core(self.request, projected)
        self.test.assertTrue(self.owner._response_flow.completed)
        return self.m._finalize_ask_response_for_ui(projected,
            language=self.request.response_language, preserve_complete=True)

    def close(self):
        if self.owner._response_flow is not None:self.owner._response_flow.close()
        self.owner.active = False
        self.session.close()


class AskConditionsFamilyAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared_fixtures.AskPreparedScopeAdmissionTests.setUpClass()
        cls.m = prepared_fixtures.AskPreparedScopeAdmissionTests.m
    @classmethod
    def tearDownClass(cls):prepared_fixtures.AskPreparedScopeAdmissionTests.tearDownClass()

    def test_owned_family_expansion_writer_ui_and_review_in_both_languages(self):
        for language in ('en', 'it'):
            for kind in ('factual', 'procedure'):
                with self.subTest(language=language, kind=kind):
                    h = OwnedFamilyFixture(self, language=language, kind=kind)
                    try:
                        result = h.run()
                        self.assertEqual(result['status'], 'answered', result)
                        self.assertEqual([x[0] for x in h.calls], ['selector','writer','reviewer'])
                        self.assertEqual(len(h.generated['_assistant_core_validation_evidence']), 5)
                        self.assertEqual(len(h.generated['citations']), 3)
                        self.assertNotIn('_assistant_core_source_ordered_procedure', h.generated)
                        self.assertNotIn('_assistant_ui_model', result)
                        self.assertFalse(any(k.startswith('_assistant_core_') for k in result))
                        self.assertIn('STOP', result['answer_html'])
                        self.assertNotIn('Ready', result['answer'])
                        for _, _, _, body in h.rows[:5]:self.assertIn(body, h.review_packets[0])
                        self.assertNotIn('Rilasciare emergenze', h.writer_packets[0])
                        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])
                        for _, _, kw in h.calls[:2]:
                            self.assertEqual(kw['max_output_tokens'], h.owner.tasks.V13_FAST_MAX_OUTPUT_TOKENS)
                            self.assertEqual(kw['timeout'], h.owner.tasks.V13_FAST_TIMEOUT_SECONDS)
                            self.assertEqual(kw['purpose'], 'ask_structured_synthesis')
                    finally:h.close()

    def test_source_revocation_after_selector_stops_before_writer(self):
        h = OwnedFamilyFixture(self, revoke=True)
        try:
            with self.assertRaises(EvidenceContractError):h.run()
            self.assertEqual([x[0] for x in h.calls], ['selector'])
        finally:h.close()

    def test_admitted_input_tamper_after_selector_stops_before_writer(self):
        h = OwnedFamilyFixture(self, tamper=True)
        try:
            with self.assertRaises(EvidenceContractError):h.run()
            self.assertEqual([x[0] for x in h.calls], ['selector'])
        finally:h.close()

    def test_eight_bounded_points_preserve_last_precaution_through_final_handoff(self):
        h = OwnedFamilyFixture(self, eight_points=True)
        try:
            result = h.run()
            self.assertEqual(result['status'], 'answered')
            self.assertIn('8. Do not enter with the feeder enabled.', result['answer'])
            self.assertIn('8. Do not enter with the feeder enabled.', h.review_packets[0])
            self.assertEqual(len(h.posts), 0)
        finally:h.close()

    def test_no_sources_does_not_fall_back_to_unrelated_family(self):
        h = OwnedFamilyFixture(self, no_sources=True)
        try:
            result = h.run()
            self.assertEqual(result['status'], 'no_sources')
            self.assertEqual(result['citations'], [])
            self.assertEqual([x[0] for x in h.calls], ['selector','writer'])
        finally:h.close()

    def test_unadmitted_parent_id_stops_before_writer(self):
        h = OwnedFamilyFixture(self, unknown_parent=True)
        try:
            with self.assertRaises(EvidenceContractError):h.run()
            self.assertEqual([x[0] for x in h.calls], ['selector'])
        finally:h.close()

    def test_same_request_budget_and_review_reserve_at_each_real_transport(self):
        now = [1000.0]
        with patch.object(request_completion, '_monotonic', side_effect=lambda:now[0]), \
             patch.object(request_budget, '_monotonic', side_effect=lambda:now[0]):
            owner = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'), started=now[0])
            h = OwnedFamilyFixture(self, transport=True)
            try:
                budget = owner.budget
                limits = (budget.deadline_monotonic, budget.max_llm_calls, budget.max_estimated_cost_usd)
                result = h.run()
                self.assertEqual(result['status'], 'answered')
                self.assertEqual(len(h.posts), 3)
                self.assertEqual(h.budget_ids, [id(budget)] * 3)
                self.assertEqual(budget.llm_calls, 3)
                self.assertEqual((budget.deadline_monotonic, budget.max_llm_calls, budget.max_estimated_cost_usd), limits)
                schedules = owner.provider_schedules
                self.assertEqual([r['review_reserve_seconds'] for r in schedules[:2]], [10.5,10.5])
                self.assertEqual(schedules[-1]['purpose'], 'assistant_core_answer_contract_verifier')
            finally:h.close(); owner.close()

    def test_late_writer_cannot_consume_reserved_review_time_or_retry(self):
        now = [1000.0]
        with patch.object(request_completion, '_monotonic', side_effect=lambda:now[0]), \
             patch.object(request_budget, '_monotonic', side_effect=lambda:now[0]):
            owner = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'), started=now[0])
            h = OwnedFamilyFixture(self, transport=True, clock=now)
            try:
                with self.assertRaises(request_budget._V13BudgetExceeded):h.run()
                self.assertEqual(len(h.posts), 1)
                self.assertEqual(owner.budget.llm_calls, 1)
                self.assertFalse(owner.provider_schedules[-1]['admitted'])
                self.assertEqual(owner.budget.deadline_monotonic, 1100.0)
            finally:h.close(); owner.close()


if __name__ == '__main__':unittest.main(verbosity=2)
