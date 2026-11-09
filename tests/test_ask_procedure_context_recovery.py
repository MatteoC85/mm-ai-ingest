"""Owned mixed-ASK family recovery. SQL/provider replies are offline fixtures.

The real preparation, typed readers, writer, independent review and final source
binding run. These tests prove contracts/source flow, not provider answer quality.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import normalize_decision, AssistantCoreV2
from machinemind.ask import execution, task_generation, procedure_review
from machinemind.ask.request_binding import run_core_request, AskRequestEvidence
from machinemind.ask import request_completion
from machinemind.infrastructure import request_budget
from types import SimpleNamespace
import time
from machinemind.evidence.contracts import EvidenceContractError, SourceIdentity, SourceScope, SourceType, SourceFormat, ScopeLevel
from machinemind.retrieval.ask_composition import AskSelection
from machinemind.retrieval.page_evidence import _PageReadCapture
import test_numeric_conditions_family as numeric
import test_ask_conditions_family_admission as owned


def router(scope=''):
    raw = numeric.raw_router()
    raw.update(procedure_scope=scope, evidence_state='unsupported')
    raw['facet_queries'].append(dict(facet='separate initial checks from the later low-speed test',
        answer_type='state_sequence', must_cover=True, dense_queries=['initial checks later low-speed test'],
        lexical_queries=['prima verifica prova'], exact_terms=[], preferred_source_types=['procedure','step']))
    raw['required_facets'].append(raw['facet_queries'][-1]['facet'])
    return raw


class ContextFixture(numeric.NumericFamilyFixture):
    def __init__(self, test, *, scope='', rows=None, raw=None, query=None, omit_review_type=None, reject_review=False, **kw):
        self.omit_review_type = omit_review_type
        self.reject_review = reject_review
        if rows is None:
            rows = deepcopy(owned.catalog()[:1])
            rows[0]['steps'][0]['description'] += '\nMaximum permitted cartridge mass: 40 kg.'
        req = numeric.NumericConditionsFamilyTests.request(test)
        req = replace(req, query=query or req.query)
        decision = normalize_decision(req, raw or router(scope))
        super().__init__(test, rows=rows, decision=decision, kind='procedure', query=req.query, **kw)
        self.owner._response_flow.close(); self.owner._response_flow = None
        self.owner._prepared = False
        self.owner.generation = replace(self.owner.generation, _v13_json_models=self.provider)
        # Actual run_core_request snapshots consumers BEFORE Core.prepare.
        er = replace(self.owner._execution_factory(), _v13_json_models=self.reviewer,
            _v13_current_budget=lambda:request_budget._v13_current_budget() or self.m._assistant_core_new_budget('ask'))
        self.runtime = execution.bind_ask_execution(er, session=self.session,
            authorize=self.owner.authorize, select=self.owner.select)

    def add_manual_noise(self, count=20):
        sources = frozenset(SourceIdentity(SourceScope(self.request.company_id,ScopeLevel.MACHINE,self.request.machine_id),
            SourceType.DOCUMENT,'noise'+str(i),SourceFormat.UNKNOWN) for i in range(count))
        self.allowance |= sources
        limits = owned.limits_fixture(1,1)[0].limits
        capture = _PageReadCapture(self.scope,limits,'structured_direct',snippet_chars=800)
        capture.expect(count,6000)
        rows, pages = [], []
        for i in range(count):
            key = 'noise'+str(i)
            body = 'Unrelated maintenance inventory page '+str(i)+'.'
            pages.append((key,self.request.machine_id,1,body,self.request.company_id,self.request.machine_id))
            rows.append(dict(bubble_document_id=key,page_from=1,page_to=1,citation_id=key+':p1',
                source_type='document',chunk_full=body,snippet=body,similarity=.1))
        capture.capture(pages)
        registered = self.session.register(request=self.request,receipt=capture.finish(rows),
            current_allowed_sources=self.allowance)
        material = [dict(item.record) for item in self.session.records(request=self.request,
            handles=registered.records,current_allowed_sources=self.allowance)]
        handles = self.owner._selection[0].records + registered.records
        complete = self.prepared['candidates'] + material
        self.prepared = {**self.prepared,'candidates':complete,'citations':complete}
        self.owner._selection = (AskSelection('candidates',handles),AskSelection('citations',handles))

    def prepare(self):
        result = self.owner.prepare(self.request, self.prepared, self.decision)
        self.preparation_result = result
        self.prepared = result['retrieval']
        return result

    def reviewer(self, messages, **kw):
        reply, model = super().reviewer(messages, **kw)
        reply['covered_answer_types'] = list(set(reply['covered_answer_types']) | {'state_sequence'})
        if self.omit_review_type:
            reply['covered_answer_types'] = [x for x in reply['covered_answer_types'] if x != self.omit_review_type]
        if self.reject_review:
            reply.update(outcome='no_sources',reply_mode='replace',answer='',citation_ids=[],
                missing_facets=[self.decision.required_facets[-1]],missing_answer_types=['state_sequence'])
        return reply, model


class AskProcedureContextRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m = owned.AskConditionsFamilyAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        owned.AskConditionsFamilyAdmissionTests.tearDownClass()

    def test_actual_r2_shape_and_scope_variants_keep_obligations_and_complete_context(self):
        for scope in ('', 'conditions_only', 'execution_steps', 'other'):
            with self.subTest(scope=scope):
                h = ContextFixture(self, scope=scope)
                try:
                    before = deepcopy(h.decision)
                    consumer = h.runtime
                    self.assertFalse(consumer.preserve_procedure_context)
                    prepared = h.prepare()
                    self.assertIs(consumer,h.runtime)
                    self.assertTrue(prepared['supported'], prepared)
                    self.assertTrue(h.runtime.preserve_procedure_context)
                    result = h.run()
                    self.assertEqual(h.decision, before)
                    self.assertEqual(result['status'], 'answered', result)
                    self.assertEqual([call[0] for call in h.calls], ['writer','reviewer'])
                    self.assertEqual(len(h.writer_packets), 1)
                    self.assertEqual(len(h.review_packets), 1)
                    for _, _, _, body in h.rows:
                        self.assertIn(body, h.writer_packets[0])
                        self.assertIn(body, h.review_packets[0])
                    self.assertIn('40 kg', result['answer'])
                    self.assertIn('low-speed test', result['answer'])
                    self.assertIn('state_sequence', h.calls[-1][1][1]['content'])
                finally:h.close()

    def test_no_seed_or_missing_canonical_relation_cannot_create_context(self):
        for missing in ('seed', 'relation'):
            h = ContextFixture(self)
            try:
                if missing == 'seed':
                    data = {**h.prepared, 'candidates':[], 'citations':[]}
                    selections = (AskSelection('candidates',()), AskSelection('citations',()))
                else:
                    h.children = {}
                    data, selections = h.prepared, h.owner._selection
                op = task_generation.TaskGenerationEvidence(request=h.request, session=h.session,
                    readers=h.owner.readers, authorize=h.owner.authorize, invoke=h.owner.invoke,
                    runtime=h.owner.generation, tasks=h.owner.tasks)
                out, selected, recovered = op.prepare_family_context(data, selections, h.decision)
                self.assertFalse(recovered)
                self.assertIs(out, data)
                self.assertEqual(selected, selections)
                self.assertEqual(h.calls, [])
            finally:h.close()

    def test_manual_ranking_noise_cannot_clip_later_family_safety_context(self):
        h = ContextFixture(self)
        try:
            h.add_manual_noise()
            prepared = h.prepare()
            self.assertTrue(prepared['supported'])
            result = h.run()
            self.assertEqual(result['status'],'answered')
            self.assertEqual([c[0] for c in h.calls],['writer','reviewer'])
            for _,_,_,body in h.rows:
                self.assertIn(body,h.writer_packets[0])
                self.assertIn(body,h.review_packets[0])
        finally:h.close()

    def test_r1_observed_conditions_shape_uses_same_owned_context_without_selector(self):
        h = ContextFixture(self,raw=numeric.raw_router())
        try:
            self.assertEqual(h.decision.information_task,'procedure_segment')
            self.assertTrue(h.prepare()['supported'])
            self.assertEqual(h.run()['status'],'answered')
            self.assertEqual([c[0] for c in h.calls],['writer','reviewer'])
        finally:h.close()

    def test_ambiguous_canonical_seed_parent_cannot_select_either_family(self):
        h = ContextFixture(self,rows=owned.catalog())
        try:
            h.children['procedure:install'].append(h.children['procedure:startup'][0])
            op = task_generation.TaskGenerationEvidence(request=h.request,session=h.session,
                readers=h.owner.readers,authorize=h.owner.authorize,invoke=h.owner.invoke,
                runtime=h.owner.generation,tasks=h.owner.tasks)
            out,_,recovered = op.prepare_family_context(h.prepared,h.owner._selection,h.decision)
            self.assertFalse(recovered)
            self.assertIs(out,h.prepared)
            self.assertEqual(h.calls,[])
        finally:h.close()

    def test_transport_has_only_writer_and_mandatory_reviewer_on_original_budget(self):
        completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company'),started=time.monotonic())
        h = ContextFixture(self,transport=True)
        try:
            budget = completion.budget
            before = (budget.max_llm_calls,budget.max_estimated_cost_usd,budget.deadline_monotonic)
            self.assertTrue(h.prepare()['supported'])
            self.assertEqual(h.run()['status'],'answered')
            self.assertEqual(len(h.posts),2)
            self.assertEqual(h.budget_ids,[id(budget)]*2)
            self.assertEqual((budget.max_llm_calls,budget.max_estimated_cost_usd,budget.deadline_monotonic),before)
        finally:h.close();completion.close()

    def test_mandatory_facet_type_omitted_by_a_pass_reply_is_rejected_before_publication(self):
        h = ContextFixture(self,omit_review_type='state_sequence')
        try:
            self.assertTrue(h.prepare()['supported'])
            with self.assertRaisesRegex(ValueError,'procedure_review_coverage_not_complete'):h.run()
            self.assertFalse(h.owner._response_flow.completed)
            self.assertEqual([c[0] for c in h.calls],['writer','reviewer'])
        finally:h.close()

    def test_no_implicit_ordered_actions_but_explicit_order_obligation_is_preserved(self):
        for explicit in (False,True):
            raw = router()
            if explicit:raw['required_answer_types'].append('ordered_actions')
            h = ContextFixture(self,raw=raw)
            try:
                self.assertTrue(h.prepare()['supported'])
                self.assertEqual(h.run()['status'],'answered')
                for _,messages,_ in h.calls:
                    line = next(x for x in messages[1]['content'].splitlines() if x.startswith('REQUIRED_ANSWER_TYPES:'))
                    self.assertEqual('ordered_actions' in line,explicit)
                    self.assertIn('state_sequence',line)
            finally:h.close()

    def test_wrong_machine_child_page_cannot_enter_family_context(self):
        h = ContextFixture(self)
        try:
            row = h.children['procedure:install'][0]
            h.children['procedure:install'][0] = (row[0],row[1],'other-machine',row[3],row[4])
            with self.assertRaises(EvidenceContractError):h.prepare()
            self.assertEqual(h.calls,[])
        finally:h.close()

    def test_duplicate_canonical_ordinals_and_ambiguous_body_capacity_fail_closed(self):
        for violation in ('ordinal','projection'):
            h = ContextFixture(self)
            try:
                row = h.children['procedure:install'][1]
                h.children['procedure:install'][1] = (row[0],1 if violation=='ordinal' else row[1],row[2],
                    row[3],('x'*100000 if violation=='projection' else row[4]))
                with self.assertRaises(EvidenceContractError):h.prepare()
                self.assertEqual(h.calls,[])
            finally:h.close()

    def test_independent_reviewer_can_reject_unavailable_states_and_missing_requested_actions(self):
        for query in (None,'Explain the complete cartridge installation and restoration procedure, the mass limit, initial checks and precautions.'):
            h = ContextFixture(self,query=query,reject_review=True)
            try:
                self.assertTrue(h.prepare()['supported'])
                self.assertEqual(h.run()['status'],'no_sources')
                self.assertEqual([c[0] for c in h.calls],['writer','reviewer'])
                self.assertIn('First check the complete QUESTION independently',h.calls[-1][1][0]['content'])
            finally:h.close()

    def test_verified_replacement_does_not_reintroduce_removed_later_phase_note(self):
        class Replacement(ContextFixture):
            def reviewer(self,messages,**kw):
                self.calls.append(('reviewer',deepcopy(messages),deepcopy(kw)))
                self.review_packets.append(messages[1]['content'])
                block=messages[1]['content']
                current=block.split('CURRENT_ANSWER:\n',1)[1].split('\n\nSOURCES:',1)[0]
                ids=[next(cid for cid in re.findall(r'\[([^\]\n]+)\]',block)
                    if cid.startswith('step:install'+str(i)+':')) for i in (1,2,3)]
                return self.reply({'outcome':'rewrite','reply_mode':'replace','edits':[],
                    'answer':'The maximum permitted cartridge mass is 40 kg. Use trained personnel with the cycle stopped and energy isolated. Wear gloves and goggles and keep STOP pressed. Verify straight unobstructed entry; do not enter with the feeder enabled. The low-speed test belongs after restoration, not to the initial checks.',
                    'reason':'Remove a later-phase observation from initial checks.',
                    'covered_facets':list(self.decision.required_facets),'missing_facets':[],
                    'covered_answer_types':list(set(self.decision.required_answer_types)|{'state_sequence'}),
                    'missing_answer_types':[],'expected_list_items':[],'covered_list_items':[],
                    'missing_list_items':[],'enumeration_requested':False,'citation_ids':ids,
                    'operation_boundary':{'complete':True,'checked_blocks':[
                        b['block_id'] for b in procedure_review.block_layout(current)['blocks']],'changes':[]}},
                    'offline-reviewer',messages,kw)
        h=Replacement(self,writer_points=[(0,'The maximum permitted cartridge mass is 40 kg. Use trained personnel and isolate energy.'),
            (3,'Check Ready before insertion.')])
        try:
            self.assertTrue(h.prepare()['supported'])
            result=h.run()
            self.assertEqual(result['status'],'answered')
            self.assertNotIn('Ready',result['answer'])
            self.assertFalse(any(c['bubble_document_id']=='step:install4' for c in result['citations']))
            self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
        finally:h.close()

    def test_owned_family_does_not_fabricate_numeric_interface_or_order_availability(self):
        for missing in ('numeric','interface','order'):
            rows=deepcopy(owned.catalog()[:1])
            raw=router()
            if missing!='numeric':rows[0]['steps'][0]['description']+='\nMaximum permitted cartridge mass: 40 kg.'
            if missing=='interface':
                raw['required_answer_types'].append('interface_locations')
                raw['facet_queries'].append(dict(facet='screen and menu path of the calibration display',
                    answer_type='interface_locations',must_cover=True,dense_queries=['calibration display menu path'],
                    lexical_queries=['calibration display menu path'],exact_terms=[],preferred_source_types=['procedure','step']))
            if missing=='order':rows[0]['steps']=rows[0]['steps'][:1]
            h=ContextFixture(self,rows=rows,raw=raw)
            try:
                prepared=h.prepare()
                self.assertFalse(prepared['supported'])
                self.assertEqual(h.calls,[])
                if missing=='numeric':self.assertEqual(prepared['retrieval']['assistant_core_decision']['numeric_viable_count'],0)
                if missing=='interface':self.assertEqual(prepared['retrieval']['assistant_core_decision']['interface_viable_count'],0)
                if missing=='order':self.assertFalse(h.runtime.preserve_procedure_context)
            finally:h.close()

    def test_real_core_binding_snapshots_both_consumers_before_preparation_and_reuses_them(self):
        h=ContextFixture(self)
        try:
            h.owner._response_flow.close();h.owner._response_flow=None
            observations=[]
            def execution_factory():
                observations.append(('execution',h.owner._prepared))
                er=h.owner._execution_factory()
                self.assertFalse(er.preserve_procedure_context)
                def review(messages,**kw):
                    h.generated=h.owner._generated_response
                    return h.reviewer(messages,**kw)
                return replace(er,_v13_json_models=review,
                    _v13_current_budget=lambda:self.m._assistant_core_new_budget('ask'))
            def validation_factory():
                observations.append(('validation',h.owner._prepared))
                return replace(h.owner._validation_factory(),_v13_current_budget=lambda:None)
            hooks=replace(h.owner.core.hooks,retrieve_neutral=lambda request:deepcopy(h.prepared),
                route_semantically=lambda request,data:router(),refine_retrieval=None,
                prepare_evidence=h.owner.prepare,build_no_evidence=h.owner._build_empty_response)
            result=run_core_request(h.request,core=AssistantCoreV2(hooks),
                runtimes=replace(h.owner.runtimes,execution=execution_factory,validation=validation_factory),
                evidence=AskRequestEvidence(session=h.session,authorize=h.owner.authorize,
                    select=h.owner.select,finalize=h.owner._finalize_core))
            self.assertEqual(observations,[('execution',False),('validation',False)])
            self.assertEqual(result['status'],'answered')
            self.assertEqual([c[0] for c in h.calls],['writer','reviewer'])
            self.assertTrue(h.owner._response_flow.completed)
            self.assertTrue(h.owner._response_flow.execution.preserve_procedure_context)
        finally:h.close()

    def test_revoked_parent_and_mutated_seed_fail_before_writer(self):
        for violation in ('revoke', 'mutate'):
            h = ContextFixture(self)
            try:
                if violation == 'revoke':
                    h.allowance = frozenset(s for s in h.allowance if s.source_id != 'install')
                else:
                    h.prepared['citations'][0]['chunk_full'] += '\nwrong source body'
                with self.assertRaises(EvidenceContractError):h.prepare()
                self.assertEqual(h.calls, [])
            finally:h.close()

    def test_pure_value_nonprocedure_degraded_and_external_modes_do_not_acquire(self):
        variants = [dict(request_kind=k) for k in ('factual','unsafe_request','out_of_scope','ambiguous')]
        variants += [dict(effective_mode=k) for k in ('root_cause','smart_diagnostic')]
        variants += [dict(degraded=True), dict(required_answer_types=('numeric_value',), facet_queries=()),
                     dict(source_type_policy='require',preferred_source_types=('document',))]
        for changes in variants:
            with self.subTest(changes=changes):
                h = ContextFixture(self)
                try:
                    op = task_generation.TaskGenerationEvidence(request=h.request,session=h.session,
                        readers=h.owner.readers,authorize=h.owner.authorize,invoke=h.owner.invoke,
                        runtime=h.owner.generation,tasks=h.owner.tasks)
                    out, _, recovered = op.prepare_family_context(h.prepared,h.owner._selection,
                        replace(h.decision,**changes))
                    self.assertFalse(recovered)
                    self.assertEqual(h.sql,[])
                    self.assertIs(out,h.prepared)
                finally:h.close()

    def test_payload_metadata_cannot_activate_owned_context(self):
        h = ContextFixture(self)
        try:
            h.prepared['assistant_core_contract']['preserve_procedure_context'] = True
            h.prepared['plan']['preserve_procedure_context'] = True
            h.owner._procedure_family_context = False
            self.assertFalse(h.runtime.preserve_procedure_context)
        finally:h.close()


if __name__ == '__main__':unittest.main()
