"""Owned mixed-ASK family recovery. SQL/provider replies are offline fixtures.

The real preparation, typed readers, writer, independent review and final source
binding run. These tests prove contracts/source flow, not provider answer quality.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import hashlib
import json
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

    def test_explicit_owned_operation_is_independent_of_numeric_or_check_output_shape(self):
        for changes in (dict(request_kind='factual',information_task='numeric_specification',
                             required_answer_types=('numeric_value',),facet_queries=()),
                        dict(request_kind='factual',information_task='document_explanation',
                             required_answer_types=('checklist','safety_conditions'),facet_queries=())):
            with self.subTest(changes=changes):
                h=ContextFixture(self)
                try:
                    h.prepared=h.owner._add_procedure_catalog(h.request,h.prepared)
                    parent=next(handle for cid,handle in h.owner._procedure_catalog.items()
                        if cid.startswith('procedure:install:'))
                    h.decision=replace(h.decision,**changes)
                    before=deepcopy(h.decision)
                    op=task_generation.TaskGenerationEvidence(request=h.request,session=h.session,
                        readers=h.owner.readers,authorize=h.owner.authorize,invoke=h.owner.invoke,
                        runtime=h.owner.generation,tasks=h.owner.tasks)
                    out,selections,recovered=op.prepare_family_context(h.prepared,h.owner._selection,h.decision,
                        selected_parent_handle=parent)
                    self.assertTrue(recovered)
                    self.assertEqual(h.decision,before)
                    material='\n'.join(row['chunk_full'] for row in out['candidates'])
                    for key,_,_,body in h.rows:
                        if key.startswith(('procedure:install','step:install')):
                            self.assertIn(body,material)
                    self.assertEqual(h.calls,[])
                finally:h.close()

    def test_explicit_owner_choice_does_not_override_refusal_or_full_procedure_lane(self):
        variants=[dict(request_kind=k) for k in ('unsafe_request','out_of_scope','ambiguous')]
        variants += [dict(effective_mode=k) for k in ('root_cause','smart_diagnostic')]
        variants += [dict(degraded=True),dict(information_task='procedure_full'),
            dict(source_type_policy='require',preferred_source_types=('document',))]
        for changes in variants:
            with self.subTest(changes=changes):
                h=ContextFixture(self)
                try:
                    data=h.owner._add_procedure_catalog(h.request,h.prepared)
                    parent=next(iter(h.owner._procedure_catalog.values()))
                    before=len(h.sql)
                    op=task_generation.TaskGenerationEvidence(request=h.request,session=h.session,
                        readers=h.owner.readers,authorize=h.owner.authorize,invoke=h.owner.invoke,
                        runtime=h.owner.generation,tasks=h.owner.tasks)
                    out,_,recovered=op.prepare_family_context(data,h.owner._selection,
                        replace(h.decision,**changes),selected_parent_handle=parent)
                    self.assertIs(out,data);self.assertFalse(recovered)
                    self.assertEqual(len(h.sql),before);self.assertEqual(h.calls,[])
                finally:h.close()

    def test_real_r5_mixed_steps_recover_all_ten_current_steps_without_foreign_pressure(self):
        # Private local QA evidence is read in place, never copied into the repo.
        # A standalone checkout without that corpus explicitly skips this replay.
        import test_ask_router_family_context as route_fixture
        from machinemind.retrieval.chunk_evidence import build_chunk_read,_source_from_columns
        work=Path(__file__).resolve().parents[2]
        inputs=[(work/'phase6-ask-correction-r5-20261109/live-client/receipts/R5-A-ASK-02.json',
                 'af8298183586f73146f45573af4a2b00928c33bafddae2d793fd9758514b0b3e'),
                (work/'phase6-ask-source-audit-20261109/current-aurelis-proc002-and-catalog.json',
                 '89620322db9e553fe738fba1e2a17b81c9dd6c1d361936ee2e996d5d5e1b8ee5'),
                (work/'phase6-multicompany-20261109/CASE-PREREGISTRATION.json',
                 'b8f93411fde49079f8d0214bda50c7aeb1a70d04934b2b905e1655b71888f85a')]
        if not all(path.exists() for path,_ in inputs):self.skipTest('Private pinned R5/source corpus not present')
        values=[]
        for path,digest in inputs:
            blob=path.read_bytes();self.assertEqual(hashlib.sha256(blob).hexdigest(),digest)
            values.append(json.loads(blob.decode('utf-8-sig')))
        receipt,source,prereg=values
        meta=receipt['response']['meta'];raw=deepcopy(meta['assistant_core_v2'])
        raw['relevant_evidence_ids']=[]  # Recorded normalized decision has no optional selections.
        candidates=deepcopy(meta['assistant_core_evidence_admission']['deterministic_signals']['candidates'])
        self.assertEqual(len(candidates),24)
        self.assertEqual(sum(c['source_type']=='step' for c in candidates),17)
        self.assertEqual(raw['information_task'],'numeric_specification')
        proc=source['procedure'];pid=proc['procedure_id']
        catalog=[]
        for entry in source['procedures_catalog']:
            item=deepcopy(proc if entry['procedure_id']==pid else entry)
            item.setdefault('steps',[]);catalog.append(item)
        # The base harness's initial one-Step receipt must really be a Step.
        catalog.sort(key=lambda entry:entry['procedure_id']==pid)
        query=next(c['prompt'] for c in prereg['cases'] if c['case_key']=='UI-A-ASK-02')
        points=[(1,'The maximum permitted coil mass is 1000 kg. Check material identification, weight, diameter and unwinding direction.'),
            (0,'Before loading and threading, stop the cycle, apply the company energy-isolation procedure and use trained personnel.'),
            (2,'Use approved lifting equipment and authorized personnel. Do not stand under suspended loads.'),
            (3,'Check full engagement of power and control connectors, cable integrity and absence of pinching. Work on connectors only with supplies isolated.'),
            (4,'Wear cut-resistant gloves and eye protection and keep STOP pressed during strip handling. STOP does not replace energy isolation.'),
            (5,'Do not enter the work area with the decoiler enabled.'),
            (9,'The later permitted-mode low-speed unwinding test is separate from initial checks; stop if the strip tensions or whips.')]
        class ActualR5(route_fixture.RouterFixture):
            def reviewer(self,messages,**kw):
                reply,model=super().reviewer(messages,**kw)
                reply['covered_answer_types']=list(set(reply['covered_answer_types']) |
                    {facet.answer_type for facet in self.decision.facet_queries if facet.must_cover})
                return reply,model
        h=ActualR5(self,raw=raw,query=query,rows=catalog,chosen=pid,choice=pid,
            mass='1000 kg',writer_points=points)
        try:
            sql=[(c['bubble_document_id'],c['chunk_index'],c['page_from'],c['page_to'],c['snippet'],
                c['chunk_full'],c['similarity'],c.get('embedding_list',[]),True,
                h.request.company_id,h.request.machine_id) for c in candidates]
            h.allowance|=frozenset(_source_from_columns(h.request.company_id,h.request.machine_id,c['bubble_document_id']) for c in candidates)
            typed=build_chunk_read(scope=h.scope,kind='dense',rows=sql,candidates=candidates,limits=owned.limits_fixture(1,1)[0].limits)
            registered=h.session.register(request=h.request,receipt=typed,current_allowed_sources=h.allowance)
            exact=[dict(item.record) for item in h.session.records(request=h.request,
                handles=registered.records,current_allowed_sources=h.allowance)]
            h.prepared={**h.prepared,'candidates':exact,'citations':exact}
            h.owner._selection=(AskSelection('candidates',registered.records),AskSelection('citations',registered.records))
            self.assertFalse(any(proc['steps'][2]['description'] in c.get('chunk_full','') for c in candidates))
            foreign=next(c['chunk_full'] for c in candidates if '5' in c['chunk_full'] and 'bar' in c['chunk_full']
                and 'PROC-001' in c['chunk_full'])
            result=h.run_core()
            self.assertEqual(result['status'],'answered')
            self.assertEqual([c[0] for c in h.calls],['router','writer','reviewer'])
            for _,_,_,body in h.rows:
                if body.startswith('SOURCE_TYPE: step'):
                    self.assertIn(body,h.writer_packets[0]);self.assertIn(body,h.review_packets[0])
            for packet in (h.writer_packets[0],h.review_packets[0]):
                self.assertIn('Non sostare sotto carichi sospesi.',packet)
                self.assertNotIn(foreign,packet)
            self.assertIn('Do not stand under suspended loads',result['answer'])
            self.assertNotRegex(result['answer'],r'5\s*[–-]\s*6\s*bar')
            self.assertEqual(h.owner._generated_response['meta']['generation_evidence']['source_sha256'],
                h.owner._response_flow._review_packet.diagnostic(reused=False)['source_sha256'])
            self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
        finally:h.close()



    def test_actual_r7_required_parent_type_keeps_complete_family_and_manual_support(self):
        # The actual private pp was not recorded: force a VALID owned choice only
        # for this counterfactual. All other normalized values and 24 rows are real.
        import test_ask_router_family_context as route_fixture
        from machinemind.retrieval.chunk_evidence import build_chunk_read,_source_from_columns
        work=Path(__file__).resolve().parents[2]
        inputs=[(work/'phase6-ask-correction-r7-20261109/live-client/receipts/R7-A-ASK-02.json',
                 '4a4c0ea4f269f472fdafa849bf443b681c451d50380eb5996142c108e3ab17ef'),
                (work/'phase6-ask-source-audit-20261109/current-aurelis-proc002-and-catalog.json',
                 '89620322db9e553fe738fba1e2a17b81c9dd6c1d361936ee2e996d5d5e1b8ee5'),
                (work/'phase6-multicompany-20261109/CASE-PREREGISTRATION.json',
                 'b8f93411fde49079f8d0214bda50c7aeb1a70d04934b2b905e1655b71888f85a')]
        if not all(p.exists()for p,_ in inputs):self.skipTest('Private pinned R7/source corpus not present')
        values=[]
        for p,digest in inputs:
            blob=p.read_bytes();self.assertEqual(hashlib.sha256(blob).hexdigest(),digest)
            values.append(json.loads(blob.decode('utf-8-sig')))
        receipt,source,prereg=values;meta=receipt['response']['meta']
        actual=deepcopy(meta['assistant_core_v2'])
        actual['relevant_evidence_ids']=list(meta['assistant_core_evidence_admission']['router_relevant_evidence_ids'])
        self.assertEqual(actual['source_type_policy'],'require')
        self.assertEqual(actual['preferred_source_types'],['procedure','document','md_photo'])
        candidates=deepcopy(meta['assistant_core_evidence_admission']['deterministic_signals']['candidates'])
        self.assertEqual(len(candidates),24)
        self.assertEqual(sum(c['source_type']=='step'for c in candidates),0)
        proc=source['procedure'];pid=proc['procedure_id'];catalog=[]
        for entry in source['procedures_catalog']:
            item=deepcopy(proc if entry['procedure_id']==pid else entry)
            item.setdefault('steps',[]);catalog.append(item)
        catalog.sort(key=lambda e:e['procedure_id']==pid)
        query=next(c['prompt']for c in prereg['cases']if c['case_key']=='UI-A-ASK-02')
        points=[(1,'The maximum permitted coil mass is 1000 kg. Check material identification, weight, diameter and unwinding direction.'),
            (0,'Before loading and threading, stop the cycle, apply the company energy-isolation procedure and use trained personnel.'),
            (2,'Use approved lifting equipment and authorized personnel. Do not stand under suspended loads.'),
            (3,'Check full engagement of power and control connectors, cable integrity and absence of pinching. Work on connectors only with supplies isolated.'),
            (4,'Wear cut-resistant gloves and eye protection and keep STOP pressed during strip handling. STOP does not replace energy isolation.'),
            (5,'Do not enter the work area with the decoiler enabled.'),
            (9,'The later permitted-mode low-speed unwinding test is separate from initial checks; stop if the strip tensions or whips.')]
        class ActualR7(route_fixture.RouterFixture):
            def reviewer(self,messages,**kw):
                reply,model=super().reviewer(messages,**kw)
                if not self.reject_review:
                    reply['covered_answer_types']=list(set(reply['covered_answer_types']) |
                        {f.answer_type for f in self.decision.facet_queries if f.must_cover})
                else:
                    reply['missing_answer_types']=list(self.decision.required_answer_types)
                return reply,model
        for manual,reject in ((False,False),(True,False),(False,True)):
            with self.subTest(manual_only_support=manual,review_rejects=reject):
                raw=deepcopy(actual);question=query;support=None
                if manual:
                    support='Calibration voltage: 24 V. Calibration display menu path: Service > Calibration > Voltage.'
                    question+=' Also give the calibration voltage and calibration display menu path.'
                    raw['required_answer_types'].append('interface_locations')
                    facet=dict(facet='calibration voltage and display menu path',answer_type='interface_locations',must_cover=True,
                        dense_queries=['calibration voltage display menu'],lexical_queries=['calibration voltage display menu'],exact_terms=[],preferred_source_types=['document'])
                    raw['facet_queries'].append(facet);raw['required_facets'].append(facet['facet'])
                h=ActualR7(self,raw=raw,query=question,rows=deepcopy(catalog),chosen=pid,choice=pid,
                    relevant_choices=tuple(actual['relevant_evidence_ids']),mass='1000 kg',writer_points=points,
                    manual=support,reject_review=reject)
                try:
                    h.decision=normalize_decision(h.request,raw)
                    sql=[(c['bubble_document_id'],c['chunk_index'],c['page_from'],c['page_to'],c['snippet'],c['chunk_full'],
                        c['similarity'],c.get('embedding_list',[]),True,h.request.company_id,h.request.machine_id)for c in candidates]
                    h.allowance|=frozenset(_source_from_columns(h.request.company_id,h.request.machine_id,c['bubble_document_id'])for c in candidates)
                    typed=build_chunk_read(scope=h.scope,kind='dense',rows=sql,candidates=candidates,limits=owned.limits_fixture(1,1)[0].limits)
                    registered=h.session.register(request=h.request,receipt=typed,current_allowed_sources=h.allowance)
                    exact=[dict(item.record)for item in h.session.records(request=h.request,handles=registered.records,current_allowed_sources=h.allowance)]
                    h.prepared={**h.prepared,'candidates':exact,'citations':exact}
                    h.owner._selection=(AskSelection('candidates',registered.records),AskSelection('citations',registered.records))
                    result=h.run_core()
                    self.assertEqual(result['status'],'no_sources'if reject else'answered',result)
                    self.assertTrue(h.owner._procedure_family_context)
                    self.assertTrue(h.refine_calls)
                    self.assertEqual([c[0]for c in h.calls],['router','writer','reviewer'])
                    expected=[body for key,_,_,body in h.rows if key=='procedure:'+pid or key.startswith('step:')]
                    self.assertEqual(len(expected),11)
                    for body in expected:
                        self.assertIn(body,h.writer_packets[0]);self.assertIn(body,h.review_packets[0])
                    for packet in (h.writer_packets[0],h.review_packets[0]):
                        self.assertIn('Non sostare sotto carichi sospesi.',packet)
                        self.assertNotIn('SOURCE_TYPE: procedure\nTITLE: Lubrificazione',packet)
                        self.assertNotRegex(packet,r'PROC-001[^\n]*\n[^\n]*5\s*[–-]\s*6\s*bar')
                    if not reject:
                        self.assertIn('Do not stand under suspended loads',result['answer'])
                        self.assertEqual(h.owner._generated_response['meta']['generation_evidence']['source_sha256'],
                            h.owner._response_flow._review_packet.diagnostic(reused=False)['source_sha256'])
                    if manual:
                        self.assertIn(support,h.writer_packets[0]);self.assertIn(support,h.review_packets[0])
                        self.assertIn('24 V',result['answer'])
                        self.assertIn('Service > Calibration > Voltage',result['answer'])
                finally:h.close()

    def test_required_parent_type_exception_does_not_expand_manual_step_or_narrow_scope(self):
        import test_ask_router_family_context as route_fixture
        from unittest.mock import patch
        raw=router();raw.update(request_kind='factual',information_task='numeric_specification',
            source_type_policy='require',preferred_source_types=['procedure','document','md_photo'])
        def operation(h,parent,decision):
            op=task_generation.TaskGenerationEvidence(request=h.request,session=h.session,readers=h.owner.readers,
                authorize=h.owner.authorize,invoke=h.owner.invoke,runtime=h.owner.generation,tasks=h.owner.tasks)
            before=len(h.sql)
            result=op.prepare_family_context(h.prepared,h.owner._selection,decision,selected_parent_handle=parent)
            self.assertFalse(result[2]);self.assertEqual(len(h.sql),before);self.assertEqual(h.calls,[])
        for preferred in (('document',),('step',),('md_photo',),('step','document')):
            h=route_fixture.RouterFixture(self,raw=raw)
            try:
                parent=next(handle for handle,item in zip(h.owner._selection[0].records,
                    h.session.records(request=h.request,handles=h.owner._selection[0].records,current_allowed_sources=h.allowance))
                    if item.context.source.source_type==SourceType.PROCEDURE and item.context.source.source_id=='install')
                operation(h,parent,replace(h.decision,source_type_policy='require',preferred_source_types=preferred))
            finally:h.close()
        # Construct a genuinely narrowed owner BEFORE its session/readers exist.
        factory=owned.condition_fixtures.AskConditionsPipelineTests
        original=factory.request_decision
        def narrow(instance,*args,**kw):
            request,decision=original(instance,*args,**kw)
            return replace(request,narrow_scope=True),decision
        with patch.object(factory,'request_decision',narrow):h=route_fixture.RouterFixture(self,raw=raw)
        try:
            self.assertTrue(h.request.narrow_scope)
            parent=next(handle for handle,item in zip(h.owner._selection[0].records,
                h.session.records(request=h.request,handles=h.owner._selection[0].records,current_allowed_sources=h.allowance))
                if item.context.source.source_type==SourceType.PROCEDURE and item.context.source.source_id=='install')
            operation(h,parent,replace(h.decision,source_type_policy='require',preferred_source_types=('procedure','document')))
        finally:h.close()

    def test_required_parent_type_exception_still_rejects_foreign_or_revoked_handles(self):
        import test_ask_router_family_context as route_fixture
        raw=router();raw.update(request_kind='factual',information_task='numeric_specification',
            source_type_policy='require',preferred_source_types=['procedure','document','md_photo'])
        h,other=route_fixture.RouterFixture(self,raw=raw),route_fixture.RouterFixture(self,raw=raw)
        try:
            foreign=next(handle for handle,item in zip(other.owner._selection[0].records,
                other.session.records(request=other.request,handles=other.owner._selection[0].records,current_allowed_sources=other.allowance))
                if item.context.source.source_type==SourceType.PROCEDURE)
            foreign_value=other.owner._values((foreign,))[0]
            self.assertIn(foreign_value['citation_id'],[c['citation_id']for c in h.prepared['candidates']])
            self.assertIn(foreign_value['bubble_document_id'],[c['bubble_document_id']for c in h.prepared['candidates']])
            op=task_generation.TaskGenerationEvidence(request=h.request,session=h.session,readers=h.owner.readers,
                authorize=h.owner.authorize,invoke=h.owner.invoke,runtime=h.owner.generation,tasks=h.owner.tasks)
            with self.assertRaises(EvidenceContractError):op.prepare_family_context(h.prepared,h.owner._selection,
                replace(h.decision,source_type_policy='require',preferred_source_types=('procedure','document')),selected_parent_handle=foreign)
            self.assertEqual(h.calls+other.calls,[])
        finally:h.close();other.close()
        h=route_fixture.RouterFixture(self,raw=raw)
        try:
            values=h.session.records(request=h.request,handles=h.owner._selection[0].records,current_allowed_sources=h.allowance)
            pairs=list(zip(h.owner._selection[0].records,values))
            foreign=next(handle for handle,item in pairs if item.context.source.source_type==SourceType.STEP
                and item.context.source.source_id.startswith('startup'))
            target=next(dict(item.record)for _,item in pairs if item.context.source.source_type==SourceType.STEP
                and item.context.source.source_id.startswith('install'))
            alias=deepcopy(h.owner._values((foreign,))[0])
            alias.update(citation_id=target['citation_id'],bubble_document_id=target['bubble_document_id'])
            with self.assertRaises(EvidenceContractError):h.session.derive_batch(request=h.request,
                views=(((foreign,),alias),),operation='offline-same-ID-wrong-parent-relabel',current_allowed_sources=h.allowance)
            self.assertEqual(h.calls,[])
        finally:h.close()
        h=route_fixture.RouterFixture(self,raw=raw,revoke=True)
        try:
            with self.assertRaises(EvidenceContractError):h.run_core()
            self.assertEqual([call[0]for call in h.calls],['router'])
            self.assertEqual(h.refine_calls,[])
        finally:h.close()

if __name__ == '__main__':unittest.main()
