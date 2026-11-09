"""One existing semantic router selects an owned operation despite noisy Steps.

SQL rows, typed initial-acquisition snapshots and provider replies are fixtures.
Real Core binding, catalog readers, refinement policy, canonical expansion,
writer/reviewer/final admission run; no network/provider spending is permitted.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from assistant_core_v2 import AssistantCoreV2
from machinemind.ask import execution,request_completion,task_generation
from machinemind.ask.request_binding import AskRequestEvidence,run_core_request
from machinemind.evidence.contracts import EvidenceContractError,SourceIdentity,SourceScope,SourceType,SourceFormat,ScopeLevel
from machinemind.retrieval.ask_composition import AskSelection
from machinemind.retrieval.page_evidence import _PageReadCapture
import test_ask_conditions_family_admission as owned
import test_ask_procedure_context_recovery as context


def catalog():
    rows=deepcopy(owned.catalog())
    rows[0]['steps'][0]['description']+='\nMaximum permitted cartridge mass: 40 kg.'
    rows.append(dict(procedure_id='lubricate',title='Daily lubrication and beginning-of-shift checks',
        short_description='Perform the daily lubrication checks.',steps=[dict(step_id='lubricate1',step_number=1,
            title='Stop and check lubrication',description='Stop the cycle and check the lubrication indicator. Qualified personnel only.')]))
    return rows


class RouterFixture(context.ContextFixture):
    def __init__(self,test,*,choice='install',revoke=False,manual=None,**kw):
        self.choice,self.revoke_catalog=choice,revoke
        self.router_raw=deepcopy(kw.get('raw') or context.router())
        self.refine_calls=[]
        super().__init__(test,rows=kw.pop('rows',catalog()),**kw)
        self.owner._response_flow.close();self.owner._response_flow=None
        self._install_seed_pool()
        self.manual=manual
        self.manual_pack=None
        if manual is not None:self._register_manual(manual)

    def _install_seed_pool(self):
        limits=owned.limits_fixture(1,1)[0].limits
        capture=_PageReadCapture(self.scope,limits,'structured_direct',snippet_chars=800)
        capture.expect(len(self.rows),6000)
        capture.capture([(*row,self.request.company_id,self.request.machine_id) for row in self.rows])
        rows=[dict(bubble_document_id=key,page_from=1,page_to=1,citation_id=key+':seed',
            source_type='procedure' if key.startswith('procedure:') else 'step',chunk_full=body,snippet=body,
            similarity=.46,semantic_similarity=.46) for key,_,_,body in self.rows]
        registered=self.session.register(request=self.request,receipt=capture.finish(rows),current_allowed_sources=self.allowance)
        records=[dict(item.record) for item in self.session.records(request=self.request,handles=registered.records,current_allowed_sources=self.allowance)]
        self.prepared={**self.prepared,'candidates':records,'citations':records}
        self.owner._selection=(AskSelection('candidates',registered.records),AskSelection('citations',registered.records))

    def _register_manual(self,text):
        source=SourceIdentity(SourceScope(self.request.company_id,ScopeLevel.MACHINE,self.request.machine_id),
            SourceType.DOCUMENT,'manual-only',SourceFormat.UNKNOWN)
        self.allowance|=frozenset((source,))
        capture=_PageReadCapture(self.scope,owned.limits_fixture(1,1)[0].limits,'structured_direct',snippet_chars=800)
        capture.expect(1,6000)
        capture.capture([('manual-only',self.request.machine_id,1,text,self.request.company_id,self.request.machine_id)])
        row=dict(bubble_document_id='manual-only',citation_id='manual-only:p1',page_from=1,page_to=1,
            source_type='document',snippet=text,chunk_full=text,similarity=.8,semantic_similarity=.8)
        registered=self.session.register(request=self.request,receipt=capture.finish([row]),current_allowed_sources=self.allowance)
        records=[dict(item.record) for item in self.session.records(request=self.request,handles=registered.records,current_allowed_sources=self.allowance)]
        self.manual_pack=(dict(candidates=records,citations=records,plan={},metrics={}),
            (AskSelection('candidates',registered.records),AskSelection('citations',registered.records)))

    def router(self,messages,**kw):
        self.calls.append(('router',deepcopy(messages),deepcopy(kw)))
        self.test.assertIn('PROCEDURE_OPERATION_CATALOG:',messages[1]['content'])
        raw=deepcopy(self.router_raw)
        choices=[self.choice] if isinstance(self.choice,str) else self.choice or []
        raw['relevant_evidence_ids']=[next((cid for cid in self.owner._procedure_catalog
            if cid.startswith('procedure:'+choice+':')),choice) for choice in choices]
        if self.revoke_catalog:self.allowance=frozenset(s for s in self.allowance if s.source_id!='install')
        return raw,'offline-router'

    def provider(self,messages,**kw):
        if self.owner._selected_catalog_parent is None:
            self.calls.append(('writer',deepcopy(messages),deepcopy(kw)))
            return self.reply({'answer_status':'no_sources','grounded_points':[]},'offline-writer',messages,kw)
        reply,model=super().provider(messages,**kw)
        if self.manual is not None and reply.get('answer_status')=='answered':
            reply['grounded_points'].append({'text':'The calibration voltage is 24 V. The calibration display menu path is Service > Calibration > Voltage.',
                'citation_ids':['manual-only:p1']})
        return reply,model

    def reviewer(self,messages,**kw):
        if self.owner._selected_catalog_parent is None:
            self.calls.append(('reviewer',deepcopy(messages),deepcopy(kw)))
            return {'outcome':'no_sources','answer':'','reason':'No supported fixture replacement.',
                'covered_facets':[],'missing_facets':list(self.decision.required_facets),
                'covered_answer_types':[],'missing_answer_types':list(self.decision.required_answer_types),
                'expected_list_items':[],'covered_list_items':[],'missing_list_items':[],
                'enumeration_requested':False,'citation_ids':[]},'offline-reviewer'
        return super().reviewer(messages,**kw)

    def initial(self,**kw):
        self.refine_calls.append(deepcopy(kw))
        if self.manual_pack is not None:return deepcopy(self.manual_pack[0]),self.manual_pack[1]
        # A typed acquisition fixture supplies the observed noisy initial pool;
        # the real bounded facet-query planner/merge/lineage policy remains live.
        return deepcopy(self.prepared),self.seed_selections

    def run_core(self):
        self.seed_selections=self.owner._selection
        neutral=deepcopy(self.prepared)
        original=lambda request,data:self.m._assistant_core_router_call(request,data)
        self.owner.core=AssistantCoreV2(replace(self.owner.core.hooks,route_semantically=original))
        self.owner._initial=self.initial
        def efactory():
            self.test.assertFalse(self.owner._prepared)
            def review(messages,**kw):
                self.generated=self.owner._generated_response
                return self.reviewer(messages,**kw)
            return replace(self.owner._execution_factory(),_v13_json_models=review,
                _v13_current_budget=lambda:self.m._assistant_core_new_budget('ask'))
        hooks=replace(self.owner.core.hooks,retrieve_neutral=lambda request:self.owner._add_procedure_catalog(request,deepcopy(neutral)),
            route_semantically=self.owner.route,refine_retrieval=self.owner.refine,prepare_evidence=self.owner.prepare,
            build_no_evidence=self.owner._build_empty_response)
        completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=self.request.company_id),started=time.monotonic())
        try:
            with patch.object(self.m,'_v13_json_models',self.router):
                return run_core_request(self.request,core=AssistantCoreV2(hooks),
                    runtimes=replace(self.owner.runtimes,execution=efactory,
                        validation=lambda:replace(self.owner._validation_factory(),_v13_current_budget=lambda:None)),
                    evidence=AskRequestEvidence(session=self.session,authorize=self.owner.authorize,
                        select=self.owner.select,finalize=self.owner._finalize_core))
        finally:completion.close()


class AskRouterFamilyContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
    @classmethod
    def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()

    def test_one_existing_router_selects_operation_from_competing_families_and_keeps_refinement(self):
        h=RouterFixture(self)
        try:
            result=h.run_core()
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual([c[0] for c in h.calls],['router','writer','reviewer'])
            self.assertTrue(h.refine_calls)
            self.assertTrue(h.owner._response_flow.completed)
            for _,_,_,body in h.rows[:5]:
                self.assertIn(body,h.writer_packets[0]);self.assertIn(body,h.review_packets[0])
            self.assertNotIn('Rilasciare emergenze',h.writer_packets[0])
            self.assertNotIn('lubrication indicator',h.writer_packets[0])
        finally:h.close()

    def test_null_ambiguous_and_forged_parent_choices_keep_strict_fallback(self):
        for choice in (None,['install','startup'],'procedure:foreign:p1'):
            with self.subTest(choice=choice):
                h=RouterFixture(self,choice=choice)
                try:
                    result=h.run_core()
                    self.assertEqual(result['status'],'no_sources')
                    self.assertIsNone(h.owner._selected_catalog_parent)
                    self.assertEqual(h.calls[0][0],'router')
                    self.assertNotIn('selector',[c[0] for c in h.calls])
                    self.assertFalse(h.owner._procedure_family_context)
                    self.assertTrue(h.refine_calls)
                finally:h.close()

    def test_revoked_catalog_parent_fails_before_refinement_writer(self):
        h=RouterFixture(self,revoke=True)
        try:
            with self.assertRaises(EvidenceContractError):h.run_core()
            self.assertEqual([c[0] for c in h.calls],['router'])
            self.assertEqual(h.refine_calls,[])
        finally:h.close()

    def test_mandatory_manual_only_value_and_menu_are_refined_and_retained(self):
        raw=context.router()
        raw['required_answer_types'].append('interface_locations')
        raw['facet_queries'].append(dict(facet='calibration voltage value and display menu path',
            answer_type='interface_locations',must_cover=True,dense_queries=['calibration voltage display menu path'],
            lexical_queries=['calibration voltage display menu path'],exact_terms=[],preferred_source_types=['document']))
        raw['required_facets'].append(raw['facet_queries'][-1]['facet'])
        h=RouterFixture(self,raw=raw,query='Before inserting the cartridge, give its maximum mass, checks and precautions. Also give the calibration voltage and calibration display menu path.',
            manual='Calibration voltage: 24 V. Calibration display menu path: Service > Calibration > Voltage.')
        try:
            result=h.run_core()
            self.assertEqual(result['status'],'answered')
            self.assertTrue(h.refine_calls)
            self.assertIn(h.manual,h.writer_packets[0])
            self.assertIn(h.manual,h.review_packets[0])
            self.assertTrue(h.owner._procedure_family_context)
            self.assertIn('24 V',result['answer'])
            self.assertIn('Service > Calibration > Voltage',result['answer'])
            self.assertTrue(any(c['bubble_document_id']=='manual-only' for c in result['citations']))
        finally:h.close()

    def test_narrow_and_non_ask_scopes_do_not_acquire_catalog(self):
        h=RouterFixture(self)
        try:
            for request in (replace(h.request,narrow_scope=True),replace(h.request,requested_mode='root_cause'),
                            replace(h.request,requested_mode='smart_diagnostic')):
                self.assertIs(h.owner._add_procedure_catalog(request,h.prepared),h.prepared)
                self.assertEqual(h.sql,[])
                self.assertEqual(h.owner._procedure_catalog,{})
        finally:h.close()

    def test_explicit_manual_only_policy_cannot_enable_selected_family_context(self):
        h=RouterFixture(self)
        try:
            h.owner._add_procedure_catalog(h.request,h.prepared)
            parent=next(handle for cid,handle in h.owner._procedure_catalog.items() if cid.startswith('procedure:install:'))
            operation=task_generation.TaskGenerationEvidence(request=h.request,session=h.session,readers=h.owner.readers,
                authorize=h.owner.authorize,invoke=h.owner.invoke,runtime=h.owner.generation,tasks=h.owner.tasks)
            current=h.owner._values(h.owner._selection[0].records)
            data={**h.prepared,'candidates':current,'citations':current}
            before=len(h.sql)
            _,_,recovered=operation.prepare_family_context(data,h.owner._selection,
                replace(h.decision,source_type_policy='require',preferred_source_types=('document',)),selected_parent_handle=parent)
            self.assertFalse(recovered)
            self.assertEqual(len(h.sql),before)
        finally:h.close()

    def test_other_request_handle_and_catalog_projection_overflow_fail_closed(self):
        first,second=RouterFixture(self),RouterFixture(self)
        try:
            second.owner._add_procedure_catalog(second.request,second.prepared)
            foreign=next(iter(second.owner._procedure_catalog.values()))
            operation=task_generation.TaskGenerationEvidence(request=first.request,session=first.session,readers=first.owner.readers,
                authorize=first.owner.authorize,invoke=first.owner.invoke,runtime=first.owner.generation,tasks=first.owner.tasks)
            with self.assertRaises(EvidenceContractError):operation.prepare_family_context(first.prepared,
                first.owner._selection,first.decision,selected_parent_handle=foreign)
            row=second.rows[0];second.rows[0]=(row[0],row[1],row[2],'x'*100000)
            with self.assertRaises(EvidenceContractError):second.owner._add_procedure_catalog(second.request,second.prepared)
            self.assertEqual(first.calls+second.calls,[])
        finally:first.close();second.close()

    def test_off_router_ignores_catalog_display_flag_and_preserves_protocol(self):
        h=RouterFixture(self)
        try:
            h.owner._add_procedure_catalog(h.request,h.prepared)
            data={**h.prepared,'candidates':h.owner._values(h.owner._selection[0].records)}
            captured=[]
            def router(messages,**kw):captured.append(messages);return context.router(),'offline-router'
            with patch.object(self.m,'_v13_json_models',router):self.m._assistant_core_router_call(h.request,data)
            self.assertEqual(len(captured),1)
            self.assertNotIn('PROCEDURE_OPERATION_CATALOG:',captured[0][1]['content'])
            self.assertNotIn('possibly incomplete, admitted',captured[0][0]['content'])
        finally:h.close()

    def test_wrong_but_authorized_parent_keeps_only_its_family_and_review_rejects(self):
        rows=catalog()
        rows[1]['steps'].append(dict(step_id='startup5',step_number=5,title='Startup test',
            description='Use a test load of 10 kg only for the startup test. Stop before adjustment.'))
        h=RouterFixture(self,rows=rows,choice='startup',chosen='startup',reject_review=True,
            writer_points=[(1,'The startup test load is 10 kg. Stop before adjustment.')])
        try:
            result=h.run_core()
            self.assertEqual(result['status'],'no_sources')
            self.assertEqual([c[0] for c in h.calls],['router','writer','reviewer'])
            self.assertIn('startup test',h.writer_packets[0])
            self.assertNotIn('SOURCE_TYPE: procedure\nTITLE: Inserimento cartuccia',h.writer_packets[0])
            self.assertIn('First check the complete QUESTION independently',h.calls[-1][1][0]['content'])
        finally:h.close()

    def test_catalog_count_and_character_bounds_filter_unemitted_model_ids(self):
        h=RouterFixture(self)
        completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=h.request.company_id),started=time.monotonic())
        try:
            captured=[]
            data=dict(candidates=[dict(citation_id='catalog-'+str(i),source_type='procedure',
                chunk_full=('Operation description '+str(i)+' ')*100,assistant_core_parent_catalog_entry=True)
                for i in range(20)])
            def model(messages,**kw):
                captured.append(messages)
                raw=context.router();raw['relevant_evidence_ids']=['catalog-19'];return raw,'offline-router'
            with patch.object(self.m,'_v13_gate_candidate_block',return_value=('',[])), \
                 patch.object(self.m,'_v13_json_models',model):
                response=self.m._assistant_core_router_call(h.request,data)
            block=captured[0][1]['content'].split('PROCEDURE_OPERATION_CATALOG:\n',1)[1].split('\n\nINDEXED_EVIDENCE:',1)[0]
            self.assertLessEqual(len(block),8000)
            self.assertLessEqual(block.count('[catalog-'),12)
            self.assertNotIn('[catalog-19]',block)
            self.assertEqual(response['relevant_evidence_ids'],[])
        finally:completion.close();h.close()

    def test_display_metadata_without_private_catalog_handle_cannot_select_family(self):
        h=RouterFixture(self)
        completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id=h.request.company_id),started=time.monotonic())
        try:
            original=h.owner._selection[0].records
            view=deepcopy(h.prepared['candidates'][0]);view['assistant_core_parent_catalog_entry']=True
            derived=h.session.derive_batch(request=h.request,views=(((original[0],),view),),
                operation='offline-display-metadata-only',current_allowed_sources=h.allowance)[0]
            handles=(derived,)+original[1:]
            values=h.owner._values(handles);data={**h.prepared,'candidates':values,'citations':values}
            h.owner._selection=(AskSelection('candidates',handles),AskSelection('citations',handles))
            h.owner.core=AssistantCoreV2(replace(h.owner.core.hooks,route_semantically=self.m._assistant_core_router_call))
            def model(messages,**kw):
                raw=context.router();raw['relevant_evidence_ids']=[view['citation_id']];return raw,'offline-router'
            with patch.object(self.m,'_v13_json_models',model):h.owner.route(h.request,data)
            self.assertIsNone(h.owner._selected_catalog_parent)
            self.assertEqual(h.owner._procedure_catalog,{})
            self.assertEqual(h.owner._family_context_handles,())
        finally:completion.close();h.close()


if __name__=='__main__':unittest.main()
