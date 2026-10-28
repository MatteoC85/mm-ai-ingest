"""Offline semantic-family selection wiring; provider replies are fixed fixtures.

Exercises the real scoped direct reader, family curation/canonical expansion,
source rendering and final review. No live model quality or authority is asserted.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import task_generation,validation,procedure_review,execution,core_intake
from machinemind.evidence.ask_input import ask_request_key
from machinemind.retrieval import procedure_families,structured


def denied(*a,**kw): raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class FamilyHarness:
    def __init__(self,m,procedures,*,chosen_key,query='Insert a new cartridge and check safe closure.',planner=None,
                 company='offline-company',machine='offline-machine',response_language='en'):
        self.m=m;self.procedures=procedures;self.chosen_key=chosen_key;self.query=query
        self.company=company;self.machine=machine;self.response_language=response_language
        self.planner=planner or {'information_task':'procedure_full','required_answer_types':['ordered_actions'],
            'required_facets':['cartridge installation'], 'facet_queries':[
                {'facet':'cartridge installation','answer_type':'ordered_actions','must_cover':True,
                 'dense_queries':['insert cartridge and secure housing'],'lexical_queries':['cartridge installation'],'exact_terms':[]}]}
        self.provider_calls=[];self.sql_reads=[];self.expansions=[];self.finalize_calls=[]
        self.forced_id=None;self.no_steps=False;self.parent_lookup=False
        self.rows=[];self.children={}
        for p in procedures:
            key='procedure:'+p['procedure_id']
            text='SOURCE_TYPE: procedure\ntitle: '+p['title']+'\nshort_description: '+p['short_description']
            self.rows.append((key,machine,1,text))
            children=[]
            for step in p['steps']:
                text='SOURCE_TYPE: step\ntitle: '+step['title']+'\nstep_number: '+str(step['step_number'])+'\ndescription: '+step['description']
                row=('step:'+step['step_id'],machine,1,text)
                self.rows.append(row);children.append((row[0],step['step_number'],machine,1,text))
            self.children[key]=children
        self.rows.sort(key=lambda row:(row[0],row[2]))
        self.tasks=m._assistant_core_task_synthesis_runtime()

    def direct(self,**kwargs):
        owner=self
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def execute(self,sql,params):
                assert 'WHERE company_id = %s' in sql and '(machine_id = %s OR machine_id IS NULL' in sql
                assert owner.company in params and params.count(owner.machine)==2
                self.params=params;self.sql=sql;owner.sql_reads.append((sql,params))
            def fetchall(self):
                # Apply the actual query's bounded record/type selection. The
                # fixture store contains only this separately scoped company.
                prefixes=[str(p)[:-1] for p in self.params if isinstance(p,str) and p.endswith(':%')]
                rows=[r for r in owner.rows if any(r[0].startswith(p) for p in prefixes)]
                if "LOWER(COALESCE(text, '')) LIKE %s" in self.sql:
                    terms=[x[1:-1] for x in self.params if isinstance(x,str) and x.startswith('%') and x.endswith('%')]
                    rows=[r for r in rows if any(t in r[3].lower() for t in terms)]
                return [(r[0],r[1],r[2],r[3][:self.params[0]]) for r in rows[:self.params[-1]]]
        class Conn:
            def cursor(self):return Cursor()
            def close(self):pass
        with patch.object(self.m,'_db_conn',lambda:Conn()):
            return self.m._ask_structured_direct_fetch_sources(**kwargs)

    def related(self,**kwargs):
        assert kwargs['company_id']==self.company and kwargs['machine_id']==self.machine
        key=kwargs['parent_source_key'];self.expansions.append(key)
        if self.no_steps:return []
        return list(self.children[key])

    def parent_rows(self,**kwargs):
        assert kwargs['company_id']==self.company and kwargs['machine_id']==self.machine
        if not self.parent_lookup:return []
        result=[]
        for parent,children in self.children.items():
            for child in children:
                if child[0] not in kwargs['child_source_keys']:continue
                parent_row=next(row for row in self.rows if row[0]==parent)
                result.append({'child_source_key':child[0],'parent_source_key':parent,
                    'ordinal':child[1],'machine_id':self.machine,'page_number':1,'parent_text':parent_row[3]})
        return result

    def curate(self,**kwargs):
        expansion=replace(self.tasks.expansion,_db_fetch_related_step_pages=self.related,_db_conn=denied)
        family=replace(self.tasks.family,_db_fetch_parent_procedure_pages_for_steps=self.parent_rows,
            _v12_expand_primary_procedure_steps=lambda **kw:structured.v12_expand_primary_procedure_steps(**kw,runtime=expansion))
        curation=replace(self.tasks.curation,_v12_choose_primary_procedure_family=lambda **kw:
            procedure_families.v12_choose_primary_procedure_family(**kw,runtime=family))
        return procedure_families.v12_curate_structured_sources(**kwargs,runtime=curation,preserve_complete=True)

    def provider(self,messages,**kwargs):
        self.provider_calls.append((deepcopy(messages),deepcopy(kwargs)))
        self.assert_catalog=[x for x in kwargs['json_schema']['schema']['properties']['selected_procedure_citation_id']['enum']]
        chosen=next(x for x in self.assert_catalog if x.startswith(self.chosen_key+':'))
        return {'selected_procedure_citation_id':self.forced_id or chosen,
                'untrusted_source_body':'Must not replace the actual source.'},'offline-model'

    def runtime(self,**changes):
        parameters=dict(preserve_complete=True,_ask_structured_direct_fetch_sources=self.direct,
            _v12_curate_structured_sources=self.curate,_v13_fetch_manual_support_deterministic=lambda **kw:[],
            _v13_json_models=self.provider,_sanitize_citations_for_response=lambda rows,**kw:deepcopy(rows),
            _build_rg_links=lambda *a,**kw:[],_finalize_ask_response_for_ui=lambda response,**kw:response)
        return replace(self.tasks,**{**parameters,**changes})

    def run(self,**changes):
        # Deliberately seed only the competing family. Discovery must make the
        # other parents reachable through the real reader, not injected seeds.
        seed=[{'citation_id':self.rows[-1][0]+':seed','bubble_document_id':self.rows[-1][0],
               'source_type':'step','chunk_full':self.rows[-1][3]}]
        return task_generation.structured_ask(q=self.query,company_id=self.company,machine_id=self.machine,
            response_language=self.response_language,top_k=6,planner=deepcopy(self.planner),seed_citations=seed,
            assurance_meta={},debug=False,runtime=self.runtime(**changes))


def synthetic_catalog():
    def step(key,n,action):
        return {'step_id':key+str(n),'step_number':n,'title':action,
            'description':'AZIONE OPERATIVA\n'+action+'\n\nNOTA DI SICUREZZA\nKeep the supply isolated.'}
    return [
        {'procedure_id':'alpha','title':'Cartridge installation','short_description':'Install and secure the filter cartridge.',
         'steps':[step('alpha',1,'Isolate the supply.'),step('alpha',2,'Insert the cartridge.'),step('alpha',3,'Secure the housing.')]},
        {'procedure_id':'beta','title':'New cartridge settings','short_description':'Configure display parameters for a new cartridge.',
         'steps':[step('beta',n,'Check new cartridge display settings.') for n in range(1,21)]},
        {'procedure_id':'gamma','title':'Housing cleaning','short_description':'Clean external housing surfaces.',
         'steps':[step('gamma',1,'Clean external housing surfaces.')]}]


class SemanticProcedureFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=ExitStack()
        cls.guard.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off','AI_INTERNAL_SECRET':'offline-fixture-only',
            'OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guard.enter_context(patch(name,denied))
        cls.m=importlib.import_module('main')
    @classmethod
    def tearDownClass(cls):cls.guard.close()

    def harness(self):return FamilyHarness(self.m,synthetic_catalog(),chosen_key='procedure:alpha')

    def test_real_bounded_reader_preserves_parent_catalog_before_repeated_steps(self):
        h=self.harness();before=deepcopy(h.procedures)
        result=h.run()
        self.assertEqual(len(h.sql_reads),1)
        sql,params=h.sql_reads[0]
        self.assertNotIn("LOWER(COALESCE(text, '')) LIKE %s",sql)
        self.assertLessEqual(params[-1],max(100,int(self.m.ASK_STRUCTURED_DIRECT_SCAN_LIMIT),int(self.m.ASK_STRUCTURED_DIRECT_MAX_ITEMS)*20))
        self.assertEqual(len(h.assert_catalog),3)
        self.assertTrue(all(cid.startswith('procedure:') for cid in h.assert_catalog))
        self.assertEqual(h.procedures,before)
        self.assertEqual(h.expansions,['procedure:alpha'])
        self.assertEqual(len(h.provider_calls),1)
        self.assertEqual(h.provider_calls[0][1]['purpose'],'ask_structured_synthesis')
        self.assertEqual(h.provider_calls[0][1]['max_output_tokens'],h.tasks.V13_FAST_MAX_OUTPUT_TOKENS)
        self.assertTrue(result['_assistant_core_force_semantic_verify'])
        self.assertTrue(result['_assistant_core_source_ordered_procedure'])
        self.assertTrue(result['_assistant_core_procedure_family_review_required'])
        self.assertIn('Insert the cartridge.',result['answer'])
        self.assertNotIn('display settings',result['answer'])
        self.assertNotIn('Must not replace',result['answer'])
        manifest=result['_assistant_core_validation_evidence']
        self.assertEqual(len(manifest),4)
        self.assertTrue(all(c['bubble_document_id']=='procedure:alpha' or c.get('parent_source_key')=='procedure:alpha' for c in manifest))
        self.assertNotIn('step:alpha',h.provider_calls[0][0][1]['content'])

    def test_any_admitted_family_can_be_selected_without_name_specific_rules(self):
        h=self.harness();h.chosen_key='procedure:gamma'
        result=h.run()
        self.assertEqual(h.expansions,['procedure:gamma'])
        self.assertEqual(len(h.provider_calls),1)
        self.assertIn('Clean external housing',result['answer'])
        self.assertNotIn('Insert the cartridge',result['answer'])

    def test_unknown_or_other_tenant_model_ids_fail_before_child_acquisition(self):
        for identifier in ('procedure:foreign-tenant:p1','unknown-source','step:alpha1:p1'):
            with self.subTest(identifier=identifier):
                h=self.harness();h.forced_id=identifier
                with self.assertRaisesRegex(task_generation.GenerationEvidenceError,'not_admitted'):h.run()
                self.assertEqual(h.expansions,[])
                self.assertEqual(len(h.provider_calls),1)

    def test_successful_selection_with_missing_model_label_does_not_consume_another_call(self):
        h=self.harness()
        def unlabeled_provider(*a,**kw):
            parsed,_=h.provider(*a,**kw)
            return parsed,None
        result=h.run(_v13_json_models=unlabeled_provider)
        self.assertEqual(len(h.provider_calls),1)
        self.assertTrue(result['_assistant_core_procedure_family_review_required'])
        self.assertTrue(result['_assistant_core_force_semantic_verify'])
        self.assertIn('Insert the cartridge.',result['answer'])

    def test_no_admitted_parent_consumes_no_selection_call(self):
        h=self.harness()
        result=h.run(_ask_structured_direct_fetch_sources=lambda **kw:[])
        self.assertIsNone(result)
        self.assertEqual(h.provider_calls,[])
        self.assertEqual(h.expansions,[])

    def test_parent_recovered_from_seed_still_requires_review_and_boundary(self):
        h=self.harness();h.parent_lookup=True
        def generic_provider(messages,**kwargs):
            h.provider_calls.append((messages,kwargs))
            return {'answer_status':'no_sources','grounded_points':[]},'offline-model'
        result=h.run(_ask_structured_direct_fetch_sources=lambda **kw:[],_v13_json_models=generic_provider)
        self.assertEqual(h.expansions,['procedure:gamma'])
        self.assertEqual(len(h.provider_calls),1)
        self.assertTrue(result['_assistant_core_source_ordered_procedure'])
        self.assertTrue(result['_assistant_core_force_semantic_verify'])
        self.assertTrue(result['_assistant_core_procedure_family_review_required'])

    def test_post_selection_missing_bundle_cannot_trigger_a_second_provider(self):
        for changes in ({'_v12_curate_structured_sources':lambda **kw:[]},
                        {'_procedure_ui_model_to_text':lambda *a,**kw:''},
                        {'_v12_curate_response_items_for_ui':lambda *a,**kw:[]}):
            with self.subTest(changes=changes):
                h=self.harness()
                with self.assertRaises(task_generation.GenerationEvidenceError):h.run(**changes)
                self.assertEqual(len(h.provider_calls),1)

    def request_decision(self,h):
        m=self.m
        request=m.AssistantCoreRequest(query=h.query,requested_mode='ask',response_language='en',
            company_id=h.company,machine_id=h.machine,ai_scope='machine_all',top_k=6,
            allowed_effective_modes=('ask',))
        decision=m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,effective_mode='ask',confidence=.99,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,evidence_policy='evidence_required',
            information_task='procedure_full',required_facets=('cartridge installation',),required_answer_types=(m.REQ_ORDERED_ACTIONS,))
        return request,decision

    def validate_bundle(self,h,result,verify):
        m=self.m;rows=result['_assistant_core_validation_evidence']
        request,decision=self.request_decision(h)
        runtime=replace(m._assistant_core_ask_validation_runtime(),
            execution_runtime=replace(m._assistant_core_ask_execution_runtime(),evidence_admission=denied),
            _assistant_core_should_semantic_verify_answer=lambda d:False,
            _assistant_core_verify_or_repair_answer=verify,_assistant_core_recover_citations=lambda *a,**kw:(rows,[]),
            _sanitize_citations_for_response=lambda rows,**kw:rows,_build_rg_links=lambda *a,**kw:[],
            _v13_current_budget=lambda:None,final_contract_enabled=True,
            source_fields=m._procedure_ui_fields,source_sections=m._procedure_ui_sections)
        with patch.object(validation,'_inputs',side_effect=lambda r,q,t,d,**kw:(r,t)), \
             patch.object(validation,'_collection',side_effect=lambda q,items,d,**kw:items), \
             patch.object(validation,'_output',side_effect=lambda r,*a,**kw:r):
            return validation.validate_response(result,request,{'citations':rows},decision,runtime=runtime)

    def valid_review(self,kw):
        # Use the real source-ordered resolver: the mock supplies only a wire
        # verdict. Audit bindings are minted by the server, not the test model.
        resolved=procedure_review.resolve_reply({'outcome':'pass','answer':'','reply_mode':'retain','edits':[],
            'covered_facets':['cartridge installation'],'missing_facets':[],
            'covered_answer_types':[self.m.REQ_ORDERED_ACTIONS],'missing_answer_types':[],
            'missing_list_items':[],'citation_ids':[c['citation_id']for c in kw['candidates']]},
            answer=kw['answer'],layout=procedure_review.block_layout(kw['answer']),
            required_facets=['cartridge installation'],required_types=[self.m.REQ_ORDERED_ACTIONS],
            source_ordered=True,source_notes=kw['repair_context']['procedure_notes'])
        resolved['operation_boundary']={'complete':True,'checked_blocks':[
            b['block_id']for b in procedure_review.block_layout(kw['answer'])['blocks']],
            'changes':[],'precedence':[]}
        return procedure_review.verify_operation_boundary(resolved,
            layout=procedure_review.block_layout(kw['answer']),
            source_texts={c['citation_id']:self.m._assistant_core_candidate_evidence_text(c)for c in kw['candidates']},
            source_ordered=True)

    def test_simple_procedure_reviewer_receives_expanded_actual_draft_and_manifest(self):
        h=self.harness();result=h.run();calls=[];rows=result['_assistant_core_validation_evidence']
        def verify(**kw):calls.append(kw);return self.valid_review(kw)
        verified=self.validate_bundle(h,result,verify)
        self.assertEqual(len(calls),1)
        self.assertIn('Insert the cartridge.',calls[0]['answer'])
        self.assertEqual(calls[0]['repair_context']['procedure_structure']['answer_sha256'],procedure_review.digest(calls[0]['answer']))
        self.assertEqual({c['citation_id']for c in calls[0]['candidates']},{c['citation_id']for c in rows})
        contract=verified['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed']);self.assertTrue(contract['complete'])
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertNotIn('_assistant_core_procedure_family_review_required',verified)

    def test_required_review_missing_or_malformed_never_inherits_deterministic_pass(self):
        invalid=[None,{}, {'outcome':'unavailable'}, {'outcome':'pass','answer':'unbound draft'},
                 {'outcome':'pass','covered_facets':'cartridge installation'},
                 {'outcome':'pass','procedure_review':[]}, {'outcome':'partial','answer':'incomplete'}]
        for value in invalid:
            for after_repair in (False,True):
                with self.subTest(value=value,after_repair=after_repair):
                    h=self.harness();result=h.run();calls=[]
                    if after_repair:result['_assistant_core_repair_attempted']=True
                    def verify(**kw):calls.append(kw);return deepcopy(value)
                    verified=self.validate_bundle(h,result,verify)
                    contract=verified['meta']['assistant_core_validation']['answer_contract']
                    self.assertFalse(contract['passed']);self.assertFalse(contract.get('complete',False))
                    self.assertEqual(len(calls),1)
                    if after_repair:
                        self.assertNotEqual(verified['status'],'answered')
                        self.assertNotIn('_assistant_core_procedure_family_review_required',verified)
                    else:
                        self.assertTrue(verified['_assistant_core_repair_needed'])
                        self.assertTrue(verified['_assistant_core_procedure_family_review_required'])

    def test_repaired_semantic_pass_without_required_audit_is_rejected(self):
        h=self.harness();result=h.run();result['_assistant_core_repair_attempted']=True
        result['_assistant_core_semantic_verified']={'outcome':'pass','answer':result['answer'],
            'covered_facets':['cartridge installation'],'covered_answer_types':[self.m.REQ_ORDERED_ACTIONS],
            'missing_facets':[],'missing_answer_types':[],'missing_list_items':[]}
        verified=self.validate_bundle(h,result,denied)
        self.assertNotEqual(verified['status'],'answered')
        contract=verified['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertEqual(contract['reason'],'procedure_family_semantic_review_incomplete')

    def test_required_review_audit_must_bind_resolved_answer_after_repair(self):
        for broken in ('digest','coverage','audit_mode','boundary_missing','boundary_digest','unchanged'):
            with self.subTest(broken=broken):
                h=self.harness();result=h.run();reviews=[]
                def verify(**kw):reviews.append(deepcopy(kw));return self.valid_review(kw)
                self.validate_bundle(h,deepcopy(result),verify)
                verified=self.valid_review(reviews[0])
                if broken=='digest':verified['procedure_review']['output_answer_sha256']='0'*64
                if broken=='coverage':verified['covered_facets']=[]
                if broken=='audit_mode':verified['procedure_review']['reply_mode']={}
                if broken=='boundary_missing':verified.pop('operation_boundary_validation')
                if broken=='boundary_digest':verified['operation_boundary_validation']['answer_sha256']='0'*64
                result['_assistant_core_repair_attempted']=True
                result['_assistant_core_semantic_verified']=verified
                final=self.validate_bundle(h,result,denied)
                contract=final['meta']['assistant_core_validation']['answer_contract']
                self.assertEqual(contract['passed'],broken=='unchanged')
                self.assertEqual(final['status']=='answered',broken=='unchanged')

    def test_execution_and_intake_latch_post_selection_error_without_generic_provider(self):
        h=self.harness();request,decision=self.request_decision(h)
        runtime=replace(self.m._assistant_core_ask_execution_runtime(),evidence_admission=denied,
            _v13_structured_ask=lambda **kw:h.run(_procedure_ui_model_to_text=lambda *a,**kw:''),
            _v13_generate_ask_response=denied,_assistant_core_recover_ask_from_evidence=denied,
            _assistant_core_verify_or_repair_answer=denied)
        # Isolate the real intake invocation/fault latch from its separate reader
        # setup. No authorization behavior is asserted by this unit fixture.
        intake=object.__new__(core_intake._CoreIntake)
        intake.request=request;intake.key=ask_request_key(request);intake.active=True;intake.fault=None
        intake.owner_invoke=lambda callback,req:callback()
        retrieval={'citations':[{'citation_id':'step:seed:p1','source_type':'step'}],'plan':{}}
        with patch.object(execution,'_admit_input',side_effect=lambda req,data,dec,**kw:data):
            with self.assertRaises(task_generation.GenerationEvidenceError) as error:
                intake.invoke(execution.synthesize_ask,request,request,retrieval,decision,runtime=runtime)
            self.assertIs(intake.fault,error.exception)
            with self.assertRaises(task_generation.GenerationEvidenceError):
                intake.invoke(denied,request)
        self.assertEqual(len(h.provider_calls),1)


if __name__=='__main__':unittest.main(verbosity=2)
