"""Phase-scoped ASK across real family expansion, admission, transport and finish.

SQL rows and model replies are synthetic. These tests prove what reaches each
consumer and that its gates hold; they do not prove a model understands a phase.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import normalize_decision, _finish_ask_validation, decorate_response
from machinemind.ask import execution, validation, request_completion, request_scope
from machinemind.infrastructure import request_budget
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_conditions_family_admission as families
import test_ask_repair_capacity as capacity


def phase_catalog():
    rows = families.catalog()
    primary = rows[0]
    primary.update(title='Pulizia e ripristino del filtro', short_description='Pulizia, calibrazione e chiusura prima del riavvio.')
    def step(n, title, action, safety):
        return dict(step_id='install'+str(n),step_number=n,title=title,
            description='AZIONE OPERATIVA\n'+action+'\n\nNOTA DI SICUREZZA\n'+safety)
    primary['steps'] = [
        step(1,'Isolare','Arrestare e isolare le energie.','Solo personale addestrato.'),
        step(2,'Pulire e calibrare','Pulire il filtro ed eseguire la calibrazione.','Indossare guanti e occhiali.'),
        step(3,'Richiudere','Richiudere e bloccare la custodia dopo pulizia o calibrazione.','Mantenere le energie isolate durante la chiusura.'),
        step(4,'Controllo finale','Rimuovere gli attrezzi e verificare che la custodia sia bloccata.','Eseguire i controlli senza alimentazione.'),
        step(5,'Riavviare','Solo dopo il controllo finale riavviare e provare a bassa velocita.','Verificare il consenso Ready durante la prova.')]
    return rows


def decision(m, query, phase):
    raw = dict(request_kind='procedure', effective_mode='ask', confidence=.98,
        requested_mode_fit=True,evidence_state='supported',evidence_policy='machine_sources_required',
        information_task='procedure_segment',procedure_scope='bounded_phase',
        procedure_phase=dict(zip(('completed_context','requested_actions','later_context'),phase)),
        required_answer_types=['ordered_actions','safety_conditions'],required_facets=['Requested restoration','Applicable precautions'],
        facet_queries=[dict(facet='Requested restoration',answer_type='ordered_actions',must_cover=True,
            dense_queries=['restore after maintenance'],lexical_queries=['restore housing']),
            dict(facet='Applicable precautions',answer_type='safety_conditions',must_cover=True,
            dense_queries=['isolation during closure'],lexical_queries=['closure precautions'])])
    request=m.AssistantCoreRequest(query=query,requested_mode='ask',response_language='en',
        company_id='offline-company',machine_id='offline-machine',ai_scope='machine_all',top_k=6)
    return normalize_decision(request,raw)


class PhaseFixture(families.OwnedFamilyFixture):
    def __init__(self,*args,reject=False,phase_tamper=False,**kwargs):
        self.reject,self.phase_tamper=reject,phase_tamper
        super().__init__(*args,**kwargs)

    def provider(self,messages,**kwargs):
        if 'selected_procedure_citation_id' in kwargs['json_schema']['schema']['properties']:
            reply=super().provider(messages,**kwargs)
            if self.phase_tamper:
                self.owner._task_input['synthesis.structured']['plan']['procedure_phase'][1]='unrequested new action'
            return reply
        self.calls.append(('writer',deepcopy(messages),deepcopy(kwargs)))
        block=messages[1]['content'].split('SOURCES:\n',1)[1]
        self.writer_packets.append(block)
        cids={key:next(cid for cid in re.findall(r'\[([^\]\n]+)\]',block) if cid.startswith(key+':'))
            for key,*_ in self.children['procedure:'+self.chosen]}
        en=self.request.response_language=='en'
        texts=(['With energy isolated, trained personnel must close and lock the housing.',
                'Remove the tools and verify that the housing is locked with power off.'] if en else
               ['Con energie isolate, il personale addestrato deve richiudere e bloccare la custodia.',
                'Rimuovere gli attrezzi e verificare che la custodia sia bloccata senza alimentazione.'])
        if self.reject:texts[0]='Close and lock the housing.'
        points=[{'text':texts[0],'citation_ids':[cids['step:install1'],cids['step:install3']]},
                {'text':texts[1],'citation_ids':[cids['step:install4']]}]
        return self.reply({'answer_status':'no_sources' if self.no_sources else 'answered',
            'grounded_points':[] if self.no_sources else points},'offline-phase-writer',messages,kwargs)

    def reply(self,reply,model,messages,kwargs):
        if self.reject and kwargs['purpose']=='assistant_core_answer_contract_verifier':
            reply={**reply,'outcome':'no_sources','reply_mode':'replace','answer':'','edits':[],
                'covered_facets':[],'missing_facets':list(self.decision.required_facets),
                'missing_answer_types':['safety_conditions'],'reason':'Proposed closure omits documented isolation.'}
        return super().reply(reply,model,messages,kwargs)


class AskBoundedPhaseAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        families.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m=families.AskConditionsFamilyAdmissionTests.m
    @classmethod
    def tearDownClass(cls):families.AskConditionsFamilyAdmissionTests.tearDownClass()

    def owned(self,query,phase,*,language='en',negative=None):
        now=[1000.0]
        with patch.object(request_completion,'_monotonic',side_effect=lambda:now[0]), \
             patch.object(request_budget,'_monotonic',side_effect=lambda:now[0]):
            completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'),started=now[0])
            h=PhaseFixture(self,query=query,decision=decision(self.m,query,phase),rows=phase_catalog(),
                kind='procedure',language=language,transport=True,reject=negative=='safety',
                revoke=negative=='revoke',phase_tamper=negative=='tamper',no_sources=negative=='no_sources')
            try:
                limits=(completion.budget.deadline_monotonic,completion.budget.max_llm_calls,
                    completion.budget.max_estimated_cost_usd)
                capacity.AskRepairCapacityTests.first_transport(self,h,failed=False)
                h.prepared['assistant_core_decision']={'supported':True,'strict_supported':True}
                h.owner._prepared_retrieval=deepcopy(h.prepared)
                if negative in {'revoke','tamper'}:
                    with self.assertRaises(EvidenceContractError):
                        execution.synthesize_ask(h.request,h.prepared,h.decision,runtime=h.runtime)
                    self.assertEqual([row[0] for row in h.calls],['selector'])
                    self.assertEqual(len(h.posts),2)
                    return
                h.generated=execution.synthesize_ask(h.request,h.prepared,h.decision,runtime=h.runtime)
                flow=h.owner._response_flow
                flow.execution=replace(flow.execution,_v13_json_models=h.reviewer)
                eruntime=replace(h.runtime,_v13_json_models=flow._verification_model)
                vruntime=validation.bind_ask_validation(h.owner._validation_factory(),execution_runtime=eruntime)
                result=validation.validate_response(h.generated,h.request,h.prepared,h.decision,runtime=vruntime)
                projected=decorate_response(_finish_ask_validation(result),h.request,h.decision)
                h.owner._finalize_core(h.request,projected)
                self.assertTrue(flow.completed)
                self.assertFalse(any(key.startswith('_assistant_core_') for key in projected))
                self.assertEqual((completion.budget.deadline_monotonic,completion.budget.max_llm_calls,
                    completion.budget.max_estimated_cost_usd),limits)
                self.assertEqual(len(set(h.budget_ids)),1)
                return dict(result=projected,generated=deepcopy(h.generated),writer=h.writer_packets,
                    review=h.review_packets,calls=deepcopy(h.calls),rows=h.rows,posts=len(h.posts))
            finally:h.close();completion.close()

    def test_phase_context_reaches_real_writer_review_and_finish_in_both_languages(self):
        cases=[('After cleaning the filter, what must I restore before restart?',
            ('After cleaning the filter','what must I restore','before restart'),'en'),
            ('Dopo la calibrazione cosa devo ripristinare prima del riavvio?',
            ('Dopo la calibrazione','cosa devo ripristinare','prima del riavvio'),'it')]
        for query,phase,language in cases:
            with self.subTest(language=language):
                x=self.owned(query,phase,language=language)
                self.assertEqual(x['result']['status'],'answered')
                self.assertEqual([c[0] for c in x['calls']],['selector','writer','reviewer'])
                self.assertEqual(x['posts'],4)
                self.assertEqual(len(x['generated']['_assistant_core_validation_evidence']),6)
                self.assertEqual(len(x['result']['citations']),3)
                self.assertNotIn('_assistant_core_source_ordered_procedure',x['generated'])
                for packet in (x['writer'][0],x['review'][0]):
                    for _,_,_,body in x['rows'][:6]:self.assertIn(body,packet)
                    self.assertNotIn(x['rows'][-1][3],packet)
                for stage, messages, kwargs in x['calls'][1:]:
                    self.assertIn('PROCEDURE_SCOPE=bounded_phase',messages[0]['content'])
                    self.assertIn(query,messages[1]['content'])
                    for value in phase:self.assertIn(value,messages[1]['content'])
                self.assertNotIn('Ready',x['result']['answer'])
                contract=x['result']['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['passed'])
                self.assertFalse(contract['procedure_structure']['source_sequence_required'])
                self.assertIn('ordered_actions',contract['required_answer_types'])

    def test_long_irrelevant_context_keeps_whole_question_and_exact_phase(self):
        query='Unrelated notes: '+('No technical observations. '*80)+'After cleaning, what must I restore before restart?'
        phase=('After cleaning','what must I restore','before restart')
        x=self.owned(query,phase)
        self.assertIn(query,x['calls'][1][1][1]['content'])
        self.assertIn(query,x['review'][0])
        self.assertEqual(x['result']['status'],'answered')

    def test_missing_precaution_is_still_rejected_without_fifth_dispatch(self):
        x=self.owned('After cleaning, what must I restore before restart?',
            ('After cleaning','what must I restore','before restart'),negative='safety')
        self.assertEqual(x['result']['result_code'],'INCOMPLETE_ANSWER_CONTRACT')
        self.assertEqual(x['posts'],4)
        self.assertFalse(x['result']['meta']['assistant_core_validation']['answer_contract']['passed'])

    def test_revocation_and_phase_metadata_tampering_block_before_writer(self):
        for negative in ('revoke','tamper'):
            with self.subTest(negative=negative):self.owned(
                'After cleaning, what must I restore before restart?',
                ('After cleaning','what must I restore','before restart'),negative=negative)

    def test_writer_no_sources_does_not_reenter_canonical_or_foreign_family(self):
        x=self.owned('After cleaning, what must I restore before restart?',
            ('After cleaning','what must I restore','before restart'),negative='no_sources')
        self.assertEqual(x['result']['status'],'no_sources')
        self.assertEqual(x['result']['citations'],[])
        self.assertEqual([c[0] for c in x['calls']],['selector','writer'])
        self.assertEqual(x['posts'],3)

    def test_scope_helper_rejects_invalid_boundaries_and_incompatible_contracts(self):
        query='After cleaning, what must I restore before restart?'
        d=decision(self.m,query,('After cleaning','what must I restore','before restart'))
        args=dict(procedure_scope=d.procedure_scope,information_task=d.information_task,
            required_answer_types=d.required_answer_types,facet_queries=d.facet_queries,
            procedure_phase=d.procedure_phase,question=query)
        self.assertTrue(request_scope.bounded_phase_contract(**args))
        for changed in ({'procedure_phase':()}, {'procedure_phase':('after cleaning','what must I restore','')},
                {'procedure_phase':('','what must I restore','')}, {'information_task':'procedure_full'},
                {'procedure_scope':'execution_steps'}, {'required_answer_types':('checklist',)}, {'facet_queries':()}):
            with self.subTest(changed=changed):self.assertFalse(request_scope.bounded_phase_contract(**{**args,**changed}))


if __name__=='__main__':unittest.main(verbosity=2)
