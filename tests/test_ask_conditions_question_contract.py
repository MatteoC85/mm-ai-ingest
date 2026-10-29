"""Question-owned obligations and source-scoped checklist review, offline.

The source session, reader adapters, admissions, compiler, transport, validator
and final handoff are real. SQL, allowance and provider replies are fixtures;
these tests prove the contract supplied to a reviewer, not model compliance.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import normalize_decision, _finish_ask_validation, decorate_response
from machinemind.ask import execution, validation, request_completion, request_scope, review_evidence, procedure_review
from machinemind.infrastructure import request_budget
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_conditions_family_admission as families
import test_ask_procedure_scope as routing
import test_ask_repair_capacity as capacity
import test_ask_prepared_scope_admission as prepared_fixtures


def raw_decision():
    raw = routing.payload()
    raw.update(request_kind='factual', evidence_state='supported')
    raw['required_facets'] = ['Guard status and safety-circuit test before insertion', 'Pre-insertion utilities']
    raw['facet_queries'][0].update(facet=raw['required_facets'][0],
        dense_queries=['safety circuit before insertion'], lexical_queries=['reset controls'], exact_terms=['RESET'])
    raw['facet_queries'][1].update(facet=raw['required_facets'][1],
        dense_queries=['utility connections'], lexical_queries=['pressure'], exact_terms=['5-6 bar'])
    return raw


class QuestionFixture(families.OwnedFamilyFixture):
    def __init__(self, *args, reject=False, **kwargs):
        self.reject = reject
        super().__init__(*args, **kwargs)

    def reply(self, reply, model, messages, kwargs):
        if self.reject == 'safety' and 'grounded_points' in reply:
            reply = {**reply, 'grounded_points':[reply['grounded_points'][i] for i in (0,2)]}
        if kwargs['purpose'] == 'assistant_core_answer_contract_verifier' and self.reject:
            reply = {**reply, 'outcome':'no_sources', 'reply_mode':'replace', 'answer':'', 'edits':[], 'covered_facets':[],
                'missing_facets':list(self.decision.required_facets),
                'reason':'The explicitly requested test has no source applicable to this operation.'}
            if self.reject == 'safety':
                reply.update(reason='Documented PPE and STOP precautions are missing.',
                    missing_list_items=['Wear gloves and goggles','Keep STOP pressed'])
        return super().reply(reply, model, messages, kwargs)


class AskConditionsQuestionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        families.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m = families.AskConditionsFamilyAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        families.AskConditionsFamilyAdmissionTests.tearDownClass()

    def decision(self, query, language='en', raw=None):
        return normalize_decision(routing.request(query, language), raw or raw_decision())

    def test_normalization_separates_search_hypotheses_without_new_or_changed_query_arrays(self):
        for language, query in (('en','Which checks and safety conditions apply before cartridge insertion?'),
                ('it','Quali controlli e condizioni di sicurezza verifico prima di inserire la cartuccia?')):
            raw=raw_decision(); before=deepcopy(raw)
            d=self.decision(query, language, raw)
            self.assertEqual(d.required_facets, (request_scope.QUESTION_CONDITIONS_REQUIREMENT,))
            self.assertEqual(len(d.facet_queries), len(raw['facet_queries']))
            self.assertEqual([f.must_cover for f in d.facet_queries], [True, False])
            self.assertEqual(d.required_answer_types, tuple(raw['required_answer_types']))
            for old, new in zip(raw['facet_queries'], d.facet_queries):
                for key in ('dense_queries','lexical_queries','exact_terms'):
                    self.assertEqual(getattr(new,key), tuple(old.get(key,())))
                self.assertEqual(new.answer_type, old['answer_type'])
            self.assertEqual(raw,before)

    def test_legacy_full_ambiguous_and_other_modes_keep_their_original_requirements(self):
        for changes in ({'procedure_scope':''}, {'procedure_scope':'execution_steps'},
                {'information_task':'procedure_full'}, {'request_kind':'ambiguous'},
                {'required_answer_types':['checklist','ordered_actions']}):
            raw=raw_decision();raw.update(changes)
            d=self.decision('How do I carry out the operation?', raw=raw)
            self.assertEqual(d.required_facets, tuple(raw['required_facets']))
            self.assertTrue(all(f.must_cover for f in d.facet_queries))
        for mode in ('root_cause','smart_diagnostic'):
            d=normalize_decision(routing.request(mode=mode), raw_decision())
            self.assertEqual(d.required_facets, tuple(raw_decision()['required_facets']))
            malformed=replace(self.decision('Which checks apply?'),effective_mode=mode)
            self.assertIs(request_scope.bind_conditions_requirements_to_question(malformed),malformed)

    def owned(self, query, language='en', *, reject=False, action=None, raw=None, finish=True):
        now=[1000.0]
        with patch.object(request_completion,'_monotonic',side_effect=lambda:now[0]), \
             patch.object(request_budget,'_monotonic',side_effect=lambda:now[0]):
            completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'),started=now[0])
            h=QuestionFixture(self, query=query, language=language, decision=self.decision(query,language,raw),
                transport=True,reject=reject)
            try:
                capacity.AskRepairCapacityTests.first_transport(self,h,failed=False)
                h.prepared['assistant_core_decision']={'supported':True,'strict_supported':True}
                h.owner._prepared_retrieval=deepcopy(h.prepared)
                h.generated=execution.synthesize_ask(h.request,h.prepared,h.decision,runtime=h.runtime)
                flow=h.owner._response_flow
                flow.execution=replace(flow.execution,_v13_json_models=h.reviewer)
                def provider(*args,**kwargs):
                    if action=='revoke':h.allowance=frozenset()
                    return flow._verification_model(*args,**kwargs)
                eruntime=replace(h.runtime,_v13_json_models=provider)
                vruntime=validation.bind_ask_validation(h.owner._validation_factory(),execution_runtime=eruntime)
                vruntime=replace(vruntime,_v13_current_budget=lambda:completion.budget)
                if action=='tamper_manifest':
                    h.generated['_assistant_core_validation_evidence'].append(h.prepared['citations'][0])
                if action in {'revoke','tamper_manifest'}:
                    with self.assertRaises(EvidenceContractError):
                        validation.validate_response(h.generated,h.request,h.prepared,h.decision,runtime=vruntime)
                    self.assertEqual(len(h.posts),3)
                    self.assertFalse(h.review_packets)
                    return
                result=validation.validate_response(h.generated,h.request,h.prepared,h.decision,runtime=vruntime)
                self.assertEqual(len(h.review_packets),1)
                writer=h.calls[1][1][1]['content'] if h.writer_packets else ''
                review=h.review_packets[0]
                self.assertIn('QUESTION:\n'+query+'\n',review)
                if writer:self.assertIn('QUESTION:\n'+query+'\n',writer)
                if finish:
                    projected=decorate_response(_finish_ask_validation(result),h.request,h.decision)
                    h.owner._finalize_core(h.request,projected)
                    self.assertTrue(flow.completed)
                    self.assertFalse(any(k.startswith('_assistant_core_') for k in projected))
                return dict(result=result,writer=writer,review=review,primary=deepcopy(h.generated['_assistant_core_validation_evidence']),
                    rows=deepcopy(h.rows),decision=h.decision,posts=len(h.posts))
            finally:h.close();completion.close()

    def test_owned_writer_and_reviewer_share_complete_family_and_no_foreign_startup(self):
        for language,query in (('en','Which checks and safety conditions apply before cartridge insertion?'),
                ('it','Quali controlli e condizioni di sicurezza verifico prima di inserire la cartuccia?')):
            x=self.owned(query,language)
            self.assertEqual(x['result']['status'],'answered')
            for _,_,_,body in x['rows'][:5]:
                self.assertIn(body,x['writer'])
                self.assertIn(body,x['review'])
            self.assertNotIn(x['rows'][-1][3],x['review'])
            contract=x['result']['meta']['assistant_core_validation']['answer_contract']
            packet=contract['semantic_verifier']['review_evidence']
            facets=json.loads(x['review'].split('FACET_CONTRACTS: ',1)[1].splitlines()[0])
            self.assertEqual([f['facet'] for f in facets],[request_scope.QUESTION_CONDITIONS_REQUIREMENT])
            self.assertTrue(all(f['must_cover'] for f in facets))
            self.assertEqual(packet['required_records'],5)
            self.assertEqual(packet['emitted_records'],5)
            self.assertEqual(packet['emitted_citation_ids'],packet['required_citation_ids'])
            self.assertTrue(packet['complete_primary_representation'])
            self.assertEqual(x['posts'],4)

    def test_explicit_missing_requirement_in_long_question_still_fails_closed(self):
        questions=[
            'Before insertion, which checks and safety conditions apply, and what safety-interlock test is required?',
            'Prima di inserire la cartuccia, quali condizioni verifico e quale pressione deve essere presente?',
            'Before insertion, verify all documented preconditions. '+('Please preserve every requested detail. '*35)
                +'Finally, state the required interlock test and the pneumatic pressure for this phase.']
        for query in questions:
            x=self.owned(query,'it' if query.startswith('Prima') else 'en',reject=True)
            self.assertEqual(x['result']['status'],'no_sources')
            self.assertEqual(x['result']['result_code'],'INCOMPLETE_ANSWER_CONTRACT')
            contract=x['result']['meta']['assistant_core_validation']['answer_contract']
            self.assertFalse(contract['passed'])
            self.assertEqual(contract['semantic_verifier']['missing_facets'],list(x['decision'].required_facets))
            self.assertFalse(x['result'].get('_assistant_core_repair_needed'))
            self.assertEqual(x['posts'],4)

    def test_source_revocation_before_reviewer_dispatch_remains_fail_closed(self):
        self.owned('Which checks apply before inserting the cartridge?',action='revoke')

    def test_omitted_documented_safety_still_rejects_a_generic_question(self):
        x=self.owned('Which checks and safety conditions apply before cartridge insertion?',reject='safety')
        current=x['review'].split('CURRENT_ANSWER:\n',1)[1].split('\n\nSOURCES:',1)[0]
        self.assertNotIn('gloves',current)
        self.assertNotIn('STOP',current)
        self.assertIn('Indossare guanti e occhiali; mantenere STOP premuto.',x['review'])
        self.assertEqual(x['result']['result_code'],'INCOMPLETE_ANSWER_CONTRACT')
        self.assertFalse(x['result']['meta']['assistant_core_validation']['answer_contract']['passed'])
        self.assertEqual(x['posts'],4)

    def test_caller_cannot_enlarge_observed_producer_manifest(self):
        self.owned('Which checks apply before inserting the cartridge?',action='tamper_manifest')

    def test_full_procedure_review_keeps_initial_search_extensions(self):
        raw=raw_decision();raw.update(procedure_scope='execution_steps',information_task='procedure_full',
            required_answer_types=['ordered_actions','safety_conditions'])
        x=self.owned('Descrivi la procedura completa di inserimento cartuccia.', 'it',raw=raw,finish=False)
        self.assertIn(x['rows'][-1][3],x['review'])
        self.assertEqual(x['decision'].required_facets,tuple(raw['required_facets']))

    def test_manual_fallback_keeps_extensions_and_rejects_a_spoofed_family_flag(self):
        base=prepared_fixtures.AskPreparedScopeAdmissionTests();base.m=self.m
        for spoof in (False,True):
            q,d,owner,data,rt,session,calls,_=base.harness()
            try:
                generated=execution.synthesize_ask(q,data,d,runtime=rt)
                self.assertNotIn('_assistant_core_procedure_family_review_required',generated)
                flow=owner._response_flow
                reviews=[];extensions=[]
                def provider(messages,**kw):
                    reviews.append(messages)
                    text=messages[1]['content'].split('CURRENT_ANSWER:\n',1)[1].split('\n\nSOURCES:',1)[0]
                    return {'outcome':'pass','reply_mode':'retain','answer':'','edits':[],
                        'covered_facets':list(d.required_facets),'missing_facets':[],
                        'covered_answer_types':list(d.required_answer_types),'missing_answer_types':[],
                        'expected_list_items':[],'covered_list_items':[],'missing_list_items':[],
                        'enumeration_requested':False,'citation_ids':[c['citation_id'] for c in generated['citations']],
                        'reason':'offline fixture','operation_boundary':{'complete':True,'changes':[],
                            'checked_blocks':[b['block_id'] for b in procedure_review.block_layout(text)['blocks']]}},'offline'
                flow.execution=replace(flow.execution,_v13_json_models=provider)
                budget=self.m._assistant_core_new_budget('ask')
                rt=replace(rt,_v13_json_models=flow._verification_model,_v13_current_budget=lambda:budget)
                vr=validation.bind_ask_validation(owner._validation_factory(),execution_runtime=rt)
                vr=replace(vr,_v13_current_budget=lambda:budget)
                compile_packet=review_evidence.compile_packet
                def capture(**kw):
                    extensions.append(list(kw['extension']))
                    return compile_packet(**kw)
                with patch.object(review_evidence,'compile_packet',capture):
                    if spoof:
                        generated['_assistant_core_procedure_family_review_required']=True
                        with self.assertRaises(EvidenceContractError):
                            validation.validate_response(generated,q,data,d,runtime=vr)
                        self.assertEqual(reviews,[])
                        self.assertEqual(extensions,[])
                    else:
                        result=validation.validate_response(generated,q,data,d,runtime=vr)
                        self.assertEqual(len(reviews),1)
                        self.assertTrue(extensions[0])
                        self.assertEqual(result['status'],'answered')
                        projected=decorate_response(_finish_ask_validation(result),q,d)
                        owner._finalize_core(q,projected)
                        self.assertTrue(flow.completed)
            finally:base.finish(owner,session)


if __name__=='__main__':unittest.main(verbosity=2)
