"""Offline repair prompt regression, using the real verifier builder and cost ledger."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, procedure_review, request_completion
from machinemind.infrastructure import request_budget


def denied(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class AskRepairPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        cls.guards.enter_context(patch.object(socket, 'create_connection', denied))
        cls.guards.enter_context(patch.object(socket.socket, 'connect', denied))
        cls.guards.enter_context(patch('requests.sessions.Session.request', denied))
        cls.guards.enter_context(patch('psycopg2.connect', denied))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def fixture(self):
        answer = ('1. Isolate the supply and keep STOP pressed.\n'
                  '2. Open the guide and insert the material.\n'
                  '3. Close the guide and secure the locking lever.\n')
        contract = {'passed': False, 'complete': False, 'reason': 'manual_procedure_semantic_review_incomplete',
            'missing_answer_facets': ['documented closure'], 'missing_evidence_facets': [],
            'semantic_missing_answer_types': ['safety_conditions'], 'missing_list_items': ['secure the locking lever'],
            'evidence_facet_coverage': 1.0, 'requirement_checks': [
                {'requirement':'safety_conditions','passed':False,'reason':'closure needs exact supporting source'}],
            'semantic_verifier': {'outcome':'no_sources','reason':'operation_boundary_quote_or_order_invalid',
                'covered_facets':['ordered actions'], 'missing_facets':['documented closure'],
                'covered_answer_types':['ordered_actions'], 'missing_answer_types':['safety_conditions'],
                'expected_list_items':['secure the locking lever'], 'covered_list_items':[],
                'missing_list_items':['secure the locking lever'],
                'answer': answer * 18,
                'operation_boundary': {'complete': False, 'checked_blocks': ['step_1','step_2','step_3'],
                    'changes': [{'block_id':'step_2', 'action_quote':'Open the guide',
                        'closing_source_citation_id':'offline:p1:c0',
                        'closing_source_quote':'Close the guide and secure the locking lever. '*8,
                        'closing_answer_quote':'Close the guide and secure the locking lever.',
                        'resolution':'restored'} for _ in range(12)]},
                'operation_boundary_validation':{'version':'ask-operation-boundary-v1','complete':False,
                    'answer_sha256':'0'*64,'checked_blocks':3,'temporary_changes':None},
                'review_evidence': {'manifest_sha256':'a'*64, 'selected_source_ids':['offline:p1:c0']},
                'reviewed_draft_binding': {'input_answer_sha256':'b'*64,'resolved_answer_sha256':'c'*64}},
            'procedure_structure': {'answer_sha256':'d'*64,'units':[{'citation_id':'offline:p1:c0','number':1}]}}
        return answer, contract

    def capture_prompt(self, sources, *, compact=True):
        m = self.m
        answer, contract = self.fixture()
        request = m.AssistantCoreRequest(query='Describe material threading and final closure.',
            requested_mode='ask', response_language='en', company_id='offline-company', machine_id='offline-machine',
            ai_scope='machine_all', top_k=8)
        decision = m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE, effective_mode='ask', confidence=.99,
            requested_mode_fit=True, evidence_state=m.EVIDENCE_SUPPORTED, evidence_policy='evidence_required',
            information_task=m.INFO_PROCEDURE_FULL, required_facets=('ordered actions','documented closure'),
            required_answer_types=(m.REQ_ORDERED_ACTIONS,m.REQ_SAFETY_CONDITIONS))
        row = {'citation_id':'offline:p1:c0','bubble_document_id':'offline', 'source_type':'document',
               'page_from':1,'page_to':1,'chunk_index':0,'chunk_full':sources,'snippet':sources}
        structure = procedure_review.observe_structure(answer, [], fields=lambda r: {}, notes=(),
            notes_present=lambda a,b: True, source_ordered=False)
        captured = []
        def provider(messages, **kwargs):
            captured.append({'messages':deepcopy(messages), **deepcopy(kwargs)})
            return {'outcome':'no_sources','answer':'','missing_facets':['documented closure']}, 'offline-model'
        runtime = replace(m._assistant_core_ask_execution_runtime(), evidence_admission=denied,
            _v13_current_budget=lambda: m._assistant_core_new_budget('ask'),
            _v13_sources_block=lambda *a, **kw:sources, _v13_json_models=provider)
        with patch.object(execution, '_admit_input', side_effect=lambda request,data,decision,**kw:data):
            with patch.object(procedure_review, 'repair_contract_feedback',
                procedure_review.repair_contract_feedback if compact else lambda value: deepcopy(value)):
                execution.verify_or_repair_answer(request=request,decision=decision,answer=answer,candidates=[row],
                    repair_context={'first_answer_contract':contract,'procedure_structure':structure,
                        'missing_answer_facets':['documented closure'],
                        'missing_answer_types':['safety_conditions'],'missing_list_items':['secure the locking lever']},
                    runtime=runtime)
        self.assertEqual(len(captured),1)
        return captured[0]

    def payload(self, captured):
        schema = captured['json_schema']
        return {'model':'gpt-5.6-terra','input':captured['messages'],'store':False,
            'reasoning':{'effort':'low','context':'current_turn'},
            'text':{'format':{'type':'json_schema','name':schema['name'],'strict':True,'schema':schema['schema']}},
            'safety_identifier':'offline-fixture'}

    def test_feedback_preserves_all_decision_reasons_without_mutating_original(self):
        _, contract = self.fixture()
        original = deepcopy(contract)
        feedback = procedure_review.repair_contract_feedback(contract)
        self.assertEqual(contract,original)
        for key in ('passed','complete','reason','missing_answer_facets','missing_evidence_facets',
                    'semantic_missing_answer_types','missing_list_items','evidence_facet_coverage','requirement_checks'):
            self.assertEqual(feedback[key], contract[key])
        self.assertEqual(feedback['semantic_verifier']['missing_facets'], ['documented closure'])
        self.assertEqual(feedback['semantic_verifier']['operation_boundary_validation'], {'complete':False})
        for key in ('answer','operation_boundary','review_evidence','reviewed_draft_binding'):
            self.assertNotIn(key,feedback['semantic_verifier'])
        self.assertNotIn('procedure_structure', feedback)
        feedback['missing_answer_facets'].append('mutated')
        self.assertEqual(contract,original)

    def test_real_verifier_retains_source_packet_answer_and_every_other_prompt_field(self):
        sources = 'SOURCE offline:p1:c0\nOpen the guide. Close the guide and secure the locking lever.\n' * 30
        old, new = self.capture_prompt(sources,compact=False), self.capture_prompt(sources)
        self.assertEqual(old['messages'][0],new['messages'][0])
        def without_contract(text):
            start=text.index('FIRST_ANSWER_CONTRACT: ')
            end=text.index('\nMACHINE_CATALOG:',start)
            return text[:start]+text[end:]
        self.assertEqual(without_contract(old['messages'][1]['content']),without_contract(new['messages'][1]['content']))
        self.assertIn('SOURCES:\n'+sources,new['messages'][1]['content'])
        self.assertEqual(old['json_schema'],new['json_schema'])
        self.assertEqual(old['max_output_tokens'],new['max_output_tokens'])

    def test_failure_diagnostics_are_bounded_and_never_retain_model_or_source_text(self):
        owner = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=False), started=time.monotonic())
        try:
            contract={'outcome':'no_sources','reason':'operation_boundary_quote_or_order_invalid',
                'answer':'PRIVATE_MODEL_RESPONSE','missing_facets':['PRIVATE_FACET'],
                'operation_boundary_validation':{'complete':False,'source':'PRIVATE_SOURCE'},
                'procedure_structure':{'source_sequence_required':True,'sequence_complete':False,
                    'source_notes_present':True,'visible_numbers':[2,7],'expected_numbers':[2,7,11],
                    'source':'PRIVATE_SOURCE'},'procedure_review':{'reply_mode':'edit','answer':'PRIVATE_TEXT'}}
            original=deepcopy(contract)
            request_completion.record_contract_observation('semantic_review',contract)
            request_completion.record_contract_observation(['PRIVATE_STAGE'],{'outcome':{},'reason':'PRIVATE_SECRET','passed':False})
            for _ in range(20):
                request_completion.record_contract_observation('final_contract', {'passed':False})
            result=owner.finish({'ok':False,'status':'error','meta':{}})
            observations=result['meta']['request_completion']['contract_checks']
            self.assertEqual(len(observations),8)
            self.assertEqual(observations[0]['reason'],'operation_boundary_quote_or_order_invalid')
            self.assertEqual(observations[0]['missing_facets_count'],1)
            self.assertEqual(observations[0]['visible_numbers_count'],2)
            self.assertEqual(observations[0]['expected_numbers_count'],3)
            self.assertFalse(observations[0]['sequence_complete'])
            self.assertTrue(observations[0]['source_notes_present'])
            self.assertEqual(observations[0]['review_reply_mode'],'edit')
            self.assertEqual(observations[1]['stage'],'other')
            self.assertEqual(observations[1]['reason'],'other')
            self.assertNotIn('PRIVATE_',json.dumps(observations))
            self.assertEqual(contract,original)
        finally:
            owner.close()

    def test_observed_repair_cost_boundary_has_room_for_full_3200_tokens_after_projection(self):
        # Reproduce the measured 44,923-byte reserved input and pre-repair cost
        # from T2-A01, with synthetic text and the REAL schema/ledger. This is
        # a deterministic cost regression, not a reconstruction of its answer.
        sources='Close the guide and secure the locking lever.\n'
        old=self.capture_prompt(sources,compact=False)
        bound,_=request_budget._text_input_bound(old['messages'],self.payload(old))
        self.assertLess(bound,44923)
        sources+='X'*(44923-bound)
        old,new=self.capture_prompt(sources,compact=False),self.capture_prompt(sources)
        observed_bound,_=request_budget._text_input_bound(old['messages'],self.payload(old))
        self.assertEqual(observed_bound,44923)
        caps=[];bounds=[]
        for captured in (old,new):
            budget=self.m._assistant_core_new_budget('ask')
            budget.max_estimated_cost_usd=.25
            budget.estimated_cost_usd=.09356286
            budget.llm_calls=3
            budget.max_llm_calls=max(4,budget.max_llm_calls)
            _,cap,_=budget.reserve_call(model='gpt-5.6-terra',purpose='assistant_core_answer_contract_verifier',
                requested_timeout=17,max_output_tokens=3200,messages=captured['messages'],request_payload=self.payload(captured))
            caps.append(cap);bounds.append(budget.call_log[-1]['reserved_input_tokens'])
            self.assertLessEqual(budget.committed_cost_usd,.25+1e-12)
        self.assertEqual(caps[0],1070)
        self.assertEqual(caps[1],3200)
        self.assertLess(bounds[1],bounds[0])
        print(json.dumps({'offline_repair_input_bound_before_after':bounds,'output_tokens_before_after':caps,
                          'request_cost_limit_usd':.25,'prior_committed_usd':.09356286}))


if __name__=='__main__':
    unittest.main(verbosity=2)
