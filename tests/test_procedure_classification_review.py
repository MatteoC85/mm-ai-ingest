"""Mixed procedure/numeric classification must receive the required boundary review.

Uses actual verifier schema, reply decoding, operation proof and final validator.
Model replies and source/admission callbacks are synthetic; external I/O is denied.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import execution, procedure_review, validation

def denied(*a, **kw):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')

class ProcedureClassificationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        cls.guards.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused',
            'MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for name in ('socket.create_connection','socket.socket.connect',
                     'requests.sessions.Session.request','psycopg2.connect'):
            cls.guards.enter_context(patch(name, denied))
        cls.m = importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def run_case(self, *, task='numeric_specification', kind='procedure',
                 mutation=None, stale_structure=False, guarded=True):
        m = self.m
        answer = ('1. Isolate the supply.\n2. Open the guide. Maximum capacity is 750 kg.\n'
                  '3. Close the guide before feeding.')
        source = ('Isolate the supply. Open the guide. Maximum capacity is 750 kg. '
                  'Close the guide before feeding.')
        rows = [{'citation_id':'offline-manual-one','source_type':'document',
                 'bubble_document_id':'offline-manual','page_from':2,'page_to':2,
                 'snippet':source,'chunk_full':source}]
        before = deepcopy(rows)
        facets = ('preparation','capacity','precautions')
        types = (m.REQ_ORDERED_ACTIONS,m.REQ_NUMERIC_VALUE,m.REQ_SAFETY_CONDITIONS)
        request = m.AssistantCoreRequest(query='What preparation and maximum capacity are documented?',
            requested_mode='ask',response_language='en',company_id='offline-company',
            machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision = m.AssistantCoreDecision(request_kind=kind,effective_mode='ask',confidence=.99,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,
            evidence_policy='evidence_required',information_task=task,
            required_facets=facets,required_answer_types=types)
        calls = []
        def provider(messages, **kw):
            calls.append((deepcopy(messages),deepcopy(kw)))
            props = kw['json_schema']['schema']['properties']
            content = messages[-1]['content']
            current = content.split('CURRENT_ANSWER:\n',1)[1].split('\n\nSOURCES:',1)[0]
            value = {'outcome':'pass','answer':current,'citation_ids':['offline-manual-one'],
                'reason':'Synthetic admitted operations and capacity are covered.',
                'covered_facets':list(facets),'missing_facets':[],
                'covered_answer_types':list(types),'missing_answer_types':[],
                'expected_list_items':[],'covered_list_items':[],'missing_list_items':[]}
            if 'reply_mode' in props:
                layout = procedure_review.block_layout(current)
                value.update(reply_mode='retain',answer='',edits=[],operation_boundary={
                    'complete':True,'checked_blocks':[b['block_id'] for b in layout['blocks']],
                    'changes':[{'block_id':'step_2','action_quote':'Open the guide',
                        'resolution':'restored','closing_source_citation_id':'offline-manual-one',
                        'closing_source_quote':'Close the guide before feeding.',
                        'closing_answer_quote':'Close the guide before feeding.'}]})
                if mutation == 'missing_boundary': value.pop('operation_boundary')
                elif mutation == 'foreign_source':
                    value['operation_boundary']['changes'][0]['closing_source_citation_id']='unadmitted-manual'
                elif mutation == 'wrong_source_quote':
                    value['operation_boundary']['changes'][0]['closing_source_quote']='Keep the guide open while feeding.'
                elif mutation == 'closure_before_open':
                    value['operation_boundary']['changes'][0]['closing_answer_quote']='Isolate the supply.'
                elif mutation == 'incomplete_review':
                    value.update(outcome='no_sources', reply_mode='replace', answer='')
                    value['covered_answer_types']=[m.REQ_ORDERED_ACTIONS,m.REQ_NUMERIC_VALUE]
                    value['missing_answer_types']=[m.REQ_SAFETY_CONDITIONS]
            return value,'offline-model'
        eruntime = replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=denied if guarded else None,
            _v13_current_budget=lambda:m._assistant_core_new_budget('ask'),
            _v13_json_models=provider,_assistant_core_candidate_evidence_text=lambda c:c['chunk_full'])
        def verify(**kw):
            if stale_structure and kw.get('repair_context'):
                kw['repair_context']=deepcopy(kw['repair_context'])
                kw['repair_context']['procedure_structure']['answer_sha256']='0'*64
            result = execution.verify_or_repair_answer(**kw,runtime=eruntime)
            if mutation == 'stale_proof' and 'operation_boundary_validation' in result:
                result['operation_boundary_validation']['answer_sha256']='0'*64
            return result
        runtime = replace(m._assistant_core_ask_validation_runtime(),execution_runtime=eruntime,
            final_contract_enabled=True,source_fields=lambda c:{},source_sections=lambda t:{},
            _assistant_core_candidate_source_type=lambda c:c['source_type'],
            _assistant_core_candidate_evidence_text=lambda c:c['chunk_full'],
            _assistant_core_recover_citations=lambda *a,**kw:(rows,[]),
            _assistant_core_redact_internal_text=m._assistant_core_redact_internal_text,
            _assistant_core_media_metadata_only=lambda text,*a:text,
            _assistant_core_should_semantic_verify_answer=lambda d:True,
            _assistant_core_verify_or_repair_answer=verify,
            _assistant_core_answer_contract_check=lambda **kw:{'passed':True,'reason':'offline_deterministic'},
            _sanitize_citations_for_response=lambda items,**kw:list(items),
            _build_rg_links=lambda *a,**kw:[],_v13_current_budget=lambda:None)
        # Final contract after the one permitted repair: rejection must be terminal.
        response={'ok':True,'status':'answered','answer':answer,'citations':rows,'meta':{},
                  '_assistant_core_repair_attempted':True}
        with patch.object(validation,'_inputs',side_effect=lambda r,q,t,d,**kw:(r,t)), \
             patch.object(validation,'_collection',side_effect=lambda q,items,d,**kw:items), \
             patch.object(validation,'_output',side_effect=lambda r,*a,**kw:r), \
             patch.object(execution,'_admit_input',side_effect=lambda q,data,d,**kw:data):
            result=validation.validate_response(response,request,{'citations':rows},decision,runtime=runtime)
        self.assertEqual(rows,before)
        self.assertEqual(len(calls),1)
        return result,calls

    def test_mixed_numeric_procedure_gets_required_boundary_and_passes_final_contract(self):
        for task in ('numeric_specification','document_explanation'):
            with self.subTest(task=task):
                result,calls=self.run_case(task=task)
                self.assertEqual(result['status'],'answered')
                self.assertIn('operation_boundary',calls[0][1]['json_schema']['schema']['required'])
                self.assertEqual(calls[0][1]['max_output_tokens'],3200)
                contract=result['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['passed']);self.assertTrue(contract['complete'])
                self.assertTrue(contract['semantic_answer_bound'])
                boundary=contract['semantic_verifier']['operation_boundary_validation']
                self.assertTrue(boundary['complete'])
                self.assertEqual(boundary['version'],'ask-operation-boundary-v1')
                self.assertIn('750 kg',result['answer'])
                self.assertLess(result['answer'].index('Open the guide'),result['answer'].index('Close the guide'))

    def test_legacy_procedure_information_tasks_keep_same_single_review(self):
        for task in ('procedure_full','procedure_segment'):
            with self.subTest(task=task):
                result,calls=self.run_case(task=task)
                self.assertEqual(result['status'],'answered')
                self.assertIn('operation_boundary',calls[0][1]['json_schema']['schema']['required'])

    def test_absent_foreign_or_wrong_order_boundary_cannot_be_published(self):
        for mutation in ('missing_boundary','foreign_source','wrong_source_quote',
                         'closure_before_open','incomplete_review','stale_proof'):
            with self.subTest(mutation=mutation):
                result,_=self.run_case(mutation=mutation)
                self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])
                self.assertNotEqual(result.get('status'),'answered')

    def test_stale_server_structure_cannot_activate_boundary_protocol(self):
        result,calls=self.run_case(stale_structure=True)
        self.assertNotIn('operation_boundary',calls[0][1]['json_schema']['schema']['properties'])
        self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])

    def test_nonprocedural_numeric_request_keeps_generic_review(self):
        result,calls=self.run_case(kind='factual')
        self.assertNotIn('operation_boundary',calls[0][1]['json_schema']['schema']['properties'])
        self.assertEqual(result['status'],'answered')

    def test_unguarded_runtime_keeps_generic_review(self):
        result,calls=self.run_case(guarded=False)
        self.assertNotIn('operation_boundary',calls[0][1]['json_schema']['schema']['properties'])
        self.assertEqual(result['status'],'answered')

if __name__ == '__main__':
    unittest.main(verbosity=2)
