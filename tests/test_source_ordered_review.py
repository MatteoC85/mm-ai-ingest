"""Source-authored ordinals cannot be replaced by a free-form reviewer list.

All calls are offline. Semantic replies are fixtures; real schema construction,
reply resolution and final validation are exercised without claiming live quality.
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
from machinemind.ask import execution, final_contract, procedure_review as review, validation


def denied(*args, **kwargs):
    raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class SourceOrderedReviewTests(unittest.TestCase):
    ANSWER = ('Qualified personnel only.\n\n2. Isolate and verify zero energy.\n\n'
              '7. Feed material through the guide.\nKeep STOP pressed.\n\n'
              '11. Close the guide and secure the lever.\n')

    def verdict(self, **changes):
        value = {'outcome':'pass','reply_mode':'retain','answer':'','edits':[],
                 'covered_facets':['sequence','safety'],'missing_facets':[],
                 'covered_answer_types':['ordered_actions','safety_conditions'],
                 'missing_answer_types':[],'missing_list_items':[]}
        value.update(changes)
        return value

    def resolve(self, value):
        return review.resolve_reply(value, answer=self.ANSWER, layout=review.block_layout(self.ANSWER),
            required_facets=['sequence','safety'], required_types=['ordered_actions','safety_conditions'],
            source_ordered=True)

    def test_schema_forbids_full_answer_and_preserves_generic_manual_protocol(self):
        legacy={'name':'test','schema':{'properties':{'answer':{'type':'string'}},'required':['answer']}}
        original=deepcopy(legacy)
        schema=review.review_schema(legacy,review.block_layout(self.ANSWER),[],[],source_ordered=True)['schema']
        self.assertEqual(schema['properties']['reply_mode']['enum'],['retain','edit','reject'])
        self.assertEqual(schema['properties']['answer'],{'type':'string','enum':['']})
        generic=review.review_schema(legacy,review.block_layout(self.ANSWER),[],[])['schema']
        self.assertEqual(generic['properties']['reply_mode']['enum'],['replace','retain','edit'])
        self.assertEqual(generic['properties']['answer'],{'type':'string'})
        self.assertEqual(legacy,original)

    def test_extensive_correction_keeps_every_source_ordinal_and_order(self):
        layout=review.block_layout(self.ANSWER)
        edits=[{'block_id':b['block_id'],'text':b['text'].strip()}
               for b in layout['blocks']]
        # Markdown repetitions of the SAME heading are harmless: the server
        # reuses its original heading and all inter-block separators.
        edits[2]['text']='**7.** Feed material through the guide.\nKeep STOP pressed.'
        result=self.resolve(self.verdict(outcome='rewrite',reply_mode='edit',edits=edits))
        self.assertEqual(result['answer'],self.ANSWER)
        self.assertTrue(result['procedure_review']['source_sequence_immutable'])
        self.assertEqual(result['procedure_review']['edited_blocks'],4)

    def test_retain_is_byte_exact_and_full_replacement_cannot_bypass_constraint(self):
        self.assertEqual(self.resolve(self.verdict())['answer'],self.ANSWER)
        for value in (self.verdict(outcome='rewrite',reply_mode='replace',answer='1. Stop.\n2. Feed.\n3. Close.'),
                      {'outcome':'rewrite','answer':'2. Stop.\n7. Feed.\n11. Close.'},
                      self.verdict(answer=self.ANSWER)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)

    def test_reject_stays_rejected_and_cannot_claim_success_or_apply_edits(self):
        value=self.verdict(outcome='no_sources',reply_mode='reject',missing_facets=['safety'])
        result=self.resolve(value)
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(result['answer'],'')
        self.assertEqual(result['missing_facets'],['safety'])
        for changes in ({'outcome':'pass'},{'edits':[{'block_id':'step_1','text':'Skip isolation.'}]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.resolve({**value,**changes})

    def test_missing_obligations_or_changed_sequence_remain_rejected(self):
        for changes in ({'covered_facets':['sequence']},{'missing_answer_types':['safety_conditions']},
                        {'outcome':'rewrite','reply_mode':'edit','edits':[{'block_id':'step_2','text':'8. Feed material.'}]},
                        {'outcome':'rewrite','reply_mode':'edit','edits':[{'block_id':'step_2','text':'Feed.\n9. Restart.'}]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.resolve(self.verdict(**changes))


class WiredSourceOrderedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards=ExitStack()
        cls.guards.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guards.enter_context(patch(name,denied))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):
        cls.guards.close()

    def run_review(self, reject=False, note_mutation=None):
        m=self.m
        numbers=[2,7,11]
        actions=['Isolate and verify zero energy.','Feed material through the guide.','Close the guide and secure the lever.']
        notes=['Keep STOP pressed.','Do not exceed 6 mA.','Keep all guards closed.']
        rows=[{'citation_id':f'offline-step-{n}','source_type':'step','fields':{
                  'step_number':n,'title':f'Operation {n}','description':a,'notes':note},
               'chunk_full':a+'\n'+note} for n,a,note in zip(numbers,actions,notes)]
        answer='\n\n'.join(f'{n}. {a}\n{note}' for n,a,note in zip(numbers,actions,notes))
        request=m.AssistantCoreRequest(query='Describe the procedure and precautions.',requested_mode='ask',
            response_language='en',company_id='offline-company',machine_id='offline-machine',ai_scope='machine_all',top_k=8)
        decision=m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,effective_mode='ask',confidence=.99,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,evidence_policy='evidence_required',
            information_task=m.INFO_PROCEDURE_SEGMENT,required_facets=('sequence','safety'),
            required_answer_types=(m.REQ_ORDERED_ACTIONS,m.REQ_SAFETY_CONDITIONS))
        calls=[]
        def provider(messages,**kwargs):
            calls.append((messages,kwargs))
            result={'outcome':'no_sources' if reject else 'rewrite','reply_mode':'reject' if reject else 'edit',
                'answer':'','edits':[] if reject else [
                    {'block_id':f'step_{i+1}','text':a+'\n'+note} for i,(a,note) in enumerate(zip(actions,notes))],
                'reason':'source insufficient' if reject else 'body reviewed','citation_ids':[r['citation_id'] for r in rows],
                'covered_facets':['sequence','safety'],'missing_facets':['safety'] if reject else [],
                'covered_answer_types':[m.REQ_ORDERED_ACTIONS,m.REQ_SAFETY_CONDITIONS],
                'missing_answer_types':[],'expected_list_items':[],'covered_list_items':[],'missing_list_items':[]}
            if note_mutation == 'delete':
                result['edits'][1]['text']=actions[1]
            elif note_mutation == 'swap':
                result['edits'][0]['text']=actions[0]+'\n'+notes[1]
                result['edits'][1]['text']=actions[1]+'\n'+notes[0]
            return result,'offline-model'
        eruntime=replace(m._assistant_core_ask_execution_runtime(),evidence_admission=denied,
            _v13_current_budget=lambda:m._assistant_core_new_budget('ask'),_v13_json_models=provider,
            _v13_sources_block=lambda *a,**kw:'\n'.join(r['chunk_full'] for r in rows))
        runtime=replace(m._assistant_core_ask_validation_runtime(),execution_runtime=eruntime,
            final_contract_enabled=True,source_fields=lambda c:c.get('fields',{}),source_sections=lambda text:{},
            _assistant_core_candidate_source_type=lambda c:c['source_type'],
            _assistant_core_candidate_evidence_text=lambda c:c['chunk_full'],
            _assistant_core_recover_citations=lambda *a,**kw:(rows,[]),
            _assistant_core_redact_internal_text=lambda text:text,
            _assistant_core_media_metadata_only=lambda text,*a:text,
            _assistant_core_should_semantic_verify_answer=lambda d:True,
            _assistant_core_verify_or_repair_answer=lambda **kw:execution.verify_or_repair_answer(**kw,runtime=eruntime),
            _assistant_core_answer_contract_check=lambda **kw:{'passed':True,'reason':'offline_deterministic'},
            _sanitize_citations_for_response=lambda items,**kw:list(items),_build_rg_links=lambda *a,**kw:[],
            _v13_current_budget=lambda:None)
        response={'ok':True,'status':'answered','answer':answer,'citations':rows,'meta':{},
                  '_assistant_core_source_ordered_procedure':True}
        with patch.object(validation,'_inputs',side_effect=lambda r,q,t,d,**kw:(r,t)), \
             patch.object(validation,'_collection',side_effect=lambda q,items,d,**kw:items), \
             patch.object(validation,'_output',side_effect=lambda r,*a,**kw:r), \
             patch.object(execution,'_admit_input',side_effect=lambda q,data,d,**kw:data):
            result=validation.validate_response(response,request,{'citations':rows},decision,runtime=runtime)
        return result,calls,answer

    def test_full_validation_preserves_sparse_source_sequence_and_independent_review(self):
        result,calls,answer=self.run_review()
        self.assertEqual(len(calls),1)
        props=calls[0][1]['json_schema']['schema']['properties']
        self.assertEqual(props['reply_mode']['enum'],['retain','edit','reject'])
        self.assertEqual(props['answer']['enum'],[''])
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed'])
        self.assertTrue(contract['complete'])
        self.assertTrue(contract['procedure_structure']['sequence_complete'])
        self.assertTrue(contract['procedure_structure']['source_notes_present'])
        self.assertEqual(contract['procedure_structure']['visible_numbers'],[2,7,11])
        self.assertEqual(result['answer'].splitlines(),[x for x in answer.splitlines() if x])
        self.assertNotIn('_assistant_core_needs_repair',result)

    def test_complete_coverage_cannot_delete_or_swap_source_safety_notes(self):
        for mutation in ('delete','swap'):
            with self.subTest(mutation=mutation):
                result,calls,_=self.run_review(note_mutation=mutation)
                self.assertEqual(len(calls),1)
                contract=result['meta']['assistant_core_validation']['answer_contract']
                self.assertFalse(contract['passed'])
                self.assertEqual(contract['semantic_verifier']['outcome'],'no_sources')
                self.assertEqual(contract['semantic_verifier']['reason'],'source_step_note_binding_incomplete')
                self.assertFalse(contract['semantic_verifier']['procedure_review']['source_note_bindings_complete'])

    def test_native_renderer_keeps_distinct_structured_notes_inside_their_step(self):
        from machinemind.presentation import responses
        row={'citation_id':'offline-step-7','source_type':'step','fields':{
            'step_number':7,'title':'Feed material','description':'Feed through the guide.',
            'notes':('Documented operating precaution. '*40)+'Do not exceed 6 mA.\nKeep STOP pressed.'}}
        runtime=replace(self.m._RESPONSE_PRESENTATION_RUNTIME(),
            procedure_ui_merge_sources=lambda rows:rows, evidence_role=lambda row:row['source_type'],
            procedure_ui_fields=lambda row:row['fields'],
            procedure_ui_sections=lambda text:{'instruction':text,'safety':'Keep hands clear.'})
        model=responses.build_structured_procedure_ui_model(structured_citations=[row],
            manual_support_citations=[],grounded_points=[],response_language='en',
            _runtime=runtime,preserve_complete=True)
        answer=responses.procedure_ui_model_to_text(model,response_language='en',_runtime=runtime,preserve_complete=True)
        body=review.block_layout(answer)['blocks'][-1]['text']
        self.assertTrue(body.startswith('7. '))
        self.assertIn('Keep hands clear.',body)
        self.assertGreater(len(row['fields']['notes']),800)
        self.assertIn(row['fields']['notes'],body)
        self.assertEqual(body.count('Do not exceed 6 mA.'),1)
        verdict={'outcome':'pass','reply_mode':'retain','answer':'','edits':[],
            'covered_facets':['safety'],'missing_facets':[],
            'covered_answer_types':['safety_conditions'],'missing_answer_types':[],'missing_list_items':[]}
        result=review.resolve_reply(verdict,answer=answer,layout=review.block_layout(answer),
            required_facets=['safety'],required_types=['safety_conditions'],source_ordered=True,
            source_notes=({'citation_id':row['citation_id'],'source_number':'7',
                           'texts':('Keep hands clear.',row['fields']['notes'])},))
        self.assertEqual(result['outcome'],'pass')
        self.assertTrue(result['procedure_review']['source_note_bindings_complete'])
        self.assertEqual(result['answer'],answer)

    def test_full_validation_does_not_promote_rejected_review(self):
        result,calls,_=self.run_review(reject=True)
        self.assertEqual(len(calls),1)
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertEqual(contract['reason'],'semantic_contract_rejected_grounded_answer')


if __name__=='__main__':
    unittest.main(verbosity=2)
