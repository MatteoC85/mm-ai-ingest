"""Cross-language procedure: one existing reviewer, original safety and scoped sources.

Provider replies are explicit translation fixtures, not evidence of model quality.
The real generation, reference decoder, review and final validator run offline.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution, procedure_review as review
import test_semantic_procedure_family as family


class ProcedureTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        family.SemanticProcedureFamilyTests.setUpClass()
        cls.m = family.SemanticProcedureFamilyTests.m
        cls.fixture = family.SemanticProcedureFamilyTests('run')
    @classmethod
    def tearDownClass(cls): family.SemanticProcedureFamilyTests.tearDownClass()

    def generated(self, *, english_source=False, response_language='en'):
        catalog = family.synthetic_catalog()
        actions = (['Isolate the supply.','Insert the cartridge.','Secure the housing.'] if english_source else
                   ['Isolare la fonte di energia.','Inserire la cartuccia nella guida.','Richiudere e fissare il contenitore.'])
        note = 'Keep the supply isolated.' if english_source else 'Mantenere la fonte di energia isolata.'
        for step,action in zip(catalog[0]['steps'],actions):
            step.update(title=action,description='AZIONE OPERATIVA\n'+action+'\n\nNOTA DI SICUREZZA\n'+note)
        if not english_source:
            catalog[0].update(title='Installazione della cartuccia',short_description='Installare la cartuccia e richiudere il contenitore.')
        h=family.FamilyHarness(self.m,catalog,chosen_key='procedure:alpha',response_language=response_language)
        manual={'citation_id':'offline-manual:p2','bubble_document_id':'offline-manual','source_type':'document',
            'ask_structured_manual_support':True,'ask_manual_support_kind':'operation','exact_machine_scope':True,
            'page_from':2,'page_to':2,'similarity':.9,
            'chunk_full':'For this cartridge installation, do not exceed 6 bar when securing the housing.'}
        result=h.run(_v13_fetch_manual_support_deterministic=lambda **kw:[deepcopy(manual)],
                     _v12_filter_manual_support_to_selected_bundle=lambda **kw:kw['manual_support_citations'])
        return h,result,note,manual

    def test_draft_defers_untranslated_appendix_but_preserves_every_manual_occurrence(self):
        h,result,note,manual=self.generated()
        self.assertEqual(len(h.provider_calls),1)
        self.assertEqual(result['_assistant_core_procedure_translation_language'],'en')
        self.assertEqual(result['_assistant_core_manual_note_sources'],[])
        self.assertNotIn('Manual support',result['answer'])
        self.assertIn(note,result['answer'])
        self.assertIn(manual['citation_id'],[c['citation_id'] for c in result['citations']])
        self.assertIn(manual,[{k:v for k,v in c.items() if k in manual} for c in result['_assistant_core_validation_evidence']])
        self.assertTrue(result['_assistant_core_procedure_family_review_required'])

    def test_same_language_draft_does_not_add_translation_protocol_or_provider(self):
        for english,language in ((True,'en'),(False,'it')):
            with self.subTest(language=language):
                h,result,_,_=self.generated(english_source=english,response_language=language)
                self.assertNotIn('_assistant_core_procedure_translation_language',result)
                self.assertEqual(len(h.provider_calls),1)

    def validated(self, fault=None, *, partial=False):
        m=self.m;h,generated,note,manual=self.generated();calls=[]
        def provider(messages,**kwargs):
            calls.append((messages,kwargs))
            prompt='\n'.join(message['content'] for message in messages)
            self.assertIn('every action, condition and ALL precautions',prompt)
            props=kwargs['json_schema']['schema']['properties']
            self.assertEqual(props['translation_review']['properties']['language']['enum'],['en'])
            blocks=json.JSONDecoder().raw_decode(prompt.split('PROCEDURE_BLOCKS:\n',1)[1])[0]
            aliases=json.JSONDecoder().raw_decode(prompt.split('PROCEDURE_SOURCE_ALIASES:\n',1)[1])[0]
            manual_alias=next(k for k,v in aliases.items() if v==manual['citation_id'])
            manual_units=json.JSONDecoder().raw_decode(prompt.split('SOURCE_UNITS '+manual_alias+':',1)[1])[0]
            self.assertIn(manual['chunk_full'],''.join(x[1] for x in manual_units))
            by_number={int(b['heading'].split('.')[0]):b for b in blocks if b['heading']}
            preamble=next(b for b in blocks if not b['heading'])
            texts={1:'Isolate the supply.',2:'Insert the cartridge through the guide.',
                   3:'Secure the housing. Do not exceed 6 bar.'}
            safety='Warning: Keep the supply isolated.\nOriginal safety wording: '+note
            edits=[{'block_id':preamble['block_id'],'original':{'first':0,'last':len(preamble['units'])-1},
                    'replacement_parts':['Cartridge installation\nInstall the cartridge and secure the housing.\nProcedure']}]
            for number,b in by_number.items():
                translated=texts[number]
                visible_safety=safety
                if fault=='warning_removed' and number==2: visible_safety='Warning: Keep the supply isolated.'
                if fault=='untranslated_precaution': visible_safety='Original safety wording: '+note
                if fault=='wrong_value' and number==3: translated=translated.replace('6 bar','60 bar')
                last=len(b['units'])-1
                if partial:
                    warning_unit=next(i for i,text in b['units'] if 'Attention:' in text)
                    last=warning_unit-1
                    visible_safety='Warning: Keep the supply isolated.\nOriginal source quotation:\n'
                edits.append({'block_id':b['block_id'],'original':{'first':0,'last':last},
                              'replacement_parts':[translated+'\n',visible_safety]})
            source_alias=next(k for k,v in aliases.items() if v.startswith('step:alpha3:'))
            units=json.JSONDecoder().raw_decode(prompt.split('SOURCE_UNITS '+source_alias+':',1)[1])[0]
            edge={'action':{'block_id':by_number[2]['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                  'next':{'block_id':by_number[3]['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                  'proof':{'source':source_alias,'range':{'first':0,'last':len(units)-1}}}
            first_source=next(k for k,v in aliases.items() if v.startswith('step:alpha1:'))
            first_units=json.JSONDecoder().raw_decode(prompt.split('SOURCE_UNITS '+first_source+':',1)[1])[0]
            prerequisite={'action':{'block_id':by_number[1]['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                'next':{'block_id':by_number[2]['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                'proof':{'source':first_source,'range':{'first':0,'last':len(first_units)-1}}}
            parent_alias=next(k for k,v in aliases.items() if v.startswith('procedure:alpha:'))
            parent_units=json.JSONDecoder().raw_decode(prompt.split('SOURCE_UNITS '+parent_alias+':',1)[1])[0]
            scope={'action':{'block_id':preamble['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                'next':{'block_id':by_number[1]['block_id'],'kind':'replacement','range':{'first':0,'last':0}},
                'proof':{'source':parent_alias,'range':{'first':0,'last':len(parent_units)-1}}}
            if fault=='order_reversed': edge['action'],edge['next']=edge['next'],edge['action']
            if fault=='foreign_source': edge['proof']['source']='foreign'
            value={'outcome':'rewrite','reply_mode':'edit','answer':'','edits':edits,
                'reason':'Fixture independently verifies translated instructions, prerequisites and manual pressure limit.',
                'citation_ids':list(aliases),'covered_facets':['cartridge installation'],'missing_facets':[],
                'covered_answer_types':['ordered_actions'],'missing_answer_types':[],
                'expected_list_items':[],'covered_list_items':[],'missing_list_items':[],
                'translation_review':{'language':'en','body_and_safety_complete':fault!='untranslated_precaution'},
                'operation_boundary':{'complete':True,'checked_blocks':[b['block_id'] for b in blocks],
                    'precedence':[edge,prerequisite,scope],'change_edges':[{'edge_index':0,'resolution':'restored'}]}}
            if fault=='missing_language_verdict': value.pop('translation_review')
            if fault=='wrong_language_verdict': value['translation_review']['language']='it'
            calls[-1][1]['offline_fixture_wire']=deepcopy(value)
            return value,'offline-model'
        eruntime=replace(m._assistant_core_ask_execution_runtime(),evidence_admission=family.denied,
            _v13_current_budget=lambda:m._assistant_core_new_budget('ask'),_v13_json_models=provider)
        def verify(**kw):
            with patch.object(execution,'_admit_input',side_effect=lambda q,data,d,**options:data):
                result=execution.verify_or_repair_answer(**kw,runtime=eruntime)
            if fault=='audit_removed' and 'procedure_review' in result:
                result['procedure_review'].pop('translation_review',None)
            return result
        result=self.fixture.validate_bundle(h,generated,verify)
        return result,calls,generated

    def test_real_generation_review_and_final_contract_keep_English_and_original_warnings(self):
        result,calls,generated=self.validated()
        c=result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(result['status'],'answered',c)
        self.assertTrue(c['passed'],c)
        self.assertTrue(c['semantic_answer_bound'])
        self.assertTrue(c['procedure_structure']['sequence_complete'])
        self.assertTrue(c['procedure_structure']['source_notes_present'])
        self.assertTrue(c['semantic_verifier']['procedure_review']['translation_review']['body_and_safety_complete'])
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['max_output_tokens'],3200)
        self.assertEqual(result['answer'].count('Warning: Keep the supply isolated.'),3)
        self.assertEqual(result['answer'].count('Original safety wording: Mantenere la fonte di energia isolata.'),3)
        self.assertIn('Do not exceed 6 bar.',result['answer'])
        self.assertEqual({c['citation_id'] for c in result['citations']},{c['citation_id'] for c in generated['citations']})
        self.assertNotIn('_assistant_core_procedure_translation_language',result)

    def test_unverified_language_cannot_become_success_through_structural_coverage(self):
        for fault in ('missing_language_verdict','wrong_language_verdict','untranslated_precaution','audit_removed'):
            with self.subTest(fault=fault):
                result,calls,_=self.validated(fault)
                c=result['meta']['assistant_core_validation']['answer_contract']
                self.assertFalse(c['passed'],c)
                self.assertEqual(len(calls),1)

    def test_translation_can_reuse_original_warning_units_without_copying_them_in_output(self):
        result,calls,_=self.validated(partial=True)
        _,full_calls,_=self.validated()
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed'],contract)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['answer'].count('Warning: Keep the supply isolated.'),3)
        self.assertEqual(result['answer'].count('Mantenere la fonte di energia isolata.'),3)
        self.assertEqual(result['answer'].count('Original source quotation:'),3)
        self.assertTrue(contract['procedure_structure']['source_notes_present'])
        wire=calls[0][1]['offline_fixture_wire']
        full_wire=full_calls[0][1]['offline_fixture_wire']
        encode=lambda value:json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8')
        self.assertLess(len(encode(wire)),len(encode(full_wire)))
        self.assertNotIn('Mantenere la fonte di energia isolata.',json.dumps(wire['edits']))

    def test_translation_does_not_waive_source_warning_order_or_identity(self):
        for fault in ('warning_removed','order_reversed','foreign_source','wrong_value'):
            with self.subTest(fault=fault):
                result,calls,_=self.validated(fault)
                c=result['meta']['assistant_core_validation']['answer_contract']
                self.assertFalse(c['passed'],c)
                self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main(verbosity=2)
