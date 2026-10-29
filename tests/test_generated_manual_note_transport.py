"""Real structured generation/finalization must carry note proof to validation.

Synthetic sources and semantic replies only; no model/network/database access.
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

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import validation, execution, procedure_review as review
from test_semantic_procedure_family import FamilyHarness, synthetic_catalog
import test_semantic_procedure_family as family_fixtures


def deny(*a,**kw):raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


class GeneratedManualNoteTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=ExitStack()
        cls.guard.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for target in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guard.enter_context(patch(target,deny))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):cls.guard.close()

    def generated(self,*,translate=False,foreign=False,aggregate=False):
        note=('Inserire la cartuccia attraverso la guida. Ripristinare la guida al termine.' if translate else
              'Insert the cartridge through the guide. Restore the guide afterwards.')
        manual={'citation_id':'manual:route','bubble_document_id':'manual','source_type':'document',
            'chunk_full':'Install and secure the cartridge. '+note,'ask_structured_manual_support':True,
            'llm_operation_note':note,'llm_operation_citation_ids':['manual:route']}
        manuals=[manual]
        if aggregate:
            manual['llm_operation_citation_ids'].append('manual:closure')
            manuals.append({**deepcopy(manual),'citation_id':'manual:closure'})
        h=FamilyHarness(self.m,synthetic_catalog(),chosen_key='procedure:alpha')
        builder=h.tasks._build_structured_procedure_ui_model
        def build(**kwargs):
            model=builder(**kwargs)
            if foreign:
                # Simulate a corrupted producer binding BEFORE the real broken
                # finalization boundary, never inject evidence into validation.
                model['manual_note_sources'][0]['citation_ids']=['foreign:manual']
            return model
        result=h.run(_v13_fetch_manual_support_deterministic=lambda **kw:deepcopy(manuals),
            _build_structured_procedure_ui_model=build,
            _finalize_ask_response_for_ui=lambda r,**kw:self.m._finalize_ask_response_for_ui(r,**kw,preserve_complete=True))
        self.assertNotIn('_assistant_ui_model',result)
        self.assertIn(note,result['answer'])
        fixture=family_fixtures.SemanticProcedureFamilyTests('run');fixture.m=self.m
        return h,result,note,fixture

    def verifier(self,*,translate=False,omit=False,reject=False):
        def verify(**kw):
            answer=kw['answer'];layout=review.block_layout(answer)
            omitted='manual:route' if omit is True else omit
            ids=[c['citation_id'] for c in kw['candidates'] if not omitted or c['citation_id']!=omitted]
            value={'outcome':'no_sources' if reject else 'pass','reply_mode':'reject' if reject else 'retain',
                'answer':'','edits':[],'covered_facets':[] if reject else ['cartridge installation'],
                'missing_facets':['cartridge installation'] if reject else [],
                'covered_answer_types':[self.m.REQ_ORDERED_ACTIONS],'missing_answer_types':[],
                'missing_list_items':[],'citation_ids':ids,
                'operation_boundary':{'complete':not reject,'checked_blocks':[b['block_id'] for b in layout['blocks']],
                    'changes':[],'precedence':[]}}
            if translate and not reject:
                note='Inserire la cartuccia attraverso la guida. Ripristinare la guida al termine.'
                projected,removed=self.m._assistant_core_filter_unsupported_claim_sentences(note,note)
                self.assertEqual(removed,[])
                last=layout['blocks'][-1]
                text='Insert the cartridge through the guide. Restore the guide afterwards.'
                value.update(outcome='rewrite',reply_mode='edit',
                    edits=[{'block_id':last['block_id'],'text':last['text'].replace(projected,text)}])
                value['operation_boundary']['precedence']=[{'before_block_id':last['block_id'],
                    'before_answer_quote':'Insert the cartridge through the guide.',
                    'after_block_id':last['block_id'],'after_answer_quote':'Restore the guide afterwards.',
                    'source_citation_id':'manual:route','source_quote':note}]
            out=review.resolve_reply(value,answer=answer,layout=layout,required_facets=['cartridge installation'],
                required_types=[self.m.REQ_ORDERED_ACTIONS],source_ordered=True,
                source_notes=kw['repair_context']['procedure_notes'])
            return review.verify_operation_boundary(out,layout=layout,
                source_texts={c['citation_id']:self.m._assistant_core_candidate_evidence_text(c) for c in kw['candidates']},
                source_ordered=True)
        return verify

    def assert_public(self,result):
        from assistant_core_v2 import _finish_ask_validation
        final=self.m._finalize_ask_response_for_ui(_finish_ask_validation(result),language='en',preserve_complete=True)
        self.assertFalse(any(k.startswith('_assistant_core_') for k in final))
        self.assertNotIn('_assistant_ui_model',final)
        return final

    def test_retain_restores_manual_reference_after_actual_generation_finalizer(self):
        h,generated,note,f=self.generated()
        result=f.validate_bundle(h,generated,self.verifier(omit=True))
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed'],contract)
        self.assertIn('manual_note_bindings',contract)
        self.assertTrue(contract['manual_note_bindings']['complete'])
        self.assertIn('manual:route',{c['citation_id'] for c in result['citations']})
        self.assert_public(result)

    def test_translation_keeps_note_source_and_no_private_output(self):
        h,generated,note,f=self.generated(translate=True,aggregate=True)
        result=f.validate_bundle(h,generated,self.verifier(translate=True))
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed'],contract)
        self.assertIn('manual_note_bindings',contract)
        self.assertTrue(contract['manual_note_bindings']['complete'])
        self.assertIn('Restore the guide afterwards',result['answer'])
        self.assert_public(result)

    def test_foreign_producer_binding_fails_closed_without_repair(self):
        h,generated,note,f=self.generated(foreign=True)
        result=f.validate_bundle(h,generated,self.verifier())
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertTrue(contract['manual_note_binding_invalid'])
        self.assertEqual(result['status'],'no_sources')
        self.assertFalse(result.get('_assistant_core_repair_needed'))
        self.assert_public(result)

    def repaired(self,*,omit=False):
        h,generated,note,f=self.generated(translate=True,aggregate=True)
        rows=generated['_assistant_core_validation_evidence']
        first=f.validate_bundle(h,generated,self.verifier(reject=True))
        self.assertTrue(first['_assistant_core_repair_needed'])
        self.assertNotIn('_assistant_core_manual_note_sources',first)
        self.assertTrue(first['_assistant_core_repair_context']['manual_note_sources'])
        request,decision=f.request_decision(h)
        runtime=replace(self.m._assistant_core_ask_execution_runtime(),evidence_admission=deny,
            _v13_current_budget=lambda:self.m._assistant_core_new_budget('ask'),
            _assistant_core_verify_or_repair_answer=self.verifier(translate=True,omit='manual:closure' if omit else False))
        with patch.object(execution,'_admit_input',side_effect=lambda q,data,d,**kw:data):
            repaired=execution.repair_response(response=first,request=request,retrieval={'citations':rows},decision=decision,runtime=runtime)
        final=f.validate_bundle(h,repaired,deny)
        self.assert_public(final)
        return final

    def test_one_bounded_repair_preserves_original_note_binding_through_translation(self):
        result=self.repaired()
        self.assertEqual(result['status'],'answered')
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed'],contract)
        self.assertTrue(contract['semantic_verifier']['manual_note_bindings']['complete'])
        self.assertIn('manual:route',{c['citation_id'] for c in result['citations']})

    def test_repair_cannot_drop_binding_required_by_translated_note(self):
        result=self.repaired(omit=True)
        self.assertEqual(result['status'],'no_sources')
        self.assertEqual(result['meta']['assistant_core_repair']['reason'],'manual_note_binding_invalid')
        self.assertEqual(result['citations'],[])


if __name__=='__main__':unittest.main(verbosity=2)
