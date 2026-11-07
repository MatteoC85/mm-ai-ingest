"""Exact source/answer references and operational ordering; no provider/network."""
from copy import deepcopy
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask import procedure_review as r

def interval(first=0,last=None):return {'first':first,'last':first if last is None else last}
def endpoint(first=0,last=None,kind='original',block='b1'):
    return {'block_id':block,'kind':kind,'range':interval(first,last)}
def proof(first=0,last=None,source='s1'):return {'source':source,'range':interval(first,last)}

class ReferenceTests(unittest.TestCase):
    ANSWER='1. Release the guide. Keep the supply isolated.\n2. Feed the strip.\n3. Secure the guide before movement.'
    SOURCE='Release the guide. Feed the strip. Secure the guide before movement. Keep the supply isolated.'

    def prepare(self,answer=None,source=None):
        answer=self.ANSWER if answer is None else answer
        source=self.SOURCE if source is None else source
        layout=r.block_layout(answer); aliases=r.review_wire_aliases(layout,['manual:one'])
        catalog=r.reference_catalog(layout,aliases,'[manual:one] source_type=document\n'+source,{'manual:one':source})
        return answer,layout,aliases,catalog,{'manual:one':source}

    def reply(self):
        return {'outcome':'pass','reply_mode':'retain','answer':'','edits':[],
            'citation_ids':['s1'],'covered_facets':['feed'],'missing_facets':[],
            'covered_answer_types':['ordered_actions'],'missing_answer_types':[],'missing_list_items':[],
            'operation_boundary':{'complete':True,'checked_blocks':['b1','b2','b3'],
                'precedence':[{'action':endpoint(),'next':endpoint(block='b3'),'proof':proof(2)}],
                'change_edges':[{'edge_index':0,'resolution':'restored'}]}}

    def execute(self,value,prepared=None,mutate=None):
        answer,layout,aliases,catalog,sources=prepared or self.prepare()
        before=deepcopy(value)
        decoded,binding=r.decode_reference_reply(value,aliases,layout=layout,catalog=catalog)
        resolved=r.resolve_reply(decoded,answer=answer,layout=layout,required_facets=['feed'],
            required_types=['ordered_actions'],source_ordered=True)
        if mutate:mutate(resolved,sources)
        out=r.verify_operation_boundary(resolved,layout=layout,source_texts=sources,
            source_ordered=True,reference_proof=binding)
        self.assertEqual(value,before)
        return out

    def test_roundtrip_units_keep_unicode_negation_decimals_and_newlines(self):
        for text in ('Do not change 2.5 psi; keep “STOP”.\r\n待機してください。', 'x'*501,
                     'one two '*90,'Do not open then close','A.\n\n B? '):
            units=r.reference_units(text)
            self.assertEqual(''.join(u[2] for u in units),text)
            self.assertTrue(all(end-start<=160 for start,end,_ in units))
            for start,end,value in units:self.assertEqual(text[start:end],value)

    def test_retain_exact_answer_and_sources(self):
        result=self.execute(self.reply())
        self.assertEqual(result['outcome'],'pass');self.assertEqual(result['answer'],self.ANSWER)
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertEqual(result['citation_ids'],['manual:one'])

    def test_edited_original_interval_replacement_is_emitted_once(self):
        value=self.reply();value.update(outcome='rewrite',reply_mode='edit')
        value['edits']=[{'block_id':'b1','original':interval(),'replacement_parts':['Release the marked guide. ']}]
        value['operation_boundary']['precedence'][0]['action']=endpoint(kind='replacement')
        result=self.execute(value)
        self.assertEqual(result['answer'],self.ANSWER.replace('Release the guide.','Release the marked guide.'))
        self.assertTrue(result['operation_boundary_validation']['complete'])

    def test_segmentation_only_same_block_proves_actual_order_without_text_change(self):
        answer='1. Release the guide then secure the guide before movement'
        prepared=self.prepare(answer,answer[3:]);value=self.reply()
        value.update(outcome='rewrite',reply_mode='edit',edits=[{'block_id':'b1','original':interval(),
            'replacement_parts':['Release the guide ', 'then secure the guide before movement']}])
        value['operation_boundary']['checked_blocks']=['b1']
        value['operation_boundary']['precedence']=[{'action':endpoint(kind='replacement'),
            'next':endpoint(1,kind='replacement'),'proof':proof()}]
        result=self.execute(value,prepared)
        self.assertEqual(result['answer'],answer)
        self.assertTrue(result['operation_boundary_validation']['complete'])

    def test_same_block_reversed_overlap_and_self_restoration_fail(self):
        answer='1. Release the guide. Secure the guide.';prepared=self.prepare(answer,answer[3:])
        for before,after in ((endpoint(1),endpoint(0)),(endpoint(0,1),endpoint(1)),(endpoint(),endpoint())):
            value=self.reply();value['operation_boundary']['checked_blocks']=['b1']
            value['operation_boundary']['precedence']=[{'action':before,'next':after,'proof':proof(0,1)}]
            out=self.execute(value,prepared)
            self.assertEqual(out['outcome'],'no_sources')
            self.assertEqual(out['reason'],'operation_reference_closure_order_invalid')

    def test_documented_terminal_hold_requires_that_exact_edge_resolution(self):
        answer='1. Keep the supply isolated.';prepared=self.prepare(answer,answer[3:])
        value=self.reply();value['operation_boundary']['checked_blocks']=['b1']
        value['operation_boundary']['precedence']=[{'action':endpoint(),'next':endpoint(),'proof':proof()}]
        value['operation_boundary']['change_edges'][0]['resolution']='documented_terminal_state'
        self.assertEqual(self.execute(value,prepared)['outcome'],'pass')
        # An identically worded OTHER edge cannot lend its terminal exemption.
        value['operation_boundary']['precedence'].append(deepcopy(value['operation_boundary']['precedence'][0]))
        value['operation_boundary']['change_edges'].append({'edge_index':1,'resolution':'restored'})
        self.assertEqual(self.execute(value,prepared)['outcome'],'no_sources')

    def test_deleted_or_partly_replaced_original_reference_fails(self):
        for ref in (endpoint(),endpoint(0,1)):
            value=self.reply();value.update(outcome='rewrite',reply_mode='edit')
            value['edits']=[{'block_id':'b1','original':interval(),'replacement_parts':['Release the marked guide. ']}]
            value['operation_boundary']['precedence'][0]['action']=ref
            with self.assertRaisesRegex(ValueError,'procedure_reference_deleted_unit'):self.execute(value)

    def test_invalid_ranges_boolean_unknown_kind_or_cross_block_replacement_fail(self):
        for ref in ({'block_id':'b1','kind':'original','range':{'first':True,'last':0}},endpoint(-1),endpoint(0,50),
                    endpoint(kind='guessed'),endpoint(kind='replacement')):
            value=self.reply();value['operation_boundary']['precedence'][0]['action']=ref
            with self.assertRaises(ValueError):self.execute(value)

    def test_wrong_source_alias_and_changed_source_binding_fail(self):
        value=self.reply();value['operation_boundary']['precedence'][0]['proof']['source']='other-tenant'
        with self.assertRaisesRegex(ValueError,'procedure_review_wire_alias_unknown'):self.execute(value)
        out=self.execute(self.reply(),mutate=lambda out,sources:sources.update({'manual:one':'Changed source.'}))
        self.assertEqual(out['reason'],'operation_reference_source_binding_changed')

    def test_catalog_rejects_unemitted_or_changed_source_body(self):
        _,layout,aliases,_,sources=self.prepare()
        for wire in ('[other-tenant] source_type=document\n'+self.SOURCE,
                     '[manual:one] source_type=document\nInvented content.',
                     '[manual:one] source_type=document\n'+self.SOURCE+'\n[manual:one] duplicate\ntext'):
            with self.assertRaises(ValueError):r.reference_catalog(layout,aliases,wire,sources)

    def test_answer_mutation_after_same_call_decode_is_rejected(self):
        result=self.execute(self.reply(),mutate=lambda out,sources:out.update(answer=out['answer'].replace('isolated','energized')))
        self.assertEqual(result['reason'],'operation_reference_answer_binding_changed')

    def test_deleting_a_clause_keeps_other_units_and_cannot_reference_deleted_text(self):
        value=self.reply();value.update(outcome='rewrite',reply_mode='edit')
        value['edits']=[{'block_id':'b1','original':interval(0,1),'replacement_parts':['Release the guide. ']}]
        value['operation_boundary']['precedence'][0]['action']=endpoint(kind='replacement')
        # This example has no source-note binding; real protected notes are checked by resolve_reply.
        result=self.execute(value)
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertNotIn('Keep the supply isolated.',result['answer'])

    def test_preserved_source_notes_cannot_be_deleted_by_reference_edit(self):
        answer,layout,aliases,catalog,_=self.prepare()
        value=self.reply();value.update(outcome='rewrite',reply_mode='edit')
        value['edits']=[{'block_id':'b1','original':interval(0,1),'replacement_parts':['Release the guide. ']}]
        value['operation_boundary']['precedence'][0]['action']=endpoint(kind='replacement')
        decoded,_=r.decode_reference_reply(value,aliases,layout=layout,catalog=catalog)
        result=r.resolve_reply(decoded,answer=answer,layout=layout,required_facets=['feed'],
            required_types=['ordered_actions'],source_ordered=True,
            source_notes=({'citation_id':'step:1','source_number':'1','texts':('Keep the supply isolated.',)},))
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(result['reason'],'source_step_note_binding_incomplete')

    def test_every_edited_block_and_each_block_review_is_still_required(self):
        for mutation in ('missing_check','duplicate_check','untouched_edit','unresolved'):
            value=self.reply()
            if mutation=='missing_check':value['operation_boundary']['checked_blocks'].pop()
            elif mutation=='duplicate_check':value['operation_boundary']['checked_blocks']=['b1','b1','b3']
            elif mutation=='unresolved':value['operation_boundary']['change_edges'][0]['resolution']='unresolved'
            else:
                value.update(outcome='rewrite',reply_mode='edit')
                value['edits']=[{'block_id':'b2','original':interval(),'replacement_parts':['Feed the flat strip.']}]
            self.assertEqual(self.execute(value)['outcome'],'no_sources')

    def test_legacy_literal_proof_is_not_a_runtime_fallback(self):
        value=self.reply();value['operation_boundary']['precedence'][0]['proof']='Secure the guide.'
        with self.assertRaisesRegex(ValueError,'procedure_reference_shape_invalid'):self.execute(value)
        value=self.reply();value.update(outcome='rewrite',reply_mode='edit')
        value['edits']=[{'block_id':'b1','old_text':'Release','new_text':'Open'}]
        with self.assertRaisesRegex(ValueError,'procedure_reference_edit_invalid'):self.execute(value)

    def test_semantic_rejection_is_not_changed_into_acceptance(self):
        value=self.reply();value.update(outcome='no_sources',reply_mode='reject',reason='unsafe_or_unsupported')
        result=self.execute(value)
        self.assertEqual(result['outcome'],'no_sources');self.assertEqual(result['reason'],'unsafe_or_unsupported')

    def test_references_allow_complete_source_span_beyond_old_wire_quote_limit(self):
        source=self.SOURCE+' Context remains literal.'*30;prepared=self.prepare(source=source)
        value=self.reply();value['operation_boundary']['precedence'][0]['proof']=proof(0,len(prepared[3]['sources']['s1']['units'])-1)
        result=self.execute(value,prepared)
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertGreater(len(result['operation_boundary']['precedence'][0]['source_quote']),400)

    def test_schema_has_typed_references_and_no_literal_proof_or_old_text(self):
        _,layout,aliases,catalog,_=self.prepare()
        legacy={'name':'fixture','schema':{'properties':{'citation_ids':{'type':'array','items':{'type':'string'}}},'required':['citation_ids']}}
        schema=r.review_schema(legacy,layout,[],[],operation_boundary=True,source_ordered=True,
            operation_source_ids=['manual:one'],wire_aliases=aliases,reference_catalogue=catalog)['schema']
        edits=schema['properties']['edits']['items']['anyOf']
        for edit in edits:self.assertEqual(set(edit['properties']),{'block_id','original','replacement_parts'})
        edge=schema['properties']['operation_boundary']['properties']['precedence']['items']['properties']
        self.assertEqual(set(edge),{'action','next','proof'})
        endpoints=schema['$defs'][edge['action']['$ref'].split('/')[-1]]['anyOf']
        self.assertEqual({kind for item in endpoints for kind in item['properties']['kind']['enum']},
                         {'original','replacement'})
        for item in endpoints:self.assertEqual(set(item['properties']),{'block_id','kind','range'})

    def test_prompt_views_reconstruct_bodies_and_keep_one_source_identity(self):
        answer,_,_,catalog,_=self.prepare()
        source_view,blocks=r.reference_views(catalog)
        self.assertEqual(''.join(b['heading']+''.join(u[1] for u in b['units'])+b['separator'] for b in blocks),answer)
        units=json.loads(source_view.split('SOURCE_UNITS s1:',1)[1])
        self.assertEqual(''.join(u[1] for u in units),self.SOURCE)
        self.assertEqual(source_view.count('[manual:one]'),1)

class ActualRedactorTests(unittest.TestCase):
    ANSWER, SOURCE = ReferenceTests.ANSWER, ReferenceTests.SOURCE
    prepare, reply, execute = ReferenceTests.prepare, ReferenceTests.reply, ReferenceTests.execute
    @classmethod
    def setUpClass(cls):
        cls.guards=ExitStack()
        cls.guards.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        def denied(*args,**kwargs):raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')
        for name in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guards.enter_context(patch(name,denied))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls):cls.guards.close()

    def test_actual_redactor_whitespace_preserves_resolved_order_and_proof(self):
        answer=self.ANSWER.replace('1. ','1.\t').replace('Release the','Release  the').replace('Secure the','Secure\tthe')
        result=self.execute(self.reply(),self.prepare(answer=answer),
            mutate=lambda out,sources:out.update(answer=self.m._assistant_core_redact_internal_text(out['answer'])))
        self.assertTrue(result['operation_boundary_validation']['complete'])
        self.assertEqual(result['answer'],self.ANSWER)

    def test_actual_redactor_content_removal_inside_span_cannot_keep_old_proof(self):
        answer=self.ANSWER.replace('Release the guide.','Release step:1234567890123x123456789012345678 the guide.')
        self.assertNotEqual(self.m._assistant_core_redact_internal_text(answer),answer)
        result=self.execute(self.reply(),self.prepare(answer=answer),
            mutate=lambda out,sources:out.update(answer=self.m._assistant_core_redact_internal_text(out['answer'])))
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(result['reason'],'operation_reference_answer_binding_changed')


if __name__=='__main__':unittest.main(verbosity=2)
