"""Mandatory operating detail must survive the bounded manual support top-k.

All sources are synthetic. Recall tests do not manufacture a semantic PASS.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machinemind.retrieval import document_readers as readers
from machinemind.ask import execution, validation, review_evidence
from test_manual_support_segment_recall import source
import test_source_ordered_review as wired_fixtures


def deny(*args, **kwargs): raise AssertionError('OFFLINE_EXTERNAL_IO_DENIED')


def facet(text, **changes):
    return {'facet':text, 'must_cover':True, 'answer_type':'ordered_actions', **changes}


class ManualSupportFacetRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=ExitStack()
        cls.guard.enter_context(patch.dict(os.environ, {'MM_USAGE_ENFORCEMENT':'off',
            'AI_INTERNAL_SECRET':'offline-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        for target in ('socket.create_connection','socket.socket.connect','requests.sessions.Session.request','psycopg2.connect'):
            cls.guard.enter_context(patch(target,deny))
        cls.m=importlib.import_module('main')

    @classmethod
    def tearDownClass(cls): cls.guard.close()

    def setUp(self):
        self.pages=[
            {'citation_id':'main', 'chunk_full':'Insert the tube through the guide and close the latch.'},
            {'citation_id':'redundant', 'chunk_full':'Adjust the guide latch position for tube insertion.'},
            {'citation_id':'route', 'chunk_full':'Route the tube through the spiral channel then the elbow and exit collar.'},
            {'citation_id':'restore', 'chunk_full':'Restore the inlet retainer with the collar disengaged.'}]
        self.plan={'information_task':'procedure_segment', 'facet_queries':[
            facet('Tube route through spiral channel elbow exit collar')]}

    def select(self, cap=2):
        return readers._manual_support_facet_selection(self.pages,self.plan,cap,
            self.m._ask_structured_manual_support_terms)

    def test_required_detail_replaces_redundant_page_and_preserves_main_source(self):
        before=deepcopy((self.pages,self.plan))
        self.assertEqual([p['citation_id'] for p in self.select()],['main','route'])
        self.assertEqual((self.pages,self.plan),before)

    def test_english_facet_can_use_existing_italian_query_variant(self):
        self.pages[2]['chunk_full']='Far passare il tubo nel canale a spirale, poi nel gomito e nel collare di uscita.'
        self.plan['facet_queries'][0]['dense_queries']=['percorso tubo canale spirale gomito collare uscita']
        self.assertEqual([p['citation_id'] for p in self.select()],['main','route'])

    def test_narrow_facet_already_in_main_page_does_not_expand_to_other_operation(self):
        self.plan['facet_queries']=[facet('Close the guide latch')]
        self.assertEqual(self.select(),self.pages[:2])

    def test_optional_or_safety_only_facets_do_not_displace_operating_support(self):
        for changes in ({'must_cover':False},{'answer_type':'safety_conditions'}):
            with self.subTest(changes=changes):
                self.plan['facet_queries']=[facet('Tube route through spiral channel elbow exit collar',**changes)]
                self.assertEqual(self.select(),self.pages[:2])

    def test_no_short_label_or_substring_can_supply_missing_facet(self):
        self.plan['facet_queries']=[facet('A B C'),facet('seal valve clamp')]
        self.pages[2]['chunk_full']='Sealed valveless equipment has clamping components.'
        self.assertEqual(self.select(),self.pages[:2])

    def test_multiple_facets_obey_existing_cap_and_leave_absent_detail_unasserted(self):
        self.plan['facet_queries'] += [facet('Restore inlet retainer collar disengaged'),facet('Calibrate optical sensor voltage')]
        self.assertEqual([p['citation_id'] for p in self.select(3)],['main','route','restore'])
        self.assertEqual(len(self.select(2)),2)
        self.assertEqual(self.select(1),self.pages[:1])
        self.assertEqual(self.select(0),[])
        self.assertTrue(all('covered_facets' not in row for row in self.pages))

    def test_nonprocedural_or_missing_contract_keeps_previous_top_k(self):
        for plan in ({},{'information_task':'document_explanation','facet_queries':self.plan['facet_queries']}):
            with self.subTest(plan=plan):
                self.plan=plan
                self.assertEqual(self.select(),self.pages[:2])

    def test_real_reader_retains_facet_source_with_same_scope_query_and_limits(self):
        rows=[source('step','open','Open housing','Open the housing and release the inlet guide latch.',3),
              source('step','insert','Insert tube','Insert the tube through the guide. Route through the spiral channel.',4),
              source('step','close','Close housing','Close the latch and restore the guide housing.',5)]
        pages=[('synthetic-manual','offline-machine',2,
            'Open housing and release inlet guide latch. Insert tube through the guide. Close latch and restore guide housing.'),
            ('synthetic-manual','offline-machine',3,
            'Open housing release inlet guide latch insert tube guide close latch restore housing.'),
            ('synthetic-manual','offline-machine',8,
            'Route the tube through the spiral channel, then the elbow, finally the exit collar.')]
        queries=[]
        class Cursor:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def execute(self,sql,params):queries.append((sql,params))
            def fetchall(self):return deepcopy(pages)
        class Connection:
            def cursor(self):return Cursor()
            def close(self):pass
        with patch.object(self.m,'_db_conn',lambda:Connection()), \
             patch.object(self.m,'_v12_filter_linkable_manual_support',lambda company,items:items):
            result=self.m._v13_fetch_manual_support_deterministic(company_id='offline-company',
                machine_id='offline-machine',q='How do I insert the tube?',planner=self.plan,structured_citations=rows)
        self.assertEqual([c['page_from'] for c in result],[2,8])
        filtered=self.m._v12_filter_manual_support_to_selected_bundle(q='How do I insert the tube?',
            structured_citations=rows,manual_support_citations=result)
        self.assertEqual([c['page_from'] for c in filtered],[2,8])
        self.assertEqual(len(queries),1)
        sql,params=queries[0]
        self.assertIn('WHERE company_id=%s',sql)
        self.assertIn("(machine_id=%s OR machine_id IS NULL OR machine_id='')",sql)
        self.assertEqual(params[1:4],('offline-company','offline-machine','offline-machine'))
        self.assertLessEqual(params[-1],300)
        self.assertLessEqual(len(result),self.m.ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS)

    def test_real_packet_and_validator_preserve_complement_after_grounded_edit(self):
        # A semantic reply is simulated, but packet construction, wire decoding,
        # quote/order proof, both claim filters and final citation projection run.
        # This proves transport/binding, not independent live answer quality.
        main_note='Release the guide before feeding. Restore the guide after feeding.'
        detail='Pass material through fixed guides P and Q, then the pivot arm R to the inlet.'
        manual=[{'citation_id':'main-manual','bubble_document_id':'manual','source_type':'document',
                 'chunk_full':main_note,'ask_structured_manual_support':True},
                {'citation_id':'complement-manual','bubble_document_id':'manual','source_type':'document',
                 'chunk_full':detail,'ask_structured_manual_support':True}]
        real_validate=validation.validate_response
        calls=[]
        def inject(response,request,retrieval,decision,*,runtime):
            response=deepcopy(response)
            response['answer']+='\nManual support\n- '+main_note
            response['_assistant_ui_model']={'manual_note_sources':[{'text':main_note,'citation_ids':['main-manual']}]}
            response['_assistant_core_validation_evidence']=manual
            primary=retrieval['citations']+manual
            m=self.m
            class Observer:
                def consume(self,name,arguments,callback):return callback()
                def procedure_review_sources(self,rows,*,max_records,max_context_chars):
                    packet=review_evidence.compile_packet(primary=primary,extension=rows,
                        render=m._v13_sources_block,max_records=max_records,max_context_chars=max_context_chars)
                    return list(packet.rows),packet.sources,packet.diagnostic(reused=False)
            def provider(messages,**kwargs):
                msg=messages[-1]['content']; calls.append(msg)
                self.assertIn(detail,msg); self.assertIn(main_note,msg)
                aliases=json.JSONDecoder().raw_decode(msg.split('PROCEDURE_SOURCE_ALIASES:\n',1)[1])[0]
                blocks=json.loads(msg.split('PROCEDURE_BLOCKS:\n',1)[1])
                target=next(b['block_id'] for b in blocks if b['step_number']==7)
                route=next(a for a,cid in aliases.items() if cid=='complement-manual')
                return {'outcome':'rewrite','reply_mode':'edit','answer':'',
                    'edits':[{'block_id':target,'old_text':'Feed material through the guide',
                              'new_text':detail.rstrip('.')}],
                    'citation_ids':list(aliases),'covered_facets':list(decision.required_facets),
                    'missing_facets':[],'covered_answer_types':list(decision.required_answer_types),
                    'missing_answer_types':[],'expected_list_items':[],'covered_list_items':[],
                    'missing_list_items':[],'reason':'offline grounded edit fixture',
                    'operation_boundary':{'complete':True,'checked_blocks':[b['block_id'] for b in blocks],
                        'change_edges':[],'precedence':[{'from':target,'action':'fixed guides P and Q',
                            'to':target,'next':'pivot arm R','source':route,'proof':detail}]}},'offline-model'
            eruntime=replace(runtime.execution_runtime,evidence_observer=Observer(),
                _v13_json_models=provider,_v13_sources_block=m._v13_sources_block)
            runtime=replace(runtime,execution_runtime=eruntime,
                _assistant_core_verify_or_repair_answer=lambda **kw:execution.verify_or_repair_answer(**kw,runtime=eruntime))
            return real_validate(response,request,retrieval,decision,runtime=runtime)
        fixture=wired_fixtures.WiredSourceOrderedReviewTests('run'); fixture.m=self.m
        with patch.object(validation,'validate_response',side_effect=inject):
            result,_,_=fixture.run_review()
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(len(calls),1)
        self.assertEqual(result['status'],'answered')
        self.assertTrue(contract['passed'],contract)
        self.assertTrue(contract['manual_note_bindings']['complete'])
        self.assertIn('fixed guides P and Q',result['answer'])
        self.assertIn('pivot arm R',result['answer'])
        self.assertIn('Restore the guide after feeding',result['answer'])
        self.assertEqual({'main-manual','complement-manual'} & {c['citation_id'] for c in result['citations']},
                         {'main-manual','complement-manual'})
        packet=contract['semantic_verifier']['review_evidence']
        self.assertIn('complement-manual',packet['required_citation_ids'])
        self.assertTrue(packet['complete_primary_representation'])
        self.assertFalse(packet['semantic_coverage_proven'])


if __name__=='__main__':unittest.main(verbosity=2)
