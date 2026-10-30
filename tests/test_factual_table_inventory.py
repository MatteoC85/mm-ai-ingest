"""Source record bookkeeping through the real verifier/final gate, offline.

Model classifications are explicit fixtures, not semantic entailment proofs.
No database/provider I/O; source authority is isolated by the adapter fixture.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_factual_list_review as shared
from machinemind.ask import execution, enumeration_review, final_contract, review_source_wire


HEADER = '- [PDF_TABLE_HEADER rows=1 columns=4] c1 r1="Assembly/action" | c2 r1="Interval" | c3 r1="Material" | c4 r1="Notes"'
ROW = '- [PDF_TABLE_RECORD rows=2-3 columns=4] c1 "Assembly/action" r2="Slide maintenance" | c2 "Interval" r2-3="Daily" | c3 "Material" r2-3="Grease ZTR-4" | c4 "Notes" r2-3="Inspect wear every 360 hours and grease if necessary." | c1 "Assembly/action" r3="Visually check cleanliness and contamination."'
CONTINUATION = '- [PDF_TABLE_RECORD rows=1 columns=4] c1 r1="Inspect wiring integrity" | c2 r1="Monthly" | c3 r1="" | c4 r1="Qualified personnel only."'
OTHER = '- [PDF_TABLE_RECORD rows=1 columns=2] c1 r1="Separate packaged unit" | c2 r1="Storage only, no installed machine relationship stated."'
SOURCES = HEADER+'\n'+ROW+'\nPage break / continuation\n'+CONTINUATION
ITEMS = ['Check slide cleanliness daily', 'Inspect slide wear every 360 hours and grease only if necessary',
         'Qualified personnel inspect wiring integrity monthly']


class FactualTableInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared.FactualListReviewTests.setUpClass()
        cls.m = shared.FactualListReviewTests.m

    @classmethod
    def tearDownClass(cls):
        shared.FactualListReviewTests.tearDownClass()

    def case(self, *, items=ITEMS, answer=None, coverage=None, outcome='pass', mode='ask',
             kind='factual', protected=True, source=SOURCES):
        m=self.m
        request=m.AssistantCoreRequest(query='Which lubrication points and maintenance intervals are documented?',
            requested_mode=mode, response_language='en',company_id='offline-company',
            machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision=m.AssistantCoreDecision(request_kind=kind,effective_mode=mode,confidence=.9,
            requested_mode_fit=True,evidence_state='supported',evidence_policy='evidence_required',
            information_task='document_explanation',required_facets=('documented obligations',),
            required_answer_types=('checklist',))
        answer='; '.join(items)+'.' if answer is None else answer
        rows=[{'citation_id':'manual:inspection','source_type':'document','chunk_full':source}]
        original=deepcopy(rows);seen=[]
        def provider(messages,**kwargs):
            seen.append((messages,kwargs))
            result={'outcome':outcome,'answer':answer,'citation_ids':['manual:inspection'],
                'covered_facets':['documented obligations'],'missing_facets':[],
                'covered_answer_types':['checklist'],'missing_answer_types':[],
                'expected_list_items':list(items),'covered_list_items':[],
                'missing_list_items':[],'reason':'offline fixture'}
            if coverage is not None:
                result['table_record_coverage']=deepcopy(coverage)
            return result,'offline-model'
        runtime=replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=(lambda *a,**kw:None) if protected else None,
            _assistant_core_enumeration_requested=lambda *a:True,
            _v13_json_models=provider,
            _v13_current_budget=lambda:m._assistant_core_new_budget('ask'))
        with patch.object(execution,'_admit_input',side_effect=lambda req,data,dec,**kw:data):
            result=execution.verify_or_repair_answer(request=request,decision=decision,
                answer=answer,candidates=rows,runtime=runtime)
        self.assertEqual(len(seen),1)
        self.assertEqual(rows,original)
        final=final_contract.final_contract(deterministic={'passed':True}, semantic=result,
            reviewed_answer=result['answer'],answer=result['answer'],semantic_complete=True,semantic_partial=False)
        return result,final,seen[0]

    @staticmethod
    def complete_coverage():
        return [{'record_id':'T1','inventory_items':[0,1],'outside_request_reason':''},
                {'record_id':'T2','inventory_items':[2],'outside_request_reason':''}]

    def test_old_self_reported_inventory_cannot_skip_continuation_record_accounting(self):
        result,final,_=self.case(items=['Grease the slide daily'])
        self.assertEqual(result['outcome'],'no_sources')
        self.assertEqual(result['reason'],'source_table_inventory_incomplete')
        self.assertTrue(result['_terminal_review_failure'])
        self.assertFalse(final['passed'])
        # This rejection is missing source-record accounting, not a claim that
        # the deterministic gate understands the unsupported daily intervention.

    def test_complete_independent_categories_and_conditional_action_survive_empty_covered_wire(self):
        result,final,(messages,kwargs)=self.case(coverage=self.complete_coverage())
        self.assertTrue(final['passed'])
        self.assertEqual(result['covered_list_items'],ITEMS)
        self.assertEqual(result['missing_list_items'],[])
        schema=kwargs['json_schema']['schema']['properties']
        self.assertEqual(schema['covered_list_items'].get('maxItems'),0)
        self.assertEqual(schema['table_record_coverage']['items']['properties']['record_id']['enum'],['T1','T2'])
        self.assertEqual(kwargs['max_output_tokens'],3200)
        wire=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
        expanded=review_source_wire.restore_sources_block(wire)
        self.assertIn('[TABLE_RECORD T1]\n'+ROW,expanded)
        self.assertIn('[TABLE_RECORD T2]\n'+CONTINUATION,expanded)

    def test_incomplete_duplicate_foreign_or_false_indices_cannot_certify_records(self):
        valid=self.complete_coverage()
        variants=[valid[:1], [valid[0],valid[0]],
            [valid[0],{**valid[1],'record_id':'OTHER'}],
            [valid[0],{**valid[1],'inventory_items':[3]}],
            [valid[0],{**valid[1],'inventory_items':[True]}],
            [valid[0],{**valid[1],'inventory_items':[]}],
            [valid[0],{**valid[1],'outside_request_reason':'contradictory classification'}]]
        for coverage in variants:
            with self.subTest(coverage=coverage):
                result,final,_=self.case(coverage=coverage)
                self.assertEqual(result['reason'],'source_table_inventory_incomplete')
                self.assertFalse(final['passed'])

    def test_record_accounting_does_not_override_omission_or_semantic_rejection(self):
        result,final,_=self.case(coverage=self.complete_coverage(),answer=ITEMS[0]+'.')
        self.assertEqual(result['missing_list_items'],ITEMS[1:])
        self.assertFalse(final['passed'])
        result,final,_=self.case(coverage=self.complete_coverage(),outcome='no_sources')
        self.assertEqual(result['outcome'],'no_sources')
        self.assertFalse(final['passed'])

    def test_wire_inventory_dedup_cannot_reassign_a_record_index(self):
        duplicate_items=[ITEMS[0],ITEMS[0],ITEMS[2]]
        coverage=[{'record_id':'T1','inventory_items':[0],'outside_request_reason':''},
                  {'record_id':'T2','inventory_items':[1],'outside_request_reason':''}]
        result,final,_=self.case(items=duplicate_items,coverage=coverage)
        self.assertEqual(result['expected_list_items'],[ITEMS[0],ITEMS[2]])
        self.assertEqual(result['reason'],'source_table_inventory_incomplete')
        self.assertFalse(final['passed'])
        # Index1 in the model's list denoted the first action again; it cannot
        # silently certify the wiring obligation after deduplication.

    def test_explicit_outside_request_record_is_not_automatic_cross_assembly_evidence(self):
        coverage=self.complete_coverage()+[
            {'record_id':'T3','inventory_items':[],
             'outside_request_reason':'Separate stored unit; no relationship to requested installed assemblies is documented.'}]
        result,final,_=self.case(coverage=coverage,source=SOURCES+'\nOther source\n'+OTHER)
        self.assertTrue(final['passed'])
        self.assertNotIn('Separate packaged unit',result['answer'])
        # The semantic classification is a fixture and must still be reviewed
        # live. No code infers applicability or equivalence from the ID.

    def test_record_projection_is_lossless_and_identical_rows_keep_wire_dedup(self):
        source='Source A\n'+SOURCES+'\nSource B\n'+SOURCES+'\n'
        annotated,ids=enumeration_review.annotate_table_records(source)
        self.assertEqual(ids,('T1','T2','T3','T4'))
        self.assertEqual(re.sub(r'\[TABLE_RECORD T\d+\]\n', '', annotated),source)
        self.assertIn('[TABLE_RECORD T1]\n'+ROW,annotated)
        self.assertIn('[TABLE_RECORD T3]\n'+ROW,annotated)
        wire=review_source_wire.compact_sources_block(annotated)
        self.assertEqual(review_source_wire.restore_sources_block(wire),annotated)
        self.assertLess(review_source_wire.serialized_content_bytes(wire),
                        review_source_wire.serialized_content_bytes(annotated))
        for malformed in ('[TABLE_RECORD T1] '+ROW,ROW.replace('r2-3="Daily"','r4="Daily"'),
                          '- [PDF_TABLE_UNASSIGNED] ambiguous words'):
            with self.subTest(malformed=malformed):
                self.assertEqual(enumeration_review.annotate_table_records(malformed),(malformed,()))

    def test_nonfactual_unprotected_and_other_modes_keep_original_wire_schema(self):
        baseline=self.m._assistant_core_contract_verifier_schema()
        for override in ({'protected':False},{'mode':'root_cause'},
                         {'mode':'smart_diagnostic'},{'kind':'procedure'}):
            with self.subTest(override=override):
                _,_,(messages,kwargs)=self.case(**override)
                self.assertEqual(kwargs['json_schema'],baseline)
                self.assertNotIn('[TABLE_RECORD ',messages[1]['content'])


if __name__=='__main__':
    unittest.main(verbosity=2)
