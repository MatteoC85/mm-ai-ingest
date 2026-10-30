"""Representation normalization through the real review and final gates.

Model judgement and source authority are fixtures; no semantic/live PASS is
inferred from successful representation or telemetry checks.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_factual_table_inventory as shared
import test_ask_source_first_inventory as owned
from machinemind.ask import enumeration_review, request_completion


class InventoryNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared.FactualTableInventoryTests.setUpClass()
        cls.m=shared.FactualTableInventoryTests.m
    @classmethod
    def tearDownClass(cls):shared.FactualTableInventoryTests.tearDownClass()

    def case(self,**kwargs):
        base=shared.FactualTableInventoryTests();base.m=self.m
        return base.case(coverage=base.complete_coverage(),outcome='rewrite',draft='',**kwargs)

    def test_one_to_one_normalization_keeps_wire_text_and_table_indices(self):
        for item in ('Check  slide cleanliness daily', 'Temperatura olio da 68–80 °C',
                     'Controllare integrita\u0300 ogni mese', '  Inspect wear every 360 hours  '):
            with self.subTest(item=item):
                items=[item,*shared.ITEMS[1:]]
                result,final,_=self.case(items=items,answer='[[FACTUAL_INVENTORY]]',expected_calls=2)
                self.assertTrue(final['passed'])
                raw_body='\n'.join('- '+s for s in items)
                expanded=enumeration_review.resolve_inventory_answer(
                    {'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]','expected_list_items':items},
                    canonical_inventory=self.m._dedup_text_values(items,limit=40),allowed=True,
                    normalize_item=lambda item:self.m._dedup_text_values([item],limit=1))
                self.assertEqual(expanded['answer'],raw_body)
                # The existing redactor still normalizes whitespace afterwards;
                # this patch never replaces visible raw text with canonical items.
                self.assertEqual(result['answer'],self.m._assistant_core_redact_internal_text(raw_body))
                self.assertEqual(result['expected_list_items'],self.m._dedup_text_values(items,limit=40))
                self.assertEqual(result['table_record_coverage'][0]['inventory_items'],[0,1])
                self.assertTrue(final['semantic_answer_bound'])

    def test_full_answer_table_coverage_uses_the_same_index_proof(self):
        items=['Check  slide cleanliness daily',*shared.ITEMS[1:]]
        body='\n'.join('- '+s for s in items)
        result,final,_=self.case(items=items,answer=body,expected_calls=2)
        self.assertTrue(final['passed'])
        self.assertEqual(result['answer'],self.m._assistant_core_redact_internal_text(body))
        self.assertEqual(result['table_record_coverage'][1]['inventory_items'],[2])

    def test_collision_dropped_item_and_duplicate_still_fail_real_adapter(self):
        for items in ([shared.ITEMS[0],'Check  slide cleanliness daily',shared.ITEMS[2]],
                      [shared.ITEMS[0],shared.ITEMS[0].upper(),shared.ITEMS[2]],
                      [shared.ITEMS[0],'  ',shared.ITEMS[2]]):
            for template in (True,False):
                with self.subTest(items=items,template=template):
                    result,final,_=self.case(items=items,answer='[[FACTUAL_INVENTORY]]' if template else '\n'.join(items))
                    self.assertFalse(final['passed'])
                    self.assertEqual(result['outcome'],'no_sources')
                    self.assertEqual(result['reason'],'factual_inventory_template_invalid' if template else 'source_table_inventory_incomplete')

    def test_position_proof_rejects_reorder_or_changed_value_not_just_same_set(self):
        normalize=lambda item:self.m._dedup_text_values([item],limit=1)
        wire=['Inspect every 360 hours','Pressure 20 MPa']
        for canonical in (list(reversed(wire)),['Inspect every 361 hours',wire[1]],wire[:1]):
            self.assertFalse(enumeration_review.inventory_positions_match(wire,canonical,normalize_item=normalize))
        self.assertTrue(enumeration_review.inventory_positions_match(wire,wire,normalize_item=normalize))

    def test_owned_recovery_normalization_then_grounding_and_final_hash(self):
        owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=True),started=time.monotonic())
        h=owned.SourceFirstFixture(self)
        try:
            h.items[0]='Check  slide cleanliness daily'
            raw=deepcopy(h.items)
            result=h.run()
            self.assertEqual(result['status'],'answered')
            self.assertEqual(h.generated['answer'],self.m._assistant_core_redact_internal_text('\n'.join('- '+s for s in raw)))
            self.assertTrue(h.final['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
            self.assertEqual(len(h.posts),2)
            self.assertNotEqual(h.reviewer_calls[0][1]['json_schema']['name'],'factual_context_review')
            self.assertEqual(h.reviewer_calls[1][1]['json_schema']['name'],'factual_context_review')
            self.assertEqual(h.final['meta']['assistant_core_validation']['answer_contract']
                             ['semantic_verifier']['independent_context_review']['outcome'],'pass')
            self.assertEqual(h.writer_calls,[])
        finally:h.close();owner.close()

    def test_debug_observer_distinguishes_model_rejection_template_and_table_gate(self):
        cases=[('model',{'outcome':'no_sources','answer':''},'other','no_sources','accepted_or_unused','accepted_or_unused'),
               ('template',{'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]',
                   'items':[shared.ITEMS[0],shared.ITEMS[0],shared.ITEMS[2]]},
                   'factual_inventory_template_invalid','rewrite','rejected','accepted_or_unused'),
               ('table',{'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]','coverage':['T1=0,1']},
                   'source_table_inventory_incomplete','rewrite','accepted_or_unused','rejected')]
        for name,kwargs,reason,model_outcome,template,table in cases:
            with self.subTest(case=name):
                owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                    payload=SimpleNamespace(company_id='offline-company',debug=True),started=time.monotonic())
                try:
                    base=shared.FactualTableInventoryTests();base.m=self.m
                    base.case(draft='',**{'coverage':base.complete_coverage(),**kwargs})
                    final=owner.finish({'ok':False,'result_code':'BUDGET_EXCEEDED','meta':{}})
                    row=final['meta']['request_completion']['contract_checks'][-1]
                    self.assertEqual(row['reason'],reason)
                    self.assertEqual(row.get('inventory_model_outcome'),model_outcome)
                    self.assertEqual(row.get('inventory_template_gate'),template)
                    self.assertEqual(row.get('table_coverage_gate'),table)
                    self.assertEqual(row.get('table_expected_count'),2)
                    self.assertNotIn('Check slide',json.dumps(row))
                    self.assertNotIn('answer',final)
                finally:owner.close()
                self.assertEqual(owner.contract_checks,[])

    def test_telemetry_details_are_debug_exact_true_bounded_and_no_content(self):
        for debug in (False,'true',True):
            owner=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company',debug=debug),started=time.monotonic())
            try:
                diagnostics={'inventory_model_outcome':'PRIVATE_TEXT','inventory_answer_form':'PRIVATE_TEXT',
                    'inventory_template_gate':'PRIVATE_TEXT','table_coverage_gate':'PRIVATE_TEXT',
                    'inventory_wire_count':100000,'inventory_canonical_count':True,'table_wire_count':-1,
                    'answer':'PRIVATE_TEXT','source':'PRIVATE_TEXT'}
                request_completion.record_contract_observation('semantic_review',{'outcome':'no_sources',
                    'reason':'factual_inventory_template_invalid'},diagnostics=diagnostics)
                row=owner.contract_checks[-1]
                self.assertNotIn('PRIVATE_TEXT',json.dumps(row))
                if debug is True:
                    self.assertEqual(row.get('inventory_wire_count'),100)
                    self.assertEqual(row.get('inventory_model_outcome'),'other')
                else:self.assertFalse(any(k.startswith('inventory_') for k in row))
                self.assertNotIn('inventory_canonical_count',row)
                self.assertNotIn('table_wire_count',row)
            finally:owner.close()


if __name__=='__main__':unittest.main(verbosity=2)
