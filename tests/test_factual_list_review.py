"""Real review adapter with recorded-style replies; no provider or database I/O.

Checks the empty-inventory defect and concrete omitted-item rejection, not live
semantic correctness. Admission is isolated here; authority has separate suites.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_final_review_window as window_fixtures
from machinemind.ask import execution, enumeration_review, final_contract


class FactualListReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        window_fixtures.FinalReviewWindowTests.setUpClass()
        cls.m = window_fixtures.FinalReviewWindowTests.m

    @classmethod
    def tearDownClass(cls):
        window_fixtures.FinalReviewWindowTests.tearDownClass()

    def case(self, *, items=(), answer='Inspect the frame.', protected=True,
             mode='ask', kind='factual', task='document_explanation', state='supported',
             outcome='pass'):
        m=self.m
        request=m.AssistantCoreRequest(query='Which inspection points and intervals are documented?',
            requested_mode=mode,response_language='en',company_id='offline-company',
            machine_id='offline-machine',ai_scope='machine_all',top_k=6)
        decision=m.AssistantCoreDecision(request_kind=kind,effective_mode=mode,confidence=.9,
            requested_mode_fit=True,evidence_state=state,evidence_policy='evidence_required',
            information_task=task,required_facets=('inspection',),required_answer_types=('checklist',))
        source='Inspect the frame daily.\nCheck pressure sensor calibration every 200 hours.'
        rows=[{'citation_id':'manual:inspection','source_type':'document','chunk_full':source}]
        seen=[]
        def provider(messages,**kwargs):
            seen.append((messages,kwargs))
            return {'outcome':outcome,'answer':answer,'citation_ids':['manual:inspection'],
                'covered_facets':['inspection'],'missing_facets':[],
                'covered_answer_types':['checklist'],'missing_answer_types':[],
                'expected_list_items':list(items),'covered_list_items':list(items),
                'missing_list_items':[],'reason':'offline fixture'},'offline-model'
        schema=m._assistant_core_contract_verifier_schema()
        baseline=deepcopy(schema)
        runtime=replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=(lambda *a,**kw:None) if protected else None,
            _assistant_core_enumeration_requested=lambda *a:True,
            _assistant_core_contract_verifier_schema=lambda:schema,
            _v13_json_models=provider,_v13_current_budget=lambda:m._assistant_core_new_budget('ask'))
        with patch.object(execution,'_admit_input',side_effect=lambda req,data,dec,**kw:data):
            result=execution.verify_or_repair_answer(request=request,decision=decision,
                answer=answer,candidates=rows,runtime=runtime)
        self.assertEqual(len(seen),1)
        self.assertEqual(schema,baseline)
        return result,seen[0],baseline

    def test_empty_claimed_pass_and_rewrite_cannot_certify_list(self):
        for outcome in ('pass','rewrite'):
            with self.subTest(outcome=outcome):
                result,_,_=self.case(outcome=outcome)
                self.assertEqual(result['outcome'],'no_sources')
                self.assertEqual(result['reason'],'supported_list_inventory_missing')

    def test_concrete_missing_row_rejected_even_when_provider_claims_coverage(self):
        item='Check pressure sensor calibration every 200 hours'
        result,_,_=self.case(items=[item])
        self.assertEqual(result['outcome'],'rewrite')
        self.assertEqual(result['missing_list_items'],[item])
        self.assertEqual(result['covered_list_items'],[])

    def test_complete_inventory_preserved_in_response_language_and_source_first(self):
        items=['Inspect the frame daily','Check pressure sensor calibration every 200 hours']
        result,(messages,kwargs),_=self.case(items=items,answer='. '.join(items)+'.')
        self.assertEqual(result['outcome'],'pass')
        self.assertEqual(result['covered_list_items'],items)
        self.assertEqual(result['missing_list_items'],[])
        self.assertEqual(next(iter(kwargs['json_schema']['schema']['properties'])),'expected_list_items')
        self.assertLess(messages[1]['content'].index('SOURCES:\n'),messages[1]['content'].index('CURRENT_ANSWER:\n'))
        self.assertIn('RESPONSE_LANGUAGE',messages[0]['content'])

    def test_nonfactual_procedural_unsupported_and_unprotected_keep_legacy_schema(self):
        for overrides in ({'protected':False},{'mode':'root_cause'},
                {'mode':'smart_diagnostic'},{'kind':'procedure','task':'procedure_segment'},
                {'task':'general_technical'},{'state':'unsupported'}):
            with self.subTest(overrides=overrides):
                result,(messages,kwargs),baseline=self.case(**overrides)
                self.assertEqual(result['outcome'],'pass')
                self.assertEqual(kwargs['json_schema'],baseline)
                self.assertNotIn('FACTUAL_LIST_REVIEW:',messages[0]['content'])

    def test_no_sources_is_not_upgraded_by_nonempty_inventory(self):
        result,_,_=self.case(items=['Inspect the frame'],outcome='no_sources')
        self.assertEqual(result['outcome'],'no_sources')

    def test_final_contract_consumes_real_reviewer_inventory_verdict(self):
        complete_items = ['Inspect the frame daily',
                          'Check pressure sensor calibration every 200 hours']
        for items, answer, expected in (
                ([], 'Inspect the frame.', False),
                (complete_items, 'Inspect the frame daily.', False),
                (complete_items, '. '.join(complete_items)+'.', True)):
            with self.subTest(items=items, answer=answer):
                reviewed, _, _ = self.case(items=items, answer=answer)
                # Exercise the real final gate with the real reviewer result.
                # Even optimistic upstream booleans cannot upgrade no_sources
                # or missing inventory items to a complete answer. This checks
                # contract consumption, not model semantic correctness or ACLs.
                result = final_contract.final_contract(
                    deterministic={'passed': True, 'reason': 'optimistic fixture'},
                    semantic=reviewed, reviewed_answer=reviewed['answer'],
                    answer=reviewed['answer'], semantic_complete=True,
                    semantic_partial=False)
                self.assertEqual(result['passed'], expected)
                self.assertEqual(result['complete'], expected)
                self.assertTrue(result['semantic_answer_bound'])
                self.assertEqual(result['semantic_verifier'], reviewed)
                if not items:
                    self.assertEqual(reviewed['reason'], 'supported_list_inventory_missing')
                    self.assertEqual(result['reason'], 'semantic_contract_incomplete')
                elif not expected:
                    self.assertEqual(result['missing_list_items'], [complete_items[1]])


if __name__=='__main__':unittest.main(verbosity=2)
