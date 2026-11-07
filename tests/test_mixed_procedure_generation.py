"""Mixed requests keep the admitted procedure scaffold and full final review.

Real reader/family/order/compiler and final contract; SQL/provider/authority
observations are synthetic. This is not a live model quality or timing test.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_semantic_procedure_family as family
from machinemind.ask import execution, procedure_review, task_generation, validation


class MixedProcedureGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        family.SemanticProcedureFamilyTests.setUpClass()
        cls.m = family.SemanticProcedureFamilyTests.m

    @classmethod
    def tearDownClass(cls):
        family.SemanticProcedureFamilyTests.tearDownClass()

    def fixture(self, *, task='procedure_full', kind='procedure', guarded=True,
                structured=True):
        m = self.m
        catalog = family.synthetic_catalog()
        catalog[0]['steps'][1]['description'] = catalog[0]['steps'][1]['description'].replace(
            '\n\nNOTA DI SICUREZZA', ' Maximum cartridge mass is 25 kg.\n\nNOTA DI SICUREZZA')
        facets = ('installation operations', 'maximum mass', 'safety precautions')
        types = (m.REQ_ORDERED_ACTIONS, m.REQ_NUMERIC_VALUE, m.REQ_SAFETY_CONDITIONS)
        planner = {'information_task':task, 'required_facets':list(facets),
                   'required_answer_types':list(types), 'facet_queries':[]}
        h = family.FamilyHarness(m, catalog, chosen_key='procedure:alpha',
            query='Explain cartridge installation, the maximum mass and all precautions.', planner=planner)
        request, decision = family.SemanticProcedureFamilyTests.request_decision(self, h)
        decision = replace(decision, request_kind=kind, information_task=task,
                           required_facets=facets, required_answer_types=types)
        rows = [{'citation_id':'seed', 'source_type':'step' if structured else 'document',
                 'bubble_document_id':'step:beta1' if structured else 'manual',
                 'chunk_full':'Configure cartridge display settings.'}]
        observed = {'generic':0, 'structured':0}
        def generic(**kw):
            observed['generic'] += 1
            return {'ok':True, 'status':'answered', 'answer':'generic fixture', 'citations':rows}
        def structured_call(**kw):
            observed['structured'] += 1
            return task_generation.structured_ask(**kw, runtime=h.runtime())
        runtime = replace(m._assistant_core_ask_execution_runtime(),
            evidence_admission=family.denied if guarded else None,
            _v13_structured_ask=structured_call, _v13_generate_ask_response=generic,
            _v13_current_budget=lambda:None)
        with patch.object(execution, '_admit_input', side_effect=lambda q,r,d,**kw:r):
            response = execution.synthesize_ask(request, {'citations':rows, 'candidates':rows},
                                                decision, runtime=runtime)
        return h, request, decision, response, observed

    def validate(self, h, request, decision, response, *, missing=False, boundary=True):
        m = self.m
        rows = response['_assistant_core_validation_evidence']
        calls = []
        def verify(**kw):
            calls.append(kw)
            resolved = procedure_review.resolve_reply({
                'outcome':'no_sources' if missing else 'pass', 'answer':'',
                'reply_mode':'reject' if missing else 'retain', 'edits':[],
                'covered_facets':list(decision.required_facets[:-1] if missing else decision.required_facets),
                'missing_facets':[decision.required_facets[-1]] if missing else [],
                'covered_answer_types':list(decision.required_answer_types), 'missing_answer_types':[],
                'missing_list_items':[], 'citation_ids':[c['citation_id'] for c in kw['candidates']]},
                answer=kw['answer'], layout=procedure_review.block_layout(kw['answer']),
                required_facets=list(decision.required_facets), required_types=list(decision.required_answer_types),
                source_ordered=True, source_notes=kw['repair_context']['procedure_notes'])
            resolved['operation_boundary'] = {'complete':boundary, 'checked_blocks':[
                b['block_id'] for b in procedure_review.block_layout(kw['answer'])['blocks']],
                'changes':[], 'precedence':[]}
            return procedure_review.verify_operation_boundary(resolved,
                layout=procedure_review.block_layout(kw['answer']),
                source_texts={c['citation_id']:m._assistant_core_candidate_evidence_text(c) for c in kw['candidates']},
                source_ordered=True)
        runtime = replace(m._assistant_core_ask_validation_runtime(),
            execution_runtime=replace(m._assistant_core_ask_execution_runtime(), evidence_admission=family.denied),
            _assistant_core_should_semantic_verify_answer=lambda d:False,
            _assistant_core_verify_or_repair_answer=verify,
            _assistant_core_recover_citations=lambda *a,**kw:(rows,[]),
            _sanitize_citations_for_response=lambda rows,**kw:rows, _build_rg_links=lambda *a,**kw:[],
            _v13_current_budget=lambda:None, final_contract_enabled=True,
            source_fields=m._procedure_ui_fields, source_sections=m._procedure_ui_sections)
        with patch.object(validation,'_inputs',side_effect=lambda r,q,t,d,**kw:(r,t)), \
             patch.object(validation,'_collection',side_effect=lambda q,items,d,**kw:items), \
             patch.object(validation,'_output',side_effect=lambda r,*a,**kw:r):
            result = validation.validate_response({**response,'_assistant_core_repair_attempted':True},
                request, {'citations':rows}, decision, runtime=runtime)
        return result, calls

    def test_mixed_full_procedure_preserves_source_mass_order_and_safety_before_review(self):
        h, request, decision, response, observed = self.fixture()
        self.assertEqual(observed, {'generic':0, 'structured':1})
        self.assertEqual(len(h.provider_calls), 1)  # family choice, no prose re-generation
        self.assertEqual(h.provider_calls[0][1]['purpose'], 'ask_structured_synthesis')
        self.assertEqual(h.provider_calls[0][1]['max_output_tokens'], h.tasks.V13_FAST_MAX_OUTPUT_TOKENS)
        self.assertTrue(response['_assistant_core_source_ordered_procedure'])
        self.assertIn('25 kg', response['answer'])
        self.assertIn('Keep the supply isolated.', response['answer'])
        self.assertLess(response['answer'].index('Insert the cartridge'), response['answer'].index('Secure the housing'))
        result, reviews = self.validate(h, request, decision, response)
        self.assertEqual(len(reviews), 1)
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])
        self.assertIn('25 kg', reviews[0]['answer'])
        self.assertEqual(result['answer'], reviews[0]['answer'])

    def test_source_scaffold_never_substitutes_for_missing_facet_or_boundary_review(self):
        for kwargs in ({'missing':True}, {'boundary':False}):
            with self.subTest(kwargs=kwargs):
                h, request, decision, response, _ = self.fixture()
                result, calls = self.validate(h, request, decision, response, **kwargs)
                self.assertEqual(len(calls),1)
                self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])
                self.assertNotEqual(result['status'],'answered')

    def test_factual_numeric_and_unprotected_routes_remain_generic(self):
        for kwargs in ({'task':'numeric_specification','kind':'factual'}, {'guarded':False},
                       {'task':'numeric_specification','kind':'procedure'}, {'structured':False}):
            with self.subTest(kwargs=kwargs):
                h, _, _, response, observed = self.fixture(**kwargs)
                self.assertEqual(observed, {'generic':1, 'structured':0})
                self.assertEqual(h.provider_calls, [])
                self.assertEqual(response['answer'],'generic fixture')

    def test_missing_or_foreign_family_cannot_be_selected(self):
        h = family.FamilyHarness(self.m, family.synthetic_catalog(), chosen_key='procedure:alpha',
            planner={'information_task':'procedure_full', 'required_answer_types':['ordered_actions','numeric_value']})
        h.forced_id='procedure:foreign-tenant:p1'
        with self.assertRaisesRegex(task_generation.GenerationEvidenceError,'not_admitted'):
            h.run()
        self.assertEqual(h.expansions,[])
        self.assertEqual(len(h.provider_calls),1)


if __name__ == '__main__':
    unittest.main()
