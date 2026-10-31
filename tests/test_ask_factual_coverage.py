"""Internal scope propagation and actual router schema; no model judgement claim."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import AssistantCoreRequest, build_router_schema, normalize_decision
from machinemind.ask import factual_coverage
import test_source_table_owned as owned


def request(mode='ask'):
    return AssistantCoreRequest(query='Which inspections and intervals are documented?',
        requested_mode=mode, response_language='en', company_id='scope-tenant',
        machine_id='scope-machine', ai_scope='machine_all', top_k=6)


def payload(value='useful', mode='ask'):
    return dict(request_kind='factual', effective_mode=mode, confidence=.9,
        evidence_state='supported', evidence_policy='machine_sources_required',
        information_task='document_explanation', factual_coverage=value,
        required_facets=['inspections', 'intervals'], required_answer_types=['checklist'],
        facet_queries=[dict(facet=category,answer_type='checklist',must_cover=True)
                       for category in ('inspections', 'intervals')])


class FactualCoverageRoutingTests(unittest.TestCase):
    def test_semantic_coverage_survives_without_changing_category_contract(self):
        for value in ('useful', 'exhaustive', 'unspecified'):
            raw=payload(value); before=deepcopy(raw)
            result=normalize_decision(request(),raw)
            self.assertEqual(result.factual_coverage,value)
            self.assertEqual(result.required_facets,('inspections','intervals'))
            self.assertEqual(result.required_answer_types,('checklist',))
            self.assertTrue(all(f.must_cover for f in result.facet_queries))
            self.assertEqual(raw,before)

    def test_legacy_invalid_and_nonfactual_cannot_enable_selected_only(self):
        for value in (None, '', 'Useful', True, [], {'useful':True}, 'all'):
            self.assertEqual(normalize_decision(request(),payload(value)).factual_coverage,'unspecified')
        raw=payload(); raw.pop('factual_coverage')
        self.assertEqual(normalize_decision(request(),raw).factual_coverage,'unspecified')
        self.assertEqual(normalize_decision(request(),None).factual_coverage,'unspecified')
        for kind in ('unsafe_request','out_of_scope','ambiguous','general_technical','fault_diagnostic'):
            raw=payload();raw['request_kind']=kind
            self.assertEqual(normalize_decision(request(),raw).factual_coverage,'unspecified')

    def test_other_modes_and_cross_mode_request_do_not_gain_scope(self):
        for requested,effective in (('root_cause','root_cause'),('smart_diagnostic','smart_diagnostic'),
                                    ('root_cause','ask'),('smart_diagnostic','ask'),('ask','root_cause')):
            self.assertEqual(normalize_decision(request(requested),payload(mode=effective)).factual_coverage,'unspecified')

    def test_schema_addition_is_required_without_mutating_legacy(self):
        base=build_router_schema(['ask']); before=deepcopy(base)
        result=factual_coverage.router_schema(base)
        self.assertEqual(base,before)
        field=result['schema']['properties']['factual_coverage']
        self.assertEqual(field['enum'],['useful','exhaustive','unspecified'])
        self.assertEqual(set(result['schema']['required']),set(result['schema']['properties']))
        self.assertFalse(result['schema']['additionalProperties'])
        result['schema']['properties'].pop('factual_coverage')
        result['schema']['required'].remove('factual_coverage')
        self.assertEqual(result,base)


class ActualRouterCoverageSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass(); cls.m=owned.SourceTableOwnedTests.m
    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()

    def test_actual_single_router_dispatch_augments_ask_only(self):
        for mode in ('ask','root_cause','smart_diagnostic'):
            with self.subTest(mode=mode):
                captured=[]
                def provider(messages,**kwargs):
                    captured.append(deepcopy(kwargs))
                    return payload(mode=mode),'offline-router'
                with patch.object(self.m,'_v13_json_models',provider), patch.object(
                        self.m,'_v13_gate_candidate_block',return_value=('No evidence.',[])):
                    self.m._assistant_core_router_call(request(mode),{})
                self.assertEqual(len(captured),1)
                schema=captured[0]['json_schema']['schema']
                self.assertEqual('factual_coverage' in schema['properties'],mode=='ask')
                self.assertEqual('factual_coverage' in schema['required'],mode=='ask')
                if mode=='smart_diagnostic':
                    self.assertEqual(schema,build_router_schema(list(request(mode).allowed_effective_modes))['schema'])
                if mode=='root_cause':
                    expected=self.m._retrieval_diagnostic_query.router_schema_with_diagnostic_basis(
                        build_router_schema(list(request(mode).allowed_effective_modes)))
                    self.assertEqual(schema,expected['schema'])


if __name__=='__main__':unittest.main(verbosity=2)
