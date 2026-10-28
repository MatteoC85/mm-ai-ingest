"""Offline ASK clarification routing; no provider, network, database or production data."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.ask.request_readiness import apply_readiness, router_schema
from assistant_core_v2 import AssistantCoreRequest, AssistantCoreHooks, AssistantCoreV2, build_router_schema


def denied(*args, **kwargs):
    raise AssertionError('Clarification must stop before retrieval refinement or answer generation')


class AskRequestReadinessTests(unittest.TestCase):
    def test_only_clarification_states_stop_before_synthesis_it_en(self):
        for state in ('underspecified','contradictory','uninterpretable'):
            for lang in ('it','en'):
                with self.subTest(state=state,lang=lang):
                    request=AssistantCoreRequest(query='Original input',requested_mode='ask',response_language=lang,
                        company_id='tenant-fixture',machine_id='machine-fixture',ai_scope='machine_all',top_k=6)
                    raw={'request_kind':'fault_diagnostic','effective_mode':'ask','confidence':0.9,'evidence_state':'partial',
                         'evidence_policy':'machine_sources_required','request_readiness':state,'dense_queries':['invented'],
                         'required_facets':['invented'], 'missing_information':['decisive condition']}
                    hooks=AssistantCoreHooks(retrieve_neutral=lambda req:{'candidates':[{'citation_id':'nearby'}]},
                        route_semantically=lambda req,retr:apply_readiness(raw,language=lang),
                        prepare_evidence=denied,synthesize_ask=denied,synthesize_root_cause=denied,
                        synthesize_general=denied,build_no_evidence=denied,refine_retrieval=denied,
                        build_clarification=lambda req,dec:{'ok':True,'status':'needs_clarification','answer':dec.clarification_question},
                        build_out_of_scope=denied,build_safety_refusal=denied)
                    result=AssistantCoreV2(hooks).run(request)
                    self.assertEqual(result['result_code'],'NEEDS_CLARIFICATION')
                    self.assertTrue(result['answer'])

    def test_missing_sources_alone_do_not_require_clarification(self):
        raw={'request_readiness':'answerable','request_kind':'factual','evidence_state':'unsupported','missing_information':['undocumented torque']}
        self.assertEqual(apply_readiness(raw),raw)

    def test_safety_and_out_of_scope_take_precedence(self):
        for kind in ('unsafe_request','out_of_scope'):
            raw={'request_kind':kind,'evidence_state':'unsupported','request_readiness':'contradictory'}
            self.assertEqual(apply_readiness(raw),raw)

    def test_payload_is_not_mutated_and_exact_question_is_preserved(self):
        raw={'request_readiness':'contradictory','clarification_question':'Was this before or after the stop?',
             'required_facets':['draft'], 'diagnostic_observables':['preserved for audit']}
        prior=copy.deepcopy(raw);result=apply_readiness(raw,language='en')
        self.assertEqual(raw,prior)
        self.assertEqual(result['clarification_question'],raw['clarification_question'])
        self.assertEqual(result['required_facets'],[])

    def test_schema_is_strict_and_does_not_mutate_legacy_schema(self):
        base=build_router_schema(['ask']);before=copy.deepcopy(base);result=router_schema(base)
        self.assertEqual(base,before)
        self.assertEqual(set(result['schema']['required']),set(result['schema']['properties']))
        self.assertFalse(result['schema']['additionalProperties'])

    def test_legacy_router_payload_not_reinterpreted(self):
        for state in (None,'invalid',False):
            raw={'request_readiness':state,'evidence_state':'supported'}
            self.assertEqual(apply_readiness(raw),raw)

if __name__=='__main__': unittest.main()
