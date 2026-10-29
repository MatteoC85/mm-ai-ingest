"""Offline normalization boundaries; these fixtures do not prove LLM classification."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_core_v2 import AssistantCoreRequest, normalize_decision, build_router_schema
from machinemind.ask import procedure_scope


QUERY = 'After cleaning the guide, what must I restore before restarting the line?'


def request(query=QUERY, mode='ask'):
    return AssistantCoreRequest(query=query, requested_mode=mode, response_language='en',
        company_id='fixture-tenant', machine_id='fixture-machine', ai_scope='machine_all', top_k=6)


def payload():
    return {'effective_mode':'ask', 'request_kind':'procedure', 'confidence':.9,
        'evidence_state':'partial', 'evidence_policy':'machine_sources_required',
        'information_task':'procedure_segment', 'procedure_scope':'bounded_phase',
        'procedure_phase':{'completed_context':'After cleaning the guide',
            'requested_actions':'what must I restore', 'later_context':'before restarting the line'},
        'required_answer_types':['ordered_actions','safety_conditions'],
        'required_facets':['restore safe operation'],
        'facet_queries':[{'facet':'restore safe operation','answer_type':'ordered_actions',
                         'must_cover':True,'dense_queries':['restore safe operation']}]}


class PhaseRouterTests(unittest.TestCase):
    def test_literal_user_phase_is_immutable_and_ordered_requirements_survive(self):
        raw=payload(); before=copy.deepcopy(raw)
        decision=normalize_decision(request(),raw)
        self.assertEqual(decision.procedure_scope,'bounded_phase')
        self.assertEqual(getattr(decision,'procedure_phase',()),tuple(raw['procedure_phase'].values()))
        self.assertIn('ordered_actions',decision.required_answer_types)
        self.assertEqual(raw,before)

    def test_either_temporal_boundary_is_sufficient_without_inventing_the_other(self):
        for empty in ('completed_context','later_context'):
            with self.subTest(empty=empty):
                raw=payload(); raw['procedure_phase'][empty]=''
                decision=normalize_decision(request(),raw)
                self.assertEqual(decision.procedure_scope,'bounded_phase')
                self.assertEqual(getattr(decision,'procedure_phase',()),tuple(raw['procedure_phase'].values()))

    def test_invalid_or_invented_excerpts_never_select_bounded_producer(self):
        variants=[None, [], {}, {'completed_context':'After cleaning the guide',
            'requested_actions':'what must I restore','later_context':'before restarting the line','extra':'bad'}]
        for key,value in (('requested_actions',''),('completed_context','AFTER CLEANING THE GUIDE'),
                          ('requested_actions','restore the emergency circuit'),
                          ('requested_actions',True),('requested_actions','x'*241),
                          ('later_context','what must I restore')):
            phase=payload()['procedure_phase']; phase[key]=value; variants.append(phase)
        phase=payload()['procedure_phase']; phase['completed_context']='';phase['later_context']='';variants.append(phase)
        for value in variants:
            with self.subTest(value=value):
                raw=payload();raw['procedure_phase']=value
                decision=normalize_decision(request(),raw)
                self.assertNotEqual(decision.procedure_scope,'bounded_phase')
                self.assertFalse(getattr(decision,'procedure_phase',()))
                self.assertIn('ordered_actions',decision.required_answer_types)

    def test_full_ambiguous_unsafe_and_non_ask_requests_cannot_acquire_phase_authority(self):
        changes=[('information_task','procedure_full'),('request_kind','ambiguous'),
                 ('request_kind','unsafe_request'),('request_kind','out_of_scope'),
                 ('evidence_state','clarify'),('evidence_state','unsupported'),
                 ('effective_mode','root_cause')]
        for key,value in changes:
            with self.subTest(key=key,value=value):
                raw=payload();raw[key]=value
                decision=normalize_decision(request(),raw)
                self.assertNotEqual(decision.procedure_scope,'bounded_phase')
                self.assertFalse(getattr(decision,'procedure_phase',()))
        for mode in ('root_cause','smart_diagnostic'):
            decision=normalize_decision(request(mode=mode),payload())
            self.assertFalse(getattr(decision,'procedure_phase',()))

    def test_other_scopes_and_legacy_payloads_drop_phase_quotes(self):
        for scope in ('conditions_only','execution_steps','other',''):
            raw=payload();raw['procedure_scope']=scope
            self.assertFalse(getattr(normalize_decision(request(),raw),'procedure_phase',()))
        raw=payload();raw.pop('procedure_scope');raw.pop('procedure_phase')
        decision=normalize_decision(request(),raw)
        self.assertEqual(decision.procedure_scope,'')
        self.assertIn('ordered_actions',decision.required_answer_types)

    def test_quote_length_is_bounded_without_truncation_and_non_english_text_is_literal(self):
        raw=payload();long_quote='restore '+('the assembly '*24)
        raw['procedure_phase']['requested_actions']=long_quote
        query='After cleaning the guide, '+long_quote+' before restarting the line'
        self.assertGreater(len(long_quote),240)
        self.assertIn(long_quote,query)
        self.assertFalse(getattr(normalize_decision(request(query),raw),'procedure_phase',()))
        raw=payload();raw['procedure_phase']={'completed_context':'Dopo la pulizia',
            'requested_actions':'cosa devo ripristinare','later_context':'prima del riavvio'}
        decision=normalize_decision(request('Dopo la pulizia, cosa devo ripristinare prima del riavvio?'),raw)
        self.assertEqual(getattr(decision,'procedure_phase',()),tuple(raw['procedure_phase'].values()))

    def test_strict_schema_has_only_three_required_excerpt_fields_and_does_not_mutate_base(self):
        base=build_router_schema(['ask']);before=copy.deepcopy(base)
        schema=procedure_scope.router_schema(base)['schema']
        self.assertEqual(base,before)
        phase=schema['properties'].get('procedure_phase',{})
        self.assertEqual(set(phase.get('required',[])),{'completed_context','requested_actions','later_context'})
        self.assertEqual(set(phase.get('properties',{})),set(phase.get('required',[])))
        self.assertIs(phase.get('additionalProperties'),False)
        self.assertEqual(schema['required'].count('procedure_phase'),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
