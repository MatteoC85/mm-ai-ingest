"""Offline router-to-selector aliases: fixtures are translations, not model proof."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assistant_core_v2 import AssistantCoreRequest, build_router_schema, normalize_decision
from machinemind.retrieval import diagnostic_query as dq, root_clue_support as cs
from machinemind.retrieval import evidence_orchestration as eo
from test_root_request_segments import reply, QUERY, SEGMENTS
import test_root_request_segments as segment_fixture

ALIASES = ["Dopo l'attrezzaggio,", "il carrello si muove ma il materiale rimane fermo.",
           "La pinza è aperta.", "Quale causa?"]


def request(query=QUERY, mode="root_cause", company="fixture-company"):
    return AssistantCoreRequest(query=query, requested_mode=mode, response_language="en",
        company_id=company, machine_id="fixture-machine", ai_scope="machine_all", top_k=6)


def classified(query=QUERY, segments=SEGMENTS, aliases=ALIASES):
    raw=reply(query, segments)
    for part, alias in zip(raw['diagnostic_basis']['request_parts'], aliases):
        part['retrieval_aliases'] = alias if isinstance(alias, list) else [alias]
    return raw


def decision(raw=None, req=None):
    req=req or request(); raw=deepcopy(raw or classified())
    profile=dq.profile_from_router(raw,dq.analyze_diagnostic_query(req.query,response_language='en'))
    raw=dq.sanitize_router_payload(raw,profile).payload
    result=dq.enforce_diagnostic_basis(raw,profile)
    if result.summary['admitted']:
        result.payload[cs.WIRE_KEY]=cs.from_profile(profile,company_id=req.company_id,machine_id=req.machine_id)
    return normalize_decision(req,result.payload),profile


class BoundClueAliasesTests(unittest.TestCase):
    def test_unknown_and_request_aliases_discarded_original_full_partition_preserved(self):
        d,p=decision()
        self.assertEqual(tuple(x.quote for x in p.parts),tuple(x[1]for x in SEGMENTS))
        self.assertEqual(p.parts[2].retrieval_aliases,())
        self.assertEqual(p.parts[3].retrieval_aliases,())
        text='La pinza è aperta.'
        self.assertEqual(cs.support_match(d,text)['matched_support_part_ids'],())
        packet=dq.reasoning_packet(p)
        self.assertNotIn('retrieval_aliases',json.dumps(packet))
        self.assertEqual(packet['original_request'],QUERY)

    def test_actual_source_match_not_facet_annotation_and_fresh_source_digest(self):
        d,_=decision()
        supported='Il carrello si muove ma il materiale rimane fermo.'
        first=cs.support_match(d,supported)
        self.assertEqual(len(first['matched_support_part_ids']),1)
        self.assertEqual(first['source_text_sha256'],hashlib.sha256(supported.encode()).hexdigest())
        second=cs.support_match(d,'Packaging and dimensions.')
        self.assertEqual(second['matched_support_part_ids'],())
        self.assertNotEqual(first['source_text_sha256'],second['source_text_sha256'])

    def test_malformed_duplicate_and_long_alias_never_invalidates_original(self):
        for aliases in ([None], {'text':'invented'}, ['x']*3, ['x'*241], [True]):
            raw=classified(); raw['diagnostic_basis']['request_parts'][1]['retrieval_aliases']=aliases
            _,p=decision(raw)
            self.assertEqual(p.interpretation_status,'valid')
            self.assertEqual(p.parts[1].retrieval_aliases,())
            self.assertEqual(p.parts[1].quote,SEGMENTS[1][1])
        raw=classified();raw['diagnostic_basis']['request_parts'][1]['retrieval_aliases']=[ALIASES[1]]*2
        _,p=decision(raw)
        self.assertEqual(p.parts[1].retrieval_aliases,(ALIASES[1],))

    def test_code_value_unit_mutation_ignored_and_neighbour_wrong_source_does_not_match(self):
        query='Fault E101 at 6 bar.'
        segments=[('observation',query,'support')]
        for alias in ('Errore E102 a 6 bar.','Errore E101 a 60 bar.','Errore E101 a 6 psi.'):
            _,p=decision(classified(query,segments,[alias]),request(query))
            self.assertEqual(p.parts[0].retrieval_aliases,())
        d,p=decision(classified(query,segments,['Errore E101 a 6 bar.']),request(query))
        self.assertTrue(p.parts[0].retrieval_aliases)
        self.assertTrue(cs.support_match(d,'Errore E101 a 6 bar.')['matched_support_part_ids'])
        for source in ('Errore E102 a 6 bar.','Errore E101 a 8 bar.','Errore E101 a 6 psi.'):
            self.assertFalse(cs.support_match(d,source)['matched_support_part_ids'])

    def test_model_header_alone_cannot_match_descriptive_observation(self):
        query='UNIT-204 motor stopped.'
        raw=classified(query,[('observation',query,'support')],['UNIT-204 motore fermo.'])
        raw['exact_terms']=['UNIT-204']
        d,_=decision(raw,request(query))
        self.assertFalse(cs.support_match(d,'UNIT-204 installation dimensions.')['matched_support_part_ids'])
        self.assertTrue(cs.support_match(d,'UNIT-204 motore fermo.')['matched_support_part_ids'])
        raw['exact_terms']=[]
        d,_=decision(raw,request(query))
        self.assertFalse(cs.support_match(d,'UNIT-204 installation dimensions.')['matched_support_part_ids'])
        code='E101'
        d,_=decision(classified(code,[('observation',code,'support')],[[]]),request(code))
        self.assertTrue(cs.support_match(d,'Alarm E101 active')['matched_support_part_ids'])
        self.assertFalse(cs.support_match(d,'Alarm E102 active')['matched_support_part_ids'])

    def test_sentence_punctuation_is_not_code_but_internal_separators_are(self):
        for code in ('E101','E101.2','AA-12','MODEL300/40T'):
            query='Fault '+code+'.'
            raw=classified(query,[('observation',query,'support')],['Errore '+code])
            d,p=decision(raw,request(query))
            self.assertEqual(p.parts[0].retrieval_aliases,('Errore '+code,))
            self.assertTrue(cs.support_match(d,'Fault '+code+' motor stopped')['matched_support_part_ids'])
            self.assertFalse(cs.support_match(d,'Fault '+code+'.9 motor stopped')['matched_support_part_ids'])
        self.assertTrue(dq.protected_literals_present('Pressure 6bar.','Pressure 6 bar'))
        self.assertFalse(dq.protected_literals_present('Pressure 6bar.','Pressure 8 bar'))
        for wrong in ('-6','+6','6.5','6,5'):
            self.assertFalse(dq.protected_literals_present('6',wrong))
        self.assertFalse(dq.protected_literals_present('6bar','16,6bar'))
        self.assertTrue(dq.protected_literals_present('6.5 bar','6.5bar'))
        self.assertTrue(dq.protected_literals_present('E101 then E101','E101 motor stopped'))
        self.assertFalse(dq.protected_literals_present('E101 then E101','E102 motor stopped'))

    def test_duplicate_action_groups_count_one_original_part_at_max_existing_weight(self):
        d,_=decision()
        metrics=cs.priority_metrics(d,"Dopo l'attrezzaggio,",source_type='document',
                                   substantive_ps=False,metrics=cs._coverage)
        self.assertAlmostEqual(metrics['base_score'],.40/(.40+.22+.14))
        self.assertEqual(metrics['groups']['discriminants'],1)
        self.assertEqual(metrics['groups']['operating_conditions'],0)
        self.assertFalse(cs.support_match(d,"Dopo l'attrezzaggio,")['matched_support_part_ids'])

    def test_long_original_clue_retains_unique_alias_and_original_identity(self):
        action = ('After completing the documented preparation procedure with the '
                  'machine in its normal operating configuration and with the '
                  'operator having followed the established setup sequence for '
                  'the next operation as described in the applicable instructions,')
        self.assertGreater(len(action),240)
        observation='the drive moves but the stock remains stationary.'
        query=action+' '+observation
        segments=[('action_performed',action,'context'),('observation',observation,'support')]
        aliases=['Dopo la preparazione documentata nella configurazione normale e la sequenza di attrezzaggio prevista,',
                 'il trascinatore si muove ma il materiale rimane fermo.']
        d,p=decision(classified(query,segments,aliases),request(query))
        self.assertEqual(d.diagnostic_discriminants,(action[:240],))
        self.assertEqual(p.parts[0].quote,action)
        metrics=cs.priority_metrics(d,aliases[0],source_type='document',substantive_ps=False,metrics=cs._coverage)
        self.assertAlmostEqual(metrics['base_score'],.40/(.40+.22+.14))
        self.assertEqual(metrics['groups']['discriminants'],1)
        self.assertEqual(metrics['groups']['operating_conditions'],0)

    def test_colliding_long_clue_prefixes_cannot_borrow_either_alias(self):
        prefix=('The operator recorded the behavior after the documented preparation '
                'procedure in the normal operating configuration while checking the '
                'established setup sequence and the relevant machine conditions '
                'before the next scheduled production operation. ')
        self.assertGreater(len(prefix),240)
        first=prefix+'The drive moves.'
        second=prefix+'The drive stops.'
        query=first+' '+second
        segments=[('observation',first,'support'),('observation',second,'support')]
        aliases=['Il motore gira.','Il motore si arresta.']
        d,_=decision(classified(query,segments,aliases),request(query))
        self.assertEqual(len(d.diagnostic_observables),1)
        for text in aliases:
            metrics=cs.priority_metrics(d,text,source_type='document',substantive_ps=False,metrics=cs._coverage)
            self.assertEqual(metrics['groups']['observables'],0)

    def test_unicode_signs_and_unit_exponents_are_whole_protected_literals(self):
        for value,wrong in [('−6 bar','6 bar'),('−6 bar','+6 bar'),('− 6 bar','6 bar'),
                            ('5mm²','5cm²'),('5mm²','5mm³'),('5 m/s²','5 m/s³')]:
            query='Measured '+value+'.'
            segments=[('observation',query,'support')]
            d,p=decision(classified(query,segments,['Misurati '+value+'.']),request(query))
            self.assertTrue(p.parts[0].retrieval_aliases)
            self.assertTrue(cs.support_match(d,'Misurati '+value+'.')['matched_support_part_ids'])
            self.assertFalse(cs.support_match(d,'Misurati '+wrong+'.')['matched_support_part_ids'])
            _,p=decision(classified(query,segments,['Misurati '+wrong+'.']),request(query))
            self.assertEqual(p.parts[0].retrieval_aliases,())

    def test_negative_observation_and_nonperformed_role_remain_original(self):
        query='The motor did not turn. We did not reset it.'
        segments=[('observed_negative','The motor did not turn.','support'),
                  ('action_not_performed','We did not reset it.','context')]
        _,p=decision(classified(query,segments,['Il motore non ha girato.','Non lo abbiamo ripristinato.']),request(query))
        self.assertEqual([(x.role,x.basis_use,x.quote)for x in p.parts],
                         [(role,use,quote)for role,quote,use in segments])
        self.assertEqual(dq.reasoning_packet(p)['request_parts'][0]['quote'],'The motor did not turn.')

    def test_forged_serialized_stale_scope_and_mutated_carriers_rejected(self):
        d,_=decision(); carrier=d.root_clue_support
        self.assertIsNotNone(carrier)
        raw=classified();raw[cs.WIRE_KEY]={'matched_support_part_ids':['0:1:observation'],'query_sha256':carrier.query_sha256}
        self.assertIsNone(normalize_decision(request(),raw).root_clue_support)
        self.assertIsNone(cs.for_request(carrier,query=QUERY+' Extra',company_id=carrier.company_id,machine_id=carrier.machine_id))
        self.assertIs(cs.for_request(carrier,query=QUERY.replace(' ', '  \n'),
                      company_id=carrier.company_id,machine_id=carrier.machine_id),carrier)
        self.assertIsNone(cs.for_request(carrier,query=QUERY.upper(),
                         company_id=carrier.company_id,machine_id=carrier.machine_id))
        self.assertIsNone(cs.for_request(carrier,query=QUERY,company_id='foreign',machine_id=carrier.machine_id))
        for mutated in (replace(carrier,query_sha256='0'*64),
                        replace(carrier,parts=(replace(carrier.parts[0],basis_use='support'),*carrier.parts[1:]))):
            self.assertEqual(cs.support_match(replace(d,root_clue_support=mutated),'Dopo l’attrezzaggio,'),{})

    def test_schema_bounded_aliases_same_unit_identity_no_quote_generation(self):
        schema=dq.router_schema_with_diagnostic_basis(build_router_schema(('root_cause','ask')),original_query=QUERY)
        part=schema['schema']['properties']['diagnostic_basis']['properties']['request_parts']['items']
        self.assertEqual(part['properties']['retrieval_aliases']['maxItems'],2)
        self.assertEqual(part['properties']['retrieval_aliases']['items']['maxLength'],240)
        self.assertNotIn('quote',part['properties'])
        self.assertEqual(part['properties']['first_unit']['maximum'],len(dq.router_request_units(QUERY)['units'])-1)


class ActualAliasWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        segment_fixture.ActualRootRouterSegmentTests.setUpClass();cls.m=segment_fixture.ActualRootRouterSegmentTests.m

    @classmethod
    def tearDownClass(cls):
        segment_fixture.ActualRootRouterSegmentTests.tearDownClass()

    def test_actual_single_router_decision_prepare_and_selector_recompute_source(self):
        m=self.m;req=request();calls=[]
        def provider(messages,**kwargs):
            calls.append(kwargs)
            return classified(),'offline-router'
        with patch.object(m,'_v13_json_models',provider),patch.object(m,'_v13_gate_candidate_block',return_value=('Fixture source.',[])):
            raw=m._assistant_core_router_call(req,{})
        d=normalize_decision(req,raw)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['max_output_tokens'],m.ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS)
        self.assertIsNotNone(d.root_clue_support)
        body='Il carrello si muove ma il materiale rimane fermo. A released clamp allows carriage motion without stock movement.'
        row={'citation_id':'fixture-doc:p1','bubble_document_id':'fixture-doc','source_type':'document',
             'page_from':1,'page_to':1,'chunk_full':body,'snippet':body,'semantic_similarity':.8,
             'causal_strength_score':.25,'subsystem_score':.2,'context_fit_score':.1,'role_group':'core',
             'exact_machine_scope':True,'v13_score':1.0}
        seen=[]
        runtime=m._assistant_core_prepare_evidence_runtime()
        def select(candidates,**kwargs):
            self.assertIs(kwargs['decision'],d)
            seen.extend(candidates)
            return m._assistant_core_root_source_selection(candidates,**kwargs)
        result=eo.assistant_core_prepare_evidence(req,{'candidates':[row]},d,runtime=replace(runtime,
            _assistant_core_root_diagnostic_evidence_assurance=lambda r,p,d:p,
            _v13_current_budget=lambda:None,_v13_rescore_root_candidates=lambda q,c:c,
            _assistant_core_root_source_selection=select))
        self.assertTrue(seen)
        self.assertTrue(result['retrieval']['candidates'])
        forged=dict(seen[0],chunk_full='Packaging and dimensions.',snippet='Packaging and dimensions.',
                    matched_support_part_ids=['forged'],source_text_sha256='0'*64)
        fresh=m._assistant_core_root_source_selection([forged],limit=6,decision=d)
        audit=fresh.summary['core_context_assessments'][0]
        self.assertFalse(audit['matched_support_part_ids'])

    def test_smart_metrics_keep_existing_tag_route_ask_has_no_carrier(self):
        m=self.m;d,_=decision()
        candidate={'source_type':'document','chunk_full':'Packaging.','assistant_core_facet_hits':[SEGMENTS[1][1]]}
        root=m._assistant_core_diagnostic_priority_metrics(candidate,d)
        smart=m._assistant_core_diagnostic_priority_metrics(candidate,replace(d,effective_mode='smart_diagnostic',root_clue_support=None))
        self.assertEqual(root['groups']['observables'],0)
        self.assertEqual(smart['groups']['observables'],1)
        self.assertIsNone(normalize_decision(request(mode='ask'),{**classified(),cs.WIRE_KEY:d.root_clue_support}).root_clue_support)


if __name__=='__main__':
    unittest.main(verbosity=2)
