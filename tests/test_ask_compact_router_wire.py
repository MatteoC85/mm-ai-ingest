"""Compact ASK wire preserves semantic values before the existing pipeline."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys,time
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import router_wire, routing_runtime
import test_ask_conditions_family_admission as owned
import test_ask_router_family_context as routed
import test_ask_procedure_context_recovery as context


def sample(rule):
    if 'enum' in rule:return deepcopy(rule['enum'][0])
    typ=rule.get('type')
    if isinstance(typ,list):typ=next(t for t in typ if t!='null')
    if typ=='object':return {k:sample(v) for k,v in rule['properties'].items()}
    if typ=='array':return []
    return {'string':'','boolean':False,'number':.98,'integer':1,'null':None}[typ]


def wire_reply(raw, wire_schema):
    """Provider fixture only; missing historical raw fields are explicit defaults."""
    value=sample(wire_schema['schema'])
    for name,item in raw.items():
        alias=router_wire.ALIASES.get(name)
        if alias not in value:continue
        if name=='procedure_scope' and item=='':
            # Older normalized fixtures use empty for unspecified scope. The
            # actual strict wire requires its explicit enum value instead.
            continue
        if name=='facet_queries':
            item=[{router_wire.FACET_ALIASES[k]:deepcopy(v) for k,v in f.items()} for f in item]
        value[alias]=deepcopy(item)
    return value


class AskCompactRouterWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.AskConditionsFamilyAdmissionTests.setUpClass();cls.m=owned.AskConditionsFamilyAdmissionTests.m
    @classmethod
    def tearDownClass(cls):owned.AskConditionsFamilyAdmissionTests.tearDownClass()

    def full_schema(self, parent=True):
        m=self.m
        full=m._ask_factual_coverage.router_schema(m._ask_procedure_scope.router_schema(
            m._ask_request_readiness.router_schema(m._retrieval_precision_facts.router_schema_with_scalar_target(
                m.build_router_schema(['ask'])))))
        if parent:
            full['schema']['properties']['selected_procedure_parent_id']={'type':['string','null'],'enum':[None,'procedure:owned:p1']}
            full['schema']['required'].append('selected_procedure_parent_id')
        return full

    def semantic_reply(self):
        value=sample(self.full_schema()['schema'])
        value.update(request_kind='factual',information_task='numeric_specification',effective_mode='ask',confidence=.98,
            requested_mode_fit=True,evidence_state='refine',evidence_policy='machine_sources_required',
            required_answer_types=['numeric_value','checklist','safety_conditions','ordered_actions'],
            dense_queries=['UNIQUE GLOBAL: pressure 192.5 kPa for assembly B; retain negation'],
            lexical_queries=['"codice-G7" "pressione iniziale"'],exact_terms=['G7','192.5 kPa'],
            preferred_source_types=['procedure','step','document'],source_type_policy='require',
            diagnostic_subsystems=['explicit subsystem'],diagnostic_observables=['observed alarm G7'],
            diagnostic_operating_conditions=['during TEST'],diagnostic_discriminants=['after change'],
            diagnostic_exclusions=['no recorded thermal alarm'],missing_information=['pressure was not measured'],
            request_readiness='answerable',procedure_scope='conditions_only',
            selected_procedure_parent_id='procedure:owned:p1')
        value['facet_queries']=[dict(facet='Controlli prima del carico; preserve full breadth',answer_type='checklist',must_cover=True,
            dense_queries=['checks before loading','controlli prima del carico'],lexical_queries=['"prima del carico"'],
            exact_terms=['kg'],preferred_source_types=['procedure','step']),
            dict(facet='Optional document cross-reference',answer_type='source_locations',must_cover=False,
                dense_queries=[],lexical_queries=[],exact_terms=[],preferred_source_types=['document'])]
        value['required_facets']=[value['facet_queries'][0]['facet']]
        value['scalar_target']={'state':'single_scalar','property_labels':['maximum permitted mass'],
            'query_quote':'maximum permitted mass'}
        value['procedure_phase']={k:'Literal user quote '+str(i) for i,k in enumerate(value['procedure_phase'])}
        return value

    def test_full_semantic_roundtrip_keeps_unique_global_probes_and_all_extensions(self):
        full=self.full_schema();before=deepcopy(full);raw=self.semantic_reply()
        wire=wire_reply(raw,router_wire.schema(full))
        expanded=router_wire.expand(wire,full,allowed_effective_modes=('ask',))
        self.assertEqual(expanded,raw);self.assertEqual(full,before)
        self.assertEqual(expanded['dense_queries'],raw['dense_queries'])
        self.assertNotIn(raw['dense_queries'][0],expanded['facet_queries'][0]['dense_queries'])
        self.assertIn('ordered_actions',expanded['required_answer_types'])
        self.assertNotIn('ordered_actions',[f['answer_type'] for f in expanded['facet_queries']])
        self.assertNotIn('effective_mode',wire);self.assertNotIn('required_facets',wire)

    def test_refusal_unknown_readiness_language_and_scope_values_are_not_reinterpreted(self):
        full=self.full_schema()
        for kind in ('unsafe_request','out_of_scope','ambiguous','fault_diagnostic','factual'):
            for readiness in ('answerable','underspecified','contradictory','uninterpretable'):
                with self.subTest(kind=kind,readiness=readiness):
                    raw=self.semantic_reply();raw.update(request_kind=kind,request_readiness=readiness,
                        clarification_question='Quale osservazione è già disponibile?',safety_reason='Non bypassare il riparo.',
                        out_of_scope_reason='Unrelated request.',selected_procedure_parent_id=None)
                    self.assertEqual(router_wire.expand(wire_reply(raw,router_wire.schema(full)),full,
                        allowed_effective_modes=('ask',)),raw)

    def test_malformed_wire_types_missing_fields_enum_and_unknown_keys_fail_closed(self):
        full=self.full_schema();good=wire_reply(self.semantic_reply(),router_wire.schema(full))
        mutations=[lambda x:x.pop('mi'),lambda x:x.update(extra='invented'),lambda x:x.update(c=True),
            lambda x:x.update(c=math.nan),lambda x:x.update(pp='unemitted-parent'),
            lambda x:x.update(pp=' procedure:owned:p1'),lambda x:x['f'][0].update(m='true'),
            lambda x:x['f'][0].update(d='not an array'),lambda x:x.update(d=['x']*6)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value=deepcopy(good);mutate(value)
                with self.assertRaises(router_wire.RouterWireError):router_wire.expand(value,full,allowed_effective_modes=('ask',))
        for modes in (['ask'],('root_cause',),('ask','root_cause')):
            with self.assertRaises(router_wire.RouterWireError):router_wire.expand(good,full,allowed_effective_modes=modes)

    def test_canonical_extensions_cannot_disappear_and_schema_uses_supported_subset(self):
        full=self.full_schema();changed=deepcopy(full)
        changed['schema']['properties']['future_semantic_extension']={'type':'string'}
        changed['schema']['required'].append('future_semantic_extension')
        with self.assertRaises(router_wire.RouterWireError):router_wire.schema(changed)
        allowed={'type','enum','properties','additionalProperties','required','items','maxItems','minItems',
            'minLength','maxLength','minimum','maximum','description','anyOf'}
        def check(rule):
            self.assertTrue(set(rule).issubset(allowed))
            if rule.get('type')=='object':
                self.assertIs(rule['additionalProperties'],False)
                self.assertEqual(set(rule['required']),set(rule['properties']))
            for child in rule.get('properties',{}).values():check(child)
            if 'items'in rule:check(rule['items'])
            for child in rule.get('anyOf',[]):check(child)
        compact=router_wire.schema(full);check(compact['schema'])
        for original,alias in router_wire.ALIASES.items():
            if original in full['schema']['properties']:
                self.assertIn(original,compact['schema']['properties'][alias]['description'])

    def active_run(self, callback):
        _,token=routing_runtime.activate(time.monotonic())
        try:return callback()
        finally:routing_runtime.deactivate(token)

    def encoded_provider(self, original):
        def provider(owner,messages,**kw):
            raw,model=original(owner,messages,**kw)
            if kw['json_schema']['name']=='machinemind_ask_compact_router_v1':
                self.assertEqual(len(kw['models']),1);self.assertLessEqual(kw['timeout'],20)
                self.assertEqual(kw['max_output_tokens'],self.m.ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS)
                raw=wire_reply(raw,kw['json_schema'])
            return raw,model
        return provider

    def test_actual_core_compact_wire_preserves_typed_family_and_manual_facts(self):
        original=routed.RouterFixture.router
        with patch.object(routed.RouterFixture,'router',self.encoded_provider(original)):
            case=routed.AskRouterFamilyContextTests();case.m=self.m
            self.active_run(case.test_mandatory_manual_only_value_and_menu_are_refined_and_retained)
            self.active_run(case.test_explicit_null_does_not_promote_optional_parent_citations)

    def test_real_r5_recorded_scope_through_compact_wire_reaches_step3_and_review(self):
        # The old raw model JSON was not retained. Existing pinned test replays
        # the exact normalized R5 scope/source subset and current family; model
        # selection/replies and missing raw extension defaults remain fixtures.
        original=routed.RouterFixture.router
        with patch.object(routed.RouterFixture,'router',self.encoded_provider(original)):
            case=context.AskProcedureContextRecoveryTests();case.m=self.m
            self.active_run(case.test_real_r5_mixed_steps_recover_all_ten_current_steps_without_foreign_pressure)

    def test_bad_compact_model_reply_does_not_fall_back_or_reach_refinement(self):
        h=routed.RouterFixture(self)
        original=routed.RouterFixture.router
        def broken(owner,messages,**kw):
            raw,model=original(owner,messages,**kw)
            raw=wire_reply(raw,kw['json_schema']);raw['c']=True
            return raw,model
        try:
            with patch.object(routed.RouterFixture,'router',broken),patch.object(self.m,
                    '_assistant_core_deterministic_router_fallback',side_effect=AssertionError('NO_GUESSED_PLAN')):
                with self.assertRaises(routing_runtime.AskRouterUnavailable):self.active_run(h.run_core)
            self.assertEqual(h.refine_calls,[])
            self.assertEqual([c[0] for c in h.calls],['router'])
        finally:h.close()

    def test_literal_output_serialization_is_smaller_without_semantic_probe_loss(self):
        raw=self.semantic_reply();wire=wire_reply(raw,router_wire.schema(self.full_schema()))
        old=json.dumps(raw,ensure_ascii=False,separators=(',',':')).encode()
        new=json.dumps(wire,ensure_ascii=False,separators=(',',':')).encode()
        self.assertLess(len(new),len(old))
        self.assertEqual(router_wire.expand(wire,self.full_schema(),allowed_effective_modes=('ask',)),raw)


if __name__=='__main__':unittest.main()
