"""Narrow reviewer routing; real packet/decoder/validation, no provider calls.

Model replies and elapsed provider durations are fixtures, not quality or speed
measurements. The source-order integration harness denies network and database I/O.
"""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import execution, review_evidence
from machinemind.infrastructure import openai_transport
from machinemind.infrastructure.provider_failures import ProviderResponseError
import test_source_ordered_review as wired


class PacketObserver:
    def __init__(self, render, variant):
        self.render, self.variant = render, variant

    def consume(self, name, arguments, work):
        return work()

    def procedure_review_sources(self, rows, **limits):
        if self.variant == 'missing_packet':
            return None
        packet = review_evidence.compile_packet(primary=rows, extension=[],
            render=self.render, **limits)
        diagnostic = packet.diagnostic(reused=False)
        if self.variant == 'incomplete_packet':
            diagnostic['complete_primary_representation'] = False
        sources = packet.sources
        if self.variant == 'source_body_changed':
            sources = sources.replace('zero energy', 'live voltage')
        return list(packet.rows), sources, diagnostic


class ProcedureReviewerRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wired.WiredSourceOrderedReviewTests.setUpClass()
        cls.m = wired.WiredSourceOrderedReviewTests.m

    @classmethod
    def tearDownClass(cls):
        wired.WiredSourceOrderedReviewTests.tearDownClass()

    def scenario(self, *, variant=None, wire_change=None, fallback_failure=None,
                 execution_module=None, **fixture):
        fixture.setdefault('reply_mode', 'edit' if wire_change in {'out_of_bounds','wrong_number'}
                           or fixture.get('note_mutation') else 'retain')
        # The external offline comparison can supply the actual pre-change
        # module. It never synthesizes a "baseline" by rewriting model options.
        target = execution_module or execution
        original = target.verify_or_repair_answer
        captured, attempts, simulated_seconds = [], [], []
        before_after = []
        fallback_budget = SimpleNamespace(llm_calls=0,
            grant_retry_allowance=lambda **kw: None)

        def invoke(**kw):
            runtime = kw['runtime']
            provider = runtime._v13_json_models
            observer = PacketObserver(runtime._v13_sources_block, variant)
            context = deepcopy(kw['repair_context'])
            if variant in {'sequence_incomplete', 'notes_missing'}:
                context['procedure_structure'][
                    'sequence_complete' if variant == 'sequence_incomplete' else 'source_notes_present'] = False
            if variant == 'mixed':
                context['procedure_structure']['source_sequence_required'] = False
            if variant == 'stale_draft':
                context['procedure_structure']['answer_sha256'] = '0' * 64
            if variant == 'missing_draft_note':
                kw['answer'] = kw['answer'].replace('Keep STOP pressed.','')
                context['procedure_structure']['answer_sha256'] = execution._procedure_review.digest(kw['answer'])
            if variant == 'translation':
                context['procedure_translation_language'] = 'en'
            repair_variants = {'prior_contract':'first_answer_contract',
                'repair_facets':'missing_answer_facets', 'repair_types':'missing_answer_types',
                'repair_items':'missing_list_items'}
            if variant in repair_variants:
                context[repair_variants[variant]] = {'complete':False} if variant == 'prior_contract' else ['missing']
            kw['repair_context'] = context
            if variant == 'factual':
                kw['decision'] = replace(kw['decision'], information_task='factual', request_kind='information')
            if variant in {'root', 'smart'}:
                kw['request'] = replace(kw['request'], requested_mode=variant)
                kw['decision'] = replace(kw['decision'], effective_mode=variant)
            options = dict(evidence_observer=observer,
                V13_FAST_MODEL='gpt-5.6-terra', V13_PLANNER_MODEL='gpt-5.6-luna')
            budget = runtime._v13_current_budget()
            options['_v13_current_budget'] = lambda: budget
            if variant == 'custom_models':
                options['V13_FAST_MODEL'] = 'custom-reviewer'
            if variant == 'reversed_configuration':
                options.update(V13_FAST_MODEL='gpt-5.6-luna', V13_PLANNER_MODEL='gpt-5.6-terra')
            if variant == 'unguarded':
                options['evidence_admission'] = None

            def model(messages, **parameters):
                captured.append((deepcopy(messages), deepcopy(parameters)))

                def respond(messages, *, model, **parameters):
                    attempts.append(model)
                    fallback_budget.llm_calls += 1
                    if fallback_failure and len(attempts) == 1:
                        simulated_seconds.append(25.001)
                        if fallback_failure == 'budget':
                            raise self.m._V13BudgetExceeded('fixture budget exhausted')
                        raise ProviderResponseError('output_limit')
                    simulated_seconds.append(6.884)
                    props = parameters['json_schema']['schema']['properties']
                    if ('operation_boundary' not in props
                            or 'enum' not in props['citation_ids']['items']):
                        # These variants deliberately leave the compact
                        # source-ordered protocol. A valid semantic rejection
                        # avoids making a fixture KeyError look like a real
                        # provider failure and triggering a second dispatch.
                        return {'outcome':'no_sources', 'answer':'',
                            'reason':'offline ineligible-protocol fixture',
                            'citation_ids':[], 'covered_facets':[],
                            'missing_facets':['safety'],
                            'covered_answer_types':[],
                            'missing_answer_types':['safety_conditions'],
                            'expected_list_items':[], 'covered_list_items':[],
                            'missing_list_items':[]}
                    out, _ = provider(messages, **parameters)
                    if wire_change == 'out_of_bounds':
                        out['edits'][0]['original']['last'] = 999
                    elif wire_change == 'missing_facets':
                        out['covered_facets'] = []
                        out['missing_facets'] = ['safety']
                    elif wire_change == 'missing_types':
                        out['covered_answer_types'] = []
                        out['missing_answer_types'] = ['safety_conditions']
                    elif wire_change == 'boundary_incomplete':
                        out['operation_boundary']['complete'] = False
                    elif wire_change == 'missing_checked_block':
                        out['operation_boundary']['checked_blocks'].pop()
                    elif wire_change == 'wrong_number':
                        out['edits'][0]['replacement_parts'] = ['9. Feed while guards are open.']
                    return out

                return openai_transport.json_models(messages, **parameters,
                    default_model='gpt-5.6-terra', normalize_models_fn=lambda values:list(values),
                    current_budget_fn=lambda:fallback_budget, responses_json_fn=respond,
                    chat_json_fn=respond, budget_exceeded_type=self.m._V13BudgetExceeded)

            options['_v13_json_models'] = model
            kw['runtime'] = replace(runtime, **options)
            before = deepcopy({key:kw[key] for key in ('answer','candidates','repair_context')})
            try:
                # Freeze only remaining wall time. This is the real budget
                # object, with its call/cost limits and exception behavior;
                # model choice must not appear to alter timeout by a second
                # merely because two offline scenarios ran at different times.
                with patch.object(budget, 'remaining', return_value=100.0):
                    return original(**kw)
            finally:
                after = {key:kw[key] for key in before}
                before_after.append((before, deepcopy(after)))

        completion = None if variant == 'no_completion' else SimpleNamespace()
        with ExitStack() as guards:
            guards.enter_context(patch.object(execution, 'verify_or_repair_answer', side_effect=invoke))
            guards.enter_context(patch.object(target, '_current_completion', return_value=completion))
            if target is not execution:
                guards.enter_context(patch.object(target, '_admit_input',
                    side_effect=lambda q,data,d,**kw:data))
            result, _, answer = wired.WiredSourceOrderedReviewTests().run_review(
                family_required=True, **fixture)
        for before, after in before_after:
            self.assertEqual(before, after)
        return result, captured, attempts, sum(simulated_seconds), answer

    def test_complete_owned_packet_uses_existing_luna_first_and_real_final_guards(self):
        result, calls, attempts, _, answer = self.scenario(reply_mode='retain')
        self.assertEqual(attempts, ['gpt-5.6-luna'])
        self.assertEqual(calls[0][1]['models'], ['gpt-5.6-luna','gpt-5.6-terra'])
        self.assertEqual(calls[0][1]['effort'], 'low')
        self.assertEqual(calls[0][1]['max_output_tokens'], 3200)
        contract = result['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['complete'])
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertTrue(contract['semantic_verifier']['operation_boundary_validation']['complete'])
        self.assertTrue(contract['semantic_verifier']['procedure_review']['source_note_bindings_complete'])
        self.assertEqual(result['answer'].splitlines(), [x for x in answer.splitlines() if x])

    def test_all_ineligible_boundaries_keep_configured_order(self):
        variants = ('missing_packet','incomplete_packet','sequence_incomplete','notes_missing',
            'translation','prior_contract','repair_facets','repair_types','repair_items',
            'mixed','stale_draft','factual','root','smart','unguarded','no_completion')
        for variant in variants:
            with self.subTest(variant=variant):
                _, calls, _, _, _ = self.scenario(variant=variant)
                self.assertEqual(calls[0][1]['models'], ['gpt-5.6-terra','gpt-5.6-luna'])
        for variant, models in (
            ('custom_models',['custom-reviewer','gpt-5.6-luna']),
            ('reversed_configuration',['gpt-5.6-luna','gpt-5.6-terra'])):
            with self.subTest(variant=variant):
                _, calls, _, _, _ = self.scenario(variant=variant)
                self.assertEqual(calls[0][1]['models'], models)

    def test_independent_source_safety_schema_and_limits_remain_required(self):
        _, calls, _, _, _ = self.scenario(reply_mode='retain')
        parameters = calls[0][1]
        schema = parameters['json_schema']['schema']
        props = schema['properties']
        self.assertEqual(props['reply_mode']['enum'], ['retain','reject'])
        self.assertEqual(props['outcome']['enum'], ['pass','no_sources'])
        self.assertEqual(props['edits']['maxItems'], 0)
        self.assertEqual(props['answer']['enum'], [''])
        self.assertEqual(props['citation_ids']['items']['enum'], ['s1','s2','s3'])
        self.assertIn('operation_boundary', schema['required'])
        self.assertEqual(props['operation_boundary']['properties']['complete'], {'type':'boolean'})
        for obligation in ('covered_facets','missing_facets','covered_answer_types',
                           'missing_answer_types'):
            self.assertIn(obligation, schema['required'])
        self.assertEqual(parameters['timeout'], 100)
        self.assertEqual(parameters['max_output_tokens'], 3200)

    def test_semantic_rejection_does_not_try_another_model(self):
        result, _, attempts, _, _ = self.scenario(reject=True)
        self.assertEqual(attempts, ['gpt-5.6-luna'])
        self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])
        self.assertEqual(result['meta']['assistant_core_validation']['answer_contract']
                         ['semantic_verifier']['outcome'], 'no_sources')

    def test_invalid_wire_safety_order_and_bindings_remain_failed(self):
        cases = [dict(note_mutation='delete'), dict(note_mutation='swap'), dict(unknown_alias=True),
            dict(wire_change='out_of_bounds'), dict(wire_change='wrong_number'),
            dict(wire_change='missing_facets'), dict(malformed='missing_audit'),
            dict(proof_tamper='source'), dict(proof_tamper='closure'),
            dict(proof_tamper='raw_audit_digest')]
        for options in cases:
            with self.subTest(options=options):
                result, _, attempts, _, _ = self.scenario(**options)
                self.assertEqual(attempts, ['gpt-5.6-luna'])
                self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])

    def test_source_catalog_mismatch_rejects_before_any_dispatch(self):
        result, calls, attempts, _, _ = self.scenario(variant='source_body_changed')
        self.assertEqual(calls, [])
        self.assertEqual(attempts, [])
        self.assertEqual(result['status'], 'no_sources')

    def test_existing_provider_failure_can_fall_back_but_budget_failure_cannot(self):
        result, _, attempts, _, _ = self.scenario(fallback_failure='output_limit', reply_mode='retain')
        self.assertEqual(attempts, ['gpt-5.6-luna','gpt-5.6-terra'])
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['complete'])
        with self.assertRaises(self.m._V13BudgetExceeded):
            self.scenario(fallback_failure='budget')

if __name__ == '__main__':
    unittest.main(verbosity=2)
