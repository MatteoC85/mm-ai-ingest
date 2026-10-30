"""Final family-context support across real admission, review and completion.

Only SQL rows, allowance and model replies are fixtures. The tests do not prove
LLM selection/coverage quality; they exercise exact requirement identities and
the scope of lossless supplements after a successful source-bound replacement.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import _finish_ask_validation, decorate_response
from machinemind.ask import execution, validation, procedure_review, request_completion
from machinemind.infrastructure import request_budget
from machinemind.evidence.contracts import EvidenceContractError
import test_ask_conditions_family_admission as families
import test_ask_conditions_question_contract as questions
import test_ask_bounded_phase_admission as phases
import test_ask_repair_capacity as capacity


class ReplacementFixture(families.OwnedFamilyFixture):
    def __init__(self, *args, negative=None, **kwargs):
        self.negative = negative
        super().__init__(*args, **kwargs)

    def provider(self, messages, **kwargs):
        if 'selected_procedure_citation_id' in kwargs['json_schema']['schema']['properties']:
            return super().provider(messages, **kwargs)
        self.calls.append(('writer', deepcopy(messages), deepcopy(kwargs)))
        block = messages[1]['content'].split('SOURCES:\n', 1)[1]
        self.writer_packets.append(block)
        self.ids = {key: next(cid for cid in re.findall(r'\[([^\]\n]+)\]', block) if cid.startswith(key+':'))
                    for key, *_ in self.children['procedure:'+self.chosen]}
        return self.reply({'answer_status':'answered', 'grounded_points':[
            {'text':'Use trained personnel and isolate energy.', 'citation_ids':[self.ids['step:install1']]},
            {'text':'Check Ready after the restoration.', 'citation_ids':[self.ids['step:install4']]}]},
            'offline-writer', messages, kwargs)

    def reviewer(self, messages, **kwargs):
        self.calls.append(('reviewer', deepcopy(messages), deepcopy(kwargs)))
        self.review_packets.append(messages[1]['content'])
        self.schema = deepcopy(kwargs['json_schema'])
        current = messages[1]['content'].split('CURRENT_ANSWER:\n', 1)[1].split('\n\nSOURCES:', 1)[0]
        self.reviewed = ('Use trained personnel with the cycle stopped and energy isolated.\n'
            'Wear gloves and goggles and keep STOP pressed.\n'
            'Verify straight unobstructed entry; do not enter with the feeder enabled.')
        if self.negative == 'claim':
            self.reviewed += '\nSet pressure to 900 bar.'
        cids = [self.ids['step:install'+str(n)] for n in (1, 2, 3)]
        if self.negative == 'foreign':
            cids.append(self.prepared['citations'][0]['citation_id'])
        covered = list(self.decision.required_facets)
        if self.negative == 'translated_key':
            covered = ['Controlli e precauzioni della fase richiesta']
        reply = {'outcome':'rewrite', 'reply_mode':'replace', 'answer':self.reviewed,
            'edits':[], 'reason':'Offline replacement excludes unrelated startup, retains applicable safety.',
            'covered_facets':covered, 'missing_facets':[],
            'covered_answer_types':list(self.decision.required_answer_types), 'missing_answer_types':[],
            'expected_list_items':[], 'covered_list_items':[], 'missing_list_items':[],
            'enumeration_requested':False, 'citation_ids':cids,
            'operation_boundary':{'complete':self.negative != 'boundary',
                'checked_blocks':[b['block_id'] for b in procedure_review.block_layout(current)['blocks']], 'changes':[]}}
        if self.negative == 'decline':
            reply.update(outcome='no_sources', answer='', covered_facets=[],
                missing_facets=['Additional explicitly requested category lacks source support'])
        if self.negative == 'retain':
            reply.update(outcome='pass', reply_mode='retain', answer='',
                citation_ids=[self.ids['step:install1']])
        result = self.reply(reply, 'offline-reviewer', messages, kwargs)
        if self.negative == 'revoke':
            self.allowance = frozenset(s for s in self.allowance if s.source_id != 'install2')
        return result


class ReviewedNoteScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        families.AskConditionsFamilyAdmissionTests.setUpClass()
        cls.m = families.AskConditionsFamilyAdmissionTests.m

    @classmethod
    def tearDownClass(cls):
        families.AskConditionsFamilyAdmissionTests.tearDownClass()

    def scenario(self, *, negative=None, bounded=False):
        now = [1000.0]
        with patch.object(request_completion, '_monotonic', side_effect=lambda:now[0]), \
             patch.object(request_budget, '_monotonic', side_effect=lambda:now[0]):
            completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'), started=now[0])
            query = 'Before cartridge insertion, which checks and safety conditions apply?'
            q = questions.AskConditionsQuestionContractTests()
            d = q.decision(query)
            if bounded:
                query = 'After cleaning, restore the housing before restart.'
                d = phases.decision(self.m, query, ('After cleaning', 'restore the housing', 'before restart'))
            h = ReplacementFixture(self, decision=d, query=query, kind='procedure', transport=True, negative=negative)
            try:
                capacity.AskRepairCapacityTests.first_transport(self, h, failed=False)
                h.prepared['assistant_core_decision'] = {'supported':True, 'strict_supported':True}
                h.owner._prepared_retrieval = deepcopy(h.prepared)
                h.generated = execution.synthesize_ask(h.request, h.prepared, h.decision, runtime=h.runtime)
                flow = h.owner._response_flow
                flow.execution = replace(flow.execution, _v13_json_models=h.reviewer)
                eruntime = replace(h.runtime, _v13_json_models=flow._verification_model)
                vruntime = validation.bind_ask_validation(h.owner._validation_factory(), execution_runtime=eruntime)
                vruntime = replace(vruntime, _v13_current_budget=lambda:completion.budget)
                if negative == 'revoke':
                    with self.assertRaises(EvidenceContractError):
                        validation.validate_response(h.generated, h.request, h.prepared, h.decision, runtime=vruntime)
                    self.assertEqual(len(h.posts), 4)
                    return
                result = validation.validate_response(h.generated, h.request, h.prepared, h.decision, runtime=vruntime)
                projected = decorate_response(_finish_ask_validation(result), h.request, h.decision)
                h.owner._finalize_core(h.request, projected)
                self.assertTrue(flow.completed)
                self.assertEqual(len(h.posts), 4)
                self.assertEqual([x[0] for x in h.calls], ['selector', 'writer', 'reviewer'])
                self.assertFalse(any(k.startswith('_assistant_core_') for k in projected))
                return {'result':projected, 'schema':h.schema, 'decision':h.decision,
                    'review':h.review_packets[0], 'writer':h.writer_packets[0], 'rows':deepcopy(h.rows),
                    'reviewed':h.reviewed, 'ids':h.ids,
                    'seed':h.prepared['citations'][0]['citation_id']}
            finally:
                h.close()
                completion.close()

    def test_exact_facet_schema_and_additional_missing_category(self):
        legacy = self.m._assistant_core_contract_verifier_schema()
        layout = procedure_review.block_layout('1. Keep the energy isolated.')
        facets = ['Original requirement identity', 'Another explicit category']
        schema = procedure_review.review_schema(legacy, layout, facets, ['safety_conditions'])['schema']
        self.assertEqual(schema['properties']['covered_facets']['items'].get('enum'), facets)
        self.assertNotIn('enum', schema['properties']['missing_facets']['items'])
        self.assertNotIn('minItems', schema['properties']['covered_facets'])
        empty = procedure_review.review_schema(legacy, layout, [], [])['schema']
        self.assertNotIn('enum', empty['properties']['covered_facets']['items'])
        self.assertEqual(legacy, self.m._assistant_core_contract_verifier_schema())

    def test_full_review_projects_final_notes_and_keeps_new_selected_safety(self):
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                x = self.scenario(bounded=bounded)
                result = x['result']; c = result['meta']['assistant_core_validation']['answer_contract']
                self.assertEqual(result['status'], 'answered', c)
                self.assertTrue(c['complete'])
                self.assertTrue(c['semantic_answer_bound'])
                self.assertTrue(c['source_preservation']['all_present'])
                self.assertEqual({n['citation_id'] for n in c['source_preservation']['source_units']},
                    {x['ids']['step:install'+str(n)] for n in (1, 2, 3)})
                self.assertNotIn(x['ids']['step:install4'], {c['citation_id'] for c in result['citations']})
                self.assertNotIn('Ready', result['answer'])
                for phrase in ('Solo personale addestrato.',
                        'Indossare guanti e occhiali; mantenere STOP premuto.',
                        'Non entrare con alimentatore abilitato.'):
                    self.assertIn(phrase, result['answer'])
                for _, _, _, body in x['rows'][:5]:
                    self.assertIn(body, x['writer'])
                    self.assertIn(body, x['review'])
                self.assertTrue(c['semantic_verifier']['review_evidence']['complete_primary_representation'])

    def test_foreign_unreviewed_source_is_not_added_to_supplements(self):
        x = self.scenario(negative='foreign')
        self.assertEqual(x['result']['status'], 'answered')
        self.assertNotIn(x['seed'], {c['citation_id'] for c in x['result']['citations']})
        self.assertNotIn('Avvio solo volontario.', x['result']['answer'])

    def test_retain_cannot_discard_an_unchanged_note_binding_by_omitting_its_id(self):
        x = self.scenario(negative='retain')
        self.assertEqual(x['result']['status'], 'answered')
        self.assertIn('Verificare il consenso Ready dopo il ripristino.', x['result']['answer'])
        self.assertIn(x['ids']['step:install4'], {c['citation_id'] for c in x['result']['citations']})
        proof = x['result']['meta']['assistant_core_validation']['answer_contract']['source_preservation']
        self.assertIn(x['ids']['step:install4'], {n['citation_id'] for n in proof['source_units']})

    def test_decline_translated_identity_and_bad_boundary_do_not_gain_acceptance(self):
        for negative in ('decline', 'translated_key', 'boundary'):
            with self.subTest(negative=negative):
                x = self.scenario(negative=negative)
                c = x['result']['meta']['assistant_core_validation']['answer_contract']
                self.assertEqual(x['result']['status'], 'no_sources')
                self.assertFalse(c['complete'])
                self.assertFalse(x['result'].get('_assistant_core_repair_needed'))
                if negative == 'translated_key':
                    self.assertTrue(c['semantic_answer_bound'])
                    self.assertEqual(c['reason'], 'procedure_family_semantic_review_incomplete')

    def test_claim_removed_after_review_still_breaks_answer_binding(self):
        x = self.scenario(negative='claim')
        c = x['result']['meta']['assistant_core_validation']['answer_contract']
        self.assertEqual(x['result']['status'], 'no_sources')
        self.assertFalse(c['semantic_answer_bound'])
        self.assertEqual(c['reason'], 'semantic_answer_changed_after_review')

    def test_current_authority_revocation_after_review_blocks_final_support(self):
        self.scenario(negative='revoke')


if __name__ == '__main__':
    unittest.main(verbosity=2)
