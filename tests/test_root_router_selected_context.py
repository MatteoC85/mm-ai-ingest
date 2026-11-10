"""Current router-selected source context is recall, never causal evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from assistant_core_v2 import normalize_decision
from machinemind.retrieval import root_clue_support as cs
import test_root_request_segments as segments
from test_root_supported_context_capacity import candidate

BODY = ('The setup selector releases the gripping force so that the stock is not pulled. '
        'The slide can complete its travel without moving the material.')


class SelectedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        segments.ActualRootRouterSegmentTests.setUpClass.__func__(cls)
        if Path(cls.m.__file__).resolve() != Path(__file__).resolve().parents[1] / 'main.py':
            raise AssertionError('ROOT_SELECTED_CONTEXT_WRONG_SOURCE_MODULE')
        for name in ('socket.socket.connect', 'socket.getaddrinfo', 'google.auth.default', 'sqlite3.connect'):
            cls.guards.enter_context(patch(name, segments.denied))
    tearDownClass = classmethod(segments.ActualRootRouterSegmentTests.tearDownClass.__func__)

    def request(self, **changes):
        values = dict(query=segments.QUERY, requested_mode='root_cause', response_language='en',
                      company_id='fixture-company', machine_id='fixture-machine', ai_scope='machine_all', top_k=6)
        values.update(changes)
        return segments.AssistantCoreRequest(**values)

    def route(self, row=None, *, chosen=True, mutate_during_provider=False, raw=None, request=None):
        row = row or candidate('fixture-manual:p1', source='fixture-manual', chunk_full=BODY)
        request = request or self.request()
        raw = deepcopy(raw or segments.reply())
        raw['relevant_evidence_ids'] = [row['citation_id']] if chosen else []
        events = []
        def provider(*args, **kwargs):
            events.append(kwargs['purpose'])
            if mutate_during_provider:
                row['chunk_full'] = 'Packaging dimensions and paint colour only.'
            return raw, 'offline-router'
        with patch.object(self.m, '_v13_json_models', provider), patch.object(
                self.m, '_v13_gate_candidate_block', return_value=('Source excerpt.', [row])):
            out = self.m._assistant_core_router_call(request, {'candidates': [row]})
        self.assertEqual(events, ['assistant_core_v2_semantic_router'])
        return request, normalize_decision(request, out), row

    def select(self, row, decision, extra=(), limit=8):
        return self.m._assistant_core_root_source_selection([*extra, row], limit=limit, decision=decision)

    def test_real_router_selected_context_restores_core_without_inventing_observation_match(self):
        _, d, row = self.route()
        proof = cs.support_match(d, BODY, source_id=row['citation_id'])
        self.assertEqual(proof['matched_support_part_ids'], ())
        self.assertTrue(proof['router_context_selected'])
        before = deepcopy(row)
        selected = self.select(row, d)
        self.assertEqual([r['citation_id'] for r in selected.candidates], [row['citation_id']])
        self.assertEqual(row, before)
        audit = selected.summary['core_context_assessments'][0]
        self.assertTrue(audit['eligible']); self.assertTrue(audit['router_context_selected'])
        self.assertEqual(audit['matched_support_part_ids'], [])
        self.assertEqual(audit['source_text_sha256'], hashlib.sha256(BODY.encode()).hexdigest())

    def test_same_ID_changed_during_or_after_router_cannot_self_legitimize_new_body(self):
        for during in (False, True):
            _, d, row = self.route(mutate_during_provider=during)
            if not during:
                row['chunk_full'] = 'Packaging dimensions and paint colour only.'
            proof = cs.support_match(d, row['chunk_full'], source_id=row['citation_id'])
            with self.subTest(during=during):
                self.assertFalse(proof['router_context_selected'])
                self.assertEqual(self.select(row, d).candidates, ())

    def test_unselected_unsupplied_duplicate_body_and_candidate_flags_have_no_lane(self):
        _, unselected, row = self.route(chosen=False)
        row.update(router_context_selected=True, router_context_source_id=row['citation_id'],
                   core_context_eligible=True, matched_support_part_ids=['0:10:observation'])
        self.assertEqual(self.select(row, unselected).candidates, ())
        _, selected, _ = self.route()
        other = {**row, 'citation_id': 'other-source:p1'}
        self.assertFalse(cs.support_match(selected, BODY, source_id=other['citation_id'])['router_context_selected'])
        ambiguous = cs.snapshot_router_sources([(row['citation_id'], BODY), (row['citation_id'], 'Different body')])
        self.assertEqual(ambiguous, ())

    def test_scope_carrier_or_mode_change_cannot_enable_context_in_ASK_or_Smart(self):
        request, d, row = self.route()
        for mode in ('ask', 'smart_diagnostic'):
            self.assertEqual(cs.support_match(replace(d, effective_mode=mode), BODY, source_id=row['citation_id']), {})
        for req in (self.request(company_id='foreign-company'), self.request(machine_id='foreign-machine'),
                    self.request(query=segments.QUERY + ' Another observation.')):
            raw = segments.reply(); raw[cs.WIRE_KEY] = d.root_clue_support
            result = normalize_decision(req, raw)
            self.assertIsNone(result.root_clue_support)
        changed = replace(d.root_clue_support, router_source_hashes=((row['citation_id'], '0' * 64),))
        self.assertEqual(cs.support_match(replace(d, root_clue_support=changed), BODY, source_id=row['citation_id']), {})

    def test_all_existing_absolute_capacity_and_exclusion_guards_remain(self):
        _, d, row = self.route()
        high = candidate('ps:high', source='ps-high', source_type='ps', semantic_similarity=1,
            causal_strength_score=.4, subsystem_score=.4, context_fit_score=.3,
            assistant_core_facet_coverage=1, assistant_core_diagnostic_priority={
                'score':1,'base_score':.8,'exact_case_bonus':.2,
                'groups':{'discriminants':1,'observables':1,'subsystems':1}})
        result = self.select(row, d, extra=[high])
        self.assertIn(row['citation_id'], [x['citation_id'] for x in result.candidates])
        self.assertGreater(result.summary['novel_quality_floor'], .35)
        for changes in ({'hard_excluded':True}, {'assistant_core_root_viable':False},
                        {'role_group':'support'}, {'causal_strength_score':0}, {'subsystem_score':0},
                        {'assistant_core_facet_coverage':0, 'assistant_core_diagnostic_priority':{},
                         'causal_strength_score':.08, 'subsystem_score':.01}):
            with self.subTest(changes=changes):
                self.assertEqual(self.select({**row, **changes}, d).candidates, ())
        self.assertEqual(self.select(row, d, limit=0).candidates, ())
        self.assertLessEqual(len(self.select(row, d, extra=[high], limit=1).candidates), 1)

    def test_no_extra_priority_fact_or_unknown_observation_and_existing_UI_rejection(self):
        _, d, row = self.route()
        before = cs.priority_metrics(d, BODY, source_type='document', substantive_ps=False, metrics=cs._coverage)
        profile = replace(segments.dq.analyze_diagnostic_query(segments.QUERY, response_language='en'),
                          parts=d.root_clue_support.parts, interpretation_status='valid', diagnostic_projection=True)
        legacy = replace(d, root_clue_support=cs.from_profile(profile,
            company_id='fixture-company', machine_id='fixture-machine'))
        self.assertEqual(before, cs.priority_metrics(legacy, BODY, source_type='document',
            substantive_ps=False, metrics=cs._coverage))
        cleared = replace(d, root_clue_support=replace(d.root_clue_support, router_source_hashes=()))
        # Removing the carrier invalidates its context binding and cannot create fact coverage.
        proof = cs.support_match(d, BODY, source_id=row['citation_id'])
        self.assertEqual(proof['matched_support_part_ids'], ())
        self.assertFalse(cs.support_match(cleared, BODY, source_id=row['citation_id']))
        self.assertNotIn(segments.SEGMENTS[2][1], before['covered'])
        from machinemind.retrieval import root_response_quality as quality
        rejected = quality.rejected_response(response={}, meta={}, language='en', result_code='TECHNICAL_ERROR')
        self.assertEqual(rejected['status'], 'error'); self.assertEqual(rejected['result_code'], 'TECHNICAL_ERROR')
        self.assertEqual(rejected['possible_causes'], []); self.assertEqual(rejected['citations'], [])
        self.assertFalse(rejected['meta']['root_response_quality']['validated_diagnosis'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
