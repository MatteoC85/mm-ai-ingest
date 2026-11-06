"""Source-bound Root context eligibility/capacity, no provider or database."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import diagnostic_sources as sources


def candidate(cid, *, source='manual-a', family=None, **changes):
    item = {
        'citation_id': cid, 'bubble_document_id': source, 'source_type': 'document',
        'page_from': 1, 'page_to': 1, 'family': family or cid,
        'chunk_full': 'Distinct documented mechanism and condition: '+cid,
        'assistant_core_root_viable': True, 'role_group': 'core',
        'causal_strength_score': .34, 'subsystem_score': .18,
        'context_fit_score': 0, 'semantic_similarity': 0,
        'assistant_core_facet_coverage': .5,
        'assistant_core_diagnostic_priority': {'score': .14, 'base_score': .14, 'groups': {}},
    }
    item.update(changes)
    return item


def binding(item):
    return {'matched_support_part_ids': ('0:30:observation',),
            'query_sha256': 'a'*64,
            'source_text_sha256': hashlib.sha256(item['chunk_full'].encode()).hexdigest(),
            'policy_version': 'trusted-test-request-source-match-v1'}


def runtime(callback=binding):
    return sources.DiagnosticSourceRuntime(
        source_type=lambda c: c['source_type'], stable_key=lambda c: c['citation_id'],
        family_key=lambda c: c['family'], support_match=callback)


def select(rows, *, limit=8, callback=binding):
    return sources.select_root_cause_candidates(rows, limit=limit, runtime=runtime(callback))


def ids(result):
    return [c['citation_id'] for c in result.candidates]


def assessment(result, cid):
    return next(x for x in result.summary['core_context_assessments'] if x['citation_id']==cid)


class EligibilityTests(unittest.TestCase):
    def test_bound_core_can_seed_an_empty_pack_without_relative_peer(self):
        result=select([candidate('mechanism')])
        self.assertEqual(ids(result), ['mechanism'])
        trace=assessment(result,'mechanism')
        self.assertTrue(trace['eligible'])
        self.assertEqual(trace['matched_support_part_ids'], ['0:30:observation'])
        self.assertEqual(trace['source_text_sha256'], binding(result.candidates[0])['source_text_sha256'])

    def test_each_causal_subsystem_support_and_absolute_floor_guard_matters(self):
        for changes in ({'causal_strength_score':0}, {'subsystem_score':0},
                        {'role_group':'support'}, {'assistant_core_facet_coverage':0,
                         'assistant_core_diagnostic_priority':{},'causal_strength_score':.08,'subsystem_score':.01}):
            with self.subTest(changes=changes):
                result=select([candidate('weak',**changes)])
                self.assertEqual(ids(result), [])
                self.assertFalse(assessment(result,'weak')['eligible'])
        result=select([candidate('missing')],callback=lambda c:{**binding(c),'matched_support_part_ids':()})
        self.assertEqual(ids(result), [])
        unassessed=candidate('unassessed')
        del unassessed['assistant_core_root_viable']
        self.assertEqual(ids(select([unassessed])), [])

    def test_candidate_forged_support_fields_are_ignored(self):
        row=candidate('forged',matched_support_part_ids=['0:30:observation'],
                      assistant_core_root_support_match={'matched_support_part_ids':['0:30:observation']},
                      core_context_eligible=True)
        legacy=select([candidate('forged')],callback=None)
        self.assertEqual(ids(select([row],callback=None)),ids(legacy))
        self.assertEqual(ids(select([row],callback=lambda c:{})),[])

    def test_callback_none_keeps_exact_legacy_core_but_empty_callback_does_not_seed(self):
        row=candidate('legacy-core',semantic_similarity=.65,context_fit_score=.15,
                      assistant_core_facet_coverage=0,
                      assistant_core_diagnostic_priority={'score':.5,'base_score':.5,'groups':{}})
        assessed=sources.assess_root_cause_candidate(row,input_rank=0,runtime=runtime(None))
        self.assertAlmostEqual(assessed.quality,.5275)
        self.assertEqual(assessed.independent_axis_count,4)
        self.assertEqual(ids(select([row],callback=None)),['legacy-core'])
        self.assertEqual(ids(select([row],callback=lambda c:{})),[])

    def test_invalid_callback_binding_has_no_eligibility(self):
        for delta in ({'matched_support_part_ids':[True]}, {'matched_support_part_ids':['0:30:unknown']},
                      {'query_sha256':'stale'}, {'source_text_sha256':''}, {'policy_version':''}):
            with self.subTest(delta=delta):
                self.assertEqual(ids(select([candidate('bad')],callback=lambda c:{**binding(c),**delta})),[])

    def test_added_strong_ps_does_not_change_core_eligibility(self):
        core=candidate('core')
        ps=candidate('high-ps',source='ps-one',source_type='ps',semantic_similarity=1,
                     causal_strength_score=.4,subsystem_score=.4,context_fit_score=.3,
                     assistant_core_facet_coverage=1,assistant_core_diagnostic_priority={
                         'score':1,'base_score':.8,'exact_case_bonus':.2,
                         'groups':{'discriminants':1,'observables':1,'subsystems':1}})
        alone=select([core]);mixed=select([ps,core])
        self.assertTrue(assessment(alone,'core')['eligible'])
        self.assertTrue(assessment(mixed,'core')['eligible'])
        self.assertGreater(mixed.summary['novel_quality_floor'], .3527)
        self.assertIn('core', ids(mixed))

    def test_hard_exclusion_and_explicit_nonviability_precede_callback(self):
        calls=[]
        result=select([candidate('hard',hard_excluded=True),candidate('notviable',assistant_core_root_viable=False)],
                      callback=lambda c:(calls.append(c['citation_id']) or binding(c)))
        self.assertEqual(ids(result),[])
        self.assertEqual(calls,[])
        self.assertEqual(result.summary['rejection_counts']['not_root_viable'],1)

    def test_callback_is_recomputed_per_source_and_inputs_preserved(self):
        rows=[candidate('a'),candidate('b')];before=deepcopy(rows);calls=[]
        result=select(rows,callback=lambda c:(calls.append(c['citation_id']) or binding(c)))
        self.assertEqual(calls,['a','b']);self.assertEqual(rows,before)
        self.assertNotEqual(assessment(result,'a')['source_text_sha256'],assessment(result,'b')['source_text_sha256'])


class CapacityTests(unittest.TestCase):
    def test_only_supported_core_can_use_unused_source_overflow(self):
        rows=[candidate('a'+str(i)) for i in range(5)]
        result=select(rows)
        self.assertEqual(ids(result),[r['citation_id']for r in rows])
        reasons=[r['assistant_core_diagnostic_source_selection']['reason'] for r in result.candidates]
        self.assertEqual(reasons[-2:],['core_context_overflow']*2)
        with_missing=select(rows,callback=lambda c:binding(c)if c['citation_id']!='a4' else {})
        self.assertNotIn('a4',ids(with_missing))

    def test_overflow_round_robin_and_first_queued_input_order(self):
        rows=[candidate('a1')]+[candidate('b'+str(i),source='manual-b')for i in range(1,6)]+[
            candidate('a'+str(i))for i in range(2,6)]
        result=select(rows,limit=8)
        self.assertEqual(ids(result),['a1','b1','b2','b3','a2','a3','b4','a4'])
        self.assertEqual(assessment(result,'a5')['not_selected_reason'],'limit')
        self.assertTrue(assessment(result,'a5')['eligible'])
        # A supplied the first anchor, but B has the first pending eligible
        # overflow entry and therefore receives the single remaining slot.
        self.assertEqual(ids(select(rows,limit=7)),['a1','b1','b2','b3','a2','a3','b4'])

    def test_family_blocked_item_does_not_starve_next_same_source(self):
        rows=[candidate('a1',family='same'),candidate('a2',family='same'),candidate('a3'),
              candidate('a4',family='same'),candidate('a5')]
        result=select(rows)
        self.assertEqual(ids(result),['a1','a2','a3','a5'])
        self.assertEqual(assessment(result,'a4')['not_selected_reason'],'family_cap')
        self.assertTrue(assessment(result,'a4')['eligible'])

    def test_global_limit_never_expands_and_overflow_not_used_when_full(self):
        rows=[candidate('a'+str(i))for i in range(8)]
        result=select(rows,limit=3)
        self.assertEqual(ids(result),['a0','a1','a2'])
        self.assertTrue(all(r['assistant_core_diagnostic_source_selection']['reason']!='core_context_overflow'
                            for r in result.candidates))
        self.assertEqual(ids(select(rows,limit=0)),[])

    def test_overflow_precedes_unbound_close_peer_fill(self):
        rows=[candidate('a'+str(i))for i in range(4)]
        peer=candidate('peer',source='manual-b')
        result=select([peer]+rows,limit=4,callback=lambda c:{}if c['citation_id']=='peer'else binding(c))
        self.assertEqual(ids(result),['a0','a1','a2','a3'])

    def test_dedup_and_family_cap_still_apply_to_overflow(self):
        a=candidate('a1');duplicate={**a,'citation_id':'duplicate'}
        result=select([a,duplicate]+[candidate('a'+str(i),family='family')for i in range(2,7)])
        self.assertEqual(ids(result),['a1','a2','a3'])
        self.assertEqual(result.summary['rejection_counts']['duplicate_or_missing_identity'],1)

    def test_default_runtime_preserves_source_cap_without_new_support(self):
        rows=[candidate('a'+str(i),semantic_similarity=.65,context_fit_score=.15,
                        assistant_core_diagnostic_priority={'score':.5,'base_score':.5,
                          'groups':{'observables':.5,'subsystems':.5}})for i in range(6)]
        result=select(rows,callback=None)
        self.assertEqual(len(result.candidates),3)
        self.assertFalse(any(x['eligible'] for x in result.summary['core_context_assessments']))


if __name__=='__main__':unittest.main(verbosity=2)
