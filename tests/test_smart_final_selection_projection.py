"""A terminal tie has no selected winner in any public Bubble-compatible view.

Fixtures are explicit offline reviewed decisions, not claims of model accuracy.
The real endpoint/signing/review replay execute. A separate frozen oracle
compares the actual pre-change display callable with this candidate.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_smart_review import SmartReviewEndpointTests, denied
from test_smart_continuity_endpoints import direct_post


class SmartFinalSelectionProjectionTests(unittest.TestCase):
    setUpClass = classmethod(SmartReviewEndpointTests.setUpClass.__func__)
    tearDownClass = classmethod(SmartReviewEndpointTests.tearDownClass.__func__)
    setup_fixture = SmartReviewEndpointTests.setUp
    start = SmartReviewEndpointTests.start
    post = direct_post

    def setUp(self):
        self.setup_fixture()

    def hypotheses(self, weights=(50,50), statuses=None):
        statuses = statuses or ('open',) * len(weights)
        return [{'id':f'H{i+1}', 'rank':i+1, 'label':f'Documented mechanism {i+1}',
                 'why':'The message type and circuit remain unknown.',
                 'status':statuses[i], 'probability_pct':weight, 'probability_band':'unknown',
                 'checks':[], 'evidence_ids':['manual:p1']}
                for i,weight in enumerate(weights)]

    def test_tie_projection_neutralizes_only_public_selection_and_preserves_claims(self):
        hs = self.hypotheses()
        final = self.m._sd_canonicalize_final_result({}, hs, 'en')
        before = deepcopy((final,hs))
        current = self.m._sd_final_result_for_display(final,hs,'en')
        self.assertEqual(final['most_likely_hypothesis_id'],'H1')
        self.assertEqual(final['probability_pct'],50)
        self.assertEqual(current['most_likely_hypothesis_id'],'')
        self.assertEqual(current['probability_pct'],0)
        self.assertEqual(current['probability_band'],'unknown')
        self.assertIn('they do not identify a leading cause',current['summary'])
        self.assertEqual(current['recommended_checks'],final['recommended_checks'])
        self.assertEqual((final,hs),before)

    def test_only_actual_positive_top_tie_changes_selection(self):
        for weights,statuses in [((70,15,15),None), ((100,),None), ((0,0),None),
                                 ((50,50),('confirmed','open')), ((40,40),('excluded','open'))]:
            with self.subTest(weights=weights,statuses=statuses):
                hs=self.hypotheses(weights,statuses)
                final=self.m._sd_canonicalize_final_result({},hs,'en')
                current=self.m._sd_final_result_for_display(final,hs,'en')
                for key in ('most_likely_hypothesis_id','most_likely_label','probability_pct','probability_band'):
                    self.assertEqual(current.get(key),final.get(key))
        for hs in (self.hypotheses((40,40,20)), self.hypotheses((49.96,50.04))):
            # Existing display convention rounds relative weights to one decimal.
            final=self.m._sd_canonicalize_final_result({},hs,'en')
            self.assertEqual(self.m._sd_final_result_for_display(final,hs,'en')['most_likely_hypothesis_id'],'')

    def test_signed_finalize_replay_flat_nested_serialized_and_sources_are_consistent(self):
        self.raw['hypotheses']=self.raw['hypotheses'][:2]
        for h in self.raw['hypotheses']:
            h.update(status='open', probability_pct=50.0, probability_band='unknown',score_raw=.5,checks=[])
        self.raw['question']['target_hypotheses']=[h['id'] for h in self.raw['hypotheses']]
        start=self.start()
        before=deepcopy(json.loads(start['session_state_json']))
        provider_before=self.provider.call_count
        with patch.object(self.m,'_v13_json_models',side_effect=denied):
            first=self.post('finalize',start['session_state_json'])
            self.assertEqual(first.status_code,200,first.text)
            body=first.json()
            again=self.post('finalize',body['session_state_json'])
        self.assertEqual(again.status_code,200,again.text)
        self.assertEqual(self.provider.call_count,provider_before)
        final=body['final_result']
        self.assertEqual(final,json.loads(body['final_result_json']))
        self.assertEqual(final,again.json()['final_result'])
        self.assertEqual(final['most_likely_hypothesis_id'],'')
        self.assertEqual(final['most_likely_label'],'No cause identified')
        self.assertEqual(body['final_most_likely_label'],final['most_likely_label'])
        self.assertEqual(body['final_probability_pct'],0)
        self.assertEqual(body['final_probability_band'],'unknown')
        self.assertEqual(body['operator_summary'],body['final_summary_text'])
        self.assertEqual(final['summary'],body['final_summary_text'])
        self.assertIn('are equal',body['final_summary_text'])
        self.assertEqual(body['hypotheses'],start['hypotheses'])
        self.assertEqual([h['probability_pct'] for h in body['hypotheses']],[0,0])
        self.assertEqual([h['relative_weight_pct'] for h in body['hypotheses']],[50,50])
        after=json.loads(body['session_state_json'])
        for key in ('hypotheses','history','grounding_review','grounding_packet'):
            self.assertEqual(after[key],before[key])
        self.m._sd_validate_state_binding(after,company_id='fixture-company',machine_id='fixture-machine',session_id='fixture-session')
        self.assertEqual(body['citations'],again.json()['citations'])
        self.assertEqual(body['rg_links'],again.json()['rg_links'])
        self.assertEqual(body['final_recommended_checks_json'],'[]')
        self.assertEqual(body['meta']['smart_review']['finalization_provider_calls'],0)


del SmartReviewEndpointTests

if __name__=='__main__': unittest.main(verbosity=2)
