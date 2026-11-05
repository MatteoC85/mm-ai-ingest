"""Source-first draft uses its bounded output for JSON; final review remains independent."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_factual_context_owned as fixture


class AskFactualProducerReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FactualContextOwnedTests.setUpClass()
        cls.m=fixture.FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        fixture.FactualContextOwnedTests.tearDownClass()

    case=fixture.FactualContextOwnedTests.case
    assert_bound=fixture.FactualContextOwnedTests.assert_bound

    def test_source_first_none_then_independent_low_through_real_owned_transport(self):
        completion,f=self.case()
        limits=(completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd,
            completion.budget.max_llm_calls)
        result=f.run();self.assert_bound(result,f)
        self.assertEqual([row['payload']['reasoning']['effort'] for row in f.post_calls],['none','low'])
        self.assertEqual(len(f.post_calls),2)
        self.assertEqual([row['payload']['max_output_tokens'] for row in f.post_calls],[3200,3200])
        self.assertEqual(limits,(completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd,
            completion.budget.max_llm_calls))
        self.assertTrue(any(row['review_reserve_seconds']==25 for row in completion.provider_schedules))
        self.assertEqual(completion.provider_schedules[-1]['review_reserve_seconds'],0)

    def test_unknown_configured_model_retains_low_effort_on_both_stages(self):
        # Set the override before creating the real request owner, whose bound
        # runtime is intentionally immutable for the lifetime of the request.
        with patch.object(self.m,'V13_FAST_MODEL','unverified-override'):
            completion,f=self.case()
            result=f.run()
        self.assert_bound(result,f)
        self.assertEqual([row['payload']['reasoning']['effort'] for row in f.post_calls],['low','low'])


if __name__=='__main__':
    unittest.main(verbosity=2)
