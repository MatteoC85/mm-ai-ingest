"""Successful operation labels do not outrank reviewed explicit changes.

The owned path uses real admission, source decoder, transport budget, independent
review and final validation. Provider replies are synthetic, not incident wire.
"""
from copy import deepcopy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_factual_context_owned as owned
import test_factual_context_review as plain
from machinemind.ask import factual_context_review as review, request_completion
from machinemind.ask.response_evidence import ResponseEvidenceError


class FactualReviewOperationLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.FactualContextOwnedTests.setUpClass()
        cls.m = owned.FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.FactualContextOwnedTests.tearDownClass()

    def run_owned(self, change):
        owner = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=False), started=time.monotonic())
        f = owned.ContextFixture(self, second=change)
        limits = (owner.budget.max_llm_calls, owner.budget.max_estimated_cost_usd, owner.budget.deadline_monotonic)
        try:
            result = f.run()
            self.assertEqual(len(f.post_calls),2)
            self.assertEqual(len(f.rendered),1)
            self.assertEqual(limits,(owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd,owner.budget.deadline_monotonic))
            self.assertEqual({x['budget_id'] for x in f.calls},{id(owner.budget)})
            return result, deepcopy(f.validated['meta']['assistant_core_validation']['answer_contract'])
        finally:
            f.close(); owner.close()

    def test_pass_or_rewrite_applies_same_explicit_safety_edit_and_table_context(self):
        answers = []
        for label in ('pass','rewrite'):
            with self.subTest(label=label):
                result, contract = self.run_owned(lambda r:r.update(outcome=label,
                    edits=[{'target':'item:1','text':owned.SAFE_ITEM}]))
                self.assertEqual(result['status'],'answered')
                self.assertTrue(contract['passed'])
                self.assertTrue(contract['semantic_answer_bound'])
                self.assertEqual(contract['semantic_verifier']['independent_context_review']['outcome'],'rewrite')
                self.assertIn('personal protective equipment',result['answer'])
                self.assertIn('halve them under heavy loading',result['answer_html'])
                answers.append((result['answer'],result['citations']))
        self.assertEqual(answers[0],answers[1])

    def test_rewrite_with_no_explicit_change_retains_the_exact_reviewed_answer(self):
        f=plain.FactualContextReviewTests()
        for template in (False,True):
            with self.subTest(template=template):
                _, first, frame, _ = f.candidate(table=False,template=template)
                out=review.apply_review(first,f.verdict(table=False,outcome='rewrite'),frame)
                self.assertEqual(out['answer'],first['answer'])
                self.assertEqual(out['expected_list_items'],first['expected_list_items'])
                self.assertEqual(out['independent_context_review']['outcome'],'pass')
                self.assertEqual(out['independent_context_review']['answer_sha256'],review._digest(first['answer']))

    def test_explicit_extra_admitted_support_needs_no_fake_text_edit(self):
        f=plain.FactualContextReviewTests();_,first,frame,_=f.candidate(table=False)
        out=review.apply_review(first,f.verdict(table=False,citations=['manual:p1','manual:p2']),frame)
        self.assertEqual(out['answer'],first['answer'])
        self.assertEqual(out['citation_ids'],['manual:p1','manual:p2'])
        self.assertEqual(out['independent_context_review']['outcome'],'rewrite')

    def test_no_sources_and_each_failed_semantic_check_remain_terminal(self):
        f=plain.FactualContextReviewTests();_,first,frame,_=f.candidate(table=False)
        for mutation in [lambda r:r.update(outcome='no_sources')]+[
                lambda r,name=name:r['checks'].update({name:False}) for name in review.CHECKS]:
            for label in ('pass','rewrite'):
                with self.subTest(label=label,mutation=mutation):
                    verdict=f.verdict(table=False,outcome=label,
                        edits=[{'target':'item:0','text':'Inspect the level; top up only if necessary.'}])
                    mutation(verdict)
                    out=review.apply_review(first,verdict,frame)
                    self.assertEqual(out['outcome'],'no_sources')
                    self.assertEqual(out['answer'],'')
                    self.assertTrue(out['_terminal_review_failure'])

    def test_success_labels_cannot_bypass_immutable_target_or_source_identity(self):
        for change in (
            lambda r:r.update(outcome='pass',edits=[{'target':'item:0','text':'Invented daily interval.'}]),
            lambda r:r.update(outcome='pass',citation_ids=['foreign:p1']),
            lambda r:r['table_contexts'][0]['source_quotes'][0].update(quote='Unsupported prerequisite.'),
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ResponseEvidenceError,'factual_context_review_invalid'):
                    self.run_owned(change)

    def test_success_labels_cannot_bypass_exact_frame_and_template_bindings(self):
        f=plain.FactualContextReviewTests();_,first,frame,_=f.candidate(table=False)
        for edits in ([{'target':'item:99','text':'New fact.'}],
                      [{'target':'item:0','text':'A'},{'target':'item:0','text':'B'}],
                      [{'target':'template-answer','text':'Inventory silently omitted.'}]):
            verdict=f.verdict(table=False,outcome='pass',edits=edits)
            out=review.apply_review(first,verdict,frame)
            self.assertEqual(out['outcome'],'no_sources')
        changed={**first,'answer':first['answer']+' Changed after review.'}
        out=review.apply_review(changed,f.verdict(table=False),frame)
        self.assertEqual(out['independent_context_review']['gate'],'frame_mismatch')


if __name__ == '__main__': unittest.main()
