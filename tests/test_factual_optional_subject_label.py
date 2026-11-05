"""Optional labels can be edited by the real review path without losing facts.

SQL evidence and provider replies are fixtures. These tests exercise owned ASK
admission, the independent edit decoder, grounding and final digest; they do not
prove that a live reviewer will follow the new instruction.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_factual_context_owned as owned
from context_review_wire_fixture import original_context_sources
from machinemind.ask.response_evidence import ResponseEvidenceError

FACT = 'Replace the cooling water every 600 hours, with the documented isolation conditions.'
LABELLED = 'For the chiller ABX-210: ' + FACT


class OptionalSubjectLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.FactualContextOwnedTests.setUpClass()
        cls.m = owned.FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.FactualContextOwnedTests.tearDownClass()

    case = owned.FactualContextOwnedTests.case
    assert_bound = owned.FactualContextOwnedTests.assert_bound

    def prepare(self, *, initial=LABELLED, source=FACT, reviewed=FACT, reject=False):
        def second(reply):
            if reject:
                reply.update(outcome='no_sources', edits=[])
                reply['checks']['scope_applicability'] = False
            elif reviewed is not None:
                reply.update(edits=[{'target':'item:1', 'text':reviewed}])
        with patch.object(owned, 'FIRST_ITEM', initial), patch.object(owned, 'SAFE_ITEM', source):
            owner, fixture = self.case(second=second, debug=True)
        original = fixture.owner._response_flow.execution._v13_json_models
        def provider(messages, **kwargs):
            with patch.object(owned, 'FIRST_ITEM', initial):
                return original(messages, **kwargs)
        fixture.owner._response_flow.execution = replace(
            fixture.owner._response_flow.execution, _v13_json_models=provider)
        return owner, fixture

    def assert_rejected_by_grounding(self, fixture):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ResponseEvidenceError): fixture.run()
        contract = fixture.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertFalse(contract['semantic_answer_bound'])
        self.assertEqual(contract['reason'], 'semantic_answer_changed_after_review')
        self.assertFalse(fixture.owner._response_flow.completed)
        self.assertEqual(len(fixture.post_calls), 2)

    def test_unsupported_optional_label_reproduces_binding_failure_without_review_edit(self):
        owner, fixture = self.prepare(reviewed=None)
        self.assert_rejected_by_grounding(fixture)
        self.assertEqual(owner.contract_checks[-1]['initial_removed_claims_count'], 1)
        hints = fixture.calls[1]['messages'][1]['content'].split('\n\nGROUNDING_CHECKS:\n')[1]
        self.assertEqual(json.loads(hints)[0]['claims'], [{'kind':'code','raw':'ABX-210','unit':'code'}])

    def test_independent_label_edit_keeps_supported_interval_conditions_sources_and_digest(self):
        owner, fixture = self.prepare()
        budget = (owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd,
                  owner.budget.max_llm_calls)
        with redirect_stdout(io.StringIO()): result = fixture.run()
        contract = self.assert_bound(result, fixture)
        self.assertIn(FACT, result['answer'])
        self.assertNotIn('ABX-210', result['answer'])
        self.assertIn('ABX-210', fixture.first_candidate)
        self.assertEqual(owner.contract_checks[-1]['initial_removed_claims_count'], 0)
        self.assertEqual(contract['semantic_verifier']['independent_context_review']['edit_count'], 1)
        self.assertEqual(len(fixture.post_calls), 2)
        first = owned.source_packet(fixture.calls[0]['messages'][1]['content'], '\n\nCURRENT_ANSWER:')
        second = owned.source_packet(fixture.calls[1]['messages'][1]['content'], '\n\nCOMPLETE_CANDIDATE:')
        self.assertEqual(first, original_context_sources(second, first))
        self.assertNotIn('ABX-210', first)
        self.assertEqual((owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd,
                          owner.budget.max_llm_calls), budget)
        # Prompt contract accompanies the executable tests; not a semantic PASS.
        prompt = fixture.calls[1]['messages'][0]['content']
        self.assertIn('QUESTION alone is not evidence', prompt)
        self.assertIn('is not optional', prompt)
        self.assertIn('Never generalize a variant-specific instruction', prompt)

    def test_optional_label_edit_cannot_introduce_a_wrong_quantity(self):
        _, fixture = self.prepare(reviewed=FACT.replace('600', '900'))
        self.assert_rejected_by_grounding(fixture)

    def test_optional_label_edit_cannot_introduce_a_wrong_unit(self):
        _, fixture = self.prepare(reviewed=FACT.replace('600 hours', '600 bar'))
        self.assert_rejected_by_grounding(fixture)

    def test_source_supported_required_code_stays_exact(self):
        source = 'Use only grade QTR-7. ' + FACT
        initial = 'For the chiller ABX-210: ' + source
        _, fixture = self.prepare(initial=initial, source=source, reviewed=source)
        with redirect_stdout(io.StringIO()): result = fixture.run()
        self.assert_bound(result, fixture)
        self.assertIn('QTR-7', result['answer'])
        self.assertIn(FACT, result['answer'])
        self.assertNotIn('ABX-210', result['answer'])

    def test_required_code_substitution_still_fails(self):
        source = 'Use only grade QTR-7. ' + FACT
        _, fixture = self.prepare(initial=source, source=source, reviewed=source.replace('QTR-7','QRT-7'))
        self.assert_rejected_by_grounding(fixture)

    def test_independent_applicability_rejection_is_not_overridden_by_optional_label_rule(self):
        _, fixture = self.prepare(reject=True)
        with redirect_stdout(io.StringIO()): result = fixture.run()
        self.assertEqual(result['status'], 'no_sources')
        self.assertEqual(result.get('citations') or [], [])
        self.assertNotIn(FACT, result['answer'])
        self.assertEqual(len(fixture.post_calls), 2)


if __name__ == '__main__': unittest.main(verbosity=2)
