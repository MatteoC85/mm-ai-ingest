"""Owned grounding hints are advisory; final proof and authority remain real.

SQL observations and both model replies are fixtures. No semantic model quality
or identity of the rejected live T48 claim is inferred by these tests.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_factual_context_owned as owned
from context_review_wire_fixture import original_context_sources
from machinemind.ask import factual_context_review as review, request_completion
from machinemind.ask.response_evidence import ResponseEvidenceError

SUPPORTED = 'The documented grade is QTR-7; the interval is 600 hours.'
UNSUPPORTED = 'The documented grade is QRT-7; the interval is 600 hours.'


class FactualGroundingPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.FactualContextOwnedTests.setUpClass()
        cls.m = owned.FactualContextOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        owned.FactualContextOwnedTests.tearDownClass()

    case = owned.FactualContextOwnedTests.case
    assert_bound = owned.FactualContextOwnedTests.assert_bound

    def prepare(self, *, initial=UNSUPPORTED, corrected=True, debug=True):
        second = (lambda r:r.update(edits=[{'target':'item:1','text':SUPPORTED}])) if corrected else None
        # Literal evidence is registered before any source handle is acquired.
        with patch.object(owned, 'FIRST_ITEM', initial), patch.object(owned, 'SAFE_ITEM', SUPPORTED):
            owner, fixture = self.case(second=second, debug=debug)
        # The fixture's provider reads FIRST_ITEM at dispatch, hence keep that
        # value in its first-reply hook instead of changing admitted sources.
        original = fixture.owner._response_flow.execution._v13_json_models
        def provider(messages, **kwargs):
            with patch.object(owned, 'FIRST_ITEM', initial):
                return original(messages, **kwargs)
        from dataclasses import replace
        fixture.owner._response_flow.execution = replace(fixture.owner._response_flow.execution, _v13_json_models=provider)
        return owner, fixture

    def hints(self, fixture):
        message = fixture.calls[1]['messages'][1]['content']
        self.assertIn('\n\nGROUNDING_CHECKS:\n', message)
        return json.loads(message.split('\n\nGROUNDING_CHECKS:\n',1)[1])

    def test_same_owned_sources_reach_reviewer_with_targeted_hint_and_correct_literal_passes(self):
        owner, f = self.prepare()
        limits = (owner.budget.deadline_monotonic, owner.budget.max_estimated_cost_usd, owner.budget.max_llm_calls)
        with patch('sys.stdout', new=io.StringIO()): result = f.run()
        contract = self.assert_bound(result, f)
        self.assertEqual(self.hints(f), [{'target':'item:1','unsupported_count':1,
            'claims':[{'kind':'code','raw':'QRT-7','unit':'code'}], 'omitted_details':0}])
        self.assertIn(UNSUPPORTED, f.first_candidate)
        self.assertIn('QTR-7', result['answer']); self.assertNotIn('QRT-7', result['answer'])
        first = owned.source_packet(f.calls[0]['messages'][1]['content'], '\n\nCURRENT_ANSWER:')
        second = owned.source_packet(f.calls[1]['messages'][1]['content'], '\n\nCOMPLETE_CANDIDATE:')
        self.assertEqual(first, original_context_sources(second, first))
        self.assertIn(SUPPORTED, first)
        self.assertEqual(len(f.post_calls), 2)
        self.assertEqual({c['budget_id'] for c in f.calls}, {id(owner.budget)})
        self.assertEqual((owner.budget.deadline_monotonic,owner.budget.max_estimated_cost_usd,owner.budget.max_llm_calls), limits)
        self.assertEqual(owner.contract_checks[-1]['initial_removed_claims_count'], 0)
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertNotIn('GROUNDING_CHECKS', json.dumps(result))

    def test_ignoring_hint_still_rejects_and_initial_diagnostic_survives_budget_terminal(self):
        owner, f = self.prepare(corrected=False)
        stream = io.StringIO()
        with patch('sys.stdout', new=stream):
            with self.assertRaisesRegex(ResponseEvidenceError, 'final rejected draft lacks observed no-evidence builder'):
                f.run()
        contract = f.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['semantic_answer_bound'])
        self.assertEqual(contract['reason'], 'semantic_answer_changed_after_review')
        row = owner.contract_checks[-1]
        self.assertIn('initial_removed_claims_count', row)
        self.assertEqual(row['initial_removed_claims_count'], 1)
        self.assertEqual(self.hints(f)[0]['claims'][0]['raw'], 'QRT-7')
        self.assertNotEqual(row['initial_redacted_sha256'], row['initial_grounded_sha256'])
        self.assertEqual(row['initial_grounded_sha256'], row['initial_media_sha256'])
        self.assertEqual(row['post_review_removed_claims_count'], 0)
        terminal = owner.finish({'ok':False,'status':'error','result_code':'BUDGET_EXCEEDED','meta':{}})
        self.assertEqual(terminal['meta']['request_completion']['contract_checks'][-1], row)
        for secret in ('QRT-7', 'QTR-7', 'documented grade', '600 hours'):
            self.assertNotIn(secret, json.dumps(terminal)); self.assertNotIn(secret, stream.getvalue())
        self.assertIn('ASK_INITIAL_GROUNDING', stream.getvalue())
        self.assertFalse(f.owner._response_flow.completed)

    def test_no_preflight_issue_does_not_approve_new_bad_second_edit(self):
        owner, f = self.prepare(initial=SUPPORTED, corrected=False)
        # The independent reply can itself introduce an unsupported spelling.
        original = f.owner._response_flow.execution._v13_json_models
        def provider(messages, **kwargs):
            if kwargs['json_schema']['name'] != 'factual_context_review':
                return original(messages, **kwargs)
            result, model = original(messages, **kwargs)
            result = deepcopy(result)
            result['edits'] = [{'target':'item:1','text':UNSUPPORTED}]
            return result, model
        from dataclasses import replace
        f.owner._response_flow.execution = replace(f.owner._response_flow.execution, _v13_json_models=provider)
        with patch('sys.stdout', new=io.StringIO()):
            with self.assertRaises(ResponseEvidenceError): f.run()
        self.assertNotIn('\n\nGROUNDING_CHECKS:', f.calls[1]['messages'][1]['content'])
        self.assertFalse(f.validated['meta']['assistant_core_validation']['answer_contract']['semantic_answer_bound'])
        self.assertEqual(owner.contract_checks[-1]['initial_removed_claims_count'], 1)

    def test_bounded_details_keep_every_affected_target_and_original_text(self):
        items = [' '.join('QRT-'+str(i*4+j+1) for j in range(4)) for i in range(8)]
        findings = review._grounding_findings(items, '[[FACTUAL_INVENTORY]]',
            tuple('item:'+str(i) for i in range(8)), 'QTR-7', lambda x:x)
        self.assertEqual([row[0] for row in findings], ['item:'+str(i) for i in range(8)])
        self.assertEqual([row[1] for row in findings], [4]*8)
        self.assertEqual(sum(len(row[2]) for row in findings), 12)
        self.assertTrue(all(len(row[2]) <= 3 for row in findings))
        self.assertTrue(all(not row[2] for row in findings[4:]))
        self.assertEqual(items[0], 'QRT-1 QRT-2 QRT-3 QRT-4')

    def test_debug_gate_bounds_hashes_and_clears_between_request_owners(self):
        private = 'PRIVATE MODEL CLAIM OR SOURCE'
        for debug in (True, False, 'true'):
            owner, f = self.prepare(debug=debug)
            stream=io.StringIO()
            with patch('sys.stdout', new=stream):
                request_completion.record_contract_observation('final_contract',
                    {'passed':False,'reason':'semantic_answer_changed_after_review'}, diagnostics={
                        'initial_grounding': {'input':private, 'redacted':private,
                            'grounded':'', 'media':'', 'removed_claims_count':999,
                            'raw_claim':private}, 'source_text':private})
            row=owner.contract_checks[-1]
            if debug is True:
                self.assertEqual(row['initial_removed_claims_count'],100)
                self.assertEqual(row['initial_input_sha256'], hashlib.sha256(private.encode()).hexdigest())
                self.assertIn('ASK_INITIAL_GROUNDING',stream.getvalue())
            else:
                self.assertFalse(any(key.startswith('initial_') for key in row))
                self.assertEqual(stream.getvalue(),'')
            self.assertNotIn(private, json.dumps(row)+stream.getvalue())
            f.close(); owner.close()
            self.assertEqual(owner.contract_checks,[])
        owner, f = self.prepare()
        self.assertEqual(owner.contract_checks,[])


if __name__ == '__main__': unittest.main(verbosity=2)
