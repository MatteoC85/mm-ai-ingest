"""Source-limited answers and fixed debug decision codes, with real owned gates.

Authority observations/model outputs are fixtures. Passing these tests does not
establish why the unretained T49 reviewer rejected its candidate.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_ask_source_first_inventory as fx
from machinemind.ask import request_completion as completion, factual_context_review as review
import test_factual_context_review as pure
from context_review_wire_fixture import assembled_context_candidate, original_context_sources

LIMITATION = 'The supplied excerpts do not state the capacity of another assembly; no capacity is inferred.'
PRIVATE_REASON = 'Private source text: a documented safety condition is missing.'


class IndependentDecisionDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fx.AskSourceFirstInventoryTests.setUpClass()
        cls.m = fx.AskSourceFirstInventoryTests.m

    @classmethod
    def tearDownClass(cls): fx.AskSourceFirstInventoryTests.tearDownClass()

    def owner(self, debug=True):
        owner = completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=debug),started=time.monotonic())
        self.addCleanup(owner.close)
        return owner

    def fixture(self, *, verdict='pass', debug=True):
        owner = self.owner(debug)
        f = fx.SourceFirstFixture(self)
        self.addCleanup(f.close)
        original = f.owner._response_flow.execution._v13_json_models
        def provider(messages, **kwargs):
            out, model = original(messages, **kwargs)
            out = deepcopy(out)
            if kwargs['json_schema']['name'] == 'factual_context_review':
                out.update(outcome=verdict, reason='missing_prerequisite' if verdict=='no_sources' else 'accepted')
                if verdict == 'rewrite':
                    out['edits'] = [{'target':'template-answer','text':'Available documented facts:\n[[FACTUAL_INVENTORY]]\n'+LIMITATION}]
                elif verdict == 'no_sources': out['checks']['safety'] = False
            else:
                # No source mutation: a documentary limitation is part of the
                # first model fixture, then independently reviewed as a whole.
                out['answer'] = 'Documented facts:\n[[FACTUAL_INVENTORY]]\n'+LIMITATION
            return out, model
        f.owner._response_flow.execution = replace(f.owner._response_flow.execution,_v13_json_models=provider)
        return owner, f

    def observe(self, owner, reason, *, checks=None):
        decision = {'checks': checks or {'safety':False,'completeness':True}, 'reason':reason}
        with patch('sys.stdout', new=io.StringIO()) as logs:
            completion.record_contract_observation('semantic_review',
                {'outcome':'no_sources','reason':'factual_context_review_rejected'}, diagnostics={
                    'independent_review_gate':'model_rejected','independent_review_outcome':'no_sources',
                    'independent_review_decision':decision})
        return owner.contract_checks[-1], logs.getvalue()

    def test_limited_answer_pass_and_rewrite_keep_actual_owned_calls_and_final_hash(self):
        for verdict in ('pass','rewrite'):
            with self.subTest(verdict=verdict):
                owner, f = self.fixture(verdict=verdict)
                limits=(owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd)
                with patch('sys.stdout',new=io.StringIO()):result=f.run()
                self.assertEqual(result['status'],'answered')
                self.assertIn(LIMITATION,result['answer'])
                for fact in f.items:self.assertIn(fact,result['answer'])
                contract=f.final['meta']['assistant_core_validation']['answer_contract']
                self.assertTrue(contract['semantic_answer_bound'])
                self.assertTrue(contract['passed'])
                self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
                self.assertEqual(len(f.posts),2)
                self.assertEqual(f.budget_ids,[id(owner.budget)]*2)
                self.assertEqual(f.writer_calls,[])
                self.assertEqual((owner.budget.deadline_monotonic,owner.budget.max_llm_calls,owner.budget.max_estimated_cost_usd),limits)
                self.assertEqual(owner.contract_checks[0]['independent_review_outcome'],verdict)
                f.close();owner.close()

    def test_model_rejection_remains_terminal_with_fixed_debug_reason_and_checks(self):
        owner,f=self.fixture(verdict='no_sources')
        with patch('sys.stdout',new=io.StringIO()) as logs:result=f.run()
        self.assertEqual(result['status'],'no_sources')
        self.assertEqual(len(f.posts),2)
        row=owner.contract_checks[-1]
        self.assertEqual(row['independent_review_gate'],'model_rejected')
        self.assertIn('independent_review_checks',row)
        self.assertFalse(row['independent_review_checks']['safety'])
        self.assertEqual(row['independent_review_reason_code'],'missing_prerequisite')
        self.assertFalse(any('summary' in key or 'reason_sha' in key for key in row))
        terminal=owner.finish(result)
        observed=terminal['meta']['request_completion']['contract_checks'][-1]
        self.assertEqual(observed,row)
        self.assertEqual(terminal['result_code'],result['result_code'])

    def test_debug_must_be_literal_true(self):
        for debug in (False,'true',1,None):
            owner=self.owner(debug)
            row,logs=self.observe(owner,'missing_prerequisite')
            self.assertFalse(any(key.startswith('independent_review_reason') or key in (
                'independent_review_checks','independent_review_decision_summary') for key in row))
            self.assertEqual(logs,'')
            owner.close()

    def test_only_static_codes_and_known_bool_checks_are_retained(self):
        owner=self.owner()
        row,logs=self.observe(owner,'incorrect_scope',checks={'safety':False,'language':True,'unknown':'PRIVATE','completeness':1})
        self.assertEqual(row['independent_review_checks'],{'safety':False,'language':True})
        self.assertEqual(row['independent_review_reason_code'],'incorrect_scope')
        self.assertNotIn('PRIVATE',json.dumps(row)+logs)
        event=json.loads(logs.split('ASK_INDEPENDENT_REVIEW ',1)[1])
        self.assertEqual(event['failed_checks'],['safety'])
        self.assertEqual(event['independent_review_reason_code'],'incorrect_scope')
        for code in review.DECISION_REASONS:
            self.assertEqual(completion._decision_observation({'independent_review_decision':{'reason':code}})['independent_review_reason_code'],code)

    def test_arbitrary_text_never_survives_even_as_hash_in_terminal_metadata_or_logs(self):
        # Generated, nonfunctional fixtures; no credential-shaped literal is
        # stored in repository source or bypasses the release secret scanner.
        values=[PRIVATE_REASON,'password'+': harmless-test-value','API_'+'KEY = demo','Bear'+'er demo-token',
            'sk'+'-'+'syntheticfixture'*3,'https://example.invalid/private','www.example.invalid',
            '-----BEGIN'+' PRIVATE'+' KEY-----','ey'+'J0eXAiOiJKV1Q.'+'ey'+'JzdWIiOiJ0ZXN0In0.fakeSignature',
            'aB3dE7gH9jK2mN4p'+'R6sT8vW0xY5zC1qF',
            'ｐａｓｓｗｏｒｄ： demo','pass\u200bword: demo','ｈｔｔｐｓ：／／example.invalid']
        for value in values:
            with self.subTest(value=value):
                owner=self.owner();row,logs=self.observe(owner,value)
                self.assertEqual(row['independent_review_reason_code'],'other')
                self.assertFalse(any('summary' in key or 'reason_sha' in key for key in row))
                self.assertNotIn(value,json.dumps(row)+logs)
                self.assertEqual(row['independent_review_checks'],{'safety':False,'completeness':True})
                terminal=owner.finish({'ok':False,'status':'error','result_code':'TECHNICAL_ERROR','meta':{}})
                self.assertNotIn(value,json.dumps(terminal))
                self.assertEqual(terminal['meta']['request_completion']['contract_checks'][-1],row)
                owner.close()

    def test_unknown_values_and_request_cleanup_do_not_change_verdict(self):
        for value in ('x'*501,{},''):
            owner=self.owner();row,_=self.observe(owner,value)
            self.assertEqual(row['independent_review_reason_code'],'other')
            self.assertNotIn('independent_review_decision_summary',row)
            self.assertEqual(row['outcome'],'no_sources')
            owner.close();self.assertEqual(owner.contract_checks,[])
        next_owner=self.owner()
        self.assertEqual(next_owner.contract_checks,[])

    def test_reason_enum_counts_toward_atomic_schema_limit_without_source_pruning(self):
        helper=pure.FactualContextReviewTests()
        _,first,frame,_=helper.candidate()
        overhead=3+len(frame.editable_targets)+len(review._context_targets(frame))+len(frame.admitted_citation_ids)+len(review.DECISION_REASONS)
        capacity=1000-overhead
        units=tuple(('Q'+str(i),'Literal unit '+str(i),(frame.admitted_citation_ids[0],))for i in range(1,capacity+2))
        exact=replace(frame,quote_units=units[:-1])
        overflow=replace(frame,quote_units=units)
        self.assertEqual(len(review._atomic_references(exact)),capacity)
        self.assertIsNone(review._atomic_references(overflow))
        def enum_count(node):
            if type(node)is dict:return len(node.get('enum',[]))+sum(enum_count(v)for v in node.values())
            if type(node)is list:return sum(enum_count(v)for v in node)
            return 0
        self.assertEqual(enum_count(review.schema(exact)),1000)
        self.assertLessEqual(enum_count(review.schema(overflow)),1000)
        self.assertEqual(review.schema(frame)['schema']['properties']['reason']['enum'],list(review.DECISION_REASONS))
        # Capacity-only reference fixture; overflow uses full original candidate
        # and source view, not a shortened subset or invented source authority.
        messages=review.build_messages(overflow,question='Which facts?',response_language='en',sources_block=helper.sources)
        self.assertEqual(assembled_context_candidate(messages),first['answer'])
        shown=messages[1]['content'].split('\n\nSOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(shown),helper.sources)


if __name__=='__main__':unittest.main(verbosity=2)
