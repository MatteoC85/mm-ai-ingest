"""Content-free debug logs survive missing response delivery without gate changes."""
from contextlib import redirect_stdout
from copy import deepcopy
import io,json
from pathlib import Path
import sys,time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_final_contract_diagnostics as fixture
from machinemind.ask import request_completion


class IndependentReviewLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FinalContractDiagnosticsTests.setUpClass()
        cls.m=fixture.FinalContractDiagnosticsTests.m

    @classmethod
    def tearDownClass(cls):fixture.FinalContractDiagnosticsTests.tearDownClass()

    def owner(self,debug=True):
        result=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=debug,request_id='PRIVATE_CLIENT_ID'),
            started=time.monotonic())
        self.addCleanup(result.close)
        return result

    def observe(self,**diagnostics):
        request_completion.record_contract_observation('semantic_review',
            {'outcome':'unavailable','reason':'factual_context_review_invalid'},diagnostics={
                'independent_review_outcome':'invalid','independent_review_gate':'quote_source',
                'independent_review_edit_count':3,'answer':'PRIVATE_ANSWER','source':'PRIVATE_SOURCE',**diagnostics})

    def events(self,stream):
        return [json.loads(line.split(' ',1)[1])for line in stream.getvalue().splitlines()
                if line.startswith('ASK_INDEPENDENT_REVIEW ')]

    def test_event_precedes_terminal_delivery_and_contains_only_bounded_server_fields(self):
        owner=self.owner();stream=io.StringIO()
        before=(owner.budget.llm_calls,owner.budget.estimated_cost_usd,owner.budget.deadline_monotonic)
        with redirect_stdout(stream):
            self.observe(independent_review_edit_count=1000)
            # Deliberately discard the returned response: logging was already emitted.
            owner.finish({'ok':False,'status':'error','result_code':'TECHNICAL_ERROR','meta':{}})
        events=self.events(stream);self.assertEqual(len(events),1)
        self.assertEqual(events[0],{'event':'independent_review','outcome':'invalid','gate':'quote_source',
            'reason':'factual_context_review_invalid','request_id':owner.budget.request_id,'edit_count':100})
        self.assertRegex(events[0]['request_id'],r'^[0-9a-f]{32}$')
        self.assertNotIn('PRIVATE',stream.getvalue())
        self.assertEqual(before,(owner.budget.llm_calls,owner.budget.estimated_cost_usd,owner.budget.deadline_monotonic))

    def test_debug_must_be_literal_true_and_other_observations_do_not_log(self):
        for debug in (False,None,'true',1):
            with self.subTest(debug=debug):
                owner=self.owner(debug);stream=io.StringIO()
                with redirect_stdout(stream):self.observe()
                self.assertEqual(self.events(stream),[]);owner.close()
        owner=self.owner();stream=io.StringIO()
        with redirect_stdout(stream):
            request_completion.record_contract_observation('final_contract',{'passed':False},
                diagnostics={'independent_review_gate':'quote_source'})
            request_completion.record_contract_observation('semantic_review',{'outcome':'rewrite'},diagnostics={})
        self.assertEqual(self.events(stream),[]);owner.close()

    def test_unknown_values_are_sanitized_count_and_event_count_are_bounded(self):
        owner=self.owner();stream=io.StringIO()
        with redirect_stdout(stream):
            for _ in range(20):
                self.observe(independent_review_outcome='PRIVATE_PROVIDER_TEXT',
                    independent_review_gate='PRIVATE_QUOTE',independent_review_edit_count=True,
                    raw_reply={'token':'PRIVATE_SECRET'})
        events=self.events(stream);self.assertEqual(len(events),8)
        for event in events:
            self.assertEqual(event['outcome'],'other');self.assertEqual(event['gate'],'other')
            self.assertNotIn('edit_count',event)
        self.assertNotIn('PRIVATE',stream.getvalue())
        self.assertEqual(len(owner.contract_checks),8)

    def test_logger_failure_and_new_owner_do_not_change_contract_or_share_request_state(self):
        owner=self.owner();contract={'outcome':'unavailable','reason':'factual_context_review_invalid'}
        diagnostics={'independent_review_outcome':'invalid','independent_review_gate':'quote_source','independent_review_edit_count':0}
        original=deepcopy((contract,diagnostics))
        with patch('builtins.print',side_effect=OSError('PRIVATE_LOGGER_ERROR')):
            request_completion.record_contract_observation('semantic_review',contract,diagnostics=diagnostics)
        self.assertEqual((contract,diagnostics),original)
        self.assertEqual(owner.contract_checks[0]['independent_review_gate'],'quote_source')
        first_id=owner.budget.request_id;owner.close()
        self.assertEqual(owner.contract_checks,[]);self.assertIsNone(request_completion.current())
        stream=io.StringIO()
        with redirect_stdout(stream):self.observe()
        self.assertEqual(self.events(stream),[])
        next_owner=self.owner();self.assertEqual(next_owner.contract_checks,[])
        with redirect_stdout(stream):self.observe()
        self.assertNotEqual(first_id,self.events(stream)[0]['request_id'])


if __name__=='__main__':unittest.main(verbosity=2)
