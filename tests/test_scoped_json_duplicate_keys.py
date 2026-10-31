"""The keyed wire rejects lost classifications after usage is settled."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_ask_review_reasoning_budget as setup
from machinemind.infrastructure import openai_transport as transport


class ScopedJsonDuplicateKeysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):setup.AskReviewReasoningBudgetTests.setUpClass();cls.m=setup.AskReviewReasoningBudgetTests.m
    @classmethod
    def tearDownClass(cls):setup.AskReviewReasoningBudgetTests.tearDownClass()

    def dispatch(self,kind,raw,marker=True):
        budget=self.m._assistant_core_new_budget('ask');posts=[]
        limits=(budget.max_llm_calls,budget.max_estimated_cost_usd,budget.deadline_monotonic)
        schema={'name':'offline_keyed_fixture','strict':True,'schema':{'type':'object','additionalProperties':False,
            'properties':{'table_record_coverage':{'type':'object','additionalProperties':False,
                'properties':{'T1':{'type':['string','null']}},'required':['T1']}},'required':['table_record_coverage']}}
        if marker is not None:schema['reject_duplicate_keys']=marker
        original=deepcopy(schema)
        class Response:
            status_code=200
            def json(self):
                value={'usage':{'input_tokens':100,'output_tokens':50}}
                if kind=='responses':value.update(status='completed',output_text=raw)
                else:value['choices']=[{'finish_reason':'stop','message':{'content':raw}}]
                return value
        def post(url,**kw):posts.append(kw['json']);return Response()
        common=dict(model='gpt-5.6-terra',json_schema=schema,timeout=20,max_output_tokens=800,
            api_key='offline-unused',url='https://offline.invalid/json',post_fn=post,
            current_budget_fn=lambda:budget,purpose='assistant_core_answer_contract_verifier')
        result=error=None
        try:
            if kind=='responses':result=transport.responses_json([{'role':'user','content':'Fixture'}],
                effort='low',company_id='offline-company',response_text_fn=transport.response_text,
                safety_identifier_fn=transport.safety_identifier,**common)
            else:result=transport.chat_json([{'role':'user','content':'Fixture'}],default_model='gpt-5.6-terra',**common)
        except ValueError as exc:error=exc
        self.assertEqual(schema,original);self.assertEqual(len(posts),1);self.assertEqual(budget.llm_calls,1)
        self.assertEqual(limits,(budget.max_llm_calls,budget.max_estimated_cost_usd,budget.deadline_monotonic))
        self.assertEqual(budget.call_log[0]['accounting_state'],'settled')
        self.assertEqual(budget.output_tokens,50)
        self.assertNotIn('reject_duplicate_keys',json.dumps(posts[0]))
        return result,error,budget,posts[0]

    def test_conflicting_nested_and_escaped_keys_fail_on_both_transports(self):
        cases=[
            '{"table_record_coverage":{"T1":"=0","T1":"!excluded"}}',
            '{"table_record_coverage":{"T1":"=0","\\u0054\\u0031":"!excluded"}}',
            '{"table_record_coverage":{"T1":"=0"},"other":{"private-diagnostic":"one","private-diagnostic":"two"}}',
            '{"table_record_coverage":{"T1":true,"T1":1}}',
            '{"table_record_coverage":{"T1":[0,1],"T1":[1,0]}}',
            '{"table_record_coverage":{"T1":{"x":[true]},"T1":{"x":[1]}}}',
            '{"table_record_coverage":{"T1":1,"T1":1.0}}']
        for kind in ('responses','chat'):
            for raw in cases:
                with self.subTest(kind=kind,raw=cases.index(raw)):
                    result,error,budget,_=self.dispatch(kind,raw)
                    self.assertIsNone(result);self.assertEqual(str(error),'structured_output_duplicate_key')
                    self.assertTrue(budget.call_log[0]['failed'])
                    self.assertNotIn('private-diagnostic',str(error));self.assertGreater(budget.estimated_cost_usd,0)

    def test_only_deep_identical_values_coalesce_without_changing_types_or_order(self):
        cases=[
            '{"table_record_coverage":{"T1":"=0","T1":"=0"}}',
            '{"table_record_coverage":{"T1":"=0","\\u0054\\u0031":"=0"}}',
            '{"table_record_coverage":{"T1":null,"T1":null}}',
            '{"table_record_coverage":{"T1":{"a":[true,1,"x"],"b":null},"T1":{"b":null,"a":[true,1,"x"]}}}']
        for kind in ('responses','chat'):
            for raw in cases:
                with self.subTest(kind=kind,raw=cases.index(raw)):
                    result,error,budget,_=self.dispatch(kind,raw)
                    self.assertIsNone(error);self.assertEqual(result,json.loads(raw))
                    self.assertFalse(budget.call_log[0].get('failed',False))

    def test_success_payload_and_reservation_equal_with_private_flag_absent(self):
        raw='{"table_record_coverage":{"T1":"=0"}}'
        for kind in ('responses','chat'):
            a,ae,ab,ap=self.dispatch(kind,raw,True);b,be,bb,bp=self.dispatch(kind,raw,None)
            self.assertIsNone(ae);self.assertIsNone(be);self.assertEqual(a,b);self.assertEqual(ap,bp)
            for key in ('reserved_input_tokens','max_output_tokens','reserved_cost_usd','estimated_cost_usd'):
                self.assertEqual(ab.call_log[0][key],bb.call_log[0][key])

    def test_opt_in_is_exact_server_boolean_and_legacy_parsing_stays_unchanged(self):
        raw='{"table_record_coverage":{"T1":"=0","T1":"!excluded"}}'
        for marker in (None,False,'true',1):
            schema={}if marker is None else{'reject_duplicate_keys':marker}
            self.assertEqual(transport.parse_model_json(raw,schema),json.loads(raw))
        for kind in ('responses','chat'):
            result,error,budget,_=self.dispatch(kind,raw,None)
            self.assertIsNone(error);self.assertEqual(result['table_record_coverage'],{'T1':'!excluded'})
            self.assertFalse(budget.call_log[0].get('failed',False))


if __name__=='__main__':unittest.main(verbosity=2)
