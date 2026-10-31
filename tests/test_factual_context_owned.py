"""Two-review factual ASK through real owner, authority, transport and finish.

Only SQL observations and model replies are fixtures. This establishes safe
orchestration and binding, not independent model quality or live latency.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_source_table_owned as table_fixture
from context_review_wire_fixture import encode_context_reply, assembled_context_candidate, original_context_sources
from machinemind.ask import factual_context_review, request_completion
from machinemind.ask.response_evidence import ResponseEvidenceError
from machinemind.evidence.contracts import EvidenceContractError
from machinemind.infrastructure import openai_transport, request_budget
from requests.exceptions import ReadTimeout

FIRST_ITEM = 'Check the oil level periodically and top up only if necessary.'
SAFE_ITEM = FIRST_ITEM + ' Before adding oil, use the documented personal protective equipment.'


def source_packet(message, end):
    return message.split('\n\nSOURCES:\n', 1)[1].split(end, 1)[0]


class ContextFixture(table_fixture.OwnedTableFixture):
    def __init__(self, test, *, second=None, first_change=None, fault=None):
        # The conditions are source evidence before registration, never inserted
        # into a mutable admitted row after its owner/session boundary.
        with patch.object(table_fixture, 'CONDITION', SAFE_ITEM):
            super().__init__(test)
        self.calls, self.rendered, self.post_calls = [], [], []
        self.initial_reads = self.session.summary()['registered_reads']
        renderer = self.owner._response_flow.execution._v13_sources_block
        def render(rows, **kwargs):
            block = renderer(rows, **kwargs)
            self.rendered.append(block)
            return block
        def provider(messages, **kwargs):
            stage = kwargs['json_schema']['name']
            self.calls.append({'schema': stage, 'messages': deepcopy(messages), 'kwargs': deepcopy(kwargs),
                               'budget_id': id(request_budget._v13_current_budget())})
            is_second = stage == 'factual_context_review'
            if is_second:
                self.first_candidate = assembled_context_candidate(messages)
                reply = {'outcome':'rewrite', 'edits':[], 'citation_ids':['inspection:p1','inspection:p2'],
                         'table_contexts':table_fixture.table_contexts(),
                         'checks':{name:True for name in factual_context_review.CHECKS}, 'reason':'offline independent fixture'}
                if second is not None:second(reply)
                first_sources = source_packet(self.calls[0]['messages'][1]['content'],'\n\nCURRENT_ANSWER:')
                reply = encode_context_reply(reply, messages, original_sources=first_sources)
            else:
                reply = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]', reason='offline first fixture',
                    citation_ids=['inspection:p1','inspection:p2'], covered_facets=['documented obligations'],
                    missing_facets=[], covered_answer_types=['checklist'], missing_answer_types=[],
                    expected_list_items=['[[SOURCE_TABLE:G1]]',FIRST_ITEM], covered_list_items=[],
                    missing_list_items=[], table_record_coverage=['T1=0','T2=0'])
                if first_change is not None:first_change(reply)
            class Response:
                status_code = 200
                def json(self):
                    return {'status':'completed', 'output_text':json.dumps(reply),
                            'usage':{'input_tokens':100,'output_tokens':100}}
            def post(url, **kw):
                self.post_calls.append({'stage':stage,'timeout':kw['timeout'],'payload':deepcopy(kw['json'])})
                if is_second:
                    if fault == 'timeout':raise ReadTimeout('offline second review timeout')
                    if fault == 'revoke':self.allowance = frozenset()
                    if fault == 'mutate':self.prepared['citations'][0]['chunk_full'] += '\nUNTRUSTED CHANGE'
                return Response()
            budget = request_budget._v13_current_budget()
            parsed = openai_transport.responses_json(messages, model=self.m.V13_FAST_MODEL,
                json_schema=kwargs['json_schema'], effort=kwargs['effort'], reasoning_mode=kwargs['reasoning_mode'],
                timeout=kwargs['timeout'], max_output_tokens=kwargs['max_output_tokens'],
                company_id=self.request.company_id, purpose=kwargs['purpose'], api_key='offline-unused',
                url='https://offline.invalid/responses', post_fn=post, current_budget_fn=lambda:budget,
                response_text_fn=openai_transport.response_text, safety_identifier_fn=openai_transport.safety_identifier)
            return parsed, 'offline-independent' if is_second else 'offline-first'
        self.owner._response_flow.execution = replace(self.owner._response_flow.execution,
            _v13_json_models=provider, _v13_sources_block=render)


class FactualContextOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        table_fixture.SourceTableOwnedTests.setUpClass()
        cls.m = table_fixture.SourceTableOwnedTests.m

    @classmethod
    def tearDownClass(cls):
        table_fixture.SourceTableOwnedTests.tearDownClass()

    def case(self, **kwargs):
        debug = kwargs.pop('debug', False)
        completion = request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
            payload=SimpleNamespace(company_id='offline-company',debug=debug), started=time.monotonic())
        f = ContextFixture(self, **kwargs)
        self.addCleanup(completion.close); self.addCleanup(f.close)
        return completion, f

    def assert_bound(self, result, f):
        self.assertEqual(result['status'], 'answered')
        contract = f.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['passed']); self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(hashlib.sha256(result['answer'].encode()).hexdigest(),contract['answer_sha256'])
        self.assertTrue(f.owner._response_flow.completed)
        self.assertEqual({c['citation_id'] for c in result['citations']},{'inspection:p1','inspection:p2'})
        self.assertEqual(f.session.summary()['registered_reads'],f.initial_reads)
        self.assertEqual(f.writer_calls,[])
        return contract

    def test_second_context_rewrite_preserves_source_packet_inventory_and_bound_answer(self):
        completion, f = self.case()
        limits = (completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd)
        result = f.run(); contract = self.assert_bound(result,f)
        self.assertEqual(len(f.post_calls),2); self.assertEqual(len(f.rendered),1)
        self.assertEqual(f.calls[-1]['schema'],'factual_context_review')
        self.assertEqual({c['budget_id'] for c in f.calls},{id(completion.budget)})
        first_sources = source_packet(f.calls[0]['messages'][1]['content'],'\n\nCURRENT_ANSWER:')
        second_sources = source_packet(f.calls[1]['messages'][1]['content'],'\n\nCOMPLETE_CANDIDATE:')
        self.assertEqual(first_sources,original_context_sources(second_sources,first_sources))
        self.assertNotIn('SOURCE_TABLE_PREVIEWS',f.calls[1]['messages'][1]['content'])
        self.assertNotIn('offline first fixture',f.calls[1]['messages'][1]['content'])
        self.assertEqual(contract['semantic_verifier']['outcome'],'rewrite')
        self.assertEqual(contract['semantic_verifier']['independent_context_review']['outcome'],'rewrite')
        self.assertIn(table_fixture.TABLE_QUALIFIER,contract['semantic_verifier']['answer'])
        self.assertNotIn(table_fixture.TABLE_QUALIFIER,f.first_candidate)
        self.assertIn('halve them under heavy loading',result['answer_html'])
        self.assertIn(self.m._assistant_core_redact_internal_text(FIRST_ITEM),contract['semantic_verifier']['answer'])
        self.assertEqual((completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd),limits)
        ceiling = min(4000,max(3200,self.m.V13_FAST_MAX_OUTPUT_TOKENS))
        self.assertEqual([call['kwargs']['max_output_tokens'] for call in f.calls], [ceiling,ceiling])
        self.assertTrue(all(row['payload']['max_output_tokens'] <= ceiling for row in f.post_calls))
        self.assertGreater(f.calls[1]['kwargs']['timeout'],25)
        self.assertGreater(f.post_calls[1]['timeout'],25)
        self.assertTrue(any(row['review_reserve_seconds']==25 for row in completion.provider_schedules))
        self.assertEqual(completion.provider_schedules[-1]['review_reserve_seconds'],0)

    def test_independent_non_table_ppe_edit_keeps_source_quotations_immutable(self):
        completion, f = self.case(second=lambda r:r.update(outcome='rewrite',edits=[{'target':'item:1','text':SAFE_ITEM}]))
        result = f.run(); contract = self.assert_bound(result,f)
        verified = contract['semantic_verifier']
        self.assertIn(table_fixture.TABLE_QUALIFIER,verified['answer'])
        self.assertEqual(verified['independent_context_review']['outcome'],'rewrite')
        self.assertEqual(verified['independent_context_review']['edit_count'],1)
        for sentence in SAFE_ITEM.split('. '):
            self.assertIn(sentence.rstrip('.'),result['answer'])
        self.assertIn('Inspect wear every 360 hours and grease only if necessary',result['answer'])
        self.assertEqual([row['inventory_items'] for row in verified['table_record_coverage']],[[0],[0]])
        self.assertEqual(len(f.post_calls),2)

    def test_invalid_independent_verdict_never_falls_back_to_first_draft(self):
        changes = [
            lambda r:r.update(outcome='no_sources'),
            lambda r:r['checks'].update(safety=False),
            lambda r:r.update(outcome='rewrite',edits=[{'target':'item:0','text':'Grease the slide every day.'}]),
            lambda r:r.update(citation_ids=['foreign-tenant:p1','inspection:p2']),
            lambda r:r.update(outcome='rewrite',edits=[{'target':'item:1','text':'[[SOURCE_TABLE:G1]]'}]),
            lambda r:r.update(table_contexts=[]),
            lambda r:r['table_contexts'][0].update(target='item:1'),
            lambda r:r['table_contexts'][0]['source_quotes'][0].update(quote='Invented condition not in any emitted source.'),
            lambda r:r['table_contexts'][0]['source_quotes'][0].update(citation_id='inspection:p2'),
            lambda r:r['table_contexts'][0].update(documentary_limits_checked=False),
        ]
        for index,change in enumerate(changes):
            with self.subTest(case=index):
                completion, f = self.case(second=change,debug=True)
                if index in (0,1,9):
                    result=f.run()
                    self.assertEqual(result['status'],'no_sources')
                    self.assertEqual(result.get('citations') or [],[])
                    self.assertNotIn(FIRST_ITEM,json.dumps(result))
                else:
                    with self.assertRaisesRegex(ResponseEvidenceError,'canonical verifier unavailable: factual_context_review_invalid'):
                        f.run()
                    self.assertFalse(f.owner._response_flow.completed)
                    terminal=completion.finish({'ok':False,'status':'error','result_code':'TECHNICAL_ERROR','meta':{}})
                    self.assertNotIn(FIRST_ITEM,json.dumps(terminal))
                row=completion.contract_checks[-1]
                self.assertEqual(row['independent_review_gate'],[
                    'model_rejected','checks_false','edit_target','citation_alias',
                    'edit_target','table_count','table_target','quote_source','quote_source',
                    'documentary_limits_unchecked'][index])
                self.assertEqual(len(f.post_calls),2)
                f.close(); completion.close()

    def test_atomic_quote_and_current_citations_need_not_be_repeated_as_additional_ids(self):
        completion, f = self.case(second=lambda r:r.update(citation_ids=[]))
        result=f.run()
        self.assert_bound(result,f)
        self.assertEqual(len(f.post_calls),2)
        self.assertIn('additional_citation_ids',f.calls[1]['kwargs']['json_schema']['schema']['required'])
        self.assertIn(table_fixture.TABLE_QUALIFIER,result['answer'])
        self.assertEqual({c['citation_id']for c in result['citations']},{'inspection:p1','inspection:p2'})

    def test_rejected_or_malformed_first_reply_does_not_trigger_second_review(self):
        changes=[lambda r:r.update(outcome='no_sources'),
                 lambda r:r.update(expected_list_items=['[[SOURCE_TABLE:G99]]',FIRST_ITEM]),
                 lambda r:r.update(missing_facets=['documented obligations'])]
        for index,change in enumerate(changes):
            with self.subTest(case=index):
                completion,f=self.case(first_change=change)
                result=f.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(len(f.post_calls),1)
                self.assertEqual(result.get('citations') or [],[])
                f.close();completion.close()

    def test_second_transport_timeout_cannot_publish_first_candidate(self):
        completion,f=self.case(fault='timeout')
        # The fixture omits the outer HTTP terminal mapper. The owned consumer
        # retains its transport fault and cannot hand off the first candidate.
        with self.assertRaises(ReadTimeout):f.run()
        self.assertFalse(hasattr(f,'generated'))
        self.assertFalse(f.owner._response_flow.completed)
        self.assertEqual(len(f.post_calls),2)
        self.assertEqual(completion.budget.call_log[-1]['accounting_state'],'uncertain')

    def test_current_authority_checked_after_second_provider(self):
        for fault in ('revoke','mutate'):
            with self.subTest(fault=fault):
                completion,f=self.case(fault=fault)
                with self.assertRaises(EvidenceContractError):f.run()
                self.assertEqual(len(f.post_calls),2)
                self.assertFalse(f.owner._response_flow.completed)
                f.close();completion.close()

    def test_second_rewrite_cannot_bypass_existing_quantity_grounding_and_final_hash(self):
        completion,f=self.case(second=lambda r:r.update(outcome='rewrite',edits=[
            {'target':'item:1','text':'Top up the assembly every 999 hours.'}]))
        # As in the table fixture, the outer bounded repair loop is deliberately
        # absent. A rejected, modified draft cannot finish as an approved result.
        with self.assertRaisesRegex(ResponseEvidenceError,'final rejected draft lacks observed no-evidence builder'):
            f.run()
        contract=f.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed']);self.assertFalse(contract['semantic_answer_bound'])
        self.assertNotIn('999 hours',f.validated.get('answer') or '')
        self.assertEqual(len(f.post_calls),2)
        self.assertFalse(f.owner._response_flow.completed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
