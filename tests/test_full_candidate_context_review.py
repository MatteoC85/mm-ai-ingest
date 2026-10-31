"""Complete factual review view and existing corrections; no model-judgment claim."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import factual_context_review as review, review_source_wire as compression
from machinemind.ask import source_table_review as tables, enumeration_review, request_completion
from context_review_wire_fixture import assembled_context_candidate, original_context_sources
import test_context_citation_roundtrip as variants
import test_source_table_review as source_fixture
import test_source_table_owned as owned

SCOPE = 'This maintenance table concerns the stated assembly.'
ACTUAL_RECORD = ('- [PDF_TABLE_RECORD rows=5-6 columns=5] c1 "Componenti" r5="Impianto lubrificazione Automatica" | '
    'c2 "Ore di Funzionamento" r5-6="Ogni giorno" | c3 "Tipo di lubrificante" r5-6="ISO SAE 220" | '
    'c4 "Quantità" r5-6="Q.B." | c5 "NOTE" r5-6="" | '
    'c1 "Componenti" r6="Verifica livello olio nel lubrificatore con eventuale aggiunta"')


def candidate_view(messages):
    return json.loads(messages[1]['content'].split('\n\nCOMPLETE_CANDIDATE:\n',1)[1].split('\n\n',1)[0])


class FullCandidateViewTests(unittest.TestCase):
    def assert_full(self, frame, messages):
        view = candidate_view(messages)
        self.assertEqual([item['text'] for item in view['inventory']], list(frame.items))
        self.assertEqual([item['target'] for item in view['inventory']], ['item:'+str(i) for i in range(len(frame.items))])
        self.assertEqual([item['immutable'] for item in view['inventory']], [i in frame.immutable_indices for i in range(len(frame.items))])
        self.assertFalse(any('source_table' in item or 'caption' in item for item in view['inventory']))
        self.assertEqual(assembled_context_candidate(messages), frame.answer)

    def test_actual_shared_span_and_contradictory_prose_are_both_visible_without_reconstruction(self):
        body = SCOPE+'\n'+ACTUAL_RECORD
        prose = 'Controllare periodicamente il livello dell’olio.'
        rows = [source_fixture.row('unseen-table:p74',body,74), source_fixture.row('unseen-prose:p62',prose,62)]
        sources, views, ids = source_fixture.pack(rows)
        catalog = tables.compile_catalog(sources, views, 'it')
        self.assertEqual(len(catalog.entries), 1)
        false_claim = 'Per il livello è indicato soltanto periodicamente, senza una frequenza precisa.'
        raw = dict(outcome='rewrite', answer='[[FACTUAL_INVENTORY]]\n'+false_claim,
            expected_list_items=[catalog.entries[0].token, false_claim], citation_ids=list(views),
            table_record_coverage=[rid+'=0' for rid in ids], missing_facets=[], missing_answer_types=[], missing_list_items=[])
        decoded = tables.resolve_reply(raw,catalog,admitted_citation_ids=list(views))
        first = enumeration_review.resolve_inventory_answer(decoded,canonical_inventory=decoded['expected_list_items'],allowed=True)
        frame = review.prepare_review(raw,first,catalog=catalog,admitted_citation_ids=list(views),sources_block=sources)
        messages = review.build_messages(frame,question='Quali frequenze sono documentate?',response_language='it',sources_block=sources)
        self.assert_full(frame,messages)
        shown = candidate_view(messages)
        self.assertIn('Ore di Funzionamento: «Ogni giorno»',shown['inventory'][0]['text'])
        self.assertIn('Verifica livello olio nel lubrificatore con eventuale aggiunta',shown['inventory'][0]['text'])
        self.assertEqual(shown['inventory'][1]['text'],false_claim)
        self.assertIn(false_claim,shown['template'])
        emitted=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
        self.assertEqual(original_context_sources(emitted,sources),sources)
        self.assertIn('r5-6="Ogni giorno"',sources)
        # Visibility establishes that the model can compare these claims. The
        # fixture does not claim the model actually recognizes contradiction.

    def test_bound_atomic_and_complete_fallback_views_are_exact_with_dictionary_sources(self):
        helper = variants.CitationRoundtripTests()
        used_dictionary=False
        for mode in ('bound','unbound_atomic','non_atomic'):
            first, frame, sources, messages, _, _ = helper.tables(mode)
            with self.subTest(mode=mode,compressed=False):self.assert_full(frame,messages)
            compact = compression.compact_sources_block(sources)
            used_dictionary=used_dictionary or compact.startswith(compression._START)
            # Rebuild through the real frame boundary rather than changing its
            # source hash or copying quote units into a manufactured frame.
            bodies=dict(frame.source_bodies)
            pages={header[1]:int(re.search(r'\bp=(\d+)',header[0])[1]) for header in tables._HEADER.finditer(sources)}
            views={cid:{'body':body,'page_from':pages[cid],'page_to':pages[cid]} for cid,body in bodies.items()}
            catalog=tables.compile_catalog(sources,views,'en')
            bycid={entry.citation_id:entry for entry in catalog.entries}
            raw={**deepcopy(first),'answer':frame.template,'expected_list_items':list(frame.items)}
            for index,cid in frame.table_owners:raw['expected_list_items'][index]=bycid[cid].token
            rebuilt=review.prepare_review(raw,first,catalog=catalog,admitted_citation_ids=frame.admitted_citation_ids,sources_block=compact)
            messages=review.build_messages(rebuilt,question='Which intervals?',response_language='en',sources_block=compact)
            with self.subTest(mode=mode,compressed=True):
                self.assert_full(rebuilt,messages)
                emitted=messages[1]['content'].split('SOURCES:\n',1)[1].split('\n\nCOMPLETE_CANDIDATE:',1)[0]
                self.assertEqual(original_context_sources(emitted,compact),compact)
                self.assertEqual(compression.restore_sources_block(compact),sources)
        self.assertTrue(used_dictionary)

    def test_unseen_document_language_and_categorical_values_remain_literal(self):
        for doc,subject,value in [('press-manual','Filter differential pressure','Weekly'),
                                  ('pompe-guide','Contrôle du niveau','Chaque jour')]:
            body=SCOPE+'\n'+source_fixture.BODY.replace('Slides',subject).replace('Daily',value)
            sources,views,_=source_fixture.pack([source_fixture.row(doc+':p1',body)])
            catalog=tables.compile_catalog(sources,views,'en');entry=catalog.entries[0]
            raw=dict(outcome='rewrite',answer='[[FACTUAL_INVENTORY]]',expected_list_items=[entry.token],
                citation_ids=[entry.citation_id],table_record_coverage=[rid+'=0'for rid in entry.record_ids])
            first=tables.resolve_reply(raw,catalog)
            first=enumeration_review.resolve_inventory_answer(first,canonical_inventory=first['expected_list_items'],allowed=True)
            frame=review.prepare_review(raw,first,catalog=catalog,admitted_citation_ids=list(views),sources_block=sources)
            messages=review.build_messages(frame,question='Which documented values?',response_language='en',sources_block=sources)
            self.assert_full(frame,messages)
            self.assertIn(subject,candidate_view(messages)['inventory'][0]['text'])
            self.assertIn(value,candidate_view(messages)['inventory'][0]['text'])


class FullCandidateOwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):owned.SourceTableOwnedTests.setUpClass();cls.m=owned.SourceTableOwnedTests.m
    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()

    def test_existing_edits_correct_item_and_template_with_bound_public_result_same_limits(self):
        false='The narrative says periodically; no precise frequency is documented.'
        corrected='The narrative says periodically; the maintenance table states Daily for the level-check entry.'
        table_body=(SCOPE+'\n- [PDF_TABLE_RECORD rows=5-6 columns=2] c1 "Component" r5="Lubrication system" | '
            'c2 "Interval" r5-6="Daily" | c1 "Component" r6="Level-check entry"\n'
            '- [PDF_TABLE_RECORD rows=1 columns=2] c1 r1="Electrical integrity" | c2 r1="Monthly"')
        with ExitStack() as cleanup:
            completion=request_completion.RequestCompletion(runtime=self.m._assistant_core_request_flow_runtime(),
                payload=SimpleNamespace(company_id='offline-company'),started=time.monotonic())
            cleanup.callback(completion.close)
            def first(reply):reply.update(answer='[[FACTUAL_INVENTORY]]\n'+false,expected_list_items=['[[SOURCE_TABLE:G1]]',false])
            def second(reply):reply.update(edits=[{'target':'item:1','text':corrected},
                {'target':'template-answer','text':'[[FACTUAL_INVENTORY]]\n'+corrected}])
            with patch.object(owned,'TABLE_BODY',table_body),patch.object(owned,'TABLE_QUALIFIER',SCOPE),patch.object(owned,'CONDITION','Check the level periodically.'):
                f=owned.OwnedTableFixture(self,change=first,second_change=second)
            cleanup.callback(f.close)
            limits=(completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd)
            before=deepcopy(f.prepared);reads=f.session.summary()['registered_reads']
            # ContextFixture generates its verdict lazily, so retain the exact
            # source-owned context during that call as well.
            with patch.object(owned,'TABLE_QUALIFIER',SCOPE):result=f.run()
            self.assertEqual(result['status'],'answered')
            self.assertEqual(len(f.posts),2);self.assertEqual(f.writer_calls,[])
            self.assertEqual(f.prepared,before);self.assertEqual(f.session.summary()['registered_reads'],reads)
            self.assertEqual((completion.budget.deadline_monotonic,completion.budget.max_estimated_cost_usd),limits)
            self.assertEqual(f.budget_ids,[id(completion.budget)]*2)
            self.assertNotIn(false,result['answer']);self.assertIn(corrected,result['answer'])
            self.assertNotIn(false,result['answer_html']);self.assertIn('Daily',result['answer_html'])
            contract=f.validated['meta']['assistant_core_validation']['answer_contract']
            self.assertTrue(contract['passed']);self.assertTrue(contract['semantic_answer_bound'])
            self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
            self.assertEqual(contract['semantic_verifier']['independent_context_review']['edit_count'],2)
            messages=f.reviewer_calls[1][0];view=candidate_view(messages)
            self.assertIn('Level-check entry',view['inventory'][0]['text'])
            self.assertIn('Daily',view['inventory'][0]['text'])
            self.assertEqual(view['inventory'][1]['text'],false)


if __name__=='__main__':unittest.main(verbosity=2)
