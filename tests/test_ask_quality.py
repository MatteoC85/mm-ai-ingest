"""Offline ASK evidence regressions; no claim about live model answer quality."""
import copy
import importlib
import os
from pathlib import Path
import random
import socket
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from machinemind.retrieval import candidate_ranking as ranking

def denied(*args,**kwargs):raise AssertionError('OFFLINE_TEST_EXTERNAL_IO_DENIED')

def candidate(index,text,**extra):
    return {'citation_id':f'manual:p6:c{index}','bubble_document_id':'manual',
            'page_from':6,'page_to':6,'chunk_index':index,'snippet':text,
            'chunk_full':text,'similarity':0.8,'semantic_similarity':0.8,
            'retrieval_score':0.8,'exact_machine_scope':True,'source_type':'document',**extra}

def dedup(items,limit=24,**kwargs):
    return ranking.dedup_citations_by_snippet(items,limit,
        runtime=ranking.DedupCitationsBySnippetRuntime(lambda text:text),**kwargs)

class AskDedupQualityTests(unittest.TestCase):
    def test_shared_long_heading_keeps_distinct_operational_steps_it_en(self):
        for prefix,tails in (
            ('Verifiche della sequenza di avvio. '*18,('Disattivare la pressione prima della regolazione.','Confermare il sensore prima del riavvio.')),
            ('Checks for the starting sequence. '*18,('Release pressure before adjustment.','Confirm the sensor before restarting.')),
        ):
            with self.subTest(language=prefix[:10]):
                items=[candidate(i,prefix+tail) for i,tail in enumerate(tails)]
                self.assertEqual([x['citation_id'] for x in dedup(items)],[x['citation_id'] for x in items])

    def test_same_visible_snippet_does_not_hide_different_full_context(self):
        items=[candidate(i,'Common display preview',chunk_full='Complete source '+str(i)) for i in range(2)]
        self.assertEqual(len(dedup(items)),2)

    def test_standalone_numbers_remain_distinct_evidence(self):
        items=[candidate(i,f'Pressure setting\n{n}\nbar') for i,n in enumerate((4,9))]
        self.assertEqual(len(dedup(items)),2)

    def test_repeated_actions_are_not_removed_from_identity(self):
        self.assertEqual(len(dedup([candidate(0,'Open\nClose\nOpen'),candidate(1,'Open\nClose')])),2)

    def test_case_sensitive_unit_prefixes_are_not_merged(self):
        self.assertEqual(len(dedup([candidate(0,'Measured value 1 MPa'),candidate(1,'Measured value 1 mPa')])),2)

    def test_true_duplicate_prefers_best_occurrence_and_keeps_lineage(self):
        first=candidate(0,'Inspect the sensor before restarting.',retrieval_score=0.5)
        best=candidate(1,'Inspect the sensor before restarting.',retrieval_score=0.9)
        observed=[];result=dedup([first,best],lineage=observed.append)
        self.assertEqual(len(result),1);self.assertIs(result[0],best)
        self.assertEqual(observed,[(((0,1),),)])

    def test_different_source_locations_remain_distinct(self):
        items=[candidate(0,'Same text'),candidate(1,'Same text',bubble_document_id='other'),candidate(2,'Same text',page_from=9,page_to=9)]
        self.assertEqual(len(dedup(items)),3)

    def test_bounded_output_and_input_metadata_unchanged(self):
        items=[candidate(i,'Distinct source '+str(i),company_id='tenant-local',machine_id='machine-local') for i in range(8)]
        original=copy.deepcopy(items);result=dedup(items,3)
        self.assertEqual(len(result),3);self.assertEqual(items,original)
        self.assertTrue(all(x['company_id']=='tenant-local' and x['machine_id']=='machine-local' for x in result))
        self.assertEqual(dedup([],5),[])

    def test_unseen_ids_and_long_prefixes_do_not_control_selection(self):
        rng=random.Random(68217)
        for _ in range(25):
            prefix=' '.join(str(rng.randrange(10000)) for _ in range(90))
            items=[candidate(i,prefix+' unique trailing observation '+str(i),citation_id=str(rng.getrandbits(100))) for i in range(4)]
            self.assertEqual(len(dedup(items)),4)

class AskRealPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards=ExitStack()
        cls.guards.enter_context(patch.dict(os.environ,{'MM_USAGE_ENFORCEMENT':'off','AI_INTERNAL_SECRET':'offline-ask-fixture-only','OPENAI_API_KEY':'offline-unused','MM_INGEST_LEDGER_AUTO_DDL':'0'}))
        cls.guards.enter_context(patch.object(socket,'create_connection',denied))
        cls.guards.enter_context(patch('requests.sessions.Session.request',denied))
        cls.guards.enter_context(patch('psycopg2.connect',denied))
        cls.m=importlib.import_module('main')
    @classmethod
    def tearDownClass(cls):cls.guards.close()

    def request_and_decision(self,query,language):
        m=self.m
        request=m.AssistantCoreRequest(query=query,requested_mode='ask',response_language=language,
            company_id='offline-company',machine_id='offline-machine',ai_scope='machine_all',top_k=5)
        decision=m.AssistantCoreDecision(request_kind=m.KIND_PROCEDURE,effective_mode='ask',confidence=0.95,
            requested_mode_fit=True,evidence_state=m.EVIDENCE_SUPPORTED,evidence_policy='evidence_required',
            information_task=m.INFO_PROCEDURE_FULL,required_answer_types=(m.REQ_ORDERED_ACTIONS,))
        return request,decision

    def test_actual_preparation_preserves_complementary_passages_and_links(self):
        for query,language in [('Come rimetto in funzione il movimento dopo una regolazione?','it'),('How can motion resume after adjustment?','en')]:
            with self.subTest(language=language):
                request,decision=self.request_and_decision(query,language)
                prefix=('Documented restart checks. '*20)
                items=[candidate(i,prefix+tail) for i,tail in enumerate(('First isolate the pressure supply before adjustment.','Then confirm the position sensor before restarting.'))]
                result=self.m._assistant_core_prepare_evidence(request,{'candidates':items},decision)
                self.assertTrue(result['supported'])
                chosen=result['retrieval']['citations']
                self.assertEqual({c['citation_id'] for c in chosen},{c['citation_id'] for c in items})
                self.assertTrue(all(c['bubble_document_id']=='manual' and c['page_from']==6 for c in chosen))
                links=self.m._build_rg_links('offline-company',chosen,file_map_fn=lambda company,ids:{'manual':'https://files.example.test/manual.pdf'})
                self.assertTrue(links)

    def test_actual_preparation_has_no_answer_evidence_when_sources_empty(self):
        request,decision=self.request_and_decision('Explain an undocumented adjustment.','en')
        result=self.m._assistant_core_prepare_evidence(request,{'candidates':[]},decision)
        self.assertFalse(result['supported']);self.assertEqual(result['retrieval']['citations'],[])

if __name__=='__main__':unittest.main(verbosity=2)
