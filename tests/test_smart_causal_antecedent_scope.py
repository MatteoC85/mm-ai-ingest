"""Producer contract and immutable source-review boundaries, never model quality."""
from copy import deepcopy
from pathlib import Path
import json,sys,unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_smart_review as fixture
import test_smart_generation_scheduling as scheduling
from machinemind.retrieval import smart_review as review,smart_evidence

def make_case(machine='press-z',direction='increase',condition='stock thickness'):
    scope={'company_id':'synthetic-company','machine_id':machine,'ai_scope':'machine_all'}
    causal=f'A documented {direction} in {condition} caused intermittent feed loss when the grip setting was not adapted.'
    prevention='Prevention: check the grip setting whenever material, tooling or stock dimensions change.'
    source={'citation_id':machine+':p8','bubble_document_id':machine+'-manual','source_type':'document',
        'company_id':scope['company_id'],'machine_id':machine,'page_from':8,'page_to':8,
        'display_title':'Synthetic setup record','display_label':'Synthetic setup record page8',
        'snippet_clean':'Display metadata only','chunk_full':causal+'\n'+prevention+
        '\nThe setup log records the prior and new stock values. Use existing records to distinguish that condition.'}
    packet=smart_evidence.build([source],scope=scope)
    _,step=fixture.fixture();h=deepcopy(step['hypotheses'][0])
    h.update(label='Possible grip-setting mismatch',description=f'If {condition} had a documented {direction} and the grip setting was not adapted, intermittent feed loss is possible.',
        why=f'The reported feed loss is compatible; whether that {direction} in {condition} occurred remains unknown.',
        probability_pct=40,probability_band='unknown',status='open',checks=[],evidence_ids=[source['citation_id']])
    q=step['question'];q.update(question_text=f'From already available setup records, was there an {direction} in {condition}? Do not inspect or operate the machine.',
        why_asked='This establishes the specific antecedent of the documented mechanism without assuming it occurred.',
        safety_level='normal',safety_note='',target_hypotheses=['H1'],
        options=[{'id':'yes','label_it':'Sì','label_en':'Yes'},
                 {'id':'unknown','label_it':'Non so','label_en':"I don't know"}])
    step.update(hypotheses=[h],operator_summary='The antecedent is unresolved.',final_result={})
    return source,packet,step,causal,prevention

class CausalAntecedentScopeTests(unittest.TestCase):
    def prepare(self,packet,step):
        return review.prepare(step=step,packet=smart_evidence.review_packet(packet),
            symptom_text='Intermittent feed loss after a setup change; the prior and new stock values have not been checked.',history=[])

    def test_broader_prevention_trigger_remains_an_immutable_rejected_causal_alternative(self):
        source,packet,step,_,_=make_case()
        step['hypotheses'][0]['description']='If material or tooling or any stock dimension differs, grip-setting mismatch can cause the feed loss.'
        prepared=self.prepare(packet,step);before=deepcopy(prepared)
        parsed=fixture.parsed_review(prepared,rejected=(0,),reject_question=True)
        parsed['decisions'][0].update(reason='unsupported',blocking_checks=[],note='The prevention list does not document each alternative as causal.')
        parsed['decisions'][1].update(reason='wrong_target',blocking_checks=[],note='The declared hypothesis is rejected.')
        with self.assertRaisesRegex(review.SmartReviewError,'no_supported_hypotheses')as caught:
            review.resolve(prepared=prepared,parsed=parsed,probability_band=lambda _: 'unknown',continuity_policy=review.continuity.POLICY_VERSION)
        self.assertEqual(prepared,before)
        self.assertEqual(caught.exception.validated_summary['accepted_causes'],0)
        self.assertIn(step['hypotheses'][0]['description'],fixture.refs.canonical(prepared['references']['frozen']))

    def test_supported_direction_and_unknown_survive_without_expanding_to_preventive_conditions(self):
        for machine,direction,condition in [('press-z','increase','stock thickness'),('roller-q','decrease','strip width')]:
            with self.subTest(machine=machine):
                source,packet,step,causal,prevention=make_case(machine,direction,condition)
                prepared=self.prepare(packet,step)
                packet_before=deepcopy(packet);step_before=deepcopy(step)
                parsed=fixture.parsed_review(prepared,rejected=())
                # This is an explicitly supplied semantic verdict, not a claim
                # that the validator or the prompt proves causal entailment.
                out,seal,summary=review.resolve(prepared=prepared,parsed=parsed,probability_band=lambda _: 'unknown',continuity_policy=review.continuity.POLICY_VERSION)
                self.assertEqual(packet,packet_before);self.assertEqual(step,step_before)
                self.assertEqual(out['hypotheses'][0]['description'],step['hypotheses'][0]['description'])
                self.assertEqual(out['hypotheses'][0]['why'],step['hypotheses'][0]['why'])
                self.assertEqual(out['hypotheses'][0]['checks'],[])
                self.assertEqual(out['hypotheses'][0]['evidence_ids'],[source['citation_id']])
                self.assertEqual(out['question']['options'][-1]['id'],'unknown')
                self.assertEqual(seal['input_step']['question']['question_text'],step['question']['question_text'])
                self.assertEqual(summary['accepted_hypothesis_ids'],['H1'])
                self.assertEqual(review.claims_digest(out),seal['output_claims_digest'])
                source_text=fixture.refs.canonical(prepared['references']['model_packet'])
                self.assertIn(causal,source_text);self.assertIn(prevention,source_text)

class CausalAntecedentProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduling.SmartGenerationSchedulingTests.setUpClass();cls.m=scheduling.SmartGenerationSchedulingTests.m
    @classmethod
    def tearDownClass(cls):scheduling.SmartGenerationSchedulingTests.tearDownClass()
    def test_start_and_answer_use_same_contract_exact_packet_schema_and_single_generation(self):
        source,packet,step,causal,prevention=make_case()
        evidence=smart_evidence.evidence_block(packet)
        state={'symptom_text':'Feed loss after setup; dimensions unknown.','history':[],
            'evidence':[source],'grounding_packet':packet,'company_id':'synthetic-company','machine_id':'press-z',
            'max_questions':6,'current_question':step['question'],'hypotheses':step['hypotheses']}
        calls=[]
        def provider(messages,**kwargs):calls.append((deepcopy(messages),deepcopy(kwargs)));return deepcopy(step)
        with patch.object(self.m,'_assistant_core_sd_json_models',side_effect=provider):
            self.m._sd_llm_step_start(symptom_text=state['symptom_text'],language='en',max_questions=6,max_hypotheses=2,
                evidence_block=evidence,evidence_ids=[source['citation_id']])
            self.m._sd_llm_step_answer(state=state,answer={'value':'unknown','label':"I don't know"},language='en',max_hypotheses=2)
        self.assertEqual(len(calls),2)
        for phase,(messages,kwargs)in zip(('start','answer'),calls):
            self.assertEqual(kwargs['phase'],phase)
            self.assertEqual(kwargs['json_schema'],self.m._sd_schema(max_hypotheses=2,max_options=4))
            self.assertEqual(kwargs['timeout'],self.m.SMART_DIAGNOSTIC_LLM_TIMEOUT)
            self.assertEqual(kwargs['models'],[self.m.SMART_DIAGNOSTIC_MODEL,self.m.DIAGNOSTIC_EVIDENCE_MODEL,self.m.OPENAI_CHAT_MODEL])
            self.assertIn(review.generation_hypothesis_instruction(),messages[0]['content'])
            supplied=messages[1]['content'].split('INDEXED_MACHINE_EVIDENCE:\n',1)[1].split('\n\nReturn JSON.',1)[0]
            self.assertEqual(supplied,evidence)
            self.assertIn(causal,supplied);self.assertIn(prevention,supplied)
        answer_text=calls[1][0][1]['content']
        self.assertIn('"value": "unknown"',answer_text)

if __name__=='__main__':unittest.main(verbosity=2)
