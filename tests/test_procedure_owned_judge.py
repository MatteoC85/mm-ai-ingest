"""A complete literal draft is judged, never silently edited or approved.

This profile changes no semantic truth rule. Model responses are fixtures;
packet compilation, reference decoding and the final contract are real.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from machinemind.ask import procedure_review as review
from test_procedure_reference_owners import accepts
import test_procedure_reviewer_routing as routing

class CompleteOwnedJudgeTests(unittest.TestCase):
    setUpClass=classmethod(routing.ProcedureReviewerRoutingTests.setUpClass.__func__)
    tearDownClass=classmethod(routing.ProcedureReviewerRoutingTests.tearDownClass.__func__)
    scenario=routing.ProcedureReviewerRoutingTests.scenario

    def schema(self,count):
        text=' '.join(f'Clause{i}.'for i in range(count))
        layout=review.block_layout('1. '+text)
        aliases=review.review_wire_aliases(layout,['manual:fixture'])
        catalog=review.reference_catalog(layout,aliases,
            '[manual:fixture] source_type=document\n'+text,{'manual:fixture':text})
        self.assertEqual(len(catalog['blocks']['b1']['units']),count)
        legacy={'name':'fixture','strict':True,'schema':{'type':'object','additionalProperties':False,
            'properties':{'outcome':{'type':'string'},'answer':{'type':'string'},
                'citation_ids':{'type':'array','items':{'type':'string'}}},
            'required':['outcome','answer','citation_ids']}}
        return review.review_schema(legacy,layout,[],[],operation_boundary=True,source_ordered=True,
            operation_source_ids=['manual:fixture'],wire_aliases=aliases,
            reference_catalogue=catalog,judge_only=True)['schema']

    def test_generated_replacement_domain_is_unrepresentable_and_originals_owner_bounded(self):
        for count in (1,2,8,20):
            with self.subTest(count=count):
                schema=self.schema(count)
                edge=schema['properties']['operation_boundary']['properties']['precedence']['items']
                original={'block_id':'b1','kind':'original','range':{'first':0,'last':count-1}}
                self.assertTrue(accepts(original,edge['properties']['action'],schema))
                bad=deepcopy(original);bad['range']['last']=count
                self.assertFalse(accepts(bad,edge['properties']['action'],schema))
                generated=deepcopy(original);generated['kind']='replacement'
                self.assertFalse(accepts(generated,edge['properties']['action'],schema))
                self.assertFalse(accepts(generated,edge['properties']['next'],schema))
                self.assertEqual(schema['properties']['edits']['maxItems'],0)
                self.assertEqual(schema['properties']['reply_mode']['enum'],['retain','reject'])
                self.assertEqual(schema['properties']['outcome']['enum'],['pass','no_sources'])

    def test_complete_flags_do_not_override_independent_semantic_safety_rejection(self):
        result,calls,attempts,_,_=self.scenario(reject=True)
        contract=result['meta']['assistant_core_validation']['answer_contract']
        self.assertFalse(contract['passed'])
        self.assertEqual(contract['semantic_verifier']['outcome'],'no_sources')
        self.assertIn('safety',contract['semantic_verifier']['missing_facets'])
        self.assertEqual(attempts,['gpt-5.6-luna'])
        self.assertEqual(len(calls),1)

    def test_unsolicited_edit_is_terminal_rejection_without_retaining_its_draft(self):
        result,_,attempts,_,_=self.scenario(reply_mode='edit')
        semantic=result['meta']['assistant_core_validation']['answer_contract']['semantic_verifier']
        self.assertEqual(semantic['outcome'],'no_sources')
        self.assertEqual(semantic['protocol_failure_code'],'procedure_review_judge_reply_invalid')
        self.assertTrue(semantic['_terminal_review_failure'])
        self.assertEqual(semantic['answer'],'')
        self.assertEqual(semantic['citation_ids'],[])
        self.assertEqual(attempts,['gpt-5.6-luna'])

    def test_server_guard_rejects_forged_modes_edits_and_replacement_even_if_schema_bypassed(self):
        base={'reply_mode':'retain','outcome':'pass','answer':'','edits':[],
              'operation_boundary':{'complete':True,'checked_blocks':[],
                    'precedence':[],'change_edges':[]}}
        malformed=[{**base,'outcome':'rewrite'},{**base,'reply_mode':'edit'},
            {**base,'answer':'Unreviewed replacement.'},
            {**base,'edits':[{'block_id':'b1','original':{'first':0,'last':0},'replacement_parts':['Changed.']}]},
            {**base,'reply_mode':'reject','outcome':'pass'},
            {**base,'reply_mode':'retain','outcome':'no_sources'}]
        for side in ('action','next'):
            audit=deepcopy(base['operation_boundary'])
            audit['precedence']=[{side:{'block_id':'b1','kind':'replacement','range':{'first':0,'last':8}}}]
            malformed.append({**base,'operation_boundary':audit})
        for value in malformed:
            before=deepcopy(value)
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'procedure_review_judge_reply_invalid'):
                review.validate_judge_reply(value)
            self.assertEqual(value,before)
        for mode,outcome in (('retain','pass'),('reject','no_sources')):
            value={**base,'reply_mode':mode,'outcome':outcome}
            before=deepcopy(value);review.validate_judge_reply(value);self.assertEqual(value,before)

    def test_source_coverage_hash_and_boundary_guards_still_run_on_valid_retain(self):
        for kwargs in ({'unknown_alias':True},{'malformed':'missing_audit'},
                       {'proof_tamper':'source'},{'proof_tamper':'raw_audit_digest'},
                       {'wire_change':'missing_facets'},{'wire_change':'missing_types'},
                       {'wire_change':'boundary_incomplete'},{'wire_change':'missing_checked_block'},
                       {'variant':'missing_draft_note'}):
            with self.subTest(kwargs=kwargs):
                result,_,attempts,_,_=self.scenario(reply_mode='retain',**kwargs)
                self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])
                self.assertEqual(attempts,['gpt-5.6-luna'])

    def test_repairs_and_other_ineligible_consumers_keep_full_edit_protocol(self):
        for variant in ('missing_packet','incomplete_packet','sequence_incomplete','notes_missing',
                        'translation','prior_contract','repair_facets','repair_types','repair_items',
                        'mixed','stale_draft','factual','root','smart','unguarded','no_completion',
                        'custom_models','reversed_configuration'):
            with self.subTest(variant=variant):
                _,calls,_,_,_=self.scenario(variant=variant)
                props=calls[0][1]['json_schema']['schema']['properties']
                if 'reply_mode' in props:
                    self.assertIn('edit',props['reply_mode']['enum'])
                else:
                    self.assertNotIn('edits',props)
        repaired,calls,_,_,_=self.scenario(variant='prior_contract',reply_mode='edit')
        self.assertTrue(repaired['meta']['assistant_core_validation']['answer_contract']['complete'])
        self.assertIn('edit',calls[0][1]['json_schema']['schema']['properties']['reply_mode']['enum'])

    def test_profile_cannot_be_requested_on_manual_translation_or_unbound_catalog(self):
        layout=review.block_layout('1. Keep supply isolated.')
        legacy={'name':'fixture','schema':{'properties':{},'required':[]}}
        for kwargs in ({},{'source_ordered':True},{'source_ordered':True,'operation_boundary':True},
                       {'source_ordered':True,'operation_boundary':True,'translation_language':'en'}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'procedure_review_judge_reply_invalid'):
                review.review_schema(legacy,layout,[],[],judge_only=True,**kwargs)

if __name__=='__main__':unittest.main(verbosity=2)
