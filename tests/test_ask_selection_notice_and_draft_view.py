"""Synthetic replies test transport contracts, not new model judgements."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.ask import source_table_review as st
import test_useful_table_selection as useful
import test_source_ordered_review as procedure

class SelectionNoticeTests(unittest.TestCase):
    def test_single_language_enum_rejects_model_or_factual_qualifications(self):
        catalog, _, _ = useful.catalog_for([useful.pure.row('synthetic-manual:p1', useful.duplicates.SCOPE+'\n'+useful.duplicates.CELLS)])
        base={'schema':{'type':'object','properties':{},'required':[],'additionalProperties':False}}
        for language in ('it','en'):
            expected=st.selection_notice(language)
            schema=st.reference_schema(base,catalog,selected_records_only=True,expected_scope_note=expected)
            self.assertEqual(schema['schema']['properties']['selection_scope_note']['enum'],[expected])
            for notice in ('This is exhaustive.', expected+' Only for model TEST-18.', st.selection_notice('it' if language=='en' else 'en'), ''):
                raw={**useful.wire(catalog),'selection_scope_note':notice}
                result=st.resolve_reply(raw,catalog,selected_records_only=True,expected_scope_note=expected,
                    admitted_citation_ids={'synthetic-manual:p1'})
                self.assertEqual(result['outcome'],'no_sources')
                self.assertEqual(raw['selection_scope_note'],notice)  # No post-hoc stripping of conditions.
        self.assertEqual(base['schema']['properties'],{})
        with self.assertRaises(ValueError):st.selection_notice('unapproved-language')

    def test_neutral_notice_does_not_supply_facts_or_citation_authority(self):
        catalog, _, _ = useful.catalog_for([useful.pure.row('synthetic-manual:p1', useful.duplicates.SCOPE+'\n'+useful.duplicates.CELLS)])
        expected=st.selection_notice('en'); raw=useful.wire(catalog)
        for changes in ({'expected_list_items':[]},{'citation_ids':['foreign-manual:p1']},
                        {'expected_list_items':['[[SOURCE_TABLE:G999]]']}):
            result=st.resolve_reply({**raw,**changes},catalog,selected_records_only=True,
                expected_scope_note=expected,admitted_citation_ids={'synthetic-manual:p1'})
            self.assertEqual(result['outcome'],'no_sources')
        good=st.resolve_reply(raw,catalog,selected_records_only=True,expected_scope_note=expected,
            admitted_citation_ids={'synthetic-manual:p1'})
        self.assertNotIn('independent_context_review',good)
        self.assertEqual(good['expected_list_items'][0],catalog.entries[0].text)

class OwnedSelectionNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):useful.UsefulTableOwnedTests.setUpClass()
    @classmethod
    def tearDownClass(cls):useful.UsefulTableOwnedTests.tearDownClass()

    def test_actual_execution_supplies_enum_and_keeps_conditions_in_reviewed_answer(self):
        harness=useful.UsefulTableOwnedTests()
        with ExitStack() as cleanup:
            _,f=harness.fixture(cleanup)
            before=deepcopy(f.prepared)
            result=f.run()
            self.assertEqual(result['status'],'answered')
            schema=f.posts[0]['text']['format']['schema']
            self.assertEqual(schema['properties']['selection_scope_note']['enum'],[st.selection_notice('en')])
            self.assertEqual(len(f.posts),2)
            self.assertEqual(f.prepared,before)
            self.assertIn(useful.owned.TABLE_QUALIFIER,result['answer'])
            self.assertIn(st.selection_notice('en'),result['answer'])

    def test_provider_cannot_hide_unsupported_identity_or_condition_in_immutable_notice(self):
        harness=useful.UsefulTableOwnedTests()
        for note in (st.selection_notice('en')+' Applies to TEST-18.',
                     'Only when depressurized; selected information.'):
            with ExitStack() as cleanup:
                _,f=harness.fixture(cleanup,change=lambda reply:reply.update(selection_scope_note=note))
                result=f.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(len(f.posts),1)
                self.assertNotIn(note,result.get('answer',''))

    def test_neutral_notice_does_not_override_any_independent_review_check(self):
        harness=useful.UsefulTableOwnedTests()
        for check in ('source_fidelity','scope_applicability','provenance','completeness','safety','language'):
            with self.subTest(check=check),ExitStack() as cleanup:
                _,f=harness.fixture(cleanup,second_change=lambda reply:reply['checks'].update({check:False}))
                result=f.run()
                self.assertEqual(result['status'],'no_sources')
                self.assertEqual(len(f.posts),2)

class LosslessDraftViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):procedure.WiredSourceOrderedReviewTests.setUpClass()
    @classmethod
    def tearDownClass(cls):procedure.WiredSourceOrderedReviewTests.tearDownClass()

    def test_actual_review_packet_represents_draft_once_without_recovery_placeholder(self):
        result,calls,original=procedure.WiredSourceOrderedReviewTests().run_review(family_required=True,reply_mode='retain')
        self.assertEqual(len(calls),1)
        messages,kwargs=calls[0]
        user=messages[1]['content']; system=messages[0]['content']
        self.assertNotIn('CURRENT_ANSWER:\nRecover',user)
        self.assertIn('CURRENT_ANSWER_FORMAT: lossless PROCEDURE_BLOCKS',user)
        self.assertIn('Format labels are metadata',system)
        blocks=json.JSONDecoder().raw_decode(user.split('PROCEDURE_BLOCKS:\n',1)[1])[0]
        represented=''.join(b['heading']+''.join(unit[1] for unit in b['units'])+b['separator'] for b in blocks)
        self.assertEqual(represented.splitlines(),[line for line in original.splitlines() if line])
        self.assertEqual(result['answer'],represented)
        self.assertEqual(user.count('PROCEDURE_BLOCKS:\n'),1)
        self.assertEqual(kwargs['max_output_tokens'],3200)
        self.assertTrue(result['meta']['assistant_core_validation']['answer_contract']['passed'])

    def test_clear_draft_label_does_not_bypass_rejection_notes_or_boundary_checks(self):
        for args in ({'reject':True},{'note_mutation':'delete'},{'note_mutation':'swap'},
                     {'unknown_alias':True},{'proof_tamper':'closure'}):
            with self.subTest(args=args):
                result,calls,_=procedure.WiredSourceOrderedReviewTests().run_review(family_required=True,**args)
                self.assertEqual(len(calls),1)
                self.assertFalse(result['meta']['assistant_core_validation']['answer_contract']['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
