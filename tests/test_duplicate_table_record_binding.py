"""Only missing exact-table occurrences are accounted for, with mandatory audits.

Database observations and model verdicts are fixtures, never model quality proof.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import test_source_table_owned as owned
import test_source_table_review as pure
import test_duplicate_table_context as contexts
import test_factual_table_inventory as legacy
from context_review_wire_fixture import encode_context_reply
from machinemind.ask import source_table_review as st,enumeration_review as er,factual_context_review as review
from machinemind.ask.response_evidence import ResponseEvidenceError


class DuplicateTableRecordBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):owned.SourceTableOwnedTests.setUpClass();cls.m=owned.SourceTableOwnedTests.m
    @classmethod
    def tearDownClass(cls):owned.SourceTableOwnedTests.tearDownClass()
    case=owned.SourceTableOwnedTests.case

    def catalog(self,second=contexts.CELLS):
        helper=pure.SourceTableReviewTests()
        return helper.catalog([pure.row('doc:p1',contexts.SCOPE+'\n'+contexts.CELLS),
            pure.row('doc:p2',contexts.QUALIFIER+'\n'+second)])

    def decode(self,catalog,rows=(),opt=True,**extra):
        raw=dict(outcome='rewrite',answer='[[FACTUAL_INVENTORY]]',
            expected_list_items=[catalog.entries[0].token],citation_ids=['doc:p1'],
            table_record_coverage=list(rows),derive_duplicate_records=True)
        raw.update(extra);snapshot=deepcopy(raw)
        out=st.resolve_reply(raw,catalog,admitted_citation_ids={'doc:p1','doc:p2'},derive_duplicate_records=opt)
        self.assertEqual(raw,snapshot)
        checked=er.validate_table_coverage(out,tuple(rid for entry in catalog.entries for rid in entry.record_ids),
            wire_inventory=out['expected_list_items'])
        return raw,out,checked

    def test_only_missing_duplicates_are_derived_and_no_citation_is_granted(self):
        catalog=self.catalog()
        raw,out,checked=self.decode(catalog)
        self.assertEqual(checked['outcome'],'rewrite')
        self.assertEqual(out['citation_ids'],['doc:p1'])
        self.assertEqual(out['expected_list_items'],[catalog.entries[0].text])
        self.assertEqual(checked['table_record_coverage'][0]['inventory_items'],[0])
        self.assertEqual(checked['table_record_coverage'][1]['inventory_items'],[])
        explicit='T2!Different source context, reviewed separately'
        _,out,checked=self.decode(catalog,[explicit])
        self.assertIn(explicit,out['table_record_coverage'])
        self.assertEqual(checked['outcome'],'rewrite')
        _,_,checked=self.decode(catalog,opt=False)
        self.assertEqual(checked['outcome'],'no_sources')

    def test_explicit_conflicts_unknowns_and_foreign_owners_are_never_replaced(self):
        catalog=self.catalog()
        for rows in (['T1!excluded'],['T2=0'],['T2=1'],['T2!x','T2!x'],['T99!unknown'],['T2=00']):
            with self.subTest(rows=rows):
                self.assertEqual(self.decode(catalog,rows)[2]['outcome'],'no_sources')
        raw=dict(outcome='rewrite',answer='[[FACTUAL_INVENTORY]]',expected_list_items=[catalog.entries[0].token],
            citation_ids=['doc:p1'],table_record_coverage=[])
        out=st.resolve_reply(raw,catalog,admitted_citation_ids={'doc:p1'},derive_duplicate_records=True)
        self.assertEqual(out['outcome'],'no_sources')
        self.assertEqual(self.decode(catalog,citation_ids=['foreign:p1'])[2]['outcome'],'no_sources')

    def test_subset_changed_values_geometry_headers_and_order_do_not_count_as_duplicates(self):
        lines=contexts.CELLS.splitlines()
        variants=[lines[-1],contexts.CELLS.replace('Periodic','Annual'),
                  contexts.CELLS.replace('Oil temperature','Different heading'),
                  contexts.CELLS.replace('rows=2','rows=3').replace('r2=','r3='),
                  '\n'.join(reversed(lines))]
        for second in variants:
            with self.subTest(second=second):
                catalog=self.catalog(second)
                self.assertEqual(len(catalog.entries),2)
                self.assertNotEqual(catalog.entries[0].table_lines,catalog.entries[1].table_lines)
                self.assertEqual(self.decode(catalog)[2]['outcome'],'no_sources')

    def test_nonempty_draft_cannot_activate_opt_in_from_model_fields(self):
        helper=legacy.FactualTableInventoryTests();helper.m=self.m
        original=st.resolve_reply
        def injected(result,catalog,**kwargs):
            # An untrusted provider property is not the server-owned argument.
            result={**result,'derive_duplicate_records':True}
            return original(result,catalog,**kwargs)
        with patch.object(st,'resolve_reply',side_effect=injected)as observed:
            result,final,(messages,spec)=helper.case(items=['[[SOURCE_TABLE:G1]]'],
                coverage=['T1=0','T2=0'],outcome='rewrite',answer='[[FACTUAL_INVENTORY]]',
                draft='An already produced source-bound draft.',complete_catalog=True,expected_calls=1)
        self.assertEqual(result['outcome'],'rewrite');self.assertTrue(final['passed'])
        self.assertIs(observed.call_args.kwargs['derive_duplicate_records'],False)
        self.assertNotIn('byte-identical occurrences',spec['json_schema']['schema']['properties']['table_record_coverage']['description'])

    def test_owned_missing_duplicates_require_all_contexts_before_html_and_final_hash(self):
        for omit_context in (False,True):
            with self.subTest(omit_context=omit_context):
                with patch.object(owned,'TABLE_BODY',contexts.SCOPE+'\n'+contexts.CELLS),patch.object(owned,'CONDITION',contexts.QUALIFIER+'\n'+contexts.CELLS):
                    owner,f=self.case(change=lambda r:r.update(expected_list_items=['[[SOURCE_TABLE:G1]]'],
                        citation_ids=['inspection:p1'],table_record_coverage=[]))
                original=f.owner._response_flow.execution._v13_json_models
                def provider(messages,**kw):
                    out,model=original(messages,**kw)
                    if kw['json_schema']['name']=='factual_context_review':
                        audits=[dict(target='item:0',context=contexts.SCOPE,documentary_limits_checked=True,
                            source_quotes=[dict(citation_id='inspection:p1',quote=contexts.SCOPE)])]
                        if not omit_context:
                            audits.append(dict(target='item:0:context:1',context=contexts.QUALIFIER,documentary_limits_checked=True,
                                source_quotes=[dict(citation_id='inspection:p2',quote=contexts.QUALIFIER)]))
                        reply=dict(outcome='rewrite',edits=[],citation_ids=['inspection:p1','inspection:p2'],table_contexts=audits,
                            checks={k:True for k in review.CHECKS},reason='accepted')
                        first_sources=f.reviewer_calls[0][0][1]['content'].split('SOURCES:\n',1)[1].split('\n\nCURRENT_ANSWER:',1)[0]
                        out=encode_context_reply(reply,messages,original_sources=first_sources)
                    return out,model
                f.owner._response_flow.execution=replace(f.owner._response_flow.execution,_v13_json_models=provider)
                if omit_context:
                    with self.assertRaises(ResponseEvidenceError):f.run()
                else:
                    result=f.run();self.assertEqual(result['status'],'answered')
                    self.assertEqual(f.wire_reply['table_record_coverage'],[])
                    self.assertEqual(f.wire_reply['citation_ids'],['inspection:p1'])
                    for text in (contexts.SCOPE.rstrip('.'),contexts.QUALIFIER.rstrip('.')):
                        self.assertIn(text,result['answer']);self.assertIn(text,result['answer_html'])
                    contract=result['meta']['assistant_core_validation']['answer_contract']
                    self.assertTrue(contract['semantic_answer_bound'])
                    self.assertEqual(contract['answer_sha256'],hashlib.sha256(result['answer'].encode()).hexdigest())
                    self.assertEqual([r['inventory_items']for r in contract['semantic_verifier']['table_record_coverage']],[[0],[]])
                self.assertEqual(len(f.posts),2);self.assertEqual(f.writer_calls,[])
                self.assertEqual(f.budget_ids,[id(owner.budget)]*2)
                f.close();owner.close()


if __name__=='__main__':unittest.main(verbosity=2)
