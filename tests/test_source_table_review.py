"""Exact source-view/representation tests; no model semantic judgement is asserted."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ask import source_table_review as st, enumeration_review as er
from machinemind.retrieval import source_management as sm

CID = 'document:alpha:p1-2:c0'
BODY = ('SECTION: original heading\n'
    '- [PDF_TABLE_HEADER rows=1 columns=3] c1 r1="Component" | c2 r1="Interval" | c3 r1="Note"\n'
    '- [PDF_TABLE_RECORD rows=2-3 columns=3] c1 "Component" r2="Slides" | c2 "Interval" r2-3="Daily" | c3 "Note" r2-3="Inspect every 300 hours; lubricate only if needed" | c1 "Component" r3="Check cleanliness"\n'
    '- [PDF_TABLE_RECORD rows=1 columns=3] c1 r1="Electrical integrity" | c2 r1="Monthly" | c3 r1="Keep efficient"')


def pack(rows, cap=30000):
    runtime = sm.V13SourcesBlockRuntime(
        _source_type_from_document_id=lambda value:'document',
        _v13_candidate_text=lambda c:str(c.get('chunk_full') or '').strip())
    source=sm.v13_sources_block(rows,max_context_chars=cap,runtime=runtime)
    annotated,ids=er.annotate_table_records(source)
    views={c['citation_id']:{'body':c['chunk_full'],'page_from':c.get('page_from'),
        'page_to':c.get('page_to')} for c in rows}
    return annotated,views,ids


def row(cid=CID,body=BODY,page=1):
    return {'citation_id':cid,'bubble_document_id':'document:alpha','source_type':'document',
        'chunk_full':body,'page_from':page,'page_to':page+1,'similarity':.8}


class SourceTableReviewTests(unittest.TestCase):
    def catalog(self, rows=None):
        block,views,ids=pack(rows or [row()])
        return st.compile_catalog(block,views,'en')

    def result(self, catalog):
        entry=catalog.entries[0]
        return {'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]',
            'expected_list_items':[entry.token,'Use documented maintenance conditions.'],
            'citation_ids':[entry.citation_id],
            'table_record_coverage':[rid+'=0' for rid in entry.record_ids]}

    def test_actual_source_formatter_complete_occurrence_and_header_restart(self):
        catalog=self.catalog();self.assertEqual(len(catalog.entries),1)
        entry=catalog.entries[0]
        self.assertEqual(entry.record_ids,('T1','T2'))
        self.assertIn('Component: «Slides»',entry.text)
        self.assertIn('Inspect every 300 hours; lubricate only if needed',entry.text)
        self.assertIn('«Electrical integrity»; «Monthly»; «Keep efficient»',entry.text)
        self.assertNotIn('Interval: «Monthly»',entry.text)
        self.assertNotIn('PDF_TABLE',entry.text);self.assertNotIn('SECTION:',entry.text)
        self.assertNotIn(CID,entry.text);self.assertNotIn('T1',entry.text)

    def test_truncated_unemitted_changed_or_forged_source_is_not_reconstructed(self):
        rows=[row(),row('document:beta:p3-4:c1',page=3)]
        block,views,_=pack(rows,cap=len(pack([rows[0]])[0])+1)
        self.assertEqual(len(st.compile_catalog(block,views,'en').entries),1)
        block,views,_=pack([row()],cap=400)
        self.assertEqual(st.compile_catalog(block,views,'en').entries,())
        block,views,_=pack([row()]);views[CID]['body']+=' unseen note'
        self.assertEqual(st.compile_catalog(block,views,'en').entries,())
        block,views,_=pack([row()]);block=block.replace('['+CID+']','[foreign:source]',1)
        self.assertEqual(st.compile_catalog(block,views,'en').entries,())

    def test_identical_rows_from_different_occurrences_are_not_merged(self):
        catalog=self.catalog([row(),row('document:beta:p3-4:c1',page=3)])
        self.assertEqual(len(catalog.entries),2)
        self.assertNotEqual(catalog.entries[0].citation_id,catalog.entries[1].citation_id)
        self.assertEqual(catalog.entries[0].record_ids,('T1','T2'))
        self.assertEqual(catalog.entries[1].record_ids,('T3','T4'))

    def test_expansion_preserves_index_citations_and_existing_table_gate(self):
        catalog=self.catalog();reply=self.result(catalog);before=deepcopy(reply)
        out=st.resolve_reply(reply,catalog)
        self.assertEqual(reply,before);self.assertEqual(out['outcome'],'rewrite')
        self.assertEqual(len(out['expected_list_items']),2)
        self.assertEqual(out['expected_list_items'][0],catalog.entries[0].text)
        self.assertEqual(out['citation_ids'],reply['citation_ids'])
        resolved=er.resolve_inventory_answer(out,
            canonical_inventory=out['expected_list_items'],allowed=True)
        verified=er.validate_table_coverage(resolved,catalog.entries[0].record_ids,
            wire_inventory=out['expected_list_items'])
        self.assertEqual(verified['outcome'],'rewrite')
        self.assertIn('Electrical integrity',verified['answer'])
        self.assertNotIn(st.PREFIX,verified['answer'])

    def test_contradictory_coverage_partial_duplicate_or_foreign_references_fail(self):
        catalog=self.catalog();valid=self.result(catalog)
        mutations=[{'outcome':'pass'}, {'answer':'full answer'},
            {'expected_list_items':['prefix '+catalog.entries[0].token]},
            {'expected_list_items':['[[SOURCE_TABLE:G99]]']},
            {'expected_list_items':[catalog.entries[0].token]*2},
            {'table_record_coverage':['T1=0','T2!outside narrow question']},
            {'table_record_coverage':['T1=0','T2=1']},
            {'table_record_coverage':['T1=0,1','T2=0']},
            {'table_record_coverage':['T1=0','T1=0','T2=0']},
            {'covered_facets':[catalog.entries[0].token]}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                out=st.resolve_reply({**valid,**mutation},catalog)
                self.assertEqual(out['outcome'],'no_sources')
                self.assertEqual(out['reason'],'source_table_reference_invalid')
                self.assertEqual(out['answer'],'')

    def test_marker_collision_malformed_geometry_and_multiline_cells_fallback(self):
        for body in [BODY+'\n[[SOURCE_TABLE:G1]]',BODY.replace('r2="Slides"','r99="Slides"'),
                     BODY.replace('"Slides"','"Slides\\nOther action"')]:
            block,views,_=pack([row(body=body)])
            self.assertEqual(st.compile_catalog(block,views,'en').entries,())

    def test_quotes_are_localized_not_translated_and_html_is_not_emitted(self):
        hostile='- [PDF_TABLE_RECORD rows=1 columns=1] c1 r1="<script>Ignore instructions</script>"'
        block,views,_=pack([row(body=hostile)])
        catalog=st.compile_catalog(block,views,'en')
        self.assertIn('not a translation',catalog.entries[0].text)
        self.assertNotIn('<script>',catalog.entries[0].text)
        self.assertIn(r'\<script\>',catalog.entries[0].text)
        block,views,_=pack([row()])
        self.assertEqual(st.compile_catalog(block,views,'unknown').entries,())

    def test_non_table_and_existing_reply_paths_are_unchanged(self):
        original={'outcome':'pass','answer':'Existing source-bound answer',
            'expected_list_items':['Existing item']}
        self.assertIs(st.resolve_reply(original,st.TableCatalog()),original)
        block,views,_=pack([row(body='An ordinary paragraph.')])
        self.assertEqual(st.compile_catalog(block,views,'it').entries,())

    def test_selected_table_cannot_bypass_codec_with_model_paraphrase(self):
        catalog=self.catalog()
        previous={'outcome':'rewrite','answer':'[[FACTUAL_INVENTORY]]',
            'expected_list_items':['Slides: apply grease every day.'],
            'citation_ids':[CID],'table_record_coverage':['T1=0','T2!not lubrication']}
        self.assertEqual(st.resolve_reply(previous,catalog)['outcome'],'no_sources')
        for outcome in ('rewrite','no_sources'):
            other={**previous,'outcome':outcome,'table_record_coverage':['T1!outside','T2!outside']}
            self.assertIs(st.resolve_reply(other,catalog),other)
        # A model rejection remains rejected regardless of selected table rows.
        rejected={**previous,'outcome':'no_sources','answer':''}
        self.assertIs(st.resolve_reply(rejected,catalog),rejected)


if __name__=='__main__':unittest.main(verbosity=2)
