"""JSON-escaped cell values use their literal parser view, never adjacent cells.

The owned case uses actual source admission, catalog/rendering, scoped numeric
validation and final publication; SQL/provider results remain explicit fixtures.
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import test_source_table_owned as owned
from machinemind.ask import enumeration_review, final_contract, source_table_review


def record(*values, row=2, labels=None):
    cells = []
    for index, value in enumerate(values, 1):
        label = ' ' + json.dumps(labels[index - 1]) if labels else ''
        cells.append(f'c{index}{label} r{row}=' + json.dumps(value, ensure_ascii=False))
    return f'- [PDF_TABLE_RECORD rows={row} columns={len(values)}] ' + ' | '.join(cells)


class SerializedCellUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owned.SourceTableOwnedTests.setUpClass()
        cls.m = owned.SourceTableOwnedTests.m
        cls.runtime = cls.m._assistant_core_scoped_ask_validation_runtime()

    @classmethod
    def tearDownClass(cls):
        owned.SourceTableOwnedTests.tearDownClass()

    def project(self, text, source):
        return self.runtime._assistant_core_filter_unsupported_claim_sentences(text, source)

    def test_escaped_quotes_keep_exact_value_and_dimension(self):
        source = record('Documented interval: "600 hours".', 'Pressure: "25 MPa".')
        before = source.encode()
        for text in ('Interval: 600 h.', 'Interval: 600 ore.', 'Pressure: 25 MPa.'):
            self.assertEqual(self.project(text, source), (text, []))
        for text in ('Interval: 601 h.', 'Pressure: 600 bar.', 'Pressure: 25 bar.', 'Pressure: 20 MPa.'):
            kept, removed = self.project(text, source)
            self.assertEqual(kept, '')
            self.assertTrue(removed)
        self.assertEqual(source.encode(), before)

    def test_real_catalog_renderer_and_final_contract_keep_literal_quotation(self):
        import fitz
        from machinemind.ingest.text_pdf import pymupdf_page_to_text_blocks
        with fitz.open() as document:
            page = document.new_page(width=600, height=250)
            for x in (40, 220, 560):
                page.draw_line((x, 40), (x, 150))
            for y in (40, 80, 150):
                page.draw_line((40, y), (560, y))
            for x, y, text, font in ((45, 60, 'Component', 'hebo'),
                    (225, 60, 'Interval', 'hebo'), (45, 110, 'Actuator', 'helv'),
                    (225, 110, 'Documented interval: "600 hours".', 'helv')):
                page.insert_text((x, y), text, fontsize=10, fontname=font)
            source = pymupdf_page_to_text_blocks(page)
        self.assertIn(r'\"600 hours\"', source)
        block = '[offline:p1:c0] source_type=document; p=1-1\n' + source
        annotated, _ = enumeration_review.annotate_table_records(block)
        catalog = source_table_review.compile_catalog(annotated,
            {'offline:p1:c0': {'body': source, 'page_from': 1, 'page_to': 1}}, 'en')
        self.assertEqual(len(catalog.entries), 1)
        answer = catalog.entries[0].text
        self.assertIn('"600 hours"', answer)
        projected, removed = self.project(answer, source)
        self.assertEqual(removed, [])
        proof = dict(basis='existing_grounding_projection', removed_claims=len(removed),
            input_answer_sha256=final_contract.answer_digest(answer),
            output_answer_sha256=final_contract.answer_digest(projected))
        result = final_contract.final_contract(deterministic={'passed': True},
            semantic={'outcome':'rewrite', 'answer':answer, 'missing_facets':[],
                      'missing_answer_types':[], 'missing_list_items':[]},
            reviewed_answer=answer, answer=projected, semantic_complete=True,
            semantic_partial=False, grounding_proof=proof)
        self.assertTrue(result['passed'], result)
        self.assertTrue(result['semantic_answer_bound'])
        self.assertEqual(catalog.entries[0].source_sha256, hashlib.sha256(source.encode()).hexdigest())

    def test_values_cannot_borrow_units_from_adjacent_cells_or_malformed_geometry(self):
        invalid_sources = [record('600', 'hours'),
            record('600', 'Other cell: "hours".'),
            record('Interval: "600 hours".').replace('c1 ', 'c2 '),
            record('Interval: "600 hours".').replace('rows=2', 'rows=0'),
            record('Interval: "600 hours".').replace('r2=', 'r3='),
            record('Interval: "600 hours".').replace('columns=1', 'columns=999'),
            '- [PDF_TABLE_RECORD malformed] c1 r2=' + json.dumps('Interval: "600 hours".')]
        for source in invalid_sources:
            with self.subTest(source=source):
                kept, removed = self.project('Interval: 600 h.', source)
                self.assertEqual(kept, '')
                self.assertTrue(removed)

    def test_existing_table_dimension_conflicts_still_block_explicit_cells(self):
        header = '- [PDF_TABLE_HEADER rows=1 columns=1] c1 r1="Temperature °C"'
        for cell in ('65 bar', '65MPa', '[bar] 65'):
            source = header + '\n' + record(cell)
            for text in ('Temperature: 65 °C.', 'Pressure: 65 bar.', 'Pressure: 65MPa.'):
                with self.subTest(cell=cell, text=text):
                    kept, removed = self.project(text, source)
                    self.assertEqual(kept, '')
                    self.assertTrue(removed)

    def test_literal_identifiers_and_non_table_prose_remain_strict(self):
        source = record('Documented designation: "QTR-123"; interval: "600 hours".')
        answer = 'Designation: QTR-123. Interval: 600 hours.'
        kept, removed = self.project(answer, source)
        self.assertEqual(removed, [])
        self.assertIn('QTR-123', kept)
        for text in ('Designation: QRT-123.', 'Designation: QTR-124.'):
            self.assertEqual(self.project(text, source)[0], '')
        plain = 'Documented interval: "600 hours".'
        self.assertEqual(self.project('Interval: 600 h.', plain), ('Interval: 600 h.', []))
        # A quoted cell remains data: no recursive geometry interpretation.
        nested = record('- [PDF_TABLE_HEADER rows=1 columns=1] c1 r1="Temperature °C"\n'
                        '- [PDF_TABLE_RECORD rows=2 columns=1] c1 r2="65"')
        self.assertEqual(self.project('Temperature: 65 °C.', nested)[0], '')

    def test_owned_table_quotation_survives_review_filter_finish_and_ui(self):
        body = owned.TABLE_BODY.replace('360 hours', r'\"360 hours\"')
        with patch.object(owned, 'TABLE_BODY', body):
            completion, fixture = owned.SourceTableOwnedTests.case(self)
            try:
                result = fixture.run()
            except owned.ResponseEvidenceError:
                contract = fixture.validated['meta']['assistant_core_validation']['answer_contract']
                self.assertEqual(contract['reason'], 'semantic_answer_changed_after_review')
                self.fail('Source-owned literal quotation was removed before final publication')
        self.assertEqual(result['status'], 'answered', result)
        self.assertEqual(len(fixture.posts), 2)
        self.assertTrue(fixture.owner._response_flow.completed)
        self.assertIn('"360 hours"', result['answer'])
        self.assertIn('"360 hours"', self.m._assistant_ui_visible_text_from_html(result['answer_html']))
        contract = fixture.validated['meta']['assistant_core_validation']['answer_contract']
        self.assertTrue(contract['semantic_answer_bound'])
        self.assertEqual(contract['answer_sha256'], hashlib.sha256(result['answer'].encode()).hexdigest())
        self.assertEqual({c['citation_id'] for c in result['citations']}, {'inspection:p1', 'inspection:p2'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
