"""Offline generic detector ambiguity; no customer documents in fixtures."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitz
from machinemind.ingest import pdf_tables
from machinemind.ingest.text_pdf import pymupdf_page_to_text_blocks
from test_pdf_table_geometry import clean_and_chunk, page_with_grid


def table(bbox=(0, 0, 200, 100), *, complete=False, rows=2, cols=2):
    x0, y0, x1, y1 = bbox
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    cells = [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1)]
    if complete:
        cells.append((xm, ym, x1, y1))
    return SimpleNamespace(bbox=bbox, row_count=rows, col_count=cols,
                           cells=cells, header=None)


def words_at(items):
    return [(x, y, x + width, y + 8, text, i, 0, 0)
            for i, (x, y, width, text) in enumerate(items)]


def page_double(tables, words, blocks=None):
    if blocks is None:
        blocks = [(*w[:4], w[4], w[5], 0) for w in words]
    return SimpleNamespace(
        find_tables=lambda **kwargs: SimpleNamespace(tables=tables),
        get_text=lambda mode: {'words': words, 'blocks': blocks,
                               'dict': {'blocks': []}}[mode])


class PdfLayoutQuarantineTests(unittest.TestCase):
    def setUp(self):
        self.words = words_at([(5, 5, 35, 'PartA'), (105, 5, 35, 'ValueB'),
                              (5, 65, 35, 'PartC')])

    def test_incomplete_table_is_not_published_as_records_or_flat_prose(self):
        candidate = table()
        with self.assertRaisesRegex(pdf_tables.PdfTableLayoutError, 'unresolved_cells'):
            pdf_tables._table(candidate, self.words)
        text = pymupdf_page_to_text_blocks(page_double([candidate], self.words))
        self.assertNotIn('PDF_TABLE_RECORD', text)
        self.assertNotIn('candidate_rows=', text)
        for word in self.words:
            line = next(line for line in text.splitlines() if word[4] in line)
            self.assertIn('PDF_TABLE_UNASSIGNED relationship=ambiguous', line)
            self.assertIn('no row/column relationship or diagram connection is asserted', line)
            self.assertEqual(text.count(word[4]), 1)

    def test_detector_dimension_mismatch_does_not_invent_missing_cells(self):
        text = pymupdf_page_to_text_blocks(page_double([table(rows=4, cols=3)], self.words))
        self.assertEqual(text.count('PDF_TABLE_UNASSIGNED'), 3)
        self.assertNotIn('PDF_TABLE_RECORD', text)

    def test_valid_parent_cell_wins_over_contained_spurious_subgrid(self):
        parent = table(complete=True)
        child = table((10, 10, 80, 40))
        words = words_at([(15, 13, 12, 'Alpha'), (50, 13, 12, 'Beta'),
                          (15, 30, 12, 'Gamma'), (110, 10, 20, 'Right'),
                          (10, 65, 20, 'Below'), (110, 65, 20, 'Last')])
        expected = pymupdf_page_to_text_blocks(page_double([parent], words))
        for candidates in ([parent, child], [child, parent]):
            self.assertEqual(pymupdf_page_to_text_blocks(page_double(candidates, words)), expected)
        self.assertIn('PDF_TABLE_RECORD', expected)
        self.assertNotIn('PDF_TABLE_UNASSIGNED', expected)

    def test_subgrid_crossing_parent_cells_quarantines_entire_table(self):
        parent, child = table(complete=True), table((50, 0, 150, 100))
        words = words_at([(55, 5, 25, 'Alpha'), (105, 5, 25, 'Beta'),
                          (55, 65, 25, 'Gamma'), (160, 65, 25, 'Delta')])
        text = pymupdf_page_to_text_blocks(page_double([parent, child], words))
        self.assertNotIn('PDF_TABLE_RECORD', text)
        for word in words:
            self.assertEqual(text.count(word[4]), 1)
            self.assertIn('PDF_TABLE_UNASSIGNED', next(l for l in text.splitlines() if word[4] in l))

    def test_containment_cannot_hide_words_not_owned_by_parent(self):
        parent, child = table(complete=True), table((10, 10, 80, 40))
        words = words_at([(15, 13, 12, 'Alpha'), (50, 13, 12, 'Beta'),
                          (15, 30, 12, 'Gamma'), (110, 10, 20, 'Right'),
                          (10, 65, 20, 'Below'), (110, 65, 20, 'Last')])
        original = pdf_tables._table
        def missing_owned(candidate, *args):
            result = original(candidate, *args)
            if candidate is parent:
                return result[0], result[1], result[2] - {0}
            return result
        with patch.object(pdf_tables, '_table', side_effect=missing_owned):
            text = pymupdf_page_to_text_blocks(page_double([parent, child], words))
        self.assertNotIn('PDF_TABLE_RECORD', text)
        self.assertEqual(text.count('Alpha'), 1)

    def test_overlapping_uncertain_regions_preserve_words_once(self):
        first, second = table(), table((50, 0, 250, 100))
        words = self.words + words_at([(210, 5, 20, 'Outside')])
        words[-1] = (*words[-1][:5], 3, 0, 0)
        def uncertain(*args):
            raise pdf_tables.PdfTableLayoutError('table_grid_dimensions_ambiguous')
        with patch.object(pdf_tables, '_table', side_effect=uncertain):
            text = pymupdf_page_to_text_blocks(page_double([first, second], words))
        for word in words:
            self.assertEqual(text.count(word[4]), 1)

    def test_shared_cross_boundary_word_merges_nonoverlapping_regions(self):
        first, second = table((0, 0, 100, 100)), table((101, 0, 201, 100))
        words = words_at([(90, 5, 25, 'Crossing')])
        def uncertain(*args):
            raise pdf_tables.PdfTableLayoutError('table_grid_dimensions_ambiguous')
        with patch.object(pdf_tables, '_table', side_effect=uncertain):
            text = pymupdf_page_to_text_blocks(page_double([first, second], words))
        self.assertEqual(text.count('Crossing'), 1)
        self.assertIn('PDF_TABLE_UNASSIGNED', text)

    def test_unaffected_table_and_prose_preserved_beside_quarantine(self):
        good = table((300, 0, 500, 100), complete=True)
        words = self.words + words_at([(305, 5, 25, 'GoodA'), (405, 5, 25, 'GoodB'),
                                       (305, 65, 25, 'GoodC'), (405, 65, 25, 'GoodD'),
                                       (5, 140, 50, 'OrdinaryProse')])
        words = [(*w[:5], i, 0, 0) for i, w in enumerate(words)]
        text = pymupdf_page_to_text_blocks(page_double([table(), good], words))
        self.assertEqual(text.count('PDF_TABLE_RECORD'), 2)
        self.assertEqual(text.count('PDF_TABLE_UNASSIGNED'), 3)
        self.assertIn('\n\nOrdinaryProse', text)
        self.assertNotIn('PDF_TABLE_UNASSIGNED', next(l for l in text.splitlines() if 'GoodA' in l))

    def test_unknown_geometry_errors_and_runtime_errors_remain_fatal(self):
        for error in (pdf_tables.PdfTableLayoutError('new_unreviewed_error'), RuntimeError('broken')):
            with patch.object(pdf_tables, '_table', side_effect=error):
                with self.assertRaises(type(error)):
                    pymupdf_page_to_text_blocks(page_double([table()], self.words))

    def test_invalid_region_is_not_silently_quarantined(self):
        candidate = table()
        candidate.bbox = (float('nan'), 0, 200, 100)
        with self.assertRaisesRegex(pdf_tables.PdfTableLayoutError, 'rectangle_invalid'):
            pymupdf_page_to_text_blocks(page_double([candidate], self.words))

    def test_unmapped_text_still_fails_without_silent_loss(self):
        blocks = [(*w[:4], w[4] + ' UNMAPPED', w[5], 0) for w in self.words]
        with self.assertRaisesRegex(pdf_tables.PdfTableLayoutError, 'not_fully_mapped'):
            pymupdf_page_to_text_blocks(page_double([table()], self.words, blocks))

    def test_every_quarantined_line_keeps_warning_after_cleaning_chunking(self):
        text = pymupdf_page_to_text_blocks(page_double([table()], self.words))
        _, chunks = clean_and_chunk([text], target=60)
        for word in self.words:
            hits = [c['chunk_text'] for c in chunks if word[4] in c['chunk_text']]
            self.assertTrue(hits)
            for hit in hits:
                self.assertIn('PDF_TABLE_UNASSIGNED relationship=ambiguous', hit)
                self.assertIn('no row/column relationship or diagram connection is asserted', hit)

    def test_vector_panel_with_unresolved_detector_keeps_labels_unassigned(self):
        doc = fitz.open()
        page = doc.new_page()
        for rect, label in [((40, 40, 140, 100), 'Alpha'), ((140, 40, 240, 100), 'Beta'),
                            ((40, 100, 140, 160), 'Gamma')]:
            page.draw_rect(rect)
            page.insert_text((rect[0] + 5, rect[1] + 20), label)
        # Simulate a detector's incomplete partition, while using actual PDF
        # word/block extraction. Simple rectangles may be auto-completed by the
        # pinned detector even where no bottom-right border was drawn.
        candidate = table((40, 40, 240, 160))
        wrapped = SimpleNamespace(get_text=page.get_text,
                                  find_tables=lambda **kw: SimpleNamespace(tables=[candidate]))
        text = pymupdf_page_to_text_blocks(wrapped)
        self.assertEqual(text.count('PDF_TABLE_UNASSIGNED'), 3)
        self.assertNotIn('PDF_TABLE_RECORD', text)
        doc.close()

    def test_rotated_valid_grid_uses_same_text_coordinate_space(self):
        expected = None
        for rotation in (0, 90, 180, 270):
            doc, page = page_with_grid(merged=True)
            page.set_rotation(rotation)
            text = pymupdf_page_to_text_blocks(page)
            if expected is None:
                expected = text
            self.assertEqual(text, expected)
            self.assertEqual(page.rotation, rotation)
            self.assertIn('PDF_TABLE_RECORD rows=2-3', text)
            self.assertEqual(text.count('Daily'), 1)
            self.assertNotIn('PDF_TABLE_UNASSIGNED', text)
            doc.close()

    def test_failed_detection_restores_page_rotation(self):
        page = SimpleNamespace(rotation=90)
        page.set_rotation = lambda value: setattr(page, 'rotation', value)
        def broken(**kwargs):
            self.assertEqual(page.rotation, 0)
            raise RuntimeError('geometry failure')
        page.find_tables = broken
        with self.assertRaisesRegex(pdf_tables.PdfTableLayoutError, 'table_detection_failed'):
            pdf_tables.page_text_with_tables(page, [])
        self.assertEqual(page.rotation, 90)

    def test_equal_bbox_wordsets_do_not_hide_conflicting_grid_claims(self):
        first, second = table(complete=True), table(complete=True)
        owned = set(range(len(self.words)))
        conflicting = [(first.bbox, 'Claim A', owned), (second.bbox, 'Claim B', owned)]
        with patch.object(pdf_tables, '_table', side_effect=conflicting):
            with self.assertRaisesRegex(pdf_tables.PdfTableLayoutError, 'regions_overlap'):
                pymupdf_page_to_text_blocks(page_double([first, second], self.words))


if __name__ == '__main__':
    unittest.main()
