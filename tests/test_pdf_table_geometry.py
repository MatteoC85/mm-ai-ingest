"""Offline synthetic PDF geometry through the production cleaning/chunking path."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fitz
from machinemind.ingest import pdf_cleaning_chunking as cleaning
from machinemind.ingest.pdf_tables import PdfTableLayoutError, _table
from machinemind.ingest.text_pdf import (
    extract_pages_with_layout_blocks, normalize_text_keep_lines,
    normalize_unicode_advanced, pymupdf_page_to_text_blocks,
)


def legacy(page):
    return '\n\n'.join(str(b[4]).strip() for b in sorted(page.get_text('blocks'), key=lambda b:(b[1],b[0])) if str(b[4]).strip())


def page_with_grid(*, merged=False, header=True, outside=True):
    doc=fitz.open();page=doc.new_page(width=600,height=500)
    xs=[40,220,330,560];ys=[80,125,190,265]
    for x in xs:page.draw_line((x,ys[0]),(x,ys[-1]))
    for y in (ys[0],ys[1],ys[-1]):page.draw_line((xs[0],y),(xs[-1],y))
    page.draw_line((xs[0],ys[2]),(xs[1] if merged else xs[-1],ys[2]))
    rows=[['Component','Frequency','Condition'],['Guide inspection','Daily','Check clean. See page 12 before work.'],['Remove contaminants','Every 250 h','Inspect condition. Lubricate only if needed.']]
    for r,row in enumerate(rows):
        for c,text in enumerate(row):
            if merged and r==2 and c>0:continue
            if merged and r==1 and c==2:text='Inspect every 250 h. Lubricate only if needed.'
            rect=fitz.Rect(xs[c]+5,ys[r]+5,xs[c+1]-5,ys[r+1]-5)
            if merged and r==1 and c>0:rect.y1=ys[-1]-5
            assert page.insert_textbox(rect,text,fontsize=10,fontname='hebo' if header and r==0 else 'helv')>=0
    if outside:
        page.insert_text((40,50),'Before access isolate all energy.',fontsize=10)
        page.insert_text((40,300),'After completion restore the protection.',fontsize=10)
    return doc,page


def clean_and_chunk(pages,target=190):
    norm=partial(cleaning.hf_norm_line,normalize_text_keep_lines=normalize_text_keep_lines)
    strip=partial(cleaning.strip_page_noise_prefix,normalize_unicode_advanced=normalize_unicode_advanced)
    noise=partial(cleaning.is_page_noise_line,hf_norm_line_fn=norm)
    header,footer=cleaning.detect_repeated_headers_footers(pages,
        extract_top_bottom_lines_fn=cleaning.extract_top_bottom_lines,hf_norm_line_fn=norm)
    result=[]
    for p,text in enumerate(pages,1):
        text=cleaning.remove_headers_footers_from_page(text,header,footer,
            strip_page_noise_prefix_fn=strip,is_page_noise_line_fn=noise,hf_norm_line_fn=norm)
        text=cleaning.reflow_paragraphs_conservative(text,looks_like_bullet_fn=cleaning.looks_like_bullet,
            looks_like_table_fn=cleaning.looks_like_table,looks_like_title_fn=cleaning.looks_like_title)
        text=cleaning.maybe_remove_toc(text,looks_like_toc_line_fn=cleaning.looks_like_toc_line,
            strip_toc_lines_fn=partial(cleaning.strip_toc_lines,looks_like_toc_line_fn=cleaning.looks_like_toc_line))
        result.append((p,text))
    chunks=cleaning.chunk_sentences_with_pages(result,target,40,60,
        split_sentences_fn=cleaning.split_sentences_conservative,
        looks_like_section_header_fn=cleaning.looks_like_section_header)
    for chunk in chunks:
        chunk['chunk_text']=cleaning.strip_hf_from_chunk_text(chunk['chunk_text'],header,footer,
            hf_norm_line_fn=norm,strip_page_noise_prefix_fn=strip,is_page_noise_line_fn=noise)
    return result,chunks


class PdfTableGeometryTests(unittest.TestCase):
    def test_cells_keep_component_interval_and_conditions_in_atomic_chunks(self):
        doc,page=page_with_grid()
        text=pymupdf_page_to_text_blocks(page)
        self.assertIn('[PDF_TABLE_HEADER',text)
        record=next(l for l in text.splitlines() if 'Guide inspection' in l)
        self.assertIn('"Frequency" r2="Daily"',record)
        self.assertIn('"Condition" r2="Check clean. See page 12 before work."',record)
        cleaned,chunks=clean_and_chunk([text,text,text])
        for _,value in cleaned:self.assertIn(record,value)
        self.assertTrue(any(record in c['chunk_text'] for c in chunks))
        self.assertEqual(text.count('Before access isolate all energy.'),1)
        self.assertEqual(text.count('After completion restore the protection.'),1)
        self.assertTrue(all('page 12' in c['chunk_text'] for c in chunks if 'Guide inspection' in c['chunk_text']))
        doc.close()

    def test_rowspan_preserves_all_original_cells_once_without_fill_down(self):
        doc,page=page_with_grid(merged=True)
        text=pymupdf_page_to_text_blocks(page)
        self.assertIn('[PDF_TABLE_RECORD rows=2-3',text)
        record=next(l for l in text.splitlines() if '[PDF_TABLE_RECORD rows=2-3' in l)
        self.assertIn('r2-3="Daily"',record)
        self.assertIn('r2-3="Inspect every 250 h. Lubricate only if needed."',record)
        self.assertIn('r3="Remove contaminants"',record)
        self.assertEqual(text.count('Daily'),1)
        self.assertEqual(text.count('Remove contaminants'),1)
        _,chunks=clean_and_chunk([text],target=100)
        self.assertTrue(any(record in c['chunk_text'] for c in chunks))
        doc.close()

    def test_first_row_is_not_silently_treated_as_header(self):
        doc,page=page_with_grid(header=False)
        text=pymupdf_page_to_text_blocks(page)
        self.assertNotIn('[PDF_TABLE_HEADER',text)
        self.assertIn('[PDF_TABLE_RECORD rows=1',text)
        self.assertEqual(text.count('Frequency'),1)
        doc.close()

    def test_colspan_preserves_geometry_without_invented_subcells(self):
        table=SimpleNamespace(bbox=(0,0,200,90),row_count=3,col_count=2,
            cells=[(0,0,200,30),(0,30,100,60),(100,30,200,60),(0,60,100,90),(100,60,200,90)],header=None)
        words=[(5,5,100,15,'Section',0,0,0),(5,35,70,45,'Valve',1,0,0),
               (105,35,170,45,'Monthly',2,0,0),(5,65,70,75,'Seal',3,0,0),(105,65,170,75,'Daily',4,0,0)]
        _,text,_=_table(table,words)
        self.assertIn('c1-2 r1="Section"',text)
        self.assertEqual(text.count('Section'),1)
        self.assertNotIn('HEADER',text)

    def test_cross_boundary_words_remain_unassigned_not_attached_to_row(self):
        table=SimpleNamespace(bbox=(0,0,200,100),row_count=2,col_count=2,
            cells=[(0,0,100,50),(100,0,200,50),(0,50,100,100),(100,50,200,100)],header=None)
        words=[(5,5,50,15,'Valve',0,0,0),(105,5,150,15,'Daily',1,0,0),
               (5,65,50,75,'Seal',2,0,0),(105,65,150,75,'Monthly',3,0,0),
               (105,45,160,65,'Uncertain',4,0,0)]
        _,text,owned=_table(table,words)
        line=next(l for l in text.splitlines() if 'Uncertain' in l)
        self.assertIn('PDF_TABLE_UNASSIGNED',line)
        self.assertIn('candidate_rows=1,2',line)
        self.assertEqual(text.count('Uncertain'),1)
        self.assertEqual(owned,set(range(5)))
        _,chunks=clean_and_chunk([text])
        self.assertTrue(any(line in c['chunk_text'] for c in chunks))

    def test_ambiguous_true_grid_fails_closed(self):
        table=SimpleNamespace(bbox=(0,0,200,100),row_count=2,col_count=2,
            cells=[(0,0,100,50),(100,0,200,50),(0,50,100,100)],header=None)
        words=[(5,5,50,15,'Valve',0,0,0),(105,5,150,15,'Daily',1,0,0),(5,65,50,75,'Seal',2,0,0)]
        with self.assertRaisesRegex(PdfTableLayoutError,'unresolved_cells'):_table(table,words)

    def test_nested_page_frame_and_picture_panel_gaps_are_not_data_tables(self):
        words=[(5,5,50,15,'Valve',0,0,0),(105,5,150,15,'Daily',1,0,0),(5,65,50,75,'Drawing',2,0,0)]
        frame=SimpleNamespace(bbox=(0,0,200,100),row_count=2,col_count=2,
            cells=[(0,0,200,100),(0,0,100,50),(100,0,200,50),(0,50,100,100)],header=None)
        self.assertIsNone(_table(frame,words))
        panel=SimpleNamespace(bbox=(0,0,200,100),row_count=2,col_count=2,
            cells=[(0,0,100,50),(100,0,200,50),(0,50,100,100)],header=None)
        self.assertIsNone(_table(panel,words,images=[(10,55,90,95)]))
        # A missing cell containing text is still ambiguous, even near an image.
        with self.assertRaisesRegex(PdfTableLayoutError,'unresolved_cells'):
            _table(panel,words+[(105,65,150,75,'Required',3,0,0)],images=[(10,55,90,95)])

    def test_api_missing_legacy_and_real_detector_failure_explicit(self):
        blocks=[(1,2,20,30,'Legacy text.',0,0)]
        page=SimpleNamespace(get_text=lambda mode:blocks)
        self.assertEqual(pymupdf_page_to_text_blocks(page),'Legacy text.')
        def broken(**kwargs):raise RuntimeError('bad geometry')
        page.find_tables=broken
        with self.assertRaisesRegex(PdfTableLayoutError,'table_detection_failed'):
            pymupdf_page_to_text_blocks(page)

    def test_prose_warning_frame_and_graph_grid_retain_legacy_bytes(self):
        doc=fitz.open();page=doc.new_page()
        page.insert_text((30,50),'Ordinary prose before a warning.')
        for x in (30,80,400):page.draw_line((x,80),(x,220))
        for y in (80,120,160,220):page.draw_line((30,y),(400,y))
        for y,text in [(100,'Stop equipment.'),(140,'Isolate energy.'),(190,'Use trained personnel.')]:
            page.insert_text((100,y),text)
        self.assertEqual(pymupdf_page_to_text_blocks(page),legacy(page))
        doc.close()
        doc=fitz.open();page=doc.new_page()
        for x in (30,100,170,240):page.draw_line((x,80),(x,220))
        for y in (80,120,160,220):page.draw_line((30,y),(240,y))
        for x,value in [(40,'0.0'),(110,'0.5'),(180,'1.0')]:page.insert_text((x,210),value)
        self.assertEqual(pymupdf_page_to_text_blocks(page),legacy(page))
        doc.close()

    def test_extraction_entrypoint_preserves_page_count_and_no_table_legacy(self):
        doc,page=page_with_grid(merged=True)
        page=doc.new_page();page.insert_text((30,50),'No table here. Preserve this text.')
        expected=normalize_text_keep_lines(legacy(page))
        pages=extract_pages_with_layout_blocks(doc.tobytes(),fitz_module=fitz)
        self.assertEqual(len(pages),2)
        self.assertIn('PDF_TABLE_RECORD',pages[0])
        self.assertEqual(pages[1],expected)
        doc.close()


if __name__=='__main__':unittest.main()
