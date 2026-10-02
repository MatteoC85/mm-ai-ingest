"""Offline regression for chunker SECTION wrappers leaking into Step notes."""
import ast
from copy import deepcopy
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.ingest import pdf_cleaning_chunking as chunker
from machinemind.presentation import citations
from machinemind.retrieval import source_parsing as parser
from machinemind.ask.final_contract import source_safety_notes


RAW_RUNTIME = parser.ProcedureUiRawTextRuntime()
CLEAN_RUNTIME = parser.ProcedureUiCleanRuntime(_normalize_unicode_advanced=lambda value: value, re=re)


def fields(citation):
    return parser.procedure_ui_fields(citation, runtime=parser.ProcedureUiFieldsRuntime(
        _normalize_unicode_advanced=lambda value: value, re=re,
        _parse_structured_source_fields=lambda text: citations.parse_structured_source_fields(
            text, clean_text_fn=citations.clean_display_text),
        _procedure_ui_raw_text=lambda row: parser.procedure_ui_raw_text(row, runtime=RAW_RUNTIME)))


def sections(value):
    return parser.procedure_ui_sections(value, runtime=parser.ProcedureUiSectionsRuntime(
        re=re, _procedure_ui_clean=lambda text: parser.procedure_ui_clean(text, runtime=CLEAN_RUNTIME)))


def production_collapsed_step(raw):
    chunks = chunker.chunk_sentences_with_pages([(1, raw)], 250, 50, 30,
        split_sentences_fn=chunker.split_sentences_conservative,
        looks_like_section_header_fn=chunker.looks_like_section_header)
    # Execute only the existing pure composition-root function, without importing
    # main or initializing its database/provider/credential dependencies.
    path = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_collapse_structured_chunks')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_collapse_structured_chunks'](chunks)[0]['chunk_text']


class StepSectionWrapperTests(unittest.TestCase):
    def test_real_ingestion_shape_preserves_complete_note_without_wrapper_suffix(self):
        raw = ('SOURCE_TYPE: step\nSTEP_NUMBER: 1\nTITLE: Arrestare il ciclo\n'
               'DESCRIPTION: PROCEDURA: PROC-X\nAZIONE OPERATIVA\nApplicare la procedura di isolamento.\n'
               'NOTA DI SICUREZZA\nPericolo di schiacciamento, taglio e proiezione del capo nastro.\n'
               'RIFERIMENTI TECNICI\nManuale pag. 93.\n')
        ingested = production_collapsed_step(raw)
        self.assertIn('SECTION: RIFERIMENTI TECNICI\nRIFERIMENTI TECNICI', ingested)
        citation = {'citation_id': 'step:synthetic:p1-1:c1', 'source_type': 'step', 'chunk_full': ingested}
        before = deepcopy(citation)
        parsed = fields(citation)
        parts = sections(parsed['description'])
        self.assertEqual(parts['safety'], 'Pericolo di schiacciamento, taglio e proiezione del capo nastro.')
        self.assertEqual(parts['technical_sources'], 'Manuale pag. 93.')
        notes = source_safety_notes([citation], fields=fields, sections=sections, source_type=lambda row: row['source_type'])
        self.assertEqual(notes[0]['texts'], (parts['safety'],))
        self.assertEqual(citation, before)

    def test_original_headings_numbers_negation_and_multiline_content_survive(self):
        raw = ('SOURCE_TYPE: step\nSTEP_NUMBER: 5\nTITLE: Prova\nDESCRIPTION: AZIONE OPERATIVA\n'
               'Non superare 1,5 mm.\nConservare il valore 12 mA, non 12 MA.\n'
               'SECTION: NOTA DI SICUREZZA\nNOTA DI SICUREZZA\nNon disabilitare il controllo.\n'
               'SECTION: RIFERIMENTI TECNICI\nRIFERIMENTI TECNICI\nManuale p. 21.\n')
        parsed = fields({'chunk_full': raw})
        self.assertIn('NOTA DI SICUREZZA', parsed['description'])
        self.assertIn('RIFERIMENTI TECNICI', parsed['description'])
        parts = sections(parsed['description'])
        self.assertEqual(parts['instruction'], 'Non superare 1,5 mm. Conservare il valore 12 mA, non 12 MA.')
        self.assertEqual(parts['safety'], 'Non disabilitare il controllo.')
        self.assertEqual(parsed['step_number'], '5')

    def test_english_labels_work_without_a_language_specific_wrapper_rule(self):
        raw = ('SOURCE_TYPE: step\nDESCRIPTION: OPERATIONAL ACTION\nCheck the indication.\n'
               'SECTION: SAFETY NOTE\nSAFETY NOTE\nDo not bypass the guard.\n'
               'SECTION: TECHNICAL REFERENCES\nTECHNICAL REFERENCES\nManual p. 21.')
        parts = sections(fields({'chunk_full': raw})['description'])
        self.assertEqual(parts['safety'], 'Do not bypass the guard.')
        self.assertEqual(parts['technical_sources'], 'Manual p. 21.')

    def test_no_global_stripping_of_inline_or_unmatched_section_text(self):
        for body in ('The indicator reads SECTION: CLOSED; preserve this label.',
                     'SECTION: SAFETY DETAILS\nDifferent following heading\nDo not remove this.',
                     'SECTION:',
                     'SECTION: Safety note\nSAFETY NOTE\nCase difference is not exact duplication.'):
            with self.subTest(body=body):
                raw = 'SOURCE_TYPE: step\nDESCRIPTION: ' + body
                parsed = fields({'chunk_full': raw})
                self.assertIn('SECTION:', parsed['description'])

    def test_document_or_untyped_text_is_not_reinterpreted_as_ingested_step(self):
        body = 'DESCRIPTION: Note text.\nSECTION: HEADING\nHEADING\nAnother source paragraph.'
        for raw in (body, 'SOURCE_TYPE: document\n' + body, 'SOURCE_TYPE: procedure\n' + body):
            with self.subTest(raw=raw):
                self.assertIn('SECTION: HEADING', fields({'chunk_full': raw})['description'])

    def test_plain_registered_step_is_unchanged(self):
        raw = ('SOURCE_TYPE: step\nSTEP_NUMBER: 8\nTITLE: Validare\nDESCRIPTION: AZIONE OPERATIVA\n'
               'Monitorare almeno 30 cicli.\nNOTA DI SICUREZZA\nEscalare se ritorna.\n'
               'RIFERIMENTI TECNICI\nManuale HMI p. 20-24.')
        parsed = fields({'chunk_full': raw})
        self.assertEqual(parsed['step_number'], '8')
        self.assertEqual(sections(parsed['description'])['safety'], 'Escalare se ritorna.')


if __name__ == '__main__':
    unittest.main()
