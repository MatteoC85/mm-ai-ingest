"""Presentation-only invariants: canonical words, numbers, safety and scope stay put."""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.presentation import responses as ui


def runtime():
    return SimpleNamespace(
        normalize_unicode=lambda value: unicodedata.normalize('NFKD', str(value)),
        procedure_ui_clean=lambda value: str(value).strip(),
        procedure_ui_complete_excerpt=lambda value, **kwargs: str(value).strip(),
        safe_int=lambda value, default=0: int(value) if str(value).isdigit() else default,
        strip_inline_citation_markers_for_display=lambda value: value,
        evidence_role=lambda row: row.get('evidence_role') or row.get('source_type', ''),
        source_type_from_document_id=lambda value: str(value).split(':')[0],
        structured_source_types={'procedure', 'step', 'ps'},
        unique_non_empty_strings=lambda values, limit: list(dict.fromkeys(str(x).strip() for x in values if str(x).strip()))[:limit],
        assistant_ui_max_html_chars=100000,
        assistant_ui_render_version='offline-root-render',
        assistant_ask_ui_render_version='offline-ask-render',
        ask_ui_structured_max_links=30, ask_ui_max_links=30,
        ask_ui_structured_max_citations=30, ask_ui_max_citations=30,
    )


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return ''.join(c.text() if isinstance(c, Node) else c for c in self.children)

    def find(self, **attrs):
        found = [self] if all(self.attrs.get(k) == v for k, v in attrs.items()) else []
        for child in self.children:
            if isinstance(child, Node):
                found.extend(child.find(**attrs))
        return found


class Tree(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.root = Node('root')
        self.stack = [self.root]
        self.feed(value)
        if len(self.stack) != 1:
            raise AssertionError('unclosed markup')

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'br', 'hr', 'img', 'input', 'meta', 'link'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if self.stack[-1].tag != tag:
            raise AssertionError(f'unbalanced {tag}')
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)


class ResponseVisualHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.runtime = runtime()

    def render(self, text, *, language='it', selected=False):
        return ui.assistant_ui_lossless_html(text, response_language=language,
            source_procedure_selected=selected, _runtime=self.runtime)

    def fixture(self):
        text = ('Preparazione della linea\nDescrizione generale completa.\nPrima di iniziare\n'
                'Personale qualificato: tecnico formato.\nLivello di sicurezza: CRITICO.\nProcedura\n'
                '1. Arrestare il ciclo\nApplicare l’isolamento prescritto.\n'
                'Attenzione: STOP non equivale a isolamento delle energie.\n'
                '5. Inserire il materiale\nSeguire il percorso documentato.\n'
                'Attenzione: Tenere le mani lontane dagli organi mobili.\n'
                '6. Ripristinare la guida\nRichiudere la guida prima del riavvio.\n'
                'Supporto dal manuale\n- Conservare l’ordine delle operazioni.\n'
                'La chiusura deve precedere il riavvio.')
        return dict(status='answered', effective_mode='ask', answer=text,
                    citations=[{'bubble_document_id': 'procedure:synthetic', 'source_type': 'procedure'},
                               {'bubble_document_id': 'step:synthetic', 'source_type': 'step'}],
                    rg_links=[{'bubble_document_id': 'procedure:synthetic', 'source_type': 'procedure',
                               'url': 'https://example.invalid/procedure'}],
                    meta={'procedure_source_selection': {'version': 'admitted-procedure-segment-v1',
                          'selected_step_numbers': [5, 6]},
                          'assistant_core_validation': {'answer_contract': {'complete': True, 'sequence_complete': True}},
                          'request_authority': {'scope': 'synthetic-only'}})

    def test_real_finalizer_keeps_canonical_contract_citations_and_input_unchanged(self):
        before = self.fixture()
        untouched = deepcopy(before)
        result = ui.assistant_ui_finalize_response(before, language='it', preserve_complete=True, _runtime=self.runtime)
        self.assertEqual(before, untouched)
        for key in ('answer', 'citations', 'rg_links', 'status', 'effective_mode'):
            self.assertEqual(result[key], before[key])
        for key in before['meta']:
            self.assertEqual(result['meta'][key], before['meta'][key])
        self.assertTrue(result['meta']['answer_html_renderer_fallback_used'])
        self.assertEqual(result['meta']['answer_html_token_coverage'], 1.0)
        tree = Tree(result['answer_html']).root
        self.assertEqual(len(tree.find(**{'data-mm-ui-note': 'source-step-selection'})), 1)
        self.assertEqual(len(tree.find(**{'data-mm-procedure-title': 'canonical'})), 1)

    def test_step_owns_its_title_body_and_safety_before_the_next_step(self):
        tree = Tree(self.render(self.fixture()['answer'], selected=True)).root
        first = tree.find(**{'data-mm-list-number': '1'})[0]
        fifth = tree.find(**{'data-mm-list-number': '5'})[0]
        self.assertIn('Applicare l’isolamento prescritto.', first.text())
        self.assertIn('STOP non equivale a isolamento', first.text())
        self.assertNotIn('Inserire il materiale', first.text())
        self.assertIn('Tenere le mani lontane', fifth.text())
        self.assertEqual([n.attrs['data-mm-list-number'] for n in tree.find(role='listitem')], ['1', '5', '6'])
        self.assertEqual([n.tag for n in fifth.children[1].children if isinstance(n, Node)][0], 'h4')
        self.assertIn('font-size:16px', fifth.children[1].children[0].attrs['style'])

    def test_generic_nonfallback_also_preserves_numbers_with_qualified_note(self):
        fixture = self.fixture()
        fixture['answer'] = 'Procedure\n1. Isolate the line\n5. Restore the guide'
        result = ui.assistant_ui_finalize_response(fixture, language='en', _runtime=self.runtime)
        self.assertFalse(result['meta']['answer_html_renderer_fallback_used'])
        self.assertEqual(result['answer'], fixture['answer'])
        tree = Tree(result['answer_html']).root
        self.assertEqual(len(tree.find(**{'data-mm-ui-note': 'source-step-selection'})), 1)
        items = tree.find(role='listitem')
        self.assertEqual([n.attrs['data-mm-list-number'] for n in items], ['1', '5'])
        self.assertTrue(all('font-size:16px' in n.children[1].attrs['style'] for n in items))

    def test_rich_procedure_nonfallback_has_same_note_and_title_hierarchy(self):
        fixture = self.fixture()
        model = {'kind': 'procedure', 'title': 'Guide preparation', 'summary': 'Documented instructions.',
                 'steps': [{'display_number': 1, 'title': 'Isolate the line', 'instruction': 'Apply the documented isolation.', 'safety': 'STOP is not isolation.'},
                           {'display_number': 5, 'title': 'Restore the guide', 'instruction': 'Restore before restart.', 'safety': 'Keep hands clear.'}]}
        fixture['answer'] = ui.procedure_ui_model_to_text(model, response_language='en', preserve_complete=True, _runtime=self.runtime)
        fixture['_assistant_ui_model'] = model
        result = ui.assistant_ui_finalize_response(fixture, language='en', preserve_complete=True, _runtime=self.runtime)
        self.assertFalse(result['meta']['answer_html_renderer_fallback_used'])
        self.assertEqual(result['answer'], fixture['answer'])
        tree = Tree(result['answer_html']).root
        self.assertEqual(len(tree.find(**{'data-mm-ui-note': 'source-step-selection'})), 1)
        self.assertEqual([n.attrs['data-mm-list-number'] for n in tree.find(role='listitem')], ['1', '5'])
        headings = [n for n in tree.find() if n.tag == 'h4']
        self.assertEqual([n.text() for n in headings], ['Isolate the line', 'Restore the guide'])
        self.assertTrue(all('font-size:16px' in n.attrs['style'] for n in headings))

    def test_direct_rich_and_generic_rendering_cannot_infer_source_selection(self):
        model = {'kind': 'procedure', 'title': 'A procedure', 'steps': [
            {'display_number': 1, 'title': 'First'}, {'display_number': 5, 'title': 'Fifth'}]}
        rendered = ui.procedure_ui_model_to_html(model, links=[], response_language='en', _runtime=self.runtime)
        self.assertNotIn('data-mm-ui-note', rendered)
        rendered = ui.assistant_ui_generic_html('1. First\n5. Fifth', links=[], response_language='en', _runtime=self.runtime)
        self.assertNotIn('data-mm-ui-note', rendered)

    def test_large_source_number_is_preserved_without_range_sized_allocation(self):
        # A source ordinal must not control the amount of memory used by the
        # presentation-only gap note. No cap, renumbering or truncation is added.
        large_number = 10 ** 12
        model = {'kind': 'procedure', 'title': 'Source numbering', 'steps': [
            {'display_number': 1, 'title': 'First'},
            {'display_number': large_number, 'title': 'Original later step'}]}
        rendered = ui.procedure_ui_model_to_html(model, links=[], response_language='en',
            source_procedure_selected=True, _runtime=self.runtime)
        tree = Tree(rendered).root
        self.assertEqual([n.attrs['data-mm-list-number'] for n in tree.find(role='listitem')],
                         ['1', str(large_number)])
        self.assertEqual(len(tree.find(**{'data-mm-ui-note': 'source-step-selection'})), 1)
        self.assertLess(len(rendered), 10000)

    def test_generic_complete_labels_become_h3_without_keyword_inference(self):
        text = 'Before starting\nQualified personnel: trained operator.\nManual support\nProcedure calibration requires a check.'
        rendered = ui.assistant_ui_generic_html(text, links=[], response_language='en', _runtime=self.runtime)
        tree = Tree(rendered).root
        self.assertEqual([n.text() for n in tree.find() if n.tag == 'h3'], ['Before starting', 'Manual support'])
        self.assertIn('Procedure calibration requires a check.', tree.text())

    def test_known_sections_are_headings_and_manual_support_is_outside_steps(self):
        tree = Tree(self.render(self.fixture()['answer'], selected=True)).root
        sections = [n for n in tree.find() if n.tag == 'section']
        # find() without attributes traverses every node.
        headings = [n.text() for n in tree.find() if n.tag == 'h3']
        self.assertIn('Prima di iniziare', headings)
        self.assertIn('Procedura', headings)
        self.assertIn('Supporto dal manuale', headings)
        procedure = next(n for n in sections if n.children[0].text() == 'Procedura')
        self.assertNotIn('Supporto dal manuale', procedure.text())
        self.assertNotIn('La chiusura deve precedere', procedure.text())

    def test_english_equivalent_retains_source_numbers_and_safety(self):
        text = ('Line preparation\nBefore starting\nQualified personnel: trained technician.\n'
                'Procedure\n2. Open the guide\nUse the documented isolation procedure.\n'
                'Warning: Keep the machine isolated.\n\n5. Close the guide\n'
                'Restore the original position.\nManual support\nCheck the original procedure.')
        rendered = self.render(text, language='en', selected=True)
        tree = Tree(rendered).root
        self.assertIn('Selected steps from the original procedure.', tree.text())
        self.assertEqual([n.attrs['data-mm-list-number'] for n in tree.find(role='listitem')], ['2', '5'])
        self.assertEqual(len(tree.find(**{'data-mm-safety': 'explicit-label'})), 1)
        self.assertIn('Keep the machine isolated.', tree.find(**{'data-mm-list-number': '2'})[0].text())

    def test_gap_note_requires_attested_procedure_context_and_actual_gap(self):
        for selected, text in ((False, '1. First\n5. Fifth'), (True, '1. First\n2. Second')):
            self.assertNotIn('data-mm-ui-note', self.render(text, selected=selected))
        for change in ('no_sources', 'unknown_selection'):
            fixture = self.fixture()
            if change == 'no_sources':
                fixture['citations'] = []
            else:
                fixture['meta']['procedure_source_selection']['version'] = 'unverified'
            result = ui.assistant_ui_finalize_response(fixture, language='it', _runtime=self.runtime)
            self.assertNotIn('data-mm-ui-note', result.get('answer_html', ''))

    def test_decimal_measurements_are_not_step_numbers_or_rewritten(self):
        text = '1.25 bar\n0.75 l/min\n1,25 bar\n100 psi\nProcedure calibration requires a check.'
        tree = Tree(self.render(text, language='en')).root
        self.assertEqual(tree.find(role='listitem'), [])
        for line in text.splitlines():
            self.assertIn(line, tree.text())
        self.assertEqual([n.text() for n in tree.find() if n.tag == 'h3'], [])

    def test_long_paragraphs_and_line_order_are_never_truncated(self):
        long_text = 'A complete conditional instruction. ' * 200 + 'Only after the final condition.'
        text = long_text + '\nProcedure\n1. Original title\n' + long_text + '\nWarning: Retain the closing condition.'
        tree = Tree(self.render(text, language='en', selected=True)).root
        self.assertEqual(tree.text().count(long_text), 2)
        self.assertEqual(tree.find(**{'data-mm-procedure-title': 'canonical'}), [])
        self.assertIn('Retain the closing condition.', tree.find(**{'data-mm-list-number': '1'})[0].text())

    def test_html_and_quoted_attributes_are_escaped_in_every_visual_slot(self):
        text = '**<img src=x onerror=alert(1)>**\n1. <script>alert(2)</script>\nWarning: <svg onload=alert(3)> & "quoted"'
        rendered = self.render(text, language='en')
        tree = Tree(rendered).root
        self.assertFalse(any(n.tag in {'script', 'img', 'svg'} for n in tree.find()))
        self.assertIn('<script>alert(2)</script>', tree.text())
        self.assertIn('<svg onload=alert(3)> & "quoted"', tree.text())

    def test_safety_is_only_explicit_label_not_keyword_in_prose(self):
        text = '1. Check warning indicators\nThe warning indicator remains unknown.\n**Warning:** Do not bypass protection.'
        tree = Tree(self.render(text, language='en')).root
        self.assertEqual(len(tree.find(**{'data-mm-safety': 'explicit-label'})), 1)
        self.assertIn('Warning: Do not bypass protection.', tree.find(**{'data-mm-safety': 'explicit-label'})[0].text())

    def test_blank_lines_crlf_and_bullets_remain_in_the_original_step(self):
        text = 'Procedure\r\n1. Title\r\n\r\nInstruction.\r\n- Original subpoint\r\nWarning: Closing condition.\r\n2. Second\r\nSecond instruction.'
        tree = Tree(self.render(text, language='en')).root
        first = tree.find(**{'data-mm-list-number': '1'})[0]
        self.assertIn('Instruction.', first.text())
        self.assertIn('Original subpoint', first.text())
        self.assertIn('Closing condition.', first.text())
        self.assertNotIn('Second instruction.', first.text())

    def test_root_preserves_newlines_and_all_diagnostic_fields(self):
        response = dict(status='answered', problem_summary='Unconfirmed observation.',
                        possible_causes=[dict(rank=1, cause='Possible documented cause',
                            why='First statement\nA condition remains unknown.',
                            checks=['Check the original observation.\nDo not treat unknown as false.'])],
                        recommended_next_checks=['Additional check.\nKeep the safety condition.'])
        untouched = deepcopy(response)
        rendered = ui.assistant_ui_root_cause_html(response, response_language='en', _runtime=self.runtime)
        self.assertEqual(response, untouched)
        tree = Tree(rendered).root
        for value in (response['possible_causes'][0]['why'], response['possible_causes'][0]['checks'][0], response['recommended_next_checks'][0]):
            self.assertIn(value, tree.text())
            matching = [n for n in tree.find() if n.tag in {'p', 'li'} and value in n.text()]
            self.assertTrue(matching)
            self.assertTrue(all('white-space:pre-line' in n.attrs.get('style', '') for n in matching))


if __name__ == '__main__':
    unittest.main(verbosity=2)
