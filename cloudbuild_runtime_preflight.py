"""Cloud Build runtime preflight for MachineMind.

This file is intentionally stored in the repository instead of embedding a
large Python program inside one Cloud Build argument. Google Cloud Build
limits every individual `args` value to 10,000 characters.
"""
import json
import os
import urllib.request

def get(path):
    with urllib.request.urlopen(
        'http://127.0.0.1:8080' + path,
        timeout=10,
    ) as response:
        return json.load(response)

ping = get('/ping')
version = get('/version')
openapi = get('/openapi.json')

assert ping.get('ok') is True, ping
assert version.get('ok') is True, version
assert (
    version.get('electrical_code_marker')
    == os.environ['EXPECTED_CODE_MARKER']
), version
assert (
    version.get('electrical_structured_pipeline_marker')
    == os.environ['EXPECTED_CODE_MARKER']
), version
assert version.get('electrical_structured_detector_model') == os.environ['EXPECTED_STRUCTURED_MODEL'], version
assert version.get('electrical_structured_extractor_model') == os.environ['EXPECTED_STRUCTURED_MODEL'], version
assert version.get('electrical_structured_verifier_model') == os.environ['EXPECTED_STRUCTURED_MODEL'], version
assert (
    version.get('electrical_structured_detector_prompt_version')
    == os.environ['EXPECTED_DETECTOR_PROMPT']
), version
assert (
    version.get('electrical_structured_extractor_prompt_version')
    == os.environ['EXPECTED_EXTRACTOR_PROMPT']
), version
assert (
    version.get('electrical_structured_verifier_prompt_version')
    == os.environ['EXPECTED_VERIFIER_PROMPT']
), version
assert (
    version.get('electrical_structured_materializer_version')
    == os.environ['EXPECTED_MATERIALIZER_VERSION']
), version

assert version.get('electrical_terminals_enabled') is True, version
assert (
    version.get('electrical_terminals_pipeline_marker')
    == os.environ['EXPECTED_TERMINAL_MARKER']
), version
assert (
    version.get('electrical_terminals_detector_model')
    == os.environ['EXPECTED_TERMINAL_MODEL']
), version
assert (
    version.get('electrical_terminals_extractor_model')
    == os.environ['EXPECTED_TERMINAL_MODEL']
), version
assert (
    version.get('electrical_terminals_verifier_model')
    == os.environ['EXPECTED_TERMINAL_MODEL']
), version
assert (
    version.get('electrical_terminals_detector_prompt_version')
    == os.environ['EXPECTED_TERMINAL_DETECTOR_PROMPT']
), version
assert (
    version.get('electrical_terminals_extractor_prompt_version')
    == os.environ['EXPECTED_TERMINAL_EXTRACTOR_PROMPT']
), version
assert (
    version.get('electrical_terminals_verifier_prompt_version')
    == os.environ['EXPECTED_TERMINAL_VERIFIER_PROMPT']
), version
assert (
    version.get('electrical_terminals_materializer_version')
    == os.environ['EXPECTED_TERMINAL_MATERIALIZER']
), version

assert version.get('electrical_bom_enabled') is True, version
assert (
    version.get('electrical_bom_pipeline_marker')
    == os.environ['EXPECTED_BOM_MARKER']
), version
assert (
    version.get('electrical_bom_detector_model')
    == os.environ['EXPECTED_BOM_MODEL']
), version
assert (
    version.get('electrical_bom_extractor_model')
    == os.environ['EXPECTED_BOM_MODEL']
), version
assert (
    version.get('electrical_bom_verifier_model')
    == os.environ['EXPECTED_BOM_MODEL']
), version
assert (
    version.get('electrical_bom_detector_prompt_version')
    == os.environ['EXPECTED_BOM_DETECTOR_PROMPT']
), version
assert (
    version.get('electrical_bom_extractor_prompt_version')
    == os.environ['EXPECTED_BOM_EXTRACTOR_PROMPT']
), version
assert (
    version.get('electrical_bom_verifier_prompt_version')
    == os.environ['EXPECTED_BOM_VERIFIER_PROMPT']
), version
assert (
    version.get('electrical_bom_materializer_version')
    == os.environ['EXPECTED_BOM_MATERIALIZER']
), version

from electrical_bom import get_electrical_bom_runtime_config
bom_config = get_electrical_bom_runtime_config()
assert bom_config['enabled'] is True, bom_config
assert (
    bom_config['pipeline_marker']
    == os.environ['EXPECTED_BOM_MARKER']
), bom_config

assert version.get('electrical_graph_enabled') is True, version
assert (
    version.get('electrical_graph_pipeline_marker')
    == os.environ['EXPECTED_GRAPH_MARKER']
), version
assert (
    version.get('electrical_graph_detector_model')
    == os.environ['EXPECTED_GRAPH_MODEL']
), version
assert (
    version.get('electrical_graph_extractor_model')
    == os.environ['EXPECTED_GRAPH_MODEL']
), version
assert (
    version.get('electrical_graph_verifier_model')
    == os.environ['EXPECTED_GRAPH_MODEL']
), version
assert (
    version.get('electrical_graph_detector_prompt_version')
    == os.environ['EXPECTED_GRAPH_DETECTOR_PROMPT']
), version
assert (
    version.get('electrical_graph_extractor_prompt_version')
    == os.environ['EXPECTED_GRAPH_EXTRACTOR_PROMPT']
), version
assert (
    version.get('electrical_graph_verifier_prompt_version')
    == os.environ['EXPECTED_GRAPH_VERIFIER_PROMPT']
), version
assert (
    version.get('electrical_graph_materializer_version')
    == os.environ['EXPECTED_GRAPH_MATERIALIZER']
), version

from electrical_graph import get_electrical_graph_runtime_config
graph_config = get_electrical_graph_runtime_config()
assert graph_config['enabled'] is True, graph_config
assert (
    graph_config['pipeline_marker']
    == os.environ['EXPECTED_GRAPH_MARKER']
), graph_config

from electrical_terminals import get_electrical_terminal_runtime_config
terminal_config = get_electrical_terminal_runtime_config()
assert terminal_config['enabled'] is True, terminal_config
assert (
    terminal_config['pipeline_marker']
    == os.environ['EXPECTED_TERMINAL_MARKER']
), terminal_config

assert version.get('electrical_source_snapshot_enabled') is True, version
assert (
    version.get('electrical_source_snapshot_bucket')
    == 'mm-ai-electrical-sources-443517556116'
), version

from electrical_structured import (
    _adjudicate_io_context,
    _adjudicate_row_semantics,
    _looks_artificially_fragmented_text,
    _text_row_accounting,
)

safety_context = _adjudicate_io_context(
    page_type='safety_io_table',
    model_io_type='digital_input',
    model_is_safety=False,
)
assert safety_context['final_io_type'] == 'safety_input', safety_context
assert safety_context['final_is_safety'] is True, safety_context

output_context = _adjudicate_io_context(
    page_type='safety_io_table',
    model_io_type='mixed',
    model_is_safety=True,
    verified_page_io_type='safety_output',
    verified_page_is_safety=True,
    verified_page_confidence=0.98,
    verified_region_io_type='safety_output',
    verified_region_is_safety=True,
    verified_region_confidence=0.97,
)
assert output_context['final_io_type'] == 'safety_output', output_context
assert output_context['direction_conflict'] is False, output_context

conflict_context = _adjudicate_io_context(
    page_type='safety_io_table',
    model_io_type='safety_input',
    model_is_safety=True,
    verified_page_io_type='safety_output',
    verified_page_is_safety=True,
    verified_page_confidence=0.98,
    verified_region_io_type='safety_input',
    verified_region_is_safety=True,
    verified_region_confidence=0.97,
)
assert conflict_context['direction_conflict'] is True, conflict_context

marker_accounting = _text_row_accounting(
    signal_name='***',
    description='***',
    expected_normal_state='',
)
assert marker_accounting['requires_text_review'] is False, marker_accounting
assert set(marker_accounting['marker_only_text_fields']) == {
    'signal_name', 'description'
}, marker_accounting

substantive_accounting = _text_row_accounting(
    signal_name='SPARE 1',
    description='SPARE 1',
    expected_normal_state='',
)
assert substantive_accounting['requires_text_review'] is True, substantive_accounting

placeholder_semantics = _adjudicate_row_semantics(
    model_row_role='other_data',
    model_is_placeholder=False,
    channel_ref='2',
    connector_ref='X1',
    plc_address='',
    wire_reference='**',
    terminal_reference='',
    signal_name='***',
    description='***',
    expected_normal_state='',
)
assert placeholder_semantics['final_row_role'] == 'placeholder', placeholder_semantics
assert placeholder_semantics['is_placeholder'] is True, placeholder_semantics

blank_semantics = _adjudicate_row_semantics(
    model_row_role='placeholder',
    model_is_placeholder=True,
    channel_ref='A',
    connector_ref='X2',
    plc_address='',
    wire_reference='',
    terminal_reference='',
    signal_name='',
    description='',
    expected_normal_state='',
)
assert blank_semantics['final_row_role'] == 'blank_unused', blank_semantics
assert blank_semantics['is_placeholder'] is False, blank_semantics

assert _looks_artificially_fragmented_text(
    'SBL OCC O EL ETT ROSER. AN TERI ORE'
) is True
assert _looks_artificially_fragmented_text(
    'MICRO SERIE PROTEZIONE MACCHINA'
) is False

from electrical_terminals import (
    _adjudicate_uncovered_regions,
    _apply_field_support_decisions,
    _apply_overrides,
    _decision_lookup,
    _field_collision_issues,
    _unrepresented_source_evidence,
    _validate_page,
    _verifier_issue_is_field_assignment_related,
)

ambiguous_terminal = {
    'row_id': 'r8',
    'source_slot_ids': ['S008'],
    'confidence': 0.95,
    'wire_number_original': '3L+',
    'potential_original': '3L+',
    'cable_reference_original': '',
    'conductor_color_original': '',
    'conductor_cross_section_original': '',
    'side_a_origin_original': '',
    'side_b_destination_original': '',
    'side_a_description_original': '',
    'side_b_description_original': '',
}
field_decision = {
    'region_id': 'R1',
    'row_id': 'r8',
    'source_text_original': '3L+',
    'source_slot_ids': ['S008'],
    'visual_occurrence_count': 1,
    'supported_fields': ['wire_number_original'],
    'unsupported_fields': ['potential_original'],
    'shared_semantics_explicitly_supported': False,
    'confidence': 0.98,
    'reason': 'Single visible cell belongs to one physical field lane.',
}
terminal_extractions = [{
    'region_id': 'R1',
    'terminals': [ambiguous_terminal],
}]
_apply_field_support_decisions(
    terminal_extractions,
    [field_decision],
)
assert ambiguous_terminal['wire_number_original'] == '3L+', ambiguous_terminal
assert ambiguous_terminal['potential_original'] == '', ambiguous_terminal
assert (
    ambiguous_terminal['verifier_field_support_decisions'][0]['applied']
    is True
), ambiguous_terminal

decision_lookup = _decision_lookup({
    'field_support_decisions': [field_decision],
})
assert _field_collision_issues(
    terminal=ambiguous_terminal,
    region_id='R1',
    row_id='r8',
    verifier_decisions=decision_lookup,
) == []

unresolved_terminal = dict(ambiguous_terminal)
unresolved_terminal['potential_original'] = '3L+'
unresolved_terminal.pop('verifier_field_support_decisions', None)
unresolved_terminal.pop('verifier_overrides', None)
unresolved_issues = _field_collision_issues(
    terminal=unresolved_terminal,
    region_id='R1',
    row_id='r8',
    verifier_decisions={},
)
assert len(unresolved_issues) == 1, unresolved_issues
assert unresolved_issues[0]['severity'] == 'high', unresolved_issues
assert _verifier_issue_is_field_assignment_related({
    'issue_type': 'wrong_field_assignment',
}) is True
assert _verifier_issue_is_field_assignment_related({
    'issue_type': 'terminal_number_not_visually_supported',
}) is False

# Phase 2T V1.2: strip-like empty layout grids are classified and
# audited, but they do not become missing data-bearing strips.
coverage_detector = {
    'missing_visible_strips': [
        'upper strip-like empty grid',
        'lower strip-like empty grid',
    ],
}
coverage_verifier = {
    'all_strip_like_regions_classified': True,
    'all_data_terminal_strips_accounted_for': True,
    'unaccounted_data_terminal_strip_regions': [],
    'confidence': 0.98,
    'uncovered_region_adjudications': [
        {
            'region_ref': 'upper strip-like empty grid',
            'visual_region_kind': 'auxiliary_description_grid',
            'is_data_terminal_strip': False,
            'has_strip_tag': False,
            'has_terminal_number_axis': False,
            'has_numbered_terminal_rows': False,
            'has_connection_semantics': False,
            'accounted': True,
            'confidence': 0.98,
            'reason': 'Empty auxiliary layout without terminal data.',
        },
        {
            'region_ref': 'lower strip-like empty grid',
            'visual_region_kind': 'auxiliary_description_grid',
            'is_data_terminal_strip': False,
            'has_strip_tag': False,
            'has_terminal_number_axis': False,
            'has_numbered_terminal_rows': False,
            'has_connection_semantics': False,
            'accounted': True,
            'confidence': 0.97,
            'reason': 'Empty auxiliary layout without terminal data.',
        },
    ],
}
region_audit, coverage_issues = _adjudicate_uncovered_regions(
    detector=coverage_detector,
    verifier=coverage_verifier,
)
assert len(region_audit) == 2, region_audit
assert not [
    issue
    for issue in coverage_issues
    if issue.get('severity') in {'high', 'critical'}
], coverage_issues

# A real uncovered strip must still block.
blocking_verifier = {
    **coverage_verifier,
    'all_data_terminal_strips_accounted_for': False,
    'unaccounted_data_terminal_strip_regions': ['uncovered data strip'],
    'uncovered_region_adjudications': [
        {
            'region_ref': 'upper strip-like empty grid',
            'visual_region_kind': 'terminal_strip',
            'is_data_terminal_strip': True,
            'has_strip_tag': True,
            'has_terminal_number_axis': True,
            'has_numbered_terminal_rows': True,
            'has_connection_semantics': True,
            'accounted': True,
            'confidence': 0.98,
            'reason': 'Actual numbered terminal strip.',
        },
        coverage_verifier['uncovered_region_adjudications'][1],
    ],
}
_, blocking_coverage_issues = _adjudicate_uncovered_regions(
    detector=coverage_detector,
    verifier=blocking_verifier,
)
assert [
    issue
    for issue in blocking_coverage_issues
    if issue.get('severity') in {'high', 'critical'}
], blocking_coverage_issues

# Positive and negative overrides must move exact visible text to its
# physically supported field without losing it.
spare_row = {
    'row_id': 'r24',
    'wire_number_original': '',
    'side_b_description_original': 'RESERVE',
}
spare_extractions = [{
    'region_id': 'R1',
    'terminals': [spare_row],
}]
_apply_overrides(
    spare_extractions,
    [
        {
            'region_id': 'R1',
            'row_id': 'r24',
            'field_name': 'wire_number_original',
            'approved_text': 'RESERVE',
            'confidence': 0.98,
            'reason': 'Visible text is printed in the wire lane.',
        },
        {
            'region_id': 'R1',
            'row_id': 'r24',
            'field_name': 'side_b_description_original',
            'approved_text': '',
            'confidence': 0.98,
            'reason': 'No text is printed in the destination lane.',
        },
    ],
)
assert spare_row['wire_number_original'] == 'RESERVE', spare_row
assert spare_row['side_b_description_original'] == '', spare_row

# Deterministic source-evidence coverage blocks silent text loss.
word_map = {
    1: {'id': 1, 'text': '24'},
    2: {'id': 2, 'text': 'RESERVE'},
}
evidence_row = {
    'source_word_ids': [1, 2],
    'terminal_number_original': '24',
    'wire_number_original': '',
}
missing_evidence = _unrepresented_source_evidence(
    evidence_row,
    word_map,
)
assert [
    item['text_original'] for item in missing_evidence
] == ['RESERVE'], missing_evidence
evidence_row['wire_number_original'] = 'RESERVE'
assert _unrepresented_source_evidence(
    evidence_row,
    word_map,
) == []

# Phase 2T V1.2.1: a verifier can correctly return review_required for the
# pre-correction row and also return the exact positive/negative corrections.
# The final gate must revalidate the corrected row, not blindly reuse the
# pre-correction verdict. A missing positive reassignment must still block.
def _post_override_fixture_extractions():
    return [{
        'region_id': 'R1',
        'strip_tag_original': 'X1',
        'source_side_label_original': '',
        'destination_side_label_original': '',
        'boundary_rows': [],
        'confidence': 0.99,
        'issues': [],
        'terminals': [{
            'row_id': 'r1',
            'visual_order': 1,
            'row_role': 'spare_terminal',
            'terminal_number_original': '1',
            'level_ref_original': '',
            'side_a_origin_original': '',
            'side_b_destination_original': '',
            'wire_number_original': '',
            'cable_reference_original': '',
            'potential_original': '',
            'conductor_color_original': '',
            'conductor_cross_section_original': '',
            'side_a_description_original': '',
            'side_b_description_original': 'RESERVE',
            'source_slot_ids': ['S001'],
            'source_word_ids': [1, 2],
            'bbox_pt': [0, 0, 1, 1],
            'confidence': 0.99,
            'evidence_notes': '',
        }],
    }]

post_override_proposals = [{
    'region_id': 'R1',
    'slot_candidates': [{'slot_id': 'S001'}],
}]
post_override_detector = {
    'missing_visible_strips': [],
    'issues': [],
    'regions': [{
        'region_id': 'R1',
        'is_terminal_strip': True,
        'strip_tag_original': 'X1',
        'expected_terminal_rows': 1,
        'expected_boundary_rows': 0,
        'visible_number_sequence': ['1'],
        'confidence': 0.99,
    }],
}
post_override_word_map = {
    1: {'id': 1, 'text': '1'},
    2: {'id': 2, 'text': 'RESERVE'},
}
post_override_corrections = [
    {
        'region_id': 'R1',
        'row_id': 'r1',
        'field_name': 'wire_number_original',
        'approved_text': 'RESERVE',
        'confidence': 0.99,
        'reason': 'Visible text belongs to the wire lane.',
    },
    {
        'region_id': 'R1',
        'row_id': 'r1',
        'field_name': 'side_b_description_original',
        'approved_text': '',
        'confidence': 0.99,
        'reason': 'No text is visible in the description lane.',
    },
]
post_override_verifier = {
    'verdict': 'review_required',
    'all_visible_strips_accounted_for': True,
    'all_strip_like_regions_classified': True,
    'all_data_terminal_strips_accounted_for': True,
    'all_visible_terminal_rows_accounted_for': True,
    'all_strip_tags_supported_by_headers': True,
    'all_terminal_numbers_visually_supported': True,
    'all_published_fields_visually_supported': False,
    'uncovered_region_adjudications': [],
    'unaccounted_data_terminal_strip_regions': [],
    'field_support_decisions': [],
    'field_overrides': post_override_corrections,
    'region_checks': [{
        'region_id': 'R1',
        'strip_tag_original': 'X1',
        'expected_terminal_rows': 1,
        'extracted_terminal_rows': 1,
        'expected_terminal_number_sequence': ['1'],
        'verified_terminal_number_sequence': ['1'],
        'boundary_rows_accounted_for': True,
        'pass': True,
        'confidence': 0.99,
        'notes': 'The row is complete after the returned corrections.',
    }],
    'issues': [{
        'issue_type': 'wrong_field_assignment',
        'severity': 'high',
        'message': 'Visible text was assigned to the wrong field.',
        'region_id': 'R1',
        'row_ids': ['r1'],
        'confidence': 0.99,
    }],
    'confidence': 0.99,
}

post_override_extractions = _post_override_fixture_extractions()
_apply_overrides(
    post_override_extractions,
    post_override_corrections,
)
post_override_passed, post_override_rows, post_override_issues = (
    _validate_page(
        page={'id': 1},
        proposals=post_override_proposals,
        detector=post_override_detector,
        extractions=post_override_extractions,
        verifier=post_override_verifier,
        word_map=post_override_word_map,
    )
)
assert post_override_passed is True, post_override_issues
assert post_override_rows[0]['wire_number_original'] == 'RESERVE'
assert post_override_rows[0]['side_b_description_original'] == ''
assert not [
    issue
    for issue in post_override_issues
    if issue.get('severity') in {'high', 'critical'}
], post_override_issues
assert any(
    issue.get('issue_type')
    == 'terminal-verifier-verdict-superseded-post-override'
    for issue in post_override_issues
), post_override_issues

missing_positive_extractions = _post_override_fixture_extractions()
missing_positive_corrections = [post_override_corrections[1]]
_apply_overrides(
    missing_positive_extractions,
    missing_positive_corrections,
)
missing_positive_verifier = {
    **post_override_verifier,
    'field_overrides': missing_positive_corrections,
}
missing_positive_passed, _, missing_positive_issues = _validate_page(
    page={'id': 1},
    proposals=post_override_proposals,
    detector=post_override_detector,
    extractions=missing_positive_extractions,
    verifier=missing_positive_verifier,
    word_map=post_override_word_map,
)
assert missing_positive_passed is False, missing_positive_issues
assert any(
    issue.get('issue_type')
    == 'terminal-visible-source-evidence-unrepresented'
    for issue in missing_positive_issues
), missing_positive_issues

# Phase 2B V1.1: deterministic BOM publication tests.
import copy
from electrical_bom import (
    _apply_overrides as _apply_bom_overrides,
    _assign_row_glyphs_to_cells as _bom_assign_row_glyphs,
    _build_table_proposal as _bom_build_table_proposal,
    _candidate_tables as _bom_candidate_tables,
    _canonical_row_candidate_accounting as _bom_row_candidate_accounting,
    _component_tag_sequence_source_exact_adjudication as _bom_tag_sequence_adjudication,
    _detect_sidecar_column_specs as _bom_detect_sidecar_columns,
    _field_evidence_audit as _bom_field_evidence_audit,
    _fallback_page_proposal as _bom_fallback_page_proposal,
    _glyph_map as _bom_glyph_map,
    _reconcile_exact_cell_glyph_evidence as _bom_reconcile_glyph_cell_evidence,
    _canonicalize_verifier_row_references as _bom_canonicalize_verifier_rows,
    _parse_quantity as _parse_bom_quantity,
    _physical_bom_key as _bom_physical_key,
    _reconcile_post_override_rows as _bom_reconcile_post_override_rows,
    _semantic_character_signature as _bom_character_signature,
    _validate_page as _validate_bom_page,
    _word_map as _bom_word_map_from_page,
)

assert str(_parse_bom_quantity('2')) == '2'
assert str(_parse_bom_quantity('2,500')) == '2.500'
assert _parse_bom_quantity('K1') is None
assert _parse_bom_quantity('3VA5180-4EC31-0AA0') is None
assert _bom_character_signature('PN-1') != _bom_character_signature('PN1')
assert _bom_character_signature('PN – 1') == _bom_character_signature('PN-1')

# Full-page fallback must expose exact vector-word IDs, text and geometry to
# the extractor; a bare list of opaque IDs would be unverifiable.
import fitz as _bom_fitz
_bom_doc = _bom_fitz.open()
try:
    _bom_page_obj = _bom_doc.new_page(width=100, height=100)
    _bom_fallback = _bom_fallback_page_proposal(
        source_page=_bom_page_obj,
        inventory_page={'pdf_page_number': 66, 'page_sha256': 'fixture'},
        word_map={
            1: {
                'id': 1, 'text': 'PN-1',
                'x0': 10, 'y0': 20, 'x1': 30, 'y1': 25,
            },
        },
    )
    assert _bom_fallback['fallback_page_word_ids'] == [1]
    assert _bom_fallback['fallback_page_words'] == [{
        'word_id': 1,
        'bbox_pt': [10.0, 20.0, 30.0, 25.0],
        'text_original': 'PN-1',
    }]
finally:
    _bom_doc.close()

# A visually populated adjacent column must be recovered when find_tables
# stops at the last fully bordered lane. The rule is row-alignment based and
# does not know any language, header word or manufacturer name.
_sidecar_doc = _bom_fitz.open()
try:
    _sidecar_page = _sidecar_doc.new_page(width=120, height=60)
    _sidecar_rows = [
        _bom_fitz.Rect(5, index * 10, 60, (index + 1) * 10)
        for index in range(5)
    ]
    _sidecar_word_map = {}
    _wid = 1
    for index in range(5):
        y0 = index * 10 + 2
        _sidecar_word_map[_wid] = {
            'id': _wid, 'text': f'BASE{index}',
            'x0': 10, 'y0': y0, 'x1': 25, 'y1': y0 + 5,
        }
        _wid += 1
        _sidecar_word_map[_wid] = {
            'id': _wid, 'text': f'MAKER{index}',
            'x0': 67, 'y0': y0, 'x1': 86, 'y1': y0 + 5,
        }
        _wid += 1
    _sidecar_specs = _bom_detect_sidecar_columns(
        source_page=_sidecar_page,
        table_bbox=_bom_fitz.Rect(5, 0, 60, 50),
        row_rects=_sidecar_rows,
        word_map=_sidecar_word_map,
    )
    assert len(_sidecar_specs) == 1, _sidecar_specs
    assert _sidecar_specs[0]['side'] == 'right', _sidecar_specs
    assert _sidecar_specs[0]['support_row_count'] == 5, _sidecar_specs
    assert _sidecar_specs[0]['bbox_pt'][0] >= 60, _sidecar_specs

    _sparse_sidecar_specs = _bom_detect_sidecar_columns(
        source_page=_sidecar_page,
        table_bbox=_bom_fitz.Rect(5, 0, 60, 50),
        row_rects=_sidecar_rows,
        word_map={
            1: {
                'id': 1,
                'text': 'UNRELATED',
                'x0': 65,
                'y0': 2,
                'x1': 90,
                'y1': 7,
            },
        },
    )
    assert _sparse_sidecar_specs == [], _sparse_sidecar_specs
finally:
    _sidecar_doc.close()

# Integration fixture: PyMuPDF detects only the three fully ruled columns,
# while the fourth adjacent populated lane is recovered and included in every
# row candidate. This reproduces the failure mode found on the first real BOM
# page without using any page number, language or manufacturer name in logic.
_integration_doc = _bom_fitz.open()
try:
    _integration_page = _integration_doc.new_page(width=400, height=250)
    _x0 = 40.0
    _widths = [55.0, 150.0, 80.0]
    _x_edges = [_x0]
    for _width in _widths:
        _x_edges.append(_x_edges[-1] + _width)
    _y0 = 30.0
    _row_height = 28.0
    _row_count = 6
    for _x in _x_edges:
        _integration_page.draw_line(
            (_x, _y0),
            (_x, _y0 + _row_count * _row_height),
            color=(0, 0, 0),
            width=0.8,
        )
    for _row_index in range(_row_count + 1):
        _y = _y0 + _row_index * _row_height
        _integration_page.draw_line(
            (_x0, _y),
            (_x_edges[-1], _y),
            color=(0, 0, 0),
            width=0.8,
        )
    for _column_index, _header in enumerate(
        ['FIELD-A', 'FIELD-B', 'FIELD-C']
    ):
        _integration_page.insert_text(
            (_x_edges[_column_index] + 3, _y0 + 18),
            _header,
            fontsize=8,
        )
    _integration_page.insert_text(
        (_x_edges[-1] + 8, _y0 + 18),
        'FIELD-D',
        fontsize=8,
    )
    for _row_index in range(1, _row_count):
        _y = _y0 + _row_index * _row_height + 18
        for _column_index, _value in enumerate(
            [
                f'A{_row_index}',
                f'B{_row_index}',
                f'C{_row_index}',
            ]
        ):
            _integration_page.insert_text(
                (_x_edges[_column_index] + 3, _y),
                _value,
                fontsize=8,
            )
        _integration_page.insert_text(
            (_x_edges[-1] + 8, _y),
            f'D{_row_index}',
            fontsize=8,
        )

    _integration_words = list(
        _integration_page.get_text('words', sort=True) or []
    )
    _integration_inventory_page = {
        'pdf_page_number': 1,
        'page_sha256': 'sidecar-integration-fixture',
        'words': [list(word) for word in _integration_words],
    }
    _integration_word_map = _bom_word_map_from_page(
        _integration_inventory_page
    )
    _integration_tables = _bom_candidate_tables(_integration_page)
    assert len(_integration_tables) == 1, _integration_tables
    assert int(_integration_tables[0].col_count or 0) == 3
    _integration_proposal = _bom_build_table_proposal(
        table=_integration_tables[0],
        proposal_index=1,
        source_page=_integration_page,
        inventory_page=_integration_inventory_page,
        word_map=_integration_word_map,
    )
    assert _integration_proposal['deterministic_column_count'] == 4, (
        _integration_proposal
    )
    assert _integration_proposal['geometry_recovery'][
        'recovered_column_count'
    ] == 1, _integration_proposal
    assert all(
        len(row.get('cells') or []) == 4
        for row in _integration_proposal['row_candidates']
    ), _integration_proposal
    assert all(
        (row.get('cells') or [])[-1].get('geometry_source')
        == 'row_aligned_sidecar_recovery_v1'
        for row in _integration_proposal['row_candidates']
    ), _integration_proposal
finally:
    _integration_doc.close()

bom_page = {
    'id': 900,
    'pdf_page_number': 66,
    'page_width_pt': 80,
    'page_height_pt': 40,
}
bom_word_map = {
    1: {'id': 1, 'text': 'K1', 'x0': 0, 'y0': 10, 'x1': 5, 'y1': 15},
    2: {'id': 2, 'text': '1', 'x0': 10, 'y0': 10, 'x1': 12, 'y1': 15},
    3: {'id': 3, 'text': 'ACME', 'x0': 20, 'y0': 10, 'x1': 30, 'y1': 15},
    4: {'id': 4, 'text': 'PN-1', 'x0': 40, 'y0': 10, 'x1': 50, 'y1': 15},
    5: {'id': 5, 'text': 'Relay', 'x0': 60, 'y0': 10, 'x1': 70, 'y1': 15},
    6: {'id': 6, 'text': 'K1', 'x0': 0, 'y0': 20, 'x1': 5, 'y1': 25},
    7: {'id': 7, 'text': '1', 'x0': 10, 'y0': 20, 'x1': 12, 'y1': 25},
    8: {'id': 8, 'text': 'ACME', 'x0': 20, 'y0': 20, 'x1': 30, 'y1': 25},
    9: {'id': 9, 'text': 'PN-1', 'x0': 40, 'y0': 20, 'x1': 50, 'y1': 25},
    10: {'id': 10, 'text': 'Coil', 'x0': 60, 'y0': 20, 'x1': 70, 'y1': 25},
    11: {'id': 11, 'text': 'Tag', 'x0': 0, 'y0': 0, 'x1': 5, 'y1': 5},
    12: {'id': 12, 'text': 'Qty', 'x0': 10, 'y0': 0, 'x1': 15, 'y1': 5},
    13: {'id': 13, 'text': 'Maker', 'x0': 20, 'y0': 0, 'x1': 30, 'y1': 5},
    14: {'id': 14, 'text': 'Code', 'x0': 40, 'y0': 0, 'x1': 50, 'y1': 5},
    15: {'id': 15, 'text': 'Description', 'x0': 60, 'y0': 0, 'x1': 75, 'y1': 5},
}
bom_proposals = [{
    'region_id': 'P66-BOM01',
    'row_candidates': [
        {'source_row_candidate_id': 'R001', 'word_ids': [11, 12, 13, 14, 15]},
        {'source_row_candidate_id': 'R002', 'word_ids': [1, 2, 3, 4, 5]},
        {'source_row_candidate_id': 'R003', 'word_ids': [6, 7, 8, 9, 10]},
    ],
}]
bom_detector = {
    'page_id': 900,
    'all_visible_bom_tables_accounted_for': True,
    'proposal_assessments': [{
        'region_id': 'P66-BOM01',
        'visible': True,
        'distinct_table': True,
        'kind': 'bom_table',
        'expected_header_rows': 1,
        'expected_item_rows': 2,
        'expected_column_count': 5,
        'confidence': 0.99,
        'notes': 'fixture',
    }],
    'missing_visible_bom_tables': [],
    'confidence': 0.99,
    'issues': [],
}


def _bom_item(
    *,
    row_id,
    source_row_candidate_id,
    visual_order,
    ids,
    tag,
    quantity,
    manufacturer,
    part_number,
    description,
):
    values = {
        'item_position': '',
        'component_tag': tag,
        'quantity_text': quantity,
        'unit': '',
        'description': description,
        'part_number': part_number,
        'manufacturer': manufacturer,
    }
    row = {
        'row_id': row_id,
        'source_row_candidate_id': source_row_candidate_id,
        'visual_order': visual_order,
        'row_role': 'item',
        'field_evidence': [
            {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': [ids[0]]},
            {'field_name': 'quantity_text_original', 'source_column_index': 1, 'source_word_ids': [ids[1]]},
            {'field_name': 'manufacturer_original', 'source_column_index': 2, 'source_word_ids': [ids[2]]},
            {'field_name': 'part_number_original', 'source_column_index': 3, 'source_word_ids': [ids[3]]},
            {'field_name': 'description_original', 'source_column_index': 4, 'source_word_ids': [ids[4]]},
        ],
        'source_word_ids': ids,
        'bbox_pt': [0, visual_order * 10, 80, visual_order * 10 + 8],
        'confidence': 0.98,
        'evidence_notes': 'fixture',
    }
    for field_name, value in values.items():
        row[field_name + '_original'] = value
        row[field_name + '_normalized'] = value
    return row


def _bom_fixture_extractions():
    return [{
        'page_id': 900,
        'region_id': 'P66-BOM01',
        'header_row_candidate_ids': ['R001'],
        'non_item_rows': [],
        'source_column_roles': [
            {'source_column_index': 0, 'canonical_roles': ['component_tag'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 1, 'canonical_roles': ['quantity'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 2, 'canonical_roles': ['manufacturer'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 3, 'canonical_roles': ['part_number'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 4, 'canonical_roles': ['description'], 'confidence': 0.99, 'reason': 'fixture'},
        ],
        'rows': [
            _bom_item(
                row_id='r1',
                source_row_candidate_id='R002',
                visual_order=1,
                ids=[1, 2, 3, 4, 5],
                tag='K1',
                quantity='1',
                manufacturer='ACME',
                part_number='PN-1',
                description='Relay',
            ),
            _bom_item(
                row_id='r2',
                source_row_candidate_id='R003',
                visual_order=2,
                ids=[6, 7, 8, 9, 10],
                tag='K1',
                quantity='1',
                manufacturer='ACME',
                part_number='PN-1',
                description='Coil',
            ),
        ],
        'unaccounted_row_candidate_ids': [],
        'duplicate_row_ids': [],
        'confidence': 0.99,
        'issues': [],
    }]


bom_verifier = {
    'page_id': 900,
    'verdict': 'pass',
    'all_visible_bom_tables_accounted_for': True,
    'all_visible_item_rows_accounted_for': True,
    'all_visible_columns_accounted_for': True,
    'all_published_fields_visually_supported': True,
    'all_source_evidence_represented': True,
    'duplicates_preserved': True,
    'region_checks': [{
        'region_id': 'P66-BOM01',
        'expected_item_rows': 2,
        'verified_item_rows': 2,
        'verified_row_ids': ['r1', 'r2'],
        'verified_component_tag_sequence': ['K1', 'K1'],
        'pass': True,
        'confidence': 0.99,
        'notes': 'Repeated content belongs to two distinct physical rows.',
    }],
    'field_overrides': [],
    'missing_region_ids': [],
    'missing_row_ids': [],
    'duplicate_physical_keys': [],
    'unaccounted_visual_evidence': [],
    'confidence': 0.99,
    'issues': [],
}

bom_extractions = _bom_fixture_extractions()
bom_passed, bom_rows, bom_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_passed is True, bom_issues
assert len(bom_rows) == 2, bom_rows
assert [row['component_tag_original'] for row in bom_rows] == ['K1', 'K1']
assert [row['part_number_original'] for row in bom_rows] == ['PN-1', 'PN-1']
assert all(row['source_evidence_coverage']['complete'] for row in bom_rows)

# A header echoed in both legacy header containers is one physical
# classification and must not create a false row-accounting failure.
bom_header_alias_extractions = _bom_fixture_extractions()
bom_header_alias_extractions[0]['non_item_rows'] = [{
    'source_row_candidate_id': 'R001',
    'kind': 'header',
    'reason': 'legacy duplicate header classification',
    'confidence': 0.99,
}]
header_ids, non_item_ids, item_ids = _bom_row_candidate_accounting(
    bom_header_alias_extractions[0]
)
assert header_ids == {'R001'}
assert non_item_ids == set()
assert item_ids == {'R002', 'R003'}
bom_header_alias_passed, _, bom_header_alias_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_header_alias_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_header_alias_passed is True, bom_header_alias_issues

# Artificial spacing differences in verifier tag sequences do not alter the
# source characters and therefore do not create a false row mismatch.
bom_spacing_verifier = copy.deepcopy(bom_verifier)
bom_spacing_verifier['region_checks'][0][
    'verified_component_tag_sequence'
] = ['K 1', 'K1']
bom_spacing_passed, _, bom_spacing_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=_bom_fixture_extractions(),
    verifier=bom_spacing_verifier,
    word_map=bom_word_map,
)
assert bom_spacing_passed is True, bom_spacing_issues

# Repeated field values are not duplicate physical rows. A conservative
# verifier report is locally adjudicated only because row candidate IDs and
# visual orders are independently unique.
bom_repeated_value_verifier = copy.deepcopy(bom_verifier)
bom_repeated_value_verifier['duplicates_preserved'] = False
bom_repeated_value_verifier['duplicate_physical_keys'] = ['K1', 'PN-1']
bom_repeated_value_passed, _, bom_repeated_value_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=_bom_fixture_extractions(),
    verifier=bom_repeated_value_verifier,
    word_map=bom_word_map,
)
assert bom_repeated_value_passed is True, bom_repeated_value_issues
assert any(
    issue.get('issue_type')
    == 'bom-repeated-values-preserved-as-distinct-rows'
    and issue.get('severity') == 'info'
    for issue in bom_repeated_value_issues
), bom_repeated_value_issues

bom_unknown_duplicate_verifier = copy.deepcopy(bom_verifier)
bom_unknown_duplicate_verifier['duplicate_physical_keys'] = [
    'UNKNOWN-PHYSICAL-KEY'
]
bom_unknown_duplicate_passed, _, bom_unknown_duplicate_issues = (
    _validate_bom_page(
        page=bom_page,
        proposals=bom_proposals,
        detector=bom_detector,
        extractions=_bom_fixture_extractions(),
        verifier=bom_unknown_duplicate_verifier,
        word_map=bom_word_map,
    )
)
assert bom_unknown_duplicate_passed is False, bom_unknown_duplicate_issues
assert any(
    issue.get('issue_type') == 'bom-verifier-duplicate-physical-keys'
    and issue.get('severity') == 'high'
    for issue in bom_unknown_duplicate_issues
), bom_unknown_duplicate_issues

# An exact positive override restores a field without losing source evidence.
bom_override_extractions = _bom_fixture_extractions()
bom_override_extractions[0]['rows'][0]['manufacturer_original'] = ''
_apply_bom_overrides(
    bom_override_extractions,
    [{
        'region_id': 'P66-BOM01',
        'row_id': 'r1',
        'field_name': 'manufacturer_original',
        'approved_text': 'ACME',
        'confidence': 0.99,
        'reason': 'Exact visible manufacturer cell.',
    }],
)
assert bom_override_extractions[0]['rows'][0]['manufacturer_original'] == 'ACME'
assert _bom_field_evidence_audit(
    row=bom_override_extractions[0]['rows'][0],
    expected_word_ids=[1, 2, 3, 4, 5],
    word_map=bom_word_map,
)['complete'] is True

# Silent source-text loss must block publication.
bom_loss_extractions = _bom_fixture_extractions()
bom_loss_extractions[0]['rows'][0]['description_original'] = ''
bom_loss_extractions[0]['rows'][0]['description_normalized'] = ''
bom_loss_passed, _, bom_loss_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_loss_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_loss_passed is False, bom_loss_issues
assert any(
    issue.get('issue_type') == 'bom-visible-source-evidence-unrepresented'
    for issue in bom_loss_issues
), bom_loss_issues

# Technical-code punctuation cannot disappear during normalization.
bom_code_change_extractions = _bom_fixture_extractions()
bom_code_change_extractions[0]['rows'][0]['part_number_normalized'] = 'PN1'
bom_code_change_passed, _, bom_code_change_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_code_change_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_code_change_passed is False, bom_code_change_issues
assert any(
    issue.get('issue_type') == 'bom-normalized-text-changed-source-content'
    for issue in bom_code_change_issues
), bom_code_change_issues

# One visible word cannot populate two canonical fields.
bom_collision_extractions = _bom_fixture_extractions()
bom_collision_row = bom_collision_extractions[0]['rows'][0]
bom_collision_row['part_number_original'] = 'K1'
bom_collision_row['part_number_normalized'] = 'K1'
bom_collision_row['field_evidence'][3]['source_word_ids'] = [1]
bom_collision_passed, _, bom_collision_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_collision_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_collision_passed is False, bom_collision_issues
assert any(
    issue.get('issue_type') == 'bom-visible-source-evidence-unrepresented'
    for issue in bom_collision_issues
), bom_collision_issues

# Duplicate physical order blocks; repeated tag/part values do not.
bom_duplicate_extractions = _bom_fixture_extractions()
bom_duplicate_extractions[0]['rows'][1]['visual_order'] = 1
bom_duplicate_passed, _, bom_duplicate_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_duplicate_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_duplicate_passed is False, bom_duplicate_issues
assert any(
    issue.get('issue_type') == 'bom-visual-order-invalid'
    for issue in bom_duplicate_issues
), bom_duplicate_issues

# Missing semantic column accounting blocks publication.
bom_column_loss_extractions = _bom_fixture_extractions()
bom_column_loss_extractions[0]['source_column_roles'].pop()
bom_column_loss_passed, _, bom_column_loss_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_column_loss_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_column_loss_passed is False, bom_column_loss_issues
assert any(
    issue.get('issue_type') == 'bom-source-column-accounting-mismatch'
    for issue in bom_column_loss_issues
), bom_column_loss_issues

# A field cannot be published from a physical column bound to another role.
bom_wrong_lane_extractions = _bom_fixture_extractions()
bom_wrong_lane_extractions[0]['source_column_roles'][4][
    'canonical_roles'
] = ['manufacturer']
bom_wrong_lane_passed, _, bom_wrong_lane_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_wrong_lane_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_wrong_lane_passed is False, bom_wrong_lane_issues
assert any(
    issue.get('issue_type') == 'bom-field-column-binding-mismatch'
    for issue in bom_wrong_lane_issues
), bom_wrong_lane_issues

# Two materialized rows cannot claim the same deterministic physical row.
bom_source_row_duplicate_extractions = _bom_fixture_extractions()
bom_source_row_duplicate_extractions[0]['rows'][1][
    'source_row_candidate_id'
] = 'R002'
bom_source_row_duplicate_passed, _, bom_source_row_duplicate_issues = (
    _validate_bom_page(
        page=bom_page,
        proposals=bom_proposals,
        detector=bom_detector,
        extractions=bom_source_row_duplicate_extractions,
        verifier=bom_verifier,
        word_map=bom_word_map,
    )
)
assert bom_source_row_duplicate_passed is False, bom_source_row_duplicate_issues
assert any(
    issue.get('issue_type') == 'bom-source-row-candidate-duplicate'
    for issue in bom_source_row_duplicate_issues
), bom_source_row_duplicate_issues

# Masked source values remain real evidence even without alphanumeric text.
mask_word_map = {
    1: {'id': 1, 'text': '********', 'x0': 0, 'y0': 0, 'x1': 20, 'y1': 5},
}
mask_row = {
    'part_number_original': '********',
    'field_evidence': [{
        'field_name': 'part_number_original',
        'source_column_index': 0,
        'source_word_ids': [1],
    }],
}
assert _bom_field_evidence_audit(
    row=mask_row,
    expected_word_ids=[1],
    word_map=mask_word_map,
)['complete'] is True
mask_row['part_number_original'] = '*******'
assert _bom_field_evidence_audit(
    row=mask_row,
    expected_word_ids=[1],
    word_map=mask_word_map,
)['complete'] is False

# A verifier issue is downgraded only when every linked exact override was
# actually applied before deterministic publication validation.
bom_resolved_extractions = _bom_fixture_extractions()
bom_resolved_extractions[0]['rows'][0]['manufacturer_original'] = ''
resolved_override = {
    'region_id': 'P66-BOM01',
    'row_id': 'r1',
    'field_name': 'manufacturer_original',
    'approved_text': 'ACME',
    'confidence': 0.99,
    'reason': 'Exact visible manufacturer cell.',
}
_apply_bom_overrides(bom_resolved_extractions, [resolved_override])
bom_resolved_verifier = copy.deepcopy(bom_verifier)
bom_resolved_verifier['field_overrides'] = [resolved_override]
bom_resolved_verifier['issues'] = [{
    'issue_type': 'wrong_field_assignment',
    'severity': 'high',
    'message': 'Manufacturer was missing before the exact correction.',
    'region_id': 'P66-BOM01',
    'row_ids': ['r1'],
    'confidence': 0.99,
    'resolution_status': 'resolved_by_exact_overrides',
    'related_overrides': [{
        'region_id': 'P66-BOM01',
        'row_id': 'r1',
        'field_name': 'manufacturer_original',
    }],
}]
bom_resolved_passed, _, bom_resolved_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_resolved_extractions,
    verifier=bom_resolved_verifier,
    word_map=bom_word_map,
)
assert bom_resolved_passed is True, bom_resolved_issues
assert any(
    issue.get('source_stage') == 'verifier_post_override_resolved'
    and issue.get('severity') == 'info'
    for issue in bom_resolved_issues
), bom_resolved_issues

bom_false_resolution_verifier = copy.deepcopy(bom_resolved_verifier)
bom_false_resolution_passed, _, bom_false_resolution_issues = (
    _validate_bom_page(
        page=bom_page,
        proposals=bom_proposals,
        detector=bom_detector,
        extractions=_bom_fixture_extractions(),
        verifier=bom_false_resolution_verifier,
        word_map=bom_word_map,
    )
)
assert bom_false_resolution_passed is False, bom_false_resolution_issues
assert any(
    issue.get('source_stage') == 'deterministic_validator'
    and issue.get('severity') == 'high'
    and issue.get('post_override_resolution', {}).get('validated') is False
    for issue in bom_false_resolution_issues
), bom_false_resolution_issues

# Invalid row geometry must fail closed.
bom_bad_bbox_extractions = _bom_fixture_extractions()
bom_bad_bbox_extractions[0]['rows'][0]['bbox_pt'] = [20, 10, 10, 15]
bom_bad_bbox_passed, _, bom_bad_bbox_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=bom_bad_bbox_extractions,
    verifier=bom_verifier,
    word_map=bom_word_map,
)
assert bom_bad_bbox_passed is False, bom_bad_bbox_issues
assert any(
    issue.get('issue_type') == 'bom-row-bbox-invalid'
    for issue in bom_bad_bbox_issues
), bom_bad_bbox_issues

# Identical content on two physical rows receives different immutable keys.
key1 = _bom_physical_key(
    version_id=2,
    page_id=68,
    region_id='P66-BOM01',
    visual_order=1,
)
key2 = _bom_physical_key(
    version_id=2,
    page_id=68,
    region_id='P66-BOM01',
    visual_order=2,
)
assert key1 != key2


# Phase 2B V1.2: exact source x geometry must repair a technical code whose
# vector fragments were returned in non-physical order. The deterministic
# source characters win over a visually plausible O/0 substitution.
_bom_source_order_row = {
    'row_id': 'ROW-CODE-ORDER',
    'part_number_original': '3VA9137OFK31',
    'part_number_normalized': '0FK33VA91371',
    'field_evidence': [{
        'field_name': 'part_number_original',
        'source_column_index': 2,
        'source_word_ids': [2, 1, 3],
    }],
    'verifier_overrides': {
        'part_number_original': {
            'before': '0FK3 3VA9137 1',
            'after': '3VA9137OFK31',
            'confidence': 0.99,
            'reason': 'Visual candidate contained an O/0 ambiguity.',
        },
    },
}
_bom_source_order_word_map = {
    1: {
        'id': 1, 'text': '3VA9137',
        'x0': 10, 'y0': 10, 'x1': 30, 'y1': 15,
    },
    2: {
        'id': 2, 'text': '0FK3',
        'x0': 31, 'y0': 10, 'x1': 42, 'y1': 15,
    },
    3: {
        'id': 3, 'text': '1',
        'x0': 43, 'y0': 10, 'x1': 45, 'y1': 15,
    },
}
_bom_source_order_audit = _bom_reconcile_post_override_rows(
    proposals=[],
    extractions=[{'rows': [_bom_source_order_row]}],
    word_map=_bom_source_order_word_map,
)
assert _bom_source_order_row['part_number_original'] == '3VA91370FK31', (
    _bom_source_order_row
)
assert _bom_source_order_row['part_number_normalized'] == '3VA91370FK31', (
    _bom_source_order_row
)
assert _bom_source_order_audit['source_order_row_ids'] == [
    'ROW-CODE-ORDER'
], _bom_source_order_audit

# Exact visual word-order repair may adjudicate a source-string mismatch only
# when it preserves the complete source character multiset. A substitution
# without deterministic source authority must still fail closed.
_bom_reordered_description = {
    'description_original': 'interruttore automatico 3va5 ul frame 125',
    'field_evidence': [{
        'field_name': 'description_original',
        'source_column_index': 1,
        'source_word_ids': [1, 2, 3, 4, 5, 6],
    }],
    'verifier_overrides': {
        'description_original': {
            'before': 'frame interruttore automatico 3va5 ul 125',
            'after': 'interruttore automatico 3va5 ul frame 125',
            'confidence': 0.99,
            'reason': 'Exact visible reading order.',
        },
    },
}
_bom_reordered_words = {
    1: {'id': 1, 'text': 'frame', 'x0': 1, 'y0': 1, 'x1': 2, 'y1': 2},
    2: {'id': 2, 'text': 'interruttore', 'x0': 3, 'y0': 1, 'x1': 4, 'y1': 2},
    3: {'id': 3, 'text': 'automatico', 'x0': 5, 'y0': 1, 'x1': 6, 'y1': 2},
    4: {'id': 4, 'text': '3va5', 'x0': 7, 'y0': 1, 'x1': 8, 'y1': 2},
    5: {'id': 5, 'text': 'ul', 'x0': 9, 'y0': 1, 'x1': 10, 'y1': 2},
    6: {'id': 6, 'text': '125', 'x0': 11, 'y0': 1, 'x1': 12, 'y1': 2},
}
_bom_reordered_audit = _bom_field_evidence_audit(
    row=_bom_reordered_description,
    expected_word_ids=[1, 2, 3, 4, 5, 6],
    word_map=_bom_reordered_words,
)
assert _bom_reordered_audit['complete'] is True, _bom_reordered_audit
assert len(
    _bom_reordered_audit['adjudicated_field_text_mismatches']
) == 1, _bom_reordered_audit

_bom_substitution = copy.deepcopy(_bom_reordered_description)
_bom_substitution['description_original'] = (
    'interruttore automatico 3va5 uI frame 125'
)
_bom_substitution['verifier_overrides']['description_original']['after'] = (
    _bom_substitution['description_original']
)
_bom_substitution_audit = _bom_field_evidence_audit(
    row=_bom_substitution,
    expected_word_ids=[1, 2, 3, 4, 5, 6],
    word_map=_bom_reordered_words,
)
assert _bom_substitution_audit['complete'] is False, (
    _bom_substitution_audit
)

# A pre-correction verifier review_required may be superseded only after the
# exact override is applied, the normalized display value is synchronized,
# and every deterministic publication check is clean.
_bom_v12_page = {
    'id': 901,
    'pdf_page_number': 66,
    'page_width_pt': 100,
    'page_height_pt': 50,
}
_bom_v12_word_map = {
    1: {'id': 1, 'text': 'K1', 'x0': 1, 'y0': 10, 'x1': 5, 'y1': 15},
    2: {'id': 2, 'text': 'Relay', 'x0': 10, 'y0': 10, 'x1': 25, 'y1': 15},
    3: {'id': 3, 'text': 'PN-1', 'x0': 30, 'y0': 10, 'x1': 45, 'y1': 15},
    4: {'id': 4, 'text': 'ACME', 'x0': 50, 'y0': 10, 'x1': 65, 'y1': 15},
    5: {'id': 5, 'text': 'Tag', 'x0': 1, 'y0': 0, 'x1': 5, 'y1': 5},
    6: {'id': 6, 'text': 'Desc', 'x0': 10, 'y0': 0, 'x1': 25, 'y1': 5},
    7: {'id': 7, 'text': 'Code', 'x0': 30, 'y0': 0, 'x1': 45, 'y1': 5},
    8: {'id': 8, 'text': 'Maker', 'x0': 50, 'y0': 0, 'x1': 65, 'y1': 5},
}
_bom_v12_proposals = [{
    'region_id': 'R-V12',
    'row_candidates': [
        {'source_row_candidate_id': 'H1', 'word_ids': [5, 6, 7, 8]},
        {'source_row_candidate_id': 'S1', 'word_ids': [1, 2, 3, 4]},
    ],
}]
_bom_v12_detector = {
    'page_id': 901,
    'all_visible_bom_tables_accounted_for': True,
    'missing_visible_bom_tables': [],
    'confidence': 0.99,
    'issues': [],
    'proposal_assessments': [{
        'region_id': 'R-V12',
        'visible': True,
        'distinct_table': True,
        'kind': 'bom_table',
        'expected_header_rows': 1,
        'expected_item_rows': 1,
        'expected_column_count': 4,
        'confidence': 0.99,
        'notes': 'fixture',
    }],
}
_bom_v12_row = {
    'row_id': 'r-v12',
    'source_row_candidate_id': 'S1',
    'visual_order': 1,
    'row_role': 'item',
    'confidence': 0.99,
    'bbox_pt': [0, 9, 70, 16],
    'source_word_ids': [1, 2, 3, 4],
    'evidence_notes': 'fixture',
    'field_evidence': [
        {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': [1]},
        {'field_name': 'description_original', 'source_column_index': 1, 'source_word_ids': [2]},
        {'field_name': 'part_number_original', 'source_column_index': 2, 'source_word_ids': [3]},
        {'field_name': 'manufacturer_original', 'source_column_index': 3, 'source_word_ids': [4]},
    ],
    'item_position_original': '',
    'item_position_normalized': '',
    'component_tag_original': 'K1',
    'component_tag_normalized': 'K1',
    'quantity_text_original': '',
    'quantity_text_normalized': '',
    'unit_original': '',
    'unit_normalized': '',
    'description_original': 'y Rela',
    'description_normalized': 'yRela',
    'part_number_original': 'PN-1',
    'part_number_normalized': 'PN-1',
    'manufacturer_original': 'ACME',
    'manufacturer_normalized': 'ACME',
}
_bom_v12_extractions = [{
    'page_id': 901,
    'region_id': 'R-V12',
    'header_row_candidate_ids': ['H1'],
    'non_item_rows': [],
    'source_column_roles': [
        {'source_column_index': 0, 'canonical_roles': ['component_tag'], 'confidence': 0.99, 'reason': 'fixture'},
        {'source_column_index': 1, 'canonical_roles': ['description'], 'confidence': 0.99, 'reason': 'fixture'},
        {'source_column_index': 2, 'canonical_roles': ['part_number'], 'confidence': 0.99, 'reason': 'fixture'},
        {'source_column_index': 3, 'canonical_roles': ['manufacturer'], 'confidence': 0.99, 'reason': 'fixture'},
    ],
    'rows': [_bom_v12_row],
    'unaccounted_row_candidate_ids': [],
    'duplicate_row_ids': [],
    'confidence': 0.99,
    'issues': [],
}]
_bom_v12_override = {
    'region_id': 'R-V12',
    'row_id': 'r-v12',
    'field_name': 'description_original',
    'approved_text': 'Relay',
    'confidence': 0.99,
    'reason': 'Exact visible transcription.',
}
_bom_v12_verifier = {
    'page_id': 901,
    'verdict': 'review_required',
    'all_visible_bom_tables_accounted_for': True,
    'all_visible_item_rows_accounted_for': True,
    'all_visible_columns_accounted_for': True,
    'all_published_fields_visually_supported': False,
    'all_source_evidence_represented': True,
    'duplicates_preserved': True,
    'region_checks': [{
        'region_id': 'R-V12',
        'expected_item_rows': 1,
        'verified_item_rows': 1,
        'verified_row_ids': ['r-v12'],
        'verified_component_tag_sequence': ['K1'],
        'pass': False,
        'confidence': 0.99,
        'notes': 'Pre-correction normalized text was stale.',
    }],
    'field_overrides': [_bom_v12_override],
    'missing_region_ids': [],
    'missing_row_ids': [],
    'duplicate_physical_keys': [],
    'unaccounted_visual_evidence': [],
    'confidence': 0.99,
    'issues': [{
        'issue_type': 'normalized_text_not_spacing_only',
        'severity': 'high',
        'message': 'Pre-correction normalized text was stale.',
        'region_id': 'R-V12',
        'row_ids': ['r-v12'],
        'confidence': 0.99,
        'resolution_status': 'open',
        'related_overrides': [],
    }],
}
_apply_bom_overrides(
    _bom_v12_extractions,
    [_bom_v12_override],
)
_bom_v12_passed, _bom_v12_rows, _bom_v12_issues = _validate_bom_page(
    page=_bom_v12_page,
    proposals=_bom_v12_proposals,
    detector=_bom_v12_detector,
    extractions=_bom_v12_extractions,
    verifier=_bom_v12_verifier,
    word_map=_bom_v12_word_map,
)
assert _bom_v12_passed is True, _bom_v12_issues
assert _bom_v12_rows[0]['description_original'] == 'Relay'
assert _bom_v12_rows[0]['description_normalized'] == 'Relay'
assert not [
    issue for issue in _bom_v12_issues
    if issue.get('severity') in {'high', 'critical'}
], _bom_v12_issues
assert any(
    issue.get('issue_type')
    == 'bom-verifier-verdict-superseded-post-override'
    for issue in _bom_v12_issues
), _bom_v12_issues

_bom_v12_unresolved_verifier = copy.deepcopy(_bom_v12_verifier)
_bom_v12_unresolved_verifier['issues'].append({
    'issue_type': 'unresolved_visual_ambiguity',
    'severity': 'high',
    'message': 'No exact correction is available.',
    'region_id': 'R-V12',
    'row_ids': ['r-v12'],
    'confidence': 0.99,
    'resolution_status': 'open',
    'related_overrides': [],
})
_bom_v12_unresolved_passed, _, _bom_v12_unresolved_issues = (
    _validate_bom_page(
        page=_bom_v12_page,
        proposals=_bom_v12_proposals,
        detector=_bom_v12_detector,
        extractions=copy.deepcopy(_bom_v12_extractions),
        verifier=_bom_v12_unresolved_verifier,
        word_map=_bom_v12_word_map,
    )
)
assert _bom_v12_unresolved_passed is False, _bom_v12_unresolved_issues
assert any(
    issue.get('issue_type') == 'unresolved_visual_ambiguity'
    and issue.get('severity') == 'high'
    for issue in _bom_v12_unresolved_issues
), _bom_v12_unresolved_issues


# Phase 2B V1.3: a whole source word can be segmented into the adjacent
# physical column. Publication may repair this only when two exact verifier
# overrides form one closed character transfer and one unique edge word proves
# the evidence movement. The final source coverage, normalized values, region
# decision, field-support flag, source-evidence flag and page verdict must all
# be revalidated after the transfer.
_bom_v13_page = {
    'id': 902,
    'pdf_page_number': 67,
    'page_width_pt': 200,
    'page_height_pt': 50,
}
_bom_v13_word_map = {
    1: {'id': 1, 'text': '30S1', 'x0': 1, 'y0': 10, 'x1': 8, 'y1': 15},
    2: {'id': 2, 'text': 'MODULO', 'x0': 10, 'y0': 10, 'x1': 20, 'y1': 15},
    3: {'id': 3, 'text': 'DI', 'x0': 21, 'y0': 10, 'x1': 24, 'y1': 15},
    4: {'id': 4, 'text': 'CONTATTO,', 'x0': 25, 'y0': 10, 'x1': 40, 'y1': 15},
    5: {'id': 5, 'text': '1NO,', 'x0': 41, 'y0': 10, 'x1': 47, 'y1': 15},
    6: {'id': 6, 'text': 'MORSETTO', 'x0': 48, 'y0': 10, 'x1': 60, 'y1': 15},
    7: {'id': 7, 'text': 'A', 'x0': 61, 'y0': 10, 'x1': 63, 'y1': 15},
    8: {'id': 8, 'text': 'VITE,', 'x0': 64, 'y0': 10, 'x1': 70, 'y1': 15},
    9: {'id': 9, 'text': 'PIASTRA', 'x0': 71, 'y0': 10, 'x1': 81, 'y1': 15},
    10: {'id': 10, 'text': 'FRONTAL', 'x0': 82, 'y0': 10, 'x1': 92, 'y1': 15},
    11: {'id': 11, 'text': 'E', 'x0': 93, 'y0': 10, 'x1': 94, 'y1': 15},
    12: {'id': 12, 'text': '3SU', 'x0': 95, 'y0': 10, 'x1': 100, 'y1': 15},
    13: {'id': 13, 'text': '1400', 'x0': 101, 'y0': 10, 'x1': 106, 'y1': 15},
    14: {'id': 14, 'text': '-1AA10', 'x0': 107, 'y0': 10, 'x1': 116, 'y1': 15},
    15: {'id': 15, 'text': '-1BA0', 'x0': 117, 'y0': 10, 'x1': 125, 'y1': 15},
    16: {'id': 16, 'text': 'SIEMENS', 'x0': 130, 'y0': 10, 'x1': 145, 'y1': 15},
    17: {'id': 17, 'text': 'Sigla', 'x0': 1, 'y0': 0, 'x1': 8, 'y1': 5},
    18: {'id': 18, 'text': 'Descrizione', 'x0': 10, 'y0': 0, 'x1': 30, 'y1': 5},
    19: {'id': 19, 'text': 'Codice', 'x0': 95, 'y0': 0, 'x1': 105, 'y1': 5},
    20: {'id': 20, 'text': 'Costruttore', 'x0': 130, 'y0': 0, 'x1': 150, 'y1': 5},
}
_bom_v13_proposals = [{
    'region_id': 'R-V13',
    'row_candidates': [
        {'source_row_candidate_id': 'H1', 'word_ids': [17, 18, 19, 20]},
        {'source_row_candidate_id': 'S1', 'word_ids': list(range(1, 17))},
    ],
}]
_bom_v13_detector = {
    'page_id': 902,
    'all_visible_bom_tables_accounted_for': True,
    'missing_visible_bom_tables': [],
    'confidence': 0.99,
    'issues': [],
    'proposal_assessments': [{
        'region_id': 'R-V13',
        'visible': True,
        'distinct_table': True,
        'kind': 'bom_table',
        'expected_header_rows': 1,
        'expected_item_rows': 1,
        'expected_column_count': 4,
        'confidence': 0.99,
        'notes': 'fixture',
    }],
}


def _bom_v13_fixture_extractions():
    return [{
        'page_id': 902,
        'region_id': 'R-V13',
        'header_row_candidate_ids': ['H1'],
        'non_item_rows': [],
        'source_column_roles': [
            {'source_column_index': 0, 'canonical_roles': ['component_tag'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 1, 'canonical_roles': ['description'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 2, 'canonical_roles': ['part_number'], 'confidence': 0.99, 'reason': 'fixture'},
            {'source_column_index': 3, 'canonical_roles': ['manufacturer'], 'confidence': 0.99, 'reason': 'fixture'},
        ],
        'rows': [{
            'row_id': 'ROW-X',
            'source_row_candidate_id': 'S1',
            'visual_order': 1,
            'row_role': 'item',
            'confidence': 0.99,
            'bbox_pt': [0, 9, 160, 16],
            'source_word_ids': list(range(1, 17)),
            'evidence_notes': 'fixture cross-column vector split',
            'field_evidence': [
                {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': [1]},
                {'field_name': 'description_original', 'source_column_index': 1, 'source_word_ids': list(range(2, 11))},
                {'field_name': 'part_number_original', 'source_column_index': 2, 'source_word_ids': [11, 12, 13, 14, 15]},
                {'field_name': 'manufacturer_original', 'source_column_index': 3, 'source_word_ids': [16]},
            ],
            'item_position_original': '',
            'item_position_normalized': '',
            'component_tag_original': '30S1',
            'component_tag_normalized': '30S1',
            'quantity_text_original': '',
            'quantity_text_normalized': '',
            'unit_original': '',
            'unit_normalized': '',
            'description_original': 'MODULO DI CONTATTO, 1NO, MORSETTO A VITE, PIASTRA FRONTAL',
            'description_normalized': 'MODULO DI CONTATTO, 1NO, MORSETTO A VITE, PIASTRA FRONTAL',
            'part_number_original': 'E 3SU 1400 -1AA10 -1BA0',
            'part_number_normalized': 'E3SU1400-1AA10-1BA0',
            'manufacturer_original': 'SIEMENS',
            'manufacturer_normalized': 'SIEMENS',
        }],
        'unaccounted_row_candidate_ids': [],
        'duplicate_row_ids': [],
        'confidence': 0.99,
        'issues': [],
    }]


_bom_v13_description_override = {
    'region_id': 'R-V13',
    'row_id': 'ROW-X',
    'field_name': 'description_original',
    'approved_text': (
        'MODULO DI CONTATTO, 1NO, MORSETTO A VITE, PIASTRA FRONTALE'
    ),
    'confidence': 0.99,
    'reason': 'The final whole source word belongs to the description cell.',
}
_bom_v13_part_override = {
    'region_id': 'R-V13',
    'row_id': 'ROW-X',
    'field_name': 'part_number_original',
    'approved_text': '3SU1400-1AA10-1BA0',
    'confidence': 0.99,
    'reason': 'The code cell begins with 3SU, not the leaked final letter.',
}
_bom_v13_verifier = {
    'page_id': 902,
    'verdict': 'review_required',
    'all_visible_bom_tables_accounted_for': True,
    'all_visible_item_rows_accounted_for': True,
    'all_visible_columns_accounted_for': True,
    'all_published_fields_visually_supported': False,
    'all_source_evidence_represented': False,
    'duplicates_preserved': True,
    'region_checks': [{
        'region_id': 'R-V13',
        'expected_item_rows': 1,
        'verified_item_rows': 1,
        'verified_row_ids': ['ROW-X'],
        'verified_component_tag_sequence': ['30S1'],
        'pass': False,
        'confidence': 0.99,
        'notes': 'Pre-correction candidate contains one cross-column split.',
    }],
    'field_overrides': [
        _bom_v13_description_override,
        _bom_v13_part_override,
    ],
    'missing_region_ids': [],
    'missing_row_ids': [],
    'duplicate_physical_keys': [],
    'unaccounted_visual_evidence': [],
    'confidence': 0.99,
    'issues': [
        {
            'issue_type': 'original_field_transcription_errors',
            'severity': 'high',
            'message': 'One source word crossed the visible column boundary.',
            'region_id': 'R-V13',
            'row_ids': ['ROW-X'],
            'confidence': 0.99,
            'resolution_status': 'resolved_by_exact_overrides',
            'related_overrides': [
                {'region_id': 'R-V13', 'row_id': 'ROW-X', 'field_name': 'description_original'},
                {'region_id': 'R-V13', 'row_id': 'ROW-X', 'field_name': 'part_number_original'},
            ],
        },
        {
            'issue_type': 'normalized_fields_not_faithful_to_source',
            'severity': 'high',
            'message': 'Pre-correction normalized fields were stale.',
            'region_id': 'R-V13',
            'row_ids': ['ROW-X'],
            'confidence': 0.99,
            'resolution_status': 'open',
            'related_overrides': [],
        },
    ],
}
_bom_v13_extractions = _bom_v13_fixture_extractions()
_apply_bom_overrides(
    _bom_v13_extractions,
    [_bom_v13_description_override, _bom_v13_part_override],
)
_bom_v13_passed, _bom_v13_rows, _bom_v13_issues = _validate_bom_page(
    page=_bom_v13_page,
    proposals=_bom_v13_proposals,
    detector=_bom_v13_detector,
    extractions=_bom_v13_extractions,
    verifier=_bom_v13_verifier,
    word_map=_bom_v13_word_map,
)
assert _bom_v13_passed is True, _bom_v13_issues
assert _bom_v13_rows[0]['description_original'].endswith('FRONTALE')
assert _bom_v13_rows[0]['description_normalized'].endswith('FRONTALE')
assert _bom_v13_rows[0]['part_number_original'] == '3SU1400-1AA10-1BA0'
assert _bom_v13_rows[0]['part_number_normalized'] == '3SU1400-1AA10-1BA0'
assert _bom_v13_rows[0]['source_evidence_coverage']['complete'] is True
assert _bom_v13_rows[0]['field_evidence'][1]['source_word_ids'][-1] == 11
assert 11 not in _bom_v13_rows[0]['field_evidence'][2]['source_word_ids']
assert _bom_v13_rows[0]['cross_field_evidence_transfers'][0][
    'moved_source_word_ids'
] == [11]
assert not [
    issue
    for issue in _bom_v13_issues
    if issue.get('severity') in {'high', 'critical'}
], _bom_v13_issues
assert any(
    issue.get('issue_type')
    == 'bom-cross-field-source-evidence-reconciled-post-override'
    for issue in _bom_v13_issues
), _bom_v13_issues
assert any(
    issue.get('issue_type')
    == 'bom-verifier-source-evidence-flag-superseded-post-override'
    for issue in _bom_v13_issues
), _bom_v13_issues

# An unmatched transfer must remain blocking: a receiver cannot gain X while
# the adjacent donor loses E.
_bom_v13_unmatched_extractions = _bom_v13_fixture_extractions()
_bom_v13_unmatched_description = {
    **_bom_v13_description_override,
    'approved_text': (
        'MODULO DI CONTATTO, 1NO, MORSETTO A VITE, PIASTRA FRONTALX'
    ),
}
_bom_v13_unmatched_verifier = copy.deepcopy(_bom_v13_verifier)
_bom_v13_unmatched_verifier['field_overrides'] = [
    _bom_v13_unmatched_description,
    _bom_v13_part_override,
]
_apply_bom_overrides(
    _bom_v13_unmatched_extractions,
    [_bom_v13_unmatched_description, _bom_v13_part_override],
)
_bom_v13_unmatched_passed, _, _bom_v13_unmatched_issues = (
    _validate_bom_page(
        page=_bom_v13_page,
        proposals=_bom_v13_proposals,
        detector=_bom_v13_detector,
        extractions=_bom_v13_unmatched_extractions,
        verifier=_bom_v13_unmatched_verifier,
        word_map=_bom_v13_word_map,
    )
)
assert _bom_v13_unmatched_passed is False, _bom_v13_unmatched_issues
assert any(
    issue.get('issue_type')
    == 'bom-visible-source-evidence-unrepresented'
    and issue.get('severity') == 'high'
    for issue in _bom_v13_unmatched_issues
), _bom_v13_unmatched_issues

# Two identical edge words are ambiguous evidence. No arbitrary word ID may be
# moved even when the character multiset would fit.
_bom_v13_ambiguous_extractions = _bom_v13_fixture_extractions()
_bom_v13_ambiguous_row = _bom_v13_ambiguous_extractions[0]['rows'][0]
_bom_v13_ambiguous_word_map = copy.deepcopy(_bom_v13_word_map)
_bom_v13_ambiguous_word_map[21] = {
    'id': 21,
    'text': 'E',
    'x0': 92.0,
    'y0': 10,
    'x1': 92.8,
    'y1': 15,
}
_bom_v13_ambiguous_row['source_word_ids'].append(21)
_bom_v13_ambiguous_row['field_evidence'][2]['source_word_ids'] = [
    21, 11, 12, 13, 14, 15
]
_bom_v13_ambiguous_row['part_number_original'] = (
    'E E 3SU 1400 -1AA10 -1BA0'
)
_bom_v13_ambiguous_row['part_number_normalized'] = (
    'EE3SU1400-1AA10-1BA0'
)
_bom_v13_ambiguous_part_override = {
    **_bom_v13_part_override,
    'approved_text': 'E3SU1400-1AA10-1BA0',
}
_bom_v13_ambiguous_verifier = copy.deepcopy(_bom_v13_verifier)
_bom_v13_ambiguous_verifier['field_overrides'] = [
    _bom_v13_description_override,
    _bom_v13_ambiguous_part_override,
]
_apply_bom_overrides(
    _bom_v13_ambiguous_extractions,
    [_bom_v13_description_override, _bom_v13_ambiguous_part_override],
)
_bom_v13_ambiguous_reconciliation = _bom_reconcile_post_override_rows(
    proposals=[],
    extractions=_bom_v13_ambiguous_extractions,
    word_map=_bom_v13_ambiguous_word_map,
)
assert _bom_v13_ambiguous_reconciliation[
    'cross_field_transfer_row_ids'
] == [], _bom_v13_ambiguous_reconciliation

# Phase 2B V1.4: exact source punctuation in a component tag is
# authoritative when the verifier emits only a compact alphanumeric sequence.
# The printed dot must be preserved; this is not permission to ignore arbitrary
# punctuation or any alphanumeric disagreement.
_bom_v14_word_map = copy.deepcopy(bom_word_map)
_bom_v14_word_map[1] = {
    **_bom_v14_word_map[1],
    'text': '35.AP1',
}
_bom_v14_extractions = _bom_fixture_extractions()
_bom_v14_row = _bom_v14_extractions[0]['rows'][0]
_bom_v14_row['component_tag_original'] = '35.AP1'
_bom_v14_row['component_tag_normalized'] = '35.AP1'
_bom_v14_verifier = copy.deepcopy(bom_verifier)
_bom_v14_verifier['verdict'] = 'review_required'
_bom_v14_verifier['all_published_fields_visually_supported'] = False
_bom_v14_verifier['all_source_evidence_represented'] = False
_bom_v14_verifier['region_checks'][0]['pass'] = False
_bom_v14_verifier['region_checks'][0][
    'verified_component_tag_sequence'
] = ['35AP1', 'K1']
_bom_v14_verifier['region_checks'][0]['notes'] = (
    'The verifier compacted source punctuation in one tag.'
)
_bom_v14_adjudication = _bom_tag_sequence_adjudication(
    actual_rows=_bom_v14_extractions[0]['rows'],
    verified_row_ids=['r1', 'r2'],
    verified_tags=['35AP1', 'K1'],
    word_map=_bom_v14_word_map,
)
assert _bom_v14_adjudication['validated'] is True, (
    _bom_v14_adjudication
)
assert _bom_v14_adjudication[
    'source_authoritative_row_ids'
] == ['r1'], _bom_v14_adjudication
_bom_v14_passed, _bom_v14_rows, _bom_v14_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=_bom_v14_extractions,
    verifier=_bom_v14_verifier,
    word_map=_bom_v14_word_map,
)
assert _bom_v14_passed is True, _bom_v14_issues
assert _bom_v14_rows[0]['component_tag_original'] == '35.AP1'
assert _bom_v14_rows[0]['component_tag_normalized'] == '35.AP1'
assert _bom_v14_rows[0]['component_tag_sequence_adjudication'][
    'validated'
] is True
assert any(
    issue.get('issue_type')
    == 'bom-verifier-component-tag-sequence-source-exact-adjudicated'
    and issue.get('severity') == 'info'
    for issue in _bom_v14_issues
), _bom_v14_issues
assert not [
    issue
    for issue in _bom_v14_issues
    if issue.get('severity') in {'high', 'critical'}
], _bom_v14_issues

# Any alphanumeric disagreement remains blocking even when punctuation also
# differs. Source authority cannot turn 35.AP1 into 35BP1.
_bom_v14_wrong_tag_verifier = copy.deepcopy(_bom_v14_verifier)
_bom_v14_wrong_tag_verifier['verdict'] = 'pass'
_bom_v14_wrong_tag_verifier['all_published_fields_visually_supported'] = True
_bom_v14_wrong_tag_verifier['all_source_evidence_represented'] = True
_bom_v14_wrong_tag_verifier['region_checks'][0]['pass'] = True
_bom_v14_wrong_tag_verifier['region_checks'][0][
    'verified_component_tag_sequence'
] = ['35BP1', 'K1']
_bom_v14_wrong_passed, _, _bom_v14_wrong_issues = _validate_bom_page(
    page=bom_page,
    proposals=bom_proposals,
    detector=bom_detector,
    extractions=copy.deepcopy(_bom_v14_extractions),
    verifier=_bom_v14_wrong_tag_verifier,
    word_map=_bom_v14_word_map,
)
assert _bom_v14_wrong_passed is False, _bom_v14_wrong_issues
assert any(
    issue.get('issue_type') == 'bom-verifier-row-sequence-mismatch'
    and issue.get('severity') == 'high'
    and issue.get('component_tag_sequence_adjudication', {}).get(
        'failure_reason'
    ).startswith('non_punctuation_component_tag_difference:')
    for issue in _bom_v14_wrong_issues
), _bom_v14_wrong_issues

# Punctuation cannot be invented by the materializer. If exact vector evidence
# says 35AP1 while the candidate says 35.AP1, source-evidence validation blocks.
_bom_v14_invented_word_map = copy.deepcopy(_bom_v14_word_map)
_bom_v14_invented_word_map[1]['text'] = '35AP1'
_bom_v14_invented_extractions = copy.deepcopy(_bom_v14_extractions)
_bom_v14_invented_verifier = copy.deepcopy(_bom_v14_wrong_tag_verifier)
_bom_v14_invented_verifier['region_checks'][0][
    'verified_component_tag_sequence'
] = ['35AP1', 'K1']
_bom_v14_invented_passed, _, _bom_v14_invented_issues = (
    _validate_bom_page(
        page=bom_page,
        proposals=bom_proposals,
        detector=bom_detector,
        extractions=_bom_v14_invented_extractions,
        verifier=_bom_v14_invented_verifier,
        word_map=_bom_v14_invented_word_map,
    )
)
assert _bom_v14_invented_passed is False, _bom_v14_invented_issues
assert any(
    issue.get('issue_type') == 'bom-verifier-row-sequence-mismatch'
    and issue.get('severity') == 'high'
    and issue.get('component_tag_sequence_adjudication', {}).get(
        'failure_reason'
    ).startswith('component_tag_not_exact_vector_source:')
    for issue in _bom_v14_invented_issues
), _bom_v14_invented_issues


# Phase 2B V1.5: when the verifier correctly removes one whole edge word
# from a donor field but omits the matching positive receiver override, exact
# adjacent-column geometry can complete the transfer. This reproduces the
# real failure mode generically without any page, language, code or word rule.
_bom_v15_extractions = _bom_v13_fixture_extractions()
_bom_v15_part_original_override = {
    'region_id': 'R-V13',
    'row_id': 'ROW-X',
    'field_name': 'part_number_original',
    'approved_text': '3SU1400-1AA10-1BA0',
    'confidence': 0.99,
    'reason': 'One whole edge word visibly belongs outside the code cell.',
}
_bom_v15_part_normalized_override = {
    'region_id': 'R-V13',
    'row_id': 'ROW-X',
    'field_name': 'part_number_normalized',
    'approved_text': '3SU1400-1AA10-1BA0',
    'confidence': 0.99,
    'reason': 'Readable code after the exact edge-word removal.',
}
_bom_v15_verifier = {
    'page_id': 902,
    'verdict': 'review_required',
    'all_visible_bom_tables_accounted_for': True,
    'all_visible_item_rows_accounted_for': True,
    'all_visible_columns_accounted_for': True,
    'all_published_fields_visually_supported': False,
    'all_source_evidence_represented': False,
    'duplicates_preserved': True,
    'region_checks': [{
        'region_id': 'R-V13',
        'expected_item_rows': 1,
        'verified_item_rows': 1,
        'verified_row_ids': ['ROW-X'],
        'verified_component_tag_sequence': ['30S1'],
        'pass': False,
        'confidence': 0.99,
        'notes': 'Pre-correction candidate contains one edge-word spill.',
    }],
    'field_overrides': [
        _bom_v15_part_original_override,
        _bom_v15_part_normalized_override,
    ],
    'missing_region_ids': [],
    'missing_row_ids': [],
    'duplicate_physical_keys': [],
    'unaccounted_visual_evidence': [],
    'confidence': 0.99,
    'issues': [{
        'issue_type': 'column_overflow',
        'severity': 'high',
        'message': 'One source edge word leaked into the code field.',
        'region_id': 'R-V13',
        'row_ids': ['ROW-X'],
        'confidence': 0.99,
        'resolution_status': 'resolved_by_exact_overrides',
        'related_overrides': [
            {'region_id': 'R-V13', 'row_id': 'ROW-X', 'field_name': 'part_number_original'},
            {'region_id': 'R-V13', 'row_id': 'ROW-X', 'field_name': 'part_number_normalized'},
        ],
    }],
}
_apply_bom_overrides(
    _bom_v15_extractions,
    [
        _bom_v15_part_original_override,
        _bom_v15_part_normalized_override,
    ],
)
_bom_v15_passed, _bom_v15_rows, _bom_v15_issues = _validate_bom_page(
    page=_bom_v13_page,
    proposals=_bom_v13_proposals,
    detector=_bom_v13_detector,
    extractions=_bom_v15_extractions,
    verifier=_bom_v15_verifier,
    word_map=_bom_v13_word_map,
)
assert _bom_v15_passed is True, _bom_v15_issues
assert _bom_v15_rows[0]['part_number_original'] == '3SU1400-1AA10-1BA0'
assert _bom_v15_rows[0]['part_number_normalized'] == '3SU1400-1AA10-1BA0'
assert _bom_v15_rows[0]['description_original'].endswith('FRONTAL E')
assert _bom_v15_rows[0]['description_normalized'].endswith('FRONTALE')
assert 11 in _bom_v15_rows[0]['field_evidence'][1]['source_word_ids']
assert 11 not in _bom_v15_rows[0]['field_evidence'][2]['source_word_ids']
assert _bom_v15_rows[0]['source_evidence_coverage']['complete'] is True
assert _bom_v15_rows[0]['cross_field_evidence_transfers'][0][
    'version'
] == 'one-sided-boundary-spill-transfer-v1'
assert not [
    issue
    for issue in _bom_v15_issues
    if issue.get('severity') in {'high', 'critical'}
], _bom_v15_issues

# If the removed word is not contiguous with the adjacent receiver, no
# transfer is allowed and source x-order must not silently reintroduce it.
_bom_v15_far_word_map = copy.deepcopy(_bom_v13_word_map)
_bom_v15_far_word_map[11] = {
    **_bom_v15_far_word_map[11],
    'x0': 150.0,
    'x1': 151.0,
}
_bom_v15_far_extractions = _bom_v13_fixture_extractions()
_apply_bom_overrides(
    _bom_v15_far_extractions,
    [
        _bom_v15_part_original_override,
        _bom_v15_part_normalized_override,
    ],
)
_bom_v15_far_reconciliation = _bom_reconcile_post_override_rows(
    proposals=[],
    extractions=_bom_v15_far_extractions,
    word_map=_bom_v15_far_word_map,
)
assert _bom_v15_far_reconciliation[
    'cross_field_transfer_row_ids'
] == [], _bom_v15_far_reconciliation
assert _bom_v15_far_extractions[0]['rows'][0][
    'part_number_original'
] == '3SU1400-1AA10-1BA0'
assert 'part_number_original' not in (
    _bom_v15_far_extractions[0]['rows'][0].get('deterministic_overrides') or {}
), _bom_v15_far_extractions
_bom_v15_far_audit = _bom_field_evidence_audit(
    row=_bom_v15_far_extractions[0]['rows'][0],
    expected_word_ids=list(range(1, 17)),
    word_map=_bom_v15_far_word_map,
)
assert _bom_v15_far_audit['complete'] is False, _bom_v15_far_audit

# Two eligible adjacent receiver fields are ambiguous. No arbitrary receiver
# may be selected, and the row must remain fail-closed.
_bom_v15_ambiguous_extractions = _bom_v13_fixture_extractions()
_bom_v15_ambiguous_row = _bom_v15_ambiguous_extractions[0]['rows'][0]
_bom_v15_ambiguous_row['item_position_original'] = 'MODULO DI CONTATTO, 1NO, MORSETTO A VITE, PIASTRA FRONTAL'
_bom_v15_ambiguous_row['item_position_normalized'] = _bom_v15_ambiguous_row['item_position_original']
_bom_v15_ambiguous_row['field_evidence'].append({
    'field_name': 'item_position_original',
    'source_column_index': 1,
    'source_word_ids': list(range(2, 11)),
})
# Duplicate assignment itself must prevent transfer before any arbitrary choice.
_apply_bom_overrides(
    _bom_v15_ambiguous_extractions,
    [
        _bom_v15_part_original_override,
        _bom_v15_part_normalized_override,
    ],
)
_bom_v15_ambiguous_reconciliation = _bom_reconcile_post_override_rows(
    proposals=[],
    extractions=_bom_v15_ambiguous_extractions,
    word_map=_bom_v13_word_map,
)
assert _bom_v15_ambiguous_reconciliation[
    'cross_field_transfer_row_ids'
] == [], _bom_v15_ambiguous_reconciliation


# Phase 2B V2: glyph/cell evidence is exact-character authority and is
# independent of PDF word grouping, content-stream order and font name.
def _v2_ids_in_rect(word_map, rect):
    out = []
    for wid, word in word_map.items():
        cx = (float(word['x0']) + float(word['x1'])) / 2.0
        cy = (float(word['y0']) + float(word['y1'])) / 2.0
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            out.append(int(wid))
    return sorted(out)

for _font_name in ['helv', 'Times-Roman', 'Courier']:
    _v2_doc = _bom_fitz.open()
    try:
        _v2_page = _v2_doc.new_page(width=520, height=120)
        _v2_y0, _v2_y1 = 30.0, 70.0
        _v2_edges = [10.0, 105.0, 345.0, 430.0, 510.0]
        for _edge in _v2_edges:
            _v2_page.draw_line((_edge, _v2_y0), (_edge, _v2_y1), width=0.5)
        _v2_page.draw_line((_v2_edges[0], _v2_y0), (_v2_edges[-1], _v2_y0), width=0.5)
        _v2_page.draw_line((_v2_edges[0], _v2_y1), (_v2_edges[-1], _v2_y1), width=0.5)
        _baseline = 55.0
        # Deliberately scramble content-stream insertion order. Physical x
        # geometry must still yield the same source text for every font.
        _v2_page.insert_text((292, _baseline), '********', fontsize=8, fontname=_font_name)
        _v2_page.insert_text((360, _baseline), '************', fontsize=8, fontname=_font_name)
        _v2_page.insert_text((112, _baseline), 'Alimentatore azionamenti', fontsize=8, fontname=_font_name)
        _v2_page.insert_text((445, _baseline), 'SIGMATEK', fontsize=8, fontname=_font_name)
        _v2_page.insert_text((18, _baseline), '200A1', fontsize=8, fontname=_font_name)
        _v2_words_raw = list(_v2_page.get_text('words', sort=True) or [])
        _v2_inventory = {'words': [list(word) for word in _v2_words_raw]}
        _v2_word_map = _bom_word_map_from_page(_v2_inventory)
        _v2_cells = []
        for _column in range(4):
            _rect = _bom_fitz.Rect(
                _v2_edges[_column], _v2_y0,
                _v2_edges[_column + 1], _v2_y1,
            )
            _ids = _v2_ids_in_rect(_v2_word_map, _rect)
            _v2_cells.append({
                'source_column_index': _column,
                'bbox_pt': [float(_rect.x0), float(_rect.y0), float(_rect.x1), float(_rect.y1)],
                'word_ids': _ids,
                'word_text_original': ' '.join(_v2_word_map[i]['text'] for i in _ids),
                'deterministic_cell_text_original': '',
                'geometry_source': 'fixture',
            })
        _v2_row_word_ids = sorted({
            wid for cell in _v2_cells for wid in cell['word_ids']
        })
        _v2_proposals = [{
            'region_id': 'V2-R1',
            'row_candidates': [{
                'source_row_candidate_id': 'R002',
                'source_row_index': 1,
                'bbox_pt': [_v2_edges[0], _v2_y0, _v2_edges[-1], _v2_y1],
                'word_ids': _v2_row_word_ids,
                'cells': _v2_cells,
            }],
        }]
        _v2_row = {
            'row_id': 'ROW_R002',
            'source_row_candidate_id': 'R002',
            'visual_order': 1,
            'component_tag_original': '200A1',
            'component_tag_normalized': '200A1',
            'description_original': 'Alimentatore azionamenti *******',
            'description_normalized': 'Alimentatore azionamenti *******',
            'part_number_original': '***********',
            'part_number_normalized': '***********',
            'manufacturer_original': 'ATEK SIGM',
            'manufacturer_normalized': 'SIGMATEK',
            'item_position_original': '', 'item_position_normalized': '',
            'quantity_text_original': '', 'quantity_text_normalized': '',
            'unit_original': '', 'unit_normalized': '',
            'field_evidence': [
                {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': _v2_cells[0]['word_ids']},
                {'field_name': 'description_original', 'source_column_index': 1, 'source_word_ids': _v2_cells[1]['word_ids']},
                {'field_name': 'part_number_original', 'source_column_index': 2, 'source_word_ids': _v2_cells[2]['word_ids']},
                {'field_name': 'manufacturer_original', 'source_column_index': 3, 'source_word_ids': _v2_cells[3]['word_ids']},
            ],
            'source_word_ids': _v2_row_word_ids,
            'bbox_pt': [_v2_edges[0], _v2_y0, _v2_edges[-1], _v2_y1],
            'confidence': 0.99,
        }
        _v2_extractions = [{
            'region_id': 'V2-R1',
            'source_column_roles': [
                {'source_column_index': 0, 'canonical_roles': ['component_tag'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 1, 'canonical_roles': ['description'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 2, 'canonical_roles': ['part_number'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 3, 'canonical_roles': ['manufacturer'], 'confidence': 0.99, 'reason': 'fixture'},
            ],
            'rows': [_v2_row],
        }]
        _v2_glyphs = _bom_glyph_map(_v2_page)
        _v2_audit = _bom_reconcile_glyph_cell_evidence(
            extractions=_v2_extractions,
            proposals=_v2_proposals,
            glyph_map=_v2_glyphs,
            word_map=_v2_word_map,
        )
        assert _v2_audit['field_count'] == 4, (_font_name, _v2_audit)
        assert _v2_row['description_original'] == 'Alimentatore azionamenti ********', (_font_name, _v2_row)
        assert _v2_row['part_number_original'] == '************', (_font_name, _v2_row)
        assert _v2_row['manufacturer_original'] == 'SIGMATEK', (_font_name, _v2_row)
        assert _v2_row['description_normalized'] == _v2_row['description_original']
        assert _v2_row['part_number_normalized'] == _v2_row['part_number_original']
        assert _v2_row['manufacturer_normalized'] == _v2_row['manufacturer_original']
        assert _bom_field_evidence_audit(
            row=_v2_row,
            expected_word_ids=_v2_row_word_ids,
            word_map=_v2_word_map,
        )['complete'] is True
    finally:
        _v2_doc.close()



# Phase 2B V2.1: adjacent rows own glyphs exclusively. Many CAD PDFs place
# each text baseline exactly on the lower row border. Baseline-inclusive row
# tests duplicated the upper-row text into the next row. Bbox-centre,
# half-open row ownership must work identically across common font metrics.
for _font_name in ['helv', 'Times-Roman', 'Courier']:
    _adj_doc = _bom_fitz.open()
    try:
        _adj_page = _adj_doc.new_page(width=360, height=100)
        _adj_edges = [10.0, 90.0, 220.0, 280.0, 350.0]
        _adj_y = [30.0, 50.0, 70.0]
        for _edge in _adj_edges:
            _adj_page.draw_line((_edge, _adj_y[0]), (_edge, _adj_y[-1]), width=0.5)
        for _y in _adj_y:
            _adj_page.draw_line((_adj_edges[0], _y), (_adj_edges[-1], _y), width=0.5)
        # The baseline equals each row's lower border, reproducing the real
        # shared-boundary failure without any page/tag/manufacturer special case.
        _adj_page.insert_text((18, 50.0), 'A1', fontsize=8, fontname=_font_name)
        _adj_page.insert_text((292, 50.0), 'BOSCH', fontsize=8, fontname=_font_name)
        _adj_page.insert_text((18, 70.0), 'A2', fontsize=8, fontname=_font_name)
        _adj_page.insert_text((292, 70.0), 'SIGMATEK', fontsize=8, fontname=_font_name)

        _adj_glyphs = _bom_glyph_map(_adj_page)
        _adj_words_raw = list(_adj_page.get_text('words', sort=True) or [])
        _adj_word_map = _bom_word_map_from_page({
            'words': [list(word) for word in _adj_words_raw],
        })

        def _adj_candidate(candidate_id, y0, y1):
            cells = []
            for column in range(4):
                rect = _bom_fitz.Rect(
                    _adj_edges[column], y0,
                    _adj_edges[column + 1], y1,
                )
                ids = _v2_ids_in_rect(_adj_word_map, rect)
                cells.append({
                    'source_column_index': column,
                    'bbox_pt': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    'word_ids': ids,
                    'word_text_original': ' '.join(_adj_word_map[i]['text'] for i in ids),
                    'deterministic_cell_text_original': '',
                    'geometry_source': 'shared-boundary-fixture',
                })
            return {
                'source_row_candidate_id': candidate_id,
                'bbox_pt': [_adj_edges[0], y0, _adj_edges[-1], y1],
                'word_ids': sorted({wid for cell in cells for wid in cell['word_ids']}),
                'cells': cells,
            }

        _adj_candidate_1 = _adj_candidate('R002', 30.0, 50.0)
        _adj_candidate_2 = _adj_candidate('R003', 50.0, 70.0)
        _adj_assignment_1 = _bom_assign_row_glyphs(
            row_candidate=_adj_candidate_1,
            glyph_map=_adj_glyphs,
        )
        _adj_assignment_2 = _bom_assign_row_glyphs(
            row_candidate=_adj_candidate_2,
            glyph_map=_adj_glyphs,
        )
        assert _adj_assignment_1['row_ownership_version'] == 'glyph-row-center-exclusive-v1'
        assert _adj_assignment_2['row_ownership_version'] == 'glyph-row-center-exclusive-v1'
        assert _adj_assignment_1['cells']['3']['source_text_physical_order'] == 'BOSCH', (
            _font_name, _adj_assignment_1
        )
        assert _adj_assignment_2['cells']['3']['source_text_physical_order'] == 'SIGMATEK', (
            _font_name, _adj_assignment_2
        )
        assert not (
            set(_adj_assignment_1['row_glyph_ids'])
            & set(_adj_assignment_2['row_glyph_ids'])
        ), (_font_name, _adj_assignment_1, _adj_assignment_2)

        _adj_rows = [
            {
                'row_id': 'ROW_R002',
                'source_row_candidate_id': 'R002',
                'visual_order': 1,
                'component_tag_original': 'A1',
                'component_tag_normalized': 'A1',
                'description_original': '', 'description_normalized': '',
                'part_number_original': '', 'part_number_normalized': '',
                'manufacturer_original': 'BOSCH',
                'manufacturer_normalized': 'BOSCH',
                'item_position_original': '', 'item_position_normalized': '',
                'quantity_text_original': '', 'quantity_text_normalized': '',
                'unit_original': '', 'unit_normalized': '',
                'field_evidence': [
                    {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': _adj_candidate_1['cells'][0]['word_ids']},
                    {'field_name': 'manufacturer_original', 'source_column_index': 3, 'source_word_ids': _adj_candidate_1['cells'][3]['word_ids']},
                ],
                'source_word_ids': _adj_candidate_1['word_ids'],
                'bbox_pt': _adj_candidate_1['bbox_pt'],
                'confidence': 0.99,
            },
            {
                'row_id': 'ROW_R003',
                'source_row_candidate_id': 'R003',
                'visual_order': 2,
                'component_tag_original': 'A2',
                'component_tag_normalized': 'A2',
                'description_original': '', 'description_normalized': '',
                'part_number_original': '', 'part_number_normalized': '',
                'manufacturer_original': 'ATEK SIGM',
                'manufacturer_normalized': 'ATEKSIGM',
                'item_position_original': '', 'item_position_normalized': '',
                'quantity_text_original': '', 'quantity_text_normalized': '',
                'unit_original': '', 'unit_normalized': '',
                'field_evidence': [
                    {'field_name': 'component_tag_original', 'source_column_index': 0, 'source_word_ids': _adj_candidate_2['cells'][0]['word_ids']},
                    {'field_name': 'manufacturer_original', 'source_column_index': 3, 'source_word_ids': _adj_candidate_2['cells'][3]['word_ids']},
                ],
                'source_word_ids': _adj_candidate_2['word_ids'],
                'bbox_pt': _adj_candidate_2['bbox_pt'],
                'confidence': 0.99,
            },
        ]
        _adj_extractions = [{
            'region_id': 'ADJ-R1',
            'source_column_roles': [
                {'source_column_index': 0, 'canonical_roles': ['component_tag'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 1, 'canonical_roles': ['description'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 2, 'canonical_roles': ['part_number'], 'confidence': 0.99, 'reason': 'fixture'},
                {'source_column_index': 3, 'canonical_roles': ['manufacturer'], 'confidence': 0.99, 'reason': 'fixture'},
            ],
            'rows': _adj_rows,
        }]
        _adj_audit = _bom_reconcile_glyph_cell_evidence(
            extractions=_adj_extractions,
            proposals=[{
                'region_id': 'ADJ-R1',
                'row_candidates': [_adj_candidate_1, _adj_candidate_2],
            }],
            glyph_map=_adj_glyphs,
            word_map=_adj_word_map,
        )
        assert _adj_audit['validated'] is True, (_font_name, _adj_audit)
        assert _adj_audit['validated_row_count'] == 2, (_font_name, _adj_audit)
        assert _adj_audit['unvalidated_row_ids'] == [], (_font_name, _adj_audit)
        assert _adj_rows[1]['manufacturer_original'] == 'SIGMATEK', (
            _font_name, _adj_rows[1]
        )
    finally:
        _adj_doc.close()

# Full V2 gate: a verifier may return review_required for its own wrong mask
# count; exact glyph-cell evidence must correct the row and supersede only the
# stale pre-correction flags, without weakening any structural gate.
# Reuse the last synthetic font fixture objects still in scope.
_v2_gate_page = {'id': 902, 'pdf_page_number': 69, 'page_width_pt': 520, 'page_height_pt': 120}
_v2_gate_extractions = copy.deepcopy(_v2_extractions)
_v2_gate_extractions[0].update({
    'page_id': 902,
    'header_row_candidate_ids': [],
    'non_item_rows': [],
    'unaccounted_row_candidate_ids': [],
    'duplicate_row_ids': [],
    'confidence': 0.99,
    'issues': [],
})
_v2_gate_row = _v2_gate_extractions[0]['rows'][0]
_v2_gate_row.update({
    'row_role': 'item',
    'description_original': 'Alimentatore azionamenti *******',
    'description_normalized': 'Alimentatore azionamenti *******',
    'part_number_original': '***********',
    'part_number_normalized': '***********',
    'manufacturer_original': 'ATEK SIGM',
    'manufacturer_normalized': 'SIGMATEK',
    'evidence_notes': 'fixture',
})
_v2_gate_detector = {
    'page_id': 902,
    'all_visible_bom_tables_accounted_for': True,
    'missing_visible_bom_tables': [],
    'confidence': 0.99,
    'issues': [],
    'proposal_assessments': [{
        'region_id': 'V2-R1', 'visible': True, 'distinct_table': True,
        'kind': 'bom_table', 'expected_header_rows': 0,
        'expected_item_rows': 1, 'expected_column_count': 4,
        'confidence': 0.99, 'notes': 'fixture',
    }],
}
_v2_gate_verifier = {
    'page_id': 902,
    'verdict': 'review_required',
    'all_visible_bom_tables_accounted_for': True,
    'all_visible_item_rows_accounted_for': True,
    'all_visible_columns_accounted_for': True,
    'all_published_fields_visually_supported': False,
    'all_source_evidence_represented': False,
    'duplicates_preserved': True,
    'region_checks': [{
        'region_id': 'V2-R1', 'expected_item_rows': 1,
        'verified_item_rows': 1, 'verified_row_ids': ['ROW001'],
        'verified_component_tag_sequence': ['200A1'], 'pass': False,
        'confidence': 0.99, 'notes': 'Wrong visual mask count before source-exact reconciliation.',
    }],
    'field_overrides': [
        {'region_id': 'V2-R1', 'row_id': 'ROW001', 'field_name': 'description_original',
         'approved_text': 'Alimentatore azionamenti *******', 'confidence': 0.99, 'reason': 'visual estimate'},
        {'region_id': 'V2-R1', 'row_id': 'ROW001', 'field_name': 'part_number_original',
         'approved_text': '***********', 'confidence': 0.99, 'reason': 'visual estimate'},
        {'region_id': 'V2-R1', 'row_id': 'ROW001', 'field_name': 'manufacturer_original',
         'approved_text': 'SIGMATEK', 'confidence': 0.99, 'reason': 'visual reading'},
    ],
    'missing_region_ids': [], 'missing_row_ids': [],
    'duplicate_physical_keys': [], 'unaccounted_visual_evidence': [],
    'confidence': 0.99,
    'issues': [{
        'issue_type': 'masked_code_star_count_uncertain', 'severity': 'warning',
        'message': 'Visual mask count uncertain.', 'region_id': 'V2-R1',
        'row_ids': ['ROW001'], 'confidence': 0.75,
        'resolution_status': 'open', 'related_overrides': [],
    }],
}
_v2_gate_alias = _bom_canonicalize_verifier_rows(
    extractions=_v2_gate_extractions,
    verifier=_v2_gate_verifier,
)
assert _v2_gate_alias['mappings'], _v2_gate_alias
_apply_bom_overrides(
    _v2_gate_extractions,
    _v2_gate_verifier['field_overrides'],
)
_v2_gate_passed, _v2_gate_rows, _v2_gate_issues = _validate_bom_page(
    page=_v2_gate_page,
    proposals=_v2_proposals,
    detector=_v2_gate_detector,
    extractions=_v2_gate_extractions,
    verifier=_v2_gate_verifier,
    word_map=_v2_word_map,
    glyph_map=_v2_glyphs,
)
assert _v2_gate_passed is True, _v2_gate_issues
assert _v2_gate_rows[0]['description_original'] == 'Alimentatore azionamenti ********'
assert _v2_gate_rows[0]['part_number_original'] == '************'
assert _v2_gate_rows[0]['manufacturer_original'] == 'SIGMATEK'
assert not [
    issue for issue in _v2_gate_issues
    if issue.get('severity') in {'high', 'critical'}
], _v2_gate_issues
assert any(
    issue.get('issue_type') == 'bom-verifier-verdict-superseded-post-override'
    for issue in _v2_gate_issues
), _v2_gate_issues

# A vector row with a non-exclusive or incomplete glyph ledger must now block
# at the endpoint gate instead of passing and being caught only by SQL later.
_v21_bad_glyphs = copy.deepcopy(_v2_glyphs)
_v21_bad_id = max(_v21_bad_glyphs) + 1
_v21_bad_glyphs[_v21_bad_id] = {
    'id': _v21_bad_id, 'text': 'Z',
    'x0': 470.0, 'y0': 45.0, 'x1': 475.0, 'y1': 55.0,
    'origin_x': 470.0, 'origin_y': 55.0,
    'block_no': 999, 'line_no': 999, 'span_no': 0, 'char_no': 0,
    'dir_x': 1.0, 'dir_y': 0.0, 'font_audit': '', 'size_audit': 8.0,
}
_v21_bad_extractions = copy.deepcopy(_v2_gate_extractions)
_v21_bad_passed, _, _v21_bad_issues = _validate_bom_page(
    page=_v2_gate_page,
    proposals=_v2_proposals,
    detector=_v2_gate_detector,
    extractions=_v21_bad_extractions,
    verifier=copy.deepcopy(_v2_gate_verifier),
    word_map=_v2_word_map,
    glyph_map=_v21_bad_glyphs,
)
assert _v21_bad_passed is False, _v21_bad_issues
assert any(
    issue.get('issue_type') == 'bom-glyph-row-evidence-incomplete'
    and issue.get('severity') == 'high'
    for issue in _v21_bad_issues
), _v21_bad_issues

# Raster/outlined sources without extractable characters remain fail-closed:
# V2 never invents exact mask counts when no glyph ledger exists.
_v2_no_glyph_row = copy.deepcopy(_v2_row)
_v2_no_glyph_row['description_original'] = 'Alimentatore azionamenti *******'
_v2_no_glyph_row['description_normalized'] = 'Alimentatore azionamenti *******'
_v2_no_glyph_extractions = copy.deepcopy(_v2_extractions)
_v2_no_glyph_extractions[0]['rows'] = [_v2_no_glyph_row]
_v2_no_glyph_audit = _bom_reconcile_glyph_cell_evidence(
    extractions=_v2_no_glyph_extractions,
    proposals=_v2_proposals,
    glyph_map={},
    word_map=_v2_word_map,
)
assert _v2_no_glyph_audit['field_count'] == 0, _v2_no_glyph_audit
assert _v2_no_glyph_row['description_original'].endswith('*******')

# More than one populated canonical owner in one physical cell is ambiguous.
_v2_ambiguous_extractions = copy.deepcopy(_v2_extractions)
_v2_ambiguous_row = _v2_ambiguous_extractions[0]['rows'][0]
_v2_ambiguous_row['item_position_original'] = 'EXTRA'
_v2_ambiguous_row['item_position_normalized'] = 'EXTRA'
_v2_ambiguous_row['field_evidence'].append({
    'field_name': 'item_position_original',
    'source_column_index': 1,
    'source_word_ids': _v2_cells[1]['word_ids'],
})
_v2_ambiguous_extractions[0]['source_column_roles'][1]['canonical_roles'] = [
    'description', 'item_position'
]
_v2_ambiguous_audit = _bom_reconcile_glyph_cell_evidence(
    extractions=_v2_ambiguous_extractions,
    proposals=_v2_proposals,
    glyph_map=_v2_glyphs,
    word_map=_v2_word_map,
)
assert not any(
    item.get('field_name') in {'description_original', 'item_position_original'}
    for item in _v2_ambiguous_audit['audits']
), _v2_ambiguous_audit

# Verifier-local row labels are mapped only when complete physical order and
# tag sequence agree.
_v2_alias_extractions = [{
    'region_id': 'V2-R1',
    'rows': [
        {'row_id': 'ROW_R002', 'visual_order': 1, 'component_tag_original': 'K1'},
        {'row_id': 'ROW_R003', 'visual_order': 2, 'component_tag_original': 'K2'},
    ],
}]
_v2_alias_verifier = {
    'region_checks': [{
        'region_id': 'V2-R1',
        'verified_row_ids': ['ROW001', 'ROW002'],
        'verified_component_tag_sequence': ['K1', 'K2'],
    }],
    'field_overrides': [{
        'region_id': 'V2-R1', 'row_id': 'ROW002',
        'field_name': 'description_original', 'approved_text': 'X',
        'confidence': 0.99, 'reason': 'fixture',
    }],
    'issues': [], 'missing_row_ids': [],
}
_v2_alias_audit = _bom_canonicalize_verifier_rows(
    extractions=_v2_alias_extractions,
    verifier=_v2_alias_verifier,
)
assert _v2_alias_audit['mappings'], _v2_alias_audit
assert _v2_alias_verifier['region_checks'][0]['verified_row_ids'] == [
    'ROW_R002', 'ROW_R003'
]
assert _v2_alias_verifier['field_overrides'][0]['row_id'] == 'ROW_R003'


# Phase 2G V1: font-independent character evidence, exact reference
# resolution and fail-closed electrical topology validation.
from electrical_graph import (
    GRAPH_PATCH_PLAN_VERSION as _graph_patch_plan_version,
    _apply_graph_patch_plan as _graph_apply_patch_plan,
    _apply_verifier_evidence_recovery as _graph_apply_recovery,
    _edge_bbox_valid as _graph_edge_bbox_valid,
    _glyph_registry as _graph_glyph_registry,
    _reconcile_graph_geometry_from_evidence as _graph_reconcile_geometry,
    _resolve_references as _graph_resolve_references,
    _resolve_references_for_verifier_v1 as _graph_resolve_for_verifier_v1,
    _review_cause_family_counts as _graph_review_cause_family_counts,
    _validate_candidate_graph as _graph_validate_candidate,
    _validate_patched_graph as _graph_validate_patched,
    _verifier_schema as _graph_v3_verifier_schema,
)

_graph_font_texts = []
for _font_name in ('helv', 'times-roman', 'cour'):
    _graph_doc = _bom_fitz.open()
    try:
        _graph_page_obj = _graph_doc.new_page(width=220, height=80)
        _graph_page_obj.insert_text(
            (15, 35),
            'K1 35.AP1 ********',
            fontsize=11,
            fontname=_font_name,
        )
        _glyph_rows = _graph_glyph_registry(_graph_page_obj)
        _glyph_rows.sort(key=lambda item: item['bbox_pt'][0])
        _graph_font_texts.append(''.join(
            item['text_original'] for item in _glyph_rows
        ))
    finally:
        _graph_doc.close()
assert _graph_font_texts == [
    'K1 35.AP1 ********',
    'K1 35.AP1 ********',
    'K1 35.AP1 ********',
], _graph_font_texts

_graph_registry = {
    'bom': [
        {'id': 1, 'page_id': 66, 'component_tag': 'K1', 'manufacturer': 'A', 'part_number': 'P1', 'description': 'Relay', 'confidence': 0.99},
        {'id': 2, 'page_id': 66, 'component_tag': 'K1', 'manufacturer': 'A', 'part_number': 'P2', 'description': 'Accessory', 'confidence': 0.98},
    ],
    'io': [
        {'id': 10, 'page_id': 51, 'module_tag': 'M1', 'channel_ref': '8', 'plc_address': '', 'io_type': 'safety_input', 'is_safety': True, 'signal_name': 'Reset', 'description': 'Reset', 'wire_reference': '33/1', 'terminal_reference': '', 'confidence': 0.98},
    ],
    'terminals': [
        {'id': 20, 'page_id': 62, 'strip_tag': 'X1', 'terminal_number': '9', 'level_ref': '', 'side_a_origin': '', 'side_b_destination': '', 'wire_number': '33/1', 'cable_reference': '', 'potential': '', 'confidence': 0.98},
    ],
    'pages': [
        {'id': 30, 'pdf_page_number': 44, 'sheet_code': '205', 'sheet_title': 'Axis', 'page_type': 'schematic'},
    ],
    'cross_references': [],
}
_graph_entities = [
    {'occurrence_id': 'C1', 'region_id': 'R1', 'entity_type': 'component_occurrence', 'subtype': 'relay', 'tag_original': 'K1', 'label_original': '', 'description_original': '', 'function_text_original': '', 'symbol_code': '', 'location_code': '', 'reference_value_original': '', 'reference_context_original': '', 'bbox_pt': [10, 10, 20, 20], 'source_glyph_ids': [1], 'source_word_ids': [1], 'confidence': 0.98, 'evidence_notes': 'fixture'},
    {'occurrence_id': 'C2', 'region_id': 'R1', 'entity_type': 'coil', 'subtype': 'coil', 'tag_original': 'K1', 'label_original': '', 'description_original': '', 'function_text_original': '', 'symbol_code': '', 'location_code': '', 'reference_value_original': '', 'reference_context_original': '', 'bbox_pt': [40, 10, 50, 20], 'source_glyph_ids': [2], 'source_word_ids': [2], 'confidence': 0.98, 'evidence_notes': 'fixture'},
    {'occurrence_id': 'IO1', 'region_id': 'R1', 'entity_type': 'io_reference', 'subtype': '', 'tag_original': 'M1', 'label_original': '', 'description_original': '', 'function_text_original': '', 'symbol_code': '', 'location_code': '', 'reference_value_original': '8', 'reference_context_original': 'M1/8', 'bbox_pt': [10, 30, 20, 40], 'source_glyph_ids': [3], 'source_word_ids': [3], 'confidence': 0.98, 'evidence_notes': 'fixture'},
    {'occurrence_id': 'T1', 'region_id': 'R1', 'entity_type': 'terminal_reference', 'subtype': '', 'tag_original': 'X1', 'label_original': '', 'description_original': '', 'function_text_original': '', 'symbol_code': '', 'location_code': '', 'reference_value_original': '9', 'reference_context_original': 'X1-9', 'bbox_pt': [30, 30, 40, 40], 'source_glyph_ids': [4], 'source_word_ids': [4], 'confidence': 0.98, 'evidence_notes': 'fixture'},
    {'occurrence_id': 'P1', 'region_id': 'R1', 'entity_type': 'page_reference', 'subtype': '', 'tag_original': '205', 'label_original': '', 'description_original': '', 'function_text_original': '', 'symbol_code': '', 'location_code': '', 'reference_value_original': '', 'reference_context_original': '205.4', 'bbox_pt': [50, 30, 60, 40], 'source_glyph_ids': [5], 'source_word_ids': [5], 'confidence': 0.98, 'evidence_notes': 'fixture'},
]
_graph_resolution = _graph_resolve_references(
    {'entities': _graph_entities},
    _graph_registry,
)
assert _graph_resolution['all_reference_entities_resolved'] is True, _graph_resolution
assert _graph_resolution['match_counts'] == {
    'bom': 4, 'io': 1, 'terminal': 1, 'page': 1
}, _graph_resolution

_graph_edges = [{
    'edge_id': 'E1',
    'source_occurrence_id': 'C1',
    'target_occurrence_id': 'C2',
    'relation_type': 'electrically_connected_to',
    'is_directed': False,
    'potential_original': '',
    'wire_reference_original': '33/1',
    'bbox_pt': [20, 10, 40, 20],
    'source_glyph_ids': [],
    'source_drawing_ids': [1],
    'source_link_ids': [],
    'confidence': 0.98,
    'evidence_notes': 'visible conductor',
}]
_graph_extraction = {
    'page_id': 1,
    'entities': _graph_entities,
    'edges': _graph_edges,
    'unresolved_visual_evidence': [],
    'confidence': 0.98,
    'issues': [],
}
_graph_detector = {
    'page_id': 1,
    'all_visible_circuit_regions_accounted_for': True,
    'regions': [{
        'region_id': 'R1', 'region_kind': 'mixed_circuit',
        'bbox_pt': [0, 0, 100, 100],
        'visible_component_count': 5,
        'visible_connection_count': 1,
        'confidence': 0.98, 'notes': 'fixture',
    }],
    'uncovered_visual_regions': [],
    'confidence': 0.98,
    'issues': [],
}
_graph_verifier = {
    'page_id': 1,
    'verdict': 'pass',
    'all_visible_entities_accounted_for': True,
    'all_visible_connections_accounted_for': True,
    'all_entity_text_visually_supported': True,
    'all_connection_geometry_supported': True,
    'all_references_resolved_or_explicitly_unresolved': True,
    'duplicates_preserved': True,
    'verified_entity_ids': ['C1', 'C2', 'IO1', 'T1', 'P1'],
    'verified_edge_ids': ['E1'],
    'rejected_entity_ids': [],
    'rejected_edge_ids': [],
    'confidence': 0.98,
    'issues': [],
}
_graph_page = {
    'id': 1,
    'page_width_pt': 100,
    'page_height_pt': 100,
}
_graph_glyphs = [
    {'glyph_id': i, 'text_original': str(i), 'bbox_pt': [i, 1, i + 1, 2]}
    for i in range(1, 6)
]
_graph_words = [
    {'word_id': i, 'text_original': str(i), 'bbox_pt': [i, 1, i + 1, 2]}
    for i in range(1, 6)
]
_graph_drawings = [{'drawing_id': 1, 'bbox_pt': [20, 10, 40, 20]}]
_graph_passed, _, _, _graph_issues = _graph_validate_candidate(
    page=_graph_page,
    detector=_graph_detector,
    extraction=_graph_extraction,
    verifier=_graph_verifier,
    resolution=_graph_resolution,
    glyphs=_graph_glyphs,
    words=_graph_words,
    drawings=_graph_drawings,
    links=[],
)
assert _graph_passed is True, _graph_issues

_graph_no_geometry = copy.deepcopy(_graph_extraction)
_graph_no_geometry['edges'][0]['source_drawing_ids'] = []
_graph_failed, _, _, _graph_failed_issues = _graph_validate_candidate(
    page=_graph_page,
    detector=_graph_detector,
    extraction=_graph_no_geometry,
    verifier=_graph_verifier,
    resolution=_graph_resolution,
    glyphs=_graph_glyphs,
    words=_graph_words,
    drawings=_graph_drawings,
    links=[],
)
assert _graph_failed is False, _graph_failed_issues
assert any(
    issue.get('issue_type') == 'graph-edge-geometry-evidence-missing'
    for issue in _graph_failed_issues
), _graph_failed_issues

# Phase 2G V1.1: conductor paths can be vertical or horizontal line segments.
assert _graph_edge_bbox_valid([20, 10, 20, 70], _graph_page) is True
assert _graph_edge_bbox_valid([20, 10, 70, 10], _graph_page) is True
assert _graph_edge_bbox_valid([20, 10, 20, 10], _graph_page) is False
assert _graph_edge_bbox_valid([70, 10, 20, 10], _graph_page) is False

# Final materialization resolves printed sheet/grid references locally, while
# the verifier request keeps the exact V1 projection for artifact reuse.
_graph_page_ref_entity = {
    'occurrence_id': 'P-GRID',
    'region_id': 'R1',
    'entity_type': 'page_reference',
    'subtype': 'cross_reference',
    'tag_original': 'K9',
    'label_original': '205.3',
    'description_original': '',
    'function_text_original': '',
    'symbol_code': '',
    'location_code': '',
    'reference_value_original': '205.3',
    'reference_context_original': 'a',
    'bbox_pt': [60, 30, 75, 40],
    'source_glyph_ids': [5],
    'source_word_ids': [5],
    'confidence': 0.98,
    'evidence_notes': 'fixture',
}
_graph_page_ref_registry = {
    **_graph_registry,
    'pages': [
        {'id': 30, 'pdf_page_number': 44, 'sheet_code': '205',
         'sheet_title': 'Axis', 'page_type': 'schematic'},
    ],
    'cross_references': [],
}
_graph_final_ref = _graph_resolve_references(
    {'entities': [_graph_page_ref_entity]},
    _graph_page_ref_registry,
)
assert _graph_final_ref['all_reference_entities_resolved'] is True, (
    _graph_final_ref
)
assert _graph_final_ref['match_counts']['page'] == 1, _graph_final_ref
_graph_legacy_ref = _graph_resolve_for_verifier_v1(
    {'entities': [_graph_page_ref_entity]},
    _graph_page_ref_registry,
)
assert _graph_legacy_ref['version'] == 'exact-certified-reference-resolution-v1'
assert _graph_legacy_ref['all_reference_entities_resolved'] is False, (
    _graph_legacy_ref
)

# A visible I/O reference absent from the certified registry is preserved as
# explicitly unresolved. A false topology edge is pruned, the verified local
# conductor remains, and the raw review verdict is superseded only after the
# final projection passes every deterministic check.
_graph_ref_entities = [
    copy.deepcopy(_graph_entities[0]),
    {
        'occurrence_id': 'IO-MISSING',
        'region_id': 'R1',
        'entity_type': 'io_reference',
        'subtype': 'safety_output',
        'tag_original': 'MISSING-MODULE',
        'label_original': 'MISSING-MODULE',
        'description_original': 'Visible reference',
        'function_text_original': 'OUTPUT',
        'symbol_code': '',
        'location_code': '',
        'reference_value_original': '1',
        'reference_context_original': 'sheet/grid',
        'bbox_pt': [40, 10, 55, 20],
        'source_glyph_ids': [2],
        'source_word_ids': [2],
        'confidence': 0.98,
        'evidence_notes': 'fixture',
    },
]
_graph_ref_edges = [
    {
        'edge_id': 'E-KEEP',
        'source_occurrence_id': 'C1',
        'target_occurrence_id': 'IO-MISSING',
        'relation_type': 'electrically_connected_to',
        'is_directed': False,
        'potential_original': '',
        'wire_reference_original': 'W1',
        'bbox_pt': [20, 15, 40, 15],
        'source_glyph_ids': [],
        'source_drawing_ids': [1],
        'source_link_ids': [],
        'confidence': 0.98,
        'evidence_notes': 'visible conductor',
    },
    {
        'edge_id': 'E-REMOVE',
        'source_occurrence_id': 'C1',
        'target_occurrence_id': 'IO-MISSING',
        'relation_type': 'controls',
        'is_directed': True,
        'potential_original': '',
        'wire_reference_original': '',
        'bbox_pt': [20, 10, 40, 20],
        'source_glyph_ids': [],
        'source_drawing_ids': [],
        'source_link_ids': [9],
        'confidence': 0.95,
        'evidence_notes': 'not local conductor geometry',
    },
]
_graph_ref_extraction = {
    'page_id': 1,
    'entities': _graph_ref_entities,
    'edges': _graph_ref_edges,
    'unresolved_visual_evidence': [
        'Reference is printed but absent from certified registry.'
    ],
    'confidence': 0.98,
    'issues': [],
}
_graph_ref_resolution = _graph_resolve_references(
    _graph_ref_extraction,
    {**_graph_registry, 'io': []},
)
assert _graph_ref_resolution[
    'all_reference_entities_resolved_or_explicitly_unresolved'
] is True, _graph_ref_resolution
assert _graph_ref_resolution['unresolved_reference_entity_ids'] == [
    'IO-MISSING'
], _graph_ref_resolution
_graph_ref_verifier = {
    'page_id': 1,
    'verdict': 'review_required',
    'all_visible_entities_accounted_for': True,
    'all_visible_connections_accounted_for': True,
    'all_entity_text_visually_supported': False,
    'all_connection_geometry_supported': False,
    'all_references_resolved_or_explicitly_unresolved': True,
    'duplicates_preserved': True,
    'verified_entity_ids': ['C1'],
    'verified_edge_ids': ['E-KEEP'],
    'rejected_entity_ids': ['IO-MISSING'],
    'rejected_edge_ids': ['E-REMOVE'],
    'recovery_entities': [],
    'recovery_edges': [],
    'visual_evidence_adjudications': [{
        'evidence_index': 0,
        'evidence_text_original': (
            'Reference is printed but absent from certified registry.'
        ),
        'status': 'accounted_existing_graph',
        'related_entity_ids': ['IO-MISSING'],
        'related_edge_ids': ['E-KEEP'],
        'confidence': 0.98,
        'reason': (
            'The visible reference occurrence and its surviving conductor are '
            'already represented; only the external registry match is absent.'
        ),
    }],
    'confidence': 0.96,
    'issues': [
        {
            'issue_type': 'certified_registry_mismatch',
            'severity': 'high',
            'message': 'Visible reference has no certified registry row.',
            'entity_ids': ['IO-MISSING'],
            'edge_ids': ['E-KEEP'],
            'confidence': 0.96,
        },
        {
            'issue_type': 'topology_ambiguity',
            'severity': 'high',
            'message': 'Functional edge is not locally drawn.',
            'entity_ids': ['C1'],
            'edge_ids': ['E-REMOVE'],
            'confidence': 0.96,
        },
    ],
}
_graph_ref_detector = copy.deepcopy(_graph_detector)
_graph_ref_detector['regions'][0] = {
    **_graph_ref_detector['regions'][0],
    'region_kind': 'off_page_reference',
    'visible_connection_count': 0,
    'confidence': 0.86,
}
_graph_ref_passed, _graph_ref_final_entities, _graph_ref_final_edges, (
    _graph_ref_issues
) = _graph_validate_candidate(
    page=_graph_page,
    detector=_graph_ref_detector,
    extraction=_graph_ref_extraction,
    verifier=_graph_ref_verifier,
    resolution=_graph_ref_resolution,
    glyphs=_graph_glyphs,
    words=_graph_words,
    drawings=_graph_drawings,
    links=[{'id': 9, 'bbox_pt': [20, 10, 40, 20]}],
)
assert _graph_ref_passed is True, _graph_ref_issues
assert [x['occurrence_id'] for x in _graph_ref_final_entities] == [
    'C1', 'IO-MISSING'
]
assert [x['edge_id'] for x in _graph_ref_final_edges] == ['E-KEEP']
assert _graph_ref_extraction['post_verifier_adjudication'][
    'preserved_unresolved_reference_ids'
] == ['IO-MISSING']
assert _graph_ref_extraction['post_verifier_adjudication'][
    'removed_edge_ids'
] == ['E-REMOVE']
assert not [
    issue for issue in _graph_ref_issues
    if issue.get('severity') in {'high', 'critical'}
], _graph_ref_issues

# A rejected non-reference entity required by a surviving verified edge cannot
# be silently removed.
_graph_bad_entity_verifier = copy.deepcopy(_graph_verifier)
_graph_bad_entity_verifier.update({
    'verdict': 'review_required',
    'verified_entity_ids': ['C1', 'IO1', 'T1', 'P1'],
    'rejected_entity_ids': ['C2'],
    'verified_edge_ids': ['E1'],
    'rejected_edge_ids': [],
    'issues': [{
        'issue_type': 'entity_not_supported',
        'severity': 'high',
        'message': 'Entity is not visually supported.',
        'entity_ids': ['C2'],
        'edge_ids': ['E1'],
        'confidence': 0.98,
    }],
})
_graph_bad_entity_passed, _, _, _graph_bad_entity_issues = (
    _graph_validate_candidate(
        page=_graph_page,
        detector=_graph_detector,
        extraction=copy.deepcopy(_graph_extraction),
        verifier=_graph_bad_entity_verifier,
        resolution=_graph_resolution,
        glyphs=_graph_glyphs,
        words=_graph_words,
        drawings=_graph_drawings,
        links=[],
    )
)
assert _graph_bad_entity_passed is False, _graph_bad_entity_issues
assert any(
    issue.get('issue_type')
    == 'graph-rejected-entity-required-by-verified-edge'
    for issue in _graph_bad_entity_issues
), _graph_bad_entity_issues

# Unresolved visual evidence remains blocking when independent coverage flags
# are not all true.
_graph_unaccounted_extraction = copy.deepcopy(_graph_extraction)
_graph_unaccounted_extraction['unresolved_visual_evidence'] = [
    'Unaccounted conductor.'
]
_graph_unaccounted_verifier = copy.deepcopy(_graph_verifier)
_graph_unaccounted_verifier['all_visible_connections_accounted_for'] = False
_graph_unaccounted_verifier['verdict'] = 'review_required'
_graph_unaccounted_passed, _, _, _graph_unaccounted_issues = (
    _graph_validate_candidate(
        page=_graph_page,
        detector=_graph_detector,
        extraction=_graph_unaccounted_extraction,
        verifier=_graph_unaccounted_verifier,
        resolution=_graph_resolution,
        glyphs=_graph_glyphs,
        words=_graph_words,
        drawings=_graph_drawings,
        links=[],
    )
)
assert _graph_unaccounted_passed is False, _graph_unaccounted_issues
assert any(
    issue.get('issue_type') == 'graph-unresolved-visual-evidence'
    for issue in _graph_unaccounted_issues
), _graph_unaccounted_issues

# Phase 2G V1.2: a verifier can recover a genuinely omitted graphic entity
# with exact PDF-point drawing evidence. Detector region bboxes reported in a
# rendered-image coordinate frame are reconciled from final entity geometry.
_graph_recovery_extraction = copy.deepcopy(_graph_extraction)
_graph_recovery_note = 'Visible protective-earth symbol and one-ended stub omitted.'
_graph_recovery_extraction['unresolved_visual_evidence'] = [
    _graph_recovery_note
]
_graph_recovery_detector = copy.deepcopy(_graph_detector)
_graph_recovery_detector['regions'][0]['bbox_pt'] = [20, 200, 80, 300]
_graph_recovery_verifier = copy.deepcopy(_graph_verifier)
_graph_recovery_verifier.update({
    'verdict': 'review_required',
    'all_visible_entities_accounted_for': False,
    'all_visible_connections_accounted_for': False,
    'all_entity_text_visually_supported': False,
    'all_connection_geometry_supported': False,
    'recovery_entities': [{
        'occurrence_id': 'VR-GROUND-1',
        'region_id': 'R1',
        'entity_type': 'potential',
        'subtype': 'protective_earth_symbol',
        'tag_original': '',
        'label_original': '',
        'description_original': '',
        'function_text_original': '',
        'symbol_code': 'protective_earth',
        'location_code': '',
        'reference_value_original': '',
        'reference_context_original': '',
        'bbox_pt': [10, 60, 20, 72],
        'source_glyph_ids': [],
        'source_word_ids': [],
        'source_drawing_ids': [2],
        'confidence': 0.98,
        'evidence_notes': (
            'Visible ground symbol; the short stub has no second endpoint.'
        ),
    }],
    'recovery_edges': [],
    'visual_evidence_adjudications': [{
        'evidence_index': 0,
        'evidence_text_original': _graph_recovery_note,
        'status': 'recovered_entity',
        'related_entity_ids': ['VR-GROUND-1'],
        'related_edge_ids': [],
        'confidence': 0.98,
        'reason': (
            'The graphic entity is recovered from exact drawing evidence; the '
            'one-ended stub is not promoted to a two-endpoint connection.'
        ),
    }],
    'issues': [
        {
            'issue_type': 'missing_visible_entity',
            'severity': 'warning',
            'message': 'Visible ground symbol was absent from raw candidates.',
            'entity_ids': [],
            'edge_ids': [],
            'confidence': 0.98,
        },
        {
            'issue_type': 'missing_visible_connections',
            'severity': 'warning',
            'message': 'One-ended ground stub was absent from raw candidates.',
            'entity_ids': [],
            'edge_ids': [],
            'confidence': 0.96,
        },
    ],
})
_graph_recovery_drawings = _graph_drawings + [
    {'drawing_id': 2, 'bbox_pt': [10, 60, 20, 72]}
]
_graph_recovery_passed, _graph_recovery_entities, _graph_recovery_edges, (
    _graph_recovery_issues
) = _graph_validate_candidate(
    page=_graph_page,
    detector=_graph_recovery_detector,
    extraction=_graph_recovery_extraction,
    verifier=_graph_recovery_verifier,
    resolution=copy.deepcopy(_graph_resolution),
    glyphs=_graph_glyphs,
    words=_graph_words,
    drawings=_graph_recovery_drawings,
    links=[],
)
assert _graph_recovery_passed is True, _graph_recovery_issues
assert len(_graph_recovery_entities) == len(_graph_entities) + 1
assert len(_graph_recovery_edges) == len(_graph_edges)
assert _graph_recovery_extraction['verifier_evidence_recovery'][
    'recovered_entity_ids'
] == ['VR-GROUND-1']
assert _graph_recovery_extraction['region_bbox_adjudication'][
    'adjudicated_region_count'
] == 1
assert not [
    issue for issue in _graph_recovery_issues
    if issue.get('severity') in {'high', 'critical'}
], _graph_recovery_issues

# Unknown drawing evidence must fail closed.
_graph_bad_recovery_verifier = copy.deepcopy(_graph_recovery_verifier)
_graph_bad_recovery_verifier['recovery_entities'][0][
    'source_drawing_ids'
] = [999]
_graph_bad_recovery_passed, _, _, _graph_bad_recovery_issues = (
    _graph_validate_candidate(
        page=_graph_page,
        detector=copy.deepcopy(_graph_recovery_detector),
        extraction=copy.deepcopy(_graph_recovery_extraction),
        verifier=_graph_bad_recovery_verifier,
        resolution=copy.deepcopy(_graph_resolution),
        glyphs=_graph_glyphs,
        words=_graph_words,
        drawings=_graph_recovery_drawings,
        links=[],
    )
)
assert _graph_bad_recovery_passed is False, _graph_bad_recovery_issues
assert any(
    issue.get('issue_type') == 'graph-recovery-entity-evidence-id-invalid'
    for issue in _graph_bad_recovery_issues
), _graph_bad_recovery_issues

# A recovery ledger that still declares visible evidence unresolved must block.
_graph_still_unresolved_verifier = copy.deepcopy(_graph_recovery_verifier)
_graph_still_unresolved_verifier['recovery_entities'] = []
_graph_still_unresolved_verifier['visual_evidence_adjudications'][0].update({
    'status': 'still_unresolved',
    'related_entity_ids': [],
})
_graph_still_unresolved_passed, _, _, _graph_still_unresolved_issues = (
    _graph_validate_candidate(
        page=_graph_page,
        detector=copy.deepcopy(_graph_recovery_detector),
        extraction=copy.deepcopy(_graph_recovery_extraction),
        verifier=_graph_still_unresolved_verifier,
        resolution=copy.deepcopy(_graph_resolution),
        glyphs=_graph_glyphs,
        words=_graph_words,
        drawings=_graph_recovery_drawings,
        links=[],
    )
)
assert _graph_still_unresolved_passed is False, (
    _graph_still_unresolved_issues
)
assert any(
    issue.get('issue_type') == 'graph-visual-evidence-still-unresolved'
    for issue in _graph_still_unresolved_issues
), _graph_still_unresolved_issues


# Phase 2G V2: semantic annotations on graphic-only recovery entities do not
# pretend to be printed source text. Exact drawing evidence remains mandatory.
_graph_v2_note = 'Standalone graphic symbol omitted from the raw candidate.'
_graph_v2_detector = {
    'regions': [{
        'region_id': 'R1',
        'region_kind': 'other',
        'bbox_pt': [0, 0, 100, 80],
        'visible_connection_count': 0,
        'confidence': 0.99,
    }],
}
_graph_v2_base_entity = {
    'occurrence_id': 'E1',
    'region_id': 'R1',
    'entity_type': 'potential',
    'subtype': 'supply',
    'tag_original': '24V',
    'label_original': '24V',
    'description_original': '',
    'function_text_original': '',
    'symbol_code': '',
    'location_code': '',
    'reference_value_original': '',
    'reference_context_original': '',
    'bbox_pt': [10, 10, 20, 20],
    'source_glyph_ids': [1],
    'source_word_ids': [1],
    'source_drawing_ids': [1],
    'confidence': 0.99,
    'evidence_notes': '',
}
_graph_v2_base_edge = {
    'edge_id': 'ED1',
    'source_occurrence_id': 'E1',
    'target_occurrence_id': 'E1-2',
    'relation_type': 'electrically_connected_to',
    'is_directed': False,
    'bbox_pt': [20, 10, 40, 10],
    'potential_original': '24V',
    'wire_reference_original': '',
    'source_glyph_ids': [],
    'source_drawing_ids': [1],
    'source_link_ids': [],
    'confidence': 0.99,
    'evidence_notes': '',
}
_graph_v2_companion = {
    **_graph_v2_base_entity,
    'occurrence_id': 'E1-2',
    'tag_original': '',
    'label_original': '',
    'entity_type': 'junction',
    'bbox_pt': [38, 8, 42, 12],
}
_graph_v2_recovery_verifier = {
    'confidence': 0.99,
    'recovery_entities': [{
        'occurrence_id': 'RE-GROUND',
        'region_id': 'R1',
        'entity_type': 'potential',
        'subtype': 'protective_earth_symbol',
        'tag_original': '',
        'label_original': '',
        # These are semantic annotations, not claimed printed words.
        'description_original': 'Standalone protective-earth graphic symbol',
        'function_text_original': 'Protective earth',
        'symbol_code': 'protective_earth',
        'location_code': '',
        'reference_value_original': '',
        'reference_context_original': '',
        'bbox_pt': [50, 50, 60, 65],
        'source_glyph_ids': [],
        'source_word_ids': [],
        'source_drawing_ids': [2],
        'confidence': 0.99,
        'evidence_notes': 'Graphic-only source occurrence.',
    }],
    'recovery_edges': [],
    'visual_evidence_adjudications': [{
        'evidence_index': 0,
        'evidence_text_original': _graph_v2_note,
        'status': 'recovered_entity',
        'related_entity_ids': ['RE-GROUND'],
        'related_edge_ids': [],
        'confidence': 0.99,
        'reason': 'Recovered from exact drawing evidence.',
    }],
}
_graph_v2_recovery_entities, _, _graph_v2_recovery_audit, (
    _graph_v2_recovery_issues
) = _graph_apply_recovery(
    page={'page_width_pt': 100, 'page_height_pt': 80},
    detector=_graph_v2_detector,
    extraction={'unresolved_visual_evidence': [_graph_v2_note]},
    verifier=_graph_v2_recovery_verifier,
    resolution={'resolution_status_counts': {}},
    base_entities=[_graph_v2_base_entity, _graph_v2_companion],
    base_edges=[_graph_v2_base_edge],
    glyphs=[{'glyph_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
    words=[{'word_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
    drawings=[
        {'drawing_id': 1, 'bbox_pt': [10, 10, 40, 20]},
        {'drawing_id': 2, 'bbox_pt': [50, 50, 60, 65]},
    ],
    links=[],
)
assert _graph_v2_recovery_audit['validated'] is True, (
    _graph_v2_recovery_issues
)
assert any(
    row['occurrence_id'] == 'RE-GROUND'
    and row['recovery_evidence']['text_evidence_mode']
        == 'graphic_with_semantic_annotation'
    for row in _graph_v2_recovery_entities
), _graph_v2_recovery_entities

# Literal printed text without glyph/word evidence still blocks.
_graph_v2_bad_text = copy.deepcopy(_graph_v2_recovery_verifier)
_graph_v2_bad_text['recovery_entities'][0]['tag_original'] = 'PE'
_, _, _graph_v2_bad_text_audit, _graph_v2_bad_text_issues = (
    _graph_apply_recovery(
        page={'page_width_pt': 100, 'page_height_pt': 80},
        detector=_graph_v2_detector,
        extraction={'unresolved_visual_evidence': [_graph_v2_note]},
        verifier=_graph_v2_bad_text,
        resolution={'resolution_status_counts': {}},
        base_entities=[_graph_v2_base_entity, _graph_v2_companion],
        base_edges=[_graph_v2_base_edge],
        glyphs=[{'glyph_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        words=[{'word_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        drawings=[
            {'drawing_id': 1, 'bbox_pt': [10, 10, 40, 20]},
            {'drawing_id': 2, 'bbox_pt': [50, 50, 60, 65]},
        ],
        links=[],
    )
)
assert _graph_v2_bad_text_audit['validated'] is False
assert any(
    issue.get('issue_type') == 'graph-recovery-entity-text-evidence-missing'
    for issue in _graph_v2_bad_text_issues
), _graph_v2_bad_text_issues

# Non-materializable evidence may cite existing graph items as context without
# materializing anything new. Unknown context IDs still block.
_graph_v2_context_note = 'Conductor annotation belongs to existing branch.'
_graph_v2_context_verifier = {
    'confidence': 0.99,
    'recovery_entities': [],
    'recovery_edges': [],
    'visual_evidence_adjudications': [{
        'evidence_index': 0,
        'evidence_text_original': _graph_v2_context_note,
        'status': 'accounted_non_materializable',
        'related_entity_ids': ['E1'],
        'related_edge_ids': ['ED1'],
        'confidence': 0.99,
        'reason': 'Existing items are context only.',
    }],
}
_, _, _graph_v2_context_audit, _graph_v2_context_issues = (
    _graph_apply_recovery(
        page={'page_width_pt': 100, 'page_height_pt': 80},
        detector=_graph_v2_detector,
        extraction={'unresolved_visual_evidence': [_graph_v2_context_note]},
        verifier=_graph_v2_context_verifier,
        resolution={'resolution_status_counts': {}},
        base_entities=[_graph_v2_base_entity, _graph_v2_companion],
        base_edges=[_graph_v2_base_edge],
        glyphs=[{'glyph_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        words=[{'word_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        drawings=[{'drawing_id': 1, 'bbox_pt': [10, 10, 40, 20]}],
        links=[],
    )
)
assert _graph_v2_context_audit['validated'] is True, _graph_v2_context_issues
assert _graph_v2_context_audit['visual_evidence_adjudications'][0][
    'context_link_policy'
]['materializes_new_graph_items'] is False
_graph_v2_invalid_context = copy.deepcopy(_graph_v2_context_verifier)
_graph_v2_invalid_context['visual_evidence_adjudications'][0][
    'related_entity_ids'
] = ['MISSING']
_, _, _graph_v2_invalid_context_audit, _graph_v2_invalid_context_issues = (
    _graph_apply_recovery(
        page={'page_width_pt': 100, 'page_height_pt': 80},
        detector=_graph_v2_detector,
        extraction={'unresolved_visual_evidence': [_graph_v2_context_note]},
        verifier=_graph_v2_invalid_context,
        resolution={'resolution_status_counts': {}},
        base_entities=[_graph_v2_base_entity, _graph_v2_companion],
        base_edges=[_graph_v2_base_edge],
        glyphs=[{'glyph_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        words=[{'word_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        drawings=[{'drawing_id': 1, 'bbox_pt': [10, 10, 40, 20]}],
        links=[],
    )
)
assert _graph_v2_invalid_context_audit['validated'] is False
assert any(
    issue.get('issue_type')
        == 'graph-visual-evidence-nonmaterializable-context-invalid'
    for issue in _graph_v2_invalid_context_issues
), _graph_v2_invalid_context_issues

# Inverted or otherwise invalid AI geometry is repaired only from the exact
# cited source registries. This reproduces the page-43 branch-potential failure.
_graph_v2_geometry_entities = [{
    **_graph_v2_base_entity,
    'occurrence_id': 'E-INVERTED',
    'bbox_pt': [60, 50, 20, 10],
    'source_glyph_ids': [1],
    'source_word_ids': [1],
    'source_drawing_ids': [1],
}]
_graph_v2_geometry_edges = [{
    **_graph_v2_base_edge,
    'edge_id': 'ED-INVERTED',
    'source_occurrence_id': 'E-INVERTED',
    'target_occurrence_id': 'E1-2',
    'bbox_pt': [50, 40, 20, 10],
}]
_graph_v2_geometry_audit, _graph_v2_geometry_issues = (
    _graph_reconcile_geometry(
        page={'page_width_pt': 100, 'page_height_pt': 80},
        entities=_graph_v2_geometry_entities,
        edges=_graph_v2_geometry_edges,
        glyphs=[{'glyph_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        words=[{'word_id': 1, 'bbox_pt': [10, 10, 20, 20]}],
        drawings=[{'drawing_id': 1, 'bbox_pt': [10, 10, 40, 30]}],
    )
)
assert _graph_v2_geometry_audit['validated'] is True, _graph_v2_geometry_issues
assert _graph_v2_geometry_audit['reconciled_entity_count'] == 1
assert _graph_v2_geometry_audit['reconciled_edge_count'] == 1


# Phase 2G V3: one canonical atomic patch plan covers every raw entity and
# edge, supports replacement/split/rewire, preserves explicit unresolved
# references, and validates only the final projection.
def _graph_v3_entity(
    occurrence_id,
    entity_type,
    tag,
    bbox,
    glyph_id,
    *,
    drawing_ids=None,
):
    return {
        'occurrence_id': occurrence_id,
        'region_id': 'R1',
        'entity_type': entity_type,
        'subtype': '',
        'tag_original': tag,
        'label_original': '',
        'description_original': '',
        'function_text_original': '',
        'symbol_code': '',
        'location_code': '',
        'reference_value_original': '',
        'reference_context_original': '',
        'bbox_pt': bbox,
        'source_glyph_ids': [glyph_id] if glyph_id else [],
        'source_word_ids': [glyph_id] if glyph_id else [],
        'source_drawing_ids': list(drawing_ids or []),
        'confidence': 0.98,
        'evidence_notes': 'V3 fixture',
    }


def _graph_v3_raw_entity(*args, **kwargs):
    row = _graph_v3_entity(*args, **kwargs)
    row.pop('source_drawing_ids', None)
    return row


def _graph_v3_edge(
    edge_id,
    source_id,
    target_id,
    *,
    relation_type='electrically_connected_to',
    drawing_ids=None,
    link_ids=None,
    bbox=None,
):
    return {
        'edge_id': edge_id,
        'source_occurrence_id': source_id,
        'target_occurrence_id': target_id,
        'relation_type': relation_type,
        'is_directed': False,
        'potential_original': '',
        'wire_reference_original': '',
        'bbox_pt': bbox or [20, 15, 40, 15],
        'source_glyph_ids': [],
        'source_drawing_ids': list(drawing_ids or []),
        'source_link_ids': list(link_ids or []),
        'confidence': 0.98,
        'evidence_notes': 'V3 fixture',
    }


def _graph_v3_entity_op(operation_id, action, source_id='', results=None):
    return {
        'operation_id': operation_id,
        'action': action,
        'source_entity_id': source_id,
        'result_entities': results or [],
        'evidence_indexes': [],
        'confidence': 0.98,
        'reason': 'V3 fixture',
    }


def _graph_v3_edge_op(operation_id, action, source_id='', results=None):
    return {
        'operation_id': operation_id,
        'action': action,
        'source_edge_id': source_id,
        'result_edges': results or [],
        'evidence_indexes': [],
        'confidence': 0.98,
        'reason': 'V3 fixture',
    }


def _graph_v3_make_verifier(entity_operations, edge_operations, *, issues=None):
    return {
        'page_id': 1,
        'verdict': 'apply_patch',
        'patch_plan_version': _graph_patch_plan_version,
        'entity_operations': entity_operations,
        'edge_operations': edge_operations,
        'evidence_adjudications': [],
        'final_assertions': {
            'all_raw_entities_decided': True,
            'all_raw_edges_decided': True,
            'all_visible_entities_accounted_for': True,
            'all_visible_connections_accounted_for': True,
            'all_entity_text_visually_supported': True,
            'all_connection_geometry_supported': True,
            'all_references_resolved_or_explicitly_unresolved': True,
            'duplicates_preserved': True,
            'patch_plan_safe_to_apply': True,
        },
        'confidence': 0.98,
        'issues': issues or [],
    }


_graph_v3_schema = _graph_v3_verifier_schema()['schema']
assert _graph_v3_schema['additionalProperties'] is False
assert _graph_v3_schema['properties']['entity_operations'][
    'items'
]['properties']['action']['enum'] == [
    'ADD_ENTITY', 'KEEP_ENTITY', 'REMOVE_ENTITY',
    'REPLACE_ENTITY', 'SPLIT_ENTITY'
]
assert _graph_v3_schema['properties']['edge_operations'][
    'items'
]['properties']['action']['enum'] == [
    'ADD_EDGE', 'KEEP_EDGE', 'REMOVE_EDGE', 'REWIRE_EDGE'
]
assert set(_graph_v3_schema['properties']['entity_operations'][
    'items'
]['properties']['result_entities']['items']['properties'][
    'entity_type'
]['enum']) >= {'io_reference', 'terminal_reference', 'page_reference'}

_graph_v3_page = {
    'id': 1,
    'pdf_page_number': 1,
    'sheet_code': 'S1',
    'page_width_pt': 100,
    'page_height_pt': 100,
}
_graph_v3_detector = {
    'page_id': 1,
    'all_visible_circuit_regions_accounted_for': True,
    'regions': [{
        'region_id': 'R1',
        'region_kind': 'mixed_circuit',
        'bbox_pt': [0, 0, 100, 100],
        'visible_component_count': 2,
        'visible_connection_count': 1,
        'confidence': 0.98,
        'notes': 'V3 fixture',
    }],
    'uncovered_visual_regions': [],
    'confidence': 0.98,
    'issues': [],
}
_graph_v3_words = [
    {'word_id': 1, 'text_original': 'K1', 'bbox_pt': [10, 10, 20, 20]},
    {'word_id': 2, 'text_original': '24V', 'bbox_pt': [40, 10, 50, 20]},
    {'word_id': 3, 'text_original': 'X', 'bbox_pt': [60, 10, 65, 20]},
]
_graph_v3_glyphs = [
    {'glyph_id': 1, 'text_original': 'K', 'bbox_pt': [10, 10, 15, 20]},
    {'glyph_id': 2, 'text_original': '2', 'bbox_pt': [40, 10, 45, 20]},
    {'glyph_id': 3, 'text_original': 'X', 'bbox_pt': [60, 10, 65, 20]},
]
_graph_v3_drawings = [{'drawing_id': 1, 'bbox_pt': [20, 14, 40, 16]}]
_graph_v3_registry = {
    'bom': [], 'io': [], 'terminals': [], 'pages': [],
    'cross_references': [],
}
_graph_v3_e1 = _graph_v3_raw_entity(
    'e1', 'component_occurrence', 'K1', [10, 10, 20, 20], 1
)
_graph_v3_e2 = _graph_v3_raw_entity(
    'e2', 'potential', '24V', [40, 10, 50, 20], 2
)
_graph_v3_false = _graph_v3_raw_entity(
    'e3', 'component_occurrence', 'X', [60, 10, 65, 20], 3
)
_graph_v3_raw_edge = _graph_v3_edge(
    'ed1', 'e1', 'e2', drawing_ids=[1]
)
_graph_v3_extraction = {
    'page_id': 1,
    'entities': [_graph_v3_e1, _graph_v3_e2, _graph_v3_false],
    'edges': [_graph_v3_raw_edge],
    'unresolved_visual_evidence': [],
    'confidence': 0.97,
    'issues': [],
}
_graph_v3_replacement = _graph_v3_entity(
    'f1', 'contact', 'K1', [10, 10, 20, 20], 1,
    drawing_ids=[1],
)
_graph_v3_verifier = _graph_v3_make_verifier(
    [
        _graph_v3_entity_op(
            'oe1', 'REPLACE_ENTITY', 'e1', [_graph_v3_replacement]
        ),
        _graph_v3_entity_op('oe2', 'KEEP_ENTITY', 'e2'),
        _graph_v3_entity_op('oe3', 'REMOVE_ENTITY', 'e3'),
    ],
    [
        _graph_v3_edge_op(
            'od1', 'REWIRE_EDGE', 'ed1', [
                _graph_v3_edge(
                    'fed1', 'f1', 'e2', drawing_ids=[1]
                )
            ]
        )
    ],
    issues=[{
        'issue_type': 'under_materialized_symbol',
        'severity': 'high',
        'message': 'Raw entity must be replaced and its edge rewired.',
        'entity_ids': ['e1'],
        'edge_ids': ['ed1'],
        'confidence': 0.98,
        'resolution_status': 'resolved_by_patch_plan',
        'related_operation_ids': ['oe1', 'od1'],
    }],
)
(
    _graph_v3_entities,
    _graph_v3_edges,
    _graph_v3_patch_audit,
    _graph_v3_patch_issues,
) = _graph_apply_patch_plan(
    page=_graph_v3_page,
    extraction=_graph_v3_extraction,
    verifier=_graph_v3_verifier,
)
assert _graph_v3_patch_audit['validated'] is True, _graph_v3_patch_issues
assert [row['occurrence_id'] for row in _graph_v3_entities] == ['f1', 'e2']
assert [row['edge_id'] for row in _graph_v3_edges] == ['fed1']
_graph_v3_resolution = _graph_resolve_references(
    {'entities': _graph_v3_entities}, _graph_v3_registry
)
(
    _graph_v3_passed,
    _,
    _,
    _graph_v3_final_issues,
) = _graph_validate_patched(
    page=_graph_v3_page,
    detector=copy.deepcopy(_graph_v3_detector),
    extraction=_graph_v3_extraction,
    verifier=_graph_v3_verifier,
    resolution=_graph_v3_resolution,
    entities=_graph_v3_entities,
    edges=_graph_v3_edges,
    patch_audit=_graph_v3_patch_audit,
    patch_issues=_graph_v3_patch_issues,
    glyphs=_graph_v3_glyphs,
    words=_graph_v3_words,
    drawings=_graph_v3_drawings,
    links=[],
)
assert _graph_v3_passed is True, _graph_v3_final_issues
assert any(
    issue.get('source_stage') == 'verifier_patch_plan_resolved'
    for issue in _graph_v3_final_issues
), _graph_v3_final_issues

# Merged repeated occurrences can be split and every incident edge rewired.
_graph_v3_split_extraction = {
    'page_id': 1,
    'entities': [
        _graph_v3_raw_entity(
            'merged', 'component_occurrence', 'K1',
            [10, 10, 30, 20], 1
        ),
        _graph_v3_e2,
    ],
    'edges': [
        _graph_v3_edge(
            'a', 'merged', 'e2', drawing_ids=[1],
            bbox=[30, 12, 40, 12]
        ),
        _graph_v3_edge(
            'b', 'merged', 'e2', drawing_ids=[1],
            bbox=[30, 18, 40, 18]
        ),
    ],
    'unresolved_visual_evidence': [],
    'confidence': 0.97,
    'issues': [],
}
_graph_v3_split_verifier = _graph_v3_make_verifier(
    [
        _graph_v3_entity_op('split', 'SPLIT_ENTITY', 'merged', [
            _graph_v3_entity(
                's1', 'contact', 'K1', [10, 10, 18, 20], 1,
                drawing_ids=[1],
            ),
            _graph_v3_entity(
                's2', 'contact', 'K1', [20, 10, 28, 20], 1,
                drawing_ids=[1],
            ),
        ]),
        _graph_v3_entity_op('keep-e2', 'KEEP_ENTITY', 'e2'),
    ],
    [
        _graph_v3_edge_op('rewire-a', 'REWIRE_EDGE', 'a', [
            _graph_v3_edge(
                'fa', 's1', 'e2', drawing_ids=[1],
                bbox=[18, 12, 40, 12]
            )
        ]),
        _graph_v3_edge_op('rewire-b', 'REWIRE_EDGE', 'b', [
            _graph_v3_edge(
                'fb', 's2', 'e2', drawing_ids=[1],
                bbox=[28, 18, 40, 18]
            )
        ]),
    ],
)
(
    _graph_v3_split_entities,
    _graph_v3_split_edges,
    _graph_v3_split_audit,
    _graph_v3_split_issues,
) = _graph_apply_patch_plan(
    page=_graph_v3_page,
    extraction=_graph_v3_split_extraction,
    verifier=_graph_v3_split_verifier,
)
assert _graph_v3_split_audit['entity_lineage']['merged'] == ['s1', 's2']
assert len(_graph_v3_split_edges) == 2
assert _graph_v3_split_audit['validated'] is True, _graph_v3_split_issues

# Explicit unresolved references are accounted for without a fabricated match.
_graph_v3_reference = _graph_v3_raw_entity(
    'ref', 'terminal_reference', 'XP1', [10, 10, 20, 20], 1
)
_graph_v3_reference['reference_value_original'] = '99'
_graph_v3_ref_extraction = {
    'page_id': 1,
    'entities': [_graph_v3_reference, _graph_v3_e2],
    'edges': [_graph_v3_edge(
        'ref-edge', 'ref', 'e2', relation_type='linked_to_component',
        drawing_ids=[], bbox=[20, 15, 40, 15]
    )],
    'unresolved_visual_evidence': [],
    'confidence': 0.97,
    'issues': [],
}
_graph_v3_ref_verifier = _graph_v3_make_verifier(
    [
        _graph_v3_entity_op('keep-ref', 'KEEP_ENTITY', 'ref'),
        _graph_v3_entity_op('keep-e2', 'KEEP_ENTITY', 'e2'),
    ],
    [_graph_v3_edge_op('keep-ref-edge', 'KEEP_EDGE', 'ref-edge')],
)
(
    _graph_v3_ref_entities,
    _graph_v3_ref_edges,
    _graph_v3_ref_audit,
    _graph_v3_ref_patch_issues,
) = _graph_apply_patch_plan(
    page=_graph_v3_page,
    extraction=_graph_v3_ref_extraction,
    verifier=_graph_v3_ref_verifier,
)
_graph_v3_ref_resolution = _graph_resolve_references(
    {'entities': _graph_v3_ref_entities}, _graph_v3_registry
)
assert _graph_v3_ref_resolution['unresolved_reference_entity_ids'] == ['ref']
assert _graph_v3_ref_resolution['all_reference_entities_accounted_for'] is True
_graph_v3_ref_passed, _, _, _graph_v3_ref_final_issues = (
    _graph_validate_patched(
        page=_graph_v3_page,
        detector=copy.deepcopy(_graph_v3_detector),
        extraction=_graph_v3_ref_extraction,
        verifier=_graph_v3_ref_verifier,
        resolution=_graph_v3_ref_resolution,
        entities=_graph_v3_ref_entities,
        edges=_graph_v3_ref_edges,
        patch_audit=_graph_v3_ref_audit,
        patch_issues=_graph_v3_ref_patch_issues,
        glyphs=_graph_v3_glyphs,
        words=_graph_v3_words,
        drawings=_graph_v3_drawings,
        links=[],
    )
)
assert _graph_v3_ref_passed is True, _graph_v3_ref_final_issues

# Missing coverage and a PDF-link-only conductor both remain fail-closed.
_graph_v3_missing_verifier = _graph_v3_make_verifier(
    [_graph_v3_entity_op('keep-e1', 'KEEP_ENTITY', 'e1')],
    [_graph_v3_edge_op('keep-ed1', 'KEEP_EDGE', 'ed1')],
)
_, _, _, _graph_v3_missing_issues = _graph_apply_patch_plan(
    page=_graph_v3_page,
    extraction=_graph_v3_extraction,
    verifier=_graph_v3_missing_verifier,
)
assert any(
    issue.get('issue_type') == 'graph-entity-patch-coverage-failed'
    for issue in _graph_v3_missing_issues
), _graph_v3_missing_issues

_graph_v3_link_only_extraction = {
    'page_id': 1,
    'entities': [_graph_v3_e1, _graph_v3_e2],
    'edges': [_graph_v3_edge(
        'link-edge', 'e1', 'e2', drawing_ids=[], link_ids=[9]
    )],
    'unresolved_visual_evidence': [],
    'confidence': 0.97,
    'issues': [],
}
_graph_v3_link_only_verifier = _graph_v3_make_verifier(
    [
        _graph_v3_entity_op('keep-e1', 'KEEP_ENTITY', 'e1'),
        _graph_v3_entity_op('keep-e2', 'KEEP_ENTITY', 'e2'),
    ],
    [_graph_v3_edge_op('keep-link', 'KEEP_EDGE', 'link-edge')],
)
(
    _graph_v3_link_entities,
    _graph_v3_link_edges,
    _graph_v3_link_audit,
    _graph_v3_link_patch_issues,
) = _graph_apply_patch_plan(
    page=_graph_v3_page,
    extraction=_graph_v3_link_only_extraction,
    verifier=_graph_v3_link_only_verifier,
)
_graph_v3_link_resolution = _graph_resolve_references(
    {'entities': _graph_v3_link_entities}, _graph_v3_registry
)
_graph_v3_link_passed, _, _, _graph_v3_link_issues = (
    _graph_validate_patched(
        page=_graph_v3_page,
        detector=copy.deepcopy(_graph_v3_detector),
        extraction=_graph_v3_link_only_extraction,
        verifier=_graph_v3_link_only_verifier,
        resolution=_graph_v3_link_resolution,
        entities=_graph_v3_link_entities,
        edges=_graph_v3_link_edges,
        patch_audit=_graph_v3_link_audit,
        patch_issues=_graph_v3_link_patch_issues,
        glyphs=_graph_v3_glyphs,
        words=_graph_v3_words,
        drawings=_graph_v3_drawings,
        links=[{'id': 9}],
    )
)
assert _graph_v3_link_passed is False
assert any(
    issue.get('issue_type') == 'graph-edge-geometry-evidence-missing'
    for issue in _graph_v3_link_issues
), _graph_v3_link_issues

_graph_v3_cause_counts = _graph_review_cause_family_counts([
    {
        'issue_type': 'graph-edge-geometry-evidence-missing',
        'severity': 'high',
        'source_stage': 'deterministic_validator',
    },
    {
        'issue_type': 'graph-verifier-blocked-page',
        'severity': 'high',
        'source_stage': 'deterministic_validator',
    },
])
assert _graph_v3_cause_counts['edge_rewire_or_geometry'] == 1
assert _graph_v3_cause_counts['publish_gate_cascade'] == 1

required = {
    '/v1/ai/electrical/normalize',
    '/v1/ai/electrical/extract-structured',
    '/v1/ai/electrical/extract-terminals',
    '/v1/ai/electrical/extract-bom',
    '/v1/ai/electrical/extract-graph',
    '/v1/ai/electrical/graph-plan',
    '/v1/ai/electrical/source/snapshot',
}
missing = required - set(openapi.get('paths', {}))
assert not missing, f'ROUTE MANCANTI: {sorted(missing)}'

print('IMAGE RUNTIME PREFLIGHT: OK')
print(json.dumps({
    'marker': version.get('electrical_code_marker'),
    'routes': sorted(required),
}, ensure_ascii=False))
