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
    _build_table_proposal as _bom_build_table_proposal,
    _candidate_tables as _bom_candidate_tables,
    _canonical_row_candidate_accounting as _bom_row_candidate_accounting,
    _component_tag_sequence_source_exact_adjudication as _bom_tag_sequence_adjudication,
    _detect_sidecar_column_specs as _bom_detect_sidecar_columns,
    _field_evidence_audit as _bom_field_evidence_audit,
    _fallback_page_proposal as _bom_fallback_page_proposal,
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
    extractions=_bom_v15_ambiguous_extractions,
    word_map=_bom_v13_word_map,
)
assert _bom_v15_ambiguous_reconciliation[
    'cross_field_transfer_row_ids'
] == [], _bom_v15_ambiguous_reconciliation

required = {
    '/v1/ai/electrical/normalize',
    '/v1/ai/electrical/extract-structured',
    '/v1/ai/electrical/extract-terminals',
    '/v1/ai/electrical/extract-bom',
    '/v1/ai/electrical/source/snapshot',
}
missing = required - set(openapi.get('paths', {}))
assert not missing, f'ROUTE MANCANTI: {sorted(missing)}'

print('IMAGE RUNTIME PREFLIGHT: OK')
print(json.dumps({
    'marker': version.get('electrical_code_marker'),
    'routes': sorted(required),
}, ensure_ascii=False))
