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

required = {
    '/v1/ai/electrical/normalize',
    '/v1/ai/electrical/extract-structured',
    '/v1/ai/electrical/extract-terminals',
    '/v1/ai/electrical/source/snapshot',
}
missing = required - set(openapi.get('paths', {}))
assert not missing, f'ROUTE MANCANTI: {sorted(missing)}'

print('IMAGE RUNTIME PREFLIGHT: OK')
print(json.dumps({
    'marker': version.get('electrical_code_marker'),
    'routes': sorted(required),
}, ensure_ascii=False))
