"""Claimless, source-bound observation collection after a rejected Smart Start.

No model verdict is changed. A literal alert-label witness authorizes only an
application question about information already held, never a causal conclusion
or a machine operation. Endpoint HMAC/scope validation remains mandatory.
"""
from copy import deepcopy
import re
import unicodedata

from . import review_references as refs


POLICY_VERSION = 'smart-rejected-start-observation-v1'
MARKER = 'start_observation_protocol'


def require(value, code):
    if not value:
        raise ValueError(code)


def _scope(state):
    scope = {key: state.get(key) for key in ('company_id', 'machine_id', 'session_id')}
    require(all(isinstance(value, str) and value.strip() for value in scope.values()),
            'start_observation_scope_required')
    return scope


def _normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def _anchor(state, packet):
    """Recognize a field value, not a substring, inferred cause, or model claim."""
    symptom = state.get('symptom_text')
    require(isinstance(symptom, str), 'start_observation_symptom_required')
    quoted = re.findall(r'"([^"\r\n]+)"|“([^”\r\n]+)”|«([^»\r\n]+)»', symptom)
    labels = {next(part for part in match if part).strip() for match in quoted}
    require(len(labels) == 1, 'start_observation_unambiguous_quoted_label_required')
    label = labels.pop()
    require(3 <= len(label) <= 100 and re.fullmatch(r'[\w ()/.,:+\-]+', label, re.UNICODE)
            and any(char.isalpha() for char in label), 'start_observation_label_invalid')
    scope = _scope(state)
    model = packet.get('model_packet') or {}
    context = model.get('request_context') or {}
    require(context.get('company_id') == scope['company_id']
            and context.get('selected_machine_id') == scope['machine_id']
            and context.get('selected_machine_present') is True,
            'start_observation_packet_scope_mismatch')
    # Deliberately narrow: a full admitted Message/Type (Messaggio/Tipo) record.
    # Prose mentions, prefixes, neighboring fields and general company manuals
    # do not establish this witness. The record does not establish component
    # applicability, event classification, measured values, or a causal fact.
    field = re.compile(r'(?m)^[ \t]*(?:Message|Messaggio)[ \t]*\r?\n'
                       r'[ \t]*(?P<label>[^\r\n]+?)[ \t]*\r?\n'
                       r'(?:[ \t]*\r?\n)*[ \t]*(?:Type|Tipo)[ \t]*(?:\r?\n|$)')
    for source in model.get('sources') or []:
        if source.get('machine_relation') != 'selected_machine' or source.get('text_complete') is not True:
            continue
        # `manual` is the persisted type in the actual admitted machine corpus;
        # `document` is the generic document type. Problem/solution records (`ps`)
        # cannot authorize this narrowly documentary label witness.
        if source.get('source_type') not in {'document', 'manual'} or not source.get('document_id'):
            continue
        text = source.get('text')
        if not isinstance(text, str):
            continue
        for match in field.finditer(text):
            value = match.group('label').strip()
            if _normalized(value) != _normalized(label):
                continue
            start = match.start('label') + len(match.group('label')) - len(match.group('label').lstrip())
            return {'citation_id': source['citation_id'], 'document_id': source['document_id'],
                    'label': value, 'start': start, 'end': start + len(value), 'quote': value}
    raise ValueError('start_observation_documented_label_required')


def _rejection(state):
    value = state.get('grounding_review_meta') or {}
    require(value.get('outcome') == 'rejected' and value.get('decision_validated') is True
            and value.get('failure_reason') == 'no_supported_hypotheses'
            and value.get('accepted_proposals') == 0 and value.get('rejected_proposals', 0) > 0
            and value.get('attempt_limit') == 1,
            'start_observation_validated_zero_acceptance_required')
    return value


def _question(language, label):
    english = language == 'en'
    why = (
        'This reported label occurs in documentation associated with the selected machine. '
        'That match establishes neither the applicable component or message type nor a cause. '
        'The proposed diagnostic turn did not pass independent review; no hypothesis is accepted. '
        'Report only information already available. If none is available, choose Unknown; '
        'the diagnosis will remain inconclusive.' if english else
        'Questa etichetta riportata compare nella documentazione associata alla macchina selezionata. '
        'La corrispondenza non stabilisce il componente o il tipo di messaggio applicabile, né una causa. '
        'Il turno diagnostico proposto non ha superato la revisione indipendente: nessuna ipotesi è accettata. '
        'Riporta soltanto informazioni già disponibili. Se non ne hai, scegli Non so: '
        'la diagnosi resterà inconclusiva.')
    safety = ('Do not obtain new readings, navigate machine screens, inspect, test, move or adjust the machine to answer.'
              if english else 'Per rispondere non acquisire nuove letture, navigare schermate macchina, '
              'eseguire ispezioni o prove, muovere o regolare la macchina.')
    return {'question_id': 'Q1', 'question_number': 1, 'question_type': 'single_choice',
            'application_start_observation_version': POLICY_VERSION,
            'question_text': (f'For the reported message “{label}”, do you have an observation already recorded to add in the notes? '
                              if english else f'Per il messaggio riportato “{label}”, hai un’osservazione già registrata da aggiungere nelle note? ') + safety,
            'why_asked': why, 'safety_level': 'normal', 'safety_note': safety,
            'options': [{'id': 'yes', 'label_en': 'Yes, I have an existing observation to add',
                         'label_it': 'Sì, ho un’osservazione già disponibile da aggiungere'},
                        {'id': 'unknown', 'label_en': "I don't know", 'label_it': 'Non so'}],
            'target_hypotheses': []}


def _step(marker):
    language, label = marker['language'], marker['anchor']['label']
    if marker['phase'] == 'active':
        question = _question(language, label)
        return {'status': 'in_progress', 'final_ready': False, 'hypotheses': [],
                'question': question, 'operator_summary': question['why_asked'], 'final_result': {}}
    text = ('The diagnosis is inconclusive: no cause has been identified, confirmed or excluded. '
            'The reported message label appears in the available documentation, but its applicable component, '
            'classification and causal explanation have not been established. '
            'The proposed diagnostic turn did not pass independent review. '
            'No additional observation was assessed. This conclusion requests no new inspection, test, movement or adjustment.'
            if language == 'en' else
            'La diagnosi è inconclusiva: nessuna causa è stata identificata, confermata o esclusa. '
            'L’etichetta del messaggio riportato compare nella documentazione disponibile, ma non sono stati stabiliti '
            'il componente applicabile, la classificazione o la spiegazione causale. '
            'Il turno diagnostico proposto non ha superato la revisione indipendente. '
            'Non è stata valutata alcuna osservazione aggiuntiva. Questa conclusione non richiede nuove ispezioni, prove, movimenti o regolazioni.')
    final = {'summary': text, 'most_likely_hypothesis_id': '', 'most_likely_label': '',
             'probability_pct': 0.0, 'probability_band': 'unknown', 'recommended_checks': []}
    return {'status': 'completed', 'final_ready': True, 'hypotheses': [], 'question': {},
            'operator_summary': text, 'final_result': final}


def _update(state, marker):
    updated = deepcopy(state)
    updated[MARKER] = marker
    updated.pop('grounding_review', None)
    updated.pop('grounding_review_debug', None)
    step = _step(marker)
    updated.update(status=step['status'], language=marker['language'], hypotheses=[],
                   current_question=step['question'], final_result=step['final_result'])
    return step, updated


def start(*, state, packet, language):
    require(language in ('en', 'it') and state.get('history') == []
            and not state.get('grounding_review') and not state.get('protocol_inconclusive')
            and not state.get(MARKER) and (state.get('evidence_gate') or {}).get('accepted') is True,
            'start_observation_initial_admitted_state_required')
    rejection = _rejection(state)
    marker = {'policy_version': POLICY_VERSION, 'scope': _scope(state), 'language': language,
              'symptom_digest': refs.digest(state.get('symptom_text')),
              'packet_digest': refs.digest(packet), 'anchor': _anchor(state, packet),
              'rejection': deepcopy(rejection), 'phase': 'active', 'history_digest': refs.digest([])}
    return _update(state, marker)


def replay(*, state, packet):
    marker = state.get(MARKER) or {}
    require(marker.get('policy_version') == POLICY_VERSION and marker.get('scope') == _scope(state)
            and marker.get('language') == state.get('language') and state.get('language') in ('en', 'it'),
            'start_observation_binding_changed')
    require(marker.get('packet_digest') == refs.digest(packet)
            and marker.get('symptom_digest') == refs.digest(state.get('symptom_text'))
            and marker.get('anchor') == _anchor(state, packet), 'start_observation_evidence_changed')
    require(marker.get('rejection') == _rejection(state)
            and not state.get('grounding_review') and not state.get('protocol_inconclusive'),
            'start_observation_rejection_changed')
    history = state.get('history')
    require(isinstance(history, list) and marker.get('history_digest') == refs.digest(history),
            'start_observation_history_changed')
    require(marker.get('phase') in ('active', 'completed'), 'start_observation_phase_invalid')
    if marker['phase'] == 'active':
        require(history == [], 'start_observation_active_history_invalid')
    else:
        require(len(history) <= 1, 'start_observation_terminal_history_invalid')
        if history:
            require(history[0].get('question') == _question(marker['language'], marker['anchor']['label']),
                    'start_observation_question_changed')
            _answer(state, history[0].get('answer'), 'unknown')
    step = _step(marker)
    require(state.get('status') == step['status'] and state.get('hypotheses') == []
            and state.get('current_question') == step['question'] and state.get('final_result') == step['final_result'],
            'start_observation_projection_changed')
    return step, [marker['anchor']['citation_id']]


def _answer(state, answer, value):
    marker = state[MARKER]
    question = _question(marker['language'], marker['anchor']['label'])
    option = next(item for item in question['options'] if item['id'] == value)
    require(isinstance(answer, dict) and answer.get('question_id') == question['question_id']
            and answer.get('api_value') == value and answer.get('value') == value
            and answer.get('label') == option['label_' + marker['language']],
            'start_observation_answer_changed')


def conclude(*, state, packet, answer=None):
    replay(state=state, packet=packet)
    if state[MARKER]['phase'] == 'completed':
        require(answer is None, 'start_observation_already_completed')
        return _step(state[MARKER]), deepcopy(state)
    updated = deepcopy(state)
    if answer is not None:
        _answer(state, answer, 'unknown')
        updated['history'].append({'question': deepcopy(state['current_question']), 'answer': deepcopy(answer)})
    marker = deepcopy(state[MARKER])
    marker.update(phase='completed', history_digest=refs.digest(updated['history']))
    return _update(updated, marker)


def continue_with_observation(*, state, packet, answer):
    replay(state=state, packet=packet)
    require(state[MARKER]['phase'] == 'active', 'start_observation_already_completed')
    _answer(state, answer, 'yes')
    require(isinstance(answer.get('free_text'), str) and answer['free_text'].strip(),
            'start_observation_existing_note_required')
    updated = deepcopy(state)
    updated['start_observation_origin'] = updated.pop(MARKER)
    # The normal Answer path owns generation, independent review and its budget.
    # No rejected hypothesis is provided to that path as a starting claim.
    return updated


def projection(*, state, step, packet):
    expected, ids = replay(state=state, packet=packet)
    require(step == expected, 'start_observation_response_changed')
    return ids, {'policy_version': POLICY_VERSION, 'outcome': 'observation_only' if not step['final_ready'] else 'inconclusive',
                 'current_proposal_accepted': False, 'exclusions_certified': False,
                 'literal_label_only': True, 'additional_provider_calls': 0,
                 'session_continuable': not step['final_ready']}


def validate_start_response(*, response, state, packet, expected_symptom):
    """Narrow exception to the ordinary first-turn hypothesis requirement.

    Caller must validate the emitted state HMAC and request scope first. A public
    flag alone cannot authorize this exception or a fabricated empty session.
    """
    step, ids = replay(state=state, packet=packet)
    require(state.get('symptom_text') == expected_symptom
            and state[MARKER]['phase'] == 'active' and response.get('ok') is True
            and response.get('status') == 'in_progress' and response.get('final_ready') is False
            and response.get('hypotheses') == [] and response.get('question') == step['question'],
            'start_observation_public_turn_changed')
    _, meta = projection(state=state, step=step, packet=packet)
    require((response.get('meta') or {}).get(MARKER) == meta
            and [item.get('citation_id') for item in response.get('citations') or []] == ids,
            'start_observation_public_evidence_changed')
    return True
