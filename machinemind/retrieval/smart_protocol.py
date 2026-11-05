"""Claimless Smart closure after a validated but inadmissible diagnostic update.

This does not accept a rejected cause or certify exclusions. The old independent
proof is replayed only against its old history; a separate protocol checkpoint
binds the one new closed answer and the unchanged source packet.
"""
from copy import deepcopy
from . import review_references as refs
from . import smart_review as review


POLICY_VERSION = 'smart-no-admissible-update-inconclusive-v1'


def require(value, code):
    if not value:
        raise ValueError(code)


def _scope(state):
    result = {key: state.get(key) for key in ('company_id', 'machine_id', 'session_id')}
    require(all(isinstance(value, str) and value.strip() for value in result.values()),
            'protocol_scope_required')
    return result


def _closed_record(question, answer, language):
    require(isinstance(question, dict) and question.get('question_type') in ('single_choice', 'yes_no')
            and isinstance(question.get('question_text'), str) and question['question_text'].strip(),
            'protocol_closed_question_required')
    options = question.get('options')
    require(isinstance(options, list) and 2 <= len(options) <= 10
            and all(isinstance(option, dict) and isinstance(option.get('id'), str)
                    and option['id'].strip() for option in options), 'protocol_options_required')
    ids = [option['id'] for option in options]
    require(len(ids) == len(set(ids)), 'protocol_duplicate_options')
    require(isinstance(answer, dict) and answer.get('question_id') == question.get('question_id')
            and answer.get('value') == answer.get('api_value') and answer.get('api_value') in ids,
            'protocol_closed_answer_required')
    option = next(item for item in options if item['id'] == answer['api_value'])
    label = option.get('label_' + language) or option['id']
    require(isinstance(label, str) and label.strip() and answer.get('label') == label.strip(),
            'protocol_closed_label_mismatch')
    return {'question': deepcopy(question), 'answer': deepcopy(answer)}


def _question_sources(summary, seal, packet):
    require(summary.get('question_validated') is True and not summary.get('application_question'),
            'protocol_reviewed_question_required')
    question_index = len(seal['input_step']['hypotheses'])
    ids = list(dict.fromkeys(proof['citation_id'] for proof in summary.get('support_proofs') or []
                             if proof.get('proposal_index') == question_index))
    available = {item['citation_id'] for item in packet['validator_records']}
    require(ids and set(ids).issubset(available), 'protocol_question_source_proof_required')
    return ids


def prepare(*, state, packet, answer, language, probability_band):
    """Called after endpoint HMAC/scope validation, before history/enrichment."""
    signature = state.get('state_signature')
    require(isinstance(signature, str) and len(signature) == 64
            and all(char in '0123456789abcdef' for char in signature), 'protocol_signed_state_required')
    require(state.get('status') == 'in_progress' and not state.get('protocol_inconclusive'),
            'protocol_active_prior_state_required')
    _, summary = review.replay(state=state, packet=packet, probability_band=probability_band)
    record = _closed_record(state.get('current_question'), answer, language)
    history = list(state.get('history') or []) + [record]
    seal = state['grounding_review']
    return {'policy_version': POLICY_VERSION, 'scope': _scope(state), 'language': language,
            'prior_language': state.get('language', language), 'prior_review': deepcopy(seal),
            'prior_hypotheses': deepcopy(state.get('hypotheses') or []),
            'prior_history_count': len(history) - 1, 'history_digest': refs.digest(history),
            'packet_digest': refs.digest(packet),
            'question_citation_ids': _question_sources(summary, seal, packet)}


def _terminal(history, language):
    last = history[-1]
    question = last['question']['question_text']
    label = last['answer']['label']
    if language == 'en':
        text = ('The diagnosis remains inconclusive: no cause has been identified. '
                'Machine documentation is available, but the proposed update did not support a new reliable diagnostic conclusion. '
                'This does not confirm or exclude the earlier hypotheses.\n\n'
                'Record of a previous question, not an instruction to repeat a check:\n'
                + question + '\nRecorded answer: ' + label
                + '\nThis record requests no new inspection, movement, adjustment or intervention.')
    else:
        text = ('La diagnosi resta inconclusiva: la causa non è stata identificata. '
                'La documentazione della macchina è disponibile, ma l’aggiornamento proposto non ha consentito una nuova conclusione diagnostica affidabile. '
                'Questo risultato non conferma né esclude le ipotesi precedenti.\n\n'
                'Registrazione di una domanda precedente, non un invito a ripetere una verifica:\n'
                + question + '\nRisposta registrata: ' + label
                + '\nIl riepilogo non richiede nuove ispezioni, movimenti, regolazioni o interventi.')
    final = {'summary': text, 'most_likely_hypothesis_id': '', 'most_likely_label': '',
             'probability_pct': 0.0, 'probability_band': 'unknown', 'recommended_checks': []}
    return {'status': 'completed', 'final_ready': True, 'hypotheses': [], 'question': {},
            'operator_summary': text, 'final_result': final}


def conclude(*, context, state, packet, probability_band):
    rejection = state.get('grounding_review_meta') or {}
    require(rejection.get('outcome') == 'rejected' and rejection.get('decision_validated') is True
            and rejection.get('failure_reason') == 'no_supported_hypotheses',
            'protocol_validated_rejection_required')
    updated = deepcopy(state)
    updated.pop('grounding_review', None)
    updated.pop('grounding_review_debug', None)
    marker = deepcopy(context)
    marker['rejection'] = deepcopy(rejection)
    updated['protocol_inconclusive'] = marker
    updated['language'] = context['language']
    terminal, _ = _validate(updated, packet, probability_band)
    updated.update(status='completed', hypotheses=[], current_question={}, final_result=terminal['final_result'])
    return terminal, updated


def _validate(state, packet, probability_band):
    marker = state.get('protocol_inconclusive') or {}
    require(marker.get('policy_version') == POLICY_VERSION and marker.get('scope') == _scope(state),
            'protocol_binding_invalid')
    require(marker.get('language') == state.get('language') and marker.get('language') in ('en', 'it'),
            'protocol_language_changed')
    require(marker.get('packet_digest') == refs.digest(packet), 'protocol_source_packet_changed')
    history = state.get('history') or []
    require(isinstance(history, list) and history and marker.get('prior_history_count') == len(history) - 1
            and marker.get('history_digest') == refs.digest(history), 'protocol_history_changed')
    last = history[-1]
    require(isinstance(last, dict), 'protocol_closed_record_required')
    _closed_record(last.get('question'), last.get('answer'), marker['language'])
    rejection = marker.get('rejection') or {}
    require(rejection.get('outcome') == 'rejected' and rejection.get('decision_validated') is True
            and rejection.get('failure_reason') == 'no_supported_hypotheses'
            and rejection == state.get('grounding_review_meta'), 'protocol_rejection_changed')
    prior = {'grounding_review': marker.get('prior_review'), 'history': history[:-1],
             'hypotheses': marker.get('prior_hypotheses'), 'current_question': last['question'],
             'status': 'in_progress', 'symptom_text': state.get('symptom_text', ''),
             'language': marker.get('prior_language')}
    _, summary = review.replay(state=prior, packet=packet, probability_band=probability_band)
    ids = _question_sources(summary, prior['grounding_review'], packet)
    require(ids == marker.get('question_citation_ids'), 'protocol_question_sources_changed')
    return _terminal(history, marker['language']), ids


def replay(*, state, packet, probability_band):
    terminal, ids = _validate(state, packet, probability_band)
    require(state.get('status') == 'completed' and state.get('hypotheses') == []
            and state.get('current_question') == {} and state.get('final_result') == terminal['final_result']
            and not state.get('grounding_review'), 'protocol_terminal_claims_changed')
    return terminal, ids


def validate_projection(step, terminal):
    require(all(step.get(key) == terminal[key] for key in
                ('status', 'final_ready', 'hypotheses', 'operator_summary', 'final_result'))
            and not (step.get('question') or {}).get('question_text')
            and not (step.get('question') or {}).get('options') and '_review_debug' not in step,
            'protocol_projection_changed')
