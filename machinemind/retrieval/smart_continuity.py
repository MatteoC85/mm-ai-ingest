"""Deterministic Smart session continuity, without new machine claims or I/O."""
from copy import deepcopy


POLICY_VERSION = 'smart-reviewed-continuity-v1'
REPORTING_VERSION = 'smart-report-existing-observation-v1'


def generation_answer(answer):
    """An Unknown note stays in signed history, but cannot become a fact in generation."""
    result = deepcopy(answer or {})
    value = str(result.get('api_value') or result.get('value') or '').strip().lower()
    if value == 'unknown':
        result['free_text'] = ''
    return result


def generation_history(history):
    result = deepcopy(history or [])
    for row in result:
        row['answer'] = generation_answer(row.get('answer'))
    return result


def reporting_question(*, language, number, hypothesis_ids):
    """Application-owned request for existing information, never a technical check."""
    english = language == 'en'
    return {
        'question_id': 'Q' + str(number), 'question_number': number,
        'question_type': 'single_choice', 'application_reporting_version': REPORTING_VERSION,
        'question_text': (
            'Do you already have another observation or alarm message to add in the notes?'
            if english else "Hai gia un'altra osservazione o un messaggio di allarme da aggiungere nelle note?"),
        'why_asked': (
            'The available information does not establish a safe next technical question. '
            'The remaining hypotheses are unconfirmed. Report only information you already have; '
            'this question requests no new inspection, test, movement or adjustment. '
            'If nothing else is available, choose Unknown; the diagnosis will remain inconclusive.'
            if english else
            'Le informazioni disponibili non consentono una prossima domanda tecnica sicura. '
            'Le ipotesi rimaste non sono confermate. Riporta soltanto informazioni gia disponibili: '
            'questa domanda non richiede nuove ispezioni, prove, movimenti o regolazioni. '
            'Se non hai altro da aggiungere, scegli Non so: la diagnosi restera non conclusiva.'),
        'safety_level': 'normal', 'safety_note': '',
        'options': [
            {'id': 'additional_observation', 'label_it': "Ho un'osservazione gia disponibile da aggiungere",
             'label_en': 'I have an existing observation to add'},
            {'id': 'unknown', 'label_it': 'Non so', 'label_en': "I don't know"},
        ],
        'target_hypotheses': list(hypothesis_ids),
    }


def reporting_was_answered(history):
    return any((row.get('question') or {}).get('application_reporting_version') == REPORTING_VERSION
               for row in history or [])


def confidence_projection(hypotheses):
    """Relative ranking weights are not calibrated event probabilities."""
    for hypothesis in hypotheses:
        hypothesis['probability_semantics'] = 'relative_weight_not_calibrated_probability'
        # Source entailment supplies no calibrated event likelihood. Accepting one
        # mechanism cannot turn its relative 100% weight into high confidence.
        # Keep the numeric legacy fields for the client's relative ranking.
        hypothesis['probability_band'] = 'unknown'
