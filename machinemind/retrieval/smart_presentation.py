"""Public Smart confidence fields; reviewed, signed diagnostic state stays private.

Relative ranking weights are not calibrated probabilities. This module accepts
only server-reviewed structures after signing; it neither evaluates sources nor
changes an observation, hypothesis status, check, order or internal score.
"""
from copy import deepcopy


def project(*, final_result, hypotheses, final_ready, status, language):
    public_final = deepcopy(final_result)
    public_hypotheses = deepcopy(hypotheses)
    active = [h for h in hypotheses if isinstance(h, dict)
              and h.get('status') not in ('excluded', 'rejected')]
    # Preserve the existing supported-confirmation contract. Unknown answers do
    # not manufacture a confirmed status; only the existing reviewed state can.
    if status == 'no_sources':
        return public_final, public_hypotheses, None
    for hypothesis in public_hypotheses:
        if (isinstance(hypothesis, dict)
                and hypothesis.get('status') not in ('confirmed', 'excluded', 'rejected')
                and hypothesis.get('probability_band') == 'unknown'):
            hypothesis['relative_weight_pct'] = hypothesis.get('probability_pct', 0.0)
            hypothesis['probability_pct'] = 0.0
            # score_raw, ordering, conditionals and source-bound checks remain
            # available as relative ranking information, never a likelihood.
    selected_id = str(final_result.get('most_likely_hypothesis_id') or '')
    selected_confirmed = any(h.get('id') == selected_id and h.get('status') == 'confirmed' for h in active)
    if not final_ready or selected_confirmed:
        return public_final, public_hypotheses, None
    public_final.update(
        most_likely_hypothesis_id='',
        most_likely_label=('No cause identified' if str(language or '').lower().startswith('en')
                           else 'Nessuna causa identificata'),
        probability_pct=0.0, probability_band='unknown')
    for alternative in public_final.get('alternative_hypotheses') or []:
        if isinstance(alternative, dict) and alternative.get('probability_band') == 'unknown':
            alternative['relative_weight_pct'] = alternative.get('probability_pct', 0.0)
            alternative['probability_pct'] = 0.0
    return public_final, public_hypotheses, 'inconclusive'
