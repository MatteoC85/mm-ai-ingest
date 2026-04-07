import json
import hashlib
import re
from pathlib import Path
from collections import Counter

ROOT = Path('tmp')
ASK_MATRIX = ROOT / 'ask_matrix.jsonl'
RC_MATRIX = ROOT / 'rc_matrix.jsonl'
ASK_REPEAT = ROOT / 'ask_repeat'
RC_REPEAT = ROOT / 'rc_repeat'

EN_ASK_QUERIES = {
    'the machine vibrates',
    'the machine vibrates during bending',
    'it makes noise',
    'it makes noise during bending',
    'it jams',
    'it jams during feeding',
    'it does not start',
    'it does not start in automatic mode',
}
EN_RC_QUERIES = {
    'the machine vibrates',
    'the machine vibrates during bending',
    'it makes noise',
    'it makes noise during bending',
}
CRITICAL_QUERIES = {
    'la macchina vibra quando piega',
    'fa rumore durante piegatura',
    'the machine vibrates during bending',
    'it makes noise during bending',
}
STRUCTURED_PREFIXES = {'procedure', 'step', 'ps', 'md_photo', 'md_video'}
IT_MARKERS = {
    'il','lo','la','gli','le','di','del','della','dei','delle','con','per','quando','durante','mentre','dopo','prima','non','si','una','un','può','puo','quindi','documenti'
}
EN_MARKERS = {
    'the','with','for','when','during','while','after','before','not','does','is','are','can','cannot','will','would','should','documents','this','that','these','those','mode'
}

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def source_type_from_cid(cid: str) -> str:
    cid = str(cid or '').strip()
    if ':p' in cid:
        doc = cid.split(':p', 1)[0]
    else:
        doc = cid
    if ':' in doc:
        prefix = doc.split(':', 1)[0].strip().lower()
        if prefix in STRUCTURED_PREFIXES:
            return prefix
    return 'manual'

def text_lang(text: str) -> str:
    toks = re.findall(r"[a-zà-öø-ÿ']{2,}", (text or '').lower())
    it = sum(1 for t in toks if t in IT_MARKERS)
    en = sum(1 for t in toks if t in EN_MARKERS)
    if en > it:
        return 'en'
    if it > en:
        return 'it'
    return 'unknown'

def repeat_metrics(folder: Path, kind: str):
    files = sorted(folder.glob('*.json'))
    rows = [json.loads(f.read_text()) for f in files]
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0
    statuses = [str(r.get('status') or 'null') for r in rows]
    status_consistency = max(Counter(statuses).values()) / n

    citation_keys = []
    for r in rows:
        cids = sorted((c.get('citation_id') or '') for c in (r.get('citations') or []))
        citation_keys.append('|'.join(cids))
    citation_concentration = max(Counter(citation_keys).values()) / n

    if kind == 'ask':
        texts = [re.sub(r'\s+', ' ', str(r.get('answer') or '')).strip() for r in rows]
        answer_concentration = max(Counter(texts).values()) / n
        score = 50 * status_consistency + 30 * citation_concentration + 20 * answer_concentration
        return score, status_consistency, citation_concentration
    else:
        texts = []
        for r in rows:
            causes = [str(x.get('cause') or '').strip() for x in (r.get('possible_causes') or [])]
            texts.append(' | '.join(causes))
        cause_concentration = max(Counter(texts).values()) / n
        score = 40 * status_consistency + 30 * citation_concentration + 30 * cause_concentration
        return score, status_consistency, citation_concentration

def main():
    ask_rows = load_jsonl(ASK_MATRIX)
    rc_rows = load_jsonl(RC_MATRIX)
    ask_map = {r['q']: r for r in ask_rows}
    rc_map = {r['q']: r for r in rc_rows}

    AR = 100.0 * sum(1 for r in ask_rows if r.get('status') == 'answered') / max(1, len(ask_rows))
    RR = 100.0 * sum(1 for r in rc_rows if r.get('status') == 'answered') / max(1, len(rc_rows))

    en_ask = [ask_map[q] for q in EN_ASK_QUERIES if q in ask_map]
    AE = 100.0 * sum(1 for r in en_ask if text_lang(str(r.get('answer') or '')) == 'en') / max(1, len(en_ask))

    en_rc = [rc_map[q] for q in EN_RC_QUERIES if q in rc_map]
    rc_en_ok = 0
    for r in en_rc:
        causes = ' '.join(str(c) for c in (r.get('causes') or []))
        if text_lang(causes) == 'en':
            rc_en_ok += 1
    RE = 100.0 * rc_en_ok / max(1, len(en_rc))

    ax_hits = 0
    critical_total = 0
    for q in CRITICAL_QUERIES:
        if q not in ask_map or q not in rc_map:
            continue
        critical_total += 1
        ask_cids = ask_map[q].get('citations') or []
        rc_cids = rc_map[q].get('citations') or []
        ask_manual = any(source_type_from_cid(cid) == 'manual' for cid in ask_cids)
        rc_docs = {cid.split(':p', 1)[0] if ':p' in cid else cid for cid in rc_cids}
        ask_docs = {cid.split(':p', 1)[0] if ':p' in cid else cid for cid in ask_cids}
        if ask_manual or bool(ask_docs & rc_docs):
            ax_hits += 1
    AX = 100.0 * ax_hits / max(1, critical_total)

    ask_repeat_prompts = sorted([p for p in ASK_REPEAT.iterdir() if p.is_dir()])
    ask_repeat_scores = [repeat_metrics(p, 'ask')[0] for p in ask_repeat_prompts]
    AS = sum(ask_repeat_scores) / max(1, len(ask_repeat_scores))

    rc_repeat_prompts = sorted([p for p in RC_REPEAT.iterdir() if p.is_dir()])
    rc_repeat_scores = [repeat_metrics(p, 'rc')[0] for p in rc_repeat_prompts]
    RS = sum(rc_repeat_scores) / max(1, len(rc_repeat_scores))

    rag = round(0.30 * AR + 0.25 * AS + 0.20 * AE + 0.25 * AX)
    rcf = round(0.35 * RR + 0.35 * RS + 0.15 * RE + 0.15 * AX)
    perceived = round(0.30 * AR + 0.20 * AS + 0.20 * AX + 0.20 * RR + 0.10 * RE)
    moat = round(0.45 * rag + 0.45 * rcf + 0.10 * 85)

    print(json.dumps({
        'AR_ask_recall': round(AR, 2),
        'AS_ask_stability': round(AS, 2),
        'AE_ask_english_fidelity': round(AE, 2),
        'AX_ask_evidence_alignment': round(AX, 2),
        'RR_root_cause_recall': round(RR, 2),
        'RS_root_cause_stability': round(RS, 2),
        'RE_root_cause_english_fidelity': round(RE, 2),
        'piattaforma_RAG_runtime_score': rag,
        'root_cause_finder_runtime_score': rcf,
        'valore_diagnostico_percepito_runtime_score': perceived,
        'moat_competitivo_runtime_score': moat,
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
