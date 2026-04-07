#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
from collections import Counter

BASE_DIR = Path('tmp')


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def file_hash_counts(files, extractor):
    vals = []
    for f in files:
        obj = json.loads(Path(f).read_text())
        vals.append(extractor(obj))
    return Counter(vals)


def concentration(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return max(counter.values()) / total


def status_concentration(files):
    c = Counter()
    for f in files:
        obj = json.loads(Path(f).read_text())
        c[str(obj.get('status') or 'null')] += 1
    return concentration(c)


def answer_hash(obj):
    text = ' '.join(str(obj.get('answer') or '').split())
    return hashlib.sha256(text.encode()).hexdigest()


def cause_hash(obj):
    text = ' | '.join(obj.get('possible_causes') and [c.get('cause','') for c in obj.get('possible_causes',[])] or [])
    text = ' '.join(text.split())
    return hashlib.sha256(text.encode()).hexdigest()


def citation_hash(obj):
    cits = sorted([c.get('citation_id','') for c in obj.get('citations',[])])
    text = '|'.join(cits)
    return hashlib.sha256(text.encode()).hexdigest()


def weighted_answered(rows, important_queries):
    if not rows:
        return 0.0
    total_w = 0.0
    hit_w = 0.0
    for r in rows:
        q = r.get('q','')
        w = 1.4 if q in important_queries else 1.0
        total_w += w
        if r.get('status') == 'answered':
            hit_w += w
    return 100.0 * hit_w / total_w if total_w else 0.0


def bilingual_consistency(ask_rows):
    by_q = {r['q']: r for r in ask_rows if 'q' in r}
    pairs = [
        ('la macchina vibra', 'the machine vibrates'),
        ('la macchina vibra quando piega', 'the machine vibrates during bending'),
        ('fa rumore durante piegatura', 'it makes noise during bending'),
        ('non parte in automatico', 'it does not start in automatic mode'),
    ]
    scores = []
    for a, b in pairs:
        ra, rb = by_q.get(a), by_q.get(b)
        if not ra or not rb:
            continue
        s = 0.0
        if ra.get('status') == rb.get('status') == 'answered':
            s += 0.6
        elif ra.get('status') == rb.get('status'):
            s += 0.35
        if bool(ra.get('citations')) == bool(rb.get('citations')):
            s += 0.2
        if isinstance(ra.get('answer'), str) and isinstance(rb.get('answer'), str):
            s += 0.2
        scores.append(min(1.0, s))
    return 100.0 * (sum(scores) / len(scores)) if scores else 0.0


def ask_stability_score(base: Path):
    dirs = [
        base / 'la_macchina_vibra_quando_piega',
        base / 'the_machine_vibrates_during_bending',
        base / 'fa_rumore_durante_piegatura',
        base / 'it_makes_noise_during_bending',
    ]
    comps = []
    for d in dirs:
        files = sorted(d.glob('*.json'))
        if not files:
            continue
        status_s = status_concentration(files)
        ans_s = concentration(file_hash_counts(files, answer_hash))
        cit_s = concentration(file_hash_counts(files, citation_hash))
        comps.append(100.0 * (0.35 * status_s + 0.30 * cit_s + 0.35 * ans_s))
    return sum(comps) / len(comps) if comps else 0.0


def rc_stability_score(base: Path):
    dirs = [
        base / 'la_macchina_vibra_quando_piega',
        base / 'the_machine_vibrates_during_bending',
        base / 'fa_rumore_durante_piegatura',
        base / 'it_makes_noise_during_bending',
    ]
    comps = []
    for d in dirs:
        files = sorted(d.glob('*.json'))
        if not files:
            continue
        status_s = status_concentration(files)
        cause_s = concentration(file_hash_counts(files, cause_hash))
        cit_s = concentration(file_hash_counts(files, citation_hash))
        comps.append(100.0 * (0.30 * status_s + 0.35 * cit_s + 0.35 * cause_s))
    return sum(comps) / len(comps) if comps else 0.0


def main():
    ask_rows = load_jsonl(BASE_DIR / 'ask_matrix.jsonl')
    rc_rows = load_jsonl(BASE_DIR / 'rc_matrix.jsonl')

    important_ask = {
        'la macchina vibra quando piega',
        'the machine vibrates during bending',
        'fa rumore durante piegatura',
        'it makes noise during bending',
    }
    important_rc = {
        'la macchina vibra quando piega',
        'the machine vibrates during bending',
        'fa rumore durante piegatura',
        'it makes noise during bending',
    }

    ask_answered = weighted_answered(ask_rows, important_ask)
    rc_answered = weighted_answered(rc_rows, important_rc)
    bilingual = bilingual_consistency(ask_rows)
    ask_stab = ask_stability_score(BASE_DIR / 'ask_repeat')
    rc_stab = rc_stability_score(BASE_DIR / 'rc_repeat')

    platform = round(0.36 * ask_answered + 0.22 * bilingual + 0.22 * ask_stab + 0.20 * rc_answered)
    root_cause = round(0.58 * rc_answered + 0.42 * rc_stab)
    diagnostic = round(0.34 * ask_answered + 0.16 * bilingual + 0.22 * ask_stab + 0.28 * root_cause)
    moat = round(0.72 * 84 + 0.28 * ((platform + root_cause + diagnostic) / 3.0))

    print(json.dumps({
        'metrics': {
            'ask_answered_score': round(ask_answered, 2),
            'root_cause_answered_score': round(rc_answered, 2),
            'bilingual_consistency_score': round(bilingual, 2),
            'ask_stability_score': round(ask_stab, 2),
            'root_cause_stability_score': round(rc_stab, 2),
        },
        'ratings': {
            'piattaforma_RAG': platform,
            'root_cause_finder_vero': root_cause,
            'valore_diagnostico_percepito': diagnostic,
            'moat_competitivo_attuale': moat,
        }
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
