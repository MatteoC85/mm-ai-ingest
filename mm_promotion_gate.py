import json
import sys

before = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
after = json.load(open(sys.argv[2], 'r', encoding='utf-8'))

hard_metrics = [
    'piattaforma_rag',
    'root_cause_finder_vero',
    'ask_grounded_quality',
    'valore_diagnostico_percepito',
]

regressions = []
for k in hard_metrics:
    av = float(before['scores'].get(k, 0.0))
    bv = float(after['scores'].get(k, 0.0))
    if bv < av:
        regressions.append(f'overall {k}: {av} -> {bv}')

for bucket, vals in before.get('bucket_scores', {}).items():
    after_vals = after.get('bucket_scores', {}).get(bucket, {})
    for k in ['root_cause', 'ask', 'rag', 'perceived']:
        av = float(vals.get(k, 0.0))
        bv = float(after_vals.get(k, 0.0))
        if bv + 0.01 < av:
            regressions.append(f'bucket {bucket} / {k}: {av} -> {bv}')

if regressions:
    print('FAIL')
    for r in regressions:
        print(r)
    sys.exit(1)

print('PASS')
for k in hard_metrics:
    av = float(before['scores'].get(k, 0.0))
    bv = float(after['scores'].get(k, 0.0))
    print(f'{k}: {av} -> {bv}   delta={round(bv-av, 1)}')
