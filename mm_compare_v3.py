import json
import sys

before = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
after = json.load(open(sys.argv[2], 'r', encoding='utf-8'))

print('OVERALL')
for k, v in before['scores'].items():
    av = float(v)
    bv = float(after['scores'].get(k, 0.0))
    print(f'{k}: {av} -> {bv}   delta={round(bv-av, 1)}')

print('\nBY_BUCKET')
all_buckets = sorted(set(before.get('bucket_scores', {}).keys()) | set(after.get('bucket_scores', {}).keys()))
for bucket in all_buckets:
    print(f'[{bucket}]')
    b1 = before.get('bucket_scores', {}).get(bucket, {})
    b2 = after.get('bucket_scores', {}).get(bucket, {})
    keys = sorted(set(b1.keys()) | set(b2.keys()))
    for k in keys:
        av = float(b1.get(k, 0.0))
        bv = float(b2.get(k, 0.0))
        print(f'  {k}: {av} -> {bv}   delta={round(bv-av, 1)}')
