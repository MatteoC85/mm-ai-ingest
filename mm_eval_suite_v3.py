import os
import sys
import json
import re
from urllib import request
from collections import defaultdict

AI_URL = os.environ["AI_URL"].rstrip("/")
AI_SECRET = os.environ["AI_SECRET"]
COMPANY_ID = os.environ["COMPANY_ID"]
MACHINE_ID = os.environ["MACHINE_ID"]
DOC_ID = os.environ["DOC_ID"]

CASES_PATH = sys.argv[1] if len(sys.argv) > 1 else "mm_eval_suite_cases.jsonl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "eval_suite_current.json"


def post_json(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        AI_URL + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-AI-Internal-Secret": AI_SECRET,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_any(text: str, arr):
    t = norm(text)
    return any(norm(x) in t for x in (arr or []))


def likely_language_ok(text: str, lang: str) -> bool:
    txt = f" {norm(text)} "
    if lang == "en":
        italian_markers = [" la ", " il ", " quando ", " durante ", " macchina ", " piega ", " rumore "]
        return sum(1 for m in italian_markers if m in txt) <= 1
    if lang == "it":
        english_markers = [" the ", " during ", " when ", " machine ", " noise ", " bending "]
        return sum(1 for m in english_markers if m in txt) <= 1
    return True


def evaluate_case(case: dict, resp: dict) -> dict:
    endpoint = case["endpoint"]
    lang = case.get("lang", "it")
    good = case.get("good_if_contains_any", [])
    bad = case.get("should_not_contain", [])
    status = str(resp.get("status") or "")

    if endpoint == "ask":
        body = str(resp.get("answer") or "")
    else:
        summary = str(resp.get("problem_summary") or "")
        causes = " | ".join([str(c.get("cause") or "") for c in (resp.get("possible_causes") or [])])
        body = f"{summary} || {causes}"

    good_hit = contains_any(body, good)
    bad_hit = contains_any(body, bad)
    lang_ok = likely_language_ok(body, lang)
    citations_n = len(resp.get("citations") or [])
    sim = float((((resp.get("debug") or {}).get("similarity_max")) or resp.get("similarity_max") or 0.0))

    root_score = 0.0
    if endpoint == "root-cause":
        if status == "answered":
            if good_hit and not bad_hit and lang_ok:
                root_score = 1.0
            elif good_hit and lang_ok:
                root_score = 0.6
            elif (not bad_hit) and lang_ok:
                root_score = 0.25
            else:
                root_score = 0.0

    ask_score = 0.0
    if endpoint == "ask":
        if status == "answered":
            if good_hit and not bad_hit and lang_ok:
                ask_score = 1.0
            elif good_hit and lang_ok:
                ask_score = 0.6
            elif (not bad_hit) and lang_ok:
                ask_score = 0.25
            else:
                ask_score = 0.0

    perceived = 0.0
    if status == "answered":
        perceived = 0.50
        if good_hit:
            perceived += 0.22
        if bad_hit:
            perceived -= 0.20
        if lang_ok:
            perceived += 0.08
        if citations_n >= 1:
            perceived += 0.08
        if endpoint == "root-cause" and len(resp.get("possible_causes") or []) >= 1:
            perceived += 0.07
        if endpoint == "ask" and body:
            perceived += 0.05
    perceived = max(0.0, min(1.0, perceived))

    rag = 0.0
    if status == "answered":
        rag = 0.42
        rag += min(sim, 0.70) * 0.30
        if citations_n >= 1:
            rag += 0.12
        if lang_ok:
            rag += 0.06
        if endpoint == "root-cause" and len(resp.get("possible_causes") or []) >= 1:
            rag += 0.08
        if endpoint == "ask" and body:
            rag += 0.05
    rag = max(0.0, min(1.0, rag))

    return {
        "endpoint": endpoint,
        "bucket": case.get("bucket", "misc"),
        "lang": lang,
        "query": case["query"],
        "status": status,
        "body": body,
        "similarity_max": sim,
        "citations_n": citations_n,
        "good_hit": good_hit,
        "bad_hit": bad_hit,
        "lang_ok": lang_ok,
        "root_cause_score": root_score,
        "ask_score": ask_score,
        "perceived_score": perceived,
        "rag_score": rag,
    }


cases = []
with open(CASES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

rows = []
for case in cases:
    payload = {
        "query": case["query"],
        "company_id": COMPANY_ID,
        "machine_id": MACHINE_ID,
        "document_ids": [DOC_ID],
        "top_k": 8 if case["endpoint"] == "root-cause" else 5,
        "debug": True,
    }
    if case["endpoint"] == "root-cause":
        payload["max_causes"] = 3
        path = "/v1/ai/root-cause"
    else:
        path = "/v1/ai/ask"

    try:
        resp = post_json(path, payload)
    except Exception as e:
        rows.append({
            "endpoint": case["endpoint"],
            "bucket": case.get("bucket", "misc"),
            "lang": case.get("lang", "it"),
            "query": case["query"],
            "status": "error",
            "body": str(e),
            "similarity_max": 0.0,
            "citations_n": 0,
            "good_hit": False,
            "bad_hit": False,
            "lang_ok": False,
            "root_cause_score": 0.0,
            "ask_score": 0.0,
            "perceived_score": 0.0,
            "rag_score": 0.0,
        })
        continue

    rows.append(evaluate_case(case, resp))

root_rows = [r for r in rows if r["endpoint"] == "root-cause"]
ask_rows = [r for r in rows if r["endpoint"] == "ask"]
all_rows = rows[:]


def avg(arr, key):
    if not arr:
        return 0.0
    return sum(float(x.get(key, 0.0) or 0.0) for x in arr) / len(arr)

scores = {
    "piattaforma_rag": round(avg(all_rows, "rag_score") * 100, 1),
    "root_cause_finder_vero": round(avg(root_rows, "root_cause_score") * 100, 1),
    "ask_grounded_quality": round(avg(ask_rows, "ask_score") * 100, 1),
    "valore_diagnostico_percepito": round(avg(all_rows, "perceived_score") * 100, 1),
}

bilingual_ok = sum(1 for r in all_rows if r.get("lang_ok")) / max(1, len(all_rows))
moat = (
    0.28 * scores["piattaforma_rag"] +
    0.30 * scores["root_cause_finder_vero"] +
    0.22 * scores["ask_grounded_quality"] +
    0.12 * scores["valore_diagnostico_percepito"] +
    0.08 * (bilingual_ok * 100)
)
scores["moat_competitivo_attuale"] = round(moat, 1)

by_bucket = defaultdict(lambda: {"rows": [], "rag": 0.0, "perceived": 0.0, "root": 0.0, "ask": 0.0})
for r in all_rows:
    b = by_bucket[r["bucket"]]
    b["rows"].append(r)

bucket_scores = {}
for bucket, entry in by_bucket.items():
    arr = entry["rows"]
    bucket_scores[bucket] = {
        "rag": round(avg(arr, "rag_score") * 100, 1),
        "perceived": round(avg(arr, "perceived_score") * 100, 1),
        "root_cause": round(avg([r for r in arr if r["endpoint"] == "root-cause"], "root_cause_score") * 100, 1),
        "ask": round(avg([r for r in arr if r["endpoint"] == "ask"], "ask_score") * 100, 1),
    }

out = {
    "scores": scores,
    "bucket_scores": bucket_scores,
    "rows": rows,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(json.dumps(scores, ensure_ascii=False, indent=2))
