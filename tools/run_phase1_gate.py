#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[1]
baseline=pathlib.Path(os.environ.get("MM_BASELINE_ROOT","/mnt/data/_mm_prod_baseline/mm-ai-ingest-prod"))
probe=root/"tools"/"contract_probe.py"

def run(path):
    env=dict(os.environ); env["PYTHONPATH"]=str(path)
    cp=subprocess.run([sys.executable,str(probe)],cwd=path,env=env,text=True,capture_output=True)
    if cp.returncode:
        raise SystemExit(cp.stdout+"\n"+cp.stderr)
    return json.loads(cp.stdout)

before=run(baseline); after=run(root)
if before != after:
    out=root/"tests"/"phase1_contract_diff.json"
    out.write_text(json.dumps({"before":before,"after":after},indent=2,ensure_ascii=False),"utf-8")
    print("FAIL: phase 1 runtime contract differs; see",out)
    raise SystemExit(1)
print("PASS: routes, request contracts, scope results and markers are unchanged")
