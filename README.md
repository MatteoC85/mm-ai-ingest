# MachineMind behavior-preserving modularization
## Roadmap Phase 3 — Commit 2 live fix 1

This delta corrects the live Bubble rendering defect found after the initial
Phase 3 Commit 2 deployment.

The initial offline gate inspected raw HTML list-value attributes. The real UI
showed that this was insufficient: every rendered item still appeared as `1.`
and the safety-oriented source Step 1 was moved into a separate “Prima di
iniziare” block.

The correction is restricted to:

`machinemind/presentation/responses.py`

It now:

- keeps every source-authored Procedure Step inside the ordered sequence;
- preserves the original source Step number instead of renumbering a reduced
  subset;
- emits `1.`, `2.`, … as visible escaped text rather than relying on browser
  generated list markers or `<li value>` attributes;
- applies the same robust numbering to structured, generic and lossless
  renderers, including cached canonical answers re-finalized by the current
  revision;
- leaves retrieval, reasoning, sources, links, Root Cause, Smart Diagnostic and
  ingest unchanged.

Status: **OFFLINE_VERIFIED — LIVE FIX PENDING**

Offline gate:

`python tools/run_phase3b_gate.py`

Primary documents:

- `docs/PHASE3_COMMIT2_LIVE_FIX1.md`;
- `docs/LIVE_GATE_PHASE3_COMMIT2.md`;
- `PHASE3_COMMIT2_LIVE_FIX1_VALIDATION.json`;
- `GITHUB_COMMIT.md`.
