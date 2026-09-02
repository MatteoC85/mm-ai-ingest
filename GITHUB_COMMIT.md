# GitHub operation — Phase 3 Commit 2 live fix 1

Remain on the current branch:

`refactor-phase1`

This is a **delta-only corrective package**. It assumes the initial Phase 3
Commit 2 has already been committed and deployed. Do not delete the repository
and do not copy any full snapshot.

Copy only the contents of this delta over the current repository. The only
production file intentionally modified is:

- `machinemind/presentation/responses.py`

The remaining changes are tests, validation and documentation for the live
numbering defect.

These production files must remain unchanged:

- `main.py`;
- `assistant_core_v2.py`;
- `machinemind/presentation/citations.py`;
- every module under `machinemind/api/`, `machinemind/config/`,
  `machinemind/core/` and `machinemind/infrastructure/`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`fix: preserve all source step numbers in Bubble procedure rendering`

Then use **Commit e push**. After deployment, run the exact PROC-001 question
from `docs/LIVE_GATE_PHASE3_COMMIT2.md`. Do not begin Phase 3 Commit 3 until the
visible response contains every Step 1 through 8 in order.
