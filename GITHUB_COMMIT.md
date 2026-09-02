# GitHub operation — Phase 2 Commit 3

Remain on the existing branch:

`refactor-phase1`

Extract the ZIP and copy the complete contents of the extracted folder over the current repository. Preserve the hidden `.git` directory and choose **Replace files in the destination** for matching files.

Expected source changes:

- `main.py`;
- `machinemind/infrastructure/request_budget.py`;
- `machinemind/infrastructure/execution.py`;
- Phase 2C tests, probe, gate and documentation;
- validation and checksum manifests.

These production files must remain unchanged from Commit 2:

- `machinemind/infrastructure/openai_transport.py`;
- `assistant_core_v2.py`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`refactor: extract request budgets and execution guards`

Then use **Commit e push**. The configured trigger will build and deploy from `refactor-phase1`.
