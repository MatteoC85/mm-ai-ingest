# GitHub operation — Phase 2 Commit 5

Remain on the existing branch:

`refactor-phase1`

Extract the ZIP and copy the complete contents of the extracted folder over the current repository. Preserve the hidden `.git` directory and choose **Replace files in the destination** for matching files.

Expected source changes:

- `main.py`;
- new `machinemind/infrastructure/document_transport.py`;
- new `machinemind/infrastructure/cloud_tasks.py`;
- Phase 2E probe, gate, manifests and documentation;
- `PHASE2_COMMIT5_VALIDATION.json`;
- `README.md`, `GITHUB_COMMIT.md` and `SHA256SUMS.txt`.

These production files/modules must remain unchanged from Commit 4:

- `assistant_core_v2.py`;
- `machinemind/infrastructure/database.py`;
- `machinemind/infrastructure/openai_transport.py`;
- `machinemind/infrastructure/request_budget.py`;
- `machinemind/infrastructure/execution.py`;
- `machinemind/infrastructure/semantic_cache.py`;
- all `machinemind/config/` modules;
- `machinemind/api/contracts.py`;
- `machinemind/core/scope.py`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`refactor: extract external document transport and Cloud Tasks dispatch`

Then use **Commit e push**. The configured trigger should build and deploy from `refactor-phase1`.

After deployment, use `docs/LIVE_GATE_PHASE2_COMMIT5.md`. Phase 2 is not formally closed until that live gate passes.
