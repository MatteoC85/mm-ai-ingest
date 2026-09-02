# GitHub operation — Phase 2 Commit 4

Remain on the existing branch:

`refactor-phase1`

Extract the ZIP and copy the complete contents of the extracted folder over the current repository. Preserve the hidden `.git` directory and choose **Replace files in the destination** for matching files.

Expected source changes:

- `main.py`;
- `machinemind/infrastructure/semantic_cache.py`;
- Phase 2D tests, probe, gate and documentation;
- validation and checksum manifests.

These production files must remain unchanged from Commit 3:

- `assistant_core_v2.py`;
- `machinemind/infrastructure/database.py`;
- `machinemind/infrastructure/openai_transport.py`;
- `machinemind/infrastructure/request_budget.py`;
- `machinemind/infrastructure/execution.py`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`refactor: extract semantic cache and knowledge versioning`

Then use **Commit e push**. The configured trigger will build and deploy from `refactor-phase1`.
