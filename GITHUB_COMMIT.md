# GitHub operation — Phase 3 Commit 1

Remain on the existing branch:

`refactor-phase1`

Extract the ZIP and copy the complete contents of the extracted folder over the current repository. Preserve the hidden `.git` directory and choose **Replace files in the destination** for matching files.

Expected production changes:

- `main.py`;
- new `machinemind/presentation/__init__.py`;
- new `machinemind/presentation/citations.py`.

Expected verification/documentation additions:

- `tools/presentation_contract_probe.py`;
- `tools/run_phase3a_gate.py`;
- `tests/phase3a_expected_presentation_contract.json`;
- `tests/phase3a_structure_manifest.json`;
- `PHASE3_COMMIT1_VALIDATION.json`;
- Phase 3 documentation, `README.md`, `GITHUB_COMMIT.md` and `SHA256SUMS.txt`.

These production files/modules must remain unchanged from Phase 2 Commit 5:

- `assistant_core_v2.py`;
- every module under `machinemind/api/`, `machinemind/config/`, `machinemind/core/` and `machinemind/infrastructure/`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`refactor: extract citation presentation and resource links`

Then use **Commit e push**. The configured trigger should build and deploy from `refactor-phase1`.

After deployment, apply `docs/LIVE_GATE_PHASE3_COMMIT1.md`. Do not start Commit 2 until the exact revision passes that live gate.
