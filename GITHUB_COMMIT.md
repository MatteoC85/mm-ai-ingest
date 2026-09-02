# GitHub operation — Phase 3 Commit 2

Remain on:

`refactor-phase1`

This package requires the current repository to contain the live-verified Phase
3 Commit 1 parent:

- commit `4a832e2`;
- revision `mm-ai-ingest-prod-00055-h7t`.

Extract the ZIP and copy the complete contents of the extracted folder over the
current repository. Preserve the hidden `.git` directory and choose **Replace
files in the destination** for matching files.

Expected production changes:

- modified `main.py`;
- new `machinemind/presentation/responses.py`.

The existing `machinemind/presentation/citations.py` must remain unchanged.

Expected verification/documentation additions:

- `tools/response_render_contract_probe.py`;
- `tools/run_phase3b_gate.py`;
- `tests/phase3b_expected_response_contract.json`;
- `tests/phase3b_parent_response_contract.json`;
- `tests/phase3b_expected_parent_candidate_diff.json`;
- `tests/phase3b_structure_manifest.json`;
- `tests/phase3b_mutation_results.json`;
- `PHASE3_COMMIT2_VALIDATION.json`;
- Phase 3 documentation, `README.md`, `GITHUB_COMMIT.md` and
  `SHA256SUMS.txt`.

These production files/modules must not be modified:

- `assistant_core_v2.py`;
- `machinemind/presentation/citations.py`;
- every module under `machinemind/api/`, `machinemind/config/`,
  `machinemind/core/` and `machinemind/infrastructure/`;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit message:

`refactor: extract response finalization and fix ordered-step rendering`

Then use **Commit e push**. The configured trigger should build and deploy from
`refactor-phase1`.

After deployment, apply `docs/LIVE_GATE_PHASE3_COMMIT2.md`. Do not begin Commit
3 until the exact revision passes that live gate.
