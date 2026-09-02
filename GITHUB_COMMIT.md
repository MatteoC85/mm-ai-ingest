# GitHub operation — Roadmap Phase 2, commit 1

Branch: `refactor-phase1`

## Apply without terminal

1. Extract `MachineMind_Modularization_Phase2_Commit1_Runtime_Config_DB.zip`.
2. Copy all files and folders inside the extracted directory over the current
   `mm-ai-ingest` repository.
3. Allow Windows/VS Code to replace files with the same name.
4. Do not delete the hidden `.git` directory.
5. Confirm that the active branch remains `refactor-phase1`.

Expected meaningful code changes:

- `main.py` modified;
- four new files under `machinemind/config/`;
- new `machinemind/infrastructure/` package;
- validation, documentation and offline-gate files added.

The following production files must remain unchanged:

- `assistant_core_v2.py`;
- `cloudbuild.yaml`;
- `Dockerfile`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

Commit title:

`refactor: extract runtime configuration and database infrastructure`

After **Commit e push**, the configured Cloud Build trigger should build the
`refactor-phase1` commit and deploy it to `mm-ai-ingest-prod`.
