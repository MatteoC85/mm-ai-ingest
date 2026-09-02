# Roadmap Phase 2 — Commit 1: runtime configuration and database boundary

## Parent state

- Git branch: `refactor-phase1`
- Parent live revision: `mm-ai-ingest-prod-00049-bgf`
- Parent Git commit: `9305c46` (short form shown by Cloud Build)
- Parent `main.py` SHA-256: `ed18b1c903493d8489350c2aa0dea5c73c9807d0ab21b2fb37d0cb1149d6fdba`
- Parent `main.py` lines: 37441

## Purpose

This is the first atomic commit of Roadmap Phase 2 (Infrastructure). It
mechanically extracts configuration and the low-level PostgreSQL connection
boundary from the Phase 1 code while preserving the historical names exported
by `main`.

This commit is intentionally **not** a functional improvement. It does not
change how MachineMind interprets a question, chooses evidence or writes an
answer.

## Extracted modules

- `machinemind/config/runtime.py`
  - ingest, PDF/XLSX, DB, OpenAI, ASK, structured-source and Root Cause settings;
  - regexes and source-type constants.
- `machinemind/config/assistant_runtime.py`
  - V13 and Assistant Core V2 settings, limits, models, release markers and
    engine-key generation.
- `machinemind/config/shadow_runtime.py`
  - V8 shadow endpoint settings.
- `machinemind/config/smart_diagnostic_runtime.py`
  - Smart Diagnostic settings.
- `machinemind/infrastructure/database.py`
  - PostgreSQL connection policy;
  - vector literal formatting;
  - table-column inspection.

`main.py` deliberately re-exports the existing configuration names. Existing
functions therefore continue to resolve names such as `ASK_SIM_THRESHOLD`,
`V13_ENGINE_KEY`, `SMART_DIAGNOSTIC_MODEL` and `DB_HOST` exactly as before.

The historical `_db_conn`, `_vector_literal` and `_get_table_columns` names also
remain in `main`; they are now thin compatibility adapters over the
infrastructure module.

## Explicit non-changes

This commit does not modify:

- API routes or request/response contracts;
- `assistant_core_v2.py`;
- prompts or OpenAI model choices;
- retrieval, ranking, evidence admission or source priorities;
- ASK, Root Cause or Smart Diagnostic behavior;
- PDF/XLSX ingest behavior;
- SQL executed by existing business operations;
- cache behavior;
- citation/link rendering;
- Cloud Build, Docker image contents or Python requirements.

## Offline verification performed

- Python syntax compilation: PASS.
- Existing Phase 1 runtime-contract probe: exact PASS.
- FastAPI OpenAPI JSON: exact parity with Phase 1.
- 315 configuration exports: exact default-value/type parity.
- 298 environment-variable inputs: extraction inventory complete.
- Three additional cross-domain environment scenarios: exact parity.
- Database connector: exact parity across 9 budget/assurance/error scenarios.
- Vector literal and table-column utilities: exact parity.
- Top-level definition inventory: 673/673 preserved.
- Unchanged function/class ASTs: 670/673.
- Intentional wrapper changes only:
  - `_db_conn`
  - `_vector_literal`
  - `_get_table_columns`
- `assistant_core_v2.py` SHA-256 unchanged:
  `dca19aba41becaffb7c0623f52dee22863c527dbbb7dc8ee965a724a25efd00d`.

## Status

`OFFLINE VERIFIED`

It becomes complete only after:

1. commit and push from `refactor-phase1`;
2. successful Cloud Build from that exact branch/commit;
3. healthy new Cloud Run revision at 100% traffic;
4. `/version` reports the new commit SHA while all engine/release markers remain
   unchanged;
5. a minimal live ASK/Root Cause/Smart Diagnostic smoke check confirms that the
   DB-backed paths still execute.

## Commit title

`refactor: extract runtime configuration and database infrastructure`
