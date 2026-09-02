# Roadmap Phase 2 — Commit 3: request budgets and execution guards

## Parent state

This package is based exactly on the reviewed **Phase 2 Commit 2 — OpenAI provider transport** artifact.

- Branch: `refactor-phase1`
- Parent `main.py` SHA-256: `96f12237a3528e2f4ab06709d3c04707f6805ab2e1b8a9de2473bd6cbeac1800`
- Parent `main.py` lines: `36,587`
- Frozen original PROD commit: `89a33a549930003fc0761d7a3f47b70bc22e0c84`
- Frozen `assistant_core_v2.py` SHA-256: `dca19aba41becaffb7c0623f52dee22863c527dbbb7dc8ee965a724a25efd00d`

The Git commit created by the user's preceding push is not guessed in this artifact. Cloud Build and `/version` remain the source of truth for the deployed commit.

## Purpose

This is the third atomic commit of Roadmap Phase 2 (Infrastructure). It moves the existing request-scoped time/cost accounting and generic outer execution mechanics out of `main.py` without changing product behavior.

The extraction follows the fixed migration rule:

> Move the current production behavior first. Change or simplify behavior only in a later, separately reviewed phase.

## Extracted modules

### `machinemind/infrastructure/request_budget.py`

Owns the current:

- `_V13BudgetExceeded` exception;
- `_V13RequestBudget` state;
- `_V13_BUDGET_CTX` ContextVar;
- request deadline calculations;
- maximum LLM-call enforcement;
- bounded retry allowance after failed provider attempts;
- token and provider-usage accounting;
- cached-input and cache-write cost treatment;
- embedding cost accounting;
- model-rate selection;
- public runtime metadata.

During the migration it reads configuration through the live `main` namespace supplied at startup. This preserves environment-derived values and the current runtime behavior without importing `main` circularly.

### `machinemind/infrastructure/execution.py`

Owns the generic mechanics for:

- JSON whitespace heartbeat streaming;
- outer asyncio hard timeouts for ASK and Root Cause;
- validation of the returned backend payload type;
- copied-ContextVar execution for synchronous Smart Diagnostic turns;
- bounded thread waiting and timeout callback invocation.

Product-specific timeout/error envelopes remain controlled by the existing application callbacks. No diagnostic or answer policy was moved into this module.

## Historical compatibility retained in `main.py`

The following names remain available exactly where existing code expects them:

- `_V13BudgetExceeded`;
- `_V13RequestBudget`;
- `_V13_BUDGET_CTX`;
- `_v13_current_budget`;
- `_v13_model_rates`;
- `_v13_estimate_model_cost_usd`;
- `_v13_stream_json_response`;
- `_assistant_core_json_with_hard_timeout`;
- `_assistant_core_run_smart_with_hard_timeout`.

The three execution functions remain thin adapters because the current application still owns language selection and response-envelope construction.

## Explicit non-changes

This commit does not intentionally change:

- FastAPI routes or Pydantic request contracts;
- `assistant_core_v2.py`;
- model names, prompts, effort or reasoning policy;
- OpenAI request payloads, parsing, fallback order or errors;
- request deadlines, call limits, cost rates or cost formulas;
- retrieval, ranking, evidence admission or source priorities;
- Procedure/Step/P&S behavior;
- ASK, Root Cause or Smart Diagnostic reasoning;
- semantic cache or knowledge invalidation;
- SQL behavior;
- PDF/XLSX ingest behavior;
- citations, links or UI rendering;
- Docker, Cloud Build, requirements or promotion-gate files.

## Code movement

- Parent `main.py`: `36,587` lines
- Candidate `main.py`: `36,213` lines
- Net removal from `main.py`: `374` lines
- New request-budget module: `377` lines
- New execution module: `237` lines

The repository is not being minimized in this phase. Responsibilities are being isolated while preserving the current production contract.

## Offline verification performed

### Cumulative prior contracts

- Python syntax compilation: PASS.
- Phase 1 API/request/scope/OpenAPI contract: PASS.
- Phase 2 Commit 1 configuration and database contracts: PASS.
- Phase 2 Commit 2 OpenAI transport contract: PASS.
- `assistant_core_v2.py` SHA-256 unchanged.
- OpenAI transport module SHA-256 unchanged.

### Parent/candidate differential contract

The Phase 2 Commit 2 parent and this candidate produced identical normalized output for:

- 8 historical names, modules and signatures;
- 18 request-budget/cost/context scenarios;
- ASK, Root Cause and unknown-mode initial budget policy;
- deadline, call-cap, cost-cap and unaffordable-output failures;
- token, cached-token, cache-write, reasoning-token and embedding accounting;
- failed-call annotation and bounded retry allowance;
- late-bound runtime configuration changes;
- ContextVar bind/reset behavior;
- 8 asynchronous success/error/invalid/timeout execution scenarios;
- Smart Diagnostic synchronous success and timeout envelopes;
- Smart Diagnostic ContextVar propagation into its worker thread.

### Static extraction guard

- Five implementation definitions were removed from `main.py` and supplied by the new request-budget module.
- Only the three declared execution adapters changed bodies.
- Every other top-level function/class AST remains identical to the Commit 2 parent.
- Extracted modules do not import `main`.
- Candidate and module SHA-256 values are frozen in the gate.
- Frozen production files remain unchanged.

### Mutation sensitivity

The behavioral probe detected each independent deliberate mutation:

1. cached-input cost multiplier changed;
2. timeout response status changed;
3. copied ContextVar propagation removed.

The reviewed source was restored before the final gate was run.

## What this verification does not prove

`OFFLINE_VERIFIED` is not a full product certification. It does not replace:

- Cloud Build from the exact pushed commit;
- Cloud Run startup/import verification;
- real Cloud SQL and OpenAI-backed requests;
- production concurrency, latency or soak behavior;
- live ASK, Root Cause and Smart Diagnostic smoke cases;
- the complete release certification suite.

## Status

`OFFLINE_VERIFIED`

This commit becomes live-complete only when the exact pushed commit:

1. builds successfully from `refactor-phase1`;
2. produces a healthy newest Cloud Run revision at 100% traffic;
3. appears in `/version` as the active `commit_sha` with unchanged engine markers;
4. completes the existing ASK, Root Cause and Smart Diagnostic smoke paths.

This commit does not close all of Roadmap Phase 2. The remaining infrastructure boundary is handled separately after this commit is live.

## Commit title

`refactor: extract request budgets and execution guards`
