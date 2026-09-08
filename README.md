# MachineMind — Behavior-Preserving Modularization

MachineMind is undergoing a staged refactoring of the AI backend with the goal of replacing the original monolithic implementation with a modular, testable and generalizable architecture while preserving existing product behavior, scope isolation and cost controls.

## Current status

Branch:

`refactor-phase1`

Current project stage:

**Phase 5 — Canonical Evidence Layer — IN PROGRESS**

Completed architectural phases:

- Phase 0 — Baseline freeze
- Phase 1 — API, scope and first modular boundaries
- Phase 2 — Infrastructure
- Phase 3 — Ingest and Presentation
- Phase 4 — Modular Retrieval

Phase 4 was formally closed after the final architectural gate verified the extraction of the retrieval subsystem and its behavior-preserving integration.

Current Phase 5 progress:

- P5-A — Canonical Evidence contracts and Evidence Manifest: **implemented and runtime verified**
- P5-B — Source-record adapters: **next step**
- P5-C — Manifest assembly and compatibility layer: pending
- P5-D — Final Canonical Evidence Layer gate: pending

## Architectural objective

The refactoring is designed around the following principle:

> AI reasoning should remain model-driven, while software deterministically guarantees scope, provenance, source integrity, contracts, resource limits and safe failure behavior.

The architecture must remain general across:

- different customers;
- different machines;
- different documents;
- Italian and English content and queries;
- new source combinations;
- previously unseen questions.

Machine-specific, benchmark-specific or question-specific hardcoding is not considered an acceptable solution.

## Current architecture

`main.py` remains the application composition root and HTTP entry point.

Retrieval responsibilities have been extracted into dedicated modules under:

`machinemind/retrieval/`

The extracted retrieval architecture includes:

- dense/vector retrieval;
- lexical / FTS retrieval;
- structured retrieval;
- candidate merge and ranking;
- query planning;
- retrieval policy and scoring;
- candidate admission;
- evidence orchestration;
- evidence assurance;
- source management;
- source priority;
- procedure-family selection;
- document readers;
- context expansion;
- diagnostic evidence handling;
- query fallbacks;
- retrieval primitives;
- structured-source parsing;
- legacy retrieval coordinators retained behind modular boundaries.

Phase 5 introduces the common evidence layer under:

`machinemind/evidence/`

Currently installed:

- `contracts.py`
- `manifest.py`

These modules define the canonical internal representation of evidence without yet changing the active ASK, Root Cause or Smart Diagnostic paths.

## Behavior-preserving rule

During architectural extraction:

- prompts are not intentionally changed;
- model selection is not intentionally changed;
- retrieval thresholds are not intentionally changed;
- source priorities are not intentionally changed;
- SQL scope rules are not intentionally changed;
- no new LLM call is introduced unless separately reviewed and approved;
- no new embedding call is introduced unless separately reviewed and approved;
- existing cost ceilings remain enforced;
- existing public API contracts remain compatible.

Behavioral improvements and architectural extraction are treated as separate changes whenever possible.

## Multi-tenant and scope guarantees

All AI paths must preserve the existing authorization boundaries for:

- `company_id`;
- `machine_id`;
- `document_ids`;
- `ai_scope`.

A retrieved source being semantically relevant is never sufficient to authorize its use.

Cross-company or unauthorized cross-machine evidence is not permitted.

## Cost controls

Request-level cost reservations are enforced before provider calls.

The current runtime contains separate limits for ASK, Root Cause and Smart Diagnostic requests.

The accounting model distinguishes:

- known provider usage;
- reserved cost for calls in progress;
- uncertain exposure when provider usage is unavailable after an error or timeout.

A failed or timed-out provider call must never be treated as free merely because usage information was not returned.

Refactoring must not silently increase:

- number of LLM calls;
- number of embedding calls;
- database queries;
- output-token limits;
- request budgets.

## Evidence and provenance

Evidence must preserve enough information to establish:

- source type;
- company and machine scope;
- document identity;
- page / row / sheet where applicable;
- structured-source identity;
- original source text;
- relations between sources;
- retrieval and ranking metadata;
- application references required to resolve the source.

Metadata for photos and videos must not be represented as visual analysis unless the media itself has actually been analyzed.

A citation proves provenance of a passage; it does not by itself prove that the model's causal interpretation is correct.

## Safe failure behavior

When evidence is insufficient, conflicting or not applicable, MachineMind must prefer abstention or clarification over unsupported claims.

Examples include:

- unknown machine identity;
- conflicting numeric values without a valid precedence rule;
- insufficient diagnostic observations;
- unidentified components where model-specific prescriptions would be unsafe;
- unsupported source-to-cause relationships.

A previously rejected AI answer must never be restored merely because it was more complete or had more citations.

## Known open issues

Architectural completion does not mean that all product-level AI behavior is certified.

Known open work includes:

- Root Cause final-review latency and timeout behavior;
- final validation of diagnostic quality after migration to the common Evidence Layer;
- generalization gates on unseen machines and unseen questions;
- wider multilingual validation beyond the currently tested Italian/English paths;
- Smart Diagnostic source-quality cleanup;
- final verification of application-level source-link resolution.

These issues remain explicit and must not be considered resolved merely because the relevant code has been modularized.

## Roadmap

Next phases:

### Phase 5 — Canonical Evidence Layer

Normalize Documents, XLSX, Procedures, Steps, Problems & Solutions and Photo/Video metadata behind a common evidence contract.

### Phase 6 — ASK on the common Core

Connect ASK to the Canonical Evidence Layer while preserving current capabilities and deterministic fact retrieval.

### Phase 7 — Root Cause on the common Core

Move Root Cause to the same evidence, retrieval, provenance and validation infrastructure.

### Phase 8 — Smart Diagnostic on the common Core

Move START, ANSWER and FINALIZE to the common architecture while preserving diagnostic session state.

### Phase 9 — Legacy heuristic reduction

Remove legacy keyword rules, bonuses, penalties, rescues and duplicate paths only when the common architecture demonstrably replaces them.

### Phase 10 — Final architectural cleanup

Remove dead code and reduce `main.py` toward bootstrap / composition-root responsibilities.

### Phase 11 — Electrical Evidence Provider

Integrate the electrical-schema compiler as an independent Evidence Provider.

### Phase 12 — Shadow, Canary and Rollout

Compare the new architecture against the production baseline, validate cost, latency, cache and critical failure modes, then perform controlled rollout.

## Development and deployment workflow

Source changes are committed and pushed through GitHub / GitHub Desktop.

Cloud Shell is used for:

- validation;
- runtime identity checks;
- test execution;
- report generation.

Cloud Shell must not be used to create commits or push source changes.

Each architectural block is validated with:

1. offline / deterministic tests;
2. precommit equivalence gates;
3. source-manifest verification;
4. runtime commit/revision verification after deployment.

Functional AI tests are executed separately when the change actually affects runtime behavior.

## Current principle

The objective is not to make a fixed benchmark pass.

The objective is a stable AI architecture where a new customer, machine, document set or valid question does not require a new retrieval architecture or machine-specific code.