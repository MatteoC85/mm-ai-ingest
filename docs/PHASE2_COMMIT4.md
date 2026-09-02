# Roadmap Phase 2 — Commit 4: semantic cache and knowledge versioning

## Parent state

This package is based exactly on the reviewed **Phase 2 Commit 3 — request budgets and execution guards** artifact.

- Branch: `refactor-phase1`
- Parent `main.py` SHA-256: `79192cb387d13a4ab0194d2e3387330199b7596e7fc8b1f3dfe2980f326bb5b2`
- Parent `main.py` lines: `36,213`
- Frozen original PROD commit: `89a33a549930003fc0761d7a3f47b70bc22e0c84`
- Frozen `assistant_core_v2.py` SHA-256: `dca19aba41becaffb7c0623f52dee22863c527dbbb7dc8ee965a724a25efd00d`

The Git commit created by the user's preceding push is not guessed in this artifact. Cloud Build and `/version` remain the source of truth for the deployed commit.

## Purpose

This is the fourth atomic commit of Roadmap Phase 2 (Infrastructure). It moves the current semantic response cache and per-company knowledge-version invalidation out of `main.py` without changing product behavior.

The migration rule remains:

> Move the current production behavior first. Change or simplify behavior only in a later, separately reviewed phase.

## Extracted module

### `machinemind/infrastructure/semantic_cache.py`

The module owns the existing production behavior for:

- cache bootstrap and optional DDL;
- bootstrap retry/cooldown state;
- per-company knowledge-version lookup and increment;
- invalidation after ingest, index and delete operations;
- deterministic cache-compatibility guards for identifiers, numbers, polarity and source constraints;
- exact normalized-query cache reuse;
- semantic cache lookup using the existing embedding and cosine threshold;
- signed/link refresh on cache hits;
- response quality scoring;
- canonical-response admission before cache storage;
- cache response sanitization;
- TTL and per-company row pruning;
- fail-open lookup, storage and invalidation behavior.

The module never imports `main`. Each compatibility wrapper passes the live `main` globals mapping so current configuration, DB/OpenAI callbacks, request-budget state and rollback/monkeypatch points remain late-bound.

## Historical compatibility retained in `main.py`

The following names and signatures remain available from `main`:

- `_v13_cache_bootstrap`;
- `_v13_normalize_query`;
- `_v13_scope_key`;
- `_v13_get_knowledge_version`;
- `_v13_bump_knowledge_version`;
- `_v13_invalidate_company_knowledge`;
- `_v13_cache_code_tokens`;
- `_v13_query_number_tokens`;
- `_v13_query_polarity_signature`;
- `_v13_query_source_signature`;
- `_v13_semantic_cache_compatible`;
- `_v13_jsonb_to_python`;
- `_v13_cache_lookup`;
- `_v13_response_quality`;
- `_assistant_core_cache_certified`;
- `_v13_cache_store`.

The mutable bootstrap variables remain in `main` and are updated through the live namespace:

- `_V13_CACHE_LOCK`;
- `_V13_CACHE_READY`;
- `_V13_CACHE_ERROR`;
- `_V13_CACHE_RETRY_AT`.

## Explicit non-changes

This commit does not intentionally change:

- cache enablement, DDL policy, TTL, scan limits or similarity thresholds;
- any SQL statement, conflict key, cleanup query or knowledge-version rule;
- identifier, numeric, polarity or source-preference compatibility logic;
- cache quality or canonical-response certification rules;
- embedding model, embedding transport or request-budget accounting;
- link refresh behavior;
- FastAPI routes or Pydantic request contracts;
- `assistant_core_v2.py`;
- prompts, model policy or semantic reasoning;
- retrieval, ranking, evidence admission or source priorities;
- Procedure/Step/P&S behavior;
- ASK, Root Cause or Smart Diagnostic behavior;
- PDF/XLSX ingest behavior;
- citations, rendering or Bubble response contracts;
- Docker, Cloud Build, requirements or promotion-gate files.

## Code movement

- Parent `main.py`: `36,213` lines
- Candidate `main.py`: `35,526` lines
- Net removal from `main.py`: `687` lines
- New semantic-cache module: `855` lines

The repository is not being minimized during extraction. Existing behavior is being isolated behind a normal importable module.

## Offline verification performed

### Cumulative prior contracts

- Python syntax compilation: PASS.
- Phase 1 API/request/scope/OpenAPI contract: PASS.
- Phase 2 Commit 1 configuration and database contracts: PASS.
- Phase 2 Commit 2 OpenAI transport contract: PASS.
- Phase 2 Commit 3 request-budget and execution contract: PASS.
- `assistant_core_v2.py` and all prior extracted production modules remain unchanged.

### Parent/candidate semantic-cache contract

The Commit 3 parent and this candidate produced identical normalized output for:

- all 16 historical cache function names, modules and signatures;
- query normalization and scope-key generation;
- code/identifier token guards;
- numeric and unit token guards;
- polarity and source-preference signatures;
- ASK and Root Cause semantic-cache compatibility decisions;
- JSONB decoding;
- ASK and Root Cause cache certification and quality scoring;
- disabled, already-ready, retry-cooldown, no-DDL, DDL-success and DDL-failure bootstrap paths;
- knowledge-version read, bump, blank-input and fail-open invalidation paths;
- exact cache hits and refreshed RG links;
- semantic cache hits and threshold handling;
- empty/incompatible/failing lookup paths;
- cache write bypass, certification, budget-vector, embedding, quality and DB-failure paths;
- persisted SQL parameter order, sanitized response JSON, TTL and pruning behavior;
- request-budget `semantic_cache` and route metadata.

### Static extraction guard

- All 668 top-level function/class names remain present.
- Exactly the 16 declared compatibility wrappers changed AST.
- No other top-level function/class changed.
- Semantic-cache DDL, lookup SQL and store SQL no longer remain duplicated in `main.py`.
- The extracted module does not import `main`.
- Candidate and module SHA-256 values are frozen in the gate.
- All previously extracted production modules and deployment files remain unchanged.

### Mutation sensitivity

The behavioral probe detected all three independent deliberate mutations:

1. exact identifier/code compatibility guard removed;
2. stale RG links retained in the stored cache response;
3. bootstrap retry window changed.

The reviewed source was restored before the final gate.

## What this verification does not prove

`OFFLINE_VERIFIED` is not a complete product certification. It does not replace:

- Cloud Build from the exact pushed commit;
- Cloud Run startup/import verification;
- real Cloud SQL DDL/read/write behavior;
- a real semantic-cache hit using the active embedding provider;
- production concurrency, latency or soak behavior;
- live ASK, Root Cause and Smart Diagnostic smoke cases;
- the complete release certification suite.

## Status

`OFFLINE_VERIFIED`

This commit becomes live-complete only when the exact pushed commit:

1. builds successfully from `refactor-phase1`;
2. produces a healthy newest Cloud Run revision at 100% traffic;
3. appears in `/version` as the active `commit_sha` with unchanged engine markers;
4. completes known-good ASK, Root Cause and Smart Diagnostic paths;
5. completes at least one repeated grounded request without cache/bootstrap/storage errors in Cloud Run logs.

This commit does not yet close all of Roadmap Phase 2. The remaining infrastructure boundary—Cloud Tasks and the shared external-file HTTP transport used by ingest—will be handled separately after this commit is live.

## Commit title

`refactor: extract semantic cache and knowledge versioning`
