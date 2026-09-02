# Roadmap Phase 2 — Commit 5: external document transport and Cloud Tasks dispatch

## Parent state

This package is based exactly on the repository snapshot deployed for **Phase 2 Commit 4 — semantic cache and knowledge versioning**.

- Branch: `refactor-phase1`
- Parent Git commit: `f13f8dda3bbe3be58aaa2bb86459e83403ebbcb3b`
- Parent live revision: `mm-ai-ingest-prod-00053-rfv`
- Parent `main.py` SHA-256: `2c7e4750a985a4810cfb74330e5d851e2ebfd95d8908d9ff30de4c781afa074a`
- Parent `main.py` lines: `35,526`
- Frozen original PROD commit: `89a33a549930003fc0761d7a3f47b70bc22e0c84`
- Frozen `assistant_core_v2.py` SHA-256: `dca19aba41becaffb7c0623f52dee22863c527dbbb7dc8ee965a724a25efd00d`

The parent Git commit and revision above come from the Cloud Build and `/version` evidence supplied after Commit 4 was deployed.

## Purpose

This is the fifth and final atomic commit of Roadmap **Phase 2 — Infrastructure**. It moves the remaining shared external-I/O mechanics out of `main.py` without changing the product behavior:

1. loading an ingest document from base64 or an external URL;
2. dispatching asynchronous document indexing through Google Cloud Tasks.

The migration rule remains unchanged:

> Move the current production behavior first. Change or simplify behavior only in a later, separately reviewed phase.

## Extracted modules

### `machinemind/infrastructure/document_transport.py`

The module owns the existing behavior for:

- stripping a `data:` URL prefix;
- strict base64 validation and decoding;
- preserving the historical HTTP 400 response for invalid base64;
- deriving a filename from an external URL;
- normalizing protocol-relative URLs from `//...` to `https://...`;
- fetching the external document with the existing `FETCH_TIMEOUT`;
- preserving the historical HTTP 502 response for a failed fetch;
- reading `Content-Type` and `Content-Disposition`;
- preserving filename and extension selection rules;
- returning the same ingest-file dictionary and `source_mode` values.

The module does not import `main`. `main.py` passes its current live dependencies into the module so the existing timeout, `requests.get`, `HTTPException`, URL helpers and monkeypatch points remain late-bound.

### `machinemind/infrastructure/cloud_tasks.py`

The module owns the existing behavior for:

- project precedence: `GOOGLE_CLOUD_PROJECT` → `GCP_PROJECT` → default project;
- Cloud Tasks location and queue defaults;
- service URL precedence: `SERVICE_URL` → `K_SERVICE_URL`;
- queue-path construction;
- the exact async indexing payload;
- the exact `/v1/ai/index/document` target URL;
- `Content-Type` and `X-AI-Internal-Secret` headers;
- JSON body encoding;
- `create_task` invocation;
- fail-open logging with `CLOUD_TASKS_ENQUEUE_SKIPPED`.

The module does not import `main`, Google Cloud configuration or application secrets directly. Those live dependencies are supplied by the composition root on each call.

## Historical compatibility retained in `main.py`

The following historical helper names and signatures remain available from `main`:

- `_strip_data_url_prefix(value: str) -> str`;
- `_decode_file_base64(file_base64: str) -> bytes`;
- `_detect_filename_from_url(url: str) -> str`;
- `_load_ingest_document_file(payload: IngestRequest, bubble_document_id: str) -> dict`.

A new internal adapter is introduced only to name the previously inline task-dispatch block:

- `_enqueue_document_index_task(...) -> None`.

`ingest_document` still performs the same sequence and invokes task dispatch at the same point: after page persistence, knowledge invalidation and ledger preparation, and before the final month-usage response is assembled.

## Explicit non-changes

This commit does not intentionally change:

- supported document formats;
- PDF/XLSX parsing, cleaning, limits, quota or metering;
- document page persistence;
- ingest response fields or status values;
- the Cloud Tasks queue, project, region, service URL or payload defaults;
- Cloud Tasks fail-open behavior;
- FastAPI routes or Pydantic request contracts;
- `assistant_core_v2.py`;
- runtime configuration;
- database infrastructure;
- OpenAI transport;
- request budgets, cost accounting, heartbeat or hard timeouts;
- semantic cache or knowledge-version behavior;
- prompts, models or reasoning policy;
- retrieval, ranking, evidence admission or source priorities;
- ASK, Root Cause or Smart Diagnostic behavior;
- citations, links or rendering;
- `Dockerfile`, `cloudbuild.yaml`, `requirements.txt` or `mm_promotion_gate.py`.

## Code movement

- Parent `main.py`: `35,526` lines
- Candidate `main.py`: `35,464` lines
- Net removal from `main.py`: `62` lines
- New document-transport module: `148` lines
- New Cloud Tasks module: `87` lines

The objective is responsibility separation, not minimizing the repository line count during extraction.

## Offline verification performed

### Cumulative prior contracts

The candidate re-ran and passed the cumulative non-structural contracts for:

- Phase 1 API, request, scope and OpenAPI behavior;
- Phase 2 Commit 1 runtime configuration and database behavior;
- Phase 2 Commit 2 OpenAI provider transport;
- Phase 2 Commit 3 request budgets and execution guards;
- Phase 2 Commit 4 semantic cache and knowledge versioning;
- the frozen `assistant_core_v2.py` SHA-256.

### Parent/candidate external-I/O contract

The actual Commit 4 parent and this candidate produced identical normalized output across:

- 30 external-file transport rows;
- 6 full ingest-to-Cloud-Tasks scenarios;
- all four historical helper signatures and module identities;
- plain, spaced, uppercase and multi-comma data-URL inputs;
- valid, empty, malformed and incorrectly padded base64;
- ordinary, encoded, protocol-relative, trailing-slash and fragment URLs;
- late-bound URL, base64 and filename helper replacement;
- base64 filename precedence and document-ID fallback;
- HTTP success with and without `Content-Disposition`;
- `FETCH_TIMEOUT` propagation;
- network failure normalization to HTTP 502;
- FastAPI `HTTPException` passthrough;
- ingest DB operation sequence and response shape;
- explicit environment configuration and fallback precedence;
- default project, location and queue;
- missing service URL, queue-path failure and task-creation failure;
- exact task parent, URL, headers and JSON body;
- fail-open logging and continued successful ingest response.

The expected contract was generated from the actual Commit 4 parent implementation, not from the candidate.

### Static extraction guard

- Parent top-level function/class inventory: `668`.
- Candidate inventory: `669`.
- No parent definition was removed.
- Exactly one new internal definition was added: `_enqueue_document_index_task`.
- Exactly five existing definitions changed as declared:
  - `_strip_data_url_prefix`;
  - `_decode_file_base64`;
  - `_detect_filename_from_url`;
  - `_load_ingest_document_file`;
  - `ingest_document`.
- No other top-level function or class changed AST.
- Direct base64 decoding, external HTTP loading and Cloud Tasks implementation no longer remain duplicated in `main.py`.
- Both extracted modules are importable and do not import `main`.
- Candidate and module SHA-256 values are frozen in the gate.
- All prior extracted production modules, including `semantic_cache.py`, remain byte-identical to Commit 4.

### Mutation sensitivity

The behavioral contract detected all three deliberate mutations:

1. removal of protocol-relative URL normalization;
2. change of the default Cloud Tasks queue;
3. change of the live `FETCH_TIMEOUT` passed by the composition root.

The reviewed source was restored and the complete Phase 2E gate was re-run successfully.

## What this verification does not prove

`OFFLINE_VERIFIED` is not a complete live product certification. It does not replace:

- Cloud Build from the exact pushed commit;
- Cloud Run startup/import verification;
- a real external file fetch through the deployed service;
- a real Cloud Tasks enqueue and subsequent index request;
- a real PDF/XLSX ingestion completion;
- production concurrency, latency or soak behavior;
- live ASK, Root Cause and Smart Diagnostic smoke cases;
- the complete release certification suite.

## Status and Phase 2 closure

Current artifact status:

`OFFLINE_VERIFIED`

This commit closes Roadmap Phase 2 **offline**. Phase 2 becomes formally complete only when the exact pushed commit satisfies `docs/LIVE_GATE_PHASE2_COMMIT5.md`.

No additional infrastructure extraction commit is planned after this one. The next roadmap phase is **Phase 3 — Ingest and Presentation**, and must begin only after this exact commit is live-verified.

## Commit title

`refactor: extract external document transport and Cloud Tasks dispatch`
