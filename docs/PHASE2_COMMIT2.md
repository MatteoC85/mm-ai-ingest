# Roadmap Phase 2 — Commit 2: OpenAI provider transport

## Parent state

- Git branch: `refactor-phase1`
- Parent artifact: Roadmap Phase 2 — Commit 1 (runtime configuration and database boundary)
- Parent `main.py` SHA-256: `7c3e6f0cae8d173b1afdcfc078cb70a4ec6ec3acbb6cd2dbec3f416229fb3875`
- Parent `main.py` lines: 36,870

## Purpose

This is the second atomic commit of Roadmap Phase 2 (Infrastructure). It extracts
provider-facing OpenAI HTTP mechanics from the production monolith while keeping
all historical functions and signatures available in `main` as compatibility
adapters.

This commit is a mechanical boundary extraction. It does **not** change prompts,
models, routing, retrieval, evidence ranking, source priorities, answer contracts,
ASK, Root Cause or Smart Diagnostic policy.

## Extracted module

`machinemind/infrastructure/openai_transport.py`

The module now owns:

- embedding request construction and response reconstruction;
- request-local embedding cache behavior;
- ingest embedding-usage metering callbacks;
- legacy Chat Completions text calls;
- legacy Chat Completions JSON-schema calls;
- ordered legacy model fallback;
- Responses API structured-output request construction;
- Responses output-text/refusal parsing;
- per-company safety-identifier generation;
- mixed Responses/legacy model fallback orchestration;
- provider error and malformed-response handling.

## Compatibility adapters retained in `main.py`

- `_openai_embed_texts`
- `_openai_chat`
- `_openai_chat_json`
- `_normalize_model_candidates`
- `_openai_chat_json_models`
- `_v13_safety_identifier`
- `_v13_response_text`
- `_v13_responses_json`
- `_v13_json_models`

The adapters pass runtime values and callbacks at call time. This intentionally
preserves the historical late-bound test/monkeypatch surface, current request
budget, ingest meter, configured URLs/models/API key and provider HTTP client.

## Explicit non-changes

This commit does not modify:

- API endpoints or request/response models;
- `assistant_core_v2.py`;
- runtime configuration defaults or environment-variable behavior;
- database connection policy;
- prompts or model selection policy;
- request-budget calculations or limits;
- retrieval, ranking, source selection or source priorities;
- evidence admission, validation or repair;
- ASK, Root Cause or Smart Diagnostic behavior;
- PDF/XLSX ingest logic;
- semantic cache behavior;
- citation/link rendering;
- Docker, Cloud Build, requirements or promotion-gate files.

## Offline verification performed

- Python syntax compilation for every Python file: PASS.
- All non-structural Phase 2 Commit 1 gates re-run: PASS.
- Phase 1 API/request/scope/marker contract: exact parity.
- FastAPI OpenAPI JSON: exact parity.
- 315 configuration exports and four environment scenarios: exact parity.
- Database connector and SQL utility contract: exact parity.
- Top-level `main.py` definition inventory: 673/673 preserved.
- Only the nine declared OpenAI compatibility functions changed in `main.py`.
- `assistant_core_v2.py` SHA-256 unchanged.
- OpenAI differential contract: exact byte-for-byte parity across 49 scenarios.
- Scenarios cover request payloads, headers, timeouts, default/custom models,
  JSON parsing, content-list parsing, provider HTTP errors, missing keys,
  embedding de-duplication/cache/metering, missing vectors, Responses status
  handling, usage accounting, refusals and model fallback/retry behavior.
- Negative control: an intentional provider-timeout mutation was detected by the
  gate; the reviewed source was then restored and the complete gate re-passed.

## Status

`OFFLINE VERIFIED`

The commit is complete only after build/deploy from the exact Git commit and live
verification of the new revision plus at least one real provider-backed ASK,
Root Cause and Smart Diagnostic start call.

## Commit title

`refactor: extract OpenAI provider transport from production monolith`
