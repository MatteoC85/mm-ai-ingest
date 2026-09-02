# Roadmap Phase 2 — Infrastructure completion record

Phase 2 is implemented through five atomic, behavior-preserving commits:

1. runtime configuration and database infrastructure;
2. OpenAI provider transport;
3. request budgets and execution guards;
4. semantic cache and knowledge versioning;
5. external document transport and Cloud Tasks dispatch.

Across Phase 2, `main.py` moves from `37,441` lines after Phase 1 to `35,464` lines after Commit 5, a net reduction of `1,977` lines from the composition root.

The extracted infrastructure now lives in normal Python modules under:

- `machinemind/config/`;
- `machinemind/infrastructure/database.py`;
- `machinemind/infrastructure/openai_transport.py`;
- `machinemind/infrastructure/request_budget.py`;
- `machinemind/infrastructure/execution.py`;
- `machinemind/infrastructure/semantic_cache.py`;
- `machinemind/infrastructure/document_transport.py`;
- `machinemind/infrastructure/cloud_tasks.py`.

This does **not** mean the whole refactor is complete. The following remain for later roadmap phases:

- ingest parsing/cleaning/chunking internals;
- citations, links and response presentation;
- retrieval and ranking;
- Canonical Evidence Layer;
- ASK and Root Cause consolidation;
- complete Smart Diagnostic consolidation;
- controlled removal of obsolete semantic heuristics;
- Electrical Evidence Provider;
- final shadow/canary rollout.

Phase 2 status in the artifact is `OFFLINE_VERIFIED`. It becomes `COMPLETED` only after the exact Commit 5 revision passes `LIVE_GATE_PHASE2_COMMIT5.md`.
