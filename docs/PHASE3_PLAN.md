# Roadmap Phase 3 — Ingest and Presentation

Phase 3 continues the behavior-preserving extraction. It does not redesign the AI core and does not change source priorities.

Planned atomic units:

1. **3A — Citation presentation and resource links**: labels, snippets, source blocks, citation sanitization and links.
2. **3B — Response finalization and UI rendering**: ASK/Root Cause text-to-HTML, structured UI model, link/citation de-duplication and final public envelope.
3. **3C — Shared ingest text and PDF extraction primitives**: Unicode normalization, layout extraction and page text boundaries.
4. **3D — PDF cleaning and chunking**: noise removal, header/footer handling, reflow, section-aware chunks and overlap.
5. **3E — XLSX extraction and ingest dispatch**: workbook validation, sheet/table text, limits and file-type routing.
6. **3F — Ingest persistence/metering closure if still required after 3C–3E**: only behavior that remains in the composition root after the parsing boundaries are extracted.

A unit may be split only to reduce risk; Phase 3 is complete when ingest and presentation no longer live materially inside `main.py`, all existing contracts remain valid and the exact live revision passes its final gate.


## Current status

- **3A — Citation presentation and resource links:** live-verified on commit
  `4a832e2`, revision `mm-ai-ingest-prod-00055-h7t`.
- **3B — Response finalization and UI rendering:** candidate prepared and
  offline-verified; live gate pending.
- The pre-existing exact numeric ASK `Peso macchina → 7500 Kg` remains an open
  Retrieval/Evidence blocker and is not part of the 3B presentation change.
