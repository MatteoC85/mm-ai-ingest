# Roadmap Phase 3 — Commit 1
## Citation presentation and resource-link boundary

### Purpose

This commit begins Roadmap Phase 3 (Ingest and Presentation) with the lower-risk presentation boundary. It moves citation labels, structured/XLSX snippet cleanup, model-facing source blocks and Bubble resource-link construction out of `main.py` without changing evidence selection, source priority, grounding, retrieval or response reasoning.

The parent is the complete repository snapshot delivered as:

`MachineMind_Modularization_Phase2_Commit5_External_IO`

The exact parent `main.py` SHA-256 is:

`87778093aa95c256a00ed2aa2e58cafdbfdf1b9b98d8b8c49e9e1ccd1fb6d678`

### Extracted module

`machinemind/presentation/citations.py`

The module owns only presentation mechanics for already-selected evidence:

- normalized display text;
- document titles derived from file URLs;
- parsing of the indexed structured-source label/value format;
- source type, title, location and label metadata;
- human-readable Procedure, Step, P&S, Photo and Video metadata snippets;
- compact citation-note lists;
- PDF page links and structured-source links;
- detection and display cleanup of indexed XLSX text;
- public citation sanitization;
- model-facing citation source blocks.

It does **not**:

- retrieve candidates;
- score or rank evidence;
- choose citations;
- modify source priority;
- alter Procedure/Step family selection;
- reason about ASK, Root Cause or Smart Diagnostic;
- access PostgreSQL, OpenAI, Cloud Tasks or FastAPI directly;
- import `main`.

### Compatibility adapters

The historical functions remain present in `main.py` with the same names and signatures:

- `_clean_display_text`;
- `_title_from_file_url`;
- `_parse_structured_source_fields`;
- `_source_display_meta_for_citation`;
- `_structured_source_snippet_for_display`;
- `_format_citation_note_lines`;
- `_build_rg_links`;
- `_looks_like_xlsx_indexed_text`;
- `_clean_xlsx_snippet_for_display`;
- `_sanitize_citations_for_response`;
- `_build_sources_block_from_citations`.

Each adapter injects the current composition-root helpers and limits at call time. This preserves historical late binding for `urlparse`, `unquote`, source-type resolution, structured-source detection, file URL lookup, safe integer conversion, manual-support compaction and display limits.

### Line-count change

- Parent `main.py`: 35,464 lines.
- Candidate `main.py`: 35,120 lines.
- Net reduction from the composition root: 344 lines.
- New presentation module: 617 lines, documented and isolated.

The total repository does not need to become artificially small. The gain in this commit is that presentation behavior now has one explicit module and can be tested independently from retrieval and reasoning.

### Frozen non-changes

The following remain byte-identical to the parent artifact:

- `assistant_core_v2.py`;
- every Phase 1 and Phase 2 extracted production module;
- all runtime configuration modules;
- API contracts and query scope;
- `Dockerfile`;
- `cloudbuild.yaml`;
- `requirements.txt`;
- `mm_promotion_gate.py`.

No prompt, model, threshold, SQL query, cache rule, source-priority rule or public endpoint was intentionally changed.

### Offline verification

The self-contained gate is:

`python tools/run_phase3a_gate.py`

Recorded result:

```text
PASS: syntax
PASS: prior_behavior_contracts
PASS: static_extraction
PASS: presentation_contract
PASS: mutation_sensitivity
PHASE 3A OFFLINE GATE: PASS
```

The new presentation characterization contains 62 captured scenarios and verifies:

- all eleven historical signatures and `main` module identity;
- whitespace, punctuation, Unicode and truncation behavior;
- URL title derivation;
- structured-field parsing and duplicate-field behavior;
- document, Procedure, Step, P&S, Photo and Video labels;
- page and page-range labels;
- structured snippets;
- Italian/English evidence-note headers and de-duplication;
- resource link page fragments and evidence roles;
- XLSX envelope detection and display cleanup;
- citation sanitization, file-map failure behavior and manual-support compaction;
- context-budget boundaries for model-facing source blocks;
- late-bound helper replacement behavior.

All prior Phase 1 and Phase 2 behavioral contracts are re-executed against the candidate.

### Mutation sensitivity

The gate intentionally creates three temporary incorrect variants and requires the characterization to reject all of them:

1. PDF resource links use `#p=` instead of `#page=`;
2. XLSX sheet labels change from `Foglio:` to `Sheet:`;
3. source-block boundary comparison changes from `>` to `>=`.

All three mutations are detected.

### What this proves

The offline evidence supports the statement that the extracted presentation behavior covered by the contract is equivalent to the Phase 2 Commit 5 parent, while the previous API/configuration/database/OpenAI/budget/cache/external-I/O contracts remain unchanged.

### What this does not prove

It does not prove complete product correctness, semantic answer quality, production latency, tenant isolation beyond the previously covered contracts, or every possible customer input. It does not replace exact-commit Cloud Build, Cloud Run startup and live ASK/Root Cause/Smart Diagnostic response checks.

### Next Phase 3 unit

After the exact candidate revision passes `LIVE_GATE_PHASE3_COMMIT1.md`, the next planned unit is:

**Phase 3 — Commit 2: response finalization and HTML/UI rendering**

That unit will remain behavior-preserving and will not yet alter retrieval or reasoning.
