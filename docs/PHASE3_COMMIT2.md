# Roadmap Phase 3 — Commit 2
## Response finalization and safe HTML/UI rendering

### Purpose

This commit completes Roadmap unit **3B** by moving response finalization and
safe HTML rendering out of `main.py` into a dedicated presentation module.

The parent is the exact Phase 3 Commit 1 state that passed the live gate:

- branch: `refactor-phase1`;
- Git commit: `4a832e2`;
- Cloud Run revision: `mm-ai-ingest-prod-00055-h7t`;
- parent `main.py` SHA-256: `d77dc2938d8cfbb6557b24230bb6936b545e39fdf945e94d76224a3e200ef05a`;
- parent `main.py` lines: 35,120.

### Extracted module

`machinemind/presentation/responses.py`

The module now owns:

- the deterministic structured Procedure UI model;
- Procedure plain-text and HTML rendering;
- ASK safe HTML rendering;
- Root Cause canonical text and HTML rendering;
- restrained Markdown-like inline formatting;
- Markdown-table normalization for display;
- safe HTML escaping;
- ordered and unordered list rendering;
- canonical visible-text/token coverage checks;
- lossless HTML fallback;
- final link and citation de-duplication;
- the final public answer envelope (`answer`, `answer_html`, `answer_format`,
  `answer_render_version` and presentation metadata).

The module receives every product-specific callback and limit through
`ResponsePresentationRuntime`. It does not import `main`, access PostgreSQL,
call OpenAI, inspect tenant scope, retrieve evidence or choose sources.

### Compatibility adapters

The following 24 historical functions remain in `main.py` with their original
names, signatures and `__module__ == "main"`:

- `_build_structured_procedure_ui_model`;
- `_procedure_ui_model_to_text`;
- `_assistant_ui_escape`;
- `_procedure_ui_model_to_html`;
- `_assistant_ui_inline_markup`;
- `_assistant_ui_section_kind`;
- `_assistant_ui_extract_labeled_line`;
- `_assistant_ui_split_inline_numbered`;
- `_assistant_ui_render_numbered_cards`;
- `_assistant_ui_sentence_has_any`;
- `_assistant_ui_promote_unlabelled_sections`;
- `_assistant_ui_normalize_markdown_tables`;
- `_assistant_ui_root_cause_text`;
- `_assistant_ui_generic_html`;
- `_assistant_ui_root_cause_html`;
- `_assistant_ui_normalize_url_for_key`;
- `_assistant_ui_dedupe_links`;
- `_assistant_ui_dedupe_citations`;
- `_assistant_ui_visible_text_from_html`;
- `_assistant_ui_canonical_tokens`;
- `_assistant_ui_token_coverage`;
- `_assistant_ui_lossless_html`;
- `_assistant_ui_finalize_response`;
- `_format_structured_procedure_answer_for_ui`.

A late-bound runtime factory keeps the current composition-root functions and
configuration values replaceable at call time, preserving the existing
characterization and monkeypatch surface.

### Intentional UI-only correction

The Phase 3 Commit 1 live gate exposed a real renderer defect: an ordered
Procedure answer could display every Step as `1.` because labelled detail lines
split the answer into multiple HTML `<ol>` fragments.

This commit corrects only that display behavior:

- explicit numbers are preserved with safe `<li value="N">` attributes;
- the same guard exists in the generic and lossless renderers;
- when a structured Procedure model is already attached to the response,
  finalization now uses the dedicated Procedure renderer instead of flattening
  it back through the generic renderer;
- the canonical plain-text answer, source order, citations, Bubble links and
  source snippets remain unchanged.

The correction is pinned by deterministic tests using `1, 2, 3`; removing the
explicit values or disabling the structured renderer makes the gate fail.

### Frozen non-changes

This commit does **not** change:

- retrieval, query routing, ranking or evidence admission;
- source priorities or Procedure-family selection;
- prompts, models, model effort, budgets or timeouts;
- ASK semantic synthesis;
- Root Cause reasoning or Smart Diagnostic state/logic;
- citation sanitization, source labels, resource-link construction or XLSX
  display cleanup from Phase 3 Commit 1;
- ingest PDF/XLSX parsing, cleaning, chunking, persistence or metering;
- semantic cache policy;
- API routes or request/response contracts;
- `assistant_core_v2.py`;
- every Phase 1/2 production module;
- `Dockerfile`, `cloudbuild.yaml`, `requirements.txt` and
  `mm_promotion_gate.py`.

### Known product blocker intentionally not changed

The live question:

`Qual è il peso della macchina BBX-300/40T?`

still has a documented pre-existing `no_sources` failure even though the
indexed manual contains `Peso macchina 7500 Kg`. That defect was reproduced on
both Phase 2 Commit 5 and Phase 3 Commit 1. It belongs to the later
Retrieval/Evidence correction and is **not** hidden or declared fixed by this
presentation commit.

### Line-count change

- Parent `main.py`: 35,120 lines.
- Candidate `main.py`: 33,977 lines.
- Net reduction from the composition root: **1,143 lines**.
- New `responses.py`: **936 lines**.

### Offline verification

Run:

`python tools/run_phase3b_gate.py`

Recorded result:

```text
PASS: syntax
PASS: prior_behavior_contracts
PASS: static_extraction
PASS: response_render_contract
PASS: mutation_sensitivity
PHASE 3B OFFLINE GATE: PASS
```

The gate re-runs all Phase 1/2 behavioral contracts and the complete Phase 3A
citation/link contract. The new response contract captures historical
signatures, safe escaping, label parsing, Markdown normalization, ASK/Root
Cause rendering, canonicality, lossless fallback, link/citation de-duplication,
structured Procedure finalization, late binding and the ordered-number fix.

Parent and candidate response characterizations differ only on the reviewed
HTML/Procedure paths. Root Cause finalization, link de-duplication, citation
de-duplication and every historical signature remain identical.

### Mutation sensitivity

The gate requires all three temporary defects to be detected:

1. remove explicit ordered-list `value` attributes;
2. disable the structured Procedure renderer;
3. disable HTML escaping.

All three mutations are rejected.

### What this proves

It proves that the response/UI mechanics covered by the contract have been
isolated from the production composition root, prior behavioral contracts still
pass, and the observed repeated-`1.` display bug is corrected without changing
the canonical answer or evidence payload.

### What this does not prove

It does not prove complete semantic correctness, production latency, every
browser/CSS combination, every customer document, or resolution of the known
numeric `no_sources` blocker. Exact-commit Cloud Build, Cloud Run identity and
the live checks in `LIVE_GATE_PHASE3_COMMIT2.md` remain mandatory.

### Next Phase 3 unit

After this exact candidate passes the live gate, the next planned unit is:

**Phase 3 — Commit 3 / 3C: shared ingest text and PDF extraction primitives.**

## Live validation correction — 2026-09-02

The initial candidate did **not** pass its exact live PROC-001 gate. Bubble
showed Step 1 outside the sequence and repeated `1.` for the remaining items.
Therefore the original statement that raw `<li value>` attributes were
sufficient is superseded by `PHASE3_COMMIT2_LIVE_FIX1.md`.

The reviewed correction keeps every source Step in the sequence and renders the
number as visible text. Phase 3 Commit 3 remains blocked until the corrective
revision passes the live gate.

