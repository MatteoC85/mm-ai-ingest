# Roadmap Phase 3 — Commit 2 live fix 1
## Preserve complete source Step sequence and Bubble-visible numbering

### Live failure observed

The exact PROC-001 gate request returned the correct operating content, source
links and citations, but the visible answer failed the Commit 2 exit criterion:

- source Step 1, **Verificare area e ripari**, was moved outside the ordered
  sequence into “Prima di iniziare”;
- the remaining Step titles were all displayed with the marker `1.` rather than
  the source sequence `1..8`.

The response layout confirmed that the newly extracted renderer was active, so
this was not accepted as a Phase 3 Commit 2 pass.

### Root cause

Two presentation assumptions were invalid in the real Bubble host:

1. the presentation model heuristically promoted a safety-oriented first Step to
   a prerequisite card, even though it remained an authored Step of PROC-001;
2. the HTML relied on ordered-list/value behavior that the live host did not
   reproduce reliably. The original offline test inspected raw attributes but
   did not assert the visible text after a conservative host strips attributes.

### Correction

The correction is confined to
`machinemind/presentation/responses.py`:

- every structured source Step remains in the Procedure `steps` collection;
- `display_number` is taken from the source `step_number`;
- structured, generic and lossless number renderers emit the number as escaped
  visible text in a `role=listitem` row;
- no source Step is silently moved to `before` or `final_checks`;
- canonical answer text, citations, resource links and source ordering are not
  changed.

### New deterministic live-shaped regression

The contract now constructs a full eight-Step PROC-001 model with the same Step
titles observed live. It requires:

- `before_titles == []` and `final_titles == []` for source-authored Steps;
- source numbers and display numbers exactly `[1,2,3,4,5,6,7,8]`;
- visible HTML number metadata exactly `[1,2,3,4,5,6,7,8]`;
- the literal visible sequence `1. Verificare area e ripari` through
  `8. Avviare in automatico` to survive removal of every opening-tag attribute.

The contract also keeps the existing generic, lossless, escaping, finalization,
link/citation de-duplication, Root Cause parity and all prior Phase 1/2/3A gates.

### Frozen non-changes

This fix does not modify `main.py`, `assistant_core_v2.py`, retrieval, evidence
admission, source priorities, prompts, models, semantic cache policy, Root Cause,
Smart Diagnostic, citation/link construction or ingest.

The pre-existing ASK blocker
`Qual è il peso della macchina BBX-300/40T? -> no_sources` remains open and is
not claimed as fixed here.

### Exit

Phase 3 Commit 2 remains open until the exact deployed fix revision passes every
section of `LIVE_GATE_PHASE3_COMMIT2.md`, beginning with the PROC-001 visible
sequence `1..8`.
