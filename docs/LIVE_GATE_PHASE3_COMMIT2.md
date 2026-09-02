# Live gate — Roadmap Phase 3 Commit 2
## Response finalization and safe HTML/UI rendering

Phase 3 Commit 2 remains `OFFLINE_VERIFIED` until the exact Git commit passes
this gate.

## 1. Build and revision identity

Confirm:

- Cloud Build succeeds from branch `refactor-phase1`;
- the commit title is `refactor: extract response finalization and fix ordered-step rendering`;
- Build, Push and Deploy are green;
- the newest `mm-ai-ingest-prod` revision is healthy at 100% traffic;
- `/version` returns `ok: true`;
- `/version.commit_sha` equals the Git/Cloud Build commit;
- Assistant Core V2, V13, ASK, Root Cause and Smart Diagnostic markers are
  unchanged from `mm-ai-ingest-prod-00055-h7t`.

## 2. Exact Procedure rendering regression

Use the same structured request already validated on Phase 3 Commit 1:

`Descrivimi in ordine tutti gli step della procedura “Avviamento sicuro e messa in ciclo automatico”, indicando per ogni step l’azione operativa principale.`

Require:

- the canonical response still contains all 8 PROC-001 Steps in the same order;
- the visible numbering is `1, 2, 3, 4, 5, 6, 7, 8`, never eight repeated `1.`;
- the Procedure is the primary source and all Step links remain present;
- manual sources remain secondary;
- no raw Bubble/citation identifiers appear;
- the Bubble LINK and FONTI sections are not duplicated inside the answer body.

## 3. PDF/HMI ASK

Repeat one known-good HMI/PDF factual request, for example:

`Secondo il manuale HMI, che differenza c'è tra un allarme “Signal” e un “Immediate stop”?`

Require the same correct answer, HMI document, page 23, source snippet and link
already observed on the parent revision.

## 4. XLSX presentation

Use a known indexed XLSX row or temporarily re-use the controlled QA workbook.

Require:

- the exact requested value is returned;
- `Foglio`, row and human-readable headings remain visible;
- raw ingest-envelope fields remain hidden;
- the workbook link remains valid.

## 5. Root Cause

Repeat the known-good post-emergency Automatic-mode Root Cause scenario.

Require:

- the same plausible cause families;
- pertinent source snippets and links;
- no change to Root Cause canonical text or cause/check payload;
- no answer-body duplication of LINK/FONTI.

## 6. Smart Diagnostic START

Repeat the known-good START symptom:

`Dopo un arresto di emergenza ho chiuso tutti i ripari, ma la macchina non si abilita in Automatico.`

Require a valid session, question 1/6, plausible hypotheses and sources, with no
state or rendering error.

## 7. HTML safety/fallback smoke

Use an ordinary response containing bold/list formatting and confirm:

- text is rendered, not lost;
- no raw HTML supplied by model text is executed or interpreted;
- `answer_format` remains `html` for a normal successful response;
- a presentation failure falls back to canonical text rather than publishing
  truncated/lossy HTML.

## 8. Logs

For the tested calls, there must be no new:

- import/startup exception;
- `NameError` involving `_presentation_responses` or
  `ResponsePresentationRuntime`;
- HTML serialization error;
- answer-token coverage failure;
- citation/link serialization regression.

## Tracked non-gate blocker

Do not use the existing question about the BBX-300/40T machine weight as proof
that this commit failed or succeeded: its `no_sources` behavior predates this
commit and remains an explicitly open Retrieval/Evidence blocker. It must not
be forgotten, but this UI commit does not claim to solve it.

## Exit decision

Commit 2 is complete only when the exact revision passes all sections above.
A failure blocks Phase 3 Commit 3 and must be diagnosed against
`mm-ai-ingest-prod-00055-h7t` before any further extraction.
