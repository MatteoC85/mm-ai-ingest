# Live gate — Roadmap Phase 3 Commit 1
## Citation presentation and resource links

Phase 3 Commit 1 remains `OFFLINE_VERIFIED` until the exact Git commit passes this live gate.

## 1. Build identity

Confirm in Cloud Build:

- status is successful;
- branch is `refactor-phase1`;
- commit is the commit titled `refactor: extract citation presentation and resource links`;
- Docker build, image push and Cloud Run deployment all succeeded.

## 2. Revision identity

Confirm in Cloud Run:

- the newest `mm-ai-ingest-prod` revision is healthy;
- it receives 100% of traffic;
- `/version` returns `ok: true`;
- `/version.commit_sha` equals the Cloud Build Git commit;
- Assistant Core V2, V13, ASK and Root Cause markers remain unchanged from the parent revision.

## 3. Document citation and PDF link

Run one known ASK request that is answered from a PDF/manual and verify:

- status remains `answered`;
- at least one citation is present;
- the citation has a non-empty display title/label;
- the corresponding resource link points to the expected document;
- a PDF document link contains `#page=N` with the correct source page;
- no raw internal citation identifier is exposed in the visible answer.

## 4. Structured Procedure/Step response

Run one known structured procedure request and verify:

- Procedure and Step labels remain human-readable;
- Step number/order is unchanged;
- structured links do not receive an artificial PDF `#page=` suffix;
- raw labels such as `SOURCE_TYPE:`, `STEP_NUMBER:` and `SHORT_DESCRIPTION:` are not displayed to the customer;
- the manual remains secondary support when applicable.

## 5. XLSX citation display

Run one known XLSX-backed request and verify:

- the answer and source are still returned;
- the visible snippet uses human-readable labels such as `Documento Excel:`, `Foglio:` and `Intestazioni:` when present;
- raw envelope labels such as `DOCUMENT_FILE_TYPE`, `EXTRACTION_MODE`, `SHEET_INDEX` and `DETECTED_HEADER_ROW` are not exposed;
- the cited workbook/document link remains valid.

## 6. Root Cause and Smart Diagnostic smoke

Run one known Root Cause request and one Smart Diagnostic START request and verify:

- status/result code is unchanged from the known-good parent path;
- citations and links are present when evidence is available;
- no presentation/import error appears;
- Smart Diagnostic still returns a valid session state and question/final envelope.

## 7. Logs

Check the newest revision logs for the tested calls. There must be no new:

- import or startup exception;
- `NameError` involving an extracted presentation helper;
- citation serialization failure;
- resource-link construction failure not already present in the parent;
- repeated `CITATION_FILE_MAP_FAIL` for known-good sources.

## Exit decision

Commit 1 is live-verified only when all seven sections pass on the exact revision. A failed check blocks Phase 3 Commit 2; it does not authorize a new architecture or a semantic behavior change.
