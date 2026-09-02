# Live gate — Roadmap Phase 2, Commit 4

Use the GitHub/VS Code and Google Cloud graphical interfaces.

1. Commit and push from branch `refactor-phase1`.
2. Confirm Cloud Build reports success and identifies:
   - branch `refactor-phase1`;
   - the exact new Git commit.
3. Confirm Cloud Run reports:
   - a healthy newest `mm-ai-ingest-prod` revision;
   - 100% traffic on that revision.
4. Open `/version` and confirm:
   - `ok=true`;
   - `commit_sha` equals the Cloud Build Git commit;
   - Assistant Core V2 and V13 markers/releases are unchanged.
5. Execute known-good cases through the existing application for:
   - ASK;
   - Root Cause;
   - Smart Diagnostic START.
6. Repeat one identical grounded ASK request and confirm that the response remains valid with the same source family/citations.
7. Inspect the newest Cloud Run logs for the request window and confirm there is no new:
   - `V13_CACHE_BOOTSTRAP_RETRY`;
   - `V13_CACHE_LOOKUP_FAIL_OPEN`;
   - `V13_CACHE_VECTOR_FETCH_FAIL_OPEN`;
   - `V13_CACHE_STORE_FAIL_OPEN`;
   - import/startup exception.

Any import failure, changed engine marker, wrong commit SHA, malformed response, missing source evidence or new cache infrastructure error keeps this commit open and blocks the final Infrastructure extraction.
