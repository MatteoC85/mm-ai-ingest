# Live gate — Roadmap Phase 2, Commit 2

Use the GitHub/VS Code and Google Cloud graphical interfaces.

1. Commit and push from branch `refactor-phase1`.
2. Confirm Cloud Build shows:
   - successful build;
   - source branch `refactor-phase1`;
   - the exact new commit.
3. Confirm Cloud Run shows:
   - a healthy new `mm-ai-ingest-prod` revision;
   - 100% traffic on that revision.
4. Open `/version` and confirm:
   - `ok=true`;
   - `commit_sha` equals the Cloud Build commit;
   - Assistant Core/V13 release markers are unchanged.
5. Execute one existing known-good live case for each provider-backed path:
   - ASK: answered with expected real citation/link;
   - Root Cause: answered with at least one grounded cause and citation;
   - Smart Diagnostic START: returns a valid first question or grounded final
     result with a valid session state.
6. Any import error, provider error caused by the extraction, missing citations,
   changed engine marker or wrong commit SHA keeps the commit open and blocks the
   next infrastructure extraction.
