# Live gate — Roadmap Phase 2, Commit 3

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
5. Use the existing application paths to execute one known-good case for:
   - ASK: normal grounded answer with real citation/link;
   - Root Cause: grounded causes/checks with citations;
   - Smart Diagnostic START: valid question or supported result with valid state.
6. Also confirm no unexpected timeout/provider-error response appears in those ordinary known-good cases.

Any import failure, changed engine marker, wrong commit SHA, provider regression, malformed timeout envelope, missing citations or broken Smart state keeps this commit open and blocks the next extraction.
