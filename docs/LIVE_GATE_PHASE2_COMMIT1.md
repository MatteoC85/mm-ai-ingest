# Live gate — Roadmap Phase 2, commit 1

This commit is not closed merely because Cloud Build succeeds.

Required evidence:

1. Cloud Build status: `Operazione riuscita`.
2. Build source branch: `refactor-phase1`.
3. Build source commit: the new Phase 2 commit.
4. New `mm-ai-ingest-prod` revision is healthy and serves 100% traffic.
5. `/version` reports:
   - the new revision;
   - the exact new Git commit SHA;
   - `assistant_core_v2_enabled: true`;
   - unchanged Assistant Core marker and release;
   - unchanged V13 marker/release and enabled state.
6. Minimal DB-backed live smoke:
   - one known ASK request returns normally;
   - one known Root Cause request returns normally;
   - Smart Diagnostic START returns normally when the feature is enabled.

A failure in any item keeps this commit in `LIVE VERIFICATION REQUIRED`.
