# Phase 6 isolated native validation

This branch is test-only. It does not change refactor-phase1, deploy Cloud Run or Cloudflare, access Bubble, use production secrets, or call an AI provider. PostgreSQL and pgvector run locally on the ephemeral GitHub Actions runner with synthetic identities. The compressed harness reconstructs a candidate from a SHA-pinned public baseline and verifies all source hashes before tests. No private customer fixture is included.
