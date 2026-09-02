# Phase 1 — API contracts and query scope extraction

Baseline: `prod@89a33a549930003fc0761d7a3f47b70bc22e0c84`.

This commit mechanically extracts the initial Pydantic request contracts and the tenant/machine/document scope resolver into normal importable modules. No prompt, model, SQL, source priority, score, endpoint or response code is modified.

The top-level `main` module still exports every moved name for compatibility.

Commit title:

`refactor: extract API contracts and query scope from production monolith`
