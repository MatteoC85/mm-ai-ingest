# Live gate — Roadmap Phase 2, Commit 5 and Phase 2 closure

Use the existing GitHub/VS Code and Google Cloud graphical workflow.

## 1. Exact source and build

1. Commit and push from branch `refactor-phase1`.
2. Confirm Cloud Build reports success and identifies:
   - branch `refactor-phase1`;
   - the exact new Git commit;
   - successful Build, Push and Deploy steps.

## 2. Exact live revision

Confirm Cloud Run reports:

- a healthy newest `mm-ai-ingest-prod` revision;
- 100% traffic on that revision;
- no startup/import error.

Open `/version` and confirm:

- `ok=true`;
- `commit_sha` equals the Cloud Build Git commit;
- Assistant Core V2 and V13 markers/releases are unchanged;
- ASK and Root Cause remain enabled.

## 3. Application smoke paths

Execute known-good cases through the existing application for:

- ASK;
- Root Cause;
- Smart Diagnostic START.

Each must preserve its expected response shape, language, evidence and link behavior.

## 4. Real ingest and asynchronous indexing

Using the existing application, ingest one known-good small PDF or XLSX that has previously indexed successfully.

Confirm:

- the ingest route accepts the document;
- the normal queued/pending response is returned;
- the document reaches its expected indexed/ready state;
- the asynchronous `/v1/ai/index/document` request is executed;
- the resulting document remains queryable through the existing product.

Inspect the newest Cloud Run logs for the same request window and confirm there is no new:

- `CLOUD_TASKS_ENQUEUE_SKIPPED`;
- external-file fetch failure;
- base64/file-load exception;
- import/startup exception;
- cache, database or OpenAI infrastructure regression.

A deliberately unavailable external URL may still produce the historical HTTP 502 `Fetch failed`; that is not a regression. The normal known-good path must succeed.

## 5. Phase 2 decision

Any wrong commit SHA, failed build, unhealthy revision, malformed response, failed known-good ingest, missing async index task or new infrastructure error keeps Commit 5 and Phase 2 open.

When all checks above pass, record:

> **FASE 2 DELLA ROADMAP — INFRASTRUCTURE: COMPLETATA**

The next work item is Phase 3. No Phase 2 Commit 6 is planned.
