"""Cloud Tasks dispatch boundary for asynchronous document indexing.

This is a behavior-preserving extraction from the production ingest route. The
composition root supplies live dependencies explicitly, so the existing environment
precedence, task payload, headers, queue path, fail-open logging and runtime patch
points remain unchanged while Cloud Tasks mechanics move behind an importable boundary.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def enqueue_document_index_task(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    ingest_scope: str,
    ingest_request_key: str,
    ingest_month_key: str,
    ingest_usage_event_id: str,
    ai_internal_secret: str,
    environ: Mapping[str, str],
    tasks_api: Any,
    dumps_fn: Callable[[Any], str],
    log_fn: Callable[..., Any] = print,
) -> None:
    """Enqueue the historical async-index task, failing open exactly as before."""
    try:
        project = (
            environ.get("GOOGLE_CLOUD_PROJECT")
            or environ.get("GCP_PROJECT")
            or "machinemind-ai-2a"
        ).strip()

        location = (
            environ.get("MM_CLOUD_TASKS_LOCATION")
            or "europe-west1"
        ).strip()

        queue = (
            environ.get("MM_CLOUD_TASKS_QUEUE")
            or "mm-ai-index-dev"
        ).strip()

        service_url = (
            environ.get("SERVICE_URL")
            or environ.get("K_SERVICE_URL")
            or ""
        ).strip().rstrip("/")

        if not service_url:
            raise RuntimeError(
                "SERVICE_URL/K_SERVICE_URL missing; Cloud Tasks enqueue skipped"
            )

        client = tasks_api.CloudTasksClient()
        parent = client.queue_path(project, location, queue)

        task_payload = {
            "company_id": company_id,
            "machine_id": machine_id,
            "bubble_document_id": bubble_document_id,
            "trace_id": "ingest_auto",
            "ai_scope": ingest_scope,
            "ingest_request_key": ingest_request_key,
            "ingest_month_key": ingest_month_key,
            "ingest_usage_event_id": ingest_usage_event_id,
            "ingest_metering_enabled": True,
        }

        task = {
            "http_request": {
                "http_method": tasks_api.HttpMethod.POST,
                "url": f"{service_url}/v1/ai/index/document",
                "headers": {
                    "Content-Type": "application/json",
                    "X-AI-Internal-Secret": ai_internal_secret,
                },
                "body": dumps_fn(task_payload).encode(),
            }
        }

        client.create_task(request={"parent": parent, "task": task})
    except Exception as exc:
        log_fn("CLOUD_TASKS_ENQUEUE_SKIPPED", str(exc))
