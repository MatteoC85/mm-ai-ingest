#!/usr/bin/env python3
"""Resumable client for MachineMind page-atomic electrical graph extraction.

The server remains authoritative and atomic per page. This runner only
orchestrates pages, retries transient transport failures, persists checkpoints,
and groups review pages by a stable issue signature. It never sends force=true.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, TextIO

RUNNER_VERSION = "mm-graph-batch-runner-v2"
EXPECTED_PIPELINE_MARKER = (
    "phase2-graph-v2-evidence-owned-batch-source-snapshot"
)
EXPECTED_MATERIALIZER_VERSION = "mm-electrical-graph-materializer-v2"
DEFAULT_BASE_URL = (
    "https://mm-ai-ingest-fixed-443517556116.europe-west1.run.app"
)
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
TRANSIENT_TEXT_MARKERS = (
    "timed out",
    "timeout",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "rate limit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gcloud_identity_token() -> str:
    completed = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        check=True,
        text=True,
        capture_output=True,
    )
    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("gcloud returned an empty identity token")
    return token


def json_request(
    *,
    url: str,
    payload: dict[str, Any],
    identity_token: str,
    internal_secret: str,
    timeout_seconds: int,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {identity_token}",
            "X-AI-Internal-Secret": internal_secret,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                parsed = {
                    "detail": "non-JSON response",
                    "raw_body": body[:4000],
                }
            return int(response.status), parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            parsed = {"detail": body[:4000]}
        return int(exc.code), parsed


@contextmanager
def exclusive_state_lock(state_path: Path) -> Iterator[TextIO]:
    """Prevent two runners from operating on the same checkpoint concurrently."""
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another runner holds the checkpoint lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "pid": os.getpid(),
            "acquired_at": utc_now(),
            "state_path": str(state_path),
        }, ensure_ascii=False) + "\n")
        handle.flush()
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validate_state_scope(
    *,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Fail closed if a checkpoint belongs to another service or data scope."""
    expected_scope = {
        "company_id": args.company_id,
        "machine_id": args.machine_id,
        "bubble_document_id": args.bubble_document_id,
        "version_id": args.version_id,
    }
    actual_scope = state.get("scope") or {}
    if actual_scope and actual_scope != expected_scope:
        raise ValueError(
            "checkpoint scope does not match this run: "
            f"expected={expected_scope!r}, actual={actual_scope!r}"
        )
    actual_base = str(state.get("base_url") or "").rstrip("/")
    expected_base = args.base_url.rstrip("/")
    if actual_base and actual_base != expected_base:
        raise ValueError(
            "checkpoint base_url does not match this run: "
            f"expected={expected_base!r}, actual={actual_base!r}"
        )
    state["runner_version"] = RUNNER_VERSION
    state["base_url"] = expected_base
    state["scope"] = expected_scope


def is_transient(status_code: int, response: dict[str, Any]) -> bool:
    if status_code in TRANSIENT_HTTP_CODES:
        return True
    text = json.dumps(response, ensure_ascii=False).casefold()
    return any(marker in text for marker in TRANSIENT_TEXT_MARKERS)


def parse_pages(value: str) -> list[int]:
    pages: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        page = int(token)
        if page <= 0:
            raise ValueError(f"invalid page number: {page}")
        if page not in pages:
            pages.append(page)
    return pages


def manifest_pages(path: Path) -> list[int]:
    payload = load_json(path, None)
    if isinstance(payload, dict):
        payload = payload.get("pages") or []
    if not isinstance(payload, list):
        raise ValueError("manifest must be a list or an object with pages")
    pages: list[int] = []
    for item in payload:
        if isinstance(item, dict):
            value = item.get("pdf_page_number")
        else:
            value = item
        page = int(value)
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def review_signature(response: dict[str, Any]) -> str:
    existing = str(response.get("review_signature") or "").strip()
    if existing:
        return existing
    counts = response.get("blocking_issue_type_counts") or {}
    raw = json.dumps(counts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def compact_result(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: response.get(key)
        for key in (
            "ok",
            "pdf_page_number",
            "sheet_code",
            "sheet_title",
            "graph_pipeline_marker",
            "graph_materializer_version",
            "page_passed",
            "raw_extracted_entity_count",
            "raw_extracted_edge_count",
            "extracted_entity_count",
            "extracted_edge_count",
            "published_page_entities",
            "published_page_edges",
            "blocking_issue_count_this_page",
            "warning_issue_count_this_page",
            "severity_counts",
            "blocking_issue_type_counts",
            "blocking_issue_summary",
            "review_signature",
            "graph_status",
            "graph_entity_count",
            "graph_edge_count",
            "passed_pages",
            "total_pages",
            "calls",
            "reused_calls",
            "new_input_tokens",
            "new_output_tokens",
        )
    }


def state_template(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "base_url": args.base_url.rstrip("/"),
        "scope": {
            "company_id": args.company_id,
            "machine_id": args.machine_id,
            "bubble_document_id": args.bubble_document_id,
            "version_id": args.version_id,
        },
        "pages": {},
        "runs": [],
    }


def page_status_from_response(
    status_code: int, response: dict[str, Any]
) -> str:
    if status_code == 200 and response.get("ok") is True:
        return "passed" if response.get("page_passed") is True else "review_required"
    if is_transient(status_code, response):
        return "transient_error"
    return "fatal_error"


def choose_pages(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    state: dict[str, Any],
) -> list[int]:
    if args.pages:
        requested = parse_pages(args.pages)
    elif args.manifest:
        requested = manifest_pages(Path(args.manifest))
    else:
        requested = [
            int(item["pdf_page_number"])
            for item in plan.get("pages") or []
            if (
                args.mode == "all"
                or item.get("status") == args.mode
                or (
                    args.mode == "pending"
                    and item.get("status") in {"not_started", "review_required"}
                )
            )
        ]

    plan_by_page = {
        int(item["pdf_page_number"]): item
        for item in plan.get("pages") or []
    }
    current_marker = str(plan.get("graph_pipeline_marker") or "")
    current_materializer = str(
        plan.get("graph_materializer_version") or ""
    )
    selected: list[int] = []
    for page in requested:
        plan_item = plan_by_page.get(page)
        if plan_item is None:
            raise ValueError(f"page {page} is not a schematic page in the plan")
        if plan_item.get("status") == "passed" and not args.reprocess_passed:
            continue
        state_item = (state.get("pages") or {}).get(str(page)) or {}
        if state_item.get("status") == "passed" and not args.reprocess_passed:
            continue
        persisted_same_generation_review = bool(
            plan_item.get("status") == "review_required"
            and str(plan_item.get("result_pipeline_marker") or "")
                == current_marker
            and str(plan_item.get("result_materializer_version") or "")
                == current_materializer
        )
        checkpoint_same_generation_review = bool(
            state_item.get("status") == "review_required"
            and str(state_item.get("pipeline_marker") or "")
                == current_marker
            and str(state_item.get("materializer_version") or "")
                == current_materializer
        )
        same_generation_review = bool(
            persisted_same_generation_review
            or checkpoint_same_generation_review
        )
        if (
            same_generation_review
            and not args.retry_review_same_generation
        ):
            continue
        selected.append(page)
    if args.max_pages:
        selected = selected[: args.max_pages]
    return selected


def write_summary(
    *,
    output_dir: Path,
    state: dict[str, Any],
    selected_pages: Iterable[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    review_groups: dict[str, dict[str, Any]] = {}
    selected = list(selected_pages)
    for page in selected:
        item = (state.get("pages") or {}).get(str(page)) or {}
        result = item.get("result") or {}
        row = {
            "pdf_page_number": page,
            "sheet_code": result.get("sheet_code"),
            "sheet_title": result.get("sheet_title"),
            "status": item.get("status") or "not_run",
            "attempts": item.get("attempts") or 0,
            "page_passed": result.get("page_passed"),
            "published_page_entities": result.get("published_page_entities"),
            "published_page_edges": result.get("published_page_edges"),
            "blocking_issue_count": result.get(
                "blocking_issue_count_this_page"
            ),
            "review_signature": item.get("review_signature") or "",
            "response_file": item.get("response_file") or "",
        }
        rows.append(row)
        if item.get("status") == "review_required":
            signature = item.get("review_signature") or "unknown"
            group = review_groups.setdefault(signature, {
                "review_signature": signature,
                "pages": [],
                "blocking_issue_type_counts": result.get(
                    "blocking_issue_type_counts"
                ) or {},
                "blocking_issue_summary": result.get(
                    "blocking_issue_summary"
                ) or [],
            })
            group["pages"].append(page)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    usage = {
        "calls": sum(
            int(((state.get("pages") or {}).get(str(page)) or {})
                .get("result", {}).get("calls") or 0)
            for page in selected
        ),
        "reused_calls": sum(
            int(((state.get("pages") or {}).get(str(page)) or {})
                .get("result", {}).get("reused_calls") or 0)
            for page in selected
        ),
        "new_input_tokens": sum(
            int(((state.get("pages") or {}).get(str(page)) or {})
                .get("result", {}).get("new_input_tokens") or 0)
            for page in selected
        ),
        "new_output_tokens": sum(
            int(((state.get("pages") or {}).get(str(page)) or {})
                .get("result", {}).get("new_output_tokens") or 0)
            for page in selected
        ),
    }
    summary = {
        "runner_version": RUNNER_VERSION,
        "generated_at": utc_now(),
        "selected_page_count": len(selected),
        "counts": dict(sorted(counts.items())),
        "usage": usage,
        "review_groups": list(review_groups.values()),
        "rows": rows,
    }
    atomic_json_write(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "pdf_page_number", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--expected-pipeline-marker",
        default=EXPECTED_PIPELINE_MARKER,
    )
    parser.add_argument(
        "--expected-materializer-version",
        default=EXPECTED_MATERIALIZER_VERSION,
    )
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--bubble-document-id", required=True)
    parser.add_argument("--version-id", type=int, required=True)
    parser.add_argument(
        "--secret",
        default=os.environ.get("AI_INTERNAL_SECRET", ""),
    )
    parser.add_argument(
        "--mode",
        choices=("all", "pending", "not_started", "review_required"),
        default="pending",
    )
    parser.add_argument("--pages", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--state", default="mm_graph_batch_state.json")
    parser.add_argument("--output-dir", default="mm_graph_batch_output")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=650)
    parser.add_argument("--retry-base-seconds", type=float, default=8.0)
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--reprocess-passed", action="store_true")
    parser.add_argument(
        "--retry-review-same-generation",
        action="store_true",
        help=(
            "retry review pages already attempted with the same pipeline "
            "marker/materializer; default is to wait for a new generation"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exit-zero-on-review", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    assert parse_pages("24,43,24") == [24, 43]
    assert page_status_from_response(200, {"ok": True, "page_passed": True}) == "passed"
    assert page_status_from_response(200, {"ok": True, "page_passed": False}) == "review_required"
    assert page_status_from_response(500, {"detail": "Read timed out"}) == "transient_error"
    assert review_signature({"blocking_issue_type_counts": {"x": 2}})
    args = argparse.Namespace(
        base_url="https://example.invalid",
        company_id="c",
        machine_id="m",
        bubble_document_id="d",
        version_id=2,
        expected_pipeline_marker="marker-v2",
        expected_materializer_version="materializer-v2",
        pages="",
        manifest="",
        mode="pending",
        reprocess_passed=False,
        retry_review_same_generation=False,
        max_pages=0,
    )
    state = state_template(args)
    plan = {
        "graph_pipeline_marker": "marker-v2",
        "graph_materializer_version": "materializer-v2",
        "pages": [
            {"pdf_page_number": 24, "status": "passed"},
            {
                "pdf_page_number": 43,
                "status": "review_required",
                "result_pipeline_marker": "marker-v2",
                "result_materializer_version": "materializer-v2",
            },
            {"pdf_page_number": 44, "status": "not_started"},
        ],
    }
    state["pages"]["43"] = {
        "status": "review_required",
        "pipeline_marker": "marker-v2",
        "materializer_version": "materializer-v2",
    }
    assert choose_pages(args=args, plan=plan, state=state) == [44]
    args.retry_review_same_generation = True
    assert choose_pages(args=args, plan=plan, state=state) == [43, 44]
    print("MM GRAPH BATCH RUNNER SELF-TEST: PASS")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    parser = build_parser()
    args = parser.parse_args()
    if not args.secret:
        parser.error("--secret or AI_INTERNAL_SECRET is required")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    base_url = args.base_url.rstrip("/")
    state_path = Path(args.state)
    lock_context = exclusive_state_lock(state_path)
    lock_handle = lock_context.__enter__()
    atexit.register(lock_context.__exit__, None, None, None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(state_path, state_template(args))
    validate_state_scope(state=state, args=args)
    state.setdefault("pages", {})
    token = gcloud_identity_token()
    plan_status, plan = json_request(
        url=base_url + "/v1/ai/electrical/graph-plan",
        payload={
            "company_id": args.company_id,
            "machine_id": args.machine_id,
            "bubble_document_id": args.bubble_document_id,
            "version_id": args.version_id,
            "include_passed": True,
        },
        identity_token=token,
        internal_secret=args.secret,
        timeout_seconds=60,
    )
    if plan_status != 200 or plan.get("ok") is not True:
        print(json.dumps(plan, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    if int(plan.get("electrical_version_id") or 0) != args.version_id:
        print(json.dumps({
            "error": "graph plan version mismatch",
            "expected_version_id": args.version_id,
            "actual_version_id": plan.get("electrical_version_id"),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    actual_marker = str(plan.get("graph_pipeline_marker") or "")
    actual_materializer = str(
        plan.get("graph_materializer_version") or ""
    )
    if (
        actual_marker != args.expected_pipeline_marker
        or actual_materializer != args.expected_materializer_version
    ):
        print(json.dumps({
            "error": "graph release generation mismatch",
            "expected_pipeline_marker": args.expected_pipeline_marker,
            "actual_pipeline_marker": actual_marker,
            "expected_materializer_version": (
                args.expected_materializer_version
            ),
            "actual_materializer_version": actual_materializer,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    selected = choose_pages(args=args, plan=plan, state=state)
    print(
        f"PLAN version={plan.get('electrical_version_id')} "
        f"selected={len(selected)} graph_status={plan.get('graph_status')}"
    )
    print("PAGES " + (",".join(map(str, selected)) if selected else "none"))
    if args.dry_run or not selected:
        write_summary(output_dir=output_dir, state=state, selected_pages=selected)
        return 0

    run = {
        "started_at": utc_now(),
        "selected_pages": selected,
        "mode": args.mode,
    }
    state.setdefault("runs", []).append(run)
    atomic_json_write(state_path, state)

    for position, page in enumerate(selected, start=1):
        page_key = str(page)
        page_state = state["pages"].setdefault(page_key, {
            "attempts": 0,
            "status": "pending",
        })
        final_response: dict[str, Any] = {}
        final_http = 0
        for attempt in range(1, args.max_attempts + 1):
            page_state["attempts"] = int(page_state.get("attempts") or 0) + 1
            page_state["last_attempt_at"] = utc_now()
            atomic_json_write(state_path, state)
            # Refresh the identity token for every page. Long graph batches can
            # outlive a single token and token refresh is cheap compared to AI.
            token = gcloud_identity_token()
            print(
                f"[{position}/{len(selected)}] page={page} attempt={attempt} ...",
                flush=True,
            )
            try:
                status_code, response = json_request(
                    url=base_url + "/v1/ai/electrical/extract-graph",
                    payload={
                        "company_id": args.company_id,
                        "machine_id": args.machine_id,
                        "bubble_document_id": args.bubble_document_id,
                        "version_id": args.version_id,
                        "pdf_page_numbers": [page],
                        "force": False,
                    },
                    identity_token=token,
                    internal_secret=args.secret,
                    timeout_seconds=args.timeout_seconds,
                )
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                status_code = 0
                response = {"detail": f"transport error: {exc}"}
            final_http = status_code
            final_response = response
            status = page_status_from_response(status_code, response)
            response_path = output_dir / f"page_{page}_attempt_{attempt}.json"
            atomic_json_write(response_path, {
                "http_code": status_code,
                "response": response,
            })
            page_state.update({
                "status": status,
                "http_code": status_code,
                "response_file": str(response_path),
                "result": compact_result(response),
                "review_signature": (
                    review_signature(response)
                    if status == "review_required" else ""
                ),
                "pipeline_marker": str(
                    response.get("graph_pipeline_marker")
                    or plan.get("graph_pipeline_marker")
                    or ""
                ),
                "materializer_version": str(
                    response.get("graph_materializer_version")
                    or plan.get("graph_materializer_version")
                    or ""
                ),
                "updated_at": utc_now(),
            })
            atomic_json_write(state_path, state)
            if status != "transient_error":
                break
            if attempt < args.max_attempts:
                delay = args.retry_base_seconds * (2 ** (attempt - 1))
                print(f"  transient; retry in {delay:.1f}s", flush=True)
                time.sleep(delay)

        final_status = page_state["status"]
        if final_status == "passed":
            print(
                f"  PASS entities={final_response.get('published_page_entities')} "
                f"edges={final_response.get('published_page_edges')} "
                f"reused={final_response.get('reused_calls')}/"
                f"{final_response.get('calls')}"
            )
        elif final_status == "review_required":
            print(
                f"  REVIEW blocking={final_response.get('blocking_issue_count_this_page')} "
                f"signature={page_state.get('review_signature')}"
            )
        else:
            print(
                f"  {final_status.upper()} http={final_http} "
                f"detail={str(final_response.get('detail') or '')[:300]}"
            )
        if args.sleep_between > 0 and position < len(selected):
            time.sleep(args.sleep_between)

    run["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)
    summary = write_summary(
        output_dir=output_dir,
        state=state,
        selected_pages=selected,
    )
    print(json.dumps({
        "counts": summary["counts"],
        "usage": summary["usage"],
        "review_groups": [
            {
                "review_signature": group["review_signature"],
                "pages": group["pages"],
                "blocking_issue_type_counts": group[
                    "blocking_issue_type_counts"
                ],
            }
            for group in summary["review_groups"]
        ],
        "state": str(state_path),
        "summary": str(output_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))

    counts = summary["counts"]
    if counts.get("fatal_error") or counts.get("transient_error"):
        return 1
    if counts.get("review_required") and not args.exit_zero_on_review:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
