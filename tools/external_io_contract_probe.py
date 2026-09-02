#!/usr/bin/env python3
"""Deterministic characterization of external document I/O and Cloud Tasks dispatch."""
from __future__ import annotations

import base64
import contextlib
import inspect
import io
import json
import sys
import types
from pathlib import Path
from typing import Any


def install_stubs() -> None:
    psycopg2 = types.ModuleType("psycopg2")
    psycopg2.connect = lambda **kwargs: None
    sys.modules["psycopg2"] = psycopg2

    google = sys.modules.get("google") or types.ModuleType("google")
    if not hasattr(google, "__path__"):
        google.__path__ = []
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    tasks = types.ModuleType("google.cloud.tasks_v2")

    class CloudTasksClient:
        pass

    class HttpMethod:
        POST = "POST"

    tasks.CloudTasksClient = CloudTasksClient
    tasks.HttpMethod = HttpMethod
    cloud.tasks_v2 = tasks
    google.cloud = cloud
    sys.modules["google"] = google
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.tasks_v2"] = tasks


install_stubs()
sys.path.insert(0, str(Path.cwd()))
import main  # noqa: E402


def normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {
            str(k): normalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def capture(fn, *args, **kwargs) -> dict:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            value = fn(*args, **kwargs)
        error = None
    except Exception as exc:
        value = None
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "status_code": getattr(exc, "status_code", None),
            "detail": normalize(getattr(exc, "detail", None)),
        }
    return {
        "value": normalize(value),
        "error": error,
        "stdout": [line for line in stream.getvalue().splitlines() if line.strip()],
    }


class Patch:
    def __init__(self, **values: Any):
        self.values = values
        self.originals: dict[str, Any] = {}
        self.missing: set[str] = set()

    def __enter__(self):
        for name, value in self.values.items():
            if hasattr(main, name):
                self.originals[name] = getattr(main, name)
            else:
                self.missing.add(name)
            setattr(main, name, value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name in self.values:
            if name in self.missing:
                try:
                    delattr(main, name)
                except AttributeError:
                    pass
            else:
                setattr(main, name, self.originals[name])
        return False


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"


class FakeResponse:
    def __init__(self, *, content=b"payload", headers=None, raise_exc=None):
        self.content = content
        self.headers = dict(headers or {})
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


def make_payload(**kwargs):
    values = {
        "company_id": "company-1",
        "machine_id": "machine-1",
        "bubble_document_id": "document-1",
    }
    values.update(kwargs)
    return main.IngestRequest(**values)


# ---------------------------------------------------------------------------
# Direct external-file transport characterization
# ---------------------------------------------------------------------------
file_rows: dict[str, Any] = {
    "signatures": {
        name: {
            "signature": safe_signature(getattr(main, name)),
            "module": getattr(getattr(main, name), "__module__", ""),
        }
        for name in [
            "_strip_data_url_prefix",
            "_decode_file_base64",
            "_detect_filename_from_url",
            "_load_ingest_document_file",
        ]
    }
}

for name, value in {
    "plain": "QUJD",
    "spaced": "  QUJD  ",
    "data_url": "data:application/pdf;base64,QUJD",
    "data_url_upper": "DATA:APPLICATION/PDF;BASE64,QUJD",
    "data_url_extra_comma": "data:text/plain,one,two",
    "comma_non_data": "x,y",
    "empty": "",
}.items():
    file_rows[f"strip_{name}"] = capture(main._strip_data_url_prefix, value)

for name, value in {
    "plain": base64.b64encode(b"ABC\x00xyz").decode(),
    "data_url": "data:application/pdf;base64," + base64.b64encode(b"PDFDATA").decode(),
    "empty": "",
    "invalid_chars": "@@not-base64@@",
    "invalid_padding": "QQ=",
}.items():
    file_rows[f"decode_{name}"] = capture(main._decode_file_base64, value)

for name, url in {
    "simple": "https://example.test/path/manual.pdf",
    "encoded": "https://example.test/path/My%20Manual.xlsx?token=abc",
    "trailing": "https://example.test/path/",
    "protocol_relative": "//example.test/a/b.pdf",
    "fragment": "https://example.test/a/file.pdf#page=2",
    "empty": "",
}.items():
    file_rows[f"filename_{name}"] = capture(main._detect_filename_from_url, url)

with Patch(
    urlparse=lambda _value: types.SimpleNamespace(path="/patched/Encoded%20Name.PDF"),
    unquote=lambda value: value.replace("%20", "_"),
):
    file_rows["filename_late_bound_url_helpers"] = capture(
        main._detect_filename_from_url,
        "ignored",
    )

with Patch(_strip_data_url_prefix=lambda _value: "QUJD"):
    file_rows["decode_late_bound_strip_helper"] = capture(
        main._decode_file_base64,
        "not-base64-before-patch",
    )

file_rows["load_missing"] = capture(
    main._load_ingest_document_file,
    make_payload(),
    "document-1",
)
file_rows["load_base64_explicit"] = capture(
    main._load_ingest_document_file,
    make_payload(
        file_url="https://example.test/fallback.pdf",
        file_base64=base64.b64encode(b"ABC").decode(),
        filename="Given.XLSX",
        content_type=(
            " Application/Vnd.Openxmlformats-Officedocument."
            "Spreadsheetml.Sheet ; charset=binary "
        ),
    ),
    "document-1",
)
file_rows["load_base64_url_filename"] = capture(
    main._load_ingest_document_file,
    make_payload(
        file_url="//example.test/a/My%20File.PDF",
        file_base64=base64.b64encode(b"XYZ").decode(),
    ),
    "document-fallback",
)
file_rows["load_base64_id_fallback"] = capture(
    main._load_ingest_document_file,
    make_payload(file_base64=base64.b64encode(b"XYZ").decode()),
    "document-fallback",
)

calls: list[dict[str, Any]] = []

def get_success(url, timeout):
    calls.append({"url": url, "timeout": timeout})
    return FakeResponse(
        content=b"REMOTE",
        headers={
            "Content-Type": "Application/PDF; charset=binary",
            "Content-Disposition": "attachment; filename='Remote Manual.PDF'",
        },
    )

with Patch(FETCH_TIMEOUT=17):
    original_get = main.requests.get
    try:
        main.requests.get = get_success
        row = capture(
            main._load_ingest_document_file,
            make_payload(file_url="//example.test/source/original.pdf"),
            "document-1",
        )
        row["calls"] = normalize(calls)
        file_rows["load_url_success"] = row
    finally:
        main.requests.get = original_get

calls = []

def get_no_content_disposition(url, timeout):
    calls.append({"url": url, "timeout": timeout})
    return FakeResponse(
        content=b"REMOTE2",
        headers={"Content-Type": "application/octet-stream"},
    )

original_get = main.requests.get
try:
    main.requests.get = get_no_content_disposition
    row = capture(
        main._load_ingest_document_file,
        make_payload(file_url="https://example.test/x/My%20Remote.XLSX?sig=1"),
        "document-1",
    )
    row["calls"] = normalize(calls)
    file_rows["load_url_no_content_disposition"] = row
finally:
    main.requests.get = original_get


def get_network_error(_url, timeout):
    raise RuntimeError(f"network down at {timeout}")

original_get = main.requests.get
try:
    main.requests.get = get_network_error
    file_rows["load_url_network_error"] = capture(
        main._load_ingest_document_file,
        make_payload(file_url="https://example.test/x.pdf"),
        "document-1",
    )
finally:
    main.requests.get = original_get


def get_http_exception(_url, timeout):
    return FakeResponse(
        raise_exc=main.HTTPException(
            status_code=418,
            detail={"reason": "patched-http-error", "timeout": timeout},
        )
    )

original_get = main.requests.get
try:
    main.requests.get = get_http_exception
    file_rows["load_url_http_exception_passthrough"] = capture(
        main._load_ingest_document_file,
        make_payload(file_url="https://example.test/x.pdf"),
        "document-1",
    )
finally:
    main.requests.get = original_get

with Patch(
    _decode_file_base64=lambda _value: b"PATCHED-DECODE",
    _detect_filename_from_url=lambda _url: "patched.name",
):
    file_rows["load_late_bound_helpers"] = capture(
        main._load_ingest_document_file,
        make_payload(
            file_url="https://example.test/x",
            file_base64="ignored",
        ),
        "document-1",
    )


# ---------------------------------------------------------------------------
# Full ingest-to-Cloud-Tasks integration characterization
# ---------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append({"op": "cursor_enter"})
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append({"op": "cursor_exit", "exc": getattr(exc_type, "__name__", None)})
        return False

    def execute(self, sql, params=None):
        self.events.append(
            {
                "op": "execute",
                "sql": " ".join(str(sql).split()),
                "params": normalize(params),
            }
        )


class FakeConnection:
    def __init__(self, events):
        self.events = events

    def cursor(self):
        return FakeCursor(self.events)

    def commit(self):
        self.events.append({"op": "commit"})

    def close(self):
        self.events.append({"op": "close"})


class TaskRecorder:
    def __init__(self, *, create_exc=None, path_exc=None):
        self.create_exc = create_exc
        self.path_exc = path_exc
        self.events: list[dict[str, Any]] = []
        owner = self

        class Client:
            def __init__(self):
                owner.events.append({"op": "client"})

            def queue_path(self, project, location, queue):
                owner.events.append(
                    {
                        "op": "queue_path",
                        "args": [project, location, queue],
                    }
                )
                if owner.path_exc is not None:
                    raise owner.path_exc
                return f"projects/{project}/locations/{location}/queues/{queue}"

            def create_task(self, request):
                owner.events.append(
                    {
                        "op": "create_task",
                        "request": normalize(request),
                    }
                )
                if owner.create_exc is not None:
                    raise owner.create_exc
                return {"name": "task-1"}

        class HttpMethod:
            POST = "POST"

        self.api = types.SimpleNamespace(
            CloudTasksClient=Client,
            HttpMethod=HttpMethod,
        )


def run_ingest_cloud_case(
    env: dict[str, str],
    *,
    create_exc: Exception | None = None,
    path_exc: Exception | None = None,
) -> dict[str, Any]:
    task_recorder = TaskRecorder(create_exc=create_exc, path_exc=path_exc)
    db_events: list[dict[str, Any]] = []
    side_effects: list[dict[str, Any]] = []
    logs: list[list[str]] = []

    loaded = {
        "data": b"PK\x03\x04FAKE-XLSX",
        "url": "",
        "content_type": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        "content_disposition": "",
        "detected_filename": "fixture.xlsx",
        "detected_extension": ".xlsx",
        "source_mode": "file_base64",
    }

    def month_usage(company_id, month_key):
        side_effects.append(
            {"op": "month_usage", "company_id": company_id, "month_key": month_key}
        )
        return {
            "ledger_available": True,
            "ingest_credits_used_month": 12.5,
        }

    def invalidate(company_id):
        side_effects.append({"op": "invalidate", "company_id": company_id})

    def prepare_event(**kwargs):
        side_effects.append({"op": "prepare_event", "kwargs": normalize(kwargs)})
        return True

    original_environ = main.os.environ
    try:
        main.os.environ = dict(env)
        with Patch(
            AI_INTERNAL_SECRET="secret-value",
            tasks_v2=task_recorder.api,
            print=lambda *args: logs.append([str(value) for value in args]),
            _load_ingest_document_file=lambda payload, bubble_document_id: dict(loaded),
            _looks_like_xlsx_document=lambda data, extension, content_type: True,
            XLSX_INGEST_ENABLED=True,
            MAX_XLSX_BYTES=10_000_000,
            XLSX_MIN_TEXT_CHARS=1,
            MIN_PAGE_CHARS=1,
            _extract_xlsx_sheets_as_pages=lambda data, detected_filename="": [
                "Sheet: Data\nA | B\n1 | 2"
            ],
            _effective_ingest_request_key=lambda **kwargs: "request-key-effective",
            _build_ingest_usage_event_id=lambda *args: "usage-event-effective",
            _db_conn=lambda: FakeConnection(db_events),
            _v13_invalidate_company_knowledge=invalidate,
            _ingest_prepare_event=prepare_event,
            _ingest_month_usage=month_usage,
        ):
            response = capture(
                main.ingest_document,
                main.IngestRequest(
                    company_id="company-1",
                    machine_id="machine-1",
                    bubble_document_id="document-1",
                    file_base64="ignored-by-patch",
                    filename="fixture.xlsx",
                    content_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    ai_scope="machine_all",
                    ingest_request_key="request-key-input",
                    ingest_month_key="2026-09",
                    ingest_credits_enforced=False,
                ),
                "secret-value",
            )
    finally:
        main.os.environ = original_environ

    return {
        "response": response,
        "task_events": normalize(task_recorder.events),
        "db_events": normalize(db_events),
        "side_effects": normalize(side_effects),
        "logs": normalize(logs),
    }


cloud_rows = {
    "explicit": run_ingest_cloud_case(
        {
            "GOOGLE_CLOUD_PROJECT": " project-x ",
            "MM_CLOUD_TASKS_LOCATION": " us-central1 ",
            "MM_CLOUD_TASKS_QUEUE": " queue-x ",
            "SERVICE_URL": " https://svc.test/ ",
        }
    ),
    "fallback_environment": run_ingest_cloud_case(
        {
            "GCP_PROJECT": " project-gcp ",
            "K_SERVICE_URL": " https://fallback.test/// ",
        }
    ),
    "default_project_queue_location": run_ingest_cloud_case(
        {"SERVICE_URL": "https://svc.test"}
    ),
    "missing_service_url": run_ingest_cloud_case({}),
    "queue_path_failure": run_ingest_cloud_case(
        {"SERVICE_URL": "https://svc.test"},
        path_exc=RuntimeError("path fail"),
    ),
    "create_task_failure": run_ingest_cloud_case(
        {"SERVICE_URL": "https://svc.test"},
        create_exc=RuntimeError("create fail"),
    ),
}

print(
    json.dumps(
        {
            "file_transport": file_rows,
            "ingest_cloud_tasks": cloud_rows,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
)
