from __future__ import annotations

import json
import sys
import types


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

    tasks.CloudTasksClient = CloudTasksClient
    cloud.tasks_v2 = tasks
    google.cloud = cloud
    sys.modules["google"] = google
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.tasks_v2"] = tasks


class FakeBudget:
    def __init__(self, remaining: float):
        self._remaining = float(remaining)
        self.ensure_calls: list[float] = []

    def ensure_time(self, seconds: float) -> None:
        self.ensure_calls.append(float(seconds))

    def remaining(self) -> float:
        return self._remaining


class FakeContext:
    def __init__(self, value: float = 0.0, *, error: bool = False):
        self.value = value
        self.error = error

    def get(self):
        if self.error:
            raise RuntimeError("context failure")
        return self.value


def serialize_error(exc: Exception) -> dict:
    return {
        "type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "detail": getattr(exc, "detail", str(exc)),
        "message": str(exc),
    }


def connection_case(main, *, name: str, budget, assurance, monotonic: float, missing=False, connect_error=False):
    main.DB_HOST = "" if missing else "db.internal"
    main.DB_NAME = "machinemind"
    main.DB_USER = "dbuser"
    main.DB_PASSWORD = "secret"
    main.V13_DB_CONNECT_TIMEOUT_SECONDS = 5
    main.V13_DB_STATEMENT_TIMEOUT_MS = 9000
    main._v13_current_budget = (
        (lambda: (_ for _ in ()).throw(RuntimeError("budget failure")))
        if budget == "error"
        else (lambda: budget)
    )
    main._V13_ASSURANCE_DEADLINE_CTX = assurance
    main.time_module.monotonic = lambda: float(monotonic)

    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        if connect_error:
            raise RuntimeError("connect failure")
        return "CONNECTED"

    main.psycopg2.connect = connect
    try:
        result = main._db_conn()
        error = None
    except Exception as exc:
        result = None
        error = serialize_error(exc)

    return {
        "name": name,
        "result": result,
        "error": error,
        "connect_calls": calls,
        "budget_ensure_calls": (
            list(budget.ensure_calls) if isinstance(budget, FakeBudget) else []
        ),
    }


def vector_cases(main):
    vectors = [
        [],
        [0.0],
        [1.234567891, -0.000000004, 999.5],
        [-12.25, 3.141592653589793],
    ]
    return [main._vector_literal(value) for value in vectors]


class FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params):
        self.calls.append({"query": query, "params": list(params)})

    def fetchall(self):
        return [("company_id",), ("machine_id",), ("updated_at",)]


def table_columns_case(main):
    cur = FakeCursor()
    result = main._get_table_columns(cur, "document_chunks")
    return {"columns": sorted(result), "calls": cur.calls}


install_stubs()
import main

cases = [
    connection_case(
        main, name="missing_env", budget=None, assurance=FakeContext(0.0),
        monotonic=100.0, missing=True,
    ),
    connection_case(
        main, name="plain", budget=None, assurance=FakeContext(0.0),
        monotonic=100.0,
    ),
    connection_case(
        main, name="budget_only_normal", budget=FakeBudget(10.3),
        assurance=FakeContext(0.0), monotonic=100.0,
    ),
    connection_case(
        main, name="budget_only_low", budget=FakeBudget(2.2),
        assurance=FakeContext(0.0), monotonic=100.0,
    ),
    connection_case(
        main, name="assurance_only", budget=None,
        assurance=FakeContext(103.0), monotonic=100.0,
    ),
    connection_case(
        main, name="budget_and_assurance", budget=FakeBudget(8.0),
        assurance=FakeContext(101.4), monotonic=100.0,
    ),
    connection_case(
        main, name="budget_getter_error", budget="error",
        assurance=FakeContext(0.0), monotonic=100.0,
    ),
    connection_case(
        main, name="assurance_context_error", budget=FakeBudget(7.0),
        assurance=FakeContext(error=True), monotonic=100.0,
    ),
    connection_case(
        main, name="connect_error", budget=None, assurance=FakeContext(0.0),
        monotonic=100.0, connect_error=True,
    ),
]
print(json.dumps({
    "connection_cases": cases,
    "vector_cases": vector_cases(main),
    "table_columns_case": table_columns_case(main),
}, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
