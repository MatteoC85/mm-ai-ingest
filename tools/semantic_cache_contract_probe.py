from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable


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
    if isinstance(value, dict):
        return {
            str(k): normalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize(v) for v in value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 9)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"


def call_with_log(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            value = fn(*args, **kwargs)
        error = None
    except Exception as exc:  # pragma: no cover - intentionally serialized
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
        "log": [line for line in stream.getvalue().splitlines() if line.strip()],
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


class FakeBudget:
    def __init__(self, embedding_cache: dict | None = None):
        self.semantic_cache = "initial"
        self.route = "initial"
        self.embedding_cache = dict(embedding_cache or {})


class FakeCursor:
    def __init__(
        self,
        owner: "FakeConnection",
        *,
        fetchall_values: list[Any] | None = None,
        fetchone_values: list[Any] | None = None,
        fail_on_execute: int | None = None,
    ):
        self.owner = owner
        self.fetchall_values = list(fetchall_values or [])
        self.fetchone_values = list(fetchone_values or [])
        self.fail_on_execute = fail_on_execute
        self.execute_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params: Any = None):
        self.execute_count += 1
        normalized_sql = " ".join(str(sql).split())
        self.owner.commands.append({"sql": normalized_sql, "params": normalize(params)})
        if self.fail_on_execute == self.execute_count:
            raise RuntimeError(f"sql-failure-{self.execute_count}")

    def fetchall(self):
        if self.fetchall_values:
            return self.fetchall_values.pop(0)
        return []

    def fetchone(self):
        if self.fetchone_values:
            return self.fetchone_values.pop(0)
        return None


class FakeConnection:
    def __init__(
        self,
        *,
        fetchall_values: list[Any] | None = None,
        fetchone_values: list[Any] | None = None,
        fail_on_execute: int | None = None,
    ):
        self.commands: list[dict[str, Any]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0
        self.cursor_obj = FakeCursor(
            self,
            fetchall_values=fetchall_values,
            fetchone_values=fetchone_values,
            fail_on_execute=fail_on_execute,
        )

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "commands": self.commands,
            "commits": self.commits,
            "rollbacks": self.rollbacks,
            "closes": self.closes,
        }


class ConnectionFactory:
    def __init__(self, connections: list[FakeConnection] | None = None, *, error: Exception | None = None):
        self.connections = list(connections or [])
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if not self.connections:
            raise AssertionError("No fake connection remaining")
        return self.connections.pop(0)


def reset_cache_state() -> None:
    main._V13_CACHE_READY = None
    main._V13_CACHE_ERROR = ""
    main._V13_CACHE_RETRY_AT = 0.0


output: dict[str, Any] = {}

surface_names = [
    "_v13_cache_bootstrap",
    "_v13_normalize_query",
    "_v13_scope_key",
    "_v13_get_knowledge_version",
    "_v13_bump_knowledge_version",
    "_v13_invalidate_company_knowledge",
    "_v13_cache_code_tokens",
    "_v13_query_number_tokens",
    "_v13_query_polarity_signature",
    "_v13_query_source_signature",
    "_v13_semantic_cache_compatible",
    "_v13_jsonb_to_python",
    "_v13_cache_lookup",
    "_v13_response_quality",
    "_assistant_core_cache_certified",
    "_v13_cache_store",
]
output["surface"] = {
    name: {
        "name": getattr(getattr(main, name), "__name__", None),
        "qualname": getattr(getattr(main, name), "__qualname__", None),
        "module": getattr(getattr(main, name), "__module__", None),
        "signature": safe_signature(getattr(main, name)),
    }
    for name in surface_names
}

# Pure deterministic guards and response admission/quality rules.
pure: dict[str, Any] = {}
pure["normalize"] = {
    value: main._v13_normalize_query(value)
    for value in ["  Caffè   POMPA  ", "A\nB\tC", "", "K12 / X4.7"]
}
pure["scope_keys"] = [
    main._v13_scope_key({"ai_scope": "machine_all"}),
    main._v13_scope_key({
        "ai_scope": "document_ids",
        "document_ids": ["b", "a", "a"],
        "bubble_document_id": "doc-1",
        "_v13_top_k": 7,
        "_v13_max_causes": 2,
    }),
    main._v13_scope_key({
        "ai_scope": "company_general",
        "document_ids": '["z","x"]',
        "_v13_top_k": "bad",
    }),
]
pure["code_tokens"] = {
    value: main._v13_cache_code_tokens(value)
    for value in [
        "Controlla K12, X4.7 e PLC_MAIN",
        "Protection reset manual",
        "E100-AX-55 e 1234",
        "AME/1000:SN v2.0",
    ]
}
pure["number_tokens"] = {
    value: main._v13_query_number_tokens(value)
    for value in [
        "1000 kg, 24 V, 3,5 bar e 10 mm/min",
        "1 a ogni ciclo e 2 dopo",
        "50% 80 °C 100 rpm",
        "-12.5 kW +5 A 9 s",
    ]
}
pure["polarity"] = {
    value: main._v13_query_polarity_signature(value)
    for value in [
        "prima del ciclo, non usare solo il manuale",
        "after completion except step 4",
        "procedura standard",
    ]
}
pure["source_signature"] = {
    value: main._v13_query_source_signature(value)
    for value in [
        "cosa dice il manuale?",
        "usa solo Excel",
        "spiegami la procedura",
    ]
}
compatibility_cases = [
    ("ask", "Come si resetta K12?", "Come si resetta K12?"),
    ("ask", "Come si resetta K12 dopo allarme?", "Procedura reset K12 dopo l'allarme"),
    ("ask", "Portata K12 1000 kg", "Portata K13 1000 kg"),
    ("ask", "Portata K12 1000 kg", "Portata K12 2000 kg"),
    ("ask", "Non usare K12", "Usare K12"),
    ("ask", "Solo dal manuale: reset K12", "Reset K12"),
    ("root_cause", "La macchina vibra in automatico", "Vibrazione macchina in automatico"),
    ("root_cause", "La macchina vibra in automatico", "La macchina non parte in manuale"),
]
pure["compatibility"] = [
    {
        "mode": mode,
        "current": current,
        "cached": cached,
        "compatible": main._v13_semantic_cache_compatible(mode, current, cached),
    }
    for mode, current, cached in compatibility_cases
]
pure["jsonb"] = [
    main._v13_jsonb_to_python(None, {"fallback": 1}),
    main._v13_jsonb_to_python({"x": 1}, {}),
    main._v13_jsonb_to_python('[1, 2, 3]', []),
    main._v13_jsonb_to_python("not-json", ["fallback"]),
]

valid_ask = {
    "ok": True,
    "status": "answered",
    "answer": "A sufficiently long grounded answer for the cache contract.",
    "answer_html": "<p>A sufficiently long grounded answer for the cache contract.</p>",
    "citations": [{"citation_id": "doc:p1", "exact_machine_scope": True, "evidence_role": "procedure"}],
    "meta": {
        "canonical_final_answer": True,
        "assistant_core_validation": {
            "answer_contract": {
                "passed": True,
                "missing_answer_facets": [],
                "missing_evidence_facets": [],
                "missing_list_items": [],
            }
        },
    },
}
valid_root = {
    "ok": True,
    "status": "answered",
    "citations": [{"citation_id": "ps:1"}, {"citation_id": "manual:2"}],
    "possible_causes": [
        {
            "cause": "Cause A",
            "why": "Supported explanation",
            "checks": ["Check A", "Check B"],
            "citations": ["ps:1"],
            "citation_ids": ["ps:1"],
        },
        {
            "cause": "Cause B",
            "why": "Other supported explanation",
            "checks": ["Check C"],
            "citations": ["manual:2"],
            "citation_ids": ["manual:2"],
        },
    ],
    "meta": {"canonical_final_answer": True},
}
cert_cases = {
    "ask_valid": valid_ask,
    "ask_no_canonical": {**valid_ask, "meta": {}},
    "ask_no_html": {**valid_ask, "answer_html": ""},
    "ask_missing_contract": {**valid_ask, "meta": {"canonical_final_answer": True}},
    "general_valid": {
        "ok": True,
        "status": "answered",
        "grounding": "general_technical_knowledge",
        "answer": "A sufficiently long general technical answer for validation.",
        "answer_html": "<p>A sufficiently long general technical answer for validation.</p>",
        "citations": [],
        "meta": {
            "canonical_final_answer": True,
            "assistant_core_validation": {"answer_contract": {"passed": True}},
        },
    },
    "root_valid": valid_root,
    "root_wrong_citation": {
        **valid_root,
        "possible_causes": [{**valid_root["possible_causes"][0], "citation_ids": ["missing"]}],
    },
    "root_missing_checks": {
        **valid_root,
        "possible_causes": [{**valid_root["possible_causes"][0], "checks": []}],
    },
}
pure["certified"] = {
    name: main._assistant_core_cache_certified(
        "root_cause" if name.startswith("root_") else "ask",
        response,
    )
    for name, response in cert_cases.items()
}
pure["quality"] = {
    "invalid": main._v13_response_quality("ask", {}),
    "ask_valid": main._v13_response_quality("ask", valid_ask),
    "ask_short": main._v13_response_quality("ask", {**valid_ask, "answer": "short"}),
    "root_valid": main._v13_response_quality("root_cause", valid_root),
    "root_no_causes": main._v13_response_quality("root_cause", {**valid_root, "possible_causes": []}),
}
output["pure"] = normalize(pure)

# Bootstrap state and DDL behavior.
bootstrap: dict[str, Any] = {}
with Patch(V13_SEMANTIC_CACHE_ENABLED=False):
    reset_cache_state()
    bootstrap["disabled"] = call_with_log(main._v13_cache_bootstrap)
    bootstrap["disabled_state"] = [main._V13_CACHE_READY, main._V13_CACHE_ERROR, main._V13_CACHE_RETRY_AT]

with Patch(V13_SEMANTIC_CACHE_ENABLED=True):
    reset_cache_state()
    main._V13_CACHE_READY = True
    bootstrap["already_ready"] = call_with_log(main._v13_cache_bootstrap)

clock = {"now": 100.0}
with Patch(V13_SEMANTIC_CACHE_ENABLED=True):
    reset_cache_state()
    main._V13_CACHE_READY = False
    main._V13_CACHE_RETRY_AT = 200.0
    original_monotonic = main.time_module.monotonic
    main.time_module.monotonic = lambda: clock["now"]
    try:
        bootstrap["cooldown"] = call_with_log(main._v13_cache_bootstrap)
        bootstrap["cooldown_state"] = [main._V13_CACHE_READY, main._V13_CACHE_ERROR, main._V13_CACHE_RETRY_AT]
    finally:
        main.time_module.monotonic = original_monotonic

with Patch(V13_SEMANTIC_CACHE_ENABLED=True, V13_SEMANTIC_CACHE_AUTO_DDL=False):
    reset_cache_state()
    bootstrap["auto_ddl_off"] = call_with_log(main._v13_cache_bootstrap)
    bootstrap["auto_ddl_off_state"] = [main._V13_CACHE_READY, main._V13_CACHE_ERROR, main._V13_CACHE_RETRY_AT]

success_conn = FakeConnection()
with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    V13_SEMANTIC_CACHE_AUTO_DDL=True,
    _db_conn=ConnectionFactory([success_conn]),
):
    reset_cache_state()
    bootstrap["ddl_success"] = call_with_log(main._v13_cache_bootstrap)
    bootstrap["ddl_success_state"] = [main._V13_CACHE_READY, main._V13_CACHE_ERROR, main._V13_CACHE_RETRY_AT]
    bootstrap["ddl_success_db"] = success_conn.snapshot()

failure_conn = FakeConnection(fail_on_execute=2)
clock = {"now": 500.0}
with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    V13_SEMANTIC_CACHE_AUTO_DDL=True,
    V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS=60,
    _db_conn=ConnectionFactory([failure_conn]),
):
    reset_cache_state()
    original_monotonic = main.time_module.monotonic
    main.time_module.monotonic = lambda: clock["now"]
    try:
        bootstrap["ddl_failure"] = call_with_log(main._v13_cache_bootstrap)
        bootstrap["ddl_failure_state"] = [main._V13_CACHE_READY, main._V13_CACHE_ERROR, main._V13_CACHE_RETRY_AT]
        bootstrap["ddl_failure_db"] = failure_conn.snapshot()
    finally:
        main.time_module.monotonic = original_monotonic
output["bootstrap"] = normalize(bootstrap)

# Knowledge-version read, bump and fail-open invalidation.
versions: dict[str, Any] = {}
get_conn = FakeConnection(fetchone_values=[[7]])
with Patch(_v13_cache_bootstrap=lambda: True, _db_conn=ConnectionFactory([get_conn])):
    versions["get_success"] = call_with_log(main._v13_get_knowledge_version, " company-1 ")
    versions["get_success_db"] = get_conn.snapshot()
with Patch(_v13_cache_bootstrap=lambda: True, _db_conn=ConnectionFactory(error=AssertionError("must-not-connect"))):
    versions["get_blank"] = call_with_log(main._v13_get_knowledge_version, "  ")
get_fail_conn = FakeConnection(fail_on_execute=1)
with Patch(_v13_cache_bootstrap=lambda: True, _db_conn=ConnectionFactory([get_fail_conn])):
    versions["get_failure"] = call_with_log(main._v13_get_knowledge_version, "company-2")
    versions["get_failure_db"] = get_fail_conn.snapshot()
bump_conn = FakeConnection()
with Patch(_v13_cache_bootstrap=lambda: True, _db_conn=ConnectionFactory([bump_conn])):
    versions["bump_success"] = call_with_log(main._v13_bump_knowledge_version, " company-3 ")
    versions["bump_success_db"] = bump_conn.snapshot()
with Patch(_v13_cache_bootstrap=lambda: True, _db_conn=ConnectionFactory(error=AssertionError("must-not-connect"))):
    versions["bump_blank"] = call_with_log(main._v13_bump_knowledge_version, "")
with Patch(_v13_bump_knowledge_version=lambda _company_id: (_ for _ in ()).throw(RuntimeError("bump failed"))):
    versions["invalidate_fail_open"] = call_with_log(main._v13_invalidate_company_knowledge, "company-4")
output["versions"] = normalize(versions)

# Lookup behavior: bypass, exact hit, semantic hit and fail-open branches.
lookup: dict[str, Any] = {}
lookup_scope = {"ai_scope": "machine_all", "_v13_top_k": 5, "_v13_max_causes": 0}

budget = FakeBudget()
with Patch(_v13_current_budget=lambda: budget, V13_SEMANTIC_CACHE_ENABLED=True, _v13_cache_bootstrap=lambda: True):
    lookup["debug_bypass"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="q", company_id="c", machine_id="m", scope=lookup_scope,
        language="it", debug=True,
    )
    lookup["debug_budget"] = {"semantic_cache": budget.semantic_cache, "route": budget.route}

exact_response = {
    "ok": True,
    "status": "answered",
    "answer": "cached answer",
    "citations": [{"citation_id": "doc:p1"}],
    "rg_links": [{"old": True}],
    "meta": {"x": 1},
}
exact_conn = FakeConnection(fetchall_values=[[
    (11, "  Reset   K12 ", json.dumps(exact_response), 0.91, "2026-09-01T10:00:00+00:00")
]])
budget = FakeBudget()
with Patch(
    _v13_current_budget=lambda: budget,
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _v13_get_knowledge_version=lambda _company: 4,
    _v13_scope_key=lambda _scope: "scope-key",
    _db_conn=ConnectionFactory([exact_conn]),
    _build_rg_links=lambda company, citations: [{"company": company, "count": len(citations)}],
):
    lookup["exact_hit"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="reset k12", company_id="company", machine_id="machine",
        scope=lookup_scope, language="it", debug=False,
    )
    lookup["exact_hit_budget"] = {"semantic_cache": budget.semantic_cache, "route": budget.route}
    lookup["exact_hit_db"] = exact_conn.snapshot()

no_rows_conn = FakeConnection(fetchall_values=[[]])
with Patch(
    _v13_current_budget=lambda: None,
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _v13_get_knowledge_version=lambda _company: 2,
    _v13_scope_key=lambda _scope: "scope",
    _db_conn=ConnectionFactory([no_rows_conn]),
):
    lookup["no_rows"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="q", company_id="c", machine_id="m", scope=lookup_scope,
        language="it", debug=False,
    )

incompatible_conn = FakeConnection(fetchall_values=[[
    (12, "cached incompatible", json.dumps(exact_response), 0.9, "date")
]])
embed_calls = {"count": 0}
def forbidden_embed(*_args, **_kwargs):
    embed_calls["count"] += 1
    raise AssertionError("embedding must not run")
with Patch(
    _v13_current_budget=lambda: None,
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _v13_get_knowledge_version=lambda _company: 2,
    _v13_scope_key=lambda _scope: "scope",
    _db_conn=ConnectionFactory([incompatible_conn]),
    _v13_semantic_cache_compatible=lambda *_args: False,
    _openai_embed_texts=forbidden_embed,
):
    lookup["incompatible"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="new", company_id="c", machine_id="m", scope=lookup_scope,
        language="it", debug=False,
    )
    lookup["incompatible_embed_calls"] = embed_calls["count"]

semantic_response = {
    "ok": True,
    "status": "answered",
    "answer": "semantic cached answer",
    "citations": [{"citation_id": "doc:p2"}],
    "meta": {},
}
semantic_rows = [
    (21, "cached one", json.dumps(semantic_response), 0.93, "2026-09-01"),
    (22, "cached two", json.dumps({**semantic_response, "answer": "second"}), 0.92, "2026-08-31"),
]
semantic_conn1 = FakeConnection(fetchall_values=[semantic_rows])
semantic_conn2 = FakeConnection(fetchall_values=[[(21, json.dumps([1.0, 0.0])), (22, json.dumps([0.0, 1.0]))]])
budget = FakeBudget()
with Patch(
    _v13_current_budget=lambda: budget,
    V13_SEMANTIC_CACHE_ENABLED=True,
    V13_SEMANTIC_CACHE_THRESHOLD_ASK=0.965,
    _v13_cache_bootstrap=lambda: True,
    _v13_get_knowledge_version=lambda _company: 5,
    _v13_scope_key=lambda _scope: "scope",
    _v13_semantic_cache_compatible=lambda *_args: True,
    _openai_embed_texts=lambda texts, timeout=10: [[1.0, 0.0]],
    _cosine_sim=lambda a, b: sum(x * y for x, y in zip(a, b)),
    _db_conn=ConnectionFactory([semantic_conn1, semantic_conn2]),
    _build_rg_links=lambda company, citations: [{"refreshed": True, "company": company}],
):
    lookup["semantic_hit"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="new query", company_id="c", machine_id="m", scope=lookup_scope,
        language="it", debug=False,
    )
    lookup["semantic_hit_budget"] = {"semantic_cache": budget.semantic_cache, "route": budget.route}
    lookup["semantic_hit_db"] = [semantic_conn1.snapshot(), semantic_conn2.snapshot()]

lookup_fail_conn = FakeConnection(fail_on_execute=1)
with Patch(
    _v13_current_budget=lambda: None,
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _v13_get_knowledge_version=lambda _company: 1,
    _v13_scope_key=lambda _scope: "scope",
    _db_conn=ConnectionFactory([lookup_fail_conn]),
):
    lookup["db_failure"] = call_with_log(
        main._v13_cache_lookup,
        mode="ask", q="q", company_id="c", machine_id="m", scope=lookup_scope,
        language="it", debug=False,
    )
    lookup["db_failure_db"] = lookup_fail_conn.snapshot()
output["lookup"] = normalize(lookup)

# Store behavior and persistence payload normalization.
store: dict[str, Any] = {}
store_scope = {"ai_scope": "machine_all", "_v13_top_k": 5}

with Patch(V13_SEMANTIC_CACHE_ENABLED=True, _v13_cache_bootstrap=lambda: True):
    store["debug_bypass"] = call_with_log(
        main._v13_cache_store,
        mode="ask", q="q", company_id="c", machine_id="m", scope=store_scope,
        language="it", response=valid_ask, debug=True,
    )

with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _assistant_core_cache_certified=lambda *_args: False,
):
    store["not_certified"] = call_with_log(
        main._v13_cache_store,
        mode="ask", q="q", company_id="c", machine_id="m", scope=store_scope,
        language="it", response=valid_ask, debug=False,
    )

budget_missing = FakeBudget()
with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    _v13_cache_bootstrap=lambda: True,
    _assistant_core_cache_certified=lambda *_args: True,
    _v13_response_quality=lambda *_args: 0.9,
    _v13_current_budget=lambda: budget_missing,
):
    store["budget_without_vector"] = call_with_log(
        main._v13_cache_store,
        mode="ask", q="q", company_id="c", machine_id="m", scope=store_scope,
        language="it", response=valid_ask, debug=False,
    )

embed_key = (main.OPENAI_EMBED_MODEL, "query")
budget_ready = FakeBudget({embed_key: [0.1, 0.2]})
stored_response = {
    **valid_ask,
    "debug": {"secret": True},
    "rg_links": [{"stale": True}],
    "meta": {
        **valid_ask["meta"],
        "v13_runtime": {"elapsed": 1},
        "v13_semantic_cache": {"hit": False},
        "keep": "yes",
    },
}
store_conn = FakeConnection()
with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    V13_SEMANTIC_CACHE_MIN_QUALITY=0.8,
    V13_SEMANTIC_CACHE_TTL_SECONDS=604800,
    V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY=600,
    V13_ENGINE_KEY="engine-key",
    _v13_cache_bootstrap=lambda: True,
    _assistant_core_cache_certified=lambda *_args: True,
    _v13_response_quality=lambda *_args: 0.91,
    _v13_current_budget=lambda: budget_ready,
    _openai_embed_texts=lambda texts, timeout=10: [[0.1, 0.2]],
    _v13_get_knowledge_version=lambda _company: 8,
    _v13_normalize_query=lambda _q: "normalized query",
    _v13_scope_key=lambda _scope: "scope-key",
    _db_conn=ConnectionFactory([store_conn]),
):
    store["success"] = call_with_log(
        main._v13_cache_store,
        mode="ask", q="query", company_id=" company ", machine_id="machine",
        scope=store_scope, language="it", response=stored_response, debug=False,
    )
    snapshot = store_conn.snapshot()
    # Parse the JSON parameters so comparison documents the actual sanitized payload.
    insert_params = snapshot["commands"][0]["params"] if snapshot["commands"] else []
    if isinstance(insert_params, list) and len(insert_params) >= 12:
        try:
            insert_params[10] = json.loads(insert_params[10])
        except Exception:
            pass
        try:
            insert_params[11] = json.loads(insert_params[11])
        except Exception:
            pass
    store["success_db"] = snapshot

store_fail_conn = FakeConnection(fail_on_execute=1)
with Patch(
    V13_SEMANTIC_CACHE_ENABLED=True,
    V13_SEMANTIC_CACHE_MIN_QUALITY=0.8,
    V13_ENGINE_KEY="engine-key",
    _v13_cache_bootstrap=lambda: True,
    _assistant_core_cache_certified=lambda *_args: True,
    _v13_response_quality=lambda *_args: 0.9,
    _v13_current_budget=lambda: None,
    _openai_embed_texts=lambda texts, timeout=10: [[0.3, 0.4]],
    _v13_get_knowledge_version=lambda _company: 2,
    _v13_normalize_query=lambda _q: "q",
    _v13_scope_key=lambda _scope: "scope",
    _db_conn=ConnectionFactory([store_fail_conn]),
):
    store["db_failure"] = call_with_log(
        main._v13_cache_store,
        mode="ask", q="q", company_id="c", machine_id="m", scope=store_scope,
        language="it", response=valid_ask, debug=False,
    )
    store["db_failure_db"] = store_fail_conn.snapshot()
output["store"] = normalize(store)

# Restore a neutral cache state so the final snapshot is deterministic.
reset_cache_state()
output["final_state"] = normalize({
    "ready": main._V13_CACHE_READY,
    "error": main._V13_CACHE_ERROR,
    "retry_at": main._V13_CACHE_RETRY_AT,
})

print(json.dumps(normalize(output), ensure_ascii=False, sort_keys=True))
