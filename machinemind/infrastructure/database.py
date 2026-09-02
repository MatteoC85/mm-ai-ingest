"""Low-level PostgreSQL connection and SQL utility boundary.

The production monolith historically resolved request-budget and retrieval-assurance
contexts through its own module globals. ``connect_database`` accepts that runtime
mapping explicitly so the extraction preserves those semantics without importing
``main`` or creating a circular dependency.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import HTTPException


def connect_database(
    *,
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    runtime_globals: Mapping[str, Any],
    connect_fn: Callable[..., Any],
    monotonic_fn: Callable[[], float],
):
    if not (db_host and db_user and db_password):
        raise HTTPException(status_code=500, detail="DB env missing")

    kwargs: dict[str, Any] = {
        "host": db_host,
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
    }

    # Preserve the certified behavior: only a live V13/Assistant Core request or
    # bounded retrieval-assurance operation receives connect/statement timeouts.
    budget = None
    try:
        budget_fn = runtime_globals.get("_v13_current_budget")
        budget = budget_fn() if callable(budget_fn) else None
    except Exception:
        budget = None

    assurance_remaining = None
    try:
        assurance_ctx = runtime_globals.get("_V13_ASSURANCE_DEADLINE_CTX")
        assurance_deadline = (
            float(assurance_ctx.get() or 0.0) if assurance_ctx is not None else 0.0
        )
        if assurance_deadline > 0.0:
            assurance_remaining = max(0.1, assurance_deadline - monotonic_fn())
    except Exception:
        assurance_remaining = None

    if budget is not None and assurance_remaining is None:
        budget.ensure_time(1.5)
        remaining = max(1.5, float(budget.remaining()))
        connect_limit = int(
            runtime_globals.get("V13_DB_CONNECT_TIMEOUT_SECONDS", 5) or 5
        )
        statement_limit = int(
            runtime_globals.get("V13_DB_STATEMENT_TIMEOUT_MS", 9000) or 9000
        )
        kwargs["connect_timeout"] = max(
            1, min(connect_limit, int(max(1.0, remaining - 0.5)))
        )
        kwargs["options"] = (
            "-c statement_timeout="
            f"{max(1000, min(statement_limit, int(max(1.0, remaining - 0.5) * 1000)))}"
        )
    elif budget is not None or assurance_remaining is not None:
        if budget is not None:
            budget.ensure_time(1.0)
            remaining = max(0.5, float(budget.remaining()))
        else:
            remaining = max(0.5, float(assurance_remaining or 0.5))
        if assurance_remaining is not None:
            remaining = min(remaining, max(0.5, float(assurance_remaining)))
        connect_limit = int(
            runtime_globals.get("V13_DB_CONNECT_TIMEOUT_SECONDS", 5) or 5
        )
        statement_limit = int(
            runtime_globals.get("V13_DB_STATEMENT_TIMEOUT_MS", 9000) or 9000
        )
        kwargs["connect_timeout"] = max(
            1, min(connect_limit, int(max(1.0, remaining - 0.2)))
        )
        kwargs["options"] = (
            "-c statement_timeout="
            f"{max(500, min(statement_limit, int(max(0.5, remaining - 0.2) * 1000)))}"
        )

    return connect_fn(**kwargs)


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def get_table_columns(cur: Any, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s;
        """,
        (table_name,),
    )
    return {row[0] for row in cur.fetchall()}
