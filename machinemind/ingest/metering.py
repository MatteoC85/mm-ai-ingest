"""Document-ingest usage ledger and embedding-cost metering boundary.

This module is a mechanical extraction from the production composition root.
All mutable state, connection factories, callbacks and configuration are
resolved through a late-bound runtime supplied by ``main`` so the historical
idempotency, fail-open behavior, context propagation and monkeypatch surface
remain intact.
"""
from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, MutableMapping, Optional


@dataclass(frozen=True)
class IngestMeteringRuntime:
    connect_db: Callable[[], Any]
    state_globals: MutableMapping[str, Any]
    ledger_auto_ddl: bool
    processing_stale_seconds: int
    credits_per_usd: Decimal
    embed_input_price_usd_per_million: Decimal
    embed_model: str
    pricing_version: str
    metering_version: str
    ai_internal_secret: str
    normalize_month_key_fn: Callable[[Any], str]
    credits_for_cost_fn: Callable[[Decimal], Decimal]
    ledger_bootstrap_fn: Callable[[], bool]
    event_snapshot_fn: Callable[[str], Optional[dict]]
    build_usage_event_id_fn: Callable[[str, str, str], str]
    prepare_event_fn: Callable[..., bool]
    claim_event_fn: Callable[[str], tuple[str, Optional[dict]]]
    finalize_event_fn: Callable[..., Optional[dict]]
    month_usage_fn: Callable[[str, str], dict]
    public_fields_fn: Callable[..., dict]
    json_dumps_fn: Callable[..., str]
    log_fn: Callable[..., Any] = print


def ingest_decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value if value is not None else default))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def normalize_ingest_month_key(value: Any) -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", raw):
        return raw
    return datetime.now(timezone.utc).strftime("%Y-%m")


def effective_ingest_request_key(
    *,
    company_id: str,
    bubble_document_id: str,
    requested_key: Any,
    file_sha256: str = "",
) -> str:
    base = str(requested_key or "").strip() or str(bubble_document_id or "").strip()
    if file_sha256:
        base = f"{base}:{file_sha256[:24]}"
    return base or hashlib.sha256(str(company_id or "").encode("utf-8")).hexdigest()[:24]


def build_ingest_usage_event_id(company_id: str, month_key: str, request_key: str) -> str:
    raw = f"{company_id}\n{month_key}\n{request_key}".encode("utf-8")
    return "ingest_" + hashlib.sha256(raw).hexdigest()[:32]


def ingest_credits_for_cost(
    cost_usd: Decimal,
    *,
    runtime: IngestMeteringRuntime,
) -> Decimal:
    credits = max(Decimal("0"), cost_usd) * max(
        Decimal("0"), runtime.credits_per_usd
    )
    return credits.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def ledger_bootstrap(*, runtime: IngestMeteringRuntime) -> bool:
    state = runtime.state_globals
    if state.get("_INGEST_LEDGER_READY") is True:
        return True

    lock = state["_INGEST_LEDGER_LOCK"]
    with lock:
        if state.get("_INGEST_LEDGER_READY") is True:
            return True
        if not runtime.ledger_auto_ddl:
            state["_INGEST_LEDGER_READY"] = True
            return True

        conn = None
        try:
            conn = runtime.connect_db()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.mm_ingest_usage_events (
                        usage_event_id TEXT PRIMARY KEY,
                        request_key TEXT NOT NULL,
                        company_id TEXT NOT NULL,
                        bubble_document_id TEXT NOT NULL,
                        month_key TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'queued',
                        embedding_model TEXT NOT NULL DEFAULT '',
                        embedding_calls BIGINT NOT NULL DEFAULT 0,
                        embedding_input_tokens BIGINT NOT NULL DEFAULT 0,
                        actual_cost_usd NUMERIC(20, 10) NOT NULL DEFAULT 0,
                        ingest_credits NUMERIC(20, 6) NOT NULL DEFAULT 0,
                        pricing_version TEXT NOT NULL DEFAULT '',
                        metering_version TEXT NOT NULL DEFAULT '',
                        usage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        completed_at TIMESTAMPTZ
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_mm_ingest_usage_company_month
                    ON public.mm_ingest_usage_events(company_id, month_key, updated_at);
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_mm_ingest_usage_request_month
                    ON public.mm_ingest_usage_events(company_id, month_key, request_key);
                    """
                )
            conn.commit()
            state["_INGEST_LEDGER_READY"] = True
            state["_INGEST_LEDGER_ERROR"] = ""
            return True
        except Exception as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            state["_INGEST_LEDGER_READY"] = False
            state["_INGEST_LEDGER_ERROR"] = str(exc)[:1000]
            runtime.log_fn(
                "INGEST_LEDGER_BOOTSTRAP_FAIL_OPEN",
                state["_INGEST_LEDGER_ERROR"],
            )
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


def prepare_event(
    *,
    usage_event_id: str,
    request_key: str,
    company_id: str,
    bubble_document_id: str,
    month_key: str,
    runtime: IngestMeteringRuntime,
) -> bool:
    if not runtime.ledger_bootstrap_fn():
        return False
    conn = None
    try:
        conn = runtime.connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.mm_ingest_usage_events(
                    usage_event_id, request_key, company_id, bubble_document_id,
                    month_key, status, pricing_version, metering_version
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s)
                ON CONFLICT (usage_event_id) DO NOTHING;
                """,
                (
                    usage_event_id,
                    request_key,
                    company_id,
                    bubble_document_id,
                    month_key,
                    runtime.pricing_version,
                    runtime.metering_version,
                ),
            )
        conn.commit()
        return True
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        runtime.log_fn("INGEST_LEDGER_PREPARE_FAIL_OPEN", str(exc)[:700])
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def event_snapshot(
    usage_event_id: str,
    *,
    runtime: IngestMeteringRuntime,
) -> Optional[dict]:
    if not usage_event_id or not runtime.ledger_bootstrap_fn():
        return None
    conn = None
    try:
        conn = runtime.connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, embedding_model, embedding_calls,
                       embedding_input_tokens, actual_cost_usd, ingest_credits,
                       pricing_version, metering_version, usage_json,
                       request_key, company_id, bubble_document_id, month_key,
                       attempt_count, updated_at
                FROM public.mm_ingest_usage_events
                WHERE usage_event_id=%s
                LIMIT 1;
                """,
                (usage_event_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "usage_event_id": usage_event_id,
            "status": str(row[0] or ""),
            "embedding_model": str(row[1] or ""),
            "embedding_calls": int(row[2] or 0),
            "embedding_input_tokens": int(row[3] or 0),
            "actual_cost_usd": float(row[4] or 0),
            "ingest_credits": float(row[5] or 0),
            "pricing_version": str(row[6] or ""),
            "metering_version": str(row[7] or ""),
            "usage_json": row[8] if isinstance(row[8], dict) else {},
            "request_key": str(row[9] or ""),
            "company_id": str(row[10] or ""),
            "bubble_document_id": str(row[11] or ""),
            "month_key": str(row[12] or ""),
            "attempt_count": int(row[13] or 0),
            "updated_at": row[14],
        }
    except Exception as exc:
        runtime.log_fn("INGEST_LEDGER_SNAPSHOT_FAIL_OPEN", str(exc)[:700])
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def claim_event(
    usage_event_id: str,
    *,
    runtime: IngestMeteringRuntime,
) -> tuple[str, Optional[dict]]:
    if not usage_event_id or not runtime.ledger_bootstrap_fn():
        return "unavailable", None
    conn = None
    try:
        conn = runtime.connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.mm_ingest_usage_events
                SET status='processing',
                    attempt_count=attempt_count + 1,
                    updated_at=NOW(),
                    last_error=''
                WHERE usage_event_id=%s
                  AND status <> 'completed'
                  AND (
                      status <> 'processing'
                      OR updated_at < NOW() - (%s * INTERVAL '1 second')
                  )
                RETURNING usage_event_id;
                """,
                (usage_event_id, int(runtime.processing_stale_seconds)),
            )
            claimed = cur.fetchone()
        conn.commit()
        if claimed:
            return "claimed", runtime.event_snapshot_fn(usage_event_id)
        snap = runtime.event_snapshot_fn(usage_event_id)
        if snap and snap.get("status") == "completed":
            return "completed", snap
        if snap and snap.get("status") == "processing":
            return "processing", snap
        return "unavailable", snap
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        runtime.log_fn("INGEST_LEDGER_CLAIM_FAIL_OPEN", str(exc)[:700])
        return "unavailable", None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def initialize_meter_state(
    meter: Any,
    *,
    usage_event_id: str,
    request_key: str,
    company_id: str,
    bubble_document_id: str,
    month_key: str,
) -> None:
    """Initialize the historical mutable meter fields on a caller-owned object."""

    meter.usage_event_id = usage_event_id
    meter.request_key = request_key
    meter.company_id = company_id
    meter.bubble_document_id = bubble_document_id
    meter.month_key = month_key
    meter.embedding_calls = 0
    meter.embedding_input_tokens = 0
    meter.embedding_cost_usd = Decimal("0")
    meter.provider_usage_calls = 0
    meter.fallback_usage_calls = 0


def record_meter_embedding(
    meter: Any,
    *,
    input_tokens: int,
    usage_source: str,
    runtime: IngestMeteringRuntime,
) -> None:
    tokens = max(0, int(input_tokens or 0))
    meter.embedding_calls += 1
    meter.embedding_input_tokens += tokens
    if usage_source == "provider_usage":
        meter.provider_usage_calls += 1
    else:
        meter.fallback_usage_calls += 1
    meter.embedding_cost_usd += (
        Decimal(tokens) * runtime.embed_input_price_usd_per_million
    ) / Decimal("1000000")


def meter_usage_dict(
    meter: Any,
    *,
    runtime: IngestMeteringRuntime,
) -> dict:
    cost = meter.embedding_cost_usd.quantize(
        Decimal("0.0000000001"), rounding=ROUND_HALF_UP
    )
    credits = runtime.credits_for_cost_fn(cost)
    return {
        "embedding_model": runtime.embed_model,
        "embedding_calls": meter.embedding_calls,
        "embedding_input_tokens": meter.embedding_input_tokens,
        "provider_usage_calls": meter.provider_usage_calls,
        "fallback_usage_calls": meter.fallback_usage_calls,
        "embedding_cost_usd": float(cost),
        "ingest_credits": float(credits),
    }


def meter_status(meter: Any) -> str:
    if meter.embedding_calls <= 0:
        return "measured_zero_cost"
    if meter.fallback_usage_calls > 0:
        return "estimated_partial"
    return "measured"


class IngestUsageMeter:
    """Per-request embedding meter with late-bound production configuration."""

    def __init__(
        self,
        *,
        usage_event_id: str,
        request_key: str,
        company_id: str,
        bubble_document_id: str,
        month_key: str,
        runtime_factory: Callable[[], IngestMeteringRuntime],
    ):
        initialize_meter_state(
            self,
            usage_event_id=usage_event_id,
            request_key=request_key,
            company_id=company_id,
            bubble_document_id=bubble_document_id,
            month_key=month_key,
        )
        self._runtime_factory = runtime_factory

    def record_embedding(self, *, input_tokens: int, usage_source: str) -> None:
        return record_meter_embedding(
            self,
            input_tokens=input_tokens,
            usage_source=usage_source,
            runtime=self._runtime_factory(),
        )

    def usage_dict(self) -> dict:
        return meter_usage_dict(self, runtime=self._runtime_factory())

    def metering_status(self) -> str:
        return meter_status(self)


def current_ingest_meter(
    *,
    runtime: IngestMeteringRuntime,
    meter_type: type,
) -> Optional[IngestUsageMeter]:
    context = runtime.state_globals["_INGEST_METER_CTX"]
    value = context.get()
    return value if isinstance(value, meter_type) else None


def finalize_event(
    meter: IngestUsageMeter,
    *,
    status: str,
    error_text: str = "",
    runtime: IngestMeteringRuntime,
) -> Optional[dict]:
    usage = meter.usage_dict()
    if not runtime.ledger_bootstrap_fn():
        return None

    attempt_cost = ingest_decimal(usage.get("embedding_cost_usd"))
    attempt_credits = ingest_decimal(usage.get("ingest_credits"))
    conn = None
    try:
        conn = runtime.connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.mm_ingest_usage_events
                SET status=%s,
                    embedding_model=%s,
                    embedding_calls=embedding_calls + %s,
                    embedding_input_tokens=embedding_input_tokens + %s,
                    actual_cost_usd=actual_cost_usd + %s,
                    ingest_credits=ingest_credits + %s,
                    pricing_version=%s,
                    metering_version=%s,
                    usage_json=%s::jsonb,
                    last_error=%s,
                    updated_at=NOW(),
                    completed_at=CASE WHEN %s='completed' THEN NOW() ELSE completed_at END
                WHERE usage_event_id=%s;
                """,
                (
                    status,
                    runtime.embed_model,
                    int(usage.get("embedding_calls") or 0),
                    int(usage.get("embedding_input_tokens") or 0),
                    str(attempt_cost),
                    str(attempt_credits),
                    runtime.pricing_version,
                    runtime.metering_version,
                    runtime.json_dumps_fn(usage, ensure_ascii=False),
                    str(error_text or "")[:1800],
                    status,
                    meter.usage_event_id,
                ),
            )
        conn.commit()
        return runtime.event_snapshot_fn(meter.usage_event_id)
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        runtime.log_fn("INGEST_LEDGER_FINALIZE_FAIL_OPEN", str(exc)[:700])
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def month_usage(
    company_id: str,
    month_key: str,
    *,
    runtime: IngestMeteringRuntime,
) -> dict:
    month_key = runtime.normalize_month_key_fn(month_key)
    if not runtime.ledger_bootstrap_fn():
        return {
            "company_id": company_id,
            "month_key": month_key,
            "ingest_credits_used_month": 0.0,
            "embedding_input_tokens_total": 0,
            "ledger_available": False,
            "ledger_error": str(
                runtime.state_globals.get("_INGEST_LEDGER_ERROR") or ""
            ),
        }
    conn = None
    try:
        conn = runtime.connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(ingest_credits), 0),
                       COALESCE(SUM(embedding_input_tokens), 0),
                       MAX(updated_at)
                FROM public.mm_ingest_usage_events
                WHERE company_id=%s AND month_key=%s;
                """,
                (company_id, month_key),
            )
            row = cur.fetchone() or (0, 0, None)
        return {
            "company_id": company_id,
            "month_key": month_key,
            "ingest_credits_used_month": float(row[0] or 0),
            "embedding_input_tokens_total": int(row[1] or 0),
            "updated_at": row[2].isoformat() if row[2] is not None else None,
            "ledger_available": True,
            "ledger_error": "",
        }
    except Exception as exc:
        runtime.log_fn("INGEST_LEDGER_MONTH_FAIL_OPEN", str(exc)[:700])
        return {
            "company_id": company_id,
            "month_key": month_key,
            "ingest_credits_used_month": 0.0,
            "embedding_input_tokens_total": 0,
            "ledger_available": False,
            "ledger_error": str(exc)[:700],
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def public_fields(
    *,
    meter: Optional[IngestUsageMeter],
    event_snapshot: Optional[dict],
    month_usage_value: Optional[dict],
    status_override: str = "",
    runtime: IngestMeteringRuntime,
) -> dict:
    snap = event_snapshot or {}
    usage = dict(snap.get("usage_json") or {})
    if meter is not None and not usage:
        usage = meter.usage_dict()

    actual_credits = float(
        snap.get("ingest_credits")
        if snap.get("ingest_credits") is not None
        else usage.get("ingest_credits") or 0
    )
    status = status_override or (
        meter.metering_status()
        if meter is not None
        else str(snap.get("status") or "unknown")
    )
    return {
        "request_key": str(
            snap.get("request_key") or (meter.request_key if meter else "")
        ),
        "ingest_request_key": str(
            snap.get("request_key") or (meter.request_key if meter else "")
        ),
        "ingest_usage_event_id": str(
            (meter.usage_event_id if meter else "")
            or snap.get("usage_event_id")
            or ""
        ),
        "ingest_month_key": str(
            snap.get("month_key") or (meter.month_key if meter else "")
        ),
        "ingest_credits_actual": actual_credits,
        "ingest_credits_used_month": float(
            (month_usage_value or {}).get("ingest_credits_used_month") or 0
        ),
        "ingest_pricing_version": str(
            snap.get("pricing_version") or runtime.pricing_version
        ),
        "ingest_metering_status": status,
        "ingest_metering_version": runtime.metering_version,
        "ingest_usage": usage,
    }


def meter_index_document(
    func: Callable[..., Any],
    *,
    runtime_factory: Callable[[], IngestMeteringRuntime],
    meter_cls: type[IngestUsageMeter],
) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapped(payload, x_ai_internal_secret=None):
        runtime = runtime_factory()
        if not bool(payload.ingest_metering_enabled):
            return func(payload, x_ai_internal_secret)

        if not runtime.ai_internal_secret or (
            x_ai_internal_secret or ""
        ).strip() != runtime.ai_internal_secret:
            return func(payload, x_ai_internal_secret)

        company_id = str(payload.company_id or "").strip()
        bubble_document_id = str(payload.bubble_document_id or "").strip()
        month_key = runtime.normalize_month_key_fn(payload.ingest_month_key)
        request_key = str(
            payload.ingest_request_key or payload.trace_id or bubble_document_id
        ).strip()
        usage_event_id = str(payload.ingest_usage_event_id or "").strip() or (
            runtime.build_usage_event_id_fn(company_id, month_key, request_key)
        )

        runtime.prepare_event_fn(
            usage_event_id=usage_event_id,
            request_key=request_key,
            company_id=company_id,
            bubble_document_id=bubble_document_id,
            month_key=month_key,
        )
        claim_status, existing = runtime.claim_event_fn(usage_event_id)

        if claim_status == "completed":
            month_usage_value = runtime.month_usage_fn(company_id, month_key)
            return {
                "ok": True,
                "status": "indexed",
                "company_id": company_id,
                "machine_id": str(payload.machine_id or ""),
                "bubble_document_id": bubble_document_id,
                "trace_id": payload.trace_id,
                "deduplicated": True,
                **runtime.public_fields_fn(
                    meter=None,
                    event_snapshot=existing,
                    month_usage=month_usage_value,
                    status_override="measured_deduplicated",
                ),
            }

        if claim_status == "processing":
            month_usage_value = runtime.month_usage_fn(company_id, month_key)
            return {
                "ok": True,
                "status": "processing",
                "company_id": company_id,
                "machine_id": str(payload.machine_id or ""),
                "bubble_document_id": bubble_document_id,
                "trace_id": payload.trace_id,
                "deduplicated": True,
                **runtime.public_fields_fn(
                    meter=None,
                    event_snapshot=existing,
                    month_usage=month_usage_value,
                    status_override="already_processing",
                ),
            }

        meter = meter_cls(
            usage_event_id=usage_event_id,
            request_key=request_key,
            company_id=company_id,
            bubble_document_id=bubble_document_id,
            month_key=month_key,
        )
        context = runtime.state_globals["_INGEST_METER_CTX"]
        context_token = context.set(meter)

        try:
            result = func(payload, x_ai_internal_secret)
        except Exception as exc:
            context.reset(context_token)
            snapshot = runtime.finalize_event_fn(
                meter, status="error", error_text=str(exc)
            )
            month_usage_value = runtime.month_usage_fn(company_id, month_key)
            runtime.log_fn(
                "INGEST_METER_FINAL_ERROR",
                runtime.json_dumps_fn(
                    runtime.public_fields_fn(
                        meter=meter,
                        event_snapshot=snapshot,
                        month_usage=month_usage_value,
                    ),
                    ensure_ascii=False,
                ),
            )
            raise

        context.reset(context_token)
        final_status = (
            "completed"
            if isinstance(result, dict) and result.get("ok") is True
            else "error"
        )
        snapshot = runtime.finalize_event_fn(
            meter,
            status=final_status,
            error_text=(
                ""
                if final_status == "completed"
                else str((result or {}).get("error") or "index failed")
            ),
        )
        month_usage_value = runtime.month_usage_fn(company_id, month_key)
        if isinstance(result, dict):
            result.update(
                runtime.public_fields_fn(
                    meter=meter,
                    event_snapshot=snapshot,
                    month_usage=month_usage_value,
                )
            )
        runtime.log_fn(
            "INGEST_METER_FINAL",
            runtime.json_dumps_fn(
                runtime.public_fields_fn(
                    meter=meter,
                    event_snapshot=snapshot,
                    month_usage=month_usage_value,
                ),
                ensure_ascii=False,
            ),
        )
        return result

    return wrapped


def usage_month_endpoint(
    payload: Any,
    x_ai_internal_secret: Optional[str],
    *,
    runtime: IngestMeteringRuntime,
    http_exception_cls: type[Exception],
) -> dict:
    if not runtime.ai_internal_secret:
        raise http_exception_cls(status_code=500, detail="AI_INTERNAL_SECRET missing")
    if (x_ai_internal_secret or "").strip() != runtime.ai_internal_secret:
        raise http_exception_cls(status_code=401, detail="Unauthorized")

    company_id = str(payload.company_id or "").strip()
    if not company_id:
        raise http_exception_cls(status_code=400, detail="Missing company_id")

    usage = runtime.month_usage_fn(
        company_id=company_id,
        month_key=runtime.normalize_month_key_fn(payload.month_key),
    )
    return {
        "ok": True,
        "status": "ok",
        **usage,
        "ingest_pricing_version": runtime.pricing_version,
        "ingest_metering_version": runtime.metering_version,
    }
