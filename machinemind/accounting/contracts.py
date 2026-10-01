"""Strict trusted-boundary input and conservative usage classification.

The header is NOT an end-user credential: it is accepted only with the separate
server-only usage secret. Bubble derives actor/company/plan inside a backend
custom event. No cost, refund, clock or accounting status is accepted as input.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import re
from zoneinfo import ZoneInfo

VERSION = 'p6-interactive-ledger-v1'
PATHS = {
    '/v1/ai/ask': ('ask', Decimal('0.25')),
    '/v1/ai/root-cause': ('root_cause', Decimal('0.40')),
    '/v1/ai/draft_ps': ('draft_ps', Decimal('0.40')),
    '/v1/ai/smart-diagnostic/start': ('smart_start', Decimal('0.35')),
    '/v1/ai/smart-diagnostic/answer': ('smart_answer', Decimal('0.25')),
    '/v1/ai/smart-diagnostic/finalize': ('smart_finalize', Decimal('0.25')),
}
SHADOW_PATHS = {'/v1/ai/ask-shadow', '/v1/ai/root-cause-shadow'}
IDENTIFIER = re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z')
CONTEXT_FIELDS = {'request_id','actor_id','company_id','enabled','daily_limit',
    'monthly_limit','opening_day','opening_month','day_key','month_key'}

class UsageError(Exception):
    def __init__(self, code: str, status: int = 503):
        super().__init__(code)
        self.code, self.status = code, status


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')


def strict_json(raw: bytes | str):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('duplicate key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def _count(value):
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise UsageError('USAGE_CONTEXT_INVALID', 400)
    return value


def _id(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise UsageError('USAGE_CONTEXT_INVALID', 400)
    return value


@dataclass(frozen=True)
class Context:
    request_id: str
    actor_id: str
    company_id: str
    enabled: bool
    daily_limit: int
    monthly_limit: int
    opening_day: int
    opening_month: int
    day_key: str
    month_key: str

    @classmethod
    def parse(cls, encoded: str, payload: dict, now: datetime, tz: str):
        try:
            if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4096:
                raise ValueError()
            data = strict_json(base64.b64decode(encoded, validate=True))
            if type(data) is not dict or set(data) != CONTEXT_FIELDS:
                raise ValueError()
            for key in ('request_id','actor_id','company_id'): _id(data[key])
            if len(data['request_id']) < 16 or type(data['enabled']) is not bool:
                raise ValueError()
            for key in ('daily_limit','monthly_limit','opening_day','opening_month'):
                _count(data[key])
            if type(data['day_key']) is not str or type(data['month_key']) is not str:
                raise ValueError()
            if type(payload) is not dict or payload.get('company_id') != data['company_id']:
                raise UsageError('USAGE_COMPANY_MISMATCH', 403)
            if now.tzinfo is None: raise ValueError()
            local = now.astimezone(ZoneInfo(tz))
            if data['day_key'] != local.strftime('%Y-%m-%d') or data['month_key'] != local.strftime('%Y-%m'):
                raise UsageError('USAGE_PERIOD_CHANGED', 409)
            if not data['enabled']: raise UsageError('PLAN_AI_DISABLED', 403)
            return cls(**data)
        except UsageError:
            raise
        except Exception:
            raise UsageError('USAGE_CONTEXT_INVALID', 400) from None

    def fingerprint(self, path: str, payload: dict) -> str:
        # Opening counters may change across a transport retry; they are not
        # request identity. Actor/company/path/body are always bound together.
        return hashlib.sha256(canonical({'actor':self.actor_id, 'company':self.company_id,
            'path':path, 'body':payload})).hexdigest()


@dataclass(frozen=True)
class Settlement:
    state: str
    known: Decimal
    uncertain: Decimal
    result_code: str
    refund: bool = False


def _money(value):
    if type(value) not in (str, int, float, Decimal) or isinstance(value, bool):
        raise ValueError()
    out = Decimal(str(value))
    if not out.is_finite() or out < 0 or out > Decimal('1000000'): raise ValueError()
    return out.quantize(Decimal('0.00000001'), rounding=ROUND_CEILING)


def settlement(body: dict | None, status: int, cap: Decimal) -> Settlement:
    """Classify only an in-process server response, never a client raw_response.

    Initial full liability survives process death. Missing/incomplete usage does
    not become zero; any known usage is retained even when it exceeds the cap.
    No payload, citation, session text or secret is persisted in this ledger.
    """
    body = body if type(body) is dict else {}
    meta = body.get('meta') if type(body.get('meta')) is dict else {}
    code = body.get('result_code') or body.get('error_code') or body.get('status') or 'UNCLASSIFIED'
    code = str(code).upper()
    if not re.fullmatch(r'[A-Z0-9_]{1,80}', code): code = 'UNCLASSIFIED'
    # A status code alone does not prove that no provider was dispatched. Do not
    # refund or infer zero expense from a late 4xx/5xx without execution proof.
    try:
        known = _money(meta['v13_estimated_cost_usd'])
    except (KeyError, ValueError, InvalidOperation):
        known = Decimal(0)
        return Settlement('uncertain', known, cap, code)
    try:
        committed = _money(meta['v13_committed_cost_usd'])
        uncertain = _money(meta['v13_uncertain_cost_usd'])
        # The service rounds independent float totals to 8 decimals. Permit only
        # that rounding tolerance, never a contradictory complete/zero claim.
        epsilon = Decimal('0.00000002')
        if committed + epsilon < known + uncertain:
            raise ValueError('inconsistent execution totals')
        reserved = (_money(meta['v13_reserved_cost_usd']) if 'v13_reserved_cost_usd' in meta
                    else max(Decimal(0), committed - known - uncertain))
        if 'v13_reserved_cost_usd' in meta and abs(committed-known-uncertain-reserved) > epsilon:
            raise ValueError('inconsistent reserved total')
        complete = meta.get('v13_accounting_complete') is True and reserved <= epsilon and uncertain == 0
        if complete:
            return Settlement('settled', known, Decimal(0), code)
        # Some timeout responses predate the thread's final ledger. Reserve the
        # entire unaccounted cap rather than trusting an incomplete snapshot.
        return Settlement('uncertain', known, max(uncertain + reserved, cap - known, Decimal(0)), code)
    except (KeyError, ValueError, InvalidOperation):
        return Settlement('uncertain', known, max(cap-known,Decimal(0)), code)
