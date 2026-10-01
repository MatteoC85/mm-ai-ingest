"""PostgreSQL is the sole authoritative interactive quota writer.

Every mutation obtains the same company owner row lock in a short transaction.
Network/provider calls NEVER occur inside a DB transaction. No automatic schema
creation, retries, stale-request refund or client-supplied usage is permitted.
"""
from __future__ import annotations
from contextlib import contextmanager
from decimal import Decimal
from typing import Callable
from .contracts import Context, Settlement, UsageError, VERSION


class Ledger:
    def __init__(self, connect: Callable):
        self.connect = connect

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout = '2000ms'")
                cur.execute("SET LOCAL statement_timeout = '4000ms'")
                yield cur
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _lock(cur, company: str):
        cur.execute('INSERT INTO public.mm_ai_usage_owner_v6(company_id) VALUES (%s) ON CONFLICT DO NOTHING', (company,))
        cur.execute('SELECT company_id FROM public.mm_ai_usage_owner_v6 WHERE company_id=%s FOR UPDATE', (company,))
        if not cur.fetchone(): raise UsageError('USAGE_OWNER_MISSING')

    def health(self):
        with self.transaction() as cur:
            # SELECT-only access is insufficient for admission and settlement.
            # Test each privilege separately: PostgreSQL treats a comma-separated
            # privilege list as ANY, whereas this runtime needs ALL of these.
            for table, privileges in (
                    ('mm_ai_usage_schema_v6', ('SELECT',)),
                    ('mm_ai_usage_owner_v6', ('SELECT', 'INSERT', 'UPDATE')),
                    ('mm_ai_usage_bucket_v6', ('SELECT', 'INSERT', 'UPDATE')),
                    ('mm_ai_usage_request_v6', ('SELECT', 'INSERT', 'UPDATE'))):
                for privilege in privileges:
                    cur.execute('SELECT has_table_privilege(to_regclass(%s), %s)',
                        ('public.' + table, privilege))
                    permission = cur.fetchone()
                    if not permission or permission[0] is not True:
                        raise UsageError('USAGE_SCHEMA_PRIVILEGE_INVALID')
            cur.execute('SELECT version FROM public.mm_ai_usage_schema_v6 WHERE singleton=true')
            row = cur.fetchone()
            if not row or row[0] != VERSION: raise UsageError('USAGE_SCHEMA_INVALID')
            # A marker alone is insufficient: check all columns touched at runtime.
            for table, columns in (
                    ('mm_ai_usage_owner_v6', ['company_id']),
                    ('mm_ai_usage_bucket_v6', ['company_id','period_kind','period_key']),
                    ('mm_ai_usage_request_v6', ['company_id','request_id'])):
                cur.execute('''SELECT array_agg(a.attname ORDER BY k.ord)
                    FROM pg_constraint c
                    CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY k(num,ord)
                    JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.num
                    WHERE c.conrelid=to_regclass(%s) AND c.contype='p' GROUP BY c.oid''',
                    ('public.'+table,))
                keys=cur.fetchall()
                if len(keys)!=1 or list(keys[0][0])!=columns:
                    raise UsageError('USAGE_SCHEMA_PRIMARY_KEY_INVALID')
            cur.execute('SELECT company_id FROM public.mm_ai_usage_owner_v6 LIMIT 0')
            cur.execute('SELECT company_id,period_kind,period_key,opening_count,admitted_count FROM public.mm_ai_usage_bucket_v6 LIMIT 0')
            cur.execute('SELECT company_id,request_id,actor_id,operation,payload_sha256,day_key,month_key,state,counted,cap_usd,known_usd,uncertain_usd,result_code,created_at,completed_at FROM public.mm_ai_usage_request_v6 LIMIT 0')
        return {'version':VERSION, 'database':'postgresql', 'ready':True}

    def admit(self, context: Context, operation: str, fingerprint: str, cap: Decimal):
        with self.transaction() as cur:
            self._lock(cur, context.company_id)
            cur.execute('SELECT actor_id,operation,payload_sha256,state FROM public.mm_ai_usage_request_v6 WHERE company_id=%s AND request_id=%s',
                (context.company_id, context.request_id))
            prior = cur.fetchone()
            if prior:
                if tuple(prior[:3]) != (context.actor_id, operation, fingerprint):
                    raise UsageError('REQUEST_ID_CONFLICT',409)
                # Never replay a stored answer: current source authority must not
                # be bypassed. Retrying this ID cannot cause another provider call.
                raise UsageError('REQUEST_ALREADY_' + ('STARTED' if prior[3]=='started' else 'RECORDED'),409)
            counts = {}
            buckets = [('day',context.day_key,context.opening_day,context.daily_limit),
                       ('month',context.month_key,context.opening_month,context.monthly_limit)]
            for kind,key,opening,limit in buckets:
                cur.execute('INSERT INTO public.mm_ai_usage_bucket_v6(company_id,period_kind,period_key,opening_count) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                    (context.company_id,kind,key,opening))
                cur.execute('SELECT opening_count,admitted_count FROM public.mm_ai_usage_bucket_v6 WHERE company_id=%s AND period_kind=%s AND period_key=%s',
                    (context.company_id,kind,key))
                row=cur.fetchone()
                if not row: raise UsageError('USAGE_BUCKET_MISSING')
                used=int(row[0])+int(row[1])
                # Zero is the existing unlimited-plan convention. enabled=false
                # is rejected before this transaction, not interpreted as unlimited.
                if limit > 0 and used >= limit:
                    raise UsageError('AI_' + kind.upper() + '_QUOTA_EXCEEDED',429)
                counts[kind]=used+1
            cur.execute('INSERT INTO public.mm_ai_usage_request_v6(company_id,request_id,actor_id,operation,payload_sha256,day_key,month_key,state,cap_usd,uncertain_usd) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                (context.company_id,context.request_id,context.actor_id,operation,fingerprint,
                 context.day_key,context.month_key,'started',cap,cap))
            for kind,key,_,_ in buckets:
                cur.execute('UPDATE public.mm_ai_usage_bucket_v6 SET admitted_count=admitted_count+1 WHERE company_id=%s AND period_kind=%s AND period_key=%s',
                    (context.company_id,kind,key))
            return counts

    def checkpoint(self, context: Context, fingerprint: str, known: Decimal):
        """Before any cache write: save known usage, retaining unused liability.

        A checkpoint is not final settlement and cannot refund a quota slot. If
        publication later fails, both known spend and residual liability survive.
        """
        if not known.is_finite() or known<0:raise UsageError('USAGE_CHECKPOINT_INVALID')
        with self.transaction() as cur:
            self._lock(cur,context.company_id)
            cur.execute('SELECT actor_id,payload_sha256,state,cap_usd,known_usd FROM public.mm_ai_usage_request_v6 WHERE company_id=%s AND request_id=%s',
                (context.company_id,context.request_id))
            row=cur.fetchone()
            if not row or row[0]!=context.actor_id or row[1]!=fingerprint or row[2]!='started':
                raise UsageError('USAGE_CHECKPOINT_MISMATCH',409)
            if known<row[4]:raise UsageError('USAGE_CHECKPOINT_DECREASE',409)
            cur.execute('UPDATE public.mm_ai_usage_request_v6 SET known_usd=%s,uncertain_usd=%s,result_code=%s WHERE company_id=%s AND request_id=%s',
                (known,max(row[3]-known,Decimal(0)),'CACHE_CHECKPOINT',context.company_id,context.request_id))
        return True

    def complete(self, context: Context, fingerprint: str, result: Settlement):
        with self.transaction() as cur:
            self._lock(cur,context.company_id)
            cur.execute('SELECT actor_id,payload_sha256,state,counted,day_key,month_key,known_usd,uncertain_usd,result_code FROM public.mm_ai_usage_request_v6 WHERE company_id=%s AND request_id=%s',
                (context.company_id,context.request_id))
            row=cur.fetchone()
            if not row or row[0]!=context.actor_id or row[1]!=fingerprint:
                raise UsageError('USAGE_COMPLETION_MISMATCH',409)
            if row[2]!='started':
                expected=(result.state,not result.refund,result.known,result.uncertain,result.result_code)
                actual=(row[2],row[3],row[6],row[7],row[8])
                if actual!=expected: raise UsageError('USAGE_COMPLETION_CONFLICT',409)
                return False
            if result.known < row[6]:
                raise UsageError('USAGE_COMPLETION_DECREASE',409)
            if result.refund and row[3]:
                for kind,key in [('day',row[4]),('month',row[5])]:
                    cur.execute('UPDATE public.mm_ai_usage_bucket_v6 SET admitted_count=admitted_count-1 WHERE company_id=%s AND period_kind=%s AND period_key=%s AND admitted_count>0',
                        (context.company_id,kind,key))
                    if cur.rowcount!=1:raise UsageError('USAGE_REFUND_INVARIANT')
            cur.execute('UPDATE public.mm_ai_usage_request_v6 SET state=%s,counted=%s,known_usd=%s,uncertain_usd=%s,result_code=%s,completed_at=clock_timestamp() WHERE company_id=%s AND request_id=%s',
                (result.state,not result.refund,result.known,result.uncertain,result.result_code,context.company_id,context.request_id))
            return True

    def snapshot(self, context: Context):
        # Read-only; missing buckets are zero plus the trusted opening baseline.
        with self.transaction() as cur:
            cur.execute('''SELECT period_kind, opening_count+admitted_count
                FROM public.mm_ai_usage_bucket_v6 WHERE company_id=%s
                AND ((period_kind='day' AND period_key=%s) OR
                     (period_kind='month' AND period_key=%s))''',
                (context.company_id,context.day_key,context.month_key))
            # Both counters come from one PostgreSQL statement snapshot.
            values={'day':context.opening_day,'month':context.opening_month}
            values.update({kind:int(value) for kind,value in cur.fetchall()})
            return values


def production_ledger(environ):
    import psycopg2
    def connect():
        keys={'host':'MM_DB_HOST','dbname':'MM_DB_NAME','user':'MM_DB_USER','password':'MM_DB_PASSWORD'}
        args={k:environ.get(v,'postgres' if k=='dbname' else '') for k,v in keys.items()}
        if not all(args.values()):raise UsageError('USAGE_DATABASE_CONFIGURATION_INVALID')
        args.update(connect_timeout=3,options='-c statement_timeout=4000 -c lock_timeout=2000')
        return psycopg2.connect(**args)
    return Ledger(connect)
