"""Fault injection for the production ledger's read-only permission preflight."""
import unittest

from machinemind.accounting.contracts import UsageError, VERSION
from machinemind.accounting.ledger import Ledger


class Cursor:
    def __init__(self, denied=None, missing=None):
        self.denied = denied
        self.missing = missing
        self.queries = []
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        self.queries.append((statement, params))
        self.rows = []
        if 'has_table_privilege' in statement:
            allowed = params != self.denied
            self.rows = [(None if params[0] == self.missing else allowed,)]
        elif 'SELECT version' in statement:
            self.rows = [(VERSION,)]
        elif 'pg_constraint' in statement:
            keys = {
                'public.mm_ai_usage_owner_v6': ['company_id'],
                'public.mm_ai_usage_bucket_v6': ['company_id', 'period_kind', 'period_key'],
                'public.mm_ai_usage_request_v6': ['company_id', 'request_id'],
            }
            self.rows = [(keys[params[0]],)]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, **kwargs):
        self.cur = Cursor(**kwargs)
        self.committed = self.rolled_back = self.closed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class HealthTests(unittest.TestCase):
    def test_missing_runtime_permission_is_never_ready(self):
        for table in ('owner', 'bucket', 'request'):
            for privilege in ('SELECT', 'INSERT', 'UPDATE'):
                with self.subTest(table=table, privilege=privilege):
                    conn = Connection(denied=('public.mm_ai_usage_' + table + '_v6', privilege))
                    with self.assertRaisesRegex(UsageError, 'USAGE_SCHEMA_PRIVILEGE_INVALID'):
                        Ledger(lambda: conn).health()
                    self.assertTrue(conn.rolled_back and conn.closed)
                    self.assertFalse(conn.committed)

    def test_missing_schema_permission_or_table_is_never_ready(self):
        for kwargs in (
            {'denied': ('public.mm_ai_usage_schema_v6', 'SELECT')},
            {'missing': 'public.mm_ai_usage_request_v6'},
        ):
            with self.subTest(kwargs=kwargs):
                conn = Connection(**kwargs)
                with self.assertRaisesRegex(UsageError, 'USAGE_SCHEMA_PRIVILEGE_INVALID'):
                    Ledger(lambda: conn).health()
                self.assertTrue(conn.rolled_back and conn.closed)

    def test_healthy_schema_requires_all_permissions_without_writes(self):
        conn = Connection()
        result = Ledger(lambda: conn).health()
        self.assertEqual(result, {'version': VERSION, 'database': 'postgresql', 'ready': True})
        permissions = [params for sql, params in conn.cur.queries if 'has_table_privilege' in sql]
        self.assertEqual(len(permissions), 10)
        self.assertTrue(all(',' not in privilege for _, privilege in permissions))
        self.assertTrue(all(sql.lstrip().startswith(('SELECT', 'SET LOCAL'))
                            for sql, _ in conn.cur.queries))
        self.assertTrue(conn.committed and conn.closed)
        self.assertFalse(conn.rolled_back)


if __name__ == '__main__':
    unittest.main()
