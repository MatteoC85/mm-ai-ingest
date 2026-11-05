"""Offline native-connection boundary tests; no provider, socket or database I/O."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import database
from machinemind.retrieval import root_time_budget as timing


class Clock:
    now = 0.0


class Cursor:
    def __init__(self, connection, *args, **kwargs):
        self.connection = connection
        self.closed = False
        self.rows = []

    def execute(self, query, vars=None):
        self.connection.calls.append((query, vars, self))
        if query == "SET statement_timeout = %s":
            self.connection.clock.now += self.connection.control_delay
            if self.connection.control_error:
                raise self.connection.control_error
        else:
            self.connection.clock.now += self.connection.query_delay
            if self.connection.query_error:
                raise self.connection.query_error
            self.rows = [("original-result",)]
        return self.connection.execute_result

    def close(self):
        self.closed = True

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Connection:
    def __init__(self, clock):
        self.clock = clock
        self.cursor_factory = Cursor
        self.closed = False
        self.autocommit = False
        self.calls = []
        self.cursors = []
        self.query_delay = self.control_delay = 0.0
        self.query_error = self.control_error = None
        self.execute_result = object()
        self.commits = self.rollbacks = 0

    def cursor(self, *args, cursor_factory=None, **kwargs):
        cursor = (cursor_factory or self.cursor_factory)(self, *args, **kwargs)
        self.cursors.append(cursor)
        return cursor

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commits += 1
        else:
            self.rollbacks += 1


class RootTimeDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = Clock()
        self.conn = Connection(self.clock)
        self.connect_calls = []
        self.connect_delay = 0.0
        self.budget = SimpleNamespace(remaining=lambda: 100.0-self.clock.now,
                                      ensure_time=lambda seconds: None)
        self.window = timing.Window(10.0, self.budget, optional=True)
        self.stack.enter_context(patch.object(timing.native, "_monotonic", lambda: self.clock.now))
        self.current = self.stack.enter_context(patch.object(timing, "current_window", return_value=self.window))
        self.stack.enter_context(patch("socket.create_connection", side_effect=AssertionError("No network allowed")))
        self.stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("No network allowed")))

    def connect(self, **runtime):
        def fake_connect(**kwargs):
            self.connect_calls.append(kwargs)
            self.clock.now += self.connect_delay
            return self.conn
        return database.connect_database(db_host="fixture-host", db_name="fixture-db",
            db_user="fixture-user", db_password="fixture-password", runtime_globals=runtime,
            connect_fn=fake_connect, monotonic_fn=lambda: self.clock.now)

    def test_absent_window_returns_original_connection_kwargs_and_cursor(self):
        self.current.return_value = None
        result = self.connect()
        self.assertIs(result, self.conn)
        self.assertIs(result.cursor_factory, Cursor)
        self.assertEqual(self.connect_calls, [{"host":"fixture-host", "dbname":"fixture-db",
            "user":"fixture-user", "password":"fixture-password"}])
        cursor = result.cursor()
        self.assertIs(cursor.execute("SELECT original", (1,)), self.conn.execute_result)
        self.assertEqual(len(self.conn.calls), 1)

    def test_producer_window_keeps_existing_native_timeout_behavior(self):
        self.current.return_value = timing.Window(10.0, self.budget, optional=False)
        result = self.connect(_v13_current_budget=lambda:self.budget,
                              V13_DB_CONNECT_TIMEOUT_SECONDS=4, V13_DB_STATEMENT_TIMEOUT_MS=3000)
        self.assertIs(result, self.conn)
        self.assertIs(result.cursor_factory, Cursor)
        self.assertEqual(self.connect_calls[0]["connect_timeout"], 4)
        self.assertEqual(self.connect_calls[0]["options"], "-c statement_timeout=3000")
        self.assertEqual(result.calls, [])

    def test_each_query_reclips_exact_window_and_preserves_result_and_arguments(self):
        result = self.connect()
        self.assertIs(result, self.conn)
        self.assertTrue(issubclass(result.cursor_factory, Cursor))
        self.conn.query_delay = 6.0
        params = ("value",)
        cursor = result.cursor()
        self.assertIs(cursor.execute("SELECT first WHERE key=%s", params), self.conn.execute_result)
        self.conn.query_delay = 0.0
        self.assertIs(cursor.execute("SELECT second"), self.conn.execute_result)
        calls = self.conn.calls
        self.assertEqual([row[:2] for row in calls], [
            ("SET statement_timeout = %s", (9000,)), ("SELECT first WHERE key=%s", params),
            ("SET statement_timeout = %s", (3950,)), ("SELECT second", None)])
        self.assertIs(calls[1][1], params)
        self.assertEqual(cursor.fetchall(), [("original-result",)])
        self.assertTrue(calls[0][2].closed and calls[2][2].closed)
        self.assertFalse(cursor.closed)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), (0, 0))

    def test_expiry_before_next_query_does_not_dispatch_sql(self):
        cursor = self.connect().cursor()
        self.clock.now = 10.0
        with self.assertRaises(timing.RootWindowExpired):
            cursor.execute("SELECT forbidden")
        self.assertEqual(self.conn.calls, [])

    def test_expiry_during_control_statement_prevents_data_query(self):
        cursor = self.connect().cursor()
        self.conn.control_delay = 10.0
        with self.assertRaises(timing.RootWindowExpired):
            cursor.execute("SELECT forbidden")
        self.assertEqual(len(self.conn.calls), 1)
        self.assertEqual(self.conn.calls[0][0], "SET statement_timeout = %s")
        self.assertTrue(self.conn.calls[0][2].closed)

    def test_connect_expiry_closes_connection_without_cursor_query(self):
        self.connect_delay = 10.0
        with self.assertRaises(timing.RootWindowExpired):
            self.connect()
        self.assertTrue(self.conn.closed)
        self.assertEqual(self.conn.calls, [])

    def test_slow_control_does_not_send_query_with_oversized_installed_timeout(self):
        cursor = self.connect().cursor()
        self.clock.now = 6.0
        self.conn.control_delay = 1.0
        with self.assertRaises(timing.RootWindowExpired):
            cursor.execute("SELECT forbidden")
        self.assertEqual(len(self.conn.calls), 1)
        self.assertEqual(self.conn.calls[0][:2], ("SET statement_timeout = %s", (3950,)))
        self.assertTrue(self.conn.calls[0][2].closed)

    def test_too_short_connect_window_does_not_connect(self):
        self.clock.now = 8.0
        with self.assertRaises(timing.RootWindowExpired):
            self.connect()
        self.assertEqual(self.connect_calls, [])

    def test_assurance_limit_is_not_widened_and_autocommit_is_unchanged(self):
        self.conn.autocommit = True
        context = SimpleNamespace(get=lambda:2.5)
        result = self.connect(_v13_current_budget=lambda:self.budget,
                              _V13_ASSURANCE_DEADLINE_CTX=context)
        self.assertEqual(self.connect_calls[0]["options"], "-c statement_timeout=2300")
        self.assertTrue(result.autocommit)
        result.cursor().execute("SELECT bounded")
        self.assertEqual(self.conn.calls[0][:2], ("SET statement_timeout = %s", (2300,)))
        self.assertEqual((self.conn.commits, self.conn.rollbacks), (0, 0))

    def test_bound_window_is_not_replaced_by_later_context(self):
        cursor = self.connect().cursor()
        self.current.return_value = timing.Window(100.0, self.budget, optional=True)
        self.clock.now = 10.0
        with self.assertRaises(timing.RootWindowExpired):
            cursor.execute("SELECT forbidden")
        self.assertEqual(self.conn.calls, [])

    def test_assurance_absolute_deadline_survives_across_multiple_queries(self):
        context = SimpleNamespace(get=lambda:2.5)
        cursor = self.connect(_V13_ASSURANCE_DEADLINE_CTX=context).cursor()
        self.conn.query_delay = 2.6
        cursor.execute("SELECT first")
        before = len(self.conn.calls)
        with self.assertRaises(timing.RootWindowExpired):
            cursor.execute("SELECT forbidden")
        self.assertEqual(len(self.conn.calls), before)
        self.assertLess(self.clock.now, self.window.deadline)

    def test_query_exception_identity_and_connection_context_are_preserved(self):
        self.conn.query_error = ValueError("original-query-error")
        conn = self.connect()
        with self.assertRaises(ValueError) as caught:
            with conn as same:
                self.assertIs(same, self.conn)
                same.cursor().execute("SELECT fails")
        self.assertIs(caught.exception, self.conn.query_error)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), (0, 1))
        self.assertTrue(self.conn.calls[0][2].closed)

    def test_control_failure_propagates_without_query_or_transaction_mutation(self):
        self.conn.control_error = RuntimeError("control-query-error")
        cursor = self.connect().cursor()
        with self.assertRaises(RuntimeError) as caught:
            cursor.execute("SELECT forbidden")
        self.assertIs(caught.exception, self.conn.control_error)
        self.assertEqual(len(self.conn.calls), 1)
        self.assertTrue(self.conn.calls[0][2].closed)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), (0, 0))

    def test_default_factory_remains_a_native_psycopg_cursor_subclass(self):
        from psycopg2.extensions import cursor as native_cursor
        self.conn.cursor_factory = None
        result = self.connect()
        self.assertIs(result, self.conn)
        self.assertTrue(issubclass(result.cursor_factory, native_cursor))
        self.assertIs(result.cursor_factory.__mro__[1], native_cursor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
