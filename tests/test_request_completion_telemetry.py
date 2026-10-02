"""Offline transport regression: terminal status, cancellation and safe correlation."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.infrastructure import execution as execution
from machinemind.infrastructure.request_budget import (
    _RequestControl, _REQUEST_CONTROL_CTX, _V13RequestBudget,
)


def denied(*args, **kwargs):
    raise AssertionError("OFFLINE_NETWORK_DENIED")


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guards = contextlib.ExitStack()
        self.guards.enter_context(patch.object(socket, "create_connection", denied))
        self.guards.enter_context(patch.object(socket.socket, "connect", denied))
        self.output = io.StringIO()
        self.guards.enter_context(contextlib.redirect_stdout(self.output))
        self.payload = SimpleNamespace(query="PRIVATE_USER_QUERY", language="it")
        self.kwargs = dict(mode="ask", payload=self.payload,
            x_ai_internal_secret="PRIVATE_TRANSPORT_SECRET", hard_timeout_seconds=1,
            timeout_result_code="TIMEOUT", select_response_language=lambda *a, **k: "it",
            error_payload=lambda mode, exc: {"ok": False, "status": "error",
                "error": {"detail": "PRIVATE_EXCEPTION"}})

    def tearDown(self):
        self.guards.close()

    def record(self):
        records = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "AI_REQUEST_COMPLETED")
        self.assertRegex(records[0]["request_id"], r"^[a-f0-9]{32}$")
        self.assertNotIn("PRIVATE_", self.output.getvalue())
        return records[0]

    async def test_normal_response_keeps_envelope_and_correlates_worker_budget(self):
        expected = {"ok": True, "status": "needs_clarification", "answer": "PRIVATE_ANSWER"}
        ids = []
        def work(payload, secret):
            ids.extend([_REQUEST_CONTROL_CTX.get().request_id, _V13RequestBudget("ask").request_id])
            return expected
        result = await execution.json_with_hard_timeout(sync_func=work,
            root_cause_mode="root_cause", **self.kwargs)
        self.assertIs(result, expected)
        record = self.record()
        self.assertEqual(record["status"], "needs_clarification")
        self.assertEqual(ids, [record["request_id"]] * 2)
        self.assertFalse(record["hard_timeout"])

    async def test_stream_records_terminal_timeout_despite_http_200(self):
        controls = []
        loop = asyncio.get_running_loop()
        started, finished = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        clock = [0.0]
        def work(payload, secret):
            controls.append(_REQUEST_CONTROL_CTX.get())
            loop.call_soon_threadsafe(started.set)
            try:
                release.wait()
                return {"ok": True, "status": "answered"}
            finally:
                loop.call_soon_threadsafe(finished.set)
        # Patch only the guard's clock, never asyncio's real scheduling clock.
        # Start the worker before expiring the deadline: a queued task may
        # legitimately time out without ever calling work, so a 15ms sleep race
        # cannot establish the cancellation contract this test exercises.
        with patch.object(execution, "time_module", SimpleNamespace(monotonic=lambda: clock[0])):
            response = await execution.stream_json_response(sync_func=work,
                heartbeat_seconds=0.005, heartbeat_bytes=1, **self.kwargs)
            first_heartbeat = await anext(response.body_iterator)
            try:
                await asyncio.wait_for(started.wait(), timeout=5)  # deadlock watchdog only
                clock[0] = self.kwargs["hard_timeout_seconds"] + 1.0
                body = first_heartbeat + "".join([chunk async for chunk in response.body_iterator])
            finally:
                release.set()
                if controls:
                    await asyncio.wait_for(finished.wait(), timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(body)["status"], "timeout")
        record = self.record()
        self.assertEqual(record["status"], "timeout")
        self.assertTrue(record["hard_timeout"])
        self.assertTrue(controls[0].stopped.is_set())

    async def test_stream_preserves_safe_refusal_and_records_it(self):
        expected = {"ok": True, "status": "safety_refusal", "answer": "PRIVATE_ANSWER"}
        response = await execution.stream_json_response(sync_func=lambda *a: expected,
            heartbeat_seconds=0.01, heartbeat_bytes=1, **self.kwargs)
        body = "".join([chunk async for chunk in response.body_iterator])
        self.assertEqual(json.loads(body), expected)
        self.assertEqual(self.record()["status"], "safety_refusal")

    async def test_exception_does_not_log_detail(self):
        def work(*args):
            raise ValueError("PRIVATE_EXCEPTION")
        result = await execution.json_with_hard_timeout(sync_func=work,
            root_cause_mode="root_cause", **self.kwargs)
        self.assertEqual(result["status"], "error")
        record = self.record()
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["exception_type"], "ValueError")

    async def test_log_failure_does_not_change_result_or_cancellation(self):
        controls = []
        expected = {"ok": True, "status": "answered"}
        def work(*args):
            controls.append(_REQUEST_CONTROL_CTX.get())
            return expected
        with patch("builtins.print", side_effect=OSError("log unavailable")):
            result = await execution.json_with_hard_timeout(sync_func=work,
                root_cause_mode="root_cause", **self.kwargs)
        self.assertIs(result, expected)
        self.assertTrue(controls[0].stopped.is_set())

    async def test_stream_close_records_cancelled_without_payload(self):
        def work(*args):
            time.sleep(0.03)
            return {"status": "answered"}
        response = await execution.stream_json_response(sync_func=work,
            heartbeat_seconds=0.01, heartbeat_bytes=1, **self.kwargs)
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        self.assertEqual(self.record()["status"], "cancelled")

    async def test_untrusted_status_is_not_logged(self):
        execution._log_completion(control=_RequestControl(1), mode="PRIVATE_MODE",
            started=time.monotonic(), result={"status": "PRIVATE_STATUS"}, transport="json")
        record = self.record()
        self.assertEqual(record["status"], "other")
        self.assertEqual(record["mode"], "other")

    async def test_nested_controls_share_id_but_independent_requests_do_not(self):
        first = _RequestControl(1)
        second = _RequestControl(1)
        child = _RequestControl(0.5, parent=first)
        self.assertEqual(first.request_id, child.request_id)
        self.assertNotEqual(first.request_id, second.request_id)


if __name__ == "__main__":
    unittest.main()
