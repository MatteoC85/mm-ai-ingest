import io
import json
import time
import unittest
from urllib.error import HTTPError

from machinemind.authority.bubble_directory import BubbleConnection, BubbleDirectory
from machinemind.authority.contracts import AuthorityError, AuthorityLimits, AuthorityMeter


class _Response(io.BytesIO):
    def __init__(self, record_id):
        self.code = 200
        self.headers = {"Content-Type": "application/json"}
        super().__init__(json.dumps({"response": {"_id": record_id}}).encode())


class _InterruptedResponse(_Response):
    def __init__(self, status):
        super().__init__("record")
        self.code = status

    def read(self, _size=-1):
        raise TimeoutError("response body was interrupted")


class _Opener:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request.full_url, request.get_method(), timeout))
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result


def _http_error(status):
    return HTTPError("https://directory.invalid", status, "", {}, io.BytesIO(b"{}"))


class BubbleAuthorityTransientTests(unittest.TestCase):
    def directory(self, *responses, owned=True, max_calls=12):
        connection = BubbleConnection(
            "https://directory.invalid/api/1.1/obj", "directory.invalid", "test-token",
        )
        meter = AuthorityMeter(AuthorityLimits(max_calls, 1024, 8192, 16, 8, 2, 8), time.monotonic)
        opener = _Opener(*responses)
        return BubbleDirectory(
            connection=connection, meter=meter, opener=opener, request_owned_io=owned,
        ), opener

    def test_recovers_from_transient_response_and_accounts_for_both_requests(self):
        directory, opener = self.directory(_http_error(503), _Response("record"))
        self.assertEqual(directory.get("machine", "record"), {"_id": "record"})
        self.assertEqual(opener.requests[0][:2], opener.requests[1][:2])
        self.assertEqual(directory.meter.calls, 2)
        self.assertEqual(directory.meter.failed_calls, 1)

    def test_recovers_from_a_read_timeout(self):
        directory, _ = self.directory(TimeoutError("internal"), _Response("record"))
        self.assertEqual(directory.get("machine", "record"), {"_id": "record"})
        self.assertEqual(directory.meter.calls, 2)

    def test_another_timeout_never_dispatches_a_third_attempt(self):
        directory, opener = self.directory(TimeoutError(), TimeoutError(), _Response("record"))
        with self.assertRaisesRegex(AuthorityError, "AUTHORITY_PROVIDER_UNAVAILABLE"):
            directory.get("machine", "record")
        self.assertEqual(len(opener.requests), 2)

    def test_retry_quota_is_shared_by_all_reads_in_one_request(self):
        directory, opener = self.directory(
            _http_error(503), _Response("a"), TimeoutError(), _Response("b"),
            _http_error(502), _Response("c"),
        )
        self.assertEqual(directory.get("machine", "a")["_id"], "a")
        self.assertEqual(directory.get("machine", "b")["_id"], "b")
        with self.assertRaisesRegex(AuthorityError, "AUTHORITY_PROVIDER_UNAVAILABLE"):
            directory.get("machine", "c")
        self.assertEqual(len(opener.requests), 5)

    def test_failed_or_non_owned_read_does_not_gain_permission_to_retry(self):
        for status, owned in [(403, True), (429, True), (503, False)]:
            with self.subTest(status=status, owned=owned):
                directory, opener = self.directory(_http_error(status), _Response("record"), owned=owned)
                with self.assertRaisesRegex(AuthorityError, "AUTHORITY_PROVIDER_UNAVAILABLE"):
                    directory.get("machine", "record")
                self.assertEqual(len(opener.requests), 1)

    def test_retry_cannot_extend_call_budget(self):
        directory, opener = self.directory(_http_error(504), _Response("record"), max_calls=1)
        with self.assertRaisesRegex(AuthorityError, "AUTHORITY_BUDGET_EXCEEDED"):
            directory.get("machine", "record")
        self.assertEqual(len(opener.requests), 1)

    def test_interrupted_denial_body_cannot_become_an_authorized_retry(self):
        for status in (401, 403, 404, 429):
            with self.subTest(status=status):
                directory, opener = self.directory(_InterruptedResponse(status), _Response("record"))
                with self.assertRaisesRegex(AuthorityError, "AUTHORITY_PROVIDER_UNAVAILABLE"):
                    directory.get("machine", "record")
                self.assertEqual(len(opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
