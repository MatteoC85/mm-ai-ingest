"""Read-only, bounded Bubble Data API transport. No retrieval/model dependency.

Base URL and schema are deployment configuration, never request parameters.
Only GET is implemented; redirects/proxies are disabled. Do not expose Data API
or broaden Privacy as an automatic response to an unavailable record/type.
Admin authentication needs exposed types and is not user authentication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .contracts import AuthorityError, AuthorityMeter, identifier


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_object(data: bytes) -> dict:
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate field")
            out[k] = v
        return out
    try:
        value = json.loads(data, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID") from None
    if type(value) is not dict:
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class BubbleConnection:
    """Exact HTTPS Bubble/custom-domain root including /api/1.1/obj."""
    base_url: str
    allowed_host: str
    token: str = field(repr=False)

    def __post_init__(self):
        try:
            p = urllib.parse.urlsplit(self.base_url)
            port = p.port
        except (TypeError, ValueError):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID") from None
        if (p.scheme != "https" or not self.allowed_host or p.hostname != self.allowed_host
                or self.allowed_host != self.allowed_host.lower()
                or p.username or p.password or port not in (None, 443)
                or p.query or p.fragment or p.path.endswith("/")
                or not re.fullmatch(r"(?:/version-[A-Za-z0-9_-]+)?/api/1\.1/obj", p.path)
                or type(self.token) is not str or not self.token
                or len(self.token) > 1024 or not self.token.isascii()
                or any(c.isspace() or ord(c) < 33 for c in self.token)):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")


class BubbleDirectory:
    def __init__(self, *, connection: BubbleConnection, meter: AuthorityMeter, opener=None):
        if type(connection) is not BubbleConnection or type(meter) is not AuthorityMeter:
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self._connection, self.meter = connection, meter
        # Injection is trusted deployment/test code, never an API parameter.
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())

    @staticmethod
    def _typename(typename: str) -> str:
        if type(typename) is not str or not re.fullmatch(r"[a-z0-9_]{1,128}", typename):
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        return typename

    def _get(self, path: str, params: dict | None = None, *, missing_record: bool = False) -> dict | None:
        url = self._connection.base_url + "/" + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, method="GET", headers={
            "Authorization": "Bearer " + self._connection.token,
            "Accept": "application/json", "Cache-Control": "no-store",
            "User-Agent": "MachineMind-B4l-Authority/1"})
        timeout = self.meter.begin()
        size, failed = 0, True
        try:
            try:
                response = self._opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            with response as r:
                status = r.code if hasattr(r, "code") else r.status
                raw = r.read(self.meter.limits.max_response_bytes + 1)
                size = len(raw)
                if size > self.meter.limits.max_response_bytes:
                    raise AuthorityError("AUTHORITY_RESPONSE_TOO_LARGE")
                if status != 200:
                    # 404 can also mean wrong type/environment. Only Bubble's
                    # explicit MISSING_DATA record status is treated as absent.
                    if status == 404 and missing_record:
                        error = _json_object(raw)
                        if (error.get("body") or {}).get("status") == "MISSING_DATA":
                            failed = False
                            return None
                    raise AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
                content_type = r.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/json":
                    raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
                value = _json_object(raw)
                body = value.get("response")
                if type(body) is not dict:
                    raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
                failed = False
                return body
        except AuthorityError:
            raise
        except Exception:
            raise AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE") from None
        finally:
            self.meter.finish(size, failed=failed)

    def get(self, typename: str, record_id: str) -> dict | None:
        typename, record_id = self._typename(typename), identifier(record_id)
        if record_id in {".", ".."} or any(c in record_id for c in "/\\?#%"):
            raise AuthorityError("AUTHORITY_INVALID_IDENTIFIER", 400)
        value = self._get(typename + "/" + urllib.parse.quote(record_id, safe=""), missing_record=True)
        if value is not None and value.get("_id") != record_id:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
        return value

    def search(self, typename: str, constraints: tuple[dict, ...]) -> tuple[dict, ...]:
        typename = self._typename(typename)
        if type(constraints) is not tuple or not constraints:
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        for c in constraints:
            if (type(c) is not dict or set(c) != {"key", "constraint_type", "value"}
                    or type(c["key"]) is not str or not c["key"]
                    or c["constraint_type"] not in {"equals", "in", "is_empty"}):
                raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        encoded = json.dumps(constraints, separators=(",", ":"), allow_nan=False)
        rows, seen, cursor = [], set(), 0
        while True:
            page = self._get(typename, {"constraints": encoded, "cursor": cursor,
                "limit": self.meter.limits.page_size, "sort_field": "Created Date", "descending": "false"})
            items = page.get("results")
            count, remaining, observed_cursor = page.get("count"), page.get("remaining"), page.get("cursor")
            if (type(items) is not list or type(count) is not int or type(remaining) is not int
                    or type(observed_cursor) is not int or observed_cursor != cursor
                    or count != len(items) or count > self.meter.limits.page_size
                    or remaining < 0 or (remaining and not count)):
                raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
            if len(rows) + count + remaining > self.meter.limits.max_records:
                raise AuthorityError("AUTHORITY_CATALOG_TOO_LARGE")
            for item in items:
                if type(item) is not dict:
                    raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
                uid = identifier(item.get("_id"))
                if uid in seen:
                    raise AuthorityError("AUTHORITY_CATALOG_UNSTABLE")
                seen.add(uid)
                rows.append(item)
            if not remaining:
                return tuple(rows)
            cursor += count

    def __repr__(self):
        return "BubbleDirectory(<read-only>)"
