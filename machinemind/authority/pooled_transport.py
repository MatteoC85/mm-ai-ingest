"""One ASK request's TLS connections, never its authority decisions.

No redirects, cookies, proxies, response cache or retry. Every call is an actual
GET of the configured Bubble origin. The application keeps validating the full
response, pagination, ownership and lifecycle exactly as before. The pool is
closed by RequestAdmission, including on failures. Standard Python CA/hostname
verification is retained. urllib3 is the transport already installed by requests.
"""
from __future__ import annotations

import math
import ssl
from threading import RLock
from urllib.parse import urlsplit

from .contracts import AuthorityError


class ReadOnlyPool:
    def __init__(self, base_url):
        self._base = base_url
        self._lock = RLock()
        self._pool = None
        self._closed = False

    def open(self, request, timeout):
        url = request.full_url
        parsed = urlsplit(url)
        if (request.get_method() != "GET" or request.data is not None
                or not url.startswith(self._base + "/")
                or parsed.scheme != "https" or parsed.username or parsed.password
                or parsed.fragment or parsed.port not in (None, 443)
                or type(timeout) not in (int, float)
                or not math.isfinite(timeout) or timeout <= 0):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        with self._lock:
            if self._closed:
                raise AuthorityError("AUTHORITY_REQUEST_EXPIRED")
            if self._pool is None:
                import urllib3
                context = ssl.create_default_context()
                context.set_alpn_protocols(["http/1.1"])
                self._pool = urllib3.PoolManager(num_pools=1, maxsize=6,
                    block=True, ssl_context=context)
            pool = self._pool
        # This is not requests.Session: no environment proxy/netrc/cookies are
        # consulted. A disconnected keep-alive fails; it is NEVER replayed here.
        headers = dict(request.header_items())
        headers["Accept-Encoding"] = "identity"
        headers["Cache-Control"] = "no-store"
        return pool.urlopen("GET", url, headers=headers, retries=False,
            redirect=False, preload_content=False, decode_content=False,
            timeout=timeout, pool_timeout=timeout)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._pool is not None:
                    self._pool.clear()
                    self._pool = None

    def __repr__(self):
        return "ReadOnlyPool(<request-owned>)"
