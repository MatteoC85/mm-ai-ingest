"""Application authority contracts, independent of retrieval and model output.

``AI_INTERNAL_SECRET`` authenticates the service call; it is not sufficient to
prove that a Bubble application workflow already authorized the requested
Company/Machine/scope. ``ApplicationBoundary`` therefore verifies a separate
server-only application credential. Bubble must attach that credential only
from its backend bridge after checking ``Current User`` and the requested
context. The backend then re-validates Company/Machine/source ownership; it does
not call back into Bubble to re-read the user's role on every ASK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import math
from typing import Callable

from ..evidence.contracts import EvidenceContractError

AUTHORITY_VERSION = "application-authority-p6b4l-v1"


class AuthorityError(EvidenceContractError):
    """Public, non-sensitive reason. Never includes provider bodies/credentials."""
    def __init__(self, code: str, http_status: int = 503):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


def identifier(value: object, *, optional: bool = False) -> str | None:
    if optional and value in (None, ""):
        return None
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > 512 or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise AuthorityError("AUTHORITY_INVALID_IDENTIFIER", 400)
    return value


@dataclass(frozen=True, slots=True, repr=False)
class ApplicationGrant:
    """Trusted in-process proof that the server application boundary passed."""
    _issuer: object = field(repr=False, compare=False)


class ApplicationBoundary:
    """One configured server boundary; no HTTP, env mutation, or authority cache."""
    def __init__(self, *, secret: str, legacy_secret: str):
        # Require an independently provisioned key, not re-use of an exposed key.
        if (type(secret) is not str or len(secret) < 32 or len(secret) > 256
                or any(not 33 <= ord(c) <= 126 for c in secret)
                or not legacy_secret or secret == legacy_secret):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self._secret = secret
        self._issuer = object()

    def authenticate(self, *, supplied_secret: object) -> ApplicationGrant:
        if (type(supplied_secret) is not str or len(supplied_secret) > 256
                or not supplied_secret.isascii()
                or not hmac.compare_digest(self._secret, supplied_secret)):
            raise AuthorityError("AUTH_REQUIRED", 401)
        return ApplicationGrant(self._issuer)

    def require(self, grant: ApplicationGrant) -> None:
        if type(grant) is not ApplicationGrant or grant._issuer is not self._issuer:
            raise AuthorityError("AUTH_REQUIRED", 401)

    def __repr__(self) -> str:
        return "ApplicationBoundary(<server-only>)"


@dataclass(frozen=True, slots=True)
class AuthorityLimits:
    """Transport/accounting bounds, not a price estimate or relevance threshold."""
    max_http_calls: int
    max_response_bytes: int
    max_total_bytes: int
    max_records: int
    page_size: int
    timeout_seconds: float
    total_seconds: float

    def __post_init__(self):
        for n in ("max_http_calls", "max_response_bytes", "max_total_bytes", "max_records", "page_size"):
            if type(getattr(self, n)) is not int or getattr(self, n) < 1:
                raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        if self.page_size > 100 or self.max_response_bytes > self.max_total_bytes:
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        for n in ("timeout_seconds", "total_seconds"):
            v = getattr(self, n)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")


class AuthorityMeter:
    """One request's observed HTTP exposure. No retries or silent free timeouts."""
    def __init__(self, limits: AuthorityLimits, clock: Callable[[], float]):
        if type(limits) is not AuthorityLimits or not callable(clock):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.calls = self.bytes = self.failed_calls = 0
        self._in_flight = False

    def begin(self) -> float:
        remaining = self.limits.total_seconds - (self.clock() - self.started)
        if self._in_flight or self.calls >= self.limits.max_http_calls or remaining <= 0:
            raise AuthorityError("AUTHORITY_BUDGET_EXCEEDED")
        self.calls += 1
        self._in_flight = True
        return min(self.limits.timeout_seconds, remaining)

    def finish(self, size: int, *, failed: bool) -> None:
        self._in_flight = False
        self.bytes += size
        self.failed_calls += int(failed)
        if self.bytes > self.limits.max_total_bytes or self.clock() - self.started > self.limits.total_seconds:
            raise AuthorityError("AUTHORITY_BUDGET_EXCEEDED")

    def summary(self) -> dict:
        return dict(version=AUTHORITY_VERSION, http_calls=self.calls,
                    response_bytes=self.bytes, failed_http_calls=self.failed_calls,
                    elapsed_seconds=max(0.0, self.clock() - self.started),
                    grants_cached=False, prices_measured=False)
