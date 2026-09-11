"""Bounded read of current principal attributes through a Bubble API workflow.

This is a server directory lookup, NOT authentication of the browser. The caller
must first validate the distinct ApplicationBoundary and bind its asserted ID to
backend Bubble Current User. The workflow must be Admin only, return primitives,
and perform no writes. It does not require exposing User through the Data API.

The HTTPS host and environment are inherited from BubbleConnection. No request
may supply another endpoint, a role, a Company membership, or a redirect target.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request

from .contracts import AuthorityError, AuthorityMeter, identifier
from .bubble_directory import BubbleConnection, _NoRedirect, _json_object

PRINCIPAL_CONTRACT = "b4l_principal_v1"
PRINCIPAL_WORKFLOW = "ai_authority_principal_v1"


@dataclass(frozen=True, slots=True, repr=False)
class PrincipalSnapshot:
    """Attributes freshly observed from the configured server directory."""
    user_id: str
    company_id: str | None
    is_superadmin: bool

    def __post_init__(self):
        identifier(self.user_id)
        identifier(self.company_id, optional=True)
        if type(self.is_superadmin) is not bool:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")


def parse_principal_response(raw: bytes, expected_id: str) -> PrincipalSnapshot | None:
    """Accept the declared primitive-only workflow contract, never a User record.

    Missing/null/empty membership is explicitly supported for Superadmin. An
    ordinary principal without membership is subsequently denied by policy.
    Missing user ID is accepted ONLY in the explicit found=false branch.
    """
    identifier(expected_id)
    root = _json_object(raw)
    if set(root) != {"status", "response"} or root["status"] != "success":
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    r = root["response"]
    required = {"contract_version", "principal_found", "is_superadmin"}
    allowed = required | {"principal_id", "company_id"}
    if (type(r) is not dict or not required <= r.keys() or not r.keys() <= allowed
            or r["contract_version"] != PRINCIPAL_CONTRACT
            or type(r["principal_found"]) is not bool or type(r["is_superadmin"]) is not bool):
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    try:
        uid = identifier(r.get("principal_id"), optional=True)
        company = identifier(r.get("company_id"), optional=True)
    except AuthorityError:
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID") from None
    if not r["principal_found"]:
        if uid is not None or company is not None or r["is_superadmin"]:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
        return None
    if uid != expected_id:
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    return PrincipalSnapshot(uid, company, r["is_superadmin"])


class BubblePrincipalReader:
    """POST to the one configured read-only workflow; share the request meter.

    POST is required by this workflow's contract, but the workflow only reads.
    The administrative API token has broader powers than this class uses.
    There is no fallback to /obj/user, no retry, and no cached principal.
    """
    def __init__(self, *, connection: BubbleConnection, meter: AuthorityMeter, opener=None):
        if type(connection) is not BubbleConnection or type(meter) is not AuthorityMeter:
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self._connection, self.meter = connection, meter
        self._url = connection.base_url[:-3] + "wf/" + PRINCIPAL_WORKFLOW
        self._opener = opener if opener is not None else urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())

    def read(self, user_id: str) -> PrincipalSnapshot | None:
        identifier(user_id)
        body = json.dumps({"p_user_id": user_id}, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(self._url, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._connection.token,
            "Content-Type": "application/json", "Accept": "application/json",
            "Accept-Encoding": "identity", "Cache-Control": "no-store",
            "User-Agent": "MachineMind-B4l-Principal/1"})
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
                    # Even a 404 is a missing endpoint/configuration, not a
                    # not-found user. Only the explicit JSON branch means that.
                    raise AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
                if r.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
                value = parse_principal_response(raw, user_id)
                failed = False
                return value
        except AuthorityError:
            raise
        except Exception:
            # Never include HTTP bodies, credentials, IDs or exception text.
            raise AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE") from None
        finally:
            self.meter.finish(size, failed=failed)

    def __repr__(self):
        return "BubblePrincipalReader(<server-directory>)"
