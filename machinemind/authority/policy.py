"""MachineMind's confirmed Company/Superadmin policy over fresh Bubble records.

The application directory, NOT an index row, candidate, receipt, URL, title,
model decision or cache entry, supplies grants. All users of a Company have its
sources; Superadmin may choose another Company. Step ownership is inherited
from its live Procedure relation. No User/Company record is ever modified.

Every current_sources call re-reads authority; there is no permission cache.
Remote reads are not an atomic cross-service snapshot. The caller brackets
acquisition/consumption with these checks; revocation cannot cancel a model call
already sent. Request-wide HTTP/time/size budgets are explicit and measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ApplicationBoundary, ApplicationPrincipal, AuthorityError, identifier
from .principal_reader import BubblePrincipalReader, PrincipalSnapshot
from ..evidence.contracts import SourceIdentity, SourceType, SourceScope, SourceFormat, ScopeLevel
from ..evidence.ask_input import ask_request_key
from ..retrieval.chunk_evidence import ChunkReadScope

_GENERAL = "__MM_COMPANY_GENERAL__"  # existing core.scope SQL sentinel; never stored on a source


@dataclass(frozen=True, slots=True)
class SourceSchema:
    source_type: SourceType
    typename: str
    company_field: str | None
    machine_field: str | None
    deleted_field: str | None
    parent_field: str | None = None
    # Explicit per-type schema: record_exists is NOT a fallback for missing flags.
    lifecycle: str = "flag"

    def __post_init__(self):
        if not isinstance(self.source_type, SourceType):
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        identifier(self.typename)
        if self.lifecycle == "flag":
            identifier(self.deleted_field)
        elif self.lifecycle == "record_exists":
            if self.deleted_field is not None:
                raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        else:
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        if self.source_type == SourceType.STEP:
            identifier(self.parent_field)
        else:
            identifier(self.company_field)
            identifier(self.machine_field)
            if self.parent_field is not None:
                raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        for n in ("company_field", "machine_field"):
            identifier(getattr(self, n), optional=True)


@dataclass(frozen=True, slots=True)
class BubbleSchema:
    user_type: str
    company_type: str
    machine_type: str
    user_company_field: str
    user_role_field: str
    superadmin_value: str
    machine_company_field: str
    sources: tuple[SourceSchema, ...]
    # This is an explicit deployment choice about unset booleans, not a fallback
    # after a failed request. False requires an actual JSON false on each source.
    deleted_false_or_blank: bool = False

    def __post_init__(self):
        for n in ("user_type", "company_type", "machine_type", "user_company_field", "user_role_field",
                  "superadmin_value", "machine_company_field"):
            identifier(getattr(self, n))
        if (type(self.sources) is not tuple or len(self.sources) != len(SourceType)
                or any(type(s) is not SourceSchema for s in self.sources)
                or {s.source_type for s in self.sources} != set(SourceType)
                or type(self.deleted_false_or_blank) is not bool):
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")

    def source(self, kind: SourceType) -> SourceSchema:
        return next(s for s in self.sources if s.source_type == kind)


def _row(value: Any, expected_id: str | None = None) -> dict:
    if value is None:
        raise AuthorityError("SCOPE_DENIED", 403)
    if type(value) is not dict:
        raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
    uid = identifier(value.get("_id"))
    if expected_id is not None and uid != expected_id:
        raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
    return value


def _ref(row: dict, field: str, *, optional: bool = False) -> str | None:
    if field not in row and not optional:
        raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
    try:
        return identifier(row.get(field), optional=optional)
    except AuthorityError:
        raise AuthorityError("AUTHORITY_SCHEMA_INVALID") from None


class BubbleAuthority:
    """Concrete application policy; directory offers only get/search reads."""
    def __init__(self, *, directory: Any, schema: BubbleSchema, boundary: ApplicationBoundary,
                 principal_reader: BubblePrincipalReader | None = None):
        if (type(schema) is not BubbleSchema or type(boundary) is not ApplicationBoundary
                or not callable(getattr(directory, "get", None)) or not callable(getattr(directory, "search", None))):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        if principal_reader is not None and (type(principal_reader) is not BubblePrincipalReader
                or principal_reader.meter is not directory.meter):
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self.directory, self.schema, self.boundary = directory, schema, boundary
        self.principal_reader = principal_reader

    def authorize_scope(self, principal: ApplicationPrincipal, scope: ChunkReadScope) -> tuple[str | None, str | None]:
        self.boundary.require(principal)
        if type(scope) is not ChunkReadScope:
            raise AuthorityError("AUTHORITY_SCOPE_INVALID", 400)
        s = self.schema
        if self.principal_reader is None:
            # Explicit compatibility construction for existing direct callers.
            # Production HTTP composition ALWAYS supplies the workflow reader;
            # a workflow error never falls back into this branch.
            user = _row(self.directory.get(s.user_type, principal.user_id), principal.user_id)
            company = _ref(user, s.user_company_field, optional=True)
            role = _ref(user, s.user_role_field, optional=True)
        else:
            snapshot = self.principal_reader.read(principal.user_id)
            if snapshot is None:
                raise AuthorityError("SCOPE_DENIED", 403)
            if type(snapshot) is not PrincipalSnapshot or snapshot.user_id != principal.user_id:
                raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
            company = snapshot.company_id
            # The workflow compares the real Option Set. No wire spelling or
            # case guess about a serialized Bubble role is used for permission.
            role = s.superadmin_value if snapshot.is_superadmin else None
        if role != s.superadmin_value and company != scope.company_id:
            raise AuthorityError("SCOPE_DENIED", 403)
        _row(self.directory.get(s.company_type, scope.company_id), scope.company_id)
        if scope.machine_id not in (None, _GENERAL):
            machine = _row(self.directory.get(s.machine_type, scope.machine_id), scope.machine_id)
            if _ref(machine, s.machine_company_field) != scope.company_id:
                raise AuthorityError("CONTEXT_MISMATCH", 403)
        elif scope.ai_scope == "machine_all":
            raise AuthorityError("SCOPE_DENIED", 403)
        return company, role

    def _active(self, row: dict, spec: SourceSchema) -> bool:
        if spec.lifecycle == "record_exists":
            # _row already validated that this authoritative record exists.
            # Parent/Company/Machine checks still apply; there is no index grant.
            return True
        value = row.get(spec.deleted_field)
        if type(value) is bool:
            return not value
        if value is None and self.schema.deleted_false_or_blank:
            return True
        raise AuthorityError("AUTHORITY_SCHEMA_INVALID")

    def _owned_rows(self, spec: SourceSchema, scope: ChunkReadScope) -> tuple[dict, ...]:
        # Independent authoritative catalog search. No retrieval IDs/data enter it.
        rows = self.directory.search(spec.typename, ({"key": spec.company_field,
            "constraint_type": "equals", "value": scope.company_id},))
        if type(rows) is not tuple:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
        for value in rows:
            row = _row(value)
            if _ref(row, spec.company_field) != scope.company_id:
                raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
        return rows

    @staticmethod
    def _identity(scope: ChunkReadScope, spec: SourceSchema, row: dict,
                  machine_id: str | None) -> SourceIdentity | None:
        uid = row["_id"]
        key = uid if spec.source_type == SourceType.DOCUMENT else spec.source_type.value + ":" + uid
        if machine_id == _GENERAL:
            # The SQL selector is not a real source association.
            raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        if not scope.permits(scope.company_id, machine_id, key):
            return None
        return SourceIdentity(SourceScope(scope.company_id,
            ScopeLevel.COMPANY if machine_id is None else ScopeLevel.MACHINE, machine_id),
            spec.source_type, uid,
            SourceFormat.UNKNOWN if spec.source_type == SourceType.DOCUMENT else SourceFormat.STRUCTURED)

    def current_sources(self, principal: ApplicationPrincipal, scope: ChunkReadScope) -> frozenset[SourceIdentity]:
        before = self.authorize_scope(principal, scope)
        result, parents = set(), {}
        seen = set()
        max_records = self.directory.meter.limits.max_records
        for kind in (SourceType.DOCUMENT, SourceType.PROCEDURE, SourceType.PROBLEM_SOLUTION,
                     SourceType.PHOTO, SourceType.VIDEO):
            spec = self.schema.source(kind)
            rows = self._owned_rows(spec, scope)
            for row in rows:
                key = kind, row["_id"]
                if key in seen or len(seen) >= max_records:
                    raise AuthorityError("AUTHORITY_CATALOG_UNSTABLE" if key in seen else "AUTHORITY_CATALOG_TOO_LARGE")
                seen.add(key)
                if not self._active(row, spec):
                    continue
                machine = _ref(row, spec.machine_field, optional=True)
                if kind == SourceType.PROCEDURE:
                    # Keep all live parents in this Company, even if only their
                    # child Step is explicitly selected by document_ids.
                    parents[row["_id"]] = machine
                source = self._identity(scope, spec, row, machine)
                if source is not None:
                    result.add(source)
        step = self.schema.source(SourceType.STEP)
        # Parent-based queries avoid assuming a duplicated Step.company field.
        parent_ids = tuple(parents)
        for start in range(0, len(parent_ids), 50):
            batch = parent_ids[start:start + 50]
            rows = self.directory.search(step.typename, ({"key": step.parent_field,
                "constraint_type": "in", "value": list(batch)},))
            if type(rows) is not tuple:
                raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
            for raw in rows:
                row = _row(raw)
                key = SourceType.STEP, row["_id"]
                if key in seen or len(seen) >= max_records:
                    raise AuthorityError("AUTHORITY_CATALOG_UNSTABLE" if key in seen else "AUTHORITY_CATALOG_TOO_LARGE")
                seen.add(key)
                parent = _ref(row, step.parent_field)
                if parent not in batch:
                    raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
                if not self._active(row, step):
                    continue
                machine = parents[parent]
                if step.company_field is not None and _ref(row, step.company_field) != scope.company_id:
                    raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
                if step.machine_field is not None and _ref(row, step.machine_field, optional=True) != machine:
                    raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
                source = self._identity(scope, step, row, machine)
                if source is not None:
                    result.add(source)
        # A source cannot keep its former tenant permission after its Machine
        # moves Company or is removed. This catalog is independent of retrieval.
        machine_rows = self.directory.search(self.schema.machine_type, ({
            "key": self.schema.machine_company_field, "constraint_type": "equals",
            "value": scope.company_id},))
        if type(machine_rows) is not tuple:
            raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
        machines = set()
        for raw in machine_rows:
            row = _row(raw)
            if _ref(row, self.schema.machine_company_field) != scope.company_id or row["_id"] in machines:
                raise AuthorityError("AUTHORITY_RESPONSE_INVALID")
            machines.add(row["_id"])
        result = {source for source in result if source.scope.machine_id is None
                  or source.scope.machine_id in machines}
        if self.authorize_scope(principal, scope) != before:
            raise AuthorityError("AUTHORITY_CHANGED", 403)
        return frozenset(result)


class RequestAuthority:
    """Bind the concrete provider to ONE request object and its resolved selectors.

    Close with the existing request/session owner. A caught authority fault is
    terminal; a retained callback, request clone or mutated selector is rejected.
    No second EvidenceSession, Core or global ContextVar is introduced.
    """
    def __init__(self, *, request: Any, scope: ChunkReadScope,
                 principal: ApplicationPrincipal, provider: BubbleAuthority):
        if type(provider) is not BubbleAuthority:
            raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID")
        self._request, self._scope, self._principal, self._provider = request, scope, principal, provider
        self._key = ask_request_key(request)
        self._active, self._fault = True, None
        self._check(request)
        provider.authorize_scope(principal, scope)

    def _check(self, request):
        if self._fault is not None:
            raise self._fault
        try:
            if not self._active or request is not self._request or ask_request_key(request) != self._key:
                raise AuthorityError("AUTHORITY_REQUEST_EXPIRED", 403)
            metadata = request.metadata
            expected = ChunkReadScope(request.company_id, request.machine_id, request.ai_scope,
                tuple(metadata.get("document_ids") or ()), metadata.get("bubble_document_id") or None)
            if expected != self._scope or request.requested_mode != "ask":
                raise AuthorityError("AUTHORITY_SCOPE_INVALID", 400)
        except Exception as exc:
            self._fault = exc if isinstance(exc, AuthorityError) else AuthorityError("AUTHORITY_SCOPE_INVALID", 400)
            raise self._fault from None

    def __call__(self, request) -> frozenset[SourceIdentity]:
        self._check(request)
        try:
            result = self._provider.current_sources(self._principal, self._scope)
            self._check(request)
            return result
        except Exception as exc:
            self._fault = exc if isinstance(exc, AuthorityError) else AuthorityError("AUTHORITY_PROVIDER_UNAVAILABLE")
            raise self._fault from None

    def close(self):
        self._active = False

    def __repr__(self):
        return "RequestAuthority(<request-owned>)"
