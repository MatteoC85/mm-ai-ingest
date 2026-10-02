"""Refresh legacy diagnostic file links through current scoped Bubble metadata.

The existing service endpoint owns authentication. This is not an ASK grant or
an end-user authorization migration: it verifies current tenant/machine source
ownership before replacing an already-selected document link. The native file
endpoint retains Bubble's browser-session privacy gate. No CDN URL is fetched,
de-signed, persisted, or returned through an administrator-authenticated redirect.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import time

from .bubble_directory import BubbleConnection, BubbleDirectory, _json_object
from .contracts import AuthorityError, AuthorityLimits, AuthorityMeter, identifier
from .current_files import native_reference
from .policy import _row, _ref
from ..evidence.contracts import SourceType

VERSION = "diagnostic-current-native-file-links-v1"
_STRUCTURED_PREFIXES = ("procedure:", "step:", "ps:", "md_photo:", "md_video:")


def _document_id(row):
    if type(row) is not dict:
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    key = row.get("bubble_document_id")
    if not key:
        return None
    key = identifier(key)
    if key.startswith(_STRUCTURED_PREFIXES):
        return None
    if ":" in key or str(row.get("source_type") or "document").lower() != "document":
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    return key


def _page(row):
    value = row.get("page_from")
    if type(value) is not int or value < 1:
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    return value


def current_document_file_map(*, company_id, machine_id, document_ids,
                              directory, schema, origin):
    """One bounded current lookup per selected document; caller owns transport."""
    company = identifier(company_id)
    machine = identifier(machine_id, optional=True)
    if (type(document_ids) is not tuple or len(document_ids) > 24
            or len(set(document_ids)) != len(document_ids)):
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    spec = schema.source(SourceType.DOCUMENT)
    # The table of indexed file URLs has no machine column. The current Machine
    # record and current document association supply these checks instead.
    _row(directory.get(schema.company_type, company), company)
    if machine:
        raw_machine = _row(directory.get(schema.machine_type, machine), machine)
        if _ref(raw_machine, schema.machine_company_field) != company:
            raise AuthorityError("AUTHORITY_FILE_SCOPE_CHANGED", 403)
    result = {}
    for uid in document_ids:
        uid = identifier(uid)
        raw = _row(directory.get(spec.typename, uid), uid)
        if spec.lifecycle == "flag":
            deleted = raw.get(spec.deleted_field)
            if deleted is True:
                raise AuthorityError("AUTHORITY_FILE_REVOKED", 403)
            if deleted is not False and not (deleted is None and schema.deleted_false_or_blank):
                raise AuthorityError("AUTHORITY_SCHEMA_INVALID")
        owner = _ref(raw, spec.company_field)
        source_machine = _ref(raw, spec.machine_field, optional=True)
        if owner != company or source_machine not in (None, machine):
            raise AuthorityError("AUTHORITY_FILE_SCOPE_CHANGED", 403)
        url = native_reference(raw.get(spec.file_field), origin)
        if url is None:
            raise AuthorityError("AUTHORITY_FILE_REFERENCE_MISSING", 403)
        result[uid] = url
    return result


def refresh_diagnostic_links(*, company_id, machine_id, citations, rg_links, env,
                             directory_factory=None):
    """Use the existing configured directory; OFF installations retain behavior.

    Only links belonging to a currently selected citation can be refreshed.
    Failure propagates before publication; stale SQL links are never a fallback.
    The optional factory is trusted offline-test injection, not request metadata.
    """
    from ..ask.application_authority import required, _config, _load_schema
    if not required(env):
        return rg_links
    if type(citations) is not list or type(rg_links) is not list:
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    selected = {}
    for row in citations:
        uid = _document_id(row)
        if uid is not None:
            selected[(uid, _page(row))] = row
    wanted = set()
    for link in rg_links:
        uid = _document_id(link)
        if uid is not None:
            if (uid, _page(link)) not in selected:
                raise AuthorityError("AUTHORITY_FILE_REFERENCE_UNBOUND", 403)
            wanted.add(uid)
    if not wanted:
        return rg_links
    if len(wanted) > 24:
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_TOO_LARGE")
    schema = _load_schema(_config(env, "MM_BUBBLE_AUTHORITY_SCHEMA_JSON"))
    try:
        limits = AuthorityLimits(**_json_object(_config(env, "MM_AUTHORITY_LIMITS_JSON").encode("utf-8")))
    except Exception:
        raise AuthorityError("AUTHORITY_CONFIGURATION_INVALID") from None
    # Final link refresh uses no more than one small document set, within the
    # already configured ceilings; it does not inherit a 60-second read allowance.
    limits = replace(limits, max_http_calls=min(limits.max_http_calls, len(wanted) + 2),
        timeout_seconds=min(limits.timeout_seconds, 3.0), total_seconds=min(limits.total_seconds, 8.0))
    connection = BubbleConnection(base_url=_config(env, "MM_BUBBLE_AUTHORITY_BASE_URL"),
        allowed_host=_config(env, "MM_BUBBLE_AUTHORITY_HOST"),
        token=_config(env, "MM_BUBBLE_AUTHORITY_TOKEN"))
    factory = directory_factory or BubbleDirectory
    directory = factory(connection=connection, meter=AuthorityMeter(limits, time.monotonic))
    try:
        file_map = current_document_file_map(company_id=company_id, machine_id=machine_id,
            document_ids=tuple(sorted(wanted)), directory=directory, schema=schema,
            origin="https://" + connection.allowed_host)
    finally:
        directory.close()
    result = []
    for link in rg_links:
        uid = _document_id(link)
        if uid is None:
            result.append(deepcopy(link))
        else:
            result.append({**deepcopy(link), "url": file_map[uid] + "#page=" + str(_page(link))})
    return result
