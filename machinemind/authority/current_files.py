"""Current, user-session-protected Bubble file references for admitted sources.

Never return the admin-authenticated CDN redirect, persist a signed URL, append
an API credential or change Bubble privacy. A native same-origin file URL is
opened later by the user's browser; that browser's file permission remains a
separate gate. Other source kinds retain their already-admitted SQL app links.
"""
from __future__ import annotations
from urllib.parse import urlsplit
from ..evidence.contracts import SourceType, Provenance, LinkTarget, canonical_json
from ..retrieval.supplemental_evidence import (FileReferenceRead,
    FileReferenceObservation, build_file_reference_read, storage_key)
from .contracts import AuthorityError
from .policy import _row, _ref

VERSION = "bubble-current-file-reference-v1"


def native_reference(value, origin):
    if value is None or value == "":
        return None
    if type(value) is not str or len(value) > 32768 or value != value.strip():
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    # Validate the actual value. Do not strip query parameters from a stale URL.
    if any(ord(c) < 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID")
    try:
        u, base = urlsplit(value), urlsplit(origin)
        if (u.scheme != "https" or u.hostname != base.hostname or u.port not in (None,443)
                or u.username or u.password or u.query or u.fragment or not u.path.startswith("/")):
            raise ValueError()
    except (ValueError, TypeError):
        raise AuthorityError("AUTHORITY_FILE_REFERENCE_INVALID") from None
    return value


class CurrentFileReader:
    def __init__(self, authorized, legacy_runtime):
        self.authorized, self.legacy = authorized, legacy_runtime
        # Deployment origin is validated by BubbleConnection, never by payload.
        self.origin = "https://" + authorized.provider.directory._connection.allowed_host

    def __call__(self, *, scope, sources, current_allowed_sources, limits, allow_structured):
        from ..retrieval.document_readers import read_document_file_references
        a = self.authorized
        (a.check(a.payload) if a.admission is None else a.admission.check())
        if a.scope != scope:
            raise AuthorityError("AUTHORITY_FILE_SCOPE_CHANGED")
        docs = tuple(s for s in sources if s.source_type == SourceType.DOCUMENT)
        other = tuple(s for s in sources if s.source_type != SourceType.DOCUMENT)
        observations = []
        if other:
            old = read_document_file_references(scope=scope, sources=other,
                current_allowed_sources=current_allowed_sources, limits=limits,
                runtime=self.legacy, allow_structured=allow_structured)
            observations.extend(old.observations)
        spec = a.provider.schema.source(SourceType.DOCUMENT)
        for source in docs:
            raw = _row(a.provider.directory.get(spec.typename, source.source_id), source.source_id)
            if not a.provider._active(raw, spec):
                raise AuthorityError("AUTHORITY_FILE_REVOKED", 403)
            company = _ref(raw, spec.company_field)
            machine = _ref(raw, spec.machine_field, optional=True)
            if (company != scope.company_id or machine != source.scope.machine_id or
                    a.provider._identity(scope, spec, raw, machine) != source):
                raise AuthorityError("AUTHORITY_FILE_SCOPE_CHANGED", 403)
            # The source schema gives this field its meaning; it is not guessed
            # from title, extension, query text or the previous SQL URL.
            url = native_reference(raw.get(spec.file_field), self.origin)
            if url is not None and len(url) > limits.adapter.max_aux_chars:
                raise AuthorityError("AUTHORITY_FILE_REFERENCE_TOO_LARGE")
            key = storage_key(source)
            observations.append(FileReferenceObservation(source,
                Provenance(VERSION, canonical_json(("bubble",spec.typename,company,source.source_id,spec.file_field))),
                LinkTarget("bubble_sources", key),company,key,url))
        (a.check(a.payload) if a.admission is None else a.admission.check())
        obs = tuple(observations)
        return FileReferenceRead(scope,sources,obs,limits,
            len(canonical_json((sources,obs)).encode("utf-8")),int(bool(sources)))
