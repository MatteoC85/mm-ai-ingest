"""B4n: current-authority proof for response citations, links and cache reuse.

This module is a pure contract layer. It performs no retrieval, no Bubble/API
lookup, no SQL, no cache I/O and no presentation. Callers must supply the CURRENT
provider allowance for the request. Legacy/cached citation identifiers are never
accepted as authorization by themselves.

A stored response-source manifest is audit/fingerprint metadata only. It is never
a credential: cache reuse must recompute the manifest from current authority and
compare it with the stored manifest before reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .contracts import EvidenceContractError, SourceIdentity, SourceType, canonical_json
from ..retrieval.supplemental_evidence import storage_key

RESPONSE_SOURCE_AUTHORITY_VERSION = "response-source-authority-p6b4n-v1"
MANIFEST_META_KEY = "response_source_manifest"


class ResponseSourceAuthorityError(EvidenceContractError):
    """Response/cache source proof is missing, ambiguous, stale or inconsistent."""


_PREFIX_ALIASES: dict[str, SourceType] = {
    "procedure": SourceType.PROCEDURE,
    "step": SourceType.STEP,
    "ps": SourceType.PROBLEM_SOLUTION,
    "problemsolution": SourceType.PROBLEM_SOLUTION,
    "problem_solution": SourceType.PROBLEM_SOLUTION,
    "problem-solution": SourceType.PROBLEM_SOLUTION,
    "md_photo": SourceType.PHOTO,
    "machine_detail_photo": SourceType.PHOTO,
    "photo_machine_detail": SourceType.PHOTO,
    "md_video": SourceType.VIDEO,
    "machine_detail_video": SourceType.VIDEO,
    "video_machine_detail": SourceType.VIDEO,
}

_TYPE_ALIASES: dict[str, SourceType] = {
    **_PREFIX_ALIASES,
    "document": SourceType.DOCUMENT,
    "manual": SourceType.DOCUMENT,
}


def _text(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ResponseSourceAuthorityError(f"{name}: nonblank text required")
    return value.strip()


def _normalize_type(value: Any) -> SourceType:
    raw = _text(value, "source_type")
    key = raw.strip().lower().replace(" ", "_")
    source_type = _TYPE_ALIASES.get(key)
    if source_type is None:
        raise ResponseSourceAuthorityError("unsupported response source_type")
    return source_type


def citation_storage_key(citation: dict[str, Any]) -> tuple[str, SourceType, str]:
    """Return canonical provider storage key from an existing legacy citation.

    The identifier is only a selector. Authorization still requires an exact
    match in the caller-supplied current provider allowance.
    """
    if type(citation) is not dict:
        raise ResponseSourceAuthorityError("citation mapping required")
    raw_key = _text(citation.get("bubble_document_id"), "bubble_document_id")
    assert raw_key is not None
    if ":" in raw_key:
        prefix, raw_id = raw_key.split(":", 1)
        source_type = _PREFIX_ALIASES.get(prefix.strip().lower())
        source_id = _text(raw_id, "structured source id")
        if source_type is None:
            raise ResponseSourceAuthorityError("unsupported structured source prefix")
        assert source_id is not None
        key = f"{source_type.value}:{source_id}"
    else:
        source_type = SourceType.DOCUMENT
        source_id = raw_key
        key = raw_key

    if citation.get("source_type") not in (None, ""):
        claimed_type = _normalize_type(citation.get("source_type"))
        if claimed_type != source_type:
            raise ResponseSourceAuthorityError("response source_type disagrees with storage key")
    if citation.get("source_id") not in (None, ""):
        claimed_id = _text(citation.get("source_id"), "source_id")
        if claimed_id != source_id:
            raise ResponseSourceAuthorityError("response source_id disagrees with storage key")
    return key, source_type, source_id


@dataclass(frozen=True, slots=True)
class ResponseSourceReference:
    citation_id: str
    storage_key: str
    source_type: str
    source_id: str
    company_id: str
    machine_id: str | None

    def __post_init__(self) -> None:
        for name in ("citation_id", "storage_key", "source_type", "source_id", "company_id"):
            _text(getattr(self, name), name)
        _text(self.machine_id, "machine_id", optional=True)


@dataclass(frozen=True, slots=True)
class ResponseSourceManifest:
    company_id: str
    machine_id: str
    ai_scope: str
    references: tuple[ResponseSourceReference, ...]
    version: str = RESPONSE_SOURCE_AUTHORITY_VERSION

    def __post_init__(self) -> None:
        _text(self.company_id, "company_id")
        if type(self.machine_id) is not str:
            raise ResponseSourceAuthorityError("machine_id must be text")
        _text(self.ai_scope, "ai_scope")
        if type(self.references) is not tuple or any(
            type(x) is not ResponseSourceReference for x in self.references
        ):
            raise ResponseSourceAuthorityError("immutable response source references required")
        if self.ai_scope == "machine_all" and not self.machine_id:
            raise ResponseSourceAuthorityError("machine_all manifest requires machine_id")
        if self.ai_scope == "company_general" and self.machine_id:
            raise ResponseSourceAuthorityError("company_general manifest cannot carry machine_id")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "company_id": self.company_id,
            "machine_id": self.machine_id,
            "ai_scope": self.ai_scope,
            "references": [
                {
                    "citation_id": r.citation_id,
                    "storage_key": r.storage_key,
                    "source_type": r.source_type,
                    "source_id": r.source_id,
                    "company_id": r.company_id,
                    "machine_id": r.machine_id,
                }
                for r in self.references
            ],
        }
        payload["manifest_id"] = "rsm1_" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest()
        return payload


def _current_by_storage_key(
    current_allowed_sources: frozenset[SourceIdentity], *, company_id: str
) -> dict[str, SourceIdentity]:
    if type(current_allowed_sources) is not frozenset or any(
        not isinstance(x, SourceIdentity) for x in current_allowed_sources
    ):
        raise ResponseSourceAuthorityError("CURRENT typed provider allowance required")
    by_key: dict[str, SourceIdentity] = {}
    for source in current_allowed_sources:
        if source.scope.company_id != company_id:
            raise ResponseSourceAuthorityError("current allowance contains another company")
        key = storage_key(source)
        previous = by_key.get(key)
        if previous is not None and previous != source:
            raise ResponseSourceAuthorityError("ambiguous current source storage key")
        by_key[key] = source
    return by_key


def validate_citations(
    citations: Iterable[dict[str, Any]], *, company_id: str, machine_id: str,
    ai_scope: str, current_allowed_sources: frozenset[SourceIdentity]
) -> ResponseSourceManifest:
    """Bind every output citation to one exact CURRENT provider source."""
    company_id = _text(company_id, "company_id") or ""
    if type(machine_id) is not str:
        raise ResponseSourceAuthorityError("machine_id must be text")
    ai_scope = _text(ai_scope, "ai_scope") or ""
    if ai_scope not in {"machine_all", "company_general", "document_ids"}:
        raise ResponseSourceAuthorityError("unsupported response ai_scope")
    if ai_scope == "machine_all" and not machine_id:
        raise ResponseSourceAuthorityError("machine_all requires machine_id")
    if ai_scope == "company_general" and machine_id:
        raise ResponseSourceAuthorityError("company_general cannot carry machine_id")

    by_key = _current_by_storage_key(current_allowed_sources, company_id=company_id)
    refs: list[ResponseSourceReference] = []
    for citation in citations:
        if type(citation) is not dict:
            raise ResponseSourceAuthorityError("citation mapping required")
        citation_id = _text(citation.get("citation_id"), "citation_id") or ""
        key, claimed_type, claimed_id = citation_storage_key(citation)
        current = by_key.get(key)
        if current is None:
            raise ResponseSourceAuthorityError("response citation source is not currently authorized")
        if current.source_type != claimed_type or current.source_id != claimed_id:
            raise ResponseSourceAuthorityError("response citation identity disagrees with current source")
        refs.append(ResponseSourceReference(
            citation_id=citation_id,
            storage_key=key,
            source_type=current.source_type.value,
            source_id=current.source_id,
            company_id=current.scope.company_id,
            machine_id=current.scope.machine_id,
        ))
    return ResponseSourceManifest(
        company_id=company_id,
        machine_id=machine_id,
        ai_scope=ai_scope,
        references=tuple(refs),
    )


def validate_links(links: Iterable[dict[str, Any]], *, manifest: ResponseSourceManifest) -> None:
    """Reject any presentation link not anchored to an admitted citation/source pair."""
    if type(manifest) is not ResponseSourceManifest:
        raise ResponseSourceAuthorityError("response source manifest required")
    allowed = {(r.citation_id, r.storage_key) for r in manifest.references}
    for link in links:
        if type(link) is not dict:
            raise ResponseSourceAuthorityError("link mapping required")
        citation_id = _text(link.get("citation_id"), "link citation_id") or ""
        key, _source_type, _source_id = citation_storage_key(link)
        if (citation_id, key) not in allowed:
            raise ResponseSourceAuthorityError("presentation link is not anchored to validated response evidence")


def validate_response(
    response: dict[str, Any], *, company_id: str, machine_id: str, ai_scope: str,
    current_allowed_sources: frozenset[SourceIdentity], require_existing_manifest: bool = False
) -> dict[str, Any]:
    """Return a fresh response carrying a recomputed CURRENT source manifest.

    ``require_existing_manifest`` is for cache hits. The stored manifest is never
    trusted to grant access: current allowance is checked first and a fresh
    manifest is recomputed, then compared to the cached fingerprint.
    """
    if type(response) is not dict:
        raise ResponseSourceAuthorityError("response mapping required")
    citations = response.get("citations") or []
    if type(citations) is not list or any(type(c) is not dict for c in citations):
        raise ResponseSourceAuthorityError("response citations must be a list of mappings")

    status = str(response.get("status") or "").strip().lower()
    grounding = str(response.get("grounding") or "").strip().lower()
    if response.get("ok") is True and status == "answered" and not citations and grounding != "general_technical_knowledge":
        raise ResponseSourceAuthorityError("answered grounded response requires citations")

    manifest = validate_citations(
        citations,
        company_id=company_id,
        machine_id=machine_id,
        ai_scope=ai_scope,
        current_allowed_sources=current_allowed_sources,
    )
    fresh_manifest = manifest.to_dict()
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    existing = meta.get(MANIFEST_META_KEY)
    if require_existing_manifest:
        if type(existing) is not dict:
            raise ResponseSourceAuthorityError("cached response lacks canonical source manifest")
        if existing != fresh_manifest:
            raise ResponseSourceAuthorityError("cached response source manifest is stale or inconsistent")

    links = response.get("rg_links") or []
    if type(links) is not list or any(type(x) is not dict for x in links):
        raise ResponseSourceAuthorityError("response links must be a list of mappings")
    if links:
        validate_links(links, manifest=manifest)

    out = dict(response)
    out_meta = dict(meta)
    out_meta[MANIFEST_META_KEY] = fresh_manifest
    out["meta"] = out_meta
    return out


def make_response_guard(*, company_id: str, machine_id: str, ai_scope: str,
                        current_allowed_sources: frozenset[SourceIdentity],
                        require_existing_manifest: bool = False):
    """Create a request-local pure guard for cache/finalization integration in B4o."""
    def guard(response: dict[str, Any]) -> dict[str, Any]:
        return validate_response(
            response,
            company_id=company_id,
            machine_id=machine_id,
            ai_scope=ai_scope,
            current_allowed_sources=current_allowed_sources,
            require_existing_manifest=require_existing_manifest,
        )
    return guard
