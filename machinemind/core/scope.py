"""Tenant/machine/document query-scope contract from the certified monolith."""
from typing import List, Optional, Union
from fastapi import HTTPException

COMPANY_GENERAL_MACHINE_SENTINEL = "__MM_COMPANY_GENERAL__"


def normalize_document_ids(value: Optional[Union[List[str], str]]) -> Optional[list[str]]:
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]

    if isinstance(value, list):
        value = [str(x).strip() for x in value if str(x).strip()]
        return value or None

    return None


def normalize_ai_scope(value: Optional[str]) -> str:
    s = str(value or "").strip().lower()

    if not s:
        return "machine_all"

    if s in {"machine", "machine_all", "machine_all_plus_company"}:
        return "machine_all"

    if s in {"company", "company_general", "company_only", "general"}:
        return "company_general"

    if s in {"document_ids", "documents", "document"}:
        return "document_ids"

    raise HTTPException(status_code=400, detail=f"Unsupported ai_scope: {value}")


def resolve_query_scope(
    company_id: str,
    machine_id: Optional[str],
    bubble_document_id: Optional[str] = None,
    document_ids: Optional[Union[List[str], str]] = None,
    ai_scope: Optional[str] = None,
) -> dict:
    company_id = (company_id or "").strip()
    if not company_id:
        raise HTTPException(status_code=400, detail="Missing company_id")

    explicit_scope = bool(str(ai_scope or "").strip())
    resolved_scope = normalize_ai_scope(ai_scope)

    machine_id = (machine_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip() or None
    doc_ids = normalize_document_ids(document_ids)

    if resolved_scope == "company_general":
        machine_id = COMPANY_GENERAL_MACHINE_SENTINEL
        bubble_document_id = None
        doc_ids = None
    elif resolved_scope == "document_ids" or (
        not explicit_scope and (doc_ids or bubble_document_id)
    ):
        resolved_scope = "document_ids"
        if not machine_id:
            machine_id = COMPANY_GENERAL_MACHINE_SENTINEL
    else:
        resolved_scope = "machine_all"
        if not machine_id:
            raise HTTPException(status_code=400, detail="Missing machine_id")

        if explicit_scope:
            bubble_document_id = None
            doc_ids = None

    return {
        "company_id": company_id,
        "machine_id": machine_id,
        "bubble_document_id": bubble_document_id,
        "document_ids": doc_ids,
        "ai_scope": resolved_scope,
    }


# Temporary compatibility aliases. Existing main.py callers retain their historical
# private symbol names while new modules use public names without underscore prefixes.
_normalize_document_ids = normalize_document_ids
_normalize_ai_scope = normalize_ai_scope
_resolve_query_scope = resolve_query_scope

__all__ = [
    "COMPANY_GENERAL_MACHINE_SENTINEL",
    "normalize_document_ids", "normalize_ai_scope", "resolve_query_scope",
    "_normalize_document_ids", "_normalize_ai_scope", "_resolve_query_scope",
]
