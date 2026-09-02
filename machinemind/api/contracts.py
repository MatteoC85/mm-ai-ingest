"""Stable FastAPI/Pydantic request contracts extracted from the certified monolith.

This module is a mechanical relocation. Field names, types, requiredness and defaults
must remain compatible with the production baseline frozen at Git commit
89a33a549930003fc0761d7a3f47b70bc22e0c84.
"""
from typing import List, Optional, Union
from pydantic import BaseModel


class IngestRequest(BaseModel):
    file_url: Optional[str] = None
    file_base64: Optional[str] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: str
    ai_scope: Optional[str] = None
    plan_embed_chars_limit_total: Optional[int] = None
    plan_index_storage_limit_bytes: Optional[int] = None
    embed_chars_used_total: Optional[int] = None
    index_storage_used_total: Optional[int] = None
    doc_prev_embed_chars: Optional[int] = None
    doc_prev_index_storage_bytes: Optional[int] = None

    # Optional context forwarded by the Worker. It is accepted now without making
    # the Cloud Run route responsible for plan enforcement.
    ingest_request_key: Optional[str] = None
    ingest_month_key: Optional[str] = None
    ingest_credits_limit_month: Optional[float] = None
    ingest_credits_used_before: Optional[float] = None
    ingest_credits_enforced: Optional[bool] = None
    ingest_request_already_admitted: Optional[bool] = None
    ingest_metering_version: Optional[str] = None


class IndexDocumentRequest(BaseModel):
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: str
    trace_id: Optional[str] = None
    ai_scope: Optional[str] = None
    ingest_request_key: Optional[str] = None
    ingest_month_key: Optional[str] = None
    ingest_usage_event_id: Optional[str] = None
    ingest_metering_enabled: Optional[bool] = False


class IngestUsageMonthRequest(BaseModel):
    company_id: str
    month_key: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    company_id: str
    bubble_document_id: Optional[str] = None
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    ai_scope: Optional[str] = None
    language: Optional[str] = None
    top_k: int = 5
    debug: Optional[bool] = False


class RootCauseRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    ai_scope: Optional[str] = None
    language: Optional[str] = None
    top_k: int = 8
    max_causes: int = 3
    debug: Optional[bool] = False


class DraftPSOptions(BaseModel):
    top_k: int = 8
    max_causes: int = 3


# Transitional module identity keeps nested Pydantic annotations compatible with
# the historical top-level `main.DraftPSOptions` contract.
DraftPSOptions.__module__ = "main"


class DraftPSRequest(BaseModel):
    query: str
    company_id: str
    machine_id: Optional[str] = None
    bubble_document_id: Optional[str] = None
    document_ids: Optional[Union[List[str], str]] = None
    ai_scope: Optional[str] = None
    language: Optional[str] = None
    options: Optional[DraftPSOptions] = None
    debug: Optional[bool] = False


class DeleteDocumentRequest(BaseModel):
    company_id: str
    bubble_document_id: str


class DeleteCompanyIndexRequest(BaseModel):
    company_id: str


class StructuredSourceIngestRequest(BaseModel):
    company_id: str
    machine_id: str
    source_type: str
    source_id: str
    source_url: Optional[str] = None

    title: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = None
    procedure_type: Optional[str] = None
    step_number: Optional[int] = None

    # Canonical Bubble relation for Step -> Procedure. These fields are metadata;
    # they do not change the Step text/embedding contract for existing sources.
    parent_procedure_id: Optional[str] = None
    parent_procedure_code: Optional[str] = None
    parent_procedure_title: Optional[str] = None

    category: Optional[str] = None
    solution: Optional[str] = None
    notes: Optional[str] = None

    plan_embed_chars_limit_total: Optional[int] = None
    plan_index_storage_limit_bytes: Optional[int] = None
    embed_chars_used_total: Optional[int] = None
    index_storage_used_total: Optional[int] = None
    doc_prev_embed_chars: Optional[int] = None
    doc_prev_index_storage_bytes: Optional[int] = None


for _model in [
    IngestRequest, IndexDocumentRequest, IngestUsageMonthRequest, SearchRequest,
    AskRequest, RootCauseRequest, DraftPSOptions, DraftPSRequest,
    DeleteDocumentRequest, DeleteCompanyIndexRequest, StructuredSourceIngestRequest,
]:
    _model.__module__ = "main"
del _model


__all__ = [
    "IngestRequest", "IndexDocumentRequest", "IngestUsageMonthRequest",
    "SearchRequest", "AskRequest", "RootCauseRequest", "DraftPSOptions",
    "DraftPSRequest", "DeleteDocumentRequest", "DeleteCompanyIndexRequest",
    "StructuredSourceIngestRequest",
]
