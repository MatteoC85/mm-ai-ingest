"""Deterministic file-format routing for document ingest."""
from __future__ import annotations

from collections.abc import Callable


XlsxPredicate = Callable[[bytes, str, str], bool]


def detect_ingest_source_file_type(
    data: bytes,
    detected_extension: str,
    content_type: str,
    *,
    looks_like_xlsx_fn: XlsxPredicate,
) -> str:
    """Return ``pdf``, ``xlsx`` or an empty string using production precedence."""
    pdf_magic = b"%PDF" in (data or b"")[:1024]
    looks_like_pdf = (
        pdf_magic
        or str(detected_extension or "").strip().lower() == ".pdf"
        or str(content_type or "").strip().lower() == "application/pdf"
    )
    looks_like_xlsx = looks_like_xlsx_fn(data, detected_extension, content_type)

    if looks_like_pdf:
        return "pdf"
    if looks_like_xlsx:
        return "xlsx"
    return ""
