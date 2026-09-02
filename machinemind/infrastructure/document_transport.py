"""External document-input transport used by the ingest endpoint.

This module is a behavior-preserving extraction from the production monolith. It
owns data-URL stripping, strict base64 decoding, URL filename detection and the
bounded HTTP GET used to fetch a document before parsing. The composition root
supplies live dependencies so historical monkeypatch points, FastAPI error envelopes
and runtime timeout configuration remain unchanged during the migration.
"""
from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Callable
from typing import Any


GetFn = Callable[..., Any]
DecodeFn = Callable[[str], bytes]
FilenameFn = Callable[[str], str]
ExceptionType = type[BaseException]


def strip_data_url_prefix(value: str) -> str:
    """Return the payload portion of a data URL, preserving legacy semantics."""
    raw = (value or "").strip()
    if "," in raw and raw.split(",", 1)[0].lower().startswith("data:"):
        return raw.split(",", 1)[1]
    return raw


def decode_file_base64(
    file_base64: str,
    *,
    strip_prefix_fn: Callable[[str], str],
    http_exception_cls: ExceptionType,
) -> bytes:
    """Strictly decode uploaded base64 or raise the historical FastAPI error."""
    raw = strip_prefix_fn(file_base64)
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise http_exception_cls(status_code=400, detail="Invalid file_base64")


def detect_filename_from_url(
    url: str,
    *,
    urlparse_fn: Callable[[str], Any],
    unquote_fn: Callable[[str], str],
    basename_fn: Callable[[str], str] = os.path.basename,
) -> str:
    """Derive a filename from the URL path, returning an empty string on failure."""
    try:
        url_path = unquote_fn(urlparse_fn(url).path or "")
        return basename_fn(url_path) or ""
    except Exception:
        return ""


def load_ingest_document_file(
    payload: Any,
    bubble_document_id: str,
    *,
    fetch_timeout: int,
    get_fn: GetFn,
    decode_base64_fn: DecodeFn,
    detect_filename_fn: FilenameFn,
    http_exception_cls: ExceptionType,
) -> dict:
    """Load ingest bytes from base64 or a remote URL without changing policy."""
    url = (payload.file_url or "").strip()
    if url.startswith("//"):
        url = "https:" + url

    file_base64 = (payload.file_base64 or "").strip()
    payload_filename = (payload.filename or "").strip()
    payload_content_type = (
        (payload.content_type or "").split(";", 1)[0].strip().lower()
    )

    if not url and not file_base64:
        raise http_exception_cls(
            status_code=400,
            detail="Missing file_url or file_base64",
        )

    if file_base64:
        data = decode_base64_fn(file_base64)
        detected_filename = (
            payload_filename
            or detect_filename_fn(url)
            or bubble_document_id
        )
        detected_extension = os.path.splitext(detected_filename)[1].lower()

        return {
            "data": data,
            "url": url,
            "content_type": payload_content_type,
            "content_disposition": "",
            "detected_filename": detected_filename,
            "detected_extension": detected_extension,
            "source_mode": "file_base64",
        }

    try:
        response = get_fn(url, timeout=fetch_timeout)
        response.raise_for_status()
        data = response.content
    except http_exception_cls:
        raise
    except Exception:
        raise http_exception_cls(status_code=502, detail="Fetch failed")

    content_type = (
        (response.headers.get("Content-Type") or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )
    content_disposition = (
        response.headers.get("Content-Disposition") or ""
    ).strip()

    detected_filename = detect_filename_fn(url)
    detected_extension = os.path.splitext(detected_filename)[1].lower()

    if "filename=" in content_disposition:
        cd_filename = (
            content_disposition.split("filename=", 1)[1]
            .strip()
            .strip('"')
            .strip("'")
        )
        if cd_filename:
            detected_filename = cd_filename
            detected_extension = os.path.splitext(detected_filename)[1].lower()

    return {
        "data": data,
        "url": url,
        "content_type": content_type,
        "content_disposition": content_disposition,
        "detected_filename": detected_filename,
        "detected_extension": detected_extension,
        "source_mode": "file_url",
    }
