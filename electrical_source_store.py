import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage


SOURCE_SNAPSHOT_ENABLED = (
    os.environ.get("MM_ELECTRICAL_SOURCE_SNAPSHOT_ENABLED") or "0"
).strip() == "1"
SOURCE_BUCKET = (
    os.environ.get("MM_ELECTRICAL_SOURCE_BUCKET") or ""
).strip()
SOURCE_PREFIX = (
    os.environ.get("MM_ELECTRICAL_SOURCE_PREFIX")
    or "electrical-source-v1"
).strip().strip("/")
SOURCE_TIMEOUT_SECONDS = int(
    os.environ.get("MM_ELECTRICAL_SOURCE_TIMEOUT_SECONDS", "90")
)
SOURCE_MAX_BYTES = int(
    os.environ.get("MM_ELECTRICAL_SOURCE_MAX_BYTES", str(100 * 1024 * 1024))
)


def get_electrical_source_runtime_config() -> dict[str, Any]:
    return {
        "enabled": SOURCE_SNAPSHOT_ENABLED,
        "bucket": SOURCE_BUCKET,
        "prefix": SOURCE_PREFIX,
        "timeout_seconds": SOURCE_TIMEOUT_SECONDS,
        "max_bytes": SOURCE_MAX_BYTES,
        "backend": "gcs-private-v1",
    }


def _safe_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._-")
    return (cleaned or fallback)[:180]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data or b"").hexdigest()


def build_electrical_source_object_name(
    *,
    company_id: str,
    bubble_document_id: str,
    source_sha256: str,
) -> str:
    sha = str(source_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid electrical source SHA-256")

    company = _safe_segment(company_id, fallback="company")
    document = _safe_segment(bubble_document_id, fallback="document")
    return f"{SOURCE_PREFIX}/{company}/{document}/{sha}.pdf"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    value = str(uri or "").strip()
    if not value.startswith("gs://"):
        raise ValueError("Electrical source URI must use gs://")
    remainder = value[5:]
    bucket_name, sep, object_name = remainder.partition("/")
    if not sep or not bucket_name or not object_name:
        raise ValueError("Invalid gs:// electrical source URI")
    return bucket_name, object_name


def snapshot_electrical_source_pdf(
    *,
    data: bytes,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    source_sha256: str,
    source_filename: Optional[str] = None,
) -> dict[str, Any]:
    if not SOURCE_SNAPSHOT_ENABLED:
        raise RuntimeError("MM_ELECTRICAL_SOURCE_SNAPSHOT_ENABLED is not enabled")
    if not SOURCE_BUCKET:
        raise RuntimeError("MM_ELECTRICAL_SOURCE_BUCKET is missing")
    if not data:
        raise ValueError("Electrical source PDF is empty")
    if len(data) > SOURCE_MAX_BYTES:
        raise ValueError(
            f"Electrical source PDF exceeds MM_ELECTRICAL_SOURCE_MAX_BYTES "
            f"({len(data)}>{SOURCE_MAX_BYTES})"
        )
    if b"%PDF" not in data[:1024]:
        raise ValueError("Electrical source snapshot is not a PDF")

    expected_sha = str(source_sha256 or "").strip().lower()
    actual_sha = _sha256_bytes(data)
    if actual_sha != expected_sha:
        raise ValueError("Electrical source snapshot SHA-256 mismatch")

    object_name = build_electrical_source_object_name(
        company_id=company_id,
        bubble_document_id=bubble_document_id,
        source_sha256=expected_sha,
    )

    client = storage.Client()
    bucket = client.bucket(SOURCE_BUCKET)
    blob = bucket.blob(object_name)
    reused = False

    metadata = {
        "source_sha256": expected_sha,
        "company_id": str(company_id or "")[:1024],
        "machine_id": str(machine_id or "")[:1024],
        "bubble_document_id": str(bubble_document_id or "")[:1024],
        "source_filename": str(source_filename or "")[:1024],
        "snapshot_backend": "gcs-private-v1",
    }

    try:
        blob.metadata = metadata
        blob.cache_control = "private, no-store, max-age=0"
        blob.upload_from_string(
            data,
            content_type="application/pdf",
            if_generation_match=0,
            timeout=SOURCE_TIMEOUT_SECONDS,
            checksum="auto",
        )
    except PreconditionFailed:
        reused = True
        blob.reload(timeout=SOURCE_TIMEOUT_SECONDS)
    else:
        blob.reload(timeout=SOURCE_TIMEOUT_SECONDS)

    stored_size = int(blob.size or 0)
    stored_metadata = dict(blob.metadata or {})
    stored_sha = str(stored_metadata.get("source_sha256") or "").strip().lower()

    if stored_size != len(data) or (stored_sha and stored_sha != expected_sha):
        stored_data = blob.download_as_bytes(
            timeout=SOURCE_TIMEOUT_SECONDS,
            checksum="auto",
        )
        if _sha256_bytes(stored_data) != expected_sha:
            raise RuntimeError("Existing electrical source snapshot content mismatch")

    return {
        "status": "ready",
        "backend": "gcs-private-v1",
        "bucket": SOURCE_BUCKET,
        "object": object_name,
        "uri": f"gs://{SOURCE_BUCKET}/{object_name}",
        "generation": str(blob.generation or ""),
        "size_bytes": len(data),
        "sha256": expected_sha,
        "reused": reused,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


def download_electrical_source_pdf(
    *,
    uri: str,
    expected_sha256: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> bytes:
    bucket_name, object_name = _parse_gs_uri(uri)
    max_allowed = int(max_bytes or SOURCE_MAX_BYTES)

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    blob.reload(timeout=SOURCE_TIMEOUT_SECONDS)

    size = int(blob.size or 0)
    if size <= 0:
        raise ValueError("Electrical source snapshot is empty")
    if size > max_allowed:
        raise ValueError(
            f"Electrical source snapshot exceeds configured maximum "
            f"({size}>{max_allowed})"
        )

    data = blob.download_as_bytes(
        timeout=SOURCE_TIMEOUT_SECONDS,
        checksum="auto",
    )
    if len(data) != size:
        raise RuntimeError("Electrical source snapshot download size mismatch")

    expected = str(expected_sha256 or "").strip().lower()
    if expected and _sha256_bytes(data) != expected:
        raise RuntimeError("Electrical source snapshot SHA-256 mismatch")
    return data


def delete_electrical_source_uri(uri: str) -> bool:
    value = str(uri or "").strip()
    if not value:
        return False
    bucket_name, object_name = _parse_gs_uri(value)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        blob.delete(timeout=SOURCE_TIMEOUT_SECONDS)
        return True
    except NotFound:
        return False


def delete_electrical_source_uris(uris: list[str]) -> dict[str, Any]:
    unique = []
    seen = set()
    for uri in uris or []:
        value = str(uri or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)

    deleted = 0
    missing = 0
    errors: list[str] = []
    for uri in unique:
        try:
            if delete_electrical_source_uri(uri):
                deleted += 1
            else:
                missing += 1
        except Exception as exc:
            errors.append(f"{uri}: {str(exc)[:500]}")

    return {
        "requested": len(unique),
        "deleted": deleted,
        "missing": missing,
        "errors": errors,
    }
