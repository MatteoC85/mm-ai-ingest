"""PostgreSQL persistence boundary for document ingest.

The composition root supplies the live connection factory explicitly.  These
functions preserve the SQL, transaction, fail/close behavior and public return
shapes that historically lived in ``main.py`` while keeping tenant-aware ingest
storage behind one importable module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class IngestPersistenceRuntime:
    """Late-bound persistence dependencies supplied by the composition root."""

    connect_db: Callable[[], Any]
    dumps_json: Callable[..., str]
    loads_json: Callable[[str], Any]


def upsert_document_file(
    company_id: str,
    bubble_document_id: str,
    file_url: str,
    *,
    runtime: IngestPersistenceRuntime,
) -> None:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    file_url = (file_url or "").strip()
    if not (company_id and bubble_document_id and file_url):
        return

    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.document_files(company_id, bubble_document_id, file_url, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (company_id, bubble_document_id)
                DO UPDATE SET file_url = EXCLUDED.file_url, updated_at = NOW();
                """,
                (company_id, bubble_document_id, file_url),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_cleaning_meta(
    company_id: str,
    bubble_document_id: str,
    header_norm: set[str],
    footer_norm: set[str],
    *,
    runtime: IngestPersistenceRuntime,
) -> None:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    if not (company_id and bubble_document_id):
        return

    header_list = sorted([x for x in (header_norm or set()) if x])
    footer_list = sorted([x for x in (footer_norm or set()) if x])

    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.document_cleaning_meta(company_id, bubble_document_id, header_norm, footer_norm, updated_at)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (company_id, bubble_document_id)
                DO UPDATE SET header_norm = EXCLUDED.header_norm,
                              footer_norm = EXCLUDED.footer_norm,
                              updated_at = NOW();
                """,
                (
                    company_id,
                    bubble_document_id,
                    runtime.dumps_json(header_list),
                    runtime.dumps_json(footer_list),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_cleaning_meta(
    company_id: str,
    bubble_document_id: str,
    *,
    runtime: IngestPersistenceRuntime,
) -> tuple[set[str], set[str]]:
    company_id = (company_id or "").strip()
    bubble_document_id = (bubble_document_id or "").strip()
    if not (company_id and bubble_document_id):
        return set(), set()

    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT header_norm, footer_norm
                FROM public.document_cleaning_meta
                WHERE company_id=%s AND bubble_document_id=%s
                LIMIT 1;
                """,
                (company_id, bubble_document_id),
            )
            row = cur.fetchone()
            if not row:
                return set(), set()

            header_value, footer_value = row
            if isinstance(header_value, str):
                header_value = runtime.loads_json(header_value or "[]")
            if isinstance(footer_value, str):
                footer_value = runtime.loads_json(footer_value or "[]")

            header_set = {str(x) for x in (header_value or []) if x}
            footer_set = {str(x) for x in (footer_value or []) if x}
            return header_set, footer_set
    finally:
        conn.close()


def get_index_usage(
    company_id: str,
    bubble_document_id: str | None = None,
    *,
    runtime: IngestPersistenceRuntime,
) -> dict:
    company_id = (company_id or "").strip()
    bubble_document_id = (
        (bubble_document_id or "").strip() if bubble_document_id else None
    )

    if not company_id:
        return {
            "text_chars": 0,
            "chunk_count": 0,
            "est_storage_bytes": 0,
        }

    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            if bubble_document_id:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(text_chars), 0)
                    FROM public.document_pages
                    WHERE company_id=%s
                      AND bubble_document_id=%s;
                    """,
                    (company_id, bubble_document_id),
                )
                text_chars = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.document_chunks
                    WHERE company_id=%s
                      AND bubble_document_id=%s;
                    """,
                    (company_id, bubble_document_id),
                )
                chunk_count = int(cur.fetchone()[0] or 0)
            else:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(text_chars), 0)
                    FROM public.document_pages
                    WHERE company_id=%s;
                    """,
                    (company_id,),
                )
                text_chars = int(cur.fetchone()[0] or 0)

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.document_chunks
                    WHERE company_id=%s;
                    """,
                    (company_id,),
                )
                chunk_count = int(cur.fetchone()[0] or 0)

        est_storage_bytes = int(text_chars * 3 + chunk_count * 2000)
        return {
            "text_chars": text_chars,
            "chunk_count": chunk_count,
            "est_storage_bytes": est_storage_bytes,
        }
    finally:
        conn.close()


def replace_document_pages(
    *,
    company_id: str,
    machine_id: str,
    bubble_document_id: str,
    pages_text: list[str],
    schema_qualified: bool = False,
    runtime: IngestPersistenceRuntime,
) -> None:
    """Replace the historical page rows in one transaction."""

    table_name = "public.document_pages" if schema_qualified else "document_pages"
    conn = runtime.connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table_name} WHERE company_id=%s AND bubble_document_id=%s;",
                (company_id, bubble_document_id),
            )
            for page_number, text in enumerate(pages_text, start=1):
                if schema_qualified:
                    insert_sql = """
                    INSERT INTO public.document_pages(
                        company_id,
                        machine_id,
                        bubble_document_id,
                        page_number,
                        text,
                        text_chars
                    )
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """
                else:
                    insert_sql = """
                    INSERT INTO document_pages(company_id, machine_id, bubble_document_id, page_number, text, text_chars)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """
                cur.execute(
                    insert_sql,
                    (
                        company_id,
                        machine_id,
                        bubble_document_id,
                        page_number,
                        text,
                        len(text),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
