"""Deterministic XLSX validation and structured-text extraction.

This module owns only workbook-adjacent mechanics used by document ingest:
container validation, cell normalization, header inference, row rendering,
sheet pagination and safe workbook traversal. Product configuration and the
historical ``main`` exception type are injected through :class:`XlsxRuntime`.
It does not access tenant scope, PostgreSQL, OpenAI, Cloud Tasks or FastAPI.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Optional


TextTransform = Callable[[str], str]
DisplayCleaner = Callable[..., str]
ExceptionType = type[Exception]


@dataclass(frozen=True)
class XlsxRuntime:
    """Late-bound dependencies and limits supplied by the composition root."""

    normalize_unicode_advanced: TextTransform
    clean_display_text: DisplayCleaner
    openpyxl_module: Any
    basename_fn: Callable[[str], str]
    error_cls: ExceptionType
    datetime_type: type
    date_type: type
    time_type: type
    isfinite_fn: Callable[[float], bool]
    max_xlsx_bytes: int
    max_sheets: int
    max_rows_per_sheet: int
    max_cols_per_sheet: int
    max_cells_total: int
    max_text_chars: int
    page_target_chars: int
    max_cell_chars: int
    max_row_chars: int
    include_hidden_sheets: bool


def xlsx_zip_has_expected_structure(
    xlsx_bytes: bytes,
    *,
    zipfile_module: Any = zipfile,
    bytes_io_fn: Callable[[bytes], Any] = io.BytesIO,
) -> bool:
    """Return true only for a macro-free OOXML workbook container."""
    try:
        with zipfile_module.ZipFile(bytes_io_fn(xlsx_bytes)) as archive:
            names = set(archive.namelist())
            if "xl/vbaProject.bin" in names:
                return False
            return "[Content_Types].xml" in names and "xl/workbook.xml" in names
    except Exception:
        return False


def looks_like_xlsx_document(
    xlsx_bytes: bytes,
    detected_extension: str,
    content_type: str,
    *,
    structure_predicate: Callable[[bytes], bool] = xlsx_zip_has_expected_structure,
) -> bool:
    """Apply the historical extension/content-type/magic XLSX admission rule."""
    extension = str(detected_extension or "").strip().lower()
    normalized_content_type = str(content_type or "").strip().lower()

    if extension in {".xls", ".xlsm", ".xlsb"}:
        return False

    xlsx_content_types = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "binary/octet-stream",
        "application/zip",
    }

    has_xlsx_hint = extension == ".xlsx" or normalized_content_type in xlsx_content_types
    if not has_xlsx_hint and not (xlsx_bytes or b"")[:2] == b"PK":
        return False

    return structure_predicate(xlsx_bytes)


def xlsx_document_title_from_filename(filename: str, *, runtime: XlsxRuntime) -> str:
    name = runtime.basename_fn(str(filename or "").strip())
    name = re.sub(r"\.xlsx$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_-]+", " ", name)
    return runtime.clean_display_text(name, max_len=120)


def xlsx_clean_cell_text(
    value: str,
    max_len: Optional[int] = None,
    *,
    runtime: XlsxRuntime,
) -> str:
    text = runtime.normalize_unicode_advanced(str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if max_len and len(text) > max_len:
        cut = text[: max_len - 1].rsplit(" ", 1)[0].strip() or text[: max_len - 1].strip()
        return cut + "…"
    return text


def xlsx_cell_to_text(
    cell: Any,
    *,
    runtime: XlsxRuntime,
    clean_cell_text_fn: Callable[..., str],
) -> str:
    try:
        value = cell.value
    except Exception:
        value = cell

    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, runtime.datetime_type):
        if value.second or value.microsecond:
            return value.isoformat(sep=" ", timespec="seconds")
        return value.isoformat(sep=" ", timespec="minutes")
    if isinstance(value, runtime.date_type):
        return value.isoformat()
    if isinstance(value, runtime.time_type):
        if value.second or value.microsecond:
            return value.isoformat(timespec="seconds")
        return value.isoformat(timespec="minutes")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if runtime.isfinite_fn(value):
            if value.is_integer():
                return str(int(value))
            return f"{value:.12g}"
        return ""
    return clean_cell_text_fn(str(value), max_len=runtime.max_cell_chars)


def xlsx_trim_trailing_empty(values: list[str]) -> list[str]:
    trimmed = list(values or [])
    while trimmed and not str(trimmed[-1] or "").strip():
        trimmed.pop()
    return trimmed


def xlsx_value_looks_numeric(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?(?:\s*[%€$£]|\s*[a-zA-Z]{1,6})?", text) is not None


def xlsx_detect_header_index(
    rows: list[dict],
    *,
    value_looks_numeric_fn: Callable[[str], bool],
) -> Optional[int]:
    if not rows:
        return None

    best_index: Optional[int] = None
    best_score = 0.0
    for index, row in enumerate(rows[: min(20, len(rows))]):
        values = [str(value or "").strip() for value in row.get("values") or []]
        non_empty = [value for value in values if value]
        if len(non_empty) < 2:
            continue

        textish = sum(1 for value in non_empty if not value_looks_numeric_fn(value))
        if textish < max(1, math.ceil(len(non_empty) * 0.45)):
            continue

        next_rows = rows[index + 1: index + 6]
        next_non_empty_average = 0.0
        if next_rows:
            next_non_empty_average = sum(
                len([value for value in (next_row.get("values") or []) if str(value or "").strip()])
                for next_row in next_rows
            ) / max(1, len(next_rows))

        score = (
            float(len(non_empty))
            + 0.75 * float(textish)
            + 0.35 * min(float(len(non_empty)), next_non_empty_average)
            - 0.05 * index
        )
        if score > best_score:
            best_score = score
            best_index = index

    return best_index if best_score >= 3.0 else None


def _column_letter(runtime: XlsxRuntime, column_index: int) -> str:
    module = runtime.openpyxl_module
    if module is not None:
        return module.utils.get_column_letter(column_index)
    return str(column_index)


def xlsx_make_unique_headers(
    header_values: list[str],
    max_cols: int,
    *,
    runtime: XlsxRuntime,
) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}

    for column_index in range(1, max_cols + 1):
        raw = header_values[column_index - 1] if column_index - 1 < len(header_values) else ""
        label = runtime.clean_display_text(raw, max_len=90) if raw else ""
        if not label:
            label = f"COL {_column_letter(runtime, column_index)}"

        key = label.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            label = f"{label} ({seen[key]})"
        headers.append(label)

    return headers


def xlsx_row_to_line(
    row_number: int,
    values: list[str],
    headers: Optional[list[str]],
    is_header_row: bool,
    *,
    runtime: XlsxRuntime,
    clean_cell_text_fn: Callable[..., str],
) -> str:
    row_values = list(values or [])
    if not row_values:
        return ""

    parts: list[str] = []
    for column_index, value in enumerate(row_values, start=1):
        value = clean_cell_text_fn(value, max_len=runtime.max_cell_chars)
        if not value:
            continue

        column_label = _column_letter(runtime, column_index)
        if headers and not is_header_row:
            label = headers[column_index - 1] if column_index - 1 < len(headers) else f"COL {column_label}"
            parts.append(f"{label}: {value}")
        else:
            parts.append(f"{column_label}: {value}")

    if not parts:
        return ""

    prefix = f"HEADER ROW {row_number}:" if is_header_row else f"ROW {row_number}:"
    line = prefix + " " + " | ".join(parts)
    if len(line) > runtime.max_row_chars:
        line = line[: runtime.max_row_chars - 1].rsplit(" ", 1)[0].strip() + "…"
    return line


def xlsx_append_page(pages: list[str], base_header: list[str], body_lines: list[str]) -> None:
    if not body_lines:
        return
    text = "\n".join(base_header + body_lines).strip()
    if text:
        pages.append(text)


def xlsx_sheet_rows_to_pages(
    sheet_name: str,
    rows: list[dict],
    sheet_index: int,
    document_title: str = "",
    *,
    runtime: XlsxRuntime,
    detect_header_index_fn: Callable[[list[dict]], Optional[int]],
    make_unique_headers_fn: Callable[[list[str], int], list[str]],
    row_to_line_fn: Callable[..., str],
    append_page_fn: Callable[[list[str], list[str], list[str]], None],
) -> list[str]:
    if not rows:
        return []

    max_cols = max((len(row.get("values") or []) for row in rows), default=0)
    header_index = detect_header_index_fn(rows)
    headers = None
    header_row_number = None
    if header_index is not None:
        header_values = list(rows[header_index].get("values") or [])
        headers = make_unique_headers_fn(header_values, max_cols)
        header_row_number = int(rows[header_index].get("row_number") or 0)

    document_title = runtime.clean_display_text(document_title, max_len=120)
    base_header = [
        "DOCUMENT_FILE_TYPE: XLSX",
        "DOCUMENT_KIND: Excel file; file Excel; foglio di calcolo; spreadsheet; workbook",
        "DOCUMENT_FORMAT_HINTS: xlsx excel spreadsheet workbook worksheet sheet table tabella righe colonne fogli",
    ]
    if document_title:
        base_header.append(f"DOCUMENT_TITLE: {document_title}")
    base_header.extend([
        f"SHEET: {sheet_name}",
        f"SHEET_NAME: {sheet_name}",
        f"SHEET_INDEX: {sheet_index}",
        "EXTRACTION_MODE: XLSX values converted to structured text for AI retrieval",
    ])
    if header_row_number:
        base_header.append(f"DETECTED_HEADER_ROW: {header_row_number}")

    pages: list[str] = []
    current_lines: list[str] = []
    current_chars = sum(len(item) + 1 for item in base_header)
    part_number = 1

    def flush() -> None:
        nonlocal current_lines, current_chars, part_number
        if not current_lines:
            return
        header = list(base_header)
        header.append(f"SHEET_PART: {part_number}")
        append_page_fn(pages, header, current_lines)
        part_number += 1
        current_lines = []
        current_chars = sum(len(item) + 1 for item in base_header)

    for index, row in enumerate(rows):
        row_number = int(row.get("row_number") or 0)
        values = list(row.get("values") or [])
        line = row_to_line_fn(
            row_number=row_number,
            values=values,
            headers=headers,
            is_header_row=(header_index is not None and index == header_index),
        )
        if not line:
            continue

        if current_lines and current_chars + len(line) + 1 > max(2000, runtime.page_target_chars):
            flush()

        current_lines.append(line)
        current_chars += len(line) + 1

    flush()
    return pages


def is_xlsx_page_text(text: str) -> bool:
    """Return whether an indexed page is an XLSX structured-text page."""
    normalized = str(text or "").lstrip()
    return normalized.startswith("DOCUMENT_FILE_TYPE: XLSX")


def chunk_xlsx_pages(
    pages: list[tuple[int, str]],
    target_chars: int,
    min_chars: int,
    *,
    chunk_page_fn: Callable[..., list[dict]],
) -> list[dict]:
    """Chunk XLSX pages independently so sheet/part boundaries never merge.

    Each extracted XLSX page represents one worksheet part and repeats its own
    sheet metadata.  Passing all pages to the generic PDF chunker can merge the
    first rows of several worksheets into one chunk.  That makes exact identifiers
    beyond the leading snippet invisible to retrieval and can substitute evidence
    from the first sheet.  Chunking one physical XLSX page at a time preserves the
    historical text while enforcing the worksheet boundary deterministically.
    """
    chunks: list[dict] = []
    next_chunk_index = 1

    for page_number, page_text in pages or []:
        page_chunks = chunk_page_fn(
            pages=[(int(page_number), str(page_text or ""))],
            target_chars=target_chars,
            # XLSX rows are already self-contained structured records.  Reusing the
            # PDF overlap here would duplicate nearly the whole of short worksheets
            # into many suffix chunks.  Keep overlap inside the generic PDF path only.
            overlap_chars=0,
            min_chars=min_chars,
        )
        for chunk in page_chunks or []:
            item = dict(chunk)
            item["chunk_index"] = next_chunk_index
            item["page_from"] = int(page_number)
            item["page_to"] = int(page_number)
            chunks.append(item)
            next_chunk_index += 1

    return chunks


def extract_xlsx_sheets_as_pages(
    xlsx_bytes: bytes,
    detected_filename: str = "",
    *,
    runtime: XlsxRuntime,
    bytes_io_fn: Callable[[bytes], Any],
    document_title_fn: Callable[[str], str],
    cell_to_text_fn: Callable[[Any], str],
    trim_trailing_empty_fn: Callable[[list[str]], list[str]],
    sheet_rows_to_pages_fn: Callable[..., list[str]],
) -> list[str]:
    openpyxl_module = runtime.openpyxl_module
    if openpyxl_module is None:
        raise runtime.error_cls(
            "XLSX_DEPENDENCY_MISSING",
            "Documento non indicizzabile: supporto XLSX non installato nel backend.",
        )

    if len(xlsx_bytes or b"") > runtime.max_xlsx_bytes:
        raise runtime.error_cls(
            "XLSX_FILE_TOO_LARGE",
            "Documento non indicizzabile: file XLSX troppo grande per l'ingest.",
            {"max_xlsx_bytes": runtime.max_xlsx_bytes, "actual_bytes": len(xlsx_bytes or b"")},
        )

    try:
        workbook = openpyxl_module.load_workbook(
            filename=bytes_io_fn(xlsx_bytes),
            read_only=True,
            data_only=True,
        )
    except Exception as error:
        raise runtime.error_cls(
            "XLSX_PARSE_FAILED",
            "Documento non indicizzabile: impossibile leggere il file XLSX.",
            {"detail": str(error)[:300]},
        )

    pages: list[str] = []
    total_cells = 0
    total_text_chars = 0
    processed_sheets = 0
    document_title = document_title_fn(detected_filename)

    try:
        for worksheet in workbook.worksheets:
            if processed_sheets >= max(1, runtime.max_sheets):
                break

            if (
                not runtime.include_hidden_sheets
                and str(getattr(worksheet, "sheet_state", "visible") or "visible") != "visible"
            ):
                continue

            processed_sheets += 1
            sheet_name = (
                runtime.clean_display_text(getattr(worksheet, "title", "Sheet"), max_len=90)
                or f"Sheet {processed_sheets}"
            )

            sheet_rows: list[dict] = []
            max_rows = max(1, runtime.max_rows_per_sheet)
            max_cols = max(1, runtime.max_cols_per_sheet)

            for row in worksheet.iter_rows(max_row=max_rows, max_col=max_cols):
                values = [cell_to_text_fn(cell) for cell in row]
                values = trim_trailing_empty_fn(values)
                if not any(str(value or "").strip() for value in values):
                    continue

                row_number = (
                    int(getattr(row[0], "row", len(sheet_rows) + 1) or len(sheet_rows) + 1)
                    if row
                    else len(sheet_rows) + 1
                )
                non_empty_cells = sum(1 for value in values if str(value or "").strip())
                total_cells += non_empty_cells
                if total_cells > max(1, runtime.max_cells_total):
                    raise runtime.error_cls(
                        "XLSX_TOO_MANY_CELLS",
                        "Documento non indicizzabile: file XLSX troppo grande o troppo denso di celle.",
                        {"max_cells_total": runtime.max_cells_total},
                    )

                row_text_chars = sum(len(str(value or "")) for value in values)
                total_text_chars += row_text_chars
                if total_text_chars > max(1000, runtime.max_text_chars):
                    raise runtime.error_cls(
                        "XLSX_TEXT_TOO_LARGE",
                        "Documento non indicizzabile: testo estratto da XLSX troppo grande per l'ingest sicuro.",
                        {"max_text_chars": runtime.max_text_chars},
                    )

                sheet_rows.append({"row_number": row_number, "values": values})

            new_pages = sheet_rows_to_pages_fn(
                sheet_name,
                sheet_rows,
                processed_sheets,
                document_title=document_title,
            )
            pages.extend(new_pages)

            converted_text_chars = sum(len(page or "") for page in pages)
            if converted_text_chars > max(2000, runtime.max_text_chars * 2):
                raise runtime.error_cls(
                    "XLSX_TEXT_TOO_LARGE",
                    "Documento non indicizzabile: testo strutturato da XLSX troppo grande per l'ingest sicuro.",
                    {"max_structured_text_chars": runtime.max_text_chars * 2},
                )

    finally:
        try:
            workbook.close()
        except Exception:
            pass

    pages = [page for page in pages if str(page or "").strip()]
    if not pages:
        raise runtime.error_cls(
            "XLSX_NO_READABLE_TEXT",
            "Documento non indicizzabile: nessun testo leggibile trovato nel file XLSX.",
        )

    return pages
