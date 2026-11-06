"""Literal P&S field reading for the independently reviewed Root fallback.

This reader does not establish relevance or authorize an operation. It preserves
field contents for draft construction; the original citation remains unchanged.
"""
from __future__ import annotations

import re


_FIELD = re.compile(
    r"[ \t]*(SOURCE_TYPE|TITLE|CATEGORY|DESCRIPTION|SOLUTION|NOTES)[ \t]*:[ \t]*",
    re.IGNORECASE,
)


def ps_fields(text: str) -> dict[str, str]:
    """Read canonical field spans without losing multiline continuations.

    Only the P&S envelope labels emitted by ingestion delimit fields. Other
    headings, colon lines and section wrappers remain literal field content.
    Repeated envelope labels are ambiguous, so the entire draft input is rejected
    rather than joining possibly different records. No source text is rewritten.
    """
    if not isinstance(text, str):
        return {}
    markers: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _FIELD.match(line)
        if match:
            key = match.group(1).lower()
            if key in seen:
                return {}
            seen.add(key)
            markers.append((key, offset, offset + match.end()))
        offset += len(line)
    return {
        key: text[value_start:markers[index + 1][1] if index + 1 < len(markers) else len(text)].strip()
        for index, (key, _header_start, value_start) in enumerate(markers)
    }
