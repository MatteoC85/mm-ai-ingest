"""ASK-only numerical/unit claim normalization.

The inherited unit fix is deliberately NOT installed in main's shared legacy
parser: that parser is still used by Root Cause. Only the protected ASK
intake runtime factory binds this module. There are no model calls, permissions,
query policies, source preferences or unit conversions here.

All numeric precision, bracket matching and rejection rules are those of the
reviewed unit candidate. The legacy parser remains frozen until its own phase.
"""
from __future__ import annotations
import re
from dataclasses import replace
from typing import Optional
from ..ingest.text_pdf import normalize_unicode_advanced as _normalize_unicode_advanced
from .validation import AskValidationRuntime

_ASSISTANT_CORE_UNIT_TOKEN = r"(?:mm(?:²|2|³|3)?|cm|kg|bar|k\s*pa|m\s*pa|pa|k\s*hz|hz|k\s*w|k\s*n|n\s*(?:[·*x]\s*)?m|nm|rpm|giri\s*/?\s*min(?:uto)?|min\s*(?:-?1|⁻¹)|1\s*/\s*min|ms|°\s*c|°|cst|db\s*\(?a\)?|cycles?|cicli|pieces?|pezzi|occurrences?|occorrenze|%|v|a|w|m|g|s|h)"
_ASSISTANT_CORE_TECH_UNIT_AFTER_RE = re.compile(
    # A unit immediately after a value may use matched square brackets, as in
    # technical PDF tables. Preserve the same unit vocabulary and comparisons.
    rf"^\s*(?:(?:circa|about|approx(?:imately)?|ca\.?|~)\s*)?(?P<bracket>\[\s*)?(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})(?(bracket)\s*\])(?=$|[\s,.;:/)\]])",
    re.IGNORECASE,
)
_ASSISTANT_CORE_RANGE_WITH_UNIT_RE = re.compile(
    rf"^\s*(?:-|–|—|÷|to|a)\s*[-+]?\d+(?:[.,]\d+)?\s*(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})(?=$|[\s,.;:/)\]])",
    re.IGNORECASE,
)
_ASSISTANT_CORE_TECH_UNIT_BEFORE_RE = re.compile(
    # Unit-before-value support is intentionally limited to bracketed table
    # headings such as ``[N m] 4 647`` or ``[min-1] 159,57``. Accepting any
    # nearby token would misread ordinary prose/dimension labels (for example
    # the ``h`` in ``l x h``) as a unit for the following number.
    rf"\[\s*(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})\s*\]\s*$",
    re.IGNORECASE,
)
_ASSISTANT_CORE_NUMBER_ATOM = r"[-+]?(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:[.,]\d+)?(?!\d)"
_ASSISTANT_CORE_NUMBER_RE = re.compile(
    rf"(?<![\w]){_ASSISTANT_CORE_NUMBER_ATOM}"
)
# XLSX structured rows keep values and units in adjacent labeled fields, e.g.
# ``Valore: 17.6 | Unità: bar``.  Recognize that bounded representation as one
# technical claim without changing ordinary prose parsing.
_ASSISTANT_CORE_LABELED_VALUE_UNIT_RE = re.compile(
    rf"\b(?:valore|value)\s*:\s*(?P<value>{_ASSISTANT_CORE_NUMBER_ATOM})"
    rf"\s*\|\s*(?:unità|unita|unit)\s*:\s*(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})"
    rf"(?=$|[\s|,.;:/)\]])",
    re.IGNORECASE,
)
_ASSISTANT_CORE_DIMENSION_CHAIN_RE = re.compile(
    rf"(?P<values>{_ASSISTANT_CORE_NUMBER_ATOM}(?:\s*[x×]\s*{_ASSISTANT_CORE_NUMBER_ATOM})+)\s*(?P<unit>mm(?:²|2|³|3)?|cm|m)(?=$|[\s,.;:/)\]])",
    re.IGNORECASE,
)
_ASSISTANT_CORE_DIMENSION_CHAIN_HEADING_RE = re.compile(
    # Technical tables often put the common unit in the row heading and the
    # dimension chain on the next line: ``Misure ... in mm\n2100 x ...``.
    rf"(?:dimensioni|misure|ingombro|dimensions?|sizes?|envelope)[^\n]{{0,120}}?\b(?P<unit>mm(?:²|2|³|3)?|cm|m)\b[^\n]*\n\s*(?P<values>{_ASSISTANT_CORE_NUMBER_ATOM}(?:\s*[x×]\s*{_ASSISTANT_CORE_NUMBER_ATOM})+)",
    re.IGNORECASE,
)
_ASSISTANT_CORE_CODE_RE = re.compile(
    r"\b(?=[A-Za-z0-9_./-]{5,}\b)(?=[A-Za-z0-9_./-]*[A-Za-z])(?=[A-Za-z0-9_./-]*\d)[A-Za-z0-9_./-]+\b"
)


def _assistant_core_normalize_unit(value: str) -> str:
    unit = _normalize_unicode_advanced(str(value or "")).casefold()
    unit = unit.replace("−", "-").replace("–", "-").replace("—", "-")
    unit = re.sub(r"[\s.()·*x]", "", unit)
    unit = unit.replace("⁻¹", "-1")
    aliases = {
        "n-m": "nm",
        "n/m": "nm",
        "min-1": "rpm",
        "min1": "rpm",
        "1/min": "rpm",
        "giri/min": "rpm",
        "giri/minuto": "rpm",
        "db(a)": "dba",
        "dba": "dba",
        "°c": "degc",
        "cicli": "cycle",
        "cycle": "cycle",
        "cycles": "cycle",
        "pezzi": "piece",
        "piece": "piece",
        "pieces": "piece",
        "occorrenze": "occurrence",
        "occurrence": "occurrence",
        "occurrences": "occurrence",
    }
    return aliases.get(unit, unit)


def _assistant_core_units_equivalent(left: str, right: str) -> bool:
    a = _assistant_core_normalize_unit(left)
    b = _assistant_core_normalize_unit(right)
    return bool(a and b and a == b)


def _assistant_core_numeric_variants(value: str) -> set[str]:
    raw = str(value or "").strip().replace("\u00a0", " ")
    compact = re.sub(r"\s+", "", raw)
    out = {compact.casefold(), compact.replace(",", ".").casefold()}

    # Compare equivalent decimal renderings without dropping the decimal mark.
    # Deleting all punctuation would incorrectly make 50.00 equal to 5000.
    # Work on decimal strings, never binary floats or machine-specific values.
    def exact_decimal(token: str) -> str | None:
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token):
            return None
        negative = token.startswith("-")
        whole, _, fraction = token.lstrip("+-").partition(".")
        whole = whole.lstrip("0") or "0"
        fraction = fraction.rstrip("0")
        number = whole + ("." + fraction if fraction else "")
        return ("-" if negative and number != "0" else "") + number

    interpretations = []
    if compact.count(".") + compact.count(",") <= 1:
        interpretations.append(compact.replace(",", "."))
    elif "." in compact and "," in compact:
        decimal = "." if compact.rfind(".") > compact.rfind(",") else ","
        grouping = "," if decimal == "." else "."
        integer, _, fraction = compact.rpartition(decimal)
        if (re.fullmatch(r"[-+]?\d{1,3}(?:" + re.escape(grouping) + r"\d{3})+", integer)
                and fraction.isdigit()):
            interpretations.append(integer.replace(grouping, "") + "." + fraction)
    elif re.fullmatch(r"[-+]?\d{1,3}(?:[.]\d{3}){2,}", compact):
        interpretations.append(compact.replace(".", ""))
    elif re.fullmatch(r"[-+]?\d{1,3}(?:[,]\d{3}){2,}", compact):
        interpretations.append(compact.replace(",", ""))

    # Preserve the pre-existing two readings of a single separator followed by
    # three digits. This is not a unit conversion or resolution of ambiguous data.
    if re.fullmatch(r"[-+]?\d+[.,]\d{3}", compact):
        interpretations.append(re.sub(r"[.,]", "", compact))
    for interpretation in interpretations:
        canonical = exact_decimal(interpretation)
        if canonical is not None:
            out.add(canonical)
    return {x for x in out if x}

def _assistant_core_is_list_marker(value: str, start: int, end: int, raw: str) -> bool:
    line_start = value.rfind("\n", 0, start) + 1
    prefix = value[line_start:start]
    suffix = value[end:end + 5]
    return bool(
        not prefix.strip()
        and re.fullmatch(r"\d{1,2}", raw.strip())
        and re.match(r"\s*[.)-]\s+", suffix)
    )


def _assistant_core_claims(text: str) -> list[dict]:
    value = str(text or "")
    claims: list[dict] = []
    labeled_value_spans: list[tuple[int, int]] = []
    technical_unit_spans: list[tuple[int, int]] = []
    for labeled_match in _ASSISTANT_CORE_LABELED_VALUE_UNIT_RE.finditer(value):
        raw_value = str(labeled_match.group("value") or "").strip()
        raw_unit = str(labeled_match.group("unit") or "").strip()
        if not raw_value or not raw_unit:
            continue
        technical_unit_spans.append(labeled_match.span("unit"))
        labeled_value_spans.append(
            (labeled_match.start("value"), labeled_match.end("value"))
        )
        claims.append(
            {
                "kind": "number",
                "raw": raw_value,
                "variants": _assistant_core_numeric_variants(raw_value),
                "unit": _assistant_core_normalize_unit(raw_unit),
            }
        )
    dimension_spans = [
        (m.start("values"), m.end("values"), str(m.group("unit") or ""))
        for pattern in (
            _ASSISTANT_CORE_DIMENSION_CHAIN_RE,
            _ASSISTANT_CORE_DIMENSION_CHAIN_HEADING_RE,
        )
        for m in pattern.finditer(value)
    ]
    for match in _ASSISTANT_CORE_NUMBER_RE.finditer(value):
        raw = match.group(0)
        if any(
            span_start <= match.start() and match.end() <= span_end
            for span_start, span_end in labeled_value_spans
        ):
            continue
        if _assistant_core_is_list_marker(value, match.start(), match.end(), raw):
            continue
        # A number embedded in an alphanumeric designation (for example the
        # 1000 in AME-1000SN) is validated by the complete code claim, not as an
        # independent process value.
        if (
            re.match(r"[A-Za-z_]", value[match.end():match.end() + 1])
            or re.search(r"[A-Za-z_][./_-]$", value[max(0, match.start() - 3):match.start()])
        ):
            continue
        after = value[match.end():match.end() + 48]
        before = value[max(0, match.start() - 48):match.start()]
        unit_match = _ASSISTANT_CORE_TECH_UNIT_AFTER_RE.match(after)
        # In Italian ranges, the connector ``a`` ("da 0 a 50") is not the
        # electrical unit ampere. Do not classify it as ``A``; the complete
        # range parser below will attach the unit that follows the second value.
        if (
            unit_match
            and _assistant_core_normalize_unit(unit_match.group("unit")) == "a"
            and re.match(r"^\s*a\s*[-+]?\d", after, re.IGNORECASE)
        ):
            unit_match = None
        range_unit_match = None if unit_match else _ASSISTANT_CORE_RANGE_WITH_UNIT_RE.match(after)
        before_unit_match = None if (unit_match or range_unit_match) else _ASSISTANT_CORE_TECH_UNIT_BEFORE_RE.search(before)
        matched_unit = unit_match or range_unit_match or before_unit_match
        unit = str(matched_unit.group("unit") if matched_unit else "")
        if matched_unit:
            offset = (max(0, match.start() - 48) if before_unit_match
                      else match.end())
            technical_unit_spans.append((offset + matched_unit.start("unit"),
                                         offset + matched_unit.end("unit")))
        if not unit:
            for span_start, span_end, span_unit in dimension_spans:
                if span_start <= match.start() and match.end() <= span_end:
                    unit = span_unit
                    break
        has_decimal = bool(re.search(r"[.,]\d", raw))
        digits = re.sub(r"\D", "", raw)
        large_integer = len(digits) >= 3 and int(digits or "0") >= 100
        if not (unit or has_decimal or large_integer):
            continue
        claims.append(
            {
                "kind": "number",
                "raw": raw,
                "variants": _assistant_core_numeric_variants(raw),
                "unit": _assistant_core_normalize_unit(unit),
            }
        )
    for match in _ASSISTANT_CORE_CODE_RE.finditer(value):
        if any(start <= match.start() and match.end() <= stop
               for start, stop in technical_unit_spans):
            continue
        raw = match.group(0)
        claims.append(
            {
                "kind": "code",
                "raw": raw,
                "variants": {
                    raw.casefold(),
                    re.sub(r"[^0-9a-z]", "", _normalize_unicode_advanced(raw).casefold()),
                },
                "unit": "code",
            }
        )
    return claims


def _assistant_core_claim_supported(
    claim: dict,
    source_text: str,
    source_claims: Optional[list[dict]] = None,
) -> bool:
    if str(claim.get("kind") or "") == "code" or str(claim.get("unit") or "") == "code":
        source_cf = _normalize_unicode_advanced(str(source_text or "")).casefold()
        source_alnum = re.sub(r"[^0-9a-z]", "", source_cf)
        for raw_variant in (claim.get("variants") or set()):
            variant = str(raw_variant or "").casefold()
            if not variant:
                continue
            if variant in source_cf:
                return True
            variant_alnum = re.sub(r"[^0-9a-z]", "", variant)
            if variant_alnum and variant_alnum in source_alnum:
                return True
        return False

    candidates = source_claims if source_claims is not None else _assistant_core_claims(source_text)
    claim_variants = {str(v or "").casefold() for v in (claim.get("variants") or set()) if str(v or "")}
    claim_unit = str(claim.get("unit") or "")
    for source_claim in candidates:
        if str(source_claim.get("kind") or "") != "number":
            continue
        source_variants = {
            str(v or "").casefold()
            for v in (source_claim.get("variants") or set())
            if str(v or "")
        }
        if not (claim_variants & source_variants):
            continue
        source_unit = str(source_claim.get("unit") or "")
        if not claim_unit:
            return True
        if _assistant_core_units_equivalent(claim_unit, source_unit):
            return True
    return False


def _assistant_core_sentence_units(value: str) -> list[str]:
    units: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Do not split ordinary numbered steps such as "1. Mettere in sicurezza".
        parts = re.split(
            r"(?<=[!?])\s+|(?<=[A-Za-zÀ-ÖØ-öø-ÿ)])\.\s+(?=[A-ZÀ-ÖØ-Þ0-9])",
            line,
        )
        units.extend(part.strip() for part in parts if part.strip())
    return units


def _assistant_core_filter_unsupported_claim_sentences(text: str, source_text: str) -> tuple[str, list[str]]:
    value = str(text or "").strip()
    if not value:
        return "", []
    source_claims = _assistant_core_claims(source_text)
    kept: list[str] = []
    removed: list[str] = []
    for unit_text in _assistant_core_sentence_units(value):
        claims = _assistant_core_claims(unit_text)
        unsupported = [
            c
            for c in claims
            if not _assistant_core_claim_supported(c, source_text, source_claims)
        ]
        if unsupported:
            removed.extend(str(c.get("raw") or "") for c in unsupported)
            continue
        kept.append(unit_text)
    return "\n".join(kept).strip(), removed


def bind_validation(runtime: AskValidationRuntime) -> AskValidationRuntime:
    """Return a private ASK runtime; do not mutate main or the shared Core."""
    if type(runtime) is not AskValidationRuntime:
        raise TypeError("fresh ASK validation runtime required")
    return replace(runtime,
        _assistant_core_claims=_assistant_core_claims,
        _assistant_core_claim_supported=_assistant_core_claim_supported,
        _assistant_core_filter_unsupported_claim_sentences=_assistant_core_filter_unsupported_claim_sentences)
