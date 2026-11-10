"""ASK-only numerical/unit claim normalization.

The inherited unit fix is deliberately NOT installed in main's shared legacy
parser: that parser is still used by Root Cause. Only the protected ASK
intake runtime factory binds this module. There are no model calls, permissions,
query policies or source preferences here. Exact W/kW comparisons require an
unambiguous scalar on both sides; unit names alone never establish equality.

All numeric precision, bracket matching and rejection rules are those of the
reviewed unit candidate. The legacy parser remains frozen until its own phase.
"""
from __future__ import annotations
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional
from ..ingest.text_pdf import normalize_unicode_advanced as _normalize_unicode_advanced
from .validation import AskValidationRuntime
from . import table_units

_ASSISTANT_CORE_UNIT_TOKEN = r"(?:mm(?:²|2|³|3)?|cm|kg|bar|psi|cfm|l\s*/\s*min|m(?:³|3)\s*/\s*min|k\s*pa|m\s*pa|pa|k\s*hz|hz|k\s*w|k\s*n|n\s*(?:[·*x]\s*)?m|nm|rpm|giri\s*/?\s*min(?:uto)?|min\s*(?:-?1|⁻¹)|1\s*/\s*min|ms|°\s*c|°|cst|db\s*\(?a\)?|cycles?|cicli|pieces?|pezzi|occurrences?|occorrenze|ore|ora|hours?|hrs?|%|v|a|w|m|g|s|h)"
_ASSISTANT_CORE_NUMBER_ATOM = r"[-+]?(?:(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:[.,]\d+)?|[.,]\d+)(?!\d)"
# A slash suffix is a different dimension even when spaces surround it; a dot
# followed by a word continues a designation. Stars are accepted only as a
# complete Markdown closing delimiter, never multiplication/exponent prefixes.
_ASSISTANT_CORE_UNIT_END = r"""(?!\s*/\s*\w)(?!\.\w)(?=$|[\s,.;:/)\]"'’”]|\*{1,2}(?=$|[\s,.;:/)\]"'’”]))"""
_ASSISTANT_CORE_TECH_UNIT_AFTER_RE = re.compile(
    # A unit immediately after a value may use matched square brackets, as in
    # technical PDF tables. Require a complete supported unit and delimiter.
    rf"^\s*(?:(?:circa|about|approx(?:imately)?|ca\.?|~)\s*)?(?P<bracket>\[\s*)?(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})(?(bracket)\s*\]){_ASSISTANT_CORE_UNIT_END}",
    re.IGNORECASE,
)
_ASSISTANT_CORE_RANGE_WITH_UNIT_RE = re.compile(
    rf"^\s*(?:-|–|—|÷|to|a)\s*{_ASSISTANT_CORE_NUMBER_ATOM}\s*(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN}){_ASSISTANT_CORE_UNIT_END}",
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
_ASSISTANT_CORE_NUMBER_RE = re.compile(
    rf"(?<![\w]){_ASSISTANT_CORE_NUMBER_ATOM}"
)
# XLSX structured rows keep values and units in adjacent labeled fields, e.g.
# ``Valore: 17.6 | Unità: bar``.  Recognize that bounded representation as one
# technical claim without changing ordinary prose parsing.
_ASSISTANT_CORE_LABELED_VALUE_UNIT_RE = re.compile(
    rf"\b(?:valore|value)\s*:\s*(?P<value>{_ASSISTANT_CORE_NUMBER_ATOM})"
    rf"\s*\|\s*(?:unità|unita|unit)\s*:\s*(?P<unit>{_ASSISTANT_CORE_UNIT_TOKEN})"
    rf"""(?=$|[\s|,.;:/)\]"'’”])""",
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
        "ora": "h",
        "ore": "h",
        "hour": "h",
        "hours": "h",
        "hr": "h",
        "hrs": "h",
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
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", token):
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


def _exact_power_quantity(raw: str, unit: str) -> Optional[Fraction]:
    """Compare explicit power scalars exactly, without locale/float guesses.

    Legacy same-unit matching retains its existing numeric variants. Crossing
    units is stricter: a single separator with three trailing digits may be
    decimal or grouping, so cannot justify a conversion. No scientific notation,
    arithmetic, unknown dimension or Decimal-context rounding is admitted.
    """
    scale = {'w': 1, 'kw': 1000}.get(_assistant_core_normalize_unit(unit))
    value = str(raw or '').strip().replace('\u00a0', ' ')
    if scale is None or not value or len(value) > 256:
        return None
    if ' ' in value:
        if not re.fullmatch(r'[-+]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?', value):
            return None
        value = value.replace(' ', '')
    if re.fullmatch(r'[-+]?\d+', value):
        canonical = value
    elif value.count('.') + value.count(',') == 1:
        if (not re.fullmatch(r'[-+]?(?:\d+[.,]\d+|[.,]\d+)', value)
                or re.fullmatch(r'[-+]?\d+[.,]\d{3}', value)):
            return None
        canonical = value.replace(',', '.')
    elif '.' in value and ',' in value:
        decimal = '.' if value.rfind('.') > value.rfind(',') else ','
        grouping = ',' if decimal == '.' else '.'
        integer, _, fraction = value.rpartition(decimal)
        if not (re.fullmatch(r'[-+]?\d{1,3}(?:' + re.escape(grouping) + r'\d{3})+', integer)
                and fraction.isdigit()):
            return None
        canonical = integer.replace(grouping, '') + '.' + fraction
    elif re.fullmatch(r'[-+]?\d{1,3}(?:\.\d{3}){2,}', value):
        canonical = value.replace('.', '')
    elif re.fullmatch(r'[-+]?\d{1,3}(?:,\d{3}){2,}', value):
        canonical = value.replace(',', '')
    else:
        return None
    try:
        # Fraction(Decimal) is exact. Multiplying Decimal directly would use
        # its current precision and can make unequal long scalars look equal.
        return Fraction(Decimal(canonical)) * scale
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _assistant_core_is_list_marker(value: str, start: int, end: int, raw: str) -> bool:
    line_start = value.rfind("\n", 0, start) + 1
    prefix = value[line_start:start]
    suffix = value[end:end + 5]
    return bool(
        not prefix.strip()
        and re.fullmatch(r"\d{1,2}", raw.strip())
        and re.match(r"\s*[.)-]\s+", suffix)
    )


def _assistant_core_claims(text: str, *, _parse_serialized_cells: bool = True) -> list[dict]:
    value = str(text or "")
    claims: list[dict] = []
    table_values, table_value_spans, conflicting_table_spans = (table_units.numeric_cells(value,
        unit_token=_ASSISTANT_CORE_UNIT_TOKEN, number_atom=_ASSISTANT_CORE_NUMBER_ATOM,
        normalize_unit=_assistant_core_normalize_unit) if _parse_serialized_cells else ([], [], []))
    for raw, unit in table_values:
        claims.append({"kind": "number", "raw": raw,
            "variants": _assistant_core_numeric_variants(raw),
            "unit": _assistant_core_normalize_unit(unit)})
    table_value_spans = table_value_spans + conflicting_table_spans
    if _parse_serialized_cells:
        for literal, span in table_units.decoded_value_cells(value):
            if span not in table_value_spans:
                # JSON escapes are representation, not source text. Parse each
                # decoded cell separately; keep typed/conflicting cells under
                # the existing same-column rules and never parse nested markers.
                claims.extend(_assistant_core_claims(literal, _parse_serialized_cells=False))
                table_value_spans.append(span)
    labeled_value_spans: list[tuple[int, int]] = []
    technical_unit_spans: list[tuple[int, int]] = []
    for labeled_match in _ASSISTANT_CORE_LABELED_VALUE_UNIT_RE.finditer(value):
        if any(start <= labeled_match.start() and labeled_match.end() <= end
               for start, end in table_value_spans):
            continue
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
        if any(start <= match.start() and match.end() <= end
               for start, end in table_value_spans):
            continue
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
        if re.search(r"[A-Za-z_][./_-]$", value[max(0, match.start() - 3):match.start()]):
            continue
        after = value[match.end():match.end() + 48]
        before = value[max(0, match.start() - 48):match.start()]
        unit_match = _ASSISTANT_CORE_TECH_UNIT_AFTER_RE.match(after)
        # Compact source quantities (1kW, .75l/min, 100psi) are quantities when
        # the complete suffix is a recognized unit. Keep composite model/code
        # identifiers in the exact-code path; never take a unit prefix from
        # an unknown suffix such as 1kWh or 100psi_model.
        if re.match(r"[A-Za-z_]", after[:1]) and unit_match is None:
            continue
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
            if unit_match:
                # The numeric+unit token is already checked dimensionally.
                # Do not also reinterpret .75l/min as the identifier 75l/min.
                technical_unit_spans.append((match.start(), match.end() + unit_match.end("unit")))
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
        if any(start <= match.start() and match.end() <= end for start, end in table_value_spans):
            continue
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
        # A compact numeric unit (65bar) can also match the legacy identifier
        # grammar. A conflicting PDF cell cannot become support through that
        # second representation. Mask only those exact internal lookup spans;
        # the admitted source text and rendered response stay unchanged.
        if '[PDF_TABLE_' in source_text:
            _, _, conflicts = table_units.numeric_cells(source_text,
                unit_token=_ASSISTANT_CORE_UNIT_TOKEN, number_atom=_ASSISTANT_CORE_NUMBER_ATOM,
                normalize_unit=_assistant_core_normalize_unit)
            for start, end in sorted(conflicts, reverse=True):
                source_text = source_text[:start] + (' ' * (end - start)) + source_text[end:]
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
    exact_power = _exact_power_quantity(claim.get('raw'), claim_unit)
    for source_claim in candidates:
        if str(source_claim.get("kind") or "") != "number":
            continue
        source_variants = {
            str(v or "").casefold()
            for v in (source_claim.get("variants") or set())
            if str(v or "")
        }
        source_unit = str(source_claim.get("unit") or "")
        if claim_variants & source_variants:
            if not claim_unit:
                return True
            if _assistant_core_units_equivalent(claim_unit, source_unit):
                return True
        if (exact_power is not None and claim_unit != source_unit
                and exact_power == _exact_power_quantity(source_claim.get('raw'), source_unit)):
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


_FINITE_DEPENDENT_PREFIX = re.compile(
    r"^\s*(?:for|on|in|under|from|per|su|sul|sulla|nel|nella|"
    r"it\s+is\s+(?:false|untrue|incorrect)|(?:è|e)\s+falso)\b", re.I)
_FINITE_SCOPE_OPERATOR = re.compile(
    r"\b(?:if|when|unless|only|except|provided|assuming|false|untrue|because|while|until|before|after|without|"
    r"se|quando|qualora|solo|salvo|purché|falso|perché|poiché|finché|prima|dopo|senza)\b|"
    r"not\s+true|non\s+(?:è|e)\s+vero", re.I)
_FINITE_CONTEXT_SWITCH = re.compile(
    r"\b(?:for|on|in|under|from|per|su|sul|sulla|nel|nella)\s+"
    r"(?:(?:a|an|the|un|uno|una|il|lo|la|un'altra|un'altro)\s+)?"
    r"(?:other|another|different|alternative|altro|altra|altri|altre|diverso|diversa|"
    r"model|machine|assembly|modello|macchina|gruppo)\b", re.I)
_FINITE_DENIAL = re.compile(
    r"\b(?:is|are)\s+(?:not\s+(?:permitted|allowed|compatible|admissible)|prohibited|forbidden)\b|"
    r"\b(?:must|may|can)\s+not\s+(?:be\s+)?(?:loaded|used)\b|"
    r"\bnon\s+(?:è|e|sono)\s+(?:consentit[oaie]|ammess[oaie]|compatibil[ei])\b|"
    r"\b(?:è|e|sono)\s+(?:vietat[oaie]|proibit[oaie])\b|"
    r"^\s*(?:do\s+not\s+(?:load|use)|non\s+(?:caricare|usare|utilizzare))\b", re.I)
_FINITE_FACT_PREDICATE = re.compile(r"\b(?:is|are|equals|è|e|sono)\b|\bpari\s+a\b", re.I)
_FINITE_DEPENDENT_FACT = re.compile(
    r"^\s*(?:it|this|that|these|those|they|its|quest[oaie]|ciò|esso|essa|essi|esse|tale|tali)\b", re.I)


def _finite_denial_and_fact(denial: str, fact: str, source_text: str, source_claims: list[dict]) -> bool:
    """A narrow clause boundary, not permission for the denied quantity."""
    if (_FINITE_DEPENDENT_PREFIX.search(denial) or _FINITE_SCOPE_OPERATOR.search(denial)
            or _FINITE_CONTEXT_SWITCH.search(denial)
            or not _FINITE_DENIAL.search(denial) or _FINITE_DEPENDENT_FACT.search(fact)):
        return False
    denied = _assistant_core_claims(denial)
    denied_numbers = [c for c in denied if c["kind"] == "number"]
    kept = _assistant_core_claims(fact)
    kept_numbers = [c for c in kept if c["kind"] == "number"]
    if (len(denied_numbers) != 1 or not denied_numbers[0].get("unit")
            or _assistant_core_claim_supported(denied_numbers[0], source_text, source_claims)
            or any(c["kind"] != "number" and not _assistant_core_claim_supported(c, source_text, source_claims)
                   for c in denied)
            or len(kept_numbers) != 1 or not kept_numbers[0].get("unit")
            or not all(_assistant_core_claim_supported(c, source_text, source_claims) for c in kept)):
        return False
    # A bare value, continuation or shared-unit fragment is not an autonomous
    # factual statement. Never infer omitted units or copy a lost prerequisite.
    value_at = fact.find(str(kept_numbers[0]["raw"]))
    return value_at > 0 and _FINITE_FACT_PREDICATE.search(fact[:value_at]) is not None


def _finite_outer_delimiters(text: str, delimiters: str = ":;") -> list[int]:
    stack: list[str] = []
    quote = ""
    positions: list[int] = []
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i-1] != "\\"):
                quote = ""
            continue
        if (char in "\"'“‘" or ord(char) == 96) and not (
                char == "'" and i and i+1 < len(text) and text[i-1].isalnum() and text[i+1].isalnum()):
            quote = {"“": "”", "‘": "’"}.get(char, char)
        elif char == "”" or (char == "’" and not (
                i and i+1 < len(text) and text[i-1].isalnum() and text[i+1].isalnum())):
            return []
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or "([{".index(stack.pop()) != ")]}".index(char):
                return []
        elif char in delimiters and not stack and i+1 < len(text) and text[i+1].isspace():
            positions.append(i)
    return [] if quote or stack else positions


def finite_factual_claim_filter(text: str, source_text: str) -> tuple[str, list[str]]:
    """Opt-in finite factual ASK only. All unsupported claims are still removed.

    Colon/semicolon splits require an explicit unsupported denial and one
    autonomous source-supported quantity statement. Scope/negation antecedents,
    quotations, identifiers and shared ranges remain in one unit and fail closed.
    Whole-question review and final text binding remain separate obligations.
    """
    source_claims = _assistant_core_claims(source_text)
    units: list[str] = []
    for unit in _assistant_core_sentence_units(str(text or "")):
        pieces: list[str] = []
        start = 0
        outer_periods = _finite_outer_delimiters(unit, ".")
        for match in re.finditer(r"(?P<mark>\*\*|__|\*)\.\s+(?=[A-ZÀ-ÖØ-Þ0-9])", unit):
            mark = match.group("mark")
            prefix = unit[:match.start()+len(mark)]
            if (match.start()+len(mark) in outer_periods
                    and prefix.count(mark) >= 2 and prefix.count(mark) % 2 == 0
                    and not _FINITE_DEPENDENT_PREFIX.search(prefix)
                    and not _FINITE_SCOPE_OPERATOR.search(prefix)
                    and not _FINITE_CONTEXT_SWITCH.search(prefix)
                    and not any(c["kind"] == "code" and not _assistant_core_claim_supported(
                        c, source_text, source_claims) for c in _assistant_core_claims(prefix))):
                pieces.append(unit[start:match.start()+len(mark)+1].strip())
                start = match.end()
        pieces.append(unit[start:].strip())
        for piece in pieces:
            divided = False
            for pos in _finite_outer_delimiters(piece):
                left, right = piece[:pos].strip(), piece[pos+1:].strip()
                if (_finite_denial_and_fact(left, right, source_text, source_claims)
                        or _finite_denial_and_fact(right, left, source_text, source_claims)):
                    units.extend((left, right))
                    divided = True
                    break
            if not divided:
                units.append(piece)
    return _assistant_core_filter_unsupported_claim_sentences("\n".join(units), source_text)


def bind_validation(runtime: AskValidationRuntime) -> AskValidationRuntime:
    """Return a private ASK runtime; do not mutate main or the shared Core."""
    if type(runtime) is not AskValidationRuntime:
        raise TypeError("fresh ASK validation runtime required")
    return replace(runtime,
        _assistant_core_claims=_assistant_core_claims,
        _assistant_core_claim_supported=_assistant_core_claim_supported,
        _assistant_core_filter_unsupported_claim_sentences=_assistant_core_filter_unsupported_claim_sentences)
