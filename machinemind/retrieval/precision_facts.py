"""Deterministic scalar-fact rescue for MachineMind ASK.

The normal semantic retrieval/synthesis pipeline remains authoritative.  This
module is intentionally conservative and is called only after ASK has already
failed closed with ``no_sources``.  It may recover one unambiguous scalar
property/value pair from authorized ``document_pages``.  It never guesses,
uses outside knowledge, or answers diagnostic/procedural requests.

The resolver is vocabulary-independent with respect to technical properties:
it does not know that "peso" means weight or that "pressione" means pressure.
It extracts the requested property words from the question, matches those words
against labels printed in the indexed source, and requires the numeric value and
engineering unit to be attached to that same label.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Callable, Optional, Sequence, Mapping
import re
import copy
import hashlib
import unicodedata


# Question framing only.  Technical property words are deliberately absent.
_QUERY_STOPWORDS = {
    # Italian
    "a", "ad", "ai", "al", "alla", "alle", "allo", "anche", "associata",
    "associato", "che", "codice", "come", "con", "da", "dal", "dalla",
    "dalle", "dallo", "dell", "dei", "del", "della", "delle", "dello", "di", "dimmi",
    "documento", "e", "ed", "file", "foglio", "gli", "i", "il", "in",
    "indica", "indicami", "indicata", "indicato", "l", "la", "le", "lo", "manuale",
    "mi", "modello", "nel", "nella", "nelle", "nello", "per", "qual", "quale",
    "quali", "quanto", "quanta", "quanti", "quante", "relativa", "relativo",
    "riporta", "riportami", "secondo", "specifica", "specificata", "specificato",
    "su", "sul", "sulla", "un", "una", "uno", "valore", "è", "e",
    # English
    "a", "an", "and", "associated", "code", "document", "file", "for", "from",
    "give", "how", "in", "indicated", "is", "manual", "me", "model", "of", "on",
    "report", "requested", "sheet", "show", "specified", "tell", "the", "to",
    "value", "what", "which", "with",
}

# Explicitly non-factual/question intents.  These are generic interaction words,
# not technical vocabulary.  A match causes this optional rescue to stay silent.
_NON_SCALAR_MARKERS = (
    r"\b(?:perch[eé]|why)\b",
    r"\b(?:come\s+(?:si|fare|eseguire)|how\s+to)\b",
    r"\b(?:procedura|procedure|passaggi|steps?|istruzioni|instructions?)\b",
    r"\b(?:cause?|causes?|diagnos|guasto|fault|anomali|malfunzion)\w*\b",
    r"\b(?:elenca|lista|list|tutti|tutte|all|quali\s+sono|what\s+are|confronta|compare|differenza|difference)\b",
    r"\b(?:bypass|ponticell|disattiv|esclud|override|defeat)\w*\b",
)

# A source-constrained question must stay with the existing source-selection
# pipeline.  This rescue has no source-title resolver of its own and therefore
# fails closed rather than answering from a different manual/document.
_EXPLICIT_SOURCE_MARKERS = (
    r"\b(?:secondo|nel|nella|nei|nelle|dal|dalla|from|in|according\s+to)\s+"
    r"(?:(?:il|lo|la|i|gli|le|un|una|the|a|an)\s+)?"
    r"(?:manuale|manual|documento|document|file|pdf)\b",
    r"\b(?:manuale|manual|documento|document|file|pdf)\s+"
    r"(?:dice|riporta|indica|states?|reports?|says?)\b",
)

# Engineering units only; the requested *property* is never hard-coded.
_UNIT_TOKEN = (
    r"(?:kg|g|mg|t|ton|lb|lbs|mm(?:2|3|²|³)?|cm(?:2|3|²|³)?|m(?:2|3|²|³)?|"
    r"km|um|µm|nm|ml|cl|dl|l|litri?|liters?|s|sec|secs|secondi?|seconds?|ms|"
    r"min|minuti?|minutes?|h|ore|hours?|hz|khz|mhz|rpm|min\s*-?1|1\s*/\s*min|"
    r"pa|kpa|mpa|bar|mbar|psi|n|kn|nm|n\s*[·*x/]\s*m|w|kw|mw|v|kv|a|ma|"
    r"°\s*c|°c|celsius|f|°\s*f|%|percento|percent|db(?:\s*\(?a\)?)?|cst)"
)
_NUMBER_TOKEN = r"[-+]?(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:[.,]\d+)?"
_VALUE_WITH_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<number>{_NUMBER_TOKEN})\s*(?P<unit>{_UNIT_TOKEN})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_UNIT_BEFORE_VALUE_RE = re.compile(
    rf"(?:\[|\()\s*(?P<unit>{_UNIT_TOKEN})\s*(?:\]|\))\s*(?P<number>{_NUMBER_TOKEN})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_./-]{4,}(?![A-Za-z0-9]))"
    r"(?=[A-Za-z0-9_./-]*[A-Za-z])(?=[A-Za-z0-9_./-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9_./-]*[A-Za-z0-9](?![A-Za-z0-9])"
)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


@dataclass(frozen=True)
class PrecisionFactRuntime:
    connect_db: Callable[[], Any]
    build_scope_where: Callable[..., tuple[str, list[Any]]]
    fetch_file_map: Optional[Callable[[str, list[str]], dict[str, str]]] = None
    company_general_machine_sentinel: str = "__company_general__"
    page_text_chars: int = 12000
    page_scan_limit: int = 220


@dataclass(frozen=True)
class PropertyQuery:
    normalized: str
    property_terms: tuple[str, ...]
    property_phrase: str
    code_tokens: tuple[str, ...]


@dataclass(frozen=True)
class FactPair:
    label: str
    value: str
    number: str
    unit: str
    canonical_number: str
    canonical_unit: str
    page_number: int
    bubble_document_id: str
    machine_id: str
    section: str
    context: str
    table_density: int
    same_line: bool
    label_coverage: float
    label_precision: float
    phrase_similarity: float
    score: float


@dataclass(frozen=True)
class PrecisionFactResolution:
    label: str
    value: str
    number: str
    unit: str
    canonical_number: str
    canonical_unit: str
    bubble_document_id: str
    page_number: int
    machine_id: str
    snippet: str
    property_terms: tuple[str, ...]
    supporting_pages: tuple[int, ...]
    score: float


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _word_key(value: str) -> str:
    token = _strip_accents(_normalize_text(value)).casefold()
    token = re.sub(r"[^a-z0-9]+", "", token)
    # Conservative inflection folding.  It never maps different roots/synonyms;
    # it only removes common plural endings after a sufficiently long stem.
    if len(token) >= 6:
        for suffix in ("mente", "zioni", "zione", "ments", "ment", "es", "s", "i", "e"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
    return token


def _words(value: str) -> list[str]:
    return [key for raw in _WORD_RE.findall(_normalize_text(value)) if (key := _word_key(raw))]


def _is_non_scalar_request(query: str) -> bool:
    low = _strip_accents(_normalize_text(query)).casefold()
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in _NON_SCALAR_MARKERS):
        return True
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in _EXPLICIT_SOURCE_MARKERS):
        return True
    # Phase 4A resolves scalar facts from ordinary technical documents only.
    # XLSX rows and Bubble structured records have their own exact-row/family
    # contracts and must never be bypassed by this PDF/manual rescue.
    if re.search(r"\b(?:xlsx|excel|spreadsheet|workbook|worksheet|foglio(?:\s+di\s+calcolo)?)\b", low):
        return True
    return False


def _ordinary_document_page(text: str) -> bool:
    low = _normalize_text(text).casefold()[:500]
    if "document_file_type: xlsx" in low or "extraction_mode: xlsx" in low:
        return False
    if re.search(r"(?:^|\n)source_type\s*:", low):
        return False
    return True


def extract_property_query(query: str) -> Optional[PropertyQuery]:
    """Extract technical property words without knowing any property vocabulary."""
    text = _normalize_text(query)
    if not text or _is_non_scalar_request(text):
        return None

    code_tokens = tuple(dict.fromkeys(match.group(0) for match in _CODE_RE.finditer(text)))
    without_codes = text
    for code in sorted(code_tokens, key=len, reverse=True):
        without_codes = re.sub(re.escape(code), " ", without_codes, flags=re.IGNORECASE)

    terms: list[str] = []
    seen: set[str] = set()
    for raw in _WORD_RE.findall(without_codes):
        raw_key = _word_key(raw)
        stop_key = _strip_accents(raw).casefold()
        if not raw_key or stop_key in _QUERY_STOPWORDS or raw_key in _QUERY_STOPWORDS:
            continue
        if raw_key in seen:
            continue
        seen.add(raw_key)
        terms.append(raw_key)

    # A precision rescue requires an actual property label.  Very long residual
    # text is likely explanatory/diagnostic and remains with the normal pipeline.
    if not terms or len(terms) > 6:
        return None

    return PropertyQuery(
        normalized=_strip_accents(text).casefold(),
        property_terms=tuple(terms),
        property_phrase=" ".join(terms),
        code_tokens=code_tokens,
    )


# Kept outside AssistantCoreDecision to avoid changing public/core contracts.
SCALAR_TARGET_KEY = "precision_scalar_target"
SCALAR_TARGET_POLICY = "scalar-target-label-v1"
SCALAR_TARGET_INSTRUCTION = (
    " For ASK only, also return SCALAR_TARGET. Set state=single_scalar only when "
    "the user requests one scalar numerical property. property_labels must be "
    "short equivalent source-label phrases, not sentences or retrieval queries. "
    "Keep the affected entity kind (whole machine versus a component) and every "
    "requested qualifier such as maximum/nominal/net/gross. Do not add unrequested "
    "transport, accessories or other alternative contexts. Up to four faithful "
    "Italian/English labels may be used for the SAME property. No model codes, "
    "values or units inside a label. query_quote must be copied verbatim from "
    "the part of USER_REQUEST asking for that property. For multiple properties, "
    "tables, spreadsheet rows, procedures, diagnosis or unknown intent, return "
    "state=not_applicable, property_labels=[], query_quote=''."
)


def router_schema_with_scalar_target(schema: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(schema))
    body = out["schema"]
    body["properties"]["scalar_target"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "state": {"type": "string", "enum": ["single_scalar", "not_applicable"]},
            "property_labels": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "query_quote": {"type": "string"},
        },
        "required": ["state", "property_labels", "query_quote"],
    }
    body["required"] = list(body.get("required") or []) + ["scalar_target"]
    return out


def scalar_target_from_router(query: str, router: Mapping[str, Any]) -> dict[str, Any]:
    """Validate provenance of a label proposal; never accept a value from a model."""
    proposed = router.get("scalar_target")
    if (not isinstance(proposed, Mapping)
        or router.get("effective_mode") != "ask"
        or router.get("information_task") != "numeric_specification"
        or router.get("router_degraded")
        or proposed.get("state") != "single_scalar"
        or _is_non_scalar_request(query)):
        return {}
    quote = _normalize_text(str(proposed.get("query_quote") or ""))
    if len(quote) < 5 or quote.casefold() not in _normalize_text(query).casefold():
        return {}
    labels = proposed.get("property_labels")
    if not isinstance(labels, list) or not 1 <= len(labels) <= 4:
        return {}
    clean: list[str] = []
    for raw in labels:
        if not isinstance(raw, str) or len(raw) > 120 or re.search(r"[\d?!;]", raw):
            return {}
        candidate = extract_property_query(raw)
        if candidate is None or candidate.code_tokens or len(candidate.property_terms) > 6:
            return {}
        if raw not in clean:
            clean.append(raw)
    return {
        "policy_version": SCALAR_TARGET_POLICY,
        "query_sha256": hashlib.sha256(_normalize_text(query).casefold().encode()).hexdigest(),
        "property_labels": clean,
        "query_quote": quote,
    }


def property_queries_from_scalar_target(
    original_query: str, contract: Optional[Mapping[str, Any]]
) -> tuple[PropertyQuery, ...]:
    if not isinstance(contract, Mapping) or _is_non_scalar_request(original_query):
        return ()
    if (contract.get("effective_mode") != "ask"
        or contract.get("information_task") != "numeric_specification"
        or contract.get("router_degraded")
        or contract.get("evidence_policy") != "machine_sources_required"
        or contract.get("source_type_policy") == "require"):
        return ()
    requirements = set(contract.get("required_answer_types") or [])
    if "numeric_value" not in requirements or requirements - {"numeric_value", "explanation"}:
        return ()
    target = contract.get(SCALAR_TARGET_KEY)
    if not isinstance(target, Mapping) or target.get("policy_version") != SCALAR_TARGET_POLICY:
        return ()
    fingerprint = hashlib.sha256(_normalize_text(original_query).casefold().encode()).hexdigest()
    if target.get("query_sha256") != fingerprint:
        return ()
    validated = scalar_target_from_router(original_query, {
        "effective_mode": "ask", "information_task": "numeric_specification",
        "scalar_target": {"state": "single_scalar", "property_labels": target.get("property_labels"),
                          "query_quote": target.get("query_quote")},
    })
    if not validated:
        return ()
    codes = tuple(dict.fromkeys(m.group(0) for m in _CODE_RE.finditer(_normalize_text(original_query))))
    out: list[PropertyQuery] = []
    for label in validated["property_labels"]:
        prop = extract_property_query(label)
        if prop is not None:
            out.append(PropertyQuery(_strip_accents(_normalize_text(original_query)).casefold(),
                                     prop.property_terms, prop.property_phrase, codes))
    return tuple(out)


def property_query_from_answer_contract(
    original_query: str, contract: Optional[Mapping[str, Any]]
) -> Optional[PropertyQuery]:
    """Use one scalar facet already resolved by the semantic router.

    This is not a new semantic parser or a property-synonym dictionary. It lets
    the existing router separate conversational framing/inflection from a scalar
    property. Source-only instructions, original model/code restrictions and the
    exact label/value ambiguity gate remain authoritative. Multi-part tasks and
    competing facets deliberately stay outside this optional rescue.
    """
    if not isinstance(contract, Mapping) or _is_non_scalar_request(original_query):
        return None
    if contract.get("effective_mode") != "ask" or contract.get("information_task") != "numeric_specification":
        return None
    if contract.get("router_degraded") or contract.get("evidence_policy") != "machine_sources_required":
        return None
    if contract.get("source_type_policy") == "require":
        return None
    requirements = set(contract.get("required_answer_types") or [])
    if "numeric_value" not in requirements or requirements - {"numeric_value", "explanation"}:
        return None
    facets = contract.get("facet_queries")
    if not isinstance(facets, (list, tuple)) or len(facets) != 1:
        return None
    facet = facets[0]
    if not isinstance(facet, Mapping) or facet.get("answer_type") != "numeric_value" or facet.get("must_cover") is not True:
        return None
    canonical = extract_property_query(str(facet.get("facet") or ""))
    if canonical is None:
        return None
    original_codes = tuple(dict.fromkeys(m.group(0) for m in _CODE_RE.finditer(_normalize_text(original_query))))
    original_keys = {_identifier_key(c) for c in original_codes}
    if any(_identifier_key(c) not in original_keys for c in canonical.code_tokens):
        return None
    return PropertyQuery(
        normalized=_strip_accents(_normalize_text(original_query)).casefold(),
        property_terms=canonical.property_terms,
        property_phrase=canonical.property_phrase,
        code_tokens=original_codes,
    )


def _canonical_unit(unit: str) -> str:
    value = _strip_accents(_normalize_text(unit)).casefold()
    value = re.sub(r"[\s.()·*x]", "", value)
    aliases = {
        "kgs": "kg", "lbs": "lb", "tons": "ton", "tonnellate": "t",
        "secondo": "s", "secondi": "s", "second": "s", "seconds": "s", "sec": "s", "secs": "s",
        "minuto": "min", "minuti": "min", "minute": "min", "minutes": "min",
        "ora": "h", "ore": "h", "hour": "h", "hours": "h",
        "litro": "l", "litri": "l", "liter": "l", "liters": "l",
        "°c": "degc", "celsius": "degc", "°f": "degf",
        "percento": "%", "percent": "%",
        "min1": "rpm", "1/min": "rpm",
        "n/m": "nm", "n-m": "nm",
        "dba": "dba",
    }
    return aliases.get(value, value)


def _canonical_number(raw: str) -> Optional[str]:
    value = _normalize_text(raw).replace(" ", "")
    if not value:
        return None
    sign = ""
    if value[:1] in "+-":
        sign, value = value[0], value[1:]
    if not value or not re.search(r"\d", value):
        return None

    if "," in value and "." in value:
        # The right-most separator is decimal; the other is grouping.
        decimal_sep = "," if value.rfind(",") > value.rfind(".") else "."
        grouping_sep = "." if decimal_sep == "," else ","
        value = value.replace(grouping_sep, "").replace(decimal_sep, ".")
    elif "," in value or "." in value:
        sep = "," if "," in value else "."
        parts = value.split(sep)
        if len(parts) > 2:
            # Repeated three-digit groups are thousands separators.
            if all(len(part) == 3 for part in parts[1:]):
                value = "".join(parts)
            else:
                return None
        elif len(parts) == 2:
            left, right = parts
            # A single separator followed by exactly three digits is ambiguous.
            # Industrial manuals overwhelmingly use it as a thousands separator
            # for integer mass/capacity values; retain both interpretations later
            # through the raw display, but use the integer reading for grouping.
            if len(right) == 3 and left and left != "0" and len(left) <= 3:
                value = left + right
            else:
                value = left + "." + right
    try:
        number = Decimal(sign + value)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _value_matches(line: str) -> list[tuple[re.Match[str], str, str]]:
    matches: list[tuple[re.Match[str], str, str]] = []
    for regex in (_VALUE_WITH_UNIT_RE, _UNIT_BEFORE_VALUE_RE):
        for match in regex.finditer(line):
            number = str(match.group("number") or "").strip()
            unit = str(match.group("unit") or "").strip()
            if number and unit:
                matches.append((match, number, unit))
    matches.sort(key=lambda item: item[0].start())
    return matches


def _value_is_composite(line: str, match: re.Match[str]) -> bool:
    """Reject one endpoint extracted from a range, alternative or dimension chain."""
    before = _normalize_text(line[: match.start()])
    after = _normalize_text(line[match.end():])
    connector = r"(?:/|[-–—÷]|[x×]|to|a)"
    if re.search(rf"{_NUMBER_TOKEN}\s*{connector}\s*$", before, flags=re.IGNORECASE):
        return True
    if re.match(rf"^\s*{connector}\s*{_NUMBER_TOKEN}", after, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_label(line: str) -> bool:
    text = _normalize_text(line).strip(" :;.-")
    if not text or len(text) > 180:
        return False
    words = _words(text)
    if not words or len(words) > 14:
        return False
    if _value_matches(text):
        return False
    # Reject dates, revisions and isolated headings made only of numbers/symbols.
    if re.fullmatch(r"(?:rev\.?\s*)?[0-9./-]+[a-z]?", _strip_accents(text).casefold()):
        return False
    return True


def _nearest_section(lines: Sequence[str], index: int) -> str:
    for pos in range(index - 1, max(-1, index - 12), -1):
        candidate = _normalize_text(lines[pos]).strip(" :;.-")
        if not candidate or len(candidate) > 100:
            continue
        letters = [ch for ch in candidate if ch.isalpha()]
        if len(letters) >= 4 and candidate.upper() == candidate:
            return candidate
    return ""


def _page_table_density(lines: Sequence[str]) -> int:
    density = 0
    compact = [_normalize_text(line) for line in lines if _normalize_text(line)]
    for index, line in enumerate(compact):
        if _value_matches(line):
            prefix = line[: _value_matches(line)[0][0].start()].strip(" :;-_")
            if _looks_like_label(prefix):
                density += 1
            elif index > 0 and _looks_like_label(compact[index - 1]):
                density += 1
    return density


def _label_metrics(label: str, query: PropertyQuery) -> tuple[float, float, float, tuple[str, ...]]:
    label_terms = tuple(dict.fromkeys(_words(label)))
    query_terms = tuple(query.property_terms)
    if not label_terms or not query_terms:
        return 0.0, 0.0, 0.0, ()
    label_set = set(label_terms)
    query_set = set(query_terms)
    matched = tuple(term for term in query_terms if term in label_set)
    coverage = len(matched) / max(1, len(query_set))
    precision = len(set(matched)) / max(1, len(label_set))
    phrase = SequenceMatcher(None, " ".join(query_terms), " ".join(label_terms)).ratio()
    return coverage, precision, phrase, matched


def _build_context(lines: Sequence[str], label_index: int, value_index: int, section: str, label: str, value: str) -> str:
    parts: list[str] = []
    if section and _word_key(section) not in _word_key(label):
        parts.append(section)
    parts.append(label)
    parts.append(value)
    # One short trailing qualifier may carry necessary context, but never include a
    # neighbouring property/value pair.
    after_index = max(label_index, value_index) + 1
    if after_index < len(lines):
        after = _normalize_text(lines[after_index])
        if after and len(after) <= 140 and not _looks_like_label(after) and not _value_matches(after):
            parts.append(after)
    return "\n".join(dict.fromkeys(part for part in parts if part)).strip()


def extract_fact_pairs(
    *,
    page_text: str,
    page_number: int,
    bubble_document_id: str,
    machine_id: str,
    query: PropertyQuery,
) -> list[FactPair]:
    raw_lines = [_normalize_text(line) for line in str(page_text or "").splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return []
    density = _page_table_density(lines)
    out: list[FactPair] = []
    seen: set[tuple[str, str, str]] = set()

    for index, line in enumerate(lines):
        matches = _value_matches(line)
        for match, number, unit in matches:
            if _value_is_composite(line, match):
                continue
            label = line[: match.start()].strip(" :;-_()[]")
            label_index = index
            same_line = bool(label and _looks_like_label(label))
            if not same_line:
                if index <= 0 or not _looks_like_label(lines[index - 1]):
                    continue
                label = lines[index - 1].strip(" :;-_()[]")
                label_index = index - 1
            if not label:
                continue

            coverage, precision, phrase, matched = _label_metrics(label, query)
            minimum_coverage = 1.0 if len(query.property_terms) <= 2 else 0.75
            if coverage + 1e-9 < minimum_coverage:
                continue
            # A one-word query is inherently broad.  Require an exact label token;
            # competing values are handled by the ambiguity guard below.
            if len(query.property_terms) == 1 and not matched:
                continue

            canonical_number = _canonical_number(number)
            canonical_unit = _canonical_unit(unit)
            if canonical_number is None or not canonical_unit:
                continue
            display = f"{number} {unit}".strip()
            section = _nearest_section(lines, label_index)
            context = _build_context(lines, label_index, index, section, label, display)

            exact_label = tuple(_words(label)) == tuple(query.property_terms)
            density_bonus = min(0.16, 0.02 * density)
            score = (
                0.54 * coverage
                + 0.14 * precision
                + 0.14 * phrase
                + (0.10 if exact_label else 0.0)
                + density_bonus
                + (0.03 if not same_line else 0.015)
            )
            key = (_word_key(label), canonical_number, canonical_unit)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                FactPair(
                    label=label,
                    value=display,
                    number=number,
                    unit=unit,
                    canonical_number=canonical_number,
                    canonical_unit=canonical_unit,
                    page_number=max(1, int(page_number or 1)),
                    bubble_document_id=str(bubble_document_id or ""),
                    machine_id=str(machine_id or ""),
                    section=section,
                    context=context,
                    table_density=density,
                    same_line=same_line,
                    label_coverage=coverage,
                    label_precision=precision,
                    phrase_similarity=phrase,
                    score=score,
                )
            )
    return out


def choose_unambiguous_fact(pairs: Sequence[FactPair], query: PropertyQuery) -> Optional[PrecisionFactResolution]:
    pairs = [pair for pair in pairs if pair.bubble_document_id and pair.value]
    if not pairs:
        return None

    groups: dict[tuple[str, str], list[FactPair]] = {}
    for pair in pairs:
        groups.setdefault((pair.canonical_number, pair.canonical_unit), []).append(pair)

    # A single generic property word (for example only "peso"/"weight") cannot
    # distinguish the complete machine from subassemblies when the scope contains
    # different values.  Leave that ambiguity to the normal semantic pipeline.
    if len(query.property_terms) == 1 and len(groups) > 1:
        return None

    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (
            -max(pair.score for pair in item[1]),
            -max(pair.label_coverage for pair in item[1]),
            -len(item[1]),
            min(pair.page_number for pair in item[1]),
        ),
    )
    if not ranked_groups:
        return None
    top_key, top_pairs = ranked_groups[0]
    top_score = max(pair.score for pair in top_pairs)
    top_coverage = max(pair.label_coverage for pair in top_pairs)

    # Never choose between materially competing scalar values.  Duplicate pages or
    # a technical table plus a lifting label with the same value are corroboration,
    # not ambiguity.
    for _, competing in ranked_groups[1:]:
        competing_score = max(pair.score for pair in competing)
        competing_coverage = max(pair.label_coverage for pair in competing)
        if competing_coverage >= top_coverage - 0.10 and competing_score >= top_score - 0.12:
            return None

    chosen = sorted(
        top_pairs,
        key=lambda pair: (
            -pair.label_coverage,
            -pair.table_density,
            -pair.label_precision,
            -pair.phrase_similarity,
            pair.page_number,
        ),
    )[0]
    supporting_pages = tuple(sorted({pair.page_number for pair in top_pairs}))
    return PrecisionFactResolution(
        label=chosen.label,
        value=chosen.value,
        number=chosen.number,
        unit=chosen.unit,
        canonical_number=top_key[0],
        canonical_unit=top_key[1],
        bubble_document_id=chosen.bubble_document_id,
        page_number=chosen.page_number,
        machine_id=chosen.machine_id,
        snippet=chosen.context,
        property_terms=query.property_terms,
        supporting_pages=supporting_pages,
        score=chosen.score,
    )


def _identifier_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _strip_accents(_normalize_text(value)).casefold())


def _strong_query_codes(query: PropertyQuery) -> tuple[str, ...]:
    out: list[str] = []
    for raw in query.code_tokens:
        key = _identifier_key(raw)
        # Short mixed tokens such as 400V or M12 are often values/components, not
        # a machine/document identity.  Longer identifiers are safe to require.
        if len(key) >= 6 and key not in out:
            out.append(key)
    return tuple(out)


def _source_matches_query_codes(
    *,
    query: PropertyQuery,
    source_hint: str,
    page_text: str,
) -> bool:
    required = _strong_query_codes(query)
    if not required:
        return True
    haystack = _identifier_key(" ".join((source_hint or "", page_text or "")))
    return bool(haystack) and all(code in haystack for code in required)


def _matching_scoped_fact_pairs(
    pages: Sequence[dict[str, Any]], property_query: PropertyQuery, target_machine_id: str
) -> list[FactPair]:
    pairs: list[FactPair] = []
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        page_text = str(page.get("text") or page.get("page_text") or "")
        if not _ordinary_document_page(page_text):
            continue
        pairs.extend(extract_fact_pairs(
            page_text=page_text,
            page_number=int(page.get("page_number") or page.get("page_from") or 1),
            bubble_document_id=str(page.get("bubble_document_id") or ""),
            machine_id=str(page.get("machine_id") or ""),
            query=property_query,
        ))
    target = str(target_machine_id or "").strip()
    if target:
        exact = [pair for pair in pairs if str(pair.machine_id or "").strip() == target]
        if exact:
            pairs = exact
    return pairs


def resolve_precision_fact_from_pages(
    *, query: str, pages: Sequence[dict[str, Any]], target_machine_id: str = "",
    property_query: Optional[PropertyQuery] = None,
) -> Optional[PrecisionFactResolution]:
    property_query = property_query or extract_property_query(query)
    if property_query is None:
        return None
    return choose_unambiguous_fact(
        _matching_scoped_fact_pairs(pages, property_query, target_machine_id), property_query
    )


def _fetch_scoped_pages(
    *,
    runtime: PrecisionFactRuntime,
    property_query: PropertyQuery,
    company_id: str,
    machine_id: str,
    doc_ids: Optional[list[str]],
    bubble_document_id: Optional[str],
) -> list[dict[str, Any]]:
    where_sql, base_params = runtime.build_scope_where(
        company_id=company_id,
        machine_id=machine_id,
        doc_ids=doc_ids,
        bubble_document_id=bubble_document_id,
    )
    terms = list(property_query.property_terms[:4])

    def run(require_all: bool) -> list[dict[str, Any]]:
        if not terms:
            return []
        joiner = " AND " if require_all else " OR "
        predicate = joiner.join(["LOWER(COALESCE(text, '')) LIKE %s" for _ in terms])
        sql = f"""
            SELECT bubble_document_id, machine_id, page_number,
                   LEFT(COALESCE(text, ''), %s) AS page_text
            FROM public.document_pages
            WHERE {where_sql}
              AND text IS NOT NULL
              AND length(text) > 20
              AND COALESCE(text, '') NOT ILIKE 'DOCUMENT_FILE_TYPE: XLSX%%'
              AND COALESCE(text, '') NOT ILIKE 'SOURCE_TYPE:%%'
              AND ({predicate})
            ORDER BY CASE WHEN machine_id = %s THEN 0 ELSE 1 END,
                     bubble_document_id, page_number
            LIMIT %s;
        """
        params: list[Any] = [
            max(1200, int(runtime.page_text_chars or 12000)),
            *base_params,
            *[f"%{term}%" for term in terms],
            machine_id,
            max(20, int(runtime.page_scan_limit or 220)),
        ]
        conn = runtime.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "bubble_document_id": str(bdid or ""),
                "machine_id": str(mid or ""),
                "page_number": int(page or 1),
                "text": str(text or ""),
            }
            for bdid, mid, page, text in rows
        ]

    rows = run(True)
    if not rows and len(terms) > 1:
        rows = run(False)
    return rows


def resolve_precision_fact(
    *, query: str, company_id: str, machine_id: str,
    doc_ids: Optional[list[str]], bubble_document_id: Optional[str],
    runtime: PrecisionFactRuntime, answer_contract: Optional[Mapping[str, Any]] = None,
) -> Optional[PrecisionFactResolution]:
    def fetch_pairs(property_query: PropertyQuery) -> Optional[list[FactPair]]:
        pages = _fetch_scoped_pages(
            runtime=runtime, property_query=property_query, company_id=company_id,
            machine_id=machine_id, doc_ids=doc_ids, bubble_document_id=bubble_document_id,
        )
        # A capped scan cannot establish the absence of a conflicting value.
        if len(pages) >= max(20, int(runtime.page_scan_limit or 220)):
            return None
        if _strong_query_codes(property_query):
            if runtime.fetch_file_map is None:
                return None
            document_ids = sorted({str(p.get("bubble_document_id") or "").strip()
                                   for p in pages if str(p.get("bubble_document_id") or "").strip()})
            try:
                file_map = runtime.fetch_file_map(company_id, document_ids) or {}
            except Exception:
                return None
            pages = [p for p in pages if _source_matches_query_codes(
                query=property_query,
                source_hint=str(file_map.get(str(p.get("bubble_document_id") or "")) or ""),
                page_text=str(p.get("text") or p.get("page_text") or ""),
            )]
        return _matching_scoped_fact_pairs(pages, property_query, machine_id)

    seen: set[tuple[str, ...]] = set()
    for prop in (extract_property_query(query), property_query_from_answer_contract(query, answer_contract)):
        if prop is None or prop.property_terms in seen:
            continue
        seen.add(prop.property_terms)
        pairs = fetch_pairs(prop)
        if pairs is None:
            return None
        if pairs:
            # Includes a terminal None for conflicting literal values.
            return choose_unambiguous_fact(pairs, prop)

    resolutions: list[PrecisionFactResolution] = []
    for prop in property_queries_from_scalar_target(query, answer_contract):
        if prop.property_terms in seen:
            continue
        seen.add(prop.property_terms)
        pairs = fetch_pairs(prop)
        if pairs is None:
            return None
        if not pairs:
            continue
        resolved = choose_unambiguous_fact(pairs, prop)
        if resolved is None:
            return None
        resolutions.append(resolved)
    if not resolutions:
        return None
    values = {(r.canonical_number, r.canonical_unit) for r in resolutions}
    if len(values) != 1:
        return None
    # All matched equivalent labels agree. Deterministic best source selection.
    return sorted(resolutions, key=lambda r: (-r.score, r.bubble_document_id, r.page_number))[0]


def resolution_to_candidate(resolution: PrecisionFactResolution) -> dict[str, Any]:
    page = max(1, int(resolution.page_number or 1))
    return {
        "citation_id": f"{resolution.bubble_document_id}:p{page}-{page}:precision-fact",
        "bubble_document_id": resolution.bubble_document_id,
        "chunk_index": 0,
        "page_from": page,
        "page_to": page,
        "snippet": resolution.snippet,
        "chunk_full": resolution.snippet,
        "snippet_clean": resolution.snippet,
        "source_type": "document",
        "similarity": 0.0,
        "semantic_similarity": 0.0,
        "retrieval_score": max(1.0, float(resolution.score)),
        "v13_score": max(1.0, float(resolution.score)),
        "exact_machine_scope": True,
        "precision_fact_exact": True,
        "precision_fact_label": resolution.label,
        "precision_fact_value": resolution.value,
        "precision_fact_number": resolution.number,
        "precision_fact_unit": resolution.unit,
        "precision_fact_canonical_number": resolution.canonical_number,
        "precision_fact_canonical_unit": resolution.canonical_unit,
        "precision_fact_property_terms": list(resolution.property_terms),
        "precision_fact_supporting_pages": list(resolution.supporting_pages),
        "precision_fact_score": float(resolution.score),
    }
