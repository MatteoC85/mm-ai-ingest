"""P4-C8: extracted source priority implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class AskRegexAnyRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_regex_any(text: str, patterns: list[str], *, runtime: AskRegexAnyRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    t = _normalize_unicode_advanced(text or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in (patterns or []))


@dataclass(frozen=True)
class AskHasManualModeFalsePositiveRuntime:
    _ask_regex_any: Callable[..., Any]


def ask_has_manual_mode_false_positive(q_low: str, *, runtime: AskHasManualModeFalsePositiveRuntime) -> bool:
    """True when manuale/manual means machine mode, not document source."""
    _ask_regex_any = runtime._ask_regex_any
    patterns = [
        r"\bmodalit[aà]\s+manuale\b",
        r"\bmodo\s+manuale\b",
        r"\bciclo\s+manuale\b",
        r"\bcomando\s+manuale\b",
        r"\bavanzamento\s+manuale\b",
        r"\bripart(?:ire|enza)?\s+(?:prima\s+)?in\s+manuale\b",
        r"\bfunzionamento\s+manuale\b",
        r"\bmanual\s+mode\b",
        r"\bmanual\s+operation\b",
        r"\bmanual\s+cycle\b",
        r"\bmanual\s+command\b",
        r"\bmanual\s+feed\b",
    ]
    return _ask_regex_any(q_low, patterns)


@dataclass(frozen=True)
class AskHasExplicitXlsxSourcePhraseRuntime:
    _ask_regex_any: Callable[..., Any]


def ask_has_explicit_xlsx_source_phrase(q_low: str, *, runtime: AskHasExplicitXlsxSourcePhraseRuntime) -> bool:
    _ask_regex_any = runtime._ask_regex_any
    patterns = [
        r"\b(?:nel|nello|nella|nell|dal|dallo|dalla|secondo|sul|sulla)\s+(?:file\s+)?(?:excel|xlsx|spreadsheet|workbook)\b",
        r"\b(?:nel|nello|nella|nell|dal|dallo|dalla|secondo|sul|sulla)\s+(?:foglio\s+(?:excel|di\s+calcolo)|tabella\s+excel|cartella\s+excel)\b",
        r"\b(?:file\s+excel|file\s+xlsx|excel\s+aziendale|xlsx\s+aziendale|foglio\s+di\s+calcolo|foglio\s+excel|tabella\s+excel)\b",
        r"\b(?:in|from|according\s+to)\s+(?:the\s+)?(?:excel|xlsx|spreadsheet|workbook|worksheet)\b",
        r"\b(?:excel|xlsx|spreadsheet|workbook|worksheet)\s+(?:file|document|source|table|sheet)\b",
        r"\b(?:righe|row|rows|colonne|columns|sheet|sheets|fogli)\b.{0,60}\b(?:excel|xlsx|spreadsheet|workbook)\b",
    ]
    return _ask_regex_any(q_low, patterns)


@dataclass(frozen=True)
class AskHasExplicitManualSourcePhraseRuntime:
    _ask_has_manual_mode_false_positive: Callable[..., Any]
    _ask_regex_any: Callable[..., Any]


def ask_has_explicit_manual_source_phrase(q_low: str, *, runtime: AskHasExplicitManualSourcePhraseRuntime) -> bool:
    _ask_has_manual_mode_false_positive = runtime._ask_has_manual_mode_false_positive
    _ask_regex_any = runtime._ask_regex_any
    if _ask_has_manual_mode_false_positive(q_low):
        # Still allow "nel manuale, cosa dice sulla modalità manuale?".
        override = [
            r"\b(?:nel|nello|nella|dal|dallo|dalla|secondo)\s+(?:il\s+|lo\s+|la\s+)?manuale\b",
            r"\b(?:in|from|according\s+to)\s+(?:the\s+)?(?:machine\s+manual|technical\s+manual|user\s+manual|manual(?!\s+(?:mode|operation|cycle|feed|command)))\b",
        ]
        if not _ask_regex_any(q_low, override):
            return False

    patterns = [
        r"\b(?:nel|nello|nella|dal|dallo|dalla|secondo)\s+(?:il\s+|lo\s+|la\s+)?(?:manuale|pdf|documento\s+pdf|documentazione\s+tecnica)\b",
        r"\b(?:nel|nello|nella|dal|dallo|dalla|secondo)\s+(?:manuale\s+(?:macchina|tecnico|utente)|manuale\s+della\s+macchina)\b",
        r"\b(?:cosa|che\s+cosa|quali|quanto|quando)\s+(?:dice|indica|riporta|prevede)\s+(?:il\s+)?(?:manuale|pdf|documento\s+pdf)\b",
        r"\b(?:manuale\s+della\s+macchina|manuale\s+macchina|manuale\s+tecnico|manuale\s+utente|documentazione\s+tecnica)\b",
        r"\b(?:in|from|according\s+to)\s+(?:the\s+)?(?:machine\s+manual|user\s+manual|technical\s+manual|technical\s+documentation|pdf|manual(?!\s+(?:mode|operation|cycle|feed|command)))\b",
        r"\b(?:what|which|how|when)\s+(?:does|is|are)?\s*(?:the\s+)?(?:machine\s+manual|user\s+manual|technical\s+manual|pdf|manual(?!\s+(?:mode|operation|cycle|feed|command)))\s+(?:say|state|show|indicate)\b",
    ]
    return _ask_regex_any(q_low, patterns)


@dataclass(frozen=True)
class AskHasHardOnlySourceInstructionRuntime:
    _ask_regex_any: Callable[..., Any]


def ask_has_hard_only_source_instruction(q_low: str, *, runtime: AskHasHardOnlySourceInstructionRuntime) -> bool:
    _ask_regex_any = runtime._ask_regex_any
    patterns = [
        r"\bsolo\b", r"\bsoltanto\b", r"\besclusivamente\b", r"\bunicamente\b",
        r"\bnon\s+considerare\s+(?:il\s+)?resto\b",
        r"\bnon\s+usare\s+(?:altre|altri)\s+(?:fonti|documenti|contenuti)\b",
        r"\bsenza\s+considerare\s+(?:altre|altri)\s+(?:fonti|documenti|contenuti)\b",
        r"\bonly\b", r"\bexclusively\b", r"\bsolely\b",
        r"\bdo\s+not\s+use\s+other\s+(?:sources|documents|content)\b",
        r"\bwithout\s+using\s+other\s+(?:sources|documents|content)\b",
    ]
    return _ask_regex_any(q_low, patterns)


@dataclass(frozen=True)
class AskSourcePreferenceProfileRuntime:
    _ask_has_explicit_manual_source_phrase: Callable[..., Any]
    _ask_has_explicit_xlsx_source_phrase: Callable[..., Any]
    _ask_has_hard_only_source_instruction: Callable[..., Any]
    _ask_has_manual_mode_false_positive: Callable[..., Any]
    _ask_regex_any: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_source_preference_profile(q: str, *, runtime: AskSourcePreferenceProfileRuntime) -> dict:
    """Infer soft/hard source preference from the user's wording.

    Source mentions are not treated as hard filters by default. They become:
    - strength="prefer": requested source must have precedence, but other sources may be
      used as secondary context/support;
    - strength="hard": only when the user explicitly says only/exclusively/do not use others;
    - strength="none": no source preference.
    """
    _ask_has_explicit_manual_source_phrase = runtime._ask_has_explicit_manual_source_phrase
    _ask_has_explicit_xlsx_source_phrase = runtime._ask_has_explicit_xlsx_source_phrase
    _ask_has_hard_only_source_instruction = runtime._ask_has_hard_only_source_instruction
    _ask_has_manual_mode_false_positive = runtime._ask_has_manual_mode_false_positive
    _ask_regex_any = runtime._ask_regex_any
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip()
    q_low = q_norm.lower()

    xlsx_pref = _ask_has_explicit_xlsx_source_phrase(q_low)
    manual_pref = _ask_has_explicit_manual_source_phrase(q_low)

    compare_patterns = [
        r"\bconfronta(?:re)?\b", r"\bconfronto\b", r"\bdifferenz[ae]\b", r"\brispett[oa]\s+a\b",
        r"\bcompare\b", r"\bcomparison\b", r"\bvs\b", r"\bversus\b",
    ]
    asks_comparison = bool(xlsx_pref and manual_pref and _ask_regex_any(q_low, compare_patterns))

    # Contrast logic: "Il PDF dice X, ma nel file Excel qual è Y?" means Excel
    # is the target source; the PDF is context/contrast, not the answer authority.
    contrast_markers = [" ma ", " però ", " tuttavia ", " invece ", " but ", " however ", " whereas "]
    tail = q_low
    last_contrast = -1
    for cm in contrast_markers:
        idx = q_low.rfind(cm)
        if idx > last_contrast:
            last_contrast = idx
            tail = q_low[idx + len(cm):]

    tail_xlsx = _ask_has_explicit_xlsx_source_phrase(tail)
    tail_manual = _ask_has_explicit_manual_source_phrase(tail)

    preferred_source = None
    reason = "no_explicit_source_preference"
    if asks_comparison:
        reason = "comparison_between_sources"
    elif last_contrast >= 0 and tail_xlsx and not tail_manual:
        preferred_source = "xlsx"
        reason = "contrast_target_xlsx"
    elif last_contrast >= 0 and tail_manual and not tail_xlsx:
        preferred_source = "manual"
        reason = "contrast_target_manual"
    elif xlsx_pref and not manual_pref:
        preferred_source = "xlsx"
        reason = "explicit_xlsx_source_preference"
    elif manual_pref and not xlsx_pref:
        preferred_source = "manual"
        reason = "explicit_manual_source_preference"
    elif xlsx_pref and manual_pref:
        reason = "multiple_source_mentions_no_single_preference"

    strength = "none"
    if preferred_source:
        strength = "hard" if _ask_has_hard_only_source_instruction(q_low) else "prefer"

    return {
        "preferred_source": preferred_source,
        "strength": strength,
        "reason": reason,
        "xlsx_preference": bool(xlsx_pref),
        "manual_preference": bool(manual_pref),
        "manual_mode_false_positive": _ask_has_manual_mode_false_positive(q_low),
        "asks_comparison": asks_comparison,
    }


@dataclass(frozen=True)
class IsXlsxIndexedPageTextRuntime:
    pass


def is_xlsx_indexed_page_text(text: str, *, runtime: IsXlsxIndexedPageTextRuntime) -> bool:
    t = str(text or "")
    return (
        "DOCUMENT_FILE_TYPE: XLSX" in t
        or "EXTRACTION_MODE: XLSX" in t
        or "DOCUMENT_KIND: Excel file" in t
    )


@dataclass(frozen=True)
class AskManualPriorityQueryIsMaintenanceRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def ask_manual_priority_query_is_maintenance(q: str, *, runtime: AskManualPriorityQueryIsMaintenanceRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    q_low = _normalize_unicode_advanced(q or "").lower()
    return any(
        marker in q_low
        for marker in [
            "manutenz", "maintenance", "controll", "check", "periodic",
            "frequenza", "frequency", "intervall", "interval", "ore", "hours",
            "lubr", "olio", "oil", "filtri", "filters", "ventole", "fans",
            "quadro elettrico", "electrical cabinet", "impianto elettrico", "impianto pneumatico",
            "pneumatic", "raddrizzatura", "straightening",
        ]
    )


@dataclass(frozen=True)
class AskManualPriorityPageHasRealMaintenanceContentRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_manual_priority_page_has_real_maintenance_content(text: str, *, runtime: AskManualPriorityPageHasRealMaintenanceContentRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    t_low = _normalize_unicode_advanced(text or "").lower()
    if not t_low:
        return False

    strong_markers = [
        "tabella per manutenzione", "tabella generale di manutenzione",
        "maintenance table", "general maintenance table",
        "ore di funzionamento", "hours of operation", "operating hours",
        "componenti", "tipo di lubrificante", "quantità", "quantita", "note",
        "controllare il livello", "check the level", "cambio olio", "oil change",
        "pulizia dei filtri", "cleaning the filters", "sostituzione completa dei filtri",
        "scarico della condensa", "drain condensate", "verifica integrità", "verifica integrita",
        "verifica corretto funzionamento", "lubrificazione manuale", "lubrificazione automatica",
        "impianto elettrico", "impianto pneumatico", "raddrizzatura", "riduttore",
    ]
    if any(m in t_low for m in strong_markers):
        return True

    freq_matches = re.findall(r"\bogni\s+\d{1,5}\s*(?:ore|ora|h|giorni|giorno|turno|settimane|settimana|mesi|mese|anni|anno)\b", t_low)
    freq_matches += re.findall(r"\bevery\s+\d{1,5}\s*(?:hours?|h|days?|shift|weeks?|months?|years?)\b", t_low)
    if any(x in t_low for x in ["ogni giorno", "ogni turno", "settiman", "mensil", "annual"]):
        freq_matches.append("periodic_interval")
    return bool(freq_matches)


@dataclass(frozen=True)
class AskManualPriorityPageIsMetaOrIndexRuntime:
    _ask_manual_priority_page_has_real_maintenance_content: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_manual_priority_page_is_meta_or_index(text: str, *, runtime: AskManualPriorityPageIsMetaOrIndexRuntime) -> bool:
    _ask_manual_priority_page_has_real_maintenance_content = runtime._ask_manual_priority_page_has_real_maintenance_content
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    t_low = _normalize_unicode_advanced(text or "").lower()
    if not t_low:
        return False
    weak_markers = [
        "indice manuale", "table of contents", "pagina vuota", "blank page",
        "informazioni generali", "general information", "proprietà delle informazioni",
        "property of information", "tutti i diritti sono riservati", "all rights reserved",
        "operatore la o le persone", "manutentore:", "conduttore:",
    ]
    if any(marker in t_low for marker in weak_markers):
        return True
    short_lines = [ln.strip() for ln in str(text or "").split("\n") if ln.strip()]
    numeric_line_count = sum(1 for ln in short_lines if re.fullmatch(r"\d{1,4}", ln.strip()))
    return numeric_line_count >= 8 and not _ask_manual_priority_page_has_real_maintenance_content(text)


@dataclass(frozen=True)
class AskManualPriorityPageScoreRuntime:
    COMPANY_GENERAL_MACHINE_SENTINEL: Any
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def ask_manual_priority_page_score(
    *,
    q: str,
    page_text: str,
    base_score: float,
    row_machine_id: Optional[str],
    requested_machine_id: Optional[str],
    runtime: AskManualPriorityPageScoreRuntime,
) -> float:
    """Score manual/PDF pages for explicit manual/document questions.

    This is a soft priority, not a hard filter. It improves source selection when
    the user asks for the machine manual: exact-machine manual pages and pages
    with actual maintenance tables/frequencies should outrank index/general pages,
    while company/general manuals can still appear as secondary support.
    """
    COMPANY_GENERAL_MACHINE_SENTINEL = runtime.COMPANY_GENERAL_MACHINE_SENTINEL
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q_low = _normalize_unicode_advanced(q or "").lower()
    t_low = _normalize_unicode_advanced(page_text or "").lower()
    score = float(base_score or 0.0)

    requested_mid = str(requested_machine_id or "").strip()
    row_mid = str(row_machine_id or "").strip()
    if requested_mid and requested_mid != COMPANY_GENERAL_MACHINE_SENTINEL:
        if row_mid == requested_mid:
            score += 24.0
        elif not row_mid:
            # Company/general document: still allowed as support, but not ahead of
            # the exact machine manual when the user says "manuale della macchina".
            score += 4.0
        else:
            score -= 8.0

    asks_maintenance = any(
        marker in q_low
        for marker in [
            "manutenz", "maintenance", "controll", "check", "periodic",
            "periodic", "frequenza", "frequency", "intervall", "interval",
            "ore", "hours", "lubr", "olio", "oil", "filtri", "filters",
            "ventole", "fans", "quadro elettrico", "electrical cabinet",
        ]
    )

    if asks_maintenance:
        strong_markers = [
            "tabella per manutenzione", "tabella generale di manutenzione",
            "maintenance table", "general maintenance table",
            "ore di funzionamento", "hours of operation", "operating hours",
            "componenti", "tipo di lubrificante", "quantità", "quantita", "note",
            "controllare il livello", "check the level", "cambio olio", "oil change",
            "pulizia dei filtri", "cleaning the filters", "sostituzione completa dei filtri",
            "scarico della condensa", "drain condensate", "verifica integrità", "verifica integrita",
            "verifica corretto funzionamento", "lubrificazione manuale", "lubrificazione automatica",
        ]
        for marker in strong_markers:
            if marker in t_low:
                score += 10.0

        # Frequencies/intervals are the key evidence for periodic maintenance.
        freq_matches = re.findall(r"\bogni\s+\d{1,5}\s*(?:ore|ora|h|giorni|giorno|turno|settimane|settimana|mesi|mese|anni|anno)\b", t_low)
        freq_matches += re.findall(r"\bevery\s+\d{1,5}\s*(?:hours?|h|days?|shift|weeks?|months?|years?)\b", t_low)
        if "ogni giorno" in t_low:
            freq_matches.append("ogni giorno")
        if "ogni turno" in t_low:
            freq_matches.append("ogni turno")
        if "settiman" in t_low:
            freq_matches.append("settimanale")
        if "mensil" in t_low:
            freq_matches.append("mensile")
        if "annual" in t_low:
            freq_matches.append("annuale")
        if freq_matches:
            score += min(32.0, 8.0 * len(set(freq_matches)))

        if "impianto elettrico" in t_low or "electrical" in t_low:
            score += 5.0
        if "impianto pneumatico" in t_low or "pneumatic" in t_low:
            score += 5.0
        if "raddrizzatura" in t_low or "avanzamento" in t_low or "riduttore" in t_low:
            score += 5.0

        weak_or_meta_markers = [
            "indice manuale", "table of contents", "pagina vuota", "blank page",
            "informazioni generali", "general information", "proprietà delle informazioni",
            "property of information", "tutti i diritti sono riservati", "all rights reserved",
            "operatore la o le persone", "manutentore:", "conduttore:",
        ]
        for marker in weak_or_meta_markers:
            if marker in t_low:
                score -= 24.0

        # Strong TOC heuristic: many page-number lines and section titles, but no
        # actual interval values or operative table rows.
        short_lines = [ln.strip() for ln in str(page_text or "").split("\n") if ln.strip()]
        numeric_line_count = sum(1 for ln in short_lines if re.fullmatch(r"\d{1,4}", ln.strip()))
        if numeric_line_count >= 8 and not freq_matches:
            score -= 18.0

    return score


@dataclass(frozen=True)
class AskSecondarySupportCitationsRuntime:
    pass


def ask_secondary_support_citations(
    secondary_citations: list[dict],
    primary_citations: list[dict],
    *,
    max_items: int = 5,
    runtime: AskSecondarySupportCitationsRuntime,
) -> list[dict]:
    primary_ids = {str(c.get("citation_id") or "").strip() for c in (primary_citations or []) if isinstance(c, dict)}
    primary_docs = {str(c.get("bubble_document_id") or "").strip() for c in (primary_citations or []) if isinstance(c, dict)}
    out: list[dict] = []
    seen: set[str] = set()
    for c in secondary_citations or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("citation_id") or "").strip()
        bdid = str(c.get("bubble_document_id") or "").strip()
        if not cid or cid in seen or cid in primary_ids:
            continue
        # Avoid duplicating the same document/page as secondary when it is already primary.
        if bdid in primary_docs and bool(c.get("ask_source_priority")):
            continue
        seen.add(cid)
        out.append(c)
        if len(out) >= max_items:
            break
    return out


@dataclass(frozen=True)
class AssistantCoreNumericSignalRuntime:
    _ASSISTANT_CORE_LABELED_VALUE_UNIT_RE: Any
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def assistant_core_numeric_signal(text: str, *, runtime: AssistantCoreNumericSignalRuntime) -> dict:
    _ASSISTANT_CORE_LABELED_VALUE_UNIT_RE = runtime._ASSISTANT_CORE_LABELED_VALUE_UNIT_RE
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    value = _normalize_unicode_advanced(str(text or ""))
    number_matches = re.findall(r"(?<![A-Za-z0-9_])[+\-]?\d+(?:[.,]\d+)?", value)
    unit_matches = re.findall(
        r"(?i)(?<![A-Za-z0-9_])(?:[+\-]?\d+(?:[.,]\d+)?\s*)"
        r"(?:kg|g|t|ton|lb|mm|cm|m|m/s|m/s2|m/s²|rpm|min\-?1|1/min|hz|khz|kw|w|v|a|bar|mpa|pa|nm|n|kn|°c|celsius|%|ms|s|sec|min|h|hour|hours)\b",
        value,
    )
    labeled_value_unit_matches = list(_ASSISTANT_CORE_LABELED_VALUE_UNIT_RE.finditer(value))
    return {
        "has_number": bool(number_matches),
        "has_number_with_unit": bool(unit_matches or labeled_value_unit_matches),
        "numbers": number_matches[:12],
    }


@dataclass(frozen=True)
class AssistantCoreInterfaceNavigationSignalRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def assistant_core_interface_navigation_signal(text: str, *, runtime: AssistantCoreInterfaceNavigationSignalRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    low = _normalize_unicode_advanced(str(text or "")).lower()
    markers = (
        "hmi", "operator panel", "operator interface", "touch screen", "touchscreen",
        "screen", "page", "menu", "window", "tab", "alarm history", "alarm list",
        "pannello operatore", "interfaccia operatore", "schermata", "pagina", "menu",
        "finestra", "scheda", "storico allarmi", "lista allarmi",
    )
    return any(marker in low for marker in markers)


@dataclass(frozen=True)
class AssistantCoreSequenceSignalRuntime:
    _normalize_unicode_advanced: Callable[..., Any]


def assistant_core_sequence_signal(text: str, *, runtime: AssistantCoreSequenceSignalRuntime) -> bool:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    low = _normalize_unicode_advanced(str(text or "")).lower()
    markers = (
        "first", "then", "next", "after", "before", "while", "when", "at the same time",
        "simultaneously", "during the return", "at the end", "opens", "closes", "returns",
        "prima", "poi", "quindi", "successivamente", "dopo", "prima di", "mentre",
        "quando", "contemporaneamente", "simultaneamente", "durante il ritorno",
        "a fine corsa", "apre", "chiude", "ritorna",
    )
    return sum(1 for marker in markers if marker in low) >= 2


@dataclass(frozen=True)
class AssistantCorePsIsSubstantiveRuntime:
    _assistant_core_candidate_source_type: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    _parse_structured_source_fields: Callable[..., Any]
    _v13_candidate_text: Callable[..., Any]
    re: Any


def assistant_core_ps_is_substantive(candidate: dict, *, runtime: AssistantCorePsIsSubstantiveRuntime) -> bool:
    _assistant_core_candidate_source_type = runtime._assistant_core_candidate_source_type
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _parse_structured_source_fields = runtime._parse_structured_source_fields
    _v13_candidate_text = runtime._v13_candidate_text
    re = runtime.re
    if _assistant_core_candidate_source_type(candidate) != "ps":
        return True
    raw = _v13_candidate_text(candidate)
    fields = _parse_structured_source_fields(raw)
    values = [
        str(fields.get("description") or ""),
        str(fields.get("solution") or ""),
        str(fields.get("notes") or ""),
    ]
    placeholder_values = {
        "", "-", "n/a", "na", "none", "null", "problema", "problem", "soluzione",
        "solution", "descr", "description", "test", "other",
    }
    substantive = [
        re.sub(r"\s+", " ", _normalize_unicode_advanced(v).lower()).strip(" .:;-_")
        for v in values
    ]
    substantive = [v for v in substantive if v not in placeholder_values and len(v) >= 24]
    return bool(substantive)


@dataclass(frozen=True)
class AssistantCoreSourceBonusRuntime:
    INFO_FAULT_DIAGNOSTIC: Any
    INFO_INTERFACE_NAVIGATION: Any
    INFO_NUMERIC_SPECIFICATION: Any
    INFO_PROCEDURE_FULL: Any
    INFO_PROCEDURE_SEGMENT: Any
    INFO_SEQUENCE_SYNCHRONIZATION: Any
    INFO_SOURCE_RETRIEVAL: Any


def assistant_core_source_bonus(
    source_type: str,
    request_kind: str,
    preferred_source_types: set[str],
    information_task: str,
    *, runtime: AssistantCoreSourceBonusRuntime,
) -> float:
    INFO_FAULT_DIAGNOSTIC = runtime.INFO_FAULT_DIAGNOSTIC
    INFO_INTERFACE_NAVIGATION = runtime.INFO_INTERFACE_NAVIGATION
    INFO_NUMERIC_SPECIFICATION = runtime.INFO_NUMERIC_SPECIFICATION
    INFO_PROCEDURE_FULL = runtime.INFO_PROCEDURE_FULL
    INFO_PROCEDURE_SEGMENT = runtime.INFO_PROCEDURE_SEGMENT
    INFO_SEQUENCE_SYNCHRONIZATION = runtime.INFO_SEQUENCE_SYNCHRONIZATION
    INFO_SOURCE_RETRIEVAL = runtime.INFO_SOURCE_RETRIEVAL
    bonus = 0.0
    if source_type in preferred_source_types:
        bonus += 0.18

    if information_task in {INFO_PROCEDURE_FULL, INFO_PROCEDURE_SEGMENT} or request_kind == "procedure":
        bonus += {"step": 0.22, "procedure": 0.20, "document": 0.07, "ps": 0.02}.get(source_type, 0.0)
    elif information_task == INFO_NUMERIC_SPECIFICATION:
        # Numeric values may live in a manual, a procedure or an exact Step. Do not
        # systematically demote structured records in favour of PDFs.
        bonus += {"step": 0.16, "procedure": 0.14, "document": 0.15, "ps": 0.03}.get(source_type, 0.0)
    elif information_task == INFO_INTERFACE_NAVIGATION:
        bonus += {"document": 0.22, "procedure": 0.02, "step": 0.01, "ps": -0.03}.get(source_type, 0.0)
    elif information_task == INFO_SEQUENCE_SYNCHRONIZATION:
        bonus += {"step": 0.18, "procedure": 0.14, "document": 0.17, "ps": 0.02}.get(source_type, 0.0)
    elif request_kind in {"fault_diagnostic", "guided_diagnostic"} or information_task == INFO_FAULT_DIAGNOSTIC:
        # A P&S is valuable only after the same-subsystem/facet gate below. The
        # source-type prior is intentionally modest so an unrelated P&S cannot win.
        bonus += {"ps": 0.10, "document": 0.10, "procedure": 0.07, "step": 0.08}.get(source_type, 0.0)
    elif request_kind == "source_retrieval" or information_task == INFO_SOURCE_RETRIEVAL:
        bonus += 0.10 if source_type in preferred_source_types else 0.0
    elif request_kind in {"factual", "comparison"}:
        bonus += {"document": 0.08, "step": 0.06, "procedure": 0.05, "ps": 0.02}.get(source_type, 0.0)
    return bonus


