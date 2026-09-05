"""Epistemic query preparation for MachineMind Root Cause retrieval.

The Root Cause pipeline must distinguish a machine observation from a variable the
user explicitly says was *not* read, checked, measured, observed, recorded or
otherwise made available.  Mentioning an unobserved variable is useful when asking
what to collect next, but it is not positive evidence and must not steer retrieval,
ranking or source admission.

This module is deliberately independent from MachineMind's database, models and
machine vocabulary.  It performs bounded multilingual (Italian/English) clause
analysis, creates a retrieval-safe query and sanitizes semantic-router fields.  It
does not infer a cause and it never changes ASK or Smart Diagnostic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import re
import unicodedata


POLICY_VERSION = "root-diagnostic-query-state-v1"

# These are epistemic/measurement verbs, not machine-domain keywords.  They are
# intentionally limited to high-precision forms: ordinary negative fault statements
# such as "Ready non arriva", "no alarms appeared" or "pressure does not rise" are
# observations and must remain in the retrieval query.
_IT_EPISTEMIC_PARTICIPLES = (
    r"lett[oaie]", r"controllat[oaie]", r"verificat[oaie]", r"misurat[oaie]",
    r"rilevat[oaie]", r"osservat[oaie]", r"registrat[oaie]", r"raccolt[oaie]",
    r"acquisit[oaie]", r"annotat[oaie]", r"ispezionat[oaie]", r"testat[oaie]",
    r"accertat[oaie]", r"monitorat[oaie]",
)
_IT_EPISTEMIC_INFINITIVES = (
    r"leggere", r"controllare", r"verificare", r"misurare", r"rilevare",
    r"osservare", r"registrare", r"raccogliere", r"acquisire", r"annotare",
    r"ispezionare", r"testare", r"accertare", r"monitorare",
)
_EN_EPISTEMIC_VERBS = (
    r"read", r"check(?:ed)?", r"verif(?:y|ied)", r"measur(?:e|ed)",
    r"observ(?:e|ed)", r"record(?:ed)?", r"collect(?:ed)?", r"acquir(?:e|ed)",
    r"inspect(?:ed)?", r"test(?:ed)?", r"confirm(?:ed)?", r"monitor(?:ed)?",
    r"log(?:ged)?", r"review(?:ed)?",
)

_IT_PART = "(?:" + "|".join(_IT_EPISTEMIC_PARTICIPLES) + ")"
_IT_INF = "(?:" + "|".join(_IT_EPISTEMIC_INFINITIVES) + ")"
_EN_VERB = "(?:" + "|".join(_EN_EPISTEMIC_VERBS) + ")"

# High-confidence markers for clauses whose content is explicitly unavailable.
_MISSING_MARKER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Italian active/passive and inability/absence forms.
        rf"\bnon\s+(?:ho|hai|ha|abbiamo|avete|hanno)\s+(?:ancora\s+)?{_IT_PART}\b",
        rf"\bnon\s+(?:e|è|sono|era|erano)\s+stat[oaie]\s+{_IT_PART}\b",
        rf"\b(?:non\s+(?:posso|puoi|può|possiamo|potete|possono)\s+|impossibile\s+){_IT_INF}\b",
        rf"\b(?:senza|privo\s+di|priva\s+di)\s+(?:aver\s+)?{_IT_PART}\b",
        r"\bnon\s+(?:ho|hai|ha|abbiamo|avete|hanno)\s+(?:alcun[oa]?|nessun[oa]?)\s+(?:dato|dati|informazione|informazioni|lettura|letture|misura|misure|risultato|risultati|riscontro|riscontri|log|storico|messaggio|messaggi)\b",
        r"\bnon\s+(?:e|è|sono|era|erano)\s+disponibil[ei]\s+(?:alcun[oa]?|nessun[oa]?|dati?|informazioni?|letture?|misure?|risultati?|riscontri?|log|storico|messaggi?)\b",
        r"\b(?:nessun[oa]?|alcun[oa]?)\s+(?:dato|informazione|lettura|misura|risultato|riscontro|log|storico|messaggio)\s+(?:e|è|risulta)\s+disponibile\b",
        r"\b(?:dati|informazioni|letture|misure|risultati|riscontri|log|storico|messaggi)\s+non\s+disponibil[ei]\b",
        r"\b(?:mancano|manca)\s+(?:i\s+|il\s+|la\s+|le\s+|gli\s+)?(?:dati|informazioni|letture|misure|risultati|riscontri|log|storico|messaggi)\b",
        r"\bnon\s+(?:so|sappiamo|conosco|conosciamo)\b",
        r"\b(?:stato|valore|esito|risultato)\s+(?:e|è|resta|rimane)\s+(?:sconosciut[oa]|non\s+noto)\b",
        r"\bnon\s+(?:e|è|sono)\s+stat[oaie]\s+effettuat[oaie]\s+(?:alcun[oa]?|nessun[oa]?)\s+(?:controllo|verifica|misura|lettura|test|ispezione)\b",
        r"\bnon\s+(?:ho|hai|ha|abbiamo|avete|hanno)\s+(?:effettuat[oaie]|eseguit[oaie]|fatt[oaie])\s+(?:alcun[oa]?|nessun[oa]?)?\s*(?:controll[oi]|verific[ae]|misur[ae]|lettur[ae]|test|ispezion[ei])\b",
        r"\b(?:nessun[oa]?|alcun[oa]?)\s+(?:controllo|verifica|misura|lettura|test|ispezione)\s+(?:e|è|risulta)?\s*stat[oaie]\s+(?:effettuat[oaie]|eseguit[oaie]|fatt[oaie])\b",
        r"\bnon\s+(?:posso|puoi|può|possiamo|potete|possono)\s+(?:effettuare|eseguire|fare)\s+(?:alcun[oa]?|nessun[oa]?|un[oa]?|il|la)?\s*(?:controllo|verifica|misura|lettura|test|ispezione)\b",
        r"\bnon\s+(?:ho|abbiamo|ha|hanno)\s+controllato\s+nulla\b",
        # English active/passive and availability forms.
        rf"\b(?:(?:i|we|you|they|he|she|it)\s+)?(?:have|has|had)\s+not\s+(?:yet\s+)?{_EN_VERB}\b",
        rf"\b(?:(?:i|we|you|they|he|she|it)\s+)?(?:haven't|hasn't|hadn't)\s+(?:yet\s+)?{_EN_VERB}\b",
        rf"\b(?:was|were|is|are)\s+not\s+(?:yet\s+)?{_EN_VERB}\b",
        rf"\b(?:without|before)\s+(?:having\s+)?{_EN_VERB}\b",
        r"\b(?:do|does|did)\s+not\s+(?:have|know)\s+(?:any\s+)?(?:data|information|reading|readings|measurement|measurements|result|results|log|history|message|messages)\b",
        r"\b(?:don't|doesn't|didn't)\s+(?:have|know)\s+(?:any\s+)?(?:data|information|reading|readings|measurement|measurements|result|results|log|history|message|messages)\b",
        r"\b(?:no|not\s+any)\s+(?:diagnostic\s+)?(?:data|information|readings?|measurements?|results?|observations?|logs?|history|messages?)\s+(?:is|are|was|were)\s+available\b",
        r"\b(?:data|information|readings?|measurements?|results?|observations?|logs?|history|messages?)\s+(?:is|are|was|were)\s+not\s+available\b",
        r"\b(?:status|value|result|reading)\s+(?:is|was|remains?)\s+unknown\b",
        r"\b(?:nothing|none)\s+(?:has\s+been|was)\s+(?:checked|measured|observed|recorded|collected|verified|inspected|tested)\b",
        r"\b(?:we|i)\s+(?:have|has|had)?\s*not\s+checked\s+anything\b",
        r"\b(?:(?:i|we|you|they|he|she|it)\s+)?(?:have|has|had|did)\s+not\s+(?:perform|performed|carry\s+out|carried\s+out|make|made)\s+(?:any|a|the)?\s*(?:checks?|measurements?|readings?|tests?|inspections?|verifications?)\b",
    )
)

# A global assertion that diagnostic information was not collected is strong enough
# to justify a fail-closed clarification when no separate, cause-specific observation
# remains.  These patterns are intentionally generic and multilingual.
_GLOBAL_ABSENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnon\s+(?:e|è|sono|era|erano)\s+stat[oaie]\s+(?:raccolt[oaie]|registrat[oaie]|acquisit[oaie])\s+(?:alcun[oa]?|nessun[oa]?)\s+(?:dato|informazione|osservazione|lettura|misura|riscontro)(?:\s+diagnostic[oaie])?\b",
        r"\bnon\s+(?:ho|abbiamo|ha|hanno)\s+(?:alcun[oa]?|nessun[oa]?)\s+(?:dato|informazione|osservazione|lettura|misura|riscontro)(?:\s+diagnostic[oaie])?\b",
        r"\bnon\s+(?:e|è|sono)\s+disponibil[ei]\s+(?:dati|informazioni|osservazioni|letture|misure|riscontri)(?:\s+diagnostic[oaie])?\b",
        r"\bnon\s+(?:e|è|sono)\s+stat[oaie]\s+effettuat[oaie]\s+(?:controlli|verifiche|misure|letture|osservazioni|ispezioni)\b",
        r"\b(?:nessun[oa]?|alcun[oa]?)\s+(?:controllo|verifica|misura|lettura|test|ispezione)\s+(?:e|è|risulta)?\s*stat[oaie]\s+(?:effettuat[oaie]|eseguit[oaie]|fatt[oaie])\b",
        r"\bnon\s+(?:ho|abbiamo|ha|hanno)\s+controllato\s+nulla\b",
        r"\bno\s+(?:diagnostic\s+)?(?:data|information|observations?|readings?|measurements?|checks?|results?)\s+(?:has|have|had|was|were)\s+(?:been\s+)?(?:collected|recorded|acquired|made|performed|available)\b",
        r"\b(?:no|not\s+any)\s+(?:diagnostic\s+)?(?:data|information|observations?|readings?|measurements?|results?)\s+(?:is|are|was|were)\s+available\b",
        r"\b(?:nothing|none)\s+(?:has\s+been|was)\s+(?:checked|measured|observed|recorded|collected|verified|inspected|tested)\b",
        r"\b(?:we|i)\s+(?:have\s+)?not\s+checked\s+anything\b",
    )
)

# High-precision markers of a concrete observation.  Stop/no-start/restart alone is
# deliberately omitted because it identifies an effect, not a discriminating cause.
_OBSERVATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Values, codes, explicit states and temporal changes.
        r"(?<!\w)[-+]?\d+(?:[.,]\d+)?\s*(?:bar|pa|kpa|mpa|v|a|kw|kn|nm|rpm|hz|°?c|mm|cm|m|%|ms|s|min)(?!\w)",
        r"[«\"']\s*[^«»\"']{2,80}\s*[»\"']",
        r"\b(?:allarme|errore|messaggio|codice|alarm|error|message|code)\s*[:#-]?\s*[A-Z]{1,}[A-Z0-9\-_]*\d+[A-Z0-9\-_]*\b",
        r"\b(?:compare|comparso|apparso|visualizzat[oa]|lampeggia|acces[oa]|spento|presente|assente|arriva|non\s+arriva|commuta|non\s+commuta|oscilla|stabile|instabile|aumenta|diminuisce|sale|scende|perdita|sfiato|condensa|danneggiat[oa]|spostat[oa]|sostituit[oa]|rumore|vibrazion[ei]|bava|tracce?|segni?|lucid[oaie]|cald[oa]|scalda|riscalda|surriscald|bloccato|inceppato)\b",
        r"\b(?:appears?|appeared|displayed|flashes?|flashing|lit|present|absent|missing|arrives?|does\s+not\s+arrive|switches?|does\s+not\s+switch|oscillates?|stable|unstable|rises?|falls?|increases?|decreases?|leak|hiss|condensate|damaged|moved|replaced|noise|vibration|burr|marks?|shiny|hot|heats?|heating|overheat|jammed|blocked)\b",
        r"\b(?:is|was|remains?|stays?)\s+(?:on|off)\b",
        r"\b(?:subito\s+dopo|immediatamente\s+dopo|dopo\s+(?:aver|la|il|lo|un|una)|immediately\s+after|right\s+after|after\s+(?:the|a|an))\b",
    )
)

# Boilerplate used only to decide whether a global-absence request contains a second
# concrete observation.  It is never used to retrieve or rank evidence.
_GENERIC_EFFECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:macchina|machine|impianto|linea|line)\b.*\b(?:si\s+ferma|ferma|arresta|non\s+riparte|non\s+parte|stops?|stopped|does\s+not\s+restart|won't\s+restart|does\s+not\s+start)\b",
        r"\b(?:ha\s+ripreso|riprende|riparte|resumed?|restarts?)\b",
        r"\b(?:intermittente|intermittently|a\s+volte|sometimes|occasional(?:ly)?)\b",
    )
)

_META_QUESTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:quali|quale|cosa|come|da\s+dove)\b.*\b(?:caus[ae]|diagnos|verific|controll|osservazion|informazion)\w*",
        r"\b(?:what|which|how|where)\b.*\b(?:causes?|diagnos|checks?|observations?|information)\w*",
        r"\b(?:caso|case)\s+(?:di\s+certificazione\s+)?[A-Z0-9][A-Z0-9\-_]{3,}\b",
    )
)

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
_STOPWORDS = {
    # Italian
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli", "anche", "che",
    "con", "da", "dal", "dalla", "dalle", "de", "dei", "del", "della", "delle",
    "di", "e", "ed", "è", "e", "gli", "ha", "ho", "i", "il", "in", "la", "le",
    "lo", "ma", "mi", "nei", "nel", "nella", "non", "o", "per", "piu", "più",
    "se", "si", "sono", "su", "tra", "un", "una", "uno", "questo", "questa",
    "queste", "questi", "quale", "quali", "cosa", "come", "dopo", "prima",
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "not", "of", "on", "or", "the", "then", "this", "that", "to", "was", "were",
    "what", "which", "with", "without", "we", "you", "yet",
}

_EPISTEMIC_CONTEXT_TOKENS = {
    "check", "checked", "checking", "control", "controllo", "controlli",
    "dato", "dati", "data", "information", "informazione", "informazioni",
    "lettura", "letture", "measurement", "measurements", "misura", "misure",
    "observation", "observations", "osservazione", "osservazioni", "reading",
    "readings", "result", "results", "riscontro", "riscontri", "stato", "status",
    "test", "tests", "verifica", "verifiche", "verification",
}


@dataclass(frozen=True)
class DiagnosticQueryProfile:
    """Query state passed from composition root to Root Cause retrieval."""

    original_query: str
    retrieval_query: str
    observed_text: str
    missing_spans: tuple[str, ...]
    missing_information: tuple[str, ...]
    explicit_global_absence: bool
    concrete_observation_count: int
    force_clarification: bool
    clarification_question: str
    response_language: str
    policy_version: str = POLICY_VERSION
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "original_query": self.original_query,
            "retrieval_query": self.retrieval_query,
            "observed_text": self.observed_text,
            "missing_spans": list(self.missing_spans),
            "missing_information": list(self.missing_information),
            "explicit_global_absence": bool(self.explicit_global_absence),
            "concrete_observation_count": int(self.concrete_observation_count),
            "force_clarification": bool(self.force_clarification),
            "clarification_question": self.clarification_question,
            "response_language": self.response_language,
            "reason": self.reason,
        }

    def public_summary(self) -> dict[str, Any]:
        """Bounded debug metadata; never includes the full original user request."""
        return {
            "policy_version": self.policy_version,
            "retrieval_query": self.retrieval_query[:1200],
            "missing_information": list(self.missing_information)[:12],
            "missing_span_count": len(self.missing_spans),
            "explicit_global_absence": bool(self.explicit_global_absence),
            "concrete_observation_count": int(self.concrete_observation_count),
            "force_clarification": bool(self.force_clarification),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouterSanitizationResult:
    payload: dict[str, Any]
    summary: dict[str, Any]


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _normalize(value)).casefold()
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _tokens(value: Any) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(_fold(value)):
        token = raw.strip("_")
        if len(token) < 2 or token in _STOPWORDS or token.isdigit():
            continue
        out.add(token)
    return out


def _overlap(left: Any, right: Any) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _missing_overlap(value: Any, missing_item: Any) -> float:
    """Overlap against the technical subject of a missing-information item."""
    value_tokens = _tokens(value)
    missing_tokens = _tokens(missing_item) - _EPISTEMIC_CONTEXT_TOKENS
    if not value_tokens or not missing_tokens:
        return 0.0
    return len(value_tokens & missing_tokens) / max(
        1, min(len(value_tokens), len(missing_tokens))
    )


def _sentence_end(text: str, start: int) -> int:
    match = re.search(r"[.!?](?=\s|$)", text[start:])
    return len(text) if not match else start + match.start()


def _next_adversative(text: str, start: int, hard_end: int) -> int | None:
    segment = text[start:hard_end]
    match = re.search(
        r"(?:,|;)\s*(?:ma|però|tuttavia|invece|but|however|whereas)\s+",
        segment,
        flags=re.IGNORECASE,
    )
    return None if not match else start + match.start()


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((max(0, int(a)), max(0, int(b))) for a, b in spans if b > a)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _expand_missing_start(text: str, marker_start: int) -> int:
    """Include the local grammatical subject/conjunction of a missing clause."""
    marker_start = max(0, int(marker_start))
    window_start = max(0, marker_start - 180)
    prefix = text[window_start:marker_start]
    candidates: list[int] = []

    # Sentence or comma boundaries keep an earlier positive observation intact.
    for match in re.finditer(r"[.!?;]\s*", prefix):
        candidates.append(window_start + match.end())
    for match in re.finditer(r",\s*", prefix):
        candidates.append(window_start + match.start())

    # Adversative/coordinating boundaries are useful when punctuation is omitted.
    for match in re.finditer(
        r"\b(?:ma|però|tuttavia|invece|but|however|whereas|e|and)\s+",
        prefix,
        flags=re.IGNORECASE,
    ):
        candidates.append(window_start + match.start())

    if not candidates:
        # At sentence start, include the short grammatical subject (for example
        # "la pressione" / "the pressure" / "I") so the unobserved variable
        # cannot remain in the retrieval query.
        if window_start == 0 and len(prefix) <= 120 and len(_TOKEN_RE.findall(prefix)) <= 16:
            return 0
        pronoun = re.search(r"\b(?:i|we|you|they|he|she|it)\s+$", prefix, re.IGNORECASE)
        return window_start + pronoun.start() if pronoun else marker_start

    candidate = max(candidates)
    local = text[candidate:marker_start]
    # Never reach far back through a long positive clause.
    if len(local) <= 120 and len(_TOKEN_RE.findall(local)) <= 16:
        return candidate
    return marker_start


def _missing_spans(text: str) -> tuple[list[tuple[int, int]], bool]:
    candidates: list[tuple[int, int]] = []
    explicit_global = False
    global_starts: list[int] = []
    for pattern in _GLOBAL_ABSENCE_PATTERNS:
        for match in pattern.finditer(text):
            explicit_global = True
            global_starts.append(match.start())

    for pattern in _MISSING_MARKER_PATTERNS:
        for match in pattern.finditer(text):
            start = _expand_missing_start(text, match.start())
            end = _sentence_end(text, match.end())
            adversative = _next_adversative(text, match.end(), end)
            if adversative is not None:
                end = adversative
            candidates.append((start, end))

    # A global absence followed by a colon introduces a list of unobserved variables.
    # Keep any positive prefix before the marker and remove the marker plus its list.
    for start in global_starts:
        start = _expand_missing_start(text, start)
        end = _sentence_end(text, start)
        adversative = _next_adversative(text, start, end)
        if adversative is not None:
            end = adversative
        candidates.append((start, end))

    return _merge_spans(candidates), explicit_global


def _clean_after_removal(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", value)
    value = re.sub(r"(?:^|\s)(?:ma|però|tuttavia|but|however)\s*([.!?]|$)", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*[,;:]\s*([.!?])", r"\1", value)
    value = re.sub(r"(?:[.!?]\s*){2,}", ". ", value)
    value = re.sub(r"^[,;:.!?\-–—\s]+", "", value)
    value = re.sub(r"[,;:\-–—\s]+$", "", value)
    return value.strip()


def _remove_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return _clean_after_removal(text)
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        parts.append(" ")
        cursor = end
    parts.append(text[cursor:])
    return _clean_after_removal("".join(parts))


def _strip_epistemic_prefix(value: str) -> str:
    text = _normalize(value).strip(" ,;:.-–—")
    # Remove global absence assertions before shorter verb-only markers, otherwise
    # "non è stato raccolto alcun dato diagnostico" would leave the noun phrase.
    text = re.sub(
        r"^(?:ma|but|e|and)?\s*(?:non\s+(?:e|è|sono|era|erano)\s+stat[oaie]\s+(?:raccolt[oaie]|registrat[oaie]|acquisit[oaie])\s+(?:alcun[oa]?|nessun[oa]?)\s+(?:dato|informazione|osservazione|lettura|misura|riscontro)(?:\s+diagnostic[oaie])?|no\s+(?:diagnostic\s+)?(?:data|information|observations?|readings?|measurements?|checks?|results?)\s+(?:has|have|had|was|were)\s+(?:been\s+)?(?:collected|recorded|acquired|made|performed|available))\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"^(?:ma|but|e|and|né|neither)?\s*(?:non\s+(?:ho|hai|ha|abbiamo|avete|hanno)\s+(?:ancora\s+)?{_IT_PART}\b|non\s+(?:e|è|sono|era|erano)\s+stat[oaie]\s+{_IT_PART}\b|(?:(?:i|we|you|they|he|she|it)\s+)?(?:have|has|had)\s+not\s+(?:yet\s+)?{_EN_VERB}\b|(?:(?:i|we|you|they|he|she|it)\s+)?(?:haven't|hasn't|hadn't)\s+(?:yet\s+)?{_EN_VERB}\b|(?:was|were|is|are)\s+not\s+(?:yet\s+)?{_EN_VERB}\b)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:ma|but|e|and)?\s*(?:non\s+(?:ho|hai|ha|abbiamo|avete|hanno)\s+(?:effettuat[oaie]|eseguit[oaie]|fatt[oaie])|non\s+(?:posso|puoi|può|possiamo|potete|possono)\s+(?:effettuare|eseguire|fare)|(?:(?:i|we|you|they|he|she|it)\s+)?(?:have|has|had|did)\s+not\s+(?:perform|performed|carry\s+out|carried\s+out|make|made))\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(?:e|è|risulta)?\s*stat[oaie]\s+(?:effettuat[oaie]|eseguit[oaie]|fatt[oaie])$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.lstrip(" :,-–—")
    return _normalize(text)


def _extract_missing_items(spans_text: Sequence[str]) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for span in spans_text:
        cleaned = _strip_epistemic_prefix(span)
        if not cleaned:
            continue
        # Lists introduced after a global statement commonly use commas plus
        # nessun/no.  Splitting is only for clarification/debug; it never affects
        # the retrieval query, which is removed by exact character spans above.
        chunks = re.split(
            r"\s*(?:,|;|\bné\b|\bne\s+tantomeno\b|\bnor\b|\be\s+(?=nessun)|\band\s+(?=no\b)|\be\s+(?=non\s+(?:ho|ha|abbiamo|hanno)\b)|\band\s+(?=(?:have|has|had)\s+not\b)|\bor\s+(?=(?:checked|measured|read|verified|observed|recorded)\b)|\bo\s+(?=(?:controllato|misurato|letto|verificato|osservato|registrato)\b))\s*",
            cleaned,
            flags=re.IGNORECASE,
        )
        for chunk in chunks:
            item = _normalize(chunk).strip(" ,;:.-–—")
            item = re.sub(
                r"^(?:nessun[oa]?|alcun[oa]?|no|not\s+any)\s+",
                "",
                item,
                flags=re.IGNORECASE,
            )
            item = re.sub(
                r"^(?:ma|but|e|and|or|o)\s+",
                "",
                item,
                flags=re.IGNORECASE,
            )
            item = re.sub(
                rf"^(?:checked|measured|read|verified|observed|recorded|collected|inspected|tested|controllato|misurato|letto|verificato|osservato|registrato|raccolto|ispezionato|testato)\s+",
                "",
                item,
                flags=re.IGNORECASE,
            )
            passive_it = re.match(
                rf"^(?P<subject>.+?)\s+non\s+(?:e|è|sono|era|erano)\s+stat[oaie]\s+{_IT_PART}\b",
                item,
                flags=re.IGNORECASE,
            )
            passive_en = re.match(
                rf"^(?P<subject>.+?)\s+(?:was|were|is|are)\s+not\s+(?:yet\s+)?{_EN_VERB}\b",
                item,
                flags=re.IGNORECASE,
            )
            if passive_it is not None:
                item = passive_it.group("subject")
            elif passive_en is not None:
                item = passive_en.group("subject")
            item = _normalize(item).strip(" ,;:.-–—")
            if len(item) < 3:
                continue
            key = _fold(item)
            if key in seen:
                continue
            seen.add(key)
            items.append(item[:260])
            if len(items) >= 12:
                return tuple(items)
    return tuple(items)


def _observation_specificity(observed_text: str) -> int:
    text = _normalize(observed_text)
    if not text:
        return 0
    # Remove certification labels and the explicit meta-question before counting.
    probe = text
    for pattern in _META_QUESTION_PATTERNS:
        probe = pattern.sub(" ", probe)
    generic_only = probe
    for pattern in _GENERIC_EFFECT_PATTERNS:
        generic_only = pattern.sub(" ", generic_only)
    generic_only = _clean_after_removal(generic_only)

    hits = 0
    for pattern in _OBSERVATION_PATTERNS:
        if pattern.search(generic_only):
            hits += 1
    return hits


def _clarification_question(
    missing_information: Sequence[str], response_language: str
) -> str:
    is_en = str(response_language or "").lower().startswith("en")
    items = [_normalize(item) for item in missing_information if _normalize(item)][:8]
    if items:
        joined = "; ".join(items)
        if is_en:
            return (
                "No cause is demonstrated by the observations currently available. "
                f"Before ranking causes, collect the information marked as missing: {joined}. "
                "What results do you obtain?"
            )
        return (
            "Nessuna causa è dimostrata dalle osservazioni attualmente disponibili. "
            f"Prima di attribuire priorità alle cause, raccogli le informazioni indicate come mancanti: {joined}. "
            "Quali risultati ottieni?"
        )
    if is_en:
        return (
            "No specific cause can be ranked from the observations currently available. "
            "Please provide the exact HMI message or state, the phase of the cycle, and at least one measured or directly observed condition."
        )
    return (
        "Nessuna causa specifica può essere ordinata con le osservazioni attualmente disponibili. "
        "Indica il messaggio o stato HMI esatto, la fase del ciclo e almeno una condizione misurata o osservata direttamente."
    )


def analyze_diagnostic_query(
    query: str,
    *,
    response_language: str = "",
) -> DiagnosticQueryProfile:
    original = _normalize(query)
    spans, explicit_global = _missing_spans(original)
    span_texts = tuple(_normalize(original[start:end]) for start, end in spans)
    retrieval_query = _remove_spans(original, spans)
    missing_information = _extract_missing_items(span_texts)
    concrete_observation_count = _observation_specificity(retrieval_query)

    # A global, explicit absence of diagnostic data is an epistemic stop condition
    # only when no separate cause-discriminating observation survives removal.
    # Multiple unobserved clauses plus no concrete observation is treated the same.
    force = bool(
        spans and concrete_observation_count <= 0
    )
    reason = ""
    if force:
        reason = (
            "explicit_global_absence_without_concrete_observation"
            if explicit_global
            else "unobserved_information_without_concrete_observation"
        )
    elif spans:
        reason = "unobserved_information_removed_from_retrieval"
    else:
        reason = "no_explicit_unobserved_information"

    language = str(response_language or "").strip().lower()
    if not language:
        folded = _fold(original)
        language = "en" if re.search(r"\b(?:the|after|before|what|which|machine|checked)\b", folded) else "it"

    return DiagnosticQueryProfile(
        original_query=original,
        retrieval_query=retrieval_query or original,
        observed_text=retrieval_query,
        missing_spans=span_texts,
        missing_information=missing_information,
        explicit_global_absence=explicit_global,
        concrete_observation_count=concrete_observation_count,
        force_clarification=force,
        clarification_question=_clarification_question(missing_information, language),
        response_language=language,
        reason=reason,
    )


def profile_from_mapping(
    value: Any,
    *,
    fallback_query: str = "",
    response_language: str = "",
) -> DiagnosticQueryProfile:
    if isinstance(value, DiagnosticQueryProfile):
        return value
    if not isinstance(value, Mapping):
        return analyze_diagnostic_query(
            fallback_query,
            response_language=response_language,
        )
    original = _normalize(value.get("original_query") or fallback_query)
    return DiagnosticQueryProfile(
        original_query=original,
        retrieval_query=_normalize(value.get("retrieval_query") or original),
        observed_text=_normalize(value.get("observed_text") or value.get("retrieval_query") or original),
        missing_spans=tuple(_normalize(x) for x in (value.get("missing_spans") or []) if _normalize(x))[:12],
        missing_information=tuple(_normalize(x) for x in (value.get("missing_information") or []) if _normalize(x))[:12],
        explicit_global_absence=bool(value.get("explicit_global_absence")),
        concrete_observation_count=max(0, int(value.get("concrete_observation_count") or 0)),
        force_clarification=bool(value.get("force_clarification")),
        clarification_question=_normalize(value.get("clarification_question")) or _clarification_question([], response_language),
        response_language=_normalize(value.get("response_language") or response_language),
        policy_version=_normalize(value.get("policy_version") or POLICY_VERSION),
        reason=_normalize(value.get("reason")),
    )


def _missing_dominated(value: Any, profile: DiagnosticQueryProfile) -> bool:
    text = _normalize(value)
    if not text or not profile.missing_information:
        return False
    missing_score = max(
        (_missing_overlap(text, item) for item in profile.missing_information),
        default=0.0,
    )
    observed_score = _overlap(text, profile.observed_text)
    # Short technical labels should be removed when they occur only inside an
    # unobserved list. Longer mixed evidence text is retained only when its
    # observed support is at least as strong as its missing-information overlap.
    return bool(missing_score >= 0.34 and observed_score < max(0.28, missing_score * 0.72))


def _mentions_missing_information(
    value: Any,
    profile: DiagnosticQueryProfile,
) -> bool:
    """Conservative postcondition for fields that can actively steer retrieval.

    A router-generated query/facet/clue that mentions an explicitly unobserved
    variable is discarded as a whole.  The composition root already supplies the
    retrieval-safe observed query, so retaining a mixed router rewrite is not needed
    for recall and could reintroduce the very variable removed from the user text.
    """
    text = _normalize(value)
    if not text or not profile.missing_information:
        return False
    return any(
        _missing_overlap(text, item) >= 0.50
        for item in profile.missing_information
    )


def _clean_string_list(
    values: Any,
    profile: DiagnosticQueryProfile,
    *,
    limit: int,
    reject_any_missing_mention: bool = False,
) -> tuple[list[str], int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return [], 0
    out: list[str] = []
    removed = 0
    seen: set[str] = set()
    for raw in values:
        text = _normalize(raw)
        if not text:
            continue
        reject = (
            _mentions_missing_information(text, profile)
            if reject_any_missing_mention
            else _missing_dominated(text, profile)
        )
        if reject:
            removed += 1
            continue
        key = _fold(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out, removed


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    """Bounded semantic surface for validating router-selected evidence IDs."""
    values: list[str] = []
    for key in (
        "title",
        "source_title",
        "source_name",
        "document_title",
        "procedure_title",
        "step_title",
        "ps_title",
        "snippet_clean",
        "snippet",
        "text",
        "chunk_full",
        "section_title",
        "category",
    ):
        text = _normalize(candidate.get(key))
        if text:
            values.append(text[:1600])
    return _normalize(" ".join(values))[:6000]


def _clean_relevant_evidence_ids(
    values: Any,
    profile: DiagnosticQueryProfile,
    evidence_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[str], int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return [], 0
    by_id: dict[str, Mapping[str, Any]] = {}
    for candidate in evidence_candidates:
        if not isinstance(candidate, Mapping):
            continue
        citation_id = _normalize(candidate.get("citation_id"))
        if citation_id:
            by_id[citation_id] = candidate

    out: list[str] = []
    removed = 0
    seen: set[str] = set()
    for raw in values:
        citation_id = _normalize(raw)
        if not citation_id or citation_id in seen:
            continue
        candidate = by_id.get(citation_id)
        if candidate is not None:
            surface = _candidate_text(candidate)
            if surface and _missing_dominated(surface, profile):
                removed += 1
                continue
        seen.add(citation_id)
        out.append(citation_id)
        if len(out) >= 16:
            break
    return out, removed


def sanitize_router_payload(
    raw: Mapping[str, Any],
    profile: DiagnosticQueryProfile,
    *,
    evidence_candidates: Sequence[Mapping[str, Any]] = (),
) -> RouterSanitizationResult:
    """Remove unobserved variables from router-positive diagnostic fields.

    The router may read the original natural-language request, but a phrase listed as
    not checked/measured/read cannot become a subsystem, clue, exclusion, retrieval
    query or mandatory facet.  This postcondition is deterministic and independent
    from machine vocabulary.
    """
    out = dict(raw or {})
    removed_counts: dict[str, int] = {}
    list_limits = {
        "diagnostic_subsystems": 6,
        "diagnostic_observables": 8,
        "diagnostic_operating_conditions": 8,
        "diagnostic_discriminants": 8,
        "diagnostic_exclusions": 8,
        "dense_queries": 5,
        "lexical_queries": 7,
        "exact_terms": 12,
        "required_facets": 10,
    }
    for key, limit in list_limits.items():
        cleaned, removed = _clean_string_list(
            out.get(key),
            profile,
            limit=limit,
            reject_any_missing_mention=True,
        )
        out[key] = cleaned
        if removed:
            removed_counts[key] = removed

    facets_out: list[dict[str, Any]] = []
    facets_removed = 0
    for raw_item in out.get("facet_queries") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        facet = _normalize(item.get("facet"))
        if not facet or _mentions_missing_information(facet, profile):
            facets_removed += 1
            continue
        item["facet"] = facet
        for key, limit in (("dense_queries", 4), ("lexical_queries", 5), ("exact_terms", 8)):
            cleaned, removed = _clean_string_list(
                item.get(key),
                profile,
                limit=limit,
                reject_any_missing_mention=True,
            )
            item[key] = cleaned
            if removed:
                removed_counts[f"facet_queries.{key}"] = removed_counts.get(f"facet_queries.{key}", 0) + removed
        facets_out.append(item)
        if len(facets_out) >= 10:
            break
    out["facet_queries"] = facets_out
    if facets_removed:
        removed_counts["facet_queries"] = facets_removed

    merged_missing: list[str] = []
    seen_missing: set[str] = set()
    for value in list(profile.missing_information) + list(out.get("missing_information") or []):
        text = _normalize(value)
        if not text:
            continue
        key = _fold(text)
        if key in seen_missing:
            continue
        seen_missing.add(key)
        merged_missing.append(text)
        if len(merged_missing) >= 12:
            break
    out["missing_information"] = merged_missing[:8]

    relevant_ids, removed_ids = _clean_relevant_evidence_ids(
        out.get("relevant_evidence_ids"),
        profile,
        evidence_candidates,
    )
    out["relevant_evidence_ids"] = relevant_ids
    if removed_ids:
        removed_counts["relevant_evidence_ids"] = removed_ids

    summary = {
        "policy_version": profile.policy_version,
        "missing_information_count": len(profile.missing_information),
        "removed_counts": dict(sorted(removed_counts.items())),
        "force_clarification": bool(profile.force_clarification),
    }
    out["diagnostic_query_state"] = profile.public_summary()
    out["diagnostic_query_sanitization"] = summary
    return RouterSanitizationResult(payload=out, summary=summary)
