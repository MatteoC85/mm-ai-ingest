"""P4-C6: existing query policies and diagnostic scores, without semantic changes.

All eighteen implementations below are extracted from the reviewed baseline.
Existing lexical heuristics, thresholds, limits, and exception paths are retained.
The component-inference helper still delegates to the existing model callback;
this module does not import providers, databases, FastAPI, main, or configuration.
Dependencies are passed at call time so the composition-root hooks and configured
values retain their historical behavior. No extra model call, cache or fallback
is introduced. This extraction is not a claim of cross-machine/language quality.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class InferMachineComponentsRuntime:
    DIAGNOSTIC_EVIDENCE_MODEL: Any
    OPENAI_CHAT_MODEL: Any
    ROOT_CAUSE_INTENT_MODEL: Any
    _openai_chat_json_models: Callable[..., Any]


def infer_machine_components(q: str, *, runtime: InferMachineComponentsRuntime) -> list[str]:
    DIAGNOSTIC_EVIDENCE_MODEL = runtime.DIAGNOSTIC_EVIDENCE_MODEL
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    ROOT_CAUSE_INTENT_MODEL = runtime.ROOT_CAUSE_INTENT_MODEL
    _openai_chat_json_models = runtime._openai_chat_json_models
    if not q:
        return []

    schema = {
        "name": "component_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "components": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["components"]
        }
    }

    system_msg = (
        "You are an industrial machine expert. "
        "Given a machine symptom, list machine components likely involved. "
        "Return short component names only."
    )

    user_msg = f"Symptom:\n{q}"

    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[DIAGNOSTIC_EVIDENCE_MODEL, ROOT_CAUSE_INTENT_MODEL, OPENAI_CHAT_MODEL],
            json_schema=schema,
            timeout=20,
        )

        comps = parsed.get("components") or []

        out = []
        seen = set()

        for c in comps:
            c = str(c).strip().lower()
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)

        return out[:6]

    except Exception:
        return []


@dataclass(frozen=True)
class BuildDiagnosticQueriesRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def build_diagnostic_queries(q: str, inferred_components: list[str], *, runtime: BuildDiagnosticQueriesRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q = re.sub(r"\s+", " ", (q or "").strip())
    comps = [re.sub(r"\s+", " ", str(x).strip()) for x in (inferred_components or []) if str(x).strip()]
    if not q:
        return []

    q_low = _normalize_unicode_advanced(q).lower()

    out: list[str] = []
    seen = set()

    def add(s: str):
        s = re.sub(r"\s+", " ", (s or "").strip())
        if not s:
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    def has_any(stems: list[str]) -> bool:
        return any(st in q_low for st in stems)

    symptom_aliases: list[str] = []

    if has_any(["vibr", "oscillat", "oscillaz"]):
        symptom_aliases.extend(["vibration", "vibrazione"])
    if has_any(["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"]):
        symptom_aliases.extend(["noise", "rumore"])
    if has_any(["overheat", "surriscal", "temperat", "hot", "cald"]):
        symptom_aliases.extend(["overheating", "surriscaldamento"])
    if has_any(["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"]):
        symptom_aliases.extend(["jam", "blocco"])
    if has_any(["lubric", "lubrif", "oil", "olio", "grease", "grasso"]):
        symptom_aliases.extend(["lubrication", "lubrificazione"])
    if has_any(["feed", "advance", "avanz", "wire", "filo", "material"]):
        symptom_aliases.extend(["feed", "avanzamento"])
    if has_any(["bend", "bending", "pieg", "forming", "formatura"]):
        symptom_aliases.extend(["bending", "piegatura"])

    add(q)

    if comps:
        add(q + " " + " ".join(comps[:3]))

    add(f"root cause {q}")
    add(f"causa {q}")

    if symptom_aliases:
        add(f"{q} {symptom_aliases[0]}")
        if len(symptom_aliases) > 1:
            add(f"{q} {symptom_aliases[1]}")

    if comps:
        add(f"{q} {comps[0]}")
        add(f"root cause {comps[0]} {q}")

    add(f"diagnosi {q}")

    return out[:8]


@dataclass(frozen=True)
class CollectCandidateKeywordsRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def collect_candidate_keywords(q: str, inferred_components: list[str], *, runtime: CollectCandidateKeywordsRuntime) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q_norm = _normalize_unicode_advanced(q or "").lower()
    comps = [str(x).strip().lower() for x in (inferred_components or []) if str(x).strip()]

    out = []
    seen = set()

    def add(x: str):
        x = re.sub(r"\s+", " ", (x or "").strip().lower())
        if not x or x in seen:
            return
        seen.add(x)
        out.append(x)

    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from",
        "machine", "problem", "issue", "fault", "cause", "possible", "probable",
        "abnormal", "anomalous", "anomaly", "diagnosis",
        "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "dei", "delle",
        "con", "per", "tra", "fra", "sul", "sulla", "macchina", "problema",
        "guasto", "causa", "possibile", "probabile", "anomalo", "anomala",
        "anomali", "anomale", "durante", "quando", "mentre", "dopo", "prima",
        "nel", "nella", "su"
    }

    for c in comps[:6]:
        add(c)

    for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", q_norm):
        if tok not in stopwords:
            add(tok)

    alias_groups = [
        (["vibr", "oscillat", "oscillaz"], ["vibration", "vibrazione", "oscillation", "oscillazione"]),
        (["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"], ["noise", "rumore", "rattle", "stridore"]),
        (["overheat", "surriscal", "temperat", "hot", "cald"], ["overheating", "surriscaldamento", "temperature", "temperatura"]),
        (["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"], ["jam", "blocco", "stoppage", "arresto"]),
        (["lubric", "lubrif", "oil", "olio", "grease", "grasso"], ["lubrication", "lubrificazione", "oil", "olio", "grease", "grasso"]),
        (["feed", "advance", "avanz", "wire", "filo", "material"], ["feed", "avanzamento", "wire", "filo", "materiale"]),
        (["bend", "bending", "pieg", "forming", "formatura"], ["bending", "piegatura", "forming", "formatura"]),
    ]

    for stems, aliases in alias_groups:
        if any(st in q_norm for st in stems):
            for alias in aliases:
                add(alias)

    return out[:12]


@dataclass(frozen=True)
class CountQueryTokensRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def count_query_tokens(q: str, *, runtime: CountQueryTokensRuntime) -> int:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    return len(re.findall(r"[a-zà-öø-ÿ0-9]{2,}", _normalize_unicode_advanced(q or "").lower()))


@dataclass(frozen=True)
class QuerySymptomProfileRuntime:
    _dedup_text_values: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def query_symptom_profile(q: str, *, runtime: QuerySymptomProfileRuntime) -> dict:
    _dedup_text_values = runtime._dedup_text_values
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q_norm = re.sub(r"\s+", " ", _normalize_unicode_advanced(q or "")).strip().lower()
    tokens = re.findall(r"[a-zà-öø-ÿ0-9]{2,}", q_norm)

    classes: list[str] = []
    if any(st in q_norm for st in ["vibr", "oscillat", "oscillaz"]):
        classes.append("vibration")
    if any(st in q_norm for st in ["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"]):
        classes.append("noise")

    # A commanded/emergency stop is an operating condition, not a mechanical jam.
    # Treat stop/arrest wording as a jam signal only when it is not explicitly part
    # of an emergency-stop/reset context.  This keeps the classifier generic while
    # preventing safety-chain questions from being expanded into unrelated
    # mechanical-stop retrieval queries.
    emergency_stop_context = bool(
        re.search(
            r"\b(?:arrest[oa]\s+(?:di\s+)?emergenza|pulsant[ei]\s+(?:di\s+)?emergenza|"
            r"emergency\s+stop|e-?stop|after\s+an?\s+emergency)\b",
            q_norm,
        )
    )
    jam_markers = ["jam", "stuck", "incepp", "impunt", "blocc", "si ferma", "stops unexpectedly"]
    uncommanded_stop = bool(
        re.search(r"\b(?:si\s+arresta|si\s+ferma|arresto\s+anomalo|unexpected\s+stop|stops?)\b", q_norm)
    )
    if any(st in q_norm for st in jam_markers) or (uncommanded_stop and not emergency_stop_context):
        classes.append("jam")

    if (
        any(
            st in q_norm
            for st in [
                "non parte", "non si avvia", "non si abilita", "non abilita",
                "does not start", "won't start", "will not start", "cannot start",
                "can't start", "doesnt start", "does not enable", "cannot enable",
                "can't enable", "not enabled",
            ]
        )
        or ("start" in q_norm and any(st in q_norm for st in [" non ", " not ", "won't", "will not", "cannot", "can't"]))
    ):
        classes.append("no_start")

    has_bending_anchor = any(st in q_norm for st in ["bend", "bending", "pieg", "forming", "formatura", "press", "pressa", "tool", "die", "stampo"])
    has_feed_anchor = any(st in q_norm for st in ["feed", "advance", "advancement", "avanz", "wire", "filo", "strip", "nastro", "material"])
    has_process_anchor = has_bending_anchor or has_feed_anchor or any(st in q_norm for st in ["cut", "cutting", "taglio", "drill", "fora", "weld", "sald", "straighten", "raddrizz"])
    automatic_mode = any(st in q_norm for st in ["automatic", "automatico", "ciclo automatico", "automatic cycle", "auto mode", "modalità automatica"])

    has_support_anchor = any(
        st in q_norm
        for st in [
            "lubric", "lubrif", "oil", "olio", "grease", "grasso",
            "electr", "elettric", "phase", "fasi", "voltage", "tensione",
            "pneumat", "hydraul", "idraulic", "pressure", "pression",
            "safety", "sicurezza", "door", "porta", "interlock", "microinter",
            "plc", "encoder", "sensor", "sensore", "panel", "quadro"
        ]
    )

    generic_symptom = bool(classes) and not has_process_anchor and not has_support_anchor and len(tokens) <= 6
    if "no_start" in classes and automatic_mode:
        generic_symptom = False

    return {
        "classes": _dedup_text_values(classes, limit=4),
        "has_bending_anchor": has_bending_anchor,
        "has_feed_anchor": has_feed_anchor,
        "has_process_anchor": has_process_anchor,
        "has_support_anchor": has_support_anchor,
        "automatic_mode": automatic_mode,
        "generic_symptom": generic_symptom,
    }


@dataclass(frozen=True)
class EffectiveSimilarityThresholdRuntime:
    ASK_SHORT_QUERY_SIM_THRESHOLD: Any
    _count_query_tokens: Callable[..., Any]


def effective_similarity_threshold(
    q: str,
    *,
    planner: Optional[dict] = None,
    base_threshold: float,
    runtime: EffectiveSimilarityThresholdRuntime,
) -> float:
    ASK_SHORT_QUERY_SIM_THRESHOLD = runtime.ASK_SHORT_QUERY_SIM_THRESHOLD
    _count_query_tokens = runtime._count_query_tokens
    token_count = _count_query_tokens(q)
    style = str((planner or {}).get("query_style") or "").strip().lower()

    if style in {"telegraphic", "identifier_lookup", "contact_lookup"} or token_count <= 5:
        return min(base_threshold, ASK_SHORT_QUERY_SIM_THRESHOLD)

    return base_threshold


@dataclass(frozen=True)
class ContentTermSetRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def content_term_set(text: str, limit: int = 80, *, runtime: ContentTermSetRuntime) -> set[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    stopwords = {
        "the", "and", "for", "with", "when", "while", "during", "after", "before", "from", "into", "onto",
        "this", "that", "these", "those", "question", "answer", "issue", "problem", "machine", "system",
        "document", "documents", "manual", "procedure", "step", "solution",
        "il", "lo", "la", "i", "gli", "le", "con", "per", "quando", "durante", "mentre", "dopo", "prima",
        "questo", "questa", "questi", "queste", "domanda", "risposta", "problema", "macchina", "sistema",
        "procedura", "step", "soluzione", "documenti", "documento",
    }
    out = []
    seen = set()
    for tok in re.findall(r"[a-zà-öø-ÿ0-9]{3,}", _normalize_unicode_advanced(text or "").lower()):
        if tok in stopwords:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return set(out)


@dataclass(frozen=True)
class PlannerQueryTermSetRuntime:
    _content_term_set: Callable[..., Any]


def planner_query_term_set(q: str, planner: Optional[dict], *, runtime: PlannerQueryTermSetRuntime) -> set[str]:
    _content_term_set = runtime._content_term_set
    texts = [q]
    if isinstance(planner, dict):
        texts.append(str(planner.get("normalized_query") or ""))
        texts.extend(list(planner.get("lexical_queries") or []))
    return _content_term_set("\n".join(texts), limit=60)


@dataclass(frozen=True)
class TermOverlapScoreRuntime:
    math: Any


def term_overlap_score(query_terms: set[str], text_terms: set[str], *, runtime: TermOverlapScoreRuntime) -> float:
    math = runtime.math
    if not query_terms or not text_terms:
        return 0.0
    inter = len(query_terms & text_terms)
    if inter <= 0:
        return 0.0
    return inter / math.sqrt(float(len(query_terms)) * float(len(text_terms)))


@dataclass(frozen=True)
class CandidateSpecificityScoreRuntime:
    _content_term_set: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]


def candidate_specificity_score(item: dict, *, runtime: CandidateSpecificityScoreRuntime) -> float:
    _content_term_set = runtime._content_term_set
    _source_type_from_document_id = runtime._source_type_from_document_id
    text = (item.get("chunk_full") or item.get("snippet") or "").strip()
    source_type = _source_type_from_document_id(item.get("bubble_document_id") or "")
    text_terms = _content_term_set(text, limit=100)
    char_len = len(text)
    line_count = len([ln for ln in text.split("\n") if ln.strip()])

    score = 0.0
    score += min(0.07, 0.0018 * len(text_terms))

    if 160 <= char_len <= 1800:
        score += 0.04
    elif char_len < 90:
        score -= 0.08
    elif char_len < 160:
        score -= 0.04

    if line_count >= 2:
        score += 0.02

    if source_type == "ps" and len(text_terms) < 20:
        score -= 0.08
    elif source_type in {"procedure", "step"} and len(text_terms) < 18:
        score -= 0.05

    return max(-0.14, min(0.14, score))


@dataclass(frozen=True)
class RootCauseChunkSignalSummaryRuntime:
    _extract_section_from_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def root_cause_chunk_signal_summary(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    *, runtime: RootCauseChunkSignalSummaryRuntime,
) -> dict:
    _extract_section_from_text = runtime._extract_section_from_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    txt = _normalize_unicode_advanced(chunk_text or "").lower()
    if not txt:
        return {}

    section = _normalize_unicode_advanced(_extract_section_from_text(chunk_text) or "").lower()
    q_low = _normalize_unicode_advanced(q or "").lower()

    diag_terms = []
    seen_diag = set()
    for x in diagnostic_keywords or []:
        x = re.sub(r"\s+", " ", str(x).strip().lower())
        if len(x) < 3 or x in seen_diag:
            continue
        seen_diag.add(x)
        diag_terms.append(x)

    overview_section_markers = [
        "overview",
        "general features",
        "general description",
        "caratteristiche generali",
        "descrizione generale",
        "technical data",
        "specifications",
        "dati tecnici",
        "caratteristiche tecniche",
        "intended use",
        "destinazione d'uso",
    ]

    description_section_markers = [
        "descrizione della macchina",
        "machine description",
        "description of the machine",
        "descrizione macchina",
    ]

    boilerplate_section_markers = overview_section_markers + [
        "safety",
        "warning",
        "warnings",
        "sicurezza",
        "avvertenze",
        "installation",
        "installazione",
        "electrical connections",
        "collegamenti elettrici",
        "wiring",
        "transport",
        "trasporto",
        "storage",
        "stoccaggio",
        "commissioning",
        "messa in servizio",
        "start-up",
        "startup",
        "prima accensione",
        "messa in moto",
        "foundation",
        "fondazione",
        "positioning",
        "posizionamento",
    ]

    startup_install_markers = [
        "before starting",
        "before start-up",
        "before startup",
        "prima di avviare",
        "prima dell'avviamento",
        "prima della messa in moto",
        "before commissioning",
        "messa in servizio",
        "messa in moto",
        "start the machine",
        "avviare la macchina",
        "control panel",
        "quadro di comando",
        "network voltage",
        "tensione di rete",
        "power supply",
        "alimentazione",
        "minimum level",
        "livello minimo",
        "installation",
        "installazione",
        "transport",
        "trasporto",
        "storage",
        "stoccaggio",
        "foundation",
        "fondazione",
        "positioning",
        "posizionamento",
    ]

    positioning_markers = [
        "positioning",
        "posizionamento",
        "foundation",
        "fondazione",
        "planarity",
        "planarità",
        "level",
        "livella",
        "levelling",
        "leveling",
        "livellamento",
        "support surface",
        "piano di appoggio",
        "rubber shims",
        "spessori di gomma",
        "threaded holes",
        "fori filettati",
        "mounting holes",
        "fori di fondazione",
        "near a wall",
        "vicino ad un muro",
    ]

    safety_access_markers = [
        "work area",
        "zona di lavoro",
        "access door",
        "porta di accesso",
        "porte protette",
        "safety door",
        "safety doors",
        "micro switch",
        "micro switches",
        "microswitch",
        "microinterruttor",
        "automatic cycle",
        "ciclo automatico",
        "protective guard",
        "protective guards",
        "riparo",
        "ripari",
    ]

    acoustic_protection_markers = [
        "acoustic",
        "noise emission",
        "noise emissions",
        "noise level",
        "sound pressure",
        "sound insulation",
        "soundproof",
        "soundproofing",
        "emissioni sonore",
        "livello di rumore",
        "pressione sonora",
        "isolamento acustico",
        "fonoassorb",
        "rumoros",
        "protective panel",
        "protective panels",
        "pannelli di protezione",
    ]

    lube_control_markers = [
        "lubrication circuit",
        "circuito di lubrificazione",
        "automatic lubrication",
        "lubrificazione automatica",
        "pressure switch",
        "pressostato",
        "oil level",
        "livello olio",
        "minimum level",
        "livello minimo",
        "oil tank",
        "serbatoio",
        "pressure drop",
        "cali pressione",
    ]

    strong_component_markers = [
        "bearing",
        "cuscinet",
        "gear",
        "ingran",
        "gearbox",
        "ridutt",
        "shaft",
        "albero",
        "belt",
        "cinghia",
        "chain",
        "catena",
        "roller",
        "rullo",
        "guide",
        "guida",
        "motor",
        "motore",
        "sensor",
        "sensore",
        "encoder",
        "valve",
        "valvol",
        "cylinder",
        "cilindr",
        "pump",
        "pompa",
        "brake",
        "freno",
        "alignment",
        "alline",
        "clearance",
        "gioco",
        "backlash",
        "friction",
        "attrit",
        "slitta",
        "slide",
        "die",
        "stampo",
    ]

    process_markers = [
        "feed",
        "advance",
        "avanz",
        "wire",
        "filo",
        "bend",
        "bending",
        "pieg",
        "forming",
        "formatura",
        "straighten",
        "straightening",
        "raddrizz",
    ]

    symptom_groups = [
        (["vibr", "oscillat", "oscillaz"], ["vibrat", "vibraz", "oscillat", "oscill"]),
        (["noise", "noisy", "rumor", "rumore", "rumoros", "sound", "rattle", "squeal", "strid", "sfreg"], ["noise", "rumor", "rumore", "rumoros", "rattle", "squeal", "strid", "sfreg"]),
        (["overheat", "surriscal", "temperat", "hot", "cald"], ["overheat", "surriscal", "temperat", "hot", "cald"]),
        (["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"], ["jam", "block", "stuck", "stop", "bloc", "ferma", "arrest"]),
        (["lubric", "lubrif", "oil", "olio", "grease", "grasso"], ["lubric", "lubrif", "oil", "olio", "grease", "grasso"]),
        (["feed", "advance", "avanz", "wire", "filo", "material"], ["feed", "advance", "avanz", "wire", "filo", "material"]),
        (["bend", "bending", "pieg", "forming", "formatura"], ["bend", "bending", "pieg", "forming", "formatura"]),
    ]

    symptom_markers: list[str] = []
    for query_stems, chunk_stems in symptom_groups:
        if any(st in q_low for st in query_stems):
            symptom_markers.extend(chunk_stems)

    def count_hits(markers: list[str], hay: str) -> int:
        return sum(1 for m in markers if m and m in hay)

    spec_markers = [
        "noise level",
        "livello di rumore",
        "sound pressure",
        "pressione sonora",
        "dimensions",
        "dimensioni",
        "weight",
        "peso",
        "voltage",
        "tensione",
        "frequency",
        "frequenza",
    ]

    query_install_related = any(
        st in q_low
        for st in [
            "install",
            "startup",
            "start-up",
            "start",
            "avvi",
            "messa in servizio",
            "messa in moto",
            "commission",
            "elettric",
            "power",
            "alimentaz",
            "tension",
            "posizion",
            "livell",
            "fondaz",
            "planarit",
            "piano di appoggio",
            "support surface",
            "setup",
            "mount",
        ]
    )
    query_safety_related = any(
        st in q_low
        for st in [
            "sicur",
            "safety",
            "ripar",
            "guard",
            "porta",
            "door",
            "microinter",
            "interlock",
            "arrest",
            "stop",
            "emerg",
            "zona di lavoro",
            "work area",
        ]
    )

    query_lube_related = any(
        st in q_low
        for st in [
            "lubric",
            "lubrif",
            "oil",
            "olio",
            "grease",
            "grasso",
            "pressost",
            "pressure switch",
        ]
    )

    return {
        "diag_hits": count_hits(diag_terms[:12], txt),
        "section_diag_hits": count_hits(diag_terms[:12], section),
        "boilerplate_section_hit": any(m in section for m in boilerplate_section_markers),
        "overview_section_hit": any(m in section for m in overview_section_markers),
        "description_section_hit": any(m in section for m in description_section_markers),
        "startup_install_hits": count_hits(startup_install_markers, txt),
        "positioning_hits": count_hits(positioning_markers, txt),
        "safety_access_hits": count_hits(safety_access_markers, txt),
        "acoustic_protection_hits": count_hits(acoustic_protection_markers, txt),
        "lube_control_hits": count_hits(lube_control_markers, txt),
        "strong_component_hits": count_hits(strong_component_markers, txt),
        "process_hits": count_hits(process_markers, txt),
        "symptom_hits": count_hits(list(dict.fromkeys(symptom_markers)), txt),
        "spec_hits": count_hits(spec_markers, txt),
        "query_install_related": query_install_related,
        "query_safety_related": query_safety_related,
        "query_lube_related": query_lube_related,
    }


@dataclass(frozen=True)
class ShouldDownrankGenericRootCauseChunkRuntime:
    _root_cause_chunk_signal_summary: Callable[..., Any]


def should_downrank_generic_root_cause_chunk(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    *, runtime: ShouldDownrankGenericRootCauseChunkRuntime,
) -> bool:
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return False

    if (
        sig["description_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (
        sig["safety_access_hits"] >= 2
        and not sig["query_safety_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (sig["strong_component_hits"] >= 2 or sig["process_hits"] >= 1) and (
        sig["diag_hits"] >= 1 or sig["section_diag_hits"] >= 1
    ):
        return False

    if sig["symptom_hits"] >= 1 and sig["strong_component_hits"] >= 2:
        return False

    if sig["overview_section_hit"] and (
        sig["acoustic_protection_hits"] >= 1 or sig["safety_access_hits"] >= 2
    ):
        return True

    if (
        sig["overview_section_hit"]
        and sig["symptom_hits"] >= 1
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["boilerplate_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["startup_install_hits"] >= 3
        and not sig["query_install_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
    ):
        return True

    if (
        sig["positioning_hits"] >= 2
        and not sig["query_install_related"]
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["lube_control_hits"] >= 2
        and not sig["query_lube_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if sig["spec_hits"] >= 2 and sig["diag_hits"] == 0 and sig["symptom_hits"] == 0:
        return True

    return False


@dataclass(frozen=True)
class ShouldHardExcludeRootCauseChunkRuntime:
    _root_cause_chunk_signal_summary: Callable[..., Any]


def should_hard_exclude_root_cause_chunk(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    *, runtime: ShouldHardExcludeRootCauseChunkRuntime,
) -> bool:
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return False

    if (
        sig["description_section_hit"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    if (
        sig["safety_access_hits"] >= 2
        and not sig["query_safety_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["strong_component_hits"] == 0
        and sig["process_hits"] == 0
    ):
        return True

    if (sig["strong_component_hits"] >= 2 or sig["process_hits"] >= 1) and (
        sig["diag_hits"] >= 1 or sig["section_diag_hits"] >= 1
    ):
        return False

    if sig["symptom_hits"] >= 1 and sig["strong_component_hits"] >= 2:
        return False

    if sig["overview_section_hit"] and (
        sig["acoustic_protection_hits"] >= 1 or sig["safety_access_hits"] >= 2
    ):
        return True

    if (
        sig["overview_section_hit"]
        and sig["symptom_hits"] >= 1
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["startup_install_hits"] >= 4
        and not sig["query_install_related"]
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] <= 1
        and sig["process_hits"] == 0
    ):
        return True

    if (
        sig["positioning_hits"] >= 3
        and not sig["query_install_related"]
        and sig["process_hits"] == 0
        and sig["strong_component_hits"] <= 1
    ):
        return True

    if (
        sig["acoustic_protection_hits"] >= 2
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
    ):
        return True

    if (
        sig["spec_hits"] >= 3
        and sig["diag_hits"] == 0
        and sig["section_diag_hits"] == 0
        and sig["symptom_hits"] == 0
        and sig["strong_component_hits"] == 0
    ):
        return True

    return False


@dataclass(frozen=True)
class ScoreRootCauseChunkSemanticRuntime:
    _root_cause_chunk_signal_summary: Callable[..., Any]


def score_root_cause_chunk_semantic(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    *, runtime: ScoreRootCauseChunkSemanticRuntime,
) -> dict:
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    if not sig:
        return {
            "semantic_score": -0.30,
            "semantic_band": "weak",
        }

    score = 0.0

    # segnali positivi
    score += min(0.20, 0.05 * float(sig.get("diag_hits", 0)))
    score += min(0.18, 0.08 * float(sig.get("section_diag_hits", 0)))
    score += min(0.16, 0.06 * float(sig.get("symptom_hits", 0)))
    score += min(0.22, 0.05 * float(sig.get("strong_component_hits", 0)))
    score += min(0.18, 0.07 * float(sig.get("process_hits", 0)))

    # sinergie utili
    if sig.get("symptom_hits", 0) >= 1 and sig.get("strong_component_hits", 0) >= 1:
        score += 0.08
    if sig.get("diag_hits", 0) >= 1 and (
        sig.get("strong_component_hits", 0) >= 1 or sig.get("process_hits", 0) >= 1
    ):
        score += 0.08
    if sig.get("section_diag_hits", 0) >= 1 and sig.get("process_hits", 0) >= 1:
        score += 0.06

    # penalità generali
    if sig.get("description_section_hit"):
        score -= 0.28
    if sig.get("overview_section_hit"):
        score -= 0.14
    if sig.get("boilerplate_section_hit"):
        score -= 0.10

    if sig.get("startup_install_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= min(0.18, 0.04 * float(sig.get("startup_install_hits", 0)))

    if sig.get("positioning_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= min(0.22, 0.05 * float(sig.get("positioning_hits", 0)))

    if sig.get("safety_access_hits", 0) >= 2 and not sig.get("query_safety_related"):
        score -= min(0.22, 0.05 * float(sig.get("safety_access_hits", 0)))

    if sig.get("acoustic_protection_hits", 0) >= 1 and not sig.get("query_safety_related"):
        score -= min(0.18, 0.06 * float(sig.get("acoustic_protection_hits", 0)))

    if (
        sig.get("lube_control_hits", 0) >= 2
        and not sig.get("query_lube_related")
        and sig.get("process_hits", 0) == 0
    ):
        score -= min(0.14, 0.04 * float(sig.get("lube_control_hits", 0)))

    if sig.get("spec_hits", 0) >= 2:
        score -= min(0.14, 0.04 * float(sig.get("spec_hits", 0)))

    score = max(-0.45, min(0.45, score))

    if score >= 0.18:
        band = "strong"
    elif score >= 0.02:
        band = "medium"
    else:
        band = "weak"

    return {
        "semantic_score": score,
        "semantic_band": band,
    }


@dataclass(frozen=True)
class ScoreRootCauseCausalStrengthRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    _root_cause_chunk_signal_summary: Callable[..., Any]


def score_root_cause_causal_strength(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    *, runtime: ScoreRootCauseCausalStrengthRuntime,
) -> dict:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    txt = _normalize_unicode_advanced(chunk_text or "").lower()

    if not sig or not txt:
        return {
            "causal_strength_score": -0.25,
            "causal_strength_band": "weak",
        }

    score = 0.0

    # segnali di evidenza diretta sul gruppo/processo coinvolto
    if sig.get("strong_component_hits", 0) >= 2:
        score += 0.20
    elif sig.get("strong_component_hits", 0) == 1:
        score += 0.08

    if sig.get("process_hits", 0) >= 1:
        score += 0.16

    if sig.get("symptom_hits", 0) >= 1:
        score += 0.12

    if sig.get("section_diag_hits", 0) >= 1:
        score += 0.10
    elif sig.get("diag_hits", 0) >= 1:
        score += 0.06

    # sinergie: componente + processo/sintomo = evidenza forte
    if sig.get("strong_component_hits", 0) >= 1 and sig.get("process_hits", 0) >= 1:
        score += 0.10
    if sig.get("strong_component_hits", 0) >= 1 and sig.get("symptom_hits", 0) >= 1:
        score += 0.08

    # segnali di evidenza indiretta/collaterale
    if sig.get("lube_control_hits", 0) >= 2 and not sig.get("query_lube_related"):
        score -= 0.10

    if sig.get("startup_install_hits", 0) >= 2 and not sig.get("query_install_related"):
        score -= 0.12

    if sig.get("description_section_hit"):
        score -= 0.18

    if sig.get("overview_section_hit"):
        score -= 0.08

    # penalizza chunk molto parziali / istruzioni isolate
    partial_instruction_markers = [
        "invertire le fasi",
        "invert phases",
        "contattare immediatamente",
        "contact immediately",
        "prima accensione",
        "first start-up",
        "first startup",
        "alla prima accensione",
        "freccia",
        "arrow",
    ]
    partial_hits = sum(1 for m in partial_instruction_markers if m in txt)
    if partial_hits >= 1 and sig.get("strong_component_hits", 0) == 0:
        score -= 0.10
    if partial_hits >= 2:
        score -= 0.08

    # penalizza evidenze senza vero legame col processo o col sintomo
    if (
        sig.get("strong_component_hits", 0) == 0
        and sig.get("process_hits", 0) == 0
        and sig.get("symptom_hits", 0) == 0
    ):
        score -= 0.14

    score = max(-0.40, min(0.40, score))

    if score >= 0.18:
        band = "direct"
    elif score >= 0.02:
        band = "indirect"
    else:
        band = "collateral"

    return {
        "causal_strength_score": score,
        "causal_strength_band": band,
    }


@dataclass(frozen=True)
class RootCauseTargetSubsystemsRuntime:
    _normalize_unicode_advanced: Callable[..., Any]
    re: Any


def root_cause_target_subsystems(
    q: str,
    inferred_components: list[str],
    *, runtime: RootCauseTargetSubsystemsRuntime,
) -> list[str]:
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    re = runtime.re
    q_low = _normalize_unicode_advanced(q or "").lower()
    comps_low = " ".join(
        _normalize_unicode_advanced(str(x) or "").lower()
        for x in (inferred_components or [])
    )

    out: list[str] = []
    seen = set()

    def add(name: str):
        if not name or name in seen:
            return
        seen.add(name)
        out.append(name)

    if any(t in q_low for t in ["feed", "advance", "avanz", "wire", "filo", "strip", "nastro", "material"]):
        add("material_feed")

    if any(t in q_low for t in ["bend", "bending", "pieg", "forming", "formatura", "press", "pressa", "tool", "die", "stampo"]):
        add("forming")

    if any(t in q_low for t in ["straight", "straighten", "raddrizz"]):
        add("straightening")

    # Contextualize the word chain/catena.  In industrial language it can mean a
    # mechanical transmission chain or a safety/interlock chain.  A safety-chain
    # phrase must not silently become a drive-train target merely because it
    # contains the noun "catena"/"chain".
    safety_chain_context = bool(
        re.search(
            r"\b(?:catena\s+(?:di\s+)?(?:sicurezza|emergenz[ae]|consens[oi]|interblocc[oi])|"
            r"(?:safety|emergency|interlock|consent)\s+chain)\b",
            q_low,
        )
    )
    mechanical_chain_context = bool(
        re.search(
            r"\b(?:catena\s+(?:di\s+)?(?:trasmissione|cinematica|traino|rulli)|"
            r"(?:drive|transmission|conveyor|roller)\s+chain)\b",
            q_low,
        )
    )
    non_chain_drive_anchor = any(t in q_low or t in comps_low for t in [
        "motor", "motore", "bearing", "cuscinet", "shaft", "albero",
        "gear", "ingran", "gearbox", "ridutt", "transmission", "trasmission",
        "brushless", "cam", "coupling", "giunto", "belt", "cinghia",
    ])
    bare_chain_anchor = any(t in q_low or t in comps_low for t in ["chain", "catena"])
    if non_chain_drive_anchor or mechanical_chain_context or (bare_chain_anchor and not safety_chain_context):
        add("drive_train")

    safety_control_anchor = any(t in q_low for t in [
        "safety", "sicurezza", "riparo", "ripari", "guard", "guards",
        "emergenza", "emergency", "interlock", "interblocc", "microinter",
        "porta di sicurezza", "safety door", "safety chain", "catena di sicurezza",
    ])
    if safety_control_anchor:
        add("safety_installation")

    electrical_consent_anchor = any(t in q_low for t in [
        "consenso", "consensi", "consent", "enable", "abilit",
        "interlock", "interblocc", "microinter", "elettroserratura",
        "circuito di sicurezza", "safety circuit", "reset emerg", "ripristino emerg",
    ])
    if electrical_consent_anchor:
        add("electrical_control")

    if any(t in q_low for t in ["lubric", "lubrif", "oil", "olio", "grease", "grasso", "pump", "pompa"]):
        add("lubrication")

    if any(t in q_low for t in ["pneumat", "hydraul", "idraulic", "air", "aria", "pressure", "pression", "valv", "valvol", "cylind", "cilindr"]):
        add("fluid_power")

    if any(t in q_low or t in comps_low for t in ["encoder", "sensor", "sensore", "plc", "electr", "elettric", "inverter", "control", "controllo"]):
        add("electrical_control")

    if not out:
        if any(t in q_low for t in ["vibr", "oscillat", "rumor", "rumore", "noise", "rattle", "strid"]):
            add("drive_train")

    return out[:4]


@dataclass(frozen=True)
class ScoreRootCauseSubsystemAlignmentRuntime:
    _extract_section_from_text: Callable[..., Any]
    _normalize_unicode_advanced: Callable[..., Any]


def score_root_cause_subsystem_alignment(
    q: str,
    chunk_text: str,
    target_subsystems: list[str],
    *, runtime: ScoreRootCauseSubsystemAlignmentRuntime,
) -> dict:
    _extract_section_from_text = runtime._extract_section_from_text
    _normalize_unicode_advanced = runtime._normalize_unicode_advanced
    txt = _normalize_unicode_advanced(chunk_text or "").lower()
    section = _normalize_unicode_advanced(_extract_section_from_text(chunk_text) or "").lower()

    if not txt:
        return {
            "subsystem_score": -0.25,
            "matched_subsystems": [],
        }

    subsystem_markers = {
        "material_feed": [
            "feed", "advance", "avanz", "trascin", "infeed", "material",
            "wire", "filo", "strip", "nastro", "roller", "rullo", "roll"
        ],
        "forming": [
            "bend", "bending", "pieg", "forming", "formatura",
            "press", "pressa", "tool", "die", "stampo", "punch", "punzone", "matrice"
        ],
        "straightening": [
            "straighten", "straightening", "raddrizz", "flatten"
        ],
        "drive_train": [
            "motor", "motore", "brushless", "shaft", "albero", "bearing", "cuscinet",
            "gear", "ingran", "gearbox", "ridutt", "transmission", "trasmission",
            "cam", "coupling", "giunto", "belt", "cinghia", "chain", "catena"
        ],
        "lubrication": [
            "lubric", "lubrif", "oil", "olio", "grease", "grasso",
            "pump", "pompa", "pressostato", "pressure switch"
        ],
        "fluid_power": [
            "pneumat", "hydraul", "idraulic", "aria", "air",
            "pressure", "pression", "valv", "valvol", "cylind", "cilindr"
        ],
        "electrical_control": [
            "encoder", "sensor", "sensore", "plc", "electr", "elettric",
            "inverter", "control", "controllo", "drive"
        ],
        "safety_installation": [
            "safety", "sicurezza", "riparo", "guard", "door", "porta",
            "microinter", "installation", "installazione", "startup", "start-up",
            "commission", "messa in servizio", "foundation", "fondazione",
            "positioning", "posizionamento", "livellamento", "piano di appoggio"
        ],
    }

    matched_subsystems: list[str] = []
    for name, markers in subsystem_markers.items():
        if any(m in txt or m in section for m in markers):
            matched_subsystems.append(name)

    target_set = {str(x).strip() for x in (target_subsystems or []) if str(x).strip()}
    matched_set = set(matched_subsystems)
    overlap = target_set & matched_set

    score = 0.0

    if overlap:
        score += min(0.34, 0.18 * len(overlap))

        if any(ts in section for ts in [
            "feed", "advance", "avanz", "bend", "pieg",
            "straight", "raddrizz", "trasmission", "transmission"
        ]):
            score += 0.06

    if matched_set and not overlap:
        support_only = matched_set <= {"lubrication", "fluid_power", "electrical_control", "safety_installation"}
        if support_only:
            score -= 0.16
        else:
            score -= 0.08

    if matched_set == {"lubrication"} and "lubrication" not in target_set:
        score -= 0.08

    if "safety_installation" in matched_set and not overlap:
        score -= 0.10

    score = max(-0.30, min(0.40, score))

    return {
        "subsystem_score": score,
        "matched_subsystems": matched_subsystems[:4],
    }


@dataclass(frozen=True)
class ScoreRootCauseContextFitRuntime:
    ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY: Any
    _root_cause_chunk_signal_summary: Callable[..., Any]


def score_root_cause_context_fit(
    q: str,
    chunk_text: str,
    diagnostic_keywords: list[str],
    symptom_profile: dict,
    matched_subsystems: list[str],
    *, runtime: ScoreRootCauseContextFitRuntime,
) -> dict:
    ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY = runtime.ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )

    classes = set(symptom_profile.get("classes") or [])
    matched_set = {str(x).strip() for x in (matched_subsystems or []) if str(x).strip()}
    direct_subsystems = {"drive_train", "material_feed", "forming", "straightening"}
    support_subsystems = {"lubrication", "fluid_power", "electrical_control", "safety_installation"}

    support_only = bool(matched_set) and not (matched_set & direct_subsystems) and (matched_set <= support_subsystems)
    direct_mechanism_supported = bool(matched_set & direct_subsystems)

    has_support_anchor = bool(symptom_profile.get("has_support_anchor"))
    automatic_mode = bool(symptom_profile.get("automatic_mode"))
    generic_symptom = bool(symptom_profile.get("generic_symptom"))

    score = 0.0

    if classes & {"vibration", "noise"}:
        if direct_mechanism_supported and (
            int(sig.get("strong_component_hits", 0) or 0) >= 1
            or int(sig.get("process_hits", 0) or 0) >= 1
            or int(sig.get("symptom_hits", 0) or 0) >= 1
        ):
            score += 0.14

        if support_only and not has_support_anchor:
            score -= ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY

        if int(sig.get("lube_control_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.12
        if int(sig.get("startup_install_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.12
        if int(sig.get("safety_access_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.10

    if "jam" in classes:
        if (matched_set & {"material_feed", "straightening", "forming", "drive_train"}) and (
            int(sig.get("process_hits", 0) or 0) >= 1
            or int(sig.get("strong_component_hits", 0) or 0) >= 1
        ):
            score += 0.14

        if support_only and not has_support_anchor:
            score -= ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY

        if int(sig.get("lube_control_hits", 0) or 0) >= 2 and not has_support_anchor:
            score -= 0.10

    if "no_start" in classes:
        if "electrical_control" in matched_set:
            score += 0.14
        if "safety_installation" in matched_set:
            score += 0.10
        if automatic_mode and ({"electrical_control", "safety_installation"} & matched_set):
            score += 0.06
        if matched_set == {"lubrication"} and not has_support_anchor:
            score -= 0.16
        if int(sig.get("startup_install_hits", 0) or 0) >= 2 and not ({"electrical_control", "safety_installation"} & matched_set):
            score -= 0.08

    if generic_symptom and support_only:
        score -= 0.08

    if generic_symptom and direct_mechanism_supported:
        score += 0.08

    score = max(-0.45, min(0.30, score))
    return {
        "context_fit_score": score,
        "support_only_penalized": bool(score < 0 and support_only and (classes & {"vibration", "noise", "jam"})),
        "direct_mechanism_supported": direct_mechanism_supported,
    }


