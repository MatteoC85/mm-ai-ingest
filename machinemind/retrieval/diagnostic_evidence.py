"""P4-C9: extracted diagnostic evidence implementations.

Existing prompts, heuristics, scores, limits and fallback paths are kept.
No database, provider or main imports: dependencies are injected per call.
This is structural extraction, not a semantic change or quality guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class LlmFilterDiagnosticChunksRuntime:
    DIAGNOSTIC_EVIDENCE_MODEL: Any
    OPENAI_CHAT_MODEL: Any
    OPENAI_RERANK_MODEL: Any
    RERANK_TIMEOUT: Any
    _extract_section_from_text: Callable[..., Any]
    _openai_chat_json_models: Callable[..., Any]
    _root_cause_evidence_family_key: Callable[..., Any]
    json: Any


def llm_filter_diagnostic_chunks(
    q: str,
    candidates: list[dict],
    max_keep: int,
    *, runtime: LlmFilterDiagnosticChunksRuntime,
) -> list[str]:
    DIAGNOSTIC_EVIDENCE_MODEL = runtime.DIAGNOSTIC_EVIDENCE_MODEL
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    RERANK_TIMEOUT = runtime.RERANK_TIMEOUT
    _extract_section_from_text = runtime._extract_section_from_text
    _openai_chat_json_models = runtime._openai_chat_json_models
    _root_cause_evidence_family_key = runtime._root_cause_evidence_family_key
    json = runtime.json
    if not q or not candidates:
        return []

    items = []

    for c in candidates[:18]:
        cid = str(c.get("citation_id") or "").strip()
        snippet = (c.get("snippet") or "").strip()
        section = _extract_section_from_text(c.get("chunk_full") or c.get("snippet") or "")

        items.append({
            "citation_id": cid,
            "section": section[:120],
            "evidence_family": _root_cause_evidence_family_key(c),
            "matched_subsystems": c.get("matched_subsystems") or [],
            "subsystem_score": round(float(c.get("subsystem_score", 0.0)), 4),
            "causal_strength_score": round(float(c.get("causal_strength_score", 0.0)), 4),
            "semantic_score": round(float(c.get("semantic_score", 0.0)), 4),
            "generic_downranked": bool(c.get("generic_downranked")),
            "snippet": snippet[:300]
        })

    schema = {
        "name": "diagnostic_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["selected_ids"]
        }
    }

    system_msg = (
        "Selezioni solo le fonti realmente utili per diagnosticare un problema tecnico su una macchina industriale.\n"
        "Regole:\n"
        "1) Tieni solo fonti che parlano del fenomeno o dei componenti coinvolti.\n"
        "2) Scarta fonti generiche di manutenzione, sicurezza, installazione o lubrificazione se non sono direttamente legate al sintomo.\n"
        "3) Se una fonte parla solo di controlli generici o procedure standard, scartala.\n"
        "4) Mantieni poche fonti ma molto pertinenti.\n"
        "5) Non collassare tutto su una sola fonte se esistono 2-3 aree causali diverse ben supportate.\n"
        "6) Le fonti con generic_downranked=true sono bassa priorità e vanno tenute solo se il sintomo coincide in modo diretto.\n"
        "7) Preferisci sezioni operative o di componente rispetto a overview, safety, installation, start-up o caratteristiche generali.\n"
        "8) Evita di selezionare più citation_id della stessa evidence_family se una sola fonte rappresenta già bene quell'area.\n"
        "9) Seleziona fonti che coprono aree causali diverse quando sono ben supportate.\n"
        "10) Preferisci fonti allineate ai sottosistemi dominanti implicati dal sintomo.\n"
        "11) Se esistono fonti di sottosistemi secondari, mantienile solo se spiegano una causa davvero plausibile e non indiretta.\n"
        "12) Per sintomi generici come vibrazione, rumore o blocco, non privilegiare lubrificazione, start-up, installazione o sicurezza se il testo non collega esplicitamente quel sottosistema al sintomo.\n"
        "13) Per il mancato avvio, i blocchi elettrici, interlock e consensi sono più forti di una nota generica di lubrificazione.\n"
        "14) Favorisci le evidenze con causal_strength_score e semantic_score più alti.\n"
    )

    user_msg = (
        f"PROBLEMA:\n{q}\n\n"
        f"CANDIDATI:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        f"Restituisci JSON con gli id delle fonti più utili alla diagnosi."
    )

    parsed = _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    selected = parsed.get("selected_ids") or []

    out = []
    used = set()

    for cid in selected:
        cid = str(cid).strip()
        if cid and cid not in used:
            used.add(cid)
            out.append(cid)
        if len(out) >= max_keep:
            break

    return out


@dataclass(frozen=True)
class LlmBuildDiagnosticEvidenceMatrixRuntime:
    DIAGNOSTIC_EVIDENCE_MODEL: Any
    OPENAI_CHAT_MODEL: Any
    OPENAI_RERANK_MODEL: Any
    RERANK_TIMEOUT: Any
    _extract_section_from_text: Callable[..., Any]
    _openai_chat_json_models: Callable[..., Any]
    _root_cause_evidence_family_key: Callable[..., Any]
    json: Any
    re: Any


def llm_build_diagnostic_evidence_matrix(
    q: str,
    citations: list[dict],
    max_causes: int,
    *, runtime: LlmBuildDiagnosticEvidenceMatrixRuntime,
) -> dict:
    DIAGNOSTIC_EVIDENCE_MODEL = runtime.DIAGNOSTIC_EVIDENCE_MODEL
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    RERANK_TIMEOUT = runtime.RERANK_TIMEOUT
    _extract_section_from_text = runtime._extract_section_from_text
    _openai_chat_json_models = runtime._openai_chat_json_models
    _root_cause_evidence_family_key = runtime._root_cause_evidence_family_key
    json = runtime.json
    re = runtime.re
    if not q or not citations:
        return {}

    max_causes = max(1, min(int(max_causes or 1), 3))

    items = []
    seen = set()

    for c in citations[:10]:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)

        snippet = (c.get("chunk_full") or c.get("snippet") or "").strip()
        snippet = re.sub(r"^SECTION:\s*[^\n]+\n?", "", snippet).strip()

        section = _extract_section_from_text(c.get("chunk_full") or c.get("snippet") or "")

        items.append(
            {
                "citation_id": cid,
                "section": section[:120],
                "evidence_family": _root_cause_evidence_family_key(c),
                "matched_subsystems": c.get("matched_subsystems") or [],
                "subsystem_score": round(float(c.get("subsystem_score", 0.0)), 4),
                "causal_strength_score": round(float(c.get("causal_strength_score", 0.0)), 4),
                "semantic_score": round(float(c.get("semantic_score", 0.0)), 4),
                "page_from": int(c.get("page_from") or 0),
                "page_to": int(c.get("page_to") or 0),
                "generic_downranked": bool(c.get("generic_downranked")),
                "snippet": snippet[:360],
            }
        )

    if not items:
        return {}

    schema = {
        "name": "diagnostic_evidence_matrix",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "discard_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cause_hypotheses": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cause": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "check_focus": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["cause", "evidence_ids", "check_focus"],
                    },
                },
            },
            "required": ["keep_ids", "discard_ids", "cause_hypotheses"],
        },
    }

    system_msg = (
        "Selezioni e organizzi le evidenze per una root cause analysis industriale.\n"
        "Obiettivo: tenere solo le fonti davvero utili e raggrupparle per area causale.\n"
        "Regole obbligatorie:\n"
        "1) keep_ids = solo citazioni utili alla diagnosi.\n"
        "2) discard_ids = citazioni generiche, ripetitive, di solo contesto o sicurezza.\n"
        "3) cause_hypotheses = massimo poche ipotesi distinte; non duplicare varianti della stessa causa.\n"
        "4) Ogni ipotesi deve usare solo citation_id presenti nei candidati.\n"
        "5) check_focus = verifiche pratiche brevi, non frasi lunghe.\n"
        "6) Non collassare tutto su una sola causa se le citazioni supportano aree causali diverse.\n"
        "7) keep_ids deve mantenere copertura delle aree causali utili, non solo il numero minimo di fonti.\n"
        "8) Le fonti con generic_downranked=true sono bassa priorità e non vanno usate come evidenza centrale se esistono fonti più specifiche.\n"
        "9) Preferisci sezioni operative o di componente rispetto a overview, safety, installation, start-up o caratteristiche generali.\n"
        "10) Evita di mantenere più citation_id della stessa evidence_family se una sola fonte è già rappresentativa.\n"
        "11) keep_ids e cause_hypotheses devono massimizzare la copertura di aree causali diverse, non la ripetizione della stessa area.\n"
        "12) Preferisci ipotesi coerenti con i sottosistemi dominanti implicati dal sintomo.\n"
        "13) Per sintomi generici come vibrazione, rumore o blocco, le fonti di lubrificazione, start-up, installazione o sicurezza non devono diventare ipotesi centrali senza un legame esplicito col sintomo.\n"
        "14) Per il mancato avvio, preferisci cause elettriche/interlock/consensi rispetto a note generiche di lubrificazione.\n"
        "15) Le evidenze con causal_strength_score e semantic_score più alti hanno priorità.\n"
    )

    user_msg = (
        f"SINTOMO/PROBLEMA:\n{q}\n\n"
        "CITAZIONI_CANDIDATE_JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "Restituisci JSON valido."
    )

    parsed = _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
        json_schema=schema,
        timeout=RERANK_TIMEOUT,
    )

    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class ClassifyDiagnosticRoleFromTextRuntime:
    _root_cause_chunk_signal_summary: Callable[..., Any]
    _score_root_cause_subsystem_alignment: Callable[..., Any]


def classify_diagnostic_role_from_text(
    q: str,
    chunk_text: str,
    symptom_profile: dict,
    diagnostic_keywords: list[str],
    target_subsystems: list[str],
    *, runtime: ClassifyDiagnosticRoleFromTextRuntime,
) -> dict:
    _root_cause_chunk_signal_summary = runtime._root_cause_chunk_signal_summary
    _score_root_cause_subsystem_alignment = runtime._score_root_cause_subsystem_alignment
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return {
            "role_class": "collateral",
            "role_group": "collateral",
            "role_adjustment": -0.06,
            "matched_subsystems": [],
            "role_reason": "empty_chunk",
        }

    sig = _root_cause_chunk_signal_summary(
        q=q,
        chunk_text=chunk_text,
        diagnostic_keywords=diagnostic_keywords,
    )
    subsystem = _score_root_cause_subsystem_alignment(
        q=q,
        chunk_text=chunk_text,
        target_subsystems=target_subsystems,
    )

    classes = set(symptom_profile.get("classes") or [])
    matched_set = {str(x).strip() for x in (subsystem.get("matched_subsystems") or []) if str(x).strip()}
    direct_subsystems = {"drive_train", "material_feed", "forming", "straightening"}
    support_subsystems = {"lubrication", "fluid_power", "electrical_control", "safety_installation"}
    has_support_anchor = bool(symptom_profile.get("has_support_anchor"))
    automatic_mode = bool(symptom_profile.get("automatic_mode"))

    strong_component_hits = int(sig.get("strong_component_hits", 0) or 0)
    process_hits = int(sig.get("process_hits", 0) or 0)
    symptom_hits = int(sig.get("symptom_hits", 0) or 0)
    lube_hits = int(sig.get("lube_control_hits", 0) or 0)
    startup_hits = int(sig.get("startup_install_hits", 0) or 0) + int(sig.get("positioning_hits", 0) or 0)
    safety_hits = int(sig.get("safety_access_hits", 0) or 0) + int(sig.get("acoustic_protection_hits", 0) or 0)

    direct_mechanism_supported = bool(matched_set & direct_subsystems) and (
        strong_component_hits >= 1 or process_hits >= 1 or symptom_hits >= 1 or float(subsystem.get("subsystem_score", 0.0) or 0.0) > 0.0
    )

    if "no_start" in classes:
        if "electrical_control" in matched_set:
            return {
                "role_class": "support_electrical_interlock",
                "role_group": "support",
                "role_adjustment": 0.16 if automatic_mode else 0.12,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_electrical_control",
            }
        if "safety_installation" in matched_set:
            return {
                "role_class": "support_safety",
                "role_group": "support",
                "role_adjustment": 0.10 if automatic_mode else 0.04,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_safety_interlock",
            }
        if "lubrication" in matched_set or lube_hits >= 2:
            return {
                "role_class": "support_lubrication",
                "role_group": "support",
                "role_adjustment": -0.18 if not has_support_anchor else -0.05,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "no_start_lubrication_secondary",
            }

    if direct_mechanism_supported:
        if (matched_set & {"forming", "material_feed", "straightening"}) or process_hits >= 1:
            return {
                "role_class": "core_process",
                "role_group": "core",
                "role_adjustment": 0.14,
                "matched_subsystems": sorted(matched_set),
                "role_reason": "direct_process_mechanism",
            }
        return {
            "role_class": "core_mechanical",
            "role_group": "core",
            "role_adjustment": 0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "direct_mechanical_mechanism",
        }

    if matched_set & {"fluid_power"}:
        return {
            "role_class": "support_fluid_power",
            "role_group": "support",
            "role_adjustment": 0.06 if ("jam" in classes or has_support_anchor) else -0.04,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "fluid_power_support",
        }

    if matched_set & {"electrical_control"}:
        return {
            "role_class": "support_electrical_interlock",
            "role_group": "support",
            "role_adjustment": 0.08 if ("no_start" in classes or has_support_anchor) else -0.03,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "electrical_or_control_support",
        }

    if matched_set & {"lubrication"} or lube_hits >= 2:
        return {
            "role_class": "support_lubrication",
            "role_group": "support",
            "role_adjustment": 0.04 if has_support_anchor else -0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "lubrication_support",
        }

    if startup_hits >= 2 or bool(sig.get("overview_section_hit")) or bool(sig.get("description_section_hit")):
        return {
            "role_class": "support_startup_install",
            "role_group": "support",
            "role_adjustment": -0.16 if not has_support_anchor else -0.03,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "startup_install_or_overview",
        }

    if safety_hits >= 2:
        return {
            "role_class": "support_safety",
            "role_group": "support",
            "role_adjustment": 0.03 if ("no_start" in classes and automatic_mode) else -0.12,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "safety_support",
        }

    if strong_component_hits >= 1 or process_hits >= 1:
        return {
            "role_class": "core_mechanical",
            "role_group": "core",
            "role_adjustment": 0.06,
            "matched_subsystems": sorted(matched_set),
            "role_reason": "component_or_process_anchor_without_subsystem",
        }

    return {
        "role_class": "collateral",
        "role_group": "collateral",
        "role_adjustment": -0.08,
        "matched_subsystems": sorted(matched_set),
        "role_reason": "collateral_or_weak",
    }


@dataclass(frozen=True)
class SummarizeEvidenceRolesForPromptRuntime:
    _classify_diagnostic_role_from_text: Callable[..., Any]
    _collect_candidate_keywords: Callable[..., Any]
    _infer_machine_components: Callable[..., Any]
    _query_symptom_profile: Callable[..., Any]
    _root_cause_target_subsystems: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    re: Any


def summarize_evidence_roles_for_prompt(
    q: str,
    citations: list[dict],
    *,
    symptom_profile: Optional[dict] = None,
    diagnostic_keywords: Optional[list[str]] = None,
    target_subsystems: Optional[list[str]] = None,
    max_items: int = 8,
    runtime: SummarizeEvidenceRolesForPromptRuntime,
) -> list[dict]:
    _classify_diagnostic_role_from_text = runtime._classify_diagnostic_role_from_text
    _collect_candidate_keywords = runtime._collect_candidate_keywords
    _infer_machine_components = runtime._infer_machine_components
    _query_symptom_profile = runtime._query_symptom_profile
    _root_cause_target_subsystems = runtime._root_cause_target_subsystems
    _source_type_from_document_id = runtime._source_type_from_document_id
    re = runtime.re
    symptom_profile = dict(symptom_profile or _query_symptom_profile(q))
    inferred_components = _infer_machine_components(q)
    diagnostic_keywords = list(diagnostic_keywords or _collect_candidate_keywords(q, inferred_components))
    target_subsystems = list(target_subsystems or _root_cause_target_subsystems(q, inferred_components))

    out = []
    used = set()
    for c in citations or []:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        chunk_text = (c.get("chunk_full") or c.get("snippet") or "").strip()
        role = _classify_diagnostic_role_from_text(
            q=q,
            chunk_text=chunk_text,
            symptom_profile=symptom_profile,
            diagnostic_keywords=diagnostic_keywords,
            target_subsystems=target_subsystems,
        )
        out.append(
            {
                "citation_id": cid,
                "role_class": str(c.get("role_class") or role.get("role_class") or "collateral"),
                "role_group": str(c.get("role_group") or role.get("role_group") or "collateral"),
                "role_adjustment": round(float(c.get("role_adjustment", role.get("role_adjustment", 0.0)) or 0.0), 4),
                "matched_subsystems": list(c.get("matched_subsystems") or role.get("matched_subsystems") or []),
                "source_type": str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")),
                "diagnostic_score": round(float(c.get("candidate_score", c.get("diagnostic_score", c.get("retrieval_score", c.get("similarity", 0.0)))) or 0.0), 4),
                "snippet": re.sub(r"\s+", " ", (c.get("snippet") or chunk_text or "").strip())[:260],
            }
        )
        if len(out) >= max_items:
            break
    return out


@dataclass(frozen=True)
class LlmBuildRoleAwareDiagnosticEvidenceMatrixRuntime:
    DIAGNOSTIC_EVIDENCE_MODEL: Any
    OPENAI_CHAT_MODEL: Any
    ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K: Any
    ROOT_CAUSE_RESPONSE_MODEL: Any
    _openai_chat_json_models: Callable[..., Any]
    _root_cause_evidence_family_key: Callable[..., Any]
    _source_type_from_document_id: Callable[..., Any]
    json: Any
    re: Any


def llm_build_role_aware_diagnostic_evidence_matrix(
    q: str,
    citations: list[dict],
    max_causes: int,
    *, runtime: LlmBuildRoleAwareDiagnosticEvidenceMatrixRuntime,
) -> dict:
    DIAGNOSTIC_EVIDENCE_MODEL = runtime.DIAGNOSTIC_EVIDENCE_MODEL
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K = runtime.ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K
    ROOT_CAUSE_RESPONSE_MODEL = runtime.ROOT_CAUSE_RESPONSE_MODEL
    _openai_chat_json_models = runtime._openai_chat_json_models
    _root_cause_evidence_family_key = runtime._root_cause_evidence_family_key
    _source_type_from_document_id = runtime._source_type_from_document_id
    json = runtime.json
    re = runtime.re
    if not q or not citations:
        return {}

    max_causes = max(1, min(int(max_causes or 1), 3))
    items = []
    used = set()
    for c in citations[: max(ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K, max_causes + 5)]:
        cid = str(c.get("citation_id") or "").strip()
        if not cid or cid in used:
            continue
        used.add(cid)
        items.append(
            {
                "citation_id": cid,
                "role_class": str(c.get("role_class") or "collateral"),
                "role_group": str(c.get("role_group") or "collateral"),
                "role_adjustment": round(float(c.get("role_adjustment", 0.0) or 0.0), 4),
                "matched_subsystems": list(c.get("matched_subsystems") or []),
                "evidence_family": _root_cause_evidence_family_key(c),
                "diagnostic_score": round(float(c.get("candidate_score", c.get("diagnostic_score", c.get("retrieval_score", c.get("similarity", 0.0)))) or 0.0), 4),
                "source_type": str(c.get("source_type") or _source_type_from_document_id(c.get("bubble_document_id") or "")),
                "snippet": re.sub(r"\s+", " ", (c.get("chunk_full") or c.get("snippet") or "").strip())[:360],
            }
        )

    if not items:
        return {}

    schema = {
        "name": "role_aware_diagnostic_evidence_matrix",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep_ids": {"type": "array", "items": {"type": "string"}},
                "discard_ids": {"type": "array", "items": {"type": "string"}},
                "cause_hypotheses": {
                    "type": "array",
                    "maxItems": max_causes,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "cause": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "check_focus": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["cause", "evidence_ids", "check_focus"],
                    },
                },
            },
            "required": ["keep_ids", "discard_ids", "cause_hypotheses"],
        },
    }

    system_msg = (
        "You organize evidence for industrial root-cause diagnosis. "
        "Use the role metadata strictly. core roles are preferred for generic symptoms such as vibration, noise, or jams. "
        "support roles (lubrication, startup/install, safety, electrical/interlock, fluid power) may stay only when the symptom explicitly anchors them or when no stronger core evidence is available. "
        "For no-start and automatic-mode failures, support_electrical_interlock can be primary, but support_lubrication should remain secondary unless directly anchored. "
        "Maximize distinct causal families and avoid duplicate paraphrases. Keep only the most diagnostic evidence."
    )
    user_msg = (
        f"PROBLEM:\n{q}\n\n"
        f"ROLE_AWARE_CANDIDATES_JSON:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Return valid JSON."
    )

    return _openai_chat_json_models(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        models=[DIAGNOSTIC_EVIDENCE_MODEL, ROOT_CAUSE_RESPONSE_MODEL, OPENAI_CHAT_MODEL],
        json_schema=schema,
        timeout=70,
    )


@dataclass(frozen=True)
class AskStructuredManualSupportSelectorSchemaRuntime:
    pass


def ask_structured_manual_support_selector_schema( *, runtime: AskStructuredManualSupportSelectorSchemaRuntime) -> dict:
    return {
        "name": "ask_structured_manual_support_selector_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation_support_indices": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
                "safety_support_indices": {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
                "operation_note": {"type": "string"},
                "safety_note": {"type": "string"},
                "rejected_reason": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": [
                "operation_support_indices",
                "safety_support_indices",
                "operation_note",
                "safety_note",
                "rejected_reason",
                "reason",
            ],
        },
    }


@dataclass(frozen=True)
class AskStructuredManualSupportSelectWithLlmRuntime:
    ASK_EVIDENCE_ANALYZER_MODEL: Any
    ASK_STRUCTURED_DIRECT_MODEL: Any
    ASK_STRUCTURED_DIRECT_TIMEOUT: Any
    OPENAI_API_KEY: Any
    OPENAI_CHAT_MODEL: Any
    OPENAI_RERANK_MODEL: Any
    _ask_full_context_sources_block: Callable[..., Any]
    _ask_structured_manual_support_selector_schema: Callable[..., Any]
    _clean_display_text: Callable[..., Any]
    _openai_chat_json_models: Callable[..., Any]
    re: Any


def ask_structured_manual_support_select_with_llm(
    *,
    q: str,
    response_language: str,
    structured_citations: list[dict],
    candidates: list[dict],
    runtime: AskStructuredManualSupportSelectWithLlmRuntime,
) -> dict:
    """Reason about whether manual pages directly support a structured operation.

    This selector is intentionally semantic, not keyword/dictionary based. It receives
    the user's question, the primary structured procedure/step/P&S/photo/video records,
    and candidate manual pages. It must select manual pages only when they directly add
    information for the same requested operation or a directly applicable safety
    prerequisite. Generic safety, generic maintenance, adjacent processes, or pages that
    merely share broad machine vocabulary must be rejected.
    """
    ASK_EVIDENCE_ANALYZER_MODEL = runtime.ASK_EVIDENCE_ANALYZER_MODEL
    ASK_STRUCTURED_DIRECT_MODEL = runtime.ASK_STRUCTURED_DIRECT_MODEL
    ASK_STRUCTURED_DIRECT_TIMEOUT = runtime.ASK_STRUCTURED_DIRECT_TIMEOUT
    OPENAI_API_KEY = runtime.OPENAI_API_KEY
    OPENAI_CHAT_MODEL = runtime.OPENAI_CHAT_MODEL
    OPENAI_RERANK_MODEL = runtime.OPENAI_RERANK_MODEL
    _ask_full_context_sources_block = runtime._ask_full_context_sources_block
    _ask_structured_manual_support_selector_schema = runtime._ask_structured_manual_support_selector_schema
    _clean_display_text = runtime._clean_display_text
    _openai_chat_json_models = runtime._openai_chat_json_models
    re = runtime.re
    if not OPENAI_API_KEY or not candidates:
        return {
            "operation_support_indices": [],
            "safety_support_indices": [],
            "operation_note": "",
            "safety_note": "",
            "reason": "selector disabled or no candidates",
            "rejected_reason": "",
        }

    structured_block = _ask_full_context_sources_block(
        structured_citations,
        max_context_chars=9000,
    )
    cand_parts: list[str] = []
    for c in candidates:
        idx = int(c.get("selector_index") or 0)
        label = str(c.get("display_label") or c.get("citation_id") or "Manual page").strip()
        page_text = str(c.get("chunk_full") or c.get("snippet") or "")
        page_text = re.sub(r"\s+", " ", page_text).strip()
        page_text = _clean_display_text(page_text, max_len=1800)
        if idx and page_text:
            cand_parts.append(f"[PAGE_INDEX {idx}] {label}\n{page_text}")
    candidates_block = "\n\n---\n\n".join(cand_parts)
    if not candidates_block:
        return {
            "operation_support_indices": [],
            "safety_support_indices": [],
            "operation_note": "",
            "safety_note": "",
            "reason": "no readable candidates",
            "rejected_reason": "",
        }

    system_msg = (
        "You are a strict evidence selector for an industrial AI assistant. "
        "Use semantic reasoning, not keyword matching. The structured sources are the primary source. "
        "Manual pages are optional secondary support. Select a manual page ONLY if it directly helps answer the user's exact operation/problem, "
        "or if it describes an immediate prerequisite/continuation that a technician must perform around the structured operation. "
        "If a manual page does not use the same workshop wording as the structured procedure but explains the formal manual operation that follows or supports it, it may be selected as related manual support. "
        "Reject pages that are merely generic safety, generic maintenance, setup overview, unrelated adjustment, or broadly similar machine vocabulary. "
        "Safety pages may be selected only when the safety instruction is directly applicable to the operation, not as a generic disclaimer. "
        "If unsure, select nothing. Do not infer from outside knowledge."
    )
    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"RESPONSE_LANGUAGE:\n{response_language}\n\n"
        f"PRIMARY STRUCTURED SOURCES:\n{structured_block}\n\n"
        f"CANDIDATE MANUAL PAGES:\n{candidates_block}\n\n"
        "Return JSON only. operation_support_indices are manual PAGE_INDEX values that either directly add operational instructions for the exact requested operation "
        "or explain an immediate connected manual phase/prerequisite needed around the structured operation. "
        "safety_support_indices are manual PAGE_INDEX values that provide directly applicable prerequisites/safety for that exact operation. "
        "operation_note and safety_note must be short, user-facing, and based only on selected pages. "
        "When the manual page is related support rather than the exact internal procedure, say that clearly. If no selected page exists for a note, leave it empty."
    )
    try:
        parsed = _openai_chat_json_models(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            models=[ASK_STRUCTURED_DIRECT_MODEL, ASK_EVIDENCE_ANALYZER_MODEL, OPENAI_RERANK_MODEL, OPENAI_CHAT_MODEL],
            json_schema=_ask_structured_manual_support_selector_schema(),
            timeout=min(int(ASK_STRUCTURED_DIRECT_TIMEOUT or 60), 55),
        )
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print("ASK_STRUCTURED_MANUAL_SELECTOR_FAIL", str(e)[:700])

    return {
        "operation_support_indices": [],
        "safety_support_indices": [],
        "operation_note": "",
        "safety_note": "",
        "reason": "selector failed closed",
        "rejected_reason": "selector failed",
    }


@dataclass(frozen=True)
class AskFullContextSourcesBlockRuntime:
    pass


def ask_full_context_sources_block(citations: list[dict], *, max_context_chars: int, runtime: AskFullContextSourcesBlockRuntime) -> str:
    parts: list[str] = []
    total = 0
    for c in citations or []:
        body = str(c.get("chunk_full") or c.get("snippet") or "").strip()
        if not body:
            continue
        part = (
            f"[{c['citation_id']}] "
            f"(doc={c['bubble_document_id']}, p{c['page_from']}-{c['page_to']})\n"
            f"{body}\n"
        )
        if total + len(part) > max_context_chars:
            if not parts:
                part = part[:max_context_chars]
                parts.append(part)
            break
        parts.append(part)
        total += len(part)
    return "\n".join(parts).strip()


