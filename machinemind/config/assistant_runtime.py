"""Assistant Core V2 and V13 runtime policy configuration.

The assignments below are mechanically extracted from the certified production
monolith. Defaults, clamps, markers, release identifiers and engine-key generation
are intentionally unchanged.
"""
from __future__ import annotations

import hashlib
import os

__all__ = [
    "V13_ENABLED",
    "V13_ASK_ENABLED",
    "V13_ROOT_CAUSE_ENABLED",
    "V13_CODE_MARKER",
    "V13_RELEASE_ID",
    "OPENAI_RESPONSES_URL",
    "V13_PLANNER_MODEL",
    "V13_FAST_MODEL",
    "V13_HEAVY_MODEL",
    "V13_EVIDENCE_GATE_MODEL",
    "V13_EVIDENCE_GATE_EFFORT",
    "V13_FAST_EFFORT",
    "V13_ASK_HEAVY_EFFORT",
    "V13_ROOT_HEAVY_EFFORT",
    "V13_HEAVY_REASONING_MODE",
    "ASSISTANT_CORE_V2_ENABLED",
    "ASSISTANT_CORE_V2_CODE_MARKER",
    "ASSISTANT_CORE_V2_RELEASE_ID",
    "RESULT_INCOMPLETE_ANSWER_CONTRACT",
    "ASSISTANT_UI_RENDER_VERSION",
    "ASSISTANT_ASK_UI_RENDER_VERSION",
    "ASSISTANT_UI_MAX_HTML_CHARS",
    "ASSISTANT_CORE_ROUTER_MODEL",
    "ASSISTANT_CORE_ROUTER_FALLBACK_MODEL",
    "ASSISTANT_CORE_ROUTER_EFFORT",
    "ASSISTANT_CORE_SMART_MODEL",
    "ASSISTANT_CORE_SMART_EFFORT",
    "ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS",
    "ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS",
    "ASSISTANT_CORE_ROUTER_MAX_CONTEXT_CHARS",
    "ASSISTANT_CORE_MAX_FACETS",
    "ASSISTANT_CORE_MAX_FACET_DENSE_QUERIES",
    "ASSISTANT_CORE_MAX_FACET_LEXICAL_QUERIES",
    "ASSISTANT_CORE_FACET_DENSE_CANDIDATE_K",
    "ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD",
    "ASSISTANT_CORE_REPAIR_MAX_OUTPUT_TOKENS",
    "ASSISTANT_CORE_GENERAL_KNOWLEDGE_ENABLED",
    "ASSISTANT_CORE_ASK_DEADLINE_SECONDS",
    "ASSISTANT_CORE_ROOT_CAUSE_DEADLINE_SECONDS",
    "ASSISTANT_CORE_SMART_START_DEADLINE_SECONDS",
    "ASSISTANT_CORE_SMART_TURN_DEADLINE_SECONDS",
    "ASSISTANT_CORE_HARD_TIMEOUT_SECONDS",
    "ASSISTANT_CORE_MAX_LLM_CALLS_ASK",
    "ASSISTANT_CORE_MAX_LLM_CALLS_ROOT_CAUSE",
    "ASSISTANT_CORE_MAX_LLM_CALLS_SMART_START",
    "ASSISTANT_CORE_MAX_LLM_CALLS_SMART_TURN",
    "ASSISTANT_CORE_MAX_COST_ASK_USD",
    "ASSISTANT_CORE_MAX_COST_ROOT_CAUSE_USD",
    "ASSISTANT_CORE_MAX_COST_SMART_START_USD",
    "ASSISTANT_CORE_MAX_COST_SMART_TURN_USD",
    "ASSISTANT_CORE_GENERAL_MAX_OUTPUT_TOKENS",
    "ASSISTANT_CORE_SMART_MAX_OUTPUT_TOKENS",
    "V13_ASK_DEADLINE_SECONDS",
    "V13_ROOT_CAUSE_DEADLINE_SECONDS",
    "V13_MAX_LLM_CALLS_ASK",
    "V13_MAX_LLM_CALLS_ROOT_CAUSE",
    "V13_MAX_ESTIMATED_COST_ASK_USD",
    "V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD",
    "V13_PLANNER_TIMEOUT_SECONDS",
    "V13_EVIDENCE_GATE_TIMEOUT_SECONDS",
    "V13_EVIDENCE_GATE_MAX_OUTPUT_TOKENS",
    "V13_EVIDENCE_GATE_MAX_CANDIDATES",
    "V13_EVIDENCE_GATE_MIN_CONFIDENCE",
    "V13_EVIDENCE_CLEAR_SUPPORT_SIM",
    "V13_EVIDENCE_SUPPORT_SIM_WITH_OVERLAP",
    "V13_EVIDENCE_CLEAR_REJECT_SIM",
    "V13_EVIDENCE_MIN_OVERLAP",
    "V13_RETRIEVAL_ASSURANCE_ENABLED",
    "V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ASK",
    "V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ROOT_CAUSE",
    "V13_RETRIEVAL_ASSURANCE_PRE_GATE_MAX_SECONDS",
    "V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ASK",
    "V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ROOT_CAUSE",
    "V13_RETRIEVAL_ASSURANCE_MAX_DENSE_QUERIES",
    "V13_RETRIEVAL_ASSURANCE_MAX_LEXICAL_QUERIES",
    "V13_RETRIEVAL_ASSURANCE_MAX_DOCS",
    "V13_RETRIEVAL_ASSURANCE_PAGE_RADIUS",
    "V13_RETRIEVAL_ASSURANCE_MAX_NEIGHBOR_PAGES",
    "V13_RETRIEVAL_ASSURANCE_MAX_CANDIDATES",
    "V13_RETRIEVAL_ASSURANCE_MAX_FACETS",
    "V13_RETRIEVAL_ASSURANCE_MIN_FACET_GAIN",
    "V13_RETRIEVAL_ASSURANCE_MIN_COVERAGE_GAIN",
    "V13_RETRIEVAL_ASSURANCE_MIN_SUPPORT_GAIN",
    "V13_RETRIEVAL_ASSURANCE_MIN_NEW_SEMANTIC_SIM",
    "V13_RETRIEVAL_ASSURANCE_MIN_NEW_OVERLAP",
    "V13_SOURCE_RETRIEVAL_ENABLED",
    "V13_SOURCE_RETRIEVAL_SCAN_LIMIT",
    "V13_SOURCE_RETRIEVAL_MAX_CANDIDATES",
    "V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE",
    "V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE",
    "V13_SOURCE_RETRIEVAL_MIN_TASK_CONFIDENCE",
    "V13_SOURCE_RETRIEVAL_REQUIRE_TYPE_CONFIDENCE",
    "V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW",
    "V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY",
    "V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS",
    "V13_SOURCE_RETRIEVAL_MIN_SEMANTIC_SCORE",
    "V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE",
    "V13_SOURCE_RETRIEVAL_PREFERENCE_MAX_GAP",
    "V13_SOURCE_RETRIEVAL_AMBIGUITY_DELTA",
    "V13_SOURCE_RETRIEVAL_RESULT_BAND",
    "V13_SOURCE_RETRIEVAL_MIN_FOCUS_SCORE",
    "V13_FAST_TIMEOUT_SECONDS",
    "V13_HEAVY_TIMEOUT_SECONDS",
    "V13_PLANNER_MAX_OUTPUT_TOKENS",
    "V13_FAST_MAX_OUTPUT_TOKENS",
    "V13_HEAVY_MAX_OUTPUT_TOKENS",
    "V13_FAST_CONTEXT_CHARS",
    "V13_HEAVY_CONTEXT_CHARS",
    "V13_MAX_EVIDENCE_ITEMS_ASK",
    "V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE",
    "V13_MIN_SECONDS_FOR_REFINEMENT",
    "V13_DENSE_QUERY_LIMIT",
    "V13_LEXICAL_QUERY_LIMIT",
    "V13_PAGE_SCAN_LIMIT",
    "V13_PAGE_TEXT_CHARS",
    "V13_PREFERRED_PAGE_SCAN_LIMIT",
    "V13_DB_CONNECT_TIMEOUT_SECONDS",
    "V13_DB_STATEMENT_TIMEOUT_MS",
    "V13_SEMANTIC_CACHE_ENABLED",
    "V13_SEMANTIC_CACHE_AUTO_DDL",
    "V13_SEMANTIC_CACHE_TTL_SECONDS",
    "V13_SEMANTIC_CACHE_SCAN_LIMIT",
    "V13_SEMANTIC_CACHE_THRESHOLD_ASK",
    "V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE",
    "V13_SEMANTIC_CACHE_MIN_QUALITY",
    "V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY",
    "V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS",
    "V13_PRICE_SOL_INPUT",
    "V13_PRICE_SOL_OUTPUT",
    "V13_PRICE_TERRA_INPUT",
    "V13_PRICE_TERRA_OUTPUT",
    "V13_PRICE_LUNA_INPUT",
    "V13_PRICE_LUNA_OUTPUT",
    "V13_PRICE_EMBED_INPUT",
    "V13_STREAM_HEARTBEAT_ENABLED",
    "V13_STREAM_HEARTBEAT_SECONDS",
    "V13_STREAM_HEARTBEAT_BYTES",
    "V13_ENGINE_KEY",
]

V13_ENABLED = (os.environ.get("MM_V13_ENABLED") or "1").strip() != "0"
V13_ASK_ENABLED = (os.environ.get("MM_V13_ASK_ENABLED") or "1").strip() != "0"
V13_ROOT_CAUSE_ENABLED = (os.environ.get("MM_V13_ROOT_CAUSE_ENABLED") or "1").strip() != "0"
V13_CODE_MARKER = "ask-root-v13-prod-task-aware-source-selection-hardened-stream-v5-1"
V13_RELEASE_ID = (os.environ.get("MM_V13_RELEASE_ID") or "2026-07-29.5.1").strip()
OPENAI_RESPONSES_URL = (os.environ.get("OPENAI_RESPONSES_URL") or "https://api.openai.com/v1/responses").strip()

# Model policy. Luna is used only for optional retrieval refinement; Terra handles
# precise/high-confidence synthesis and the borderline evidence-sufficiency gate;
# Sol is reserved for genuinely complex synthesis.
V13_PLANNER_MODEL = (os.environ.get("MM_V13_PLANNER_MODEL") or "gpt-5.6-luna").strip()
V13_FAST_MODEL = (os.environ.get("MM_V13_FAST_MODEL") or "gpt-5.6-terra").strip()
V13_HEAVY_MODEL = (os.environ.get("MM_V13_HEAVY_MODEL") or "gpt-5.6-sol").strip()
# The shared gate evaluates whether indexed evidence supports the exact request.
# It does not classify specific phrases, domains, greetings, or source types. Terra
# is the default because false admission is more damaging than the small extra cost
# paid only for borderline ASK cases and for Root Cause mode-fit verification.
V13_EVIDENCE_GATE_MODEL = (os.environ.get("MM_V13_EVIDENCE_GATE_MODEL") or V13_FAST_MODEL).strip()
V13_EVIDENCE_GATE_EFFORT = (os.environ.get("MM_V13_EVIDENCE_GATE_EFFORT") or "medium").strip()
V13_FAST_EFFORT = (os.environ.get("MM_V13_FAST_EFFORT") or "medium").strip()
V13_ASK_HEAVY_EFFORT = (os.environ.get("MM_V13_ASK_HEAVY_EFFORT") or "medium").strip()
V13_ROOT_HEAVY_EFFORT = (os.environ.get("MM_V13_ROOT_HEAVY_EFFORT") or "high").strip()
# Pro mode is opt-in. Standard mode is the production default because it is faster and cheaper.
V13_HEAVY_REASONING_MODE = (os.environ.get("MM_V13_HEAVY_REASONING_MODE") or "").strip().lower()
if V13_HEAVY_REASONING_MODE not in {"", "pro"}:
    V13_HEAVY_REASONING_MODE = ""

# Assistant Core V2 is the production orchestration layer. It reuses the stable V13
# retrieval/synthesis primitives but chooses the response mode only after neutral
# cross-source retrieval. Default ON; set MM_ASSISTANT_CORE_V2_ENABLED=0 only for rollback.
ASSISTANT_CORE_V2_ENABLED = (os.environ.get("MM_ASSISTANT_CORE_V2_ENABLED") or "1").strip() != "0"
ASSISTANT_CORE_V2_CODE_MARKER = "assistant-core-v2-root-evidence-assurance-v10-3-3-final-20260831-3"
ASSISTANT_CORE_V2_RELEASE_ID = (os.environ.get("MM_ASSISTANT_CORE_V2_RELEASE_ID") or "2026-08-31.3").strip()
RESULT_INCOMPLETE_ANSWER_CONTRACT = "INCOMPLETE_ANSWER_CONTRACT"
ASSISTANT_UI_RENDER_VERSION = "assistant-ui-html-lossless-v10-1-20260806-1"
ASSISTANT_ASK_UI_RENDER_VERSION = "assistant-ui-ask-root-parity-v1-20260901-1"
ASSISTANT_UI_MAX_HTML_CHARS = max(8000, min(60000, int(
    os.environ.get("MM_ASSISTANT_UI_MAX_HTML_CHARS", "32000")
)))
ASSISTANT_CORE_ROUTER_MODEL = (os.environ.get("MM_ASSISTANT_CORE_ROUTER_MODEL") or V13_FAST_MODEL).strip()
ASSISTANT_CORE_ROUTER_FALLBACK_MODEL = (
    os.environ.get("MM_ASSISTANT_CORE_ROUTER_FALLBACK_MODEL") or V13_PLANNER_MODEL
).strip()
ASSISTANT_CORE_ROUTER_EFFORT = (os.environ.get("MM_ASSISTANT_CORE_ROUTER_EFFORT") or "medium").strip()
# Smart Diagnostic uses one quality-oriented Responses API call after routing.
# Keeping it inside the 5.6 model family gives real usage accounting and an
# enforceable max_output_tokens/cost ceiling, unlike the legacy chat fallback.
ASSISTANT_CORE_SMART_MODEL = (os.environ.get("MM_ASSISTANT_CORE_SMART_MODEL") or V13_HEAVY_MODEL).strip()
ASSISTANT_CORE_SMART_EFFORT = (os.environ.get("MM_ASSISTANT_CORE_SMART_EFFORT") or "medium").strip()
ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS = max(8, min(18, int(os.environ.get("MM_ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS", "15"))))
ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS = max(1400, min(3600, int(os.environ.get("MM_ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS", "3000"))))
ASSISTANT_CORE_ROUTER_MAX_CONTEXT_CHARS = max(7000, min(18000, int(os.environ.get("MM_ASSISTANT_CORE_ROUTER_MAX_CONTEXT_CHARS", "14000"))))
ASSISTANT_CORE_MAX_FACETS = max(3, min(10, int(os.environ.get("MM_ASSISTANT_CORE_MAX_FACETS", "8"))))
ASSISTANT_CORE_MAX_FACET_DENSE_QUERIES = max(4, min(16, int(os.environ.get("MM_ASSISTANT_CORE_MAX_FACET_DENSE_QUERIES", "12"))))
ASSISTANT_CORE_MAX_FACET_LEXICAL_QUERIES = max(6, min(28, int(os.environ.get("MM_ASSISTANT_CORE_MAX_FACET_LEXICAL_QUERIES", "20"))))
ASSISTANT_CORE_FACET_DENSE_CANDIDATE_K = max(12, min(36, int(os.environ.get("MM_ASSISTANT_CORE_FACET_DENSE_CANDIDATE_K", "22"))))
ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD = max(0.20, min(0.60, float(os.environ.get("MM_ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD", "0.30"))))
ASSISTANT_CORE_REPAIR_MAX_OUTPUT_TOKENS = max(1600, min(6000, int(os.environ.get("MM_ASSISTANT_CORE_REPAIR_MAX_OUTPUT_TOKENS", "3600"))))
ASSISTANT_CORE_GENERAL_KNOWLEDGE_ENABLED = (os.environ.get("MM_ASSISTANT_CORE_GENERAL_KNOWLEDGE_ENABLED") or "1").strip() != "0"

# End-to-end target: normal responses should finish well below these values; the
# hard stream guard returns an explicit TIMEOUT before 75 seconds. The monetary caps
# are deliberately a little wider than the first proposal in favour of consistency.
# Per-mode internal deadlines remain bounded even though the outer safety ceiling is
# widened. ASK does not receive a two-minute reasoning budget; Root Cause and Smart
# Diagnostic get enough room for one quality model plus one bounded fallback.
ASSISTANT_CORE_ASK_DEADLINE_SECONDS = max(50, min(85, int(os.environ.get("MM_ASSISTANT_CORE_ASK_DEADLINE_SECONDS", "74"))))
ASSISTANT_CORE_ROOT_CAUSE_DEADLINE_SECONDS = max(65, min(105, int(os.environ.get("MM_ASSISTANT_CORE_ROOT_CAUSE_DEADLINE_SECONDS", "92"))))
ASSISTANT_CORE_SMART_START_DEADLINE_SECONDS = max(70, min(110, int(os.environ.get("MM_ASSISTANT_CORE_SMART_START_DEADLINE_SECONDS", "96"))))
ASSISTANT_CORE_SMART_TURN_DEADLINE_SECONDS = max(60, min(105, int(os.environ.get("MM_ASSISTANT_CORE_SMART_TURN_DEADLINE_SECONDS", "86"))))
ASSISTANT_CORE_HARD_TIMEOUT_SECONDS = max(95, min(120, int(os.environ.get("MM_ASSISTANT_CORE_HARD_TIMEOUT_SECONDS", "115"))))
ASSISTANT_CORE_MAX_LLM_CALLS_ASK = max(3, min(4, int(os.environ.get("MM_ASSISTANT_CORE_MAX_LLM_CALLS_ASK", "4"))))
ASSISTANT_CORE_MAX_LLM_CALLS_ROOT_CAUSE = max(3, min(4, int(os.environ.get("MM_ASSISTANT_CORE_MAX_LLM_CALLS_ROOT_CAUSE", "4"))))
ASSISTANT_CORE_MAX_LLM_CALLS_SMART_START = max(3, min(4, int(os.environ.get("MM_ASSISTANT_CORE_MAX_LLM_CALLS_SMART_START", "4"))))
ASSISTANT_CORE_MAX_LLM_CALLS_SMART_TURN = max(2, min(3, int(os.environ.get("MM_ASSISTANT_CORE_MAX_LLM_CALLS_SMART_TURN", "3"))))
ASSISTANT_CORE_MAX_COST_ASK_USD = max(0.08, float(os.environ.get("MM_ASSISTANT_CORE_MAX_COST_ASK_USD", "0.25")))
ASSISTANT_CORE_MAX_COST_ROOT_CAUSE_USD = max(0.12, float(os.environ.get("MM_ASSISTANT_CORE_MAX_COST_ROOT_CAUSE_USD", "0.40")))
ASSISTANT_CORE_MAX_COST_SMART_START_USD = max(0.12, float(os.environ.get("MM_ASSISTANT_CORE_MAX_COST_SMART_START_USD", "0.35")))
ASSISTANT_CORE_MAX_COST_SMART_TURN_USD = max(0.08, float(os.environ.get("MM_ASSISTANT_CORE_MAX_COST_SMART_TURN_USD", "0.25")))
ASSISTANT_CORE_GENERAL_MAX_OUTPUT_TOKENS = max(1200, min(4200, int(os.environ.get("MM_ASSISTANT_CORE_GENERAL_MAX_OUTPUT_TOKENS", "2600"))))
ASSISTANT_CORE_SMART_MAX_OUTPUT_TOKENS = max(2400, min(7000, int(os.environ.get("MM_ASSISTANT_CORE_SMART_MAX_OUTPUT_TOKENS", "5200"))))

# Hard synchronous budgets. These are intentionally below the proxy timeout.
V13_ASK_DEADLINE_SECONDS = max(20, min(58, int(os.environ.get("MM_V13_ASK_DEADLINE_SECONDS", "48"))))
V13_ROOT_CAUSE_DEADLINE_SECONDS = max(25, min(58, int(os.environ.get("MM_V13_ROOT_CAUSE_DEADLINE_SECONDS", "55"))))
V13_MAX_LLM_CALLS_ASK = max(1, min(2, int(os.environ.get("MM_V13_MAX_LLM_CALLS_ASK", "2"))))
V13_MAX_LLM_CALLS_ROOT_CAUSE = max(1, min(2, int(os.environ.get("MM_V13_MAX_LLM_CALLS_ROOT_CAUSE", "2"))))
V13_MAX_ESTIMATED_COST_ASK_USD = max(0.05, float(os.environ.get("MM_V13_MAX_ESTIMATED_COST_ASK_USD", "0.30")))
V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD = max(0.08, float(os.environ.get("MM_V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD", "0.45")))
V13_PLANNER_TIMEOUT_SECONDS = max(6, min(15, int(os.environ.get("MM_V13_PLANNER_TIMEOUT_SECONDS", "10"))))
V13_EVIDENCE_GATE_TIMEOUT_SECONDS = max(6, min(18, int(os.environ.get("MM_V13_EVIDENCE_GATE_TIMEOUT_SECONDS", "14"))))
V13_EVIDENCE_GATE_MAX_OUTPUT_TOKENS = max(900, min(2200, int(os.environ.get("MM_V13_EVIDENCE_GATE_MAX_OUTPUT_TOKENS", "1400"))))
V13_EVIDENCE_GATE_MAX_CANDIDATES = max(6, min(16, int(os.environ.get("MM_V13_EVIDENCE_GATE_MAX_CANDIDATES", "10"))))
V13_EVIDENCE_GATE_MIN_CONFIDENCE = max(0.50, min(0.95, float(os.environ.get("MM_V13_EVIDENCE_GATE_MIN_CONFIDENCE", "0.65"))))
# Deterministic bands decide only clear cases. Borderline support is checked
# semantically, preserving valid terse and multilingual technical requests.
V13_EVIDENCE_CLEAR_SUPPORT_SIM = max(0.45, min(0.80, float(os.environ.get("MM_V13_EVIDENCE_CLEAR_SUPPORT_SIM", "0.58"))))
V13_EVIDENCE_SUPPORT_SIM_WITH_OVERLAP = max(0.35, min(0.75, float(os.environ.get("MM_V13_EVIDENCE_SUPPORT_SIM_WITH_OVERLAP", "0.48"))))
V13_EVIDENCE_CLEAR_REJECT_SIM = max(0.08, min(0.40, float(os.environ.get("MM_V13_EVIDENCE_CLEAR_REJECT_SIM", "0.18"))))
V13_EVIDENCE_MIN_OVERLAP = max(0.01, min(0.25, float(os.environ.get("MM_V13_EVIDENCE_MIN_OVERLAP", "0.05"))))

# Bounded Retrieval Assurance. This layer never buys an extra reasoning call: it
# uses only bounded embeddings, PostgreSQL/FTS scans, adjacent-page expansion and
# explicit structured relationships. The admitted pack remains the baseline; a
# full bounded pack may replace a weaker item only when coverage, exact-identifier
# support, or true semantic support objectively improves.
V13_RETRIEVAL_ASSURANCE_ENABLED = (os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_ENABLED") or "1").strip() != "0"
V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ASK = max(2, min(10, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ASK", "6"))))
V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ROOT_CAUSE = max(3, min(12, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ROOT_CAUSE", "8"))))
# A small deterministic rescue may run before the semantic gate only when the first
# retrieval is plausibly incomplete (explicit document scope, an exact technical
# identifier, or a weak-but-nonzero evidence signal). It never answers by itself: any
# recovered pack must still pass the semantic evidence gate before synthesis.
V13_RETRIEVAL_ASSURANCE_PRE_GATE_MAX_SECONDS = max(2, min(6, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_PRE_GATE_MAX_SECONDS", "4"))))
V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ASK = max(8, min(24, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ASK", "12"))))
V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ROOT_CAUSE = max(12, min(30, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ROOT_CAUSE", "16"))))
V13_RETRIEVAL_ASSURANCE_MAX_DENSE_QUERIES = max(1, min(4, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_DENSE_QUERIES", "3"))))
V13_RETRIEVAL_ASSURANCE_MAX_LEXICAL_QUERIES = max(2, min(8, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_LEXICAL_QUERIES", "6"))))
V13_RETRIEVAL_ASSURANCE_MAX_DOCS = max(1, min(5, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_DOCS", "3"))))
V13_RETRIEVAL_ASSURANCE_PAGE_RADIUS = max(0, min(2, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_PAGE_RADIUS", "1"))))
V13_RETRIEVAL_ASSURANCE_MAX_NEIGHBOR_PAGES = max(4, min(30, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_NEIGHBOR_PAGES", "18"))))
V13_RETRIEVAL_ASSURANCE_MAX_CANDIDATES = max(12, min(40, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_CANDIDATES", "24"))))
V13_RETRIEVAL_ASSURANCE_MAX_FACETS = max(4, min(16, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MAX_FACETS", "10"))))
V13_RETRIEVAL_ASSURANCE_MIN_FACET_GAIN = max(1, min(4, int(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MIN_FACET_GAIN", "1"))))
V13_RETRIEVAL_ASSURANCE_MIN_COVERAGE_GAIN = max(0.02, min(0.30, float(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MIN_COVERAGE_GAIN", "0.08"))))
V13_RETRIEVAL_ASSURANCE_MIN_SUPPORT_GAIN = max(0.02, min(0.25, float(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MIN_SUPPORT_GAIN", "0.06"))))
V13_RETRIEVAL_ASSURANCE_MIN_NEW_SEMANTIC_SIM = max(0.30, min(0.75, float(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MIN_NEW_SEMANTIC_SIM", "0.42"))))
V13_RETRIEVAL_ASSURANCE_MIN_NEW_OVERLAP = max(0.03, min(0.25, float(os.environ.get("MM_V13_RETRIEVAL_ASSURANCE_MIN_NEW_OVERLAP", "0.08"))))

# Task-aware source retrieval. This is a monotonic ASK-only layer: it performs a
# bounded title/description scan without changing the baseline evidence pack. When a
# source title is a strong semantic/lexical match, the existing ASK evidence-gate call
# is forced to classify whether the user primarily wants to open/retrieve content. If
# that classification is not confident, the untouched V13.4 synthesis path continues.
V13_SOURCE_RETRIEVAL_ENABLED = (os.environ.get("MM_V13_SOURCE_RETRIEVAL_ENABLED") or "1").strip() != "0"
V13_SOURCE_RETRIEVAL_SCAN_LIMIT = max(100, min(2400, int(os.environ.get("MM_V13_SOURCE_RETRIEVAL_SCAN_LIMIT", "1200"))))
V13_SOURCE_RETRIEVAL_MAX_CANDIDATES = max(3, min(16, int(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MAX_CANDIDATES", "8"))))
V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE = max(0.50, min(0.95, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE", "0.68"))))
V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE = max(0.55, min(0.98, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE", "0.72"))))
V13_SOURCE_RETRIEVAL_MIN_TASK_CONFIDENCE = max(0.55, min(0.95, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MIN_TASK_CONFIDENCE", "0.72"))))
V13_SOURCE_RETRIEVAL_REQUIRE_TYPE_CONFIDENCE = max(0.72, min(0.98, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_REQUIRE_TYPE_CONFIDENCE", "0.85"))))
V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW = max(2, min(5, int(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW", "3"))))
V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY = max(4, min(12, int(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY", "8"))))
V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS = max(6, min(48, int(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS", "24"))))
# Hardening: source modality is never allowed to outrank a materially better content
# match. A semantic source-type preference may decide only inside a narrow relevance
# band; an explicit required type is enforced by the task contract and otherwise the
# direct route falls back rather than returning a nearby-but-wrong item.
V13_SOURCE_RETRIEVAL_MIN_SEMANTIC_SCORE = max(0.45, min(0.90, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MIN_SEMANTIC_SCORE", "0.62"))))
V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE = max(0.55, min(0.95, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE", "0.74"))))
V13_SOURCE_RETRIEVAL_PREFERENCE_MAX_GAP = max(0.0, min(0.15, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_PREFERENCE_MAX_GAP", "0.05"))))
V13_SOURCE_RETRIEVAL_AMBIGUITY_DELTA = max(0.0, min(0.10, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_AMBIGUITY_DELTA", "0.025"))))
V13_SOURCE_RETRIEVAL_RESULT_BAND = max(0.04, min(0.25, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_RESULT_BAND", "0.12"))))
V13_SOURCE_RETRIEVAL_MIN_FOCUS_SCORE = max(0.35, min(0.90, float(os.environ.get("MM_V13_SOURCE_RETRIEVAL_MIN_FOCUS_SCORE", "0.58"))))
V13_FAST_TIMEOUT_SECONDS = max(12, min(35, int(os.environ.get("MM_V13_FAST_TIMEOUT_SECONDS", "26"))))
V13_HEAVY_TIMEOUT_SECONDS = max(20, min(45, int(os.environ.get("MM_V13_HEAVY_TIMEOUT_SECONDS", "40"))))
V13_PLANNER_MAX_OUTPUT_TOKENS = max(800, min(2000, int(os.environ.get("MM_V13_PLANNER_MAX_OUTPUT_TOKENS", "1200"))))
V13_FAST_MAX_OUTPUT_TOKENS = max(1800, min(4200, int(os.environ.get("MM_V13_FAST_MAX_OUTPUT_TOKENS", "3000"))))
V13_HEAVY_MAX_OUTPUT_TOKENS = max(3000, min(8000, int(os.environ.get("MM_V13_HEAVY_MAX_OUTPUT_TOKENS", "6000"))))
V13_FAST_CONTEXT_CHARS = max(8000, min(28000, int(os.environ.get("MM_V13_FAST_CONTEXT_CHARS", "18000"))))
V13_HEAVY_CONTEXT_CHARS = max(16000, min(60000, int(os.environ.get("MM_V13_HEAVY_CONTEXT_CHARS", "38000"))))
V13_MAX_EVIDENCE_ITEMS_ASK = max(4, min(12, int(os.environ.get("MM_V13_MAX_EVIDENCE_ITEMS_ASK", "8"))))
V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE = max(6, min(16, int(os.environ.get("MM_V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE", "10"))))
V13_MIN_SECONDS_FOR_REFINEMENT = max(24, min(40, int(os.environ.get("MM_V13_MIN_SECONDS_FOR_REFINEMENT", "32"))))
V13_DENSE_QUERY_LIMIT = max(1, min(5, int(os.environ.get("MM_V13_DENSE_QUERY_LIMIT", "4"))))
V13_LEXICAL_QUERY_LIMIT = max(2, min(8, int(os.environ.get("MM_V13_LEXICAL_QUERY_LIMIT", "6"))))
V13_PAGE_SCAN_LIMIT = max(80, min(900, int(os.environ.get("MM_V13_PAGE_SCAN_LIMIT", "500"))))
V13_PAGE_TEXT_CHARS = max(1800, min(12000, int(os.environ.get("MM_V13_PAGE_TEXT_CHARS", "7000"))))
V13_PREFERRED_PAGE_SCAN_LIMIT = max(80, min(900, int(os.environ.get("MM_V13_PREFERRED_PAGE_SCAN_LIMIT", "500"))))
V13_DB_CONNECT_TIMEOUT_SECONDS = max(2, min(10, int(os.environ.get("MM_V13_DB_CONNECT_TIMEOUT_SECONDS", "5"))))
V13_DB_STATEMENT_TIMEOUT_MS = max(3000, min(20000, int(os.environ.get("MM_V13_DB_STATEMENT_TIMEOUT_MS", "9000"))))

# Semantic cache. It is deliberately conservative: changed codes, numbers, polarity,
# source constraints, or root-cause symptom class prevent reuse even at high similarity.
V13_SEMANTIC_CACHE_ENABLED = (os.environ.get("MM_V13_SEMANTIC_CACHE_ENABLED") or "1").strip() != "0"
V13_SEMANTIC_CACHE_AUTO_DDL = (os.environ.get("MM_V13_SEMANTIC_CACHE_AUTO_DDL") or "1").strip() != "0"
V13_SEMANTIC_CACHE_TTL_SECONDS = max(300, min(30 * 24 * 3600, int(os.environ.get("MM_V13_SEMANTIC_CACHE_TTL_SECONDS", "604800"))))
V13_SEMANTIC_CACHE_SCAN_LIMIT = max(10, min(250, int(os.environ.get("MM_V13_SEMANTIC_CACHE_SCAN_LIMIT", "80"))))
V13_SEMANTIC_CACHE_THRESHOLD_ASK = max(0.90, min(0.995, float(os.environ.get("MM_V13_SEMANTIC_CACHE_THRESHOLD_ASK", "0.965"))))
V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE = max(0.93, min(0.999, float(os.environ.get("MM_V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE", "0.978"))))
V13_SEMANTIC_CACHE_MIN_QUALITY = max(0.60, min(0.98, float(os.environ.get("MM_V13_SEMANTIC_CACHE_MIN_QUALITY", "0.80"))))
V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY = max(50, min(5000, int(os.environ.get("MM_V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY", "600"))))
V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS = max(15, min(300, int(os.environ.get("MM_V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS", "60"))))

# Standard API prices per 1M text tokens. Cache reads/writes are accounted from usage details.
V13_PRICE_SOL_INPUT = float(os.environ.get("MM_V13_PRICE_SOL_INPUT", "5.0"))
V13_PRICE_SOL_OUTPUT = float(os.environ.get("MM_V13_PRICE_SOL_OUTPUT", "30.0"))
V13_PRICE_TERRA_INPUT = float(os.environ.get("MM_V13_PRICE_TERRA_INPUT", "2.5"))
V13_PRICE_TERRA_OUTPUT = float(os.environ.get("MM_V13_PRICE_TERRA_OUTPUT", "15.0"))
V13_PRICE_LUNA_INPUT = float(os.environ.get("MM_V13_PRICE_LUNA_INPUT", "1.0"))
V13_PRICE_LUNA_OUTPUT = float(os.environ.get("MM_V13_PRICE_LUNA_OUTPUT", "6.0"))
V13_PRICE_EMBED_INPUT = float(os.environ.get("MM_V13_PRICE_EMBED_INPUT", "0.02"))

# Both ASK and Root Cause stream valid JSON whitespace while processing. Bubble still
# receives exactly one JSON object because the Worker calls response.text()/JSON.parse().
V13_STREAM_HEARTBEAT_ENABLED = (os.environ.get("MM_V13_STREAM_HEARTBEAT_ENABLED") or "1").strip() != "0"
V13_STREAM_HEARTBEAT_SECONDS = max(5, min(25, int(os.environ.get("MM_V13_STREAM_HEARTBEAT_SECONDS", "12"))))
V13_STREAM_HEARTBEAT_BYTES = max(512, min(8192, int(os.environ.get("MM_V13_STREAM_HEARTBEAT_BYTES", "4096"))))

# Semantic-cache entries are isolated by the complete active engine policy, not only
# by the public marker. Changing a model/budget creates a new cache generation.
V13_ENGINE_KEY = hashlib.sha256(
    "|".join(
        [
            V13_CODE_MARKER,
            V13_RELEASE_ID,
            ASSISTANT_CORE_V2_CODE_MARKER if ASSISTANT_CORE_V2_ENABLED else "assistant-core-v2-off",
            ASSISTANT_CORE_V2_RELEASE_ID,
            str(ASSISTANT_CORE_V2_ENABLED),
            ASSISTANT_CORE_ROUTER_MODEL,
            ASSISTANT_CORE_ROUTER_EFFORT,
            str(ASSISTANT_CORE_GENERAL_KNOWLEDGE_ENABLED),
            str(ASSISTANT_CORE_ASK_DEADLINE_SECONDS),
            str(ASSISTANT_CORE_ROOT_CAUSE_DEADLINE_SECONDS),
            str(ASSISTANT_CORE_MAX_COST_ASK_USD),
            str(ASSISTANT_CORE_MAX_COST_ROOT_CAUSE_USD),
            V13_PLANNER_MODEL,
            V13_EVIDENCE_GATE_MODEL,
            V13_EVIDENCE_GATE_EFFORT,
            str(V13_EVIDENCE_GATE_TIMEOUT_SECONDS),
            str(V13_EVIDENCE_GATE_MAX_OUTPUT_TOKENS),
            str(V13_EVIDENCE_GATE_MAX_CANDIDATES),
            str(V13_EVIDENCE_GATE_MIN_CONFIDENCE),
            str(V13_EVIDENCE_CLEAR_SUPPORT_SIM),
            str(V13_EVIDENCE_SUPPORT_SIM_WITH_OVERLAP),
            str(V13_EVIDENCE_CLEAR_REJECT_SIM),
            str(V13_EVIDENCE_MIN_OVERLAP),
            str(V13_RETRIEVAL_ASSURANCE_ENABLED),
            str(V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ASK),
            str(V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ROOT_CAUSE),
            str(V13_RETRIEVAL_ASSURANCE_PRE_GATE_MAX_SECONDS),
            str(V13_RETRIEVAL_ASSURANCE_MAX_DENSE_QUERIES),
            str(V13_RETRIEVAL_ASSURANCE_MAX_LEXICAL_QUERIES),
            str(V13_RETRIEVAL_ASSURANCE_PAGE_RADIUS),
            str(V13_RETRIEVAL_ASSURANCE_MIN_COVERAGE_GAIN),
            str(V13_RETRIEVAL_ASSURANCE_MIN_SUPPORT_GAIN),
            str(V13_RETRIEVAL_ASSURANCE_MIN_NEW_SEMANTIC_SIM),
            str(V13_RETRIEVAL_ASSURANCE_MIN_NEW_OVERLAP),
            str(V13_SOURCE_RETRIEVAL_ENABLED),
            str(V13_SOURCE_RETRIEVAL_SCAN_LIMIT),
            str(V13_SOURCE_RETRIEVAL_MAX_CANDIDATES),
            str(V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE),
            str(V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE),
            str(V13_SOURCE_RETRIEVAL_MIN_TASK_CONFIDENCE),
            str(V13_SOURCE_RETRIEVAL_REQUIRE_TYPE_CONFIDENCE),
            str(V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW),
            str(V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY),
            str(V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS),
            str(V13_SOURCE_RETRIEVAL_MIN_SEMANTIC_SCORE),
            str(V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE),
            str(V13_SOURCE_RETRIEVAL_PREFERENCE_MAX_GAP),
            str(V13_SOURCE_RETRIEVAL_AMBIGUITY_DELTA),
            str(V13_SOURCE_RETRIEVAL_RESULT_BAND),
            str(V13_SOURCE_RETRIEVAL_MIN_FOCUS_SCORE),
            V13_FAST_MODEL,
            V13_HEAVY_MODEL,
            V13_FAST_EFFORT,
            V13_ASK_HEAVY_EFFORT,
            V13_ROOT_HEAVY_EFFORT,
            V13_HEAVY_REASONING_MODE or "standard",
            str(V13_MAX_LLM_CALLS_ASK),
            str(V13_MAX_LLM_CALLS_ROOT_CAUSE),
            str(V13_FAST_CONTEXT_CHARS),
            str(V13_HEAVY_CONTEXT_CHARS),
            str(V13_FAST_MAX_OUTPUT_TOKENS),
            str(V13_HEAVY_MAX_OUTPUT_TOKENS),
        ]
    ).encode("utf-8")
).hexdigest()[:32]
