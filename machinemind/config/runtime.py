"""Production runtime configuration extracted verbatim from the certified monolith.

This module intentionally performs the same environment reads at import time and
exports the historical constant names unchanged. It contains configuration only;
no routing, retrieval, ranking, prompt or response behavior lives here.
"""
from __future__ import annotations

import os
import re
from decimal import Decimal

__all__ = [
    "AI_INTERNAL_SECRET",
    "FETCH_TIMEOUT",
    "MAX_PDF_BYTES",
    "XLSX_INGEST_ENABLED",
    "MAX_XLSX_BYTES",
    "XLSX_MAX_SHEETS",
    "XLSX_MAX_ROWS_PER_SHEET",
    "XLSX_MAX_COLS_PER_SHEET",
    "XLSX_MAX_CELLS_TOTAL",
    "XLSX_MAX_TEXT_CHARS",
    "XLSX_PAGE_TARGET_CHARS",
    "XLSX_MAX_CELL_CHARS",
    "XLSX_MAX_ROW_CHARS",
    "XLSX_MIN_TEXT_CHARS",
    "XLSX_INCLUDE_HIDDEN_SHEETS",
    "DB_HOST",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "MIN_TEXT_CHARS",
    "MIN_TEXT_CHARS_SHORT",
    "MIN_PAGE_CHARS",
    "MIN_PAGES_WITH_TEXT_ABS",
    "MIN_PAGES_WITH_TEXT_PCT",
    "CHUNK_TARGET_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "CHUNK_MIN_CHARS",
    "OPENAI_API_KEY",
    "OPENAI_EMBED_MODEL",
    "OPENAI_EMBED_URL",
    "INGEST_METERING_VERSION",
    "INGEST_PRICING_VERSION",
    "INGEST_CREDITS_PER_USD",
    "INGEST_PRICE_EMBED_INPUT_USD_PER_MILLION",
    "INGEST_LEDGER_AUTO_DDL",
    "INGEST_PROCESSING_STALE_SECONDS",
    "OPENAI_CHAT_MODEL",
    "OPENAI_CHAT_URL",
    "OPENAI_RERANK_MODEL",
    "RERANK_MAX_CANDIDATES",
    "RERANK_SNIPPET_CHARS",
    "RERANK_TIMEOUT",
    "RERANK_ENABLED",
    "RERANK_MIN_SIM_MAX",
    "RERANK_MAX_SIM_MAX",
    "RERANK_MAX_SPREAD",
    "RERANK_MIN_CANDIDATES",
    "FINAL_CITATION_LOCK_DELTA",
    "FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA",
    "FINAL_CITATION_LOCK_SET_DELTA",
    "FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA",
    "FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA",
    "ROOT_CAUSE_SET_LOCK_DELTA",
    "ASK_SIM_THRESHOLD",
    "ASK_SHORT_QUERY_SIM_THRESHOLD",
    "ASK_MAX_TOP_K",
    "ASK_SNIPPET_CHARS",
    "ASK_MAX_CONTEXT_CHARS",
    "DRAFT_PS_SIM_THRESHOLD",
    "STRUCTURED_RESCUE_ENABLED",
    "STRUCTURED_RESCUE_SCAN_LIMIT",
    "STRUCTURED_RESCUE_MAX_HITS",
    "ASK_EVIDENCE_COMPILER_ENABLED",
    "ASK_EVIDENCE_ANALYZER_MODEL",
    "ASK_EVIDENCE_ANSWER_MODEL",
    "ASK_EVIDENCE_SCOPE_PAGE_LIMIT",
    "ASK_EVIDENCE_TOP_PAGES",
    "ASK_EVIDENCE_MAX_PAGE_CHARS",
    "ASK_EVIDENCE_MAX_CONTEXT_CHARS",
    "ASK_EVIDENCE_MIN_PAGE_SCORE",
    "ASK_EVIDENCE_VERIFIER_ENABLED",
    "ASK_EVIDENCE_VERIFIER_MODEL",
    "ASK_EVIDENCE_VERIFIER_TIMEOUT",
    "ASK_EVIDENCE_VERIFIER_MAX_CONTEXT_CHARS",
    "ASK_FULL_CONTEXT_ENABLED",
    "ASK_FULL_CONTEXT_MAX_DOCS",
    "ASK_FULL_CONTEXT_MAX_PAGES",
    "ASK_FULL_CONTEXT_MAX_CHARS",
    "ASK_FULL_CONTEXT_PAGE_CHARS",
    "ASK_FULL_CONTEXT_TIMEOUT",
    "ASK_FULL_CONTEXT_MODEL",
    "ASK_STRUCTURED_DIRECT_ENABLED",
    "ASK_STRUCTURED_DIRECT_MAX_ITEMS",
    "ASK_STRUCTURED_DIRECT_SCAN_LIMIT",
    "ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS",
    "ASK_STRUCTURED_DIRECT_TEXT_CHARS",
    "ASK_STRUCTURED_DIRECT_TIMEOUT",
    "ASK_STRUCTURED_DIRECT_MODEL",
    "ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED",
    "ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS",
    "ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT",
    "ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS",
    "ASK_UI_MAX_POINTS",
    "ASK_UI_MAX_ANSWER_CHARS",
    "ASK_UI_MAX_LINKS",
    "ASK_UI_MAX_CITATIONS",
    "ASK_UI_MAX_SNIPPET_CLEAN_CHARS",
    "ASK_UI_MANUAL_SUPPORT_SNIPPET_CHARS",
    "ASK_UI_MAX_STRUCTURED_ANSWER_CHARS",
    "ASK_UI_STRUCTURED_MAX_CITATIONS",
    "ASK_UI_STRUCTURED_MAX_LINKS",
    "SEMANTIC_QUERY_PLANNER_MODEL",
    "SEMANTIC_QUERY_PLANNER_TIMEOUT",
    "SEMANTIC_MAX_DENSE_QUERIES",
    "SEMANTIC_MAX_LEXICAL_QUERIES",
    "SEMANTIC_EXACT_MACHINE_BONUS",
    "ASK_ROOT_CAUSE_CODE_MARKER",
    "ROOT_CAUSE_INTENT_MODEL",
    "ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE",
    "ROOT_CAUSE_GATE_MIN_MARGIN",
    "ROOT_CAUSE_GATE_MIN_PRELIM_SIM",
    "ROOT_CAUSE_GATE_MIN_PRELIM_HITS",
    "ROOT_CAUSE_GATE_PRELIM_TOP_K",
    "DIAGNOSTIC_PIPELINE_ENABLED",
    "DIAGNOSTIC_EVIDENCE_MODEL",
    "ROOT_CAUSE_RESPONSE_MODEL",
    "ROOT_CAUSE_EXTRA_CANDIDATE_K",
    "ROOT_CAUSE_MAX_EVIDENCE_POOL",
    "ROOT_CAUSE_MAX_PROMPT_CITATIONS",
    "ROOT_CAUSE_DIRECT_SIGNAL_BONUS",
    "ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY",
    "ROOT_CAUSE_HARD_EXCLUDE_PENALTY",
    "ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY",
    "ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES",
    "ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA",
    "ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL",
    "RESPONSE_ARB_ENABLED",
    "ASK_CANDIDATE_ENABLED",
    "ROOT_CAUSE_CANDIDATE_ENABLED",
    "ROOT_CAUSE_SKIP_CANDIDATE_IF_BASELINE_PROXY_GTE",
    "RESPONSE_ARB_KEEP_BASELINE_ON_TIE",
    "ASK_ARB_MIN_DELTA",
    "ROOT_CAUSE_ARB_MIN_DELTA",
    "ROOT_CAUSE_CANDIDATE_CORE_PROMOTION",
    "ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY",
    "ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY",
    "ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY",
    "ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY",
    "ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K",
    "ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K",
    "ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX",
    "ASK_CANDIDATE_MATRIX_TOP_K",
    "ASK_CANDIDATE_PROMPT_TOP_K",
    "URL_REGEX",
    "EMAIL_REGEX",
    "PHONE_REGEX",
    "CODE_TOKEN_REGEX",
    "URL_HINTS",
    "EMAIL_HINTS",
    "PHONE_HINTS",
    "STRUCTURED_SOURCE_TYPES",
]

AI_INTERNAL_SECRET = (os.environ.get("AI_INTERNAL_SECRET") or "").strip()
FETCH_TIMEOUT = int(os.environ.get("MM_FETCH_TIMEOUT_SECONDS", "30"))
MAX_PDF_BYTES = int(os.environ.get("MM_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

# XLSX ingest is intentionally feature-flagged and isolated from the PDF path.
# Default OFF: deploy the code safely, then enable only after PDF regression tests pass.
XLSX_INGEST_ENABLED = (os.environ.get("MM_XLSX_INGEST_ENABLED") or "0").strip() == "1"
MAX_XLSX_BYTES = int(os.environ.get("MM_MAX_XLSX_BYTES", str(20 * 1024 * 1024)))
XLSX_MAX_SHEETS = int(os.environ.get("MM_XLSX_MAX_SHEETS", "25"))
XLSX_MAX_ROWS_PER_SHEET = int(os.environ.get("MM_XLSX_MAX_ROWS_PER_SHEET", "5000"))
XLSX_MAX_COLS_PER_SHEET = int(os.environ.get("MM_XLSX_MAX_COLS_PER_SHEET", "80"))
XLSX_MAX_CELLS_TOTAL = int(os.environ.get("MM_XLSX_MAX_CELLS_TOTAL", "150000"))
XLSX_MAX_TEXT_CHARS = int(os.environ.get("MM_XLSX_MAX_TEXT_CHARS", "500000"))
XLSX_PAGE_TARGET_CHARS = int(os.environ.get("MM_XLSX_PAGE_TARGET_CHARS", "12000"))
XLSX_MAX_CELL_CHARS = int(os.environ.get("MM_XLSX_MAX_CELL_CHARS", "260"))
XLSX_MAX_ROW_CHARS = int(os.environ.get("MM_XLSX_MAX_ROW_CHARS", "2400"))
XLSX_MIN_TEXT_CHARS = int(os.environ.get("MM_XLSX_MIN_TEXT_CHARS", "120"))
XLSX_INCLUDE_HIDDEN_SHEETS = (os.environ.get("MM_XLSX_INCLUDE_HIDDEN_SHEETS") or "0").strip() == "1"

# DB
DB_HOST = (os.environ.get("MM_DB_HOST") or "").strip()
DB_NAME = (os.environ.get("MM_DB_NAME") or "postgres").strip()
DB_USER = (os.environ.get("MM_DB_USER") or "").strip()
DB_PASSWORD = (os.environ.get("MM_DB_PASSWORD") or "").strip()

# Indicizzabilità
MIN_TEXT_CHARS = int(os.environ.get("MM_MIN_TEXT_CHARS", "2000"))
MIN_TEXT_CHARS_SHORT = int(os.environ.get("MM_MIN_TEXT_CHARS_SHORT", "800"))
MIN_PAGE_CHARS = int(os.environ.get("MM_MIN_PAGE_CHARS", "30"))
MIN_PAGES_WITH_TEXT_ABS = int(os.environ.get("MM_MIN_PAGES_WITH_TEXT_ABS", "2"))
MIN_PAGES_WITH_TEXT_PCT = float(os.environ.get("MM_MIN_PAGES_WITH_TEXT_PCT", "0.20"))

# Chunking
CHUNK_TARGET_CHARS = int(os.environ.get("MM_CHUNK_TARGET_CHARS", "3800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("MM_CHUNK_OVERLAP_CHARS", "800"))
CHUNK_MIN_CHARS = int(os.environ.get("MM_CHUNK_MIN_CHARS", "200"))

# OpenAI
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_EMBED_MODEL = (os.environ.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
OPENAI_EMBED_URL = (os.environ.get("OPENAI_EMBED_URL") or "https://api.openai.com/v1/embeddings").strip()

# Document-ingest cost metering. The current document pipeline pays OpenAI only for
# embeddings; PDF/XLSX parsing, chunking and DB writes remain infrastructure work.
# 1 credit = USD 0.001 by default. Prices are versioned and configurable so future
# OCR/vision/electrical stages can be added without changing Bubble's unit.
INGEST_METERING_VERSION = "ingest-credits-v1-async-ledger"
INGEST_PRICING_VERSION = (
    os.environ.get("MM_INGEST_PRICING_VERSION") or "2026-07-embedding-v1"
).strip()
INGEST_CREDITS_PER_USD = Decimal(
    os.environ.get("MM_INGEST_CREDITS_PER_USD", "1000")
)
INGEST_PRICE_EMBED_INPUT_USD_PER_MILLION = Decimal(
    os.environ.get("MM_INGEST_PRICE_EMBED_INPUT_USD_PER_MILLION", "0.02")
)
INGEST_LEDGER_AUTO_DDL = (
    os.environ.get("MM_INGEST_LEDGER_AUTO_DDL") or "1"
).strip() != "0"
INGEST_PROCESSING_STALE_SECONDS = max(300, min(7200, int(
    os.environ.get("MM_INGEST_PROCESSING_STALE_SECONDS", "1800")
)))

# OpenAI Chat
OPENAI_CHAT_MODEL = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-5.4-mini").strip()
OPENAI_CHAT_URL = (os.environ.get("OPENAI_CHAT_URL") or "https://api.openai.com/v1/chat/completions").strip()

# OpenAI Citation Reranker (on-demand)
OPENAI_RERANK_MODEL = (os.environ.get("OPENAI_RERANK_MODEL") or "gpt-5.4-nano").strip()
RERANK_MAX_CANDIDATES = int(os.environ.get("MM_RERANK_MAX_CANDIDATES", "18"))
RERANK_SNIPPET_CHARS = int(os.environ.get("MM_RERANK_SNIPPET_CHARS", "320"))
RERANK_TIMEOUT = int(os.environ.get("MM_RERANK_TIMEOUT_SECONDS", "30"))
RERANK_ENABLED = (os.environ.get("MM_RERANK_ENABLED") or "1").strip() == "1"
RERANK_MIN_SIM_MAX = float(os.environ.get("MM_RERANK_MIN_SIM_MAX", "0.38"))
RERANK_MAX_SIM_MAX = float(os.environ.get("MM_RERANK_MAX_SIM_MAX", "0.72"))
RERANK_MAX_SPREAD = float(os.environ.get("MM_RERANK_MAX_SPREAD", "0.10"))
RERANK_MIN_CANDIDATES = int(os.environ.get("MM_RERANK_MIN_CANDIDATES", "4"))

FINAL_CITATION_LOCK_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_DELTA", "0.028"))
FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA", "0.042"))
FINAL_CITATION_LOCK_SET_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_SET_DELTA", "0.014"))
FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA", "0.020"))
FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA = float(os.environ.get("MM_FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA", "0.016"))
ROOT_CAUSE_SET_LOCK_DELTA = float(os.environ.get("MM_ROOT_CAUSE_SET_LOCK_DELTA", "0.028"))

ASK_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SIM_THRESHOLD", "0.35"))
ASK_SHORT_QUERY_SIM_THRESHOLD = float(os.environ.get("MM_ASK_SHORT_QUERY_SIM_THRESHOLD", "0.28"))
ASK_MAX_TOP_K = int(os.environ.get("MM_ASK_MAX_TOP_K", "8"))
ASK_SNIPPET_CHARS = int(os.environ.get("MM_ASK_SNIPPET_CHARS", "700"))
ASK_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_MAX_CONTEXT_CHARS", "9000"))
DRAFT_PS_SIM_THRESHOLD = float(os.environ.get("MM_DRAFT_PS_SIM_THRESHOLD", "0.42"))

# Structured-source rescue for machine-wide Ask queries.
# Purpose: when the user asks about procedures/steps/P&S/photos/videos or asks a how-to
# question, do not let long manuals dominate over exact structured records.
STRUCTURED_RESCUE_ENABLED = (os.environ.get("MM_STRUCTURED_RESCUE_ENABLED") or "1").strip() != "0"
STRUCTURED_RESCUE_SCAN_LIMIT = int(os.environ.get("MM_STRUCTURED_RESCUE_SCAN_LIMIT", "220"))
STRUCTURED_RESCUE_MAX_HITS = int(os.environ.get("MM_STRUCTURED_RESCUE_MAX_HITS", "3"))

# ASK v2 generic evidence compiler (query-agnostic, multilingual, non-hardcoded)
# This is NOT a benchmark dictionary: it does not contain expected answers, document ids,
# product codes or test questions. It improves retrieval by analyzing the user query,
# scanning authorized pages/structured sources, then verifying groundedness and completeness.
ASK_EVIDENCE_COMPILER_ENABLED = (os.environ.get("MM_ASK_EVIDENCE_COMPILER_ENABLED") or "1").strip() != "0"
ASK_EVIDENCE_ANALYZER_MODEL = (os.environ.get("MM_ASK_EVIDENCE_ANALYZER_MODEL") or OPENAI_RERANK_MODEL).strip()
ASK_EVIDENCE_ANSWER_MODEL = (os.environ.get("MM_ASK_EVIDENCE_ANSWER_MODEL") or os.environ.get("MM_ROOT_CAUSE_RESPONSE_MODEL") or "gpt-5.4").strip()
ASK_EVIDENCE_SCOPE_PAGE_LIMIT = int(os.environ.get("MM_ASK_EVIDENCE_SCOPE_PAGE_LIMIT", "900"))
ASK_EVIDENCE_TOP_PAGES = int(os.environ.get("MM_ASK_EVIDENCE_TOP_PAGES", "10"))
ASK_EVIDENCE_MAX_PAGE_CHARS = int(os.environ.get("MM_ASK_EVIDENCE_MAX_PAGE_CHARS", "12000"))
ASK_EVIDENCE_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_EVIDENCE_MAX_CONTEXT_CHARS", "24000"))
ASK_EVIDENCE_MIN_PAGE_SCORE = float(os.environ.get("MM_ASK_EVIDENCE_MIN_PAGE_SCORE", "1.5"))
ASK_EVIDENCE_VERIFIER_ENABLED = (os.environ.get("MM_ASK_EVIDENCE_VERIFIER_ENABLED") or "1").strip() != "0"
ASK_EVIDENCE_VERIFIER_MODEL = (os.environ.get("MM_ASK_EVIDENCE_VERIFIER_MODEL") or OPENAI_RERANK_MODEL).strip()
ASK_EVIDENCE_VERIFIER_TIMEOUT = int(os.environ.get("MM_ASK_EVIDENCE_VERIFIER_TIMEOUT_SECONDS", "45"))
ASK_EVIDENCE_VERIFIER_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_EVIDENCE_VERIFIER_MAX_CONTEXT_CHARS", "16000"))

# ASK full-document reader (generic, non-benchmark-specific).
# When the authorized scope is narrow enough, this gives the answer model a much larger
# evidence pack instead of only a few top chunks. It is meant to mimic how a human expert
# would scan the actual manual/procedure before answering factual technical questions.
ASK_FULL_CONTEXT_ENABLED = (os.environ.get("MM_ASK_FULL_CONTEXT_ENABLED") or "1").strip() != "0"
ASK_FULL_CONTEXT_MAX_DOCS = int(os.environ.get("MM_ASK_FULL_CONTEXT_MAX_DOCS", "3"))
ASK_FULL_CONTEXT_MAX_PAGES = int(os.environ.get("MM_ASK_FULL_CONTEXT_MAX_PAGES", "140"))
ASK_FULL_CONTEXT_MAX_CHARS = int(os.environ.get("MM_ASK_FULL_CONTEXT_MAX_CHARS", "120000"))
ASK_FULL_CONTEXT_PAGE_CHARS = int(os.environ.get("MM_ASK_FULL_CONTEXT_PAGE_CHARS", "6500"))
ASK_FULL_CONTEXT_TIMEOUT = int(os.environ.get("MM_ASK_FULL_CONTEXT_TIMEOUT_SECONDS", "120"))
ASK_FULL_CONTEXT_MODEL = (os.environ.get("MM_ASK_FULL_CONTEXT_MODEL") or ASK_EVIDENCE_ANSWER_MODEL).strip()
ASK_STRUCTURED_DIRECT_ENABLED = (os.environ.get("MM_ASK_STRUCTURED_DIRECT_ENABLED") or "1").strip() != "0"
ASK_STRUCTURED_DIRECT_MAX_ITEMS = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_MAX_ITEMS", "12"))
ASK_STRUCTURED_DIRECT_SCAN_LIMIT = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_SCAN_LIMIT", "1200"))
ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS", "28000"))
ASK_STRUCTURED_DIRECT_TEXT_CHARS = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_TEXT_CHARS", "5000"))
ASK_STRUCTURED_DIRECT_TIMEOUT = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_TIMEOUT_SECONDS", "60"))
ASK_STRUCTURED_DIRECT_MODEL = (os.environ.get("MM_ASK_STRUCTURED_DIRECT_MODEL") or ASK_EVIDENCE_ANSWER_MODEL).strip()
ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED = (os.environ.get("MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED") or "1").strip() != "0"
ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS", "2"))
ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT", "180"))
ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS = int(os.environ.get("MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS", "4200"))

# ASK user-facing output polish. Retrieval/citations remain rich; the answer box
# must not expose internal ids or become an unreadable evidence dump.
ASK_UI_MAX_POINTS = int(os.environ.get("MM_ASK_UI_MAX_POINTS", "5"))
ASK_UI_MAX_ANSWER_CHARS = int(os.environ.get("MM_ASK_UI_MAX_ANSWER_CHARS", "2200"))
ASK_UI_MAX_LINKS = int(os.environ.get("MM_ASK_UI_MAX_LINKS", "8"))
ASK_UI_MAX_CITATIONS = int(os.environ.get("MM_ASK_UI_MAX_CITATIONS", "8"))
ASK_UI_MAX_SNIPPET_CLEAN_CHARS = int(os.environ.get("MM_ASK_UI_MAX_SNIPPET_CLEAN_CHARS", "520"))
ASK_UI_MANUAL_SUPPORT_SNIPPET_CHARS = int(os.environ.get("MM_ASK_UI_MANUAL_SUPPORT_SNIPPET_CHARS", "260"))
ASK_UI_MAX_STRUCTURED_ANSWER_CHARS = int(os.environ.get("MM_ASK_UI_MAX_STRUCTURED_ANSWER_CHARS", "5200"))
ASK_UI_STRUCTURED_MAX_CITATIONS = int(os.environ.get("MM_ASK_UI_STRUCTURED_MAX_CITATIONS", "14"))
ASK_UI_STRUCTURED_MAX_LINKS = int(os.environ.get("MM_ASK_UI_STRUCTURED_MAX_LINKS", "14"))

# Shared semantic retrieval planner
SEMANTIC_QUERY_PLANNER_MODEL = (os.environ.get("MM_SEMANTIC_QUERY_PLANNER_MODEL") or "gpt-5.4-mini").strip()
SEMANTIC_QUERY_PLANNER_TIMEOUT = int(os.environ.get("MM_SEMANTIC_QUERY_PLANNER_TIMEOUT_SECONDS", "20"))
SEMANTIC_MAX_DENSE_QUERIES = int(os.environ.get("MM_SEMANTIC_MAX_DENSE_QUERIES", "5"))
SEMANTIC_MAX_LEXICAL_QUERIES = int(os.environ.get("MM_SEMANTIC_MAX_LEXICAL_QUERIES", "5"))
SEMANTIC_EXACT_MACHINE_BONUS = float(os.environ.get("MM_SEMANTIC_EXACT_MACHINE_BONUS", "0.055"))
ASK_ROOT_CAUSE_CODE_MARKER = "ask-root-v13-prod-task-aware-source-selection-hardened-stream-v5-1"

# Root-cause semantic intent gate
ROOT_CAUSE_INTENT_MODEL = (os.environ.get("MM_ROOT_CAUSE_INTENT_MODEL") or "gpt-5.4-mini").strip()
ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE", "0.33"))
ROOT_CAUSE_GATE_MIN_MARGIN = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_MARGIN", "0.05"))
ROOT_CAUSE_GATE_MIN_PRELIM_SIM = float(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_PRELIM_SIM", "0.36"))
ROOT_CAUSE_GATE_MIN_PRELIM_HITS = int(os.environ.get("MM_ROOT_CAUSE_GATE_MIN_PRELIM_HITS", "2"))
ROOT_CAUSE_GATE_PRELIM_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_GATE_PRELIM_TOP_K", "6"))
DIAGNOSTIC_PIPELINE_ENABLED = (os.environ.get("MM_DIAGNOSTIC_PIPELINE_ENABLED") or "1").strip() == "1"
DIAGNOSTIC_EVIDENCE_MODEL = (os.environ.get("MM_DIAGNOSTIC_EVIDENCE_MODEL") or "gpt-5.4-mini").strip()
ROOT_CAUSE_RESPONSE_MODEL = (os.environ.get("MM_ROOT_CAUSE_RESPONSE_MODEL") or "gpt-5.4").strip()
ROOT_CAUSE_EXTRA_CANDIDATE_K = int(os.environ.get("MM_ROOT_CAUSE_EXTRA_CANDIDATE_K", "60"))
ROOT_CAUSE_MAX_EVIDENCE_POOL = int(os.environ.get("MM_ROOT_CAUSE_MAX_EVIDENCE_POOL", "10"))
ROOT_CAUSE_MAX_PROMPT_CITATIONS = int(os.environ.get("MM_ROOT_CAUSE_MAX_PROMPT_CITATIONS", "7"))
ROOT_CAUSE_DIRECT_SIGNAL_BONUS = float(os.environ.get("MM_ROOT_CAUSE_DIRECT_SIGNAL_BONUS", "0.12"))
ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY", "0.14"))
ROOT_CAUSE_HARD_EXCLUDE_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_HARD_EXCLUDE_PENALTY", "0.30"))
ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY", "0.22"))
ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES = int(os.environ.get("MM_ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES", "2"))
ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA = int(os.environ.get("MM_ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA", "2"))
ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL = (os.environ.get("MM_ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL") or "1").strip() != "0"

RESPONSE_ARB_ENABLED = (os.environ.get("MM_RESPONSE_ARB_ENABLED") or "1").strip() != "0"
ASK_CANDIDATE_ENABLED = (os.environ.get("MM_ASK_CANDIDATE_ENABLED") or "1").strip() != "0"
ROOT_CAUSE_CANDIDATE_ENABLED = (os.environ.get("MM_ROOT_CAUSE_CANDIDATE_ENABLED") or "1").strip() != "0"

# Root Cause candidate gating:
# If the baseline proxy score is already high enough, do not run the expensive
# candidate/arbiter branch. Set to 1.30 to effectively disable this skip.
ROOT_CAUSE_SKIP_CANDIDATE_IF_BASELINE_PROXY_GTE = float(
    os.environ.get("MM_ROOT_CAUSE_SKIP_CANDIDATE_IF_BASELINE_PROXY_GTE", "0.80")
)

RESPONSE_ARB_KEEP_BASELINE_ON_TIE = (os.environ.get("MM_RESPONSE_ARB_KEEP_BASELINE_ON_TIE") or "1").strip() != "0"
ASK_ARB_MIN_DELTA = float(os.environ.get("MM_ASK_ARB_MIN_DELTA", "0.035"))
ROOT_CAUSE_ARB_MIN_DELTA = float(os.environ.get("MM_ROOT_CAUSE_ARB_MIN_DELTA", "0.040"))
ROOT_CAUSE_CANDIDATE_CORE_PROMOTION = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_CORE_PROMOTION", "0.08"))
ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY", "0.16"))
ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY", "0.20"))
ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY", "0.16"))
ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY = float(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY", "0.14"))
ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K", "10"))
ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K = int(os.environ.get("MM_ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K", "7"))
ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX = (os.environ.get("MM_ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX") or "1").strip() != "0"
ASK_CANDIDATE_MATRIX_TOP_K = int(os.environ.get("MM_ASK_CANDIDATE_MATRIX_TOP_K", "7"))
ASK_CANDIDATE_PROMPT_TOP_K = int(os.environ.get("MM_ASK_CANDIDATE_PROMPT_TOP_K", "6"))


# -----------------------------
# Entity fallback (URL / email / phone)
# -----------------------------
URL_REGEX = re.compile(r"(https?://[^\s\)\]\}]+|www\.[^\s\)\]\}]+)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

# -----------------------------
# Code token fallback (SENTINEL / part-number / codici)
# -----------------------------
CODE_TOKEN_REGEX = re.compile(r"\b[A-Z0-9_]{6,}\b")

URL_HINTS = ["sito", "website", "web site", "url", "link", "pagina", "dominio", "www"]
EMAIL_HINTS = ["email", "e-mail", "mail", "posta"]
PHONE_HINTS = ["telefono", "cell", "cellulare", "tel", "contatto", "chiamare", "numero"]

STRUCTURED_SOURCE_TYPES = {
    "procedure",
    "step",
    "ps",
    "md_photo",
    "md_video",
}
