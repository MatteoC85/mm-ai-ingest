#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"

CONFIG_ENV_KEYS = [
    'AI_INTERNAL_SECRET',
    'MM_ASK_ARB_MIN_DELTA',
    'MM_ASK_CANDIDATE_ENABLED',
    'MM_ASK_CANDIDATE_MATRIX_TOP_K',
    'MM_ASK_CANDIDATE_PROMPT_TOP_K',
    'MM_ASK_EVIDENCE_ANALYZER_MODEL',
    'MM_ASK_EVIDENCE_ANSWER_MODEL',
    'MM_ASK_EVIDENCE_COMPILER_ENABLED',
    'MM_ASK_EVIDENCE_MAX_CONTEXT_CHARS',
    'MM_ASK_EVIDENCE_MAX_PAGE_CHARS',
    'MM_ASK_EVIDENCE_MIN_PAGE_SCORE',
    'MM_ASK_EVIDENCE_SCOPE_PAGE_LIMIT',
    'MM_ASK_EVIDENCE_TOP_PAGES',
    'MM_ASK_EVIDENCE_VERIFIER_ENABLED',
    'MM_ASK_EVIDENCE_VERIFIER_MAX_CONTEXT_CHARS',
    'MM_ASK_EVIDENCE_VERIFIER_MODEL',
    'MM_ASK_EVIDENCE_VERIFIER_TIMEOUT_SECONDS',
    'MM_ASK_FULL_CONTEXT_ENABLED',
    'MM_ASK_FULL_CONTEXT_MAX_CHARS',
    'MM_ASK_FULL_CONTEXT_MAX_DOCS',
    'MM_ASK_FULL_CONTEXT_MAX_PAGES',
    'MM_ASK_FULL_CONTEXT_MODEL',
    'MM_ASK_FULL_CONTEXT_PAGE_CHARS',
    'MM_ASK_FULL_CONTEXT_TIMEOUT_SECONDS',
    'MM_ASK_MAX_CONTEXT_CHARS',
    'MM_ASK_MAX_TOP_K',
    'MM_ASK_SHORT_QUERY_SIM_THRESHOLD',
    'MM_ASK_SIM_THRESHOLD',
    'MM_ASK_SNIPPET_CHARS',
    'MM_ASK_STRUCTURED_DIRECT_ENABLED',
    'MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_ENABLED',
    'MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_MAX_ITEMS',
    'MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_SCAN_LIMIT',
    'MM_ASK_STRUCTURED_DIRECT_MANUAL_SUPPORT_TEXT_CHARS',
    'MM_ASK_STRUCTURED_DIRECT_MAX_CONTEXT_CHARS',
    'MM_ASK_STRUCTURED_DIRECT_MAX_ITEMS',
    'MM_ASK_STRUCTURED_DIRECT_MODEL',
    'MM_ASK_STRUCTURED_DIRECT_SCAN_LIMIT',
    'MM_ASK_STRUCTURED_DIRECT_TEXT_CHARS',
    'MM_ASK_STRUCTURED_DIRECT_TIMEOUT_SECONDS',
    'MM_ASK_UI_MANUAL_SUPPORT_SNIPPET_CHARS',
    'MM_ASK_UI_MAX_ANSWER_CHARS',
    'MM_ASK_UI_MAX_CITATIONS',
    'MM_ASK_UI_MAX_LINKS',
    'MM_ASK_UI_MAX_POINTS',
    'MM_ASK_UI_MAX_SNIPPET_CLEAN_CHARS',
    'MM_ASK_UI_MAX_STRUCTURED_ANSWER_CHARS',
    'MM_ASK_UI_STRUCTURED_MAX_CITATIONS',
    'MM_ASK_UI_STRUCTURED_MAX_LINKS',
    'MM_ASSISTANT_CORE_ASK_DEADLINE_SECONDS',
    'MM_ASSISTANT_CORE_FACET_DENSE_CANDIDATE_K',
    'MM_ASSISTANT_CORE_FACET_SUPPORT_THRESHOLD',
    'MM_ASSISTANT_CORE_GENERAL_KNOWLEDGE_ENABLED',
    'MM_ASSISTANT_CORE_GENERAL_MAX_OUTPUT_TOKENS',
    'MM_ASSISTANT_CORE_HARD_TIMEOUT_SECONDS',
    'MM_ASSISTANT_CORE_MAX_COST_ASK_USD',
    'MM_ASSISTANT_CORE_MAX_COST_ROOT_CAUSE_USD',
    'MM_ASSISTANT_CORE_MAX_COST_SMART_START_USD',
    'MM_ASSISTANT_CORE_MAX_COST_SMART_TURN_USD',
    'MM_ASSISTANT_CORE_MAX_FACETS',
    'MM_ASSISTANT_CORE_MAX_FACET_DENSE_QUERIES',
    'MM_ASSISTANT_CORE_MAX_FACET_LEXICAL_QUERIES',
    'MM_ASSISTANT_CORE_MAX_LLM_CALLS_ASK',
    'MM_ASSISTANT_CORE_MAX_LLM_CALLS_ROOT_CAUSE',
    'MM_ASSISTANT_CORE_MAX_LLM_CALLS_SMART_START',
    'MM_ASSISTANT_CORE_MAX_LLM_CALLS_SMART_TURN',
    'MM_ASSISTANT_CORE_REPAIR_MAX_OUTPUT_TOKENS',
    'MM_ASSISTANT_CORE_ROOT_CAUSE_DEADLINE_SECONDS',
    'MM_ASSISTANT_CORE_ROUTER_EFFORT',
    'MM_ASSISTANT_CORE_ROUTER_FALLBACK_MODEL',
    'MM_ASSISTANT_CORE_ROUTER_MAX_CONTEXT_CHARS',
    'MM_ASSISTANT_CORE_ROUTER_MAX_OUTPUT_TOKENS',
    'MM_ASSISTANT_CORE_ROUTER_MODEL',
    'MM_ASSISTANT_CORE_ROUTER_TIMEOUT_SECONDS',
    'MM_ASSISTANT_CORE_SMART_EFFORT',
    'MM_ASSISTANT_CORE_SMART_MAX_OUTPUT_TOKENS',
    'MM_ASSISTANT_CORE_SMART_MODEL',
    'MM_ASSISTANT_CORE_SMART_START_DEADLINE_SECONDS',
    'MM_ASSISTANT_CORE_SMART_TURN_DEADLINE_SECONDS',
    'MM_ASSISTANT_CORE_V2_ENABLED',
    'MM_ASSISTANT_CORE_V2_RELEASE_ID',
    'MM_ASSISTANT_UI_MAX_HTML_CHARS',
    'MM_CHUNK_MIN_CHARS',
    'MM_CHUNK_OVERLAP_CHARS',
    'MM_CHUNK_TARGET_CHARS',
    'MM_DB_HOST',
    'MM_DB_NAME',
    'MM_DB_PASSWORD',
    'MM_DB_USER',
    'MM_DIAGNOSTIC_EVIDENCE_MODEL',
    'MM_DIAGNOSTIC_PIPELINE_ENABLED',
    'MM_DRAFT_PS_SIM_THRESHOLD',
    'MM_FETCH_TIMEOUT_SECONDS',
    'MM_FINAL_CITATION_LOCK_DELTA',
    'MM_FINAL_CITATION_LOCK_DIAGNOSTIC_DELTA',
    'MM_FINAL_CITATION_LOCK_FAMILY_WITHIN_SET_DELTA',
    'MM_FINAL_CITATION_LOCK_SET_DELTA',
    'MM_FINAL_CITATION_LOCK_SET_DIAGNOSTIC_DELTA',
    'MM_INGEST_CREDITS_PER_USD',
    'MM_INGEST_LEDGER_AUTO_DDL',
    'MM_INGEST_PRICE_EMBED_INPUT_USD_PER_MILLION',
    'MM_INGEST_PRICING_VERSION',
    'MM_INGEST_PROCESSING_STALE_SECONDS',
    'MM_MAX_PDF_BYTES',
    'MM_MAX_XLSX_BYTES',
    'MM_MIN_PAGES_WITH_TEXT_ABS',
    'MM_MIN_PAGES_WITH_TEXT_PCT',
    'MM_MIN_PAGE_CHARS',
    'MM_MIN_TEXT_CHARS',
    'MM_MIN_TEXT_CHARS_SHORT',
    'MM_RERANK_ENABLED',
    'MM_RERANK_MAX_CANDIDATES',
    'MM_RERANK_MAX_SIM_MAX',
    'MM_RERANK_MAX_SPREAD',
    'MM_RERANK_MIN_CANDIDATES',
    'MM_RERANK_MIN_SIM_MAX',
    'MM_RERANK_SNIPPET_CHARS',
    'MM_RERANK_TIMEOUT_SECONDS',
    'MM_RESPONSE_ARB_ENABLED',
    'MM_RESPONSE_ARB_KEEP_BASELINE_ON_TIE',
    'MM_ROOT_CAUSE_ARB_MIN_DELTA',
    'MM_ROOT_CAUSE_CANDIDATE_CORE_PROMOTION',
    'MM_ROOT_CAUSE_CANDIDATE_ENABLED',
    'MM_ROOT_CAUSE_CANDIDATE_ENABLE_ROLE_AWARE_MATRIX',
    'MM_ROOT_CAUSE_CANDIDATE_MATRIX_TOP_K',
    'MM_ROOT_CAUSE_CANDIDATE_NO_START_LUBE_PENALTY',
    'MM_ROOT_CAUSE_CANDIDATE_PROMPT_TOP_K',
    'MM_ROOT_CAUSE_CANDIDATE_SAFETY_PENALTY',
    'MM_ROOT_CAUSE_CANDIDATE_STARTUP_PENALTY',
    'MM_ROOT_CAUSE_CANDIDATE_SUPPORT_PENALTY',
    'MM_ROOT_CAUSE_DIRECT_SIGNAL_BONUS',
    'MM_ROOT_CAUSE_EXTRA_CANDIDATE_K',
    'MM_ROOT_CAUSE_GATE_MIN_MARGIN',
    'MM_ROOT_CAUSE_GATE_MIN_PRELIM_HITS',
    'MM_ROOT_CAUSE_GATE_MIN_PRELIM_SIM',
    'MM_ROOT_CAUSE_GATE_MIN_SYMPTOM_SCORE',
    'MM_ROOT_CAUSE_GATE_PRELIM_TOP_K',
    'MM_ROOT_CAUSE_GENERIC_DOWNRANK_PENALTY',
    'MM_ROOT_CAUSE_GENERIC_SUPPORT_ONLY_PENALTY',
    'MM_ROOT_CAUSE_HARD_EXCLUDE_PENALTY',
    'MM_ROOT_CAUSE_INTENT_MODEL',
    'MM_ROOT_CAUSE_MATRIX_MIN_DISTINCT_CAUSES',
    'MM_ROOT_CAUSE_MATRIX_PROMPT_CAUSE_QUOTA',
    'MM_ROOT_CAUSE_MAX_EVIDENCE_POOL',
    'MM_ROOT_CAUSE_MAX_PROMPT_CITATIONS',
    'MM_ROOT_CAUSE_RESPONSE_MODEL',
    'MM_ROOT_CAUSE_SET_LOCK_DELTA',
    'MM_ROOT_CAUSE_SKIP_CANDIDATE_IF_BASELINE_PROXY_GTE',
    'MM_ROOT_CAUSE_USE_DETERMINISTIC_CROSSLINGUAL',
    'MM_SEMANTIC_EXACT_MACHINE_BONUS',
    'MM_SEMANTIC_MAX_DENSE_QUERIES',
    'MM_SEMANTIC_MAX_LEXICAL_QUERIES',
    'MM_SEMANTIC_QUERY_PLANNER_MODEL',
    'MM_SEMANTIC_QUERY_PLANNER_TIMEOUT_SECONDS',
    'MM_SMART_DIAGNOSTIC_ENABLED',
    'MM_SMART_DIAGNOSTIC_EVIDENCE_GATE_MIN_CONFIDENCE',
    'MM_SMART_DIAGNOSTIC_EVIDENCE_GATE_MODEL',
    'MM_SMART_DIAGNOSTIC_EVIDENCE_GATE_TIMEOUT_SECONDS',
    'MM_SMART_DIAGNOSTIC_FINAL_SOURCE_LIMIT',
    'MM_SMART_DIAGNOSTIC_LLM_TIMEOUT_SECONDS',
    'MM_SMART_DIAGNOSTIC_MAX_CONTEXT_CHARS',
    'MM_SMART_DIAGNOSTIC_MAX_EVIDENCE_IN_STATE',
    'MM_SMART_DIAGNOSTIC_MAX_HYPOTHESES',
    'MM_SMART_DIAGNOSTIC_MAX_QUESTIONS',
    'MM_SMART_DIAGNOSTIC_MODEL',
    'MM_SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_ENABLED',
    'MM_SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_MAX_NEW_EVIDENCE',
    'MM_SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_MAX_SECONDS_ANSWER',
    'MM_SMART_DIAGNOSTIC_RETRIEVAL_ASSURANCE_MAX_SECONDS_START',
    'MM_SMART_DIAGNOSTIC_TOP_K',
    'MM_STRUCTURED_RESCUE_ENABLED',
    'MM_STRUCTURED_RESCUE_MAX_HITS',
    'MM_STRUCTURED_RESCUE_SCAN_LIMIT',
    'MM_V13_ASK_DEADLINE_SECONDS',
    'MM_V13_ASK_ENABLED',
    'MM_V13_ASK_HEAVY_EFFORT',
    'MM_V13_DB_CONNECT_TIMEOUT_SECONDS',
    'MM_V13_DB_STATEMENT_TIMEOUT_MS',
    'MM_V13_DENSE_QUERY_LIMIT',
    'MM_V13_ENABLED',
    'MM_V13_EVIDENCE_CLEAR_REJECT_SIM',
    'MM_V13_EVIDENCE_CLEAR_SUPPORT_SIM',
    'MM_V13_EVIDENCE_GATE_EFFORT',
    'MM_V13_EVIDENCE_GATE_MAX_CANDIDATES',
    'MM_V13_EVIDENCE_GATE_MAX_OUTPUT_TOKENS',
    'MM_V13_EVIDENCE_GATE_MIN_CONFIDENCE',
    'MM_V13_EVIDENCE_GATE_MODEL',
    'MM_V13_EVIDENCE_GATE_TIMEOUT_SECONDS',
    'MM_V13_EVIDENCE_MIN_OVERLAP',
    'MM_V13_EVIDENCE_SUPPORT_SIM_WITH_OVERLAP',
    'MM_V13_FAST_CONTEXT_CHARS',
    'MM_V13_FAST_EFFORT',
    'MM_V13_FAST_MAX_OUTPUT_TOKENS',
    'MM_V13_FAST_MODEL',
    'MM_V13_FAST_TIMEOUT_SECONDS',
    'MM_V13_HEAVY_CONTEXT_CHARS',
    'MM_V13_HEAVY_MAX_OUTPUT_TOKENS',
    'MM_V13_HEAVY_MODEL',
    'MM_V13_HEAVY_REASONING_MODE',
    'MM_V13_HEAVY_TIMEOUT_SECONDS',
    'MM_V13_LEXICAL_QUERY_LIMIT',
    'MM_V13_MAX_ESTIMATED_COST_ASK_USD',
    'MM_V13_MAX_ESTIMATED_COST_ROOT_CAUSE_USD',
    'MM_V13_MAX_EVIDENCE_ITEMS_ASK',
    'MM_V13_MAX_EVIDENCE_ITEMS_ROOT_CAUSE',
    'MM_V13_MAX_LLM_CALLS_ASK',
    'MM_V13_MAX_LLM_CALLS_ROOT_CAUSE',
    'MM_V13_MIN_SECONDS_FOR_REFINEMENT',
    'MM_V13_PAGE_SCAN_LIMIT',
    'MM_V13_PAGE_TEXT_CHARS',
    'MM_V13_PLANNER_MAX_OUTPUT_TOKENS',
    'MM_V13_PLANNER_MODEL',
    'MM_V13_PLANNER_TIMEOUT_SECONDS',
    'MM_V13_PREFERRED_PAGE_SCAN_LIMIT',
    'MM_V13_PRICE_EMBED_INPUT',
    'MM_V13_PRICE_LUNA_INPUT',
    'MM_V13_PRICE_LUNA_OUTPUT',
    'MM_V13_PRICE_SOL_INPUT',
    'MM_V13_PRICE_SOL_OUTPUT',
    'MM_V13_PRICE_TERRA_INPUT',
    'MM_V13_PRICE_TERRA_OUTPUT',
    'MM_V13_RELEASE_ID',
    'MM_V13_RETRIEVAL_ASSURANCE_ENABLED',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_CANDIDATES',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_DENSE_QUERIES',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_DOCS',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_FACETS',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_LEXICAL_QUERIES',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_NEIGHBOR_PAGES',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ASK',
    'MM_V13_RETRIEVAL_ASSURANCE_MAX_SECONDS_ROOT_CAUSE',
    'MM_V13_RETRIEVAL_ASSURANCE_MIN_COVERAGE_GAIN',
    'MM_V13_RETRIEVAL_ASSURANCE_MIN_FACET_GAIN',
    'MM_V13_RETRIEVAL_ASSURANCE_MIN_NEW_OVERLAP',
    'MM_V13_RETRIEVAL_ASSURANCE_MIN_NEW_SEMANTIC_SIM',
    'MM_V13_RETRIEVAL_ASSURANCE_MIN_SUPPORT_GAIN',
    'MM_V13_RETRIEVAL_ASSURANCE_PAGE_RADIUS',
    'MM_V13_RETRIEVAL_ASSURANCE_PRE_GATE_MAX_SECONDS',
    'MM_V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ASK',
    'MM_V13_RETRIEVAL_ASSURANCE_RESERVE_FINAL_SECONDS_ROOT_CAUSE',
    'MM_V13_ROOT_CAUSE_DEADLINE_SECONDS',
    'MM_V13_ROOT_CAUSE_ENABLED',
    'MM_V13_ROOT_HEAVY_EFFORT',
    'MM_V13_SEMANTIC_CACHE_AUTO_DDL',
    'MM_V13_SEMANTIC_CACHE_BOOTSTRAP_RETRY_SECONDS',
    'MM_V13_SEMANTIC_CACHE_ENABLED',
    'MM_V13_SEMANTIC_CACHE_MAX_ROWS_PER_COMPANY',
    'MM_V13_SEMANTIC_CACHE_MIN_QUALITY',
    'MM_V13_SEMANTIC_CACHE_SCAN_LIMIT',
    'MM_V13_SEMANTIC_CACHE_THRESHOLD_ASK',
    'MM_V13_SEMANTIC_CACHE_THRESHOLD_ROOT_CAUSE',
    'MM_V13_SEMANTIC_CACHE_TTL_SECONDS',
    'MM_V13_SOURCE_RETRIEVAL_AMBIGUITY_DELTA',
    'MM_V13_SOURCE_RETRIEVAL_ENABLED',
    'MM_V13_SOURCE_RETRIEVAL_FORCE_GATE_SCORE',
    'MM_V13_SOURCE_RETRIEVAL_FORCE_SEMANTIC_SCORE',
    'MM_V13_SOURCE_RETRIEVAL_MAX_CANDIDATES',
    'MM_V13_SOURCE_RETRIEVAL_MAX_QUERY_TOKENS',
    'MM_V13_SOURCE_RETRIEVAL_MAX_RESULTS_FEW',
    'MM_V13_SOURCE_RETRIEVAL_MAX_RESULTS_MANY',
    'MM_V13_SOURCE_RETRIEVAL_MIN_FOCUS_SCORE',
    'MM_V13_SOURCE_RETRIEVAL_MIN_SEMANTIC_SCORE',
    'MM_V13_SOURCE_RETRIEVAL_MIN_TASK_CONFIDENCE',
    'MM_V13_SOURCE_RETRIEVAL_MIN_TITLE_SCORE',
    'MM_V13_SOURCE_RETRIEVAL_PREFERENCE_MAX_GAP',
    'MM_V13_SOURCE_RETRIEVAL_REQUIRE_TYPE_CONFIDENCE',
    'MM_V13_SOURCE_RETRIEVAL_RESULT_BAND',
    'MM_V13_SOURCE_RETRIEVAL_SCAN_LIMIT',
    'MM_V13_STREAM_HEARTBEAT_BYTES',
    'MM_V13_STREAM_HEARTBEAT_ENABLED',
    'MM_V13_STREAM_HEARTBEAT_SECONDS',
    'MM_V8_SHADOW_ASK_ENABLED',
    'MM_V8_SHADOW_MAX_CAUSES',
    'MM_V8_SHADOW_MAX_CITATIONS',
    'MM_V8_SHADOW_MIN_ASK_PROXY',
    'MM_V8_SHADOW_MIN_ROOT_PROXY',
    'MM_V8_SHADOW_MODEL',
    'MM_V8_SHADOW_REASONING_ENABLED',
    'MM_V8_SHADOW_ROOT_CAUSE_ENABLED',
    'MM_V8_SHADOW_TIMEOUT_SECONDS',
    'MM_XLSX_INCLUDE_HIDDEN_SHEETS',
    'MM_XLSX_INGEST_ENABLED',
    'MM_XLSX_MAX_CELLS_TOTAL',
    'MM_XLSX_MAX_CELL_CHARS',
    'MM_XLSX_MAX_COLS_PER_SHEET',
    'MM_XLSX_MAX_ROWS_PER_SHEET',
    'MM_XLSX_MAX_ROW_CHARS',
    'MM_XLSX_MAX_SHEETS',
    'MM_XLSX_MAX_TEXT_CHARS',
    'MM_XLSX_MIN_TEXT_CHARS',
    'MM_XLSX_PAGE_TARGET_CHARS',
    'OPENAI_API_KEY',
    'OPENAI_CHAT_MODEL',
    'OPENAI_CHAT_URL',
    'OPENAI_EMBED_MODEL',
    'OPENAI_EMBED_URL',
    'OPENAI_RERANK_MODEL',
    'OPENAI_RESPONSES_URL',
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalized_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [normalized_json(v) for v in value]
    return value


def run_json_probe(script: Path, *, overrides: dict[str, str] | None = None, clean_config: bool = False) -> Any:
    env = dict(os.environ)
    if clean_config:
        for key in CONFIG_ENV_KEYS:
            env.pop(key, None)
    if overrides:
        env.update({str(k): str(v) for k, v in overrides.items()})
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{script.name} failed with exit {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def top_definition_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    output: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
            dumped = ast.dump(node, include_attributes=False)
            output[key] = sha256_bytes(dumped.encode("utf-8"))
    return output


def assert_syntax() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        try:
            compile(path.read_text("utf-8"), str(path), "exec")
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc!r}")
    if failures:
        raise AssertionError("Python syntax failures:\n" + "\n".join(failures))


def assert_static_structure() -> None:
    manifest = json.loads((TESTS / "phase2a_parent_structure.json").read_text("utf-8"))
    actual = top_definition_hashes((ROOT / "main.py").read_text("utf-8"))
    expected = dict(manifest["parent_definition_ast_sha256"])
    allowed = set(manifest["allowed_changed_definitions"])

    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(f"Top-level definition inventory changed: missing={missing}, extra={extra}")

    changed = {key for key in expected if actual[key] != expected[key]}
    if changed != allowed:
        raise AssertionError(
            "Unexpected main.py definition changes. "
            f"Expected only {sorted(allowed)}, got {sorted(changed)}"
        )

    expected_candidate = str(manifest["candidate_main_sha256"])
    actual_candidate = sha256_file(ROOT / "main.py")
    if actual_candidate != expected_candidate:
        raise AssertionError(
            f"main.py differs from the reviewed Phase 2A candidate: {actual_candidate}"
        )


def assert_config_extraction() -> None:
    manifest = json.loads((TESTS / "phase2a_config_manifest.json").read_text("utf-8"))
    for name, item in manifest.items():
        path = ROOT / item["module"]
        text = path.read_text("utf-8")
        first_line = str(item["first_line"])
        start = text.find(first_line)
        if start < 0:
            raise AssertionError(f"{name}: extracted source start not found in {item['module']}")
        extracted = text[start:]
        actual_hash = sha256_bytes(extracted.encode("utf-8"))
        if actual_hash != item["source_sha256"]:
            raise AssertionError(
                f"{name}: configuration assignment block changed: {actual_hash}"
            )

        module_tree = ast.parse(text)
        exports: list[str] | None = None
        for node in module_tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    exports = ast.literal_eval(node.value)
                    break
        if exports != item["exports"]:
            raise AssertionError(f"{name}: __all__ does not match the frozen export inventory")


def assert_assistant_core_unchanged() -> None:
    expected = "dca19aba41becaffb7c0623f52dee22863c527dbbb7dc8ee965a724a25efd00d"
    actual = sha256_file(ROOT / "assistant_core_v2.py")
    if actual != expected:
        raise AssertionError(f"assistant_core_v2.py changed unexpectedly: {actual}")


def assert_runtime_contracts() -> None:
    expected_contract = json.loads((TESTS / "phase2a_expected_contract.json").read_text("utf-8"))
    actual_contract = run_json_probe(TOOLS / "contract_probe.py")
    if normalized_json(actual_contract) != normalized_json(expected_contract):
        raise AssertionError("API/request/scope/marker runtime contract differs from Phase 1")

    expected_database = json.loads(
        (TESTS / "phase2a_expected_database_contract.json").read_text("utf-8")
    )
    actual_database = run_json_probe(TOOLS / "database_contract_probe.py")
    if normalized_json(actual_database) != normalized_json(expected_database):
        raise AssertionError("Database connector or SQL utility behavior differs from Phase 1")

    expected_default = json.loads(
        (TESTS / "phase2a_expected_config_default.json").read_text("utf-8")
    )
    actual_default = run_json_probe(TOOLS / "config_probe.py", clean_config=True)
    if normalized_json(actual_default) != normalized_json(expected_default):
        raise AssertionError("Default runtime configuration differs from Phase 1")

    scenarios = json.loads((TESTS / "phase2a_config_scenarios.json").read_text("utf-8"))
    for index, scenario in enumerate(scenarios, start=1):
        actual = run_json_probe(
            TOOLS / "config_probe.py",
            overrides=scenario["environment"],
            clean_config=True,
        )
        if normalized_json(actual) != normalized_json(scenario["expected"]):
            raise AssertionError(
                f"Runtime configuration parity failed for scenario {index}"
            )


def main() -> int:
    checks = [
        ("syntax", assert_syntax),
        ("assistant_core_unchanged", assert_assistant_core_unchanged),
        ("static_structure", assert_static_structure),
        ("configuration_extraction", assert_config_extraction),
        ("runtime_contracts", assert_runtime_contracts),
    ]
    completed: list[str] = []
    for name, check in checks:
        check()
        completed.append(name)
        print(f"PASS: {name}")
    print(
        "PHASE 2A OFFLINE GATE: PASS — "
        "configuration and DB connection infrastructure were extracted without "
        "changing the frozen runtime contracts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
