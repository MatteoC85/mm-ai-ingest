#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 2 — Commit 3.

This gate is intentionally scoped to behavior-preserving extraction of request budget,
cost accounting and generic execution guards. It re-runs the prior configuration,
database and OpenAI provider contracts, then verifies the new boundary against the
Phase 2 Commit 2 parent.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_phase2a_gate as phase2a  # noqa: E402
import run_phase2b_gate as phase2b  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_definition_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    output: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
            output[key] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")
            ).hexdigest()
    return output


def module_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def normalized_json(value):
    if isinstance(value, dict):
        return {
            str(k): normalized_json(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [normalized_json(v) for v in value]
    return value


def assert_prior_behavior_contracts() -> None:
    # Phase 2A: API/scope/config/database behavior.
    phase2a.assert_assistant_core_unchanged()
    phase2a.assert_config_extraction()
    phase2a.assert_runtime_contracts()
    # Phase 2B: OpenAI transport payloads, parsing, errors, cache/metering and fallback.
    phase2b.assert_transport_boundary()
    phase2b.assert_openai_contract()


def assert_static_extraction() -> None:
    manifest = json.loads(
        (TESTS / "phase2c_structure_manifest.json").read_text("utf-8")
    )
    main_path = ROOT / "main.py"
    actual = top_definition_hashes(main_path.read_text("utf-8"))
    parent = dict(manifest["parent_definition_ast_sha256"])
    extracted = set(manifest["extracted_definitions"])
    allowed_changed = set(manifest["allowed_changed_definitions"])
    expected_names = set(parent) - extracted

    if set(actual) != expected_names:
        missing = sorted(expected_names - set(actual))
        extra = sorted(set(actual) - expected_names)
        raise AssertionError(
            f"Unexpected top-level definition inventory: missing={missing}, extra={extra}"
        )

    changed = {
        key for key in expected_names if actual[key] != parent[key]
    }
    if changed != allowed_changed:
        raise AssertionError(
            "Unexpected main.py behavior-surface changes. "
            f"Expected {sorted(allowed_changed)}, got {sorted(changed)}"
        )

    actual_main_sha = sha256_file(main_path)
    if actual_main_sha != str(manifest["candidate_main_sha256"]):
        raise AssertionError(
            f"main.py differs from the reviewed candidate: {actual_main_sha}"
        )

    main_text = main_path.read_text("utf-8")
    markers = [
        "configure_request_budget_runtime as _configure_request_budget_runtime",
        "_configure_request_budget_runtime(globals())",
        "_infra_stream_json_response",
        "_infra_json_with_hard_timeout",
        "_infra_run_sync_with_hard_timeout",
    ]
    for marker in markers:
        if marker not in main_text:
            raise AssertionError(f"Missing compatibility adapter marker: {marker}")

    for relative, required in manifest["required_modules"].items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"Missing extracted module: {relative}")
        text = path.read_text("utf-8")
        if "import main" in text or "from main import" in text:
            raise AssertionError(f"Circular composition-root import in {relative}")
        missing = sorted(set(required) - module_symbols(path))
        if missing:
            raise AssertionError(f"{relative} missing symbols: {missing}")
        expected_sha = str(manifest["module_sha256"][relative])
        if sha256_file(path) != expected_sha:
            raise AssertionError(f"{relative} differs from reviewed module")

    for relative, expected_sha in manifest["unchanged_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise AssertionError(f"Unexpected change to frozen file: {relative}")


def assert_budget_execution_contract() -> None:
    expected = json.loads(
        (TESTS / "phase2c_expected_budget_execution_contract.json").read_text("utf-8")
    )
    actual = phase2a.run_json_probe(
        TOOLS / "budget_execution_contract_probe.py",
        clean_config=True,
    )
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "Request budget, cost accounting, ContextVar or execution-guard behavior "
            "differs from the Phase 2 Commit 2 parent"
        )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("prior_behavior_contracts", assert_prior_behavior_contracts),
        ("static_extraction", assert_static_extraction),
        ("budget_execution_contract", assert_budget_execution_contract),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 2C OFFLINE GATE: PASS — request budgets, cost accounting, "
        "ContextVar propagation and generic hard-timeout/streaming execution guards "
        "were extracted without changing the frozen parent behavior covered by this gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
