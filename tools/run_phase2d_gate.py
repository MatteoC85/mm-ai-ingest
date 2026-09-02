#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 2 — Commit 4 (semantic cache boundary)."""
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
import run_phase2c_gate as phase2c  # noqa: E402


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
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


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
    # Re-run every cumulative non-structural contract. Previous static manifests are
    # intentionally not reused because this commit legitimately changes main.py.
    phase2a.assert_assistant_core_unchanged()
    phase2a.assert_config_extraction()
    phase2a.assert_runtime_contracts()
    phase2b.assert_transport_boundary()
    phase2b.assert_openai_contract()
    phase2c.assert_budget_execution_contract()


def assert_static_extraction() -> None:
    manifest = json.loads(
        (TESTS / "phase2d_structure_manifest.json").read_text("utf-8")
    )
    main_path = ROOT / "main.py"
    actual = top_definition_hashes(main_path.read_text("utf-8"))
    parent = dict(manifest["parent_definition_ast_sha256"])

    if set(actual) != set(parent):
        missing = sorted(set(parent) - set(actual))
        extra = sorted(set(actual) - set(parent))
        raise AssertionError(
            f"Unexpected top-level definition inventory: missing={missing}, extra={extra}"
        )

    changed = {key for key in parent if actual[key] != parent[key]}
    allowed = set(manifest["allowed_changed_definitions"])
    if changed != allowed:
        raise AssertionError(
            "Unexpected main.py behavior-surface changes. "
            f"Expected {sorted(allowed)}, got {sorted(changed)}"
        )

    if sha256_file(main_path) != str(manifest["candidate_main_sha256"]):
        raise AssertionError("main.py differs from the reviewed Phase 2D candidate")

    main_text = main_path.read_text("utf-8")
    for marker in manifest["required_main_markers"]:
        if marker not in main_text:
            raise AssertionError(f"Missing semantic-cache adapter marker: {marker}")
    for marker in manifest["forbidden_main_markers"]:
        if marker in main_text:
            raise AssertionError(
                f"Semantic-cache implementation/SQL remains duplicated in main.py: {marker}"
            )

    module_path = ROOT / str(manifest["required_module"])
    if not module_path.is_file():
        raise AssertionError("Extracted semantic-cache module missing")
    module_text = module_path.read_text("utf-8")
    if "import main" in module_text or "from main import" in module_text:
        raise AssertionError("Semantic-cache module imports the composition root")
    missing_symbols = sorted(
        set(manifest["required_module_symbols"]) - module_symbols(module_path)
    )
    if missing_symbols:
        raise AssertionError(
            f"Semantic-cache module missing symbols: {missing_symbols}"
        )
    if sha256_file(module_path) != str(manifest["module_sha256"]):
        raise AssertionError("semantic_cache.py differs from the reviewed module")

    for relative, expected_sha in manifest["unchanged_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise AssertionError(f"Unexpected change to frozen file: {relative}")


def assert_semantic_cache_contract() -> None:
    expected = json.loads(
        (TESTS / "phase2d_expected_semantic_cache_contract.json").read_text("utf-8")
    )
    actual = phase2a.run_json_probe(
        TOOLS / "semantic_cache_contract_probe.py",
        clean_config=True,
    )
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "Semantic-cache guards, SQL, bootstrap state, knowledge versioning, "
            "lookup/store behavior or fail-open policy differs from the Commit 3 parent"
        )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("prior_behavior_contracts", assert_prior_behavior_contracts),
        ("static_extraction", assert_static_extraction),
        ("semantic_cache_contract", assert_semantic_cache_contract),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 2D OFFLINE GATE: PASS — semantic response caching, compatibility "
        "guards, company knowledge versioning, SQL persistence and fail-open behavior "
        "were extracted without changing the frozen Commit 3 contract covered here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
