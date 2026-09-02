#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 2 — Commit 5 (external I/O boundary)."""
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
import run_phase2d_gate as phase2d  # noqa: E402


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
    # Re-run every cumulative behavioral contract. Prior static main.py manifests are
    # intentionally not reused because this commit legitimately changes main.py.
    phase2a.assert_assistant_core_unchanged()
    phase2a.assert_config_extraction()
    phase2a.assert_runtime_contracts()
    phase2b.assert_transport_boundary()
    phase2b.assert_openai_contract()
    phase2c.assert_budget_execution_contract()
    phase2d.assert_semantic_cache_contract()


def assert_static_extraction() -> None:
    manifest = json.loads(
        (TESTS / "phase2e_structure_manifest.json").read_text("utf-8")
    )
    main_path = ROOT / "main.py"
    actual = top_definition_hashes(main_path.read_text("utf-8"))
    parent = dict(manifest["parent_definition_ast_sha256"])

    expected_inventory = (
        set(parent)
        - set(manifest["allowed_removed_definitions"])
        | set(manifest["allowed_added_definitions"])
    )
    if set(actual) != expected_inventory:
        missing = sorted(expected_inventory - set(actual))
        extra = sorted(set(actual) - expected_inventory)
        raise AssertionError(
            f"Unexpected top-level definition inventory: missing={missing}, extra={extra}"
        )

    removed = set(parent) - set(actual)
    if removed != set(manifest["allowed_removed_definitions"]):
        raise AssertionError(
            f"Unexpected removed definitions: {sorted(removed)}"
        )

    changed = {
        key for key in set(parent) & set(actual) if actual[key] != parent[key]
    }
    allowed_changed = set(manifest["allowed_changed_definitions"])
    if changed != allowed_changed:
        raise AssertionError(
            "Unexpected main.py behavior-surface changes. "
            f"Expected {sorted(allowed_changed)}, got {sorted(changed)}"
        )

    if sha256_file(main_path) != str(manifest["candidate_main_sha256"]):
        raise AssertionError("main.py differs from the reviewed Phase 2E candidate")

    main_text = main_path.read_text("utf-8")
    for marker in manifest["required_main_markers"]:
        if marker not in main_text:
            raise AssertionError(f"Missing external-I/O adapter marker: {marker}")
    for marker in manifest["forbidden_main_markers"]:
        if marker in main_text:
            raise AssertionError(
                f"External-I/O implementation remains duplicated in main.py: {marker}"
            )

    for relative, item in manifest["modules"].items():
        module_path = ROOT / relative
        if not module_path.is_file():
            raise AssertionError(f"Extracted module missing: {relative}")
        module_text = module_path.read_text("utf-8")
        if "import main" in module_text or "from main import" in module_text:
            raise AssertionError(f"Extracted module imports composition root: {relative}")
        missing_symbols = sorted(
            set(item["required_symbols"]) - module_symbols(module_path)
        )
        if missing_symbols:
            raise AssertionError(
                f"{relative} missing symbols: {missing_symbols}"
            )
        if sha256_file(module_path) != str(item["sha256"]):
            raise AssertionError(f"{relative} differs from the reviewed module")

    for relative, expected_sha in manifest["unchanged_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise AssertionError(f"Unexpected change to frozen file: {relative}")


def assert_external_io_contract() -> None:
    expected = json.loads(
        (TESTS / "phase2e_expected_external_io_contract.json").read_text("utf-8")
    )
    actual = phase2a.run_json_probe(
        TOOLS / "external_io_contract_probe.py",
        clean_config=True,
    )
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "External file loading, helper late-binding, HTTP error envelopes, "
            "ingest persistence sequence, Cloud Tasks environment precedence, task "
            "payload/headers or fail-open behavior differs from the Commit 4 parent"
        )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("prior_behavior_contracts", assert_prior_behavior_contracts),
        ("static_extraction", assert_static_extraction),
        ("external_io_contract", assert_external_io_contract),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 2E OFFLINE GATE: PASS — external document loading and Cloud Tasks "
        "dispatch were extracted without changing the frozen Commit 4 behavior "
        "covered here. This closes the planned Phase 2 infrastructure extraction "
        "offline; exact-commit build, live ingest/task dispatch and application smoke "
        "verification remain required."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
