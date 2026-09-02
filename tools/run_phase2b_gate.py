#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 2 — Commit 2 (OpenAI transport)."""
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_definition_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    output: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
            dumped = ast.dump(node, include_attributes=False)
            output[key] = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return output


def normalized_json(value):
    if isinstance(value, dict):
        return {str(k): normalized_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [normalized_json(v) for v in value]
    return value


def assert_parent_phase_contracts() -> None:
    # Re-run every non-structural Phase 2A gate. Static structure has intentionally
    # changed in this commit and is checked against the Phase 2A parent below.
    phase2a.assert_assistant_core_unchanged()
    phase2a.assert_config_extraction()
    phase2a.assert_runtime_contracts()


def assert_static_structure() -> None:
    manifest = json.loads((TESTS / "phase2b_parent_structure.json").read_text("utf-8"))
    actual = top_definition_hashes((ROOT / "main.py").read_text("utf-8"))
    expected = dict(manifest["parent_definition_ast_sha256"])
    allowed = set(manifest["allowed_changed_definitions"])

    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise AssertionError(
            f"Top-level definition inventory changed: missing={missing}, extra={extra}"
        )

    changed = {key for key in expected if actual[key] != expected[key]}
    if changed != allowed:
        raise AssertionError(
            "Unexpected main.py definition changes relative to Phase 2A. "
            f"Expected only {sorted(allowed)}, got {sorted(changed)}"
        )

    expected_candidate = str(manifest["candidate_main_sha256"])
    actual_candidate = sha256_file(ROOT / "main.py")
    if actual_candidate != expected_candidate:
        raise AssertionError(
            f"main.py differs from the reviewed Phase 2B candidate: {actual_candidate}"
        )


def assert_transport_boundary() -> None:
    path = ROOT / "machinemind" / "infrastructure" / "openai_transport.py"
    if not path.is_file():
        raise AssertionError("OpenAI transport module missing")
    text = path.read_text("utf-8")
    tree = ast.parse(text)

    expected = {
        "normalize_model_candidates",
        "safety_identifier",
        "response_text",
        "embed_texts",
        "chat_text",
        "chat_json",
        "chat_json_models",
        "responses_json",
        "json_models",
    }
    actual = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if actual != expected:
        raise AssertionError(
            f"OpenAI transport function inventory differs: expected={sorted(expected)}, actual={sorted(actual)}"
        )

    forbidden = {"main"}
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    if imports & forbidden:
        raise AssertionError("OpenAI transport must not import the composition root")

    main_text = (ROOT / "main.py").read_text("utf-8")
    if "from machinemind.infrastructure import openai_transport as _openai_transport" not in main_text:
        raise AssertionError("main.py does not import the extracted transport boundary")


def assert_openai_contract() -> None:
    expected = json.loads(
        (TESTS / "phase2b_expected_openai_contract.json").read_text("utf-8")
    )
    actual = phase2a.run_json_probe(TOOLS / "openai_contract_probe.py")
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "OpenAI request payload, output, exception, cache, metering or fallback behavior differs from Phase 2A"
        )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("parent_phase_contracts", assert_parent_phase_contracts),
        ("static_structure", assert_static_structure),
        ("transport_boundary", assert_transport_boundary),
        ("openai_differential_contract", assert_openai_contract),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 2B OFFLINE GATE: PASS — OpenAI embeddings, legacy Chat Completions, "
        "Responses structured output and model-fallback orchestration were extracted "
        "without changing the frozen provider/runtime contract."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
