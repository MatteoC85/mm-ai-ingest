#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 3 — Commit 1 (citation presentation)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_phase2a_gate as phase2a  # noqa: E402
import run_phase2b_gate as phase2b  # noqa: E402
import run_phase2c_gate as phase2c  # noqa: E402
import run_phase2d_gate as phase2d  # noqa: E402
import run_phase2e_gate as phase2e  # noqa: E402


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
            str(key): normalized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [normalized_json(item) for item in value]
    return value


def assert_prior_behavior_contracts() -> None:
    # Prior static main.py manifests are intentionally not reused because this commit
    # changes only the frozen presentation adapters. Every cumulative behavioral
    # contract is re-executed against the candidate instead.
    phase2a.assert_assistant_core_unchanged()
    phase2a.assert_config_extraction()
    phase2a.assert_runtime_contracts()
    phase2b.assert_transport_boundary()
    phase2b.assert_openai_contract()
    phase2c.assert_budget_execution_contract()
    phase2d.assert_semantic_cache_contract()
    phase2e.assert_external_io_contract()


def assert_static_extraction() -> None:
    manifest = json.loads(
        (TESTS / "phase3a_structure_manifest.json").read_text("utf-8")
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
        raise AssertionError(
            "Unexpected top-level definition inventory: "
            f"missing={sorted(expected_inventory - set(actual))}, "
            f"extra={sorted(set(actual) - expected_inventory)}"
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
        raise AssertionError("main.py differs from the reviewed Phase 3A candidate")

    main_text = main_path.read_text("utf-8")
    for marker in manifest["required_main_markers"]:
        if marker not in main_text:
            raise AssertionError(f"Missing presentation adapter marker: {marker}")
    for marker in manifest["forbidden_main_markers"]:
        if marker in main_text:
            raise AssertionError(
                f"Presentation implementation remains duplicated in main.py: {marker}"
            )

    for relative, item in manifest["modules"].items():
        module_path = ROOT / relative
        if not module_path.is_file():
            raise AssertionError(f"Extracted module missing: {relative}")
        module_text = module_path.read_text("utf-8")
        if "import main" in module_text or "from main import" in module_text:
            raise AssertionError(f"Extracted module imports composition root: {relative}")
        tree = ast.parse(module_text)
        forbidden_roots = {"psycopg2", "requests", "fastapi", "google", "openai"}
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        if imported_roots & forbidden_roots:
            raise AssertionError(
                f"Presentation module has infrastructure imports: {sorted(imported_roots & forbidden_roots)}"
            )
        actual_symbols = module_symbols(module_path)
        required_symbols = set(item["required_symbols"])
        if actual_symbols != required_symbols:
            raise AssertionError(
                f"{relative} symbol inventory differs: "
                f"missing={sorted(required_symbols - actual_symbols)}, "
                f"extra={sorted(actual_symbols - required_symbols)}"
            )
        if sha256_file(module_path) != str(item["sha256"]):
            raise AssertionError(f"{relative} differs from the reviewed module")

    for relative, expected_sha in manifest["unchanged_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise AssertionError(f"Unexpected change to frozen file: {relative}")


def run_presentation_probe(root: Path) -> dict:
    env = os.environ.copy()
    for key in phase2a.CONFIG_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "tools/presentation_contract_probe.py"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Presentation contract probe failed: "
            f"stdout={proc.stdout[-2000:]!r} stderr={proc.stderr[-4000:]!r}"
        )
    return json.loads(proc.stdout)


def assert_presentation_contract() -> None:
    expected = json.loads(
        (TESTS / "phase3a_expected_presentation_contract.json").read_text("utf-8")
    )
    actual = run_presentation_probe(ROOT)
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "Citation labels, structured/XLSX snippets, resource links, source blocks, "
            "error envelopes or late-bound helper behavior differs from the Phase 2 "
            "Commit 5 parent"
        )


def assert_mutation_sensitivity() -> None:
    expected = json.loads(
        (TESTS / "phase3a_expected_presentation_contract.json").read_text("utf-8")
    )
    mutations = [
        (
            "document_page_fragment_changed",
            'else f"{base_url}#page={page_from}"',
            'else f"{base_url}#p={page_from}"',
        ),
        (
            "xlsx_sheet_label_changed",
            'lines.append(f"Foglio: {sheet}")',
            'lines.append(f"Sheet: {sheet}")',
        ),
        (
            "source_block_boundary_changed",
            "if total_chars + len(part) > max_context_chars:",
            "if total_chars + len(part) >= max_context_chars:",
        ),
    ]
    source_rel = Path("machinemind/presentation/citations.py")
    for name, before, after in mutations:
        with tempfile.TemporaryDirectory(prefix=f"mm-{name}-") as temp_dir:
            temp_root = Path(temp_dir) / "repo"
            shutil.copytree(ROOT, temp_root)
            module_path = temp_root / source_rel
            text = module_path.read_text("utf-8")
            if text.count(before) != 1:
                raise AssertionError(
                    f"Mutation marker is not unique for {name}: {text.count(before)}"
                )
            module_path.write_text(text.replace(before, after, 1), "utf-8")
            actual = run_presentation_probe(temp_root)
            if normalized_json(actual) == normalized_json(expected):
                raise AssertionError(
                    f"Presentation contract failed to detect intentional mutation: {name}"
                )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("prior_behavior_contracts", assert_prior_behavior_contracts),
        ("static_extraction", assert_static_extraction),
        ("presentation_contract", assert_presentation_contract),
        ("mutation_sensitivity", assert_mutation_sensitivity),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 3A OFFLINE GATE: PASS — citation metadata, structured/XLSX display "
        "sanitization, prompt source blocks and Bubble resource links were extracted "
        "without changing the frozen Phase 2 Commit 5 behavior covered here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
