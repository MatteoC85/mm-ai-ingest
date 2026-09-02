#!/usr/bin/env python3
"""Offline gate for Roadmap Phase 3 — Commit 2 (response/UI finalization)."""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_phase2a_gate as phase2a  # noqa: E402
import run_phase3a_gate as phase3a  # noqa: E402


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


def normalized_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [normalized_json(item) for item in value]
    return value


def json_diff_records(parent: Any, candidate: Any, path: str = "") -> list[dict]:
    out: list[dict] = []
    if type(parent) is not type(candidate):
        return [{"path": path, "parent": parent, "candidate": candidate}]
    if isinstance(parent, dict):
        for key in sorted(set(parent) | set(candidate)):
            child = f"{path}/{key}"
            if key not in parent or key not in candidate:
                out.append({
                    "path": child,
                    "parent": parent.get(key),
                    "candidate": candidate.get(key),
                })
            else:
                out.extend(json_diff_records(parent[key], candidate[key], child))
        return out
    if isinstance(parent, list):
        if len(parent) != len(candidate):
            out.append({
                "path": f"{path}/len",
                "parent": len(parent),
                "candidate": len(candidate),
            })
        for index, (left, right) in enumerate(zip(parent, candidate)):
            out.extend(json_diff_records(left, right, f"{path}/{index}"))
        return out
    if parent != candidate:
        out.append({"path": path, "parent": parent, "candidate": candidate})
    return out


def assert_prior_behavior_contracts() -> None:
    # Re-run every Phase 1/2 behavioral contract plus the complete Phase 3A
    # citation/link characterization. Phase 3A's exact static main.py hash is
    # intentionally not reused because this commit extracts a new boundary.
    phase3a.assert_prior_behavior_contracts()
    phase3a.assert_presentation_contract()


def assert_static_extraction() -> None:
    manifest = json.loads(
        (TESTS / "phase3b_structure_manifest.json").read_text("utf-8")
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
            "Unexpected response/UI behavior-surface changes. "
            f"Expected {sorted(allowed_changed)}, got {sorted(changed)}"
        )

    if sha256_file(main_path) != str(manifest["candidate_main_sha256"]):
        raise AssertionError("main.py differs from the reviewed Phase 3B candidate")

    main_text = main_path.read_text("utf-8")
    for marker in manifest["required_main_markers"]:
        if marker not in main_text:
            raise AssertionError(f"Missing response-presentation adapter marker: {marker}")
    for marker in manifest["forbidden_main_markers"]:
        if marker in main_text:
            raise AssertionError(
                f"Response/UI implementation remains duplicated in main.py: {marker}"
            )

    for relative, item in manifest["modules"].items():
        module_path = ROOT / relative
        if not module_path.is_file():
            raise AssertionError(f"Extracted module missing: {relative}")
        module_text = module_path.read_text("utf-8")
        if "import main" in module_text or "from main import" in module_text:
            raise AssertionError(f"Extracted module imports composition root: {relative}")
        tree = ast.parse(module_text)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        forbidden = set(item.get("forbidden_import_roots") or [])
        if imported_roots & forbidden:
            raise AssertionError(
                f"Presentation module has infrastructure imports: "
                f"{sorted(imported_roots & forbidden)}"
            )
        actual_symbols = module_symbols(module_path)
        required_symbols = set(item["required_symbols"])
        if actual_symbols != required_symbols:
            raise AssertionError(
                f"{relative} symbol inventory differs: "
                f"missing={sorted(required_symbols - actual_symbols)}, "
                f"extra={sorted(actual_symbols - required_symbols)}"
            )
        for marker in item.get("required_markers") or []:
            if marker not in module_text:
                raise AssertionError(f"Missing reviewed module marker in {relative}: {marker}")
        if sha256_file(module_path) != str(item["sha256"]):
            raise AssertionError(f"{relative} differs from the reviewed module")

    for relative, expected_sha in manifest["unchanged_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise AssertionError(f"Unexpected change to frozen file: {relative}")


def run_response_probe(root: Path) -> dict:
    env = os.environ.copy()
    for key in phase2a.CONFIG_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "tools/response_render_contract_probe.py"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Response/UI contract probe failed: "
            f"stdout={proc.stdout[-2500:]!r} stderr={proc.stderr[-5000:]!r}"
        )
    return json.loads(proc.stdout)


def assert_response_contract() -> None:
    expected = json.loads(
        (TESTS / "phase3b_expected_response_contract.json").read_text("utf-8")
    )
    actual = run_response_probe(ROOT)
    if normalized_json(actual) != normalized_json(expected):
        raise AssertionError(
            "Response finalization, HTML safety, canonicality, de-duplication or "
            "structured-procedure rendering differs from the reviewed candidate"
        )

    # Explicitly pin the live defect observed during Phase 3A validation.
    if actual["generic_numbered_restart"]["li_values"] != [1, 2, 3]:
        raise AssertionError("Generic separated ordered lists no longer preserve 1,2,3")
    if actual["lossless_numbered_restart"]["li_values"] != [1, 2, 3]:
        raise AssertionError("Lossless fallback no longer preserves explicit list values")
    if actual["procedure_html"]["li_values"] != [1, 2, 3]:
        raise AssertionError("Structured Procedure HTML no longer preserves Step numbers")

    finalized = actual["finalize_procedure"]
    if finalized["meta"]["answer_html_mode"] != "procedure":
        raise AssertionError("Structured Procedure model is not the active renderer")
    if finalized["answer_html"]["article_kinds"] != ["procedure"]:
        raise AssertionError("Structured Procedure response fell back to generic HTML")
    if finalized["meta"]["answer_html_renderer_fallback_used"]:
        raise AssertionError("Reviewed structured Procedure scenario unexpectedly fell back")
    if finalized["meta"]["answer_html_token_coverage"] < 0.97:
        raise AssertionError("Structured Procedure HTML is not canonically lossless")
    if len(finalized["citations"]) != 3 or len(finalized["rg_links"]) != 3:
        raise AssertionError("Final public envelope de-duplication changed")
    if finalized["answer_html"]["contains_sources_heading"]:
        raise AssertionError("Answer body duplicated Bubble LINK/FONTI sections")

    parent = json.loads(
        (TESTS / "phase3b_parent_response_contract.json").read_text("utf-8")
    )
    expected_diff = json.loads(
        (TESTS / "phase3b_expected_parent_candidate_diff.json").read_text("utf-8")
    )
    actual_diff = json_diff_records(parent, expected)
    if normalized_json(actual_diff) != normalized_json(expected_diff):
        raise AssertionError(
            "Parent/candidate response diff is broader than the reviewed UI-only changes"
        )

    # Everything outside the pinned HTML/Procedure-diff paths remains byte-equivalent.
    if parent["finalize_root_cause"] != expected["finalize_root_cause"]:
        raise AssertionError("Root Cause final envelope changed")
    if parent["dedupe_links"] != expected["dedupe_links"]:
        raise AssertionError("Link de-duplication behavior changed")
    if parent["dedupe_citations"] != expected["dedupe_citations"]:
        raise AssertionError("Citation de-duplication behavior changed")
    if parent["signatures"] != expected["signatures"]:
        raise AssertionError("Historical response helper signatures changed")


def assert_mutation_sensitivity() -> None:
    expected = json.loads(
        (TESTS / "phase3b_expected_response_contract.json").read_text("utf-8")
    )
    mutations = [
        (
            "explicit_list_values_removed",
            """        value_attr = f' value="{number}"' if number is not None else ""\n""",
            """        value_attr = ""\n""",
        ),
        (
            "structured_procedure_renderer_disabled",
            """    is_procedure_model = isinstance(ui_model, dict) and str(ui_model.get('kind') or '').strip().lower() == 'procedure'\n""",
            """    is_procedure_model = False\n""",
        ),
        (
            "html_escaping_disabled",
            """    return html.escape(str(value or ''), quote=True)\n""",
            """    return str(value or '')\n""",
        ),
    ]
    source_rel = Path("machinemind/presentation/responses.py")
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
            actual = run_response_probe(temp_root)
            if normalized_json(actual) == normalized_json(expected):
                raise AssertionError(
                    f"Response/UI contract failed to detect intentional mutation: {name}"
                )


def main() -> int:
    checks = [
        ("syntax", phase2a.assert_syntax),
        ("prior_behavior_contracts", assert_prior_behavior_contracts),
        ("static_extraction", assert_static_extraction),
        ("response_render_contract", assert_response_contract),
        ("mutation_sensitivity", assert_mutation_sensitivity),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(
        "PHASE 3B OFFLINE GATE: PASS — response finalization, safe HTML rendering, "
        "structured Procedure presentation and final link/citation de-duplication "
        "are isolated; the reviewed UI-only numbering correction is pinned without "
        "changing retrieval, reasoning, Root Cause or citation/link behavior."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
