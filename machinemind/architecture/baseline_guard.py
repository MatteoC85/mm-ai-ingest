"""Static migration guard for the frozen production baseline and phase metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path
from machinemind.config.baseline import (
    PRODUCTION_BASELINE_ASSISTANT_CORE_SHA256,
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_assistant_core_unchanged(repository_root: str | Path) -> None:
    root = Path(repository_root)
    actual = sha256_file(root / "assistant_core_v2.py")
    if actual != PRODUCTION_BASELINE_ASSISTANT_CORE_SHA256:
        raise RuntimeError(
            "assistant_core_v2.py differs from the frozen production baseline: "
            f"{actual}"
        )
