"""Legacy opt-in V8 shadow endpoint configuration, unchanged."""
from __future__ import annotations

import os

from machinemind.config.runtime import DIAGNOSTIC_EVIDENCE_MODEL, OPENAI_CHAT_MODEL

__all__ = [
    "V8_SHADOW_REASONING_ENABLED",
    "V8_SHADOW_ASK_ENABLED",
    "V8_SHADOW_ROOT_CAUSE_ENABLED",
    "V8_SHADOW_MODEL",
    "V8_SHADOW_TIMEOUT",
    "V8_SHADOW_MAX_CITATIONS",
    "V8_SHADOW_MIN_ASK_PROXY",
    "V8_SHADOW_MIN_ROOT_PROXY",
    "V8_SHADOW_MAX_CAUSES",
]

V8_SHADOW_REASONING_ENABLED = (os.environ.get("MM_V8_SHADOW_REASONING_ENABLED") or "1").strip() != "0"
V8_SHADOW_ASK_ENABLED = (os.environ.get("MM_V8_SHADOW_ASK_ENABLED") or "1").strip() != "0"
V8_SHADOW_ROOT_CAUSE_ENABLED = (os.environ.get("MM_V8_SHADOW_ROOT_CAUSE_ENABLED") or "1").strip() != "0"
V8_SHADOW_MODEL = (os.environ.get("MM_V8_SHADOW_MODEL") or DIAGNOSTIC_EVIDENCE_MODEL or OPENAI_CHAT_MODEL).strip()
V8_SHADOW_TIMEOUT = int(os.environ.get("MM_V8_SHADOW_TIMEOUT_SECONDS", "20"))
V8_SHADOW_MAX_CITATIONS = int(os.environ.get("MM_V8_SHADOW_MAX_CITATIONS", "4"))
V8_SHADOW_MIN_ASK_PROXY = float(os.environ.get("MM_V8_SHADOW_MIN_ASK_PROXY", "0.78"))
V8_SHADOW_MIN_ROOT_PROXY = float(os.environ.get("MM_V8_SHADOW_MIN_ROOT_PROXY", "0.86"))
V8_SHADOW_MAX_CAUSES = int(os.environ.get("MM_V8_SHADOW_MAX_CAUSES", "3"))
