#!/usr/bin/env bash
set -euo pipefail

# MachineMind AI — Smoke tests (multi-tenant anti-leak + sentinel/code fallback)
# Usage:
#   export TOKEN="v1_internal_..."
#   bash smoke_tests.sh
#
# Expected:
#   - A sees A  -> answered
#   - B sees B  -> answered
#   - A sees B  -> no_sources
#   - B sees A  -> no_sources

WORKER_URL="https://mm-ai-mock.square-sunset-7388.workers.dev/v1/ai/ask"

# ---- IDs (DEV) ----
COMPANY_A="1766334551970x848077659418132500"
MACHINE_A="1766967703577x651809929304997900"
DOC_A="1771493804876x415817609722462200"
SENT_A="SENTINEL_A_9Q7X3"

COMPANY_B="1768570969798x571640985075318800"
MACHINE_B="1771493897062x931194232803426300"
DOC_B="1771494052788x533049530417938400"
SENT_B="SENTINEL_B_K2M8P"

TOP_K=5

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN env var is required."
  echo "Example: export TOKEN=\"v1_internal_...\""
  exit 2
fi

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command '$1'"; exit 2; }; }
need_cmd curl
need_cmd python3

ask_status() {
  local company="$1"
  local machine="$2"
  local doc="$3"
  local query="$4"

  local resp
  resp="$(curl --max-time 60 --connect-timeout 10 -sS -X POST "$WORKER_URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"auth_token\":\"$TOKEN\",
      \"company_id\":\"$company\",
      \"machine_id\":\"$machine\",
      \"document_ids\":\"$doc\",
      \"query\":\"$query\",
      \"options\":{\"top_k\":$TOP_K}
    }" || true)"

  python3 - <<PY
import json
raw = """$resp"""
try:
  j = json.loads(raw)
  print(j.get("status",""))
except Exception:
  print("PARSE_ERROR")
  print(raw[:400])
PY
}

assert_eq() {
  local name="$1"
  local got="$2"
  local expected="$3"
  if [[ "$got" != "$expected" ]]; then
    echo "❌ $name: expected '$expected' but got '$got'"
    exit 1
  fi
  echo "✅ $name: $got"
}

echo "Running smoke tests against: $WORKER_URL"
echo

S1="$(ask_status "$COMPANY_A" "$MACHINE_A" "$DOC_A" "Trova la stringa $SENT_A nel documento.")"
assert_eq "A sees A" "$S1" "answered"

S2="$(ask_status "$COMPANY_B" "$MACHINE_B" "$DOC_B" "Trova la stringa $SENT_B nel documento.")"
assert_eq "B sees B" "$S2" "answered"

S3="$(ask_status "$COMPANY_A" "$MACHINE_A" "$DOC_B" "Trova la stringa $SENT_B nel documento.")"
assert_eq "A sees B (should block)" "$S3" "no_sources"

S4="$(ask_status "$COMPANY_B" "$MACHINE_B" "$DOC_A" "Trova la stringa $SENT_A nel documento.")"
assert_eq "B sees A (should block)" "$S4" "no_sources"

echo
echo "✅ All smoke tests passed."
