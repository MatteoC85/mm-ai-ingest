#!/usr/bin/env bash
set -euo pipefail

# MachineMind AI — Smoke tests (multi-tenant anti-leak + sentinel/code fallback)
# Usage:
#   export TOKEN="v1_internal_..."   # DO NOT commit the token in git
#   ./smoke_tests.sh
#
# Expected outcomes:
#   1) A sees A  -> answered
#   2) B sees B  -> answered
#   3) A sees B  -> no_sources
#   4) B sees A  -> no_sources

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

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command '$1'"; exit 2; }
}
need_cmd curl
need_cmd python3

ask_status() {
  local company="$1"
  local machine="$2"
  local doc="$3"
  local query="$4"

  curl -s -X POST "$WORKER_URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"auth_token\":\"$TOKEN\",
      \"company_id\":\"$company\",
      \"machine_id\":\"$machine\",
      \"document_ids\":\"$doc\",
      \"query\":\"$query\",
      \"options\":{\"top_k\":$TOP_K}
    }" | python3 - <<'PY'
import json,sys
try:
  j=json.load(sys.stdin)
except Exception as e:
  print("PARSE_ERROR")
  sys.exit(0)
print(j.get("status",""))
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

# 1) A sees A -> answered
S1="$(ask_status "$COMPANY_A" "$MACHINE_A" "$DOC_A" "Trova la stringa $SENT_A nel documento.")"
assert_eq "A sees A" "$S1" "answered"

# 2) B sees B -> answered
S2="$(ask_status "$COMPANY_B" "$MACHINE_B" "$DOC_B" "Trova la stringa $SENT_B nel documento.")"
assert_eq "B sees B" "$S2" "answered"

# 3) A sees B -> no_sources (anti-leak)
S3="$(ask_status "$COMPANY_A" "$MACHINE_A" "$DOC_B" "Trova la stringa $SENT_B nel documento.")"
assert_eq "A sees B (should block)" "$S3" "no_sources"

# 4) B sees A -> no_sources (anti-leak)
S4="$(ask_status "$COMPANY_B" "$MACHINE_B" "$DOC_A" "Trova la stringa $SENT_A nel documento.")"
assert_eq "B sees A (should block)" "$S4" "no_sources"

echo
echo "✅ All smoke tests passed."
