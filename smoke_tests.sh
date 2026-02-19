#!/usr/bin/env bash
set -euo pipefail

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
MAX_ATTEMPTS=6

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN env var is required."
  echo "Example: export TOKEN=\"v1_internal_...\""
  exit 2
fi

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command '$1'"; exit 2; }; }
need_cmd curl
need_cmd python3

build_payload() {
  # Args: company machine doc query
  python3 - <<PY
import json
company="${1}"
machine="${2}"
doc="${3}"
query="${4}"
token="${TOKEN}"
top_k=int("${TOP_K}")
print(json.dumps({
  "auth_token": token,
  "company_id": company,
  "machine_id": machine,
  "document_ids": doc,
  "query": query,
  "options": {"top_k": top_k},
}, ensure_ascii=False))
PY
}

parse_status() {
  # Reads any text from stdin, tries to extract the first JSON object and print .status
  python3 - <<'PY'
import json,sys
raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")

# Robust extraction: take substring from first '{' to last '}'.
# This tolerates banners/HTML/extra text before/after JSON.
i = raw.find("{")
j = raw.rfind("}")
if i == -1 or j == -1 or j <= i:
    print("PARSE_ERROR")
    print("RAW_HEAD:", raw[:200].replace("\n","\\n"))
    sys.exit(0)

candidate = raw[i:j+1]
try:
    obj = json.loads(candidate)
    print(obj.get("status",""))
except Exception:
    print("PARSE_ERROR")
    print("RAW_HEAD:", raw[:200].replace("\n","\\n"))
PY
}

ask_status() {
  local company="$1"
  local machine="$2"
  local doc="$3"
  local query="$4"

  local payload
  payload="$(build_payload "$company" "$machine" "$doc" "$query")"

  local resp=""
  local attempt=1
  local sleep_s=1

  while [[ $attempt -le $MAX_ATTEMPTS ]]; do
    set +e
    resp="$(curl --max-time 60 --connect-timeout 10 -sS -X POST "$WORKER_URL" \
      -H "Content-Type: application/json; charset=utf-8" \
      --data-binary "$payload")"
    local rc=$?
    set -e

    if [[ $rc -eq 0 && -n "$resp" ]]; then
      # Parse whatever arrived (even if not pure JSON)
      printf '%s' "$resp" | parse_status
      return 0
    fi

    if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
      sleep "$sleep_s"
      sleep_s=$((sleep_s * 2))
      if [[ $sleep_s -gt 16 ]]; then sleep_s=16; fi
    fi
    attempt=$((attempt + 1))
  done

  # After retries: fail clearly (network/empty)
  echo "NETWORK_ERROR"
  echo "RAW_HEAD: ${resp:0:200}"
  return 0
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
