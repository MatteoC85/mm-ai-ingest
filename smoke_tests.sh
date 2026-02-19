#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (override via env)
# -----------------------------
TARGET="${TARGET:-worker}"  # worker | cloudrun  (cloudrun = opzionale per isolare gateway)
WORKER_URL="${WORKER_URL:-https://mm-ai-mock.square-sunset-7388.workers.dev/v1/ai/ask}"
CLOUD_RUN_URL="${CLOUD_RUN_URL:-https://mm-ai-ingest-fixed-pvgxe6eo5q-ew.a.run.app/v1/ai/ask}"

TOP_K="${TOP_K:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
MAX_TIME="${MAX_TIME:-60}"
HTTP_FORCE_1_1="${HTTP_FORCE_1_1:-1}"  # 1 = forza http/1.1 (spesso riduce flakiness)

# -----------------------------
# IDs (DEV)
# -----------------------------
COMPANY_A="1766334551970x848077659418132500"
MACHINE_A="1766967703577x651809929304997900"
DOC_A="1771493804876x415817609722462200"
SENT_A="SENTINEL_A_9Q7X3"

COMPANY_B="1768570969798x571640985075318800"
MACHINE_B="1771493897062x931194232803426300"
DOC_B="1771494052788x533049530417938400"
SENT_B="SENTINEL_B_K2M8P"

# -----------------------------
# Preconditions
# -----------------------------
if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: TOKEN env var is required." >&2
  echo "Example: export TOKEN=\"v1_internal_...\"" >&2
  exit 2
fi

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command '$1'" >&2; exit 2; }; }
need_cmd curl
need_cmd python3
need_cmd mktemp
need_cmd head
need_cmd sed
need_cmd tr

log() { echo "$*" >&2; }

is_blank() {
  # true if empty or only whitespace
  local s="${1:-}"
  [[ -z "${s//[[:space:]]/}" ]]
}

# -----------------------------
# Payload builder (worker vs cloudrun)
# -----------------------------
build_payload() {
  # Args: company machine doc query
  local company="$1"
  local machine="$2"
  local doc="$3"
  local query="$4"

  python3 - "$company" "$machine" "$doc" "$query" <<'PY'
import json, os, sys

company, machine, doc, query = sys.argv[1:5]
token = os.environ.get("TOKEN", "")
top_k = int(os.environ.get("TOP_K", "5"))
target = os.environ.get("TARGET", "worker")

if target == "cloudrun":
    # Cloud Run /ask vuole top_k al root; secret va in header
    doc_ids = [doc] if doc else None
    payload = {
        "company_id": company,
        "machine_id": machine,
        "document_ids": doc_ids,
        "query": query,
        "top_k": top_k,
        "debug": False,
    }
else:
    # Worker vuole auth_token nel body e top_k in options
    payload = {
        "auth_token": token,
        "company_id": company,
        "machine_id": machine,
        "document_ids": doc,  # stringa singola ok
        "query": query,
        "options": {"top_k": top_k},
    }

print(json.dumps(payload, ensure_ascii=False))
PY
}

# -----------------------------
# Parse status from any response body
# - stdout: ONLY status string OR PARSE_ERROR
# - stderr: debug (non rompe command substitution)
# -----------------------------
parse_status() {
  python3 - <<'PY'
import json, sys

raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
raw_strip = raw.strip()

obj = None
# 1) try full parse
try:
    obj = json.loads(raw_strip)
except Exception:
    obj = None

# 2) fallback: extract first JSON object
if not isinstance(obj, dict):
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidate = raw[i:j+1]
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = None

if not isinstance(obj, dict):
    sys.stderr.write("parse_status: PARSE_ERROR (not JSON). RAW_HEAD=" + raw[:220].replace("\n","\\n") + "\n")
    print("PARSE_ERROR")
    sys.exit(0)

status = obj.get("status") or ""
if not status:
    sys.stderr.write("parse_status: PARSE_ERROR (missing .status). keys=" + ",".join(obj.keys()) + "\n")
    print("PARSE_ERROR")
    sys.exit(0)

# Se status=error, stampa dettagli utili su stderr
if status == "error":
    err = obj.get("error") or {}
    code = obj.get("error_code") or err.get("code") or ""
    msg = err.get("message") or ""
    sys.stderr.write(f"parse_status: API status=error code={code} message={msg}\n")

print(status)
PY
}

# -----------------------------
# One request with retries
# - stdout: ONLY final status token
# - stderr: debug on retries/fail
# -----------------------------
ask_status() {
  # Args: name company machine doc query
  local name="$1"
  local company="$2"
  local machine="$3"
  local doc="$4"
  local query="$5"

  local url=""
  if [[ "$TARGET" == "cloudrun" ]]; then
    url="$CLOUD_RUN_URL"
  else
    url="$WORKER_URL"
  fi

  local payload
  payload="$(build_payload "$company" "$machine" "$doc" "$query")"

  local attempt=1
  local backoff=1

  local last_rc=0
  local last_http=""
  local last_body=""
  local last_hdr=""
  local last_reason=""

  # curl base args
  local -a CURL_ARGS
  CURL_ARGS=(--max-time "$MAX_TIME" --connect-timeout "$CONNECT_TIMEOUT" -sS)
  if [[ "$HTTP_FORCE_1_1" == "1" ]]; then
    CURL_ARGS+=(--http1.1)
  fi

  while [[ $attempt -le $MAX_ATTEMPTS ]]; do
    local body_file hdr_file http_code rc body hdr status
    body_file="$(mktemp)"
    hdr_file="$(mktemp)"

    set +e
    if [[ "$TARGET" == "cloudrun" ]]; then
      http_code="$(curl "${CURL_ARGS[@]}" -X POST "$url" \
        -H "Content-Type: application/json; charset=utf-8" \
        -H "X-AI-Internal-Secret: ${TOKEN}" \
        --data-binary "$payload" \
        -o "$body_file" -D "$hdr_file" -w "%{http_code}")"
    else
      http_code="$(curl "${CURL_ARGS[@]}" -X POST "$url" \
        -H "Content-Type: application/json; charset=utf-8" \
        --data-binary "$payload" \
        -o "$body_file" -D "$hdr_file" -w "%{http_code}")"
    fi
    rc=$?
    set -e

    body="$(cat "$body_file" 2>/dev/null || true)"
    hdr="$(cat "$hdr_file" 2>/dev/null || true)"
    rm -f "$body_file" "$hdr_file"

    last_rc="$rc"
    last_http="${http_code:-}"
    last_body="$body"
    last_hdr="$hdr"

    # Retry rules
    if [[ $rc -ne 0 ]]; then
      last_reason="curl_rc_${rc}"
    elif [[ -z "${http_code:-}" || "${http_code:-}" == "000" ]]; then
      last_reason="no_http_code"
    elif is_blank "$body"; then
      last_reason="empty_body"
    elif [[ "$http_code" != "200" ]]; then
      last_reason="http_${http_code}"
    else
      status="$(printf '%s' "$body" | parse_status)"
      if [[ "$status" != "PARSE_ERROR" && -n "$status" ]]; then
        echo "$status"
        return 0
      fi
      last_reason="parse_error"
    fi

    if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
      log "WARN [$name] attempt $attempt/$MAX_ATTEMPTS failed ($last_reason). retrying in ${backoff}s..."
      sleep "$backoff"
      backoff=$((backoff * 2))
      [[ $backoff -gt 16 ]] && backoff=16
    fi
    attempt=$((attempt + 1))
  done

  # Final failure: print useful debug to stderr, return a clear token to stdout
  log "FAIL [$name] after $MAX_ATTEMPTS attempts ($last_reason)"
  log "  target=$TARGET url=$url"
  log "  curl_rc=$last_rc http_code=${last_http:-<none>}"

  log "  ---- response headers (first 20 lines) ----"
  log "$(printf '%s' "$last_hdr" | head -n 20 | sed -e 's/\r$//')"

  log "  ---- response body (first 600 chars) ----"
  log "$(printf '%s' "$last_body" | head -c 600 | tr '\n' ' ')"

  if [[ "$last_reason" == "parse_error" ]]; then
    echo "PARSE_ERROR"
  else
    echo "NETWORK_ERROR"
  fi
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

echo "Running smoke tests"
echo "  TARGET=$TARGET"
echo "  WORKER_URL=$WORKER_URL"
echo "  CLOUD_RUN_URL=$CLOUD_RUN_URL"
echo "  TOP_K=$TOP_K MAX_ATTEMPTS=$MAX_ATTEMPTS"
echo

# Query “pulita”: metto la sentinella per prima così il fallback code-token è deterministico
Q1="$SENT_A"
Q2="$SENT_B"

S1="$(ask_status "A sees A" "$COMPANY_A" "$MACHINE_A" "$DOC_A" "$Q1")"
assert_eq "A sees A" "$S1" "answered"

S2="$(ask_status "B sees B" "$COMPANY_B" "$MACHINE_B" "$DOC_B" "$Q2")"
assert_eq "B sees B" "$S2" "answered"

S3="$(ask_status "A sees B (should block)" "$COMPANY_A" "$MACHINE_A" "$DOC_B" "$Q2")"
assert_eq "A sees B (should block)" "$S3" "no_sources"

S4="$(ask_status "B sees A (should block)" "$COMPANY_B" "$MACHINE_B" "$DOC_A" "$Q1")"
assert_eq "B sees A (should block)" "$S4" "no_sources"

echo
echo "✅ All smoke tests passed."
