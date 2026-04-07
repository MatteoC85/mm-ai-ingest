#!/usr/bin/env bash
set -euo pipefail

: "${AI_URL:?Set AI_URL}"
: "${AI_SECRET:?Set AI_SECRET}"
: "${COMPANY_ID:?Set COMPANY_ID}"
: "${MACHINE_ID:?Set MACHINE_ID}"

run_ask () {
  local Q="$1"
  jq -n \
    --arg q "$Q" \
    --arg c "$COMPANY_ID" \
    --arg m "$MACHINE_ID" \
    '{query:$q, company_id:$c, machine_id:$m, top_k:5, debug:true}' \
  | curl -sS -X POST "$AI_URL/v1/ai/ask" \
      -H "Content-Type: application/json" \
      -H "X-AI-Internal-Secret: $AI_SECRET" \
      --data @- \
  | jq -c --arg q "$Q" '{
      q:$q,
      status:(.status // null),
      sim:(.debug.similarity_max // null),
      citations:((.citations // []) | map(.citation_id) | sort),
      answer:(.answer // null)
    }'
}

run_rc () {
  local Q="$1"
  jq -n \
    --arg q "$Q" \
    --arg c "$COMPANY_ID" \
    --arg m "$MACHINE_ID" \
    '{query:$q, company_id:$c, machine_id:$m, top_k:8, max_causes:3, debug:true}' \
  | curl -sS -X POST "$AI_URL/v1/ai/root-cause" \
      -H "Content-Type: application/json" \
      -H "X-AI-Internal-Secret: $AI_SECRET" \
      --data @- \
  | jq -c --arg q "$Q" '{
      q:$q,
      status:(.status // null),
      sim:(.debug.similarity_max // null),
      citations:((.citations // []) | map(.citation_id) | sort),
      causes:((.possible_causes // []) | map(.cause))
    }'
}

mkdir -p tmp

echo "===== ASK MATRIX ====="
{
  run_ask "la macchina vibra"
  run_ask "la macchina vibra quando piega"
  run_ask "the machine vibrates"
  run_ask "the machine vibrates during bending"
  run_ask "fa rumore durante piegatura"
  run_ask "it makes noise during bending"
  run_ask "si blocca durante avanzamento"
  run_ask "it jams during feeding"
  run_ask "non parte in automatico"
  run_ask "it does not start in automatic mode"
} | tee tmp/ask_matrix.jsonl

echo
echo "===== ROOT-CAUSE MATRIX ====="
{
  run_rc "la macchina vibra"
  run_rc "la macchina vibra quando piega"
  run_rc "the machine vibrates"
  run_rc "the machine vibrates during bending"
  run_rc "fa rumore durante piegatura"
  run_rc "it makes noise during bending"
  run_rc "si blocca durante avanzamento"
  run_rc "it jams during feeding"
} | tee tmp/rc_matrix.jsonl

echo
echo "===== REPEATABILITY ====="
mkdir -p tmp/ask_repeat tmp/rc_repeat
rm -rf tmp/ask_repeat/* tmp/rc_repeat/*

for Q in \
  "la macchina vibra quando piega" \
  "the machine vibrates during bending" \
  "fa rumore durante piegatura" \
  "it makes noise during bending"
do
  safe=$(printf '%s' "$Q" | tr ' /' '__' | tr -cd '[:alnum:]_')
  mkdir -p "tmp/ask_repeat/$safe" "tmp/rc_repeat/$safe"

  for i in $(seq 1 20); do
    jq -n \
      --arg q "$Q" \
      --arg c "$COMPANY_ID" \
      --arg m "$MACHINE_ID" \
      '{query:$q, company_id:$c, machine_id:$m, top_k:5, debug:true}' \
    | curl -sS -X POST "$AI_URL/v1/ai/ask" \
        -H "Content-Type: application/json" \
        -H "X-AI-Internal-Secret: $AI_SECRET" \
        --data @- \
    | jq -S . > "tmp/ask_repeat/$safe/$i.json"

    jq -n \
      --arg q "$Q" \
      --arg c "$COMPANY_ID" \
      --arg m "$MACHINE_ID" \
      '{query:$q, company_id:$c, machine_id:$m, top_k:8, max_causes:3, debug:true}' \
    | curl -sS -X POST "$AI_URL/v1/ai/root-cause" \
        -H "Content-Type: application/json" \
        -H "X-AI-Internal-Secret: $AI_SECRET" \
        --data @- \
    | jq -S . > "tmp/rc_repeat/$safe/$i.json"
  done

  echo
  echo "===== ASK REPEAT | $Q ====="
  echo "STATUS"
  jq -r '.status // "null"' "tmp/ask_repeat/$safe"/*.json | sort | uniq -c
  echo "ANSWER HASH"
  for f in "tmp/ask_repeat/$safe"/*.json; do
    jq -r '.answer // ""' "$f" | tr -s '[:space:]' ' ' | sha256sum | cut -d" " -f1
  done | sort | uniq -c
  echo "CITATION HASH"
  for f in "tmp/ask_repeat/$safe"/*.json; do
    jq -r '((.citations // []) | map(.citation_id) | sort | join("|"))' "$f" | sha256sum | cut -d" " -f1
  done | sort | uniq -c

  echo
  echo "===== ROOT-CAUSE REPEAT | $Q ====="
  echo "STATUS"
  jq -r '.status // "null"' "tmp/rc_repeat/$safe"/*.json | sort | uniq -c
  echo "CAUSE HASH"
  for f in "tmp/rc_repeat/$safe"/*.json; do
    jq -r '((.possible_causes // []) | map(.cause) | join(" | "))' "$f" | tr -s '[:space:]' ' ' | sha256sum | cut -d" " -f1
  done | sort | uniq -c
  echo "CITATION HASH"
  for f in "tmp/rc_repeat/$safe"/*.json; do
    jq -r '((.citations // []) | map(.citation_id) | sort | join("|"))' "$f" | sha256sum | cut -d" " -f1
  done | sort | uniq -c
done
