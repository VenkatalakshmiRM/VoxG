#!/usr/bin/env bash
# Verify PRISM (PRISMtrace) connectivity before logging the real batch.
# Loads credentials from the repo-root .env. Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

: "${PRISMTRACE_HOST:?Set PRISMTRACE_HOST in .env}"
: "${PRISMTRACE_API_KEY:?Set PRISMTRACE_API_KEY in .env}"
: "${PRISMTRACE_PROJECT_ID:?Set PRISMTRACE_PROJECT_ID in .env}"

echo "== Check 1: credential handshake =="
HANDSHAKE=$(curl -sS -X POST "$PRISMTRACE_HOST/api/setup-doctor/handshake" \
  -H "Content-Type: application/json" \
  -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY" \
  -d "{\"project_id\": \"$PRISMTRACE_PROJECT_ID\", \"send_test_trace\": true}")
echo "$HANDSHAKE"

echo
echo "== Check 2: live traffic check =="
STATUS=$(curl -sS "$PRISMTRACE_HOST/api/setup-doctor?project_id=$PRISMTRACE_PROJECT_ID" \
  -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY")
echo "$STATUS"

echo
LIVE=$(echo "$STATUS" | python -c "import sys,json; print(json.load(sys.stdin).get('live_connected', False))" 2>/dev/null || echo False)
if [[ "$LIVE" == "True" ]]; then
  echo "PRISM LIVE CONNECTED — proceed with the real batch."
else
  echo "NOT live connected (live_connected=$LIVE, blocked_step above). Fix before the evaluation session."
  exit 1
fi
