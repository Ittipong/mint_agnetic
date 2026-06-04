#!/usr/bin/env bash
# v3 chat-agent smoke — curl the live deployment through the Cloudflare tunnel.
#
# Per memory `feedback_test_via_cloudflared`, manual tests MUST hit the
# tunnel URL — NOT localhost — so the smoke exercises the same path the
# mobile client uses (TLS, header rewriting, SSE framing through CF).
#
# Per memory `reference_cloudflare_tunnel`, the tunnel is auto-started via
# launchd. If smoke fails on connectivity, kick the tunnel:
#   launchctl kickstart -k gui/$(id -u)/com.cloudflare.cloudflared.mintmoney
# DO NOT fall back to localhost — that hides bugs in TLS / framing.
#
# Exits non-zero on ANY failure (set -euo pipefail + per-assertion exits).
#
# USAGE:
#   bash scripts/smoke.sh                     # full suite
#   CHAT_HOST=https://staging.example.com bash scripts/smoke.sh    # override
#   SMOKE_USER=<uuid> bash scripts/smoke.sh   # override seed user

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST="${CHAT_HOST:-https://chat.minttechdev.uk}"
SMOKE_USER="${SMOKE_USER:-ba91d8a5-46b2-46f7-aaf4-189a54e17fe9}"
WALLET_ID="${SMOKE_WALLET:-00000000-0001-4000-a000-000000000001}"
TIMEOUT="${SMOKE_TIMEOUT:-30}"

fail() {
    echo "SMOKE FAIL: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

# ---------------------------------------------------------------------------
# Pre-flight — host reachable
# ---------------------------------------------------------------------------

step "Pre-flight: HOST=${HOST} USER=${SMOKE_USER}"

if ! command -v curl >/dev/null 2>&1; then
    fail "curl not on PATH"
fi
if ! command -v jq >/dev/null 2>&1; then
    fail "jq not on PATH (needed for healthz parse)"
fi

# ---------------------------------------------------------------------------
# 1. /healthz — fastest possible round-trip
# ---------------------------------------------------------------------------

step "1. GET /healthz"
HEALTHZ_OUT=$(curl --max-time "${TIMEOUT}" -sf "${HOST}/healthz") || \
    fail "healthz unreachable (tunnel down? service stopped?)"

# Accept either {"ok":true} or {"status":"ok"} — different lifespan rollouts
# emit slightly different envelopes; smoke only cares about 2xx + JSON.
echo "    healthz payload: ${HEALTHZ_OUT}"
echo "${HEALTHZ_OUT}" | jq . >/dev/null || fail "healthz returned non-JSON"

# ---------------------------------------------------------------------------
# 2. /chat/stream — text turn (greeting; fastest path; no DB writes)
# ---------------------------------------------------------------------------

step "2. POST /chat/stream — greeting"

THREAD_TEXT="smoke-text-$(date +%s)-$$"
TEXT_BODY=$(cat <<EOF
{"user_id":"${SMOKE_USER}","thread_id":"${THREAD_TEXT}","message":"สวัสดี","wallet_id":"${WALLET_ID}"}
EOF
)

TEXT_OUT_FILE=$(mktemp -t v3smoke_text.XXXXXX)
trap 'rm -f "${TEXT_OUT_FILE}" "${ADD_OUT_FILE:-}"' EXIT

# -N = no buffering (SSE); --max-time bounds the whole stream.
curl --max-time "${TIMEOUT}" -Nsf \
    -X POST "${HOST}/chat/stream" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    --data-raw "${TEXT_BODY}" \
    >"${TEXT_OUT_FILE}" \
    || fail "text stream POST failed"

# Wire contract: must contain at least one answer_token and a done event.
grep -q "^event: answer_token" "${TEXT_OUT_FILE}" \
    || fail "text stream missing answer_token (head dump above)"
grep -q "^event: done" "${TEXT_OUT_FILE}" \
    || fail "text stream missing done event"
echo "    text stream OK ($(wc -l <"${TEXT_OUT_FILE}") lines)"

# ---------------------------------------------------------------------------
# 3. /chat/stream — ADD turn (smoke that propose_transaction wires through)
# ---------------------------------------------------------------------------

step "3. POST /chat/stream — ADD 'เพิ่ม 50 ค่าน้ำ'"

THREAD_ADD="smoke-add-$(date +%s)-$$"
ADD_BODY=$(cat <<EOF
{"user_id":"${SMOKE_USER}","thread_id":"${THREAD_ADD}","message":"เพิ่ม 50 ค่าน้ำ","wallet_id":"${WALLET_ID}"}
EOF
)

ADD_OUT_FILE=$(mktemp -t v3smoke_add.XXXXXX)
curl --max-time "${TIMEOUT}" -Nsf \
    -X POST "${HOST}/chat/stream" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    --data-raw "${ADD_BODY}" \
    >"${ADD_OUT_FILE}" \
    || fail "ADD stream POST failed"

grep -q "^event: block" "${ADD_OUT_FILE}" \
    || fail "ADD stream emitted no block events"
grep -q "transaction_proposal" "${ADD_OUT_FILE}" \
    || fail "ADD stream did not emit a transaction_proposal block"
grep -q "^event: done" "${ADD_OUT_FILE}" \
    || fail "ADD stream missing done"
echo "    ADD stream OK ($(wc -l <"${ADD_OUT_FILE}") lines)"

# ---------------------------------------------------------------------------
# 4. /threads — list endpoint (smoke that thread CRUD still works)
# ---------------------------------------------------------------------------

step "4. GET /threads?user_id=…"

curl --max-time "${TIMEOUT}" -sf \
    "${HOST}/threads?user_id=${SMOKE_USER}&limit=5" \
    | jq -e '.threads | type == "array"' >/dev/null \
    || fail "/threads did not return {threads:[…]}"
echo "    threads listing OK"

# ---------------------------------------------------------------------------
# All green
# ---------------------------------------------------------------------------

echo
echo "===== SMOKE OK ====="
echo "host=${HOST} thread_text=${THREAD_TEXT} thread_add=${THREAD_ADD}"
exit 0
