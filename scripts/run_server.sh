#!/usr/bin/env bash
# Entrypoint for the always-on launchd daemon (uk.minttechdev.chat-agent-v3).
#
# Why a wrapper: --reload must be enabled ONLY while developing. A plist is
# static XML, so we decide here at launch time based on ENVIRONMENT in .env —
# development -> hot reload; production/staging -> stable, no reload.
set -euo pipefail

# Resolve the v3 project root (this script lives in <root>/scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
PORT="${PORT:-2026}"

# Read ENVIRONMENT from .env without sourcing the whole file (avoids executing it).
ENV_VALUE="$(grep -E '^ENVIRONMENT=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '[:space:]')"

RELOAD_ARGS=()
if [ "$ENV_VALUE" = "development" ]; then
  RELOAD_ARGS=(--reload)
  echo "[run_server] ENVIRONMENT=development -> hot reload ON"
else
  echo "[run_server] ENVIRONMENT=${ENV_VALUE:-unset} -> hot reload OFF"
fi

exec "$PY" -m uvicorn agent.server:app --port "$PORT" --app-dir src "${RELOAD_ARGS[@]}"
