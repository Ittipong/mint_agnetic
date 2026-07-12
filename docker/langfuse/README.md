# Langfuse self-host (Mint chat tracing — Phase 1 PoC)

Self-hosted Langfuse for tracing the chat agent per **user** and **conversation**.
Self-host (not cloud) because chat traces contain the user's real amounts —
CLAUDE.md forbids shipping payment info to a third party.

## What's here
- `docker-compose.yml` — adapted from the official Langfuse v3 self-host stack
  (web + worker + postgres + clickhouse + redis + minio). Only `langfuse-web` is
  published, on **127.0.0.1:3100** (host 3000/5432/6379 are taken locally).
- `.env` — local secrets + **headless bootstrap** (`LANGFUSE_INIT_*`): on first
  boot Langfuse auto-creates the org/project/user and **fixed API keys**, so no
  UI click-through is needed. ⚠️ local-only secrets — gitignored, never for prod.
- `emit_trace.py` — emits one trace through the SAME `attach_langfuse()` handler
  `server.py` uses (proves the wiring end-to-end).

## Bring it up
```bash
cd docker/langfuse
docker compose -p langfuse-mint up -d
# wait for health, then open http://localhost:3100
curl -s http://localhost:3100/api/public/health   # {"status":"OK",...}
```
UI login (from `.env` LANGFUSE_INIT_USER_*): **admin@mint.local / mintadmin123**

Fixed project keys (from `.env` LANGFUSE_INIT_PROJECT_*):
- public: `pk-lf-mint-poc-public`
- secret: `sk-lf-mint-poc-secret`
- host:   `http://localhost:3100`

## Turn on tracing in the chat service
Add to `mint_agentic/.env` (the handler is OFF until `LANGFUSE_ENABLED=on`):
```
LANGFUSE_ENABLED=on
LANGFUSE_PUBLIC_KEY=pk-lf-mint-poc-public
LANGFUSE_SECRET_KEY=sk-lf-mint-poc-secret
LANGFUSE_HOST=http://localhost:3100
```
Restart the chat service → every chat turn becomes one Langfuse trace, grouped by
`user_id` (langfuse_user_id) + conversation (`langfuse_session_id` = thread_id).

## Verify (no real chat turn needed)
```bash
PYTHONPATH=. .venv/bin/python docker/langfuse/emit_trace.py
# then check the UI: Tracing → row "mint-chat-poc-turn", or the API:
AUTH=$(printf 'pk-lf-mint-poc-public:sk-lf-mint-poc-secret' | base64)
curl -s -H "Authorization: Basic $AUTH" \
  "http://localhost:3100/api/public/traces?userId=poc-user-verify&limit=5"
```

## Stop / reset
```bash
docker compose -p langfuse-mint down          # stop (keeps data volumes)
docker compose -p langfuse-mint down -v        # stop + wipe all data
```
