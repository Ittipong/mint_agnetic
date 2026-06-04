# User Preferences — durable "remember about the user" for the chat agent

## Why this exists

The chat agent (mint_agentic_v3) gives better financial advice when it
remembers durable context about a user across threads — but only the kind of
context that has **no transaction behind it**. This feature is that memory:
a structured, consent-gated, server-side store the agent reads every turn and
writes when it learns something stable.

It is **not** the agent's free-text memory store (`memory_recall`/`memory_write`),
and **not** the identity profile (`user_profile`). See the three-layer split
below.

## The three layers (don't conflate them)

| Layer | What | Where | Who writes |
|---|---|---|---|
| **Identity** | name, email, avatar, occupation | `user_profile` | auth / onboarding |
| **Preferences** (this) | non-derivable financial context + AI style | `user_preferences` | the agent (round 1); a settings UI (later) |
| **Learned facts** | fuzzy things the agent infers mid-chat | LangGraph store (`memory_*` tools) | the agent |

## The one rule: derive, don't store

Anything with transactions behind it — actual income, debt balance, savings,
spending, budget usage, goal progress — is **derived live** by `run_python`
from `transactions` / wallets / `goal_wallets`. Storing those here would go
stale and contradict the real numbers. `user_preferences` holds ONLY what the
data can't tell us:

- **Financial context** (consent-gated): `financial_literacy_level`,
  `income_stability`, `salary_day`, `housing_status`, `dependents_count`,
  `life_stage`, `risk_tolerance`, `emergency_fund_target_months`,
  `primary_goal_priority`, `debt_payoff_strategy`, `declared_monthly_income`
  (self-stated, optional), `financial_notes`.
- **AI style** (not sensitive, always applied): `ai_preferences` jsonb —
  `ai_tone`, `ai_response_length`, `ai_language`, `ai_proactivity`,
  `use_emoji`, `coaching_frequency`, `preferred_checkin_time`, `avoid_topics`.

The write tool's whitelist enforces this — an off-list field (e.g.
`total_debt`) is rejected with a hint to use `run_python`.

## Meta columns make it a real memory, not a dumb k/v

- **`memory_consent`** (bool) — PDPA opt-in. Personal/financial context is
  neither injected into the prompt nor writable until this is `true`. AI style
  and occupation are exempt (non-sensitive / identity).
- **`field_sources`** (jsonb) — per-field provenance: `user_stated` vs
  `ai_inferred`. The prompt tells the LLM to trust user-stated over inferred,
  and never present an inferred value as confirmed for irreversible decisions.
- **`last_confirmed_at`** — lets the agent re-ask when context goes stale.
- **`onboarding_completed`** — whether the prefs intake has run.

## How the agent uses it

**Read (every turn):** `_pre_turn_hook` (graph.py) calls
`load_user_preferences(user_id)` → writes the `user_preferences` state channel
→ `_make_prompt` renders it via `format_about_user_block` into a `[# ABOUT THIS
USER]` block in the system prompt. Absent / no-consent / null fields collapse
to nothing (no empty slots for the LLM to hallucinate into).

**Write:** the `set_user_preference(field, value, source)` tool upserts one
field (ON CONFLICT user_id), stamps `field_sources` + `last_confirmed_at`, and
returns the refreshed prefs into state so the same turn sees the update.
`memory_consent` is always recorded as `user_stated` (consent is the user's act).

## Scope notes

- **Round 1 (this):** DB + agent read/write only. No backend Go endpoint, no
  mobile sync, no settings UI yet.
- **No `sync_id` yet.** This mirrors `user_profile` (server-side, one row per
  user, not Drift-synced). When the mobile settings UI lands, add `sync_id` +
  a sync handler in that round — the table is one column away from sync-ready.
- One row per user (`UNIQUE(user_id)`), `ON DELETE CASCADE` with `users`.

## Files

- Migration: `backend/migrations/sql/000029_add_user_preferences.{up,down}.sql`
- Read + render: `src/agent/user_preferences.py`
- Write tool: `src/agent/tools/user_preferences_tool.py` (registered in
  `tools/__init__.py`, 8th tool)
- Wiring: `_pre_turn_hook` + `_make_prompt` in `src/agent/graph.py`;
  `{about_user}` placeholder + `render_system_prompt` in `src/agent/prompts.py`;
  pool installed via `set_preferences_pool` in `src/agent/server.py` lifespan
- Tests: `tests/test_user_preferences.py`
