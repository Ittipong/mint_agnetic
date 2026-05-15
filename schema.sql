-- Mint Agentic AI Tables
-- References: ai_insight_card_final_spec.md v2.4

-- ============================================================
-- 1. ai_insight_logs — every AI message + user action
-- ============================================================
CREATE TABLE IF NOT EXISTS ai_insight_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,

    -- LLM output
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    cta_primary_label   TEXT NOT NULL,
    cta_primary_action  TEXT NOT NULL,
    cta_secondary_label TEXT NOT NULL,

    -- Classifier labels
    insight_type    TEXT NOT NULL,     -- celebration|insight|alert|forecast|nudge
    topic           TEXT NOT NULL,      -- goal|budget|spending|cashflow|anomaly
    urgency         TEXT NOT NULL,      -- urgent|normal|low

    -- Context
    trigger_source  TEXT NOT NULL,     -- foreground|scheduled|event:X
    agent_reasoning TEXT,
    confidence      NUMERIC(3,2),

    -- User action
    action          TEXT,              -- tap|dismiss|snooze|ignore|NULL
    action_at       TIMESTAMPTZ,

    -- Delivery
    delivered       BOOLEAN NOT NULL DEFAULT false,
    delivered_at    TIMESTAMPTZ,
    gate_result     TEXT,              -- pass|low_confidence|no_message

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted      BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_logs_user_today
    ON ai_insight_logs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_user_topic
    ON ai_insight_logs (user_id, topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_user_action
    ON ai_insight_logs (user_id, action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_delivered
    ON ai_insight_logs (user_id, delivered, created_at DESC);

-- ============================================================
-- 2. ai_user_preferences — long-term learned memory
-- ============================================================
CREATE TABLE IF NOT EXISTS ai_user_preferences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL UNIQUE,

    -- TONE (slow-changing, inferred from taps)
    preferred_tone  TEXT NOT NULL DEFAULT 'neutral',
    tone_scores     JSONB NOT NULL DEFAULT '{}',

    -- CTA RESPONSE (inferred from taps, EMA)
    cta_response_rate JSONB NOT NULL DEFAULT '{}',

    -- ACTIVITY PATTERN (inferred from app opens)
    active_hours    JSONB NOT NULL DEFAULT '[]',
    active_days     JSONB NOT NULL DEFAULT '[]',

    -- DIVERSITY TRACKING (recent topics, NOT suppression)
    recent_topics_sent JSONB NOT NULL DEFAULT '[]',

    -- LEARNING COUNTS
    total_taps      INTEGER NOT NULL DEFAULT 0,
    total_dismisses INTEGER NOT NULL DEFAULT 0,

    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pref_user ON ai_user_preferences (user_id);

-- ============================================================
-- 3. ai_insight_cooldowns — per-insight/topic/goal state
-- ============================================================
CREATE TABLE IF NOT EXISTS ai_insight_cooldowns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,

    cooldown_type   TEXT NOT NULL,      -- topic|goal_id|insight_hash|category
    cooldown_key    TEXT NOT NULL,      -- e.g. "goal:abc123" or "topic:budget"
    cooldown_scope  TEXT NOT NULL,      -- seen_permanently|dismissed_3d|fired_14d
    expires_at      TIMESTAMPTZ,        -- NULL = permanent
    reason          TEXT,                -- milestone_seen|user_dismissed|rate_limit
    extra_data      JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cd_lookup
    ON ai_insight_cooldowns (user_id, cooldown_type, cooldown_key);
CREATE INDEX IF NOT EXISTS idx_cd_expires
    ON ai_insight_cooldowns (expires_at)
    WHERE expires_at IS NOT NULL;

-- ============================================================
-- 4. chat_threads — Flutter chat session metadata
-- Messages themselves live in LangGraph's `checkpoints` table;
-- this table only stores sidebar metadata (title, preview, sort key).
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id            TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL,
    title                TEXT NOT NULL DEFAULT 'แชตใหม่',
    last_message_preview TEXT,
    message_count        INT  NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_user
    ON chat_threads (user_id, updated_at DESC);
