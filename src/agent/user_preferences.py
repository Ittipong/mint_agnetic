"""Durable user preferences — load + render the `[about_user]` prompt block.

NEW in the user-preferences feature. Companion to `entity_catalog.py`:
the catalog answers "what wallets/categories does this user have" (derived,
per-turn); THIS module answers "what do we durably KNOW about this user"
(declared / AI-inferred context that has no transaction behind it).

Design (see docs/user_preferences.md):
  - Source of truth = the BACKEND DB `user_preferences` table (+ the
    `user_profile.occupation` identity field), NOT the agent store.
  - We store ONLY non-derivable context. Anything with a transaction behind
    it (income, debt, goals, budgets) is derived live by the agent, never
    cached here.
  - PDPA: `memory_consent` gates the personal/financial block. Without it we
    still surface occupation (identity) + AI style (non-sensitive), but no
    financial context.

This module is import-safe (no DB connection at import). The backend pool is
installed once at startup via `set_preferences_pool(pool)` (FastAPI lifespan);
tests inject a fake via `set_preferences_loader(fn)`.
"""

from __future__ import annotations

from typing import Any, Optional

from src.agent.session_logger import slog


# ---------------------------------------------------------------------------
# Field taxonomy — shared with the write tool (single source of truth).
# ---------------------------------------------------------------------------

# Financial-context columns. PERSONAL → gated behind `memory_consent`.
FINANCIAL_FIELDS: tuple[str, ...] = (
    "salary_day",
    "income_stability",
    "housing_status",
    "financial_literacy_level",
    "emergency_fund_target_months",
    "primary_goal_priority",
    "debt_payoff_strategy",
    "dependents_count",
    "risk_tolerance",
    "life_stage",
    "declared_monthly_income",
    "financial_notes",
)

# Human-readable labels for the prompt block (English keys — internal context).
_LABELS: dict[str, str] = {
    "occupation": "Occupation",
    "financial_literacy_level": "Financial literacy",
    "income_stability": "Income stability",
    "salary_day": "Salary day (day of month)",
    "housing_status": "Housing",
    "dependents_count": "Dependents",
    "risk_tolerance": "Risk tolerance",
    "life_stage": "Life stage",
    "emergency_fund_target_months": "Emergency-fund target (months)",
    "primary_goal_priority": "Primary goal priority",
    "debt_payoff_strategy": "Debt payoff strategy",
    "declared_monthly_income": "Declared monthly income (self-stated)",
    "financial_notes": "Notes",
}

# Order the financial lines appear in the block.
_FINANCIAL_ORDER: tuple[str, ...] = (
    "financial_literacy_level",
    "income_stability",
    "salary_day",
    "housing_status",
    "dependents_count",
    "life_stage",
    "risk_tolerance",
    "emergency_fund_target_months",
    "primary_goal_priority",
    "debt_payoff_strategy",
    "declared_monthly_income",
    "financial_notes",
)


# ---------------------------------------------------------------------------
# SQL — psycopg uses %s placeholders. Base off the literal uid so a user with
# no preferences row (and even no profile row) still resolves to one NULL row.
# ---------------------------------------------------------------------------
_PREFERENCES_SQL = (
    "SELECT "
    "  up.salary_day, up.income_stability, up.housing_status, "
    "  up.financial_literacy_level, up.emergency_fund_target_months, "
    "  up.primary_goal_priority, up.debt_payoff_strategy, up.dependents_count, "
    "  up.risk_tolerance, up.life_stage, up.declared_monthly_income, "
    "  up.financial_notes, "
    "  COALESCE(up.memory_consent, false) AS memory_consent, "
    "  COALESCE(up.field_sources, '{}'::jsonb) AS field_sources, "
    "  COALESCE(up.onboarding_completed, false) AS onboarding_completed, "
    "  up.last_confirmed_at, "
    "  COALESCE(up.ai_preferences, '{}'::jsonb) AS ai_preferences, "
    "  (up.user_id IS NOT NULL) AS exists, "
    "  pr.occupation "
    "FROM (SELECT %s::uuid AS uid) x "
    "LEFT JOIN user_preferences up ON up.user_id = x.uid "
    "LEFT JOIN user_profile pr ON pr.user_id = x.uid"
)

_COLUMNS: tuple[str, ...] = (
    "salary_day", "income_stability", "housing_status",
    "financial_literacy_level", "emergency_fund_target_months",
    "primary_goal_priority", "debt_payoff_strategy", "dependents_count",
    "risk_tolerance", "life_stage", "declared_monthly_income",
    "financial_notes", "memory_consent", "field_sources",
    "onboarding_completed", "last_confirmed_at", "ai_preferences",
    "exists", "occupation",
)


async def load_user_preferences_from_pool(
    pool, user_id: str,
) -> Optional[dict]:
    """Read the user's preferences + occupation from the backend pool.

    Returns a plain dict (JSON/MsgPack-safe so it survives the LangGraph
    checkpointer) or None when there's nothing to surface (no pool, no
    user_id, or DB error). Never raises — a failed preferences load must not
    break a turn; the agent simply runs without the `[about_user]` block.
    """
    if pool is None or not user_id:
        return None
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_PREFERENCES_SQL, (user_id,))
                row = await cur.fetchone()
    except Exception as exc:  # noqa: BLE001 — preferences must not crash a turn
        slog("load_user_preferences", f"load failed: {type(exc).__name__}: {exc}")
        return None

    if row is None:
        return None
    data = {col: _coerce(row[i]) for i, col in enumerate(_COLUMNS)}
    slog(
        "load_user_preferences",
        f"loaded exists={data.get('exists')} consent={data.get('memory_consent')} "
        f"occupation={bool(data.get('occupation'))}",
    )
    return data


def _coerce(value: Any) -> Any:
    """Marshal DB types into JSON-safe primitives (Decimal→float, datetime→iso)."""
    # Decimal (numeric column) — keep precision as float for the prompt.
    if value is not None and value.__class__.__name__ == "Decimal":
        return float(value)
    # datetime / date — isoformat string.
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return value


# ---------------------------------------------------------------------------
# Prompt block renderer
# ---------------------------------------------------------------------------


def format_about_user_block(prefs: Optional[dict]) -> str:
    """Render preferences into the `[about_user]` prompt block.

    Rules:
      - Empty / None prefs → "" (no block; the prompt placeholder collapses).
      - Occupation (identity) + AI style (non-sensitive) ALWAYS render.
      - Financial context renders ONLY when `memory_consent` is true (PDPA).
      - NULL/blank fields are skipped so the LLM never sees empty slots to
        hallucinate into.

    Returns a string that starts with a blank line + heading, or "".
    """
    if not prefs:
        return ""

    lines: list[str] = []

    occupation = _clean(prefs.get("occupation"))
    if occupation:
        lines.append(f"- {_LABELS['occupation']}: {occupation}")

    consent = bool(prefs.get("memory_consent"))
    if consent:
        any_ai_inferred = _has_ai_inferred(prefs.get("field_sources"))
        for field in _FINANCIAL_ORDER:
            rendered = _render_value(field, prefs.get(field))
            if rendered is None:
                continue
            lines.append(f"- {_LABELS[field]}: {rendered}")

    ai_line = _format_ai_style(prefs.get("ai_preferences"))
    if ai_line:
        lines.append(f"- AI style: {ai_line}")

    if not lines:
        return ""

    header = (
        "# ABOUT THIS USER\n"
        "Durable context we remember about this user. Prefer it over re-asking. "
        "User-stated facts outrank AI-inferred ones; never present an inferred "
        "value as confirmed when the decision is irreversible."
    )
    block = header + "\n" + "\n".join(lines)

    if consent and _has_ai_inferred(prefs.get("field_sources")):
        block += (
            "\n(Some values above were AI-inferred and not yet confirmed by the "
            "user — verify before relying on them for major decisions.)"
        )
    return "\n" + block


def _render_value(field: str, value: Any) -> Optional[str]:
    """Stringify one financial field, or None to skip (null/blank)."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    if field == "declared_monthly_income":
        # Show as a plain number with thousands separators.
        try:
            return f"{float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _format_ai_style(ai_prefs: Any) -> str:
    """Compact one-line summary of the ai_preferences jsonb."""
    if not isinstance(ai_prefs, dict) or not ai_prefs:
        return ""
    parts = []
    for key, val in ai_prefs.items():
        if val is None or val == "":
            continue
        parts.append(f"{key}={val}")
    return ", ".join(parts)


def _has_ai_inferred(field_sources: Any) -> bool:
    if not isinstance(field_sources, dict):
        return False
    return any(v == "ai_inferred" for v in field_sources.values())


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# Module-level pool / loader registry (mirrors entity_catalog's pattern).
# ---------------------------------------------------------------------------

_PREFERENCES_POOL: Any = None
_PREFERENCES_LOADER: Any = None


def set_preferences_pool(pool) -> None:
    """Install the backend Postgres pool (FastAPI lifespan). Same pool object
    as the entity catalog's — both read from BACKEND_DATABASE_URL."""
    global _PREFERENCES_POOL
    _PREFERENCES_POOL = pool


def set_preferences_loader(loader) -> None:
    """Install a custom `async (user_id) -> Optional[dict]` loader for tests/DI.
    Takes precedence over the pool. Pass None to clear."""
    global _PREFERENCES_LOADER
    _PREFERENCES_LOADER = loader


async def load_user_preferences(user_id: str) -> Optional[dict]:
    """Top-level loader used by the pre-turn hook + the write tool.

    Resolution order: test loader → pool-backed loader → None. Never raises.
    """
    if _PREFERENCES_LOADER is not None:
        return await _PREFERENCES_LOADER(user_id)
    if _PREFERENCES_POOL is not None:
        return await load_user_preferences_from_pool(_PREFERENCES_POOL, user_id)
    return None


def get_preferences_pool() -> Any:
    """Expose the installed pool to the write tool (which needs to upsert)."""
    return _PREFERENCES_POOL


__all__ = [
    "FINANCIAL_FIELDS",
    "load_user_preferences",
    "load_user_preferences_from_pool",
    "format_about_user_block",
    "set_preferences_pool",
    "set_preferences_loader",
    "get_preferences_pool",
]
