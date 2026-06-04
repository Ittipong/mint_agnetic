"""`set_user_preference` — persist one durable fact/preference about the user.

Companion to the read path (`src/agent/user_preferences.py` + the `[about_user]`
prompt block). The agent calls this AFTER learning something durable and
non-derivable about the user (literacy, housing, dependents, risk tolerance,
salary day, declared figures) or to record an AI-style preference.

Hard rules baked in:
  - PDPA consent gate: a PERSONAL/financial field can only be written when
    `memory_consent` is already true. If not, the tool refuses and tells the
    agent to ask for consent (and set `memory_consent`) first.
  - Provenance: every write stamps `field_sources[field] = source`
    ("user_stated" | "ai_inferred", default ai_inferred). `memory_consent` is
    always recorded as user_stated — consent can only come from the user.
  - DON'T STORE DERIVABLE DATA: income/debt/goal/budget numbers are computed
    live by `run_python` from transactions — they are NOT writable here. The
    field whitelist enforces this; an off-list field is rejected.

Never raises out — store/validation failures degrade to a structured
{"error": ..., "kind": ...} dict.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog, slog_error
from src.agent.user_preferences import (
    FINANCIAL_FIELDS,
    get_preferences_pool,
    load_user_preferences,
)


_STATUS_WORD = "กำลังจดจำ..."

# ── Field taxonomy + validation ─────────────────────────────────────────────
# Enum columns → allowed values (mirror the DB CHECK constraints exactly).
_ENUM_VALUES: dict[str, set[str]] = {
    "income_stability": {"fixed_salary", "freelance", "irregular"},
    "housing_status": {"rent", "mortgage", "own", "with_family"},
    "financial_literacy_level": {"beginner", "intermediate", "advanced"},
    "primary_goal_priority": {
        "pay_debt", "emergency_fund", "save_house", "invest", "general",
    },
    "debt_payoff_strategy": {"avalanche", "snowball", "none"},
    "risk_tolerance": {"conservative", "moderate", "aggressive"},
    "life_stage": {"student", "single", "family", "retired"},
}
# Integer columns → (min, max) inclusive.
_INT_RANGES: dict[str, tuple[int, int]] = {
    "salary_day": (1, 31),
    "dependents_count": (0, 50),
    "emergency_fund_target_months": (0, 120),
}
_NUMERIC_FIELDS: set[str] = {"declared_monthly_income"}
_TEXT_FIELDS: set[str] = {"financial_notes"}
# Meta booleans the agent may set — NOT consent-gated.
_BOOL_META_FIELDS: set[str] = {"memory_consent", "onboarding_completed"}
# AI-style keys → stored inside the `ai_preferences` jsonb (NOT consent-gated).
_AI_STYLE_KEYS: set[str] = {
    "ai_tone", "ai_response_length", "ai_language", "ai_proactivity",
    "use_emoji", "coaching_frequency", "preferred_checkin_time", "avoid_topics",
}

_VALID_SOURCES = {"user_stated", "ai_inferred"}

# Columns that live as top-level table columns (whitelist for SQL identifier
# interpolation — every value here is a hard-coded literal, never user input).
_COLUMN_FIELDS: set[str] = (
    set(_ENUM_VALUES) | set(_INT_RANGES) | _NUMERIC_FIELDS | _TEXT_FIELDS
    | _BOOL_META_FIELDS
)


@tool
async def set_user_preference(
    field: str,
    value: Any,
    source: str = "ai_inferred",
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Remember a durable, non-derivable preference/fact about the user.

    Use AFTER you learn something stable that has NO transaction behind it:
    financial literacy, housing status, dependents, risk tolerance, salary
    day, life stage, a self-stated income figure, or an AI-style preference.
    DO NOT use this for numbers you can compute from transactions (actual
    income, debt balance, savings, spending) — derive those with run_python.

    Personal/financial fields require `memory_consent=true` first; if the user
    hasn't consented, this returns a consent_required error — ask them, then
    call set_user_preference(field="memory_consent", value=true).

    Args:
      field: One of the financial-context fields, "memory_consent",
        "onboarding_completed", or an ai_* style key (ai_tone,
        ai_response_length, ai_language, ai_proactivity, use_emoji,
        coaching_frequency, preferred_checkin_time, avoid_topics).
      value: The value (validated against the field's type/enum/range).
      source: "user_stated" when the user told you directly, "ai_inferred"
        when you deduced it (default). Ignored for memory_consent (always
        user_stated).

    Returns:
      Command(update={"user_preferences": <refreshed>, "messages": [ToolMessage]})
      The ToolMessage body is {"ok": true, "field": ...} or {"error", "kind"}.
    """
    _emit_status(_STATUS_WORD)

    user_id = state.get("user_id")
    if not user_id:
        return _err(tool_call_id, "state.user_id missing", "invalid_state")

    kind, coerced, msg = classify_and_coerce(field, value)
    if kind == "invalid":
        return _err(tool_call_id, msg, "invalid_field")
    if kind == "invalid_value":
        return _err(tool_call_id, msg, "invalid_value")

    if source not in _VALID_SOURCES:
        source = "ai_inferred"
    # Consent is always the user's own act, never an AI inference.
    if field == "memory_consent":
        source = "user_stated"

    # PDPA gate — personal/financial fields need prior consent.
    if field in FINANCIAL_FIELDS and not _has_consent(state):
        return _err(
            tool_call_id,
            "Cannot store personal financial context without consent. Ask the "
            "user if they're OK with Mint Money remembering this, then call "
            "set_user_preference(field='memory_consent', value=true) before "
            "retrying.",
            "consent_required",
        )

    pool = get_preferences_pool()
    if pool is None:
        return _err(tool_call_id, "preferences store unavailable", "store_unavailable")

    try:
        await _upsert(pool, user_id, field, coerced, source, is_ai_style=(kind == "ai_style"))
    except Exception as exc:  # noqa: BLE001
        slog_error("set_user_preference", exc)
        return _err(tool_call_id, str(exc), "store_write_failed")

    refreshed = await load_user_preferences(user_id)
    slog("set_user_preference", f"set {field}={coerced!r} source={source}")
    return Command(update={
        "user_preferences": refreshed,
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {"ok": True, "field": field, "source": source},
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call_id,
            ),
        ],
    })


# ── Validation (pure, unit-testable without a DB) ───────────────────────────


def classify_and_coerce(field: str, value: Any) -> tuple[str, Any, str]:
    """Classify a field and coerce its value.

    Returns (kind, coerced, message):
      kind ∈ {"column", "ai_style", "invalid", "invalid_value"}.
      On an "invalid*" kind, `message` explains why; coerced is None.
    """
    if not isinstance(field, str) or not field.strip():
        return "invalid", None, "field must be a non-empty string"
    field = field.strip()

    if field in _AI_STYLE_KEYS:
        return _coerce_ai_style(field, value)
    if field in _COLUMN_FIELDS:
        return _coerce_column(field, value)

    allowed = sorted(_COLUMN_FIELDS | _AI_STYLE_KEYS)
    return (
        "invalid", None,
        f"unknown field {field!r}. Allowed: {allowed}. (Derivable numbers like "
        f"actual income/debt/savings are NOT stored here — use run_python.)",
    )


def _coerce_column(field: str, value: Any) -> tuple[str, Any, str]:
    if field in _ENUM_VALUES:
        v = str(value).strip()
        if v not in _ENUM_VALUES[field]:
            return ("invalid_value", None,
                    f"{field} must be one of {sorted(_ENUM_VALUES[field])}, got {value!r}")
        return "column", v, ""
    if field in _INT_RANGES:
        lo, hi = _INT_RANGES[field]
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return "invalid_value", None, f"{field} must be an integer, got {value!r}"
        if not (lo <= iv <= hi):
            return "invalid_value", None, f"{field} must be {lo}..{hi}, got {iv}"
        return "column", iv, ""
    if field in _NUMERIC_FIELDS:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return "invalid_value", None, f"{field} must be a number, got {value!r}"
        if fv < 0:
            return "invalid_value", None, f"{field} must be >= 0, got {fv}"
        return "column", fv, ""
    if field in _BOOL_META_FIELDS:
        bv = _coerce_bool(value)
        if bv is None:
            return "invalid_value", None, f"{field} must be true/false, got {value!r}"
        return "column", bv, ""
    if field in _TEXT_FIELDS:
        sv = str(value).strip()
        if not sv:
            return "invalid_value", None, f"{field} must be non-empty text"
        return "column", sv[:500], ""  # defensive cap
    return "invalid", None, f"unhandled column {field!r}"


def _coerce_ai_style(field: str, value: Any) -> tuple[str, Any, str]:
    """AI-style values are stored as-is in jsonb; only `use_emoji` is bool."""
    if field == "use_emoji":
        bv = _coerce_bool(value)
        if bv is None:
            return "invalid_value", None, f"{field} must be true/false, got {value!r}"
        return "ai_style", bv, ""
    if value is None:
        return "invalid_value", None, f"{field} must not be null"
    if isinstance(value, (str, int, float, bool)):
        v = value.strip() if isinstance(value, str) else value
        if v == "":
            return "invalid_value", None, f"{field} must be non-empty"
        return "ai_style", v, ""
    return "invalid_value", None, f"{field} must be a scalar, got {type(value).__name__}"


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _has_consent(state: dict) -> bool:
    prefs = state.get("user_preferences")
    return bool(isinstance(prefs, dict) and prefs.get("memory_consent"))


# ── DB upsert ────────────────────────────────────────────────────────────────


async def _upsert(
    pool, user_id: str, field: str, value: Any, source: str, *, is_ai_style: bool,
) -> None:
    """Upsert one field. `field` is always from a hard-coded whitelist, so the
    identifier interpolation below is injection-safe."""
    sources_json = json.dumps({field: source})
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if is_ai_style:
                style_json = json.dumps({field: value})
                await cur.execute(
                    "INSERT INTO user_preferences "
                    "  (user_id, ai_preferences, field_sources, updated_at) "
                    "VALUES (%s, %s::jsonb, %s::jsonb, now()) "
                    "ON CONFLICT (user_id) DO UPDATE SET "
                    "  ai_preferences = user_preferences.ai_preferences || EXCLUDED.ai_preferences, "
                    "  field_sources = user_preferences.field_sources || EXCLUDED.field_sources, "
                    "  updated_at = now()",
                    (user_id, style_json, sources_json),
                )
            else:
                # `field` is whitelisted in _COLUMN_FIELDS → safe to interpolate.
                sql = (
                    f'INSERT INTO user_preferences '
                    f'  (user_id, "{field}", field_sources, last_confirmed_at, updated_at) '
                    f"VALUES (%s, %s, %s::jsonb, now(), now()) "
                    f"ON CONFLICT (user_id) DO UPDATE SET "
                    f'  "{field}" = EXCLUDED."{field}", '
                    f"  field_sources = user_preferences.field_sources || EXCLUDED.field_sources, "
                    f"  last_confirmed_at = now(), "
                    f"  updated_at = now()"
                )
                await cur.execute(sql, (user_id, value, sources_json))


# ── Command / status helpers ─────────────────────────────────────────────────


def _err(tool_call_id: str, error: str, kind: str) -> Command:
    return Command(update={
        "messages": [
            ToolMessage(
                content=json.dumps({"error": error, "kind": kind}, ensure_ascii=False),
                tool_call_id=tool_call_id,
            ),
        ],
    })


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


__all__ = ["set_user_preference", "classify_and_coerce"]
