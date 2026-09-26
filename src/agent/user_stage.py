"""How far along a user is with the app — deterministic, loaded once per turn.

Why: a brand-new user (onboarding leaves exactly one wallet, zero
transactions) was answered as if they had months of history. "Emergency fund =
3–6 × 0 = 0 บาท", "average income 11,667" (one salary ÷ 3 months), and
"no data yet, try recording" dead ends. The agent cannot tell a new user from
a quiet one by itself; this module tells it, and the NEW USER PLAYBOOK in the
system prompt says what to do at each stage.

Stages:
    new          no confirmed transaction yet
    starting     first transaction < STARTING_DAYS days ago (too little
                 history for monthly averages or trends)
    established  enough history — no special handling
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from src.agent.session_logger import slog
from src.agent.user_preferences import get_preferences_pool

STAGE_NEW = "new"
STAGE_STARTING = "starting"
STAGE_ESTABLISHED = "established"
STARTING_DAYS = 30

_SQL = """
    SELECT
      (SELECT count(*) FROM transactions t
        WHERE t.created_by_user_id = %s AND t.is_deleted = false
          AND t.status = 'confirmed')                                   AS tx_count,
      (SELECT min((t.date AT TIME ZONE 'Asia/Bangkok')::date) FROM transactions t
        WHERE t.created_by_user_id = %s AND t.is_deleted = false
          AND t.status = 'confirmed')                                   AS first_tx_date,
      (SELECT count(*) FROM transactions t
        WHERE t.created_by_user_id = %s AND t.is_deleted = false
          AND t.status = 'confirmed' AND t.type = 'income')             AS income_count,
      (SELECT count(*) FROM general_wallets w
        WHERE w.user_id = %s AND w.deleted_at IS NULL)
      + (SELECT count(*) FROM creditcard_wallets w
        WHERE w.user_id = %s AND w.deleted_at IS NULL)
      + (SELECT count(*) FROM goal_wallets w
        WHERE w.user_id = %s AND w.deleted_at IS NULL AND w.is_deleted = false) AS wallet_count
"""

_LOADER: Any = None  # tests inject an async (user_id) -> dict | None


def set_user_stage_loader(loader) -> None:
    global _LOADER
    _LOADER = loader


def compute_stage(tx_count: int, first_tx_date: Optional[date], today: date) -> str:
    if not tx_count:
        return STAGE_NEW
    if first_tx_date is None or (today - first_tx_date).days < STARTING_DAYS:
        return STAGE_STARTING
    return STAGE_ESTABLISHED


async def load_user_stage(user_id: str, today: Optional[date] = None) -> Optional[dict]:
    """{stage, tx_count, history_days, has_income, wallet_count} or None.
    Never raises — a failed load just means the playbook does not apply."""
    if not user_id:
        return None
    today = today or date.today()
    try:
        if _LOADER is not None:
            raw = await _LOADER(user_id)
        else:
            pool = get_preferences_pool()
            if pool is None:
                return None
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(_SQL, (user_id,) * 6)
                    row = await cur.fetchone()
            raw = {"tx_count": row[0], "first_tx_date": row[1],
                   "income_count": row[2], "wallet_count": row[3]} if row else None
    except Exception as exc:  # noqa: BLE001 — must not crash a turn
        slog("user_stage", f"load failed: {type(exc).__name__}: {exc}")
        return None
    if not raw:
        return None
    first = raw.get("first_tx_date")
    if isinstance(first, datetime):
        first = first.date()
    elif isinstance(first, str):
        first = date.fromisoformat(first[:10])
    tx_count = int(raw.get("tx_count") or 0)
    stage = compute_stage(tx_count, first, today)
    data = {
        "stage": stage,
        "tx_count": tx_count,
        "history_days": (today - first).days + 1 if first else 0,
        "has_income": bool(raw.get("income_count")),
        "wallet_count": int(raw.get("wallet_count") or 0),
    }
    slog("user_stage", f"{data}")
    return data


def format_user_stage_block(info: Optional[dict]) -> str:
    """One line for the END of the system prompt (the cached prefix stays the
    same). `established` renders nothing — the playbook only covers early users."""
    if not info or info.get("stage") == STAGE_ESTABLISHED:
        return ""
    return (
        "\n\n# USER STAGE (this turn — follow the NEW USER PLAYBOOK)\n"
        f"stage={info['stage']} · confirmed transactions={info['tx_count']} · "
        f"history={info['history_days']} day(s) · income recorded="
        f"{'yes' if info['has_income'] else 'no'} · wallets={info['wallet_count']}"
    )


__all__ = [
    "STAGE_ESTABLISHED",
    "STAGE_NEW",
    "STAGE_STARTING",
    "compute_stage",
    "format_user_stage_block",
    "load_user_stage",
    "set_user_stage_loader",
]
