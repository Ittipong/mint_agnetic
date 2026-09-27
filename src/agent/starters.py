"""Starter chips for the empty chat screen — personal, deterministic, fast.

Why: the empty chat showed the same ~30 canned prompts to everyone ("เติม
น้ำมัน 700" to people without a car, "ใช้จ่ายเกินงบไหม" to people without a
budget). The empty screen is the moment the user decides what to type, so a
chip that matches their life turns a record into one tap — the product's
core promise. No LLM: this has to render instantly on every chat open, and
rule-built chips are always correct.

Signals, in order:
  log  — what THIS user records at this time of day, with their usual amount
         ("กาแฟ 65" in the morning); skipped once already recorded today.
         A monthly item (salary, rent, a subscription) shows only around its
         usual day of the month, and only while this month's is not recorded.
  ask  — questions tied to their money right now: a card due within
         CARD_DUE_DAYS, a budget at ≥ BUDGET_WARN_PCT.
  fill — up to MIN_ITEMS: questions about their own data when they have
         history, generic examples when they have none.

Each item carries `kind` and `reason`, so taps can be measured per signal
once the client echoes it back (`origin` on /chat/stream). Record chips (log,
example) also carry the category (`category_id` = the user's category sync_id,
`category_name`) so the client can draw the category icon, and every chip may
carry `hint` — the one line that says why it is shown ("ใช้ไป 103% ของงบ").
`label` is always exactly what `send` sends: a chip that shows one sentence
and sends another reads as the app putting words in the user's mouth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

BKK = ZoneInfo("Asia/Bangkok")
LOOKBACK_DAYS = 90
MIN_OCCURRENCES = 3
HOUR_WINDOW = 2
CARD_DUE_DAYS = 5
BUDGET_WARN_PCT = 80
MAX_LOG = 3
MAX_ASK = 2
MIN_ITEMS = 3
MONTHLY_MAX_PER_MONTH = 1.5   # ≤ this many per month = a monthly item

EXAMPLE_HINT = "ตัวอย่าง · แตะแล้วแก้ยอดได้"
# category_name = a default category every new user has; build_starters swaps in
# that user's own sync_id (a user who renamed it just gets no icon).
EXAMPLES = [
    {"label": "กาแฟ 60", "send": "กาแฟ 60", "kind": "example", "reason": "example",
     "category_id": None, "category_name": "กาแฟ", "hint": EXAMPLE_HINT},
    {"label": "ข้าวเที่ยง 80", "send": "ข้าวเที่ยง 80", "kind": "example", "reason": "example",
     "category_id": None, "category_name": "ร้านอาหาร", "hint": EXAMPLE_HINT},
    {"label": "เดือนนี้ใช้ไปเท่าไหร่", "send": "เดือนนี้ใช้ไปเท่าไหร่", "kind": "example",
     "reason": "example", "hint": None},
    {"label": "Nimo ทำอะไรได้บ้าง", "send": "Nimo ทำอะไรได้บ้าง", "kind": "example",
     "reason": "example", "hint": None},
]


@dataclass
class TxRow:
    type: str             # expense | income
    amount: Decimal
    label: str            # the user's note, else the category name
    logged_at: datetime   # when it was recorded (local) — habit hour
    tx_date: date         # the transaction's own date (local)
    category: str = ""    # one chip per category keeps the row varied
    category_id: str = "" # the category's sync_id — the client draws its icon


def _fmt_amount(a: Decimal) -> str:
    a = Decimal(a)
    return f"{int(a):,}" if a == a.to_integral_value() else f"{a:,.2f}"


def _hour_distance(a: int, b: int) -> int:
    d = abs(a - b) % 24
    return min(d, 24 - d)


def habit_chips(rows: list[TxRow], now: datetime) -> list[dict]:
    """Rank the user's recurring records for this moment. Pure — no I/O."""
    today = now.date()
    groups: dict[str, list[TxRow]] = defaultdict(list)
    for r in rows:
        key = " ".join(r.label.split()).lower()
        if key:
            groups[key].append(r)

    scored = []
    for key, items in groups.items():
        if len(items) < MIN_OCCURRENCES:
            continue
        kind = Counter(i.type for i in items).most_common(1)[0][0]
        latest = max(items, key=lambda i: i.logged_at)
        amounts = Counter(i.amount for i in items)
        top = max(amounts.values())
        # Most common amount; on a tie the most recent one wins.
        usual = next(i.amount for i in sorted(items, key=lambda i: i.logged_at, reverse=True)
                     if amounts[i.amount] == top)
        months = {(i.tx_date.year, i.tx_date.month) for i in items}
        monthly = len(items) / len(months) <= MONTHLY_MAX_PER_MONTH
        if monthly:
            # Salary, rent, a subscription: offer it around its usual day of the
            # month, once — not every night because it was charged at 03:00.
            if (today.year, today.month) in months:
                continue
            usual_day = Counter(i.tx_date.day for i in items).most_common(1)[0][0]
            if abs(today.day - usual_day) > 1:
                continue
            score, why = 100 + len(items), f"monthly:day{usual_day}"
            hint = f"ปกติจดวันที่ {usual_day}"
        else:
            if any(i.tx_date == today for i in items):
                continue
            in_window = sum(1 for i in items
                            if _hour_distance(i.logged_at.hour, now.hour) <= HOUR_WINDOW)
            if in_window == 0:
                continue
            score, why = in_window * 3 + len(items), f"habit:{in_window}/{len(items)}@{now.hour}h"
            hint = "ปกติจดเวลานี้"
        text = f"{latest.label} {_fmt_amount(usual)}"
        if kind == "income":
            text = f"ได้{latest.label} {_fmt_amount(usual)}" if not latest.label.startswith("ได้") else text
        cat, cat_id = Counter((i.category, i.category_id) for i in items).most_common(1)[0][0]
        scored.append((score, {"label": text, "send": text, "kind": "log", "reason": why,
                               "category_id": cat_id or None, "category_name": cat or None,
                               "hint": hint}, cat))

    # Best chip per category: three coffee shops in the morning is one choice.
    scored.sort(key=lambda s: s[0], reverse=True)
    out, used_cats = [], set()
    for _, chip, cat in scored:
        if cat and cat in used_cats:
            continue
        used_cats.add(cat)
        out.append(chip)
        if len(out) >= MAX_LOG:
            break
    return out


def ask_chips(cards: list[dict], budgets: list[dict], today: date) -> list[dict]:
    """Questions about the user's money right now. Pure — no I/O."""
    from src.agent.tools.codeact.namespace import _day_of_month_on_or_after

    out: list[tuple[int, dict]] = []
    for c in cards:
        due = _day_of_month_on_or_after(today, c.get("payment_due_day"))
        used = Decimal(str(c.get("used") or 0))
        if due is None or used <= 0:
            continue
        days = (due - today).days
        if days <= CARD_DUE_DAYS:
            name = c["name"]
            when = "วันนี้" if days == 0 else f"อีก {days} วัน"
            text = f"{name} รอบนี้ต้องจ่ายเท่าไหร่"
            out.append((50 - days, {
                "label": text, "send": text, "kind": "ask", "reason": f"card_due:{when}",
                "hint": "ครบกำหนดวันนี้" if days == 0 else f"ครบกำหนดใน {days} วัน"}))
    for b in budgets:
        pct = Decimal(str(b.get("pct_used") or 0))
        if pct < BUDGET_WARN_PCT:
            continue
        name = b["name"] if str(b["name"]).startswith("งบ") else f"งบ{b['name']}"
        text = f"{name}เดือนนี้เกินไปเท่าไหร่" if pct >= 100 else f"{name}เดือนนี้เหลือเท่าไหร่"
        out.append((int(pct), {"label": text, "send": text, "kind": "ask",
                               "reason": f"budget:{int(pct)}%",
                               "hint": f"ใช้ไป {int(pct)}% ของงบ"}))
    out.sort(key=lambda s: s[0], reverse=True)
    return [c for _, c in out[:MAX_ASK]]


# Fill for a user who HAS history but no habit fits this hour: questions about
# their own data beat a generic "กาแฟ 60".
HISTORY_FILL = [
    {"label": "สัปดาห์นี้ใช้ไปเท่าไหร่", "send": "สัปดาห์นี้ใช้ไปเท่าไหร่", "kind": "ask",
     "reason": "fill:history", "hint": None},
    {"label": "เดือนนี้ใช้ไปกับอะไรบ้าง", "send": "เดือนนี้ใช้ไปกับอะไรบ้าง", "kind": "ask",
     "reason": "fill:history", "hint": None},
]


def assemble(log: list[dict], ask: list[dict], limit: int,
             has_history: bool = False) -> list[dict]:
    items = (log + ask)[:limit]
    seen = {i["send"] for i in items}
    for e in (HISTORY_FILL if has_history else EXAMPLES):
        if len(items) >= min(MIN_ITEMS, limit):
            break
        if e["send"] not in seen:
            items.append(dict(e))
    return items


# ── DB ─────────────────────────────────────────────────────────────────────

_TX_SQL = """
    SELECT t.type, t.amount::numeric AS amount,
           COALESCE(NULLIF(btrim(t.note), ''), t.category_name, '') AS label,
           (t.created_at AT TIME ZONE 'Asia/Bangkok') AS logged_at,
           (t.date AT TIME ZONE 'Asia/Bangkok')::date AS tx_date,
           COALESCE(t.category_name, '') AS category,
           COALESCE(t.category_sync_id::text, '') AS category_id
    FROM transactions t
    WHERE t.created_by_user_id = $1::uuid AND t.is_deleted = false
      AND t.status = 'confirmed' AND t.type IN ('expense', 'income')
      AND t.date >= $2::date
    ORDER BY t.created_at DESC
    LIMIT 2000
"""
_CARD_SQL = """
    SELECT name, payment_due_day, cached_used_amount::numeric AS used
    FROM creditcard_wallets WHERE user_id = $1::uuid AND deleted_at IS NULL
"""
_CAT_SQL = """
    SELECT name, sync_id::text AS sync_id FROM categories
    WHERE user_id = $1::uuid AND deleted_at IS NULL AND type = 'expense'
"""


def resolve_example_categories(items: list[dict], cats: dict[str, str]) -> list[dict]:
    """Point example chips at THIS user's category of the same name."""
    for i in items:
        if i["kind"] == "example" and i.get("category_name"):
            i["category_id"] = cats.get(i["category_name"])
    return items


async def build_starters(user_id: str, *, limit: int = 5,
                         now: Optional[datetime] = None) -> list[dict]:
    """Starter chips for `user_id`. Never raises — on any failure it returns
    the examples, so the empty screen always has something to tap."""
    from src.agent.session_logger import slog
    from src.agent.tools.codeact.db import get_pool
    from src.agent.tools.codeact.namespace import _run_query
    from src.agent.tools.codeact.schemas import QuerySpec, TimeRange

    now = now or datetime.now(BKK)
    today = now.date()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            tx = await conn.fetch(_TX_SQL, user_id, today - timedelta(days=LOOKBACK_DAYS))
            cards = [dict(r) for r in await conn.fetch(_CARD_SQL, user_id)]
            cats = {r["name"]: r["sync_id"] for r in await conn.fetch(_CAT_SQL, user_id)}
        rows = [TxRow(r["type"], Decimal(r["amount"]), r["label"] or "",
                      r["logged_at"], r["tx_date"], r["category"], r["category_id"]) for r in tx]
        budgets = await _run_query(QuerySpec(
            metric="budget_remaining",
            time_range=TimeRange(start=today, end=today, granularity="day", confidence=1.0),
        ), user_id)
        items = assemble(habit_chips(rows, now), ask_chips(cards, budgets, today), limit,
                         has_history=len(rows) >= MIN_OCCURRENCES)
        resolve_example_categories(items, cats)
    except Exception as exc:  # noqa: BLE001 — chips are nice-to-have
        slog("starters", f"failed, serving examples: {type(exc).__name__}: {exc}")
        items = [dict(e) for e in EXAMPLES[:limit]]
    slog("starters", f"user={user_id[:8]} items={[(i['kind'], i['label']) for i in items]}")
    return items


__all__ = ["EXAMPLES", "HISTORY_FILL", "TxRow", "ask_chips", "assemble", "build_starters", "habit_chips",
           "resolve_example_categories"]
