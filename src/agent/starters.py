"""Starter chips for the empty chat screen — personal records, example questions.

Why: the empty chat showed the same ~30 canned prompts to everyone ("เติม
น้ำมัน 700" to people without a car). The empty screen is the moment the user
decides what to type, so a record chip that matches their life turns logging
into one tap — the product's core promise. No LLM: this has to render
instantly on every chat open, and rule-built chips are always correct.

Two parts:
  records   — what THIS user records at this time of day, with their usual
              amount ("กาแฟ 65" in the morning); skipped once already recorded
              today. A monthly item (salary, rent, a subscription) shows only
              around its usual day, and only while this month's is not
              recorded. A user with no history gets two example records.
  questions — a long, fixed list of advisor questions (owner, 2026-09-27 /
              09-28): shows the range of what Nimo can help with — judgment,
              what-ifs, plans, not lookups the screens already show. Same
              for everyone, quotes none of the user's numbers.

Each item carries `kind` and `reason`, so taps can be measured per signal
once the client echoes it back (`origin` on /chat/stream). Record chips (log,
example) also carry the category (`category_id` = the user's category sync_id,
`category_name`) so the client can draw the category icon, and a `hint` — the
one line that says why it is shown ("ปกติจดเวลานี้"). `label` is always
exactly what `send` sends: a chip that shows one sentence and sends another
reads as the app putting words in the user's mouth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

BKK = ZoneInfo("Asia/Bangkok")
LOOKBACK_DAYS = 90
MIN_OCCURRENCES = 3
HOUR_WINDOW = 2
MAX_LOG = 3
MONTHLY_MAX_PER_MONTH = 1.5   # ≤ this many per month = a monthly item

EXAMPLE_HINT = "ตัวอย่าง · แตะแล้วแก้ยอดได้"
FREQUENT_HINT = "จดบ่อย"
FREQUENT_DAYS = 30
# category_name = a default category every new user has; build_starters swaps in
# that user's own sync_id (a user who renamed it just gets no icon).
EXAMPLES = [
    {"label": "กาแฟ 60", "send": "กาแฟ 60", "kind": "example", "reason": "example",
     "category_id": None, "category_name": "กาแฟ", "hint": EXAMPLE_HINT},
    {"label": "ข้าวเที่ยง 80", "send": "ข้าวเที่ยง 80", "kind": "example", "reason": "example",
     "category_id": None, "category_name": "ร้านอาหาร", "hint": EXAMPLE_HINT},
]

# Advisor questions in a friend's voice (owner, 2026-09-28: Nimo is a money
# advisor and a friend who never judges). The app's screens already show
# totals and lists, so the empty chat shows what only a conversation can:
# judgment, what-ifs, plans, a heads-up. Still the same for everyone and no
# amounts or presumptions ("มีหนี้บัตรหลายใบ") — they must read right for any
# user. "Nimo ทำอะไรได้บ้าง" stays last (question_chips moves it first for a
# user with no history).
QUESTIONS = [
    "เดือนนี้ใช้เงินโอเคไหม",
    "เงินจะพอใช้ถึงสิ้นเดือนไหม",
    "อยากเก็บเงินเพิ่ม ควรเริ่มลดตรงไหน",
    "เงินเดือนหายไปไหนหมด",
    "เดือนนี้มีอะไรผิดปกติไหม",
    "ฉันชอบใช้เงินหนักช่วงไหน",
    "จ่ายบัตรเต็มหรือขั้นต่ำดี",
    "ควรมีเงินสำรองเท่าไหร่ถึงจะอุ่นใจ",
    "เป้าเก็บเงินจะทันไหม",
    "ถ้าเก็บเพิ่มเดือนละนิด อีกปีจะมีเท่าไหร่",
    "งบเดือนนี้ยังไหวไหม",
    "เงินเดือนออกแล้ว แบ่งใช้แบ่งเก็บยังไงดี",
    "ช่วยดูหน่อย การเงินตอนนี้เป็นไงบ้าง",
    "Nimo ทำอะไรได้บ้าง",
]


def question_chips(has_history: bool) -> list[dict]:
    """The example questions; a user with no data meets "what can you do" first."""
    qs = QUESTIONS if has_history else [QUESTIONS[-1], *QUESTIONS[:-1]]
    return [{"label": q, "send": q, "kind": "ask", "reason": "example", "hint": None} for q in qs]


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


def frequent_chips(rows: list[TxRow], now: datetime) -> list[dict]:
    """The user's most-recorded items of the last 30 days, best first.

    Why: habit chips only match records made within ±2 h of now, so a user
    opening the chat at 1 AM (or any hour without a habit) got NO record card
    at all — the logging shortcut, the product's core promise, vanished
    (owner 2026-09-28). Monthly items (salary, rent) and anything already
    recorded today are left out; one chip per category. Pure — no I/O.
    """
    today = now.date()
    since = today - timedelta(days=FREQUENT_DAYS)
    groups: dict[str, list[TxRow]] = defaultdict(list)
    for r in rows:
        key = " ".join(r.label.split()).lower()
        if key:
            groups[key].append(r)
    scored = []
    for items in groups.values():
        recent = [i for i in items if i.tx_date >= since]
        if len(recent) < MIN_OCCURRENCES or any(i.tx_date == today for i in items):
            continue
        months = {(i.tx_date.year, i.tx_date.month) for i in items}
        if len(items) / len(months) <= MONTHLY_MAX_PER_MONTH:
            continue  # a monthly bill — offered around its day by habit_chips
        latest = max(items, key=lambda i: i.logged_at)
        amounts = Counter(i.amount for i in recent)
        top = max(amounts.values())
        usual = next(i.amount for i in sorted(recent, key=lambda i: i.logged_at, reverse=True)
                     if amounts[i.amount] == top)
        text = f"{latest.label} {_fmt_amount(usual)}"
        if Counter(i.type for i in items).most_common(1)[0][0] == "income" \
                and not latest.label.startswith("ได้"):
            text = f"ได้{text}"
        cat, cat_id = Counter((i.category, i.category_id) for i in items).most_common(1)[0][0]
        scored.append((len(recent), {"label": text, "send": text, "kind": "log",
                                     "reason": f"frequent:{len(recent)}/{FREQUENT_DAYS}d",
                                     "category_id": cat_id or None, "category_name": cat or None,
                                     "hint": FREQUENT_HINT}))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [chip for _, chip in scored]


def assemble(log: list[dict], limit: int = MAX_LOG, has_history: bool = False,
             frequent: Optional[list[dict]] = None) -> list[dict]:
    """Record card, then every example question.

    The card always has something to tap (owner 2026-09-28), filled in this
    order up to `limit`: habits for this hour → the user's frequent records →
    the system examples. One chip per category/label."""
    records: list[dict] = []
    seen: set[str] = set()
    for chip in [*log, *(frequent or []), *(dict(e) for e in EXAMPLES)]:
        key = chip.get("category_name") or chip["label"]
        if key in seen:
            continue
        seen.add(key)
        records.append(chip)
        if len(records) >= limit:
            break
    return records + question_chips(has_history)


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


async def build_starters(user_id: str, *, limit: int = MAX_LOG,
                         now: Optional[datetime] = None) -> list[dict]:
    """Starter chips for `user_id`. Never raises — on any failure it returns
    the examples, so the empty screen always has something to tap."""
    from src.agent.session_logger import slog
    from src.agent.tools.codeact.db import get_pool

    now = now or datetime.now(BKK)
    today = now.date()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            tx = await conn.fetch(_TX_SQL, user_id, today - timedelta(days=LOOKBACK_DAYS))
            cats = {r["name"]: r["sync_id"] for r in await conn.fetch(_CAT_SQL, user_id)}
        rows = [TxRow(r["type"], Decimal(r["amount"]), r["label"] or "",
                      r["logged_at"], r["tx_date"], r["category"], r["category_id"]) for r in tx]
        items = assemble(habit_chips(rows, now), limit, has_history=len(rows) >= MIN_OCCURRENCES,
                         frequent=frequent_chips(rows, now))
        resolve_example_categories(items, cats)
    except Exception as exc:  # noqa: BLE001 — chips are nice-to-have
        slog("starters", f"failed, serving examples: {type(exc).__name__}: {exc}")
        items = assemble([], limit)
    slog("starters", f"user={user_id[:8]} items={[(i['kind'], i['label']) for i in items]}")
    return items


__all__ = ["EXAMPLES", "QUESTIONS", "TxRow", "assemble", "build_starters", "frequent_chips",
           "habit_chips",
           "question_chips", "resolve_example_categories"]
