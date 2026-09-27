"""UT-ST — personal starter chips for the empty chat (starters.py)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from src.agent.starters import (BKK, EXAMPLES, TxRow, ask_chips, assemble, habit_chips,
                                resolve_example_categories)

NOW = datetime(2026, 9, 26, 8, 30, tzinfo=BKK)   # Saturday morning


def _rows(label, amount, hour, days, typ="expense", day_of_month=None):
    out = []
    for d in days:
        when = NOW - timedelta(days=d)
        if day_of_month:
            when = when.replace(day=day_of_month)
        out.append(TxRow(typ, Decimal(amount), label,
                         when.replace(hour=hour, tzinfo=None), when.date()))
    return out


def test_UT_ST01_morning_habit_with_usual_amount():
    """Coffee 12× around 8 am (mostly 65) beats lunch at noon in the morning."""
    rows = (_rows("Café Amazon", 65, 8, range(1, 11)) + _rows("Café Amazon", 95, 9, [12, 13])
            + _rows("ข้าวมันไก่", 60, 12, range(1, 20)))
    chips = habit_chips(rows, NOW)
    assert chips[0]["label"] == "Café Amazon 65" and chips[0]["send"] == "Café Amazon 65"
    assert chips[0]["kind"] == "log"
    assert all(c["label"] != "ข้าวมันไก่ 60" for c in chips)  # noon habit, 8:30 now


def test_UT_ST02_already_recorded_today_is_skipped():
    rows = _rows("Café Amazon", 65, 8, [0, 1, 2, 3, 4])
    assert habit_chips(rows, NOW) == []


def test_UT_ST03_rare_items_are_not_habits():
    assert habit_chips(_rows("ร้านใหม่", 300, 8, [1, 5]), NOW) == []


def test_UT_ST04_salary_only_near_payday_and_once_a_month():
    salary = []
    for m_back, d in [(1, 25), (2, 25), (3, 25)]:
        dt = date(2026, 9 - m_back, 25)
        salary.append(TxRow("income", Decimal(42000), "เงินเดือน", datetime(2026, 9 - m_back, 25, 7), dt))
    on_day = datetime(2026, 9, 25, 7, tzinfo=BKK)
    chips = habit_chips(salary, on_day)
    assert chips and chips[0]["label"] == "ได้เงินเดือน 42,000"
    assert habit_chips(salary, datetime(2026, 9, 10, 7, tzinfo=BKK)) == []   # not payday
    recorded = salary + [TxRow("income", Decimal(42000), "เงินเดือน",
                               datetime(2026, 9, 25, 7), date(2026, 9, 25))]
    assert habit_chips(recorded, datetime(2026, 9, 26, 7, tzinfo=BKK)) == []  # this month done


def test_UT_ST05_card_due_soon_and_budget_near_limit_become_questions():
    cards = [{"name": "บัตร KTC", "payment_due_day": 28, "used": Decimal("24943")},
             {"name": "บัตร SCB", "payment_due_day": 15, "used": Decimal("603")},   # not soon
             {"name": "บัตรว่าง", "payment_due_day": 27, "used": Decimal("0")}]      # nothing owed
    budgets = [{"name": "งบอาหาร", "pct_used": Decimal("103")},
               {"name": "เที่ยว", "pct_used": Decimal("40")}]
    chips = ask_chips(cards, budgets, date(2026, 9, 26))
    labels = [c["label"] for c in chips]
    assert labels == ["งบอาหารเดือนนี้เกินไปเท่าไหร่", "บัตร KTC รอบนี้ต้องจ่ายเท่าไหร่"]
    assert all(c["kind"] == "ask" for c in chips)
    assert chips[1]["reason"] == "card_due:อีก 2 วัน"
    assert [c["hint"] for c in chips] == ["ใช้ไป 103% ของงบ", "ครบกำหนดใน 2 วัน"]


def test_UT_ST06_examples_fill_up_to_three_only():
    assert [i["kind"] for i in assemble([], [], 5)] == ["example"] * 3
    log = [{"label": f"x{i}", "send": f"x{i}", "kind": "log", "reason": ""} for i in range(3)]
    assert [i["kind"] for i in assemble(log, [], 5)] == ["log"] * 3
    assert len(assemble(log, log, 4)) == 4


def test_UT_ST07_endpoint_never_errors(monkeypatch):
    """A DB failure still serves the examples."""
    import asyncio

    from src.agent import starters
    from src.agent.tools.codeact import db

    async def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_pool", boom)
    items = asyncio.run(starters.build_starters("u-1", limit=3, now=NOW))
    assert items == [dict(e) for e in EXAMPLES[:3]]


def test_UT_ST08_monthly_charges_are_not_nightly_habits():
    """Live: Netflix / Spotify / iCloud (charged monthly at 03:00) topped the
    list at 1 am every night. A monthly item only shows around its usual day
    and only while this month's is still unrecorded."""
    netflix = [TxRow("expense", Decimal(419), "Netflix", datetime(2026, m, 10, 3), date(2026, m, 10))
               for m in (6, 7, 8)]
    assert habit_chips(netflix, datetime(2026, 9, 26, 2, tzinfo=BKK)) == []
    due = habit_chips(netflix, datetime(2026, 9, 10, 9, tzinfo=BKK))
    assert due and due[0]["label"] == "Netflix 419" and due[0]["reason"] == "monthly:day10"


def test_UT_ST09_one_chip_per_category_and_history_fill():
    """Three coffee shops at 8 am are one choice; a user WITH history gets
    questions about their data as filler, not the generic 'กาแฟ 60'."""
    rows = []
    for shop in ("Starbucks", "Café Amazon", "All Café"):
        rows += [TxRow("expense", Decimal(80), shop, (NOW - timedelta(days=d)).replace(tzinfo=None),
                       (NOW - timedelta(days=d)).date(), "กาแฟ") for d in range(1, 6)]
    chips = habit_chips(rows, NOW)
    assert len(chips) == 1
    items = assemble(chips, [], 5, has_history=True)
    assert [i["kind"] for i in items] == ["log", "ask", "ask"]
    assert "กาแฟ 60" not in [i["label"] for i in items]


def test_UT_ST10_chip_shows_exactly_what_it_sends():
    """A chip that shows one sentence and sends another puts words in the
    user's mouth — every label must equal its send."""
    cards = [{"name": "บัตร KTC", "payment_due_day": 28, "used": Decimal("24943")}]
    budgets = [{"name": "อาหาร", "pct_used": Decimal("85")}]
    rows = _rows("Café Amazon", 65, 8, range(1, 6))
    items = (habit_chips(rows, NOW) + ask_chips(cards, budgets, date(2026, 9, 26))
             + assemble([], [], 5) + assemble([], [], 5, has_history=True))
    assert items and all(i["label"] == i["send"] for i in items)


def test_UT_ST11_record_chips_carry_category_and_why():
    """Record chips name their category (sync_id + name) so the client can draw
    the icon, and say why they are shown."""
    rows = [TxRow("expense", Decimal(65), "Café Amazon", (NOW - timedelta(days=d)).replace(tzinfo=None),
                  (NOW - timedelta(days=d)).date(), "กาแฟ", "cat-coffee") for d in range(1, 6)]
    chip = habit_chips(rows, NOW)[0]
    assert (chip["category_id"], chip["category_name"], chip["hint"]) == ("cat-coffee", "กาแฟ", "ปกติจดเวลานี้")

    items = resolve_example_categories(assemble([], [], 3), {"กาแฟ": "u-coffee", "อาหาร": "u-food"})
    assert [i.get("category_id") for i in items] == ["u-coffee", None, None]   # no ร้านอาหาร for this user
    assert items[0]["hint"] == "ตัวอย่าง · แตะแล้วแก้ยอดได้"
    assert EXAMPLES[0]["category_id"] is None   # resolving never mutates the shared examples
