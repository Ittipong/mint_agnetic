"""UT-ST — personal starter chips for the empty chat (starters.py)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from src.agent.starters import (BKK, EXAMPLES, QUESTIONS, TxRow, assemble, habit_chips,
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


def test_UT_ST05_questions_are_many_generic_examples():
    """Owner 2026-09-27: the question list shows what Nimo can answer — many
    of them, the same for everyone, quoting none of the user's numbers."""
    items = assemble([], has_history=True)
    asks = [i for i in items if i["kind"] == "ask"]
    assert len(asks) >= 10 and [a["send"] for a in asks] == QUESTIONS
    assert not any(ch.isdigit() for a in asks for ch in a["label"])
    assert all(a["hint"] is None for a in asks)


def test_UT_ST06_records_then_questions():
    """A user with no history gets the example records; one with history but
    no habit this hour gets no record card; the user's own habits win."""
    new = assemble([], has_history=False)
    assert [i["kind"] for i in new[:2]] == ["example", "example"]
    assert new[2]["label"] == "Nimo ทำอะไรได้บ้าง"          # a first-timer asks this first
    assert assemble([], has_history=True)[0]["kind"] == "ask"
    log = [{"label": f"x {i}", "send": f"x {i}", "kind": "log", "reason": ""} for i in range(5)]
    mixed = assemble(log, limit=3, has_history=True)
    assert [i["kind"] for i in mixed[:4]] == ["log", "log", "log", "ask"]


def test_UT_ST07_endpoint_never_errors(monkeypatch):
    """A DB failure still serves the examples."""
    import asyncio

    from src.agent import starters
    from src.agent.tools.codeact import db

    async def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_pool", boom)
    items = asyncio.run(starters.build_starters("u-1", limit=3, now=NOW))
    assert items == assemble([], 3)
    assert [i["kind"] for i in items[:2]] == ["example", "example"]


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
    """Three coffee shops at 8 am are one choice, and a user WITH history
    never gets the generic 'กาแฟ 60' record."""
    rows = []
    for shop in ("Starbucks", "Café Amazon", "All Café"):
        rows += [TxRow("expense", Decimal(80), shop, (NOW - timedelta(days=d)).replace(tzinfo=None),
                       (NOW - timedelta(days=d)).date(), "กาแฟ") for d in range(1, 6)]
    chips = habit_chips(rows, NOW)
    assert len(chips) == 1
    items = assemble(chips, has_history=True)
    assert "กาแฟ 60" not in [i["label"] for i in items]


def test_UT_ST10_chip_shows_exactly_what_it_sends():
    """A chip that shows one sentence and sends another puts words in the
    user's mouth — every label must equal its send."""
    rows = _rows("Café Amazon", 65, 8, range(1, 6))
    items = (habit_chips(rows, NOW) + assemble([]) + assemble([], has_history=True))
    assert items and all(i["label"] == i["send"] for i in items)


def test_UT_ST11_record_chips_carry_category_and_why():
    """Record chips name their category (sync_id + name) so the client can draw
    the icon, and say why they are shown."""
    rows = [TxRow("expense", Decimal(65), "Café Amazon", (NOW - timedelta(days=d)).replace(tzinfo=None),
                  (NOW - timedelta(days=d)).date(), "กาแฟ", "cat-coffee") for d in range(1, 6)]
    chip = habit_chips(rows, NOW)[0]
    assert (chip["category_id"], chip["category_name"], chip["hint"]) == ("cat-coffee", "กาแฟ", "ปกติจดเวลานี้")

    items = resolve_example_categories(assemble([]), {"กาแฟ": "u-coffee", "อาหาร": "u-food"})
    assert [i.get("category_id") for i in items[:2]] == ["u-coffee", None]   # no ร้านอาหาร for this user
    assert items[0]["hint"] == "ตัวอย่าง · แตะแล้วแก้ยอดได้"
    assert EXAMPLES[0]["category_id"] is None   # resolving never mutates the shared examples
