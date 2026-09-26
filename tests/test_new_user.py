"""UT-NU — first-chat experience for a brand-new user (NEW USER PLAYBOOK).

Onboarding leaves ONE wallet and no transactions. Before this, the agent
answered that user as if they had months of history ("เงินฉุกเฉิน 0 บาท",
"รายได้เฉลี่ย 11,667"), hit dead ends, and never answered the
[INTENT:wallet_created] marker mobile sends at the very first moment.
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import user_stage as us
from src.agent.nodes import classify_intent as ci
from src.agent.suggest_followups import _finalize
from src.agent.tools.codeact.calculators import starter_plan

TODAY = date(2026, 9, 26)


@pytest.mark.parametrize("tx, first, stage", [
    (0, None, "new"),
    (3, date(2026, 9, 20), "starting"),
    (40, date(2026, 8, 28), "starting"),     # 29 days — still too little
    (40, date(2026, 8, 27), "established"),  # 30 days
])
def test_UT_NU01_compute_stage(tx, first, stage):
    assert us.compute_stage(tx, first, TODAY) == stage


def test_UT_NU02_stage_block_renders_only_for_early_users():
    info = {"stage": "new", "tx_count": 0, "history_days": 0, "has_income": False,
            "wallet_count": 1}
    block = us.format_user_stage_block(info)
    assert "stage=new" in block and "NEW USER PLAYBOOK" in block
    assert us.format_user_stage_block({**info, "stage": "established"}) == ""
    assert us.format_user_stage_block(None) == ""


def test_UT_NU03_load_user_stage_uses_loader_and_never_raises():
    async def loader(uid):
        return {"tx_count": 2, "first_tx_date": date(2026, 9, 25), "income_count": 1,
                "wallet_count": 1}

    async def boom(uid):
        raise RuntimeError("db down")

    us.set_user_stage_loader(loader)
    try:
        info = asyncio.run(us.load_user_stage("u-1", today=TODAY))
        assert info == {"stage": "starting", "tx_count": 2, "history_days": 2,
                        "has_income": True, "wallet_count": 1}
        us.set_user_stage_loader(boom)
        assert asyncio.run(us.load_user_stage("u-1", today=TODAY)) is None
    finally:
        us.set_user_stage_loader(None)


def test_UT_NU04_starter_plan_numbers_are_the_users_own():
    """25,000 income, 7,000 fixed, 5,000 in the wallet, paid on the 25th."""
    p = starter_plan(25000, 7000, balance=5000, salary_day=25, today=TODAY)
    assert p["save_rate_pct"] == 20                   # fixed leaves ≥ 50%
    assert p["monthly_saving"] == Decimal("5000.00")
    assert p["flexible_monthly"] == Decimal("13000.00")
    assert p["daily_flexible"] == Decimal("433")      # floor(13000/30)
    assert p["emergency_target"] == Decimal("60000.00")  # 3 × (7000+13000)
    assert p["next_payday"] == date(2026, 10, 25)     # today (26th) is past the 25th
    assert p["days_to_payday"] == 29
    assert p["balance_per_day"] == Decimal("172")     # floor(5000/29)


@pytest.mark.parametrize("fixed, rate", [(15000, 10), (26000, 0)])
def test_UT_NU05_starter_plan_tight_budgets(fixed, rate):
    p = starter_plan(25000, fixed, today=TODAY)
    assert p["save_rate_pct"] == rate
    assert p["flexible_monthly"] >= 0
    assert p["fixed_exceeds_income"] is (fixed >= 25000)
    assert "balance_per_day" not in p                 # no balance given


def test_UT_NU06_starter_plan_without_salary_day_uses_month_end():
    p = starter_plan(30000, 0, balance=6000, today=TODAY)
    assert p["next_payday"] == date(2026, 10, 1) and p["days_to_payday"] == 5
    assert p["balance_per_day"] == Decimal("1200")


def test_UT_NU07_wallet_created_marker_gets_a_welcome_without_llm(monkeypatch):
    """The marker mobile sends right after the first wallet is created was
    passed raw to the LLM. It must be answered deterministically, router on
    OR off, with the welcome and doable chips."""
    def boom(role, *, timeout_s=30.0):
        raise AssertionError("welcome must not call the classifier LLM")

    monkeypatch.setattr(ci, "make_llm_call", boom)
    for router in ("0", "1"):
        monkeypatch.setenv("CLASSIFY_ROUTER_ENABLED", router)
        out = asyncio.run(ci.classify_intent_node(
            {"messages": [HumanMessage(ci.WALLET_CREATED_MARKER)]}))
        assert out["__classify_route__"] == "direct_propose"
        assert out["__classified_add__"] == {"__welcome__": True}

    res = asyncio.run(ci.direct_propose_node({"__classified_add__": {"__welcome__": True}}))
    assert isinstance(res["messages"][0], AIMessage)
    assert "รายได้ต่อเดือน" in res["messages"][0].content
    assert [c["send"] for c in res["suggestions_block"]["items"]][0].startswith("ช่วยวางแผน")


def test_UT_NU08_app_only_action_chips_are_dropped():
    """Live chips offered "เพิ่มบัตรเครดิต" / "ตั้งเป้าหมายออมเงิน" — the chat
    cannot do either, so a tap only earns a redirect."""
    out = _finalize([
        {"label": "เพิ่มบัตรเครดิต", "send": "เพิ่มบัตรเครดิต"},
        {"label": "ตั้งเป้าหมายออมเงิน", "send": "ตั้งเป้าหมายออมเงิน"},
        {"label": "ลบรายการนี้", "send": "ลบรายการกาแฟ"},
        {"label": "ควรออมเดือนละเท่าไหร่", "send": "ควรออมเดือนละเท่าไหร่ดี"},
        {"label": "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่", "send": "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่"},
    ])
    assert [c["label"] for c in out] == ["ควรออมเดือนละเท่าไหร่", "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่"]


def test_UT_NU09_first_record_is_celebrated(monkeypatch):
    """stage=new → the deterministic confirm line marks the first record."""
    import importlib

    pt = importlib.import_module("src.agent.tools.propose_transaction")

    async def fake_core(**kw):
        return pt.ProposeResult(state_updates={}, summary={"amount": 60, "category_name": "กาแฟ"})

    monkeypatch.setattr(ci, "_propose_core", fake_core)
    base = {"__classified_add__": {"amount": 60, "category_label": "กาแฟ"}}
    new = asyncio.run(ci.direct_propose_node({**base, "user_stage": {"stage": "new"}}))
    assert new["messages"][0].content.startswith("รายการแรกของคุณ 🎉 ขอยืนยันรายการ 60 บาท")
    old = asyncio.run(ci.direct_propose_node({**base, "user_stage": {"stage": "established"}}))
    assert old["messages"][0].content.startswith("ขอยืนยันรายการ 60 บาท")
