"""Unit tests for the analyze subgraph — focus on pure-code paths.

Why only pure code: planner/responder LLM calls are exercised in the
LangSmith eval harness; here we lock down deterministic logic so a
regression in time parsing or SQL building fails CI without spending tokens.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.graph.analyze_subgraph.nodes.gate import _aggregate_confidence, _first_low_confidence
from src.graph.analyze_subgraph.nodes.responder import _format_result, _format_clarification
from src.graph.analyze_subgraph.nodes.time_resolver import _quick_thai
from src.graph.analyze_subgraph.schemas import (
    ClarificationPayload,
    ExecRow,
    QueryPlan,
    QuerySpec,
    ResolvedEntity,
    TimeRange,
    decimal_to_display,
)
from src.graph.analyze_subgraph.sql_templates import build_query


# ── Time resolver ────────────────────────────────────────────────────────────


@pytest.fixture
def today():
    return date(2026, 5, 3)


def test_quick_thai_this_month(today):
    s, e, g = _quick_thai("เดือนนี้", today)
    assert s == date(2026, 5, 1)
    assert e == date(2026, 5, 31)
    assert g == "month"


def test_quick_thai_last_month(today):
    s, e, g = _quick_thai("เดือนที่แล้ว", today)
    assert s == date(2026, 4, 1)
    assert e == date(2026, 4, 30)


def test_quick_thai_last_year(today):
    s, e, g = _quick_thai("ปีที่แล้ว", today)
    assert s == date(2025, 1, 1)
    assert e == date(2025, 12, 31)
    assert g == "year"


def test_quick_thai_n_months_ago(today):
    s, e, _ = _quick_thai("5 เดือนที่แล้ว", today)
    # window of 5 months ending today: start = month 5 prior
    assert s == date(2025, 12, 1)
    assert e == today


def test_quick_thai_today(today):
    s, e, g = _quick_thai("วันนี้", today)
    assert s == today and e == today and g == "day"


def test_quick_thai_unknown_returns_none(today):
    assert _quick_thai("ไม่รู้จัก", today) is None


# ── Confidence aggregation ───────────────────────────────────────────────────


def _entity(score: float) -> ResolvedEntity:
    return ResolvedEntity(
        sync_id="x", display_name="x", kind="wallet", score=score
    )


def _time_range(conf: float) -> TimeRange:
    return TimeRange(start=date(2026, 5, 1), end=date(2026, 5, 31), confidence=conf)


def test_aggregate_uses_min():
    tr = _time_range(0.9)
    entities = [_entity(0.95), _entity(0.4), _entity(0.8)]
    assert _aggregate_confidence(tr, entities) == pytest.approx(0.4)


def test_aggregate_no_entities():
    assert _aggregate_confidence(_time_range(0.7), []) == pytest.approx(0.7)


def test_first_low_confidence_time_wins():
    plan = QueryPlan(metric="sum_expense")
    tr = _time_range(0.3)
    entities = [_entity(0.4)]
    c = _first_low_confidence(plan, tr, entities)
    assert c is not None and c.kind == "time"


def test_first_low_confidence_entity_when_time_ok():
    plan = QueryPlan(metric="sum_expense")
    tr = _time_range(0.9)
    entities = [_entity(0.3)]
    c = _first_low_confidence(plan, tr, entities)
    assert c is not None and c.kind == "entity"


def test_no_clarification_when_all_high():
    plan = QueryPlan(metric="sum_expense")
    tr = _time_range(0.9)
    assert _first_low_confidence(plan, tr, [_entity(0.85)]) is None


# ── SQL builder smoke ───────────────────────────────────────────────────────


def _spec(metric: str = "sum_expense") -> QuerySpec:
    return QuerySpec(
        metric=metric,
        time_range=TimeRange(start=date(2026, 5, 1), end=date(2026, 5, 31)),
    )


def test_build_sum_expense_includes_filters():
    sql, params = build_query(_spec("sum_expense"), "user-1")
    # 3 base + 1 type filter
    assert len(params) == 4
    assert "include_in_report = true" in sql
    assert "amount::numeric" in sql


def test_build_balance_uses_effect_on_wallet():
    sql, params = build_query(_spec("balance"), "user-1")
    assert "effect_on_wallet" in sql
    # balance only takes user_id + end date when no wallet/currency filter
    assert len(params) == 2


def test_build_list_caps_at_50():
    sql, _ = build_query(_spec("list"), "user-1")
    assert "LIMIT 50" in sql


def test_build_with_wallet_filter():
    spec = _spec("sum_expense")
    spec.wallets = [
        ResolvedEntity(sync_id="w1", display_name="Cash", kind="wallet", score=1.0)
    ]
    sql, params = build_query(spec, "user-1")
    assert "wallet_sync_id = ANY" in sql
    assert ["w1"] in params


# ── Responder formatting ────────────────────────────────────────────────────


def test_decimal_display_formats_thousands():
    from decimal import Decimal

    assert decimal_to_display(Decimal("1234567.891")) == "1,234,567.89"


def test_format_clarification_includes_question():
    out = _format_clarification(
        ClarificationPayload(
            kind="entity",
            question="หมายถึงอันไหน?",
            candidates=[{"name": "A", "score": 0.7}, {"name": "B", "score": 0.6}],
        )
    )
    assert "NEEDS_CLARIFICATION" in out
    assert "A" in out and "B" in out


def test_format_result_no_data():
    spec = _spec("sum_expense")
    out = _format_result(spec, [])
    assert "NO_DATA" in out


def test_format_result_separates_currencies():
    spec = _spec("sum_expense")
    rows = [
        ExecRow(currency="THB", amount="5000", count=10),
        ExecRow(currency="USD", amount="100", count=3),
    ]
    out = _format_result(spec, rows)
    assert "THB" in out and "USD" in out
    # never sums across currencies
    assert "5,000" in out and "100" in out
