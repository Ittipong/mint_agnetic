from decimal import Decimal
import pytest
from financial_agent.finance.budget import variance, utilization_pct, overspend_flag


def test_variance_over():
    diff, pct, label = variance(Decimal("5000"), Decimal("6000"))
    assert diff == Decimal("1000.00")
    assert pct == Decimal("20.0000")
    assert label == "over"


def test_variance_under():
    diff, pct, label = variance(Decimal("5000"), Decimal("4000"))
    assert diff == Decimal("-1000.00")
    assert label == "under"


def test_variance_on_track():
    diff, pct, label = variance(Decimal("5000"), Decimal("5000"))
    assert diff == Decimal("0.00")
    assert label == "on_track"


def test_utilization_60pct():
    result = utilization_pct(Decimal("3000"), Decimal("5000"))
    assert result == Decimal("60.0000")


def test_utilization_zero_budget():
    result = utilization_pct(Decimal("100"), Decimal("0"))
    assert result == Decimal("0")


def test_overspend_flag_true():
    assert overspend_flag(Decimal("6000"), Decimal("5000")) is True


def test_overspend_flag_false():
    assert overspend_flag(Decimal("4000"), Decimal("5000")) is False


def test_overspend_flag_at_threshold():
    # exactly 100% — not over (> not >=)
    assert overspend_flag(Decimal("5000"), Decimal("5000")) is False


def test_overspend_custom_threshold():
    # 80% threshold — 85% spent → over
    assert overspend_flag(Decimal("4250"), Decimal("5000"), Decimal("80")) is True
    # 75% spent → not over
    assert overspend_flag(Decimal("3750"), Decimal("5000"), Decimal("80")) is False
