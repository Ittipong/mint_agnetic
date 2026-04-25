from decimal import Decimal
import pytest
from financial_agent.finance.credit_card import (
    minimum_payment,
    daily_interest,
    statement_interest,
    utilization,
)


def test_minimum_payment_pct_wins():
    # 5% of 10000 = 500 > 100 flat
    result = minimum_payment(Decimal("10000"), Decimal("5"), Decimal("100"))
    assert result == Decimal("500.00")


def test_minimum_payment_flat_wins():
    # 5% of 500 = 25 < 100 flat
    result = minimum_payment(Decimal("500"), Decimal("5"), Decimal("100"))
    assert result == Decimal("100.00")


def test_minimum_payment_capped_at_balance():
    # balance 50, flat 100 → capped at 50
    result = minimum_payment(Decimal("50"), Decimal("5"), Decimal("100"))
    assert result == Decimal("50.00")


def test_daily_interest_18pct():
    # 10000 * 18% / 365 = 4.93150...
    result = daily_interest(Decimal("10000"), Decimal("18"))
    assert result == Decimal("4.93")


def test_statement_interest_30days():
    result = statement_interest(Decimal("10000"), Decimal("18"), 30)
    # 10000 * 0.18 / 365 * 30 ≈ 147.95
    assert result == Decimal("147.95")


def test_utilization_25pct():
    result = utilization(Decimal("5000"), Decimal("20000"))
    assert result == Decimal("25.0000")


def test_utilization_zero_limit():
    result = utilization(Decimal("100"), Decimal("0"))
    assert result == Decimal("0")


def test_utilization_over_100():
    result = utilization(Decimal("25000"), Decimal("20000"))
    assert result == Decimal("125.0000")
