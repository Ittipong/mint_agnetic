from decimal import Decimal
import pytest
from financial_agent.finance.precision import to_decimal, money, rate, percent


def test_to_decimal_from_string():
    assert to_decimal("1234.56") == Decimal("1234.56")


def test_to_decimal_from_int():
    assert to_decimal(100) == Decimal("100")


def test_to_decimal_from_decimal():
    d = Decimal("99.99")
    assert to_decimal(d) is d


def test_to_decimal_avoids_float_repr():
    # float 0.1 + 0.2 = 0.30000000000000004 in float world
    raw = 0.1 + 0.2
    result = to_decimal(raw)
    # should be Decimal('0.30000000000000004') not Decimal('0.3')
    # — the key point is we do NOT lose precision silently
    assert isinstance(result, Decimal)
    # converting through str is the safe path regardless of exact value
    assert result == Decimal(str(raw))


def test_money_rounds_half_up():
    assert money(Decimal("1.005")) == Decimal("1.01")
    assert money(Decimal("1.004")) == Decimal("1.00")


def test_money_two_decimals():
    assert money(Decimal("1234.5")) == Decimal("1234.50")


def test_money_from_float_string():
    assert money("99.999") == Decimal("100.00")


def test_rate_six_decimals():
    result = rate(Decimal("0.18"))
    assert result == Decimal("0.180000")


def test_percent_four_decimals():
    result = percent(Decimal("25.5"))
    assert result == Decimal("25.5000")
