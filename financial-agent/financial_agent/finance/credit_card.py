from decimal import Decimal
from .precision import money, rate, percent, to_decimal

_HUNDRED = Decimal("100")
_DAYS_PER_YEAR = Decimal("365")


def minimum_payment(
    balance: Decimal,
    min_pct: Decimal = Decimal("5"),
    min_flat: Decimal = Decimal("100"),
) -> Decimal:
    """Returns max(balance * min_pct%, min_flat) but not more than balance."""
    pct_amount = money(to_decimal(balance) * to_decimal(min_pct) / _HUNDRED)
    flat = to_decimal(min_flat)
    result = max(pct_amount, flat)
    return money(min(result, to_decimal(balance)))


def daily_interest(balance: Decimal, annual_rate_pct: Decimal) -> Decimal:
    """Daily interest amount. annual_rate_pct is e.g. 18 for 18%."""
    b = to_decimal(balance)
    r = to_decimal(annual_rate_pct) / _HUNDRED / _DAYS_PER_YEAR
    return money(b * r)


def statement_interest(
    avg_daily_balance: Decimal,
    annual_rate_pct: Decimal,
    days: int,
) -> Decimal:
    """Interest using average daily balance method."""
    b = to_decimal(avg_daily_balance)
    r = to_decimal(annual_rate_pct) / _HUNDRED / _DAYS_PER_YEAR
    return money(b * r * Decimal(str(days)))


def utilization(balance: Decimal, credit_limit: Decimal) -> Decimal:
    """Returns utilization percentage (0-100)."""
    limit = to_decimal(credit_limit)
    if limit == Decimal("0"):
        return Decimal("0")
    return percent(to_decimal(balance) / limit * _HUNDRED)
