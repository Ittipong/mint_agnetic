from decimal import Decimal


def _to_decimal(v) -> Decimal:
    return Decimal(str(v))


def _money(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


def _percent(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


_HUNDRED = Decimal("100")
_DAYS_PER_YEAR = Decimal("365")


def minimum_payment(
    balance: Decimal,
    min_pct: Decimal = Decimal("5"),
    min_flat: Decimal = Decimal("100"),
) -> Decimal:
    pct_amount = _money(_to_decimal(balance) * _to_decimal(min_pct) / _HUNDRED)
    flat = _to_decimal(min_flat)
    result = max(pct_amount, flat)
    return _money(min(result, _to_decimal(balance)))


def daily_interest(balance: Decimal, annual_rate_pct: Decimal) -> Decimal:
    b = _to_decimal(balance)
    r = _to_decimal(annual_rate_pct) / _HUNDRED / _DAYS_PER_YEAR
    return _money(b * r)


def statement_interest(
    avg_daily_balance: Decimal,
    annual_rate_pct: Decimal,
    days: int,
) -> Decimal:
    b = _to_decimal(avg_daily_balance)
    r = _to_decimal(annual_rate_pct) / _HUNDRED / _DAYS_PER_YEAR
    return _money(b * r * Decimal(str(days)))


def utilization(balance: Decimal, credit_limit: Decimal) -> Decimal:
    limit = _to_decimal(credit_limit)
    if limit == Decimal("0"):
        return Decimal("0")
    return _percent(_to_decimal(balance) / limit * _HUNDRED)
