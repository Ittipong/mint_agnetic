from decimal import Decimal


def _to_decimal(v) -> Decimal:
    return Decimal(str(v))


def _money(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


def _percent(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


_HUNDRED = Decimal("100")


def variance(budgeted: Decimal, actual: Decimal) -> tuple[Decimal, Decimal, str]:
    b = _to_decimal(budgeted)
    a = _to_decimal(actual)
    diff = _money(a - b)
    pct = _percent(diff / b * _HUNDRED) if b != Decimal("0") else Decimal("0")
    if diff > Decimal("0"):
        label = "over"
    elif diff < Decimal("0"):
        label = "under"
    else:
        label = "on_track"
    return (diff, pct, label)


def utilization_pct(spent: Decimal, budget: Decimal) -> Decimal:
    b = _to_decimal(budget)
    if b == Decimal("0"):
        return Decimal("0")
    return _percent(_to_decimal(spent) / b * _HUNDRED)


def overspend_flag(
    spent: Decimal,
    budget: Decimal,
    threshold_pct: Decimal = Decimal("100"),
) -> bool:
    pct = utilization_pct(spent, budget)
    return pct > _to_decimal(threshold_pct)
