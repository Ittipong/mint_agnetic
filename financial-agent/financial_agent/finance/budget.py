from decimal import Decimal
from .precision import money, percent, to_decimal

_HUNDRED = Decimal("100")


def variance(budgeted: Decimal, actual: Decimal) -> tuple[Decimal, Decimal, str]:
    b = to_decimal(budgeted)
    a = to_decimal(actual)
    diff = money(a - b)
    pct = percent(diff / b * _HUNDRED) if b != Decimal("0") else Decimal("0")
    if diff > Decimal("0"):
        label = "over"
    elif diff < Decimal("0"):
        label = "under"
    else:
        label = "on_track"
    return (diff, pct, label)


def utilization_pct(spent: Decimal, budget: Decimal) -> Decimal:
    b = to_decimal(budget)
    if b == Decimal("0"):
        return Decimal("0")
    return percent(to_decimal(spent) / b * _HUNDRED)


def overspend_flag(
    spent: Decimal,
    budget: Decimal,
    threshold_pct: Decimal = Decimal("100"),
) -> bool:
    pct = utilization_pct(spent, budget)
    return pct > to_decimal(threshold_pct)
