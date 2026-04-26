from decimal import Decimal


def _to_decimal(v) -> Decimal:
    return Decimal(str(v))


def _money(v) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


_ALPHA = Decimal("0.3")


def weighted_moving_avg(values: list[Decimal], weights: list[Decimal]) -> Decimal:
    if not values or not weights or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and equal length")
    total_weight = sum((_to_decimal(w) for w in weights), Decimal("0"))
    if total_weight == Decimal("0"):
        raise ValueError("weights sum to zero")
    weighted_sum = sum(
        _to_decimal(v) * _to_decimal(w) for v, w in zip(values, weights)
    )
    return _money(weighted_sum / total_weight)


def spending_prediction(
    monthly_history: list[Decimal],
    months_ahead: int = 1,
) -> list[Decimal]:
    if not monthly_history:
        raise ValueError("monthly_history cannot be empty")
    history = [_to_decimal(v) for v in monthly_history]
    smoothed = history[0]
    for val in history[1:]:
        smoothed = _ALPHA * val + (Decimal("1") - _ALPHA) * smoothed
    return [_money(smoothed) for _ in range(months_ahead)]


def cashflow_projection(
    income: Decimal,
    fixed_expenses: Decimal,
    variable_estimates: list[Decimal],
    months: int = 3,
) -> list[dict]:
    inc = _to_decimal(income)
    fixed = _to_decimal(fixed_expenses)
    var_avg = (
        sum((_to_decimal(v) for v in variable_estimates), Decimal("0"))
        / Decimal(str(len(variable_estimates)))
        if variable_estimates
        else Decimal("0")
    )
    result = []
    for month in range(1, months + 1):
        total_expense = _money(fixed + var_avg)
        net = _money(inc - total_expense)
        result.append({
            "month": month,
            "projected_income": _money(inc),
            "projected_expense": total_expense,
            "net": net,
        })
    return result
