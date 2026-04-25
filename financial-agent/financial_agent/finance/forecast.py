from decimal import Decimal
from .precision import money, to_decimal

_ALPHA = Decimal("0.3")


def weighted_moving_avg(values: list[Decimal], weights: list[Decimal]) -> Decimal:
    if not values or not weights or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and equal length")
    total_weight = sum((to_decimal(w) for w in weights), Decimal("0"))
    if total_weight == Decimal("0"):
        raise ValueError("weights sum to zero")
    weighted_sum = sum(
        to_decimal(v) * to_decimal(w) for v, w in zip(values, weights)
    )
    return money(weighted_sum / total_weight)


def spending_prediction(
    monthly_history: list[Decimal],
    months_ahead: int = 1,
) -> list[Decimal]:
    if not monthly_history:
        raise ValueError("monthly_history cannot be empty")
    history = [to_decimal(v) for v in monthly_history]
    smoothed = history[0]
    for val in history[1:]:
        smoothed = _ALPHA * val + (Decimal("1") - _ALPHA) * smoothed
    return [money(smoothed) for _ in range(months_ahead)]


def cashflow_projection(
    income: Decimal,
    fixed_expenses: Decimal,
    variable_estimates: list[Decimal],
    months: int = 3,
) -> list[dict]:
    inc = to_decimal(income)
    fixed = to_decimal(fixed_expenses)
    var_avg = (
        sum((to_decimal(v) for v in variable_estimates), Decimal("0"))
        / Decimal(str(len(variable_estimates)))
        if variable_estimates
        else Decimal("0")
    )
    result = []
    for month in range(1, months + 1):
        total_expense = money(fixed + var_avg)
        net = money(inc - total_expense)
        result.append({
            "month": month,
            "projected_income": money(inc),
            "projected_expense": total_expense,
            "net": net,
        })
    return result
