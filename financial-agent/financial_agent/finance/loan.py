from decimal import Decimal, ROUND_HALF_UP
import numpy_financial as npf
from .precision import money, to_decimal

_HUNDRED = Decimal("100")
_TWELVE = Decimal("12")


def amortization_schedule(
    principal: Decimal,
    annual_rate_pct: Decimal,
    months: int,
) -> list[dict]:
    p = to_decimal(principal)
    annual_r = to_decimal(annual_rate_pct) / _HUNDRED
    monthly_r = annual_r / _TWELVE

    if monthly_r == Decimal("0"):
        pmt = money(p / Decimal(str(months)))
    else:
        r_float = float(monthly_r)
        pmt_float = abs(npf.pmt(r_float, months, float(p)))
        pmt = money(Decimal(str(pmt_float)))

    schedule = []
    balance = p
    for month in range(1, months + 1):
        interest = money(balance * monthly_r)
        principal_payment = money(pmt - interest)
        if month == months:
            principal_payment = balance
            pmt = money(principal_payment + interest)
        balance = money(balance - principal_payment)
        if balance < Decimal("0"):
            balance = Decimal("0")
        schedule.append({
            "month": month,
            "payment": pmt,
            "principal": principal_payment,
            "interest": interest,
            "balance": balance,
        })
    return schedule


def remaining_balance(
    principal: Decimal,
    annual_rate_pct: Decimal,
    months_total: int,
    months_paid: int,
) -> Decimal:
    schedule = amortization_schedule(principal, annual_rate_pct, months_total)
    if months_paid >= months_total:
        return Decimal("0")
    return schedule[months_paid - 1]["balance"]


def total_interest_cost(schedule: list[dict]) -> Decimal:
    return money(sum((to_decimal(row["interest"]) for row in schedule), Decimal("0")))
