"""Financial decision calculators for the CodeAct sandbox.

NEW in Wave 5 (Advisor expansion).

These are PURE FUNCTIONS — no DB, no async, no I/O. They expose
decision-grade math to the LLM via the CodeAct namespace so the model
can compute DTI, mortgage payments, debt payoff timelines, etc. without
writing the same loops every turn (and getting them subtly wrong).

Why a separate module:
  • R3 (NO-MENTAL-ARITHMETIC) bans the LLM from doing money math in its
    head. The LLM offloads to run_python; these helpers make the common
    advisor calculations easy to call.
  • The HOME / DEBT / REFI / DISCIPLINE playbooks all reference these
    rules — by codifying them once, every advisor turn uses the same
    thresholds (DTI 40%, 28/36, 20/4/10, etc.).
  • Decimal throughout. NEVER use float for money — see CLAUDE.md
    "Money column is float8" note for the existing latent bug.

Every helper is documented in `prompts.py` # CODEACT TOOLBOX section
under "## Financial calculators". UT-NS01 / UT-P01 enforce lockstep.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any


_CENT = Decimal("0.01")
_PERCENT = Decimal("100")
_ZERO = Decimal("0")
_TWELVE = Decimal("12")


def _to_dec(x: Any) -> Decimal:
    """Coerce int/float/str/Decimal to Decimal safely."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        # Round-trip via str to avoid float-rep noise (3.5 → 3.5, not 3.4999...).
        return Decimal(str(x))
    return Decimal(x)


def _q(x: Decimal) -> Decimal:
    """Quantize to satang (2 decimals)."""
    return x.quantize(_CENT, rounding=ROUND_HALF_UP)


# ── DTI ────────────────────────────────────────────────────────────────────


def compute_dti(monthly_income, monthly_debt_payments) -> Decimal:
    """Debt-to-Income ratio = total monthly debt service / net income.

    Returned as a Decimal in 0..1 range (0.42 = 42%). Bank cap is 40%;
    > 50% is the danger zone. Returns 0 if income is 0 (caller must
    guard against the "no income" interpretation).
    """
    inc = _to_dec(monthly_income)
    debts = _to_dec(monthly_debt_payments)
    if inc <= 0:
        return _ZERO
    return (debts / inc).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# ── Mortgage / loan payment ───────────────────────────────────────────────


def mortgage_payment(principal, annual_rate_pct, term_years) -> Decimal:
    """Monthly payment for a fully-amortizing fixed-rate loan.

    Standard formula:
        P = L * c / (1 - (1+c)^-n)
    where L = principal, c = monthly rate, n = total months.

    Args:
      principal:        loan amount (Decimal-coerceable)
      annual_rate_pct:  e.g. 3.5 means 3.5%/yr
      term_years:       loan term in years
    """
    L = _to_dec(principal)
    annual_rate = _to_dec(annual_rate_pct) / _PERCENT
    n = int(_to_dec(term_years) * _TWELVE)
    if n <= 0 or L <= 0:
        return _ZERO
    c = annual_rate / _TWELVE
    if c == 0:
        return _q(L / Decimal(n))
    factor = (Decimal(1) + c) ** n
    payment = L * c * factor / (factor - Decimal(1))
    return _q(payment)


# ── Home affordability check ──────────────────────────────────────────────


def affordability_check(
    price,
    down_payment,
    annual_rate_pct,
    term_years,
    monthly_income,
    other_monthly_debts=0,
) -> dict:
    """Run the HOME playbook decision rules in one shot.

    Returns:
      {
        "monthly_payment": Decimal,
        "house_payment_ratio": Decimal,   # payment / income (0..1)
        "dti_ratio": Decimal,             # (payment + other) / income
        "stress_test_payment": Decimal,   # rate + 2%
        "verdict": "safe" | "tight" | "dangerous",
        "reasons": list[str],             # human-readable explanation
      }

    Thresholds (lockstep with HOME playbook in advice_playbook.py):
      • house_payment_ratio  ≤ 30% safe, ≤ 35% tight, > 35% danger
      • dti_ratio            ≤ 40% safe, ≤ 50% tight, > 50% danger
      • Stress test at rate +2% must still keep dti ≤ 50%
    """
    price_d = _to_dec(price)
    down_d = _to_dec(down_payment)
    loan = price_d - down_d
    payment = mortgage_payment(loan, annual_rate_pct, term_years)
    stress = mortgage_payment(loan, _to_dec(annual_rate_pct) + Decimal(2), term_years)

    inc = _to_dec(monthly_income)
    other = _to_dec(other_monthly_debts)

    if inc <= 0:
        return {
            "monthly_payment": payment,
            "house_payment_ratio": _ZERO,
            "dti_ratio": _ZERO,
            "stress_test_payment": stress,
            "verdict": "dangerous",
            "reasons": ["รายได้เป็น 0 ตัดสินใจไม่ได้ ขอข้อมูลรายได้ก่อน"],
        }

    house_ratio = (payment / inc).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    dti = ((payment + other) / inc).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    stress_dti = ((stress + other) / inc).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )

    reasons: list[str] = []
    if house_ratio > Decimal("0.35"):
        reasons.append(
            f"ค่างวดบ้าน {(house_ratio * _PERCENT):.1f}% ของรายได้ — "
            f"สูงกว่าเพดาน 30%"
        )
    elif house_ratio > Decimal("0.30"):
        reasons.append(
            f"ค่างวดบ้าน {(house_ratio * _PERCENT):.1f}% ของรายได้ — ตึงแต่พอไหว"
        )
    if dti > Decimal("0.50"):
        reasons.append(f"DTI รวม {(dti * _PERCENT):.1f}% เกิน 50% อันตราย")
    elif dti > Decimal("0.40"):
        reasons.append(f"DTI รวม {(dti * _PERCENT):.1f}% เกิน 40% เพดานแบงค์")
    if stress_dti > Decimal("0.50"):
        reasons.append(
            f"Stress test ที่ดอก +2% → DTI {(stress_dti * _PERCENT):.1f}% — "
            f"ผ่อนไม่ไหวถ้าดอกขึ้น"
        )

    if dti > Decimal("0.50") or house_ratio > Decimal("0.35"):
        verdict = "dangerous"
    elif dti > Decimal("0.40") or house_ratio > Decimal("0.30") or stress_dti > Decimal("0.50"):
        verdict = "tight"
    else:
        verdict = "safe"
        reasons.append("อยู่ในเกณฑ์ปลอดภัย — ค่างวด ≤ 30% และ DTI ≤ 40%")

    return {
        "monthly_payment": payment,
        "house_payment_ratio": house_ratio,
        "dti_ratio": dti,
        "stress_test_payment": stress,
        "verdict": verdict,
        "reasons": reasons,
    }


# ── Refinance payback ─────────────────────────────────────────────────────


def refi_payback_months(refi_fees, current_payment, new_payment) -> int:
    """Months to recoup refi fees from monthly savings.

    Returns 0 if new payment is not actually lower (refi doesn't save).
    Returns 99999 (sentinel "never") if fees > 0 but savings = 0.
    """
    fees = _to_dec(refi_fees)
    cur = _to_dec(current_payment)
    new = _to_dec(new_payment)
    monthly_savings = cur - new
    if monthly_savings <= 0:
        return 0
    if fees <= 0:
        return 0
    months = (fees / monthly_savings).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(months)


# ── Debt payoff timeline ──────────────────────────────────────────────────


def debt_payoff_months(balance, annual_rate_pct, monthly_payment) -> int:
    """Months to pay off a revolving balance at a fixed monthly payment.

    Returns 99999 (sentinel "never") if the monthly payment is too small
    to cover the monthly interest accrual.
    """
    bal = _to_dec(balance)
    rate = _to_dec(annual_rate_pct) / _PERCENT
    pay = _to_dec(monthly_payment)
    if bal <= 0:
        return 0
    if pay <= 0:
        return 99999
    monthly_rate = rate / _TWELVE
    monthly_interest = bal * monthly_rate
    if pay <= monthly_interest:
        return 99999  # payment never catches up to interest
    months = 0
    remaining = bal
    while remaining > 0 and months < 600:  # 50 years sanity cap
        interest = remaining * monthly_rate
        principal_paid = pay - interest
        remaining -= principal_paid
        months += 1
    return months


# ── Emergency fund target ─────────────────────────────────────────────────


def emergency_fund_target(monthly_expenses, months=6) -> Decimal:
    """Target emergency fund = monthly_expenses × months.

    DISCIPLINE playbook guidance:
      3 months — single, stable salaried job
      6 months — has dependents OR moderate income volatility
      12 months — freelance / commission / sole earner with family
    """
    exp = _to_dec(monthly_expenses)
    m = _to_dec(months)
    return _q(exp * m)


__all__ = [
    "compute_dti",
    "mortgage_payment",
    "affordability_check",
    "refi_payback_months",
    "debt_payoff_months",
    "emergency_fund_target",
]
