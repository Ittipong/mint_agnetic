"""UT-CALC01..06 — financial decision calculators.

Pure-function helpers exposed to the CodeAct sandbox. These guard the
math that the HOME / REFI / DEBT / DISCIPLINE playbooks depend on.
"""

from __future__ import annotations

from decimal import Decimal

from src.agent.tools.codeact.calculators import (
    affordability_check,
    compute_dti,
    debt_payoff_months,
    emergency_fund_target,
    mortgage_payment,
    refi_payback_months,
)


# ── DTI ────────────────────────────────────────────────────────────────────


def test_UT_CALC01_compute_dti_normal_case():
    # 12,000 debt on 30,000 income = 40%
    assert compute_dti(30_000, 12_000) == Decimal("0.4")


def test_UT_CALC01_compute_dti_zero_income_safe():
    # Must not raise on zero income — Decimal/Decimal would.
    assert compute_dti(0, 5000) == Decimal("0")


# ── Mortgage payment ──────────────────────────────────────────────────────


def test_UT_CALC02_mortgage_payment_standard():
    # 2.7M loan, 3.5%, 30 yr → ~12,100/mo. Tolerance for rounding.
    pay = mortgage_payment(2_700_000, 3.5, 30)
    assert Decimal("11800") < pay < Decimal("12500")


def test_UT_CALC02_mortgage_payment_zero_rate_is_principal_divided():
    pay = mortgage_payment(120_000, 0, 1)  # 12 months, no interest
    assert pay == Decimal("10000.00")


# ── Affordability ─────────────────────────────────────────────────────────


def test_UT_CALC03_affordability_safe_verdict_for_low_payment_ratio():
    # 3M house, 600k down, 3.5%, 30y, income 60k, no other debt
    result = affordability_check(
        price=3_000_000, down_payment=600_000,
        annual_rate_pct=3.5, term_years=30,
        monthly_income=60_000, other_monthly_debts=0,
    )
    assert result["verdict"] == "safe"
    assert result["house_payment_ratio"] < Decimal("0.30")


def test_UT_CALC03_affordability_dangerous_verdict_for_tight_income():
    # Same house, half the income → should fail
    result = affordability_check(
        price=3_000_000, down_payment=300_000,
        annual_rate_pct=3.5, term_years=30,
        monthly_income=30_000, other_monthly_debts=5_000,
    )
    assert result["verdict"] == "dangerous"
    assert result["reasons"]


def test_UT_CALC03_affordability_zero_income_returns_dangerous():
    """Guard against division-by-zero when the user data is missing."""
    result = affordability_check(
        price=3_000_000, down_payment=300_000,
        annual_rate_pct=3.5, term_years=30,
        monthly_income=0,
    )
    assert result["verdict"] == "dangerous"


# ── Refinance payback ─────────────────────────────────────────────────────


def test_UT_CALC04_refi_payback_normal_case():
    # 30k fees, save 2k/month → 15 months
    assert refi_payback_months(30_000, 12_000, 10_000) == 15


def test_UT_CALC04_refi_payback_no_savings_returns_zero():
    # New payment isn't lower — refi doesn't pay back
    assert refi_payback_months(30_000, 10_000, 10_000) == 0
    assert refi_payback_months(30_000, 10_000, 11_000) == 0


# ── Debt payoff ───────────────────────────────────────────────────────────


def test_UT_CALC05_debt_payoff_normal_case():
    # 80k @ 18% APR, pay 5k/mo → ~18-19 months (real result)
    months = debt_payoff_months(80_000, 18, 5_000)
    assert 16 <= months <= 20


def test_UT_CALC05_debt_payoff_never_when_payment_under_interest():
    # 100k @ 24%, pay 1k → monthly interest = 2k > payment → sentinel
    assert debt_payoff_months(100_000, 24, 1_000) == 99999


# ── Emergency fund target ─────────────────────────────────────────────────


def test_UT_CALC06_emergency_fund_target_default_six_months():
    assert emergency_fund_target(25_000) == Decimal("150000.00")


def test_UT_CALC06_emergency_fund_target_custom_months():
    assert emergency_fund_target(25_000, months=3) == Decimal("75000.00")
