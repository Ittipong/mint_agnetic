from decimal import Decimal
import pytest
from financial_agent.finance.loan import amortization_schedule, remaining_balance, total_interest_cost


def test_amortization_12_rows(amortization_100k):
    assert len(amortization_100k) == 12


def test_amortization_principal_sums_to_total(amortization_100k):
    total = sum(row["principal"] for row in amortization_100k)
    # Should equal original principal within rounding tolerance
    assert abs(total - Decimal("100000")) < Decimal("1.00")


def test_amortization_final_balance_zero(amortization_100k):
    assert amortization_100k[-1]["balance"] == Decimal("0.00")


def test_amortization_interest_decreases(amortization_100k):
    interests = [row["interest"] for row in amortization_100k]
    # Interest should generally decrease as principal reduces
    assert interests[0] > interests[-1]


def test_total_interest_positive(amortization_100k):
    result = total_interest_cost(amortization_100k)
    assert result > Decimal("0")


def test_total_interest_reasonable(amortization_100k):
    # 6% annual on 100k for 12 months — total interest should be ~3,200 range
    result = total_interest_cost(amortization_100k)
    assert Decimal("3000") < result < Decimal("4000")


def test_remaining_balance_after_half():
    balance = remaining_balance(Decimal("100000"), Decimal("6"), 12, 6)
    # After 6 months of 12, should be roughly 50k-55k
    assert Decimal("48000") < balance < Decimal("55000")


def test_remaining_balance_after_all():
    balance = remaining_balance(Decimal("100000"), Decimal("6"), 12, 12)
    assert balance == Decimal("0.00")


def test_zero_rate():
    schedule = amortization_schedule(Decimal("12000"), Decimal("0"), 12)
    assert len(schedule) == 12
    # Each payment should be exactly 1000 principal
    for row in schedule:
        assert row["interest"] == Decimal("0.00")
        assert row["principal"] == Decimal("1000.00")
