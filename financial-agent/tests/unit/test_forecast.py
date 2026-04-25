from decimal import Decimal
import pytest
from financial_agent.finance.forecast import weighted_moving_avg, spending_prediction, cashflow_projection


def test_wma_basic():
    values = [Decimal("100"), Decimal("200"), Decimal("300")]
    weights = [Decimal("1"), Decimal("2"), Decimal("3")]
    result = weighted_moving_avg(values, weights)
    # (100*1 + 200*2 + 300*3) / 6 = 1400/6 = 233.33...
    assert result == Decimal("233.33")


def test_wma_equal_weights():
    values = [Decimal("100"), Decimal("200"), Decimal("300")]
    weights = [Decimal("1"), Decimal("1"), Decimal("1")]
    result = weighted_moving_avg(values, weights)
    assert result == Decimal("200.00")


def test_wma_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_moving_avg([Decimal("100")], [Decimal("1"), Decimal("2")])


def test_wma_zero_weights():
    with pytest.raises(ValueError):
        weighted_moving_avg([Decimal("100")], [Decimal("0")])


def test_spending_prediction_returns_list(sample_monthly_history):
    result = spending_prediction(sample_monthly_history, months_ahead=3)
    assert len(result) == 3
    assert all(isinstance(v, Decimal) for v in result)


def test_spending_prediction_in_reasonable_range(sample_monthly_history):
    result = spending_prediction(sample_monthly_history, months_ahead=1)
    # All values in history are 7800-10200 range, prediction should be in that ballpark
    assert Decimal("6000") < result[0] < Decimal("12000")


def test_spending_prediction_empty():
    with pytest.raises(ValueError):
        spending_prediction([], months_ahead=1)


def test_cashflow_projection_length():
    result = cashflow_projection(
        Decimal("30000"), Decimal("15000"), [Decimal("5000"), Decimal("6000")], months=3
    )
    assert len(result) == 3


def test_cashflow_projection_net():
    result = cashflow_projection(
        Decimal("30000"), Decimal("20000"), [], months=1
    )
    assert result[0]["net"] == Decimal("10000.00")
    assert result[0]["projected_income"] == Decimal("30000.00")
