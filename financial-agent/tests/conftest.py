from decimal import Decimal
import pytest


@pytest.fixture
def sample_transactions():
    return [
        {"amount": Decimal("500.00"), "effect_on_wallet": 1, "type": "income"},
        {"amount": Decimal("200.00"), "effect_on_wallet": -1, "type": "expense"},
        {"amount": Decimal("150.00"), "effect_on_wallet": -1, "type": "expense"},
    ]


@pytest.fixture
def sample_monthly_history():
    return [Decimal(str(x)) for x in [8000, 9500, 7800, 10200, 8800, 9100]]


@pytest.fixture
def amortization_100k():
    from financial_agent.finance.loan import amortization_schedule
    return amortization_schedule(Decimal("100000"), Decimal("6"), 12)
