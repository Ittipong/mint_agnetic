from collections import defaultdict, Counter
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_agent.tools.wallet_tools import WalletTools
from financial_agent.tools.credit_card_tools import CreditCardTools
from financial_agent.tools.loan_tools import LoanTools
from financial_agent.tools.budget_tools import BudgetTools
from financial_agent.tools.forecast_tools import ForecastTools
from financial_agent.tools.db_tool import DBTools
from financial_agent.finance import wallet as _wallet_calc
from financial_agent.finance import credit_card as _cc_calc
from financial_agent.finance import loan as _loan_calc
from financial_agent.finance import budget as _budget_calc
from financial_agent.finance import forecast as _forecast_calc
from financial_agent.finance.precision import money, rate, percent, to_decimal


class FinanceCalc:
    wallet = _wallet_calc
    credit_card = _cc_calc
    loan = _loan_calc
    budget = _budget_calc
    forecast = _forecast_calc
    money = staticmethod(money)
    rate = staticmethod(rate)
    percent = staticmethod(percent)
    to_decimal = staticmethod(to_decimal)


def build_namespace(session_factory: async_sessionmaker, user_id: str) -> dict:
    return {
        "wallet": WalletTools(session_factory, user_id),
        "credit_card": CreditCardTools(session_factory, user_id),
        "loan": LoanTools(session_factory, user_id),
        "budget": BudgetTools(session_factory, user_id),
        "forecast": ForecastTools(session_factory, user_id),
        "db": DBTools(session_factory, user_id),
        "finance": FinanceCalc(),
        "Decimal": Decimal,
        "ROUND_HALF_UP": ROUND_HALF_UP,
        "defaultdict": defaultdict,
        "Counter": Counter,
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
        "timezone": timezone,
        "context_vars": {},
    }
