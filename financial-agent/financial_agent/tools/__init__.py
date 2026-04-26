from collections import defaultdict, Counter
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_agent.tools.wallet_tools import WalletTools
from financial_agent.tools.credit_card_tools import CreditCardTools
from financial_agent.tools.budget_tools import BudgetTools
from financial_agent.tools.forecast_tools import ForecastTools
from financial_agent.tools.db_tool import DBTools
from financial_agent.finance import wallet as _wallet_calc
from financial_agent.finance import credit_card as _cc_calc

from financial_agent.finance import budget as _budget_calc
from financial_agent.finance import forecast as _forecast_calc


class FinanceCalc:
    wallet = _wallet_calc
    credit_card = _cc_calc

    budget = _budget_calc
    forecast = _forecast_calc


def build_namespace(session_factory: async_sessionmaker, user_id: str) -> dict:
    return {
        "wallet": WalletTools(session_factory, user_id),
        "credit_card": CreditCardTools(session_factory, user_id),
        "budget": BudgetTools(session_factory, user_id),
        "forecast": ForecastTools(session_factory, user_id),
        "db": DBTools(session_factory, user_id),

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
