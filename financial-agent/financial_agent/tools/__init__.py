from collections import defaultdict, Counter
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_agent.tools.wallet_tools import WalletTools, running_balance, net_worth
from financial_agent.tools.transaction_tools import TransactionTools
from financial_agent.tools.db_tool import DBTools
from financial_agent.tools.transaction_functions import (
    total_expenses,
    total_income,
    net_change,
    group_by_category,
    group_by_date,
    group_by_currency,
    sum_by_category,
    sum_by_currency,
    filter_by_tags,
    filter_by_type,
    balance_from_transactions,
)


def build_namespace(session_factory: async_sessionmaker, user_id: str) -> dict:
    return {
        "wallet": WalletTools(session_factory, user_id),
        "transaction": TransactionTools(session_factory, user_id),
        "db": DBTools(session_factory, user_id),
        "running_balance": running_balance,
        "net_worth": net_worth,
        "total_expenses": total_expenses,
        "total_income": total_income,
        "net_change": net_change,
        "group_by_category": group_by_category,
        "group_by_date": group_by_date,
        "group_by_currency": group_by_currency,
        "sum_by_category": sum_by_category,
        "sum_by_currency": sum_by_currency,
        "filter_by_tags": filter_by_tags,
        "filter_by_type": filter_by_type,
        "balance_from_transactions": balance_from_transactions,
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
