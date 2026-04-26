from collections import defaultdict, Counter
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_agent.tools.wallet_tools import WalletTools


def build_namespace(session_factory: async_sessionmaker, user_id: str) -> dict:
    return {
        "wallet": WalletTools(session_factory, user_id),
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
