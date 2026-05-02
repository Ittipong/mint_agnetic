from collections import defaultdict, Counter
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker

from financial_agent.tools.wallet_tools import WalletTools
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
from financial_agent.tools.tool_registry import ToolRegistry, build_tool_registry

# Backward compatibility — prefer build_tool_registry() for new code
def build_namespace(session_factory: async_sessionmaker, user_id: str) -> dict:
    """Legacy namespace builder. Use build_tool_registry() instead."""
    return build_tool_registry(session_factory, user_id).get_namespace()
