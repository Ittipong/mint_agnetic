# Re-export all queries and aggregation functions for backward compatibility
from financial_agent.db.transaction_query import (
    fetch_transactions,
    fetch_all_wallet_balances,
    fetch_wallet_balance,
    fetch_general_wallets,
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
