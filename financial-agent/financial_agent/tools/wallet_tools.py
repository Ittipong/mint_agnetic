from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.db import queries
from financial_agent.db.wallet_query import running_balance, net_worth

__all__ = ["WalletTools", "running_balance", "net_worth"]


class WalletTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_wallets(self) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_general_wallets(session, self._user_id)

    async def get_all_balances(self) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_all_wallet_balances(session, self._user_id)

    async def get_balance(self, wallet_sync_id: str) -> Decimal:
        async with self._sf() as session:
            return await queries.fetch_wallet_balance(session, self._user_id, wallet_sync_id)
