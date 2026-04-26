from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.db import queries


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

    async def get_transactions(
        self,
        wallet_sync_id: str | None = None,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        type: str | list[str] | None = None,
        type_group: str | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit, days, start_date, end_date, type, type_group, include_in_report
            )

