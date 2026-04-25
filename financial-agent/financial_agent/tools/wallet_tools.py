from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from loguru import logger
from financial_agent.db import queries
from financial_agent.finance.precision import money, to_decimal


class WalletTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_wallets(self) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_general_wallets(session, self._user_id)

    async def get_balance(self, wallet_sync_id: str) -> Decimal:
        async with self._sf() as session:
            return await queries.fetch_wallet_balance(session, self._user_id, wallet_sync_id)

    async def get_transactions(
        self,
        wallet_sync_id: str | None = None,
        limit: int | None = 100,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit, days, start_date, end_date
            )

    async def get_net_worth(self) -> dict:
        async with self._sf() as session:
            wallets = await queries.fetch_general_wallets(session, self._user_id)
            goals = await queries.fetch_goal_wallets(session, self._user_id)
            obligations = await queries.fetch_obligation_wallets(session, self._user_id)

        # Compute real wallet balances (cached_balance may be NULL)
        wallet_balances = []
        for w in wallets:
            bal = await self.get_balance(w["sync_id"])
            wallet_balances.append(bal)

        goal_balances = []
        for g in goals:
            bal = await self.get_balance(g["sync_id"])
            goal_balances.append(bal)

        debt_balances = [to_decimal(o["outstanding_principal"]) for o in obligations]

        total_assets = money(sum(wallet_balances + goal_balances, Decimal("0")))
        total_debts = money(sum(debt_balances, Decimal("0")))

        return {
            "total_assets": total_assets,
            "total_debts": total_debts,
            "net_worth": money(total_assets - total_debts),
            "wallets": [
                {"name": w["name"], "balance": bal}
                for w, bal in zip(wallets, wallet_balances)
            ],
        }
