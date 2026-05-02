from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.db import queries


class TransactionTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

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

    async def get_expenses(
        self,
        wallet_sync_id: str | None = None,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """All outgoing transactions: expense + transfer-out + goalDeposit + creditCardPay + ..."""
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit, days, start_date, end_date,
                type=None, type_group="expense", include_in_report=include_in_report
            )

    async def get_income(
        self,
        wallet_sync_id: str | None = None,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """All incoming transactions: income + transfer-in + goalWithdraw + ..."""
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit, days, start_date, end_date,
                type=None, type_group="income", include_in_report=include_in_report
            )

    async def get_transfers(
        self,
        wallet_sync_id: str | None = None,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """Only transfer transactions (type=transfer)."""
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit, days, start_date, end_date,
                type="transfer", type_group=None, include_in_report=include_in_report
            )

    async def get_recent(
        self,
        wallet_sync_id: str | None = None,
        limit: int = 10,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """Last N transactions ordered by date desc. Use for "recent transactions" / "last N"."""
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit=limit, days=None,
                start_date=None, end_date=None, type=None, type_group=None, include_in_report=include_in_report
            )

    async def get_by_date_range(
        self,
        wallet_sync_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        type_group: str | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """Transactions within a date range. dates as "YYYY-MM-DD"."""
        async with self._sf() as session:
            return await queries.fetch_transactions(
                session, self._user_id, wallet_sync_id, limit=None, days=None,
                start_date=start_date, end_date=end_date, type=None, type_group=type_group, include_in_report=include_in_report
            )
