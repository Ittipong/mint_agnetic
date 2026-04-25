from decimal import Decimal
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import text
from financial_agent.finance import forecast as fc
from financial_agent.finance.precision import money, to_decimal


class ForecastTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def _monthly_totals(self, months_back: int = 6) -> list[Decimal]:
        since = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
        async with self._sf() as session:
            sql = text("""
                SELECT DATE_TRUNC('month', date) AS month,
                       SUM(ABS(amount)) AS total
                FROM transactions
                WHERE created_by_user_id = :user_id
                  AND is_deleted = false
                  AND type = 'expense'
                  AND include_in_report = true
                  AND date >= :since
                GROUP BY DATE_TRUNC('month', date)
                ORDER BY month
            """)
            result = await session.execute(sql, {"user_id": self._user_id, "since": since})
            rows = result.mappings().all()
        return [to_decimal(row["total"]) for row in rows]

    async def _monthly_income_totals(self, months_back: int = 6) -> list[Decimal]:
        since = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
        async with self._sf() as session:
            sql = text("""
                SELECT DATE_TRUNC('month', date) AS month,
                       SUM(amount) AS total
                FROM transactions
                WHERE created_by_user_id = :user_id
                  AND is_deleted = false
                  AND type = 'income'
                  AND include_in_report = true
                  AND date >= :since
                GROUP BY DATE_TRUNC('month', date)
                ORDER BY month
            """)
            result = await session.execute(sql, {"user_id": self._user_id, "since": since})
            rows = result.mappings().all()
        return [to_decimal(row["total"]) for row in rows]

    async def predict_spending(self, months_ahead: int = 3) -> list[dict]:
        history = await self._monthly_totals(months_back=6)
        if not history:
            return []
        predictions = fc.spending_prediction(history, months_ahead)
        return [{"month": i + 1, "predicted_expense": p} for i, p in enumerate(predictions)]

    async def project_cashflow(self, months: int = 3) -> list[dict]:
        expense_history = await self._monthly_totals(months_back=6)
        income_history = await self._monthly_income_totals(months_back=6)

        avg_income = (
            money(sum(income_history, Decimal("0")) / Decimal(str(len(income_history))))
            if income_history else Decimal("0")
        )
        avg_expense = (
            money(sum(expense_history, Decimal("0")) / Decimal(str(len(expense_history))))
            if expense_history else Decimal("0")
        )

        return fc.cashflow_projection(avg_income, avg_expense, [], months)
