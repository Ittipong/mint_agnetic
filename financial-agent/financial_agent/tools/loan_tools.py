from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.db import queries
from financial_agent.finance import loan as loan_calc
from financial_agent.finance.precision import money, to_decimal


class LoanTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_loans(self) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_obligation_wallets(session, self._user_id)

    async def get_loan_detail(self, obligation_sync_id: str) -> dict:
        loans = await self.get_loans()
        loan = next(
            (lo for lo in loans if str(lo["sync_id"]) == obligation_sync_id), None
        )
        if not loan:
            raise ValueError(f"Loan {obligation_sync_id} not found")

        async with self._sf() as session:
            txns = await queries.fetch_obligation_transactions(session, str(loan["id"]))

        total_paid = money(
            sum((to_decimal(t["amount"]) for t in txns if t["event_type"] == "payment"), Decimal("0"))
        )
        return {**loan, "transactions": txns, "total_paid": total_paid}

    async def gen_amortization_schedule(self, obligation_sync_id: str) -> list[dict]:
        detail = await self.get_loan_detail(obligation_sync_id)
        principal = to_decimal(detail["principal"])
        rate = to_decimal(detail["annual_rate"])
        months = int(detail["term_months"] or 0)
        if months == 0 or principal == Decimal("0"):
            return []
        return loan_calc.amortization_schedule(principal, rate, months)

    async def get_remaining_balance(self, obligation_sync_id: str) -> Decimal:
        detail = await self.get_loan_detail(obligation_sync_id)
        txns = detail.get("transactions", [])
        payments = [t for t in txns if t["event_type"] == "payment"]
        if payments:
            latest = sorted(payments, key=lambda t: t["event_date"])[-1]
            remaining = to_decimal(latest["remaining_balance"])
            if remaining > Decimal("0"):
                return remaining
        return to_decimal(detail.get("outstanding_principal", detail["principal"]))
