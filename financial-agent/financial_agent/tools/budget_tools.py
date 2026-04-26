from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.db import queries
from financial_agent.finance import budget as budget_calc


class BudgetTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_budgets(self, active_only: bool = True) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_budgets(session, self._user_id, active_only)

    async def get_utilization(self, budget_sync_id: str) -> dict:
        budgets = await self.get_budgets(active_only=False)
        b = next(
            (bgt for bgt in budgets if str(bgt["sync_id"]) == budget_sync_id), None
        )
        if not b:
            raise ValueError(f"Budget {budget_sync_id} not found")

        budgeted = Decimal(str(b["amount"]))
        spent = Decimal(str(b["spent_amount"]))
        diff, pct_diff, label = budget_calc.variance(budgeted, spent)
        util_pct = budget_calc.utilization_pct(spent, budgeted)

        return {
            "name": b["name"],
            "budgeted": budgeted,
            "spent": spent,
            "variance_amount": diff,
            "variance_pct": pct_diff,
            "status": label,
            "utilization_pct": util_pct,
            "is_overspent": budget_calc.overspend_flag(spent, budgeted),
        }

    async def detect_overspend(self) -> list[dict]:
        budgets = await self.get_budgets(active_only=True)
        overspent = []
        for b in budgets:
            budgeted = Decimal(str(b["amount"]))
            spent = Decimal(str(b["spent_amount"]))
            if budget_calc.overspend_flag(spent, budgeted):
                diff, pct_diff, _ = budget_calc.variance(budgeted, spent)
                overspent.append({
                    "name": b["name"],
                    "budgeted": budgeted,
                    "spent": spent,
                    "over_by": diff,
                    "over_pct": pct_diff,
                })
        return overspent
