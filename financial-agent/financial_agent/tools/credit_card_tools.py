from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import text
from financial_agent.db import queries
from financial_agent.finance import credit_card as cc
from financial_agent.finance.precision import money, to_decimal


class CreditCardTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_cards(self) -> list[dict]:
        async with self._sf() as session:
            return await queries.fetch_credit_card_wallets(session, self._user_id)

    async def get_used_amount(self, card_sync_id: str) -> Decimal:
        async with self._sf() as session:
            sql = text("""
                SELECT cw.initial_used,
                       COALESCE(SUM(t.amount * ABS(t.effect_on_wallet)), 0) AS txn_total
                FROM creditcard_wallets cw
                LEFT JOIN transactions t
                    ON t.wallet_sync_id = cw.sync_id::text
                    AND t.is_deleted = false
                    AND t.type = 'expense'
                    AND t.include_in_report = true
                WHERE cw.sync_id::text = :card_sync_id
                  AND cw.user_id = :user_id
                  AND cw.deleted_at IS NULL
                GROUP BY cw.initial_used
            """)
            result = await session.execute(
                sql, {"card_sync_id": card_sync_id, "user_id": self._user_id}
            )
            row = result.mappings().first()
            if not row:
                return Decimal("0")
            return money(to_decimal(row["initial_used"]) + to_decimal(row["txn_total"]))

    async def get_statement(self, card_sync_id: str) -> dict:
        cards = await self.get_cards()
        card = next((c for c in cards if str(c["sync_id"]) == card_sync_id), None)
        if not card:
            raise ValueError(f"Card {card_sync_id} not found")

        used = await self.get_used_amount(card_sync_id)
        limit = card["credit_limit"]
        util = cc.utilization(used, limit)
        min_pay = cc.minimum_payment(used)
        available = money(to_decimal(limit) - used)

        return {
            "name": card["name"],
            "balance": used,
            "credit_limit": limit,
            "available": available,
            "utilization_pct": util,
            "minimum_payment": min_pay,
            "billing_cycle_day": card["billing_cycle_day"],
            "payment_due_day": card["payment_due_day"],
        }

    async def calc_minimum_payment(self, card_sync_id: str) -> Decimal:
        used = await self.get_used_amount(card_sync_id)
        return cc.minimum_payment(used)

    async def calc_interest(
        self,
        card_sync_id: str,
        annual_rate_pct: Decimal | None = None,
    ) -> Decimal:
        used = await self.get_used_amount(card_sync_id)
        r = annual_rate_pct if annual_rate_pct is not None else Decimal("18")
        return cc.statement_interest(used, r, 30)
