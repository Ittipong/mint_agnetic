from decimal import Decimal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from financial_agent.tools.wallet_functions import running_balance, net_worth

__all__ = ["WalletTools", "running_balance", "net_worth"]

_BALANCE_CASE_SQL = """
    COALESCE(SUM(
        CASE
            WHEN t.type = 'income' AND t.wallet_sync_id = gw.sync_id::text
                THEN COALESCE(t.converted_amount, t.amount)
            WHEN t.type = 'expense' AND t.wallet_sync_id = gw.sync_id::text
                THEN -COALESCE(t.converted_amount, t.amount)
            WHEN t.type = 'transfer' AND t.wallet_sync_id = gw.sync_id::text
                THEN -COALESCE(t.converted_amount, t.amount)
            WHEN t.type = 'transfer' AND t.destination_wallet_sync_id = gw.sync_id::text
                THEN COALESCE(t.destination_converted_amount, t.amount)
            WHEN t.type = 'creditCardPay' AND t.wallet_sync_id = gw.sync_id::text
                THEN -COALESCE(t.converted_amount, t.amount)
            WHEN t.type = 'creditCardCashAdvance' AND t.destination_wallet_sync_id = gw.sync_id::text
                THEN COALESCE(t.destination_converted_amount, t.amount)
            ELSE 0
        END
    ), 0)
"""


class WalletTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def get_wallets(self) -> list[dict]:
        async with self._sf() as session:
            result = await session.execute(
                text("""
                    SELECT id, sync_id, name, initial_balance, cached_balance,
                           currency, wallet_category, created_at
                    FROM general_wallets
                    WHERE user_id = :user_id AND deleted_at IS NULL
                    ORDER BY created_at
                """),
                {"user_id": self._user_id},
            )
            rows = result.mappings().all()
            return [
                {
                    **dict(row),
                    "id": str(row["id"]),
                    "sync_id": str(row["sync_id"]),
                    "initial_balance": Decimal(str(row["initial_balance"])),
                    "cached_balance": Decimal(str(row["cached_balance"])),
                }
                for row in rows
            ]

    async def get_all_balances(self) -> list[dict]:
        """Return all wallets with balance computed from initial_balance + transactions up to today (past only, no future).

        Balance formula per docs/balance_calculation.md:
        balance = initial_balance + SUM(effect_on_wallet * converted_amount)
        Only confirmed transactions with date <= today are included.
        """
        async with self._sf() as session:
            result = await session.execute(
                text("""
                    SELECT gw.sync_id::text AS sync_id,
                           gw.name,
                           gw.currency,
                           gw.wallet_category,
                           gw.initial_balance,
                           gw.icon,
                           gw.initial_balance +
                           COALESCE(SUM(
                               CASE
                                   WHEN t.wallet_sync_id = gw.sync_id::text
                                       THEN t.effect_on_wallet * COALESCE(t.converted_amount, t.amount)
                                   WHEN t.destination_wallet_sync_id = gw.sync_id::text
                                       THEN COALESCE(t.effect_on_destination, 0) * COALESCE(t.destination_converted_amount, t.amount)
                                   ELSE 0
                               END
                           ), 0) AS balance
                    FROM general_wallets gw
                    LEFT JOIN transactions t
                        ON (t.wallet_sync_id = gw.sync_id::text OR t.destination_wallet_sync_id = gw.sync_id::text)
                        AND t.is_deleted = false
                        AND t.status = 'confirmed'
                        AND t.date <= CURRENT_DATE
                    WHERE gw.user_id = :user_id
                      AND gw.deleted_at IS NULL
                    GROUP BY gw.sync_id, gw.name, gw.currency, gw.wallet_category, gw.initial_balance, gw.icon, gw.created_at
                    ORDER BY gw.created_at
                """),
                {"user_id": str(self._user_id)},
            )
            rows = result.mappings().all()
            return [
                {
                    "sync_id": row["sync_id"],
                    "name": row["name"],
                    "currency": row["currency"],
                    "wallet_category": row["wallet_category"],
                    "initial_balance": Decimal(str(row["initial_balance"])),
                    "icon": row["icon"],
                    "balance": Decimal(str(row["balance"])) if row["balance"] is not None else Decimal("0"),
                }
                for row in rows
            ]

    async def get_balance(self, wallet_sync_id: str) -> Decimal:
        """Return balance for a specific wallet (past transactions only, up to today).

        Balance formula per docs/balance_calculation.md:
        balance = initial_balance + SUM(effect_on_wallet * converted_amount)
        Only confirmed transactions with date <= today are included.
        """
        async with self._sf() as session:
            result = await session.execute(
                text("""
                    SELECT gw.initial_balance +
                           COALESCE(SUM(
                               CASE
                                   WHEN t.wallet_sync_id = gw.sync_id::text
                                       THEN t.effect_on_wallet * COALESCE(t.converted_amount, t.amount)
                                   WHEN t.destination_wallet_sync_id = gw.sync_id::text
                                       THEN COALESCE(t.effect_on_destination, 0) * COALESCE(t.destination_converted_amount, t.amount)
                                   ELSE 0
                               END
                           ), 0) AS balance
                    FROM general_wallets gw
                    LEFT JOIN transactions t
                        ON (t.wallet_sync_id = gw.sync_id::text OR t.destination_wallet_sync_id = gw.sync_id::text)
                        AND t.is_deleted = false
                        AND t.status = 'confirmed'
                        AND t.date <= CURRENT_DATE
                    WHERE gw.sync_id::text = :wallet_sync_id
                      AND gw.user_id = :user_id
                      AND gw.deleted_at IS NULL
                    GROUP BY gw.initial_balance
                """),
                {"wallet_sync_id": str(wallet_sync_id), "user_id": str(self._user_id)},
            )
            row = result.mappings().first()
            if not row:
                return Decimal("0")
            return Decimal(str(row["balance"])) if row["balance"] is not None else Decimal("0")
