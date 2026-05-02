from decimal import Decimal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

__all__ = ["WalletTools"]

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

    async def get_wallets(
        self,
        id: str | None = None,
        search_text: str | None = None,
        include_balance: bool = True,
    ) -> list[dict]:
        """Return wallets with optional filtering. Returns list of dicts with keys: wallet_id, wallet_name, wallet_balance (Decimal), wallet_currency, wallet_category, wallet_icon, wallet_ai_intension. Use these exact keys - DO NOT use name/balance/currency/category/icon/sync_id.

        Args:
            id: Exact match on wallet_id (uuid). Use when user provides specific wallet ID.
            search_text: Fuzzy match on wallet_name (case-insensitive partial match). Use when user asks about a specific wallet by name.
            include_balance: If True (default), joins transactions to calculate real balance (slower). If False, returns initial_balance as balance (faster, no transaction join).

        Use when: user asks about wallets. Examples:
            - "what wallets do I have" → get_wallets(include_balance=False)
            - "list all wallet names" → get_wallets(include_balance=False)
            - "how much in Pad shop" → get_wallets(search_text="Pad shop")
            - "wallet id=xxx" → get_wallets(id="xxx")
        """
        async with self._sf() as session:
            conditions = ["gw.user_id = :user_id", "gw.deleted_at IS NULL"]
            params: dict = {"user_id": str(self._user_id)}

            if id:
                conditions.append("gw.sync_id::text = :wallet_id")
                params["wallet_id"] = id

            if search_text:
                conditions.append("LOWER(gw.name) LIKE LOWER(:search_text)")
                params["search_text"] = f"%{search_text}%"

            where_clause = " AND ".join(conditions)

            if include_balance:
                # Slow: joins transactions to calculate real balance
                result = await session.execute(
                    text(f"""
                        SELECT gw.sync_id::text AS sync_id,
                               gw.name,
                               gw.currency,
                               gw.wallet_category,
                               gw.initial_balance,
                               gw.icon,
                               gw.ai_message as wallet_ai_intension,
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
                        WHERE {where_clause}
                        GROUP BY gw.sync_id, gw.name, gw.currency, gw.wallet_category, gw.initial_balance, gw.icon, gw.ai_message, gw.created_at
                        ORDER BY gw.created_at
                    """),
                    params,
                )
            else:
                # Fast: no transaction join, returns initial_balance
                result = await session.execute(
                    text(f"""
                        SELECT gw.sync_id::text AS sync_id,
                               gw.name,
                               gw.currency,
                               gw.wallet_category,
                               gw.initial_balance,
                               gw.icon,
                               gw.ai_message as wallet_ai_intension,
                               gw.initial_balance AS balance
                        FROM general_wallets gw
                        WHERE {where_clause}
                        ORDER BY gw.created_at
                    """),
                    params,
                )
            rows = result.mappings().all()
            return [
                {
                    "wallet_id": row["sync_id"],
                    "wallet_name": row["name"],
                    "wallet_currency": row["currency"],
                    "wallet_category": row["wallet_category"],
                    "wallet_icon": row["icon"],
                    "wallet_balance": (Decimal(str(row["balance"])).quantize(Decimal("0.01")) if row["balance"] is not None else Decimal("0")),
                    "wallet_ai_intension": row["wallet_ai_intension"],
                }
                for row in rows
            ]
