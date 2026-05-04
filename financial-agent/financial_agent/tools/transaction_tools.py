from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


def _to_decimal(value) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


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
        """Fetch transactions with flexible filtering. Returns list of transaction dicts.

        Use when: user wants custom filtering with specific type/type_group, or need advanced query beyond get_expenses/get_income.
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit, days, start_date, end_date, type, type_group, include_in_report
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
        """All outgoing transactions: expense + transfer-out + goalDeposit + creditCardPay + ...

        Use when: user asks "expenses", "spending", "how much did I spend", "spent"
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit, days, start_date, end_date,
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
        """All incoming transactions: income + transfer-in + goalWithdraw + ...

        Use when: user asks "income", "how much did I receive", "money in", "received"
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit, days, start_date, end_date,
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
        """Only transfer transactions (type=transfer).

        Use when: user asks about money transfers between wallets, "transfer"
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit, days, start_date, end_date,
                type="transfer", type_group=None, include_in_report=include_in_report
            )

    async def get_recent(
        self,
        wallet_sync_id: str | None = None,
        limit: int = 10,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        """Last N transactions ordered by date desc. Use for "recent transactions" / "last N".

        Use when: user asks "recent transactions", "last 5", "latest", "recent"
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit=limit, days=None,
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
        """Transactions within a date range. dates as "YYYY-MM-DD".

        Use when: user specifies a date range like "this month", "last week", "January", "between dates"
        """
        async with self._sf() as session:
            return await self._fetch_transactions(
                session, wallet_sync_id, limit=None, days=None,
                start_date=start_date, end_date=end_date, type=None, type_group=type_group, include_in_report=include_in_report
            )

    async def get_scheduled(
        self,
        wallet_sync_id: str | None = None,
        days: int | None = 30,
        limit: int | None = None,
    ) -> list[dict]:
        """Scheduled/future transactions (status='scheduled' and date > today).

        Use when: user asks about "upcoming", "scheduled", "future transactions"

        Args:
            wallet_sync_id: Filter by specific wallet
            days: Look ahead N days (default 30). Set to None for all future.
            limit: Max number of results
        """
        async with self._sf() as session:
            conditions = ["t.created_by_user_id = :user_id", "t.is_deleted = false", "t.status = 'scheduled'"]
            params: dict = {"user_id": self._user_id}

            if wallet_sync_id:
                conditions.append("(t.wallet_sync_id = :wallet_sync_id OR t.destination_wallet_sync_id = :wallet_sync_id)")
                params["wallet_sync_id"] = wallet_sync_id

            if days is not None:
                future_date = datetime.now(timezone.utc).date() + timedelta(days=days)
                conditions.append("t.date <= :future_date")
                params["future_date"] = future_date

            where = " AND ".join(conditions)
            limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

            sql = text(f"""
                SELECT t.id, t.sync_id, t.type,
                       'expense' AS type_group,
                       t.amount, t.date, t.note,
                       t.destination_note,
                       t.wallet_sync_id, t.destination_wallet_sync_id,
                       t.category_sync_id,
                       COALESCE(t.category_name, c.name) AS category_name,
                       COALESCE(t.category_name, c.name) AS display_category,
                       t.effect_on_wallet, t.effect_on_destination,
                       t.currency_code, t.currency_symbol,
                       t.converted_amount,
                       t.destination_currency_code, t.destination_currency_symbol,
                       t.destination_converted_amount,
                       t.exchange_rate,
                       t.is_recurring, t.recurring_frequency, t.recurring_transaction_sync_id,
                       t.include_in_report, t.status, t.icon,
                       COALESCE(
                           (SELECT JSON_AGG(JSON_BUILD_OBJECT('sync_id', tg.sync_id, 'name', tg.name))
                            FROM transaction_tags tt
                            JOIN tags tg ON tg.sync_id = tt.tag_sync_id AND tg.is_deleted = false
                            WHERE tt.transaction_sync_id = t.sync_id),
                           '[]'::json
                       ) AS tags
                FROM transactions t
                LEFT JOIN categories c ON c.sync_id::text = t.category_sync_id AND c.deleted_at IS NULL
                WHERE {where}
                ORDER BY t.date ASC
                {limit_clause}
            """)

            result = await session.execute(sql, params)
            rows = result.mappings().all()

            return [
                {
                    "transaction_id": str(row["id"]),
                    "transaction_sync_id": str(row["sync_id"]),
                    "transaction_type": row["type"],
                    "transaction_type_group": row["type_group"],
                    "transaction_amount": _to_decimal(row["amount"]),
                    "transaction_date": row["date"],
                    "transaction_note": row["note"],
                    "transaction_destination_note": row["destination_note"],
                    "transaction_wallet_sync_id": row["wallet_sync_id"],
                    "transaction_destination_wallet_sync_id": row["destination_wallet_sync_id"],
                    "transaction_category_sync_id": row["category_sync_id"],
                    "transaction_category_name": row["category_name"],
                    "transaction_display_category": row["display_category"],
                    "transaction_effect_on_wallet": _to_decimal(row["effect_on_wallet"]) if row["effect_on_wallet"] is not None else None,
                    "transaction_effect_on_destination": _to_decimal(row["effect_on_destination"]) if row["effect_on_destination"] is not None else None,
                    "transaction_currency_code": row["currency_code"],
                    "transaction_currency_symbol": row["currency_symbol"],
                    "transaction_converted_amount": _to_decimal(row["converted_amount"]) if row["converted_amount"] is not None else None,
                    "transaction_destination_currency_code": row["destination_currency_code"],
                    "transaction_destination_currency_symbol": row["destination_currency_symbol"],
                    "transaction_destination_converted_amount": _to_decimal(row["destination_converted_amount"]) if row["destination_converted_amount"] is not None else None,
                    "transaction_exchange_rate": _to_decimal(row["exchange_rate"]) if row["exchange_rate"] is not None else None,
                    "transaction_is_recurring": row["is_recurring"],
                    "transaction_recurring_frequency": row["recurring_frequency"],
                    "transaction_recurring_transaction_sync_id": row["recurring_transaction_sync_id"],
                    "transaction_include_in_report": row["include_in_report"],
                    "transaction_status": row["status"],
                    "transaction_icon": row["icon"] if isinstance(row["icon"], dict) else None,
                    "transaction_tags": row["tags"] if isinstance(row["tags"], list) else [],
                }
                for row in rows
            ]

    async def _fetch_transactions(
        self,
        session,
        wallet_sync_id: str | None = None,
        limit: int | None = None,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        type: str | list[str] | None = None,
        type_group: str | None = None,
        include_in_report: bool | None = None,
    ) -> list[dict]:
        conditions = ["t.created_by_user_id = :user_id", "t.is_deleted = false"]
        params: dict = {"user_id": self._user_id}

        if wallet_sync_id:
            conditions.append("t.wallet_sync_id = :wallet_sync_id")
            params["wallet_sync_id"] = wallet_sync_id

        if type:
            if isinstance(type, list):
                placeholders = ", ".join(f":type_{i}" for i in range(len(type)))
                conditions.append(f"t.type IN ({placeholders})")
                for i, t in enumerate(type):
                    params[f"type_{i}"] = t
            else:
                conditions.append("t.type = :type")
                params["type"] = type

        if type_group == "income":
            conditions.append("t.effect_on_wallet > 0")
        elif type_group == "expense":
            conditions.append("t.effect_on_wallet < 0")

        if include_in_report is not None:
            conditions.append("t.include_in_report = :include_in_report")
            params["include_in_report"] = include_in_report

        if start_date:
            conditions.append("t.date >= :start_date")
            params["start_date"] = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        elif days:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            conditions.append("t.date >= :since")
            params["since"] = since

        if end_date:
            end_date_obj = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
            conditions.append("t.date < :end_date_exclusive")
            params["end_date_exclusive"] = end_date_obj + timedelta(days=1)

        where = " AND ".join(conditions)
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = text(f"""
            SELECT t.id, t.sync_id, t.type,
                   CASE WHEN t.effect_on_wallet > 0 THEN 'income' ELSE 'expense' END AS type_group,
                   t.amount, t.date, t.note,
                   t.destination_note,
                   t.wallet_sync_id, t.destination_wallet_sync_id,
                   t.category_sync_id,
                   COALESCE(t.category_name, c.name) AS category_name,
                   COALESCE(
                       t.category_name,
                       c.name,
                       CASE t.type
                           WHEN 'goalDeposit'          THEN 'ฝากเงิน'
                           WHEN 'goalWithdraw'         THEN 'ถอนเงิน'
                           WHEN 'transfer'             THEN 'โอนเงิน'
                           WHEN 'creditCardPay'        THEN 'จ่ายบัตรเครดิต'
                           WHEN 'creditCardCashAdvance' THEN 'เบิกเงินสดบัตรเครดิต'
                           WHEN 'creditCardCashback'   THEN 'เงินคืนบัตรเครดิต'
                       END
                   ) AS display_category,
                   t.effect_on_wallet, t.effect_on_destination,
                   t.currency_code, t.currency_symbol,
                   t.converted_amount,
                   t.destination_currency_code, t.destination_currency_symbol,
                   t.destination_converted_amount,
                   t.exchange_rate,
                   t.is_recurring, t.recurring_frequency, t.recurring_transaction_sync_id,
                   t.include_in_report, t.status, t.icon,
                   COALESCE(
                       (SELECT JSON_AGG(JSON_BUILD_OBJECT('sync_id', tg.sync_id, 'name', tg.name))
                        FROM transaction_tags tt
                        JOIN tags tg ON tg.sync_id = tt.tag_sync_id AND tg.is_deleted = false
                        WHERE tt.transaction_sync_id = t.sync_id),
                       '[]'::json
                   ) AS tags
            FROM transactions t
            LEFT JOIN categories c ON c.sync_id::text = t.category_sync_id AND c.deleted_at IS NULL
            WHERE {where}
            ORDER BY t.date DESC
            {limit_clause}
        """)

        result = await session.execute(sql, params)
        rows = result.mappings().all()

        return [
            {
                "transaction_id": str(row["id"]),
                "transaction_sync_id": str(row["sync_id"]),
                "transaction_type": row["type"],
                "transaction_type_group": row["type_group"],
                "transaction_amount": _to_decimal(row["amount"]),
                "transaction_date": row["date"],
                "transaction_note": row["note"],
                "transaction_destination_note": row["destination_note"],
                "transaction_wallet_sync_id": row["wallet_sync_id"],
                "transaction_destination_wallet_sync_id": row["destination_wallet_sync_id"],
                "transaction_category_sync_id": row["category_sync_id"],
                "transaction_category_name": row["category_name"],
                "transaction_display_category": row["display_category"],
                "transaction_effect_on_wallet": _to_decimal(row["effect_on_wallet"]) if row["effect_on_wallet"] is not None else None,
                "transaction_effect_on_destination": _to_decimal(row["effect_on_destination"]) if row["effect_on_destination"] is not None else None,
                "transaction_currency_code": row["currency_code"],
                "transaction_currency_symbol": row["currency_symbol"],
                "transaction_converted_amount": _to_decimal(row["converted_amount"]) if row["converted_amount"] is not None else None,
                "transaction_destination_currency_code": row["destination_currency_code"],
                "transaction_destination_currency_symbol": row["destination_currency_symbol"],
                "transaction_destination_converted_amount": _to_decimal(row["destination_converted_amount"]) if row["destination_converted_amount"] is not None else None,
                "transaction_exchange_rate": _to_decimal(row["exchange_rate"]) if row["exchange_rate"] is not None else None,
                "transaction_is_recurring": row["is_recurring"],
                "transaction_recurring_frequency": row["recurring_frequency"],
                "transaction_recurring_transaction_sync_id": row["recurring_transaction_sync_id"],
                "transaction_include_in_report": row["include_in_report"],
                "transaction_status": row["status"],
                "transaction_icon": row["icon"] if isinstance(row["icon"], dict) else None,
                "transaction_tags": row["tags"] if isinstance(row["tags"], list) else [],
            }
            for row in rows
        ]
