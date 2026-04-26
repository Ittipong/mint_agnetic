from decimal import Decimal
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _to_decimal(value) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


async def fetch_transactions(
    session: AsyncSession,
    user_id: str,
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
    params: dict = {"user_id": user_id}

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
        # Use exclusive upper bound (< next day) to include all timestamps on end_date
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
            **dict(row),
            "id": str(row["id"]),
            "sync_id": str(row["sync_id"]),
            "amount": _to_decimal(row["amount"]),
            "converted_amount": _to_decimal(row["converted_amount"]) if row["converted_amount"] is not None else None,
            "destination_converted_amount": _to_decimal(row["destination_converted_amount"]) if row["destination_converted_amount"] is not None else None,
            "exchange_rate": _to_decimal(row["exchange_rate"]) if row["exchange_rate"] is not None else None,
            "tags": row["tags"] if isinstance(row["tags"], list) else [],
            "icon": row["icon"] if isinstance(row["icon"], dict) else None,
        }
        for row in rows
    ]


async def fetch_all_wallet_balances(
    session: AsyncSession,
    user_id: str,
) -> list[dict]:
    sql = text(f"""
        SELECT gw.sync_id::text AS sync_id,
               gw.name,
               gw.currency,
               gw.wallet_category,
               gw.initial_balance,
               gw.icon,
               gw.initial_balance + {_BALANCE_CASE_SQL} AS balance
        FROM general_wallets gw
        LEFT JOIN transactions t
            ON (t.wallet_sync_id = gw.sync_id::text OR t.destination_wallet_sync_id = gw.sync_id::text)
            AND t.is_deleted = false
        WHERE gw.user_id = :user_id
          AND gw.deleted_at IS NULL
        GROUP BY gw.sync_id, gw.name, gw.currency, gw.wallet_category, gw.initial_balance, gw.icon, gw.created_at
        ORDER BY gw.created_at
    """)
    result = await session.execute(sql, {"user_id": str(user_id)})
    rows = result.mappings().all()
    return [
        {
            "sync_id": row["sync_id"],
            "name": row["name"],
            "currency": row["currency"],
            "wallet_category": row["wallet_category"],
            "initial_balance": _to_decimal(row["initial_balance"]),
            "icon": row["icon"],
            "balance": _to_decimal(row["balance"]),
        }
        for row in rows
    ]


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


async def fetch_wallet_balance(
    session: AsyncSession,
    user_id: str,
    wallet_sync_id: str,
) -> Decimal:
    sql = text(f"""
        SELECT gw.initial_balance + {_BALANCE_CASE_SQL} AS balance
        FROM general_wallets gw
        LEFT JOIN transactions t
            ON (t.wallet_sync_id = gw.sync_id::text OR t.destination_wallet_sync_id = gw.sync_id::text)
            AND t.is_deleted = false
        WHERE gw.sync_id::text = :wallet_sync_id
          AND gw.user_id = :user_id
          AND gw.deleted_at IS NULL
        GROUP BY gw.initial_balance
    """)
    result = await session.execute(sql, {"wallet_sync_id": str(wallet_sync_id), "user_id": str(user_id)})
    row = result.mappings().first()
    if not row:
        return Decimal("0")
    return _to_decimal(row["balance"])


async def fetch_general_wallets(session: AsyncSession, user_id: str) -> list[dict]:
    sql = text("""
        SELECT id, sync_id, name, initial_balance, cached_balance,
               currency, wallet_category, created_at
        FROM general_wallets
        WHERE user_id = :user_id AND deleted_at IS NULL
        ORDER BY created_at
    """)
    result = await session.execute(sql, {"user_id": user_id})
    rows = result.mappings().all()
    return [
        {
            **dict(row),
            "id": str(row["id"]),
            "sync_id": str(row["sync_id"]),
            "initial_balance": _to_decimal(row["initial_balance"]),
            "cached_balance": _to_decimal(row["cached_balance"]),
        }
        for row in rows
    ]

