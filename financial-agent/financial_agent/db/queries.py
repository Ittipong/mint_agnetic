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
    limit: int | None = 100,
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    conditions = ["t.created_by_user_id = :user_id", "t.is_deleted = false"]
    params: dict = {"user_id": user_id}

    if wallet_sync_id:
        conditions.append("t.wallet_sync_id = :wallet_sync_id")
        params["wallet_sync_id"] = wallet_sync_id

    if start_date:
        conditions.append("t.date >= :start_date")
        params["start_date"] = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    elif days:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        conditions.append("t.date >= :since")
        params["since"] = since

    if end_date:
        conditions.append("t.date <= :end_date")
        params["end_date"] = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    where = " AND ".join(conditions)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    sql = text(f"""
        SELECT t.id, t.sync_id, t.type, t.amount, t.date, t.note,
               t.wallet_sync_id, t.destination_wallet_sync_id,
               t.category_sync_id,
               COALESCE(t.category_name, c.name) AS category_name,
               t.effect_on_wallet, t.effect_on_destination,
               t.currency_code, t.is_recurring, t.include_in_report
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
            # transactions.amount is double precision — convert immediately
            "amount": _to_decimal(row["amount"]),
        }
        for row in rows
    ]


async def fetch_wallet_balance(
    session: AsyncSession,
    user_id: str,
    wallet_sync_id: str,
) -> Decimal:
    sql = text("""
        SELECT gw.initial_balance,
               COALESCE(SUM(t.amount * t.effect_on_wallet), 0) AS txn_total
        FROM general_wallets gw
        LEFT JOIN transactions t
            ON t.wallet_sync_id = gw.sync_id::text
            AND t.is_deleted = false
            AND t.include_in_report = true
        WHERE gw.sync_id::text = :wallet_sync_id
          AND gw.user_id = :user_id
          AND gw.deleted_at IS NULL
        GROUP BY gw.initial_balance
    """)
    result = await session.execute(sql, {"wallet_sync_id": str(wallet_sync_id), "user_id": str(user_id)})
    row = result.mappings().first()
    if not row:
        return Decimal("0")
    return _to_decimal(row["initial_balance"]) + _to_decimal(row["txn_total"])


async def fetch_budgets(
    session: AsyncSession,
    user_id: str,
    active_only: bool = True,
) -> list[dict]:
    conditions = ["b.user_id = :user_id", "b.is_deleted = false"]
    if active_only:
        conditions.append("b.end_date >= NOW()")
    where = " AND ".join(conditions)

    sql = text(f"""
        SELECT b.id, b.sync_id, b.name, b.amount, b.spent_amount,
               b.period, b.start_date, b.end_date, b.currency, b.mode
        FROM budgets b
        WHERE {where}
        ORDER BY b.start_date DESC
    """)
    result = await session.execute(sql, {"user_id": user_id})
    rows = result.mappings().all()
    return [
        {
            **dict(row),
            "id": str(row["id"]),
            "sync_id": str(row["sync_id"]),
            "amount": _to_decimal(row["amount"]),
            "spent_amount": _to_decimal(row["spent_amount"]),
        }
        for row in rows
    ]


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


async def fetch_credit_card_wallets(session: AsyncSession, user_id: str) -> list[dict]:
    sql = text("""
        SELECT id, sync_id, name, credit_limit, initial_used,
               cached_used_amount, billing_cycle_day, payment_due_day,
               currency, created_at
        FROM creditcard_wallets
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
            "credit_limit": _to_decimal(row["credit_limit"]),
            "initial_used": _to_decimal(row["initial_used"]),
            "cached_used_amount": _to_decimal(row["cached_used_amount"]),
        }
        for row in rows
    ]


async def fetch_obligation_wallets(session: AsyncSession, user_id: str) -> list[dict]:
    sql = text("""
        SELECT ow.id, ow.sync_id, ow.name, ow.lender, ow.status,
               ow.obligation_type, ow.obligation_mode, ow.currency,
               ow.monthly_payment, ow.due_day, ow.cached_total_paid,
               opv.principal, opv.outstanding_principal,
               opv.annual_rate, opv.term_months, opv.interest_type
        FROM obligation_wallets ow
        LEFT JOIN obligation_plan_versions opv
            ON opv.obligation_id = ow.id
            AND opv.effective_to IS NULL
            AND opv.deleted_at IS NULL
        WHERE ow.user_id = :user_id
          AND ow.deleted_at IS NULL
          AND ow.status = 'active'
        ORDER BY ow.created_at
    """)
    result = await session.execute(sql, {"user_id": user_id})
    rows = result.mappings().all()
    return [
        {
            **dict(row),
            "id": str(row["id"]),
            "sync_id": str(row["sync_id"]),
            "principal": _to_decimal(row["principal"]),
            "outstanding_principal": _to_decimal(row["outstanding_principal"]),
            "annual_rate": _to_decimal(row["annual_rate"]),
            "monthly_payment": _to_decimal(row["monthly_payment"]),
            "cached_total_paid": _to_decimal(row["cached_total_paid"]),
        }
        for row in rows
    ]


async def fetch_obligation_transactions(
    session: AsyncSession,
    obligation_id: str,
) -> list[dict]:
    sql = text("""
        SELECT id, sync_id, event_type, amount, event_date,
               principal_paid, interest_paid, remaining_balance,
               paid_installments, notes
        FROM obligation_transactions
        WHERE obligation_id = :obligation_id
          AND deleted_at IS NULL
        ORDER BY event_date DESC
    """)
    result = await session.execute(sql, {"obligation_id": obligation_id})
    rows = result.mappings().all()
    return [
        {
            **dict(row),
            "id": str(row["id"]),
            "sync_id": str(row["sync_id"]),
            "amount": _to_decimal(row["amount"]),
            "principal_paid": _to_decimal(row["principal_paid"]),
            "interest_paid": _to_decimal(row["interest_paid"]),
            "remaining_balance": _to_decimal(row["remaining_balance"]),
        }
        for row in rows
    ]


async def fetch_goal_wallets(session: AsyncSession, user_id: str) -> list[dict]:
    sql = text("""
        SELECT id, sync_id, name, initial_balance, cached_balance,
               target_amount, target_date, is_closed, achieved_at,
               currency, start_date, created_at
        FROM goal_wallets
        WHERE user_id = :user_id
          AND deleted_at IS NULL
          AND is_deleted = false
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
            "target_amount": _to_decimal(row["target_amount"]),
        }
        for row in rows
    ]
