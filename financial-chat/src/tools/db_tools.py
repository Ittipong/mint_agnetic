"""PostgreSQL tools for the agent — queries mint_money_dev (backend DB)."""

import json
import asyncpg
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from src.config import settings

_pools: dict[str, asyncpg.Pool] = {}


async def _get_pool() -> asyncpg.Pool:
    url = settings.backend_database_url
    if url not in _pools:
        _pools[url] = await asyncpg.create_pool(url, min_size=2, max_size=5)
    return _pools[url]


@tool
async def get_recent_transactions(
    limit: int = 10,
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's most recent transactions. Returns type, amount, date, category, and wallet."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT t.type, t.amount, t.date, t.note, t.status,
                   c.name AS category_name,
                   w.name AS wallet_name
            FROM transactions t
            LEFT JOIN categories c ON t.category_sync_id = c.sync_id
            LEFT JOIN (
                SELECT sync_id, name FROM general_wallets
                UNION ALL
                SELECT sync_id, name FROM goal_wallets
                UNION ALL
                SELECT sync_id, name FROM creditcard_wallets
            ) w ON t.wallet_sync_id::uuid = w.sync_id
            WHERE t.created_by_user_id = $1::uuid
              AND t.is_deleted = false
            ORDER BY t.date DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
    if not rows:
        return "No transactions found."
    lines = []
    for r in rows:
        lines.append(
            f"- [{r['type']}] {r['amount']:,.2f} | {r['date'].date()} | "
            f"{r['category_name'] or 'uncategorized'} | {r['wallet_name'] or 'unknown'}"
        )
    return "\n".join(lines) if lines else "No transactions."


@tool
async def get_budget_status(
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's budget status for the current period. Shows budget name, limit, spent, and remaining."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT name, amount, spent_amount, period
            FROM budgets
            WHERE user_id = $1::uuid
              AND is_deleted = false
            ORDER BY start_date DESC
            LIMIT 10
            """,
            user_id,
        )
    if not rows:
        return "No budgets found."
    lines = []
    for r in rows:
        remaining = float(r["amount"]) - float(r["spent_amount"])
        pct = (float(r["spent_amount"]) / float(r["amount"]) * 100) if float(r["amount"]) > 0 else 0
        lines.append(
            f"- {r['name']} ({r['period']}): {r['spent_amount']:,.2f} / "
            f"{r['amount']:,.2f} | remaining={remaining:,.2f} ({pct:.0f}% used)"
        )
    return "\n".join(lines) if lines else "No budgets."


@tool
async def get_savings_goals(
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's savings goals. Shows goal name, current balance, target amount, and progress."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT name, initial_balance, target_amount, target_date, is_closed
            FROM goal_wallets
            WHERE user_id = $1::uuid
              AND is_closed = false
            ORDER BY created_at DESC
            LIMIT 10
            """,
            user_id,
        )
    if not rows:
        return "No savings goals found."
    lines = []
    for r in rows:
        progress = (float(r["initial_balance"]) / float(r["target_amount"]) * 100) \
            if float(r["target_amount"]) > 0 else 0
        lines.append(
            f"- {r['name']}: {r['initial_balance']:,.2f} / "
            f"{r['target_amount']:,.2f} ({progress:.1f}%)"
            + (f" | target: {r['target_date']}" if r["target_date"] else "")
        )
    return "\n".join(lines) if lines else "No active goals."


# =============================================================================
# Template-based tools (Phase 1) — Fixed SQL templates per intent
# =============================================================================

# SPENT_BUDGET Template
SPENT_BUDGET_SQL = """
WITH user_wallets AS (
    SELECT sync_id::text AS sync_id FROM general_wallets
    WHERE user_id = $1::uuid AND deleted_at IS NULL
    UNION
    SELECT sync_id::text AS sync_id FROM creditcard_wallets
    WHERE user_id = $1::uuid AND deleted_at IS NULL
),
month_expenses AS (
    SELECT COALESCE(SUM(t.amount), 0)::numeric AS total_spent
    FROM transactions t
    WHERE t.created_by_user_id = $1::uuid
      AND t.type = 'expense'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
      AND t.date >= DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * $2::int)
      AND t.date < DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * ($2::int + 1))
),
month_budget AS (
    SELECT b.amount AS budget_amount
    FROM budgets b
    JOIN budget_wallets bw ON b.sync_id = bw.budget_sync_id
    WHERE bw.wallet_sync_id = ANY(SELECT sync_id FROM user_wallets)
      AND b.period = 'monthly'
      AND b.is_deleted = false
      AND b.start_date <= DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * ($2::int + 1))
      AND (b.end_date IS NULL OR b.end_date >= DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * $2::int))
    LIMIT 1
),
calculations AS (
    SELECT
        me.total_spent,
        mb.budget_amount,
        mb.budget_amount - me.total_spent AS remaining,
        CASE WHEN mb.budget_amount > 0
             THEN ROUND((me.total_spent::numeric / NULLIF(mb.budget_amount, 0)) * 100, 1)
             ELSE NULL END AS percent_used,
        GREATEST(0, EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * ($2::int + 1)) - INTERVAL '1 day')
            - CURRENT_DATE)) AS days_left,
        CASE WHEN mb.budget_amount > 0 AND EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * ($2::int + 1)) - INTERVAL '1 day') - CURRENT_DATE) > 0
             THEN ROUND((mb.budget_amount - me.total_spent::numeric) /
                GREATEST(1, EXTRACT(DAY FROM (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * ($2::int + 1)) - INTERVAL '1 day') - CURRENT_DATE)))
             ELSE NULL END AS daily_allowance
    FROM month_expenses me, month_budget mb
)
SELECT
    total_spent,
    budget_amount,
    remaining,
    percent_used,
    days_left,
    daily_allowance,
    TO_CHAR(DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * $2::int), 'TMMonth') AS month_name,
    EXTRACT(YEAR FROM DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month' * $2::int)) AS year
FROM calculations;
"""


@tool
async def get_spent_budget(
    month_offset: int = 0,
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's spending vs budget for a specific month.

    Args:
        month_offset: 0 = current month, -1 = last month, 1 = next month

    Returns:
        JSON string with spent, budget, remaining, percent_used, days_left, daily_allowance
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(SPENT_BUDGET_SQL, user_id, month_offset)
        if not row or (row["total_spent"] == 0 and row["budget_amount"] is None):
            return json.dumps({"spent": 0, "budget": None, "remaining": None, "month_name": None})

        return json.dumps({
            "spent": float(row["total_spent"]) if row["total_spent"] else 0,
            "budget": float(row["budget_amount"]) if row["budget_amount"] else None,
            "remaining": float(row["remaining"]) if row["remaining"] is not None else None,
            "percent_used": float(row["percent_used"]) if row["percent_used"] is not None else None,
            "days_left": int(row["days_left"]) if row["days_left"] is not None else None,
            "daily_allowance": float(row["daily_allowance"]) if row["daily_allowance"] is not None else None,
            "month_name": row["month_name"],
            "year": int(row["year"]) if row["year"] else None,
        })


# WEEKLY_SUMMARY Template
WEEKLY_SUMMARY_SQL = """
WITH week_range AS (
    SELECT
        DATE_TRUNC('week', CURRENT_DATE + INTERVAL '1 week' * $2::int) AS week_start,
        DATE_TRUNC('week', CURRENT_DATE + INTERVAL '1 week' * $2::int) + INTERVAL '6 days' AS week_end,
        DATE_TRUNC('week', CURRENT_DATE + INTERVAL '1 week' * ($2::int - 1)) AS last_week_start,
        DATE_TRUNC('week', CURRENT_DATE + INTERVAL '1 week' * ($2::int - 1)) + INTERVAL '6 days' AS last_week_end
),
weekly_total AS (
    SELECT COALESCE(SUM(t.amount), 0) AS total_spent
    FROM transactions t
    CROSS JOIN week_range wr
    WHERE t.created_by_user_id = $1::uuid
      AND t.type = 'expense'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
      AND t.date >= wr.week_start AND t.date <= wr.week_end
),
category_breakdown AS (
    SELECT
        COALESCE(t.category_name, cat.name, 'อื่นๆ') AS category,
        COALESCE(SUM(t.amount), 0) AS amount,
        COUNT(*) AS transaction_count
    FROM transactions t
    CROSS JOIN week_range wr
    LEFT JOIN categories cat ON t.category_sync_id = cat.sync_id
    WHERE t.created_by_user_id = $1::uuid
      AND t.type = 'expense'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
      AND t.date >= wr.week_start AND t.date <= wr.week_end
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 5
),
last_week_total AS (
    SELECT COALESCE(SUM(t.amount), 0) AS last_week_spent
    FROM transactions t
    CROSS JOIN week_range wr
    WHERE t.created_by_user_id = $1::uuid
      AND t.type = 'expense'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
      AND t.date >= wr.last_week_start AND t.date <= wr.last_week_end
),
change_calc AS (
    SELECT
        wt.total_spent,
        lwt.last_week_spent,
        CASE WHEN lwt.last_week_spent > 0
             THEN ROUND(((wt.total_spent::numeric - lwt.last_week_spent::numeric) / lwt.last_week_spent::numeric) * 100, 1)
             ELSE NULL END AS week_over_week_change
    FROM weekly_total wt, last_week_total lwt
)
SELECT
    cc.total_spent,
    cc.last_week_spent,
    cc.week_over_week_change,
    TO_CHAR(wr.week_start, 'TMMonth DD') AS week_start_display,
    TO_CHAR(wr.week_end, 'TMMonth DD, YYYY') AS week_end_display,
    (SELECT json_agg(json_build_object('category', cb.category, 'amount', cb.amount, 'count', cb.transaction_count))
     FROM category_breakdown cb) AS category_breakdown_json
FROM change_calc cc, week_range wr;
"""


@tool
async def get_weekly_summary(
    week_offset: int = 0,
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's weekly spending breakdown by category.

    Args:
        week_offset: 0 = current week, -1 = last week, -2 = 2 weeks ago

    Returns:
        JSON string with total_spent, last_week_spent, week_over_week_change, category_breakdown
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(WEEKLY_SUMMARY_SQL, user_id, week_offset)
        if not row:
            return json.dumps({"total_spent": 0, "category_breakdown": []})

        categories = json.loads(row["category_breakdown_json"]) if row["category_breakdown_json"] else []
        return json.dumps({
            "total_spent": float(row["total_spent"]) if row["total_spent"] else 0,
            "last_week_spent": float(row["last_week_spent"]) if row["last_week_spent"] else 0,
            "week_over_week_change": float(row["week_over_week_change"]) if row["week_over_week_change"] is not None else None,
            "week_start_display": row["week_start_display"],
            "week_end_display": row["week_end_display"],
            "category_breakdown": categories,
        })


# DEBT_BALANCE Template
DEBT_BALANCE_SQL = """
WITH active_obligations AS (
    SELECT
        ow.name,
        ow.monthly_payment,
        ow.due_day,
        (
            SELECT opv.outstanding_principal
            FROM obligation_plan_versions opv
            WHERE opv.obligation_id = ow.id
              AND opv.effective_from <= CURRENT_DATE
              AND (opv.effective_to IS NULL OR opv.effective_to >= CURRENT_DATE)
            ORDER BY opv.version DESC
            LIMIT 1
        ) AS outstanding_principal,
        (
            SELECT ot.event_date
            FROM obligation_transactions ot
            WHERE ot.obligation_id = ow.id
              AND ot.event_type = 'payment'
              AND ot.deleted_at IS NULL
            ORDER BY ot.event_date DESC
            LIMIT 1
        ) AS last_payment_date
    FROM obligation_wallets ow
    WHERE ow.user_id = $1::uuid
      AND ow.status = 'active'
      AND ow.deleted_at IS NULL
),
creditcard_debts AS (
    SELECT
        cc.name,
        cc.cached_used_amount AS outstanding_principal,
        cc.credit_limit,
        cc.payment_due_day AS due_day
    FROM creditcard_wallets cc
    WHERE cc.user_id = $1::uuid
      AND cc.deleted_at IS NULL
      AND cc.cached_used_amount > 0
),
combined AS (
    SELECT
        name,
        outstanding_principal,
        monthly_payment,
        due_day,
        last_payment_date,
        NULL::numeric AS credit_limit,
        NULL::numeric AS utilization_percent,
        'obligation' AS debt_type
    FROM active_obligations
    UNION ALL
    SELECT
        name,
        outstanding_principal,
        0 AS monthly_payment,
        due_day,
        NULL AS last_payment_date,
        credit_limit,
        CASE WHEN credit_limit > 0 THEN ROUND((outstanding_principal / credit_limit) * 100, 1) ELSE NULL END,
        'creditcard' AS debt_type
    FROM creditcard_debts
),
aggregates AS (
    SELECT
        COUNT(*) AS total_debts,
        SUM(outstanding_principal) AS total_outstanding,
        SUM(monthly_payment) AS total_monthly_payment,
        MIN(due_day) AS next_due_day,
        CASE WHEN MIN(due_day) IS NOT NULL THEN
            CASE
                WHEN MIN(due_day) >= EXTRACT(DAY FROM CURRENT_DATE)::INTEGER
                THEN DATE_TRUNC('month', CURRENT_DATE) + (MIN(due_day) - 1) * INTERVAL '1 day'
                ELSE DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (MIN(due_day) - 1) * INTERVAL '1 day'
            END
        ELSE NULL END AS next_due_date
    FROM combined
)
SELECT
    a.total_debts,
    a.total_outstanding,
    a.total_monthly_payment,
    a.next_due_day,
    a.next_due_date,
    (SELECT json_agg(json_build_object(
        'name', cd.name,
        'outstanding', cd.outstanding_principal,
        'monthly_payment', cd.monthly_payment,
        'due_day', cd.due_day,
        'last_payment', cd.last_payment_date,
        'credit_limit', cd.credit_limit,
        'utilization_percent', cd.utilization_percent,
        'debt_type', cd.debt_type
    )) FROM combined cd) AS debts_json
FROM aggregates a;
"""


@tool
async def get_debt_balance(
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's total outstanding debt across all obligations and credit cards.

    Returns:
        JSON string with total_outstanding, total_monthly_payment, next_due_date, debts array
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(DEBT_BALANCE_SQL, user_id)
        if not row or row["total_debts"] == 0:
            return json.dumps({"total_debts": 0, "debts": []})

        debts = json.loads(row["debts_json"]) if row["debts_json"] else []
        return json.dumps({
            "total_outstanding": float(row["total_outstanding"]) if row["total_outstanding"] else 0,
            "total_monthly_payment": float(row["total_monthly_payment"]) if row["total_monthly_payment"] else 0,
            "next_due_day": int(row["next_due_day"]) if row["next_due_day"] else None,
            "next_due_date": str(row["next_due_date"]) if row["next_due_date"] else None,
            "debts": debts,
        })


# GOAL_PROGRESS Template
GOAL_PROGRESS_SQL = """
SELECT
    gw.sync_id AS goal_id,
    gw.name,
    gw.target_amount,
    gw.cached_balance AS current_amount,
    CASE
        WHEN COALESCE(gw.cached_balance, 0) >= COALESCE(gw.target_amount, 0) THEN 0
        ELSE COALESCE(gw.target_amount, 0) - COALESCE(gw.cached_balance, 0)
    END AS amount_remaining,
    LEAST(100, ROUND(
        CASE
            WHEN COALESCE(gw.cached_balance, 0) >= COALESCE(gw.target_amount, 0) THEN 100
            ELSE (COALESCE(gw.cached_balance, 0)::numeric / NULLIF(COALESCE(gw.target_amount, 0), 0)) * 100
        END, 1
    )) AS progress_percent,
    gw.target_date,
    CASE
        WHEN gw.target_date IS NULL THEN NULL
        ELSE GREATEST(0, (gw.target_date - CURRENT_DATE)::integer)
    END AS days_remaining,
    CASE
        WHEN gw.target_date IS NULL OR COALESCE(gw.cached_balance, 0) >= COALESCE(gw.target_amount, 0) THEN NULL
        ELSE ROUND(
            (COALESCE(gw.target_amount, 0) - COALESCE(gw.cached_balance, 0))::numeric /
            NULLIF(((gw.target_date - CURRENT_DATE)::integer / 30.0), 0)
        , 0)
    END AS monthly_required,
    COALESCE(gw.cached_balance, 0) >= COALESCE(gw.target_amount, 0) AS is_achieved
FROM goal_wallets gw
WHERE gw.user_id = $1::uuid
  AND gw.is_deleted = false
  AND gw.is_closed = false
ORDER BY
    CASE WHEN gw.achieved_at IS NULL THEN 0 ELSE 1 END,
    gw.target_date ASC NULLS LAST;
"""


@tool
async def get_goal_progress(
    goal_id: str | None = None,
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's savings goal(s) progress.

    Args:
        goal_id: Optional specific goal UUID (null = all goals)

    Returns:
        JSON string with goals array containing name, target, current, progress_percent, is_achieved
    """
    # LLM may pass the string "null"/"none" instead of Python None
    if goal_id and goal_id.lower() in ("null", "none", ""):
        goal_id = None

    pool = await _get_pool()
    async with pool.acquire() as conn:
        if goal_id:
            single_sql = GOAL_PROGRESS_SQL.replace(
                "WHERE gw.user_id = $1::uuid",
                "WHERE gw.user_id = $1::uuid\n  AND gw.sync_id = $2::uuid",
            )
            rows = await conn.fetch(single_sql, user_id, goal_id)
        else:
            rows = await conn.fetch(GOAL_PROGRESS_SQL, user_id)

        if not rows:
            return json.dumps({"goals": [], "total_goals": 0})

        goals = []
        for r in rows:
            goals.append({
                "id": str(r["goal_id"]),
                "name": r["name"],
                "target": float(r["target_amount"]) if r["target_amount"] else 0,
                "current": float(r["current_amount"]) if r["current_amount"] else 0,
                "remaining": float(r["amount_remaining"]) if r["amount_remaining"] else 0,
                "progress_percent": float(r["progress_percent"]) if r["progress_percent"] else 0,
                "target_date": str(r["target_date"]) if r["target_date"] else None,
                "days_remaining": int(r["days_remaining"]) if r["days_remaining"] is not None else None,
                "monthly_required": float(r["monthly_required"]) if r["monthly_required"] is not None else None,
                "is_achieved": r["is_achieved"],
            })

        return json.dumps({
            "goals": goals,
            "total_goals": len(goals),
            "achieved_goals": sum(1 for g in goals if g["is_achieved"]),
            "total_saved": sum(g["current"] for g in goals),
        })


# PAYDAY Template
PAYDAY_SQL = """
WITH recurring_payday AS (
    SELECT
        rt.day_of_month,
        rt.next_occurrence,
        rt.amount
    FROM recurring_transactions rt
    WHERE rt.created_by_user_id = $1::uuid
      AND rt.type = 'income'
      AND rt.status = 'active'
      AND rt.is_deleted = false
    ORDER BY rt.day_of_month
    LIMIT 1
),
historical_payday AS (
    SELECT
        EXTRACT(DAY FROM t.date)::INTEGER AS day_of_month,
        COUNT(*) AS frequency,
        AVG(t.amount) AS avg_amount
    FROM transactions t
    WHERE t.created_by_user_id = $1::uuid
      AND t.type = 'income'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
      AND t.date >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 3
),
recurring_amount AS (
    SELECT amount FROM recurring_transactions
    WHERE created_by_user_id = $1::uuid
      AND type = 'income'
      AND status = 'active'
      AND is_deleted = false
    LIMIT 1
),
next_payday_calc AS (
    SELECT
        c.day_of_month,
        c.source,
        CASE
            WHEN c.day_of_month >= EXTRACT(DAY FROM CURRENT_DATE)::INTEGER
            THEN DATE_TRUNC('month', CURRENT_DATE) + (c.day_of_month - 1) * INTERVAL '1 day'
            ELSE DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (c.day_of_month - 1) * INTERVAL '1 day'
        END AS next_payday,
        GREATEST(0, EXTRACT(DAY FROM (
            CASE
                WHEN c.day_of_month >= EXTRACT(DAY FROM CURRENT_DATE)::INTEGER
                THEN DATE_TRUNC('month', CURRENT_DATE) + (c.day_of_month - 1) * INTERVAL '1 day'
                ELSE DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (c.day_of_month - 1) * INTERVAL '1 day'
            END - CURRENT_DATE
        ))::INTEGER) AS days_until,
        COALESCE(r.amount, (
            SELECT AVG(amount)
            FROM transactions
            WHERE created_by_user_id = $1::uuid
              AND type = 'income'
              AND status = 'confirmed'
              AND is_deleted = false
              AND date >= CURRENT_DATE - INTERVAL '6 months'
        )) AS typical_amount
    FROM (
        SELECT day_of_month, 'recurring' AS source FROM recurring_payday
        UNION ALL
        SELECT day_of_month, 'historical' AS source FROM historical_payday
    ) c
    LEFT JOIN recurring_amount r ON c.source = 'recurring'
    ORDER BY c.source = 'recurring' DESC, c.day_of_month DESC
    LIMIT 1
)
SELECT
    np.day_of_month AS typical_payday,
    np.source,
    np.next_payday,
    np.days_until,
    np.typical_amount,
    CASE WHEN np.days_until = 0 THEN true ELSE false END AS is_today,
    CASE WHEN np.days_until = 1 THEN true ELSE false END AS is_tomorrow,
    (SELECT json_agg(json_build_object('day', day_of_month, 'freq', frequency, 'avg', avg_amount))
     FROM (SELECT day_of_month, frequency, avg_amount FROM historical_payday) hp) AS historical_patterns
FROM next_payday_calc np;
"""


@tool
async def get_payday(
    user_id: Annotated[str, InjectedState("user_id")] = "",
) -> str:
    """Get the user's payday information from recurring income or historical patterns.

    Returns:
        JSON string with next_payday, days_until, typical_payday, typical_amount, source
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(PAYDAY_SQL, user_id)
        if not row or row["typical_payday"] is None:
            return json.dumps({"payday": None, "message": "ยังไม่เห็น pattern เงินเดือนเลยนะ"})

        patterns = json.loads(row["historical_patterns"]) if row["historical_patterns"] else []
        return json.dumps({
            "next_payday": str(row["next_payday"]) if row["next_payday"] else None,
            "days_until": int(row["days_until"]) if row["days_until"] is not None else None,
            "typical_payday": int(row["typical_payday"]) if row["typical_payday"] else None,
            "typical_amount": float(row["typical_amount"]) if row["typical_amount"] else None,
            "source": row["source"],
            "is_today": row["is_today"],
            "is_tomorrow": row["is_tomorrow"],
            "historical_patterns": patterns,
        })
