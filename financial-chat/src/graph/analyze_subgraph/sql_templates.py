"""SQL templates per metric — never produced by the LLM.

Each builder returns (sql, params) ready for asyncpg. All numeric aggregations
cast `amount` (double precision in the existing schema) to `numeric` so the
result lands as Decimal — preventing float drift on sums.

Filters applied uniformly:
- t.is_deleted = false
- t.status = 'confirmed'
- t.user_id matches the caller
- t.include_in_report = true (income/expense only — never for transfer/balance)

Joins:
- t.wallet_sync_id (text) ↔ general_wallets.sync_id (uuid) — cast on the wallet side
- t.category_sync_id (text) ↔ categories.sync_id (text) — direct
- transaction_tags links transaction_sync_id ↔ tag_sync_id, both text
"""

from __future__ import annotations

from typing import Any

from src.graph.analyze_subgraph.schemas import QuerySpec


# ── Filter fragments ─────────────────────────────────────────────────────────


def _wallet_filter(spec: QuerySpec, params: list[Any]) -> str:
    if not spec.wallets:
        return ""
    ids = [w.sync_id for w in spec.wallets]
    params.append(ids)
    return f"AND t.wallet_sync_id = ANY(${len(params)}::text[])"


def _category_filter(spec: QuerySpec, params: list[Any]) -> str:
    """Filter by display name, not sync_id.

    Categories are duplicated per wallet (each wallet owns its own copy of the
    system categories), so a single sync_id only covers transactions in one
    wallet. We filter on `transactions.category_name` (denormalized) instead
    so 'อาหาร' matches across every wallet.
    """
    if not spec.categories:
        return ""
    names = [c.display_name for c in spec.categories]
    params.append(names)
    return f"AND t.category_name = ANY(${len(params)}::text[])"


def _tag_filter(spec: QuerySpec, params: list[Any]) -> str:
    if not spec.tags:
        return ""
    ids = [t.sync_id for t in spec.tags]
    params.append(ids)
    return (
        f"AND EXISTS (SELECT 1 FROM transaction_tags tt "
        f"WHERE tt.transaction_sync_id = t.sync_id "
        f"AND tt.tag_sync_id = ANY(${len(params)}::text[]) "
        f"AND tt.is_deleted = false)"
    )


def _currency_filter(spec: QuerySpec, params: list[Any]) -> str:
    if spec.currency == "ALL":
        return ""
    params.append(spec.currency)
    return f"AND t.currency_code = ${len(params)}"


def _common_filters(spec: QuerySpec, params: list[Any]) -> str:
    parts = [
        _wallet_filter(spec, params),
        _category_filter(spec, params),
        _tag_filter(spec, params),
        _currency_filter(spec, params),
    ]
    return "\n        ".join(p for p in parts if p)


def _base_params(spec: QuerySpec, user_id: str) -> list[Any]:
    """First 3 params are always: $1 user_id, $2 start, $3 end."""
    return [user_id, spec.time_range.start, spec.time_range.end]


# ── Metric builders ──────────────────────────────────────────────────────────


def _sum_by_type(spec: QuerySpec, user_id: str, type_value: str) -> tuple[str, list]:
    params = _base_params(spec, user_id)
    params.append(type_value)
    type_idx = len(params)
    extra = _common_filters(spec, params)
    sql = f"""
        SELECT
            COALESCE(t.currency_code, 'THB') AS currency,
            SUM(t.amount::numeric)           AS amount,
            COUNT(*)                          AS cnt
        FROM transactions t
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.type = ${type_idx}
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          AND t.include_in_report = true
          {extra}
        GROUP BY COALESCE(t.currency_code, 'THB')
        ORDER BY currency
    """
    return sql, params


def build_sum_income(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    return _sum_by_type(spec, user_id, "income")


def build_sum_expense(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    return _sum_by_type(spec, user_id, "expense")


def build_count(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    params = _base_params(spec, user_id)
    extra = _common_filters(spec, params)
    sql = f"""
        SELECT
            COALESCE(t.currency_code, 'THB') AS currency,
            COUNT(*)                          AS cnt
        FROM transactions t
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          {extra}
        GROUP BY COALESCE(t.currency_code, 'THB')
        ORDER BY currency
    """
    return sql, params


def build_list(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Detailed rows. `order_by` + `limit` come from the planner so users can
    ask for "top 5 by amount" or "latest 20 by date" without a new metric."""
    params = _base_params(spec, user_id)
    extra = _common_filters(spec, params)
    order_clause = (
        "ORDER BY t.amount::numeric DESC, t.date DESC"
        if spec.order_by == "amount_desc"
        else "ORDER BY t.date DESC"
    )
    cap = max(1, min(spec.limit or 50, 100))
    sql = f"""
        SELECT
            t.sync_id            AS sync_id,
            t.date::date         AS date,
            t.type               AS type,
            t.amount::numeric    AS amount,
            COALESCE(t.currency_code, 'THB') AS currency,
            t.note               AS note,
            t.category_name      AS category_name,
            w.name               AS wallet_name
        FROM transactions t
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          {extra}
        {order_clause}
        LIMIT {cap}
    """
    return sql, params


def build_balance(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Per-wallet balance as of `time_range.end`.

    initial_balance + sum of (effect_on_wallet * amount) for transactions
    on or before the end date. Uses effect_on_wallet so transfers behave correctly.
    """
    params = [user_id, spec.time_range.end]
    wallet_clause = ""
    if spec.wallets:
        ids = [w.sync_id for w in spec.wallets]
        params.append(ids)
        wallet_clause = f"AND w.sync_id::text = ANY(${len(params)}::text[])"

    currency_clause = ""
    if spec.currency != "ALL":
        params.append(spec.currency)
        currency_clause = f"AND w.currency = ${len(params)}"

    sql = f"""
        SELECT
            w.sync_id::text                                              AS wallet_sync_id,
            w.name                                                        AS wallet_name,
            w.currency                                                    AS currency,
            (w.initial_balance + COALESCE(SUM(
                t.effect_on_wallet * t.amount::numeric
            ) FILTER (
                WHERE t.is_deleted = false
                  AND t.status = 'confirmed'
                  AND t.date < ($2::date + INTERVAL '1 day')
            ), 0))                                                        AS amount
        FROM general_wallets w
        LEFT JOIN transactions t ON t.wallet_sync_id = w.sync_id::text
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          {wallet_clause}
          {currency_clause}
        GROUP BY w.sync_id, w.name, w.currency, w.initial_balance
        ORDER BY w.name
    """
    return sql, params


def build_sum_by_category(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Group total expense (default) by category. Income vs expense decided
    by metric — falls back to expense unless the planner asked for income."""
    params = _base_params(spec, user_id)
    # default to expense breakdown (most common ask)
    params.append("expense")
    type_idx = len(params)
    extra = _common_filters(spec, params)
    sql = f"""
        SELECT
            COALESCE(t.category_name, '(uncategorized)') AS bucket,
            COALESCE(t.currency_code, 'THB')             AS currency,
            SUM(t.amount::numeric)                        AS amount,
            COUNT(*)                                      AS cnt
        FROM transactions t
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.type = ${type_idx}
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          AND t.include_in_report = true
          {extra}
        GROUP BY COALESCE(t.category_name, '(uncategorized)'),
                 COALESCE(t.currency_code, 'THB')
        ORDER BY amount DESC
        LIMIT 30
    """
    return sql, params


def build_sum_by_wallet(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    params = _base_params(spec, user_id)
    params.append("expense")
    type_idx = len(params)
    extra = _common_filters(spec, params)
    sql = f"""
        SELECT
            w.name                            AS bucket,
            COALESCE(t.currency_code, 'THB') AS currency,
            SUM(t.amount::numeric)            AS amount,
            COUNT(*)                          AS cnt
        FROM transactions t
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.user_id = $1
          AND w.deleted_at IS NULL
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.type = ${type_idx}
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          AND t.include_in_report = true
          {extra}
        GROUP BY w.name, COALESCE(t.currency_code, 'THB')
        ORDER BY amount DESC
    """
    return sql, params


# ── Dispatcher ───────────────────────────────────────────────────────────────


# ── Budget metrics ───────────────────────────────────────────────────────────


def build_budget_list(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Every non-deleted budget the user has — definition only, no compute.

    No date filter: 'งบที่ตั้งไว้มีอะไรบ้าง' should answer with the full set,
    even ones that have already ended. Optionally narrowed by
    `budget_name_phrase` (ILIKE) when the user named a specific budget.
    """
    params: list = [user_id]
    name_clause = ""
    if spec.budget_name_phrase:
        params.append(f"%{spec.budget_name_phrase}%")
        name_clause = f"AND b.name ILIKE ${len(params)}"
    sql = f"""
        SELECT
            b.sync_id::text         AS sync_id,
            b.name                   AS name,
            b.amount::numeric        AS amount,
            b.currency               AS currency,
            b.period                 AS period,
            b.start_date::date       AS start_date,
            b.end_date::date         AS end_date,
            b.mode                   AS mode
        FROM budgets b
        WHERE b.is_deleted = false
          AND b.user_id = $1
          {name_clause}
        ORDER BY b.start_date DESC
    """
    return sql, params


def build_budget_remaining(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Per-budget amount/spent/remaining/pct, scoped to the requested window.

    Budget filter: budgets whose [start_date, end_date] overlap the requested
    `time_range`. So "งบเดือนนี้" picks budgets covering this month, "งบเมษา"
    picks budgets covering April, "all-time" picks every non-deleted budget.

    Spent-amount filter:
      - wallet match: `t.wallet_sync_id IN (budget's wallet sync_ids)` (or no
        wallet filter when the budget has no wallets).
      - category match by NAME, not sync_id. Categories are duplicated per
        wallet (each wallet has its own copy), so the budget's stored
        category_sync_ids cover only one wallet — using those alone misses
        every transaction in other wallets that share the category name. We
        join `budget_categories → categories` to get the canonical name set,
        then compare `transactions.category_name` against it.
      - mode `expenseOnly` restricts to `t.type='expense'`.
      - currency must match the budget's declared currency (no FX).
    """
    params: list = [user_id, spec.time_range.start, spec.time_range.end]
    name_clause = ""
    if spec.budget_name_phrase:
        params.append(f"%{spec.budget_name_phrase}%")
        name_clause = f"AND b.name ILIKE ${len(params)}"
    sql = f"""
        WITH scoped_budgets AS (
            SELECT
                b.sync_id           AS budget_sync_id,
                b.name              AS name,
                b.amount::numeric   AS amount,
                b.currency          AS currency,
                b.period            AS period,
                b.start_date        AS start_date,
                b.end_date          AS end_date,
                b.mode              AS mode
            FROM budgets b
            WHERE b.is_deleted = false
              AND b.user_id = $1
              AND b.start_date::date <= $3::date
              AND b.end_date::date   >= $2::date
              {name_clause}
        ),
        budget_filters AS (
            SELECT
                sb.budget_sync_id,
                array_remove(array_agg(DISTINCT bw.wallet_sync_id), NULL) AS wallet_ids,
                array_remove(
                    array_agg(DISTINCT c.name) FILTER (WHERE c.name IS NOT NULL),
                    NULL
                ) AS category_names
            FROM scoped_budgets sb
            LEFT JOIN budget_wallets bw
                   ON bw.budget_sync_id = sb.budget_sync_id AND bw.is_deleted = false
            LEFT JOIN budget_categories bc
                   ON bc.budget_sync_id = sb.budget_sync_id AND bc.is_deleted = false
            LEFT JOIN categories c
                   ON c.sync_id = bc.category_sync_id AND c.is_deleted = false
            GROUP BY sb.budget_sync_id
        ),
        budget_spent AS (
            SELECT
                sb.budget_sync_id,
                COALESCE(SUM(t.amount::numeric), 0) AS spent
            FROM scoped_budgets sb
            LEFT JOIN budget_filters bf ON bf.budget_sync_id = sb.budget_sync_id
            LEFT JOIN transactions t
                   ON t.is_deleted = false
                  AND t.status = 'confirmed'
                  AND t.include_in_report = true
                  AND (sb.mode <> 'expenseOnly' OR t.type = 'expense')
                  AND t.date >= sb.start_date
                  AND t.date <  (sb.end_date + INTERVAL '1 day')
                  AND (cardinality(bf.wallet_ids)     = 0 OR t.wallet_sync_id = ANY(bf.wallet_ids))
                  AND (cardinality(bf.category_names) = 0 OR t.category_name  = ANY(bf.category_names))
                  AND COALESCE(t.currency_code, 'THB') = sb.currency
            GROUP BY sb.budget_sync_id
        )
        SELECT
            sb.budget_sync_id::text             AS sync_id,
            sb.name                              AS name,
            sb.amount                            AS amount,
            COALESCE(bs.spent, 0)                AS spent,
            (sb.amount - COALESCE(bs.spent, 0))  AS remaining,
            CASE WHEN sb.amount > 0
                 THEN ROUND((COALESCE(bs.spent, 0) / sb.amount) * 100, 1)
                 ELSE 0 END                      AS pct_used,
            -- days_remaining: clamp at 0 so an ended budget reads as "expired"
            -- rather than producing a negative number.
            GREATEST(0, (sb.end_date::date - CURRENT_DATE)) AS days_remaining,
            -- daily_allowance: NULL when the budget already ended (so the
            -- responder can render it as "expired" instead of dividing by zero).
            CASE
                WHEN sb.end_date::date < CURRENT_DATE THEN NULL
                WHEN GREATEST(1, (sb.end_date::date - CURRENT_DATE + 1)) > 0
                    THEN ROUND(
                        GREATEST(0, sb.amount - COALESCE(bs.spent, 0))
                        / GREATEST(1, (sb.end_date::date - CURRENT_DATE + 1)),
                        2
                    )
                ELSE NULL
            END                                  AS daily_allowance,
            sb.currency                          AS currency,
            sb.period                            AS period,
            sb.start_date::date                  AS start_date,
            sb.end_date::date                    AS end_date,
            sb.mode                              AS mode
        FROM scoped_budgets sb
        LEFT JOIN budget_spent bs ON bs.budget_sync_id = sb.budget_sync_id
        ORDER BY sb.start_date DESC
    """
    return sql, params


def build_budget_transactions(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Drill-down: every transaction that the budget counts.

    Same filter logic as build_budget_remaining (wallet + category-name +
    mode + currency + budget date window) but emits the transaction rows
    instead of an aggregate. Picks ALL budgets whose name matches
    `budget_name_phrase` — when the user named the budget; otherwise picks
    every active-in-window budget. Honors `order_by` and `limit`.
    """
    params: list = [user_id, spec.time_range.start, spec.time_range.end]
    name_clause = ""
    if spec.budget_name_phrase:
        params.append(f"%{spec.budget_name_phrase}%")
        name_clause = f"AND b.name ILIKE ${len(params)}"

    order_clause = (
        "ORDER BY t.amount::numeric DESC, t.date DESC"
        if spec.order_by == "amount_desc"
        else "ORDER BY t.date DESC"
    )
    cap = max(1, min(spec.limit or 50, 100))

    sql = f"""
        WITH scoped_budgets AS (
            SELECT
                b.sync_id     AS budget_sync_id,
                b.name        AS budget_name,
                b.currency    AS currency,
                b.start_date  AS start_date,
                b.end_date    AS end_date,
                b.mode        AS mode
            FROM budgets b
            WHERE b.is_deleted = false
              AND b.user_id = $1
              AND b.start_date::date <= $3::date
              AND b.end_date::date   >= $2::date
              {name_clause}
        ),
        budget_filters AS (
            SELECT
                sb.budget_sync_id,
                array_remove(array_agg(DISTINCT bw.wallet_sync_id), NULL) AS wallet_ids,
                array_remove(
                    array_agg(DISTINCT c.name) FILTER (WHERE c.name IS NOT NULL),
                    NULL
                ) AS category_names
            FROM scoped_budgets sb
            LEFT JOIN budget_wallets bw
                   ON bw.budget_sync_id = sb.budget_sync_id AND bw.is_deleted = false
            LEFT JOIN budget_categories bc
                   ON bc.budget_sync_id = sb.budget_sync_id AND bc.is_deleted = false
            LEFT JOIN categories c
                   ON c.sync_id = bc.category_sync_id AND c.is_deleted = false
            GROUP BY sb.budget_sync_id
        )
        SELECT
            t.sync_id                          AS sync_id,
            t.date::date                       AS date,
            t.type                             AS type,
            t.amount::numeric                  AS amount,
            COALESCE(t.currency_code, 'THB')  AS currency,
            t.note                             AS note,
            t.category_name                    AS category_name,
            w.name                             AS wallet_name,
            sb.budget_name                     AS budget_name
        FROM scoped_budgets sb
        JOIN budget_filters bf ON bf.budget_sync_id = sb.budget_sync_id
        JOIN transactions t
              ON t.is_deleted = false
             AND t.status = 'confirmed'
             AND t.include_in_report = true
             AND (sb.mode <> 'expenseOnly' OR t.type = 'expense')
             AND t.date >= sb.start_date
             AND t.date <  (sb.end_date + INTERVAL '1 day')
             AND (cardinality(bf.wallet_ids)     = 0 OR t.wallet_sync_id = ANY(bf.wallet_ids))
             AND (cardinality(bf.category_names) = 0 OR t.category_name  = ANY(bf.category_names))
             AND COALESCE(t.currency_code, 'THB') = sb.currency
        JOIN general_wallets w ON w.sync_id::text = t.wallet_sync_id
        WHERE w.deleted_at IS NULL
        {order_clause}
        LIMIT {cap}
    """
    return sql, params


_BUILDERS = {
    "sum_income": build_sum_income,
    "sum_expense": build_sum_expense,
    "balance": build_balance,
    "list": build_list,
    "count": build_count,
    "sum_by_category": build_sum_by_category,
    "sum_by_wallet": build_sum_by_wallet,
    "budget_list": build_budget_list,
    "budget_remaining": build_budget_remaining,
    "budget_transactions": build_budget_transactions,
}


def build_query(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Pick a builder by metric — KeyError surfaces as a programming bug."""
    return _BUILDERS[spec.metric](spec, user_id)
