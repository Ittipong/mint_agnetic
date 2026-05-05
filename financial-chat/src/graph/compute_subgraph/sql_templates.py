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

from src.graph.compute_subgraph.schemas import QuerySpec


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


# ── Hybrid FX expression ─────────────────────────────────────────────────────


# Best-effort transaction → THB amount. Priority:
#   1. native THB → use raw amount.
#   2. user-recorded converted_amount when it differs from raw amount
#      (rules out the case where the input pipeline left it equal to amount).
#   3. per-tx exchange_rate.
#   4. today's rate from the currencies table.
#   5. final fallback: raw amount (treats as THB; should not normally hit).
#
# Uses table aliases `t` (transactions) and `c` (currencies) — every caller
# that uses this expression must JOIN currencies as `c` on the tx currency.
_AMOUNT_THB_EXPR = """COALESCE(
    CASE
        WHEN COALESCE(t.currency_code, 'THB') = 'THB' THEN t.amount::numeric
        WHEN t.converted_amount IS NOT NULL
             AND t.converted_amount::numeric <> t.amount::numeric
            THEN t.converted_amount::numeric
        ELSE NULL
    END,
    CASE
        WHEN t.exchange_rate IS NOT NULL AND t.exchange_rate > 0
            THEN t.amount::numeric / t.exchange_rate::numeric
        ELSE NULL
    END,
    t.amount::numeric / NULLIF(c.rate, 0),
    t.amount::numeric
)"""


def _currency_join() -> str:
    """JOIN clause for the currencies lookup, only needed when converting."""
    return (
        "LEFT JOIN currencies c "
        "ON c.code = COALESCE(t.currency_code, 'THB')"
    )


def _wallet_join() -> str:
    """LEFT JOIN all 3 wallet types so transactions from any wallet are visible."""
    return """
        LEFT JOIN general_wallets gw ON gw.sync_id::text = t.wallet_sync_id
        LEFT JOIN creditcard_wallets cc ON cc.sync_id::text = t.wallet_sync_id
        LEFT JOIN goal_wallets gl ON gl.sync_id::text = t.wallet_sync_id"""


def _wallet_name_expr() -> str:
    """COALESCE wallet name from all 3 wallet types."""
    return "COALESCE(gw.name, cc.name, gl.name) AS wallet_name"


def _wallet_user_check() -> str:
    """WHERE clause to filter by user_id from any wallet type (non-deleted).
    Use as: WHERE 1=1 {_wallet_user_check()}"""
    return """
        AND (gw.user_id = $1 AND gw.deleted_at IS NULL
             OR cc.user_id = $1 AND cc.deleted_at IS NULL
             OR gl.user_id = $1 AND gl.deleted_at IS NULL)"""


# ── Metric builders ──────────────────────────────────────────────────────────


def _sum_by_type(spec: QuerySpec, user_id: str, type_value: str) -> tuple[str, list]:
    params = _base_params(spec, user_id)
    params.append(type_value)
    type_idx = len(params)
    extra = _common_filters(spec, params)
    if spec.convert_to_thb:
        sql = f"""
            SELECT
                'THB'                AS currency,
                SUM({_AMOUNT_THB_EXPR}) AS amount,
                COUNT(*)              AS cnt
            FROM transactions t
            {_wallet_join()}
            {_currency_join()}
            WHERE 1=1 {_wallet_user_check()}
              AND t.date >= $2
              AND t.date <  ($3::date + INTERVAL '1 day')
              AND t.type = ${type_idx}
              AND t.is_deleted = false
              AND t.status = 'confirmed'
              AND t.include_in_report = true
              {extra}
        """
    else:
        sql = f"""
            SELECT
                COALESCE(t.currency_code, 'THB') AS currency,
                SUM(t.amount::numeric)           AS amount,
                COUNT(*)                          AS cnt
            FROM transactions t
            {_wallet_join()}
            WHERE 1=1 {_wallet_user_check()}
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
        {_wallet_join()}
        WHERE 1=1 {_wallet_user_check()}
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
    ask for "top 5 by amount" or "latest 20 by date" without a new metric.
    `transaction_type` (income/expense/transfer/creditCardPay) narrows by
    `t.type` so 'ดูรายการรายรับ' returns income rows only."""
    params = _base_params(spec, user_id)
    extra = _common_filters(spec, params)
    type_clause = ""
    if spec.transaction_type is not None:
        params.append(spec.transaction_type)
        type_clause = f"AND t.type = ${len(params)}"
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
            {_wallet_name_expr()}
        FROM transactions t
        {_wallet_join()}
        WHERE 1=1 {_wallet_user_check()}
          AND t.date >= $2
          AND t.date <  ($3::date + INTERVAL '1 day')
          AND t.is_deleted = false
          AND t.status = 'confirmed'
          {type_clause}
          {extra}
        {order_clause}
        LIMIT {cap}
    """
    return sql, params


def build_balance(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Per-wallet balance as of `time_range.end`.

    initial_balance + sum of (effect_on_wallet * amount) for transactions
    on or before the end date. Uses effect_on_wallet so transfers behave correctly.

    When `convert_to_thb=True`: convert wallet's native balance to THB via
    the currencies table (initial_balance + per-tx amounts both run through
    the hybrid FX expression). Wallets stay as separate rows, but each row's
    `amount` is now in THB and the row's `currency` reads 'THB'.
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

    if spec.convert_to_thb:
        # initial_balance is in wallet.currency; wallet has no per-tx record
        # so we use the currencies table (cw) for the initial conversion.
        # Per-transaction amounts use the hybrid chain via _AMOUNT_THB_EXPR.
        sql = f"""
            SELECT
                w.sync_id::text                                              AS wallet_sync_id,
                w.name                                                        AS wallet_name,
                'THB'                                                         AS currency,
                (
                    (w.initial_balance / NULLIF(cw.rate, 0))
                    + COALESCE(SUM(
                        t.effect_on_wallet * {_AMOUNT_THB_EXPR}
                    ) FILTER (
                        WHERE t.is_deleted = false
                          AND t.status = 'confirmed'
                          AND t.date < ($2::date + INTERVAL '1 day')
                    ), 0)
                )                                                             AS amount
            FROM general_wallets w
            LEFT JOIN currencies cw ON cw.code = w.currency
            LEFT JOIN transactions t ON t.wallet_sync_id = w.sync_id::text
            LEFT JOIN currencies c ON c.code = COALESCE(t.currency_code, 'THB')
            WHERE w.user_id = $1
              AND w.deleted_at IS NULL
              {wallet_clause}
              {currency_clause}
            GROUP BY w.sync_id, w.name, w.initial_balance, cw.rate
            ORDER BY w.name
        """
    else:
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
    if spec.convert_to_thb:
        sql = f"""
            SELECT
                COALESCE(t.category_name, '(uncategorized)') AS bucket,
                'THB'                                          AS currency,
                SUM({_AMOUNT_THB_EXPR})                        AS amount,
                COUNT(*)                                       AS cnt
            FROM transactions t
            {_wallet_join()}
            {_currency_join()}
            WHERE 1=1 {_wallet_user_check()}
              AND t.date >= $2
              AND t.date <  ($3::date + INTERVAL '1 day')
              AND t.type = ${type_idx}
              AND t.is_deleted = false
              AND t.status = 'confirmed'
              AND t.include_in_report = true
              {extra}
            GROUP BY COALESCE(t.category_name, '(uncategorized)')
            ORDER BY amount DESC
            LIMIT 30
        """
    else:
        sql = f"""
            SELECT
                COALESCE(t.category_name, '(uncategorized)') AS bucket,
                COALESCE(t.currency_code, 'THB')             AS currency,
                SUM(t.amount::numeric)                        AS amount,
                COUNT(*)                                      AS cnt
            FROM transactions t
            {_wallet_join()}
            WHERE 1=1 {_wallet_user_check()}
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
    if spec.convert_to_thb:
        sql = f"""
            SELECT
                {_wallet_name_expr().replace(' AS wallet_name', ' AS bucket')} AS bucket,
                'THB'                    AS currency,
                SUM({_AMOUNT_THB_EXPR}) AS amount,
                COUNT(*)                 AS cnt
            FROM transactions t
            {_wallet_join()}
            {_currency_join()}
            WHERE 1=1 {_wallet_user_check()}
              AND t.date >= $2
              AND t.date <  ($3::date + INTERVAL '1 day')
              AND t.type = ${type_idx}
              AND t.is_deleted = false
              AND t.status = 'confirmed'
              AND t.include_in_report = true
              {extra}
            GROUP BY bucket
            ORDER BY amount DESC
        """
    else:
        sql = f"""
            SELECT
                {_wallet_name_expr().replace(' AS wallet_name', ' AS bucket')} AS bucket,
                COALESCE(t.currency_code, 'THB') AS currency,
                SUM(t.amount::numeric)            AS amount,
                COUNT(*)                          AS cnt
            FROM transactions t
            {_wallet_join()}
            WHERE 1=1 {_wallet_user_check()}
              AND t.date >= $2
              AND t.date <  ($3::date + INTERVAL '1 day')
              AND t.type = ${type_idx}
              AND t.is_deleted = false
              AND t.status = 'confirmed'
              AND t.include_in_report = true
              {extra}
            GROUP BY bucket, COALESCE(t.currency_code, 'THB')
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
            {_wallet_name_expr()},
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
        {_wallet_join()}
        WHERE (gw.sync_id IS NOT NULL OR cc.sync_id IS NOT NULL OR gl.sync_id IS NOT NULL)
          AND (gw.deleted_at IS NULL OR cc.deleted_at IS NULL OR gl.deleted_at IS NULL)
        {order_clause}
        LIMIT {cap}
    """
    return sql, params


# ── Credit card wallets ──────────────────────────────────────────────────────


def build_creditcard_list(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Every non-deleted credit-card wallet with limit, used, available.

    `used` is the backend-maintained `cached_used_amount`. When the cache is
    stale or has never been written, used / available come back as NULL on
    purpose — we never fall back to `initial_used` because doing so hides a
    stale cache from the user (a "0 used" reading on a card that has been
    swiped is far worse than an explicit "unknown").
    """
    params: list = [user_id]
    sql = """
        SELECT
            cc.sync_id::text                                AS sync_id,
            cc.name                                          AS name,
            cc.credit_limit::numeric                         AS credit_limit,
            cc.cached_used_amount::numeric                   AS used,
            CASE
                WHEN cc.cached_used_amount IS NULL THEN NULL
                ELSE cc.credit_limit::numeric - cc.cached_used_amount::numeric
            END                                              AS available,
            cc.currency                                      AS currency,
            cc.cache_updated_at                              AS cache_updated_at
        FROM creditcard_wallets cc
        WHERE cc.deleted_at IS NULL
          AND cc.user_id = $1
        ORDER BY cc.name
    """
    return sql, params


# ── Goal metrics ─────────────────────────────────────────────────────────────


def build_goal_list(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Every non-deleted savings goal — definition only, no compute."""
    params: list = [user_id]
    name_clause = ""
    if spec.goal_name_phrase:
        params.append(f"%{spec.goal_name_phrase}%")
        name_clause = f"AND g.name ILIKE ${len(params)}"
    sql = f"""
        SELECT
            g.sync_id::text         AS sync_id,
            g.name                   AS name,
            g.target_amount::numeric AS amount,
            g.currency               AS currency,
            g.target_date            AS target_date,
            g.start_date             AS start_date,
            g.is_closed              AS is_closed,
            g.achieved_at            AS achieved_at,
            g.note                   AS note
        FROM goal_wallets g
        WHERE g.is_deleted = false
          AND g.user_id = $1
          {name_clause}
        ORDER BY g.target_date NULLS LAST
    """
    return sql, params


def build_goal_progress(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Per-goal balance + completion stats.

    Goal balance is computed live from transactions whose
    `destination_wallet_sync_id` matches the goal's sync_id (deposits) less
    any whose `wallet_sync_id` matches (withdrawals from the goal). The
    cached `goal_wallets.cached_balance` is ignored — the test data shows it
    is often null/stale.

    Outputs: target, balance, remaining_to_target, pct_completed,
    days_left (clamped at 0), daily_required (NULL if expired or already
    achieved). currency stays in the goal's native currency — no FX.
    """
    params: list = [user_id]
    name_clause = ""
    if spec.goal_name_phrase:
        params.append(f"%{spec.goal_name_phrase}%")
        name_clause = f"AND g.name ILIKE ${len(params)}"

    sql = f"""
        WITH goals AS (
            SELECT
                g.sync_id            AS goal_sync_id,
                g.name               AS name,
                g.target_amount::numeric AS target,
                g.currency           AS currency,
                g.target_date        AS target_date,
                g.start_date         AS start_date,
                g.initial_balance::numeric AS initial_balance,
                g.is_closed          AS is_closed,
                g.achieved_at        AS achieved_at
            FROM goal_wallets g
            WHERE g.is_deleted = false
              AND g.user_id = $1
              {name_clause}
        ),
        bal AS (
            SELECT
                gs.goal_sync_id,
                COALESCE(SUM(
                    CASE
                        WHEN t.destination_wallet_sync_id = gs.goal_sync_id::text
                            THEN COALESCE(t.destination_converted_amount::numeric, t.amount::numeric)
                        WHEN t.wallet_sync_id = gs.goal_sync_id::text
                            THEN -t.amount::numeric
                        ELSE 0
                    END
                ) FILTER (
                    WHERE t.is_deleted = false AND t.status = 'confirmed'
                ), 0) AS deposited
            FROM goals gs
            LEFT JOIN transactions t
                   ON (t.destination_wallet_sync_id = gs.goal_sync_id::text
                    OR t.wallet_sync_id = gs.goal_sync_id::text)
            GROUP BY gs.goal_sync_id
        )
        SELECT
            gs.goal_sync_id::text                        AS sync_id,
            gs.name                                       AS name,
            gs.target                                     AS amount,
            (gs.initial_balance + COALESCE(bal.deposited, 0)) AS balance,
            (gs.target - (gs.initial_balance + COALESCE(bal.deposited, 0))) AS remaining_to_target,
            CASE WHEN gs.target > 0
                 THEN ROUND(
                    ((gs.initial_balance + COALESCE(bal.deposited, 0)) / gs.target) * 100,
                    1
                 )
                 ELSE 0 END                                AS pct_completed,
            CASE WHEN gs.target_date IS NULL THEN NULL
                 ELSE GREATEST(0, gs.target_date - CURRENT_DATE)
            END                                            AS days_left,
            CASE
                WHEN gs.is_closed THEN NULL
                WHEN gs.target_date IS NULL THEN NULL
                WHEN gs.target_date < CURRENT_DATE THEN NULL
                WHEN gs.target <= (gs.initial_balance + COALESCE(bal.deposited, 0)) THEN 0
                ELSE ROUND(
                    GREATEST(0, gs.target - (gs.initial_balance + COALESCE(bal.deposited, 0)))
                    / GREATEST(1, (gs.target_date - CURRENT_DATE)),
                    2
                )
            END                                            AS daily_required,
            gs.currency                                    AS currency,
            gs.target_date                                 AS target_date,
            gs.start_date                                  AS start_date,
            gs.is_closed                                   AS is_closed,
            gs.achieved_at                                 AS achieved_at
        FROM goals gs
        LEFT JOIN bal ON bal.goal_sync_id = gs.goal_sync_id
        ORDER BY gs.target_date NULLS LAST
    """
    return sql, params


def build_goal_transactions(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Drill-down: deposits/withdrawals on a goal wallet."""
    params: list = [user_id]
    name_clause = ""
    if spec.goal_name_phrase:
        params.append(f"%{spec.goal_name_phrase}%")
        name_clause = f"AND g.name ILIKE ${len(params)}"

    order_clause = (
        "ORDER BY t.amount::numeric DESC, t.date DESC"
        if spec.order_by == "amount_desc"
        else "ORDER BY t.date DESC"
    )
    cap = max(1, min(spec.limit or 50, 100))

    sql = f"""
        WITH goals AS (
            SELECT g.sync_id AS goal_sync_id, g.name AS goal_name
            FROM goal_wallets g
            WHERE g.is_deleted = false AND g.user_id = $1
              {name_clause}
        )
        SELECT
            t.sync_id                AS sync_id,
            t.date::date             AS date,
            t.type                   AS type,
            CASE
                WHEN t.destination_wallet_sync_id = gs.goal_sync_id::text
                    THEN 'deposit'
                ELSE 'withdraw'
            END                       AS direction,
            CASE
                WHEN t.destination_wallet_sync_id = gs.goal_sync_id::text
                    THEN COALESCE(t.destination_converted_amount::numeric, t.amount::numeric)
                ELSE t.amount::numeric
            END                       AS amount,
            COALESCE(t.currency_code, 'THB') AS currency,
            t.note                    AS note,
            gs.goal_name              AS goal_name
        FROM goals gs
        JOIN transactions t
              ON (t.destination_wallet_sync_id = gs.goal_sync_id::text
               OR t.wallet_sync_id = gs.goal_sync_id::text)
             AND t.is_deleted = false
             AND t.status = 'confirmed'
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
    "creditcard_list": build_creditcard_list,
    "goal_list": build_goal_list,
    "goal_progress": build_goal_progress,
    "goal_transactions": build_goal_transactions,
}


def build_query(spec: QuerySpec, user_id: str) -> tuple[str, list]:
    """Pick a builder by metric — KeyError surfaces as a programming bug."""
    return _BUILDERS[spec.metric](spec, user_id)
