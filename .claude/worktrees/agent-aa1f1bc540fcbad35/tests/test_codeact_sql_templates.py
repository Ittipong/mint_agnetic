"""Unit tests for src.agent.tools.codeact.sql_templates.

Covers UT-SQL01..03 — the SQL templates take a QuerySpec and emit a
(sql, params) tuple that asyncpg accepts. No DB connection touched —
these are pure string-shape + parameter-binding assertions.

Templates are FROZEN per memory `project_codeact_dual_impl` — drift between
build_query and namespace wrappers is what produced the 1,234.56
hallucination, so any template change here must come with a UT update.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.agent.tools.codeact.schemas import (
    QuerySpec,
    ResolvedEntity,
    TimeRange,
)
from src.agent.tools.codeact.sql_templates import build_query


def _bare(metric: str, **kw) -> QuerySpec:
    """Construct a minimal QuerySpec for `metric` over a default May-2026 window."""
    return QuerySpec(
        metric=metric,
        time_range=TimeRange(
            start=kw.pop("start", date(2026, 5, 1)),
            end=kw.pop("end", date(2026, 5, 31)),
        ),
        **kw,
    )


# ── UT-SQL01 — sum_expense shape & params ──────────────────────────────────


def test_UT_SQL01_sum_expense_emits_expected_shape():
    """UT-SQL01: build_query for sum_expense binds $1 user_id, $2 start,
    $3 end, and the literal 'expense' as the type filter. Cast through
    `t.amount::numeric` so aggregates land as Decimal (memory
    `project_chat_money_column_float8`)."""
    spec = _bare("sum_expense")
    sql, params = build_query(spec, user_id="u-1")

    assert params[0] == "u-1"
    assert params[1] == date(2026, 5, 1)
    assert params[2] == date(2026, 5, 31)
    # The fourth param locks the metric to expense (sum_income would bind 'income').
    assert "expense" in params

    # The SQL must restrict to confirmed, undeleted, in-report rows.
    assert "t.is_deleted = false" in sql
    assert "t.status = 'confirmed'" in sql
    assert "t.include_in_report = true" in sql
    # Numeric cast for Decimal-safe aggregates.
    assert "t.amount::numeric" in sql


def test_UT_SQL01b_sum_expense_convert_to_thb_drops_per_currency_group():
    """When convert_to_thb=True the GROUP BY currency disappears and the
    output is one THB-converted row — guards the THB-fast-path the LLM
    relies on for single-number totals."""
    spec = _bare("sum_expense", convert_to_thb=True)
    sql, _ = build_query(spec, user_id="u-1")
    assert "'THB'" in sql
    # convert path uses the hybrid FX expression
    assert "currency_code" in sql
    assert "GROUP BY" not in sql.split("FROM")[1] or "GROUP BY COALESCE" not in sql


# ── UT-SQL02 — time-range filter is applied ────────────────────────────────


def test_UT_SQL02_time_range_applied_inclusively():
    """UT-SQL02: the [start, end] window MUST appear as
    `t.date >= $2 AND t.date < ($3::date + INTERVAL '1 day')` — half-open
    on the upper bound so a transaction on `end` is included regardless of
    its time-of-day component (DB stores timestamps for some fixtures)."""
    spec = _bare("sum_expense",
                 start=date(2026, 4, 1), end=date(2026, 4, 30))
    sql, params = build_query(spec, user_id="u-1")

    assert params[1] == date(2026, 4, 1)
    assert params[2] == date(2026, 4, 30)
    assert "t.date >= $2" in sql
    # Half-open upper bound — guards "transaction at end-of-day end" inclusion.
    assert "($3::date + INTERVAL '1 day')" in sql


def test_UT_SQL02b_wallet_filter_appears_only_when_wallets_supplied():
    """Wallets passed in spec emit a `t.wallet_sync_id = ANY(...)` clause
    matched against EITHER side of a transfer (source OR destination) so
    incoming transfers aren't silently dropped from the receiving wallet."""
    spec_no_wallets = _bare("sum_expense")
    sql_no, _ = build_query(spec_no_wallets, user_id="u-1")
    assert "wallet_sync_id = ANY" not in sql_no

    spec_with = _bare(
        "sum_expense",
        wallets=[ResolvedEntity(
            sync_id="w-1", display_name="W1", kind="wallet", score=1.0,
        )],
    )
    sql_with, params = build_query(spec_with, user_id="u-1")
    assert "t.wallet_sync_id = ANY" in sql_with
    assert "t.destination_wallet_sync_id = ANY" in sql_with
    assert ["w-1"] in params  # bound as a list for ::text[] cast


# ── UT-SQL03 — note search is parameterized (no injection) ──────────────────


def test_UT_SQL03_note_query_is_parameterized():
    """UT-SQL03: note keywords are passed as $N parameters wrapped in `%kw%`,
    never spliced into the SQL string — so user-supplied keywords cannot
    inject SQL (e.g. `'; DROP TABLE`). The clause uses ILIKE for case-
    insensitive Thai/English matches."""
    spec = _bare(
        "list",
        note_query=["'; DROP TABLE transactions; --"],
    )
    sql, params = build_query(spec, user_id="u-1")

    # The injection payload must NOT appear in the SQL string.
    assert "DROP TABLE transactions" not in sql
    # It MUST appear in params, wrapped for ILIKE.
    assert "%'; DROP TABLE transactions; --%" in params
    # The placeholder uses ILIKE, not raw concat.
    assert "ILIKE" in sql


def test_UT_SQL03b_metric_dispatch_unknown_raises_keyerror():
    """An unknown metric name surfaces KeyError — that's a programming bug
    (the QuerySpec.metric Literal SHOULD have caught it earlier), not a
    user error."""
    # Bypass Pydantic validation to force an unknown metric into the dispatch.
    spec = _bare("sum_expense")
    spec_dict = spec.model_dump()
    spec_dict["metric"] = "totally_made_up"
    # Reconstruct outside of QuerySpec to avoid the Literal check.
    class _FakeSpec:
        def __init__(self, d):
            self.__dict__.update(d)
        def __getattr__(self, k):
            return self.__dict__.get(k)

    fake = _FakeSpec({"metric": "totally_made_up"})
    with pytest.raises(KeyError):
        build_query(fake, user_id="u-1")  # type: ignore[arg-type]


# ── UT-SQL-CB — sum_by_category pulls the category icon for the card ────────


def test_UT_SQL_CB01_sum_by_category_selects_icon_and_joins_categories():
    """UT-SQL-CB01: build_query for sum_by_category LEFT JOINs `categories cat`
    and selects its `icon` so the mobile category_breakdown card can render the
    real per-category glyph. `icon` aggregates via array_agg (NOT MAX) because
    the column is JSONB and Postgres has no max(jsonb)."""
    spec = _bare("sum_by_category")
    sql, _ = build_query(spec, user_id="u-1")
    assert "LEFT JOIN categories cat ON cat.sync_id = t.category_sync_id" in sql
    assert "icon" in sql
    # Guard against the MAX(jsonb) regression that would raise at runtime.
    assert "MAX(cat.icon)" not in sql
    assert "array_agg(cat.icon" in sql


def test_UT_SQL_CB02_sum_by_category_thb_branch_also_has_icon():
    """UT-SQL-CB02: the convert_to_thb branch keeps the icon select + join too,
    using alias `cat` so it never collides with the `c` currencies alias."""
    spec = _bare("sum_by_category", convert_to_thb=True)
    sql, _ = build_query(spec, user_id="u-1")
    assert "LEFT JOIN categories cat ON cat.sync_id = t.category_sync_id" in sql
    assert "array_agg(cat.icon" in sql
    # currencies join still uses alias `c` — both coexist.
    assert "LEFT JOIN currencies c" in sql
