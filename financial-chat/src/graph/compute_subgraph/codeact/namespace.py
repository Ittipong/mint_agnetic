"""Sandbox-safe wrappers around the SQL templates.

The wrappers:
  - take simple kwargs (str dates, str names) instead of QuerySpec
  - resolve wallet/category/tag NAMES against the user's EntityCatalog
  - execute through the shared asyncpg pool (sync facade for the sandbox)
  - return plain `list[dict]` so user code can compose without ORM types

Anything not in `build_namespace()` is unreachable from sandbox code.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from src.entity_catalog import EntityCatalog
from src.graph.compute_subgraph.db import get_pool
from src.graph.compute_subgraph.schemas import (
    QuerySpec,
    ResolvedEntity,
    TimeRange,
)
from src.graph.compute_subgraph.sql_templates import build_query


# ── Async core (runs on the main loop) ───────────────────────────────────────


async def _run_query(spec: QuerySpec, user_id: str) -> list[dict]:
    """Execute a QuerySpec and return rows as plain dicts.

    Decimal values stay as Decimal — the sandbox accepts them and arithmetic
    on Decimal is exact, so we keep them rather than stringifying.
    """
    sql, params = build_query(spec, user_id)
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(sql, *params)
    return [dict(r) for r in records]


# ── Sync facades (called from inside sandbox code) ───────────────────────────


class _Wrappers:
    """Holds per-turn context (user_id, catalog, today, main loop) so the
    namespace functions can stay simple kwargs-only callables.
    """

    def __init__(
        self,
        *,
        user_id: str,
        catalog: EntityCatalog,
        today: date,
        main_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._user_id = user_id
        self._catalog = catalog
        self._today = today
        self._loop = main_loop

    # -- Helpers ---------------------------------------------------------

    def _to_date(self, value: str | date) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    def _resolve_wallets(self, names: list[str] | None) -> list[ResolvedEntity]:
        if not names:
            return []
        out: list[ResolvedEntity] = []
        for name in names:
            match = next(
                (w for w in self._catalog.wallets if w.name == name), None
            )
            if match is None:
                raise ValueError(
                    f"unknown wallet name {name!r} — not in user's catalog"
                )
            out.append(
                ResolvedEntity(
                    sync_id=match.sync_id,
                    display_name=match.name,
                    kind="wallet",
                    score=1.0,
                )
            )
        return out

    def _resolve_categories(self, names: list[str] | None) -> list[ResolvedEntity]:
        # Categories filter by name (denormalized) — sync_id is informational.
        if not names:
            return []
        return [
            ResolvedEntity(
                sync_id="",  # not used: SQL filters on category_name
                display_name=name,
                kind="category",
                score=1.0,
            )
            for name in names
        ]

    def _resolve_tags(self, names: list[str] | None) -> list[ResolvedEntity]:
        if not names:
            return []
        out: list[ResolvedEntity] = []
        for name in names:
            match = next(
                (t for t in self._catalog.tags if t.name == name), None
            )
            if match is None:
                raise ValueError(
                    f"unknown tag name {name!r} — not in user's catalog"
                )
            out.append(
                ResolvedEntity(
                    sync_id=match.sync_id,
                    display_name=match.name,
                    kind="tag",
                    score=1.0,
                )
            )
        return out

    def _spec(
        self,
        *,
        metric: str,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        order_by: str = "date_desc",
        limit: int | None = None,
        budget_name_phrase: str | None = None,
        goal_name_phrase: str | None = None,
        convert_to_thb: bool = False,
        transaction_type: str | None = None,
    ) -> QuerySpec:
        return QuerySpec(
            metric=metric,  # type: ignore[arg-type]
            wallets=self._resolve_wallets(wallet_names),
            categories=self._resolve_categories(category_names),
            tags=self._resolve_tags(tag_names),
            time_range=TimeRange(
                start=self._to_date(start),
                end=self._to_date(end),
                granularity="day",
                confidence=1.0,
            ),
            currency=currency,  # type: ignore[arg-type]
            order_by=order_by,  # type: ignore[arg-type]
            limit=limit,
            budget_name_phrase=budget_name_phrase,
            goal_name_phrase=goal_name_phrase,
            convert_to_thb=convert_to_thb,
            transaction_type=transaction_type,  # type: ignore[arg-type]
        )

    def _exec(self, spec: QuerySpec) -> list[dict]:
        """Bridge sync sandbox → async DB. Only safe to call from a worker
        thread (the main event loop must be running)."""
        coro = _run_query(spec, self._user_id)
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=15)

    # -- Public template wrappers (these are what the LLM calls) --------

    def sum_income(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        convert_to_thb: bool = False,
    ) -> list[dict]:
        """Total income per currency in [start, end].

        With convert_to_thb=True the per-currency split is dropped and you
        get ONE row in THB via the hybrid FX chain. Use this whenever you
        need a single THB total — never sum across currency rows yourself.
        """
        return self._exec(
            self._spec(
                metric="sum_income",
                start=start,
                end=end,
                wallet_names=wallet_names,
                category_names=category_names,
                tag_names=tag_names,
                currency=currency,
                convert_to_thb=convert_to_thb,
            )
        )

    def sum_expense(self, **kw) -> list[dict]:
        """Total expense per currency. Same kwargs as sum_income, including
        convert_to_thb=True for a single THB total."""
        kw["metric"] = "sum_expense"
        return self._exec(self._spec(**kw))

    def sum_by_category(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        currency: str = "ALL",
        convert_to_thb: bool = False,
    ) -> list[dict]:
        """Spending breakdown by category (expense only)."""
        return self._exec(
            self._spec(
                metric="sum_by_category",
                start=start,
                end=end,
                wallet_names=wallet_names,
                currency=currency,
                convert_to_thb=convert_to_thb,
            )
        )

    def sum_by_wallet(
        self,
        *,
        start: str | date,
        end: str | date,
        currency: str = "ALL",
        convert_to_thb: bool = False,
    ) -> list[dict]:
        """Spending breakdown by wallet (expense only)."""
        return self._exec(
            self._spec(
                metric="sum_by_wallet",
                start=start,
                end=end,
                currency=currency,
                convert_to_thb=convert_to_thb,
            )
        )

    def list_transactions(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        order_by: str = "date_desc",
        limit: int | None = 50,
        transaction_type: str | None = None,
    ) -> list[dict]:
        """Individual transactions matching filters.

        `transaction_type` narrows to one of: 'income', 'expense', 'transfer',
        'creditCardPay'. Leave None for all types."""
        return self._exec(
            self._spec(
                metric="list",
                start=start,
                end=end,
                wallet_names=wallet_names,
                category_names=category_names,
                tag_names=tag_names,
                currency=currency,
                order_by=order_by,
                limit=limit,
                transaction_type=transaction_type,
            )
        )

    def balance(
        self,
        *,
        as_of: str | date | None = None,
        wallet_names: list[str] | None = None,
        currency: str = "ALL",
        convert_to_thb: bool = False,
    ) -> list[dict]:
        """Per-wallet balance as of `as_of` (defaults to today). With
        convert_to_thb=True every wallet's balance is rendered in THB."""
        end = self._to_date(as_of) if as_of is not None else self._today
        return self._exec(
            self._spec(
                metric="balance",
                start=date(1900, 1, 1),
                end=end,
                wallet_names=wallet_names,
                currency=currency,
                convert_to_thb=convert_to_thb,
            )
        )

    def budget_remaining(
        self,
        *,
        start: str | date | None = None,
        end: str | date | None = None,
        budget_name_phrase: str | None = None,
    ) -> list[dict]:
        """Active budgets in [start, end] with amount/spent/remaining/pct/days."""
        return self._exec(
            self._spec(
                metric="budget_remaining",
                start=self._to_date(start) if start else date(1900, 1, 1),
                end=self._to_date(end) if end else self._today,
                budget_name_phrase=budget_name_phrase,
            )
        )

    def budget_transactions(
        self,
        *,
        budget_name_phrase: str,
        order_by: str = "amount_desc",
        limit: int = 20,
    ) -> list[dict]:
        """Transactions counted toward a specific budget (drill-down)."""
        return self._exec(
            self._spec(
                metric="budget_transactions",
                start=date(1900, 1, 1),
                end=self._today,
                budget_name_phrase=budget_name_phrase,
                order_by=order_by,
                limit=limit,
            )
        )

    def budget_list(self) -> list[dict]:
        """All non-deleted budgets the user has ever set up."""
        return self._exec(
            self._spec(
                metric="budget_list",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )

    def goal_list(self) -> list[dict]:
        """All savings goals the user has set up (target, target_date, currency)."""
        return self._exec(
            self._spec(
                metric="goal_list",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )

    def goal_progress(
        self,
        *,
        goal_name_phrase: str | None = None,
    ) -> list[dict]:
        """Per-goal progress: target, current balance, remaining, pct_completed,
        days_left, daily_required (NULL when expired/achieved)."""
        return self._exec(
            self._spec(
                metric="goal_progress",
                start=date(1900, 1, 1),
                end=self._today,
                goal_name_phrase=goal_name_phrase,
            )
        )

    def goal_transactions(
        self,
        *,
        goal_name_phrase: str,
        order_by: str = "date_desc",
        limit: int = 50,
    ) -> list[dict]:
        """Deposits/withdrawals to/from a specific goal wallet."""
        return self._exec(
            self._spec(
                metric="goal_transactions",
                start=date(1900, 1, 1),
                end=self._today,
                goal_name_phrase=goal_name_phrase,
                order_by=order_by,
                limit=limit,
            )
        )


# ── Public entry point ───────────────────────────────────────────────────────


def build_namespace(
    *,
    user_id: str,
    catalog: EntityCatalog,
    today: date,
    main_loop: asyncio.AbstractEventLoop,
) -> dict[str, Any]:
    """Build the dict that the sandbox `exec()` runs against.

    Anything not in this dict is invisible to user code — the AST validator
    additionally blocks `import`, attribute access on dunders, etc.
    """
    w = _Wrappers(user_id=user_id, catalog=catalog, today=today, main_loop=main_loop)
    return {
        # Templates (the only DB access)
        "sum_income":          w.sum_income,
        "sum_expense":         w.sum_expense,
        "sum_by_category":     w.sum_by_category,
        "sum_by_wallet":       w.sum_by_wallet,
        "list_transactions":   w.list_transactions,
        "balance":             w.balance,
        "budget_remaining":    w.budget_remaining,
        "budget_transactions": w.budget_transactions,
        "budget_list":         w.budget_list,
        "goal_list":           w.goal_list,
        "goal_progress":       w.goal_progress,
        "goal_transactions":   w.goal_transactions,
        # Decimal-safe primitives
        "Decimal":             Decimal,
        "date":                date,
        "timedelta":           timedelta,
        "today":               lambda: today,
        # Output marker — the LLM sets this to terminate the loop.
        "result":              None,
    }
