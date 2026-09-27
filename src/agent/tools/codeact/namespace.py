"""Sandbox-safe wrappers around the SQL templates.

Ported from v2 codeact_subgraph in Wave 2 — namespace API is FROZEN to
prevent the 1,234.56 hallucination regression (per memory
`project_codeact_dual_impl`). Only import adjustments are allowed.

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
import calendar
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from src.agent.entity_catalog import EntityCatalog
from .resolvers import (
    clarify,
    make_resolve_budget,
    make_resolve_category,
    make_resolve_goal,
    make_resolve_tag,
    make_resolve_wallet,
    parse_period as _parse_period,
)
from .calculators import (
    affordability_check,
    compute_dti,
    debt_payoff_months,
    emergency_fund_target,
    mortgage_payment,
    refi_payback_months,
)
from .db import get_pool
from .schemas import (
    QuerySpec,
    ResolvedEntity,
    TimeRange,
)
from .sql_templates import build_query


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
        from src.agent.tools.codeact.resolvers import is_root_dominant

        pool = self._catalog.categories_all or self._catalog.categories
        return [
            ResolvedEntity(
                sync_id="",  # not used: SQL filters on names
                display_name=name,
                kind="category",
                score=1.0,
                group_match=is_root_dominant(name, pool),
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
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
        granularity: str = "day",
    ) -> QuerySpec:
        # Allow `note_query="Starbucks"` for the common single-keyword case.
        if isinstance(note_query, str):
            note_query = [note_query]
        return QuerySpec(
            metric=metric,  # type: ignore[arg-type]
            wallets=self._resolve_wallets(wallet_names),
            categories=self._resolve_categories(category_names),
            tags=self._resolve_tags(tag_names),
            time_range=TimeRange(
                start=self._to_date(start),
                end=self._to_date(end),
                granularity=granularity,  # type: ignore[arg-type]
                confidence=1.0,
            ),
            currency=currency,  # type: ignore[arg-type]
            order_by=order_by,  # type: ignore[arg-type]
            limit=limit,
            budget_name_phrase=budget_name_phrase,
            goal_name_phrase=goal_name_phrase,
            convert_to_thb=convert_to_thb,
            transaction_type=transaction_type,  # type: ignore[arg-type]
            note_query=note_query,
            match_destination_note=match_destination_note,
            has_note=has_note,
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
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
        # Accepted-and-ignored: the metric name already fixes the type, but the
        # LLM frequently over-generalises `transaction_type` (most other tools
        # take it) onto the typed sum_* wrappers. Swallowing it here keeps a
        # harmless redundant kwarg from raising TypeError and burning a whole
        # ReAct round-trip on the retry. We do NOT use **kwargs — a real param
        # typo must still surface, not be silently dropped.
        transaction_type: str | None = None,
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
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
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
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
        transaction_type: str | None = None,  # accepted-and-ignored (see sum_income)
        by_parent: bool = False,
    ) -> list[dict]:
        """Spending breakdown by category (expense only). `by_parent=True`
        folds sub-categories into their parent (the app's "แยกตามหมวด")."""
        spec = self._spec(
            metric="sum_by_category",
            start=start,
            end=end,
            wallet_names=wallet_names,
            currency=currency,
            convert_to_thb=convert_to_thb,
            note_query=note_query,
            match_destination_note=match_destination_note,
            has_note=has_note,
        )
        spec.category_level = "parent" if by_parent else "leaf"
        return self._exec(spec)

    def fixed_costs(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
    ) -> dict:
        """Fixed monthly costs (rent, bills, subscriptions, insurance,
        installments) in THB: {total, months, monthly_avg, rows}. Each row:
        {bucket (leaf category), parent, category_sync_id, amount, cnt}.
        `months` = the window in months (a near-whole count is rounded: a
        calendar month = 1, a rolling "3 เดือน" = 3). Use this — never classify
        fixed vs variable yourself."""
        rows = self._exec(self._spec(
            metric="fixed_costs", start=start, end=end, wallet_names=wallet_names,
        ))
        s, e = self._to_date(start), self._to_date(end)
        months = Decimal(max((e - s).days + 1, 1)) / Decimal("30.4375")
        # A calendar month (30 d → 0.99) or "3 เดือน" (92 d → 3.02) is a whole
        # number of months to a person — average by that, not by days.
        if abs(months - round(months)) <= Decimal("0.1") and round(months) >= 1:
            months = Decimal(round(months))
        total = sum((Decimal(str(r["amount"] or 0)) for r in rows), Decimal(0))
        return {
            "total": total,
            "months": round(months, 2),
            "monthly_avg": round(total / months, 2) if rows else Decimal(0),
            "rows": rows,
        }

    def month_end_outlook(self, *, as_of: str | date | None = None) -> dict:
        """"เงินจะพอใช้ถึงสิ้นเดือนไหม" — spendable cash left at month end.

        month_end_cash = cash_now − typical_daily × days_left − fixed_pending
                         − card_due_before_month_end

        - cash_now: general wallets only (goals are savings, cards are debt).
          Money already spent this month is ALREADY out of cash_now — never
          subtract spent-so-far or a projected month total from it (the chat
          once did: "เหลือ 107,309" instead of ~152,000).
        - typical_daily: variable spend (expense − fixed costs) over the last
          90 days ÷ 90, so a one-off (an iPad) is diluted, not extrapolated.
        - fixed_pending: this month's usual fixed costs not yet recorded.
        - card_due_before_month_end: statement amounts due by month end;
          month_end_cash_after_all_card_dues also pays every other statement.
        No future income is added (conservative). Returns all components.
        """
        from datetime import timedelta as _td

        today = self._to_date(as_of) if as_of else self._today
        nxt = (today.replace(day=28) + _td(days=4)).replace(day=1)
        month_end = nxt - _td(days=1)
        days_left = (month_end - today).days

        # balance() reads general wallets only — spendable cash.
        cash_now = sum((Decimal(str(r.get("amount") or 0))
                        for r in self.balance(convert_to_thb=True)), Decimal(0))

        start90 = today - _td(days=89)
        spent90 = sum((Decimal(str(r.get("amount") or 0)) for r in
                       self.sum_expense(start=start90, end=today, convert_to_thb=True)),
                      Decimal(0))
        fixed90 = self.fixed_costs(start=start90, end=today)
        typical_daily = (spent90 - fixed90["total"]) / Decimal(90)

        fixed_month = self.fixed_costs(start=today.replace(day=1), end=today)
        fixed_pending = max(fixed90["monthly_avg"] - fixed_month["total"], Decimal(0))

        card_due = Decimal(0)
        all_card_due = Decimal(0)
        cards_due = []
        for c in self.creditcard_list():
            due, when = c.get("amount_due"), c.get("next_due_date")
            if not due:
                continue
            all_card_due += Decimal(str(due))
            cards_due.append({"name": c["name"], "amount_due": due, "due": when})
            if when and when <= month_end:
                card_due += Decimal(str(due))

        variable_left = typical_daily * days_left
        month_end_cash = cash_now - variable_left - fixed_pending - card_due
        q = Decimal("1")
        return {
            "as_of": today, "month_end": month_end, "days_left": days_left,
            "cash_now": cash_now.quantize(q),
            "typical_daily_spend": typical_daily.quantize(q),
            "variable_spend_left": variable_left.quantize(q),
            "fixed_pending": fixed_pending.quantize(q),
            "card_due_before_month_end": card_due.quantize(q),
            "cards_due": cards_due,
            "month_end_cash": month_end_cash.quantize(q),
            "enough": month_end_cash > 0,
            # "ถ้าจ่ายบัตรเต็ม เหลือพอไหม": every statement paid, whatever its
            # due date (the chat once answered without subtracting them).
            "all_card_due": all_card_due.quantize(q),
            "month_end_cash_after_all_card_dues":
                (month_end_cash - (all_card_due - card_due)).quantize(q),
        }

    def spending_by_weekday(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
    ) -> list[dict]:
        """Expense per day of week, THB, biggest first. Each row: {weekday
        (1=Mon … 7=Sun), weekday_th, amount, cnt, days_with_spend}. For habit
        questions ("ใช้เงินหนักวันไหน") — never sum list_transactions."""
        names = {1: "วันจันทร์", 2: "วันอังคาร", 3: "วันพุธ", 4: "วันพฤหัสบดี",
                 5: "วันศุกร์", 6: "วันเสาร์", 7: "วันอาทิตย์"}
        rows = self._exec(self._spec(
            metric="spending_by_weekday", start=start, end=end, wallet_names=wallet_names,
        ))
        for r in rows:
            r["weekday_th"] = names.get(int(r["weekday"]), str(r["weekday"]))
        return rows

    def sum_by_wallet(
        self,
        *,
        start: str | date,
        end: str | date,
        currency: str = "ALL",
        convert_to_thb: bool = False,
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
        transaction_type: str | None = None,  # accepted-and-ignored (see sum_income)
    ) -> list[dict]:
        """Spending breakdown by wallet (expense only)."""
        return self._exec(
            self._spec(
                metric="sum_by_wallet",
                start=start,
                end=end,
                currency=currency,
                convert_to_thb=convert_to_thb,
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
            )
        )

    def sum_by_tag(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        convert_to_thb: bool = False,
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
        transaction_type: str | None = None,  # accepted-and-ignored (see sum_income)
    ) -> list[dict]:
        """Spending breakdown by tag (expense only). Untagged transactions are
        excluded — pass tag_names=[...] to restrict to a subset of tags.

        Returns rows like sum_by_category: {bucket, currency, amount, cnt}
        where bucket = tag name (verbatim from the catalog, with or without `#`).
        """
        return self._exec(
            self._spec(
                metric="sum_by_tag",
                start=start,
                end=end,
                wallet_names=wallet_names,
                tag_names=tag_names,
                currency=currency,
                convert_to_thb=convert_to_thb,
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
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
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
    ) -> list[dict]:
        """Individual transactions matching filters.

        `transaction_type` narrows to one of: 'income', 'expense', 'transfer',
        'creditCardPay'. Leave None for all types.

        Note search:
          - `note_query` = single keyword or a list. Case-insensitive LIKE
            OR across the keywords. With `match_destination_note=True` (default)
            also searches `destination_note` so transfers are caught.
          - `has_note=True/False` filters by note presence.
        """
        rows = self._exec(
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
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
            )
        )
        from src.agent.tools.codeact.sql_templates import LIST_ROW_CAP

        # The SQL caps every list at LIST_ROW_CAP whatever `limit` asked for,
        # so compare against the EFFECTIVE cap (limit=500 once returned 100
        # rows silently and a weekday total came out a third short).
        cap = min(limit or 50, LIST_ROW_CAP)
        if len(rows) >= cap:
            # stdout reaches the LLM. Summing a capped list once turned KTC's
            # 15,524 into 14,574 and a fixed-cost pass lost the rent.
            print(
                f"[list_transactions] TRUNCATED at {cap} rows: older rows are "
                "missing. Do NOT sum/count/average these rows — use sum_expense, "
                "sum_by_category, sum_by_wallet or count_transactions for totals."
            )
        return rows

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
        """Budgets covering the [start, end] window with amount/spent/remaining/pct/days.

        Default (no args): **active-only** — returns only budgets whose period
        covers TODAY (start_date <= today <= end_date). Showing already-ended
        budgets in an overview misleads the user, so the LLM must explicitly
        ask for them by passing a `start`/`end` that overlaps the older period
        (e.g. last month).
        """
        resolved_start = self._to_date(start) if start else self._today
        resolved_end = self._to_date(end) if end else self._today
        return self._exec(
            self._spec(
                metric="budget_remaining",
                start=resolved_start,
                end=resolved_end,
                budget_name_phrase=budget_name_phrase,
            )
        )

    def creditcard_list(self) -> list[dict]:
        """Every non-deleted credit-card wallet — limit, used, available, currency.

        Each row: {sync_id, name, credit_limit, used, available, currency,
        billing_cycle_day, payment_due_day, last_statement_date, next_due_date,
        statement_balance, amount_due, unbilled, cache_updated_at}. `amount_due`
        = the closed statement minus payments since — the figure to pay by
        next_due_date (None when the card has no billing day). `unbilled` = the
        swipes since that statement, which bill next cycle (`used` − the
        statement still owed). `used` mirrors the backend's cached_used_amount
        and is the TOTAL owed: amount_due + unbilled.
        When the cache has never been written it stays NULL (and `available`
        is NULL too) — never substitute a 0 or initial_used here, an unknown
        usage must read as unknown so the user can see the cache is stale.
        The two dates are computed from the injected `today` (not SQL now())
        so time-warped QA replays stay deterministic.
        """
        rows = self._exec(
            self._spec(
                metric="creditcard_list",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )
        for r in rows:
            r["last_statement_date"] = _day_of_month_on_or_before(
                self._today, r.get("billing_cycle_day"))
            r["next_due_date"] = _day_of_month_on_or_after(
                self._today, r.get("payment_due_day"))
            # "ต้องจ่ายเท่าไหร่" = what the closed statement billed minus what was
            # paid since, NOT the live `used` (which also holds the open cycle's
            # swipes that bill next month).
            r["statement_balance"] = r["amount_due"] = r["unbilled"] = None
            if r["last_statement_date"] is not None:
                spec = self._spec(metric="creditcard_statement",
                                  start=date(1900, 1, 1), end=r["last_statement_date"])
                spec.wallets = [ResolvedEntity(sync_id=r["sync_id"], display_name=r["name"],
                                               kind="wallet", score=1.0)]
                st = self._exec(spec)
                if st:
                    bal = Decimal(str(st[0]["statement_balance"] or 0))
                    paid = Decimal(str(st[0]["paid_since_statement"] or 0))
                    r["statement_balance"] = bal
                    r["amount_due"] = max(bal - paid, Decimal(0))
                    # Unclamped owed-on-statement, so an overpayment lowers the
                    # next cycle instead of vanishing. The LLM once read the
                    # total `used` (24,943) as "next cycle" on top of 24,111 due.
                    if r.get("used") is not None:
                        r["unbilled"] = Decimal(str(r["used"])) - (bal - paid)
        return rows

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

    # ── Counts / discovery / analytics ─────────────────────────────────────

    def count_transactions(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        transaction_type: str | None = None,
        note_query: str | list[str] | None = None,
        match_destination_note: bool = True,
        has_note: bool | None = None,
    ) -> list[dict]:
        """Number of transactions matching filters, per currency. Use this for
        'กี่ครั้ง / how many' questions instead of len(list_transactions(...))."""
        return self._exec(
            self._spec(
                metric="count",
                start=start,
                end=end,
                wallet_names=wallet_names,
                category_names=category_names,
                tag_names=tag_names,
                currency=currency,
                transaction_type=transaction_type,
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
            )
        )

    def wallet_list(self) -> list[dict]:
        """Every non-deleted wallet the user owns (general / creditcard / goal).
        Returns rows with `kind` so the LLM picks the right downstream metric.
        Use this whenever the user asks 'บัญชีฉันมีอะไรบ้าง' before drilling in."""
        return self._exec(
            self._spec(
                metric="wallet_list",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )

    def category_list(self, *, transaction_type: str | None = None) -> list[dict]:
        """All non-deleted categories the user has. Pass transaction_type
        ('expense' / 'income' / 'transfer') to filter."""
        return self._exec(
            self._spec(
                metric="category_list",
                start=date(1900, 1, 1),
                end=self._today,
                transaction_type=transaction_type,
            )
        )

    def tag_list(self) -> list[dict]:
        """All non-deleted tags + how often each was used (all-time confirmed
        only). Sorted by usage_count DESC so the most-used tag is first."""
        return self._exec(
            self._spec(
                metric="tag_list",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )

    def spending_trend(
        self,
        *,
        start: str | date,
        end: str | date,
        group_by: str = "month",
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        transaction_type: str | None = "expense",
        convert_to_thb: bool = False,
        note_query: str | list[str] | None = None,
        has_note: bool | None = None,
    ) -> list[dict]:
        """Time-series totals grouped by `day` / `week` / `month` / `quarter`
        / `year`. Default = monthly expense. Use this for trend / 'เทรนด์' /
        'แต่ละเดือน' questions.

        Returns rows: {bucket (date), currency, amount, cnt}.
        With convert_to_thb=True you get one THB-converted row per bucket.
        """
        return self._exec(
            self._spec(
                metric="spending_trend",
                start=start,
                end=end,
                wallet_names=wallet_names,
                category_names=category_names,
                tag_names=tag_names,
                currency=currency,
                transaction_type=transaction_type,
                convert_to_thb=convert_to_thb,
                note_query=note_query,
                has_note=has_note,
                granularity=group_by,
            )
        )

    def transaction_stats(
        self,
        *,
        start: str | date,
        end: str | date,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        transaction_type: str | None = "expense",
        convert_to_thb: bool = False,
        note_query: str | list[str] | None = None,
        has_note: bool | None = None,
    ) -> list[dict]:
        """Min / max / avg / median / sum / count for the period.

        Use for 'เฉลี่ยใช้วันละเท่าไร' (avg), 'รายการแพงสุด' (max),
        'ถูกสุด' (min). Per-currency rows unless convert_to_thb=True.
        """
        return self._exec(
            self._spec(
                metric="transaction_stats",
                start=start,
                end=end,
                wallet_names=wallet_names,
                category_names=category_names,
                tag_names=tag_names,
                currency=currency,
                transaction_type=transaction_type,
                convert_to_thb=convert_to_thb,
                note_query=note_query,
                has_note=has_note,
            )
        )

    def top_transactions(
        self,
        *,
        start: str | date,
        end: str | date,
        limit: int = 5,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        currency: str = "ALL",
        transaction_type: str | None = "expense",
        note_query: str | list[str] | None = None,
        has_note: bool | None = None,
    ) -> list[dict]:
        """Top N transactions by absolute amount. Convenience wrapper around
        list_transactions(order_by='amount_desc') — saves the LLM from
        having to remember the sort flag."""
        return self.list_transactions(
            start=start,
            end=end,
            wallet_names=wallet_names,
            category_names=category_names,
            tag_names=tag_names,
            currency=currency,
            order_by="amount_desc",
            limit=limit,
            transaction_type=transaction_type,
            note_query=note_query,
            has_note=has_note,
        )

    def currency_rate(self, *, code: str | None = None) -> list[dict]:
        """FX rates from the currencies table. Pass `code='USD'` for one row
        or omit for every rated currency. Rate semantics: 1 unit of the code
        ≈ (1 / rate) THB. Use for 'X กี่บาท / convert N currency to THB'."""
        return self._exec(
            self._spec(
                metric="currency_rate",
                start=date(1900, 1, 1),
                end=self._today,
                currency=(code or "ALL"),
            )
        )

    def active_period(self) -> list[dict]:
        """User's first / last confirmed transaction date + active-day count.
        Use for 'ใช้แอปมานานเท่าไร / first transaction'."""
        return self._exec(
            self._spec(
                metric="active_period",
                start=date(1900, 1, 1),
                end=self._today,
            )
        )

    # ── Composite analytics (Python compose, not new SQL) ──────────────────

    def compare_periods(
        self,
        *,
        period1_start: str | date,
        period1_end: str | date,
        period2_start: str | date | None = None,
        period2_end: str | date | None = None,
        by: str = "category",
        transaction_type: str | None = "expense",
        currency: str = "ALL",
        convert_to_thb: bool = True,
    ) -> dict:
        """Compare TWO periods side-by-side. `by` ∈ {'category','subcategory',
        'wallet','tag','total'}. Returns {period1, period2, diff, pct_change}
        per bucket — one tool call instead of two + Python diff. 'category'
        groups like the app report (sub-categories fold into their parent);
        'subcategory' is the leaf level for drilling into one parent.

        This is a 2-period diff, NOT a multi-period trend. For "compare the
        last N months / see the trend over months", call
        `spending_trend(group_by='month')` instead.
        """
        # Defend the call surface: a missing period2 or a time-bucket `by`
        # (e.g. by="month") is the classic mis-call when the LLM conflates
        # "เทียบ N เดือน" (a trend) with a 2-period diff. Fail LOUD with an
        # actionable message the ReAct loop can act on (R10), instead of
        # letting Python raise a cryptic TypeError on the missing kwargs.
        bucket_metric = {
            "category": "sum_by_category",
            "subcategory": "sum_by_category",
            "wallet":   "sum_by_wallet",
            "tag":      "sum_by_tag",
            "total":    ("sum_expense" if transaction_type == "expense"
                         else "sum_income"),
        }.get(by)
        if bucket_metric is None:
            raise ValueError(
                f"compare_periods: by={by!r} is not valid — use one of "
                "{'category','subcategory','wallet','tag','total'}. For a trend across "
                f"months/weeks, call spending_trend(group_by={by!r}) instead."
            )
        if period2_start is None or period2_end is None:
            raise ValueError(
                "compare_periods compares TWO periods — pass period2_start AND "
                "period2_end. For a trend across N months, call "
                "spending_trend(group_by='month') instead."
            )

        def _fetch(s: str | date, e: str | date) -> list[dict]:
            spec = self._spec(
                metric=bucket_metric,
                start=s,
                end=e,
                currency=currency,
                convert_to_thb=convert_to_thb,
                transaction_type=transaction_type if by == "total" else None,
            )
            # Leaf grouping made "ช้อปปิ้ง" mean only items filed on the root
            # itself (16,900 vs 0) while the report says 25,345 vs 7,244.
            spec.category_level = "parent" if by == "category" else "leaf"
            return self._exec(spec)

        rows1 = _fetch(period1_start, period1_end)
        rows2 = _fetch(period2_start, period2_end)
        # Index by bucket for diff. For "total", both are single-row sums.
        def _key(r: dict) -> str:
            return str(r.get("bucket", "TOTAL"))

        def _amt(r: dict) -> Decimal:
            from decimal import Decimal as _D
            v = r.get("amount") or r.get("amount_thb") or 0
            try:
                return _D(str(v))
            except Exception:
                return _D(0)

        idx1 = {_key(r): _amt(r) for r in rows1}
        idx2 = {_key(r): _amt(r) for r in rows2}
        from decimal import Decimal as _D
        out = []
        for k in sorted(set(idx1) | set(idx2)):
            a, b = idx1.get(k, _D(0)), idx2.get(k, _D(0))
            diff = b - a
            pct = (diff / a * 100) if a else None
            out.append({
                "bucket": k,
                "period1_amount": str(a),
                "period2_amount": str(b),
                "diff": str(diff),
                "pct_change": (str(pct) if pct is not None else None),
            })

        return {
            "by": by,
            "period1": f"{period1_start} → {period1_end}",
            "period2": f"{period2_start} → {period2_end}",
            "rows": out,
        }

    def spending_pace(
        self,
        *,
        as_of: str | date | None = None,
        wallet_names: list[str] | None = None,
        category_names: list[str] | None = None,
        currency: str = "ALL",
        convert_to_thb: bool = True,
    ) -> dict:
        """Project end-of-month spend based on daily pace from the 1st →
        as_of (default today). Returns {spent_so_far, days_elapsed,
        days_in_month, daily_avg, projected_total, projected_remaining}.
        """
        from datetime import date as _date
        from decimal import Decimal as _D
        cur = self._to_date(as_of) if as_of else self._today
        month_start = cur.replace(day=1)
        # last day of month
        if cur.month == 12:
            next_month = _date(cur.year + 1, 1, 1)
        else:
            next_month = _date(cur.year, cur.month + 1, 1)
        from datetime import timedelta as _td
        month_end = next_month - _td(days=1)
        days_elapsed = (cur - month_start).days + 1
        days_in_month = (month_end - month_start).days + 1

        rows = self._exec(
            self._spec(
                metric="sum_expense",
                start=month_start,
                end=cur,
                wallet_names=wallet_names,
                category_names=category_names,
                currency=currency,
                convert_to_thb=convert_to_thb,
            )
        )
        spent = _D(0)
        for r in rows:
            v = r.get("amount") or r.get("amount_thb") or 0
            try:
                spent += _D(str(v))
            except Exception:
                pass
        daily_avg = spent / days_elapsed if days_elapsed else _D(0)
        projected = daily_avg * days_in_month
        return {
            "month_start": month_start.isoformat(),
            "as_of": cur.isoformat(),
            "month_end": month_end.isoformat(),
            "days_elapsed": days_elapsed,
            "days_in_month": days_in_month,
            "spent_so_far": str(spent),
            "daily_avg": str(daily_avg),
            "projected_total": str(projected),
            "projected_remaining": str(projected - spent),
        }

    def anomaly(
        self,
        *,
        category_names: list[str] | None = None,
        wallet_names: list[str] | None = None,
        lookback_days: int = 30,
        as_of: str | date | None = None,
    ) -> dict:
        """Flag whether today's spend (in optional category/wallet) is
        unusually high vs the daily average over the last `lookback_days`.

        Returns {today_spent, lookback_avg, ratio, times_vs_avg,
        pct_above_avg, anomaly_level} where:
          • ratio / times_vs_avg — today ÷ daily-average, a MULTIPLIER
            (e.g. 85.8 means today is 85.8× a usual day — NOT 85.8%).
          • pct_above_avg — percent ABOVE the daily average
            (e.g. 8483 means 8,483% above; use this if you want a %).
          • anomaly_level ∈ {'normal','elevated','high','very_high'}.
        """
        from datetime import date as _date, timedelta as _td
        from decimal import Decimal as _D
        cur = self._to_date(as_of) if as_of else self._today
        win_start = cur - _td(days=lookback_days)

        def _spend(s: _date, e: _date) -> _D:
            rows = self._exec(
                self._spec(
                    metric="sum_expense",
                    start=s,
                    end=e,
                    wallet_names=wallet_names,
                    category_names=category_names,
                    convert_to_thb=True,
                )
            )
            total = _D(0)
            for r in rows:
                v = r.get("amount") or r.get("amount_thb") or 0
                try:
                    total += _D(str(v))
                except Exception:
                    pass
            return total

        today_spent = _spend(cur, cur)
        window_total = _spend(win_start, cur - _td(days=1))
        lookback_avg = window_total / lookback_days if lookback_days else _D(0)
        ratio = (today_spent / lookback_avg) if lookback_avg else None
        if ratio is None or today_spent == 0:
            level = "normal"
        elif ratio >= 3:
            level = "very_high"
        elif ratio >= 2:
            level = "high"
        elif ratio >= 1.5:
            level = "elevated"
        else:
            level = "normal"
        # ratio is a MULTIPLIER (today ÷ daily-avg), not a percent. Expose
        # both a rounded multiplier and an explicit "% above average" so the
        # LLM never has to guess the unit (it used to render 85.8 as "85%").
        times_vs_avg = (
            ratio.quantize(_D("0.1")) if ratio is not None else None
        )
        pct_above_avg = (
            int((ratio - _D(1)) * _D(100)) if ratio is not None else None
        )
        return {
            "as_of": cur.isoformat(),
            "lookback_days": lookback_days,
            "today_spent": str(today_spent),
            "lookback_avg": str(lookback_avg),
            "ratio": (str(ratio) if ratio is not None else None),
            "times_vs_avg": (str(times_vs_avg) if times_vs_avg is not None else None),
            "pct_above_avg": pct_above_avg,
            "anomaly_level": level,
        }


def _clamped(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _day_of_month_on_or_after(today: date, day: int | None) -> date | None:
    """Next date (today inclusive) that falls on `day` of a month, clamped to
    the month length — e.g. a card's next payment-due date."""
    if not day:
        return None
    d = _clamped(today.year, today.month, day)
    if d >= today:
        return d
    y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return _clamped(y, m, day)


def _day_of_month_on_or_before(today: date, day: int | None) -> date | None:
    """Latest date (today inclusive) on `day` of a month — e.g. the last
    statement (billing-cycle close) date."""
    if not day:
        return None
    d = _clamped(today.year, today.month, day)
    if d <= today:
        return d
    y, m = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    return _clamped(y, m, day)


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

    REGRESSION GUARD: every key returned here MUST be documented in the
    `# CODEACT TOOLBOX` section of the system prompt (`prompts.py`). The
    prompt is the LLM's contract; an undocumented helper invites the model
    to invent fake helper names (the 1,234.56 hallucination cause — see
    memory `project_codeact_dual_impl`). UT-NS01 + UT-P01 enforce lockstep.
    """
    w = _Wrappers(
        user_id=user_id,
        catalog=catalog,
        today=today,
        main_loop=main_loop,
    )

    # Shared so spend_for() and the namespace key use the SAME resolver instance.
    resolve_category = make_resolve_category(catalog, main_loop)

    def spend_for(
        term: str,
        *,
        start: str | date,
        end: str | date,
        transaction_type: str = "expense",
        note_term: str | None = None,
    ) -> dict:
        """Deterministic "spent/earned on <term>" lookup that decides CATEGORY vs
        NOTE instead of letting the LLM improvise the branching (which drifts —
        see the 5,205 phantom). NEVER merges the two: a category total and a note
        search answer different questions, so this returns BOTH separately plus a
        `mode` telling the caller which one is meaningful (or to clarify).

        mode:
          "category"  — `term` is a real category, the note adds nothing outside it
                        → answer cat_total.
          "note"      — `term` is NOT a category (resolver force-picked an unrelated
                        one) → the spend lives in notes → answer note_total.
          "ambiguous" — `term` IS a category AND the note search finds money OUTSIDE
                        that category → caller should clarify(cat_total vs note_total).

        Totals come from sum_income/sum_expense (SQL SUM — exact, uncapped); the
        *_txns lists are display rows (capped) for the breakdown.
        """
        # Note keyword: strip the generic Thai bill prefix "ค่า" so "ค่าโทรศัพท์"
        # searches notes for "โทรศัพท์". resolve_category still gets the full term.
        kw = note_term or (
            term[len("ค่า") :] if term.startswith("ค่า") and len(term) > len("ค่า") else term
        )
        try:
            cats = resolve_category(term)
        except Exception:
            cats = []
        # Case-insensitive: users type "subscription" for the "Subscription"
        # category; a case-sensitive check fell to the note search and reported 0.
        cat_match = any(kw.casefold() in c.casefold() for c in cats)

        is_income = transaction_type == "income"
        sum_fn = w.sum_income if is_income else w.sum_expense

        def _sum(**flt) -> tuple[Decimal, int]:
            rows = sum_fn(start=start, end=end, convert_to_thb=True, **flt)
            # SUM over zero matching rows comes back as NULL, not 0 — Decimal("None")
            # raised and the LLM retried the same call until the recursion limit.
            if rows and rows[0].get("amount") is not None:
                return Decimal(str(rows[0]["amount"])), int(rows[0].get("cnt") or 0)
            return Decimal(0), 0

        cat_total, cat_cnt = _sum(category_names=cats) if cat_match else (Decimal(0), 0)
        note_total, note_cnt = _sum(note_query=kw)
        # Money matched by note that sits OUTSIDE the resolved category → the fork.
        note_in_cat, _ = _sum(note_query=kw, category_names=cats) if cat_match else (Decimal(0), 0)
        has_outside = cat_match and (note_total - note_in_cat) > 0

        if not cat_match:
            mode = "note"
        elif has_outside:
            mode = "ambiguous"
        else:
            mode = "category"

        cat_txns = (
            w.list_transactions(
                start=start, end=end, category_names=cats,
                transaction_type=transaction_type, order_by="amount_desc",
            )
            if cat_match
            else []
        )
        note_txns = w.list_transactions(
            start=start, end=end, note_query=kw,
            transaction_type=transaction_type, order_by="amount_desc",
        )
        return {
            "mode": mode,
            "term": term,
            "keyword": kw,
            "type": transaction_type,
            "cat_total": cat_total, "cat_cnt": cat_cnt, "cat_txns": cat_txns,
            "note_total": note_total, "note_cnt": note_cnt, "note_txns": note_txns,
        }

    return {
        # Templates (the only DB access)
        "sum_income":          w.sum_income,
        "sum_expense":         w.sum_expense,
        "sum_by_category":     w.sum_by_category,
        "sum_by_wallet":       w.sum_by_wallet,
        "fixed_costs":         w.fixed_costs,
        "spending_by_weekday": w.spending_by_weekday,
        "month_end_outlook":   w.month_end_outlook,
        "sum_by_tag":          w.sum_by_tag,
        "list_transactions":   w.list_transactions,
        "balance":             w.balance,
        "budget_remaining":    w.budget_remaining,
        "budget_transactions": w.budget_transactions,
        "budget_list":         w.budget_list,
        "creditcard_list":     w.creditcard_list,
        "goal_list":           w.goal_list,
        "goal_progress":       w.goal_progress,
        "goal_transactions":   w.goal_transactions,
        # Counts / discovery / analytics
        "count_transactions":  w.count_transactions,
        "wallet_list":         w.wallet_list,
        "category_list":       w.category_list,
        "tag_list":            w.tag_list,
        "spending_trend":      w.spending_trend,
        "transaction_stats":   w.transaction_stats,
        "top_transactions":    w.top_transactions,
        "currency_rate":       w.currency_rate,
        "active_period":       w.active_period,
        "compare_periods":     w.compare_periods,
        "spending_pace":       w.spending_pace,
        "anomaly":             w.anomaly,
        # Spent/earned-on-<term> with deterministic category-vs-note resolution.
        "spend_for":           spend_for,
        # Smart CodeAct helpers — entity & time resolution + clarification
        "resolve_wallet":      make_resolve_wallet(catalog, main_loop),
        "resolve_category":    resolve_category,
        "resolve_tag":         make_resolve_tag(catalog, main_loop),
        "resolve_budget":      make_resolve_budget(catalog, main_loop),
        "resolve_goal":        make_resolve_goal(catalog, main_loop),
        "parse_period":        lambda phrase=None: _parse_period(phrase, today),
        "clarify":             clarify,
        # Financial decision calculators (Wave 5) — pure Decimal math
        "compute_dti":           compute_dti,
        "mortgage_payment":      mortgage_payment,
        "affordability_check":   affordability_check,
        "refi_payback_months":   refi_payback_months,
        "debt_payoff_months":    debt_payoff_months,
        "emergency_fund_target": emergency_fund_target,
        # Decimal-safe primitives
        "Decimal":             Decimal,
        "date":                date,
        "timedelta":           timedelta,
        "today":               lambda: today,
        # Output marker — the LLM sets this to terminate the loop.
        "result":              None,
    }
