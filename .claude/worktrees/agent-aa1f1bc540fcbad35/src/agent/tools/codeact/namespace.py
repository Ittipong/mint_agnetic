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


def _to_wire_number(value: Any) -> Any:
    """Coerce a Decimal/number to a JSON-native number for the mobile wire.

    Blocks are serialized with a plain json.dumps (no Decimal default), so a
    Decimal would raise. We round to 2 dp (money) then drop to int when the
    result is integral (the mobile schema shows `850`, not `850.0`). Returns
    the input unchanged if it is not numeric so a malformed field surfaces in
    validation rather than being silently zeroed.
    """
    if isinstance(value, bool):  # bool is an int subclass — keep as-is
        return value
    if isinstance(value, Decimal):
        q = value.quantize(Decimal("0.01"))
        return int(q) if q == q.to_integral_value() else float(q)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return value


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
        block_sink: list[dict],
    ) -> None:
        self._user_id = user_id
        self._catalog = catalog
        self._today = today
        self._loop = main_loop
        # Option C list-sharing: `block_sink` is the SAME list object that
        # build_namespace put under the namespace dict's `__block_sink__` key.
        # The namespace dict is passed by reference into the worker thread
        # (execute(code, namespace) runs under asyncio.to_thread), and these
        # data methods run on that worker thread — so appending here is visible
        # to run_python after the thread joins, with no state/writer plumbing.
        # The dunder key is unreachable from LLM code (AST validator bans `.`
        # access on dunders + import), so this stays a server-side channel.
        self._block_sink = block_sink

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

    # -- category_breakdown emission (Option C list-sharing) -------------

    def _stage_breakdown_block(
        self,
        *,
        title: str,
        items: list[dict],
        total: Any,
        period_label: str | None = None,
    ) -> None:
        """Stage a `category_breakdown` block onto the shared sink.

        Guard C2b (deterministic, last-wins): the mobile card only earns its
        place when there are ≥ 2 items to compare — a single-row "breakdown"
        is noise the prose already covers. We clear the sink before appending
        so that when several breakdowns run in ONE turn (e.g. the model loops
        over months) only the LAST one survives; run_python reads the tail.
        The decision is made HERE in deterministic Python, never by the LLM —
        there is no emit flag for the model to set (C2b).

        Numbers are passed through verbatim from the SQL rows (Decimal-safe);
        we only shape them into the frozen wire schema. We never recompute an
        amount in Python here beyond the SUM that built `total` upstream.
        """
        if not isinstance(items, list) or len(items) < 2:
            return
        # The block leaves the server via json.dumps WITHOUT a Decimal-aware
        # default (sse_adapter serializes blocks plainly), so every numeric
        # field MUST be a JSON-native number here. We coerce Decimal → int when
        # integral else float — display-only at the wire edge; all upstream
        # math stayed in Decimal. Non-numeric / None numerics are dropped from
        # the optional fields so the mobile decoder never sees a string amount.
        wire_items: list[dict] = []
        for it in items:
            wi: dict[str, Any] = {
                "name": it.get("name"),
                "amount": _to_wire_number(it.get("amount")),
            }
            if it.get("icon") is not None:
                wi["icon"] = it["icon"]  # raw JSONB passthrough
            if it.get("diff") is not None:
                wi["diff"] = _to_wire_number(it["diff"])
            if it.get("pct") is not None:
                wi["pct"] = _to_wire_number(it["pct"])
            wire_items.append(wi)
        block: dict[str, Any] = {
            "type": "category_breakdown",
            "title": title,
            "items": wire_items,
        }
        if total is not None:
            block["total"] = _to_wire_number(total)
        if period_label:
            block["period_label"] = period_label
        # last-wins dedup: a single turn shows at most one breakdown card.
        self._block_sink.clear()
        self._block_sink.append(block)

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
    ) -> list[dict]:
        """Spending breakdown by category (expense only)."""
        rows = self._exec(
            self._spec(
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
        )
        # Stage the mobile category_breakdown card (single-period: no diff/pct).
        # Side effect only — the returned rows are unchanged so existing prose
        # grounding (R3) and tests are untouched.
        self._stage_category_breakdown_single(rows)
        return rows

    def _stage_category_breakdown_single(self, rows: list[dict]) -> None:
        """Build a single-period category_breakdown from sum_by_category rows.

        Each sum_by_category row is one (bucket, currency) pair. The wire
        schema carries ONE `total` number, so a multi-currency result has no
        single coherent total — we skip emission in that case (prose still
        renders every currency; the card is suppressed, not faked). For the
        common single-currency case we keep the helper's amount-desc order and
        pass `icon` (raw JSONB / None) straight through like wallet icons.
        diff/pct are omitted — there is no comparison period here.
        """
        if not rows:
            return
        currencies = {str(r.get("currency", "THB")) for r in rows}
        if len(currencies) > 1:
            return  # multi-currency: no single total → suppress card (correct)
        items: list[dict] = []
        total = Decimal(0)
        for r in rows:
            amt = r.get("amount")
            try:
                amt_dec = Decimal(str(amt))
            except Exception:
                continue
            total += amt_dec
            items.append({
                "name": str(r.get("bucket", "")),
                "amount": amt_dec,
                "icon": r.get("icon"),  # raw JSONB passthrough; None ok
            })
        self._stage_breakdown_block(
            title="หมวดที่ใช้จ่าย",
            items=items,
            total=total,
        )

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
                note_query=note_query,
                match_destination_note=match_destination_note,
                has_note=has_note,
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
        cache_updated_at}. `used` mirrors the backend's cached_used_amount.
        When the cache has never been written it stays NULL (and `available`
        is NULL too) — never substitute a 0 or initial_used here, an unknown
        usage must read as unknown so the user can see the cache is stale.
        """
        return self._exec(
            self._spec(
                metric="creditcard_list",
                start=date(1900, 1, 1),
                end=self._today,
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
        """Compare TWO periods side-by-side. `by` ∈ {'category','wallet',
        'tag','total'}. Returns {period1, period2, diff, pct_change} per
        bucket — one tool call instead of two + Python diff.

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
            "wallet":   "sum_by_wallet",
            "tag":      "sum_by_tag",
            "total":    ("sum_expense" if transaction_type == "expense"
                         else "sum_income"),
        }.get(by)
        if bucket_metric is None:
            raise ValueError(
                f"compare_periods: by={by!r} is not valid — use one of "
                "{'category','wallet','tag','total'}. For a trend across "
                f"months/weeks, call spending_trend(group_by={by!r}) instead."
            )
        if period2_start is None or period2_end is None:
            raise ValueError(
                "compare_periods compares TWO periods — pass period2_start AND "
                "period2_end. For a trend across N months, call "
                "spending_trend(group_by='month') instead."
            )

        def _fetch(s: str | date, e: str | date) -> list[dict]:
            return self._exec(
                self._spec(
                    metric=bucket_metric,
                    start=s,
                    end=e,
                    currency=currency,
                    convert_to_thb=convert_to_thb,
                    transaction_type=transaction_type if by == "total" else None,
                )
            )

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
        # icon by bucket (category compares only) — prefer the period the
        # bucket actually appears in; both periods share the same icon for a
        # given category_name. Raw JSONB passthrough, None when absent.
        icon_by_bucket: dict[str, Any] = {}
        for r in (*rows2, *rows1):  # rows2 first so the current period wins
            icon_by_bucket.setdefault(_key(r), r.get("icon"))
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

        # Stage the mobile category_breakdown card for a category compare.
        # `amount` = current period (period2), `diff`/`pct` from the same
        # Decimal math used above (no head-math). Side effect only.
        if by == "category":
            self._stage_category_breakdown_compare(
                out, icon_by_bucket,
                period_label=f"{period2_start} → {period2_end} vs "
                             f"{period1_start} → {period1_end}",
            )

        return {
            "by": by,
            "period1": f"{period1_start} → {period1_end}",
            "period2": f"{period2_start} → {period2_end}",
            "rows": out,
        }

    def _stage_category_breakdown_compare(
        self,
        rows: list[dict],
        icon_by_bucket: dict[str, Any],
        *,
        period_label: str,
    ) -> None:
        """Build a 2-period category_breakdown from compare_periods rows.

        `amount` is the current period (period2_amount); `diff`/`pct` carry the
        comparison so the mobile card can render the up/down delta. Buckets
        whose current-period amount is 0 (present only in the old period) still
        appear — the card shows them as a drop. Values reuse the Decimal math
        already computed in compare_periods (parsed back from the string repr);
        no new arithmetic is introduced.
        """
        items: list[dict] = []
        total = Decimal(0)
        for r in rows:
            try:
                amount = Decimal(str(r.get("period2_amount", "0")))
                diff = Decimal(str(r.get("diff", "0")))
            except Exception:
                continue
            pct_raw = r.get("pct_change")
            pct = None
            if pct_raw is not None:
                try:
                    pct = Decimal(str(pct_raw))
                except Exception:
                    pct = None
            total += amount
            bucket = str(r.get("bucket", ""))
            items.append({
                "name": bucket,
                "amount": amount,
                "icon": icon_by_bucket.get(bucket),
                "diff": diff,
                "pct": pct,
            })
        # Order biggest-current-spend first so the card matches prose ranking.
        items.sort(key=lambda it: it["amount"], reverse=True)
        self._stage_breakdown_block(
            title="เปรียบเทียบรายหมวด",
            items=items,
            total=total,
            period_label=period_label,
        )

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
    # Option C: a single mutable list shared between the namespace dict (under
    # the LLM-unreachable `__block_sink__` dunder key) and the _Wrappers data
    # methods. The data methods append staged category_breakdown blocks here;
    # run_python reads `namespace["__block_sink__"]` after the worker thread
    # joins and forwards the tail into emitted_blocks_this_turn. The dunder key
    # is banned for LLM code by the AST validator (no dunder attr access).
    block_sink: list[dict] = []
    w = _Wrappers(
        user_id=user_id,
        catalog=catalog,
        today=today,
        main_loop=main_loop,
        block_sink=block_sink,
    )
    return {
        # Server-side block emission channel (NOT callable by the LLM; the AST
        # validator bans dunder access so user code can never touch this).
        "__block_sink__": block_sink,
        # Templates (the only DB access)
        "sum_income":          w.sum_income,
        "sum_expense":         w.sum_expense,
        "sum_by_category":     w.sum_by_category,
        "sum_by_wallet":       w.sum_by_wallet,
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
        # Smart CodeAct helpers — entity & time resolution + clarification
        "resolve_wallet":      make_resolve_wallet(catalog, main_loop),
        "resolve_category":    make_resolve_category(catalog, main_loop),
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
