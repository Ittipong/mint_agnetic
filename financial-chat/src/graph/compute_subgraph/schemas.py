"""Pydantic models for the analyze pipeline.

QueryPlan is what the LLM produces; QuerySpec is the resolved, code-only
representation that drives the SQL builder. ResolvedEntity carries confidence
so the gate can branch into clarification when needed.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

# Whitelist of metrics the SQL builder knows how to translate.
# Keep this list small and explicit — adding a new metric must be a code change,
# never an LLM choice.
Metric = Literal[
    "sum_income",
    "sum_expense",
    "balance",
    "list",
    "count",
    "sum_by_category",
    "sum_by_wallet",
    # Budget family — see sql_templates.build_budget_*.
    "budget_list",          # definitions only (name, amount, period, dates)
    "budget_remaining",     # name, amount, spent, remaining, pct_used per budget
    "budget_transactions",  # individual transactions counted toward a budget
    # Goal family — savings goals from the goal_wallets table.
    "goal_list",            # definitions: name, target_amount, target_date, currency
    "goal_progress",        # + current_balance, pct_completed, days_left, daily_required
    "goal_transactions",    # deposits/withdrawals on a specific goal wallet
    # Credit card family — from the creditcard_wallets table.
    "creditcard_list",     # all credit cards: name, credit_limit, used_amount, available
    # Templates-as-Tools escape hatch — used for compose / diff / anomaly /
    # multi-step queries that no single metric can answer. The LLM writes
    # Python that calls the templates above; the sandbox blocks anything else.
    "freeform_codeact",
]

Granularity = Literal["day", "week", "month", "quarter", "year", "all"]
GroupBy = Literal["day", "week", "month", "year", "wallet", "category", "tag"]
SlotKind = Literal["wallet", "category", "tag"]
# Transaction `type` column has 4 distinct values in this schema. Used by
# `metric=list` so users asking "ดูรายการรายรับ" actually get income only.
TxType = Literal["income", "expense", "transfer", "creditCardPay"]


class TimeRange(BaseModel):
    start: date
    end: date
    granularity: Granularity = "month"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    raw_phrase: str | None = None


class EntityMention(BaseModel):
    """A textual mention extracted by the planner before resolution."""

    kind: SlotKind
    text: str


class ResolvedEntity(BaseModel):
    """Resolver output: a candidate the SQL layer can join on."""

    sync_id: str
    display_name: str
    kind: SlotKind
    score: float = Field(ge=0.0, le=1.0)
    alternatives: list[dict] = Field(default_factory=list)
    # For category expansion: additional category NAMES to include in SQL filter
    # e.g., if user asks about "เดินทาง" (travel), expand_ids includes ["แท็กซี่", "BTS/MRT"]
    # These are CATEGORY NAMES (not sync_ids) because the SQL filter matches category_name text
    expand_ids: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """LLM output — high-level intent + raw mentions before resolution.

    Combined intent + plan to save a round-trip (Option B).
    """

    metric: Metric
    entity_mentions: list[EntityMention] = Field(default_factory=list)
    time_phrase: str | None = None
    group_by: GroupBy | None = None
    currency: Literal["THB", "USD", "ALL"] = "ALL"
    # Ordering + cap — used by metric=list and the breakdown metrics.
    # `amount_desc` covers "top N ใหญ่สุด"; default `date_desc` is the
    # "show me what happened" pattern.
    order_by: Literal["date_desc", "amount_desc"] = "date_desc"
    limit: int | None = None  # planner sets explicit "top N"; None = default cap
    # Used by metric=budget_* — filter to budgets whose name matches.
    # Set whenever the user mentions a specific budget ("งบอาหาร" → "อาหาร").
    budget_name_phrase: str | None = None
    # Used by metric=goal_* — filter to savings goals whose name matches.
    goal_name_phrase: str | None = None
    # Cross-currency rollup. When True, aggregation metrics drop the
    # currency split and report a single THB total (using the hybrid FX
    # chain: converted_amount → exchange_rate → currencies.rate). Default
    # False keeps the per-currency separation that prevents implicit FX.
    convert_to_thb: bool = False
    # Filter rows by transaction.type — only meaningful for metric=list and
    # metric=budget_transactions. Aggregation metrics already imply the type
    # (sum_income, sum_expense). Leave None to include all types.
    transaction_type: TxType | None = None
    note: str | None = None  # planner's free-text rationale (debug only)


class QuerySpec(BaseModel):
    """Fully-resolved spec passed to the SQL builder. No LLM beyond this point."""

    metric: Metric
    wallets: list[ResolvedEntity] = Field(default_factory=list)
    categories: list[ResolvedEntity] = Field(default_factory=list)
    tags: list[ResolvedEntity] = Field(default_factory=list)
    time_range: TimeRange
    group_by: GroupBy | None = None
    currency: Literal["THB", "USD", "ALL"] = "ALL"
    order_by: Literal["date_desc", "amount_desc"] = "date_desc"
    limit: int | None = None
    budget_name_phrase: str | None = None
    goal_name_phrase: str | None = None
    convert_to_thb: bool = False
    transaction_type: TxType | None = None


class ClarificationPayload(BaseModel):
    """Sent to the user when confidence is too low to proceed."""

    kind: Literal["entity", "time", "metric"]
    question: str
    candidates: list[dict] = Field(default_factory=list)


class ExecRow(BaseModel):
    """Generic row returned by the executor — values are stringified Decimal."""

    bucket: str | None = None  # group_by label
    currency: str | None = None
    amount: str | None = None  # Decimal serialized as string
    count: int | None = None
    extra: dict = Field(default_factory=dict)


def decimal_to_display(value: Decimal | None, places: int = 2) -> str:
    """Format Decimal for display: thousands separator, fixed places.

    Done in code so the LLM never sees raw numbers — eliminates math hallucination.
    """
    if value is None:
        return "0"
    quantized = value.quantize(Decimal(10) ** -places)
    return f"{quantized:,}"
