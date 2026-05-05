"""Pydantic models for the analyze pipeline (Smart CodeAct).

QuerySpec is the code-only representation that drives the SQL builder.
ResolvedEntity carries the canonical name + sync_id so SQL filters use
the exact catalog entry. Both are constructed inside namespace.py wrappers
when the codeact loop calls a SQL helper.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Whitelist of metrics the SQL builder knows how to translate.
# Keep this list small and explicit — adding a new metric must be a code change.
Metric = Literal[
    "sum_income",
    "sum_expense",
    "balance",
    "list",
    "count",
    "sum_by_category",
    "sum_by_wallet",
    # Budget family
    "budget_list",
    "budget_remaining",
    "budget_transactions",
    # Goal family
    "goal_list",
    "goal_progress",
    "goal_transactions",
    # Credit card family
    "creditcard_list",
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


class ResolvedEntity(BaseModel):
    """Resolver output: a candidate the SQL layer can join on."""

    sync_id: str
    display_name: str
    kind: SlotKind
    score: float = Field(ge=0.0, le=1.0)
    alternatives: list[dict] = Field(default_factory=list)
    # For category expansion: additional category NAMES to include in SQL filter
    # (e.g., user asks "เดินทาง" → expand_ids = ["แท็กซี่", "BTS/MRT"]).
    expand_ids: list[str] = Field(default_factory=list)


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
