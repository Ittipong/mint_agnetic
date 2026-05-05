"""User entity catalog — single source of truth for wallet/category/tag names.

Fetched once per turn (cheap small-N query through the shared pool) and
threaded into every LLM call that might mention an entity:

  reason_node (system prompt) → planner (user message context) →
  entity_resolver (rerank candidates) → responder (tool output header)

Goal: every LLM in the pipeline sees the canonical names verbatim, so it
never paraphrases "TrueMonney" → "TrueMoney" or invents categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# `get_pool` is imported lazily inside the function to avoid a circular import:
# src.entity_catalog ← imported by src.graph.compute_subgraph.graph
# (the package's __init__ pulls graph.py, which would re-enter this module).


@dataclass(frozen=True)
class WalletEntry:
    sync_id: str
    name: str
    currency: str
    wallet_type: str = "general"  # 'general' | 'creditcard' | 'goal'
    initial_balance: float | None = None
    wallet_category: str | None = None  # e.g. 'savings', 'cash', 'eWallet', 'promptPay', 'salary', 'sales', 'other'
    cached_balance: float | None = None
    ai_message: str | None = None
    icon: str | None = None


@dataclass(frozen=True)
class CategoryEntry:
    sync_id: str
    name: str
    type: str  # 'income' | 'expense' | other
    parent_id: str | None = None  # parent category sync_id for hierarchy
    keywords: list[str] | None = None  # synonyms / related terms for semantic matching


@dataclass(frozen=True)
class TagEntry:
    sync_id: str
    name: str


@dataclass(frozen=True)
class BudgetEntry:
    sync_id: str
    name: str
    currency: str
    period: str  # 'monthly' | 'yearly' | …


@dataclass(frozen=True)
class GoalEntry:
    sync_id: str
    name: str
    currency: str
    target_amount: str  # stringified Decimal — display only, not for math


@dataclass(frozen=True)
class EntityCatalog:
    """Snapshot of one user's named entities at a point in time."""

    wallets: list[WalletEntry] = field(default_factory=list)
    categories: list[CategoryEntry] = field(default_factory=list)
    tags: list[TagEntry] = field(default_factory=list)
    budgets: list[BudgetEntry] = field(default_factory=list)
    goals: list[GoalEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "wallets": [
                {"sync_id": w.sync_id, "name": w.name, "currency": w.currency,
                 "wallet_type": w.wallet_type, "initial_balance": w.initial_balance,
                 "wallet_category": w.wallet_category, "cached_balance": w.cached_balance,
                 "ai_message": w.ai_message, "icon": w.icon}
                for w in self.wallets
            ],
            "categories": [
                {"sync_id": c.sync_id, "name": c.name, "type": c.type,
                 "parent_id": c.parent_id, "keywords": c.keywords}
                for c in self.categories
            ],
            "tags": [{"sync_id": t.sync_id, "name": t.name} for t in self.tags],
            "budgets": [
                {"sync_id": b.sync_id, "name": b.name, "currency": b.currency, "period": b.period}
                for b in self.budgets
            ],
            "goals": [
                {"sync_id": g.sync_id, "name": g.name, "currency": g.currency,
                 "target_amount": g.target_amount}
                for g in self.goals
            ],
        }

    def render_for_prompt(self) -> str:
        """Markdown rendering — used inside system/user prompts.

        Names are wrapped in backticks so the LLM is biased to copy them
        verbatim (the tokenizer treats backticked spans as single units more
        often than not).
        """
        lines = []

        if self.wallets:
            lines.append("### Wallets")
            for w in self.wallets:
                parts = [f"`{w.name}`"]
                parts.append(f"type: {w.wallet_type}")
                parts.append(f"currency: {w.currency}")
                if w.wallet_category:
                    parts.append(f"category: {w.wallet_category}")
                if w.initial_balance is not None:
                    parts.append(f"initial: {w.initial_balance}")
                if w.ai_message:
                    parts.append(f"note: {w.ai_message}")
                lines.append("- " + ", ".join(parts))
        else:
            lines.append("### Wallets\n(no wallets)")

        if self.categories:
            lines.append("\n### Categories")
            # Group by type so the planner can pick income vs expense more easily
            income = [c for c in self.categories if c.type == "income"]
            expense = [c for c in self.categories if c.type == "expense"]
            other = [c for c in self.categories if c.type not in {"income", "expense"}]
            if income:
                lines.append("Income:")
                lines.extend(f"- `{c.name}`" for c in income)
            if expense:
                lines.append("Expense:")
                lines.extend(f"- `{c.name}`" for c in expense)
            if other:
                lines.append("Other:")
                lines.extend(f"- `{c.name}` ({c.type})" for c in other)

        if self.tags:
            lines.append("\n### Tags")
            lines.extend(f"- `{t.name}`" for t in self.tags)

        if self.budgets:
            lines.append("\n### Budgets")
            lines.extend(
                f"- `{b.name}` ({b.period}, {b.currency})" for b in self.budgets
            )

        if self.goals:
            lines.append("\n### Savings Goals")
            lines.extend(
                f"- `{g.name}` (target {g.target_amount} {g.currency})"
                for g in self.goals
            )

        return "\n".join(lines)


# ── Fetch ────────────────────────────────────────────────────────────────────


_WALLETS_SQL = (
    "SELECT "
    "  sync_id::text AS sync_id, "
    "  name, "
    "  currency, "
    "  'general' AS wallet_type, "
    "  initial_balance::float, "
    "  wallet_category, "
    "  cached_balance::float, "
    "  ai_message, "
    "  icon "
    "FROM general_wallets "
    "WHERE user_id = $1 AND deleted_at IS NULL "
    "UNION ALL "
    "SELECT "
    "  sync_id::text AS sync_id, "
    "  name, "
    "  currency, "
    "  'creditcard' AS wallet_type, "
    "  NULL::float AS initial_balance, "
    "  NULL::text AS wallet_category, "
    "  cached_used_amount::float AS cached_balance, "
    "  ai_message, "
    "  icon "
    "FROM creditcard_wallets "
    "WHERE user_id = $1 AND deleted_at IS NULL "
    "UNION ALL "
    "SELECT "
    "  sync_id::text AS sync_id, "
    "  name, "
    "  currency, "
    "  'goal' AS wallet_type, "
    "  NULL::float AS initial_balance, "
    "  NULL::text AS wallet_category, "
    "  NULL::float AS cached_balance, "
    "  NULL::text AS ai_message, "
    "  NULL::text AS icon "
    "FROM goal_wallets "
    "WHERE user_id = $1 AND deleted_at IS NULL "
    "ORDER BY name"
)

_CATEGORIES_SQL = (
    "SELECT sync_id, name, type, parent_sync_id "
    "FROM categories "
    "WHERE (user_id = $1 OR user_id IS NULL) "
    "ORDER BY display_order"
)

_TAGS_SQL = (
    "SELECT sync_id, name "
    "FROM tags "
    "WHERE created_by_user_id = $1 AND is_deleted = false "
    "ORDER BY name"
)

_BUDGETS_SQL = (
    "SELECT sync_id, name, currency, period "
    "FROM budgets "
    "WHERE user_id = $1 AND is_deleted = false "
    "ORDER BY start_date DESC"
)

_GOALS_SQL = (
    "SELECT sync_id, name, currency, target_amount "
    "FROM goal_wallets "
    "WHERE user_id = $1 AND is_deleted = false "
    "ORDER BY target_date NULLS LAST"
)


async def fetch_user_catalog(user_id: str) -> EntityCatalog:
    """Single call that pulls everything the LLM may need to name."""
    if not user_id:
        return EntityCatalog()

    from src.graph.compute_subgraph.db import get_pool  # local — see note above

    pool = await get_pool()
    async with pool.acquire() as conn:
        wallets_rows = await conn.fetch(_WALLETS_SQL, user_id)
        categories_rows = await conn.fetch(_CATEGORIES_SQL, user_id)
        tags_rows = await conn.fetch(_TAGS_SQL, user_id)
        budgets_rows = await conn.fetch(_BUDGETS_SQL, user_id)
        goals_rows = await conn.fetch(_GOALS_SQL, user_id)

    # Dedup categories by (name, type). Each wallet has its own copy of the
    # system categories (so "อาหาร" appears 6× for a 6-wallet user). For the
    # planner/resolver/responder, only the display name matters — the SQL
    # builder filters by `transactions.category_name` (denormalized).
    seen: set[tuple[str, str]] = set()
    deduped_categories: list[CategoryEntry] = []
    for r in categories_rows:
        key = (r["name"], r["type"])
        if key in seen:
            continue
        seen.add(key)
        deduped_categories.append(
            CategoryEntry(
                sync_id=str(r["sync_id"]),
                name=r["name"],
                type=r["type"],
                parent_id=str(r["parent_sync_id"]) if r["parent_sync_id"] else None,
                keywords=None,  # not available in DB schema
            )
        )

    # Dedup budgets by name too — a user can roll a "monthly" budget every
    # period, leaving multiple rows with the same name. The catalog only
    # cares about the name for grounding LLM output, not the period instance.
    seen_b: set[str] = set()
    deduped_budgets: list[BudgetEntry] = []
    for r in budgets_rows:
        if r["name"] in seen_b:
            continue
        seen_b.add(r["name"])
        deduped_budgets.append(
            BudgetEntry(
                sync_id=str(r["sync_id"]),
                name=r["name"],
                currency=r["currency"],
                period=r["period"],
            )
        )

    return EntityCatalog(
        wallets=[
            WalletEntry(sync_id=str(r["sync_id"]), name=r["name"], currency=r["currency"])
            for r in wallets_rows
        ],
        categories=deduped_categories,
        tags=[TagEntry(sync_id=str(r["sync_id"]), name=r["name"]) for r in tags_rows],
        budgets=deduped_budgets,
        goals=[
            GoalEntry(
                sync_id=str(r["sync_id"]),
                name=r["name"],
                currency=r["currency"],
                target_amount=str(r["target_amount"]),
            )
            for r in goals_rows
        ],
    )
