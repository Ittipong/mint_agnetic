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
    # Usage signal — drives top-N truncation in render_for_prompt_truncated.
    # Null when the SQL didn't surface it (e.g. legacy fetch path).
    usage_count: int = 0
    last_used_at: str | None = None


@dataclass(frozen=True)
class CategoryEntry:
    sync_id: str
    name: str
    type: str  # 'income' | 'expense' | other
    parent_id: str | None = None  # parent category sync_id for hierarchy
    keywords: list[str] | None = None  # synonyms / related terms for semantic matching
    usage_count: int = 0
    last_used_at: str | None = None


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
        """Full markdown rendering — used by slip/quick-add prompts that
        need EVERY name available (no lookup_entity tool there)."""
        return self._render(wallets=self.wallets, categories=self.categories)

    def render_for_prompt_truncated(self, top_n: int = 20) -> str:
        """Truncated rendering for the ReAct system prompt.

        Wallets and categories are pre-sorted by `usage_count DESC` in the
        SQL fetch, so slicing `[:top_n]` keeps the most-used ones the user
        is likely to mention. Tags/budgets/goals are usually small so we
        render them in full.

        Adds a footer pointing the LLM to `lookup_entity` for anything not
        in the visible top-N — without this the LLM would either guess a
        sync_id or refuse to answer when the user mentions a long-tail
        wallet/category.
        """
        top_wallets = self.wallets[:top_n] if top_n > 0 else self.wallets
        # Categories are grouped by income/expense downstream; truncating
        # the flat list before splitting keeps both groups balanced toward
        # the most-used ones.
        top_categories = (
            self.categories[:top_n] if top_n > 0 else self.categories
        )
        rendered = self._render(wallets=top_wallets, categories=top_categories)
        omitted_w = max(0, len(self.wallets) - len(top_wallets))
        omitted_c = max(0, len(self.categories) - len(top_categories))
        if omitted_w or omitted_c:
            rendered += (
                f"\n\n_แสดง top {top_n} โดยการใช้งานล่าสุด. "
                f"อีก {omitted_w} wallet + {omitted_c} category ที่ใช้น้อยกว่า"
                " ถ้า user พูดถึงให้เรียก tool `lookup_entity` เพื่อค้นหา"
                " sync_id (อย่าเดาเอง)._"
            )
        return rendered

    def _render(
        self,
        *,
        wallets: list[WalletEntry],
        categories: list[CategoryEntry],
    ) -> str:
        """Shared markdown renderer — called by both full and truncated
        variants with the desired subset.
        """
        lines = []

        if wallets:
            lines.append("### Wallets")
            for w in wallets:
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

        if categories:
            lines.append("\n### Categories")
            # Group by type so the planner can pick income vs expense more easily
            income = [c for c in categories if c.type == "income"]
            expense = [c for c in categories if c.type == "expense"]
            other = [c for c in categories if c.type not in {"income", "expense"}]
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
    # Transactions table has no user_id column — ownership comes from
    # joining the wallet tables. We aggregate per wallet_sync_id which
    # is already the join key.
    "WITH owned_wallet_ids AS ( "
    "  SELECT sync_id::text AS sync_id FROM general_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text FROM creditcard_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text FROM goal_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "), "
    "usage AS ( "
    "  SELECT t.wallet_sync_id, "
    "         COUNT(*)::int AS usage_count, "
    "         MAX(t.date)::text AS last_used_at "
    "  FROM transactions t "
    "  JOIN owned_wallet_ids o ON o.sync_id = t.wallet_sync_id "
    "  WHERE t.is_deleted = false "
    "  GROUP BY t.wallet_sync_id "
    ") "
    "SELECT "
    "  w.sync_id, w.name, w.currency, w.wallet_type, "
    "  w.initial_balance, w.wallet_category, w.cached_balance, "
    "  w.ai_message, w.icon, "
    "  COALESCE(u.usage_count, 0) AS usage_count, "
    "  u.last_used_at "
    "FROM ( "
    "  SELECT "
    "    sync_id::text AS sync_id, "
    "    name, "
    "    currency, "
    "    'general' AS wallet_type, "
    "    initial_balance::float AS initial_balance, "
    "    wallet_category, "
    "    cached_balance::float AS cached_balance, "
    "    ai_message, "
    "    icon "
    "  FROM general_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT "
    "    sync_id::text AS sync_id, "
    "    name, "
    "    currency, "
    "    'creditcard' AS wallet_type, "
    "    NULL::float AS initial_balance, "
    "    NULL::text AS wallet_category, "
    "    cached_used_amount::float AS cached_balance, "
    "    ai_message, "
    "    icon "
    "  FROM creditcard_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT "
    "    sync_id::text AS sync_id, "
    "    name, "
    "    currency, "
    "    'goal' AS wallet_type, "
    "    NULL::float AS initial_balance, "
    "    NULL::text AS wallet_category, "
    "    NULL::float AS cached_balance, "
    "    NULL::text AS ai_message, "
    "    NULL::text AS icon "
    "  FROM goal_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    ") w "
    "LEFT JOIN usage u ON u.wallet_sync_id = w.sync_id "
    "ORDER BY usage_count DESC, last_used_at DESC NULLS LAST, w.name"
)

_CATEGORIES_SQL = (
    "WITH owned_wallet_ids AS ( "
    "  SELECT sync_id::text AS sync_id FROM general_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text FROM creditcard_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text FROM goal_wallets "
    "  WHERE user_id = $1 AND deleted_at IS NULL "
    "), "
    "usage AS ( "
    "  SELECT t.category_sync_id, "
    "         COUNT(*)::int AS usage_count, "
    "         MAX(t.date)::text AS last_used_at "
    "  FROM transactions t "
    "  JOIN owned_wallet_ids o ON o.sync_id = t.wallet_sync_id "
    "  WHERE t.is_deleted = false "
    "    AND t.category_sync_id IS NOT NULL "
    "  GROUP BY t.category_sync_id "
    ") "
    "SELECT c.sync_id, c.name, c.type, c.parent_sync_id, "
    "       COALESCE(u.usage_count, 0) AS usage_count, "
    "       u.last_used_at "
    "FROM categories c "
    "LEFT JOIN usage u ON u.category_sync_id = c.sync_id::text "
    "WHERE (c.user_id = $1 OR c.user_id IS NULL) "
    "ORDER BY c.display_order"
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


# Slip flow needs categories grouped by wallet_sync_id (so the LLM
# can match "wallet first → category from that wallet's list"). We
# query separately rather than touching the cached EntityCatalog so
# regular chat flow stays cheap and deduped.
_CATEGORIES_BY_WALLET_SQL = (
    "SELECT wallet_sync_id, sync_id::text AS sync_id, name, type "
    "FROM categories "
    "WHERE user_id = $1 "
    "  AND is_deleted = false "
    "  AND is_active = true "
    "ORDER BY wallet_sync_id, type, display_order, name"
)


async def fetch_categories_by_wallet(user_id: str) -> dict[str, list[dict]]:
    """Return {wallet_sync_id: [{sync_id, name, type}, ...]}.

    Used by slip_node only. The DB stores one categories row per
    wallet (a user with 5 wallets has 5 copies of every system
    category) — preserve that shape so the LLM picks ids that match
    the wallet it just matched.
    """
    from src.graph.compute_subgraph.db import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_CATEGORIES_BY_WALLET_SQL, user_id)

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        bucket = grouped.setdefault(str(r["wallet_sync_id"]), [])
        bucket.append(
            {"sync_id": r["sync_id"], "name": r["name"], "type": r["type"]}
        )
    return grouped


def render_for_slip(
    catalog: "EntityCatalog",
    categories_by_wallet: dict[str, list[dict]],
) -> str:
    """Render wallets + their nested category lists for the slip prompt.

    **Only `wallet_type=general` wallets are surfaced** — slip parsing
    is scoped to cash/bank accounts. Credit-card and goal wallets are
    intentionally hidden so the LLM cannot pick them for income/expense
    proposals (a credit-card wallet ≠ a savings account; a goal wallet
    is a sinking-fund construct, not a real account that receives a
    payslip).

    Output shape (sync_ids surfaced so the LLM can copy them into
    `propose_transaction` arguments verbatim):

        ### Wallet: `TrueMonney` (sync_id=w1, currency=THB)
          Expense categories:
          - sync_id=`c1` name=`อาหาร`
          - ...
          Income categories:
          - sync_id=`c2` name=`เงินเดือน`

    Wallets with zero categories still appear — the LLM may still
    pick the wallet and leave `category_id` null.
    """
    general_wallets = [w for w in catalog.wallets if w.wallet_type == "general"]
    if not general_wallets:
        return (
            "(no general wallets — slip flow requires at least one "
            "cash/bank account; ask user to add one first)"
        )

    lines: list[str] = []
    for w in general_wallets:
        lines.append(
            f"### Wallet: `{w.name}` (sync_id=`{w.sync_id}`, "
            f"currency={w.currency})"
        )
        cats = categories_by_wallet.get(w.sync_id, [])
        if not cats:
            lines.append("  (no categories)")
            continue
        expense = [c for c in cats if c["type"] == "expense"]
        income = [c for c in cats if c["type"] == "income"]
        if expense:
            lines.append("  Expense categories:")
            for c in expense:
                lines.append(f"  - sync_id=`{c['sync_id']}` name=`{c['name']}`")
        if income:
            lines.append("  Income categories:")
            for c in income:
                lines.append(f"  - sync_id=`{c['sync_id']}` name=`{c['name']}`")
    return "\n".join(lines)


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
    # Dedup categories by (name, type) while accumulating usage across
    # duplicates (each wallet has its own copy of system categories — a
    # 6-wallet user has 6× "อาหาร" rows, and usage_count must sum).
    seen: dict[tuple[str, str], int] = {}
    deduped_categories: list[CategoryEntry] = []
    for r in categories_rows:
        key = (r["name"], r["type"])
        u = int(r["usage_count"] or 0)
        last = r["last_used_at"]
        if key in seen:
            idx = seen[key]
            existing = deduped_categories[idx]
            merged_last = (
                existing.last_used_at
                if last is None
                else (
                    last
                    if existing.last_used_at is None
                    else max(existing.last_used_at, last)
                )
            )
            deduped_categories[idx] = CategoryEntry(
                sync_id=existing.sync_id,
                name=existing.name,
                type=existing.type,
                parent_id=existing.parent_id,
                keywords=existing.keywords,
                usage_count=existing.usage_count + u,
                last_used_at=merged_last,
            )
            continue
        seen[key] = len(deduped_categories)
        deduped_categories.append(
            CategoryEntry(
                sync_id=str(r["sync_id"]),
                name=r["name"],
                type=r["type"],
                parent_id=str(r["parent_sync_id"]) if r["parent_sync_id"] else None,
                keywords=None,  # not available in DB schema
                usage_count=u,
                last_used_at=last,
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
            WalletEntry(
                sync_id=str(r["sync_id"]),
                name=r["name"],
                currency=r["currency"],
                wallet_type=r["wallet_type"],
                usage_count=int(r["usage_count"] or 0),
                last_used_at=r["last_used_at"],
            )
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
