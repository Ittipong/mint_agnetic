"""User entity catalog — canonical names + sync_ids for the chat graph.

Ported from `mint_agentic/src/entity_catalog.py` and adapted for
mint_agentic_v2:

  - psycopg pool (AsyncConnectionPool) instead of asyncpg.
  - Loader takes the pool as an explicit argument so tests can inject
    mocks (`load_entity_catalog(pool=None, ...)` returns an empty catalog).
  - Only the fields the v2 chat graph reads are surfaced; slip-flow
    helpers from the original module are intentionally omitted.

Why we re-fetch every turn: small-N (~tens of wallets/categories) per
user, cheap query, and grounding the LLM with canonical names prevents
paraphrase hallucinations like "TrueMonney" → "TrueMoney".
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WalletEntry:
    sync_id: str
    name: str
    currency: str
    wallet_type: str = "general"        # general | creditcard | goal
    is_default: bool = False
    initial_balance: float | None = None
    wallet_category: str | None = None
    cached_balance: float | None = None
    usage_count: int = 0
    last_used_at: str | None = None
    # Raw icon JSON string from the DB `icon` column, e.g.
    # '{"type":"asset","value":"assets/.../wallet.png","bg":"0xFFE4E0FF"}'.
    # Passed through verbatim to the chat client (NOT injected into the LLM
    # prompt) so mobile can render the wallet's icon in a clarification chip.
    icon: str | None = None


@dataclass(frozen=True)
class CategoryEntry:
    sync_id: str
    name: str
    type: str
    parent_id: str | None = None
    usage_count: int = 0
    # Optional list of synonyms / Thai-English keywords used by the codeact
    # resolver to disambiguate fuzzy matches. Defaults to None so existing
    # call sites (tests, loader) keep working without changes.
    keywords: list[str] | None = None
    # The wallet this category belongs to. Categories are wallet-scoped in the
    # DB (each wallet owns its own copy with a distinct sync_id); None means a
    # global/system category shared by every wallet. The ADD-path resolver
    # filters candidates by this so "เก้าอี้" on a wallet without เฟอร์นิเจอร์
    # never resolves to another wallet's category.
    wallet_sync_id: str | None = None


@dataclass(frozen=True)
class TagEntry:
    sync_id: str
    name: str


@dataclass(frozen=True)
class EntityCatalog:
    """Snapshot of one user's named entities."""

    wallets: list[WalletEntry] = field(default_factory=list)
    # Deduped by (name, type) — one canonical entry per category name, used for
    # the LLM prompt and the codeact/sandbox resolvers (they don't need wallet
    # scope and must not see 6 copies of "อาหาร").
    categories: list[CategoryEntry] = field(default_factory=list)
    # Full per-wallet list (deduped within (wallet, name, type)) — each wallet's
    # own sync_id is preserved so the ADD resolver can scope to one wallet.
    categories_all: list[CategoryEntry] = field(default_factory=list)
    tags: list[TagEntry] = field(default_factory=list)

    def categories_for(self, wallet_sync_id: str | None) -> list[CategoryEntry]:
        """Candidates for ADD-path category resolution scoped to one wallet:
        that wallet's own categories PLUS global ones (`wallet_sync_id IS NULL`,
        shared by every wallet).

        Falls back to the deduped `categories` list when `categories_all` is
        empty (e.g. a hand-built test catalog that only set `categories`); those
        entries default to `wallet_sync_id=None` and so count as global.
        """
        pool = self.categories_all or self.categories
        if wallet_sync_id is None:
            return list(pool)
        return [c for c in pool if c.wallet_sync_id in (wallet_sync_id, None)]

    def wallet_by_sync_id(self, sync_id: str | None) -> "WalletEntry | None":
        """Find a wallet by sync_id, or None."""
        if not sync_id:
            return None
        for w in self.wallets:
            if w.sync_id == sync_id:
                return w
        return None

    def slip_wallet(self, preferred_sync_id: str | None) -> "WalletEntry | None":
        """Resolve the wallet a slip's transactions land in.

        Honors the user's explicit pick (`preferred_sync_id`, any wallet type)
        when it exists; otherwise falls back to the first general wallet
        (`wallets` is ordered general → creditcard → goal, so this is the same
        index-0 spending-wallet fallback the ADD path uses), then the first
        wallet outright. None only when the user has no wallets at all.
        """
        chosen = self.wallet_by_sync_id(preferred_sync_id)
        if chosen is not None:
            return chosen
        for w in self.wallets:
            if w.wallet_type == "general":
                return w
        return self.wallets[0] if self.wallets else None

    def render_for_slip(self, wallet_sync_id: str | None) -> str:
        """Render ONE wallet's categories for the slip prompt — expense/income
        grouped, sync_id surfaced verbatim so the vision model copies them into
        `category_sync_id` without paraphrasing.

        Scoped to the slip's target wallet (the user already picked it), so —
        unlike the v1 slip renderer — we do NOT list other wallets for the
        model to choose among. `categories_for` adds the global/system
        categories shared by every wallet.
        """
        cats = self.categories_for(wallet_sync_id)
        expense = [c for c in cats if c.type == "expense"]
        income = [c for c in cats if c.type == "income"]
        lines: list[str] = []
        if expense:
            lines.append("Expense categories:")
            lines.extend(f"- sync_id=`{c.sync_id}` name=`{c.name}`" for c in expense)
        if income:
            lines.append("Income categories:")
            lines.extend(f"- sync_id=`{c.sync_id}` name=`{c.name}`" for c in income)
        return "\n".join(lines) if lines else "(no categories for this wallet)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "wallets": [
                {"sync_id": w.sync_id, "name": w.name, "currency": w.currency,
                 "wallet_type": w.wallet_type, "is_default": w.is_default,
                 "usage_count": w.usage_count}
                for w in self.wallets
            ],
            "categories": [
                {"sync_id": c.sync_id, "name": c.name, "type": c.type,
                 "parent_id": c.parent_id, "usage_count": c.usage_count}
                for c in self.categories
            ],
            "tags": [{"sync_id": t.sync_id, "name": t.name} for t in self.tags],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityCatalog":
        """Rehydrate from the marshaled dict in state.user_context.

        Round-trips against `to_dict()`. We pre-filter each row to the
        fields declared on the corresponding dataclass so extra keys
        (e.g. legacy `default_currency_code`) don't raise TypeError, and
        missing optional keys (e.g. `icon`) fall back to the dataclass
        defaults. `categories_all` is reconstructed from `categories`
        when absent — `categories_for()` already tolerates that shape.
        """
        # Field whitelists are derived from the dataclasses themselves so this
        # patch stays in sync if a field is added/removed upstream.
        wallet_fields = {f.name for f in fields(WalletEntry)}
        category_fields = {f.name for f in fields(CategoryEntry)}
        tag_fields = {f.name for f in fields(TagEntry)}

        wallets = [
            WalletEntry(**{k: v for k, v in w.items() if k in wallet_fields})
            for w in d.get("wallets", [])
        ]
        categories = [
            CategoryEntry(**{k: v for k, v in c.items() if k in category_fields})
            for c in d.get("categories", [])
        ]
        # `categories_all` is the per-wallet view; fall back to `categories`
        # when callers serialized only the deduped list.
        raw_all = d.get("categories_all") or d.get("categories", [])
        categories_all = [
            CategoryEntry(**{k: v for k, v in c.items() if k in category_fields})
            for c in raw_all
        ]
        tags = [
            TagEntry(**{k: v for k, v in t.items() if k in tag_fields})
            for t in d.get("tags", [])
        ]
        return cls(
            wallets=wallets,
            categories=categories,
            categories_all=categories_all,
            tags=tags,
        )

    def render_for_prompt(self, *, top_n: int = 20) -> str:
        """Compact markdown rendering. Top-N most-used wallets/categories.

        The footer steers the LLM toward `lookup_entity` for long-tail
        names instead of guessing a sync_id.
        """
        lines: list[str] = []

        wallets = self.wallets[:top_n] if top_n > 0 else self.wallets
        if wallets:
            lines.append("### Wallets")
            for w in wallets:
                default_tag = " (default)" if w.is_default else ""
                lines.append(
                    f"- `{w.name}` sync_id=`{w.sync_id}` "
                    f"type={w.wallet_type} currency={w.currency}{default_tag}"
                )
        else:
            lines.append("### Wallets\n(no wallets)")

        categories = self.categories[:top_n] if top_n > 0 else self.categories
        if categories:
            lines.append("\n### Categories")
            income = [c for c in categories if c.type == "income"]
            expense = [c for c in categories if c.type == "expense"]
            other = [c for c in categories if c.type not in {"income", "expense"}]
            if income:
                lines.append("Income:")
                lines.extend(f"- `{c.name}` sync_id=`{c.sync_id}`" for c in income)
            if expense:
                lines.append("Expense:")
                lines.extend(f"- `{c.name}` sync_id=`{c.sync_id}`" for c in expense)
            if other:
                lines.append("Other:")
                lines.extend(f"- `{c.name}` sync_id=`{c.sync_id}` ({c.type})"
                             for c in other)

        if self.tags:
            lines.append("\n### Tags")
            lines.extend(f"- `{t.name}` sync_id=`{t.sync_id}`" for t in self.tags)

        omitted_w = max(0, len(self.wallets) - len(wallets))
        omitted_c = max(0, len(self.categories) - len(categories))
        if omitted_w or omitted_c:
            lines.append(
                f"\n_top {top_n} by usage. {omitted_w} more wallets + "
                f"{omitted_c} more categories hidden — call `lookup_entity` "
                "if the user names something not listed (do not guess)._"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL — psycopg uses %s placeholders (not $1 like asyncpg).
# ---------------------------------------------------------------------------
_WALLETS_SQL = (
    "WITH owned_wallets AS ( "
    "  SELECT sync_id::text AS sync_id, name, currency, "
    "         'general' AS wallet_type, false AS is_default, icon "
    "  FROM general_wallets "
    "  WHERE user_id = %s AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text, name, currency, 'creditcard', false, icon "
    "  FROM creditcard_wallets "
    "  WHERE user_id = %s AND deleted_at IS NULL "
    "  UNION ALL "
    "  SELECT sync_id::text, name, currency, 'goal', false, icon "
    "  FROM goal_wallets "
    "  WHERE user_id = %s AND deleted_at IS NULL "
    "), "
    "usage AS ( "
    "  SELECT t.wallet_sync_id, COUNT(*) AS usage_count, "
    "         MAX(t.date)::text AS last_used_at "
    "  FROM transactions t "
    "  JOIN owned_wallets o ON o.sync_id = t.wallet_sync_id "
    "  WHERE t.is_deleted = false "
    "  GROUP BY t.wallet_sync_id "
    ") "
    # `icon` is appended LAST so existing row indices 0-6 stay stable;
    # the WalletEntry constructor reads it at r[7].
    "SELECT w.sync_id, w.name, w.currency, w.wallet_type, w.is_default, "
    "       COALESCE(u.usage_count, 0) AS usage_count, u.last_used_at, "
    "       w.icon "
    "FROM owned_wallets w "
    "LEFT JOIN usage u ON u.wallet_sync_id = w.sync_id "
    # wallet_type FIRST so index-0 (the no-wallet fallback in transactions._handle_add)
    # is deterministically the general wallet, then creditcard, then goal — a
    # default transaction must land in the spending wallet, never a credit card
    # or a savings goal just because it happens to be the most-used. Within a
    # type, most-used wins (usage_count DESC), name as the final stable tiebreak.
    "ORDER BY CASE w.wallet_type "
    "           WHEN 'general' THEN 0 WHEN 'creditcard' THEN 1 "
    "           WHEN 'goal' THEN 2 ELSE 3 END, "
    "         usage_count DESC, last_used_at DESC NULLS LAST, w.name"
)

_CATEGORIES_SQL = (
    # wallet_sync_id appended LAST so existing row indices 0-3 stay stable.
    "SELECT sync_id::text, name, type, parent_sync_id::text, wallet_sync_id::text "
    "FROM categories "
    "WHERE (user_id = %s OR user_id IS NULL) "
    "  AND is_deleted = false "
    "ORDER BY display_order, name"
)

_TAGS_SQL = (
    "SELECT sync_id::text, name FROM tags "
    "WHERE created_by_user_id = %s AND is_deleted = false "
    "ORDER BY name"
)


async def load_entity_catalog(pool, user_id: str) -> EntityCatalog:
    """Load the catalog from Postgres. Returns an empty catalog if pool
    is None (test/DI escape hatch) or user_id is empty.

    The caller is responsible for catching `psycopg.OperationalError`
    if the pool is unhealthy — we don't swallow it here because a silent
    empty catalog would make hallucination worse, not better.
    """
    if pool is None or not user_id:
        return EntityCatalog()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_WALLETS_SQL, (user_id, user_id, user_id))
            wallet_rows = await cur.fetchall()
            await cur.execute(_CATEGORIES_SQL, (user_id,))
            category_rows = await cur.fetchall()
            await cur.execute(_TAGS_SQL, (user_id,))
            tag_rows = await cur.fetchall()

    wallets = [
        WalletEntry(
            sync_id=r[0], name=r[1], currency=r[2], wallet_type=r[3],
            is_default=bool(r[4]),
            usage_count=int(r[5] or 0),
            last_used_at=r[6],
            # icon appended last in the SELECT; raw DB string, may be None.
            icon=r[7] if len(r) > 7 else None,
        )
        for r in wallet_rows
    ]

    # Two views of the same rows:
    #  - `deduped`: one canonical entry per (name, type) for the LLM prompt and
    #    the codeact/sandbox resolvers — a 6-wallet user has 6 copies of "อาหาร"
    #    and they only need one.
    #  - `all_cats`: the full per-wallet list (deduped within (wallet, name,
    #    type)) so the ADD resolver can scope to one wallet and keep that
    #    wallet's own sync_id.
    seen: dict[tuple[str, str], int] = {}
    deduped: list[CategoryEntry] = []
    seen_all: set[tuple[str | None, str, str]] = set()
    all_cats: list[CategoryEntry] = []
    for r in category_rows:
        wsid = r[4] if len(r) > 4 else None
        entry = CategoryEntry(
            sync_id=r[0], name=r[1], type=r[2],
            parent_id=r[3] if r[3] else None,
            wallet_sync_id=wsid,
        )
        akey = (wsid, r[1], r[2])
        if akey not in seen_all:
            seen_all.add(akey)
            all_cats.append(entry)
        key = (r[1], r[2])
        if key in seen:
            continue
        seen[key] = len(deduped)
        deduped.append(entry)

    tags = [TagEntry(sync_id=r[0], name=r[1]) for r in tag_rows]

    return EntityCatalog(
        wallets=wallets, categories=deduped, categories_all=all_cats, tags=tags
    )


# ---------------------------------------------------------------------------
# Module-level catalog-loader registry
# ---------------------------------------------------------------------------
#
# `get_user_context` (Wave 3 tool) needs to call `load_catalog_for_user(uid)`
# WITHOUT plumbing a pool through every tool signature — the LangGraph tool
# surface only has `state` available. The FastAPI lifespan installs the
# backend pool once at startup via `set_catalog_pool(pool)`; tests inject a
# fake loader via `set_catalog_loader(fn)` to bypass the pool path entirely.
#
# This is the same pattern used by `session_logger` (ContextVar) and the v2
# `catalog_loader` closure — except surfaced as a process-level setter for
# the tool to pick up. Tests override it cleanly per-test via setattr.

_CATALOG_POOL: Any = None
_CATALOG_LOADER: Any = None


def set_catalog_pool(pool) -> None:
    """Install the backend Postgres pool used by `load_catalog_for_user`.

    Called once by the FastAPI lifespan (Wave 5). Tests usually skip this and
    install a fake loader via `set_catalog_loader` instead.
    """
    global _CATALOG_POOL
    _CATALOG_POOL = pool


def set_catalog_loader(loader) -> None:
    """Install a custom async loader for tests / DI.

    `loader` is `async (user_id) -> EntityCatalog`. When set, it takes
    precedence over the pool-backed default. Pass `None` to clear.
    """
    global _CATALOG_LOADER
    _CATALOG_LOADER = loader


async def load_catalog_for_user(user_id: str) -> EntityCatalog:
    """Top-level catalog loader for `get_user_context`.

    Resolution order:
      1. Test/DI loader installed via `set_catalog_loader` (if any)
      2. Pool-backed loader using `_CATALOG_POOL`
      3. Empty catalog (when neither is configured)

    Never raises — pool errors propagate through `load_entity_catalog`'s own
    contract, but missing wiring degrades to an empty catalog so a misconfigured
    test surfaces obviously instead of crashing the tool.
    """
    if _CATALOG_LOADER is not None:
        return await _CATALOG_LOADER(user_id)
    if _CATALOG_POOL is not None:
        return await load_entity_catalog(_CATALOG_POOL, user_id)
    return EntityCatalog()


__all__ = [
    "EntityCatalog",
    "WalletEntry",
    "CategoryEntry",
    "TagEntry",
    "load_entity_catalog",
    "load_catalog_for_user",
    "set_catalog_pool",
    "set_catalog_loader",
]
