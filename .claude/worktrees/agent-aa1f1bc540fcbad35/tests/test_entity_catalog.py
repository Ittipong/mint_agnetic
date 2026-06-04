"""Unit tests for src.agent.entity_catalog.

Two cases:
  1. `load_entity_catalog` reads from a mocked psycopg pool and returns
     dataclass instances populated from the canned rows.
  2. `render_for_prompt` produces a compact string containing the
     canonical names + sync_ids — this is what gets injected into the
     understand/codeact LLM system prompt.
"""

from __future__ import annotations

import pytest

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    TagEntry,
    WalletEntry,
    load_entity_catalog,
)


# ---------------------------------------------------------------------------
# Minimal async-context-manager mocks mimicking psycopg's pool/conn/cursor API
# (async with pool.connection() as conn; async with conn.cursor() as cur)
# ---------------------------------------------------------------------------
class _MockCursor:
    def __init__(self, batches: list[list[tuple]]):
        self._batches = batches
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def execute(self, _sql, _params=None):
        return None

    async def fetchall(self):
        # Each call returns the next batch — matches the loader's
        # sequence: wallets → categories → tags.
        batch = self._batches[self._idx]
        self._idx += 1
        return batch


class _MockConn:
    def __init__(self, batches: list[list[tuple]]):
        self._cursor = _MockCursor(batches)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    def cursor(self):
        return self._cursor


class _MockPool:
    def __init__(self, batches: list[list[tuple]]):
        self._conn = _MockConn(batches)

    def connection(self):
        return self._conn


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_UT_EC_001_load_returns_dataclasses_from_mocked_pool():
    """load_entity_catalog feeds rows from the pool into the dataclass
    constructors in the right order (wallets → categories → tags)."""

    wallet_rows = [
        # sync_id, name, currency, wallet_type, is_default, usage_count,
        # last_used_at, icon (icon appended LAST — see _WALLETS_SQL)
        ("w-cash", "Cash", "THB", "general", True, 12, "2026-05-17",
         '{"type":"asset","value":"assets/cetegory_icons/wallet.png","bg":"0xFFE4E0FF"}'),
        # No icon (NULL in DB) → WalletEntry.icon must be None.
        ("w-cc",   "Credit Card", "THB", "creditcard", False, 3, "2026-05-10",
         None),
    ]
    category_rows = [
        # sync_id, name, type, parent_sync_id
        ("c-food", "อาหาร", "expense", None),
        ("c-salary", "เงินเดือน", "income", None),
        # duplicate (name, type) — must be deduped
        ("c-food-dup", "อาหาร", "expense", None),
    ]
    tag_rows = [
        ("t-work", "งาน"),
    ]

    pool = _MockPool([wallet_rows, category_rows, tag_rows])
    catalog = await load_entity_catalog(pool, "user-1")

    assert len(catalog.wallets) == 2
    assert catalog.wallets[0] == WalletEntry(
        sync_id="w-cash", name="Cash", currency="THB",
        wallet_type="general", is_default=True,
        usage_count=12, last_used_at="2026-05-17",
        icon='{"type":"asset","value":"assets/cetegory_icons/wallet.png","bg":"0xFFE4E0FF"}',
    )
    # Raw icon JSON string passed through verbatim from the DB row.
    assert catalog.wallets[0].icon == (
        '{"type":"asset","value":"assets/cetegory_icons/wallet.png",'
        '"bg":"0xFFE4E0FF"}'
    )
    # NULL icon in DB → None on the dataclass.
    assert catalog.wallets[1].icon is None
    # Categories deduped by (name, type) — 3 input rows → 2 unique entries
    assert len(catalog.categories) == 2
    names = {c.name for c in catalog.categories}
    assert names == {"อาหาร", "เงินเดือน"}
    assert catalog.tags == [TagEntry(sync_id="t-work", name="งาน")]


def test_UT_EC_002_render_for_prompt_contains_names_and_sync_ids():
    """render_for_prompt produces a compact markdown string that
    surfaces each entity's canonical name and sync_id — the LLM
    consumes this to return resolved sync_ids in slots."""

    catalog = EntityCatalog(
        wallets=[
            WalletEntry(sync_id="w-cash", name="Cash", currency="THB",
                        is_default=True),
            WalletEntry(sync_id="w-cc", name="Credit Card", currency="THB",
                        wallet_type="creditcard"),
        ],
        categories=[
            CategoryEntry(sync_id="c-food", name="อาหาร", type="expense"),
            CategoryEntry(sync_id="c-sal", name="เงินเดือน", type="income"),
        ],
        tags=[TagEntry(sync_id="t-work", name="งาน")],
    )

    rendered = catalog.render_for_prompt(top_n=20)

    # Every entity name and sync_id must appear so the LLM can pick them up.
    for expected in (
        "Cash", "w-cash", "Credit Card", "w-cc",
        "อาหาร", "c-food", "เงินเดือน", "c-sal",
        "งาน", "t-work",
    ):
        assert expected in rendered, f"missing {expected!r} in render"

    # Section headers signal structure to the LLM.
    assert "### Wallets" in rendered
    assert "### Categories" in rendered
    assert "### Tags" in rendered


def test_UT_EC_003_render_for_prompt_omits_icon():
    """UT-EC-003: the wallet icon is a client-only field — it must NOT
    leak into the LLM prompt. Adding `icon` to WalletEntry must leave
    render_for_prompt output unchanged (no icon JSON, no `icon=` token)."""

    icon_json = (
        '{"type":"asset","value":"assets/cetegory_icons/wallet.png",'
        '"bg":"0xFFE4E0FF"}'
    )
    catalog = EntityCatalog(
        wallets=[
            WalletEntry(sync_id="w-cash", name="Cash", currency="THB",
                        is_default=True, icon=icon_json),
        ],
    )

    rendered = catalog.render_for_prompt(top_n=20)

    # The wallet still renders by name + sync_id...
    assert "Cash" in rendered
    assert "w-cash" in rendered
    # ...but the icon never appears in the prompt (neither the JSON blob
    # nor any `icon` key / asset path).
    assert icon_json not in rendered
    assert "icon" not in rendered
    assert "wallet.png" not in rendered
    assert "0xFFE4E0FF" not in rendered
