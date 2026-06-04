"""Unit tests for src.agent.tools.get_user_context.

Covers UT-T05 (idempotent within turn) + UT-T05b (retry-once on empty wallets
per memory `project_wallet_index0_ordering`).

We bypass the real Postgres path by installing a fake catalog loader via
`set_catalog_loader` (test/DI hook in entity_catalog.py).

Wave 4 update — Command(update=) return type:
  After Issue 1 fix, `get_user_context` returns `Command(update=...)`
  instead of a plain dict. The `_invoke_ctx` helper invokes via the
  ToolCall protocol (required when InjectedToolCallId is in the signature)
  and unwraps both the Command and the JSON ToolMessage body so the
  existing dict-based assertions stay readable.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from langgraph.types import Command

from src.agent import entity_catalog as ec_module
from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
    set_catalog_loader,
)
from src.agent.tools.get_user_context import get_user_context


async def _invoke_ctx(
    *, state: dict, tool_call_id: str = "tc-test",
) -> tuple[Command, dict]:
    """Invoke get_user_context via the ToolCall protocol.

    Returns:
      (command, payload) where `command` is the raw Command(update=...) and
      `payload` is the parsed ToolMessage content (the user_context dict
      OR {"error": ..., "kind": ...}).
    """
    tool_call = {
        "name": "get_user_context",
        "args": {"state": state},
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await get_user_context.ainvoke(tool_call)
    messages = (cmd.update or {}).get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content)
    return cmd, {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _full_catalog() -> EntityCatalog:
    return EntityCatalog(
        wallets=[
            WalletEntry(sync_id="w-cc", name="KCard", currency="THB",
                        wallet_type="creditcard"),
            WalletEntry(sync_id="w-gen", name="Cash", currency="THB",
                        wallet_type="general", is_default=True),
            WalletEntry(sync_id="w-goal", name="House", currency="THB",
                        wallet_type="goal"),
        ],
        categories=[
            CategoryEntry(sync_id="c-1", name="อาหาร", type="expense"),
            CategoryEntry(sync_id="c-2", name="เงินเดือน", type="income"),
        ],
    )


def _empty_catalog() -> EntityCatalog:
    return EntityCatalog(wallets=[], categories=[])


@pytest.fixture(autouse=True)
def _reset_loader():
    """Each test installs its own loader; clear afterward to keep tests isolated."""
    set_catalog_loader(None)
    yield
    set_catalog_loader(None)


# ─────────────────────────────────────────────────────────────────────────────
# UT-T05 — idempotent within a turn (cache hit)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T05_get_user_context_is_idempotent_within_turn():
    """UT-T05: with `state.user_context` already populated (the graph's
    per-turn cache), a fresh tool call MUST short-circuit on the cache and
    NOT hit the loader at all.

    Wave 4 graph wiring is responsible for SETTING `state.user_context` after
    the first call (via Command(update=...) or a reducer). In Wave 3 we test
    the tool's contract: if the cache is present, the loader stays cold.
    """
    call_count = {"n": 0}

    async def fake_loader(user_id: str) -> EntityCatalog:
        call_count["n"] += 1
        return _full_catalog()

    set_catalog_loader(fake_loader)

    async def run():
        # Simulate the cache being populated by a prior call (Wave 4 graph).
        cached = {
            "wallets": [{"sync_id": "w-cached", "name": "Pre-Cached",
                         "currency": "THB", "wallet_type": "general",
                         "is_default": True, "usage_count": 0}],
            "categories": [],
            "budgets": [],
            "goals": [],
            "tags": [],
            "wallet_id": "w-cached",
            "default_currency_code": "THB",
            "fetched_at": "2026-05-28T00:00:00Z",
        }
        state = {"user_id": "u-1", "user_context": cached}

        cmd, out = await _invoke_ctx(state=state)

        # Cache hit — loader was NOT called.
        assert call_count["n"] == 0, "loader must not run when cache is hot"
        # ToolMessage payload IS the cached dict (same contents — wallet
        # sync_id comes from the cache, not the loader's fresh data).
        assert out["wallet_id"] == "w-cached"
        assert out["wallets"][0]["sync_id"] == "w-cached"
        # Command(update=) carries the same dict so a downstream
        # `propose_transaction` reads it via state.user_context.
        assert cmd.update["user_context"]["wallet_id"] == "w-cached"

    asyncio.run(run())


def test_UT_T05_cold_cache_populates_via_loader():
    """UT-T05 supporting: with no cache, the loader runs exactly once and
    the resulting dict is what the ReAct agent receives. The graph's
    reducer (Wave 4) takes the return value and writes it into state."""
    call_count = {"n": 0}

    async def fake_loader(user_id: str) -> EntityCatalog:
        call_count["n"] += 1
        return _full_catalog()

    set_catalog_loader(fake_loader)

    async def run():
        state = {"user_id": "u-1"}
        cmd, out = await _invoke_ctx(state=state)
        assert call_count["n"] == 1
        assert len(out["wallets"]) == 3
        # Default wallet is the catalog's flagged default (the general one).
        assert out["wallet_id"] == "w-gen"
        # Command propagates the same context for the graph reducer to write.
        assert cmd.update["user_context"]["wallet_id"] == "w-gen"

    asyncio.run(run())


def test_UT_T05_wallets_ordered_general_creditcard_goal():
    """UT-T05 supporting: wallets returned in `user_context` are ordered
    general → creditcard → goal so a downstream index-0 fallback in
    propose_transaction lands on a spending wallet (memory
    `project_wallet_index0_ordering`)."""

    async def fake_loader(user_id: str) -> EntityCatalog:
        # Return wallets in REVERSE order to prove the tool re-sorts them.
        return EntityCatalog(
            wallets=[
                WalletEntry(sync_id="w-goal", name="House", currency="THB",
                            wallet_type="goal"),
                WalletEntry(sync_id="w-cc", name="KCard", currency="THB",
                            wallet_type="creditcard"),
                WalletEntry(sync_id="w-gen", name="Cash", currency="THB",
                            wallet_type="general"),
            ],
        )

    set_catalog_loader(fake_loader)

    async def run():
        state = {"user_id": "u-1"}
        _cmd, ctx = await _invoke_ctx(state=state)
        types = [w["wallet_type"] for w in ctx["wallets"]]
        assert types == ["general", "creditcard", "goal"]

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T05b — retry-once on empty wallets
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T05b_retries_once_on_empty_wallets():
    """UT-T05b: a transient empty read must trigger ONE retry before the
    tool concludes the user is genuinely wallet-less. Memory
    `project_wallet_index0_ordering` documents the v2 flake this guards
    against (false-onboarding for existing users)."""
    attempts = {"n": 0}

    async def flaky_loader(user_id: str) -> EntityCatalog:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _empty_catalog()       # transient empty
        return _full_catalog()             # second call returns real data

    set_catalog_loader(flaky_loader)

    async def run():
        state = {"user_id": "u-1"}
        _cmd, ctx = await _invoke_ctx(state=state)

        assert attempts["n"] == 2, "expected exactly 2 catalog load attempts"
        assert len(ctx["wallets"]) == 3, "second-attempt catalog should populate"

    asyncio.run(run())


def test_UT_T05b_truly_wallet_less_user_still_returns_empty():
    """UT-T05b supporting: when BOTH attempts come back empty, we DO surface
    the empty state — the retry only forgives transient blips, it doesn't
    invent wallets. wallet_required_cta is the LLM's next move."""
    attempts = {"n": 0}

    async def empty_loader(user_id: str) -> EntityCatalog:
        attempts["n"] += 1
        return _empty_catalog()

    set_catalog_loader(empty_loader)

    async def run():
        state = {"user_id": "new-user"}
        cmd, ctx = await _invoke_ctx(state=state)

        assert attempts["n"] == 2
        assert ctx["wallets"] == []
        assert ctx["wallet_id"] is None
        # Empty-but-real context still propagates — `propose_transaction`
        # detects wallets=[] and pivots to wallet_required_cta.
        assert cmd.update["user_context"]["wallets"] == []

    asyncio.run(run())


def test_UT_T05_missing_user_id_returns_error():
    """Without `state.user_id` the tool MUST return a structured error, not
    raise — ReAct relies on tool messages for retry."""

    async def run():
        cmd, out = await _invoke_ctx(state={})
        assert out.get("error") == "state.user_id missing"
        assert out.get("kind") == "invalid_state"
        # No user_context lands in update on error.
        assert "user_context" not in (cmd.update or {})

    asyncio.run(run())
