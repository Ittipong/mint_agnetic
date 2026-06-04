"""Unit tests for src.agent.tools.codeact.resolvers.

Covers UT-RS01..03 — lifted from v2 `test_resolvers_llm_only.py` +
`test_add_category_resolver.py` with import paths adjusted for the v3
module layout.

Resolver design (FROZEN per memory `project_codeact_dual_impl`):
  - LLM-only — no fuzzy / string-similarity fast-path.
  - sandbox (sync) path bridges via run_coroutine_threadsafe.
  - ADD-path (async) coroutines: resolve_wallet_choice_async,
    resolve_category_for_add_async (wallet+type scoped, top-1 no-threshold).
"""

from __future__ import annotations

import pytest

from src.agent.tools.codeact import resolvers
from src.agent.tools.codeact.exceptions import AmbiguousMatchError
from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    TagEntry,
    WalletEntry,
)


pytestmark = pytest.mark.asyncio


def _catalog():
    return EntityCatalog(
        wallets=[
            WalletEntry(sync_id="w-true", name="TrueMonney", currency="THB"),
            WalletEntry(sync_id="w-kbank", name="KBank Savings", currency="THB"),
        ],
        categories=[
            CategoryEntry(sync_id="c-food", name="อาหาร", type="expense"),
            CategoryEntry(sync_id="c-travel", name="เดินทาง", type="expense"),
            CategoryEntry(sync_id="c-taxi", name="แท็กซี่", type="expense",
                          parent_id="c-travel"),
        ],
        tags=[
            TagEntry(sync_id="t-work", name="งาน"),
            TagEntry(sync_id="t-trip", name="เที่ยว"),
        ],
    )


def _patch_llm(monkeypatch, *, sync_id, confidence=0.95):
    """Patch `_llm_rerank` so resolution is deterministic and never touches
    the network. Returns the chosen candidate from the supplied list."""

    async def fake_rerank(kind, query, candidates):
        chosen = next((c for c in candidates if c.sync_id == sync_id), None)
        if chosen is None:
            return None
        return chosen, confidence

    monkeypatch.setattr(resolvers, "_llm_rerank", fake_rerank)


def _patch_llm_nopick(monkeypatch):
    async def fake_rerank(kind, query, candidates):
        return None

    monkeypatch.setattr(resolvers, "_llm_rerank", fake_rerank)


# ── UT-RS01 — wallet resolver (async) ──────────────────────────────────────
async def test_UT_RS01_wallet_llm_pick_returns_candidate(monkeypatch):
    """UT-RS01: cross-language wallet resolution via the LLM. 'true money'
    must surface the catalog entry 'TrueMonney' with its sync_id intact —
    callers index by sync_id, not by name."""
    _patch_llm(monkeypatch, sync_id="w-true")
    chosen = await resolvers.resolve_wallet_choice_async(_catalog(), "true money")
    assert chosen.sync_id == "w-true"
    assert chosen.name == "TrueMonney"


async def test_UT_RS01b_wallet_no_pick_raises_with_available(monkeypatch):
    """No-pick MUST raise ValueError naming the available candidates so the
    LLM can read the error and pick from the listed set on retry — silent
    None would loop forever."""
    _patch_llm_nopick(monkeypatch)
    with pytest.raises(ValueError) as ei:
        await resolvers.resolve_wallet_choice_async(_catalog(), "xxx")
    msg = str(ei.value)
    assert "no wallet matches 'xxx'" in msg
    assert "Available" in msg
    assert "TrueMonney" in msg


# ── UT-RS02 — ADD-path category resolver (wallet + type scoped) ─────────────
def _scoped_catalog():
    """One wallet (w-cardx) with same-named expense+income 'อื่นๆ', a second
    wallet's category, and a global (wallet None) category — exercise the
    ADD resolver's wallet + type scoping (memory: ADDR category for ADD)."""
    cats = [
        CategoryEntry("c-other-exp", "อื่นๆ", "expense", wallet_sync_id="w-cardx"),
        CategoryEntry("c-other-inc", "อื่นๆ", "income", wallet_sync_id="w-cardx"),
        CategoryEntry("c-shop", "ช้อปปิ้ง", "expense", wallet_sync_id="w-cardx"),
        CategoryEntry("c-furn", "เฟอร์นิเจอร์", "expense", wallet_sync_id="w-test"),
        CategoryEntry("c-global", "ทั่วไป", "expense", wallet_sync_id=None),
    ]
    return EntityCatalog(categories=cats, categories_all=cats)


def _patch_llm_capture(monkeypatch, *, pick_sync_id, confidence=0.9):
    captured: dict = {}

    async def fake_rerank(kind, query, candidates):
        captured["candidates"] = list(candidates)
        chosen = next((c for c in candidates if c.sync_id == pick_sync_id), None)
        return (chosen, confidence) if chosen else None

    monkeypatch.setattr(resolvers, "_llm_rerank", fake_rerank)
    return captured


async def test_UT_RS02_add_resolver_scopes_by_wallet_and_type(monkeypatch):
    """UT-RS02: the ADD resolver only offers candidates of the SAME wallet
    (+ global) AND the SAME txn_type — an EXPENSE never sees the income
    'อื่นๆ' of the same wallet, nor another wallet's category.

    This is the Other-floor sibling: when the LLM picks a wrong-type
    candidate the answer can mislabel a slip's discount (memory
    `project_chat_null_category_other_floor`). Scoping prevents that."""
    cap = _patch_llm_capture(monkeypatch, pick_sync_id="c-other-exp")
    chosen = await resolvers.resolve_category_for_add_async(
        _scoped_catalog(), "เก้าอี้", "w-cardx", "expense")
    ids = {c.sync_id for c in cap["candidates"]}
    assert "c-other-inc" not in ids   # wrong type (income) excluded
    assert "c-furn" not in ids        # other wallet excluded
    assert {"c-other-exp", "c-shop", "c-global"} <= ids
    assert chosen.sync_id == "c-other-exp"


async def test_UT_RS02b_add_resolver_returns_none_on_empty_candidates(monkeypatch):
    """When no candidate matches the txn_type, the ADD resolver MUST short-
    circuit to None without an LLM call — the caller writes back
    `category_sync_id=None` and the propose_transaction Other-floor takes
    over (memory `project_chat_null_category_other_floor`)."""
    called = {"n": 0}

    async def fake_rerank(kind, query, candidates):
        called["n"] += 1
        return None

    monkeypatch.setattr(resolvers, "_llm_rerank", fake_rerank)
    cat = EntityCatalog(
        categories=[CategoryEntry(
            "e", "อาหาร", "expense", wallet_sync_id="w-cardx",
        )],
        categories_all=[CategoryEntry(
            "e", "อาหาร", "expense", wallet_sync_id="w-cardx",
        )],
    )
    chosen = await resolvers.resolve_category_for_add_async(
        cat, "เงินเดือน", "w-cardx", "income")
    assert chosen is None
    assert called["n"] == 0  # filtered to empty → never hit the LLM


# ── UT-RS03 — tag resolver (sandbox closure) + budget/goal echo ─────────────
async def test_UT_RS03_tag_closure_returns_canonical_name(monkeypatch):
    """UT-RS03: the sandbox tag resolver (sync closure) bridges to the LLM
    via the running loop and returns the canonical tag name. We drive it on
    a worker thread to mimic the sandbox's calling pattern."""
    import asyncio

    _patch_llm(monkeypatch, sync_id="t-trip")
    loop = asyncio.get_running_loop()
    resolve_tag = resolvers.make_resolve_tag(_catalog(), loop)
    name = await asyncio.to_thread(resolve_tag, "vacation")
    assert name == "เที่ยว"


async def test_UT_RS03b_budget_resolver_echoes_query():
    """Budgets are SQL ILIKE'd at the template level — the resolver only
    validates that the phrase is non-empty, then echoes it back. Keeps
    API symmetry with wallet/category resolvers."""
    import asyncio

    loop = asyncio.get_running_loop()
    resolve_budget = resolvers.make_resolve_budget(EntityCatalog(), loop)
    assert resolve_budget("งบอาหาร") == "งบอาหาร"
    with pytest.raises(ValueError):
        resolve_budget("")


async def test_UT_RS03c_low_confidence_pick_raises_ambiguous(monkeypatch):
    """A confident-but-still-uncertain LLM choice (conf < 0.5) raises
    AmbiguousMatchError so the LLM commits to a candidate on the next
    iteration instead of silently picking a guess."""
    _patch_llm(monkeypatch, sync_id="c-food", confidence=0.3)
    with pytest.raises(AmbiguousMatchError):
        await resolvers.resolve_category_choice_async(_catalog(), "??")
