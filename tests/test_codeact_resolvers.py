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


# ── UT-RS04 — schema hint present in EVERY rerank prompt (no-pick trap guard) ─
@pytest.mark.parametrize("kind", ["wallet", "category", "tag"])
async def test_UT_RS04_every_rerank_prompt_describes_confidence(monkeypatch, kind):
    """UT-RS04 (regression, project_resolver_structured_schema_trap): the
    structured adapter does NOT inject the pydantic schema, so every rerank
    prompt must describe the `confidence` field by hand. Without it the model
    omits the field → ValidationError → swallowed → silent no-pick. 'tag' used
    to lack this hint; this pins all three kinds."""
    import src.agent.llm as llm_mod

    captured: dict[str, str] = {}

    class _FakeStructured:
        async def ainvoke(self, prompt):
            captured["prompt"] = prompt
            return resolvers._LLMChoice(sync_id="x-1", confidence=0.9, reason="ok")

    class _FakeLLM:
        def with_structured_output(self, model):
            return _FakeStructured()

    monkeypatch.setattr(llm_mod, "llm", _FakeLLM())

    cand = [resolvers._Candidate(sync_id="x-1", name="ทดสอบ")]
    await resolvers._llm_rerank(kind, "q", cand)

    assert "confidence" in captured["prompt"], (
        f"{kind} rerank prompt omits the confidence-field description → silent no-pick"
    )
    assert resolvers._CHOICE_SCHEMA_HINT in captured["prompt"]


# ── UT-RS05 — parent expansion across per-wallet duplicate copies ────────────
def _duplicated_catalog():
    """Mimic the real per-wallet category cloning + (name, type) dedupe:

    wallet A and wallet B both own a copy of 'อาหาร' with children. Dedupe
    keeps wallet A's 'อาหาร' but wallet B's 'ร้านอาหาร' — whose parent_id
    points at wallet B's 'อาหาร' copy that dedupe DROPPED. This is the exact
    shape that made a category-scoped sum silently drop transactions
    (970-vs-1290 bug)."""
    a_food = CategoryEntry("a-food", "อาหาร", "expense",
                           wallet_sync_id="w-a")
    a_coffee = CategoryEntry("a-coffee", "กาแฟ", "expense",
                             parent_id="a-food", wallet_sync_id="w-a")
    b_food = CategoryEntry("b-food", "อาหาร", "expense",
                           wallet_sync_id="w-b")
    b_rest = CategoryEntry("b-rest", "ร้านอาหาร", "expense",
                           parent_id="b-food", wallet_sync_id="w-b")
    return EntityCatalog(
        # deduped view — the survivor mix the resolver candidates see:
        # 'ร้านอาหาร' survived from wallet B, its parent copy did not.
        categories=[a_food, a_coffee, b_rest],
        # full per-wallet list — hierarchy is only complete here.
        categories_all=[a_food, a_coffee, b_food, b_rest],
    )


async def test_UT_RS05_expand_covers_children_of_dropped_parent_copies(monkeypatch):
    """UT-RS05: resolve_category on a parent MUST include child names whose
    parent_id points at ANY per-wallet copy of that parent — not only the
    copy that survived dedupe. Pre-fix this returned ['อาหาร', 'กาแฟ'] and
    the SQL name-filter never saw 'ร้านอาหาร' spend."""
    import asyncio

    _patch_llm(monkeypatch, sync_id="a-food")
    loop = asyncio.get_running_loop()
    resolve_category = resolvers.make_resolve_category(_duplicated_catalog(), loop)
    names = await asyncio.to_thread(resolve_category, "อาหาร")
    assert names == ["อาหาร", "กาแฟ", "ร้านอาหาร"]


async def test_UT_RS05b_expand_leaf_is_not_widened(monkeypatch):
    """UT-RS05b: a leaf pick (no copy anywhere has children pointing at it)
    stays `[name]` — the don't-widen-a-leaf contract survives the fix."""
    import asyncio

    _patch_llm(monkeypatch, sync_id="a-coffee")
    loop = asyncio.get_running_loop()
    resolve_category = resolvers.make_resolve_category(_duplicated_catalog(), loop)
    names = await asyncio.to_thread(resolve_category, "กาแฟ")
    assert names == ["กาแฟ"]


def _name_collision_catalog():
    """The credit-card seed (`category_seed_data.dart`) nests 'อาหาร' under
    'ช้อปปิ้ง' while 'อาหาร' is its OWN root category in the general wallets.
    Because category filtering is by NAME, widening 'ช้อปปิ้ง' into 'อาหาร'
    would vacuum the standalone food category's transactions into the shopping
    total (the real 5,205-vs-5,085 contamination).

    'อาหาร' is root-DOMINANT here (2 general roots vs 1 nested under 'ช้อปปิ้ง'),
    mirroring real data (root×7 vs child×2) → the strict majority test drops it.
    A genuine leaf 'เสื้อผ้า' (never a root) survives."""
    g_food = CategoryEntry("g-food", "อาหาร", "expense", wallet_sync_id="w-g")
    g2_food = CategoryEntry("g2-food", "อาหาร", "expense", wallet_sync_id="w-g2")
    cc_shop = CategoryEntry("cc-shop", "ช้อปปิ้ง", "expense", wallet_sync_id="w-cc")
    cc_food = CategoryEntry("cc-food", "อาหาร", "expense",
                            parent_id="cc-shop", wallet_sync_id="w-cc")
    cc_clothes = CategoryEntry("cc-clothes", "เสื้อผ้า", "expense",
                               parent_id="cc-shop", wallet_sync_id="w-cc")
    return EntityCatalog(
        categories=[g_food, cc_shop, cc_clothes],
        categories_all=[g_food, g2_food, cc_shop, cc_food, cc_clothes],
    )


def _legit_nesting_catalog():
    """Mirror 'น้ำมัน': child-DOMINANT (2 copies under 'เดินทาง') with a lone
    stray root copy. The strict majority test must KEEP it — transport queries
    still include fuel. This is the regression guard for the RS05c fix."""
    g_transport = CategoryEntry("g-tx", "เดินทาง", "expense", wallet_sync_id="w-g")
    cc_transport = CategoryEntry("cc-tx", "เดินทาง", "expense", wallet_sync_id="w-cc")
    g_fuel = CategoryEntry("g-fuel", "น้ำมัน", "expense",
                           parent_id="g-tx", wallet_sync_id="w-g")
    cc_fuel = CategoryEntry("cc-fuel", "น้ำมัน", "expense",
                            parent_id="cc-tx", wallet_sync_id="w-cc")
    stray_fuel_root = CategoryEntry("x-fuel", "น้ำมัน", "expense", wallet_sync_id="w-x")
    return EntityCatalog(
        categories=[g_transport, g_fuel],
        categories_all=[g_transport, cc_transport, g_fuel, cc_fuel, stray_fuel_root],
    )


async def test_UT_RS05c_expand_drops_name_colliding_root_child(monkeypatch):
    """UT-RS05c: a child whose name is PREDOMINANTLY a top-level category is
    DROPPED from parent expansion. Widening 'ช้อปปิ้ง' must NOT include 'อาหาร'
    (it would name-match the standalone food category's spend), but the genuine
    leaf 'เสื้อผ้า' survives."""
    import asyncio

    _patch_llm(monkeypatch, sync_id="cc-shop")
    loop = asyncio.get_running_loop()
    resolve_category = resolvers.make_resolve_category(_name_collision_catalog(), loop)
    names = await asyncio.to_thread(resolve_category, "ช้อปปิ้ง")
    assert names == ["ช้อปปิ้ง", "เสื้อผ้า"]  # NOT "อาหาร"


async def test_UT_RS05d_expand_keeps_legitimately_nested_child(monkeypatch):
    """UT-RS05d: a child that is PREDOMINANTLY nested (more child-of-parent than
    root copies, like 'น้ำมัน' under 'เดินทาง') is KEPT despite a stray root copy.
    Regression guard so the RS05c fix doesn't strip fuel from transport sums."""
    import asyncio

    _patch_llm(monkeypatch, sync_id="g-tx")
    loop = asyncio.get_running_loop()
    resolve_category = resolvers.make_resolve_category(_legit_nesting_catalog(), loop)
    names = await asyncio.to_thread(resolve_category, "เดินทาง")
    assert names == ["เดินทาง", "น้ำมัน"]  # fuel KEPT
