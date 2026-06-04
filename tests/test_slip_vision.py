"""Unit tests for the slip-vision extraction pipeline (Wave 7b adaptation).

Adapted from v2 `tests/test_slip_vision.py`. v3 collapsed the slip vision into
`src.agent.endpoints.slip_handler.extract_slip` (analog of v2's
`run_slip_vision`). These tests focus on the vision extraction semantics
deeper than `test_slip_handler.py` (Wave 5) already covers:

- UT-SLIP-01: render_for_slip scoping (catalog method, untouched in v3)
- UT-SLIP-02: _pick_slip_wallet fallback semantics
- UT-SLIP-03: _to_data_uri wrap + passthrough
- UT-SLIP-12: discount line preserves include_in_report=false
- UT-SLIP-21: unresolvable category -> Other floor (same direction)
- UT-SLIP-26: null income category -> income Other (not first income cat)
- UT-SLIP-27: no Other category -> degrades to None (mobile picker)
- UT-SLIP-30: prompt forbids null + maps discount to "เงินคืน/Refund"
- UT-SLIP-22: zero/missing amount lines skipped
- UT-SLIP-23: readable=true but empty -> unreadable

Overlap with existing v3 tests:
- UT-SLIP-10 / UT-SLIP-11 single/multi payload -> `test_slip_handler.py` UT-SL01
- UT-SLIP-20 unreadable readable=false -> `test_slip_handler.py` UT-SL01b
- UT-SLIP-24 / UT-SLIP-25 dev vs prod JSON behavior -> v3 ALWAYS degrades
  (Postel's-law gating not surfaced in extract_slip), so those two cases are
  NOT lifted — they would test a v2-only contract.

The vision LLM is mocked (an async callable returning a canned JSON string)
so these are deterministic and offline — the real-model path is covered by
`tests_integration/`.
"""

from __future__ import annotations

import json

import pytest

from src.agent.endpoints.slip_handler import (
    SLIP_UNREADABLE_MESSAGE,
    _build_slip_prompt,
    _pick_slip_wallet,
    _to_data_uri,
    extract_slip,
)
from src.agent.entity_catalog import CategoryEntry, EntityCatalog, WalletEntry


# ---------------------------------------------------------------------------
# Fixtures — a hand-built catalog: 1 general wallet "w1" + 1 creditcard "w2",
# w1 has expense (food/coffee) + income (refund) categories, plus GLOBAL
# expense + income "อื่นๆ"/Other catch-alls (the never-null fallback floor).
# ---------------------------------------------------------------------------
def _catalog() -> EntityCatalog:
    wallets = [
        WalletEntry(sync_id="w1", name="เงินสด", currency="THB", wallet_type="general"),
        WalletEntry(sync_id="w2", name="บัตรเครดิต", currency="THB",
                    wallet_type="creditcard"),
    ]
    cats_all = [
        CategoryEntry(sync_id="c_food", name="อาหาร", type="expense",
                      wallet_sync_id="w1"),
        CategoryEntry(sync_id="c_coffee", name="กาแฟ", type="expense",
                      wallet_sync_id="w1"),
        CategoryEntry(sync_id="c_refund", name="เงินคืน", type="income",
                      wallet_sync_id="w1"),
        CategoryEntry(sync_id="c_w2", name="ผ่อนชำระ", type="expense",
                      wallet_sync_id="w2"),
        CategoryEntry(sync_id="c_other_exp", name="อื่นๆ", type="expense",
                      wallet_sync_id=None),
        CategoryEntry(sync_id="c_other_inc", name="อื่นๆ", type="income",
                      wallet_sync_id=None),
    ]
    return EntityCatalog(wallets=wallets, categories=cats_all,
                         categories_all=cats_all)


def _vision(payload: dict):
    """Build an async vision_call returning `payload` as a JSON string."""
    async def call(_messages):
        return json.dumps(payload, ensure_ascii=False)
    return call


W1 = "w1"


# ---------------------------------------------------------------------------
# Catalog slip helpers (pure functions — no LLM)
# ---------------------------------------------------------------------------
def test_ut_slip_01_render_scopes_to_wallet_plus_global():
    """UT-SLIP-01: render_for_slip shows ONLY the target wallet's categories
    (+ global), grouped expense/income, with verbatim sync_ids — never another
    wallet's category."""
    out = _catalog().render_for_slip(W1)
    assert "c_food" in out and "c_coffee" in out and "c_refund" in out
    assert "c_other_exp" in out            # global is shared
    assert "c_w2" not in out               # other wallet's category excluded
    assert "Expense categories:" in out and "Income categories:" in out


def test_ut_slip_02_pick_slip_wallet_honors_pick_then_falls_back():
    """UT-SLIP-02: _pick_slip_wallet returns the explicit pick when valid;
    otherwise the first general wallet (not the creditcard); None when no
    wallets exist."""
    cat = _catalog()
    assert _pick_slip_wallet(cat, "w2").sync_id == "w2"        # explicit pick
    assert _pick_slip_wallet(cat, None).sync_id == "w1"        # fallback general
    assert _pick_slip_wallet(cat, "nope").sync_id == "w1"      # invalid -> general
    assert _pick_slip_wallet(EntityCatalog(), None) is None    # no wallets


def test_ut_slip_03_data_uri_wrap_and_passthrough():
    """UT-SLIP-03: raw base64 is wrapped as a jpeg data URI; an existing data
    URI is passed through unchanged."""
    assert _to_data_uri("AAAA").startswith("data:image/jpeg;base64,AAAA")
    assert _to_data_uri("data:image/png;base64,XYZ") == "data:image/png;base64,XYZ"


# ---------------------------------------------------------------------------
# extract_slip — payload-shape regression cases
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ut_slip_12_discount_line_preserves_include_in_report_false():
    """UT-SLIP-12: a discount/income line keeps include_in_report=false so it
    doesn't inflate income reports."""
    cat = _catalog()
    vc = _vision({"readable": True, "transactions": [
        {"type": "expense", "amount": 500, "category_sync_id": "c_food",
         "note": "ของชำ"},
        {"type": "income", "amount": 50, "category_sync_id": "c_refund",
         "note": "ส่วนลด", "include_in_report": False},
    ]})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    assert res.payloads[0]["include_in_report"] is True
    assert res.payloads[1]["include_in_report"] is False
    assert res.payloads[1]["type"] == "income"


@pytest.mark.asyncio
async def test_ut_slip_21_unresolvable_category_falls_back_to_other_same_direction():
    """UT-SLIP-21: a category_sync_id NOT in the wallet's catalog (hallucinated
    or cross-wallet) never leaks another wallet's sync_id AND is never left null
    — it falls back to the wallet's "อื่นๆ"/Other category of the SAME
    direction so the proposal always carries a sane category."""
    cat = _catalog()
    vc = _vision({"readable": True, "transactions": [
        # belongs to w2 (out of this wallet's scope) -> must NOT leak
        {"type": "expense", "amount": 99, "category_sync_id": "c_w2", "note": "x"},
        # nonexistent id -> falls to Other-expense
        {"type": "expense", "amount": 10, "category_sync_id": "ghost", "note": "y"},
    ]})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    assert all(p["category_sync_id"] == "c_other_exp" for p in res.payloads)
    assert all(p["category"] == "อื่นๆ" for p in res.payloads)
    assert all(p["category_sync_id"] != "c_w2" for p in res.payloads)


@pytest.mark.asyncio
async def test_ut_slip_26_null_income_category_falls_back_to_income_other_not_first():
    """UT-SLIP-26 (regression — discount mislabelled as เงินเดือน): a discount /
    income line the model couldn't categorise (`category_sync_id: null`) must
    resolve to the INCOME "อื่นๆ"/Other category — never be left null (which the
    mobile card silently turns into the first income category "เงินเดือน")."""
    cat = _catalog()
    vc = _vision({"readable": True, "transactions": [
        {"type": "expense", "amount": 12500, "category_sync_id": "c_coffee",
         "note": "เมล็ดกาแฟ"},
        {"type": "income", "amount": 625, "category_sync_id": None,
         "note": "ส่วนลด", "include_in_report": False},
    ]})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    discount = res.payloads[1]
    assert discount["type"] == "income"
    assert discount["category_sync_id"] == "c_other_inc"   # income Other
    assert discount["category"] == "อื่นๆ"
    assert discount["category_sync_id"] is not None
    assert discount["include_in_report"] is False


@pytest.mark.asyncio
async def test_ut_slip_27_no_other_category_degrades_to_none():
    """UT-SLIP-27: the fallback is best-effort — if the wallet genuinely has NO
    "อื่นๆ"/Other category of that direction, the line degrades to None (the
    card lets the user pick) rather than borrowing a wrong category."""
    wallets = [
        WalletEntry(sync_id="w1", name="เงินสด", currency="THB",
                    wallet_type="general"),
    ]
    cats = [
        CategoryEntry(sync_id="c_food", name="อาหาร", type="expense",
                      wallet_sync_id="w1"),
    ]
    cat = EntityCatalog(wallets=wallets, categories=cats, categories_all=cats)
    vc = _vision({"readable": True, "transactions": [
        {"type": "income", "amount": 50, "category_sync_id": None, "note": "ส่วนลด"},
    ]})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    assert res.payloads[0]["category_sync_id"] is None
    assert res.payloads[0]["category"] == "other"


def test_ut_slip_30_prompt_forbids_null_and_maps_discount_to_refund():
    """UT-SLIP-30: the slip prompt must (a) tell the model a discount line maps
    to the "เงินคืน/Refund" income category and (b) forbid `null` — every line
    carries a category_sync_id (Other as the last resort)."""
    prompt = _build_slip_prompt(
        current_date="2026-05-27",
        wallet_name="กระเป๋าหลัก",
        wallet_currency="THB",
        category_map="(categories)",
        default_currency_code="THB",
        default_currency_symbol="฿",
    )
    assert "เงินคืน/Refund" in prompt
    assert "<id-from-list-or-null>" not in prompt
    assert "use `null`" not in prompt and "use null" not in prompt


@pytest.mark.asyncio
async def test_ut_slip_22_zero_and_missing_amount_lines_skipped():
    """UT-SLIP-22: lines with no/zero/negative amount are skipped; if NOTHING
    usable remains, the turn degrades to unreadable."""
    cat = _catalog()
    vc = _vision({"readable": True, "transactions": [
        {"type": "expense", "amount": 0, "category_sync_id": "c_food",
         "note": "free"},
        {"type": "expense", "category_sync_id": "c_food", "note": "no amount"},
        {"type": "expense", "amount": 80, "category_sync_id": "c_food",
         "note": "ของจริง"},
    ]})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    assert len(res.payloads) == 1 and res.payloads[0]["amount"] == 80


@pytest.mark.asyncio
async def test_ut_slip_23_readable_true_but_empty_is_unreadable():
    """UT-SLIP-23: readable=true with no usable transactions -> unreadable."""
    cat = _catalog()
    vc = _vision({"readable": True, "transactions": []})
    res = await extract_slip(
        vc, user_id="u1", image_b64="AAAA", catalog=cat,
        wallet=_pick_slip_wallet(cat, W1),
        default_currency_code="THB", default_currency_symbol="฿",
    )
    assert res.readable is False
    assert res.message == SLIP_UNREADABLE_MESSAGE
