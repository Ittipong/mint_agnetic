"""Onboarding stage machine — Wave 7b lift (compute_stage + augment_blocks).

Lifted from v2 `tests/test_onboarding.py`. The schema/helper subset
(UT-ON101..115) maps directly onto v3's surviving `src.agent.onboarding`
module (compute_stage / augment_blocks / STAGE_* constants are identical).

v2-only cases NOT lifted:
- UT-ON121..123 finalize integration — `agent.nodes.finalize.finalize` is
  removed in v3 (no node graph); the augment_blocks contract is asserted at
  the helper layer instead.
- UT-ON131..134 onboarding_hint prompt content — `onboarding_hint` removed
  in v3 (prompt structure changed for the ReAct loop).
- UT-ON132 CODEACT_SYSTEM content — `CODEACT_SYSTEM` removed (prompt now
  built per-turn in `prompts.py`).
- UT-WS001 wallet_id scoping — `CodeActNode` removed; equivalent assertion
  for v3 lives in `test_get_user_context.py` (wallet flows through
  user_context cache, never through the analyst tool's args).

The lifted subset uses `src.agent.entity_catalog.WalletEntry` (same shape
as v2).
"""

from __future__ import annotations

from src.agent.entity_catalog import WalletEntry
from src.agent.onboarding import (
    STAGE_DONE,
    STAGE_NO_TRANSACTION,
    STAGE_NO_WALLET,
    augment_blocks,
    compute_stage,
)


def _w(wallet_type: str = "general", usage_count: int = 0) -> WalletEntry:
    return WalletEntry(sync_id="s", name="n", currency="THB",
                       wallet_type=wallet_type, usage_count=usage_count)


class _Catalog:
    def __init__(self, wallets):
        self.wallets = wallets


# ---------------------------------------------------------------------------
# compute_stage
# ---------------------------------------------------------------------------
def test_UT_ON101_no_wallet_when_no_general_wallet():
    """Only a credit-card wallet -> still no_wallet (must create a general one)."""
    cat = _Catalog([_w(wallet_type="creditcard")])
    assert compute_stage(cat, has_pool=True) == STAGE_NO_WALLET
    assert compute_stage(_Catalog([]), has_pool=True) == STAGE_NO_WALLET


def test_UT_ON102_no_transaction_when_wallet_but_zero_txn():
    cat = _Catalog([_w(wallet_type="general", usage_count=0)])
    assert compute_stage(cat, has_pool=True) == STAGE_NO_TRANSACTION


def test_UT_ON103_done_when_general_wallet_has_transactions():
    cat = _Catalog([_w(wallet_type="general", usage_count=3)])
    assert compute_stage(cat, has_pool=True) == STAGE_DONE


def test_UT_ON104_no_pool_disables_onboarding():
    """Tests (pool=None) must never be flagged as onboarding."""
    assert compute_stage(_Catalog([]), has_pool=False) == STAGE_DONE


# ---------------------------------------------------------------------------
# augment_blocks per stage
# ---------------------------------------------------------------------------
def test_UT_ON111_no_wallet_appends_cta_only():
    """With no wallet the only meaningful action is "create wallet": emit the
    CTA and NO suggestion chips (chips would just distract / duplicate it)."""
    out = augment_blocks([], stage=STAGE_NO_WALLET)
    types = [b["type"] for b in out]
    assert types == ["wallet_required"], types


def test_UT_ON112_no_wallet_skips_duplicate_cta():
    """A worker already emitted the CTA -> nothing extra to add (no chips
    either)."""
    existing = [{"type": "wallet_required", "text": "x", "action": {}}]
    out = augment_blocks(existing, stage=STAGE_NO_WALLET)
    assert out == [], out


def test_UT_ON113_no_transaction_chips_have_manual_deeplink():
    out = augment_blocks([], stage=STAGE_NO_TRANSACTION)
    assert [b["type"] for b in out] == ["suggestions"], out
    items = out[0]["items"]
    assert any(it.get("deeplink") for it in items), items   # "กรอกเอง" chip
    assert any(it.get("send") for it in items), items       # text chip


def test_UT_ON114_done_chips_only_on_just_completed():
    assert augment_blocks([], stage=STAGE_DONE, just_completed=False) == []
    out = augment_blocks([], stage=STAGE_DONE, just_completed=True)
    assert [b["type"] for b in out] == ["suggestions"], out


def test_UT_ON115_chip_schema_label_plus_send_xor_deeplink():
    """no_wallet emits no chips, so schema check covers no_transaction (and
    any other chip-bearing stage) only."""
    for stage in (STAGE_NO_TRANSACTION,):
        out = augment_blocks([], stage=stage)
        items = next(b for b in out if b["type"] == "suggestions")["items"]
        for it in items:
            assert it.get("label"), it
            assert bool(it.get("send")) ^ bool(it.get("deeplink")), it
