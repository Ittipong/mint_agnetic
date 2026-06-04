"""Wallet-required CTA path (Wave 7b adaptation, repurposed from v2).

Source: `mint_agentic_v2/tests/test_wallet_clarification_block.py`. Q6 in v3
dropped the wallet picker (`option_kind: "wallet"` clarification), so the
original UT-WC tests asserting "no wallet picker emitted" against an
UnderstandNode/TransactionsNode chain no longer apply (those nodes are gone).

This file REPURPOSES the same behaviour assertions against v3's two-tool
contract: `propose_transaction` + `wallet_required_cta`. The ReAct loop is
expected to call them in sequence, but at unit level we drive each tool
directly and assert the wire shape the ReAct chain depends on.

Cases:
- UT-WRC01: zero-wallet user -> propose_transaction returns kind="no_wallet"
  pointing at `wallet_required_cta`, NO transaction_proposal block emitted.
- UT-WRC02: zero-wallet path then `wallet_required_cta` invoked -> emits the
  `wallet_required` block with action.target = "/wallets/create".
- UT-WRC03: wallets present, no default, no last-used -> propose_transaction
  falls back to wallet INDEX 0 (general before creditcard before goal),
  emits transaction_proposal carrying that wallet's sync_id, NEVER a
  wallet-picker clarification (Q6: picker dropped).
- UT-WRC04: category clarification is unrelated to wallet path — propose_
  transaction never asks for a wallet, just emits transaction_proposal once
  the wallet cascade completes.

Note: UT-WC003 (transient empty catalog retry) is covered by
`test_get_user_context.py::test_UT_T05b_retries_once_on_empty_wallets` —
the retry now lives in `get_user_context`, not the transactions node. Not
duplicated here.
"""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest

from src.agent.tools.propose_transaction import propose_transaction
from src.agent.tools.wallet_required_cta import wallet_required_cta

# Import the module (not the @tool object) so monkeypatch.setattr can swap
# out `_resolve_pair_async` where the tool reads it.
_pt_module = importlib.import_module("src.agent.tools.propose_transaction")
_PairChoice = _pt_module._PairChoice


@pytest.fixture(autouse=True)
def _stub_pair_resolver(monkeypatch):
    """Default joint-LLM stub for the wallet-cascade tests — picks the
    first candidate wallet and any cat whose name exact-matches the hint.
    These tests focus on the wallet RESOLUTION cascade (default → index-0),
    not LLM mechanics, so a deterministic stub is sufficient.
    """

    async def fake_resolver(
        *, candidate_wallets, candidate_cats_by_wallet,
        category_hint, description, **_kwargs,
    ):
        if not candidate_wallets:
            return None
        wallet = candidate_wallets[0]
        hint = (category_hint or description or "").strip().lower()
        cat_id = None
        if hint:
            for c in candidate_cats_by_wallet.get(wallet.sync_id, []):
                if c.name.lower() == hint:
                    cat_id = c.sync_id
                    break
        return _PairChoice(
            wallet_sync_id=wallet.sync_id,
            category_sync_id=cat_id,
            confidence=1.0,
            reason="stub",
        )

    monkeypatch.setattr(_pt_module, "_resolve_pair_async", fake_resolver)
    yield


async def _invoke(tool, args: dict, tool_call_id: str = "tc-1"):
    """Drive a tool via the ToolCall protocol; return (cmd, parsed_payload)."""
    tool_call = {
        "name": tool.name,
        "args": args,
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await tool.ainvoke(tool_call)
    update = cmd.update or {}
    state = args.get("state")
    if isinstance(state, dict):
        if "emitted_blocks_this_turn" in update:
            state.setdefault("emitted_blocks_this_turn", []).extend(
                update["emitted_blocks_this_turn"]
            )
        for k, v in update.items():
            if k in ("emitted_blocks_this_turn", "messages"):
                continue
            state[k] = v
    messages = update.get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content)
    return cmd, {}


def _wallet(sync_id: str, *, name: str = "Cash", wallet_type: str = "general",
            is_default: bool = False, currency: str = "THB") -> dict:
    """Build a wallet dict in the get_user_context wire shape."""
    return {
        "sync_id": sync_id,
        "name": name,
        "currency": currency,
        "wallet_type": wallet_type,
        "is_default": is_default,
        "usage_count": 0,
    }


def _category(sync_id: str, *, name: str = "อาหาร",
              ctype: str = "expense") -> dict:
    return {
        "sync_id": sync_id,
        "name": name,
        "type": ctype,
        "parent_id": None,
        "usage_count": 0,
    }


def _context(wallets: list[dict], categories: list[dict] | None = None,
             default: str | None = None) -> dict:
    return {
        "wallets": wallets,
        "categories": categories or [],
        "tags": [],
        "wallet_id": default,
        "default_currency_code": "THB",
        "fetched_at": "2026-05-28T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# UT-WRC01 — zero wallets => inline wallet_required CTA, no proposal
# ---------------------------------------------------------------------------
def test_UT_WRC01_zero_wallets_emits_wallet_required_block_inline():
    """A user with zero wallets MUST NOT receive a transaction_proposal block.

    Option-4 rewrite: instead of returning a `no_wallet` error and forcing a
    second LLM round through `wallet_required_cta`, propose_transaction now
    emits the wallet_required CTA block IN-PLACE so onboarding finishes in
    the same LLM round. The payload carries kind='no_wallet' + cta_emitted
    so the LLM knows the CTA was already handled and just needs to reply
    with a short confirmation.
    """

    async def run():
        state = {
            "user_id": "u1",
            "user_context": _context(wallets=[]),
            "proposals": [],
            "emitted_blocks_this_turn": [],
            "tool_outputs_this_turn": [],
        }
        cmd, out = await _invoke(propose_transaction, {
            "amount": 50,
            "type": "expense",
            "description": "กาแฟ",
            "state": state,
        })

        # New shape: structured CTA payload, no `error` key.
        assert out["cta_emitted"] is True
        assert out["kind"] == "no_wallet"

        # The wallet_required CTA block landed via Command(update=...).
        update_blocks = (cmd.update or {}).get("emitted_blocks_this_turn") or []
        assert any(b.get("type") == "wallet_required" for b in update_blocks)
        # No transaction_proposal block was emitted by this tool.
        assert not any(b.get("type") == "transaction_proposal"
                       for b in update_blocks)
        # Deep-link target matches the dedicated wallet_required_cta tool.
        wreq = next(b for b in update_blocks if b["type"] == "wallet_required")
        assert wreq["action"]["target"] == "/wallets/create"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-WRC02 — wallet_required_cta tool emits the block when called next
# ---------------------------------------------------------------------------
def test_UT_WRC02_wallet_required_cta_emits_block_with_deep_link():
    """After propose_transaction signals no_wallet, the ReAct loop calls
    wallet_required_cta which emits exactly one `wallet_required` block with
    action.target = '/wallets/create' (mobile deep-link contract). The block's
    text mentions wallets in Thai."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, out = await _invoke(wallet_required_cta, {"state": state})

        assert out["cta_emitted"] is True

        blocks = cmd.update["emitted_blocks_this_turn"]
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "wallet_required"
        assert "กระเป๋า" in block["text"]
        assert block["action"]["target"] == "/wallets/create"
        assert block["action"]["label"]   # any non-empty default

    asyncio.run(run())


class _FakeRepo:
    """Stand-in for ProposalRepo — just records insert_pending_proposal."""

    def __init__(self):
        self.inserts: list[dict] = []

    async def insert_pending_proposal(self, *, proposal_id, user_id, kind,
                                      payload):
        self.inserts.append({"proposal_id": proposal_id, "user_id": user_id,
                             "kind": kind, "payload": payload})
        return proposal_id


# ---------------------------------------------------------------------------
# UT-WRC03 — wallets present, no picker, index-0 fallback
# ---------------------------------------------------------------------------
def test_UT_WRC03_no_default_no_pick_falls_back_to_index_0_no_picker():
    """When wallets exist but neither caller-supplied wallet_sync_id nor
    wallet_id are set, propose_transaction falls back to wallet
    INDEX 0 and emits a transaction_proposal bound to that wallet — NEVER a
    `clarification` block with `option_kind: wallet` (Q6: picker dropped)."""

    async def run():
        wallets = [
            # general comes first (index-0 fallback target)
            _wallet("w-cash", name="Cash"),
            _wallet("w-cc", name="Credit Card", wallet_type="creditcard"),
        ]
        state = {
            "user_id": "u1",
            "user_context": _context(
                wallets=wallets,
                categories=[_category("c-food")],
            ),
            "proposals": [],
            "pending_proposal": None,
            "emitted_blocks_this_turn": [],
            "tool_outputs_this_turn": [],
            "__repo__": _FakeRepo(),
        }
        cmd, out = await _invoke(propose_transaction, {
            "amount": 50,
            "type": "expense",
            "note": "กาแฟ",
            "category_label": "อาหาร",
            "state": state,
        })

        # Success payload has no "error" key.
        assert "error" not in out
        update_blocks = state.get("emitted_blocks_this_turn") or []

        # NO wallet-picker clarification anywhere.
        assert not any(
            b.get("type") == "clarification" and b.get("option_kind") == "wallet"
            for b in update_blocks
        )

        # Exactly one transaction_proposal bound to index-0 (general).
        proposals = [b for b in update_blocks
                     if b.get("type") == "transaction_proposal"]
        assert len(proposals) == 1
        txn = proposals[0]["transaction"]
        assert txn["wallet_sync_id"] == "w-cash"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-WRC04 — explicit wallet_id is honored over index-0
# ---------------------------------------------------------------------------
def test_UT_WRC04_wallet_id_honored_when_set():
    """If `wallet_id` points at a non-index-0 wallet, that
    explicit pick wins — index-0 is the FALLBACK, not an override. (Mirror of
    v2 cascade behavior.)"""

    async def run():
        wallets = [
            _wallet("w-cash", name="Cash"),
            _wallet("w-cc", name="Credit Card", wallet_type="creditcard",
                    is_default=True),
        ]
        state = {
            "user_id": "u1",
            "user_context": _context(
                wallets=wallets,
                categories=[_category("c-food")],
                default="w-cc",
            ),
            "proposals": [],
            "pending_proposal": None,
            "emitted_blocks_this_turn": [],
            "tool_outputs_this_turn": [],
            "__repo__": _FakeRepo(),
        }
        cmd, out = await _invoke(propose_transaction, {
            "amount": 100,
            "type": "expense",
            "note": "ค่าน้ำมัน",
            "category_label": "อาหาร",
            "state": state,
        })

        assert "error" not in out
        update_blocks = state.get("emitted_blocks_this_turn") or []
        prop = next(b for b in update_blocks
                    if b["type"] == "transaction_proposal")
        # default beats index-0 — this is the cascade's whole point.
        assert prop["transaction"]["wallet_sync_id"] == "w-cc"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-WRC05 — client's chat-input wallet pick (state["wallet_id"]) beats the
# catalog default (user_context["wallet_id"])
# ---------------------------------------------------------------------------
def test_UT_WRC05_client_pick_beats_catalog_default():
    """The wallet the user selected in the chat input MUST win over the
    catalog default. Server seeds the pick into state["wallet_id"]; the
    cascade reads it (step 2) ahead of user_context["wallet_id"] (step 3).

    Regression for the dropped-pick bug: the seed used to land under a key the
    cascade never read, so the user's selection was silently ignored and every
    transaction fell to the catalog default.
    """

    async def run():
        wallets = [
            _wallet("w-cash", name="Cash"),                       # index-0
            _wallet("w-cc", name="Credit Card", wallet_type="creditcard"),
        ]
        state = {
            "user_id": "u1",
            # Catalog default points at Cash …
            "user_context": _context(
                wallets=wallets,
                categories=[_category("c-food")],
                default="w-cash",
            ),
            # … but the user explicitly picked Credit Card in the chat input.
            "wallet_id": "w-cc",
            "proposals": [],
            "pending_proposal": None,
            "emitted_blocks_this_turn": [],
            "tool_outputs_this_turn": [],
            "__repo__": _FakeRepo(),
        }
        cmd, out = await _invoke(propose_transaction, {
            "amount": 100,
            "type": "expense",
            "note": "ค่าน้ำมัน",
            "category_label": "อาหาร",
            "state": state,
        })

        assert "error" not in out
        update_blocks = state.get("emitted_blocks_this_turn") or []
        prop = next(b for b in update_blocks
                    if b["type"] == "transaction_proposal")
        # Client pick (w-cc) wins over the catalog default (w-cash).
        assert prop["transaction"]["wallet_sync_id"] == "w-cc"

    asyncio.run(run())
