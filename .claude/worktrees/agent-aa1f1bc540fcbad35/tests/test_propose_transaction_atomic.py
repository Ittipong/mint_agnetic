"""Unit tests for src.agent.tools.propose_transaction — atomic discard guard.

Covers UT-T01..T04. THIS IS THE MOST CRITICAL TOOL TEST in Wave 3 — the
atomic guard (Decision X) is what closes the late-Confirm hole in v2's
state machine (memory `project_proposal_discard_state_machine`).

- UT-T01: emits transaction_proposal block + state.pending_proposal set
- UT-T02: when state.pending_proposal exists → emits discard_proposal FIRST
          then transaction_proposal (2 blocks in that order)
- UT-T03: no wallet → returns error string (does NOT raise)
- UT-T04: category resolve fails → Other-floor applied (category_sync_id=None,
          display label "อื่นๆ", per memory `project_chat_null_category_other_floor`)

We bypass the LLM category resolver via monkeypatching so these tests run
offline. The resolver itself is unit-tested in test_codeact_resolvers.py.

Wave 4 update — Command(update=) return type:
  After Issue 1 fix, `propose_transaction` returns `Command(update=...)`
  instead of a plain dict. Tests use the `_invoke_propose` helper to call
  via the ToolCall protocol (required when InjectedToolCallId is in the
  signature) and inspect `update["messages"][0].content` for the JSON
  result + `update["pending_proposal"]` for the canonical state write.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from decimal import Decimal

import pytest

from langgraph.types import Command

from src.agent.entity_catalog import CategoryEntry
from src.agent.tools.propose_transaction import propose_transaction

# `src.agent.tools.__init__` re-exports the @tool-decorated `propose_transaction`,
# so `import src.agent.tools.propose_transaction as pt_module` returns the
# StructuredTool object, NOT the module — the parent-package re-export wins
# over the submodule attribute. `importlib.import_module` always returns the
# module itself, which is what `monkeypatch.setattr` needs in order to swap
# out `_resolve_pair_async` where the tool reads it.
pt_module = importlib.import_module("src.agent.tools.propose_transaction")
_PairChoice = pt_module._PairChoice


# ─────────────────────────────────────────────────────────────────────────────
# Invocation helper — ToolCall protocol (required by InjectedToolCallId)
# ─────────────────────────────────────────────────────────────────────────────


async def _invoke_propose(
    *, state: dict, args: dict, tool_call_id: str = "tc-test",
) -> tuple[Command, dict]:
    """Invoke propose_transaction via the ToolCall protocol and unwrap.

    Returns:
      (command, result_json) where `command` is the raw Command(update=...)
      and `result_json` is the parsed ToolMessage content (success payload
      OR {"error": ..., "kind": ...}).

    The `state` kwarg is passed directly through `args` so direct-invocation
    bypasses the LangGraph state-injection machinery (which only fires inside
    ToolNode). InjectedToolCallId mandates the ToolCall envelope.
    """
    tool_call = {
        "name": "propose_transaction",
        "args": {**args, "state": state},
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await propose_transaction.ainvoke(tool_call)
    update = cmd.update or {}
    # Simulate LangGraph reducers so existing tests can inspect `state` directly.
    # `emitted_blocks_this_turn` uses append_reducer → concat onto existing list.
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


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _user_context(with_categories: bool = True) -> dict:
    """A two-wallet, two-category cached catalog (general + creditcard)."""
    cats = (
        [
            {"sync_id": "c-food", "name": "อาหาร", "type": "expense",
             "parent_id": None, "usage_count": 0,
             "wallet_sync_id": "w-cash"},
            {"sync_id": "c-salary", "name": "เงินเดือน", "type": "income",
             "parent_id": None, "usage_count": 0,
             "wallet_sync_id": "w-cash"},
        ]
        if with_categories else []
    )
    return {
        "wallets": [
            {"sync_id": "w-cash", "name": "Cash", "currency": "THB",
             "wallet_type": "general", "is_default": True, "usage_count": 0},
            {"sync_id": "w-card", "name": "KCard", "currency": "THB",
             "wallet_type": "creditcard", "is_default": False,
             "usage_count": 0},
        ],
        "categories": cats,
        "budgets": [],
        "goals": [],
        "tags": [],
        "wallet_id": "w-cash",
        "default_currency_code": "THB",
        "fetched_at": "2026-05-28T00:00:00Z",
    }


class _FakeRepo:
    """In-memory repo standin — records insert_pending_proposal calls."""

    def __init__(self):
        self.inserts: list[dict] = []

    async def insert_pending_proposal(self, *, proposal_id, user_id, kind, payload):
        self.inserts.append({
            "proposal_id": proposal_id, "user_id": user_id,
            "kind": kind, "payload": payload,
        })
        return proposal_id


@pytest.fixture
def base_state() -> dict:
    return {
        "user_id": "u-1",
        "user_context": _user_context(),
        "emitted_blocks_this_turn": [],
        "proposals": [],
        "pending_proposal": None,
        "__repo__": _FakeRepo(),
    }


@pytest.fixture(autouse=True)
def _stub_pair_resolver(monkeypatch):
    """Default stub for the joint wallet+category LLM call.

    Picks `candidate_wallets[0]` and, when `category_hint` matches a candidate
    category by name (case-insensitive), the matching cat. Otherwise returns
    a pair with `category_sync_id=None` so the propose flow applies
    Other-floor. Tests that want to force a transport-level LLM failure
    override this to return None (→ joint_resolve_failed error envelope).
    """

    async def fake_resolver(
        *,
        candidate_wallets,
        candidate_cats_by_wallet,
        amount,
        type_hint,
        category_hint,
        description,
        wallet_label,
        multi_wallet,
    ):
        if not candidate_wallets:
            return None
        # Single-wallet path always picks index 0; joint path here also picks
        # index 0 by default (a label-aware stub would diverge — none of the
        # current tests need that).
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

    monkeypatch.setattr(pt_module, "_resolve_pair_async", fake_resolver)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# UT-T01 — emits transaction_proposal + updates state.pending_proposal
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T01_emits_transaction_proposal_and_sets_pending(base_state):
    """UT-T01: a fresh ADD must (a) emit ONE transaction_proposal block and
    (b) set state.pending_proposal to the same proposal_id. No discard block
    when nothing was pending before."""

    async def run():
        cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense", "category_label": "อาหาร"},
        )

        # Successful return (no error key).
        assert "error" not in out
        assert out["proposal_id"].startswith("prop_")
        assert out["discarded_proposal_id"] is None

        # Exactly ONE block emitted, and it's the proposal.
        blocks = base_state["emitted_blocks_this_turn"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "transaction_proposal"
        assert blocks[0]["proposal_id"] == out["proposal_id"]

        # Payload matches the wire contract.
        txn = blocks[0]["transaction"]
        assert txn["type"] == "expense"
        assert txn["amount"] == 250        # integral → int (Decimal-safe)
        assert txn["category_sync_id"] == "c-food"
        assert txn["wallet_sync_id"] == "w-cash"
        assert txn["currency_code"] == "THB"

        # Shared list mutation: `proposals.append(...)` survives the pydantic
        # shallow-copy boundary (list reference shared), so direct test
        # invocation sees the new entry. In the real graph, this list also
        # gets the entry via the same path.
        assert base_state["proposals"][-1]["proposal_id"] == out["proposal_id"]
        assert base_state["proposals"][-1]["status"] == "pending"

        # Command(update=) carries the canonical scalar writes (Issue 1 fix).
        # `pending_proposal` and `last_txn` ONLY propagate via this channel
        # — the in-place `state["pending_proposal"] = ...` is defensive only.
        assert cmd.update["pending_proposal"]["proposal_id"] == out["proposal_id"]
        assert cmd.update["pending_proposal"]["status"] == "pending"
        assert cmd.update["last_txn"]["id"] == out["proposal_id"]
        assert cmd.update["last_txn"]["pending"] is True

        # Repo received the persist call.
        assert len(base_state["__repo__"].inserts) == 1
        assert base_state["__repo__"].inserts[0]["proposal_id"] == out["proposal_id"]

    asyncio.run(run())


def test_UT_T01_decimal_amounts_round_trip_without_float_drift(base_state):
    """UT-T01 supporting: a non-integral amount (250.50) must land on the
    wire EXACTLY as 250.5 (the round-trippable float), not as 250.4999999.
    We verify by re-parsing the wire value through Decimal."""

    async def run():
        await _invoke_propose(
            state=base_state,
            args={"amount": 250.50, "type": "expense",
                  "category_label": "อาหาร"},
        )
        wire = base_state["emitted_blocks_this_turn"][0]["transaction"]["amount"]
        # Re-Decimalize from the wire value — must equal Decimal("250.50").
        assert Decimal(str(wire)) == Decimal("250.50")

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T02 — atomic discard guard (Decision X)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T02_discards_pending_before_new_propose(base_state):
    """UT-T02: when a proposal is already pending, propose_transaction MUST:
      (1) emit a discard_proposal block targeting the old proposal_id,
      (2) emit a transaction_proposal block for the new one,
      (3) the discard MUST come FIRST in emitted_blocks_this_turn,
      (4) state.pending_proposal points at the NEW proposal_id,
      (5) the old proposal is marked status='discarded' in state.proposals.

    This is the atomic guard from Decision X — what makes "correction = new
    ADD" work without stacking two pending proposals.
    """
    # Seed an existing pending proposal in state.
    old_proposal = {
        "proposal_id": "prop_OLDXX",
        "intent_type": "ADD_TRANSACTION",
        "type": "ADD_TRANSACTION",
        "payload": {"amount": 100, "category": "อาหาร"},
        "status": "pending",
        "created_at": "2026-05-28T00:00:00Z",
        "confirmed_at": None,
        "cancelled_at": None,
        "discarded_at": None,
    }
    base_state["pending_proposal"] = old_proposal
    base_state["proposals"] = [old_proposal]

    async def run():
        cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense", "category_label": "อาหาร"},
        )

        # The new proposal_id is different from the old one and is reported.
        assert out["proposal_id"] != "prop_OLDXX"
        assert out["discarded_proposal_id"] == "prop_OLDXX"

        blocks = base_state["emitted_blocks_this_turn"]
        # Exactly 2 blocks, in the right ORDER.
        assert len(blocks) == 2
        assert blocks[0]["type"] == "discard_proposal"
        assert blocks[0]["target"] == "prop_OLDXX"
        assert blocks[1]["type"] == "transaction_proposal"
        assert blocks[1]["proposal_id"] == out["proposal_id"]

        # Old proposal in state.proposals is marked discarded (the in-place
        # mutation on the dict inside the list IS shared across the
        # pydantic shallow-copy boundary — the list ref + the dict ref).
        old = next(
            p for p in base_state["proposals"]
            if p["proposal_id"] == "prop_OLDXX"
        )
        assert old["status"] == "discarded"
        assert old["discarded_at"] is not None

        # New proposal appended at the end.
        assert base_state["proposals"][-1]["proposal_id"] == out["proposal_id"]
        assert base_state["proposals"][-1]["status"] == "pending"

        # Command(update=) now points pending_proposal at the NEW entry.
        assert cmd.update["pending_proposal"]["proposal_id"] == out["proposal_id"]

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T03 — no wallet → returns error (does NOT raise)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T03_no_wallet_emits_onboarding_cta_inline(base_state):
    """UT-T03: when the user has no wallets, the tool emits the
    `wallet_required` CTA block IN-PLACE and returns a structured
    {cta_emitted, kind=no_wallet} payload — no exception, no proposal,
    no extra LLM round routed through `wallet_required_cta`.

    This is the onboarding shortcut from Option 4: by handling the empty-
    wallet path inside the same tool call, a "เพิ่ม 50 ค่าน้ำ" turn for
    a brand-new user finishes onboarding in ONE LLM round instead of two
    (the LLM-round savings was the main motivation for the rewrite — see
    log 0023, trace 019e741a-…).
    """
    # Empty out the wallets in the cached context.
    base_state["user_context"]["wallets"] = []
    base_state["user_context"]["wallet_id"] = None

    async def run():
        cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense", "category_label": "อาหาร"},
        )

        # Structured onboarding payload — NOT an exception, NOT a proposal.
        assert out["cta_emitted"] is True
        assert out["kind"] == "no_wallet"
        # The wallet_required CTA block landed via Command(update=...).
        blocks = (cmd.update or {}).get("emitted_blocks_this_turn") or []
        assert len(blocks) == 1
        assert blocks[0]["type"] == "wallet_required"
        assert blocks[0]["action"]["target"] == "/wallets/create"
        # No pending_proposal landed (neither in state nor in Command.update).
        assert base_state.get("pending_proposal") is None
        assert "pending_proposal" not in (cmd.update or {})
        # No insert was attempted.
        assert base_state["__repo__"].inserts == []

    asyncio.run(run())


def test_UT_T03_missing_context_auto_loads_and_propagates(monkeypatch, base_state):
    """UT-T03 supporting: when `state.user_context` is None, the tool no
    longer errors with `missing_context` — it auto-loads via the shared
    `load_user_context_dict` helper and persists the loaded context via
    Command(update={"user_context": ...}) so downstream tools in the same
    turn see the cached catalog too.

    Rationale (Option 4 rewrite): the LLM frequently skipped the
    `get_user_context` bootstrap call for one-shot ADD turns. The old gate
    cost a full LLM round to recover. Auto-loading removes the failure
    mode entirely while keeping the same data + same resolver chain.
    """
    cached_ctx = base_state["user_context"]
    # Wipe the cached context the test fixture pre-populated.
    base_state["user_context"] = None

    # Stub the loader so the test doesn't need a real DB. Return the same
    # shape the fixture pre-populated — that way step 4-7 logic is exercised
    # exactly as in the happy path. Wave 5.1: signature is now 3-tuple
    # (context_dict, catalog, error); we rehydrate a catalog from the dict
    # so per-wallet category scoping works against the fixture.
    from src.agent.entity_catalog import EntityCatalog
    fake_catalog = EntityCatalog.from_dict(cached_ctx)

    async def fake_load(user_id: str):  # noqa: ARG001 — signature must match
        return cached_ctx, fake_catalog, None

    monkeypatch.setattr(pt_module, "load_user_context_dict", fake_load)

    async def run():
        cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense", "category_label": "อาหาร"},
        )
        # Proposal landed — auto-load successful, full propose flow ran.
        assert "proposal_id" in out
        # Command propagates the freshly-loaded context so a follow-up
        # run_python in the same turn skips the DB hit.
        assert (cmd.update or {}).get("user_context") == cached_ctx

    asyncio.run(run())


def test_UT_T03_missing_context_with_failed_load_errors(monkeypatch, base_state):
    """UT-T03 supporting: if the auto-load itself fails (e.g. DB pool
    blip both attempts), the tool still returns a structured error so the
    ReAct loop can retry — no exception escapes."""
    base_state["user_context"] = None

    async def fake_load(user_id: str):  # noqa: ARG001
        # Wave 5.1 — 3-tuple shape: (context, catalog, error).
        return None, None, RuntimeError("simulated pool exhausted")

    monkeypatch.setattr(pt_module, "load_user_context_dict", fake_load)

    async def run():
        _cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense", "category_label": "อาหาร"},
        )
        assert out["kind"] == "catalog_load_failed"
        assert "simulated pool exhausted" in out["error"]

    asyncio.run(run())


def test_UT_T03_invalid_amount_returns_error(base_state):
    """UT-T03 supporting: a non-positive or unparseable amount returns
    kind='invalid_amount' — the LLM can retry with a corrected number."""

    async def run():
        for amt in (0, -10):
            fresh_state = dict(
                base_state, emitted_blocks_this_turn=[], proposals=[],
                pending_proposal=None, __repo__=_FakeRepo(),
            )
            _cmd, out = await _invoke_propose(
                state=fresh_state,
                args={"amount": amt, "type": "expense",
                      "category_label": "อาหาร"},
            )
            assert out["kind"] == "invalid_amount", f"amount={amt} should error"

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T04 — Other-floor when category resolve fails
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T04_llm_picks_no_category_applies_other_floor(
    base_state, monkeypatch,
):
    """UT-T04: when the joint LLM returns a valid pair with
    `category_sync_id=null` (LLM picked a wallet but found no plausible
    category), the proposal MUST land with category_sync_id=None and a
    display label "อื่นๆ" (Other-floor). Memory
    `project_chat_null_category_other_floor` — a null fallback used to pick
    whatever the first income/expense category was, which produced wrong
    proposals (slip discount labelled "เงินเดือน")."""

    async def pick_no_cat(
        *, candidate_wallets, candidate_cats_by_wallet, **_kwargs,
    ):
        # LLM picked the wallet but explicitly returned null for category.
        wallet = candidate_wallets[0]
        return _PairChoice(
            wallet_sync_id=wallet.sync_id,
            category_sync_id=None,
            confidence=0.4,
            reason="no plausible category",
        )

    monkeypatch.setattr(pt_module, "_resolve_pair_async", pick_no_cat)

    async def run():
        _cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 250, "type": "expense",
                  "category_label": "totally-unknown-thing"},
        )
        assert "error" not in out

        blocks = base_state["emitted_blocks_this_turn"]
        assert len(blocks) == 1
        txn = blocks[0]["transaction"]
        # Other-floor: NULL sync_id (mobile can edit), display label "อื่นๆ".
        assert txn["category_sync_id"] is None
        assert txn["category"] == "อื่นๆ"
        # `note` defaults to the user's original label (not "อื่นๆ").
        assert txn["note"] == "totally-unknown-thing"

    asyncio.run(run())


def test_UT_T04_llm_call_failure_returns_joint_error(
    base_state, monkeypatch,
):
    """UT-T04 supporting: when the joint LLM call itself fails (transport
    timeout, validation error, out-of-set wallet pick — anything the
    resolver maps to None), the tool MUST return a structured error
    `kind="joint_resolve_failed"` so ReAct can retry or surface a message
    to the user (Q10b — hard fail-loud per spec).

    NOTE: this is a deliberate behavior change from the legacy 2-call
    chain, where a resolver raise/None silently fell through to Other-floor.
    The new joint design treats LLM failure as a first-class error because
    it cannot distinguish "LLM unsure" from "LLM offline".
    """

    async def llm_down(**_kwargs):
        return None  # mirrors what _resolve_pair_async does on any failure

    monkeypatch.setattr(pt_module, "_resolve_pair_async", llm_down)

    async def run():
        _cmd, out = await _invoke_propose(
            state=base_state,
            args={"amount": 99, "type": "expense",
                  "category_label": "อาหาร"},
        )
        assert out["kind"] == "joint_resolve_failed"
        # No proposal block was emitted (just the discard if any), so the
        # mobile renderer doesn't draw a phantom card.
        blocks = base_state["emitted_blocks_this_turn"]
        assert not any(b["type"] == "transaction_proposal" for b in blocks)

    asyncio.run(run())


# Re-export FakeRepo for the UT-T03 amount test that builds its own state
_FakeRepoExport = _FakeRepo
