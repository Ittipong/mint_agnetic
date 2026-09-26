"""Unit tests for `src.agent.endpoints.slip_handler`.

Covers UT-SL01..SL04. The slip flow short-circuits the ReAct loop entirely
— one vision call + one transaction_proposal_group block. Tests mock the
vision call and pass a hand-built EntityCatalog so no DB / OpenRouter is
needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from src.agent.endpoints.slip_handler import (
    SLIP_UNREADABLE_MESSAGE,
    SlipData,
    extract_slip,
    handle_slip_chat,
)
from src.agent.entity_catalog import CategoryEntry, EntityCatalog, WalletEntry


# ---------------------------------------------------------------------------
# Helpers — fake catalog + vision call
# ---------------------------------------------------------------------------


def _catalog() -> EntityCatalog:
    """Small but realistic catalog: 1 general wallet, 1 creditcard wallet,
    1 expense + 1 income category + Other categories for both directions."""
    return EntityCatalog(
        wallets=[
            WalletEntry(
                sync_id="wallet-cash",
                name="เงินสด",
                currency="THB",
                wallet_type="general",
                is_default=True,
            ),
            WalletEntry(
                sync_id="wallet-cc",
                name="บัตรเครดิต",
                currency="THB",
                wallet_type="creditcard",
            ),
        ],
        categories=[
            CategoryEntry(sync_id="cat-coffee", name="กาแฟ", type="expense"),
            CategoryEntry(sync_id="cat-salary", name="เงินเดือน", type="income"),
            CategoryEntry(sync_id="cat-other-exp", name="อื่นๆ", type="expense"),
            CategoryEntry(sync_id="cat-other-inc", name="อื่นๆ", type="income"),
        ],
    )


def _make_vision_call(returned_json: str):
    """Build a vision callable that returns a fixed JSON string."""
    calls: list[list[dict]] = []

    async def vision(messages: list[dict]) -> str:
        calls.append(messages)
        return returned_json

    vision.calls = calls  # type: ignore[attr-defined]
    return vision


def _make_failing_vision_call(exc: Exception):
    async def vision(messages: list[dict]) -> str:
        raise exc

    return vision


async def _collect(gen: AsyncIterator[dict]) -> list[dict]:
    out: list[dict] = []
    async for ev in gen:
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# UT-SL01 — vision -> transaction_proposal_group block
# ---------------------------------------------------------------------------


def test_UT_SL01_vision_emits_transaction_proposal_group() -> None:
    """UT-SL01: a successful vision parse with one expense line emits a
    `transaction_proposal_group` block with the row, the wallet's
    sync_id, and a positive total."""
    vision_call = _make_vision_call(json.dumps({
        "readable": True,
        "transactions": [
            {
                "type": "expense",
                "amount": 120,
                "category_sync_id": "cat-coffee",
                "note": "ค่ากาแฟ",
                "include_in_report": True,
            }
        ],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["fake-jpeg-b64"],
            thread_id="t-sl1",
            user_id="u-1",
        )
    ))
    # status_token events first, then one block, then done.
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    payload = json.loads(block_events[0]["data"])
    assert payload["type"] == "transaction_proposal_group"
    assert payload["wallet_sync_id"] == "wallet-cash"  # default general wallet
    assert payload["total"] == 120  # int (integral)
    assert len(payload["transactions"]) == 1
    txn = payload["transactions"][0]
    assert txn["type"] == "expense"
    assert txn["amount"] == 120
    assert txn["category_sync_id"] == "cat-coffee"
    assert txn["wallet_sync_id"] == "wallet-cash"
    assert txn["note"] == "ค่ากาแฟ"
    # Done event last.
    assert out[-1]["event"] == "done"


def test_UT_SL01b_extract_slip_returns_unreadable_on_failure() -> None:
    """UT-SL01b: vision call exception degrades to SlipData(readable=False)
    with the Thai unreadable sentence — never raises."""
    vision_call = _make_failing_vision_call(RuntimeError("openrouter 502"))
    catalog = _catalog()
    result = asyncio.run(
        extract_slip(
            vision_call,
            user_id="u",
            image_b64="x",
            catalog=catalog,
            wallet=catalog.wallets[0],
        )
    )
    assert isinstance(result, SlipData)
    assert result.readable is False
    assert result.message == SLIP_UNREADABLE_MESSAGE
    assert result.payloads == []


def test_UT_SL01c_unreadable_emits_answer_block() -> None:
    """UT-SL01c: when vision returns {"readable": false}, the handler
    emits an `answer` block with the Thai unreadable sentence + done.
    No `transaction_proposal_group` reaches the wire."""
    vision_call = _make_vision_call(json.dumps({
        "readable": False,
        "transactions": [],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["x"],
            thread_id="t",
            user_id="u",
        )
    ))
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    payload = json.loads(block_events[0]["data"])
    assert payload["type"] == "answer"
    assert SLIP_UNREADABLE_MESSAGE in payload["text"]


# ---------------------------------------------------------------------------
# UT-SL02 (B1) — only first image processed even when 2 provided
# ---------------------------------------------------------------------------


def test_UT_SL02_B1_only_first_image_processed() -> None:
    """UT-SL02 (B1): when the mobile client sends `image_b64s = [a, b]`,
    the handler processes ONLY `a`. Vision is called exactly once. Multi-
    image slips are out of v3 scope; behavior matches v2 byte-for-byte."""
    vision_call = _make_vision_call(json.dumps({
        "readable": True,
        "transactions": [
            {"type": "expense", "amount": 50, "category_sync_id": "cat-coffee",
             "note": "img-1", "include_in_report": True}
        ],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["image-A-b64", "image-B-b64"],
            thread_id="t-sl2",
            user_id="u-1",
        )
    ))
    # Vision called exactly once.
    assert len(vision_call.calls) == 1
    # The image part inside the user message is image-A (the first).
    user_msg = vision_call.calls[0][1]
    parts = user_msg["content"]
    image_part = next(p for p in parts if p.get("type") == "image_url")
    assert "image-A-b64" in image_part["image_url"]["url"]
    # And the output is one group block.
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    payload = json.loads(block_events[0]["data"])
    assert payload["type"] == "transaction_proposal_group"


def test_UT_SL02b_empty_image_list_emits_unreadable() -> None:
    """UT-SL02b: an empty `image_b64s` list (mobile bug or accidental
    omission) degrades to the unreadable sentence. Vision NOT called."""
    vision_call = _make_vision_call("never used")
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=[],
            thread_id="t",
            user_id="u",
        )
    ))
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    payload = json.loads(block_events[0]["data"])
    assert payload["type"] == "answer"
    assert SLIP_UNREADABLE_MESSAGE in payload["text"]
    assert vision_call.calls == []


# ---------------------------------------------------------------------------
# UT-SL03 — group_id == proposal_id in emitted block
# ---------------------------------------------------------------------------


def test_UT_SL03_group_id_equals_proposal_id() -> None:
    """UT-SL03 (memory `project_slip_vision_as_node`): the emitted
    `transaction_proposal_group` block carries BOTH `group_id` and
    `proposal_id` with the SAME value. Mobile history replay annotates
    block status by matching `blk["proposal_id"]`, so without this
    duplication a reloaded group card stays status=None ("never saved")."""
    vision_call = _make_vision_call(json.dumps({
        "readable": True,
        "transactions": [
            {"type": "expense", "amount": 99, "category_sync_id": "cat-coffee",
             "note": "test", "include_in_report": True}
        ],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["x"],
            thread_id="t-sl3",
            user_id="u-1",
        )
    ))
    block_events = [e for e in out if e["event"] == "block"]
    payload = json.loads(block_events[0]["data"])
    assert payload["group_id"], "group_id must be non-empty"
    assert payload["proposal_id"], "proposal_id must be non-empty"
    assert payload["group_id"] == payload["proposal_id"], (
        f"group_id ({payload['group_id']}) and proposal_id "
        f"({payload['proposal_id']}) must match — mobile history replay "
        "depends on it (memory: project_slip_vision_as_node)"
    )


def test_UT_SL03b_default_currency_code_passed_through() -> None:
    """UT-SL03b: the `currency_code` field is None on the wire (mobile
    fills it from the wallet) — matches v2 contract."""
    vision_call = _make_vision_call(json.dumps({
        "readable": True,
        "transactions": [
            {"type": "expense", "amount": 10, "category_sync_id": "cat-coffee",
             "note": "x", "include_in_report": True}
        ],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["x"],
            thread_id="t",
            user_id="u",
        )
    ))
    block_events = [e for e in out if e["event"] == "block"]
    payload = json.loads(block_events[0]["data"])
    assert payload["currency_code"] is None


# ---------------------------------------------------------------------------
# UT-SL04 — slip/confirm persistence targets a node that EXISTS in the graph
# ---------------------------------------------------------------------------


def test_UT_SL04_as_node_is_a_real_terminal_node() -> None:
    """UT-SL04: out-of-graph writes (slip group seed, confirm, cancel) must pass
    `as_node="post_turn"` — v3's terminal node. v2 used "finalize", which v3
    does not have; writing as a missing node raises InvalidUpdateError (the old
    confirm/cancel 500). Guards both sides: the code uses post_turn, and the
    compiled graph really has a post_turn node wired to END.
    """
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver

    from src.agent.graph import build_graph

    src = (Path(__file__).resolve().parents[1] / "src/agent/server.py").read_text(
        encoding="utf-8")
    assert 'as_node="post_turn"' in src
    assert 'as_node="finalize"' not in src.replace('`as_node="finalize"`', "")

    from tests.test_react_loop import FakeToolingModel

    async def nodes():
        g = await build_graph(model=FakeToolingModel(responses=[]),
                              checkpointer=MemorySaver())
        return g.get_graph()

    graph = asyncio.run(nodes())
    assert "post_turn" in graph.nodes
    assert "finalize" not in graph.nodes
    assert any(e.source == "post_turn" and e.target == "__end__" for e in graph.edges)


def test_UT_SL04b_wallet_id_pick_honored() -> None:
    """UT-SL04b: when the mobile client passes `wallet_id`, the slip
    handler honors it (not the catalog default). Memory:
    `project_wallet_index0_ordering` (the fallback only triggers when
    no client pick is provided)."""
    vision_call = _make_vision_call(json.dumps({
        "readable": True,
        "transactions": [
            {"type": "expense", "amount": 1, "category_sync_id": "cat-coffee",
             "note": "x", "include_in_report": True}
        ],
    }))
    out = asyncio.run(_collect(
        handle_slip_chat(
            vision_call=vision_call,
            catalog=_catalog(),
            image_b64s=["x"],
            thread_id="t",
            user_id="u",
            wallet_id="wallet-cc",  # explicit pick
        )
    ))
    block_events = [e for e in out if e["event"] == "block"]
    payload = json.loads(block_events[0]["data"])
    assert payload["wallet_sync_id"] == "wallet-cc"
    # Inner txn rows also point to the chosen wallet.
    assert payload["transactions"][0]["wallet_sync_id"] == "wallet-cc"


@pytest.mark.parametrize("raw, expected", [
    ("2026-09-24", "2026-09-24"),                    # slip's own date kept
    ("2569-09-24", "2026-09-24"),                    # BE year left unconverted
    ("2026-09-24T12:41:00+07:00", "2026-09-24"),     # datetime → date only
    ("2026-12-01", "2026-09-26T10:00:00+00:00"),     # future → now
    (None, "2026-09-26T10:00:00+00:00"),             # not printed → now
    ("24 ก.ย.", "2026-09-26T10:00:00+00:00"),        # unparseable → now
])
def test_UT_SL08_slip_date_uses_printed_date(raw, expected) -> None:
    """UT-SL08: a slip dated "24 ก.ย. 2569" was saved as TODAY — the vision
    prompt never asked for a date, so every line fell back to now()."""
    from src.agent.endpoints.slip_handler import _valid_date
    assert _valid_date(raw, "2026-09-26T10:00:00+00:00") == expected


def test_UT_SL08b_vision_prompt_asks_for_the_date() -> None:
    from src.agent.endpoints.slip_handler import _build_slip_prompt
    import inspect
    src = inspect.getsource(_build_slip_prompt)
    assert '"date": "2026-09-24"' in src and "Buddhist Era" in src
