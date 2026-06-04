"""Unit tests for the `gen_suggestions` graph node (src.agent.graph).

The node owns follow-up-chip GATING + data assembly; `build_suggestions_block`
(suggest_followups.py) owns the LLM call. These tests stub the LLM via
`suggest_followups._generate` and assert the node feeds the right signals and
writes the right `suggestions_block` value to state.

On the live flat path the node reads this-turn signals directly from state
(no subgraph boundary): answer + user text from `messages`, tool output from
`tool_outputs_this_turn`, the ADD signal from `emitted_blocks_this_turn` /
`__classify_route__`.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from src.agent import suggest_followups as sf
from src.agent.graph import _gen_suggestions


def _state(answer: str, user: str, **extra) -> dict:
    """Minimal state: a user turn + a final (tool-call-free) AI answer."""
    return {
        "messages": [HumanMessage(content=user), AIMessage(content=answer)],
        **extra,
    }


def test_UT_GS01_analyst_turn_writes_suggestions_block(monkeypatch) -> None:
    """UT-GS01: a non-ADD answer turn → node writes a `suggestions` block with
    the LLM's free-form chips (verbatim, no auto-wildcard)."""
    monkeypatch.setenv("SUGGESTIONS_ENABLED", "1")

    captured: dict = {}

    async def fake_generate(**kw):
        captured.update(kw)
        return {"skip": False, "reason": "ok", "items": [
            {"label": "ดูเทรนด์ 3 เดือน", "send": "ดูเทรนด์ 3 เดือน"},
            {"label": "หมวดไหนเยอะสุด", "send": "หมวดไหนใช้เยอะสุด"},
        ]}

    monkeypatch.setattr(sf, "_generate", fake_generate)

    state = _state(
        "เดือนนี้ใช้ไป 12,500 บาทครับ", "เดือนนี้ใช้ไปเท่าไร",
        tool_outputs_this_turn=[{"tool": "run_python", "result": "total=12500"}],
    )
    out = asyncio.run(_gen_suggestions(state))

    block = out["suggestions_block"]
    assert block is not None and block["type"] == "suggestions"
    sends = [it["send"] for it in block["items"]]
    assert sends == ["ดูเทรนด์ 3 เดือน", "หมวดไหนใช้เยอะสุด"]
    # The node fed the LLM the real answer + this-turn tool data (flat-path
    # parity — no dropped DATA HOOK).
    assert captured["answer_text"] == "เดือนนี้ใช้ไป 12,500 บาทครับ"
    assert "total=12500" in captured["tool_data"]


def test_UT_GS02_add_turn_via_emitted_block_skips(monkeypatch) -> None:
    """UT-GS02: an ADD turn (transaction_proposal in emitted_blocks_this_turn)
    → node writes None even though the LLM would have produced chips."""
    monkeypatch.setenv("SUGGESTIONS_ENABLED", "1")

    async def fake_generate(**_kw):
        return {"skip": False, "items": [{"label": "x", "send": "ไม่ควรโผล่"}]}

    monkeypatch.setattr(sf, "_generate", fake_generate)

    state = _state(
        "บันทึก 250 บาทแล้วครับ", "เพิ่ม 250 กาแฟ",
        emitted_blocks_this_turn=[{"type": "transaction_proposal",
                                   "proposal_id": "p1"}],
    )
    out = asyncio.run(_gen_suggestions(state))
    assert out["suggestions_block"] is None


def test_UT_GS03b_add_turn_via_propose_tool_call_skips(monkeypatch) -> None:
    """UT-GS03b: the react loop's ADD is a `propose_transaction` tool call in
    `messages`. Detected via the message scan (works on flat AND prebuilt,
    since `messages` merges across the subgraph boundary) → skip chips."""
    monkeypatch.setenv("SUGGESTIONS_ENABLED", "1")

    async def fake_generate(**_kw):
        return {"skip": False, "items": [{"label": "x", "send": "x"}]}

    monkeypatch.setattr(sf, "_generate", fake_generate)

    state = {"messages": [
        HumanMessage(content="เพิ่ม 250 กาแฟ"),
        AIMessage(content="", tool_calls=[
            {"name": "propose_transaction", "args": {}, "id": "tc1",
             "type": "tool_call"},
        ]),
        AIMessage(content="บันทึก 250 บาทแล้วครับ"),
    ]}
    out = asyncio.run(_gen_suggestions(state))
    assert out["suggestions_block"] is None


def test_UT_GS04_crisis_turn_skips(monkeypatch) -> None:
    """UT-GS04: a self-harm signal in the user message hard-skips chips
    (safety gate inside build_suggestions_block, code-side)."""
    monkeypatch.setenv("SUGGESTIONS_ENABLED", "1")

    async def fake_generate(**_kw):
        return {"skip": False, "items": [{"label": "x", "send": "x"}]}

    monkeypatch.setattr(sf, "_generate", fake_generate)

    state = _state(
        "ฟังแล้วเป็นห่วงมากเลย ลองโทร 1323 นะ",
        "ไม่ไหวแล้ว ไม่อยากอยู่ต่อแล้ว หนี้ท่วม",
    )
    out = asyncio.run(_gen_suggestions(state))
    assert out["suggestions_block"] is None


def test_UT_GS05_no_answer_skips(monkeypatch) -> None:
    """UT-GS05: no final answer text → nothing to follow up on → None."""
    monkeypatch.setenv("SUGGESTIONS_ENABLED", "1")

    async def fake_generate(**_kw):
        return {"skip": False, "items": [{"label": "x", "send": "x"}]}

    monkeypatch.setattr(sf, "_generate", fake_generate)

    # Only a user message, no AI answer.
    state = {"messages": [HumanMessage(content="เดือนนี้ใช้ไปเท่าไร")]}
    out = asyncio.run(_gen_suggestions(state))
    assert out["suggestions_block"] is None
