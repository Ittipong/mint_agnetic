"""Unit tests for src.agent.tools.emit_suggestions.

Covers UT-T08 (caps at 4 items) + UT-T08b (empty → error).

Wave 4 update — Command(update=) return type:
  After Issue 1 fix, `emit_suggestions` returns `Command(update=...)`.
  Tests use the `_invoke` helper to drive the tool via the ToolCall
  protocol (required by `InjectedToolCallId`) and inspect both the
  Command's update channel AND the in-place mutation on `state`.
"""

from __future__ import annotations

import asyncio
import json

from langgraph.types import Command

from src.agent.tools.emit_suggestions import emit_suggestions


async def _invoke(*, items, state, tool_call_id="tc-test"):
    """Invoke emit_suggestions via the ToolCall protocol; unwrap (Command,
    payload). `payload` is the parsed ToolMessage content."""
    tool_call = {
        "name": "emit_suggestions",
        "args": {"items": items, "state": state},
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await emit_suggestions.ainvoke(tool_call)
    messages = (cmd.update or {}).get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content)
    return cmd, {}


# ─────────────────────────────────────────────────────────────────────────────
# UT-T08 — caps at 4 items
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T08_caps_at_four_items():
    """UT-T08: more than 4 items must be silently truncated to the first 4
    so mobile renders a stable chip strip (spec section 8 schema)."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, out = await _invoke(
            items=["a", "b", "c", "d", "e", "f"], state=state,
        )
        assert out["suggestions_emitted"] == 4

        # Command(update=) carries the new block for the outer graph's
        # append_reducer to merge.
        update_blocks = cmd.update["emitted_blocks_this_turn"]
        assert len(update_blocks) == 1
        block = update_blocks[0]
        assert block["type"] == "suggestions"
        labels = [c["label"] for c in block["items"]]
        assert labels == ["a", "b", "c", "d"]

    asyncio.run(run())


def test_UT_T08_each_item_becomes_a_chip_with_label_and_send():
    """Each chip carries identical {label, send} so tapping re-sends the
    label as the next user message."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, _out = await _invoke(
            items=["ตั้งงบเดือนนี้", "ดูเทรนด์ 3 เดือน"], state=state,
        )
        block = cmd.update["emitted_blocks_this_turn"][0]
        assert block["items"] == [
            {"label": "ตั้งงบเดือนนี้", "send": "ตั้งงบเดือนนี้"},
            {"label": "ดูเทรนด์ 3 เดือน", "send": "ดูเทรนด์ 3 เดือน"},
        ]

    asyncio.run(run())


def test_UT_T08_long_items_are_trimmed_to_60_chars():
    """Items longer than 60 chars trimmed defensively — the chip strip is
    visually narrow and a 200-char chip overflows mobile layout."""
    long_text = "x" * 200

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, _out = await _invoke(items=[long_text], state=state)
        block = cmd.update["emitted_blocks_this_turn"][0]
        assert len(block["items"][0]["label"]) == 60

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T08b — empty items → structured error
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T08b_empty_items_returns_error_does_not_emit_block():
    """UT-T08b: zero items must NOT emit an empty `suggestions` block —
    mobile renders that as a bug. Tool returns a structured error so the
    LLM can retry with non-empty items."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, out = await _invoke(items=[], state=state)

        assert out["kind"] == "empty_suggestions"
        assert "no items" in out["error"].lower()
        # No `emitted_blocks_this_turn` key in update on the error path.
        assert "emitted_blocks_this_turn" not in (cmd.update or {})
        # No in-place block was added.
        assert state["emitted_blocks_this_turn"] == []

    asyncio.run(run())


def test_UT_T08b_items_that_become_empty_after_strip_still_error():
    """Items that are all whitespace must NOT count — the tool filters
    them and reports the same empty_suggestions error.

    Note: pydantic's `list[str]` schema rejects literal None at the tool
    boundary, so we only test whitespace + empty strings here. The None
    branch in `emit_suggestions._clean` is defensive belt-and-braces."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, out = await _invoke(items=["  ", "", "\t\n"], state=state)
        assert out["kind"] == "empty_suggestions"
        assert "emitted_blocks_this_turn" not in (cmd.update or {})
        assert state["emitted_blocks_this_turn"] == []

    asyncio.run(run())
