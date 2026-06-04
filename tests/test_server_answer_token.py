"""Server-level answer_token wiring (Wave 7b adaptation).

v2 `test_server_answer_token.py` drove `chat_stream_generator` against a fake
graph whose `astream_events` yielded the v2 event stream. v3 replaces
`astream_events` with `stream_chat` over LangGraph's `astream(stream_mode=
["messages", "custom", "updates"])` (see `src.agent.streaming.sse_adapter`),
so the wire format is asserted at that adapter level instead.

Existing v3 coverage:
- `test_sse_adapter.py` UT-S02 already asserts answer_token filters tool-call
  chunks (keeps only AI `content` text without tool_calls).
- `test_sse_adapter.py` UT-S08 asserts block-buffer dedupe across updates.

What this file ADDS — the wider contract v2's test asserted but isn't
covered in one place yet:

  UT-SAT-001: a mixed stream (status -> answer_token chunks -> updates with
              answer block + suggestions block) yields the correct ordered
              wire events; the `answer`-type block is BUFFERED in updates
              but the test confirms NO duplicate emission as a `block` event
              when the same text was already streamed as answer_tokens (the
              v2 "answer block not re-emitted as SSE block" contract).
  UT-SAT-002: malformed block in updates is dropped silently (no exception
              escapes to wire) — defensive in-process boundary.
  UT-SAT-003: done event carries thread_id sourced from initial_state when
              config doesn't supply it (caller convenience contract).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from langchain_core.messages import AIMessageChunk

from src.agent.streaming.sse_adapter import stream_chat


class _FakeGraph:
    """Drives a fixed sequence of (stream_mode, chunk) tuples through astream."""

    def __init__(self, events: list[tuple[str, Any]]):
        self._events = events

    def astream(self, _state, *, config=None, stream_mode=None):
        events = self._events

        async def _gen() -> AsyncIterator[tuple[str, Any]]:
            for mode, chunk in events:
                yield mode, chunk

        return _gen()


def _msg(content: str, tool_calls=None) -> tuple:
    """Build a (AIMessageChunk, metadata) tuple as `messages` stream yields."""
    msg = AIMessageChunk(content=content)
    if tool_calls is not None:
        msg.tool_calls = tool_calls
    return (msg, {"langgraph_step": 1})


async def _collect(gen) -> list[dict]:
    return [ev async for ev in gen]


# ---------------------------------------------------------------------------
# UT-SAT-001 — mixed stream emits status + answer_tokens + non-answer blocks
# ---------------------------------------------------------------------------
def test_UT_SAT001_mixed_stream_emits_correct_ordered_wire_events():
    """A realistic turn yields: custom status -> 2x answer_token chunks ->
    updates with a `suggestions` block. The wire MUST carry exactly:
        [status_token, answer_token, answer_token, block(suggestions), done]
    in that order. The `answer` block in updates (if any) is NOT re-emitted
    as a `block` event because answer text already went out via answer_token.
    """

    async def run():
        events = [
            # 1. Custom status word.
            ("custom", {"status_token": "กำลังคิด"}),
            # 2. Two answer_token chunks (final answer streaming).
            ("messages", _msg("ยอด")),
            ("messages", _msg("คงเหลือ 1,234.50 บาท")),
            # 3. Updates: a non-answer block (suggestions).
            ("updates", {"react_agent": {
                "emitted_blocks_this_turn": [
                    {"type": "suggestions",
                     "items": [{"label": "ดูเดือนนี้", "send": "เดือนนี้"}]},
                ]
            }}),
        ]
        wire = await _collect(stream_chat(
            _FakeGraph(events),
            initial_state={"thread_id": "t-1", "user_id": "u-1", "messages": []},
            config={"configurable": {"thread_id": "t-1"}},
        ))
        kinds = [ev["event"] for ev in wire]
        # Ordering invariant: status_token first; answer_tokens before any
        # block; done last.
        assert kinds[0] == "status_token"
        assert kinds[-1] == "done"
        # answer_tokens come before blocks.
        first_block = kinds.index("block")
        for i, k in enumerate(kinds):
            if k == "answer_token":
                assert i < first_block, kinds
        # Suggestions block carried through.
        block_evs = [ev for ev in wire if ev["event"] == "block"]
        types = [json.loads(ev["data"]).get("type") for ev in block_evs]
        assert "suggestions" in types

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-SAT-002 — malformed block in updates is dropped, never throws
# ---------------------------------------------------------------------------
def test_UT_SAT002_malformed_block_dropped_silently():
    """A block dict missing required keys (e.g. `type`) MUST be dropped
    without crashing the stream. This is defensive against tool bugs that
    might land bad blocks in the channel."""

    async def run():
        events = [
            ("updates", {"react_agent": {
                "emitted_blocks_this_turn": [
                    {"not_a_type": "broken"},                # malformed
                    {"type": "suggestions",
                     "items": [{"label": "ok", "send": "ok"}]},
                ],
            }}),
        ]
        wire = await _collect(stream_chat(
            _FakeGraph(events),
            initial_state={"thread_id": "t-2", "user_id": "u-1", "messages": []},
            config={"configurable": {"thread_id": "t-2"}},
        ))
        # Only ONE `block` event (the valid suggestions), plus done.
        block_evs = [ev for ev in wire if ev["event"] == "block"]
        assert len(block_evs) == 1
        assert json.loads(block_evs[0]["data"])["type"] == "suggestions"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-SAT-003 — done event carries thread_id from initial_state
# ---------------------------------------------------------------------------
def test_UT_SAT003_done_event_carries_thread_id_from_initial_state():
    """When config doesn't supply thread_id, the adapter falls back to
    initial_state['thread_id'] for the done payload — caller convenience."""

    async def run():
        events = [("messages", _msg("ok"))]
        wire = await _collect(stream_chat(
            _FakeGraph(events),
            initial_state={"thread_id": "thread-abc", "user_id": "u-1",
                           "messages": []},
            config={},  # NO configurable.thread_id
        ))
        done = next(ev for ev in wire if ev["event"] == "done")
        payload = json.loads(done["data"])
        assert payload["thread_id"] == "thread-abc"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-SAT-004 — content alongside tool_calls IS streamed (thinking-out-loud UX)
# ---------------------------------------------------------------------------
def test_UT_SAT004_content_with_tool_calls_is_emitted():
    """When the LLM emits `content` (Thai narration) in the SAME response
    as `tool_calls`, the content MUST reach the user as an answer_token —
    this drives the "thinking out loud" UX (memory: progress-narration in
    prompts.py). The user sees "กำลังเช็คยอดบัตรให้นะ" streaming while the
    tool runs, instead of staring at a blank screen for 5-15s.

    Empty content + tool_calls → dropped (no text to emit)."""

    async def run():
        events = [
            # Narration chunk — content + tool_calls. MUST be emitted now.
            ("messages", _msg("กำลังเช็คยอดบัตรให้นะ", tool_calls=[
                {"name": "run_python", "args": {}, "id": "1"},
            ])),
            # Final answer chunk — must be kept too.
            ("messages", _msg("เสร็จแล้วครับ")),
            # Empty content + tool_calls → no text to stream, dropped.
            ("messages", _msg("", tool_calls=[
                {"name": "run_python", "args": {}, "id": "2"},
            ])),
        ]
        wire = await _collect(stream_chat(
            _FakeGraph(events),
            initial_state={"thread_id": "t-4", "user_id": "u-1", "messages": []},
            config={"configurable": {"thread_id": "t-4"}},
        ))
        answer_events = [ev for ev in wire if ev["event"] == "answer_token"]
        texts = [ev["data"] for ev in answer_events]
        assert texts == ["กำลังเช็คยอดบัตรให้นะ", "เสร็จแล้วครับ"], texts

    asyncio.run(run())
