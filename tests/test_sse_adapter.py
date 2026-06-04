"""Unit tests for `src.agent.streaming.sse_adapter`.

Covers UT-S01..S08. The adapter is the wire-format owner — bugs here are
silent on mobile (decoder drops malformed events) so the test surface is
deliberately end-to-end at the event-shape level, not just the pure
helpers.

Mocks: a fake `graph.astream` is built per test by yielding a programmed
sequence of `(stream_mode, payload)` tuples. No real LLM / DB / pool.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessageChunk

from src.agent.streaming.sse_adapter import BlockBuffer, stream_chat


# ---------------------------------------------------------------------------
# Helpers — fake graph + collect generator output
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal graph stub whose `astream` replays a hand-built event list."""

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self._events = events

    def astream(self, init_state: dict, config: dict, stream_mode: list[str]):
        async def gen() -> AsyncIterator[tuple[str, Any]]:
            for mode, payload in self._events:
                yield mode, payload

        return gen()


class _RaisingGraph:
    """Graph whose astream raises mid-stream — exercises the error branch."""

    def __init__(self, raised: Exception) -> None:
        self._raised = raised

    def astream(self, init_state: dict, config: dict, stream_mode: list[str]):
        async def gen():
            yield "custom", {"status_token": "กำลังคิด..."}
            raise self._raised

        return gen()


async def _collect(gen: AsyncIterator[dict]) -> list[dict]:
    """Drain an async generator into a list — sync-test helper."""
    out: list[dict] = []
    async for ev in gen:
        out.append(ev)
    return out


def _ai_msg_chunk(content: str, *, tool_calls: list | None = None) -> AIMessageChunk:
    """Build an AIMessageChunk with optional tool_calls (intermediate)."""
    kwargs: dict = {"content": content}
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    return AIMessageChunk(**kwargs)


# ---------------------------------------------------------------------------
# UT-S01 — status_token from custom event reaches output
# ---------------------------------------------------------------------------


def test_UT_S01_status_token_from_custom_event() -> None:
    """UT-S01: a `custom` stream entry with `status_token` is forwarded as
    `event: status_token` with the raw value as `data`."""
    graph = _FakeGraph([
        ("custom", {"status_token": "กำลังคิด..."}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t1"}, {}, thread_id="t1")
    ))
    # Expect: status_token event, then done event (no answer text → no answer block).
    assert out[0] == {"event": "status_token", "data": "กำลังคิด..."}
    assert out[-1]["event"] == "done"
    assert json.loads(out[-1]["data"]) == {"thread_id": "t1"}


def test_UT_S01b_status_key_from_tool_emit_status_reaches_wire() -> None:
    """UT-S01b: a `custom` entry keyed `status` (what every tool's
    `_emit_status` + the numerical validator push) MUST surface as
    `event: status_token`. Regression: the adapter previously read only
    `status_token`, silently dropping all 9 tool-level progress emissions
    so the user saw one preamble then 5-15s of silence."""
    graph = _FakeGraph([
        ("custom", {"status": "กำลังเช็คข้อมูล..."}),
        ("custom", {"status": "กำลังคำนวณ..."}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t1"}, {}, thread_id="t1")
    ))
    statuses = [e["data"] for e in out if e["event"] == "status_token"]
    assert statuses == ["กำลังเช็คข้อมูล...", "กำลังคำนวณ..."]


# ---------------------------------------------------------------------------
# UT-S02 — answer_token filters tool_call chunks (intermediate drops)
# ---------------------------------------------------------------------------


def test_UT_S02_answer_token_filters_tool_call_chunks() -> None:
    """UT-S02: AIMessageChunks with non-empty tool_calls are intermediate
    reasoning and MUST be dropped. Only chunks with empty tool_calls AND
    non-empty content reach the wire as `answer_token`."""
    graph = _FakeGraph([
        # Intermediate tool-call chunk — drop.
        ("messages", (_ai_msg_chunk("", tool_calls=[
            {"name": "propose_transaction", "args": {}, "id": "1"}
        ]), {})),
        # Real answer token — forward.
        ("messages", (_ai_msg_chunk("เพิ่ม "), {})),
        ("messages", (_ai_msg_chunk("250 ค่ากาแฟ"), {})),
        # Empty content — drop.
        ("messages", (_ai_msg_chunk(""), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t2"}, {}, thread_id="t2")
    ))
    answer_tokens = [e for e in out if e["event"] == "answer_token"]
    assert [e["data"] for e in answer_tokens] == ["เพิ่ม ", "250 ค่ากาแฟ"]
    # End-of-stream: answer block flushed, done emitted.
    assert any(e["event"] == "block" for e in out)
    answer_block = next(
        json.loads(e["data"]) for e in out if e["event"] == "block"
    )
    assert answer_block == {"type": "answer", "text": "เพิ่ม 250 ค่ากาแฟ"}


# ---------------------------------------------------------------------------
# UT-S13 — narration (content + tool_call) routes to narration_token, never answer
# ---------------------------------------------------------------------------


def test_UT_S13_narration_with_tool_call_routes_to_narration() -> None:
    """UT-S13: the LIVE subgraph path delivers "narrate + call a tool" as ONE
    AIMessage carrying BOTH content (the Thai "thinking out loud" preamble)
    AND a tool_call. That content is DYNAMIC NARRATION — it must stream as a
    `narration_token` (NOT a STATIC `status_token`, NOT an `answer_token`), and
    MUST NOT leak into the persisted answer block. Only the later tool-call-FREE
    content is the real answer.
    """
    graph = _FakeGraph([
        # Narration bundled with the run_python tool_call → narration, not answer.
        ("messages", (_ai_msg_chunk(
            "กำลังรวมยอดเงินคงเหลือให้สักครู่นะครับ",
            tool_calls=[{"name": "run_python", "args": {}, "id": "1"}],
        ), {})),
        # The real answer arrives tool-call-free → answer_token + answer block.
        ("messages", (_ai_msg_chunk("ยอดเงินรวม -3,533,392 บาทครับ"), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t13"}, {}, thread_id="t13")
    ))

    # Narration surfaced as a narration_token (DYNAMIC LLM text stream).
    narration_tokens = [e["data"] for e in out if e["event"] == "narration_token"]
    assert "กำลังรวมยอดเงินคงเหลือให้สักครู่นะครับ" in narration_tokens

    # It must NOT have leaked onto the STATIC status_token stream.
    status_tokens = [e["data"] for e in out if e["event"] == "status_token"]
    assert "กำลังรวมยอดเงินคงเหลือให้สักครู่นะครับ" not in status_tokens

    # answer_token stream carries ONLY the real answer — no narration.
    answer_tokens = [e["data"] for e in out if e["event"] == "answer_token"]
    assert answer_tokens == ["ยอดเงินรวม -3,533,392 บาทครับ"]

    # The persisted answer block is clean — narration prefix is gone.
    answer_block = next(
        json.loads(e["data"]) for e in out
        if e["event"] == "block" and json.loads(e["data"]).get("type") == "answer"
    )
    assert answer_block == {"type": "answer", "text": "ยอดเงินรวม -3,533,392 บาทครับ"}
    assert "กำลังรวม" not in answer_block["text"]


def test_UT_S15_event_routing_table() -> None:
    """UT-S15: full routing table in one turn (the split-stream contract).

    Sources → events:
      - tool writer custom `status` (every tool's `_emit_status`) → `status_token`
        (STATIC tool/progress words — kept as-is).
      - AIMessage content + tool_calls (narration / "thinking out loud") →
        `narration_token` (DYNAMIC LLM text — RE-ROUTED in v3).
      - AIMessage content WITHOUT tool_calls (the real answer) → `answer_token`.

    NOTE: the streaming preamble (server.py `_text_graph_stream`) is the OTHER
    `narration_token` source; it lives in the server layer (not the adapter), so
    its routing is covered by the server/integration tests. Here we pin the two
    adapter-owned routes.
    """
    graph = _FakeGraph([
        # 1. STATIC tool progress word → status_token (unchanged).
        ("custom", {"status": "กำลังเช็คยอด..."}),
        # 2. DYNAMIC narration bundled with a tool_call → narration_token.
        ("messages", (_ai_msg_chunk(
            "กำลังรวมยอดให้นะครับ",
            tool_calls=[{"name": "run_python", "args": {}, "id": "1"}],
        ), {})),
        # 3. Tool-call-FREE content → answer_token (unchanged).
        ("messages", (_ai_msg_chunk("ยอดรวม 1,250 บาทครับ"), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t15"}, {}, thread_id="t15")
    ))

    status = [e["data"] for e in out if e["event"] == "status_token"]
    narration = [e["data"] for e in out if e["event"] == "narration_token"]
    answer = [e["data"] for e in out if e["event"] == "answer_token"]

    # Static tool words stay on status_token; nothing else leaks in.
    assert status == ["กำลังเช็คยอด..."]
    # Dynamic narration is on its own stream now.
    assert narration == ["กำลังรวมยอดให้นะครับ"]
    # The real answer is the only thing on answer_token.
    assert answer == ["ยอดรวม 1,250 บาทครับ"]

    # The answer block buffers ONLY the answer — narration never feeds it.
    answer_block = next(
        json.loads(e["data"]) for e in out
        if e["event"] == "block" and json.loads(e["data"]).get("type") == "answer"
    )
    assert answer_block == {"type": "answer", "text": "ยอดรวม 1,250 บาทครับ"}


def test_UT_S12_strips_internal_retry_markers_from_answer() -> None:
    """UT-S12: a weak model sometimes echoes the tool-error guard's
    `[TOOL_ERROR_RETRY]` instruction (or the validator's `[VALIDATOR_RETRY]`)
    as narration after the hook rewrites a ToolMessage to force a retry. Those
    internal control markers MUST be stripped from BOTH the live answer_token
    stream and the final answer block — they must never reach the user."""
    graph = _FakeGraph([
        ("messages", (_ai_msg_chunk("กำลังรวมยอดให้สักครู่นะ"), {})),
        # Echoed internal instruction — a whole token of marker text → dropped.
        ("messages", (_ai_msg_chunk(
            "[TOOL_ERROR_RETRY 1/2] The tool call above FAILED with the error "
            "shown. Do NOT apologise to the user and do NOT answer yet. Read "
            "the error, fix your code, and call the tool again now."), {})),
        ("messages", (_ai_msg_chunk("เดือนนี้ใช้ไป 106,667 บาทครับ"), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t12"}, {}, thread_id="t12")
    ))

    answer_tokens = [e["data"] for e in out if e["event"] == "answer_token"]
    # The marker token is gone; the real narration + answer survive.
    assert answer_tokens == [
        "กำลังรวมยอดให้สักครู่นะ", "เดือนนี้ใช้ไป 106,667 บาทครับ"
    ]
    assert not any("TOOL_ERROR_RETRY" in t for t in answer_tokens)

    answer_block = next(
        json.loads(e["data"]) for e in out
        if e["event"] == "block" and json.loads(e["data"]).get("type") == "answer"
    )
    assert "TOOL_ERROR_RETRY" not in answer_block["text"]
    assert "106,667" in answer_block["text"]


# ---------------------------------------------------------------------------
# UT-S03 — block JSON matches mobile schema (golden byte diff)
# ---------------------------------------------------------------------------


def test_UT_S03_block_json_matches_mobile_schema() -> None:
    """UT-S03: an `updates` event whose state delta contains a new
    transaction_proposal block reaches the wire as the EXACT JSON shape
    `chat_api_datasource.dart::decodeBlockJson` expects.

    Golden expectation:
      event: block
      data: {"type":"transaction_proposal","proposal_id":"prop-7","transaction":{...},"low_confidence":false}
    """
    txn = {
        "sync_id": "sync-7",
        "type": "expense",
        "amount": 250.0,
        "wallet_sync_id": "wallet-1",
        "category_sync_id": "cat-coffee",
        "note": "ค่ากาแฟ",
        "date": "2026-05-29",
        "currency_code": "THB",
    }
    proposal_block = {
        "type": "transaction_proposal",
        "proposal_id": "prop-7",
        "transaction": txn,
        "low_confidence": False,
    }
    graph = _FakeGraph([
        ("updates", {"react": {"emitted_blocks_this_turn": [proposal_block]}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t3"}, {}, thread_id="t3")
    ))
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    decoded = json.loads(block_events[0]["data"])
    assert decoded == proposal_block
    # Golden verification — Thai pass-through + snake_case keys preserved.
    assert "ค่ากาแฟ" in block_events[0]["data"]
    assert "proposal_id" in block_events[0]["data"]
    assert "wallet_sync_id" in block_events[0]["data"]


# ---------------------------------------------------------------------------
# UT-S04 — malformed block dropped + logged
# ---------------------------------------------------------------------------


def test_UT_S04_malformed_block_dropped() -> None:
    """UT-S04: a block whose schema fails `validate_block` is dropped from
    the wire. We don't crash — mobile would silently drop it anyway, so
    we surface the failure as a server-side drop instead."""
    bad_block = {
        "type": "transaction_proposal",
        # missing proposal_id + transaction → fails validate_block.
    }
    good_block = {"type": "answer", "text": "ดูยอดสรุปแล้วครับ"}
    graph = _FakeGraph([
        ("updates", {"react": {"emitted_blocks_this_turn": [bad_block, good_block]}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t4"}, {}, thread_id="t4")
    ))
    block_events = [e for e in out if e["event"] == "block"]
    # Only the good block survives.
    assert len(block_events) == 1
    assert json.loads(block_events[0]["data"]) == good_block


# ---------------------------------------------------------------------------
# UT-S05 — done event emitted at graph end with thread_id
# ---------------------------------------------------------------------------


def test_UT_S05_done_event_carries_thread_id() -> None:
    """UT-S05: every successful turn ends with `event: done` carrying the
    `{thread_id}` payload mobile uses to dedupe the final ack."""
    graph = _FakeGraph([
        ("custom", {"status_token": "เสร็จ"}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "abc-123"}, {}, thread_id="abc-123")
    ))
    assert out[-1] == {"event": "done", "data": json.dumps({"thread_id": "abc-123"})}


# ---------------------------------------------------------------------------
# UT-S06 — error event carries no stack trace (PII guard)
# ---------------------------------------------------------------------------


def test_UT_S06_error_event_no_stack_trace() -> None:
    """UT-S06: an uncaught exception inside the graph becomes `event:
    error` with `{code, message}` ONLY. The full traceback goes to the
    session log; the wire never carries it (PII risk + opaque to users).
    """
    err = RuntimeError("psycopg connection closed")
    graph = _RaisingGraph(err)
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t6"}, {}, thread_id="t6")
    ))
    # Last event must be `error`.
    error_events = [e for e in out if e["event"] == "error"]
    assert len(error_events) == 1
    payload = json.loads(error_events[0]["data"])
    assert payload["code"] == "RuntimeError"
    assert "psycopg connection closed" in payload["message"]
    # PII guard — no traceback string, no python file paths.
    assert "traceback" not in payload
    assert "Traceback" not in error_events[0]["data"]
    # Even when the raised exception has an empty message, we never leak
    # the file:line of the traceback to mobile.
    assert "/Users" not in error_events[0]["data"]


# ---------------------------------------------------------------------------
# UT-S07 — CRLF format matches v2 (inspection test)
# ---------------------------------------------------------------------------


def test_UT_S07_wire_format_matches_v2_dict_shape() -> None:
    """UT-S07: this adapter yields dict-shape events ({"event", "data"})
    EXACTLY like v2 server.py — same keys, same string types.
    `sse_starlette.EventSourceResponse` (used by both v2 and Wave 6's
    server) wraps them with `event:`/`data:` lines + CRLF framing.
    Per memory `project_sse_ngrok_framing`, ngrok mangles CRLF/LF on
    chunk boundaries; sse_starlette frames correctly and the mobile
    decoder normalizes the difference. We only assert OUR contract
    (dict shape) here; the CRLF wrapper is sse_starlette's job.

    v2 reference: server.py line 736 (`yield {"event": "answer_token",
    "data": data}`) and line 832-833 (`yield {"event": "done", "data":
    json.dumps({"thread_id": ...})}`).
    """
    graph = _FakeGraph([
        ("messages", (_ai_msg_chunk("ok"), {})),
        ("updates", {"x": {"emitted_blocks_this_turn": [
            {"type": "answer", "text": "ok"}
        ]}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t7"}, {}, thread_id="t7")
    ))
    # All events are dicts with exactly {"event": str, "data": str}.
    for ev in out:
        assert isinstance(ev, dict)
        assert set(ev.keys()) == {"event", "data"}
        assert isinstance(ev["event"], str)
        assert isinstance(ev["data"], str)
    # Event names are exactly the v2 set, plus the v3-added `narration_token`.
    names = {e["event"] for e in out}
    assert names <= {
        "status_token", "narration_token", "answer_token", "block", "done", "error"
    }


# ---------------------------------------------------------------------------
# UT-S08 — BlockBuffer dedupes across updates
# ---------------------------------------------------------------------------


def test_UT_S08_block_buffer_dedupes_across_updates() -> None:
    """UT-S08: when LangGraph delivers the same channel value across two
    update events (the second update appends one new block to the prior
    list), the buffer forwards ONLY the new entry."""
    b1 = {"type": "answer", "text": "a"}
    b2 = {"type": "answer", "text": "b"}
    b3 = {"type": "answer", "text": "c"}

    # First updates entry has [b1]; second has [b1, b2]; third has [b1, b2, b3].
    # Buffer must forward b1 once, b2 once, b3 once — never re-forward.
    graph = _FakeGraph([
        ("updates", {"n": {"emitted_blocks_this_turn": [b1]}}),
        ("updates", {"n": {"emitted_blocks_this_turn": [b1, b2]}}),
        ("updates", {"n": {"emitted_blocks_this_turn": [b1, b2, b3]}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t8"}, {}, thread_id="t8")
    ))
    block_events = [json.loads(e["data"]) for e in out if e["event"] == "block"]
    assert block_events == [b1, b2, b3]


# ---------------------------------------------------------------------------
# UT-S09..S11 — follow-up chips: the adapter is now a PURE FORWARDER.
# Generation + gating moved to the `gen_suggestions` graph node (see
# tests/test_gen_suggestions.py). The adapter only emits the block the node
# wrote to the `suggestions_block` channel, AFTER the answer block.
# ---------------------------------------------------------------------------


def test_UT_S09_forwards_suggestions_block_after_answer() -> None:
    """UT-S09: a `suggestions_block` arriving on the `updates` stream is
    emitted as a `block` event AFTER the answer block and BEFORE `done`."""
    graph = _FakeGraph([
        ("messages", (_ai_msg_chunk("เดือนนี้ใช้ไป 12,500 บาทครับ"), {})),
        ("updates", {"gen_suggestions": {"suggestions_block": {
            "type": "suggestions",
            "items": [{"label": "ดูเทรนด์ 3 เดือน", "send": "ดูเทรนด์ 3 เดือน"}],
        }}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t9"}, {}, thread_id="t9")
    ))
    events = [e["event"] for e in out]
    blocks = [json.loads(e["data"]) for e in out if e["event"] == "block"]
    types = [b["type"] for b in blocks]
    assert "suggestions" in types
    # Ordering: answer block before suggestions, done last.
    assert types.index("answer") < types.index("suggestions")
    assert events[-1] == "done"
    sugg = next(b for b in blocks if b["type"] == "suggestions")
    assert [it["send"] for it in sugg["items"]] == ["ดูเทรนด์ 3 เดือน"]


def test_UT_S10_no_suggestions_block_emits_nothing() -> None:
    """UT-S10: when the node writes `suggestions_block=None` (skip — ADD /
    crisis / no-answer), the adapter emits NO suggestions block."""
    graph = _FakeGraph([
        ("messages", (_ai_msg_chunk("บันทึก 250 บาทแล้วครับ"), {})),
        ("updates", {"gen_suggestions": {"suggestions_block": None}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t10"}, {}, thread_id="t10")
    ))
    types = [json.loads(e["data"])["type"] for e in out if e["event"] == "block"]
    assert "suggestions" not in types


def test_UT_S11_malformed_suggestions_block_dropped() -> None:
    """UT-S11: a malformed `suggestions_block` (fails schema) is dropped, not
    forwarded — the adapter never yields invalid JSON to mobile."""
    graph = _FakeGraph([
        ("messages", (_ai_msg_chunk("คำตอบครับ"), {})),
        ("updates", {"gen_suggestions": {"suggestions_block": {
            "type": "suggestions",  # missing required `items`
        }}}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t11"}, {}, thread_id="t11")
    ))
    types = [json.loads(e["data"])["type"] for e in out if e["event"] == "block"]
    assert "suggestions" not in types


def test_UT_S16_category_breakdown_dropped_never_emitted() -> None:
    """UT-S16: the per-category breakdown CARD was retired — breakdowns render
    as a table in the answer again. A stale `category_breakdown` block reaching
    the `updates` stream (old code path, replayed history, etc.) must be DROPPED
    by validate_block, never forwarded to mobile (its decoder no longer knows
    the type). The answer still flows through normally."""
    breakdown_block = {
        "type": "category_breakdown",
        "title": "ค่าใช้จ่ายเดือนนี้",
        "items": [
            {"name": "อาหาร", "amount": 4200.0},
            {"name": "เดินทาง", "amount": 1800.0},
        ],
    }
    graph = _FakeGraph([
        ("updates", {"react": {"emitted_blocks_this_turn": [breakdown_block]}}),
        ("messages", (_ai_msg_chunk("เดือนนี้ใช้ไป 6,000 บาทครับ"), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t16"}, {}, thread_id="t16")
    ))
    events = [e["event"] for e in out]
    types = [json.loads(e["data"])["type"] for e in out if e["event"] == "block"]
    # The stale card is dropped; only the answer block survives.
    assert "category_breakdown" not in types
    assert "answer" in types
    assert events[-1] == "done"


def test_UT_S16b_proposal_block_stays_live_before_answer() -> None:
    """UT-S16b: a `transaction_proposal` staged on the
    `emitted_blocks_this_turn` channel before the answer MUST still emit live
    (its mid-turn card UX is intentional), so it lands BEFORE the answer
    block."""
    proposal_block = {
        "type": "transaction_proposal",
        "proposal_id": "prop-16",
        "transaction": {
            "sync_id": "sync-16",
            "type": "expense",
            "amount": 250.0,
            "wallet_sync_id": "wallet-1",
        },
    }
    graph = _FakeGraph([
        ("updates", {"react": {"emitted_blocks_this_turn": [proposal_block]}}),
        ("messages", (_ai_msg_chunk("บันทึกให้แล้วครับ"), {})),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t16b"}, {}, thread_id="t16b")
    ))
    types = [json.loads(e["data"])["type"] for e in out if e["event"] == "block"]
    assert types.index("transaction_proposal") < types.index("answer")


def test_UT_S08b_block_buffer_unit_behavior() -> None:
    """UT-S08b: direct exercise of BlockBuffer's take_new — high-water
    mark advances + reset on shrinking list."""
    buf = BlockBuffer()
    # Same list reference, growing — high-water mark advances.
    blocks: list[dict] = []
    assert buf.take_new(blocks) == []
    blocks.append({"type": "answer", "text": "1"})
    # Same list ref, count went 0 -> 1: returns [first].
    assert buf.take_new(blocks) == [{"type": "answer", "text": "1"}]
    # No change → nothing new.
    assert buf.take_new(blocks) == []
    # Reset (new list object, empty) → buffer resets.
    new_blocks: list[dict] = []
    assert buf.take_new(new_blocks) == []
    # Now add a block on the NEW list — buffer starts from 0 again.
    new_blocks.append({"type": "answer", "text": "fresh"})
    assert buf.take_new(new_blocks) == [{"type": "answer", "text": "fresh"}]
    # None input is a no-op.
    assert buf.take_new(None) == []


# ---------------------------------------------------------------------------
# UT-S13 / UT-S14 — validator-reject emits a DISTINCT message (not the generic
# model-error fallback), with a dev error code in non-prod only.
# ---------------------------------------------------------------------------


def test_UT_S13_validator_reject_distinct_message_dev(monkeypatch) -> None:
    """UT-S13: a `validator_failed` custom event + no answer tokens → the
    adapter emits the validator-specific reject message, and in dev
    (ENVIRONMENT != production) appends the [DEV] error code so devs see it."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    graph = _FakeGraph([
        ("custom", {"status_token": "กำลังตรวจ..."}),
        ("custom", {"validator_failed": {
            "offending_number": "400000",
            "reason": "number_not_in_tool_outputs",
            "tool_numbers": ["99608.33", "2000000"],
        }}),
        # Validator dropped the answer → NO messages event with content.
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t13"}, {}, thread_id="t13")
    ))
    # The validator_failed custom event must NOT surface as a status_token.
    status = [e for e in out if e["event"] == "status_token"]
    assert [e["data"] for e in status] == ["กำลังตรวจ..."]
    # A distinct reject answer block reached the wire.
    blocks = [json.loads(e["data"]) for e in out if e["event"] == "block"]
    answer_blocks = [b for b in blocks if b.get("type") == "answer"]
    assert len(answer_blocks) == 1
    text = answer_blocks[0]["text"]
    assert "ยืนยันความถูกต้องของตัวเลข" in text   # validator-specific, not "ระบบขัดข้อง"
    assert "[DEV] validator-reject (__validator_failed__)" in text
    assert "offending=400000" in text


def test_UT_S14_validator_reject_no_dev_code_in_prod(monkeypatch) -> None:
    """UT-S14: in production the reject message is clean — no [DEV] code leaks."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    graph = _FakeGraph([
        ("custom", {"validator_failed": {
            "offending_number": "9961",
            "reason": "number_not_in_tool_outputs",
            "tool_numbers": ["99608.33"],
        }}),
    ])
    out = asyncio.run(_collect(
        stream_chat(graph, {"thread_id": "t14"}, {}, thread_id="t14")
    ))
    answer_blocks = [
        json.loads(e["data"]) for e in out
        if e["event"] == "block" and json.loads(e["data"]).get("type") == "answer"
    ]
    assert len(answer_blocks) == 1
    text = answer_blocks[0]["text"]
    assert "ยืนยันความถูกต้องของตัวเลข" in text
    assert "[DEV]" not in text
    assert "__validator_failed__" not in text
