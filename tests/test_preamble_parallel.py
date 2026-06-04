"""UT-PRE — astream_with_preamble: preamble races the ReAct stream concurrently.

Regression guard for the 8s-stall bug (logs 0027/0028): a slow/dead preamble
model used to block the whole turn because each branch awaited the preamble to
drain BEFORE starting stream_chat. The merge helper now runs both at once —
preamble shows only if it beats the answer, and is cancelled (never blocks) the
moment the answer starts or the stream ends.

These tests stub `astream_preamble` so no live OpenRouter round-trip happens,
and force `PREAMBLE_ENABLED=1` (conftest disables it globally).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from src.agent import preamble as preamble_mod
from src.agent.preamble import astream_with_preamble


async def _fake_main(events: list[dict], *, delay: float = 0.0) -> AsyncIterator[dict]:
    """Yield each event, optionally sleeping `delay` before each (to lose the race)."""
    for ev in events:
        if delay:
            await asyncio.sleep(delay)
        yield ev


def _stub_preamble(monkeypatch, tokens: list[str], *, delay: float = 0.0) -> None:
    async def _fake(_msg: str) -> AsyncIterator[str]:
        for t in tokens:
            if delay:
                await asyncio.sleep(delay)
            yield t

    monkeypatch.setattr(preamble_mod, "astream_preamble", _fake)


# UT-PRE-01: preamble finishes before the answer → emitted as ONE event, first.
async def test_preamble_wins_emitted_before_answer(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "1")
    _stub_preamble(monkeypatch, ["ได้เลย", "ครับ"])  # instant
    main = _fake_main(
        [
            {"event": "status_token", "data": "กำลังคิด"},
            {"event": "answer_token", "data": "คำตอบ"},
            {"event": "done", "data": "{}"},
        ],
        delay=0.05,  # main lags so the preamble wins the race
    )

    out = [
        ev
        async for ev in astream_with_preamble(
            "เหลือเงินเท่าไหร่", main, event_type="narration_token"
        )
    ]

    narr = [e for e in out if e["event"] == "narration_token"]
    assert narr, "preamble should be emitted when it wins the race"
    assert narr[0]["data"] == "ได้เลยครับ"  # tokens joined into one event
    idx_narr = next(i for i, e in enumerate(out) if e["event"] == "narration_token")
    idx_ans = next(i for i, e in enumerate(out) if e["event"] == "answer_token")
    assert idx_narr < idx_ans, "preamble must precede the first answer token"


# UT-PRE-02: answer arrives before the preamble → preamble dropped, never shown.
async def test_answer_wins_preamble_dropped(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "1")
    _stub_preamble(monkeypatch, ["ได้เลยครับ"], delay=0.2)  # slow preamble
    main = _fake_main(
        [
            {"event": "answer_token", "data": "คำตอบ"},
            {"event": "done", "data": "{}"},
        ]
    )  # instant main

    out = [
        ev
        async for ev in astream_with_preamble(
            "x", main, event_type="narration_token"
        )
    ]

    assert not any(e["event"] == "narration_token" for e in out)
    assert any(e["event"] == "answer_token" for e in out)


# UT-PRE-03: a hung preamble does NOT block the turn — bounded by main, then cancelled.
async def test_hung_preamble_does_not_block(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "1")

    async def _hung(_msg: str) -> AsyncIterator[str]:
        await asyncio.sleep(5)  # simulates the dead-model 8s ReadTimeout
        yield "late"

    monkeypatch.setattr(preamble_mod, "astream_preamble", _hung)
    main = _fake_main(
        [
            {"event": "answer_token", "data": "เร็ว"},
            {"event": "done", "data": "{}"},
        ]
    )

    loop = asyncio.get_event_loop()
    start = loop.time()
    out = [
        ev
        async for ev in astream_with_preamble(
            "x", main, event_type="narration_token"
        )
    ]
    elapsed = loop.time() - start

    assert elapsed < 1.0, f"hung preamble blocked the turn ({elapsed:.2f}s)"
    assert not any(e["event"] == "narration_token" for e in out)


# UT-PRE-04: a block (e.g. proposal/wallet_required) before any answer token also
# closes the preamble window — preamble emitted after a block would sit after the
# real card, so it must be dropped.
async def test_block_before_preamble_drops_it(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "1")
    _stub_preamble(monkeypatch, ["ได้เลยครับ"], delay=0.2)  # slow
    main = _fake_main(
        [
            {"event": "block", "data": '{"type":"proposal"}'},
            {"event": "done", "data": "{}"},
        ]
    )

    out = [
        ev
        async for ev in astream_with_preamble(
            "x", main, event_type="narration_token"
        )
    ]

    assert not any(e["event"] == "narration_token" for e in out)
    assert any(e["event"] == "block" for e in out)


# UT-PRE-05: PREAMBLE_ENABLED=0 → straight passthrough, astream_preamble untouched.
async def test_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "0")
    calls = {"n": 0}

    async def _spy(_msg: str) -> AsyncIterator[str]:
        calls["n"] += 1
        return
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(preamble_mod, "astream_preamble", _spy)
    main = _fake_main([{"event": "answer_token", "data": "hi"}])

    out = [
        ev
        async for ev in astream_with_preamble(
            "x", main, event_type="narration_token"
        )
    ]

    assert calls["n"] == 0, "preamble must not be invoked when disabled"
    assert [e["event"] for e in out] == ["answer_token"]


# UT-PRE-06: empty user message → passthrough, no preamble attempt.
async def test_empty_message_passthrough(monkeypatch):
    monkeypatch.setenv("PREAMBLE_ENABLED", "1")
    calls = {"n": 0}

    async def _spy(_msg: str) -> AsyncIterator[str]:
        calls["n"] += 1
        return
        yield  # pragma: no cover

    monkeypatch.setattr(preamble_mod, "astream_preamble", _spy)
    main = _fake_main([{"event": "answer_token", "data": "hi"}])

    out = [
        ev
        async for ev in astream_with_preamble(
            "   ", main, event_type="narration_token"
        )
    ]

    assert calls["n"] == 0
    assert [e["event"] for e in out] == ["answer_token"]
