"""Unit tests for `src.agent.endpoints.voice_handler`.

Covers UT-VC01..VC03. The voice endpoint is the simpler of the two
pre-processors (one STT call, then delegate to ReAct stream). Tests mock
both the STT callable and the graph so no real OpenRouter / DB is touched.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from src.agent.endpoints.voice_handler import (
    NO_SPEECH_SENTINEL,
    STTResult,
    handle_voice_chat,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GraphYieldingDone:
    """Fake graph whose astream yields exactly one done-shaped event so
    handle_voice_chat returns after the transcript block."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def astream(self, init_state: dict, config: dict, stream_mode: list[str]):
        self.calls.append({"init_state": init_state, "config": config})

        async def gen():
            # Empty event stream — stream_chat will emit done itself.
            if False:
                yield
            return

        return gen()


def _make_stt_call(returned_text: str = "ใช้กาแฟห้าสิบบาท"):
    """Build an STT callable that returns a fixed transcript."""
    calls: list[list[dict]] = []

    async def stt(messages: list[dict]) -> str:
        calls.append(messages)
        return returned_text

    stt.calls = calls  # type: ignore[attr-defined]
    return stt


def _make_failing_stt_call(exc: Exception):
    async def stt(messages: list[dict]) -> str:
        raise exc

    return stt


async def _collect(gen: AsyncIterator[dict]) -> list[dict]:
    out: list[dict] = []
    async for ev in gen:
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# UT-VC01 — Gemini STT m4a -> text via mock callable
# ---------------------------------------------------------------------------


def test_UT_VC01_stt_m4a_to_text() -> None:
    """UT-VC01: `transcribe_audio` calls the injected STT callable with the
    expected multimodal message shape (audio_b64 + format=m4a) and returns
    the cleaned transcript."""
    stt_call = _make_stt_call("กาแฟห้าสิบบาท")
    audio_bytes = b"fake-m4a-bytes"

    result = asyncio.run(transcribe_audio(stt_call, audio_bytes=audio_bytes))
    assert isinstance(result, STTResult)
    assert result.text == "กาแฟห้าสิบบาท"
    assert result.error_reason is None
    # Verify the STT call shape — system prompt + user-content with input_audio.
    assert len(stt_call.calls) == 1
    messages = stt_call.calls[0]
    assert messages[0]["role"] == "system"
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    audio_part = next(p for p in parts if p.get("type") == "input_audio")
    assert audio_part["input_audio"]["format"] == "m4a"
    # Bytes should be base64-encoded into the prompt.
    import base64
    expected_b64 = base64.b64encode(audio_bytes).decode("ascii")
    assert audio_part["input_audio"]["data"] == expected_b64


def test_UT_VC01b_stt_strips_wrapping_quotes() -> None:
    """UT-VC01b: a transcript wrapped in quotes (model artifact) is cleaned."""
    stt_call = _make_stt_call('"กาแฟห้าสิบ"')
    result = asyncio.run(transcribe_audio(stt_call, audio_bytes=b"x"))
    assert result.text == "กาแฟห้าสิบ"


def test_UT_VC01c_stt_transport_failure_maps_to_friendly_reason() -> None:
    """UT-VC01c: a transport / model failure becomes
    `error_reason="transcription_failed"` — never an HTTP 500."""
    stt_call = _make_failing_stt_call(RuntimeError("openrouter timeout"))
    result = asyncio.run(transcribe_audio(stt_call, audio_bytes=b"x"))
    assert result.text is None
    assert result.error_reason == "transcription_failed"


# ---------------------------------------------------------------------------
# UT-VC02 — empty / NO_SPEECH transcript -> stt_error block
# ---------------------------------------------------------------------------


def test_UT_VC02_empty_audio_emits_stt_error_block() -> None:
    """UT-VC02a: empty audio bytes return STTResult(error_reason='no_speech')
    WITHOUT calling the model — saves a wasted round-trip."""
    stt_call = _make_stt_call("nope")
    result = asyncio.run(transcribe_audio(stt_call, audio_bytes=b""))
    assert result.error_reason == "no_speech"
    # Model NOT called.
    assert stt_call.calls == []


def test_UT_VC02b_no_speech_sentinel_returns_error() -> None:
    """UT-VC02b: the model's `NO_SPEECH` sentinel maps to
    `error_reason='no_speech'`."""
    stt_call = _make_stt_call(NO_SPEECH_SENTINEL)
    result = asyncio.run(transcribe_audio(stt_call, audio_bytes=b"x"))
    assert result.text is None
    assert result.error_reason == "no_speech"


def test_UT_VC02c_handle_voice_chat_emits_stt_error_then_done() -> None:
    """UT-VC02c: when STT yields no_speech, the orchestrator emits
    `event: block` {type: stt_error, reason: no_speech} followed by
    `event: done` — and SKIPS the ReAct stream entirely (graph.astream
    never called)."""
    graph = _GraphYieldingDone()
    stt_call = _make_failing_stt_call(RuntimeError("provider down"))

    out = asyncio.run(_collect(
        handle_voice_chat(
            graph=graph,
            stt_call=stt_call,
            audio_bytes=b"audio",
            thread_id="t-vc",
            user_id="u-1",
        )
    ))
    # First N events are status_token (animated label).
    assert all(e["event"] == "status_token" for e in out if e["event"] != "block" and e["event"] != "done")
    # An stt_error block must appear.
    block_events = [e for e in out if e["event"] == "block"]
    assert len(block_events) == 1
    payload = json.loads(block_events[0]["data"])
    assert payload == {"type": "stt_error", "reason": "transcription_failed"}
    # done is last.
    assert out[-1]["event"] == "done"
    # Graph was NEVER called.
    assert graph.calls == []


# ---------------------------------------------------------------------------
# UT-VC03 — success emits transcript block BEFORE graph stream
# ---------------------------------------------------------------------------


def test_UT_VC03_success_emits_transcript_before_graph_stream() -> None:
    """UT-VC03: a successful transcription emits
    `event: block` {type: transcript, text: <heard>}` BEFORE the ReAct
    stream begins. We verify the order by recording the event index of
    the transcript block + checking the graph was called AFTER.
    """
    graph = _GraphYieldingDone()
    stt_call = _make_stt_call("เพิ่ม 50 ค่าน้ำ")

    out = asyncio.run(_collect(
        handle_voice_chat(
            graph=graph,
            stt_call=stt_call,
            audio_bytes=b"audio",
            thread_id="t-vc3",
            user_id="u-1",
        )
    ))
    # Locate transcript block.
    transcript_idx = next(
        i for i, e in enumerate(out)
        if e["event"] == "block"
        and json.loads(e["data"]).get("type") == "transcript"
    )
    payload = json.loads(out[transcript_idx]["data"])
    assert payload == {"type": "transcript", "text": "เพิ่ม 50 ค่าน้ำ"}

    # All preceding events are status tokens (animated label).
    for ev in out[:transcript_idx]:
        assert ev["event"] == "status_token"

    # Graph was called with the transcript as the user message.
    assert len(graph.calls) == 1
    init_state = graph.calls[0]["init_state"]
    assert init_state["messages"][0]["content"] == "เพิ่ม 50 ค่าน้ำ"
    assert init_state["messages"][0]["role"] == "user"
    # Voice carries no wallet pick → wallet_id is empty so
    # propose_transaction's index-0 fallback fires (memory:
    # project_wallet_index0_ordering).
    assert init_state["wallet_id"] == ""
    # Done is the last event (from the empty graph stream).
    assert out[-1]["event"] == "done"


def test_UT_VC03b_wallet_id_propagated_when_provided() -> None:
    """UT-VC03b: when the caller passes wallet_id, it propagates into
    init_state so propose_transaction sees the right input (caller may
    add wallet support to voice in a future release)."""
    graph = _GraphYieldingDone()
    stt_call = _make_stt_call("เพิ่ม 30 บาท")
    asyncio.run(_collect(
        handle_voice_chat(
            graph=graph,
            stt_call=stt_call,
            audio_bytes=b"x",
            thread_id="t",
            user_id="u",
            wallet_id="w-explicit",
        )
    ))
    assert graph.calls[0]["init_state"]["wallet_id"] == "w-explicit"
