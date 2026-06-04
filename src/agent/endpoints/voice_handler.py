"""Voice endpoint pre-processor — Gemini STT, then ReAct.

NEW (TRANSFORM) in Wave 5. Source: v2 `stt.py` for the Gemini STT call.

v3 deviation from v2:
  - The merged STT+classify fast path (v2 `voice_understand.py`) is DROPPED
    per memory `project_voice_fast_path` and the Wave 5 kickoff: pure ReAct
    does its own classification implicitly through tool calls, so there's
    no `UnderstandResult` to pre-build. One STT round-trip + the ReAct loop
    is faster than v2's merged path turned out to be in practice (since
    the merged path still produced no proposal; ReAct still had to run).

Mobile contract (frozen, identical to v2 voice flow):
  1. `event: status_token` "กำลังถอดเสียง..." (one word at a time)
  2. EITHER:
     - `event: block` {"type":"stt_error","reason":"<code>"} + `event: done`
       on transcription failure (mobile shows retry UI), OR
     - `event: block` {"type":"transcript","text":"<heard>"} (heard-text
       bubble) followed by the normal ReAct stream from `stream_chat`.

m4a passthrough — Gemini STT (`STT_MODEL`) transcribes AAC/m4a directly per
memory `project_voice_stt_real`; we never transcode. OpenRouter needs >=
$0.50 balance for audio routing — that's an env/billing concern, not code.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from src.agent.preamble import astream_with_preamble
from src.agent.session_logger import slog, slog_error
from src.agent.streaming.sse_adapter import stream_chat


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Exact sentinel the model emits when the audio carries no intelligible
# speech (matches v2 `stt.py::NO_SPEECH_SENTINEL`). Keep ASCII so it never
# collides with a real Thai transcript.
NO_SPEECH_SENTINEL = "NO_SPEECH"

# Thai status word streamed at turn start so mobile shows an animated label
# while STT runs. Each space-separated word becomes one `status_token`
# event — v2's `_stream_status_tokens` convention preserved.
_STATUS_TRANSCRIBING = "กำลังถอดเสียง..."

_STT_SYSTEM_PROMPT = (
    "You are a speech-to-text engine for a Thai personal-finance app. "
    "Transcribe the user's spoken audio to text VERBATIM in Thai.\n"
    "Rules:\n"
    "- Output ONLY the transcript text — no translation, no explanation, "
    "no quotes, no labels.\n"
    '- Keep numbers and amounts exactly as spoken (e.g. "200 บาท").\n'
    "- Do not invent words; transcribe only what is actually said.\n"
    f"- If the audio has no intelligible speech, output exactly: {NO_SPEECH_SENTINEL}"
)


# ---------------------------------------------------------------------------
# 2. STT result + Gemini call
# ---------------------------------------------------------------------------


@dataclass
class STTResult:
    """Outcome of one STT call.

    On success: `text` is the cleaned transcript; `error_reason` is None.
    On failure: `text` is None; `error_reason` is one of the codes mobile
    voice UI understands (`no_speech` | `too_short` | `transcription_failed`).
    """

    text: Optional[str] = None
    error_reason: Optional[str] = None


def _build_stt_messages(audio_b64: str, audio_format: str = "m4a") -> list[dict]:
    """OpenAI-compatible multimodal messages list for one STT call.

    `input_audio.format` tells the provider how to interpret the bytes;
    Gemini handles m4a/AAC natively (no transcode needed).
    """
    return [
        {"role": "system", "content": _STT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ถอดข้อความจากเสียงนี้"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": audio_format},
                },
            ],
        },
    ]


def _clean_transcript(raw: str) -> str:
    """Strip wrapping quotes/backticks the model sometimes adds despite the
    'output only the transcript' instruction."""
    t = (raw or "").strip()
    for q in ('"', "'", "`"):
        if len(t) >= 2 and t.startswith(q) and t.endswith(q):
            return t[1:-1].strip()
    return t


async def transcribe_audio(
    stt_call: Callable[[list[dict]], Awaitable[str]],
    *,
    audio_bytes: bytes,
    audio_format: str = "m4a",
) -> STTResult:
    """Transcribe one voice note via the injected Gemini callable.

    Empty bytes → no_speech (don't waste a model call). A transport / model
    failure raised by `stt_call` is mapped to `transcription_failed` so the
    caller can show the mobile voice-error UI rather than surfacing an HTTP
    500.

    `stt_call` is `make_multimodal_call("stt")` in production; tests pass a
    stub that returns a fixed string.
    """
    if not audio_bytes:
        slog("voice_handler", "empty audio payload → no_speech")
        return STTResult(error_reason="no_speech")

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    try:
        raw = await stt_call(_build_stt_messages(audio_b64, audio_format))
    except Exception as e:  # noqa: BLE001 — any STT failure surfaces friendly
        slog_error("voice_handler", e)
        return STTResult(error_reason="transcription_failed")

    text = _clean_transcript(raw)
    if not text or text.upper().startswith(NO_SPEECH_SENTINEL):
        slog("voice_handler", "no intelligible speech")
        return STTResult(error_reason="no_speech")
    slog("voice_handler", f"transcript: {text!r}")
    return STTResult(text=text)


# ---------------------------------------------------------------------------
# 3. SSE generator — orchestrates the voice flow
# ---------------------------------------------------------------------------


async def _stream_status_tokens(text: str) -> AsyncIterator[dict]:
    """Yield one `status_token` event per space-separated word.

    Mirrors v2 `server.py::_stream_status_tokens` (~line 350); the mobile
    cubit accumulates words into the running status label so multi-word
    Thai status strings animate naturally.
    """
    for word in text.split():
        yield {"event": "status_token", "data": word}


def _emit_stt_error_block(reason: str) -> dict:
    """One `event: block` dict carrying the voice-only stt_error payload."""
    return {
        "event": "block",
        "data": json.dumps(
            {"type": "stt_error", "reason": reason},
            ensure_ascii=False,
        ),
    }


def _emit_transcript_block(text: str) -> dict:
    """One `event: block` dict carrying the voice-only transcript payload."""
    return {
        "event": "block",
        "data": json.dumps(
            {"type": "transcript", "text": text},
            ensure_ascii=False,
        ),
    }


def _emit_done(thread_id: str) -> dict:
    """Terminal done event."""
    return {"event": "done", "data": json.dumps({"thread_id": thread_id})}


async def handle_voice_chat(
    *,
    graph: Any,
    stt_call: Callable[[list[dict]], Awaitable[str]],
    audio_bytes: bytes,
    thread_id: str,
    user_id: str,
    audio_format: str = "m4a",
    wallet_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Orchestrate one voice turn: STT -> transcript block -> ReAct stream.

    Event sequence (mobile contract — frozen):
      status_token* -> [stt_error + done] OR
      status_token* -> transcript -> (ReAct events from stream_chat)

    The ReAct events include answer_token + block + done; the caller does
    NOT need to add its own `done` after a successful transcribe — stream_chat
    emits it. Only the stt_error path appends a manual `done`.

    `stt_call` is injected so tests can mock without an OpenRouter dependency;
    Wave 6's server.py wires `make_multimodal_call("stt")` here.

    `wallet_id` is the mobile-selected wallet (None for voice — voice carries
    no wallet selection per v2 contract). Propagated into initial_state so
    propose_transaction's index-0 cascade sees the right input.
    """
    # 1. Status — animated label while STT runs.
    async for ev in _stream_status_tokens(_STATUS_TRANSCRIBING):
        yield ev

    # 2. STT.
    result = await transcribe_audio(
        stt_call, audio_bytes=audio_bytes, audio_format=audio_format
    )
    if not result.text:
        # 2a. Transcription failed → emit stt_error block + done; skip ReAct.
        reason = result.error_reason or "transcription_failed"
        slog("voice_handler", f"STT failure → {reason}")
        yield _emit_stt_error_block(reason)
        yield _emit_done(thread_id)
        return

    # 3. Transcript bubble — mobile renders the heard text BEFORE the
    # proposal/answer stream so the user can see what STT heard.
    yield _emit_transcript_block(result.text)

    # 4. Drive the ReAct graph with the transcript as the user message,
    # reusing the text-path streaming adapter. The transcript becomes the
    # message content; voice carries no wallet, so wallet_id is left empty
    # for propose_transaction's index-0 cascade.
    init_state: dict = {
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [{"role": "user", "content": result.text}],
        # Per-turn append channels — pre_turn_hook clears these, but seeding
        # them explicitly keeps the test fixtures simple.
        "tool_outputs_this_turn": [],
        "emitted_blocks_this_turn": [],
        "user_context": None,
        # Voice convention: no client-side wallet pick. The ADD cascade
        # falls back to index-0 (memory `project_wallet_index0_ordering`).
        "wallet_id": wallet_id or "",
    }
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    # Streaming preamble — same "thinking out loud" UX as the text branch
    # (server.py::_text_graph_stream), now racing the ReAct loop CONCURRENTLY
    # via astream_with_preamble instead of awaiting it ahead of stream_chat.
    # A slow/dead preamble model used to add its whole timeout to the turn
    # (observed: an 8s ReadTimeout stalling ReAct); now the preamble is emitted
    # as ONE ephemeral `status_token` only if it wins the race against the first
    # answer token, and is dropped + cancelled the moment the answer starts or
    # the stream ends. Gated by `PREAMBLE_ENABLED` (default on) inside the
    # helper. The status_token lands in the mobile typing label and vanishes
    # when the real answer bubble starts, so it never sticks before the answer.
    async for ev in astream_with_preamble(
        result.text,
        stream_chat(graph, init_state, config, thread_id=thread_id),
        event_type="status_token",
    ):
        yield ev


__all__ = [
    "NO_SPEECH_SENTINEL",
    "STTResult",
    "transcribe_audio",
    "handle_voice_chat",
]
