"""Speech-to-transcript node for the voice-chat lane.

Runs before the rest of the graph when the client uploaded an audio
clip via `POST /chat/voice`. The transcript is injected back into the
latest HumanMessage so every downstream node — intent classifier,
quick-add, reason loop — sees the turn as if the user had typed it.
After the node finishes the audio bytes are dropped from state so the
PostgreSQL checkpointer never persists a multi-megabyte blob.

Why a dedicated node instead of inlining STT in `server.py`:
- Keeps the HTTP layer thin: it just shovels bytes into the graph and
  forwards SSE events out. All AI / model handling lives inside the
  graph the same way it does for vision (`slip_node`).
- Makes the transcript and failure modes visible as graph state, so a
  future test harness can drive the voice flow without HTTP at all.
- Uses the same `adispatch_custom_event` machinery as `slip_node`, so
  the SSE plumbing in `server.py` learns one event-name pattern
  instead of two.

Failure handling:
- Empty / whitespace transcript     → `stt_error = "no_speech"`
- Audio bytes ≲ 0.5s (≈4KB raw)     → `stt_error = "too_short"`
- Model exception / malformed reply → `stt_error = "transcription_failed"`

In production the heuristics are tolerated (Postel's Law per project
CLAUDE.md). In dev/test the same exceptions propagate so a broken
upload format fails the run loudly instead of silently emitting an
SSE error.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.config import settings
from src.debug_log import LogLevel as _LogLevel, log as _log
from src.graph.state import AgentState
from src.llm import stt_llm


def _stt_log(tag: str, msg: str, **kwargs: Any) -> None:
    level = kwargs.pop("_level", _LogLevel.MILESTONE)
    _log(tag, msg, level=level, **kwargs)


# Below this raw-byte threshold we assume the upload is not real speech
# — a 0.5s AAC clip at 64kbps is ~4KB. Tuned conservatively so genuine
# short utterances ("ใช่") still pass.
_TOO_SHORT_BYTES = 4 * 1024

# Hard upper bound enforced one more time defensively. The HTTP layer
# already rejects oversized uploads, but if a future caller bypasses
# that gate the model would still time out.
_MAX_BYTES = 6 * 1024 * 1024  # 6 MB


# Map of mime → format string accepted by the OpenAI-compatible
# `input_audio` content block. OpenRouter passes the format through to
# Gemini, which accepts mp3 / wav / flac / m4a / aac / ogg / webm.
_MIME_FORMAT_MAP = {
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


def _format_from_mime(mime: str | None) -> str:
    """Pick the `input_audio.format` value Gemini expects.

    In production (Postel's Law) an unknown mime falls back to "m4a"
    — that is the format the mobile client uploads. In dev/test an
    unknown mime raises so the caller catches the misconfiguration
    early.
    """
    if mime:
        normalized = mime.split(";")[0].strip().lower()
        if normalized in _MIME_FORMAT_MAP:
            return _MIME_FORMAT_MAP[normalized]

    if settings.is_production:
        _stt_log(
            "STT",
            "unknown audio mime — falling back to m4a (production)",
            mime=mime or "",
        )
        return "m4a"
    raise ValueError(
        f"stt_node: unsupported audio mime {mime!r} — "
        f"supported: {sorted(_MIME_FORMAT_MAP)}"
    )


_SYSTEM_PROMPT = (
    "You are a speech-to-text transcriber for a Thai personal-finance "
    "chat app. Listen to the attached audio and output the transcript "
    "in the language the speaker used — usually Thai. Output ONLY the "
    "transcript text, nothing else. No greetings, no markdown, no "
    "punctuation cleanup, no language tags. If the audio is silent or "
    "contains no speech, output an empty string."
)


async def stt_node(state: AgentState, config: RunnableConfig) -> dict:
    """Transcribe `state['audio_data']` with `stt_llm` then inject the
    text into the latest HumanMessage. Drops the audio bytes from
    state on exit so the checkpoint stays lean.
    """
    audio_data = state.get("audio_data")
    audio_mime = state.get("audio_mime")
    msgs = state.get("messages") or []

    if not audio_data:
        # Should never happen — the entry router only sends us here when
        # audio_data is truthy. Treat as a programmer error in dev; in
        # prod fall back to no_speech so the user gets a clean error.
        if settings.is_production:
            _stt_log("STT", "no audio_data — forcing no_speech (production)")
            return {
                "stt_error": "no_speech",
                "audio_data": None,
                "audio_mime": None,
            }
        raise ValueError("stt_node invoked without audio_data")

    raw_size = len(audio_data)
    _stt_log(
        "STT",
        "stt_node entered",
        bytes=raw_size,
        mime=audio_mime or "",
        msg_count=len(msgs),
    )

    if raw_size > _MAX_BYTES:
        # HTTP layer should have caught this; here we bail before
        # paying the model.
        _stt_log("STT", "audio exceeds max size — failing", bytes=raw_size)
        return _terminate_with_error(
            "transcription_failed",
            reason="oversize_audio",
            bytes=raw_size,
        )

    if raw_size < _TOO_SHORT_BYTES:
        _stt_log("STT", "audio below short threshold", bytes=raw_size)
        return await _dispatch_and_terminate("too_short", bytes=raw_size)

    # Format detection — Postel's Law applies inside `_format_from_mime`.
    try:
        audio_format = _format_from_mime(audio_mime)
    except ValueError as exc:
        _stt_log("STT", "format detection failed", error=str(exc))
        # In dev/test this re-raise surfaces the bug at the API edge.
        # We never want a silent mp3-mistaken-for-wav in prod, but the
        # production branch of `_format_from_mime` already handled it.
        raise

    b64_audio = base64.b64encode(audio_data).decode("ascii")
    content = [
        {
            "type": "text",
            "text": (
                "Transcribe the following audio in Thai. Output only the "
                "transcript text, nothing else."
            ),
        },
        {
            "type": "input_audio",
            "input_audio": {"data": b64_audio, "format": audio_format},
        },
    ]

    started = time.monotonic()
    try:
        response = await stt_llm.ainvoke(
            [HumanMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        _stt_log(
            "STT",
            "model call failed",
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return await _dispatch_and_terminate(
            "transcription_failed",
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    transcript = _extract_text(response).strip()
    _stt_log(
        "STT",
        "transcription complete",
        duration_ms=duration_ms,
        transcript_len=len(transcript),
        transcript_preview=transcript[:80].replace("\n", " "),
    )

    if not transcript:
        return await _dispatch_and_terminate(
            "no_speech",
            duration_ms=duration_ms,
        )

    # Surface the transcript over SSE BEFORE any downstream LLM token
    # streams. `server.py` maps `stt_transcript` → an SSE `data:` line
    # with `{"type": "transcript", "text": <transcript>}`.
    await adispatch_custom_event("stt_transcript", {"text": transcript})

    # Replace the latest HumanMessage's content with the transcript so
    # downstream nodes (intent classifier, quick-add, reason) see the
    # turn as if the user had typed the transcript. If no HumanMessage
    # exists yet (caller invoked the voice endpoint with an empty
    # messages list — only happens in tests), append a new one.
    new_messages: list = []
    replaced = False
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            # Use the same `id` so `add_messages` overwrites the
            # placeholder instead of appending a duplicate. LangChain
            # auto-generates an id when one is missing — preserve it.
            mid = getattr(m, "id", None)
            new_messages.append(
                HumanMessage(content=transcript, id=mid) if mid
                else HumanMessage(content=transcript)
            )
            replaced = True
            break
    if not replaced:
        new_messages.append(HumanMessage(content=transcript))

    return {
        "messages": new_messages,
        "transcript": transcript,
        "stt_error": None,
        # Drop the bytes — they served their purpose and have no place
        # in the persistent checkpoint.
        "audio_data": None,
        "audio_mime": None,
    }


async def stt_error_node(state: AgentState, config: RunnableConfig) -> dict:
    """Terminal node when STT failed.

    The `stt_error` reason is already set on state; this node exists
    only so the graph has somewhere clean to land before END (rather
    than abandoning a half-formed turn). The error event was already
    dispatched by `stt_node` via `_dispatch_and_terminate`. We just
    clear the audio fields and return.
    """
    reason = state.get("stt_error") or "transcription_failed"
    _stt_log("STT", "stt_error_node entered", reason=reason)
    return {
        "audio_data": None,
        "audio_mime": None,
    }


# ── helpers ──────────────────────────────────────────────────────────


def _extract_text(response: Any) -> str:
    """Pull plain text from an AIMessage with either str or block content."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text = blk.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _terminate_with_error(reason: str, **fields) -> dict:
    """Synchronous path — produce the state delta without dispatching
    an SSE event. Reserved for cases where dispatching makes no sense
    (e.g. oversize bytes — we already failed defensively)."""
    _stt_log("STT", "terminating with error", reason=reason, **fields)
    return {
        "stt_error": reason,
        "audio_data": None,
        "audio_mime": None,
    }


async def _dispatch_and_terminate(reason: str, **fields) -> dict:
    """Emit an `stt_error_event` to SSE and return the state delta
    that short-circuits the graph onto `stt_error_node`.
    """
    _stt_log("STT", "dispatching stt_error_event", reason=reason, **fields)
    await adispatch_custom_event("stt_error_event", {"reason": reason})
    return {
        "stt_error": reason,
        "audio_data": None,
        "audio_mime": None,
    }
