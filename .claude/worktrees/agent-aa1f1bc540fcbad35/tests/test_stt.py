"""STT pipeline unit tests (Wave 7b adaptation).

Adapted from v2 `tests/test_stt.py`. In v3 the STT lives at
`src.agent.endpoints.voice_handler` — `run_stt` was renamed to
`transcribe_audio` and the helpers are private (`_build_stt_messages`,
`_clean_transcript`). These tests assert the SPEC of those helpers and the
transcribe_audio outcome map; the streaming/endpoint integration is covered
by `test_voice_handler.py` (Wave 5).

Overlap notes vs `test_voice_handler.py`:
- UT-STT-10 success transcript -> UT-VC01 already covers (kept here as a
  unit-level sanity check on transcribe_audio's own contract).
- UT-STT-11/12 no_speech sentinel + empty -> UT-VC02/02b.
- UT-STT-13 transport failure -> UT-VC01c.

This file ADDS:
- UT-STT-01: _build_stt_messages carries audio part with format passthrough.
- UT-STT-02: _clean_transcript strips one wrapping quote pair.
"""

from __future__ import annotations

import pytest

from src.agent.endpoints.voice_handler import (
    NO_SPEECH_SENTINEL,
    STTResult,
    _build_stt_messages,
    _clean_transcript,
    transcribe_audio,
)


def _stt(reply: str):
    async def call(_messages):
        return reply
    return call


def _stt_raises(exc: Exception):
    async def call(_messages):
        raise exc
    return call


# ---------------------------------------------------------------------------
# message builder + transcript cleaning
# ---------------------------------------------------------------------------
def test_ut_stt_01_build_messages_carries_input_audio():
    """UT-STT-01: the user turn carries an input_audio part with the base64
    data + container format the multimodal client forwards unchanged."""
    msgs = _build_stt_messages(audio_b64="QUJD", audio_format="m4a")
    assert msgs[0]["role"] == "system"
    audio_parts = [
        p for p in msgs[1]["content"] if p.get("type") == "input_audio"
    ]
    assert len(audio_parts) == 1
    assert audio_parts[0]["input_audio"] == {"data": "QUJD", "format": "m4a"}


def test_ut_stt_02_clean_transcript_strips_one_quote_pair():
    """UT-STT-02: a single wrapping pair of quotes/backticks is stripped; inner
    text and un-paired quotes are left intact."""
    assert _clean_transcript('  "กินข้าว 200 บาท"  ') == "กินข้าว 200 บาท"
    assert _clean_transcript("`เติมน้ำมัน`") == "เติมน้ำมัน"
    assert _clean_transcript("กาแฟ 60") == "กาแฟ 60"


# ---------------------------------------------------------------------------
# transcribe_audio outcomes (unit-level — endpoint streaming covered by VC tests)
# ---------------------------------------------------------------------------
# v3 transcribe_audio takes raw bytes (it base64-encodes internally); a
# 1-byte payload is enough to bypass the empty-bytes guard so the stub stt_call
# fires and we exercise the cleaning + sentinel logic.
_AUDIO = b"\x01"


@pytest.mark.asyncio
async def test_ut_stt_10_success_returns_transcript():
    """UT-STT-10: a real transcript comes back as text, no error."""
    res = await transcribe_audio(_stt("กินข้าว 200 บาท"), audio_bytes=_AUDIO)
    assert res == STTResult(text="กินข้าว 200 บาท")


@pytest.mark.asyncio
async def test_ut_stt_11_no_speech_sentinel_maps_to_no_speech():
    """UT-STT-11: the model's NO_SPEECH sentinel -> error_reason 'no_speech',
    never a garbage transcript."""
    res = await transcribe_audio(_stt(NO_SPEECH_SENTINEL), audio_bytes=_AUDIO)
    assert res.text is None and res.error_reason == "no_speech"


@pytest.mark.asyncio
async def test_ut_stt_12_empty_reply_maps_to_no_speech():
    """UT-STT-12: an empty/whitespace transcript -> 'no_speech'."""
    res = await transcribe_audio(_stt("   "), audio_bytes=_AUDIO)
    assert res.text is None and res.error_reason == "no_speech"


@pytest.mark.asyncio
async def test_ut_stt_13_model_exception_maps_to_transcription_failed():
    """UT-STT-13: a transport/model exception -> 'transcription_failed' (the
    caller shows the voice-error UI instead of a 500), never raised."""
    res = await transcribe_audio(
        _stt_raises(RuntimeError("upstream 503")), audio_bytes=_AUDIO,
    )
    assert res.text is None and res.error_reason == "transcription_failed"


@pytest.mark.asyncio
async def test_ut_stt_14_empty_audio_bytes_short_circuits_to_no_speech():
    """UT-STT-14: empty audio bytes -> no_speech without calling the model
    (cost-saver). The STT stub must NOT be invoked."""
    calls = {"n": 0}

    async def call(_messages):
        calls["n"] += 1
        return "ignored"

    res = await transcribe_audio(call, audio_bytes=b"")
    assert res.text is None and res.error_reason == "no_speech"
    assert calls["n"] == 0
