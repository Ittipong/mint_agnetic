"""Deterministic helpers in `stt_node` — mime resolution and the
Postel's Law toggle. Keeps the model call out of CI.

The actual transcription path is exercised by the live voice endpoint
(see the curl snippet in the PR description) — anything involving the
network needs a real OpenRouter key and would burn tokens on every run.
"""

from __future__ import annotations

import pytest

from src.config import settings
from src.graph.stt_node import _format_from_mime, _MIME_FORMAT_MAP


# ── _format_from_mime — known mimes map straight through ────────────────────


@pytest.mark.parametrize(
    "mime,expected",
    [
        ("audio/m4a", "m4a"),
        ("audio/x-m4a", "m4a"),
        ("audio/mp4", "m4a"),
        ("audio/aac", "aac"),
        ("audio/mpeg", "mp3"),
        ("audio/mp3", "mp3"),
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),
        ("audio/ogg", "ogg"),
        ("audio/webm", "webm"),
        ("audio/flac", "flac"),
    ],
)
def test_format_from_mime_known(mime: str, expected: str):
    assert _format_from_mime(mime) == expected


def test_format_from_mime_strips_codec_suffix():
    # Some clients append `; codecs=...`. The resolver must ignore it.
    assert _format_from_mime("audio/webm; codecs=opus") == "webm"


def test_format_from_mime_case_insensitive():
    assert _format_from_mime("AUDIO/M4A") == "m4a"


# ── Postel's Law — unknown mime fails loudly in dev, falls back in prod ─────


def test_format_from_mime_unknown_raises_in_dev(monkeypatch):
    monkeypatch.setattr(settings, "environment", "development")
    with pytest.raises(ValueError, match="unsupported audio mime"):
        _format_from_mime("audio/wat-is-this")


def test_format_from_mime_unknown_falls_back_in_prod(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    # Production tolerates garbage mime by defaulting to the mobile
    # client's normal upload format. The warning is logged but nothing
    # raises — that's the whole point of Postel's Law on this path.
    assert _format_from_mime("audio/wat-is-this") == "m4a"


def test_format_from_mime_none_in_prod_returns_m4a(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    assert _format_from_mime(None) == "m4a"


def test_format_from_mime_none_in_dev_raises(monkeypatch):
    monkeypatch.setattr(settings, "environment", "development")
    with pytest.raises(ValueError):
        _format_from_mime(None)


# ── Map sanity — every value is a single lowercase token Gemini accepts ─────


def test_mime_map_values_are_lowercase_tokens():
    for fmt in _MIME_FORMAT_MAP.values():
        assert fmt == fmt.lower()
        assert " " not in fmt
        assert "/" not in fmt
