"""Unit tests for `src.agent.utils.version_router`.

Covers UT-VR01..VR03 from `docs/v3/phase3_implementation_plan.md` §3.

The router is pure-env-read with a one-shot warning gate, so tests just
monkeypatch CHAT_AGENT_VERSION and assert the public API + reset the
one-shot warned flag between cases so the warning path is exercised.
"""

from __future__ import annotations

import logging

import pytest

from src.agent.utils import version_router
from src.agent.utils.version_router import (
    REFUSE_REASON,
    get_chat_version,
    is_v3_active,
)


@pytest.fixture(autouse=True)
def _reset_warned_flag():
    """The malformed-value warning is one-shot per process — reset it so
    each test starts clean and UT-VR03 can verify the log path triggers."""
    version_router._WARNED["emitted"] = False
    yield
    version_router._WARNED["emitted"] = False


# ---------------------------------------------------------------------------
# UT-VR01 — env unset returns default
# ---------------------------------------------------------------------------


def test_UT_VR01_env_unset_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-VR01: with CHAT_AGENT_VERSION unset, `get_chat_version()` returns
    the safe default `"v2"`. An unconfigured deployment must NOT silently
    flip to v3 — the v3 server would then refuse to boot, which is the
    explicit guarantee."""
    monkeypatch.delenv("CHAT_AGENT_VERSION", raising=False)
    assert get_chat_version() == "v2"
    assert is_v3_active() is False


def test_UT_VR01b_env_empty_string_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-VR01b: empty string is treated the same as unset (no spurious
    warning, falls back to default)."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "")
    assert get_chat_version() == "v2"
    assert is_v3_active() is False


# ---------------------------------------------------------------------------
# UT-VR02 — v3 activates this codebase
# ---------------------------------------------------------------------------


def test_UT_VR02_v3_activates(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-VR02: with CHAT_AGENT_VERSION=v3, the router reports v3 and
    `is_v3_active()` is True. This is the ONLY value that activates this
    server (match v2 deployment convention exactly — lowercase, no padding)."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "v3")
    assert get_chat_version() == "v3"
    assert is_v3_active() is True


def test_UT_VR02b_v2_explicit_returns_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-VR02b: an explicit `v2` is honored (operator-controlled fallback
    during cutover) and does NOT trigger the malformed-value warning."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "v2")
    assert get_chat_version() == "v2"
    assert is_v3_active() is False


# ---------------------------------------------------------------------------
# UT-VR03 — invalid value falls back to default + logs once
# ---------------------------------------------------------------------------


def test_UT_VR03_invalid_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """UT-VR03: a typo'd value (`v4`, `V3`, padded ` v3 `) falls back to the
    safe default AND emits a one-shot warning so the operator can correct
    the misconfiguration. The default protects against silent activation of
    the wrong codebase."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "v4")
    with caplog.at_level(logging.WARNING, logger="src.agent.utils.version_router"):
        assert get_chat_version() == "v2"
        assert is_v3_active() is False
    # The warning fired exactly once with the bad value visible so an
    # operator can grep their boot log and see the typo.
    matching = [r for r in caplog.records if "v4" in r.getMessage()]
    assert len(matching) == 1, (
        f"expected exactly one warning containing 'v4', got {len(matching)}: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_UT_VR03b_case_sensitive_uppercase_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-VR03b: uppercase `V3` is NOT honored — convention requires
    lowercase. This guard catches a common copy-paste mistake."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "V3")
    assert get_chat_version() == "v2"


def test_UT_VR03c_warning_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """UT-VR03c: the malformed-value warning fires AT MOST once per process
    so repeated calls per request don't spam the log."""
    monkeypatch.setenv("CHAT_AGENT_VERSION", "wrong")
    with caplog.at_level(logging.WARNING, logger="src.agent.utils.version_router"):
        for _ in range(5):
            assert get_chat_version() == "v2"
    matching = [r for r in caplog.records if "wrong" in r.getMessage()]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Refuse-reason constant — covered here because every consumer (lifespan,
# diagnostics) references the same string. If the operator-facing message
# regresses, this test catches it.
# ---------------------------------------------------------------------------


def test_REFUSE_REASON_mentions_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refuse reason names CHAT_AGENT_VERSION explicitly so an operator
    reading the lifespan error message knows which knob to flip."""
    assert "CHAT_AGENT_VERSION" in REFUSE_REASON
    assert "v3" in REFUSE_REASON
