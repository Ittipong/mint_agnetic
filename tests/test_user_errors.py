"""UT-UE — user-facing error events + ops alerts (streaming/user_errors.py)."""

from __future__ import annotations

import json

import pytest

from src.agent.streaming import user_errors as ue

_CREDITS_TEXT = (
    "Error code: 402 - {'error': {'message': 'Insufficient credits. Add more using "
    "https://openrouter.ai/settings/credits', 'code': 402}}"
)


@pytest.fixture(autouse=True)
def _fresh_alerts(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ue, "_last_alert", {})
    yield tmp_path


def test_UT_UE01_credit_exhaustion_never_reaches_the_user(_fresh_alerts):
    """UT-UE01: when OpenRouter ran out of credits, users saw "Error code: 402 -
    …Insufficient credits. Add more using https://openrouter.ai/…" in the chat.
    The event must carry a stable code + a Thai message, and ops must be told."""
    ev = ue.error_event(RuntimeError(_CREDITS_TEXT), "sse_adapter")
    payload = json.loads(ev["data"])
    assert payload == {"code": "llm_unavailable",
                       "message": ue.USER_MESSAGES[ue.ErrorKind.LLM_CREDITS]}
    assert "openrouter" not in ev["data"].lower() and "402" not in ev["data"]
    alerts = (_fresh_alerts / "ops_alerts.log").read_text(encoding="utf-8").splitlines()
    assert len(alerts) == 1 and json.loads(alerts[0])["kind"] == "llm_unavailable"


def test_UT_UE02_alerts_are_throttled_per_kind(_fresh_alerts):
    """UT-UE02: a credit outage hits every turn — one alert per window, not per turn."""
    for _ in range(5):
        ue.note_llm_failure(RuntimeError(_CREDITS_TEXT), "llm.classify")
    ue.note_llm_failure(RuntimeError("Error code: 429 rate limit"), "llm.react")
    kinds = [json.loads(l)["kind"] for l in
             (_fresh_alerts / "ops_alerts.log").read_text(encoding="utf-8").splitlines()]
    assert kinds == ["llm_unavailable", "llm_busy"]


@pytest.mark.parametrize("exc, kind", [
    (RuntimeError(_CREDITS_TEXT), ue.ErrorKind.LLM_CREDITS),
    (RuntimeError("Client error '402 Payment Required'"), ue.ErrorKind.LLM_CREDITS),
    (RuntimeError("Error code: 429 - rate limit exceeded"), ue.ErrorKind.LLM_RATE_LIMITED),
    (RuntimeError("Recursion limit of 25 reached"), ue.ErrorKind.AGENT_LOOP),
    (TimeoutError("read timed out"), ue.ErrorKind.TIMEOUT),
    (RuntimeError("amount 4020 invalid"), ue.ErrorKind.INTERNAL),  # 4020 is not HTTP 402
])
def test_UT_UE03_classify_error(exc, kind):
    assert ue.classify_error(exc) == kind


def test_UT_UE04_cause_chain_is_inspected():
    """A wrapped provider error (raise X from openrouter_err) still maps right."""
    try:
        try:
            raise RuntimeError(_CREDITS_TEXT)
        except RuntimeError as inner:
            raise ValueError("graph step failed") from inner
    except ValueError as outer:
        assert ue.classify_error(outer) == ue.ErrorKind.LLM_CREDITS


def test_UT_UE05_non_alert_kinds_stay_quiet(_fresh_alerts):
    ue.error_event(RuntimeError("psycopg connection closed"), "server")
    assert not (_fresh_alerts / "ops_alerts.log").exists()
