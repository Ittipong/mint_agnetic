"""Unit tests for src/billing/cost_extractor.py.

Pure-function tests — no network, no fixtures. Safety net 1
(generation API) is covered separately because it requires monkey-
patching httpx; we add one such test with a mock to lock in the
contract.
"""

from __future__ import annotations

import pytest

from src.billing import cost_extractor


def test_extract_cost_uses_usage_cost_when_present():
    out = cost_extractor.extract_cost(
        {"cost": 0.005, "prompt_tokens": 100, "completion_tokens": 50},
        model="anything/model",
    )
    assert out.cost_usd_micro == 5_000
    assert out.prompt_tokens == 100
    assert out.completion_tokens == 50
    assert out.source == "usage_cost"
    assert out.is_billable


def test_extract_cost_zero_cost_is_billable_when_tokens_present():
    # Free model — still billable for audit purposes (cost == 0 but
    # tokens > 0 means the call really happened).
    out = cost_extractor.extract_cost(
        {"cost": 0, "prompt_tokens": 10, "completion_tokens": 0},
        model="free/model",
    )
    assert out.cost_usd_micro == 0
    assert out.is_billable


def test_extract_cost_falls_back_to_local_pricing_table():
    # No `cost` field → safety net 2 (we don't pass generation_id so
    # safety net 1 is skipped).
    out = cost_extractor.extract_cost(
        {"prompt_tokens": 1000, "completion_tokens": 1000},
        model="google/gemini-2.5-flash",
    )
    # 1000 prompt @ 0.0001/1k + 1000 completion @ 0.0004/1k = 0.0005 USD
    assert out.cost_usd_micro == 500
    assert out.source == "fallback_table"


def test_extract_cost_unknown_model_returns_none():
    # No `cost`, no generation_id, no fallback row → source='none'.
    out = cost_extractor.extract_cost(
        {"prompt_tokens": 100, "completion_tokens": 50},
        model="totally-made-up/model",
    )
    assert out.cost_usd_micro == 0
    assert out.source == "none"
    # Tokens > 0 → still billable for audit (Go logs the call even at zero cost).
    assert out.is_billable


def test_extract_cost_empty_usage_meta_is_not_billable():
    out = cost_extractor.extract_cost(None, model="anything")
    assert out.cost_usd_micro == 0
    assert out.prompt_tokens == 0
    assert out.completion_tokens == 0
    assert out.source == "none"
    assert not out.is_billable


def test_extract_cost_negative_cost_clamped_to_zero():
    # Defensive: providers should never send negative cost but we
    # clamp instead of asserting so a glitch doesn't crash chat.
    out = cost_extractor.extract_cost(
        {"cost": -0.001, "prompt_tokens": 10, "completion_tokens": 5},
        model="any",
    )
    assert out.cost_usd_micro == 0
    # When cost is negative we fall through to net 2 / none; tokens
    # remain billable for audit.
    assert out.is_billable


def test_extract_cost_generation_api_path(monkeypatch: pytest.MonkeyPatch):
    """Safety net 1 — when inline usage has no cost AND a generation_id
    is present, we call OpenRouter's /api/v1/generation."""

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": {"total_cost": 0.0042}}

    def _fake_get(url, params=None, headers=None, timeout=None):
        assert "generation" in url
        assert params["id"] == "gen_xyz"
        return _FakeResp()

    monkeypatch.setattr(cost_extractor.httpx, "get", _fake_get)
    monkeypatch.setattr(cost_extractor.settings, "openrouter_api_key", "fake-key", raising=False)

    out = cost_extractor.extract_cost(
        {"prompt_tokens": 100, "completion_tokens": 50},
        model="anything",
        generation_id="gen_xyz",
    )
    # 0.0042 USD → 4200 micro
    assert out.cost_usd_micro == 4_200
    assert out.source == "generation_api"
