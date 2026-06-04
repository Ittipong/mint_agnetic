"""Regression tests for the codeact sandbox DSN resolution precedence.

Wave 7b — lifted from v2 (test_resolve_dsn.py). Import path adjusted from
`agent.codeact_subgraph.shared.db` (v2's sub-graph layout) to
`src.agent.tools.codeact.db` (v3 collapses the sub-graph into a tool).
The function body is unchanged across versions — semantics asserted here.

Bug history: `_resolve_dsn` once preferred `DATABASE_URL` (the agent's
checkpoint DB, which has NO financial tables) over `BACKEND_DATABASE_URL`
(the backend DB holding general_wallets/transactions/categories). The fix
prefers BACKEND_DATABASE_URL (memory `reference_agentic_v2_two_databases`).
"""

from __future__ import annotations

import pytest

from src.agent.tools.codeact.db import _resolve_dsn

_BACKEND = "postgresql://u:p@localhost:5432/mint_money_dev"
_AGENT = "postgresql://u:p@localhost:5432/mint_agentic"


def test_UT_DSN_001_prefers_backend_when_both_set(monkeypatch):
    """UT-DSN-001: both env set -> BACKEND_DATABASE_URL wins (the bug repro)."""
    monkeypatch.setenv("BACKEND_DATABASE_URL", _BACKEND)
    monkeypatch.setenv("DATABASE_URL", _AGENT)
    assert _resolve_dsn() == _BACKEND


def test_UT_DSN_002_falls_back_to_database_url(monkeypatch):
    """UT-DSN-002: only DATABASE_URL set -> used as fallback (single-DB dev)."""
    monkeypatch.delenv("BACKEND_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", _AGENT)
    assert _resolve_dsn() == _AGENT


def test_UT_DSN_003_uses_backend_when_only_backend_set(monkeypatch):
    """UT-DSN-003: only BACKEND_DATABASE_URL set -> used."""
    monkeypatch.setenv("BACKEND_DATABASE_URL", _BACKEND)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert _resolve_dsn() == _BACKEND


def test_UT_DSN_004_raises_when_neither_set(monkeypatch):
    """UT-DSN-004: neither set -> loud RuntimeError, not a silent bad pool."""
    monkeypatch.delenv("BACKEND_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        _resolve_dsn()
