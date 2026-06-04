"""Unit tests for src.agent.tools.codeact.db.

Covers UT-DB01 — the pool MUST target the financial-data DB
(`BACKEND_DATABASE_URL` -> `mint_money_dev`), NOT the agent's own
checkpoint/thread DB (`DATABASE_URL` -> `mint_agentic`).

Per memory `reference_agentic_v2_two_databases`: getting this wrong sends
every codeact query to a DB with no financial tables -> `UndefinedTableError`.

Strategy: monkey-patch env vars + the asyncpg.create_pool factory so we
assert what DSN the pool *would* open against, WITHOUT actually opening
a network connection.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agent.tools.codeact import db as codeact_db


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _reset_pool_state():
    """Force the lazy pool back to uninitialized between tests so each test
    re-runs the env-resolution path. The module caches `_pool` globally."""
    codeact_db._pool = None


class _FakePool:
    """A stand-in for asyncpg.Pool that records its init kwargs."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def close(self):
        self.closed = True


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveDSN:
    def test_UT_DB01_a_prefers_backend_database_url(self, monkeypatch):
        """UT-DB01a: when BOTH env vars are set, _resolve_dsn() returns the
        BACKEND_DATABASE_URL — financial data lives there."""
        monkeypatch.setenv(
            "BACKEND_DATABASE_URL",
            "postgresql://app:pw@localhost:5432/mint_money_dev",
        )
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://app:pw@localhost:5432/mint_agentic",
        )
        dsn = codeact_db._resolve_dsn()
        assert "mint_money_dev" in dsn
        assert "mint_agentic" not in dsn

    def test_UT_DB01_b_falls_back_to_database_url_when_backend_unset(self, monkeypatch):
        """UT-DB01b: dev/test single-DB setups (only DATABASE_URL set) still work."""
        monkeypatch.delenv("BACKEND_DATABASE_URL", raising=False)
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://app:pw@localhost:5432/mint_agentic",
        )
        dsn = codeact_db._resolve_dsn()
        assert "mint_agentic" in dsn

    def test_UT_DB01_c_raises_when_neither_set(self, monkeypatch):
        """UT-DB01c: pool init must fail LOUD when no DSN is configured —
        silent fallback hides config bugs."""
        monkeypatch.delenv("BACKEND_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            codeact_db._resolve_dsn()

    def test_UT_DB01_d_normalizes_sqlalchemy_form(self, monkeypatch):
        """UT-DB01d: asyncpg rejects `postgresql+asyncpg://`; _normalize_dsn
        strips the `+asyncpg` driver suffix."""
        monkeypatch.setenv(
            "BACKEND_DATABASE_URL",
            "postgresql+asyncpg://u:p@h/mint_money_dev",
        )
        monkeypatch.delenv("DATABASE_URL", raising=False)
        dsn = codeact_db._resolve_dsn()
        assert dsn.startswith("postgresql://")
        assert "+asyncpg" not in dsn


class TestGetPool:
    def test_UT_DB01_e_opens_against_backend_url(self, monkeypatch):
        """UT-DB01e: get_pool() actually instantiates the pool with the
        BACKEND_DATABASE_URL DSN — proves the resolve path is wired into pool
        creation, not just exposed as a helper."""
        _reset_pool_state()
        captured: dict = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return _FakePool(**kwargs)

        monkeypatch.setattr(codeact_db.asyncpg, "create_pool", fake_create_pool)
        monkeypatch.setenv(
            "BACKEND_DATABASE_URL",
            "postgresql://app:pw@localhost:5432/mint_money_dev",
        )
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://app:pw@localhost:5432/mint_agentic",
        )
        pool = asyncio.run(codeact_db.get_pool())
        try:
            assert isinstance(pool, _FakePool)
            assert "mint_money_dev" in captured["dsn"]
            assert "mint_agentic" not in captured["dsn"]
            # Sandbox-side timeout + size limits carried over from v2.
            assert captured["command_timeout"] == 15
            assert captured["max_size"] == 5
        finally:
            asyncio.run(codeact_db.close_pool())
            _reset_pool_state()
