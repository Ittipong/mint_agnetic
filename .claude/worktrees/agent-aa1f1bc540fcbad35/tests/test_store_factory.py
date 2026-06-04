"""Unit tests for src.agent.utils.store_factory.

Covers UT-SF01. The factory's primary job is graceful degradation — when
DATABASE_URL is unset OR the postgres extras are missing, it returns None
so the rest of the agent boots without a long-term-memory store.

We never open a real Postgres connection here. The "AsyncPostgresStore
opened with embed index" sanity check uses a monkeypatched stub so the test
asserts the factory CONSTRUCTS the store correctly without paying network
or DDL cost.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from src.agent.utils import store_factory


# ─────────────────────────────────────────────────────────────────────────────
# UT-SF01 — AsyncPostgresStore opened with embed index
# ─────────────────────────────────────────────────────────────────────────────


class _FakeStore:
    """Stand-in for AsyncPostgresStore — records the index config it sees."""

    def __init__(self) -> None:
        self.setup_called = False

    async def setup(self) -> None:
        self.setup_called = True


class _FakeAsyncCM:
    """Async context manager that yields a single _FakeStore instance."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def __aenter__(self) -> _FakeStore:
        return self._store

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeAsyncPostgresStore:
    """Stand-in module surface for AsyncPostgresStore.from_conn_string.

    Records every call so the test can assert the DSN and index config the
    factory passed in.
    """

    calls: list[dict[str, Any]] = []

    @classmethod
    def from_conn_string(cls, dsn: str, *, index=None, **kwargs):
        cls.calls.append({"dsn": dsn, "index": index, **kwargs})
        return _FakeAsyncCM(_FakeStore())


def test_UT_SF01_builds_store_with_embed_index(monkeypatch):
    """UT-SF01: build_store(dsn=..., embed="openai:text-embedding-3-small")
    MUST construct an AsyncPostgresStore via `from_conn_string` and pass
    `index={"dims": 1536, "embed": "openai:text-embedding-3-small"}` so
    semantic recall works.

    We monkeypatch the import so no real Postgres connection opens.
    """
    _FakeAsyncPostgresStore.calls.clear()

    # Patch the LANGGRAPH import the factory does lazily inside build_store.
    fake_module = type(
        "fake_aio", (),
        {"AsyncPostgresStore": _FakeAsyncPostgresStore},
    )
    import sys
    sys.modules["langgraph.store.postgres.aio"] = fake_module

    try:
        async def run():
            store = await store_factory.build_store(
                dsn="postgresql://u:p@h/db?options=-c%20search_path%3Dv3",
            )
            return store

        store = asyncio.run(run())

        # 1. We got a usable store (not None).
        assert store is not None, "build_store should not return None when DSN given"
        # 2. The store's setup() was called.
        assert isinstance(store, _FakeStore)
        assert store.setup_called is True
        # 3. The factory passed the DSN through unchanged (search_path lives
        # there per Q1, no second transformation).
        assert len(_FakeAsyncPostgresStore.calls) == 1
        call = _FakeAsyncPostgresStore.calls[0]
        assert "search_path%3Dv3" in call["dsn"]
        # 4. The embed index is wired with the expected defaults.
        assert call["index"] == {
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    finally:
        sys.modules.pop("langgraph.store.postgres.aio", None)


def test_UT_SF01_returns_none_when_dsn_missing(monkeypatch):
    """UT-SF01 supporting: no DATABASE_URL + no `dsn=` arg → return None.
    Long-term memory is OPTIONAL — `memory_recall` / `memory_write` already
    gate on a None store. Boot must not crash."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    async def run():
        return await store_factory.build_store()

    out = asyncio.run(run())
    assert out is None


def test_UT_SF01_returns_none_on_import_error(monkeypatch):
    """UT-SF01 supporting: when `langgraph.store.postgres.aio` fails to
    import (e.g. extras not installed), return None. We force the import to
    fail by inserting a sentinel that raises on attribute access."""
    import sys

    class _BrokenModule:
        def __getattr__(self, name):
            raise ImportError(f"simulated missing extras for {name}")

    sys.modules["langgraph.store.postgres.aio"] = _BrokenModule()
    try:
        async def run():
            return await store_factory.build_store(dsn="postgresql://x/y")

        out = asyncio.run(run())
        assert out is None
    finally:
        sys.modules.pop("langgraph.store.postgres.aio", None)


def test_UT_SF01_embed_none_disables_index(monkeypatch):
    """UT-SF01 supporting: passing `embed=None` (explicit) skips the index
    config — store still constructs, just without semantic search."""
    _FakeAsyncPostgresStore.calls.clear()
    import sys
    fake_module = type(
        "fake_aio", (),
        {"AsyncPostgresStore": _FakeAsyncPostgresStore},
    )
    sys.modules["langgraph.store.postgres.aio"] = fake_module

    try:
        async def run():
            return await store_factory.build_store(
                dsn="postgresql://u/db", embed="",
            )

        store = asyncio.run(run())
        assert store is not None
        assert _FakeAsyncPostgresStore.calls[0]["index"] is None
    finally:
        sys.modules.pop("langgraph.store.postgres.aio", None)


def test_UT_SF01_redact_dsn_hides_password():
    """Defensive: `_redact_dsn` must hide passwords in logged DSNs."""
    redacted = store_factory._redact_dsn(
        "postgresql://user:secret@host:5432/db?x=y"
    )
    assert "secret" not in redacted
    assert "user:***" in redacted
