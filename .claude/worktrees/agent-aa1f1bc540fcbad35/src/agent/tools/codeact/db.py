"""Shared asyncpg pool with Decimal codec for the CodeAct sandbox.

Ported from v2 (`codeact_subgraph/shared/db.py`) in Wave 1 — relocated
under `src/agent/tools/codeact/` because v3 collapses the CodeAct
sub-graph into a single `run_python` tool.

A single pool per process — created lazily on first call. asyncpg's default
numeric -> Decimal mapping is preserved (it's the default for `numeric`
columns; we cast double precision -> numeric in SQL templates so aggregates
land as Decimal here, per memory `project_chat_money_column_float8`).

IMPORTANT: this pool is for the FINANCIAL data DB (`mint_money_dev` via
`BACKEND_DATABASE_URL`), NOT the agent's checkpoint/thread DB
(`DATABASE_URL` -> `mint_agentic`). See memory
`reference_agentic_v2_two_databases`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import asyncpg

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


def _normalize_dsn(url: str) -> str:
    """asyncpg requires `postgresql://`, not the SQLAlchemy `+asyncpg` form."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _resolve_dsn() -> str:
    """Env-driven DSN for the codeact sandbox's SQL wrappers.

    The sandbox reads FINANCIAL data (general_wallets, transactions,
    categories, …) which lives in the BACKEND database — so prefer
    `BACKEND_DATABASE_URL`. Falling back to `DATABASE_URL` (the agent's own
    checkpoint/thread DB) only when no backend URL is set keeps dev/test
    (single-DB) working. With the two pointing at different DBs, preferring
    `DATABASE_URL` here would send every query to the agent DB — which has
    no financial tables → `UndefinedTableError`.
    Raises RuntimeError if neither is set so pool init fails loud, not silent.
    """
    url = os.getenv("BACKEND_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "neither DATABASE_URL nor BACKEND_DATABASE_URL is set — "
            "the codeact sandbox cannot reach Postgres without a DSN"
        )
    return _normalize_dsn(url)


async def get_pool() -> asyncpg.Pool:
    """Lazy-initialize and return the shared pool.

    `server_settings.timezone='Asia/Bangkok'` pins the PG session to the
    user-facing calendar. Without this, asyncpg encodes Python `datetime.date`
    as a `timestamptz` rooted in the client process's local TZ (Asia/Bangkok
    on the production host); when the server interprets that timestamptz in
    the default UTC session, every date boundary shifts back 7 hours and
    period filters silently leak in / out neighbouring days. The symptom on
    "เดือนที่แล้ว" was sum_income counting the boundary rows of both
    March-end and April-end — returning 2× the real income.
    """
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=_resolve_dsn(),
                min_size=1,
                max_size=5,
                command_timeout=15,
                server_settings={"timezone": "Asia/Bangkok"},
            )
    return _pool


async def close_pool() -> None:
    """Close on shutdown — call from server lifespan."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
