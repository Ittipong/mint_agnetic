"""Shared asyncpg pool with Decimal codec.

A single pool per process — created lazily on first call. asyncpg's default
numeric → Decimal mapping is preserved (it's the default for `numeric` columns;
we cast double precision → numeric in SQL templates so aggregates land as
Decimal here).
"""

from __future__ import annotations

import asyncpg
import asyncio
from typing import Optional

from src.config import settings

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


def _normalize_dsn(url: str) -> str:
    """asyncpg requires `postgresql://`, not the SQLAlchemy `+asyncpg` form."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_pool() -> asyncpg.Pool:
    """Lazy-initialize and return the shared pool."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=_normalize_dsn(settings.backend_database_url),
                min_size=1,
                max_size=5,
                command_timeout=15,
            )
    return _pool


async def close_pool() -> None:
    """Close on shutdown — call from server lifespan."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
