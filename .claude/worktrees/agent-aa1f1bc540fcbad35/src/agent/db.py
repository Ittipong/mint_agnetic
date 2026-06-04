"""Postgres connection helpers + ProposalRepo backed by psycopg pool.

The pool is shared with LangGraph's AsyncPostgresSaver checkpointer
(see server.py lifespan). We expose a minimal Proposal repo here; the
agent itself never touches real entity tables — those are only written
inside REST confirm/cancel endpoints.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

from psycopg_pool import AsyncConnectionPool


DEFAULT_POOL_KWARGS = dict(min_size=1, max_size=10, open=False)


def make_pool(dsn: Optional[str] = None, **kwargs) -> AsyncConnectionPool:
    """Build (do not yet open) an AsyncConnectionPool.

    The DSN preference order:
      1. explicit arg
      2. DATABASE_URL env var
      3. raises — we never default to a localhost guess in prod
    """
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    merged = {**DEFAULT_POOL_KWARGS, **kwargs}
    # psycopg AsyncConnectionPool requires open=False or explicit open() — we
    # leave it closed so the caller (lifespan) can `await pool.open()`.
    return AsyncConnectionPool(conninfo=dsn, **merged)


def make_backend_pool(**kwargs) -> AsyncConnectionPool:
    """Pool for the backend DB (wallets/categories/transactions used by
    entity_catalog). Falls back to DATABASE_URL when BACKEND_DATABASE_URL
    is unset so single-DB deployments keep working unchanged.

    The agent's own metadata (pending_proposals, threads, langgraph
    checkpoints) lives on the DATABASE_URL pool — keep them separate so
    backend can be a read replica without leaking writes.
    """
    dsn = os.getenv("BACKEND_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "neither BACKEND_DATABASE_URL nor DATABASE_URL is set"
        )
    merged = {**DEFAULT_POOL_KWARGS, **kwargs}
    return AsyncConnectionPool(conninfo=dsn, **merged)


PENDING_PROPOSALS_DDL = """
CREATE TABLE IF NOT EXISTS pending_proposals (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    kind         TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ttl_seconds  INTEGER NOT NULL DEFAULT 1800,
    consumed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS pending_proposals_user_idx
    ON pending_proposals(user_id) WHERE consumed_at IS NULL;
"""


class ProposalRepo:
    """Async repository for pending_proposals. The agent only INSERTs;
    REST confirm/cancel handlers SELECT + UPDATE."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def insert_pending_proposal(
        self, *, proposal_id: str, user_id: str, kind: str, payload: dict[str, Any],
    ) -> str:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO pending_proposals (id, user_id, kind, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (proposal_id, user_id, kind, json.dumps(payload)),
                )
        return proposal_id

    async def get_pending_proposal(self, proposal_id: str) -> Optional[dict[str, Any]]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, user_id, kind, payload, created_at, ttl_seconds, consumed_at "
                    "FROM pending_proposals WHERE id = %s",
                    (proposal_id,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "user_id": row[1], "kind": row[2],
            "payload": row[3], "created_at": row[4],
            "ttl_seconds": row[5], "consumed_at": row[6],
        }

    async def mark_consumed(self, proposal_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE pending_proposals SET consumed_at = NOW() "
                    "WHERE id = %s AND consumed_at IS NULL",
                    (proposal_id,),
                )

    async def delete_pending_proposal(self, proposal_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM pending_proposals WHERE id = %s",
                    (proposal_id,),
                )
