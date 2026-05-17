"""Async CRUD for chat_threads metadata.

Messages live in LangGraph's `checkpoints` table; this module only manages
sidebar metadata (title, preview, message_count, timestamps) and provides
cascade-delete across all checkpointer tables.

Pool is the shared `psycopg_pool.AsyncConnectionPool` created in server.py.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg_pool import AsyncConnectionPool


_PREVIEW_LIMIT = 120
_TITLE_LIMIT = 40
_DEFAULT_TITLE = "แชตใหม่"


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


async def create_thread(
    pool: AsyncConnectionPool,
    user_id: str,
    title: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Create a new chat thread row. Returns the inserted row."""
    tid = thread_id or str(uuid.uuid4())
    final_title = title.strip() if title else _DEFAULT_TITLE

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO chat_threads (thread_id, user_id, title)
                VALUES (%s, %s, %s)
                RETURNING thread_id, user_id, title,
                          last_message_preview, message_count,
                          created_at, updated_at
                """,
                (tid, user_id, final_title),
            )
            row = await cur.fetchone()
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))


async def list_threads(
    pool: AsyncConnectionPool,
    user_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List user's threads sorted by updated_at DESC."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, user_id, title,
                       last_message_preview, message_count,
                       created_at, updated_at
                FROM chat_threads
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]


async def get_thread(
    pool: AsyncConnectionPool,
    thread_id: str,
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, user_id, title,
                       last_message_preview, message_count,
                       created_at, updated_at
                FROM chat_threads
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))


async def rename_thread(
    pool: AsyncConnectionPool,
    thread_id: str,
    title: str,
) -> dict[str, Any] | None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE chat_threads
                SET title = %s, updated_at = NOW()
                WHERE thread_id = %s
                RETURNING thread_id, title, updated_at
                """,
                (title.strip(), thread_id),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))


async def upsert_on_first_message(
    pool: AsyncConnectionPool,
    thread_id: str,
    user_id: str,
    first_message: str,
) -> bool:
    """Create thread row if not exists, using first message as title.
    Returns True if inserted, False if already existed."""
    title = _truncate(first_message, _TITLE_LIMIT) or _DEFAULT_TITLE
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO chat_threads (thread_id, user_id, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                RETURNING thread_id
                """,
                (thread_id, user_id, title),
            )
            row = await cur.fetchone()
            return row is not None


async def update_after_message(
    pool: AsyncConnectionPool,
    thread_id: str,
    assistant_reply: str,
) -> None:
    """Update preview + counters after a finished AI turn."""
    preview = _truncate(assistant_reply, _PREVIEW_LIMIT)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE chat_threads
                SET last_message_preview = %s,
                    message_count = message_count + 2,
                    updated_at = NOW()
                WHERE thread_id = %s
                """,
                (preview, thread_id),
            )


async def update_after_marker(
    pool: AsyncConnectionPool,
    thread_id: str,
    marker_preview: str,
) -> None:
    """Update preview + counters after a single marker append.

    Used by `POST /chat/intent` which persists ONLY a HumanMessage
    marker (no ack AIMessage), so the count bumps by 1 — not 2.
    """
    preview = _truncate(marker_preview, _PREVIEW_LIMIT)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE chat_threads
                SET last_message_preview = %s,
                    message_count = message_count + 1,
                    updated_at = NOW()
                WHERE thread_id = %s
                """,
                (preview, thread_id),
            )


async def delete_thread_cascade(
    pool: AsyncConnectionPool,
    thread_id: str,
) -> bool:
    """Hard-delete thread + all LangGraph checkpoint rows in one transaction.
    Returns True if anything was deleted."""
    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                    (thread_id,),
                )
                await cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                    (thread_id,),
                )
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )
                await cur.execute(
                    "DELETE FROM chat_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                return cur.rowcount > 0
