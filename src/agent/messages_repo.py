"""Per-message transcript repo.

The checkpointer (AsyncPostgresSaver) keeps agent *state* (last_txn,
proposals, episodes) but NOT the user-facing chat bubbles. Without a
transcript, returning to a thread shows an empty pane. This repo stores
one row per message so the client can reload a thread's full conversation
on switch — ChatGPT-style history.

`content` is JSONB shaped by role:
  - user:      {"text": "..."}
  - assistant: {"blocks": [...response_blocks...], "answer": "..."|null}
"""

from __future__ import annotations

import json
from typing import Any

from psycopg_pool import AsyncConnectionPool


MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS thread_messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS thread_messages_thread_idx
    ON thread_messages(thread_id, created_at, id);
"""


class MessagesRepo:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def append(
        self, *, thread_id: str, user_id: str, role: str, content: dict[str, Any],
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO thread_messages (thread_id, user_id, role, content) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (thread_id, user_id, role,
                     json.dumps(content, ensure_ascii=False)),
                )

    async def list_for_thread(self, thread_id: str) -> list[dict[str, Any]]:
        # `id` is the tiebreaker — user + assistant rows of the same turn share
        # a NOW() created_at, so insertion order (id ASC) keeps them in order.
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT role, content, created_at FROM thread_messages "
                    "WHERE thread_id = %s ORDER BY created_at ASC, id ASC",
                    (thread_id,),
                )
                rows = await cur.fetchall()
        return [
            {"role": r[0], "content": r[1], "created_at": r[2]} for r in rows
        ]

    async def delete_for_thread(self, thread_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM thread_messages WHERE thread_id = %s",
                    (thread_id,),
                )
