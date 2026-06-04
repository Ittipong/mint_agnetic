"""Threads metadata repo (separate from the checkpointer's tables).

The checkpointer (AsyncPostgresSaver) stores raw state per thread_id; this
repo tracks the user-facing thread *metadata* (created_at, last_active,
title) so we can list/delete from the API.

Wire format note: the mobile client speaks the v1 contract (ThreadOut with
`thread_id` / `updated_at` / `last_message_preview` / `message_count`), so
the read methods here return that shape directly — the DB columns `id` and
`last_active` are mapped to `thread_id` and `updated_at` on the way out.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from psycopg_pool import AsyncConnectionPool


THREADS_DDL = """
CREATE TABLE IF NOT EXISTS threads (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    title         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS threads_user_idx ON threads(user_id);
"""

# Default title until the first user message overrides it (matches v1).
_DEFAULT_TITLE = "แชตใหม่"

# A slip/voice turn stores a machine marker as the user message text (mobile
# uses it to render its own photo bubble on replay, so we can't drop it). When
# that marker is the latest text it becomes the list preview — show a friendly
# label instead of leaking the raw `[INTENT:...]` code. Display-only: the stored
# text is untouched.
_FRIENDLY_SLIP_PREVIEW = "📷 สลิป"


def _friendly_preview(preview: Optional[str]) -> Optional[str]:
    """Map an internal intent marker used as a message's text to a human label;
    pass real text through unchanged."""
    if preview and preview.startswith("[INTENT:"):
        return _FRIENDLY_SLIP_PREVIEW
    return preview


def _thread_out(
    *, id: str, user_id: str, title: Optional[str],
    created_at: Any, last_active: Any,
    message_count: int = 0, last_message_preview: Optional[str] = None,
) -> dict:
    """Map a DB row to the v1 ThreadOut wire shape."""
    return {
        "thread_id": id,
        "user_id": user_id,
        "title": title or _DEFAULT_TITLE,
        "last_message_preview": _friendly_preview(last_message_preview),
        "message_count": message_count,
        "created_at": created_at,
        "updated_at": last_active,
    }


class ThreadsRepo:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def upsert(self, *, thread_id: str, user_id: str,
                     title: Optional[str] = None) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # The first-turn title wins: keep the existing title and only
                # fall back to the incoming one when the row had none. (The old
                # order let every later turn clobber it — fine when the client
                # resent the same title, but wrong now that the server derives
                # one per turn from that turn's content.) Explicit renames go
                # through `rename()`, not this upsert, so they are unaffected.
                await cur.execute(
                    "INSERT INTO threads (id, user_id, title) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET last_active = NOW(), "
                    "title = COALESCE(threads.title, EXCLUDED.title)",
                    (thread_id, user_id, title),
                )

    async def create(self, *, user_id: str,
                     title: Optional[str] = None) -> dict:
        """Allocate a fresh thread_id and insert an empty metadata row.

        Returns the new thread in v1 ThreadOut shape (count 0, no preview).
        """
        thread_id = str(uuid.uuid4())
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO threads (id, user_id, title) "
                    "VALUES (%s, %s, %s) "
                    "RETURNING id, user_id, title, created_at, last_active",
                    (thread_id, user_id, title),
                )
                r = await cur.fetchone()
        return _thread_out(
            id=r[0], user_id=r[1], title=r[2],
            created_at=r[3], last_active=r[4],
        )

    async def rename(self, thread_id: str, title: str) -> Optional[dict]:
        """Update the display title. Returns RenameThreadOut shape, or None
        if the thread doesn't exist."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE threads SET title = %s, last_active = NOW() "
                    "WHERE id = %s "
                    "RETURNING id, title, last_active",
                    (title, thread_id),
                )
                r = await cur.fetchone()
        if not r:
            return None
        return {"thread_id": r[0], "title": r[1], "updated_at": r[2]}

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        """List a user's threads (most recent activity first), each enriched
        with `message_count` and the latest message's text as
        `last_message_preview` — computed in one grouped query."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # `last_message_preview` = text of the most recent message that
                # actually has text. Assistant turns may be blocks-only (empty
                # `answer`), so NULLIF('') skips them and falls back to the
                # latest message that does carry text.
                await cur.execute(
                    "SELECT t.id, t.user_id, t.title, t.created_at, t.last_active, "
                    "       COUNT(tm.id) AS message_count, "
                    "       (ARRAY_AGG("
                    "           COALESCE(NULLIF(tm.content->>'answer', ''), "
                    "                    NULLIF(tm.content->>'text', '')) "
                    "           ORDER BY tm.created_at DESC, tm.id DESC"
                    "       ) FILTER (WHERE COALESCE("
                    "           NULLIF(tm.content->>'answer', ''), "
                    "           NULLIF(tm.content->>'text', '')) IS NOT NULL))[1] "
                    "           AS last_message_preview "
                    "FROM threads t "
                    "LEFT JOIN thread_messages tm ON tm.thread_id = t.id "
                    "WHERE t.user_id = %s "
                    "GROUP BY t.id, t.user_id, t.title, t.created_at, t.last_active "
                    "ORDER BY t.last_active DESC "
                    "LIMIT %s",
                    (user_id, limit),
                )
                rows = await cur.fetchall()
        return [
            _thread_out(
                id=r[0], user_id=r[1], title=r[2],
                created_at=r[3], last_active=r[4],
                message_count=r[5] or 0, last_message_preview=r[6],
            )
            for r in rows
        ]

    async def get(self, thread_id: str) -> Optional[dict]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, user_id, title, created_at, last_active "
                    "FROM threads WHERE id = %s",
                    (thread_id,),
                )
                r = await cur.fetchone()
        if not r:
            return None
        return _thread_out(
            id=r[0], user_id=r[1], title=r[2],
            created_at=r[3], last_active=r[4],
        )

    async def delete(self, thread_id: str) -> bool:
        """Delete the metadata row. Returns True if a row was removed,
        False if it didn't exist (so the API can report `found`)."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM threads WHERE id = %s", (thread_id,)
                )
                return cur.rowcount > 0
