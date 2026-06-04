"""Factory for the LangGraph long-term-memory `BaseStore`.

NEW in Wave 4. Spec: `docs/v3/phase2_state_and_graph.md` §8.

Why a factory:
- `AsyncPostgresStore` is async-context-managed (`from_conn_string` returns
  an async context manager). The FastAPI lifespan owns the context window;
  this factory packages the construction + DDL into one async call that
  returns the live store *and* its cleanup hook.
- Tests should never open a live Postgres connection. `build_store(...)`
  returns `None` when `DATABASE_URL` is unset OR when the
  `langgraph.store.postgres` import fails (e.g. missing dependency in CI).
  Callers MUST treat `None` as "memory tools disabled" — `memory_recall`
  and `memory_write` already do.

Q1 (locked, phase3_decisions.md):
  Schema isolation lives in the DATABASE_URL itself:
    DATABASE_URL=postgresql://…?options=-c%20search_path%3Dv3
  We do not re-set search_path here; the connection string carries it.

Embedding choice:
  Wave 4 starts with `openai:text-embedding-3-small` — same model v2's
  long-term memory used. It runs through `langchain_openai`'s embeddings
  shim, which honors OpenAI-compatible base URLs (so an OPENAI_BASE_URL
  pointed at OpenRouter works). Phase 4 may swap to a cheaper provider; the
  factory accepts an override via the `embed` argument.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

# `BaseStore` is the abstract contract that `memory_recall` / `memory_write`
# expect via `InjectedStore`. The PostgresStore impl is loaded lazily so a
# missing extras dependency in tests doesn't break collection.
from langgraph.store.base import BaseStore


_LOG = logging.getLogger(__name__)

# Default embedding spec. Disabled by default — `openai:text-embedding-3-small`
# requires OPENAI_API_KEY which we don't ship (the chat LLM goes through
# OpenRouter, not OpenAI direct). Set `LANGMEM_EMBED` to re-enable, e.g.
# `LANGMEM_EMBED=openai:text-embedding-3-small`. With embed=None the store
# still works for k/v reads — only semantic recall is unavailable.
_DEFAULT_EMBED = os.getenv("LANGMEM_EMBED") or None
_DEFAULT_DIMS = int(os.getenv("LANGMEM_EMBED_DIMS", "1536"))


async def build_store(
    *,
    dsn: Optional[str] = None,
    embed: Optional[str] = None,
    dims: int = _DEFAULT_DIMS,
) -> Optional[BaseStore]:
    """Build the long-term-memory store. Returns `None` to indicate disabled.

    Args:
      dsn:   Postgres DSN. Defaults to `os.environ["DATABASE_URL"]`. Per Q1,
             the DSN should carry `?options=-c%20search_path%3Dv3` so the
             store's tables sit under the `v3` schema separate from v2.
      embed: Embedding model spec (`"openai:text-embedding-3-small"`-style).
             Defaults to `_DEFAULT_EMBED`; pass through unchanged to the
             `index` config the store accepts. None or "" disables
             semantic-search index — store still works for k/v reads, just
             slower on `memory_recall(topic=...)`.
      dims:  Embedding dimensionality (1536 for openai 3-small). Ignored
             when `embed` is None.

    Returns:
      A constructed `BaseStore` (semantically `AsyncPostgresStore`) with
      `setup()` already called, OR `None` when the store can't be built
      (no DSN, missing dependency, or DDL failure).

    Notes:
      - Caller MUST keep a reference to the returned store; releasing it
        closes the underlying connection. In `server.py`'s lifespan, this
        lives on `app.state.store`.
      - The DDL is idempotent — repeat lifespans are safe.
    """
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        _LOG.info("build_store: DATABASE_URL unset — long-term memory disabled")
        return None

    try:
        # Lazy import — pulls in psycopg + asyncpg machinery. Skipping it
        # gracefully keeps unit tests fast and runnable on machines without
        # the extras installed.
        from langgraph.store.postgres.aio import AsyncPostgresStore
    except Exception as exc:  # noqa: BLE001 — surface ANY import failure
        _LOG.warning(
            "build_store: langgraph.store.postgres unavailable (%s) — "
            "long-term memory disabled",
            exc,
        )
        return None

    embed_spec = embed if embed is not None else _DEFAULT_EMBED
    index_config: Optional[dict[str, Any]] = None
    if embed_spec:
        index_config = {"dims": dims, "embed": embed_spec}

    try:
        # `from_conn_string` returns an async context manager; we open it
        # via `__aenter__` so the caller (lifespan) owns the close. The
        # alternative — `async with from_conn_string(...) as store` —
        # forces the lifespan to nest the context, which couples the store
        # lifecycle to a single coroutine. Manual aenter/aexit keeps the
        # store available across the whole FastAPI lifespan.
        cm = AsyncPostgresStore.from_conn_string(dsn, index=index_config)
        store = await cm.__aenter__()
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("build_store: AsyncPostgresStore construction failed: %s", exc)
        return None

    try:
        await store.setup()
    except Exception as exc:  # noqa: BLE001
        # DDL failure is recoverable — the store may already be initialized
        # by a previous lifespan; log and continue rather than crash boot.
        _LOG.warning(
            "build_store: store.setup() failed (%s) — continuing with "
            "existing schema if any",
            exc,
        )

    _LOG.info(
        "build_store: AsyncPostgresStore ready dsn=%s embed=%s dims=%d",
        _redact_dsn(dsn), embed_spec, dims,
    )
    return store


def _redact_dsn(dsn: str) -> str:
    """Hide the password before logging the DSN — same redaction we use in
    session_logger; we duplicate it here to avoid a circular import."""
    try:
        # postgresql://user:pass@host:port/db -> postgresql://user:***@host…
        scheme, _, rest = dsn.partition("://")
        if not scheme or "@" not in rest:
            return dsn
        creds, _, host_rest = rest.partition("@")
        if ":" in creds:
            user, _, _pw = creds.partition(":")
            return f"{scheme}://{user}:***@{host_rest}"
        return dsn
    except Exception:  # noqa: BLE001 — best-effort
        return dsn


__all__ = ["build_store"]
