"""Per-request user-id context — same pattern as `debug_log.request_id`.

LangChain's BaseCallbackHandler doesn't get the user_id passed in to
each `on_llm_end` event, and we don't want to surgery-patch every
chain to thread it through state. A ContextVar set by the HTTP entry
point (server.py `_stream_graph`) lets the billing callback pick up
the right user automatically.

When the context is empty (e.g. a CodeAct subgraph spawned outside an
HTTP request, or an offline script), `current_user_id()` returns None
and the billing callback skips the report — better than guessing.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_USER_ID: ContextVar[str | None] = ContextVar("billing_user_id", default=None)


def set_user_id(user_id: str | None) -> None:
    """Set the active user for billing callbacks. Pass None to clear."""
    _USER_ID.set(user_id)


def current_user_id() -> str | None:
    """Return the user_id stamped on the active request, or None."""
    return _USER_ID.get()


@contextmanager
def user_context(user_id: str | None) -> Iterator[None]:
    """Scoped helper for short-lived code paths that want to set, run,
    and reliably reset the user context — e.g. a one-off CLI call or
    a fixture in a test.

    Server.py uses the raw set_user_id() instead because the lifespan
    of the request is bounded by the ASGI middleware, not a python
    `with` block.
    """
    token = _USER_ID.set(user_id)
    try:
        yield
    finally:
        _USER_ID.reset(token)
