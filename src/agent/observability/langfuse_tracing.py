"""Langfuse tracing — PoC (Phase 1: chat).

Attaches a Langfuse LangChain CallbackHandler to the graph run so every ReAct/
CodeAct LLM call, tool call, and node becomes a span/generation in Langfuse,
grouped per USER (langfuse_user_id) and per CONVERSATION (langfuse_session_id =
thread_id). This lets you trace "everything a user did in chat" in one place,
alongside cost/latency/token usage.

DESIGN (load-bearing):
  • OPT-IN + OFF by default (`LANGFUSE_ENABLED`). Until it's on with valid keys,
    every function here is a no-op — the chat path is byte-for-byte unchanged.
  • NEVER raises. Observability must not break a user's turn: import failures,
    a missing instance, or a bad key all degrade to "no tracing", logged once.
  • Credentials come from env (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
    LANGFUSE_HOST) — the same vars both langfuse v2 and v3 read.
  • Trace attributes (user_id / session_id / tags) are passed via the run
    config's `metadata` keys (langfuse_user_id / langfuse_session_id /
    langfuse_tags), which BOTH v2 and v3 honour — so the handler construction
    stays version-agnostic.

⚠️ PRIVACY (finance app): a trace captures the LLM prompts/completions, which
include the user's real amounts/categories. Use a SELF-HOSTED Langfuse and lock
down access before enabling in prod (CLAUDE.md: never ship payment info to a
third-party). Masking is a later phase.

Sibling of react_callback.py (the session-log bridge); both are plain
LangChain callbacks and coexist in the same `callbacks=[...]` list.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..session_logger import slog

_TAG = "obs.langfuse"

# Lazily-built singleton handler (+ a flag so we only log the outcome once). The
# handler is stateless w.r.t. a run — trace attributes ride on the config
# metadata per request — so one instance is safe to share across requests.
_handler: Optional[Any] = None
_resolved: bool = False


def langfuse_enabled() -> bool:
    """True only when explicitly switched on. Default OFF keeps the PoC inert."""
    return os.getenv("LANGFUSE_ENABLED", "off").strip().lower() in ("on", "true", "1")


def _build_handler() -> Optional[Any]:
    """Construct the Langfuse LangChain CallbackHandler, tolerant of the v2/v3
    package split. Returns None (never raises) if langfuse isn't installed or the
    client can't initialise."""
    # v3: `from langfuse.langchain import CallbackHandler` — credentials come from
    # env via the global client, so the constructor takes no keys.
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore

        return CallbackHandler()
    except Exception:  # noqa: BLE001 — any failure -> try the older layout
        pass
    # v2: `from langfuse.callback import CallbackHandler` — pass creds explicitly.
    try:
        from langfuse.callback import CallbackHandler  # type: ignore

        return CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
    except Exception as exc:  # noqa: BLE001
        slog(_TAG, f"langfuse unavailable ({exc!r}) -> tracing disabled")
        return None


def _get_handler() -> Optional[Any]:
    """Lazy singleton; resolves the handler once and caches the outcome."""
    global _handler, _resolved
    if _resolved:
        return _handler
    _resolved = True
    _handler = _build_handler()
    if _handler is not None:
        slog(_TAG, f"langfuse tracing ON (host={os.getenv('LANGFUSE_HOST', 'default')})")
    return _handler


def attach_langfuse(
    config: dict,
    *,
    user_id: str = "",
    session_id: str = "",
    tags: Optional[list[str]] = None,
) -> dict:
    """Return `config` with the Langfuse handler + trace attributes merged in.

    No-op (returns the config unchanged) when tracing is disabled or the handler
    can't be built. Merges rather than overwrites, so an existing callback (e.g.
    the session-log bridge) and metadata are preserved. Never raises."""
    try:
        if not langfuse_enabled():
            return config
        handler = _get_handler()
        if handler is None:
            return config
        out = dict(config)
        out["callbacks"] = [*(out.get("callbacks") or []), handler]
        # v2 + v3 both read these keys off the run metadata to stamp the trace.
        meta = dict(out.get("metadata") or {})
        if user_id:
            meta["langfuse_user_id"] = user_id
        if session_id:
            meta["langfuse_session_id"] = session_id
        if tags:
            meta["langfuse_tags"] = list(tags)
        out["metadata"] = meta
        return out
    except Exception as exc:  # noqa: BLE001 — tracing must never break a turn
        slog(_TAG, f"attach_langfuse failed ({exc!r}) -> tracing skipped this turn")
        return config
