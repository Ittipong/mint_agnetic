"""Unified debug logger for the chat server + graph nodes.

Single source of truth — fixes two prior bugs:
1. Each caller computed `Path(__file__).parent.parent.parent`, which
   resolved to different directories depending on file depth. Logs
   were silently split across `mint_agnetic/logs/` and
   `mint_agnetic/financial-chat/logs/`. This module pins one path.
2. Log filename was previously frozen at import time, so a long-
   running server kept appending to yesterday's date file. The path is
   now recomputed per write.

Adds per-request correlation via a ContextVar (`request_id`) so every
log line emitted inside a single HTTP request shares the same id, even
across `server.py` and the LangGraph nodes that run inside it.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.config import settings


# All logs land under the financial-chat project, not its parent.
# Resolved once (the directory itself doesn't change), but the filename
# inside it is recomputed on every write — see `_log_path`.
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)


class LogLevel(Enum):
    """Verbosity tier for a log line.

    `MILESTONE` — high-signal events (request entry, stream start/done,
    tool boundaries, LLM tool_calls summary, errors). Always written.

    `DETAIL` — node-internal mechanics (state_keys, history compression,
    catalog counts, user_id echoes). Suppressed when `LOG_MODE=production`
    so prod logs stay grep-friendly and small.
    """

    MILESTONE = "milestone"
    DETAIL = "detail"


# Per-request correlation id. Set by the HTTP middleware in server.py;
# read by every `log()` call inside the request scope. Default `None`
# for background tasks / startup logs.
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def _log_path() -> Path:
    """Today's log file path, recomputed per write so the date rolls
    at midnight without restarting the server."""
    return _LOG_DIR / f"agent_{datetime.now().strftime('%Y-%m-%d')}.log"


def log(
    tag: str,
    msg: str,
    *,
    level: LogLevel = LogLevel.DETAIL,
    **fields,
) -> None:
    """Write a structured log line.

    Format:
        [ISO_TS] [TAG] [req=<short>] message k1=v1 k2=v2 ...

    `req=` shows the first 12 chars of the per-request correlation id
    (full id available via `get_request_id()` if you need to join
    across systems). `req=-` means there's no active request — typically
    a startup/shutdown log. 12 chars is enough for `grep req=<prefix>`
    to uniquely pick one UUIDv4 within a day's log without showing the
    full 36-char id on every line.
    """
    if level is LogLevel.DETAIL and not settings.is_debug_log:
        return

    rid = _request_id_ctx.get()
    rid_short = rid[:12] if rid else "-"
    parts = [f"[{datetime.now().isoformat()}] [{tag}] [req={rid_short}] {msg}"]
    for k, v in fields.items():
        parts.append(f" {k}={v}")
    line = "".join(parts) + "\n"

    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:  # noqa: BLE001 - logging must never fail the turn
        pass
    logging.info(line.rstrip("\n"))
