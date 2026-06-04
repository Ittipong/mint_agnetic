"""Per-session file logging for DEBUG / root-cause analysis.

Why: a single chat session (one `thread_id`) should land in ONE human-readable
log file so an investigator can read it top-to-bottom and trace the whole
chain — LLM prompts/responses, intent + confidence, routing, CodeAct code +
sandbox output, errors, and proposal confirm/cancel.

Routing is ContextVar-based (mirrors `user_context.py`): FastAPI sets the
active logger on request entry; nodes/tools read it via the module-level
`slog*` helpers and NO-OP when none is set, so node code never needs the
logger plumbed through its signature. This is async-safe across concurrent
sessions — each `astream_events` task carries its own ContextVar copy.

Gate: enabled in dev/local/staging (LOG_MODE=debug OR ENVIRONMENT != production),
disabled in production so these verbose files never ship.

NEVER log secrets (API keys, tokens, passwords). Financial amounts / transaction
data ARE allowed here — this is a dev debug channel only.
"""

from __future__ import annotations

import os
import re
import threading
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Gate ───────────────────────────────────────────────────────────────────

def _logging_enabled() -> bool:
    """Enable in dev/local/staging, off in production.

    Why both flags: LOG_MODE=debug is an explicit opt-in; ENVIRONMENT guards
    the implicit default so we never write verbose files in prod even if
    LOG_MODE drifts.
    """
    if os.getenv("LOG_MODE", "").lower() == "debug":
        return True
    return os.getenv("ENVIRONMENT", "development").lower() != "production"


def session_log_dir() -> str:
    return os.getenv("SESSION_LOG_DIR", "logs")


# ── Logger ───────────────────────────────────────────────────────────────────

class SessionLogger:
    """Bound to one `thread_id`; appends to `<log_dir>/session_<thread_id>.log`.

    Open-append-write-close per call keeps it crash-safe (no dangling handle if
    the process dies mid-turn) and correct across the worker-thread sandbox,
    which writes from a different thread than the event loop. A per-instance
    lock serializes the small fast writes. Debug-grade: simplicity over
    throughput.
    """

    def __init__(self, path: Path, thread_id: str):
        self._path = path
        self._thread_id = thread_id
        self._lock = threading.Lock()

    @staticmethod
    def _ts() -> str:
        # Millisecond precision — enough to order events within a turn.
        return datetime.now().strftime("%H:%M:%S.") + f"{datetime.now().microsecond // 1000:03d}"

    def _write(self, text: str) -> None:
        # Best-effort: a logging failure must never break a chat turn.
        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(text)
        except Exception:  # noqa: BLE001 — debug logging is non-critical
            pass

    # -- public API --------------------------------------------------------

    def log(self, tag: str, message: str) -> None:
        """Single line: `[HH:MM:SS.mmm] <tag> → <message>`."""
        self._write(f"[{self._ts()}] {tag} → {message}\n")

    def log_section(self, title: str) -> None:
        bar = "═" * 16
        self._write(f"\n{bar} {title} {bar}\n")

    def log_block(self, tag: str, label: str, content: str) -> None:
        """Multi-line content (LLM prompt/response, code, SQL) indented under a
        header. For debug we keep FULL content but stamp its length so an
        investigator knows nothing was silently dropped."""
        body = content if content is not None else ""
        n = len(body)
        indented = "\n".join("    " + ln for ln in body.splitlines()) or "    (empty)"
        self._write(
            f"[{self._ts()}] {tag} ── {label} ({n} chars):\n{indented}\n"
        )

    def log_error(self, tag: str, error: "Exception | str", stack: Optional[str] = None) -> None:
        msg = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        out = f"[{self._ts()}] {tag} ✗ ERROR: {msg}\n"
        if stack:
            out += "\n".join("    " + ln for ln in stack.splitlines()) + "\n"
        self._write(out)

    def turn_header(
        self,
        user_id: Optional[str],
        message: str,
        wallet_id: Optional[str],
        extra: Optional[dict] = None,
    ) -> None:
        bar = "═" * 16
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        extra_str = ""
        if extra:
            extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())
        self._write(
            f"\n{bar} TURN @ {stamp} {bar}\n"
            f"user={user_id} wallet={wallet_id}{extra_str} "
            f'msg="{message}"\n\n'
        )

    def turn_footer(self, duration_s: float) -> None:
        bar = "─" * 12
        self._write(f"{bar} END TURN ({duration_s:.1f}s) {bar}\n")


# ── No-op fallback ─────────────────────────────────────────────────────────

class _NoopLogger(SessionLogger):
    """Returned when logging is disabled — every method is a no-op so callers
    stay uniform and pay no IO cost in production."""

    def __init__(self):  # noqa: D401 — intentionally skips file path
        pass

    def _write(self, text: str) -> None:  # noqa: D401
        return


_NOOP = _NoopLogger()


# ── ContextVar routing (mirrors user_context.py) ───────────────────────────

_session_logger_var: ContextVar[Optional[SessionLogger]] = ContextVar(
    "session_logger", default=None
)


def set_session_logger(logger: Optional[SessionLogger]) -> None:
    _session_logger_var.set(logger)


def get_session_logger() -> Optional[SessionLogger]:
    return _session_logger_var.get()


def _next_index(directory: Path) -> int:
    """Next running index = max numeric prefix of existing `NNNN_*.log` + 1.

    Derived from directory contents, not a persisted counter — we keep no index
    file, so deleting all logs restarts numbering at 1. Best-effort: any name we
    can't parse is skipped rather than aborting the turn."""
    highest = 0
    try:
        for child in directory.glob("[0-9]*_*.log"):
            m = re.match(r"^(\d+)_", child.name)
            if m:
                highest = max(highest, int(m.group(1)))
    except Exception:  # noqa: BLE001 — best-effort, fall back to index 1
        pass
    return highest + 1


def open_session_logger(thread_id: str, user_id: Optional[str] = None) -> SessionLogger:
    """Build the per-thread logger. Returns a no-op logger when logging is
    disabled (production) so callers never branch on the gate themselves.

    Filename: `<NNNN>_<YYYY-MM-DD_HHMMSS>_<thread_id>.log` — the index + creation
    timestamp make logs sort chronologically (raw UUID thread_ids did not),
    while the thread_id suffix preserves LangSmith cross-referencing.

    A session spans multiple turns. We glob `*_<thread_id>.log` and reuse the
    first match so later turns APPEND into the same file (this also matches the
    legacy `session_<thread_id>.log` name, so in-flight sessions keep their
    file); a new index + timestamp is minted only when no file for this thread
    exists yet. No registry/index file is kept — both ordering and the
    thread→file mapping live entirely in the filenames."""
    if not _logging_enabled() or not thread_id:
        return _NOOP
    # Keep the filename filesystem-safe without losing readability.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in thread_id)
    directory = Path(session_log_dir())
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 — fall back to no-op if dir can't be made
        return _NOOP
    # Reuse this thread's existing file so later turns append, not fork.
    existing = sorted(directory.glob(f"*_{safe}.log"))
    if existing:
        return SessionLogger(existing[0], thread_id)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{_next_index(directory):04d}_{stamp}_{safe}.log"
    return SessionLogger(directory / name, thread_id)


# ── Module-level convenience fns (fetch current logger, no-op if unset) ─────

def slog(tag: str, msg: str) -> None:
    lg = get_session_logger()
    if lg is not None:
        lg.log(tag, msg)


def slog_section(title: str) -> None:
    lg = get_session_logger()
    if lg is not None:
        lg.log_section(title)


def slog_block(tag: str, label: str, content: str) -> None:
    lg = get_session_logger()
    if lg is not None:
        lg.log_block(tag, label, content)


def slog_error(tag: str, err: "Exception | str", stack: Optional[str] = None) -> None:
    lg = get_session_logger()
    if lg is not None:
        lg.log_error(tag, err, stack)
