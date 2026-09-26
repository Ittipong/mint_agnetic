"""User-facing errors + ops alerts for the chat stream.

Why this exists: every stream endpoint used to forward `str(exc)` to the
client, and mobile renders the `message` of an `error` event verbatim. When
OpenRouter ran out of credits, users saw "Error code: 402 - {'error':
{'message': 'Insufficient credits. Add more using https://openrouter.ai/...'}}"
in the chat bubble, and nobody on our side was told.

Contract: an `error` SSE event carries a STABLE `code` (one of ErrorKind)
and a short Thai `message` written for the user. Internals — exception text,
provider names, URLs, tracebacks — go to the session log only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.agent.session_logger import slog, slog_error


class ErrorKind(str, Enum):
    LLM_CREDITS = "llm_unavailable"      # provider billing/credits exhausted
    LLM_RATE_LIMITED = "llm_busy"        # 429 / provider overloaded
    TIMEOUT = "timeout"
    AGENT_LOOP = "agent_loop"            # recursion limit — the agent went round in circles
    INTERNAL = "internal_error"


USER_MESSAGES: dict[ErrorKind, str] = {
    ErrorKind.LLM_CREDITS: "ตอนนี้ Nimo ใช้งานไม่ได้ชั่วคราว ทีมงานกำลังแก้ไขอยู่ ลองใหม่อีกครั้งในอีกสักครู่นะครับ",
    ErrorKind.LLM_RATE_LIMITED: "ตอนนี้มีคนใช้งานเยอะ ลองส่งใหม่อีกครั้งในอีกสักครู่นะครับ",
    ErrorKind.TIMEOUT: "ใช้เวลานานเกินไป ลองส่งใหม่อีกครั้งได้เลยนะครับ",
    ErrorKind.AGENT_LOOP: "ขอโทษครับ รอบนี้ Nimo หาคำตอบไม่สำเร็จ ลองถามใหม่อีกครั้ง หรือถามให้สั้นลงได้นะครับ",
    ErrorKind.INTERNAL: "ขออภัยครับ ระบบขัดข้องชั่วคราว ลองพิมพ์ใหม่อีกครั้งได้เลยนะครับ",
}

# Kinds that need a human on our side, not just a retry by the user.
_ALERT_KINDS = {ErrorKind.LLM_CREDITS, ErrorKind.LLM_RATE_LIMITED}
_ALERT_COOLDOWN_S = 600
_last_alert: dict[ErrorKind, float] = {}


def _chain_text(exc: BaseException) -> str:
    """Type names + messages along the __cause__/__context__ chain (lowercased)."""
    parts, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__} {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts).lower()


def classify_error(exc: BaseException) -> ErrorKind:
    text = _chain_text(exc)
    if re.search(r"\b402\b", text) or "insufficient credits" in text or "payment required" in text:
        return ErrorKind.LLM_CREDITS
    if re.search(r"\b429\b", text) or "rate limit" in text or "ratelimit" in text or "overloaded" in text:
        return ErrorKind.LLM_RATE_LIMITED
    if "graphrecursionerror" in text or "recursion limit" in text:
        return ErrorKind.AGENT_LOOP
    if "timeout" in text or "timed out" in text:
        return ErrorKind.TIMEOUT
    return ErrorKind.INTERNAL


def ops_alert(kind: ErrorKind, where: str, detail: str) -> bool:
    """Tell US (not the user): one JSON line in logs/ops_alerts.log + an
    `[OPS-ALERT]` line on stderr (the launchd err log). Throttled per kind so a
    credit outage does not write one alert per chat turn. Returns True when an
    alert was written."""
    now = time.monotonic()
    if now - _last_alert.get(kind, -_ALERT_COOLDOWN_S - 1) < _ALERT_COOLDOWN_S:
        return False
    _last_alert[kind] = now
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind.value,
        "where": where,
        "detail": detail[:500],
    }
    line = json.dumps(record, ensure_ascii=False)
    try:
        log_dir = os.getenv("SESSION_LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "ops_alerts.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # alerting must never break the turn
    print(f"[OPS-ALERT] {line}", file=sys.stderr, flush=True)
    slog("ops_alert", line)
    return True


def note_llm_failure(exc: BaseException, where: str) -> ErrorKind:
    """Classify an LLM failure and alert when it needs a human. Used where the
    failure is swallowed (classifier, resolvers) so the outage is still seen."""
    kind = classify_error(exc)
    if kind in _ALERT_KINDS:
        ops_alert(kind, where, str(exc))
    return kind


def error_event(exc: BaseException, where: str, tb: Optional[str] = None) -> dict:
    """The SSE `error` event for `exc`: stable code + Thai message, nothing
    internal. Logs the full error and raises an ops alert when needed."""
    slog_error(where, exc, tb)
    kind = note_llm_failure(exc, where)
    return {
        "event": "error",
        "data": json.dumps(
            {"code": kind.value, "message": USER_MESSAGES[kind]},
            ensure_ascii=False,
        ),
    }


__all__ = [
    "ErrorKind",
    "USER_MESSAGES",
    "classify_error",
    "error_event",
    "note_llm_failure",
    "ops_alert",
]
