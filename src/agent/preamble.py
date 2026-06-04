"""Streaming preamble — "thinking out loud" warm-up before the ReAct loop.

UX rationale: the main ReAct loop takes 5-15s per turn (system prompt 25K
chars + tool calls + Gemini reasoning). Users perceive that as "ค้าง" if
they see nothing. This module emits a 1-2 sentence Thai preamble streamed
from a fast cheap model (gemini-2.0-flash-001) within ~300ms, so the chat
bubble shows text immediately. The real answer then continues in the same
bubble after a `\\n\\n` separator.

The preamble is intentionally NUMBER-FREE and PROMISE-FREE so it can't
contradict whatever the ReAct loop ends up computing. It's a warm
acknowledgment + intent statement only.

Model: `PREAMBLE_MODEL` env (default `google/gemini-2.5-flash-lite`).
Failure: yields nothing — the main loop still streams the real answer
unimpeded. We never block the turn waiting for a preamble.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

import httpx

from src.agent.session_logger import slog, slog_error


_SYSTEM_PROMPT = """\
You are "Mint Money" — a warm Thai financial assistant.

Your ONLY job here: write a SHORT 1-sentence Thai acknowledgment of what
the user just asked, so they see text immediately while the real answer
is being computed.

HARD RULES:
- Reply in Thai only.
- Maximum 15 words.
- Acknowledge + state intent (e.g. "ได้เลยครับ เดี๋ยวเช็คยอดบัตรให้นะ").
- NEVER include numbers, amounts, percentages, dates.
- NEVER give the actual answer.
- NEVER ask a clarifying question.
- NEVER use emojis.
- End with "นะ" or "ครับ" (warm, not robotic).
- Output ONLY the sentence — no quotes, no prefix.

Examples:
- user: "เหลือเงินเท่าไหร่?" → "ได้เลยครับ เดี๋ยวรวมยอดทุกกระเป๋าให้นะ"
- user: "บัตรเหลือเท่าไหร่" → "ได้เลย ขอเช็คยอดบัตรเครดิตให้สักครู่นะครับ"
- user: "เพิ่ม กาแฟ 50" → "รับทราบครับ เดี๋ยวบันทึกรายการให้นะ"
- user: "เดือนนี้ใช้ไปเท่าไหร่" → "ได้ครับ ขอรวมยอดใช้จ่ายเดือนนี้ให้สักครู่"
- user: "สวัสดี" → "สวัสดีครับ มีอะไรให้ช่วยมั้ย"
"""


def _default_model() -> str:
    return os.getenv("PREAMBLE_MODEL", "google/gemini-2.5-flash-lite")


def _build_request(user_message: str) -> tuple[str, dict, dict]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    body = {
        "model": _default_model(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 80,
        "stream": True,
    }
    return url, headers, body


async def astream_preamble(user_message: str) -> AsyncIterator[str]:
    """Stream preamble content tokens. Yields nothing on failure.

    Caller is responsible for forwarding each chunk to the SSE stream
    (typically as an `answer_token` event) and appending a `\\n\\n`
    separator after this generator drains.
    """
    msg = (user_message or "").strip()
    if not msg:
        return
    try:
        url, headers, body = _build_request(msg)
    except Exception as exc:  # noqa: BLE001 — preamble must never break the turn
        slog_error("preamble", exc)
        return

    timeout_s = float(os.getenv("PREAMBLE_TIMEOUT_S", "8"))
    chars_total = 0
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    slog_error(
                        "preamble",
                        RuntimeError(f"preamble HTTP {r.status_code}"),
                    )
                    return
                async for raw in r.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        chars_total += len(delta)
                        yield delta
        slog("preamble", f"streamed {chars_total} chars model={_default_model()}")
    except Exception as exc:  # noqa: BLE001 — never raise into the turn
        slog_error("preamble", exc)
        return


async def astream_with_preamble(
    user_message: str,
    main_stream: AsyncIterator[dict],
    *,
    event_type: str,
) -> AsyncIterator[dict]:
    """Merge the preamble stream with the main ReAct stream — CONCURRENTLY.

    Why: previously each branch awaited `astream_preamble` to drain fully
    *before* starting `stream_chat`. A slow/dead preamble model then added its
    whole timeout to the turn (observed: an 8s ReadTimeout stalling ReAct).
    Here both run at once; the preamble races the answer:

      - preamble finishes first (the normal ~300-500ms case) → emit it as ONE
        `event_type` event so the bubble shows warm text immediately.
      - the real answer (first `answer_token`/`block`) arrives first → the
        preamble has lost its purpose (it only fills the pre-answer silence),
        so it is DROPPED — never shown after the answer started.
      - the main stream ends → any still-pending preamble task is cancelled,
        so a hung preamble can never delay or outlive the turn.

    `event_type` is the SSE event name the joined preamble text is wrapped in:
    `narration_token` for the text branch, `status_token` for voice — both
    ephemeral labels that the mobile cubit clears once the real answer starts.

    Honors `PREAMBLE_ENABLED` (default on); when off, the main stream passes
    through untouched (keeps unit tests preamble-free, see tests/conftest.py).
    """
    enabled = os.getenv("PREAMBLE_ENABLED", "1") not in ("0", "false", "False")
    if not enabled or not (user_message or "").strip():
        async for ev in main_stream:
            yield ev
        return

    _PREAMBLE, _MAIN, _MAIN_DONE = "preamble", "main", "main_done"
    queue: asyncio.Queue = asyncio.Queue()

    async def _drain_preamble() -> None:
        chars: list[str] = []
        try:
            async for tok in astream_preamble(user_message):
                chars.append(tok)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — preamble must never break the turn
            slog_error("preamble", exc)
        await queue.put((_PREAMBLE, "".join(chars)))

    async def _drain_main() -> None:
        try:
            async for ev in main_stream:
                await queue.put((_MAIN, ev))
        finally:
            await queue.put((_MAIN_DONE, None))

    pre_task = asyncio.create_task(_drain_preamble())
    main_task = asyncio.create_task(_drain_main())

    answer_started = False
    try:
        while True:
            kind, payload = await queue.get()
            if kind == _MAIN_DONE:
                break
            if kind == _MAIN:
                # First real answer/block closes the preamble window — anything
                # after this is the actual answer bubble, not pre-answer filler.
                if payload.get("event") in ("answer_token", "block"):
                    answer_started = True
                yield payload
            elif not answer_started and payload:  # _PREAMBLE, still in window
                yield {"event": event_type, "data": payload}
    finally:
        for task in (pre_task, main_task):
            if not task.done():
                task.cancel()
        for task in (pre_task, main_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


__all__ = ["astream_preamble", "astream_with_preamble"]
