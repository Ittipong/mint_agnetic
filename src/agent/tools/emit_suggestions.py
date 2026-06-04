"""`emit_suggestions` — follow-up suggestion chips.

NEW in Wave 3. Phase 2 spec: docs/v3/phase2_tools_design.md Tool 4.

Use at the end of advisor / analyst turns to surface 2-4 short Thai
follow-up phrases the user can tap. Each item becomes a chip that
re-sends its label as the next user message.

Schema constraints (matches `docs/mint_agentic_v3_pure_react_spec.html`
section 8 schema):
- Cap at 4 items — any extras are dropped silently.
- Each item is trimmed to 60 chars.
- Empty list → structured error (do NOT emit an empty `suggestions` block,
  mobile renders it as a bug).
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog


_STATUS_WORD = "กำลังคิดคำถามต่อ..."
_MAX_ITEMS = 4
_MAX_ITEM_LEN = 60


@tool
async def emit_suggestions(
    items: list[str],
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Emit follow-up suggestions for the user. Use at end of turn for
    advisor/analyst flows.

    Args:
      items: 2-4 short Thai phrases (e.g. ["ตั้ง budget เดือนนี้",
             "ดูเทรนด์ 3 เดือน"]). Items beyond the 4th are dropped;
             items longer than 60 chars are trimmed.

    Returns:
      Command(update={
        "emitted_blocks_this_turn": [<suggestions block>],
        "messages": [ToolMessage(JSON of {"suggestions_emitted": int})],
      })
      OR Command with {"error": ..., "kind": "empty_suggestions"} payload
      (and NO emitted_blocks_this_turn update) when cleanup leaves nothing.
    """
    _emit_status(_STATUS_WORD)

    # Trim + cap. None items are filtered defensively.
    cleaned: list[str] = []
    for raw in (items or []):
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        cleaned.append(text[:_MAX_ITEM_LEN])
        if len(cleaned) >= _MAX_ITEMS:
            break

    if not cleaned:
        slog("emit_suggestions", "empty after cleanup — returning error")
        return Command(update={
            "messages": [
                ToolMessage(
                    content=json.dumps({"error": "no items provided",
                                        "kind": "empty_suggestions"}),
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    block = {
        "type": "suggestions",
        "items": [{"label": s, "send": s} for s in cleaned],
    }
    # NOTE: do NOT mutate state.emitted_blocks_this_turn in-place. LangGraph
    # injects the live channel value as `state`; an in-place append duplicates
    # the block because the append_reducer also applies our Command(update=)
    # below, producing two identical blocks on the SSE wire (issue #SSE-DUP).
    # Tests that need to observe the block read it from cmd.update instead.
    slog("emit_suggestions", f"emitted {len(cleaned)} chip(s)")

    return Command(update={
        "emitted_blocks_this_turn": [block],
        "messages": [
            ToolMessage(
                content=json.dumps({"suggestions_emitted": len(cleaned)}),
                tool_call_id=tool_call_id,
            ),
        ],
    })


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


__all__ = ["emit_suggestions"]
