"""Transaction confirmation node.

Fires when mobile sends a system marker after the user pressed save
or delete on a transaction card we proposed earlier. The marker
arrives as the latest HumanMessage:

    [INTENT:transaction_saved] group_id=<uuid>
    [INTENT:transaction_dismissed] group_id=<uuid>

The node walks back through chat history to find the matching
proposal (most recent AIMessage with `tool_calls`), extracts the
proposed items, then asks the chat LLM to generate a short warm
Thai confirmation that references the actual transaction.

We deliberately don't validate the group_id against a backend store —
quick-add proposals aren't persisted server-side; they live on
mobile until sync. The chat history is the source of truth for
"what was on the card the user just acted on", which is the most
recent tool_call AIMessage.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.debug_log import LogLevel as _LogLevel, log as _log
from src.graph.state import AgentState
from src.llm import transaction_llm


SAVED_MARKER_RE = re.compile(r"\[INTENT:transaction_saved\]")
DISMISSED_MARKER_RE = re.compile(r"\[INTENT:transaction_dismissed\]")


def _cn_log(tag: str, msg: str, **kwargs: Any) -> None:
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


def is_confirmation_marker(text: str) -> bool:
    """True when `text` is one of the save/dismiss markers from mobile."""
    if not text:
        return False
    return bool(SAVED_MARKER_RE.search(text) or DISMISSED_MARKER_RE.search(text))


def _classify_action(text: str) -> str:
    """Return 'saved' or 'dismissed' from the marker text."""
    if DISMISSED_MARKER_RE.search(text):
        return "dismissed"
    return "saved"


def _find_prior_tool_call(msgs: list) -> list[dict]:
    """Walk backwards past the current HumanMessage marker and find
    the most recent AIMessage with `tool_calls`. Return the parsed
    propose_transaction items (an empty list if none found).
    """
    seen_current_human = False
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and not seen_current_human:
            # Skip the marker HumanMessage at the tail.
            seen_current_human = True
            continue
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            if not tool_calls:
                continue
            items: list[dict] = []
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name != "propose_transaction":
                    continue
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                items.append(dict(args or {}))
            if items:
                return items
    return []


def _render_items_for_prompt(items: list[dict]) -> str:
    """One line per item — feed the LLM enough to write a natural reply."""
    if not items:
        return "(no transaction details available in history)"
    lines = []
    for it in items:
        type_ = it.get("type", "expense")
        amount = it.get("amount", 0)
        currency = it.get("currency_code", "THB")
        note = (it.get("note") or "").strip()
        merchant = (it.get("merchant_name") or "").strip()
        bits = [f"type={type_}", f"amount={amount} {currency}"]
        if note:
            bits.append(f"note='{note}'")
        if merchant:
            bits.append(f"merchant='{merchant}'")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines)


SAVED_PROMPT = """You are Mint Money's chat assistant. The user just pressed
SAVE on a transaction card you proposed. Below is what they saved:

{items}

Write ONE short, warm Thai sentence confirming the save. Rules:
- Reference the transaction naturally — mention the merchant or note
  and the amount if there's a single item. For multiple items, say
  "บันทึก N รายการ" + the total.
- Use a positive, friendly tone ("เรียบร้อย", "เก็บไว้แล้ว").
- End with a tick "✓" or no punctuation at all.
- NO suggestions tag, NO follow-up questions, NO restating fields
  the user already saw on the card.
- Output only the sentence. No markdown, no quotes, no preamble."""


DISMISSED_PROMPT = """You are Mint Money's chat assistant. The user just pressed
DELETE / DISMISS on a transaction card you proposed. Below is what
they removed:

{items}

Write ONE short, friendly Thai sentence acknowledging the dismissal.
Rules:
- Reference the transaction lightly if helpful ("ยกเลิก '<note>'
  แล้วครับ"). Don't lecture, don't ask why.
- Tone: neutral, no judgement, no guilt-tripping.
- Offer to log a new one if natural ("ถ้าจะบันทึกใหม่ บอกได้เลยครับ")
  but keep the whole reply to ≤ 25 Thai words total.
- NO suggestions tag, NO follow-up questions about analytics, NO
  emojis, NO markdown.
- Output only the sentence(s). No quotes, no preamble."""


async def confirmation_node(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the AI's reply when the user saves/deletes a card."""
    msgs = state.get("messages") or []
    if not msgs:
        return {"messages": [AIMessage(content="รับทราบครับ")]}

    # The marker is the latest HumanMessage.
    marker_text = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                marker_text = content
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        marker_text = str(blk.get("text", ""))
                        break
            break

    action = _classify_action(marker_text)
    items = _find_prior_tool_call(msgs)
    rendered = _render_items_for_prompt(items)

    _cn_log(
        "CONFIRM",
        "node entered",
        action=action,
        item_count=len(items),
        marker_preview=marker_text[:80].replace("\n", " "),
        _level=_LogLevel.MILESTONE,
    )

    template = SAVED_PROMPT if action == "saved" else DISMISSED_PROMPT
    system_msg = SystemMessage(content=template.format(items=rendered))
    # Empty body — the system prompt carries all the data the LLM
    # needs. We still send a HumanMessage stub because most chat
    # endpoints reject system-only inputs.
    human_msg = HumanMessage(
        content="(user just pressed " + action + " on the card)"
    )

    started = time.monotonic()
    response = await transaction_llm.ainvoke([system_msg, human_msg])
    duration_ms = int((time.monotonic() - started) * 1000)

    text = ""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text = str(blk.get("text", "")).strip()
                break

    if not text:
        # Belt-and-suspenders fallback so the user never sees an empty turn.
        text = (
            "บันทึกเรียบร้อยแล้วครับ ✓"
            if action == "saved"
            else "ยกเลิกรายการแล้วครับ"
        )
        response = AIMessage(content=text)

    _cn_log(
        "CONFIRM",
        "LLM response",
        action=action,
        duration_ms=duration_ms,
        text_preview=text[:200],
        _level=_LogLevel.MILESTONE,
    )

    return {"messages": [response]}
