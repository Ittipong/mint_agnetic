"""Entry-router intent classifier.

Runs on every text-only turn before the ReAct loop to decide whether
the user wants to RECORD a new transaction (quick-add lane) or do
anything else (regular ReAct lane).

Why a dedicated classifier instead of giving propose_transaction to
the ReAct LLM directly: comment in `nodes.py` (the "intentionally NOT
here" block) — exposing the tool to the chat LLM produces hallucinated
cards on plain-text intent markers. We separate the decision from the
execution so quick-add only fires when a cheap, specialized model says
the user's intent is clearly "add transaction".

Failure mode: any error (timeout, parse failure, API down) falls back
to `other` so the user lands in the regular reason loop. The quick-add
lane is an optimization — losing it must not break chat.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config import settings
from src.debug_log import LogLevel as _LogLevel, log as _log
from src.graph.state import AgentState
from src.llm import (
    cached_system_content,
    intent_classifier_fallback_llm,
    intent_classifier_llm,
)


def _ic_log(tag: str, msg: str, **kwargs: Any) -> None:
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


_SYSTEM_PROMPT = """You classify a user turn into one of two intents
for a Thai personal-finance chat app. The user turn may be standalone
OR a follow-up to a previous assistant question — context matters.

Output STRICT JSON only — no prose, no markdown — exactly:
{"intent": "add_transaction"} or {"intent": "other"}

## add_transaction
The user is telling the app to RECORD a new spending or income event.
A turn qualifies if ANY of these hold:

1. **Amount + activity** — explicit money figure with a verb/noun of
   spending/eating/buying/paying/receiving (e.g. "กิน", "ซื้อ",
   "จ่าย", "โอน", "เติม", "ได้เงิน", "รับเงิน").
2. **Bare transaction noun** — a short noun phrase that names a
   common expense/income category by itself, with no question mark
   and no question word. Treat these as the user starting to log a
   transaction but forgetting the amount — the next turn will ask
   for it. Examples: "เที่ยว", "กินข้าว", "น้ำมัน", "ค่าน้ำ",
   "ค่าไฟ", "ค่าเทอม", "ค่ารถ", "เงินเดือน", "โบนัส", "ค่ากาแฟ".
3. **Follow-up amount in an active add_transaction conversation** —
   if the LAST assistant message asked for a missing field (amount,
   merchant, etc.) for a transaction the user just started, treat
   the current user turn as the continuation. Even a bare number
   like "200" or "100 บาท" counts here.
4. **Explicit save/record command** — user directly tells the app to
   record/save something, even without any item/amount detail. The
   quick-add lane will follow up by asking what to record. Cover the
   obvious verbs AND semantically-similar paraphrases:
   - Thai: "บันทึก", "บันทึกรายการ", "บันทึกรายจ่าย",
     "บันทึกรายได้", "บันทึก transaction", "จด", "จดให้หน่อย",
     "ลงรายการ", "เพิ่มรายการ", "เก็บรายการ", "เซฟ", "เซฟไว้",
     "อัพเดทรายจ่าย", "อัพเดทรายได้".
   - English: "save", "save transaction", "record", "log", "add".
   - Synonyms / casual paraphrases that mean the same thing (e.g.
     "ช่วยจดทีนะ", "เก็บข้อมูลให้หน่อย", "ลงให้หน่อย") also count.

   CRITICAL: it must be the **verb / imperative form** ("please
   record"). If "บันทึก" appears as a **noun referring to past
   records** ("ดูบันทึก" = view records, "บันทึกของเดือนที่แล้ว"
   = last month's records) it is a QUERY → "other".

## other
Everything else: analytics questions, advice requests, chit-chat,
greetings, summaries, comparisons, anything ending in a question
mark or starting with a question word.

Examples (→ add_transaction):
- "กิน kfc 100บาท"             (rule 1)
- "จ่ายค่าไฟ 100"               (rule 1)
- "ได้เงินเดือน 35000"          (rule 1)
- "เที่ยว"                       (rule 2 — bare noun)
- "กินข้าว"                      (rule 2)
- "น้ำมัน"                       (rule 2)
- "ค่าน้ำ"                       (rule 2)
- "200" (after assistant asked "จำนวนเงินเท่าไหร่ครับ?")  (rule 3)
- "100 บาท" (after assistant asked for amount)             (rule 3)
- "kbank" (after assistant asked which wallet)             (rule 3)
- "บันทึกให้ฉันหน่อย"           (rule 4 — explicit save command)
- "บันทึกรายการ"                (rule 4)
- "บันทึกรายจ่าย"               (rule 4)
- "save transaction"            (rule 4 — English)
- "ช่วยจดทีนะ"                   (rule 4 — synonym for record)
- "เซฟไว้หน่อย"                  (rule 4 — semantic paraphrase)
- "อัพเดทรายจ่ายให้"            (rule 4 — semantic paraphrase)
- "ลงรายการให้หน่อย"            (rule 4)

Examples (→ other):
- "เดือนนี้ใช้เงินไปเท่าไหร่"
- "ขอดูรายการอาหารหน่อย"
- "ยอดเงินใน wallet KBank เหลือเท่าไหร่"
- "สวัสดี"
- "ขอบคุณ"
- "เปรียบเทียบกับเดือนที่แล้ว"
- "กิน kfc อร่อยไหม"   (question — ends with ไหม)
- "อยากกิน kfc"         (future intent, not a recorded event)
- "200" (with NO prior assistant question asking for amount)
- "เที่ยวที่ไหนดี"      (question word "ที่ไหน")
- "ดูบันทึกย้อนหลัง"   (noun form — query past records, not save)
- "บันทึกของเดือนที่แล้วเท่าไหร่"  (noun + question — query)

When in doubt → "other". The cost of misclassifying an analytics
question as add_transaction is much higher (wrong card shown) than
misclassifying an add as analytics (user re-types). EXCEPTION: when
the user gives an explicit save command (rule 4), default to
add_transaction — the quick-add lane will gracefully ask for the
missing detail.

## Input format
You will receive the conversation context as JSON in the user message:
{
  "last_assistant": "<previous assistant message text, or empty>",
  "user": "<the current user turn>"
}
Classify the `user` field using both fields as evidence."""


async def classify_intent_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Classify the latest user turn → 'add_transaction' or 'other'.

    Sends both the latest HumanMessage AND the last AIMessage text to
    the classifier so it can recognize follow-ups in an active
    add_transaction conversation (e.g. user replies "200" after the
    assistant asked "จำนวนเงินเท่าไหร่ครับ?"). Without the AI context
    a bare number always classifies as "other" and the multi-turn
    quick-add lane breaks.
    """
    msgs = state.get("messages") or []
    user_text = _last_human_text(msgs)
    last_ai_text = _last_ai_text(msgs)

    user_text = (user_text or "").strip()
    if not user_text:
        _ic_log("INTENT", "empty user text — defaulting to other")
        return {"intent": "other"}

    # The slip flow uses an explicit marker we must never re-route.
    # Belt-and-suspenders: the entry router already excludes image
    # turns, but if a marker ever reaches this node it must pass through.
    if user_text.startswith("[INTENT:"):
        _ic_log("INTENT", "explicit intent marker — defaulting to other", marker=user_text[:40])
        return {"intent": "other"}

    _ic_log(
        "INTENT",
        "classifying",
        user_text_preview=user_text[:120].replace("\n", " "),
        has_last_ai=bool(last_ai_text),
        _level=_LogLevel.MILESTONE,
    )

    # Build the structured user message — JSON so the model parses both
    # fields unambiguously rather than guessing where context ends.
    context_payload = json.dumps(
        {"last_assistant": (last_ai_text or "")[:300], "user": user_text},
        ensure_ascii=False,
    )

    # The system prompt is multi-KB and 100% static — cache it on every
    # turn so Nova (which doesn't auto-cache like DeepSeek) only pays full
    # price on the first turn of a thread. `cached_system_content` returns
    # plain text for providers that ignore the marker, so this is safe to
    # apply unconditionally.
    messages = [
        SystemMessage(
            content=cached_system_content(
                _SYSTEM_PROMPT, settings.intent_classifier_model
            )
        ),
        HumanMessage(content=context_payload),
    ]

    intent, source, duration_ms = await _classify_with_fallback(messages)
    _ic_log(
        "INTENT",
        "classified",
        intent=intent,
        duration_ms=duration_ms,
        source=source,
        _level=_LogLevel.MILESTONE,
    )
    return {"intent": intent}


async def _classify_with_fallback(messages: list) -> tuple[str, str, int]:
    """Run the primary classifier; on any failure retry on the optional
    fallback. Returns (intent, source, total_duration_ms).

    `source` distinguishes "primary", "fallback", "primary_failed_default",
    or "fallback_failed_default" so logs can attribute classification
    decisions back to a specific model.
    """
    started = time.monotonic()
    try:
        response = await intent_classifier_llm.ainvoke(messages)
        raw = (getattr(response, "content", "") or "").strip()
        intent = _parse_intent(raw)
        return intent, "primary", int((time.monotonic() - started) * 1000)
    except Exception as exc:
        primary_ms = int((time.monotonic() - started) * 1000)
        _ic_log(
            "INTENT",
            "primary classifier failed",
            duration_ms=primary_ms,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            has_fallback=intent_classifier_fallback_llm is not None,
            _level=_LogLevel.MILESTONE,
        )

    if intent_classifier_fallback_llm is None:
        return "other", "primary_failed_default", int(
            (time.monotonic() - started) * 1000
        )

    try:
        response = await intent_classifier_fallback_llm.ainvoke(messages)
        raw = (getattr(response, "content", "") or "").strip()
        intent = _parse_intent(raw)
        return intent, "fallback", int((time.monotonic() - started) * 1000)
    except Exception as exc:
        _ic_log(
            "INTENT",
            "fallback classifier also failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            _level=_LogLevel.MILESTONE,
        )
        return "other", "fallback_failed_default", int(
            (time.monotonic() - started) * 1000
        )


def _last_human_text(msgs: list) -> str:
    """Find the most recent HumanMessage's text content, skipping
    system markers like `[INTENT:transaction_saved]`. Markers are
    routing signals from mobile, not the user's voice.
    """
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            text = _text_of(m)
            if text.strip().startswith("[INTENT:"):
                continue
            return text
    return ""


def _last_ai_text(msgs: list) -> str:
    """Find the most recent AIMessage text from the CURRENT session.

    Two kinds of session boundary stop the search:
    1. **AIMessage with tool_calls** — a quick-add proposal was
       dispatched as a transaction card; the user has since saved or
       rejected it on mobile, so anything before is closed.
    2. **HumanMessage that's a marker** — `[INTENT:transaction_saved
       /dismissed]`. The confirmation AIMessage that follows belongs
       to the closed session, not to the new turn — skip it.

    Returning "" tells the classifier "no prior question to anchor
    against" so it judges the user's text on its own merits.
    """
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            if getattr(m, "tool_calls", None):
                return ""
            text = _text_of(m)
            if text.strip():
                return text
        elif isinstance(m, HumanMessage):
            text = _text_of(m)
            if text.strip().startswith("[INTENT:"):
                # Marker = session boundary. Any earlier AI message
                # belongs to a closed conversation.
                return ""
    return ""


def _text_of(message) -> str:
    """Pull plain text out of a Human/AIMessage (string or list content)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                return str(blk.get("text", ""))
    return ""


def _parse_intent(raw: str) -> str:
    """Tolerate minor format slips. Anything not parseable → 'other'.

    Gemini Flash Lite occasionally wraps JSON in ```json fences or
    leaks a stray space. We strip those before json.loads. If parsing
    still fails we substring-scan as a last resort so a model that
    answers 'add_transaction' bare still works.
    """
    if not raw:
        return "other"
    text = raw.strip()
    if text.startswith("```"):
        # Strip triple-backtick fences with optional language tag.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            val = obj.get("intent")
            if val == "add_transaction":
                return "add_transaction"
            return "other"
    except json.JSONDecodeError:
        pass
    # Substring fallback — only triggers when JSON parse failed.
    lowered = text.lower()
    if "add_transaction" in lowered:
        return "add_transaction"
    return "other"
