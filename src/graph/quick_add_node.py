"""Quick-add subgraph node — text-only counterpart to slip_node.

Triggered by `intent_classifier_node` when the user typed a short
journal-style entry like "กิน kfc 100บาท". One LLM hop binds
`propose_transaction` over the user's wallet + category catalog,
then `propose_validation_node` (reused from the slip lane) validates
the ids and dispatches the SSE card to mobile.

We deliberately skip the slip prompt's VAT/discount/payslip chain —
text entries are always a single line item. If the user wants to
split a receipt they should use the slip flow.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.debug_log import LogLevel as _LogLevel, log as _log
from src.entity_catalog import (
    fetch_categories_by_wallet,
    fetch_user_catalog,
    render_for_slip,
)
from src.graph.state import AgentState
from src.llm import transaction_llm
from src.tools.transaction import propose_transaction

_logger = logging.getLogger(__name__)


def _qa_log(tag: str, msg: str, **kwargs: Any) -> None:
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


def _build_quick_add_prompt(current_date: str, wallet_category_map: str) -> str:
    """Multi-turn prompt — ask back when required fields are missing,
    fire `propose_transaction` when the user has provided enough info.
    """
    return f"""You are a quick-add agent inside Mint Money.

**Today's date:** {current_date}

The user is logging a single spending/income event through chat.
This may take ONE turn ("กิน kfc 100บาท" → done) or MULTIPLE turns
("เที่ยว" → you ask for amount → user replies "200" → done). You
will see the full recent conversation each turn — merge what the
user said earlier with the latest reply before deciding what to do.

## Wallets + their categories (catalog for matching)

ONLY `general` (cash/bank) wallets are listed — credit-card and goal
wallets are hidden. Every `sync_id` you pass to the tool MUST appear
verbatim below. Never fabricate one.

{wallet_category_map}

## REQUIRED fields (gate the tool call on these two)

1. **name / note** — what was bought, paid, or received. A short
   Thai noun phrase. Examples: "เที่ยว", "กินข้าว", "ค่าไฟ",
   "ค่ากาแฟ", "เงินเดือน". The user's first turn almost always
   gives you this.
2. **amount** — a positive number. May be in the first turn
   ("กิน kfc 100") or in a follow-up turn after you ask.

If EITHER is missing, do NOT call the tool. Ask back instead
(see "Asking back" below).

## OPTIONAL fields — apply defaults silently

The user usually omits these. Pick a reasonable default WITHOUT
asking the user:

- **`type`** — default `expense`. Use `income` if the name clearly
  implies receiving money (เงินเดือน, โบนัส, รับโอน, ได้เงิน).
- **`date`** — default today ({current_date}). Quick-add never
  back-dates; slip / manual form handle that.
- **`wallet_id`** — pick using this cascade:
  1. Explicit wallet name in any user turn (e.g. "จ่าย 100 จาก KBank").
  2. Bank/payment-app keyword (KBank, SCB, TrueMoney) → wallet whose
     name contains that keyword.
  3. Functional fit — wallet whose name best fits the transaction
     purpose (e.g. "ค่าใช้จ่ายบ้าน" for utilities).
  4. Fallback — the first wallet in the catalog.
- **`category_id`** — pick a category from the matched wallet's list
  whose `type` matches. Prefer the most specific name that fits; use
  a generic catch-all ("อาหาร", "ช้อปปิ้ง") only when no narrower
  name applies. "อื่นๆ"/"Other" is a LAST resort.
- **`currency_code` / `currency_symbol`** — always THB / ฿.
- **`merchant_name`** — fill if the user named a brand/vendor (KFC,
  Starbucks); else leave empty.
- **`include_in_report`** — always true.

DO NOT ask the user about any optional field. The mobile card lets
them edit anything before saving.

## Asking back (when required fields are missing)

Reply with ONE short Thai sentence that:
1. **Recaps** what you captured already (the name, plus any optional
   fields the user mentioned).
2. **Asks for the missing required field** (almost always amount).

Format: `รับทราบว่าจะบันทึก '<name>'<optional bits> จำนวนเงินเท่าไหร่ครับ?`

Examples:
- User said "เที่ยว" → reply
  `รับทราบว่าจะบันทึก 'เที่ยว' จำนวนเงินเท่าไหร่ครับ?`
- User said "กินข้าว ที่ MK" → reply
  `รับทราบว่าจะบันทึก 'กินข้าว' ที่ MK จำนวนเงินเท่าไหร่ครับ?`
- User said "ค่าไฟ จาก KBank" → reply
  `รับทราบว่าจะบันทึก 'ค่าไฟ' จาก KBank จำนวนเงินเท่าไหร่ครับ?`

Do NOT call the tool. Do NOT add suggestions. Do NOT add fluff
sentences ("ได้เลยครับ", "เดี๋ยวบันทึกให้นะคะ"). Just the recap +
question.

If only the AMOUNT is given without a name ("100", "200 บาท") and
there is NO name in the prior conversation either, reply:
`กรุณาบอกด้วยว่าจะบันทึกอะไรครับ (เช่น "ค่ากาแฟ" / "เที่ยว")`

## Firing the tool (when both required fields are present)

Once you have both `name` AND `amount` (across the whole conversation
window — merge prior turns with the current reply), call
`propose_transaction` EXACTLY ONCE with:
- `type` = expense | income (per the rules above)
- `amount` = positive number
- `date` = "{current_date}" (unless user explicitly back-dated)
- `note` = the name from the user (Thai, short)
- `wallet_id` = matched sync_id (cascade above)
- `category_id` = matched sync_id (cascade above)
- `merchant_name` = brand/vendor string, else empty
- `currency_code` = "THB"
- `currency_symbol` = "฿"
- `include_in_report` = true

After the tool call, reply with exactly this Thai sentence:
"ดูข้อมูลในการ์ดด้านบนได้เลยครับ — กดบันทึกถ้าถูกต้อง"

Do NOT restate the transaction details. Do NOT emit a `<suggestions>`
tag. Do NOT call any other tool.
"""


# How many recent messages to feed the LLM. Quick-add conversations
# are usually 2-3 turns ("เที่ยว" → AI asks → "200" → AI fires); 8
# leaves plenty of headroom for back-and-forth without bloating the
# token budget. Older turns are irrelevant — the user already moved
# on if they're back in quick-add mode.
_HISTORY_WINDOW = 8


async def quick_add_node(state: AgentState, config: RunnableConfig) -> dict:
    """Text → ask-back OR propose_transaction tool call.

    Multi-turn: the LLM sees the recent conversation window so it can
    merge a prior "เที่ยว" with a follow-up "200" into one transaction.

    Returns:
        - `messages: [AIMessage with tool_calls]` when enough info is
          present (propose_validation_node executes next).
        - `messages: [AIMessage with text only]` when something is
          missing (graph routes straight to END; user replies; next
          turn re-enters this node with the reply included).
    """
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("user_id is required for quick_add_node")

    msgs = state.get("messages") or []
    if not msgs:
        raise ValueError("quick_add_node invoked with no messages")

    catalog = await fetch_user_catalog(user_id)
    by_wallet = await fetch_categories_by_wallet(user_id)
    wallet_category_map = render_for_slip(catalog, by_wallet)

    current_date = date.today().isoformat()
    system_msg = SystemMessage(
        content=_build_quick_add_prompt(current_date, wallet_category_map)
    )

    # Pass a sliding window of recent messages so the LLM can stitch
    # together a multi-turn quick-add. We drop empty AIMessages and
    # ToolMessages — they're routing artifacts, not conversational
    # signal, and including them confuses the model.
    history = _recent_dialog(msgs, _HISTORY_WINDOW)
    last_user_preview = ""
    for m in reversed(history):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                last_user_preview = content[:120]
            break
    _qa_log(
        "QUICK_ADD",
        "node entered",
        user_id=user_id,
        history_len=len(history),
        last_user_preview=last_user_preview.replace("\n", " "),
        _level=_LogLevel.MILESTONE,
    )

    llm_with_tool = transaction_llm.bind_tools([propose_transaction])

    started = time.monotonic()
    response = await llm_with_tool.ainvoke([system_msg, *history])
    duration_ms = int((time.monotonic() - started) * 1000)

    tool_calls = getattr(response, "tool_calls", None) or []
    text_preview = _text_preview(response)
    _qa_log(
        "QUICK_ADD",
        "LLM response",
        duration_ms=duration_ms,
        tool_calls=len(tool_calls),
        ask_back=not tool_calls,
        text_preview=text_preview[:200],
        _level=_LogLevel.MILESTONE,
    )

    # No tool_call = ask-back. Persist the AIMessage so the next user
    # turn has it as context for the classifier + this node. Empty
    # text is a model glitch — surface a generic Thai ask so the user
    # never sees a blank reply.
    if not tool_calls and not text_preview.strip():
        response = AIMessage(
            content="กรุณาระบุจำนวนเงินด้วยครับ (เช่น 100 บาท)"
        )

    return {
        "messages": [response],
        "user_id": user_id,
        "current_date": current_date,
    }


def _recent_dialog(msgs: list, window: int) -> list:
    """Take the last `window` messages from the CURRENT quick-add
    session only.

    A previous AIMessage carrying `tool_calls` marks the moment a
    transaction card was dispatched to mobile — the user has saved or
    rejected it on the mobile UI by now and that conversation is
    closed. Anything before that boundary must NOT bleed into the
    new session: a user typing "เที่ยว" after a saved "กิน kfc 100"
    must NOT have "kfc" merged into their next transaction.

    Algorithm: drop everything up to and including the last AIMessage
    with tool_calls; from what remains, keep only Human/AI text
    messages (Tool messages and empty AI messages are routing
    artifacts that confuse the LLM). Also drop `[INTENT:...]` markers
    from mobile — those are system signals, not conversational text.
    """
    # Find the index of the most recent AIMessage with tool_calls —
    # that's the session boundary. Anything ≤ this index is closed.
    cutoff = -1
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            cutoff = i
    relevant = msgs[cutoff + 1:] if cutoff >= 0 else msgs

    filtered = []
    for m in relevant:
        if isinstance(m, HumanMessage):
            text = _text_preview(m).strip()
            if text.startswith("[INTENT:"):
                # Routing signal from mobile — not user speech.
                continue
            filtered.append(m)
        elif isinstance(m, AIMessage):
            text = _text_preview(m).strip()
            if text:
                # Strip tool_calls before re-sending so the model isn't
                # tempted to "complete" a stale, partial tool chain.
                filtered.append(AIMessage(content=text))
    return filtered[-window:]


def _text_preview(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content[:200].replace("\n", " ")
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                return str(blk.get("text", ""))[:200].replace("\n", " ")
    return ""
