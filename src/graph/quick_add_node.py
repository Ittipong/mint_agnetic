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

import json
import logging
import re
import time
import uuid
from datetime import date
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from src.config import settings
from src.debug_log import LogLevel as _LogLevel, log as _log
from src.entity_catalog import (
    fetch_categories_by_wallet,
    fetch_user_catalog,
    render_for_slip,
)
from src.graph.state import AgentState
from src.llm import cached_system_content, transaction_llm
from src.tools.transaction import propose_transaction

_logger = logging.getLogger(__name__)


def _qa_log(tag: str, msg: str, **kwargs: Any) -> None:
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


def _build_quick_add_prompt(
    current_date: str,
    wallet_category_map: str,
    currency_code: str,
    currency_symbol: str,
) -> str:
    """Multi-turn prompt — ask back when required fields are missing,
    fire `propose_transaction` when the user has provided enough info.

    `currency_code` / `currency_symbol` come from the user's app
    settings (forwarded on every chat request). The prompt does NOT
    contain any THB-specific rule any more — the LLM just copies the
    forwarded values into the tool call.
    """
    return f"""You are a quick-add agent inside Mint Money.

**Today's date:** {current_date}

## OUTPUT FORMAT — CRITICAL

You have ONLY two valid response shapes:
1. **Tool call** — invoke `propose_transaction` via the tool-calling API.
   Do not echo its arguments in assistant text.
2. **Plain Thai text** — a short ask-back sentence when a required
   field is missing. No JSON, no braces, no key/value pairs.

NEVER write JSON, dict syntax, or anything that starts with `{{` in
the assistant content. NEVER write the tool call as Python function
syntax in the assistant content either. If you find yourself about
to type `propose_transaction(type=...)` or `{{"type": "..."}}`,
STOP — fire the tool through the tool-calling API.

## Scope — ADD ONLY (no in-chat editing)

Quick-add is the **add-transaction** lane only. If the user is
trying to correct a card that's already on screen (the mobile
client opens an Edit sheet for that), an earlier guard intercepts
the turn before this prompt runs — you will never see it. So treat
EVERY turn that reaches you as a fresh ADD intent. There is no
`corrects_group_id` to worry about; do not set it.

The user is logging a single spending/income event through chat.
This may take ONE turn ("กิน kfc 100บาท" → done) or MULTIPLE turns
("เที่ยว" → you ask for amount → user replies "200" → done). You
will see the full recent conversation each turn — merge what the
user said earlier with the latest reply before deciding what to do.

## Currency defaults (from the user's app settings)

- `currency_code` = `{currency_code}`
- `currency_symbol` = `{currency_symbol}`

Copy these EXACT strings verbatim into the tool call. NEVER substitute
a different currency code (USD, EUR, RUB, …) or a different symbol
(₽, $, €, …), even if the user typed something that looks like a
foreign currency word. The mobile client already decided the user's
currency — your job is only to copy it into the tool args.

## Wallets + their categories (catalog for matching)

ONLY `general` (cash/bank) wallets are listed — credit-card and goal
wallets are hidden. Every `sync_id` you pass to the tool MUST appear
verbatim below. Never fabricate one.

{wallet_category_map}

## REQUIRED fields — gate the tool call on EXACTLY these two

1. **name / note** — what was bought, paid, or received. A short
   Thai noun phrase. Examples: "เที่ยว", "กินข้าว", "ค่าไฟ",
   "ค่ากาแฟ", "เงินเดือน". The user's first turn almost always
   gives you this.
2. **amount** — a positive number. May be in the first turn
   ("กิน kfc 100") or in a follow-up turn after you ask. Strip any
   currency symbol or word the user typed ("$", "฿", "บาท",
   "ดอลลาร์") — keep only the numeric value.

**DECISION RULE (read this carefully):**

If BOTH name and amount are available across the conversation
window (count what the user said in ANY prior turn, not just the
latest reply) → CALL THE TOOL IMMEDIATELY. Do not ask anything
else. Optional fields below all have defaults — use them.

If EITHER is missing → ask back (see "Asking back" below).

NEVER ask about anything OTHER than name and amount. Wallet,
category, date, merchant, type — these are NEVER worth a question;
you have defaults for all of them.

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
- **`currency_code` / `currency_symbol`** — use the values supplied
  in the "Currency defaults" section below verbatim. The mobile
  client decides the currency for the user.
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
- `currency_code` = "{currency_code}"   (see "Currency defaults")
- `currency_symbol` = "{currency_symbol}"   (see "Currency defaults")
- `include_in_report` = true
- `corrects_group_id` = ALWAYS null (chat doesn't edit — see "Scope" above)

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

    # Pre-check: if a proposal card is already on screen AND the user's
    # latest message looks like an edit instruction, short-circuit the
    # LLM. Dispatch an `open_edit_sheet` SSE event so mobile opens its
    # in-app editor for the pending card and reply with a one-liner.
    # Chat is intentionally add-only; this avoids the entire class of
    # "LLM-parses-natural-language-edit" failures (which is why we
    # don't have an in-chat correction flow any more).
    pending_group_id = _pending_group_id(msgs)
    if pending_group_id and _looks_like_edit(_latest_user_text(msgs)):
        ack_text = (
            "กำลังเปิดหน้าแก้ไขรายการให้ครับ — ปรับค่าได้ตามต้องการ"
        )
        # Stream the ack as a token so mobile shows a normal assistant
        # bubble explaining WHY the editor sheet is popping up. Without
        # this, the sheet appears out of nowhere and the chat trail has
        # no record of the redirect — confusing UX.
        await adispatch_custom_event("assistant_text", {"text": ack_text})
        # Then tell mobile to actually pop the editor for the matching
        # pending card.
        await adispatch_custom_event(
            "structured_data",
            {
                "type": "open_edit_sheet",
                "data": {"group_id": pending_group_id},
            },
        )
        _qa_log(
            "QUICK_ADD",
            "open_edit_sheet dispatched (chat-edit redirect)",
            group_id=pending_group_id,
            _level=_LogLevel.MILESTONE,
        )
        return {
            "messages": [AIMessage(content=ack_text)],
            "user_id": user_id,
            "current_date": date.today().isoformat(),
        }

    catalog = await fetch_user_catalog(user_id)
    by_wallet = await fetch_categories_by_wallet(user_id)
    wallet_category_map = render_for_slip(catalog, by_wallet)

    current_date = date.today().isoformat()
    currency_code = (state.get("default_currency_code") or "THB").strip() or "THB"
    currency_symbol = (state.get("default_currency_symbol") or "฿").strip() or "฿"

    # Wrap with cache_control so Nova (the configured transaction model)
    # only pays full price for the first turn of a session. The wrapper
    # is a no-op for auto-caching providers (DeepSeek/Gemini), so swapping
    # the model later doesn't break anything.
    system_msg = SystemMessage(
        content=cached_system_content(
            _build_quick_add_prompt(
                current_date,
                wallet_category_map,
                currency_code,
                currency_symbol,
            ),
            settings.transaction_llm_model,
        )
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
    try:
        response = await llm_with_tool.ainvoke([system_msg, *history])
    except Exception as exc:
        # Provider-side failure (OpenRouter "Provider returned error",
        # rate limit, timeout, etc.). If a card is already pending, the
        # user is likely trying to edit it (keyword match missed) — fall
        # back to the same redirect path so they aren't stuck staring at
        # a raw error bubble. Without a pending card we have no safe
        # recovery, so re-raise and let the stream emit its `error` event.
        if pending_group_id:
            _qa_log(
                "QUICK_ADD",
                "LLM error → open_edit_sheet fallback",
                group_id=pending_group_id,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                _level=_LogLevel.MILESTONE,
            )
            ack_text = (
                "กำลังเปิดหน้าแก้ไขรายการให้ครับ — ปรับค่าได้ตามต้องการ"
            )
            await adispatch_custom_event(
                "assistant_text", {"text": ack_text}
            )
            await adispatch_custom_event(
                "structured_data",
                {
                    "type": "open_edit_sheet",
                    "data": {"group_id": pending_group_id},
                },
            )
            return {
                "messages": [AIMessage(content=ack_text)],
                "user_id": user_id,
                "current_date": current_date,
            }
        raise
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

    # Force-set currency_code / currency_symbol in the tool args. The
    # mobile client decides the user's currency (forwarded on every
    # request) — the LLM's value is at best redundant and at worst
    # wrong (DeepSeek-Flash sometimes picks ₽ / ₹ / $ when the user
    # types Thai money phrases like "เงินเดือน"). Slip lane is NOT
    # affected — slip_node has its own LLM that legitimately reads
    # currency off receipts.
    if tool_calls:
        for tc in tool_calls:
            args = tc.get("args") if isinstance(tc, dict) else None
            if isinstance(args, dict):
                args["currency_code"] = currency_code
                args["currency_symbol"] = currency_symbol
        response = AIMessage(
            content=getattr(response, "content", "") or "",
            tool_calls=tool_calls,
        )

    # Defensive recovery: Nova occasionally dumps the tool args as a
    # raw JSON string in `content` instead of invoking the tool API
    # (especially when the prompt grows in size). Detect that shape
    # and synthesize the tool_call so downstream graph nodes don't
    # have to deal with it — mobile would otherwise render the JSON
    # as a plain text bubble.
    if not tool_calls:
        recovered_args = _recover_tool_args_from_text(_full_text(response))
        if recovered_args is not None:
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "propose_transaction",
                    "args": recovered_args,
                    "id": f"recovered_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )
            tool_calls = response.tool_calls
            _qa_log(
                "QUICK_ADD",
                "recovered tool_call from JSON text",
                args_keys=sorted(recovered_args.keys()),
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


# Keywords that signal the user wants to edit the active proposal
# rather than add a new transaction. Match is intentionally narrow —
# the pre-check is gated by "pending card exists", so false positives
# are bounded. Anything richer (regex with word boundaries) is overkill
# for Thai where word boundaries aren't space-delimited.
#
# `ยอด` (amount/total) is included because it almost always refers to
# editing the amount on the pending card — and it also catches the
# common typo "แค่ยอด..." (autocorrect of "แก้ยอด...").
_EDIT_KEYWORDS = (
    "ผิด",
    "ไม่ใช่",
    "แก้",
    "ขอแก้",
    "เปลี่ยน",
    "ที่จริง",
    "ยอด",
)


def _looks_like_edit(text: str) -> bool:
    """True if the user's latest turn reads like a correction to an
    already-on-screen card. Gated by `_pending_group_id` returning
    a non-empty id at the caller site."""
    if not text:
        return False
    return any(kw in text for kw in _EDIT_KEYWORDS)


def _latest_user_text(msgs: list) -> str:
    """Pull the most recent non-marker HumanMessage text."""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            text = _text_preview(m).strip()
            if text.startswith("[INTENT:"):
                continue
            content = getattr(m, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return str(blk.get("text", ""))
            return ""
    return ""


def _pending_group_id(msgs: list) -> str:
    """Return the `group_id` of a still-pending proposal, or "" if
    no proposal is pending.

    A proposal is "still pending" when there is an `AIMessage` with
    `tool_calls` AND no subsequent `[INTENT:...]` marker (the marker
    only appears after mobile fires /chat/intent on Save / Discard).
    The id is read from the trailing `ToolMessage`'s content
    (`(proposal dispatched, group_id=<uuid>)`), backfilled by
    `propose_validation_node`.
    """
    last_tool_idx = -1
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_tool_idx = i

    if last_tool_idx < 0:
        return ""

    for j in range(last_tool_idx + 1, len(msgs)):
        m = msgs[j]
        if isinstance(m, HumanMessage):
            text = _text_preview(m).strip()
            if text.startswith("[INTENT:"):
                return ""  # session closed by Save / Discard
        elif isinstance(m, ToolMessage):
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and "group_id=" in content:
                marker = "group_id="
                idx = content.find(marker)
                tail = content[idx + len(marker):].strip()
                return tail.rstrip(")").split()[0] if tail else ""

    return ""


def _recent_dialog(msgs: list, window: int) -> list:
    """Take the last `window` messages from the CURRENT quick-add
    session only.

    A previous AIMessage carrying `tool_calls` marks the moment a
    transaction card was dispatched to mobile. Two cases:

    1. **Closed proposal** — a later `[INTENT:transaction_saved]` /
       `[INTENT:transaction_dismissed]` HumanMessage follows the
       tool-call AIMessage. The user has acted on the card; the
       conversation is closed and must NOT bleed into the next
       session (a user typing "เที่ยว" after a saved "กิน kfc 100"
       must NOT have "kfc" merged into their new transaction).
    2. **Still pending** — NO intent marker follows. The user is
       likely correcting the just-dispatched proposal ("ผิด ไม่ใช่
       100 แต่ 200"). We must KEEP the pending proposal in context
       so the LLM can read its `group_id` from the trailing
       ToolMessage and pass it back as `corrects_group_id`, which
       lets mobile auto-discard the stale card.

    Algorithm: scan for the last AIMessage with tool_calls. If a
    following HumanMessage with `[INTENT:...]` exists, treat the
    block as closed and cut everything up to and including it.
    Otherwise keep the block intact (including its trailing
    ToolMessages, which carry the resolved group_id). From what
    remains, keep Human/AI text messages PLUS the still-pending
    proposal's AIMessage(tool_calls) + ToolMessage so the LLM can
    reason about a correction.
    """
    # Find the index of the most recent AIMessage with tool_calls.
    last_tool_idx = -1
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_tool_idx = i

    cutoff = -1
    if last_tool_idx >= 0:
        # Look for a `[INTENT:...]` marker AFTER the tool call. If we
        # find one, that proposal is closed — cut up to and including
        # the marker. If we don't, leave `cutoff = -1` so the entire
        # history (including the still-pending proposal block) stays.
        for j in range(last_tool_idx + 1, len(msgs)):
            m = msgs[j]
            if isinstance(m, HumanMessage):
                text = _text_preview(m).strip()
                if text.startswith("[INTENT:"):
                    cutoff = j
                    break
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
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                # Quick-add no longer chains corrections in chat — the
                # pre-check at the top of `quick_add_node` intercepts
                # edits and dispatches an `open_edit_sheet` event
                # instead. So the LLM doesn't need to see the prior
                # tool_call args. We DO replace it with a short text
                # ack though: an `AIMessage(content="")` makes
                # DeepSeek-Flash blank on its next response, which
                # silently drops new add intents.
                filtered.append(
                    AIMessage(content="(เสนอรายการก่อนหน้าแล้ว — รอผู้ใช้กดบันทึก)")
                )
            elif text:
                filtered.append(AIMessage(content=text))
        elif isinstance(m, ToolMessage):
            # Carries `group_id=<uuid>` substring — required reading
            # for the LLM to populate `corrects_group_id` on a
            # follow-up correction. Skip empty ones (routing noise).
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and content.strip():
                filtered.append(m)
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


def _full_text(response: Any) -> str:
    """Full assistant text (not truncated like _text_preview)."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
        return "".join(parts)
    return ""


# Required fields a recovered payload must carry to be usable as a
# tool call. Optional fields fall back to the tool's own defaults.
_REQUIRED_RECOVERY_FIELDS = ("amount",)

# Match the first balanced-looking JSON object in a blob. Nova
# sometimes wraps the JSON in code fences or chats around it; this
# strips those.
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

# Match `propose_transaction(...)` Python-style call. DeepSeek-Flash
# occasionally writes the tool call as pseudo-code in the content
# instead of invoking the tool API — particularly on multi-turn
# corrections after we inject the pending args. Capture group 1 holds
# the inner argument list so the parser below can split key=value pairs.
_PY_CALL_RE = re.compile(
    r"propose_transaction\s*\(\s*([\s\S]*?)\s*\)", re.IGNORECASE
)


def _recover_tool_args_from_text(text: str) -> dict | None:
    """Try to parse a propose_transaction args dict out of free text.

    Returns the args dict on success, None when no recoverable payload
    is present. Two shapes are recovered:
      1. JSON dict (Nova's failure mode)
      2. Python call syntax (DeepSeek's failure mode on corrections)

    The defender lives here so the rest of the graph can stay
    tool-call-only.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    candidates: list[dict] = []

    # ── JSON dict path ─────────────────────────────────────────
    if "{" in stripped:
        json_candidates: list[str] = []
        if stripped.startswith("{"):
            json_candidates.append(stripped)
        match = _JSON_OBJECT_RE.search(stripped)
        if match and match.group(0) not in json_candidates:
            json_candidates.append(match.group(0))
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

    # ── Python call syntax path ────────────────────────────────
    for m in _PY_CALL_RE.finditer(stripped):
        parsed = _parse_python_call_args(m.group(1))
        if parsed:
            candidates.append(parsed)

    for parsed in candidates:
        amount = parsed.get("amount")
        if not isinstance(amount, (int, float)) or amount <= 0:
            continue
        if not all(field in parsed for field in _REQUIRED_RECOVERY_FIELDS):
            continue
        return parsed
    return None


# `key=value` pair parser for the Python-call recovery path. Handles
# the conservative subset DeepSeek emits: bare numbers (int/float),
# bools (`True`/`False`/`true`/`false`), `None`/`null`, and quoted
# strings (single or double quotes). Anything fancier (nested dicts,
# expressions) is intentionally unsupported — we'd rather fall through
# to ask-back than parse arbitrary Python.
_KV_RE = re.compile(
    r"""(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?P<val>"[^"]*"|'[^']*'|[-\d.]+|[A-Za-z_][A-Za-z0-9_]*)\s*,?""",
)


def _parse_python_call_args(arg_text: str) -> dict | None:
    """Parse `k1="v1", k2=2, k3=true, k4=None` into a dict.

    Returns None when nothing parseable was found. Permissive on
    extras (we just skip pairs we can't decode rather than failing
    the whole recovery).
    """
    if not arg_text or not arg_text.strip():
        return None
    out: dict = {}
    for m in _KV_RE.finditer(arg_text):
        key = m.group("key")
        raw = m.group("val")
        out[key] = _coerce_py_value(raw)
    return out or None


def _coerce_py_value(raw: str) -> Any:
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in ("true",):
        return True
    if lowered in ("false",):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
