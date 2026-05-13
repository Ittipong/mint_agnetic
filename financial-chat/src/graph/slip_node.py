"""Slip → transaction subgraph node.

Routes around the ReAct reason loop when the user attached a slip
image. We don't need analyze_user_finances or CodeAct for this turn —
the slip text is the source of truth and the LLM just needs to match
wallet + category from the user's catalog and call propose_transaction.

Why a separate node (not a tool in reason_node):
- The chat LLM (deepseek-v4-flash) is text-only. Routing slips into
  it would force an unconditional upgrade to a vision model for every
  chat turn, which costs more for non-slip questions.
- Vision LLM call shape (content blocks with image_url) differs from
  the regular HumanMessage(str) the chat loop sends.
- Errors in vision processing should not poison the chat thread's
  ReAct state — keeping it isolated means a vision failure surfaces
  cleanly as an SSE error without leaving partial tool calls behind.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.entity_catalog import (
    fetch_categories_by_wallet,
    fetch_user_catalog,
    render_for_slip,
)
from src.graph.state import AgentState
from src.llm import vision_llm
from src.tools.transaction import propose_transaction

_logger = logging.getLogger(__name__)


def _build_slip_system_prompt(
    current_date: str,
    wallet_category_map: str,
) -> str:
    """Single-shot prompt — focused, no ReAct reasoning."""
    return f"""You are a slip-parsing agent inside Mint Money.

**Today's date:** {current_date}

Your only job: look at the attached slip image, extract the
transaction, and call the `propose_transaction` tool **exactly once**
with the matched values. Do NOT describe the transaction in chat — the
mobile client renders it as a card.

## Scope (slip flow only)

This flow handles **bank/cash slips** that hit a real account —
deposits, payments, transfers, payroll deposits, ATM receipts, retail
receipts paid by cash/bank. The wallet list below shows ONLY the
user's general (cash/bank) wallets. Credit-card wallets and goal
wallets are out of scope and intentionally hidden — you must NOT
invent or guess sync_ids for those types.

## Wallets + their categories (for matching)

Use the **sync_id** values verbatim when calling the tool, NOT names.
Every `sync_id` you pass MUST appear in the list below — never
fabricate one. If you cannot find a confident match, set
`wallet_id`/`category_id` to **null** rather than guessing.

{wallet_category_map}

## Matching rules

1. **Wallet** — look for bank logo, account name, account number
   prefix, payment method, or wallet name. Pick a wallet from the
   list above whose name best matches. The wallets listed above are
   already filtered to `type=general` — never suggest a credit-card
   or goal wallet here. If nothing clearly matches, set `wallet_id`
   to null.
2. **Category** — search **only** within the matched wallet's
   category list. Match by merchant/item keywords ("ร้านอาหาร" /
   "อาหาร", "Cafe Amazon" / "เครื่องดื่ม", "BTS" / "เดินทาง", "Shopee"
   / "ช้อปปิ้ง", "เงินเดือน" / "salary"). If the matched wallet has no
   matching category, set `category_id` to null.
3. **Amount** = total paid / total received (look for "รวม", "ยอดรวม",
   "Total", "Amount", "รายได้สุทธิ").
4. **Date** = ISO 8601. If the slip has no timestamp, use today.
   Pay attention to Buddhist Era — if a year looks like 25xx, subtract
   543 to convert to AD before formatting.
5. **Type** — pick by money direction relative to the user:
   - `income` — money INTO the user's account (payroll / payslip /
     incoming transfer / cashback / refund).
   - `expense` — money OUT of the user's account (purchase / outgoing
     transfer / bill payment / withdrawal).
6. **Note** = one-line summary. For receipts with multiple line items,
   list comma-separated (e.g. "ข้าวมันไก่ 60, ชา 25").
7. **Merchant** = store/vendor name (for expense) or employer name
   (for income payslips). Pull from the slip header.

## Failure mode

If the image is **clearly not a slip** (random photo, screenshot of
the app UI, blank, unreadable) → do NOT call the tool. Reply with a
short Thai sentence asking the user to send a real slip, then stop.

## After the tool call

Reply with **one short Thai sentence** like
"ดูข้อมูลในการ์ดด้านบนได้เลยครับ — กดบันทึกถ้าถูกต้อง". Do not restate
the transaction details, do not emit a `<suggestions>` tag.
"""


async def slip_node(state: AgentState, config: RunnableConfig) -> dict:
    """Vision LLM call → propose_transaction → end.

    Single LLM hop. The graph routes here only when `state["images"]`
    is non-empty, so we trust that invariant and don't re-check.
    """
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("user_id is required for slip_node")

    images = state.get("images") or []
    if not images:
        raise ValueError("slip_node invoked with no images")

    # Pull both shapes of the catalog: regular for diagnostics,
    # wallet-grouped for the prompt.
    catalog = await fetch_user_catalog(user_id)
    by_wallet = await fetch_categories_by_wallet(user_id)
    wallet_category_map = render_for_slip(catalog, by_wallet)

    current_date = date.today().isoformat()
    system_msg = SystemMessage(
        content=_build_slip_system_prompt(current_date, wallet_category_map)
    )

    # Pull the latest human message text (typically the intent marker)
    # so the LLM has a place to attach the image content block. If no
    # text exists, send a minimal placeholder — the image is what
    # matters here.
    msgs = state.get("messages") or []
    user_text = ""
    for m in reversed(msgs):
        # Heuristic: HumanMessage carries `.type == "human"` in LC v0.3+
        if getattr(m, "type", None) == "human":
            content = getattr(m, "content", "")
            if isinstance(content, str):
                user_text = content
            break
    if not user_text:
        user_text = "[INTENT:parse_transaction_from_slip]"

    # Build multimodal content blocks. `detail: "low"` keeps token cost
    # down — slip layouts are simple enough that low-res understanding
    # is sufficient for amount/merchant extraction.
    image_blocks: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
        for url in images
    ]
    human_msg = HumanMessage(
        content=[{"type": "text", "text": user_text}, *image_blocks]
    )

    llm_with_tools = vision_llm.bind_tools([propose_transaction])

    # No try/except: any vision LLM failure should propagate so the
    # FastAPI stream wraps it in an SSE `error` event and the mobile
    # client shows a snackbar (matches the "throw error to frontend"
    # requirement — no silent fallback).
    #
    # The vision model occasionally produces an empty response on the
    # first call (no tool_calls AND no text). Retry once before
    # giving up — single retry is cheap and recovers most flakes
    # without leaving the mobile UI hanging.
    response = await llm_with_tools.ainvoke([system_msg, human_msg])
    if _is_empty(response):
        _logger.warning(
            "slip_node: empty response on first call, retrying once"
        )
        response = await llm_with_tools.ainvoke([system_msg, human_msg])

    # Debug trace — surfaces whether the vision LLM produced a tool
    # call or text refusal so failing rounds can be diagnosed without
    # re-running.
    tool_calls = getattr(response, "tool_calls", None) or []
    _logger.warning(
        "slip_node response — image_count=%d tool_calls=%d text=%r",
        len(images),
        len(tool_calls),
        _text_preview(response),
    )

    # If the LLM returned tool calls, propose_validation_node will
    # execute them on the next hop. We just return the message.
    return {
        "messages": [response],
        "user_id": user_id,
        "current_date": current_date,
    }


def _is_empty(response: Any) -> bool:
    """An AIMessage is "empty" when it carries neither a tool call
    nor any text. The vision model occasionally returns this when it
    fails to process the image — usually transient."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return False
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict):
                text = blk.get("text", "")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return True


def _text_preview(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content[:120].replace("\n", " ")
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                return str(blk.get("text", ""))[:120].replace("\n", " ")
    return ""


# ──────────────────────── validation node ────────────────────────


def _combine_note(note: str, merchant: str | None) -> str:
    """Merge the LLM's note with the slip merchant into one user-
    facing field. Returns "" when both are empty — the mobile card
    hides the note row in that case."""
    note = (note or "").strip()
    merchant = (merchant or "").strip()
    if note and merchant:
        return f"{note} — {merchant}"
    return note or merchant


async def propose_validation_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Intercept `propose_transaction` tool calls — validate ids
    against the user's catalog and dispatch the slim SSE payload.

    Why a custom node instead of LangGraph's prebuilt ToolNode:
    1. **Anti-hallucination** — vision LLMs occasionally invent UUIDs.
       We need to confirm `wallet_id` and `category_id` exist for
       this user before the mobile client persists them. Validation
       requires DB access, which the bare `@tool` body doesn't get.
    2. **Slim payload** — mobile expects only sync_ids; display
       fields (wallet name / category name / icon / color) are
       resolved on-device from those ids. The LLM also wants to
       pass `merchant_name` (it sees it in the slip), but the
       mobile card has no merchant slot — so we fold it into `note`.

    Emits the dispatched payload as a `structured_data` custom event;
    the FastAPI stream forwards that as a `data` SSE event.
    """
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("user_id required for propose_validation_node")

    msgs = state.get("messages") or []
    if not msgs:
        return {}
    last = msgs[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {}

    # Load the catalog once — used to validate every tool_call below.
    catalog = await fetch_user_catalog(user_id)
    valid_wallet_ids = {
        w.sync_id for w in catalog.wallets if w.wallet_type == "general"
    }

    # Build the wallet → categories map BEFORE iterating tool_calls so
    # we can validate `category_id` against the matched wallet's
    # categories only. Using the deduped catalog.categories here would
    # false-negative: the same category name exists once per wallet,
    # and dedup keeps only one sync_id, dropping the others as invalid
    # even though they belong to a real wallet.
    by_wallet = await fetch_categories_by_wallet(user_id)

    tool_messages: list[ToolMessage] = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name != "propose_transaction":
            continue
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        args = dict(args or {})

        # ── id validation ───────────────────────────────────────
        wallet_id_in = args.get("wallet_id")
        category_id_in = args.get("category_id")
        wallet_id = wallet_id_in if wallet_id_in in valid_wallet_ids else None
        if wallet_id_in and not wallet_id:
            _logger.warning(
                "propose_validation_node: dropped hallucinated wallet_id=%s",
                wallet_id_in,
            )

        # Category must belong to the matched wallet's category list.
        # If wallet didn't match, no category context — drop the
        # category id rather than guessing. Match by sync_id directly
        # OR by (name, type) when sync_id is a stale duplicate of a
        # categorically equivalent row in the same wallet.
        category_id: str | None = None
        if wallet_id and category_id_in:
            wallet_cats = by_wallet.get(wallet_id, [])
            wallet_cat_ids = {c["sync_id"] for c in wallet_cats}
            if category_id_in in wallet_cat_ids:
                category_id = category_id_in
            else:
                # Stale-duplicate rescue: the LLM may have picked a
                # different sync_id of a category with the same name
                # under a different wallet (catalog dedups by name).
                # Find a same-name row under the matched wallet.
                stale_name = None
                for c in catalog.categories:
                    if c.sync_id == category_id_in:
                        stale_name = c.name
                        break
                if stale_name:
                    for c in wallet_cats:
                        if c["name"] == stale_name:
                            category_id = c["sync_id"]
                            _logger.info(
                                "propose_validation_node: remapped "
                                "category_id %s → %s (same name '%s' "
                                "in matched wallet)",
                                category_id_in, category_id, stale_name,
                            )
                            break
            if not category_id:
                _logger.warning(
                    "propose_validation_node: dropped category_id=%s "
                    "(not in wallet %s)",
                    category_id_in, wallet_id,
                )

        # ── note merge ──────────────────────────────────────────
        final_note = _combine_note(args.get("note"), args.get("merchant_name"))

        # ── slim payload ────────────────────────────────────────
        payload = {
            "type": "propose_transaction",
            "data": {
                "sync_id": str(uuid.uuid4()),
                "type": args.get("type", "expense"),
                "amount": float(args.get("amount") or 0),
                "date": args.get("date"),
                "wallet_id": wallet_id,
                "category_id": category_id,
                "note": final_note,
                "currency_code": args.get("currency_code") or "THB",
                "currency_symbol": args.get("currency_symbol") or "฿",
            },
        }
        await adispatch_custom_event("structured_data", payload)

        tc_id = (
            tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "tc-0")
        )
        tool_messages.append(
            ToolMessage(
                content="(proposal dispatched)",
                tool_call_id=tc_id,
                name="propose_transaction",
            )
        )

    return {"messages": tool_messages}
