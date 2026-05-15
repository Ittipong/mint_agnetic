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
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
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

# ── Debug log (delegates to unified logger) ──────────────────────────
from src.debug_log import LogLevel as _LogLevel, log as _log


def _slip_log(tag: str, msg: str, **kwargs) -> None:
    """Write a slip-flow line via the unified debug logger.

    All slip-internal mechanics (mode, retries, payload shapes) are
    `DETAIL`; production mode shows only the high-signal vision call
    boundaries via dedicated MILESTONE calls. Callers can override
    by passing `_level=_LogLevel.MILESTONE`.
    """
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


def _build_slip_system_prompt(
    current_date: str,
    wallet_category_map: str,
) -> str:
    """Single-shot prompt — three-step chain-of-thought, single call."""
    return f"""You are a slip-parsing agent inside Mint Money.

**Today's date:** {current_date}

Your job: look at the attached image, decide whether it is a slip /
receipt, and turn it into one or more transactions by calling the
`propose_transaction` tool (once per line item). The mobile client
renders each call as a transaction card — do NOT describe them in
chat.

## Scope

This flow accepts **bank transfer slips** and **retail receipts**
(in any language, any format). Anything else — random photos, app
screenshots, blank images, unreadable images — is out of scope:
reply with the exact Thai sentence "ไม่สามารถอ่านสลิปได้" and do
NOT call the tool.

## Wallets + their categories (catalog for matching)

The wallet list below shows ONLY the user's `general` (cash/bank)
wallets — credit-card and goal wallets are intentionally hidden.
Every `sync_id` you pass to the tool MUST appear verbatim in this
catalog — never fabricate one.

{wallet_category_map}

## Workflow — think through these steps in order, then act

### Step 0 — Classify the image into one of five types

Before extracting anything, decide which of these the image is.
This single decision drives merchant extraction, category
mapping, and how many tool calls you emit.

| slip_type        | What it looks like                              |
|------------------|-------------------------------------------------|
| `retail_receipt` | Store header with tax id, **multiple** itemised lines with prices, sub-total/grand total, often cash/change. Examples: 7-Eleven, Lotus, Big C, Tesco, Tops, MaxValu. |
| `restaurant_bill`| Food/drink items, often a table number or order id, may include service charge / VAT. Examples: cafés, ร้านอาหาร, MK, Sukishi. |
| `transfer_slip`  | Bank, e-wallet, or QR-payment confirmation: **single amount**, "โอนเงิน"/"Transfer"/"ชำระ"/"Pay", sender → recipient names, reference number. Examples: KBank / SCB / TrueMoney / PromptPay / ร้านถุงเงิน confirmations. |
| `utility_bill`   | "ใบแจ้งหนี้"/"Invoice"/"Statement" from a service provider — electricity, water, internet, mobile, credit card. Has billing period and/or **due date**, account/customer number. The amount may be marked "ยอดที่ต้องชำระ" / "Total Due". Examples: MEA, PEA, MWA, AIS, TRUE, 3BB, credit-card statements. |
| `reject`         | Not a paid transaction: ใบเสนอราคา (quote), proforma invoice, order page still pending payment, wallet balance screenshot, generic photo, blurry image. |

**If `slip_type == reject`:** reply with the exact Thai sentence
"ไม่สามารถอ่านสลิปได้" and do NOT call the tool. Stop here.

**Ambiguity tiebreakers:**
- A utility/credit-card payment **confirmation slip** (showing
  the bill was paid via a bank/e-wallet) is `transfer_slip`, not
  `utility_bill`. `utility_bill` is the bill itself before/at
  payment.
- A receipt with only 1 line item is still `retail_receipt` /
  `restaurant_bill` — what matters is the format, not the count.
- If a receipt has both food items and a tax-invoice header from
  a non-restaurant chain (e.g. Lotus food court), treat as
  `retail_receipt`.

### Step 1 — Extract raw context (in your head)

Pull these fields from the image:

- `amount` — the money figure for the slip / each line item.
- `date_time` — ISO 8601; Buddhist Era 25xx → subtract 543 for AD.
  Missing → use **today** ({current_date}).
- `transaction_type` — `expense` (money leaves the user) or
  `income` (money enters the user). Missing / ambiguous →
  default to **expense**.
- `merchant_or_recipient_name` — the **actual** counterparty for
  the transaction. The exact source depends on `slip_type`:
    - `retail_receipt` / `restaurant_bill` → the **store name**
      from the header.
    - `transfer_slip` → the **recipient name** (or sender, if
      this is an incoming slip). Strip payment-app wrappers
      such as "ร้านถุงเงิน", "TrueMoney Wallet", "PromptPay",
      "Rabbit LINE Pay", "ShopeePay", "GrabPay", "K PLUS",
      "SCB EASY", and similar banking apps / e-wallets — these
      are the payment rails, not the merchant. If the slip
      shows `<provider> (<real store>)`, extract only
      `<real store>`.
    - `utility_bill` → the **service provider** (the company
      billing you, e.g. MEA, PEA, MWA, AIS, TRUE, 3BB,
      บัตรเครดิตธนาคารกรุงเทพ). NOT the payment rail you used
      to pay. If the bill is unpaid, there is no payment rail
      anyway — only the provider matters.
  Examples:
    "ร้านถุงเงิน (ร้านณสา)"        → "ร้านณสา"
    "TrueMoney Wallet (ร้านสมชาย)" → "ร้านสมชาย"
    "Tesco Lotus"                  → "Tesco Lotus" (no wrapper → keep)
    utility_bill from MEA          → "การไฟฟ้านครหลวง"
    utility_bill from AIS          → "AIS"
  Pull from the slip header / provider block.
- `bank_name` — issuing bank logo / abbreviation (KBANK, SCB,
  TrueMoney, etc.) if visible.
- `account_number` — sender or recipient account if shown.
- `items` — list of line items on a retail receipt with each
  item's name and amount.
- `summary` — one short Thai sentence describing what the slip is
  about (e.g. "ซื้อของชำที่ Tesco", "จ่ายค่ากาแฟที่ Amazon",
  "โอนเงินจาก KBANK ไปยังร้านข้าวมันไก่"). This is your reasoning
  context for picking the wallet and category in steps 2 and 3.
- `raw_note` — any other free text visible on the slip.

**Required:** only `amount`. If you cannot extract `amount`, treat
the image as unreadable — reply with "ไม่สามารถอ่านสลิปได้" and stop.

### Step 1b — Identify the *paid* transaction boundary

A retail slip almost always has TWO sections separated by a
dashed line, blank space, or footer text. Only the **upper
section** is the transaction. Everything else is metadata or
advertising and MUST be ignored.

**Inside the transaction (use these):**
- Merchant header, tax id, branch
- Itemised line items with prices
- Sub-total / VAT line / **applied** discount line
- Grand total (e.g. `ยอดรวม`, `รวมทั้งสิ้น`, `Total`,
  `Grand Total`, `Net Amount`) — this is the **source of truth**
- Cash tendered / change (`เงินสด`, `เงินทอน`, `Cash`, `Change`)
- Date / time of purchase

**Outside the transaction (SKIP — do NOT create tool calls for
these):**
- Loyalty/points info (`แต้มสะสม`, `แต้มจากยอดซื้อ`,
  `บัตรคลับการ์ด`, `Points`, `Reward`)
- Member card numbers and expiry dates of points
- Coupon/promo codes for **future** use — telltale signals:
    · a redemption code string (e.g. `LT15SHOP`, `USE CODE`,
      `ใส่โค้ด`)
    · a future or open-ended date range
    · a condition (`เมื่อช้อปครบ ...`, `เมื่อซื้อครบ`,
      `When you spend …`, `Next purchase`)
    · references an app / website / URL
- Cashback / "earn X back" promotions tied to future spend
- Survey / feedback prompts, social handles, store hours

**Heuristic — applied discount vs promo ad:**
- An **applied discount** appears *above* the grand total and is
  mathematically reflected in it (items − discount = grand total).
- A line that mentions "ส่วนลด … บาท" but appears *below* the
  grand total, has a code/condition/future date, or whose value
  is NOT consistent with `items_sum − grand_total` is a **promo
  advertisement** — skip it.

**Reconciliation check (mandatory before Step 4):**
1. Compute `items_sum` from your extracted line items.
2. Compare with the slip's grand total.
3. If `items_sum == grand_total` → emit items only. Do NOT
   invent a discount line just because the word "ส่วนลด"
   appears anywhere on the slip.
4. If `items_sum − applied_discount == grand_total` → the
   discount above the total is real; emit it as the discount
   line.
5. If numbers cannot be reconciled → trust the grand total and
   emit a single transaction for that amount instead of split
   items.

### Step 2 — Map wallet (`wallet_id`)

Pick exactly one wallet from the catalog for this entire slip
(all line items share the same wallet). Match priority:

1. Best match by **wallet name + bank context** (e.g. slip shows
   "KBANK" → wallet whose name contains "KBank" / "กสิกร").
2. If nothing is a confident match → **the first wallet** in the
   catalog as a last-resort fallback.

`wallet_id` is **required** — it must always be one of the
sync_ids in the catalog. Never null, never fabricated.

### Step 3 — Map category (`category_id`) per line item

For each line item / VAT line / discount line, pick a category
**from the matched wallet's category list only**. The catalog is
fully user-defined — you only see `name` (and `type`). Reason
about the names; do not invent ids.

**Reasoning strategy (apply in order):**

1. **Filter by type.** Keep only categories whose `type` matches
   the line's `transaction_type`. Expense lines → expense
   categories; income / discount → income categories.

2. **Identify the item's function/domain in 1–3 words** before
   looking at the catalog. Examples of the *kind of thinking*
   (not literal mappings):
     - Item used on the body for hygiene → personal-care domain.
     - Item used to clean the house / surfaces → home-cleaning
       domain.
     - Raw food bought to cook later → grocery domain.
     - Prepared food/drink served at a shop → dining domain.
     - Bill, utility, subscription → bills domain.
   Then read each candidate category name as the question
   *"does this name naturally cover the item's domain?"* Use the
   item word, the merchant context, and the slip `summary`.

3. **Prefer the most specific name that fits.** If multiple
   categories overlap, the narrower name wins (e.g. a coffee
   purchase fits a "กาแฟ"-style category better than a generic
   "อาหาร" or "ช้อปปิ้ง"). Specific personal-care / household /
   grocery / transport / utility names all beat generic ones.

4. **Generic catch-alls only if no specific name fits.** A broad
   name like "ช้อปปิ้ง" / "อาหาร" / "บันเทิง" is OK when the item
   is plausibly inside its scope but no narrower name applies.

5. **"อื่นๆ" / "Other" is a LAST RESORT.** Only pick a category
   named "อื่นๆ" (or any obvious "Other"/`expense_other`-style
   bucket) when **every other** name in the filtered list has no
   semantic overlap with the item at all. If even one name has
   any plausible coverage — prefer it over "อื่นๆ".

6. **Never-null safety net.** If after all the above no category
   feels right, fall back to the first category of the matching
   type in the wallet.

`category_id` is **required** — never null, never fabricated.
VAT / ภาษี / ค่าธรรมเนียม follow the same rules: pick a
tax/fee-named category if one exists, else nearest by name.

### Step 4 — Build transactions (call the tool)

Emit `propose_transaction` tool calls in a single response. All
calls share the same `wallet_id`, `date`, `currency_code`,
`merchant_name`. The shape depends on `slip_type`:

#### `transfer_slip` → exactly **1** tool call
- `type` = `expense` if money leaves the user, `income` if it
  arrives.
- `amount` = the single transfer amount.
- `note` = a short purpose if visible (e.g. "ค่ากาแฟ"), else
  empty — the server appends the recipient.

#### `utility_bill` → exactly **1** tool call
- `type = "expense"`.
- `amount` = the amount due / paid on the bill (use the grand
  total after any applied discount; ignore promo ads).
- `category_id` prefers a bills / utilities-named category in
  the wallet (e.g. "ค่าไฟ", "ค่าบิล") per Step 3 rules.
- `note` = a short label of what the bill is for (e.g.
  "ค่าไฟ", "อินเทอร์เน็ต", "บัตรเครดิต") — no merchant.

#### `retail_receipt` / `restaurant_bill` → **N + optional**
Emit one call per **transaction-relevant line** in the upper
section of the slip (see Step 1b). Each line is optional except
the item lines themselves:

- **Each item line** (always — there must be ≥1) →
  `type="expense"`, `amount` = the item's price,
  `note` = item name **only** (e.g. "นม", "ค่ากาแฟ") — no
  merchant, `include_in_report=true`.
- **VAT line** (only if shown as a separate paid amount above
  the grand total; skip if "VAT INCLUDED" / bundled) →
  `type="expense"`, `amount` = the VAT amount,
  `category_id` = a tax/fee-named category if present, else
  nearest by name, `note` = "VAT 7%" (actual rate),
  `include_in_report=true`.
- **Applied discount line** (only if the reconciliation check
  in Step 1b proved it reduces the grand total; never emit one
  for promo / coupon ads) → `type="income"`,
  `amount` = discount magnitude (positive),
  `category_id` = nearest match in income categories (refund /
  cashback / discount names beat salary-style names),
  `note` = "ส่วนลด" (no merchant),
  **`include_in_report=false`** — discounts are tracked but
  must NOT inflate income totals on reports.
- **Service charge** (restaurants, if listed separately) →
  emit as a normal expense line with note = "ค่าบริการ"
  (or "Service Charge"), `include_in_report=true`.

#### `reject` → **0** tool calls
Reply with the exact Thai sentence "ไม่สามารถอ่านสลิปได้".

## Common conventions

- `currency_code` / `currency_symbol` — default THB / ฿ unless the
  slip clearly shows another currency.
- `merchant_name` — the store/vendor/employer string; the server
  appends it to `note` for display.
- Do NOT split a transfer slip into multiple transactions — splitting
  is only for receipts that list individual items.

## After the tool calls

Reply with **one short Thai sentence** like
"ดูข้อมูลในการ์ดด้านบนได้เลยครับ — กดบันทึกถ้าถูกต้อง". Do not
restate the transaction details and do not emit a `<suggestions>` tag.
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
    _slip_log(
        "SLIP",
        "slip_node entered",
        user_id=user_id,
        image_count=len(images),
        msg_count=len(state.get("messages") or []),
    )
    if not images:
        raise ValueError("slip_node invoked with no images")

    # Log per-image size + detected mime so we can diagnose later why a
    # given slip turn confused the vision model. Data URLs look like
    # `data:image/jpeg;base64,xxxx` — pull the mime prefix verbatim and
    # compute the decoded byte size from the base64 length.
    for idx, url in enumerate(images):
        mime = "unknown"
        b64_len = len(url)
        if url.startswith("data:") and ";base64," in url:
            header, _, _ = url.partition(",")
            mime = header[5:].split(";")[0]  # data:<mime>;base64
            b64_only_len = b64_len - header.__len__() - 1
            # base64 → bytes ratio is 4:3; subtract padding bytes too
            approx_bytes = (b64_only_len * 3) // 4
        else:
            approx_bytes = (b64_len * 3) // 4
        _slip_log(
            "SLIP",
            "image metadata",
            index=idx,
            mime=mime,
            approx_kb=approx_bytes // 1024,
            data_url_chars=b64_len,
        )

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

    # Build multimodal content blocks. We previously set `detail: "low"`
    # to save tokens, but on Gemini-2.5-flash-lite (via OpenRouter) that
    # downscale plus the mobile-side compression (1280px @ q70 JPEG) was
    # producing empty responses on real slips — the small-font amounts
    # weren't legible after triple downscale. Letting the model pick the
    # detail tier costs ~1000 extra tokens per slip but reliably parses
    # the amounts (which is the whole point of this feature).
    image_blocks: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": url}}
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
        _slip_log("SLIP", "vision LLM empty on first call — retrying")
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
    # Pull model metadata so empty-response investigations can see the
    # actual finish_reason / token counts / model name without re-running.
    resp_meta = getattr(response, "response_metadata", None) or {}
    usage_meta = getattr(response, "usage_metadata", None) or {}
    _slip_log(
        "SLIP",
        "vision LLM response",
        image_count=len(images),
        tool_calls=len(tool_calls),
        text_preview=_text_preview(response)[:200],
        model=resp_meta.get("model_name"),
        finish_reason=resp_meta.get("finish_reason"),
        input_tokens=usage_meta.get("input_tokens"),
        output_tokens=usage_meta.get("output_tokens"),
    )

    # No tool_calls means the vision model couldn't (or wouldn't) parse
    # the slip. Two flavors we've observed:
    #   1. Truly empty response (model silently failed).
    #   2. Text refusal — model returned a Thai sentence like
    #      "กรุณาส่งสลิปที่ชัดเจนกว่านี้" instead of calling the tool.
    # Both end the slip flow without a `structured_data` dispatch, so
    # mobile gets `reading_slip` → `done` and the user sees a dots
    # indicator with no card and no explanation. Surface either case as
    # an SSE `error` event (server.py:567-569 maps the raised exception
    # → `{"type": "error", "message": ...}` → mobile snackbar). When the
    # model gave a refusal text, prefer it as the message so the user
    # sees the model's actual feedback ("ภาพไม่ใช่สลิป" / "ภาพไม่ชัด").
    if not tool_calls:
        # Spec: surface a single, user-friendly Thai sentence whenever
        # the model could not produce a transaction — whether it
        # refused (text reply) or silently failed (empty). The model is
        # already told to use this exact phrase, so honour it here too
        # instead of leaking the raw refusal text (which may say things
        # like "the image appears blurry" in inconsistent wording).
        refusal_text = _text_preview(response).strip()
        error_msg = "ไม่สามารถอ่านสลิปได้"
        _slip_log(
            "SLIP",
            "no tool_calls — surfacing error to client",
            image_count=len(images),
            had_text=bool(refusal_text),
            refusal_preview=refusal_text[:200],
            error_msg=error_msg,
        )
        raise ValueError(error_msg)

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
    general_wallets = [
        w for w in catalog.wallets if w.wallet_type == "general"
    ]
    valid_wallet_ids = {w.sync_id for w in general_wallets}
    # Last-resort wallet for the "wallet_id must never be null" rule.
    # Mirrors the prompt's fallback chain: if the LLM picked an invalid
    # / null id we still need to surface a real wallet to mobile, so
    # use the first general wallet in catalog order.
    fallback_wallet_id: str | None = (
        general_wallets[0].sync_id if general_wallets else None
    )

    # Build the wallet → categories map BEFORE iterating tool_calls so
    # we can validate `category_id` against the matched wallet's
    # categories only. Using the deduped catalog.categories here would
    # false-negative: the same category name exists once per wallet,
    # and dedup keeps only one sync_id, dropping the others as invalid
    # even though they belong to a real wallet.
    by_wallet = await fetch_categories_by_wallet(user_id)

    tool_messages: list[ToolMessage] = []
    # Collect every validated transaction here, then dispatch ONE
    # bundled `propose_transaction_group` event at the end. Mobile
    # renders the entire turn as a single card with N rows + a wallet/
    # total footer + a single set of group-level actions — sending
    # multiple events would produce N stacked cards instead of a group.
    group_transactions: list[dict[str, Any]] = []
    group_wallet_id: str | None = None
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name != "propose_transaction":
            continue
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        args = dict(args or {})

        # ── id validation with never-null fallback ──────────────
        # Spec: wallet_id / category_id must ALWAYS hold a real value.
        # If the LLM passed an invalid id (hallucinated / stale dedup
        # leftover), fall back to the same chain the prompt instructs:
        # nearest valid match → first wallet / first category of the
        # matching type.
        wallet_id_in = args.get("wallet_id")
        category_id_in = args.get("category_id")
        txn_type = args.get("type", "expense")

        if wallet_id_in in valid_wallet_ids:
            wallet_id = wallet_id_in
        else:
            if wallet_id_in:
                _logger.warning(
                    "propose_validation_node: dropped invalid wallet_id=%s "
                    "→ falling back to %s",
                    wallet_id_in, fallback_wallet_id,
                )
            wallet_id = fallback_wallet_id

        category_id: str | None = None
        if wallet_id:
            wallet_cats = by_wallet.get(wallet_id, [])
            wallet_cat_ids = {c["sync_id"] for c in wallet_cats}
            if category_id_in in wallet_cat_ids:
                category_id = category_id_in
            elif category_id_in:
                # Stale-duplicate rescue: same category name under a
                # different wallet in the deduped catalog. Find a
                # same-name row under the matched wallet first.
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
            # Final never-null fallback: first category of the matching
            # type under this wallet. Mirrors the prompt rule.
            if not category_id:
                for c in wallet_cats:
                    if c.get("type") == txn_type:
                        category_id = c["sync_id"]
                        break
                # Last resort: any category under this wallet.
                if not category_id and wallet_cats:
                    category_id = wallet_cats[0]["sync_id"]
                if category_id_in:
                    _logger.warning(
                        "propose_validation_node: dropped category_id=%s "
                        "(not in wallet %s) → fell back to %s",
                        category_id_in, wallet_id, category_id,
                    )

        # ── note merge ──────────────────────────────────────────
        final_note = _combine_note(args.get("note"), args.get("merchant_name"))

        # The group shares one wallet — first valid wallet_id wins.
        # If later items pick a different wallet (LLM noise) we coerce
        # them onto the group's wallet so the group footer stays honest.
        if group_wallet_id is None:
            group_wallet_id = wallet_id
        elif wallet_id != group_wallet_id:
            _logger.warning(
                "propose_validation_node: coercing item wallet_id %s → "
                "group wallet %s (group shares one wallet)",
                wallet_id, group_wallet_id,
            )
            wallet_id = group_wallet_id
            # Re-resolve category against the group wallet's list so
            # the category still belongs to a real category of the
            # final wallet.
            wallet_cats = by_wallet.get(group_wallet_id, [])
            if category_id not in {c["sync_id"] for c in wallet_cats}:
                category_id = None
                for c in wallet_cats:
                    if c.get("type") == txn_type:
                        category_id = c["sync_id"]
                        break
                if not category_id and wallet_cats:
                    category_id = wallet_cats[0]["sync_id"]

        include_in_report = bool(args.get("include_in_report", True))
        item = {
            "sync_id": str(uuid.uuid4()),
            "type": txn_type,
            "amount": float(args.get("amount") or 0),
            "date": args.get("date"),
            "wallet_id": wallet_id,
            "category_id": category_id,
            "note": final_note,
            "currency_code": args.get("currency_code") or "THB",
            "currency_symbol": args.get("currency_symbol") or "฿",
            "includeInReport": include_in_report,
        }
        group_transactions.append(item)
        _slip_log(
            "PROPOSE",
            "validated tool_call",
            wallet_id=wallet_id,
            category_id=category_id,
            amount=item["amount"],
            type=item["type"],
            include_in_report=include_in_report,
            sync_id=item["sync_id"],
        )

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

    # ── bundle + dispatch ────────────────────────────────────────
    # Mobile contract: ALWAYS emit a `propose_transaction_group` even
    # for a single-item slip (Q2=B). The card UI renders a single
    # consistent group layout regardless of item count — backend
    # doesn't try to second-guess whether to send "single" vs "group".
    if group_transactions:
        # `total` = net amount the user actually paid out of pocket.
        # Per spec Q3: expense + (VAT-ish expense) − income (discounts).
        # Iterate the validated items rather than re-parsing args so the
        # total stays in lockstep with what mobile renders.
        total = 0.0
        for t in group_transactions:
            if t["type"] == "expense":
                total += t["amount"]
            else:  # income (discounts / rebates)
                total -= t["amount"]
        group_payload = {
            "type": "propose_transaction_group",
            "data": {
                "group_id": str(uuid.uuid4()),
                "wallet_id": group_wallet_id,
                "currency_code": (
                    group_transactions[0].get("currency_code") or "THB"
                ),
                "currency_symbol": (
                    group_transactions[0].get("currency_symbol") or "฿"
                ),
                "total": total,
                "transactions": group_transactions,
            },
        }
        _slip_log(
            "PROPOSE",
            "dispatching group structured_data",
            group_id=group_payload["data"]["group_id"],
            wallet_id=group_wallet_id,
            item_count=len(group_transactions),
            total=total,
        )
        await adispatch_custom_event("structured_data", group_payload)
        _slip_log("PROPOSE", "dispatched group OK")

    return {"messages": tool_messages}


async def slip_cleanup_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Terminal step of the slip flow — strip every message added during
    this turn so the checkpoint has no trace of the slip exchange.

    Slip turns produce three artifacts that are useless once the
    proposal SSE event has been emitted to the mobile client:
        1. `HumanMessage("[INTENT:parse_transaction_from_slip]")` — the
           routing marker from mobile; the image content blocks attached
           to it bloat the checkpoint with base64 payloads
        2. `AIMessage(tool_calls=[propose_transaction(...)])` — vision
           LLM output; not user-facing
        3. `ToolMessage("(proposal dispatched)")` — placeholder from
           `propose_validation_node`

    Removing them here means:
      • No orphan markers in chat history reloads (no need for mobile
        client to filter — the data simply isn't there)
      • ReAct turns that follow can't reference "the slip I just sent",
        but that's acceptable: the mobile UI keeps the proposal card
        in its own state and the user interacts with it directly
      • Checkpoint stays lean — base64 image blocks aren't persisted
    """
    msgs = state.get("messages") or []
    if not msgs:
        return {"images": []}
    # Find the slip turn's start: walk back to the most recent
    # HumanMessage. Everything from there onward is part of this turn
    # (text marker + vision AIMessage + optional ToolMessage).
    start = -1
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            start = i
            break
    if start < 0:
        return {"images": []}
    removals: list[RemoveMessage] = []
    for m in msgs[start:]:
        mid = getattr(m, "id", None)
        if mid:
            removals.append(RemoveMessage(id=mid))
    _logger.info(
        "slip_cleanup_node: removing %d messages from current turn",
        len(removals),
    )
    _slip_log(
        "CLEANUP",
        "removing slip turn",
        removal_count=len(removals),
        total_msgs=len(msgs),
    )
    return {"messages": removals, "images": []}
