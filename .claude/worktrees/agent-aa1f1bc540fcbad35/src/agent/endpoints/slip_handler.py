"""Slip endpoint pre-processor — Gemini vision -> transaction_proposal_group.

NEW (TRANSFORM) in Wave 5. Source: v2 `vision.py` (slip-parsing prompt + the
payload-builder). Decision alpha (locked Phase 2): the slip flow SKIPS the
ReAct loop entirely — one vision call + a single `transaction_proposal_group`
block. Per Wave 5 kickoff B1, only the FIRST image is processed when the
mobile client sends `image_b64s = [a, b]` (matches v2 behavior; spec defers
multi-image to a later release).

Mobile contract (frozen, identical to v2 slip flow):
  1. `event: status_token` "กำลังอ่านสลิป..." (one word at a time).
  2. EITHER:
     - `event: block` {"type":"answer","text":"<unreadable sentence>"} +
       `event: done` (vision returned readable=false), OR
     - `event: block` {"type":"transaction_proposal_group", ...} +
       `event: done` (vision parsed >= 1 line).

Wave 6 expectation (documented but NOT executed here): when Wave 6's
`server.py` persists the group proposal via `graph.aupdate_state(...)`, it
MUST pass `as_node="finalize"`. Per memory `project_slip_vision_as_node`,
seeding state for a turn that short-circuited BEFORE the graph ran (slip
flow doesn't enter the graph at all) leaves LangGraph unable to infer the
write's owning node → InvalidUpdateError. The terminal `finalize` node (=
edge to END) is the safe target.

The group block carries `proposal_id == group_id` per the same memory: the
mobile history-replay path reads `blk["proposal_id"]` to annotate status
when the user reopens the thread.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
)
from src.agent.session_logger import slog, slog_block, slog_error
from src.agent.utils.safe_json import safe_json


# ---------------------------------------------------------------------------
# 1. Constants — preserved byte-for-byte from v2 so the mobile UX is identical
# ---------------------------------------------------------------------------

# The exact Thai sentence the mobile UI expects when an image carries no
# readable money figure. Lifted from v2 `vision.py::SLIP_UNREADABLE_MESSAGE`
# — keep byte-for-byte identical.
SLIP_UNREADABLE_MESSAGE = "ไม่สามารถอ่านสลิปได้"

_STATUS_READING_SLIP = "กำลังอ่านสลิป..."

# System "catch-all" category names. When a line's category can't be resolved
# we fall back to the Other category of the line's DIRECTION — never to None
# (the mobile card silently turns null into the first income category
# "เงินเดือน", mislabelling e.g. a discount line). Memory:
# `project_chat_null_category_other_floor`.
_OTHER_CATEGORY_NAMES = {"อื่นๆ", "Other"}


# ---------------------------------------------------------------------------
# 2. Slip result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SlipData:
    """Outcome of one slip-vision call.

    `readable=False` → the image carried no readable money. `message` holds
    the Thai sentence the client renders; `payloads` is empty.
    `readable=True` → `payloads` is a non-empty list of confirm-ready
    proposal rows (each row = one transaction line).
    """

    readable: bool
    message: Optional[str] = None
    payloads: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Helpers — lifted from v2 vision.py (kept here so v3 has no v2 import)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Current UTC datetime as ISO 8601 — matches v2 transactions._now_iso."""
    return datetime.now(timezone.utc).isoformat()


def _new_sync_id() -> str:
    """Fresh sync_id for one proposal row — 32-char hex (UUID4 no dashes)."""
    return uuid.uuid4().hex


def _new_group_id() -> str:
    """Fresh proposal/group id — separate from per-row sync_ids."""
    return uuid.uuid4().hex


def _jsonable_amount(canon: Decimal) -> int | float:
    """Decimal → JSON-safe number (int when integral, float otherwise).

    Mirrors `tools/propose_transaction.py::_jsonable_amount` so the slip
    payload looks identical to a typed-ADD proposal on the wire.
    """
    if canon == canon.to_integral_value():
        return int(canon)
    return float(canon)


def _to_data_uri(b64: str) -> str:
    """Wrap raw base64 JPEG as a data URI for the `image_url` content block."""
    s = b64.strip()
    if s.startswith("data:"):
        return s
    return f"data:image/jpeg;base64,{s}"


def _valid_date(raw: Any, fallback_iso: str) -> str:
    """Accept a real ISO date string; fall back to today on malformed input."""
    if not isinstance(raw, str) or not raw.strip():
        return fallback_iso
    try:
        datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return raw
    except ValueError:
        return fallback_iso


def _other_category(
    cat_by_id: dict[str, CategoryEntry], ttype: str
) -> Optional[CategoryEntry]:
    """Find the wallet's "อื่นๆ"/Other category for direction `ttype`.

    Memory `project_chat_null_category_other_floor`: a null category leak
    on a slip discount line gets silently rendered as "เงินเดือน". Floor
    every line at Other-of-direction before degrading to None.
    """
    for c in cat_by_id.values():
        if c.type == ttype and c.name in _OTHER_CATEGORY_NAMES:
            return c
    return None


def _pick_slip_wallet(
    catalog: EntityCatalog, preferred_sync_id: Optional[str]
) -> Optional[WalletEntry]:
    """Slip wallet cascade — honor mobile pick, else index-0 (general).

    Same shape as v2 `EntityCatalog.slip_wallet`. We can't reuse that method
    directly because Wave 5 test fixtures inject a hand-built catalog that
    sometimes mocks just `wallets`. Wallets are pre-ordered general ->
    creditcard -> goal (memory `project_wallet_index0_ordering`), so index-0
    is the spending-wallet default.
    """
    if preferred_sync_id:
        wallet = catalog.wallet_by_sync_id(preferred_sync_id)
        if wallet is not None:
            return wallet
    for w in catalog.wallets:
        if w.wallet_type == "general":
            return w
    return catalog.wallets[0] if catalog.wallets else None


# ---------------------------------------------------------------------------
# 4. Prompt builders — slimmed from v2 vision.py (same prompt, repackaged)
# ---------------------------------------------------------------------------


def _build_slip_prompt(
    *,
    current_date: str,
    wallet_name: str,
    wallet_currency: str,
    category_map: str,
    default_currency_code: str,
    default_currency_symbol: str,
) -> str:
    """Single-shot slip-parsing prompt scoped to ONE wallet.

    Ported byte-for-byte from v2 `vision.py::build_slip_prompt` so the
    vision model's outputs match v2's expectations and no eval regression
    creeps in. We only inline it here to keep v3 free of v2 imports.
    """
    return f"""You are a slip-parsing agent inside Mint Money.

**Today's date:** {current_date}
**Target wallet:** `{wallet_name}` (currency {wallet_currency}) — ALL
transactions you emit belong to this wallet. Do NOT choose a wallet.
**User's default currency (fallback only):** {default_currency_code} / {default_currency_symbol}

Your job: look at the attached image, decide whether it is a slip /
receipt with a readable money amount, and turn it into one or more
transactions. The single gate is the `amount`: if you can read at least
one money figure, process it. If no money figure can be read (fully
blurry, unreadable, or not money-related), the image is unreadable.

## Step 0 — Classify the image into one slip_type

| slip_type        | What it looks like                              |
|------------------|-------------------------------------------------|
| `retail_receipt` | Store header + **multiple** itemised priced lines + sub-total/grand total. |
| `restaurant_bill`| Food/drink items, table/order id, may include service charge / VAT. |
| `transfer_slip`  | Bank / e-wallet / QR confirmation: **single amount**, sender → recipient, ref number. |
| `service_bill`   | "ใบแจ้งหนี้" / "Invoice" / "Statement" for a **single-amount** service payment. |
| `payslip`        | Pay period header + earnings/deductions columns + **net pay** at the bottom. |
| `reject`         | **No extractable amount** — fully blurry, unreadable, or not money-related. |

**Has-amount fallback:** any image with a readable money figure that
fits no specific type → treat as `transfer_slip` with that single amount.
Only `reject` when no amount can be read at all.

**If `slip_type == reject`:** Return `{{"readable": false, "transactions": []}}` and stop.

## Step 1 — Extract raw context
- `amount` per slip / line item. Buddhist Era 25xx dates → subtract 543.
- `transaction_type` — `expense` (money leaves) or `income` (money enters); ambiguous → expense.
- `merchant_or_recipient` — store / recipient / issuer / employer (strip payment-rail wrappers).

## Reconciliation (MANDATORY)
- VAT bundled vs added-on-top: bundled → no separate VAT line; added → emit VAT line.
- Σ(expense) − Σ(income/discount) MUST equal grand_total; on mismatch fall back to a single grand-total line.

## Category mapping (per line) — pick from THIS wallet only
{category_map}

`category_sync_id` MUST be copied verbatim from the list. Reasoning order:
1. Filter by type (expense/income).
2. Pick the category whose name covers the item's domain.
3. Discount lines → income "เงินคืน/Refund" if available else "อื่นๆ"/Other (income).
4. NEVER return null — every line carries a `category_sync_id`. "อื่นๆ"/Other is the last resort.

## Step 4 — Emit transactions
- `transfer_slip` / `service_bill` → exactly 1 transaction.
- `retail_receipt` / `restaurant_bill` → one per item line; VAT only when added-on-top; discount as income with `include_in_report=false`.
- `payslip` → one income line per earnings row, one expense line per deduction; reconcile to net pay.

## Output — STRICT JSON only (no prose, no markdown fences)
{{"readable": true, "transactions": [
  {{"type": "expense", "amount": 120.0, "category_sync_id": "<id>", "note": "ค่ากาแฟ", "include_in_report": true}}
]}}

Unreadable image → {{"readable": false, "transactions": []}}.
"""


def _build_slip_messages(prompt: str, image_b64: str) -> list[dict]:
    """Two-message multimodal request: system prompt + a user turn with the
    image. The text part anchors the image content block."""
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Parse the attached slip/receipt image."},
                {"type": "image_url", "image_url": {"url": _to_data_uri(image_b64)}},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 5. Payload builder — mirrors propose_transaction's wire shape
# ---------------------------------------------------------------------------


def _build_payload(
    *,
    txn: dict,
    user_id: str,
    wallet: WalletEntry,
    cat_by_id: dict[str, CategoryEntry],
    today_iso: str,
) -> Optional[dict]:
    """Build one confirm-ready proposal payload from a vision transaction dict.

    Mirrors `tools/propose_transaction.py`'s payload exactly so the mobile
    proposal card + `/transactions/confirm` treat it like a typed ADD.
    Returns None when the line is unusable (no positive amount) — caller
    skips.
    """
    raw_amount = txn.get("amount")
    if raw_amount is None:
        return None
    try:
        amount = _jsonable_amount(Decimal(str(raw_amount)))
    except Exception:  # noqa: BLE001 — malformed amount, skip line
        return None
    if not isinstance(amount, (int, float)) or amount <= 0:
        return None

    ttype = txn.get("type") if txn.get("type") in ("expense", "income") else "expense"

    # Category resolve: explicit pick → Other-floor (Decimal `project_chat_null_category_other_floor`)
    # → finally None (last resort, only when wallet truly has no Other-of-direction).
    cat_id = txn.get("category_sync_id")
    chosen = cat_by_id.get(cat_id) if cat_id else None
    if chosen is None:
        chosen = _other_category(cat_by_id, ttype)
    category_sync_id = chosen.sync_id if chosen else None
    category_label = chosen.name if chosen else "other"

    note = (txn.get("note") or "").strip() or "สลิป"
    include_in_report = txn.get("include_in_report")
    if not isinstance(include_in_report, bool):
        include_in_report = True

    return {
        # Canonical mobile-side write contract:
        "sync_id": _new_sync_id(),
        "type": ttype,
        "amount": amount,
        "category_sync_id": category_sync_id,
        "wallet_sync_id": wallet.sync_id,
        "note": note,
        "date": _valid_date(txn.get("date"), today_iso),
        "currency_code": None,  # mobile fills from the wallet's default
        # Legacy keys preserved for in-flight clients / confirm handler:
        "user_id": user_id,
        "wallet_id": wallet.sync_id,
        "category": category_label,
        "description": note,
        "include_in_report": include_in_report,
    }


# ---------------------------------------------------------------------------
# 6. extract_slip — one Gemini vision call -> SlipData
# ---------------------------------------------------------------------------


async def extract_slip(
    vision_call: Callable[[list[dict]], Awaitable[str]],
    *,
    user_id: str,
    image_b64: str,
    catalog: EntityCatalog,
    wallet: WalletEntry,
    default_currency_code: str = "THB",
    default_currency_symbol: str = "฿",
    today_iso: Optional[str] = None,
) -> SlipData:
    """Run one slip image through the vision model -> confirm-ready payloads.

    `vision_call` is an injected `make_multimodal_call("vision")` callable
    (model + fallback chain resolved from env). The wallet is already
    chosen by the caller (`_pick_slip_wallet`); we only resolve categories
    scoped to that wallet.

    Failures (malformed JSON, model error) degrade gracefully to
    `readable=False` with the unreadable sentence — the mobile voice UI
    shows a retry, which is friendlier than a 500.
    """
    today_iso = today_iso or _now_iso()

    cat_by_id = {
        c.sync_id: c for c in catalog.categories_for(wallet.sync_id)
    }

    prompt = _build_slip_prompt(
        current_date=date.today().isoformat(),
        wallet_name=wallet.name,
        wallet_currency=wallet.currency,
        category_map=catalog.render_for_slip(wallet.sync_id),
        default_currency_code=default_currency_code or "THB",
        default_currency_symbol=default_currency_symbol or "฿",
    )
    messages = _build_slip_messages(prompt, image_b64)

    slog(
        "slip_handler",
        f"slip parse start — wallet={wallet.sync_id} cats={len(cat_by_id)}",
    )

    try:
        content = await vision_call(messages)
    except Exception as e:  # noqa: BLE001 — vision transport failure
        slog_error("slip_handler", e)
        return SlipData(readable=False, message=SLIP_UNREADABLE_MESSAGE)

    try:
        parsed = safe_json(content)
    except ValueError as e:
        slog_error("slip_handler", e)
        return SlipData(readable=False, message=SLIP_UNREADABLE_MESSAGE)

    if not parsed.get("readable", False):
        slog("slip_handler", "slip unreadable (model)")
        return SlipData(readable=False, message=SLIP_UNREADABLE_MESSAGE)

    raw_txns = parsed.get("transactions") or []
    payloads: list[dict] = []
    for txn in raw_txns:
        if not isinstance(txn, dict):
            continue
        payload = _build_payload(
            txn=txn,
            user_id=user_id,
            wallet=wallet,
            cat_by_id=cat_by_id,
            today_iso=today_iso,
        )
        if payload is not None:
            payloads.append(payload)

    if not payloads:
        # readable=true but nothing usable → treat as unreadable for the user.
        slog("slip_handler", "slip readable but no usable transactions")
        return SlipData(readable=False, message=SLIP_UNREADABLE_MESSAGE)

    slog_block(
        "slip_handler",
        f"slip parsed — {len(payloads)} proposal(s)",
        "\n".join(
            f"{p['type']} {p['amount']} {p['category']} — {p['note']}"
            for p in payloads
        ),
    )
    return SlipData(readable=True, payloads=payloads)


# ---------------------------------------------------------------------------
# 7. SSE generator — orchestrates the slip flow
# ---------------------------------------------------------------------------


async def _stream_status_tokens(text: str) -> AsyncIterator[dict]:
    """Yield one `status_token` event per space-separated word (v2 parity)."""
    for word in text.split():
        yield {"event": "status_token", "data": word}


def _emit_answer_block(text: str) -> dict:
    """One answer block (used for the unreadable / no-wallet fallback)."""
    return {
        "event": "block",
        "data": json.dumps({"type": "answer", "text": text}, ensure_ascii=False),
    }


def _emit_done(thread_id: str) -> dict:
    """Terminal done event."""
    return {"event": "done", "data": json.dumps({"thread_id": thread_id})}


async def handle_slip_chat(
    *,
    vision_call: Callable[[list[dict]], Awaitable[str]],
    catalog: EntityCatalog,
    image_b64s: list[str],
    thread_id: str,
    user_id: str,
    wallet_id: Optional[str] = None,
    default_currency_code: str = "THB",
    default_currency_symbol: str = "฿",
) -> AsyncIterator[dict]:
    """Drive one slip turn: vision -> transaction_proposal_group block.

    B1 (locked Wave 5 kickoff): only the FIRST image in `image_b64s` is
    processed. Multi-image slips are out of v3 scope — matches v2 behavior.

    NOTE for Wave 6 (server.py persistence):
        When persisting the group proposal via `graph.aupdate_state(...)`
        the caller MUST pass `as_node="finalize"`. Per memory
        `project_slip_vision_as_node`, the slip turn short-circuits BEFORE
        the graph runs so LangGraph has no pending task to attribute the
        write to → InvalidUpdateError otherwise. Attributing it to the
        terminal `finalize` node (edge to END) writes the proposals and
        schedules no pending tasks — a clean completed state, indistinguishable
        from a normal turn that ended at finalize.

    Wave 6 also reads `proposal_id == group_id` from the emitted block to
    annotate status during history replay — keep both keys in lockstep
    (this generator writes them together below).
    """
    # 1. Status — animated label while vision runs.
    async for ev in _stream_status_tokens(_STATUS_READING_SLIP):
        yield ev

    # 2. Wallet pick — honor mobile selection, fall back to index-0.
    wallet = _pick_slip_wallet(catalog, wallet_id)
    if wallet is None:
        # No wallet at all → polite onboarding nudge + done. The user
        # creates a wallet, then the next slip works.
        slog("slip_handler", "no wallet available — emitting onboarding fallback")
        yield _emit_answer_block(
            "ยังไม่มีบัญชีสำหรับบันทึกสลิป เพิ่มบัญชีก่อนนะครับ"
        )
        yield _emit_done(thread_id)
        return

    # 3. Vision — B1: process ONLY the first image. Multi-image deferred.
    if not image_b64s:
        slog("slip_handler", "no images attached — unreadable fallback")
        yield _emit_answer_block(SLIP_UNREADABLE_MESSAGE)
        yield _emit_done(thread_id)
        return
    first_image_b64 = image_b64s[0]

    result = await extract_slip(
        vision_call,
        user_id=user_id,
        image_b64=first_image_b64,
        catalog=catalog,
        wallet=wallet,
        default_currency_code=default_currency_code,
        default_currency_symbol=default_currency_symbol,
    )

    if not result.readable:
        # 3a. Vision rejected the slip → render the Thai unreadable
        # sentence as an answer block + done. Mobile shows it inline.
        yield _emit_answer_block(result.message or SLIP_UNREADABLE_MESSAGE)
        yield _emit_done(thread_id)
        return

    # 4. One group block. group_id doubles as proposal_id — mobile writes the
    # rows to its own DB and fires ONE confirm on group_id, so the server
    # persists exactly one proposal keyed by group_id whose payload carries
    # the rows. Both keys MUST be set on the wire (memory:
    # project_slip_vision_as_node).
    rows = result.payloads
    group_id = _new_group_id()

    # Net total — Σ(expense) − Σ(income). Summed in Decimal so the printed
    # net (e.g. payslip รายได้สุทธิ) never drifts vs the lines.
    total_dec: Decimal = Decimal(0)
    for r in rows:
        amount_dec = Decimal(str(r["amount"]))
        sign = Decimal(-1) if r["type"] == "income" else Decimal(1)
        total_dec += amount_dec * sign
    total = _jsonable_amount(total_dec)

    block = {
        "type": "transaction_proposal_group",
        # Both keys carry the SAME id on purpose. `proposal_id` is the
        # canonical confirm/status key shared with single proposals;
        # `group_id` is the group-card contract mobile reads to batch-save.
        "proposal_id": group_id,
        "group_id": group_id,
        "total": total,
        "wallet_sync_id": wallet.sync_id,
        "currency_code": None,
        "low_confidence": False,
        "transactions": rows,
    }

    yield {
        "event": "block",
        "data": json.dumps(block, ensure_ascii=False),
    }
    yield _emit_done(thread_id)


__all__ = [
    "SLIP_UNREADABLE_MESSAGE",
    "SlipData",
    "extract_slip",
    "handle_slip_chat",
]
