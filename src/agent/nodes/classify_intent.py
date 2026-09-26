"""`classify_intent` + `direct_propose` — the stateful classify-router shortcut.

NEW in Wave 8.

Goal: skip the full ReAct loop for an UNAMBIGUOUS, COMPLETE ADD ("กาแฟ 50")
by classifying intent up-front and routing straight to a deterministic
`direct_propose` node that reuses `propose_transaction._propose_core`. Every other
input (Analyst, Advisor, incomplete ADD, ambiguous, classifier failure, or
router disabled) falls through to `react` unchanged.

Why a stateful classifier:
  A STATELESS classifier sees only the latest user text in isolation, which
  mis-routes the E9 case: a bare "รถ 2 ล้าน" that ANSWERS a prior advisor
  question ("อยากซื้อรถราคาเท่าไหร่ดี") looks like a fresh ADD. This classifier
  sees WINDOWED conversation history, so it keeps that turn on the advisor flow.

LLM call: routes through `make_llm_call("classify")` — the classify role is a
first-class LLM role (CLASSIFY_MODEL + CLASSIFY_FALLBACK_MODELS), validated at
startup by `validate_llm_env`, with usage/token logging via the shared call
path. Returns raw text; we parse the JSON object ourselves (tolerating fences),
same pattern as `propose_transaction._resolve_pair_async`.

Kill-switch + env contract:
  CLASSIFY_ROUTER_ENABLED   1 | 0 (default 0 → router OFF → every turn → react,
                            and `classify_intent` SKIPS the LLM call entirely).
  CLASSIFY_MODEL            slug (env always set; validated at startup).
  CLASSIFY_FALLBACK_MODELS  JSON array | comma list (≥2 slugs, validated).
  CLASSIFY_TIMEOUT_S        float (default 4.0) → passed to make_llm_call.

Completeness Gate (นิยาม B): route to `direct_propose` ONLY when
  intent == "ADD" AND complete AND amount > 0 AND (category_label OR note).
Conservative: any doubt → react.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable as _ls_traceable

from src.agent.llm_openrouter import OpenRouterError, make_llm_call
from src.agent.session_logger import slog, slog_block, slog_error
from src.agent.tools.propose_transaction import (
    _extract_json_object,
    _propose_core,
    _propose_group_core,
    is_saved_card_correction,
    latest_card,
)


def latest_card_note(state: dict) -> str:
    """One line describing the newest card and whether the user already saved
    it. Confirm/cancel happen out-of-band (REST), so neither the classifier nor
    ReAct can see them in the conversation text — without this line "แก้เป็น 85"
    after a CONFIRMED 75 became a second card and the coffee was counted twice."""
    card = latest_card(state)
    if not card:
        return ""
    p = card.get("payload") or {}
    if p.get("transactions"):  # a group card (multi-item text or slip)
        p = {"amount": p.get("total"), "category": f"({len(p['transactions'])} รายการ)"}
    status = card.get("status") or "pending"
    label = {"confirmed": "CONFIRMED (already saved to the ledger)",
             "cancelled": "CANCELLED (not saved)",
             "discarded": "DISCARDED (not saved)"}.get(status, "PENDING (not saved yet)")
    return f"{p.get('amount')} บาท {p.get('category') or ''} — {label}".strip()


# Route the conditional edge reads. Distinct from any LLM intent label.
_ROUTE_DIRECT_PROPOSE = "direct_propose"
_ROUTE_REACT = "react"


# ─────────────────────────────────────────────────────────────────────────────
# Env helpers. The classify role's model/fallback envs are owned by
# `llm_openrouter` (_MODEL_ENV / _FALLBACK_ENV) and validated at startup by
# `validate_llm_env`; here we only read the runtime toggle + timeout.
# ─────────────────────────────────────────────────────────────────────────────


def is_classify_router_enabled() -> bool:
    """True when `CLASSIFY_ROUTER_ENABLED=1`. Default OFF — every turn → react."""
    return os.getenv("CLASSIFY_ROUTER_ENABLED", "0") in ("1", "true", "True")


def _timeout_s() -> float:
    try:
        return float(os.getenv("CLASSIFY_TIMEOUT_S", "4.0"))
    except (TypeError, ValueError):
        return 4.0


def _history_window_turns() -> int:
    """Reuse REACT_HISTORY_WINDOW_TURNS so the classifier and ReAct agree on
    how much history is in scope. Default 16 (matches graph._history_window)."""
    raw = os.getenv("REACT_HISTORY_WINDOW_TURNS")
    if not raw:
        return 16
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 16
    return n if n > 0 else 16


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — stateful: shows the recent conversation so the classifier can tell a
# fresh ADD from a bare amount that ANSWERS a prior advisor/analyst question.
# ─────────────────────────────────────────────────────────────────────────────


_SYSTEM = (
    "You classify the user's LATEST message in a Thai personal-finance chat. "
    "You are given the recent conversation for context. Output JSON only — no "
    "markdown, no prose."
)

_INSTRUCTIONS = """\
Decide the intent of the LATEST user message, using the conversation context.

intents:
  ADD     - user wants to RECORD a transaction they made (amount + what for)
  OTHER   - anything else: asking about their data, asking for guidance,
            greeting, app help, OR a bare value that ANSWERS the assistant's
            previous question (NOT a record request)

Own-wallet transfer rule:
  Moving money between the user's OWN wallets / credit cards / savings goals
  ("โอนเงินเข้า TrueMoney 500", "จ่ายบัตร KTC 5000", "เก็บเข้าเป้า 3000",
  "ถอนเงินสด 2000") is NOT ADD → intent=OTHER. Paying another person or a shop
  ("โอนให้แม่ 3000", "โอนค่าเช่าให้เจ้าของห้อง") IS an expense ADD.

Saved-card rule:
  If the latest transaction card is CONFIRMED and the LATEST message corrects
  it ("แก้เป็น 85", "เปลี่ยนหมวดเป็นอาหาร", "ไม่ใช่ 75"), that is NOT a new
  ADD → intent=OTHER (the saved row has to be edited in the app).

CRITICAL context rule:
  If the assistant's previous turn ASKED the user something (e.g. "อยากซื้อรถ
  ราคาเท่าไหร่", "ตั้งเป้าออมเดือนละเท่าไหร่"), then a bare amount or short
  reply from the user is ANSWERING that question → intent=OTHER, NOT ADD.
  Only classify ADD when the LATEST message itself expresses intent to record
  a transaction.

For ADD only, extract:
  amount          number (THB), required for complete=true
  type            "expense" (default) | "income" (ได้/รับ/เงินเดือน/โบนัส/รายได้)
                  | "auto" only when genuinely ambiguous (e.g. "โอนเงิน 500")
  category_label  short Thai noun phrase spent on / earned ("กาแฟ","เงินเดือน")
  wallet_label    "เงินสด" / "KBank" / null
  note            free-text description / null
  date_iso        resolve "เมื่อวาน"/"อาทิตย์ก่อน" to ISO; today = {today}

complete = TRUE only when amount AND (category_label OR note) are present.
multi    = TRUE when the LATEST message lists 2+ SEPARATE transactions, each
           with its own amount ("กาแฟ 60 ข้าว 80", "ค่าน้ำ 100, ค่าไฟ 900").
           A quantity inside one item is NOT multi ("ข้าว 2 จาน 120").
items    = when multi, EVERY transaction in order, each
           {{"amount","type","category_label","note","date_iso"}}; the top-level
           amount/category_label describe the first item. [] when not multi.

Return EXACTLY this JSON shape:
{{"intent":"ADD"|"OTHER","complete":true|false,"multi":true|false,\
"amount":number|null,"type":"expense"|"income"|"auto"|null,\
"category_label":string|null,"wallet_label":string|null,"note":string|null,\
"date_iso":string|null,"items":[...]}}
"""


def _last_user_text(messages: list) -> str:
    """The latest HumanMessage content (str), or ''."""
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c if isinstance(p, dict)]
                return " ".join(p for p in parts if p).strip()
            return ""
    return ""


def _render_history(messages: list) -> str:
    """Render the windowed history as compact role-tagged lines for the prompt.

    We only need ROLE + TEXT for intent disambiguation — tool-call payloads and
    block JSON would add noise + tokens. Skips empty-content narration steps.
    """
    lines: list[str] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            continue  # skip ToolMessage / System — not useful for intent
        content = m.content if isinstance(m.content, str) else ""
        content = content.strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier LLM call — routes through `make_llm_call("classify")` so the role
# shares provider routing, fallback chain, timeout, and usage/token logging with
# every other LLM role. make_llm_call returns raw text (no response_format
# support), so we parse the JSON object ourselves, tolerating fences — same
# pattern as propose_transaction._resolve_pair_async.
# ─────────────────────────────────────────────────────────────────────────────


@_ls_traceable(name="llm.classify_router", run_type="llm", tags=["classify_router"])
async def _run_classify_llm(
    *, history_text: str, user_text: str, today: date, card_note: str = ""
) -> Optional[dict]:
    """One short classification call. Returns parsed dict or None on ANY failure.

    None → the caller fail-safe routes to react. Never raises.
    """
    yesterday = (today - timedelta(days=1)).isoformat()
    user_msg = (
        _INSTRUCTIONS.format(today=today.isoformat(), yesterday=yesterday)
        + "\n\nConversation (most recent last):\n"
        + history_text
        + (f"\n\nLatest transaction card: {card_note}" if card_note else "")
        + f"\n\nLATEST user message to classify:\n{user_text}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    slog_block(
        "classify_router",
        f"PROMPT timeout={_timeout_s()}s",
        f"latest={user_text!r}",
    )
    try:
        call = make_llm_call("classify", timeout_s=_timeout_s())
    except OpenRouterError as exc:
        # CLASSIFY_MODEL unconfigured (should never happen — validate_llm_env
        # guards startup) → fail-safe to react.
        slog("classify_router", f"CLASSIFY_MODEL not configured: {exc}")
        return None
    try:
        raw = await call(messages)
    except OpenRouterError as exc:
        slog_error("classify_router", exc)
        return None
    try:
        parsed = _extract_json_object(raw)
    except (TypeError, ValueError) as exc:
        slog_error("classify_router", exc)
        return None
    if not isinstance(parsed, dict):
        slog("classify_router", f"non-dict result: {type(parsed).__name__}")
        return None
    return parsed


def _coerce_slots(parsed: dict) -> dict:
    """Coerce + validate the classifier JSON into clean ADD slots.

    Re-derives completeness from the actual presence of amount + (label|note)
    so the LLM can't get us to direct_propose a half-extracted ADD (Gate B).
    """
    intent = parsed.get("intent")
    amount = parsed.get("amount")
    if amount is not None:
        try:
            amount = float(amount)
            if amount <= 0:
                amount = None
        except (TypeError, ValueError):
            amount = None

    tx_type = parsed.get("type")
    if tx_type not in ("expense", "income", "auto"):
        tx_type = "expense"

    label = parsed.get("category_label")
    if label is not None:
        label = str(label).strip() or None
    note = parsed.get("note")
    if note is not None:
        note = str(note).strip() or None
    wallet = parsed.get("wallet_label")
    if wallet is not None:
        wallet = str(wallet).strip() or None
    date_iso = parsed.get("date_iso")
    if date_iso is not None:
        try:
            date.fromisoformat(str(date_iso))
        except (TypeError, ValueError):
            date_iso = None

    items: list[dict] = []
    for raw in parsed.get("items") or []:
        if not isinstance(raw, dict):
            continue
        try:
            amt = float(raw.get("amount"))
        except (TypeError, ValueError):
            continue
        lbl = str(raw.get("category_label") or "").strip() or None
        nte = str(raw.get("note") or "").strip() or None
        if amt <= 0 or not (lbl or nte):
            continue
        it_date = raw.get("date_iso") or date_iso
        try:
            date.fromisoformat(str(it_date)) if it_date else None
        except (TypeError, ValueError):
            it_date = date_iso
        items.append({
            "amount": amt,
            "type": raw.get("type") if raw.get("type") in ("expense", "income", "auto") else tx_type,
            "category_label": lbl, "note": nte, "date_iso": it_date,
        })

    derived_complete = amount is not None and (label is not None or note is not None)
    complete = bool(parsed.get("complete")) and derived_complete

    return {
        "intent": intent,
        "complete": complete,
        "multi": parsed.get("multi") is True,
        "items": items,
        "amount": amount,
        "type": tx_type,
        "category_label": label,
        "wallet_label": wallet,
        "note": note,
        "date_iso": date_iso,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


async def classify_intent_node(state: dict) -> dict:
    """Decide whether this turn can skip ReAct via the direct_propose shortcut.

    Router OFF (default) → set route=react and return IMMEDIATELY (no LLM call,
    zero added latency/cost). Router ON → classify the latest message against
    windowed history, apply Completeness Gate B, and route. Any failure
    (LLM transport, parse, ambiguity) fail-safe routes to react.
    """
    # Kill-switch: OFF → no LLM call, every turn → react.
    if not is_classify_router_enabled():
        return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}

    messages = state.get("messages") or []
    user_text = _last_user_text(messages)
    if not user_text:
        slog("classify_router", "no user text → react")
        return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}

    # Window history the same way ReAct does so the two share context scope.
    windowed = _window_messages(messages, _history_window_turns())
    history_text = _render_history(windowed)

    # Deterministic guard: a correction right after a SAVED card can never be a
    # fast-path ADD (it would record the item twice) — react redirects it.
    if is_saved_card_correction(state):
        slog("classify_router", "correction of a confirmed card → react")
        return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}

    parsed = await _run_classify_llm(
        history_text=history_text, user_text=user_text, today=date.today(),
        card_note=latest_card_note(state),
    )
    if parsed is None:
        # Fail-safe: classifier unavailable / failed → react handles the turn.
        return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}

    slots = _coerce_slots(parsed)
    intent = slots["intent"]
    complete = slots["complete"]
    amount = slots["amount"]
    has_label = bool(slots["category_label"] or slots["note"])

    # Completeness Gate B. A multi-item message shortcuts only when EVERY item
    # parsed complete (→ one group card); otherwise it goes to react (R7: one
    # card now, the rest after) — never silently drop items 2..n.
    multi_ok = slots["multi"] and len(slots["items"]) >= 2
    route_direct = (
        intent == "ADD"
        and complete
        and (multi_ok or not slots["multi"])
        and amount is not None
        and amount > 0
        and has_label
    )
    route = _ROUTE_DIRECT_PROPOSE if route_direct else _ROUTE_REACT
    slog(
        "classify_router",
        f"intent={intent} complete={complete} multi={slots['multi']} amount={amount} "
        f"label={slots['category_label']!r} note={slots['note']!r} → route={route}",
    )
    if route == _ROUTE_DIRECT_PROPOSE:
        return {"__classify_route__": route, "__classified_add__": slots}
    return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}


def route_after_classify(state: dict) -> str:
    """Conditional-edge router: read the decision classify_intent_node wrote."""
    return (
        _ROUTE_DIRECT_PROPOSE
        if state.get("__classify_route__") == _ROUTE_DIRECT_PROPOSE
        else _ROUTE_REACT
    )


def route_after_direct_propose(state: dict) -> str:
    """After direct_propose: success → post_turn; a failed propose (onboarding,
    off-list wallet, resolve error) → react, which can clarify or retry."""
    return _ROUTE_REACT if state.get("__classify_route__") == _ROUTE_REACT else "post_turn"


# Deterministic confirmation message — 0 LLM calls. Mirrors the EX-ADD few-shot
# shape. The save is NOT done yet (user must tap the confirm card) — so we say
# "ขอยืนยัน … กดยืนยันเพื่อบันทึก", never "บันทึก…แล้ว" (which falsely implies
# the save already happened). We avoid naming the card's position ("ด้านล่าง"/
# "ด้านบน") so the copy never goes stale if the UI layout moves the card.
def _confirm_text(amount: Any, category_name: str) -> str:
    return (
        f"ขอยืนยันรายการ {_fmt_amount(amount)} บาท หมวด{category_name} "
        "— กดยืนยันเพื่อบันทึกได้เลยครับ"
    )


def _confirm_group_text(rows: list, total: Any) -> str:
    """"ขอยืนยัน 3 รายการ รวม 184 บาท (กาแฟ 60 · อาหาร 80 · BTS/MRT 44) — …".

    A mix of income and expense has no single meaningful "รวม" (the net read
    as "รวม -180 บาท"), so it states each side instead.
    """
    parts = " · ".join(f"{cat} {_fmt_amount(amt)}" for amt, cat, *_ in rows)
    types = {r[2] if len(r) > 2 else "expense" for r in rows}
    if types == {"expense"}:
        head = f"รวม {_fmt_amount(total)} บาท"
    elif types == {"income"}:
        head = f"รวมรายรับ {_fmt_amount(abs(total))} บาท"
    else:
        sums = {"expense": Decimal(0), "income": Decimal(0)}
        for amt, _cat, typ in rows:
            sums[typ] += Decimal(str(amt))
        head = (f"จ่าย {_fmt_amount(_num(sums['expense']))} บาท · "
                f"รับ {_fmt_amount(_num(sums['income']))} บาท")
    return (
        f"ขอยืนยัน {len(rows)} รายการ {head} ({parts}) "
        "— กดยืนยันเพื่อบันทึกทั้งหมดได้เลยครับ"
    )


def _num(d: Decimal) -> int | float:
    return int(d) if d == d.to_integral_value() else float(d)


def _fmt_amount(amount: Any) -> str:
    """Render the wire amount (int or float) with thousands separators.

    The amount comes straight from `_propose_core`'s Decimal canonicalization
    (int when integral, else float) — we only FORMAT for display here, never
    re-derive the value (no LLM/float arithmetic on money).
    """
    try:
        if isinstance(amount, int):
            return f"{amount:,}"
        if isinstance(amount, float):
            # Satang always shows two digits ("1,250.50", never "1,250.5").
            return f"{amount:,.2f}"
    except (TypeError, ValueError):
        pass
    return str(amount)


async def direct_propose_node(state: dict) -> dict:
    """ReAct-free ADD: run `_propose_core`, stream a deterministic confirmation.

    On error (onboarding / no-wallet / resolve-fail) → route back to `react`
    so the full agent can handle the edge case (clarify, onboard, retry). We do
    NOT try to recover here — the goal is a fast HAPPY path, not a second
    error handler.

    On success → merge `_propose_core`'s state writes (proposal block, pending,
    last_txn, proposals) and append a SINGLE AIMessage holding the deterministic
    confirmation. The SSE adapter's `messages` stream mode DOES forward this
    node-authored AIMessage as `answer_token` events (this node is a plain outer-
    StateGraph node, not inside the create_react_agent subgraph), and also
    buffers it into the closing answer block + checkpoint/history — all from ONE
    source. Route straight to `post_turn`.

    GOTCHA 1 (corrected): an earlier version ALSO pushed the same text via
    `get_stream_writer()({"answer_token": ...})`, on the assumption that
    messages-mode does NOT auto-stream a node-authored AIMessage. A real
    session log (CLASSIFY_ROUTER_ENABLED=1, direct_propose) proved that assumption
    WRONG: messages-mode DOES stream this node's AIMessage, so the two sources
    duplicated the confirmation (answer_chars=110 ≈ 55*2 — the same sentence
    printed twice, live tokens AND the buffered answer block). The custom
    writer is removed; the AIMessage is the single source of truth.
    """
    slots = state.get("__classified_add__") or {}
    group = slots.get("multi") and len(slots.get("items") or []) >= 2
    if group:
        result = await _propose_group_core(
            items=slots["items"], wallet_label=slots.get("wallet_label"), state=state,
        )
    else:
        result = await _propose_core(
            amount=slots.get("amount"),
            type=slots.get("type") or "expense",
            category_label=slots.get("category_label"),
            wallet_label=slots.get("wallet_label"),
            note=slots.get("note"),
            date_iso=slots.get("date_iso"),
            state=state,
        )

    if not result.ok:
        # Onboarding OR error → hand the turn to react (no recovery here).
        kind = result.error_kind or ("onboarding" if result.onboarding else "?")
        slog(
            "classify_router",
            f"direct_propose → react (propose_core not ok: kind={kind})",
        )
        return {"__classify_route__": _ROUTE_REACT, "__classified_add__": {}}

    summary = result.summary
    confirm = (
        _confirm_group_text(summary.get("rows") or [], summary.get("amount"))
        if group
        else _confirm_text(summary.get("amount"), summary.get("category_name", "อื่นๆ"))
    )

    updates: dict[str, Any] = dict(result.state_updates)
    # SINGLE source for the confirmation: this node-authored AIMessage is both
    # streamed live (messages-mode → answer_token) AND persisted to
    # checkpoint/history. See the corrected GOTCHA 1 in the docstring — pushing
    # it ALSO via get_stream_writer duplicated the text on the wire.
    updates["messages"] = [AIMessage(content=confirm)]
    slog(
        "classify_router",
        f"direct_propose OK amount={summary.get('amount')} "
        f"cat={summary.get('category_name')!r} → post_turn",
    )
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# History windowing — imported from graph.py to keep ONE definition. Lazy import
# avoids a circular import at module load (graph imports this package's nodes).
# ─────────────────────────────────────────────────────────────────────────────


def _window_messages(messages: list, max_turns: int) -> list:
    from src.agent.graph import _window_messages as _gw

    return _gw(messages, max_turns)


__all__ = [
    "classify_intent_node",
    "direct_propose_node",
    "is_classify_router_enabled",
    "route_after_classify",
]
