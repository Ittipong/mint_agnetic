"""Slip → transaction proposal tool.

The ReAct agent (vision LLM in slip_node) calls this tool to surface
an extracted transaction. The tool body itself is intentionally a
**no-op stub** — it only declares the schema so the LLM can produce
structured arguments. Actual validation + custom-event dispatch
happens in `propose_validation_node` (see graph/slip_node.py), where
we have access to the user's catalog and can null-out hallucinated
sync_ids before any payload reaches the mobile client.

Schema is the wire contract — keep it in lockstep with the slim
payload mobile expects (no display fields; mobile resolves names /
icons / colors from sync_ids via its own repositories).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool
async def propose_transaction(
    type: Literal["expense", "income"],
    amount: float,
    date: str,
    note: str = "",
    wallet_id: str | None = None,
    category_id: str | None = None,
    merchant_name: str | None = None,
    currency_code: str = "THB",
    currency_symbol: str = "฿",
    include_in_report: bool = True,
) -> str:
    """Propose a transaction extracted from a slip image.

    Call this tool **once per line item** when a slip image is
    attached. A retail receipt with 3 items + VAT + 1 discount =
    5 separate tool calls in one response (3 items + 1 VAT +
    1 discount), each with its own category and amount.

    Extract slip data from the image, match wallet/category against
    the user's catalog (provided in the system prompt under
    `Slip context` with sync_ids), and pass the matched ids here.

    `wallet_id` / `category_id` **MUST** be a sync_id (UUID) that
    appears verbatim in the system-prompt catalog. Always pass a
    value — the prompt's fallback rules guarantee at least one match
    (nearest by name → first wallet / first category of the matched
    type). Never pass null; the server expects every id to resolve.

    Args:
        type: "expense" for money leaving the wallet, "income" for
              money entering it. Discounts are modeled as "income"
              with `include_in_report=false`.
        amount: Positive numeric amount for this line item only
              (NOT the receipt total when items are split).
        date: ISO 8601 timestamp. Convert Buddhist Era to AD
              (subtract 543) before formatting. If the slip has no
              timestamp, use today's date.
        note: One-line summary describing what this transaction is
              about — e.g. "นม @ Tesco", "VAT 7%", "ส่วนลด @ Tesco",
              "มื้อเที่ยง @ KFC". The server merges merchant_name into
              this field automatically.
        wallet_id: sync_id of the matched **general** wallet. Always
                   required.
        category_id: sync_id of the matched category from the matched
                     wallet's category list. Always required.
        merchant_name: Optional merchant/employer name from the slip
                       header — server merges it into the final note.
        currency_code: ISO currency code (default THB).
        currency_symbol: Display symbol (default ฿).
        include_in_report: Whether this transaction should appear in
                           income/expense reports. Default true. Set
                           to **false** for discount lines so they
                           don't inflate the user's income totals.

    Returns:
        A short status string. The validation node intercepts the
        tool call and dispatches the actual SSE event — the LLM
        should reply briefly after with something like
        "ดูข้อมูลในการ์ดด้านบนได้เลยครับ".
    """
    # Stub — see propose_validation_node for the real implementation.
    # We return a stable string so the LLM gets a clean ToolMessage
    # and continues to the wrap-up reply.
    return "(proposal received — validation deferred to server)"
