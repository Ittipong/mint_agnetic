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
) -> str:
    """Propose a transaction extracted from a slip image.

    Call this **only** when the user message starts with
    `[INTENT:parse_transaction_from_slip]` and a slip image is
    attached. Extract slip data from the image, match wallet/category
    against the user's catalog (provided in the system prompt under
    `Slip context` with sync_ids), and pass the matched ids here.

    `wallet_id` / `category_id` **MUST** be a sync_id (UUID) that
    appears verbatim in the system-prompt catalog. The server-side
    validation node will null-out any id not in the user's catalog,
    so guessing only loses you the match. When in doubt, pass null.

    Args:
        type: "expense" or "income". Pick by money direction relative
              to the user.
        amount: Positive numeric amount (the slip total).
        date: ISO 8601 timestamp. Convert Buddhist Era to AD
              (subtract 543) before formatting.
        note: One-line summary. For receipts with multiple items,
              concatenate them comma-separated. Server will append
              `merchant_name` after a separator automatically.
        wallet_id: sync_id of the matched **general** wallet, or null.
        category_id: sync_id of the matched category (from the
                     matched wallet's category list), or null.
        merchant_name: Optional merchant/employer name from the slip
                       header — server merges it into the final note.
        currency_code: ISO currency code (default THB).
        currency_symbol: Display symbol (default ฿).

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
