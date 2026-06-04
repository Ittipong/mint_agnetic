"""`wallet_required_cta` — onboarding CTA block + builder.

NEW in Wave 3. Phase 2 spec: docs/v3/phase2_tools_design.md Tool 3.

ROLE CHANGE (post-wave 5): no longer registered as a ReAct tool. The
mint_v3 agent emits the wallet_required block in-place from the data tools
that detect the no-wallet condition:

  - `propose_transaction` (ADD path) — onboarding shortcut on `wallets=[]`.
  - `run_python` (analyst path) — same gate before invoking the sandbox.

Both call `build_wallet_required_block()` so the mobile renderer treats
both paths identically. The standalone `@tool` wrapper is kept for
backward-compatibility but is NOT in `ALL_TOOLS` — the LLM cannot reach it.

The block carries `action.target = "/wallets/create"` which the mobile
client deep-links into the wallet-create flow.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog


_STATUS_WORD = "กำลังเช็คกระเป๋า..."
DEFAULT_WALLET_CTA_TEXT = "ยังไม่มีกระเป๋าครับ สร้างก่อนเพื่อเริ่มบันทึกรายการ"
DEFAULT_WALLET_CTA_BUTTON = "สร้างกระเป๋าเงิน"
WALLET_CTA_DEEP_LINK = "/wallets/create"


def build_wallet_required_block(
    *,
    message: str = DEFAULT_WALLET_CTA_TEXT,
    button_label: str = DEFAULT_WALLET_CTA_BUTTON,
) -> dict[str, Any]:
    """Return the wallet_required block payload — the single source of truth
    for the block shape, reused by `propose_transaction`, `run_python`, and
    the legacy `wallet_required_cta` tool wrapper."""
    return {
        "type": "wallet_required",
        "text": message,
        "action": {
            "label": button_label,
            "target": WALLET_CTA_DEEP_LINK,
        },
    }


@tool
async def wallet_required_cta(
    message: str = DEFAULT_WALLET_CTA_TEXT,
    button_label: str = DEFAULT_WALLET_CTA_BUTTON,
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Emit a CTA prompting the user to create a wallet.

    Retained for tests + direct invocation. Not exposed to the ReAct LLM —
    `propose_transaction` / `run_python` emit this block in-place when they
    detect `wallets=[]`.
    """
    _emit_status(_STATUS_WORD)

    block = build_wallet_required_block(message=message, button_label=button_label)
    # NOTE: do NOT mutate state in-place — see issue #SSE-DUP in
    # emit_suggestions.py. The Command(update=) return alone is the canonical
    # write; the append_reducer applies it once, no duplicate on the wire.
    slog("wallet_required_cta", f"emitted CTA: {message!r}")

    return Command(update={
        "emitted_blocks_this_turn": [block],
        "messages": [
            ToolMessage(
                content=json.dumps({"cta_emitted": True}),
                tool_call_id=tool_call_id,
            ),
        ],
    })


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


__all__ = [
    "wallet_required_cta",
    "build_wallet_required_block",
    "DEFAULT_WALLET_CTA_TEXT",
    "DEFAULT_WALLET_CTA_BUTTON",
    "WALLET_CTA_DEEP_LINK",
]
