"""Unit tests for src.agent.tools.wallet_required_cta.

Covers UT-T11.

Wave 4 update — Command(update=) return type:
  After Issue 1 fix, `wallet_required_cta` returns `Command(update=...)`.
  The `_invoke` helper drives the tool via the ToolCall protocol and
  unwraps both the Command and the JSON ToolMessage content.
"""

from __future__ import annotations

import asyncio
import json

from src.agent.tools.wallet_required_cta import wallet_required_cta


async def _invoke(*, args: dict, tool_call_id: str = "tc-test"):
    """Invoke wallet_required_cta via the ToolCall protocol."""
    tool_call = {
        "name": "wallet_required_cta",
        "args": args,
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await wallet_required_cta.ainvoke(tool_call)
    messages = (cmd.update or {}).get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content)
    return cmd, {}


def test_UT_T11_emits_wallet_required_block_with_action_target():
    """UT-T11: the tool must emit ONE `wallet_required` block carrying
    action.label + action.target. Mobile reads `action.target` to deep-link
    into the create-wallet flow."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, out = await _invoke(args={"state": state})

        assert out["cta_emitted"] is True

        # Command(update=) carries the new block — that's what the outer
        # graph's append_reducer reads at the subgraph boundary.
        update_blocks = cmd.update["emitted_blocks_this_turn"]
        assert len(update_blocks) == 1
        block = update_blocks[0]
        assert block["type"] == "wallet_required"
        assert "กระเป๋า" in block["text"]
        assert block["action"]["label"] == "สร้างกระเป๋าเงิน"
        assert block["action"]["target"] == "/wallets/create"

        # In-place mutation also reflects the new block for direct-test
        # inspection of `state`.
        assert state["emitted_blocks_this_turn"] == update_blocks

    asyncio.run(run())


def test_UT_T11_custom_message_and_button_overrides_default():
    """The LLM can pass a custom Thai message/button label. The tool honors
    both without other behavior changes."""

    async def run():
        state = {"emitted_blocks_this_turn": []}
        cmd, _out = await _invoke(args={
            "message": "เริ่มต้นด้วยการสร้างกระเป๋านะครับ",
            "button_label": "เริ่มเลย",
            "state": state,
        })
        block = cmd.update["emitted_blocks_this_turn"][0]
        assert block["text"] == "เริ่มต้นด้วยการสร้างกระเป๋านะครับ"
        assert block["action"]["label"] == "เริ่มเลย"
        # target is fixed — mobile deep-link contract.
        assert block["action"]["target"] == "/wallets/create"

    asyncio.run(run())
