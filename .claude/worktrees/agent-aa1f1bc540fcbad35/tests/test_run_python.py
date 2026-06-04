"""Unit tests for src.agent.tools.codeact (the `run_python` tool wrapper).

Covers UT-T06..T07.

The `run_python` tool wraps the sandbox + namespace at the LangChain @tool
boundary. Tests assert:
- UT-T06: returns repr + stdout and writes an entry to
  `tool_outputs_this_turn` (via Command(update=)) so the numerical
  validator (Wave 1) can cross-check the answer's money numbers.
- UT-T07: ClarificationNeeded raised inside the sandbox is caught here and
  surfaced as a `clarification` block via Command(update=) — it MUST NOT
  escape the tool, or the ReAct loop dead-loops.

Wave 4 update — Command(update=) return type:
  After Issue 1 fix, `run_python` returns Command instead of dict. The
  `_invoke` helper drives via the ToolCall protocol and unwraps the
  ToolMessage payload back into the dict tests previously asserted on.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
)
from src.agent.tools.codeact import run_python


async def _invoke(*, code: str, state: dict, tool_call_id: str = "tc-test"):
    """Invoke run_python via the ToolCall protocol.

    Returns (command, payload) where `payload` is the parsed ToolMessage
    JSON — same shape `out` used to be in the pre-Wave-4 tests.
    """
    tool_call = {
        "name": "run_python",
        "args": {"code": code, "state": state},
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await run_python.ainvoke(tool_call)
    messages = (cmd.update or {}).get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content)
    return cmd, {}


def _user_context_dict() -> dict:
    """Marshal a tiny catalog to the wire shape `get_user_context` produces.

    Mirrors EntityCatalog.to_dict() — but as a hand-built dict so we keep
    the test independent of the loader path.
    """
    return {
        "wallets": [
            {"sync_id": "w-1", "name": "Cash", "currency": "THB",
             "wallet_type": "general", "is_default": True, "usage_count": 0},
        ],
        "categories": [
            {"sync_id": "c-1", "name": "อาหาร", "type": "expense",
             "parent_id": None, "usage_count": 0},
        ],
        "tags": [],
        "wallet_id": "w-1",
        "default_currency_code": "THB",
        "fetched_at": "2026-05-15T00:00:00Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# UT-T06 — run_python returns repr + stdout + records to validator scratch
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T06_run_python_returns_result_and_records_for_validator():
    """UT-T06: a simple expression `result = 1 + 2` returns the repr in
    `result`, captures stdout, and pushes an entry into
    `tool_outputs_this_turn` (via Command(update=)) so the numerical
    validator (Wave 1) can cross-check money numbers in the final answer."""

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context_dict(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }
        cmd, out = await _invoke(
            code="print('hello')\nresult = 1 + 2",
            state=state,
        )

        assert out["error"] is None
        assert out["result"] == "3"        # repr of int(3)
        assert "hello" in out["stdout"]

        # Validator scratch is propagated via Command(update=) so the outer
        # graph's append_reducer sees it. (In-place state mutation also
        # reflects it for direct-invocation tests — both views agree.)
        cmd_outputs = cmd.update.get("tool_outputs_this_turn") or []
        assert len(cmd_outputs) == 1
        assert cmd_outputs[0]["tool"] == "run_python"
        assert cmd_outputs[0]["result"] == "3"
        # In-place mirror (defensive write inside the tool):
        assert state["tool_outputs_this_turn"][0]["result"] == "3"

    asyncio.run(run())


def test_UT_T06b_run_python_lazy_loads_context_when_missing():
    """Without `user_context` the tool loads the catalog inline. Three
    legitimate outcomes when no DB is wired in unit tests:

      - real DB available → `result=1`, no error.
      - DB raises → structured `context_load_failed` error, never a raise.
      - DB returns an empty catalog → tool emits the `wallet_required`
        block in-place + `kind="wallet_required"` (post-Wave 5 onboarding
        gate; see `build_wallet_required_block`).

    The shared invariant: we never short-circuit with the old
    `missing_context` error and never raise from inside the tool."""

    async def run():
        state = {"user_id": "u-1"}  # user_context deliberately omitted
        _cmd, out = await _invoke(code="result = 1", state=state)
        assert out.get("kind") != "missing_context"
        if out.get("error"):
            assert out["kind"] in ("context_load_failed", "wallet_required")

    asyncio.run(run())


def test_UT_T06c_run_python_returns_error_for_bad_syntax():
    """Syntax errors from the sandbox surface as `error` strings, NOT as
    raises — the LLM reads the error and retries with corrected code."""

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context_dict(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }
        _cmd, out = await _invoke(code="result = ", state=state)
        assert out["result"] is None
        assert out["error"] is not None
        assert "SyntaxError" in out["error"]

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T07 — ClarificationNeeded → clarification block, no raise
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T07_clarification_emits_block_without_raising(monkeypatch):
    """UT-T07: when sandbox code calls `clarify(...)`, the
    ClarificationNeeded exception MUST be caught inside the tool and
    surfaced as a `clarification` block in Command(update=).
    The tool returns `{"clarification_emitted": True, ...}` so the LLM
    knows to stop and wait.
    """

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context_dict(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }
        cmd, out = await _invoke(
            code="clarify('กระเป๋าไหน', ['Cash', 'KBank'])",
            state=state,
        )

        # No raise — return value signals to the LLM that we paused.
        assert out["clarification_emitted"] is True
        assert "กระเป๋าไหน" in out["question"]
        assert out["options"] == ["Cash", "KBank"]

        # Block landed in the Command's update channel for the graph's
        # append_reducer to merge.
        update_blocks = cmd.update.get("emitted_blocks_this_turn") or []
        assert len(update_blocks) == 1
        assert update_blocks[0]["type"] == "clarification"
        assert update_blocks[0]["text"] == "กระเป๋าไหน"
        assert {"label": "Cash", "send": "Cash"} in update_blocks[0]["options"]
        # In-place mirror also reflects the block for backward-compat tests.
        assert state["emitted_blocks_this_turn"] == update_blocks

    asyncio.run(run())
