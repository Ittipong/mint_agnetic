"""CodeAct sandbox variable persistence — Wave 7b adaptation.

v2 `tests/test_codeact_state_persistence.py` asserted that user variables
defined in step N of the CodeAct subgraph survived into step N+1's namespace
(REPL semantics). v3 removed the subgraph: `run_python` is a SINGLE-shot
tool invoked once per ReAct step.

In v3 the "carry forward across reasoning steps" is the ReAct loop's job —
the LLM re-sends prior tool results via `messages`, then re-emits Python
when it needs another computation. There is no shared sandbox namespace
across tool calls by design (statelessness is a defense against subtle
data leaks across users / requests).

DECISION (Wave 7b): the v2 REPL-persistence contract does NOT apply to v3.
Re-asserting "user variables persist" would CONTRADICT v3's design.
Instead, this file asserts the inverse:

  UT-CSP-001 (v3): two SEPARATE run_python invocations on the SAME
                   conversational state DO NOT share a namespace —
                   a variable defined in call #1 is undefined in call #2.

Flagged for human review:
  Whether v3 needs an explicit `run_python_session_namespace` parameter for
  multi-step computations that genuinely need carry-forward. Current bet is
  the LLM re-emits the prior variable as a literal — the messages history
  is the carry-forward channel. If perf becomes an issue, this decision
  can be revisited.
"""

from __future__ import annotations

import asyncio
import json

from src.agent.tools.codeact import run_python


def _user_context() -> dict:
    return {
        "wallets": [
            {"sync_id": "w-1", "name": "Cash", "currency": "THB",
             "wallet_type": "general", "is_default": True, "usage_count": 0},
        ],
        "categories": [],
        "tags": [],
        "wallet_id": "w-1",
        "default_currency_code": "THB",
        "fetched_at": "2026-05-28T00:00:00Z",
    }


async def _invoke(code: str, state: dict, tool_call_id: str = "tc-csp"):
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


# ---------------------------------------------------------------------------
# UT-CSP-001 (v3) — run_python invocations do NOT share namespace
# ---------------------------------------------------------------------------
def test_UT_CSP001_sandbox_namespace_does_not_persist_across_invocations():
    """A variable defined in run_python call #1 is UNDEFINED in call #2 on
    the same state — sandboxes are stateless across invocations by design.

    This is the inverse of v2's CodeAct REPL-persistence contract; v3 relies
    on the LLM re-emitting prior context via messages instead of shared
    Python state across tool calls."""

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }

        # Call 1: define `x`.
        _cmd1, out1 = await _invoke("x = 42\nresult = x", state)
        assert out1["error"] is None
        assert out1["result"] == "42"

        # Call 2: reference `x` -> NameError (fresh namespace, x is unknown).
        _cmd2, out2 = await _invoke("result = x", state)
        assert out2["result"] is None
        assert out2["error"] is not None
        assert "NameError" in out2["error"] or "x" in out2["error"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-CSP-002 (v3) — namespace builtins ARE available every call
# ---------------------------------------------------------------------------
def test_UT_CSP002_namespace_builtins_available_each_call():
    """Even though user variables don't persist, the canonical sandbox
    namespace (Decimal, date, the SQL wrappers, etc.) IS available on EVERY
    call — that's `build_namespace`'s contract."""

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }

        # Call 1: use Decimal.
        _cmd1, out1 = await _invoke('result = Decimal("1.50")', state)
        assert out1["error"] is None
        # Repr matches Decimal output (str(Decimal('1.50')) -> '1.50').
        assert "1.50" in out1["result"]

        # Call 2: use Decimal again — must still be available.
        _cmd2, out2 = await _invoke('result = Decimal("99.99")', state)
        assert out2["error"] is None
        assert "99.99" in out2["result"]

    asyncio.run(run())
