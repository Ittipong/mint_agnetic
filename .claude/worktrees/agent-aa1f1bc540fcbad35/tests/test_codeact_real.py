"""Real-sandbox CodeAct execution tests (Wave 7b adaptation).

Source: v2 `tests/test_codeact_real.py`. The v2 file exercised the
`build_codeact_subgraph` multi-step LLM-driven loop. v3 collapses that loop
into the single `run_python` tool inside the ReAct loop, so the multi-step
"LLM emits code -> sandbox runs -> LLM reads observation -> LLM emits more
code" structure is now a ReAct-loop concern, not a sub-graph one.

What's still load-bearing at unit level: the sandbox MUST actually EXECUTE
code (never return a stub observation) and errors MUST surface as structured
results (never as a fabricated number). That was the 1,234.56 hallucination
root cause (memory `project_codeact_dual_impl`) — we re-prove the contract
here against v3's tool, with no LLM in the loop.

Overlap with existing v3 unit tests:
- UT-T06 simple-result + validator-scratch -> `test_run_python.py`.
- UT-T06b missing context -> `test_run_python.py`.
- UT-T06c syntax error -> `test_run_python.py`.
- UT-NS01 namespace keys -> `test_codeact_namespace.py`.

What this file ADDS:
- UT-CR-201 (v3): the sandbox EXECUTES code — `result` reflects the real
  output. Hand-built code returns a Decimal-grounded amount; no stub.
- UT-CR-202 (v3): empty `result` propagates as the literal repr "[]" — no
  number ever appears in the observation that the next ReAct step reads.
- UT-CR-103 (v3): a banned construct (`import`) is rejected by the AST
  validator as an error; the next sandbox run with valid code recovers.
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


async def _invoke(code: str, state: dict | None = None, tool_call_id="tc-cr"):
    state = state or {
        "user_id": "u-1",
        "user_context": _user_context(),
        "tool_outputs_this_turn": [],
        "emitted_blocks_this_turn": [],
    }
    tool_call = {
        "name": "run_python",
        "args": {"code": code, "state": state},
        "type": "tool_call",
        "id": tool_call_id,
    }
    cmd = await run_python.ainvoke(tool_call)
    messages = (cmd.update or {}).get("messages") or []
    if messages:
        return cmd, json.loads(messages[0].content), state
    return cmd, {}, state


# ---------------------------------------------------------------------------
# UT-CR-201 (v3) — sandbox EXECUTES code, no stub observation
# ---------------------------------------------------------------------------
def test_UT_CR201_sandbox_executes_code_no_stub():
    """The sandbox MUST run the code and reflect its real output as `result`.
    The 1,234.56 hallucination was caused by an old code path that returned
    a placeholder observation without ever running the code — re-asserting
    here against v3 keeps that regression from reappearing."""

    async def run():
        _cmd, out, _ = await _invoke(
            'result = {"amount": str(Decimal("42.00"))}',
        )
        assert out["error"] is None
        # Real execution produced this exact repr — never a stub string.
        parsed = (out["result"] or "")
        assert "'amount': '42.00'" in parsed or '"amount": "42.00"' in parsed
        assert "stub" not in parsed.lower()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-CR-202 (v3) — empty result propagates as "[]"; no fabricated number
# ---------------------------------------------------------------------------
def test_UT_CR202_empty_result_propagates_as_no_data():
    """The no-wallet / no-data shape: `result = []`. The sandbox's `result`
    repr MUST be the literal "[]" — never a number — so the validator can
    detect "no data" without the next ReAct step inventing one."""

    async def run():
        _cmd, out, _ = await _invoke("result = []")
        assert out["error"] is None
        assert out["result"] == "[]"
        # No digit characters anywhere in the observation.
        assert not any(ch.isdigit() for ch in out["result"])

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-CR-103 (v3) — banned construct -> error, no crash, retry works
# ---------------------------------------------------------------------------
def test_UT_CR103_banned_import_is_rejected_then_retry_succeeds():
    """The AST validator rejects banned constructs (e.g. `import`) with a
    structured error. A SUBSEQUENT call with valid code on the SAME state
    proceeds normally — the tool MUST be stateless across invocations
    (no `result` shadowing from the bad first call)."""

    async def run():
        state = {
            "user_id": "u-1",
            "user_context": _user_context(),
            "tool_outputs_this_turn": [],
            "emitted_blocks_this_turn": [],
        }
        # First call: banned import -> error, no result.
        _cmd1, out1, _ = await _invoke("import os\nresult = 1", state=state)
        assert out1["result"] is None
        assert out1["error"] is not None
        # The error mentions the rejection (sandbox-error_kind / kind / msg
        # — exact text may evolve; assert on the presence of error+kind).
        assert "kind" in out1 or "import" in out1["error"].lower() \
            or "banned" in out1["error"].lower() \
            or "forbidden" in out1["error"].lower()

        # Second call: valid code -> recovers cleanly with a real result.
        _cmd2, out2, _ = await _invoke("result = 'recovered'", state=state)
        assert out2["error"] is None
        assert out2["result"] == "'recovered'"

        # Both calls logged into the validator scratch (per turn).
        outputs = state["tool_outputs_this_turn"]
        assert len(outputs) == 2

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-CR-203 (v3) — error result NEVER carries a number
# ---------------------------------------------------------------------------
def test_UT_CR203_error_result_carries_no_number():
    """A code path that raises must surface as `error` with `result=None` —
    NEVER carry a numeric repr forward. This is the validator's safety net
    against hallucination on the next ReAct step."""

    async def run():
        _cmd, out, _ = await _invoke("result = 1 / 0")
        assert out["result"] is None
        assert out["error"] is not None
        # The error message may contain "0" (e.g. "division by zero") — that's
        # ok; the contract is that `result` is None, so no number is read
        # forward as the answer.
        assert "ZeroDivisionError" in out["error"] or "division" in out["error"].lower()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# UT-CR-CB (v3) — run_python drains the category_breakdown block sink
# ---------------------------------------------------------------------------
def test_UT_CR_CB01_run_python_emits_category_breakdown(monkeypatch):
    """When sandbox code calls sum_by_category and the result has 2+ buckets,
    run_python MUST forward the staged category_breakdown block via
    Command(update={'emitted_blocks_this_turn': [...]}) — the Option C sink
    drain. `_run_query` is patched so no DB is touched; the cached
    user_context path supplies the catalog."""
    from src.agent.tools.codeact import namespace as ns_mod
    from decimal import Decimal

    async def fake_run_query(spec, user_id):
        return [
            {"bucket": "อาหาร", "currency": "THB",
             "amount": Decimal("900"), "cnt": 3,
             "icon": {"type": "asset", "value": "food.png"}},
            {"bucket": "เดินทาง", "currency": "THB",
             "amount": Decimal("800"), "cnt": 2, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        code = (
            "rows = sum_by_category(start='2026-05-01', end='2026-05-31')\n"
            "result = {'rows': len(rows)}"
        )
        cmd, out, _state = await _invoke(code)
        assert out["error"] is None
        update = cmd.update or {}
        blocks = update.get("emitted_blocks_this_turn") or []
        assert len(blocks) == 1, f"expected one breakdown block, got {blocks}"
        block = blocks[0]
        assert block["type"] == "category_breakdown"
        assert block["total"] == 1700
        assert {it["name"] for it in block["items"]} == {"อาหาร", "เดินทาง"}
        # The block survives a plain json.dumps (the sse_adapter path).
        json.dumps(block, ensure_ascii=False)

    asyncio.run(run())


def test_UT_CR_CB02_run_python_no_block_when_single_category(monkeypatch):
    """A single-category result stages NO block (C2b guard) — run_python's
    Command must NOT carry emitted_blocks_this_turn for the breakdown."""
    from src.agent.tools.codeact import namespace as ns_mod
    from decimal import Decimal

    async def fake_run_query(spec, user_id):
        return [
            {"bucket": "อาหาร", "currency": "THB",
             "amount": Decimal("900"), "cnt": 3, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        code = (
            "rows = sum_by_category(start='2026-05-01', end='2026-05-31')\n"
            "result = {'rows': len(rows)}"
        )
        cmd, out, _state = await _invoke(code)
        assert out["error"] is None
        update = cmd.update or {}
        assert "emitted_blocks_this_turn" not in update
        assert update.get("tool_outputs_this_turn")  # normal scratch still set

    asyncio.run(run())
