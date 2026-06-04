"""Unit tests for src.agent.graph — the v3 ReAct + CodeAct graph builder.

Covers Wave 4 UT-G01..UT-G04.

Strategy:
  We drive the graph with a `FakeToolingModel` so every test runs offline
  and produces deterministic tool-call sequences. The fake replays a fixed
  list of AIMessages — each test ships its own script. This keeps the
  ReAct loop honest: every step the agent takes is driven by what the
  fake emits next, and we never depend on a real LLM's flakiness or cost.

  The store + checkpointer are wired via test factories so no Postgres
  connection ever opens.

Why the entity_catalog loader gets patched:
  `get_user_context` calls into `entity_catalog.load_catalog_for_user`.
  We swap that out via `set_catalog_loader(...)` so the catalog returns a
  small in-memory `EntityCatalog` instead of hitting BACKEND_DATABASE_URL.

UT-G04 mechanism — validator retry:
  The fake LLM emits a FIRST answer with a number NOT in any tool output
  ("เดือนนี้ใช้ไป 99999 บาท" while the only tool said 250). The
  post_model_hook fires, injects a [VALIDATOR_RETRY] SystemMessage, and
  the fake's second response (corrected) uses the right number. We assert
  `__validator_retries__ == 1` at end-of-turn.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any, Iterator, Sequence
from unittest.mock import patch

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
    set_catalog_loader,
)
from src.agent.graph import build_graph


# ─────────────────────────────────────────────────────────────────────────────
# Fake LLM — replays a pre-canned list of AIMessages
# ─────────────────────────────────────────────────────────────────────────────


class FakeToolingModel(BaseChatModel):
    """Deterministic chat model for ReAct loop tests.

    Pops one canned AIMessage per `_generate` call. `bind_tools` is a no-op
    so `create_react_agent`'s internal binding step doesn't crash on a
    fake. Each test instantiates a fresh model with its own script —
    DON'T share an instance across tests, the cursor would carry over.
    """

    responses: list = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake_react"

    def _generate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        idx = self.cursor
        # Pydantic models freeze attrs; use object.__setattr__ to bump.
        object.__setattr__(self, "cursor", idx + 1)
        if idx >= len(self.responses):
            raise RuntimeError(
                f"FakeToolingModel ran out of responses at call #{idx + 1}"
            )
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        return self._generate(messages, stop=stop, **kwargs)

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        # Fake models don't actually need to know about tools; the
        # AIMessages we replay already carry hand-crafted tool_calls.
        return self


def _seed_catalog() -> None:
    """Install a fake catalog loader so `get_user_context` never hits the DB."""

    async def loader(user_id: str) -> EntityCatalog:
        return EntityCatalog(
            wallets=[
                WalletEntry(sync_id="w-cash", name="Cash", currency="THB",
                            wallet_type="general", is_default=True),
            ],
            categories=[
                CategoryEntry(sync_id="c-coffee", name="กาแฟ", type="expense"),
            ],
        )

    set_catalog_loader(loader)


@pytest.fixture(autouse=True)
def _reset_catalog_loader():
    """Each test gets a fresh loader binding (cleared after)."""
    set_catalog_loader(None)
    yield
    set_catalog_loader(None)


async def _build_test_graph(model: BaseChatModel, *, react_impl: str | None = None):
    """Build a graph with an in-memory checkpointer + no store.

    The MemorySaver lets each test resume / continue threads without
    needing a Postgres pool. Store is None — memory tools simply degrade.

    `react_impl` pins REACT_IMPL for this build (restored after). The numerical
    validator + tool-error guard live ONLY on the prebuilt path now (removed from
    flat by request), so tests for that behavior pass `react_impl="prebuilt"`.
    """
    async def memsaver_factory():
        return MemorySaver()

    prev = os.environ.get("REACT_IMPL")
    if react_impl is not None:
        os.environ["REACT_IMPL"] = react_impl
    try:
        return await build_graph(
            repo=None,
            model=model,
            checkpointer_factory=memsaver_factory,
        )
    finally:
        if react_impl is not None:
            if prev is None:
                os.environ.pop("REACT_IMPL", None)
            else:
                os.environ["REACT_IMPL"] = prev


# ─────────────────────────────────────────────────────────────────────────────
# UT-G01 — simple ADD ends with transaction_proposal block
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G01_simple_add_emits_transaction_proposal_block():
    """UT-G01: a fresh ADD turn ("เพิ่ม 250 กาแฟ") flowing
    get_user_context → propose_transaction → final answer MUST end with a
    `transaction_proposal` block in `emitted_blocks_this_turn`."""
    _seed_catalog()

    # Fake script:
    #   step 1 — call get_user_context to bootstrap the catalog
    #   step 2 — call propose_transaction with the parsed values
    #   step 3 — final answer (text confirmation, no tool calls)
    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_user_context",
                "args": {},
                "id": "tc-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "propose_transaction",
                "args": {
                    "amount": 250,
                    "type": "expense",
                    "category_label": "กาแฟ",
                    "note": "กาแฟ",
                },
                "id": "tc-2",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="บันทึก 250 บาทหมวดกาแฟแล้วครับ ยืนยันด้านบนได้เลย"),
    ]
    model = FakeToolingModel(responses=responses)

    async def run():
        graph = await _build_test_graph(model)
        out = await graph.ainvoke(
            {
                "user_id": "u-1",
                "thread_id": "t-G01",
                "messages": [HumanMessage("เพิ่ม 250 กาแฟ")],
            },
            config={"configurable": {"thread_id": "t-G01"},
                    "recursion_limit": 25},
        )

        # 1. A transaction_proposal block landed in emitted_blocks_this_turn.
        block_types = [
            b.get("type") for b in (out.get("emitted_blocks_this_turn") or [])
        ]
        assert "transaction_proposal" in block_types, (
            f"expected transaction_proposal block; got {block_types}"
        )

        # 2. pending_proposal points at the new proposal (Command(update=)).
        pending = out.get("pending_proposal")
        assert pending is not None
        assert pending["status"] == "pending"
        assert pending["payload"]["amount"] == 250

        # 3. post_turn_hook wrote last_txn from the new proposal.
        last_txn = out.get("last_txn")
        assert last_txn is not None
        assert last_txn["pending"] is True
        assert last_txn["amount"] == 250

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-G02 — pre_turn_hook clears scratch
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G02_pre_turn_hook_clears_scratch_between_turns():
    """UT-G02: turn N's `tool_outputs_this_turn` / `emitted_blocks_this_turn`
    / `user_context` / validator scratch MUST NOT leak into turn N+1.

    We drive two turns on the same thread_id. The first turn populates
    scratch. The second turn starts with a fake LLM that gives an
    immediate answer (no tools, no proposal). After turn 2 ends, the per-
    turn lists should reflect ONLY turn 2's activity (empty).
    """
    _seed_catalog()

    # Turn 1 — call get_user_context + propose, then answer.
    turn1_responses = [
        AIMessage(content="", tool_calls=[
            {"name": "get_user_context", "args": {},
             "id": "tc1", "type": "tool_call"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "propose_transaction",
             "args": {"amount": 100, "type": "expense",
                      "category_label": "กาแฟ"},
             "id": "tc2", "type": "tool_call"},
        ]),
        AIMessage(content="ok turn1"),
    ]
    # Turn 2 — straight text answer, no tool calls.
    turn2_responses = [
        AIMessage(content="สวัสดีครับ"),
    ]
    model = FakeToolingModel(responses=turn1_responses + turn2_responses)

    async def run():
        graph = await _build_test_graph(model)
        config = {"configurable": {"thread_id": "t-G02"}, "recursion_limit": 25}

        # Turn 1
        out1 = await graph.ainvoke(
            {"user_id": "u-1", "thread_id": "t-G02",
             "messages": [HumanMessage("เพิ่ม 100 กาแฟ")]},
            config=config,
        )
        # Sanity: turn 1 produced a proposal block.
        block_types_1 = [b.get("type") for b in out1.get("emitted_blocks_this_turn", [])]
        assert "transaction_proposal" in block_types_1

        # Turn 2 — new user message, same thread.
        out2 = await graph.ainvoke(
            {"messages": [HumanMessage("สวัสดี")]},
            config=config,
        )

        # 1. Per-turn lists reflect ONLY turn 2 (empty — no tools, no blocks).
        assert out2.get("tool_outputs_this_turn", []) == [], (
            "tool_outputs_this_turn should be empty after pre_turn_hook reset"
        )
        assert out2.get("emitted_blocks_this_turn", []) == [], (
            "emitted_blocks_this_turn should be empty after pre_turn_hook reset"
        )
        # 2. Validator scratch was reset.
        assert out2.get("__validator_retries__", 0) == 0
        assert out2.get("__validator_failed__", False) is False
        # 3. user_context cleared at turn boundary — turn 2 didn't call
        # get_user_context so it stays None.
        assert out2.get("user_context") is None

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-G03 — recursion_limit prevents infinite loops
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G03_recursion_limit_raises_graph_recursion_error():
    """UT-G03: a model that keeps emitting tool_calls forever MUST trip
    `recursion_limit` and raise `GraphRecursionError` rather than hang.

    Strategy: feed a long stream of AIMessages, EACH carrying a tool_call.
    The ReAct loop will call the tool, the tool returns, the model emits
    ANOTHER tool_call — repeat. We set recursion_limit=4 (small) so the
    test is fast.
    """
    _seed_catalog()

    # Pathological: every step calls get_user_context. Repeat 50× so we
    # don't run out of responses before recursion fires.
    pathological = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_user_context",
                "args": {},
                "id": f"tc-{i}",
                "type": "tool_call",
            }],
        )
        for i in range(50)
    ]
    model = FakeToolingModel(responses=pathological)

    async def run():
        graph = await _build_test_graph(model)
        with pytest.raises(GraphRecursionError):
            await graph.ainvoke(
                {"user_id": "u-1", "thread_id": "t-G03",
                 "messages": [HumanMessage("hang me")]},
                config={"configurable": {"thread_id": "t-G03"},
                        "recursion_limit": 4},
            )

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-G04 — post_model_hook validator: ungrounded number → SOFT-WARN (no retry)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G04_validator_soft_warns_ungrounded_number():
    """UT-G04: when the LLM emits a final answer with a money number NOT
    grounded in any tool output, the validator MUST NOT retry or reject. It
    appends a warning footer to the SAME answer and delivers it. So
    `__validator_retries__` stays 0, `__validator_failed__` stays False, the
    original number survives, and the warning marker is present.

    Script (only THREE responses — no correction turn):
      step 1 — call get_user_context (so `user_context` is populated for run_python)
      step 2 — call run_python returning {amount: 250}
      step 3 — final answer claiming "เดือนนี้ใช้ไป 99999 บาท" — ungrounded
               → soft-warned, delivered as-is with footer
    """
    from src.agent.validators.numerical import _WARNING_MARKER

    _seed_catalog()

    # Patch the codeact sandbox so run_python doesn't actually execute Python.
    # We return a controlled result + stdout. The tool imports `execute`
    # by name (`from .sandbox import execute`) into its own module, so we
    # need to patch the BINDING IN THE TOOL'S MODULE, not the source.
    import importlib
    codeact_mod = importlib.import_module("src.agent.tools.codeact")

    def fake_execute(code, namespace):
        # Mimic the (result, stdout, error) tuple `execute` returns.
        return ({"amount_thb": 250}, "", None)

    responses = [
        # Step 1: bootstrap context
        AIMessage(content="", tool_calls=[
            {"name": "get_user_context", "args": {},
             "id": "tc-1", "type": "tool_call"},
        ]),
        # Step 2: run_python — produces the ground-truth 250.
        AIMessage(content="", tool_calls=[
            {"name": "run_python",
             "args": {"code": "result = sum_expense(...)"},
             "id": "tc-2", "type": "tool_call"},
        ]),
        # Step 3: ungrounded 99,999 ≠ 250 → soft-warned + delivered (no retry).
        AIMessage(content="เดือนนี้ใช้ไป 99,999 บาทครับ"),
    ]
    model = FakeToolingModel(responses=responses)

    async def run():
        with patch.object(codeact_mod, "execute", fake_execute):
            graph = await _build_test_graph(model, react_impl="prebuilt")
            out = await graph.ainvoke(
                {"user_id": "u-1", "thread_id": "t-G04",
                 "messages": [HumanMessage("เดือนนี้ใช้ไปเท่าไหร่")]},
                config={"configurable": {"thread_id": "t-G04"},
                        "recursion_limit": 25},
            )

        # 1. No retry loop — the validator never bounces the model back.
        assert out.get("__validator_retries__", 0) == 0, (
            f"expected 0 retries (soft-warn), got {out.get('__validator_retries__')}"
        )
        # 2. No terminal rejection / silent error.
        assert out.get("__validator_failed__", False) is False

        # 3. The model's answer is delivered as-is (number NOT dropped).
        ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
        assert ai_messages, "no AIMessage in final state"
        final_text = ai_messages[-1].content
        assert "99,999" in final_text, (
            f"answer was dropped/altered instead of delivered: {final_text!r}"
        )

        # 4. The warning footer is appended to that same answer.
        assert _WARNING_MARKER in final_text, (
            f"soft-warn footer missing from delivered answer: {final_text!r}"
        )

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-G05 — tool-error finalize guard forces a retry, then recovers
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G05_tool_error_guard_forces_retry_then_recovers():
    """UT-G05: structural R10 enforcement (the "เทียบ 3 เดือน" trace bug).

    When the model tries to finalize with a give-up apology while THIS turn's
    most recent run_python ended in a RETRYABLE error, the post_model_hook
    MUST drop the apology and force a loop-back — NOT accept the give-up.
    The corrected second run_python succeeds and the final answer is grounded.

    Script:
      1. get_user_context (bootstrap)
      2. run_python (BAD call) → fake_execute returns a TypeError
      3. apology "ขออภัย..." ← guard fires: removes it, rewrites the failed
         ToolMessage, routes back to agent (__tool_error_retries__ → 1)
      4. run_python (CORRECTED) → fake_execute returns 5520
      5. final answer "เทรนด์ล่าสุด 5,520 บาท" — grounded, passes
    """
    _seed_catalog()

    import importlib
    codeact_mod = importlib.import_module("src.agent.tools.codeact")

    calls = {"n": 0}

    def fake_execute(code, namespace):
        calls["n"] += 1
        if calls["n"] == 1:
            # Exactly the trace failure: wrong compare_periods signature.
            return (
                None,
                "",
                "TypeError: compare_periods() missing 2 required "
                "keyword-only arguments: 'period2_start' and 'period2_end'",
            )
        return ({"amount_thb": 5520}, "", None)

    responses = [
        AIMessage(content="", tool_calls=[
            {"name": "get_user_context", "args": {},
             "id": "tc-1", "type": "tool_call"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "run_python",
             "args": {"code": "result = compare_periods(period1_start=s, period1_end=e)"},
             "id": "tc-2", "type": "tool_call"},
        ]),
        # Give-up apology — the buggy behavior the guard must reject.
        AIMessage(content="ขออภัยครับ ตอนนี้ดึงข้อมูลไม่สำเร็จ ลองใหม่อีกครั้งนะครับ"),
        # Forced retry — corrected call.
        AIMessage(content="", tool_calls=[
            {"name": "run_python",
             "args": {"code": "result = spending_trend(start=s, end=e, group_by='month')"},
             "id": "tc-3", "type": "tool_call"},
        ]),
        AIMessage(content="เทรนด์ล่าสุดอยู่ที่ 5,520 บาทครับ"),
    ]
    model = FakeToolingModel(responses=responses)

    async def run():
        with patch.object(codeact_mod, "execute", fake_execute):
            graph = await _build_test_graph(model, react_impl="prebuilt")
            out = await graph.ainvoke(
                {"user_id": "u-1", "thread_id": "t-G05",
                 "messages": [HumanMessage("เทียบค่าใช้จ่าย 3 เดือน")]},
                config={"configurable": {"thread_id": "t-G05"},
                        "recursion_limit": 25},
            )

        # 1. Guard fired EXACTLY once.
        assert out.get("__tool_error_retries__") == 1, (
            f"expected 1 tool-error retry, got {out.get('__tool_error_retries__')}"
        )
        # 2. Both run_python calls happened (the retry was real, not skipped).
        assert calls["n"] == 2, f"expected 2 sandbox calls, got {calls['n']}"
        # 3. Final answer is the grounded recovery, NOT the apology.
        ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
        final_text = ai_messages[-1].content
        assert "5,520" in final_text
        assert "ขออภัย" not in final_text
        # 4. The failed ToolMessage carries the corrective marker — proof the
        # guard rewrote it to route back (not a lucky re-emit).
        assert any(
            "[TOOL_ERROR_RETRY" in (getattr(m, "content", "") or "")
            for m in out["messages"]
        ), "expected a ToolMessage carrying the [TOOL_ERROR_RETRY marker"

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-G06 — guard gives up after MAX retries → R10 Thai fallback stands
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_G06_tool_error_guard_lets_fallback_stand_after_max_retries():
    """UT-G06: when every retry still errors, the guard MUST stop forcing
    loop-backs after _TOOL_ERROR_RETRY_MAX (=2) and let the model's Thai R10
    fallback be the final answer — not loop forever or crash.

    Script: run_python errors on EVERY call; the model apologises after each.
    Guard fires on the 1st and 2nd apology, then on the 3rd it lets the
    apology through.
    """
    _seed_catalog()

    import importlib
    codeact_mod = importlib.import_module("src.agent.tools.codeact")

    def fake_execute(code, namespace):
        return (None, "", "ValueError: boom")

    apology = "ขออภัยครับ ตอนนี้ดึงข้อมูลไม่สำเร็จ ลองใหม่อีกครั้งนะครับ"

    # Each run_python AIMessage needs a unique tool_call id or add_messages
    # would collide — build fresh ones per attempt.
    def _rp(idx: int) -> AIMessage:
        return AIMessage(content="", tool_calls=[
            {"name": "run_python", "args": {"code": "result = boom()"},
             "id": f"tc-rp{idx}", "type": "tool_call"},
        ])

    responses = [
        _rp(1), AIMessage(content=apology),   # attempt 1 → guard retry (1)
        _rp(2), AIMessage(content=apology),   # attempt 2 → guard retry (2)
        _rp(3), AIMessage(content=apology),   # attempt 3 → guard exhausted → stands
    ]
    model = FakeToolingModel(responses=responses)

    async def run():
        with patch.object(codeact_mod, "execute", fake_execute):
            graph = await _build_test_graph(model, react_impl="prebuilt")
            out = await graph.ainvoke(
                {"user_id": "u-1", "thread_id": "t-G06",
                 "messages": [HumanMessage("เทียบค่าใช้จ่าย 3 เดือน")]},
                config={"configurable": {"thread_id": "t-G06"},
                        "recursion_limit": 25},
            )

        # Guard capped at MAX (2) — did not loop forever.
        assert out.get("__tool_error_retries__") == 2, (
            f"expected guard to stop at 2, got {out.get('__tool_error_retries__')}"
        )
        # Final answer is the Thai R10 fallback.
        ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
        assert "ขออภัย" in ai_messages[-1].content
        # Did NOT escalate to a validator failure (apology has no numbers).
        assert out.get("__validator_failed__", False) is False

    asyncio.run(run())


