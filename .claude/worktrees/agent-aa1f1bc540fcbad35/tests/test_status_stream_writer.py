"""Unit tests for per-tool `stream_writer` status_token emission.

Covers UT-SW01..UT-SW02.

Goal:
  Every block-emitting / data tool MUST push exactly one `{"status": ...}`
  payload via `langgraph.config.get_stream_writer` at entry. The SSE adapter
  then forwards this as `event: status_token`.

  - UT-SW01a..g: each of the 7 tools fires exactly ONE status push per
    single invocation.
  - UT-SW02: when a tool is invoked TWICE in the same turn (e.g. two
    `run_python` calls during an advisor flow), TWO status pushes are
    emitted — one per invocation. We do NOT collapse repeated status
    tokens server-side; the mobile cubit handles the visual coalescing.

Tool invocation discipline (matches existing v3 tool tests):
  - Tools that declare `InjectedToolCallId` (propose_transaction,
    get_user_context, wallet_required_cta, emit_suggestions, run_python)
    MUST be invoked via the full ToolCall envelope: `{"name", "args",
    "type": "tool_call", "id"}`. `state` rides inside `args` because the
    direct .ainvoke path bypasses LangGraph's InjectedState machinery.
  - Tools without InjectedToolCallId (memory_recall, memory_write) use
    plain kwarg dict invocation.

Mocking strategy:
  Patch `langgraph.config.get_stream_writer` at each tool's import path
  (`src.agent.tools.codeact.get_stream_writer` etc) → a factory that
  returns a list-appending recorder. Assert the recorder captured the
  expected status word exactly once.

Spec sources:
  - Tool status words: `docs/mint_agentic_v3_pure_react_spec.html` §5.2 +
    `docs/v3/phase2_tools_design.md` per-tool Status sections.
  - The v3 spec lists run_python status as "กำลังคำนวน..." but the code
    in `tools/codeact/__init__.py::_STATUS_WORD` is "กำลังคำนวณ..." (ณ).
    Code is the source of truth; test asserts what the tool actually emits.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from langgraph.store.memory import InMemoryStore

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
    set_catalog_loader,
)


# ---------------------------------------------------------------------------
# Recorder writer factory
# ---------------------------------------------------------------------------


def _make_writer_recorder() -> tuple[Any, list[dict]]:
    """Return `(get_writer_factory, calls)`.

    `get_writer_factory` is callable-shaped like `get_stream_writer()` —
    each call returns the SAME `writer(payload)` that appends to `calls`.
    Tools internally do `writer = get_stream_writer(); writer({"status": ...})`
    so this recorder captures one entry per status push.
    """
    calls: list[dict] = []

    def writer(payload: Any) -> None:
        calls.append(payload)

    def factory() -> Any:
        return writer

    return factory, calls


# Tool registry — module path the tool imports `get_stream_writer` from.
_TOOL_WRITER_PATHS = {
    "propose_transaction": "src.agent.tools.propose_transaction.get_stream_writer",
    "get_user_context": "src.agent.tools.get_user_context.get_stream_writer",
    "wallet_required_cta": "src.agent.tools.wallet_required_cta.get_stream_writer",
    "emit_suggestions": "src.agent.tools.emit_suggestions.get_stream_writer",
    "run_python": "src.agent.tools.codeact.get_stream_writer",
    "memory_recall": "src.agent.tools.memory_tools.get_stream_writer",
    "memory_write": "src.agent.tools.memory_tools.get_stream_writer",
}


# Expected status word per tool — code is the source of truth.
_EXPECTED_STATUS = {
    "propose_transaction": "กำลังบันทึก...",
    "get_user_context": "กำลังเช็คข้อมูล...",
    "wallet_required_cta": "กำลังเช็คกระเป๋า...",
    "emit_suggestions": "กำลังคิดคำถามต่อ...",
    "run_python": "กำลังคำนวณ...",
    "memory_recall": "กำลังนึกย้อน...",
    "memory_write": "กำลังจดจำ...",
}


# ---------------------------------------------------------------------------
# Helpers — minimal fixtures (state, catalog, store)
# ---------------------------------------------------------------------------


def _seeded_user_context() -> dict:
    """Minimal `user_context` so propose_transaction proceeds past its
    context guard. Status emission still happens at entry regardless."""
    return {
        "wallets": [
            {
                "sync_id": "w-cash",
                "name": "เงินสด",
                "currency": "THB",
                "wallet_type": "general",
                "is_default": True,
                "usage_count": 0,
                "icon": None,
            }
        ],
        "categories": [
            {
                "sync_id": "c-coffee",
                "name": "กาแฟ",
                "type": "expense",
                "parent_id": None,
                "usage_count": 0,
                "wallet_sync_id": "w-cash",
            }
        ],
        "budgets": [],
        "goals": [],
        "tags": [],
        "wallet_id": "w-cash",
        "default_currency_code": "THB",
        "fetched_at": "2026-05-29T00:00:00+00:00",
    }


def _store() -> InMemoryStore:
    """Real LangGraph InMemoryStore — pydantic validates the store kwarg
    against `BaseStore`, so a duck-typed stub fails Tool validation. The
    real store is cheap; both methods complete locally and status emission
    happens BEFORE the store hop regardless."""
    return InMemoryStore()


def _install_test_catalog() -> None:
    """Swap entity_catalog loader so get_user_context returns offline."""
    catalog = EntityCatalog(
        wallets=[
            WalletEntry(
                sync_id="w-cash",
                name="เงินสด",
                currency="THB",
                wallet_type="general",
                is_default=True,
                usage_count=0,
                icon=None,
            )
        ],
        categories=[
            CategoryEntry(
                sync_id="c-coffee",
                name="กาแฟ",
                type="expense",
                parent_id=None,
                usage_count=0,
                wallet_sync_id="w-cash",
            )
        ],
        tags=[],
    )

    async def fake_loader(user_id: str) -> EntityCatalog:
        return catalog

    set_catalog_loader(fake_loader)


def _tool_call(name: str, args: dict, call_id: str = "tc-1") -> dict:
    """Build a full LangChain ToolCall envelope (required for tools with
    InjectedToolCallId)."""
    return {"name": name, "args": args, "type": "tool_call", "id": call_id}


# ---------------------------------------------------------------------------
# UT-SW01a — propose_transaction emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01a_propose_transaction_emits_status_once() -> None:
    """UT-SW01a: a single `propose_transaction` invocation pushes exactly
    one `{"status": "กำลังบันทึก..."}` via stream_writer."""
    from src.agent.tools.propose_transaction import propose_transaction

    factory, calls = _make_writer_recorder()
    state = {
        "user_id": "u-1",
        "thread_id": "t-1",
        "user_context": _seeded_user_context(),
        "proposals": [],
        "pending_proposal": None,
        "emitted_blocks_this_turn": [],
    }

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["propose_transaction"], factory):
            await propose_transaction.ainvoke(_tool_call(
                "propose_transaction",
                {
                    "amount": 250.0,
                    "type": "expense",
                    "category_label": "กาแฟ",
                    "state": state,
                },
                "call-1",
            ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["propose_transaction"]


# ---------------------------------------------------------------------------
# UT-SW01b — get_user_context emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01b_get_user_context_emits_status_once() -> None:
    """UT-SW01b: a single `get_user_context` invocation pushes exactly one
    `{"status": "กำลังเช็คข้อมูล..."}` BEFORE the catalog load."""
    _install_test_catalog()
    from src.agent.tools.get_user_context import get_user_context

    factory, calls = _make_writer_recorder()
    state = {"user_id": "u-1", "thread_id": "t-1"}

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["get_user_context"], factory):
            await get_user_context.ainvoke(_tool_call(
                "get_user_context", {"state": state}, "call-2",
            ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["get_user_context"]


# ---------------------------------------------------------------------------
# UT-SW01c — wallet_required_cta emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01c_wallet_required_cta_emits_status_once() -> None:
    """UT-SW01c: one `wallet_required_cta` invocation pushes exactly one
    `{"status": "กำลังเช็คกระเป๋า..."}`."""
    from src.agent.tools.wallet_required_cta import wallet_required_cta

    factory, calls = _make_writer_recorder()
    state = {"user_id": "u-1", "emitted_blocks_this_turn": []}

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["wallet_required_cta"], factory):
            await wallet_required_cta.ainvoke(_tool_call(
                "wallet_required_cta", {"state": state}, "call-3",
            ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["wallet_required_cta"]


# ---------------------------------------------------------------------------
# UT-SW01d — emit_suggestions emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01d_emit_suggestions_emits_status_once() -> None:
    """UT-SW01d: one `emit_suggestions` invocation pushes exactly one
    `{"status": "กำลังคิดคำถามต่อ..."}` BEFORE the cleanup hop."""
    from src.agent.tools.emit_suggestions import emit_suggestions

    factory, calls = _make_writer_recorder()
    state = {"emitted_blocks_this_turn": []}

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["emit_suggestions"], factory):
            await emit_suggestions.ainvoke(_tool_call(
                "emit_suggestions",
                {
                    "items": ["ตั้ง budget เดือนนี้", "ดูสรุป 3 เดือน"],
                    "state": state,
                },
                "call-4",
            ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["emit_suggestions"]


# ---------------------------------------------------------------------------
# UT-SW01e — run_python emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01e_run_python_emits_status_once() -> None:
    """UT-SW01e: one `run_python` invocation pushes exactly one
    `{"status": "กำลังคำนวณ..."}` BEFORE the sandbox executes.

    We exit early via the missing-context guard (state.user_context is
    None) — status MUST still have been emitted at entry."""
    from src.agent.tools.codeact import run_python

    factory, calls = _make_writer_recorder()
    state = {"user_id": "u-1", "user_context": None}

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["run_python"], factory):
            await run_python.ainvoke(_tool_call(
                "run_python",
                {"code": "result = 1 + 1", "state": state},
                "call-5",
            ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["run_python"]


# ---------------------------------------------------------------------------
# UT-SW01f — memory_recall emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01f_memory_recall_emits_status_once() -> None:
    """UT-SW01f: one `memory_recall` invocation pushes exactly one
    `{"status": "กำลังนึกย้อน..."}`. memory_recall has NO
    InjectedToolCallId so direct kwarg invocation works."""
    from src.agent.tools.memory_tools import memory_recall

    factory, calls = _make_writer_recorder()

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["memory_recall"], factory):
            await memory_recall.ainvoke({
                "topic": "debt",
                "k": 3,
                "state": {"user_id": "u-1"},
                "store": _store(),
            })

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["memory_recall"]


# ---------------------------------------------------------------------------
# UT-SW01g — memory_write emits exactly one status_token
# ---------------------------------------------------------------------------


def test_UT_SW01g_memory_write_emits_status_once() -> None:
    """UT-SW01g: one `memory_write` invocation pushes exactly one
    `{"status": "กำลังจดจำ..."}`."""
    from src.agent.tools.memory_tools import memory_write

    factory, calls = _make_writer_recorder()

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["memory_write"], factory):
            await memory_write.ainvoke({
                "note": "user prefers conservative debt-paydown plan",
                "tags": ["debt", "preference"],
                "state": {"user_id": "u-1"},
                "store": _store(),
            })

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 1
    assert statuses[0]["status"] == _EXPECTED_STATUS["memory_write"]


# ---------------------------------------------------------------------------
# UT-SW02 — multiple invocations emit one status per call (no collapse)
# ---------------------------------------------------------------------------


def test_UT_SW02_repeat_run_python_calls_emit_one_status_each() -> None:
    """UT-SW02: invoking `run_python` THREE times in the same patch context
    emits THREE status pushes — one per invocation. The server does NOT
    collapse repeated status tokens; mobile cubit handles visual coalescing.

    Inverse of "double-emit per single call" — exactly one push PER
    invocation, but invocations multiply naturally during an advisor flow
    (e.g. baseline + current + anomaly run_python queries)."""
    from src.agent.tools.codeact import run_python

    factory, calls = _make_writer_recorder()

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["run_python"], factory):
            for n in range(3):
                state = {"user_id": "u-1", "user_context": None}
                await run_python.ainvoke(_tool_call(
                    "run_python",
                    {"code": f"result = {n}", "state": state},
                    f"call-{n}",
                ))

    asyncio.run(_run())
    statuses = [c for c in calls if isinstance(c, dict) and "status" in c]
    assert len(statuses) == 3
    assert all(s["status"] == _EXPECTED_STATUS["run_python"] for s in statuses)


# ---------------------------------------------------------------------------
# UT-SW02b — two distinct tools both emit one status each
# ---------------------------------------------------------------------------


def test_UT_SW02b_two_distinct_tools_emit_distinct_statuses() -> None:
    """UT-SW02b: a turn that fires `get_user_context` then
    `wallet_required_cta` records TWO status pushes — first
    "กำลังเช็คข้อมูล..." then "กำลังเช็คกระเป๋า...". Different status
    words, in invocation order. Guards against a future refactor that
    might inadvertently share a writer reference and double-push."""
    _install_test_catalog()
    from src.agent.tools.get_user_context import get_user_context
    from src.agent.tools.wallet_required_cta import wallet_required_cta

    factory, calls = _make_writer_recorder()

    async def _run() -> None:
        with patch(_TOOL_WRITER_PATHS["get_user_context"], factory), patch(
            _TOOL_WRITER_PATHS["wallet_required_cta"], factory
        ):
            await get_user_context.ainvoke(_tool_call(
                "get_user_context",
                {"state": {"user_id": "u-1", "thread_id": "t-1"}},
                "call-a",
            ))
            await wallet_required_cta.ainvoke(_tool_call(
                "wallet_required_cta",
                {"state": {"user_id": "u-1", "emitted_blocks_this_turn": []}},
                "call-b",
            ))

    asyncio.run(_run())
    statuses = [c["status"] for c in calls if isinstance(c, dict) and "status" in c]
    assert statuses == [
        _EXPECTED_STATUS["get_user_context"],
        _EXPECTED_STATUS["wallet_required_cta"],
    ]
