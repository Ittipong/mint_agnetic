"""Unit tests for src.agent.tools.memory_tools.

Covers UT-T09 (recall what you wrote) + UT-T12 (caps 200 chars / 3 tags).
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.store.memory import InMemoryStore

from src.agent.tools.memory_tools import memory_recall, memory_write


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


# ─────────────────────────────────────────────────────────────────────────────
# UT-T09 — write → recall roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T09_memory_write_then_recall_returns_what_was_written(store):
    """UT-T09: after `memory_write`, a subsequent `memory_recall` for the
    same user must return the note. Roundtrip through the live
    InMemoryStore — no mocks. Exercises the real LangGraph store path."""

    async def run():
        state = {"user_id": "u-1"}

        write_out = await memory_write.ainvoke({
            "note": "ผู้ใช้ตั้งเป้าลดค่ากาแฟเดือนละ 1500",
            "tags": ["goal", "coffee"],
            "state": state,
            "store": store,
        })
        assert write_out["memory_id"].startswith("mem_")

        recall_out = await memory_recall.ainvoke({
            "topic": "coffee",
            "k": 5,
            "state": state,
            "store": store,
        })
        memories = recall_out["memories"]
        assert len(memories) == 1
        assert memories[0]["note"] == "ผู้ใช้ตั้งเป้าลดค่ากาแฟเดือนละ 1500"
        assert memories[0]["tags"] == ["goal", "coffee"]
        assert memories[0]["id"] == write_out["memory_id"]

    asyncio.run(run())


def test_UT_T09_memory_namespaced_per_user(store):
    """Memories written for user A must NOT leak to user B. Namespace is
    `("user", user_id, "facts")` — different user_id → different namespace."""

    async def run():
        await memory_write.ainvoke({
            "note": "fact for A",
            "tags": ["a"],
            "state": {"user_id": "user-A"},
            "store": store,
        })
        # User B should see nothing.
        out = await memory_recall.ainvoke({
            "topic": "anything",
            "state": {"user_id": "user-B"},
            "store": store,
        })
        assert out["memories"] == []

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-T12 — caps (200 chars, 3 tags)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_T12_note_trimmed_to_200_chars(store):
    """UT-T12: notes longer than 200 chars must be trimmed silently so
    long-term storage doesn't blow up with prose dumps from the LLM."""

    async def run():
        long_note = "x" * 500
        out = await memory_write.ainvoke({
            "note": long_note,
            "tags": ["t"],
            "state": {"user_id": "u-1"},
            "store": store,
        })
        assert "error" not in out

        recall = await memory_recall.ainvoke({
            "topic": "x",
            "state": {"user_id": "u-1"},
            "store": store,
        })
        assert len(recall["memories"]) == 1
        assert len(recall["memories"][0]["note"]) == 200

    asyncio.run(run())


def test_UT_T12_tags_capped_at_three(store):
    """UT-T12: more than 3 tags must be silently dropped to the first 3."""

    async def run():
        await memory_write.ainvoke({
            "note": "n",
            "tags": ["a", "b", "c", "d", "e"],
            "state": {"user_id": "u-1"},
            "store": store,
        })
        recall = await memory_recall.ainvoke({
            "topic": "n",
            "state": {"user_id": "u-1"},
            "store": store,
        })
        assert recall["memories"][0]["tags"] == ["a", "b", "c"]

    asyncio.run(run())


def test_UT_T12_empty_note_returns_error(store):
    """An empty / all-whitespace note must error rather than persist nothing."""

    async def run():
        out = await memory_write.ainvoke({
            "note": "   ",
            "tags": ["a"],
            "state": {"user_id": "u-1"},
            "store": store,
        })
        assert out["kind"] == "invalid_note"

    asyncio.run(run())


def test_UT_T12_recall_k_is_clamped(store):
    """k > 20 must be clamped to 20 defensively. We seed 5 facts so the
    cap doesn't matter for content; what we test is that the tool accepts
    a pathological k without crashing."""

    async def run():
        for i in range(5):
            await memory_write.ainvoke({
                "note": f"fact-{i}",
                "tags": ["t"],
                "state": {"user_id": "u-1"},
                "store": store,
            })
        out = await memory_recall.ainvoke({
            "topic": "anything",
            "k": 9999,
            "state": {"user_id": "u-1"},
            "store": store,
        })
        # All 5 returned (k=9999 clamped to 20).
        assert len(out["memories"]) == 5

    asyncio.run(run())
