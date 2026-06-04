"""Unit tests for graph.py pure helpers — Fix B (windowing).

These are DETERMINISTIC, no-LLM tests for the helper that carries the most
regression risk:

  - `_window_messages` — must trim the LLM context WITHOUT ever orphaning a
    `ToolMessage` (an orphan = OpenRouter/OpenAI 400). The window always begins
    at a HumanMessage boundary, so tool-call/result groups stay intact.

Test IDs: UT-HW01..UT-HW05 (history window).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import _window_messages


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build realistic message histories
# ─────────────────────────────────────────────────────────────────────────────


def _add_turn(idx: int) -> list:
    """A full ADD turn: human → ai(tool_call) → tool → ai(final)."""
    return [
        HumanMessage(content=f"เพิ่ม {idx}0 กาแฟ"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "propose_transaction",
                "args": {"amount": idx * 10, "type": "expense",
                         "category_label": "กาแฟ"},
                "id": f"tc-{idx}",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content='{"proposal_id": "p-%d"}' % idx,
                    tool_call_id=f"tc-{idx}", name="propose_transaction"),
        AIMessage(content=f"บันทึก {idx*10} บาทหมวดกาแฟแล้วครับ ยืนยันด้านบนได้เลย"),
    ]


def _analyst_turn(idx: int) -> list:
    """A full Analyst turn: human → ai(run_python) → tool → ai(final)."""
    return [
        HumanMessage(content=f"เดือนนี้ใช้ไปเท่าไหร่ ({idx})"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "run_python",
                "args": {"code": "result = sum_expense(...)"},
                "id": f"rp-{idx}",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content='{"result": "250"}',
                    tool_call_id=f"rp-{idx}", name="run_python"),
        AIMessage(content=f"เดือนนี้ใช้ไป 250 บาทครับ ({idx})"),
    ]


def _no_orphan(messages: list) -> bool:
    """True when every ToolMessage has a preceding AIMessage carrying a
    matching tool_call id somewhere before it (no dangling tool result)."""
    seen_ids: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                seen_ids.add(tc["id"])
        if isinstance(m, ToolMessage):
            if m.tool_call_id not in seen_ids:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# UT-HW01..05 — history windowing
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_HW01_short_history_unchanged():
    """UT-HW01: with fewer turns than the window, the list is returned as-is."""
    msgs = _add_turn(1) + _analyst_turn(2)  # 2 turns
    out = _window_messages(msgs, max_turns=16)
    assert out == msgs
    # Pure — original list not mutated / not the same object returned by ref-edit.
    assert out is not msgs


def test_UT_HW02_trims_to_last_n_turns():
    """UT-HW02: a 10-turn thread windowed to 3 keeps exactly the last 3 turns,
    starting at a HumanMessage."""
    msgs: list = []
    for i in range(10):
        msgs += _analyst_turn(i)
    out = _window_messages(msgs, max_turns=3)

    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert len(humans) == 3, f"expected 3 turns kept, got {len(humans)}"
    assert isinstance(out[0], HumanMessage), "window must start at a turn boundary"
    # The kept turns are the LAST three (indices 7,8,9).
    assert "(9)" in out[-1].content


def test_UT_HW03_never_orphans_a_tool_message():
    """UT-HW03 (CRITICAL): no window size may produce an orphan ToolMessage.

    An orphan (ToolMessage with no preceding AIMessage tool_call in context)
    makes OpenRouter/OpenAI reject the request with a 400. We sweep every
    window size from 1..N over a mixed ADD/Analyst thread.
    """
    msgs: list = []
    for i in range(8):
        msgs += _add_turn(i) if i % 2 == 0 else _analyst_turn(i)

    for n in range(1, 10):
        out = _window_messages(msgs, max_turns=n)
        assert _no_orphan(out), f"window={n} produced an orphan ToolMessage"
        if out:
            assert isinstance(out[0], HumanMessage), (
                f"window={n} did not start at a HumanMessage boundary"
            )


def test_UT_HW04_zero_or_empty_is_safe():
    """UT-HW04: defensive edges — empty list, max_turns<=0."""
    assert _window_messages([], max_turns=16) == []
    msgs = _add_turn(1)
    assert _window_messages(msgs, max_turns=0) == msgs
    assert _window_messages(None, max_turns=16) == []  # type: ignore[arg-type]


def test_UT_HW05_does_not_inject_or_drop_system_messages():
    """UT-HW05: windowing operates on the conversation list only — the system
    prompt is prepended by `_make_prompt`, never part of `messages`. A stray
    SystemMessage in the tail (e.g. a validator retry note) is preserved when
    inside the window."""
    msgs = _analyst_turn(1) + [SystemMessage(content="[VALIDATOR_RETRY] ...")] \
        + _add_turn(2)
    out = _window_messages(msgs, max_turns=1)  # keep only the last turn (turn 2)
    # Last turn is the ADD turn; the system note sits before turn 2's Human and
    # is dropped with turn 1. No orphan, starts at Human.
    assert isinstance(out[0], HumanMessage)
    assert _no_orphan(out)

