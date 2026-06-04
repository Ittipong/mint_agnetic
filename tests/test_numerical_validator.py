"""Unit tests for src.agent.validators.numerical.

Covers UT-V01 .. UT-V07 per `docs/v3/phase2_validator.md` §9.

The validator is the safety net against the 1,234.56 hallucination class
of bugs (memory `project_codeact_dual_impl`) — every test here asserts
EXPECTED CORRECT behaviour (Testing Rule #1: not what the validator
currently does, but what the SLO contract requires).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.agent.validators.numerical import (
    extract_money_numbers,
    validate_numerical_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# UT-V01 — extract_money_numbers covers Thai + English + magnitudes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("เดือนนี้ใช้ไป 12,500 บาท", [Decimal("12500")]),
    ("รายได้ 32k", [Decimal("32000")]),
    ("ออม 1.5 ล้าน", [Decimal("1500000")]),
    (
        "หนี้ 42,500 รายได้ 32,000 ใช้ 30,800",
        [Decimal("42500"), Decimal("32000"), Decimal("30800")],
    ),
    ("เริ่มที่ 2,500 บาท แล้วค่อยขยับ", [Decimal("2500")]),
    ("ไม่มีตัวเลข", []),
    ("฿50,000", [Decimal("50000")]),
    ("ราว 30k-50k", [Decimal("30000"), Decimal("50000")]),
])
def test_UT_V01_extract_money_numbers(text, expected):
    """UT-V01: extract_money_numbers covers Thai + English + magnitudes."""
    assert extract_money_numbers(text) == expected


# ─────────────────────────────────────────────────────────────────────────────
# UT-V02 .. UT-V07 — validate_numerical_response cross-check
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_V02_pass_within_tolerance():
    """UT-V02: 1% tolerance passes for rounding-for-display (12,500 vs 12,499.50)."""
    result = validate_numerical_response(
        "ใช้ไป 12,500 บาท",
        ["{'amount_thb': 12499.50}"],
    )
    assert result.pass_, result.reason


def test_UT_V03_fail_outside_tolerance():
    """UT-V03: > 1% diff fails (13,000 vs 12,500 = 3.8% — outside tolerance)."""
    result = validate_numerical_response(
        "ใช้ไป 13,000 บาท",
        ["{'amount_thb': 12500}"],
    )
    assert not result.pass_
    assert result.offending_number == Decimal("13000")
    assert result.reason == "number_not_in_tool_outputs"


def test_UT_V04_no_numbers_passes():
    """UT-V04: answer with no money numbers always passes — empathetic
    EMOTIONAL replies + chitchat must not trigger the validator."""
    result = validate_numerical_response("สวัสดีครับ", [])
    assert result.pass_
    assert result.reason == "no_numbers_in_answer"


def test_UT_V05_numbers_no_tool_call_fails():
    """UT-V05: numbers in answer but no tool was called -> fail.
    Forces a retry; the LLM must call run_python to ground the number."""
    result = validate_numerical_response("ใช้ไป 12,500 บาท", [])
    assert not result.pass_
    assert result.reason == "no_tool_numbers_to_validate_against"


def test_UT_V06_magnitude_normalization():
    """UT-V06: '1.5 ล้าน' in answer matches tool output '1500000'."""
    result = validate_numerical_response(
        "ออม 1.5 ล้านบาท",
        ["1500000"],
    )
    assert result.pass_, result.reason


def test_UT_V07_user_amount_passthrough():
    """UT-V07: user typed '250' -> propose_transaction returned 250 ->
    answer says '250 บาท' (ADD flow). Validator must pass via the tool
    return dict carrying 250. Don't add a hardcoded skip (would weaken
    the safety net per phase2_validator.md §10.5)."""
    result = validate_numerical_response(
        "บันทึก 250 บาทแล้วครับ",
        ["{'transaction_sync_id': 'tx1', 'amount': 250}"],
    )
    assert result.pass_, result.reason


# ─────────────────────────────────────────────────────────────────────────────
# UT-V08 — a numerical mismatch SOFT-WARNS, never rejects
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_V08_mismatch_soft_warns_no_reject(monkeypatch):
    """UT-V08: an ungrounded number must NOT reject the answer. The hook
    appends a warning footer to the SAME answer AIMessage (id preserved →
    add_messages updates in place), does NOT set `__validator_failed__`, and
    does NOT push a `validator_failed` custom event. The user always receives
    the full streamed answer plus a 'numbers may be off' note."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    import langgraph.config as lg_config
    from src.agent.validators.numerical import (
        _WARNING_MARKER,
        make_validator_post_model_hook,
    )

    captured: list = []
    monkeypatch.setattr(lg_config, "get_stream_writer", lambda: captured.append)

    hook = make_validator_post_model_hook()
    state = {
        "messages": [
            HumanMessage(content="รถ 2 ล้าน"),
            # clean tool output (no "error" key) — grounds 99,608 only
            ToolMessage(content="{'avg_income': 99608}",
                        tool_call_id="x", name="run_python"),
            # ungrounded 400,000 (LLM mental math) → soft-warned, NOT rejected
            AIMessage(content="เงินดาวน์ 400,000 บาทนะครับ", id="ans-1"),
        ],
    }
    out = hook(state)

    # No reject path: no terminal flag, no custom event.
    assert out.get("__validator_failed__") is None
    assert captured == []
    # Answer rewritten in place (same id) with the warning footer appended.
    msgs = out["messages"]
    assert len(msgs) == 1
    assert msgs[0].id == "ans-1"
    assert msgs[0].content.startswith("เงินดาวน์ 400,000 บาทนะครับ")
    assert _WARNING_MARKER in msgs[0].content


def test_UT_V08b_soft_warn_is_idempotent():
    """The footer is appended at most once — if the hook re-enters on an
    already-warned answer it returns a no-op (no stacked footers)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from src.agent.validators.numerical import (
        _WARNING_FOOTER,
        make_validator_post_model_hook,
    )

    hook = make_validator_post_model_hook()
    state = {
        "messages": [
            HumanMessage(content="รถ 2 ล้าน"),
            ToolMessage(content="{'avg_income': 99608}",
                        tool_call_id="x", name="run_python"),
            AIMessage(content="เงินดาวน์ 400,000 บาท" + _WARNING_FOOTER, id="ans-1"),
        ],
    }
    out = hook(state)
    assert out == {}
