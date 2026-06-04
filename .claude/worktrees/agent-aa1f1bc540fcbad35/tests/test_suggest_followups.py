"""Unit tests for the follow-up suggestion generator.

Covers UT-SG01..UT-SG12 — `src/agent/suggest_followups.py`. The generator is
called by the SSE adapter after the answer streams; here we test its pure
orchestrator (`build_suggestions_block`) and helpers in isolation, with the
LLM call (`_generate`) patched so nothing hits the network.

Chips are free-form QUESTIONS the user would ask next (no fixed-slot formula,
no locked wildcard, no heuristic fallback). On LLM failure we emit nothing.
"""

from __future__ import annotations

from unittest.mock import patch

from src.agent import suggest_followups as sf
from src.agent.streaming.block_emitter import validate_block


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _gen(skip=False, items=None):
    return {"skip": skip, "reason": "t", "items": items or []}


async def _build(*, gen_return=None, **kwargs):
    """Run build_suggestions_block with `_generate` patched.

    Forces SUGGESTIONS_ENABLED=1 so the suite is robust to an ambient
    `SUGGESTIONS_ENABLED=0` in the shell env (UT-SG05 overrides it back to 0
    via its own patch.dict)."""
    async def fake_generate(**_kw):
        return gen_return

    defaults = dict(
        user_text="เดือนนี้ใช้ไปเท่าไร",
        answer_text="เดือนนี้ใช้ไป 12,500 บาทครับ หมวดอาหาร 4,200",
        tool_data="[run_python] total=12500",
        user_context={"wallets": [], "budgets": [], "goals": []},
        proposal_emitted=False,
    )
    defaults.update(kwargs)
    with patch.dict("os.environ", {"SUGGESTIONS_ENABLED": "1"}), \
            patch.object(sf, "_generate", fake_generate):
        return await sf.build_suggestions_block(**defaults)


def _sends(block):
    return [it["send"] for it in (block or {}).get("items", [])]


# ─────────────────────────────────────────────────────────────────────────────
# Gate tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_UT_SG01_add_turn_skips():
    """UT-SG01: a turn that emitted a transaction proposal is ADD → None."""
    block = await _build(proposal_emitted=True,
                         gen_return=_gen(items=[{"label": "x", "send": "x"}]))
    assert block is None


async def test_UT_SG02_crisis_lexicon_skips():
    """UT-SG02: a self-harm signal hard-skips in code (never trusts the LLM)."""
    block = await _build(
        user_text="ไม่ไหวแล้ว ไม่อยากอยู่ต่อแล้ว หนี้ท่วม",
        answer_text="ฟังแล้วเป็นห่วงมากเลย",
        gen_return=_gen(items=[{"label": "x", "send": "x"}]),
    )
    assert block is None


async def test_UT_SG03_no_answer_skips():
    """UT-SG03: no final answer text → nothing to follow up on → None."""
    block = await _build(answer_text="   ", gen_return=_gen(items=[{"label": "x", "send": "x"}]))
    assert block is None


async def test_UT_SG04_llm_skip_true_emits_nothing():
    """UT-SG04: pure-emotional / clarifying-question turn — LLM gate
    returns skip=true → None."""
    block = await _build(
        user_text="วันนี้เหนื่อยมาก",
        answer_text="เห็นใจนะ พักก่อนได้เลย",
        gen_return=_gen(skip=True),
    )
    assert block is None


async def test_UT_SG05_disabled_env_skips():
    """UT-SG05: SUGGESTIONS_ENABLED=0 disables the generator entirely."""
    async def fake_generate(**_kw):
        return _gen(items=[{"label": "x", "send": "x"}])

    with patch.dict("os.environ", {"SUGGESTIONS_ENABLED": "0"}), \
            patch.object(sf, "_generate", fake_generate):
        block = await sf.build_suggestions_block(
            user_text="เดือนนี้ใช้ไปเท่าไร",
            answer_text="ใช้ไป 12,500 บาท",
            user_context={"wallets": [], "budgets": [], "goals": []},
        )
    assert block is None


# ─────────────────────────────────────────────────────────────────────────────
# Free-form assembly — pure LLM chips, no wildcard, no forced count
# ─────────────────────────────────────────────────────────────────────────────


async def test_UT_SG06_emits_llm_chips_verbatim_no_wildcard():
    """UT-SG06: LLM chips pass through unchanged — no locked wildcard is
    appended (the old action-chip mechanism is gone)."""
    block = await _build(
        gen_return=_gen(items=[
            {"label": "หมวดไหนใช้เยอะสุด", "send": "หมวดไหนใช้เยอะสุด"},
            {"label": "เทรนด์ 3 เดือน", "send": "เทรนด์ย้อนหลัง 3 เดือนเป็นยังไง"},
            {"label": "เทียบเดือนก่อน", "send": "เดือนนี้ใช้เยอะกว่าเดือนก่อนไหม"},
        ]),
    )
    sends = _sends(block)
    assert len(sends) == 3
    assert "ตั้งงบประมาณเดือนนี้" not in sends  # no auto wildcard
    assert sends == [
        "หมวดไหนใช้เยอะสุด",
        "เทรนด์ย้อนหลัง 3 เดือนเป็นยังไง",
        "เดือนนี้ใช้เยอะกว่าเดือนก่อนไหม",
    ]


async def test_UT_SG07_two_chips_stay_two_not_padded():
    """UT-SG07: LLM returns 2 relevant chips → exactly 2 emitted (we never
    pad to 3 with a filler/wildcard)."""
    block = await _build(
        gen_return=_gen(items=[
            {"label": "หมวดไหนใช้เยอะสุด", "send": "หมวดไหนใช้เยอะสุด"},
            {"label": "เทรนด์ 3 เดือน", "send": "เทรนด์ย้อนหลัง 3 เดือน"},
        ]),
    )
    assert len(_sends(block)) == 2


async def test_UT_SG08_dedup_collapses_duplicates():
    """UT-SG08: chips with the same `send` (ignoring spaces) collapse to one."""
    block = await _build(
        gen_return=_gen(items=[
            {"label": "หมวดไหนเยอะ", "send": "หมวดไหนใช้เยอะสุด"},
            {"label": "หมวดเยอะ", "send": "หมวดไหนใช้ เยอะสุด"},  # dup after strip
            {"label": "เทรนด์", "send": "เทรนด์ย้อนหลัง 3 เดือน"},
        ]),
    )
    sends = _sends(block)
    assert len(sends) == 2
    assert sends == ["หมวดไหนใช้เยอะสุด", "เทรนด์ย้อนหลัง 3 เดือน"]


async def test_UT_SG09_caps_at_three():
    """UT-SG09: 5 chips → keep first 3 (mobile renders at most 3)."""
    items = [{"label": f"c{i}", "send": f"send-{i}"} for i in range(5)]
    block = await _build(gen_return=_gen(items=items))
    sends = _sends(block)
    assert len(sends) == 3
    assert sends == ["send-0", "send-1", "send-2"]


async def test_UT_SG10_llm_failure_emits_nothing():
    """UT-SG10: LLM returns None (error/timeout) → None. No heuristic fallback
    (chips are nice-to-have, a low-relevance chip is worse than none)."""
    block = await _build(
        tool_data="[run_python] total=12500",
        gen_return=None,
    )
    assert block is None


async def test_UT_SG11_plain_string_items_normalized():
    """UT-SG11: legacy plain-string items are coerced to {label, send}."""
    block = await _build(gen_return=_gen(items=["หมวดไหนใช้เยอะสุด", "  ", "เทรนด์"]))
    items = block["items"]
    assert len(items) == 2  # blank dropped
    assert items[0] == {"label": "หมวดไหนใช้เยอะสุด", "send": "หมวดไหนใช้เยอะสุด"}


async def test_UT_SG12_emitted_block_is_valid():
    """UT-SG12: the emitted block passes the wire validator (so the SSE
    adapter forwards it instead of dropping it)."""
    block = await _build(
        gen_return=_gen(items=[{"label": "เทรนด์", "send": "เทรนด์ย้อนหลัง"}]),
    )
    assert block["type"] == "suggestions"
    assert validate_block(block), f"block failed validation: {block}"
    for it in block["items"]:
        assert it["label"] and it["send"]
