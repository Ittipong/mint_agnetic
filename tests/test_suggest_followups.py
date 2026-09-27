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


# ─────────────────────────────────────────────────────────────────────────────
# UT-SG13 — lenient JSON parse (the chip-generator reply tolerance)
#
# Regression for the production `JSONDecodeError` that silently dropped every
# chip whenever the fallback model (gemini-flash-lite via OpenRouter models[]
# routing) ignored `response_format` and returned a fenced / single-quoted /
# smart-quoted reply. `_loads_lenient` MUST recover all three shapes.
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_SG13a_loads_lenient_strips_markdown_fence():
    """A ```json fenced reply (the `char 0` failure) parses to the object."""
    out = sf._loads_lenient(
        '```json\n{"skip": false, "items": [{"label":"ก","send":"ข"}]}\n```'
    )
    assert out["skip"] is False
    assert out["items"][0]["send"] == "ข"


def test_UT_SG13b_loads_lenient_handles_single_quotes():
    """A Python-dict single-quoted reply (the production `char 1` failure)
    is recovered via the literal-eval fallback — incl. JSON `null`/`false`."""
    out = sf._loads_lenient(
        "{'skip': false, 'reason': null, "
        "'items': [{'label': 'ช้อปปิ้ง', 'send': 'เดือนนี้ใช้เท่าไหร่'}]}"
    )
    assert out["skip"] is False
    assert out["reason"] is None
    assert out["items"][0]["label"] == "ช้อปปิ้ง"


def test_UT_SG13c_loads_lenient_normalizes_smart_quotes():
    """Typographic “smart” double quotes are normalized to ASCII before parse."""
    out = sf._loads_lenient('{“skip”: false, “items”: [{“label”:“ก”,“send”:“ข”}]}')
    assert out["skip"] is False
    assert out["items"][0]["send"] == "ข"


def test_UT_SG13d_loads_lenient_drops_surrounding_prose():
    """Prose wrapped around a fenced object is discarded — only the object
    survives (models occasionally add 'Sure! Here is the JSON:')."""
    out = sf._loads_lenient(
        'Sure! Here you go:\n```json\n{"skip": true, "items": []}\n```\nHope it helps'
    )
    assert out["skip"] is True
    assert out["items"] == []


def test_UT_SG13e_loads_lenient_raises_on_garbage():
    """No recoverable object → raise (caller maps this to 'emit no chips')."""
    import pytest

    with pytest.raises((ValueError, SyntaxError)):
        sf._loads_lenient("totally not json at all")


# ─────────────────────────────────────────────────────────────────────────────
# UT-SG14..UT-SG15 — quick-reply MODE
#
# When the assistant's answer ENDS WITH a clarifying question back to the user,
# the generator no longer skips the whole turn. It returns skip=false with chips
# that are the user's likely ANSWERS (carrying the question's topic so the next
# turn routes as a continuation, not as a new ADD). These tests pin the
# orchestrator path (quick-reply items are emitted, not gated) and guard the
# prompt so the new mode can't be silently stripped.
# ─────────────────────────────────────────────────────────────────────────────


async def test_UT_SG14_quick_reply_items_emit_a_block():
    """UT-SG14: the answer was a clarifying question back; the LLM gate returns
    quick-reply answer-chips (skip=false). The orchestrator must EMIT them as a
    `suggestions` block — proving the clarifying-question turn is no longer
    blanket-skipped. `send` carries the topic (anti-ADD-misroute), not a bare
    number."""
    block = await _build(
        user_text="อยากวางแผนเก็บเงิน",
        answer_text="รายจ่ายคงที่ของคุณเดือนละเท่าไรครับ",
        gen_return=_gen(items=[
            {"label": "ประมาณ 10,000", "send": "รายจ่ายคงที่ประมาณเดือนละ 10,000 บาท"},
            {"label": "ไม่มีรายจ่ายคงที่", "send": "ไม่มีรายจ่ายคงที่ประจำ"},
            {"label": "ช่วยประมาณให้", "send": "ช่วยประมาณรายจ่ายคงที่ให้หน่อย"},
        ]),
    )
    assert block is not None
    assert block["type"] == "suggestions"
    sends = _sends(block)
    assert len(sends) == 3
    assert sends[0] == "รายจ่ายคงที่ประมาณเดือนละ 10,000 บาท"
    # Anti-misroute: no chip's `send` is a bare number (would be read as ADD).
    assert all(not s.strip().replace(",", "").replace(".", "").isdigit() for s in sends)


def test_UT_SG15_prompt_declares_quick_reply_mode():
    """UT-SG15: prompt-guard. The two-mode design (esp. MODE-QUICKREPLY and its
    anti-misroute rule) must be present in `_PROMPT_TEMPLATE` so a future edit
    can't silently revert to skipping every clarifying-question turn."""
    tmpl = sf._PROMPT_TEMPLATE
    assert "MODE-QUICKREPLY" in tmpl
    assert "MODE-NORMAL" in tmpl
    # The anti-ADD-misroute guarantee: `send` must carry the topic, not a number.
    assert "anti-misroute" in tmpl
    assert "bare number" in tmpl


def test_UT_SUG_APPONLY_action_chips_the_chat_cannot_do_are_dropped():
    """Live chips offered "เพิ่มบัตรเครดิต" / "ตั้งเป้าหมายออมเงิน" — the chat
    cannot do either (the prompt already bans them), so a tap only earns a
    redirect. They are filtered deterministically after the chip LLM."""
    from src.agent.suggest_followups import _finalize

    out = _finalize([
        {"label": "เพิ่มบัตรเครดิต", "send": "เพิ่มบัตรเครดิต"},
        {"label": "ตั้งเป้าหมายออมเงิน", "send": "ตั้งเป้าหมายออมเงิน"},
        {"label": "ลบรายการนี้", "send": "ลบรายการกาแฟ"},
        {"label": "ควรออมเดือนละเท่าไหร่", "send": "ควรออมเดือนละเท่าไหร่ดี"},
        {"label": "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่", "send": "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่"},
    ])
    assert [c["label"] for c in out] == ["ควรออมเดือนละเท่าไหร่", "ตั้งแต่ต้นเดือนใช้ไปเท่าไหร่"]


# ─────────────────────────────────────────────────────────────────────────────
# UT-SG20..SG25 — chip-chain fixes (docs/qa_chip_chain_2026-09-27.md)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_SG20_repeat_of_an_earlier_question_is_dropped():
    """UT-SG20: a 6-tap chain offered "เดือนนี้หมวดไหนใช้จ่ายเยอะที่สุด" again
    after the thread already asked it. A word-for-word repeat of an earlier
    question must be dropped (spacing differences included)."""
    out = sf._finalize(
        [
            {"kind": "deeper", "label": "หมวดไหนเยอะสุด", "send": "เดือนนี้หมวดไหน ใช้จ่ายเยอะที่สุด"},
            {"kind": "deeper", "label": "เทียบเดือนก่อน", "send": "ช้อปปิ้งเทียบเดือนก่อนเป็นยังไง"},
        ],
        asked_before=["เดือนนี้ใช้จ่ายไปเท่าไหร่", "เดือนนี้หมวดไหนใช้จ่ายเยอะที่สุด"],
    )
    assert [c["label"] for c in out] == ["เทียบเดือนก่อน"]


def test_UT_SG21_mix_keeps_one_wider_chip_and_puts_it_last():
    """UT-SG21: the owner wants drill-downs mixed with ONE sideways question.
    Extra "wider" chips are dropped and the wider one always renders last."""
    out = sf._finalize([
        {"kind": "wider", "label": "วันไหนใช้เยอะสุด", "send": "วันไหนของสัปดาห์ที่ฉันใช้เงินเยอะสุด"},
        {"kind": "deeper", "label": "ทำไมช้อปปิ้งสูง", "send": "ทำไมช้อปปิ้งเดือนนี้สูง"},
        {"kind": "wider", "label": "ร้านประจำ", "send": "ร้านที่ฉันจ่ายบ่อยที่สุดคือที่ไหน"},
        {"kind": "deeper", "label": "เทียบเดือนก่อน", "send": "เทียบกับเดือนก่อน"},
    ])
    assert [c["label"] for c in out] == ["ทำไมช้อปปิ้งสูง", "เทียบเดือนก่อน", "วันไหนใช้เยอะสุด"]


def test_UT_SG22_asks_back_detects_thai_questions_without_question_mark():
    """UT-SG22: reply chips need the real answer, so a speculative chip call is
    only reused when the answer does NOT end by asking the user back."""
    assert sf.asks_back("รายจ่ายคงที่ประมาณเท่าไรครับ?")
    assert sf.asks_back("สรุปยอดให้แล้วนะครับ\n\nอยากลองเช็คดูไหมครับ")
    assert not sf.asks_back("เดือนนี้ใช้ไป **44,399 บาท** ครับ")
    assert not sf.asks_back("")


def test_UT_SG23_asked_before_skips_current_turn_and_intent_markers():
    """UT-SG23: history for the prompt = earlier user questions only."""
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("[INTENT:wallet_created]"),
            AIMessage("a2"), HumanMessage("q2"), AIMessage("a3"), HumanMessage("current")]
    assert sf.asked_before(msgs) == ["q1", "q2"]


async def _spec_result(v):
    return v


def test_UT_SG24_speculative_chips_used_when_answer_is_a_statement():
    """UT-SG24: chips started during the answer are reused — no second LLM call."""
    import asyncio

    calls = []

    async def fake_generate(**kw):
        calls.append(kw)
        return _gen(items=[{"label": "B", "send": "B?"}])

    async def run():
        task = asyncio.ensure_future(_spec_result(_gen(items=[{"label": "A", "send": "A?"}])))
        with patch.object(sf, "_generate", fake_generate), \
             patch.dict("os.environ", {"SUGGESTIONS_ENABLED": "1"}):
            return await sf.build_suggestions_block(
                user_text="เดือนนี้ใช้ไปเท่าไร", answer_text="ใช้ไป 44,399 บาทครับ",
                speculative=task)

    block = asyncio.run(run())
    assert [c["label"] for c in block["items"]] == ["A"]
    assert calls == []


def test_UT_SG25_answer_that_asks_back_regenerates_reply_chips():
    """UT-SG25: when the answer ends with a question to the user, the
    speculative (MODE-NORMAL) chips are wrong — regenerate with the answer."""
    import asyncio

    calls = []

    async def fake_generate(**kw):
        calls.append(kw)
        return _gen(items=[{"kind": "reply", "label": "คำนวณจากข้อมูลของฉัน",
                            "send": "ช่วยคำนวณรายจ่ายคงที่จากข้อมูลของฉันให้หน่อย"}])

    async def run():
        task = asyncio.ensure_future(_spec_result(_gen(items=[{"label": "A", "send": "A?"}])))
        with patch.object(sf, "_generate", fake_generate), \
             patch.dict("os.environ", {"SUGGESTIONS_ENABLED": "1"}):
            return await sf.build_suggestions_block(
                user_text="อยากออมเพิ่ม", answer_text="รายจ่ายคงที่เดือนละเท่าไรครับ",
                speculative=task)

    block = asyncio.run(run())
    assert [c["label"] for c in block["items"]] == ["คำนวณจากข้อมูลของฉัน"]
    assert len(calls) == 1 and calls[0]["answer_text"] == "รายจ่ายคงที่เดือนละเท่าไรครับ"


def test_UT_SG26_self_flagged_repeat_and_reminder_chips_are_dropped():
    """UT-SG26: round-3 chains still re-offered questions in other words and
    offered "ตั้งแจ้งเตือนวันสรุปยอด" twice (the chat can't set reminders —
    both taps dead-ended). The LLM's own `repeats` flag and reminder chips
    are dropped in code."""
    out = sf._finalize([
        {"kind": "deeper", "repeats": "เดือนที่แล้วหมวดหมู่ไหนที่ฉันใช้จ่ายเยอะที่สุด",
         "label": "หมวดที่ใช้เยอะสุด", "send": "หมวดหมู่ที่ฉันใช้จ่ายเยอะที่สุดในเดือนที่แล้วคืออะไร"},
        {"kind": "deeper", "label": "ตั้งแจ้งเตือนวันสรุปยอด", "send": "ช่วยตั้งแจ้งเตือนวันสรุปยอดบัตรเครดิตให้ฉันหน่อย"},
        {"kind": "wider", "label": "เตือนเมื่อถึงวันสรุปยอด", "send": "เตือนเมื่อถึงวันสรุปยอด"},
        {"kind": "deeper", "repeats": "", "label": "ค่าเช่าเทียบปีก่อน", "send": "ค่าที่อยู่อาศัยเทียบกับเดือนก่อน"},
    ])
    assert [c["label"] for c in out] == ["ค่าเช่าเทียบปีก่อน"]


def test_UT_SG27_not_now_chips_are_dropped():
    """UT-SG27: "ยังไม่ต้องการตอนนี้" reached the wire in round 4 despite the
    prompt ban; tapping it earns a dead turn. Filtered in code."""
    out = sf._finalize([
        {"kind": "reply", "label": "ยังไม่ต้องการตอนนี้", "send": "ยังไม่ต้องการตอนนี้"},
        {"kind": "reply", "label": "ไว้ก่อน", "send": "ไว้ก่อนนะ"},
        {"kind": "reply", "label": "ช่วยวางแผนการโอนเงิน", "send": "ช่วยวางแผนการโอนเงินให้หน่อย"},
    ])
    assert [c["label"] for c in out] == ["ช่วยวางแผนการโอนเงิน"]


async def test_UT_SG28_chip_equal_to_the_current_question_is_dropped():
    """UT-SG28: round 7 offered "ยอดรวมเงินสดและบัญชีออมทรัพย์ของฉันตอนนี้เท่าไร"
    as a chip on the very turn that asked it."""
    q = "ยอดรวมเงินสดและบัญชีออมทรัพย์ของฉันตอนนี้เท่าไร"
    block = await _build(
        user_text=q,
        gen_return=_gen(items=[{"label": "ยอดรวม", "send": q},
                               {"label": "มาจากไหน", "send": "เงินใน KBank มาจากไหน"}]),
    )
    assert _sends(block) == ["เงินใน KBank มาจากไหน"]
