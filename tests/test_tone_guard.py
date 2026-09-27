"""UT-TG01..03 — src/agent/streaming/tone_guard.py."""

from src.agent.streaming.tone_guard import strip_unasked_empathy

_BODY = "\n\nแนะนำจ่ายเต็ม 24,111 บาทก่อน 5 ต.ค. ครับ"


def test_UT_TG01_projected_worry_without_user_feeling_is_dropped():
    head = "ฟังแล้วเข้าใจเลยครับว่ายอดนี้น่ากังวลใจ" + _BODY
    assert strip_unasked_empathy(head, "ควรจ่ายบัตรเท่าไหร่ดี") == _BODY.strip()


def test_UT_TG02_plain_acknowledgment_and_substance_openers_stay():
    ack = "เข้าใจเลยครับ การเก็บเข้าบัญชีออมเป็นจุดเริ่มต้นที่ดี" + _BODY
    assert strip_unasked_empathy(ack, "เก็บเข้าบัญชีออมปกติ") == ack
    # A long opener carries substance — never dropped.
    long = "ยอดนี้น่ากังวลใจ " + "ก" * 250 + _BODY
    assert strip_unasked_empathy(long, "ควรจ่ายเท่าไหร่") == long


def test_UT_TG03_never_empties_and_respects_voiced_feelings():
    only = "ฟังแล้วเข้าใจเลยครับว่าน่ากังวลใจ"
    assert strip_unasked_empathy(only, "ควรจ่ายเท่าไหร่") == only
    head = "ฟังแล้วเข้าใจเลยครับว่าน่ากังวลใจ" + _BODY
    assert strip_unasked_empathy(head, "เครียดมาก หนี้บัตรเยอะ") == head


def test_UT_TG04_progress_opener_is_dropped_from_the_answer():
    """UT-TG04: the answer opened with "กำลังคำนวณยอดเงินคงเหลือให้ครับ" — a step
    the waiting UI already showed (2026-09-28). Dropped; a lone line stays."""
    from src.agent.streaming.tone_guard import strip_progress_opener

    head = "กำลังคำนวณยอดเงินคงเหลือให้ครับ\n\nเงินพอถึงสิ้นเดือนแน่นอนครับ"
    assert strip_progress_opener(head) == "เงินพอถึงสิ้นเดือนแน่นอนครับ"
    assert strip_progress_opener("กำลังดูให้ครับ") == "กำลังดูให้ครับ"
    long_first = "กำลังจะบอกว่า " + "ก" * 90 + "\n\nต่อ"
    assert strip_progress_opener(long_first) == long_first
