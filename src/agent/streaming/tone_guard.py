"""Drop an empathy opener the user did not ask for.

R12 ("EMPATHY PLACEMENT") already says: open with empathy only when the user
voices a feeling; a neutral question about debt gets the numbers first. The
model still broke it in ~1.5% of chip-chain turns (docs/qa_chip_chain_2026-09-27.md),
always on card / overspending questions — e.g. "ควรจ่ายบัตร KTC เท่าไหร่ดี" →
"ฟังแล้วเข้าใจเลยครับว่ายอด 24,111 บาท … น่ากังวลใจ". Telling a calm user they
are worried reads as judgment, so the opening paragraph is removed in code
when it projects a feeling the user never expressed.
"""

from __future__ import annotations

import re

# Feelings the user voices (R12 + P0 trigger words, plus milder ones seen in
# chains: "ตึงไป", "จะเป็นอะไรไหม"). Any of these → empathy is welcome.
_USER_FEELING = re.compile(
    r"เครียด|ท้อ|ไม่ไหว|อาย|กลัว|สิ้นหวัง|โกรธตัวเอง|อยากหนี|นอนไม่หลับ|ร้องไห้|"
    r"หมดหนทาง|หมดแรง|กังวล|หนักใจ|กดดัน|ตึง|เหนื่อย|ทุกข์|แย่|เป็นอะไรไหม|ห่วง"
)

# Opening lines that ascribe a feeling to the user. Plain "เข้าใจเลยครับ"
# (an acknowledgment) is deliberately NOT here.
_PROJECTED_FEELING = re.compile(
    r"น่ากังวล|กังวลใจ|กดดัน|ไม่ใช่ความล้มเหลว|หนักใจ|ฟังแล้วเข้าใจ|เหนื่อยแทน"
)

# An opener longer than this is carrying substance too — keep it.
_MAX_OPENER_CHARS = 220


def user_voiced_feeling(user_text: str) -> bool:
    return bool(_USER_FEELING.search(user_text or ""))


def strip_unasked_empathy(answer_head: str, user_text: str) -> str:
    """Return `answer_head` without its first paragraph when that paragraph
    projects a feeling the user did not voice. Otherwise unchanged.

    `answer_head` must contain the whole first paragraph (up to the first
    blank line). Nothing is dropped when no other paragraph follows it, so an
    answer can never be emptied.
    """
    if user_voiced_feeling(user_text):
        return answer_head
    head = answer_head.lstrip()
    first, sep, rest = head.partition("\n\n")
    if not sep or not rest.strip():
        return answer_head
    if len(first) > _MAX_OPENER_CHARS or not _PROJECTED_FEELING.search(first):
        return answer_head
    return rest.lstrip("\n")


# "กำลังคำนวณยอดให้ครับ" as the answer's first line: a progress status the
# model wrote into the final answer. The waiting UI already showed that step
# (and lists it under "คิดอยู่ N วิ"), so in the answer it is noise.
_PROGRESS_OPENER = re.compile(r"^กำลัง\S[^\n]{0,70}$")


def strip_progress_opener(answer_head: str) -> str:
    """Drop a short "กำลัง…" first paragraph when another paragraph follows."""
    head = answer_head.lstrip()
    first, sep, rest = head.partition("\n\n")
    if sep and rest.strip() and _PROGRESS_OPENER.match(first.strip()):
        return rest.lstrip("\n")
    return answer_head
