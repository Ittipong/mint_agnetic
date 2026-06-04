"""`get_app_capability` — lookup what the chat can / can't do.

NEW in Wave 5 (Advisor expansion).

The app-capability FAQ used to live inline in the system prompt's MODE
GUIDE "Chit-chat / app help" section (~25 lines). It only matters when
the user actually asks about capabilities (~5% of turns), so we moved
it behind a tool. Topics are intentionally narrow Literals so the LLM
maps fuzzy questions ("ลบรายการได้มั้ย", "ทำซ้ำทุกเดือนได้มั้ย") onto
a canonical key.

Returned shape per topic:
  {
    "topic_label": str,
    "chat_can_do": list[str],     # what to tell the user
    "chat_cannot_do": list[str],  # what to tell the user
    "where_to_do_it": str,        # in-app screen if not in chat
    "phrase_examples": list[str], # things user can type that DO work
    "redirect_text": str | None,  # ready-to-use Thai redirect message
  }

R1 (NUMBERS-FROM-TOOLS-ONLY) is irrelevant here — this tool returns no
monetary values. Safe to call from chit-chat / Q&A turns without R9 pre-tool.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog


_STATUS_WORD = "กำลังเช็คฟีเจอร์..."


# ── Capability content ────────────────────────────────────────────────────

CAPABILITIES: dict[str, dict] = {
    "edit_confirmed_txn": {
        "topic_label": "แก้ไขรายการที่ confirm แล้ว",
        "chat_can_do": [
            "แก้รายการที่ยัง **pending** (ยังไม่กดยืนยัน) ได้ในเทิร์นเดียวกัน "
            "เช่น เพิ่ง propose แล้วพิมพ์ 'แก้เป็น 200 ค่าอาหาร'",
        ],
        "chat_cannot_do": [
            "แก้รายการที่กดยืนยันไปแล้วในแชท",
        ],
        "where_to_do_it": "หน้า transaction list — กดที่รายการนั้นเพื่อแก้ไข",
        "phrase_examples": ["แก้เป็น 200", "เปลี่ยนหมวดเป็นอาหาร"],
        "redirect_text": (
            "รายการที่ confirm แล้วต้องไปแก้ที่หน้ารายการ (transaction list) "
            "กดที่รายการนั้นได้เลย แชทนี้แก้ได้แค่ตอนยัง pending"
        ),
    },

    "delete_confirmed_txn": {
        "topic_label": "ลบรายการที่ confirm แล้ว",
        "chat_can_do": [
            "ยกเลิกรายการที่ยัง pending ได้ — แค่บอก 'ยกเลิก' หรือ propose ใหม่",
        ],
        "chat_cannot_do": [
            "ลบรายการที่ confirm แล้วในแชท",
        ],
        "where_to_do_it": "หน้า transaction list — ปัดซ้ายที่รายการ หรือกดเข้าไปเพื่อลบ",
        "phrase_examples": ["ยกเลิก", "ไม่เอาแล้ว"],
        "redirect_text": (
            "ลบรายการที่ confirm แล้วในแชทไม่ได้นะ — ไปที่หน้ารายการ "
            "(transaction list) ปัดซ้ายหรือกดเข้าไปลบได้เลย"
        ),
    },

    "backdate": {
        "topic_label": "บันทึกย้อนหลัง",
        "chat_can_do": [
            "บันทึกย้อนหลังได้ — บอกวันที่ในประโยคได้เลย",
            "ใช้คำธรรมชาติ: 'เมื่อวาน', 'อาทิตย์ที่แล้ว', '2 วันก่อน', "
            "หรือวันที่เฉพาะ '15 พ.ค.'",
        ],
        "chat_cannot_do": [],
        "where_to_do_it": "ทำในแชทได้เลย",
        "phrase_examples": [
            "เพิ่ม 80 ค่าน้ำ เมื่อวาน",
            "จ่ายค่าไฟ 1200 อาทิตย์ที่แล้ว",
            "ซื้อกาแฟ 50 วันที่ 15 พ.ค.",
        ],
        "redirect_text": None,
    },

    "manage_wallet": {
        "topic_label": "จัดการกระเป๋า (สร้าง / แก้ / ลบ)",
        "chat_can_do": [
            "ดูรายชื่อกระเป๋าได้ — ถามว่า 'กระเป๋ามีอะไรบ้าง'",
        ],
        "chat_cannot_do": [
            "สร้าง / แก้ไข / ลบกระเป๋าในแชท",
        ],
        "where_to_do_it": "เมนูในแอป → กระเป๋าเงิน",
        "phrase_examples": ["กระเป๋ามีอะไรบ้าง", "ยอดในแต่ละกระเป๋า"],
        "redirect_text": (
            "สร้าง/แก้กระเป๋าทำในแชทไม่ได้นะ ไปที่เมนูกระเป๋าเงินในแอปได้เลย "
            "แต่ถามยอดในกระเป๋าผ่านแชทได้"
        ),
    },

    "manage_budget": {
        "topic_label": "จัดการ Budget (ตั้ง / แก้ / ลบ)",
        "chat_can_do": [
            "ดู budget ที่เหลือ / รายการที่ใช้กับ budget นี้",
            "ถามว่า 'งบเดือนนี้เหลือเท่าไร' / 'รายการในงบอาหาร'",
        ],
        "chat_cannot_do": [
            "สร้าง / แก้ / ลบ budget ในแชท",
        ],
        "where_to_do_it": "เมนูในแอป → Budget",
        "phrase_examples": ["งบเหลือเท่าไร", "ใช้งบอาหารไปกี่บาทแล้ว"],
        "redirect_text": (
            "ตั้ง/แก้ budget ต้องไปที่เมนู Budget ในแอป แต่ถามยอดที่เหลือ "
            "หรือดูรายการในงบผ่านแชทได้นะ"
        ),
    },

    "manage_goal": {
        "topic_label": "จัดการ Goal (ตั้ง / แก้ / ลบ)",
        "chat_can_do": [
            "ดูความคืบหน้าเป้าหมาย — 'เป้าออมเงินไปถึงไหนแล้ว'",
            "ดูรายการที่เก็บเข้าเป้า",
        ],
        "chat_cannot_do": [
            "สร้าง / แก้ / ลบ goal ในแชท",
        ],
        "where_to_do_it": "เมนูในแอป → เป้าหมาย",
        "phrase_examples": [
            "เป้าทริปญี่ปุ่นเก็บไปเท่าไรแล้ว",
            "เหลืออีกเท่าไรถึงเป้า",
        ],
        "redirect_text": (
            "ตั้ง/แก้เป้าหมายต้องไปที่เมนูเป้าหมายในแอป "
            "แต่ดูความคืบหน้าผ่านแชทได้"
        ),
    },

    "split_bill": {
        "topic_label": "หารบิล / แบ่งจ่าย",
        "chat_can_do": [],
        "chat_cannot_do": [
            "หารบิล (split bill) ในแชท — ฟีเจอร์นี้ยังไม่มีในแอป",
        ],
        "where_to_do_it": "ยังไม่มีในแอปตอนนี้",
        "phrase_examples": [],
        "redirect_text": (
            "ตอนนี้แอปยังไม่รองรับการหารบิลในแชทนะ ถ้าอยากบันทึกเฉพาะส่วนของตัวเอง "
            "บอกยอดที่จ่ายจริงได้เลย เช่น 'เพิ่ม 250 ค่าข้าว'"
        ),
    },

    "export_data": {
        "topic_label": "export ข้อมูล / ส่งออกรายการ",
        "chat_can_do": [
            "ดูรายการในแชทได้ (list_transactions)",
        ],
        "chat_cannot_do": [
            "export เป็นไฟล์ (CSV / Excel / PDF) จากแชท",
        ],
        "where_to_do_it": "เมนูในแอป → การตั้งค่า → ส่งออกข้อมูล (ถ้ามี)",
        "phrase_examples": ["ดูรายการเดือนนี้", "list รายการอาหารเดือนที่แล้ว"],
        "redirect_text": (
            "Export ไฟล์ทำในแชทไม่ได้ ลองดูในเมนูตั้งค่าของแอป "
            "แต่ดูรายการในแชทได้นะ"
        ),
    },

    "recurring": {
        "topic_label": "รายการประจำ / ทำซ้ำอัตโนมัติ",
        "chat_can_do": [
            "บันทึกรายการเดิมซ้ำเองทุกครั้งที่เกิดขึ้น (พิมพ์เข้ามาเหมือนเดิมได้)",
        ],
        "chat_cannot_do": [
            "ตั้งให้บันทึกอัตโนมัติทุกเดือนในแชท",
        ],
        "where_to_do_it": "เมนูในแอป → รายการประจำ (ถ้ามี)",
        "phrase_examples": [],
        "redirect_text": (
            "ตั้งรายการประจำอัตโนมัติทำในแชทไม่ได้ ลองดูในเมนูแอป "
            "ถ้าอยากบันทึกตอนนี้พิมพ์เข้ามาได้เลย"
        ),
    },

    "general": {
        "topic_label": "ภาพรวม — แชทนี้ทำอะไรได้บ้าง",
        "chat_can_do": [
            "**บันทึก** รายรับ / รายจ่าย (พิมพ์เป็นภาษาคนได้เลย)",
            "**บันทึกย้อนหลัง** (บอกวันที่ในประโยค)",
            "**วิเคราะห์ / สรุป / ตอบคำถาม** เรื่องการเงิน",
            "**แก้รายการที่ยัง pending** ในเทิร์นเดียวกัน",
            "ช่วย**วางแผน** ซื้อบ้าน/รถ/แก้หนี้/ออม/ลงทุน (advisor mode)",
            "**รับฟัง**ตอนเครียดเรื่องเงิน",
        ],
        "chat_cannot_do": [
            "แก้/ลบรายการที่ confirm แล้ว",
            "สร้าง/แก้/ลบ กระเป๋า/budget/goal",
            "หารบิล / export ไฟล์ / ตั้งรายการประจำอัตโนมัติ",
        ],
        "where_to_do_it": "ฟีเจอร์ที่แชททำไม่ได้ → ทำในเมนูแอป",
        "phrase_examples": [
            "เพิ่ม 50 ค่ากาแฟ",
            "เพิ่ม 80 ค่าน้ำ เมื่อวาน",
            "เดือนนี้ใช้ไปเท่าไร",
            "หนี้บัตรเยอะมาก ทำไงดี",
        ],
        "redirect_text": None,
    },
}


_VALID_TOPICS = tuple(CAPABILITIES.keys())


@tool
async def get_app_capability(
    topic: Literal[
        "edit_confirmed_txn", "delete_confirmed_txn", "backdate",
        "manage_wallet", "manage_budget", "manage_goal",
        "split_bill", "export_data", "recurring", "general",
    ],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """Look up what the Mint Money chat can vs. can't do for a topic.

    Call when the user asks about app features: "ทำ X ในแชทได้มั้ย",
    "ลบรายการได้ไหม", "บันทึกย้อนหลังยังไง", "หารบิลได้ป่าว",
    "แอปนี้ทำอะไรได้บ้าง" (use "general"), etc.

    Args:
      topic: closest matching capability bucket. Use "general" when the
             user asks broadly about the app.

    Returns: ToolMessage with JSON of {
      topic, topic_label, chat_can_do, chat_cannot_do, where_to_do_it,
      phrase_examples, redirect_text
    }
    """
    _emit_status(_STATUS_WORD)

    capability = CAPABILITIES.get(topic)
    if capability is None:
        slog("get_app_capability", f"unknown topic: {topic!r}")
        return ToolMessage(
            content=json.dumps({
                "error": f"unknown topic {topic!r}",
                "valid_topics": list(_VALID_TOPICS),
            }),
            tool_call_id=tool_call_id,
        )

    slog("get_app_capability", f"loaded topic={topic}")
    return ToolMessage(
        content=json.dumps({"topic": topic, **capability}, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


__all__ = ["get_app_capability", "CAPABILITIES"]
