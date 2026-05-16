"""Standalone intent classifier benchmark.

Measures accuracy and latency for ~20 realistic Thai chat inputs.
Covers: add_transaction with amount, bare nouns, follow-ups, analytics
questions, chit-chat, and adversarial cases. Calls the classifier node
directly so latency reflects ONLY the classifier's LLM call (no graph
overhead, no DB fetch, no other nodes).
"""
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.intent_classifier import classify_intent_node


# ── test cases: (user_msg, prior_ai_msg_or_None, expected_intent) ───
CASES = [
    # ─── add_transaction with explicit amount ───────────────
    ("กิน kfc 100",            None, "add_transaction"),
    ("ซื้อกาแฟ 65 บาท",        None, "add_transaction"),
    ("จ่ายค่าไฟ 1200",          None, "add_transaction"),
    ("โอน 500 ให้แม่",          None, "add_transaction"),
    ("เติมน้ำมัน 800",          None, "add_transaction"),
    ("ได้เงินเดือน 35000",      None, "add_transaction"),
    ("กาแฟ 50",                None, "add_transaction"),

    # ─── add_transaction bare noun (no amount, no question) ─
    ("กาแฟ",                   None, "add_transaction"),
    ("เที่ยว",                  None, "add_transaction"),
    ("ค่าน้ำ",                  None, "add_transaction"),
    ("น้ำมัน",                  None, "add_transaction"),

    # ─── add_transaction follow-up (bare number after AI ask) ─
    ("100", "รับทราบว่าจะบันทึก 'กาแฟ' จำนวนเงินเท่าไหร่ครับ?", "add_transaction"),
    ("200 บาท", "รับทราบว่าจะบันทึก 'เที่ยว' จำนวนเงินเท่าไหร่ครับ?", "add_transaction"),

    # ─── other: analytics ───────────────────────────────────
    ("เดือนนี้ใช้เงินเท่าไหร่",  None, "other"),
    ("ขอดูรายการอาหาร",          None, "other"),
    ("ยอด KBank เหลือเท่าไหร่",  None, "other"),
    ("หมวดไหนใช้เยอะสุด",        None, "other"),
    ("เปรียบเทียบกับเดือนที่แล้ว", None, "other"),

    # ─── other: chit-chat ───────────────────────────────────
    ("สวัสดี",                    None, "other"),
    ("ขอบคุณ",                    None, "other"),

    # ─── other: adversarial (looks like add but isn't) ──────
    ("กิน kfc อร่อยไหม",          None, "other"),       # question
    ("อยากกิน kfc",               None, "other"),       # future intent
    ("เที่ยวที่ไหนดี",            None, "other"),       # question word
    ("100",                       None, "other"),       # bare number, no context

    # ─── other: bare number with NON-question AI prior ──────
    ("100", "บันทึกค่ากาแฟ 50 บาทเรียบร้อยแล้ว ✓", "other"),
]


async def main():
    print(f"{'#':3} {'expected':>17} {'actual':>17} {'src':>9} {'ms':>5}  match  message")
    print("─" * 110)
    n_total = 0
    n_correct = 0
    durations = []
    for i, (user_msg, prior_ai, expected) in enumerate(CASES, 1):
        msgs = []
        if prior_ai:
            msgs.append(AIMessage(content=prior_ai))
        msgs.append(HumanMessage(content=user_msg))
        state = {"messages": msgs}

        t0 = time.monotonic()
        result = await classify_intent_node(state, None)
        ms = int((time.monotonic() - t0) * 1000)

        actual = result.get("intent", "?")
        ok = actual == expected
        durations.append(ms)
        n_total += 1
        if ok:
            n_correct += 1
        mark = "✓" if ok else "✗"
        ctx = "+ctx" if prior_ai else ""
        print(f"{i:3} {expected:>17} {actual:>17} {ctx:>9} {ms:>5}  {mark}      {user_msg[:50]}")

    print("─" * 110)
    avg = sum(durations) / len(durations)
    p50 = sorted(durations)[len(durations) // 2]
    p95 = sorted(durations)[int(len(durations) * 0.95)]
    fast = sum(1 for d in durations if d < 500)
    print(f"\nAccuracy: {n_correct}/{n_total}  ({n_correct/n_total*100:.0f}%)")
    print(f"Latency : avg={avg:.0f}ms  p50={p50}ms  p95={p95}ms  min={min(durations)}ms  max={max(durations)}ms")
    print(f"        : <500ms: {fast}/{n_total}  <1000ms: {sum(1 for d in durations if d < 1000)}/{n_total}")


asyncio.run(main())
