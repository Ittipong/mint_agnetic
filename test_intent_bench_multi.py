"""Multi-model intent classifier benchmark.

Runs the same CASES from test_intent_bench.py against 4 candidate models
side-by-side. Bypasses the module-level singleton + BillingCallback so
each model is a clean comparison.
"""
import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter

from src.config import settings
from src.graph import intent_classifier as ic_module


MODELS = [
    "openai/gpt-oss-safeguard-20b",
    "amazon/nova-lite-v1",
    "inception/mercury-2",
    "liquid/lfm-2-24b-a2b",
    "ibm-granite/granite-4.1-8b",
    "nvidia/nemotron-nano-9b-v2",
    "mistralai/devstral-small",
    "openai/gpt-5-nano",
]

CASES = [
    # add_transaction with explicit amount
    ("กิน kfc 100",            None, "add_transaction"),
    ("ซื้อกาแฟ 65 บาท",        None, "add_transaction"),
    ("จ่ายค่าไฟ 1200",          None, "add_transaction"),
    ("โอน 500 ให้แม่",          None, "add_transaction"),
    ("เติมน้ำมัน 800",          None, "add_transaction"),
    ("ได้เงินเดือน 35000",      None, "add_transaction"),
    ("กาแฟ 50",                None, "add_transaction"),
    # bare noun
    ("กาแฟ",                   None, "add_transaction"),
    ("เที่ยว",                  None, "add_transaction"),
    ("ค่าน้ำ",                  None, "add_transaction"),
    ("น้ำมัน",                  None, "add_transaction"),
    # follow-up bare number
    ("100", "รับทราบว่าจะบันทึก 'กาแฟ' จำนวนเงินเท่าไหร่ครับ?", "add_transaction"),
    ("200 บาท", "รับทราบว่าจะบันทึก 'เที่ยว' จำนวนเงินเท่าไหร่ครับ?", "add_transaction"),
    # analytics → other
    ("เดือนนี้ใช้เงินเท่าไหร่",  None, "other"),
    ("ขอดูรายการอาหาร",          None, "other"),
    ("ยอด KBank เหลือเท่าไหร่",  None, "other"),
    ("หมวดไหนใช้เยอะสุด",        None, "other"),
    ("เปรียบเทียบกับเดือนที่แล้ว", None, "other"),
    # chit-chat
    ("สวัสดี",                    None, "other"),
    ("ขอบคุณ",                    None, "other"),
    # adversarial
    ("กิน kfc อร่อยไหม",          None, "other"),
    ("อยากกิน kfc",               None, "other"),
    ("เที่ยวที่ไหนดี",            None, "other"),
    ("100",                       None, "other"),
    ("100", "บันทึกค่ากาแฟ 50 บาทเรียบร้อยแล้ว ✓", "other"),
]


def make_llm(model: str):
    """Plain ChatOpenRouter — no BillingCallback, no fallback list."""
    return ChatOpenRouter(
        model=model,
        api_key=settings.openrouter_api_key,
        temperature=0.0,
        timeout=60_000,
        max_retries=1,
    )


async def run_one(model: str):
    # Monkey-patch the module singletons + settings so classify_intent_node
    # uses our LLM and picks the right cache_control shape per model.
    ic_module.intent_classifier_llm = make_llm(model)
    settings.intent_classifier_model = model

    n_correct = 0
    durations = []
    failures = []
    mismatches = []

    for i, (user_msg, prior_ai, expected) in enumerate(CASES, 1):
        msgs = []
        if prior_ai:
            msgs.append(AIMessage(content=prior_ai))
        msgs.append(HumanMessage(content=user_msg))
        state = {"messages": msgs}

        t0 = time.monotonic()
        try:
            result = await ic_module.classify_intent_node(state, None)
            ms = int((time.monotonic() - t0) * 1000)
            actual = result.get("intent", "?")
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            actual = "ERROR"
            failures.append((i, user_msg, type(exc).__name__, str(exc)[:120]))

        durations.append(ms)
        if actual == expected:
            n_correct += 1
        else:
            mismatches.append((i, user_msg, expected, actual, ms))

    return {
        "model": model,
        "n_total": len(CASES),
        "n_correct": n_correct,
        "durations": durations,
        "mismatches": mismatches,
        "failures": failures,
    }


async def main():
    results = []
    for model in MODELS:
        print(f"\n>>> Benchmarking: {model}")
        r = await run_one(model)
        results.append(r)
        durs = sorted(r["durations"])
        avg = sum(durs) / len(durs)
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)]
        acc = r["n_correct"] / r["n_total"] * 100
        print(f"    acc={r['n_correct']}/{r['n_total']} ({acc:.0f}%) "
              f"avg={avg:.0f}ms p50={p50}ms p95={p95}ms "
              f"errors={len(r['failures'])}")

    # Final comparison table
    print("\n" + "=" * 78)
    print(f"{'Model':<42} {'Acc':>8} {'p50':>7} {'p95':>7} {'Err':>5}")
    print("-" * 78)
    for r in results:
        durs = sorted(r["durations"])
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)]
        acc = r["n_correct"] / r["n_total"] * 100
        print(f"{r['model']:<42} {acc:>7.0f}% {p50:>5}ms {p95:>5}ms {len(r['failures']):>5}")
    print("=" * 78)

    # Mismatch detail per model
    for r in results:
        if r["mismatches"] or r["failures"]:
            print(f"\n--- {r['model']} misses ---")
            for i, msg, expected, actual, ms in r["mismatches"]:
                print(f"  [{i:2}] expected={expected:<16} actual={actual:<16} {ms:>4}ms  {msg[:50]}")
            for i, msg, etype, emsg in r["failures"]:
                print(f"  [{i:2}] FAILED {etype}: {emsg}  ({msg[:40]})")


asyncio.run(main())
