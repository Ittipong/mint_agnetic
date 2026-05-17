"""Multi-model benchmark for quick_add_node.

Bypasses DB by injecting a hand-crafted catalog into the system prompt.
For each case, runs an LLM bound to `propose_transaction` and scores:

  - Tool-fire cases   : did the model call the tool? args correct?
  - Ask-back cases    : did the model reply with text (no tool_call)?

Covers: single-turn complete add, income detection, wallet match,
multi-turn merge, bare amount, bare noun, currency-lock attempt.
"""
import asyncio
import time
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from src.config import settings
from src.graph.quick_add_node import _build_quick_add_prompt
from src.tools.transaction import propose_transaction


MODELS = [
    "inception/mercury-2",
    "mistralai/devstral-small",
    "nvidia/nemotron-nano-9b-v2",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-coder-30b-a3b-instruct",
]

# ── Mock catalog ────────────────────────────────────────────
# Two general wallets so we can test wallet-matching logic.
WALLET_MAP = """### Wallet: `KBank` (sync_id=`wallet_kbank`, currency=THB)
  Expense categories:
  - sync_id=`cat_food` name=`อาหาร`
  - sync_id=`cat_utility` name=`ค่าใช้จ่ายบ้าน`
  - sync_id=`cat_transport` name=`คมนาคม`
  - sync_id=`cat_shopping` name=`ช้อปปิ้ง`
  - sync_id=`cat_other_exp` name=`อื่นๆ`
  Income categories:
  - sync_id=`cat_salary` name=`เงินเดือน`
  - sync_id=`cat_other_inc` name=`อื่นๆ (รายได้)`
### Wallet: `Cash` (sync_id=`wallet_cash`, currency=THB)
  Expense categories:
  - sync_id=`cat_food_cash` name=`อาหาร`
  - sync_id=`cat_other_cash` name=`อื่นๆ`"""

TODAY = date.today().isoformat()
SYSTEM_PROMPT = _build_quick_add_prompt(TODAY, WALLET_MAP, "THB", "฿")


# Each case: (id, messages, expect_dict)
# expect_dict keys (all optional):
#   tool: True/False         — must fire propose_transaction?
#   amount: float            — exact amount expected
#   type: "expense"/"income"
#   wallet_id: str           — must equal
#   note_contains: str       — substring expected in note (case-insensitive)
#   ask_contains: list[str]  — substring(s) expected in ask-back text
CASES = [
    (
        "single_expense",
        [HumanMessage(content="กิน kfc 100")],
        {"tool": True, "amount": 100, "type": "expense", "note_contains": "kfc"},
    ),
    (
        "single_expense_baht",
        [HumanMessage(content="ซื้อกาแฟ 65 บาท")],
        {"tool": True, "amount": 65, "type": "expense", "note_contains": "กาแฟ"},
    ),
    (
        "income_salary",
        [HumanMessage(content="ได้เงินเดือน 35000")],
        {"tool": True, "amount": 35000, "type": "income"},
    ),
    (
        "utility_wallet_match",
        [HumanMessage(content="จ่ายค่าไฟ 1200 จาก KBank")],
        {"tool": True, "amount": 1200, "type": "expense", "wallet_id": "wallet_kbank"},
    ),
    (
        "bare_noun_askback",
        [HumanMessage(content="เที่ยว")],
        {"tool": False, "ask_contains": ["เที่ยว"]},
    ),
    (
        "bare_amount_askback",
        [HumanMessage(content="100")],
        {"tool": False},  # should ask for name
    ),
    (
        "multi_turn_merge",
        [
            HumanMessage(content="เที่ยว"),
            AIMessage(content="รับทราบว่าจะบันทึก 'เที่ยว' จำนวนเงินเท่าไหร่ครับ?"),
            HumanMessage(content="200"),
        ],
        {"tool": True, "amount": 200, "type": "expense", "note_contains": "เที่ยว"},
    ),
    (
        "multi_turn_with_unit",
        [
            HumanMessage(content="กาแฟ"),
            AIMessage(content="รับทราบว่าจะบันทึก 'กาแฟ' จำนวนเงินเท่าไหร่ครับ?"),
            HumanMessage(content="50 บาท"),
        ],
        {"tool": True, "amount": 50, "type": "expense", "note_contains": "กาแฟ"},
    ),
    (
        "currency_lock_dollar",
        # Even when user types $ the prompt requires using THB/฿.
        [HumanMessage(content="ซื้อหนังสือ $20")],
        {"tool": True, "amount": 20, "type": "expense"},
    ),
    (
        "transfer_to_mom",
        [HumanMessage(content="โอน 500 ให้แม่")],
        {"tool": True, "amount": 500, "type": "expense"},
    ),
]


def score_case(case_id, expect, response):
    """Return (ok: bool, reason: str). Reason explains failure."""
    tool_calls = getattr(response, "tool_calls", None) or []
    text = ""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "")

    want_tool = expect.get("tool", True)
    if want_tool and not tool_calls:
        return False, f"expected tool_call, got text: {text[:80]!r}"
    if not want_tool and tool_calls:
        args = tool_calls[0].get("args") if isinstance(tool_calls[0], dict) else {}
        return False, f"expected ask-back, got tool_call amount={args.get('amount')}"

    if not want_tool:
        # Ask-back path
        asks = expect.get("ask_contains") or []
        for sub in asks:
            if sub not in text:
                return False, f"ask-back missing {sub!r}; text={text[:80]!r}"
        return True, "ok"

    # Tool-fire path
    args = tool_calls[0].get("args") if isinstance(tool_calls[0], dict) else {}
    if not isinstance(args, dict):
        return False, f"tool args not dict: {args!r}"

    if "amount" in expect:
        got_amount = args.get("amount")
        if not isinstance(got_amount, (int, float)):
            return False, f"amount not numeric: {got_amount!r}"
        if abs(float(got_amount) - float(expect["amount"])) > 0.01:
            return False, f"amount mismatch: want={expect['amount']} got={got_amount}"
    if "type" in expect:
        if args.get("type") != expect["type"]:
            return False, f"type mismatch: want={expect['type']} got={args.get('type')}"
    if "wallet_id" in expect:
        if args.get("wallet_id") != expect["wallet_id"]:
            return False, f"wallet mismatch: want={expect['wallet_id']} got={args.get('wallet_id')}"
    if "note_contains" in expect:
        note = (args.get("note") or "").lower()
        if expect["note_contains"].lower() not in note:
            return False, f"note missing {expect['note_contains']!r}: got={note!r}"

    return True, "ok"


async def run_one(model: str):
    llm = ChatOpenRouter(
        model=model,
        api_key=settings.openrouter_api_key,
        temperature=0.1,
        timeout=120_000,
        max_retries=1,
    ).bind_tools([propose_transaction])

    results = []
    for case_id, msgs, expect in CASES:
        full = [SystemMessage(content=SYSTEM_PROMPT), *msgs]
        t0 = time.monotonic()
        try:
            response = await llm.ainvoke(full)
            ms = int((time.monotonic() - t0) * 1000)
            ok, reason = score_case(case_id, expect, response)
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            ok, reason = False, f"EXCEPTION {type(exc).__name__}: {str(exc)[:120]}"
        results.append((case_id, ok, reason, ms))
    return results


async def main():
    all_runs = []
    for model in MODELS:
        print(f"\n>>> Benchmarking: {model}")
        rs = await run_one(model)
        all_runs.append((model, rs))
        passed = sum(1 for _, ok, _, _ in rs if ok)
        durs = sorted(ms for _, _, _, ms in rs)
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)]
        avg = sum(durs) / len(durs)
        print(f"    pass={passed}/{len(rs)} avg={avg:.0f}ms p50={p50}ms p95={p95}ms")
        for cid, ok, reason, ms in rs:
            mark = "✓" if ok else "✗"
            print(f"      {mark} {cid:<28} {ms:>5}ms  {reason if not ok else ''}")

    # Summary table
    print("\n" + "=" * 92)
    print(f"{'Model':<48} {'Pass':>7} {'p50':>7} {'p95':>7} {'avg':>7}")
    print("-" * 92)
    for model, rs in all_runs:
        passed = sum(1 for _, ok, _, _ in rs if ok)
        durs = sorted(ms for _, _, _, ms in rs)
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)]
        avg = int(sum(durs) / len(durs))
        pct = passed / len(rs) * 100
        print(f"{model:<48} {passed}/{len(rs)} ({pct:>3.0f}%) {p50:>5}ms {p95:>5}ms {avg:>5}ms")
    print("=" * 92)


asyncio.run(main())
