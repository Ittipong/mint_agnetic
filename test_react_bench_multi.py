"""Multi-model benchmark for the ReAct reason_node.

Bypasses DB by injecting a hand-crafted EntityCatalog. For each case,
runs an LLM bound to ALL_TOOLS through the real `build_system_prompt`
and scores the resulting tool-call decision + key argument fields.

Cases focus on the ReAct system prompt's hardest rules:
  - Tool selection (analyze_user_finances vs lookup_entity vs direct chat)
  - Task-string quality (English, ISO dates, entity names verbatim)
  - Time defaults / window vs single-point / all-time keywords
  - Buddhist Era conversion
  - Note search vs category routing
  - Currency conversion trigger
"""
import asyncio
import time
from datetime import date
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import settings
from src.entity_catalog import EntityCatalog, WalletEntry, CategoryEntry, TagEntry
from src.graph.nodes import ALL_TOOLS, build_system_prompt
from src.llm_openrouter import ChatOpenRouterREST


MODELS = [
    "inception/mercury-2",
    "mistralai/devstral-small",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash",
    "stepfun/step-3.5-flash",
    "openai/gpt-oss-120b",
    "openai/gpt-4o-mini",
    "google/gemma-4-31b-it",
]


# ── Mock catalog (top wallets + categories user would mention) ──────
MOCK_CATALOG = EntityCatalog(
    wallets=[
        WalletEntry(sync_id="w_kbank", name="KBank", currency="THB", wallet_type="general"),
        WalletEntry(sync_id="w_truemoney", name="TrueMonney", currency="THB", wallet_type="general"),
        WalletEntry(sync_id="w_cash", name="Cash", currency="THB", wallet_type="general"),
        WalletEntry(sync_id="w_card", name="KBank Credit", currency="THB", wallet_type="creditcard"),
    ],
    categories=[
        CategoryEntry(sync_id="c_food", name="อาหาร", type="expense"),
        CategoryEntry(sync_id="c_transport", name="คมนาคม", type="expense"),
        CategoryEntry(sync_id="c_bill", name="ค่าใช้จ่ายบ้าน", type="expense"),
        CategoryEntry(sync_id="c_shop", name="ช้อปปิ้ง", type="expense"),
        CategoryEntry(sync_id="c_other", name="อื่นๆ", type="expense"),
        CategoryEntry(sync_id="c_salary", name="เงินเดือน", type="income"),
    ],
    tags=[TagEntry(sync_id="t_meeting", name="#ประชุมงาน")],
)

# Pin "today" so date-based assertions are stable across days. Matches
# the project's system reminder default.
TODAY = "2026-05-17"
SYSTEM_PROMPT = build_system_prompt("user_test", TODAY, MOCK_CATALOG)

# Pre-computed dates relative to TODAY = 2026-05-17:
#   - default-window start = last month 1st = 2026-04-01
#   - this-month range = 2026-05-01 ... 2026-05-31
#   - last-month single = 2026-04-01 ... 2026-04-30


# Each case: (id, user_text, scorer_fn)
# scorer_fn(response) -> (ok: bool, reason: str)


def _tool_call(response):
    """Extract first tool call as dict, or None."""
    tcs = getattr(response, "tool_calls", None) or []
    if not tcs:
        return None
    tc = tcs[0]
    return tc if isinstance(tc, dict) else None


def _text(response):
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return "".join(parts)
    return ""


def _expect_tool(response, expected_name):
    tc = _tool_call(response)
    if tc is None:
        return False, f"expected tool={expected_name}, got text={_text(response)[:80]!r}"
    if tc.get("name") != expected_name:
        return False, f"expected tool={expected_name}, got tool={tc.get('name')}"
    return True, ""


def score_direct_chat(response):
    """No tool call expected — greeting should be answered directly."""
    tc = _tool_call(response)
    if tc is not None:
        return False, f"expected no tool, got {tc.get('name')}"
    if not _text(response).strip():
        return False, "empty response"
    return True, "ok"


def score_default_window(response):
    """Default time window: last month 1st → today (2026-04-01 → 2026-05-17)."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "")
    if "อาหาร" not in task:
        return False, f"task missing entity 'อาหาร': {task[:120]}"
    if "2026-04" not in task and "last month" not in task.lower():
        return False, f"task missing default window start: {task[:120]}"
    return True, "ok"


def score_this_month(response):
    """SINGLE-POINT this month: 2026-05-01 → 2026-05-31."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "")
    if "2026-05-01" not in task:
        return False, f"task missing this-month start: {task[:120]}"
    if "2026-05-31" not in task:
        return False, f"task missing this-month end (should be last day, not today): {task[:120]}"
    return True, "ok"


def score_all_time(response):
    """All-time sentinel (2020-01-01) + note search for 'Starbucks'."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "")
    if "2020-01-01" not in task:
        return False, f"task missing all-time sentinel start=2020-01-01: {task[:120]}"
    if "Starbucks" not in task:
        return False, f"task missing 'Starbucks': {task[:120]}"
    return True, "ok"


def score_buddhist_era(response):
    """BE 2568 must be converted to AD 2025."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "")
    if "2568" in task:
        return False, f"task leaked BE year '2568': {task[:120]}"
    if "2025" not in task:
        return False, f"task missing converted AD year '2025': {task[:120]}"
    return True, "ok"


def score_lookup_entity(response):
    """Long-tail wallet 'KKP' (not in mock catalog) → must call lookup_entity FIRST."""
    ok, why = _expect_tool(response, "lookup_entity")
    if not ok:
        return False, why
    args = _tool_call(response).get("args", {})
    if "KKP" not in str(args.get("name_query", "")):
        return False, f"lookup_entity name_query missing 'KKP': {args}"
    return True, "ok"


def score_top_n(response):
    """Top 5 expensive → analyze_user_finances with top_transactions hint."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "").lower()
    if "top" not in task or "5" not in task:
        return False, f"task missing 'top 5': {task[:120]}"
    return True, "ok"


def score_currency_convert(response):
    """Mixed-currency total request → task must mention 'THB' conversion."""
    ok, why = _expect_tool(response, "analyze_user_finances")
    if not ok:
        return False, why
    task = _tool_call(response).get("args", {}).get("task", "")
    if "THB" not in task and "baht" not in task.lower():
        return False, f"task missing currency conversion hint 'THB': {task[:120]}"
    if "convert" not in task.lower() and "in thb" not in task.lower() and "to thb" not in task.lower():
        return False, f"task missing 'convert/in THB/to THB' hint: {task[:120]}"
    return True, "ok"


CASES = [
    ("direct_chat",         "สวัสดี",                                                   score_direct_chat),
    ("default_window",      "ใช้เงินกับอาหารไปเท่าไร",                                   score_default_window),
    ("this_month",          "เดือนนี้ใช้เงินเท่าไหร่",                                    score_this_month),
    ("all_time_starbucks",  "เคยใช้ Starbucks กี่ครั้งแล้ว",                              score_all_time),
    ("buddhist_era",        "ปี 2568 รายจ่ายรวมเท่าไหร่",                                 score_buddhist_era),
    ("lookup_long_tail",    "wallet KKP เหลือเท่าไหร่",                                   score_lookup_entity),
    ("top_n",               "ขอดู 5 รายการแพงสุดในเดือนนี้",                              score_top_n),
    ("currency_convert",    "รายจ่ายเดือนเมษาทั้งหมดคิดเป็นบาทเท่าไร",                     score_currency_convert),
]


async def run_one(model: str):
    llm = ChatOpenRouterREST(
        model=model,
        api_key=settings.openrouter_api_key,
        temperature=0.0,
        timeout=180.0,
        max_retries=1,
    ).bind_tools(ALL_TOOLS)

    results = []
    for case_id, user_text, scorer in CASES:
        msgs = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
        t0 = time.monotonic()
        try:
            response = await llm.ainvoke(msgs)
            ms = int((time.monotonic() - t0) * 1000)
            ok, reason = scorer(response)
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            ok, reason = False, f"EXCEPTION {type(exc).__name__}: {str(exc)[:140]}"
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
        avg = int(sum(durs) / len(durs))
        print(f"    pass={passed}/{len(rs)} avg={avg}ms p50={p50}ms p95={p95}ms")
        for cid, ok, reason, ms in rs:
            mark = "✓" if ok else "✗"
            tail = "" if ok else f"  ← {reason[:120]}"
            print(f"      {mark} {cid:<22} {ms:>5}ms{tail}")

    # Summary table
    print("\n" + "=" * 92)
    print(f"{'Model':<42} {'Pass':>10} {'p50':>8} {'p95':>8} {'avg':>8}")
    print("-" * 92)
    for model, rs in all_runs:
        passed = sum(1 for _, ok, _, _ in rs if ok)
        durs = sorted(ms for _, _, _, ms in rs)
        p50 = durs[len(durs) // 2]
        p95 = durs[int(len(durs) * 0.95)]
        avg = int(sum(durs) / len(durs))
        pct = passed / len(rs) * 100
        print(f"{model:<42} {passed}/{len(rs)} ({pct:>3.0f}%) {p50:>6}ms {p95:>6}ms {avg:>6}ms")
    print("=" * 92)


asyncio.run(main())
