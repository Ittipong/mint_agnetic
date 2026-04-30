"""Benchmark: compare N models on a single query. Run one query at a time.

Usage:
    python benchmark.py                        # run first query
    python benchmark.py wallet_balance
    python benchmark.py uc1_1_total_balance
    python benchmark.py uc5_1_health_score
    python benchmark.py calc01_wallet_sum        # 10 calculation test cases
    python benchmark.py calc02_txn_total
    python benchmark.py calc03_spending_by_cat
    python benchmark.py calc04_budget_variance
    python benchmark.py calc05_net_worth
    python benchmark.py calc06_cc_utilization
    python benchmark.py calc07_daily_avg_spend
    python benchmark.py calc08_monthly_cashflow
    python benchmark.py calc09_mom_change_pct
"""
import asyncio
import sys
import time
from decimal import Decimal

from financial_agent.agent import FinancialCodeActAgent
from financial_agent.config import Settings

USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"

QUERIES = {
    # --- existing ---
    "wallet_balance":   "แสดง wallet ทั้งหมดและยอดคงเหลือของแต่ละ wallet",

    # --- Calculation Test Suite (10 cases) ---
    # ทดสอบความสามารถในการคำนวณของ agent

    # 1. รวมยอด wallet ทั้งหมด → คำตอบ: 255,000 THB
    "calc01_wallet_sum": "Calculate the sum of all wallet balances. What is the total?",

    # 2. รวมยอด transactions ทั้งหมดในเดือนปัจจุบัน → sum income vs expense
    "calc02_txn_total": "Calculate total income and total expenses for all transactions in April 2026. Show both amounts and the net.",

    # 3. รวมค่าใช้จ่ายแต่ละ category → breakdown by category_name
    "calc03_spending_by_cat": "Show total spending grouped by category_name for the last 30 days. List each category and its total.",

    # 4. คำนวณ budget variance % → (spent - budgeted) / budgeted * 100
    "calc04_budget_variance": "For each active budget, calculate the variance percentage: (spent - budgeted) / budgeted * 100. Show which budgets are overspent.",

    # 5. คำนวณ net worth → total_assets - total_debts
    "calc05_net_worth": "Calculate my net worth: total assets minus total debts. Show the breakdown.",

    # 6. คำนวณ credit card utilization % → (used / limit) * 100
    "calc06_cc_utilization": "For each credit card, calculate the utilization percentage: (used_amount / credit_limit) * 100. Show which card has highest utilization.",

    # 7. คำนวณ average daily spending → total_spending / days
    "calc07_daily_avg_spend": "Calculate the average daily spending over the last 30 days. Show total spending divided by number of days with transactions.",

    # 8. คำนวณ monthly cash flow → income - expenses for current month
    "calc08_monthly_cashflow": "Calculate net cash flow for April 2026: total income minus total expenses. Show both numbers and the net.",

    # 9. คำนวณ % change MoM → (this_month - last_month) / last_month * 100
    "calc09_mom_change_pct": "Calculate the month-over-month percentage change in total spending. Compare April 2026 vs March 2026. Show the % increase or decrease.",

    # 10. คำนวณ loan interest → outstanding_principal * annual_rate / 12

}

# sorted by total cost (in + out) per 1M tokens — ascending
MODELS = [
    "google/gemini-2.5-flash-lite",
    "openai/gpt-oss-safeguard-20b:nitro",
    "qwen/qwen3-vl-30b-a3b-instruct",
    # "openai/gpt-oss-120b",          # in $0.039  out $0.19   total $0.229/M
    # "qwen/qwen3.5-9b",              # in $0.10   out $0.15   total $0.25/M
    # "qwen/qwen3-8b",                # in $0.05   out $0.40   total $0.45/M
    # "google/gemma-4-31b-it",        # in $0.13   out $0.38   total $0.51/M
    # "x-ai/grok-4.1-fast",           # in $0.20   out $0.50   total $0.70/M
    # "qwen/qwen3-coder-next",        # in $0.14   out $0.80   total $0.94/M
]


async def run_query(model: str, task: str) -> dict:
    settings = Settings(model=model, complex_model=model, max_steps=3)
    t0 = time.perf_counter()
    try:
        async with FinancialCodeActAgent(settings=settings) as agent:
            result = await agent.solve(task, user_id=USER_ID)
        elapsed = time.perf_counter() - t0
        return {
            "model": model,
            "status": result.status,
            "steps": result.total_steps,
            "elapsed_s": round(elapsed, 1),
            "result": result.result,
            "breakdown": result.breakdown,
            "metadata": result.metadata,
            "error": None,
            "generated_steps": [
                {"step": s.step, "code": s.code, "status": s.execution.status, "error": s.execution.error}
                for s in result.steps
            ],
        }
    except Exception as e:
        return {
            "model": model,
            "status": "exception",
            "steps": 0,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "result": None,
            "breakdown": [],
            "metadata": {},
            "error": str(e)[:200],
        }


def _fmt_decimal(v) -> str:
    if v is None:
        return "—"
    try:
        d = Decimal(str(v))
        return f"{d:,.2f} THB"
    except Exception:
        return str(v)


def print_result(r: dict):
    status_icon = {"completed": "✅", "partial": "⚠️", "error": "❌", "exception": "💥"}.get(r["status"], "?")
    print(f"\n{'─'*60}")
    print(f"  Model : {r['model']}")
    print(f"  Status: {status_icon} {r['status']}  |  Steps: {r['steps']}  |  Time: {r['elapsed_s']}s")

    if r["error"]:
        print(f"  Error : {r['error']}")
        return

    print(f"  Answer: {_fmt_decimal(r['result'])}")

    caveat = r["metadata"].get("caveat", "")
    if caveat:
        print(f"  Note  : {caveat}")

    if r["breakdown"]:
        print(f"  {'─'*50}")
        print(f"  {'Category':<32} {'Amount':>15}")
        print(f"  {'─'*50}")
        for b in r["breakdown"]:
            label = b.get("label") or "Uncategorized"
            value = b.get("value", "")
            try:
                val_str = f"{Decimal(str(value)):>15,.2f}"
            except Exception:
                val_str = f"{str(value):>15}"
            print(f"  {label:<32} {val_str}")

    if r.get("generated_steps"):
        print(f"\n  {'─'*50}")
        print(f"  Generated Code")
        for gs in r["generated_steps"]:
            print(f"\n  [Step {gs['step']} — {gs['status']}]")
            for line in gs["code"].splitlines():
                print(f"    {line}")
            if gs.get("error"):
                print(f"    # ERROR: {gs['error']}")


async def main():
    # Pick query from CLI arg or default
    query_key = sys.argv[1] if len(sys.argv) > 1 else list(QUERIES.keys())[0]
    if query_key not in QUERIES:
        print(f"Unknown query '{query_key}'. Available: {', '.join(QUERIES)}")
        sys.exit(1)

    task = QUERIES[query_key]
    print(f"\n{'='*60}")
    print(f"  Query : {query_key}")
    print(f"  Task  : {task}")
    print(f"  Models: {len(MODELS)}  |  Max steps: 3")
    print(f"{'='*60}")

    tasks = [run_query(model, task) for model in MODELS]
    results = await asyncio.gather(*tasks)

    # Print each model result
    for r in sorted(results, key=lambda x: (x["status"] != "completed", x["elapsed_s"])):
        print_result(r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {query_key}")
    print(f"{'='*60}")
    print(f"  {'Model':<36} {'Status':<10} {'Steps':>5} {'Time':>7}")
    print(f"  {'─'*56}")
    for r in sorted(results, key=lambda x: (x["status"] != "completed", x["elapsed_s"])):
        icon = {"completed": "✅", "partial": "⚠️", "error": "❌", "exception": "💥"}.get(r["status"], "?")
        print(f"  {r['model']:<36} {icon} {r['status']:<8} {r['steps']:>5} {r['elapsed_s']:>6}s")


if __name__ == "__main__":
    asyncio.run(main())
