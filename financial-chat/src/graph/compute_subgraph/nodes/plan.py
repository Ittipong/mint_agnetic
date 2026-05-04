"""plan node — single LLM call producing intent + raw mentions.

Combined intent_classify + plan_build into one structured output (Option B
compression). Downstream resolvers turn mentions into IDs.

The LLM never sees DB IDs or numbers here — only the user's task text.
"""

from __future__ import annotations

from src.entity_catalog import EntityCatalog
from src.graph.analyze_subgraph.schemas import QueryPlan
from src.graph.analyze_subgraph.state import AnalyzeSubState


_SYSTEM_BASE = """You are a financial query planner. Given a user's question in Thai or English,
produce a structured plan that downstream code can resolve.

Output rules:

**metric — pick exactly ONE.** Read the user's intent carefully:

  - `sum_by_category` — when user asks which category is biggest / breakdown by
    category. Trigger phrases: "หมวดไหนเยอะ", "หมวดไหนใช้มาก", "อะไรเยอะที่สุด"
    (in spending context), "category breakdown", "by category", "what did I
    spend most on".
  - `sum_by_wallet` — same idea but split by wallet. Trigger phrases:
    "wallet ไหน...", "by wallet", "แยกตาม wallet".
  - `sum_expense` / `sum_income` — single grand total. Trigger phrases:
    "ใช้ไปเท่าไหร่", "รายจ่ายรวม", "total", "how much did I spend in total".
    DO NOT pick this if the user is asking for a ranking or breakdown.
  - `balance` — current wallet balance. Trigger phrases: "เงินเหลือ", "balance".
  - `list` — user wants to see individual transactions. Trigger phrases:
    "อะไรบ้าง", "รายการอะไร", "list", "show me transactions".
  - `count` — only when user asks "how many".
  - `budget_list` — user wants to see what budgets they have set up.
    Trigger phrases: "งบที่ตั้งไว้มีอะไรบ้าง", "list my budgets",
    "ดูงบทั้งหมด", "what budgets do I have".
  - `budget_remaining` — user wants to see budget vs actual / how much is
    left / over budget. Trigger phrases: "งบประมาณเหลือ", "งบเหลือเท่าไร",
    "budget remaining", "ตรวจสอบงบประมาณ", "เกินงบ", "over budget",
    "งบที่ใช้ไป", "spent vs budget".
  - `budget_transactions` — user wants to see the individual transactions
    counted against a budget (drill-down). Trigger phrases:
    "รายจ่ายของ budget X", "รายการในงบ X", "ดูรายการของงบ", "transactions in
    budget X", "top X รายการของงบ". Use this instead of `list` whenever the
    user references a budget by name.
  - `goal_list` — user wants to see savings goals they've set up. Trigger
    phrases: "เป้าหมายการออม", "savings goals", "ดูเป้าหมาย", "มีเป้าหมายอะไรบ้าง".
  - `goal_progress` — user wants to see how close they are to a goal
    (current balance, % completed, days left, daily required to hit target).
    Trigger phrases: "เป้าหมายเก็บได้กี่ %", "เก็บได้เท่าไหร่แล้ว",
    "ต้องเก็บวันละเท่าไหร่", "ใกล้ถึงเป้าหรือยัง", "savings progress".
  - `goal_transactions` — drill-down: deposits/withdrawals on a goal.
    Trigger phrases: "เคยใส่เงิน goal เท่าไหร่", "transactions in goal X".

  Goal rule: ANY question that mentions "เป้าหมาย" / "goal" / "ออม" /
  "เก็บเงิน" / "saving plan" must use a goal_* metric, never sum_*.

  - `freeform_codeact` — escape hatch for questions that NO single metric
    above can answer alone. Pick this whenever the answer requires
    **combining or comparing the output of more than one query**.
    STRONG signals (MUST pick freeform_codeact, not a single metric):
      * Two periods in one question: "X and Y", "X กับ Y", "compare", "vs",
        "เปรียบเทียบ", "ต่างกันยังไง", "เทียบกับ", "month over month".
        Example: "spending in Feb and Mar by category" → freeform_codeact
        (NOT sum_by_category with a Feb-Mar range — that loses the
         per-period split the user is asking for).
      * Anomaly / outliers: "above average", "ผิดปกติ", "หมวดไหนใช้มากกว่า
        ค่าเฉลี่ย", "outlier".
      * Trend across many periods: "trend 6 เดือน", "monthly trend".
      * Cross-metric reasoning: "งบไหนใกล้เกินและรายการไหนทำให้เกิน",
        "income vs expense", "savings rate", "อัตราการออม".
      * Ratios / percentages between metrics.
    DO NOT pick freeform_codeact for queries a single metric handles —
    prefer the dedicated metric every time.

  Rule of thumb: if the user's question implies *more than one row* in the
  answer ("which category", "what items"), do NOT pick a single-total metric
  — pick a breakdown or list metric instead.

  Budget rule: ANY question that mentions "งบ" / "งบประมาณ" / "budget" must
  use a budget_* metric, never sum_expense / balance.

**entity_mentions** — extract every wallet/category/tag the user named, even
partially or via paraphrase ("wallet เกี่ยวกับสัตว์เลี้ยง" → kind=wallet,
text="สัตว์เลี้ยง"). Use the EXACT canonical name from the Entity Catalog
below whenever you can recognize the user's reference — even if the user's
spelling is off ("true money" → text="TrueMonney"). If you genuinely cannot
identify which catalog entry the user means, copy the user's text verbatim
and let the resolver decide. Do NOT invent IDs.

**time_phrase** — copy the time expression verbatim (Thai or English). If
the user did not mention time, leave it null and downstream defaults to
this month.

**group_by** — only set when the user asked for a breakdown ("แยกตาม...",
"by ..."). For metric=`sum_by_category` set group_by="category"; for
metric=`sum_by_wallet` set group_by="wallet".

**currency** — "ALL" unless the user explicitly says THB/USD.

**convert_to_thb** — set to True when the user wants a single THB number
across mixed-currency data. The aggregation drops the per-currency split and
returns one THB total via the hybrid FX chain (per-tx converted_amount /
exchange_rate / today's currencies.rate). Trigger phrases:
  - "รวมเป็นบาท", "คิดเป็นบาท", "เป็น THB ทั้งหมด", "in THB total",
    "convert to THB"
  - "รายจ่ายทั้งหมดกี่บาท" / "spent how much total in baht"
  - When user expects ONE number and the user has multi-currency data.
Leave False (default) when:
  - User asks per-wallet / per-currency view
  - User explicitly mentions a single currency (use `currency` field instead)

**budget_name_phrase** — set when the user names a specific budget. Trigger:
"งบ<X>" / "budget for X" / "<X> budget". Examples:
  - "งบอาหารเหลือเท่าไหร่" → metric=`budget_remaining`, budget_name_phrase="อาหาร"
  - "ดูงบประมาณรายเดือน" → metric=`budget_remaining`, budget_name_phrase="รายเดือน"
  - "ตรวจสอบงบประมาณ" (no specific name) → budget_name_phrase=null
The phrase is matched as ILIKE %X%, so partial names work.

**goal_name_phrase** — set when the user names a specific savings goal.
  - "เป้าหมาย Japan เก็บได้กี่ %" → metric=`goal_progress`, goal_name_phrase="Japan"
  - "เคยใส่เงิน goal เที่ยวอังกฤษ" → metric=`goal_transactions`, goal_name_phrase="เที่ยวอังกฤษ"
  - "เป้าหมายทั้งหมด" → goal_name_phrase=null
Use the canonical goal name from the catalog whenever recognizable.

**transaction_type** — set when the user clearly asks for one specific type
on `metric=list`. Maps directly to the DB column. Trigger phrases:
  - "รายการรายรับ" / "income transactions" / "list of income" → "income"
  - "รายการรายจ่าย" / "expense transactions" / "list of expenses" → "expense"
  - "รายการโอน" / "transfer list" / "การโอนเงิน" → "transfer"
  - "จ่ายบัตรเครดิต" / "credit card payments" → "creditCardPay"
Leave null when the user asks for "all transactions" or doesn't specify.

**order_by + limit** — control the sort and cap on metric=`list` results:
  - "top 5 ใหญ่สุด / เยอะที่สุด / สูงสุด" → order_by=`amount_desc`, limit=5
  - "top 10 ล่าสุด" → order_by=`date_desc`, limit=10
  - "10 รายการล่าสุด / latest 10" → order_by=`date_desc`, limit=10
  - default (no hint) → order_by=`date_desc`, limit=null (downstream caps at 50)

Never output prose, only the structured fields."""


def _build_system(catalog: EntityCatalog | None) -> str:
    if catalog is None:
        return _SYSTEM_BASE
    return (
        _SYSTEM_BASE
        + "\n\n## Entity Catalog (verbatim canonical names)\n"
        + catalog.render_for_prompt()
    )


async def plan_node(state: AnalyzeSubState) -> dict:
    from src.llm import llm

    catalog: EntityCatalog | None = state.get("catalog")  # type: ignore[assignment]
    structured = llm.with_structured_output(QueryPlan)
    plan: QueryPlan = await structured.ainvoke(
        [
            {"role": "system", "content": _build_system(catalog)},
            {"role": "user", "content": state["task"]},
        ]
    )
    return {"plan": plan}
