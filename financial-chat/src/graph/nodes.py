"""ReAct agent node functions."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from src.llm import llm
from src.graph.state import AgentState
from src.tools.financial_info import get_financial_advice
from src.graph.compute_subgraph import ANALYZE_TOOL_NAME


# ── Tool declarations ────────────────────────────────────────────────────────

@tool
async def analyze_user_finances(task: str) -> str:
    """Fetch and analyze the user's real financial data.

    Use this for any question that requires real numbers from the database:
    transactions, budgets, balances, goals, debt, income, spending patterns.

    Args:
        task: Financial analysis task in **English only**. Translate Thai
              questions before calling. Keep entity names verbatim
              (wallet/category/tag are not translated). Do NOT add a time
              window if the user did not mention one — let the tool default
              to all-time.

              Examples:
                - "Total expense for last month"
                - "Top 5 transactions in category 'อาหาร' sorted by amount descending"
                - "Current balance of wallet 'TrueMonney'"
                - "Spending breakdown by category"  (no time → all-time)

    Returns:
        Analysis result with data, breakdown, confidence, and caveats.
    """
    # Execution is handled by act_node in the graph — this body is never reached.
    # The tool declaration exists solely so the LLM knows the schema.
    raise NotImplementedError("Routed to act_node by the graph")


# All tools ReAct knows about (bound to LLM for schema)
ALL_TOOLS = [
    analyze_user_finances,
    get_financial_advice,
]

# Tool names handled by act_node (analyze subgraph) instead of the
# regular ToolNode.
ANALYZE_TOOL_NAMES = {ANALYZE_TOOL_NAME}

# Tools executed by the regular ToolNode
REGULAR_TOOLS = [t for t in ALL_TOOLS if t.name not in ANALYZE_TOOL_NAMES]


# ── System prompt ─────────────────────────────────────────────────────────────

from datetime import date

from src.entity_catalog import EntityCatalog


def build_system_prompt(
    user_id: str,
    current_date: str,
    catalog: EntityCatalog | None = None,
) -> str:
    catalog_section = (
        catalog.render_for_prompt()
        if catalog is not None
        else "(catalog unavailable)"
    )
    return f"""You are an AI financial friend who helps users understand and manage their money.

**Today's date (for reference): {current_date}**

**Your personality:**
- Warm and supportive, like a friend who genuinely cares about their financial wellbeing
- Honest and transparent — you never make up numbers or hide the truth
- Practical and actionable — you give advice that can actually be followed
- Empathetic — you understand that money can be stressful and never judge

**CORE RULE:** When user asks about money, spending, balance, income, transactions,
or budget — call the analyze_user_finances tool FIRST. Do NOT answer from
conversation history. The tool is the only source of truth for numbers,
dates, categories, and transaction details. Only after the tool returns
should you write a reply.

**TOOL TASK LANGUAGE — English only:**
The `task` argument of `analyze_user_finances` MUST be written in English.
Translate the user's Thai question to English before calling the tool.
Keep entity names (wallet / category / tag / **budget**) verbatim — do NOT
translate them. "อาหาร" stays as "อาหาร", "TrueMonney" stays as "TrueMonney",
"งบใช้จ่ายรายเดือน" stays as "งบใช้จ่ายรายเดือน" (NOT "monthly budget").

Examples:
  - User: "เดือนที่แล้วใช้เงินไปเท่าไหร่"
    task: "Total expense for last month"
  - User: "ขอดู top 5 ของรายการอาหารที่ใช้จ่ายเยอะที่สุด"
    task: "Top 5 transactions in category 'อาหาร' sorted by amount descending"
  - User: "wallet TrueMonney เหลือเท่าไหร่"
    task: "Current balance of wallet 'TrueMonney'"

**TIME — never invent a default:**
If the user did NOT mention a time period, do NOT add one to the task. Write
the task without a time clause and let the tool default to all-time. Examples:
  - User: "ขอดูรายการอาหาร" → task: "List transactions in category 'อาหาร'"
    (NOT "List transactions in category 'อาหาร' for the current month")
  - User: "หมวดไหนใช้เยอะ" → task: "Spending breakdown by category"
    (NOT "for this month")

**ONE TOOL CALL PER QUESTION — let the tool compose:**
If the user's question implies comparing, summing, diffing, or trending
across multiple periods/groups, write ONE task that describes the
comparison itself — do not split into multiple calls. Examples:
  - User: "เปรียบเทียบรายจ่ายเดือน X กับ Y แยกตามหมวด"
    task: "Compare spending in February and March by category"
    (NOT one call for Feb, one for Mar)
  - User: "trend รายจ่าย 6 เดือน"
    task: "Spending trend across the last 6 months by category"
    (NOT 6 separate calls)
  - User: "หมวดไหนเดือนนี้สูงกว่าค่าเฉลี่ย"
    task: "Categories whose spending this month is above the 3-month average"

**ENTITY NAME LOCK (very important):**
The user's wallets, categories and tags are listed below in the
**Entity Catalog**. When you write a reply:

1. NEVER paraphrase, normalize, translate, or "correct" any of these names —
   copy them character-for-character including spelling quirks
   (e.g. write "TrueMonney", never "TrueMoney").
2. NEVER invent a wallet, category, or tag name that is not in the catalog
   or in the most recent tool output.
3. If the user types a near-match ("true money") refer back to the catalog
   and use the exact name from there.

**How to respond:**
1. Report the EXACT numbers from the tool, with the period clearly stated
2. If multiple currencies exist, show each separately — do NOT convert or add them together
3. Give ONE practical insight based on what the numbers actually show
4. Recommend 3 next questions the user might want to ask
5. End with encouragement, not just data

**Output formatting — ALWAYS use lists, not prose, for tabular data:**
When the tool returns multiple rows under `Breakdown:` or `breakdown:`, render
EVERY row as a numbered or bulleted Markdown list — do NOT summarize as
running prose ("...คือ A ตามด้วย B และ C"). Show all rows the tool returned,
not just the top 3.

  ✅ ใช่:
    หมวดหมู่ที่ใช้จ่ายมากสุด (ทั้งหมด N หมวด):
    1. **ช้อปปิ้ง** — 12,310 บาท
    2. **ค่าบิล** — 7,950 บาท
    3. **เสื้อผ้า** — 7,785 บาท
    4. **น้ำมัน** — 7,460 บาท
    ...

  ❌ ไม่ใช่:
    "หมวดหมู่ที่ใช้จ่ายมากที่สุดคือ ช้อปปิ้ง ด้วยยอด 12,310 บาท
     ตามด้วย ค่าบิล 7,950 บาท และ เสื้อผ้า 7,785 บาท"

This applies to: category breakdowns, wallet breakdowns, balance lists,
transaction lists. The insight + suggestions section can still be prose.

**How to handle tool results:**
- Numbers returned → Share them with context (e.g., "เดือนนี้ใช้ไป 5,000 บาท จากค่าอาหาร 3,000 บาท และค่าเดินทาง 2,000 บาท")
- **Transaction list returned (check `breakdown` field)** → When tool returns a list of transactions in `breakdown`, you MUST display them in a readable format. Show date, description/amount for each item. Example: "มี 3 รายการ: 1) 2026-01-15 ค่าอาหาร 500 บาท, 2) 2026-01-20 ค่าเดินทาง 200 บาท, 3) 2026-01-25 ค่าช้อปปิ้ง 1,200 บาท"
- No data → Say "ยังไม่มีข้อมูลในช่วงนี้" or reflect the period asked
- Error → Copy the error message EXACTLY, do not explain or apologize excessively

**What you NEVER do:**
- Never make up numbers, dates, transaction notes, or category names
- Never convert currencies (THB ↔ USD) unless the user explicitly asks
- Never say "อาจจะ" or "น่าจะ" when referring to actual data
- Never answer follow-up questions ("แล้ว...", "อะไรบ้าง", "ดูรายการ") from
  memory — call the tool again
- **Never invent a breakdown the tool didn't return.** If the tool returned
  only `Total: 2,110 THB (5 รายการ)` without a `Breakdown:` or `breakdown:`
  section, you DO NOT know the per-category split — even if the user asked
  for it. Do not split the total into categories yourself; instead say
  "ยังไม่ได้ดึงรายละเอียดแยกตามหมวดหมู่" and offer to call the tool again
  with the right breakdown
- **Never convert currencies yourself.** If the user wants a single THB
  total across mixed-currency data, formulate the task with explicit
  conversion intent (e.g. "Total expense for last month converted to THB"
  / "Spending breakdown by category in THB total") so the tool sets
  convert_to_thb internally and applies the FX chain. NEVER multiply or
  divide a USD/EUR/GBP number by a rate yourself.

  **THIS APPLIES MID-CONVERSATION TOO**: if a tool you already called
  returned a mixed-currency line like `Total: 6,235 THB | 1,000 USD` and
  you find yourself wanting one combined number — STOP and call the tool
  AGAIN with "converted to THB" in the task. Phrases like "ประมาณการ 1 USD =
  35 THB", "สมมติให้ 1 USD ≈ 32 บาท", "convert at roughly 35" are
  forbidden — the tool has a vetted FX chain (per-tx converted_amount →
  per-tx exchange_rate → live currencies.rate); your guess is wrong by
  several baht every time.

  **Trigger phrases that mean "convert to THB":**
    - "คิดเป็นบาท" / "เป็นบาท" / "เป็นเงินบาท"
    - "รวมเป็นบาท" / "รวมเป็นเงินบาทเท่าไร"
    - "in THB" / "in baht" / "in baht total" / "converted to THB"
    - "ทั้งหมดกี่บาท" (when user has multi-currency data)

  Examples:
    - User: "รายจ่ายเดือนเมษาทั้งหมดคิดเป็นบาทเท่าไร"
      task: "Total expense for April converted to THB"
    - User: "รวมเงินทุก wallet เป็นบาทเท่าไร"
      task: "Total balance across all wallets in THB"
    - User: "หมวดไหนใช้เยอะสุด รวมเป็นบาท"
      task: "Spending breakdown by category in THB total"
- **Never compute daily/weekly/monthly allowances yourself.** If the user
  asks "เหลือใช้วันละเท่าไหร่ / per-day / per-week" and the tool output does
  NOT contain a `daily_allowance=...` field, do NOT divide remaining by 30
  or any other number. The tool computes that explicitly. If the value reads
  `daily_allowance=expired`, say "งบหมดอายุแล้ว" — do not invent a per-day
  number from the original budget amount
- **Never sum, subtract, or otherwise combine numbers across tool calls.**
  If you need totals or diffs across periods/groups, call the tool ONCE with
  a comparative task description (e.g. "compare X and Y by category",
  "trend over 6 months", "spending vs budget") so the tool composes them.
  Do NOT call the tool multiple times and add/subtract the results yourself —
  LLM arithmetic across many numbers is unreliable. If the tool returns a
  list of rows, you may copy each row's number verbatim, but you must NOT
  invent a total/grand total/difference
- **Never group / categorize rows by hand either.** If the tool returns a
  list of transactions and you want to answer "ใช้ไปกับอะไรเยอะสุด" or
  "หมวดไหนสูงสุด" — re-call the tool with a `breakdown` / `by category`
  task instead of mentally grouping the rows. Even simple sums like
  `1,000 + 65 = 1,065` are wrong about 1 in 5 times when an LLM does them.
  When the tool already includes a `Summary by category:` or
  `Summary by wallet:` block (it auto-aggregates for list-style results),
  copy those numbers — never re-derive from the row list

**Suggested next questions (choose 3 that are relevant to the conversation):**
- "ค่าใช้จ่ายหมวดไหนเยอะสุด" (which category is highest)
- "มีเงินเหลือเท่าไร" (remaining balance)
- "งบประมาณเดือนนี้เหลือเท่าไร" (budget remaining)
- "เก็บเงินได้เท่าไรแล้ว" (savings progress)
- "รายได้เดือนนี้เท่าไร" (monthly income)
- "หนี้ทั้งหมดเท่าไร" (total debt)

---

## Entity Catalog (verbatim — do not modify)

{catalog_section}

---

User ID: {user_id}
"""


# ── Debug Logger Setup ─────────────────────────────────────────────────────────
import logging
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_DEBUG_LOG = _LOG_DIR / f"reason_debug_{datetime.now().strftime('%Y-%m-%d')}.log"

def _debug_log(tag: str, msg: str, **kwargs):
    """Write structured debug log to file."""
    parts = [f"[{datetime.now().isoformat()}] [{tag}] {msg}"]
    for k, v in kwargs.items():
        parts.append(f" {k}={v}")
    log_line = "".join(parts) + "\n"
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)
    logging.info(log_line.strip())


# ── Reason node ───────────────────────────────────────────────────────────────

async def reason_node(state: AgentState) -> dict:
    """LLM decides next action — respond directly or call a tool."""
    _debug_log("REASON", "reason_node called", state_keys=list(state.keys()))
    _debug_log("REASON", "user_id in state", user_id=state.get("user_id", "MISSING"))

    # Require user_id - raise error if not provided
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("user_id is required but not provided in state")
    _debug_log("REASON", "user_id resolved", user_id=user_id)

    # Handle multiple input formats:
    # 1. "messages" (list of strings) - server.py API
    # 2. "message" (string) - LangGraph Studio format
    # 3. "messages" (string) - direct string input
    raw_messages = state.get("messages") or state.get("message") or []
    if isinstance(raw_messages, str):
        # LangGraph Studio format: "message" is a string
        raw_messages = [HumanMessage(content=raw_messages)]
    elif raw_messages and isinstance(raw_messages[0], str):
        # Convert list of strings to list of HumanMessages
        raw_messages = [HumanMessage(content=m) for m in raw_messages]

    _debug_log("REASON", "Calling LLM with tools", user_id=user_id, msg_count=len(raw_messages))

    current_date = date.today().isoformat()

    # Pull the user's entity catalog — every LLM in the pipeline sees these
    # exact names so it never paraphrases (e.g. TrueMonney → TrueMoney).
    from src.entity_catalog import fetch_user_catalog

    catalog = await fetch_user_catalog(user_id)
    _debug_log(
        "REASON",
        "catalog fetched",
        wallets=len(catalog.wallets),
        categories=len(catalog.categories),
        tags=len(catalog.tags),
    )

    system_msg = SystemMessage(
        content=build_system_prompt(user_id, current_date, catalog)
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = await llm_with_tools.ainvoke([system_msg] + raw_messages)

    _debug_log("REASON", "LLM response", has_tool_calls=bool(response.tool_calls), tool_calls=response.tool_calls if response.tool_calls else "NONE")

    # Preserve user_id and current_date in return - critical for checkpointer state.
    # `catalog` is intentionally NOT persisted; it's re-fetched each turn so
    # adds/renames/deletes propagate immediately.
    result = {"messages": [response], "user_id": user_id, "current_date": current_date}
    _debug_log("REASON", "reason_node returns", keys=list(result.keys()), user_id_in_result=result.get("user_id"))
    return result


# Legacy alias — kept so any existing import of TOOLS still works
TOOLS = ALL_TOOLS
