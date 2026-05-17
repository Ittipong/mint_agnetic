"""ReAct agent node functions."""

import asyncio
import time

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from src.llm import llm
from src.graph.state import AgentState
from src.tools.entity_lookup import lookup_entity
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


# All tools ReAct knows about (bound to LLM for schema).
# NOTE: propose_transaction is intentionally NOT here — it's only
# available in the slip subgraph (slip_node), which has the vision
# context required to populate its args correctly. Exposing it to
# the text ReAct loop lets the chat LLM fire it on plain-text
# intent markers with no image, producing empty/hallucinated cards.
ALL_TOOLS = [
    analyze_user_finances,
    get_financial_advice,
    lookup_entity,
]

# Tool names handled by act_node (analyze subgraph) instead of the
# regular ToolNode.
ANALYZE_TOOL_NAMES = {ANALYZE_TOOL_NAME}

# Tools executed by the regular ToolNode
REGULAR_TOOLS = [t for t in ALL_TOOLS if t.name not in ANALYZE_TOOL_NAMES]


# ── System prompt ─────────────────────────────────────────────────────────────

from datetime import date

from src.entity_catalog import EntityCatalog


def _render_slip_context(catalog: EntityCatalog | None) -> str:
    """Render wallets + expense categories with sync_ids for the slip-
    parsing flow. Sync_ids are required so `propose_transaction` can
    reference entities by their canonical key, not display name."""
    if catalog is None:
        return "(no wallets / categories — ask user to set up first)"

    lines: list[str] = []
    if catalog.wallets:
        lines.append("Wallets (use sync_id when calling the tool):")
        for w in catalog.wallets:
            lines.append(
                f"- sync_id=`{w.sync_id}` name=`{w.name}` "
                f"type={w.wallet_type} currency={w.currency}"
            )
    else:
        lines.append("Wallets: (none)")

    expense_cats = [c for c in catalog.categories if c.type == "expense"]
    if expense_cats:
        lines.append("\nExpense categories (use sync_id when calling the tool):")
        for c in expense_cats:
            lines.append(f"- sync_id=`{c.sync_id}` name=`{c.name}`")
    else:
        lines.append("\nExpense categories: (none)")

    income_cats = [c for c in catalog.categories if c.type == "income"]
    if income_cats:
        lines.append("\nIncome categories (use sync_id when calling the tool):")
        for c in income_cats:
            lines.append(f"- sync_id=`{c.sync_id}` name=`{c.name}`")

    return "\n".join(lines)


# How many wallets/categories to render verbatim in the ReAct system
# prompt. Users with >20 entries see only the top-N by usage; the rest
# are reachable via the `lookup_entity` tool. Keeps the per-turn input
# under ~1KB for the entity section even for power users.
_CATALOG_TOP_N = 20


def build_system_prompt(
    user_id: str,
    current_date: str,
    catalog: EntityCatalog | None = None,
    *,
    top_n: int = _CATALOG_TOP_N,
) -> str:
    catalog_section = (
        catalog.render_for_prompt_truncated(top_n=top_n)
        if catalog is not None
        else "(catalog unavailable)"
    )
    slip_context = _render_slip_context(catalog)
    return f"""You are an AI financial friend who helps users understand and manage their money.

**Today's date (for reference): {current_date}**

**SLIP → TRANSACTION (special intent — takes priority over everything else):**

If the user's message starts with `[INTENT:parse_transaction_from_slip]`,
the rest of the message is OCR text extracted from a payment slip /
receipt on the mobile client. Your job is to:

1. Parse the OCR text — find the amount paid, date/time, merchant, and
   any line items.
2. Pick the most likely wallet from the **Slip context** below by
   matching bank logos / wallet names / payment method keywords.
   If no clear match, leave `wallet_id` null.
3. Pick the most likely expense category from the **Slip context** by
   matching merchant type / item keywords (e.g. "ร้านอาหาร" / "อาหาร",
   "Cafe Amazon" / "เครื่องดื่ม", "BTS" / "เดินทาง"). If no clear match,
   leave `category_id` null.
4. Call the `propose_transaction` tool with the matched values. Use
   the catalog `sync_id` for `wallet_id` and `category_id` — NOT the
   display name. Include a one-line `note` summarizing line items
   when the slip is a receipt with multiple items.
5. After the tool call, reply with **one short Thai sentence** like
   "ดูข้อมูลในการ์ดด้านบนได้เลยครับ — กดบันทึกถ้าถูกต้อง" — do NOT
   restate the transaction details, the client renders a card.
6. If the OCR text is empty / unreadable / clearly not a slip, do NOT
   call the tool. Instead reply in plain Thai asking the user for the
   details (amount, merchant) and skip the suggestions tag.

For slip-intent turns you do NOT need to:
- Call `analyze_user_finances` (the slip text is the source of truth).
- Emit a `<suggestions>` tag.
- Translate to English (the tool args are already structured).

### Slip context (wallets + categories for this user)

{slip_context}

---

(The instructions below apply ONLY when the user message does NOT
start with the slip-intent marker.)

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

**TIME — default to last-month-1st → today when user did not specify:**
If the user did NOT mention a time period, ALWAYS add an explicit ISO date
range to the task using:
  - start = the 1st day of the PREVIOUS calendar month
  - end   = today's date (above)
Compute the dates yourself from "Today's date" — never let the tool guess.
Use the format `start = YYYY-MM-DD, end = YYYY-MM-DD`.

EXCEPTIONS — these phrases mean ALL-TIME, NOT the default window:
  - "ทั้งหมด" / "รวมทั้งหมด" / "ตั้งแต่เริ่ม" / "ตั้งแต่ใช้แอป" /
    "ที่ผ่านมาทั้งหมด" / "all time" / "เคย...บ้าง" / "กี่ครั้งแล้ว"
When ANY of these phrases appear ANYWHERE in the question (even at the end,
e.g. "transfer เข้า kbank ทั้งหมดมีเท่าไหร่"), set start = 2020-01-01 (or any
earlier sentinel date that predates the user's data) and end = today. Do NOT
apply the last-month default.

CRITICAL — "ทั้งหมด" is ambiguous: it can mean "total amount" (sum) OR
"all time" (period). Default to ALL-TIME unless the question already names
a period — e.g. "เดือนนี้ใช้ทั้งหมดเท่าไร" the period = this month, not
all-time. But "X ทั้งหมดมีเท่าไหร่" / "เคยใช้ X ทั้งหมดเท่าไหร่" with no
named period = ALL-TIME.

Examples:
  - User: "tag #ประชุมงาน ใช้ไปทั้งหมดเท่าไหร่"
    task: "Total expense tagged '#ประชุมงาน', start = 2020-01-01,
           end = 2026-05-05"
  - User: "transfer เข้า kbank ทั้งหมดมีเท่าไหร่"
    task: "List transfers into wallet 'kbank' (incoming),
           start = 2020-01-01, end = 2026-05-05"
  - User: "เคยใช้ Starbucks กี่ครั้งแล้ว"
    task: "Count transactions whose note contains 'Starbucks',
           start = 2020-01-01, end = 2026-05-05"

Why this default: a single calendar month is too narrow when today is early in
the month (the user usually wants to see recent activity that includes last
month). All-time is too broad and confuses the user with stale data.

Examples (assume today = 2026-05-05 → default start = 2026-04-01):
  - User: "ใช้เงินกับอาหารไปเท่าไร" (no period)
    task: "Total expense in category 'อาหาร',
           start = 2026-04-01, end = 2026-05-05"
  - User: "ขอดูรายการอาหาร" (no period)
    task: "List transactions in category 'อาหาร',
           start = 2026-04-01, end = 2026-05-05"
  - User: "หมวดไหนใช้เยอะ" (no period)
    task: "Spending breakdown by category,
           start = 2026-04-01, end = 2026-05-05"

This default does NOT apply to balance / credit-card / goal / budget queries
(they are point-in-time or active-scope by design — leave them without dates).
When the user DOES name a period (เดือนนี้ / เดือนที่แล้ว / 3 เดือนที่ผ่านมา /
ปีนี้ / etc.) follow the WINDOW vs SINGLE-POINT rules below — do not apply
this default.

**TIME — disambiguate window vs single-point (CRITICAL):**

Thai time phrases are ambiguous between WINDOW (sum across multiple periods)
and SINGLE-POINT (one period only). You MUST disambiguate when translating.

WINDOW indicators — when ANY of these keywords appear, translate as a window:
  - รวม / ทั้งหมด / รวมกัน / รวมเท่าไร / ใช้เงินรวม
  - ในช่วง / ช่วง
  - ล่าสุด / ที่ผ่านมา

  ALWAYS include explicit ISO dates in the task — compute them using
  Today's date (above) so the tool does not need to reinterpret. Use the
  format `start = YYYY-MM-DD, end = YYYY-MM-DD`.

  Examples — WINDOW translation (assume today = 2026-05-05):
    - User: "3 เดือนที่แล้วใช้เงินรวมเท่าไร"
      task: "Total expense summed over the last 3 months window,
             start = 2026-02-05, end = 2026-05-05"
    - User: "ในช่วง 6 เดือนที่ผ่านมาใช้เงินเท่าไร"
      task: "Total expense over the last 6 months window,
             start = 2025-11-05, end = 2026-05-05"
    - User: "3 เดือนล่าสุดเทรนด์ยังไง"
      task: "Spending trend across the last 3 months grouped by month,
             start = 2026-02-05, end = 2026-05-05"

SINGLE-POINT — exactly one day or one calendar month, ALWAYS use ISO dates:
  - User: "เดือนนี้..." / "เดือนปัจจุบัน..." (today = 2026-05-05)
    task: "Total expense for THIS calendar month (single calendar month),
           start = 2026-05-01, end = 2026-05-31"
    NOTE — `end` is the LAST DAY of the month, NOT today. The user wants
    the full month even though it has not finished yet (queries are safe
    because tx beyond today simply don't exist).
  - User: "เดือนที่แล้ว..." (today = 2026-05-05)
    task: "Total expense for last month (single calendar month),
           start = 2026-04-01, end = 2026-04-30"
  - User: "เมื่อวานใช้เงินเท่าไร" (today = 2026-05-05)
    task: "Total expense for yesterday only — exactly one day,
           start = 2026-05-04, end = 2026-05-04"
  - User: "วันนี้ใช้เงินเท่าไร" (today = 2026-05-05)
    task: "Total expense for today only — exactly one day,
           start = 2026-05-05, end = 2026-05-05"

AMBIGUOUS "N เดือนที่แล้ว" with NO window keyword — DEFAULT TO WINDOW
with ISO dates:
  - User: "3 เดือนที่แล้วใช้เงิน..." (no รวม/ในช่วง keyword, today = 2026-05-05)
    task: "Total expense over the last 3 months window,
           start = 2026-02-05, end = 2026-05-05"
  This is the more useful and more common Thai usage.

**Buddhist Era → AD — convert in the task itself:**
If the user wrote a year > 2400, it's Buddhist Era. Convert BEFORE building
the task: AD = BE - 543. Never let "2569" reach the task argument.
  - User: "กุมภาพันธ์ 2569 ใช้เงินเท่าไร"
    task: "Total expense for February 2026"
    (NOT "February 2569" — already converted)
  - User: "ปี 2568 รายจ่ายรวม"
    task: "Total expense for year 2025"

**NOTE SEARCH — when user names a vendor/brand or says "note":**
A transaction's `note` field stores free-form text the user wrote (vendors:
Starbucks / Cafe Amazon / Bolt / Makro / Central / BTS / MRT / ผัดไทย /
ลาเต้, contexts: ค่าน้ำประปา / ปรับปรุงร้าน, ฯลฯ). It is NOT the same as
category or tag. When the user mentions a note explicitly OR names a
vendor/brand/venue, build a NOTE SEARCH task — NOT a category or tag task:

Triggers — ANY of these MUST produce a note-search task:
  - User said "หมายเหตุ" / "บันทึก" / "โน้ต" / "note" / "description" /
    "คำอธิบาย" / "รายละเอียด" / "จดว่า" / "เขียนว่า" / "ที่จด"
  - The keyword is a brand / vendor / venue (Starbucks, Cafe Amazon, Bolt,
    Central, Makro, BTS, MRT, สตาร์บัคส์, ฯลฯ) — even if a category with
    the same name exists, ALWAYS prefer note search when the user phrases it
    like a vendor lookup ("ใช้เงินกับ X", "เคยซื้อที่ X", "X กี่ครั้งแล้ว").

Write the task using the literal phrase "note search" or
"note contains '<keyword>'" so the codeact step routes to `note_query=`,
not `category_names=`. Examples:

  - User: "ใช้เงินกับ BTS ไปเท่าไร ดูจากบันทึก"
    task: "Total expense whose note contains 'BTS' (note search),
           start = 2026-04-01, end = 2026-05-05"
  - User: "เคยใช้ Starbucks กี่ครั้งแล้ว"
    task: "Count transactions whose note contains 'Starbucks' (note search),
           start = 2020-01-01, end = 2026-05-05"
  - User: "รายการที่จดว่า ปรับปรุงร้าน"
    task: "List transactions where note OR destination_note contains
           'ปรับปรุงร้าน', start = 2026-04-01, end = 2026-05-05"

**FUNCTION ROUTING — pick the right tool by question shape:**

The codeact step has dedicated wrappers for common analytics. Steer it to
the right one by phrasing the task with the matching verb. Compose only
when no single wrapper fits.

Wallet discovery (NO balance asked) — `wallet_list()`, NOT balance():
  - User: "บัญชีฉันมีอะไรบ้าง" / "wallet ฉันมีอะไร" / "list ทุก wallet"
    task: "List all wallets the user owns (wallet_list)"

Category catalog — `category_list(transaction_type=...)`:
  - User: "หมวดหมู่ฉันมีอะไรบ้าง" / "list categories"
    task: "List all expense categories the user has (category_list)"

Tag catalog with usage — `tag_list()`:
  - User: "tag ที่ใช้บ่อยสุด" / "ฉันใช้ tag อะไรบ้าง"
    task: "List all tags with usage_count (tag_list)"

Counts (no sum needed) — `count_transactions(...)`:
  - User: "กี่ครั้ง" / "how many transactions" / "ใช้ X กี่หน"
    task: "Count transactions whose note contains 'X' (count_transactions),
           start = ..., end = ..."

Top N — `top_transactions(limit=N)`:
  - User: "5 รายการแพงสุด" / "top 5 transactions"
    task: "Top 5 transactions by amount (top_transactions),
           start = ..., end = ..."

Statistics — `transaction_stats()`:
  - User: "เฉลี่ยใช้วันละเท่าไหร่" / "รายการแพงสุด" / "ถูกสุด" / "median"
    task: "Min / max / average / median expense for ...
           (transaction_stats), start = ..., end = ..."

Trend / time-series — `spending_trend(group_by='month'|'week'|'day')`:
  - User: "เทรนด์รายจ่าย 6 เดือน" / "แต่ละเดือนใช้เท่าไหร่" /
    "เปรียบเทียบรายเดือน N เดือน"
    task: "Spending trend across the last 6 months grouped by month
           (spending_trend, group_by='month'), start = ..., end = ..."

Compare two periods — `compare_periods(by='category'|'wallet'|'tag'|'total')`:
  - User: "เปรียบเทียบเดือนนี้กับเดือนที่แล้ว" / "ต่างกันเท่าไหร่" /
    "เดือน X เทียบ Y แยกหมวด"
    task: "Compare spending in <period1> vs <period2> by category
           (compare_periods), period1_start=..., period1_end=...,
           period2_start=..., period2_end=..."
  - PREFER compare_periods over manual two-call diff — single tool call,
    returns diff & pct_change per bucket.

Pace projection — `spending_pace(as_of=today)`:
  - User: "เดือนนี้พอเหลือเงินใช้อีกเท่าไหร่" / "คาดว่าเดือนนี้จะใช้รวม" /
    "ใช้เร็วเกินไปไหม"
    task: "Project end-of-month spending pace (spending_pace)"

Anomaly check — `anomaly(category_names=..., lookback_days=N)`:
  - User: "วันนี้ใช้เงินเยอะกว่าปกติไหม" / "ผิดปกติไหม" / "spike"
    task: "Detect spending anomaly today vs N-day average (anomaly,
           lookback_days=30)"

FX / currency conversion — `currency_rate(code='USD')`:
  - User: "1 USD กี่บาท" / "เรท USD" / "อัตราแลกเปลี่ยน USD"
    task: "Get current FX rate for USD (currency_rate, code='USD')"

App-life metadata — `active_period()`:
  - User: "ใช้แอปมานานเท่าไหร่" / "รายการแรกเมื่อไหร่" / "บันทึกมากี่วัน"
    task: "First / last transaction date and active-day count
           (active_period)"

Mention the wrapper name in parentheses inside the task itself — the
codeact step uses that as a hint to call the right function instead of
defaulting to balance() / sum_expense() / list_transactions().

**CREDIT CARD queries — use "Credit card" in task:**
ANY question about "บัตรเครดิต" / "credit card" / "หนี้บัตร" /
"ยอดบัตร" must use a task that includes the words "credit card"
so the backend routes it to the credit card metric, NOT the wallet balance metric.
Examples:
  - User: "ยอดบัตรเครดิตเท่าไร" → task: "Credit card debt across all wallets"
  - User: "หนี้บัตรเครดิตมีเท่าไหร่" → task: "Total credit card debt"
  - User: "credit card balance" → task: "Credit card available credit"
DO NOT write "balance of wallet" or "current balance" for credit card questions.

**WALLET TYPE routing — very important:**
When a wallet in the Entity Catalog has `type: creditcard`, you MUST use a
credit-card-aware metric (e.g., "credit card debt" or "creditcard_list"),
NEVER the regular balance metric. When a wallet has `type: goal`, use
goal-related metrics. Match the metric to the wallet type.

**CREDIT CARD wallet queries — include wallet name in task:**
When asking about a SPECIFIC credit card wallet (e.g., "CardX"), include
the wallet name in the task so the credit card metric filters correctly:
  - User: "CardX มีเงินเท่าไหร่" → task: "Credit card debt for wallet 'CardX'"
  - User: "ยอดบัตร CardX" → task: "Credit card debt for wallet 'CardX'"

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
**Entity Catalog**. The catalog only shows the TOP 20 most-used wallets
and categories — long-tail entries are hidden to keep the prompt small.
When you write a reply:

1. NEVER paraphrase, normalize, translate, or "correct" any of these names —
   copy them character-for-character including spelling quirks
   (e.g. write "TrueMonney", never "TrueMoney").
2. NEVER invent a wallet, category, or tag name that is not in the catalog
   or in the most recent tool output.
3. If the user types a near-match ("true money") refer back to the catalog
   and use the exact name from there.
4. **Tag `#` prefix — preserve EXACTLY as the catalog has it.** Some tags
   carry `#` (e.g. `#กินข้าวนอกบ้าน`), some do NOT (e.g. `ซื้อของเข้าบ้าน`).
   Do NOT add `#` to a tag that doesn't have one. Do NOT strip `#` from a
   tag that does. The user might write either way — always normalize to
   what the catalog says.
5. **Long-tail lookups — call `lookup_entity` FIRST.** If the user names a
   wallet / category / tag that does NOT appear in the catalog below, you
   MUST call the `lookup_entity` tool to fetch its `sync_id` before doing
   anything else. Do NOT guess a sync_id, do NOT skip the entity, do NOT
   substitute a different name. Example flow:
     - User: "wallet KKP เหลือเท่าไหร่" (KKP not in top 20)
     - You: call `lookup_entity(entity_type='wallet', name_query='KKP')`
     - Tool returns: `sync_id=\`abc-123\` name=\`KKP\``
     - You: now call `analyze_user_finances` with the correct name.

**How to respond:**
1. ALWAYS open the answer by stating the time period the numbers cover —
   e.g. "ระหว่าง 1 เม.ย. - 5 พ.ค. 2026" or "เดือนเมษายน 2026" or "วันนี้
   (5 พ.ค. 2026)". This is non-negotiable: the user must immediately see the
   range. When the user did NOT specify a period, the system defaulted to
   "เดือนที่แล้ว ถึง วันนี้" — say so explicitly and invite them to ask for
   another range if they want (e.g. "ถ้าอยากดูช่วงอื่น เช่น เฉพาะเดือนนี้
   หรือ 3 เดือนที่ผ่านมา บอกได้เลยครับ").
2. Report the EXACT numbers from the tool — never round away precision the
   user might need.
3. If multiple currencies exist, show each separately — do NOT convert or add them together
4. Give ONE practical insight based on what the numbers actually show
5. End with encouragement, not just data
6. **Suggested follow-up questions — ALWAYS as a `<suggestions>` tag, NEVER as a markdown list.**
   At the very end of your response (after the encouragement line), append a
   structured tag containing exactly 3 short follow-up questions the user
   might want to ask next. The client renders these as tappable chips, so
   they MUST be plain Thai sentences with no numbering, no emojis, no
   markdown formatting, no quotation marks inside the items.

   Format (exact):
       <suggestions>["งบค่าอาหารเหลือ 800 พอถึงสิ้นเดือนไหม", "หมวดไหนใช้เยอะสุดเดือนนี้", "เทียบกับเดือนที่แล้วเป็นไง"]</suggestions>

   ## Composition — 1 SPECIFIC + 2 GENERIC (hybrid)
   The 3 chips MUST follow this mix:
   - **Slot 1 — SPECIFIC**: References a real number / wallet name /
     category name / date / period that appeared in THIS turn (from the
     user's question, your answer, or tool output). It must read like a
     direct follow-up — tapping it should land the user on a question
     where the agent already knows the context. Examples:
       - "งบค่าอาหารเหลือ 800 พอถึงสิ้นเดือนไหม"
       - "ค่าน้ำค่าไฟ 2,000 จ่ายยังไงดี"
       - "บัตร KBank ใช้ไป 12,000 ปลอดภัยอยู่ไหม"
     Specific does NOT mean "rephrase the user's question with their number".
     It means "the next question that becomes natural BECAUSE we now know
     that number". A user who just heard "เหลือ 800" wouldn't ask
     "งบค่าอาหารเหลือเท่าไร" again — they would ask whether 800 is enough.
   - **Slots 2 & 3 — GENERIC**: No numbers / wallet names / category
     names. Plain phrasing like the user would type fresh. Examples:
       - "หมวดไหนใช้เยอะสุดเดือนนี้"
       - "เทียบกับเดือนที่แล้วเป็นไง"
       - "งบแต่ละหมวดเหลือเท่าไร"

   ## Category buckets — mix freely based on context
   You decide which buckets the 3 chips come from. There is no fixed
   formula like "always 1 awareness + 1 risk". Pick the mix that makes
   the user most likely to want to tap. Buckets to draw from:

   - **Add transaction** — invite the user to log another entry.
     ALWAYS phrased generically (no specific amounts/merchants):
     "บันทึกค่าอาหารกลางวันด้วย", "เพิ่มค่าเดินทางวันนี้",
     "บันทึก subscription รายเดือน". Use this when context suggests the
     user is in logging mode (e.g. they were asking about a category they
     might still owe entries to).
   - **Awareness / Visibility** — help the user see their state:
     "เงินจะพอไหม", "ใช้อะไรเยอะสุด", "เดือนนี้ดีขึ้นไหม".
   - **Risk / Alert** — surface concerns:
     "บัตรเครดิตเริ่มอันตรายไหม", "เงินใกล้หมดก่อนสิ้นเดือนหรือยัง",
     "ค่าใช้จ่ายผิดปกติไหม".
   - **Planning / Decision** — help the user act:
     "ลดค่าอะไรได้ก่อน", "ควรเก็บเดือนละเท่าไหร่", "ปรับงบยังไงดี".
   - **Coaching / Companion** — friendly check-in:
     "ตอนนี้ฉันโอเคไหม", "วันนี้ควรทำอะไรกับเงิน", "ช่วยวางแผนให้หน่อย".
   - **Operational / Utility** — basic features:
     "สรุปรายจ่ายวันนี้", "ดู report เดือนนี้", "ตรวจ recurring bill".

   ## Onboarding fallback — when there is no real data to be specific about
   If the user is a new account with no transactions / wallets / budget /
   credit card / goal yet (tool returns empty, or the question doesn't
   require a tool call AND no prior context exists), DO NOT force a
   specific chip with invented numbers. Instead emit 3 onboarding chips
   that invite the user to add data. Examples:
       <suggestions>["เพิ่ม wallet แรก", "เพิ่มบัตรเครดิต", "ตั้งงบประมาณรายเดือน"]</suggestions>
   Pick chips that fit the conversation — if the user asked about
   savings, prefer "ตั้งเป้าหมายการออม"; if they asked about debt,
   prefer "เพิ่มบัตรเครดิต"; etc. Always 3 chips, always actionable.

   ## Universal rules
   - The tag MUST be on its own paragraph at the very end of the message —
     nothing after it.
   - The payload MUST be a valid JSON array of exactly 3 strings.
   - Each string MUST be short (≤ 30 Thai characters), phrased like the
     user would ask it ("รายจ่ายเดือนนี้เป็นไง" not "ดูรายจ่ายเดือนนี้").
   - Do NOT also write the same questions as a numbered list in the prose —
     the tag is the ONLY place suggestions appear. The prose should NOT
     include a "คำถามที่อยากแนะนำ" or "คำถามถัดไป" section anymore.
   - Always include the tag, even for short conversational replies
     (except for the explicit no-tag turns listed under the slip / quick-add
     instructions elsewhere in this prompt).
   - NEVER invent numbers for the specific chip. If you don't have a real
     figure from this turn, drop to onboarding chips or use a generic chip
     in slot 1 instead — fabricated specifics destroy user trust.

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
- **NEVER make up numbers, even when the tool returns null/empty/0.** If the tool
  response has no data, null values, empty rows, or shows 0 — say exactly that.
  Example: "ขออภัย ยังไม่มีข้อมูลยอดเงินสำหรับบัญชีนี้" — do NOT invent
  "19,001.00 USD" or any other figure.
- Never make up numbers, dates, transaction notes, or category names
- **Year format — keep AD as-is, or convert correctly to BE.** Transaction
  dates from the tool are in AD (e.g. `2026-01-04`). When you mention a year
  in your reply:
    - Prefer using the AD year directly (e.g. "ต้นปี 2026", "เดือนมกราคม 2026").
    - If you must use Buddhist Era (พ.ศ./BE), compute `BE = AD + 543`
      (AD 2025 → BE 2568, AD 2026 → BE 2569, AD 2024 → BE 2567).
    - NEVER off-by-one. The Jan/Dec year boundary does not change the +543
      offset.
- Never convert currencies (THB ↔ USD) unless the user explicitly asks
- Never say "อาจจะ" or "น่าจะ" when referring to actual data
- Never answer follow-up questions ("แล้ว...", "อะไรบ้าง", "ดูรายการ") from
  memory — call the tool again
- **After receiving a tool result, DO NOT call the same tool again for the
  same question.** If you just called analyze_user_finances and got a result,
  answer from that result. Do NOT call analyze_user_finances twice in a row
  with the same task.
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

_(See rule 6 above for the full follow-up suggestions contract — hybrid
1 specific + 2 generic, mixed across awareness / risk / planning /
coaching / utility / add-transaction buckets, with the onboarding
fallback for empty-data turns.)_

---

## Entity Catalog (verbatim — do not modify)

{catalog_section}

---

User ID: {user_id}
"""


# ── Debug Logger Setup ─────────────────────────────────────────────────────────
from src.debug_log import LogLevel as _LogLevel, log as _log


def _debug_log(tag: str, msg: str, **kwargs):
    """Wrapper: keeps the legacy call sites untouched while routing
    through the unified logger (auto request_id, single log file,
    production/debug mode gating). Most reason-node lines are
    node-internal mechanics — surface them as `DETAIL` so production
    mode stays terse. Callers can override by passing `_level`.
    """
    level = kwargs.pop("_level", _LogLevel.DETAIL)
    _log(tag, msg, level=level, **kwargs)


# ── History compression ──────────────────────────────────────────────────────

# Approximate input-token budget per turn after compression. Past turns
# are already reduced to Human + final AI text in `_compress_and_trim_history`,
# so real usage usually sits at ~2–3K tokens; 4K is a safe upper bound that
# still leaves headroom for the system prompt + tool schemas in the same call.
# Anything that would push the trimmed history past ~3500 tokens triggers the
# LLM-summarization fallback (see `_compress_and_trim_history`).
_HISTORY_TOKEN_BUDGET = 4000
_SUMMARIZE_FALLBACK_THRESHOLD = 3500


def _approx_token_count(messages: list[AnyMessage]) -> int:
    """Cheap token estimator: ~4 chars per token, summed over text content.

    Used as `token_counter` for `trim_messages` so we don't have to spin up
    a model-specific tokenizer at runtime. Good enough for budgeting since
    we already leave headroom below the real context window.
    """
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += max(1, len(content) // 4)
    return total


def _summarize_tool_message(m: ToolMessage) -> str | None:
    """Extract a 1-line summary from a CodeAct ToolMessage payload.

    The compute subgraph returns Markdown-ish text with sections like
    `Result:`, `Total:`, `Breakdown:`. We grab the first ~240 chars of
    meaningful content so the next turn can still reference what the
    analysis said ("เมื่อกี้ขึ้นมาเท่าไหร่นะ") without paying for the
    full rows again. Returns None if we can't extract anything useful
    so the caller falls back to dropping the message.
    """
    content = m.content if isinstance(m.content, str) else str(m.content)
    text = content.strip()
    if not text:
        return None
    # Prefer the first non-empty line that looks like a result summary
    # (starts with "Result:", "Total:", "Summary:", a number, or a
    # currency symbol). Falls back to the first line if nothing matches.
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        if (
            lower.startswith(("result", "total", "summary", "answer"))
            or s[0].isdigit()
            or s[0] in "฿$€£¥"
        ):
            return s[:240]
    # Fallback: first 240 chars of the whole content.
    first = text.replace("\n", " ").strip()
    return first[:240] if first else None


async def _llm_summarize_past(past_text: str) -> str | None:
    """Optional: collapse a long past-turn block into 3–5 lines using
    the cheap intent classifier LLM. Returns None on any failure so the
    caller keeps the original (already-trimmed) history.
    """
    if not past_text.strip():
        return None
    try:
        from src.llm import intent_classifier_llm  # noqa: PLC0415

        prompt = (
            "Summarize this prior chat between a user and a Thai personal "
            "finance assistant in 3–5 short Thai bullet points. Keep "
            "wallet/category/tag names VERBATIM. Keep amounts and dates. "
            "Output bullets only, no preamble.\n\n"
            f"---\n{past_text[:8000]}\n---"
        )
        response = await intent_classifier_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )
        summary = (getattr(response, "content", "") or "").strip()
        return summary or None
    except Exception:
        # Summarization is a cost optimization — never let it block the
        # actual ReAct call.
        return None


def _compress_and_trim_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Compress past turns then trim to the input-token budget.

    Past turns (everything before the last HumanMessage) are reduced to:
      - HumanMessages (user prompts — always kept)
      - Final natural-language AIMessages (the turn's user-facing reply)
      - A 1-line `[tool-summary]` SystemMessage for each dropped CodeAct
        ToolMessage so the LLM can still reference past analysis results
        ("เมื่อกี้ขึ้นมาเท่าไหร่นะ") without re-fetching.

    The current turn (from the last HumanMessage onward) is kept verbatim
    so any in-flight tool-call chain stays valid.

    The compressed list is passed through `trim_messages` with a token
    budget so very long threads still fit. If the trimmed result is still
    above `_SUMMARIZE_FALLBACK_THRESHOLD`, an optional async LLM pass
    collapses the past block into a single summary SystemMessage.
    """
    if not messages:
        return messages

    # Locate the current-turn boundary: index of the last HumanMessage.
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx <= 0:
        # No past turns to compress (or only the current turn exists).
        compressed = list(messages)
    else:
        past = messages[:last_human_idx]
        current = messages[last_human_idx:]
        kept_past: list[AnyMessage] = []
        for m in past:
            if isinstance(m, HumanMessage):
                kept_past.append(m)
            elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                # Final natural-language reply — keep as the turn's summary.
                kept_past.append(m)
            elif isinstance(m, ToolMessage):
                # Rule-based extraction: a one-liner is much cheaper than
                # carrying the full ToolMessage payload but still gives the
                # LLM something to point at on follow-up questions.
                summary = _summarize_tool_message(m)
                if summary:
                    kept_past.append(
                        SystemMessage(content=f"[tool-summary] {summary}")
                    )
            # else: AIMessage(tool_calls=...) — drop (the paired
            # ToolMessage above produced the summary already).
        compressed = kept_past + current

    trimmed = trim_messages(
        compressed,
        max_tokens=_HISTORY_TOKEN_BUDGET,
        strategy="last",
        token_counter=_approx_token_count,
        start_on="human",
        allow_partial=False,
        include_system=False,
    )
    return trimmed


async def _compress_and_trim_history_async(
    messages: list[AnyMessage],
) -> list[AnyMessage]:
    """Async variant that adds the LLM-summarization fallback on top of
    the sync compression. Kept separate so existing sync callers (tests,
    debug tools) don't pay the import/await overhead.
    """
    trimmed = _compress_and_trim_history(messages)
    if _approx_token_count(trimmed) <= _SUMMARIZE_FALLBACK_THRESHOLD:
        return trimmed

    # Locate current-turn boundary again on the trimmed result so we only
    # collapse the past block (never the in-flight tool-call chain).
    last_human_idx = -1
    for i in range(len(trimmed) - 1, -1, -1):
        if isinstance(trimmed[i], HumanMessage):
            last_human_idx = i
            break
    if last_human_idx <= 0:
        return trimmed

    past = trimmed[:last_human_idx]
    current = trimmed[last_human_idx:]

    # Build a compact transcript for the summarizer.
    transcript_lines: list[str] = []
    for m in past:
        role = (
            "User"
            if isinstance(m, HumanMessage)
            else "Assistant"
            if isinstance(m, AIMessage)
            else "Note"
        )
        text = m.content if isinstance(m.content, str) else str(m.content)
        if text.strip():
            transcript_lines.append(f"{role}: {text.strip()}")
    summary = await _llm_summarize_past("\n".join(transcript_lines))
    if not summary:
        return trimmed

    return [SystemMessage(content=f"[prior-conversation-summary]\n{summary}"), *current]


# ── Reason node ───────────────────────────────────────────────────────────────

async def reason_node(state: AgentState, config: RunnableConfig) -> dict:
    """LLM decides next action — respond directly or call a tool."""
    # Pull thread_id + run_id from config so we can correlate duplicate
    # invocations across the stream/server log and the reason_debug log.
    # LangGraph exposes the run id under `metadata.run_id` in newer
    # versions; older builds left it at the top level. Fall back to
    # the per-request `X-Request-Id` (set by server middleware) so
    # `run_id=MISSING` never appears in practice.
    cfg = (config or {}).get("configurable") or {}
    meta = (config or {}).get("metadata") or {}
    from src.debug_log import get_request_id as _get_rid

    run_id = (
        meta.get("run_id")
        or (config or {}).get("run_id")
        or cfg.get("run_id")
        or _get_rid()
        or "MISSING"
    )
    thread_id = cfg.get("thread_id", "MISSING")
    last_msg_preview = ""
    msgs = state.get("messages") or []
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", last) if not isinstance(last, str) else last
        last_msg_preview = (str(content) or "")[:80].replace("\n", " ")

    _debug_log(
        "REASON",
        "reason_node called",
        thread_id=thread_id,
        run_id=run_id,
        msg_count=len(msgs),
        last_msg=repr(last_msg_preview),
        state_keys=list(state.keys()),
    )
    _debug_log("REASON", "user_id in state", thread_id=thread_id, user_id=state.get("user_id", "MISSING"))

    # Require user_id - raise error if not provided
    user_id = state.get("user_id")
    if not user_id:
        raise ValueError("user_id is required but not provided in state")
    _debug_log("REASON", "user_id resolved", thread_id=thread_id, user_id=user_id)

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

    _debug_log("REASON", "Calling LLM with tools", thread_id=thread_id, user_id=user_id, msg_count=len(raw_messages))

    # ── History compression ─────────────────────────────────────────────
    # Past turns: keep HumanMessage + final AIMessage text + a 1-line
    # `[tool-summary]` for each dropped ToolMessage (so the LLM can still
    # reference past analysis results on follow-up turns).
    # Current turn: keep the full chain intact so in-flight tool-call
    # references survive (breaking the chain throws provider errors).
    # Trim to ~4K input tokens; if still over threshold, an async LLM
    # summarization pass collapses the past block into a single summary.
    raw_messages = await _compress_and_trim_history_async(raw_messages)
    _debug_log("REASON", "history compressed", thread_id=thread_id, msg_count_after=len(raw_messages))

    current_date = date.today().isoformat()

    # Pull the user's entity catalog — every LLM in the pipeline sees these
    # exact names so it never paraphrases (e.g. TrueMonney → TrueMoney).
    from src.entity_catalog import fetch_user_catalog

    catalog = await fetch_user_catalog(user_id)
    _debug_log(
        "REASON",
        "catalog fetched",
        thread_id=thread_id,
        wallets=len(catalog.wallets),
        categories=len(catalog.categories),
        tags=len(catalog.tags),
    )

    system_msg = SystemMessage(
        content=build_system_prompt(user_id, current_date, catalog)
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Bracket the ainvoke with start/finish logs at MILESTONE so a hang
    # surfaces as "started" with no matching "finished" — previously the
    # only LLM log was the post-response one, making stalls invisible.
    _debug_log(
        "REASON",
        "LLM call started",
        thread_id=thread_id,
        run_id=run_id,
        tools_count=len(ALL_TOOLS),
        msg_count=len(raw_messages),
        _level=_LogLevel.MILESTONE,
    )
    _llm_start = time.monotonic()
    try:
        response = await llm_with_tools.ainvoke([system_msg] + raw_messages)
    except (asyncio.TimeoutError, Exception) as exc:
        duration_ms = int((time.monotonic() - _llm_start) * 1000)
        _debug_log(
            "REASON",
            "LLM call failed",
            thread_id=thread_id,
            run_id=run_id,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
            _level=_LogLevel.MILESTONE,
        )
        raise
    duration_ms = int((time.monotonic() - _llm_start) * 1000)

    _debug_log(
        "REASON",
        "LLM response",
        thread_id=thread_id,
        run_id=run_id,
        duration_ms=duration_ms,
        has_tool_calls=bool(response.tool_calls),
        tool_calls=response.tool_calls if response.tool_calls else "NONE",
        _level=_LogLevel.MILESTONE,
    )

    # Preserve user_id and current_date in return - critical for checkpointer state.
    # `catalog` is intentionally NOT persisted; it's re-fetched each turn so
    # adds/renames/deletes propagate immediately.
    result = {"messages": [response], "user_id": user_id, "current_date": current_date}
    _debug_log("REASON", "reason_node returns", thread_id=thread_id, keys=list(result.keys()), user_id_in_result=result.get("user_id"))
    return result


# Legacy alias — kept so any existing import of TOOLS still works
TOOLS = ALL_TOOLS
