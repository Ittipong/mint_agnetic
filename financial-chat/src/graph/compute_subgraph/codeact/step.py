"""One iteration of the CodeAct loop: gen → execute → record.

The LLM sees the task, the catalog, the tool reference, and any prior steps
(code + stdout + error). It must either set `result = ...` to terminate or
emit more code that gets fed back into the next iteration.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from datetime import date
from decimal import Decimal
from typing import Any

from src.entity_catalog import EntityCatalog
from src.graph.compute_subgraph.codeact.exceptions import ClarificationNeeded
from src.graph.compute_subgraph.codeact.namespace import build_namespace
from src.graph.compute_subgraph.codeact.sandbox import execute
from src.graph.compute_subgraph.state import ComputeSubState

# Loop bound — picked to cover compose / diff / multi-step but reject runaway.
MAX_STEPS = 5


# ── Prompt scaffolding ───────────────────────────────────────────────────────


_TOOL_REFERENCE = """\
You have ONLY these functions available. They each query the database and
return a list[dict]. Decimal values are real Decimal objects — arithmetic
on them is exact. Dates are ISO strings or date objects.

## Resolvers — ALWAYS go through these for user-supplied names

The user may type fuzzy / partial / alternate-language names ("true money",
"เดินทาง", "kbank"). Don't guess the canonical name yourself; let the
resolvers do it deterministically against the catalog.

  resolve_wallet(query) -> str
      # 'true money' → 'TrueMonney'; 'kbank' → 'KBank Savings'
      # Raises ValueError("no wallet matches 'xxx'. Available: [...]") on miss.
      # Raises AmbiguousMatchError when multiple candidates tie.

  resolve_category(query) -> list[str]
      # Returns canonical name(s). For a parent category, expands to include
      # all direct children — so 'เดินทาง' → ['เดินทาง', 'แท็กซี่', 'BTS/MRT', 'น้ำมัน'].
      # For a leaf, returns [name] only.
      # Pass the full list to category_names= when calling SQL wrappers.

  resolve_tag(query) -> str
      # Same shape as resolve_wallet.

  resolve_budget(query) -> str
      # Echoes query — backend uses ILIKE %query%. Provided for symmetry.

  resolve_goal(query) -> str
      # Same — backend ILIKE %query%.

  parse_period(phrase=None) -> (start_date, end_date)
      # Thai/English time phrase → inclusive (start, end) tuple of date objs.
      # phrase=None or '' → all time (1900-01-01 → today).
      # Recognized: 'เดือนนี้', 'เดือนที่แล้ว', '3 เดือนที่แล้ว',
      #             'ปีนี้', 'ปีที่แล้ว', 'วันนี้', 'เมื่อวาน',
      #             'February 2026', 'กุมภาพันธ์ 2569' (BE→AD), 'last 7 days', etc.
      # Raises ValueError on unrecognized input — fall back to date(...) literal.

  clarify(question, options=[...]) -> never
      # Terminate the loop with a question to the user. Use ONLY when the
      # user's intent is genuinely ambiguous and you cannot resolve it from
      # context (e.g. 2 wallets named similarly). Example:
      #   clarify("คุณหมายถึง wallet ไหน?", options=['TrueMonney', 'TrueMoneyTH'])

## SQL wrappers — the only DB access

  sum_income(start, end, *, wallet_names=None, category_names=None,
             tag_names=None, currency='ALL',
             convert_to_thb=False) -> list[dict]
      # rows: [{"currency": ..., "amount": Decimal, "cnt": int}, ...]
      # With convert_to_thb=True you get ONE row in THB instead of one per
      # currency. PREFER this over manually summing across currency rows.

  sum_expense(start, end, *, wallet_names=None, category_names=None,
              tag_names=None, currency='ALL',
              convert_to_thb=False) -> list[dict]

  sum_by_category(start, end, *, wallet_names=None, currency='ALL',
                  convert_to_thb=False) -> list[dict]
      # rows: [{"bucket": <category_name>, "currency": ..., "amount": Decimal,
      #         "cnt": int}, ...]

  sum_by_wallet(start, end, *, currency='ALL',
                convert_to_thb=False) -> list[dict]

  sum_by_tag(start, end, *, wallet_names=None, tag_names=None,
             currency='ALL', convert_to_thb=False) -> list[dict]
      # rows: [{"bucket": <tag_name>, "currency": ..., "amount": Decimal,
      #         "cnt": int}, ...]
      # Untagged transactions are excluded. Use this for "tag/แท็ก/ป้าย/
      # หัวข้อ/แฮชแท็ก/กลุ่มไหนใช้เงินเยอะสุด" — never compose by looping.
      # `bucket` is the tag name verbatim from the catalog (preserve `#`
      # exactly as it appears — do not add or strip).

  list_transactions(start, end, *, wallet_names=None, category_names=None,
                    tag_names=None, currency='ALL',
                    order_by='date_desc' | 'amount_desc', limit=50,
                    transaction_type=None,
                    note_query=None, match_destination_note=True,
                    has_note=None) -> list[dict]
      # rows: [{"date": ..., "type": ..., "amount": Decimal, "currency": ...,
      #         "note": ..., "category_name": ..., "wallet_name": ...}, ...]
      # `transaction_type` ∈ {'income','expense','transfer','creditCardPay'}
      # narrows to one type. Use this for 'รายการรายรับ' / 'income only'.
      # Note search: `note_query` accepts a single string or a list of
      #   keywords (case-insensitive LIKE OR). With match_destination_note=True
      #   (default) also searches `destination_note` — important for transfers
      #   where the meaningful label lives on the receiving side.
      # `has_note=True` keeps only transactions with a non-empty note;
      #   `has_note=False` keeps only those without one.

  # Note search semantics — applies to sum_income / sum_expense /
  # sum_by_category / sum_by_wallet / sum_by_tag / count too via the same
  # parameters (`note_query`, `match_destination_note`, `has_note`).
  #
  # WHEN to reach for note_query (NOT category_names / tag_names):
  #   - User said "หมายเหตุ" / "บันทึก" / "โน้ต" / "note" / "description" /
  #     "คำอธิบาย" / "รายละเอียด" / "จดว่า" / "เขียนว่า" → ALWAYS note_query.
  #   - The user's keyword is a brand / vendor / venue name (Starbucks,
  #     Cafe Amazon, Bolt, Makro, BTS, MRT, ผัดไทย, ลาเต้, ร้านหมูกระทะ,
  #     สตาร์บัคส์, ฯลฯ) — these are stored in `note`, not in category. Use
  #     note_query, NOT a fuzzy category match.
  #   - "transfer เข้า X" / "โอนเข้า X" / "รับโอนจาก X" — use
  #     list_transactions(transaction_type='transfer', wallet_names=['X'])
  #     or note_query='X'. The wallet_filter automatically matches both
  #     source and destination wallet, so any transfer involving X comes back.
  #
  # Multi-keyword examples:
  #   "ค่ากาแฟทั้งหมด" → note_query=["Starbucks", "Cafe Amazon", "ลาเต้", "กาแฟ"]
  #   "รายการ Central กับ Makro" → note_query=["Central", "Makro"]
  # Single-keyword:
  #   "รายการที่หมายเหตุมี Starbucks" → note_query="Starbucks"
  #   "transfer เข้า kbank" → list_transactions(transaction_type='transfer',
  #     wallet_names=['kbank'], start=..., end=...)
  #
  # Result-empty fallback — IMPORTANT:
  #   If your first call returns count=0, sum=None, or an empty list AND
  #   the user did NOT name an explicit period (so the default last-month-
  #   to-today window kicked in), retry ONCE with the all-time window:
  #     start = date(2020, 1, 1), end = today()
  #   Then state BOTH periods in the answer so the user sees that the
  #   default window was empty but the all-time view is not.

  balance(*, as_of=None, wallet_names=None, currency='ALL',
          convert_to_thb=False) -> list[dict]
      # rows: [{"wallet_name": ..., "currency": ..., "amount": Decimal}, ...]
      # With convert_to_thb=True each wallet's balance is rolled up in THB.

  budget_remaining(*, start=None, end=None, budget_name_phrase=None) -> list[dict]
      # rows: [{"name": ..., "amount": ..., "spent": ..., "remaining": ...,
      #         "pct_used": ..., "days_remaining": ..., "daily_allowance": ...,
      #         "currency": ..., "period": ..., "start_date": ...,
      #         "end_date": ...}, ...]
      # DEFAULT (no args): active-only — only budgets whose period covers
      # TODAY. Pass an explicit start/end overlapping an older period to
      # include ended budgets (e.g. 'last month's budget'). For a generic
      # overview, ALWAYS call it with no args — never expose ended budgets
      # unless the user asked for them by name or period.

  budget_transactions(*, budget_name_phrase, order_by='amount_desc',
                      limit=20) -> list[dict]

  budget_list() -> list[dict]

  creditcard_list() -> list[dict]
      # rows: [{"name": ..., "credit_limit": Decimal, "used": Decimal | None,
      #         "available": Decimal | None, "currency": ...,
      #         "cache_updated_at": ...}, ...]
      # `used` and `available` are NULL when the backend cache has not been
      # written. NEVER substitute 0 or initial_used as a fallback — an
      # unknown reading must be relayed verbatim to the user.

  goal_list() -> list[dict]
  goal_progress(*, goal_name_phrase=None) -> list[dict]
  goal_transactions(*, goal_name_phrase, order_by='date_desc', limit=50) -> list[dict]

## Counts / discovery / analytics

  count_transactions(start, end, *, wallet_names=None, category_names=None,
                     tag_names=None, currency='ALL', transaction_type=None,
                     note_query=None, has_note=None) -> list[dict]
      # rows: [{"currency": ..., "cnt": int}, ...]
      # Use for "กี่ครั้ง / how many" — never compose len(list_transactions).

  wallet_list() -> list[dict]
      # rows: [{"kind": 'general'|'creditcard'|'goal', "sync_id": ...,
      #         "name": ..., "currency": ..., "initial_balance": Decimal,
      #         "icon": ...}, ...]
      # Use for "บัญชีฉันมีอะไรบ้าง" — quicker than balance() for discovery.

  category_list(*, transaction_type=None) -> list[dict]
      # rows: [{"sync_id": ..., "name": ..., "type": 'expense'|'income',
      #         "parent_sync_id": ..., "icon": ..., "is_active": bool}, ...]
      # Pass transaction_type to filter. Use for "หมวดหมู่ฉันมีอะไรบ้าง".

  tag_list() -> list[dict]
      # rows: [{"sync_id": ..., "name": ..., "usage_count": int}, ...]
      # Sorted by usage DESC. Use for "tag/แท็กที่ใช้บ่อยสุด".

  spending_trend(start, end, *, group_by='month'|'week'|'day'|'quarter'|'year',
                 wallet_names=None, category_names=None, tag_names=None,
                 currency='ALL', transaction_type='expense',
                 convert_to_thb=False, note_query=None,
                 has_note=None) -> list[dict]
      # rows: [{"bucket": date, "currency": ..., "amount": Decimal,
      #         "cnt": int}, ...]
      # Use for "เทรนด์ / แต่ละเดือน / 6 เดือนล่าสุด" trend questions.

  transaction_stats(start, end, *, wallet_names=None, category_names=None,
                    tag_names=None, currency='ALL',
                    transaction_type='expense', convert_to_thb=False,
                    note_query=None, has_note=None) -> list[dict]
      # rows: [{"currency": ..., "cnt": int, "min": Decimal, "max": Decimal,
      #         "avg": Decimal, "median": Decimal, "sum": Decimal}, ...]
      # Use for "เฉลี่ย / แพงสุด / ถูกสุด / median" questions.

  top_transactions(start, end, *, limit=5, wallet_names=None,
                   category_names=None, tag_names=None, currency='ALL',
                   transaction_type='expense', note_query=None,
                   has_note=None) -> list[dict]
      # Convenience for "Top N / รายการแพงสุด N รายการ".

  currency_rate(*, code=None) -> list[dict]
      # rows: [{"code": "USD", "name": ..., "symbol": ..., "rate": Decimal,
      #         "updated_at": ...}, ...]
      # Conversion: <amount> THB = <amount in code> / rate. Use for
      # "1 USD กี่บาท / FX rate".

  active_period() -> list[dict]
      # rows: [{"first_date": date, "last_date": date,
      #         "active_days": int, "total_tx": int}]
      # Use for "ใช้แอปมานานเท่าไร / รายการแรกเมื่อไหร่".

## Composite analytics (Python compose, single function call)

  compare_periods(*, period1_start, period1_end, period2_start, period2_end,
                  by='category'|'wallet'|'tag'|'total',
                  transaction_type='expense', currency='ALL',
                  convert_to_thb=True) -> dict
      # Returns {by, period1, period2, rows: [{bucket, period1_amount,
      # period2_amount, diff, pct_change}, ...]}.
      # Use for "เปรียบเทียบเดือนนี้กับเดือนที่แล้ว แยกหมวด" — single call.

  spending_pace(*, as_of=None, wallet_names=None, category_names=None,
                currency='ALL', convert_to_thb=True) -> dict
      # Returns end-of-month projection from current pace.
      # Keys: month_start, as_of, month_end, days_elapsed, days_in_month,
      #       spent_so_far, daily_avg, projected_total, projected_remaining.
      # Use for "เดือนนี้พอเหลือเงินอีกเท่าไร / projection".

  anomaly(*, category_names=None, wallet_names=None, lookback_days=30,
          as_of=None) -> dict
      # Returns {today_spent, lookback_avg, ratio, anomaly_level}.
      # anomaly_level ∈ {'normal','elevated','high','very_high'}.
      # Use for "วันนี้ใช้เยอะกว่าปกติไหม".

## Primitives

  Decimal(s)        # safe number, never use float() for money
  date(y, m, d)     # construct a date
  timedelta(days=N)
  today()           # returns today's date

## Rules

  - **Always resolve user names through resolve_*() before passing to SQL
    wrappers.** Never hand-type a wallet/category/tag name from the user's
    question — it may be a typo or paraphrase.

  - **Always go through parse_period() for time phrases** the user wrote.
    Only construct date(YYYY, M, D) yourself when the user gave you literal
    ISO dates or absolute dates.

  - On ValueError("no X matches ...") read the Available: list in the error
    message and re-call with one from there.

  - On AmbiguousMatchError read the candidates list. If your context tells
    you which one — pick it. Otherwise call clarify() to ask the user.

  - Set `result = <answer>` when finished (dict/list/Decimal/str — under 5 KB).

  - NEVER use float() for money — keep everything Decimal.

  - NEVER sum across currency rows yourself. If you need a single THB total
    across mixed-currency data, pass convert_to_thb=True to the SQL wrapper.

  - NEVER call functions not listed above.
  - NEVER use import, open, exec, eval, getattr, dunder access.

  - You may print() to log intermediate values; stdout is captured and shown
    back to you next step.

  - Multi-step is OK: leave `result = None` to keep going. You have at most
    MAX_STEPS_PLACEHOLDER steps.
"""


_INSTRUCTION = """\
You are writing Python in a sandbox. Your job is to answer the user's
financial question by composing resolvers + SQL wrappers + Python.

Write ONLY a Python code block — no explanation, no markdown fences.

## Standard flow

1. **Resolve names first** — pass user-supplied wallet/category/tag/budget/goal
   names through resolve_*() to canonicalize them. Never type them by hand.
2. **Resolve time** — if the task contains explicit ISO dates, use them
   verbatim via date(YYYY, M, D). If the user mentioned a Thai/English period
   phrase without ISO dates, call parse_period(). If NEITHER is present
   (the task has no period at all), default to last-month-1st through today:
       start = date(today.year, today.month - 1, 1) if today.month > 1 \
               else date(today.year - 1, 12, 1)
       end   = today
   This default does NOT apply to point-in-time queries (balance,
   creditcard_list, goal_list) or active-scope queries (budget_remaining
   with no args). For sum / list / count / breakdown queries, ALWAYS pass
   start and end so the answer can state the period back to the user.
3. **Pick the right SQL wrapper** for the question's intent.
4. **Compose** in Python only when you need to combine multiple queries
   (compare, trend, ratio).
5. Set `result = <payload>`.

## Date handling — CRITICAL

  - **If the task contains explicit ISO dates** like "start = 2026-02-05,
    end = 2026-05-05", USE THEM VERBATIM via `date(YYYY, M, D)`. Do NOT
    re-derive from relative phrases. Do NOT shrink a window to a single
    month. Do NOT call parse_period() when ISO dates are already given.

  - **Window vs single-point** — read the task verb:
      "summed over ... window" / "across the last N months window"
        → WINDOW. start..end is a multi-month range. Pass to sum_expense as-is.
      "for last month (single calendar month)" / "for yesterday only"
        → SINGLE. start..end is one day or one month. Pass as-is.

  - Only fall back to `parse_period(<phrase>)` when the task gives a Thai
    phrase without ISO dates (e.g. "for เดือนที่แล้ว" with no start/end).

## Composition rules

  - For overview / dashboard / 'ภาพรวมการเงิน' / 'สรุป' questions, call ALL of:
    sum_income, sum_expense, balance, creditcard_list, budget_remaining
    (active-only — no args), goal_progress. Use convert_to_thb=True on
    income/expense for a single THB total.
  - NEVER set a field to None just because you didn't query it. If a query
    returned empty list, use "0" (sums) or [] (breakdowns) — never None.
  - NEVER include ended budgets in an overview. budget_remaining() with no
    args already returns active-only.

## Example — fuzzy wallet name

    # User: "true money เหลือเท่าไร"
    wallet = resolve_wallet('true money')          # → 'TrueMonney'
    rows = balance(wallet_names=[wallet])
    result = rows

## Example — fuzzy category with subcategory expansion

    # User: "ดูรายการค่าเดินทาง"
    cats = resolve_category('เดินทาง')              # → ['เดินทาง','แท็กซี่','BTS/MRT','น้ำมัน']
    rows = list_transactions(
        start=date(1900, 1, 1), end=today(),
        category_names=cats,
    )
    result = rows

## Example — relative time (Thai phrase, no ISO dates)

    # Task: "Total expense for เดือนที่แล้ว"
    start, end = parse_period('เดือนที่แล้ว')
    rows = sum_expense(start=start, end=end, convert_to_thb=True)
    result = {'amount_thb': str(rows[0]['amount']) if rows else '0',
              'period': f"{start} → {end}"}

## Example — task with explicit ISO dates (ALWAYS use verbatim)

    # Task: "Total expense summed over the last 3 months window,
    #        start = 2026-02-05, end = 2026-05-05"
    start = date(2026, 2, 5)
    end = date(2026, 5, 5)
    rows = sum_expense(start=start, end=end, convert_to_thb=True)
    result = {'amount_thb': str(rows[0]['amount']) if rows else '0',
              'period': f"{start} → {end}"}

## Example — single day from ISO dates

    # Task: "Total expense for yesterday only,
    #        start = 2026-05-04, end = 2026-05-04"
    start = date(2026, 5, 4)
    end = date(2026, 5, 4)
    rows = sum_expense(start=start, end=end, convert_to_thb=True)
    result = {'amount_thb': str(rows[0]['amount']) if rows else '0',
              'period': f"{start} → {end}"}

## Example — compare two months

    march_start, march_end = parse_period('March 2026')
    april_start, april_end = parse_period('April 2026')
    march = sum_by_category(start=march_start, end=march_end)
    april = sum_by_category(start=april_start, end=april_end)
    by_cat = {}
    for r in march:
        by_cat.setdefault(r['bucket'], {})['march'] = r['amount']
    for r in april:
        by_cat.setdefault(r['bucket'], {})['april'] = r['amount']
    diff = []
    for cat, vals in by_cat.items():
        m = vals.get('march', Decimal('0'))
        a = vals.get('april', Decimal('0'))
        diff.append({'category': cat, 'march': m, 'april': a, 'change': a - m})
    diff.sort(key=lambda x: abs(x['change']), reverse=True)
    result = diff[:10]

## Example — handle resolver error in next step

    # First step:
    wallet = resolve_wallet('savings')             # raises AmbiguousMatchError
    # Error message tells you the candidates: ['KBank Savings', 'Goal Savings'].
    # Next step pick from context, e.g.:
    rows = balance(wallet_names=['KBank Savings'])
    result = rows

## Example — overview

    today_date = today()
    inc = sum_income(start=date(1900, 1, 1), end=today_date, convert_to_thb=True)
    exp = sum_expense(start=date(1900, 1, 1), end=today_date, convert_to_thb=True)
    bal = balance(as_of=today_date)
    cc = creditcard_list()
    bud = budget_remaining()                       # active-only by design
    goals = goal_progress()
    result = {
        'date': today_date.isoformat(),
        'total_income_thb': str(inc[0]['amount']) if inc else '0',
        'total_expense_thb': str(exp[0]['amount']) if exp else '0',
        'wallet_balances': bal,
        'credit_cards': cc,
        'active_budgets': bud,
        'savings_goals': goals,
    }
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior steps)"
    out = []
    for i, h in enumerate(history, 1):
        out.append(f"--- step {i} ---")
        out.append("CODE:")
        out.append(h.get("code", ""))
        if h.get("stdout"):
            out.append("STDOUT:")
            out.append(h["stdout"][:1000])
        if h.get("error"):
            out.append(f"ERROR: {h['error']}")
        elif h.get("result") is not None:
            out.append(f"RESULT (preview): {_preview(h['result'])}")
    return "\n".join(out)


def _preview(value: Any, limit: int = 800) -> str:
    """Stringify a result for showing back to the LLM — bounded length."""
    try:
        s = json.dumps(value, default=_json_default, ensure_ascii=False)
    except Exception:
        s = str(value)
    if len(s) > limit:
        s = s[:limit] + "… (truncated)"
    return s


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, (date,)):
        return o.isoformat()
    return str(o)


# ── Node ─────────────────────────────────────────────────────────────────────


async def codeact_step_node(state: ComputeSubState) -> dict:
    """Run one iteration. The graph loops until the LLM sets `result`
    or `MAX_STEPS` is hit."""
    from src.llm import codeact_llm as llm  # imported lazily to avoid eager API client init

    history: list[dict] = state.get("codeact_history") or []
    if len(history) >= MAX_STEPS:
        return {
            "codeact_history": history,
            "codeact_done": True,
            "codeact_final": _last_non_null_result(history),
        }

    catalog: EntityCatalog | None = state.get("catalog")  # type: ignore[assignment]
    today = date.fromisoformat(state["today"])

    catalog_section = (
        catalog.render_for_prompt() if catalog else "(catalog unavailable)"
    )

    system = (
        _INSTRUCTION
        + "\n\n## Tool reference\n\n"
        + _TOOL_REFERENCE.replace("MAX_STEPS_PLACEHOLDER", str(MAX_STEPS))
        + "\n\n## Entity Catalog (use these names verbatim)\n\n"
        + catalog_section
    )
    user = (
        f"Today: {today.isoformat()}\n"
        f"Question: {state['task']}\n\n"
        f"## Prior steps\n{_format_history(history)}\n\n"
        f"Write the next Python block. If you have the answer, set `result`."
    )

    response = await llm.ainvoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    code = _extract_code(response.content if hasattr(response, "content") else str(response))

    main_loop = asyncio.get_running_loop()
    namespace = build_namespace(
        user_id=state["user_id"],
        catalog=catalog or EntityCatalog(),
        today=today,
        main_loop=main_loop,
    )

    # Sandbox runs in a worker thread so it can call back into the main loop
    # via run_coroutine_threadsafe (DB queries) without blocking it.
    try:
        result, stdout, error = await asyncio.to_thread(execute, code, namespace)
    except ClarificationNeeded as exc:
        # Sandbox helper signaled the user must answer first — terminate the
        # loop and surface the question. We still record the step for trace.
        history = [
            *history,
            {
                "step": len(history) + 1,
                "code": code,
                "stdout": "",
                "error": f"ClarificationNeeded: {exc.question}",
                "result": None,
            },
        ]
        return {
            "codeact_history": history,
            "codeact_done": True,
            "codeact_final": None,
            "needs_clarification": True,
            "clarification_question": exc.question,
            "clarification_options": exc.options,
        }

    history = [
        *history,
        {
            "step": len(history) + 1,
            "code": code,
            "stdout": stdout,
            "error": error,
            "result": result,
        },
    ]

    done = result is not None or len(history) >= MAX_STEPS
    final = result if result is not None else _last_non_null_result(history)
    return {
        "codeact_history": history,
        "codeact_done": done,
        "codeact_final": final,
    }


def _last_non_null_result(history: list[dict]) -> Any:
    for h in reversed(history):
        if h.get("result") is not None:
            return h["result"]
    return None


def _extract_code(text: str) -> str:
    """Strip markdown fences if the LLM ignored the instruction."""
    s = text.strip()
    if s.startswith("```"):
        # remove first fence + optional language
        s = s[3:]
        if s.startswith("python"):
            s = s[6:]
        s = s.lstrip("\n")
        if s.endswith("```"):
            s = s[:-3]
    return textwrap.dedent(s).strip()
